"""Main video processing pipeline."""
import asyncio
import base64
import json
import logging
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from .. import config
from .. import cloud_cost
from . import detector, vlm, mosaic, ocr_detector

logger = logging.getLogger(__name__)

VLM_CONCURRENCY = 8  # max concurrent VLM API calls
HIGHLIGHT_BATCH_SIZE = 20
VLM_FRAME_BATCH_SIZE = config.VLM_FRAME_BATCH_SIZE
AUTO_MOSAIC_VLM_TYPES = {"血腥暴力"}
BLOOD_DESCRIPTION_CUES = (
    "血", "流血", "血迹", "出血", "伤口", "创口", "割伤",
    "血肉", "喷溅", "断肢", "残肢", "血污", "血浆",
)
HIGHLIGHT_START_PATTERN = re.compile(
    r'"start_time"\s*:\s*"?(?P<value>\d+(?:\.\d+)?s?)"?',
    re.IGNORECASE,
)


def get_video_info(video_path: str) -> dict:
    """Get video metadata using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {os.path.basename(video_path)}")
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"ffprobe returned invalid JSON for {os.path.basename(video_path)}")
    vs = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    if not vs:
        raise RuntimeError(f"No video stream found in {os.path.basename(video_path)}")
    duration = float(data.get("format", {}).get("duration", 0))
    fps_parts = vs.get("r_frame_rate", "25/1").split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 25.0
    return {
        "width": int(vs.get("width", 1920)),
        "height": int(vs.get("height", 1080)),
        "fps": fps,
        "duration": duration,
    }


def extract_frames_smart(video_path: str, output_dir: str,
                         base_fps: int = 2) -> list[tuple[float, str]]:
    """Extract frames at base_fps + scene changes.
    Returns list of (timestamp, frame_path).
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval = max(1, int(fps / base_fps))

    prev_gray = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ts = frame_idx / fps
        is_interval = (frame_idx % interval == 0)

        # Scene change detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        is_scene_change = False
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            score = np.mean(diff)
            if score > config.SCENE_THRESHOLD:
                is_scene_change = True
        prev_gray = gray

        if is_interval or is_scene_change:
            path = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(path, frame)
            frames.append((ts, path))

        frame_idx += 1

    cap.release()
    return frames


def _normalize_execution_profile(execution_profile: str | None = None) -> str:
    """Normalize a task-level cloud execution profile."""
    profile = str(execution_profile or config.CLOUD_EXECUTION_PROFILE).strip().lower()
    if profile not in {"local_first", "balanced"}:
        return config.CLOUD_EXECUTION_PROFILE
    return profile


def _is_local_first_profile(execution_profile: str | None = None) -> bool:
    """Whether cloud calls should be minimized around local GPU signals."""
    return _normalize_execution_profile(execution_profile) == "local_first"


def _effective_highlight_sample_interval(execution_profile: str | None = None) -> float:
    """Highlight sampling interval used by the current runtime profile."""
    if _is_local_first_profile(execution_profile):
        return max(float(config.VLM_SAMPLE_INTERVAL), config.HIGHLIGHT_SAMPLE_INTERVAL_SECONDS)
    return max(0.1, float(config.VLM_SAMPLE_INTERVAL))


def _effective_vlm_baseline_interval(execution_profile: str | None = None) -> float:
    """Base VLM frame audit interval used by the current runtime profile."""
    if _is_local_first_profile(execution_profile):
        return max(float(config.VLM_SAMPLE_INTERVAL), config.VLM_LOCAL_FIRST_BASELINE_INTERVAL_SECONDS)
    return max(0.1, float(config.VLM_SAMPLE_INTERVAL))


def _sample_highlight_frames(frames: list[tuple[float, str]],
                             sample_interval: float | None = None,
                             execution_profile: str | None = None) -> list[tuple[float, str]]:
    """Sample frames for highlight detection."""
    if not frames:
        return []
    interval = max(0.1, sample_interval or _effective_highlight_sample_interval(execution_profile))
    sampled = []
    last_ts = -interval
    for ts, path in frames:
        if ts - last_ts >= interval:
            sampled.append((ts, path))
            last_ts = ts
    return sampled


def _coalesce_timestamps(timestamps: set[float] | list[float],
                         min_gap_seconds: float) -> list[float]:
    """Collapse dense local risk timestamps into a smaller stable set."""
    if not timestamps:
        return []
    min_gap = max(0.0, float(min_gap_seconds))
    merged = []
    for ts in sorted(float(item) for item in timestamps):
        if not merged or ts - merged[-1] >= min_gap:
            merged.append(ts)
    return merged


def _compute_red_metrics(image: np.ndarray | None) -> dict:
    """Compute red / dark-red ratios with skin suppression."""
    empty = {
        "red_ratio": 0.0,
        "dark_red_ratio": 0.0,
        "spot_ratio": 0.0,
        "spot_pixels": 0,
        "has_spot_evidence": False,
        "has_evidence": False,
        "strong_evidence": False,
    }
    if image is None or image.size == 0:
        return empty

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    blue = image[:, :, 0].astype(np.int16)
    green = image[:, :, 1].astype(np.int16)
    red = image[:, :, 2].astype(np.int16)

    hue_red = (hue <= 10) | (hue >= 170)
    red_dominant = (
        (red >= green + 18) &
        (red >= blue + 12) &
        (red >= 60)
    )
    skin_like = (
        (cr >= 133) & (cr <= 178) &
        (cb >= 77) & (cb <= 135) &
        (val >= 45)
    )
    vivid_red = hue_red & red_dominant & (sat >= 90) & (val >= 40) & (~skin_like)
    dark_red = vivid_red & (val <= 150)
    vivid_red_mask = vivid_red.astype(np.uint8)
    spot_pixels = 0
    if np.any(vivid_red_mask):
        component_count, _, stats, _ = cv2.connectedComponentsWithStats(vivid_red_mask, connectivity=8)
        if component_count > 1:
            component_areas = stats[1:, cv2.CC_STAT_AREA]
            if component_areas.size:
                spot_pixels = int(component_areas.max())

    red_ratio = float(np.mean(vivid_red))
    dark_red_ratio = float(np.mean(dark_red))
    spot_ratio = float(spot_pixels / vivid_red_mask.size)
    has_spot_evidence = (
        spot_pixels >= config.VLM_BLOOD_SPOT_MIN_PIXELS and
        spot_ratio >= config.VLM_BLOOD_SPOT_RATIO_MIN
    )
    has_evidence = (
        red_ratio >= config.VLM_BLOOD_RED_RATIO_MIN and
        dark_red_ratio >= config.VLM_BLOOD_DARK_RED_RATIO_MIN
    )
    strong_evidence = red_ratio >= config.VLM_BLOOD_STRONG_RED_RATIO_MIN
    return {
        "red_ratio": red_ratio,
        "dark_red_ratio": dark_red_ratio,
        "spot_ratio": spot_ratio,
        "spot_pixels": spot_pixels,
        "has_spot_evidence": has_spot_evidence,
        "has_evidence": has_evidence,
        "strong_evidence": strong_evidence,
    }


def _score_frame_blood_candidate(image_path: str | None) -> float:
    """Cheap whole-frame blood cue score for cloud-call candidate selection."""
    if not image_path:
        return 0.0
    frame = cv2.imread(image_path)
    if frame is None or frame.size == 0:
        return 0.0
    height, width = frame.shape[:2]
    scale = min(1.0, 256.0 / max(height, width, 1))
    if scale < 1.0:
        frame = cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))))
    metrics = _compute_red_metrics(frame)
    return round(metrics["red_ratio"] + metrics["dark_red_ratio"] * 1.8, 4)


def _collect_blood_candidate_timestamps(frames: list[tuple[float, str]],
                                        execution_profile: str | None = None) -> set[float]:
    """Pick a small set of whole-frame red-heavy timestamps for VLM review."""
    if not _is_local_first_profile(execution_profile) or not frames:
        return set()
    scored = []
    for ts, path in frames:
        score = _score_frame_blood_candidate(path)
        if score >= config.VLM_LOCAL_FIRST_BLOOD_FRAME_SCORE_MIN:
            scored.append((score, ts))
    scored.sort(reverse=True)
    limit = config.VLM_LOCAL_FIRST_MAX_BLOOD_CANDIDATES
    return {ts for _, ts in scored[:limit] if limit > 0}


def _build_vlm_sample_frames(frames: list[tuple[float, str]],
                             all_nudenet_dets: dict[float, list[dict]],
                             execution_profile: str | None = None) -> list[tuple[float, str]]:
    """Select frames for expensive VLM frame audit."""
    if not frames:
        return []

    if not _is_local_first_profile(execution_profile):
        sampled = []
        last_ts = -float(config.VLM_SAMPLE_INTERVAL)
        for ts, path in frames:
            if ts - last_ts >= float(config.VLM_SAMPLE_INTERVAL):
                sampled.append((ts, path))
                last_ts = ts
        covered_frames = set()
        for ts, dets in all_nudenet_dets.items():
            for det in dets:
                if det["class"] in ("FEMALE_BREAST_COVERED",) and det["score"] >= 0.6:
                    covered_frames.add(ts)
        for ts, path in frames:
            if ts in covered_frames and (ts, path) not in sampled:
                sampled.append((ts, path))
        sampled.sort(key=lambda item: item[0])
        return sampled

    sampled_by_ts = {}
    baseline_interval = _effective_vlm_baseline_interval(execution_profile)
    last_ts = -baseline_interval
    for ts, path in frames:
        if ts - last_ts >= baseline_interval:
            sampled_by_ts[round(ts, 3)] = (ts, path)
            last_ts = ts

    risk_timestamps = set()
    for ts, dets in all_nudenet_dets.items():
        for det in dets:
            score = float(det.get("score", 0.0) or 0.0)
            support = _find_temporal_support_score(
                ts,
                det,
                all_nudenet_dets,
                window_seconds=config.NUDENET_TEMPORAL_WINDOW_SECONDS,
            )
            if det["class"] in detector.CENSOR_IF_HIGH_SCORE:
                if score >= detector.HIGH_SCORE_THRESHOLD or support is not None:
                    risk_timestamps.add(ts)
                    break
    risk_timestamps.update(_collect_blood_candidate_timestamps(frames, execution_profile))
    risk_timestamps = set(
        _coalesce_timestamps(
            risk_timestamps,
            min_gap_seconds=max(1.5, config.VLM_LOCAL_FIRST_RISK_WINDOW_SECONDS * 2.0),
        )
    )

    if risk_timestamps:
        window = config.VLM_LOCAL_FIRST_RISK_WINDOW_SECONDS
        for ts, path in frames:
            if any(abs(ts - risk_ts) <= window for risk_ts in risk_timestamps):
                sampled_by_ts[round(ts, 3)] = (ts, path)

    sampled = list(sampled_by_ts.values())
    sampled.sort(key=lambda item: item[0])
    return sampled


def _normalize_highlights(highlights: list[dict], ts_list: list[float],
                          duration: float) -> list[dict]:
    """Normalize VLM highlight output into stable start/end timestamps."""
    normalized = []
    for idx, highlight in enumerate(highlights or []):
        if not isinstance(highlight, dict):
            continue
        item = dict(highlight)
        if "start_time" not in item and "timestamp" in item:
            item["start_time"] = item.pop("timestamp")
        if "start_time" not in item:
            if idx >= len(ts_list):
                continue
            item["start_time"] = f"{ts_list[idx]:.1f}s"
        try:
            start = max(0.0, float(str(item["start_time"]).rstrip("s")))
        except ValueError:
            continue
        start = min(start, duration)
        item["start_time"] = f"{start:.1f}s"

        if "end_time" in item:
            try:
                end = float(str(item["end_time"]).rstrip("s"))
            except ValueError:
                end = start + config.VLM_SAMPLE_INTERVAL
        else:
            end = start + config.VLM_SAMPLE_INTERVAL
        end = max(start, min(end, duration))
        item["end_time"] = f"{end:.1f}s"
        normalized.append(item)
    return normalized


def _dedupe_highlights(highlights: list[dict]) -> list[dict]:
    """Keep highlight list stable when multiple batches overlap slightly."""
    deduped = []
    seen = set()
    for item in sorted(
        highlights,
        key=lambda h: float(str(h.get("start_time", "0")).rstrip("s") or 0.0),
    ):
        start = round(float(str(item.get("start_time", "0")).rstrip("s") or 0.0), 1)
        end = round(
            float(str(item.get("end_time", item.get("start_time", "0"))).rstrip("s") or 0.0),
            1,
        )
        key = (start, end, item.get("description", "").strip())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _chunk_items(items: list, batch_size: int):
    """Yield stable fixed-size batches."""
    for index in range(0, len(items), batch_size):
        yield items[index:index + batch_size]


def _resolve_vlm_visual_box(violation: dict, frame_width: int, frame_height: int) -> list[int] | None:
    """Resolve a visual violation into a concrete mosaic box."""
    box = violation.get("box")
    if isinstance(box, (list, tuple)) and len(box) == 4:
        return [int(v) for v in box]
    bbox = violation.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return vlm._bbox_percent_to_pixels(list(bbox), frame_width, frame_height)
    region = violation.get("region")
    if region:
        return vlm._region_to_box(region, frame_width, frame_height)
    return None


def _normalize_confidence(value, fallback: float | None = None) -> float | None:
    """Clamp confidence to 0-1 when present.

    Accept both fractional (0-1) and percentage-like (0-100) inputs.
    Values slightly above 1.0 are treated as invalid confidence and clamped.
    """
    if value is None:
        return fallback
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback

    if 1.0 < number <= 100.0:
        # Percentage-like confidence (e.g. 78 -> 0.78)
        if number >= 2.0:
            number /= 100.0
        else:
            # Fractional confidence should not exceed 1.0.
            number = 1.0

    return round(max(0.0, min(1.0, number)), 3)


def _confidence_from_severity(severity: str | None) -> float:
    """Map VLM severity into a user-facing confidence estimate."""
    level = str(severity or "medium").lower()
    return {
        "high": 0.92,
        "medium": 0.78,
        "low": 0.58,
    }.get(level, 0.78)


def _combine_confidence(*values) -> float | None:
    """Average multiple confidence inputs into a stable single value."""
    normalized = [item for item in (_normalize_confidence(v) for v in values) if item is not None]
    if not normalized:
        return None
    return round(sum(normalized) / len(normalized), 3)


def _box_iou(box_a: list[int] | tuple[int, ...] | None,
             box_b: list[int] | tuple[int, ...] | None) -> float:
    """Calculate IoU for two [x1, y1, x2, y2] boxes."""
    if not box_a or not box_b or len(box_a) != 4 or len(box_b) != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter_area
    if denom <= 0:
        return 0.0
    return inter_area / denom


def _boxes_similar(box_a: list[int] | tuple[int, ...] | None,
                   box_b: list[int] | tuple[int, ...] | None,
                   min_iou: float = 0.08,
                   max_center_distance_ratio: float = 0.85) -> bool:
    """Check whether two boxes likely describe the same region across nearby frames."""
    if not box_a or not box_b or len(box_a) != 4 or len(box_b) != 4:
        return False
    if _box_iou(box_a, box_b) >= min_iou:
        return True
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    acx, acy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
    bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
    aw, ah = max(1.0, ax2 - ax1), max(1.0, ay2 - ay1)
    bw, bh = max(1.0, bx2 - bx1), max(1.0, by2 - by1)
    scale = max(aw, ah, bw, bh)
    center_distance = ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5
    return center_distance <= scale * max_center_distance_ratio


def _find_temporal_support_score(timestamp: float, detection: dict,
                                 detections_by_ts: dict[float, list[dict]],
                                 window_seconds: float) -> float | None:
    """Return the strongest nearby same-class detection score, if any."""
    det_label = detection.get("class") or detection.get("type")
    det_box = detection.get("box")
    best = None
    for other_ts, other_detections in detections_by_ts.items():
        if other_ts == timestamp or abs(other_ts - timestamp) > window_seconds:
            continue
        for other in other_detections:
            other_label = other.get("class") or other.get("type")
            if det_label and other_label != det_label:
                continue
            if det_box and other.get("box") and not _boxes_similar(det_box, other["box"]):
                continue
            score = other.get("score")
            if score is None:
                score = other.get("confidence")
            normalized = _normalize_confidence(score)
            if normalized is None:
                continue
            best = max(best or 0.0, normalized)
    return round(best, 3) if best is not None else None


def _build_nudenet_violation(timestamp: float, detection: dict,
                             support_score: float | None = None,
                             review_basis: str = "NudeNet本地检测",
                             need_mosaic: bool = False,
                             extra_evidence: str | None = None) -> dict:
    """Build a consistent user-facing NudeNet violation payload."""
    evidence_parts = [f"模型分 {detection['score']:.2f}"]
    if support_score is not None:
        evidence_parts.append(f"相邻帧复现 {support_score * 100:.0f}%")
    if extra_evidence:
        evidence_parts.append(extra_evidence)
    return {
        "timestamp": round(timestamp, 2),
        "source": "nudenet",
        "type": detection["class"],
        "score": detection["score"],
        "confidence": _combine_confidence(detection["score"], support_score),
        "review_basis": review_basis,
        "evidence": " · ".join(evidence_parts),
        "need_mosaic": need_mosaic,
    }


def _bbox_area_percent(bbox: list[int] | tuple[int, ...] | None) -> float | None:
    """Return bbox area in percent-space for VLM [x, y, w, h] boxes."""
    if not bbox or len(bbox) != 4:
        return None
    try:
        _, _, width, height = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    return round(max(0.0, width) * max(0.0, height), 3)


def _should_auto_mosaic_vlm_visual(v_type: str,
                                   severity: str,
                                   support_score: float | None = None,
                                   bbox: list[int] | tuple[int, ...] | None = None) -> bool:
    """Conservative auto-mosaic decision for VLM-only visual violations."""
    severity = str(severity or "medium").lower()
    area = _bbox_area_percent(bbox)
    has_direct_box = area is not None and area <= config.VLM_VISUAL_HIGH_DIRECT_MAX_BOX_AREA
    has_support = support_score is not None

    if v_type in AUTO_MOSAIC_VLM_TYPES:
        if severity == "high":
            return has_support or has_direct_box
        return has_support

    if v_type in ("低俗色情", "低俗内容"):
        if severity != "high":
            return False
        return has_support or has_direct_box

    return False


def _build_vlm_support_map(vlm_results: list, frame_width: int, frame_height: int) -> dict[float, list[dict]]:
    """Normalize VLM visual results so we can use temporal support checks."""
    support_map: dict[float, list[dict]] = {}
    for item in vlm_results:
        if isinstance(item, Exception):
            continue
        ts, result = item
        if result.get("safe", True):
            continue
        normalized = []
        for violation in result.get("violations", []):
            normalized.append({
                "type": violation.get("type", "unknown"),
                "box": _resolve_vlm_visual_box(violation, frame_width, frame_height),
                "confidence": _confidence_from_severity(violation.get("severity")),
            })
        if normalized:
            support_map[ts] = normalized
    return support_map


def _has_blood_description_cue(description: str | None) -> bool:
    """Require explicit gore wording before trusting blood auto-mosaic."""
    text = str(description or "").strip()
    if not text:
        return False
    return any(keyword in text for keyword in BLOOD_DESCRIPTION_CUES)


def _analyze_blood_visual_evidence(image_path: str | None,
                                   box: list[int] | tuple[int, ...] | None) -> dict:
    """Inspect a candidate blood box for local red / dark-red evidence."""
    empty = {
        "has_evidence": False,
        "strong_evidence": False,
        "has_spot_evidence": False,
        "red_ratio": 0.0,
        "dark_red_ratio": 0.0,
        "spot_ratio": 0.0,
        "spot_pixels": 0,
    }
    if not image_path or not box or len(box) != 4:
        return empty

    frame = cv2.imread(image_path)
    if frame is None:
        return empty

    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(frame.shape[1] - 1, x1))
    x2 = max(0, min(frame.shape[1], x2))
    y1 = max(0, min(frame.shape[0] - 1, y1))
    y2 = max(0, min(frame.shape[0], y2))
    if x2 <= x1 or y2 <= y1:
        return empty

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return empty
    metrics = _compute_red_metrics(crop)
    return {
        "has_evidence": metrics["has_evidence"],
        "strong_evidence": metrics["strong_evidence"],
        "has_spot_evidence": metrics["has_spot_evidence"],
        "red_ratio": round(metrics["red_ratio"], 3),
        "dark_red_ratio": round(metrics["dark_red_ratio"], 3),
        "spot_ratio": round(metrics["spot_ratio"], 3),
        "spot_pixels": int(metrics["spot_pixels"]),
    }


def _should_auto_mosaic_blood_violation(description: str | None,
                                        severity: str,
                                        support_score: float | None,
                                        bbox: list[int] | tuple[int, ...] | None,
                                        image_path: str | None,
                                        visual_box: list[int] | tuple[int, ...] | None) -> tuple[bool, dict]:
    """Conservative auto-mosaic decision for blood scenes using image evidence."""
    metrics = _analyze_blood_visual_evidence(image_path, visual_box)
    has_cue = _has_blood_description_cue(description)
    metrics["has_description_cue"] = has_cue
    severity = str(severity or "medium").lower()
    area = _bbox_area_percent(bbox)
    has_direct_box = area is not None and area <= config.VLM_VISUAL_HIGH_DIRECT_MAX_BOX_AREA

    if not _should_auto_mosaic_vlm_visual(
        "血腥暴力",
        severity,
        support_score=support_score,
        bbox=bbox,
    ):
        allow = (
            severity == "medium" and
            has_direct_box and
            has_cue and
            metrics.get("has_spot_evidence", False)
        )
        return allow, metrics

    has_evidence = metrics.get("has_evidence", False)
    has_spot = metrics.get("has_spot_evidence", False)
    strong = metrics.get("strong_evidence", False)

    if support_score is not None:
        # Temporal support alone is not enough for blood mosaic; require semantic cue + visual cue,
        # or a strong visual combination to reduce red-object false positives.
        allow = (has_cue and (has_evidence or has_spot)) or (has_evidence and has_spot and strong)
    elif severity == "high":
        # No temporal support: keep strict, require cue + spot-like evidence.
        allow = has_cue and has_spot and (has_evidence or strong)
    else:
        # Medium/no-support must include both cue and local blood-color structure.
        allow = has_cue and has_direct_box and has_spot and has_evidence
    return allow, metrics


def _collect_nearby_ocr_text(timestamp: float,
                             ocr_frame_texts: dict[float, list[str]],
                             window_seconds: float = 2.0) -> str:
    """Collect nearby OCR subtitles so VLM text calls need textual corroboration."""
    texts = []
    for other_ts, items in ocr_frame_texts.items():
        if abs(other_ts - timestamp) > window_seconds:
            continue
        texts.extend(item.strip() for item in items if str(item).strip())
    deduped = []
    seen = set()
    for text in texts:
        if text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return " ".join(deduped)


def _should_keep_vlm_frame_text_violation(timestamp: float,
                                          severity: str,
                                          ocr_frame_texts: dict[float, list[str]],
                                          window_seconds: float = 2.0) -> tuple[bool, str]:
    """Require OCR subtitle corroboration before keeping frame-level VLM text hits."""
    nearby_text = _collect_nearby_ocr_text(timestamp, ocr_frame_texts, window_seconds=window_seconds)
    if not nearby_text:
        return False, ""
    return ocr_detector._should_keep_vlm_text_violation(nearby_text, severity), nearby_text


def _extract_highlight_time(text: str, field: str) -> str | None:
    """Best-effort extraction of a highlight time field from malformed JSON."""
    match = re.search(
        rf'"{field}"\s*:\s*"?(?P<value>\d+(?:\.\d+)?s?)"?',
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    return match.group("value")


def _extract_highlight_text(text: str, field: str) -> str | None:
    """Best-effort extraction of a text field from malformed highlight JSON."""
    match = re.search(
        rf'"{field}"\s*:\s*"(?P<value>.*?)(?="\s*,\s*"(?:start_time|end_time|description|reason)"|\s*}})',
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    value = match.group("value").strip().strip(",")
    return value or None


def _salvage_highlights(raw: str) -> list[dict]:
    """Recover highlight timestamps when VLM returns malformed JSON."""
    highlights = []
    starts = list(HIGHLIGHT_START_PATTERN.finditer(raw))
    for idx, match in enumerate(starts):
        start_pos = match.start()
        end_pos = starts[idx + 1].start() if idx + 1 < len(starts) else len(raw)
        chunk = raw[start_pos:end_pos]
        start_time = match.group("value")
        end_time = _extract_highlight_time(chunk, "end_time") or start_time
        description = _extract_highlight_text(chunk, "description") or "高光片段"
        reason = _extract_highlight_text(chunk, "reason") or ""
        highlights.append({
            "start_time": start_time,
            "end_time": end_time,
            "description": description,
            "reason": reason,
        })
    return highlights


def _parse_highlight_response(raw: str, ts_list: list[float], duration: float) -> list[dict]:
    """Parse VLM highlight response with fallback for malformed JSON."""
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start < 0 or end <= start:
        return []

    payload = raw[start:end]
    try:
        result = json.loads(payload)
        highlights = result.get("highlights", [])
    except json.JSONDecodeError as exc:
        highlights = _salvage_highlights(payload)
        if not highlights:
            raise
        logger.warning(
            "Highlight response JSON malformed, salvaged %d highlights: %s",
            len(highlights),
            exc,
        )
    return _normalize_highlights(highlights, ts_list, duration)


def _detect_highlight_batch(sampled_batch: list[tuple[float, str]],
                            duration: float) -> list[dict]:
    """Call VLM for one highlight batch."""
    if not sampled_batch:
        return []

    content_parts = [{"type": "text", "text": vlm.HIGHLIGHT_PROMPT}]
    ts_list = []
    for ts, path in sampled_batch:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        content_parts.append({"type": "text", "text": f"[{ts:.1f}s]"})
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
        ts_list.append(ts)

    messages = [{"role": "user", "content": content_parts}]
    raw = asyncio.run(vlm._call_vlm_async(messages, max_tokens=2048))
    return _parse_highlight_response(raw, ts_list, duration)


async def _detect_vlm_frames_async(sampled_frames: list[tuple[float, str]],
                                   frame_width: int, frame_height: int) -> list:
    """Run VLM frame audit across all sampled frames in manageable batches."""
    sem = asyncio.Semaphore(VLM_CONCURRENCY)
    results = []

    async def _detect_one(ts, path):
        async with sem:
            return ts, await vlm.detect_frame_vlm_async(
                path, frame_width=frame_width, frame_height=frame_height
            )

    for batch in _chunk_items(sampled_frames, VLM_FRAME_BATCH_SIZE):
        batch_tasks = [_detect_one(ts, path) for ts, path in batch]
        results.extend(await asyncio.gather(*batch_tasks, return_exceptions=True))
    return results


def detect_highlights(video_path: str, frames: list[tuple[float, str]],
                      duration: float,
                      sampled: list[tuple[float, str]] | None = None,
                      execution_profile: str | None = None) -> list[dict]:
    """Use VLM to detect highlight moments. Sample every VLM_SAMPLE_INTERVAL."""
    if not frames:
        return []
    sampled = sampled or _sample_highlight_frames(frames, execution_profile=execution_profile)

    if not sampled:
        return []

    highlights = []
    for batch_start in range(0, len(sampled), HIGHLIGHT_BATCH_SIZE):
        batch = sampled[batch_start: batch_start + HIGHLIGHT_BATCH_SIZE]
        try:
            highlights.extend(_detect_highlight_batch(batch, duration))
        except Exception as e:
            batch_no = batch_start // HIGHLIGHT_BATCH_SIZE + 1
            logger.error(f"Highlight detection batch {batch_no} failed: {e}")
    return _dedupe_highlights(highlights)


def process_video(task_id: str, video_path: str,
                  progress_callback=None,
                  execution_profile: str | None = None) -> dict:
    """Main pipeline: extract -> detect -> mosaic -> encode.
    Returns {"output_path", "violations", "highlights"}.
    """
    import time as _time
    t_start = _time.time()
    info = get_video_info(video_path)
    execution_profile = _normalize_execution_profile(execution_profile)
    logger.info(f"Task {task_id}: video {info}")

    with tempfile.TemporaryDirectory(prefix="vc_") as tmpdir:
        # Step 1: Extract frames
        if progress_callback:
            progress_callback(0.1, "extracting_frames")
        t_step = _time.time()
        frames = extract_frames_smart(video_path, tmpdir)
        logger.info(f"Task {task_id}: Step 1 (extract) done [{_time.time()-t_step:.1f}s] - {len(frames)} frames")

        # Step 2 + 2.5: NudeNet and OCR run in parallel (independent models)
        if progress_callback:
            progress_callback(0.3, "detecting_nudity")

        # Separate result containers for each thread (no shared writes)
        nudenet_frame_violations = {}   # {ts: [mosaic dets]}
        nudenet_all_dets = {}           # {ts: [all dets]}
        nudenet_violations_list = []    # violation records
        pending_vlm_confirm = []        # [(ts, path, det)] borderline EXPOSED needing VLM confirm

        ocr_frame_violations = {}       # {ts: [mosaic dets]}
        ocr_violations_list = []        # violation records
        ocr_frame_texts = {}            # {ts: [text_strings]}
        pending_vlm_texts = []          # [(ts, subtitle_info)]
        vlm_confirm_attempts = 0
        vlm_text_batch_calls = 0

        # Build ts->path lookup for VLM confirmation
        ts_to_path = {ts: path for ts, path in frames}

        def _run_nudenet():
            """NudeNet detection on all frames."""
            t = _time.time()
            for i, (ts, path) in enumerate(frames):
                dets = detector.detect_frame(path)
                if dets:
                    nudenet_all_dets[ts] = dets
                    for d in dets:
                        if d.get("need_mosaic"):
                            # Borderline EXPOSED (low score): VLM confirmation
                            if (d["class"] in detector.CENSOR_CLASSES and
                                    d["score"] < config.NUDENET_VLM_CONFIRM_THRESHOLD):
                                pending_vlm_confirm.append((ts, path, d))
                            # COVERED very high score (>=0.7): mosaic directly
                            elif d["class"] in detector.CENSOR_IF_HIGH_SCORE and d["score"] >= 0.7:
                                if ts not in nudenet_frame_violations:
                                    nudenet_frame_violations[ts] = []
                                nudenet_frame_violations[ts].append(d)
                            # COVERED borderline (0.65-0.7): VLM confirmation
                            elif d["class"] in detector.CENSOR_IF_HIGH_SCORE:
                                pending_vlm_confirm.append((ts, path, d))
                            else:
                                # High confidence EXPOSED: mosaic directly
                                if ts not in nudenet_frame_violations:
                                    nudenet_frame_violations[ts] = []
                                nudenet_frame_violations[ts].append(d)
                        nudenet_violations_list.append({
                            "timestamp": round(ts, 2),
                            "source": "nudenet",
                            "type": d["class"],
                            "score": d["score"],
                            "confidence": _normalize_confidence(d["score"], 0.0),
                            "review_basis": "NudeNet本地检测",
                            "evidence": f"模型分 {d['score']:.2f}",
                            "need_mosaic": d["need_mosaic"],
                        })
                # NOTE: progress_callback cannot be called from thread
                # (Celery update_state requires request context)
            logger.info(f"Task {task_id}: Step 2 (NudeNet) done [{_time.time()-t:.1f}s]")

        def _run_ocr():
            """OCR subtitle detection, sampled every 2s."""
            t = _time.time()
            ocr_last_ts = -2.0
            for ts, path in frames:
                if ts - ocr_last_ts < 2.0:
                    continue
                ocr_last_ts = ts
                try:
                    subtitles = ocr_detector.extract_subtitles(path)
                    if not subtitles:
                        continue
                    ocr_frame_texts[ts] = [s["text"] for s in subtitles]
                    # Layer 1: instant keyword match
                    instant = ocr_detector.check_instant_violations(subtitles)
                    for iv in instant:
                        ocr_violations_list.append({
                            "timestamp": round(ts, 2),
                            "source": "ocr",
                            "type": "违规文字",
                            "description": f"字幕违规: {iv['reason']} (原文: {iv['text']})",
                            "confidence": _combine_confidence(iv.get("confidence"), 0.9),
                            "review_basis": "OCR关键词直判",
                            "evidence": iv["reason"],
                            "need_mosaic": True,
                        })
                        if ts not in ocr_frame_violations:
                            ocr_frame_violations[ts] = []
                        ocr_frame_violations[ts].append({
                            "box": iv["box"], "need_mosaic": True,
                            "det_type": "text",
                        })
                        # Expand to adjacent frames with same text
                        for adj_ts, adj_texts in ocr_frame_texts.items():
                            if adj_ts != ts and iv["text"] in " ".join(adj_texts):
                                if adj_ts not in ocr_frame_violations:
                                    ocr_frame_violations[adj_ts] = []
                                ocr_frame_violations[adj_ts].append({
                                    "box": iv["box"], "need_mosaic": True,
                                    "det_type": "text",
                                })
                    # Layer 2: collect non-keyword texts for VLM batch check
                    instant_texts = {iv["text"] for iv in instant}
                    for sub in subtitles:
                        if sub["text"] not in instant_texts:
                            pending_vlm_texts.append((ts, sub))
                except Exception as e:
                    logger.warning(f"OCR detection error at {ts}s: {e}")
            logger.info(f"Task {task_id}: Step 2.5 (OCR) done [{_time.time()-t:.1f}s]")

        t_step = _time.time()
        with ThreadPoolExecutor(max_workers=2) as executor:
            fut_nudenet = executor.submit(_run_nudenet)
            fut_ocr = executor.submit(_run_ocr)
            fut_nudenet.result()
            fut_ocr.result()
        logger.info(f"Task {task_id}: Step 2+2.5 (NudeNet+OCR parallel) done [{_time.time()-t_step:.1f}s]")

        # Merge OCR results first; NudeNet will be rebuilt from raw detections with
        # temporal support + VLM confirmation to reduce isolated false positives.
        frame_violations = {}
        for ts, dets in ocr_frame_violations.items():
            if ts not in frame_violations:
                frame_violations[ts] = []
            frame_violations[ts].extend(dets)
        nudenet_frame_violations = {}
        nudenet_violations_list = []
        pending_vlm_confirm = []
        all_nudenet_dets = nudenet_all_dets

        for ts, dets in all_nudenet_dets.items():
            for det in dets:
                support_score = _find_temporal_support_score(
                    ts, det, all_nudenet_dets,
                    window_seconds=config.NUDENET_TEMPORAL_WINDOW_SECONDS,
                )
                has_support = support_score is not None
                is_exposed = det["class"] in detector.CENSOR_CLASSES
                is_covered = det["class"] in detector.CENSOR_IF_HIGH_SCORE

                if is_exposed:
                    if (
                        det["score"] >= config.NUDENET_EXPOSED_DIRECT_SCORE
                        or (has_support and det["score"] >= config.NUDENET_VLM_CONFIRM_THRESHOLD)
                    ):
                        frame_violations.setdefault(ts, []).append(det)
                        nudenet_violations_list.append(_build_nudenet_violation(
                            ts,
                            det,
                            support_score=support_score,
                            review_basis="NudeNet连续帧一致" if has_support else "NudeNet本地检测",
                            need_mosaic=True,
                        ))
                    elif det["score"] >= config.NUDENET_VLM_CONFIRM_THRESHOLD or has_support:
                        pending_vlm_confirm.append({
                            "timestamp": ts,
                            "path": ts_to_path.get(ts),
                            "det": det,
                            "support_score": support_score,
                        })
                    else:
                        logger.info(
                            "Skip isolated weak NudeNet detection %s@%.2f at %.2fs",
                            det["class"], det["score"], ts,
                        )
                    continue

                if is_covered:
                    if has_support and det["score"] >= config.NUDENET_COVERED_DIRECT_SCORE:
                        frame_violations.setdefault(ts, []).append(det)
                        nudenet_violations_list.append(_build_nudenet_violation(
                            ts,
                            det,
                            support_score=support_score,
                            review_basis="NudeNet连续帧一致",
                            need_mosaic=True,
                        ))
                    elif has_support or det["score"] >= detector.HIGH_SCORE_THRESHOLD:
                        pending_vlm_confirm.append({
                            "timestamp": ts,
                            "path": ts_to_path.get(ts),
                            "det": det,
                            "support_score": support_score,
                        })
                    else:
                        logger.info(
                            "Skip isolated covered NudeNet detection %s@%.2f at %.2fs",
                            det["class"], det["score"], ts,
                        )
                    continue

                if has_support or det["score"] >= config.NUDENET_EXPOSED_DIRECT_SCORE:
                    nudenet_violations_list.append(_build_nudenet_violation(
                        ts,
                        det,
                        support_score=support_score,
                        review_basis="NudeNet连续帧一致" if has_support else "NudeNet本地检测",
                        need_mosaic=False,
                    ))

        if pending_vlm_confirm:
            logger.info(
                f"Task {task_id}: VLM confirming {len(pending_vlm_confirm)} refined NudeNet detections"
            )
            try:
                async def _run_vlm_confirm():
                    sem = asyncio.Semaphore(VLM_CONCURRENCY)

                    async def _confirm_one(item):
                        async with sem:
                            result = await vlm.confirm_violation_async(
                                item["path"], item["det"].get("class")
                            )
                            return item, result

                    tasks_c = [_confirm_one(item) for item in pending_vlm_confirm if item.get("path")]
                    return await asyncio.gather(*tasks_c, return_exceptions=True)

                vlm_confirm_attempts = len([item for item in pending_vlm_confirm if item.get("path")])
                confirm_results = asyncio.run(_run_vlm_confirm())

                confirmed_count = 0
                rejected_count = 0
                fallback_count = 0
                for item in confirm_results:
                    if isinstance(item, Exception):
                        logger.warning(f"VLM confirm error: {item}")
                        continue
                    meta, result = item
                    ts = meta["timestamp"]
                    det = meta["det"]
                    support_score = meta.get("support_score")
                    decision = result.get("is_violation")
                    if decision is None and support_score is not None:
                        decision = True
                        fallback_count += 1
                    if decision is True:
                        frame_violations.setdefault(ts, []).append(det)
                        basis = "NudeNet + VLM复核"
                        if support_score is not None:
                            basis += " + 连续帧一致"
                        nudenet_violations_list.append(_build_nudenet_violation(
                            ts,
                            det,
                            support_score=support_score,
                            review_basis=basis,
                            need_mosaic=True,
                            extra_evidence=result.get("reason"),
                        ))
                        confirmed_count += 1
                    else:
                        rejected_count += 1
                        logger.info(
                            "VLM rejected NudeNet %s@%.2f at %.2fs: %s",
                            det["class"], det["score"], ts, result.get("reason", ""),
                        )
                logger.info(
                    f"Task {task_id}: VLM confirm done: {confirmed_count} confirmed, "
                    f"{rejected_count} rejected, {fallback_count} fallback-kept"
                )
            except Exception as e:
                logger.warning(
                    f"VLM confirm batch failed: {e}, only keeping temporally supported detections"
                )
                for meta in pending_vlm_confirm:
                    support_score = meta.get("support_score")
                    if support_score is None:
                        continue
                    ts = meta["timestamp"]
                    det = meta["det"]
                    frame_violations.setdefault(ts, []).append(det)
                    nudenet_violations_list.append(_build_nudenet_violation(
                        ts,
                        det,
                        support_score=support_score,
                        review_basis="NudeNet连续帧一致(复核失败)",
                        need_mosaic=True,
                        extra_evidence="VLM复核失败，按连续帧一致保留",
                    ))

        all_violations = nudenet_violations_list + ocr_violations_list

        # VLM batch check for remaining subtitle texts (one API call)
        if progress_callback:
            progress_callback(0.45, "detecting_subtitles")
        if pending_vlm_texts:
            try:
                subs_for_vlm = [s for _, s in pending_vlm_texts]
                ts_for_vlm = [t for t, _ in pending_vlm_texts]
                vlm_text_batch_calls = 1
                viol_indices = ocr_detector.check_vlm_violations_sync(
                    subs_for_vlm, ts_for_vlm
                )
                for idx in viol_indices:
                    if 0 <= idx < len(pending_vlm_texts):
                        ts, sub = pending_vlm_texts[idx]
                        all_violations.append({
                            "timestamp": round(ts, 2),
                            "source": "ocr_vlm",
                            "type": "违规文字",
                            "description": f"字幕违规(AI判定): {sub['text']}",
                            "confidence": _combine_confidence(sub.get("confidence"), 0.82),
                            "review_basis": "OCR + VLM文本复核",
                            "evidence": "字幕文本经云端复核后判定违规",
                            "need_mosaic": True,
                        })
                        if ts not in frame_violations:
                            frame_violations[ts] = []
                        frame_violations[ts].append({
                            "box": sub["box"], "need_mosaic": True,
                            "det_type": "text",
                        })
                        # Expand to adjacent frames with same text
                        for adj_ts, adj_texts in ocr_frame_texts.items():
                            if adj_ts != ts and sub["text"] in " ".join(adj_texts):
                                if adj_ts not in frame_violations:
                                    frame_violations[adj_ts] = []
                                frame_violations[adj_ts].append({
                                    "box": sub["box"], "need_mosaic": True,
                                    "det_type": "text",
                                })
                        logger.info(f"VLM text violation at {ts}s: '{sub['text']}'")
            except Exception as e:
                logger.warning(f"VLM text batch check failed: {e}")

        # Step 3 + 4: VLM frame detection and highlight detection run in parallel
        if progress_callback:
            progress_callback(0.5, "vlm_detection")
        vlm_sampled = _build_vlm_sample_frames(frames, all_nudenet_dets, execution_profile)
        highlight_sampled = _sample_highlight_frames(frames, execution_profile=execution_profile)

        async def _vlm_frame_detect():
            return await _detect_vlm_frames_async(
                vlm_sampled, info["width"], info["height"]
            )

        async def _highlight_detect_async():
            """Run highlight detection in a thread to avoid blocking the event loop."""
            return await asyncio.get_event_loop().run_in_executor(
                None, detect_highlights, video_path, frames, info["duration"], highlight_sampled, execution_profile
            )

        async def _run_vlm_and_highlights():
            t_vlm = _time.time()
            vlm_task = asyncio.create_task(_vlm_frame_detect())
            hl_task = asyncio.create_task(_highlight_detect_async())
            results = await asyncio.gather(vlm_task, hl_task, return_exceptions=True)
            logger.info(f"Task {task_id}: Step 3+4 (VLM+Highlights parallel) done [{_time.time()-t_vlm:.1f}s]")
            return results

        t_step = _time.time()
        try:
            _parallel_results = asyncio.run(_run_vlm_and_highlights())
            vlm_results = _parallel_results[0] if not isinstance(_parallel_results[0], Exception) else []
            highlights = _parallel_results[1] if not isinstance(_parallel_results[1], Exception) else []
            if isinstance(_parallel_results[0], Exception):
                logger.error(f"VLM batch error: {_parallel_results[0]}")
            if isinstance(_parallel_results[1], Exception):
                logger.error(f"Highlight detection error: {_parallel_results[1]}")
        except Exception as e:
            logger.error(f"VLM+Highlight concurrent batch failed: {e}")
            vlm_results = []
            highlights = []
        logger.info(f"Task {task_id}: Step 3+4 wall time [{_time.time()-t_step:.1f}s]")
        vlm_support_map = _build_vlm_support_map(vlm_results, info["width"], info["height"])

        for item in vlm_results:
            if isinstance(item, Exception):
                logger.warning(f"VLM detection error: {item}")
                continue
            ts, result = item
            if not result.get("safe", True):
                for v in result.get("violations", []):
                    v_type = v.get("type", "unknown")
                    visual_box = _resolve_vlm_visual_box(v, info["width"], info["height"])
                    image_path = ts_to_path.get(ts)
                    support_score = _find_temporal_support_score(
                        ts,
                        {
                            "type": v_type,
                            "box": visual_box,
                            "confidence": _confidence_from_severity(v.get("severity")),
                        },
                        vlm_support_map,
                        window_seconds=config.VLM_VISUAL_TEMPORAL_WINDOW_SECONDS,
                    ) if visual_box else None
                    violation = {
                        "timestamp": round(ts, 2),
                        "source": "vlm",
                        "type": v_type,
                        "description": v.get("description", ""),
                        "severity": v.get("severity", "medium"),
                        "confidence": _normalize_confidence(
                            _confidence_from_severity(v.get("severity"))
                        ),
                        "review_basis": "VLM画面复核",
                        "evidence": (
                            f"区域 {v.get('region')}" if v.get("region") else "云端画面语义命中"
                        ),
                        "need_mosaic": False,
                    }
                    if support_score is not None:
                        violation["confidence"] = _combine_confidence(
                            violation.get("confidence"), support_score
                        )
                        violation["evidence"] += f" · 相邻帧复现 {support_score * 100:.0f}%"
                    # For "低俗色情"/"低俗内容": only mosaic when NudeNet confirms EXPOSED class,
                    # or VLM severity=high. VLM-only + medium = report only, no mosaic.
                    if v_type in ("低俗色情", "低俗内容") and v.get("severity") in ("high", "medium"):
                        # Find nearest NudeNet EXPOSED detections within 0.5 seconds
                        exposed_boxes = []
                        covered_boxes = []
                        for nts, ndets in all_nudenet_dets.items():
                            if abs(nts - ts) <= 0.5:
                                for d in ndets:
                                    if d["class"] in detector.CENSOR_CLASSES and d["score"] >= 0.4:
                                        exposed_boxes.append(d)
                                    elif d["class"] in ("FEMALE_BREAST_COVERED",) and d["score"] >= 0.65:
                                        covered_boxes.append(d)

                        if ts not in frame_violations:
                            frame_violations[ts] = []

                        if exposed_boxes:
                            # NudeNet EXPOSED confirmed: always mosaic with precise bbox
                            violation["need_mosaic"] = True
                            violation["confidence"] = _combine_confidence(
                                violation.get("confidence"),
                                max((box.get("score", 0.0) for box in exposed_boxes), default=None),
                            )
                            violation["review_basis"] = "VLM画面复核 + NudeNet局部确认"
                            for eb in exposed_boxes:
                                frame_violations[ts].append({
                                    "box": eb["box"], "need_mosaic": True
                                })
                            logger.info(f"VLM+NudeNet EXPOSED mosaic at {ts}s: {len(exposed_boxes)} regions")
                        elif v.get("severity") == "high":
                            # VLM-only high severity now requires temporal support or a compact bbox.
                            if covered_boxes:
                                violation["confidence"] = _combine_confidence(
                                    violation.get("confidence"),
                                    max((box.get("score", 0.0) for box in covered_boxes), default=None),
                                )
                                violation["review_basis"] = "VLM高风险 + NudeNet辅助定位"
                                for cb in covered_boxes:
                                    frame_violations[ts].append({
                                        "box": cb["box"], "need_mosaic": True
                                    })
                                logger.info(f"VLM-high+NudeNet COVERED mosaic at {ts}s: {len(covered_boxes)} regions")
                            else:
                                bbox = v.get("bbox")
                                if _should_auto_mosaic_vlm_visual(
                                    v_type,
                                    v.get("severity"),
                                    support_score=support_score,
                                    bbox=bbox,
                                ) and visual_box:
                                    violation["need_mosaic"] = True
                                    if support_score is not None:
                                        violation["review_basis"] = "VLM画面复核 + 连续帧一致"
                                    frame_violations[ts].append({
                                        "box": visual_box, "need_mosaic": True, "det_type": "vlm_region",
                                    })
                                    logger.info(
                                        f"VLM-high mosaic at {ts}s: bbox={bbox}, support={support_score}, box={visual_box}"
                                    )
                                else:
                                    logger.info(
                                        f"VLM-high report-only at {ts}s: bbox={bbox}, support={support_score}"
                                    )
                        else:
                            # VLM medium without NudeNet EXPOSED: report only, no mosaic
                            logger.info(f"VLM medium-only at {ts}s: no NudeNet confirm, skip mosaic")
                    if v_type in AUTO_MOSAIC_VLM_TYPES and v.get("severity") in ("high", "medium"):
                        allow_blood_mosaic, blood_metrics = _should_auto_mosaic_blood_violation(
                            v.get("description"),
                            v.get("severity"),
                            support_score=support_score,
                            bbox=v.get("bbox"),
                            image_path=image_path,
                            visual_box=visual_box,
                        )
                        if blood_metrics.get("has_description_cue"):
                            violation["evidence"] += " · 血腥描述命中"
                        if blood_metrics.get("has_evidence"):
                            violation["evidence"] += (
                                f" · 血色证据 {blood_metrics['red_ratio']:.0%}/"
                                f"{blood_metrics['dark_red_ratio']:.0%}"
                            )
                        if blood_metrics.get("has_spot_evidence"):
                            violation["evidence"] += (
                                f" · 血滴局部证据 {blood_metrics['spot_pixels']}px/"
                                f"{blood_metrics['spot_ratio']:.0%}"
                            )
                        if visual_box and allow_blood_mosaic:
                            violation["need_mosaic"] = True
                            violation["review_basis"] = (
                                "VLM血腥暴力判定 + 连续帧一致"
                                if support_score is not None else "VLM血腥暴力判定 + 本地血色/血滴证据"
                            )
                            violation["confidence"] = _combine_confidence(
                                violation.get("confidence"),
                                support_score or _confidence_from_severity(v.get("severity")),
                            )
                            if ts not in frame_violations:
                                frame_violations[ts] = []
                            frame_violations[ts].append({
                                "box": visual_box, "need_mosaic": True, "det_type": "vlm_visual",
                            })
                            logger.info(f"Visual mosaic at {ts}s: type={v_type}, box={visual_box}")
                        elif visual_box:
                            logger.info(
                                "Visual report-only at %ss: type=%s, severity=%s, support=%s, cue=%s, "
                                "red=%s, dark_red=%s, spot=%s/%s",
                                ts,
                                v_type,
                                v.get("severity"),
                                support_score,
                                blood_metrics.get("has_description_cue"),
                                blood_metrics.get("red_ratio"),
                                blood_metrics.get("dark_red_ratio"),
                                blood_metrics.get("spot_pixels"),
                                blood_metrics.get("spot_ratio"),
                            )

                    # For violating text/subtitles, mosaic the text region
                    if v_type in ("违规文字",):
                        keep_text, nearby_text = _should_keep_vlm_frame_text_violation(
                            ts,
                            v.get("severity", "medium"),
                            ocr_frame_texts,
                        )
                        if keep_text:
                            region = v.get("region", "subtitle")
                            text_box = vlm._text_region_to_box(region, info["width"], info["height"])
                            violation["need_mosaic"] = True
                            violation["review_basis"] = "VLM文字区域判定 + OCR字幕佐证"
                            violation["evidence"] += f" · OCR:{nearby_text[:60]}"
                            if ts not in frame_violations:
                                frame_violations[ts] = []
                            frame_violations[ts].append({
                                "box": text_box, "need_mosaic": True,
                                "det_type": "text",
                            })
                            logger.info(f"Text mosaic at {ts}s: region={region}, box={text_box}")
                        else:
                            logger.info(
                                "Skip VLM-only text violation at %ss: severity=%s, nearby_ocr=%s",
                                ts,
                                v.get("severity"),
                                nearby_text,
                            )
                            continue
                    all_violations.append(violation)

        # Deduplicate violations: same source+type within 2s → keep highest score
        def _dedup_violations(violations):
            if not violations:
                return violations
            deduped: list[dict] = []
            window_seconds = 2.0

            # Prefer higher-score (or confidence) entries first, then earlier timestamp.
            def _rank(v: dict) -> tuple[float, float]:
                score = v.get("score")
                if score is None:
                    score = v.get("confidence")
                norm = _normalize_confidence(score, 0.0) or 0.0
                return (norm, -float(v.get("timestamp", 0) or 0.0))

            for v in sorted(violations, key=_rank, reverse=True):
                key = (v.get("source"), v.get("type"))
                ts = float(v.get("timestamp", 0) or 0.0)
                if any(
                    (existing.get("source"), existing.get("type")) == key
                    and abs(float(existing.get("timestamp", 0) or 0.0) - ts) < window_seconds
                    for existing in deduped
                ):
                    continue
                deduped.append(v)

            return sorted(deduped, key=lambda item: float(item.get("timestamp", 0) or 0.0))

        all_violations = _dedup_violations(all_violations)

        # Step 5: Apply mosaic
        if progress_callback:
            progress_callback(0.8, "applying_mosaic")
        t_step = _time.time()
        output_path = os.path.join(
            config.OUTPUT_DIR, f"{task_id}_censored.mp4"
        )
        mosaic.apply_mosaic(
            video_path, output_path, frame_violations,
            info["fps"], info["width"], info["height"],
        )
        logger.info(f"Task {task_id}: Step 5 (mosaic+encode) done [{_time.time()-t_step:.1f}s]")

        if progress_callback:
            progress_callback(1.0, "done")

        total_time = _time.time() - t_start
        text_batch_chars = sum(len(sub.get("text", "")) for _, sub in pending_vlm_texts)
        cloud_usage = cloud_cost.estimate_vlm_cost(
            execution_profile=execution_profile,
            video_duration_seconds=info.get("duration"),
            total_extracted_frames=len(frames),
            vlm_frame_calls=len(vlm_sampled),
            vlm_confirm_calls=vlm_confirm_attempts,
            text_batch_calls=vlm_text_batch_calls,
            text_batch_chars=text_batch_chars,
            highlight_images=len(highlight_sampled),
            highlight_batches=len(list(_chunk_items(highlight_sampled, HIGHLIGHT_BATCH_SIZE))),
        )
        logger.info(f"Task {task_id}: DONE in {total_time:.1f}s, "
                     f"{len(all_violations)} violations, {len(frames)} frames")

        return {
            "output_path": output_path,
            "violations": all_violations,
            "highlights": highlights,
            "cloud_usage": cloud_usage,
            "frame_count": len(frames),
            "violation_count": len(all_violations),
        }
