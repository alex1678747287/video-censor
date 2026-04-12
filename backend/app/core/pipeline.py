"""Main video processing pipeline."""
import asyncio
import json
import logging
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from .. import config
from . import detector, vlm, mosaic, ocr_detector

logger = logging.getLogger(__name__)

VLM_CONCURRENCY = 8  # max concurrent VLM API calls


def get_video_info(video_path: str) -> dict:
    """Get video metadata using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)
    vs = next((s for s in data["streams"] if s["codec_type"] == "video"), {})
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
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = int(fps / base_fps)

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


def detect_highlights(video_path: str, frames: list[tuple[float, str]],
                      duration: float) -> list[dict]:
    """Use VLM to detect highlight moments. Sample every VLM_SAMPLE_INTERVAL."""
    if not frames:
        return []

    # Sample frames for highlight detection (every N seconds)
    sampled = []
    last_ts = -config.VLM_SAMPLE_INTERVAL
    for ts, path in frames:
        if ts - last_ts >= config.VLM_SAMPLE_INTERVAL:
            sampled.append((ts, path))
            last_ts = ts

    if not sampled:
        return []

    # Send batch of frames to VLM for highlight analysis
    import base64
    content_parts = [{"type": "text", "text": vlm.HIGHLIGHT_PROMPT}]
    ts_list = []
    for ts, path in sampled[:20]:  # max 20 frames per batch
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        content_parts.append({
            "type": "text", "text": f"[{ts:.1f}s]"
        })
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
        ts_list.append(ts)

    # Calculate sample interval for fallback end_time estimation
    sample_interval = config.VLM_SAMPLE_INTERVAL

    try:
        messages = [{"role": "user", "content": content_parts}]
        loop = asyncio.new_event_loop()
        raw = loop.run_until_complete(vlm._call_vlm_async(messages, max_tokens=2048))
        loop.close()
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(raw[start:end])
            highlights = result.get("highlights", [])
            # Ensure each highlight has start_time and end_time
            for h in highlights:
                if "start_time" not in h and "timestamp" in h:
                    # Backward compat: convert old single timestamp
                    h["start_time"] = h.pop("timestamp")
                if "start_time" not in h:
                    # Fallback: assign from ts_list if possible
                    idx = highlights.index(h)
                    if idx < len(ts_list):
                        h["start_time"] = f"{ts_list[idx]:.1f}s"
                if "end_time" not in h:
                    # Estimate end_time as start_time + sample_interval
                    try:
                        st = float(h["start_time"].rstrip("s"))
                        et = min(st + sample_interval, duration)
                        h["end_time"] = f"{et:.1f}s"
                    except (ValueError, KeyError):
                        pass
            return highlights
    except Exception as e:
        logger.error(f"Highlight detection failed: {e}")
    return []


def process_video(task_id: str, video_path: str,
                  progress_callback=None) -> dict:
    """Main pipeline: extract -> detect -> mosaic -> encode.
    Returns {"output_path", "violations", "highlights"}.
    """
    import time as _time
    t_start = _time.time()
    info = get_video_info(video_path)
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

        # Merge NudeNet and OCR results into shared dicts
        frame_violations = dict(nudenet_frame_violations)
        for ts, dets in ocr_frame_violations.items():
            if ts not in frame_violations:
                frame_violations[ts] = []
            frame_violations[ts].extend(dets)
        all_violations = nudenet_violations_list + ocr_violations_list
        all_nudenet_dets = nudenet_all_dets

        # VLM confirmation for borderline NudeNet EXPOSED detections
        if pending_vlm_confirm:
            logger.info(f"Task {task_id}: VLM confirming {len(pending_vlm_confirm)} borderline EXPOSED detections")
            try:
                async def _run_vlm_confirm():
                    sem = asyncio.Semaphore(VLM_CONCURRENCY)
                    async def _confirm_one(ts, path, det):
                        async with sem:
                            result = await vlm.confirm_violation_async(path)
                            return ts, det, result
                    tasks_c = [_confirm_one(ts, p, d) for ts, p, d in pending_vlm_confirm]
                    return await asyncio.gather(*tasks_c, return_exceptions=True)

                loop = asyncio.new_event_loop()
                confirm_results = loop.run_until_complete(_run_vlm_confirm())
                loop.close()

                confirmed_count = 0
                rejected_count = 0
                for item in confirm_results:
                    if isinstance(item, Exception):
                        logger.warning(f"VLM confirm error: {item}")
                        continue
                    ts, det, result = item
                    if result.get("is_violation", True):
                        # VLM confirmed: add to frame_violations
                        if ts not in frame_violations:
                            frame_violations[ts] = []
                        frame_violations[ts].append(det)
                        confirmed_count += 1
                    else:
                        rejected_count += 1
                        logger.info(f"VLM rejected NudeNet {det['class']}@{det['score']:.2f} at {ts}s: {result.get('reason', '')}")
                logger.info(f"Task {task_id}: VLM confirm done: {confirmed_count} confirmed, {rejected_count} rejected")
            except Exception as e:
                logger.warning(f"VLM confirm batch failed: {e}, keeping all borderline detections")
                for ts, path, det in pending_vlm_confirm:
                    if ts not in frame_violations:
                        frame_violations[ts] = []
                    frame_violations[ts].append(det)

        # VLM batch check for remaining subtitle texts (one API call)
        if progress_callback:
            progress_callback(0.45, "detecting_subtitles")
        if pending_vlm_texts:
            try:
                subs_for_vlm = [s for _, s in pending_vlm_texts]
                ts_for_vlm = [t for t, _ in pending_vlm_texts]
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
        vlm_sampled = []
        last_ts = -config.VLM_SAMPLE_INTERVAL
        for ts, path in frames:
            if ts - last_ts >= config.VLM_SAMPLE_INTERVAL:
                vlm_sampled.append((ts, path))
                last_ts = ts

        # Also add frames with high-score COVERED detections for VLM review
        covered_frames = set()
        for ts, dets in all_nudenet_dets.items():
            for d in dets:
                if d["class"] in ("FEMALE_BREAST_COVERED",) and d["score"] >= 0.6:
                    covered_frames.add(ts)
        for ts, path in frames:
            if ts in covered_frames and (ts, path) not in vlm_sampled:
                vlm_sampled.append((ts, path))
        vlm_sampled.sort(key=lambda x: x[0])

        vlm_batch = vlm_sampled[:40]

        async def _vlm_frame_detect():
            sem = asyncio.Semaphore(VLM_CONCURRENCY)
            async def _detect_one(ts, path):
                async with sem:
                    return ts, await vlm.detect_frame_vlm_async(
                        path, frame_width=info["width"], frame_height=info["height"]
                    )
            tasks_list = [_detect_one(ts, path) for ts, path in vlm_batch]
            return await asyncio.gather(*tasks_list, return_exceptions=True)

        async def _highlight_detect_async():
            """Run highlight detection in a thread to avoid blocking the event loop."""
            return await asyncio.get_event_loop().run_in_executor(
                None, detect_highlights, video_path, frames, info["duration"]
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
            loop = asyncio.new_event_loop()
            _parallel_results = loop.run_until_complete(_run_vlm_and_highlights())
            loop.close()
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

        for item in vlm_results:
            if isinstance(item, Exception):
                logger.warning(f"VLM detection error: {item}")
                continue
            ts, result = item
            if not result.get("safe", True):
                for v in result.get("violations", []):
                    v_type = v.get("type", "unknown")
                    violation = {
                        "timestamp": round(ts, 2),
                        "source": "vlm",
                        "type": v_type,
                        "description": v.get("description", ""),
                        "severity": v.get("severity", "medium"),
                        "need_mosaic": False,
                    }
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
                            for eb in exposed_boxes:
                                frame_violations[ts].append({
                                    "box": eb["box"], "need_mosaic": True
                                })
                            logger.info(f"VLM+NudeNet EXPOSED mosaic at {ts}s: {len(exposed_boxes)} regions")
                        elif v.get("severity") == "high":
                            # VLM high severity: mosaic even without NudeNet, use best bbox
                            violation["need_mosaic"] = True
                            if covered_boxes:
                                for cb in covered_boxes:
                                    frame_violations[ts].append({
                                        "box": cb["box"], "need_mosaic": True
                                    })
                                logger.info(f"VLM-high+NudeNet COVERED mosaic at {ts}s: {len(covered_boxes)} regions")
                            else:
                                bbox = v.get("bbox")
                                if bbox and len(bbox) == 4:
                                    # Validate bbox size: skip if > 25% of frame area
                                    bw, bh = bbox[2], bbox[3]
                                    if bw * bh > 625:  # 25*25 = 625 (25% x 25%)
                                        logger.info(f"VLM bbox too large at {ts}s: {bbox}, skipping mosaic")
                                    else:
                                        fb = vlm._bbox_percent_to_pixels(bbox, info["width"], info["height"])
                                        frame_violations[ts].append({
                                            "box": fb, "need_mosaic": True, "det_type": "vlm_region",
                                        })
                                        logger.info(f"VLM-high mosaic at {ts}s: bbox={bbox}, box={fb}")
                                else:
                                    logger.info(f"VLM-high no bbox at {ts}s, skipping mosaic")
                        else:
                            # VLM medium without NudeNet EXPOSED: report only, no mosaic
                            logger.info(f"VLM medium-only at {ts}s: no NudeNet confirm, skip mosaic")

                    # For violating text/subtitles, mosaic the text region
                    if v_type in ("违规文字",):
                        region = v.get("region", "subtitle")
                        text_box = vlm._text_region_to_box(region, info["width"], info["height"])
                        violation["need_mosaic"] = True
                        if ts not in frame_violations:
                            frame_violations[ts] = []
                        frame_violations[ts].append({
                            "box": text_box, "need_mosaic": True,
                            "det_type": "text",
                        })
                        logger.info(f"Text mosaic at {ts}s: region={region}, box={text_box}")
                    all_violations.append(violation)

        # Deduplicate violations: same source+type within 2s → keep highest score
        def _dedup_violations(violations):
            if not violations:
                return violations
            deduped = []
            seen = set()
            sorted_v = sorted(violations, key=lambda v: (v.get("timestamp", 0), -v.get("score", 0)))
            for v in sorted_v:
                key = (v.get("source"), v.get("type"))
                ts = v.get("timestamp", 0)
                dup = False
                for sk, sts in seen:
                    if sk == key and abs(sts - ts) < 2.0:
                        dup = True
                        break
                if not dup:
                    deduped.append(v)
                    seen.add((key, ts))
            return deduped

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
        logger.info(f"Task {task_id}: DONE in {total_time:.1f}s, "
                     f"{len(all_violations)} violations, {len(frames)} frames")

        return {
            "output_path": output_path,
            "violations": all_violations,
            "highlights": highlights,
            "frame_count": len(frames),
            "violation_count": len(all_violations),
        }
