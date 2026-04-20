"""Drama (multi-episode) processing pipeline.
Stages: censor each episode -> speed up -> add labels -> hook clips -> concat -> trim.
"""
import logging
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata

from .. import config
from .pipeline import process_video, get_video_info

logger = logging.getLogger(__name__)
_OVERLAY_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _build_episode_offsets(episode_durations: list[dict], speed_factor: float) -> list[dict]:
    """Build cumulative time offsets for processed episodes."""
    offsets = []
    cumulative = 0.0
    for ep in episode_durations:
        processed_duration = _get_processed_duration(ep, speed_factor)
        offsets.append({
            **ep,
            "start": cumulative,
            "end": cumulative + processed_duration,
        })
        cumulative += processed_duration
    return offsets


def _find_episode_by_timestamp(ts: float, offsets: list[dict]) -> dict | None:
    """Find which processed episode contains the given global timestamp."""
    for ep in offsets:
        if ep["end"] - ep["start"] <= 1e-6:
            continue
        if ep["start"] <= ts < ep["end"]:
            return ep
    return None


def _parse_seconds(value, default: float = 0.0) -> float:
    """Parse a float-ish timestamp field like '12.3s' into seconds."""
    try:
        return float(str(value).rstrip("s"))
    except (TypeError, ValueError):
        return default


def _effective_hook_end_guard_seconds(end_guard_seconds: float | None = None) -> float:
    """Keep hook windows clear of the tail clip we remove before stitching."""
    base_guard = (
        config.DRAMA_HOOK_END_GUARD_SECONDS
        if end_guard_seconds is None else max(0.0, end_guard_seconds)
    )
    return max(base_guard, config.DRAMA_EPISODE_TAIL_TRIM_SECONDS + 0.25)


def _video_has_audio(input_path: str) -> bool:
    """Check whether the source clip contains an audio stream."""
    probe = subprocess.run(
        [
            "ffprobe", "-v", "quiet", "-select_streams", "a",
            "-show_entries", "stream=index", "-of", "csv=p=0", input_path,
        ],
        capture_output=True, text=True, timeout=30,
    )
    return bool(probe.stdout.strip())


def _build_precise_clip_cmd(input_path: str, output_path: str,
                            start: float | None = None,
                            duration: float | None = None) -> list[str]:
    """Build an accurate clip command using re-encode instead of keyframe copy."""
    src_info = get_video_info(input_path)
    cmd = ["ffmpeg", "-y", "-i", input_path]
    if start is not None and start > 0.001:
        cmd.extend(["-ss", f"{start:.3f}"])
    if duration is not None:
        cmd.extend(["-t", f"{max(0.1, duration):.3f}"])
    cmd.extend([
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-r", str(src_info["fps"]),
    ])
    if _video_has_audio(input_path):
        cmd.extend(["-c:a", "aac"])
    else:
        cmd.append("-an")
    cmd.extend(["-movflags", "+faststart", output_path])
    return cmd


def _ffmpeg_run(cmd: list[str], desc: str = ""):
    """Run FFmpeg command, raise on failure."""
    logger.info(f"FFmpeg [{desc}]: {' '.join(cmd[:8])}...")
    env = os.environ.copy()
    env["LC_ALL"] = "C.UTF-8"
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600,
                            encoding="utf-8", errors="replace", env=env)
    if result.returncode != 0:
        # Extract actual error from stderr (skip ffmpeg banner)
        lines = result.stderr.strip().splitlines()
        err_lines = [l for l in lines if not l.startswith(("  ", "ffmpeg version", "  built", "  configuration", "  lib"))]
        err_msg = "\n".join(err_lines[-10:]) if err_lines else result.stderr[-1000:]
        logger.error(f"FFmpeg [{desc}] failed:\n{err_msg}")
        raise RuntimeError(f"FFmpeg {desc} failed")
    return result


def trim_tail(input_path: str, output_path: str, trim_seconds: float | None = None) -> str:
    """Remove last N seconds from video (e.g. slow-motion outro)."""
    info = get_video_info(input_path)
    trim_seconds = config.DRAMA_EPISODE_TAIL_TRIM_SECONDS if trim_seconds is None else max(0.0, trim_seconds)
    trim_seconds = min(trim_seconds, max(0.0, info["duration"] - 1.0))
    if trim_seconds <= 0.01:
        shutil.copy2(input_path, output_path)
        return output_path
    new_duration = max(1.0, info["duration"] - trim_seconds)
    cmd = _build_precise_clip_cmd(input_path, output_path, duration=new_duration)
    _ffmpeg_run(cmd, f"trim_tail -{trim_seconds}s")
    return output_path


def _map_highlights_to_episodes(highlights: list, episode_durations: list[dict],
                                 speed_factor: float) -> list[dict]:
    """Map global highlights back to per-episode based on cumulative time offsets.

    episode_durations: [{"episode_num": 1, "episode_id": "uuid", "original_duration": 60.0}, ...]
    highlights: [{"start_time": "10.5s", "end_time": "15.0s", "description": "..."}, ...]

    Returns updated episode_durations with "highlights" key added.
    """
    offsets = _build_episode_offsets(episode_durations, speed_factor)

    # Assign each highlight to the episode it falls in
    for ep in offsets:
        ep["highlights"] = []

    for hl in highlights:
        ts_str = str(hl.get("start_time", hl.get("timestamp", "0")))
        try:
            ts = float(ts_str.rstrip("s"))
        except (TypeError, ValueError):
            logger.warning("Skip malformed highlight timestamp: %s", ts_str)
            continue
        ep = _find_episode_by_timestamp(ts, offsets)
        if not ep:
            continue
        # Convert to episode-local time
        local_hl = dict(hl)
        local_start = ts - ep["start"]
        local_hl["start_time"] = f"{local_start:.1f}s"
        if hl.get("end_time"):
            try:
                end_ts = float(str(hl["end_time"]).rstrip("s"))
            except (TypeError, ValueError):
                end_ts = ts
            local_end = max(local_start, min(end_ts, ep["end"]) - ep["start"])
            local_hl["end_time"] = f"{local_end:.1f}s"
        ep["highlights"].append(local_hl)

    return offsets


def _count_violations_by_episode(violations: list[dict], episode_durations: list[dict],
                                 speed_factor: float) -> dict[int, int]:
    """Count processed-video violations for each episode after trim and speed-up."""
    return {
        episode_num: len(items)
        for episode_num, items in _map_violations_to_episodes(
            violations, episode_durations, speed_factor
        ).items()
    }


def _map_violations_to_episodes(violations: list[dict], episode_durations: list[dict],
                                speed_factor: float) -> dict[int, list[dict]]:
    """Map global processed-video violations back to per-episode local timestamps."""
    offsets = _build_episode_offsets(episode_durations, speed_factor)
    mapped = {ep["episode_num"]: [] for ep in offsets}
    for violation in violations or []:
        try:
            ts = float(str(violation.get("timestamp", "0")).rstrip("s"))
        except ValueError:
            continue
        ep = _find_episode_by_timestamp(ts, offsets)
        if not ep:
            continue
        local_violation = dict(violation)
        local_violation["global_timestamp"] = round(ts, 2)
        local_violation["timestamp"] = round(ts - ep["start"], 2)
        mapped[ep["episode_num"]].append(local_violation)
    return mapped


def _get_processed_duration(ep: dict, speed_factor: float) -> float:
    """Return the effective episode duration after speed-up and global trim."""
    if ep.get("processed_duration") is not None:
        return max(0.0, float(ep["processed_duration"]))
    original_duration = float(ep.get("original_duration", 0.0) or 0.0)
    if speed_factor <= 0:
        return 0.0
    return max(0.0, original_duration / speed_factor)


def _limit_episode_durations(episode_durations: list[dict], speed_factor: float,
                             total_duration: float) -> list[dict]:
    """Clip per-episode durations to the globally trimmed merged video length."""
    limited = []
    remaining = max(0.0, total_duration)
    for ep in episode_durations:
        full_duration = _get_processed_duration(ep, speed_factor)
        processed_duration = min(full_duration, remaining)
        limited.append({
            **ep,
            "processed_duration": processed_duration,
            "trimmed_out": processed_duration <= 0.01,
        })
        remaining = max(0.0, remaining - processed_duration)
    return limited


def split_video_at_boundaries(input_path: str, episode_durations: list[dict],
                               speed_factor: float, output_dir: str) -> list[str]:
    """Split a merged+sped video back into per-episode segments using time boundaries.

    Uses -c copy for speed (no re-encode).
    Returns list of split file paths in episode order.
    """
    paths = []
    offset = 0.0
    for ep in episode_durations:
        ep_num = ep["episode_num"]
        sped_dur = _get_processed_duration(ep, speed_factor)
        if sped_dur <= 0.01:
            paths.append(None)
            continue
        out = os.path.join(output_dir, f"split_ep{ep_num}.mp4")
        cmd = _build_precise_clip_cmd(input_path, out, start=offset, duration=sped_dur)
        try:
            _ffmpeg_run(cmd, f"split_ep{ep_num}")
            paths.append(out)
        except Exception as e:
            logger.warning(f"split ep{ep_num} failed: {e}")
            paths.append(None)
        offset += sped_dur
    return paths


def _select_hook_window(highlights: list, episode_duration: float, clip_duration: float,
                        pre_roll_seconds: float | None = None,
                        end_guard_seconds: float | None = None) -> dict | None:
    """Pick a hook window that avoids episode tails and favors stable highlights."""
    if not highlights:
        return None
    pre_roll = config.DRAMA_HOOK_PRE_ROLL_SECONDS if pre_roll_seconds is None else max(0.0, pre_roll_seconds)
    intro_guard = max(0.0, config.DRAMA_HOOK_INTRO_SKIP_SECONDS)
    end_guard = _effective_hook_end_guard_seconds(end_guard_seconds)
    usable_start = intro_guard
    usable_end = max(0.0, episode_duration - end_guard)
    best = None
    usable_highlights = []

    for index, highlight in enumerate(highlights):
        start = max(0.0, _parse_seconds(highlight.get("start_time", highlight.get("timestamp", "0"))))
        end = max(start, _parse_seconds(highlight.get("end_time", start), start))
        if end <= usable_start + 0.35:
            continue
        if end > usable_end + 0.05:
            continue
        usable_highlights.append(highlight)
        clip_start = max(usable_start, start - pre_roll)
        clip_room = usable_end - clip_start
        if clip_room <= 0.6:
            continue
        clip_len = min(clip_duration, clip_room)
        if clip_len < max(1.0, clip_duration * 0.45):
            continue
        span = max(0.1, end - start)
        score = (span * 4.0) - (start * 0.04)
        candidate = {
            "clip_start": clip_start,
            "clip_duration": clip_len,
            "highlight": highlight,
            "score": score,
            "index": index,
        }
        if not best or candidate["score"] > best["score"]:
            best = candidate

    if best:
        return best

    if not usable_highlights:
        return None

    fallback = min(
        usable_highlights,
        key=lambda item: _parse_seconds(item.get("start_time", item.get("timestamp", "0"))),
    )
    start = max(0.0, _parse_seconds(fallback.get("start_time", fallback.get("timestamp", "0"))))
    clip_start = max(
        usable_start,
        min(start - pre_roll, max(usable_start, usable_end - max(1.0, clip_duration))),
    )
    clip_len = min(clip_duration, max(1.0, usable_end - clip_start))
    return {
        "clip_start": clip_start,
        "clip_duration": clip_len,
        "highlight": fallback,
        "score": -1.0,
        "index": 0,
    }


def _select_fallback_hook_window(episode_duration: float, clip_duration: float,
                                 intro_skip_seconds: float | None = None,
                                 end_guard_seconds: float | None = None) -> dict | None:
    """Fallback teaser window when an episode has no usable mapped highlights."""
    intro_skip = (
        config.DRAMA_HOOK_INTRO_SKIP_SECONDS
        if intro_skip_seconds is None else max(0.0, intro_skip_seconds)
    )
    end_guard = _effective_hook_end_guard_seconds(end_guard_seconds)
    usable_end = max(0.0, episode_duration - end_guard)
    if usable_end <= 0.6:
        return None

    clip_start = min(intro_skip, max(0.0, usable_end - max(1.0, clip_duration)))
    available = usable_end - clip_start
    if available <= 0.6:
        clip_start = 0.0
        available = usable_end
    if available <= 0.6:
        return None

    clip_len = min(clip_duration, available)
    return {
        "clip_start": clip_start,
        "clip_duration": max(1.0, clip_len),
        "highlight": None,
        "score": -2.0,
        "index": -1,
    }


def extract_highlight_clip(video_path: str, highlights: list, output_path: str,
                           duration: float = 3.0, episode_duration: float | None = None) -> bool:
    """Extract best highlight clip from video. Returns True if clip created."""
    if not highlights:
        return False
    if episode_duration is None:
        episode_duration = get_video_info(video_path)["duration"]
    window = _select_hook_window(highlights, episode_duration, duration)
    if not window:
        return False
    cmd = _build_precise_clip_cmd(
        video_path,
        output_path,
        start=window["clip_start"],
        duration=window["clip_duration"],
    )
    try:
        _ffmpeg_run(cmd, "extract_highlight")
        return os.path.exists(output_path)
    except Exception as e:
        logger.warning(f"Highlight clip extraction failed: {e}")
        return False


def extract_hook_clip(video_path: str, highlights: list, output_path: str,
                      duration: float = 3.0, episode_duration: float | None = None) -> bool:
    """Extract next-episode hook clip, preferring highlights and falling back to an early teaser."""
    if episode_duration is None:
        episode_duration = get_video_info(video_path)["duration"]
    window = _select_hook_window(highlights, episode_duration, duration)
    if not window:
        window = _select_fallback_hook_window(episode_duration, duration)
    if not window:
        return False
    cmd = _build_precise_clip_cmd(
        video_path,
        output_path,
        start=window["clip_start"],
        duration=window["clip_duration"],
    )
    try:
        _ffmpeg_run(cmd, "extract_hook")
        return os.path.exists(output_path)
    except Exception as e:
        logger.warning(f"Hook clip extraction failed: {e}")
        return False


def speed_up_video(input_path: str, output_path: str, factor: float = 1.5) -> str:
    """Speed up video by factor. Returns output path."""
    if abs(factor - 1.0) < 0.01:
        shutil.copy2(input_path, output_path)
        return output_path
    # Build atempo chain for factors > 2.0 (atempo range: 0.5-2.0)
    atempo_filters = []
    remaining = factor
    while remaining > 2.0:
        atempo_filters.append("atempo=2.0")
        remaining /= 2.0
    atempo_filters.append(f"atempo={remaining:.4f}")
    atempo_chain = ",".join(atempo_filters)

    # Get source frame rate to preserve it
    src_info = get_video_info(input_path)
    src_fps = src_info["fps"]

    # Check if video has audio stream
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", input_path],
        capture_output=True, text=True, timeout=30,
    )
    has_audio = bool(probe.stdout.strip())

    if has_audio:
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-filter_complex",
            f"[0:v]setpts=PTS/{factor}[v];[0:a]{atempo_chain}[a]",
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-r", str(src_fps),
            "-c:a", "aac", "-movflags", "+faststart", output_path,
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-vf", f"setpts=PTS/{factor}",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-r", str(src_fps),
            "-an", "-movflags", "+faststart", output_path,
        ]
    _ffmpeg_run(cmd, f"speed_up x{factor}")
    return output_path


def _find_cjk_font(bold: bool = True) -> str:
    """Find a usable CJK font path, preferring bold or regular variant."""
    preferred = config.FONT_PATH if bold else config.FONT_PATH_REGULAR
    fallback = config.FONT_PATH_REGULAR if bold else config.FONT_PATH
    candidates = [
        preferred,
        fallback,
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
        config.FONT_PATH,
        config.FONT_PATH_REGULAR,
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return preferred


def _escape_ffmpeg_path(path: str) -> str:
    """Escape special chars in path for FFmpeg filter expressions."""
    path = path.replace("\\", "/")
    for ch in (":", "'", "[", "]", ";", ","):
        path = path.replace(ch, f"\\{ch}")
    return path


def _safe_concat_path(path: str) -> str:
    """Validate and escape path for FFmpeg concat list file."""
    abs_path = os.path.abspath(path)
    allowed = [os.path.abspath(d) + os.sep for d in (config.DRAMA_DIR, config.OUTPUT_DIR, "/tmp")]
    if not any(abs_path.startswith(d) for d in allowed):
        raise ValueError("Path outside allowed directory")
    return abs_path.replace("'", "'\\''")


def _sanitize_overlay_text(text: str, fallback: str) -> str:
    """Normalize overlay text to reduce garbled spacing and hidden control chars."""
    raw = unicodedata.normalize("NFKC", str(text or ""))
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    raw = _OVERLAY_CONTROL_CHARS.sub("", raw)
    raw = re.sub(r"[ \t]+", " ", raw)
    cleaned_lines = [line.strip() for line in raw.split("\n") if line.strip()]
    cleaned = "\n".join(cleaned_lines)
    return cleaned or fallback


def _build_label_filter(esc_title: str, esc_disc: str, esc_title_font: str,
                        esc_disc_font: str | None = None) -> str:
    """Build a stable watermark layout: larger title + clearer bottom disclaimer."""
    esc_disc_font = esc_disc_font or esc_title_font
    return (
        f"drawtext=textfile='{esc_title}':fontsize=72:fontcolor=#f6d76b:"
        f"x=(w-text_w)/2:y=h*0.055:fontfile='{esc_title_font}':"
        f"text_shaping=1:fix_bounds=1:expansion=none:"
        f"borderw=3:bordercolor=black@0.92:box=1:boxcolor=black@0.26:boxborderw=22:"
        f"shadowcolor=black@0.65:shadowx=2:shadowy=2,"
        f"drawtext=textfile='{esc_disc}':fontsize=42:fontcolor=white:"
        f"x=(w-text_w)/2:y=h-text_h-h*0.055:fontfile='{esc_disc_font}':"
        f"text_shaping=1:fix_bounds=1:expansion=none:line_spacing=8:"
        f"borderw=2:bordercolor=black@0.86:box=1:boxcolor=black@0.62:boxborderw=18:"
        f"shadowcolor=black@0.6:shadowx=1:shadowy=1"
    )


def add_labels(input_path: str, output_path: str, title: str,
               disclaimer: str, font_path: str = "") -> str:
    """Add top title label and bottom disclaimer to video."""
    title = _sanitize_overlay_text(title, "精彩短剧")
    disclaimer = _sanitize_overlay_text(disclaimer, "影视演绎效果 请勿模仿剧情行为")
    title_font_path = font_path or _find_cjk_font(True)
    body_font_path = _find_cjk_font(False)
    # Write text to temp files to avoid CJK encoding issues in CLI
    title_path = None
    disc_path = None
    try:
        title_f = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
        title_path = title_f.name
        title_f.write(title)
        title_f.close()
        disc_f = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
        disc_path = disc_f.name
        disc_f.write(disclaimer)
        disc_f.close()
        esc_title = _escape_ffmpeg_path(title_path)
        esc_disc = _escape_ffmpeg_path(disc_path)
        esc_title_font = _escape_ffmpeg_path(title_font_path)
        esc_disc_font = _escape_ffmpeg_path(body_font_path)
        vf = _build_label_filter(esc_title, esc_disc, esc_title_font, esc_disc_font)
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-vf", vf,
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-c:a", "copy", "-movflags", "+faststart", output_path,
        ]
        _ffmpeg_run(cmd, "add_labels")
    finally:
        if title_path and os.path.exists(title_path):
            os.unlink(title_path)
        if disc_path and os.path.exists(disc_path):
            os.unlink(disc_path)
    return output_path


def concat_with_hook(episode_path: str, hook_clip_path: str,
                     output_path: str) -> str:
    """Concat episode video with next episode's highlight hook clip."""
    if not hook_clip_path or not os.path.exists(hook_clip_path):
        shutil.copy2(episode_path, output_path)
        return output_path
    # Hook clip may have different encoding params, so re-encode for safety
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(f"file '{_safe_concat_path(episode_path)}'\n")
        f.write(f"file '{_safe_concat_path(hook_clip_path)}'\n")
        concat_list = f.name
    try:
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_list,
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-c:a", "aac", "-movflags", "+faststart", output_path,
        ]
        _ffmpeg_run(cmd, "concat_hook")
    finally:
        os.unlink(concat_list)
    return output_path


def concat_episodes(episode_paths: list[str], output_path: str) -> str:
    """Concat all episodes into one video with consistent frame rate."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for p in episode_paths:
            f.write(f"file '{_safe_concat_path(p)}'\n")
        concat_list = f.name
    try:
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_list,
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-c:a", "aac", "-vsync", "cfr",
            "-movflags", "+faststart", output_path,
        ]
        _ffmpeg_run(cmd, "concat_all")
    finally:
        os.unlink(concat_list)
    return output_path


def trim_video(input_path: str, output_path: str, max_seconds: int) -> str:
    """Trim video to max duration."""
    info = get_video_info(input_path)
    if info["duration"] <= max_seconds:
        shutil.copy2(input_path, output_path)
        return output_path
    cmd = _build_precise_clip_cmd(input_path, output_path, duration=float(max_seconds))
    _ffmpeg_run(cmd, f"trim to {max_seconds}s")
    return output_path


def process_drama(drama_id: str, episodes_info: list[dict],
                   title: str, speed_factor: float, max_duration: int,
                   disclaimer: str, progress_callback=None,
                   execution_profile: str | None = None) -> dict:
    """Main drama pipeline.

    Flow:
      1. Preprocess (sort + trim_tail)
      2. Concat + speed_up + trim
      3. Censor (process_video on merged)
      4. Highlight detection (from censor result)
      5. Split at episode boundaries -> extract highlights -> insert hooks
      6. Re-concat hooked episodes
      7. Add labels
    """
    drama_dir = os.path.join(config.DRAMA_DIR, drama_id)
    os.makedirs(drama_dir, exist_ok=True)
    intermediates = []  # track files for cleanup

    # Sort by episode number
    episodes_info = sorted(episodes_info, key=lambda e: e["episode_num"])
    total_eps = len(episodes_info)

    # ---- Stage 1: Preprocess — trim tail (0-5%) ----
    if progress_callback:
        progress_callback(0.01, "preprocessing")
    trimmed_paths = []
    episode_durations = []
    for ep in episodes_info:
        ep_num = ep["episode_num"]
        out = os.path.join(drama_dir, f"trimmed_ep{ep_num}.mp4")
        try:
            trim_tail(ep["video_path"], out)
            trimmed_paths.append(out)
        except Exception as e:
            logger.warning(f"trim_tail ep{ep_num} failed: {e}, using original")
            trimmed_paths.append(ep["video_path"])
        try:
            info = get_video_info(trimmed_paths[-1])
            episode_durations.append({
                "episode_num": ep_num,
                "episode_id": ep.get("episode_id"),
                "original_duration": info["duration"],
            })
        except Exception:
            episode_durations.append({
                "episode_num": ep_num,
                "episode_id": ep.get("episode_id"),
                "original_duration": 0.0,
            })
    intermediates.extend(trimmed_paths)

    # ---- Stage 2: Concat all episodes (5-10%) ----
    if progress_callback:
        progress_callback(0.05, "concatenating")
    merged_path = os.path.join(drama_dir, "merged.mp4")
    concat_episodes(trimmed_paths, merged_path)
    intermediates.append(merged_path)

    # ---- Stage 3: Global speed up (10-15%) ----
    if progress_callback:
        progress_callback(0.10, "speeding_up")
    sped_path = os.path.join(drama_dir, "sped.mp4")
    speed_up_video(merged_path, sped_path, speed_factor)
    intermediates.append(sped_path)

    # ---- Stage 4: Trim to max duration (15-20%) ----
    if progress_callback:
        progress_callback(0.15, "trimming")
    trimmed_global = os.path.join(drama_dir, "trimmed_global.mp4")
    trim_video(sped_path, trimmed_global, max_duration)
    intermediates.append(trimmed_global)
    trimmed_duration = get_video_info(trimmed_global)["duration"]
    episode_durations = _limit_episode_durations(episode_durations, speed_factor, trimmed_duration)

    # ---- Stage 5: Content censor + mosaic (20-70%) ----
    if progress_callback:
        progress_callback(0.20, "censoring")
    logger.info(f"Drama {drama_id}: censoring merged video")

    def _censor_progress(p: float, stage: str):
        if progress_callback:
            progress_callback(0.20 + p * 0.50, f"censoring_{stage}")

    censor_result = process_video(
        f"{drama_id}_merged", trimmed_global,
        progress_callback=_censor_progress,
        execution_profile=execution_profile,
    )
    censored_path = censor_result["output_path"]
    all_highlights = censor_result.get("highlights", [])
    per_episode_violations = _map_violations_to_episodes(
        censor_result.get("violations", []),
        episode_durations,
        speed_factor,
    )
    violation_counts = _count_violations_by_episode(
        censor_result.get("violations", []),
        episode_durations,
        speed_factor,
    )
    intermediates.append(censored_path)

    # ---- Stage 6: Map highlights to episodes (70%) ----
    if progress_callback:
        progress_callback(0.70, "mapping_highlights")
    ep_mapped = _map_highlights_to_episodes(all_highlights, episode_durations, speed_factor)

    # Build episode_results for DB storage
    episode_results = []
    for ep in ep_mapped:
        episode_results.append({
            "episode_num": ep["episode_num"],
            "episode_id": ep.get("episode_id"),
            "status": "trimmed_out" if ep.get("trimmed_out") else "censored",
            "violations": violation_counts.get(ep["episode_num"], 0),
            "violation_details": per_episode_violations.get(ep["episode_num"], []),
            "highlights": ep.get("highlights", []),
        })

    # ---- Stage 7: Split + Hook insertion (70-85%) ----
    if progress_callback:
        progress_callback(0.72, "splitting_episodes")

    # Split censored video at episode boundaries
    split_paths = split_video_at_boundaries(
        censored_path, episode_durations, speed_factor, drama_dir
    )
    intermediates.extend([p for p in split_paths if p])

    # Extract highlight clips from each split episode
    highlight_clips = {}  # {episode_num: clip_path}
    for i, sp in enumerate(split_paths):
        if not sp or not os.path.exists(sp):
            continue
        ep_num = episode_durations[i]["episode_num"]
        ep_highlights = ep_mapped[i].get("highlights", [])
        hl_clip = os.path.join(drama_dir, f"hl_clip_ep{ep_num}.mp4")
        if extract_hook_clip(
            sp,
            ep_highlights,
            hl_clip,
            config.DRAMA_HOOK_DURATION,
            episode_duration=_get_processed_duration(episode_durations[i], speed_factor),
        ):
            highlight_clips[ep_num] = hl_clip
            intermediates.append(hl_clip)

    if progress_callback:
        progress_callback(0.78, "inserting_hooks")

    # Insert next episode's highlight at each episode's end (except last)
    hooked_paths = []
    for i, sp in enumerate(split_paths):
        if not sp or not os.path.exists(sp):
            hooked_paths.append(sp)
            continue
        ep_num = episode_durations[i]["episode_num"]
        # Get next episode's highlight clip
        next_ep_num = episode_durations[i + 1]["episode_num"] if i + 1 < total_eps else None
        next_hook = highlight_clips.get(next_ep_num) if next_ep_num else None

        if next_hook and os.path.exists(next_hook):
            hooked_out = os.path.join(drama_dir, f"hooked_ep{ep_num}.mp4")
            try:
                concat_with_hook(sp, next_hook, hooked_out)
                hooked_paths.append(hooked_out)
                intermediates.append(hooked_out)
            except Exception as e:
                logger.warning(f"Hook ep{ep_num} failed: {e}, using split directly")
                hooked_paths.append(sp)
        else:
            # Last episode or no highlight available
            hooked_paths.append(sp)

    # ---- Stage 8: Re-concat hooked episodes (85-90%) ----
    if progress_callback:
        progress_callback(0.85, "recombining")
    valid_hooked = [p for p in hooked_paths if p and os.path.exists(p)]
    if len(valid_hooked) > 1:
        recombined_path = os.path.join(drama_dir, "recombined.mp4")
        concat_episodes(valid_hooked, recombined_path)
        intermediates.append(recombined_path)
    elif valid_hooked:
        recombined_path = valid_hooked[0]
    else:
        recombined_path = censored_path

    # ---- Stage 9: Add labels / watermark (90-98%) ----
    if progress_callback:
        progress_callback(0.90, "adding_labels")
    final_path = os.path.join(config.OUTPUT_DIR, f"{drama_id}_drama.mp4")
    try:
        add_labels(recombined_path, final_path, title, disclaimer)
    except Exception as e:
        logger.warning(f"add_labels failed: {e}, using previous output")
        shutil.copy2(recombined_path, final_path)

    if progress_callback:
        progress_callback(1.0, "done")

    # ---- Cleanup intermediate files ----
    for p in intermediates:
        if p and p != final_path and os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass

    logger.info(f"Drama {drama_id}: DONE -> {final_path}")
    return {
        "output_path": final_path,
        "episode_results": episode_results,
        "cloud_usage": censor_result.get("cloud_usage", {}),
    }
