"""Mosaic application using OpenCV frame-by-frame processing + FFmpeg re-encode."""
import logging
import subprocess
import shutil
import tempfile
import os
import cv2
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


def _build_mosaic_map(frame_violations: dict[float, list[dict]],
                      fps: float) -> dict[int, list[tuple]]:
    """Convert timestamp-based violations to frame-number-based mosaic regions.

    Returns: {frame_number: [(x1, y1, x2, y2, block_size, det_type), ...]}
    det_type: "text" for subtitle overlay, "body"/"vlm_region" for pixelation.
    """
    mosaic_map: dict[int, list[tuple]] = {}

    for ts, detections in frame_violations.items():
        for det in detections:
            if not det.get("need_mosaic", False):
                continue
            x1, y1, x2, y2 = [int(v) for v in det["box"]]
            det_type = det.get("det_type", "body")

            # Dynamic padding (smaller for text, larger for body)
            if det_type == "text":
                pad_x = max(5, (x2 - x1) // 15)
                pad_y = max(3, (y2 - y1) // 10)
            else:
                pad_x = max(15, (x2 - x1) // 8)
                pad_y = max(15, (y2 - y1) // 8)
            x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            x2, y2 = x2 + pad_x, y2 + pad_y
            if (x2 - x1) < 10 or (y2 - y1) < 10:
                continue

            # Time windows per type
            if det_type == "text":
                f_start = max(0, int((ts - 0.2) * fps))
                f_end = int((ts + 1.8) * fps)
                blk = 20
            elif det_type == "vlm_region":
                f_start = max(0, int((ts - 0.25) * fps))
                f_end = int((ts + 0.75) * fps)
                blk = 12
            else:
                f_start = max(0, int((ts - 0.5) * fps))
                f_end = int((ts + 1.0) * fps)
                blk = 15

            entry = (x1, y1, x2, y2, blk, det_type)
            for f in range(f_start, f_end + 1):
                if f not in mosaic_map:
                    mosaic_map[f] = []
                mosaic_map[f].append(entry)

    # Gap filling: connect nearby same-region detections
    if mosaic_map:
        max_gap = int(fps * 1.5)
        sorted_frames = sorted(mosaic_map.keys())
        for i, fn in enumerate(sorted_frames):
            for region in mosaic_map[fn]:
                rx1, ry1, rx2, ry2, rblk, rtype = region
                for j in range(i + 1, len(sorted_frames)):
                    fn2 = sorted_frames[j]
                    if fn2 - fn > max_gap:
                        break
                    for region2 in mosaic_map[fn2]:
                        rx1b, ry1b, rx2b, ry2b, rblk2, rtype2 = region2
                        if (rtype == rtype2 and
                                abs(rx1 - rx1b) < 50 and abs(ry1 - ry1b) < 50 and
                                abs(rx2 - rx2b) < 50 and abs(ry2 - ry2b) < 50):
                            for fill_f in range(fn + 1, fn2):
                                if fill_f not in mosaic_map:
                                    mosaic_map[fill_f] = []
                                if region not in mosaic_map[fill_f]:
                                    mosaic_map[fill_f].append(region)

    return mosaic_map


def _blur_region(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                 block_size: int = 15) -> None:
    """Apply pixelated mosaic blur to a region in-place."""
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    rh, rw = roi.shape[:2]
    # Downscale then upscale for pixelation effect
    small = cv2.resize(roi, (max(1, rw // block_size), max(1, rh // block_size)),
                       interpolation=cv2.INTER_LINEAR)
    frame[y1:y2, x1:x2] = cv2.resize(small, (rw, rh),
                                       interpolation=cv2.INTER_NEAREST)
    # For large regions, add gaussian blur to ensure thorough obscuring
    if rw > 200 or rh > 200:
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (21, 21), 0)


def _cover_text_region(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
    """Cover subtitle text with a solid dark bar, making text completely invisible.
    Uses the average background color of surrounding area for a natural look.
    """
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    # Sample background color from a thin strip above the text region
    sample_y1 = max(0, y1 - 10)
    sample_y2 = y1
    if sample_y2 > sample_y1:
        bg_sample = frame[sample_y1:sample_y2, x1:x2]
        avg_color = bg_sample.mean(axis=(0, 1)).astype(np.uint8)
    else:
        avg_color = np.array([0, 0, 0], dtype=np.uint8)
    # Fill with background-matched color (slightly darker for natural blend)
    fill_color = np.clip(avg_color.astype(int) - 20, 0, 255).astype(np.uint8)
    frame[y1:y2, x1:x2] = fill_color


def apply_mosaic(input_video: str, output_video: str,
                 frame_violations: dict[float, list[dict]],
                 fps: float, width: int, height: int) -> bool:
    """Apply mosaic to video using OpenCV frame processing + FFmpeg re-encode.

    Strategy: read video with OpenCV, apply mosaic on flagged frames,
    write raw frames to pipe, let FFmpeg mux with original audio.
    This avoids complex filter_complex chains that break with many regions.
    """
    if not frame_violations:
        logger.info("No frame_violations, copying original video")
        shutil.copy2(input_video, output_video)
        return True

    mosaic_map = _build_mosaic_map(frame_violations, fps)
    if not mosaic_map:
        logger.info("No mosaic regions after building map, copying original video")
        shutil.copy2(input_video, output_video)
        return True

    logger.info(f"Mosaic map: {len(mosaic_map)} frames to process, "
                f"from {len(frame_violations)} violation timestamps")

    # Debug: log all violation timestamps in compact format
    ts_summary = {ts: len(dets) for ts, dets in frame_violations.items()}
    logger.info(f"  violation timestamps ({len(ts_summary)} total): {ts_summary}")

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {input_video}")
        shutil.copy2(input_video, output_video)
        return False

    v_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    v_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    v_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # FFmpeg process: read raw frames from stdin, mux with original audio
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{v_w}x{v_h}", "-r", str(v_fps),
        "-i", "pipe:0",
        "-i", input_video,
        "-map", "0:v", "-map", "1:a?",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "ultrafast", "-crf", "23",
        "-c:a", "copy",
        "-movflags", "+faststart",
        output_video,
    ]

    try:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

        frame_idx = 0
        mosaic_applied = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in mosaic_map:
                for region in mosaic_map[frame_idx]:
                    x1, y1, x2, y2, blk, det_type = region
                    if det_type == "text":
                        # Solid color cover for text: completely hides subtitle
                        _cover_text_region(frame, x1, y1, x2, y2)
                    else:
                        # Pixelated mosaic for body/vlm regions
                        _blur_region(frame, x1, y1, x2, y2, block_size=blk)
                mosaic_applied += 1

            proc.stdin.write(frame.tobytes())
            frame_idx += 1

        cap.release()
        proc.stdin.close()
        proc.wait(timeout=600)

        if proc.returncode != 0:
            stderr = proc.stderr.read().decode(errors="ignore")
            logger.error(f"FFmpeg encode failed: {stderr[:500]}")
            shutil.copy2(input_video, output_video)
            return False

        logger.info(f"Mosaic applied to {mosaic_applied}/{frame_idx} frames, "
                     f"{len(mosaic_map)} unique frames targeted")
        return True

    except Exception as e:
        logger.error(f"Mosaic application failed: {e}")
        shutil.copy2(input_video, output_video)
        return False
