"""Scan videos for likely blood-heavy frames using local color evidence."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
import re

import cv2
import numpy as np


def _find_binary(name: str) -> str | None:
    path = shutil.which(name)
    if path:
        return path
    if not name.lower().endswith(".exe"):
        return shutil.which(f"{name}.exe")
    return None


def _parse_ffmpeg_duration(stderr: str) -> float:
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", stderr)
    if not match:
        return 0.0
    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = float(match.group(3))
    return hours * 3600 + minutes * 60 + seconds


def _probe_duration(video_path: Path) -> float:
    ffprobe_bin = _find_binary("ffprobe")
    if ffprobe_bin:
        cmd = [
            ffprobe_bin,
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        payload = json.loads(result.stdout)
        return float(payload.get("format", {}).get("duration", 0.0) or 0.0)

    ffmpeg_bin = _find_binary("ffmpeg")
    if not ffmpeg_bin:
        raise FileNotFoundError("Neither ffprobe nor ffmpeg is available in PATH")
    cmd = [ffmpeg_bin, "-i", str(video_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return _parse_ffmpeg_duration(result.stderr)


def _extract_frame(video_path: Path, timestamp: float, output_path: Path) -> None:
    ffmpeg_bin = _find_binary("ffmpeg")
    if not ffmpeg_bin:
        raise FileNotFoundError("ffmpeg is not available in PATH")
    cmd = [
        ffmpeg_bin,
        "-y",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-update",
        "1",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _score_frame(image_path: Path) -> dict:
    frame = cv2.imread(str(image_path))
    if frame is None:
        return {
            "red_ratio": 0.0,
            "dark_red_ratio": 0.0,
            "score": 0.0,
        }

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    blue = frame[:, :, 0].astype(np.int16)
    green = frame[:, :, 1].astype(np.int16)
    red = frame[:, :, 2].astype(np.int16)

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

    red_ratio = float(np.mean(vivid_red))
    dark_red_ratio = float(np.mean(dark_red))
    score = dark_red_ratio * 0.7 + red_ratio * 0.3
    return {
        "red_ratio": round(red_ratio, 6),
        "dark_red_ratio": round(dark_red_ratio, 6),
        "score": round(float(score), 6),
    }


def scan_video(video_path: Path, interval: float, temp_dir: Path) -> list[dict]:
    duration = _probe_duration(video_path)
    rows = []
    index = 0
    timestamp = 0.0
    while timestamp < duration:
        frame_path = temp_dir / f"{video_path.stem}_{index:04d}.jpg"
        _extract_frame(video_path, timestamp, frame_path)
        metrics = _score_frame(frame_path)
        rows.append({
            "video": video_path.name,
            "timestamp": round(timestamp, 1),
            **metrics,
            "frame_path": frame_path,
        })
        index += 1
        timestamp += interval
    return rows


def _iter_videos(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(
        path for path in input_path.rglob("*.mp4")
        if path.is_file()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="video file or directory")
    parser.add_argument("--interval", type=float, default=8.0, help="sample interval in seconds")
    parser.add_argument("--top", type=int, default=12, help="number of top candidates to keep")
    parser.add_argument(
        "--output-dir",
        default="data/debug_frames/blood_scan",
        help="directory to save top frames and report",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = _iter_videos(input_path)
    if not videos:
        raise SystemExit(f"No mp4 videos found under {input_path}")

    with tempfile.TemporaryDirectory(prefix="blood_scan_") as tmpdir:
        temp_dir = Path(tmpdir)
        rows = []
        for video_path in videos:
            rows.extend(scan_video(video_path, args.interval, temp_dir))

        rows.sort(key=lambda item: item["score"], reverse=True)
        top_rows = rows[: max(1, args.top)]

        saved = []
        for rank, item in enumerate(top_rows, 1):
            frame_name = f"{rank:02d}_{Path(item['video']).stem}_t{int(item['timestamp']):03d}.jpg"
            target = output_dir / frame_name
            shutil.copy2(item["frame_path"], target)
            saved.append({
                "rank": rank,
                "video": item["video"],
                "timestamp": item["timestamp"],
                "score": item["score"],
                "red_ratio": item["red_ratio"],
                "dark_red_ratio": item["dark_red_ratio"],
                "frame": target.name,
            })

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(saved, ensure_ascii=False, indent=2))
    print(f"\nSaved report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
