"""API routes for drama (multi-episode) processing."""
import json
import os
import re
import uuid
import shutil
from typing import List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from ..models import SessionLocal, Drama, DramaEpisode
from ..queue_metrics import build_queue_metrics
from ..task_recovery import recover_stale_processing
from ..worker import process_drama_task
from .. import config

router = APIRouter(prefix="/api")

ALLOWED_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv"}
MAX_FILE_SIZE = 4 * 1024 * 1024 * 1024  # 4GB per file
UUID_RE = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')


def _normalize_execution_profile(value: str | None) -> str:
    profile = str(value or config.CLOUD_EXECUTION_PROFILE).strip().lower()
    return profile if profile in {"local_first", "balanced"} else config.CLOUD_EXECUTION_PROFILE


@router.post("/dramas")
async def create_drama(
    files: List[UploadFile] = File(...),
    title: str = Form(...),
    speed_factor: float = Form(config.DRAMA_SPEED_DEFAULT),
    max_duration: int = Form(900),
    execution_profile: str = Form(config.CLOUD_EXECUTION_PROFILE),
    disclaimer: str = Form("热门漫剧 影视效果 无任何不良引导 请勿模仿"),
    episode_start: int = Form(1),
    episode_end: int = Form(0),
):
    """Upload multiple episodes and create drama processing task."""
    if not files:
        raise HTTPException(400, "No files provided")

    # Fix encoding: python-multipart may decode UTF-8 bytes as Latin-1,
    # or Windows curl may send GBK bytes which also get Latin-1 decoded
    def _fix_encoding(s: str) -> str:
        try:
            raw = s.encode('latin-1')
        except (UnicodeDecodeError, UnicodeEncodeError):
            return s
        # Try UTF-8 first, then GBK (Windows system encoding)
        for codec in ('utf-8', 'gbk', 'gb18030'):
            try:
                return raw.decode(codec)
            except (UnicodeDecodeError, LookupError):
                continue
        return s
    title = _fix_encoding(title)
    disclaimer = _fix_encoding(disclaimer)

    if episode_start < 1:
        raise HTTPException(400, "episode_start must be >= 1")
    if not (0.5 <= speed_factor <= 4.0):
        raise HTTPException(400, "speed_factor must be between 0.5 and 4.0")
    if max_duration < 60 or max_duration > 7200:
        raise HTTPException(400, "max_duration must be between 60 and 7200")

    drama_id = str(uuid.uuid4())
    run_token = str(uuid.uuid4())
    execution_profile = _normalize_execution_profile(execution_profile)
    drama_dir = os.path.join(config.DRAMA_DIR, drama_id)
    os.makedirs(drama_dir, exist_ok=True)

    if episode_end <= 0:
        episode_end = episode_start + len(files) - 1

    db = SessionLocal()
    try:
        from ..core.pipeline import get_video_info

        total_source_duration = 0.0
        drama = Drama(
            id=drama_id, title=title, status="pending",
            episode_start=episode_start, episode_end=episode_end,
            speed_factor=speed_factor, max_duration=max_duration,
            disclaimer=disclaimer,
        )
        drama.run_token = run_token
        drama.retry_count = 0
        drama.execution_profile = execution_profile
        db.add(drama)

        episodes_info = []
        for i, f in enumerate(files):
            ep_num = episode_start + i
            fname = _fix_encoding(f.filename or f"episode_{ep_num}.mp4")
            ext = os.path.splitext(fname)[1].lower() or ".mp4"
            if ext not in ALLOWED_EXTS:
                raise HTTPException(400, f"Unsupported format: {ext}")
            ep_id = str(uuid.uuid4())
            save_path = os.path.join(drama_dir, f"ep{ep_num}{ext}")
            # Stream write with size limit
            written = 0
            try:
                with open(save_path, "wb") as out:
                    while chunk := f.file.read(8 * 1024 * 1024):  # 8MB chunks
                        written += len(chunk)
                        if written > MAX_FILE_SIZE:
                            out.close()
                            os.remove(save_path)
                            raise HTTPException(413, f"File too large (max {MAX_FILE_SIZE // (1024**3)}GB)")
                        out.write(chunk)
            finally:
                await f.close()
            try:
                total_source_duration += float(get_video_info(save_path).get("duration") or 0.0)
            except Exception:
                pass
            episode = DramaEpisode(
                id=ep_id, drama_id=drama_id, episode_num=ep_num,
                filename=fname,
                status="pending", upload_path=save_path,
            )
            db.add(episode)
            episodes_info.append({
                "episode_num": ep_num,
                "video_path": save_path,
                "episode_id": ep_id,
            })

        drama.source_duration = total_source_duration or None
        db.commit()
    except Exception:
        # Clean up orphan files on failure
        import shutil as _shutil
        _shutil.rmtree(drama_dir, ignore_errors=True)
        db.rollback()
        raise
    finally:
        db.close()

    process_drama_task.delay(
        drama_id, episodes_info, title,
        speed_factor, max_duration, disclaimer, run_token, execution_profile,
    )
    return {
        "drama_id": drama_id,
        "status": "pending",
        "episodes": len(files),
        "execution_profile": execution_profile,
    }


@router.get("/dramas")
async def list_dramas(page: int = 1, size: int = 20):
    page = max(1, page)
    size = max(1, min(size, 100))
    db = SessionLocal()
    try:
        recover_stale_processing(db)
        total = db.query(Drama).count()
        queue_metrics = build_queue_metrics(db)
        dramas = (db.query(Drama)
                  .order_by(Drama.created_at.desc())
                  .offset((page - 1) * size).limit(size).all())
        return {
            "total": total, "page": page,
            "items": [_drama_to_dict(d, queue_metrics=queue_metrics) for d in dramas],
        }
    finally:
        db.close()


@router.get("/dramas/{drama_id}")
async def get_drama(drama_id: str):
    if not UUID_RE.match(drama_id):
        raise HTTPException(400, "Invalid drama ID")
    db = SessionLocal()
    try:
        recover_stale_processing(db)
        queue_metrics = build_queue_metrics(db)
        drama = db.query(Drama).filter(Drama.id == drama_id).first()
        if not drama:
            raise HTTPException(404, "Drama not found")
        return _drama_to_dict(drama, detail=True, queue_metrics=queue_metrics)
    finally:
        db.close()


@router.get("/dramas/{drama_id}/download")
async def download_drama(drama_id: str):
    if not UUID_RE.match(drama_id):
        raise HTTPException(400, "Invalid drama ID")
    db = SessionLocal()
    try:
        drama = db.query(Drama).filter(Drama.id == drama_id).first()
        if not drama or not drama.output_path:
            raise HTTPException(404, "Output not ready")
        # Validate output path is within allowed directory
        abs_path = os.path.abspath(drama.output_path)
        allowed_prefix = os.path.abspath(config.OUTPUT_DIR) + os.sep
        if not abs_path.startswith(allowed_prefix):
            raise HTTPException(403, "Invalid output path")
        if not os.path.exists(drama.output_path):
            raise HTTPException(404, "Output file missing")
        return FileResponse(
            drama.output_path, media_type="video/mp4",
            filename=f"{drama.title}_合集.mp4",
        )
    finally:
        db.close()


@router.delete("/dramas/{drama_id}")
async def delete_drama(drama_id: str):
    if not UUID_RE.match(drama_id):
        raise HTTPException(400, "Invalid drama ID")
    db = SessionLocal()
    try:
        drama = db.query(Drama).filter(Drama.id == drama_id).first()
        if not drama:
            raise HTTPException(404, "Drama not found")
        # Remove drama directory
        drama_dir = os.path.join(config.DRAMA_DIR, drama_id)
        if os.path.exists(drama_dir):
            shutil.rmtree(drama_dir, ignore_errors=True)
        if drama.output_path and os.path.exists(drama.output_path):
            os.remove(drama.output_path)
        # Remove episodes and drama
        db.query(DramaEpisode).filter(DramaEpisode.drama_id == drama_id).delete()
        db.delete(drama)
        db.commit()
        return {"ok": True}
    finally:
        db.close()


def _drama_to_dict(drama: Drama, detail: bool = False, queue_metrics: dict | None = None) -> dict:
    metric = (queue_metrics or {}).get(f"drama:{drama.id}", {})
    cloud_usage = json.loads(drama.cloud_usage or "{}")
    d = {
        "id": drama.id,
        "title": drama.title,
        "status": drama.status,
        "execution_profile": drama.execution_profile or config.CLOUD_EXECUTION_PROFILE,
        "progress": drama.progress,
        "stage": drama.stage or "",
        "episode_start": drama.episode_start,
        "episode_end": drama.episode_end,
        "speed_factor": drama.speed_factor,
        "max_duration": drama.max_duration,
        "source_duration": drama.source_duration,
        "created_at": drama.created_at.isoformat() if drama.created_at else None,
        "started_at": drama.started_at.isoformat() if drama.started_at else None,
        "completed_at": drama.completed_at.isoformat() if drama.completed_at else None,
        "queue_position": metric.get("queue_position", 0),
        "waiting_count": metric.get("waiting_count", 0),
        "processing_slots": metric.get("processing_slots", config.PROCESSING_SLOTS),
        "estimated_duration_seconds": metric.get("estimated_duration_seconds"),
        "estimated_wait_seconds": metric.get("estimated_wait_seconds", 0),
        "estimated_remaining_seconds": metric.get("estimated_remaining_seconds"),
        "estimated_start_at": metric.get("estimated_start_at"),
        "estimated_finish_at": metric.get("estimated_finish_at"),
        "actual_runtime_seconds": metric.get("actual_runtime_seconds"),
        "cloud_profile": cloud_usage.get("cloud_profile"),
        "local_gpu_profile": cloud_usage.get("local_gpu_profile"),
        "estimated_cloud_cost_cny": cloud_usage.get("estimated_cost_cny"),
    }
    if detail:
        d["disclaimer"] = drama.disclaimer
        d["cloud_usage"] = cloud_usage
        d["error"] = drama.error
        d["episodes"] = [
            {
                "id": ep.id,
                "episode_num": ep.episode_num,
                "filename": ep.filename,
                "status": ep.status,
                "highlights": json.loads(ep.highlights or "[]"),
                "violations": json.loads(ep.violations or "[]"),
                "violations_count": ep.violations_count or 0,
            }
            for ep in drama.episodes
        ]
    return d
