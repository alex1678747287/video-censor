"""API routes for video censor service."""
import json
import os
import re
import uuid
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from ..models import SessionLocal, Task
from ..queue_metrics import build_queue_metrics
from ..task_recovery import recover_stale_processing
from ..worker import process_video_task
from .. import config

router = APIRouter(prefix="/api")

ALLOWED_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv"}
MAX_FILE_SIZE = 4 * 1024 * 1024 * 1024  # 4GB
UUID_RE = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')


def _normalize_execution_profile(value: str | None) -> str:
    profile = str(value or config.CLOUD_EXECUTION_PROFILE).strip().lower()
    return profile if profile in {"local_first", "balanced"} else config.CLOUD_EXECUTION_PROFILE


@router.post("/tasks")
async def create_task(
    file: UploadFile = File(...),
    execution_profile: str = Form(config.CLOUD_EXECUTION_PROFILE),
):
    """Upload video and create processing task."""
    if not file.filename:
        raise HTTPException(400, "No file provided")

    # Sanitize filename to prevent path traversal in response headers
    safe_name = os.path.basename(file.filename)
    ext = os.path.splitext(safe_name)[1].lower() or ".mp4"
    if ext not in ALLOWED_EXTS:
        raise HTTPException(400, f"Unsupported format: {ext}")

    task_id = str(uuid.uuid4())
    run_token = str(uuid.uuid4())
    execution_profile = _normalize_execution_profile(execution_profile)
    save_path = os.path.join(config.UPLOAD_DIR, f"{task_id}{ext}")

    # Stream write with size limit
    written = 0
    try:
        with open(save_path, "wb") as f:
            while chunk := file.file.read(8 * 1024 * 1024):  # 8MB chunks
                written += len(chunk)
                if written > MAX_FILE_SIZE:
                    f.close()
                    os.remove(save_path)
                    raise HTTPException(413, f"File too large (max {MAX_FILE_SIZE // (1024**3)}GB)")
                f.write(chunk)
    finally:
        await file.close()

    db = SessionLocal()
    try:
        duration = None
        try:
            from ..core.pipeline import get_video_info
            duration = float(get_video_info(save_path).get("duration") or 0.0) or None
        except Exception:
            duration = None
        task = Task(id=task_id, filename=safe_name, status="pending")
        task.duration = duration
        task.run_token = run_token
        task.retry_count = 0
        task.execution_profile = execution_profile
        db.add(task)
        db.commit()
    except Exception:
        if os.path.exists(save_path):
            os.remove(save_path)
        db.rollback()
        raise
    finally:
        db.close()

    process_video_task.delay(task_id, save_path, run_token, execution_profile)
    return {"task_id": task_id, "status": "pending", "execution_profile": execution_profile}


@router.get("/tasks")
async def list_tasks(page: int = 1, size: int = 20):
    """List all tasks with pagination."""
    page = max(1, page)
    size = max(1, min(size, 100))
    db = SessionLocal()
    try:
        recover_stale_processing(db)
        total = db.query(Task).count()
        queue_metrics = build_queue_metrics(db)
        tasks = (db.query(Task)
                 .order_by(Task.created_at.desc())
                 .offset((page - 1) * size).limit(size).all())
        return {
            "total": total,
            "page": page,
            "items": [_task_to_dict(t, queue_metrics=queue_metrics) for t in tasks],
        }
    finally:
        db.close()


@router.get("/tasks/{task_id}")
async def get_task(task_id: str):
    """Get task detail with violations and highlights."""
    if not UUID_RE.match(task_id):
        raise HTTPException(400, "Invalid task ID")
    db = SessionLocal()
    try:
        recover_stale_processing(db)
        queue_metrics = build_queue_metrics(db)
        task = db.query(Task).filter(Task.id == task_id).first()
        if not task:
            raise HTTPException(404, "Task not found")
        return _task_to_dict(task, detail=True, queue_metrics=queue_metrics)
    finally:
        db.close()


def _validate_output_path(path: str) -> bool:
    """Validate output path is within allowed directories."""
    abs_path = os.path.abspath(path)
    allowed = [os.path.abspath(config.OUTPUT_DIR) + os.sep]
    return any(abs_path.startswith(d) for d in allowed)


@router.get("/tasks/{task_id}/download")
async def download_result(task_id: str):
    """Download censored video."""
    if not UUID_RE.match(task_id):
        raise HTTPException(400, "Invalid task ID")
    db = SessionLocal()
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        if not task or not task.output_path:
            raise HTTPException(404, "Output not ready")
        if not _validate_output_path(task.output_path):
            raise HTTPException(403, "Invalid output path")
        if not os.path.exists(task.output_path):
            raise HTTPException(404, "Output file missing")
        return FileResponse(
            task.output_path,
            media_type="video/mp4",
            filename=f"censored_{task.filename}",
        )
    finally:
        db.close()


@router.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    """Delete a task and its associated files."""
    if not UUID_RE.match(task_id):
        raise HTTPException(400, "Invalid task ID")
    db = SessionLocal()
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        if not task:
            raise HTTPException(404, "Task not found")
        # Remove uploaded file
        for ext in (".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv"):
            p = os.path.join(config.UPLOAD_DIR, f"{task_id}{ext}")
            if os.path.exists(p):
                os.remove(p)
        # Remove output file
        if task.output_path and os.path.exists(task.output_path):
            os.remove(task.output_path)
        db.delete(task)
        db.commit()
        return {"ok": True}
    except HTTPException:
        raise
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _task_to_dict(task: Task, detail: bool = False, queue_metrics: dict | None = None) -> dict:
    metric = (queue_metrics or {}).get(f"task:{task.id}", {})
    cloud_usage = json.loads(task.cloud_usage or "{}")
    d = {
        "id": task.id,
        "filename": task.filename,
        "status": task.status,
        "execution_profile": task.execution_profile or config.CLOUD_EXECUTION_PROFILE,
        "progress": task.progress,
        "stage": task.stage or "",
        "created_at": task.created_at.isoformat() if task.created_at else None,
        "started_at": task.started_at.isoformat() if task.started_at else None,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
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
        d["violations"] = json.loads(task.violations or "[]")
        d["highlights"] = json.loads(task.highlights or "[]")
        d["cloud_usage"] = cloud_usage
        d["error"] = task.error
        d["duration"] = task.duration
    return d
