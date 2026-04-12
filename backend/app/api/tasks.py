"""API routes for video censor service."""
import json
import os
import uuid
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from ..models import SessionLocal, Task
from ..worker import process_video_task
from .. import config

router = APIRouter(prefix="/api")


@router.post("/tasks")
async def create_task(file: UploadFile = File(...)):
    """Upload video and create processing task."""
    if not file.filename:
        raise HTTPException(400, "No file provided")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv"):
        raise HTTPException(400, f"Unsupported format: {ext}")

    task_id = str(uuid.uuid4())
    save_path = os.path.join(config.UPLOAD_DIR, f"{task_id}{ext}")

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    db = SessionLocal()
    try:
        task = Task(id=task_id, filename=file.filename, status="pending")
        db.add(task)
        db.commit()
    finally:
        db.close()

    process_video_task.delay(task_id, save_path)
    return {"task_id": task_id, "status": "pending"}


@router.get("/tasks")
async def list_tasks(page: int = 1, size: int = 20):
    """List all tasks with pagination."""
    db = SessionLocal()
    try:
        total = db.query(Task).count()
        tasks = (db.query(Task)
                 .order_by(Task.created_at.desc())
                 .offset((page - 1) * size).limit(size).all())
        return {
            "total": total,
            "page": page,
            "items": [_task_to_dict(t) for t in tasks],
        }
    finally:
        db.close()


@router.get("/tasks/{task_id}")
async def get_task(task_id: str):
    """Get task detail with violations and highlights."""
    db = SessionLocal()
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        if not task:
            raise HTTPException(404, "Task not found")
        return _task_to_dict(task, detail=True)
    finally:
        db.close()


@router.get("/tasks/{task_id}/download")
async def download_result(task_id: str):
    """Download censored video."""
    db = SessionLocal()
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        if not task or not task.output_path:
            raise HTTPException(404, "Output not ready")
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
    finally:
        db.close()


def _task_to_dict(task: Task, detail: bool = False) -> dict:
    d = {
        "id": task.id,
        "filename": task.filename,
        "status": task.status,
        "progress": task.progress,
        "created_at": task.created_at.isoformat() if task.created_at else None,
    }
    if detail:
        d["violations"] = json.loads(task.violations or "[]")
        d["highlights"] = json.loads(task.highlights or "[]")
        d["error"] = task.error
        d["duration"] = task.duration
    return d
