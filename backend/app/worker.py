"""Celery worker for async video processing."""
import json
import logging
from celery import Celery
from . import config
from .models import SessionLocal, Task
from .core.pipeline import process_video

logger = logging.getLogger(__name__)

celery_app = Celery("video_censor", broker=config.REDIS_URL, backend=config.REDIS_URL)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    worker_prefetch_multiplier=1,
)


def _update_task(task_id: str, **kwargs):
    db = SessionLocal()
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        if task:
            for k, v in kwargs.items():
                setattr(task, k, v)
            db.commit()
    finally:
        db.close()


@celery_app.task(name="process_video_task", bind=True)
def process_video_task(self, task_id: str, video_path: str):
    """Celery task to process a video."""
    logger.info(f"Starting task {task_id}")
    _update_task(task_id, status="processing", progress=0.0)

    def on_progress(progress: float, stage: str):
        _update_task(task_id, progress=progress)
        self.update_state(state="PROGRESS", meta={"progress": progress, "stage": stage})

    try:
        result = process_video(task_id, video_path, progress_callback=on_progress)
        _update_task(
            task_id,
            status="done",
            progress=1.0,
            output_path=result["output_path"],
            violations=json.dumps(result["violations"], ensure_ascii=False),
            highlights=json.dumps(result["highlights"], ensure_ascii=False),
        )
        return {"status": "done", "task_id": task_id}
    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}", exc_info=True)
        _update_task(task_id, status="failed", error=str(e))
        return {"status": "failed", "task_id": task_id, "error": str(e)}
