"""Celery worker for async video processing."""
import json
import logging
import threading
from datetime import datetime, timezone
from celery import Celery
from . import config
from .models import SessionLocal, Task
from .core.pipeline import process_video
from .core.drama_pipeline import process_drama

logger = logging.getLogger(__name__)

celery_app = Celery("video_censor", broker=config.REDIS_URL, backend=config.REDIS_URL)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    broker_connection_retry_on_startup=True,
    broker_transport_options={
        "visibility_timeout": config.CELERY_VISIBILITY_TIMEOUT_SECONDS,
    },
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _update_task(task_id: str, force_touch: bool = False, run_token: str | None = None, **kwargs):
    db = SessionLocal()
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        if task:
            if run_token and task.run_token and task.run_token != run_token:
                return False
            for k, v in kwargs.items():
                setattr(task, k, v)
            if force_touch:
                task.updated_at = _utcnow()
            db.commit()
            return True
    except Exception as e:
        logger.error(f"Failed to update task {task_id}: {e}")
        db.rollback()
    finally:
        db.close()
    return False


class _ProgressHeartbeat:
    """Periodically touches a running task so long stages stay observable."""

    def __init__(self, updater, item_id: str, *, interval_seconds: int | None = None,
                 run_token: str | None = None):
        self._updater = updater
        self._item_id = item_id
        self._run_token = run_token
        self._interval = max(
            0.05, float(interval_seconds or config.TASK_HEARTBEAT_INTERVAL_SECONDS)
        )
        self._state = {"progress": 0.0, "stage": "initializing"}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def update(self, *, progress: float | None = None, stage: str | None = None):
        with self._lock:
            if progress is not None:
                self._state["progress"] = progress
            if stage is not None:
                self._state["stage"] = stage

    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=self._interval + 1.0)

    def _snapshot(self) -> dict:
        with self._lock:
            return dict(self._state)

    def _run(self):
        while not self._stop_event.wait(self._interval):
            state = self._snapshot()
            self._updater(
                self._item_id,
                progress=state.get("progress"),
                stage=state.get("stage"),
                force_touch=True,
                run_token=self._run_token,
            )


@celery_app.task(name="process_video_task", bind=True)
def process_video_task(self, task_id: str, video_path: str, run_token: str | None = None,
                       execution_profile: str | None = None):
    """Celery task to process a video."""
    logger.info(f"Starting task {task_id}")
    claimed = _update_task(
        task_id,
        status="processing",
        progress=0.0,
        stage="initializing",
        started_at=_utcnow(),
        completed_at=None,
        run_token=run_token,
    )
    if not claimed:
        logger.warning("Ignoring stale video task run: %s token=%s", task_id, run_token)
        return {"status": "ignored", "task_id": task_id}
    heartbeat = _ProgressHeartbeat(_update_task, task_id, run_token=run_token)
    heartbeat.start()

    def on_progress(progress: float, stage: str):
        heartbeat.update(progress=progress, stage=stage)
        _update_task(task_id, progress=progress, stage=stage, run_token=run_token)
        self.update_state(state="PROGRESS", meta={"progress": progress, "stage": stage})

    try:
        result = process_video(
            task_id, video_path,
            progress_callback=on_progress,
            execution_profile=execution_profile,
        )
        _update_task(
            task_id,
            status="done",
            progress=1.0,
            stage="done",
            output_path=result["output_path"],
            violations=json.dumps(result["violations"], ensure_ascii=False),
            highlights=json.dumps(result["highlights"], ensure_ascii=False),
            cloud_usage=json.dumps(result.get("cloud_usage", {}), ensure_ascii=False),
            completed_at=_utcnow(),
            run_token=run_token,
        )
        return {"status": "done", "task_id": task_id}
    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}", exc_info=True)
        _update_task(
            task_id,
            status="failed",
            stage="failed",
            error=str(e)[:2000],
            completed_at=_utcnow(),
            run_token=run_token,
        )
        return {"status": "failed", "task_id": task_id, "error": "Processing failed"}
    finally:
        heartbeat.stop()


def _update_drama(drama_id: str, force_touch: bool = False, run_token: str | None = None, **kwargs):
    from .models import Drama
    db = SessionLocal()
    try:
        drama = db.query(Drama).filter(Drama.id == drama_id).first()
        if drama:
            if run_token and drama.run_token and drama.run_token != run_token:
                return False
            for k, v in kwargs.items():
                setattr(drama, k, v)
            if force_touch:
                drama.updated_at = _utcnow()
            db.commit()
            return True
    except Exception as e:
        logger.error(f"Failed to update drama {drama_id}: {e}")
        db.rollback()
    finally:
        db.close()
    return False


def _update_episode(episode_id: str, **kwargs):
    from .models import DramaEpisode
    db = SessionLocal()
    try:
        ep = db.query(DramaEpisode).filter(DramaEpisode.id == episode_id).first()
        if ep:
            for k, v in kwargs.items():
                setattr(ep, k, v)
            db.commit()
    except Exception as e:
        logger.error(f"Failed to update episode {episode_id}: {e}")
        db.rollback()
    finally:
        db.close()


def _build_episode_updates(er: dict) -> dict:
    """Build DB update payload for one drama episode result."""
    updates = {"status": er.get("status", "done")}
    if "highlights" in er:
        updates["highlights"] = json.dumps(er.get("highlights", []), ensure_ascii=False)
    if "violation_details" in er:
        updates["violations"] = json.dumps(er.get("violation_details", []), ensure_ascii=False)
    if "violations" in er:
        updates["violations_count"] = int(er.get("violations") or 0)
    return updates


@celery_app.task(name="process_drama_task", bind=True)
def process_drama_task(self, drama_id: str, episodes_info: list,
                       title: str, speed_factor: float,
                       max_duration: int, disclaimer: str,
                       run_token: str | None = None,
                       execution_profile: str | None = None):
    """Celery task to process a full drama."""
    logger.info(f"Starting drama task {drama_id} ({len(episodes_info)} episodes)")
    claimed = _update_drama(
        drama_id,
        status="processing",
        progress=0.0,
        stage="initializing",
        started_at=_utcnow(),
        completed_at=None,
        run_token=run_token,
    )
    if not claimed:
        logger.warning("Ignoring stale drama task run: %s token=%s", drama_id, run_token)
        return {"status": "ignored", "drama_id": drama_id}
    heartbeat = _ProgressHeartbeat(_update_drama, drama_id, run_token=run_token)
    heartbeat.start()

    def on_progress(progress: float, stage: str):
        heartbeat.update(progress=progress, stage=stage)
        _update_drama(drama_id, progress=progress, stage=stage, run_token=run_token)
        self.update_state(state="PROGRESS", meta={"progress": progress, "stage": stage})

    try:
        result = process_drama(
            drama_id, episodes_info, title,
            speed_factor, max_duration, disclaimer,
            progress_callback=on_progress,
            execution_profile=execution_profile,
        )
        # Update episode statuses and highlights
        for er in result.get("episode_results", []):
            if er.get("episode_id"):
                _update_episode(er["episode_id"], **_build_episode_updates(er))

        _update_drama(
            drama_id, status="done", progress=1.0, stage="done",
            output_path=result["output_path"],
            cloud_usage=json.dumps(result.get("cloud_usage", {}), ensure_ascii=False),
            completed_at=_utcnow(),
            run_token=run_token,
        )
        return {"status": "done", "drama_id": drama_id}
    except Exception as e:
        logger.error(f"Drama task {drama_id} failed: {e}", exc_info=True)
        _update_drama(
            drama_id,
            status="failed",
            error=str(e)[:2000],
            stage="failed",
            completed_at=_utcnow(),
            run_token=run_token,
        )
        return {"status": "failed", "drama_id": drama_id, "error": "Processing failed"}
    finally:
        heartbeat.stop()
