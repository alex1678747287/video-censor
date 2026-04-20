"""Recover orphaned processing jobs that no longer receive progress updates."""
from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from . import config
from .models import Drama, DramaEpisode, Task

logger = logging.getLogger(__name__)

STALE_PROCESSING_ERROR = (
    "\u4efb\u52a1\u957f\u65f6\u95f4\u6ca1\u6709\u8fdb\u5ea6\u66f4\u65b0\uff0c"
    "\u53ef\u80fd\u5df2\u56e0 worker \u91cd\u542f\u6216\u4e2d\u65ad\u800c\u505c\u6b62\uff0c"
    "\u8bf7\u91cd\u65b0\u63d0\u4ea4\u3002"
)


def _normalize_dt(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _last_activity_at(item: Any) -> datetime | None:
    for attr in ("updated_at", "started_at", "created_at"):
        value = _normalize_dt(getattr(item, attr, None))
        if value is not None:
            return value
    return None


def _is_stale_processing(item: Any, *, now: datetime, timeout_seconds: int) -> bool:
    if getattr(item, "status", None) != "processing":
        return False
    if getattr(item, "completed_at", None):
        return False
    last_activity = _last_activity_at(item)
    if last_activity is None:
        return False
    return last_activity <= now - timedelta(seconds=max(1, int(timeout_seconds)))


def _mark_failed(item: Any, *, now: datetime):
    item.status = "failed"
    if hasattr(item, "stage"):
        item.stage = "failed"
    if hasattr(item, "completed_at"):
        item.completed_at = now
    if hasattr(item, "error") and not getattr(item, "error", None):
        item.error = STALE_PROCESSING_ERROR


def _guess_task_upload_path(task: Task) -> str | None:
    _, ext = os.path.splitext(task.filename or "")
    candidates = []
    if ext:
        candidates.append(os.path.join(config.UPLOAD_DIR, f"{task.id}{ext.lower()}"))
    for extra in (".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv"):
        path = os.path.join(config.UPLOAD_DIR, f"{task.id}{extra}")
        if path not in candidates:
            candidates.append(path)
    return next((path for path in candidates if os.path.exists(path)), None)


def _requeue_task(task: Task) -> tuple[str, tuple]:
    from .worker import process_video_task

    upload_path = _guess_task_upload_path(task)
    if not upload_path:
        raise FileNotFoundError(f"Missing upload for task {task.id}")
    new_token = str(uuid.uuid4())
    task.status = "pending"
    task.progress = 0.0
    task.stage = "retrying"
    task.started_at = None
    task.completed_at = None
    task.error = None
    task.output_path = None
    task.highlights = "[]"
    task.violations = "[]"
    task.run_token = new_token
    task.retry_count = int(task.retry_count or 0) + 1
    return (
        "task",
        (task.id, upload_path, new_token, getattr(task, "execution_profile", None)),
        process_video_task,
    )


def _requeue_drama(drama: Drama) -> tuple[str, tuple]:
    from .worker import process_drama_task

    ordered_episodes = sorted(
        [ep for ep in drama.episodes if ep.upload_path],
        key=lambda ep: ep.episode_num,
    )
    if not ordered_episodes:
        raise FileNotFoundError(f"Missing episodes for drama {drama.id}")
    episodes_info = []
    for ep in ordered_episodes:
        if not os.path.exists(ep.upload_path):
            raise FileNotFoundError(f"Missing upload for drama {drama.id} ep{ep.episode_num}")
        ep.status = "pending"
        ep.highlights = "[]"
        ep.violations = "[]"
        ep.violations_count = 0
        episodes_info.append({
            "episode_num": ep.episode_num,
            "video_path": ep.upload_path,
            "episode_id": ep.id,
        })
    new_token = str(uuid.uuid4())
    drama.status = "pending"
    drama.progress = 0.0
    drama.stage = "retrying"
    drama.started_at = None
    drama.completed_at = None
    drama.error = None
    drama.output_path = None
    drama.run_token = new_token
    drama.retry_count = int(drama.retry_count or 0) + 1
    return (
        "drama",
        (
            drama.id,
            episodes_info,
            drama.title,
            drama.speed_factor,
            drama.max_duration,
            drama.disclaimer,
            new_token,
            getattr(drama, "execution_profile", None),
        ),
        process_drama_task,
    )


def _recover_stale_items(
    tasks: list[Any],
    dramas: list[Any],
    *,
    now: datetime | None = None,
    task_timeout_seconds: int | None = None,
    drama_timeout_seconds: int | None = None,
) -> dict[str, int]:
    now = _normalize_dt(now) or _now()
    task_timeout_seconds = int(task_timeout_seconds or config.STALE_TASK_TIMEOUT_SECONDS)
    drama_timeout_seconds = int(drama_timeout_seconds or config.STALE_DRAMA_TIMEOUT_SECONDS)
    recovered = {"tasks": 0, "dramas": 0, "episodes": 0}

    for task in tasks:
        if not _is_stale_processing(task, now=now, timeout_seconds=task_timeout_seconds):
            continue
        _mark_failed(task, now=now)
        recovered["tasks"] += 1

    for drama in dramas:
        if not _is_stale_processing(drama, now=now, timeout_seconds=drama_timeout_seconds):
            continue
        _mark_failed(drama, now=now)
        recovered["dramas"] += 1
        for episode in getattr(drama, "episodes", []) or []:
            if getattr(episode, "status", None) in {"pending", "processing", ""}:
                episode.status = "failed"
                recovered["episodes"] += 1

    return recovered


def recover_stale_processing(db_session) -> dict[str, int]:
    """Mark processing rows as failed when they have been silent for too long."""
    now = _now()
    tasks = db_session.query(Task).filter(Task.status == "processing").all()
    dramas = db_session.query(Drama).filter(Drama.status == "processing").all()
    requeued_calls = []
    recovered = {"tasks": 0, "dramas": 0, "episodes": 0, "requeued_tasks": 0, "requeued_dramas": 0}

    orphan_cutoff = now - timedelta(seconds=config.ORPHAN_REQUEUE_GRACE_SECONDS)
    for task in tasks:
        last_activity = _last_activity_at(task)
        if not last_activity or last_activity > orphan_cutoff:
            continue
        if int(task.retry_count or 0) >= config.MAX_AUTO_REQUEUE_ATTEMPTS:
            _mark_failed(task, now=now)
            recovered["tasks"] += 1
            continue
        try:
            kind, args, task_func = _requeue_task(task)
        except FileNotFoundError:
            _mark_failed(task, now=now)
            recovered["tasks"] += 1
            continue
        requeued_calls.append((kind, task.id, args, task_func))
        recovered["requeued_tasks"] += 1

    for drama in dramas:
        last_activity = _last_activity_at(drama)
        if not last_activity or last_activity > orphan_cutoff:
            continue
        if int(drama.retry_count or 0) >= config.MAX_AUTO_REQUEUE_ATTEMPTS:
            _mark_failed(drama, now=now)
            recovered["dramas"] += 1
            for episode in getattr(drama, "episodes", []) or []:
                if getattr(episode, "status", None) in {"pending", "processing", ""}:
                    episode.status = "failed"
                    recovered["episodes"] += 1
            continue
        try:
            kind, args, task_func = _requeue_drama(drama)
        except FileNotFoundError:
            _mark_failed(drama, now=now)
            recovered["dramas"] += 1
            for episode in getattr(drama, "episodes", []) or []:
                if getattr(episode, "status", None) in {"pending", "processing", ""}:
                    episode.status = "failed"
                    recovered["episodes"] += 1
            continue
        requeued_calls.append((kind, drama.id, args, task_func))
        recovered["requeued_dramas"] += 1

    older_tasks = [task for task in tasks if task.status == "processing"]
    older_dramas = [drama for drama in dramas if drama.status == "processing"]
    stale_counts = _recover_stale_items(older_tasks, older_dramas, now=now)
    recovered["tasks"] += stale_counts["tasks"]
    recovered["dramas"] += stale_counts["dramas"]
    recovered["episodes"] += stale_counts["episodes"]
    if any(recovered.values()):
        db_session.commit()
        for kind, item_id, args, task_func in requeued_calls:
            try:
                task_func.delay(*args)
            except Exception:
                logger.exception("Auto requeue dispatch failed for %s %s", kind, item_id)
    return recovered
