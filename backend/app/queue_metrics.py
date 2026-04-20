"""Queue position and ETA helpers for shared task/drama processing."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from heapq import heappop, heappush
from typing import Any

from . import config
from .models import Drama, Task


def _normalize_dt(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _seconds_between(start: datetime | None, end: datetime | None) -> float | None:
    start = _normalize_dt(start)
    end = _normalize_dt(end)
    if start is None or end is None:
        return None
    return max(0.0, (end - start).total_seconds())


def _mean(values: list[float], fallback: float) -> float:
    cleaned = [float(v) for v in values if v is not None and v > 0]
    if not cleaned:
        return fallback
    return sum(cleaned) / len(cleaned)


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _build_job(kind: str, item: Any) -> dict[str, Any]:
    if kind == "task":
        input_seconds = float(item.duration or 0.0) if getattr(item, "duration", None) else None
        episode_count = 1
    else:
        source_duration = float(item.source_duration or 0.0) if getattr(item, "source_duration", None) else None
        effective_limit = float(item.max_duration or 0.0) if getattr(item, "max_duration", None) else None
        if source_duration and effective_limit:
            input_seconds = min(source_duration, effective_limit)
        else:
            input_seconds = source_duration or effective_limit or None
        episode_count = max(1, int((item.episode_end or 1) - (item.episode_start or 1) + 1))
    return {
        "key": f"{kind}:{item.id}",
        "id": item.id,
        "kind": kind,
        "status": item.status,
        "created_at": _normalize_dt(item.created_at),
        "started_at": _normalize_dt(getattr(item, "started_at", None)),
        "completed_at": _normalize_dt(getattr(item, "completed_at", None)),
        "input_seconds": input_seconds,
        "episode_count": episode_count,
    }


def _build_profiles(jobs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    profiles: dict[str, dict[str, float]] = {}
    for kind, fallback_runtime, fallback_ratio in (
        ("task", float(config.ETA_DEFAULT_SINGLE_SECONDS), 2.4),
        ("drama", float(config.ETA_DEFAULT_DRAMA_SECONDS), 1.4),
    ):
        sample_jobs = [job for job in jobs if job["kind"] == kind]
        runtimes = []
        ratios = []
        for job in sample_jobs:
            runtime = _seconds_between(job["started_at"], job["completed_at"])
            if runtime is None:
                continue
            runtimes.append(runtime)
            input_seconds = float(job["input_seconds"] or 0.0)
            if input_seconds >= 30:
                ratios.append(runtime / input_seconds)
        profiles[kind] = {
            "avg_runtime": _mean(runtimes, fallback_runtime),
            "avg_ratio": _mean(ratios, fallback_ratio),
        }
    return profiles


def _estimate_total_seconds(job: dict[str, Any], profiles: dict[str, dict[str, float]]) -> float:
    profile = profiles[job["kind"]]
    input_seconds = float(job["input_seconds"] or 0.0)
    if job["kind"] == "task":
        if input_seconds > 0:
            estimate = input_seconds * profile["avg_ratio"]
        else:
            estimate = profile["avg_runtime"]
        return _clip(estimate, 60.0, 3600.0)

    if input_seconds > 0:
        estimate = input_seconds * profile["avg_ratio"]
    else:
        estimate = profile["avg_runtime"]
    estimate = max(estimate, float(job["episode_count"] or 1) * 45.0)
    return _clip(estimate, 180.0, 4 * 3600.0)


def _estimate_remaining_seconds(total_seconds: float, elapsed_seconds: float) -> float:
    remaining = total_seconds - max(0.0, elapsed_seconds)
    if remaining > 0:
        return remaining
    return max(float(config.ETA_MIN_REMAINING_SECONDS), total_seconds * 0.12)


def _serialize_eta(now: datetime, seconds_from_now: float) -> str:
    return (now + timedelta(seconds=max(0.0, seconds_from_now))).isoformat()


def build_queue_metrics_for_jobs(
    jobs: list[dict[str, Any]], now: datetime | None = None, slots: int | None = None
) -> dict[str, dict[str, Any]]:
    now = _normalize_dt(now) or _now()
    slots = max(1, int(slots or config.PROCESSING_SLOTS))
    jobs = sorted(
        jobs,
        key=lambda job: (
            job["created_at"] or now,
            job["started_at"] or now,
            job["kind"],
            job["id"],
        ),
    )
    profiles = _build_profiles(jobs)
    metrics: dict[str, dict[str, Any]] = {}

    for job in jobs:
        actual_runtime = _seconds_between(job["started_at"], job["completed_at"])
        metrics[job["key"]] = {
            "processing_slots": slots,
            "queue_position": 0,
            "waiting_count": 0,
            "estimated_duration_seconds": round(_estimate_total_seconds(job, profiles)),
            "estimated_wait_seconds": 0,
            "estimated_remaining_seconds": None,
            "estimated_start_at": None,
            "estimated_finish_at": None,
            "actual_runtime_seconds": round(actual_runtime) if actual_runtime is not None else None,
        }

    processing_jobs = [job for job in jobs if job["status"] == "processing"]
    pending_jobs = [job for job in jobs if job["status"] == "pending"]
    processing_jobs.sort(key=lambda job: (job["started_at"] or job["created_at"] or now, job["id"]))
    pending_jobs.sort(key=lambda job: (job["created_at"] or now, job["id"]))

    slot_heap: list[float] = []
    for job in processing_jobs:
        total_seconds = _estimate_total_seconds(job, profiles)
        elapsed = _seconds_between(job["started_at"], now) or 0.0
        remaining = _estimate_remaining_seconds(total_seconds, elapsed)
        heappush(slot_heap, remaining)
        metrics[job["key"]].update({
            "estimated_duration_seconds": round(total_seconds),
            "estimated_remaining_seconds": round(remaining),
            "estimated_start_at": job["started_at"].isoformat() if job["started_at"] else now.isoformat(),
            "estimated_finish_at": _serialize_eta(now, remaining),
        })

    while len(slot_heap) < slots:
        heappush(slot_heap, 0.0)

    for index, job in enumerate(pending_jobs):
        slot_available_at = heappop(slot_heap) if slot_heap else 0.0
        total_seconds = _estimate_total_seconds(job, profiles)
        wait_seconds = max(0.0, slot_available_at)
        finish_seconds = wait_seconds + total_seconds
        heappush(slot_heap, finish_seconds)
        metrics[job["key"]].update({
            "queue_position": index + 1,
            "waiting_count": index,
            "estimated_duration_seconds": round(total_seconds),
            "estimated_wait_seconds": round(wait_seconds),
            "estimated_remaining_seconds": round(finish_seconds),
            "estimated_start_at": _serialize_eta(now, wait_seconds),
            "estimated_finish_at": _serialize_eta(now, finish_seconds),
        })

    return metrics


def build_queue_metrics(db_session) -> dict[str, dict[str, Any]]:
    jobs = []
    jobs.extend(_build_job("task", task) for task in db_session.query(Task).all())
    jobs.extend(_build_job("drama", drama) for drama in db_session.query(Drama).all())
    return build_queue_metrics_for_jobs(jobs)
