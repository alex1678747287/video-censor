from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest import TestCase

from app.task_recovery import STALE_PROCESSING_ERROR, _recover_stale_items


class TaskRecoveryTests(TestCase):
    def test_recover_stale_processing_marks_failed_and_updates_episodes(self):
        now = datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc)
        task = SimpleNamespace(
            status="processing",
            stage="vlm_detection",
            updated_at=now - timedelta(hours=5),
            started_at=now - timedelta(hours=6),
            created_at=now - timedelta(hours=6, minutes=5),
            completed_at=None,
            error=None,
        )
        drama = SimpleNamespace(
            status="processing",
            stage="censoring_vlm_detection",
            updated_at=now - timedelta(hours=9),
            started_at=now - timedelta(hours=10),
            created_at=now - timedelta(hours=10, minutes=10),
            completed_at=None,
            error=None,
            episodes=[
                SimpleNamespace(status="pending"),
                SimpleNamespace(status="processing"),
                SimpleNamespace(status="censored"),
            ],
        )

        recovered = _recover_stale_items(
            [task],
            [drama],
            now=now,
            task_timeout_seconds=4 * 3600,
            drama_timeout_seconds=8 * 3600,
        )

        self.assertEqual(recovered, {"tasks": 1, "dramas": 1, "episodes": 2})
        self.assertEqual(task.status, "failed")
        self.assertEqual(task.stage, "failed")
        self.assertEqual(task.completed_at, now)
        self.assertEqual(task.error, STALE_PROCESSING_ERROR)
        self.assertEqual(drama.status, "failed")
        self.assertEqual(drama.stage, "failed")
        self.assertEqual(drama.completed_at, now)
        self.assertEqual(drama.error, STALE_PROCESSING_ERROR)
        self.assertEqual([ep.status for ep in drama.episodes], ["failed", "failed", "censored"])

    def test_recent_processing_items_are_not_recovered(self):
        now = datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc)
        task = SimpleNamespace(
            status="processing",
            stage="vlm_detection",
            updated_at=now - timedelta(minutes=30),
            started_at=now - timedelta(hours=1),
            created_at=now - timedelta(hours=1, minutes=5),
            completed_at=None,
            error=None,
        )
        drama = SimpleNamespace(
            status="processing",
            stage="censoring_vlm_detection",
            updated_at=now - timedelta(hours=2),
            started_at=now - timedelta(hours=3),
            created_at=now - timedelta(hours=3, minutes=10),
            completed_at=None,
            error=None,
            episodes=[SimpleNamespace(status="pending")],
        )

        recovered = _recover_stale_items(
            [task],
            [drama],
            now=now,
            task_timeout_seconds=4 * 3600,
            drama_timeout_seconds=8 * 3600,
        )

        self.assertEqual(recovered, {"tasks": 0, "dramas": 0, "episodes": 0})
        self.assertEqual(task.status, "processing")
        self.assertEqual(drama.status, "processing")
        self.assertEqual(drama.episodes[0].status, "pending")
