from datetime import datetime, timedelta, timezone
from unittest import TestCase

from app.queue_metrics import build_queue_metrics_for_jobs


class QueueMetricsTests(TestCase):
    def test_build_queue_metrics_respects_slots_and_queue_order(self):
        now = datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc)
        jobs = [
            {
                "key": "task:done-sample",
                "id": "done-sample",
                "kind": "task",
                "status": "done",
                "created_at": now - timedelta(hours=2),
                "started_at": now - timedelta(hours=2),
                "completed_at": now - timedelta(hours=2) + timedelta(minutes=5),
                "input_seconds": 100.0,
                "episode_count": 1,
            },
            {
                "key": "drama:done-drama",
                "id": "done-drama",
                "kind": "drama",
                "status": "done",
                "created_at": now - timedelta(hours=3),
                "started_at": now - timedelta(hours=3),
                "completed_at": now - timedelta(hours=3) + timedelta(minutes=15),
                "input_seconds": 600.0,
                "episode_count": 10,
            },
            {
                "key": "task:processing",
                "id": "processing",
                "kind": "task",
                "status": "processing",
                "created_at": now - timedelta(minutes=5),
                "started_at": now - timedelta(minutes=2),
                "completed_at": None,
                "input_seconds": 100.0,
                "episode_count": 1,
            },
            {
                "key": "task:pending-a",
                "id": "pending-a",
                "kind": "task",
                "status": "pending",
                "created_at": now - timedelta(minutes=1),
                "started_at": None,
                "completed_at": None,
                "input_seconds": 100.0,
                "episode_count": 1,
            },
            {
                "key": "drama:pending-b",
                "id": "pending-b",
                "kind": "drama",
                "status": "pending",
                "created_at": now - timedelta(seconds=30),
                "started_at": None,
                "completed_at": None,
                "input_seconds": 600.0,
                "episode_count": 12,
            },
        ]

        metrics = build_queue_metrics_for_jobs(jobs, now=now, slots=2)

        processing = metrics["task:processing"]
        self.assertEqual(processing["queue_position"], 0)
        self.assertGreater(processing["estimated_remaining_seconds"], 0)
        self.assertEqual(processing["estimated_duration_seconds"], 300)

        pending_a = metrics["task:pending-a"]
        self.assertEqual(pending_a["queue_position"], 1)
        self.assertEqual(pending_a["waiting_count"], 0)
        self.assertEqual(pending_a["estimated_wait_seconds"], 0)
        self.assertEqual(pending_a["estimated_duration_seconds"], 300)

        pending_b = metrics["drama:pending-b"]
        self.assertEqual(pending_b["queue_position"], 2)
        self.assertEqual(pending_b["waiting_count"], 1)
        self.assertEqual(pending_b["estimated_wait_seconds"], 180)
        self.assertEqual(pending_b["estimated_duration_seconds"], 900)
