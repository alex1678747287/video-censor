import unittest
from time import sleep

from app import worker


class WorkerTests(unittest.TestCase):
    def test_build_episode_updates_preserves_zero_violations(self):
        updates = worker._build_episode_updates({
            "status": "censored",
            "violations": 0,
            "highlights": [],
        })

        self.assertEqual(updates["status"], "censored")
        self.assertEqual(updates["violations_count"], 0)
        self.assertEqual(updates["highlights"], "[]")

    def test_celery_reliability_settings_enabled(self):
        self.assertTrue(worker.celery_app.conf.task_acks_late)
        self.assertTrue(worker.celery_app.conf.task_reject_on_worker_lost)
        self.assertEqual(
            worker.celery_app.conf.broker_transport_options["visibility_timeout"],
            worker.config.CELERY_VISIBILITY_TIMEOUT_SECONDS,
        )

    def test_progress_heartbeat_touches_latest_stage(self):
        calls = []

        def fake_updater(item_id, **kwargs):
            calls.append((item_id, kwargs))

        heartbeat = worker._ProgressHeartbeat(fake_updater, "job-1", interval_seconds=0.05)
        heartbeat.update(progress=0.45, stage="vlm_detection")
        heartbeat.start()
        sleep(0.12)
        heartbeat.update(progress=0.8, stage="applying_mosaic")
        sleep(0.08)
        heartbeat.stop()

        self.assertGreaterEqual(len(calls), 2)
        self.assertTrue(all(call[0] == "job-1" for call in calls))
        self.assertTrue(all(call[1]["force_touch"] for call in calls))
        self.assertTrue(any(call[1]["stage"] == "vlm_detection" for call in calls))
        self.assertEqual(calls[-1][1]["stage"], "applying_mosaic")


if __name__ == "__main__":
    unittest.main()
