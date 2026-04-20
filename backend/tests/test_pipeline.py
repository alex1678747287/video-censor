import unittest
from unittest.mock import patch
import asyncio
import os
import tempfile

import cv2
import numpy as np

from app.core import pipeline


class PipelineHighlightTests(unittest.TestCase):
    def test_detect_highlights_processes_all_batches(self):
        frames = [(float(i * 2), f"/tmp/frame_{i:03d}.jpg") for i in range(45)]
        calls = []

        def fake_detect(sampled_batch, duration):
            calls.append([ts for ts, _ in sampled_batch])
            start = sampled_batch[0][0]
            return [{
                "start_time": f"{start:.1f}s",
                "end_time": f"{start + 2:.1f}s",
                "description": f"batch-{len(calls)}",
            }]

        with patch.object(pipeline.config, "CLOUD_EXECUTION_PROFILE", "local_first"), \
                patch.object(pipeline.config, "HIGHLIGHT_SAMPLE_INTERVAL_SECONDS", 2.0), \
                patch.object(pipeline, "_detect_highlight_batch", side_effect=fake_detect):
            highlights = pipeline.detect_highlights("demo.mp4", frames, duration=120.0)

        self.assertEqual([len(batch) for batch in calls], [20, 20, 5])
        self.assertEqual(
            [item["description"] for item in highlights],
            ["batch-1", "batch-2", "batch-3"],
        )

    def test_build_vlm_sample_frames_local_first_focuses_on_risk_windows(self):
        frames = [(float(i), f"/tmp/frame_{i:03d}.jpg") for i in range(13)]
        all_nudenet_dets = {
            6.0: [{"class": "FEMALE_BREAST_COVERED", "score": 0.74}],
        }

        with patch.object(pipeline.config, "CLOUD_EXECUTION_PROFILE", "local_first"), \
                patch.object(pipeline.config, "VLM_LOCAL_FIRST_BASELINE_INTERVAL_SECONDS", 5.0), \
                patch.object(pipeline.config, "VLM_LOCAL_FIRST_RISK_WINDOW_SECONDS", 1.0), \
                patch.object(pipeline, "_collect_blood_candidate_timestamps", return_value=set()):
            sampled = pipeline._build_vlm_sample_frames(frames, all_nudenet_dets)

        sampled_ts = [ts for ts, _ in sampled]
        self.assertEqual(sampled_ts[0], 0.0)
        self.assertIn(6.0, sampled_ts)
        self.assertIn(7.0, sampled_ts)
        self.assertLess(len(sampled_ts), len(frames))

    def test_parse_highlight_response_salvages_malformed_json(self):
        raw = """
        ```json
        {
          "highlights": [
            {"start_time": "42.0s", "end_time": "45.0s", "description": "反转开场", "reason": "冲突升级"},
            {"start_time": "88.0s", "end_time": "91.0s", "description": "台词出现 "爆点"", "reason": "情绪拉满"}
          ]
        }
        ```
        """

        highlights = pipeline._parse_highlight_response(
            raw,
            ts_list=[42.0, 88.0],
            duration=120.0,
        )

        self.assertEqual(len(highlights), 2)
        self.assertEqual(highlights[0]["start_time"], "42.0s")
        self.assertEqual(highlights[0]["end_time"], "45.0s")
        self.assertEqual(highlights[1]["start_time"], "88.0s")
        self.assertEqual(highlights[1]["end_time"], "91.0s")

    def test_detect_vlm_frames_processes_all_batches(self):
        sampled = [(float(i * 2), f"/tmp/frame_{i:03d}.jpg") for i in range(85)]
        seen = []

        async def fake_detect(path, frame_width=1920, frame_height=1080):
            seen.append(path)
            return {"violations": [], "safe": True}

        with patch.object(pipeline.vlm, "detect_frame_vlm_async", side_effect=fake_detect):
            results = asyncio.run(
                pipeline._detect_vlm_frames_async(sampled, frame_width=720, frame_height=1280)
            )

        self.assertEqual(len(results), 85)
        self.assertEqual(len(seen), 85)
        self.assertEqual(seen[0], "/tmp/frame_000.jpg")
        self.assertEqual(seen[-1], "/tmp/frame_084.jpg")

    def test_resolve_vlm_visual_box_prefers_existing_box(self):
        box = pipeline._resolve_vlm_visual_box(
            {"box": [10, 20, 30, 40], "region": "full"},
            frame_width=720,
            frame_height=1280,
        )

        self.assertEqual(box, [10, 20, 30, 40])

    def test_confidence_helpers_normalize_and_combine(self):
        self.assertEqual(pipeline._normalize_confidence(82), 0.82)
        self.assertEqual(pipeline._confidence_from_severity("high"), 0.92)
        self.assertEqual(pipeline._combine_confidence(0.9, 80), 0.85)

    def test_find_temporal_support_score_matches_nearby_same_class(self):
        support = pipeline._find_temporal_support_score(
            10.0,
            {"class": "FEMALE_BREAST_COVERED", "score": 0.62, "box": [100, 100, 180, 220]},
            {
                8.9: [{"class": "FEMALE_BREAST_COVERED", "score": 0.71, "box": [102, 104, 182, 224]}],
                12.0: [{"class": "FACE_FEMALE", "score": 0.99, "box": [10, 10, 80, 80]}],
            },
            window_seconds=1.5,
        )

        self.assertEqual(support, 0.71)

    def test_find_temporal_support_score_ignores_distant_or_mismatched_boxes(self):
        support = pipeline._find_temporal_support_score(
            10.0,
            {"class": "MALE_GENITALIA_EXPOSED", "score": 0.66, "box": [100, 100, 150, 180]},
            {
                11.6: [{"class": "MALE_GENITALIA_EXPOSED", "score": 0.8, "box": [400, 400, 450, 480]}],
                13.0: [{"class": "MALE_GENITALIA_EXPOSED", "score": 0.9, "box": [102, 100, 152, 182]}],
            },
            window_seconds=1.2,
        )

        self.assertIsNone(support)

    def test_should_auto_mosaic_vlm_visual_requires_support_for_medium_blood(self):
        self.assertFalse(
            pipeline._should_auto_mosaic_vlm_visual(
                "血腥暴力", "medium", support_score=None, bbox=[10, 10, 8, 8]
            )
        )
        self.assertTrue(
            pipeline._should_auto_mosaic_vlm_visual(
                "血腥暴力", "medium", support_score=0.78, bbox=[10, 10, 8, 8]
            )
        )

    def test_should_auto_mosaic_vlm_visual_allows_small_high_severity_box(self):
        self.assertTrue(
            pipeline._should_auto_mosaic_vlm_visual(
                "低俗色情", "high", support_score=None, bbox=[20, 20, 10, 10]
            )
        )
        self.assertFalse(
            pipeline._should_auto_mosaic_vlm_visual(
                "低俗色情", "high", support_score=None, bbox=[20, 20, 30, 30]
            )
        )

    def test_has_blood_description_cue_requires_explicit_gore_wording(self):
        self.assertTrue(pipeline._has_blood_description_cue("近景特写出现伤口流血"))
        self.assertFalse(pipeline._has_blood_description_cue("两人激烈打斗，动作冲突明显"))

    def test_should_keep_vlm_frame_text_violation_rejects_benign_dialogue(self):
        keep, nearby_text = pipeline._should_keep_vlm_frame_text_violation(
            10.0,
            "medium",
            {10.0: ["爸破脂羊是什么呀"]},
        )

        self.assertFalse(keep)
        self.assertIn("破脂羊", nearby_text)

    def test_should_keep_vlm_frame_text_violation_keeps_explicit_risk_text(self):
        keep, nearby_text = pipeline._should_keep_vlm_frame_text_violation(
            10.0,
            "high",
            {10.0: ["我现在就杀了你"]},
        )

        self.assertTrue(keep)
        self.assertIn("杀了你", nearby_text)

    def test_analyze_blood_visual_evidence_detects_dark_red_patch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "blood.jpg")
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            image[20:80, 20:80] = [0, 0, 170]
            cv2.imwrite(image_path, image)

            metrics = pipeline._analyze_blood_visual_evidence(image_path, [20, 20, 80, 80])

        self.assertTrue(metrics["has_evidence"])
        self.assertTrue(metrics["strong_evidence"])
        self.assertGreater(metrics["red_ratio"], 0.1)
        self.assertGreater(metrics["dark_red_ratio"], 0.05)

    def test_analyze_blood_visual_evidence_rejects_warm_non_blood_patch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "warm.jpg")
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            image[20:80, 20:80] = [40, 90, 130]
            cv2.imwrite(image_path, image)

            metrics = pipeline._analyze_blood_visual_evidence(image_path, [20, 20, 80, 80])

        self.assertFalse(metrics["has_evidence"])
        self.assertFalse(metrics["strong_evidence"])
        self.assertLess(metrics["red_ratio"], 0.05)

    def test_analyze_blood_visual_evidence_rejects_skin_like_closeup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "skin.jpg")
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            image[:, :] = [95, 140, 185]
            cv2.imwrite(image_path, image)

            metrics = pipeline._analyze_blood_visual_evidence(image_path, [0, 0, 100, 100])

        self.assertFalse(metrics["has_evidence"])
        self.assertFalse(metrics["strong_evidence"])
        self.assertLess(metrics["red_ratio"], 0.05)

    def test_analyze_blood_visual_evidence_detects_small_drop_on_skin(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "drop.jpg")
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            image[:, :] = [95, 140, 185]
            cv2.circle(image, (54, 34), 8, (0, 0, 165), -1)
            cv2.rectangle(image, (52, 40), (56, 74), (0, 0, 150), -1)
            cv2.imwrite(image_path, image)

            metrics = pipeline._analyze_blood_visual_evidence(image_path, [20, 10, 88, 86])

        self.assertTrue(metrics["has_spot_evidence"])
        self.assertGreater(metrics["spot_ratio"], 0.01)

    def test_should_auto_mosaic_blood_violation_requires_cue_or_image_evidence(self):
        with patch.object(
            pipeline,
            "_analyze_blood_visual_evidence",
            return_value={
                "has_evidence": False,
                "strong_evidence": False,
                "red_ratio": 0.0,
                "dark_red_ratio": 0.0,
            },
        ):
            allowed, metrics = pipeline._should_auto_mosaic_blood_violation(
                "激烈冲突动作",
                "high",
                support_score=0.81,
                bbox=[10, 10, 8, 8],
                image_path="/tmp/demo.jpg",
                visual_box=[100, 100, 200, 200],
            )

        self.assertFalse(allowed)
        self.assertFalse(metrics["has_description_cue"])

        with patch.object(
            pipeline,
            "_analyze_blood_visual_evidence",
            return_value={
                "has_evidence": True,
                "strong_evidence": True,
                "red_ratio": 0.24,
                "dark_red_ratio": 0.18,
            },
        ):
            allowed, metrics = pipeline._should_auto_mosaic_blood_violation(
                "画面出现伤口流血",
                "high",
                support_score=None,
                bbox=[10, 10, 8, 8],
                image_path="/tmp/demo.jpg",
                visual_box=[100, 100, 200, 200],
            )

        self.assertTrue(allowed)
        self.assertTrue(metrics["has_description_cue"])

    def test_should_auto_mosaic_blood_violation_allows_medium_direct_drop_with_cue(self):
        with patch.object(
            pipeline,
            "_analyze_blood_visual_evidence",
            return_value={
                "has_evidence": False,
                "strong_evidence": False,
                "has_spot_evidence": True,
                "red_ratio": 0.018,
                "dark_red_ratio": 0.011,
                "spot_ratio": 0.018,
                "spot_pixels": 42,
            },
        ):
            allowed, metrics = pipeline._should_auto_mosaic_blood_violation(
                "手指伤口滴血特写",
                "medium",
                support_score=None,
                bbox=[10, 10, 8, 8],
                image_path="/tmp/demo.jpg",
                visual_box=[100, 100, 180, 220],
            )

        self.assertTrue(allowed)
        self.assertTrue(metrics["has_description_cue"])


if __name__ == "__main__":
    unittest.main()
