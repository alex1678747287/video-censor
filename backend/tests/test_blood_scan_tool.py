import os
import tempfile
import unittest

import cv2
import numpy as np

from tools import scan_blood_candidates


class BloodScanToolTests(unittest.TestCase):
    def test_score_frame_detects_dark_red_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "blood.jpg")
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            image[20:80, 20:80] = [0, 0, 170]
            cv2.imwrite(image_path, image)

            metrics = scan_blood_candidates._score_frame(image_path)

        self.assertGreater(metrics["score"], 0.1)
        self.assertGreater(metrics["dark_red_ratio"], 0.02)

    def test_score_frame_rejects_warm_non_blood_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "warm.jpg")
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            image[20:80, 20:80] = [40, 90, 130]
            cv2.imwrite(image_path, image)

            metrics = scan_blood_candidates._score_frame(image_path)

        self.assertLess(metrics["score"], 0.05)
        self.assertLess(metrics["dark_red_ratio"], 0.02)

    def test_score_frame_rejects_skin_like_closeup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "skin.jpg")
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            image[:, :] = [95, 140, 185]
            cv2.imwrite(image_path, image)

            metrics = scan_blood_candidates._score_frame(image_path)

        self.assertLess(metrics["score"], 0.05)
        self.assertLess(metrics["dark_red_ratio"], 0.02)


if __name__ == "__main__":
    unittest.main()
