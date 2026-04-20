import unittest

from app.core import ocr_detector


class OcrDetectorTests(unittest.TestCase):
    def test_extract_vlm_violation_indices_handles_malformed_json(self):
        raw = """
        {"violations": [
          {"index": 10, "text": "你打算做破脂羊吗", "reason": "\\"破脂\\"为"破处"的谐音变体", "severity": "high"},
          {"index": 11, "text": "普通台词", "reason": "无风险", "severity": "low"},
          {"index": 12, "text": "继续暗示", "reason": "低俗暗示", "severity": "medium"}
        ], "has_violation": true}
        """

        indices = ocr_detector._extract_vlm_violation_indices(raw, subtitles_count=20)

        self.assertEqual(indices, [])

    def test_collect_vlm_violation_indices_suppresses_benign_livestock_context(self):
        indices = ocr_detector._collect_vlm_violation_indices(
            [{
                "index": 0,
                "text": "破脂羊只能是母的",
                "reason": "谐音低俗暗示",
                "severity": "high",
            }],
            subtitles_count=1,
        )

        self.assertEqual(indices, [])

    def test_collect_vlm_violation_indices_keeps_explicit_risk_text(self):
        indices = ocr_detector._collect_vlm_violation_indices(
            [{
                "index": 0,
                "text": "今晚陪睡就放过你",
                "reason": "明显性交易暗示",
                "severity": "high",
            }],
            subtitles_count=1,
        )

        self.assertEqual(indices, [0])

    def test_extract_vlm_violation_indices_fallback_applies_text_filter(self):
        raw = """
        {"violations": [
          {"index": 0, "text": "破脂羊只能是母的", "reason": "低俗谐音", "severity": "high"},
          {"index": 1, "text": "今晚陪睡就放过你", "reason": "性交易暗示", "severity": "high"}
        ], "has_violation": true}
        """

        indices = ocr_detector._extract_vlm_violation_indices(raw, subtitles_count=5)

        self.assertEqual(indices, [1])


if __name__ == "__main__":
    unittest.main()
