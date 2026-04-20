import unittest
from unittest.mock import patch

from app import cloud_cost


class CloudCostTests(unittest.TestCase):
    def test_estimate_vlm_cost_reports_cost_and_reduction(self):
        with patch.object(cloud_cost.config, "CLOUD_EXECUTION_PROFILE", "local_first"), \
                patch.object(cloud_cost.config, "LOCAL_GPU_PROFILE", "rtx4060_8g"), \
                patch.object(cloud_cost.config, "VLM_SAMPLE_INTERVAL", 2.0), \
                patch.object(cloud_cost.config, "ARK_PRICE_INPUT_PER_MTOKEN", 0.6), \
                patch.object(cloud_cost.config, "ARK_PRICE_OUTPUT_PER_MTOKEN", 3.6):
            usage = cloud_cost.estimate_vlm_cost(
                video_duration_seconds=120.0,
                total_extracted_frames=180,
                vlm_frame_calls=18,
                vlm_confirm_calls=3,
                text_batch_calls=1,
                text_batch_chars=120,
                highlight_images=30,
                highlight_batches=2,
            )

        self.assertEqual(usage["cloud_profile"], "local_first")
        self.assertEqual(usage["local_gpu_profile"], "rtx4060_8g")
        self.assertGreater(usage["estimated_input_tokens"], 0)
        self.assertGreater(usage["estimated_output_tokens"], 0)
        self.assertGreater(usage["estimated_cost_cny"], 0.0)
        self.assertEqual(usage["frame_audit_baseline_calls"], 60)
        self.assertGreater(usage["frame_call_reduction_ratio"], 0.6)


if __name__ == "__main__":
    unittest.main()
