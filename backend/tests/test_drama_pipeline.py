import unittest

from app.core import drama_pipeline


class DramaPipelineTests(unittest.TestCase):
    def test_limit_episode_durations_respects_global_trim(self):
        episodes = [
            {"episode_num": 1, "original_duration": 60.0},
            {"episode_num": 2, "original_duration": 60.0},
            {"episode_num": 3, "original_duration": 60.0},
        ]

        limited = drama_pipeline._limit_episode_durations(
            episodes, speed_factor=1.2, total_duration=80.0
        )

        self.assertEqual(
            [round(ep["processed_duration"], 1) for ep in limited],
            [50.0, 30.0, 0.0],
        )
        self.assertEqual([ep["trimmed_out"] for ep in limited], [False, False, True])

        highlights = drama_pipeline._map_highlights_to_episodes(
            [
                {"start_time": "70.0s", "end_time": "90.0s", "description": "late twist"},
                {"start_time": "95.0s", "end_time": "98.0s", "description": "trimmed away"},
            ],
            limited,
            speed_factor=1.2,
        )

        self.assertEqual(len(highlights[0]["highlights"]), 0)
        self.assertEqual(len(highlights[1]["highlights"]), 1)
        self.assertEqual(highlights[1]["highlights"][0]["start_time"], "20.0s")
        self.assertEqual(highlights[1]["highlights"][0]["end_time"], "30.0s")
        self.assertEqual(len(highlights[2]["highlights"]), 0)

    def test_build_label_filter_uses_top_and_bottom_slots(self):
        vf = drama_pipeline._build_label_filter("title.txt", "disc.txt", "title.ttc", "disc.ttc")

        self.assertIn("fontsize=72", vf)
        self.assertIn("fontsize=42", vf)
        self.assertIn("y=h*0.055", vf)
        self.assertIn("y=h-text_h-h*0.055", vf)
        self.assertIn("box=1", vf)
        self.assertIn("text_shaping=1", vf)

    def test_count_violations_by_episode_respects_processed_boundaries(self):
        episodes = [
            {"episode_num": 1, "original_duration": 60.0},
            {"episode_num": 2, "original_duration": 60.0},
            {"episode_num": 3, "original_duration": 60.0},
        ]
        limited = drama_pipeline._limit_episode_durations(
            episodes, speed_factor=1.2, total_duration=80.0
        )

        counts = drama_pipeline._count_violations_by_episode(
            [
                {"timestamp": 10.0},
                {"timestamp": "49.9s"},
                {"timestamp": 50.0},
                {"timestamp": 79.9},
                {"timestamp": 80.0},
                {"timestamp": 95.0},
            ],
            limited,
            speed_factor=1.2,
        )

        self.assertEqual(counts[1], 2)
        self.assertEqual(counts[2], 2)
        self.assertEqual(counts[3], 0)

    def test_map_violations_to_episodes_converts_to_local_timestamps(self):
        episodes = [
            {"episode_num": 1, "original_duration": 60.0},
            {"episode_num": 2, "original_duration": 60.0},
        ]
        limited = drama_pipeline._limit_episode_durations(
            episodes, speed_factor=1.2, total_duration=100.0
        )

        mapped = drama_pipeline._map_violations_to_episodes(
            [
                {"timestamp": 12.5, "type": "违规文字", "description": "A"},
                {"timestamp": "55.0s", "type": "违规文字", "description": "B"},
            ],
            limited,
            speed_factor=1.2,
        )

        self.assertEqual(len(mapped[1]), 1)
        self.assertEqual(mapped[1][0]["timestamp"], 12.5)
        self.assertEqual(mapped[1][0]["global_timestamp"], 12.5)
        self.assertEqual(len(mapped[2]), 1)
        self.assertEqual(mapped[2][0]["timestamp"], 5.0)
        self.assertEqual(mapped[2][0]["global_timestamp"], 55.0)

    def test_select_hook_window_avoids_tail_transition(self):
        window = drama_pipeline._select_hook_window(
            [
                {"start_time": "26.2s", "end_time": "28.4s", "description": "tail slow motion"},
                {"start_time": "8.0s", "end_time": "10.4s", "description": "fight"},
            ],
            episode_duration=29.0,
            clip_duration=3.0,
            pre_roll_seconds=0.3,
            end_guard_seconds=1.0,
        )

        self.assertIsNotNone(window)
        self.assertEqual(window["highlight"]["description"], "fight")
        self.assertAlmostEqual(window["clip_start"], 7.7, places=1)
        self.assertAlmostEqual(window["clip_duration"], 3.0, places=1)

    def test_select_hook_window_rejects_highlight_that_spills_into_tail_zone(self):
        window = drama_pipeline._select_hook_window(
            [
                {"start_time": "27.0s", "end_time": "29.0s", "description": "tail teaser"},
                {"start_time": "12.0s", "end_time": "13.1s", "description": "fight"},
            ],
            episode_duration=30.0,
            clip_duration=3.0,
            pre_roll_seconds=0.35,
            end_guard_seconds=1.8,
        )

        self.assertIsNotNone(window)
        self.assertEqual(window["highlight"]["description"], "fight")

    def test_select_hook_window_skips_intro_transition_highlight(self):
        window = drama_pipeline._select_hook_window(
            [
                {"start_time": "0.2s", "end_time": "1.6s", "description": "opening transition"},
                {"start_time": "4.5s", "end_time": "6.0s", "description": "real teaser"},
            ],
            episode_duration=30.0,
            clip_duration=3.0,
            pre_roll_seconds=0.35,
            end_guard_seconds=1.0,
        )

        self.assertIsNotNone(window)
        self.assertEqual(window["highlight"]["description"], "real teaser")
        self.assertGreaterEqual(window["clip_start"], 1.0)

    def test_select_hook_window_returns_none_when_only_intro_transition_exists(self):
        window = drama_pipeline._select_hook_window(
            [
                {"start_time": "0.1s", "end_time": "1.3s", "description": "opening transition"},
            ],
            episode_duration=18.0,
            clip_duration=3.0,
            pre_roll_seconds=0.35,
            end_guard_seconds=1.0,
        )

        self.assertIsNone(window)

    def test_select_hook_window_fallback_respects_tail_guard(self):
        window = drama_pipeline._select_hook_window(
            [
                {"start_time": "8.0s", "end_time": "8.2s", "description": "late tease"},
            ],
            episode_duration=10.0,
            clip_duration=3.0,
            pre_roll_seconds=0.35,
            end_guard_seconds=1.8,
        )

        self.assertIsNotNone(window)
        self.assertAlmostEqual(window["clip_start"], 5.2, places=1)
        self.assertAlmostEqual(window["clip_duration"], 3.0, places=1)

    def test_select_fallback_hook_window_skips_intro_and_tail(self):
        window = drama_pipeline._select_fallback_hook_window(
            episode_duration=40.0,
            clip_duration=3.0,
            intro_skip_seconds=1.2,
            end_guard_seconds=1.8,
        )

        self.assertIsNotNone(window)
        self.assertAlmostEqual(window["clip_start"], 1.2, places=1)
        self.assertAlmostEqual(window["clip_duration"], 3.0, places=1)

    def test_map_highlights_to_episodes_skips_invalid_timestamps(self):
        episodes = [
            {"episode_num": 1, "original_duration": 60.0},
            {"episode_num": 2, "original_duration": 60.0},
        ]

        mapped = drama_pipeline._map_highlights_to_episodes(
            [
                {"start_time": "bad", "end_time": "10.0s", "description": "broken"},
                {"start_time": "55.0s", "end_time": "59.0s", "description": "valid"},
            ],
            episodes,
            speed_factor=1.0,
        )

        self.assertEqual(len(mapped[0]["highlights"]), 1)
        self.assertEqual(mapped[0]["highlights"][0]["description"], "valid")
        self.assertEqual(len(mapped[1]["highlights"]), 0)


if __name__ == "__main__":
    unittest.main()
