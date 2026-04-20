import unittest

from app.core import vlm


class VlmPromptTests(unittest.TestCase):
    def test_confirm_prompt_for_type_uses_breast_prompt(self):
        prompt = vlm._confirm_prompt_for_type("FEMALE_BREAST_COVERED")
        self.assertIn("胸部区域", prompt)

    def test_confirm_prompt_for_type_uses_lower_body_prompt(self):
        prompt = vlm._confirm_prompt_for_type("MALE_GENITALIA_EXPOSED")
        self.assertIn("下体或臀部区域", prompt)

    def test_confirm_prompt_for_type_falls_back_to_general_prompt(self):
        prompt = vlm._confirm_prompt_for_type("UNKNOWN")
        self.assertIn("人体敏感区域", prompt)


if __name__ == "__main__":
    unittest.main()
