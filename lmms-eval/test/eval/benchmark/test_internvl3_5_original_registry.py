import unittest

from lmms_eval import models


class InternVL3_5OriginalRegistryTest(unittest.TestCase):
    def test_internvl3_5_original_uses_simple_official_path(self):
        self.assertIn("internvl3_5_original", models.AVAILABLE_SIMPLE_MODELS)
        self.assertNotIn(
            "internvl3_5_original",
            models.AVAILABLE_CHAT_TEMPLATE_MODELS,
        )

        model_cls = models.get_model("internvl3_5_original")

        self.assertTrue(model_cls.is_simple)
        self.assertEqual(model_cls.__name__, "InternVL3_5_Original")


if __name__ == "__main__":
    unittest.main()
