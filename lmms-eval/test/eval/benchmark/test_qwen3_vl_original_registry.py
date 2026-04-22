import unittest

from lmms_eval import models


class Qwen3OriginalRegistryTest(unittest.TestCase):
    def test_qwen3_original_uses_simple_task_path(self):
        self.assertIn("qwen3_vl_original", models.AVAILABLE_SIMPLE_MODELS)
        self.assertNotIn("qwen3_vl_original", models.AVAILABLE_CHAT_TEMPLATE_MODELS)

        model_cls = models.get_model("qwen3_vl_original")

        self.assertTrue(model_cls.is_simple)
        self.assertEqual(model_cls.__name__, "Qwen3_VL_Original")


if __name__ == "__main__":
    unittest.main()
