import unittest

from lmms_eval import models
from lmms_eval.models.chat.internvl_hf import InternVLHf


class InternVL3_5OriginalRegistryTest(unittest.TestCase):
    def test_internvl3_5_original_uses_hf_chat_backend(self):
        self.assertNotIn("internvl3_5_original", models.AVAILABLE_SIMPLE_MODELS)
        self.assertIn(
            "internvl3_5_original",
            models.AVAILABLE_CHAT_TEMPLATE_MODELS,
        )

        model_cls = models.get_model("internvl3_5_original")

        self.assertFalse(model_cls.is_simple)
        self.assertEqual(model_cls.__name__, "InternVL3_5_Original")
        self.assertTrue(issubclass(model_cls, InternVLHf))


if __name__ == "__main__":
    unittest.main()
