from unittest import TestCase
from unittest.mock import MagicMock, patch

from lmms_eval.models.simple import internvl3


class TestInternVL3LoaderCompat(TestCase):
    def test_loader_falls_back_to_attn_implementation_when_legacy_flag_is_rejected(self):
        sentinel_model = MagicMock()

        def fake_from_pretrained(_pretrained, **kwargs):
            if "use_flash_attn" in kwargs:
                raise TypeError("InternVLModel.__init__() got an unexpected keyword argument 'use_flash_attn'")
            self.assertEqual(kwargs["attn_implementation"], "flash_attention_2")
            self.assertEqual(kwargs["device_map"], "auto")
            self.assertTrue(kwargs["trust_remote_code"])
            return sentinel_model

        with patch.object(internvl3.AutoModel, "from_pretrained", side_effect=fake_from_pretrained) as mocked_loader:
            loaded_model = internvl3._load_internvl_model(
                "OpenGVLab/InternVL3_5-8B",
                "auto",
                True,
            )

        self.assertIs(loaded_model, sentinel_model)
        self.assertEqual(mocked_loader.call_count, 2)
