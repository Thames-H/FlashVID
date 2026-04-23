import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import torch


def _load_llava_mmtok_module():
    repo_root = Path(__file__).resolve().parents[4]
    flashvid_root = repo_root / "flashvid"
    mmtok_root = flashvid_root / "mmtok"
    llava_root = mmtok_root / "llava_onevision"

    package_roots = {
        "flashvid": flashvid_root,
        "flashvid.mmtok": mmtok_root,
        "flashvid.mmtok.llava_onevision": llava_root,
    }
    for package_name, package_root in package_roots.items():
        if package_name in sys.modules:
            continue
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_root)]
        sys.modules[package_name] = package

    if "loguru" not in sys.modules:
        fake_loguru = types.ModuleType("loguru")
        fake_loguru.logger = SimpleNamespace(
            info=lambda *args, **kwargs: None,
            warning=lambda *args, **kwargs: None,
            error=lambda *args, **kwargs: None,
        )
        sys.modules["loguru"] = fake_loguru

    module_name = "flashvid.mmtok.llava_onevision.llava_onevision_mmtok"
    if module_name in sys.modules:
        return sys.modules[module_name]

    module_path = llava_root / "llava_onevision_mmtok.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


llava_mmtok = _load_llava_mmtok_module()


class _DummyLanguageModel:
    def __call__(
        self,
        *,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        **kwargs,
    ):
        return SimpleNamespace(
            last_hidden_state=inputs_embeds,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )


class _DummyLlavaModel:
    def __init__(self):
        self.language_model = _DummyLanguageModel()

    def get_placeholder_mask(
        self,
        input_ids,
        *,
        inputs_embeds=None,
        image_features=None,
        video_features=None,
    ):
        image_mask = torch.tensor(
            [[[False, False], [True, True], [True, True]]],
            dtype=torch.bool,
        )
        return image_mask, None


class TestLlavaOnevisionMMTokCompat(TestCase):
    def test_concat_token_features_accepts_feature_lists(self):
        combined = llava_mmtok._concat_token_features(
            [
                torch.tensor([[1.0, 2.0]]),
                torch.tensor([[3.0, 4.0]]),
            ]
        )

        self.assertTrue(
            torch.equal(
                combined,
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            )
        )

    @patch.object(llava_mmtok, "_extract_image_features_compat")
    def test_forward_without_mmtok_accepts_list_image_features(
        self,
        mock_extract_image_features,
    ):
        mock_extract_image_features.return_value = [
            torch.tensor([[1.0, 2.0]]),
            torch.tensor([[3.0, 4.0]]),
        ]
        model = _DummyLlavaModel()

        outputs = llava_mmtok._forward_without_mmtok(
            model,
            input_ids=torch.tensor([[11, 12, 13]]),
            pixel_values=torch.ones(1, 1, 1, 1),
            image_sizes=[(1, 1)],
            pixel_values_videos=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=torch.zeros(1, 3, 2),
            vision_feature_layer=0,
            vision_feature_select_strategy=None,
            vision_aspect_ratio=None,
            batch_num_images=None,
            use_cache=False,
        )

        expected_hidden_states = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        expected_last_hidden_state = torch.tensor(
            [[[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]]]
        )

        self.assertTrue(torch.equal(outputs.image_hidden_states, expected_hidden_states))
        self.assertTrue(
            torch.equal(outputs.last_hidden_state, expected_last_hidden_state)
        )
