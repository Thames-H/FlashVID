import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase

import torch


def _load_mmtok_core_module():
    repo_root = Path(__file__).resolve().parents[4]
    flashvid_root = repo_root / "flashvid"
    mmtok_root = flashvid_root / "mmtok"
    core_root = mmtok_root / "core"

    package_roots = {
        "flashvid": flashvid_root,
        "flashvid.mmtok": mmtok_root,
        "flashvid.mmtok.core": core_root,
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

    semantic_module_name = "flashvid.mmtok.core.semantic_selector"
    if semantic_module_name not in sys.modules:
        semantic_module = types.ModuleType(semantic_module_name)
        semantic_module.SemanticTokenSelector = object
        sys.modules[semantic_module_name] = semantic_module

    text_module_name = "flashvid.mmtok.core.text_processor"
    if text_module_name not in sys.modules:
        text_module = types.ModuleType(text_module_name)
        text_module.VQATextProcessor = object
        sys.modules[text_module_name] = text_module

    module_name = "flashvid.mmtok.core.mmtok_core"
    if module_name in sys.modules:
        return sys.modules[module_name]

    module_path = core_root / "mmtok_core.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


mmtok_core_module = _load_mmtok_core_module()
MMTokCore = mmtok_core_module.MMTokCore


class _DummySelector:
    def __init__(self, target_vision_tokens=None):
        self.target_vision_tokens = target_vision_tokens

    def mm_coverage_selection(
        self,
        text_token_embedding,
        vision_tokens,
        vision_tokens_clip,
        tv_temp=0.01,
        vv_temp=0.2,
        padding_patch_indices=None,
    ):
        k_max = min(int(self.target_vision_tokens), vision_tokens.shape[0])
        selected_indices = list(range(k_max))
        return selected_indices, vision_tokens[selected_indices]


class TestMMTokCoreTargetTokens(TestCase):
    def _build_core(self, target_vision_tokens=None):
        core = object.__new__(MMTokCore)
        core.target_vision_tokens = target_vision_tokens
        core.token_selector = _DummySelector(target_vision_tokens)
        core.softmax_tv_temperature = 0.02
        core.softmax_vv_temperature = 0.2
        core.clean_text = False
        core._encode_text_with_token_pooling = lambda text: torch.ones(1, 2)
        return core

    def test_apply_selection_preprocess_qwen_uses_explicit_target_when_core_default_is_none(
        self,
    ):
        core = self._build_core(target_vision_tokens=None)
        image_embeds = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=torch.float32,
        )

        selected_indices, selected_features = core.apply_selection_preprocess_qwen(
            image_embeds=image_embeds,
            image_features=image_embeds,
            question_text="What is shown?",
            target_vision_tokens=2,
        )

        self.assertEqual(selected_indices, [0, 1])
        self.assertTrue(torch.equal(selected_features, image_embeds[:2]))

    def test_select_vision_tokens_returns_full_features_when_no_target_is_available(self):
        core = self._build_core(target_vision_tokens=None)
        image_embeds = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=torch.float32,
        )
        selected_features, selected_indices = core.select_vision_tokens(
            vision_features=image_embeds,
            vision_features_clip=image_embeds,
            text_token_embedding=torch.ones(1, 2),
        )

        self.assertTrue(torch.equal(selected_features, image_embeds.unsqueeze(0)))
        self.assertEqual(selected_indices, [list(range(image_embeds.shape[0]))])
