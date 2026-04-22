import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase

import torch
from PIL import Image


def _load_visual_compare_utils_module():
    repo_root = Path(__file__).resolve().parents[4]
    module_path = (
        repo_root
        / "lmms-eval"
        / "lmms_eval"
        / "models"
        / "chat"
        / "llava_onevision_visual_compare_utils.py"
    )
    spec = importlib.util.spec_from_file_location(
        "llava_onevision_visual_compare_utils_test",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_visual_compare_utils = _load_visual_compare_utils_module()


def _make_config():
    return SimpleNamespace(
        image_grid_pinpoints=[(8, 8)],
        vision_config=SimpleNamespace(image_size=4, patch_size=2),
    )


class TestLlavaOnevisionVisualCompareMetadata(TestCase):
    def test_build_visual_compare_metadata_extracts_onevision_token_mapping(self):
        image = Image.new("RGB", (8, 8), color=(10, 20, 30))

        metadata = _visual_compare_utils.build_visual_compare_metadata(
            image_inputs=[image],
            video_inputs=None,
            model_config=_make_config(),
            n_visual_tokens_scored=24,
            vision_aspect_ratio="anyres_max_9",
        )

        self.assertTrue(metadata["visual_compare_eligible"])
        self.assertEqual(tuple(metadata["image_preview"].shape), (8, 8, 3))
        self.assertEqual(metadata["image_size"], [8, 8])
        self.assertEqual(tuple(metadata["token_boxes"].shape), (24, 4))
        self.assertEqual(tuple(metadata["token_is_spatial"].shape), (24,))
        self.assertEqual(int(metadata["token_is_spatial"].sum().item()), 20)
        self.assertEqual(metadata["token_source"].count("base"), 4)
        self.assertEqual(metadata["token_source"].count("crop"), 16)
        self.assertEqual(metadata["token_source"].count("newline"), 4)

    def test_build_visual_compare_metadata_applies_stage1_keep_subset(self):
        image = Image.new("RGB", (8, 8), color=(10, 20, 30))

        metadata = _visual_compare_utils.build_visual_compare_metadata(
            image_inputs=[image],
            video_inputs=None,
            model_config=_make_config(),
            n_visual_tokens_scored=3,
            vision_aspect_ratio="anyres_max_9",
            stage1_keep_local=torch.tensor([0, 4, 5], dtype=torch.long),
        )

        self.assertTrue(metadata["visual_compare_eligible"])
        self.assertEqual(tuple(metadata["token_boxes"].shape), (3, 4))
        self.assertEqual(metadata["token_source"], ["base", "crop", "crop"])

    def test_build_visual_compare_metadata_marks_video_samples_ineligible(self):
        metadata = _visual_compare_utils.build_visual_compare_metadata(
            image_inputs=None,
            video_inputs=[object()],
            model_config=_make_config(),
            n_visual_tokens_scored=0,
            vision_aspect_ratio="anyres_max_9",
        )

        self.assertFalse(metadata["visual_compare_eligible"])
        self.assertEqual(metadata["visual_compare_skip_reason"], "video_input")

    def test_build_visual_compare_metadata_marks_length_mismatch_ineligible(self):
        image = Image.new("RGB", (8, 8), color=(10, 20, 30))

        metadata = _visual_compare_utils.build_visual_compare_metadata(
            image_inputs=[image],
            video_inputs=None,
            model_config=_make_config(),
            n_visual_tokens_scored=25,
            vision_aspect_ratio="anyres_max_9",
        )

        self.assertFalse(metadata["visual_compare_eligible"])
        self.assertEqual(metadata["visual_compare_skip_reason"], "token_mapping_mismatch")
