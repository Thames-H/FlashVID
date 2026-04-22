import importlib.util
from pathlib import Path
from unittest import TestCase

import torch
from PIL import Image


def _load_visual_compare_utils_module():
    repo_root = Path(__file__).resolve().parents[4]
    module_path = repo_root / "lmms-eval" / "lmms_eval" / "models" / "chat" / "qwen3_vl_visual_compare_utils.py"
    spec = importlib.util.spec_from_file_location(
        "qwen3_vl_visual_compare_utils_test",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_visual_compare_utils = _load_visual_compare_utils_module()


class TestQwen3VLVisualCompareMetadata(TestCase):
    def test_attach_visual_compare_metadata_merges_fields_into_artifact(self):
        artifact = {"method": "fetp", "metadata": {}}
        visual_metadata = {
            "visual_compare_eligible": True,
            "visual_compare_skip_reason": None,
            "image_preview": torch.zeros((4, 6, 3), dtype=torch.uint8),
            "image_size": [4, 6],
            "token_grid_size": [2, 3],
        }

        updated = _visual_compare_utils.attach_visual_compare_metadata(
            artifact,
            visual_metadata,
            target_layer=16,
        )

        self.assertEqual(updated["metadata"]["target_layer"], 16)
        self.assertTrue(updated["visual_compare_eligible"])
        self.assertEqual(updated["image_size"], [4, 6])
        self.assertEqual(updated["token_grid_size"], [2, 3])

    def test_build_visual_compare_metadata_extracts_single_image_preview_and_grid(self):
        image = Image.new("RGB", (6, 4), color=(10, 20, 30))

        metadata = _visual_compare_utils.build_visual_compare_metadata(
            image_inputs=[image],
            video_inputs=None,
            image_grid_thw=torch.tensor([[1, 4, 6]], dtype=torch.long),
            n_visual_tokens_scored=6,
            spatial_merge_size=2,
        )

        self.assertTrue(metadata["visual_compare_eligible"])
        self.assertEqual(tuple(metadata["image_preview"].shape), (4, 6, 3))
        self.assertEqual(metadata["image_size"], [4, 6])
        self.assertEqual(metadata["token_grid_size"], [2, 3])

    def test_build_visual_compare_metadata_marks_video_samples_ineligible(self):
        metadata = _visual_compare_utils.build_visual_compare_metadata(
            image_inputs=None,
            video_inputs=[object()],
            image_grid_thw=None,
            n_visual_tokens_scored=0,
            spatial_merge_size=2,
        )

        self.assertFalse(metadata["visual_compare_eligible"])
        self.assertEqual(metadata["visual_compare_skip_reason"], "video_input")
