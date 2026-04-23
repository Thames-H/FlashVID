from unittest import TestCase

import numpy as np
import torch

from sink_analysis.analyze.exp2_sink_retention import compute_sink_retention
from sink_analysis.analyze.exp4_spatial_visuals import render_full_comparison
from sink_analysis.analyze.exp6_summary import generate_summary_table


def _build_artifact_without_mmtok():
    return {
        "sample_id": "demo",
        "model": "llava-onevision",
        "benchmark": "gqa",
        "image_preview": np.zeros((32, 32, 3), dtype=np.uint8),
        "patch_mapping": {
            "token_coords": [(8, 8), (8, 24), (24, 8), (24, 24)],
            "patch_pixel_size": (16, 16),
        },
        "alpha": torch.tensor([[0.9, 0.1, 0.1, 0.1]], dtype=torch.float32),
        "values": torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [1.0, -1.0],
                [-1.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        "query_outputs": torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        "selections": {
            "50%": {
                "attention": {
                    "indices": torch.tensor([0, 1], dtype=torch.long),
                    "scores": torch.tensor([1.0, 0.5], dtype=torch.float32),
                },
                "fetp": {
                    "indices": torch.tensor([1, 2], dtype=torch.long),
                    "scores": torch.tensor([0.7, 0.6], dtype=torch.float32),
                },
            }
        },
    }


class TestSinkAnalysisAnalysisRobustness(TestCase):
    def test_compute_sink_retention_treats_missing_methods_as_zero(self):
        artifact = _build_artifact_without_mmtok()

        results = compute_sink_retention([artifact], "50%")

        self.assertIn("mmtok", results)
        self.assertEqual(results["mmtok"], 0.0)

    def test_generate_summary_table_treats_missing_mmtok_as_empty_overlap(self):
        artifact = _build_artifact_without_mmtok()

        frame = generate_summary_table({"llava-onevision": [artifact]}, keep_ratio="50%")

        self.assertEqual(frame.iloc[0]["IoU (FETP vs MMTok)"], "0.000")

    def test_render_full_comparison_handles_missing_mmtok_selection(self):
        artifact = _build_artifact_without_mmtok()

        figure = render_full_comparison(artifact, keep_ratio="50%")

        self.assertEqual(len(figure.axes), 8)
