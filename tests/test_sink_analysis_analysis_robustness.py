from unittest import TestCase
from unittest.mock import patch

import numpy as np
import torch

from sink_analysis.analyze.exp5_ablation import build_ablation_selections
from sink_analysis.analyze.exp2_sink_retention import compute_sink_retention
from sink_analysis.analyze.exp4_spatial_visuals import render_full_comparison
from sink_analysis.analyze.exp6_summary import (
    generate_summary_table,
    generate_summary_tables_by_ratio,
)


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
    def test_compute_sink_retention_treats_missing_methods_as_nan(self):
        artifact = _build_artifact_without_mmtok()

        results = compute_sink_retention([artifact], "50%")

        self.assertIn("mmtok", results)
        self.assertTrue(np.isnan(results["mmtok"]))

    def test_generate_summary_table_marks_missing_mmtok_as_na(self):
        artifact = _build_artifact_without_mmtok()

        frame = generate_summary_table({"llava-onevision": [artifact]}, keep_ratio="50%")

        self.assertEqual(frame.iloc[0]["IoU (FETP vs MMTok)"], "NA")

    def test_generate_summary_tables_by_ratio_adds_keep_ratio_rows(self):
        artifact = _build_artifact_without_mmtok()
        artifact["selections"]["25%"] = artifact["selections"]["50%"]

        frame = generate_summary_tables_by_ratio({"llava-onevision": [artifact]})

        self.assertEqual(frame["Keep Ratio"].tolist(), ["25%", "50%"])
        self.assertTrue((frame["Model"] == "llava-onevision").all())

    def test_render_full_comparison_handles_missing_mmtok_selection(self):
        artifact = _build_artifact_without_mmtok()

        figure = render_full_comparison(artifact, keep_ratio="50%")

        self.assertEqual(len(figure.axes), 8)

    @patch("sink_analysis.analyze.exp5_ablation.identify_sink_tokens")
    def test_build_ablation_selections_replaces_lowest_fetp_scores_for_sink_injection(self, mock_identify):
        artifact = _build_artifact_without_mmtok()
        artifact["selections"]["50%"]["fetp"] = {
            "indices": torch.tensor([2, 3], dtype=torch.long),
            "scores": torch.tensor([0.9, 0.8, 0.7, 0.1], dtype=torch.float32),
        }
        artifact["selections"]["50%"]["attention"]["scores"] = torch.tensor(
            [1.0, 0.9, 0.4, 0.3],
            dtype=torch.float32,
        )
        mock_identify.return_value = (
            torch.tensor([True, True, False, False]),
            torch.zeros(4, dtype=torch.float32),
            torch.zeros(4, dtype=torch.float32),
        )

        selections = build_ablation_selections(artifact, "50%")

        self.assertTrue(torch.equal(selections["D: FETP+Sink"], torch.tensor([0, 1], dtype=torch.long)))
