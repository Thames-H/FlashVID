from pathlib import Path
from unittest import TestCase

import torch


class TestSinkAnalysisCore(TestCase):
    def test_identify_sink_tokens_marks_high_attention_low_deviation_tokens(self):
        from sink_analysis.collect.sink_metrics import identify_sink_tokens

        alpha = torch.tensor(
            [
                [0.92, 0.21, 0.08],
                [0.88, 0.19, 0.09],
            ],
            dtype=torch.float32,
        )
        values = torch.tensor(
            [
                [1.0, 1.0],
                [4.0, 4.0],
                [6.0, 6.0],
            ],
            dtype=torch.float32,
        )
        query_outputs = torch.tensor(
            [
                [1.1, 1.0],
                [1.0, 1.1],
            ],
            dtype=torch.float32,
        )

        sink_mask, mean_attn, value_dev = identify_sink_tokens(
            alpha=alpha,
            values=values,
            query_outputs=query_outputs,
            attn_percentile=60,
            dev_percentile=40,
        )

        self.assertEqual(mean_attn.shape[0], 3)
        self.assertEqual(value_dev.shape[0], 3)
        self.assertListEqual(sink_mask.tolist(), [True, False, False])

    def test_paths_use_expected_output_roots(self):
        from sink_analysis.paths import SinkAnalysisPaths

        paths = SinkAnalysisPaths.from_repo_root(Path.cwd())

        self.assertTrue(str(paths.partial_root).endswith("sink_analysis\\artifacts_partial"))
        self.assertTrue(str(paths.artifact_root).endswith("sink_analysis\\artifacts"))
        self.assertTrue(str(paths.figure_root).endswith("sink_analysis\\figures"))
        self.assertTrue(str(paths.data_root).endswith("sink_analysis\\data"))

    def test_build_partial_record_payload_keeps_selection_tensors_and_metadata(self):
        from sink_analysis.collect.writer import build_partial_record_payload
        from sink_analysis.schema import SinkAnalysisExportConfig

        payload = build_partial_record_payload(
            export_config=SinkAnalysisExportConfig(
                output_root=Path("sink_analysis/artifacts_partial"),
                model_name="qwen3-vl",
                method_name="fetp",
                keep_ratio_label="50%",
            ),
            task_name="gqa",
            doc_id=7,
            benchmark="gqa",
            target="book",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is on the table?"},
                    ],
                }
            ],
            answer="book",
            export_payload={
                "target_layer": 12,
                "num_visual_tokens": 4,
                "indices": torch.tensor([1, 3]),
                "scores": torch.tensor([0.1, 0.8, 0.2, 0.5]),
                "alpha": torch.tensor([[0.7, 0.2, 0.05, 0.05]]),
                "values": torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
                "query_outputs": torch.tensor([[1.5]]),
            },
            patch_mapping={
                "token_coords": [(0, 0), (0, 1), (1, 0), (1, 1)],
                "token_source": ["spatial"] * 4,
                "patch_pixel_size": (16, 16),
            },
        )

        self.assertEqual(payload["sample_id"], "gqa__7")
        self.assertEqual(payload["question"], "What is on the table?")
        self.assertEqual(payload["selection"]["indices"], [1, 3])
        self.assertEqual(payload["target_layer"], 12)
        self.assertEqual(payload["num_visual_tokens"], 4)
        self.assertIn("patch_mapping", payload)
