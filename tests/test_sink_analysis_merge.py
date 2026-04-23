from unittest import TestCase

import torch


class TestSinkAnalysisMerge(TestCase):
    def test_merge_partial_records_combines_methods_into_one_artifact(self):
        from sink_analysis.collect.merge_runs import merge_partial_records

        partials = [
            {
                "sample_id": "gqa__7",
                "model": "qwen3-vl",
                "benchmark": "gqa",
                "question": "What is on the table?",
                "ground_truth": "book",
                "method": "fetp",
                "keep_ratio": "50%",
                "selection": {"indices": [1, 3], "scores": [0.8, 0.5]},
                "answer": "book",
            },
            {
                "sample_id": "gqa__7",
                "model": "qwen3-vl",
                "benchmark": "gqa",
                "question": "What is on the table?",
                "ground_truth": "book",
                "method": "attention",
                "keep_ratio": "50%",
                "selection": {"indices": [0, 1], "scores": [0.9, 0.7]},
                "answer": "magazine",
            },
            {
                "sample_id": "gqa__7",
                "model": "qwen3-vl",
                "benchmark": "gqa",
                "question": "What is on the table?",
                "ground_truth": "book",
                "method": "mmtok",
                "keep_ratio": "50%",
                "selection": {"indices": [2, 3], "scores": [1.0, 0.6]},
                "answer": "book",
            },
        ]

        artifact = merge_partial_records(partials)

        self.assertEqual(artifact["sample_id"], "gqa__7")
        self.assertEqual(artifact["answers"]["50%"]["fetp"], "book")
        self.assertEqual(
            artifact["selections"]["50%"]["attention"]["indices"].tolist(),
            [0, 1],
        )

    def test_build_ablation_selections_produces_all_four_configs(self):
        from sink_analysis.analyze.exp5_ablation import build_ablation_selections

        artifact = {
            "alpha": torch.tensor([[0.95, 0.40, 0.10, 0.85]], dtype=torch.float32),
            "values": torch.tensor([[1.0], [5.0], [7.0], [1.1]], dtype=torch.float32),
            "query_outputs": torch.tensor([[1.0]], dtype=torch.float32),
            "selections": {
                "50%": {
                    "fetp": {
                        "indices": torch.tensor([0, 2]),
                        "scores": torch.tensor([0.9, 0.1, 0.8, 0.2]),
                    },
                    "attention": {
                        "indices": torch.tensor([0, 3]),
                        "scores": torch.tensor([0.95, 0.3, 0.2, 0.92]),
                    },
                }
            },
        }

        configs = build_ablation_selections(artifact, "50%")

        self.assertIn("A: Attention", configs)
        self.assertIn("B: Attention-Sink", configs)
        self.assertIn("C: FETP", configs)
        self.assertIn("D: FETP+Sink", configs)

