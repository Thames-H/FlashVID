import importlib.util
from pathlib import Path
from unittest import TestCase

import torch


def _load_semantic_selector_module():
    repo_root = Path(__file__).resolve().parents[4]
    module_path = repo_root / "flashvid" / "mmtok" / "core" / "semantic_selector.py"
    spec = importlib.util.spec_from_file_location(
        "flashvid_mmtok_semantic_selector_test",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


compute_initial_marginal_gain = _load_semantic_selector_module().compute_initial_marginal_gain


class TestQwen3VLMMTokSelection(TestCase):
    def test_compute_initial_marginal_gain_sums_rows_per_token(self):
        combined = torch.tensor(
            [
                [0.10, 0.20, 0.30],
                [0.50, 0.10, 0.00],
                [0.40, 0.20, 0.10],
            ]
        )

        gains = compute_initial_marginal_gain(combined)

        expected = torch.tensor([1.00, 0.50, 0.40])
        self.assertTrue(torch.allclose(gains, expected, atol=1e-6))
