import importlib.util
from pathlib import Path
from unittest import TestCase

import torch


def _load_pca_compare_module():
    repo_root = Path(__file__).resolve().parents[4]
    module_path = repo_root / "tools" / "qwen3_vl_token_pruning_pca_compare.py"
    spec = importlib.util.spec_from_file_location(
        "qwen3_vl_pca_compare_plotting_test",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_pca_compare = _load_pca_compare_module()


class TestQwen3VLPCAComparePlotting(TestCase):
    def test_resolve_layer_root_appends_layer_suffix(self):
        resolved = _pca_compare._resolve_layer_root(Path("/tmp/pca_compare"), 20)

        self.assertEqual(resolved, Path("/tmp/pca_compare/layer_20"))

    def test_compute_subset_groups_partitions_overlaps(self):
        groups = _pca_compare._compute_subset_groups(
            torch.tensor([0, 1, 2, 6]),
            torch.tensor([1, 2, 3, 6]),
            torch.tensor([2, 4, 5, 6]),
        )

        self.assertEqual(groups["fetp_only"].tolist(), [0])
        self.assertEqual(groups["attention_only"].tolist(), [3])
        self.assertEqual(groups["mmtok_only"].tolist(), [4, 5])
        self.assertEqual(groups["fetp_attention"].tolist(), [1])
        self.assertEqual(groups["fetp_mmtok"].tolist(), [])
        self.assertEqual(groups["attention_mmtok"].tolist(), [])
        self.assertEqual(groups["all_three"].tolist(), [2, 6])

        covered = set()
        for indices in groups.values():
            current = set(indices.tolist())
            self.assertTrue(covered.isdisjoint(current))
            covered |= current

        self.assertEqual(covered, {0, 1, 2, 3, 4, 5, 6})
