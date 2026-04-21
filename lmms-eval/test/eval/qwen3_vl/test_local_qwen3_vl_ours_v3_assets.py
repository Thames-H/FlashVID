import ast
from pathlib import Path
from unittest import TestCase


class TestLocalQwen3VLOursV3Assets(TestCase):
    def test_local_compare_runner_exists_and_references_v2_and_v3(self):
        repo_root = Path(__file__).resolve().parents[4]
        runner_path = repo_root / "playground" / "qwen3_vl_ours_v3_compare.py"

        self.assertTrue(runner_path.exists(), "local compare runner should exist")

        runner_ast = ast.parse(runner_path.read_text(encoding="utf-8"))
        class_names = {
            node.name for node in runner_ast.body if isinstance(node, ast.ClassDef)
        }
        self.assertIn("ComparisonArguments", class_names)

        string_constants = [
            node.value
            for node in ast.walk(runner_ast)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]
        self.assertIn("qwen3_vl_ours_v2", string_constants)
        self.assertIn("qwen3_vl_ours_v3", string_constants)
        self.assertIn("pruning_scoring_time_ms", string_constants)
        self.assertIn("pruning_reference_speedup", string_constants)
