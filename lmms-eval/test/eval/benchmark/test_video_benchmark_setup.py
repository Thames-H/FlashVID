import ast
from pathlib import Path
from unittest import TestCase


class TestVideoBenchmarkSetup(TestCase):
    def test_download_script_exists_with_expected_video_benchmark_manifest(self):
        repo_root = Path(__file__).resolve().parents[4]
        script_path = repo_root / "tools" / "download_video_benchmarks.py"

        self.assertTrue(script_path.exists(), "video benchmark download script should exist")

        tree = ast.parse(script_path.read_text(encoding="utf-8"))
        strings = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]

        self.assertIn("/root/autodl-fs/videomme", strings)
        self.assertIn("/root/autodl-fs/longvideobench", strings)
        self.assertIn("lmms-lab/Video-MME", strings)
        self.assertIn("longvideobench/LongVideoBench", strings)

    def test_task_configs_point_to_fixed_video_benchmark_roots(self):
        repo_root = Path(__file__).resolve().parents[4]
        file_expectations = {
            repo_root / "lmms-eval" / "lmms_eval" / "tasks" / "videomme" / "videomme.yaml": "/root/autodl-fs/videomme",
            repo_root / "lmms-eval" / "lmms_eval" / "tasks" / "longvideobench" / "longvideobench_val_v.yaml": "/root/autodl-fs/longvideobench",
            repo_root / "lmms-eval" / "lmms_eval" / "tasks" / "longvideobench" / "utils.py": "/root/autodl-fs/longvideobench",
        }

        for path, expected in file_expectations.items():
            text = path.read_text(encoding="utf-8")
            self.assertIn(expected, text, f"{path} should reference {expected}")
