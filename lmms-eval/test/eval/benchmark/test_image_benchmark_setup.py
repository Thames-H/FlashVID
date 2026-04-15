import ast
from pathlib import Path
from unittest import TestCase


class TestImageBenchmarkSetup(TestCase):
    def test_download_script_exists_with_expected_benchmark_manifest(self):
        repo_root = Path(__file__).resolve().parents[4]
        script_path = repo_root / "tools" / "download_image_benchmarks.py"

        self.assertTrue(script_path.exists(), "download script should exist")

        tree = ast.parse(script_path.read_text(encoding="utf-8"))
        strings = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]

        self.assertIn("/root/autodl-tmp/benchmark", strings)
        self.assertIn("lmms-lab/GQA", strings)
        self.assertIn("lmms-lab/ScienceQA", strings)
        self.assertIn("lmms-lab/MMBench", strings)
        self.assertIn("lmms-lab/MME", strings)
        self.assertIn("lmms-lab/POPE", strings)
        self.assertIn("lmms-lab/OCRBench-v2", strings)

    def test_task_configs_point_to_local_benchmark_root(self):
        repo_root = Path(__file__).resolve().parents[4]
        benchmark_root = "/root/autodl-tmp/benchmark"

        file_expectations = {
            repo_root / "lmms-eval" / "lmms_eval" / "tasks" / "gqa" / "gqa.yaml": benchmark_root + "/GQA",
            repo_root / "lmms-eval" / "lmms_eval" / "tasks" / "gqa" / "utils.py": benchmark_root + "/GQA/testdev_balanced_images/testdev-00000-of-00001.parquet",
            repo_root / "lmms-eval" / "lmms_eval" / "tasks" / "scienceqa" / "scienceqa_img.yaml": benchmark_root + "/ScienceQA",
            repo_root / "lmms-eval" / "lmms_eval" / "tasks" / "mmbench" / "_default_template_mmbench_en_yaml": benchmark_root + "/MMBench",
            repo_root / "lmms-eval" / "lmms_eval" / "tasks" / "mme" / "mme.yaml": benchmark_root + "/MME",
            repo_root / "lmms-eval" / "lmms_eval" / "tasks" / "pope" / "pope.yaml": benchmark_root + "/POPE",
            repo_root / "lmms-eval" / "lmms_eval" / "tasks" / "ocrbench" / "ocrbench.yaml": benchmark_root + "/OCRBench",
        }

        for path, expected in file_expectations.items():
            text = path.read_text(encoding="utf-8")
            self.assertIn(expected, text, f"{path} should reference {expected}")
