import ast
from pathlib import Path
from unittest import TestCase


class TestLocalQwen3VLOursV2Assets(TestCase):
    def test_local_runner_and_download_helper_exist_with_root_model_path(self):
        repo_root = Path(__file__).resolve().parents[4]
        runner_path = repo_root / "playground" / "qwen3_vl_ours_v2_local.py"
        downloader_path = repo_root / "tools" / "download_qwen3_vl_2b.py"

        self.assertTrue(runner_path.exists(), "local runner should exist")
        self.assertTrue(downloader_path.exists(), "download helper should exist")

        runner_ast = ast.parse(runner_path.read_text(encoding="utf-8"))
        downloader_ast = ast.parse(downloader_path.read_text(encoding="utf-8"))

        runner_class_names = {
            node.name for node in runner_ast.body if isinstance(node, ast.ClassDef)
        }
        self.assertIn("InferenceArguments", runner_class_names)

        default_strings = []
        for node in ast.walk(runner_ast):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                default_strings.append(node.value)
        self.assertIn("Qwen/Qwen3-VL-2B-Instruct", default_strings)
        self.assertIn("Qwen3-VL-2B-Instruct", default_strings)

        downloader_strings = []
        for node in ast.walk(downloader_ast):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                downloader_strings.append(node.value)
        self.assertIn("Qwen/Qwen3-VL-2B-Instruct", downloader_strings)
        self.assertIn("Qwen3-VL-2B-Instruct", downloader_strings)
