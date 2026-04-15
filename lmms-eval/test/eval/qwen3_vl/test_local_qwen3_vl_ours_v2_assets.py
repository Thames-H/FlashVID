import ast
from pathlib import Path
from unittest import TestCase


class TestLocalQwen3VLOursV2Assets(TestCase):
    def test_local_runner_and_download_helper_exist_with_root_model_path(self):
        repo_root = Path(__file__).resolve().parents[4]
        runner_path = repo_root / "playground" / "qwen3_vl_ours_v2_local.py"
        downloader_path = repo_root / "tools" / "download_qwen3_vl_8b.py"

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
        self.assertIn("Qwen/Qwen3-VL-8B-Instruct", default_strings)
        self.assertIn("Qwen3-VL-8B-Instruct", default_strings)
        self.assertIn("QWEN3_VL_MODEL_PATH", default_strings)
        self.assertIn("autodl-tmp", default_strings)

        downloader_strings = []
        for node in ast.walk(downloader_ast):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                downloader_strings.append(node.value)
        self.assertIn("Qwen/Qwen3-VL-8B-Instruct", downloader_strings)
        self.assertIn("Qwen3-VL-8B-Instruct", downloader_strings)
        self.assertIn("QWEN3_VL_MODEL_PATH", downloader_strings)
        self.assertIn("autodl-tmp", downloader_strings)

    def test_qwen3_wrappers_default_to_8b_instruct(self):
        repo_root = Path(__file__).resolve().parents[4]
        simple_model_path = (
            repo_root / "lmms-eval" / "lmms_eval" / "models" / "simple" / "qwen3_vl.py"
        )
        chat_model_path = (
            repo_root / "lmms-eval" / "lmms_eval" / "models" / "chat" / "qwen3_vl_ours_v2.py"
        )

        simple_ast = ast.parse(simple_model_path.read_text(encoding="utf-8"))
        chat_ast = ast.parse(chat_model_path.read_text(encoding="utf-8"))

        simple_strings = [
            node.value
            for node in ast.walk(simple_ast)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]
        chat_strings = [
            node.value
            for node in ast.walk(chat_ast)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]

        self.assertIn("Qwen/Qwen3-VL-8B-Instruct", simple_strings)
        self.assertIn("Qwen/Qwen3-VL-8B-Instruct", chat_strings)

    def test_image_benchmark_script_targets_requested_tasks(self):
        repo_root = Path(__file__).resolve().parents[4]
        script_path = repo_root / "scripts" / "ours_v2" / "qwen3_vl_8b_img.sh"

        self.assertTrue(script_path.exists(), "image benchmark script should exist")

        script_text = script_path.read_text(encoding="utf-8")
        self.assertIn("Qwen/Qwen3-VL-8B-Instruct", script_text)
        self.assertIn('"gqa"', script_text)
        self.assertIn('"scienceqa_img"', script_text)
        self.assertIn('"mmbench_en"', script_text)
        self.assertIn('"mme"', script_text)
        self.assertIn('"pope"', script_text)
        self.assertIn('"ocrbench"', script_text)
        self.assertIn("--model qwen3_vl_ours_v2", script_text)
        self.assertIn("autodl-tmp/Qwen3-VL-8B-Instruct", script_text)
