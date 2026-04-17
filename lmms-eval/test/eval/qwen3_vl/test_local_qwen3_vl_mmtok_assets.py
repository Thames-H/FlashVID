import ast
from pathlib import Path
from unittest import TestCase


class TestLocalQwen3VLMMTokAssets(TestCase):
    def test_local_runner_exists_and_uses_local_model_defaults(self):
        repo_root = Path(__file__).resolve().parents[4]
        runner_path = repo_root / "playground" / "qwen3_vl_mmtok_local.py"

        self.assertTrue(runner_path.exists(), "local MMTok runner should exist")

        runner_ast = ast.parse(runner_path.read_text(encoding="utf-8"))
        runner_class_names = {
            node.name for node in runner_ast.body if isinstance(node, ast.ClassDef)
        }
        self.assertIn("InferenceArguments", runner_class_names)

        string_constants = [
            node.value
            for node in ast.walk(runner_ast)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]
        self.assertIn("QWEN3_VL_MODEL_PATH", string_constants)
        self.assertIn("Qwen3-VL-2B-Instruct", string_constants)
        self.assertIn("assets", string_constants)
        self.assertIn("method.png", string_constants)
        self.assertIn("Qgr4dcsY-60.mp4", string_constants)

    def test_image_benchmark_script_targets_mme_gqa_pope(self):
        repo_root = Path(__file__).resolve().parents[4]
        script_path = repo_root / "scripts" / "qwen3_vl_mmtok_img.sh"

        self.assertTrue(script_path.exists(), "Qwen3-VL MMTok image benchmark script should exist")

        script_text = script_path.read_text(encoding="utf-8")
        self.assertIn("Qwen/Qwen3-VL-8B-Instruct", script_text)
        self.assertIn("autodl-tmp/Qwen3-VL-8B-Instruct", script_text)
        self.assertIn('"gqa"', script_text)
        self.assertIn('"mme"', script_text)
        self.assertIn('"pope"', script_text)
        self.assertIn("--model qwen3_vl_mmtok", script_text)
        self.assertIn('MAX_NUM_FRAMES="${MAX_NUM_FRAMES:-8}"', script_text)
        self.assertIn("max_num_frames=$MAX_NUM_FRAMES", script_text)
        self.assertIn('ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"', script_text)
        self.assertIn('RETENTION_RATIOS=(0.20)', script_text)
