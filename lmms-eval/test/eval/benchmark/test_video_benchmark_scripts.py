from pathlib import Path
from unittest import TestCase


class TestVideoBenchmarkScripts(TestCase):
    def test_qwen_and_llava_scripts_target_requested_video_tasks(self):
        repo_root = Path(__file__).resolve().parents[4]
        paths = [
            repo_root / "scripts" / "qwen3_vl.sh",
            repo_root / "scripts" / "baseline" / "qwen3_vl.sh",
            repo_root / "scripts" / "llava_ov.sh",
            repo_root / "scripts" / "baseline" / "llava_ov.sh",
        ]

        for path in paths:
            text = path.read_text(encoding="utf-8")
            self.assertIn('"videomme"', text, f"{path} should target videomme")
            self.assertIn('"longvideobench_val_v"', text, f"{path} should target longvideobench_val_v")

    def test_internvl_video_script_exists_and_uses_expected_model(self):
        repo_root = Path(__file__).resolve().parents[4]
        baseline_script = repo_root / "scripts" / "baseline" / "internvl3_5_video.sh"

        self.assertTrue(baseline_script.exists(), "InternVL3.5 baseline video script should exist")

        baseline_text = baseline_script.read_text(encoding="utf-8")

        self.assertIn("--model internvl3_5", baseline_text)
        self.assertIn("modality=video", baseline_text)
        self.assertIn('"videomme"', baseline_text)
        self.assertIn('"longvideobench_val_v"', baseline_text)

    def test_internvl_original_video_script_matches_official_reference_budget(self):
        repo_root = Path(__file__).resolve().parents[4]
        script_path = (
            repo_root
            / "scripts"
            / "ours_v3"
            / "internvl3_5_8b_video_original.sh"
        )

        self.assertTrue(
            script_path.exists(),
            "InternVL3.5 original video script should exist",
        )

        text = script_path.read_text(encoding="utf-8")
        self.assertIn("--model internvl3_5_original", text)
        self.assertIn('NUM_FRAME="${NUM_FRAME:-8}"', text)
        self.assertIn('MAX_NUM="${MAX_NUM:-1}"', text)
        self.assertIn('USE_FLASH_ATTN="${USE_FLASH_ATTN:-true}"', text)
        self.assertIn('"videomme"', text)
        self.assertIn('"longvideobench_val_v"', text)

    def test_qwen3_original_video_entry_exists(self):
        repo_root = Path(__file__).resolve().parents[4]
        script_path = (
            repo_root
            / "scripts"
            / "ours_v3"
            / "qwen3_vl_8b_video_original.sh"
        )
        models_init = (
            repo_root
            / "lmms-eval"
            / "lmms_eval"
            / "models"
            / "__init__.py"
        )

        self.assertTrue(
            script_path.exists(),
            "Qwen3-VL original video script should exist",
        )

        script_text = script_path.read_text(encoding="utf-8")
        self.assertIn("--model qwen3_vl_original", script_text)
        self.assertIn('"videomme"', script_text)
        self.assertIn('"longvideobench_val_v"', script_text)

        models_text = models_init.read_text(encoding="utf-8")
        self.assertIn(
            '"qwen3_vl_original": "lmms_eval.models.chat.qwen3_vl_original.Qwen3_VL_Original"',
            models_text,
        )

    def test_ours_v3_video_scripts_support_single_process_model_parallelism(self):
        repo_root = Path(__file__).resolve().parents[4]
        paths = [
            repo_root / "scripts" / "ours_v3" / "qwen3_vl_8b_video.sh",
            repo_root / "scripts" / "ours_v3" / "qwen3_vl_8b_video_original.sh",
            repo_root / "scripts" / "ours_v3" / "llava_onevision_7b_video.sh",
            repo_root / "scripts" / "ours_v3" / "internvl3_5_8b_video.sh",
        ]

        for path in paths:
            text = path.read_text(encoding="utf-8")
            self.assertIn('if [[ "$NUM_PROCESSES" == "1" ]]; then', text)
            self.assertIn('DEVICE_MAP="${DEVICE_MAP:-$DEVICE_MAP_DEFAULT}"', text)
            self.assertIn('device_map=$DEVICE_MAP', text)

    def test_internvl_ours_v3_video_script_uses_safe_default_budget(self):
        repo_root = Path(__file__).resolve().parents[4]
        path = repo_root / "scripts" / "ours_v3" / "internvl3_5_8b_video.sh"
        text = path.read_text(encoding="utf-8")

        self.assertIn('CANDIDATE_RATIO="${CANDIDATE_RATIO:-0.25}"', text)
        self.assertIn('MAX_SCORE_TEXT_TOKENS="${MAX_SCORE_TEXT_TOKENS:-8}"', text)
        self.assertIn('MAX_SCORE_HEADS="${MAX_SCORE_HEADS:-8}"', text)
        self.assertIn('TEXT_CHUNK_SIZE="${TEXT_CHUNK_SIZE:-16}"', text)
        self.assertIn('MAX_PATCHES="${MAX_PATCHES:-4}"', text)
        self.assertIn('NUM_FRAMES="${NUM_FRAMES:-8}"', text)
        self.assertIn('ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"', text)

    def test_ours_v3_video_scripts_enable_resume_cache_by_default(self):
        repo_root = Path(__file__).resolve().parents[4]
        paths = [
            repo_root / "scripts" / "ours_v3" / "qwen3_vl_8b_video.sh",
            repo_root / "scripts" / "ours_v3" / "qwen3_vl_8b_video_original.sh",
            repo_root / "scripts" / "ours_v3" / "internvl3_5_8b_video.sh",
        ]

        for path in paths:
            text = path.read_text(encoding="utf-8")
            self.assertIn(
                'export LMMS_EVAL_USE_CACHE="${LMMS_EVAL_USE_CACHE:-True}"',
                text,
                f"{path} should enable lmms-eval JSONL cache by default",
            )
            self.assertIn(
                'export LMMS_EVAL_HOME="${LMMS_EVAL_HOME:-$PROJECT_ROOT/.cache/lmms-eval}"',
                text,
                f"{path} should use a stable repo-local cache root by default",
            )
