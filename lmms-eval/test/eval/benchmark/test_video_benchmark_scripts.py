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

    def test_internvl_original_video_script_uses_hf_backend_defaults(self):
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
        self.assertIn('AUTODL_MODEL_PATH="${HOME}/autodl-tmp/InternVL3_5-8B-HF"', text)
        self.assertIn('DEFAULT_PRETRAINED="OpenGVLab/InternVL3_5-8B-HF"', text)
        self.assertIn('NUM_FRAMES="${NUM_FRAMES:-8}"', text)
        self.assertIn('MAX_PATCHES="${MAX_PATCHES:-12}"', text)
        self.assertIn('ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"', text)
        self.assertIn('LOW_CPU_MEM_USAGE="${LOW_CPU_MEM_USAGE:-true}"', text)
        self.assertIn('num_frames=$NUM_FRAMES', text)
        self.assertIn('max_patches=$MAX_PATCHES', text)
        self.assertIn('attn_implementation=$ATTN_IMPLEMENTATION', text)
        self.assertIn('low_cpu_mem_usage=$LOW_CPU_MEM_USAGE', text)
        self.assertIn('"videomme"', text)
        self.assertIn('"longvideobench_val_v"', text)
