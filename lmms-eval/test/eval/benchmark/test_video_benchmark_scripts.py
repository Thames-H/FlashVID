from pathlib import Path
import re
from unittest import TestCase


class TestVideoBenchmarkScripts(TestCase):
    def test_ours_v3_scripts_use_in_file_configuration_only(self):
        repo_root = Path(__file__).resolve().parents[4]
        scripts_dir = repo_root / "scripts" / "ours_v3"
        script_paths = sorted(scripts_dir.glob("*.sh"))

        self.assertGreater(len(script_paths), 0, "Expected to validate ours_v3 scripts")

        for path in script_paths:
            text = path.read_text(encoding="utf-8")
            self.assertIn("# Editable configuration. Change values here instead of exporting env vars.", text)
            self.assertNotRegex(text, r"\$\{[^}]+:-", f"{path} should not use shell default overrides")
            self.assertNotIn("TASKS_CSV", text, f"{path} should not read task lists from env vars")
            self.assertNotIn("RETENTION_RATIOS_CSV", text, f"{path} should not read retention ratios from env vars")

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
        self.assertIn('AUTODL_MODEL_PATH="$HOME/autodl-tmp/InternVL3_5-8B-HF"', text)
        self.assertIn('DEFAULT_PRETRAINED="OpenGVLab/InternVL3_5-8B-HF"', text)
        self.assertRegex(text, re.compile(r"^NUM_FRAMES=8$", re.MULTILINE))
        self.assertRegex(text, re.compile(r"^MAX_PATCHES=12$", re.MULTILINE))
        self.assertRegex(text, re.compile(r'^ATTN_IMPLEMENTATION="flash_attention_2"$', re.MULTILINE))
        self.assertRegex(text, re.compile(r'^LOW_CPU_MEM_USAGE="true"$', re.MULTILINE))
        self.assertIn('num_frames=$NUM_FRAMES', text)
        self.assertIn('max_patches=$MAX_PATCHES', text)
        self.assertIn('attn_implementation=$ATTN_IMPLEMENTATION', text)
        self.assertIn('low_cpu_mem_usage=$LOW_CPU_MEM_USAGE', text)
        self.assertRegex(text, re.compile(r'^TASKS=\("videomme" "longvideobench_val_v"\)$', re.MULTILINE))
        self.assertIn('"videomme"', text)
        self.assertIn('"longvideobench_val_v"', text)
        self.assertNotIn("<<<", text)
