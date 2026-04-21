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

    def test_ours_v3_video_scripts_support_single_process_model_parallelism(self):
        repo_root = Path(__file__).resolve().parents[4]
        paths = [
            repo_root / "scripts" / "ours_v3" / "qwen3_vl_8b_video.sh",
            repo_root / "scripts" / "ours_v3" / "llava_onevision_7b_video.sh",
            repo_root / "scripts" / "ours_v3" / "internvl3_5_8b_video.sh",
        ]

        for path in paths:
            text = path.read_text(encoding="utf-8")
            self.assertIn('if [[ "$NUM_PROCESSES" == "1" ]]; then', text)
            self.assertIn('DEVICE_MAP="${DEVICE_MAP:-$DEVICE_MAP_DEFAULT}"', text)
            self.assertIn('device_map=$DEVICE_MAP', text)
