from pathlib import Path
from unittest import TestCase


class TestSinkAnalysisScripts(TestCase):
    def test_sink_analysis_scripts_exist(self):
        repo_root = Path(__file__).resolve().parents[4]
        expected = [
            repo_root / "scripts" / "sink_analysis" / "run_all.sh",
            repo_root / "scripts" / "sink_analysis" / "collect_llava.sh",
            repo_root / "scripts" / "sink_analysis" / "collect_qwen3.sh",
            repo_root / "scripts" / "sink_analysis" / "merge.sh",
            repo_root / "scripts" / "sink_analysis" / "analyze.sh",
        ]

        for path in expected:
            self.assertTrue(path.exists(), f"{path} should exist")

    def test_run_all_mentions_resume_and_pipeline_stages(self):
        repo_root = Path(__file__).resolve().parents[4]
        script_path = repo_root / "scripts" / "sink_analysis" / "run_all.sh"
        self.assertTrue(script_path.exists(), f"{script_path} should exist")

        text = script_path.read_text(encoding="utf-8")
        self.assertIn("--resume", text)
        self.assertIn("fetp", text)
        self.assertIn("attention", text)
        self.assertIn("mmtok", text)
        self.assertIn("analyze", text)

    def test_run_all_invokes_collect_merge_and_ablation_steps(self):
        repo_root = Path(__file__).resolve().parents[4]
        script_path = repo_root / "scripts" / "sink_analysis" / "run_all.sh"
        self.assertTrue(script_path.exists(), f"{script_path} should exist")

        text = script_path.read_text(encoding="utf-8")
        self.assertIn("collect_llava.sh", text)
        self.assertIn("collect_qwen3.sh", text)
        self.assertIn("build-ablation", text)
        self.assertIn("rerun-ablation", text)
        self.assertIn("merge.sh", text)
        self.assertIn("analyze.sh", text)
