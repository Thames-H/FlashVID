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

    def test_collect_scripts_handle_full_keep_ratio_without_float_conversion_error(self):
        repo_root = Path(__file__).resolve().parents[4]
        llava_script = repo_root / "scripts" / "sink_analysis" / "collect_llava.sh"
        qwen_script = repo_root / "scripts" / "sink_analysis" / "collect_qwen3.sh"

        llava_text = llava_script.read_text(encoding="utf-8")
        qwen_text = qwen_script.read_text(encoding="utf-8")

        self.assertIn("label.lower() == 'full'", llava_text)
        self.assertIn("label.lower() == 'full'", qwen_text)

    def test_collect_scripts_use_registered_full_model_names_and_guard_resume_find(self):
        repo_root = Path(__file__).resolve().parents[4]
        llava_script = repo_root / "scripts" / "sink_analysis" / "collect_llava.sh"
        qwen_script = repo_root / "scripts" / "sink_analysis" / "collect_qwen3.sh"

        llava_text = llava_script.read_text(encoding="utf-8")
        qwen_text = qwen_script.read_text(encoding="utf-8")

        self.assertIn('MODEL_NAME="llava_hf"', llava_text)
        self.assertIn('MODEL_NAME="qwen3_vl"', qwen_text)
        self.assertIn('-d "${RUN_ROOT}"', llava_text)
        self.assertIn('-d "${RUN_ROOT}"', qwen_text)
        self.assertIn("No results json produced under", llava_text)
        self.assertIn("No results json produced under", qwen_text)

    def test_code_wrapper_scripts_exist_for_pull_and_llava_only_run(self):
        repo_root = Path(__file__).resolve().parents[4]
        pull_script = repo_root / "code" / "pull_attn_sink.sh"
        run_script = repo_root / "code" / "run_llava_sink_analysis.sh"
        focused_run_script = repo_root / "code" / "run_llava_sink_analysis_10pct.sh"

        self.assertTrue(pull_script.exists(), f"{pull_script} should exist")
        self.assertTrue(run_script.exists(), f"{run_script} should exist")
        self.assertTrue(focused_run_script.exists(), f"{focused_run_script} should exist")

        pull_text = pull_script.read_text(encoding="utf-8")
        self.assertIn("git fetch", pull_text)
        self.assertIn("attn-sink", pull_text)
        self.assertIn("git pull", pull_text)

        run_text = run_script.read_text(encoding="utf-8")
        self.assertIn("collect_llava.sh", run_text)
        self.assertNotIn("collect_qwen3.sh", run_text)
        self.assertIn("build-ablation", run_text)
        self.assertIn("merge.sh", run_text)
        self.assertIn("analyze.sh", run_text)
        self.assertIn("/root/autodl-tmp/llava-onevision-qwen2-7b-ov-hf", run_text)
        self.assertIn('export RATIOS_CSV="${RATIOS_CSV:-5%,10%,20%,25%,50%,75%}"', run_text)
        self.assertIn('export TARGET_LAYER="${TARGET_LAYER:-15}"', run_text)
        self.assertIn('echo "target layer: ${TARGET_LAYER}"', run_text)
        self.assertIn('bash "${SINK_SCRIPT_DIR}/collect_llava.sh"', run_text)
        self.assertIn('bash "${SINK_SCRIPT_DIR}/merge.sh"', run_text)
        self.assertIn('bash "${SINK_SCRIPT_DIR}/analyze.sh"', run_text)

        focused_text = focused_run_script.read_text(encoding="utf-8")
        self.assertIn('export LIMIT="${LIMIT:-64}"', focused_text)
        self.assertIn('export RATIOS_CSV="${RATIOS_CSV:-10%}"', focused_text)
        self.assertIn('export TARGET_LAYER="${TARGET_LAYER:-15}"', focused_text)
        self.assertIn('bash "${SCRIPT_DIR}/run_llava_sink_analysis.sh"', focused_text)
