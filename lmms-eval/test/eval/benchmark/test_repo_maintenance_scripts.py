from pathlib import Path
from unittest import TestCase


class TestRepoMaintenanceScripts(TestCase):
    def test_pull_main_script_is_safe_for_resume_caches(self):
        repo_root = Path(__file__).resolve().parents[4]
        script_path = repo_root / "scripts" / "pull_main.sh"

        self.assertTrue(script_path.exists(), "pull_main.sh should exist")

        text = script_path.read_text(encoding="utf-8")
        self.assertIn("git fetch origin main", text)
        self.assertIn("git pull --ff-only origin main", text)
        self.assertIn("git diff --name-only --diff-filter=U", text)
        self.assertIn("git status --porcelain=v1 --untracked-files=no", text)
        self.assertIn("Untracked files are preserved", text)
