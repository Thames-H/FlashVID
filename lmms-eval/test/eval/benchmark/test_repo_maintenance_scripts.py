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

    def test_repo_enforces_lf_for_shell_scripts(self):
        repo_root = Path(__file__).resolve().parents[4]
        gitattributes_path = repo_root / ".gitattributes"

        self.assertTrue(gitattributes_path.exists(), ".gitattributes should exist")
        gitattributes_text = gitattributes_path.read_text(encoding="utf-8")
        self.assertIn("*.sh text eol=lf", gitattributes_text)

        for script_path in (repo_root / "scripts").rglob("*.sh"):
            content = script_path.read_bytes()
            self.assertNotIn(
                b"\r\n",
                content,
                f"{script_path} should use LF line endings for Linux bash compatibility",
            )
