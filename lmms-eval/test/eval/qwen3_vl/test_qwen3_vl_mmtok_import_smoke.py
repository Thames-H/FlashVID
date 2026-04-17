from pathlib import Path
from unittest import TestCase

from lmms_eval.models import get_model
from lmms_eval.models.chat import qwen3_vl_mmtok


class TestQwen3VLMMTokImportSmoke(TestCase):
    def test_get_model_returns_qwen3_vl_mmtok_class(self):
        model_cls = get_model("qwen3_vl_mmtok")
        self.assertEqual(model_cls.__name__, "Qwen3_VL_MMTok")

    def test_local_mmtok_repo_root_exists(self):
        repo_root = qwen3_vl_mmtok._resolve_local_mmtok_repo_root()
        self.assertTrue(repo_root.exists())
        self.assertTrue((repo_root / "mmtok" / "qwen" / "qwen3_vl_mmtok.py").exists())

    def test_helper_resolves_workspace_copy(self):
        repo_root = qwen3_vl_mmtok._resolve_local_mmtok_repo_root()
        expected_suffix = Path("reference_mmtok") / "MMTok"
        self.assertEqual(repo_root.parts[-2:], expected_suffix.parts)
