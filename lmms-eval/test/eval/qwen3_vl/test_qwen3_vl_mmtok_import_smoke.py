from pathlib import Path
from unittest import TestCase

from lmms_eval.models import get_model
from lmms_eval.models.chat import qwen3_vl_mmtok


class TestQwen3VLMMTokImportSmoke(TestCase):
    def test_get_model_returns_qwen3_vl_mmtok_class(self):
        model_cls = get_model("qwen3_vl_mmtok")
        self.assertEqual(model_cls.__name__, "Qwen3_VL_MMTok")

    def test_flashvid_repo_root_exists(self):
        repo_root = qwen3_vl_mmtok._resolve_flashvid_repo_root()
        self.assertTrue(repo_root.exists())
        self.assertTrue(
            (repo_root / "flashvid" / "mmtok" / "qwen" / "qwen3_vl_mmtok.py").exists()
        )

    def test_helper_resolves_flashvid_workspace_root(self):
        repo_root = qwen3_vl_mmtok._resolve_flashvid_repo_root()
        self.assertEqual(repo_root.name, "FlashVID")

    def test_wrapper_loads_from_bundled_flashvid_package(self):
        wrapper = qwen3_vl_mmtok._load_mmtok_qwen3_wrapper()
        self.assertEqual(wrapper.__module__, "flashvid.mmtok.qwen.qwen3_vl_mmtok")

    def test_question_extractor_accepts_batched_hf_messages(self):
        question_text = qwen3_vl_mmtok._load_mmtok_qwen3_wrapper().__globals__[
            "extract_question_from_messages"
        ](
            [
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": "dummy"},
                            {"type": "text", "text": "What is shown in the image?"},
                        ],
                    },
                ]
            ]
        )
        self.assertEqual(question_text, "What is shown in the image?")
