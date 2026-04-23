from pathlib import Path
from unittest import TestCase

from lmms_eval.models.chat.llava_hf import LlavaHf


class TestLlavaChatVideoInputs(TestCase):
    def test_prepare_chat_media_inputs_uses_fixed_frame_budget_for_videos(self):
        model = object.__new__(LlavaHf)
        model.max_frames_num = 16
        calls = []

        def fake_load_video(video_path, max_frames_num):
            calls.append((video_path, max_frames_num))
            return "sampled-video"

        model.load_video = fake_load_video

        visuals, videos = model._prepare_chat_media_inputs([], ["video.mp4"])

        self.assertIsNone(visuals)
        self.assertEqual(videos, ["sampled-video"])
        self.assertEqual(calls, [(["video.mp4"], 16)])

    def test_llava_ours_v3_video_script_uses_conservative_default_frame_budget(self):
        repo_root = Path(__file__).resolve().parents[4]
        script_path = repo_root / "scripts" / "ours_v3" / "llava_onevision_7b_video.sh"

        self.assertTrue(script_path.exists(), "LLaVA-OneVision ours_v3 video script should exist")

        text = script_path.read_text(encoding="utf-8")
        self.assertIn('MAX_FRAMES_NUM="${MAX_FRAMES_NUM:-16}"', text)
        self.assertIn("--model llava_onevision_ours_v3", text)
