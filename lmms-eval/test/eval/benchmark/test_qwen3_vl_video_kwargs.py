import unittest

from lmms_eval.models.chat.qwen3_vl_ours_v2 import (
    _build_qwen_message_video_kwargs,
    _build_qwen_processor_video_kwargs,
)


class Qwen3VideoKwargsTest(unittest.TestCase):
    def test_build_message_video_kwargs_uses_nframes_without_fps(self):
        video_kwargs = _build_qwen_message_video_kwargs(
            max_pixels=1605632,
            min_pixels=200704,
            max_num_frames=8,
            fps=None,
        )

        self.assertEqual(
            video_kwargs,
            {
                "max_pixels": 1605632,
                "min_pixels": 200704,
                "nframes": 8,
            },
        )

    def test_build_message_video_kwargs_uses_fps_and_max_frames(self):
        video_kwargs = _build_qwen_message_video_kwargs(
            max_pixels=1605632,
            min_pixels=200704,
            max_num_frames=8,
            fps=2.0,
        )

        self.assertEqual(
            video_kwargs,
            {
                "max_pixels": 1605632,
                "min_pixels": 200704,
                "fps": 2.0,
                "max_frames": 8,
            },
        )

    def test_build_processor_video_kwargs_keeps_only_processor_safe_values(self):
        processor_kwargs = _build_qwen_processor_video_kwargs(
            {
                "do_sample_frames": False,
                "fps": [2.0],
                "nframes": 8,
                "max_frames": 8,
                "max_pixels": 1605632,
            }
        )

        self.assertEqual(processor_kwargs, {"do_sample_frames": False})


if __name__ == "__main__":
    unittest.main()
