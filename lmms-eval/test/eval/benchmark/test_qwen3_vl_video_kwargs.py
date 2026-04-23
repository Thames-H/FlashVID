import unittest

from lmms_eval.models.chat.qwen3_vl_ours_v2 import (
    _build_message_video_kwargs,
    _build_processor_video_kwargs,
)


class TestQwen3VLVideoKwargs(unittest.TestCase):
    def test_message_video_kwargs_use_nframes_for_presampled_path(self):
        kwargs = _build_message_video_kwargs(
            min_pixels=10,
            max_pixels=20,
            max_num_frames=8,
            fps=None,
        )
        self.assertEqual(kwargs["min_pixels"], 10)
        self.assertEqual(kwargs["max_pixels"], 20)
        self.assertEqual(kwargs["nframes"], 8)
        self.assertNotIn("do_sample_frames", kwargs)

    def test_processor_video_kwargs_strip_sampling_keys(self):
        kwargs = _build_processor_video_kwargs(
            {"do_sample_frames": False, "fps": [2.0], "max_frames": 8}
        )
        self.assertEqual(kwargs, {"do_sample_frames": False})


if __name__ == "__main__":
    unittest.main()