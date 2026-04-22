import unittest
from types import SimpleNamespace

from lmms_eval.models.chat.llava_hf import _build_llava_processor_kwargs


class LlavaOnevisionVideoKwargsTest(unittest.TestCase):
    def test_build_processor_kwargs_enables_video_sampling_for_onevision(self):
        config = SimpleNamespace(
            model_type="llava_onevision",
            vision_config=SimpleNamespace(image_size=384),
        )

        images_kwargs, videos_kwargs = _build_llava_processor_kwargs(
            model_config=config,
            max_frames_num=8,
        )

        self.assertEqual(images_kwargs, {})
        self.assertEqual(
            videos_kwargs,
            {
                "num_frames": 8,
                "do_sample_frames": True,
                "size": {"height": 384, "width": 384},
            },
        )

    def test_build_processor_kwargs_skips_video_sampling_for_non_onevision(self):
        config = SimpleNamespace(model_type="llava")

        images_kwargs, videos_kwargs = _build_llava_processor_kwargs(
            model_config=config,
            max_frames_num=8,
        )

        self.assertEqual(images_kwargs, {})
        self.assertEqual(videos_kwargs, {})
