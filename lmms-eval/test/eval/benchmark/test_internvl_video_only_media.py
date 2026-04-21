from unittest import TestCase

from lmms_eval.models.chat.internvl_hf import _prepare_internvl_media_inputs


class TestInternVLVideoOnlyMedia(TestCase):
    def test_video_only_batch_uses_none_for_images(self):
        visuals = []
        videos = ["demo.mp4"]

        normalized_visuals, normalized_videos, image_sizes = (
            _prepare_internvl_media_inputs(visuals, videos)
        )

        self.assertIsNone(normalized_visuals)
        self.assertEqual(normalized_videos, videos)
        self.assertEqual(image_sizes, [])
