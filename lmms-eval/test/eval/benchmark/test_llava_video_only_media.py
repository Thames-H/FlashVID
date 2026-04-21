import unittest

from lmms_eval.models.chat.llava_hf import _prepare_llava_media_inputs


class LlavaVideoOnlyMediaTest(unittest.TestCase):
    def test_prepare_media_inputs_normalizes_empty_images_for_video_only_requests(self):
        visuals, videos, image_sizes = _prepare_llava_media_inputs([], ["demo.mp4"])

        self.assertIsNone(visuals)
        self.assertEqual(videos, ["demo.mp4"])
        self.assertEqual(image_sizes, [])
