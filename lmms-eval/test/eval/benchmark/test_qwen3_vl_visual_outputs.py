import unittest

import torch

from lmms_eval.models.chat.qwen3_vl_ours_v2 import _unpack_visual_outputs


class Qwen3VisualOutputsTest(unittest.TestCase):
    def test_unpack_preserves_split_feature_rank(self):
        split_visual_embeds = (
            torch.randn(2, 4, 8),
            torch.randn(1, 4, 8),
        )
        deepstack_features = [torch.randn(3, 8), torch.randn(3, 8)]

        visual_embeds, deepstack = _unpack_visual_outputs(
            (split_visual_embeds, deepstack_features)
        )

        self.assertEqual(visual_embeds.shape, (3, 4, 8))
        self.assertEqual(len(deepstack), 2)
        self.assertEqual(deepstack[0].shape, (3, 8))
        self.assertEqual(deepstack[1].shape, (3, 8))


if __name__ == "__main__":
    unittest.main()
