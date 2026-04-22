import unittest

import torch

from lmms_eval.models.chat.qwen3_vl_ours_v2 import (
    _slice_position_embeddings,
    _unpack_visual_outputs,
)


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

    @unittest.skipUnless(
        torch.cuda.is_available(),
        "CUDA is required to validate cross-device position slicing",
    )
    def test_slice_position_embeddings_moves_indices_to_embedding_device(self):
        cos = torch.randn(1, 10, 8, device="cuda")
        sin = torch.randn(1, 10, 8, device="cuda")
        positions = torch.tensor([0, 4, 9], device="cpu")

        sliced_cos, sliced_sin = _slice_position_embeddings(
            (cos, sin),
            positions,
        )

        self.assertEqual(sliced_cos.device.type, "cuda")
        self.assertEqual(sliced_sin.device.type, "cuda")
        self.assertEqual(tuple(sliced_cos.shape), (1, 3, 8))
        self.assertEqual(tuple(sliced_sin.shape), (1, 3, 8))


if __name__ == "__main__":
    unittest.main()
