import unittest

import torch

from lmms_eval.models.chat.qwen3_vl_ours_v2 import (
    _maybe_merge_qwen_visual_outputs,
    _slice_position_embeddings,
    _unpack_visual_outputs,
)


class _FakeMerger(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states.view(-1, 4, hidden_states.shape[-1]).mean(dim=1)


class _FakeVisual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_merge_size = 2
        self.merger = _FakeMerger()
        self.deepstack_merger_list = torch.nn.ModuleList(
            [_FakeMerger(), _FakeMerger()]
        )


class _FakeConfig:
    class VisionConfig:
        spatial_merge_size = 2

    vision_config = VisionConfig()


class _FakeModel:
    def __init__(self):
        self.visual = _FakeVisual()
        self.config = _FakeConfig()


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

    def test_merge_qwen_visual_outputs_merges_raw_token_counts(self):
        model = _FakeModel()
        visual_embeds = torch.randn(12, 8)
        deepstack_features = [torch.randn(12, 8), torch.randn(12, 8)]
        grid_thw = torch.tensor([[1, 4, 2], [1, 2, 2]])

        merged_visual_embeds, merged_deepstack = _maybe_merge_qwen_visual_outputs(
            model,
            visual_embeds,
            deepstack_features,
            grid_thw,
        )

        self.assertEqual(tuple(merged_visual_embeds.shape), (3, 8))
        self.assertEqual(len(merged_deepstack), 2)
        self.assertEqual(tuple(merged_deepstack[0].shape), (3, 8))
        self.assertEqual(tuple(merged_deepstack[1].shape), (3, 8))

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
