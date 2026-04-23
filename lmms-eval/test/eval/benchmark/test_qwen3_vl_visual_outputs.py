import unittest

import torch

from lmms_eval.models.chat.qwen3_vl_ours_v2 import _unpack_visual_outputs


class _FakeOutputs:
    def __init__(self, last_hidden_state=None, deepstack_features=None):
        self.last_hidden_state = last_hidden_state
        self.deepstack_features = deepstack_features


class _FakeMerger:
    class _Norm:
        normalized_shape = (3,)

    class _Linear:
        out_features = 3

    norm = _Norm()
    linear_fc2 = _Linear()

    def __call__(self, tensor):
        if tensor.ndim == 3:
            return tensor.mean(dim=1)
        if tensor.ndim == 2:
            return tensor.reshape(-1, 4, tensor.shape[-1]).mean(dim=1)
        return tensor


class TestQwen3VLVisualOutputs(unittest.TestCase):
    def test_unpack_merges_raw_3d_visual_tokens_before_flattening(self):
        raw_features = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
        deepstack_feature = raw_features + 100

        outputs = _FakeOutputs(
            last_hidden_state=raw_features,
            deepstack_features=[deepstack_feature],
        )

        merged_features, merged_deepstack = _unpack_visual_outputs(
            outputs, merger=_FakeMerger()
        )

        self.assertEqual(merged_features.shape, (2, 3))
        self.assertTrue(torch.equal(merged_features, raw_features.mean(dim=1)))
        self.assertEqual(len(merged_deepstack), 1)
        self.assertEqual(merged_deepstack[0].shape, (2, 3))
        self.assertTrue(
            torch.equal(merged_deepstack[0], deepstack_feature.mean(dim=1))
        )

    def test_unpack_merges_raw_2d_visual_tokens_before_returning(self):
        raw_features = torch.arange(8 * 3, dtype=torch.float32).reshape(8, 3)
        outputs = _FakeOutputs(last_hidden_state=raw_features)

        merged_features, merged_deepstack = _unpack_visual_outputs(
            outputs, merger=_FakeMerger()
        )

        self.assertEqual(merged_features.shape, (2, 3))
        self.assertTrue(
            torch.equal(merged_features, raw_features.reshape(2, 4, 3).mean(dim=1))
        )
        self.assertIsNone(merged_deepstack)


if __name__ == "__main__":
    unittest.main()