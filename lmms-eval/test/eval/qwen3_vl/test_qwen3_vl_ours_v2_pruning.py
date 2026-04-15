from unittest import TestCase

import torch

from lmms_eval.models.chat.qwen3_vl_ours_v2 import _compute_fes_scores


class TestQwen3VLOursV2Pruning(TestCase):
    def test_compute_fes_scores_supports_non_contiguous_visual_positions(self):
        attn_logits = torch.zeros(1, 1, 8, 8)
        value_states = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0],
                        [1.0, 0.0],
                        [2.0, 0.0],
                        [3.0, 0.0],
                        [4.0, 0.0],
                        [5.0, 0.0],
                        [6.0, 0.0],
                        [7.0, 0.0],
                    ]
                ]
            ]
        )
        visual_positions = torch.tensor([1, 2, 5, 6])
        text_positions = torch.tensor([7])

        scores = _compute_fes_scores(
            attn_logits=attn_logits,
            value_states=value_states,
            visual_positions=visual_positions,
            text_positions=text_positions,
        )

        self.assertEqual(tuple(scores.shape), (4,))
        self.assertTrue(torch.all(scores >= 0))
