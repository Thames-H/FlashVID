from unittest import TestCase

import torch

from lmms_eval.models.chat.qwen3_vl_ours_v2 import (
    _compute_fes_scores,
    _compute_fes_scores_from_compact_inputs,
)


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

    def test_compact_fes_scores_match_full_tensor_version(self):
        attn_logits = torch.tensor(
            [
                [
                    [
                        [0.1, 0.3, -0.2, 0.0, 0.5, -0.1],
                        [0.0, -0.2, 0.4, 0.6, 0.1, 0.2],
                        [0.2, 0.5, 0.1, -0.4, 0.2, 0.0],
                        [0.3, -0.1, 0.2, 0.1, 0.4, 0.2],
                        [0.4, 0.2, -0.3, 0.5, 0.0, 0.1],
                        [0.6, -0.4, 0.3, 0.2, 0.1, -0.2],
                    ],
                    [
                        [0.2, -0.3, 0.1, 0.4, 0.0, 0.2],
                        [0.1, 0.2, -0.5, 0.3, 0.6, -0.1],
                        [0.0, 0.4, 0.2, -0.2, 0.1, 0.5],
                        [0.5, 0.0, -0.1, 0.2, 0.3, 0.4],
                        [0.3, 0.1, 0.2, -0.4, 0.5, 0.0],
                        [-0.2, 0.6, 0.4, 0.1, -0.1, 0.3],
                    ],
                ]
            ],
            dtype=torch.float32,
        )
        value_states = torch.tensor(
            [
                [
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [2.0, 1.0],
                        [1.0, 2.0],
                        [0.5, 1.5],
                        [1.5, 0.5],
                    ]
                ]
            ],
            dtype=torch.float32,
        )
        visual_positions = torch.tensor([1, 3, 5])
        text_positions = torch.tensor([0, 4])

        full_scores = _compute_fes_scores(
            attn_logits=attn_logits,
            value_states=value_states,
            visual_positions=visual_positions,
            text_positions=text_positions,
        )
        compact_logits = attn_logits.index_select(2, text_positions).index_select(
            3, visual_positions
        )
        compact_values = value_states.index_select(2, visual_positions)
        compact_scores = _compute_fes_scores_from_compact_inputs(
            text_to_vis_logits=compact_logits,
            visual_value_states=compact_values,
        )

        self.assertTrue(torch.allclose(full_scores, compact_scores, atol=1e-6))
