from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import torch

from lmms_eval.models.chat.qwen3_vl_ours_v3 import (
    _aggregate_anchor_scores,
    _compute_fes_scores_from_compact_inputs,
    _compute_fes_scores_from_visual_logits,
    _forward_extract,
    _normalize_anchor_layers,
    _summarize_pruning_stats,
)


class TestQwen3VLOursV3Pruning(TestCase):
    def test_normalize_anchor_layers_sorts_and_deduplicates(self):
        normalized = _normalize_anchor_layers(
            anchor_layers=[15, 3, 15, 8],
            num_layers=24,
        )

        self.assertEqual(normalized, (3, 8, 15))

    def test_aggregate_anchor_scores_supports_weights(self):
        scores = _aggregate_anchor_scores(
            [
                torch.tensor([1.0, 3.0, 5.0]),
                torch.tensor([2.0, 4.0, 6.0]),
            ],
            weights=[1.0, 3.0],
        )

        expected = torch.tensor([1.75, 3.75, 5.75])
        self.assertTrue(torch.allclose(scores, expected))

    def test_summarize_pruning_stats_includes_comparison_speedup(self):
        stats = _summarize_pruning_stats(
            scoring_method="anchor",
            anchor_layers=(3, 8, 15),
            num_visual_tokens=512,
            num_keep=128,
            scoring_time_s=0.4,
            total_pruning_time_s=0.55,
            reference_method="shallow",
            reference_scoring_time_s=0.8,
            topk_overlap=0.625,
        )

        self.assertEqual(stats["pruning_scoring_method"], "anchor")
        self.assertEqual(stats["pruning_anchor_layers"], "3,8,15")
        self.assertEqual(stats["pruning_num_visual_tokens"], 512)
        self.assertEqual(stats["pruning_num_keep"], 128)
        self.assertAlmostEqual(stats["pruning_scoring_time_ms"], 400.0)
        self.assertAlmostEqual(stats["pruning_total_time_ms"], 550.0)
        self.assertEqual(stats["pruning_reference_method"], "shallow")
        self.assertAlmostEqual(stats["pruning_reference_scoring_time_ms"], 800.0)
        self.assertAlmostEqual(stats["pruning_reference_speedup"], 2.0)
        self.assertAlmostEqual(stats["pruning_topk_overlap"], 0.625)

    def test_compute_fes_scores_from_visual_logits_reduces_query_dimension(self):
        attn_logits = torch.zeros(1, 2, 3, 5)
        value_states = torch.tensor(
            [
                [
                    [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]],
                    [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0]],
                ]
            ]
        )
        visual_positions = torch.tensor([1, 3, 4])

        scores = _compute_fes_scores_from_visual_logits(
            attn_logits=attn_logits,
            value_states=value_states,
            visual_positions=visual_positions,
        )

        self.assertEqual(tuple(scores.shape), (3,))
        self.assertTrue(torch.all(scores >= 0))

    def test_compute_fes_scores_from_compact_inputs_matches_chunked_reference(self):
        text_to_vis_logits = torch.tensor(
            [
                [
                    [[1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [2.0, 0.0, -2.0]],
                    [[0.0, 1.0, -1.0], [1.0, 0.0, -1.0], [1.0, 1.0, -2.0]],
                ]
            ]
        )
        visual_value_states = torch.tensor(
            [
                [
                    [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                    [[0.5, 1.5], [1.5, 0.5], [0.0, 2.0]],
                ]
            ]
        )

        full_scores, full_alpha, full_deviation = (
            _compute_fes_scores_from_compact_inputs(
                text_to_vis_logits=text_to_vis_logits,
                visual_value_states=visual_value_states,
                text_chunk_size=16,
            )
        )
        chunked_scores, chunked_alpha, chunked_deviation = (
            _compute_fes_scores_from_compact_inputs(
                text_to_vis_logits=text_to_vis_logits,
                visual_value_states=visual_value_states,
                text_chunk_size=2,
            )
        )

        self.assertTrue(torch.allclose(chunked_scores, full_scores, atol=1e-6))
        self.assertTrue(torch.allclose(chunked_alpha, full_alpha, atol=1e-6))
        self.assertTrue(
            torch.allclose(chunked_deviation, full_deviation, atol=1e-6)
        )

    def test_compute_fes_scores_from_compact_inputs_supports_attention_only_proxy(self):
        text_to_vis_logits = torch.tensor(
            [
                [
                    [[1.0, 0.0, -1.0], [0.0, 2.0, -2.0]],
                    [[0.5, 1.5, -0.5], [1.0, -1.0, 0.0]],
                ]
            ]
        )
        visual_value_states = torch.tensor(
            [
                [
                    [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                    [[0.5, 1.5], [1.5, 0.5], [0.0, 2.0]],
                ]
            ]
        )

        scores, alpha_mean, deviation_mean = _compute_fes_scores_from_compact_inputs(
            text_to_vis_logits=text_to_vis_logits,
            visual_value_states=visual_value_states,
            use_alpha=True,
            use_deviation=False,
        )

        alpha_per_text = torch.softmax(text_to_vis_logits.float(), dim=-1).mean(dim=1)[0]
        expected_scores = alpha_per_text.pow(2).mean(dim=0).sqrt()

        self.assertTrue(torch.allclose(scores, expected_scores, atol=1e-6))
        self.assertTrue(torch.allclose(alpha_mean, alpha_per_text.mean(dim=0), atol=1e-6))
        self.assertEqual(tuple(deviation_mean.shape), (3,))

    def test_forward_extract_stops_before_full_attention_reconstruction(self):
        class FakeSelfAttention:
            head_dim = 2
            num_key_value_groups = 1
            num_heads = 1
            scaling = 1.0

            @staticmethod
            def q_proj(hidden_states):
                return hidden_states

            @staticmethod
            def k_proj(hidden_states):
                return hidden_states

            @staticmethod
            def v_proj(hidden_states):
                return hidden_states

            @staticmethod
            def q_norm(hidden_states):
                return hidden_states

            @staticmethod
            def k_norm(hidden_states):
                return hidden_states

            @staticmethod
            def o_proj(hidden_states):
                raise AssertionError(
                    "full attention reconstruction should not run"
                )

        class FakeLayer:
            def __init__(self):
                self.self_attn = FakeSelfAttention()
                self.input_layernorm = lambda hidden_states: hidden_states
                self.post_attention_layernorm = (
                    lambda hidden_states: hidden_states
                )
                self.mlp = lambda hidden_states: hidden_states

            def __call__(self, *args, **kwargs):
                raise AssertionError("non-target layer path should not run")

        language_model = SimpleNamespace(
            config=SimpleNamespace(),
            layers=[FakeLayer()],
            rotary_emb=lambda hidden, rope_ids: (
                torch.ones(1, hidden.shape[1], hidden.shape[-1]),
                torch.zeros(1, hidden.shape[1], hidden.shape[-1]),
            ),
        )
        inputs_embeds = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]]]
        )
        position_ids = torch.arange(4).view(1, 1, 4).expand(3, -1, -1)
        cache_position = torch.arange(4)
        text_positions = torch.tensor([2, 3])
        visual_positions = torch.tensor([0, 1])

        with patch(
            "lmms_eval.models.chat.qwen3_vl_ours_v3.create_causal_mask",
            return_value=None,
        ), patch(
            "lmms_eval.models.chat.qwen3_vl_ours_v3.apply_rotary_pos_emb",
            side_effect=lambda query, key, cos, sin: (query, key),
        ):
            text_to_vis_logits, visual_value_states = _forward_extract(
                language_model=language_model,
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                cache_position=cache_position,
                num_layers=1,
                attn_layer=0,
                text_positions=text_positions,
                visual_positions=visual_positions,
            )

        self.assertEqual(tuple(text_to_vis_logits.shape), (1, 1, 2, 2))
        self.assertEqual(tuple(visual_value_states.shape), (1, 1, 2, 2))
