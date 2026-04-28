from unittest import TestCase

import torch

from lmms_eval.models.chat.fetp_pruning_policies import (
    adaptive_topk_prune,
    frame_aware_prune,
    infer_tokens_per_frame,
    select_pruning_indices,
    uniform_prune,
)


class TestFETPPruningPolicies(TestCase):
    def test_frame_aware_prune_keeps_each_frame_when_budget_allows(self):
        scores = torch.tensor(
            [
                0.1,
                0.2,
                9.0,
                0.3,
                8.0,
                7.0,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
            ]
        )

        keep = frame_aware_prune(
            scores,
            tokens_per_frame=4,
            num_keep=5,
            min_keep_per_frame=1,
        )

        kept_frames = (keep // 4).unique().tolist()
        self.assertEqual(kept_frames, [0, 1, 2])
        self.assertEqual(int(keep.numel()), 5)

    def test_adaptive_topk_cuts_at_first_large_score_gap(self):
        scores = torch.tensor([10.0, 9.0, 8.0, 1.0, 0.9])

        keep = adaptive_topk_prune(
            scores,
            max_keep=5,
            min_keep=1,
            gap_percentile=0.8,
        )

        self.assertEqual(keep.tolist(), [0, 1, 2])

    def test_uniform_prune_evenly_spreads_indices(self):
        keep = uniform_prune(10, 4, device=torch.device("cpu"))

        self.assertEqual(keep.tolist(), [0, 3, 6, 9])

    def test_select_frame_adaptive_uses_adaptive_budget_then_frame_constraint(self):
        scores = torch.tensor(
            [
                10.0,
                9.0,
                8.0,
                7.0,
                1.0,
                0.9,
                0.8,
                0.7,
            ]
        )

        keep, stats = select_pruning_indices(
            scores,
            num_keep=8,
            pruning_policy="frame_aware_adaptive",
            tokens_per_frame=4,
            min_keep_per_frame=1,
            gap_percentile=0.8,
        )

        self.assertLess(int(keep.numel()), 8)
        self.assertEqual(stats["pruning_policy"], "frame_aware_adaptive")
        self.assertEqual(stats["pruning_num_frames"], 2)

    def test_infer_tokens_per_frame_from_qwen_grid(self):
        tokens_per_frame = infer_tokens_per_frame(
            128,
            video_grid_thw=torch.tensor([[8, 8, 8]]),
        )

        self.assertEqual(tokens_per_frame, 16)

    def test_infer_tokens_per_frame_handles_llava_video_newline(self):
        pixel_values_videos = torch.zeros(1, 4, 3, 336, 336)

        tokens_per_frame = infer_tokens_per_frame(
            785,
            pixel_values_videos=pixel_values_videos,
        )

        self.assertEqual(tokens_per_frame, 196)
