# MMTok: Multimodal Coverage Maximization for Efficient Inference of VLMs
# Paper: https://arxiv.org/abs/2508.18264

"""Greedy maximum-coverage token selection for MMTok."""

import os
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


@torch.jit.script
def greedy_merged_jit_kernel(
    combined: torch.Tensor,
    k_max: int,
    exclude_indices: torch.Tensor,
) -> torch.Tensor:
    total_rows, n = combined.shape
    device = combined.device
    dtype = combined.dtype

    best_combined = torch.zeros(total_rows, device=device, dtype=dtype)
    score_mask = torch.zeros(n, device=device, dtype=dtype)
    neg_inf_tensor = torch.tensor(float("-inf"), dtype=dtype, device=device)

    if exclude_indices.numel() > 0:
        score_mask.index_fill_(0, exclude_indices, neg_inf_tensor.item())

    selected_indices = torch.zeros(k_max, dtype=torch.long, device=device)

    for i in range(k_max):
        delta = (combined - best_combined.unsqueeze(1)).clamp_min_(0).sum(dim=0)
        delta.add_(score_mask)
        best_idx = torch.argmax(delta)
        idx_1d = best_idx.view(1)
        selected_indices[i] = best_idx

        current_col = combined.index_select(1, idx_1d).squeeze(1)
        score_mask.scatter_(0, idx_1d, neg_inf_tensor)
        best_combined = torch.maximum(best_combined, current_col)

    return selected_indices


class SemanticTokenSelector:
    def __init__(
        self,
        target_vision_tokens: int = 32,
        alpha: float = 0.5,
    ):
        self.target_vision_tokens = target_vision_tokens
        self.alpha = alpha

    @staticmethod
    def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    def mm_coverage_selection(
        self,
        text_token_embedding: torch.Tensor,
        vision_tokens: torch.Tensor,
        vision_tokens_clip: torch.Tensor,
        tv_temp: float = 0.01,
        vv_temp: float = 0.2,
        padding_patch_indices: Optional[List[int]] = None,
    ) -> Tuple[List[int], torch.Tensor]:
        device = vision_tokens.device
        x_norm = self._l2_normalize(vision_tokens).float()
        x_clip_norm = self._l2_normalize(vision_tokens_clip).float()
        z_norm = self._l2_normalize(text_token_embedding).float()

        text_to_vision = z_norm @ x_norm.T
        vision_to_vision = x_clip_norm @ x_clip_norm.T
        m, n = text_to_vision.shape
        text_to_vision = F.softmax(text_to_vision * (1.0 / tv_temp), dim=1) / m
        vision_to_vision = F.softmax(vision_to_vision * (1.0 / vv_temp), dim=1) / n
        k_max = min(self.target_vision_tokens, n)
        combined = torch.cat(
            [text_to_vision, vision_to_vision * getattr(self, "alpha", 0.5)],
            dim=0,
        )

        n_threshold = int(os.getenv("MMTok_JIT_N_THRESHOLD", "500"))
        k_max_threshold = int(os.getenv("MMTok_JIT_K_MAX_THRESHOLD", "20"))
        use_jit = (n >= n_threshold) or (k_max >= k_max_threshold)
        if padding_patch_indices is not None:
            exclude_indices = torch.tensor(
                padding_patch_indices,
                device=device,
                dtype=torch.long,
            )
        else:
            exclude_indices = torch.empty(0, dtype=torch.long, device=device)

        if use_jit:
            selected_indices = greedy_merged_jit_kernel(
                combined,
                k_max,
                exclude_indices,
            )
        else:
            best_combined = torch.zeros(m + n, device=device, dtype=torch.float32)
            score_mask = torch.zeros(n, device=device, dtype=torch.float32)
            if padding_patch_indices is not None:
                score_mask[exclude_indices] = float("-inf")
            selected_indices = torch.empty(k_max, dtype=torch.long, device=device)
            neg_inf = float("-inf")
            for i in range(k_max):
                delta = (combined - best_combined.unsqueeze(1)).clamp_min_(0).sum(0)
                delta.add_(score_mask)
                best_idx = torch.argmax(delta)
                selected_indices[i] = best_idx
                torch.maximum(best_combined, combined[:, best_idx], out=best_combined)
                score_mask[best_idx] = neg_inf

        selected_indices, _ = selected_indices.sort()
        selected_tokens = vision_tokens[selected_indices]
        return selected_indices.tolist(), selected_tokens
