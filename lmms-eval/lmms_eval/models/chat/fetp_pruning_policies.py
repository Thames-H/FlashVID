"""Selection policies for FETP/FES visual-token pruning experiments."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


TOPK_POLICY = "topk"
UNIFORM_POLICY = "uniform"
FRAME_AWARE_POLICY = "frame_aware"
ADAPTIVE_TOPK_POLICY = "adaptive_topk"
FRAME_AWARE_ADAPTIVE_POLICY = "frame_aware_adaptive"
TEMPORAL_THEN_FES_POLICY = "temporal_then_fes"


_POLICY_ALIASES = {
    "": TOPK_POLICY,
    "baseline": TOPK_POLICY,
    "fes": TOPK_POLICY,
    "fes_topk": TOPK_POLICY,
    "topk": TOPK_POLICY,
    "top_k": TOPK_POLICY,
    "uniform": UNIFORM_POLICY,
    "uniform_sample": UNIFORM_POLICY,
    "uniform_sampling": UNIFORM_POLICY,
    "uniform_frame": UNIFORM_POLICY,
    "uniform_frames": UNIFORM_POLICY,
    "frame": FRAME_AWARE_POLICY,
    "frame_aware": FRAME_AWARE_POLICY,
    "frame_aware_prune": FRAME_AWARE_POLICY,
    "frame_aware_pruning": FRAME_AWARE_POLICY,
    "adaptive": ADAPTIVE_TOPK_POLICY,
    "adaptive_topk": ADAPTIVE_TOPK_POLICY,
    "adaptive_top_k": ADAPTIVE_TOPK_POLICY,
    "score_gap": ADAPTIVE_TOPK_POLICY,
    "score_gap_adaptive": ADAPTIVE_TOPK_POLICY,
    "frame_adaptive": FRAME_AWARE_ADAPTIVE_POLICY,
    "frame_aware_adaptive": FRAME_AWARE_ADAPTIVE_POLICY,
    "adaptive_frame_aware": FRAME_AWARE_ADAPTIVE_POLICY,
    "temporal": TEMPORAL_THEN_FES_POLICY,
    "temporal_fes": TEMPORAL_THEN_FES_POLICY,
    "temporal_then_fes": TEMPORAL_THEN_FES_POLICY,
}


def normalize_pruning_policy(policy: Optional[str]) -> str:
    key = str(policy or "").strip().lower().replace("-", "_")
    if key in _POLICY_ALIASES:
        return _POLICY_ALIASES[key]
    supported = ", ".join(sorted(set(_POLICY_ALIASES.values())))
    raise ValueError(
        f"Unknown pruning_policy={policy!r}. Supported policies: {supported}"
    )


def infer_tokens_per_frame(
    num_visual_tokens: int,
    *,
    configured_tokens_per_frame: Optional[int] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.Tensor] = None,
) -> Optional[int]:
    """Infer post-vision-encoder visual tokens per frame when possible."""
    if configured_tokens_per_frame is not None:
        configured = int(configured_tokens_per_frame)
        if configured > 0:
            return configured

    num_visual_tokens = int(num_visual_tokens)
    if num_visual_tokens <= 0:
        return None

    if video_grid_thw is not None and video_grid_thw.numel() >= 3:
        grid = video_grid_thw.detach()
        if grid.ndim == 1:
            total_frames = int(grid[0].item())
        else:
            total_frames = int(grid[:, 0].sum().item())
        if total_frames > 0 and num_visual_tokens % total_frames == 0:
            return num_visual_tokens // total_frames

    if pixel_values_videos is not None and pixel_values_videos.ndim >= 5:
        total_frames = int(pixel_values_videos.shape[1])
        if total_frames > 0:
            aligned_tokens = num_visual_tokens
            # LLaVA-OneVision appends one learned newline token after the video.
            if (num_visual_tokens - 1) > 0 and (
                (num_visual_tokens - 1) % total_frames == 0
            ):
                aligned_tokens = num_visual_tokens - 1
            if aligned_tokens % total_frames == 0:
                return aligned_tokens // total_frames

    return None


def _empty_indices(device: torch.device) -> torch.Tensor:
    return torch.empty(0, dtype=torch.long, device=device)


def _safe_num_keep(num_keep: int, num_tokens: int) -> int:
    return max(0, min(int(num_keep), int(num_tokens)))


@torch.no_grad()
def topk_prune(scores: torch.Tensor, num_keep: int) -> torch.Tensor:
    num_keep = _safe_num_keep(num_keep, scores.numel())
    if num_keep == 0:
        return _empty_indices(scores.device)
    if num_keep >= scores.numel():
        return torch.arange(scores.numel(), dtype=torch.long, device=scores.device)
    return scores.float().topk(num_keep).indices.sort().values


@torch.no_grad()
def uniform_prune(
    num_tokens: int,
    num_keep: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    num_tokens = int(num_tokens)
    num_keep = _safe_num_keep(num_keep, num_tokens)
    if num_keep == 0:
        return _empty_indices(device)
    if num_keep >= num_tokens:
        return torch.arange(num_tokens, dtype=torch.long, device=device)
    if num_keep == 1:
        return torch.zeros(1, dtype=torch.long, device=device)

    keep = torch.linspace(
        0,
        num_tokens - 1,
        steps=num_keep,
        device=device,
    ).round().long().unique(sorted=True)
    if keep.numel() < num_keep:
        mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        mask[keep] = False
        fill = torch.arange(num_tokens, dtype=torch.long, device=device)[mask]
        keep = torch.cat([keep, fill[: num_keep - keep.numel()]])
    return keep.sort().values


def _frame_layout(
    scores: torch.Tensor,
    tokens_per_frame: Optional[int],
) -> Tuple[int, int, Optional[int]]:
    if tokens_per_frame is None:
        return 0, 0, None
    tokens_per_frame = int(tokens_per_frame)
    if tokens_per_frame <= 0:
        return 0, 0, None

    n_vis = int(scores.numel())
    n_frames = n_vis // tokens_per_frame
    aligned_tokens = n_frames * tokens_per_frame
    if n_frames <= 0 or aligned_tokens <= 0:
        return 0, 0, None
    return n_frames, aligned_tokens, tokens_per_frame


@torch.no_grad()
def frame_aware_prune(
    scores: torch.Tensor,
    tokens_per_frame: Optional[int],
    num_keep: int,
    *,
    min_keep_per_frame: int = 1,
) -> torch.Tensor:
    n_vis = int(scores.numel())
    num_keep = _safe_num_keep(num_keep, n_vis)
    if num_keep == 0:
        return _empty_indices(scores.device)
    if num_keep >= n_vis:
        return torch.arange(n_vis, dtype=torch.long, device=scores.device)

    n_frames, aligned_tokens, tokens_per_frame = _frame_layout(
        scores,
        tokens_per_frame,
    )
    if tokens_per_frame is None:
        return topk_prune(scores, num_keep)

    frame_scores = scores[:aligned_tokens].float().view(
        n_frames,
        tokens_per_frame,
    )
    frame_offsets = (
        torch.arange(n_frames, device=scores.device).unsqueeze(1)
        * tokens_per_frame
    )

    min_keep_per_frame = max(1, int(min_keep_per_frame))
    per_frame_k = min(min_keep_per_frame, tokens_per_frame)

    if per_frame_k * n_frames <= num_keep:
        _, frame_top = frame_scores.topk(per_frame_k, dim=-1)
        mandatory = (frame_offsets + frame_top).flatten()
    elif num_keep < n_frames:
        frame_indices = uniform_prune(
            n_frames,
            num_keep,
            device=scores.device,
        )
        _, frame_top = frame_scores.index_select(0, frame_indices).topk(
            1,
            dim=-1,
        )
        mandatory = frame_indices * tokens_per_frame + frame_top.flatten()
    else:
        base_k = max(1, num_keep // n_frames)
        base_k = min(base_k, tokens_per_frame)
        _, frame_top = frame_scores.topk(base_k, dim=-1)
        mandatory = (frame_offsets + frame_top).flatten()
        if mandatory.numel() > num_keep:
            local_scores = scores.index_select(0, mandatory).float()
            mandatory = mandatory.index_select(
                0,
                local_scores.topk(num_keep).indices,
            )

    remaining_budget = num_keep - int(mandatory.numel())
    if remaining_budget > 0:
        mask = torch.ones(n_vis, dtype=torch.bool, device=scores.device)
        mask[mandatory] = False
        candidate_scores = scores.float().clone()
        candidate_scores[~mask] = -float("inf")
        extra = candidate_scores.topk(
            min(remaining_budget, int(mask.sum().item()))
        ).indices
        keep = torch.cat([mandatory, extra])
    else:
        keep = mandatory

    return keep.unique().sort().values[:num_keep]


@torch.no_grad()
def adaptive_topk_prune(
    scores: torch.Tensor,
    max_keep: int,
    *,
    min_keep: int = 1,
    gap_percentile: float = 0.8,
) -> torch.Tensor:
    n_vis = int(scores.numel())
    max_keep = _safe_num_keep(max_keep, n_vis)
    if max_keep == 0:
        return _empty_indices(scores.device)
    if max_keep >= n_vis:
        max_keep = n_vis

    min_keep = max(1, min(int(min_keep), max_keep))
    sorted_scores, sorted_indices = scores.float().sort(descending=True)
    sorted_scores = sorted_scores[:max_keep]

    if sorted_scores.numel() <= 2:
        return sorted_indices[:max_keep].sort().values

    gaps = sorted_scores[:-1] - sorted_scores[1:]
    positive_gaps = gaps[gaps > 0]
    if positive_gaps.numel() == 0:
        cutoff = max_keep
    else:
        percentile = min(max(float(gap_percentile), 0.0), 1.0)
        threshold = positive_gaps.quantile(percentile)
        large_gap_positions = torch.where(gaps >= threshold)[0]
        cutoff = (
            int(large_gap_positions[0].item()) + 1
            if large_gap_positions.numel() > 0
            else max_keep
        )
        cutoff = max(min_keep, min(cutoff, max_keep))

    return sorted_indices[:cutoff].sort().values


@torch.no_grad()
def temporal_then_fes_prune(
    scores: torch.Tensor,
    tokens_per_frame: Optional[int],
    num_keep: int,
    *,
    temporal_ratio: float = 0.5,
) -> torch.Tensor:
    n_vis = int(scores.numel())
    num_keep = _safe_num_keep(num_keep, n_vis)
    if num_keep == 0:
        return _empty_indices(scores.device)
    if num_keep >= n_vis:
        return torch.arange(n_vis, dtype=torch.long, device=scores.device)

    n_frames, aligned_tokens, tokens_per_frame = _frame_layout(
        scores,
        tokens_per_frame,
    )
    if tokens_per_frame is None:
        return topk_prune(scores, num_keep)

    temporal_budget = int(num_keep * min(max(float(temporal_ratio), 0.0), 1.0))
    temporal_budget = max(1, min(num_keep, temporal_budget))
    n_keep_frames = max(1, min(n_frames, temporal_budget // tokens_per_frame))
    frame_indices = torch.linspace(
        0,
        n_frames - 1,
        steps=n_keep_frames,
        device=scores.device,
    ).long().unique(sorted=True)
    if frame_indices.numel() < n_keep_frames:
        mask = torch.ones(n_frames, dtype=torch.bool, device=scores.device)
        mask[frame_indices] = False
        fill = torch.arange(n_frames, dtype=torch.long, device=scores.device)[mask]
        frame_indices = torch.cat(
            [frame_indices, fill[: n_keep_frames - frame_indices.numel()]]
        ).sort().values

    frame_scores = scores[:aligned_tokens].float().view(
        n_frames,
        tokens_per_frame,
    )
    per_frame_budget = max(1, temporal_budget // int(frame_indices.numel()))
    temporal_keeps = []
    for frame_idx in frame_indices:
        _, top_in_frame = frame_scores[frame_idx].topk(
            min(per_frame_budget, tokens_per_frame)
        )
        temporal_keeps.append(frame_idx * tokens_per_frame + top_in_frame)

    temporal_keep = torch.cat(temporal_keeps).unique()
    if temporal_keep.numel() > num_keep:
        local_scores = scores.index_select(0, temporal_keep).float()
        temporal_keep = temporal_keep.index_select(
            0,
            local_scores.topk(num_keep).indices,
        )

    remaining = num_keep - int(temporal_keep.numel())
    if remaining > 0:
        mask = torch.ones(n_vis, dtype=torch.bool, device=scores.device)
        mask[temporal_keep] = False
        candidate_scores = scores.float().clone()
        candidate_scores[~mask] = -float("inf")
        extra = candidate_scores.topk(
            min(remaining, int(mask.sum().item()))
        ).indices
        keep = torch.cat([temporal_keep, extra])
    else:
        keep = temporal_keep

    return keep.unique().sort().values[:num_keep]


@torch.no_grad()
def select_pruning_indices(
    scores: torch.Tensor,
    num_keep: int,
    *,
    pruning_policy: Optional[str] = TOPK_POLICY,
    tokens_per_frame: Optional[int] = None,
    min_keep_per_frame: int = 1,
    gap_percentile: float = 0.8,
    temporal_ratio: float = 0.5,
    adaptive_min_keep: int = 1,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    policy = normalize_pruning_policy(pruning_policy)
    max_keep = _safe_num_keep(num_keep, scores.numel())

    if policy == TOPK_POLICY:
        keep = topk_prune(scores, max_keep)
    elif policy == UNIFORM_POLICY:
        keep = uniform_prune(
            scores.numel(),
            max_keep,
            device=scores.device,
        )
    elif policy == FRAME_AWARE_POLICY:
        keep = frame_aware_prune(
            scores,
            tokens_per_frame,
            max_keep,
            min_keep_per_frame=min_keep_per_frame,
        )
    elif policy == ADAPTIVE_TOPK_POLICY:
        keep = adaptive_topk_prune(
            scores,
            max_keep,
            min_keep=adaptive_min_keep,
            gap_percentile=gap_percentile,
        )
    elif policy == FRAME_AWARE_ADAPTIVE_POLICY:
        adaptive_keep = adaptive_topk_prune(
            scores,
            max_keep,
            min_keep=adaptive_min_keep,
            gap_percentile=gap_percentile,
        )
        keep = frame_aware_prune(
            scores,
            tokens_per_frame,
            int(adaptive_keep.numel()),
            min_keep_per_frame=min_keep_per_frame,
        )
    elif policy == TEMPORAL_THEN_FES_POLICY:
        keep = temporal_then_fes_prune(
            scores,
            tokens_per_frame,
            max_keep,
            temporal_ratio=temporal_ratio,
        )
    else:
        raise AssertionError(f"Unhandled pruning policy: {policy}")

    n_frames, aligned_tokens, resolved_tpf = _frame_layout(
        scores,
        tokens_per_frame,
    )
    stats: Dict[str, object] = {
        "pruning_policy": policy,
        "pruning_max_keep_budget": int(max_keep),
        "pruning_effective_keep": int(keep.numel()),
        "pruning_tokens_per_frame": int(resolved_tpf or 0),
        "pruning_num_frames": int(n_frames),
        "pruning_aligned_visual_tokens": int(aligned_tokens),
        "pruning_min_keep_per_frame": int(min_keep_per_frame),
        "pruning_gap_percentile": float(gap_percentile),
        "pruning_temporal_ratio": float(temporal_ratio),
    }
    return keep, stats
