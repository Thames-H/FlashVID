"""Query-Aware Anchor Propagation for Qwen2.5-VL.

Implements a 5-stage pipeline: Vision Encoding → Video Partition → Anchor
Frame Token Selection (query-aware) → Non-anchor Frame Propagation →
Aggregation & LLM Inference.

The core idea is to use the user query to guide token selection via shallow
LLM forward passes, then propagate anchor selections to non-anchor frames
through lightweight feature matching.
"""

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLModel,
    Qwen2_5_VLModelOutputWithPast,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, is_torchdynamo_compiling

try:
    import decord
except ImportError:
    decord = None

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)
from lmms_eval.models.simple.qwen2_5_vl import Qwen2_5_VL as Qwen2_5_VLSimple
from lmms_eval.protocol import ChatMessages

process_vision_info, _has_qwen_vl = optional_import(
    "qwen_vl_utils", "process_vision_info"
)
if not _has_qwen_vl:
    eval_logger.warning(
        "Failed to import qwen_vl_utils; "
        "Please install it via `pip install qwen-vl-utils`"
    )


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AnchorInfo:
    """Information stored per local anchor for propagation."""

    feature: torch.Tensor  # [d] original feature (immutable)
    grid_position: Tuple[int, int]  # (row, col)
    relevance_score: float
    neighbor_radius: int
    prev_match_feature: torch.Tensor  # [d] updated per-frame


# ---------------------------------------------------------------------------
# Stage 1: Video Partition (DySeg)
# ---------------------------------------------------------------------------

def _dyseg(
    frame_embeds: torch.Tensor,
    segment_threshold: float,
    min_segment_num: int,
) -> torch.Tensor:
    """Segment video into coherent groups based on frame-level similarity.

    Re-implements the DySeg logic from ``flashvid/utils.py:segment()``.

    Args:
        frame_embeds: [F, d] per-frame feature embeddings (e.g. from
            pixel-level GAP or LLM embeddings).
        segment_threshold: Cosine similarity threshold for scene cuts.
        min_segment_num: Minimum number of segments to produce.

    Returns:
        segment_lengths: 1-D tensor of per-segment frame counts.
    """
    num_frames = frame_embeds.shape[0]

    normed = frame_embeds / frame_embeds.norm(p=2, dim=-1, keepdim=True)
    transition_sims = (normed[:-1] * normed[1:]).sum(dim=-1)  # [F-1]

    cut_indices = torch.where(transition_sims < segment_threshold)[0]

    # Ensure at least min_segment_num segments via complementary cuts.
    num_segments = cut_indices.numel() + 1
    if num_segments < min_segment_num:
        remaining = min_segment_num - num_segments
        sims_copy = transition_sims.clone()
        sims_copy[sims_copy < segment_threshold] = 1.0
        extra = torch.topk(
            sims_copy,
            k=min(remaining, sims_copy.shape[0]),
            largest=False,
        ).indices
        cut_indices = torch.cat([cut_indices, extra]).sort().values

    padded = F.pad(cut_indices, (1, 1), value=0)
    padded[0] = -1
    padded[-1] = num_frames - 1
    segment_lengths = torch.diff(padded, n=1, dim=0)
    return segment_lengths


# ---------------------------------------------------------------------------
# Stage 2: Anchor frame selection helpers
# ---------------------------------------------------------------------------

def _select_anchor_frame(segment_features: torch.Tensor) -> int:
    """Select the most representative frame in a segment.

    The anchor frame is the one with the highest average cosine similarity
    to all other frames in the segment.

    Args:
        segment_features: [S, N_v, d] features for frames in a segment.

    Returns:
        Index of the anchor frame within the segment.
    """
    num_frames = segment_features.shape[0]
    if num_frames == 1:
        return 0
    # GAP → [S, d]
    frame_embeds = segment_features.mean(dim=1)
    normed = frame_embeds / frame_embeds.norm(p=2, dim=-1, keepdim=True)
    sim_matrix = normed @ normed.T  # [S, S]
    # Average similarity to other frames (exclude self).
    sim_matrix.fill_diagonal_(0.0)
    avg_sim = sim_matrix.sum(dim=1) / (num_frames - 1)
    return avg_sim.argmax().item()


def _get_neighbor_radius(retention_ratio: float) -> int:
    """Determine spatial neighbor radius based on retention ratio."""
    if retention_ratio >= 0.25:
        return 2
    if retention_ratio >= 0.10:
        return 1
    return 0


def _get_spatial_neighbors(
    idx: int,
    grid_h: int,
    grid_w: int,
    radius: int,
) -> List[int]:
    """Return indices of spatial neighbors (including self) on a grid."""
    row, col = divmod(idx, grid_w)
    neighbors = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            nr, nc = row + dr, col + dc
            if 0 <= nr < grid_h and 0 <= nc < grid_w:
                neighbors.append(nr * grid_w + nc)
    return neighbors


# ---------------------------------------------------------------------------
# Stage 2b: Shallow LLM forward + cross-attention extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def _shallow_forward_and_extract(
    language_model,
    anchor_visual_embeds: torch.Tensor,
    full_inputs_embeds: torch.Tensor,
    full_position_ids: torch.Tensor,
    visual_start_index: int,
    visual_end_index: int,
    shallow_layers: int,
) -> torch.Tensor:
    """Run first K LLM layers and extract text→visual relevance scores.

    Instead of concatenating raw [visual; text] tokens, this function
    preserves the full chat template structure by replacing the multi-frame
    video region with a single anchor frame's tokens.

    Args:
        language_model: The LLM backbone (``self.language_model``).
        anchor_visual_embeds: [N_v, d] anchor frame visual embeddings.
        full_inputs_embeds: [1, total_seq, d] full input embeddings with
            template tokens (system prompt, special tokens, etc.).
        full_position_ids: [3 or 4, 1, total_seq] position ids for the
            full sequence.
        visual_start_index: Start index of video tokens in the sequence.
        visual_end_index: End index of video tokens in the sequence.
        shallow_layers: Number of layers K to run.

    Returns:
        relevance: [N_v] per-visual-token query-relevance score.
    """
    n_vis = anchor_visual_embeds.shape[0]

    # Build shallow input: [prefix | anchor_frame_tokens | suffix]
    # This preserves the chat template (system prompt, <|vision_start|>,
    # <|vision_end|>, user query, etc.).
    prefix_embeds = full_inputs_embeds[:, :visual_start_index]
    suffix_embeds = full_inputs_embeds[:, visual_end_index:]
    hidden = torch.cat(
        [prefix_embeds, anchor_visual_embeds.unsqueeze(0), suffix_embeds],
        dim=1,
    )  # [1, prefix + N_v + suffix, d]

    # Build corresponding position ids.
    prefix_pos = full_position_ids[:, :, :visual_start_index]
    suffix_pos = full_position_ids[:, :, visual_end_index:]
    # For anchor visual tokens, use contiguous positions starting from
    # the visual start position.
    anchor_pos_start = full_position_ids[:, :, visual_start_index:visual_start_index + 1]
    anchor_vis_pos = anchor_pos_start + torch.arange(
        n_vis, device=anchor_visual_embeds.device
    ).view(1, 1, -1)
    pos_ids = torch.cat([prefix_pos, anchor_vis_pos, suffix_pos], dim=2)

    seq_len = hidden.shape[1]
    vis_start = visual_start_index
    vis_end = visual_start_index + n_vis

    # pos_ids may be [4, batch, seq] where dim-0 = [text_pos, t, h, w].
    # rotary_emb expects only the 3 multimodal dims (t, h, w).
    if pos_ids.shape[0] == 4:
        rope_pos = pos_ids[1:]  # [3, 1, seq_len]
    else:
        rope_pos = pos_ids  # [3, 1, seq_len]

    # Compute position embeddings (RoPE).
    position_embeddings = language_model.rotary_emb(hidden, rope_pos)

    # Forward through first K layers.
    attn_weights = None
    for layer_idx in range(shallow_layers):
        is_last = layer_idx == shallow_layers - 1
        layer = language_model.layers[layer_idx]
        layer_out = layer(
            hidden,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            output_attentions=is_last,
            use_cache=False,
            cache_position=torch.arange(seq_len, device=hidden.device),
            position_embeddings=position_embeddings,
        )
        hidden = layer_out[0]
        if is_last:
            attn_weights = layer_out[1]  # [1, n_heads, seq, seq]

    if attn_weights is None:
        # Fallback: uniform relevance.
        return torch.ones(n_vis, device=anchor_visual_embeds.device)

    # attn_weights: [1, n_heads, seq_len, seq_len]
    # Extract text→visual attention.
    # Text tokens = everything NOT in [vis_start, vis_end).
    # We average attention from suffix text tokens (the user query,
    # after <|vision_end|>) to the visual tokens.
    suffix_start = vis_end
    text_to_visual = attn_weights[
        0, :, suffix_start:, vis_start:vis_end
    ]  # [n_heads, N_suffix, N_v]
    # Average over heads and text tokens → [N_v].
    relevance = text_to_visual.mean(dim=0).mean(dim=0).float()
    return relevance


# ---------------------------------------------------------------------------
# Stage 2c: MMDP global exploration
# ---------------------------------------------------------------------------

def _mmdp_select(
    features: torch.Tensor,
    anchor_indices: torch.Tensor,
    num_select: int,
    all_features: torch.Tensor,
) -> torch.Tensor:
    """Max-Min Distance Problem greedy selection for global exploration.

    Args:
        features: [N_v, d] all frame features.
        anchor_indices: 1-D tensor of already-selected indices.
        num_select: Number of exploration tokens to select.
        all_features: same as features (included for clarity).

    Returns:
        selected: 1-D tensor of additionally selected indices.
    """
    if num_select <= 0:
        return torch.tensor([], dtype=torch.long, device=features.device)

    n_total = features.shape[0]
    device = features.device

    # Build candidate mask.
    is_anchor = torch.zeros(n_total, dtype=torch.bool, device=device)
    if anchor_indices.numel() > 0:
        is_anchor[anchor_indices] = True
    candidates = torch.where(~is_anchor)[0]

    if candidates.numel() == 0 or num_select <= 0:
        return torch.tensor([], dtype=torch.long, device=device)

    num_select = min(num_select, candidates.numel())

    # Normalise for cosine distance.
    normed = features / features.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)

    # Compute initial min-distance from each candidate to nearest anchor.
    if anchor_indices.numel() > 0:
        anchor_feats = normed[anchor_indices]  # [A, d]
        cand_feats = normed[candidates]  # [C, d]
        sim = cand_feats @ anchor_feats.T  # [C, A]
        min_dist = (1.0 - sim).min(dim=1).values  # [C]
    else:
        min_dist = torch.ones(candidates.numel(), device=device)

    selected = []
    for _ in range(num_select):
        best_idx = min_dist.argmax().item()
        selected.append(candidates[best_idx].item())
        # Update min_dist.
        new_feat = normed[candidates[best_idx]]  # [d]
        new_sim = (normed[candidates] * new_feat).sum(dim=-1)  # [C]
        new_dist = 1.0 - new_sim
        min_dist = torch.min(min_dist, new_dist)
        min_dist[best_idx] = -float("inf")  # Mark as selected.

    return torch.tensor(selected, dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# Core patched forward
# ---------------------------------------------------------------------------

def _make_ours_forward(
    original_forward,
    retention_ratio: float,
    shallow_layers: int,
    alpha: float,
    segment_threshold: float,
    min_segment_num: int,
    T_match: float,
    lambda_min: float,
    r_max: float,
    r_explore_min: float,
    use_pixel_segment: bool,
):
    """Create a patched Qwen2_5_VLModel.forward with query-aware anchor
    propagation for video token compression."""

    def patched_forward(
        self: Qwen2_5_VLModel,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen2_5_VLModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_embeds = self.get_image_features(
                pixel_values, image_grid_thw
            )
            image_embeds = torch.cat(image_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask, _ = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask, image_embeds
            )

        n_video_tokens = None
        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(
                pixel_values_videos, video_grid_thw
            )
            video_embeds = torch.cat(video_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            _, video_mask = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                video_features=video_embeds,
            )
            n_video_tokens = video_embeds.shape[0]
            inputs_embeds = inputs_embeds.masked_scatter(
                video_mask, video_embeds
            )

        if position_ids is None:
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (
                    inputs_embeds is not None
                    and inputs_embeds.shape[1] != 1
                )
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (
                    past_key_values is None
                    or past_key_values.get_seq_length() == 0
                )
            )
            if (
                prefill_compiled_stage or prefill_noncompiled_stage
            ) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                position_ids = torch.arange(
                    seq_length, device=inputs_embeds.device
                )
                position_ids = position_ids.view(1, 1, -1).expand(
                    3, batch_size, -1
                )
                if cache_position is not None:
                    delta = (cache_position[0] + self.rope_deltas).to(
                        inputs_embeds.device
                    )
                else:
                    delta = torch.zeros(
                        (batch_size, seq_length),
                        device=inputs_embeds.device,
                    )
                delta = delta.repeat_interleave(
                    batch_size // delta.shape[0], dim=1
                )
                position_ids = position_ids + delta.to(position_ids.device)

        # ---------------------------------------------------------------
        # Query-Aware Anchor Propagation: prefill + video tokens present
        # Only applied for single-sample batches (batch_size=1) since
        # the per-frame compression logic assumes a single video.
        # ---------------------------------------------------------------
        if (
            n_video_tokens is not None
            and position_ids.shape[-1] > 1
            and retention_ratio < 1.0
            and inputs_embeds.shape[0] == 1
            and video_grid_thw.shape[0] == 1
        ):
            device = inputs_embeds.device
            dtype = inputs_embeds.dtype
            bsz, total_seq, hidden_size = inputs_embeds.shape

            # -- Locate video tokens in the sequence --
            visual_start_index = (
                torch.where(
                    input_ids[0] == self.config.video_token_id
                )[0][0].item()
            )
            visual_end_index = visual_start_index + n_video_tokens

            # -- Infer per-frame grid dimensions from video_grid_thw --
            # video_grid_thw: [num_videos, 3] = (T, H_pre, W_pre) where
            # H_pre/W_pre are BEFORE spatial merge.  The vision encoder
            # merges spatial_merge_size x spatial_merge_size patches into
            # one token, so actual tokens per frame = H_pre*W_pre / merge^2.
            num_frames = video_grid_thw[0, 0].item()
            grid_h_pre = video_grid_thw[0, 1].item()
            grid_w_pre = video_grid_thw[0, 2].item()
            n_vis_per_frame = n_video_tokens // num_frames
            # Derive merged (post-merge) grid dimensions for spatial
            # neighbor calculations.
            merge_size_sq = (grid_h_pre * grid_w_pre) // n_vis_per_frame
            merge_size = int(math.sqrt(merge_size_sq))
            grid_h = grid_h_pre // merge_size
            grid_w = grid_w_pre // merge_size

            # -- Extract video features [F, N_v, d] --
            video_features = inputs_embeds[
                0, visual_start_index:visual_end_index
            ].view(num_frames, n_vis_per_frame, hidden_size)

            # -- Per-frame budget --
            budget_per_frame = max(1, int(n_vis_per_frame * retention_ratio))
            neighbor_radius = _get_neighbor_radius(retention_ratio)
            b_local_max = int(budget_per_frame * r_max)
            m_global_min = max(3, int(budget_per_frame * r_explore_min))

            # ==========================================================
            # Stage 1: Video Partition (DySeg)
            # ==========================================================
            if use_pixel_segment and pixel_values_videos is not None:
                # Use pixel-level features for segmentation:
                # more sensitive to visual scene changes.
                num_pixels = pixel_values_videos.shape[0]
                pixels_per_frame = num_pixels // num_frames
                pixel_frame_embeds = pixel_values_videos.view(
                    num_frames, pixels_per_frame, -1
                ).mean(dim=1).float()  # [F, pixel_dim]
                segment_lengths = _dyseg(
                    pixel_frame_embeds, segment_threshold, min_segment_num
                )
            else:
                # Fallback: use LLM embeddings (post-projector).
                llm_frame_embeds = video_features.mean(dim=1)  # [F, d]
                segment_lengths = _dyseg(
                    llm_frame_embeds, segment_threshold, min_segment_num
                )

            # ==========================================================
            # Stages 2-3: Per-segment anchor selection + propagation
            # ==========================================================
            all_frame_keep_local = [None] * num_frames  # per-frame local indices

            offset = 0
            for seg_idx in range(segment_lengths.shape[0]):
                seg_len = segment_lengths[seg_idx].item()
                seg_frame_indices = list(range(offset, offset + seg_len))
                seg_features = video_features[offset: offset + seg_len]

                # -- Stage 2a: Select anchor frame --
                anchor_local = _select_anchor_frame(seg_features)
                anchor_global = offset + anchor_local
                anchor_frame_features = video_features[anchor_global]  # [N_v, d]

                # -- Stage 2b: Shallow LLM forward for relevance --
                relevance = _shallow_forward_and_extract(
                    self.language_model,
                    anchor_frame_features,
                    inputs_embeds,
                    position_ids,
                    visual_start_index,
                    visual_end_index,
                    shallow_layers,
                )  # [N_v]

                # -- Stage 2b: Adaptive threshold --
                mu_r = relevance.mean()
                sigma_r = relevance.std()
                threshold = mu_r + alpha * sigma_r
                core_anchor_mask = relevance > threshold
                core_anchor_indices = torch.where(core_anchor_mask)[0]

                # -- Stage 2b: Neighborhood expansion --
                anchor_with_neighbors = set()
                anchor_infos: List[AnchorInfo] = []
                for idx_tensor in core_anchor_indices:
                    idx = idx_tensor.item()
                    neighbors = _get_spatial_neighbors(
                        idx, grid_h, grid_w, neighbor_radius
                    )
                    anchor_with_neighbors.update(neighbors)
                    anchor_infos.append(
                        AnchorInfo(
                            feature=anchor_frame_features[idx].clone(),
                            grid_position=divmod(idx, grid_w),
                            relevance_score=relevance[idx].item(),
                            neighbor_radius=neighbor_radius,
                            prev_match_feature=anchor_frame_features[idx].clone(),
                        )
                    )

                # -- Stage 2d: Budget allocation --
                b_local_raw = len(anchor_with_neighbors)

                if b_local_raw > b_local_max:
                    # Truncate by relevance: keep top anchors until budget met.
                    sorted_anchors = sorted(
                        zip(core_anchor_indices.tolist(), anchor_infos),
                        key=lambda x: x[1].relevance_score,
                        reverse=True,
                    )
                    anchor_with_neighbors = set()
                    kept_anchor_infos = []
                    for a_idx, a_info in sorted_anchors:
                        neighbors = _get_spatial_neighbors(
                            a_idx, grid_h, grid_w, neighbor_radius
                        )
                        candidate = anchor_with_neighbors | set(neighbors)
                        if len(candidate) > b_local_max:
                            break
                        anchor_with_neighbors = candidate
                        kept_anchor_infos.append(a_info)
                    anchor_infos = kept_anchor_infos
                    b_local_raw = len(anchor_with_neighbors)

                b_explore = max(
                    m_global_min, budget_per_frame - b_local_raw
                )
                # Hard budget constraint.
                if b_local_raw + b_explore > budget_per_frame:
                    b_explore = budget_per_frame - b_local_raw
                    b_explore = max(0, b_explore)

                # -- Stage 2c: MMDP global exploration --
                anchor_idx_tensor = torch.tensor(
                    sorted(anchor_with_neighbors),
                    dtype=torch.long,
                    device=device,
                )
                explore_idx = _mmdp_select(
                    anchor_frame_features,
                    anchor_idx_tensor,
                    b_explore,
                    anchor_frame_features,
                )

                # -- Anchor frame final kept tokens --
                anchor_keep = set(anchor_with_neighbors)
                anchor_keep.update(explore_idx.tolist())
                all_frame_keep_local[anchor_global] = sorted(anchor_keep)

                # =======================================================
                # Stage 3: Propagation to non-anchor frames
                # =======================================================
                # Propagate forward then backward from anchor.
                def _propagate_direction(frame_order: List[int]):
                    # Reset prev_match_feature at the start of each direction.
                    for a_info in anchor_infos:
                        a_info.prev_match_feature = a_info.feature.clone()

                    for f_idx in frame_order:
                        if f_idx == anchor_global:
                            continue
                        frame_feats = video_features[f_idx]  # [N_v, d]

                        # Dynamic lambda based on temporal distance.
                        dist = abs(f_idx - anchor_global)
                        lam = max(
                            lambda_min, 1.0 - dist / max(seg_len, 1)
                        )

                        # Collect per-anchor match results, sorted by
                        # relevance so we can truncate by importance.
                        match_results = []  # (relevance, best_idx, a_info)
                        for a_info in anchor_infos:
                            normed_anchor = a_info.feature / a_info.feature.norm().clamp(min=1e-8)
                            normed_prev = a_info.prev_match_feature / a_info.prev_match_feature.norm().clamp(min=1e-8)
                            normed_frame = frame_feats / frame_feats.norm(
                                dim=-1, keepdim=True
                            ).clamp(min=1e-8)

                            score = (
                                lam * (normed_frame @ normed_anchor)
                                + (1.0 - lam) * (normed_frame @ normed_prev)
                            )  # [N_v]

                            best_score, best_idx = score.max(dim=0)
                            if best_score.item() >= T_match:
                                match_results.append(
                                    (a_info.relevance_score, best_idx.item(), a_info)
                                )

                        # Sort by relevance (high → low) and greedily add
                        # neighborhoods, respecting b_local_max budget.
                        match_results.sort(key=lambda x: x[0], reverse=True)
                        matched_indices = set()
                        for rel, best_idx_val, a_info in match_results:
                            neighbors = _get_spatial_neighbors(
                                best_idx_val,
                                grid_h,
                                grid_w,
                                neighbor_radius,
                            )
                            candidate = matched_indices | set(neighbors)
                            if len(candidate) > b_local_max:
                                break
                            matched_indices = candidate
                            # Update prev_match_feature.
                            a_info.prev_match_feature = (
                                frame_feats[best_idx_val].clone()
                            )

                        # Budget control (same logic as anchor frame).
                        b_matched = len(matched_indices)
                        b_explore_prop = max(
                            m_global_min, budget_per_frame - b_matched
                        )
                        if b_matched + b_explore_prop > budget_per_frame:
                            b_explore_prop = budget_per_frame - b_matched
                            b_explore_prop = max(0, b_explore_prop)

                        # MMDP exploration for this frame.
                        matched_tensor = torch.tensor(
                            sorted(matched_indices),
                            dtype=torch.long,
                            device=device,
                        )
                        explore = _mmdp_select(
                            frame_feats,
                            matched_tensor,
                            b_explore_prop,
                            frame_feats,
                        )

                        keep = set(matched_indices)
                        keep.update(explore.tolist())
                        all_frame_keep_local[f_idx] = sorted(keep)

                # Forward propagation: anchor+1, anchor+2, ...
                forward_order = list(
                    range(offset, offset + seg_len)
                )
                # Backward propagation: anchor-1, anchor-2, ...
                backward_order = list(
                    range(offset + seg_len - 1, offset - 1, -1)
                )

                _propagate_direction(forward_order)
                _propagate_direction(backward_order)

                offset += seg_len

            # ==========================================================
            # Stage 4: Build keep_global_indices and filter
            # ==========================================================
            keep_visual_local_list = []
            for f_idx in range(num_frames):
                frame_local = all_frame_keep_local[f_idx]
                if frame_local is None:
                    # Shouldn't happen, but fallback: keep first budget tokens.
                    frame_local = list(range(
                        min(budget_per_frame, n_vis_per_frame)
                    ))
                # Safety truncation (budget should already be enforced above,
                # but guard against edge cases).
                if len(frame_local) > budget_per_frame:
                    frame_local = frame_local[:budget_per_frame]
                frame_global_offset = f_idx * n_vis_per_frame
                keep_visual_local_list.extend(
                    [frame_global_offset + i for i in frame_local]
                )

            keep_visual_local = torch.tensor(
                keep_visual_local_list, dtype=torch.long, device=device
            ).sort().values

            # Build global keep indices: prefix + kept video + suffix.
            global_indices = torch.arange(total_seq, device=device)
            keep_global_indices = torch.cat(
                [
                    global_indices[:visual_start_index],
                    global_indices[visual_start_index:visual_end_index][
                        keep_visual_local
                    ],
                    global_indices[visual_end_index:],
                ],
                dim=0,
            )

            inputs_embeds = torch.gather(
                inputs_embeds,
                dim=1,
                index=keep_global_indices.view(1, -1, 1).expand(
                    bsz, -1, hidden_size
                ),
            )
            position_ids = position_ids[:, :, keep_global_indices]
            attention_mask = attention_mask[:, keep_global_indices]
            cache_position = cache_position[keep_global_indices]

        # ---------------------------------------------------------------
        # Language model forward
        # ---------------------------------------------------------------
        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        output = Qwen2_5_VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
        return output if return_dict else output.to_tuple()

    return patched_forward


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------


@register_model("qwen2_5_vl_ours")
class Qwen2_5_VL_Ours(Qwen2_5_VLSimple):
    """Qwen2.5-VL with Query-Aware Anchor Propagation.

    Implements a 5-stage pipeline that uses the user query to guide video
    token selection via shallow LLM attention, then propagates anchor
    selections to non-anchor frames through feature matching.
    """

    is_simple = False

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        attn_implementation: Optional[str] = None,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,
        max_image_size: Optional[int] = None,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        # Query-aware anchor propagation parameters.
        retention_ratio: float = 0.25,
        shallow_layers: int = 2,
        alpha: float = 1.0,
        segment_threshold: float = 0.9,
        min_segment_num: int = 8,
        T_match: float = 0.7,
        lambda_min: float = 0.4,
        r_max: float = 0.7,
        r_explore_min: float = 0.15,
        use_pixel_segment: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            pretrained=pretrained,
            device=device,
            device_map=device_map,
            batch_size=batch_size,
            use_cache=use_cache,
            attn_implementation=attn_implementation,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            max_num_frames=max_num_frames,
            use_custom_video_loader=use_custom_video_loader,
            fps=fps,
            max_image_size=max_image_size,
            system_prompt=system_prompt,
            interleave_visuals=interleave_visuals,
            reasoning_prompt=reasoning_prompt,
            enable_flashvid=False,
        )

        self.retention_ratio = retention_ratio
        eval_logger.info(
            f"[Qwen2_5_VL_Ours] retention_ratio={retention_ratio}, "
            f"shallow_layers={shallow_layers}, alpha={alpha}, "
            f"segment_threshold={segment_threshold}, "
            f"min_segment_num={min_segment_num}, T_match={T_match}, "
            f"lambda_min={lambda_min}, r_max={r_max}, "
            f"r_explore_min={r_explore_min}, "
            f"use_pixel_segment={use_pixel_segment}"
        )

        # Monkey-patch the model's forward.
        original_forward = Qwen2_5_VLModel.forward
        Qwen2_5_VLModel.forward = _make_ours_forward(
            original_forward,
            retention_ratio=retention_ratio,
            shallow_layers=shallow_layers,
            alpha=alpha,
            segment_threshold=segment_threshold,
            min_segment_num=min_segment_num,
            T_match=T_match,
            lambda_min=lambda_min,
            r_max=r_max,
            r_explore_min=r_explore_min,
            use_pixel_segment=use_pixel_segment,
        )

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            return x[0], x[0]

        re_ords = utils.Collator(
            [reg.args for reg in requests],
            _collate,
            group_fn=lambda x: x[2],
            grouping=True,
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = (
            len(requests) // self.batch_size
            if len(requests) % self.batch_size == 0
            else len(requests) // self.batch_size + 1
        )
        pbar = tqdm(
            total=num_iters,
            disable=(self.rank != 0),
            desc="Model Responding",
        )
        e2e_latency = 0
        total_tokens = 0
        for chunk in chunks:
            (
                ctx,
                doc_to_messages,
                all_gen_kwargs,
                doc_id,
                task,
                split,
            ) = zip(*chunk)
            chat_messages = [
                doc_to_messages[idx](self.task_dict[task][split][ids])
                for idx, (ids, task, split) in enumerate(
                    zip(doc_id, task, split)
                )
            ]
            chat_messages: List[ChatMessages] = [
                ChatMessages(**{"messages": message})
                for message in chat_messages
            ]
            visuals = []
            videos = []
            for messages in chat_messages:
                visual, video, _ = messages.extract_media()
                visuals.append(visual)
                videos.append(video)
            visuals = self.flatten(visuals)
            videos = self.flatten(videos)
            gen_kwargs = all_gen_kwargs[0]

            video_kwargs = {
                "max_pixels": self.max_pixels,
                "min_pixels": self.min_pixels,
            }
            if self.fps is not None:
                video_kwargs["fps"] = self.fps
            else:
                if videos and decord is not None:
                    try:
                        video_path = videos[0]
                        vr = decord.VideoReader(video_path)
                        video_total_frames = len(vr)
                        nframes = min(
                            self.max_num_frames, video_total_frames
                        )
                        nframes = (nframes // 2) * 2
                        nframes = max(2, nframes)
                        video_kwargs["nframes"] = nframes
                    except Exception as e:
                        eval_logger.warning(
                            f"Failed to probe video {videos[0]}: {e}, "
                            "using default nframes"
                        )
                        video_kwargs["nframes"] = self.max_num_frames
                else:
                    video_kwargs["nframes"] = self.max_num_frames

            batched_messages = [
                chat_message.to_hf_messages(video_kwargs=video_kwargs)
                for chat_message in chat_messages
            ]
            texts = self.processor.apply_chat_template(
                batched_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(
                batched_messages
            )
            padding_side = "left" if self.batch_size > 1 else "right"
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                padding_side=padding_side,
                return_tensors="pt",
            )

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            default_gen_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.0,
                "top_p": None,
                "num_beams": 1,
            }
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id

            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None
                current_gen_kwargs["top_k"] = None

            start_time = time.time()
            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=current_gen_kwargs["do_sample"],
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                top_k=current_gen_kwargs.get("top_k", None),
                use_cache=self.use_cache,
            )
            end_time = time.time()

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, cont)
            ]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            e2e_latency += end_time - start_time
            total_tokens += sum(
                len(ids) for ids in generated_ids_trimmed
            )

            for ans, context in zip(answers, texts):
                clean_ans = parse_reasoning_model_answer(ans)
                res.append(clean_ans)
                self.cache_hook.add_partial(
                    "generate_until",
                    (context, gen_kwargs),
                    clean_ans,
                )

                eval_logger.debug(f"Question: {context}")
                eval_logger.debug(f"Model Raw Response: {ans}")
                eval_logger.debug(f"Model Clean Response: {clean_ans}")
            pbar.update(1)
        res = re_ords.get_original(res)

        avg_speed = total_tokens / e2e_latency if e2e_latency > 0 else 0
        metric_dict = {
            "total_tokens": total_tokens,
            "e2e_latency": e2e_latency,
            "avg_speed": avg_speed,
            "additional_metrics": {
                "rank": self.rank,
            },
        }
        log_metrics(**metric_dict)

        pbar.close()
        return res
