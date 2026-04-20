"""Functionally Equivalent Token Pruning V3 (FETP-v3) for Qwen3-VL.

This adapts the per-text-token FETP-v3 scoring used in the external
``qwen2_5_vl_ours_v3.py`` implementation to Qwen3-VL's DeepStack-based
multimodal stack.
"""

import json
import math
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModel,
    Qwen3VLModelOutputWithPast,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, is_torchdynamo_compiling

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)
from lmms_eval.models.simple.qwen3_vl import Qwen3_VL as Qwen3_VLSimple
from lmms_eval.protocol import ChatMessages

from .qwen3_vl_ours_v2 import (
    _get_suffix_text_positions,
    _merge_visual_inputs,
    _slice_attention_mask,
    _slice_position_embeddings,
    _unpack_visual_outputs,
)

process_vision_info, _has_qwen_vl = optional_import(
    "qwen_vl_utils", "process_vision_info"
)
if not _has_qwen_vl:
    eval_logger.warning(
        "Failed to import qwen_vl_utils; "
        "Please install it via `pip install qwen-vl-utils`"
    )


# ---------------------------------------------------------------------------
# Compatibility helpers kept for the existing local tests / playground.
# ---------------------------------------------------------------------------


def _normalize_anchor_layers(
    anchor_layers: Optional[Union[str, Sequence[int]]],
    num_layers: int,
) -> Tuple[int, ...]:
    if num_layers <= 0:
        return tuple()

    if anchor_layers is None:
        quarter = min(num_layers - 1, max(0, num_layers // 4))
        middle = min(num_layers - 1, max(0, num_layers // 2))
        three_quarter = min(num_layers - 1, max(0, (3 * num_layers) // 4))
        anchor_layers = [quarter, middle, three_quarter]
    elif isinstance(anchor_layers, str):
        anchor_layers = [
            int(part.strip())
            for part in anchor_layers.split(",")
            if part.strip()
        ]

    cleaned = sorted(
        {
            int(layer_idx)
            for layer_idx in anchor_layers
            if 0 <= int(layer_idx) < num_layers
        }
    )
    if not cleaned:
        return (min(num_layers - 1, max(0, num_layers // 2)),)
    return tuple(cleaned)


def _aggregate_anchor_scores(
    score_list: Sequence[torch.Tensor],
    weights: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    if not score_list:
        return torch.empty(0)

    stacked = torch.stack(score_list, dim=0)
    if weights is None:
        weight_tensor = torch.ones(
            stacked.shape[0], device=stacked.device, dtype=stacked.dtype
        )
    else:
        weight_tensor = torch.tensor(
            weights,
            device=stacked.device,
            dtype=stacked.dtype,
        )
    weight_tensor = weight_tensor / weight_tensor.sum().clamp_min(1e-12)
    return (stacked * weight_tensor.view(-1, 1)).sum(dim=0)


def _summarize_pruning_stats(
    scoring_method: str,
    anchor_layers: Sequence[int],
    num_visual_tokens: int,
    num_keep: int,
    scoring_time_s: float,
    total_pruning_time_s: float,
    reference_method: Optional[str] = None,
    reference_scoring_time_s: Optional[float] = None,
    topk_overlap: Optional[float] = None,
    candidate_size: Optional[int] = None,
    score_query_tokens: Optional[int] = None,
    score_heads: Optional[int] = None,
) -> Dict[str, Union[str, int, float]]:
    stats: Dict[str, Union[str, int, float]] = {
        "pruning_scoring_method": scoring_method,
        "pruning_anchor_layers": ",".join(str(layer) for layer in anchor_layers),
        "pruning_num_visual_tokens": int(num_visual_tokens),
        "pruning_num_keep": int(num_keep),
        "pruning_scoring_time_ms": float(scoring_time_s * 1000.0),
        "pruning_total_time_ms": float(total_pruning_time_s * 1000.0),
    }
    if candidate_size is not None:
        stats["pruning_candidate_size"] = int(candidate_size)
    if score_query_tokens is not None:
        stats["pruning_score_query_tokens"] = int(score_query_tokens)
    if score_heads is not None:
        stats["pruning_score_heads"] = int(score_heads)
    if reference_method is not None:
        stats["pruning_reference_method"] = reference_method
    if reference_scoring_time_s is not None:
        stats["pruning_reference_scoring_time_ms"] = float(
            reference_scoring_time_s * 1000.0
        )
        if scoring_time_s > 0:
            stats["pruning_reference_speedup"] = float(
                reference_scoring_time_s / scoring_time_s
            )
    if topk_overlap is not None:
        stats["pruning_topk_overlap"] = float(topk_overlap)
    return stats


# ---------------------------------------------------------------------------
# Core scoring and stats collection.
# ---------------------------------------------------------------------------


_fes_distribution_stats: List[dict] = []


@torch.no_grad()
def _compute_fes_scores_from_compact_inputs(
    text_to_vis_logits: torch.Tensor,
    visual_value_states: torch.Tensor,
    use_alpha: bool = True,
    use_deviation: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute FETP-v3 scores from compact text-to-visual logits.

    Args:
        text_to_vis_logits: [1, n_heads, n_text, n_vis] logits before softmax.
        visual_value_states: [1, n_kv_heads, n_vis, head_dim] visual values.

    Returns:
        (scores, alpha_mean, deviation_mean), each shaped [n_vis].
    """
    n_vis = visual_value_states.shape[2]
    if n_vis == 0:
        empty = torch.empty(0, device=visual_value_states.device)
        return empty, empty, empty

    if text_to_vis_logits.shape[2] == 0:
        alpha_per_text = torch.ones(
            1,
            n_vis,
            device=visual_value_states.device,
            dtype=torch.float32,
        ) / n_vis
    else:
        text_to_vis_alpha = F.softmax(text_to_vis_logits.float(), dim=-1)
        alpha_per_text = text_to_vis_alpha.mean(dim=1)[0]

    vis_values = visual_value_states[0].float()
    vis_values = vis_values.permute(1, 0, 2).reshape(n_vis, -1)

    o_per_text = alpha_per_text @ vis_values
    diff = vis_values.unsqueeze(0) - o_per_text.unsqueeze(1)
    dev_sq = diff.norm(dim=-1).pow(2)

    alpha_factor = alpha_per_text.pow(2) if use_alpha else torch.ones_like(dev_sq)
    deviation_factor = dev_sq if use_deviation else torch.ones_like(dev_sq)
    scores = (alpha_factor * deviation_factor).mean(dim=0).sqrt()

    alpha_mean = alpha_per_text.mean(dim=0)
    deviation_mean = dev_sq.sqrt().mean(dim=0)
    return scores, alpha_mean, deviation_mean


@torch.no_grad()
def _compute_fes_scores_from_visual_logits(
    attn_logits: torch.Tensor,
    value_states: torch.Tensor,
    visual_positions: torch.Tensor,
    use_alpha: bool = True,
    use_deviation: bool = True,
) -> torch.Tensor:
    if visual_positions.numel() == 0:
        return torch.empty(0, device=value_states.device)

    compact_logits = attn_logits.index_select(3, visual_positions)
    compact_values = value_states.index_select(2, visual_positions)
    scores, _, _ = _compute_fes_scores_from_compact_inputs(
        text_to_vis_logits=compact_logits,
        visual_value_states=compact_values,
        use_alpha=use_alpha,
        use_deviation=use_deviation,
    )
    return scores


def _tensor_summary(tensor: torch.Tensor) -> dict:
    tensor = tensor.float()
    return {
        "mean": tensor.mean().item(),
        "std": tensor.std().item() if tensor.numel() > 1 else 0.0,
        "min": tensor.min().item(),
        "max": tensor.max().item(),
        "median": tensor.median().item(),
        "num_tokens": tensor.numel(),
    }


def _summarize_float_values(values: List[float]) -> dict:
    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "mean": float(mean),
        "std": float(math.sqrt(variance)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


@torch.no_grad()
def _diversity_prune(
    features: torch.Tensor,
    keep_ratio: float = 0.5,
) -> torch.Tensor:
    """Keep the most diverse visual tokens via cosine-similarity redundancy."""
    num_tokens = features.shape[0]
    num_keep = max(1, int(num_tokens * keep_ratio))
    if num_keep >= num_tokens:
        return torch.arange(num_tokens, device=features.device)

    feature_norm = F.normalize(features.float(), dim=-1)
    similarity = feature_norm @ feature_norm.T
    similarity.fill_diagonal_(0.0)
    redundancy = similarity.mean(dim=-1)
    _, keep_indices = redundancy.topk(num_keep, largest=False)
    return keep_indices.sort().values


def _slice_visual_inputs(
    visual_pos_masks: Optional[torch.Tensor],
    deepstack_visual_embeds,
    keep_global_indices: torch.Tensor,
):
    if visual_pos_masks is None:
        return visual_pos_masks, deepstack_visual_embeds

    original_visual_positions = torch.where(visual_pos_masks[0])[0]
    joint_visual_keep_indices = torch.where(
        torch.isin(original_visual_positions, keep_global_indices)
    )[0]
    visual_pos_masks = visual_pos_masks[:, keep_global_indices]
    if deepstack_visual_embeds is not None:
        deepstack_visual_embeds = [
            embed[joint_visual_keep_indices]
            for embed in deepstack_visual_embeds
        ]
    return visual_pos_masks, deepstack_visual_embeds


# ---------------------------------------------------------------------------
# Forward extraction and patched forward.
# ---------------------------------------------------------------------------


@torch.no_grad()
def _forward_extract(
    language_model,
    inputs_embeds: torch.Tensor,
    position_ids: torch.Tensor,
    cache_position: torch.Tensor,
    num_layers: int,
    attn_layer: int,
    text_positions: torch.Tensor,
    visual_positions: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    hidden = inputs_embeds

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        rope_position_ids = position_ids[1:]
    else:
        text_position_ids = position_ids[0]
        rope_position_ids = position_ids

    attention_mask = create_causal_mask(
        config=language_model.config,
        input_embeds=inputs_embeds,
        attention_mask=None,
        cache_position=cache_position,
        past_key_values=None,
        position_ids=text_position_ids,
    )
    position_embeddings = language_model.rotary_emb(hidden, rope_position_ids)

    attn_logits_out = None
    visual_value_states_out = None
    num_layers = min(num_layers, len(language_model.layers))

    for layer_idx in range(num_layers):
        layer = language_model.layers[layer_idx]

        if layer_idx == attn_layer:
            residual = hidden
            hidden_normed = layer.input_layernorm(hidden)

            attn_module = layer.self_attn
            bsz, seq_len, _ = hidden_normed.size()

            visual_hidden = hidden_normed.index_select(1, visual_positions)
            visual_shape = (
                bsz,
                visual_positions.numel(),
                -1,
                attn_module.head_dim,
            )
            key_states = attn_module.k_proj(visual_hidden).view(visual_shape)
            value_states = attn_module.v_proj(visual_hidden).view(visual_shape)

            key_states = attn_module.k_norm(key_states).transpose(1, 2)
            visual_value_states_out = value_states.transpose(1, 2).float().clone()

            visual_cos, visual_sin = _slice_position_embeddings(
                position_embeddings,
                visual_positions,
            )
            key_states, _ = apply_rotary_pos_emb(
                key_states,
                key_states,
                visual_cos,
                visual_sin,
            )
            key_states_expanded = repeat_kv(
                key_states, attn_module.num_key_value_groups
            )

            if text_positions.numel() == 0:
                attn_logits_out = hidden.new_empty(
                    bsz,
                    attn_module.num_heads,
                    0,
                    visual_positions.numel(),
                ).float()
            else:
                text_hidden = hidden_normed.index_select(1, text_positions)
                text_shape = (
                    bsz,
                    text_positions.numel(),
                    -1,
                    attn_module.head_dim,
                )
                query_states = attn_module.q_proj(text_hidden).view(text_shape)
                query_states = attn_module.q_norm(query_states).transpose(1, 2)

                text_cos, text_sin = _slice_position_embeddings(
                    position_embeddings,
                    text_positions,
                )
                query_states, _ = apply_rotary_pos_emb(
                    query_states,
                    query_states,
                    text_cos,
                    text_sin,
                )

                attn_logits_out = torch.matmul(
                    query_states,
                    key_states_expanded.transpose(2, 3),
                ) * attn_module.scaling
                attn_logits_out = attn_logits_out.float().clone()

            full_query_states = attn_module.q_proj(hidden_normed).view(
                bsz, seq_len, -1, attn_module.head_dim
            )
            full_key_states = attn_module.k_proj(hidden_normed).view(
                bsz, seq_len, -1, attn_module.head_dim
            )
            full_value_states = attn_module.v_proj(hidden_normed).view(
                bsz, seq_len, -1, attn_module.head_dim
            )

            full_query_states = attn_module.q_norm(full_query_states).transpose(1, 2)
            full_key_states = attn_module.k_norm(full_key_states).transpose(1, 2)
            full_value_states = full_value_states.transpose(1, 2)

            cos, sin = position_embeddings
            full_query_states, full_key_states = apply_rotary_pos_emb(
                full_query_states,
                full_key_states,
                cos,
                sin,
            )

            full_key_states_expanded = repeat_kv(
                full_key_states, attn_module.num_key_value_groups
            )
            full_value_states_expanded = repeat_kv(
                full_value_states, attn_module.num_key_value_groups
            )

            full_attn_logits = torch.matmul(
                full_query_states,
                full_key_states_expanded.transpose(2, 3),
            ) * attn_module.scaling
            causal_mask = torch.triu(
                torch.ones(
                    seq_len,
                    seq_len,
                    device=hidden.device,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
            full_attn_logits = full_attn_logits.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0),
                float("-inf"),
            )

            full_attn_weights = F.softmax(
                full_attn_logits, dim=-1, dtype=torch.float32
            )
            attn_output = torch.matmul(
                full_attn_weights.to(full_value_states_expanded.dtype),
                full_value_states_expanded,
            )
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, seq_len, -1)
            attn_output = attn_module.o_proj(attn_output)

            hidden = residual + attn_output
            residual = hidden
            hidden = residual + layer.mlp(
                layer.post_attention_layernorm(hidden)
            )
        else:
            layer_out = layer(
                hidden,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=None,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                output_attentions=False,
            )
            hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out

    return attn_logits_out, visual_value_states_out


def _resolve_scoring_plan(
    scoring_method: str,
    shallow_layers: int,
    target_layer: int,
    num_layers: int,
) -> Tuple[str, int, int]:
    resolved_scoring_method = scoring_method
    if resolved_scoring_method == "anchor":
        resolved_scoring_method = "shallow"

    if resolved_scoring_method not in {"full", "shallow"}:
        raise ValueError(
            "scoring_method must be one of {'full', 'shallow', 'anchor'}"
        )

    if resolved_scoring_method == "full":
        num_run_layers = num_layers
    else:
        num_run_layers = min(max(1, shallow_layers), num_layers)

    if target_layer < 0:
        extract_at = num_run_layers + target_layer
    else:
        extract_at = target_layer
    extract_at = min(max(extract_at, 0), num_run_layers - 1)

    return resolved_scoring_method, num_run_layers, extract_at


def _make_fetp_forward(
    retention_ratio: float,
    scoring_method: str,
    shallow_layers: int,
    target_layer: int,
    use_alpha: bool = True,
    use_deviation: bool = True,
    two_stage: bool = False,
):
    def patched_forward(
        self: Qwen3VLModel,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.IntTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLModelOutputWithPast]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        is_prefill_stage = (
            (cache_position is not None and cache_position[0] == 0)
            or (
                past_key_values is None
                or past_key_values.get_seq_length() == 0
            )
        )
        if is_prefill_stage or not hasattr(self, "_fetp_last_pruning_stats"):
            self._fetp_last_pruning_stats = {}

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length()
                if past_key_values is not None
                else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        image_mask = None
        video_mask = None
        deepstack_image_embeds = None
        deepstack_video_embeds = None
        n_image_tokens = None
        n_video_tokens = None

        if pixel_values is not None:
            image_outputs = self.get_image_features(
                pixel_values, image_grid_thw
            )
            image_embeds, deepstack_image_embeds = _unpack_visual_outputs(
                image_outputs
            )
            image_embeds = image_embeds.to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask, _ = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_embeds,
            )
            n_image_tokens = image_embeds.shape[0]
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask, image_embeds
            )

        if pixel_values_videos is not None:
            video_outputs = self.get_video_features(
                pixel_values_videos, video_grid_thw
            )
            video_embeds, deepstack_video_embeds = _unpack_visual_outputs(
                video_outputs
            )
            video_embeds = video_embeds.to(
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

        visual_pos_masks, deepstack_visual_embeds = _merge_visual_inputs(
            image_mask,
            video_mask,
            deepstack_image_embeds,
            deepstack_video_embeds,
        )

        if position_ids is None:
            attention_mask_tensor = (
                attention_mask
                if not isinstance(attention_mask, dict)
                else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(
                    attention_mask_tensor[:, 0], dim1=1, dim2=2
                )
                if attention_mask_tensor.dtype.is_floating_point:
                    min_value = torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = attention_mask_tensor / min_value
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

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
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(
                    seq_length, device=inputs_embeds.device
                )
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(
                        batch_size // delta.shape[0], dim=0
                    )
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        n_visual_tokens = n_video_tokens or n_image_tokens
        if n_video_tokens is not None:
            visual_token_id = self.config.video_token_id
        elif n_image_tokens is not None:
            visual_token_id = self.config.image_token_id
        else:
            visual_token_id = None

        if (
            n_visual_tokens is not None
            and position_ids.shape[-1] > 1
            and retention_ratio != 0
            and inputs_embeds.shape[0] == 1
            and input_ids is not None
        ):
            original_n_visual_tokens = int(n_visual_tokens)
            n_visual_tokens_after_stage1 = None
            visual_positions = torch.where(input_ids[0] == visual_token_id)[0]

            if (
                visual_positions.numel() > 0
                and visual_positions.numel() == n_visual_tokens
            ):
                total_pruning_start = time.perf_counter()

                if two_stage and n_visual_tokens > 1:
                    layer0 = self.language_model.layers[0]
                    hidden_normed = layer0.input_layernorm(inputs_embeds)
                    visual_hidden = hidden_normed.index_select(1, visual_positions)
                    stage1_values = layer0.self_attn.v_proj(visual_hidden)[0]
                    stage1_keep = _diversity_prune(
                        stage1_values,
                        keep_ratio=0.5,
                    )

                    non_visual_positions = torch.where(
                        input_ids[0] != visual_token_id
                    )[0]
                    keep_global_indices = torch.cat(
                        [
                            non_visual_positions,
                            visual_positions[stage1_keep],
                        ],
                        dim=0,
                    ).sort().values

                    hidden_size = inputs_embeds.shape[-1]
                    gather_index = keep_global_indices.view(1, -1, 1).expand(
                        inputs_embeds.shape[0],
                        -1,
                        hidden_size,
                    )
                    inputs_embeds = torch.gather(
                        inputs_embeds,
                        dim=1,
                        index=gather_index,
                    )
                    input_ids = input_ids[:, keep_global_indices]
                    position_ids = position_ids[:, :, keep_global_indices]
                    attention_mask = _slice_attention_mask(
                        attention_mask,
                        keep_global_indices,
                    )
                    cache_position = cache_position[keep_global_indices]
                    (
                        visual_pos_masks,
                        deepstack_visual_embeds,
                    ) = _slice_visual_inputs(
                        visual_pos_masks,
                        deepstack_visual_embeds,
                        keep_global_indices,
                    )

                    n_visual_tokens = int(stage1_keep.numel())
                    n_visual_tokens_after_stage1 = n_visual_tokens
                    if n_video_tokens is not None:
                        n_video_tokens = n_visual_tokens
                    else:
                        n_image_tokens = n_visual_tokens
                    visual_positions = torch.where(
                        input_ids[0] == visual_token_id
                    )[0]

                if retention_ratio < 1.0:
                    num_keep = max(
                        1,
                        int(original_n_visual_tokens * retention_ratio),
                    )
                else:
                    num_keep = max(
                        1,
                        min(int(retention_ratio), original_n_visual_tokens),
                    )
                num_keep = min(num_keep, int(n_visual_tokens))

                text_positions = _get_suffix_text_positions(
                    input_ids[0],
                    visual_positions,
                    self.config,
                )
                (
                    resolved_scoring_method,
                    num_run_layers,
                    extract_at,
                ) = _resolve_scoring_plan(
                    scoring_method=scoring_method,
                    shallow_layers=shallow_layers,
                    target_layer=target_layer,
                    num_layers=len(self.language_model.layers),
                )

                extract_start = time.perf_counter()
                text_to_vis_logits, visual_value_states = _forward_extract(
                    self.language_model,
                    inputs_embeds,
                    position_ids,
                    cache_position,
                    num_layers=num_run_layers,
                    attn_layer=extract_at,
                    text_positions=text_positions,
                    visual_positions=visual_positions,
                )
                scoring_time_s = time.perf_counter() - extract_start

                if (
                    text_to_vis_logits is not None
                    and visual_value_states is not None
                ):
                    scores, alpha_mean, deviation_mean = (
                        _compute_fes_scores_from_compact_inputs(
                            text_to_vis_logits=text_to_vis_logits,
                            visual_value_states=visual_value_states,
                            use_alpha=use_alpha,
                            use_deviation=use_deviation,
                        )
                    )
                    _fes_distribution_stats.append(
                        {
                            "sample_index": len(_fes_distribution_stats),
                            "n_visual_tokens_original": original_n_visual_tokens,
                            "n_visual_tokens_after_stage1": (
                                n_visual_tokens_after_stage1
                            ),
                            "n_visual_tokens": int(n_visual_tokens),
                            "num_keep": int(num_keep),
                            "n_visual_tokens_used": int(num_keep),
                            "alpha": _tensor_summary(alpha_mean),
                            "deviation": _tensor_summary(deviation_mean),
                            "score": _tensor_summary(scores),
                        }
                    )
                    keep_visual_local = scores.topk(num_keep).indices.sort().values
                    score_heads = int(text_to_vis_logits.shape[1])
                else:
                    eval_logger.warning(
                        "FETP(Qwen3-VL/V3): attention extraction failed, "
                        "falling back to uniform selection."
                    )
                    step = n_visual_tokens / num_keep
                    keep_visual_local = (
                        torch.arange(num_keep, device=inputs_embeds.device)
                        .float()
                        .mul(step)
                        .long()
                    )
                    _fes_distribution_stats.append(
                        {
                            "sample_index": len(_fes_distribution_stats),
                            "n_visual_tokens_original": original_n_visual_tokens,
                            "n_visual_tokens_after_stage1": (
                                n_visual_tokens_after_stage1
                            ),
                            "n_visual_tokens": int(n_visual_tokens),
                            "num_keep": int(num_keep),
                            "n_visual_tokens_used": int(num_keep),
                            "fallback": True,
                        }
                    )
                    score_heads = self.language_model.config.num_attention_heads

                non_visual_positions = torch.where(
                    input_ids[0] != visual_token_id
                )[0]
                keep_global_indices = torch.cat(
                    [
                        non_visual_positions,
                        visual_positions[keep_visual_local],
                    ],
                    dim=0,
                ).sort().values

                hidden_size = inputs_embeds.shape[-1]
                gather_index = keep_global_indices.view(1, -1, 1).expand(
                    inputs_embeds.shape[0],
                    -1,
                    hidden_size,
                )
                inputs_embeds = torch.gather(
                    inputs_embeds,
                    dim=1,
                    index=gather_index,
                )
                input_ids = input_ids[:, keep_global_indices]
                position_ids = position_ids[:, :, keep_global_indices]
                attention_mask = _slice_attention_mask(
                    attention_mask,
                    keep_global_indices,
                )
                cache_position = cache_position[keep_global_indices]
                (
                    visual_pos_masks,
                    deepstack_visual_embeds,
                ) = _slice_visual_inputs(
                    visual_pos_masks,
                    deepstack_visual_embeds,
                    keep_global_indices,
                )

                self._fetp_last_pruning_stats = _summarize_pruning_stats(
                    scoring_method=resolved_scoring_method,
                    anchor_layers=(extract_at,),
                    num_visual_tokens=original_n_visual_tokens,
                    num_keep=num_keep,
                    scoring_time_s=scoring_time_s,
                    total_pruning_time_s=time.perf_counter() - total_pruning_start,
                    candidate_size=int(n_visual_tokens),
                    score_query_tokens=int(text_positions.numel()),
                    score_heads=score_heads,
                )
                self._fetp_last_pruning_stats["pruning_two_stage"] = bool(
                    two_stage
                )
                eval_logger.info(
                    "[Qwen3_VL_Ours_V3 / FETP-v3] "
                    f"retention_ratio={retention_ratio}, "
                    f"scoring_method={resolved_scoring_method}, "
                    f"target_layer={extract_at}, "
                    f"two_stage={two_stage}, "
                    f"pruning_ms="
                    f"{self._fetp_last_pruning_stats['pruning_total_time_ms']:.2f}"
                )
            else:
                eval_logger.warning(
                    "FETP(Qwen3-VL/V3): visual placeholder count does not match "
                    "the extracted visual token count, skipping pruning."
                )

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        return Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )

    return patched_forward


def _make_fetp_anchor_forward(
    retention_ratio: float,
    scoring_method: str,
    shallow_layers: int,
    target_layer: int,
    anchor_layers: Optional[Union[str, Sequence[int]]] = None,
    anchor_weights: Optional[Union[str, Sequence[float]]] = None,
    candidate_ratio: float = 0.5,
    max_score_text_tokens: Optional[int] = 8,
    max_score_heads: Optional[int] = 8,
    use_alpha: bool = True,
    use_deviation: bool = True,
    profile_reference_scoring: bool = False,
    reference_scoring_method: str = "shallow",
    two_stage: bool = False,
):
    """Backward-compatible wrapper for the older anchor-layer API."""
    del anchor_weights
    del candidate_ratio
    del max_score_text_tokens
    del max_score_heads
    del profile_reference_scoring

    resolved_target_layer = target_layer
    if scoring_method == "anchor" and anchor_layers is not None:
        parsed_layers = _normalize_anchor_layers(anchor_layers, num_layers=10_000)
        if parsed_layers:
            resolved_target_layer = parsed_layers[len(parsed_layers) // 2]

    resolved_scoring_method = scoring_method
    if resolved_scoring_method == "anchor":
        resolved_scoring_method = reference_scoring_method

    return _make_fetp_forward(
        retention_ratio=retention_ratio,
        scoring_method=resolved_scoring_method,
        shallow_layers=shallow_layers,
        target_layer=resolved_target_layer,
        use_alpha=use_alpha,
        use_deviation=use_deviation,
        two_stage=two_stage,
    )


# ---------------------------------------------------------------------------
# Model class.
# ---------------------------------------------------------------------------


@register_model("qwen3_vl_ours_v3")
class Qwen3_VL_Ours_V3(Qwen3_VLSimple):
    is_simple = False

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-VL-8B-Instruct",
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
        retention_ratio: float = 0.25,
        scoring_method: str = "full",
        shallow_layers: int = 4,
        target_layer: int = 15,
        use_alpha: bool = True,
        use_deviation: bool = True,
        two_stage: bool = False,
        stats_output_path: Optional[str] = None,
        # Backward-compatible legacy v3 arguments.
        anchor_layers: Optional[Union[str, Sequence[int]]] = None,
        anchor_weights: Optional[Union[str, Sequence[float]]] = None,
        candidate_ratio: float = 0.5,
        max_score_text_tokens: Optional[int] = 8,
        max_score_heads: Optional[int] = 8,
        profile_reference_scoring: bool = False,
        reference_scoring_method: str = "shallow",
        **kwargs,
    ) -> None:
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

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
        self.stats_output_path = stats_output_path

        if scoring_method == "anchor":
            eval_logger.warning(
                "[Qwen3_VL_Ours_V3 / FETP-v3] "
                "legacy anchor scoring requested; falling back to the "
                "reference FETP-v3 path."
            )
            Qwen3VLModel.forward = _make_fetp_anchor_forward(
                retention_ratio=retention_ratio,
                scoring_method=scoring_method,
                shallow_layers=shallow_layers,
                target_layer=target_layer,
                anchor_layers=anchor_layers,
                anchor_weights=anchor_weights,
                candidate_ratio=candidate_ratio,
                max_score_text_tokens=max_score_text_tokens,
                max_score_heads=max_score_heads,
                use_alpha=use_alpha,
                use_deviation=use_deviation,
                profile_reference_scoring=profile_reference_scoring,
                reference_scoring_method=reference_scoring_method,
                two_stage=two_stage,
            )
        else:
            Qwen3VLModel.forward = _make_fetp_forward(
                retention_ratio=retention_ratio,
                scoring_method=scoring_method,
                shallow_layers=shallow_layers,
                target_layer=target_layer,
                use_alpha=use_alpha,
                use_deviation=use_deviation,
                two_stage=two_stage,
            )

        eval_logger.info(
            f"[Qwen3_VL_Ours_V3 / FETP-v3] "
            f"retention_ratio={retention_ratio}, "
            f"scoring_method={scoring_method}, "
            f"shallow_layers={shallow_layers}, "
            f"target_layer={target_layer}, "
            f"two_stage={two_stage}, "
            f"stats_output_path={stats_output_path}"
        )

    def generate_until(self, requests: List[Instance]) -> List[str]:
        _fes_distribution_stats.clear()
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
        e2e_latency = 0.0
        total_tokens = 0
        pruning_metric_sums: Dict[str, float] = {}
        pruning_metric_counts: Dict[str, int] = {}
        pruning_metric_last: Dict[str, Union[str, int, float, bool]] = {}

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
            chat_messages = [
                ChatMessages(**{"messages": message})
                for message in chat_messages
            ]
            gen_kwargs = all_gen_kwargs[0]

            video_kwargs = {
                "max_pixels": self.max_pixels,
                "min_pixels": self.min_pixels,
            }
            if self.fps is not None:
                video_kwargs["fps"] = self.fps
                video_kwargs["max_frames"] = self.max_num_frames
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

            image_inputs, video_inputs, video_kwargs_qwen = process_vision_info(
                batched_messages,
                return_video_kwargs=True,
                image_patch_size=16,
                return_video_metadata=True,
            )
            video_kwargs = {**video_kwargs, **video_kwargs_qwen}

            video_metadatas = None
            if video_inputs is not None:
                video_inputs, video_metadatas = zip(*video_inputs)
                video_inputs = list(video_inputs)
                video_metadatas = list(video_metadatas)

            if self.batch_size > 1:
                inputs = self.processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    video_metadata=video_metadatas,
                    **video_kwargs,
                    do_resize=False,
                    padding=True,
                    padding_side="left",
                    return_tensors="pt",
                )
            else:
                inputs = self.processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    video_metadata=video_metadatas,
                    **video_kwargs,
                    do_resize=False,
                    return_tensors="pt",
                )

            inputs.pop("token_type_ids", None)

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

            pruning_stats = getattr(
                self.model.model, "_fetp_last_pruning_stats", {}
            )
            for key, value in pruning_stats.items():
                if isinstance(value, bool):
                    pruning_metric_last[key] = value
                elif isinstance(value, (int, float)):
                    pruning_metric_sums[key] = pruning_metric_sums.get(
                        key, 0.0
                    ) + float(value)
                    pruning_metric_counts[key] = pruning_metric_counts.get(
                        key, 0
                    ) + 1
                else:
                    pruning_metric_last[key] = value

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
            total_tokens += sum(len(ids) for ids in generated_ids_trimmed)

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
        additional_metrics: Dict[str, Union[int, float, str, bool]] = {
            "rank": self.rank,
        }
        for key, total in pruning_metric_sums.items():
            count = pruning_metric_counts.get(key, 1)
            additional_metrics[key] = total / max(1, count)
        additional_metrics.update(pruning_metric_last)

        metric_dict = {
            "total_tokens": total_tokens,
            "e2e_latency": e2e_latency,
            "avg_speed": avg_speed,
            "additional_metrics": additional_metrics,
        }
        log_metrics(**metric_dict)

        if _fes_distribution_stats:
            all_alpha_means = [
                stat["alpha"]["mean"]
                for stat in _fes_distribution_stats
                if "alpha" in stat
            ]
            all_deviation_means = [
                stat["deviation"]["mean"]
                for stat in _fes_distribution_stats
                if "deviation" in stat
            ]
            all_score_means = [
                stat["score"]["mean"]
                for stat in _fes_distribution_stats
                if "score" in stat
            ]

            output = {
                "global_summary": {
                    "num_samples": len(_fes_distribution_stats),
                    "rank": self.rank,
                    "alpha_mean_across_samples": _summarize_float_values(
                        all_alpha_means
                    ),
                    "deviation_mean_across_samples": _summarize_float_values(
                        all_deviation_means
                    ),
                    "score_mean_across_samples": _summarize_float_values(
                        all_score_means
                    ),
                },
                "per_sample": _fes_distribution_stats,
            }

            if self.stats_output_path:
                stats_dir = self.stats_output_path
            else:
                stats_dir = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "..",
                    "..",
                    "logs",
                )
            os.makedirs(stats_dir, exist_ok=True)
            stats_path = os.path.join(
                stats_dir,
                f"qwen3_fes_v3_distribution_stats_r{self.retention_ratio}_rank{self.rank}.json",
            )
            with open(stats_path, "w", encoding="utf-8") as file:
                json.dump(output, file, indent=2)
            eval_logger.info(
                f"[FETP-v3] FES distribution stats written to {stats_path}"
            )

        pbar.close()
        return res
