import inspect
import time
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.models.internvl.modeling_internvl import (
    InternVLModel,
    InternVLModelOutputWithPast,
)
from transformers.models.qwen2.modeling_qwen2 import (
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.chat.internvl_hf import (
    InternVLHf,
    _prepare_internvl_media_inputs,
)
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.protocol import ChatMessages

warnings.filterwarnings("ignore")


def _call_mask_fn(
    mask_fn,
    *,
    config,
    inputs_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
):
    if position_ids is not None:
        cache_position = (
            position_ids[0]
            if position_ids.ndim > 1
            else position_ids
        ).to(inputs_embeds.device)
    else:
        cache_position = torch.arange(
            inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )

    kwargs = {
        "config": config,
        "attention_mask": attention_mask,
        "cache_position": cache_position,
        "past_key_values": None,
        "position_ids": position_ids,
    }
    try:
        parameters = inspect.signature(mask_fn).parameters
    except (TypeError, ValueError):
        parameters = {}

    embed_arg_name = (
        "inputs_embeds"
        if "inputs_embeds" in parameters or "input_embeds" not in parameters
        else "input_embeds"
    )
    kwargs[embed_arg_name] = inputs_embeds
    return mask_fn(**kwargs)


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


def _resolve_scoring_plan(
    scoring_method: str,
    shallow_layers: int,
    target_layer: int,
    anchor_layers: Optional[Union[str, Sequence[int]]],
    num_layers: int,
) -> Tuple[str, int, int]:
    resolved_target_layer = target_layer
    if scoring_method == "anchor" and anchor_layers is not None:
        parsed_layers = _normalize_anchor_layers(anchor_layers, num_layers)
        if parsed_layers:
            resolved_target_layer = parsed_layers[len(parsed_layers) // 2]

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

    if resolved_target_layer < 0:
        extract_at = num_run_layers + resolved_target_layer
    else:
        extract_at = resolved_target_layer
    extract_at = min(max(extract_at, 0), num_run_layers - 1)

    return resolved_scoring_method, num_run_layers, extract_at


def _get_suffix_text_positions(
    input_ids: torch.Tensor,
    visual_positions: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    non_visual_positions = torch.where(input_ids != image_token_id)[0]
    if visual_positions.numel() == 0:
        return non_visual_positions

    suffix_mask = non_visual_positions > visual_positions.max()
    suffix_positions = non_visual_positions[suffix_mask]
    if suffix_positions.numel() > 0:
        return suffix_positions
    return non_visual_positions


def _truncate_text_positions(
    text_positions: torch.Tensor,
    max_text_tokens: Optional[int],
) -> torch.Tensor:
    if max_text_tokens is None or max_text_tokens <= 0:
        return text_positions
    if text_positions.numel() <= max_text_tokens:
        return text_positions
    return text_positions[-max_text_tokens:]


def _select_score_head_indices(
    num_heads: int,
    max_score_heads: Optional[int],
    device: torch.device,
) -> Optional[torch.Tensor]:
    if max_score_heads is None or max_score_heads <= 0:
        return None
    if max_score_heads >= num_heads:
        return None

    step = max(1, num_heads // max_score_heads)
    head_indices = torch.arange(0, num_heads, step, device=device)
    return head_indices[:max_score_heads]


def _select_candidate_indices(
    coarse_scores: torch.Tensor,
    num_keep: int,
    candidate_ratio: float,
) -> torch.Tensor:
    total = coarse_scores.numel()
    if total == 0:
        return torch.empty(0, dtype=torch.long, device=coarse_scores.device)
    if candidate_ratio >= 1.0:
        return torch.arange(total, device=coarse_scores.device)

    num_candidate = max(num_keep, int(total * candidate_ratio))
    num_candidate = min(total, num_candidate)
    return coarse_scores.topk(num_candidate).indices.sort().values


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


@torch.no_grad()
def _diversity_prune(
    features: torch.Tensor,
    keep_ratio: float = 0.5,
) -> torch.Tensor:
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


def _summarize_pruning_stats(
    scoring_method: str,
    anchor_layers: Sequence[int],
    num_visual_tokens: int,
    num_keep: int,
    scoring_time_s: float,
    total_pruning_time_s: float,
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
    return stats


def _slice_sequence_tensor(
    tensor: Optional[torch.Tensor],
    keep_indices: torch.Tensor,
) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if tensor.ndim == 1:
        return tensor.index_select(0, keep_indices)
    if tensor.ndim == 2:
        return tensor.index_select(1, keep_indices)
    if tensor.ndim == 3:
        return tensor.index_select(-1, keep_indices)
    if tensor.ndim == 4:
        tensor = tensor.index_select(-1, keep_indices)
        return tensor.index_select(-2, keep_indices)
    return tensor


def _slice_position_embeddings(position_embeddings, positions: torch.Tensor):
    cos, sin = position_embeddings
    return cos.index_select(1, positions), sin.index_select(1, positions)


def _extract_pooler_output(outputs):
    pooler_output = getattr(outputs, "pooler_output", None)
    if pooler_output is not None:
        return pooler_output
    if isinstance(outputs, torch.Tensor):
        return outputs
    if isinstance(outputs, (tuple, list)):
        if len(outputs) > 1:
            return outputs[1]
        if outputs:
            return outputs[0]
    return outputs


@torch.no_grad()
def _compute_fes_scores_from_compact_inputs(
    text_to_vis_logits: torch.Tensor,
    visual_value_states: torch.Tensor,
    use_alpha: bool = True,
    use_deviation: bool = True,
    text_chunk_size: Optional[int] = 32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_visual = visual_value_states.shape[2]
    if num_visual == 0:
        empty = torch.empty(0, device=visual_value_states.device)
        return empty, empty, empty

    if text_to_vis_logits.shape[2] == 0:
        alpha_per_text = torch.ones(
            1,
            num_visual,
            device=visual_value_states.device,
            dtype=torch.float32,
        ) / num_visual
    else:
        text_to_vis_alpha = F.softmax(text_to_vis_logits.float(), dim=-1)
        alpha_per_text = text_to_vis_alpha.mean(dim=1)[0]

    vis_values = visual_value_states[0].float()
    vis_values = vis_values.permute(1, 0, 2).reshape(num_visual, -1)

    total_text_tokens = alpha_per_text.shape[0]
    if text_chunk_size is None or text_chunk_size <= 0:
        text_chunk_size = total_text_tokens

    vis_norm_sq = vis_values.pow(2).sum(dim=-1)
    alpha_sum = torch.zeros(
        num_visual, device=vis_values.device, dtype=torch.float32
    )
    deviation_sum = torch.zeros(
        num_visual, device=vis_values.device, dtype=torch.float32
    )
    score_sum = torch.zeros(
        num_visual, device=vis_values.device, dtype=torch.float32
    )

    for start in range(0, total_text_tokens, text_chunk_size):
        end = min(start + text_chunk_size, total_text_tokens)
        alpha_chunk = alpha_per_text[start:end]
        pooled_chunk = alpha_chunk @ vis_values
        pooled_norm_sq = pooled_chunk.pow(2).sum(dim=-1, keepdim=True)
        deviation_sq_chunk = (
            vis_norm_sq.unsqueeze(0)
            + pooled_norm_sq
            - 2.0 * (pooled_chunk @ vis_values.T)
        ).clamp_min_(0.0)

        if use_alpha:
            alpha_factor = alpha_chunk.pow(2)
        else:
            alpha_factor = torch.ones_like(deviation_sq_chunk)

        if use_deviation:
            deviation_factor = deviation_sq_chunk
        else:
            deviation_factor = torch.ones_like(deviation_sq_chunk)

        score_sum += (alpha_factor * deviation_factor).sum(dim=0)
        alpha_sum += alpha_chunk.sum(dim=0)
        deviation_sum += deviation_sq_chunk.sqrt().sum(dim=0)

    scores = (score_sum / total_text_tokens).sqrt()
    alpha_mean = alpha_sum / total_text_tokens
    deviation_mean = deviation_sum / total_text_tokens
    return scores, alpha_mean, deviation_mean


@torch.no_grad()
def _forward_extract(
    language_model,
    inputs_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    num_layers: int,
    attn_layer: int,
    text_positions: torch.Tensor,
    visual_positions: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    hidden = inputs_embeds

    if not isinstance(causal_mask_mapping := attention_mask, dict):
        causal_mask_mapping = {
            "full_attention": _call_mask_fn(
                create_causal_mask,
                config=language_model.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
            ),
        }
        if getattr(language_model, "has_sliding_layers", False):
            causal_mask_mapping["sliding_attention"] = (
                _call_mask_fn(
                    create_sliding_window_causal_mask,
                    config=language_model.config,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
            )

    position_embeddings = language_model.rotary_emb(hidden, position_ids)

    for layer_idx in range(min(num_layers, len(language_model.layers))):
        layer = language_model.layers[layer_idx]

        if layer_idx == attn_layer:
            hidden_normed = layer.input_layernorm(hidden)
            attn_module = layer.self_attn
            batch_size = hidden_normed.size(0)
            num_visual = visual_positions.numel()

            visual_hidden = hidden_normed.index_select(1, visual_positions)
            visual_shape = (
                batch_size,
                num_visual,
                -1,
                attn_module.head_dim,
            )
            key_states = attn_module.k_proj(visual_hidden).view(
                visual_shape
            ).transpose(1, 2)
            value_states = attn_module.v_proj(visual_hidden).view(
                visual_shape
            ).transpose(1, 2)

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
                key_states,
                attn_module.num_key_value_groups,
            )

            if text_positions.numel() == 0:
                empty_logits = hidden.new_empty(
                    batch_size,
                    attn_module.num_heads,
                    0,
                    num_visual,
                )
                return empty_logits, value_states.float()

            text_hidden = hidden_normed.index_select(1, text_positions)
            text_shape = (
                batch_size,
                text_positions.numel(),
                -1,
                attn_module.head_dim,
            )
            query_states = attn_module.q_proj(text_hidden).view(
                text_shape
            ).transpose(1, 2)

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
            text_to_vis_logits = torch.matmul(
                query_states,
                key_states_expanded.transpose(2, 3),
            ) * attn_module.scaling
            return text_to_vis_logits.float(), value_states.float()

        layer_type = language_model.config.layer_types[layer_idx]
        layer_out = layer(
            hidden,
            attention_mask=causal_mask_mapping[layer_type],
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            position_embeddings=position_embeddings,
        )
        hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out

    return None, None


def _make_fetp_forward(
    retention_ratio: float,
    scoring_method: str,
    shallow_layers: int,
    target_layer: int,
    anchor_layers: Optional[Union[str, Sequence[int]]] = None,
    candidate_ratio: float = 1.0,
    max_score_text_tokens: Optional[int] = None,
    max_score_heads: Optional[int] = None,
    use_alpha: bool = True,
    use_deviation: bool = True,
    two_stage: bool = False,
    text_chunk_size: Optional[int] = 32,
):
    def patched_forward(
        self: InternVLModel,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        vision_feature_layer: int | list[int] | None = None,
        vision_feature_select_strategy: str | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | InternVLModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        is_prefill_stage = past_key_values is None
        if hasattr(past_key_values, "get_seq_length"):
            is_prefill_stage = past_key_values.get_seq_length() == 0
        if is_prefill_stage or not hasattr(self, "_fetp_last_pruning_stats"):
            self._fetp_last_pruning_stats = {}

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_hidden_states = None
        if pixel_values is not None:
            image_features = _extract_pooler_output(
                self.get_image_features(
                    pixel_values=pixel_values,
                    vision_feature_layer=vision_feature_layer,
                    vision_feature_select_strategy=vision_feature_select_strategy,
                    return_dict=True,
                )
            )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            special_image_mask = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_features,
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
            image_hidden_states = image_features

        if position_ids is None:
            position_ids = torch.arange(
                inputs_embeds.shape[1],
                device=inputs_embeds.device,
            ).unsqueeze(0)

        if (
            pixel_values is not None
            and retention_ratio != 0
            and inputs_embeds.shape[0] == 1
            and input_ids is not None
            and is_prefill_stage
        ):
            visual_positions = torch.where(input_ids[0] == self.config.image_token_id)[0]
            num_visual_tokens = int(visual_positions.numel())
            original_num_visual_tokens = num_visual_tokens
            num_visual_tokens_after_stage1 = None

            if num_visual_tokens > 0:
                total_pruning_start = time.perf_counter()
                try:
                    if two_stage and num_visual_tokens > 1:
                        layer0 = self.language_model.layers[0]
                        hidden_normed = layer0.input_layernorm(inputs_embeds)
                        visual_hidden = hidden_normed.index_select(1, visual_positions)
                        stage1_values = layer0.self_attn.v_proj(visual_hidden)[0]
                        stage1_keep = _diversity_prune(
                            stage1_values,
                            keep_ratio=0.5,
                        )

                        non_visual_positions = torch.where(
                            input_ids[0] != self.config.image_token_id
                        )[0]
                        keep_indices = torch.cat(
                            [
                                non_visual_positions,
                                visual_positions.index_select(0, stage1_keep),
                            ],
                            dim=0,
                        ).sort().values

                        hidden_size = inputs_embeds.shape[-1]
                        gather_index = keep_indices.view(1, -1, 1).expand(
                            inputs_embeds.shape[0],
                            -1,
                            hidden_size,
                        )
                        inputs_embeds = torch.gather(
                            inputs_embeds,
                            dim=1,
                            index=gather_index,
                        )
                        input_ids = input_ids.index_select(1, keep_indices)
                        attention_mask = _slice_sequence_tensor(attention_mask, keep_indices)
                        position_ids = _slice_sequence_tensor(position_ids, keep_indices)

                        visual_positions = torch.where(
                            input_ids[0] == self.config.image_token_id
                        )[0]
                        num_visual_tokens = int(visual_positions.numel())
                        num_visual_tokens_after_stage1 = num_visual_tokens

                    if retention_ratio < 1.0:
                        num_keep = max(
                            1,
                            int(original_num_visual_tokens * retention_ratio),
                        )
                    else:
                        num_keep = max(
                            1,
                            min(int(retention_ratio), original_num_visual_tokens),
                        )
                    num_keep = min(num_keep, int(num_visual_tokens))

                    text_positions = _get_suffix_text_positions(
                        input_ids[0],
                        visual_positions,
                        self.config.image_token_id,
                    )
                    (
                        resolved_scoring_method,
                        num_run_layers,
                        extract_at,
                    ) = _resolve_scoring_plan(
                        scoring_method=scoring_method,
                        shallow_layers=shallow_layers,
                        target_layer=target_layer,
                        anchor_layers=anchor_layers,
                        num_layers=len(self.language_model.layers),
                    )

                    scoring_start = time.perf_counter()
                    text_to_vis_logits, visual_value_states = _forward_extract(
                        self.language_model,
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        num_layers=num_run_layers,
                        attn_layer=extract_at,
                        text_positions=text_positions,
                        visual_positions=visual_positions,
                    )
                    if (
                        text_to_vis_logits is None
                        or visual_value_states is None
                    ):
                        raise RuntimeError("compact attention extraction failed")

                    if two_stage:
                        final_scores, alpha_mean, deviation_mean = (
                            _compute_fes_scores_from_compact_inputs(
                                text_to_vis_logits=text_to_vis_logits,
                                visual_value_states=visual_value_states,
                                use_alpha=use_alpha,
                                use_deviation=use_deviation,
                                text_chunk_size=text_chunk_size,
                            )
                        )
                        keep_within_candidate = (
                            final_scores.topk(num_keep).indices.sort().values
                        )
                        keep_visual_positions = visual_positions.index_select(
                            0,
                            keep_within_candidate,
                        )
                        score_query_tokens = int(text_positions.numel())
                        score_heads = int(text_to_vis_logits.shape[1])
                        candidate_size = int(num_visual_tokens)
                    else:
                        coarse_text_positions = _truncate_text_positions(
                            text_positions,
                            max_score_text_tokens,
                        )
                        coarse_logits = text_to_vis_logits
                        if coarse_text_positions.numel() != text_positions.numel():
                            coarse_logits = coarse_logits[
                                :,
                                :,
                                -coarse_text_positions.numel():,
                                :,
                            ]

                        coarse_head_indices = _select_score_head_indices(
                            text_to_vis_logits.shape[1],
                            max_score_heads,
                            text_to_vis_logits.device,
                        )
                        if coarse_head_indices is not None:
                            coarse_logits = coarse_logits.index_select(
                                1,
                                coarse_head_indices,
                            )

                        coarse_scores, _, _ = (
                            _compute_fes_scores_from_compact_inputs(
                                text_to_vis_logits=coarse_logits,
                                visual_value_states=visual_value_states,
                                use_alpha=use_alpha,
                                use_deviation=use_deviation,
                                text_chunk_size=text_chunk_size,
                            )
                        )

                        candidate_local = _select_candidate_indices(
                            coarse_scores,
                            num_keep=num_keep,
                            candidate_ratio=candidate_ratio,
                        )
                        candidate_visual_positions = visual_positions.index_select(
                            0,
                            candidate_local,
                        )

                        if candidate_visual_positions.numel() != visual_positions.numel():
                            final_logits = text_to_vis_logits.index_select(
                                3,
                                candidate_local,
                            )
                            final_value_states = visual_value_states.index_select(
                                2,
                                candidate_local,
                            )
                        else:
                            final_logits = text_to_vis_logits
                            final_value_states = visual_value_states

                        final_scores, alpha_mean, deviation_mean = (
                            _compute_fes_scores_from_compact_inputs(
                                text_to_vis_logits=final_logits,
                                visual_value_states=final_value_states,
                                use_alpha=use_alpha,
                                use_deviation=use_deviation,
                                text_chunk_size=text_chunk_size,
                            )
                        )
                        keep_within_candidate = (
                            final_scores.topk(num_keep).indices.sort().values
                        )
                        keep_visual_positions = candidate_visual_positions.index_select(
                            0,
                            keep_within_candidate,
                        )
                        score_query_tokens = int(coarse_text_positions.numel())
                        score_heads = (
                            text_to_vis_logits.shape[1]
                            if coarse_head_indices is None
                            else int(coarse_head_indices.numel())
                        )
                        candidate_size = int(candidate_visual_positions.numel())

                    keep_visual_positions = keep_visual_positions.sort().values

                    non_visual_positions = torch.where(input_ids[0] != self.config.image_token_id)[0]
                    keep_indices = torch.cat(
                        [non_visual_positions, keep_visual_positions],
                        dim=0,
                    ).sort().values

                    hidden_size = inputs_embeds.shape[-1]
                    gather_index = keep_indices.view(1, -1, 1).expand(
                        inputs_embeds.shape[0],
                        -1,
                        hidden_size,
                    )
                    inputs_embeds = torch.gather(inputs_embeds, dim=1, index=gather_index)
                    input_ids = input_ids.index_select(1, keep_indices)
                    attention_mask = _slice_sequence_tensor(attention_mask, keep_indices)
                    position_ids = _slice_sequence_tensor(position_ids, keep_indices)

                    if image_hidden_states is not None:
                        kept_visual_mask = input_ids[0] == self.config.image_token_id
                        kept_visual_indices = torch.where(kept_visual_mask)[0]
                        image_hidden_states = inputs_embeds.index_select(1, kept_visual_indices)

                    scoring_time_s = time.perf_counter() - scoring_start
                    self._fetp_last_pruning_stats = _summarize_pruning_stats(
                        scoring_method=resolved_scoring_method,
                        anchor_layers=(extract_at,),
                        num_visual_tokens=original_num_visual_tokens,
                        num_keep=num_keep,
                        scoring_time_s=scoring_time_s,
                        total_pruning_time_s=time.perf_counter() - total_pruning_start,
                        candidate_size=candidate_size,
                        score_query_tokens=score_query_tokens,
                        score_heads=score_heads,
                    )
                    self._fetp_last_pruning_stats["pruning_two_stage"] = bool(two_stage)
                    if num_visual_tokens_after_stage1 is not None:
                        self._fetp_last_pruning_stats["pruning_num_visual_tokens_after_stage1"] = int(
                            num_visual_tokens_after_stage1
                        )
                    self._fetp_last_pruning_stats["pruning_score_summary_mean"] = float(
                        final_scores.float().mean().item()
                    )
                    self._fetp_last_pruning_stats["pruning_alpha_mean"] = float(
                        alpha_mean.float().mean().item()
                    )
                    self._fetp_last_pruning_stats["pruning_deviation_mean"] = float(
                        deviation_mean.float().mean().item()
                    )
                    eval_logger.info(
                        "[InternVL3_5_Ours_V3 / FETP] "
                        f"retention_ratio={retention_ratio}, "
                        f"scoring_method={resolved_scoring_method}, "
                        f"target_layer={extract_at}, "
                        f"two_stage={two_stage}, "
                        f"pruning_ms={self._fetp_last_pruning_stats['pruning_total_time_ms']:.2f}"
                    )
                except Exception as exc:
                    self._fetp_last_pruning_stats = {
                        "pruning_error": str(exc),
                        "pruning_num_visual_tokens": original_num_visual_tokens,
                        "pruning_two_stage": bool(two_stage),
                    }
                    eval_logger.warning(
                        f"InternVL3.5 FETP pruning skipped due to scoring failure: {exc}"
                    )

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        return InternVLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_hidden_states,
        )

    return patched_forward


@register_model("internvl3_5_ours_v3")
class InternVL3_5_Ours_V3(InternVLHf):
    is_simple = False

    def __init__(
        self,
        pretrained: str = "OpenGVLab/InternVL3_5-8B-HF",
        retention_ratio: float = 0.1,
        scoring_method: str = "full",
        shallow_layers: int = 4,
        target_layer: int = 20,
        anchor_layers: Optional[Union[str, Sequence[int]]] = None,
        anchor_weights: Optional[Union[str, Sequence[float]]] = None,
        candidate_ratio: float = 1.0,
        max_score_text_tokens: Optional[int] = None,
        max_score_heads: Optional[int] = None,
        use_alpha: bool = True,
        use_deviation: bool = True,
        two_stage: bool = True,
        text_chunk_size: Optional[int] = 32,
        profile_reference_scoring: bool = False,
        reference_scoring_method: str = "shallow",
        **kwargs,
    ) -> None:
        del anchor_weights
        del profile_reference_scoring
        del reference_scoring_method

        super().__init__(pretrained=pretrained, **kwargs)

        self.retention_ratio = retention_ratio
        eval_logger.info(
            "[InternVL3_5_Ours_V3 / FETP] "
            f"retention_ratio={retention_ratio}, "
            f"scoring_method={scoring_method}, "
            f"shallow_layers={shallow_layers}, "
            f"target_layer={target_layer}, "
            f"two_stage={two_stage}, "
            f"candidate_ratio={candidate_ratio}, "
            f"max_score_text_tokens={max_score_text_tokens}, "
            f"max_score_heads={max_score_heads}, "
            f"text_chunk_size={text_chunk_size}"
        )

        InternVLModel.forward = _make_fetp_forward(
            retention_ratio=retention_ratio,
            scoring_method=scoring_method,
            shallow_layers=shallow_layers,
            target_layer=target_layer,
            anchor_layers=anchor_layers,
            candidate_ratio=candidate_ratio,
            max_score_text_tokens=max_score_text_tokens,
            max_score_heads=max_score_heads,
            use_alpha=use_alpha,
            use_deviation=use_deviation,
            two_stage=two_stage,
            text_chunk_size=text_chunk_size,
        )

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res: List[str] = []

        def _collate(x):
            return x[2], x[2]

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
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        e2e_latency = 0.0
        total_tokens = 0
        pruning_metric_sums: Dict[str, float] = {}
        pruning_metric_counts: Dict[str, int] = {}
        pruning_metric_last: Dict[str, Union[str, int, float]] = {}

        for chunk in chunks:
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            chat_messages = [doc_to_messages[0](self.task_dict[task][split][ids]) for ids in doc_id]
            chat_messages = [ChatMessages(**{"messages": message}) for message in chat_messages]
            visuals = []
            videos = []
            for messages in chat_messages:
                visual, video, _ = messages.extract_media()
                visuals.append(visual)
                videos.append(video)
            visuals = self.flatten(visuals)
            videos = self.flatten(videos)

            images_kwargs = {}
            videos_kwargs = {}
            if self.min_patches is not None:
                images_kwargs["min_patches"] = self.min_patches
            if self.max_patches is not None:
                images_kwargs["max_patches"] = self.max_patches
            if self.num_frames is not None:
                videos_kwargs["num_frames"] = self.num_frames
            if self.fps is not None:
                videos_kwargs["fps"] = self.fps

            messages = chat_messages[0].model_dump()["messages"]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if self.accelerator.is_main_process and doc_id[0] % 100 == 0:
                eval_logger.debug(f"Prompt for doc ID {doc_id[0]}:\n\n{text}\n")

            visuals, videos, image_sizes = _prepare_internvl_media_inputs(
                visuals,
                videos,
            )
            inputs = self.processor(
                images=visuals,
                videos=videos,
                text=text,
                return_tensors="pt",
                **images_kwargs,
                **videos_kwargs,
            ).to(self.device, self.model.dtype)

            gen_kwargs = all_gen_kwargs[0]
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            do_sample = gen_kwargs["temperature"] > 0
            start_time = time.time()
            cont = self.model.generate(
                **inputs,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eot_token_id,
                do_sample=do_sample,
                temperature=gen_kwargs["temperature"] if do_sample else None,
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
            )
            end_time = time.time()

            pruning_stats = getattr(self.model.model, "_fetp_last_pruning_stats", {})
            for key, value in pruning_stats.items():
                if isinstance(value, bool):
                    pruning_metric_last[key] = value
                elif isinstance(value, (int, float)):
                    pruning_metric_sums[key] = pruning_metric_sums.get(key, 0.0) + float(value)
                    pruning_metric_counts[key] = pruning_metric_counts.get(key, 0) + 1
                else:
                    pruning_metric_last[key] = value

            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            e2e_latency += end_time - start_time
            total_tokens += sum(len(ids) for ids in generated_ids_trimmed)

            if self.accelerator.is_main_process and doc_id[0] % 100 == 0:
                eval_logger.debug(f"Generated text for doc ID {doc_id[0]}:\n\n{answers}\n")

            for answer in answers:
                res.append(answer)
                self.cache_hook.add_partial("generate_until", (text, gen_kwargs), answer)

            pbar.update(1)

        res = re_ords.get_original(res)

        metric_dict = {
            "total_tokens": total_tokens,
            "e2e_latency": e2e_latency,
            "avg_speed": total_tokens / e2e_latency if e2e_latency > 0 else 0,
            "additional_metrics": {
                "rank": self.rank,
                **{
                    key: pruning_metric_sums[key] / max(1, pruning_metric_counts.get(key, 1))
                    for key in pruning_metric_sums
                },
                **pruning_metric_last,
            },
        }
        log_metrics(**metric_dict)

        pbar.close()
        return res
