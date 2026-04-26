"""FETP-v3 inference wrapper for LLaVA-OneVision-1.5."""

import importlib
import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from loguru import logger as eval_logger
from transformers.cache_utils import DynamicCache
from transformers.utils import is_torchdynamo_compiling

from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model

from .llava_onevision1_5 import Llava_OneVision1_5 as LlavaOneVision1_5Chat


def _get_module_dtype(module) -> torch.dtype:
    for parameter in module.parameters():
        return parameter.dtype
    for buffer in module.buffers():
        return buffer.dtype
    raise ValueError(f"Cannot infer dtype for module {module.__class__.__name__}")


def _cast_tensor_for_module(tensor: torch.Tensor, module) -> torch.Tensor:
    target_dtype = _get_module_dtype(module)
    if tensor.dtype == target_dtype:
        return tensor
    return tensor.to(dtype=target_dtype)


def _slice_attention_mask(attention_mask, keep_indices: torch.Tensor):
    if attention_mask is None:
        return None
    if isinstance(attention_mask, dict):
        return {
            key: _slice_attention_mask(value, keep_indices)
            for key, value in attention_mask.items()
        }
    if attention_mask.ndim == 2:
        return attention_mask[:, keep_indices]
    if attention_mask.ndim == 4:
        return attention_mask[:, :, keep_indices][:, :, :, keep_indices]
    return attention_mask


def _slice_position_ids(position_ids: torch.Tensor, keep_indices: torch.Tensor):
    if position_ids is None:
        return None
    if position_ids.ndim == 2:
        return position_ids[:, keep_indices]
    if position_ids.ndim == 3:
        return position_ids[:, :, keep_indices]
    return position_ids


def _slice_position_embeddings(
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    positions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos, sin = position_embeddings
    if cos.ndim == 3:
        return cos.index_select(1, positions), sin.index_select(1, positions)
    if cos.ndim == 4:
        return cos.index_select(2, positions), sin.index_select(2, positions)
    raise ValueError(f"Unsupported rotary embedding shape: {cos.shape}")


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch,
        num_key_value_heads,
        n_rep,
        slen,
        head_dim,
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _get_suffix_text_positions(
    input_ids: torch.Tensor,
    visual_positions: torch.Tensor,
    config,
) -> torch.Tensor:
    visual_related_token_ids = torch.tensor(
        [config.image_token_id, config.video_token_id],
        device=input_ids.device,
    )
    visual_related_mask = torch.isin(input_ids, visual_related_token_ids)
    non_visual_positions = torch.where(~visual_related_mask)[0]
    if non_visual_positions.numel() == 0:
        return non_visual_positions

    last_visual_related_position = torch.where(visual_related_mask)[0].max()
    text_positions = non_visual_positions[
        non_visual_positions > last_visual_related_position
    ]
    if text_positions.numel() > 0:
        return text_positions
    return non_visual_positions[non_visual_positions > visual_positions[-1]]


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


@torch.no_grad()
def _compute_fes_scores_from_compact_inputs(
    text_to_vis_logits: torch.Tensor,
    visual_value_states: torch.Tensor,
    use_alpha: bool = True,
    use_deviation: bool = True,
    text_chunk_size: Optional[int] = 32,
) -> torch.Tensor:
    n_vis = visual_value_states.shape[2]
    if n_vis == 0:
        return torch.empty(0, device=visual_value_states.device)

    if text_to_vis_logits.shape[2] == 0:
        return torch.ones(n_vis, device=visual_value_states.device)

    text_to_vis_alpha = F.softmax(text_to_vis_logits.float(), dim=-1)
    alpha_per_text = text_to_vis_alpha.mean(dim=1)[0]

    vis_values = visual_value_states[0].float()
    vis_values = vis_values.permute(1, 0, 2).reshape(n_vis, -1)

    total_text_tokens = alpha_per_text.shape[0]
    if text_chunk_size is None or text_chunk_size <= 0:
        text_chunk_size = total_text_tokens

    vis_norm_sq = vis_values.pow(2).sum(dim=-1)
    alpha_sum = torch.zeros(n_vis, device=vis_values.device, dtype=torch.float32)
    deviation_sum = torch.zeros(
        n_vis, device=vis_values.device, dtype=torch.float32
    )
    score_sum = torch.zeros(n_vis, device=vis_values.device, dtype=torch.float32)

    for start in range(0, total_text_tokens, text_chunk_size):
        end = min(start + text_chunk_size, total_text_tokens)
        alpha_chunk = alpha_per_text[start:end]
        o_chunk = alpha_chunk @ vis_values
        o_norm_sq = o_chunk.pow(2).sum(dim=-1, keepdim=True)
        deviation_sq_chunk = (
            vis_norm_sq.unsqueeze(0)
            + o_norm_sq
            - 2.0 * (o_chunk @ vis_values.T)
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
    if use_alpha and use_deviation:
        return scores

    alpha_mean = alpha_sum / total_text_tokens
    deviation_mean = deviation_sum / total_text_tokens
    if use_alpha:
        return alpha_mean
    if use_deviation:
        return deviation_mean
    return torch.ones(n_vis, device=visual_value_states.device)


def _resolve_scoring_plan(
    scoring_method: str,
    shallow_layers: int,
    target_layer: int,
    num_layers: int,
) -> Tuple[int, int]:
    if scoring_method not in {"full", "shallow"}:
        raise ValueError("scoring_method must be one of {'full', 'shallow'}")

    if scoring_method == "full":
        num_run_layers = num_layers
    else:
        num_run_layers = min(max(1, shallow_layers), num_layers)

    if target_layer < 0:
        extract_at = num_run_layers + target_layer
    else:
        extract_at = target_layer
    extract_at = min(max(extract_at, 0), num_run_layers - 1)
    return num_run_layers, extract_at


@torch.no_grad()
def _forward_extract(
    language_model,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values,
    num_layers: int,
    attn_layer: int,
    text_positions: torch.Tensor,
    visual_positions: torch.Tensor,
):
    hidden = _cast_tensor_for_module(
        inputs_embeds,
        language_model.layers[0].input_layernorm,
    )
    causal_mask = language_model._update_causal_mask(
        attention_mask,
        hidden,
        cache_position,
        past_key_values,
        False,
    )
    position_embeddings = language_model.rotary_emb(hidden, position_ids)

    for layer_idx in range(min(num_layers, len(language_model.layers))):
        layer = language_model.layers[layer_idx]
        hidden = _cast_tensor_for_module(hidden, layer.input_layernorm)
        hidden_normed = layer.input_layernorm(hidden)

        if layer_idx == attn_layer:
            attn_module = layer.self_attn
            bsz = hidden_normed.size(0)
            n_vis = visual_positions.numel()

            visual_hidden = hidden_normed.index_select(1, visual_positions)
            visual_hidden_for_k = _cast_tensor_for_module(
                visual_hidden,
                attn_module.k_proj,
            )
            visual_hidden_for_v = _cast_tensor_for_module(
                visual_hidden,
                attn_module.v_proj,
            )
            visual_shape = (bsz, n_vis, -1, attn_module.head_dim)
            key_states = attn_module.k_proj(visual_hidden_for_k).view(visual_shape)
            key_states = attn_module.k_norm(key_states).transpose(1, 2)
            value_states = attn_module.v_proj(visual_hidden_for_v).view(
                visual_shape
            ).transpose(1, 2)

            visual_cos, visual_sin = _slice_position_embeddings(
                position_embeddings,
                visual_positions,
            )
            visual_cos = visual_cos.to(dtype=key_states.dtype)
            visual_sin = visual_sin.to(dtype=key_states.dtype)
            key_states, _ = _apply_rotary_pos_emb(
                key_states,
                key_states,
                visual_cos,
                visual_sin,
            )
            key_states_expanded = _repeat_kv(
                key_states,
                attn_module.num_key_value_groups,
            )

            if text_positions.numel() == 0:
                empty_logits = hidden.new_empty(
                    bsz,
                    attn_module.num_heads,
                    0,
                    n_vis,
                )
                return empty_logits, value_states.float()

            text_hidden = hidden_normed.index_select(1, text_positions)
            text_hidden = _cast_tensor_for_module(
                text_hidden,
                attn_module.q_proj,
            )
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
            text_cos = text_cos.to(dtype=query_states.dtype)
            text_sin = text_sin.to(dtype=query_states.dtype)
            query_states, _ = _apply_rotary_pos_emb(
                query_states,
                query_states,
                text_cos,
                text_sin,
            )
            scaling = getattr(attn_module, "scaling", 1.0 / math.sqrt(attn_module.head_dim))
            text_to_vis_logits = torch.matmul(
                query_states,
                key_states_expanded.transpose(2, 3),
            ) * scaling
            return text_to_vis_logits.float(), value_states.float()

        layer_outputs = layer(
            hidden,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

    return None, None


def _make_fetp_forward(
    output_cls,
    retention_ratio: float,
    scoring_method: str,
    shallow_layers: int,
    target_layer: int,
    use_alpha: bool = True,
    use_deviation: bool = True,
    two_stage: bool = False,
    text_chunk_size: Optional[int] = 32,
):
    def patched_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        rope_deltas=None,
        cache_position=None,
    ):
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
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        is_prefill = (
            past_key_values is None
            or (
                hasattr(past_key_values, "get_seq_length")
                and past_key_values.get_seq_length() == 0
            )
        )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            if pixel_values is not None:
                image_embeds = self.get_image_features(pixel_values, image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if (
                    not is_torchdynamo_compiling()
                    and n_image_tokens != n_image_features
                ):
                    raise ValueError(
                        "Image features and image tokens do not match: "
                        f"tokens: {n_image_tokens}, features {n_image_features}"
                    )
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                video_embeds = self.get_video_features(
                    pixel_values_videos,
                    video_grid_thw,
                )
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if (
                    not is_torchdynamo_compiling()
                    and n_video_tokens != n_video_features
                ):
                    raise ValueError(
                        "Video features and video tokens do not match: "
                        f"tokens: {n_video_tokens}, features {n_video_features}"
                    )
                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        if use_cache is True and past_key_values is None:
            past_key_values = DynamicCache()

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

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if (
            input_ids is not None
            and inputs_embeds.shape[0] == 1
            and is_prefill
            and retention_ratio != 0
        ):
            image_token_count = int((input_ids[0] == self.config.image_token_id).sum())
            video_token_count = int((input_ids[0] == self.config.video_token_id).sum())
            if image_token_count > 0 and video_token_count == 0:
                visual_token_id = self.config.image_token_id
                n_visual_tokens = image_token_count
            elif video_token_count > 0 and image_token_count == 0:
                visual_token_id = self.config.video_token_id
                n_visual_tokens = video_token_count
            elif image_token_count == 0 and video_token_count == 0:
                visual_token_id = None
                n_visual_tokens = None
            else:
                eval_logger.warning(
                    "FETP(LLaVA-OneVision-1.5): simultaneous image+video "
                    "inputs are not pruned; using the original forward path."
                )
                visual_token_id = None
                n_visual_tokens = None

            if n_visual_tokens is not None:
                visual_positions = torch.where(input_ids[0] == visual_token_id)[0]
                if (
                    visual_positions.numel() > 0
                    and visual_positions.numel() == n_visual_tokens
                ):
                    original_input_ids = input_ids
                    original_inputs_embeds = inputs_embeds
                    original_position_ids = position_ids
                    original_attention_mask = attention_mask
                    original_cache_position = cache_position

                    try:
                        original_n_visual_tokens = n_visual_tokens
                        if two_stage and n_visual_tokens > 1:
                            layer0 = self.language_model.layers[0]
                            stage1_hidden = _cast_tensor_for_module(
                                inputs_embeds,
                                layer0.input_layernorm,
                            )
                            hidden_normed = layer0.input_layernorm(stage1_hidden)
                            visual_hidden = hidden_normed.index_select(
                                1,
                                visual_positions,
                            )
                            visual_hidden = _cast_tensor_for_module(
                                visual_hidden,
                                layer0.self_attn.v_proj,
                            )
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
                                    visual_positions.index_select(0, stage1_keep),
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
                            input_ids = input_ids.index_select(
                                1,
                                keep_global_indices,
                            )
                            position_ids = _slice_position_ids(
                                position_ids,
                                keep_global_indices,
                            )
                            attention_mask = _slice_attention_mask(
                                attention_mask,
                                keep_global_indices,
                            )
                            cache_position = torch.arange(
                                inputs_embeds.shape[1],
                                device=inputs_embeds.device,
                            )

                            visual_positions = torch.where(
                                input_ids[0] == visual_token_id
                            )[0]
                            n_visual_tokens = int(visual_positions.numel())

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
                        num_keep = min(num_keep, n_visual_tokens)

                        num_run_layers, extract_at = _resolve_scoring_plan(
                            scoring_method=scoring_method,
                            shallow_layers=shallow_layers,
                            target_layer=target_layer,
                            num_layers=len(self.language_model.layers),
                        )
                        text_positions = _get_suffix_text_positions(
                            input_ids[0],
                            visual_positions,
                            self.config,
                        )

                        text_to_vis_logits, visual_value_states = _forward_extract(
                            self.language_model,
                            inputs_embeds,
                            attention_mask,
                            position_ids,
                            cache_position,
                            past_key_values,
                            num_layers=num_run_layers,
                            attn_layer=extract_at,
                            text_positions=text_positions,
                            visual_positions=visual_positions,
                        )

                        if (
                            text_to_vis_logits is not None
                            and visual_value_states is not None
                        ):
                            scores = _compute_fes_scores_from_compact_inputs(
                                text_to_vis_logits=text_to_vis_logits,
                                visual_value_states=visual_value_states,
                                use_alpha=use_alpha,
                                use_deviation=use_deviation,
                                text_chunk_size=text_chunk_size,
                            )
                            _, top_indices = scores.topk(num_keep)
                            keep_visual_local = top_indices.sort().values
                        else:
                            eval_logger.warning(
                                "FETP(LLaVA-OneVision-1.5): attention extraction "
                                "failed, falling back to uniform selection."
                            )
                            step = n_visual_tokens / num_keep
                            keep_visual_local = (
                                torch.arange(
                                    num_keep,
                                    device=inputs_embeds.device,
                                )
                                .float()
                                .mul(step)
                                .long()
                            )

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
                        position_ids = _slice_position_ids(
                            position_ids,
                            keep_global_indices,
                        )
                        attention_mask = _slice_attention_mask(
                            attention_mask,
                            keep_global_indices,
                        )
                        cache_position = torch.arange(
                            inputs_embeds.shape[1],
                            device=inputs_embeds.device,
                        )
                    except Exception as exc:
                        input_ids = original_input_ids
                        inputs_embeds = original_inputs_embeds
                        position_ids = original_position_ids
                        attention_mask = original_attention_mask
                        cache_position = original_cache_position
                        eval_logger.warning(
                            "FETP(LLaVA-OneVision-1.5): pruning skipped due to "
                            f"scoring failure: {exc}"
                        )
                else:
                    eval_logger.warning(
                        "FETP(LLaVA-OneVision-1.5): visual placeholder count "
                        "does not match extracted visual token count; skipping "
                        "pruning."
                    )

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
        )

        output = output_cls(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=getattr(self, "rope_deltas", None),
        )
        return output if return_dict else output.to_tuple()

    return patched_forward


def _patch_llava_onevision1_5_model(
    model,
    *,
    retention_ratio: float,
    scoring_method: str,
    shallow_layers: int,
    target_layer: int,
    use_alpha: bool,
    use_deviation: bool,
    two_stage: bool,
    text_chunk_size: Optional[int],
) -> None:
    inner_model = getattr(model, "model", None)
    if inner_model is None or not hasattr(inner_model, "language_model"):
        raise TypeError(
            "Expected a LLaVA-OneVision-1.5 model with a nested `.model` "
            "and `.language_model`."
        )

    remote_module = importlib.import_module(inner_model.__class__.__module__)
    output_cls = getattr(remote_module, "LLaVAOneVision1_5_ModelOutputWithPast")

    inner_model.__class__.forward = _make_fetp_forward(
        output_cls=output_cls,
        retention_ratio=retention_ratio,
        scoring_method=scoring_method,
        shallow_layers=shallow_layers,
        target_layer=target_layer,
        use_alpha=use_alpha,
        use_deviation=use_deviation,
        two_stage=two_stage,
        text_chunk_size=text_chunk_size,
    )


@register_model("llava_onevision1_5_ours_v3")
class LlavaOneVision1_5OursV3(LlavaOneVision1_5Chat):
    is_simple = False

    def __init__(
        self,
        *args,
        retention_ratio: float = 0.25,
        scoring_method: str = "full",
        shallow_layers: int = 4,
        target_layer: int = 20,
        use_alpha: bool = True,
        use_deviation: bool = True,
        two_stage: bool = False,
        text_chunk_size: Optional[int] = 32,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        retention_ratio = float(retention_ratio)
        scoring_method = str(scoring_method)
        shallow_layers = int(shallow_layers)
        target_layer = int(target_layer)
        text_chunk_size = (
            None if text_chunk_size is None else int(text_chunk_size)
        )

        self.retention_ratio = retention_ratio
        self.scoring_method = scoring_method
        self.shallow_layers = shallow_layers
        self.target_layer = target_layer
        self.use_alpha = bool(use_alpha)
        self.use_deviation = bool(use_deviation)
        self.two_stage = bool(two_stage)
        self.text_chunk_size = text_chunk_size

        eval_logger.info(
            "[LlavaOneVision1_5OursV3 / FETP-v3] "
            f"retention_ratio={retention_ratio}, "
            f"scoring_method={scoring_method}, "
            f"shallow_layers={shallow_layers}, "
            f"target_layer={target_layer}, "
            f"use_alpha={use_alpha}, "
            f"use_deviation={use_deviation}, "
            f"two_stage={two_stage}, "
            f"text_chunk_size={text_chunk_size}"
        )

        _patch_llava_onevision1_5_model(
            self.model,
            retention_ratio=retention_ratio,
            scoring_method=scoring_method,
            shallow_layers=shallow_layers,
            target_layer=target_layer,
            use_alpha=self.use_alpha,
            use_deviation=self.use_deviation,
            two_stage=self.two_stage,
            text_chunk_size=text_chunk_size,
        )

    def generate_until(self, requests: List[Instance]) -> List[str]:
        original_requests = list(requests)
        cached_responses, pending_requests = self.split_requests_by_cache(
            original_requests
        )
        if not pending_requests:
            return self.merge_cached_and_new_responses(
                original_requests,
                cached_responses,
                [],
                [],
            )

        pending_responses = super().generate_until(pending_requests)
        for request, response in zip(pending_requests, pending_responses):
            self.add_request_response_to_cache(request, response)

        return self.merge_cached_and_new_responses(
            original_requests,
            cached_responses,
            pending_requests,
            pending_responses,
        )
