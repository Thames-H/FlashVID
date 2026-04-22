"""FETP-v3 for LLaVA-OneVision (HF format)."""

import inspect
import os
import time
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.models.llava_onevision.modeling_llava_onevision import (
    LlavaOnevisionModel,
    LlavaOnevisionModelOutputWithPast,
    image_size_to_num_patches,
)
from transformers.models.qwen2.modeling_qwen2 import (
    apply_rotary_pos_emb,
    repeat_kv,
)

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.protocol import ChatMessages

from .llava_hf import (
    LlavaHf as LlavaHfChat,
    _build_llava_processor_kwargs,
    _prepare_llava_media_inputs,
)
from .llava_onevision_visual_compare_utils import (
    attach_visual_compare_metadata,
    build_visual_compare_metadata,
)


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


def _call_feature_extractor(
    feature_fn,
    *args,
    return_dict: Optional[bool] = None,
    **kwargs,
):
    try:
        parameters = inspect.signature(feature_fn).parameters
    except (TypeError, ValueError):
        parameters = {}

    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if return_dict is not None and (
        "return_dict" in parameters or accepts_kwargs
    ):
        kwargs["return_dict"] = return_dict

    return feature_fn(*args, **kwargs)


def _filter_vision_tower_kwargs(vision_tower, kwargs) -> dict:
    if not kwargs:
        return {}

    try:
        parameters = inspect.signature(vision_tower.forward).parameters
    except (TypeError, ValueError, AttributeError):
        parameters = {}

    reserved = {"output_hidden_states", "return_dict"}
    return {
        key: value
        for key, value in kwargs.items()
        if key not in reserved and key in parameters
    }


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


def _concat_token_features(features):
    if isinstance(features, torch.Tensor):
        return features
    if isinstance(features, (tuple, list)):
        tensors = tuple(features)
        if not tensors:
            raise ValueError("Expected non-empty image feature sequence")
        return torch.cat(tensors, dim=0)
    raise TypeError(f"Unsupported image feature type: {type(features)!r}")


def _select_vision_features(
    vision_outputs,
    vision_feature_layer,
    vision_feature_select_strategy: Optional[str],
) -> torch.Tensor:
    if isinstance(vision_feature_layer, int):
        selected_feature = vision_outputs.hidden_states[vision_feature_layer]
    else:
        hidden_states_pool = [
            vision_outputs.hidden_states[layer_idx]
            for layer_idx in vision_feature_layer
        ]
        selected_feature = torch.cat(hidden_states_pool, dim=-1)

    if vision_feature_select_strategy == "default":
        selected_feature = selected_feature[:, 1:]
    return selected_feature


def _extract_image_features_compat(
    model: LlavaOnevisionModel,
    pixel_values: torch.Tensor,
    image_sizes,
    vision_feature_layer,
    vision_feature_select_strategy: Optional[str],
    vision_aspect_ratio: Optional[str],
    batch_num_images=None,
    **kwargs,
) -> torch.Tensor:
    vision_tower_kwargs = _filter_vision_tower_kwargs(
        model.vision_tower,
        kwargs,
    )
    if batch_num_images is None:
        need_patching = [True] * len(image_sizes)
    else:
        need_patching = [n == 1 for n in batch_num_images for _ in range(n)]

    image_num_patches = [
        image_size_to_num_patches(
            image_size=image_size,
            grid_pinpoints=model.config.image_grid_pinpoints,
            patch_size=model.config.vision_config.image_size,
        )
        if should_patch
        else 1
        for image_size, should_patch in zip(image_sizes, need_patching)
    ]

    if pixel_values.dim() == 5:
        pixel_values = torch.cat(
            [
                pixel_value[:num_patch]
                for pixel_value, num_patch in zip(pixel_values, image_num_patches)
            ],
            dim=0,
        )
    elif pixel_values.dim() != 4:
        raise ValueError(
            f"pixel_values of shape {pixel_values.shape}, expect to be 4D or 5D"
        )

    vision_outputs = model.vision_tower(
        pixel_values,
        output_hidden_states=True,
        return_dict=True,
        **vision_tower_kwargs,
    )
    selected_feature = _select_vision_features(
        vision_outputs,
        vision_feature_layer,
        vision_feature_select_strategy,
    )
    selected_feature = _cast_tensor_for_module(
        selected_feature,
        model.multi_modal_projector,
    )
    image_features = model.multi_modal_projector(selected_feature)
    image_features = torch.split(image_features, image_num_patches, dim=0)

    image_features, _ = model.pack_image_features(
        image_features,
        image_sizes,
        image_newline=model.image_newline.to(dtype=image_features[0].dtype),
        vision_aspect_ratio=vision_aspect_ratio,
    )
    return image_features


def _extract_video_features_compat(
    model: LlavaOnevisionModel,
    pixel_values: torch.Tensor,
    vision_feature_layer,
    vision_feature_select_strategy: Optional[str],
    **kwargs,
) -> torch.Tensor:
    vision_tower_kwargs = _filter_vision_tower_kwargs(
        model.vision_tower,
        kwargs,
    )
    batch_size, frames, channels, height, width = pixel_values.shape
    pixel_values = pixel_values.view(batch_size * frames, channels, height, width)
    vision_outputs = model.vision_tower(
        pixel_values,
        output_hidden_states=True,
        return_dict=True,
        **vision_tower_kwargs,
    )
    selected_feature = _select_vision_features(
        vision_outputs,
        vision_feature_layer,
        vision_feature_select_strategy,
    )
    selected_feature = _cast_tensor_for_module(
        selected_feature,
        model.multi_modal_projector,
    )
    video_features = model.multi_modal_projector(selected_feature)
    video_features = model.apply_pooling(video_features)
    video_features = video_features.reshape(
        batch_size,
        frames * video_features.shape[1],
        -1,
    )
    return video_features


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


def _cpu_tensor(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    return tensor.detach().cpu()


def _sanitize_artifact_component(value: Union[str, int]) -> str:
    text = str(value)
    sanitized = [
        ch if ch.isalnum() or ch in {"-", "_", "."} else "_"
        for ch in text
    ]
    return "".join(sanitized)


def _write_sample_artifact(
    stats_output_path: Optional[str],
    method_name: str,
    task_name: str,
    doc_id: Union[str, int],
    artifact: dict,
) -> Optional[str]:
    if not stats_output_path:
        return None

    artifact_dir = os.path.join(stats_output_path, "artifacts", method_name)
    os.makedirs(artifact_dir, exist_ok=True)
    artifact_name = (
        f"{_sanitize_artifact_component(task_name)}"
        f"__doc{_sanitize_artifact_component(doc_id)}.pt"
    )
    artifact_path = os.path.join(artifact_dir, artifact_name)
    torch.save(artifact, artifact_path)
    return artifact_path


def _build_fes_sample_artifact(
    question_text: str,
    visual_embeddings: torch.Tensor,
    visual_value_states: Optional[torch.Tensor],
    fetp_scores: torch.Tensor,
    attention_only_scores: torch.Tensor,
    fetp_keep_local: torch.Tensor,
    attention_only_keep_local: torch.Tensor,
    stage1_keep_local: Optional[torch.Tensor],
    num_keep: int,
    target_layer: int,
    scoring_method: str,
    n_visual_tokens_original: int,
    n_visual_tokens_after_stage1: Optional[int],
) -> dict:
    return {
        "method": "fetp",
        "question_text": question_text,
        "visual_embeddings": _cpu_tensor(visual_embeddings.float()),
        "visual_value_states": _cpu_tensor(visual_value_states.float())
        if visual_value_states is not None
        else None,
        "scores": {
            "fetp": _cpu_tensor(fetp_scores.float()),
            "attention_only": _cpu_tensor(attention_only_scores.float()),
        },
        "selection": {
            "fetp_keep_local": _cpu_tensor(fetp_keep_local.long()),
            "attention_only_keep_local": _cpu_tensor(
                attention_only_keep_local.long()
            ),
            "stage1_keep_local": _cpu_tensor(stage1_keep_local.long())
            if stage1_keep_local is not None
            else None,
            "num_keep": int(num_keep),
        },
        "metadata": {
            "target_layer": int(target_layer),
            "scoring_method": scoring_method,
            "n_visual_tokens_original": int(n_visual_tokens_original),
            "n_visual_tokens_after_stage1": (
                int(n_visual_tokens_after_stage1)
                if n_visual_tokens_after_stage1 is not None
                else None
            ),
            "n_visual_tokens_scored": int(visual_embeddings.shape[0]),
        },
    }


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


@torch.no_grad()
def _forward_extract(
    language_model,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    num_layers: int,
    attn_layer: int,
    text_positions: torch.Tensor,
    visual_positions: torch.Tensor,
):
    hidden = _cast_tensor_for_module(
        inputs_embeds,
        language_model.layers[0].input_layernorm,
    )

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
        hidden = _cast_tensor_for_module(hidden, layer.input_layernorm)

        if layer_idx == attn_layer:
            hidden_normed = layer.input_layernorm(hidden)
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
            key_states = attn_module.k_proj(visual_hidden_for_k).view(
                visual_shape
            ).transpose(1, 2)
            value_states = attn_module.v_proj(visual_hidden_for_v).view(
                visual_shape
            ).transpose(1, 2)

            visual_cos, visual_sin = _slice_position_embeddings(
                position_embeddings,
                visual_positions,
            )
            visual_cos = visual_cos.to(dtype=key_states.dtype)
            visual_sin = visual_sin.to(dtype=key_states.dtype)
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
            query_states = attn_module.q_proj(text_hidden).view(
                text_shape
            ).transpose(1, 2)

            text_cos, text_sin = _slice_position_embeddings(
                position_embeddings,
                text_positions,
            )
            text_cos = text_cos.to(dtype=query_states.dtype)
            text_sin = text_sin.to(dtype=query_states.dtype)
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
    use_alpha: bool = True,
    use_deviation: bool = True,
    two_stage: bool = False,
    text_chunk_size: Optional[int] = 32,
):
    def patched_forward(
        self,
        input_ids=None,
        pixel_values=None,
        image_sizes=None,
        pixel_values_videos=None,
        image_sizes_videos=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        vision_feature_layer=None,
        vision_feature_select_strategy=None,
        vision_aspect_ratio=None,
        batch_num_images=None,
        use_cache=None,
        **kwargs,
    ):
        vision_feature_layer = (
            vision_feature_layer
            if vision_feature_layer is not None
            else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )
        vision_aspect_ratio = (
            vision_aspect_ratio
            if vision_aspect_ratio is not None
            else getattr(self.config, "vision_aspect_ratio", "anyres_max_9")
        )
        use_cache = (
            use_cache
            if use_cache is not None
            else getattr(self.language_model.config, "use_cache", True)
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_features = None
        n_image_tokens = None
        if pixel_values is not None:
            image_features = _extract_image_features_compat(
                self,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                vision_aspect_ratio=vision_aspect_ratio,
                batch_num_images=batch_num_images,
                **kwargs,
            )
            image_features = _concat_token_features(image_features).to(
                inputs_embeds.device,
                inputs_embeds.dtype,
            )
            special_image_mask, _ = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_features,
            )
            n_image_tokens = image_features.shape[0]
            inputs_embeds = inputs_embeds.masked_scatter(
                special_image_mask,
                image_features,
            )

        video_features = None
        n_video_tokens = None
        if pixel_values_videos is not None:
            video_features = _extract_video_features_compat(
                self,
                pixel_values=pixel_values_videos,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                **kwargs,
            )
            image_newline = (
                self.image_newline[None, None, :]
                .repeat(video_features.shape[0], 1, 1)
                .to(video_features.device)
            )
            video_features = torch.cat(
                (video_features, image_newline),
                dim=1,
            )
            video_features = video_features.flatten(0, 1).to(
                inputs_embeds.device,
                inputs_embeds.dtype,
            )
            _, special_video_mask = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                video_features=video_features,
            )
            n_video_tokens = video_features.shape[0]
            inputs_embeds = inputs_embeds.masked_scatter(
                special_video_mask,
                video_features,
            )

        is_prefill = (
            past_key_values is None
            or (
                hasattr(past_key_values, "get_seq_length")
                and past_key_values.get_seq_length() == 0
            )
        )

        if position_ids is None:
            position_ids = torch.arange(
                inputs_embeds.shape[1],
                device=inputs_embeds.device,
            ).unsqueeze(0)

        if is_prefill or not hasattr(self, "_fetp_last_sample_artifact"):
            self._fetp_last_sample_artifact = None

        if (
            input_ids is not None
            and inputs_embeds.shape[0] == 1
            and is_prefill
            and retention_ratio != 0
        ):
            if n_image_tokens is not None and n_video_tokens is None:
                visual_token_id = self.config.image_token_id
                n_visual_tokens = n_image_tokens
            elif n_video_tokens is not None and n_image_tokens is None:
                visual_token_id = self.config.video_token_id
                n_visual_tokens = n_video_tokens
            elif n_image_tokens is None and n_video_tokens is None:
                visual_token_id = None
                n_visual_tokens = None
            else:
                eval_logger.warning(
                    "FETP(LLaVA-OneVision): simultaneous image+video inputs "
                    "are not pruned; using the original forward path."
                )
                visual_token_id = None
                n_visual_tokens = None

            if n_visual_tokens is not None:
                visual_positions = torch.where(
                    input_ids[0] == visual_token_id
                )[0]
                if (
                    visual_positions.numel() > 0
                    and visual_positions.numel() == n_visual_tokens
                ):
                    original_input_ids = input_ids
                    original_inputs_embeds = inputs_embeds
                    original_position_ids = position_ids
                    original_attention_mask = attention_mask

                    try:
                        original_n_visual_tokens = n_visual_tokens
                        artifact_stage1_keep_local = None
                        scored_visual_embeddings = image_features
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
                            artifact_stage1_keep_local = stage1_keep
                            scored_visual_embeddings = image_features.index_select(
                                0,
                                stage1_keep,
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
                            position_ids = position_ids[:, keep_global_indices]
                            attention_mask = _slice_attention_mask(
                                attention_mask,
                                keep_global_indices,
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

                        if scoring_method == "full":
                            n_run = len(self.language_model.layers)
                            extract_at = target_layer
                            if extract_at < 0:
                                extract_at = n_run + extract_at
                        else:
                            n_run = min(
                                max(1, shallow_layers),
                                len(self.language_model.layers),
                            )
                            extract_at = target_layer
                            if extract_at < 0:
                                extract_at = n_run + extract_at
                            extract_at = min(extract_at, n_run - 1)

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
                            num_layers=n_run,
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
                            attention_only_scores = _compute_fes_scores_from_compact_inputs(
                                text_to_vis_logits=text_to_vis_logits,
                                visual_value_states=visual_value_states,
                                use_alpha=True,
                                use_deviation=False,
                                text_chunk_size=text_chunk_size,
                            )
                            _, top_indices = scores.topk(num_keep)
                            keep_visual_local = top_indices.sort().values
                            _, attention_only_top = attention_only_scores.topk(num_keep)
                            attention_only_keep_local = (
                                attention_only_top.sort().values
                            )
                            self._fetp_last_sample_artifact = _build_fes_sample_artifact(
                                question_text="",
                                visual_embeddings=scored_visual_embeddings,
                                visual_value_states=visual_value_states,
                                fetp_scores=scores,
                                attention_only_scores=attention_only_scores,
                                fetp_keep_local=keep_visual_local,
                                attention_only_keep_local=attention_only_keep_local,
                                stage1_keep_local=artifact_stage1_keep_local,
                                num_keep=num_keep,
                                target_layer=extract_at,
                                scoring_method=scoring_method,
                                n_visual_tokens_original=original_n_visual_tokens,
                                n_visual_tokens_after_stage1=(
                                    n_visual_tokens
                                    if artifact_stage1_keep_local is not None
                                    else None
                                ),
                            )
                        else:
                            eval_logger.warning(
                                "FETP(LLaVA-OneVision): attention extraction "
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
                            attention_only_keep_local = keep_visual_local
                            self._fetp_last_sample_artifact = None

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
                        position_ids = position_ids[:, keep_global_indices]
                        attention_mask = _slice_attention_mask(
                            attention_mask,
                            keep_global_indices,
                        )
                    except Exception as exc:
                        input_ids = original_input_ids
                        inputs_embeds = original_inputs_embeds
                        position_ids = original_position_ids
                        attention_mask = original_attention_mask
                        eval_logger.warning(
                            "FETP(LLaVA-OneVision): pruning skipped due to "
                            f"scoring failure: {exc}"
                        )
                else:
                    eval_logger.warning(
                        "FETP(LLaVA-OneVision): visual placeholder count does "
                        "not match extracted visual token count; skipping "
                        "pruning."
                    )

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        return LlavaOnevisionModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features,
            video_hidden_states=video_features,
        )

    return patched_forward


@register_model("llava_onevision_ours_v3")
class LlavaOnevisionOursV3(LlavaHfChat):
    is_simple = False

    def __init__(
        self,
        pretrained: str = "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        revision: str = "main",
        device: str = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "float16",
        batch_size: int = 1,
        trust_remote_code: Optional[bool] = False,
        attn_implementation: Optional[str] = None,
        device_map: str = "",
        chat_template: Optional[str] = None,
        use_cache: bool = True,
        max_frames_num: Optional[int] = 32,
        retention_ratio: float = 0.25,
        scoring_method: str = "full",
        shallow_layers: int = 4,
        target_layer: int = 15,
        use_alpha: bool = True,
        use_deviation: bool = True,
        two_stage: bool = True,
        text_chunk_size: Optional[int] = 32,
        stats_output_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            pretrained=pretrained,
            revision=revision,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
            device_map=device_map,
            chat_template=chat_template,
            use_cache=use_cache,
            max_frames_num=max_frames_num,
            **kwargs,
        )

        self.retention_ratio = retention_ratio
        self.stats_output_path = stats_output_path
        eval_logger.info(
            "[LlavaOnevisionOursV3 / FETP-v3] "
            f"retention_ratio={retention_ratio}, "
            f"scoring_method={scoring_method}, "
            f"shallow_layers={shallow_layers}, "
            f"target_layer={target_layer}, "
            f"use_alpha={use_alpha}, "
            f"use_deviation={use_deviation}, "
            f"two_stage={two_stage}, "
            f"text_chunk_size={text_chunk_size}, "
            f"stats_output_path={stats_output_path}"
        )

        LlavaOnevisionModel.forward = _make_fetp_forward(
            retention_ratio=retention_ratio,
            scoring_method=scoring_method,
            shallow_layers=shallow_layers,
            target_layer=target_layer,
            use_alpha=use_alpha,
            use_deviation=use_deviation,
            two_stage=two_stage,
            text_chunk_size=text_chunk_size,
        )

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

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
        pbar = tqdm(
            total=num_iters,
            disable=(self.rank != 0),
            desc="Model Responding",
        )
        e2e_latency = 0
        total_tokens = 0

        for chunk in chunks:
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            task_name = task[0]
            split_name = split[0]
            chat_messages = [
                doc_to_messages[0](self.task_dict[task_name][split_name][ids])
                for ids in doc_id
            ]
            chat_messages = [
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
            assert self.batch_size_per_gpu == 1, (
                "Do not support batch_size_per_gpu > 1 for now"
            )

            messages = chat_messages[0].model_dump()["messages"]
            text = self._image_processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            visuals, videos, image_sizes = _prepare_llava_media_inputs(
                visuals,
                videos,
            )
            images_kwargs, videos_kwargs = _build_llava_processor_kwargs(
                self.model.config,
                self.max_frames_num,
            )
            inputs = self._prepare_processor_inputs(
                self._image_processor(
                    images=visuals,
                    videos=videos,
                    text=text,
                    return_tensors="pt",
                    **images_kwargs,
                    **videos_kwargs,
                )
            )

            gen_kwargs = dict(all_gen_kwargs[0])
            gen_kwargs["image_sizes"] = image_sizes
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            do_sample = gen_kwargs["temperature"] > 0

            try:
                start_time = time.time()
                cont = self.model.generate(
                    **inputs,
                    do_sample=do_sample,
                    temperature=gen_kwargs["temperature"] if do_sample else None,
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                    pad_token_id=self.eot_token_id,
                    eos_token_id=self.eot_token_id,
                )
                end_time = time.time()
                cont = cont[:, inputs["input_ids"].shape[-1] :]
                e2e_latency += end_time - start_time
                total_tokens += cont.shape[-1] if len(cont.shape) > 1 else len(cont)
            except Exception as error:
                eval_logger.error(f"Error {error} in generating")
                cont = ""

            sample_artifact = getattr(
                self.model.model,
                "_fetp_last_sample_artifact",
                None,
            )
            text_outputs = (
                self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
                if cont != ""
                else ""
            )

            res.append(text_outputs)
            self.cache_hook.add_partial("generate_until", (text, gen_kwargs), text_outputs)

            if sample_artifact is not None and len(chunk) == 1:
                artifact_to_write = dict(sample_artifact)
                artifact_to_write["task_name"] = task_name
                artifact_to_write["doc_id"] = doc_id[0]
                artifact_to_write["question_text"] = text
                artifact_to_write["model_response"] = text_outputs
                artifact_to_write = attach_visual_compare_metadata(
                    artifact_to_write,
                    build_visual_compare_metadata(
                        image_inputs=visuals,
                        video_inputs=videos,
                        model_config=self.model.config,
                        n_visual_tokens_scored=artifact_to_write["metadata"][
                            "n_visual_tokens_scored"
                        ],
                        vision_aspect_ratio=getattr(
                            self.model.config,
                            "vision_aspect_ratio",
                            "anyres_max_9",
                        ),
                        stage1_keep_local=artifact_to_write["selection"].get(
                            "stage1_keep_local"
                        ),
                    ),
                    target_layer=artifact_to_write["metadata"].get("target_layer"),
                )
                artifact_path = _write_sample_artifact(
                    stats_output_path=self.stats_output_path,
                    method_name="fetp",
                    task_name=task_name,
                    doc_id=doc_id[0],
                    artifact=artifact_to_write,
                )
                if artifact_path:
                    eval_logger.info(
                        f"[LLaVA-OneVision / FETP-v3] wrote sample artifact to {artifact_path}"
                    )

            self.model.model._fetp_last_sample_artifact = None
            pbar.update(1)

        res = re_ords.get_original(res)
        metric_dict = {
            "total_tokens": total_tokens,
            "e2e_latency": e2e_latency,
            "avg_speed": total_tokens / e2e_latency if e2e_latency > 0 else 0,
            "additional_metrics": {
                "rank": self.rank,
            },
        }
        log_metrics(**metric_dict)

        pbar.close()
        return res
