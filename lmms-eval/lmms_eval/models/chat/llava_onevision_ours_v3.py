"""FETP-v3 for LLaVA-OneVision (HF format)."""

from typing import Optional, Union

import torch
import torch.nn.functional as F
from loguru import logger as eval_logger
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.models.llava_onevision.modeling_llava_onevision import (
    LlavaOnevisionModel,
    LlavaOnevisionModelOutputWithPast,
)
from transformers.models.qwen2.modeling_qwen2 import (
    apply_rotary_pos_emb,
    repeat_kv,
)

from lmms_eval.api.registry import register_model

from .llava_hf import LlavaHf as LlavaHfChat


def _slice_position_embeddings(position_embeddings, positions: torch.Tensor):
    cos, sin = position_embeddings
    return cos.index_select(1, positions), sin.index_select(1, positions)


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
def _compute_fes_scores_from_compact_inputs(
    text_to_vis_logits: torch.Tensor,
    visual_value_states: torch.Tensor,
    use_alpha: bool = True,
    use_deviation: bool = True,
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

    pooled_per_text = alpha_per_text @ vis_values
    diff = vis_values.unsqueeze(0) - pooled_per_text.unsqueeze(1)
    deviation_sq = diff.norm(dim=-1).pow(2)
    scores = (alpha_per_text.pow(2) * deviation_sq).mean(dim=0).sqrt()

    if use_alpha and use_deviation:
        return scores

    alpha_mean = alpha_per_text.mean(dim=0)
    deviation_mean = deviation_sq.sqrt().mean(dim=0)
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
    hidden = inputs_embeds

    if not isinstance(causal_mask_mapping := attention_mask, dict):
        mask_kwargs = {
            "config": language_model.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": None,
            "past_key_values": None,
            "position_ids": position_ids,
        }
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
        }
        if getattr(language_model, "has_sliding_layers", False):
            causal_mask_mapping["sliding_attention"] = (
                create_sliding_window_causal_mask(**mask_kwargs)
            )

    position_embeddings = language_model.rotary_emb(hidden, position_ids)

    for layer_idx in range(min(num_layers, len(language_model.layers))):
        layer = language_model.layers[layer_idx]

        if layer_idx == attn_layer:
            hidden_normed = layer.input_layernorm(hidden)
            attn_module = layer.self_attn
            bsz = hidden_normed.size(0)
            n_vis = visual_positions.numel()

            visual_hidden = hidden_normed.index_select(1, visual_positions)
            visual_shape = (bsz, n_vis, -1, attn_module.head_dim)
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
                    bsz,
                    attn_module.num_heads,
                    0,
                    n_vis,
                )
                return empty_logits, value_states.float()

            text_hidden = hidden_normed.index_select(1, text_positions)
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
            image_features = self.get_image_features(
                pixel_values,
                image_sizes,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                vision_aspect_ratio=vision_aspect_ratio,
                batch_num_images=batch_num_images,
                return_dict=True,
            ).pooler_output
            image_features = torch.cat(image_features, dim=0).to(
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
            video_features = self.get_video_features(
                pixel_values_videos,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                return_dict=True,
            ).pooler_output
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
                    if retention_ratio < 1.0:
                        num_keep = max(
                            1,
                            int(n_visual_tokens * retention_ratio),
                        )
                    else:
                        num_keep = max(
                            1,
                            min(int(retention_ratio), n_visual_tokens),
                        )

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
                        )
                        _, top_indices = scores.topk(num_keep)
                        keep_visual_local = top_indices.sort().values
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
        dtype: Optional[Union[str, torch.dtype]] = "auto",
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
        eval_logger.info(
            "[LlavaOnevisionOursV3 / FETP-v3] "
            f"retention_ratio={retention_ratio}, "
            f"scoring_method={scoring_method}, "
            f"shallow_layers={shallow_layers}, "
            f"target_layer={target_layer}, "
            f"use_alpha={use_alpha}, "
            f"use_deviation={use_deviation}"
        )

        LlavaOnevisionModel.forward = _make_fetp_forward(
            retention_ratio=retention_ratio,
            scoring_method=scoring_method,
            shallow_layers=shallow_layers,
            target_layer=target_layer,
            use_alpha=use_alpha,
            use_deviation=use_deviation,
        )
