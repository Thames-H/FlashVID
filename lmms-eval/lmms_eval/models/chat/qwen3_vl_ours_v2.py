"""Functionally Equivalent Token Pruning (FETP) for Qwen3-VL.

This mirrors the pruning flow from ``qwen2_5_vl_ours_v2.py`` but adapts the
patched forward path to Qwen3-VL's DeepStack-based multimodal stack.
"""

import time
from typing import List, Optional, Union

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

process_vision_info, _has_qwen_vl = optional_import(
    "qwen_vl_utils", "process_vision_info"
)
if not _has_qwen_vl:
    eval_logger.warning(
        "Failed to import qwen_vl_utils; "
        "Please install it via `pip install qwen-vl-utils`"
    )


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
        alpha = torch.ones(n_vis, device=visual_value_states.device) / n_vis
    else:
        text_to_vis_alpha = F.softmax(text_to_vis_logits.float(), dim=-1)
        alpha = text_to_vis_alpha.mean(dim=1).mean(dim=1)[0]

    vis_values = visual_value_states[0].float()
    vis_values = vis_values.permute(1, 0, 2).reshape(n_vis, -1)
    pooled_value = (alpha.unsqueeze(-1) * vis_values).sum(dim=0)
    deviation = (vis_values - pooled_value.unsqueeze(0)).norm(dim=-1)

    if use_alpha and use_deviation:
        return alpha * deviation
    if use_alpha:
        return alpha
    if use_deviation:
        return deviation
    return torch.ones(n_vis, device=visual_value_states.device)


@torch.no_grad()
def _compute_fes_scores(
    attn_logits: torch.Tensor,
    value_states: torch.Tensor,
    visual_positions: torch.Tensor,
    text_positions: torch.Tensor,
    use_alpha: bool = True,
    use_deviation: bool = True,
) -> torch.Tensor:
    compact_logits = attn_logits.index_select(2, text_positions).index_select(
        3, visual_positions
    )
    compact_values = value_states.index_select(2, visual_positions)
    return _compute_fes_scores_from_compact_inputs(
        text_to_vis_logits=compact_logits,
        visual_value_states=compact_values,
        use_alpha=use_alpha,
        use_deviation=use_deviation,
    )


def _get_suffix_text_positions(
    input_ids: torch.Tensor,
    visual_positions: torch.Tensor,
    config,
) -> torch.Tensor:
    visual_related_token_ids = torch.tensor(
        [
            config.vision_start_token_id,
            config.vision_end_token_id,
            config.image_token_id,
            config.video_token_id,
        ],
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


def _flatten_visual_tensor(hidden_states, merger=None):
    if hidden_states is None:
        return None
    if isinstance(hidden_states, (list, tuple)):
        parts = [
            _flatten_visual_tensor(part, merger=merger)
            for part in hidden_states
        ]
        parts = [part for part in parts if part is not None]
        if not parts:
            return None
        return torch.cat(parts, dim=0)
    if hidden_states.ndim == 3:
        if merger is not None:
            return merger(hidden_states)
        return hidden_states.reshape(-1, hidden_states.shape[-1])
    return hidden_states


def _unpack_visual_outputs(outputs, merger=None):
    last_hidden_state = getattr(outputs, "last_hidden_state", None)
    deepstack_features = getattr(outputs, "deepstack_features", None)

    if last_hidden_state is None and isinstance(outputs, (tuple, list)):
        if outputs:
            last_hidden_state = outputs[0]
        if len(outputs) > 1:
            deepstack_features = outputs[-1]

    last_hidden_state = _flatten_visual_tensor(
        last_hidden_state, merger=merger
    )
    if deepstack_features is not None:
        deepstack_features = [
            _flatten_visual_tensor(feature, merger=merger)
            for feature in deepstack_features
        ]

    return last_hidden_state, deepstack_features


def _merge_visual_inputs(
    image_mask,
    video_mask,
    deepstack_image_embeds,
    deepstack_video_embeds,
):
    visual_pos_masks = None
    deepstack_visual_embeds = None

    if image_mask is not None:
        image_mask = image_mask[..., 0]
    if video_mask is not None:
        video_mask = video_mask[..., 0]

    if image_mask is not None and video_mask is not None:
        visual_pos_masks = image_mask | video_mask
        deepstack_visual_embeds = []
        image_mask_joint = image_mask[visual_pos_masks]
        video_mask_joint = video_mask[visual_pos_masks]

        for img_embed, vid_embed in zip(
            deepstack_image_embeds or [], deepstack_video_embeds or []
        ):
            merged = img_embed.new_zeros(
                (int(visual_pos_masks.sum().item()), img_embed.shape[-1])
            )
            merged[image_mask_joint, :] = img_embed
            merged[video_mask_joint, :] = vid_embed
            deepstack_visual_embeds.append(merged)
    elif image_mask is not None:
        visual_pos_masks = image_mask
        deepstack_visual_embeds = deepstack_image_embeds
    elif video_mask is not None:
        visual_pos_masks = video_mask
        deepstack_visual_embeds = deepstack_video_embeds

    return visual_pos_masks, deepstack_visual_embeds


def _slice_attention_mask(attention_mask, keep_indices):
    if attention_mask is None:
        return None

    if isinstance(attention_mask, dict):
        pruned = {}
        for key, value in attention_mask.items():
            pruned[key] = _slice_attention_mask(value, keep_indices)
        return pruned

    if attention_mask.ndim == 2:
        return attention_mask[:, keep_indices]

    if attention_mask.ndim == 4 and attention_mask.shape[-1] == attention_mask.shape[-2]:
        return attention_mask[:, :, keep_indices][:, :, :, keep_indices]

    return attention_mask


def _slice_position_embeddings(position_embeddings, positions: torch.Tensor):
    cos, sin = position_embeddings
    if positions.numel() == 0:
        return cos[:, :0, :], sin[:, :0, :]
    return cos.index_select(1, positions), sin.index_select(1, positions)


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
) -> tuple:
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

    for layer_idx in range(num_layers):
        layer = language_model.layers[layer_idx]

        if layer_idx == attn_layer:
            residual = hidden
            hidden_normed = layer.input_layernorm(hidden)

            attn_module = layer.self_attn
            bsz = hidden_normed.size(0)
            n_vis = visual_positions.numel()

            visual_hidden = hidden_normed.index_select(1, visual_positions)
            visual_shape = (bsz, n_vis, -1, attn_module.head_dim)
            key_states = attn_module.k_proj(visual_hidden).view(visual_shape)
            value_states = attn_module.v_proj(visual_hidden).view(visual_shape)

            key_states = attn_module.k_norm(key_states).transpose(1, 2)
            visual_value_states = value_states.transpose(1, 2)

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
                text_to_vis_logits = hidden.new_empty(
                    bsz,
                    attn_module.num_heads,
                    0,
                    n_vis,
                )
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

                text_to_vis_logits = torch.matmul(
                    query_states, key_states_expanded.transpose(2, 3)
                ) * attn_module.scaling

            return text_to_vis_logits, visual_value_states
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
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

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
                image_outputs, merger=self.visual.merger
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
                video_outputs, merger=self.visual.merger
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
            visual_positions = torch.where(input_ids[0] == visual_token_id)[0]
            if visual_positions.numel() > 0 and visual_positions.numel() == n_visual_tokens:
                if retention_ratio < 1.0:
                    num_keep = max(1, int(n_visual_tokens * retention_ratio))
                else:
                    num_keep = max(
                        1, min(int(retention_ratio), n_visual_tokens)
                    )

                if scoring_method == "full":
                    n_run = len(self.language_model.layers)
                    extract_at = target_layer
                else:
                    n_run = shallow_layers
                    extract_at = min(target_layer, shallow_layers - 1)

                text_positions = _get_suffix_text_positions(
                    input_ids[0],
                    visual_positions,
                    self.config,
                )

                text_to_vis_logits, visual_value_states = _forward_extract(
                    self.language_model,
                    inputs_embeds,
                    position_ids,
                    cache_position,
                    num_layers=n_run,
                    attn_layer=extract_at,
                    text_positions=text_positions,
                    visual_positions=visual_positions,
                )

                if text_to_vis_logits is not None and visual_value_states is not None:
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
                        "FETP(Qwen3-VL): attention extraction failed, "
                        "falling back to uniform selection."
                    )
                    step = n_visual_tokens / num_keep
                    keep_visual_local = (
                        torch.arange(num_keep, device=inputs_embeds.device)
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
                    inputs_embeds.shape[0], -1, hidden_size
                )
                inputs_embeds = torch.gather(
                    inputs_embeds, dim=1, index=gather_index
                )
                position_ids = position_ids[:, :, keep_global_indices]
                attention_mask = _slice_attention_mask(
                    attention_mask, keep_global_indices
                )
                cache_position = cache_position[keep_global_indices]

                if visual_pos_masks is not None:
                    original_visual_positions = torch.where(
                        visual_pos_masks[0]
                    )[0]
                    joint_visual_keep_indices = torch.where(
                        torch.isin(
                            original_visual_positions, keep_global_indices
                        )
                    )[0]
                    visual_pos_masks = visual_pos_masks[:, keep_global_indices]
                    if deepstack_visual_embeds is not None:
                        deepstack_visual_embeds = [
                            embed[joint_visual_keep_indices]
                            for embed in deepstack_visual_embeds
                        ]
            else:
                eval_logger.warning(
                    "FETP(Qwen3-VL): visual placeholder count does not match "
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


@register_model("qwen3_vl_ours_v2")
class Qwen3_VL_Ours_V2(Qwen3_VLSimple):
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
        eval_logger.info(
            f"[Qwen3_VL_Ours_V2 / FETP] "
            f"retention_ratio={retention_ratio}, "
            f"scoring_method={scoring_method}, "
            f"shallow_layers={shallow_layers}, "
            f"target_layer={target_layer}, "
            f"use_alpha={use_alpha}, "
            f"use_deviation={use_deviation}"
        )

        Qwen3VLModel.forward = _make_fetp_forward(
            retention_ratio=retention_ratio,
            scoring_method=scoring_method,
            shallow_layers=shallow_layers,
            target_layer=target_layer,
            use_alpha=use_alpha,
            use_deviation=use_deviation,
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

            # Hugging Face's Qwen3-VL docs explicitly drop token_type_ids
            # before generation.
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