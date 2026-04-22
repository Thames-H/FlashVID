# MMTok: Multimodal Coverage Maximization for Efficient Inference of VLMs
# Paper: https://arxiv.org/abs/2508.18264

import math
from typing import Optional, Union

import torch
import torch.nn as nn
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModel,
    Qwen3VLModelOutputWithPast,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, is_torchdynamo_compiling


def _cpu_tensor(tensor: Optional[torch.Tensor]):
    if tensor is None:
        return None
    return tensor.detach().cpu()


def _flatten_visual_tensor(hidden_states):
    if hidden_states is None:
        return None
    if isinstance(hidden_states, (list, tuple)):
        parts = [_flatten_visual_tensor(part) for part in hidden_states]
        parts = [part for part in parts if part is not None]
        if not parts:
            return None
        return torch.cat(parts, dim=0)
    if hidden_states.ndim == 3:
        return hidden_states.reshape(-1, hidden_states.shape[-1])
    return hidden_states


def _unpack_qwen3_visual_outputs(outputs):
    final_embeds = None
    selection_features = None
    deepstack_features = None

    if isinstance(outputs, (tuple, list)):
        if outputs:
            final_embeds = outputs[0]
        if len(outputs) > 1:
            selection_features = outputs[1]
        if len(outputs) > 2:
            deepstack_features = outputs[2]
    else:
        final_embeds = getattr(outputs, "pooler_output", None)
        selection_features = getattr(outputs, "selection_features", None)
        deepstack_features = getattr(outputs, "deepstack_features", None)

    final_embeds = _flatten_visual_tensor(final_embeds)
    selection_features = _flatten_visual_tensor(selection_features)
    if deepstack_features is not None:
        deepstack_features = [
            _flatten_visual_tensor(feature) for feature in deepstack_features
        ]

    return final_embeds, selection_features, deepstack_features


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

    if (
        attention_mask.ndim == 4
        and attention_mask.shape[-1] == attention_mask.shape[-2]
    ):
        return attention_mask[:, :, keep_indices][:, :, :, keep_indices]

    return attention_mask


def _filter_deepstack_by_sequence_indices(
    visual_pos_masks: Optional[torch.Tensor],
    keep_global_indices: torch.LongTensor,
    deepstack_visual_embeds: Optional[list[torch.Tensor]],
):
    if visual_pos_masks is None:
        return visual_pos_masks, deepstack_visual_embeds

    original_visual_positions = torch.where(visual_pos_masks[0])[0]
    joint_visual_keep_indices = torch.where(
        torch.isin(original_visual_positions, keep_global_indices)
    )[0]
    pruned_mask = visual_pos_masks[:, keep_global_indices]

    if deepstack_visual_embeds is None:
        return pruned_mask, deepstack_visual_embeds

    pruned_deepstack = [
        embed[joint_visual_keep_indices] for embed in deepstack_visual_embeds
    ]
    return pruned_mask, pruned_deepstack


def _compute_target_vision_tokens(num_visual_tokens: int, retain_ratio: float) -> int:
    if num_visual_tokens <= 0:
        return 0
    if retain_ratio <= 0:
        return 0
    if retain_ratio >= 1:
        return num_visual_tokens
    return max(1, min(num_visual_tokens, math.ceil(num_visual_tokens * retain_ratio)))


def _build_keep_indices(
    input_ids: torch.Tensor,
    image_keep_local: Optional[torch.LongTensor],
    video_keep_local: Optional[torch.LongTensor],
    config,
):
    non_visual_positions = torch.where(
        (input_ids != config.image_token_id)
        & (input_ids != config.video_token_id)
    )[0]

    keep_parts = [non_visual_positions]

    if image_keep_local is not None:
        image_positions = torch.where(input_ids == config.image_token_id)[0]
        keep_parts.append(image_positions[image_keep_local])

    if video_keep_local is not None:
        video_positions = torch.where(input_ids == config.video_token_id)[0]
        keep_parts.append(video_positions[video_keep_local])

    return torch.cat(keep_parts, dim=0).sort().values


class Qwen3_VL_MMTok(nn.Module):
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: Optional[torch.LongTensor] = None,
    ):
        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        video_embeds, selection_features, deepstack_video_embeds = self.visual(
            pixel_values_videos,
            grid_thw=video_grid_thw,
        )
        split_sizes = (
            video_grid_thw.prod(-1) // self.visual.spatial_merge_size**2
        ).tolist()
        video_embeds = torch.split(video_embeds, split_sizes)
        selection_features = torch.split(selection_features, split_sizes)
        return video_embeds, selection_features, deepstack_video_embeds

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
    ):
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds, selection_features, deepstack_image_embeds = self.visual(
            pixel_values,
            grid_thw=image_grid_thw,
        )
        split_sizes = (
            image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2
        ).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        selection_features = torch.split(selection_features, split_sizes)
        return image_embeds, selection_features, deepstack_image_embeds

    def forward(
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
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLModelOutputWithPast]:
        self._mmtok_last_sample_artifact = None
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

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
        image_embeds = None
        video_embeds = None
        image_selection_features = None
        video_selection_features = None

        if pixel_values is not None:
            image_outputs = self.get_image_features(pixel_values, image_grid_thw)
            (
                image_embeds,
                image_selection_features,
                deepstack_image_embeds,
            ) = _unpack_qwen3_visual_outputs(image_outputs)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            image_selection_features = image_selection_features.to(
                inputs_embeds.device,
                inputs_embeds.dtype,
            )
            image_mask, _ = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_outputs = self.get_video_features(
                pixel_values_videos, video_grid_thw
            )
            (
                video_embeds,
                video_selection_features,
                deepstack_video_embeds,
            ) = _unpack_qwen3_visual_outputs(video_outputs)
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            video_selection_features = video_selection_features.to(
                inputs_embeds.device,
                inputs_embeds.dtype,
            )
            _, video_mask = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                video_features=video_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

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
                    attention_mask_tensor[:, 0],
                    dim1=1,
                    dim2=2,
                )
                if attention_mask_tensor.dtype.is_floating_point:
                    min_value = torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = attention_mask_tensor / min_value
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
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
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(
                        batch_size // delta.shape[0], dim=0
                    )
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        if (
            input_ids is not None
            and inputs_embeds.shape[0] == 1
            and position_ids.shape[-1] > 1
            and hasattr(self, "_mmtok_core")
        ):
            question = ""
            if hasattr(self, "get_question"):
                question = self.get_question() or ""

            image_keep_local = None
            selected_image_embeds = None
            if image_embeds is not None and image_selection_features is not None:
                target_vision_tokens = _compute_target_vision_tokens(
                    image_embeds.shape[0],
                    getattr(self._mmtok_core, "retain_ratio", 1.0),
                )
                if target_vision_tokens == 0:
                    image_keep_local = torch.empty(
                        0,
                        device=inputs_embeds.device,
                        dtype=torch.long,
                    )
                    selected_image_embeds = image_embeds[:0]
                elif target_vision_tokens < image_embeds.shape[0]:
                    image_keep_local, selected_image_embeds = (
                        self._mmtok_core.apply_selection_preprocess_qwen(
                            image_embeds=image_embeds,
                            image_features=image_selection_features,
                            question_text=question,
                            target_vision_tokens=target_vision_tokens,
                        )
                    )
                    image_keep_local = torch.tensor(
                        image_keep_local,
                        device=inputs_embeds.device,
                        dtype=torch.long,
                    )
                    selected_image_embeds = selected_image_embeds.to(
                        inputs_embeds.device, inputs_embeds.dtype
                    )
                else:
                    image_keep_local = torch.arange(
                        image_embeds.shape[0],
                        device=inputs_embeds.device,
                        dtype=torch.long,
                    )
                    selected_image_embeds = image_embeds

            video_keep_local = None
            selected_video_embeds = None
            if video_embeds is not None and video_selection_features is not None:
                target_vision_tokens = _compute_target_vision_tokens(
                    video_embeds.shape[0],
                    getattr(self._mmtok_core, "retain_ratio", 1.0),
                )
                if target_vision_tokens == 0:
                    video_keep_local = torch.empty(
                        0,
                        device=inputs_embeds.device,
                        dtype=torch.long,
                    )
                    selected_video_embeds = video_embeds[:0]
                elif target_vision_tokens < video_embeds.shape[0]:
                    video_keep_local, selected_video_embeds = (
                        self._mmtok_core.apply_selection_preprocess_qwen(
                            image_embeds=video_embeds,
                            image_features=video_selection_features,
                            question_text=question,
                            target_vision_tokens=target_vision_tokens,
                        )
                    )
                    video_keep_local = torch.tensor(
                        video_keep_local,
                        device=inputs_embeds.device,
                        dtype=torch.long,
                    )
                    selected_video_embeds = selected_video_embeds.to(
                        inputs_embeds.device, inputs_embeds.dtype
                    )
                else:
                    video_keep_local = torch.arange(
                        video_embeds.shape[0],
                        device=inputs_embeds.device,
                        dtype=torch.long,
                    )
                    selected_video_embeds = video_embeds

            if image_keep_local is not None or video_keep_local is not None:
                artifact_visual_embeds = None
                artifact_keep_local = None
                artifact_initial_gain = None
                artifact_combined_rows = None

                if image_keep_local is not None:
                    image_positions = torch.where(
                        input_ids[0] == self.config.image_token_id
                    )[0]
                    if image_positions.numel() != image_embeds.shape[0]:
                        raise ValueError(
                            "Image placeholder count does not match image embeds."
                    )
                    inputs_embeds[:, image_positions[image_keep_local]] = (
                        selected_image_embeds
                    )
                    if video_keep_local is None:
                        selection_info = getattr(
                            self._mmtok_core.token_selector,
                            "last_selection_info",
                            {},
                        )
                        artifact_visual_embeds = image_embeds
                        artifact_keep_local = image_keep_local
                        artifact_initial_gain = selection_info.get(
                            "initial_marginal_gain"
                        )
                        artifact_combined_rows = selection_info.get(
                            "combined_rows"
                        )

                if video_keep_local is not None:
                    video_positions = torch.where(
                        input_ids[0] == self.config.video_token_id
                    )[0]
                    if video_positions.numel() != video_embeds.shape[0]:
                        raise ValueError(
                            "Video placeholder count does not match video embeds."
                    )
                    inputs_embeds[:, video_positions[video_keep_local]] = (
                        selected_video_embeds
                    )
                    if image_keep_local is None:
                        selection_info = getattr(
                            self._mmtok_core.token_selector,
                            "last_selection_info",
                            {},
                        )
                        artifact_visual_embeds = video_embeds
                        artifact_keep_local = video_keep_local
                        artifact_initial_gain = selection_info.get(
                            "initial_marginal_gain"
                        )
                        artifact_combined_rows = selection_info.get(
                            "combined_rows"
                        )

                keep_global_indices = _build_keep_indices(
                    input_ids[0],
                    image_keep_local,
                    video_keep_local,
                    self.config,
                )
                hidden_size = inputs_embeds.shape[-1]
                gather_index = keep_global_indices.view(1, -1, 1).expand(
                    inputs_embeds.shape[0], -1, hidden_size
                )
                inputs_embeds = torch.gather(
                    inputs_embeds,
                    dim=1,
                    index=gather_index,
                )
                position_ids = position_ids[:, :, keep_global_indices]
                attention_mask = _slice_attention_mask(
                    attention_mask,
                    keep_global_indices,
                )
                cache_position = cache_position[keep_global_indices]
                (
                    visual_pos_masks,
                    deepstack_visual_embeds,
                ) = _filter_deepstack_by_sequence_indices(
                    visual_pos_masks,
                    keep_global_indices,
                    deepstack_visual_embeds,
                )

                if artifact_visual_embeds is not None and artifact_keep_local is not None:
                    if artifact_initial_gain is None:
                        artifact_initial_gain = torch.ones(
                            artifact_visual_embeds.shape[0],
                            dtype=torch.float32,
                        )
                    self._mmtok_last_sample_artifact = {
                        "method": "mmtok",
                        "question_text": question,
                        "visual_embeddings": _cpu_tensor(
                            artifact_visual_embeds.float()
                        ),
                        "scores": {
                            "initial_marginal_gain": _cpu_tensor(
                                artifact_initial_gain.float()
                            ),
                        },
                        "selection": {
                            "mmtok_keep_local": _cpu_tensor(
                                artifact_keep_local.long()
                            ),
                            "num_keep": int(artifact_keep_local.numel()),
                        },
                        "metadata": {
                            "combined_rows": int(artifact_combined_rows)
                            if artifact_combined_rows is not None
                            else None,
                            "n_visual_tokens_scored": int(
                                artifact_visual_embeds.shape[0]
                            ),
                        },
                    }

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
