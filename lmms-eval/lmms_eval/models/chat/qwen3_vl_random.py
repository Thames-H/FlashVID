"""Random visual token pruning baseline for Qwen3-VL.

This mirrors the patched-forward integration pattern from
``qwen2_5_vl_random.py`` while reusing the Qwen3-VL multimodal handling
already validated in ``qwen3_vl_ours_v2.py``.
"""

from typing import Optional, Union

import torch
from loguru import logger as eval_logger
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModel,
    Qwen3VLModelOutputWithPast,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, is_torchdynamo_compiling

from lmms_eval.api.registry import register_model
from lmms_eval.models.chat.qwen3_vl_ours_v2 import (
    Qwen3_VL_Ours_V2,
    _merge_visual_inputs,
    _slice_attention_mask,
    _unpack_visual_outputs,
)
from lmms_eval.models.simple.qwen3_vl import Qwen3_VL as Qwen3_VLSimple


def _sample_random_visual_positions(
    visual_positions: torch.Tensor,
    num_keep: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if visual_positions.numel() == 0:
        return visual_positions

    num_keep = max(1, min(int(num_keep), int(visual_positions.numel())))
    sampled_indices = torch.randperm(
        visual_positions.numel(),
        device=visual_positions.device,
        generator=generator,
    )[:num_keep]
    return visual_positions[sampled_indices].sort().values


def _make_random_forward(retention_ratio: float):
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
        n_image_tokens = 0
        n_video_tokens = 0

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
            if (
                attention_mask_tensor is not None
                and attention_mask_tensor.ndim == 4
            ):
                attention_mask_tensor = torch.diagonal(
                    attention_mask_tensor[:, 0], dim1=1, dim2=2
                )
                if attention_mask_tensor.dtype.is_floating_point:
                    min_value = torch.finfo(
                        attention_mask_tensor.dtype
                    ).min
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
                    (cache_position[0] + self.rope_deltas).to(
                        inputs_embeds.device
                    )
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(
                    seq_length, device=inputs_embeds.device
                )
                position_ids = position_ids.view(1, -1).expand(
                    batch_size, -1
                )
                if cache_position is not None:
                    delta = delta.repeat_interleave(
                        batch_size // delta.shape[0], dim=0
                    )
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        n_visual_tokens = n_image_tokens + n_video_tokens
        visual_token_ids = []
        if n_image_tokens:
            visual_token_ids.append(self.config.image_token_id)
        if n_video_tokens:
            visual_token_ids.append(self.config.video_token_id)

        if (
            n_visual_tokens > 0
            and position_ids.shape[-1] > 1
            and retention_ratio < 1.0
            and inputs_embeds.shape[0] == 1
            and input_ids is not None
            and visual_token_ids
        ):
            visual_token_ids_tensor = torch.tensor(
                visual_token_ids,
                device=input_ids.device,
            )
            visual_token_mask = torch.isin(
                input_ids[0], visual_token_ids_tensor
            )
            visual_positions = torch.where(visual_token_mask)[0]

            if visual_positions.numel() == n_visual_tokens:
                num_keep = max(1, int(n_visual_tokens * retention_ratio))

                keep_visual_positions = _sample_random_visual_positions(
                    visual_positions,
                    num_keep=num_keep,
                )
                non_visual_positions = torch.where(~visual_token_mask)[0]
                keep_global_indices = torch.cat(
                    [non_visual_positions, keep_visual_positions],
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
                    visual_pos_masks = visual_pos_masks[
                        :, keep_global_indices
                    ]
                    if deepstack_visual_embeds is not None:
                        deepstack_visual_embeds = [
                            embed[joint_visual_keep_indices]
                            for embed in deepstack_visual_embeds
                        ]
            else:
                eval_logger.warning(
                    "Random(Qwen3-VL): visual placeholder count does not "
                    "match the extracted visual token count, skipping pruning."
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


@register_model("qwen3_vl_random")
class Qwen3_VL_Random(Qwen3_VL_Ours_V2):
    """Qwen3-VL with random visual token retention."""

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
        **kwargs,
    ) -> None:
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        Qwen3_VLSimple.__init__(
            self,
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
            f"[Qwen3_VL_Random] retention_ratio={retention_ratio}"
        )

        Qwen3VLModel.forward = _make_random_forward(
            retention_ratio=retention_ratio
        )
