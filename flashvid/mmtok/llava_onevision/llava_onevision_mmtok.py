# MMTok: Multimodal Coverage Maximization for Efficient Inference of VLMs
# Paper: https://arxiv.org/abs/2508.18264

import inspect
import types
from typing import Optional, Union

import torch
import torch.nn as nn
from loguru import logger as eval_logger
from transformers.models.llava_onevision.modeling_llava_onevision import (
    LlavaOnevisionModel,
    LlavaOnevisionModelOutputWithPast,
    image_size_to_num_patches,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from ..core import MMTokCore
from ..core.adapter_utils import (
    concat_token_features,
    compute_target_vision_tokens,
    extract_question_from_messages,
    gather_sequence_hidden_states,
    slice_attention_mask,
    slice_position_ids,
)


def mmtok_llava_onevision(
    llava_model,
    language_tokenizer=None,
    processor=None,
    retain_ratio=0.2,
    target_vision_tokens: Optional[int] = None,
    **mmtok_kwargs,
):
    mmtok_config = {
        "alpha": 0.5,
        "softmax_tv_temperature": 0.01,
        "softmax_vv_temperature": 0.2,
        "device": llava_model.device,
        "remove_padding_indices": False,
        **mmtok_kwargs,
    }

    eval_logger.info(
        f"[MMTok-LLaVA-OneVision] Injecting MMTok: retain_ratio={retain_ratio}, "
        f"target_vision_tokens={target_vision_tokens}, device={mmtok_config['device']}"
    )
    mmtok_core = MMTokCore(**mmtok_config)
    mmtok_core.retain_ratio = retain_ratio
    mmtok_core.target_vision_tokens = target_vision_tokens
    mmtok_core._main_model_embed_tokens = llava_model.get_input_embeddings()
    mmtok_core._language_tokenizer = language_tokenizer

    llava_model.model._mmtok_core = mmtok_core
    llava_model.model._question_for_vision = None
    llava_model.set_question = types.MethodType(_set_question, llava_model)
    llava_model.model.get_question = types.MethodType(_get_question, llava_model.model)
    llava_model.model.forward = types.MethodType(
        LlavaOnevision_MMTok.forward,
        llava_model.model,
    )

    if processor is not None:
        patch_processor_for_question_hook(processor, llava_model)
    else:
        eval_logger.warning(
            "[MMTok-LLaVA-OneVision] No processor provided, skipping question hook patch"
        )

    return llava_model, processor


def _set_question(self, question: str):
    self.model._question_for_vision = question


def _get_question(self):
    return self._question_for_vision


def patch_processor_for_question_hook(processor, mmtok_model_instance):
    original_apply_chat_template = processor.apply_chat_template

    def patched_apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    ):
        question_text = extract_question_from_messages(messages)
        if question_text:
            mmtok_model_instance.set_question(question_text)
        return original_apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )

    processor.apply_chat_template = patched_apply_chat_template


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


def _get_module_dtype(module) -> Optional[torch.dtype]:
    for parameter in module.parameters():
        return parameter.dtype
    for buffer in module.buffers():
        return buffer.dtype
    return None


def _cast_tensor_for_module(tensor: torch.Tensor, module) -> torch.Tensor:
    target_dtype = _get_module_dtype(module)
    if target_dtype is None or tensor.dtype == target_dtype:
        return tensor
    return tensor.to(dtype=target_dtype)


def _concat_token_features(features):
    return concat_token_features(features)


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
    vision_tower_kwargs = _filter_vision_tower_kwargs(model.vision_tower, kwargs)
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
    selected_feature = _cast_tensor_for_module(selected_feature, model.multi_modal_projector)
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
    vision_tower_kwargs = _filter_vision_tower_kwargs(model.vision_tower, kwargs)
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
    selected_feature = _cast_tensor_for_module(selected_feature, model.multi_modal_projector)
    video_features = model.multi_modal_projector(selected_feature)
    video_features = model.apply_pooling(video_features)
    return video_features.reshape(batch_size, frames * video_features.shape[1], -1)


def _should_apply_mmtok(self, input_ids, inputs_embeds) -> bool:
    return (
        input_ids is not None
        and inputs_embeds.shape[0] == 1
        and hasattr(self, "_mmtok_core")
    )


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


class LlavaOnevision_MMTok(nn.Module):
    def forward(
        self: LlavaOnevisionModel,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_sizes: torch.LongTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_sizes_videos: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        vision_feature_layer: int | list[int] | None = None,
        vision_feature_select_strategy: str | None = None,
        vision_aspect_ratio: str | None = None,
        batch_num_images: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, LlavaOnevisionModelOutputWithPast]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_hidden_states = None
        video_hidden_states = None

        if not _should_apply_mmtok(self, input_ids, inputs_embeds):
            return _forward_without_mmtok(
                self,
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                pixel_values_videos=pixel_values_videos,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                vision_aspect_ratio=vision_aspect_ratio,
                batch_num_images=batch_num_images,
                use_cache=use_cache,
                **kwargs,
            )

        try:
            self._sink_analysis_last_export = None
            image_keep_local = None
            video_keep_local = None
            original_image_feature_count = None
            original_video_feature_count = None

            if pixel_values is not None:
                image_features = _concat_token_features(
                    _extract_image_features_compat(
                        self,
                        pixel_values=pixel_values,
                        image_sizes=image_sizes,
                        vision_feature_layer=vision_feature_layer,
                        vision_feature_select_strategy=vision_feature_select_strategy,
                        vision_aspect_ratio=vision_aspect_ratio,
                        batch_num_images=batch_num_images,
                        **kwargs,
                    )
                ).to(inputs_embeds.device, inputs_embeds.dtype)
                original_image_feature_count = image_features.shape[0]
                image_hidden_states = image_features
                question = self.get_question() if hasattr(self, "get_question") else ""
                target_tokens = compute_target_vision_tokens(
                    image_features.shape[0],
                    getattr(self._mmtok_core, "retain_ratio", None),
                    getattr(self._mmtok_core, "target_vision_tokens", None),
                )
                if target_tokens < image_features.shape[0]:
                    image_keep_local, selected_image_features = self._mmtok_core.apply_selection_preprocess_qwen(
                        image_embeds=image_features,
                        image_features=image_features,
                        question_text=question or "",
                        target_vision_tokens=target_tokens,
                    )
                    image_keep_local = torch.tensor(
                        image_keep_local,
                        device=inputs_embeds.device,
                        dtype=torch.long,
                    )
                    image_hidden_states = _concat_token_features(
                        selected_image_features
                    ).to(
                        inputs_embeds.device,
                        inputs_embeds.dtype,
                    )

            if pixel_values_videos is not None:
                video_features = _concat_token_features(
                    _extract_video_features_compat(
                        self,
                        pixel_values=pixel_values_videos,
                        vision_feature_layer=vision_feature_layer,
                        vision_feature_select_strategy=vision_feature_select_strategy,
                        **kwargs,
                    )
                )
                image_newline = self.image_newline[None, None, :].repeat(
                    video_features.shape[0],
                    1,
                    1,
                ).to(video_features.device, video_features.dtype)
                video_features = torch.cat((video_features, image_newline), dim=1)
                video_features = video_features.flatten(0, 1).to(
                    inputs_embeds.device,
                    inputs_embeds.dtype,
                )
                original_video_feature_count = video_features.shape[0]
                video_hidden_states = video_features
                question = self.get_question() if hasattr(self, "get_question") else ""
                target_tokens = compute_target_vision_tokens(
                    video_features.shape[0],
                    getattr(self._mmtok_core, "retain_ratio", None),
                    getattr(self._mmtok_core, "target_vision_tokens", None),
                )
                if target_tokens < video_features.shape[0]:
                    video_keep_local, selected_video_features = self._mmtok_core.apply_selection_preprocess_qwen(
                        image_embeds=video_features,
                        image_features=video_features,
                        question_text=question or "",
                        target_vision_tokens=target_tokens,
                    )
                    video_keep_local = torch.tensor(
                        video_keep_local,
                        device=inputs_embeds.device,
                        dtype=torch.long,
                    )
                    video_hidden_states = _concat_token_features(
                        selected_video_features
                    ).to(
                        inputs_embeds.device,
                        inputs_embeds.dtype,
                    )

            if image_keep_local is None and video_keep_local is None:
                return _forward_without_mmtok(
                    self,
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    image_sizes=image_sizes,
                    pixel_values_videos=pixel_values_videos,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    vision_feature_layer=vision_feature_layer,
                    vision_feature_select_strategy=vision_feature_select_strategy,
                    vision_aspect_ratio=vision_aspect_ratio,
                    batch_num_images=batch_num_images,
                    use_cache=use_cache,
                    **kwargs,
                )

            image_positions = torch.where(input_ids[0] == self.config.image_token_id)[0]
            if image_hidden_states is not None:
                if image_positions.numel() == 0:
                    raise ValueError("Image placeholders not found in input_ids.")
                if image_positions.numel() != original_image_feature_count:
                    raise ValueError("Image placeholder count does not match image features.")
                if image_keep_local is None:
                    image_keep_local = torch.arange(
                        image_hidden_states.shape[0],
                        device=inputs_embeds.device,
                        dtype=torch.long,
                    )
                if image_positions.numel() < int(image_keep_local.numel()):
                    raise ValueError("Selected image token count exceeds placeholder count.")
                if image_keep_local.numel() > 0:
                    inputs_embeds[:, image_positions[image_keep_local], :] = image_hidden_states.unsqueeze(0)

            video_positions = torch.where(input_ids[0] == self.config.video_token_id)[0]
            if video_hidden_states is not None:
                if video_positions.numel() == 0:
                    raise ValueError("Video placeholders not found in input_ids.")
                if video_positions.numel() != original_video_feature_count:
                    raise ValueError("Video placeholder count does not match video features.")
                if video_keep_local is None:
                    video_keep_local = torch.arange(
                        video_hidden_states.shape[0],
                        device=inputs_embeds.device,
                        dtype=torch.long,
                    )
                if video_positions.numel() < int(video_keep_local.numel()):
                    raise ValueError("Selected video token count exceeds placeholder count.")
                if video_keep_local.numel() > 0:
                    inputs_embeds[:, video_positions[video_keep_local], :] = video_hidden_states.unsqueeze(0)

            keep_indices = _build_keep_indices(
                input_ids[0],
                image_keep_local,
                video_keep_local,
                self.config,
            )
            inputs_embeds = gather_sequence_hidden_states(inputs_embeds, keep_indices)
            attention_mask = slice_attention_mask(attention_mask, keep_indices)
            position_ids = slice_position_ids(position_ids, keep_indices)
            num_visual_tokens = 0
            selected_keep = None
            if original_image_feature_count is not None:
                num_visual_tokens = int(original_image_feature_count)
                selected_keep = image_keep_local
            elif original_video_feature_count is not None:
                num_visual_tokens = int(original_video_feature_count)
                selected_keep = video_keep_local
            if selected_keep is not None:
                scores = torch.zeros(
                    num_visual_tokens,
                    device=inputs_embeds.device,
                    dtype=torch.float32,
                )
                if selected_keep.numel() > 0:
                    scores[selected_keep] = 1.0
                self._sink_analysis_last_export = {
                    "method": "mmtok",
                    "num_visual_tokens": num_visual_tokens,
                    "indices": selected_keep.detach().cpu(),
                    "scores": scores.detach().cpu(),
                }
        except Exception as error:
            eval_logger.warning(
                f"[MMTok-LLaVA-OneVision] Falling back to full visual tokens due to selection failure: {error}"
            )
            return _forward_without_mmtok(
                self,
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                pixel_values_videos=pixel_values_videos,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                vision_aspect_ratio=vision_aspect_ratio,
                batch_num_images=batch_num_images,
                use_cache=use_cache,
                **kwargs,
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
            image_hidden_states=image_hidden_states,
            video_hidden_states=video_hidden_states,
        )


def _forward_without_mmtok(
    self: LlavaOnevisionModel,
    *,
    input_ids,
    pixel_values,
    image_sizes,
    pixel_values_videos,
    attention_mask,
    position_ids,
    past_key_values,
    inputs_embeds,
    vision_feature_layer,
    vision_feature_select_strategy,
    vision_aspect_ratio,
    batch_num_images,
    use_cache,
    **kwargs,
):
    image_hidden_states = None
    if pixel_values is not None:
        image_hidden_states = _concat_token_features(
            _extract_image_features_compat(
                self,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                vision_aspect_ratio=vision_aspect_ratio,
                batch_num_images=batch_num_images,
                **kwargs,
            )
        ).to(inputs_embeds.device, inputs_embeds.dtype)
        special_image_mask, _ = self.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_hidden_states,
        )
        inputs_embeds = inputs_embeds.masked_scatter(
            special_image_mask,
            image_hidden_states,
        )

    video_hidden_states = None
    if pixel_values_videos is not None:
        video_hidden_states = _concat_token_features(
            _extract_video_features_compat(
                self,
                pixel_values=pixel_values_videos,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                **kwargs,
            )
        )
        image_newline = self.image_newline[None, None, :].repeat(
            video_hidden_states.shape[0],
            1,
            1,
        ).to(video_hidden_states.device, video_hidden_states.dtype)
        video_hidden_states = torch.cat(
            (video_hidden_states, image_newline),
            dim=1,
        ).flatten(0, 1).to(inputs_embeds.device, inputs_embeds.dtype)
        _, special_video_mask = self.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            video_features=video_hidden_states,
        )
        inputs_embeds = inputs_embeds.masked_scatter(
            special_video_mask,
            video_hidden_states,
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
        image_hidden_states=image_hidden_states,
        video_hidden_states=video_hidden_states,
    )


__all__ = ["mmtok_llava_onevision"]
