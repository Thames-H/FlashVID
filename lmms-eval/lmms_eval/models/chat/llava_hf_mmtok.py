import types
from typing import List, Optional, Tuple, Union

import torch
from loguru import logger as eval_logger
from transformers.models.llava_onevision.modeling_llava_onevision import (
    LlavaOnevisionModel,
    LlavaOnevisionModelOutputWithPast,
)

from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.chat.llava_hf import LlavaHf as LlavaHfChat
from lmms_eval.models.chat.mmtok_hf_common import (
    build_keep_indices,
    gather_sequence_hidden,
    get_newline_mask,
    get_question_from_model,
    load_bundled_mmtok_core,
    patch_processor_for_question_hook,
    resolve_flashvid_repo_root,
    select_vision_token_indices,
    set_question_on_model,
    slice_attention_mask,
    slice_position_ids,
)


MMTokCore = load_bundled_mmtok_core()


def _set_question(self, question: Optional[str]):
    set_question_on_model(self.model, question)


def _get_question(self):
    return get_question_from_model(self)


def _llava_onevision_forward(
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
    **kwargs,
):
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    question = get_question_from_model(self)
    enable_mmtok = (
        hasattr(self, "_mmtok_core")
        and input_ids is not None
        and input_ids.shape[0] == 1
        and input_ids.shape[1] > 1
    )

    image_features = None
    kept_image_positions = None
    if pixel_values is not None:
        image_features = self.get_image_features(
            pixel_values,
            image_sizes,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            batch_num_images=batch_num_images,
            return_dict=True,
        ).pooler_output
        image_features = torch.cat(image_features, dim=0)
        image_positions = torch.where(input_ids[0] == self.config.image_token_id)[0]
        if image_positions.numel() != image_features.shape[0]:
            raise ValueError(
                "Image placeholder count does not match packed image features."
            )
        kept_image_positions = image_positions

        if enable_mmtok and image_features.shape[0] > 0:
            image_keep_local = select_vision_token_indices(
                self._mmtok_core,
                projected_features=image_features,
                coverage_features=image_features,
                question_text=question,
                retain_ratio=getattr(self._mmtok_core, "retain_ratio", 1.0),
                always_keep_mask=get_newline_mask(image_features, self.image_newline),
            )
            if image_keep_local.numel() < image_features.shape[0]:
                kept_image_positions = image_positions[image_keep_local]
                image_features = image_features[image_keep_local]

    video_features = None
    kept_video_positions = None
    if pixel_values_videos is not None:
        video_features = self.get_video_features(
            pixel_values_videos,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            return_dict=True,
        ).pooler_output

        if video_features.shape[0] != 1:
            raise ValueError("LlavaOneVision MMTok currently supports batch_size=1 only.")

        flat_video_features = video_features[0]
        newline_embedding = self.image_newline[None, :].to(
            flat_video_features.device, flat_video_features.dtype
        )
        video_positions = torch.where(input_ids[0] == self.config.video_token_id)[0]
        if video_positions.numel() != flat_video_features.shape[0] + 1:
            raise ValueError(
                "Video placeholder count does not match pooled video features."
            )
        kept_video_positions = video_positions

        if enable_mmtok and flat_video_features.shape[0] > 0:
            video_keep_local = select_vision_token_indices(
                self._mmtok_core,
                projected_features=flat_video_features,
                coverage_features=flat_video_features,
                question_text=question,
                retain_ratio=getattr(self._mmtok_core, "retain_ratio", 1.0),
            )
            if video_keep_local.numel() < flat_video_features.shape[0]:
                kept_video_positions = torch.cat(
                    [video_positions[video_keep_local], video_positions[-1:]], dim=0
                )
                flat_video_features = flat_video_features[video_keep_local]

        video_features = torch.cat([flat_video_features, newline_embedding], dim=0)

    should_prune_sequence = False
    if pixel_values is not None and kept_image_positions is not None:
        should_prune_sequence |= kept_image_positions.numel() != torch.sum(
            input_ids[0] == self.config.image_token_id
        ).item()
    if pixel_values_videos is not None and kept_video_positions is not None:
        should_prune_sequence |= kept_video_positions.numel() != torch.sum(
            input_ids[0] == self.config.video_token_id
        ).item()

    if should_prune_sequence:
        keep_global_indices = build_keep_indices(
            input_ids[0],
            image_token_id=self.config.image_token_id,
            video_token_id=self.config.video_token_id,
            kept_image_positions=kept_image_positions,
            kept_video_positions=kept_video_positions,
        )
        pruned_input_ids = input_ids[:, keep_global_indices]
        inputs_embeds = gather_sequence_hidden(inputs_embeds, keep_global_indices)
        attention_mask = slice_attention_mask(attention_mask, keep_global_indices)
        position_ids = slice_position_ids(position_ids, keep_global_indices)

        if image_features is not None:
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        if video_features is not None:
            video_features = video_features.to(inputs_embeds.device, inputs_embeds.dtype)

        special_image_mask, special_video_mask = self.get_placeholder_mask(
            pruned_input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_features,
            video_features=video_features,
        )
        if image_features is not None:
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
        if video_features is not None:
            inputs_embeds = inputs_embeds.masked_scatter(special_video_mask, video_features)
    else:
        if image_features is not None:
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            special_image_mask, _ = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_features,
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        if video_features is not None:
            video_features = video_features.to(inputs_embeds.device, inputs_embeds.dtype)
            _, special_video_mask = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                video_features=video_features,
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_video_mask, video_features)

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


@register_model("llava_hf_mmtok")
class LlavaHfMMTok(LlavaHfChat):
    def __init__(
        self,
        pretrained: str = "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        device: str = "cuda",
        device_map: str = "auto",
        batch_size: Union[int, str] = 1,
        use_cache: bool = True,
        retain_ratio: float = 0.2,
        alpha: float = 0.5,
        softmax_tv_temperature: float = 0.01,
        softmax_vv_temperature: float = 0.2,
        **kwargs,
    ) -> None:
        if int(batch_size) != 1:
            raise AssertionError("LlavaOneVision MMTok currently supports batch_size=1 only.")

        super().__init__(
            pretrained=pretrained,
            device=device,
            device_map=device_map,
            batch_size=batch_size,
            use_cache=use_cache,
            **kwargs,
        )

        base_model = self.model
        if getattr(base_model.config, "model_type", None) != "llava_onevision":
            raise ValueError(
                "llava_hf_mmtok only supports Hugging Face Llava-OneVision checkpoints."
            )

        flashvid_repo_root = resolve_flashvid_repo_root()
        eval_logger.info(
            f"[LlavaOneVision-MMTok] Using bundled FlashVID MMTok from "
            f"{flashvid_repo_root / 'flashvid' / 'mmtok'}"
        )

        mmtok_core = MMTokCore(
            alpha=alpha,
            softmax_tv_temperature=softmax_tv_temperature,
            softmax_vv_temperature=softmax_vv_temperature,
            device=base_model.device,
            remove_padding_indices=False,
        )
        mmtok_core.retain_ratio = retain_ratio
        mmtok_core._main_model_embed_tokens = base_model.get_input_embeddings()
        mmtok_core._language_tokenizer = self._tokenizer

        base_model.model._mmtok_core = mmtok_core
        base_model.model._question_for_vision = None
        base_model.set_question = types.MethodType(_set_question, base_model)
        base_model.model.get_question = types.MethodType(_get_question, base_model.model)
        base_model.model.forward = types.MethodType(_llava_onevision_forward, base_model.model)
        patch_processor_for_question_hook(self._image_processor, base_model)

        self.retain_ratio = retain_ratio

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for LlavaHfMMTok.")
