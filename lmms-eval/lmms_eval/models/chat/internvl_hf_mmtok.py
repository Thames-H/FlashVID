import types
from typing import Optional

import torch
from loguru import logger as eval_logger
from transformers.models.internvl.modeling_internvl import (
    InternVLModel,
    InternVLModelOutputWithPast,
)

from lmms_eval.api.registry import register_model
from lmms_eval.models.chat.internvl_hf import InternVLHf
from lmms_eval.models.chat.mmtok_hf_common import (
    build_keep_indices,
    gather_sequence_hidden,
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


def _internvl_get_image_features(
    self: InternVLModel,
    pixel_values: torch.FloatTensor,
    vision_feature_layer: int | list[int] | None = None,
    vision_feature_select_strategy: str | None = None,
    **kwargs,
):
    pixel_values = pixel_values.to(dtype=self.dtype)

    downsample_ratio = self.config.downsample_ratio
    if vision_feature_layer != -1:
        kwargs["output_hidden_states"] = True
    vision_outputs = self.vision_tower(pixel_values=pixel_values, return_dict=True, **kwargs)
    if vision_feature_layer == -1:
        vision_features = vision_outputs.last_hidden_state
    else:
        vision_features = vision_outputs.hidden_states[vision_feature_layer]
    if vision_feature_select_strategy == "default":
        vision_features = vision_features[:, 1:, :]

    channels = vision_features.shape[1]
    feature_size = int(channels**0.5)
    batch_size = vision_features.shape[0]

    vision_features = vision_features.reshape(batch_size, feature_size, feature_size, -1)
    vision_features = self.pixel_shuffle(vision_features, scale_factor=downsample_ratio)
    selection_features = vision_features.reshape(batch_size, -1, vision_features.shape[-1])

    vision_outputs.pooler_output = self.multi_modal_projector(selection_features)
    vision_outputs.selection_features = selection_features
    return vision_outputs


def _internvl_forward(
    self: InternVLModel,
    input_ids: torch.LongTensor | None = None,
    pixel_values: torch.FloatTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values=None,
    inputs_embeds: torch.FloatTensor | None = None,
    vision_feature_layer: int | list[int] | None = None,
    vision_feature_select_strategy: str | None = None,
    **kwargs,
):
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    image_features = None
    enable_mmtok = (
        hasattr(self, "_mmtok_core")
        and input_ids is not None
        and input_ids.shape[0] == 1
        and input_ids.shape[1] > 1
    )

    if pixel_values is not None:
        image_outputs = self.get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            return_dict=True,
        )
        image_features = image_outputs.pooler_output
        selection_features = getattr(image_outputs, "selection_features", image_features)
        image_positions = torch.where(input_ids[0] == self.config.image_token_id)[0]
        if image_positions.numel() != image_features.shape[1]:
            raise ValueError(
                "Image placeholder count does not match InternVL image features."
            )

        kept_image_positions = image_positions
        if enable_mmtok and image_features.shape[1] > 0:
            image_keep_local = select_vision_token_indices(
                self._mmtok_core,
                projected_features=image_features[0],
                coverage_features=selection_features[0],
                question_text=get_question_from_model(self),
                retain_ratio=getattr(self._mmtok_core, "retain_ratio", 1.0),
            )
            if image_keep_local.numel() < image_features.shape[1]:
                kept_image_positions = image_positions[image_keep_local]
                image_features = image_features[:, image_keep_local, :]

        if kept_image_positions.numel() != image_positions.numel():
            keep_global_indices = build_keep_indices(
                input_ids[0],
                image_token_id=self.config.image_token_id,
                kept_image_positions=kept_image_positions,
            )
            pruned_input_ids = input_ids[:, keep_global_indices]
            inputs_embeds = gather_sequence_hidden(inputs_embeds, keep_global_indices)
            attention_mask = slice_attention_mask(attention_mask, keep_global_indices)
            position_ids = slice_position_ids(position_ids, keep_global_indices)
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            special_image_mask = self.get_placeholder_mask(
                pruned_input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_features,
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
        else:
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            special_image_mask = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_features,
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

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
        image_hidden_states=image_features if pixel_values is not None else None,
    )


@register_model("internvl_hf_mmtok")
class InternVLHfMMTok(InternVLHf):
    def __init__(
        self,
        pretrained: str = "OpenGVLab/InternVL3_5-8B-HF",
        batch_size: int = 1,
        retain_ratio: float = 0.2,
        alpha: float = 0.5,
        softmax_tv_temperature: float = 0.01,
        softmax_vv_temperature: float = 0.2,
        **kwargs,
    ) -> None:
        if int(batch_size) != 1:
            raise AssertionError("InternVL MMTok currently supports batch_size=1 only.")

        super().__init__(
            pretrained=pretrained,
            batch_size=batch_size,
            **kwargs,
        )

        base_model = self.model
        if getattr(base_model.config, "model_type", None) != "internvl":
            raise ValueError("internvl_hf_mmtok only supports Hugging Face InternVL checkpoints.")

        flashvid_repo_root = resolve_flashvid_repo_root()
        eval_logger.info(
            f"[InternVL-MMTok] Using bundled FlashVID MMTok from "
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
        base_model.model.get_image_features = types.MethodType(
            _internvl_get_image_features,
            base_model.model,
        )
        base_model.model.forward = types.MethodType(_internvl_forward, base_model.model)
        patch_processor_for_question_hook(self.processor, base_model)

        self.retain_ratio = retain_ratio
