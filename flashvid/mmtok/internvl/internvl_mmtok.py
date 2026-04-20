# MMTok: Multimodal Coverage Maximization for Efficient Inference of VLMs
# Paper: https://arxiv.org/abs/2508.18264

import types
from typing import Optional, Union

import torch
import torch.nn as nn
from loguru import logger as eval_logger
from transformers.models.internvl.modeling_internvl import (
    InternVLModel,
    InternVLModelOutputWithPast,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from ..core import MMTokCore
from ..core.adapter_utils import (
    compute_target_vision_tokens,
    extract_pooler_output,
    extract_question_from_messages,
    gather_sequence_hidden_states,
    slice_attention_mask,
    slice_position_ids,
)


def mmtok_internvl(
    internvl_model,
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
        "device": internvl_model.device,
        "remove_padding_indices": False,
        **mmtok_kwargs,
    }

    eval_logger.info(
        f"[MMTok-InternVL] Injecting MMTok: retain_ratio={retain_ratio}, "
        f"target_vision_tokens={target_vision_tokens}, device={mmtok_config['device']}"
    )
    mmtok_core = MMTokCore(**mmtok_config)
    mmtok_core.retain_ratio = retain_ratio
    mmtok_core.target_vision_tokens = target_vision_tokens
    mmtok_core._main_model_embed_tokens = internvl_model.get_input_embeddings()
    mmtok_core._language_tokenizer = language_tokenizer

    internvl_model.model._mmtok_core = mmtok_core
    internvl_model.model._question_for_vision = None
    internvl_model.set_question = types.MethodType(_set_question, internvl_model)
    internvl_model.model.get_question = types.MethodType(
        _get_question,
        internvl_model.model,
    )
    internvl_model.model.forward = types.MethodType(
        InternVL_MMTok.forward,
        internvl_model.model,
    )

    if processor is not None:
        patch_processor_for_question_hook(processor, internvl_model)
    else:
        eval_logger.warning(
            "[MMTok-InternVL] No processor provided, skipping question hook patch"
        )

    return internvl_model, processor


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


def _should_apply_mmtok(self, input_ids, inputs_embeds) -> bool:
    return (
        input_ids is not None
        and inputs_embeds.shape[0] == 1
        and hasattr(self, "_mmtok_core")
    )


def _build_keep_indices(input_ids: torch.Tensor, image_keep_local: torch.LongTensor, config):
    non_visual_positions = torch.where(input_ids != config.image_token_id)[0]
    image_positions = torch.where(input_ids == config.image_token_id)[0]
    keep_parts = [non_visual_positions]
    if image_keep_local is not None:
        keep_parts.append(image_positions[image_keep_local])
    return torch.cat(keep_parts, dim=0).sort().values


class InternVL_MMTok(nn.Module):
    def forward(
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
    ) -> Union[tuple, InternVLModelOutputWithPast]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_hidden_states = None
        if pixel_values is not None:
            image_outputs = self.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                return_dict=True,
            )
            image_features = extract_pooler_output(image_outputs)
            if image_features.ndim == 2:
                image_features = image_features.unsqueeze(0)
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            image_hidden_states = image_features

            if _should_apply_mmtok(self, input_ids, inputs_embeds):
                try:
                    flat_image_features = image_features[0]
                    target_tokens = compute_target_vision_tokens(
                        flat_image_features.shape[0],
                        getattr(self._mmtok_core, "retain_ratio", None),
                        getattr(self._mmtok_core, "target_vision_tokens", None),
                    )

                    if target_tokens < flat_image_features.shape[0]:
                        question = self.get_question() if hasattr(self, "get_question") else ""
                        keep_local, selected_features = self._mmtok_core.apply_selection_preprocess_qwen(
                            image_embeds=flat_image_features,
                            image_features=flat_image_features,
                            question_text=question or "",
                            target_vision_tokens=target_tokens,
                        )
                        keep_local = torch.tensor(
                            keep_local,
                            device=inputs_embeds.device,
                            dtype=torch.long,
                        )
                        selected_features = selected_features.to(
                            inputs_embeds.device,
                            inputs_embeds.dtype,
                        )
                        image_positions = torch.where(input_ids[0] == self.config.image_token_id)[0]
                        if image_positions.numel() != flat_image_features.shape[0]:
                            raise ValueError(
                                "Image placeholder count does not match image features."
                            )

                        if keep_local.numel() > 0:
                            inputs_embeds[:, image_positions[keep_local], :] = selected_features.unsqueeze(0)
                        keep_indices = _build_keep_indices(input_ids[0], keep_local, self.config)
                        inputs_embeds = gather_sequence_hidden_states(inputs_embeds, keep_indices)
                        attention_mask = slice_attention_mask(attention_mask, keep_indices)
                        position_ids = slice_position_ids(position_ids, keep_indices)
                        image_hidden_states = selected_features.unsqueeze(0)
                    else:
                        special_image_mask = self.get_placeholder_mask(
                            input_ids,
                            inputs_embeds=inputs_embeds,
                            image_features=image_features,
                        )
                        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
                except Exception as error:
                    eval_logger.warning(
                        f"[MMTok-InternVL] Falling back to full visual tokens due to selection failure: {error}"
                    )
                    special_image_mask = self.get_placeholder_mask(
                        input_ids,
                        inputs_embeds=inputs_embeds,
                        image_features=image_features,
                    )
                    inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
            else:
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
            image_hidden_states=image_hidden_states,
        )


__all__ = ["mmtok_internvl"]
