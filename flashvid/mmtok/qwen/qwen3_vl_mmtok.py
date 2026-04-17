# MMTok: Multimodal Coverage Maximization for Efficient Inference of VLMs
# Paper: https://arxiv.org/abs/2508.18264

import types

from loguru import logger as eval_logger

from ..core import MMTokCore
from .modeling_qwen3_vl_mmtok import Qwen3VLVisionModel_MMTok
from .qwen3_VLmodel_mmtok import Qwen3_VL_MMTok


def mmtok_qwen3_vl(
    qwen_model,
    language_tokenizer=None,
    processor=None,
    retain_ratio=0.2,
    **mmtok_kwargs,
):
    mmtok_config = {
        "alpha": 0.5,
        "softmax_tv_temperature": 0.01,
        "softmax_vv_temperature": 0.2,
        "device": qwen_model.device,
        "remove_padding_indices": False,
        **mmtok_kwargs,
    }

    eval_logger.info(
        f"[MMTok-Qwen3] Injecting MMTok: retain_ratio={retain_ratio}, "
        f"device={mmtok_config['device']}"
    )
    mmtok_core = MMTokCore(**mmtok_config)
    mmtok_core.retain_ratio = retain_ratio
    mmtok_core._main_model_embed_tokens = qwen_model.get_input_embeddings()
    mmtok_core._language_tokenizer = language_tokenizer

    qwen_model.model._mmtok_core = mmtok_core
    qwen_model.model._question_for_vision = None
    qwen_model.set_question = types.MethodType(_set_question, qwen_model)
    qwen_model.model.get_question = types.MethodType(_get_question, qwen_model.model)
    qwen_model.model.forward = types.MethodType(Qwen3_VL_MMTok.forward, qwen_model.model)
    qwen_model.model.get_video_features = types.MethodType(
        Qwen3_VL_MMTok.get_video_features,
        qwen_model.model,
    )
    qwen_model.model.get_image_features = types.MethodType(
        Qwen3_VL_MMTok.get_image_features,
        qwen_model.model,
    )
    qwen_model.model.visual.forward = types.MethodType(
        Qwen3VLVisionModel_MMTok.forward,
        qwen_model.model.visual,
    )

    if processor is not None:
        patch_qwen3_vl_processor_for_question_hook(processor, qwen_model)
    else:
        eval_logger.warning(
            "[MMTok-Qwen3] No processor provided, skipping question hook patch"
        )

    return qwen_model, processor


def _set_question(self, question: str):
    self.model._question_for_vision = question


def _get_question(self):
    return self._question_for_vision


def patch_qwen3_vl_processor_for_question_hook(processor, mmtok_model_instance):
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


def extract_question_from_messages(messages):
    question_parts = []
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content", [])
        if isinstance(content, str):
            if content:
                question_parts.append(content)
            continue
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_content = item.get("text", "")
                if text_content:
                    question_parts.append(text_content)
    return " ".join(question_parts).strip()


__all__ = ["mmtok_qwen3_vl", "extract_question_from_messages"]
