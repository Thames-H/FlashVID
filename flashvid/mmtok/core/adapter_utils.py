# MMTok: Multimodal Coverage Maximization for Efficient Inference of VLMs
# Paper: https://arxiv.org/abs/2508.18264

"""Shared helpers for bundled MMTok adapters."""

import math
from typing import Optional

import torch


def extract_question_from_messages(messages) -> str:
    question_parts = []
    for message in _iter_message_dicts(messages):
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
            if not isinstance(item, dict):
                continue
            if item.get("type") != "text":
                continue
            text_content = item.get("text", "")
            if text_content:
                question_parts.append(text_content)
    return " ".join(question_parts).strip()


def compute_target_vision_tokens(
    num_visual_tokens: int,
    retain_ratio: Optional[float],
    target_vision_tokens: Optional[int],
) -> int:
    if num_visual_tokens <= 0:
        return 0
    if target_vision_tokens is not None:
        return max(0, min(num_visual_tokens, int(target_vision_tokens)))
    if retain_ratio is None:
        return num_visual_tokens
    if retain_ratio <= 0:
        return 0
    if retain_ratio >= 1:
        return num_visual_tokens
    return max(1, min(num_visual_tokens, math.ceil(num_visual_tokens * retain_ratio)))


def slice_attention_mask(attention_mask, keep_indices: torch.LongTensor):
    if attention_mask is None:
        return None
    if isinstance(attention_mask, dict):
        return {
            key: slice_attention_mask(value, keep_indices)
            for key, value in attention_mask.items()
        }
    if attention_mask.ndim == 2:
        return attention_mask[:, keep_indices]
    if attention_mask.ndim == 3:
        return attention_mask[:, :, keep_indices]
    if (
        attention_mask.ndim == 4
        and attention_mask.shape[-1] == attention_mask.shape[-2]
    ):
        return attention_mask[:, :, keep_indices][:, :, :, keep_indices]
    return attention_mask


def slice_position_ids(position_ids, keep_indices: torch.LongTensor):
    if position_ids is None:
        return None
    if position_ids.ndim == 1:
        return position_ids[keep_indices]
    if position_ids.ndim == 2:
        return position_ids[:, keep_indices]
    if position_ids.ndim == 3:
        return position_ids[:, :, keep_indices]
    return position_ids


def gather_sequence_hidden_states(
    hidden_states: torch.Tensor,
    keep_indices: torch.LongTensor,
) -> torch.Tensor:
    gather_index = keep_indices.view(1, -1, 1).expand(
        hidden_states.shape[0],
        -1,
        hidden_states.shape[-1],
    )
    return torch.gather(hidden_states, dim=1, index=gather_index)


def extract_pooler_output(outputs):
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


def concat_token_features(features):
    if isinstance(features, torch.Tensor):
        return features
    if isinstance(features, (tuple, list)):
        tensors = tuple(features)
        if not tensors:
            raise ValueError("Expected non-empty feature sequence")
        return torch.cat(tensors, dim=0)
    raise TypeError(f"Unsupported feature type: {type(features)!r}")


def _iter_message_dicts(messages):
    for item in messages:
        if isinstance(item, dict):
            yield item
        elif isinstance(item, list):
            yield from _iter_message_dicts(item)
