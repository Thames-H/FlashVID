import sys
from pathlib import Path
from typing import Optional

import torch


def resolve_flashvid_repo_root() -> Path:
    repo_root = Path(__file__).resolve().parents[4]
    flashvid_pkg = repo_root / "flashvid"
    if not flashvid_pkg.exists():
        raise FileNotFoundError(
            f"FlashVID package not found at {flashvid_pkg}. "
            "Expected the workspace copy under flashvid/."
        )
    return repo_root


def load_bundled_mmtok_core():
    repo_root = resolve_flashvid_repo_root()
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    from flashvid.mmtok.core import MMTokCore

    return MMTokCore


def iter_message_dicts(messages):
    for item in messages:
        if isinstance(item, dict):
            yield item
        elif isinstance(item, list):
            yield from iter_message_dicts(item)


def extract_question_from_messages(messages) -> str:
    question_parts = []
    for message in iter_message_dicts(messages):
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


def patch_processor_for_question_hook(processor, mmtok_model_instance) -> None:
    if getattr(processor, "_mmtok_question_hook_patched", False):
        return

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
    processor._mmtok_question_hook_patched = True


def set_question_on_model(model, question: Optional[str]) -> None:
    model._question_for_vision = question


def get_question_from_model(model) -> str:
    return getattr(model, "_question_for_vision", None) or ""


def compute_target_vision_tokens(num_visual_tokens: int, retain_ratio: float) -> int:
    if num_visual_tokens <= 0 or retain_ratio <= 0:
        return 0
    if retain_ratio >= 1:
        return num_visual_tokens
    return max(1, min(num_visual_tokens, int(torch.ceil(torch.tensor(num_visual_tokens * retain_ratio)).item())))


def select_vision_token_indices(
    mmtok_core,
    projected_features: torch.Tensor,
    question_text: str,
    retain_ratio: float,
    coverage_features: Optional[torch.Tensor] = None,
    always_keep_mask: Optional[torch.BoolTensor] = None,
) -> torch.LongTensor:
    device = projected_features.device
    total_tokens = projected_features.shape[0]

    if coverage_features is None:
        coverage_features = projected_features

    if always_keep_mask is None:
        always_keep_mask = torch.zeros(total_tokens, dtype=torch.bool, device=device)

    always_keep_indices = torch.where(always_keep_mask)[0]
    candidate_indices = torch.where(~always_keep_mask)[0]

    if candidate_indices.numel() == 0:
        return always_keep_indices

    target_vision_tokens = compute_target_vision_tokens(candidate_indices.numel(), retain_ratio)
    if target_vision_tokens == 0:
        selected_candidate_indices = candidate_indices[:0]
    elif target_vision_tokens >= candidate_indices.numel():
        selected_candidate_indices = candidate_indices
    else:
        selected_local_indices, _ = mmtok_core.apply_selection_preprocess_qwen(
            image_embeds=projected_features[candidate_indices],
            image_features=coverage_features[candidate_indices],
            question_text=question_text,
            target_vision_tokens=target_vision_tokens,
        )
        selected_candidate_indices = candidate_indices[
            torch.tensor(selected_local_indices, device=device, dtype=torch.long)
        ]

    return torch.cat([always_keep_indices, selected_candidate_indices], dim=0).sort().values


def gather_sequence_hidden(hidden_states: torch.Tensor, keep_indices: torch.LongTensor) -> torch.Tensor:
    hidden_size = hidden_states.shape[-1]
    gather_index = keep_indices.view(1, -1, 1).expand(hidden_states.shape[0], -1, hidden_size)
    return torch.gather(hidden_states, dim=1, index=gather_index)


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

    if attention_mask.ndim == 4 and attention_mask.shape[-1] == attention_mask.shape[-2]:
        return attention_mask[:, :, keep_indices][:, :, :, keep_indices]

    return attention_mask


def slice_position_ids(position_ids, keep_indices: torch.LongTensor):
    if position_ids is None:
        return None
    if position_ids.ndim == 2:
        return position_ids[:, keep_indices]
    if position_ids.ndim == 3:
        return position_ids[:, :, keep_indices]
    return position_ids


def build_keep_indices(
    input_ids_row: torch.LongTensor,
    image_token_id: Optional[int] = None,
    video_token_id: Optional[int] = None,
    kept_image_positions: Optional[torch.LongTensor] = None,
    kept_video_positions: Optional[torch.LongTensor] = None,
) -> torch.LongTensor:
    keep_mask = torch.ones_like(input_ids_row, dtype=torch.bool)
    if image_token_id is not None:
        keep_mask &= input_ids_row != image_token_id
    if video_token_id is not None:
        keep_mask &= input_ids_row != video_token_id

    keep_parts = [torch.where(keep_mask)[0]]
    if kept_image_positions is not None:
        keep_parts.append(kept_image_positions)
    if kept_video_positions is not None:
        keep_parts.append(kept_video_positions)

    return torch.cat(keep_parts, dim=0).sort().values


def get_newline_mask(features: torch.Tensor, newline_embedding: torch.Tensor) -> torch.BoolTensor:
    if features.numel() == 0:
        return torch.zeros(features.shape[0], dtype=torch.bool, device=features.device)

    newline_embedding = newline_embedding.to(features.device, features.dtype)
    return torch.isclose(
        features,
        newline_embedding.unsqueeze(0),
        atol=1e-6,
        rtol=1e-4,
    ).all(dim=-1)
