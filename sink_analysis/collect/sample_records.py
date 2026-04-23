from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from sink_analysis.schema import build_sample_id


def extract_question_from_messages(messages: Iterable[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content", [])
        if isinstance(content, str):
            if content:
                parts.append(content)
            continue
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "").strip()
                if text:
                    parts.append(text)
    return " ".join(parts).strip()


def build_image_preview(image: Any, max_side: int = 256) -> np.ndarray | None:
    if image is None:
        return None
    try:
        preview = image.copy()
        preview.thumbnail((max_side, max_side))
        return np.asarray(preview.convert("RGB"), dtype=np.uint8)
    except Exception:
        return None


def build_image_size(image: Any) -> tuple[int, int] | None:
    if image is None:
        return None
    try:
        width, height = image.size
        return int(height), int(width)
    except Exception:
        return None


def extract_ground_truth_from_doc(doc: Any) -> Any:
    if doc is None or not isinstance(doc, dict):
        return doc
    for key in (
        "ground_truth",
        "answer",
        "answers",
        "label",
        "target",
        "gt",
    ):
        if key in doc:
            return doc[key]
    return None


def build_base_sample_record(
    *,
    task_name: str,
    doc_id: int | str,
    model_name: str,
    benchmark: str,
    target: Any,
    messages: Iterable[dict[str, Any]] | None,
    image: Any = None,
) -> dict[str, Any]:
    return {
        "sample_id": build_sample_id(task_name, doc_id),
        "model": model_name,
        "benchmark": benchmark,
        "question": extract_question_from_messages(messages or []),
        "ground_truth": target,
        "image_preview": build_image_preview(image),
        "image_size": build_image_size(image),
    }
