from __future__ import annotations

from typing import Any

import torch


def _to_tensor(values: Any) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values
    return torch.tensor(values)


def merge_partial_records(partials: list[dict[str, Any]]) -> dict[str, Any]:
    if not partials:
        raise ValueError("Expected at least one partial record to merge.")

    base: dict[str, Any] = {}
    for partial in partials:
        for key, value in partial.items():
            if key in {"method", "keep_ratio", "selection", "answer"}:
                continue
            if key not in base or base[key] is None:
                base[key] = value
    base["selections"] = {}
    base["answers"] = {}

    for partial in partials:
        keep_ratio = partial["keep_ratio"]
        method = partial["method"]
        if method == "full" or keep_ratio == "full":
            base["answers"]["full"] = partial["answer"]
            continue
        selection = partial.get("selection")
        if selection is None:
            continue
        base["selections"].setdefault(keep_ratio, {})
        base["answers"].setdefault(keep_ratio, {})
        base["selections"][keep_ratio][method] = {
            "indices": _to_tensor(selection["indices"]).long(),
            "scores": _to_tensor(selection["scores"]).float(),
        }
        base["answers"][keep_ratio][method] = partial["answer"]

    return base
