from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SinkAnalysisExportConfig:
    output_root: Path
    model_name: str
    method_name: str
    keep_ratio_label: str
    include_tensors: bool = False


def build_sample_id(task_name: str, doc_id: int | str) -> str:
    return f"{task_name}__{doc_id}"


def keep_ratio_to_label(retention_ratio: float | int | str) -> str:
    if isinstance(retention_ratio, str):
        return retention_ratio if retention_ratio.endswith("%") else retention_ratio
    if retention_ratio < 1:
        return f"{int(round(float(retention_ratio) * 100))}%"
    return str(int(retention_ratio))


def artifact_file_name(model_name: str, sample_id: str) -> str:
    return f"{model_name}__{sample_id}.pt"


def partial_file_name(
    model_name: str,
    method_name: str,
    keep_ratio_label: str,
    sample_id: str,
) -> str:
    keep_slug = keep_ratio_label.replace("%", "pct")
    return f"{model_name}__{method_name}__{keep_slug}__{sample_id}.pt"


def coerce_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def strip_none_values(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}

