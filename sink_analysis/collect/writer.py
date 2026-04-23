from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch

from sink_analysis.collect.sample_records import build_base_sample_record
from sink_analysis.schema import (
    SinkAnalysisExportConfig,
    artifact_file_name,
    partial_file_name,
    strip_none_values,
)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _to_builtin(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def build_partial_record_payload(
    *,
    export_config: SinkAnalysisExportConfig,
    task_name: str,
    doc_id: int | str,
    benchmark: str,
    target: Any,
    messages: Iterable[dict[str, Any]] | None,
    answer: str,
    export_payload: dict[str, Any] | None = None,
    patch_mapping: dict[str, Any] | None = None,
    image: Any = None,
) -> dict[str, Any]:
    payload = build_base_sample_record(
        task_name=task_name,
        doc_id=doc_id,
        model_name=export_config.model_name,
        benchmark=benchmark,
        target=target,
        messages=messages,
        image=image,
    )
    payload.update(
        {
            "method": export_config.method_name,
            "keep_ratio": export_config.keep_ratio_label,
            "answer": answer,
        }
    )

    if export_payload is None:
        return strip_none_values(payload)

    payload.update(
        {
            "num_visual_tokens": export_payload.get("num_visual_tokens"),
            "patch_mapping": patch_mapping,
            "target_layer": export_payload.get("target_layer"),
            "alpha": export_payload.get("alpha"),
            "values": export_payload.get("values"),
            "query_outputs": export_payload.get("query_outputs"),
            "selection": {
                "indices": _to_builtin(export_payload.get("indices", [])),
                "scores": _to_builtin(export_payload.get("scores", [])),
            },
        }
    )
    return strip_none_values(payload)


def load_override_map(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    override_path = path if isinstance(path, Path) else Path(path)
    if not override_path.exists():
        return None
    with override_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def lookup_override_indices(
    override_map: dict[str, Any] | None,
    *,
    model_name: str,
    sample_id: str,
    keep_ratio_label: str,
) -> list[int] | None:
    if not override_map:
        return None
    model_overrides = override_map.get(model_name, {})
    sample_overrides = model_overrides.get(sample_id, {})
    override_indices = sample_overrides.get(keep_ratio_label)
    if override_indices is None:
        return None
    return [int(index) for index in override_indices]


def write_partial_record(output_root: Path, payload: dict[str, Any]) -> Path:
    output_path = output_root / partial_file_name(
        payload["model"],
        payload["method"],
        payload["keep_ratio"],
        payload["sample_id"],
    )
    ensure_parent(output_path)
    torch.save(payload, output_path)
    return output_path


def write_artifact(output_root: Path, payload: dict[str, Any]) -> Path:
    model_root = output_root / payload["model"]
    output_path = model_root / artifact_file_name(payload["model"], payload["sample_id"])
    ensure_parent(output_path)
    torch.save(payload, output_path)
    return output_path


def load_pt_records(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.pt")):
        records.append(torch.load(path, map_location="cpu", weights_only=False))
    return records


def write_json(path: Path, payload: Any) -> Path:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def group_records(records: Iterable[dict[str, Any]], *keys: str) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        group_key = tuple(record.get(key) for key in keys)
        grouped.setdefault(group_key, []).append(record)
    return grouped
