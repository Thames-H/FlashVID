from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch

from sink_analysis.analyze.ratio_utils import sort_keep_ratios
from sink_analysis.collect.sink_metrics import identify_sink_tokens


def _infer_keep_ratios(artifacts_by_model: dict[str, list[dict]]) -> list[str]:
    return sort_keep_ratios(
        keep_ratio
        for artifacts in artifacts_by_model.values()
        for artifact in artifacts
        for keep_ratio in artifact.get("selections", {})
    )


def _format_metric(values: list[float], *, percentage: bool = False, precision: int = 3) -> str:
    if not values:
        return "NA"
    mean_value = float(np.mean(values))
    if percentage:
        return f"{mean_value:.1%}"
    return f"{mean_value:.{precision}f}"


def _selection_tensor(selection: dict | None, field: str) -> torch.Tensor | None:
    if selection is None:
        return None
    value = selection.get(field)
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value
    return torch.tensor(value)


def _build_summary_rows(artifacts_by_model: dict[str, list[dict]], keep_ratios: Sequence[str]) -> list[dict]:
    rows = []
    for keep_ratio in keep_ratios:
        for model_name, artifacts in artifacts_by_model.items():
            sink_ret_attn = []
            sink_ret_fetp = []
            sink_ret_mmtok = []
            iou_fa = []
            iou_fm = []
            sink_counts = []
            for artifact in artifacts:
                sink_mask, _, _ = identify_sink_tokens(
                    artifact["alpha"],
                    artifact["values"],
                    artifact["query_outputs"],
                )
                sink_set = set(torch.where(sink_mask)[0].tolist())
                selection = artifact.get("selections", {}).get(keep_ratio, {})
                fetp_tensor = _selection_tensor(selection.get("fetp"), "indices")
                attn_tensor = _selection_tensor(selection.get("attention"), "indices")
                mmtok_tensor = _selection_tensor(selection.get("mmtok"), "indices")
                fetp = set(fetp_tensor.tolist()) if fetp_tensor is not None else None
                attention = set(attn_tensor.tolist()) if attn_tensor is not None else None
                mmtok = set(mmtok_tensor.tolist()) if mmtok_tensor is not None else None
                sink_counts.append(len(sink_set))
                if sink_set and attention is not None:
                    sink_ret_attn.append(len(attention & sink_set) / len(sink_set))
                if sink_set and fetp is not None:
                    sink_ret_fetp.append(len(fetp & sink_set) / len(sink_set))
                if sink_set and mmtok is not None:
                    sink_ret_mmtok.append(len(mmtok & sink_set) / len(sink_set))
                if fetp is not None and attention is not None:
                    iou_fa.append(len(fetp & attention) / max(1, len(fetp | attention)))
                if fetp is not None and mmtok is not None:
                    iou_fm.append(len(fetp & mmtok) / max(1, len(fetp | mmtok)))

            rows.append(
                {
                    "Keep Ratio": keep_ratio,
                    "Model": model_name,
                    "Samples": len(artifacts),
                    "Avg Sink Count": f"{np.mean(sink_counts) if sink_counts else 0.0:.1f}",
                    "Sink Retention (Attn)": _format_metric(sink_ret_attn, percentage=True),
                    "Sink Retention (FETP)": _format_metric(sink_ret_fetp, percentage=True),
                    "Sink Retention (MMTok)": _format_metric(sink_ret_mmtok, percentage=True),
                    "IoU (FETP vs Attn)": _format_metric(iou_fa, precision=3),
                    "IoU (FETP vs MMTok)": _format_metric(iou_fm, precision=3),
                }
            )
    return rows


def generate_summary_table(artifacts_by_model: dict[str, list[dict]], keep_ratio: str = "50%") -> pd.DataFrame:
    frame = pd.DataFrame(_build_summary_rows(artifacts_by_model, [keep_ratio]))
    if "Keep Ratio" in frame.columns:
        frame = frame.drop(columns=["Keep Ratio"])
    return frame


def generate_summary_tables_by_ratio(
    artifacts_by_model: dict[str, list[dict]],
    keep_ratios: Sequence[str] | None = None,
) -> pd.DataFrame:
    resolved_keep_ratios = list(keep_ratios) if keep_ratios is not None else _infer_keep_ratios(artifacts_by_model)
    return pd.DataFrame(_build_summary_rows(artifacts_by_model, resolved_keep_ratios))
