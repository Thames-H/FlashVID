from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from sink_analysis.collect.sink_metrics import identify_sink_tokens


def generate_summary_table(artifacts_by_model: dict[str, list[dict]], keep_ratio: str = "50%") -> pd.DataFrame:
    rows = []
    for model_name, artifacts in artifacts_by_model.items():
        sink_ret_attn = []
        sink_ret_fetp = []
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
            fetp = set(selection.get("fetp", {}).get("indices", torch.tensor([], dtype=torch.long)).tolist())
            attention = set(selection.get("attention", {}).get("indices", torch.tensor([], dtype=torch.long)).tolist())
            mmtok = set(selection.get("mmtok", {}).get("indices", torch.tensor([], dtype=torch.long)).tolist())
            sink_counts.append(len(sink_set))
            if sink_set:
                sink_ret_attn.append(len(attention & sink_set) / len(sink_set))
                sink_ret_fetp.append(len(fetp & sink_set) / len(sink_set))
            iou_fa.append(len(fetp & attention) / max(1, len(fetp | attention)))
            iou_fm.append(len(fetp & mmtok) / max(1, len(fetp | mmtok)))

        rows.append(
            {
                "Model": model_name,
                "Samples": len(artifacts),
                "Avg Sink Count": f"{np.mean(sink_counts) if sink_counts else 0.0:.1f}",
                "Sink Retention (Attn)": f"{np.mean(sink_ret_attn) if sink_ret_attn else 0.0:.1%}",
                "Sink Retention (FETP)": f"{np.mean(sink_ret_fetp) if sink_ret_fetp else 0.0:.1%}",
                "IoU (FETP vs Attn)": f"{np.mean(iou_fa) if iou_fa else 0.0:.3f}",
                "IoU (FETP vs MMTok)": f"{np.mean(iou_fm) if iou_fm else 0.0:.3f}",
            }
        )
    return pd.DataFrame(rows)
