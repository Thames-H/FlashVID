from __future__ import annotations

import matplotlib.pyplot as plt
import torch

from sink_analysis.collect.sink_metrics import identify_sink_tokens


def _compute_decomposition(alpha: torch.Tensor, values: torch.Tensor, query_outputs: torch.Tensor):
    attn_factor = alpha.pow(2).sum(dim=0).sqrt()
    diff = values.unsqueeze(0) - query_outputs.unsqueeze(1)
    diff_norm_sq = diff.pow(2).sum(dim=-1)
    weighted_dev = (alpha.pow(2) * diff_norm_sq).sum(dim=0)
    dev_factor = (weighted_dev / attn_factor.pow(2).clamp(min=1e-8)).sqrt()
    return attn_factor, dev_factor


def plot_score_decomposition(artifacts: list[dict], model_name: str):
    sink_attn = []
    sink_dev = []
    nonsink_attn = []
    nonsink_dev = []

    for artifact in artifacts:
        sink_mask, _, _ = identify_sink_tokens(
            artifact["alpha"],
            artifact["values"],
            artifact["query_outputs"],
        )
        attn_factor, dev_factor = _compute_decomposition(
            artifact["alpha"],
            artifact["values"],
            artifact["query_outputs"],
        )
        sink_attn.append(attn_factor[sink_mask])
        sink_dev.append(dev_factor[sink_mask])
        nonsink_attn.append(attn_factor[~sink_mask])
        nonsink_dev.append(dev_factor[~sink_mask])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.boxplot(
        [torch.cat(sink_attn).numpy(), torch.cat(nonsink_attn).numpy()],
        labels=["Sink", "Non-sink"],
        showfliers=False,
    )
    ax1.set_title("Attention Factor")
    ax2.boxplot(
        [torch.cat(sink_dev).numpy(), torch.cat(nonsink_dev).numpy()],
        labels=["Sink", "Non-sink"],
        showfliers=False,
    )
    ax2.set_title("Value Deviation Factor")
    fig.suptitle(f"{model_name}: FETP Score Decomposition")
    fig.tight_layout()
    return fig

