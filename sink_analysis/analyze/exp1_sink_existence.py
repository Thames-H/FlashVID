from __future__ import annotations

import matplotlib.pyplot as plt
import torch

from sink_analysis.collect.sink_metrics import identify_sink_tokens


def _minmax(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    denominator = (values.max() - values.min()).clamp(min=1e-8)
    return (values - values.min()) / denominator


def plot_sink_existence(artifacts: list[dict], model_name: str):
    fig, ax = plt.subplots(figsize=(8, 8))
    all_attn = []
    all_dev = []

    for artifact in artifacts:
        _, mean_attn, value_dev = identify_sink_tokens(
            artifact["alpha"],
            artifact["values"],
            artifact["query_outputs"],
        )
        all_attn.append(_minmax(mean_attn))
        all_dev.append(_minmax(value_dev))

    if all_attn and all_dev:
        attn_values = torch.cat(all_attn).numpy()
        dev_values = torch.cat(all_dev).numpy()
        ax.hexbin(attn_values, dev_values, gridsize=60, cmap="YlOrRd", mincnt=1)
        plt.colorbar(ax.collections[0], ax=ax, label="Token Count")

    ax.set_xlabel("Normalized Mean Attention")
    ax.set_ylabel("Normalized Value Deviation")
    ax.set_title(f"{model_name}: Attention vs Value Deviation")
    ax.axvspan(0.7, 1.0, ymin=0.0, ymax=0.3, color="blue", alpha=0.08)
    ax.text(0.84, 0.15, "Sink\nZone", color="blue", ha="center", va="center")
    fig.tight_layout()
    return fig

