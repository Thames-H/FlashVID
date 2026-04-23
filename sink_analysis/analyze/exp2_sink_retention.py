from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from sink_analysis.analyze.ratio_utils import sort_keep_ratios
from sink_analysis.collect.sink_metrics import identify_sink_tokens


def compute_sink_retention(artifacts: list[dict], keep_ratio: str) -> dict[str, float]:
    results = {method: [] for method in ("attention", "mmtok", "fetp")}
    for artifact in artifacts:
        sink_mask, _, _ = identify_sink_tokens(
            artifact["alpha"],
            artifact["values"],
            artifact["query_outputs"],
        )
        sink_indices = set(torch.where(sink_mask)[0].tolist())
        if not sink_indices:
            continue
        ratio_selection = artifact.get("selections", {}).get(keep_ratio, {})
        for method in results:
            selection = ratio_selection.get(method)
            if selection is None:
                continue
            selected = set(selection["indices"].tolist())
            results[method].append(len(selected & sink_indices) / len(sink_indices))
    return {
        method: float(np.mean(values)) if values else float("nan")
        for method, values in results.items()
    }


def plot_sink_retention(artifacts_by_model: dict[str, list[dict]]):
    ratios = sort_keep_ratios(
        keep_ratio
        for artifacts in artifacts_by_model.values()
        for artifact in artifacts
        for keep_ratio in artifact.get("selections", {})
    )
    methods = ["attention", "mmtok", "fetp"]
    colors = {"attention": "#e74c3c", "mmtok": "#f39c12", "fetp": "#2ecc71"}
    fig, axes = plt.subplots(
        1,
        max(1, len(artifacts_by_model)),
        figsize=(6 * max(1, len(artifacts_by_model)), 5),
        sharey=True,
    )
    if len(artifacts_by_model) == 1:
        axes = [axes]

    for ax, (model_name, artifacts) in zip(axes, artifacts_by_model.items()):
        x = np.arange(len(ratios))
        width = 0.25
        for offset, method in enumerate(methods):
            values = [compute_sink_retention(artifacts, ratio)[method] for ratio in ratios]
            ax.bar(x + offset * width, values, width, color=colors[method], label=method)
        ax.set_xticks(x + width)
        ax.set_xticklabels(ratios)
        ax.set_ylim(0, 1)
        ax.set_title(model_name)
        ax.set_xlabel("Keep Ratio")
        ax.set_ylabel("Sink Token Retention Rate")
        ax.legend()

    fig.tight_layout()
    return fig
