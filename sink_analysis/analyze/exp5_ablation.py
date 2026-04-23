from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from sink_analysis.collect.sink_metrics import identify_sink_tokens


def build_ablation_selections(artifact: dict, keep_ratio: str) -> dict[str, torch.Tensor]:
    selected = artifact["selections"][keep_ratio]
    k = len(selected["fetp"]["indices"])

    sink_mask, _, _ = identify_sink_tokens(
        artifact["alpha"],
        artifact["values"],
        artifact["query_outputs"],
    )
    sink_set = set(torch.where(sink_mask)[0].tolist())

    attention_scores = selected["attention"]["scores"]
    fetp_scores = selected["fetp"]["scores"]

    attention_rank = attention_scores.argsort(descending=True).tolist()
    config_a = selected["attention"]["indices"]
    config_b = torch.tensor(
        [idx for idx in attention_rank if idx not in sink_set][:k],
        dtype=torch.long,
    )

    config_c = selected["fetp"]["indices"]
    config_d_selected = set(config_c.tolist())
    removable = sorted(
        (idx for idx in config_d_selected if idx not in sink_set),
        key=lambda idx: float(fetp_scores[idx]),
    )
    missing_sinks = sorted(
        (idx for idx in sink_set if idx not in config_d_selected),
        key=lambda idx: float(attention_scores[idx]),
        reverse=True,
    )
    for sink_idx, remove_idx in zip(missing_sinks, removable):
        config_d_selected.remove(remove_idx)
        config_d_selected.add(sink_idx)

    return {
        "A: Attention": config_a,
        "B: Attention-Sink": config_b,
        "C: FETP": config_c,
        "D: FETP+Sink": torch.tensor(sorted(config_d_selected), dtype=torch.long),
    }


def plot_ablation(results_by_model: dict[str, dict[str, dict[str, float]]]):
    if not results_by_model:
        raise ValueError("No ablation results available for plotting.")

    configs = [
        "A: Attention",
        "B: Attention-Sink",
        "C: FETP",
        "D: FETP+Sink",
    ]
    colors = {
        "A: Attention": "#e74c3c",
        "B: Attention-Sink": "#e67e22",
        "C: FETP": "#2ecc71",
        "D: FETP+Sink": "#95a5a6",
    }
    ratios = ["25%", "50%", "75%"]
    fig, axes = plt.subplots(1, len(results_by_model), figsize=(7 * len(results_by_model), 6), sharey=True)
    if len(results_by_model) == 1:
        axes = [axes]

    for ax, (model_name, ratio_payload) in zip(axes, results_by_model.items()):
        x = np.arange(len(ratios))
        width = 0.2
        for offset, config_name in enumerate(configs):
            values = [ratio_payload.get(ratio, {}).get(config_name, np.nan) for ratio in ratios]
            ax.bar(
                x + offset * width,
                values,
                width,
                color=colors[config_name],
                label=config_name,
            )
        ax.set_xticks(x + 1.5 * width)
        ax.set_xticklabels(ratios)
        ax.set_xlabel("Keep Ratio")
        ax.set_ylabel("Accuracy")
        ax.set_title(model_name)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9)

    fig.suptitle("Ablation: Effect of Sink Token Removal", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig
