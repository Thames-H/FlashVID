from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sink_analysis.collect.writer import write_json


def write_report(
    report_path: Path,
    summary_table: pd.DataFrame,
    figure_paths: list[Path],
    ablation_path: Path | None = None,
) -> Path:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Sink Analysis Report",
        "",
        "## Figures",
        "",
    ]
    for figure_path in figure_paths:
        lines.append(f"- `{figure_path}`")
    lines.extend(["", "## Summary Table", "", summary_table.to_markdown(index=False)])
    if ablation_path is not None:
        lines.extend(["", "## Ablation Results", "", f"- `{ablation_path}`"])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def write_summary_outputs(data_root: Path, summary_table: pd.DataFrame) -> tuple[Path, Path]:
    data_root.mkdir(parents=True, exist_ok=True)
    csv_path = data_root / "per_model_summary.csv"
    json_path = data_root / "summary.json"
    summary_table.to_csv(csv_path, index=False)
    write_json(json_path, summary_table.to_dict(orient="records"))
    return csv_path, json_path


def plot_summary_table(summary_table: pd.DataFrame):
    fig_height = max(2.5, 1.0 + 0.45 * (len(summary_table) + 1))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=summary_table.values,
        colLabels=summary_table.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    ax.set_title("Sink Analysis Summary", fontsize=14, pad=14)
    fig.tight_layout()
    return fig
