from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd
import torch

from sink_analysis.analyze.exp1_sink_existence import plot_sink_existence
from sink_analysis.analyze.exp2_sink_retention import plot_sink_retention
from sink_analysis.analyze.exp3_score_decomposition import plot_score_decomposition
from sink_analysis.analyze.exp4_spatial_visuals import (
    render_full_comparison,
    select_representative_samples,
)
from sink_analysis.analyze.exp5_ablation import (
    build_ablation_selections,
    plot_ablation,
)
from sink_analysis.analyze.exp6_summary import generate_summary_table
from sink_analysis.analyze.report import (
    plot_summary_table,
    write_report,
    write_summary_outputs,
)
from sink_analysis.collect.merge_runs import merge_partial_records
from sink_analysis.collect.sink_metrics import identify_sink_tokens
from sink_analysis.collect.writer import (
    group_records,
    load_pt_records,
    read_json,
    write_artifact,
    write_json,
)
from sink_analysis.paths import SinkAnalysisPaths

ABLATION_CONFIG_ALIASES = {
    "A": "A: Attention",
    "A: ATTENTION": "A: Attention",
    "A: ATTENTION-ONLY": "A: Attention",
    "B": "B: Attention-Sink",
    "B: ATTENTION-SINK": "B: Attention-Sink",
    "B: ATTENTION−SINK": "B: Attention-Sink",
    "C": "C: FETP",
    "C: FETP": "C: FETP",
    "D": "D: FETP+Sink",
    "D: FETP+SINK": "D: FETP+Sink",
}

METHOD_TO_ABLATION_CONFIG = {
    "attention": "A: Attention",
    "fetp": "C: FETP",
    "ablation_b": "B: Attention-Sink",
    "ablation_d": "D: FETP+Sink",
}


def _keep_ratio_slug(value: str) -> str:
    return value.replace("%", "pct")


def _keep_ratio_label(value: str) -> str:
    return value.replace("pct", "%")


def _load_artifacts(artifact_root: Path) -> dict[str, list[dict]]:
    artifacts_by_model: dict[str, list[dict]] = {}
    if not artifact_root.exists():
        return artifacts_by_model
    for path in sorted(artifact_root.rglob("*.pt")):
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        artifacts_by_model.setdefault(artifact["model"], []).append(artifact)
    return artifacts_by_model


def _resolve_ablation_config(config_name: str) -> str:
    normalized = str(config_name).strip().upper()
    if normalized in ABLATION_CONFIG_ALIASES:
        return ABLATION_CONFIG_ALIASES[normalized]
    raise SystemExit(f"Unsupported ablation config: {config_name}")


def _build_per_sample_stats(artifacts_by_model: dict[str, list[dict]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name, artifacts in artifacts_by_model.items():
        for artifact in artifacts:
            sink_mask, _, _ = identify_sink_tokens(
                artifact["alpha"],
                artifact["values"],
                artifact["query_outputs"],
            )
            sink_indices = set(torch.where(sink_mask)[0].tolist())
            for keep_ratio, selections in artifact.get("selections", {}).items():
                row: dict[str, Any] = {
                    "model": model_name,
                    "sample_id": artifact["sample_id"],
                    "benchmark": artifact.get("benchmark"),
                    "keep_ratio": keep_ratio,
                    "sink_count": len(sink_indices),
                }
                for method_name in ("attention", "fetp", "mmtok"):
                    selection = selections.get(method_name)
                    if selection is None:
                        continue
                    selected = set(selection["indices"].tolist())
                    row[f"{method_name}_selected"] = len(selected)
                    row[f"{method_name}_sink_retention"] = (
                        len(selected & sink_indices) / max(1, len(sink_indices))
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def _latest_results_json(run_root: Path) -> Path | None:
    candidates = sorted(
        run_root.rglob("*_results.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _extract_score_from_results(result_path: Path | None) -> float | None:
    if result_path is None or not result_path.exists():
        return None
    payload = read_json(result_path)
    if not payload:
        return None
    task_results = payload.get("results", {})
    task_scores: list[float] = []
    for metrics in task_results.values():
        for metric_name, metric_value in metrics.items():
            name = metric_name.split(",")[0]
            if name.endswith("_stderr") or name == "alias":
                continue
            if isinstance(metric_value, (int, float)):
                score = float(metric_value)
                if score > 1.0:
                    score = score / 100.0
                task_scores.append(score)
                break
    if not task_scores:
        return None
    return float(sum(task_scores) / len(task_scores))


def _collect_ablation_results(paths: SinkAnalysisPaths) -> dict[str, dict[str, dict[str, float]]]:
    results_by_model: dict[str, dict[str, dict[str, float]]] = {}
    if not paths.eval_output_root.exists():
        return results_by_model

    for model_root in sorted(path for path in paths.eval_output_root.iterdir() if path.is_dir()):
        for method_slug, config_name in METHOD_TO_ABLATION_CONFIG.items():
            method_root = model_root / method_slug
            if not method_root.exists():
                continue
            for ratio_root in sorted(path for path in method_root.iterdir() if path.is_dir()):
                ratio_label = _keep_ratio_label(ratio_root.name)
                score = _extract_score_from_results(_latest_results_json(ratio_root))
                if score is None:
                    continue
                results_by_model.setdefault(model_root.name, {}).setdefault(ratio_label, {})[config_name] = score
    return results_by_model


def run_merge(args: argparse.Namespace) -> int:
    paths = SinkAnalysisPaths.from_repo_root(Path(args.repo_root))
    paths.ensure_output_dirs()
    records = load_pt_records(paths.partial_root)
    if not records:
        raise SystemExit("No partial artifacts found under sink_analysis/artifacts_partial.")

    merged_paths = []
    grouped = group_records(records, "model", "sample_id")
    for (_, _), partials in grouped.items():
        artifact = merge_partial_records(partials)
        merged_paths.append(str(write_artifact(paths.artifact_root, artifact)))
    write_json(paths.data_root / "merged_artifacts.json", merged_paths)
    return 0


def run_build_ablation(args: argparse.Namespace) -> int:
    paths = SinkAnalysisPaths.from_repo_root(Path(args.repo_root))
    artifacts_by_model = _load_artifacts(paths.artifact_root)
    payload = {}
    for model_name, artifacts in artifacts_by_model.items():
        payload[model_name] = {}
        for artifact in artifacts:
            sample_payload = {}
            for keep_ratio in artifact.get("selections", {}):
                sample_payload[keep_ratio] = {
                    name: tensor.tolist()
                    for name, tensor in build_ablation_selections(artifact, keep_ratio).items()
                }
            payload[model_name][artifact["sample_id"]] = sample_payload
    write_json(paths.data_root / "ablation_selections.json", payload)
    return 0


def run_analyze(args: argparse.Namespace) -> int:
    paths = SinkAnalysisPaths.from_repo_root(Path(args.repo_root))
    paths.ensure_output_dirs()
    artifacts_by_model = _load_artifacts(paths.artifact_root)
    if not artifacts_by_model:
        raise SystemExit("No merged artifacts found under sink_analysis/artifacts.")

    figure_paths: list[Path] = []
    for model_name, artifacts in artifacts_by_model.items():
        fig = plot_sink_existence(artifacts, model_name)
        path = paths.figure_root / f"exp1_sink_existence_{model_name}.pdf"
        fig.savefig(path)
        figure_paths.append(path)
        fig.clf()

        fig = plot_score_decomposition(artifacts, model_name)
        path = paths.figure_root / f"exp3_score_decomposition_{model_name}.pdf"
        fig.savefig(path)
        figure_paths.append(path)
        fig.clf()

        for index, artifact in enumerate(select_representative_samples(artifacts), start=1):
            fig = render_full_comparison(artifact)
            path = paths.figure_root / f"exp4_visual_{model_name}_sample{index}.pdf"
            fig.savefig(path)
            figure_paths.append(path)
            fig.clf()

    fig = plot_sink_retention(artifacts_by_model)
    path = paths.figure_root / "exp2_sink_retention.pdf"
    fig.savefig(path)
    figure_paths.append(path)
    fig.clf()

    summary_table = generate_summary_table(artifacts_by_model)
    _, summary_json = write_summary_outputs(paths.data_root, summary_table)

    per_sample_stats = _build_per_sample_stats(artifacts_by_model)
    per_sample_path = paths.data_root / "per_sample_stats.csv"
    per_sample_stats.to_csv(per_sample_path, index=False)

    fig = plot_summary_table(summary_table)
    summary_path = paths.figure_root / "exp6_summary_table.pdf"
    fig.savefig(summary_path)
    figure_paths.append(summary_path)
    fig.clf()

    ablation_results = _collect_ablation_results(paths)
    ablation_path = None
    if ablation_results:
        ablation_path = write_json(paths.data_root / "ablation_results.json", ablation_results)
        fig = plot_ablation(ablation_results)
        path = paths.figure_root / "exp5_ablation.pdf"
        fig.savefig(path)
        figure_paths.append(path)
        fig.clf()

    write_report(paths.report_path, summary_table, figure_paths, ablation_path)
    return 0


def run_rerun_ablation(args: argparse.Namespace) -> int:
    paths = SinkAnalysisPaths.from_repo_root(Path(args.repo_root))
    source_path = paths.data_root / "ablation_selections.json"
    selections = read_json(source_path)
    if not selections:
        raise SystemExit("Ablation selections not found. Run `build-ablation` first.")

    config_name = _resolve_ablation_config(args.config)
    config_slug = config_name.split(":", 1)[0].lower().replace(" ", "_")
    overrides: dict[str, dict[str, dict[str, list[int]]]] = {}
    for model_name, samples in selections.items():
        model_payload: dict[str, dict[str, list[int]]] = {}
        for sample_id, ratio_payload in samples.items():
            sample_payload: dict[str, list[int]] = {}
            for keep_ratio, config_payload in ratio_payload.items():
                indices = config_payload.get(config_name)
                if indices is not None:
                    sample_payload[keep_ratio] = indices
            if sample_payload:
                model_payload[sample_id] = sample_payload
        if model_payload:
            overrides[model_name] = model_payload

    output_path = paths.data_root / f"{config_slug}_overrides.json"
    write_json(output_path, overrides)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("merge")
    subparsers.add_parser("build-ablation")
    subparsers.add_parser("analyze")
    rerun = subparsers.add_parser("rerun-ablation")
    rerun.add_argument("--config", required=True)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "merge":
        return run_merge(args)
    if args.command == "build-ablation":
        return run_build_ablation(args)
    if args.command == "analyze":
        return run_analyze(args)
    if args.command == "rerun-ablation":
        return run_rerun_ablation(args)
    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
