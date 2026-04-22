import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _resolve_layer_root(base_root: Path, target_layer: int | None) -> Path:
    if target_layer is None:
        return base_root

    layer_dir = f"layer_{int(target_layer)}"
    if base_root.name == layer_dir:
        return base_root
    return base_root / layer_dir


def _load_artifacts(artifact_root: Path, method_name: str) -> Dict[Tuple[str, str], dict]:
    method_dir = artifact_root / "artifacts" / method_name
    if not method_dir.exists():
        return {}

    artifacts = {}
    for path in sorted(method_dir.glob("*.pt")):
        payload = torch.load(path, map_location="cpu")
        task_name = str(payload["task_name"])
        doc_id = str(payload["doc_id"])
        payload["_artifact_path"] = str(path)
        artifacts[(task_name, doc_id)] = payload
    return artifacts


def _compute_pca_coords(visual_embeddings: torch.Tensor) -> torch.Tensor:
    x = visual_embeddings.float()
    if x.ndim != 2:
        raise ValueError(f"visual_embeddings must be 2D, got {tuple(x.shape)}")
    if x.shape[0] == 0:
        return torch.empty((0, 2), dtype=torch.float32)
    if x.shape[0] == 1:
        return torch.zeros((1, 2), dtype=torch.float32)

    x = x - x.mean(dim=0, keepdim=True)
    q = min(4, min(x.shape[0], x.shape[1]))
    if q < 2:
        first = x[:, :1]
        second = torch.zeros_like(first)
        return torch.cat([first, second], dim=1)

    _, _, v = torch.pca_lowrank(x, q=q, center=False)
    coords = x @ v[:, :2]
    if coords.shape[1] < 2:
        coords = torch.cat([coords, torch.zeros_like(coords)], dim=1)
    return coords[:, :2]


def _normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    scores = scores.float()
    if scores.numel() == 0:
        return scores
    min_value = scores.min()
    max_value = scores.max()
    scale = (max_value - min_value).clamp_min(1e-8)
    return (scores - min_value) / scale


def _pairwise_mean_distance(coords: torch.Tensor) -> float:
    if coords.shape[0] < 2:
        return 0.0
    dist = torch.cdist(coords.float(), coords.float())
    upper = dist[torch.triu(torch.ones_like(dist, dtype=torch.bool), diagonal=1)]
    return float(upper.mean().item()) if upper.numel() > 0 else 0.0


def _coverage_area(coords: torch.Tensor) -> float:
    if coords.shape[0] < 2:
        return 0.0
    centered = coords.float() - coords.float().mean(dim=0, keepdim=True)
    cov = centered.T @ centered / max(1, coords.shape[0] - 1)
    det = torch.det(cov).clamp_min(0.0)
    return float(det.sqrt().item())


def _edge_bias(subset_coords: torch.Tensor, all_coords: torch.Tensor) -> float:
    if subset_coords.numel() == 0 or all_coords.numel() == 0:
        return 0.0
    center = all_coords.float().mean(dim=0, keepdim=True)
    all_radii = torch.norm(all_coords.float() - center, dim=1)
    denom = all_radii.mean().clamp_min(1e-8)
    subset_radii = torch.norm(subset_coords.float() - center, dim=1)
    return float((subset_radii.mean() / denom).item())


def _jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    set_a = set(int(x) for x in a.tolist())
    set_b = set(int(x) for x in b.tolist())
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def _compute_subset_groups(
    fetp_keep: torch.Tensor,
    attn_keep: torch.Tensor,
    mmtok_keep: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    fetp = set(int(x) for x in fetp_keep.long().tolist())
    attn = set(int(x) for x in attn_keep.long().tolist())
    mmtok = set(int(x) for x in mmtok_keep.long().tolist())

    all_three = fetp & attn & mmtok
    fetp_attention = (fetp & attn) - all_three
    fetp_mmtok = (fetp & mmtok) - all_three
    attention_mmtok = (attn & mmtok) - all_three
    fetp_only = fetp - attn - mmtok
    attention_only = attn - fetp - mmtok
    mmtok_only = mmtok - fetp - attn

    return {
        "fetp_only": torch.tensor(sorted(fetp_only), dtype=torch.long),
        "attention_only": torch.tensor(sorted(attention_only), dtype=torch.long),
        "mmtok_only": torch.tensor(sorted(mmtok_only), dtype=torch.long),
        "fetp_attention": torch.tensor(sorted(fetp_attention), dtype=torch.long),
        "fetp_mmtok": torch.tensor(sorted(fetp_mmtok), dtype=torch.long),
        "attention_mmtok": torch.tensor(sorted(attention_mmtok), dtype=torch.long),
        "all_three": torch.tensor(sorted(all_three), dtype=torch.long),
    }


def _plot_subset_pca(
    coords: torch.Tensor,
    fetp_keep: torch.Tensor,
    attn_keep: torch.Tensor,
    mmtok_keep: torch.Tensor,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    subset_groups = _compute_subset_groups(fetp_keep, attn_keep, mmtok_keep)

    ax.scatter(coords[:, 0], coords[:, 1], s=14, c="#bdbdbd", alpha=0.35, label="All visual tokens", zorder=1)

    group_styles = [
        ("fetp_only", "FETP only", dict(marker="o", s=42, c="#d73027", alpha=0.92, linewidths=0.8, edgecolors="white", zorder=4)),
        (
            "attention_only",
            "Attention-only only",
            dict(marker="s", s=42, c="#4575b4", alpha=0.90, linewidths=0.8, edgecolors="white", zorder=4),
        ),
        ("mmtok_only", "MMTok only", dict(marker="^", s=50, c="#1a9850", alpha=0.90, linewidths=0.8, edgecolors="white", zorder=4)),
        (
            "fetp_attention",
            "FETP ∩ Attention",
            dict(marker="D", s=58, facecolors="white", edgecolors="#984ea3", linewidths=1.8, alpha=1.0, zorder=5),
        ),
        (
            "fetp_mmtok",
            "FETP ∩ MMTok",
            dict(marker="P", s=64, facecolors="white", edgecolors="#ff7f00", linewidths=1.8, alpha=1.0, zorder=5),
        ),
        (
            "attention_mmtok",
            "Attention ∩ MMTok",
            dict(marker="X", s=64, facecolors="white", edgecolors="#4daf4a", linewidths=1.8, alpha=1.0, zorder=5),
        ),
        ("all_three", "All three", dict(marker="*", s=110, c="#111111", alpha=0.95, linewidths=0.6, edgecolors="white", zorder=6)),
    ]

    for group_name, label, style in group_styles:
        indices = subset_groups[group_name]
        if indices.numel() == 0:
            continue
        ax.scatter(
            coords[indices, 0],
            coords[indices, 1],
            label=f"{label} ({indices.numel()})",
            **style,
        )
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_score_landscape(
    coords: torch.Tensor,
    fetp_scores: torch.Tensor,
    attn_scores: torch.Tensor,
    mmtok_scores: torch.Tensor,
    title: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
    method_panels = [
        ("FETP", _normalize_scores(fetp_scores), axes[0]),
        ("Attention-only", _normalize_scores(attn_scores), axes[1]),
        ("MMTok", _normalize_scores(mmtok_scores), axes[2]),
    ]

    for name, scores, ax in method_panels:
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=scores,
            s=20,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_title(name)
        ax.set_xlabel("PC1")
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("PC2")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _iter_matched_samples(
    artifact_root: Path,
) -> Iterable[Tuple[Tuple[str, str], dict, dict]]:
    fetp_artifacts = _load_artifacts(artifact_root, "fetp")
    mmtok_artifacts = _load_artifacts(artifact_root, "mmtok")
    common_keys = sorted(set(fetp_artifacts) & set(mmtok_artifacts))
    for key in common_keys:
        yield key, fetp_artifacts[key], mmtok_artifacts[key]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-samples", type=int, default=3)
    parser.add_argument("--target-layer", type=int, default=None)
    parser.add_argument(
        "--report-title",
        type=str,
        default="Qwen3-VL Token Pruning PCA Compare",
    )
    args = parser.parse_args()

    artifact_root = _resolve_layer_root(args.artifact_root, args.target_layer)
    output_dir = _resolve_layer_root(args.output_dir, args.target_layer)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    matched_rows: List[dict] = []
    report_keys: List[str] = []

    for (task_name, doc_id), fetp_artifact, mmtok_artifact in _iter_matched_samples(artifact_root):
        visual_embeddings = fetp_artifact["visual_embeddings"].float()
        coords = _compute_pca_coords(visual_embeddings)

        fetp_keep = fetp_artifact["selection"]["fetp_keep_local"].long()
        attn_keep = fetp_artifact["selection"]["attention_only_keep_local"].long()
        mmtok_keep = mmtok_artifact["selection"]["mmtok_keep_local"].long()

        fetp_scores = fetp_artifact["scores"]["fetp"].float()
        attn_scores = fetp_artifact["scores"]["attention_only"].float()
        mmtok_scores = mmtok_artifact["scores"]["initial_marginal_gain"].float()

        sample_slug = f"{task_name}__doc{doc_id}"
        question_text = str(fetp_artifact.get("question_text", "")).strip().replace("\n", " ")
        title = f"{task_name} / doc {doc_id}"

        subset_plot = plots_dir / f"{sample_slug}__subset.png"
        landscape_plot = plots_dir / f"{sample_slug}__landscape.png"
        _plot_subset_pca(coords, fetp_keep, attn_keep, mmtok_keep, title, subset_plot)
        _plot_score_landscape(coords, fetp_scores, attn_scores, mmtok_scores, title, landscape_plot)

        fetp_coords = coords[fetp_keep]
        attn_coords = coords[attn_keep]
        mmtok_coords = coords[mmtok_keep]
        metrics = {
            "task_name": task_name,
            "doc_id": doc_id,
            "num_tokens": int(coords.shape[0]),
            "num_keep": int(fetp_keep.numel()),
            "fetp_spread": _pairwise_mean_distance(fetp_coords),
            "attn_spread": _pairwise_mean_distance(attn_coords),
            "mmtok_spread": _pairwise_mean_distance(mmtok_coords),
            "fetp_area": _coverage_area(fetp_coords),
            "attn_area": _coverage_area(attn_coords),
            "mmtok_area": _coverage_area(mmtok_coords),
            "fetp_edge_bias": _edge_bias(fetp_coords, coords),
            "attn_edge_bias": _edge_bias(attn_coords, coords),
            "mmtok_edge_bias": _edge_bias(mmtok_coords, coords),
            "jaccard_fetp_attn": _jaccard(fetp_keep, attn_keep),
            "jaccard_fetp_mmtok": _jaccard(fetp_keep, mmtok_keep),
            "jaccard_attn_mmtok": _jaccard(attn_keep, mmtok_keep),
            "question_text": question_text,
            "subset_plot": str(subset_plot.relative_to(output_dir)),
            "landscape_plot": str(landscape_plot.relative_to(output_dir)),
        }
        matched_rows.append(metrics)
        if len(report_keys) < args.report_samples:
            report_keys.append(sample_slug)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "artifact_root": str(artifact_root),
                "num_matched_samples": len(matched_rows),
                "samples": matched_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report_lines = [
        f"# {args.report_title}",
        "",
        f"- Artifact root: `{artifact_root}`",
        f"- Matched samples: `{len(matched_rows)}`",
        "",
        "## Summary Table",
        "",
        "| Task | Doc | N | K | FETP spread | Attn spread | MMTok spread | FETP edge | Attn edge | MMTok edge | F/A | F/M | A/M |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in matched_rows:
        report_lines.append(
            f"| {row['task_name']} | {row['doc_id']} | {row['num_tokens']} | {row['num_keep']} | "
            f"{row['fetp_spread']:.4f} | {row['attn_spread']:.4f} | {row['mmtok_spread']:.4f} | "
            f"{row['fetp_edge_bias']:.4f} | {row['attn_edge_bias']:.4f} | {row['mmtok_edge_bias']:.4f} | "
            f"{row['jaccard_fetp_attn']:.4f} | {row['jaccard_fetp_mmtok']:.4f} | {row['jaccard_attn_mmtok']:.4f} |"
        )

    report_lines.extend(["", "## Representative Samples", ""])
    for row in matched_rows[: args.report_samples]:
        report_lines.extend(
            [
                f"### {row['task_name']} / doc {row['doc_id']}",
                "",
                f"- Question: {row['question_text']}",
                f"- Subset PCA: ![]({row['subset_plot']})",
                f"- Score landscape: ![]({row['landscape_plot']})",
                "",
            ]
        )

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Wrote summary to {summary_path}")
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
