import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_artifacts(
    artifact_root: Path, method_name: str
) -> Dict[Tuple[str, str], dict]:
    method_dir = artifact_root / "artifacts" / method_name
    if not method_dir.exists():
        return {}

    artifacts: Dict[Tuple[str, str], dict] = {}
    for path in sorted(method_dir.glob("*.pt")):
        payload = torch.load(path, map_location="cpu")
        task_name = str(payload["task_name"])
        doc_id = str(payload["doc_id"])
        payload["_artifact_path"] = str(path)
        artifacts[(task_name, doc_id)] = payload
    return artifacts


def _selection_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    set_a = set(int(x) for x in a.long().tolist())
    set_b = set(int(x) for x in b.long().tolist())
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def _compute_sink_indices(
    attention_keep: torch.Tensor, fetp_keep: torch.Tensor
) -> torch.Tensor:
    sink = sorted(
        set(int(x) for x in attention_keep.long().tolist())
        - set(int(x) for x in fetp_keep.long().tolist())
    )
    return torch.tensor(sink, dtype=torch.long)


def _tensor_image_to_float(image_preview: torch.Tensor) -> torch.Tensor:
    image = image_preview.detach().cpu()
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(
            f"image_preview must have shape (H, W, 3), got {tuple(image.shape)}"
        )
    return image.float() / 255.0


def _selection_source_counts(
    keep_indices: torch.Tensor,
    token_is_spatial: torch.Tensor,
    token_source: List[str],
) -> dict:
    counts = {
        "base": 0,
        "crop": 0,
        "non_spatial": 0,
    }
    for idx in keep_indices.long().tolist():
        source = token_source[int(idx)]
        if not bool(token_is_spatial[int(idx)].item()):
            counts["non_spatial"] += 1
        elif source == "base":
            counts["base"] += 1
        else:
            counts["crop"] += 1
    return counts


def _overlay_token_boxes(
    image_preview: torch.Tensor,
    token_boxes: torch.Tensor,
    token_is_spatial: torch.Tensor,
    keep_indices: torch.Tensor,
    color: Tuple[float, float, float],
    token_source: Optional[List[str]] = None,
    allowed_sources: Optional[set[str]] = None,
    alpha: float = 0.45,
    dim_factor: float = 0.55,
) -> torch.Tensor:
    image = _tensor_image_to_float(image_preview)
    overlay = image * dim_factor
    color_tensor = torch.tensor(color, dtype=torch.float32).view(1, 1, 3)

    for idx in keep_indices.long().tolist():
        idx = int(idx)
        if not bool(token_is_spatial[idx].item()):
            continue
        if allowed_sources is not None and token_source is not None:
            if token_source[idx] not in allowed_sources:
                continue
        x0, y0, x1, y1 = token_boxes[idx].tolist()
        left = max(0, min(image.shape[1] - 1, int(round(x0))))
        top = max(0, min(image.shape[0] - 1, int(round(y0))))
        right = max(left + 1, min(image.shape[1], int(round(x1))))
        bottom = max(top + 1, min(image.shape[0], int(round(y1))))
        patch = overlay[top:bottom, left:right]
        overlay[top:bottom, left:right] = torch.clamp(
            patch * (1.0 - alpha) + color_tensor * alpha,
            0.0,
            1.0,
        )
    return overlay


def _prepare_sample_metrics(
    fetp_artifact: dict,
    mmtok_artifact: dict,
) -> dict:
    attn_keep = fetp_artifact["selection"]["attention_only_keep_local"].long()
    fetp_keep = fetp_artifact["selection"]["fetp_keep_local"].long()
    mmtok_keep = mmtok_artifact["selection"]["mmtok_keep_local"].long()

    token_boxes = fetp_artifact.get("token_boxes")
    token_is_spatial = fetp_artifact.get("token_is_spatial")
    token_source = fetp_artifact.get("token_source")
    num_tokens = int(fetp_artifact["metadata"]["n_visual_tokens_scored"])

    eligible = True
    skip_reason = None
    if "visual_compare_eligible" in fetp_artifact:
        eligible = bool(fetp_artifact.get("visual_compare_eligible"))
        skip_reason = fetp_artifact.get("visual_compare_skip_reason")

    if eligible and token_boxes is None:
        eligible = False
        skip_reason = "missing_token_boxes"
    elif eligible and token_is_spatial is None:
        eligible = False
        skip_reason = "missing_token_is_spatial"
    elif eligible and token_source is None:
        eligible = False
        skip_reason = "missing_token_source"
    elif eligible and int(token_boxes.shape[0]) != num_tokens:
        eligible = False
        skip_reason = "token_box_count_mismatch"

    sink_indices = _compute_sink_indices(attn_keep, fetp_keep)
    attn_source_counts = _selection_source_counts(
        attn_keep,
        token_is_spatial,
        token_source,
    )
    fetp_source_counts = _selection_source_counts(
        fetp_keep,
        token_is_spatial,
        token_source,
    )
    mmtok_source_counts = _selection_source_counts(
        mmtok_keep,
        token_is_spatial,
        token_source,
    )
    sink_source_counts = _selection_source_counts(
        sink_indices,
        token_is_spatial,
        token_source,
    )

    return {
        "task_name": str(fetp_artifact["task_name"]),
        "doc_id": str(fetp_artifact["doc_id"]),
        "question_text": str(fetp_artifact.get("question_text", "")).strip().replace(
            "\n",
            " ",
        ),
        "eligible": eligible,
        "skip_reason": skip_reason,
        "iou_fetp_attn": _selection_iou(fetp_keep, attn_keep),
        "iou_fetp_mmtok": _selection_iou(fetp_keep, mmtok_keep),
        "sink_ratio": float(sink_indices.numel() / max(1, attn_keep.numel())),
        "sink_count": int(sink_indices.numel()),
        "attn_source_counts": attn_source_counts,
        "fetp_source_counts": fetp_source_counts,
        "mmtok_source_counts": mmtok_source_counts,
        "sink_source_counts": sink_source_counts,
    }


def _select_representative_sample(rows: List[dict]) -> Optional[dict]:
    eligible_rows = [row for row in rows if row.get("eligible")]
    if not eligible_rows:
        return None
    return min(eligible_rows, key=lambda row: row["iou_fetp_attn"])


def _plot_overview(
    fetp_artifact: dict,
    mmtok_artifact: dict,
    output_path: Path,
) -> None:
    image_preview = fetp_artifact["image_preview"]
    token_boxes = fetp_artifact["token_boxes"]
    token_is_spatial = fetp_artifact["token_is_spatial"]
    token_source = fetp_artifact["token_source"]

    panels = [
        ("Original", _tensor_image_to_float(image_preview)),
        (
            "Attention-only",
            _overlay_token_boxes(
                image_preview,
                token_boxes,
                token_is_spatial,
                fetp_artifact["selection"]["attention_only_keep_local"].long(),
                (0.14, 0.37, 0.88),
                token_source=token_source,
            ),
        ),
        (
            "MMTok",
            _overlay_token_boxes(
                image_preview,
                token_boxes,
                token_is_spatial,
                mmtok_artifact["selection"]["mmtok_keep_local"].long(),
                (0.12, 0.65, 0.30),
                token_source=token_source,
            ),
        ),
        (
            "FETP",
            _overlay_token_boxes(
                image_preview,
                token_boxes,
                token_is_spatial,
                fetp_artifact["selection"]["fetp_keep_local"].long(),
                (0.85, 0.22, 0.18),
                token_source=token_source,
            ),
        ),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, (title, image) in zip(axes, panels):
        ax.imshow(image.numpy())
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_structure(
    fetp_artifact: dict,
    mmtok_artifact: dict,
    output_path: Path,
) -> None:
    image_preview = fetp_artifact["image_preview"]
    token_boxes = fetp_artifact["token_boxes"]
    token_is_spatial = fetp_artifact["token_is_spatial"]
    token_source = fetp_artifact["token_source"]
    fetp_keep = fetp_artifact["selection"]["fetp_keep_local"].long()
    attn_keep = fetp_artifact["selection"]["attention_only_keep_local"].long()
    mmtok_keep = mmtok_artifact["selection"]["mmtok_keep_local"].long()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    base_overlay = _overlay_token_boxes(
        image_preview,
        token_boxes,
        token_is_spatial,
        fetp_keep,
        (0.95, 0.57, 0.10),
        token_source=token_source,
        allowed_sources={"base"},
    )
    crop_overlay = _overlay_token_boxes(
        image_preview,
        token_boxes,
        token_is_spatial,
        fetp_keep,
        (0.74, 0.24, 0.14),
        token_source=token_source,
        allowed_sources={"crop"},
    )
    axes[0].imshow(base_overlay.numpy())
    axes[0].set_title("FETP base tokens")
    axes[0].axis("off")
    axes[1].imshow(crop_overlay.numpy())
    axes[1].set_title("FETP crop tokens")
    axes[1].axis("off")

    attn_counts = _selection_source_counts(attn_keep, token_is_spatial, token_source)
    mmtok_counts = _selection_source_counts(
        mmtok_keep,
        token_is_spatial,
        token_source,
    )
    fetp_counts = _selection_source_counts(fetp_keep, token_is_spatial, token_source)
    methods = ["Attention", "MMTok", "FETP"]
    base_values = [
        attn_counts["base"],
        mmtok_counts["base"],
        fetp_counts["base"],
    ]
    crop_values = [
        attn_counts["crop"],
        mmtok_counts["crop"],
        fetp_counts["crop"],
    ]
    non_spatial_values = [
        attn_counts["non_spatial"],
        mmtok_counts["non_spatial"],
        fetp_counts["non_spatial"],
    ]
    axes[2].bar(methods, base_values, label="Base", color="#fdae61")
    axes[2].bar(
        methods,
        crop_values,
        bottom=base_values,
        label="Crop",
        color="#d73027",
    )
    axes[2].bar(
        methods,
        non_spatial_values,
        bottom=[
            base_values[idx] + crop_values[idx]
            for idx in range(len(methods))
        ],
        label="Non-spatial",
        color="#636363",
    )
    axes[2].set_title("Selection composition")
    axes[2].set_ylabel("Token count")
    axes[2].legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_sink(
    fetp_artifact: dict,
    output_path: Path,
) -> None:
    image_preview = fetp_artifact["image_preview"]
    token_boxes = fetp_artifact["token_boxes"]
    token_is_spatial = fetp_artifact["token_is_spatial"]
    token_source = fetp_artifact["token_source"]
    attn_keep = fetp_artifact["selection"]["attention_only_keep_local"].long()
    fetp_keep = fetp_artifact["selection"]["fetp_keep_local"].long()
    sink_indices = _compute_sink_indices(attn_keep, fetp_keep)
    sink_counts = _selection_source_counts(
        sink_indices,
        token_is_spatial,
        token_source,
    )

    base_image = _tensor_image_to_float(image_preview)
    attention_overlay = _overlay_token_boxes(
        image_preview,
        token_boxes,
        token_is_spatial,
        attn_keep,
        (0.14, 0.37, 0.88),
        token_source=token_source,
    )
    sink_highlight = _overlay_token_boxes(
        image_preview,
        token_boxes,
        token_is_spatial,
        sink_indices,
        (1.0, 0.78, 0.12),
        token_source=token_source,
        alpha=0.65,
        dim_factor=0.65,
    )
    fetp_overlay = _overlay_token_boxes(
        image_preview,
        token_boxes,
        token_is_spatial,
        fetp_keep,
        (0.85, 0.22, 0.18),
        token_source=token_source,
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    panels = [
        ("Original", base_image),
        (
            f"Attention + sink\n(spatial={sink_counts['base'] + sink_counts['crop']}, non-spatial={sink_counts['non_spatial']})",
            torch.clamp(
                (attention_overlay * 0.75) + (sink_highlight * 0.55),
                0.0,
                1.0,
            ),
        ),
        ("FETP", fetp_overlay),
    ]
    for ax, (title, image) in zip(axes, panels):
        ax.imshow(image.numpy())
        ax.set_title(title)
        ax.axis("off")
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


def generate_visual_compare_report(
    artifact_root: Path,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    matched_artifacts: Dict[Tuple[str, str], Tuple[dict, dict]] = {}
    for key, fetp_artifact, mmtok_artifact in _iter_matched_samples(artifact_root):
        rows.append(_prepare_sample_metrics(fetp_artifact, mmtok_artifact))
        matched_artifacts[key] = (fetp_artifact, mmtok_artifact)

    representative = _select_representative_sample(rows)
    if representative is not None:
        rep_key = (
            representative["task_name"],
            representative["doc_id"],
        )
        fetp_artifact, mmtok_artifact = matched_artifacts[rep_key]
        sample_slug = f"{representative['task_name']}__doc{representative['doc_id']}"
        overview_plot = plots_dir / f"{sample_slug}__overview.png"
        structure_plot = plots_dir / f"{sample_slug}__structure.png"
        sink_plot = plots_dir / f"{sample_slug}__sink.png"
        _plot_overview(fetp_artifact, mmtok_artifact, overview_plot)
        _plot_structure(fetp_artifact, mmtok_artifact, structure_plot)
        _plot_sink(fetp_artifact, sink_plot)
        representative["overview_plot"] = str(overview_plot.relative_to(output_dir))
        representative["structure_plot"] = str(
            structure_plot.relative_to(output_dir)
        )
        representative["sink_plot"] = str(sink_plot.relative_to(output_dir))

    average_row = None
    if rows:
        average_row = {
            "task_name": "Average",
            "doc_id": "-",
            "iou_fetp_attn": sum(row["iou_fetp_attn"] for row in rows) / len(rows),
            "iou_fetp_mmtok": sum(row["iou_fetp_mmtok"] for row in rows) / len(rows),
            "sink_ratio": sum(row["sink_ratio"] for row in rows) / len(rows),
            "fetp_non_spatial": sum(
                row["fetp_source_counts"]["non_spatial"] for row in rows
            )
            / len(rows),
            "attn_non_spatial": sum(
                row["attn_source_counts"]["non_spatial"] for row in rows
            )
            / len(rows),
            "mmtok_non_spatial": sum(
                row["mmtok_source_counts"]["non_spatial"] for row in rows
            )
            / len(rows),
        }

    summary = {
        "artifact_root": str(artifact_root),
        "num_matched_samples": len(rows),
        "representative_sample": representative,
        "average": average_row,
        "samples": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report_lines = [
        "# LLaVA-OneVision Visual Compare",
        "",
        f"- Artifact root: `{artifact_root}`",
        f"- Matched samples: `{len(rows)}`",
        "",
        "## Summary Table",
        "",
        "| Sample | FETP vs Attn IoU | FETP vs MMTok IoU | Attn-only ratio removed by FETP | FETP non-spatial | Attn non-spatial | MMTok non-spatial | Eligible |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        sample_name = f"{row['task_name']}/{row['doc_id']}"
        eligibility = "yes"
        if not row["eligible"]:
            eligibility = f"no ({row['skip_reason']})"
        report_lines.append(
            f"| {sample_name} | {row['iou_fetp_attn']:.4f} | {row['iou_fetp_mmtok']:.4f} | {row['sink_ratio']:.4f} | "
            f"{row['fetp_source_counts']['non_spatial']} | {row['attn_source_counts']['non_spatial']} | "
            f"{row['mmtok_source_counts']['non_spatial']} | {eligibility} |"
        )
    if average_row is not None:
        report_lines.append(
            f"| Average | {average_row['iou_fetp_attn']:.4f} | {average_row['iou_fetp_mmtok']:.4f} | {average_row['sink_ratio']:.4f} | "
            f"{average_row['fetp_non_spatial']:.2f} | {average_row['attn_non_spatial']:.2f} | {average_row['mmtok_non_spatial']:.2f} | - |"
        )

    report_lines.extend(["", "## Representative Sample", ""])
    if representative is None:
        report_lines.append("No eligible image-space sample was found.")
    else:
        report_lines.extend(
            [
                f"### {representative['task_name']} / doc {representative['doc_id']}",
                "",
                f"- Question: {representative['question_text']}",
                f"- FETP vs Attn IoU: {representative['iou_fetp_attn']:.4f}",
                f"- FETP vs MMTok IoU: {representative['iou_fetp_mmtok']:.4f}",
                f"- Attn-only ratio removed by FETP: {representative['sink_ratio']:.4f}",
                f"- Overview: ![]({representative['overview_plot']})",
                f"- Base/crop composition: ![]({representative['structure_plot']})",
                f"- Sink view: ![]({representative['sink_plot']})",
                "",
                "PCA supplement is generated separately under `../pca_compare/report.md`.",
                "",
            ]
        )

    (output_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summary = generate_visual_compare_report(
        artifact_root=args.artifact_root,
        output_dir=args.output_dir,
    )
    print(f"Wrote summary to {args.output_dir / 'summary.json'}")
    print(f"Wrote report to {args.output_dir / 'report.md'}")
    if summary["representative_sample"] is None:
        print("No eligible representative sample found.")


if __name__ == "__main__":
    main()
