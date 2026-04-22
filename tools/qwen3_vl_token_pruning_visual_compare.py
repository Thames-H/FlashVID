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


def _minmax(scores: torch.Tensor) -> torch.Tensor:
    scores = scores.float()
    if scores.numel() == 0:
        return scores
    min_value = scores.min()
    max_value = scores.max()
    denom = (max_value - min_value).clamp_min(1e-8)
    return (scores - min_value) / denom


def _build_patch_mask(
    num_tokens: int,
    grid_size: Tuple[int, int],
    keep_indices: torch.Tensor,
) -> torch.Tensor:
    grid_h, grid_w = int(grid_size[0]), int(grid_size[1])
    if grid_h * grid_w != int(num_tokens):
        raise ValueError(
            f"grid_size {grid_size} does not match num_tokens {num_tokens}"
        )
    mask = torch.zeros(num_tokens, dtype=torch.float32)
    if keep_indices.numel() > 0:
        mask[keep_indices.long()] = 1.0
    return mask.view(grid_h, grid_w)


def _tensor_image_to_float(image_preview: torch.Tensor) -> torch.Tensor:
    image = image_preview.detach().cpu()
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(
            f"image_preview must have shape (H, W, 3), got {tuple(image.shape)}"
        )
    return image.float() / 255.0


def _upsample_mask(mask: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
    target_h, target_w = int(image_size[0]), int(image_size[1])
    resized = torch.nn.functional.interpolate(
        mask.view(1, 1, *mask.shape).float(),
        size=(target_h, target_w),
        mode="nearest",
    )
    return resized[0, 0]


def _overlay_binary_mask(
    image_preview: torch.Tensor,
    patch_mask: torch.Tensor,
    color: Tuple[float, float, float],
    alpha: float = 0.45,
    dim_factor: float = 0.55,
) -> torch.Tensor:
    base = _tensor_image_to_float(image_preview)
    overlay = base * dim_factor
    mask = _upsample_mask(patch_mask, base.shape[:2]).unsqueeze(-1)
    color_tensor = torch.tensor(color, dtype=torch.float32).view(1, 1, 3)
    return torch.clamp(overlay * (1.0 - mask * alpha) + color_tensor * (mask * alpha), 0.0, 1.0)


def _prepare_sample_metrics(
    fetp_artifact: dict,
    mmtok_artifact: dict,
) -> dict:
    attn_keep = fetp_artifact["selection"]["attention_only_keep_local"].long()
    fetp_keep = fetp_artifact["selection"]["fetp_keep_local"].long()
    mmtok_keep = mmtok_artifact["selection"]["mmtok_keep_local"].long()

    image_preview = fetp_artifact.get("image_preview")
    grid_size = tuple(fetp_artifact.get("token_grid_size", []))
    num_tokens = int(fetp_artifact["metadata"]["n_visual_tokens_scored"])

    eligible = True
    skip_reason = None
    if "visual_compare_eligible" in fetp_artifact:
        eligible = bool(fetp_artifact.get("visual_compare_eligible"))
        skip_reason = fetp_artifact.get("visual_compare_skip_reason")

    if eligible and image_preview is None:
        eligible = False
        skip_reason = "missing_image_preview"
    elif eligible and len(grid_size) != 2:
        eligible = False
        skip_reason = "missing_token_grid_size"
    elif eligible and int(grid_size[0]) * int(grid_size[1]) != num_tokens:
        eligible = False
        skip_reason = "grid_token_mismatch"

    sink_indices = _compute_sink_indices(attn_keep, fetp_keep)
    return {
        "task_name": str(fetp_artifact["task_name"]),
        "doc_id": str(fetp_artifact["doc_id"]),
        "question_text": str(fetp_artifact.get("question_text", "")).strip().replace("\n", " "),
        "eligible": eligible,
        "skip_reason": skip_reason,
        "iou_fetp_attn": _selection_iou(fetp_keep, attn_keep),
        "iou_fetp_mmtok": _selection_iou(fetp_keep, mmtok_keep),
        "sink_ratio": float(sink_indices.numel() / max(1, attn_keep.numel())),
        "sink_count": int(sink_indices.numel()),
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
    grid_size = tuple(int(x) for x in fetp_artifact["token_grid_size"])
    num_tokens = int(fetp_artifact["metadata"]["n_visual_tokens_scored"])

    attn_mask = _build_patch_mask(
        num_tokens,
        grid_size,
        fetp_artifact["selection"]["attention_only_keep_local"].long(),
    )
    mmtok_mask = _build_patch_mask(
        num_tokens,
        grid_size,
        mmtok_artifact["selection"]["mmtok_keep_local"].long(),
    )
    fetp_mask = _build_patch_mask(
        num_tokens,
        grid_size,
        fetp_artifact["selection"]["fetp_keep_local"].long(),
    )

    panels = [
        ("Original", _tensor_image_to_float(image_preview)),
        ("Attention-only", _overlay_binary_mask(image_preview, attn_mask, (0.14, 0.37, 0.88))),
        ("MMTok", _overlay_binary_mask(image_preview, mmtok_mask, (0.12, 0.65, 0.30))),
        ("FETP", _overlay_binary_mask(image_preview, fetp_mask, (0.85, 0.22, 0.18))),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, (title, image) in zip(axes, panels):
        ax.imshow(image.numpy())
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_sink(
    fetp_artifact: dict,
    output_path: Path,
) -> None:
    image_preview = fetp_artifact["image_preview"]
    grid_size = tuple(int(x) for x in fetp_artifact["token_grid_size"])
    num_tokens = int(fetp_artifact["metadata"]["n_visual_tokens_scored"])
    attn_keep = fetp_artifact["selection"]["attention_only_keep_local"].long()
    fetp_keep = fetp_artifact["selection"]["fetp_keep_local"].long()
    sink_indices = _compute_sink_indices(attn_keep, fetp_keep)

    attn_mask = _build_patch_mask(num_tokens, grid_size, attn_keep)
    fetp_mask = _build_patch_mask(num_tokens, grid_size, fetp_keep)
    sink_mask = _build_patch_mask(num_tokens, grid_size, sink_indices)

    base_image = _tensor_image_to_float(image_preview)
    attention_overlay = _overlay_binary_mask(
        image_preview,
        attn_mask,
        (0.14, 0.37, 0.88),
    )
    sink_highlight = _overlay_binary_mask(
        image_preview,
        sink_mask,
        (1.0, 0.78, 0.12),
        alpha=0.65,
        dim_factor=0.65,
    )
    fetp_overlay = _overlay_binary_mask(
        image_preview,
        fetp_mask,
        (0.85, 0.22, 0.18),
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    panels = [
        ("Original", base_image),
        ("Attention + sink", torch.clamp((attention_overlay * 0.75) + (sink_highlight * 0.55), 0.0, 1.0)),
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
        sink_plot = plots_dir / f"{sample_slug}__sink.png"
        _plot_overview(fetp_artifact, mmtok_artifact, overview_plot)
        _plot_sink(fetp_artifact, sink_plot)
        representative["overview_plot"] = str(overview_plot.relative_to(output_dir))
        representative["sink_plot"] = str(sink_plot.relative_to(output_dir))

    average_row = None
    if rows:
        average_row = {
            "task_name": "Average",
            "doc_id": "-",
            "iou_fetp_attn": sum(row["iou_fetp_attn"] for row in rows) / len(rows),
            "iou_fetp_mmtok": sum(row["iou_fetp_mmtok"] for row in rows) / len(rows),
            "sink_ratio": sum(row["sink_ratio"] for row in rows) / len(rows),
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
        "# Qwen3-VL Visual Compare",
        "",
        f"- Artifact root: `{artifact_root}`",
        f"- Matched samples: `{len(rows)}`",
        "",
        "## Summary Table",
        "",
        "| Sample | FETP vs Attn IoU | FETP vs MMTok IoU | Attn-only ratio removed by FETP | Eligible |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        sample_name = f"{row['task_name']}/{row['doc_id']}"
        eligibility = "yes"
        if not row["eligible"]:
            eligibility = f"no ({row['skip_reason']})"
        report_lines.append(
            f"| {sample_name} | {row['iou_fetp_attn']:.4f} | {row['iou_fetp_mmtok']:.4f} | {row['sink_ratio']:.4f} | "
            f"{eligibility} |"
        )
    if average_row is not None:
        report_lines.append(
            f"| Average | {average_row['iou_fetp_attn']:.4f} | {average_row['iou_fetp_mmtok']:.4f} | {average_row['sink_ratio']:.4f} | - |"
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
                f"- Sink view: ![]({representative['sink_plot']})",
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
