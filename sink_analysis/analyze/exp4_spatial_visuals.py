from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch

from sink_analysis.collect.sink_metrics import identify_sink_tokens


def render_patch_overlay(
    image: np.ndarray,
    patch_mapping: dict,
    indices: Iterable[int],
    color: tuple[int, int, int],
) -> np.ndarray:
    overlay = image.copy()
    coords = patch_mapping.get("token_coords", [])
    patch_h, patch_w = patch_mapping.get("patch_pixel_size") or (16, 16)
    for index in indices:
        if index >= len(coords):
            continue
        coord = coords[index]
        if coord is None:
            continue
        y, x = coord
        y0 = max(0, y - patch_h // 2)
        y1 = min(overlay.shape[0], y + patch_h // 2)
        x0 = max(0, x - patch_w // 2)
        x1 = min(overlay.shape[1], x + patch_w // 2)
        overlay[y0:y1, x0:x1] = (
            0.55 * overlay[y0:y1, x0:x1] + 0.45 * np.asarray(color, dtype=np.uint8)
        ).astype(np.uint8)
    return overlay


def select_representative_samples(artifacts: list[dict], keep_ratio: str = "50%", n: int = 3) -> list[dict]:
    scored = []
    for artifact in artifacts:
        fetp = set(artifact["selections"][keep_ratio]["fetp"]["indices"].tolist())
        attention = set(artifact["selections"][keep_ratio]["attention"]["indices"].tolist())
        union = fetp | attention
        iou = len(fetp & attention) / max(1, len(union))
        scored.append((iou, artifact))
    scored.sort(key=lambda item: item[0])
    return [artifact for _, artifact in scored[:n]]


def render_full_comparison(artifact: dict, keep_ratio: str = "50%"):
    image = artifact["image_preview"]
    patch_mapping = artifact["patch_mapping"]
    selections = artifact["selections"][keep_ratio]
    sink_mask, _, _ = identify_sink_tokens(
        artifact["alpha"],
        artifact["values"],
        artifact["query_outputs"],
    )

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original")

    methods = [
        ("attention", (230, 76, 60), "Attention-only"),
        ("mmtok", (243, 156, 18), "MMTok"),
        ("fetp", (46, 204, 113), "FETP"),
    ]
    for col, (method, color, title) in enumerate(methods, start=1):
        overlay = render_patch_overlay(
            image,
            patch_mapping,
            selections[method]["indices"].tolist(),
            color,
        )
        axes[0, col].imshow(overlay)
        axes[0, col].set_title(title)

    sink_indices = torch.where(sink_mask)[0].tolist()
    axes[1, 0].imshow(render_patch_overlay(image, patch_mapping, sink_indices, (255, 0, 0)))
    axes[1, 0].set_title("Sink Tokens")

    attention_idx = selections["attention"]["indices"].tolist()
    attention_sink = [idx for idx in attention_idx if sink_mask[idx]]
    attention_nonsink = [idx for idx in attention_idx if not sink_mask[idx]]
    overlay = render_patch_overlay(image, patch_mapping, attention_nonsink, (52, 152, 219))
    overlay = render_patch_overlay(overlay, patch_mapping, attention_sink, (255, 0, 0))
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title("Attention sink split")

    fetp_idx = selections["fetp"]["indices"].tolist()
    fetp_sink = [idx for idx in fetp_idx if sink_mask[idx]]
    fetp_nonsink = [idx for idx in fetp_idx if not sink_mask[idx]]
    overlay = render_patch_overlay(image, patch_mapping, fetp_nonsink, (46, 204, 113))
    overlay = render_patch_overlay(overlay, patch_mapping, fetp_sink, (255, 0, 0))
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title("FETP sink split")

    fetp_unique = [idx for idx in fetp_idx if idx not in set(attention_idx)]
    axes[1, 3].imshow(render_patch_overlay(image, patch_mapping, fetp_unique, (0, 255, 255)))
    axes[1, 3].set_title("FETP unique")

    for axis in axes.flat:
        axis.axis("off")
    fig.tight_layout()
    return fig

