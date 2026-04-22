from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
from PIL import Image


def _to_uint8_hwc_tensor(image_input: Any) -> torch.Tensor:
    if isinstance(image_input, Image.Image):
        rgb = image_input.convert("RGB")
        return torch.from_numpy(np.array(rgb, dtype=np.uint8, copy=True))

    if isinstance(image_input, torch.Tensor):
        image = image_input.detach().cpu()
        if image.ndim == 3 and image.shape[0] == 3:
            image = image.permute(1, 2, 0)
        if image.dtype != torch.uint8:
            image = image.clamp(0, 255).to(torch.uint8)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(
                f"Unsupported tensor image shape {tuple(image.shape)}"
            )
        return image.contiguous()

    raise TypeError(f"Unsupported image input type: {type(image_input)!r}")


def build_visual_compare_metadata(
    image_inputs: Optional[list],
    video_inputs: Optional[list],
    image_grid_thw: Optional[torch.Tensor],
    n_visual_tokens_scored: int,
    spatial_merge_size: int,
) -> dict:
    metadata = {
        "visual_compare_eligible": False,
        "visual_compare_skip_reason": None,
    }

    if video_inputs:
        metadata["visual_compare_skip_reason"] = "video_input"
        return metadata

    if not image_inputs:
        metadata["visual_compare_skip_reason"] = "missing_image_input"
        return metadata

    if len(image_inputs) != 1:
        metadata["visual_compare_skip_reason"] = "multi_image_input"
        return metadata

    if image_grid_thw is None or image_grid_thw.numel() != 3:
        metadata["visual_compare_skip_reason"] = "missing_image_grid"
        return metadata

    grid_tensor = image_grid_thw.detach().cpu().view(-1).long()
    temporal, grid_h_raw, grid_w_raw = (int(x) for x in grid_tensor.tolist())
    if temporal != 1:
        metadata["visual_compare_skip_reason"] = "non_single_frame_grid"
        return metadata

    if spatial_merge_size <= 0:
        metadata["visual_compare_skip_reason"] = "invalid_merge_size"
        return metadata

    if (grid_h_raw % spatial_merge_size) != 0 or (
        grid_w_raw % spatial_merge_size
    ) != 0:
        metadata["visual_compare_skip_reason"] = "non_divisible_grid"
        return metadata

    token_grid_h = grid_h_raw // spatial_merge_size
    token_grid_w = grid_w_raw // spatial_merge_size

    image_preview = _to_uint8_hwc_tensor(image_inputs[0])
    metadata["image_preview"] = image_preview
    metadata["image_size"] = [
        int(image_preview.shape[0]),
        int(image_preview.shape[1]),
    ]
    metadata["token_grid_size"] = [int(token_grid_h), int(token_grid_w)]

    if int(n_visual_tokens_scored) != int(token_grid_h * token_grid_w):
        metadata["visual_compare_skip_reason"] = "grid_token_mismatch"
        return metadata

    metadata["visual_compare_eligible"] = True
    return metadata


def attach_visual_compare_metadata(
    artifact: dict,
    visual_metadata: dict,
    target_layer: Optional[int] = None,
) -> dict:
    updated = dict(artifact)
    updated["metadata"] = dict(updated.get("metadata", {}))
    if target_layer is not None:
        updated["metadata"]["target_layer"] = int(target_layer)

    updated["visual_compare_eligible"] = bool(
        visual_metadata.get("visual_compare_eligible", False)
    )
    updated["visual_compare_skip_reason"] = visual_metadata.get(
        "visual_compare_skip_reason"
    )

    for key in ("image_preview", "image_size", "token_grid_size"):
        if key in visual_metadata:
            updated[key] = visual_metadata[key]

    return updated
