from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from transformers.models.llava_onevision.modeling_llava_onevision import (
    get_anyres_image_grid_shape,
    unpad_image,
)


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


def _grid_boxes(
    image_height: int,
    image_width: int,
    grid_height: int,
    grid_width: int,
) -> torch.Tensor:
    y_edges = torch.linspace(0, image_height, steps=grid_height + 1)
    x_edges = torch.linspace(0, image_width, steps=grid_width + 1)
    boxes = []
    for row in range(grid_height):
        for col in range(grid_width):
            y0 = float(y_edges[row].item())
            x0 = float(x_edges[col].item())
            y1 = float(y_edges[row + 1].item())
            x1 = float(x_edges[col + 1].item())
            boxes.append([x0, y0, x1, y1])
    return torch.tensor(boxes, dtype=torch.float32)


def _resolve_crop_grid_size(
    image_height: int,
    image_width: int,
    model_config,
    vision_aspect_ratio: str,
) -> tuple[int, int]:
    base_grid_side = (
        int(model_config.vision_config.image_size)
        // int(model_config.vision_config.patch_size)
    )
    patch_rows, patch_cols = get_anyres_image_grid_shape(
        (image_height, image_width),
        model_config.image_grid_pinpoints,
        int(model_config.vision_config.image_size),
    )
    dummy = torch.zeros(
        1,
        int(patch_rows) * base_grid_side,
        int(patch_cols) * base_grid_side,
        dtype=torch.float32,
    )
    unpadded = unpad_image(dummy, (image_height, image_width))
    crop_grid_height = int(unpadded.shape[1])
    crop_grid_width = int(unpadded.shape[2])

    if vision_aspect_ratio.startswith("anyres_max_"):
        try:
            max_num_patches = int(vision_aspect_ratio.split("_")[-1])
        except ValueError:
            max_num_patches = None
        if max_num_patches and crop_grid_height > 0 and crop_grid_width > 0:
            ratio = math.sqrt(
                (crop_grid_height * crop_grid_width)
                / (max_num_patches * (base_grid_side**2))
            )
            if ratio > 1.1:
                crop_grid_height = max(1, int(crop_grid_height // ratio))
                crop_grid_width = max(1, int(crop_grid_width // ratio))

    return crop_grid_height, crop_grid_width


def _build_onevision_token_mapping(
    image_height: int,
    image_width: int,
    model_config,
    vision_aspect_ratio: str,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    base_grid_side = (
        int(model_config.vision_config.image_size)
        // int(model_config.vision_config.patch_size)
    )
    crop_grid_height, crop_grid_width = _resolve_crop_grid_size(
        image_height=image_height,
        image_width=image_width,
        model_config=model_config,
        vision_aspect_ratio=vision_aspect_ratio,
    )

    base_boxes = _grid_boxes(
        image_height=image_height,
        image_width=image_width,
        grid_height=base_grid_side,
        grid_width=base_grid_side,
    )
    crop_boxes = _grid_boxes(
        image_height=image_height,
        image_width=image_width,
        grid_height=crop_grid_height,
        grid_width=crop_grid_width,
    )

    boxes = []
    is_spatial = []
    sources: list[str] = []

    for box in base_boxes:
        boxes.append(box.tolist())
        is_spatial.append(True)
        sources.append("base")

    crop_index = 0
    for _row in range(crop_grid_height):
        for _col in range(crop_grid_width):
            boxes.append(crop_boxes[crop_index].tolist())
            is_spatial.append(True)
            sources.append("crop")
            crop_index += 1
        boxes.append([0.0, 0.0, 0.0, 0.0])
        is_spatial.append(False)
        sources.append("newline")

    return (
        torch.tensor(boxes, dtype=torch.float32),
        torch.tensor(is_spatial, dtype=torch.bool),
        sources,
    )


def _apply_stage1_keep(
    token_boxes: torch.Tensor,
    token_is_spatial: torch.Tensor,
    token_source: list[str],
    stage1_keep_local: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    if stage1_keep_local is None:
        return token_boxes, token_is_spatial, token_source

    keep = stage1_keep_local.detach().cpu().long()
    token_boxes = token_boxes.index_select(0, keep)
    token_is_spatial = token_is_spatial.index_select(0, keep)
    token_source = [token_source[int(idx)] for idx in keep.tolist()]
    return token_boxes, token_is_spatial, token_source


def build_visual_compare_metadata(
    image_inputs: Optional[list],
    video_inputs: Optional[list],
    model_config,
    n_visual_tokens_scored: int,
    vision_aspect_ratio: str,
    stage1_keep_local: Optional[torch.Tensor] = None,
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

    image_preview = _to_uint8_hwc_tensor(image_inputs[0])
    image_height = int(image_preview.shape[0])
    image_width = int(image_preview.shape[1])

    token_boxes, token_is_spatial, token_source = _build_onevision_token_mapping(
        image_height=image_height,
        image_width=image_width,
        model_config=model_config,
        vision_aspect_ratio=vision_aspect_ratio,
    )
    token_boxes, token_is_spatial, token_source = _apply_stage1_keep(
        token_boxes=token_boxes,
        token_is_spatial=token_is_spatial,
        token_source=token_source,
        stage1_keep_local=stage1_keep_local,
    )

    metadata["image_preview"] = image_preview
    metadata["image_size"] = [image_height, image_width]
    metadata["token_boxes"] = token_boxes
    metadata["token_is_spatial"] = token_is_spatial
    metadata["token_source"] = token_source

    if int(token_boxes.shape[0]) != int(n_visual_tokens_scored):
        metadata["visual_compare_skip_reason"] = "token_mapping_mismatch"
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

    for key in (
        "image_preview",
        "image_size",
        "token_boxes",
        "token_is_spatial",
        "token_source",
    ):
        if key in visual_metadata:
            updated[key] = visual_metadata[key]

    return updated
