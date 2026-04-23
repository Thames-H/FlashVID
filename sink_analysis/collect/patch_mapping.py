from __future__ import annotations

import math
from typing import Any

import torch
from transformers.models.llava_onevision.modeling_llava_onevision import (
    get_anyres_image_grid_shape,
)


def build_qwen3vl_mapping(
    image_size: tuple[int, int] | None,
    grid_thw: tuple[int, int, int] | list[int] | torch.Tensor | None,
) -> dict[str, Any]:
    if image_size is None or grid_thw is None:
        return {
            "token_coords": [],
            "token_source": [],
            "patch_pixel_size": None,
        }

    if isinstance(grid_thw, torch.Tensor):
        grid_thw = grid_thw.tolist()

    _, grid_h, grid_w = [int(value) for value in grid_thw]
    image_h, image_w = image_size
    coords: list[tuple[int, int]] = []
    sources: list[str] = []

    for row in range(grid_h):
        for col in range(grid_w):
            y = int((row + 0.5) / max(1, grid_h) * image_h)
            x = int((col + 0.5) / max(1, grid_w) * image_w)
            coords.append((y, x))
            sources.append("spatial")

    return {
        "token_coords": coords,
        "token_source": sources,
        "patch_pixel_size": (
            max(1, image_h // max(1, grid_h)),
            max(1, image_w // max(1, grid_w)),
        ),
    }


def _unpad_dimensions(
    current_height: int,
    current_width: int,
    original_height: int,
    original_width: int,
) -> tuple[int, int]:
    original_aspect_ratio = original_width / max(1, original_height)
    current_aspect_ratio = current_width / max(1, current_height)

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / max(1, original_width)
        new_height = int(round(original_height * scale_factor, 7))
        return max(1, new_height), current_width

    scale_factor = current_height / max(1, original_height)
    new_width = int(round(original_width * scale_factor, 7))
    return current_height, max(1, new_width)


def build_onevision_mapping(
    image_size: tuple[int, int] | None,
    image_grid_pinpoints: list | None,
    vision_image_size: int,
    vision_patch_size: int,
    vision_aspect_ratio: str = "anyres_max_9",
) -> dict[str, Any]:
    if image_size is None:
        return {
            "token_coords": [],
            "token_source": [],
            "patch_pixel_size": None,
        }

    image_h, image_w = [int(value) for value in image_size]
    base_side = max(1, vision_image_size // vision_patch_size)

    coords: list[tuple[int, int] | None] = []
    sources: list[str] = []

    for row in range(base_side):
        for col in range(base_side):
            y = int((row + 0.5) / base_side * image_h)
            x = int((col + 0.5) / base_side * image_w)
            coords.append((y, x))
            sources.append("base")

    if not image_grid_pinpoints:
        return {
            "token_coords": coords,
            "token_source": sources,
            "patch_pixel_size": (
                max(1, image_h // base_side),
                max(1, image_w // base_side),
            ),
        }

    patch_grid_h, patch_grid_w = get_anyres_image_grid_shape(
        (image_h, image_w),
        image_grid_pinpoints,
        vision_image_size,
    )
    crop_grid_h = patch_grid_h * base_side
    crop_grid_w = patch_grid_w * base_side
    crop_grid_h, crop_grid_w = _unpad_dimensions(
        crop_grid_h,
        crop_grid_w,
        image_h,
        image_w,
    )

    max_num_patches = int(str(vision_aspect_ratio).strip("anyres_max_") or "9")
    ratio = math.sqrt(
        (crop_grid_h * crop_grid_w) / max(1, max_num_patches * (base_side**2))
    )
    if ratio > 1.1:
        crop_grid_h = max(1, int(crop_grid_h // ratio))
        crop_grid_w = max(1, int(crop_grid_w // ratio))

    for row in range(crop_grid_h):
        for col in range(crop_grid_w):
            y = int((row + 0.5) / crop_grid_h * image_h)
            x = int((col + 0.5) / crop_grid_w * image_w)
            coords.append((y, x))
            sources.append("crop")
        coords.append(None)
        sources.append("newline")

    return {
        "token_coords": coords,
        "token_source": sources,
        "patch_pixel_size": (
            max(1, image_h // max(1, crop_grid_h)),
            max(1, image_w // max(1, crop_grid_w)),
        ),
    }

