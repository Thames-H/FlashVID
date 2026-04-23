from __future__ import annotations

import math
from typing import Tuple

from .schema import PatchMapping


class PatchMappingBuilder:
    @staticmethod
    def build(model_name: str, **kwargs) -> PatchMapping:
        builders = {
            "qwen2.5-vl": QwenPatchMappingBuilder,
            "qwen3-vl": QwenPatchMappingBuilder,
            "llava-onevision": OneVisionPatchMappingBuilder,
            "internvl3.5": InternVLPatchMappingBuilder,
            "internvl3_5": InternVLPatchMappingBuilder,
            "internvl": InternVLPatchMappingBuilder,
        }
        if model_name not in builders:
            raise KeyError(f"Unsupported model_name: {model_name}")
        return builders[model_name].build(**kwargs)


class QwenPatchMappingBuilder:
    @staticmethod
    def build(image_size, grid_thw, **kwargs) -> PatchMapping:
        _, grid_h, grid_w = tuple(grid_thw)
        H, W = image_size
        coords = []
        sources = []

        if grid_h <= 0 or grid_w <= 0:
            return PatchMapping(coords, sources, (0, 0), 0)

        for r in range(grid_h):
            for c in range(grid_w):
                y = int((r + 0.5) / grid_h * H)
                x = int((c + 0.5) / grid_w * W)
                coords.append((y, x))
                sources.append("spatial")

        patch_pixel_size = (H // grid_h, W // grid_w)
        return PatchMapping(
            token_coords=coords,
            token_source=sources,
            patch_pixel_size=patch_pixel_size,
            spatial_token_count=grid_h * grid_w,
        )


class OneVisionPatchMappingBuilder:
    @staticmethod
    def build(
        image_size,
        base_grid: Tuple[int, int] | None = None,
        crop_grid: Tuple[int, int] | None = None,
        crop_layout: Tuple[int, int] | None = None,
        has_newline: bool = True,
        **kwargs,
    ) -> PatchMapping:
        H, W = image_size
        coords = []
        sources = []

        if base_grid is None or crop_grid is None or crop_layout is None:
            approx_num_tokens = int(kwargs.get("num_visual_tokens", 0))
            if approx_num_tokens <= 0:
                return PatchMapping(coords, sources, (0, 0), 0)

            side = max(1, int(round(math.sqrt(approx_num_tokens))))
            for idx in range(approx_num_tokens):
                r = idx // side
                c = idx % side
                y = int((r + 0.5) / max(side, 1) * H)
                x = int((c + 0.5) / max(side, 1) * W)
                coords.append((y, x))
                sources.append("base")
            return PatchMapping(
                token_coords=coords,
                token_source=sources,
                patch_pixel_size=(max(H // max(side, 1), 1), max(W // max(side, 1), 1)),
                spatial_token_count=len(coords),
            )

        bh, bw = base_grid
        if bh > 0 and bw > 0:
            for r in range(bh):
                for c in range(bw):
                    y = int((r + 0.5) / bh * H)
                    x = int((c + 0.5) / bw * W)
                    coords.append((y, x))
                    sources.append("base")

            if has_newline:
                coords.append(None)
                sources.append("newline")

        crop_rows, crop_cols = crop_layout
        ch, cw = crop_grid
        if crop_rows > 0 and crop_cols > 0 and ch > 0 and cw > 0:
            for cr in range(crop_rows):
                for cc in range(crop_cols):
                    y0 = int(cr / crop_rows * H)
                    x0 = int(cc / crop_cols * W)
                    y1 = int((cr + 1) / crop_rows * H)
                    x1 = int((cc + 1) / crop_cols * W)
                    for r in range(ch):
                        for c in range(cw):
                            y = y0 + int((r + 0.5) / ch * max(y1 - y0, 1))
                            x = x0 + int((c + 0.5) / cw * max(x1 - x0, 1))
                            coords.append((y, x))
                            sources.append("crop")

                    if has_newline:
                        coords.append(None)
                        sources.append("newline")

        # spatial coverage proxy from original image split for downstream metrics
        patch_pixel_size = (max(H // max(crop_rows * ch, 1), 1), max(W // max(crop_cols * cw, 1), 1))
        spatial_count = sum(1 for c in coords if c is not None)
        return PatchMapping(
            token_coords=coords,
            token_source=sources,
            patch_pixel_size=patch_pixel_size,
            spatial_token_count=spatial_count,
        )


class InternVLPatchMappingBuilder:
    @staticmethod
    def build(
        image_size,
        num_patches_list,
        tile_size: int = 448,
        token_per_tile_side: int = 16,
        has_thumbnail: bool = True,
        **kwargs,
    ) -> PatchMapping:
        H, W = image_size
        coords = []
        sources = []

        if H <= 0 or W <= 0:
            return PatchMapping(coords, sources, (0, 0), 0)

        if isinstance(num_patches_list, int):
            num_patches_list = [int(num_patches_list)]
        if num_patches_list is None:
            num_patches_list = []

        n_tiles = max(len(num_patches_list) - (1 if has_thumbnail else 0), 0)
        if n_tiles == 0:
            total = int(sum(int(v) for v in num_patches_list)) if num_patches_list else 0
            if total <= 0:
                return PatchMapping(coords, sources, (0, 0), 0)
            side = max(1, int(round(math.sqrt(total))))
            for idx in range(total):
                r = idx // side
                c = idx % side
                y = int((r + 0.5) / max(side, 1) * H)
                x = int((c + 0.5) / max(side, 1) * W)
                coords.append((y, x))
                sources.append("tile")
            return PatchMapping(coords, sources, (max(H // max(side, 1), 1), max(W // max(side, 1), 1)), total)

        tile_layout = InternVLPatchMappingBuilder._infer_tile_layout(
            image_size=image_size,
            n_tiles=n_tiles,
            tile_size=tile_size,
        )
        tile_rows, tile_cols = tile_layout

        # thumbnail
        if has_thumbnail and num_patches_list:
            n_thumb_tokens = int(num_patches_list[0] or 0)
            if n_thumb_tokens > 0:
                th = int(math.isqrt(n_thumb_tokens))
                if th * th != n_thumb_tokens:
                    th = max(int(round(math.sqrt(n_thumb_tokens))), 1)
                th = max(th, 1)
                for r in range(th):
                    for c in range(th):
                        y = int((r + 0.5) / th * H)
                        x = int((c + 0.5) / th * W)
                        coords.append((y, x))
                        sources.append("thumbnail")

        # tiles
        tile_start = 1 if has_thumbnail else 0
        tile_idx = 0
        for tr in range(tile_rows):
            for tc in range(tile_cols):
                if tile_start + tile_idx >= len(num_patches_list):
                    break
                n_tokens = int(num_patches_list[tile_start + tile_idx] or 0)
                side = int(math.isqrt(n_tokens))
                if side * side != n_tokens:
                    side = max(int(round(math.sqrt(n_tokens))), 1)

                y0 = int(tr / max(tile_rows, 1) * H)
                x0 = int(tc / max(tile_cols, 1) * W)
                y1 = int((tr + 1) / max(tile_rows, 1) * H)
                x1 = int((tc + 1) / max(tile_cols, 1) * W)
                if side > 0:
                    h_scale = max(y1 - y0, 1)
                    w_scale = max(x1 - x0, 1)
                    for r in range(side):
                        for c in range(side):
                            y = y0 + int((r + 0.5) / side * h_scale)
                            x = x0 + int((c + 0.5) / side * w_scale)
                            coords.append((y, x))
                            sources.append("tile")
                tile_idx += 1

        spatial_count = sum(1 for c in coords if c is not None)
        patch_pixel_size = (H // max(tile_rows * token_per_tile_side, 1), W // max(tile_cols * token_per_tile_side, 1))
        return PatchMapping(
            token_coords=coords,
            token_source=sources,
            patch_pixel_size=patch_pixel_size,
            spatial_token_count=spatial_count,
        )

    @staticmethod
    def _infer_tile_layout(image_size, n_tiles, tile_size):
        H, W = image_size
        if n_tiles <= 0:
            return (1, 1)
        aspect = W / H
        best_layout = (1, n_tiles)
        best_diff = float("inf")
        for rows in range(1, n_tiles + 1):
            if n_tiles % rows != 0:
                continue
            cols = n_tiles // rows
            layout_aspect = cols / rows
            diff = abs(layout_aspect - aspect)
            if diff < best_diff:
                best_diff = diff
                best_layout = (rows, cols)
        return best_layout
