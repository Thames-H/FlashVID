from __future__ import annotations

from typing import Optional, Tuple


class VisualTokenLocator:
    @staticmethod
    def locate(model_name: str, input_ids, **kwargs):
        locators = {
            "qwen2.5-vl": QwenTokenLocator,
            "qwen3-vl": QwenTokenLocator,
            "llava-onevision": OneVisionTokenLocator,
            "internvl3.5": InternVLTokenLocator,
            "internvl3_5": InternVLTokenLocator,
            "internvl": InternVLTokenLocator,
        }
        if model_name not in locators:
            raise KeyError(f"Unsupported model_name: {model_name}")
        return locators[model_name].locate(input_ids, **kwargs)


class QwenTokenLocator:
    @staticmethod
    def locate(input_ids, image_token_id: int = 151655, **kwargs) -> Tuple[int, int, int, int]:
        ids = input_ids.squeeze(0).detach().cpu().tolist()
        visual_positions = [i for i, token_id in enumerate(ids) if token_id == image_token_id]
        if not visual_positions:
            raise ValueError("No visual tokens found for qwen-style locator.")

        visual_start = visual_positions[0]
        visual_end = visual_positions[-1] + 1
        query_start = visual_end
        query_end = len(ids)
        return visual_start, visual_end, query_start, query_end


class OneVisionTokenLocator:
    @staticmethod
    def locate(
        input_ids,
        image_token_id: Optional[int] = None,
        newline_token_id: Optional[int] = None,
        has_newline: bool = True,
        **kwargs,
    ) -> Tuple[int, int, int, int]:
        ids = input_ids.squeeze(0).detach().cpu().tolist()
        vis_start = kwargs.get("vis_start")
        vis_end = kwargs.get("vis_end")
        query_start = kwargs.get("query_start")
        query_end = kwargs.get("query_end")

        if vis_start is None or vis_end is None:
            if image_token_id is None:
                image_token_id = kwargs.get("vision_token_id")

            if image_token_id is not None:
                image_positions = [i for i, token_id in enumerate(ids) if token_id == image_token_id]
                if image_positions:
                    vis_start = image_positions[0]
                    vis_end = image_positions[-1] + 1

            if vis_start is None or vis_end is None:
                # fallback with newline separators
                if newline_token_id is None:
                    raise ValueError(
                        "OneVisionTokenLocator needs vis bounds, image_token_id, or newline_token_id."
                    )
                newline_positions = [i for i, token_id in enumerate(ids) if token_id == newline_token_id]
                if not newline_positions:
                    vis_start, vis_end = 0, len(ids)
                elif len(newline_positions) == 1:
                    vis_start, vis_end = 0, newline_positions[0]
                else:
                    vis_start = newline_positions[0] + 1
                    vis_end = newline_positions[1]

        if query_start is None:
            query_start = vis_end if vis_end is not None else len(ids)
        if query_end is None:
            query_end = len(ids)

        return vis_start, vis_end, query_start, query_end


class InternVLTokenLocator:
    @staticmethod
    def locate(
        input_ids,
        image_token_id: Optional[int] = None,
        img_start_token_id: Optional[int] = None,
        img_end_token_id: Optional[int] = None,
        **kwargs,
    ) -> Tuple[int, int, int, int]:
        ids = input_ids.squeeze(0).detach().cpu().tolist()

        if image_token_id is None:
            image_token_id = kwargs.get("img_context_token_id")
        if image_token_id is None:
            image_token_id = kwargs.get("vision_token_id")

        if image_token_id is not None:
            image_positions = [i for i, token_id in enumerate(ids) if token_id == image_token_id]
            if image_positions:
                visual_start = image_positions[0]
                visual_end = image_positions[-1] + 1
                query_start = visual_end
                query_end = len(ids)
                return visual_start, visual_end, query_start, query_end

        if img_start_token_id is None:
            img_start_token_id = kwargs.get("vision_start_token_id")
        if img_end_token_id is None:
            img_end_token_id = kwargs.get("vision_end_token_id")

        if img_start_token_id is None or img_end_token_id is None:
            raise ValueError("InternVL locator needs img_start_token_id and img_end_token_id.")

        try:
            start_pos = ids.index(img_start_token_id)
            end_pos = ids.index(img_end_token_id)
        except ValueError as exc:
            raise ValueError("No <img> ... </img> marker pair found.") from exc

        if start_pos >= end_pos:
            raise ValueError("Invalid InternVL visual token boundaries.")

        visual_start = start_pos + 1
        visual_end = end_pos
        query_start = end_pos + 1
        query_end = len(ids)
        return visual_start, visual_end, query_start, query_end
