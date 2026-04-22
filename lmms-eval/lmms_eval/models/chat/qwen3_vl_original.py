from typing import Optional, Union

from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.qwen3_vl import Qwen3_VL as Qwen3VLSimple


_FLASHVID_ONLY_ARGS = {
    "retention_ratio",
    "do_segment",
    "segment_threshold",
    "min_segment_num",
    "complementary_segment",
    "token_selection_method",
    "alpha",
    "temporal_threshold",
    "expansion",
    "pruning_layer",
    "llm_retention_ratio",
}


@register_model("qwen3_vl_original")
class Qwen3_VL_Original(Qwen3VLSimple):
    is_simple = True

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        attn_implementation: Optional[str] = None,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,
        max_image_size: Optional[int] = None,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        enable_flashvid = kwargs.pop("enable_flashvid", False)
        flashvid_kwargs = {
            key: kwargs.pop(key)
            for key in list(kwargs)
            if key in _FLASHVID_ONLY_ARGS
        }
        if enable_flashvid or flashvid_kwargs:
            raise ValueError(
                "qwen3_vl_original disables FlashVID and pruning kwargs. "
                "Use qwen3_vl or qwen3_vl_ours_* for optimized variants."
            )

        super().__init__(
            pretrained=pretrained,
            device=device,
            device_map=device_map,
            batch_size=batch_size,
            use_cache=use_cache,
            attn_implementation=attn_implementation,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            max_num_frames=max_num_frames,
            use_custom_video_loader=use_custom_video_loader,
            fps=fps,
            max_image_size=max_image_size,
            system_prompt=system_prompt,
            interleave_visuals=interleave_visuals,
            reasoning_prompt=reasoning_prompt,
            enable_flashvid=False,
            **kwargs,
        )
