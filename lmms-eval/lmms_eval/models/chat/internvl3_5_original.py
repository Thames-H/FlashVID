from typing import Optional, Union

from lmms_eval.api.registry import register_model
from lmms_eval.models.chat.internvl_hf import InternVLHf


@register_model("internvl3_5_original")
class InternVL3_5_Original(InternVLHf):
    is_simple = False

    def __init__(
        self,
        pretrained: str = "OpenGVLab/InternVL3_5-8B-HF",
        revision: str = "main",
        device: str = "cuda",
        device_map: str = "auto",
        batch_size: Union[int, str] = 1,
        min_patches: int = 1,
        max_patches: int = 12,
        num_frames: int = 32,
        fps: Optional[float] = None,
        trust_remote_code: Optional[bool] = False,
        low_cpu_mem_usage: Optional[bool] = False,
        attn_implementation: Optional[str] = None,
        use_cache: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            pretrained=pretrained,
            revision=revision,
            device=device,
            device_map=device_map,
            batch_size=batch_size,
            min_patches=min_patches,
            max_patches=max_patches,
            num_frames=num_frames,
            fps=fps,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=low_cpu_mem_usage,
            attn_implementation=attn_implementation,
            use_cache=use_cache,
            **kwargs,
        )
