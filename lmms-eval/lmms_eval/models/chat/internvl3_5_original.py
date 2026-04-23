from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.internvl3_5 import InternVL3_5


@register_model("internvl3_5_original")
class InternVL3_5_Original(InternVL3_5):
    is_simple = True

    def __init__(
        self,
        pretrained: str = "OpenGVLab/InternVL3_5-8B",
        modality: str = "video",
        device: str = "cuda:0",
        device_map: str = "auto",
        batch_size: str = "1",
        num_frame: int = 8,
        max_num: int = 1,
        use_flash_attn: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            pretrained=pretrained,
            modality=modality,
            device=device,
            device_map=device_map,
            batch_size=batch_size,
            num_frame=num_frame,
            max_num=max_num,
            use_flash_attn=use_flash_attn,
            **kwargs,
        )
