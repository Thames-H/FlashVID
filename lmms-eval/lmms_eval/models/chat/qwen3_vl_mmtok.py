import sys
from pathlib import Path
from typing import Optional, Union

from loguru import logger as eval_logger

from lmms_eval.api.registry import register_model
from lmms_eval.models.chat.qwen3_vl import Qwen3_VL as Qwen3_VL_Chat


def _resolve_flashvid_repo_root() -> Path:
    repo_root = Path(__file__).resolve().parents[4]
    flashvid_pkg = repo_root / "flashvid"
    if not flashvid_pkg.exists():
        raise FileNotFoundError(
            f"FlashVID package not found at {flashvid_pkg}. "
            "Expected the workspace copy under flashvid/."
        )
    return repo_root


def _load_mmtok_qwen3_wrapper():
    repo_root = _resolve_flashvid_repo_root()
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    from flashvid.mmtok.qwen import mmtok_qwen3_vl

    return mmtok_qwen3_vl


_MMTOK_QWEN3_VL = _load_mmtok_qwen3_wrapper()


@register_model("qwen3_vl_mmtok")
class Qwen3_VL_MMTok(Qwen3_VL_Chat):
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
        retain_ratio: float = 0.2,
        enable_flashvid: bool = False,
        **kwargs,
    ) -> None:
        if int(batch_size) != 1:
            raise AssertionError("Qwen3-VL MMTok currently supports batch_size=1 only.")
        if enable_flashvid:
            raise ValueError("qwen3_vl_mmtok does not support enable_flashvid=True.")

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

        flashvid_repo_root = _resolve_flashvid_repo_root()
        eval_logger.info(
            f"[Qwen3-VL-MMTok] Using bundled FlashVID MMTok from {flashvid_repo_root / 'flashvid' / 'mmtok'}"
        )

        if hasattr(self, "accelerator") and self.accelerator.num_processes > 1:
            _, self.processor = _MMTOK_QWEN3_VL(
                self.model,
                language_tokenizer=self._tokenizer,
                processor=self.processor,
                retain_ratio=retain_ratio,
            )
        else:
            self._model, self.processor = _MMTOK_QWEN3_VL(
                self._model,
                language_tokenizer=self._tokenizer,
                processor=self.processor,
                retain_ratio=retain_ratio,
            )

        self.retain_ratio = retain_ratio
