import sys
from pathlib import Path
from typing import Optional, Union

import torch
from loguru import logger as eval_logger

from lmms_eval.api.registry import register_model
from lmms_eval.models.chat.llava_hf import LlavaHf as LlavaHfChat


def _resolve_flashvid_repo_root() -> Path:
    repo_root = Path(__file__).resolve().parents[4]
    flashvid_pkg = repo_root / "flashvid"
    if not flashvid_pkg.exists():
        raise FileNotFoundError(
            f"FlashVID package not found at {flashvid_pkg}. "
            "Expected the workspace copy under flashvid/."
        )
    return repo_root


def _load_mmtok_llava_onevision_wrapper():
    repo_root = _resolve_flashvid_repo_root()
    bundled_pkg_root = str(repo_root / "flashvid")
    if bundled_pkg_root not in sys.path:
        sys.path.insert(0, bundled_pkg_root)
    from mmtok.llava_onevision import mmtok_llava_onevision

    return mmtok_llava_onevision


_MMTOK_LLAVA_ONEVISION = _load_mmtok_llava_onevision_wrapper()


@register_model("llava_onevision_mmtok")
class LlavaOnevisionMMTok(LlavaHfChat):
    is_simple = False

    def __init__(
        self,
        pretrained: str = "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        revision: str = "main",
        device: str = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "float16",
        batch_size: Union[int, str] = 1,
        trust_remote_code: Optional[bool] = False,
        attn_implementation: Optional[str] = None,
        device_map: str = "",
        chat_template: Optional[str] = None,
        use_cache: bool = True,
        max_frames_num: Optional[int] = 32,
        retain_ratio: float = 0.2,
        target_vision_tokens: Optional[int] = None,
        **kwargs,
    ) -> None:
        if int(batch_size) != 1:
            raise AssertionError("LLaVA-OneVision MMTok currently supports batch_size=1 only.")

        super().__init__(
            pretrained=pretrained,
            revision=revision,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
            device_map=device_map,
            chat_template=chat_template,
            use_cache=use_cache,
            max_frames_num=max_frames_num,
            **kwargs,
        )

        flashvid_repo_root = _resolve_flashvid_repo_root()
        eval_logger.info(
            f"[LLaVA-OneVision-MMTok] Using bundled FlashVID MMTok from {flashvid_repo_root / 'flashvid' / 'mmtok'}"
        )

        if hasattr(self, "accelerator") and self.accelerator.num_processes > 1:
            _, self._image_processor = _MMTOK_LLAVA_ONEVISION(
                self.model,
                language_tokenizer=self._tokenizer,
                processor=self._image_processor,
                retain_ratio=retain_ratio,
                target_vision_tokens=target_vision_tokens,
            )
        else:
            self._model, self._image_processor = _MMTOK_LLAVA_ONEVISION(
                self._model,
                language_tokenizer=self._tokenizer,
                processor=self._image_processor,
                retain_ratio=retain_ratio,
                target_vision_tokens=target_vision_tokens,
            )

        self.retain_ratio = retain_ratio
        self.target_vision_tokens = target_vision_tokens
