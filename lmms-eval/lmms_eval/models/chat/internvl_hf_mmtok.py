import sys
from pathlib import Path
from typing import Optional, Union

from loguru import logger as eval_logger

from lmms_eval.api.registry import register_model
from lmms_eval.models.chat.internvl_hf import InternVLHf


def _resolve_flashvid_repo_root() -> Path:
    repo_root = Path(__file__).resolve().parents[4]
    flashvid_pkg = repo_root / "flashvid"
    if not flashvid_pkg.exists():
        raise FileNotFoundError(
            f"FlashVID package not found at {flashvid_pkg}. "
            "Expected the workspace copy under flashvid/."
        )
    return repo_root


def _load_mmtok_internvl_wrapper():
    repo_root = _resolve_flashvid_repo_root()
    bundled_pkg_root = str(repo_root / "flashvid")
    if bundled_pkg_root not in sys.path:
        sys.path.insert(0, bundled_pkg_root)
    from mmtok.internvl import mmtok_internvl

    return mmtok_internvl


_MMTOK_INTERNVL = _load_mmtok_internvl_wrapper()


@register_model("internvl_hf_mmtok")
class InternVLHfMMTok(InternVLHf):
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
        retain_ratio: float = 0.2,
        target_vision_tokens: Optional[int] = None,
        **kwargs,
    ) -> None:
        if int(batch_size) != 1:
            raise AssertionError("InternVL MMTok currently supports batch_size=1 only.")

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

        flashvid_repo_root = _resolve_flashvid_repo_root()
        eval_logger.info(
            f"[InternVL-MMTok] Using bundled FlashVID MMTok from {flashvid_repo_root / 'flashvid' / 'mmtok'}"
        )

        if hasattr(self, "accelerator") and self.accelerator.num_processes > 1:
            _, self.processor = _MMTOK_INTERNVL(
                self.model,
                language_tokenizer=self._tokenizer,
                processor=self.processor,
                retain_ratio=retain_ratio,
                target_vision_tokens=target_vision_tokens,
            )
        else:
            self._model, self.processor = _MMTOK_INTERNVL(
                self._model,
                language_tokenizer=self._tokenizer,
                processor=self.processor,
                retain_ratio=retain_ratio,
                target_vision_tokens=target_vision_tokens,
            )

        self.retain_ratio = retain_ratio
        self.target_vision_tokens = target_vision_tokens
