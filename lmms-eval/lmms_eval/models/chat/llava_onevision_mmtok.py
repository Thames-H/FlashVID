import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Union

import torch
from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.chat.llava_hf import (
    LlavaHf as LlavaHfChat,
    _build_llava_processor_kwargs,
    _prepare_llava_media_inputs,
)
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.protocol import ChatMessages
from .llava_onevision_visual_compare_utils import (
    attach_visual_compare_metadata,
    build_visual_compare_metadata,
)


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


def _sanitize_artifact_component(value: Union[str, int]) -> str:
    text = str(value)
    sanitized = [
        ch if ch.isalnum() or ch in {"-", "_", "."} else "_"
        for ch in text
    ]
    return "".join(sanitized)


def _write_sample_artifact(
    stats_output_path: Optional[str],
    task_name: str,
    doc_id: Union[str, int],
    artifact: dict,
) -> Optional[str]:
    if not stats_output_path:
        return None

    artifact_dir = os.path.join(stats_output_path, "artifacts", "mmtok")
    os.makedirs(artifact_dir, exist_ok=True)
    artifact_name = (
        f"{_sanitize_artifact_component(task_name)}"
        f"__doc{_sanitize_artifact_component(doc_id)}.pt"
    )
    artifact_path = os.path.join(artifact_dir, artifact_name)
    torch.save(artifact, artifact_path)
    return artifact_path


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
        stats_output_path: Optional[str] = None,
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
        self.stats_output_path = stats_output_path

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            return x[2], x[2]

        re_ords = utils.Collator(
            [reg.args for reg in requests],
            _collate,
            group_fn=lambda x: x[2],
            grouping=True,
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = (
            len(requests) // self.batch_size
            if len(requests) % self.batch_size == 0
            else len(requests) // self.batch_size + 1
        )
        pbar = tqdm(
            total=num_iters,
            disable=(self.rank != 0),
            desc="Model Responding",
        )
        e2e_latency = 0
        total_tokens = 0

        for chunk in chunks:
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            task_name = task[0]
            split_name = split[0]
            chat_messages = [
                doc_to_messages[0](self.task_dict[task_name][split_name][ids])
                for ids in doc_id
            ]
            chat_messages = [
                ChatMessages(**{"messages": message})
                for message in chat_messages
            ]
            visuals = []
            videos = []
            for messages in chat_messages:
                visual, video, _ = messages.extract_media()
                visuals.append(visual)
                videos.append(video)
            visuals = self.flatten(visuals)
            videos = self.flatten(videos)
            assert self.batch_size_per_gpu == 1, (
                "Do not support batch_size_per_gpu > 1 for now"
            )

            messages = chat_messages[0].model_dump()["messages"]
            text = self._image_processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            visuals, videos, image_sizes = _prepare_llava_media_inputs(
                visuals,
                videos,
            )
            images_kwargs, videos_kwargs = _build_llava_processor_kwargs(
                self.model.config,
                self.max_frames_num,
            )
            inputs = self._prepare_processor_inputs(
                self._image_processor(
                    images=visuals,
                    videos=videos,
                    text=text,
                    return_tensors="pt",
                    **images_kwargs,
                    **videos_kwargs,
                )
            )

            gen_kwargs = dict(all_gen_kwargs[0])
            gen_kwargs["image_sizes"] = image_sizes
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            do_sample = gen_kwargs["temperature"] > 0

            try:
                start_time = time.time()
                cont = self.model.generate(
                    **inputs,
                    do_sample=do_sample,
                    temperature=gen_kwargs["temperature"] if do_sample else None,
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                    pad_token_id=self.eot_token_id,
                    eos_token_id=self.eot_token_id,
                )
                end_time = time.time()
                cont = cont[:, inputs["input_ids"].shape[-1] :]
                e2e_latency += end_time - start_time
                total_tokens += cont.shape[-1] if len(cont.shape) > 1 else len(cont)
            except Exception as error:
                eval_logger.error(f"Error {error} in generating")
                cont = ""

            sample_artifact = getattr(
                self.model.model,
                "_mmtok_last_sample_artifact",
                None,
            )
            text_outputs = (
                self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
                if cont != ""
                else ""
            )

            res.append(text_outputs)
            self.cache_hook.add_partial("generate_until", (text, gen_kwargs), text_outputs)

            if sample_artifact is not None and len(chunk) == 1:
                artifact_to_write = dict(sample_artifact)
                artifact_to_write["task_name"] = task_name
                artifact_to_write["doc_id"] = doc_id[0]
                artifact_to_write["question_text"] = text
                artifact_to_write["model_response"] = text_outputs
                artifact_to_write = attach_visual_compare_metadata(
                    artifact_to_write,
                    build_visual_compare_metadata(
                        image_inputs=visuals,
                        video_inputs=videos,
                        model_config=self.model.config,
                        n_visual_tokens_scored=artifact_to_write["metadata"][
                            "n_visual_tokens_scored"
                        ],
                        vision_aspect_ratio=getattr(
                            self.model.config,
                            "vision_aspect_ratio",
                            "anyres_max_9",
                        ),
                    ),
                )
                artifact_path = _write_sample_artifact(
                    stats_output_path=self.stats_output_path,
                    task_name=task_name,
                    doc_id=doc_id[0],
                    artifact=artifact_to_write,
                )
                if artifact_path:
                    eval_logger.info(
                        f"[LLaVA-OneVision-MMTok] wrote sample artifact to {artifact_path}"
                    )

            self.model.model._mmtok_last_sample_artifact = None
            pbar.update(1)

        res = re_ords.get_original(res)
        metric_dict = {
            "total_tokens": total_tokens,
            "e2e_latency": e2e_latency,
            "avg_speed": total_tokens / e2e_latency if e2e_latency > 0 else 0,
            "additional_metrics": {
                "rank": self.rank,
            },
        }
        log_metrics(**metric_dict)

        pbar.close()
        return res
