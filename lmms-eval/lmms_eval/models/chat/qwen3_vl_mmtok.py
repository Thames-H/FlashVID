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
from lmms_eval.models.chat.qwen3_vl import (
    Qwen3_VL as Qwen3_VL_Chat,
    process_vision_info,
)
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)
from lmms_eval.protocol import ChatMessages


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
        stats_output_path: Optional[str] = None,
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
        self.stats_output_path = stats_output_path

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            return x[0], x[0]

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
        e2e_latency = 0.0
        total_tokens = 0

        for chunk in chunks:
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            chat_messages = [
                doc_to_messages[idx](self.task_dict[task][split][ids])
                for idx, (ids, task, split) in enumerate(zip(doc_id, task, split))
            ]
            chat_messages = [
                ChatMessages(**{"messages": message})
                for message in chat_messages
            ]
            gen_kwargs = all_gen_kwargs[0]

            video_kwargs = {
                "max_pixels": self.max_pixels,
                "min_pixels": self.min_pixels,
            }
            if self.fps is not None:
                video_kwargs["fps"] = self.fps
                video_kwargs["max_frames"] = self.max_num_frames
            else:
                video_kwargs["nframes"] = self.max_num_frames

            batched_messages = [
                chat_message.to_hf_messages(video_kwargs=video_kwargs)
                for chat_message in chat_messages
            ]
            texts = self.processor.apply_chat_template(
                batched_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs, video_kwargs_qwen = process_vision_info(
                batched_messages,
                return_video_kwargs=True,
                image_patch_size=16,
                return_video_metadata=True,
            )
            video_kwargs = {**video_kwargs, **video_kwargs_qwen}

            video_metadatas = None
            if video_inputs is not None:
                video_inputs, video_metadatas = zip(*video_inputs)
                video_inputs = list(video_inputs)
                video_metadatas = list(video_metadatas)

            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                video_metadata=video_metadatas,
                **video_kwargs,
                do_resize=False,
                return_tensors="pt",
            )

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            default_gen_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.0,
                "top_p": None,
                "num_beams": 1,
            }
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id

            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None
                current_gen_kwargs["top_k"] = None

            start_time = time.time()
            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=current_gen_kwargs["do_sample"],
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                top_k=current_gen_kwargs.get("top_k", None),
                use_cache=self.use_cache,
            )
            end_time = time.time()

            sample_artifact = getattr(
                self.model.model,
                "_mmtok_last_sample_artifact",
                None,
            )

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, cont)
            ]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            e2e_latency += end_time - start_time
            total_tokens += sum(len(ids) for ids in generated_ids_trimmed)

            for idx, (ans, context) in enumerate(zip(answers, texts)):
                clean_ans = parse_reasoning_model_answer(ans)
                res.append(clean_ans)
                self.cache_hook.add_partial(
                    "generate_until",
                    (context, gen_kwargs),
                    clean_ans,
                )

                eval_logger.debug(f"Question: {context}")
                eval_logger.debug(f"Model Raw Response: {ans}")
                eval_logger.debug(f"Model Clean Response: {clean_ans}")

                if sample_artifact is not None and len(answers) == 1:
                    artifact_to_write = dict(sample_artifact)
                    artifact_to_write["task_name"] = task[idx]
                    artifact_to_write["doc_id"] = doc_id[idx]
                    artifact_to_write["question_text"] = context
                    artifact_to_write["model_response"] = clean_ans
                    artifact_path = _write_sample_artifact(
                        stats_output_path=self.stats_output_path,
                        task_name=task[idx],
                        doc_id=doc_id[idx],
                        artifact=artifact_to_write,
                    )
                    if artifact_path:
                        eval_logger.info(
                            f"[Qwen3-VL-MMTok] wrote sample artifact to {artifact_path}"
                        )

            self.model.model._mmtok_last_sample_artifact = None
            pbar.update(1)

        res = re_ords.get_original(res)
        avg_speed = total_tokens / e2e_latency if e2e_latency > 0 else 0
        log_metrics(
            total_tokens=total_tokens,
            e2e_latency=e2e_latency,
            avg_speed=avg_speed,
            additional_metrics={"rank": self.rank},
        )

        pbar.close()
        return res
