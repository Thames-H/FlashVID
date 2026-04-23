import math
import time
from typing import List, Optional, Sequence, Union

import numpy as np
import PIL
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

from flashvid.experiments.token_subset import ExperimentPipeline


def _clean_image_placeholder(text: str) -> str:
    return text.replace("<image>", " ").strip()


def _parse_keep_ratios(
    keep_ratios: Optional[Union[str, Sequence[float], Sequence[str], float, int]],
) -> tuple[float, ...]:
    if keep_ratios is None:
        return (0.1, 0.25, 0.5, 0.75)
    if isinstance(keep_ratios, (float, int)):
        return (float(keep_ratios),)
    if isinstance(keep_ratios, str):
        ratios = []
        for piece in keep_ratios.split(","):
            piece = piece.strip()
            if piece:
                ratios.append(float(piece))
        return tuple(ratios) if ratios else (0.1, 0.25, 0.5, 0.75)
    return tuple(float(v) for v in keep_ratios)


def _parse_methods(methods: Optional[Union[str, Sequence[str]]]) -> tuple[str, ...]:
    if methods is None:
        return ("fetp", "attention", "mmtok")
    if isinstance(methods, str):
        methods = [m.strip() for m in methods.split(",") if m.strip()]
    return tuple(str(m).lower() for m in methods)


@register_model("llava_onevision_subset_exp")
class LlavaOnevisionSubsetExp(LlavaHfChat):
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
        artifact_path: Optional[str] = None,
        keep_ratios: Optional[Union[str, Sequence[float], Sequence[str]]] = None,
        methods: Optional[Union[str, Sequence[str]]] = None,
        target_layer: int = 15,
        run_downstream: bool = False,
        **kwargs,
    ) -> None:
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

        self.target_layer = int(target_layer)
        self.run_downstream = bool(run_downstream)
        self.subset_keep_ratios = _parse_keep_ratios(keep_ratios)
        self.subset_methods = _parse_methods(methods)
        self.artifact_path = artifact_path
        self._subset_pipeline = ExperimentPipeline(
            model_name="llava-onevision",
            model=self.model,
            processor=self._image_processor,
            target_layer=self.target_layer,
            methods=self.subset_methods,
            keep_ratios=self.subset_keep_ratios,
            artifact_path=artifact_path,
            eos_token_id=self.eot_token_id,
            pad_token_id=self.eot_token_id,
            downstream_kwargs={"use_cache": self.use_cache},
        )

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            return x[2], x[2]

        re_ords = utils.Collator([reg.args for reg in requests], _collate, group_fn=lambda x: x[2], grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        e2e_latency = 0.0
        total_tokens = 0
        subset_artifacts = 0

        for chunk in chunks:
            contexts, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            task_name = task[0]
            split_name = split[0]
            chat_messages = [doc_to_messages[0](self.task_dict[task_name][split_name][ids]) for ids in doc_id]
            visuals_batch = []
            videos_batch = []
            for msg in chat_messages:
                chat_message = msg if hasattr(msg, "extract_media") else None
                if chat_message is None:
                    from lmms_eval.protocol import ChatMessages

                    chat_message = ChatMessages(**{"messages": msg})
                visuals, videos, _ = chat_message.extract_media()
                visuals_batch.append(list(visuals) if visuals else [])
                videos_batch.append(list(videos) if videos else [])

            visuals = self.flatten(visuals_batch)
            videos = self.flatten(videos_batch)
            assert self.batch_size_per_gpu == 1, "Do not support batch_size_per_gpu > 1 for now"
            gen_kwargs = all_gen_kwargs[0]

            messages = chat_messages[0]
            if not isinstance(messages, list):
                messages = messages.model_dump()["messages"]
            text = self._image_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            visuals, videos, image_sizes = _prepare_llava_media_inputs(visuals, videos)
            try:
                visuals, videos = self._prepare_chat_media_inputs(visuals, videos)
            except Exception as exc:
                eval_logger.error(f"Error {exc} when loading video: {videos}")
                res.append("")
                pbar.update(1)
                continue

            images_kwargs, videos_kwargs = _build_llava_processor_kwargs(
                self.model.config,
                self.max_frames_num,
            )
            if videos is not None and len(videos) > 0 and not isinstance(videos[0], str):
                videos_kwargs.pop("num_frames", None)
                videos_kwargs.pop("do_sample_frames", None)

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

            gen_kwargs["image_sizes"] = image_sizes
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            do_sample = True if gen_kwargs["temperature"] > 0 else False

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
            except Exception as exc:
                eval_logger.error(f"Error {exc} in generating")
                cont = ""

            text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0] if cont != "" else ""
            res.append(text_outputs)
            self.cache_hook.add_partial("generate_until", (text, gen_kwargs), text_outputs)

            if self.artifact_path is not None:
                doc = self.task_dict[task_name][split_name][doc_id[0]]
                ground_truth = str(doc.get("answer", "")) if isinstance(doc, dict) else ""

                preview = np.zeros((1, 1, 3), dtype=np.uint8)
                image_size = (0, 0)
                if visuals and isinstance(visuals[0], PIL.Image.Image):
                    preview = np.asarray(visuals[0].convert("RGB"), dtype=np.uint8)
                    image_size = (int(preview.shape[0]), int(preview.shape[1]))

                image_token_id = int(getattr(self.model.config, "image_token_id", -200))
                input_ids = inputs["input_ids"]
                image_positions = torch.where(input_ids[0] == image_token_id)[0]
                n_vis = int(image_positions.numel())
                approx_side = max(1, int(math.sqrt(max(n_vis, 1))))

                sample = {
                    "sample_id": f"{task_name}/{split_name}/{doc_id[0]}",
                    "benchmark": str(task_name),
                    "question": _clean_image_placeholder(str(contexts[0])),
                    "ground_truth": ground_truth,
                    "image_preview": preview,
                    "image_size": image_size,
                    "grid_thw": None,
                    "locator_kwargs": {"image_token_id": image_token_id},
                    "patch_mapping_kwargs": {
                        "base_grid": (approx_side, approx_side),
                        "crop_grid": (0, 0),
                        "crop_layout": (0, 0),
                        "has_newline": False,
                    },
                }
                try:
                    self._subset_pipeline.process_sample(
                        sample=sample,
                        model_inputs=inputs,
                        base_answer=text_outputs,
                        keep_ratios=self.subset_keep_ratios,
                        run_downstream=self.run_downstream,
                    )
                    subset_artifacts += 1
                except Exception as exc:  # pragma: no cover
                    eval_logger.warning(
                        "[llava_onevision_subset_exp] subset processing failed "
                        f"for doc_id={doc_id[0]}: {exc}"
                    )

            pbar.update(1)

        res = re_ords.get_original(res)
        metric_dict = {
            "total_tokens": total_tokens,
            "e2e_latency": e2e_latency,
            "avg_speed": total_tokens / e2e_latency if e2e_latency > 0 else 0,
            "additional_metrics": {
                "rank": self.rank,
                "subset_artifacts": subset_artifacts,
            },
        }
        log_metrics(**metric_dict)

        pbar.close()
        return res
