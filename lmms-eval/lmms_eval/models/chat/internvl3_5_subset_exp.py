import time
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.chat.internvl_hf import (
    InternVLHf,
    _prepare_internvl_media_inputs,
)
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.protocol import ChatMessages

from flashvid.experiments.token_subset import ExperimentPipeline


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


@register_model("internvl3_5_subset_exp")
class InternVL3_5_SubsetExp(InternVLHf):
    is_simple = False

    def __init__(
        self,
        pretrained: str = "OpenGVLab/InternVL3_5-8B-HF",
        revision: str = "main",
        device: str = "cuda",
        device_map: str = "auto",
        batch_size: int = 1,
        min_patches: int = 1,
        max_patches: int = 12,
        num_frames: int = 8,
        fps: Optional[float] = None,
        trust_remote_code: Optional[bool] = False,
        low_cpu_mem_usage: Optional[bool] = True,
        attn_implementation: Optional[str] = "flash_attention_2",
        use_cache: bool = True,
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
        self.target_layer = int(target_layer)
        self.run_downstream = bool(run_downstream)
        self.subset_keep_ratios = _parse_keep_ratios(keep_ratios)
        self.subset_methods = _parse_methods(methods)
        self.artifact_path = artifact_path
        self._subset_pipeline = ExperimentPipeline(
            model_name="internvl3.5",
            model=self.model,
            processor=self.processor,
            target_layer=self.target_layer,
            methods=self.subset_methods,
            keep_ratios=self.subset_keep_ratios,
            artifact_path=artifact_path,
            eos_token_id=self.eot_token_id,
            pad_token_id=self.pad_token_id,
            downstream_kwargs={"use_cache": self.use_cache},
        )

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res: List[str] = []

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
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            task_name = task[0]
            split_name = split[0]

            chat_messages = [doc_to_messages[0](self.task_dict[task_name][split_name][ids]) for ids in doc_id]
            chat_messages = [ChatMessages(**{"messages": message}) for message in chat_messages]
            visuals = []
            videos = []
            for messages in chat_messages:
                visual, video, _ = messages.extract_media()
                visuals.append(visual)
                videos.append(video)
            visuals = self.flatten(visuals)
            videos = self.flatten(videos)

            images_kwargs = {}
            videos_kwargs = {}
            if self.min_patches is not None:
                images_kwargs["min_patches"] = self.min_patches
            if self.max_patches is not None:
                images_kwargs["max_patches"] = self.max_patches
            if self.num_frames is not None:
                videos_kwargs["num_frames"] = self.num_frames
            if self.fps is not None:
                videos_kwargs["fps"] = self.fps

            messages = chat_messages[0].model_dump()["messages"]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            visuals, videos, image_sizes = _prepare_internvl_media_inputs(
                visuals,
                videos,
            )
            inputs = self.processor(
                images=visuals,
                videos=videos,
                text=text,
                return_tensors="pt",
                **images_kwargs,
                **videos_kwargs,
            ).to(self.device, self.model.dtype)

            gen_kwargs = all_gen_kwargs[0]
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
                    pad_token_id=self.pad_token_id,
                    eos_token_id=self.eot_token_id,
                    do_sample=do_sample,
                    temperature=gen_kwargs["temperature"] if do_sample else None,
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                )
                end_time = time.time()
                generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, cont)]
                answers = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                e2e_latency += end_time - start_time
                total_tokens += sum(len(ids) for ids in generated_ids_trimmed)
            except Exception as exc:
                eval_logger.error(f"Error {exc} in generating")
                answers = [""]

            for answer in answers:
                res.append(answer)
                self.cache_hook.add_partial("generate_until", (text, gen_kwargs), answer)

            if self.artifact_path is not None:
                doc = self.task_dict[task_name][split_name][doc_id[0]]
                ground_truth = str(doc.get("answer", "")) if isinstance(doc, dict) else ""

                preview = np.zeros((1, 1, 3), dtype=np.uint8)
                image_size = (0, 0)
                if visuals and hasattr(visuals[0], "convert"):
                    preview = np.asarray(visuals[0].convert("RGB"), dtype=np.uint8)
                    image_size = (int(preview.shape[0]), int(preview.shape[1]))

                image_token_id = int(getattr(self.model.config, "image_token_id", -1))
                n_vis = int((inputs["input_ids"][0] == image_token_id).sum().item()) if image_token_id >= 0 else 0
                sample = {
                    "sample_id": f"{task_name}/{split_name}/{doc_id[0]}",
                    "benchmark": str(task_name),
                    "question": str(ctx[0]),
                    "ground_truth": ground_truth,
                    "image_preview": preview,
                    "image_size": image_size,
                    "locator_kwargs": {"image_token_id": image_token_id},
                    "patch_mapping_kwargs": {
                        "num_patches_list": [n_vis] if n_vis > 0 else None,
                    },
                }
                try:
                    self._subset_pipeline.process_sample(
                        sample=sample,
                        model_inputs=inputs,
                        base_answer=answers[0] if answers else "",
                        keep_ratios=self.subset_keep_ratios,
                        run_downstream=self.run_downstream,
                    )
                    subset_artifacts += 1
                except Exception as exc:  # pragma: no cover
                    eval_logger.warning(
                        "[internvl3_5_subset_exp] subset processing failed "
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
