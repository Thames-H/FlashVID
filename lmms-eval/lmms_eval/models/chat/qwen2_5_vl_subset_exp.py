import time
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.chat.qwen3_vl_subset_exp import (
    _clean_image_placeholder,
    _first_grid_for_sample,
    _parse_keep_ratios,
    _parse_methods,
)
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)
from lmms_eval.models.simple.qwen2_5_vl import Qwen2_5_VL as Qwen2_5_VL_Chat
from lmms_eval.protocol import ChatMessages

from flashvid.experiments.token_subset import ExperimentPipeline

process_vision_info, _has_qwen_vl = optional_import(
    "qwen_vl_utils", "process_vision_info"
)
if not _has_qwen_vl:
    eval_logger.warning(
        "Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`"
    )


@register_model("qwen2_5_vl_subset_exp")
class Qwen2_5_VL_SubsetExp(Qwen2_5_VL_Chat):
    """Qwen2.5-VL evaluation wrapper that logs token-subset properties."""

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-7B-Instruct",
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
        artifact_path: Optional[str] = None,
        keep_ratios: Optional[Union[str, Sequence[float], Sequence[str]]] = None,
        methods: Optional[Union[str, Sequence[str]]] = None,
        target_layer: int = 15,
        run_downstream: bool = False,
        enable_flashvid: bool = False,
        **kwargs,
    ) -> None:
        if enable_flashvid:
            eval_logger.warning(
                "[qwen2_5_vl_subset_exp] enable_flashvid is ignored; baseline model is used for subset extraction."
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

        self.target_layer = int(target_layer)
        self.run_downstream = bool(run_downstream)
        self.subset_keep_ratios = _parse_keep_ratios(keep_ratios)
        self.subset_methods = _parse_methods(methods)
        self.artifact_path = artifact_path
        self._subset_pipeline = ExperimentPipeline(
            model_name="qwen2.5-vl",
            model=self.model,
            processor=self.processor,
            target_layer=self.target_layer,
            methods=self.subset_methods,
            keep_ratios=self.subset_keep_ratios,
            artifact_path=artifact_path,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            downstream_kwargs={"use_cache": self.use_cache},
        )

    @staticmethod
    def _select_batch_tensor(value: torch.Tensor, sample_idx: int, batch: int):
        if value.ndim > 0 and value.shape[0] == batch:
            return value[sample_idx : sample_idx + 1]
        return value

    def _make_sample_payload(
        self,
        task: str,
        split: str,
        doc_id,
        context: str,
        visual_groups: list,
        sample_idx: int,
        image_grid_thw,
        visual_count_prefix: int,
        image_token_id: int,
    ) -> dict:
        doc = self.task_dict[task][split][doc_id] if task in self.task_dict and split in self.task_dict[task] else {}
        ground_truth = str(doc.get("answer", "")) if isinstance(doc, dict) else ""

        sample_visuals = visual_groups[sample_idx] if sample_idx < len(visual_groups) else []
        visual_count = len(sample_visuals) if sample_visuals else 0
        grid_thw = _first_grid_for_sample(
            image_grid_thw=image_grid_thw,
            sample_visual_count=visual_count,
            visual_start_idx=visual_count_prefix,
        )

        preview = np.zeros((1, 1, 3), dtype=np.uint8)
        image_size = (0, 0)
        if sample_visuals:
            first_visual = sample_visuals[0]
            if hasattr(first_visual, "convert"):
                preview = np.asarray(first_visual.convert("RGB"), dtype=np.uint8)
                if preview.ndim == 3 and preview.shape[2] == 3:
                    image_size = (int(preview.shape[0]), int(preview.shape[1]))
                else:
                    preview = np.zeros((1, 1, 3), dtype=np.uint8)

        return {
            "sample_id": f"{task}/{split}/{doc_id}",
            "benchmark": str(task),
            "question": _clean_image_placeholder(context),
            "ground_truth": ground_truth,
            "image_preview": preview,
            "image_size": image_size,
            "grid_thw": grid_thw,
            "locator_kwargs": {"grid_thw": grid_thw, "image_token_id": image_token_id},
            "patch_mapping_kwargs": {},
            "visual_count": visual_count,
            "visual_start": visual_count_prefix,
        }

    def generate_until(self, requests: List[Instance]) -> List[str]:
        if self.batch_size > 1:
            eval_logger.warning(
                "[qwen2_5_vl_subset_exp] batch_size>1: per-sample artifact metadata may be aligned conservatively."
            )

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
        subset_artifacts = 0

        for chunk in chunks:
            contexts, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]

            chat_messages = [
                ChatMessages(**{"messages": doc_to_messages[idx](self.task_dict[task][split][ids])})
                for idx, ids in enumerate(doc_id)
            ]
            gen_kwargs = all_gen_kwargs[0]

            visual_groups = []
            for message in chat_messages:
                visuals, _, _ = message.extract_media()
                visual_groups.append(list(visuals) if visuals is not None else [])

            batched_messages = [
                chat_message.to_hf_messages(
                    video_kwargs={
                        "max_pixels": self.max_pixels,
                        "min_pixels": self.min_pixels,
                        **({"fps": self.fps, "max_frames": self.max_num_frames}
                           if self.fps is not None else {"nframes": self.max_num_frames}),
                    }
                )
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
            video_kwargs = {"max_pixels": self.max_pixels, "min_pixels": self.min_pixels}
            video_kwargs.update(video_kwargs_qwen or {})

            video_metadatas = None
            if video_inputs is not None:
                video_inputs, video_metadatas = zip(*video_inputs)
                video_inputs = list(video_inputs)
                video_metadatas = list(video_metadatas)

            if self.batch_size > 1:
                inputs = self.processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    video_metadata=video_metadatas,
                    **video_kwargs,
                    do_resize=False,
                    padding=True,
                    padding_side="left",
                    return_tensors="pt",
                )
            else:
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

            batch_inputs = len(answers)
            image_grid_thw = inputs.get("image_grid_thw")
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.tolist()

            visual_start = 0
            image_token_id = int(getattr(self.model.config, "image_token_id", 151655))
            for i, (ans, context) in enumerate(zip(answers, contexts)):
                clean_ans = parse_reasoning_model_answer(ans)
                res.append(clean_ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), clean_ans)

                if self.artifact_path is not None:
                    sample = self._make_sample_payload(
                        task=task,
                        split=split,
                        doc_id=doc_id[i],
                        context=context if isinstance(context, str) else str(context),
                        visual_groups=visual_groups,
                        sample_idx=i,
                        image_grid_thw=image_grid_thw,
                        visual_count_prefix=visual_start,
                        image_token_id=image_token_id,
                    )

                    sample_visuals = visual_groups[i] if i < len(visual_groups) else []
                    visual_start += len(sample_visuals)

                    if sample["grid_thw"] is not None:
                        sample_inputs = {}
                        for key, value in inputs.items():
                            if torch.is_tensor(value):
                                sample_inputs[key] = self._select_batch_tensor(
                                    value,
                                    sample_idx=i,
                                    batch=batch_inputs,
                                )
                            else:
                                sample_inputs[key] = value

                        try:
                            self._subset_pipeline.process_sample(
                                sample=sample,
                                model_inputs=sample_inputs,
                                base_answer=clean_ans,
                                keep_ratios=self.subset_keep_ratios,
                                run_downstream=self.run_downstream,
                            )
                            subset_artifacts += 1
                        except Exception as exc:  # pragma: no cover
                            eval_logger.warning(
                                "[qwen2_5_vl_subset_exp] subset processing failed "
                                f"for doc_id={doc_id[i]}: {exc}"
                            )
                else:
                    sample_visuals = visual_groups[i] if i < len(visual_groups) else []
                    visual_start += len(sample_visuals)

                pbar.update(1)

        res = re_ords.get_original(res)
        avg_speed = total_tokens / e2e_latency if e2e_latency > 0 else 0
        log_metrics(
            total_tokens=total_tokens,
            e2e_latency=e2e_latency,
            avg_speed=avg_speed,
            additional_metrics={
                "rank": self.rank,
                "subset_artifacts": subset_artifacts,
            },
        )
        pbar.close()
        return res
