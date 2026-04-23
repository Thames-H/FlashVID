import time
from pathlib import Path
from typing import List, Optional

import torch
from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)
from lmms_eval.models.simple.qwen3_vl import Qwen3_VL as Qwen3_VLSimple
from lmms_eval.protocol import ChatMessages
from sink_analysis.collect.patch_mapping import build_qwen3vl_mapping
from sink_analysis.collect.sample_records import extract_ground_truth_from_doc
from sink_analysis.collect.writer import (
    build_partial_record_payload,
    load_override_map,
    lookup_override_indices,
    write_partial_record,
)
from sink_analysis.schema import SinkAnalysisExportConfig, build_sample_id, keep_ratio_to_label

process_vision_info, _has_qwen_vl = optional_import("qwen_vl_utils", "process_vision_info")
if not _has_qwen_vl:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


@register_model("qwen3_vl_chat")
class Qwen3_VL(Qwen3_VLSimple):
    is_simple = False

    def __init__(
        self,
        *args,
        sink_analysis_output_root: Optional[str] = None,
        sink_analysis_method_name: Optional[str] = None,
        sink_analysis_keep_ratio: Optional[str] = None,
        sink_analysis_override_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._sink_analysis_config = None
        if sink_analysis_output_root and sink_analysis_method_name and sink_analysis_keep_ratio:
            self._sink_analysis_config = SinkAnalysisExportConfig(
                output_root=Path(sink_analysis_output_root),
                model_name="qwen3-vl",
                method_name=sink_analysis_method_name,
                keep_ratio_label=keep_ratio_to_label(sink_analysis_keep_ratio),
            )
        self._sink_analysis_override_map = load_override_map(sink_analysis_override_path)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        # A dummy collate here to sort by doc id
        def _collate(x):
            return x[0], x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator(
            [reg.args for reg in requests],
            _collate,
            group_fn=lambda x: x[2],
            grouping=True,
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        e2e_latency = 0
        total_tokens = 0
        for chunk in chunks:
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            chat_messages = [doc_to_messages[idx](self.task_dict[task][split][ids]) for idx, (ids, task, split) in enumerate(zip(doc_id, task, split))]
            chat_messages: List[ChatMessages] = [ChatMessages(**{"messages": message}) for message in chat_messages]
            visuals = []
            videos = []
            for messages in chat_messages:
                visual, video, _ = messages.extract_media()
                visuals.append(visual)
                videos.append(video)
            visuals = self.flatten(visuals)
            videos = self.flatten(videos)
            gen_kwargs = all_gen_kwargs[0]

            # Apply chat template
            video_kwargs = {
                "max_pixels": self.max_pixels,
                "min_pixels": self.min_pixels,
            }
            if self.fps is not None:
                video_kwargs["fps"] = self.fps
                # limit the number of frames in case fps is set
                video_kwargs["max_frames"] = self.max_num_frames
            else:
                video_kwargs["nframes"] = self.max_num_frames
            batched_messages = [chat_message.to_hf_messages(video_kwargs=video_kwargs) for chat_message in chat_messages]
            texts = self.processor.apply_chat_template(batched_messages, tokenize=False, add_generation_prompt=True)
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
                video_inputs, video_metadatas = (
                    list(video_inputs),
                    list(video_metadatas),
                )

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

            # Set default generation kwargs
            default_gen_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.0,  # Set to 0 for greedy default
                "top_p": None,
                "num_beams": 1,
            }
            # Update with provided kwargs
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id

            export_source = getattr(self.model, "model", self.model)
            export_config = getattr(self, "_sink_analysis_config", None)
            if export_source is not None:
                setattr(export_source, "_sink_analysis_last_export", None)
                setattr(export_source, "_sink_analysis_override_keep_indices", None)
            if export_config is not None and len(doc_id) == 1 and export_source is not None:
                sample_id = build_sample_id(str(task[0]), doc_id[0])
                override_indices = lookup_override_indices(
                    getattr(self, "_sink_analysis_override_map", None),
                    model_name=export_config.model_name,
                    sample_id=sample_id,
                    keep_ratio_label=export_config.keep_ratio_label,
                )
                if override_indices is not None:
                    setattr(
                        export_source,
                        "_sink_analysis_override_keep_indices",
                        torch.tensor(
                            override_indices,
                            device=self.device,
                            dtype=torch.long,
                        ),
                    )

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

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            # Calculate timing metrics for batch
            e2e_latency += end_time - start_time
            total_tokens += sum(len(ids) for ids in generated_ids_trimmed)

            for index, (ans, context) in enumerate(zip(answers, texts)):
                clean_ans = parse_reasoning_model_answer(ans)
                res.append(clean_ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), clean_ans)

                eval_logger.debug(f"Question: {context}")
                eval_logger.debug(f"Model Raw Response: {ans}")
                eval_logger.debug(f"Model Clean Response: {clean_ans}")
                if export_config is not None and len(doc_id) == 1 and index == 0:
                    export_payload = getattr(
                        export_source,
                        "_sink_analysis_last_export",
                        None,
                    )
                    sample_doc = self.task_dict[task[index]][split[index]][doc_id[index]]
                    image = visuals[index] if index < len(visuals) else None
                    image_size = None
                    if image is not None:
                        raw_size = getattr(image, "size", None)
                        if raw_size is not None and not callable(raw_size):
                            image_size = (int(raw_size[1]), int(raw_size[0]))
                    image_grid = inputs.get("image_grid_thw")
                    if image_grid is not None and torch.is_tensor(image_grid) and image_grid.ndim > 1:
                        image_grid = image_grid[0]
                    payload = build_partial_record_payload(
                        export_config=export_config,
                        task_name=str(task[index]),
                        doc_id=doc_id[index],
                        benchmark=str(task[index]),
                        target=extract_ground_truth_from_doc(sample_doc),
                        messages=chat_messages[index].model_dump()["messages"],
                        answer=clean_ans,
                        export_payload=export_payload,
                        patch_mapping=build_qwen3vl_mapping(
                            image_size=image_size,
                            grid_thw=image_grid,
                        )
                        if export_payload is not None
                        else None,
                        image=image,
                    )
                    write_partial_record(export_config.output_root, payload)
            if export_source is not None:
                setattr(export_source, "_sink_analysis_override_keep_indices", None)
                setattr(export_source, "_sink_analysis_last_export", None)
            # reorder this group of results back to original unsorted form
            pbar.update(1)
        res = re_ords.get_original(res)

        # Calculate average speed
        avg_speed = total_tokens / e2e_latency if e2e_latency > 0 else 0
        # Log metrics
        metric_dict = {
            "total_tokens": total_tokens,
            "e2e_latency": e2e_latency,
            "avg_speed": avg_speed,
            "additional_metrics": {
                "rank": self.rank,
            },
        }
        log_metrics(**metric_dict)

        pbar.close()
        return res
