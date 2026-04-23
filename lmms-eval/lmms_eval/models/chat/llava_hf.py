import time
import warnings
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.protocol import ChatMessages
from sink_analysis.collect.patch_mapping import build_onevision_mapping
from sink_analysis.collect.sample_records import extract_ground_truth_from_doc
from sink_analysis.collect.writer import (
    build_partial_record_payload,
    load_override_map,
    lookup_override_indices,
    write_partial_record,
)
from sink_analysis.schema import SinkAnalysisExportConfig, build_sample_id, keep_ratio_to_label

warnings.filterwarnings("ignore")

from loguru import logger as eval_logger

from lmms_eval.models.simple.llava_hf import LlavaHf as LlavaHfSimple

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

# Default chat for llava-hf/llava-1.5 models: https://huggingface.co/collections/llava-hf/llava-15-65f762d5b6941db5c2ba07e0
VICUNA_CHAT_TEMPLATE = "{% for message in messages %}{% if loop.index0 == 0 %}A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {{ message['content'] }} {% elif message['role'] == 'user' %}USER: {{ message['content'] }} {% else %} ASSISTANT: {{ message['content'] }}{{ eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"


@register_model("llava_hf_chat")
class LlavaHf(LlavaHfSimple):
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
                model_name="llava-onevision",
                method_name=sink_analysis_method_name,
                keep_ratio_label=keep_ratio_to_label(sink_analysis_keep_ratio),
            )
        self._sink_analysis_override_map = load_override_map(sink_analysis_override_path)

    def _prepare_chat_media_inputs(self, visuals, videos):
        prepared_visuals = visuals if len(visuals) != 0 else None
        prepared_videos = None
        if len(videos) != 0:
            prepared_videos = [self.load_video(videos, self.max_frames_num)]
        return prepared_visuals, prepared_videos

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        # A dummy collate here to sort by doc id
        def _collate(x):
            return x[2], x[2]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, group_fn=lambda x: x[2], grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        e2e_latency = 0
        total_tokens = 0
        for chunk in chunks:
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            chat_messages = [doc_to_messages[0](self.task_dict[task][split][ids]) for ids in doc_id]
            chat_messages: List[ChatMessages] = [ChatMessages(**{"messages": message}) for message in chat_messages]
            visuals = []
            videos = []
            for messages in chat_messages:
                visual, video, _ = messages.extract_media()
                visuals.append(visual)
                videos.append(video)
            visuals = self.flatten(visuals)
            videos = self.flatten(videos)
            assert self.batch_size_per_gpu == 1, "Do not support batch_size_per_gpu > 1 for now"
            preview_image = visuals[0] if visuals else None

            # Apply chat template
            messages = chat_messages[0].model_dump()["messages"]
            text = self._image_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if self.accelerator.is_main_process and doc_id[0] % 100 == 0:
                eval_logger.debug(f"Prompt for doc ID {doc_id[0]}:\n\n{text}\n")

            try:
                visuals, videos = self._prepare_chat_media_inputs(visuals, videos)
            except Exception as e:
                eval_logger.error(f"Error {e} when loading video: {videos}")
                res.append("")
                pbar.update(1)
                continue

            inputs = self._prepare_processor_inputs(
                self._image_processor(
                    images=visuals,
                    videos=videos,
                    text=text,
                    return_tensors="pt",
                )
            )

            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]

            gen_kwargs["image_sizes"] = [] if visuals is None else [visuals[idx].size for idx in range(len(visuals))]
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            do_sample = True if gen_kwargs["temperature"] > 0 else False
            export_source = getattr(self.model, "model", self.model)
            export_config = getattr(self, "_sink_analysis_config", None)
            if export_source is not None:
                setattr(export_source, "_sink_analysis_last_export", None)
                setattr(export_source, "_sink_analysis_override_keep_indices", None)
            if export_config is not None and len(doc_id) == 1 and export_source is not None:
                sample_id = build_sample_id(str(task), doc_id[0])
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

                # Calculate timing metrics
                e2e_latency += end_time - start_time
                total_tokens += cont.shape[-1] if len(cont.shape) > 1 else len(cont)

            except Exception as e:
                eval_logger.error(f"Error {e} in generating")
                cont = ""
                e2e_latency += 0
                total_tokens += 0

            text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0] if cont != "" else ""

            if self.accelerator.is_main_process and doc_id[0] % 100 == 0:
                eval_logger.debug(f"Generated text for doc ID {doc_id[0]}:\n\n{text_outputs}\n")

            res.append(text_outputs)
            self.cache_hook.add_partial("generate_until", (text, gen_kwargs), text_outputs)
            if export_config is not None and len(doc_id) == 1:
                export_payload = getattr(export_source, "_sink_analysis_last_export", None)
                sample_doc = self.task_dict[task][split][doc_id[0]]
                vision_config = getattr(self.model.config, "vision_config", None)
                preview_image_size = None
                if preview_image is not None:
                    raw_size = getattr(preview_image, "size", None)
                    if raw_size is not None and not callable(raw_size):
                        preview_image_size = (int(raw_size[1]), int(raw_size[0]))
                payload = build_partial_record_payload(
                    export_config=export_config,
                    task_name=str(task),
                    doc_id=doc_id[0],
                    benchmark=str(task),
                    target=extract_ground_truth_from_doc(sample_doc),
                    messages=messages,
                    answer=text_outputs,
                    export_payload=export_payload,
                    patch_mapping=build_onevision_mapping(
                        image_size=preview_image_size,
                        image_grid_pinpoints=getattr(self.model.config, "image_grid_pinpoints", None),
                        vision_image_size=getattr(vision_config, "image_size", 384),
                        vision_patch_size=getattr(vision_config, "patch_size", 14),
                        vision_aspect_ratio=getattr(self.model.config, "vision_aspect_ratio", "anyres_max_9"),
                    )
                    if export_payload is not None
                    else None,
                    image=preview_image,
                )
                write_partial_record(export_config.output_root, payload)
            if export_source is not None:
                setattr(export_source, "_sink_analysis_override_keep_indices", None)
                setattr(export_source, "_sink_analysis_last_export", None)
            pbar.update(1)
        # reorder this group of results back to original unsorted form
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
