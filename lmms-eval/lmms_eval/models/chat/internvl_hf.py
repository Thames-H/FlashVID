import json
import os
import time
import warnings
from typing import List, Optional, Tuple
from datetime import timedelta

import torch
from tqdm import tqdm
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from accelerate.utils import InitProcessGroupKwargs
from tqdm import tqdm
from transformers import (
    InternVLForConditionalGeneration,
    AutoProcessor,
)

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.protocol import ChatMessages

warnings.filterwarnings("ignore")

from loguru import logger as eval_logger


def _prepare_internvl_media_inputs(
    visuals,
    videos,
) -> Tuple[Optional[list], Optional[list], List[Tuple[int, int]]]:
    normalized_visuals = visuals or None
    normalized_videos = videos or None
    image_sizes: List[Tuple[int, int]] = []
    if normalized_visuals is not None:
        image_sizes = [
            tuple(map(int, visual.size))
            for visual in normalized_visuals
            if hasattr(visual, "size")
        ]
    return normalized_visuals, normalized_videos, image_sizes


def _resolve_internvl_video_size(model_config) -> Optional[dict]:
    image_size = getattr(model_config, "image_size", None)
    if image_size is None:
        vision_config = getattr(model_config, "vision_config", None)
        image_size = getattr(vision_config, "image_size", None)
    if image_size is None:
        return None

    if isinstance(image_size, (list, tuple)):
        if len(image_size) >= 2:
            height, width = image_size[0], image_size[1]
        elif len(image_size) == 1:
            height = width = image_size[0]
        else:
            return None
    else:
        height = width = image_size

    try:
        return {"height": int(height), "width": int(width)}
    except (TypeError, ValueError):
        return None


def _build_internvl_processor_kwargs(
    *,
    model_config,
    min_patches: Optional[int],
    max_patches: Optional[int],
    num_frames: Optional[int],
    fps: Optional[float],
) -> Tuple[dict, dict]:
    images_kwargs = {}
    videos_kwargs = {}
    if min_patches is not None:
        images_kwargs["min_patches"] = min_patches
    if max_patches is not None:
        images_kwargs["max_patches"] = max_patches

    if num_frames is not None:
        videos_kwargs["num_frames"] = num_frames
        videos_kwargs["do_sample_frames"] = True
    if fps is not None:
        videos_kwargs["fps"] = fps
        videos_kwargs["do_sample_frames"] = True

    video_size = _resolve_internvl_video_size(model_config)
    if video_size is not None:
        videos_kwargs["size"] = video_size

    return images_kwargs, videos_kwargs


def _validate_internvl_hf_checkpoint(pretrained: str) -> None:
    if not pretrained or not os.path.isdir(pretrained):
        return

    config_path = os.path.join(pretrained, "config.json")
    if not os.path.isfile(config_path):
        return

    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
    except Exception:
        return

    model_type = config.get("model_type")
    if model_type == "internvl_chat":
        raise ValueError(
            "The checkpoint at "
            f"'{pretrained}' is an original InternVL chat-format model "
            "(model_type='internvl_chat'), but internvl_hf / "
            "internvl3_5_ours_v3 expects the Hugging Face-format "
            "InternVL checkpoint (for example "
            "'OpenGVLab/InternVL3_5-8B-HF' or a local "
            "'.../InternVL3_5-8B-HF' directory)."
        )


@register_model("internvl_hf_chat")
class InternVLHf(lmms):
    """
    InternVL Model for Hugging Face Transformers: https://huggingface.co/docs/transformers/v4.55.4/en/model_doc/internvl
    At present, the OpenGVLab has provided the HF format model weights for InternVL3 and InternVL3.5:
        InternVL3-1B: https://huggingface.co/OpenGVLab/InternVL3-1B-hf
        InternVL3-2B: https://huggingface.co/OpenGVLab/InternVL3-2B-hf
        InternVL3-8B: https://huggingface.co/OpenGVLab/InternVL3-8B-hf
        InternVL3-14B: https://huggingface.co/OpenGVLab/InternVL3-14B-hf
        InternVL3-38B: https://huggingface.co/OpenGVLab/InternVL3-38B-hf
        InternVL3-78B: https://huggingface.co/OpenGVLab/InternVL3-78B-hf

        InternVL3.5-1B: https://huggingface.co/OpenGVLab/InternVL3_5-1B-HF
        InternVL3.5-2B: https://huggingface.co/OpenGVLab/InternVL3_5-2B-HF
        InternVL3.5-4B: https://huggingface.co/OpenGVLab/InternVL3_5-4B-HF
        InternVL3.5-8B: https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF
        InternVL3.5-14B: https://huggingface.co/OpenGVLab/InternVL3_5-14B-HF
        InternVL3.5-38B: https://huggingface.co/OpenGVLab/InternVL3_5-38B-HF
        ...

    Example usage:

    accelerate launch --num_processes=8 --main_process_port 12345 -m lmms_eval \
        --model internvl_hf \
        --model_args pretrained=OpenGVLab/InternVL3_5-8B-HF \
        --tasks seedbench \
        --batch_size 1 \
        --output_path ./logs/ \
        --log_samples
    """

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
        num_frames: int = 32,
        fps: Optional[float] = None,
        trust_remote_code: Optional[bool] = False,
        low_cpu_mem_usage: Optional[bool] = False,
        attn_implementation: Optional[str] = None,
        use_cache: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        self.path = pretrained
        self.min_patches = min_patches
        self.max_patches = max_patches
        self.num_frames = num_frames
        self.fps = fps

        batch_size_int = int(batch_size)
        assert batch_size_int == 1, f"Batch size should be 1 for InternVLHf, but got {batch_size_int}."
        self.batch_size_per_gpu = batch_size_int

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator

        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        _validate_internvl_hf_checkpoint(self.path)

        self._model = InternVLForConditionalGeneration.from_pretrained(
            self.path,
            revision=revision,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=low_cpu_mem_usage,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
            device_map=self.device_map,
        ).eval()
        self._config = self._model.config

        self.processor = AutoProcessor.from_pretrained(
            self.path,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        self._tokenizer = self.processor.tokenizer
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")

            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        """Return the model configuration."""
        return self._config

    @property
    def tokenizer(self):
        """Return the tokenizer."""
        return self._tokenizer

    @property
    def model(self):
        """Return the unwrapped model."""
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        """Return the end-of-sentence token ID to replace the end-of-text token ID."""
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self):
        """Return the padding token ID."""
        return self.tokenizer.pad_token_id

    @property
    def max_length(self):
        """Return the maximum sequence length."""
        return self._max_length

    @property
    def batch_size(self):
        """Return the batch size per GPU."""
        return self.batch_size_per_gpu

    @property
    def device(self):
        """Return the device."""
        return self._device

    @property
    def rank(self):
        """Return the process rank."""
        return self._rank

    @property
    def world_size(self):
        """Return the world size."""
        return self._world_size

    def flatten(self, input):
        """Flatten a nested list."""
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Generate responses for a list of requests.

        Args:
            requests: List of Instance objects containing generation requests.

        Returns:
            List of generated response strings.
        """
        self.load_cache()
        cached_responses, pending_requests = self.partition_loaded_cache_requests(
            requests
        )
        if not pending_requests:
            return self.merge_cached_and_generated_responses(
                requests,
                cached_responses,
                {},
            )

        generated_responses: Dict[Tuple[str, int], str] = {}

        # A dummy collate here to sort by doc id
        def _collate(x):
            return x.args[2], x.args[2]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator(
            pending_requests,
            _collate,
            group_fn=lambda x: x.args[2],
            grouping=True,
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = (
            len(pending_requests) // self.batch_size
            if len(pending_requests) % self.batch_size == 0
            else len(pending_requests) // self.batch_size + 1
        )
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        e2e_latency = 0
        total_tokens = 0
        for chunk in chunks:
            chunk_requests = list(chunk)
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(
                *[req.args for req in chunk_requests]
            )
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
            visuals, videos, image_sizes = _prepare_internvl_media_inputs(
                visuals,
                videos,
            )

            images_kwargs, videos_kwargs = _build_internvl_processor_kwargs(
                model_config=self.model.config,
                min_patches=self.min_patches,
                max_patches=self.max_patches,
                num_frames=self.num_frames,
                fps=self.fps,
            )

            # Apply chat template
            messages = chat_messages[0].model_dump()["messages"]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if self.accelerator.is_main_process and doc_id[0] % 100 == 0:
                eval_logger.debug(f"Prompt for doc ID {doc_id[0]}:\n\n{text}\n")

            inputs = self.processor(
                images=visuals,
                videos=videos,
                text=text,
                return_tensors="pt",
                **images_kwargs,
                **videos_kwargs,
            ).to(self.device, self.model.dtype)

            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]

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

                generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
                answers = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                # Calculate timing metrics
                e2e_latency += end_time - start_time
                total_tokens += sum(len(ids) for ids in generated_ids_trimmed)
            except Exception as e:
                eval_logger.error(f"Error {e} in generating")
                cont = ""
                e2e_latency += 0
                total_tokens += 0

            if self.accelerator.is_main_process and doc_id[0] % 100 == 0:
                eval_logger.debug(f"Generated text for doc ID {doc_id[0]}:\n\n{answers}\n")

            for req, answer in zip(chunk_requests, answers):
                generated_responses[(req.task_name, req.doc_id)] = answer
                self.add_request_response_to_cache(req, answer)
                self.cache_hook.add_partial("generate_until", (text, gen_kwargs), answer)
            pbar.update(1)

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
        return self.merge_cached_and_generated_responses(
            requests,
            cached_responses,
            generated_responses,
        )

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Compute log-likelihood for requests. Not implemented for InternVLHf."""
        # TODO: Implement log-likelihood computation for InternVLHf.
        raise NotImplementedError("Loglikelihood is not implemented for InternVLHf.")

    def generate_until_multi_round(self, requests) -> List[str]:
        """Generate multi-round responses. Not implemented for InternVLHf."""
        raise NotImplementedError("Multi-round generation is not implemented for InternVLHf.")
