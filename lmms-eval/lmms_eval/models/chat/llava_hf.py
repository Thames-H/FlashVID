import time
import warnings
from typing import List

from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.protocol import ChatMessages

warnings.filterwarnings("ignore")

from loguru import logger as eval_logger

from lmms_eval.models.simple.llava_hf import LlavaHf as LlavaHfSimple

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

# Default chat for llava-hf/llava-1.5 models: https://huggingface.co/collections/llava-hf/llava-15-65f762d5b6941db5c2ba07e0
VICUNA_CHAT_TEMPLATE = "{% for message in messages %}{% if loop.index0 == 0 %}A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {{ message['content'] }} {% elif message['role'] == 'user' %}USER: {{ message['content'] }} {% else %} ASSISTANT: {{ message['content'] }}{{ eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"


def _prepare_llava_media_inputs(visuals, videos):
    normalized_visuals = visuals if visuals else None
    normalized_videos = videos if videos else None
    image_sizes = [visual.size for visual in visuals] if visuals else []
    return normalized_visuals, normalized_videos, image_sizes


def _resolve_llava_video_size(model_config):
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


def _build_llava_processor_kwargs(model_config, max_frames_num):
    images_kwargs = {}
    videos_kwargs = {}
    if getattr(model_config, "model_type", None) != "llava_onevision":
        return images_kwargs, videos_kwargs

    if max_frames_num is not None:
        videos_kwargs["num_frames"] = int(max_frames_num)
        videos_kwargs["do_sample_frames"] = True

    video_size = _resolve_llava_video_size(model_config)
    if video_size is not None:
        videos_kwargs["size"] = video_size

    return images_kwargs, videos_kwargs


@register_model("llava_hf_chat")
class LlavaHf(LlavaHfSimple):
    is_simple = False

    def _prepare_chat_media_inputs(self, visuals, videos):
        prepared_visuals = visuals if visuals else None
        prepared_videos = None
        if videos:
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

            # Apply chat template
            messages = chat_messages[0].model_dump()["messages"]
            text = self._image_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if self.accelerator.is_main_process and doc_id[0] % 100 == 0:
                eval_logger.debug(f"Prompt for doc ID {doc_id[0]}:\n\n{text}\n")

            visuals, videos, image_sizes = _prepare_llava_media_inputs(
                visuals,
                videos,
            )
            try:
                visuals, videos = self._prepare_chat_media_inputs(visuals, videos)
            except Exception as e:
                eval_logger.error(f"Error {e} when loading video: {videos}")
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
