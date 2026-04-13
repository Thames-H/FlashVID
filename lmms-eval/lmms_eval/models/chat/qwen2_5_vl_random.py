import time
from typing import List, Optional, Union

import torch
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLModel,
    Qwen2_5_VLModelOutputWithPast,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, is_torchdynamo_compiling

try:
    import decord
except ImportError:
    decord = None

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)
from lmms_eval.models.simple.qwen2_5_vl import Qwen2_5_VL as Qwen2_5_VLSimple
from lmms_eval.protocol import ChatMessages

process_vision_info, _has_qwen_vl = optional_import(
    "qwen_vl_utils", "process_vision_info"
)
if not _has_qwen_vl:
    eval_logger.warning(
        "Failed to import qwen_vl_utils; "
        "Please install it via `pip install qwen-vl-utils`"
    )


def _make_random_forward(
    original_forward,
    retention_ratio: float,
):
    """Create a patched forward that randomly drops video tokens."""

    def patched_forward(
        self: Qwen2_5_VLModel,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen2_5_VLModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask, _ = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        n_video_tokens = None
        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(
                pixel_values_videos, video_grid_thw
            )
            video_embeds = torch.cat(video_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            _, video_mask = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                video_features=video_embeds,
            )
            n_video_tokens = video_embeds.shape[0]
            inputs_embeds = inputs_embeds.masked_scatter(
                video_mask, video_embeds
            )

        if position_ids is None:
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (
                    inputs_embeds is not None
                    and inputs_embeds.shape[1] != 1
                )
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (
                    past_key_values is None
                    or past_key_values.get_seq_length() == 0
                )
            )
            if (
                prefill_compiled_stage or prefill_noncompiled_stage
            ) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                position_ids = torch.arange(
                    seq_length, device=inputs_embeds.device
                )
                position_ids = position_ids.view(1, 1, -1).expand(
                    3, batch_size, -1
                )
                if cache_position is not None:
                    delta = (cache_position[0] + self.rope_deltas).to(
                        inputs_embeds.device
                    )
                else:
                    delta = torch.zeros(
                        (batch_size, seq_length),
                        device=inputs_embeds.device,
                    )
                delta = delta.repeat_interleave(
                    batch_size // delta.shape[0], dim=1
                )
                position_ids = position_ids + delta.to(position_ids.device)

        # Random video token selection: only during prefill when we have
        # video tokens and the sequence length is > 1.
        if (
            n_video_tokens is not None
            and position_ids.shape[-1] > 1
            and retention_ratio < 1.0
        ):
            visual_start_index = (
                torch.where(input_ids[0] == self.config.video_token_id)[0][0]
                .item()
            )
            visual_end_index = visual_start_index + n_video_tokens
            num_keep = max(1, int(n_video_tokens * retention_ratio))

            # Randomly select indices to keep among video tokens.
            perm = torch.randperm(
                n_video_tokens, device=inputs_embeds.device
            )[:num_keep]
            keep_visual_local = perm.sort().values

            # Build global keep indices: prefix + kept video + suffix.
            global_indices = torch.arange(
                input_ids.shape[-1], device=inputs_embeds.device
            )
            keep_global_indices = torch.cat(
                [
                    global_indices[:visual_start_index],
                    global_indices[visual_start_index:visual_end_index][
                        keep_visual_local
                    ],
                    global_indices[visual_end_index:],
                ],
                dim=0,
            )

            bsz, _, hidden_size = inputs_embeds.shape
            inputs_embeds = torch.gather(
                inputs_embeds,
                dim=1,
                index=keep_global_indices.view(1, -1, 1).expand(
                    bsz, -1, hidden_size
                ),
            )
            position_ids = position_ids[:, :, keep_global_indices]
            attention_mask = attention_mask[:, keep_global_indices]
            cache_position = cache_position[keep_global_indices]

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        output = Qwen2_5_VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
        return output if return_dict else output.to_tuple()

    return patched_forward


@register_model("qwen2_5_vl_random")
class Qwen2_5_VL_Random(Qwen2_5_VLSimple):
    """Qwen2.5-VL with random video token selection.

    Instead of FlashVID's attention-based compression, this model randomly
    retains ``retention_ratio`` of video tokens before sending them to the
    LLM backbone.
    """

    is_simple = False

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-3B-Instruct",
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
        retention_ratio: float = 0.25,
        **kwargs,
    ) -> None:
        # Pass enable_flashvid=False to base class to skip FlashVID setup.
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
        )

        self.retention_ratio = retention_ratio
        eval_logger.info(
            f"[Qwen2_5_VL_Random] retention_ratio={retention_ratio}"
        )

        # Monkey-patch the model's forward with random token selection.
        original_forward = Qwen2_5_VLModel.forward
        Qwen2_5_VLModel.forward = _make_random_forward(
            original_forward, retention_ratio
        )

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
        e2e_latency = 0
        total_tokens = 0
        for chunk in chunks:
            (
                ctx,
                doc_to_messages,
                all_gen_kwargs,
                doc_id,
                task,
                split,
            ) = zip(*chunk)
            chat_messages = [
                doc_to_messages[idx](self.task_dict[task][split][ids])
                for idx, (ids, task, split) in enumerate(
                    zip(doc_id, task, split)
                )
            ]
            chat_messages: List[ChatMessages] = [
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
            gen_kwargs = all_gen_kwargs[0]

            video_kwargs = {
                "max_pixels": self.max_pixels,
                "min_pixels": self.min_pixels,
            }
            if self.fps is not None:
                video_kwargs["fps"] = self.fps
            else:
                if videos and decord is not None:
                    try:
                        video_path = videos[0]
                        vr = decord.VideoReader(video_path)
                        video_total_frames = len(vr)
                        nframes = min(
                            self.max_num_frames, video_total_frames
                        )
                        nframes = (nframes // 2) * 2
                        nframes = max(2, nframes)
                        video_kwargs["nframes"] = nframes
                    except Exception as e:
                        eval_logger.warning(
                            f"Failed to probe video {videos[0]}: {e}, "
                            "using default nframes"
                        )
                        video_kwargs["nframes"] = self.max_num_frames
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
            image_inputs, video_inputs = process_vision_info(
                batched_messages
            )
            padding_side = "left" if self.batch_size > 1 else "right"
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                padding_side=padding_side,
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

            for ans, context in zip(answers, texts):
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
            pbar.update(1)
        res = re_ords.get_original(res)

        avg_speed = total_tokens / e2e_latency if e2e_latency > 0 else 0
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
