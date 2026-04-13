"""Functionally Equivalent Token Pruning (FETP) for Qwen2.5-VL.

Implements the theory-driven token compression framework based on the
Functional Equivalence Score (FES).  The optimal importance score is:

    s_i = alpha_i * ||v_i - o||

where alpha_i is the attention weight from the query to visual token i,
and o is the attention-weighted mean of all visual value vectors.

Two modes are supported (controlled by ``scoring_method``):
  - "full":    Run a full LLM forward pass to extract exact alpha_i from
               a specified layer (approach 3 -- theoretical upper bound).
  - "shallow": Run only the first K layers to approximate alpha_i
               (approach 1 -- practical).
"""

import math
import time
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLModel,
    Qwen2_5_VLModelOutputWithPast,
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
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


# ---------------------------------------------------------------------------
# Core scoring: FES-based importance
# ---------------------------------------------------------------------------


@torch.no_grad()
def _compute_fes_scores(
    attn_logits: torch.Tensor,
    value_states: torch.Tensor,
    vis_start: int,
    vis_end: int,
    use_alpha: bool = True,
    use_deviation: bool = True,
) -> torch.Tensor:
    """Compute FES importance scores for visual tokens.

    Following the FES theory, alpha_i is defined with softmax only over
    the visual token set (not all tokens).  We take the raw attention
    logits and re-normalise over visual positions only.

    Args:
        attn_logits: [1, n_heads, seq_len, seq_len] raw attention logits
            (before softmax) from a single LLM layer.
        value_states: [1, n_kv_heads, seq_len, head_dim] value states
            from the same layer (before GQA expansion, raw V projections).
        vis_start: Start index of visual tokens in the sequence.
        vis_end: End index of visual tokens in the sequence.
        use_alpha: Whether to include alpha_i (attention weight) in score.
        use_deviation: Whether to include ||v_i - o|| (value deviation) in score.

    Returns:
        scores: [n_vis] importance score per visual token.
    """
    n_vis = vis_end - vis_start

    # --- Compute alpha_i: query-to-visual attention weights ---
    # Extract logits from text (query) positions to visual (key) positions.
    text_start = vis_end
    # [n_heads, n_text, n_vis]
    text_to_vis_logits = attn_logits[0, :, text_start:, vis_start:vis_end]
    # Re-softmax ONLY over the visual token dimension (dim=-1),
    # so alpha sums to 1 over visual tokens — matching FES definition.
    text_to_vis_alpha = F.softmax(
        text_to_vis_logits.float(), dim=-1
    )  # [n_heads, n_text, n_vis]
    # Average over heads and text positions -> [n_vis]
    alpha = text_to_vis_alpha.mean(dim=0).mean(dim=0)

    # --- Compute ||v_i - o|| ---
    # value_states may have fewer heads (GQA). We average over kv_heads.
    # [n_kv_heads, n_vis, head_dim]
    vis_values = value_states[0, :, vis_start:vis_end, :].float()
    # Flatten to [n_vis, n_kv_heads * head_dim]
    vis_values = vis_values.permute(1, 0, 2).reshape(n_vis, -1)

    # Weighted mean o = sum(alpha_i * v_i)
    # alpha: [n_vis], vis_values: [n_vis, D]
    o = (alpha.unsqueeze(-1) * vis_values).sum(dim=0)  # [D]

    # ||v_i - o|| for each token
    deviation = (vis_values - o.unsqueeze(0)).norm(dim=-1)  # [n_vis]

    # s_i = alpha_i * ||v_i - o|| (with ablation switches)
    if use_alpha and use_deviation:
        scores = alpha * deviation
    elif use_alpha:
        scores = alpha
    elif use_deviation:
        scores = deviation
    else:
        # Both disabled: uniform scores (no pruning preference).
        scores = torch.ones(n_vis, device=alpha.device)
    return scores


# ---------------------------------------------------------------------------
# Forward extraction: run LLM layers and extract attn + values
# ---------------------------------------------------------------------------


@torch.no_grad()
def _forward_extract(
    language_model,
    inputs_embeds: torch.Tensor,
    position_ids: torch.Tensor,
    cache_position: torch.Tensor,
    num_layers: int,
    attn_layer: int,
) -> tuple:
    """Run LLM layers and extract attention weights + values from one layer.

    For the target layer, we manually compute eager attention (not flash)
    to obtain the full attention weight matrix. For all other layers, we
    use the standard layer forward (which may use flash attention).

    Args:
        language_model: The LLM backbone (``model.language_model``).
        inputs_embeds: [1, seq_len, d] input embeddings.
        position_ids: [3or4, 1, seq_len] position ids.
        cache_position: [seq_len] cache positions.
        num_layers: How many layers to run (all layers for "full",
            K layers for "shallow").
        attn_layer: Which layer to extract attention from (0-indexed).

    Returns:
        (attn_logits, value_states):
            attn_logits: [1, n_heads, seq_len, seq_len] float32
                Raw attention logits BEFORE softmax, so that the caller
                can re-normalise over any subset of positions.
            value_states: [1, n_kv_heads, seq_len, head_dim] float32
    """
    hidden = inputs_embeds
    seq_len = hidden.shape[1]

    # Compute RoPE position embeddings.
    if position_ids.shape[0] == 4:
        rope_pos = position_ids[1:]
    else:
        rope_pos = position_ids
    position_embeddings = language_model.rotary_emb(hidden, rope_pos)

    attn_logits_out = None
    value_states_out = None

    for layer_idx in range(num_layers):
        layer = language_model.layers[layer_idx]

        if layer_idx == attn_layer:
            # --- Manual eager attention to extract weights + values ---
            residual = hidden
            hidden_normed = layer.input_layernorm(hidden)

            attn_module = layer.self_attn
            bsz, q_len, _ = hidden_normed.size()

            # Q/K/V projections.
            query_states = attn_module.q_proj(hidden_normed)
            key_states = attn_module.k_proj(hidden_normed)
            value_states = attn_module.v_proj(hidden_normed)

            query_states = query_states.view(
                bsz, q_len, -1, attn_module.head_dim
            ).transpose(1, 2)
            key_states = key_states.view(
                bsz, q_len, -1, attn_module.head_dim
            ).transpose(1, 2)
            value_states = value_states.view(
                bsz, q_len, -1, attn_module.head_dim
            ).transpose(1, 2)

            # Apply M-RoPE (multimodal rotary position embeddings).
            cos, sin = position_embeddings
            mrope_section = attn_module.rope_scaling["mrope_section"]
            query_states, key_states = apply_multimodal_rotary_pos_emb(
                query_states, key_states, cos, sin, mrope_section,
            )

            # Save raw value states (before GQA expansion).
            value_states_out = value_states.float().clone()

            # Expand KV heads for GQA to compute attention.
            key_states_expanded = repeat_kv(
                key_states, attn_module.num_key_value_groups
            )
            value_states_expanded = repeat_kv(
                value_states, attn_module.num_key_value_groups
            )

            # Compute attention logits (eager, no flash).
            attn_logits = torch.matmul(
                query_states, key_states_expanded.transpose(2, 3)
            ) * attn_module.scaling

            # Apply causal mask.
            causal_mask = torch.triu(
                torch.ones(
                    q_len, q_len,
                    device=hidden.device, dtype=torch.bool,
                ),
                diagonal=1,
            )
            attn_logits = attn_logits.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )

            # Save raw logits (before softmax) for FES scoring.
            attn_logits_out = attn_logits.float().clone()

            # Softmax for computing attention output (to continue forward).
            attn_w = F.softmax(attn_logits, dim=-1, dtype=torch.float32)

            # Compute attention output to continue the forward pass.
            attn_output = torch.matmul(
                attn_w.to(value_states_expanded.dtype),
                value_states_expanded,
            )
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, -1)
            attn_output = attn_module.o_proj(attn_output)

            hidden = residual + attn_output

            # Post-attention MLP.
            residual = hidden
            hidden = residual + layer.mlp(
                layer.post_attention_layernorm(hidden)
            )
        else:
            # --- Standard layer forward (may use flash attention) ---
            layer_out = layer(
                hidden,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                output_attentions=False,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden = layer_out[0]

    return attn_logits_out, value_states_out


# ---------------------------------------------------------------------------
# Core patched forward
# ---------------------------------------------------------------------------


def _make_fetp_forward(
    original_forward,
    retention_ratio: float,
    scoring_method: str,
    shallow_layers: int,
    target_layer: int,
    use_alpha: bool = True,
    use_deviation: bool = True,
):
    """Create a patched Qwen2_5_VLModel.forward with FES-based pruning."""

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

        n_image_tokens = None
        if pixel_values is not None:
            image_embeds = self.get_image_features(
                pixel_values, image_grid_thw
            )
            image_embeds = torch.cat(image_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask, _ = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_embeds,
            )
            n_image_tokens = image_embeds.shape[0]
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask, image_embeds
            )

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

        # ---------------------------------------------------------------
        # FETP: Functionally Equivalent Token Pruning
        # Applied during prefill with visual tokens (image or video),
        # batch_size=1.
        # ---------------------------------------------------------------
        n_visual_tokens = n_video_tokens or n_image_tokens
        if n_video_tokens is not None:
            visual_token_id = self.config.video_token_id
        elif n_image_tokens is not None:
            visual_token_id = self.config.image_token_id
        else:
            visual_token_id = None

        if (
            n_visual_tokens is not None
            and position_ids.shape[-1] > 1
            and retention_ratio != 0
            and inputs_embeds.shape[0] == 1
        ):
            device = inputs_embeds.device
            bsz, total_seq, hidden_size = inputs_embeds.shape

            # -- Locate visual tokens in the sequence --
            visual_positions = torch.where(
                input_ids[0] == visual_token_id
            )[0]
            visual_start_index = visual_positions[0].item()
            visual_end_index = visual_start_index + n_visual_tokens
            # retention_ratio < 1: proportion; >= 1: absolute token count.
            if retention_ratio < 1.0:
                num_keep = max(1, int(n_visual_tokens * retention_ratio))
            else:
                num_keep = max(1, min(int(retention_ratio), n_visual_tokens))

            # -- Determine how many layers to run and which to extract --
            if scoring_method == "full":
                n_run = len(self.language_model.layers)
                extract_at = target_layer
            else:  # "shallow"
                n_run = shallow_layers
                extract_at = min(target_layer, shallow_layers - 1)

            # -- Extract attention logits and value states --
            attn_logits, val_s = _forward_extract(
                self.language_model,
                inputs_embeds,
                position_ids,
                cache_position,
                num_layers=n_run,
                attn_layer=extract_at,
            )

            if attn_logits is not None and val_s is not None:
                # -- Compute FES importance scores --
                scores = _compute_fes_scores(
                    attn_logits, val_s,
                    visual_start_index, visual_end_index,
                    use_alpha=use_alpha,
                    use_deviation=use_deviation,
                )  # [n_visual_tokens]

                # -- Select top-k tokens by score --
                _, top_indices = scores.topk(num_keep)
                keep_visual_local = top_indices.sort().values
            else:
                # Fallback: uniform selection.
                eval_logger.warning(
                    "FETP: attention extraction failed, "
                    "falling back to uniform selection."
                )
                step = n_visual_tokens / num_keep
                keep_visual_local = torch.arange(
                    num_keep, device=device
                ).float().mul(step).long()

            # -- Build global keep indices: prefix + kept visual + suffix --
            global_indices = torch.arange(total_seq, device=device)
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

        # ---------------------------------------------------------------
        # Language model forward (with pruned tokens)
        # ---------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------


@register_model("qwen2_5_vl_ours_v2")
class Qwen2_5_VL_Ours_V2(Qwen2_5_VLSimple):
    """Qwen2.5-VL with Functionally Equivalent Token Pruning (FETP).

    Uses the FES-derived importance score s_i = alpha_i * ||v_i - o|| to
    select the most important visual tokens before LLM inference.
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
        # FETP parameters.
        retention_ratio: float = 0.25,
        scoring_method: str = "full",
        shallow_layers: int = 4,
        target_layer: int = 15,
        use_alpha: bool = True,
        use_deviation: bool = True,
        **kwargs,
    ) -> None:
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
            f"[Qwen2_5_VL_Ours_V2 / FETP] "
            f"retention_ratio={retention_ratio}, "
            f"scoring_method={scoring_method}, "
            f"shallow_layers={shallow_layers}, "
            f"target_layer={target_layer}, "
            f"use_alpha={use_alpha}, "
            f"use_deviation={use_deviation}"
        )

        # Monkey-patch the model's forward.
        original_forward = Qwen2_5_VLModel.forward
        Qwen2_5_VLModel.forward = _make_fetp_forward(
            original_forward,
            retention_ratio=retention_ratio,
            scoring_method=scoring_method,
            shallow_layers=shallow_layers,
            target_layer=target_layer,
            use_alpha=use_alpha,
            use_deviation=use_deviation,
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
            total_tokens += sum(
                len(ids) for ids in generated_ids_trimmed
            )

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
