"""Functionally Equivalent Token Pruning (FETP) for LLaVA-1.5 (HF format).

Applies the same FES-based scoring (s_i = alpha_i * ||v_i - o||) to
LLaVA-1.5's image tokens.  The LLM backbone is LLaMA/Vicuna, which uses
standard RoPE (not M-RoPE like Qwen2.5-VL).

Two modes are supported (controlled by ``scoring_method``):
  - "full":    Run all LLM layers, extract attention from target_layer.
  - "shallow": Run first K layers, extract attention from target_layer.
"""

import time
import warnings
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
)

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.chat.llava_hf import _prepare_llava_media_inputs
from lmms_eval.models.simple.llava_hf import LlavaHf as LlavaHfSimple
from lmms_eval.protocol import ChatMessages

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Core scoring: FES-based importance (same logic as Qwen2.5-VL version)
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

    Args:
        attn_logits: [1, n_heads, seq_len, seq_len] raw attention logits
            (before softmax) from a single LLM layer.
        value_states: [1, n_kv_heads, seq_len, head_dim] value states.
        vis_start: Start index of visual tokens in the sequence.
        vis_end: End index of visual tokens in the sequence.
        use_alpha: Whether to include alpha_i (attention weight) in score.
        use_deviation: Whether to include ||v_i - o|| (value deviation) in score.

    Returns:
        scores: [n_vis] importance score per visual token.
    """
    n_vis = vis_end - vis_start

    # alpha_i: text -> visual attention, softmax only over visual tokens.
    text_start = vis_end
    text_to_vis_logits = attn_logits[0, :, text_start:, vis_start:vis_end]
    text_to_vis_alpha = F.softmax(
        text_to_vis_logits.float(), dim=-1
    )  # [n_heads, n_text, n_vis]
    alpha = text_to_vis_alpha.mean(dim=0).mean(dim=0)  # [n_vis]

    # ||v_i - o||
    vis_values = value_states[0, :, vis_start:vis_end, :].float()
    vis_values = vis_values.permute(1, 0, 2).reshape(n_vis, -1)
    o = (alpha.unsqueeze(-1) * vis_values).sum(dim=0)
    deviation = (vis_values - o.unsqueeze(0)).norm(dim=-1)

    if use_alpha and use_deviation:
        scores = alpha * deviation
    elif use_alpha:
        scores = alpha
    elif use_deviation:
        scores = deviation
    else:
        scores = torch.ones(n_vis, device=alpha.device)
    return scores


# ---------------------------------------------------------------------------
# Forward extraction: run LLM layers and extract attn logits + values
# ---------------------------------------------------------------------------


@torch.no_grad()
def _forward_extract(
    language_model,
    inputs_embeds: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    cache_position: torch.Tensor,
    num_layers: int,
    attn_layer: int,
) -> tuple:
    """Run LLM layers and extract attention logits + values from one layer.

    For the target layer, manually compute eager attention (no flash) to
    get the full attention weight matrix.  For other layers, use standard
    layer forward.

    Args:
        language_model: The LLaMA model (``self.model.language_model``).
        inputs_embeds: [1, seq_len, d].
        position_ids: [1, seq_len].
        attention_mask: [1, seq_len] or None.
        cache_position: [seq_len] or None.
        num_layers: How many layers to run.
        attn_layer: Which layer to extract from (0-indexed).

    Returns:
        (attn_logits, value_states) from the target layer.
    """
    hidden = inputs_embeds
    seq_len = hidden.shape[1]

    # Compute RoPE position embeddings.
    position_embeddings = language_model.rotary_emb(hidden, position_ids)

    attn_logits_out = None
    value_states_out = None

    for layer_idx in range(num_layers):
        layer = language_model.layers[layer_idx]

        if layer_idx == attn_layer:
            # --- Manual eager attention ---
            residual = hidden
            hidden_normed = layer.input_layernorm(hidden)

            attn_module = layer.self_attn
            bsz, q_len, _ = hidden_normed.size()
            hidden_shape = (bsz, q_len, -1, attn_module.head_dim)

            query_states = attn_module.q_proj(hidden_normed).view(
                hidden_shape
            ).transpose(1, 2)
            key_states = attn_module.k_proj(hidden_normed).view(
                hidden_shape
            ).transpose(1, 2)
            value_states = attn_module.v_proj(hidden_normed).view(
                hidden_shape
            ).transpose(1, 2)

            # Apply standard RoPE.
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin,
            )

            # Save raw value states (before GQA expansion).
            value_states_out = value_states.float().clone()

            # Expand KV heads for GQA.
            key_states_expanded = repeat_kv(
                key_states, attn_module.num_key_value_groups
            )
            value_states_expanded = repeat_kv(
                value_states, attn_module.num_key_value_groups
            )

            # Compute attention logits.
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

            # Save raw logits.
            attn_logits_out = attn_logits.float().clone()

            # Softmax to continue forward pass.
            attn_w = F.softmax(attn_logits, dim=-1, dtype=torch.float32)
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
            # --- Standard layer forward ---
            hidden = layer(
                hidden,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            # LlamaDecoderLayer returns a tensor directly (not a tuple)
            # in newer transformers, but some versions return tuple.
            if isinstance(hidden, tuple):
                hidden = hidden[0]

    return attn_logits_out, value_states_out


# ---------------------------------------------------------------------------
# Patched LlavaModel.forward
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
    """Create a patched LlavaModel.forward with FES-based token pruning."""

    def patched_forward(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        vision_feature_layer=None,
        vision_feature_select_strategy=None,
        cache_position=None,
        image_sizes=None,
        **kwargs,
    ):
        vision_feature_layer = (
            vision_feature_layer
            if vision_feature_layer is not None
            else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        n_image_tokens = None
        image_features = None
        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                image_sizes=image_sizes,
            )
            image_features = torch.cat(image_features, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            special_image_mask = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_features,
            )
            n_image_tokens = special_image_mask.sum().item() // inputs_embeds.shape[-1]
            inputs_embeds = inputs_embeds.masked_scatter(
                special_image_mask, image_features
            )

        # ---------------------------------------------------------------
        # FETP: Functionally Equivalent Token Pruning
        # ---------------------------------------------------------------
        if (
            n_image_tokens is not None
            and n_image_tokens > 0
            and retention_ratio != 0
            and inputs_embeds.shape[0] == 1
            and (past_key_values is None or
                 (hasattr(past_key_values, 'get_seq_length') and
                  past_key_values.get_seq_length() == 0))
        ):
            device = inputs_embeds.device
            bsz, total_seq, hidden_size = inputs_embeds.shape
            # retention_ratio < 1: proportion; >= 1: absolute token count.
            if retention_ratio < 1.0:
                num_keep = max(1, int(n_image_tokens * retention_ratio))
            else:
                num_keep = max(1, min(int(retention_ratio), n_image_tokens))

            # Locate image tokens.
            image_token_id = self.config.image_token_id
            visual_positions = torch.where(
                input_ids[0] == image_token_id
            )[0]
            visual_start_index = visual_positions[0].item()
            visual_end_index = visual_start_index + n_image_tokens

            # Determine layers to run.
            if scoring_method == "full":
                n_run = len(self.language_model.layers)
                extract_at = target_layer
            else:
                n_run = shallow_layers
                extract_at = min(target_layer, shallow_layers - 1)

            # Build position_ids if not provided.
            if position_ids is None:
                pos_ids = torch.arange(
                    total_seq, device=device
                ).unsqueeze(0)
            else:
                pos_ids = position_ids

            if cache_position is None:
                c_pos = torch.arange(total_seq, device=device)
            else:
                c_pos = cache_position

            # Extract attention logits and value states.
            attn_logits, val_s = _forward_extract(
                self.language_model,
                inputs_embeds,
                pos_ids,
                attention_mask,
                c_pos,
                num_layers=n_run,
                attn_layer=extract_at,
            )

            if attn_logits is not None and val_s is not None:
                scores = _compute_fes_scores(
                    attn_logits, val_s,
                    visual_start_index, visual_end_index,
                    use_alpha=use_alpha,
                    use_deviation=use_deviation,
                )
                _, top_indices = scores.topk(num_keep)
                keep_visual_local = top_indices.sort().values
            else:
                eval_logger.warning(
                    "FETP: extraction failed, uniform fallback."
                )
                step = n_image_tokens / num_keep
                keep_visual_local = torch.arange(
                    num_keep, device=device
                ).float().mul(step).long()

            # Build global keep indices.
            global_indices = torch.arange(total_seq, device=device)
            keep_global = torch.cat(
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
                index=keep_global.view(1, -1, 1).expand(
                    bsz, -1, hidden_size
                ),
            )
            if attention_mask is not None:
                attention_mask = attention_mask[:, keep_global]
            if position_ids is not None:
                position_ids = position_ids[:, keep_global]
            if cache_position is not None:
                cache_position = cache_position[keep_global]

        # ---------------------------------------------------------------
        # Language model forward (with pruned tokens)
        # ---------------------------------------------------------------
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        from transformers.models.llava.modeling_llava import (
            LlavaModelOutputWithPast,
        )
        return LlavaModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=(
                image_features if pixel_values is not None else None
            ),
        )

    return patched_forward


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"


@register_model("llava_hf_ours_v2")
class LlavaHfOursV2(LlavaHfSimple):
    """LLaVA-1.5 (HF) with Functionally Equivalent Token Pruning (FETP).

    Uses the FES-derived importance score s_i = alpha_i * ||v_i - o|| to
    prune image tokens before LLM inference.
    """

    is_simple = False

    def __init__(
        self,
        pretrained: str = "llava-hf/llava-1.5-7b-hf",
        revision: str = "main",
        device: str = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: int = 1,
        trust_remote_code: Optional[bool] = False,
        attn_implementation: Optional[str] = None,
        device_map: str = "",
        chat_template: Optional[str] = None,
        use_cache: bool = True,
        max_frames_num: Optional[int] = 32,
        # FETP parameters.
        retention_ratio: float = 0.25,
        scoring_method: str = "full",
        shallow_layers: int = 4,
        target_layer: int = 15,
        use_alpha: bool = True,
        use_deviation: bool = True,
        **kwargs,
    ) -> None:
        # Remove FETP params from kwargs before passing to parent.
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
        )

        self.retention_ratio = retention_ratio
        eval_logger.info(
            f"[LlavaHfOursV2 / FETP] "
            f"retention_ratio={retention_ratio}, "
            f"scoring_method={scoring_method}, "
            f"shallow_layers={shallow_layers}, "
            f"target_layer={target_layer}, "
            f"use_alpha={use_alpha}, "
            f"use_deviation={use_deviation}"
        )

        # Monkey-patch LlavaModel.forward.
        from transformers.models.llava.modeling_llava import LlavaModel
        LlavaModel.forward = _make_fetp_forward(
            LlavaModel.forward,
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
            return x[2], x[2]

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
            task = task[0]
            split = split[0]
            chat_messages = [
                doc_to_messages[0](self.task_dict[task][split][ids])
                for ids in doc_id
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
            assert self.batch_size_per_gpu == 1

            # Apply chat template.
            messages = chat_messages[0].model_dump()["messages"]
            text = self._image_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

            visuals, videos, image_sizes = _prepare_llava_media_inputs(
                visuals,
                videos,
            )
            inputs = self._image_processor(
                images=visuals, videos=videos, text=text,
                return_tensors="pt",
            ).to(self._device, self.model.dtype)

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
            do_sample = gen_kwargs["temperature"] > 0

            try:
                start_time = time.time()
                cont = self.model.generate(
                    **inputs,
                    do_sample=do_sample,
                    temperature=(
                        gen_kwargs["temperature"] if do_sample else None
                    ),
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                    pad_token_id=self.eot_token_id,
                    eos_token_id=self.eot_token_id,
                )
                end_time = time.time()
                cont = cont[:, inputs["input_ids"].shape[-1]:]
                e2e_latency += end_time - start_time
                total_tokens += (
                    cont.shape[-1] if len(cont.shape) > 1 else len(cont)
                )
            except Exception as e:
                eval_logger.error(f"Error {e} in generating")
                cont = ""
                e2e_latency += 0
                total_tokens += 0

            text_outputs = (
                self.tokenizer.batch_decode(
                    cont, skip_special_tokens=True
                )[0]
                if cont != "" else ""
            )

            res.append(text_outputs)
            self.cache_hook.add_partial(
                "generate_until", (text, gen_kwargs), text_outputs,
            )
            pbar.update(1)

        res = re_ords.get_original(res)

        metric_dict = {
            "total_tokens": total_tokens,
            "e2e_latency": e2e_latency,
            "avg_speed": (
                total_tokens / e2e_latency if e2e_latency > 0 else 0
            ),
            "additional_metrics": {"rank": self.rank},
        }
        log_metrics(**metric_dict)

        pbar.close()
        return res
