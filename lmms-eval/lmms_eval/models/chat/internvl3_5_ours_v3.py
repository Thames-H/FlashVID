import time
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers.models.internvl.modeling_internvl import (
    InternVLModel,
    InternVLModelOutputWithPast,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.chat.internvl_hf import InternVLHf
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.protocol import ChatMessages

warnings.filterwarnings("ignore")


def _normalize_anchor_layers(
    anchor_layers: Optional[Union[str, Sequence[int]]],
    num_layers: int,
) -> Tuple[int, ...]:
    if num_layers <= 0:
        return tuple()

    if anchor_layers is None:
        quarter = min(num_layers - 1, max(0, num_layers // 4))
        middle = min(num_layers - 1, max(0, num_layers // 2))
        three_quarter = min(num_layers - 1, max(0, (3 * num_layers) // 4))
        anchor_layers = [quarter, middle, three_quarter]
    elif isinstance(anchor_layers, str):
        anchor_layers = [
            int(part.strip())
            for part in anchor_layers.split(",")
            if part.strip()
        ]

    cleaned = sorted(
        {
            int(layer_idx)
            for layer_idx in anchor_layers
            if 0 <= int(layer_idx) < num_layers
        }
    )
    if not cleaned:
        return (min(num_layers - 1, max(0, num_layers // 2)),)
    return tuple(cleaned)


def _resolve_scoring_plan(
    scoring_method: str,
    shallow_layers: int,
    target_layer: int,
    anchor_layers: Optional[Union[str, Sequence[int]]],
    num_layers: int,
) -> Tuple[str, int, int]:
    resolved_target_layer = target_layer
    if scoring_method == "anchor" and anchor_layers is not None:
        parsed_layers = _normalize_anchor_layers(anchor_layers, num_layers)
        if parsed_layers:
            resolved_target_layer = parsed_layers[len(parsed_layers) // 2]

    resolved_scoring_method = scoring_method
    if resolved_scoring_method == "anchor":
        resolved_scoring_method = "shallow"

    if resolved_scoring_method not in {"full", "shallow"}:
        raise ValueError(
            "scoring_method must be one of {'full', 'shallow', 'anchor'}"
        )

    if resolved_scoring_method == "full":
        num_run_layers = num_layers
    else:
        num_run_layers = min(max(1, shallow_layers), num_layers)

    if resolved_target_layer < 0:
        extract_at = num_run_layers + resolved_target_layer
    else:
        extract_at = resolved_target_layer
    extract_at = min(max(extract_at, 0), num_run_layers - 1)

    return resolved_scoring_method, num_run_layers, extract_at


def _get_suffix_text_positions(
    input_ids: torch.Tensor,
    visual_positions: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    non_visual_positions = torch.where(input_ids != image_token_id)[0]
    if visual_positions.numel() == 0:
        return non_visual_positions

    suffix_mask = non_visual_positions > visual_positions.max()
    suffix_positions = non_visual_positions[suffix_mask]
    if suffix_positions.numel() > 0:
        return suffix_positions
    return non_visual_positions


def _truncate_text_positions(
    text_positions: torch.Tensor,
    max_text_tokens: Optional[int],
) -> torch.Tensor:
    if max_text_tokens is None or max_text_tokens <= 0:
        return text_positions
    if text_positions.numel() <= max_text_tokens:
        return text_positions
    return text_positions[-max_text_tokens:]


def _select_score_head_indices(
    num_heads: int,
    max_score_heads: Optional[int],
    device: torch.device,
) -> Optional[torch.Tensor]:
    if max_score_heads is None or max_score_heads <= 0:
        return None
    if max_score_heads >= num_heads:
        return None

    step = max(1, num_heads // max_score_heads)
    head_indices = torch.arange(0, num_heads, step, device=device)
    return head_indices[:max_score_heads]


def _select_candidate_indices(
    coarse_scores: torch.Tensor,
    num_keep: int,
    candidate_ratio: float,
) -> torch.Tensor:
    total = coarse_scores.numel()
    if total == 0:
        return torch.empty(0, dtype=torch.long, device=coarse_scores.device)
    if candidate_ratio >= 1.0:
        return torch.arange(total, device=coarse_scores.device)

    num_candidate = max(num_keep, int(total * candidate_ratio))
    num_candidate = min(total, num_candidate)
    return coarse_scores.topk(num_candidate).indices.sort().values


def _tensor_summary(tensor: torch.Tensor) -> dict:
    tensor = tensor.float()
    return {
        "mean": tensor.mean().item(),
        "std": tensor.std().item() if tensor.numel() > 1 else 0.0,
        "min": tensor.min().item(),
        "max": tensor.max().item(),
        "median": tensor.median().item(),
        "num_tokens": tensor.numel(),
    }


def _summarize_pruning_stats(
    scoring_method: str,
    anchor_layers: Sequence[int],
    num_visual_tokens: int,
    num_keep: int,
    scoring_time_s: float,
    total_pruning_time_s: float,
    candidate_size: Optional[int] = None,
    score_query_tokens: Optional[int] = None,
    score_heads: Optional[int] = None,
) -> Dict[str, Union[str, int, float]]:
    stats: Dict[str, Union[str, int, float]] = {
        "pruning_scoring_method": scoring_method,
        "pruning_anchor_layers": ",".join(str(layer) for layer in anchor_layers),
        "pruning_num_visual_tokens": int(num_visual_tokens),
        "pruning_num_keep": int(num_keep),
        "pruning_scoring_time_ms": float(scoring_time_s * 1000.0),
        "pruning_total_time_ms": float(total_pruning_time_s * 1000.0),
    }
    if candidate_size is not None:
        stats["pruning_candidate_size"] = int(candidate_size)
    if score_query_tokens is not None:
        stats["pruning_score_query_tokens"] = int(score_query_tokens)
    if score_heads is not None:
        stats["pruning_score_heads"] = int(score_heads)
    return stats


def _slice_sequence_tensor(
    tensor: Optional[torch.Tensor],
    keep_indices: torch.Tensor,
) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if tensor.ndim == 1:
        return tensor.index_select(0, keep_indices)
    if tensor.ndim == 2:
        return tensor.index_select(1, keep_indices)
    if tensor.ndim == 3:
        return tensor.index_select(-1, keep_indices)
    if tensor.ndim == 4:
        tensor = tensor.index_select(-1, keep_indices)
        return tensor.index_select(-2, keep_indices)
    return tensor


@torch.no_grad()
def _run_scoring_forward(
    language_model,
    inputs_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    num_run_layers: int,
):
    original_layers = language_model.layers
    original_attn_impl = getattr(language_model.config, "_attn_implementation", None)

    try:
        if original_attn_impl is not None:
            language_model.config._attn_implementation = "eager"
        if num_run_layers < len(original_layers):
            language_model.layers = nn.ModuleList(list(original_layers[:num_run_layers]))

        return language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
    finally:
        language_model.layers = original_layers
        if original_attn_impl is not None:
            language_model.config._attn_implementation = original_attn_impl


@torch.no_grad()
def _extract_value_states(
    language_model,
    hidden_states: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    layer = language_model.layers[layer_idx]
    hidden_normed = layer.input_layernorm(hidden_states)
    attn_module = layer.self_attn
    batch_size, seq_len, _ = hidden_normed.shape
    value_states = attn_module.v_proj(hidden_normed).view(
        batch_size,
        seq_len,
        -1,
        attn_module.head_dim,
    )
    return value_states.transpose(1, 2).float().clone()


@torch.no_grad()
def _compute_fes_scores(
    attention_weights: torch.Tensor,
    value_states: torch.Tensor,
    text_positions: torch.Tensor,
    visual_positions: torch.Tensor,
    use_alpha: bool = True,
    use_deviation: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_visual = visual_positions.numel()
    if num_visual == 0:
        empty = torch.empty(0, device=value_states.device)
        return empty, empty, empty

    if text_positions.numel() == 0:
        scores = torch.ones(num_visual, device=value_states.device)
        return scores, scores.clone(), scores.clone()

    text_to_vis_attn = attention_weights.index_select(2, text_positions)
    text_to_vis_attn = text_to_vis_attn.index_select(3, visual_positions).float()
    denom = text_to_vis_attn.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    alpha = text_to_vis_attn / denom
    alpha_per_text = alpha.mean(dim=1)[0]

    vis_values = value_states[0].index_select(1, visual_positions).float()
    vis_values = vis_values.permute(1, 0, 2).reshape(num_visual, -1)
    pooled_values = alpha_per_text @ vis_values
    diff = vis_values.unsqueeze(0) - pooled_values.unsqueeze(1)
    deviation = diff.norm(dim=-1)

    alpha_mean = alpha_per_text.mean(dim=0)
    deviation_mean = deviation.mean(dim=0)

    if use_alpha and use_deviation:
        scores = ((alpha_per_text**2) * (deviation**2)).mean(dim=0).sqrt()
    elif use_alpha:
        scores = alpha_mean
    elif use_deviation:
        scores = deviation_mean
    else:
        scores = torch.ones(num_visual, device=value_states.device)

    return scores, alpha_mean, deviation_mean


def _make_fetp_forward(
    retention_ratio: float,
    scoring_method: str,
    shallow_layers: int,
    target_layer: int,
    anchor_layers: Optional[Union[str, Sequence[int]]] = None,
    candidate_ratio: float = 1.0,
    max_score_text_tokens: Optional[int] = None,
    max_score_heads: Optional[int] = None,
    use_alpha: bool = True,
    use_deviation: bool = True,
):
    def patched_forward(
        self: InternVLModel,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        vision_feature_layer: int | list[int] | None = None,
        vision_feature_select_strategy: str | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | InternVLModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        is_prefill_stage = past_key_values is None
        if hasattr(past_key_values, "get_seq_length"):
            is_prefill_stage = past_key_values.get_seq_length() == 0
        if is_prefill_stage or not hasattr(self, "_fetp_last_pruning_stats"):
            self._fetp_last_pruning_stats = {}

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_hidden_states = None
        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                return_dict=True,
            ).pooler_output
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            special_image_mask = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_features,
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
            image_hidden_states = image_features

        if (
            pixel_values is not None
            and retention_ratio != 0
            and inputs_embeds.shape[0] == 1
            and input_ids is not None
            and is_prefill_stage
        ):
            visual_positions = torch.where(input_ids[0] == self.config.image_token_id)[0]
            num_visual_tokens = int(visual_positions.numel())

            if num_visual_tokens > 0:
                if retention_ratio < 1.0:
                    num_keep = max(1, int(num_visual_tokens * retention_ratio))
                else:
                    num_keep = max(1, min(int(retention_ratio), num_visual_tokens))

                text_positions = _get_suffix_text_positions(
                    input_ids[0],
                    visual_positions,
                    self.config.image_token_id,
                )
                (
                    resolved_scoring_method,
                    num_run_layers,
                    extract_at,
                ) = _resolve_scoring_plan(
                    scoring_method=scoring_method,
                    shallow_layers=shallow_layers,
                    target_layer=target_layer,
                    anchor_layers=anchor_layers,
                    num_layers=len(self.language_model.layers),
                )

                scoring_start = time.perf_counter()
                try:
                    scoring_outputs = _run_scoring_forward(
                        self.language_model,
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        num_run_layers=num_run_layers,
                    )

                    attention_weights = scoring_outputs.attentions[extract_at]
                    value_states = _extract_value_states(
                        self.language_model,
                        scoring_outputs.hidden_states[extract_at],
                        extract_at,
                    )

                    coarse_text_positions = _truncate_text_positions(
                        text_positions,
                        max_score_text_tokens,
                    )
                    coarse_attention = attention_weights
                    coarse_head_indices = _select_score_head_indices(
                        attention_weights.shape[1],
                        max_score_heads,
                        attention_weights.device,
                    )
                    if coarse_head_indices is not None:
                        coarse_attention = coarse_attention.index_select(1, coarse_head_indices)

                    coarse_scores, _, _ = _compute_fes_scores(
                        coarse_attention,
                        value_states,
                        coarse_text_positions,
                        visual_positions,
                        use_alpha=use_alpha,
                        use_deviation=use_deviation,
                    )

                    candidate_local = _select_candidate_indices(
                        coarse_scores,
                        num_keep=num_keep,
                        candidate_ratio=candidate_ratio,
                    )
                    candidate_visual_positions = visual_positions.index_select(0, candidate_local)

                    if candidate_visual_positions.numel() != visual_positions.numel():
                        final_scores, alpha_mean, deviation_mean = _compute_fes_scores(
                            attention_weights,
                            value_states,
                            text_positions,
                            candidate_visual_positions,
                            use_alpha=use_alpha,
                            use_deviation=use_deviation,
                        )
                    else:
                        final_scores, alpha_mean, deviation_mean = _compute_fes_scores(
                            attention_weights,
                            value_states,
                            text_positions,
                            visual_positions,
                            use_alpha=use_alpha,
                            use_deviation=use_deviation,
                        )

                    keep_within_candidate = final_scores.topk(num_keep).indices.sort().values
                    keep_visual_positions = candidate_visual_positions.index_select(
                        0,
                        keep_within_candidate,
                    )
                    keep_visual_positions = keep_visual_positions.sort().values

                    non_visual_positions = torch.where(input_ids[0] != self.config.image_token_id)[0]
                    keep_indices = torch.cat(
                        [non_visual_positions, keep_visual_positions],
                        dim=0,
                    ).sort().values

                    hidden_size = inputs_embeds.shape[-1]
                    gather_index = keep_indices.view(1, -1, 1).expand(
                        inputs_embeds.shape[0],
                        -1,
                        hidden_size,
                    )
                    inputs_embeds = torch.gather(inputs_embeds, dim=1, index=gather_index)
                    input_ids = input_ids.index_select(1, keep_indices)
                    attention_mask = _slice_sequence_tensor(attention_mask, keep_indices)
                    position_ids = _slice_sequence_tensor(position_ids, keep_indices)

                    if image_hidden_states is not None:
                        kept_visual_mask = input_ids[0] == self.config.image_token_id
                        kept_visual_indices = torch.where(kept_visual_mask)[0]
                        image_hidden_states = inputs_embeds.index_select(1, kept_visual_indices)

                    scoring_time_s = time.perf_counter() - scoring_start
                    score_heads = (
                        attention_weights.shape[1]
                        if coarse_head_indices is None
                        else int(coarse_head_indices.numel())
                    )
                    self._fetp_last_pruning_stats = _summarize_pruning_stats(
                        scoring_method=resolved_scoring_method,
                        anchor_layers=(extract_at,),
                        num_visual_tokens=num_visual_tokens,
                        num_keep=num_keep,
                        scoring_time_s=scoring_time_s,
                        total_pruning_time_s=scoring_time_s,
                        candidate_size=int(candidate_visual_positions.numel()),
                        score_query_tokens=int(coarse_text_positions.numel()),
                        score_heads=score_heads,
                    )
                    self._fetp_last_pruning_stats["pruning_score_summary_mean"] = float(
                        final_scores.float().mean().item()
                    )
                    self._fetp_last_pruning_stats["pruning_alpha_mean"] = float(
                        alpha_mean.float().mean().item()
                    )
                    self._fetp_last_pruning_stats["pruning_deviation_mean"] = float(
                        deviation_mean.float().mean().item()
                    )
                    eval_logger.info(
                        "[InternVL3_5_Ours_V3 / FETP] "
                        f"retention_ratio={retention_ratio}, "
                        f"scoring_method={resolved_scoring_method}, "
                        f"target_layer={extract_at}, "
                        f"candidate_ratio={candidate_ratio}, "
                        f"pruning_ms={self._fetp_last_pruning_stats['pruning_total_time_ms']:.2f}"
                    )
                except Exception as exc:
                    self._fetp_last_pruning_stats = {
                        "pruning_error": str(exc),
                        "pruning_num_visual_tokens": num_visual_tokens,
                    }
                    eval_logger.warning(
                        f"InternVL3.5 FETP pruning skipped due to scoring failure: {exc}"
                    )

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        return InternVLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_hidden_states,
        )

    return patched_forward


@register_model("internvl3_5_ours_v3")
class InternVL3_5_Ours_V3(InternVLHf):
    is_simple = False

    def __init__(
        self,
        pretrained: str = "OpenGVLab/InternVL3_5-8B-HF",
        retention_ratio: float = 0.1,
        scoring_method: str = "full",
        shallow_layers: int = 4,
        target_layer: int = 20,
        anchor_layers: Optional[Union[str, Sequence[int]]] = None,
        anchor_weights: Optional[Union[str, Sequence[float]]] = None,
        candidate_ratio: float = 1.0,
        max_score_text_tokens: Optional[int] = None,
        max_score_heads: Optional[int] = None,
        use_alpha: bool = True,
        use_deviation: bool = True,
        profile_reference_scoring: bool = False,
        reference_scoring_method: str = "shallow",
        **kwargs,
    ) -> None:
        del anchor_weights
        del profile_reference_scoring
        del reference_scoring_method

        super().__init__(pretrained=pretrained, **kwargs)

        self.retention_ratio = retention_ratio
        eval_logger.info(
            "[InternVL3_5_Ours_V3 / FETP] "
            f"retention_ratio={retention_ratio}, "
            f"scoring_method={scoring_method}, "
            f"shallow_layers={shallow_layers}, "
            f"target_layer={target_layer}, "
            f"candidate_ratio={candidate_ratio}, "
            f"max_score_text_tokens={max_score_text_tokens}, "
            f"max_score_heads={max_score_heads}"
        )

        InternVLModel.forward = _make_fetp_forward(
            retention_ratio=retention_ratio,
            scoring_method=scoring_method,
            shallow_layers=shallow_layers,
            target_layer=target_layer,
            anchor_layers=anchor_layers,
            candidate_ratio=candidate_ratio,
            max_score_text_tokens=max_score_text_tokens,
            max_score_heads=max_score_heads,
            use_alpha=use_alpha,
            use_deviation=use_deviation,
        )

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res: List[str] = []

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
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        e2e_latency = 0.0
        total_tokens = 0
        pruning_metric_sums: Dict[str, float] = {}
        pruning_metric_counts: Dict[str, int] = {}
        pruning_metric_last: Dict[str, Union[str, int, float]] = {}

        for chunk in chunks:
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            chat_messages = [doc_to_messages[0](self.task_dict[task][split][ids]) for ids in doc_id]
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
            if self.accelerator.is_main_process and doc_id[0] % 100 == 0:
                eval_logger.debug(f"Prompt for doc ID {doc_id[0]}:\n\n{text}\n")

            if len(videos) == 0:
                videos = None
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

            do_sample = gen_kwargs["temperature"] > 0
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

            pruning_stats = getattr(self.model.model, "_fetp_last_pruning_stats", {})
            for key, value in pruning_stats.items():
                if isinstance(value, bool):
                    pruning_metric_last[key] = value
                elif isinstance(value, (int, float)):
                    pruning_metric_sums[key] = pruning_metric_sums.get(key, 0.0) + float(value)
                    pruning_metric_counts[key] = pruning_metric_counts.get(key, 0) + 1
                else:
                    pruning_metric_last[key] = value

            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            e2e_latency += end_time - start_time
            total_tokens += sum(len(ids) for ids in generated_ids_trimmed)

            if self.accelerator.is_main_process and doc_id[0] % 100 == 0:
                eval_logger.debug(f"Generated text for doc ID {doc_id[0]}:\n\n{answers}\n")

            for answer in answers:
                res.append(answer)
                self.cache_hook.add_partial("generate_until", (text, gen_kwargs), answer)

            pbar.update(1)

        res = re_ords.get_original(res)

        metric_dict = {
            "total_tokens": total_tokens,
            "e2e_latency": e2e_latency,
            "avg_speed": total_tokens / e2e_latency if e2e_latency > 0 else 0,
            "additional_metrics": {
                "rank": self.rank,
                **{
                    key: pruning_metric_sums[key] / max(1, pruning_metric_counts.get(key, 1))
                    for key in pruning_metric_sums
                },
                **pruning_metric_last,
            },
        }
        log_metrics(**metric_dict)

        pbar.close()
        return res
