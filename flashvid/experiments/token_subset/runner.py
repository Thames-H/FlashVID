from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Sequence, Tuple

import numpy as np
import torch

from .extractor import AttentionExtractor
from .locator import VisualTokenLocator
from .mapping import PatchMappingBuilder
from .pruners import TokenPruner
from .properties import SubsetPropertyComputer
from .schema import AnswerResult, PatchMapping, SelectionResult, TokenSubsetArtifact


def _slice_attention_mask(mask: Any, keep_positions: torch.Tensor) -> Any:
    if mask is None:
        return None
    if isinstance(mask, dict):
        return {k: _slice_attention_mask(v, keep_positions) for k, v in mask.items()}
    if not torch.is_tensor(mask):
        return mask
    if mask.ndim == 2:
        return mask.index_select(1, keep_positions)
    if mask.ndim == 4:
        return mask.index_select(-1, keep_positions).index_select(-2, keep_positions)
    return mask


def _default_score(answer: str, ground_truth: str) -> float:
    if not ground_truth:
        return 0.0
    return float(difflib.SequenceMatcher(None, answer.strip(), ground_truth.strip()).ratio())


class ExperimentPipeline:
    def __init__(
        self,
        model_name: str,
        model,
        processor,
        target_layer: int = 15,
        methods: Sequence[str] | None = None,
        keep_ratios: Sequence[float] | None = None,
        artifact_path: str | None = None,
        downstream_kwargs: Dict[str, Any] | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.model = model
        self.processor = processor
        self.target_layer = target_layer
        self.methods = list(methods or ("fetp", "attention", "mmtok"))
        self.keep_ratios = list(keep_ratios or (0.1, 0.25, 0.5, 0.75))
        self.extractor = AttentionExtractor(model, model_name, target_layer)
        self.artifact_path = Path(artifact_path) if artifact_path else None
        self._artifact_handle = None

        self.eos_token_id = eos_token_id if eos_token_id is not None else getattr(model.config, "eos_token_id", None)
        self.pad_token_id = pad_token_id if pad_token_id is not None else getattr(model.config, "pad_token_id", None)
        if self.pad_token_id is None:
            self.pad_token_id = 0
        if self.eos_token_id is None:
            self.eos_token_id = self.pad_token_id

        self.downstream_kwargs: Dict[str, Any] = {
            "max_new_tokens": 128,
            "temperature": 0.0,
            "top_p": None,
            "top_k": None,
            "num_beams": 1,
            "use_cache": True,
        }
        if downstream_kwargs:
            self.downstream_kwargs.update(downstream_kwargs)

        if self.artifact_path is not None:
            self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
            self._artifact_handle = self.artifact_path.open("a", encoding="utf-8")

    def close(self):
        if self._artifact_handle is not None:
            self._artifact_handle.close()
            self._artifact_handle = None

    @torch.no_grad()
    def process_sample(
        self,
        sample: Dict[str, Any],
        model_inputs: Dict[str, torch.Tensor],
        base_answer: str = "",
        keep_ratios: Sequence[float] | None = None,
        run_downstream: bool = False,
        score_fn: Callable[[str, str], float] | None = None,
    ) -> TokenSubsetArtifact:
        keep_ratios = list(keep_ratios or self.keep_ratios)
        if "input_ids" not in model_inputs:
            raise ValueError("model_inputs must contain input_ids")

        self.extractor.clear()
        forward_kwargs = {
            key: value
            for key, value in model_inputs.items()
            if isinstance(value, (torch.Tensor, list, tuple, dict))
        }
        try:
            _ = self.model(**forward_kwargs, output_attentions=True)
        except TypeError:
            _ = self.model(**forward_kwargs)
        except Exception:
            _ = self.model(**forward_kwargs, output_attentions=False)

        input_ids = model_inputs["input_ids"]
        locator_kwargs = sample.get("locator_kwargs", {})
        vis_start, vis_end, q_start, q_end = VisualTokenLocator.locate(
            self.model_name,
            input_ids=input_ids,
            **locator_kwargs,
        )

        Q, K, V, alpha = self.extractor.get_visual_token_tensors(
            (vis_start, vis_end),
            (q_start, q_end),
        )

        num_visual_tokens = V.shape[0]
        if num_visual_tokens == 0:
            raise ValueError("No visual tokens extracted from this sample.")

        image_size = sample.get("image_size", (0, 0))
        image_preview = sample.get("image_preview", np.zeros((1, 1, 3), dtype=np.uint8))
        if image_size == (0, 0):
            image_size = (
                (image_preview.shape[0], image_preview.shape[1])
                if isinstance(image_preview, np.ndarray) and image_preview.ndim == 3
                else (0, 0)
            )

        patch_mapping = self._build_patch_mapping(sample, num_visual_tokens)

        selections: Dict[str, Dict[str, SelectionResult]] = {}
        answers: Dict[str, Dict[str, AnswerResult]] = {}
        properties: Dict[str, Dict[str, Dict[str, float]]] = {}

        for ratio in keep_ratios:
            ratio_key = str(ratio)
            method_map: Dict[str, SelectionResult] = {}
            ans_map: Dict[str, AnswerResult] = {}
            prop_map: Dict[str, Dict[str, float]] = {}

            for method in self.methods:
                k = max(1, min(num_visual_tokens, int(num_visual_tokens * float(ratio))))
                indices, scores = TokenPruner.prune(method=method, Q=Q, K=K, V=V, alpha=alpha, k=k)
                if indices.numel() > k:
                    indices = indices[:k]

                selected = SelectionResult(
                    indices=indices.detach().cpu(),
                    scores=scores.detach().cpu(),
                )
                method_map[method] = selected

                if run_downstream:
                    down_answer, down_score = self._run_downstream(
                        model_inputs=model_inputs,
                        visual_token_range=(vis_start, vis_end),
                        selected_indices=indices,
                        base_answer=base_answer,
                        ground_truth=str(sample.get("ground_truth", "")),
                        score_fn=score_fn,
                    )
                    ans_map[method] = AnswerResult(text=down_answer, score=float(down_score))
                else:
                    ans_map[method] = AnswerResult(text=base_answer, score=0.0)

                prop_map[method] = SubsetPropertyComputer.compute_all(Q, K, V, alpha, indices, patch_mapping)

            selections[ratio_key] = method_map
            answers[ratio_key] = ans_map
            properties[ratio_key] = prop_map

        artifact = TokenSubsetArtifact(
            sample_id=str(sample.get("sample_id", "")),
            model_name=self.model_name,
            benchmark=str(sample.get("benchmark", "")),
            question=str(sample.get("question", "")),
            ground_truth=str(sample.get("ground_truth", "")),
            image_preview=np.asarray(image_preview, dtype=np.uint8),
            image_size=tuple(int(x) for x in image_size),
            num_visual_tokens=num_visual_tokens,
            patch_mapping=patch_mapping,
            target_layer=self.target_layer,
            queries=Q.cpu(),
            keys=K.cpu(),
            values=V.cpu(),
            alpha=alpha.cpu(),
            selections=selections,
            answers=answers,
            properties=properties,
        )

        if self._artifact_handle is not None:
            self._artifact_handle.write(json.dumps(artifact.to_dict()) + "\n")
            self._artifact_handle.flush()

        return artifact

    def _resolve_language_model_module(self):
        candidates = (
            lambda m: m.language_model,
            lambda m: m.model.language_model,
            lambda m: m.model.model.language_model,
            lambda m: m.model,
        )
        for getter in candidates:
            try:
                module = getter(self.model)
                if module is not None:
                    return module
            except Exception:
                continue
        raise RuntimeError("Cannot resolve language_model module for downstream pruning.")

    def _run_downstream(
        self,
        model_inputs: Dict[str, Any],
        visual_token_range: Tuple[int, int],
        selected_indices: torch.Tensor,
        base_answer: str,
        ground_truth: str,
        score_fn: Callable[[str, str], float] | None,
    ) -> Tuple[str, float]:
        if "input_ids" not in model_inputs or not torch.is_tensor(model_inputs["input_ids"]):
            return base_answer, 0.0

        input_ids = model_inputs["input_ids"]
        seq_len = int(input_ids.shape[1])
        vis_start, vis_end = visual_token_range
        if vis_end <= vis_start:
            return base_answer, 0.0

        selected_indices = selected_indices.detach().long().to(input_ids.device)
        selected_indices = selected_indices[(selected_indices >= 0) & (selected_indices < (vis_end - vis_start))]
        if selected_indices.numel() == 0:
            return base_answer, 0.0

        keep_visual_global = (selected_indices + vis_start).long()
        left = torch.arange(0, vis_start, device=input_ids.device, dtype=torch.long)
        right = torch.arange(vis_end, seq_len, device=input_ids.device, dtype=torch.long)
        keep_positions = torch.cat([left, keep_visual_global, right], dim=0).sort().values

        lm_module = self._resolve_language_model_module()
        hook_state = {"applied": False}

        def _prefill_hook(module, args, kwargs):
            if hook_state["applied"]:
                return args, kwargs

            local_kwargs = dict(kwargs)

            local_seq_len = None
            for key in ("input_ids", "inputs_embeds"):
                value = local_kwargs.get(key)
                if torch.is_tensor(value) and value.ndim >= 2:
                    local_seq_len = int(value.shape[1])
                    break
            if local_seq_len is None:
                return args, kwargs

            if local_seq_len < int(keep_positions.max().item()) + 1:
                return args, kwargs

            kp = keep_positions.to(next(v.device for v in local_kwargs.values() if torch.is_tensor(v)))

            if torch.is_tensor(local_kwargs.get("input_ids")) and local_kwargs["input_ids"].ndim >= 2:
                local_kwargs["input_ids"] = local_kwargs["input_ids"].index_select(1, kp)

            if torch.is_tensor(local_kwargs.get("inputs_embeds")) and local_kwargs["inputs_embeds"].ndim >= 3:
                local_kwargs["inputs_embeds"] = local_kwargs["inputs_embeds"].index_select(1, kp)

            if torch.is_tensor(local_kwargs.get("position_ids")) and local_kwargs["position_ids"].ndim >= 2:
                local_kwargs["position_ids"] = local_kwargs["position_ids"].index_select(1, kp)

            if torch.is_tensor(local_kwargs.get("cache_position")):
                cache_position = local_kwargs["cache_position"]
                if cache_position.ndim == 1 and cache_position.shape[0] >= kp.shape[0]:
                    local_kwargs["cache_position"] = cache_position.index_select(0, kp)

            local_kwargs["attention_mask"] = _slice_attention_mask(local_kwargs.get("attention_mask"), kp)

            hook_state["applied"] = True
            return args, local_kwargs

        handle = lm_module.register_forward_pre_hook(_prefill_hook, with_kwargs=True)
        try:
            temperature = self.downstream_kwargs.get("temperature", 0.0)
            do_sample = bool(self.downstream_kwargs.get("do_sample", False) or (temperature is not None and temperature > 0))
            generated = self.model.generate(
                **model_inputs,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.pad_token_id,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=self.downstream_kwargs.get("top_p", None),
                top_k=self.downstream_kwargs.get("top_k", None),
                num_beams=self.downstream_kwargs.get("num_beams", 1),
                max_new_tokens=self.downstream_kwargs.get("max_new_tokens", 128),
                use_cache=self.downstream_kwargs.get("use_cache", True),
            )
        except Exception:
            handle.remove()
            return base_answer, 0.0
        finally:
            try:
                handle.remove()
            except Exception:
                pass

        if generated is None:
            return base_answer, 0.0

        prompt_len = int(input_ids.shape[1])
        trimmed = [out_ids[prompt_len:] for out_ids in generated]
        try:
            decoded = self.processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except Exception:
            decoded = [""]

        answer = str(decoded[0]) if decoded else ""
        if score_fn is None:
            return answer, _default_score(answer, ground_truth)

        try:
            return answer, float(score_fn(answer, ground_truth))
        except Exception:
            return answer, 0.0

    def _build_patch_mapping(self, sample: Dict[str, Any], num_visual_tokens: int) -> PatchMapping:
        image_size = sample.get("image_size", (0, 0))
        if tuple(image_size) == (0, 0):
            return PatchMapping(token_coords=[], token_source=[], patch_pixel_size=(0, 0), spatial_token_count=0)

        if self.model_name in {"qwen2.5-vl", "qwen3-vl"}:
            grid_thw = sample.get("grid_thw")
            if grid_thw is None:
                side = max(1, int(num_visual_tokens ** 0.5))
                grid_thw = (1, side, max(1, num_visual_tokens // side))
            return PatchMappingBuilder.build(
                "qwen2.5-vl",
                image_size=image_size,
                grid_thw=grid_thw[0] if isinstance(grid_thw, (list, tuple)) and len(grid_thw) > 0 and isinstance(grid_thw[0], (list, tuple)) else grid_thw,
            )

        if self.model_name == "llava-onevision":
            mapping_kwargs = dict(sample.get("patch_mapping_kwargs", {}))
            if "base_grid" not in mapping_kwargs:
                side = max(1, int(num_visual_tokens ** 0.5))
                mapping_kwargs.update(
                    {
                        "base_grid": (side, side),
                        "crop_grid": (0, 0),
                        "crop_layout": (0, 0),
                        "has_newline": False,
                    }
                )
            return PatchMappingBuilder.build(
                "llava-onevision",
                image_size=image_size,
                num_visual_tokens=num_visual_tokens,
                **mapping_kwargs,
            )

        if self.model_name in {"internvl3.5", "internvl3_5", "internvl"}:
            mapping_kwargs = dict(sample.get("patch_mapping_kwargs", {}))
            if "num_patches_list" not in mapping_kwargs:
                mapping_kwargs["num_patches_list"] = [num_visual_tokens]
            return PatchMappingBuilder.build(
                "internvl3.5",
                image_size=image_size,
                num_visual_tokens=num_visual_tokens,
                **mapping_kwargs,
            )

        return PatchMapping(
            token_coords=[],
            token_source=[],
            patch_pixel_size=(0, 0),
            spatial_token_count=0,
        )
