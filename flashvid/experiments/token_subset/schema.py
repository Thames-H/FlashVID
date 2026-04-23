from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def _tensor_to_list(tensor: Optional[torch.Tensor]) -> Optional[list]:
    if tensor is None:
        return None
    return tensor.detach().float().cpu().numpy().tolist()


def _ndarray_to_list(array: Optional[np.ndarray]) -> Optional[list]:
    if array is None:
        return None
    return array.tolist()


@dataclass
class PatchMapping:
    token_coords: List[Tuple[int, int] | None]
    token_source: List[str]
    patch_pixel_size: Tuple[int, int]
    spatial_token_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token_coords": self.token_coords,
            "token_source": self.token_source,
            "patch_pixel_size": self.patch_pixel_size,
            "spatial_token_count": int(self.spatial_token_count),
        }


@dataclass
class SelectionResult:
    indices: torch.Tensor
    scores: torch.Tensor

    def to_dict(self) -> Dict[str, Any]:
        return {
            "indices": _tensor_to_list(self.indices),
            "scores": _tensor_to_list(self.scores),
            "count": int(self.indices.numel()),
        }


@dataclass
class AnswerResult:
    text: str
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "score": float(self.score)}


@dataclass
class TokenSubsetArtifact:
    sample_id: str
    model_name: str
    benchmark: str
    question: str
    ground_truth: str
    image_preview: np.ndarray
    image_size: Tuple[int, int]
    num_visual_tokens: int
    patch_mapping: PatchMapping
    target_layer: int
    queries: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor
    alpha: torch.Tensor
    selections: Dict[str, Dict[str, SelectionResult]]
    answers: Dict[str, Dict[str, AnswerResult]]
    properties: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)

    def to_dict(self, max_tensors: int = 0) -> Dict[str, Any]:
        queries = self.queries
        keys = self.keys
        values = self.values
        alpha = self.alpha

        if max_tensors > 0:
            if queries.numel() > max_tensors:
                queries = queries[:1, :]
            if keys.numel() > max_tensors:
                keys = keys[:1, :]
            if values.numel() > max_tensors:
                values = values[:1, :]
            if alpha.numel() > max_tensors:
                alpha = alpha[:1, :1]

        serializable_selections: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for ratio, method_map in self.selections.items():
            serializable_selections[ratio] = {
                method: result.to_dict()
                for method, result in method_map.items()
            }

        serializable_answers: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for ratio, method_map in self.answers.items():
            serializable_answers[ratio] = {
                method: result.to_dict()
                for method, result in method_map.items()
            }

        return {
            "sample_id": self.sample_id,
            "model_name": self.model_name,
            "benchmark": self.benchmark,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "image_size": self.image_size,
            "image_preview": _ndarray_to_list(self.image_preview),
            "num_visual_tokens": int(self.num_visual_tokens),
            "patch_mapping": self.patch_mapping.to_dict(),
            "target_layer": int(self.target_layer),
            "queries": _tensor_to_list(queries),
            "keys": _tensor_to_list(keys),
            "values": _tensor_to_list(values),
            "alpha": _tensor_to_list(alpha),
            "selections": serializable_selections,
            "answers": serializable_answers,
            "properties": self.properties,
        }
