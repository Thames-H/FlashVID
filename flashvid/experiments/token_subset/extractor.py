from __future__ import annotations

import math
from typing import Any, Tuple

import torch


def _resolve_attr(obj: Any, names):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None


def _split_qkv(qkv: torch.Tensor, num_heads: int, head_dim: int):
    if qkv.shape[-1] % (3 * head_dim) == 0:
        q, k, v = qkv.chunk(3, dim=-1)
    elif qkv.shape[-1] % 3 == 0:
        per_proj = qkv.shape[-1] // 3
        q, k, v = torch.split(qkv, per_proj, dim=-1)
    else:
        # Last-resort fallback: assume q/k/v are concatenated with equal dims
        if num_heads <= 0:
            raise ValueError("Cannot infer Q/K/V splits for fused projection.")
        d = qkv.shape[-1] // num_heads // 3
        if d <= 0:
            raise ValueError("Invalid fused qkv projection dimension.")
        q = qkv[..., : num_heads * d]
        k = qkv[..., num_heads * d : 2 * num_heads * d]
        v = qkv[..., 2 * num_heads * d :]
        q, k, v = q.to(qkv.dtype), k.to(qkv.dtype), v.to(qkv.dtype)
    return q, k, v


def _repeat_kv(tensor: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats <= 1:
        return tensor
    return tensor.repeat_interleave(repeats, dim=1)


class AttentionExtractor:
    def __init__(self, model, model_name: str, target_layer: int):
        self.model_name = model_name
        self.target_layer = target_layer
        self.captured = {}
        self._last_hook = None
        self._module = self._get_layer(model, target_layer)
        self._layer_handle = self._module.register_forward_hook(self._hook_fn)

    def _get_layer(self, model, layer_idx):
        candidates = [
            lambda m: m.model.language_model.layers[layer_idx].self_attn,
            lambda m: m.language_model.layers[layer_idx].self_attn,
            lambda m: m.model.layers[layer_idx].self_attn,
            lambda m: m.layers[layer_idx].self_attn,
            lambda m: m.model.model.language_model.layers[layer_idx].self_attn,
            lambda m: m.model.model.layers[layer_idx].self_attn,
        ]
        last_error = None
        for getter in candidates:
            try:
                return getter(model)
            except Exception as err:
                last_error = err
                continue
        raise RuntimeError(f"Cannot resolve attention module for layer {layer_idx}") from last_error

    def _hook_fn(self, module, inputs, output):
        # keep the latest full-step captures for this layer
        if inputs:
            self.captured["inputs"] = inputs[0]
        else:
            self.captured["inputs"] = None
        self.captured["attention_output"] = output

    def clear(self):
        self.captured = {}

    def remove(self):
        if self._layer_handle is not None:
            self._layer_handle.remove()
            self._layer_handle = None

    @torch.no_grad()
    def _compute_qkv(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = self.captured.get("inputs")
        if hidden_states is None:
            raise RuntimeError("No attention inputs captured.")

        if hidden_states.ndim != 3:
            raise ValueError(f"Expected 3D hidden_states, got {hidden_states.shape}")

        attn = self._module
        q_proj = _resolve_attr(attn, ["q_proj", "query", "q_linear"])
        k_proj = _resolve_attr(attn, ["k_proj", "key", "k_linear"])
        v_proj = _resolve_attr(attn, ["v_proj", "value", "v_linear"])
        if q_proj is None or k_proj is None or v_proj is None:
            qkv_proj = _resolve_attr(attn, ["qkv_proj", "c_attn", "in_proj"])
            if qkv_proj is None:
                raise RuntimeError("Attention module misses q/k/v projection modules.")
            qkv = qkv_proj(hidden_states).to(torch.float32)
            num_heads = _resolve_attr(attn, ["num_heads", "num_attention_heads"]) or 1
            head_dim = attn.head_dim if hasattr(attn, "head_dim") else qkv.shape[-1] // (3 * num_heads)
            q, k, v = _split_qkv(qkv, num_heads=num_heads, head_dim=head_dim)
        else:
            q = q_proj(hidden_states).to(torch.float32)
            k = k_proj(hidden_states).to(torch.float32)
            v = v_proj(hidden_states).to(torch.float32)

        num_heads = _resolve_attr(attn, ["num_heads", "num_attention_heads"])
        if num_heads is None:
            num_heads = _resolve_attr(attn, ["num_attention_heads", "num_heads"])
        num_key_value_heads = _resolve_attr(attn, ["num_key_value_heads", "num_kv_heads"])

        if num_heads is None:
            # fallback for small implementations
            num_heads = attn.num_heads if hasattr(attn, "num_heads") else 1
        if num_key_value_heads is None:
            num_key_value_heads = num_heads

        head_dim = attn.head_dim if hasattr(attn, "head_dim") else q.shape[-1] // num_heads
        if head_dim <= 0:
            raise ValueError("Invalid attention head dimension.")

        q_shape = q.shape[:-1] + (num_heads, head_dim)
        k_shape = k.shape[:-1] + (num_key_value_heads, head_dim)
        v_shape = v.shape[:-1] + (num_key_value_heads, head_dim)
        q = q.view(*q_shape).transpose(1, 2)
        k = k.view(*k_shape).transpose(1, 2)
        v = v.view(*v_shape).transpose(1, 2)

        if num_key_value_heads < num_heads:
            repeat_factor = num_heads // num_key_value_heads
            k = _repeat_kv(k, repeat_factor)
            v = _repeat_kv(v, repeat_factor)

        q_states = q.mean(dim=1)
        k_states = k.mean(dim=1)
        v_states = v.mean(dim=1)
        return q_states, k_states, v_states

    def get_visual_token_tensors(
        self,
        visual_token_range,
        query_token_range,
    ):
        Q_full, K_full, V_full = self._compute_qkv()

        q_start, q_end = query_token_range
        v_start, v_end = visual_token_range
        if q_end <= q_start or v_end <= v_start:
            raise ValueError("Invalid query or visual token range.")

        q_start = int(q_start)
        q_end = int(q_end)
        v_start = int(v_start)
        v_end = int(v_end)

        Q = Q_full[:, q_start:q_end, :].squeeze(0)
        K = K_full[:, v_start:v_end, :].squeeze(0)
        V = V_full[:, v_start:v_end, :].squeeze(0)

        scale = math.sqrt(max(Q.size(-1), 1))
        if K.numel() == 0:
            alpha = torch.zeros(Q.shape[0], 0, device=Q.device, dtype=Q.dtype)
        else:
            logits = torch.matmul(Q, K.T) / scale
            alpha = torch.softmax(logits, dim=-1)

        return Q, K, V, alpha
