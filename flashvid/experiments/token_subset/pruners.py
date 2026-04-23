from __future__ import annotations

import torch


class FETPPruner:
    @staticmethod
    def select(Q, K, V, alpha, k, **kwargs):
        Q = Q.to(torch.float32)
        K = K.to(torch.float32)
        V = V.to(torch.float32)
        alpha = alpha.to(torch.float32)

        k = int(max(1, min(k, V.shape[0])))
        if k <= 0 or V.shape[0] == 0:
            return torch.empty(0, dtype=torch.long), torch.empty(0, device=V.device)

        o = alpha @ V
        diff = V.unsqueeze(0) - o.unsqueeze(1)
        diff_sq = diff.pow(2).sum(dim=-1)
        weighted = alpha.pow(2) * diff_sq
        scores = weighted.sum(dim=0).sqrt()
        _, indices = torch.topk(scores, k)
        return indices.sort().values, scores


class AttentionPruner:
    @staticmethod
    def select(Q, K, V, alpha, k, **kwargs):
        alpha = alpha.to(torch.float32)
        k = int(max(1, min(k, alpha.shape[1])))
        scores = alpha.mean(dim=0)
        _, indices = torch.topk(scores, k)
        return indices.sort().values, scores


class MMTokPruner:
    @staticmethod
    def select(Q, K, V, alpha, k, **kwargs):
        V = V.to(torch.float32)
        n_tokens = V.shape[0]
        k = int(max(1, min(k, n_tokens)))
        if n_tokens == 0:
            return torch.empty(0, dtype=torch.long), torch.empty(0, device=V.device)
        if k >= n_tokens:
            scores = torch.ones(n_tokens, device=V.device, dtype=V.dtype)
            return torch.arange(n_tokens, device=V.device), scores

        selected = []
        scores = torch.zeros(n_tokens, device=V.device, dtype=V.dtype)
        centroid = V.mean(dim=0)
        dists = (V - centroid).norm(dim=-1)
        first = dists.argmax().item()
        selected.append(first)
        scores[first] = dists[first]

        min_dists = (V - V[first]).norm(dim=-1)
        for _ in range(k - 1):
            next_idx = min_dists.argmax().item()
            selected.append(next_idx)
            scores[next_idx] = min_dists[next_idx]
            new_dists = (V - V[next_idx]).norm(dim=-1)
            min_dists = torch.min(min_dists, new_dists)

        selected = torch.tensor(selected, device=V.device)
        return selected.sort().values, scores


class TokenPruner:
    @staticmethod
    def prune(method: str, Q, K, V, alpha, k, **kwargs):
        pruners = {
            "fetp": FETPPruner,
            "attention": AttentionPruner,
            "mmtok": MMTokPruner,
        }
        if method not in pruners:
            raise ValueError(f"Unsupported pruning method: {method}")
        return pruners[method].select(Q, K, V, alpha, int(k), **kwargs)


def _as_selection_result(token_indices, scores):
    import torch
    from .schema import SelectionResult

    if token_indices.dtype != torch.long:
        token_indices = token_indices.long()
    return SelectionResult(indices=token_indices, scores=scores)
