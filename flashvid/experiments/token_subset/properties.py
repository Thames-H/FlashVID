from __future__ import annotations

import torch
import torch.nn.functional as F


class SubsetPropertyComputer:
    @staticmethod
    def _mask_empty_tensor(t: torch.Tensor, fallback: float = 0.0):
        if t.numel() == 0:
            return torch.tensor(float(fallback), device=t.device)
        return t

    @staticmethod
    def compute_all(Q, K, V, alpha, indices, patch_mapping=None):
        d = Q.shape[-1]
        N = V.shape[0]
        k = int(indices.numel()) if hasattr(indices, "numel") else int(len(indices))
        if k == 0:
            raise ValueError("Subset selection is empty.")

        if indices.dtype != torch.long:
            indices = indices.long()

        V_sel = V[indices]
        K_sel = K[indices]
        alpha_sub = alpha[:, indices]
        props = {}

        # P1-P3: attention quality
        per_query_mass = alpha_sub.sum(dim=-1)
        props["mean_attn_mass"] = float(per_query_mass.mean().item())
        props["min_attn_mass"] = float(per_query_mass.min().item())
        props["std_attn_mass"] = float(per_query_mass.std().item() if per_query_mass.numel() > 1 else 0.0)

        # P4: attention entropy of subset
        mass = alpha_sub / (alpha_sub.sum(dim=-1, keepdim=True).clamp_min_(1e-8))
        entropy = -(mass * mass.log().clamp_min(-30)).sum(dim=-1)
        props["mean_attn_entropy"] = float(entropy.mean().item())

        # P5: effective rank
        V_centered = V_sel - V_sel.mean(dim=0)
        try:
            S = torch.linalg.svdvals(V_centered.float())
            S_norm = S / S.sum().clamp_min_(1e-8)
            valid = S_norm > 1e-10
            if valid.any():
                eff_rank = torch.exp(-(S_norm[valid] * S_norm[valid].log()).sum())
            else:
                eff_rank = torch.tensor(0.0, device=V.device)
        except RuntimeError:
            eff_rank = torch.tensor(0.0, device=V.device)
        props["effective_rank"] = float(eff_rank.item())

        # P6-P7: diversity metrics
        V_normed = F.normalize(V_sel, dim=-1)
        cos_sim = V_normed @ V_normed.T
        mask = ~torch.eye(k, dtype=torch.bool, device=V.device)
        if k > 1:
            props["mean_pairwise_cosine_dist"] = float((1 - cos_sim[mask]).mean().item())
            l2_dists = torch.cdist(V_sel, V_sel)
            props["mean_pairwise_l2_dist"] = float(l2_dists[mask].mean().item())
            if patch_mapping is not None:
                source_counts = {}
                for idx in indices.tolist():
                    source = patch_mapping.token_source[idx] if idx < len(patch_mapping.token_source) else "special"
                    source_counts[source] = source_counts.get(source, 0) + 1
                total = max(int(sum(source_counts.values())), 1)
                source_ratio = {k: v / total for k, v in sorted(source_counts.items())}
                props["source_ratio"] = source_ratio
                props["source_entropy"] = float(-(torch.tensor(list(source_ratio.values()), device=V.device).clamp_min_(1e-8).log() * torch.tensor(list(source_ratio.values()), device=V.device)).sum().item())
            props["pairwise_nonempty"] = 1.0
        else:
            props["mean_pairwise_cosine_dist"] = 0.0
            props["mean_pairwise_l2_dist"] = 0.0
            props["pairwise_nonempty"] = 0.0
            if patch_mapping is not None:
                props["source_ratio"] = {}
                props["source_entropy"] = 0.0

        # P8-P11: function fidelity
        alpha_full = torch.softmax(Q @ K.T / (d ** 0.5), dim=-1)
        o_full = alpha_full @ V
        alpha_sub_full = torch.softmax(Q @ K_sel.T / (d ** 0.5), dim=-1)
        o_sub = alpha_sub_full @ V_sel

        per_query_error = (o_full - o_sub).pow(2).sum(dim=-1)
        props["mean_output_error"] = float(per_query_error.mean().item())
        props["max_output_error"] = float(per_query_error.max().item())
        props["std_output_error"] = float(per_query_error.std().item() if per_query_error.numel() > 1 else 0.0)

        cos_out = F.cosine_similarity(o_full, o_sub, dim=-1)
        props["mean_output_cosine_sim"] = float(cos_out.mean().item())
        props["min_output_cosine_sim"] = float(cos_out.min().item())

        # P12: attention distribution distortion
        alpha_kept_orig = alpha_full[:, indices]
        alpha_kept_renorm = alpha_kept_orig / alpha_kept_orig.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        kl = (alpha_sub_full * (
            alpha_sub_full.log().clamp(min=-30) - alpha_kept_renorm.log().clamp(min=-30)
        )).sum(dim=-1)
        props["mean_kl_divergence"] = float(kl.mean().item())

        # P13: sink retention
        mean_attn = alpha.mean(dim=0)
        value_dev = (V.unsqueeze(0) - o_full.unsqueeze(1)).norm(dim=-1).mean(dim=0)
        sink_mask = (mean_attn >= torch.quantile(mean_attn, 0.9)) & (value_dev <= torch.quantile(value_dev, 0.3))
        sink_set = torch.where(sink_mask)[0]
        sel_set = indices.to(alpha.device)
        if sink_set.numel() == 0:
            props["sink_retention"] = 0.0
        else:
            intersect = torch.isin(sel_set, sink_set).sum().item()
            props["sink_retention"] = float(intersect / max(int(sink_set.numel()), 1))

        return props
