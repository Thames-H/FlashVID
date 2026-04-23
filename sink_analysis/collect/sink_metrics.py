from __future__ import annotations

import torch


def identify_sink_tokens(
    alpha: torch.Tensor,
    values: torch.Tensor,
    query_outputs: torch.Tensor,
    attn_percentile: float = 90,
    dev_percentile: float = 30,
):
    mean_attn = alpha.mean(dim=0)
    diff = values.unsqueeze(0) - query_outputs.unsqueeze(1)
    value_dev = diff.norm(dim=-1).mean(dim=0)

    attn_thresh = torch.quantile(mean_attn, attn_percentile / 100.0)
    dev_thresh = torch.quantile(value_dev, dev_percentile / 100.0)
    sink_mask = (mean_attn >= attn_thresh) & (value_dev <= dev_thresh)

    return sink_mask, mean_attn, value_dev

