# MMTok: Multimodal Coverage Maximization for Efficient Inference of VLMs
# Paper: https://arxiv.org/abs/2508.18264

import torch
import torch.nn.functional as F
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel


class Qwen3VLVisionModel_MMTok(Qwen3VLVisionModel):
    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs):
        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2],
            grid_thw[:, 0],
        ).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if layer_num in self.deepstack_visual_indexes:
                merger_idx = self.deepstack_visual_indexes.index(layer_num)
                deepstack_feature = self.deepstack_merger_list[merger_idx](
                    hidden_states
                )
                deepstack_feature_lists.append(deepstack_feature)

        selection_features = hidden_states.view(
            -1, self.spatial_merge_unit, hidden_states.shape[-1]
        ).mean(dim=1)
        hidden_states = self.merger(hidden_states)

        return hidden_states, selection_features, deepstack_feature_lists
