# FlashVID 方法执行逻辑详解（以 Qwen2.5-VL 为例）

## 总览

FlashVID 是一个 **training-free**（无需训练）的视频 token 压缩框架，通过 monkey-patch 的方式替换模型的 forward 方法，在推理过程中自动压缩视觉 token，从而加速 Video LLM 的推理。

整体流程分为 **三个阶段**：

```
视频输入
  │
  ▼
┌─────────────────────────────────────────────┐
│ Stage 1: Vision Encoder (ViT)               │
│   提取视频帧特征 + CLS attention weights    │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│ Stage 2: Vision-Side Compression            │
│   ├─ DySeg: 动态视频分段                    │
│   ├─ ADTS: 注意力+多样性 token 选择         │
│   └─ TSTM: 树状时空 token 合并 + DPC-kNN    │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│ Stage 3: Inner-LLM Pruning (FastV)          │
│   在 LLM 第 K 层根据 attention 剪枝视觉token│
└─────────────────────┬───────────────────────┘
                      │
                      ▼
                  文本输出
```

---

## Stage 1: Vision Encoder — 提取特征与 CLS Attention

**入口**: `Qwen2_5_VLModel_get_video_features` → `Qwen2_5_VisionTransformerPretrainedModel_forward`

**代码位置**: `flashvid/modeling_qwen2_5_vl.py:590-608` 和 `192-265`

### 输入
- `pixel_values_videos`: 视频像素值，shape `(total_patches, channels)`
- `video_grid_thw`: 视频的 (时间帧数 T, 高度 H, 宽度 W)，shape `(1, 3)`

### 处理过程
1. **Patch Embedding**: 将像素值通过 `patch_embed`（Conv3D）转成 patch 特征
2. **Rotary Position Embedding**: 计算视觉 token 的旋转位置编码
3. **Window Attention + Full Attention**: 交替通过 ViT 的 block 层
4. **最后一层提取 CLS Attention**: 在 ViT 的最后一个 block 中（`return_logits=True`），手动计算 attention weights：
   ```python
   # Qwen2_5_VLVisionAttention_forward (line 574-584)
   # 对最后一层，手动计算 Q*K^T 得到 attention 矩阵
   attn_weights = torch.matmul(q, k.transpose(-1, -2)) / sqrt(head_dim)
   attn_weights = softmax(attn_weights)
   attn_weights = attn_weights.mean(1).mean(1)  # 对 head 和 query 维度取平均
   ```
5. **Spatial Merge**: 通过 `merger` 将相邻 spatial token 合并（2×2 → 1）
6. **CLS Attention 后处理**: 将 attention weights reshape 成 `(num_frames, num_tokens_per_frame)` 的形式

### 输出
- `video_embeds`: 视频特征，shape `(num_frames × num_tokens_per_frame, hidden_dim)`
- `cls_attention`: 每帧每个 token 的重要性分数，shape `(num_frames, num_tokens_per_frame)`

> **CLS Attention 的含义**: 它代表 ViT 最后一层中，每个 token 被其他 token 关注的平均程度。分数越高，说明该 token 包含的视觉信息越重要。

---

## Stage 2: Vision-Side Compression — 视觉侧压缩

**入口**: `Qwen2_5_VLModel_forward` 中调用 `flashvid_compression`

**代码位置**: `flashvid/utils.py:23-78` 和 `flashvid/modeling_qwen2_5_vl.py:461-505`

### 输入
- `video_features`: 视频特征，reshape 成 `(num_frames, num_tokens_per_frame, hidden_dim)`
- `cls_attention`: CLS attention，shape `(num_frames, num_tokens_per_frame)`
- `flashvid_config`: 配置参数

### Token 预算计算
```python
token_budget = ceil(num_tokens_per_frame × retention_ratio × expansion)
#            = ceil(num_tokens_per_frame × 0.10 × 1.25)
# 例如: ceil(256 × 0.10 × 1.25) = 32 tokens per frame

num_adts_tokens = ceil(token_budget × alpha)     # = ceil(32 × 0.7) = 23
num_sttm_tokens = token_budget - num_adts_tokens  # = 32 - 23 = 9
```

### 2.1 DySeg — 动态视频分段

**代码位置**: `flashvid/utils.py:191-249`

**目的**: 将视频按场景变化切分成多个片段（segment），每个片段内部场景相似，后续压缩在片段内独立进行。

**算法**:
1. 对每帧的特征取 L2 归一化，计算相邻帧之间的余弦相似度
2. 相似度低于 `segment_threshold`（默认 0.9）的位置作为切割点
3. 如果切割出的段数不够 `min_segment_num`（默认 4），则从剩余位置中选相似度最低的 top-K 位置作为补充切割点

**示例**: 32 帧视频 → 切割成 [8, 6, 10, 8] 四个片段

### 2.2 ADTS — 注意力+多样性 token 选择

**代码位置**: `flashvid/token_selection.py:28-76`（ADTSv1，Qwen2.5-VL 使用此版本）

**目的**: 在每个片段内，为每帧选出最重要且最多样化的 token。

**算法（Max-Min Diversity + CLS Attention 校准）**:

1. **计算余弦距离矩阵**: 每帧 token 之间的两两余弦距离
   ```
   dist_matrix[i][j] = 1 - cosine_similarity(token_i, token_j)
   ```

2. **CLS Attention 校准**: 用 CLS attention 对距离矩阵加权
   ```
   dist_matrix = dist_matrix × cls_attention  # 注意力高的 token 权重更大
   ```

3. **贪心选择（Max-Min）**:
   - 第 1 个 token: 选距离矩阵中"最近邻距离最大"的 token（离其他 token 最远的）
   - 第 2~N 个 token: 每次选"到已选 token 集合的最小距离最大"的 token
   - 这保证了选出的 token 既重要（CLS attention 校准），又互相不重复（多样性）

**每帧选出 `num_adts_tokens` 个 token（如 23 个）**

### 2.3 TSTM — 树状时空 token 合并

**代码位置**: `flashvid/utils.py:335-401`

**目的**: 对 ADTS 没选中的 token，利用时序冗余进行跨帧合并。

**算法**:

1. **构建时序关联树**: 对相邻帧之间的未选中 token 计算余弦相似度
   ```
   cosine_sim[frame_t][token_i] = max similarity to any token in frame_{t-1}
   ```

2. **标记合并**: 如果 `cosine_sim > temporal_threshold`（默认 0.8），则该 token 被标记为"合并到前一帧的锚点 token"

3. **从后向前执行合并**: 从最后一帧开始向前遍历
   - 被标记的 token：将其特征加权累加到前一帧对应的锚点 token 上
   - 未被标记的 token：保留为当前帧的独立 token
   - 合并后执行加权平均（按累加的 token 计数归一化）

4. **DPC-kNN 空间聚类**（对合并后仍超出预算的帧）:
   - 使用 Density Peak Clustering with k-Nearest Neighbors
   - 计算每个 token 的局部密度 ρ 和到更高密度点的最小距离 δ
   - 聚类得分 = ρ × δ，选 top-K 作为聚类中心
   - 每个 token 归入最近的聚类中心，对同一聚类内的 token 取平均

### Vision-Side Compression 的整体结果

压缩后，将所有片段的 token 按原始全局顺序排列，得到压缩后的视频 token 序列。

**示例（32 帧，每帧 256 个 token）**:
```
原始:  32 × 256 = 8192 tokens
ADTS:  32 × 23  =  736 tokens（选中的重要 token）
TSTM:  ~288 tokens（合并后的上下文 token）
总计:  ~1024 tokens（约 12.5% = retention_ratio × expansion）
```

### 嵌入到 LLM 输入序列

**代码位置**: `flashvid/modeling_qwen2_5_vl.py:461-505`

压缩完成后，需要将压缩后的 token 替换到 LLM 的输入序列中：

1. 找到 `input_ids` 中视频 token 的位置范围 `[visual_start, visual_end]`
2. 用 `scatter_` 将压缩后的特征写入 `inputs_embeds` 的对应位置
3. 用 `torch.gather` 只保留压缩后的位置，删除被丢弃的 token
4. 同步更新 `position_ids`、`attention_mask`、`cache_position`

---

## Stage 3: Inner-LLM Pruning (FastV)

**入口**: `Qwen2_5_VLTextModel_forward` 中，在第 `pruning_layer` 层执行

**代码位置**: `flashvid/modeling_qwen2_5_vl.py:120-156` 和 `flashvid/utils.py:404-458`

### 目的
经过 Vision-Side Compression 后，仍有约 `retention_ratio × expansion`（如 12.5%）的视觉 token。在 LLM 内部进一步根据 LLM 自身的 attention 评估哪些视觉 token 真正有用，再做一轮剪枝。

### 触发机制
在 LLM 的 decoder 层循环中：
```python
for layer_idx, decoder_layer in enumerate(self.layers):
    if layer_idx == pruning_layer - 1:
        output_attentions = True          # 强制输出 attention weights
    elif layer_idx == pruning_layer:
        output_attentions = False
        attn = layer_outputs[1]           # 获取上一层的 attention
        fastv_prune(...)                  # 执行剪枝
```

### 算法（`fastv_prune`）

1. **获取 Attention 信号**: 取第 `pruning_layer - 1` 层的 attention weights，只看最后一个 token（即最新生成的 token）对所有 token 的注意力
   ```python
   attn = mean(attentions[:, :, -1, :], dim=1)  # 所有 head 取平均
   ```

2. **按 Attention 排序选择**: 从当前剩余的视觉 token 中，选出 attention 最高的 `visual_token_length × llm_retention_ratio` 个 token
   ```python
   num_retained = ceil(visual_token_length × 0.3)  # 保留 30%
   ```

3. **重建序列**: 将保留的视觉 token 与所有非视觉 token（system prompt、用户问题等）合并，更新 `hidden_states`、`causal_mask`、`position_ids`、`position_embeddings`

### 最终 token 数量

**示例（32 帧，每帧 256 个 token）**:
```
原始:          8192 tokens
Vision-Side:  ~1024 tokens (retention_ratio × expansion = 12.5%)
Inner-LLM:    ~307 tokens  (1024 × llm_retention_ratio = 30%)
最终保留率:    ~3.75% (= 0.10 × 1.25 × 0.3)
```

---

## 完整数据流图

```
                        视频 (32帧)
                            │
                            ▼
               ┌────────────────────────┐
               │   Vision Encoder (ViT)  │
               │   Patch Embed + Blocks  │
               └─────────┬──────────────┘
                         │
              ┌──────────┴──────────┐
              │                     │
         video_embeds          cls_attention
      (32×256, hidden)         (32, 256)
              │                     │
              └──────────┬──────────┘
                         │
                         ▼
               ┌──────────────────┐
               │   DySeg 动态分段  │
               │ 32帧 → 4个片段    │
               └────────┬─────────┘
                        │
            ┌───────────┼───────────┐
            ▼           ▼           ▼
         Seg 1       Seg 2      Seg 3 ...
            │           │           │
            ▼           ▼           ▼
    ┌───────────┐ ┌───────────┐
    │   ADTS    │ │   ADTS    │ ...   每帧选 23 个重要token
    │ (Max-Min) │ │ (Max-Min) │
    └─────┬─────┘ └─────┬─────┘
          │              │
          ▼              ▼
    ┌───────────┐ ┌───────────┐
    │   TSTM    │ │   TSTM    │ ...   跨帧合并冗余token
    │ 时空合并   │ │ 时空合并   │
    └─────┬─────┘ └─────┬─────┘
          │              │
          ▼              ▼
    ┌───────────┐ ┌───────────┐
    │  DPC-kNN  │ │  DPC-kNN  │ ...   帧内空间聚类
    │  空间聚类  │ │  空间聚类  │
    └─────┬─────┘ └─────┬─────┘
          │              │
          └──────┬───────┘
                 │
                 ▼
          ~1024 视觉 tokens
     (原始 8192 的 12.5%)
                 │
                 ▼
     ┌───────────────────────────┐
     │  拼接到 LLM 输入序列       │
     │  [sys_prompt] [视觉] [问题]│
     └──────────┬────────────────┘
                │
                ▼
     ┌───────────────────────────┐
     │  LLM Decoder Layer 0~19   │
     │  正常 Transformer 前向传播  │
     └──────────┬────────────────┘
                │
                ▼  (第19层输出 attention)
     ┌───────────────────────────┐
     │  FastV Prune (第20层)      │
     │  按 attention 保留 30%     │
     │  ~1024 → ~307 视觉 tokens  │
     └──────────┬────────────────┘
                │
                ▼
     ┌───────────────────────────┐
     │  LLM Decoder Layer 20~27  │
     │  用更少的 token 继续推理    │
     └──────────┬────────────────┘
                │
                ▼
            文本输出 (A/B/C/D)
```

---

## 关键配置参数一览

| 参数 | 默认值 | 含义 |
|---|---|---|
| `retention_ratio` | 0.10 | 视觉侧整体保留比例 |
| `expansion` | 1.25 | 过选因子，实际保留 = retention_ratio × expansion |
| `alpha` | 0.70 | ADTS 占总 token budget 的比例 |
| `temporal_threshold` | 0.8 | TSTM 中跨帧合并的余弦相似度阈值 |
| `do_segment` | True | 是否启用 DySeg 动态分段 |
| `segment_threshold` | 0.9 | DySeg 场景切割的相似度阈值 |
| `min_segment_num` | 4 | DySeg 最少分段数 |
| `pruning_layer` | 20 | Inner-LLM Pruning 执行的 LLM 层号 |
| `llm_retention_ratio` | 0.3 | Inner-LLM Pruning 保留比例 |
| `token_selection_method` | `attn_div` | Token 选择方法（Qwen2.5-VL 用 ADTSv1） |

---

## Monkey-Patch 机制

FlashVID 不修改模型权重，而是通过替换模型类的方法实现（`flashvid/__init__.py`）：

```python
# 替换 ViT 层 — 提取 CLS attention
Qwen2_5_VLVisionAttention.forward = Qwen2_5_VLVisionAttention_forward
Qwen2_5_VLVisionBlock.forward = Qwen2_5_VLVisionBlock_forward
Qwen2_5_VisionTransformerPretrainedModel.forward = ...  # 返回 (embeds, cls_attn)

# 替换视频特征提取 — 返回 cls_attention
Qwen2_5_VLModel.get_video_features = Qwen2_5_VLModel_get_video_features

# 替换模型 forward — 加入 Vision-Side Compression
Qwen2_5_VLModel.forward = Qwen2_5_VLModel_forward

# 替换 LLM forward — 加入 Inner-LLM Pruning
Qwen2_5_VLTextModel.forward = Qwen2_5_VLTextModel_forward

# 替换 LLM Attention — 支持手动计算 attention weights
Qwen2_5_VLAttention.forward = Qwen2_5_VLAttention_forward

# 替换 generate — 记录视觉 token 位置信息
Qwen2_5_VLForConditionalGeneration.generate = ..._generate
```

这样做的好处是完全 **training-free**：加载原始模型后调用 `flashvid(model)` 即可，无需微调。
