# FlashVID 参数说明

## 1. RETENTION_RATIOS — 视觉 token 保留比例

```bash
RETENTION_RATIOS=(0.15 0.25)
```

控制 Vision-Side Compression 阶段每帧保留多少比例的视觉 token。值越小压缩越激进、推理越快，但可能丢失细节。

实际保留量 = `RETENTION_RATIO × EXPANSION`，例如：
- 0.15 × 1.25 = 18.75%
- 0.25 × 1.25 = 31.25%

以每帧 256 个 token、32 帧为例：

| retention_ratio | Vision-Side 保留 token 数 | 再经 Inner-LLM Pruning (×0.3) |
|---|---|---|
| 0.15 | 32 × ceil(256×0.15×1.25) = 32×48 = 1536 | ~461 |
| 0.25 | 32 × ceil(256×0.25×1.25) = 32×80 = 2560 | ~768 |

---

## 2. DySeg 参数 — 动态视频分段

### DO_SEGMENT

```bash
DO_SEGMENT=True
```

是否启用动态视频分段。开启后，视频会按场景变化切成多个片段，每个片段独立做 token 压缩。关闭则整段视频当作一个片段处理。

### MIN_SEGMENT_NUM

```bash
MIN_SEGMENT_NUM=4
```

最少分成多少个片段。如果按 `SEGMENT_THRESHOLD` 切出来的片段数不够这个值，会从剩余帧间相似度最低的位置补充切割点。

例如 32 帧视频，如果场景变化不明显只切出了 2 段，就会在相似度最低的位置再补切 2 刀，凑够 4 段。

### COMPLEMENTARY_SEGMENT

```bash
COMPLEMENTARY_SEGMENT=True
```

是否启用补充分段。与 `MIN_SEGMENT_NUM` 配合使用：当实际切出的段数 < `MIN_SEGMENT_NUM` 时，从相邻帧相似度最低的 top-K 位置补充切割点。设为 False 则不补充，实际段数可能少于 `MIN_SEGMENT_NUM`。

### SEGMENT_THRESHOLD（代码中默认 0.9，脚本未显式设置）

相邻帧余弦相似度低于此阈值的位置被认为是场景切换点。0.9 表示只有相似度显著下降（< 0.9）才切割。

---

## 3. ADTS 和 TSTM 参数 — token 选择与合并

### TOKEN_SELECTION_METHOD

```bash
TOKEN_SELECTION_METHOD=attn_div
```

Token 选择算法。可选值：

| 值 | 算法 | 说明 |
|---|---|---|
| `attn_div` | ADTSv1 | CLS attention 校准的 Max-Min 多样性选择，Qwen2.5-VL 用这个 |
| `attn_div_v2` | ADTSv2 | 在 v1 基础上额外加入局部事件相关性校准，LLaVA 系列用这个 |
| `attn` | 纯 Attention | 只按 CLS attention 分数 top-K 选择 |
| `div` | 纯多样性 | 只按 Max-Min 多样性选择，不看 attention |

### ALPHA

```bash
ALPHA=0.70
```

ADTS 选出的 token 占总 token budget 的比例。剩余 `1 - ALPHA` 给 TSTM（时空合并）。

```
token_budget = ceil(num_tokens_per_frame × retention_ratio × expansion)

ADTS 分配: ceil(token_budget × 0.70)  → 重要且多样的 token
TSTM 分配: token_budget - ADTS 分配    → 通过跨帧合并保留的上下文 token
```

ALPHA 越大，越依赖 ADTS 显式选择；越小，越依赖 TSTM 的时空合并。

### TEMPORAL_THRESHOLD

```bash
TEMPORAL_THRESHOLD=0.8
```

TSTM 跨帧合并的余弦相似度阈值。相邻帧中同一位置的 token 如果余弦相似度 > 0.8，就合并到前一帧的对应 token 上（加权平均）。

- 值越高：合并条件越严格，保留越多独立 token
- 值越低：合并越激进，更多 token 被合并掉

---

## 4. Inner-LLM Pruning 参数 — LLM 内部剪枝

### EXPANSION

```bash
EXPANSION=1.25
```

Vision-Side 阶段的过选因子。实际保留 token 数 = `retention_ratio × expansion × num_tokens_per_frame`。

设 1.25 是因为后面还有一轮 Inner-LLM Pruning 会再砍掉一部分，所以 Vision-Side 阶段故意多保留一些，给 LLM 更多选择余地。

### PRUNING_LAYER

```bash
PRUNING_LAYER=20
```

在 LLM 的第几层执行 Inner-LLM Pruning。Qwen2.5-VL-7B 共 28 层，第 20 层意味着前 20 层用完整的视觉 token 做理解，第 20 层之后用精简后的 token 继续推理。

选择依据：层数太浅，LLM 还没充分理解视觉内容，attention 信号不准；层数太深，节省的计算量太少。20/28 ≈ 71% 的位置是一个折中。

### LLM_RETENTION_RATIO

```bash
LLM_RETENTION_RATIO=0.3
```

Inner-LLM Pruning 保留的视觉 token 比例。在第 `PRUNING_LAYER` 层，取上一层最后一个 token 对所有视觉 token 的 attention 分数，保留 top 30%，丢弃剩余 70%。

---

## 参数间的关系总结

最终视觉 token 保留率 = `RETENTION_RATIO × EXPANSION × LLM_RETENTION_RATIO`

| RETENTION_RATIO | × EXPANSION | × LLM_RETENTION_RATIO | = 最终保留率 |
|---|---|---|---|
| 0.15 | × 1.25 | × 0.3 | = 5.625% |
| 0.25 | × 1.25 | × 0.3 | = 9.375% |

以 32 帧 × 256 token = 8192 原始 token 为例：

| RETENTION_RATIO | Vision-Side 后 | Inner-LLM 后 | 压缩倍数 |
|---|---|---|---|
| 0.15 | ~1536 | ~461 | 17.8× |
| 0.25 | ~2560 | ~768 | 10.7× |
