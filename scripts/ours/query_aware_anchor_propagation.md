# Query-Aware Anchor Propagation for Visual Token Pruning in Video Question Answering

## 方法技术文档（详细设计 + 工程细节）

---

## 1. 问题定义

### 1.1 输入

- **视频帧序列**：$F$ 帧，均匀采样自原始视频（如 $F=32$ 或 $F=64$）
- **用户 Query**：一个自然语言问题（如 "What does the person do after picking up the cup?"）
- **Vision Encoder 输出**：每帧经过 ViT 编码后得到 $N_v$ 个 visual token，总共 $F \times N_v$ 个 token
  - 例：LLaVA-OneVision 使用 SigLIP，每帧 196 个 token（14×14 grid），32 帧共 6272 个 visual token

### 1.2 输出

- **剪枝后的 visual token 子集**：从 $F \times N_v$ 中选出远少于原始数量的 token
- 与 query 的 text token 拼接后送入 LLM 生成答案

### 1.3 目标

- 在大幅减少 visual token 数量（如保留 10%-25%）的同时，保持甚至提升视频问答性能
- 核心思路：利用 query 信息指导 token 选择，将 pruning 从 "query-agnostic 的冗余压缩" 转化为 "query-driven 的视觉证据保留"

---

## 2. 方法总览

整体 Pipeline 分为 5 个阶段：

```
阶段 0: 视频编码 (Vision Encoding)
    ↓
阶段 1: 视频分段 (Video Partition)
    ↓
阶段 2: 锚点帧 Token 选择 (Anchor Frame Token Selection) ← query-aware
    ↓
阶段 3: 后续帧 Token 传播与选择 (Token Propagation for Non-anchor Frames)
    ↓
阶段 4: 汇总与 LLM 推理 (Aggregation & LLM Inference)
```

---

## 3. 阶段 0：视频编码

### 3.1 流程

1. 对输入视频均匀采样 $F$ 帧
2. 每帧独立通过 Vision Encoder（如 SigLIP）编码，得到：
   - Visual token 序列：$E_v^{(f)} \in \mathbb{R}^{N_v \times d}$，$f = 1, \dots, F$
   - 每个 token 对应原图中一个 patch（如 14×14 grid 中的一个位置）
3. 通过 modality connector（如 MLP projector）将视觉特征投射到 LLM 的文本空间
4. Query 文本通过 tokenizer 编码为 text token 序列 $H_t \in \mathbb{R}^{N_t \times d}$

### 3.2 工程细节

- **ViT 中间层特征保存**：除了最后一层输出，建议同时缓存 ViT 若干中间层的输出特征（如第 6、12、18、24 层），供后续阶段使用（层间变化度计算、多层距离矩阵等）
- **数据格式**：每帧的 token 需要附带其在 grid 中的空间坐标 $(row, col)$，后续邻域保留时需要用到
- **内存管理**：所有帧的 visual token 在编码后全部保存在 GPU 显存中，后续阶段的操作均为 in-place 或轻量计算

```python
# 伪代码：视频编码
visual_tokens = []  # List of [N_v, d] tensors
for frame in sampled_frames:
    tokens = vision_encoder(frame)          # [N_v, d]
    tokens = modality_connector(tokens)     # [N_v, d_llm]
    visual_tokens.append(tokens)

visual_tokens = torch.stack(visual_tokens)  # [F, N_v, d_llm]
text_tokens = tokenizer_embed(query)        # [N_t, d_llm]
```

---

## 4. 阶段 1：视频分段 (Video Partition)

### 4.1 目的

将 $F$ 帧分成若干语义连续的 segment，确保同一 segment 内场景一致，避免跨场景的 token 被混合处理。每个 segment 独立进行后续的锚点选择和传播。

### 4.2 方法

1. 对每帧 visual token 做全局平均池化，得到帧级 embedding：
   $$f_e^{(i)} = \text{GAP}(E_v^{(i)}) \in \mathbb{R}^{d}$$

2. 计算相邻帧之间的 transition similarity：
   $$t_i = \cos(f_e^{(i)}, f_e^{(i+1)}), \quad i = 1, \dots, F-1$$

3. 分段规则：
   - 如果 $t_i < S_\tau$（场景切换阈值），在第 $i$ 帧和第 $i+1$ 帧之间切分
   - 保证最少有 $M_s$ 个 segment（如果自然切分不够，对剩余最长的 segment 继续按最低 transition similarity 切分）

### 4.3 超参数

| 参数 | 含义 | 推荐值 | 说明 |
|------|------|--------|------|
| $S_\tau$ | 场景切换阈值 | 0.9 | 低于此值认为发生场景切换 |
| $M_s$ | 最少 segment 数量 | 8 | 保证足够的时间粒度 |

### 4.4 工程细节

- 分段操作仅涉及帧级 embedding 的余弦相似度计算，计算量极小（$O(F \cdot d)$）
- 输出为 segment 列表，每个 segment 记录其起止帧索引：`segments = [(0, 5), (6, 12), (13, 20), ...]`

```python
# 伪代码：视频分段
frame_embeddings = visual_tokens.mean(dim=1)  # [F, d]
transition_sim = cosine_similarity(frame_embeddings[:-1], frame_embeddings[1:])  # [F-1]

# 按阈值切分
cut_points = [0]
for i, sim in enumerate(transition_sim):
    if sim < S_tau:
        cut_points.append(i + 1)
cut_points.append(F)

segments = [(cut_points[i], cut_points[i+1]) for i in range(len(cut_points)-1)]

# 如果 segment 数不足 M_s，对最长的 segment 继续切分
while len(segments) < M_s:
    longest_idx = argmax([end - start for start, end in segments])
    start, end = segments[longest_idx]
    if end - start <= 1:
        break  # 无法再切
    # 在该 segment 内找 transition similarity 最低的位置切分
    sub_sims = transition_sim[start:end-1]
    split_pos = start + argmin(sub_sims) + 1
    segments[longest_idx] = (start, split_pos)
    segments.insert(longest_idx + 1, (split_pos, end))
```

---

## 5. 阶段 2：锚点帧 Token 选择

### 5.1 锚点帧的确定

每个 segment 选择一帧作为锚点帧（anchor frame）。

**选择策略**：选择 segment 内与其他帧平均相似度最高的帧（最具代表性）。

```python
# 伪代码：锚点帧选择
def select_anchor_frame(segment_tokens):
    """segment_tokens: [L, N_v, d]，L 为 segment 内帧数"""
    frame_embs = segment_tokens.mean(dim=1)           # [L, d]
    sim_matrix = cosine_similarity_matrix(frame_embs)  # [L, L]
    avg_sim = sim_matrix.mean(dim=1)                   # [L]
    anchor_idx = avg_sim.argmax()
    return anchor_idx
```

**备选策略**：直接选每个 segment 的第一帧，实现更简单，且与"前向传播"的设计更自然。

### 5.2 锚点帧上的两类 Token 选择

在锚点帧上，需要选出两类 token：

- **局部锚点 (Local Anchors)**：与 query 高度相关的关键区域 token
- **全局探索 token (Global Exploration Tokens)**：覆盖锚点未关注区域的补充 token

两类 token 的选择是一个统一的流程，分为两个阶段。

#### 5.2.1 第一阶段：局部锚点选择（Query-Aware）

**目标**：找到锚点帧中与 query 最相关的 visual token。

**方法**：将锚点帧的 visual token 和 query 的 text token 一起送入 LLM 的前 $K$ 层，提取 cross-modal attention 作为 query relevance score。

**具体步骤**：

1. **构造输入**：拼接锚点帧的 visual token 和 query 的 text token
   $$H = [\text{visual\_tokens}^{(\text{anchor})}; \text{text\_tokens}]$$

2. **LLM 前向传播前 $K$ 层**：
   $$H^{(K)} = \text{LLM\_Layer}_{1:K}(H)$$

3. **提取 cross-modal attention**：在第 $K$ 层的注意力矩阵中，提取所有 text token 对每个 visual token 的注意力权重，聚合为每个 visual token 的 query relevance score：
   $$r_j = \frac{1}{N_t} \sum_{i=1}^{N_t} \text{Attn}^{(K)}(t_i, v_j)$$
   
   其中 $t_i$ 是第 $i$ 个 text token，$v_j$ 是第 $j$ 个 visual token。
   
   > **替代方案**：也可以取 max 而非 mean，即 $r_j = \max_{i} \text{Attn}^{(K)}(t_i, v_j)$，这会更关注与 query 中某个关键词强相关的 token。两种方式都可以实验对比。

4. **自适应阈值选择**：
   $$\text{threshold} = \mu(r) + \alpha \cdot \sigma(r)$$
   
   其中 $\mu(r)$ 和 $\sigma(r)$ 分别是 relevance score 的均值和标准差，$\alpha$ 控制选择的严格程度。选择所有 $r_j > \text{threshold}$ 的 token 作为局部锚点。

5. **邻域扩展**：对每个选中的局部锚点，额外保留其在空间 grid 中的邻域 token。
   - 邻域大小可以固定（如 3×3，即上下左右各扩一格）
   - 或按重要性自适应：$\text{neighborhood\_size}(a_i) = \text{base\_size} + \lfloor r_i / r_{\max} \cdot \text{extra\_size} \rfloor$

```python
# 伪代码：局部锚点选择
def select_local_anchors(visual_tokens_anchor, text_tokens, llm, K, alpha=1.0):
    """
    visual_tokens_anchor: [N_v, d] 锚点帧的 visual token
    text_tokens: [N_t, d] query 的 text token
    llm: 大语言模型
    K: 使用前 K 层
    alpha: 阈值系数
    """
    # 拼接输入
    combined = torch.cat([visual_tokens_anchor, text_tokens], dim=0)  # [N_v + N_t, d]
    
    # LLM 前 K 层前向，提取注意力
    _, attention_maps = llm.forward_layers(combined, layers=range(K), output_attentions=True)
    
    # 提取第 K 层的 cross-modal attention
    # attn_K: [num_heads, N_v + N_t, N_v + N_t]
    attn_K = attention_maps[K - 1]
    
    # text-to-visual attention: text token 对 visual token 的注意力
    # 取所有 head 的平均
    cross_attn = attn_K[:, N_v:, :N_v].mean(dim=0)  # [N_t, N_v]
    
    # 每个 visual token 的 query relevance score
    relevance_scores = cross_attn.mean(dim=0)  # [N_v]
    # 备选：relevance_scores = cross_attn.max(dim=0).values
    
    # 自适应阈值
    mu = relevance_scores.mean()
    sigma = relevance_scores.std()
    threshold = mu + alpha * sigma
    
    # 选择局部锚点
    anchor_mask = relevance_scores > threshold  # [N_v] bool
    anchor_indices = anchor_mask.nonzero(as_tuple=True)[0]  # 选中的 token 索引
    
    # 邻域扩展
    anchor_indices_with_neighbors = expand_neighbors(
        anchor_indices, 
        grid_size=(H_grid, W_grid),  # 如 14×14
        neighbor_radius=1            # 3×3 邻域
    )
    
    return anchor_indices_with_neighbors, relevance_scores


def expand_neighbors(indices, grid_size, neighbor_radius):
    """将 token 索引扩展到包含空间邻域"""
    H, W = grid_size
    expanded = set()
    for idx in indices.tolist():
        row, col = idx // W, idx % W
        for dr in range(-neighbor_radius, neighbor_radius + 1):
            for dc in range(-neighbor_radius, neighbor_radius + 1):
                nr, nc = row + dr, col + dc
                if 0 <= nr < H and 0 <= nc < W:
                    expanded.add(nr * W + nc)
    return torch.tensor(sorted(expanded))
```

#### 5.2.2 第二阶段：全局探索 Token 选择（Coverage-Aware MMDP）

**目标**：在局部锚点未覆盖的区域中，选出一组分散的探索 token，保证画面的全局覆盖。

**关键设计**：以局部锚点为已选集合的初始化，在剩余 token 上做 MMDP 扩展。

**具体步骤**：

1. **确定候选集合**：所有未被局部锚点覆盖的 token
   $$\text{candidates} = \{1, \dots, N_v\} \setminus \text{anchor\_indices\_with\_neighbors}$$

2. **构造距离矩阵**：在候选 token 之间以及候选 token 与锚点 token 之间计算余弦距离
   $$D(i, j) = 1 - \cos(e_i, e_j)$$

3. **以锚点为初始集合做 MMDP 贪心扩展**：

```python
# 伪代码：全局探索 token 选择（以锚点初始化的 MMDP）
def select_global_exploration_tokens(
    visual_tokens_frame,     # [N_v, d] 当前帧所有 token
    anchor_indices,          # 已选的局部锚点索引（含邻域）
    M_global,                # 需要选出的全局探索 token 数量
):
    N_v = visual_tokens_frame.shape[0]
    all_indices = set(range(N_v))
    anchor_set = set(anchor_indices.tolist())
    candidate_indices = sorted(all_indices - anchor_set)
    
    if len(candidate_indices) == 0 or M_global == 0:
        return torch.tensor([], dtype=torch.long)
    
    # 初始化：计算每个候选 token 到已选集合（锚点）的最小距离
    anchor_features = visual_tokens_frame[anchor_indices]  # [M_anchor, d]
    candidate_features = visual_tokens_frame[candidate_indices]  # [M_cand, d]
    
    # [M_cand, M_anchor] 距离矩阵
    dist_to_anchors = 1 - cosine_similarity_matrix(candidate_features, anchor_features)
    
    # 每个候选到已选集合的最小距离
    min_dist = dist_to_anchors.min(dim=1).values  # [M_cand]
    
    selected_global = []
    
    for _ in range(min(M_global, len(candidate_indices))):
        # 选择距离已选集合最远的候选
        best_local_idx = min_dist.argmax().item()
        
        if min_dist[best_local_idx] <= 0:
            break  # 所有候选都已经和某个已选 token 完全重合
        
        selected_global.append(candidate_indices[best_local_idx])
        
        # 更新 min_dist：加入新选的 token 后，部分候选的最小距离可能需要更新
        new_token_feature = candidate_features[best_local_idx].unsqueeze(0)  # [1, d]
        dist_to_new = 1 - cosine_similarity_matrix(candidate_features, new_token_feature).squeeze(1)  # [M_cand]
        min_dist = torch.min(min_dist, dist_to_new)
        
        # 已选的不再参与后续选择（设为 -inf）
        min_dist[best_local_idx] = -float('inf')
    
    return torch.tensor(selected_global, dtype=torch.long)
```

#### 5.2.3 锚点帧最终保留的 Token

$$\text{Token}_{\text{anchor}} = \text{anchor\_indices\_with\_neighbors} \cup \text{global\_exploration\_indices}$$

#### 5.2.4 锚点信息的存储

每个局部锚点需要存储以下信息，用于后续帧的传播：

```python
@dataclass
class AnchorInfo:
    feature: torch.Tensor           # [d] 锚点在锚点帧上的原始特征（不变，用于锚定）
    grid_position: Tuple[int, int]  # (row, col) 在 grid 中的空间位置
    relevance_score: float          # query relevance score
    neighbor_radius: int            # 邻域半径（由 R 决定）
    prev_match_feature: torch.Tensor  # [d] 上一帧匹配到的特征（逐帧更新，用于适应性）
```

### 5.3 超参数

| 参数 | 含义 | 推荐值 | 说明 |
|------|------|--------|------|
| $K$ | LLM 前向传播层数 | 2-4 | 越深 query-awareness 越强，但计算开销越大 |
| $\alpha$ | 阈值系数 | 1.0 | 控制"质量门槛"，越大锚点越少 |
| neighbor_radius | 邻域半径 | 由 $R$ 派生 | 见 §5.4.3，$R$ 越小邻域越小 |
| $M_{\text{global}}$ | 全局探索 token 数量 | 由预算分配决定 | 见 §5.4.2 |

### 5.4 剪枝率参数 $R$ 的全局控制机制

剪枝率 $R$（retention ratio）是用户指定的唯一核心控制参数，所有其他行为都应该由 $R$ 驱动或联动。本节详述 $R$ 如何逐层传导到 pipeline 的每个环节。

#### 5.4.1 第一层：全局预算 → 每帧预算

每帧保留的 token 数量：

$$B = \lfloor R \times N_v \rfloor$$

**推荐采用均匀分配**：每帧预算相同。原因是 query-aware 的选择机制已经自适应地为不同帧分配了不同的"信息密度"——重要帧上锚点多、探索少，不重要帧上锚点少（匹配失败）、探索多，但总量一致，便于实现和对齐 token budget。

> **进阶方案（可选 ablation）**：允许帧间预算不均匀。比如根据每帧与锚点帧的相似度分配权重，相似度高的帧（内容接近锚点帧，锚点匹配容易）给少一些预算，相似度低的帧（内容变化大，需要更多探索）给多一些预算，但总量满足 $\sum_{f=1}^{F} B_f = R \times F \times N_v$。

#### 5.4.2 第二层：每帧预算 → 锚点 vs 探索的分配

**核心策略："锚点优先，探索兜底，预算硬约束"**

```
给定每帧预算 B：

1. 自适应阈值选出锚点候选 → 含邻域后总数为 B_local_raw
2. 计算锚点预算上限：B_local_max = floor(B × r_max)
3. 计算探索预算下限：M_global_min = max(M_explore_floor, floor(B × r_explore_min))

4. 如果 B_local_raw ≤ B - M_global_min:
       → 保留所有锚点，剩余全给探索
       → B_local = B_local_raw
       → B_global = B - B_local
       
5. 如果 B_local_raw > B - M_global_min 且 B_local_raw ≤ B_local_max:
       → 保留所有锚点，探索被压缩到 M_global_min
       → B_local = B_local_raw
       → B_global = M_global_min
       → 此帧实际保留 B_local + M_global_min 个 token（可能略超 B，或截断锚点）
       
6. 如果 B_local_raw > B_local_max:
       → 锚点太多，按 relevance score 排序截断
       → 从高到低逐个加入锚点（含邻域），直到用完 B_local_max
       → B_global = B - B_local_max
```

**参数说明**：

| 参数 | 含义 | 推荐值 | 说明 |
|------|------|--------|------|
| $r_{\max}$ | 锚点预算占比上限 | 0.7 | 锚点最多占每帧预算的 70% |
| $r_{\text{explore\_min}}$ | 探索预算占比下限 | 0.15 | 探索 token 至少占 15% |
| $M_{\text{explore\_floor}}$ | 探索 token 绝对下限 | 3 | 不论预算多紧张，至少保留 3 个探索 token |

```python
# 伪代码：预算分配
def allocate_budget(
    B: int,                     # 每帧总预算
    B_local_raw: int,           # 自适应阈值选出的锚点数（含邻域）
    relevance_scores: torch.Tensor,  # 每个锚点的 relevance score
    anchor_indices_raw: torch.Tensor,  # 锚点索引（含邻域）
    core_anchor_indices: torch.Tensor,  # 核心锚点索引（不含邻域）
    r_max: float = 0.7,
    r_explore_min: float = 0.15,
    M_explore_floor: int = 3,
) -> Tuple[torch.Tensor, int]:
    """
    返回：(最终锚点索引, 全局探索 token 预算)
    """
    B_local_max = int(B * r_max)
    M_global_min = max(M_explore_floor, int(B * r_explore_min))
    
    if B_local_raw <= B - M_global_min:
        # 情况 1：锚点不多，预算充裕
        final_anchor_indices = anchor_indices_raw
        B_global = B - B_local_raw
        
    elif B_local_raw <= B_local_max:
        # 情况 2：锚点较多但未超上限，压缩探索到下限
        final_anchor_indices = anchor_indices_raw
        B_global = M_global_min
        
    else:
        # 情况 3：锚点太多，需要截断
        # 按 relevance score 从高到低逐个加入核心锚点
        sorted_core = core_anchor_indices[relevance_scores[core_anchor_indices].argsort(descending=True)]
        
        selected = set()
        for idx in sorted_core.tolist():
            # 加入该锚点及其邻域
            neighbors = get_neighbor_indices(idx, grid_size, neighbor_radius)
            candidate = selected | set(neighbors)
            if len(candidate) > B_local_max:
                break  # 再加就超预算了
            selected = candidate
        
        final_anchor_indices = torch.tensor(sorted(selected))
        B_global = B - len(final_anchor_indices)
    
    # 安全检查：确保探索预算不为负
    B_global = max(B_global, M_global_min)
    
    return final_anchor_indices, B_global
```

#### 5.4.3 $R$ 对邻域大小的联动控制

邻域扩展会放大锚点的 token 占用。$R$ 越小，预算越紧张，邻域应相应缩小：

$$\text{neighbor\_radius}(R) = \begin{cases} 2 & \text{if } R \geq 0.25 \\\ 1 & \text{if } 0.10 \leq R < 0.25 \\\ 0 & \text{if } R < 0.10 \end{cases}$$

邻域半径对单个锚点 token 占用的影响：

| 半径 | 邻域形状 | 单锚点占用 token 数 | 适用 $R$ |
|------|---------|-------------------|---------|
| 0 | 仅锚点自身 | 1 | $R < 0.10$ |
| 1 | 3×3 | 最多 9 | $0.10 \leq R < 0.25$ |
| 2 | 5×5 | 最多 25 | $R \geq 0.25$ |

> 注意：边界位置的邻域不足完整方块，实际占用会小于上限。

```python
def get_neighbor_radius(R: float) -> int:
    if R >= 0.25:
        return 2
    elif R >= 0.10:
        return 1
    else:
        return 0
```

#### 5.4.4 $R$ 对自适应阈值的影响策略

有两种设计哲学，推荐**方案 B**：

**方案 A：$\alpha$ 与 $R$ 联动（不推荐）**

$$\alpha(R) = \alpha_{\text{base}} + \beta \cdot (1 - R)$$

$R$ 越小 → $\alpha$ 越大 → 阈值越高 → 锚点越少。问题是把两个不同语义的控制（"什么质量的 token 算锚点" vs "总共保留多少 token"）耦合在了一起，不好调试。

**方案 B：$\alpha$ 固定，用预算硬约束截断（推荐）**

$\alpha$ 保持固定（如 1.0），让自适应阈值自由决定"什么样的 token 有资格成为锚点"。然后通过 §5.4.2 的预算分配机制来硬约束最终数量。

好处是职责分离：
- $\alpha$ 控制**质量门槛**（"多相关才算锚点"）
- $R$ 控制**数量上限**（"最终保留多少"）
- 两者独立可调，互不干扰

#### 5.4.5 $R$ 在传播阶段的影响

$R$ 对传播阶段（阶段 3）的影响是间接的、自然传导的：

1. **$R$ 小 → 每帧预算 $B$ 小 → 邻域半径小 → 锚点匹配区域占用少**
   → 更多预算给全局探索 → method 自动从"精确跟踪"模式转向"广泛探索"模式

2. **$R$ 大 → 每帧预算 $B$ 大 → 邻域半径大 → 锚点匹配区域占用多**
   → 全局探索适量补充 → method 处于"充分跟踪 + 适度探索"模式

3. **$T_{\text{match}}$ 不与 $R$ 联动**：匹配质量阈值控制的是"对不对"，不是"多不多"，应保持独立。即使预算充裕也不应降低匹配质量标准。

#### 5.4.6 $R$ 的行为总结表

| $R$ | $B$ (per frame, $N_v=196$) | 邻域半径 | 锚点行为 | 探索行为 | 整体模式 |
|-----|---------------------------|---------|---------|---------|---------|
| 0.05 | ~10 | 0 | 仅保留 3-5 个最相关 token | 5-7 个探索 token | 极度聚焦 |
| 0.10 | ~20 | 1 | 保留核心锚点 + 小邻域 | ~6 个探索 token | 聚焦 + 基本覆盖 |
| 0.15 | ~29 | 1 | 保留多数锚点 + 邻域 | ~10 个探索 token | 平衡模式 |
| 0.20 | ~39 | 1 | 保留几乎所有锚点 + 邻域 | ~12-15 个探索 token | 宽裕跟踪 |
| 0.25 | ~49 | 2 | 保留所有锚点 + 大邻域 | ~15-20 个探索 token | 充分保留 |

#### 5.4.7 完整的预算控制 Config

```python
@dataclass
class BudgetConfig:
    """由 R 驱动的预算控制配置"""
    
    R: float                    # 核心参数：retention ratio (0.05 ~ 0.30)
    N_v: int = 196              # 每帧 visual token 数
    
    # === 以下参数由 R 自动派生或有推荐默认值 ===
    
    # 每帧总预算
    @property
    def B(self) -> int:
        return int(self.R * self.N_v)
    
    # 邻域半径
    @property
    def neighbor_radius(self) -> int:
        if self.R >= 0.25:
            return 2
        elif self.R >= 0.10:
            return 1
        else:
            return 0
    
    # 锚点预算占比上限
    r_max: float = 0.7
    
    # 探索预算占比下限
    r_explore_min: float = 0.15
    
    # 探索 token 绝对下限
    M_explore_floor: int = 3
    
    # 自适应阈值系数（固定，不随 R 变化）
    alpha: float = 1.0
    
    # 匹配有效性阈值（固定，不随 R 变化）
    T_match: float = 0.7
    
    # 锚定权重下限
    lambda_min: float = 0.4
    
    # === 派生的预算数值 ===
    @property
    def B_local_max(self) -> int:
        return int(self.B * self.r_max)
    
    @property
    def M_global_min(self) -> int:
        return max(self.M_explore_floor, int(self.B * self.r_explore_min))
    
    def summary(self) -> str:
        return (
            f"R={self.R:.0%} | B={self.B} tokens/frame | "
            f"neighbor_r={self.neighbor_radius} | "
            f"anchor_max={self.B_local_max} | explore_min={self.M_global_min}"
        )
```

#### 5.4.8 端到端示例：$R=0.10$，$N_v=196$

```
输入：R=0.10, N_v=196
 → B = 19 tokens/frame
 → neighbor_radius = 1 (3×3)
 → B_local_max = floor(19 × 0.7) = 13
 → M_global_min = max(3, floor(19 × 0.15)) = 3

锚点帧上：
  自适应阈值选出 4 个核心锚点
  → 含 3×3 邻域后（去重）约 28 个 token
  → 28 > B_local_max (13)，触发截断
  → 按 relevance 排序，保留 top-2 锚点 + 邻域 ≈ 13 个 token
  → 全局探索预算 = 19 - 13 = 6 个 token
  → 最终：13 锚点区域 + 6 探索 = 19 tokens ✓

传播帧上：
  2 个锚点传播匹配，1 个成功 1 个失败
  → 成功的占 ~7 个 token（锚点 + 3×3 邻域去重）
  → 全局探索预算 = 19 - 7 = 12 个 token
  → 最终：7 锚点区域 + 12 探索 = 19 tokens ✓
  → 注意：匹配失败导致更多预算自动流向探索，行为合理
```

---

## 6. 阶段 3：后续帧 Token 传播与选择

### 6.1 概述

对于每个 segment 内锚点帧之后（以及之前，如果锚点帧不在 segment 开头）的每一帧，通过轻量特征匹配传播锚点信息，选择保留的 token。

**核心原则**：不再调用 LLM，仅通过 visual token 之间的特征匹配完成 token 选择。

### 6.2 锚点传播（Anchor Propagation）

对于当前帧 $f$ 和锚点集合 $\{a_1, a_2, \dots, a_M\}$，执行以下匹配操作：

#### 6.2.1 特征匹配

对每个锚点 $a_i$，在当前帧的所有 visual token 中找最佳匹配：

$$\text{match}(a_i) = \arg\max_{j \in \{1, \dots, N_v\}} \text{score}(a_i, e_j^{(f)})$$

**匹配分数的计算**（混合锚定策略）：

$$\text{score}(a_i, e_j^{(f)}) = \lambda \cdot \cos(f_i^{\text{anchor}}, e_j^{(f)}) + (1 - \lambda) \cdot \cos(f_i^{\text{prev}}, e_j^{(f)})$$

其中：
- $f_i^{\text{anchor}}$：锚点 $a_i$ 在锚点帧上的**原始特征**（不变，提供锚定）
- $f_i^{\text{prev}}$：锚点 $a_i$ 在上一帧中匹配到的 token 的**特征**（逐帧更新，提供适应性）
- $\lambda$：锚定权重，根据当前帧与锚点帧的时间距离动态调节

**$\lambda$ 的调节策略**：

$$\lambda = \max\left(\lambda_{\min}, \; 1 - \frac{|f - f_{\text{anchor}}|}{\text{segment\_length}}\right)$$

- 离锚点帧近时 $\lambda \approx 1$（更信任原始锚点特征）
- 离锚点帧远时 $\lambda$ 衰减到 $\lambda_{\min}$（更依赖逐帧传递）
- $\lambda_{\min}$ 设为 0.3-0.5，保证始终有一定的锚定效果，防止完全漂移

#### 6.2.2 匹配有效性判断

只有当最佳匹配的相似度超过阈值时，才认为匹配成功：

$$\text{match\_valid}(a_i) = \begin{cases} \text{True}, & \text{if } \max_j \text{score}(a_i, e_j^{(f)}) \geq T_{\text{match}} \\ \text{False}, & \text{otherwise} \end{cases}$$

匹配失败说明该锚点对应的物体/区域在当前帧中不可见（遮挡或离开画面）。

**离线设定下的简单处理**：匹配失败的锚点在当前帧中不保留任何 token，但该锚点仍然保留在集合中，继续参与后续帧的匹配（物体可能重新出现）。

#### 6.2.3 邻域保留

匹配成功的锚点，保留其匹配位置及空间邻域：

```python
def propagate_anchors(
    anchors: List[AnchorInfo],
    current_frame_tokens: torch.Tensor,   # [N_v, d]
    anchor_frame_idx: int,
    current_frame_idx: int,
    segment_length: int,
    T_match: float = 0.7,
    lambda_min: float = 0.3,
    grid_size: Tuple[int, int] = (14, 14),
):
    """
    返回：
    - matched_indices: 匹配成功的 token 索引（含邻域）
    - updated_prev_features: 用于下一帧传播的特征
    """
    # 动态调节 lambda
    time_ratio = abs(current_frame_idx - anchor_frame_idx) / max(segment_length, 1)
    lam = max(lambda_min, 1.0 - time_ratio)
    
    matched_indices = set()
    updated_prev_features = {}
    
    for i, anchor in enumerate(anchors):
        # 混合匹配分数
        sim_to_anchor = cosine_similarity(anchor.feature.unsqueeze(0), current_frame_tokens)  # [1, N_v]
        
        if anchor.prev_match_feature is not None:
            sim_to_prev = cosine_similarity(anchor.prev_match_feature.unsqueeze(0), current_frame_tokens)
            score = lam * sim_to_anchor + (1 - lam) * sim_to_prev
        else:
            score = sim_to_anchor
        
        score = score.squeeze(0)  # [N_v]
        best_idx = score.argmax().item()
        best_score = score[best_idx].item()
        
        if best_score >= T_match:
            # 匹配成功：保留匹配 token 及邻域
            neighbor_indices = get_neighbor_indices(best_idx, grid_size, anchor.neighbor_radius)
            matched_indices.update(neighbor_indices)
            
            # 更新 prev_match_feature 用于下一帧
            updated_prev_features[i] = current_frame_tokens[best_idx].clone()
        else:
            # 匹配失败：不保留 token，prev_feature 保持不变
            updated_prev_features[i] = anchor.prev_match_feature
    
    return torch.tensor(sorted(matched_indices)), updated_prev_features
```

#### 6.2.4 一对多匹配的冲突处理

多个锚点可能匹配到相同或重叠的 token，处理方式：

- **允许重叠**：不同锚点匹配到相邻位置是正常的（可能是同一大物体的不同部分），邻域取并集即可
- **完全相同匹配的处理**：如果两个锚点匹配到完全相同的 token，说明这两个锚点在当前帧中指向同一区域，保留一份邻域即可（自然去重）

### 6.3 全局探索 Token 选择

与锚点帧上的逻辑完全一致：以当前帧中匹配成功的锚点区域为已选集合，对剩余 token 做 MMDP 扩展。

```python
# 复用 §5.2.2 的 select_global_exploration_tokens 函数
global_indices = select_global_exploration_tokens(
    visual_tokens_frame=current_frame_tokens,
    anchor_indices=matched_indices,  # 锚点匹配区域（含邻域）
    M_global=B - len(matched_indices),  # 剩余预算
)
```

### 6.4 当前帧最终保留的 Token

$$\text{Token}^{(f)} = \text{matched\_indices} \cup \text{global\_exploration\_indices}$$

### 6.5 完整的 Segment 内处理流程

```python
def process_segment(
    segment_tokens: torch.Tensor,      # [L, N_v, d] segment 内所有帧的 token
    text_tokens: torch.Tensor,          # [N_t, d]
    llm: LLM,
    anchor_frame_local_idx: int,        # 锚点帧在 segment 内的索引
    budget_config: BudgetConfig,        # 由 R 驱动的预算配置
) -> List[torch.Tensor]:
    """
    返回：每帧保留的 token 索引列表
    """
    L, N_v, d = segment_tokens.shape
    B = budget_config.B                              # 每帧总预算
    neighbor_radius = budget_config.neighbor_radius  # 由 R 决定的邻域半径
    retained_indices = [None] * L
    
    # ========== Step 1: 锚点帧处理 ==========
    anchor_tokens = segment_tokens[anchor_frame_local_idx]  # [N_v, d]
    
    # 1a. 用 LLM 浅层 attention 选择核心锚点（不含邻域）
    core_anchor_indices, relevance_scores = select_local_anchors(
        anchor_tokens, text_tokens, llm, 
        K=budget_config.K, alpha=budget_config.alpha
    )
    # core_anchor_indices: 仅超过阈值的 token 索引，不含邻域
    
    # 1b. 邻域扩展
    anchor_with_neighbors = expand_neighbors(
        core_anchor_indices, 
        grid_size=(budget_config.grid_H, budget_config.grid_W),
        neighbor_radius=neighbor_radius
    )
    B_local_raw = len(anchor_with_neighbors)
    
    # 1c. 预算分配：锚点 vs 探索
    final_anchor_indices, B_global = allocate_budget(
        B=B,
        B_local_raw=B_local_raw,
        relevance_scores=relevance_scores,
        anchor_indices_raw=anchor_with_neighbors,
        core_anchor_indices=core_anchor_indices,
        r_max=budget_config.r_max,
        r_explore_min=budget_config.r_explore_min,
        M_explore_floor=budget_config.M_explore_floor,
    )
    
    # 1d. 构造锚点信息（仅记录核心锚点，不含纯邻域 token）
    # 筛选出截断后仍保留的核心锚点
    final_anchor_set = set(final_anchor_indices.tolist())
    anchors = []
    for idx in core_anchor_indices.tolist():
        if idx in final_anchor_set:  # 该核心锚点未被截断
            anchors.append(AnchorInfo(
                feature=anchor_tokens[idx].clone(),
                grid_position=(idx // budget_config.grid_W, idx % budget_config.grid_W),
                relevance_score=relevance_scores[idx].item(),
                neighbor_radius=neighbor_radius,
                prev_match_feature=anchor_tokens[idx].clone(),
            ))
    
    # 1e. 选择全局探索 token
    global_indices = select_global_exploration_tokens(
        anchor_tokens, final_anchor_indices, M_global=B_global
    )
    
    retained_indices[anchor_frame_local_idx] = torch.cat(
        [final_anchor_indices, global_indices]
    ).unique()
    
    # ========== Step 2: 前向传播（锚点帧之后的帧） ==========
    for f in range(anchor_frame_local_idx + 1, L):
        current_tokens = segment_tokens[f]  # [N_v, d]
        
        # 2a. 锚点传播匹配
        matched_indices, updated_prev = propagate_anchors(
            anchors, current_tokens,
            anchor_frame_idx=anchor_frame_local_idx,
            current_frame_idx=f,
            segment_length=L,
            T_match=budget_config.T_match,
            lambda_min=budget_config.lambda_min,
            grid_size=(budget_config.grid_H, budget_config.grid_W),
        )
        
        # 更新 prev_match_feature
        for i, feat in updated_prev.items():
            if feat is not None:
                anchors[i].prev_match_feature = feat
        
        # 2b. 预算约束：如果匹配区域已超预算，截断
        if len(matched_indices) > budget_config.B_local_max:
            # 按锚点 relevance score 排序，优先保留高分锚点的匹配区域
            matched_indices = truncate_matched_by_relevance(
                matched_indices, anchors, budget_config.B_local_max,
                grid_size=(budget_config.grid_H, budget_config.grid_W),
                neighbor_radius=neighbor_radius,
            )
        
        # 2c. 全局探索
        remaining_budget = B - len(matched_indices)
        remaining_budget = max(remaining_budget, budget_config.M_global_min)
        
        global_indices = select_global_exploration_tokens(
            current_tokens, matched_indices,
            M_global=remaining_budget
        )
        
        retained_indices[f] = torch.cat([matched_indices, global_indices]).unique()
        
        # 安全检查：确保不超总预算（unique 后可能 <= B + M_global_min）
        if len(retained_indices[f]) > B + budget_config.M_explore_floor:
            # 优先保留锚点匹配区域，截断探索 token
            n_keep_global = B - len(matched_indices)
            retained_indices[f] = torch.cat([
                matched_indices, global_indices[:max(0, n_keep_global)]
            ]).unique()
    
    # ========== Step 3: 反向传播（锚点帧之前的帧） ==========
    # 重置 prev_match_feature
    for anchor in anchors:
        anchor.prev_match_feature = anchor.feature.clone()
    
    for f in range(anchor_frame_local_idx - 1, -1, -1):
        current_tokens = segment_tokens[f]
        
        matched_indices, updated_prev = propagate_anchors(
            anchors, current_tokens,
            anchor_frame_idx=anchor_frame_local_idx,
            current_frame_idx=f,
            segment_length=L,
            T_match=budget_config.T_match,
            lambda_min=budget_config.lambda_min,
            grid_size=(budget_config.grid_H, budget_config.grid_W),
        )
        
        for i, feat in updated_prev.items():
            if feat is not None:
                anchors[i].prev_match_feature = feat
        
        if len(matched_indices) > budget_config.B_local_max:
            matched_indices = truncate_matched_by_relevance(
                matched_indices, anchors, budget_config.B_local_max,
                grid_size=(budget_config.grid_H, budget_config.grid_W),
                neighbor_radius=neighbor_radius,
            )
        
        remaining_budget = max(B - len(matched_indices), budget_config.M_global_min)
        global_indices = select_global_exploration_tokens(
            current_tokens, matched_indices, M_global=remaining_budget
        )
        
        retained_indices[f] = torch.cat([matched_indices, global_indices]).unique()
    
    return retained_indices
```

### 6.6 超参数

| 参数 | 含义 | 推荐值 | 说明 |
|------|------|--------|------|
| $T_{\text{match}}$ | 匹配有效性阈值 | 0.7 | 低于此值认为匹配失败 |
| $\lambda_{\min}$ | 最小锚定权重 | 0.3-0.5 | 防止完全依赖逐帧传递导致漂移 |
| $B$ | 每帧 token 预算 | $R \times N_v$ | 由 retention ratio 决定 |

---

## 7. 阶段 4：汇总与 LLM 推理

### 7.1 Token 汇总

将所有帧的保留 token 按时间顺序拼接：

$$H_v = [\text{Token}^{(1)}; \text{Token}^{(2)}; \dots; \text{Token}^{(F)}]$$

与 query 的 text token 拼接后送入 LLM：

$$H = [H_v; H_t]$$

### 7.2 位置编码处理

**重要工程细节**：剪枝后的 visual token 需要保持正确的位置编码。

- 对于使用绝对位置编码或 RoPE 的模型：每个保留的 token 应使用其在原始完整序列中的位置索引，而不是在剪枝后序列中的连续索引
- 对于使用 MRoPE 的模型（如 Qwen2.5-VL）：每个 token 的 (temporal, height, width) 三维位置应保持不变

```python
def assemble_pruned_tokens(
    visual_tokens: torch.Tensor,       # [F, N_v, d]
    retained_indices: List[torch.Tensor],  # 每帧保留的索引
    text_tokens: torch.Tensor,          # [N_t, d]
    grid_size: Tuple[int, int],
):
    """
    汇总剪枝后的 token，并构造正确的位置编码信息
    """
    all_tokens = []
    all_positions = []  # (frame_idx, row, col) for each token
    
    H, W = grid_size
    for f, indices in enumerate(retained_indices):
        tokens = visual_tokens[f][indices]  # [M_f, d]
        all_tokens.append(tokens)
        
        for idx in indices.tolist():
            row, col = idx // W, idx % W
            all_positions.append((f, row, col))
    
    pruned_visual = torch.cat(all_tokens, dim=0)          # [sum(M_f), d]
    combined = torch.cat([pruned_visual, text_tokens], dim=0)  # [sum(M_f) + N_t, d]
    
    return combined, all_positions
```

### 7.3 可选：Hybrid Compression（LLM 内部二次剪枝）

参考 FlashVID 的策略，可以在 LLM 前保留稍多的 token（如 $f_e = 1.25$ 倍预算），然后在 LLM 的第 $K_{\text{prune}}$ 层（如第 20 层）再做一次基于 attention 的 pruning：

1. 前 $K_{\text{prune}}$ 层处理 $f_e \cdot B$ 个 visual token
2. 在第 $K_{\text{prune}}$ 层，根据 text-to-visual attention 排序，只保留 top-$B$ 个 visual token
3. 后续层处理 $B$ 个 visual token

这一步是可选的，增加少量计算开销但可能提升性能。

---

## 8. 完整 Pipeline 伪代码

```python
def query_aware_anchor_propagation(
    video_frames: List[Image],
    query: str,
    vision_encoder: ViT,
    llm: LLM,
    R: float = 0.10,           # 唯一核心参数：retention ratio
) -> str:
    """
    完整的 Query-Aware Anchor Propagation Pipeline
    
    Args:
        video_frames: 采样后的视频帧列表
        query: 用户问题
        vision_encoder: 视觉编码器
        llm: 大语言模型
        R: 剪枝率，保留多少比例的 visual token (0.05 ~ 0.30)
    
    Returns: LLM 生成的答案字符串
    """
    
    # ===== 构造由 R 驱动的预算配置 =====
    budget_config = BudgetConfig(R=R, N_v=196)  # N_v 根据 ViT 确定
    print(budget_config.summary())
    # 例："R=10% | B=19 tokens/frame | neighbor_r=1 | anchor_max=13 | explore_min=3"
    
    # ===== 阶段 0: 视频编码 =====
    visual_tokens = encode_video(video_frames, vision_encoder)  # [F, N_v, d]
    text_tokens = encode_text(query, llm.tokenizer, llm.embed)  # [N_t, d]
    
    # ===== 阶段 1: 视频分段 =====
    segments = partition_video(visual_tokens, S_tau=0.9, M_s=8)
    # segments: List[Tuple[int, int]]，每个元素为 (start_frame, end_frame)
    
    # ===== 阶段 2 & 3: 逐 Segment 处理 =====
    F = len(video_frames)
    all_retained_indices = [None] * F
    
    for seg_start, seg_end in segments:
        segment_tokens = visual_tokens[seg_start:seg_end]  # [L, N_v, d]
        
        # 选择锚点帧
        anchor_local_idx = select_anchor_frame(segment_tokens)
        
        # 在该 segment 内进行锚点选择和传播
        seg_retained = process_segment(
            segment_tokens=segment_tokens,
            text_tokens=text_tokens,
            llm=llm,
            anchor_frame_local_idx=anchor_local_idx,
            budget_config=budget_config,    # 传入由 R 驱动的预算配置
        )
        
        # 写入全局索引
        for local_f, indices in enumerate(seg_retained):
            global_f = seg_start + local_f
            all_retained_indices[global_f] = indices
    
    # ===== 阶段 4: 汇总与 LLM 推理 =====
    combined_tokens, positions = assemble_pruned_tokens(
        visual_tokens, all_retained_indices, text_tokens,
        grid_size=(budget_config.grid_H, budget_config.grid_W)
    )
    
    # 统计实际压缩率
    total_kept = sum(len(idx) for idx in all_retained_indices)
    total_original = F * budget_config.N_v
    actual_R = total_kept / total_original
    print(f"Actual retention: {actual_R:.1%} ({total_kept}/{total_original} tokens)")
    
    # LLM 推理
    answer = llm.generate(combined_tokens, positions)
    
    return answer
```

---

## 9. 计算开销分析

### 9.1 各阶段开销

| 阶段 | 操作 | 复杂度 | 说明 |
|------|------|--------|------|
| 视频编码 | ViT 前向 | $O(F \cdot N_v^2 \cdot d)$ | 不可避免，与 baseline 相同 |
| 视频分段 | 帧级余弦相似度 | $O(F \cdot d)$ | 极小 |
| 锚点帧 token 选择 | LLM 前 K 层前向 | $O(S \cdot K \cdot (N_v + N_t)^2 \cdot d)$ | $S$ 为 segment 数，主要开销 |
| 后续帧传播 | 余弦相似度匹配 | $O((F - S) \cdot M \cdot N_v \cdot d)$ | $M$ 为锚点数，很轻量 |
| MMDP 选择 | 贪心扩展 | $O(F \cdot M_{\text{global}} \cdot N_v \cdot d)$ | 轻量 |
| LLM 推理 | LLM 全层前向 | $O(L \cdot (B \cdot F + N_t)^2 \cdot d)$ | 主要收益来源 |

### 9.2 相比 baseline 的开销对比

- **额外开销**：每个 segment 的锚点帧需要过 LLM 前 K 层（$K \ll L$），$S$ 个 segment 共需 $S$ 次
- **节省的开销**：LLM 推理时 token 数量从 $F \times N_v$ 降到 $F \times B$（$B \ll N_v$），由于 attention 是二次复杂度，节省非常显著
- **净效果**：只要 $R$ 不太高（如 $\leq 25\%$），节省的 LLM 推理开销远大于锚点选择的额外开销



## 10. 全部超参数汇总

### 10.1 核心控制参数（用户直接指定）

| 参数 | 含义 | 推荐范围 | 说明 |
|------|------|---------|------|
| $R$ | **Retention ratio（剪枝率）** | 0.05-0.30 | **唯一的核心参数**，控制保留多少比例的 visual token |

### 10.2 由 $R$ 自动派生的参数

| 参数 | 派生规则 | $R=0.10$ 时 | $R=0.25$ 时 | 说明 |
|------|---------|------------|------------|------|
| $B$ | $\lfloor R \times N_v \rfloor$ | 19 | 49 | 每帧 token 预算 |
| neighbor_radius | $R \geq 0.25 → 2$; $R \geq 0.10 → 1$; else $0$ | 1 | 2 | 邻域半径随 $R$ 缩放 |
| $B_{\text{local\_max}}$ | $\lfloor B \times r_{\max} \rfloor$ | 13 | 34 | 锚点预算上限 |
| $M_{\text{global\_min}}$ | $\max(M_{\text{floor}}, \lfloor B \times r_{\text{explore\_min}} \rfloor)$ | 3 | 7 | 探索预算下限 |

### 10.3 独立超参数（不随 $R$ 变化）

| 参数 | 所属阶段 | 含义 | 推荐值 | 调节建议 |
|------|---------|------|--------|---------|
| $S_\tau$ | 视频分段 | 场景切换阈值 | 0.9 | 静态视频可调高，动态视频可调低 |
| $M_s$ | 视频分段 | 最少 segment 数 | 8 | 长视频可增大 |
| $K$ | 锚点选择 | LLM 前向层数 | 2-4 | 越大 query-awareness 越强但开销越大 |
| $\alpha$ | 锚点选择 | 自适应阈值系数 | 1.0 | 控制"质量门槛"，与 $R$ 独立 |
| $T_{\text{match}}$ | 锚点传播 | 匹配有效性阈值 | 0.7 | 控制"匹配质量"，与 $R$ 独立 |
| $\lambda_{\min}$ | 锚点传播 | 最小锚定权重 | 0.3-0.5 | 越大越防漂移，越小越适应变化 |

### 10.4 预算分配参数（通常不需要调节）

| 参数 | 含义 | 推荐值 | 说明 |
|------|------|--------|------|
| $r_{\max}$ | 锚点预算占比上限 | 0.7 | 锚点最多占每帧预算的 70% |
| $r_{\text{explore\_min}}$ | 探索预算占比下限 | 0.15 | 保证探索 token 的最低覆盖 |
| $M_{\text{explore\_floor}}$ | 探索 token 绝对下限 | 3 | 极端压缩下也保留基本全局视野 |