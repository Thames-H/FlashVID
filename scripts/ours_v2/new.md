# Functionally Equivalent Token Pruning（功能等效性Token剪枝）

# 一套理论驱动的视觉Token压缩框架

---

## 目录

1. 背景与动机
2. 问题定义
3. 核心度量：Functional Equivalence Score（FES）
4. 核心定理：误差分解
5. 最优剪枝策略的推导
6. 与现有方法的统一解释
7. 算法设计
8. 预期实验现象
9. 讨论与展望

---

## 1. 背景与动机

### 1.1 给非技术读者的背景

想象你在看一部两小时的电影，然后有人问你："电影里那个穿红衣服的人最后去了哪里？"

你的大脑不需要回忆电影的每一帧画面——你会自动忽略无关的风景镜头，聚焦在那个穿红衣服的人出现的关键场景上。这就是人类大脑天生擅长的"信息压缩"。

现在的AI视频理解模型（Video-LMM）面临类似的挑战：一段一小时的视频在1 FPS采样下会产生3600帧，每帧又会被切成很多小块（patch token），总共可能有几十万个token。把这些全部送给语言模型处理，计算成本是不可承受的。

**所以我们需要挑选出最关键的那些token，丢弃不重要的。**

但问题来了：**怎么判断哪些token重要，哪些不重要？**

目前的方法大多靠直觉和经验：有人均匀抽样，有人挑"看起来不一样"的帧，有人挑和问题最相关的帧。这些方法效果参差不齐，也缺乏理论解释。

本文提出一个全新的视角：**不是问"哪个token重要"，而是问"压缩后的token集合能不能在功能上替代原始的完整集合"。** 我们从这个视角出发，推导出一个有严格数学保证的最优剪枝策略。

### 1.2 给技术读者的背景

在多模态大语言模型（MLLM）中，视觉编码器将视频帧编码为视觉token序列 $H_v \in \mathbb{R}^{N \times d}$，随后作为KV cache供LLM的注意力层使用。当 $N$ 很大时（长视频场景下 $N$ 可达数万至数十万），计算和内存开销成为瓶颈。

现有的token压缩方法可分为以下几类：

- **均匀采样**：时间或空间上等间隔抽取token
- **基于注意力分数**：保留attention权重高的token（如FastV）
- **基于多样性**：合并相似token（如ToMe）
- **基于问题相关性**：保留与用户问题cosine相似度高的token

这些方法的核心问题是：**缺乏统一的理论框架来定义"什么是好的压缩"，以及"最优压缩策略是什么"。** 每种方法都基于不同的启发式直觉，无法相互比较，也无法判断距离理论最优还有多远。

---

## 2. 问题定义

### 2.1 直观版本

我们有一个视频，经过视觉编码器后变成了 $N$ 个token（比如 $N = 3600$）。我们的目标是：

> **从 $N$ 个token中选出 $k$ 个（比如 $k = 500$），使得语言模型用这 $k$ 个token回答问题时，和用全部 $N$ 个token回答问题时，效果尽可能接近。**

### 2.2 形式化版本

设视觉token的Key和Value矩阵分别为：

$$K = [k_1, k_2, \ldots, k_N]^\top \in \mathbb{R}^{N \times d}, \quad V = [v_1, v_2, \ldots, v_N]^\top \in \mathbb{R}^{N \times d}$$

用户问题经过embedding后为 $Q = [q_1, q_2, \ldots, q_{L_q}]^\top \in \mathbb{R}^{L_q \times d}$，其中 $L_q$ 是问题的token数。

我们要选择一个子集 $S \subseteq \{1, 2, \ldots, N\}$，大小 $|S| = k$，使得LLM使用子集 $(K_S, V_S)$ 时的行为尽可能接近使用完整 $(K, V)$ 时的行为。

**核心挑战**：如何量化"行为接近"？这就是FES要解决的问题。

---

## 3. 核心度量：Functional Equivalence Score（FES）

### 3.1 直观解释

LLM在处理视觉信息时，每个问题token会通过注意力机制"查阅"视觉token，得到一个汇总向量。FES衡量的就是：

> **用压缩后的token做注意力汇总，和用全部token做注意力汇总，得到的结果有多接近？**

如果很接近（FES接近1），说明压缩没有丢失对回答问题有用的信息。如果差距很大（FES远小于1），说明压缩掉了关键信息。

### 3.2 数学定义

对于单个问题token $q_l$，完整KV cache下的注意力输出为：

$$o_l = \text{Attn}(q_l, K, V) = \sum_{i=1}^{N} \alpha_{l,i} \cdot v_i$$

其中注意力权重为：

$$\alpha_{l,i} = \frac{\exp(q_l^\top k_i / \sqrt{d})}{\sum_{j=1}^{N} \exp(q_l^\top k_j / \sqrt{d})}$$

（解释：$q_l^\top k_i$ 衡量问题token $q_l$ 和视觉token $k_i$ 的匹配程度；除以 $\sqrt{d}$ 是为了控制数值范围；softmax保证所有权重之和为1。）

压缩后子集 $S$ 下的注意力输出为：

$$o_l^S = \text{Attn}(q_l, K_S, V_S) = \sum_{i \in S} \beta_{l,i} \cdot v_i$$

其中：

$$\beta_{l,i} = \frac{\exp(q_l^\top k_i / \sqrt{d})}{\sum_{j \in S} \exp(q_l^\top k_j / \sqrt{d})}$$

**注意 $\alpha_{l,i} \neq \beta_{l,i}$**，即使token $i$ 被保留了。这是因为softmax的分母变了——被丢弃的token原本分走的概率质量会被重新分配给保留的token。这是一个容易忽略但非常重要的点。

**FES的定义**：

$$\boxed{\text{FES}(S; K, V, Q) = \frac{1}{L_q} \sum_{l=1}^{L_q} \cos(o_l, \; o_l^S)}$$

其中 $\cos(a, b) = \frac{a^\top b}{\|a\| \cdot \|b\|}$ 是cosine相似度。

### 3.3 为什么用cosine而不是L2距离？

- **尺度不变性**：cosine只关心方向，不关心长度。attention输出的绝对大小受softmax温度等因素影响，但方向才是真正承载语义信息的部分。
- **有界性**：$\text{FES} \in [-1, 1]$，方便解释和比较。FES = 1意味着完美保持，FES = 0意味着信息完全无关。
- **与下游任务的关系**：LLM后续层中的残差连接和LayerNorm会标准化向量长度，因此方向的偏差比长度的偏差对最终输出的影响更大。

### 3.4 FES的基本性质

**性质1（有界性）**：$-1 \leq \text{FES}(S; K, V, Q) \leq 1$

**性质2（完美还原）**：当 $S = \{1, 2, \ldots, N\}$ 时，$\text{FES} = 1$

**性质3（单调性）**：若 $S_1 \subset S_2$，则在温和条件下 $\text{FES}(S_1) \leq \text{FES}(S_2)$

（保留越多token，FES越高。这符合直觉：信息越多，近似越好。）

---

## 4. 核心定理：误差分解

### 4.1 定理陈述

这是本框架最核心的理论结果。

**定理（Attention输出误差分解）**：对于任意问题token $q$，设完整输出为 $o = \sum_{i=1}^{N} \alpha_i v_i$，子集输出为 $o_S = \sum_{i \in S} \beta_i v_i$，定义被丢弃token的总概率质量为 $\delta = \sum_{i \notin S} \alpha_i$，则：

$$\boxed{o - o_S = \frac{1}{1 - \delta} \sum_{i \notin S} \alpha_i (v_i - o)}$$

### 4.2 完整证明

**第一步：展开误差。**

$$o - o_S = \sum_{i=1}^{N} \alpha_i v_i - \sum_{i \in S} \beta_i v_i$$

将第一项按子集内外拆分：

$$= \sum_{i \in S} \alpha_i v_i + \sum_{i \notin S} \alpha_i v_i - \sum_{i \in S} \beta_i v_i$$

$$= \underbrace{\sum_{i \notin S} \alpha_i v_i}_{\text{(A) 被丢弃token的贡献}} + \underbrace{\sum_{i \in S} (\alpha_i - \beta_i) v_i}_{\text{(B) 权重偏移导致的误差}}$$

**直觉解释**：误差由两部分构成——(A) 被丢掉的token本身携带的信息，(B) 因为丢掉了一些token导致softmax重新归一化，保留token的权重发生偏移。

**第二步：推导权重偏移量。**

保留token $i \in S$ 的原始权重为 $\alpha_i$，压缩后的权重为 $\beta_i$。

由softmax的定义：

$$\beta_i = \frac{\exp(q^\top k_i / \sqrt{d})}{\sum_{j \in S} \exp(q^\top k_j / \sqrt{d})}$$

而原始权重可以写成：

$$\alpha_i = \frac{\exp(q^\top k_i / \sqrt{d})}{\sum_{j=1}^{N} \exp(q^\top k_j / \sqrt{d})}$$

两者的分子相同，分母的关系为：

$$\sum_{j \in S} \exp(q^\top k_j / \sqrt{d}) = \sum_{j=1}^{N} \exp(q^\top k_j / \sqrt{d}) - \sum_{j \notin S} \exp(q^\top k_j / \sqrt{d})$$

$$= \sum_{j=1}^{N} \exp(q^\top k_j / \sqrt{d}) \cdot \left(1 - \sum_{j \notin S} \alpha_j\right) = \sum_{j=1}^{N} \exp(q^\top k_j / \sqrt{d}) \cdot (1 - \delta)$$

因此：

$$\beta_i = \frac{\alpha_i}{1 - \delta}$$

**关键结论**：压缩后，每个保留token的权重被统一放大了 $\frac{1}{1-\delta}$ 倍。这是因为被丢弃token释放的概率质量 $\delta$ 被均匀地重新分配了。

**第三步：代入权重偏移。**

$$\alpha_i - \beta_i = \alpha_i - \frac{\alpha_i}{1 - \delta} = \alpha_i \cdot \frac{(1-\delta) - 1}{1-\delta} = \frac{-\alpha_i \delta}{1-\delta}$$

将此代入误差(B)项：

$$\sum_{i \in S} (\alpha_i - \beta_i) v_i = \frac{-\delta}{1-\delta} \sum_{i \in S} \alpha_i v_i$$

注意到 $\sum_{i \in S} \alpha_i v_i = o - \sum_{i \notin S} \alpha_i v_i$，所以：

$$\text{(B)} = \frac{-\delta}{1-\delta} \left(o - \sum_{i \notin S} \alpha_i v_i\right)$$

**第四步：合并(A)和(B)。**

$$o - o_S = \sum_{i \notin S} \alpha_i v_i + \frac{-\delta}{1-\delta} \left(o - \sum_{i \notin S} \alpha_i v_i\right)$$

$$= \sum_{i \notin S} \alpha_i v_i + \frac{-\delta}{1-\delta} o + \frac{\delta}{1-\delta} \sum_{i \notin S} \alpha_i v_i$$

$$= \left(1 + \frac{\delta}{1-\delta}\right) \sum_{i \notin S} \alpha_i v_i - \frac{\delta}{1-\delta} o$$

$$= \frac{1}{1-\delta} \sum_{i \notin S} \alpha_i v_i - \frac{\delta}{1-\delta} o$$

$$= \frac{1}{1-\delta} \left(\sum_{i \notin S} \alpha_i v_i - \delta \cdot o\right)$$

注意到 $\delta = \sum_{i \notin S} \alpha_i$，所以 $\delta \cdot o = \sum_{i \notin S} \alpha_i \cdot o$，最终：

$$o - o_S = \frac{1}{1-\delta} \sum_{i \notin S} \alpha_i (v_i - o) \quad \blacksquare$$

### 4.3 定理的意义

这个结果揭示了一个非常优雅的结构：**误差完全由被丢弃token的"偏差"决定**。

这里的"偏差"指的是 $v_i - o$，即第 $i$ 个被丢弃token的value向量与全局注意力输出 $o$（所有token的加权平均）之间的差异。

这意味着：

1. **如果被丢弃的token的value接近全局平均 $o$**：即 $\|v_i - o\| \approx 0$，那么即使它的attention权重 $\alpha_i$ 很大，误差也很小。直觉：这个token虽然被关注，但它说的和"大家的共识"一样，丢掉它不影响结论。

2. **如果被丢弃的token的attention权重很小**：即 $\alpha_i \approx 0$，那么即使它的value很独特（$\|v_i - o\|$ 很大），误差也很小。直觉：这个token虽然独特，但没人关注它，丢掉它也没影响。

3. **只有同时被高度关注（$\alpha_i$ 大）且信息独特（$\|v_i - o\|$ 大）的token被丢弃时，才会造成大误差。** 这就是"重要token"的精确数学定义。

---

## 5. 最优剪枝策略的推导

### 5.1 优化目标

我们要选子集 $S$（大小为 $k$）使FES最大化，等价于最小化误差的平方范数：

$$S^* = \arg\min_{|S|=k} \|o - o_S\|^2 = \arg\min_{|S|=k} \frac{1}{(1-\delta)^2} \left\|\sum_{i \notin S} \alpha_i(v_i - o)\right\|^2$$

由于 $\delta$ 本身也依赖于 $S$ 的选择，严格优化比较复杂。但在 $\delta$ 较小的情况下（即压缩比不太极端），$\frac{1}{(1-\delta)^2}$ 近似为常数，优化简化为：

$$S^* \approx \arg\min_{|S|=k} \left\|\sum_{i \notin S} r_i\right\|^2$$

其中 $r_i = \alpha_i(v_i - o)$ 是第 $i$ 个token的**加权残差向量**。

### 5.2 简化策略：独立打分

将目标展开：

$$\left\|\sum_{i \notin S} r_i\right\|^2 = \sum_{i \notin S} \|r_i\|^2 + \sum_{\substack{i,j \notin S \\ i \neq j}} r_i^\top r_j$$

在高维空间（$d$ 很大，如 $d = 1024$ 或 $4096$）中，不同token的残差向量方向趋于正交，交叉项 $r_i^\top r_j \approx 0$。此时：

$$\left\|\sum_{i \notin S} r_i\right\|^2 \approx \sum_{i \notin S} \|r_i\|^2 = \sum_{i \notin S} \alpha_i^2 \|v_i - o\|^2$$

要最小化被丢弃token的 $\alpha_i^2 \|v_i - o\|^2$ 之和，等价于**保留 $\alpha_i \|v_i - o\|$ 最大的 $k$ 个token**。

**最优重要性分数**：

$$\boxed{s_i = \alpha_i \cdot \|v_i - o\|}$$

**保留分数最大的 $k$ 个token，丢弃分数最小的token。**

### 5.3 分数的直觉解释

$s_i$ 是两个因子的乘积：

| 因子 | 含义 | 类比 |
|------|------|------|
| $\alpha_i$ | 问题对该token的关注程度 | 这个证人被法官传唤的次数 |
| $\|v_i - o\|$ | 该token携带的独特信息量 | 这个证人的证词与其他人有多不同 |

一个证人只有同时满足"被频繁传唤"和"证词独特"时，才是关键证人。不常被传唤的证人丢了没事；证词和大家一样的证人丢了也没事。

### 5.4 完整策略：考虑交叉项（贪心算法）

当交叉项不可忽略时（例如低维空间或token之间高度相关），简单打分不再最优。此时需要贪心算法：

**算法思路**：从"丢弃全部token"的状态开始，逐步"保留"token。每一步选择一个token保留，使得剩余的丢弃集合的误差 $\|\sum_{i \notin S} r_i\|^2$ 下降最多。

等价地，可以从"保留全部token"开始，逐步"丢弃"token。每一步选择丢弃后误差增加最小的token丢弃。

设当前的累计误差向量为 $e = \sum_{i \notin S_{\text{current}}} r_i$，则丢弃token $i$ 后新的误差向量为 $e + r_i$，范数为 $\|e + r_i\|^2$。选择使 $\|e + r_i\|^2$ 最小的 $i$ 丢弃。

$$\|e + r_i\|^2 = \|e\|^2 + 2e^\top r_i + \|r_i\|^2$$

所以每步只需计算 $2e^\top r_i + \|r_i\|^2$，复杂度为 $O(d)$，总复杂度 $O((N-k) \cdot N \cdot d)$。

**贪心算法的理论保证**：误差函数在取负后是次模函数（边际收益递减），贪心算法可以达到 $(1 - 1/e) \approx 0.63$ 的近似比。即贪心解的误差不会超过最优解误差的 $\frac{1}{1-1/e} \approx 1.58$ 倍。

---

## 6. 与现有方法的统一解释

本框架最优雅的地方在于，它可以将现有的各种token剪枝方法解释为最优分数 $s_i = \alpha_i \cdot \|v_i - o\|$ 的特殊情况或近似。

### 6.1 纯注意力分数方法（FastV等）

这类方法只用 $\alpha_i$ 来选择token。

**在我们的框架下**：等价于假设所有token的独特性相同，即 $\|v_i - o\| = c$（常数）。此时 $s_i = c \cdot \alpha_i$，排序等价于按 $\alpha_i$ 排序。

**何时最优**：当所有视觉token的value向量与全局均值的距离都差不多时。
**何时失效**：当某些token的value非常独特（远离 $o$）但attention权重中等时，会被错误丢弃。

### 6.2 纯多样性方法（ToMe等）

Token Merging合并cosine相似度高的token对，间接保留了多样性。

**在我们的框架下**：近似等价于最大化保留token的value空间覆盖率，间接地使 $\|v_i - o\|$ 项在保留集合上分布均匀。但完全忽略了 $\alpha_i$，可能保留了很多"多样但没人关注"的token。

**何时最优**：当attention分布接近均匀时，所有 $\alpha_i \approx 1/N$，此时多样性就是决定性因素。
**何时失效**：当attention高度集中在少数token时，多样性无关紧要。

### 6.3 问题引导方法（cosine相似度选择）

这类方法按问题embedding和视觉token的cosine相似度选择。

**在我们的框架下**：cosine相似度大致与softmax前的logit $q^\top k_i$ 正相关，因此粗略近似于 $\alpha_i$。但由于没有经过softmax（缺少归一化和竞争效应），且忽略了 $\|v_i - o\|$ 项，是一个双重近似。

**何时最优**：当key空间和value空间高度对齐，且softmax温度较高（分布接近线性）时。
**何时失效**：当softmax使注意力分布变得很尖锐（赢家通吃）时，线性近似失效。

### 6.4 均匀采样

**在我们的框架下**：不使用任何信息，随机选择。期望上，均匀采样的 $\sum_{i \notin S} \alpha_i^2 \|v_i - o\|^2$ 正比于总体均值乘以被丢弃的数量，没有利用分数分布的不均匀性。

**何时最优**：当所有token的重要性分数 $s_i$ 完全相同时（实际中几乎不会发生）。

### 6.5 统一视角表格

| 方法 | 使用的分数 | 最优策略的近似 | 忽略的信息 |
|------|-----------|---------------|-----------|
| FastV | $\alpha_i$ | $s_i = \alpha_i \cdot c$ | value的独特性 $\|v_i - o\|$ |
| ToMe | 多样性（间接） | 最大化 $\|v_i - o\|$ 的覆盖 | 注意力权重 $\alpha_i$ |
| 问题cosine | $q^\top k_i$（无softmax） | $s_i \approx \tilde{\alpha}_i \cdot c$ | softmax竞争 + value独特性 |
| 均匀采样 | 无 | $s_i = c$（常数） | 全部信息 |
| **本方法** | $\alpha_i \cdot \|v_i - o\|$ | **完整最优策略** | **无（在高维近似下）** |

---

## 7. 算法设计

### 7.1 简单版算法（独立打分）

```
输入：Key矩阵 K ∈ R^{N×d}, Value矩阵 V ∈ R^{N×d},
      问题embedding Q ∈ R^{L_q×d}, 保留数量 k
输出：保留的token索引集合 S

步骤1：计算注意力权重
  对每个问题token q_l：
    logits_l = q_l · K^T / √d          # [N]维向量
    α_l = softmax(logits_l)              # [N]维向量
  α = mean(α_1, α_2, ..., α_{L_q})      # 对所有问题token取平均

步骤2：计算全局输出
  o = α · V = Σ_i α_i · v_i             # [d]维向量

步骤3：计算每个token的重要性分数
  对每个token i：
    s_i = α_i × ||v_i - o||              # 标量

步骤4：选择top-k
  S = argtop-k(s_1, s_2, ..., s_N)      # 选分数最大的k个

返回 S
```

**时间复杂度**：$O(L_q \cdot N \cdot d + N \cdot d)$，其中第一项是attention计算，第二项是分数计算。

**空间复杂度**：$O(N \cdot d)$

### 7.2 贪心版算法（考虑交叉项）

```
输入：同上
输出：保留的token索引集合 S

步骤1-2：同简单版，得到 α 和 o

步骤3：计算加权残差向量
  对每个token i：
    r_i = α_i × (v_i - o)               # [d]维向量

步骤4：初始化
  误差向量 e = Σ_{i=1}^{N} r_i          # 所有token都在丢弃集中
  已丢弃集合 D = {1, 2, ..., N}

步骤5：逐步"保留"token（共k步）
  重复k次：
    对每个候选 i ∈ D：
      # 如果保留i，误差向量变为 e - r_i
      score_i = ||e - r_i||^2
    best = argmin score_i
    D = D \ {best}                       # 从丢弃集中移除
    e = e - r_{best}                     # 更新误差向量

步骤6：
  S = {1,...,N} \ D

返回 S
```

**时间复杂度**：$O(k \cdot N \cdot d)$

### 7.3 计算FES分数

```
输入：完整的 K, V ∈ R^{N×d}，
      子集索引 S，问题 Q ∈ R^{L_q×d}
输出：FES分数

步骤1：计算完整输出
  对每个 q_l：
    α_l = softmax(q_l · K^T / √d)
    o_l = α_l · V                        # [d]维向量

步骤2：计算子集输出
  对每个 q_l：
    β_l = softmax(q_l · K_S^T / √d)     # 注意分母只在S上求和
    o_l^S = β_l · V_S                    # [d]维向量

步骤3：计算FES
  FES = (1/L_q) × Σ_l cos(o_l, o_l^S)

返回 FES
```

### 7.4 实用建议：两阶段策略

在实际部署中，推荐以下两阶段策略：

**阶段一：离线粗筛（视频上传时预计算，与问题无关）**

使用value空间覆盖率作为代理指标，选出候选集 $C$（比如从3600个token选500个）。
这一步不需要问题信息，可以在视频编码后立即执行并缓存。

**阶段二：在线精选（用户提问时实时计算，与问题相关）**

在候选集 $C$ 上使用完整的FES策略（$s_i = \alpha_i \|v_i - o\|$），选出最终的 $k$ 个token。
由于候选集已经很小（500而非3600），这一步很快。

---

## 8. 预期实验现象

### 8.1 FES与下游准确率的相关性

**预期**：在同一个视频上，用不同压缩方法计算FES，画FES vs. 下游任务准确率（如Video QA准确率）的散点图，应该呈现**强正相关**（Pearson相关系数 > 0.85）。

**意义**：证明FES是一个好的代理指标，可以不跑LLM推理就预判压缩质量。

**为什么这是预期的**：根据我们的理论推导，FES直接控制了LLM隐状态的偏差上界，而隐状态的偏差通过Lipschitz传播最终影响输出logits。

### 8.2 不同压缩方法的FES对比

**预期**：在相同的token预算（相同的 $k$）下，各方法的FES排序应该为：

$$\text{FES}_{\text{ours}} > \text{FES}_{\text{FastV}} > \text{FES}_{\text{uniform}} > \text{FES}_{\text{random}}$$

ToMe的排名取决于具体场景——在attention分布较均匀的视频上可能接近我们的方法，在attention高度集中的视频上会显著落后。

### 8.3 误差分解的验证

**预期**：计算实际误差 $\|o - o_S\|$ 和理论预测 $\frac{1}{1-\delta}\|\sum_{i \notin S} \alpha_i(v_i - o)\|$，两者应该**精确相等**（这是恒等式，不是近似）。

进一步，分解误差为"丢弃贡献"和"权重偏移"两项，验证在不同压缩比下两项的相对大小变化。

### 8.4 逐层FES分析

**预期**：LLM不同层的FES对同一压缩策略的敏感度不同。

- 浅层的attention通常更分散（高熵），FES对剪枝更敏感，需要保留更多token
- 深层的attention更集中（低熵），FES对剪枝更鲁棒，可以激进压缩

这为逐层自适应分配token预算提供了理论和实验依据。

### 8.5 高维近似的验证

**预期**：随着hidden dimension $d$ 的增大，简单版（独立打分）和贪心版的性能差距缩小。因为高维空间中交叉项 $r_i^\top r_j \to 0$，简单版的近似更加精确。

在 $d = 4096$（常见LLM维度）时，简单版应该已经非常接近贪心版。

### 8.6 注意力集中度与压缩潜力

**预期**：对于不同类型的视频内容：

- **信息密集型**（如多人对话、复杂场景）：attention分布分散，$\delta$ 值大，FES下降快，压缩空间有限
- **信息稀疏型**（如风景、静态场景）：attention集中在少数关键帧，$\delta$ 值小，FES保持高值，可以大幅压缩

---

## 9. 讨论与展望

### 9.1 循环依赖问题

计算分数 $s_i = \alpha_i \|v_i - o\|$ 需要先算 $\alpha_i$ 和 $o$，而这需要完整的KV cache做一次前向。

**解决方案**：这个前向只需要在prefill阶段做一次。对于视频QA场景，prefill是一次性成本，之后的decode阶段（逐token生成答案）是重复成本。剪枝减少的是decode阶段每一步的attention计算量，因此只要答案长度超过一定阈值，总计算量就会下降。

### 9.2 多层KV cache的处理

实际LLM有多层，每层的KV cache不同。两种策略：

- **共享选择**：只用第一层（或中间某层）的attention计算分数，所有层用同一个子集。简单但可能次优。
- **逐层选择**：每层独立选择子集。最优但存储和计算开销更大。

### 9.3 多头注意力的处理

每个attention head的 $\alpha_i$ 分布不同。可以对所有head的 $s_i$ 取平均，或者每个head独立选择不同的子集。后者更灵活但实现更复杂。

### 9.4 与可学习方法的关系

本框架给出的是**免训练的最优策略**。而Hour-LLaVA的MemAug是一个可学习的方法——通过cross-attention隐式地实现了类似的功能。可以预期，在充分训练后，MemAug的效果应该优于我们的免训练方法（因为它可以端到端优化），但我们的方法不需要任何训练，且有理论保证。

两者是互补的：我们的理论框架解释了为什么MemAug有效（它在隐式地最大化FES），同时提供了一个不需要训练的强baseline。

### 9.5 超越视频理解

FES框架不局限于视频token剪枝。它适用于任何需要压缩KV cache的场景：

- 长文本的KV cache压缩
- 多图像理解中的图像token选择
- RAG系统中检索文档的压缩

核心思想是通用的：**好的压缩 = 压缩后的KV cache在功能上等效于原始KV cache。**

---

## 附录：符号表

| 符号 | 含义 |
|------|------|
| $N$ | 视觉token总数 |
| $k$ | 保留的token数 |
| $d$ | 隐藏层维度 |
| $L_q$ | 问题token数 |
| $K, V$ | 完整的Key和Value矩阵 |
| $K_S, V_S$ | 子集 $S$ 对应的Key和Value |
| $\alpha_i$ | token $i$ 在完整KV下的注意力权重 |
| $\beta_i$ | token $i$ 在压缩KV下的注意力权重 |
| $\delta$ | 被丢弃token的总注意力质量 |
| $o$ | 完整KV下的注意力输出 |
| $o_S$ | 压缩KV下的注意力输出 |
| $s_i$ | token $i$ 的重要性分数 |
| $r_i$ | token $i$ 的加权残差向量 $\alpha_i(v_i - o)$ |
| FES | Functional Equivalence Score |

---

*本文档为研究讨论稿，涵盖从动机、理论推导到方法设计的完整框架。*