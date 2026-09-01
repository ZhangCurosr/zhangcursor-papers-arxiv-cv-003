---
title: "WHAT-MEMORY-COMPOSITION-DOES-NOT-TELL-US-ABOUT-ANOMALY-DETEC"
source: https://arxiv.org/pdf/2608.23295v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 06:16:30"
field: "工业异常检测"
keywords: ["anomaly detection", "memory-based methods", "contaminated training", "coreset selection", "out-of-bag filtering", "industrial defect detection"]
innovations: ["揭示全局覆盖选择器对稀疏训练污染的放大效应（16-44×）", "提出 CLEANCON OOB 交叉图像门控方法在固定检测器下将污染降至≈0%", "建立污染度与检测性能非单调关系的悖论发现"]
benchmarks: ["MVTec AD", "VisA"]
---

# 论文速读：WHAT-MEMORY-COMPOSITION-DOES-NOT-TELL-US-ABOUT-ANOMALY-DETECTION

## 一句话总结
本文揭示了基于记忆的异常检测中全局覆盖选择器对稀疏训练污染的过度放大效应，并提出 CLEANCON（out-of-bag 交叉图像支持门控）方法，在固定表征、内存预算和推理规则下将最终内存污染降至近似零，且在全部 12 组匹配对比中提升 category-macro P-AP。关键发现是"最干净的内存不一定是最优内存"——污染度与检测性能并不单调相关。

## 研究问题与动机
- **核心问题**：基于记忆的异常检测器（如 PatchCore）通过 coreset 覆盖选择候选训练 patch 作为正常参考，但现有方法未检查被选中 patch 的几何稀有性是否足以保证其"正常可靠性"。
- **稀疏污染脆弱性**：当少量异常图像混入名义训练集时，其稀有 patch 易被全局最远优先遍历（Global FF）选入内存，形成污染参考。
- **关键疑问**：减少污染参考是否必然提升检测性能？内存纯度是否足以排序有效内存？
- **研究缺口**：现有去噪方法（SoftPatch、InReaCh、FUN-AD 等）改变检测器或训练流程，缺乏在固定检测器骨架下隔离"候选图像资格"影响的受控实验。

## 核心贡献（创新点）
- **揭示全局覆盖的污染放大效应**：发现 Global FF 在不同数据集-表征组合下将污染放大 16–44 倍，而限制为图像内竞争（Local FF）或随机/medoid 选择可将放大降至 1–3 倍。
- **提出 CLEANCON 受控干预框架**：首次固定表征、绝对内存大小、builder 和推理规则，仅通过 out-of-bag 交叉图像投影残差对候选图像进行排名，改变候选图像资格而不改变下游组件。
- **建立"纯度≠性能"的悖论发现**：通过保留率扫描证明，最低污染内存并非最高 P-AP，且性能可在污染上升时继续改善，挑战"去污即最优"的直觉。
- **提供半径证书分析**：从几何覆盖角度证明被选中的污染中心中至少有一部分是维持已实现覆盖半径所必需的（certify 0.58–6.32% 的污染中心）。
- **系统比较多种选择器与基线**：在 MVTec AD 和 VisA 的 4 种数据集-表征组合、3 个污染折叠下，与 SoftPatch、InReaCh、FUN-AD 等进行公平对比。

## 方法详解
### CLEANCON 框架设计
1. **Out-of-Bag 支持银行构建**：
   - 对每个 category 的 $n$ 张训练图像，采样 $B = 20$ 个支持银行 $I_b$，每个银行包含 $\lceil 0.20n \rceil$ 张图像。
   - 目标图像 $i$ 仅 against 不包含自身的银行 $B_i = \{b \mid i \notin I_b\}$，防止自引用偏差。

2. **软投影残差计算**：
   - 对描述符 $z_{i,p}^{\ell}$，在每个银行中检索 $k=5$ 个最近邻 $m_j$。
   - 计算 softmax 权重：$w_{i,p,j}^{\ell,b} = \frac{\exp(-d_j^2/\tau)}{\sum_{t=1}^k \exp(-d_t^2/\tau)}$。
   - 软投影：$\hat{z}_{i,p}^{\ell,b} = \sum_{j=1}^k w_{i,p,j}^{\ell,b} m_j$，残差：$r_{i,p}^{\ell,b} = \|z_{i,p}^{\ell} - \hat{z}_{i,p}^{\ell,b}\|_2$。

3. **残差聚合与图像评分**：
   - 层内跨银行取中值，跨深度平均：$\mu_{i,p} = \frac{1}{|\mathcal{L}|} \sum_{\ell \in \mathcal{L}} \text{median}_{b \in B_i} r_{i,p}^{\ell,b}$。
   - 图像分数取最高 0.5% patch 残差的均值：$a_i = \text{TopMean}_{0.5\%}(\{\mu_{i,p}\}_{p \in \mathcal{P}})$。

4. **图像级过滤**：
   - 在每个 category 内保留得分 $\leq$ 中位数的图像：$\mathcal{T} = \{i \mid a_i \leq \text{median}_j a_j\}$。
   - 过滤后所有保留图像的描述符进入原始 memory builder。

5. **匹配内存构建**：
   - 固定绝对目标 $K = 1\% |\mathcal{X}|$，CLEANCON 与 vanilla 使用相同 builder（Global FF）和推理规则（PatchCore 或 ProCon）。

### 污染度量
- 候选池流行度：$p_{\text{pool}} = |\mathcal{Q}|/|\mathcal{X}|$
- 最终内存占用：$p_{\text{mem}} = |\mathcal{M}_K \cap \mathcal{Q}|/K$
- 放大倍数：$A_K = p_{\text{mem}}/p_{\text{pool}}$

## 实验与结果
### 实验设置
- **数据集**：MVTec AD（15 categories）、VisA（12 categories）
- **表征**：WRN50（PatchCore 推理）、DINOv2（ProCon 推理）
- **污染协议**：Exact-Final 5% 图像级污染，3 个 No-Overlap 折叠
- **评估指标**：I-AUROC、P-AUROC、P-AP（主要定位指标）、AUPRO

### 核心结果
**污染放大效应（Table A3）**：
- Global FF 放大倍数：MVTec-WRN50 = 16.41×, VisA-WRN50 = 31.17×, MVTec-DINOv2 = 23.83×, VisA-DINOv2 = 43.57×
- Local FF 仅 2.55–3.42×，Random/Medoid ≈ 1×

**CLEANCON 匹配对比（Table 1）**：
| Setting | Vanilla $p_{\text{mem}}$ | CLEANCON $p_{\text{mem}}$ | Vanilla P-AP | CLEANCON P-AP | $\Delta$ P-AP |
|---------|--------------------------|---------------------------|--------------|---------------|---------------|
| MVTec-WRN50 | 6.481±0.370% | 0.000% | 57.704±2.033 | 61.141±0.433 | +3.438±1.625 |
| MVTec-DINOv2 | 9.462±0.611% | 0.000% | 69.342±0.995 | 72.573±0.373 | +3.231±0.645 |
| VisA-WRN50 | 3.678±0.500% | 0.003±0.003% | 39.345±0.586 | 39.725±0.309 | +0.380±0.429 |
| VisA-DINOv2 | 4.748±0.814% | 0.002±0.000% | 47.271±0.901 | 51.590±1.821 | +4.319±1.139 |

**保留率扫描悖论（Table 2）**：
- R80 在所有 12 组对比中超越 R50，且 11 组污染更高
- MVTec：P-AP 持续上升直到 R90，同时污染从 0% 升至 0.037%
- VisA：P-AP 在 R80 达峰值后 R90 下降，污染持续上升

**与基线对比（Table 3）**：
- CLEANCON-DINOv2 在 MVTec 达 I-AUROC 99.147、P-AP 72.255，优于 InReaCh、SoftPatch、FUN-AD
- VisA-WRN50 是唯一 CLEANCON 略逊于 SoftPatch-style control 的设置（40.061 vs 41.495）

## 相关工作脉络
- **PatchCore (Roth et al., 2022)**：基于近似贪心 coreset 的内存构建，本文揭示其对稀疏污染的敏感性，并提出 CLEANCON 作为后处理门控。
- **SoftPatch (Jiang et al., 2022)**：使用 patch-level 异常分数调整内存构建与推理，本文在其检测器骨架内实现匹配对比，证明 CLEANCON 在 5/6 设置中更优。
- **InReaCh (McIntosh & Branzan Albu, 2023)**：基于跨图像通道的方法，与 CLEANCON 在不同设计哲学下对比（Table 3）。
- **FUN-AD (Im et al., 2025)**：迭代重建内存银行，本文使用 No-Overlap 协议确保公平比较。
- **Coreset/覆盖选择理论 (Gonzalez, 1985; Charikar et al., 2001)**：farthest-first traversal 的几何性质解释 Global FF 对稀有点的偏好，与异常检测的耦合机制。
- **数据选择与主动学习 (Sener & Savarese, 2018; Mirzasoleiman et al., 2020)**：区分几何覆盖与下游性能的文献传统，本文在异常检测内存构建中验证这一分离。

## 局限性与未来方向
- **未解决稀有正常 patch 选择**：CLEANCON 仅去除高污染风险图像，未能恢复被全局覆盖排除的"稀有但有效"的正常锚点。
- **固定超参数评估**：保留率扫描仅在 R∈{50,60,70,80,90}% 五个点评估，未优化自适应保留策略。
- **半径证书的局限性**：仅 certify 0.58–6.32% 的被选污染中心为维持覆盖半径所必需，其余污染中心的必要性未证明。
- **VisA-WRN50 表现较弱**：CLEANCON 在该设置提升有限且略逊于 SoftPatch，可能反映 WRN50 表征对污染更敏感或 VisA 数据分布特性。
- **未来方向**：结合 OOB 分数调整 coverage builder（risk-aware coverage）、探索自适应保留阈值、研究正常源扩展与控制组的机制差异。

## 研究启发与可借鉴点
- **受控干预设计范式**：固定表征、builder、推理规则和绝对内存大小，仅改变候选图像资格的匹配比较方法，为去噪算法评估提供标准化框架。
- **OOB 交叉图像支持机制**：利用随机森林式思想（Breiman, 2001）构建不支持目标图像的银行集合，避免自引用偏差，可迁移至其他内存压缩场景。
- **污染放大倍数 $A_K$ 作为诊断指标**：量化选择器对稀有污染的偏好程度，可用于快速评估不同 coreset 构建策略的鲁棒性。
- **"纯度≠性能"的悖论启示**：提示在异常检测中，内存的语义多样性、局部流形结构可能与纯度同等重要，建议引入更丰富的内存质量度量。
- **保留率扫描实验设计**：通过系统变化候选集大小而非仅比较清洁/污染两种极端，揭示性能-污染非单调关系，值得在其他鲁棒性研究中借鉴。

## 关键术语表
- **Memory-based anomaly detection**：基于记忆的异常检测，通过将名义训练 patch 特征存入内存，以测试 patch 与最近邻记忆的距离作为异常分数。
- **Global FF (Farthest-First traversal)**：全局最远优先遍历，贪心选择距离当前内存最远的候选点，追求特征空间覆盖最大化。
- **CLEANCON**：本文提出的 Out-of-Bag Cross-image support CONtrol，通过交叉图像投影残差筛选候选图像，固定下游组件实现受控干预。
- **Contamination amplification ($A_K$)**：污染放大倍数，定义为内存污染比例与候选池污染比例之比，衡量选择器对稀疏污染的偏好强度。
- **No-Overlap evaluation**：无重叠评估协议，注入训练的异常图像从评估集中移除，确保测试干净。
- **Category-macro P-AP**：类别宏观平均像素平均精度，对每个类别计算 P-AP 后取平均，作为主要定位评估指标。
- **Achieved radius certificate**：已实现半径证书，从几何覆盖角度证明被选污染中心中至少有一部分对维持覆盖半径是必要的。
- **Soft projection residual**：软投影残差，目标描述符与其对 OOB 银行描述的 softmax 加权投影之间的 L2 距离。

## 可复现要素
- **数据集**：MVTec AD、VisA（公开数据集）
- **代码开源**：https://github.com/jw-chae/cleancon
- **关键超参**：
  - OOB 银行数 $B = 20$
  - 每银行图像数 $\lceil 0.20n \rceil$
  - 最近邻数 $k = 5$
  - 温度参数 $\tau$（从确定性描述符子样本距离尺度校准）
  - 最终内存大小 $K = 1\% |\mathcal{X}|$
  - 图像筛选阈值：类别内中位数
  - Patch 残差聚合：Top 0.5% mean
- **随机种子**：Approximate-greedy builder 使用 NumPy RNG seed 0
- **特征提取**：WRN50（224×224）、DINOv2（392×392），冻结预训练权重
- **评估协议**：3 个污染折叠，Exact-Final 5% 图像级污染，No-Overlap
