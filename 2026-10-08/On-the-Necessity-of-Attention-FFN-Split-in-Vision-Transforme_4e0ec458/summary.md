---
title: "On-the-Necessity-of-Attention-FFN-Split-in-Vision-Transforme"
source: https://arxiv.org/pdf/2610.10303v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 17:19:40"
field: "视觉Transformer架构设计"
keywords: ["Vision Transformer", "inductive bias", "Attention-FFN dichotomy", "architecture analysis", "unified module", "rank collapse", "small-scale efficiency"]
innovations: ["提出AttenFeed模块统一Attention与FFN功能", "设计uViT作为消除Attention-FFN分离的控制组架构", "首次系统证明Attention-FFN严格分离在小模型规模下构成有害归纳偏置"]
benchmarks: ["ImageNet-1k", "Places365", "iNaturalist 2021", "CIFAR-10/100", "SVHN", "STL-10", "ImageNet-Seg", "Pascal-VOC"]
---

# 论文速读：On the Necessity of Attention–FFN Split in Vision Transformers

## 一句话总结
本文系统探究了 Vision Transformer 中交替的 Attention–FFN 结构是否为必要设计，提出统一的 AttenFeed 模块与 uViT 架构，通过理论分析与大规模实验证明：**Attention–FFN 严格分离在小模型规模下反而构成有害的归纳偏置，阻碍有效学习**。

## 研究问题与动机
1. **核心问题**：Transformer 最基础的结构设计——交替排列的 Attention 与 FFN——所施加的归纳偏置（inductive bias）从未被系统性分析过，其"必要性"仍是黑盒。
2. **MoE 研究的启示**：SkipGPT 等工作发现，连续的同构层交互（如 Attention–Attention 或 FFN–FFN）偶尔比标准交替模式更有效，暗示当前"硬编码"的交替结构可能并非不可移除。
3. **小模型下的参数分配困境**：标准 ViT 在小参数预算下将参数刚性分配到 Attention 与 FFN 两个固定功能模块，无法根据任务需求动态调配容量，导致小模型性能受限。
4. **缺乏统一分析工具**：现有工作（如 MLP-Mixer、GAU）虽涉及结构简化，但均未提供可同时统一表征 Attention 与 FFN 功能的分析模块，无法直接对"分离 vs 统一"进行受控比较。

## 核心贡献（创新点）
1. **提出 AttenFeed 模块**：在标准 Attention 的值投影后引入非线性激活函数（GELU），使其在数学与实证上同时泛化 Attention 与 FFN 的功能特性。与 GAU 等先前工作的本质区别在于：GAU 旨在弥补线性注意力的性能缺陷，而 AttenFeed 以最小化归纳偏置、系统性分析 Attention–FFN 分离必要性为目标。
2. **设计 uViT 统一架构**：用全 AttenFeed 模块序列完全替代标准 Transformer 的交替 Attention–FFN 结构，使 uViT 成为研究该结构分离必要性的完美对照组。与 SkipGPT 等动态路由方法的本质区别在于：uViT 是确定性的架构简化，而非训练时动态选择是否包含某类层。
3. **揭示小尺度下的归纳偏置代价**：通过 uViT 与 ViT 的多尺度对比实验发现，Attention–FFN 严格分离在小模型规模（Small）下显著阻碍性能，但随着模型容量增大（Base/Large）差距逐渐缩小直至边缘化，首次为"何时需要交替结构"提供了定量答案。

## 方法详解
**AttenFeed 模块**：基于 UMoE 将标准 Attention 的 OV circuit 解释为无激活函数的 FFN 这一洞察，仅在值投影后加入非线性激活 σ（默认 GELU）：

$$\mathrm{AttenFeed}^{\ell,h}(\mathbf{X}^{\ell-1})_i := \sum_j a_{i,j}^{\ell,h} \, \sigma(\mathbf{x}_j^{\ell-1} \mathbf{W}_V^{\ell,h}) \, \mathbf{W}_O^{\ell,h}$$

其中 $a_{i,j}^{\ell,h}$ 为标准 softmax 注意力权重（QK circuit），$\mathbf{W}_V$ 和 $\mathbf{W}_O$ 可同时被解释为 Attention 的价值/输出矩阵或 FFN 的输入/输出矩阵。

**三个数学命题**：
- **Proposition 4.1**：当 AttenFeed 头为 self-focused（即 $a_{i,i} \to 1$）时，退化为位置-wise FFN。
- **Proposition 4.2**：当 GELU 预激活输入趋近于 0 时，GELU 近似恒等映射，AttenFeed 退化为标准 Attention。
- **Proposition 4.3**：在 Prop 4.1 的 self-focused 条件下，仍存在一组参数化使 AttenFeed 近似标准 Attention（更宽松的条件）。

**uViT 架构**：每层仅含一个 AttenFeed 模块（无独立 FFN），残差更新：

$$\mathbf{x}_i^{\ell} = \mathbf{x}_i^{\ell-1} + \sum_h \mathrm{AttenFeed}^{\ell,h}(\mathrm{LN}^{\ell}(\mathbf{X}^{\ell-1}))_i$$

为保持参数量公平对比，uViT 通过增加层数和模型维度来补偿移除 FFN 后的参数减少（约减少 3 倍），具体配置见 Table 1（如 ViT-S 21M vs uViT-S 20M；ViT-L 303M vs uViT-L 289M）。

## 实验与结果
**评估基线与数据集**：ImageNet-1k、Places365、iNaturalist 2021（预训练）；CIFAR-10/100、SVHN、STL-10、ImageNet-Seg、Pascal-VOC（下游/分析）。

**主要预训练结果（Table 5）**：
- **Small 规模优势显著**：uViT-S 在 ImageNet-1k 上达 **67.11%**，相比 ViT-S（53.64%）**提升 +13.47 个百分点**；Places365（50.25% vs 44.63%，+5.62%）；iNaturalist（66.81% vs 52.82%，+13.99%）。
- **Base 规模差距缩小**：uViT-B 74.90% vs ViT-B 74.22%，仅 **+0.68%**。
- **Large 规模基本持平**：uViT-L 77.18% vs ViT-L 77.33%，**-0.15%**（uViT-L 在 iNaturalist 上略低）。

**传输学习（Table 4）**：uViT 在所有数据集的线性探测（Linear Probing）下均超越 ViT，全微调（Full Fine-tuning）下大多数场景也更强，表明 uViT 表征更具通用性、归纳偏置更弱。

**表征秩分析（Table 2, 6）**：uViT 在所有配置的 pre-head rank 上均等于或高于仅含 Attention 的 xViT（去除 FFN 的变体），且伴随更高测试精度，证实 AttenFeed 继承了 FFN 的保秩特性。

**注意力图分割（Table 3）**：uViT 的注意力图分割性能略低于 ViT 但差距不大，uViT-L 在 ImageNet-Seg mAP 上甚至超过 ViT-L（0.631 vs 0.666 差距极小），证明 AttenFeed 保留了足够的注意力特性。

**参数空间分析（Figure 3, 5, 6）**：uViT 的 $\mathbf{W}_V$ 和 $\mathbf{W}_O$ 值分布在标准 Attention 与 FFN 参数分布之间，呈现一致的过渡态特征，且该规律在 Small/Base/Large 各规模下均成立。

## 相关工作脉络
1. **MLP-Mixer / ResMLP**：用纯 MLP 替换 self-attention 以最小化 token mixing 的归纳偏置，在大规模数据上表现优异。本文定位差异：MLP-Mixer 用 MLP 替代 Attention，而本文保留两者功能但消除硬性分离，是"融合"而非"替换"的思路。
2. **Swin Transformer**：通过分层结构与 shifted window attention 重新引入局部性与平移不变性归纳偏置。本文定位差异：Swin 增强特定偏置以提升性能，本文则研究移除最基础的功能分离偏置本身。
3. **UMoE（Yang et al., 2025）**：用共享专家统一 Attention 与 FFN。本文定位差异：UMoE 是工程上的统一方案，本文将其 OV circuit 解释为无激活 FFN 作为理论起点，并在此基础上系统化验证分离偏置的影响。
4. **SkipGPT（Zhao et al., 2025）**：训练路由器条件性地跳过 Attention 或 FFN 块，发现连续同构交互有时更有效。本文定位差异：SkipGPT 是训练时的动态稀疏化策略，本文是确定性的架构简化，提供更干净的控制变量对照。
5. **GAU（Hua et al., 2022）**：融合 GLU 与 Attention，但主要针对线性注意力的性能缺陷进行补偿。本文定位差异：GAU 以性能提升为目标，AttenFeed 以"系统性分析归纳偏置必要性"为目标，是研究工具而非纯性能优化。
6. **Attention-only Transformer 系列**（Huben & Morris, 2023; Wang et al., 2025）：探索仅用 Attention 实现 FFN 功能的可能性。本文定位差异：这些工作关注"能否用 Attention 替代 FFN"，本文关注"分离是否必要"，通过统一模块而非单方面替代进行研究。

## 局限性与未来方向
1. **仅限视觉域**：所有实验均在图像数据集上完成，Attention–FFN 分离的必要性在语言建模、音频处理等其他模态中可能表现不同，需跨域验证。
2. **使用标准训练配方**：为公平对比，uViT 与 ViT 使用完全相同的训练超参数与正则化策略；uViT 可能存在更优的专属优化景观（如不同正则化、学习率调度或数据增强），未做针对性调优。
3. **未探索动态混合比**：AttenFeed 中 GELU 引入的"注意力 vs FFN 行为"比例是隐式学习的，作者未系统分析或控制这一混合比，也未探索手动调度或可学习门控的变体。
4. **推理效率未深入分析**：uViT 每层参数量相近但层数更多，未详细讨论推理延迟、显存占用等实际部署指标。

## 研究启发与可借鉴点
1. **控制变量法的架构分析范式**：以 uViT 作为 ViT 的"对照组"，通过移除单一结构差异（Attention–FFN 分离）来孤立评估其影响，这一实验设计思路可迁移至其他架构组件（如 LayerNorm 顺序、残差连接位置等）的必要性分析。
2. **小模型应优先考虑架构灵活性**：结论提示在参数量受限的场景（端侧部署、边缘计算）中，刚性结构偏置的代价更高，可优先考虑融合型模块设计而非直接套用标准 ViT。
3. **Rank 分析作为表征质量诊断工具**：通过 SVD 计算的 normalized rank（pre-head / post-head）能敏锐反映 FFN 对避免 token uniformity 的关键作用，这一分析手段可用于快速诊断其他 Transformer 变体的表征健康度。
4. **Attention 图的可解释性延续性验证**：AttenFeed 模块既能保持类似 Attention 的分割性能，又能发挥 FFN 的保秩功能，提示未来的统一模块设计可同时追求多项可验证的中间性质，而非仅关注端到端准确率。
5. **可探索的延伸方向**：将 AttenFeed 思想应用于 LLM（如结合 MoE 架构做动态混合）、探索不同激活函数（SiLU、ReLU² 等）对混合行为的影响、开发 uViT 专属训练策略以实现更大规模的性能突破。

## 关键术语表
- **AttenFeed 模块**：在标准 Attention 值投影后引入非线性激活（GELU），使其在数学与实证上同时泛化 Attention 与 FFN 功能的统一模块。
- **uViT（Unified Vision Transformer）**：由全 AttenFeed 模块序列构成的 Vision Transformer 变体，消除了标准 ViT 中交替的 Attention–FFN 结构分离。
- **归纳偏置（Inductive Bias）**：架构设计中对解空间的先验约束，可加速小数据训练但也可能限制模型在充足数据下的性能上限。
- **Rank Collapse（秩坍缩）**：随网络深度增加，不同 token 的隐式表征趋向相关、表征空间有效维度降低的现象，FFN 的非线性激活有助于缓解此问题。
- **OV Circuit / QK Circuit**：分别指 Attention 中的值-输出变换路径（$\mathbf{x} \mathbf{W}_V \mathbf{W}_O$）和注意力权重计算路径（Query-Key 相似度），后者决定前者的空间混合权重。
- **Self-focused Attention**：注意力权重对角占优的状态（$a_{i,i} \to 1$），此时 AttenFeed 退化为位置-wise FFN。
- **Linear Probing**：冻结预训练模型 backbone，仅训练顶层分类器以评估表征迁移能力的标准协议。
- **Normalized Rank**：通过 SVD 计算的有效秩与最大可能秩之比（以 95% 方差解释阈值为标准），用于量化表征空间的维度丰富程度。

## 可复现要素
- **数据集**：ImageNet-1k、Places365、iNaturalist 2021、CIFAR-10/100、SVHN、STL-10、ImageNet-Seg、Pascal-VOC（均为公开数据集）。
- **代码/权重**：论文未明确声明开源仓库与模型权重链接（论文为 arXiv 2610.10303v1，尚未提供 supplementary GitHub）。
- **关键超参**：大模型预训练使用 AdamW，peak lr = $3 \times 10^{-3}$（batch=4096 参考，线性缩放），weight decay=0.3，梯度裁剪 $l_2$ norm=1.0，warmup 10K 步，cosine decay，FP16 mixed precision，300 epochs；小模型训练 100 epochs，batch=128，peak lr=$5 \times 10^{-4}$，weight decay=$10^{-3}$，dropout=0.1；Patch size=16（大模型）/ 4（小模型）；随机种子、数据划分、优化器、LR schedule、dropout、数据增强在两模型间严格一致。
