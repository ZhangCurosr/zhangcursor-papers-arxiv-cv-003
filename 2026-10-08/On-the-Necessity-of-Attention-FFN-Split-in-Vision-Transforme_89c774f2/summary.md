---
title: "On-the-Necessity-of-Attention-FFN-Split-in-Vision-Transforme"
source: https://arxiv.org/pdf/2610.10303v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 03:15:16"
field: "视觉Transformer架构分析"
keywords: ["Vision Transformer", "inductive bias", "attention-FFN dichotomy", "AttenFeed", "uViT", "model scaling"]
innovations: ["提出AttenFeed模块，在Attention的OV circuit中嵌入GELU激活，统一Attention和FFN功能", "构建uViT控制架构，消除ViT中严格的Attention-FFN交替结构", "揭示Attention-FFN分离在小模型规模下构成有害归纳偏置，uViT-S在ImageNet-1k上较ViT-S提升13.47%"]
benchmarks: ["ImageNet-1k", "Places365", "iNaturalist 2021", "CIFAR-10", "CIFAR-100", "SVHN", "STL-10", "ImageNet-Segmentation", "Pascal-VOC"]
---

# 论文速读：On the Necessity of Attention–FFN Split in Vision Transformers

## 一句话总结
本文提出 AttenFeed 模块，通过在标准 Attention 的值投影（OV circuit）中引入非线性激活函数，将 Attention 和 FFN 统一为单一组件，并构建基于此的 uViT 架构；实验表明，ViT 中严格的 Attention–FFN 交替结构在小模型规模下构成有害的归纳偏置，而 uViT 通过放松该约束实现了显著性能提升。

## 研究问题与动机
- ViT 的核心架构依赖"Attention → FFN"交替结构，但这一结构性假设的系统性归纳偏置从未被严格检验。
- 近期 MoE 相关工作（如 SkipGPT）发现连续 Attention-Attention 或 FFN-FFN 交互有时优于交替模式，暗示固定交替结构可能存在冗余。
- 小模型参数预算有限，ViT 强制将参数分配到固定功能的 Attention 和 FFN 两层，导致参数利用率不灵活，可能制约性能上限。
- 当前研究缺乏一套"控制变量"架构来系统对比有/无 Attention–FFN 分离时模型的学习行为差异。

## 核心贡献（创新点）
- **提出 AttenFeed 模块**：在 Attention 的 OV circuit 中嵌入 GELU 激活，数学上证明其可同时退化为标准 Attention 和标准 FFN；与 UMoE 等"共享专家"思路的本质区别在于 AttenFeed 不依赖路由/门控，而是直接通过激活函数将两者统一于同一张量计算。
- **构建 uViT 控制架构**：用纯 AttenFeed 序列替代 ViT 的交替 Attention–FFN 结构，形成可对比的控制组；本质区别于以往简化注意力（如 Pure Attention、Attention-only Transformer）的工作——uViT 并非删除 FFN，而是让每层同时具备两者功能。
- **揭示规模相关的归纳偏置效应**：通过 uViT 对比发现，Attention–FFN 分离在小规模时阻碍有效学习（uViT-S 在 ImageNet-1k 上 +13.47% Top-1），随规模增大差异逐渐消失；这是首次从"分离必要性"角度系统性解释规模与结构偏置的关系。

## 方法详解
- **AttenFeed 公式**：$\text{AttenFeed}^{\ell,h}(\mathbf{X}^{\ell-1})_i = \sum_j a_{i,j}^{\ell,h} \sigma(\mathbf{x}_j^{\ell-1}\mathbf{W}_V^{\ell,h})\mathbf{W}_O^{\ell,h}$，其中 $\sigma$ 默认取 GELU；标准 Attention 对应 $\sigma$ 省略（OV 无激活），标准 FFN 对应 self-focus（$a_{i,i}\to1$）时的退化形式。
- **uViT 更新规则**：$\mathbf{x}_i^\ell = \mathbf{x}_i^{\ell-1} + \sum_h \text{AttenFeed}^{\ell,h}(\text{LN}^\ell(\mathbf{X}^{\ell-1}))_i$，即每层仅含 AttenFeed，不再保留独立 FFN 层。
- **参数对齐策略**：uViT 去掉 FFN 后参数量约为 ViT 的 1/3，为公平比较按同比例增加层数与维度（使总参数量近似匹配）。
- **理论命题**：Prop 4.1（self-focused 时退化为 FFN）、Prop 4.2（预激活接近零时退化为标准 Attention）、Prop 4.3（在更宽松条件下仍可近似标准 Attention）。

## 实验与结果
- **预训练（主结果）**：在 ImageNet-1k 上，uViT-S 取得 67.11% Top-1，相比 ViT-S（53.64%）**提升 +13.47%**；uViT-B 74.90% vs ViT-B 74.22%（+0.68%）；uViT-L 77.18% vs ViT-L 77.33%（-0.15%，差距基本消失）。Places365 和 iNaturalist 2021 趋势一致。
- **表征秩（Rank）实验**：uViT 在 CIFAR-10/100/SVHN/STL-10 上均保持比无 FFN 版本（xViT）更高的 pre-head 秩，且准确率同步提升，证明 AttenFeed 继承了 FFN 防止秩坍缩的能力。
- **注意力图分割**：在 ImageNet-Seg 和 Pascal-VOC 上，uViT 的注意力图可生成与 ViT 相近质量的分割 mask（mAP 差距极小，uViT-L 甚至在 ImageNet-Seg 上 mAP 略高于 ViT-L）。
- **迁移学习**：线性探针在 SVHN/CIFAR10/100/STL10 四个数据集上 uViT 全面超越 ViT；全微调多数场景也占优，表明 uViT 表征更灵活、归纳偏置更弱。

## 相关工作脉络
- **MLP-Mixer / ResMLP**：以纯 MLP 替代 Attention 降低归纳偏置；本文与之不同，AttenFeed 不是用 MLP 替换 Attention，而是在 Attention 内部融合 FFN 能力，保留注意力机制的同时放松功能分离。
- **UMoE（Yang et al., 2025）**：通过共享专家统一 Attention 和 FFN；本文受其启发但路线不同——AttenFeed 无需专家路由，直接修改标准 Attention 的计算结构。
- **GAU（Hua et al., 2022）**：融合 GLU 与 Attention，目标是弥补线性注意力性能不足；本文动机是系统研究结构归纳偏置，而非弥补性能短板。
- **Pure Attention / Attention-only Transformer（Huben & Morris, 2023; Wang et al., 2025）**：完全去除 FFN；本文与之相反，主张的是"统一而非移除"，AttenFeed 兼具两方功能。
- **SkipGPT（Zhao et al., 2025）**：通过路由器动态跳过 Attention 或 FFN 块；本文不引入额外路由机制，而是从模块层面统一两者。

## 局限性与未来方向
- 实验仅局限于视觉域，未在 NLP 或音频等其他模态验证；Attention–FFN 分离的必要性可能因领域而异。
- 使用标准 ViT 训练配方（未针对 uViT 优化超参）；专属训练策略（正则化、学习率调度、数据增强）可能进一步拉开性能差距或改变规模行为。
- AttenFeed 的激活函数选择（本文固定 GELU）及其他激活变体的影响未充分探索。
- 缺乏对自注意力 head 在 uViT 中如何分工（哪些更接近 Attention、哪些更接近 FFN）的细粒度分析。

## 研究启发与可借鉴点
- **"最小修改"范式**：在已有模块中加入最少改动（一个 GELU）即可实现功能统一，这种思路可迁移到 SSM、State-Space 等新兴模块的统一研究中。
- **控制组实验设计**：用 uViT 作为"无分离"对照组来量化结构归纳偏置的影响，这一实验范式可推广到研究其他"约定俗成"的结构假设（如 residual 连接、LayerNorm 位置等）。
- **规模-偏置权衡启示**：小模型应优先考虑结构灵活性，大模型可利用强归纳偏置加速收敛；对资源受限场景（边缘设备、移动端）的 ViT 设计有直接参考价值。
- **秩分析作为诊断工具**：用 pre-head/post-head 秩作为表征丰富度的代理指标，可在不依赖下游任务的情况下快速评估架构设计的有效性。

## 关键术语表
**AttenFeed 模块**：在标准 Attention 的 OV circuit 中引入非线性激活（如 GELU）后得到的统一组件，理论上可退化为标准 Attention 或标准 FFN。
**uViT（Unified Vision Transformer）**：完全由 AttenFeed 模块堆叠而成的 Vision Transformer，去除了 ViT 中严格的 Attention–FFN 交替结构。
**Attention–FFN dichotomy**：ViT 中 Attention 层与 FFN 层严格分离、交替出现的设计惯例，本文认为这是一种潜在的有害归纳偏置。
**归纳偏置（Inductive Bias）**：架构设计中对解空间的结构性约束，帮助模型快速收敛，但也可能限制其在特定规模下的表达上限。
**Rank Collapse（秩坍缩）**：仅含 Attention 的网络随深度增加导致表征矩阵秩急剧下降、token 表征趋同的现象；FFN 的非线性激活可缓解此问题。
**OV Circuit**：Attention 中值向量经 $\mathbf{W}_V$ 投影后再经 $\mathbf{W}_O$ 输出的部分，负责将输入信息映射到输出空间。

## 可复现要素
- **数据集**：ImageNet-1k（公开）、Places365（公开）、iNaturalist 2021（公开）、CIFAR-10/100（公开）、SVHN（公开）、STL-10（公开）；代码仓库论文未明确声明，需联系作者或查看 arXiv 附页。
- **关键超参**：
  - ViT-S (21M): L=12, H=6, D=384；uViT-S (20M): L=12, H=8, D=640
  - ViT-B (86M): L=12, H=12, D=768；uViT-B (80M): L=12, H=16, D=1280
  - ViT-L (303M): L=24, H=16, D=1024；uViT-L (289M): L=28, H=20, D=1600
  - 优化器：AdamW，peak LR=3×10⁻³（batch=4096 时线性缩放），weight decay=0.3
  - 训练：300 epochs，linear warmup 10k steps + cosine decay，FP16 mixed precision
  - 数据增强：RandomResizedCrop + RandomFlip(p=0.5) + RandAugment(mag=9)，MixUp(α=1.0) / CutMix(α=1.0)
  - 激活：GELU（AttenFeed 中）；patch size=16，分辨率 224×224
