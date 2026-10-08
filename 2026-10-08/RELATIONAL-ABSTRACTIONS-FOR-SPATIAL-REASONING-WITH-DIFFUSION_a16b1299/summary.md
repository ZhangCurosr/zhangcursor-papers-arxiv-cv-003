---
title: "RELATIONAL-ABSTRACTIONS-FOR-SPATIAL-REASONING-WITH-DIFFUSION"
source: https://arxiv.org/pdf/2610.09780v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 09:51:48"
field: "视觉生成与空间推理"
keywords: ["diffusion models", "spatial reasoning", "object-centric learning", "relational abstraction", "generative reasoning", "slot attention"]
innovations: ["提出RDM框架将无监督对象发现与关系抽象注入扩散模型", "设计可扩展Slot解码器支持数百级槽数推理", "构建百万级多谜题基准测试证明关系抽象提升3倍推理能力"]
benchmarks: ["Akari", "Coldoku", "Tangram", "LogicFace", "Tildoku"]
---

# 论文速读：RELATIONAL-ABSTRACTIONS-FOR-SPATIAL-REASONING-WITH-DIFFUSION

## 一句话总结
本文提出 RDM（Relational Abstractions for Diffusion Models）框架，通过无监督对象发现（Slot Attention）和学习对象间关系抽象，将结构化的关系知识注入扩散模型，从而显著提升其在空间推理任务上的条件生成能力，并引入了包含 Akari、Coldoku、Tangram、LogicFace 四个大规模谜题数据集的基准测试。

## 研究问题与动机
- **扩散模型在结构化推理中的缺陷**：当前先进生成模型在处理具有隐含逻辑结构的条件数据分布任务（如谜题）时，往往只能逼近像素级分布，而未能学习推理所需的底层逻辑规则。
- **现有方法的局限性**：传统视觉推理方法多为判别式分类任务（如选择答案），允许模型利用表层视觉启发式方法；而基于符号程序综合的方法难以扩展到像素级图像生成。
- **推理范式的转变需求**：尽管近期研究开始将空间推理重构为生成问题（如 ARC 挑战），但现有扩散模型方法主要依赖采样时的推理策略调整，未从生成过程内部引入高层语义抽象。

## 核心贡献（创新点）
- **提出 RDM 框架**：首次将无监督对象中心表示与关系抽象模块结合，使扩散模型能够在训练和推理过程中利用结构先验引导生成过程。（本质区别：不同于仅调整采样策略的方法，从生成源头增强推理能力）
- **可扩展的 Slot 解码器设计**：提出基于可学习位置网格的缩放型 Slot 解码器，将槽数量从传统的 k ≤ 10 扩展到数百级，支撑大规模网格推理任务。（本质区别：突破原有计算瓶颈，适配不同粒度的对象划分需求）
- **引入大规模空间推理基准**：构建了包含百万级样本的四个新数据集（Akari、Coldoku、Tangram、LogicFace），覆盖从网格补全到多上下文生成的多种推理范式。（本质区别：现有基准多基于 MNIST 变体，本文提供更具挑战性和真实感的数据集）
- **证明关系抽象对规则学习的可解释性**：通过线性探针实验证明，学习到的关系嵌入能够线性分离有效与违反规则的谜题配置，表明模型确实学习了逻辑规则而非简单模式匹配。（本质区别：为黑箱扩散模型提供了可解释的推理证据）

## 方法详解
- **整体架构**：RDM 由两个模块组成——Discoverer（无监督对象发现）和 Reasoner（关系指导的扩散模型）。Discoverer 使用 Slot Attention 将输入图像解耦为 k 个槽嵌入，每个槽包含对象特征 $\tilde{\mathbf{f}}_{1:k}$ 和位置 $\tilde{\mathbf{p}}_{1:k}$。
- **可扩展 Slot 解码器**：为避免传统方法中高槽数导致的内存开销，引入可学习位置网格 $\mathbf{g} \in \mathbb{R}^{h \times w \times d}$，通过计算 $\tilde{\alpha} = \text{softmax}(\mathbf{g} \cdot \mathbf{s}_{1:k}^\top)$ 估计每个位置的槽权重，再聚合特征图 $\tilde{\mathbf{s}} = \sum_i \tilde{\alpha}_i \mathbf{s}_i$，最后经解码器生成统一重建。
- **关系抽象模块（Abstractor）**：采用 Slot Abstractors 架构，由 L=6 层抽象层堆叠而成，每层包含交叉注意力、自注意力与前馈网络，逐层将位置嵌入与特征嵌入交互，输出关系嵌入 $\mathbf{r}_{1:k}^* = \mathcal{R}(\tilde{\mathbf{f}}_{1:k}^*, \tilde{\mathbf{p}}_{1:k}^*)$。
- **条件扩散建模**：Reasoner 使用 Flow Matching 训练的单参数化 U-Net，同时学习两种分布——含解 guided 分布 $p_\theta(\mathbf{x}_0 | \mathbf{x}_c, \mathcal{R}(\mathbf{z}^c, \mathbf{z}^0))$ 和纯线索 unguided 分布 $p_\theta(\mathbf{x}_0 | \mathbf{x}_c, \mathcal{R}(\mathbf{z}^c))$。训练时以 $p_{\text{drop}}=0.5$ 概率丢弃解嵌入模拟推理时不可用状态。
- **三种条件设置**：(i) clue-only：仅用线索关系；(ii) solution：训练时用解关系、推理时不用；(iii) full：训练用两者、推理仅用线索。

## 实验与结果
- **数据集规模**：Akari（1M 样本）、Coldoku（1M 样本）、Tangram（1,013 样本）、LogicFace（30,000 样本）。
- **Akari 结果**：RDM[clue, 121 slots] 在 ID 设置下达 83.12% 准确率，较 Base 的 80.04% 提升；OoD low regime（5-10% 墙壁密度）下保持 72.48%（vs Base 39.48%），展现强泛化能力。
- **Coldoku 结果**：RDM[full, 16 slots] 在 Easy（50-60 clues）达 97.48% ACC、Medium 达 71.64% ACC；但在 Hard  regime SRM 略优（20.78% vs 12.58%）。81 slot 配置在 Easy 降至 90.36%，表明过度细粒度可能损害关系模块性能。
- **Tangram 结果**：RDM[full] 达 46.76% ACC，较 Base 的 15.46% 提升约 **3 倍**；shape accuracy 达 65.12%。solution 条件贡献显著（clue 34.37% → solution 38.57% → full 46.76%）。
- **LogicFace 结果**：RDM[clue] 达 74.74% ACC，远超 Base 的 11.24%；四个逻辑运算符准确率均超过随机基线（OR: 90.19%, AND: 97.00%, XOR: 97.26%, IMPLIES: 87.27%）。
- **Tildoku（倾斜 Coldoku）**：RDM[solution] 在中等倾斜度下达 28.40% ACC，而 SRM 在偏移增加时性能骤降，证明 RDM 对网格几何变形更具鲁棒性。
- **消融实验**：纯 slot-conditioned diffusion（非关系抽象）在 Tangram 仅达 27.09%，显著低于 RDM 的 34.37%，证明关系抽象的必要性。线性探针实验显示关系嵌入对有效/无效谜题的分类准确率分别达 89.98% 和 82.59%。

## 相关工作脉络
- **Diffusion Forcing / SRM（Wewer et al., 2025）**：基于扩散强制的推理方法，依赖 patch 级不确定性估计进行渐进式采样。本文证明该方法在 Grid 结构对齐的任务（MNIST-Sudoku）有效，但在 Akari 和 Tildoku 等推理变量非均匀分布或几何变形场景下性能崩溃。
- **Object-Centric Slot Diffusion（Jiang et al., 2023; Wu et al., 2023）**：将 Slot Attention 嵌入扩散模型用于 compositional 图像合成。本文指出此类工作未针对推理任务优化，且缺乏显式关系抽象机制。
- **Abstract Visual Reasoning 判别式方法（Pekar et al., 2020; Shi et al., 2024）**：基于深层潜变量模型生成 RPM 缺失面板。本文认为这些方法依赖手工归纳偏置，且无法处理像素级生成任务。
- **Symbolic Program Synthesis for ARC（Chollet et al., 2024）**：ARC 挑战主流解法依赖符号程序合成。本文走相反路径，在连续像素空间内通过学到的关系抽象实现类符号推理。
- **Classifier-Free Guidance（CFG, Ho & Salimans, 2021）**：传统 CFG 在采样时线性组合条件/无条件分布。本文采用相似训练策略但不做采样时组合，因推理时无解 embedding 可供 conditional 分支使用。

## 局限性与未来方向
- **对象发现的可扩展性瓶颈**：当前基于神经网络的对象发现架构存在固有可扩展性限制，难以直接应用于"野外"场景中数千对象的无约束场景。（自述局限）
- **自然场景的开放世界推理挑战**：作者承认方法对齐人类认知约束（对象数量有限），但开放世界视觉推理仍需更强大的关系视觉表示。（未来方向）
- **评估器的估计误差**：基于神经网络的自动化评估器可能存在轻微估计误差，虽公平比较但非完美。（自述局限）
- **架构选择的探索空间**：初步尝试 DiT（Diffusion Transformer）作为骨干网络在空间推理任务上难以优化，暗示卷积归纳偏置在此类任务上的优势；混合架构设计留待未来研究。

## 研究启发与可借鉴点
- **"发现+推理"的两阶段范式可迁移**：先用无监督方法解耦场景对象，再将对象关系注入生成过程——这一范式可推广至其他结构化生成任务（如电路图生成、化学分子设计）。
- **关系嵌入的可解释性验证方法**：通过线性探针测试有效/无效配置的线性可分性，为黑箱生成模型提供可解释证据，值得在后续工作中复用。
- **条件丢弃策略的设计灵活性**：采用 $p_{\text{drop}}$ 随机丢弃解嵌入模拟推理场景，这一训练技巧可用于任何"训练时有额外信息、推理时无可用"的条件生成任务。
- **可扩展 Slot 解码器降低内存开销**：基于位置网格的加权聚合设计避免了每槽独立重建的高内存消耗，可直接复用于其他需要高槽数的对象中心任务。
- **与采样时推理的兼容性**：作者指出 RDM 原则上可与 SRM 等采样时推理方法结合，暗示多层推理增强（训练时抽象 + 推理时细化）是可行的改进方向。

## 关键术语表
- **Slot Attention**：无监督对象发现方法，通过迭代注意力机制将图像特征分配给固定数量的"槽"向量，每个槽代表场景中的一个对象。
- **Flow Matching**：扩散模型训练技巧，学习将数据分布映射到高斯先验的常微分方程向量场，相比 noise prediction 更稳定高效。
- **Relational Abstraction（关系抽象）**：将对象中心表示中的特征与位置信息交互，学习对象间的高层语义关系与空间依赖的模块。
- **Classifier-Free Guidance（CFG）**：扩散模型条件控制技巧，通过训练条件/无条件分支并在采样时线性组合来实现更强的条件控制。
- **OoD Generalization（分布外泛化）**：模型在训练分布之外（如极端参数设置、几何变形）保持推理能力的特性。
- **Diffusion Forcing**：将自回归 next-token prediction 思想引入扩散模型，通过patch级不确定性指导渐进式去噪顺序的方法。
- **Tildoku**：本文提出的 Coldoku 变体，对拼图图像施加随机倾斜变换，测试模型对几何变形的鲁棒性。
- **Object-Centric Representation（对象中心表示）**：将复杂视觉场景分解为离散、可交互的对象嵌入的表征学习范式。

## 可复现要素
- **代码开源**：GitHub 仓库 https://github.com/anaezquerro/rdm（作者声明）
- **数据集**：四个新数据集（Akari、Coldoku、Tangram、LogicFace）随代码一同开源
- **关键超参**：
  - Discoverer：AdamW, lr=4×10⁻⁴, batch_size=64, warmup=10K steps
  - Reasoner：AdamW, lr=1×10⁻⁴, batch_size=128, warmup=5K steps
  - Abstractor 层数 L=6，slot 维度 d=128/256/512，槽数 k=6/7/16/81/121（视任务）
  - 训练步数：300K（S/M 配置）至 400K（L 配置）
  - 混合精度：bfloat16
  - 采样步数 τ=10 或 τ=100
- **模型规模**：三档 S/M/L，U-Net 参数量 25M/35M/120M，Discoverer <8M 参数
