---
title: "RELATIONAL-ABSTRACTIONS-FOR-SPATIAL-REASONING-WITH-DIFFUSION"
source: https://arxiv.org/pdf/2610.09780v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 22:52:37"
field: "视觉空间推理与生成模型"
keywords: ["diffusion models", "spatial reasoning", "object-centric learning", "relational abstraction", "generative modeling", "visual reasoning", "slot attention"]
innovations: ["将无监督对象发现与关系抽象引入扩散模型，通过对象中心关系嵌入引导条件生成以实现空间推理", "提出可扩展Slot解码器，通过可学习位置网格将高slot数量下的显存开销大幅降低", "发布包含Akari、Coldoku、Tangram、LogicFace四数据集的大规模空间推理基准，评估生成式规则学习能力"]
benchmarks: ["Akari", "Coldoku", "Tildoku", "Tangram", "LogicFace", "MNIST-Sudoku", "Counting Polygons"]
---

# 论文速读：RELATIONAL-ABSTRACTIONS-FOR-SPATIAL-REASONING-WITH-DIFFUSION

## 一句话总结
本文提出 RDM（Relational Abstractions for Diffusion Models），将无监督物体发现与对象关系抽象引入扩散模型，以克服其在空间推理任务中仅拟合像素分布、无法学习隐含逻辑规则的局限；同时发布包含四个谜题数据集（Akari、Coldoku、Tangram、LogicFace）的大规模基准。

## 研究问题与动机
1. **扩散模型缺乏结构化推理能力**：现有扩散模型擅长图像合成，但在满足复杂空间逻辑约束的条件生成任务中，往往仅拟合像素级分布，无法真正推断底层的逻辑规则。
2. **判别式视觉推理的局限性**：传统方法（如 RPM、Bongard 风格问题）将视觉推理建模为分类选择，易被表面视觉启发式策略利用，难以体现系统性规则推断。
3. **生成式推理方法的不足**：ARC 等生成式方法依赖符号程序综合，而在像素级图像生成上仍具挑战；近期基于扩散模型的方法（如 SRM）主要依赖采样时自适应策略，本质仍局限于像素空间重构。
4. **缺乏通用且可评估的空间推理基准**：现有基准（MNIST-Sudoku 等）过于简单或任务单一，无法系统评估模型在多类型空间推理任务中的规则学习能力和 OOD 泛化能力。

## 核心贡献（创新点）
1. **提出 RDM 框架**：通过无监督 Slot Attention 发现场景对象，并利用 Abstractor 学习对象间的关系嵌入，将其注入扩散模型 U-Net 瓶颈，实现关系引导的生成过程；与纯像素级条件扩散的本质区别在于引入了可解释的对象中心表征作为推理先验。
2. **设计可扩展 Slot 解码器**：通过可学习位置网格（position grid）替代逐 slot 全分辨率重建，大幅降低高 slot 数量（如 k=121）下的显存开销；与原始 Slot Attention 架构的本质区别在于将瓶颈计算复杂度从 O(k) 独立解码降为聚合后单次解码。
3. **提出关系引导训练策略**：训练时以 p_drop=0.5 概率丢弃 solution 的关系嵌入来模拟推理时无解的情况，类似 CFG 但不进行采样时加权融合，而是直接训练单参数化模型；与 classifier-free guidance 的本质区别在于不依赖 guide/unguide 线性插值采样。
4. **发布四大空间推理基准数据集**：涵盖网格图像补全（Akari、Coldoku）和多上下文图像生成（Tangram、LogicFace），每个数据集多达 100 万配对样本；与现有基准的本质区别在于涵盖多元推理类型且要求模型从数据中隐式推断逻辑规则而非显式给定。

## 方法详解
**整体架构**：RDM 由两个模块组成——Discoverer（无监督对象发现）和 Reasoner（关系引导的扩散模型）。

**Discoverer（Slot Attention 自编码器）**：
- 输入图像经卷积编码器得到 m 个特征嵌入 f_{1:m} 和位置嵌入 p_{1:m}。
- Slot Attention 模块初始化 k 个 slot s_{1:k}~N(0,I)，经 η=3 轮 GRU 增强的交叉注意力迭代更新，将对象绑定到各 slot。
- 输出 object-specific 特征 f̃_{1:k} 和位置 p̃_{1:k}，组成 z'=(f̃',p̃')。
- **可扩展 Slot 解码器**：引入可学习位置网格 g∈R^{h×w×d}，计算 ã=softmax(g·s^T) 作为 soft mask，再对 slot 加权求和得ṡ∈R^{h×w×d}，最后经卷积层统一解码重建输入，避免 O(k) 独立高分辨率卷积。

**Reasoner（关系抽象扩散模型）**：
- 关系抽象模块 R 基于 Slot Abstractor，将 clue 和 solution 的 z^c、z^0 拼接后送入 L=6 层抽象器，每层包含 Cross-Attention→Self-Attention→FFN，逐层融合对象特征与位置信息，输出关系嵌入 r*。
- 扩散模型（基于 Flow Matching 的 CNF）在 U-Net 瓶颈处以 cross-attention 注入 r*。
- 训练策略：以 p_drop=0.5 随机丢弃 solution 的 z^0 嵌入，使模型同时学习条件分布 p_θ(x_0|x_c,R(z^c)) 和 solution-guided 分布 p_θ(x_0|x_c,R(z^c,z^0))；推理时仅使用 clue 的关系嵌入。
- Flow matching 损失：L(θ)=E‖p_θ(x_t,t)−(ε−x_0)‖²，线性路径 μ_t(x_0)=(1−t)x_0，σ_t(x_0)=t。

## 实验与结果
**数据集与评估**：四个数据集各含最多 100 万样本，采用任务定制的确定性评估管线和神经网络评估器（LogicFace 用 ResNet-18 分类器）。

**Akari（网格光照谜题）**：
- ID  regime：RDM[S] clue (k=121) 达 ACC=83.12%，对比 Base 的 80.04%。
- OOD Low regime（最难）：RDM[S] clue (k=16) 达 ACC=72.48%，对比 Base 的 39.48%（提升近 2×），证明关系抽象显著改善 OOD 泛化。
- RDM 在低密度 regime 明显优于 SRM（ACC=0.00%），因 SRM 的不确定性引导在稀疏约束下失效。

**Coldoku（彩色数独）**：
- Easy：RDM[L] full (k=16) 达 ACC=97.48%，DGT=80.12%，CLR=88.22%。
- Hard：SRM 略优（ACC=20.78% vs RDM full 12.58%），但 RDM solution (k=16) 在 Hard 达到 ACC=3.34% vs Base 3.88%，整体在 Easy/Medium 全面领先。
- Tildoku（旋转数独）：RDM[L] solution 在中等难度各倾斜比例下平均 ACC=28.40%，大幅优于 SRM sequential（约 7%），验证了网格无关性。

**Tangram（七巧板）**：
- RDM[S] full 达 ACC=46.76%，约为 Base（15.46%）的 **3×**，SHAPE 准确率 65.12%。
- Solution 嵌入的贡献显著：clue(34.37) → solution(38.57) → full(46.76)，说明 solution 提供了显式的轮廓分割信息。

**LogicFace（面部逻辑运算）**：
- RDM[S] clue 达 AVG=92.93%，ACC=74.74%，远超 Base（ACC=11.24%）和 SRM sequential（ACC=5.70%）。
- 各逻辑运算符准确率均超 trivial 策略：OR=90.19%，AND=97.00%，XOR=97.26%，IMPLIES=87.27%。

**消融验证**：
- 纯 slot-conditioning（无关系抽象）在 Tangram 上 ACC=27.09%，远低于 RDM clue 的 34.37%，证明关系抽象是关键。
- 线性探针实验：冻结抽象器后，在有效/无效 Coldoku 嵌入上分类准确率达 89.98%/82.59%，证实抽象嵌入编码了逻辑规则。

## 相关工作脉络
1. **SRM（Wewer et al., 2025）**：基于 diffusion forcing 的 patch-level 不确定性感知的空间推理方法，对规则规则网格（如 Sudoku）有效，但无法处理非均匀分布约束（Akari）和网格扭曲场景（Tildoku）；RDM 通过关系抽象实现任务无关的通用推理。
2. **Classifier-Free Guidance（Ho & Salimans, 2021）**：经典的条件生成引导技术；RDM 借用其训练范式（部分丢弃条件），但在采样时不做线性插值，而是直接采样 unguided 分布，因推理时 solution 不可用。
3. **Slot Attention（Locatello et al., 2020）**：无监督对象发现的基础方法；本文将其从判别式任务扩展至生成式扩散引导，并通过可扩展解码器突破 k≤10 的限制。
4. **SlotDiffusion（Jiang et al., 2023）/ Wu et al. (2023)**：将 slot 表征用于扩散模型生成；本文对比实验表明纯 slot-conditioning 不如引入关系抽象，强调显式关系建模对复杂推理的必要性。
5. **Abstractors（Mondal et al., 2024; Altabaa et al., 2024）**：用于判别式视觉推理的对象关系抽象Transformer；本文将其引入扩散模型的条件注入，实现生成式关系推理。
6. **Diffusion Forcing（Chen et al., 2024a）**：SRM 的核心技术；本文指出其在网格外任务（Tildoku）中因 patch 与对象不对齐而失效，展示了关系抽象相比采样时启发式策略的鲁棒优势。

## 局限性与未来方向
1. **对象发现的可扩展性上限**：当前基于 Slot Attention 的神经网络架构在大规模无约束场景（数千对象）下存在内在伸缩性限制；作者认为这与人类认知的对象数量边界相符，但开放世界视觉推理仍需更强的关系表征。
2. **评估器的潜在误差**：神经评估器（LogicFace 的 ResNet-18）和确定性评估管线可能存在轻微估计误差，虽对所有模型公平但非绝对精确。
3. **未来方向**：（1）探索更强关系视觉表征以提升开放世界场景可扩展性；（2）将关系抽象与采样时推理策略（如 SRM）结合，探索混合架构；（3）探索 Conv+Transformer 混合骨干（本文 DiT 实验未收敛）。

## 研究启发与可借鉴点
1. **关系抽象引导扩散生成的范式**：将无监督对象发现（Slot Attention）+ 关系抽象（Abstractor）的组合引入扩散模型的条件注入，是一种有效的结构化推理增强手段，可迁移至其他需满足离散约束的生成任务（如电路布局、建筑平面图生成）。
2. **可扩展 Slot 解码器设计**：用可学习位置网格 g 替代逐 slot 高分辨率卷积，将内存复杂度从 O(k·H·W) 降为 O(h·w)，对高 slot 数量任务极具参考价值。
3. **Solution 关系嵌入作为隐式指导信号**：训练时随机丢弃 solution 嵌入的 CFG-like 策略，在推理时无需 solution 即可生成合理结果，这一思想可推广至其他"答案辅助训练"场景。
4. **任务定制化确定性评估 + 神经网络评估的结合**：Grid 类任务用确定性约束验证器，Real-image 类任务用专用分类器，这种混合评估策略为多类型推理基准建设提供了模板。
5. **OOD 泛化是关系抽象的核心价值**：在 Akari low regime 下 RDM 比 Base 提升 33 个百分点，提示关系抽象方法在分布外场景的潜力，值得在更多 OOD 设定下验证。

## 关键术语表
**RDM (Relational Abstractions for Diffusion Models)**：本文提出的核心框架，通过无监督对象发现和关系抽象来引导扩散模型进行空间推理。
**Slot Attention**：一种无监督对象发现机制，通过可学习的 slot 向量与输入特征的迭代交叉注意力，将场景解耦为离散的对象表征。
**Relational Abstraction (Abstractor)**：基于 Slot Abstractor 的关系抽象模块，通过多层交叉/自注意力将对象特征与位置信息融合为关系嵌入，编码对象间的语义和空间交互。
**Flow Matching**：一种连续时间扩散训练范式，通过最小化预测向量场与真实流量场之间的 MSE 来训练扩散模型。
**Classifier-Free Guidance (CFG)**：一种条件生成训练技术，通过随机丢弃条件嵌入来同时学习条件和无条件分布，本文借鉴其训练策略但改变采样方式。
**Coldoku**：本文提出的彩色数独数据集，同时满足数字和颜色两重数独约束，需同步求解两个正交的数独问题。
**Tildoku**：Coldoku 的旋转变体，图像经过随机仿射倾斜，用于测试模型对网格结构变化的鲁棒性。
**LogicFace**：基于 CelebA-HQ 面部图像的逻辑运算数据集，要求模型根据 OR/AND/XOR/IMPLIES 四种逻辑运算符组合面部属性生成一致的输出图像。

## 可复现要素
- **代码**：已开源，地址 https://github.com/anaezquerro/rdm
- **数据集**：论文声明代码和数据集均已公开（见 Reproducibility Statement）
- **关键超参**：slot attention 迭代 η=3；抽象器层数 L=6；p_drop=0.5；flow matching 线性路径；U-Net bottleneck 中 cross-attention 注入关系嵌入；Discoverer 学习率 4e-4，Reasoner 学习率 1e-4，batch size 128；训练步数 100K–400K 视任务而定
- **模型尺寸**：S（U-Net 25M）、M（40M）、L（120M）三档，Discoverer <8M
- **离散化**：默认 τ=100 步 Euler 采样，部分实验使用 τ=10
