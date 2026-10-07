---
title: "TRANSFER-COEFFICIENT-TRANSFER-FOR-EFFICIENT-MODEL-MERGING"
source: https://arxiv.org/pdf/2610.07819v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-07 10:43:10"
field: "模型合并与参数效率"
keywords: ["model merging", "coefficient transfer", "proxy model", "task vector", "large language model", "vision transformer", "hyperparameter transfer"]
innovations: ["首次在模型合并领域提出小模型代理系数迁移范式", "建立统一合并形式化框架并设计跨尺度层别映射策略", "通过Spearman相关性量化验证系数可迁移性并提供Hybrid Search扩展"]
benchmarks: ["DTD", "EuroSAT", "FER2013", "Food101", "GTSRB", "RESISC45", "Stanford Cars", "SUN397", "Usefulness Judge", "IFEval", "Banking77", "DDXPlus"]
---

# 论文速读：αTRANSFER: COEFFICIENT TRANSFER FOR EFFICIENT MODEL MERGING

## 一句话总结
论文提出αTransfer方法，通过在小型同系代理模型上搜索最优合并系数，直接迁移到大型目标模型进行合并，从而解决模型合并中系数搜索的计算成本与显存开销瓶颈，在视觉Transformer上实现最高6倍加速和70%内存节省，在大语言模型上实现最高20倍加速和85%内存节省，同时保持与直接在目标模型上搜索相当的性能。

## 研究问题与动机
- **核心问题**：模型合并（Model Merging）需要通过调整权重空间中的合并系数来融合多个微调checkpoint，但搜索最优系数在当前大模型场景下计算成本极高，且随着模型尺寸和任务数量增加呈指数级增长。
- **现有方法不足（评估类）**：基于网格搜索、进化搜索或贝叶斯优化的方法虽然减少了评估次数，但每次评估仍需在目标大模型上进行完整前向推理，单次运行耗时超10小时，总计算预算巨大。
- **现有方法不足（梯度类）**：AdaMerging、DivMerge等基于梯度的方法需要同时加载多个微调模型进行反向传播，显存消耗随任务数和模型尺寸双重增长，单GPU难以部署。
- **关键洞察**：论文发现同一家族模型在不同尺度下，对合并系数的性能分布呈现高度相似性（Spearman相关性达0.94），这意味着最优系数本质上反映任务间相关性与相对重要性，由任务特性与模型族决定，而非模型规模。

## 核心贡献（创新点）
1. **提出αTransfer框架**：首次将在小模型上搜索的合并系数迁移到大模型的范式引入模型合并领域，将搜索成本解耦于目标模型规模。
2. **建立统一的合并方法形式化**：提出通用表达式$\theta_{merged} = \mathcal{M}(\theta_0, \{\tau_i\}; \alpha)$，将Task Arithmetic、TIES、DARE、AdaMerging等六种方法纳入同一分析框架，按编辑函数$f$、搜索策略、系数粒度（全局/任务级/层级别）进行分类。
3. **设计层别迁移策略**：针对代理模型与目标模型深度不同的场景，提出Copy-based和Interpolation-based两种相对深度映射方法，实现层别系数的跨尺度传递。
4. **提出Hybrid Search扩展**：将传递的系数作为初始化点，在目标模型上进行少量局部精修，兼顾效率与性能，在多数情况下可完全追平甚至超越直接搜索。
5. **系统性实证验证**：在CLIP-ViT、SigLIP、Qwen3三个模型族，六种合并方法，十二个任务（8视觉+4语言）上验证，证明跨架构、跨任务、跨尺度的泛化能力。

## 方法详解
- **统一形式化**：将模型合并没有视为独立方法，而是分解为编辑阶段$f(\{\tau_i\}) \rightarrow \tilde{\tau}$与加权阶段$\theta_0 + \sum \alpha_i \tilde{\tau}_i$，系数$\alpha$的粒度分为全局（单一标量）、任务级（每任务一个标量）、层别（每任务每层一个标量）。
- **全局/任务级迁移**：在代理模型上求解$\alpha^* = \arg\min_\alpha \mathcal{L}^p(\theta_0^p + \sum \alpha_i \tilde{\tau}_i^p)$，然后将$\alpha^*$直接应用于目标模型：$\theta_{merged}^t = \theta_0^t + \sum \alpha_i^* \tilde{\tau}_i^t$，其中编辑函数$f$仍在目标模型的任务向量上原生计算。
- **层别迁移策略**：
  - **Copy-based**：目标模型第$b$个block的层，映射到代理模型第$\lceil \frac{b}{B_t} \cdot B_p \rceil$个block的对应层类型。
  - **Interpolation-based**：根据目标层连续相对深度，对相邻两个代理block的系数做线性插值。
- **Hybrid Search**：将传递系数作为初始点，分配小部分预算在目标模型上精修。对评估类方法执行粗粒网格搜索后的局部精化；对梯度类方法平分优化步数，前半段在代理上优化，后半段以传递系数为起点在目标上优化。

## 实验与结果
- **数据集**：视觉任务8个（DTD、EuroSAT、FER2013、Food101、GTSRB、RESISC45、Stanford Cars、SUN397）；语言任务4个（Usefulness Judge、IFEval、Banking77、DDXPlus）。
- **模型族**：CLIP-ViT（ViT-B/32、ViT-B/16、ViT-L/14）、SigLIP（Base、Large）、Qwen3（0.6B、1.7B、4B）。
- **基线**：Origin（直接在目标模型搜索）、SA（简单平均无搜索）、Hybrid（传递+局部精修）。
- **视觉模型主要结果**：
  - SigLIP-Base→SigLIP-Large：TIES仅下降0.11%、DARE下降0.32%，Task Arithmetic下降1.39%；速度提升3.01×，内存减少62.3%。
  - ViT-B/32→ViT-L/14：Task Arithmetic仅下降0.43%、DARE下降0.49%；速度提升6.07×，内存减少68.7%。
  - 梯度方法（AdaMerging/AdaMerging++/DivMerge）内存减少高达70.4%。
- **LLM主要结果**：
  - Qwen3-0.6B→4B：Task Arithmetic零性能损失；TIES-Pairwise零损失；速度提升19.97×，内存从40.22GB降至5.96GB（减少85.2%）。
  - Qwen3-0.6B→1.7B：误差<2%，速度提升11-19×，内存减少65%。
- **相关系数验证**：Table 2显示Spearman相关系数在0.58（ViT-B/32→L/14，因patch size差异）至0.94（Qwen3-0.6B→1.7B）之间，支撑迁移可行性。
- **最强结果**：Qwen3-0.6B→4B的Task Arithmetic和TIES-Pairwise传递后性能与直接搜索完全一致（76.03%和73.69%），同时实现约20倍加速和85%显存节省。

## 相关工作脉络
- **Task Arithmetic (Ilharco et al., 2023)**：最早的模型合并方法，用简单加权平均合并任务向量，采用全局网格搜索系数。本文在其框架内验证迁移可行性。
- **TIES (Yadav et al., 2023)**：引入trim和sign election解决冲突，需搜索任务级系数。本文通过αTransfer将其搜索成本大幅降低。
- **DARE (Yu et al., 2024)**：引入随机掩码稀疏化，本文验证其在不同尺度间的系数可迁移性。
- **AdaMerging / AdaMerging++ (Yang et al., 2024)**：基于梯度优化预测熵最小化，需同时加载多模型。本文通过代理模型规避显存瓶颈。
- **DivMerge (Touayouch et al., 2026)**：基于Jensen-Shannon散度优化，论文指出其在代理模型上更快收敛。
- **Evolutionary/Bayesian Search (Akiba et al., 2025; Lee et al., 2025)**：减少评估次数的元启发式方法，但每次评估仍昂贵。αTransfer与之正交，可直接叠加使用。
- **μTransfer (Yang et al., 2021) 与 DoReMi (Xie et al., 2023)**：已有在预训练超参数和数据混合权重上的小模型代理研究，本文是首次将其思想应用于合并系数搜索。

## 局限性与未来方向
- **层别迁移在patch size差异大时效果下降**：ViT-B/32→ViT-L/14因patch size不同导致相关性仅0.58，性能差距1-3%。
- **高稀疏度影响AdaMerging++迁移**：当k=20（保留20%权重）时，强掩码放大代理与目标的结构性差异，限制可迁移性；k≥60时效果良好。
- **跨架构迁移未验证**：目前仅在模型族内（相同架构、不同尺度）验证，跨architecture（如ViT→CNN）的系数映射尚未探索。
- **未来方向**：论文建议探索其他合并超参数的跨尺度迁移性、跨recipe的系数映射、以及更精细的层间关系建模。

## 研究启发与可借鉴点
- **可复用的方法**：任何涉及超参数搜索且目标模型较大的场景（如学习率、数据配比、稀疏度）均可考虑小模型代理+系数迁移思路，与μTransfer、DoReMi形成方法论呼应。
- **实验设计借鉴**：用Spearman秩相关量化代理与目标的性能分布对齐程度，为迁移可行性提供可量化的先验验证指标。
- **层间映射策略**：Copy-based与Interpolation-based的相对深度映射可用于其他跨尺度参数对齐问题（如知识蒸馏中的层对齐）。
- **Hybrid Search范式**：先代理粗搜、再目标精修的混合策略可作为通用加速模板，适用于各类昂贵搜索问题。
- **统一形式化价值**：将多种合并方法抽象为"编辑+加权"两阶段结构，有助于系统性地比较和扩展新方法。

## 关键术语表
- **Model Merging（模型合并）**：在参数空间直接加权融合多个微调模型权重，无需重新训练即可获得多任务能力的方法。
- **Task Vector（任务向量）**：微调模型参数与预训练模型参数的差值$\tau_i = \theta_i - \theta_0$，捕捉任务特定的知识更新。
- **αTransfer**：本文提出的方法，在小型代理模型上搜索最优合并系数后迁移到大型目标模型。
- **Evaluation-based Search（评估类搜索）**：通过多次前向推理评估不同系数组合性能来寻找最优解的方法，计算成本高。
- **Gradient-based Optimization（梯度类优化）**：直接对系数求梯度并优化代理目标（如预测熵、散度），需同时加载多模型，显存成本高。
- **Global/Task-wise/Layer-wise（全局/任务级/层别）**：合并系数$\alpha$的粒度，分别对应单标量、每任务一标量、每任务每层一标量。
- **Spearman's ρ（斯皮尔曼秩相关）**：用于衡量代理与目标模型在系数空间中性能排序一致性的统计量。
- **Hybrid Search（混合搜索）**：将传递系数作为初始化，在目标模型上进行少量局部精修的混合策略。

## 可复现要素
- **数据集**：8个视觉分类数据集（DTD、EuroSAT、FER2013、Food101、GTSRB、RESISC45、Stanford Cars、SUN397）和4个语言任务（Usefulness Judge、IFEval、Banking77、DDXPlus），均为公开基准。
- **代码/权重**：论文未明确声明代码开源状态，模型权重使用CLIP OpenAI发布版、SigLIP Google发布版、Qwen3发布版。
- **关键超参**：
  - 学习率：视觉任务{1e-5, 3e-5}，LLM固定4e-5
  - Batch size：视觉{32, 48}，LLM固定36
  - Epochs：视觉{5, 8, 10, 20}，LLM最多6
  - TIES/DARE k=20（top-20%保留）
  - DARE drop rate=0.5
  - AdaMerging：500步，lr=1e-3，1000样本
  - DivMerge：100步，lr=1e-2，200样本，JS散度
  - 网格搜索范围：Task Arithmetic (0,1] 16点，TIES/DARE (0,1.8] 18点
