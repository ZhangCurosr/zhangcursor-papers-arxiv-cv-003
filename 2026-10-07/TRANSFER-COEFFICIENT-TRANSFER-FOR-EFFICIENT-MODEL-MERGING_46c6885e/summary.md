---
title: "TRANSFER-COEFFICIENT-TRANSFER-FOR-EFFICIENT-MODEL-MERGING"
source: https://arxiv.org/pdf/2610.07819v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 02:39:03"
field: "模型合并与效率优化"
keywords: ["model merging", "coefficient transfer", "proxy model", "task vector", "model scaling", "multi-task learning"]
innovations: ["提出在小型代理模型上搜索合并系数并迁移至大型目标模型的αTransfer范式", "建立编辑+加权的统一合并形式化框架并设计跨层深相对深度映射策略", "证明同家族模型间合并系数性能分布的高度跨尺度可迁移性"]
benchmarks: ["CLIP-ViT, SigLIP 视觉分类", "Qwen3 LLM 语言任务", "8项视觉任务, 4项语言任务"]
---

# 论文速读：αTRANSFER: COEFFICIENT TRANSFER FOR EFFICIENT MODEL MERGING

## 一句话总结
αTransfer 提出在小型代理模型上搜索最优合并系数，再直接迁移至同家族大型目标模型的高效模型合并方法，在视觉 Transformer 上实现最高 6× 加速和 70% 显存节省，在 LLM 上实现最高 20× 加速和 85% 显存节省，同时保持与直接在目标模型上搜索相当的精度。

## 研究问题与动机
- **系数搜索是模型合并可扩展性的核心瓶颈**：现有合并方法（Task Arithmetic、TIES、DARE 等）的性能高度依赖合并系数的选择，而每个候选系数组合都需要完整评估一个合并模型，搜索成本随任务数量和模型规模指数增长。
- **基于评估的方法受计算成本制约**：均匀系数、进化搜索、贝叶斯优化等策略虽减少了评估次数，但每次评估的代价随模型增大而剧增，总计算预算仍然巨大。
- **基于梯度的方法受显存消耗制约**：AdaMerging、DivMerge 等方法直接对系数求梯度，需同时持有多个微调模型，显存占用随任务数量和模型规模共同放大，在大模型上难以运行（如目标模型只能将 batch size 从 16 降至 4）。
- **同家族模型间存在跨尺度分布相似性**：论文通过可视化性能景观和 Spearman 秩相关分析发现，同一模型家族中不同规模的模型在合并系数空间上的性能分布高度对齐（Qwen3-0.6B → 4B 相关系数达 0.90），为系数迁移提供了理论基础。

## 核心贡献（创新点）
- **提出 αTransfer 范式**：在小型代理模型上搜索合并系数并迁移至大型目标模型，首次将"小模型搜索→大模型应用"的思路引入模型合并领域，突破了直接搜索的计算和显存瓶颈。
- **建立统一的模型合并形式化框架**：将六种主流合并方法（Task Arithmetic、TIES、DARE、AdaMerging、AdaMerging++、DivMerge）统一表述为"编辑 + 加权"两阶段操作，便于系统分析比较并证明系数迁移的通用适用性。
- **设计跨层深的层-wise 系数迁移策略**：针对代理模型和目标模型层数不同的情况，提出基于复制（Copy）和线性插值（Interpolation）的相对深度映射方案，有效处理 Transformer 重复块之间的系数传递。
- **提出混合搜索扩展方案**：将代理搜索得到的系数作为目标模型的初始化，再用少量预算进行局部精修，在保持加速的同时达到甚至超越直接搜索的精度。
- **全面实验验证跨架构、跨方法、跨尺度的系数迁移可行性**：在 CLIP-ViT、SigLIP 和 Qwen3 三个模型家族、八项视觉任务和四项语言任务上验证，覆盖六种合并方法，结果具有一致性和可推广性。

## 方法详解
**统一合并形式化**：所有合并方法可表述为 $\theta_{merged} = \mathcal{M}(\theta_0, \{\tau_i\}_{i=1}^T; \boldsymbol{\alpha})$，其中任务向量 $\tau_i = \theta_i - \theta_0$，$\mathcal{M}$ 包含编辑函数 $f(\cdot)$ 和系数加权两个阶段。按粒度分为全局（单个标量 $\alpha$）、任务级（每个任务独立 $\alpha_i$）和层级别（每层每任务独立 $\alpha_{i,l}$）。

**αTransfer 核心机制**：
1. 在代理模型 $\theta_0^p$ 上执行任意合并方法搜索最优系数：$\boldsymbol{\alpha}^* = \arg\min_{\boldsymbol{\alpha}} \mathcal{L}^p(\theta_0^p + \sum_i \alpha_i \cdot \tilde{\tau}_i^p)$，其中 $\tilde{\tau}_i^p = f(\tau_i^p)$。
2. 将 $\boldsymbol{\alpha}^*$ 直接应用于目标模型：$\theta_{merged}^t = \theta_0^t + \sum_i \alpha_i^* \cdot \tilde{\tau}_i^t$，编辑函数 $f(\cdot)$ 仍在目标模型上原生执行。

**层-wise 迁移策略**：
- **复制迁移**：目标模型第 $b$ 块的对应代理块编号为 $b_{proxy} = \lceil \frac{b}{B_t} \cdot B_p \rceil$，其中 $B_t$ 和 $B_p$ 分别为目标模型和代理模型的块数。
- **插值迁移**：根据目标块在相对深度上的连续位置，线性插值最近两个代理块的系数以获得更平滑过渡。

**混合搜索**：将代理系数作为目标端初始值，分配少量计算预算进行目标端精修。对基于评估的方法执行局部网格搜索，对基于梯度的方法平分优化步数。

## 实验与结果
**数据集与任务**：视觉模型使用 CLIP-ViT 和 SigLIP，覆盖 DTD、EuroSAT、FER2013、Food101、GTSRB、RESISC45、Stanford Cars、SUN397 共 8 项图像分类任务；语言模型使用 Qwen3，覆盖 Banking77、DDXPlus、IFEval、Usefulness Judge 共 4 项语言任务。

**主要结果**：
- **视觉模型**（Table 3）：在 ViT-B/32 → ViT-L/14 上，Task Arithmetic 从 11.43h 降至 1.88h（6.07× 加速），显存从 2.12GB 降至 0.67GB（68.7% 节省）；SigLIP-Base → SigLIP-Large 上，TIES 精度损失仅 0.11%（77.50→77.39%），显存节省 62.3%。
- **LLM**（Table 4）：Qwen3-0.6B → 4B 上，Task Arithmetic 实现零精度损失（76.03→76.03%），加速 11.03×，显存从 40.22GB 降至 5.96GB（85.2% 节省）；TIES-Pairwise 加速 19.97×，精度 73.69%（与 Origin 一致）。
- **层-wise 迁移**（Table 5）：Copy 和 Interpolation 策略性能差距始终低于 0.5%，在多数设置下匹配或超越直接目标端优化。
- **混合搜索**（Table 6, Table 8）：对所有评估方法和 LLM 组合，混合搜索均恢复或直接超过完整目标端优化的精度，同时保持 1.20×–4.35× 的加速。
- **最强结果**：Qwen3-0.6B → 4B 上 Task Arithmetic 实现零精度损失的 11.03× 加速和 85.2% 显存节省。

## 相关工作脉络
- **Task Arithmetic (Ilharco et al., 2023)**：最早提出通过任务向量加权合并多模型的基础方法，使用均匀系数（grid search），本文在其框架内验证系数迁移的有效性。
- **TIES (Yadav et al., 2023)**：引入 trimming 和符号选举解决任务间冲突，采用全局系数搜索；本文发现其在 DDXPlus 任务上存在方向冲突被过度过滤的问题，提出 TIES-Pairwise 变体。
- **AdaMerging / AdaMerging++ (Yang et al., 2024)**：基于梯度的自适应合并方法，需同时加载多个模型，本文证明通过代理模型搜索可规避其显存瓶颈，但 AdaMerging++ 在高稀疏度（k=20）下迁移性受限。
- **DARE (Yu et al., 2024)**：通过随机掩码和重缩放合并，本文验证其在跨尺度下的系数迁移保持了较高精度（最大损失 0.49%）。
- **DivMerge (Touayouch et al., 2026)**：基于 Jensen-Shannon 散度的梯度优化方法，显存节省达 70.4%，说明其对内存瓶颈的缓解效果显著。
- **µTransfer (Yang et al., 2021) / DoReMi (Xie et al., 2023)**：同类"小模型搜索→大模型迁移"思路已在训练超参数和数据混合权重优化中成功应用，本文首次将该范式引入模型合并领域。

## 局限性与未来方向
- **代理与目标的系数相关系数影响迁移效果**：ViT-B/32 → ViT-L/14 场景因 patch size 差异大导致相关系数仅 0.58，迁移性能下降 1-3%，而非直接失败，表明架构一致性是影响迁移的关键因素。
- **AdaMerging++ 在高稀疏度下迁移性受限**：当 top-k 保留率 k=20 时，强烈掩码放大了代理和目标模型间的结构差异，降低了迁移效果；降低稀疏度可改善但可能影响单模型性能。
- **层-wise 迁移在非重复层（embedding、输出层）仅支持直接映射**：对于结构差异较大的模型对（如不同 patch size），重复块之间的相对深度映射可能不够精确。
- **未来方向**：探索跨配方（cross-recipe）和跨架构的系数映射、验证其他合并超参数的跨尺度迁移性、以及向更多下游场景（如多模态大模型）推广。

## 研究启发与可借鉴点
- **代理模型搜索范式的通用迁移价值**：αTransfer 的核心思想（在小模型上优化 → 迁移到大模型）可推广至模型合并之外的领域，如超参数搜索、数据采样策略优化、网络结构搜索等，可作为通用的"缩放加速"方法论。
- **统一形式化框架便于方法比较与组合**：将不同合并方法抽象为"编辑函数 + 系数搜索 + 粒度"三维度，为系统性理解现有方法提供了清晰的分析工具，可借鉴用于新方法的设计与评测。
- **混合搜索策略提供了效率-精度的灵活权衡**：从纯代理搜索到全目标搜索之间建立连续谱系，允许根据可用计算预算动态调整，此思路可应用于其他需要昂贵搜索的 ML 任务。
- **层-wise 相对深度映射的可复用设计**：针对跨层数模型的结构对齐问题，基于相对深度的插值/复制策略具有通用性，可迁移至其他跨规模模型适配场景。
- **与团队方向的结合机会**：若团队涉及多任务 LLM 微调或模型融合，可直接将 αTransfer 集成到现有合并流水线中，省去在目标模型上反复评估/优化的成本，特别适用于频繁迭代合并配方的场景。

## 关键术语表
**Task Vector（任务向量）**：微调后模型与预训练模型之间的参数差（$\tau_i = \theta_i - \theta_0$），编码了特定任务的知识，是大多数合并方法的基础表示。
**Model Merging（模型合并）**：在参数空间直接组合多个微调检查点的权重，无需访问原始训练数据即可获得多任务能力的单一模型。
**Alpha Transfer（系数迁移）**：在小型代理模型上搜索最优合并系数并迁移至同家族大型目标模型的核心方法。
**Editing Function（编辑函数）**：合并方法中对任务向量进行变换的操作（如 TIES 的 trimming+符号选举、DARE 的随机掩码），嵌入在合并算子 $\mathcal{M}$ 中。
**Coefficient Granularity（系数粒度）**：合并系数定义的层次，包括全局（single scalar）、任务级（per-task）和层级别（per-layer per-task）。
**Spearman Rank Correlation（Spearman 秩相关）**：用于量化代理和目标模型在合并系数空间中性能排序一致性的统计量，本文用于验证跨尺度分布相似性。
**Hybrid Search（混合搜索）**：将代理搜索得到的系数作为目标端初始值，再用少量目标端预算进行局部精修的扩展方案。
**TIES-Pairwise**：本文提出的 TIES 变体，分两阶段合并以避免 DDXPlus 等任务因方向冲突被过度过滤。

## 可复现要素
- **数据集**：视觉上 DTD、EuroSAT、FER2013、Food101、GTSRB、RESISC45、Stanford Cars、SUN397；语言上 Banking77、DDXPlus、IFEval、Usefulness Judge（来自 Nemotron-SFT-Instruction-Following-Chat-v2 和 Chen et al. 2026）。均为公开数据集。
- **代码**：论文未提及代码开源状态。
- **模型权重**：使用 OpenAI CLIP-ViT、Google SigLIP 和 Qwen3 的官方预训练权重，均为公开可用。
- **关键超参**：Fine-tuning 使用 AdamW（$\beta_1=0.9, \beta_2=0.999$），学习率 {1e-5, 3e-5}（视觉）或 {1e-5, 4e-5}（LLM），batch size 32/48（视觉）或 36（LLM），epoch 5-20 不等。合并搜索范围：Task Arithmetic 在 (0,1] 取 16 点，DARE/TIES 在 (0,1.8] 取 18 点。梯度优化步数：AdaMerging 500 步，DivMerge 100 步。
