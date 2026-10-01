---
title: "Just-MLPs-Eficient-Visual-State-Reconstruction-for-Multimoda"
source: https://arxiv.org/pdf/2609.34972v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 04:09:57"
field: "多模态大模型高效推理"
keywords: ["multimodal large language models", "visual token compression", "efficient inference", "low-rank adaptation", "knowledge distillation", "visual memory prediction"]
innovations: ["用低秩适配器逐层重建视觉记忆并复用冻结KV投影", "提出视觉-文本计算解耦的轻量适配范式", "证明层级视觉状态可压缩且可预测并与剪枝互补"]
benchmarks: ["MMStar", "RWQA", "GQA", "MMB", "MMB-CN", "MME", "POPE", "SQA", "VQA-v2", "MuirBench", "Video-MME", "MVBench"]
---

# 论文速读：Just-MLPs-Eficient-Visual-State-Reconstruction-for-Multimoda

## 一句话总结
论文提出 δ-Vision，一种通过轻量级低秩适配器替代多模态大语言模型中重复的视觉 Transformer 计算的视觉记忆预测框架，在保留全部视觉令牌的前提下显著降低推理开销，同时在图像与视频基准上优于视觉令牌剪枝方法。

## 研究问题与动机
- 多模态大语言模型（MLLM）处理高分辨率图像/视频时，会产生大量视觉令牌，且这些令牌在每个 Transformer 层都要参与注意力与前馈计算，造成主要计算开销。
- 现有主流方案通过剪枝冗余视觉令牌来缩短序列长度，但剪枝是不可逆的：被丢弃的视觉证据在后续层无法再被文本检索利用。
- 作者提出一个互补视角：不减少视觉令牌数量，而是降低每个视觉令牌在每层中被完整 Transformer 更新的计算代价。
- 初步分析发现，视觉隐状态的隐藏维度方向具有强压缩性（低有效秩），且各层视觉状态可从初始嵌入用轻量 MLP 高精度预测，从而支撑"用适配器重建视觉记忆"的设计动机。

## 核心贡献（创新点）
- 发现 MLLM 中层级视觉状态在隐藏维度上存在显著冗余，并具有较强的跨层可预测性；与剪枝类方法只关注"是否保留令牌"不同，本文关注"如何以更低维方式重建层级视觉表示"。
- 提出 δ-Vision 的层内低秩残差参数化，使每层视觉记忆更新被约束在一个低维子空间内；与直接蒸馏或微调整个骨干的方法相比，仅引入轻量适配器并保持主干冻结。
- 给出两种记忆构造变体（Embedding Adapter 与 Recurrent Adapter），分别对应"每层独立从初始嵌入校正"和"跨层逐步累积校正"，并与仅改变序列长度的剪枝形成互补效率轴。
- 使用 Supervised-KD 训练，利用冻结原始 MLLM 作为教师提供词级分布监督；与 SFT 和 on-policy 蒸馏相比，在保持最高平均性能的同时避免高昂的自回归 rollout 开销。
- 在多种 MLLM 骨干与图像/多图/视频基准上验证，证明该方法在可比或更低计算预算下取得更好准确率，并可与现有剪枝方法组合使用。

## 方法详解
- 层-wise 视觉记忆预测：给定输入视觉状态 $X \in \mathbb{R}^{N_v \times d}$，第 $l$ 层适配器定义为 $A_l(X) = X + \Delta_l(X)$，其中 $\Delta_l(X) = \phi(X D_l) U_l$，$D_l \in \mathbb{R}^{d \times r}$、$U_l \in \mathbb{R}^{r \times d}$ 为低秩参数，$r \ll d$，$\phi$ 为 SiLU；初始化 $U_l = 0$，使初始映射恒等。
- Embedding Adapter：每层记忆直接从初始视觉嵌入预测，$M_l = E + \Delta_l(E)$，各层记忆在给定 $E$ 条件下条件独立，无需跨层显式传播。
- Recurrent Adapter：记忆通过逐层残差更新累积，$M_0 = E$，$M_l = M_{l-1} + \Delta_l(M_{l-1})$，以紧凑低维转移近似完整 Transformer 演进轨迹。
- 视觉记忆作为 KV 上下文：将 $M_l$ 送入冻结的层归一化与原始 $W_{K,l}$、$W_{V,l}$，得到 $K^v_l$、$V^v_l$；文本侧按原式计算 $Q^t_l$、$K^t_l$、$V^t_l$，文本查询联合注意力 $O^t_l = \text{Attn}(Q^t_l, [K^v_l; K^t_l], [V^v_l; V^t_l])$，因果结构保持不变。
- 视觉侧不再计算 $Q^v_l$、视觉注意力输出与视觉 FFN；每层 $M_l$ 仅作为只读 KV 上下文被消费一次，下一层 $M_{l+1}$ 由对应适配器重新生成，而非由文本注意力更新。
- 训练目标采用 Supervised-KD：冻结视觉编码器、投影器与语言主干，仅训练适配器；在答案位置集合 $\Omega$ 上最小化教师与学生条件分布的前向 KL 散度 $\mathcal{L}_{KD} = \frac{1}{|\Omega|}\sum_{t \in \Omega} D_{KL}(p_T(\cdot|I,x_{1:T},y_{<t}) \| p_S(\cdot|I,x_{1:T},y_{<t}))$。
- 低秩干预分析支撑设计：对视觉注意力输出等量做谱分析，保留 95% 奇异值能量所需秩远低于全维；在逐层顺序低秩干预下，恢复少量方向即可恢复大部分性能，且不同深度层对视觉信息的依赖呈显著非均匀性。

## 实验与结果
- 模型与设置：默认以 Qwen3-VL-4B-Instruct 与 Embedding Adapter 为主配置；适配器瓶颈秩 $r=128$；AdamW、学习率 $5\times10^{-5}$、权重衰减 0.01、梯度裁剪 1.0、余弦调度与 3% warmup；单图训练 2000 步、批量 32，多图/视频训练 4000 步、同硬件与批量。
- 单图基准：在 5% 视觉令牌保留下，δ-Vision（Emb）平均 74.4，比最强训练无关 Baseline DivPrune 的 65.3 高 9.1 分；亦优于训练型 Baseline LLaVA-Mini（65.0）与 EPIC（66.9）。Emb 平均 74.4，Rec 进一步提升至 75.4。
- 效率（Video-MME，5% 保留对照）：δ-Vision 使用 17.90% FLOPs，总加速 1.30×，prefill 加速 1.50×，与 DivPrune/VisionZip 等同档 FLOPs 剪枝方案效率相当，但精度明显更高。
- 跨骨干泛化：在 LLaVA-1.5-7B、Qwen3-VL-30B-A3B、Qwen3.5-4B 上均保留约 94%–96% 原始性能；Qwen3-VL-30B-A3B 平均 78.6（DivPrune 71.3），Qwen3.5-4B 平均 71.6（DivPrune 62.7）。
- 多图与视频：仅单图训练的 Emb 在 MuirBench/Video-MME/MVBench 平均 49.5；加入多图/视频数据后提升至 52.0，仍优于 5% 保留下最强训练无关 Baseline 的 50.2。
- 附加发现：与 DART/DivPrune 组合时，50% 保留仅使 Emb 平均从 74.4 下降 1.2–1.3 分；层跳过可在精度小幅下降时进一步降低 FLOPs（如 17.90%→14.89%）并提升 prefill 加速。

## 相关工作脉络
- 视觉表征冗余与子空间压缩：LRCP 用 PCA 识别主视觉子空间并据此选择令牌；本文与之不同，不仅压缩跨令牌冗余，还直接替换跨层重复 Transformer 演化，用适配器重建各层视觉记忆。
- 层内计算跳过/去除：EE-MLLM 去掉视觉 self-attention，ShortV 等在无效层冻结视觉令牌；本文不删除或冻结整段计算，而是用低秩残差修正近似其效果。
- 训练无关令牌剪枝：FastV、VisionZip、DivPrune、SparseVLM、DART、ZOO-Prune 等通过缩短视觉序列降成本；本文的核心差异是保留全部令牌并为每层构造独立 KV 上下文。
- 训练型压缩：LLaVA-Mini、EPIC 等学习更紧凑的视觉表示；本文通过冻结主干+蒸馏适配器的路径，避免重新训练大体积视觉/语言组件。
- KV 预测：KV Prediction 用辅助小模型预测更大模型的 KV cache；本文在投影前用轻量适配器预测视觉记忆，并复用冻结的 $W_K$、$W_V$。
- 跨层 bypass：SwiftVLM 用跨层旁路与残差恢复未选令牌；本文不使用旁路恢复，而以每层独立记忆构造取代跨层传播。

## 局限性与未来方向
- 需要额外适配训练；虽然主干冻结且参数小，但在零样本/无训练场景中无法直接部署。
- 极端压缩与复杂长视频下性能仍有下降空间；当前最佳结果多基于单图训练，多图/视频性能依赖额外数据。
- 层依赖呈非均匀性，但未给出自动化的层选择或动态跳过策略，主要依赖固定跳过作为附加分析。
- 主要在 Qwen 与 LLaVA 系列验证；对更多原生架构（如含不同注意力变体的模型）的普适性仍需扩展。
- 低秩方向以固定秩为主，实际任务间对 $r$ 的敏感度未充分自适应化。

## 研究启发与可借鉴点
- 将"保留令牌数量"与"压缩表征演化"视为两个正交效率轴，为后续工作提供可组合的压缩思路。
- 低秩残差适配器以恒等初始化并配合 Teacher-Forced KD，可在不破坏已有能力的情况下快速适配新效率路径。
- 层-wise 干预与谱分析可用于诊断各层视觉依赖强度，指导选择性计算分配。
- 视觉记忆仅作只读 KV 上下文、不参与文本反向更新，这一设计避免了训练与推理时的状态耦合，便于通用接入。
- 本团队可将其与序列剪枝、动态层跳过、或跨任务蒸馏结合，探索更高压缩率下的稳定部署方案。

## 关键术语表
- **δ-Vision**：一种用轻量低秩适配器逐层重建视觉记忆、并以冻结 KV 投影供文本检索的效率方法。
- **Embedding Adapter**：从初始视觉嵌入直接预测每层视觉记忆的适配器变体。
- **Recurrent Adapter**：从前一层视觉记忆出发逐级累积低维残差修正的适配器变体。
- **Supervised-KD**：以冻结原始 MLLM 为教师、在答案位置做分布匹配的蒸馏损失。
- **visual token pruning**：通过减少视觉令牌数量来降低 MLLM 推理开销的训练无关方法族。
- **low-rank adaptation**：用低秩矩阵参数化表示更新，以降低适配参数与计算代价的技术。
- **effective rank / r_95**：分别用熵与奇异值能量阈值刻画矩阵内在维度的谱指标。
- **layer-wise skipping**：在有选择地跳过部分层适配器以进一步降低计算开销的策略。

## 可复现要素
- 数据集：allenai/pixmo-ask-model-anythang、allenai/Molmo2-MultiImageQA、lmms-lab/M4-Instruct-Data、Video-R1/Video-R1-data，以及 MMStar、RWQA、GQA、MMB、MMB-CN、MME、POPE、SQA、VQA-v2、MuirBench、Video-MME、MVBench；论文公开研究与 Hugging Face 数据集链接，数据可获取。
- 代码/权重：论文提供 GitHub 链接（DeCLaRE Lab），但未在当前正文给出具体仓库 URL，需以论文页脚为准。
- 关键超参：适配器瓶颈秩 $r=128$；AdamW；学习率 $5\times10^{-5}$；权重衰减 0.01；梯度裁剪 1.0；余弦调度、3% warmup、最低比率 0.1；单图 2000 步、批量 32；多图/视频 4000 步、同硬件与批量；随机种子 44；禁用 DeepStack。
