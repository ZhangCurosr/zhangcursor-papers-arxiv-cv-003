---
title: "Just-MLPs-Eficient-Visual-State-Reconstruction-for-Multimoda"
source: https://arxiv.org/pdf/2609.34972v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 04:09:24"
field: "多模态模型效率优化"
keywords: ["多模态大语言模型", "视觉token压缩", "低秩适配", "参数高效微调", "推理加速", "知识蒸馏"]
innovations: ["揭示视觉表征在隐藏维度的低秩可压缩性与跨层可预测性", "提出δ-Vision框架用轻量级低秩adapter替代逐层Transformer演化", "设计embedding与recurrent两种记忆构建变体并验证与token剪枝的正交性"]
benchmarks: ["MMStar", "RWQA", "GQA", "MMB", "MME", "POPE", "SQA", "VQAv2", "MuirBench", "Video-MME", "MVBench"]
---

# 论文速读：Just-MLPs- Efficient Visual State Reconstruction for Multimodal Language Models

## 一句话总结
论文提出δ-Vision，通过轻量级低秩adapter替代多模态大语言模型（MLLM）中视觉token在全Transformer层中的重复演化，在保留全部视觉token的前提下显著降低计算开销。研究发现视觉信息在隐藏维度上高度可压缩且具有强可预测性，从而实现更优的精度-效率权衡。

## 研究问题与动机
- **核心问题**：MLLM中长视觉token序列导致巨大的推理计算开销，现有方法通过剪枝冗余视觉token来缩短序列，但会永久丢弃可能在后续层中重获重要性的视觉证据。
- **动机**：视觉token是否必须经过完整的逐层Transformer计算？能否保留所有视觉token的同时，用更低成本的方式构建层特定的视觉表示？

## 核心贡献（创新点）
1. **揭示视觉表征的低维可压缩性与跨层可预测性**：通过低秩干预实验表明，阻断视觉到文本注意力后仅恢复少数方向即可恢复大部分准确率，且轻量级MLP能以高余弦相似度预测各层视觉状态。
2. **提出δ-Vision框架**：用低秩adapter替代视觉token的完整Transformer演化，保留所有视觉token作为只读KV上下文，而非削减token数量。
3. **设计两种记忆构建变体**：Embedding Adapter直接从初始视觉embedding预测各层记忆，Recurrent Adapter通过逐层低秩修正累积记忆轨迹。
4. **提供全面的精度-效率分析**：在单图、多图、视频基准上均优于视觉token剪枝基线，在Qwen3-VL-4B上以5% token保留量的剪枝方法相比提升9.1平均得分，同时FLOPs降至17.90%。

## 方法详解
- **视觉记忆构建**：给定初始视觉嵌入$E \in \mathbb{R}^{N_v \times d}$，第$l$层的adapter构造残差修正$\Delta_l(X) = \phi(XD_l)U_l$，其中$D_l \in \mathbb{R}^{d \times r}$、$U_l \in \mathbb{R}^{r \times d}$为低秩矩阵，$r \ll d$，$\phi$为SiLU激活。
- **Embedding Adapter**：$M_l = E + \Delta_l(E)$，每层独立从初始嵌入预测记忆，层间条件独立。
- **Recurrent Adapter**：$M_0 = E$，$M_l = M_{l-1} + \Delta_l(M_{l-1})$，通过序列低秩修正累积视觉记忆。
- **作为上下文使用**：预测记忆经冻结的LN和KV投影得到$K_l^v, V_l^v$，文本query与之计算注意力，视觉query、视觉注意力输出、视觉FFN被完全省略。
- **训练目标**：冻结主干网络，仅优化adapter，采用Supervised-KD（老师强制下的KL散度损失），在答案token位置上最小化教师与学生预测分布的差异。

## 实验与结果
- **数据集与基准**：单图（MMStar、RWQA、GQA、MMB、MMB-CN、MME、POPE、SQA、VQAv2）、多图（MuirBench）、视频（Video-MME、MVBench）。
- **基线**：免训练剪枝（FastV、VisionZip、DART、DivPrune、ZOO-Prune、SparseVLM）与训练-based（LLaVA-Mini、EPIC）。
- **主要结果**：Qwen3-VL-4B上δ-Vision（5%等效计算预算）平均得分74.4，较最强剪枝基线DivPrune（5% retention）提升9.1分，保留 uncompressed 模型92.9%性能；Recurrent Adapter进一步提升至75.4。
- **效率**：Video-MME上FLOPs为原始模型的17.90%，总加速1.30×，prefill加速1.50×，显存峰值仅略增（12.97GB vs 12.94GB）。
- **跨架构泛化**：在LLaVA-1.5-7B、Qwen3-VL-30B-A3B、Qwen3.5-4B上均保持94-96%的vanilla性能，增益尤其在Qwen系列显著。

## 相关工作脉络
- **视觉token剪枝**（FastV、VisionZip、DART、DivPrune等）：通过缩短序列长度降低计算，但不可逆丢弃视觉证据；δ-Vision改变视觉状态的构建方式而非削减token数量，提供正交的效率轴。
- **视觉表示冗余利用**（LRCP、EE-MLLM、ShortV）：识别主成分子空间或跳过无效层；δ-Vision学习轻量级adapter直接预测层特定记忆，而非移除或简化原始计算。
- **KV缓存预测**（KV Prediction）：用辅助模型预测大模型的KV cache；δ-Vision在KV投影前预测视觉记忆并复用冻结投影，结构更简单。
- **跨层bypass**（SwiftVLM）：通过残差连接恢复未选中token；δ-Vision完全避免token选择，所有视觉token始终以低秩修正形式参与。
- **低秩自适应**（LoRA等）：通常用于参数高效微调；δ-Vision将其应用于视觉状态的重构而非权重更新。
- **蒸馏方法**（OPD、Supervised-KD）：对比显示teacher-forced KD避免on-policy rollout开销，训练时间减少约8倍。

## 局限性与未来方向
- **adapter秩的选择**：论文显示增大秩$r$可带来适度提升，但未给出跨任务的最优秩自动搜索策略。
- **层跳过与泛化**：虽发现早期和晚期层对视觉依赖较低，可安全跳过以提升效率，但跳跃策略可能依赖特定模型架构，跨架构普适性待验证。
- **多模态数据分布**：训练数据主要来自单图、多图和视频指令数据，对极端长视频或高分辨率图像的处理效率未充分探讨。
- **与结构化剪枝的结合**：虽然展示可与DART/DivPrune组合，但联合优化的理论边界和实际收益仍需深入分析。
- **推理部署**：当前分析基于FLOPs和速度模拟，实际硬件上的内存访问模式和算子融合效率未全面评估。

## 研究启发与可借鉴点
- **低秩干预实验范式**：通过阻断信息流并逐步恢复低秩方向来量化任务相关信息的维度，可迁移至其他模态（如音频、文本）的冗余分析。
- **层特定可预测性验证**：使用独立轻量级MLP预测各层隐藏状态，评估余弦相似度和MSE，可作为通用方法用于分析任何神经网络中间表示的压缩潜力。
- **记忆构造与上下文分离**：将状态构建（adapter）与状态消费（冻结KV投影）解耦的设计，可推广至序列建模中的外部记忆机制。
- **Supervised-KD训练策略**：冻结主干、仅训练轻量模块的teacher-forced蒸馏，在保持效率的同时避免on-policy rollout的高成本，适用于各类参数高效适配场景。
- **正交效率轴思维**：与token剪枝正交的“重构而非削减”思路，为多模态推理优化提供了新的设计空间，可启发其他维度的效率改进。

## 关键术语表
- **δ-Vision**：一种轻量级视觉记忆预测框架，用低秩adapter替代MLLM中视觉token的逐层Transformer演化。
- **Embedding Adapter**：直接从初始视觉嵌入预测每层视觉记忆的变体，层间记忆条件独立。
- **Recurrent Adapter**：通过逐层低秩修正累积视觉记忆的变体，保留显式的层间状态传播。
- **Supervised-KD**：在答案token位置上最小化教师与学生预测分布KL散度的训练目标，采用teacher-forcing方式。
- **低秩干预**：将视觉隐藏状态投影到低维子空间再重构，以分析视觉信息的有效维度。
- **视觉记忆**：adapter预测的层特定视觉表示，作为只读KV上下文供文本查询使用。
- **r_95**：保留95%奇异值能量所需的最低秩，用于衡量视觉计算谱结构的集中度。
- **有效秩（ER）**：基于归一化奇异值谱计算的熵度量，反映实际利用的独立方向数。

## 可复现要素
- **数据集**：allenai/pixmo-ask-modelanything（单图）、allenai/Molmo2-MultiImageQA（多图）、lmms-lab/M4-Instruct-Data、Video-R1/Video-R1-data（视频），均已公开。
- **代码**：论文提供GitHub链接（DeCLaRe Lab），但具体仓库地址未在本Markdown中明确列出。
- **权重**：未提及预训练adapter权重是否开源。
- **关键超参**：瓶颈秩$r=128$，AdamW优化器，学习率$5 \times 10^{-5}$，weight decay 0.01，梯度裁剪1.0，余弦学习率调度（3% warmup，最小学习率比0.1），固定随机种子44。单图训练2000步，多图/视频训练4000步，全局batch size 32，8 GPU。
