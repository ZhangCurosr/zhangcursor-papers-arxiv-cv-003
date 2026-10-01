---
title: "MICO-MUTUAL-INFORMATION-COVERAGE-OPTI-MIZATION-THROUGH-SEMAN"
source: https://arxiv.org/pdf/2609.34330v1.pdf
model: agnes-2.5-flash
chunks: 5
summarized_at: "2026-10-01 09:56:18"
---

# 论文速读：MICO-MUTUAL-INFORMATION-COVERAGE-OPTI-MIZATION-THROUGH-SEMAN

## 一句话总结
本文提出 MiCo，一种基于互信息覆盖理论优化的无训练两阶段视觉 token 剪枝方法，从任务 log-loss 上界推导出可观测的单调次模代理目标，在严格预算约束下贪心选取代表性且任务相关的 token，实现多模态大模型推理的显著加速与性能保留。

## 研究问题与动机
- 多模态大语言模型（MLLM）处理海量视觉 token 带来高昂计算与显存成本，亟需高效剪枝。
- 现有剪枝方法多依赖启发式规则，注意力排名易保留语义冗余 token，覆盖类方法可能选入与当前查询无关的区域。
- 两者结合方案在高剪枝率下性能下降明显，缺乏从信息论角度严谨推导并统一指导 token 选择的理论目标。
- 如何在不微调模型的前提下，兼顾视觉特征的内在代表性与当前语言查询的任务相关性，是提升 MLLM 推理效率的关键瓶颈。

## 核心贡献（创新点）
- **从任务 log-loss 推导互信息覆盖目标**：通过语义擦除模型建立理论因子与可计算代理的映射，将剪枝优化转化为最大化任务加权互信息下界的问题。
- **提出无训练两阶段贪心剪枝框架**：Stage 1 依赖纯视觉信号粗筛候选池，Stage 2 利用文本到视觉注意力精筛至目标预算，全程无需梯度更新或微调。
- **设计单调次模可观测代理函数**：将信息量、可靠性、任务相关性与余弦相似度覆盖分解为范数与注意力权重的乘积，严格满足次模性。
- **提供 $(1-1/e)$ 近似比理论保证**：在基数约束下贪心迭代添加增量最大 token，保证代理目标至少达到全局最优解的 $63\%$。
- **跨架构与跨任务的 SOTA 验证**：在 7B–13B 主流 MLLM 及图像/视频/grounding/OCR 等多样化基准上，以极高剪枝率取得领先性能保留率，且开销可解析分解。

## 方法详解
- **两阶段选择架构**：
  - **Stage 1（视觉代表性粗筛）**：视觉 token 进入 LLM 前，仅使用视觉隐藏状态 $L_2$ 范数 $\|x_v^{\text{vis}}\|_2$ 与视觉编码器 CLS 到 token 的平均注意力 $\overline{A}_{\text{CLS}\to v}^{\text{vis}}$，从全量 token 中选出大小为 $N_1=2T$ 的候选池。
  - **Stage 2（任务相关性精筛）**：在解码器指定层 $\ell_\star$ 后，利用文本到视觉 token 的平均注意力 $\overline{A}_{\text{text}\to v}^{(\ell_\star)}$ 进行子集选择，使各层保留 token 平均数严格等于目标预算 $T$。
- **信息论推导与代理目标**：
  - 理论目标是最大化任务加权互信息 $\mathcal{I}_{\text{task}}(S;q,g)=\sum_v \Pr(J=v|Q=q) I(U_v;\widehat{U}_v(S)|q,g)$，其对应的 Bayes log-loss 风险上界为 $\Delta R^*(S) \le H(\mathbf{U}_V|q,g) - \mathcal{I}_{\text{task}}(S)$。
  - 通过语义擦除模型将不可观测项拆解为三个可计算代理的乘积：信息量 $\widehat{\mathcal{H}}_v$、可靠性 $\widehat{\mathcal{R}}_v$、任务相关性 $\widehat{\mathcal{P}}_v$，以及覆盖代理 $\widehat{M}_v(S)=\max_{s\in S}\kappa_X(v,s)$（余弦相似度）。
  - 最终代理目标函数为 $\widehat{\mathcal{T}}_{\text{task}}(S)=\sum_v \widehat{W}_v \max_{s\in S}\kappa_X(v,s)$，其中 $\widehat{W}_v=\widehat{\mathcal{H}}_v\widehat{\mathcal{R}}_v\widehat{\mathcal{P}}_v$。
- **贪心优化与性质**：
  - 目标函数经归一化后满足非负、单调、次模性。
  - 在基数约束 $|S|\le B$ 下，贪心依次添加边际增量最大的 token，理论保证 $\widehat{\mathcal{T}}_{\text{task}}(S_b) \ge (1-1/e)\widehat{\mathcal{T}}_{\text{task}}(S^\star)$。
  - 注：该保证仅为单次 selector 调用的局部性质，不承诺跨两阶段的联合全局最优。

## 实验与结果
- **数据集与基准**：图像理解（GQA、ScienceQA-IMG、AI2D、MMMU、MMStar、TextVQA、MMBench-EN/CN、SEED、POPE、HallusionBench、MME）、细粒度/OCR（ChartQA、DocVQA、OCRBench、RefCOCO/+/gRefCOCO、VTC-Bench）、视频（VideoMME、MVBench、LongVideoBench、MLVU）。
- **评估模型**：LLaVA-1.5/NeXT（7B/13B）、Qwen2.5-VL/3-VL/3.5-VL（7B/8B/9B）、InternVL3-8B、LLaVA-Video-7B。
- **主要结果数字**：
  - LLaVA-NeXT-13B：仅用 **5.6%** 视觉 token 保留 **97.5%** 基线性能，POPE F1 达 **99.62%**，单 token 加速 **3.83×**。
  - LLaVA-1.5/NeXT（**94.4%** 剪枝率）：性能保留率最高达 **97.5%**（NeXT-13B）；在 88.9% 剪枝下 NeXT-13
