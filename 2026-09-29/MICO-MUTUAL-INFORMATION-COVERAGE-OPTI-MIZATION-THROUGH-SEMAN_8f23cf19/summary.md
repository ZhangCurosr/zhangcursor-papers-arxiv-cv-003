---
title: "MICO-MUTUAL-INFORMATION-COVERAGE-OPTI-MIZATION-THROUGH-SEMAN"
source: https://arxiv.org/pdf/2609.34330v1.pdf
model: agnes-2.5-flash
chunks: 5
summarized_at: "2026-10-01 09:57:04"
---

# 论文速读：MICO-MUTUAL-INFORMATION-COVERAGE-OPTI-MIZATION-THROUGH-SEMAN

## 一句话总结
论文从任务 log-loss 出发建立语义擦除信道模型，推导出一般化的互信息覆盖目标，并据此提出免训练的两阶段贪心剪枝方法 MiCo；该方法利用单调次模代理目标实现视觉 token 的精准筛选，在极低预算下保留主流 MLLM 绝大部分性能的同时显著降低推理延迟与显存占用。

## 研究问题与动机
- MLLM 面对海量视觉 token 导致计算成本与显存开销过高，现有启发式剪枝策略易误删关键视觉信息，造成性能大幅下降。
- 单一维度的重要性代理（如仅依赖注意力或特征范数）难以同时刻画信息量、可靠性和任务相关性，缺乏统一的信息论选择准则。
- 视频与长序列场景下，固定比例保留或单层剪枝无法适配动态任务需求，且部分剪枝方法自身的计算开销反而抵消加速收益。
- 现有压缩方法在图像降采样、细粒度 OCR/图表、指代表达理解等困难任务上鲁棒性不足，缺乏系统性的理论保证与广泛基准验证。

## 核心贡献（创新点）
- 从任务 log-loss 严格推导互信息覆盖目标，建立语义擦除信道模型并将信息增益上界分解为可靠性与覆盖率之积，为 token 剪枝提供信息论依据。
- 提出免训练两阶段贪心选择器 MiCo，Stage 1 在视觉编码器侧粗筛候选池，Stage 2 在 LLM 解码器内基于文本查询完成任务感知精炼，全程无需微调。
- 证明代理目标函数满足单调非负次模性质，贪心增量选择可保证达到全局最优解的 $(1-1/e)$ 近似比，兼顾理论保证与工程效率。
- 在图像理解、视频理解、指代定位与细粒度文本/图表等多类基准上系统验证，MiCo 以 5.6%–10% 的 token 保留率实现接近 Vanilla 的性能并保持亚秒级额外开销。

## 方法详解
- **信息论推导**：预测 log-loss 可分解为条件熵 $H(Y|Z,q,g)$ 与 KL 散度项，最小可达 Bayes 风险为 $R^*_Z(q,g)=H(Y|Z,q,g)$；恢复信息增益上界为 $\Delta R^*(S;q,g)\le H(\mathbf{U}_V|q,g)-\mathcal{I}_{task}(S;q,g)$，其中 $\mathcal{I}_{task}$ 为语义查询的任务加权互信息。
- **语义擦除信道建模**：将 token $v$ 的保留过程建模为信道，成功恢复概率等于可靠性 $\Pr(C_v=1|q,g)$ 与覆盖率 $M_v(S;q,g)$ 的乘积，从而得到代理互信息上界。
- **代理目标函数**：$\widehat{\mathcal{I}}_{task}(S)=\sum_{v\in V}\widehat{\mathcal{H}}_v\,\widehat{\mathcal{R}}_v\,\widehat{\mathcal{P}}_v\,\widehat{M}_v(S)$，覆盖函数 $\widehat{M}_v(S)=\max_{s\in S}[\frac{x_v^\top x_s}{\|x_v\|\|x_s\|}]_+$ 取候选集中与目标 token 最大余弦相似度的正部分。
- **Stage 1（视觉编码器侧）**：在 token 进入 LLM 前构建规模 $N_1=2T$ 的候选池。信息量代理为 $\widehat{\mathcal{H}}_v=\|x_v^{\text{vis}}\|_2$，可靠性代理为 $\widehat{\mathcal{R}}_v=\overline{A}_{\text{CLS}\to v}^{\text{vis}}$，任务相关性代理置为 1。
- **Stage 2（LLM 解码器侧）**：在选定层 $\ell_\star$ 后对候选池进行任务感知子集选择。信息量代理为 $\widehat{\mathcal{H}}_v=\|x_v^{(\ell_\star)}\|_2$，可靠性代理置为 1，任务相关性代理为所有文本 token 到视觉 token 的平均注意力 $\overline{A}_{\text{text}\to v}^{(\ell_\star)}$。
- **贪心选择与权重融合**：统一权重 $\widehat{W}_v=\widehat{\mathcal{H}}_v\widehat{\mathcal{R}}_v\widehat{\mathcal{P}}_v$，从空集开始迭代选取使加权覆盖增量最大的 token，直至满足预算 $|S|\le B$；理论证明该贪心策略至少达到最优值的 $(1-1/e)$。

## 实验与结果
- **评测基准与协议**：涵盖 GQA、ScienceQA-IMG、AI2D、MMMU、MMStar、POPE、HallusionBench、MME、TextVQA、MMBench、SEED-Bench、ChartQA、DocVQA、OCRBench、RefCOCO/+/g、VTC-Bench、VideoMME、MVBench、LongVideoBench、MLVU、gRefCOCO；解码统一采用 batch size=1 确定性贪心、最大生成 2048 token。
- **LLaVA 系列**：LLaVA-NeXT-13B 仅保留 5.6% token（160/2880）即保留 97.5% 性能，单 token 推理加速 3.8×（POPE F1 99.62%）；12 组 model-budget 设置中 11 组综合分最高（仅 NeXT-7B @77.8% 被 VScan 以 +0.3 分小幅超越）。
- **先进架构**：在 Qwen2.5-VL-7B/8B、Qwen3-VL-
