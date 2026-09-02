---
title: "TempCloze-Can-Video-LLMs-Identify-the-Missing-Middle"
source: https://arxiv.org/pdf/2609.01515v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 22:22:13"
field: "多模态大模型评测"
keywords: ["Video-LLM", "时序推理", "视频完形填空", "benchmark", "视觉推理", "Temporal Reasoning"]
innovations: ["提出TempCloze视频完形填空基准，以纯视觉候选替代文本选项消除语言捷径", "设计三维度(Semantic/Alignment/Progression)同源干扰项机制，系统化拆解视觉时序推理能力", "发现Alignment(时间对齐)是当前Video-LLM视觉时序推理的主要瓶颈"]
benchmarks: ["TempCloze", "TempCloze-Mixed", "TempCloze-Hard"]
---

# 论文速读：TempCloze: Can Video-LLMs Identify the Missing Middle?

## 一句话总结
论文提出了 TempCloze，一个基于视频完形填空范式的基准，用于评估 Video-LLMs 的视觉时序推理能力；通过让模型从四个候选片段中选出正确的"缺失中间段"，有效规避了传统 VideoQA 中文本选项带来的语言捷径问题，并发现当前模型的主要瓶颈在于**时间对齐（Alignment）**而非语义理解。

## 研究问题与动机
- 现有 Video-LLM 时序推理评测（如 TempCompass、TVBench）仍依赖自然语言提问与文本选项，模型可通过选项措辞、答案相关性或语言先验获得虚假高分，无法真实反映视觉时序推理能力。
- 预训练语言模型在无需多模态上下文的情况下即可超越随机基线 25% 以上，说明语言捷径问题严重。
- 缺少直接要求模型在视觉片段之间进行证据比较的评测范式，难以诊断模型在"事件发生时间"这一维度上的真实水平。
- 现有完形填空（Cloze）任务多用于视频表示学习的训练目标（如 VideoBERT、VideoMAE），尚未被系统性地用于 Video-LLM 的评测。

## 核心贡献（创新点）
- **提出 TempCloze 视频完形填空基准**，以 1,521 个视频、三个维度（Semantic/Alignment/Progression）系统化评估 Video-LLM 的视觉时序推理能力；与已有工作本质区别在于将完形填空从训练目标转变为评测工具，并以纯视觉候选替代文本选项消除语言捷径。
- **设计同源干扰项机制**，所有干扰片段来自同一视频，共享场景与物体，从语义、时间对齐、事件进程三个可控维度构造区分度强的干扰项；与已有 VideoQA 基准使用自由文本或无关选项的本质区别在于强制模型依赖时序证据而非外观匹配。
- **系统评估 31 个 Video-LLM（10 私有 + 21 开源）并定位 Alignment 为主要瓶颈**；发现最强模型在 Alignment 维度仅达 76.92%，远低于人类基线的 98%，揭示当前模型的时序推理存在结构性短板；与已有评测仅报告综合分数的本质区别在于提供了多维度拆解与误差模式分析。
- **提供 Error Pattern Analysis 与 Behavioral Sensitivity Analysis**，分析候选顺序、上下文方向、可见跨度、帧密度和测试时扩展对模型选择的影响，揭示了模型时序推理的不稳定性来源。

## 方法详解
- **任务定义**：视频 $V$ 分为三段 $V = [B \mid M \mid E]$，其中 $B = C_{0,s}$（开头）、$M = C_{s,e}$（缺失中间）、$E = C_{e,T}$（结尾）。模型需从候选集合 $\mathcal{Y}_d = \{M, D_d^1, D_d^2, D_d^3\}$ 中选择正确中间片段。
- **三维度干扰项设计**：
  - **Semantic（S）**：使用同持续时间的非重叠区间片段 $D_S = C_{u, u+\ell}$，$(u, u+\ell) \cap (s, e) = \emptyset$，测试模型对"应发生什么事件"的判断。
  - **Alignment（A）**：在目标区间附近扰动边界，包括 Advanced（前移 $\ell/2$）、Deferred（后移 $\ell/2$）、Expanded（向两侧各扩展 $\ell/2$），测试"事件应何时发生"。
  - **Progression（P）**：对目标区间进行 Reversed（反转播放）、Reordered（子事件重排）、Repeated（重复短片段）变换，测试"事件应如何展开"。
- **数据构建流程**：从七个公开源筛选视频，先按时长（12-90秒）与 LLM 判断（使用 GPT-o3 评估时序连贯性），再经质量过滤（码率 > 200kbps、锐度 > 30），最后用 Farnebäck 光流验证运动量（平均幅度 > 1.0），最终保留 1,521 个视频。
- **帧采样策略**：采用 Bin-Centered Sampling 在每个 clip 内等分 bin 并取中心点采样，避免边界帧泄露；默认每 clip 采样 16 帧，每个维度共 96 帧。

## 实验与结果
- **数据集**：TempCloze 共 1,521 个视频，来源包括 LVD-2M（515）、EgoLife（437）、MiraData（198）、FAVOR（145）、CaReBench（94）、Video-TT（89）、Daily-Omni（43）；另构造 TempCloze-Mixed（300 个随机混合干扰）和 TempCloze-Hard（150 个最难样本）。
- **评估基线**：10 个私有模型（Seed1.8、Qwen3.5-Plus、Gemini2.5-Pro/Flash、GPT5.4、Claude4.6-Sonnet/Opus、Gemini3-Flash、Seed1.6、Grok4.1）和 21 个开源模型；人类基线在 100 个随机样本上测得 S/A/P 分别为 96%/98%/97%。
- **主要结果**：
  - **Alignment 是最大瓶颈**：私有模型平均 Alignment 准确率 48.13%（最高 Seed1.8 为 76.92%），开源模型仅 26.54%；语义理解方面私有平均 70.73%，进展方面 67.72%。
  - **最强私有模型** Seed1.8（thinking）达到 S=96.25%、A=76.92%、P=92.11%，均值 83.43%，3/3 完整正确率 56.94%。
  - **最强开源模型** Qwen3.5-397B-A17B 达到 S=75.94%、A=50.62%、P=78.24%，均值 68.27%，3/3 正确率 35.77%。
  - 私有 vs 开源在 3/3 正确率上差距达 4.6 倍（33.24% vs 7.15%）。
- **误差模式**：Alignment 错误以 Expanded（扩展）为主（如 Seed1.8 的 75% Alignment 错误为 Expanded）；Progression 错误以 Reversed（反转）为主（GPT-5.4 的 67% Progression 错误为 Reversed）。
- **行为敏感性发现**：模型更依赖开头上下文而非结尾；增加帧密度反而降低 Alignment 表现；测试时扩展（Pass@k）能提升绝对分数但不改变维度间排序。

## 相关工作脉络
- **Temporal-Bench、TVBench、TempCompass**：同为时序推理基准，但均依赖自然语言问答，存在语言捷径风险；TempCloze 以纯视觉候选替代文本选项，实现更严格的时序评测。
- **LongVideoBench、MVBench、VideoMME**：通用长视频理解基准，涵盖多样任务但时序推理非核心焦点；TempCloze 聚焦于缺失中间段的精确时序对齐。
- **MovieFIB、FIBER**：文本完形填空基准，评估视频理解但通过文本补全而非视觉推理；TempCloze 将完形填空范式迁移至纯视觉领域。
- **VideoBERT、VCP、MaskFeat、VideoMAE**：将 Cloze 作为预训练目标用于视频表征学习；TempCloze 首次将视频完形填空系统设计为 Video-LLM 的评测基准。
- **NExT-QA、ActivityNet-QA、TGIF-QA**：早期 VideoQA 数据集，关注显式动作与时空关系问答；TempCloze 在形式上更严格，消除了文本选项带来的偏差。
- **CaReBench、Daily-Omni、EgoLife、FAVOR、LVD-2M、MiraData、Video-TT**：TempCloze 使用的七个数据来源，涵盖了长镜头、第一人称视角和细粒度动作视频，为时序连续性提供了丰富的语料基础。

## 局限性与未来方向
- TempCloze 是受控的诊断性基准，未覆盖开放生成、对话式叙事理解、音频 grounding 推理和自由形式未来预测等场景。
- 固定候选格式意味着性能应相对于设计的干扰项解读，不能作为通用自由推理能力的度量。
- 数据来源侧重长镜头、第一人称和细粒度动作视频，不能覆盖所有领域和摄像机风格。
- 模型结果为当前快照，API 和解码行为会随时间变化。
- 未来方向：需要视频原生机制来整合双向时序信息、选择关键证据并显式表征时序结构；基准可扩展至更长视频并控制语言捷径。

## 研究启发与可借鉴点
- **三维度干扰项设计思路可迁移**：Semantic/Alignment/Progression 的正交分解为评估其他模态推理能力（如音频时序、多模态因果推理）提供了可复用的维度拆解框架。
- **同源候选消除外观捷径的实验设计**值得借鉴：在需要排除视觉混淆因素的评测场景中，可以从同一源视频生成干扰项来强制模型依赖目标维度信号。
- **Bin-Centered Sampling 避免边界帧泄露**的技术细节对任何涉及视频片段匹配的评测均有参考价值。
- **Error Pattern + Behavioral Sensitivity 的组合分析方法**可同时定位模型"错在哪里"和"为何不稳定"，对模型诊断有通用方法论价值。
- **发现 Alignment 为最大瓶颈**提示未来 Video-LLM 训练可能需要加强时间边界敏感度，可探索视频原生时序编码或直接引入时序对齐损失。

## 关键术语表
- **TempCloze**：一个视频完形填空基准，要求模型从四个候选片段中识别出正确的缺失中间段，以评估 Video-LLM 的视觉时序推理能力。
- **Alignment（时间对齐）**：评估模型判断事件应"何时"发生的维度，干扰项通过平移或扩展目标区间构造。
- **Semantic（语义）**：评估模型判断事件应"是什么"的维度，干扰项为不同时间段的同持续时长片段。
- **Progression（事件进程）**：评估模型判断事件应"如何展开"的维度，干扰项包括反转、重排和重复子片段。
- **Bin-Centered Sampling**：将 clip 等分为若干 bin 并采样每个 bin 中心帧的帧采样策略，避免边界帧泄露。
- **Farnebäck Optical Flow**：用于验证视频运动量的光流算法，用于过滤近乎静态的视频片段。
- **Expanded distractor**：Alignment 干扰项的一种，将候选区间向两侧扩展，使选中片段包含目标区间之外的相邻内容。
- **Test-time Scaling（Pass@k）**：通过多次采样取最优结果的测试时扩展策略，用于估计模型性能上界。

## 可复现要素
- **数据集**：TempCloze 由七个公开数据集构建，数据本身基于公开来源，但经过去重和筛选后的最终数据集论文未明确声明是否公开（需查阅项目页面确认）。
- **代码/权重**：论文未明确声明代码和权重是否开源。
- **关键超参**：视频时长 12-90 秒；缺失区间占全视频 20%-40%，位于中间 50%；每 clip 采样 16 帧；光流阈值平均幅度 > 1.0；质量阈值码率 > 200kbps、锐度 > 30。
- **硬件**：开源模型部署于 A6000 GPU，使用 vLLM 推理。
- **提示词**：论文附录 G 提供了 LLM 筛选提示词和评估提示词的完整内容。
