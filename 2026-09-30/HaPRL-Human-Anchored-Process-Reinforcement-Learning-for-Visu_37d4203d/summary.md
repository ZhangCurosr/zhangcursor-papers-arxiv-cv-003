---
title: "HaPRL-Human-Anchored-Process-Reinforcement-Learning-for-Visu"
source: https://arxiv.org/pdf/2609.37190v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 21:42:41"
field: "视觉搜索强化学习"
keywords: ["visual search", "reinforcement learning", "process supervision", "human traces", "reward design", "multimodal reasoning", "vision-language model"]
innovations: ["提出人类锚定过程强化学习框架，首次将人类搜索行为作为奖励锚点", "设计任务自适应的过程评价器与答案门控机制", "通过早期过程监督显著提升后续结果扩展增益（6.7倍）"]
benchmarks: ["VisualProbe Easy/Medium/Hard", "V*QA", "HR-Bench 4K/8K"]
---

# 论文速读：HaPRL: Human-Anchored Process Reinforcement Learning for Visual Search Agent

## 一句话总结
HaPRL 首次将人类搜索行为作为奖励锚点引入多轮视觉搜索代理的强化学习训练，通过任务自适应的过程评分替代仅依赖最终答案的结果奖励，有效纠正错误搜索路径，显著提升搜索质量与准确率。

## 研究问题与动机
1. **结果奖励的缺陷**：现有视觉搜索代理的强化学习仅以最终答案正确性作为奖励，导致错误搜索路径（如随机猜测）也能获得全量奖励，使模型沿错误路径优化。
2. **过程监督缺失**：基于结果的 GRPO 在同一组采样轨迹中标准化奖励，无法区分搜索质量优劣，例如高效定位证据的轨迹与无效搜索轨迹获得相同奖励。
3. **人类行为信号未利用**：人类在搜索过程中产生的细粒度行为信号（如移动轨迹、停留时间、缩放操作）蕴含丰富搜索质量信息，但被现有方法丢弃。
4. **核心挑战**：需解决两个问题：(G1) 系统获取人类搜索过程中的细粒度行为信号；(G2) 将代理训练与这些人类信号对齐以实现密集过程监督。

## 核心贡献（创新点）
1. **构建人类搜索行为数据集**：开发浏览器标注平台，收集 1,104 个视觉搜索问题的细粒度行为信号，并蒸馏为按轮次的注意力与证据获取轨迹。
2. **提出 HaPRL 训练范式**：首个将视觉搜索代理与人类过程信号对齐的框架，通过答案门控、任务自适应的过程评分，以人类轨迹为锚点排序正确轨迹。
3. **显著提升性能与搜索质量**：在四个骨干网络和六个基准上，HaPRL 一致优于基于结果的强化学习；早期过程监督为后续结果扩展带来 **6.7×** 更大增益（+6.13 vs +0.91）。
4. **揭示过程对齐的重要性**：实验证明将代理行为与人类过程标注信号对齐能有效纠正错误搜索，为超越视觉搜索的基础模型训练提供新见解。

## 方法详解
方法分为数据收集、过程评价器和策略优化三部分。

1. **人类过程数据采集**：
   - 使用浏览器平台记录 1,104 个 VisualProbe 训练问题的交互流，事件包括指针位置、速度、加速度、缩放框等。
   - 离线蒸馏为紧凑的按轮次文本轨迹 $H^* = (r_1^*, \ldots, r_L^*)$，每轮记录检查区域、源视图、归一化裁剪及揭示的证据。
   - 训练样本为 $\bar{z} = (I_0, q, a^*, E^*, H^*)$，策略仅观察 $(I_0, q)$，轨迹用于奖励计算。

2. **人类锚定过程评价器**：
   - 冻结的视觉语言模型 $J_\phi$ 接收 $(I_0, q, \tau, H^*)$，输出五个维度的分数向量 $\mathbf{s}(\tau) \in [0,1]^5$：目标语义、证据获取、搜索进展、工具纪律、通信纪律。
   - 根据问题类型分配任务自适应权重 $\mathbf{w}^{(p)}$（如 OCR/text 侧重证据可读性，count/relation 侧重空间覆盖）。
   - 过程得分 $r_{\text{proc}}(\tau) = \langle \mathbf{w}^{(p)}, \mathbf{s}(\tau) \rangle$。

3. **答案门控过程优化**：
   - 冻结的答案评价器 $J_{\text{ans}}$ 得到二元正确性 $r_{\text{out}} \in \{0,1\}$。
   - 训练奖励 $R_{\text{HaPRL}}(\tau) = r_{\text{out}}(\tau) \cdot r_{\text{proc}}(\tau)$，错误答案得零分，正确轨迹按搜索质量排名。
   - 组内标准化优势：$A_i = \frac{R_i - \bar{R}}{\sqrt{G^{-1}\sum (R_j - \bar{R})^2 + \epsilon}}$。
   - 使用裁剪 GRPO 目标更新策略：$\mathscr{L}(\theta) = -\mathbb{E}[\min(\rho A_i, \text{clip}(\rho,1-\varepsilon,1+\varepsilon) A_i)] + \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$。

## 实验与结果
- **数据集与基准**：VisualProbe Easy/Medium/Hard、V*QA、HR-Bench 4K/8K；骨干网络：Qwen3-VL-4B/8B-Instruct、LLaVA-OneVision-1.5-4B/8B。
- **评估基线**：Base model、SFT cold start、Outcome-RL。
- **主要结果**：
  - HaPRL 在所有 24 个骨干-基准组合中优于 Outcome-RL；Qwen3-VL-4B 宏观平均准确率提升 **30.3%**（44.26 vs 33.98）。
  - 修复 SFT 导致的准确率下降，所有四个骨干均超越 base model。
  - **搜索质量**：HaPRL 在答案正确 rollout 上平均提升 **+2.78** 分（84.77 vs 81.99），而 Outcome-RL 仅 +0.08；在 VisualProbe Hard 上，Outcome-RL 准确率提升 28.6% 但过程得分下降 8.4%，HaPRL 恢复 14.8 分。
- **缩放行为**：过程标注数据从 200 增至 1,104，准确率单调上升（+20.9%）。早期过程监督（400 条）后接结果阶段（600 条）比全程结果监督提升 **6.7×**（+6.13 vs +0.91）。
- **消融实验**：打乱人类轨迹配对导致性能降至 28.63；去掉人类轨迹仅用评价器得 36.41；添加 10% 噪声后仍保持 41.74。

## 相关工作脉络
1. **视觉搜索代理**（V*, Visual Sketchpad, DeepEyes, Pixel Reasoner, Chain-of-Focus, Mini-o3）：依赖终端正确性或手动设计的辅助信号，HaPRL 使用人类搜索轨迹作为实例特定参考。
2. **过程监督**（PRM800K, Math-Shepherd, OmegaPRM, VisualPRM）：评估数学推理中间步骤，但视觉搜索的感知行动相互依赖，需整体轨迹评估。
3. **评分规则强化学习**（Rubrics as Rewards, RL from Checklist Feedback, Rubric Anchors）：主要用于最终响应评估，HaPRL 将任务自适应评分规则应用于搜索过程，并以人类轨迹为锚点。

## 局限性与未来方向
- **局限性**：人类轨迹采集成本较高（约 27.6 人时），可能限制大规模应用。
- **未来方向**：将人类过程对齐方法扩展到其他领域，如 3D rigging、GUI agents 等可借鉴人类标注信号的领域。

## 研究启发与可借鉴点
1. **过程监督的实例化锚定**：使用与问题匹配的人类行为轨迹作为评价参考，而非通用规则，可提高奖励精度，适用于其他序列决策任务。
2. **答案门控机制**：将过程评分与最终正确性相乘，既保留结果目标又引入过程偏好，可推广至其他强化学习场景。
3. **任务自适应权重**：根据不同任务类型动态调整过程维度重要性，提高评价器适应性，可借鉴于多模态推理中的奖励设计。
4. **冷启动修复价值**：过程监督能显著弥补 SFT 导致的性能损失，为多阶段训练提供有益初始化，可探索过程数据在少样本学习中的作用。

## 关键术语表
- **HaPRL**：Human-Anchored Process Reinforcement Learning，本文提出的人类锚定过程强化学习框架。
- **VisualProbe**：包含不同难度级别的视觉搜索基准数据集，用于评估代理的多轮搜索能力。
- **GRPO**：Group Relative Policy Optimization，组相对策略优化，通过组内标准化优势更新策略的强化学习算法。
- **Process Judge**：过程评价器，冻结的视觉语言模型，依据五个维度对人类轨迹锚定的 rollout 进行评分。
- **Task-adaptive Weights**：任务自适应权重，根据问题类型动态分配的评分维度权重。
- **Answer-gated Reward**：答案门控奖励，将过程得分与二元正确性相乘的奖励机制。
- **Cold Start**：冷启动，代理在强化学习前通过 SFT 获得基本工具使用能力的训练阶段。
- **Trace Distillation**：轨迹蒸馏，将人类交互事件流压缩为按轮次的文本轨迹。

## 可复现要素
- **数据集**：VisualProbe 训练集（1,104 个标注问题）公开，人工标注轨迹已开源。
- **代码/权重**：代码和权重已开源，github.com/zhangquanchen/HAPRL。
- **关键超参**：RL 使用 GRPO，8 条 rollout 每 prompt，prompt batch size 48，actor learning rate 5e-7，KL 系数 3e-3；评价器为冻结的 Qwen3-VL-30B-A3B-Instruct；训练 3 个 epoch。
