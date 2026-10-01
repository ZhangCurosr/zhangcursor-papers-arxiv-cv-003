---
title: "MG-Thinker-Bi-Axial-Self-Reflection-for-Multi-Image-Reasonin"
source: https://arxiv.org/pdf/2609.37374v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 21:43:53"
field: "多模态推理与视觉定位"
keywords: ["多图像推理定位", "多模态大语言模型", "强化学习", "Chain-of-Thought", "组相对优势", "奖励分解"]
innovations: ["提出沿信号轴与能力轴双轴解耦的 BiA-DAPO 组相对 RL 更新机制", "构建含层级 CoT 的 25K MRG 冷启动数据与伪推理过滤+IoU 增益后处理", "将 AND 聚合的粗到细奖励（format→image-id→IoU）与双轴训练结合在统一框架"]
benchmarks: ["MIG-Bench", "LISA-Grounding", "LLMSeg-Grounding", "ReVOS-Grounding", "ReasonVOS-Grounding", "MuirBench", "BLINK", "MMIU", "MIBench", "RefCOCO/+/g"]
---

# 论文速读：MG-Thinker: Bi-Axial Self-Reflection for Multi-Image Reasoning Grounding

## 一句话总结

本文提出 MG-Thinker，一个面向多图像推理定位（Multi-image Reasoning Grounding, MRG）的后训练强化学习框架，通过构建具有粗到细层级 Chain-of-Thought 标注的 25K 数据，以及提出的 BiA-DAPO 算法（沿信号轴和能力轴双轴分解 rollout 优势），解决了先方法忽视的 MRG 特有层级推理模式和任务–样本难度异质性问题，在 MIG-Bench 上达到 SOTA（76.71%），且同时泛化至多图像理解与通用多模态基准。

## 研究问题与动机

- **多图像推理定位（MRG）是新范式的必要补充**：现有 MLLM 的多图像能力多停留在自由格式理解层面，缺乏面向像素级精确定位、需要跨图像推理支撑的 fine-grained grounding 能力；直接移植单图 CoT 会产生不忠实、误导注意力的推理链，放大定位误差。
- **先验 RL 方法忽视 MRG 的两个本质特征**：① MRG 天然要求"语义→图ID→边界框"的粗到细层级推理结构，通用任务的模板化推理或直接预测不足以满足；② MRG 存在显著的任务–样本难度异质性——基座模型在 MRG 上的均值奖励远低于通用任务，而 CoT-SFT 模型在 MRG 子任务间波动剧烈，通用任务却相对稳定。
- **直接端到端定位存在漂移问题**：现有 MLLM 依赖 end-to-end 直接预测，面临 grounding drift、输出格式不稳定、跨图像对齐弱等问题，尤其在"一次推理同时定位多个跨图目标"的场景下显著不足。

## 核心贡献（创新点）

1. **提出 MRG 新范式并配套层级 CoT 数据管线**：将 MGrounding-630K 重构蒸馏为 25K MRG 专属 CoT 样本，以 task-adaptive cue prompts 注入多视角证据搜寻，迫使模型在结论前先跨图检索证据；与已有工作区别在于，这是首个显式把"CoT→image-id→bbox"层级结构嵌入奖励设计与数据构造的 MRG 方案。
2. **BiA-DAPO：沿双轴解耦的组相对 RL 更新机制**：提出 Group-Informativeness Assessment（GIA，沿组内方差轴剔除优势信号稀疏组）与 Cascaded-Reward Stratification（CRS，沿组间均值轴按能力成熟度排序更新），两者共享候选池以稳定统计；与 UniVG-R1（仅用 mIoU 加权）和 AdaRFT（仅课程调度）的单轴方案相比，BiA-DAPO 同时在信号轴和能力轴起作用，避免单一统计量混淆两种故障模式。
3. **分层阶段式训练与双阈值 IoU 奖励**：Stage-1 SFT（480K 重构样本，激活 grounding）→ Stage-2 CoT-SFT（25K 冷启动，耦合推理与定位）→ RL 后训练（BiA-DAPO），并以 format + image-id + IoU（双阈值 τh=0.9、τl=0.1）的组合奖励防止 reward hacking；与 R1 类方法相比，其粗到细奖励是 MRG 特有的 AND 聚合结构。
4. **系统验证跨任务泛化与零样本迁移**：不仅在 MIG-Bench 10 个子任务上达到 SOTA，还证明在 LISA/LLMSeg-Grounding、ReVOS/ReasonVOS-Grounding、MuirBench、BLINK、MMIU 等基准上均有持续提升，且单图 RefCOCO/+/g 上不会退化。

## 方法详解

**MRG 形式化。** 给定图像集 $V = \{I_1,...,I_m\}$、提示 $P$、可选参考 $R$，模型输出推理链 $S$ 与定位结果 $\{(b_k, i_k)\}_{k=1}^n$：$O = \{S, \{(b_k, i_k)\}\} = \mathcal{M}_{sg/rg}(V, P)$；自发定位（SG）无需参考，参照定位（RG）需利用文本/视觉参考，共 10 个子任务（Static、Robust、Common、OT、MV、Region、Refer、GG、Reason、Co-Re）。

**数据管线（Fig. 3）。** ① 基于规则的 MGrounding-630K 筛选：去除问题类型同质、分辨率过载、冗余对话样本，重写特殊 token 与 JSON 结构；② Qwen2.5-VL-72B 再生无法自动改写的标签并引入 image-id 键值对；③ 剩余 160K 样本按 4 类 MRG 任务（视觉比较、空间感知、时间感知、视觉语义/逻辑关联）分配 cue prompt，注入 comparison/searching/observation/tracking/association 算子驱动分步证据搜寻，且 <think> 块禁止透露答案或 gt bbox；④ IoU-improvement 后处理保留满足以下之一者：直接预测正确且 CoT 后 IoU 增益 ≥10%、直接预测错误但被 CoT 纠正、直接预测错误且 CoT 后 IoU 增益 ≥20%；最终得到 25K 冷启动样本。

**BiA-DAPO（Sec. 3.3, Fig. 4）。** 基于 DAPO 的组相对优势 $\hat{A}_i = (acc_i - \bar{r}_g)/(\sigma_g + \varepsilon)$，分子编码组间能力轴（Axis-2，$\bar{r}_g$），分母编码组内信号轴（Axis-1，$\sigma_g$），两统计量正交：
- **GIA**：按组内奖励方差 $\sigma_g^2$ 筛选"高信息"组，阈值 $\alpha$ 随训练阶段线性衰减（4 阶段：$\alpha_0=0.05$，每阶段减半），早期只保留高方差组、后期逐渐放宽，缓解优势信号稀疏漂移。
- **CRS**：在 GIA 保留组上，按组均值 $\bar{r}_g$ 排序为"能力成熟→能力发展中"的阶层，策略逐层更新，使梯度步落在与当前能力匹配的难度带上，缓解能力–难度失配。
- **共享候选池**：复用 DAPO 的 generation batch 作为候选池（训练 batch 的 3 倍，pool=48），保证组统计稳定，无需额外扩大生成预算。

**奖励建模（Sec. 3.4）。** $R_{total} = R_{format} + R_{acc}$，$R_{acc} = R_{id} + R_{IoU}$；$R_{format}$ 校验 `<｜think＞</｜think>` 与 `<｜answer＞</｜answer>` 配对（正确得 0.25）；$R_{id}$ 用匈牙利匹配验证 image-id（多图像）或直接匹配（单目标），仅 id 正确才继续 IoU 评估；$R_{IoU}$ 采用双阈值截断：$R_{IoU}=1$（IoU≥0.9）、$R_{IoU}=\text{IoU}$（0.1≤IoU<0.9）、$R_{IoU}=0$（IoU<0.1），防止退化解 hack。

**DAPO 目标。** 非对称 clip $\varepsilon_l=0.2$、$\varepsilon_h=0.28$，token-mean 聚合，KL 以 low_var_kl 形式作显式损失（coef=0.01），而非 reward shaping；至少 1 个且不全等于 gt 的多样性约束维持正负例学习。

## 实验与结果

**数据集与评估。** 主基准 MIG-Bench 10 个子任务（Acc@0.5），零样本迁移到 LISA-val/test、LLMSeg-Grounding、ReVOS、ReasonVOS-Grounding；多图像理解基准 MuirBench、BLINK、MMIU、MIBench；单图 RefCOCO/+/g。

**主要结果。** MG-Thinker（7B）在 MIG-Bench 平均 **76.71%**，超越最强 baseline UniVG-R1†（74.22%，+2.49）与 Migician（63.54%，+13.17）；相比 Qwen2.5-VL-72B（46.23%）提升 **+30.48 点**。单图 RefCOCO/+/g 上仍保持 88.18% 的竞争力，验证不过度 specialization。

**对比消融（Tab. 6）。** Stage-1 SFT vs. 基座平均提升约 2×；CoT-SFT 进一步提升跨图对应/推理密集型子任务；移除 GIA（w/o GIA）使 OT、Co-Re 等组内方差敏感任务下降（GIA 缺失导致 σg≈0 组混入）；移除 CRS（w/o CRS）使 Robust、MV、GG 等组间均值跨度大的任务下降；二者联合达到最优。

**零样本迁移（Tab. 3）。** 在 LISA/LLMSeg/ReVOS/ReasonVOS-Grounding 上平均 59.02%，略逊于用单图推理数据做 RL 的 UniVG-R1，但强调"纯跨图推理先验可自然迁移到时序场景"。

**多图像理解（Tab. 5）。** 在 MuirBench、BLINK、MIBench、MMIU 上平均 60.43%，超越 UniVG-R1（54.11%）与 Migician（58.85%），说明 grounding-anchored CoT 强化跨图像注意力，并非任务过拟合。

**任务异质性分析（Tab. 4）。** V-Het（能力轴主导）上 BiA-DAPO 领先 AdaRFT† 4.41 点；R-Het（信号轴主导）上领先 UniVG-R1† 4.29 点；L-Het 上两者接近，说明双轴缺一不可。

**训练动态（Fig. 5/6）。** GRPO/DAPO 因 sparsity 与 misalignment 耦合导致 PFA 非单调震荡；BiA-DAPO 呈平滑上升，且 σg 分布集中在中/高信息区间，而非 GRPO 滞留 σg=0 的退化组。

## 相关工作脉络

1. **视觉定位（Visual Grounding）**：从 RefCOCO 系的单图指代表达定位，到 LISA-Grounding 的推理密集型指令定位，再到 Mantis/LLaVA-OV-Interleave 的多图理解，本文聚焦的是从自由格式多图 grounding 到像素级定位的"最后一公里"空白。
2. **R1 类 RL 推理**：DeepSeek-R1、VLM-R1、Ground-R1 等主要验证了规则奖励 + GRPO/DAPO 在单图视觉推理上的有效性；本文指出其直接移植到多图会因"模板化推理"与"单图先验"引发 reasoning drift，需引入层级 CoT 与双轴奖励分解。
3. **VL-Rethinker / AdaRFT**：VL-Rethinker 用高价值样本 replay 缓解 advantage vanishing；AdaRFT 做离线课程调度；两者均只处理单一维度，而 BiA-DAPO 沿两条正交轴同时干预。
4. **Migician / UniVG-R1**：Migician 提出首个多图 grounding 端到端范式；UniVG-R1 首次尝试 GRPO + grounding 专用 loss；本文 fair re-implementation（UniVG-R1†、AdaRFT†）显示它们在单轴上已有上限，双轴才是完整方案。
5. **多图像理解基准**：BLINK、MuirBench、MMIU、MIBench 衡量"看见但非感知"的能力；本文将 grounding 锚定引入理解任务，提升跨图注意力质量而非仅优化 grounding 精度。

## 局限性与未来方向

- **CoT 生成依赖 72B 标注器**：25K 样本用 Qwen2.5-VL-72B 生成，若标注器本身存在系统性偏差（如图像理解盲区），会污染冷启动 prior。
- **奖励为确定性规则且 AND 聚合**：一旦图像 ID 匹配失败即跳过 IoU，导致梯度稀疏且不可微；对部分模糊图像难以给出部分 credit。
- **仅在 MRG 任务上验证迁移性**：虽然零样本在多模态基准有增益，但未系统探索 BiA-DAPO 在纯文本推理（如数学）、视频等多时序任务上的可迁移上限。
- **候选池固定 3× 未做规模化扩展**：对于更大数据量或更大 batch 的场景，候选池稳定性可能需要进一步研究。
- **未讨论长 CoT 的推理漂移控制**：尽管有 pseudo-reasoning 过滤与 IoU-improvement 校验，但对于极端复杂推理链中的"中途跑偏"现象，缺少机制性保障。

## 研究启发与可借鉴点

1. **"双轴解耦"可作为 RL 后训练的通用设计模式**：将组内方差（信息量轴）与组间均值（能力轴）视为正交统计量，分别做 gating 与分层调度，可复用到任何"任务难度异质性强 + 易出现优势信号稀疏"的 RL 场景（如视频推理、长 CoT 数学）。
2. **CoT 数据必须配合 reward 设计**：本文的"伪推理过滤 + IoU 增益验证"两步后处理值得复用，确保冷启动 prior 既忠实又有效，避免 RL 后期因 shortcut 而退化。
3. **AND 聚合奖励的层叠饱和现象可作为诊断工具**：从图 6 的 σg 分布迁移轨迹可见，层叠饱和是逐步发生的，可用于监控各层能力的训练动力学。
4. **单图先验保护机制**：BiA-DAPO 的组相对更新不依赖 modal-specific loss，因此 multi-image 训练不会压制 single-image 能力；这一"先验保持"原则在跨模态扩展时应继续遵循。
5. **四阶段 GIA 衰减阈值的启示**：从 0.05 逐阶段减半的经验设置，可推广为"初期严格、后期放宽"的通用 curriculum 设计范式，适配不同任务的信号稀疏曲线。

## 关键术语表

**Multi-image Reasoning Grounding (MRG)**：面向多图像集的精细定位任务，要求模型在跨图像证据基础上输出带 image-id 标记的像素级 bbox。
**Bi-Axial DAPO (BiA-DAPO)**：沿组内方差轴（信号轴）和组间均值轴（能力轴）双轴分解 DAPO 优势的 RL 更新机制。
**Group-Informativeness Assessment (GIA)**：基于组内奖励方差 $\sigma_g^2$ 筛选高信息 rollout 组的 gating 机制，缓解优势信号稀疏漂移。
**Cascaded-Reward Stratification (CRS)**：按组均值奖励 $\bar{r}_g$ 排序分组并逐级更新，使策略难度匹配当前能力，缓解能力–难度失配。
**Post-Format Accuracy (PFA)**：格式通过后 $R_{acc}=R_{id}+R_{IoU}$ 的奖励分量，是 RL 阶段的主导信号。
**CoT-SFT**：在 Stage-1 grounding 激活之后、RL 之前的第二级监督微调，使用 25K 带层级 CoT 的冷启动样本。
**Spontaneous vs. Referential Grounding**：前者无需参考（纯跨图推理找目标），后者需结合文本/视觉参考（考察跨图对应与定位耦合）。
**Advantage-Signal Sparsity**：rollout 组内奖励方差接近零导致组相对优势 $\hat{A}_i$ 退化，梯度信息缺失的训练病态。

## 可复现要素

- **数据集**：MIG-Bench（MGrounding-630K 重构），320K Stage-1 SFT + 25K CoT-SFT + 7K RL 预留；论文未公开代码/权重，仅说明基于 VeRL 框架实现。
- **代码/权重**：论文未开源代码与最终 checkpoint；训练基于 Qwen2.5-VL-7B，硬件 8×A100-80GB。
- **关键超参**：SFT lr=3e-6、batch=48；CoT-SFT/RL lr=1e-6；RL batch=16、responses/prompt=16、max len=1024、temp=1.0；GIA 初始阈值 α0=0.05、4 阶段减半；clip εl=0.2、εh=0.28；KL coef=0.01；候选池 size=48（3× batch）；IoU 阈值 τh=0.9、τl=0.1。
