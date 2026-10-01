---
title: "Motion-Concept-Unlearning-in-Video-Difusion-Models"
source: https://arxiv.org/pdf/2609.36832v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 21:44:53"
field: "视频生成模型安全与可控性"
keywords: ["motion concept erasure", "text-to-video diffusion", "classifier-free guidance", "training-free unlearning", "concept direction"]
innovations: ["首次系统性探究视频DiT中运动概念的神经编码位置，证明cross-attention适合选择性干预而temporal positional encoding全局支持动态", "提出MUTE：基于token neutralization和自派生spatial gate的training-free输出级逐步骤擦除方法，在CFG前移除概念贡献", "揭示CFG残差缩放机制是权重级擦除效果受限的根本原因，推导出三要求框架指导方法设计"]
benchmarks: ["X-CLIP MCS", "LPIPS", "SSIM", "Wan2.1-T2V-1.3B", "CogVideoX-2B"]
---

# 论文速读：Motion-Concept-Unlearning-in-Video-Difusion-Models

## 一句话总结
本文首次系统性研究视频扩散模型中**运动概念的定向擦除**问题，提出训练-free的方法MUTE（Motion Concept Unlearning in Text-to-video gEneration），通过在每个去噪步骤中估计并减去目标运动贡献，实现仅抑制目标动作而保留场景外观与非目标动态的效果，在20个运动概念上显著优于现有基线。

## 研究问题与动机
- **安全问题驱动**：T2V模型能生成踢、刺、射击等危险动作的逼真视频，仅靠后生成安全过滤器易被对抗性提示绕过，需从根本上修改生成过程以抑制目标概念。
- **运动概念擦除不同于静态概念**：现有概念擦除工作主要针对静态物体、风格、身份，或视频中的静态内容；而运动概念在时间维度展开，视频DiT引入的temporal positional encoding等新组件使问题更复杂。
- **现有方法不足**：直接将ESD等图像权重级擦除方法适配到视频cross-attention，抑制效果有限且不均匀（mean ΔMCS仅+0.23），且残差信号经CFG放大后仍会残留运动信息。
- **编码位置不明确**：视频DiT中运动信息是集中在text-conditioning attention中，还是分布到temporal positional encoding等时间组件中，尚未有系统探究。

## 核心贡献（创新点）
1. **首次将运动概念擦除定义为独立问题并提出系统性因果干预分析**：通过实验证明cross-attention支持概念选择性干预，而temporal positional encoding抑制会导致全局动态崩溃，从而确定text-conditioning attention为理想擦除通道。
2. **揭示权重级擦除失败的本质原因**：指出CFG对conditional-unconditional差值的缩放机制使得单一权重编辑难以彻底消除残差运动信号，由此引出需要输出级逐步骤干预的思路。
3. **提出MUTE训练-free输出级擦除框架**：基于三大要求（concept specificity、spatial selectivity、temporal naturalness）推导出token neutralization提取概念方向、self-derived spatial gate实现空间聚焦、per-step velocity subtraction在CFG前执行三个组件，仅需单个超参α=5.0。
4. **跨架构验证泛化性**：在Wan2.1-T2V（separate cross-attention）和CogVideoX（joint attention）两个不同架构上均取得强抑制效果，证明方法不依赖特定注意力设计。

## 方法详解
**核心思路**：在每个去噪步骤t，估计目标运动概念C对velocity输出的贡献d_t，并通过空间门控M_t限制干预范围，最终在CFG应用前从conditional velocity中减去校正项。

**关键组件**：
1. **概念方向估计（Token Neutralization）**：构建修改后的条件ĉ，将目标运动动词子词（如"kicks"）的上下文嵌入替换为整个固定长度上下文的均值，其余位置不变，计算 $d_t = v_θ(x_t, t, c) - v_θ(x_t, t, \tilde{c})$，作为该步骤目标概念的代理贡献。
2. **自派生空间门控**：利用d_t的空间集中性（peak-to-mean比>20:1），计算每位置的channel-norm $I_t(i,j,k) = \|d_t(:,i,j,k)\|_2$，归一化得 $M_t = I_t / \max(I_t)$，在运动区域≈1，背景≈0，无需外部分割模型。
3. **逐步骤校正与CFG融合**：修正后的velocity为 $v_{erased} = v_θ(x_t, t, c) - α·M_t ⊙ d_t$，再代入标准CFG公式 $v_{final} = v_∅ + s·(v_{erased} - v_∅)$。整体可简化为 $v_{final} = v_{cfg} - s·α·M_t ⊙ d_t$，本质上是concept-specific negative guidance。

**超参**：单一强度超参α=5.0，每步需3次forward pass（conditional、neutralized、unconditional），约47%时间开销。

## 实验与结果
**数据集与模型**：Wan2.1-T2V-1.3B（832×480，81帧）和CogVideoX-2B（720×480，49帧）；20个运动概念（kick、punch、slap、push、stab等），每概念最多3个prompt。

**评估指标**：X-CLIP MCS（运动一致性分数，ΔMCS越大抑制越强）、LPIPS、SSIM；人工评估。

**主要结果（Wan2.1-T2V）**：
- MUTE mean ΔMCS = **+1.24**，显著优于neg prompt（+0.24）、UCE（+0.09）、ESD（+0.33）、T2VUnlearning（+0.50）、VideoEraser（+1.09）。
- 在局部化动作上优势最明显：punch（+3.81 vs +2.43）、slam（+1.75 vs +0.55）、slap（+1.87 vs +1.05）。
- CogVideoX上mean MCS从0.90降至0.61（joint attention下抑制略弱但方向一致）。

**消融**：α=5.0为默认；去掉spatial gate在定量指标上变化不大但定性上引入背景伪影；random direction取代d_t导致抑制下降35%。

**人工评估**：MUTE在motion suppression上获57.9%投票（vs VE 12.1%），visual quality上获82.4%投票（vs VE 7.6%）。

## 相关工作脉络
1. **ESD/Gandikota et al.**：图像领域代表性weight-level概念擦除方法，通过fine-tune cross-attention Q/K/V投影消除概念；本文证明其直接迁移到视频motion概念效果不佳，因CFG残差缩放机制未被考虑。
2. **UCE/Gandikota et al.**：closed-form unified concept editing；同样针对静态概念设计，未处理运动的时间维度特性。
3. **VideoEraser/Xu et al.**：首个专为此视频推理时概念擦除方法，但目标为静态object和NSFW content；本文指出其易产生phantom object伪影且会连带移除appearance细节。
4. **T2VUnlearning/Ye et al.**：fine-tuning-based video unlearning方法，仍主要针对静态概念；本文证明output-level逐步骤干预比单一权重编辑更有效。
5. **ConceptVoid/Huang et al.**：multi-concept erasure in video；聚焦多概念同时擦除而非运动概念的精确选择性抑制。
6. **Attention控制相关（Prompt-to-Prompt/DAMa等）**：关注文本-视觉对齐与信息定位；本文沿此思路但聚焦于"擦除"而非"控制生成"，且首次将token neutralization引入视频运动概念场景。

## 局限性与未来方向
- **跨architecture的抑制强度差异**：Joint attention架构（CogVideoX）下抑制效果弱于separate cross-attention（Wan2.1），因text和visual token共享同一attention层导致neutralization效应更分散，可能需要不同α值或更精细的token定位策略。
- **全身体动作保真度下降**：tackle、slam等大空间范围动作的LPIPS较高（0.82~0.84），说明空间门控难以完全避免大范围修正对整体画面的影响。
- **依赖X-CLIP MCS评估**：MCS基于预训练视频理解模型，可能与人类感知存在偏差；且仅衡量"运动是否消失"，未系统评估语义完整性和长期时序一致性。
- **单动作擦除为主**：Joint motion-object erasure初步验证可行，但多动作组合或复合运动的擦除策略尚未探索。
- **计算开销**：每步3次forward pass相比标准2次增加约47%延迟，对实时应用构成限制。

## 研究启发与可借鉴点
1. **Token Neutralization作为概念方向估计工具**：将目标token嵌入替换为均值是一种简洁有效的概念隔离策略，无需梯度更新即可估计概念对model输出的贡献，可迁移至图像编辑、风格控制等任务。
2. **自派生空间门控避免外部模块依赖**：利用概念方向本身的spatial concentration特征派生门控，比引入segmentation model更轻量且端到端兼容，此设计思路可用于其他空间选择性生成控制任务。
3. **CFG前的输出级干预优于权重级干预**：发现CFG会放大conditional-unconditional残差的scaling视角，启发了"在干扰被放大前直接移除"的策略，这对任何基于classifier-free guidance的生成模型均有参考价值。
4. **三大设计要求（specificity/selectivity/naturalness）可作为方法论框架**：将抽象约束映射到具体模块设计，形成可复用的方法推导范式，适用于未来其他类型的概念修改任务。
5. **跨架构可迁移性验证**：在Wan2.1和CogVideoX上用相同公式取得成功，提示训练-free方法可通过architecture-agnostic设计实现更高泛化性，值得在LLM等领域探索。

## 关键术语表
**Token Neutralization**：将目标token的上下文嵌入替换为整个上下文的均值，以估计该token对模型输出的概念贡献方向。

**MCS (Motion Consistency Score)**：基于X-CLIP的运动一致性度量，分数越高表示视频中目标运动存在程度越高，用于评估擦除效果。

**CFG (Classifier-Free Guidance)**：通过$v_{cfg} = v_∅ + s·(v_c - v_∅)$放大条件与无条件预测差异的技术，本文发现其对残差运动信号的缩放是权重级擦除效果受限的关键原因。

**DiT (Diffusion Transformer)**：采用纯Transformer架构的扩散模型，视频DiT引入3D RoPE等时间位置编码以建模视频时序关系。

**Cross-Attention vs Joint Attention**：Wan2.1使用独立的cross-attention层（visual tokens attend to text embeddings），CogVideoX使用joint attention（text和visual tokens拼接到同一序列）；两者在概念擦除中的敏感性不同。

**Spatial Gate / Influence Map**：基于概念方向d_t的空间norm自派生的门控矩阵M_t，仅在运动相关区域（M_t≈1）施加校正，避免背景和非目标区域被污染。

## 可复现要素
- **数据集**：使用自定义prompt生成数据，非公开基准；20个motion concept各有up to 3 prompts。
- **代码**：论文未明确声明开源，仅提及supplementary material含baseline实现细节。
- **模型权重**：使用开源的Wan2.1-T2V-1.3B和CogVideoX-2B，可从官方仓库获取。
- **关键超参**：α=5.0（唯一超参）、CFG scale s按模型默认、spatial gating开启、mean-replacement neutralization。
