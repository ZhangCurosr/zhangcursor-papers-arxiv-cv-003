---
title: "Motion-Concept-Unlearning-in-Video-Difusion-Models"
source: https://arxiv.org/pdf/2609.36832v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 21:45:04"
field: "视频生成模型安全与可控"
keywords: ["motion concept erasure", "video diffusion models", "text-to-video", "concept unlearning", "classifier-free guidance", "attention analysis"]
innovations: ["提出免训练推理时干预方法MUTE，通过token neutralization提取概念方向并结合自导出空间门控实现运动概念选择性消除", "首次系统探究视频DiT中运动信息的编码位置，证明text-conditioning attention支持概念特定干预而temporal positional encoding支持全局动态", "揭示MUTE与CFG的结构对称性，导出紧凑形式v_final = v_cfg - s·α·M_t⊙d_t"]
benchmarks: ["Wan2.1-T2V-1.3B", "CogVideoX-2B"]
---

# 论文速读：Motion-Concept-Unlearning-in-Video-Difusion-Models

## 一句话总结
本文研究了文本到视频扩散模型中"运动概念"（如踢、刺、射击等动作模式）的选择性消除问题，提出了一种免训练推理时干预方法 MUTE，通过在每个去噪步提取概念方向并结合空间门控，实现目标运动的抑制而保留场景外观与非目标动态。

## 研究问题与动机
- **运动概念的独立性**：现有概念消除工作主要针对静态概念（物体、风格、身份），而"如何移动"这一动态模式（如 kick、stab、shoot）是独立且未被系统研究的问题。
- **视频 DiT 的独特性**：视频扩散 Transformer 引入了时间位置编码（temporal positional encoding）等图像模型中没有的组件，运动信息可能分布在这些新通道中，需要先探究其编码位置。
- **权重级方法的局限**：直接将图像领域的权重级消除方法（如 ESD）适配到视频模型，会产生残留的概念信号，该信号会通过 classifier-free guidance (CFG) 被放大，导致消除效果不均。
- **安全部署需求**：后生成安全过滤器容易被对抗性提示绕过，需要在生成过程中直接抑制危险动作概念的生成能力。

## 核心贡献（创新点）
- **系统性地定义了运动概念消除问题**：通过因果干预实验，首次证明 text-conditioning attention 携带概念特定的运动信息且支持选择性干预，而 temporal positional encoding 支持全局动态、不适合概念特定干预，这与图像模型中的假设形成对比。
- **提出了免训练的推理时干预方法 MUTE**：无需修改模型权重或训练，在每个去噪步通过 token neutralization 提取概念方向、从方向自导出空间门控、在 CFG 放大前从速度输出中减去校正项，实现概念特异性、空间选择性与时间自然性三个要求。
- **跨架构验证与扩展**：在 Wan2.1-T2V（分离式 cross-attention）和 CogVideoX（联合 attention）两个不同架构上验证方法有效性，并证明可自然扩展到联合运动-物体消除。
- **揭示 CFG 与概念消除的结构关联**：推导出 MUTE 的紧凑形式 $v_{\text{final}} = v_{\text{cfg}} - s \cdot \alpha \cdot M_t \odot d_t$，表明 MUTE 相当于一种具有空间选择性的"概念特定负向引导"。

## 方法详解
- **问题形式化**：给定视频 DiT 的速度预测 $v_\theta(x_t, t, c)$ 和目标运动概念 C，寻求校正项 $\Delta_C$ 使得 $v_{\text{erased}} = v_\theta(x_t, t, c) - \Delta_C(x_t, t, c)$，满足三个要求：(R1) 概念特异性、(R2) 空间选择性、(R3) 时间自然性。
- **概念方向提取（R1）**：通过 token neutralization 操作，将提示中目标关键词子词位置的上下文嵌入替换为整个固定长度上下文的均值嵌入（保持其他位置不变），得到修改后的条件 $\tilde{c}$，概念方向为 $d_t = v_\theta(x_t, t, c) - v_\theta(x_t, t, \tilde{c})$。该操作作用于 context 序列而非特定注意力布局，适用于分离式和联合式 attention。
- **自导出空间门控（R2, R3）**：观察到概念方向 $d_t$ 在空间上高度集中（峰值与均值比超过 20:1），计算每位置的通道范数作为影响图 $I_t(i,j,k) = \|d_t(:,i,j,k)\|_2$，归一化后得到空间门控 $M_t = I_t / \max(I_t)$，在运动相关区域 $M_t \approx 1$，其他区域接近 0，无需外部分割模型。
- **完整校正公式**：校正项 $\Delta_C = \alpha \cdot M_t \odot d_t$，擦除后的速度为 $v_{\text{erased}} = v_\theta(x_t, t, c) - \alpha \cdot M_t \odot d_t$，代入 CFG 公式得紧凑形式 $v_{\text{final}} = v_{\text{cfg}} - s \cdot \alpha \cdot M_t \odot d_t$，其中 $\alpha$ 为唯一超参数（默认 5.0）。
- **推理开销**：每步需要三次前向传播（条件、neutralized、无条件），相比标准 CFG 增加约 47% 的 wall-clock 时间。

## 实验与结果
- **数据集与模型**：在 Wan2.1-T2V-1.3B（832×480，81帧）和 CogVideoX-2B（720×480，49帧）上评估，涵盖 20 个运动概念（kick、punch、slap 等），每个概念最多 3 个提示。
- **评估指标**：X-CLIP 运动一致性分数（MCS，下降表示抑制）、LPIPS（感知距离）、SSIM（结构相似性）及定性评估。
- **最强结果**：在 Wan2.1-T2V 上，MUTE 的平均 ΔMCS 为 +1.24，显著优于 negative prompting（+0.24）、UCE（+0.09）、weight-level ESD（+0.33）、T2VUnlearning（+0.50）和最强的 baselines VideoEraser（+1.09）。对 punch 的消除效果最好（ΔMCS +3.81 vs. VE +2.43），对 kick 达到 +0.75。
- **跨架构迁移**：在 CogVideoX 上，平均 MCS 从 0.90 降至 0.61，相同算法配置无需额外调参即可工作，尽管 joint attention 设计导致效果略弱于分离式 cross-attention。
- **保真度**：Wan2.1-T2V 上平均 LPIPS 0.69、SSIM 0.31；CogVideoX 上 LPIPS 0.45、SSIM 0.54。定位动作（如 drag、kick）比全身动作（如 tackle）保真度更高。
- **人工评估**：17 名受试者盲评，MUTE 在运动消除方面获得 57.9% 投票（VideoEraser 12.1%），在视觉质量方面获得 82.4% 投票（VideoEraser 7.6%）。

## 相关工作脉络
- **ESD / UCE**（文本到图像概念消除）：权重级方法，通过微调 cross-attention 或 closed-form 编辑消除静态概念；本文发现直接适配到运动概念时，因 CFG 放大残留信号导致效果有限，促使转向输出级干预。
- **VideoEraser / T2VUnlearning**：针对视频模型的静态概念消除方法；本文指出这些方法将运动视为静态属性处理，导致 phantom objects 等伪影，而 MUTE 通过概念方向和空间门控避免这些问题。
- **ConceptVoid / PROBE**：视频多概念消除与残留概念诊断；本文补充了运动概念这一独特维度的研究空白，并提供了系统性探针分析。
- **Cross-attention 语义分析**（Prompt-to-Prompt、DAAM）：图像领域证明 cross-attention 携带类别级语义；本文将其扩展到视频 motion，但发现 motion 在 temporal positional encoding 中有额外支撑，需要更精细的选择性干预。
- **Video Unlearning via Low-Rank Refusal Vector**：通过低秩拒绝向量在权重级别抑制不安全概念；本文定位为不同的范式——不改变权重，而是在推理时动态估计并移除概念贡献。
- **Classifier-free Guidance**：标准 CFG 放大条件与无条件预测的差异；本文揭示 MUTE 的结构本质是概念特定的负向引导，扩展了 CFG 的应用语义。

## 局限性与未来方向
- **token neutralization 的估计偏差**：部分运动信息可能通过文本编码器或 cross-attention 中的上下文交互分布到非目标 token，导致 $d_t$ 低估实际概念贡献，需要 $\alpha > 1$ 补偿，但仍有残留信号。
- **joint attention 架构的局限性**：在 CogVideoX 中，text 和 visual tokens 共享同一注意力层，neutralization 对 velocity field 的影响更分散，消除效果弱于分离式 cross-attention。
- **全身动作的保真度挑战**：tackle、slam 等全身动作的 LPIPS 较高（0.82、0.84），空间门控难以完全隔离运动区域与背景。
- **未探索持续学习场景**：多概念相继消除时的retain-forget entanglement 问题尚未涉及。
- **计算开销**：每步增加约 47% wall-clock 时间，未来可考虑更高效的近似方法。

## 研究启发与可借鉴点
- **探针驱动方法设计**：先通过因果干预实验（零化 motion-token embedding、缩放 temporal RoPE）定位信息编码位置，再据此设计干预策略，避免盲目适配已有方法。
- **从 CFG 公式推导方法形式**：将 MUTE 表示为 $v_{\text{final}} = v_{\text{cfg}} - s \cdot \alpha \cdot M_t \odot d_t$，揭示了与 CFG 的结构对称性，这种"反向引导"视角可迁移到其他控制任务。
- **自导出空间门控的设计**：利用概念方向自身的空间集中度（peak-to-mean ratio > 20:1）自动构建门控，无需外部分割模型，这一思想可用于其他需要空间局部化的编辑任务。
- **跨架构泛化验证**：在分离式 cross-attention 和联合 attention 两种架构上统一验证，证明方法不依赖特定实现细节，为后续工作提供通用评估范式。
- **联合消除的扩展路径**：通过独立计算多个概念方向和门控并叠加校正，可自然扩展至多概念联合消除，为复杂场景控制提供思路。

## 关键术语表
**Motion Concept Erasure**：从文本到视频生成模型中选择性抑制特定动作模式（如踢、刺）的概念消除，区别于静态物体或风格的消除。

**Token Neutralization**：将提示中目标关键词子词位置的上下文嵌入替换为整个上下文的均值嵌入，以隔离该词对速度预测的贡献。

**Concept Direction ($d_t$)**：条件预测与 neutralized 条件预测的速度差，估计目标运动概念在每一步的贡献。

**Spatial Gate ($M_t$)**：由概念方向的空间范数归一化得到的影响图，用于将校正限制在运动相关区域。

**Classifier-Free Guidance (CFG)**：通过放大条件与无条件预测的差异来增强生成质量的推理技术，公式为 $v_{\text{cfg}} = v_\theta(x_t, t, \emptyset) + s \cdot (v_\theta(x_t, t, c) - v_\theta(x_t, t, \emptyset))$。

**Text-Conditioning Attention**：视频 DiT 中将文本条件注入生成过程的注意力层，在 Wan2.1-T2V 中为分离式 cross-attention，在 CogVideoX 中为联合 attention。

**Temporal Positional Encoding**：视频 DiT 中编码帧时序关系的位置编码，本文发现其支持全局时间动态而非概念特定的运动信息。

**Motion Consistency Score (MCS)**：基于 X-CLIP 的运动一致性评估指标，衡量生成视频中目标运动的强度，ΔMCS 越大表示抑制越强。

## 可复现要素
- **数据集**：论文使用自构建的 20 个运动概念提示集合（每个概念最多 3 个提示），未提及公开数据集。
- **代码/权重**：论文未明确声明开源，但提到 Wan2.1-T2V 和 CogVideoX 为开源模型；补充材料提供 baseline 实现细节。
- **关键超参**：$\alpha = 5.0$（消融实验显示 1.0–10.0 范围内平滑 trade-off）；默认使用 spatial gating 和 mean-replacement neutralization。
