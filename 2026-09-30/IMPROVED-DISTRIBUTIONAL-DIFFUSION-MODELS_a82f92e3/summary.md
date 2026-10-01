---
title: "IMPROVED-DISTRIBUTIONAL-DIFFUSION-MODELS"
source: https://arxiv.org/pdf/2609.37147v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 17:29:09"
field: "生成模型与快速采样"
keywords: ["Diffusion Models", "Few-step Generation", "Distributional Denoising", "Flow Matching", "Energy Score", "Scoring Rules"]
innovations: ["延迟粒子扩展降低多粒子训练开销至1.5x FM", "基于动力学相变的时变评分规则调度(lambda(t), beta(t))", "单阶段from scratch训练实现4步FID 4.48"]
benchmarks: ["ImageNet-256^2", "MS-COCO T2I"]
---

# 论文速读：IMPROVED-DISTRIBUTIONAL-DIFFUSION-MODELS

## 一句话总结
论文提出了 iDDM（Improved Distributional Diffusion Models），通过**延迟粒子扩展**和**时变评分规则调度**解决了分布扩散模型在大规模图像生成中的可扩展性问题，使得模型在单阶段、无蒸馏条件下实现了 4 步 FID 4.48 / 50 步 FID 2.38 的优异 Few-Step 生成效果。

## 研究问题与动机
- **Few-step 生成的方差挑战**：在少量采样步（如 4 步）下，条件分布 $p(x_1|x_t)$ 方差较大，仅建模条件均值的标准去噪器无法充分刻画后验分布，导致采样质量严重下降。
- **原始 DDM 的计算开销瓶颈**：原 DDM 对每个样本需生成 $m$ 个粒子并分别完整通过整个 Transformer，计算开销随粒子数线性放大（4 倍于 FM），难以扩展到大模型规模。
- **固定超参数的局限性**：原始 DDM 使用全局固定的 $(\lambda, \beta)$ 评分规则超参数，无法适配扩散轨迹不同阶段（噪声端 vs. 数据端）的后验分布差异——噪声端需要多样性，数据端需要保真度。
- **缺乏从 scratch 训练的实用方案**：现有 Few-step 方法多依赖多阶段蒸馏或预训练教师模型，难以直接复用。

## 核心贡献（创新点）
1. **延迟粒子扩展（Deferred Population Expansion）**：仅在 Transformer 最后若干层展开粒子并注入噪声 $\xi$，前期共享表示提取，将训练开销从 4×FM 降至约 1.5×FM，同时保留了分布建模能力。
2. **时变评分规则调度（Time-dependent Scoring Rule Schedules）**：基于 Biroli 等（2024）的动力学相变理论，将 $(\lambda(t), \beta(t))$ 设计为时间函数，在低 SNR 阶段偏向分布学习目标、在高 SNR 阶段偏向回归目标，实现单一模型适配 4–50 步全范围采样。
3. **$\xi$ 注入机制的系统化探索**：对比了通道拼接、残差加法、AdaNorm、Register Token 四种方案，发现固定内宽拼接（fixed-inner-width concatenation）在少步场景下表现最优，提供了实用的工程设计指导。
4. **从 scratch 单阶段训练的实证验证**：在 DiT-B 和 DiT-XL/2 上均实现了无需蒸馏、无 CFG 训练、无自引导的强 Few-step 生成，且 FID 随步数增加单调下降（4.48→2.38），证明了方案的有效性和一致性。

## 方法详解
**延迟粒子扩展架构**：
- 设 Transformer 共 $L$ 层，在前 $\ell_\text{start}$ 层以 batch size $B$ 运行，共享中间表示；在第 $\ell_\text{start}$ 层进行粒子复制（batch size 变为 $B \times m$），并将辅助噪声 $\xi_j$ 注入到隐藏状态中；后续 $L - \ell_\text{start}$ 层以 $B \times m$ 的 batch size 运行。
- 计算开销比例：$(\ell_\text{start} + (L - \ell_\text{start}) \cdot m) / (L \cdot m)$，DiT-XL（$L=28, \ell_\text{start}=24, m=4$）下约 36% 的原始 DDM 开销。
- $\xi$ 注入方式采用**固定内宽拼接**：将 $\tilde{\xi}_j$ 拼接到 residual stream（维度 $d + d_\text{cat}$），但后续层的 attention 和 MLP 内部宽度保持 $d$，velocity 从前 $d$ 个通道读出。

**时变评分规则调度**：
- 广义能量评分（Generalized Energy Score）：
$$S_{\lambda,\beta}(p, y) = -\mathbb{E}_p[\|X - y\|_2^\beta] + \frac{\lambda}{2}\mathbb{E}_{p \otimes p}[\|X - X'\|_2^\beta]$$
- 调度公式：$\lambda(t) = \lambda_\text{max} \cdot s_\lambda(t)$，$\beta(t) = 2 - (2 - \beta_\text{min}) \cdot s_\beta(t)$，默认采用线性调度 $s(t) = 1 - t$，$\beta_\text{min} = 0.1$。
- 调度依据 Biroli et al. (2024) 的三个动力学阶段：Regime I-II（前 speciation 和 speciation 到 collapse 之间，分布广泛，偏好分布性目标）；Regime IIIa（collapse 后模糊窗口）；Regime IIIb（高 SNR 数据端，偏好回归目标）。
- 训练时 $t$ 采样使用 jit 分布（mode ≈ 0.25，偏向噪声端），以增强分布损失在关键区域的覆盖。

**训练流程**：
- 整体为单阶段训练，无教师模型蒸馏、无自蒸馏、训练时无 CFG；推理时可应用 Classifier-Free Guidance。
- 使用 DiT 架构 + 潜变量（Latent，f8d4，REPA-E VAE）。

## 实验与结果
**数据集与设置**：ImageNet-256²（类条件），DiT-B（depth 12, width 768）训练 400k 步；DiT-XL/2（depth 28, width 1152）训练 200 epochs。评估使用 FID@50k、FD-DINOv2、KID、KDD、Inception Score、Precision、Recall。

**主要结果（DiT-B，Table 1 & 4a）**：
- 4 步 FID：iDDM **13.13** vs. 原始 DDM 43.68（提升 30.55），vs. Flow Matching 26.53（提升 13.40）。
- 50 步 FID：iDDM **4.57** vs. FM 4.97（提升 0.40），vs. 原始 DDM 10.04。
- 计算代价：iDDM 每步仅 1.5× FM，远低于原始 DDM 的 4×。

**系统级结果（DiT-XL/2，Table 4b）**：
- 4 步 FID：**4.48**，50 步 FID：**2.38**。
- 在 XL 尺度下优于 MeanFlow-XL/2（4 步 2.93，但 iMF 训练 800 epochs 对比 iDDM 200 epochs）。
- 与 iMF/IMM 相比，iDDM 优势在于**单阶段 from scratch 训练** + **4–50 步范围内 FID 单调非递增**。

**Text-to-Image 迁移（Table 13）**：
- MS-COCO 4 步 FID：iDDM **41.05** vs. FM 78.20（几乎减半）；8 步 FID：19.40 vs. 26.33。

**消融验证**：
- 延迟扩展起始层 $\ell_\text{start} \in [8,10]$ 效果最佳（Table 2a）。
- Fixed-inner-width concatenation 优于 Ada-norm、additive、tokens 等方案（Table 2b）。
- 线性调度最优，dyn-reg 次之（Table 3b）。
- jit 时间采样优于 uniform、logit-normal、imf（Table 3c）。

## 相关工作脉络
- **DDM（De Bortoli et al., 2025b）**：本文的直接前身，使用固定 $(\lambda, \beta)$ 和全网络多粒子传播，计算开销大；本文的核心改进在于使其可扩展。
- **Flow Matching（Lipman et al., 2023）**：确定性 velocity field 学习基准；iDDM 在其基础上引入分布性目标，在少步场景下取得显著优势。
- **MeanFlow / iMF（Geng et al., 2025, 2026）**：确定性 Few-step flow map 方法，需多层深度（iMF 48 层）和大量训练步数；iDDM 以更低训练成本实现相近甚至更优的少步效果。
- **Consistency Models / Shortcut（Song et al., 2023；Frans et al., 2025）**：基于自蒸馏的少步生成；iDDM 无需蒸馏，目标不同。
- **IMM（Zhou et al., 2025a）**：多粒子 MMD 训练方法，与 DDM 共享多粒子哲学但右侧目标不同；iDDM 聚焦分布建模与评分规则。
- **MFM（Potaptchik et al., 2026）**：基于对角 flow matching 和一致性损失的随机流映射，多阶段训练；iDDM 走评分规则路线，单阶段训练。

## 局限性与未来方向
- **动力学相变阈值为均值场近似**：$\rho_s, \rho_c, \rho_\text{sep}$ 来源于无穷维分析，在有限尺度下为近似值，平滑调度比硬阈值更鲁棒。
- **1–2 步极限下仍弱于蒸馏方法**：在极端少步场景下，teacher-based distillation（如 sCD、ADD）依然更强。
- **$\lambda(t)$ 与 $\beta(t)$ 的最优调度尚未完全明确**：虽然线性调度表现良好，但理论上最优的时间依赖形式仍是开放问题。
- **目前主要在 ImageNet-256² 验证**：虽已迁移到 T2I，但在更高分辨率、视频或其他模态上的泛化仍需探索。

## 研究启发与可借鉴点
- **延迟计算扩展的工程范式**：将粒子特有的噪声注入推迟到网络末端，可在几乎不增加参数的前提下实现低成本的多粒子训练，该思路可推广到其他需要多样本采样的生成模型（如 Flow Matching 的分布扩展）。
- **基于动力学相变的超参数调度**：利用理论分析（如 Biroli 的 speciation/collapse 时间）指导超参数的时间变化，为评分规则、loss weighting 等其他超参数提供了可借鉴的调度设计思路。
- **噪声注入机制的消融比较**：对 $\xi$ 注入方式（concat vs. additive vs. AdaNorm vs. tokens）的系统消融提供了清晰的工程选型参考，尤其在少步生成场景下。
- **单阶段训练的 Few-step 方案**：无需蒸馏即可达到强 Few-step 效果，降低了部署复杂度，适合对训练资源敏感的场景。

## 关键术语表
**Distributional Diffusion Model (DDM)**：通过广义能量评分规则训练，学习条件分布 $p(x_1|x_t)$ 的随机近似而非仅条件均值的扩散模型变体。

**Deferred Population Expansion**：将多粒子复制和噪声注入延迟到 Transformer 后半段层执行，以减少多粒子训练的计算开销。

**Generalized Energy Score**：参数化为 $(\lambda, \beta)$ 的评分规则，$\lambda$ 控制粒子间交互（多样性）权重，$\beta$ 控制保真度范数阶次，$\lambda=0,\beta=2$ 时退化为标准 MSE。

**Speciation Time ($\tau_s$)**：反向扩散过程中，数据协方差主特征值对应的噪声方差降至 1 的时刻，标志着数据结构开始从噪声中分离。

**Collapse Time ($\tau_c$)**：反向扩散过程中，数据分布熵等于分离高斯混合熵的时刻，标志着后验分布开始向单个训练样本聚集。

**Classifier-Free Guidance (CFG)**：推理时通过条件/无条件预测的差异增强生成质量的常用技术，本文训练阶段不使用 CFG。

**Flow Matching (FM)**：学习从噪声到数据的 time-dependent velocity field 并通过 ODE 积分采样的生成建模框架。

**JIT Sampling**：一种偏向低 SNR（噪声端）的时间采样分布，mode 约在 $t \approx 0.25$，有利于分布损失的充分训练。

## 可复现要素
- **数据集**：ImageNet-256²（公开）；T2I 使用 COYO 子集（公开）。
- **代码**：https://github.com/CompVis/iDDM（已开源）。
- **预训练权重**：已随代码发布。
- **关键超参**：$m=4$（粒子数），$\ell_\text{start}=24$（DiT-XL）或 $10$（DiT-B），$\beta_\text{min}=0.1$，$\lambda_\text{max}=1$，线性调度 $s(t)=1-t$，jit 时间采样（$\mu=-0.8, s=0.8$），学习率 $1\times10^{-4}$，AdamW，batch size 256（B）/1024（T2I）。
- **训练设备**：B 规模 4×H200 约 12 小时；XL 规模 8×H200。
