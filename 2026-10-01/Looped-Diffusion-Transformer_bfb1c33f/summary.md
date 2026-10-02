---
title: "Looped-Diffusion-Transformer"
source: https://arxiv.org/pdf/2609.40305v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 01:45:52"
---

# 论文速读：Looped-Diffusion-Transformer

## 一句话总结
本文提出 Looped-DiT，通过在每个去噪步内重复运行共享 Transformer 块来增加有效计算深度，并结合深度监督与自调制注意力机制解决朴素循环导致的表征退化问题；260M 参数的模型在多项文生图基准上超越参数量大 6.5 倍的 SOTA 模型，且推理计算量降低约 4.9 倍。

## 研究问题与动机
- 当前文生图模型（如 FLUX.2、Qwen-Image）依赖盲目扩大参数量或增加去噪步数来提升生成质量，部署与推理成本高昂。
- 循环 Transformer（Looped Transformers）可通过共享模块重复计算扩展有效深度而不增加参数量，但在直接应用于文生图扩散模型时，朴素循环无法稳定提升图像质量。
- 实验诊断发现性能退化的两大根源：① 中间循环缺乏直接监督信号，梯度难以有效回溯；② 重复注意力更新不断覆盖或削弱图像 token 的局部空间信息（位置线性可解码性随循环次数显著下降）。
- 亟需一种既能安全放大推理计算、又能保持视觉表征完整性的新扩展范式，而非单纯堆砌参数或步数。

## 核心贡献（创新点）
1. **首次系统验证循环计算在文生图扩散模型中的可行性与优势**：证明在参数量与推理计算完全匹配的条件下，循环深度仍可带来显著性能提升，打破了“仅靠加深网络或增加采样步数”的传统缩放路径。
2. **提出深度监督（Deep Supervision）机制**：将共享后段解码器应用于每个中间循环输出，统一施加 flow-matching 目标，使早期隐状态获得直接且充分的梯度信号，避免长迭代路径中的监督缺失。
3. **提出自调制注意力（Self-Modulating Attention）并给出两种实现**：通过 token/head 依赖的门控（Gated Attention）或无参数的正交投影（XSA）动态调节每次循环的注意力更新幅度，防止重复变换对空间表征的渐进式侵蚀。
4. **揭示循环计算支持潜在视觉推理（Latent Visual Reasoning）**：更深的循环能逐步修正早期错误、解析空间与程序约束；在固定推理 FLOPs 预算下，增加循环深度比增加去噪步数带来更高的性能收益，且与显式文本 CoT 呈互补关系。

## 方法详解
- **整体架构**：基于像素空间 MMDiT（MiniT2I）骨干，将 17 个 Transformer 块划分为三段：前段 A（6块）、循环段 B（5块，参数跨循环共享）、后段 C（6块）。每个去噪步内，$h^{(0)}=\mathcal{A}(h_{\text{input}})$，$h^{(r)}=\mathcal{B}(h^{(r-1)}), r=1,\dots,N$，$\hat{x}_0=\mathcal{C}(h^{(N)})$。训练时 $N=4$，推理时可自由调整。
- **深度监督损失**：对每个循环 $n$，经后段 C 解码得到 $\hat{x}_0^{(n)}$，与干净图像 $x_0$ 计算 flow-matching MSE 损失 $\ell_n = \mathbb{E}\left[\frac{\|\hat{x}_0^{(n)}-x_0\|_2^2}{d_x c(t)^2}\right]$，其中 $c(t)=\max\{1-t, \tau\}$。总损失 $\mathcal{L}=\sum_{n=1}^N w_n \ell_n$，最终采用 Final+Mean 加权 $(1/3,1/3,1/3,1)$。中间预测仅在训练时参与反向传播，推理零开销。
- **自调制注意力**：仅作用于循环段 B。标准头输出 $o_{i,h}=\sum_j \alpha_{ij,h}v_{j,h}$ 经调制因子 $G_{i,h}$ 缩放后拼接：$z_{i,h}=G_{i,h}o_{i,h}$。
  - *Gated Attention*：$G_{i,h}^{\text{gate}}=\sigma(w_{g,h}^\top u_i+b_{g,h})$，可学习标量门控显式控制更新强度。
  - *Exclusive Self Attention (XSA)*：$G_{i,h}^{\text{xsa}}=I-\hat{v}_{i,h}\hat{v}_{i,h}^\top$（$\hat{v}_{i,h}$ 为 token 自身 value 方向的单位向量），将输出投影至与自身 value 正交的子空间，参数-free 且天然非扩张，有效切断重复循环中的冗余自更新。
- **训练设置**：像素空间 flow-matching，timestep 服从 LogitNormal(-0.8, 0.8)，AdamW 优化（峰值 lr 4e-4，batch 1024），条件 dropout 0.1，EMA decay 0.99995。推理使用 Euler 采样 100 步，CFG scale=6.0。

## 实验与结果
- **数据与模型**：预训练 CC12M，微调 BLIP3o-60K/DALL-E 3/ShareGPT-4o-Image。主模型 Looped-DiT B/16（260M，patch=16，N=4），消融用 B/32（patch=32）。
- **基准覆盖**：GenEval、DPG-Bench、PRISM、T2I-CoReBench、SpatialGenEval、TIIF-Short（B/16 全 6 个；B/32 聚焦后 4 个推理相关基准）。
- **参数量效率**：Looped-DiT B/16 平均得分 71.5，超越 InternVL-U（1.7B, 69.0）等同期模型，参数量仅为后者的 ~1/6.5，推理 FLOPs 降低约 4.9 倍；在 DPG-Bench (87.0)、PRISM (67.0)、CoRe (53.5)、Spatial (54.6)、TIIF-Short (79.7) 五项基准上均达最优。
- **计算分配效率**：固定推理预算下，将计算用于增加循环深度（1→4）的收益显著高于用于增加去噪步数（25→50步），且两者独立可调。
- **与 CoT 的互补性**：循环在空间/关系/程序约束解析上增益更大（PRISM +9.1），文本 CoT 提示重写对隐含内容推断更有效（CoRe Generalization +22.2），两者结合取得最高综合分。
- **最强结果**：Looped-DiT B/16 平均 71.5，相对参数量匹配的 MiniT2I 基线（55.2）提升 +16.3 分，相对次优非 CoT 模型 InternVL-U 提升 +2.5 分。

## 相关工作脉络
- **Looped Transformers / Universal Transformers**：Dehghani 等 [7]、Geiping 等 [14] 在语言建模中验证共享层循环可扩展计算深度；本文首次将其系统引入文生图扩散场景，并针对视觉生成特性设计专属监督与调制机制。
- **Elastic Looped Transformers (ELT)** [18]：针对类条件图像/视频生成研究可变循环深度，依赖循环内自蒸馏缓解浅层退出退化；本文直接对中间循环施加与终态相同的 training objective，无需教师配置或额外蒸馏损失。
- **扩散模型缩放路线**：PixArt-α [5]、FLUX [25]、Qwen-Image [54] 等依赖扩大参数或网络宽度；本文证明
