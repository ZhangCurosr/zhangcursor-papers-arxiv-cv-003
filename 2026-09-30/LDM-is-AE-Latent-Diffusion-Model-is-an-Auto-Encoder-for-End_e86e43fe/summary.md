---
title: "LDM-is-AE-Latent-Diffusion-Model-is-an-Auto-Encoder-for-End"
source: https://arxiv.org/pdf/2609.37080v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 17:30:31"
---

# 论文速读：LDM-is-AE: Latent Diffusion Model is an Auto-Encoder for End-to-End Image Generation

## 一句话总结
本文揭示传统潜在扩散模型（LDM）的 Transformer 骨干网络本身隐含“潜在→特征→潜在”的解码-编码变换，提出 **LDM-is-AE** 端到端单阶段训练框架，通过显式对齐中间特征与图像域，消除对预训练 Tokenizer 的依赖，实现扩散原生潜在空间与去噪动力学的联合学习。

## 研究问题与动机
1. **两阶段范式固有缺陷**：现有 LDM 普遍先预训练独立 AE 定义固定潜在空间，再冻结该空间训练扩散模型，导致潜在表示仅针对重建优化，无法自适应去噪动力学，产生 Representation Mismatch。
2. **现有联合训练方案仍依赖外部模块**：如 REPA-E、UNITE 等虽尝试端到端训练，但仍保留独立的编码器/解码器或强依赖外部视觉基础模型（VFM），计算开销大且未真正融合表征与生成过程。
3. **隐式结构未被挖掘**：DiT 骨干天然具备前半段扩张、后半段压缩的对称结构，可解释为内部解码-编码过程，但此前缺乏显式机制将其与图像域连接，阻碍了潜在空间的动态演化。
4. **训练效率与性能平衡需求**：像素级扩散（如 JiT、PixelFlow）虽省去潜在编码但算力成本极高；本文希望在潜在空间内以更低开销实现媲美甚至超越两阶段方法的生成质量。

## 核心贡献（创新点）
1. **揭示 DiT 骨干的隐式自编码器本质**：首次从理论层面证明扩散骨干在去噪步天然执行潜在到图像特征再到潜在的变换，可直接作为内生 AE 使用；与已有工作将 AE 视为独立预处理模块不同，本文证明该结构无需额外预训练即可被激活。
2. **提出 DiT-D/DiT-E 显式分解与图像空间对齐机制**：将骨干拆分为解码分支（DiT-D）与编码分支（DiT-E），并在中间插入轻量 MLP 头施加像素空间 MSE+LPIPS 损失，建立显式潜在-图像-潜在路径；区别于 REPA-E/UNITE 保留独立编解码器或依赖 VFM 的方案，本方法仅用单一生成器完成表征学习与去噪。
3. **设计时间感知辅助特征混合（Time-aware Auxiliary Feature Mixing）**：通过 $\gamma(t)=t^k$ 门控动态混合图像对齐特征与辅助特征，在零噪声步保障 AE 路径有效、在噪声步保留去噪容量；避免了对高维特征的过度约束，这是已有单阶段方法未涉及的训练稳定性设计。
4. **构建纯单阶段端到端训练框架**：联合优化扩散 v-loss 与图像对齐损失，潜在表示在整个训练过程中持续演化；相比传统两阶段范式及依赖外部辅助数据的基线，显著降低训练算力（Generator-only FLOPs 减少约 50%）与流水线复杂度。

## 方法详解
- **基础扩散设定**：采用 Flow Matching 框架，噪声潜变量 $z_t = t z_1 + (1-t) z_0$，骨干 $f_\theta$ 预测 $\hat{z}_1$，使用 v-loss：
  $\mathcal{L}_{\text{ldm}} = \mathbb{E}\left[\|(f_\theta(z_t,t,c)-z_t)/(1-t) - (z_1-z_t)/(1-t)\|^2\right]$
- **骨干分解**：将 DiT 前半段定义为 **DiT-D**（解码），后半段定义为 **DiT-E**（编码）。DiT-D 将 $z_t$ 映射为中间特征 $F$，DiT-E 将 $F$ 映射回潜在空间 $\hat{z}_1$。
- **图像空间对齐损失**：对输入图像 $x$ 执行 Pixel Unshuffle（步长 $p$）得 $x_u$，在 DiT-D 输出后接轻量 MLP 投影使通道数与 $x_u$ 一致，施加：
  $\mathcal{L}_{\text{toimg}} = \mathbb{E}_{t,x}\left[w(t)\left(\|F - x_u\|^2 + w_{\text{lpips}} \text{LPIPS}(F, x_u)\right)\right]$
  其中 $w(t) = 1/(1-t)^2$ 与 v-loss 量级对齐，$w_{\text{lpips}}=1$。
- **时间感知辅助特征混合**：DiT-D 输出通道均分给对齐特征 $F$ 与辅助特征 $F'$。输入 DiT-E 的特征为：
  $F_{\text{full}} = \gamma(t) F + (1-\gamma(t)) F', \quad \gamma(t) = t^k$
  $t\to1$ 时 $F$ 主导以保证零噪声步的 AE 路径；噪声步时 $F'$ 提供额外容量，防止过度约束。
- **残差 DiT-E 设计**：DiT-E 输出经归一化后加到从输入特征插值的 skip connection 上，使编码分支聚焦于修正项，稳定训练动态。
- **单阶段训练流程**：给定图像 $x$，先 detach 地通过 DiT-E 获得初始 $z_1$（避免显式重建梯度污染编码路径），再随机采样 $t$ 构造 $z_t$，经 DiT-D 得 $F, F'$，混合后过 DiT-E 得 $\hat{z}_1$。总损失 $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ldm}} + w_{\text{toimg}} \mathcal{L}_{\text{toimg}}$，全部组件联合反向传播。

## 实验与结果
- **数据集**：ImageNet 分类条件生成（256×256 / 512×512），GenEval 文生图基准。
- **256×256 结果**：FID **1.80**，IS **314**。在无 VFM 的单阶段方法中 FID 排名第二，IS 为全场最高；Generator-only 训练 FLOPs 仅 7.02×10^19，为 JiT-H/16（14.0）的一半。
- **512×512 结果**：FID **1.90**，IS **320**，**超越所有对比方法**（包括两阶段 DiT-XL/2 的 3.04、SiT-XL/2 的 2.62，以及像素基线 JiT-H/32 的 1.94），最强结果提升幅度达 **2.1%**（FID 相对 JiT-H/32）。
- **潜空间分析**：在 ImageNet val-50K 上，PSNR 达 **27.57**（优于 VAVAE 26.59、SDVAE 25.94、REPA-E 25.11）；gFID 仅 **1.821**，显著低于对比方法，且生成 FID（1.80）与重建 gFID（1.82）极度接近，验证了 Diffusion-native 潜在空间的成立。
- **消融结论**：最佳 latent channel=128；aux mixing exponent $k=3.0$；DiT-E/DiT-D 深度分割 **2/10** 最优；LPIPS 监督带来正交增益；框架可无缝迁移至 SiT
