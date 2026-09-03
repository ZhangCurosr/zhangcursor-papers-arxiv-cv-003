---
title: "Training-Free-Inpainting-Across-Domains-with-a-Frozen-Text-t"
source: https://arxiv.org/pdf/2609.00862v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 00:49:46"
field: "扩散模型推理时控制与零训练修复"
keywords: ["training-free inpainting", "frozen diffusion model", "test-time control", "PI controller", "cross-domain transfer", "latent diffusion", "boundary-interior feedback", "release schedule"]
innovations: ["将 PI 控制结构引入冻结 SD1.5 的 DDIM 逆向轨迹，分离边界-内部反馈与持久状态", "四字段预设释放调度实现逆向进程中的动作调制，无需任何权重更新", "仅用 CelebA-HQ pilot 调参即实现到 AFHQ 和 Places2 的配置锁定零调整迁移"]
benchmarks: ["Main35 (AFHQ 1300 + CelebA-HQ 1400 + Places2 800, 35 protocols)", "LanPaint native v1.5.5 ComfyUI", "PILOT official DDIM", "SD-Inpaint", "BrushNet", "PixelHacker (Places2-finetuned)"]
---

# 论文速读：Training-Free Inpainting Across Domains with a Frozen Text-to-Image Diffusion Model

## 一句话总结
本文提出 Step-PI，一种零训练的闭环潜变量控制器，利用冻结的通用 SD1.5 文生图扩散模型在三个自然图像域（AFHQ、CelebA-HQ、Places2）上完成条件修复（inpainting），无需任何权重更新或专用修复条件通道。

## 研究问题与动机
1. **现有修复方法依赖任务专用权重**：主流扩散修复器（如 SD-Inpaint、BrushNet、PixelHacker）通过大量离线训练获得修复能力，使用了专用的潜在接口或双分支架构，无法直接复用冻结的通用预训练模型。
2. **通用文生图模型缺乏修复特定资源**：原始 SD1.5 没有掩码编码机制、无修复微调权重，也无比对已知/未知区域的显式控制逻辑，其直接用于修复效果有限。
3. **现有零训练方法的接口异质性阻碍因果归因**：LanPaint（Langevin + ODE 采样）、PILOT（DDIM 中优化潜变量）、RePaint（重采样）、LatentPaint 等基础框架和条件接口各不相同，难以进行组件级对比。
4. **测试时控制（test-time control）是一条被低估的路线**：通过将经典 PI（比例-积分）控制思想迁移到扩散逆向轨迹上，可以在不修改模型权重的情况下引入边界连续性与内部结构连贯性的显式约束。

## 核心贡献（创新点）
1. **零训练跨域配置锁定验证**：仅用 CelebA-HQ 上的少量 pilot 调参，固定配置零调整直接迁移到 AFHQ 和 Places2，证明单冻结通用 T2I 先验可支持多域修复。
2. **Step-PI 控制器设计**：将 PI 结构（当前反馈 + 持久历史状态）引入冻结 SD1.5 的 DDIM 逆向轨迹，分离边界/内部目标、指数折扣记忆与四字段预设释放调度，共同调制每次逆向步的控制动作。
3. **逐组件可控的对照实验设计**：通过 DDIM-Proj → P-Guidance → PI → Step-PI 的阶梯式增量对照，精确分离持久状态和预设调度各自的因果贡献，全部 15 个数据集-指标格均显著改善；同时提供五种子重启鲁棒性审计和提示词存在性干预。
4. **推理效率-质量的清晰定位**：明确说明本方法以在线梯度计算换取离线参数训练，A100 上 30.48s/image（15.55 GiB），不是速度最优方案，但为零训练路径提供了可比的全新参考点。

## 方法详解
**整体接口（闭环控制结构）**：在每步 $k$（从 $N$ 到 1）执行：先做已知区域投影 $\Pi_{t_k}(z^k)$，再经冻结 U-Net + CFG 得到去噪估计，通过 DDIM 转移得到基线潜变量 $z_{\mathrm{base}}^{k-1}$，控制器 $\mathcal{C}_k$ 在当前解码估计 $\widehat{x}_0^k$、原图 $y$、掩码 $M$、持久状态和历史释放向量上生成动作 $a_k$，最后加到 $z_{\mathrm{base}}^{k-1}$ 并再次投影：$z^{k-1} = \Pi_{t_{k-1}}(z_{\mathrm{base}}^{k-1} + a_k)$。最终 RGB 合成时已知像素精确复原。

**已知区域投影**：由可见像素构造无泄漏参考潜变量 $z_0^{\mathrm{known}}$，沿固定噪声轨迹 $q_t^{\mathrm{known}}$ 在每一步回填已知区域坐标，使已知部分不随逆向过程漂移。

**边界–内部反馈目标**：
- 边界目标 $\mathcal{L}_B = \lambda_1^B \mathcal{L}_{\mathrm{known}} + \lambda_2^B \mathcal{L}_{\mathrm{pair}} + \lambda_3^B \mathcal{L}_{\mathrm{TV}} + \lambda_4^B \mathcal{L}_{\mathrm{boundary-grad}}$，分别对应已知区 RGB 保真、接缝配对、支持掩码 TV 平滑和 RGB 梯度匹配；权重 $(0.50, 1.00, 0.05, 0.20)$。
- 内部目标 $\mathcal{L}_I = \lambda_1^I \mathcal{L}_{\mathrm{lowfreq}} + \lambda_2^I \mathcal{L}_{\mathrm{interior}} + \lambda_3^I \mathcal{L}_{\mathrm{ring}} + \lambda_4^I \mathcal{L}_{\mathrm{frequency}}$，促进多尺度（半径 16/32/64）上下文连贯、深度加权、环形均值/方差对齐与高频一致性；权重 $(0.20, 0.15, 0.02, 未给出具体值)$。
- 反馈方向 $g_B^k = \mathrm{Normalize}_2[-M_z \odot \nabla_{\tilde{z}^k} \mathcal{L}_B]$，$g_I^k = \mathrm{Normalize}_2[-D_z \odot \nabla_{\tilde{z}^k} \mathcal{L}_I]$，经张量 $\ell_2$ 归一化消除跨目标/跨时间步尺度差异。

**持久 PI 结构状态**：边界与内部各维护一个指数折扣记忆 $\xi_B^k$、$\xi_I^k$，递推为 $\xi_j^k = \mathcal{P}_j[m_j \odot \xi_j^{k+1} + (1-m_j) \odot (w_j \odot g_j^k)]$，其中边界域用空间变化保留率 $\beta_B$（内边界 0.95、深内部 0.70）和深度衰减权重 $W_B$，内部域使用统一保留率 $\gamma_I = 0.90$；边界状态受半径 $\nu_B=1$ 的范数约束。该递推等价于当前方向与历史方向的有界凸组合（Proposition 1）。

**四字段预设释放调度 $\rho_k$**：
- $r_B(t_k)$、$r_I(t_k)$ 依赖 scheduler 累积系数 $\bar{\alpha}_{t_k}$，分别控制边界记忆积分增益和最终内部输出增益，前者从 0.50 线性升至 1.00，后者取 $\max(\bar{\alpha}_{t_k}, 0.05)$。
- $q_k$（公共内部增益）和 $h_k$（额外内部记忆释放）依赖归一化逆向步进度 $p_k$，$q_k$ 采用分段 cubic smoothstep 从 $q_0=0.25$ 升至 $q_m=1.00$（$p_k \in [0.10, 0.30]$），保持平台至 $p_k=0.70$，再降至 $q_f=0.15$（$p_k \in [0.70, 0.90]$）；$h_k$ 在 $p_k > 0.55$ 后按 smoothstep 衰减至 0。
- 最终动作 $a_k = \mathrm{Clamp}[-a_{\max}, a_{\max}][M_z \odot (u_B^k + u_I^k)]$，其中 $a_{\max} = 0.12$，内部增益 $K_P^I = 0.20, K_I^I = 0.18$，边界 $K_P^B = 0.08, K_I^B = 0.40$。

**关键区别**：本方法不反向传播过 DDIM 转移（无梯度穿过 frozen U-Net/VAE），反馈在当前投影潜变量上计算，动作施加在转移后的潜变量上，是因果的单次前向施加（causal one-pass actuation），不是经典意义上的梯度下降。

## 实验与结果
**数据集与协议**：Main35 共 3,500 组配对样本，来自 AFHQ v2（1,300）、CelebA-HQ（1,400）、Places2（800），覆盖 35 个固定 100 样本掩码协议（mask area 2.6%–40.1%）。控制器仅在 Main35-disjoint 的 CelebA-HQ pilot 上调参并冻结，AFHQ 和 Places2 为纯迁移目标。

**评估指标**：Masked L1、Boundary L1、Masked LPIPS、Boundary LPIPS、CLIP-Q（更高更好），共 5 项。

**主要结果（Table 1/2）**：
- 内部阶梯：Step-PI 在所有 15 个 dataset–metric 格中均最优，单调改进沿 DDIM-Proj → P-Guidance → PI → Step-PI。
  - CelebA-HQ Masked L1：0.1611 → 0.1278（Step-PI 相对 DDIM-Proj 提升约 20.7%）
  - CelebA-HQ Boundary L1：0.0376 → 0.0236（提升约 37.2%）
  - Places2 Boundary L1：0.0472 → 0.0317（提升约 32.8%）
- 持久状态贡献（PI vs P-Guidance）：全部 15 格改善，均值相对增益 Masked L1 +2.4%、Boundary L1 +3.9%、Boundary LPIPS +2.6%、CLIP-Q +1.0%。
- 释放调度贡献（Step-PI vs PI）：全部 15 格改善，均值相对增益 Masked L1 +7.8%、Boundary L1 +24.4%、Boundary LPIPS +12.6%、Masked LPIPS +4.7%、CLIP-Q +4.6%。
- 外部对比（Table 2）：在零训练组内，Step-PI 在全部 5 项 equal-dataset macro 指标上领先 LanPaint 和 PILOT。
- 绝对最优仍属于训练型方法：SD-Inpaint Masked L1=0.1198、Boundary LPIPS=0.0733；PixelHacker Boundary L1=0.0209、Boundary LPIPS=0.0713；但它们需要 440k 步 / 430k 步 / 200k 步的离线训练，且 PixelHacker 还在 Places2 上做了适配微调。

**鲁棒性审计**：
- 五种子逆序测试（MS175，875 对）：PI 在全部 5 指标上均优于 P-Guidance（95% CI 全排除 0）；Step-PI 在全部 5 指标上均优于 PI（14/15 CI 排除 0，Places2 CLIP-Q 区间含 0 但仍为正）。
- 单字段均匀化审计：将四字段中任意一个替换为 1（其余保持预设），全部 20 个（4 字段 × 5 指标）均值对比均指向完整 Step-PI 更优，且每对比 175/0/0 全胜。

**计算成本（A100 35 样本审计）**：Step-PI 中位 30.48 s/image，峰值 15.55 GiB；相同步骤数的 DDIM-Proj 仅 3.07 s/image（4.18 GiB）。每步 2 次反馈梯度（共 100 次 autograd call），无参数更新。

## 相关工作脉络
1. **SD-Inpaint / BrushNet / PixelHacker**：训练型修复代表，使用修复专用 checkpoint 或双分支结构化接口，通过大量离线优化获得任务能力；本文将其作为"最强质量参考"而非匹配对照，用于资源语境对照。
2. **LanPaint（2025）**：同基于 vanilla SD1.5 的零训练方法，结合 Langevin 动力学与 ODE 采样，在官方默认配置下被本文作为外部基准；两者 native route 不同，本文强调这是描述性定位而非 tuning-matched 排序。
3. **PILOT（2024）**：在 DDIM 采样期间对潜变量做周期性梯度优化（每 10 步 10 次潜梯度），也是 vanilla SD1.5 零训练路线，被本文同等对待；Step-PI 的区别在于使用闭环 PI 反馈而非直接潜优化。
4. **GradPaint（2024）**：对 masked MSE 和边界损失做反向传播，适用于 pixel-space 模型；本文指其接口异质，不做量化对比。
5. **HiGS（2026）**：利用 denoiser-prediction 历史作通用采样增强，不针对 inpainting 评估；本文未纳入定量比较。
6. **通用测试时控制文献（DPS、Universal Guidance 等）**：涵盖 posterior gradient、可微 guidance、轨迹优化等；本文强调这些方法接口多样，不利于组件级因果归因，而 Step-PI 通过统一 SD1.5 基座 + 字段等同对照实现可解释消融。

## 局限性与未来方向
1. **推理成本高**：30.48 s/image 远高于训练型方法和 LanPaint（11.38 s）/ PILOT（23.51 s），不适合实时场景。
2. **迁移范围受限**：仅验证了从 CelebA-HQ pilot 到 AFHQ、Places2 的配置锁定向外迁移，未测试跨 backbone、VAE、分辨率、步数或 sampler 的迁移。
3. **提示词粒度有限**：使用的提示词为简单 class-level（如"realistic portrait photo of a person"），未测试细粒度 masked-content 控制、指令跟随或 paraphrase 鲁棒性。
4. **释放调度的非最优性声明**：作者明确说明四字段为人工预设、未证明全局最优或最小性，也未与共享标量或替代调度做充分对比。
5. **统计推断边界**：Bootstrap 区间条件于固定的 35 协议和样本池，不能泛化到未见 mask 族或新域。

## 研究启发与可借鉴点
1. **PI 结构在扩散逆向轨迹上的迁移**：将经典比例-积分控制的"当前反馈 + 折扣历史"分解引入确定性感知的 DDIM 轨迹，是一种可复用的控制思想框架，可用于其他零训练扩散任务（如超分、去噪、编辑）。
2. **字段等同（field-identical）对照实验范式**：通过仅改变一个组件（持久状态 / 释放调度）来隔离因果效应，并提供五种子逆序 + 单字段均匀化审计，是非常规范的消融策略，可迁移到其它复杂 diffusion controller 研究中。
3. **边界-内部目标分离的设计思路**：将局部接缝连续性和远距离结构连贯性解耦为两套独立目标，并以深度加权 $D_z$ 调制内部方向，这一思路可直接复用到任何需要"边缘平滑 + 内部生成"协同的任务。
4. **测试时控制换离线训练的 tradeoff 分析**：本文清晰给出了"算力从训练迁移到推理"的定位（100 次 autograd call vs 数百万步权重训练），为团队选择方案时的成本预算提供明确参照。
5. **确定性 DDIM（η=0）作为可控 substrate**：选择无随机噪声的逆向采样以消除跨步随机性干扰，使机制证据更干净；未来扩展到 stochastic sampler 时需重新评估。

## 关键术语表
**Step-PI**：本文提出的零训练闭环潜变量控制器，基于冻结 SD1.5 和确定性 DDIM，在逆向轨迹上叠加边界-内部反馈、持久 PI 状态与四字段预设释放调度。
**Known-region projection ($\Pi_t$)**：在每一步逆向中将已知区域坐标替换为沿固定噪声轨迹的参考潜变量，避免已知信息漂移。
**Boundary–interior feedback**：从当前解码估计上对边界目标 $\mathcal{L}_B$ 和内部目标 $\mathcal{L}_I$ 求梯度，得到归一化方向 $g_B^k$、$g_I^k$，分别约束接缝连续性和内部结构连贯性。
**Persistent PI state ($\xi_B, \xi_I$)**：指数折扣的反馈方向历史记忆，递推为当前与历史方向的有界凸组合（Proposition 1），提供跨步方向累积。
**Release schedule ($\rho_k$)**：四字段向量 $(r_B, q_k, h_k, r_I)$，分别在边界积分、公共内部增益、额外记忆释放和最终内部输出上随逆向进程调制动作幅度。
**DDIM-Proj / P-Guidance / PI / Step-PI**：四种内部对照变体，依次增加投影、当前反馈、持久状态、完整释放调度，构成控制器的阶梯式构建链。
**Main35**：本文统一的 3,500 样本评估集，来自 AFHQ（1,300）、CelebA-HQ（1,400）、Places2（800），覆盖 35 个固定 100 样本掩码协议。
**MS175**：从 Main35 中按协议分层随机抽取的 175 样本无结果盲子集，用于持久状态和释放调度的五种子鲁棒性审计。

## 可复现要素
- **数据集**：AFHQ v2、CelebA-HQ（原生 512px）、Places365 Standard validation（取 800 样本为 Places2），均为公开数据集；论文声明代码、实验配置、样本列表和评估脚本将在发表后开源。
- **代码/权重**：未发布；论文承诺开源 Step-PI 实现、case 列表、评估脚本和表图生成代码；第三方 checkpoint（SD1.5 revision 451f4fe1、original VAE、openai/clip-vit-base-patch32）通过官方渠道获取。
- **关键超参**：50-step DDIM、CFG=7.5、$\eta=0$、输入尺寸 512×512、VAE 缩放 $s_{\mathrm{VAE}}=0.18215$、FP16 backbone + FP32 反馈解码；边界权重 $(0.50, 1.00, 0.05, 0.20)$、内部权重 $(0.20, 0.15, 0.02, \cdots)$；增益 $(K_P^B, K_I^B)=(0.08, 0.40)$、$(K_P^I, K_I^I)=(0.20, 0.18)$；保留率 $\mu_{B,S}=0.95, \mu_{B,I}=0.70, \gamma_I=0.90$；clamp $a_{\max}=0.12$；释放常数见 Appendix A.4 Eq.41。
- **GPU 环境**：NVIDIA A100-SXM4-80GB，batch=1。
