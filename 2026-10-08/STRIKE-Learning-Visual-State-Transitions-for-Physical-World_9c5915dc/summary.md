---
title: "STRIKE-Learning-Visual-State-Transitions-for-Physical-World"
source: https://arxiv.org/pdf/2610.09514v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 22:54:15"
field: "物理感知视频生成"
keywords: ["physical world modeling", "visual state transition", "video generation", "physics consistency", "robotic manipulation", "event-aligned supervision", "flow matching"]
innovations: ["事件对齐监督：VLM辅助从训练视频提取状态对，监督显式状态转换学习", "可组合状态转换模型：基于相对时间条件图像模型递归预测未来视觉状态", "状态条件化密集视频生成：预测状态作为软条件输入视频扩散模型"]
benchmarks: ["Physics-IQ Verified", "PhyGenBench", "Pisa-Experiments", "RoboTwin2.0"]
---

# 论文速读：STRIKE-Learning-Visual-State-Transitions-for-Physical-World

## 一句话总结
论文提出 STREKE 框架，通过显式学习视觉状态转换（visual state transitions）并将其作为中间表示来条件化密集视频生成，从而提升物理世界建模中的物理一致性与操作视频保真度。

## 研究问题与动机
- **核心问题**：现有视频生成模型倾向于生成视觉连贯但物理不一致的运动，缺乏对交互结果（如接触、附着、变形）的显式预测能力。
- **现有方法不足**：
  1. 端到端视频生成模型（如 CogVideoX、Wan）直接生成完整视频，未显式建模交互引起的状态跃迁。
  2. 基于关键帧的方法（如 KeyWorld、SKIP）依赖稀疏插值，缺乏对转换规范的显式建模。
  3. 物理感知条件化方法主要关注视频级生成或控制，而非单个交互的结果预测。

## 核心贡献（创新点）
1. **事件对齐监督构建**：提出 VLM 辅助的数据标注流水线，从训练视频中提取事件对齐的状态对 $(s^-, u, \Delta\tau, s^+)$，监督转换模型学习显式状态跃迁。
   - 与已有工作区别：不同于通用视频级 caption 监督，监督信号来自真实观测的状态对而非合成标注。

2. **可组合的状态转换模型**：基于 Qwen-Image-Edit 微调的相对时间条件图像转换模型，支持递归预测未来视觉状态序列。
   - 与已有工作区别：引入首帧 sink 机制缓解自回归误差累积，区别于仅依赖全局任务描述的关键帧生成方法。

3. **状态条件化密集视频生成**：将预测的视觉状态及其时间位置作为软条件输入到 CogVideoX/Wan 动态模型中生成完整视频。
   - 与已有工作区别：状态作为条件而非硬约束，允许视频生成器在状态之间建模连续动态，优于直接插值方法。

4. **系统性评估**：在 Physics-IQ Verified、PhyGenBench、Pisa-Experiments、RoboTwin2.0 四个基准上验证有效性。
   - 与已有工作区别：同时评估物理一致性与操作视频保真度，覆盖从物理现象到机器人操作的完整谱系。

## 方法详解

**整体架构**（图2）：
- 预处理阶段：VLM 规划器 $q_\psi$ 接收初始图像 $x_0$ 和任务上下文 $c$，预测局部转换规范 $\{u_k\}$ 及目标时间 $\{\tau_k\}$。
- 状态转换阶段：共享转换模型 $q_\phi$ 递归应用，从当前状态 $s_{k-1}$、转换规范 $u_k$ 和elapsed time $\Delta\tau_k$ 预测下一状态 $s_k$。
- 视频生成阶段：条件动态模型 $q_\theta$ 以 $(x_0, c, \{(s_k, \tau_k)\})$ 为条件生成完整视频 rollout。

**数据标注流程**（图3a）：
- Stage I：GPT-5.5 识别主要实体、可见事件及时间结构，分解为子事件序列。
- Stage II：选择信息丰富的状态帧（如接触、变形、稳定配置），避免冗余中间帧。
- Stage III：Qwen3-VL 作为 VQA judge 验证状态帧与描述的视觉一致性，返回 binary match  verdict。

**状态转换模型训练**：
- 基于 Qwen-Image-Edit 微调，采用 flow-matching 目标（公式8）：
  $$\mathcal{L}_{\text{trans}}(\phi) = \mathbb{E}\left[\|\nu_\phi(z_\lambda^+, \lambda; s^-, s_0, u, d) - (\epsilon - z^+)\|_2^2\right]$$
- 相对时间编码：通过 RoPE 将源/目标图像 token 的时间位置设为 0 和 $d_k$，不增加可训练参数。
- 首帧 sink 机制：将 $s_0$ 作为 persistent visual reference 输入，缓解自回归 exposure bias。
- 训练数据：WISA (59,298 episodes)、NVIDIA PhysicalAI (39,873 episodes)、PhyCo/Kubric (31,109 episodes)、PICA-100K (105,085 image pairs)，共 353,840 个事件对齐状态对。

**视频动态模型训练**：
- 使用 CogVideoX-5B-I2V 和 Wan2.2-5B-TI2V 作为 backbone。
- 预测状态经 VAE 编码后投影为 spatial memory tokens，时间戳映射到 temporal grid。
- 注意力机制：$H_{\text{attn}} = [M_1; \ldots; M_K; H_{\text{video}}]$，仅从 video-token 输出读取 denoising prediction。
- 训练数据：69,344 episodes (35,114 WISA + 34,230 OpenVid) for CogVideoX，同等规模 for Wan。

## 实验与结果

**数据集**：
- Physics-IQ Verified：评估重力、碰撞、流体、材料、光照、磁性等物理现象。
- PhyGenBench：27 条物理定律（力学、光学、热学、材料）。
- Pisa-Experiments：421 个物体下落视频，评估轨迹几何精度。
- RoboTwin2.0 val500：50 个任务、5 种机械臂的 500 个 held-out episode。

**主要结果**：
- **Physics-IQ Verified**（Table 1）：
  - Ours (CogVideoX-5B): Score 41.9（+11.4 vs baseline），ST-IoU 53.5，WS-IoU 36.0。
  - Ours (Wan2.2-5B): Score 39.7（+20.3 vs baseline）。
  - 超越 Cosmos3-Super (64B) 0.8 分，证明轻量 backbone + 大型辅助模型可替代单纯扩大视频生成器规模。

- **PhyGenBench**（Table 2）：
  - CogVideoX variant: Avg 76.91（+22.12 vs baseline），超越 CausalMotion 6.70 分。
  - Wan variant: Avg 75.62（+24.79 vs baseline）。
  - 光学：80.67（最佳），热学：82.22（最佳），材料：76.67（最佳）。

- **Pisa-Experiments**（Table 3）：
  - CogVideoX variant: L2 最低（0.1293，-19.6% vs baseline），CD 最低（0.3310，-17.3%），IoU 最高（0.1753）。

- **RoboTwin2.0 val500**（Table 4）：
  - CogVideoX variant: EWMScore-15 = 60.51（+2.47 vs baseline），轨迹分数 +43.7%。
  - Wan variant: EWMScore-15 = 62.53（+3.37 vs baseline），交互质量 +18.2%，指令遵循 +31.8%。

**消融实验**（Table 5-6）：
- 移除首帧 sink：Physics-IQ 降至 40.60，Pisa-IoU 降至 0.1528。
- 移除时间条件：Physics-IQ 41.63，Pisa-IoU 0.1617。
- 移除合成数据：Physics-IQ 降至 37.91（最大降幅 4.02）。
- 预训练 Qwen-Image-Edit 未微调：PSNR 17.90 → 24.88，SSIM 0.674 → 0.863，LPIPS -57.6%，L1 -63.5%。

## 相关工作脉络

1. **物理感知生成世界模型**（Sec 2.1）：IRASim、Unified World Models 等关注视频级生成或联合动作扩散，本文聚焦显式预测单个交互结果，定位差异在于中间状态表示的显式学习。

2. **关键帧/事件引导生成**（Sec 2.2）：KeyWorld、SKIP、DCARL、RoboEnvision、CausalMotion 等方法使用稀疏关键帧或插值，本文直接监督共享转换模型于事件对齐状态对，条件接口更结构化（当前状态 + 转换规范 + 时间间隔）。

3. **视觉子目标与状态转换预测**（Sec 2.3）：MPI、SuSIE、TaKSIE 学习表示或子目标用于策略指导，本文预测跨物理现象的交互条件转换，并使用时序视觉结果条件化密集视频动态。

4. **视频生成基础模型**：CogVideoX、Wan、Cosmos 等作为 backbone，本文在其上叠加状态转换模块，改进物理一致性而不改变基础生成架构。

5. **物理后训练方法**：PISA Experiments、LaMo、PhysVid 等通过物理感知条件或 latent motion prior 提升物理现实感，本文额外引入显式状态转换学习作为中间表示。

## 局限性与未来方向

- **非物理模拟器**：框架是学习的视觉预测器而非显式物理模拟器，合理状态不保证正确轨迹、接触动态或守恒律。
- **误差累积**：训练使用观测状态，推理使用递归预测状态，长 horizon 下误差可能累积。
- **成本增加**：引入规划与图像生成额外开销，依赖模型辅助标注质量。
- **未来方向**：探索更高效的 transition model 架构、结合物理约束损失、扩展至多体交互与流体场景。

## 研究启发与可借鉴点

1. **事件对齐监督范式**：VLM-assisted annotation pipeline（事件规划 → 状态选择 → VQA 验证）可用于其他需要物理一致性的视频生成任务，如液体模拟、软体变形等。

2. **首帧 sink 机制**：在自回归状态生成中引入 persistent reference 缓解 exposure bias，可迁移至任何递归状态预测任务。

3. **状态条件化视频生成**：将离散状态作为软条件输入视频扩散模型，而非硬约束对应帧，允许生成器建模状态间连续动态，适用于需要物理一致性的可控视频生成。

4. **相对时间编码**：通过 RoPE 编码 elapsed time 无需额外参数，可迁移至任何需要时序条件化的图像/视频生成模型。

5. **分层生成架构**：先预测稀疏状态再填充密集视频的分层策略，可有效解耦物理逻辑与视觉细节，适用于复杂物理场景建模。

## 关键术语表

**Visual State（视觉状态）**：描述场景配置的图像，包含可观察的对象属性与关系（如接触、支撑、附着），不涉及速度等显式物理量。

**Transition Specification（转换规范）**：描述局部变化及其预期可见结果的文本指令，如"释放"表示对象从夹爪分离。

**Event-aligned Supervision（事件对齐监督）**：从训练视频提取的状态对 $(s^-, u, \Delta\tau, s^+)$，监督信号来自真实观测而非通用视频 caption。

**Flow Matching（流匹配）**：一种扩散模型训练目标，优化噪声预测与目标 velocity 之间的 MSE。

**First-frame Sink（首帧 sink）**：将初始图像 $s_0$ 作为 persistent visual reference 输入转换模型，缓解自回归误差累积。

**EWMScore-15**：WorldArena 的 15 指标适配版，覆盖视觉质量、运动质量、物理一致性、3D 精度与控制性。

**Timed Event Planning（定时事件规划）**：VLM 规划器从初始图像和任务上下文预测转换规范序列及其目标时间。

**Soft Conditioning（软条件）**：状态图像作为条件输入视频生成器但不强制对应帧精确匹配，允许动态模型自由生成中间状态。

## 可复现要素

- **数据集**：WISA、NVIDIA PhysicalAI、PhyCo/Kubric、PICA-100K、OpenVid-1M、RoboTwin2.0；论文未提及代码/权重开源声明，Project Page 为 https://strike-applied.github.io。
- **关键超参**：
  - Transition model：AdamW，lr=$10^{-5}$，warmup=100 steps，64×A100，5 epochs，batch size=1/GPU。
  - Dynamics model：AdamW，lr=$10^{-5}$，weight decay=0.01，32×GPU，5000 steps，batch size=2/GPU。
  - CogVideoX：49-frame clips @ 720×480；Wan：81-frame clips @ 832×480。
  - 每 backbone 预留 2,000 episodes 作内部验证。
