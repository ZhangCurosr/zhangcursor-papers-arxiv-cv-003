---
title: "RLHND-VIDEO-FOUNDATION-MODELS-AS-PHYSICALLY-GROUNDED-HAND-TR"
source: https://arxiv.org/pdf/2610.09455v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 09:51:48"
field: "具身视觉与机器人学习"
keywords: ["hand tracking", "video foundation model", "tactile estimation", "robot learning", "physically grounded", "shape caching", "anatomical pose"]
innovations: ["将Cosmos 3视频扩散模型通过clean-latent conditioning转换为确定性clip级特征提取器，并联合估计3D姿态与稠密触觉", "提出shape caching与解剖学约束姿态参数化，消除手部尺度漂移并符合人体运动学", "解耦触觉专家流+LBS-based feature spreading，在pose冻结下实现顶点级接触与力预测"]
benchmarks: ["HOT3D", "ARCTIC", "EgoDex", "OpenTouch", "PressureVisionDB", "DexYCB"]
---

# 论文速读：RLHND: VIDEO FOUNDATION MODELS AS PHYSICALLY GROUNDED HAND TRACKERS FOR ROBOT LEARNING

## 一句话总结
本文提出 RLHND，一个基于 Cosmos 3 视频扩散基础模型的手部追踪方法，通过 clean-latent conditioning 将预训练 backbone 转为确定性 clip 级特征提取器，联合估计度量级精确的三维双工手部运动与手部表面的稠密接触和力信息，显著优于现有基线并在机器人学习中验证了实用性。

## 研究问题与动机
1. **现有手部追踪方法在 3D 深度轴上存在严重抖动和漂移**：多数方法主要优化 2D 重投影或图像空间对齐，透视投影下的深度-尺度歧义导致同一视频中估计的手部大小波动可达 ~8 mm，难以支撑对运动精度敏感的机器人策略训练。
2. **现有方法缺乏物理线索（接触、力）**：人类示教视频通常不包含触觉信息，而近年的 VLA/机器人基础模型日益依赖触觉输入区分"成功抓取"与"悬停"，现有接触/力估计模型（如 HACO、HOPE）相比姿态估计仍较不成熟。
3. **基于 crop 的帧级 tracker 继承检测框抖动且时序一致性弱**：HaMeR、WiLoR、HaWoR、HaPTIC 等独立处理每帧，即使加入时序上下文，仍受限于 detector box jitter 与独立的帧级形状回归。
4. **视频扩散模型蕴含强大手部-物体交互先验但未充分迁移至追踪任务**：Wang et al. (2026)、Liu et al. (2026) 已展示预训练视频扩散模型在手部重建上的潜力，但其 feed-clean-video 的设计与预训练接口不对齐，且仍按窗口回归 shape 参数。

## 核心贡献（创新点）
1. **将 Cosmos 3 视频扩散 backbone 通过 clean-latent conditioning 接口转换为确定性 clip 级特征提取器**：不同于 ACE-Ego-Hand 的 denoising 接口（σ=0），RLHND 对齐预训练 context-frame 接口并附加空 prompt，保留视频先验的同时得到稳定时空特征。
2. **Shape caching：单视频共享一个 MANO shape 参数**：从首个充分可见 clip 估计 β 并缓存复用，彻底消除图1所示的手部大小帧间漂移（σ_shape → 0），且支持一次性预标定 β 直接注入。
3. **解剖学约束的姿态参数化（29 DoF）**：将 MANO 的 45 DoF 限制为沿解剖轴的 twist/spread/bend 分解后，冻结 PIP/DIP 不可实现的轴，仅保留 29 个可行自由度，得到更符合人体结构的姿态估计。
4. **解耦触觉专家流 + LBS-based feature spreading**：触觉流在 pose 流冻结下训练，避免触觉监督扰动姿态；通过将 16 个 bone token 经固定 MANO LBS 权重直接扩展到 778 个顶点，避免 per-vertex attention 的高计算代价。

## 方法详解
- **Clean-latent 编码器**：输入视频被 Wan2.2 VAE 编码为 clean latent z = E_VAE(C_w)，随后送入 Cosmos 3 Nano 主干；通过 LoRA（rank 64）微调 patch embedding 与生成路径，输出确定性时空特征网格 F ∈ R^{T' × h × w × D_f}。
- **Pose expert stream**：对 F 拼接 spatial positional 与 Fourier-encoded ray embeddings 后 token 化，用交替 spatial cross-attention + bidirectional temporal self-attention 解码；hand tokens X_H 由 learnable query q_H 与零初始化的 MLP(g_β(β_cache)) 构造，实现 shape conditioning。关节旋转按 anatomical axes 分解（Eq.2），在约束集 C 下回归 29 DoF（Eq.3），标签通过 damped Gauss-Newton 逆运动学投影到 C（Eq.4-5）。深度/平移由 ray-based mixed-PnP solver（ACE-Ego-Hand）恢复。
- **Tactile expert stream**：在 16 个骨骼关节上各附一个 bone token，经 LBS 扩展至 778 顶点（Eq.6），再由两个小 MLP head 分别预测 contact logit 与 force-distribution logit；单只手的总力由 softplus 读出后 softmax 分配到顶点。Stage-2 训练时 pose 流完全冻结。
- **两阶段训练**：Stage-1 在 egocentric + exocentric 手部视频混合数据集上仅训练 pose 流，损失包含旋转、shape、3D/2D 关节、平移、presence、时序平滑、ray 多项（Eq.7）；Stage-2 在含力标签的 OpenTouch、PressureVisionDB 及 mesh-derived contact 数据上训练 tactile 流，损失为加权 BCE（接触）+ 总力 MAE + 分布 CE（Eq.8），contact-positive 权重为 10。

## 实验与结果
- **运动重建**：在 HOT3D、ARCTIC ego、EgoDex（held-out）三基准全面评估。RLHND 在 HOT3D 上 MPJPE-p=13.01 mm、PA-p=6.72 mm、EPE2D-p=6.84 px、Jitter=4.38 mm/frame²，较 ACE-Ego-Hand（MPJPE-p=24.41）几乎减半；β-cache 变体 σ_shape=0。在 ARCTIC 与 EgoDex 上同样保持 SOTA。
- **触觉估计**：在 OpenTouch、DexYCB、HOT3D、ARCTIC 接触基准与 OpenTouch、PVDB 力基准上取得最高 F1/AUROC 与最低 MAE/RMSE。例如 OpenTouch 接触 F1=0.696、AUROC=0.980；OpenTouch 力 MAE=0.489 kPa、RMSE=2.508 kPa，优于 HOPE（F1=0.663、MAE=1.781）。
- **机器人学习**：在 5 种灵巧手（Sharpa Wave、WUJI v2、Shadow、Inspire RH56、ALLEX）上 retargeting，RLHND 平均 Q-err 最低（如 Shadow 9.6° vs ACE-Ego-Hand 10.8°），Jerk 同样最平滑。真实机器人 DPP 实验中（200 人类示教 + 200 遥操作），DPP+RLHND 在双工任务（塑料袋 59.4%→75.0%，组装纸巾 90.6%）提升显著，总体成功率 87.5% vs 基线 80.8%。
- **消融**：去掉 Cosmos 3 编码器（A0）在所有指标上显著退化；去掉解剖约束（A1）或 β-conditioning（A2）均恶化 MPJPE；去掉 contact-only 数据（B2）使接触 F1 骤降（如 HOT3D 0.589→0.145），说明多源接触监督不可或缺。

## 相关工作脉络
1. **HaMeR / WiLoR / HaWoR / HaPTIC**：基于单帧 crop + 检测框的 3D 手部重建，时序一致性弱、继承 detector jitter，且每帧独立估计 shape 导致尺度漂移。RLHND 以 clip 级视频为输入并引入 shape caching，显著降低 Jitter 与 σ_shape。
2. **ACE-Ego-Hand (Liu et al., 2026)**：首个将视频扩散 denoising 接口（σ=0）用于 egocentric 手部重建的工作；RLHND 改用 pre-training 原生的 conditioning-frame 接口，并对齐 empty prompt，保留更强先验且避免 scale/depth 耦合的不稳定 shape 回归。
3. **HandFlow (Xu et al., 2026)**：生成式 4D 手部恢复；同样为 clip 级，但无 shape caching 与解剖约束。RLHND 在 MPJPE-p 与 Jitter 上全面超越。
4. **HACO (Jung & Lee, 2025)**：基于 HOI 数据 mesh 距离生成 contact 标签训练接触模型；RLHND 使用统一 HOPE 标签格式并引入解耦 tactile expert，接触与力估计同步达 SOTA。
5. **HOPE (Jeon et al., 2026)**：首个统一 contact + force 于 MANO 表面的估计模型；RLHND 在其基础上以视频扩散先验与 LBS feature spreading 进一步提升稠密触觉精度。
6. **PressureVision / PressureVision++**：仅预测图像平面压力，无法提供手部表面稠密触觉；RLHND 输出 MANO 顶点级 contact + force，可直接用于基于 contact 的机器人控制。

## 局限性与未来方向
1. **β-cache 在公开 benchmark 上依赖首个可见 clip 估计，而非真实预标定**：外部标定仅在 ARCTIC 上有 GT β 可验证；在首 clip 可见性差或遮挡严重时鲁棒性未充分评估。
2. **触觉监督数据多样性有限**：力标签仅来自单一触觉手套（OpenTouch）和单一平面压力垫（PVDB），压力幅度与接触表面分布受限；EgoPressure 等新数据集因传感器标定与表面参数化不一致而尚未接入。
3. **从手部标注到机器人动作的映射仍依赖固定 IK 与启发式 contact offset**：未探索力的大小/方向对灵巧操作的直接利用，也未处理不同 kinematics/compliance 的跨 embodiment 适配。
4. **离线 clip 处理与 7.8B 参数 backbone**：推理非实时（~144 FPS，单 clip 一次前向），难以直接部署于在线遥操作/闭环控制；与 crop-based 方法（>500 FPS）相比延迟更高。

## 研究启发与可借鉴点
1. **Clean-latent conditioning 接口优于 denoising-at-σ=0**：对于将视频扩散模型用作确定性编码器，直接使用预训练 context-frame 接口并配合空 prompt 能获得更稳定的时空表征，值得推广到其他具身视觉任务。
2. **Shape caching 作为"时间一致性正则"的简洁范式**：对任意需要跨帧保持物体/身体几何一致性的任务（如手部、人脸、工具），在首个可靠帧估计 shape/SSR 后缓存复用，可直接消除尺度漂移而不需复杂时序正则。
3. **解剖学约束 + IK 投影重构标签**：将 45 DoF MANO 标签投影到 29 DoF 解剖可行集，以 damped Gauss-Newton 三步求解成本低且精度保持（中位数误差 ~0.5 mm），可为其他受约束人体/手部建模提供标签预处理方案。
4. **LBS-based feature spreading 替代 per-vertex attention**：在 mesh-based 输出（顶点/像素级预测）中，用固定 skinned weights 将关节/骨骼级特征广播至顶点，避免 O(|V|²) 注意力开销，适用于任何基于模板 mesh 的稠密预测头设计。
5. **解耦多任务专家（pose / tactile）两阶段冻结训练**：当某类监督信号稀缺且可能干扰主任务时，stage-2 冻结主分支仅训专家分支，可有效防止负迁移；这一范式可迁移到多模态输出（姿态+接触+语义）联合学习。

## 关键术语表
- **RLHND**：本文提出的基于视频基础模型的手部追踪器，联合估计 3D 姿态与稠密触觉（接触+力）。
- **Cosmos 3 Nano**：NVIDIA 发布的 multi-modal 视频扩散基础模型；本文以其 Nano 版作为确定性时空特征编码器。
- **Clean-latent conditioning**：利用 Cosmos 3 预训练的 context-frame 输入接口（而非 denoising 接口）传入无噪声 latent，使模型对齐预训练分布以提取稳定特征。
- **Shape caching (β-cache)**：在视频内固定使用从首个充分可见 clip 估计出的 MANO shape 参数，消除帧间手部尺度漂移。
- **Anatomical pose parameterization**：将 MANO 的 45 DoF 旋转分解为 twist/spread/bend 轴并冻结不可实现轴，得到 29 DoF 的解剖可行姿态表示。
- **LBS-based feature spreading**：利用 MANO 固定的线性blend-skinning 权重将 16 个 bone token 特征扩展到 778 个顶点，避免 per-vertex attention。
- **Dexterous Point Policy (DPP)**：Kim et al. (2026a) 提出的基于关键点的手部策略，本文将其与 RLHND 结合在真实 RB-Y1 双工机器人上验证。
- **Jitter / Q-err**：Jitter 衡量关节位置的二阶时间差分（加速度噪声，mm/frame²）；Q-err 为 retargeting 后关节指令与 oracle 指令的平均绝对偏差（°）。

## 可复现要素
- **数据集**：训练使用公开 egocentric/exocentric 手部视频（ARCTIC、HOT3D、H2O、OakInk2、DexYCB、HO3D、HRDexDB 等）及触觉数据集（OpenTouch、PressureVisionDB）；全部为公开数据。
- **代码/权重**：论文声明将开源代码与预训练权重（https://seungjun-moon.github.io/rlhnd/）。
- **关键超参**：clip 长度 T=81 帧、输入高 480；decoder D=384、4 层、8 heads、4 register tokens；LoRA rank=64；AdamW β₁=0.9, β₂=0.95, weight decay=0.01；stage-1 20k steps on 8×H100，batch=32；stage-2 20k steps on 1×A100，batch=2；contact BCE 正类权重=10；λ_c=λ_F=λ_π=1。
