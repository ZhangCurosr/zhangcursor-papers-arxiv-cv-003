---
title: "RLHND-VIDEO-FOUNDATION-MODELS-AS-PHYSICALLY-GROUNDED-HAND-TR"
source: https://arxiv.org/pdf/2610.09455v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 22:52:38"
field: "机器人视觉与触觉感知"
keywords: ["hand tracking", "video foundation model", "tactile estimation", "robot learning from human videos", "MANO", "physics-aware reconstruction"]
innovations: ["将 Cosmos 3 视频扩散 backbone 通过 clean-latent 接口转为确定性特征编码器并联合估计姿态与触觉", "Shape caching 消除 monocular 手重建中的尺度-深度歧义导致的时序手型漂移", "解剖学约束 29-DoF 姿态参数化 + 解耦触觉专家 + LBS-based vertex spreading"]
benchmarks: ["HOT3D", "ARCTIC", "EgoDex", "OpenTouch", "PressureVisionDB", "DexYCB"]
---

# 论文速读：RLHND: VIDEO FOUNDATION MODELS AS PHYSICALLY GROUNDED HAND TRACKERS FOR ROBOT LEARNING

## 一句话总结
RLHND 将预训练的 Cosmos 3 视频扩散模型转化为确定性视频特征编码器，联合估计单目第一人称视频中双手的 metric 级 3D 姿态和手部表面密集接触/力分布，为机器人学习提供物理一致的轨迹与触觉信号。

## 研究问题与动机
- **现有手追踪方法 3D 深度方向抖动严重**：基于裁帧的 tracker（如 HaMeR、WiLoR 等）对每帧独立回归，导致手腕深度在 850 帧序列中出现严重振荡；同一视频中估计的手部尺寸标准差可达 ~8 mm，无法用于对精度敏感的机器人动作标注。
- **尺度-深度歧义导致形状估计不稳定**：透视投影下深度误差会被吸收进 MANO shape 参数，使 clip-level 方法也会在窗口间产生手型漂移。
- **人类数据集缺乏物理触觉信息**：现有视觉-语言-动作模型需要接触/力信号区分"抓取"与"悬停"，但大多数人类 egocentric 视频只有 RGB，压力/触觉数据需穿戴手套或压力板，难以规模化获取。
- **现有触觉估计方法仍不充分**：HACO 等仅利用几何距离预测接触，无法监督力；HOPE 虽统一了接触与力标签，但未结合视频 foundation model 的时空先验，性能仍有提升空间。

## 核心贡献（创新点）
1. **首次将视频 foundation model 的 clean-latent 条件接口用于确定性价手重建**：与 ACE-Ego-Hand 直接通过去噪接口喂干净帧不同，RLHND 利用 Cosmos 3 原生 conditioning-frame 接口，保持与预训练一致，提取更丰富的时空表示并联合估计 contact/force。
2. **Shape Caching：整个视频共享单一 shape 参数**：从首个清晰可见 clip 估计 β 并全局缓存，消除手型漂移（σ_shape 降至 0），同时支持一次性预标定 β 的直接注入，解决尺度-深度歧义。
3. **解剖学约束的 29-DoF 姿态参数化**：沿 anatomical twist/spread/bend 轴分解旋转，冻结PIP/DIP 关节的 twist 和 spread，仅回归 29 DoF（MANO 原为 45 DoF），通过阻尼 Gauss-Newton IK 将原始标签投影到可行流形。
4. **解耦触觉专家流 + LBS-based feature spreading**：stage-2 冻结姿态流单独训练触觉专家，避免触觉监督扰动姿态；通过 MANO 固定 LBS 权重将 16 个骨级 token 展开至 778 个顶点，免去逐顶点注意力的高计算代价。

## 方法详解
**架构总览**：输入为 W 个 clip（每 clip T=81 帧），依次经三个阶段处理：
- **Clean-latent 编码器**：Wan2.2 VAE 将 clip 编码为 clean latent z，经微调的 patch embedding + LoRA（rank=64）的 Cosmos 3 Nano 主干输出确定性特征网格 F ∈ R^{T'×h×w×D_f}。与 Liu et al. 设置 σ=0 走去噪接口的做法不同，RLHND 走 conditioning-frame 接口，并输入空 prompt。
- **Pose Expert Stream**：特征网格 token 化后加入空间位置编码与 Fourier-encoded ray embedding，经交替 spatial cross-attention + bidirectional temporal self-attention 解码。构造两个 hand token X_H^w = q_H + g_β(β_cache)，其中 g_β 零初始化，β_cache 取自首个可见度超阈值 T_v 的 clip；随后通过解剖学约束的 29-DoF 参数化（式 2-5）预测旋转，shape head 预测 β（仅对未 teacher-force 样本计算 L_β），ray head 预测像素视线用于混合 PnP 求解器恢复 metric 相机空间平移 τ_t。
- **Tactile Expert Stream**：每个手 16 个骨 token（X_B ∈ R^{2|B|×D}）表征骨架特征，经 LBS 展开至 778 顶点（式 6）：h_{t,v} = Σ_j W_lbs,vj [X_B^L]_{t,j} + m_v。两个小 MLP 分别输出 per-vertex contact logit 和 force-distribution logit，总力 F̂_t 经 softplus 从 expert token 读出，各顶点力 f̂_{t,v} = F̂_t · softmax_v(·)。

**损失函数**：
- Stage-1（姿态流）：L_pose = λ_rot L_rot + λ_β L_β + λ_3D L_3D + λ_2D L_2D + λ_τ L_τ + λ_pres L_pres + λ_tmp L_tmp + λ_ray L_ray，含 geodesic+Frobenius 旋转损失、L1 shape 损失、腕相对/相机帧 3D 关节损失、soft-argmax 锚点 + 重投影 2D 损失、平移 L1、存在/可见性 BCE、二阶时间差分平滑、ray cosine distance。
- Stage-2（触觉流，姿态冻结）：L_tactile = λ_c BCE_w(ĉ, c) + λ_F |Σ_v f̂_t,v - Σ_v f_t,v|/|V| + λ_π CE(f̂_t/Σf̂, f_t/Σf)，正样本加权 BCE + 总力 L1 + 分布 CE。

**β-cache 推理协议**：逐 clip 解码，第一个可见度 >T_v 的 clip 的 β 成为全局缓存，之前等待的 clip 用缓存形状重新解码；若无满足条件的 clip 则回退至 per-window 估计。

## 实验与结果
**数据集与评估**：姿态评估在 HOT3D、ARCTIC（ego split）和 EgoDex（zero-shot，训练未见）；触觉评估在 OpenTouch、PressureVisionDB（PVDB）的 HOPE test split，以及 DexYCB、HOT3D、ARCTIC 的 mesh-derived contact 标签。所有 baseline 使用官方权重与推理代码在同一批帧上重跑。

**姿态估计 SOTA**（Table 1）：
- HOT3D：RLHND MPJPE-p = 13.01 mm（ACE-Ego-Hand 24.41），F_Acc = 0.996，Jitter = 4.38 mm/frame²，σ_shape = 1.30（β-cache 后为 0.00）。
- ARCTIC ego：MPJPE-p = 13.36（ACE-Ego-Hand 14.86），EPE2D-p = 4.19（3.52）。
- EgoDex（zero-shot）：MPJPE-p = 19.90（ACE-Ego-Hand 20.95），F_Acc = 1.000。
- RLHND 在所有三个数据集的所有指标上全面领先。

**触觉估计 SOTA**（Table 2）：
- Contact：OpenTouch F1=0.696/AUROC=0.980，DexYCB F1=0.572/AUROC=0.915，HOT3D F1=0.589/AUROC=0.959，ARCTIC F1=0.602/AUROC=0.935，四项均第一。
- Force：OpenTouch MAE=0.489 kPa / RMSE=2.508；PVDB MAE=0.274 / RMSE=2.066，competitive。

**机器人学习验证**（Table 3-4）：
- Retargeting 到五款灵巧手（Sharpa Wave、WUJI v2、Shadow、Inspire RH56、ALLEX），RLHND 的 Q-err 最低（如 Shadow 9.6° vs. ACE-Ego-Hand 10.8°），Jerk 最优（Shadow 0.19 vs. 0.21）。
- DPP + RLHND 在真实 RB-Y1 双臂机器人上，复杂双手指令任务（塑料袋翻转、纸巾装配）成功率显著提升：Bimanual Avg 从 64.1% 提升至 87.5%，All↑ 从 80.8% 提升至 87.5%。

**消融**（Table 5）：
- 移除 Cosmos 3 主干（A0）：MPJPE-p 从 13.36 → 16.62（ARCTIC），encoder 是最大贡献来源。
- 移除解剖约束（A1）：MPJPE-p 从 13.36 → 14.26。
- 移除 β-conditioning（A2）：MPJPE-p 从 13.36 → 14.87。
- 移除 LBS spread（B1）：DexYCB F1 从 0.572 → 0.047（灾难性下降）。
- 移除 contact-only 数据（B2）：DexYCB F1 从 0.572 → 0.532，HOT3D F1 从 0.589 → 0.548，contact 显著下降。

## 相关工作脉络
- **HaMeR / WiLoR / HaWoR / HaPTIC**：单帧/裁帧 transformer，独立回归每帧 MANO 参数，时序一致性弱，深度抖动严重（Jitter 15-51 mm/frame² vs. RLHND 4.38）。RLHND 以 clip 级视频输入从根本上改善时序稳定性。
- **HandFlow**：基于 flow matching 的全生成 4D 手恢复，虽为 clip-level，但 per-window 估计 shape 仍导致 σ_shape 达 3.68（HOT3D），RLHND 的 β-cache 将其降至 0。
- **ACE-Ego-Hand（Liu et al., 2026）**：最近的视频扩散 backbone 手追踪工作，通过去噪接口（σ=0）利用 Wan2.2，但不约束 pose DoF、不缓存 shape、无触觉分支。RLHND 在此基础上增加解剖约束、shape caching 和解耦触觉专家，形成更全面的能力。
- **HACO（Jung & Lee, 2025）**：利用 HOI 数据集中手-物距离生成 contact pseudo-label，仅预测接触无力的监督。RLHND 使用真实力标签（OpenTouch 手套、PVDB 压力板），且通过解耦专家避免姿态被稀疏触觉信号干扰。
- **HOPE（Jeon et al., 2026）**：统一 contact + force 标签并在 MANO 表面训练触觉模型，但采用 per-window 裁剪输入，未充分利用视频时空先验。RLHND 的 clip-level 特征提取 + LBS spreading 在接触 F1（HOT3D 0.589 vs. HOPE 0.197）和力 MAE（OpenTouch 0.489 vs. 1.781）上大幅超越。
- **Dexterous Point Policy（DPP, Kim et al., 2026a）**：以关键点+指尖接触标签训练灵巧手策略的 baseline；RLHND 直接替代其手部 tracker 并提供自动 contact 标注，省去人工标注，证明 end-to-end 可用性。

## 局限性与未来方向
- **β-cache 在公开 benchmark 上无法利用预标定 shape**：因缺乏 per-subject 校准数据，只能从首个清晰 clip 自动估计，真实场景中首次校准的收益仅在 ARCTIC A4 消融中演示，鲁棒性待验证。
- **触觉监督数据多样性有限**：力标签仅来自单一触觉手套（OpenTouch）和单一压力板（PVDB），压力范围和接触表面覆盖不足；EgoPressure 等新数据集因传感器标定和表面参数化不一致难以直接融合。
- **从手标签到机器人动作的映射未完全解决**：retargeting 使用固定 IK 求解器（锁定 abduction joints）和启发式接触偏移，未探索 contact geometry 差异、柔顺性建模及力反馈的充分利用。
- **离线 clip 处理，非实时**：Cosmos 3 Nano 主干（7.8B 参数）推理约 6.9 ms/frame，虽优于裁帧方法（含 detector），但远达不到实时控制需求（ACE-Ego-Hand 仅 1.8 ms/frame）。
- **公共数据集的多样性有限**：伦理声明指出性能可能随受试者、手型、肤色变化而波动，未在评估集中充分覆盖。

## 研究启发与可借鉴点
1. **Video diffusion backbone 作为确定性特征提取器的 clean-latent 接口设计**：利用预训练模型的 conditioning-frame 而非 denoising interface，既保留时空先验又避免训练-推理接口不一致，可作为 video foundation model 微调的通用范式。
2. **Shape caching 解决尺度-深度歧义的简洁工程技巧**：用一个 clip 估计 shape 并全局复用，零额外计算成本即可消除手型漂移；此思路可迁移至其他 monocular 人体/手部重建任务。
3. **解耦多任务训练的冻结策略**：姿态流与触觉流分 stage 训练且 stage-2 冻结 stage-1，避免触觉监督（数据稀疏且噪声大）污染姿态表征，值得在多任务视觉-物理联合学习中推广。
4. **LBS-based feature spreading 替代 per-vertex attention**：用固定 MANO 皮肤权重将低维骨级特征展开到 778 顶点，以极低计算代价获得密集顶点表征，可推广至任意 SMPL/SMPL-X 类 mesh 模型的密集预测任务。
5. **Anatomical DoF 约束提升物理合理性**：通过 IK 将 45-DoF MANO 标签投影到 29-DoF 解剖可行流形，使网络输出天然符合生物力学，可减少 retargeting 到机器人时的反向运动学失败率。

## 关键术语表
**Cosmos 3 Nano**：NVIDIA 发布的大规模视频扩散 foundation model 的轻量版本，RLHND 以其 VAE + 主干作为确定性视频特征编码器。
**MANO**：Embodied Hands 参数化模型，用 10 维 shape 参数 β 和 45-DoF 旋转参数 θ 表示单只手，含 778 个顶点和 16 个关节。
**β-cache**：将从首个可见 clip 估计的 MANO shape 参数缓存并在整段视频中复用的机制，消除 hand-size 时序抖动。
**Anatomical twist-spread-bend parameterization**：沿手指解剖轴（twist/spread/bend）分解关节旋转，冻结不可行轴（PIP/DIP 的 twist 和 spread），将 DoF 从 45 降至 29。
**LBS（Linear Blend Skinning）**：MANO 提供的固定权重矩阵 W_lbs ∈ R^{778×16}，将 16 个骨级特征线性插值展开至 778 个顶点。
**Clean-latent conditioning**：将 VAE 编码后的干净 latent 送入 diffusion backbone 的条件接口（而非去噪接口），使预训练权重与微调输入分布保持一致。
**Dexterous Point Policy（DPP）**：以关键点位移为动作空间的灵巧手策略，RLHND 的输出可直接替代其原始 tracker 并自动提供 contact 标签。
**Mixed PnP solver**：结合 per-frame depth head 直接预测 wrist 深度 z 与 in-plane 最小二乘闭式解的手-相机平移恢复方法。

## 可复现要素
- **数据集**：训练数据全部来自公开数据集（HOT3D、ARCTIC、DexYCB、HO3D、H2O、OakInk2、HRDexDB、OpenTouch、PressureVisionDB、EgoDex）；评估均在公开 benchmark 上进行。论文未提及自建数据集。
- **代码**：论文声明代码将在 https://seungjun-moon.github.io/rlhnd/ 公开（截至论文发表时未上线）。
- **权重**：预训练权重随代码一起发布。
- **关键超参**：clip 长度 T=81，输入高度 480px；decoder D=384、4 层、8 头、4 register tokens；LoRA rank=64；AdamW（β1=0.9, β2=0.95, wd=0.01）；warmup 200 步 + cosine decay；stage-1 学习率 decoder/ray head/tactile 2e-4、LoRA 1e-4、patch embedding 2e-5，8×H100、eff. batch=32、20k steps；stage-2 单 A100、batch=2、20k steps，λ_c=λ_F=λ_π=1，BCE_w 正样本权重 10。
