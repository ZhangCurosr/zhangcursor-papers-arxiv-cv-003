---
title: "INFINIHAND-STREAMING-WORLD-SPACE-HAND-MOTION-ESTIMATION-FROM"
source: https://arxiv.org/pdf/2609.35743v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 04:07:51"
field: "Ego-centric 3D 手运动估计与世界空间重建"
keywords: ["world-space hand estimation", "ego-centric video", "streaming reconstruction", "MANO", "SLAM integration", "feed-forward framework"]
innovations: ["端到端流式前馈统一手定位/MANO/相机轨迹", "GCA持久时空记忆+手中心双路特征融合", "稀疏BA关键帧池校正长序列漂移"]
benchmarks: ["ARCTIC", "HOT3D", "EgoDex", "HOI4D"]
---

# 论文速读：INFINIHAND: STREAMING WORLD-SPACE HAND MOTION ESTIMATION FROM EGOCENTRIC VIDEO

## 一句话总结
论文提出了 InfiniHand，一个流式前馈框架，直接从无标定ego-centric视频联合估计手部位置、MANO参数和相机轨迹，将手重建与SLAM统一为一个端到端架构，避免了传统级联方案中误差累积和计算冗余的问题。

## 研究问题与动机
- **现有方法的级联缺陷**：HaWoR 等方法依赖独立的 hand detector + camera-space hand network + DROID-SLAM + Metric3D 等组件级联，各组件间的检测抖动、相机跟踪漂移和尺度不一致会沿管线传播，损害世界空间重建精度。
- **泛化瓶颈**：WiLoR、HaWoR 等未在有大规模多样化的无约束 ego-centric 视频上进行充分预训练，面对快速相机运动和复杂手物交互时表现明显退化。
- **缺乏统一的 streaming 世界空间估计**：既有方法要么只做 camera-space 单帧重建，要么先做单独 hand 重建再用外部 SLAM 转换到 world-space，二者并未联合优化。
- **下游应用驱动**：ego-centric 视频中蕴含着大量人类示范，将其自动转换为世界坐标系中的 3D 手运动标注，对具身智能的 human-to-robot 动作迁移和 in-context imitation 至关重要。

## 核心贡献（创新点）
1. **首个端到端 streaming 前馈的世界空间手运动估计框架**，将手部定位、MANO 参数预测与相机轨迹估计统一在单一架构中；与 ViDiHand/HaWoR 的本质区别在于消除了级联管线、显式耦合了全局相机运动与局部手几何。
2. **持久时空记忆（Geometric Context Attention）耦合手中心视觉特征**，使 streaming 过程中历史几何约束得以持续保留；区别于 LoGeR/LingBot-Map 纯场景重建，本文专门设计了 hand-centered 的几何+外观双路融合机制。
3. **Two-stage progressive training**：Stage I 先学 robust 的 camera-space 手 prior，Stage II 再扩展到 streaming world-space 联合优化；与 Dyn-HaMR 的 multi-stage 不同，本文两阶段共享 backbone 并在 streaming 窗口间持续传递状态。
4. **稀疏 Bundle Adjustment（基于 DROID-SLAM 后端）** 作为 streaming 输出的后优化模块，选择性维护 keyframe pool 以校正长序列相机漂移；与 MASt3R-SLAM/VGGT-SLAM 的密集优化相比，本文保持在线 streaming 吞吐的同时仅做稀疏关键帧修正。
5. **聚合约 5,000 小时公开 ego-centric 视频构建标准化训练语料**（ARCTIC、HOT3D、EgoDex 等九数据集），并设计清洗管线过滤 30–40% 噪声标注；这是目前规模最大、覆盖最广的 hand 世界空间估计训练集之一。

## 方法详解
### 整体架构
输入 RGB 序列 $\mathcal{I} = \{\mathbf{I}_t\}_{t=1}^T$，模型输出：相机参数 $\hat{P}_t = (\hat{\mathbf{K}}_t, \hat{\mathbf{R}}_t, \hat{\mathbf{u}}_t)$；每只手 $s \in \{L, R\}$ 的 MANO pose $\hat{\mathbf{e}}_t^s$、shape $\hat{\beta}_t^s$、全局朝向 $\hat{\Phi}_t^{w,s}$、根平移 $\hat{\mathbf{t}}_t^{w,s}$。

### Stage I：Camera-space 手定位与重建
- **手定位**：基于 LingBot-Map backbone 提取多层特征 $\{\mathbf{F}_t^{(\ell)}\}$，经 DPT decoder 后接 mask head 得到双手掩码：
$$\mathbf{D}_t = \mathcal{D}_{\text{DPT}}(\{\mathbf{F}_t^{(\ell)}\}), \quad [\hat{\mathbf{M}}_t^L, \hat{\mathbf{M}}_t^R] = \sigma(\mathcal{H}_{\text{mask}}(\mathbf{D}_t))$$
损失：$\mathcal{L}_{\text{mask}} = \lambda_{\text{BCE}}\mathcal{L}_{\text{BCE}} + \lambda_{\text{Dice}}\mathcal{L}_{\text{Dice}}$
- **手重建**：对每个 hand box $\mathbf{b}_t^s$，分别提取几何特征（via adapter $\mathcal{A}$）与外观特征（via WiLoR encoder $\mathcal{E}_{\text{WiLoR}}$），拼接后经 $1\times1$ conv 融合：
$$\mathbf{Z}_t^s = \text{Conv}_{1\times1}([\mathcal{A}(\mathbf{F}_t, \mathbf{b}_t^s);\; \mathcal{E}_{\text{WiLoR}}(\text{Crop}(\mathbf{I}_t, \mathbf{b}_t^s))])$$
MANO head 回归 $(\hat{\Theta}, \hat{\beta}, \hat{\Phi}^h, \hat{\ell}_z)$，辅助 2D landmark head 回归 $\hat{\mathbf{p}}$。
- **横向平移恢复（LSP）**：固定深度 $\hat{t}_z^h$，通过可微分最小二乘投影对齐 3D joints 与 2D landmarks：
$$(\hat{t}_x^h, \hat{t}_y^h) = \arg\min_{a,b} \sum_{j\in\mathcal{V}} \|\pi(\bar{\mathbf{J}}_j^h + [a,b,\hat{t}_z^h]^\top; \mathbf{K}^h) - \hat{\mathbf{p}}_j^{\text{pix}}\|_2^2$$
- **坐标转换**：将 hand-frame 参数旋转到原 camera frame：
$$\text{Rot}(\hat{\Phi}^c) = \mathbf{R}_{h\to c}^\top \text{Rot}(\hat{\Phi}^h), \quad \hat{\mathbf{t}}^c = \mathbf{R}_{h\to c}^\top \hat{\mathbf{t}}^h$$
- **Stage I 总损失**：$\mathcal{L}_{\text{MANO}}$（含 orient/pose/shape/trans/joints/reproj 六项）+ $\mathcal{L}_{\text{2D}}$ + $\mathcal{L}_{\text{mask}}$。

### Stage II：Streaming World-space 联合估计
- **Streaming 布局**：clip 由 4 个 anchor 帧 + 两个连续 16 帧窗口构成（$4 + 16\times2$ 共 36 帧），窗口间 streaming state 持续，clip 间重置。
- **GCA 记忆机制**：状态 $\mathcal{M}_k = (\mathcal{M}_{\text{anchor}}, \mathcal{W}_k, \mathcal{T}_k)$，含锚点特征、近期稠密窗口特征、压缩后的轨迹 memory tokens。
$$({\mathbf{F}}_k, \mathcal{M}_k) = \mathcal{B}_{\text{GCA}}(\mathbf{I}_k, \mathcal{M}_{k-1})$$
- **多任务预测**：${\hat{P}}_t = \mathcal{H}_{\text{cam}}(\mathbf{F}_k)_t$，$\hat{\mathbf{M}}_t = \mathcal{H}_{\text{loc}}(\mathbf{F}_k)_t$，$\hat{y}_t^s = \mathcal{H}_{\text{rec}}(\mathbf{I}_t, \mathbf{F}_t, \mathbf{b}_t^s)$。
- **World-space 变换**：${\text{Rot}}(\hat{\Phi}_t^{w,s}) = \hat{\mathbf{R}}_t {\text{Rot}}(\hat{\Phi}_t^{c,s})$，$\hat{\mathbf{t}}_t^{w,s} = \hat{\mathbf{R}}_t \hat{\mathbf{t}}_t^{c,s} + \hat{\mathbf{u}}_t$。
- **Streaming 总损失**：
$$\mathcal{L}_{\text{joint}} = \lambda_{\text{cam}}\mathcal{L}_{\text{camera}} + \lambda_w \mathcal{L}_{\text{joints}}^w + \lambda_{\text{temp}}\mathcal{L}_{\text{temp}} + \mathcal{L}_{\text{MANO}} + \lambda_{2\text{D}}\mathcal{L}_{2\text{D}} + \mathcal{L}_{\text{mask}}$$
其中 $\mathcal{L}_{\text{camera}}$ 含绝对位姿（平移/四元数/FOV）与相对位姿两项；$\mathcal{L}_{\text{joints}}^w$ 用 smooth-$\mathcal{L}_1$ 监督世界空间 joints。

### 稀疏 Bundle Adjustment（BA）
- 维护 binary keyframe pool $\mathcal{P}=\{(f_i, b_i)\}$，超阈值后每隔 $K$ 选一次最老条目，交由 DROID-SLAM 后端精化。
- 被选中条目翻转 $b_i$：保留的 $0\to1$ 继续留存，已保留的 $1\to0$ 参与本次 BA 后剔除，实现"旧观测被回顾但不永久占用内存"。
- BA 仅优化相机姿态，手预测器保持纯前馈。

## 实验与结果
### 数据集
- 训练：聚合 ARCTIC、HOT3D、EgoDex、DexYCB、HO3D、H2O-3D、EgoVerse、EgoLive、Xperience-10M 共约 **5,000 小时**，采样 10 FPS，清洗丢弃 30–40% 噪声样本。
- 测试（camera-space）：**ARCTIC**（34 场景）、**HOT3D**（10 场景）、**EgoDex**（100 场景）、**HOI4D**（166 场景）；in-the-wild：Ego4D、Xperience-10M。
- 测试（world-space）：ARCTIC、HOT3D、EgoDex。

### 评估指标
- 检测：FAcc、Recall、F1
- Camera-space：MP-p、PA-p、EPE-p、GO-p、CT-p（均含漏检惩罚）
- World-space：PA-MPJPE、W-MPJPE、WA-MPJPE

### 核心结果（ARCTIC camera-space）
| 方法 | PA-p (mm) ↓ |
|------|-------------|
| ViDiHand | 9.82 |
| **InfiniHand (Ours)** | **7.72** |
→ 较 ViDiHand 降低 **21.4%**，为 SOTA。

### 核心结果（EgoDex camera-space）
| 方法 | PA-p (mm) ↓ |
|------|-------------|
| ViDiHand | 17.22 |
| **InfiniHand (Ours)** | **7.29** |
→ 较 ViDiHand 降低 **57.7%**。

### World-space（W-MPJPE，mm）
| 方法 | ARCTIC | HOT3D | EgoDex |
|------|--------|-------|--------|
| WiLoR-SLAM | 65.86 | 106.77 | 96.02 |
| Dyn-HaMR | 114.10 | 296.72 | 78.41 |
| HaWoR | 95.39 | 93.70 | 103.85 |
| **InfiniHand (Ours)** | **59.21** | **87.37** | **28.77** |
→ 较 WiLoR-SLAM（ARCTIC）降低 10.1%，较 Dyn-HaMR（EgoDex）降低 **63.3%**。

### 效率
- **11.19 FPS**，较 HaWoR（5.48 FPS）快 **2.04×**，较 Dyn-HaMR（0.81 FPS）快 13.8×。
- Pipeline 双 worker 重叠执行：worker1 处理下一窗口（4 anchor + 16 帧），worker2 同时解码上一窗口的手/相机预测；worker3 后台跑稀疏 BA。

### Ablation（ARCTIC）
- w/o WiLoR 外观特征：MP-p 从 17.09 → 33.31 mm（近翻倍）
- w/o LSP 横向恢复：PA-p 从 7.72 → 8.02 mm
- w/o BA：W-MPJPE 从 59.21 → 80.48 mm（camera-space 不变）
- w/o Stage II：W-MPJPE 从 59.21 → 185.76 mm（世界空间严重退化）
- w/o Mask Head（改用 HaWoR mask）：PA-p 从 7.72 → 25.50 mm

## 相关工作脉络
1. **WiLoR**（Potamias et al., 2024）：end-to-end 单帧手定位+重建，但未估计相机轨迹/世界空间；InfiniHand 在此基础上增加了相机 heads 与 streaming 记忆。
2. **HaWoR**（Zhang et al., 2025）：级联 hand detector + camera-space network + DROID-SLAM；InfiniHand 将其替换为统一前馈 streaming 架构。
3. **Dyn-HaMR**（Yu et al., 2025）：多阶段 hand-camera 联合优化，但依赖离线多阶段而非 streaming；InfiniHand 采用 $4+16\times2$ 流式窗口。
4. **ViDiHand**（Wang et al., 2026）：视频 diffusion prior + hand-overlay supervision；InfiniHand 明确建模世界空间并与 SLAM 联合。
5. **LingBot-Map**（Chen et al., 2026）：GCA 记忆的 streaming 3D 场景重建 backbone；InfiniHand 以此为基础，加入 hand 专用 head 与 hand-centered crop 机制。
6. **DROID-SLAM**（Teed & Deng, 2021）：dense learned SLAM；InfiniHand 将其作为稀疏 BA 后端而非主 pipeline。
7. **MASt3R-SLAM / VGGT-SLAM**：learning-augmented 密集 SLAM；InfiniHand 不走显式 cross-view 优化路线，保持纯前馈 streaming。

## 局限性与未来方向
- **尺度恢复依赖外部对齐**：LingBot-Map backbone 本身无 metric scale，需辅助 post-processing alignment，其误差会传播到手位置与世界轨迹。
- **数据覆盖不足**：高质量含复杂大幅双手电机和相机动力学的 ego-centric 数据集仍然稀缺，限制了 rapid viewpoint change / severe occlusion / intermittent visibility 场景的泛化。
- **监督的时间尺度漂移**：monocular 标注自带的 time-varying scale drift 会污染长期轨迹训练目标，导致跨帧不一致。
- **未来方向**：（1）引入 metric depth prior 或 stereo 约束实现内禀尺度；（2）扩展至完整 body + object 联合估计；（3）结合 robot 端真实 trajectory 进行 sim-to-real fine-tuning。

## 研究启发与可借鉴点
1. **GCA streaming memory + hand-centered crop 双路融合**（几何 + 外观）可直接迁移到 finger-level 精细重建或 finger-object contact 估计。
2. **Two-stage progressive training**（先 camera-space 后 world-space）策略适用于任何"局部 → 全局"联合学习问题，如 hand+face 同步估计、全身估计等。
3. **LSP 横向平移可微分恢复**（固定深度求解 lateral translation）是解决"depth-only monocular ambiguity"的简洁技巧，可复用于其他单目 3D 定位任务。
4. **Sparse BA on binary keyframe pool** 实现了"精确校正"与"在线吞吐"的折衷，思路可移植到其他 streaming 3D 任务（footage SLAM、object trajectory tracking）。
5. **5,000h 大规模 ego-centric 语料 + 自动清洗管线**（Sapiens2 keypoint 对齐 + 90th percentile 阈值过滤）提供了可复用的数据工程范式，对其他多源视觉预训练有参考价值。

## 关键术语表
- **InfiniHand**：端到端流式前馈框架，联合估计 ego-centric 视频中的世界空间手运动与相机轨迹。
- **MANO**：参数化人手模型（pose 15×3, shape 10, 旋转轴角），广泛用于 hand mesh regression。
- **GCA（Geometric Context Attention）**：LingBot-Map 提出的 streaming 记忆机制，维护 anchor 特征、稠密窗口、压缩轨迹 token 三类状态。
- **PA-p（Procrustes-aligned error with penalty）**：Procrustes 对齐后 3D joint 误差，计入漏检惩罚（mm）。
- **W-MPJPE**：World-space Mean Per-Joint Position Error，未做额外对齐的世界空间 joint 误差（mm）。
- **LSP（Least-Squares Projection）**：固定深度下通过可微分投影最小二乘求解手 frame 内的横向平移。
- **DROID-SLAM**：基于 learned recurrent updates 和 differentiable dense BA 的 monocular SLAM 系统。
- **Sparse Bundle Adjustment**：仅对选出的关键帧执行 BA 后优化，平衡精度与 streaming 延迟。

## 可复现要素
- **数据集**：ARCTIC、HOT3D、EgoDex、DexYCB、HO3D、H2O-3D、EgoVerse、EgoLive、Xperience-10M（均为公开数据集）；自建清洗管线后合计约 5,000 小时。
- **代码/权重**：项目页 https://infinihand.github.io/，论文未明确声明开源仓库与权重地址；需访问项目页确认。
- **关键超参**：Stage I 300k steps，Stage II 100k steps；batch size 64；AdamW + weight decay 0.01；图像尺寸 378×518；hand crop 256×256（box 放大 1.5×）；learning rate Stage I mask/joint $10^{-4}$、MANO $5\times10^{-5}$；Stage II camera $3\times10^{-5}$、其余 $10^{-5}$；梯度范数 clip 1.0；linear warmup + cosine annealing。
