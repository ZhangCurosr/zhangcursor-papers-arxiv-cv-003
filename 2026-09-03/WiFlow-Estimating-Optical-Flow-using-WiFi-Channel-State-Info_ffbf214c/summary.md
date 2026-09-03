---
title: "WiFlow-Estimating-Optical-Flow-using-WiFi-Channel-State-Info"
source: https://arxiv.org/pdf/2609.02452v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 10:18:57"
field: "无线感知与低层视觉交叉"
keywords: ["WiFi Sensing", "Channel State Information", "Optical Flow", "Device-free Sensing", "Dataset", "Motion Estimation"]
innovations: ["首次提出端到端CSI-to-optical-flow映射框架WiFlow", "设计三种差异化解耦稀疏运动的架构（Simple/RoI/Combo）", "构建首个含同步CSI与伪真值光流的多接收端 indoor 数据集"]
benchmarks: ["WiFlow Dataset (sideview/birdview/birdview+)", "EPE / EPE_M / EPE_S / EPE_A", "Subject Split / Time Split"]
---

# 论文速读：WiFlow-Estimating-Optical-Flow-using-WiFi-Channel-State-Info

## 一句话总结
本文首次探索仅使用WiFi信道状态信息（CSI）估计场景光流（optical flow）的可行性，提出WiFlow框架及三种不同复杂度的模型架构，并构建首个CSI+光流同步数据集。

## 研究问题与动机
- **核心问题**：能否仅凭WiFi CSI数据恢复场景中的稠密光流场，替代相机方案？
- **相机方案的局限**：采集图像会暴露隐私，且在低光照条件下性能大幅下降；而CSI天然具备暗环境鲁棒性。
- **CSI的运动表征能力**：CSI反映信号在多径环境中的传播变化，人体等移动物体会引起结构化时变特征，具有潜在的运动编码能力。
- **现有数据集不足**：既有光流数据集无CSI同步，既有CSI数据集缺乏高分辨率多接收端、视频同步标注，无法直接训练该任务。

## 核心贡献（创新点）
1. **首个CSI-光流同步数据集WiFlow Dataset**：包含多接收端（4 Rx×4天线）、双视角视频（sideview/birdview）、9类动作、10名被试，并提供基于SOTA方法集成生成的伪真值（Pseudo_GT）。
2. **三种差异化架构WiFlow_Simple / WiFlow_RoI / WiFlow_Combo**：从单流端到"掩码+流"两阶段/并行分支，显式建模运动稀疏性。
3. **面向静止背景主导场景的改进损失**：对RAFT序列损失进行指数加权，放大非零流像素误差，避免模型退化为全零预测。
4. **系统性CSI预处理分析**：对比Raw、Quotient、Fourier、SavGol、PCA五种预处理，发现Quotient在移动像素EPE_M及综合指标EPE_A上最优。
5. **多接收端空间分集的有效性验证**：1→4个接收机的消融表明，多个空间分离接收端可显著提升光流推断精度。

## 方法详解
- **输入表示**：CSI复值张量 $H_i \in \mathbb{C}^{A \times K \times N}$，其中 $A=T\cdot D\cdot R=16$（1发射、4接收、每接收4天线），$K$ 为80 MHz带宽下子载波数，$N\in\{10,100\}$ 为时间窗口长度。
- **输出目标**：光流场 $F_i \in \mathbb{R}^{2\times H\times W}$，分辨率默认 $168\times128$，由两张同步视频帧经SOTA光流集成生成Pseudo_GT。
- **基础模块（WiFlow Block）**：借鉴RAFT设计，含Feature Network（ResNet变体）、Context Network（ResNet变体）、Refinement Network（ConvGRU迭代细化，M=4次）。
- **三种架构**：
  - $\mathrm{WiFlow}_{\mathrm{Simple}}$：单个Flow Block直接回归光流。
  - $\mathrm{WiFlow}_{\mathrm{RoI}}$：Mask Block先预测运动区域→提取轮廓边界框→RoIAlign结合Context/Feature特征→Decoder在ROI内估计光流。
  - $\mathrm{WiFlow}_{\mathrm{Combo}}$：Flow Block与Mask Block并行，最终结果 = 预测流 $\odot$ 预测掩码（逐点乘）。
- **训练策略**：Mask Block先以MSE预训练；整体采用余弦退火学习率（$1e^{-3}$）+ AdamW，60k步。
- **改进损失函数**：
  $$\mathcal{L} = \sum_{m=1}^{M} \gamma^{M-m} (F_{gt} - F_m)^4, \quad \gamma=0.8$$
  对迭代 refinements 降权、对非零像素高权重，缓解背景静止主导带来的零预测塌陷。

## 实验与结果
- **数据集规模**：328段序列，共164分钟；9种动作（slow/fast-walk、together、collect、knee、mill、waving、off-area、void），10名被试。
- **评估指标**：EPE（总体）、$\mathrm{EPE_M}$（移动像素）、$\mathrm{EPE_S}$（静态像素）、$\mathrm{EPE_A}$（4倍放大的综合指标）。
- **最佳结果（birdview+ subject split）**：
  - $\mathrm{WiFlow}_{\mathrm{Combo}}$：EPE=**0.18**、$\mathrm{EPE_M}$=3.50、$\mathrm{EPE_S}$=**0.02**、$\mathrm{EPE_A}$未列但综合最优。
  - $\mathrm{WiFlow}_{\mathrm{RoI}}$：EPE=0.21、$\mathrm{EPE_M}$=3.41、$\mathrm{EPE_S}$=0.05。
  - $\mathrm{WiFlow}_{\mathrm{Simple}}$：EPE=0.45、$\mathrm{EPE_M}$=3.24（移动像素误差最低但背景噪声大）。
- **跨受试泛化**：Subject split与Time split结果接近，证明模型不依赖特定被试特征。
- **分辨率影响**：将输出分辨率从 $128\times168$ 提升至 $256\times336$ 后，所有EPE约翻倍，说明误差主要受限于CSI本身的表征能力而非网络容量。
- **计算开销（RTX 6000 Ada）**：Simple 23ms/22 GFLOPs/380MB；RoI 26ms/22 GFLOPs/380MB；Combo 47ms/43 GFLOPs/763MB。
- **预处理结论**：Quotient在$\mathrm{EPE_M}$和$\mathrm{EPE_A}$上均最优（Quotient: EPE=0.41, $\mathrm{EPE_M}$=2.89, $\mathrm{EPE_A}$=0.78），Raw虽总体EPE最低（0.40）但移动像素误差高（3.71），证实预处理必要性。

## 相关工作脉络
- **传统/深度学习光流**：FlowNet、PWC-Net、RAFT等从视频帧预测光流；本文首次将光流估计源从可见光替换为CSI。
- **CSI重建视觉内容**：CSI2Image/CSI2Depth等生成RGB或深度图，但目标仍是空间重建而非时间对应关系学习。
- **CSI人体姿态/分割**：Person-in-WiFi、MMFi等输出人体mask或骨架，属于人中心输出；本文输出场景级稠密流场。
- **CSI运动轨迹/定位**：WiDir/WiVelo/Widar等预测行走方向、速度加速度等物理量；本文输出的是图像平面稠密运动表示。
- **CSI预处理技术**：Fourier/Stft微多普勒、Wavelet去噪、PCA降维、Quotient多天线归一化、SavGol平滑；本文系统对比并验证Quotient最优。
- **定位差异**： prior work 要么重建空间内容（RGB/深度/网格）、要么预测度量空间运动（轨迹/速度）；本文是首个端到端CSI-to-optical-flow映射，推理时无需视频输入。

## 局限性与未来方向
- **环境泛化受限**：与多数WiFi感知工作类似，模型仅能在训练房间/布局下工作，跨场景迁移未验证。
- **伪真值不一致**：阴影运动在CSI频率域中不可见，导致Pseudo_GT存在固有噪声/偏差。
- **多目标精度下降**：多人场景下精度低于单人，数据集多人样本比例偏低；RoI分支在处理遮挡时仍有局限。
- **输出分辨率与细节有限**：相比相机光流，CSI派生光流的细节水平较低，可能制约实际部署。
- **未来方向**：提升分辨率与细节表征、探索跨环境/跨房间泛化、融合更多Rx/Tx拓扑、探索自监督/半监督训练以降低对Pseudo_GT依赖。

## 研究启发与可借鉴点
1. **稀疏运动建模范式**：将"掩码定位 + 局部精细估计"的两阶段思想引入CSI感知任务，可有效抑制静态背景噪声；对于稀疏响应信号（如RF/音频事件定位）具有迁移价值。
2. **面向非均匀误差分布的加权损失设计**：背景静止占比极高时，简单MSE易退化为零预测；指数衰减迭代加权 + 非零像素放大是一种通用正则策略。
3. **多接收端空间分集的价值量化**：通过逐设备消融揭示感知增益饱和点，为硬件部署成本-精度权衡提供可复用评估方法。
4. **预处理的系统化对比协议**：将Raw/Quotient/Fourier/SavGol/PCA放在同一任务下统一评测，可为其他CSI下游任务（姿态、轨迹、动作识别）提供预处理选型参考。
5. **Pseudo_GT集成生成管线**：使用多个SOTA光流模型（rpknet、MS-RAFT、SEA-RAFT、MemFlow、DPFlow）集成产生训练标签，可在缺乏真值的跨模态任务中推广。

## 关键术语表
- **Channel State Information (CSI)**：WiFi接收端对多径信道在多个子载波上的复频域估计，携带环境结构与运动信息。
- **Optical Flow（光流）**：连续两帧图像间像素级表观位移场，表征场景中物体的二维运动。
- **Device-free Sensing（无设备感知）**：无需被感知者携带任何设备，仅利用无线信号与环境/人体的交互进行感知的范式。
- **Pseudo Ground Truth (Pseudo_GT)**：利用多个SOTA光流模型集成并后处理生成的训练监督信号，替代昂贵的人工标注。
- **Quotient Preprocessing**：以参考天线为基准对多天线CSI做复数除法，消除同设备共享相位偏移的预处理方法。
- **RAFT (Recurrent All-pairs Field Transforms)**：基于全局相关体积与迭代精炼的卷积光流骨干网络，本文将其模块移植至CSI域。
- **EPE / EPE_M / EPE_S / EPE_A**：端到端误差（总体/移动像素/静态像素/4倍加权综合），用于衡量光流估计精度。
- **NexmonCSI**：在商用华硕路由器上通过固件补丁获取原始CSI的工具链，支撑本数据集的高采样率采集。

## 可复现要素
- **数据集**：WiFlow Dataset，论文声明代码与数据已在 https://visinf.github.io/wiflow 开源。
- **代码/权重**：论文明确"Code and data are available"，具体仓库地址见主页；权重与训练脚本需从官方链接获取。
- **关键超参**：学习率 $1e^{-3}$（余弦退火）、AdamW优化器、60k训练步、4次迭代细化、损失衰减因子 $\gamma=0.8$、CSI窗口 $N=10/100$、子载波80 MHz、采样率1 kHz、输出分辨率 $168\times128$（可选 $256\times336$）。
- **硬件**：采集端 USRP N2954-R（Tx）+ 4× Asus RT-AC86U（Rx，各4天线）；训练/推理 NVIDIA RTX 6000 Ada。
- **同步方式**：侧视30 Hz、俯视50 Hz视频，与1000 Hz CSI按时间戳配对；保留每帧前K个CSI（K=10或100）。
