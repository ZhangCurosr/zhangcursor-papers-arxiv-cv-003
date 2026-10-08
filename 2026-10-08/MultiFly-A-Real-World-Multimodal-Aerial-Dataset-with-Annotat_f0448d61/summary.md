---
title: "MultiFly-A-Real-World-Multimodal-Aerial-Dataset-with-Annotat"
source: https://arxiv.org/pdf/2610.10359v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 09:49:12"
field: "多模态无人机语义感知"
keywords: ["UAV dataset", "multimodal semantic segmentation", "LiDAR", "radar", "thermal imagery", "label transfer", "cross-modal consistency"]
innovations: ["从115张RGB标注图像通过几何驱动传播生成17,272帧四模态（RGB/热红外/LiDAR/雷达）语义标签，平均转移准确率89.93%", "首次提供公开的四模态同步低空无人机语义分割基准，跨模态语义一致性达90.94%", "揭示稀疏雷达上稀疏卷积（SpUNet）优于Point Transformer的发现"]
benchmarks: ["RGB semantic segmentation (Firefly/SegFormer/UPerNet)", "Thermal semantic segmentation", "LiDAR semantic segmentation (SpUNet/LitePT/PTv3)", "Radar semantic segmentation (SpUNet/LitePT/PTv3)"]
---

# 论文速读：MultiFly: A Real-World Multimodal Aerial Dataset with Annotation-Efficient Label Transfer and Cross-Modal Semantic Consistency

## 一句话总结
论文提出了 MultiFly，一个真实世界低空无人机四模态语义分割数据集（RGB、热红外、LiDAR、雷达），并通过几何驱动的流程将仅 115 张手动标注的 RGB 图像上的标签传播到全部 17,272 帧数据中，平均转移准确率达 89.93%，跨模态语义一致性达 90.94%，同时建立了四个模态的语义分割基准。

## 研究问题与动机
- 现有大规模多模态数据集主要来自地面自动驾驶（如 nuScenes），而无人机（UAV）感知数据在多模态、大尺度方面严重匮乏。
- 无人机场景具有独特的统计特性（ nadir 到斜视的多角度、不同尺度目标），无法简单复用地面数据集的训练范式。
- 多模态语义标注成本极高且各模态几何特性差异大（RGB/热图为密集像素，LiDAR 为中等密度点云，雷达为稀疏点云+多 Ghost target），独立标注会引入模态间不一致。
- 现有 UAV 数据集大多仅含单模态或两种模态组合，且无公开数据集提供 RGB+热红外+LiDAR+雷达的全同步帧级语义标注。

## 核心贡献（创新点）
- **首个四模态低空无人机同步数据集**：MultiFly 提供 17,272 帧同步 RGB/热红外/LiDAR/雷达数据，覆盖 4 个郊区场景、两个高度（30 m/50 m）、15 个语义类别，并附完整传感器标定与 GNSS-RTK/IMU 位姿。
- **标注高效的几何驱动标签传播**：仅需 115 张手动标注的 RGB 图像（占总量 0.67%）即可自动为剩余 17,157 张 RGB、全部热红外图像、8.4 亿 LiDAR 点和 340 万雷达点生成语义标签，且无需人工修正。
- **跨模态语义一致性量化评估**：提出并量化六种模态对的语义一致性指标，平均达 90.94%，证明几何驱动传播在多种模态间保持高一致性。
- **四模态语义分割基准**：建立 RGB、热红外、LiDAR、雷达四个模态各自的语义分割基准模型（Firefly、SpUNet、PTv3 等），揭示密集 LiDAR 与稀疏雷达在架构行为上的显著差异。

## 方法详解

### 一、硬件平台与时间同步
- **传感器配置**：1 台前视 RGB（FLIR Blackfly S，10 Hz）、1 台热红外（Ouster microbolometer，20 Hz raw）、1 台旋转 LiDAR（Ouster OS1-128，10 Hz）、2 台前视雷达（Continental ARS 548，10 Hz）、1 套 GNSS-RTK/INS（OxTS xRED，100 Hz）。
- **整体云台下倾 45°**，1 台雷达额外下倾 14° 以扩大垂直视场。
- **时间同步**：以 INS 为 PTP grandmaster，通过 Precision Time Protocol 分发统一时钟；LiDAR 硬件触发 RGB，雷达通过 `cycle_offset` 参数补偿相位偏移（LiDAR-Radar 偏差 ±3 ms），热红外通过 GigE Vision Scheduled Action 软件触发。

### 二、传感器标定与配准
- **外参标定**：以 LiDAR 为参考帧（calibration tree）：
  - **RGB-LiDAR**：利用结构化场景的角点，求解 PnP 问题。
  - **Radar-LiDAR**：放置角反射器，通过 SVD 闭式求解刚性变换。
  - **RGB-Thermal**：棋盘格立体标定。
- **内参标定**：RGB 与热红外分别用 Zhang 平面标定法，LiDAR 与雷达使用出厂参数。
- **配准**：RGB-Thermal 配准依赖 RGB 重建提供的稠密几何（而非 LiDAR 深度），原因是在同一重建基础上联合完成配准与标签传播，避免引入额外的几何表示；热红外与 LiDAR、RGB 与 LiDAR/Radar 的投影则利用标定好的内外参直接实现。

### 三、标签传播流程（核心方法）

**步骤总览（SegFly 的 2D→3D→2D 范式扩展至四模态）**：

#### 3.1 RGB 与热红外标签传播
1. **2D→3D**：对每张手动标注的 RGB 源图像，利用已重建的稠密 RGB 点云 $\mathcal{P}^{\mathrm{RGB}}$ 与图像-点映射，将语义类 $c_m$ 提升为语义点云：
   $$\mathcal{P}_{\mathrm{sem}}^{\mathrm{RGB}} = \{( \mathbf{x}_m^{\mathrm{RGB}}, c_m )\}_{m=1}^{M_{\mathrm{RGB}}}$$
2. **3D→2D**：对语义点云使用 SegFly 的可见性感知 Z-buffer 渲染，将标签投影到所有剩余 RGB 帧与热红外帧，得到稠密伪标签图：
   $$\widehat{\mathcal{V}}^{\mathrm{RGB}} = \{\widehat{\mathbf{Y}}_n^{\mathrm{RGB}}\}_{n=1}^{N_{\mathrm{RGB}}}, \quad \widehat{\mathcal{V}}^{\mathrm{Th}} = \{\widehat{\mathbf{Y}}_n^{\mathrm{Th}}\}_{n=1}^{N_{\mathrm{Th}}}$$
3. 由于配准与渲染共用同一 metric 重建与固定 RGB-Thermal 外参，两类伪标签天然保持几何一致性。

#### 3.2 LiDAR 标签传播
- 将所有 LiDAR scan 变换到统一场景坐标系，聚合为场景级点云 $\mathcal{P}^{\mathrm{Lid}}$，保留每点与原始 scan 的关联。
- 将 $\mathcal{P}^{\mathrm{Lid}}$ 渲染到各标注 RGB 源视图，对每个点取多数投票，当相对频率超过阈值 $\tau_{\mathrm{Lid}} = 0.75$ 时赋予对应类别，否则标记为 $c_{\mathrm{unl}}$（ignore label）。
- 将语义标签反向映射回原始 scan，得到帧级 LiDAR 标注：
  $$\widehat{\mathcal{V}}^{\mathrm{Lid}} = \{\widehat{\mathbf{Y}}_k^{\mathrm{Lid}}\}_{k=1}^{K_{\mathrm{Lid}}}$$

#### 3.3 雷达标签传播（含预处理）
- 将两台雷达视为一个逻辑雷达，聚合为场景级点云 $\mathcal{P}^{\mathrm{Rad}}$。
- **预处理**：雷达数据约 18.8% 为 ghost target 和异常点，需两步清洗：
  1. RANSAC 平面拟合（$\tau = 0.2$ m），去除低于拟合地面 2 个标准差以下的点。
  2. DBSCAN 聚类（$\varepsilon = 1.5$ m, min\_points = 25），去除稀疏 ghost 检测点。
- 被剔除的点标记为 $c_{\mathrm{inv}}$（无效），未投票通过的点标记为 $c_{\mathrm{unl}}$。
- 清洗后与 LiDAR 相同的标签传播流程，得到：
  $$\widehat{\mathcal{V}}^{\mathrm{Rad}} = \{\widehat{\mathbf{Y}}_k^{\mathrm{Rad}}\}_{k=1}^{K_{\mathrm{Rad}}}$$

## 实验与结果

### 数据集规模
- 4 个郊区场景 × 2 个高度（30 m / 50 m）× 4 模态 = **17,272 帧同步样本**，覆盖 **56,700 m²**。
- 15 个语义类别：building, barrier, pole, truck, car, cyclist, motorcycle, bus, train, person, bike, bike rider, roof, vegetation, ground。

### 标签转移质量（Table IV）
| 模态 | 30m | 50m | 平均 |
|------|-----|-----|------|
| RGB | 91.78% | 92.47% | **92.12%** |
| Thermal | 90.51% | 90.11% | **90.31%** |
| LiDAR | 91.80% | 91.11% | **91.45%** |
| Radar | 85.92% | 85.76% | **85.84%** |
| **平均** | 90.00% | 89.86% | **89.93%** |

### 跨模态语义一致性（Table V）
- 六种模态对平均一致性：**90.94%**。
- LiDAR-Radar（3D-3D）最高：**93.42%**；且对近邻半径 $r$ 鲁棒（$r=1$ m 时 93%，$r=\infty$ 时降至 86%）。
- RGB-Thermal（2D-2D）：**91.02%**；RGB-LiDAR（2D-3D）：**91.56%**。

### 语义分割基准（Tables VI-VII）
**RGB / Thermal（Table VI）**：
- Firefly 表现最强：RGB mIoU 44.05%，Thermal mIoU 40.24%。
- RGB-to-Thermal 迁移有效（SegFly 协议），多模态适配有增益。

**LiDAR / Radar（Table VII）**：
- LiDAR：PTv3 最佳，mAcc 51.76%，mIoU 37.09%。
- Radar：**SpUNet 最佳**（mAcc 25.95%，mIoU 14.50%），说明稀疏卷积对稀疏雷达点云仍优于 Point Transformer，这是一个值得关注的发现。
- 雷达性能显著低于 LiDAR，主因是稀疏性与 ghost target。

## 相关工作脉络
- **SegFly [5]**：本文的直接基础，提出了 2D→3D→2D 的 RGB-Thermal 标签传播范式；MultiFly 将其扩展至四模态（新增 LiDAR 和 Radar）。
- **nuScenes [13]**：自动驾驶领域的多模态标杆数据集，启发本文在无人机领域的对标工作，但 nuScenes 缺少热红外与雷达。
- **UAVScenes [6]**：含 RGB+LiDAR 的 UAV 数据集，但仅 2 模态且标注量为 12 万帧，无帧级语义标签。
- **KITTI-360 [21] / WildScenes [22]**：几何驱动标签传播的先驱工作，将图像标注投射到 3D 点云，本文借鉴其 Z-buffer 渲染思路但面向 UAV 四模态场景。
- **SegFly 对比表 I**：已有 UAV 数据集（IndraEye、MVUAV、CART、Kust4K、SegFly、UAVScenes）均只覆盖 ≤2 种模态，且无公开的跨模态语义一致性标注；MultiFly 首次实现四模态同步帧级标注。
- **Flow4r [25]**：4D 重建与运动场估计；本文为未来解决动态物体标签传播问题提及了 4D 重建作为潜在方向。

## 局限性与未来方向
- **静态 3D 重建**：标签传播依赖静态重建，对运动物体（行人、车辆）的标签存在误差，尤其在高帧移场景下。
- **雷达预处理敏感性**：RANSAC 地面拟合和 DBSCAN 超参（$\varepsilon = 1.5$ m, min\_points = 25）对场景适应性未充分讨论。
- **场景单一性**：仅 4 个郊区场景，缺乏城市密集区、室内/半室外、极端光照等场景。
- **动态 4D 重建**：论文明确提到 4D 重建（Flow4r 等）是未来方向，可用于处理动态物体的标签传播。
- **雷达语义性能偏低**：mIoU 仅 ~14.5%，表明当前基准距离实用仍有较大差距。

## 研究启发与可借鉴点
- **几何驱动的跨模态标签传播可极大降低标注成本**：仅需 0.67% 的 RGB 手动标注即可生成全量四模态语义标签，这一范式可直接迁移到其他多传感器 UAV 平台。
- **雷达稀疏点的两阶段预处理（RANSAC 地面+DBSCAN 去噪）值得借鉴**：对于其他稀疏点云任务（毫米波雷达语义分割），这一预处理管线可作为强 baseline。
- **LiDAR-Radar 跨模态一致性评估（近邻半径敏感性实验）提供了一个可复用的评估范式**：用固定距离阈值衡量不同密度点云的语义对齐质量。
- **稀疏卷积（SpUNet）在稀疏雷达上优于 Point Transformer**：这一发现挑战了"attention 模型处处更强"的直觉，提示团队在未来雷达感知工作中优先考虑 sparse convolution 基线。
- **RGB-to-Thermal 跨模态迁移的三阶段训练协议**（RGB 训练 → RGB-to-Thermal 适配 → Thermal fine-tune）可作为团队多模态 domain adaptation 的实验设计参考。

## 关键术语表
- **MultiFly**：本文提出的首个包含 RGB+热红外+LiDAR+雷达四模态同步帧级语义标注的真实世界低空 UAV 数据集。
- **Annotation-Efficient Label Transfer**：仅需少量手动标注（115 张 RGB）即可通过几何传播为全部数据生成语义标签的方法。
- **Cross-Modal Semantic Consistency**：不同模态之间对同一场景内容赋予相同语义标签的一致性度量，本文平均达 90.94%。
- **Z-buffered Rendering（可见性感知深度缓冲渲染）**：将 3D 语义点云渲染到 2D 图像时考虑深度遮挡，确保标签投影的几何正确性。
- **Ghost Target**：雷达因多径反射等产生的虚假检测点，约占本文雷达数据的 18.8%。
- **GNSS-RTK/IMU**：全球导航卫星系统实时动态（RTK）结合惯性测量单元（IMU），提供厘米级定位与 6 DoF 位姿。
- **Firefly**：SegFly 论文提出的 RGB-Thermal 跨模态语义分割网络，在本文基准中表现最佳。
- **SpUNet / PTv3 / LitePT**：分别代表稀疏卷积（SpUNet）、轻量 Point Transformer（LitePT）和 Point Transformer V3（PTv3）三个 LiDAR/Radar 分割基线模型。

## 可复现要素
- **数据集**：已公开，GitHub 地址 https://github.com/markus-42/multifly。
- **代码**：论文未提及开源代码仓库（仅数据链接）。
- **传感器参数**：完整的相机内参/外参、LiDAR/雷达模型、同步时间戳已在数据中提供。
- **关键超参**：LiDAR/Radar 标签传播阈值 $\tau = 0.75$；雷达预处理 RANSAC $\tau = 0.2$ m，DBSCAN $\varepsilon = 1.5$ m, min\_points = 25；跨模态最近邻半径 $r = 1$ m（敏感性实验覆盖 0.1/0.25/0.5/1.0/2.0 m）。
- **训练协议**：RGB/Thermal 三阶段共 70 epoch（30+10+30）；LiDAR/Radar 各自 20 epoch；雷达 voxelization grid size 从 0.05 m 调整为 0.50 m。
- **硬件**：EmQopter Q6500 UAV + Aetina AIB MX22 边缘计算单元 + ROS 2 Humble + Docker。
