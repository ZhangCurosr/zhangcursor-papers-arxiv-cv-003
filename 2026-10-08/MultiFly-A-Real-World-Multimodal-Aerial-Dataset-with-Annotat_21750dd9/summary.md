---
title: "MultiFly-A-Real-World-Multimodal-Aerial-Dataset-with-Annotat"
source: https://arxiv.org/pdf/2610.10359v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 22:50:35"
field: "多模态 aerial 感知与数据集构建"
keywords: ["UAV dataset", "multi-modal perception", "semantic segmentation", "label transfer", "aerial robotics", "cross-modal consistency", "LiDAR", "radar"]
innovations: ["首个公开的真实世界低空四模态（RGB/Thermal/LiDAR/Radar）同步语义分割数据集", "几何驱动的标注高效标签迁移方法，仅用 0.67% RGB 手动标注自动生成全部模态语义标注且无需人工修正", "首次系统评估 aerial 场景中稀疏雷达的语义分割性能，揭示稀疏卷积优于 point transformer"]
benchmarks: ["MultiFly Semantic Segmentation Benchmark", "Cross-Modal Semantic Consistency Evaluation", "RGB-Thermal Domain Adaptation Benchmark"]
---

# 论文速读：MultiFly: A Real-World Multimodal Aerial Dataset with Annotation-Efficient Label Transfer and Cross-Modal Semantic Consistency

## 一句话总结
本文提出了 MultiFly，首个公开的真实世界低空无人机四模态（RGB、热成像、LiDAR、雷达）语义分割数据集；通过几何驱动的标注迁移方法，仅用 115 张手动标注的 RGB 图像（占比 0.67%），自动生成 17,272 个跨模态一致的帧级语义标注，平均一致率达到 89.93%，跨模态语义一致性达 90.94%。

## 研究问题与动机
- **低空 UAV 多模态感知数据匮乏**：现有大规模多模态数据集（如自动驾驶领域的 nuScenes）主要覆盖地面场景，UAV 感知特有的俯视/斜视几何、物体尺度分布和场景统计特征缺乏代表性数据支撑。
- **现有 aerial 数据集模态覆盖不全**：已有 aerial 数据集多为单模态（如 UAVScenes 仅提供 RGB+LiDAR）或仅两模态组合，公开真实世界中尚无同时提供 RGB、热成像、LiDAR 和雷达四种模态帧级语义标注的数据集。
- **跨模态标注成本高且一致性难保证**：对各模态分别独立标注不仅成本高昂，还会因模态间几何与采样差异导致标注边界和语义不一致，亟需一种共享几何参考的统一标注策略。
- **雷达在 aerial 感知中的研究空白**：现有工作对稀疏雷达点云的语义分割缺乏系统评估，尚未建立可靠的 aerial 场景雷达分割基准。

## 核心贡献（创新点）
1. **MultiFly 数据集发布**：构建首个公开的真实世界低空四模态 aerial 数据集，包含 17,272 个同步样本、4 个城郊场景、15 个语义类别及完整传感器标定与位姿信息，覆盖面积 56,700 m²，超越现有数据集（如 SegFly 的 15,007 样本）的模态广度。
2. **标注高效的几何驱动标签迁移框架**：将仅 115 张 RGB 手动标注通过共享 3D 几何表示传播至全部四种模态，生成 17,157 张额外 RGB 图像、17,272 张热成像、8.4 亿 LiDAR 点和 340 万雷达点的语义标注，无需人工修正，平均一致性达 89.93%，区别于先前方法（如 SegFly）需人工验证与修正的局限。
3. **跨模态语义一致性量化评估与基准建立**：首次对所有六种模态对进行跨模态语义一致性评测（平均 90.94%），并建立四模态语义分割基准，揭示密集 LiDAR 与稀疏雷达在架构行为上的显著差异，填补 aerial 雷达分割基准空白。

## 方法详解
- **硬件平台设计**：基于 EmQopter Q6500 UAV，搭载前向 RGB 相机（FLIR Blackfly S，10 Hz，2048×1536）、热成像相机（Ouster microbolometer，20 Hz raw→10 Hz 选中，1280×1024）、 spinning LiDAR（Ouster OS1-128，10 Hz，360° 方位/45° 俯仰）和两个前向雷达（Continental ARS 548 RDI，10 Hz，其中一个下倾 14° 扩大垂直视场），所有传感器以 LiDAR 为参考帧进行外参标定，并通过 PTP 协议实现时间同步（LiDAR-Radar 偏差 ±3 ms）。
- **RGB-Thermal 配准**：采用 SegFly 的 2D-3D-2D 管线，利用 RGB 重建提供场景几何，通过固定 RGB-Thermal 外参将热像相机位姿绑定至 RGB 重建位姿，投影可见地标至两者的无畸变针孔域，计算共重视图支撑区域，将 RGB 图像重采样并映射至热像畸变空间，裁剪并缩放至热像分辨率，实现像素级对齐。
- **RGB 与 Thermal 标签迁移**（2D-3D-2D）：将稀疏手动标注的 RGB 图像通过图像-点云对应关系提升（lift）至 3D 重建点云，经邻域补全得到语义点云 $\mathcal{P}_{\text{sem}}^{\text{RGB}}$，再用可见性感知 Z-buffer 渲染至所有剩余 RGB 帧与热成像帧，生成稠密伪标签图 $\widehat{\mathcal{V}}^{\text{RGB}}$ 与 $\widehat{\mathcal{V}}^{\text{Th}}$。
- **LiDAR 标签迁移**：将各高度独立处理的 LiDAR 扫描变换至场景统一坐标系，聚合为场景级点云 $\mathcal{P}^{\text{Lid}}$，用可见性感知渲染将其投射至标注 RGB 源视图采样标签，多点观测时按多数投票规则分配标签（频率超过阈值 $\tau_{\text{Lid}} = 0.75$），否则标记为未标注 $c_{\text{unl}}$，保留点-扫描关联映射回原始扫描得到帧级标注 $\widehat{\mathcal{V}}^{\text{Lid}}$。
- **Radar 标签迁移与预处理**：将双雷达视为逻辑单雷达，聚合为场景级点云 $\mathcal{P}^{\text{Rad}}$；针对雷达数据含 18.8% 平均虚假目标的问题，先通过 RANSAC 平面拟合（$\tau = 0.2$ m）移除低于拟合平面两个标准差的异常点，再用 DBSCAN 聚类（$\varepsilon = 1.5$ m, min_points = 25）去除稀疏鬼影检测，被移除点标记为无效 $c_{\text{inv}}$；后续标签迁移流程与 LiDAR 相同（阈值 $\tau_{\text{Rad}} = 0.75$），生成帧级标注 $\widehat{\mathcal{V}}^{\text{Rad}}$。
- **跨模态一致性评估**：2D-2D（RGB-Thermal）通过几何映射比较像素标签；2D-3D（RGB/Thermal 与 LiDAR/Radar）通过标定外参将 3D 点投影至 2D 图像比较标签；3D-3D（LiDAR-Radar）通过最近邻关联（距离阈值 $r = 1$ m，覆盖 71% 雷达点）比较标签，评估几何保真度与语义一致性。

## 实验与结果
- **数据集统计**：4 个城郊场景（德国 Ingolstadt），2 个飞行高度（30 m、50 m），17,272 个同步样本，覆盖 56,700 m²，15 个语义类别，含完整 IMU/GNSS-RTK 位姿。
- **模态标签迁移评估**（Table IV）：RGB 平均准确率 92.12%（30m: 91.78%, 50m: 92.47%），Thermal 90.31%（30m: 90.51%, 50m: 90.11%），LiDAR 91.45%（30m: 91.80%, 50m: 91.11%），Radar 85.84%（30m: 85.92%, 50m: 85.76%），总体平均 89.93%。
- **跨模态语义一致性**（Table V）：RGB-Thermal 2D-2D 平均 91.02%，RGB-LiDAR 2D-3D 91.56%，RGB-Radar 88.73%，Thermal-LiDAR 91.88%，Thermal-Radar 89.06%，LiDAR-Radar 3D-3D 93.42%，六种模态对总体平均 90.94%。
- **语义分割基准**（Table VI, VII）：RGB-Thermal 实验中 Firefly 最优（RGB mIoU 44.05%, Thermal mIoU 40.24%）；LiDAR 实验中 PTv3 最优（mIoU 37.09%）；Radar 实验中 SpUNet 最优（mIoU 14.50%），表明稀疏卷积在稀疏雷达点云上优于 point transformer 架构。
- **最强结果**：LiDAR-Radar 3D-3D 一致性最高达 93.42%；Firefly 在 RGB-Thermal 分割任务上取得最佳 mIoU；PTv3 在 LiDAR 分割上取得最佳 mIoU 37.09%。

## 相关工作脉络
- **SegFly [5]**：两模态（RGB-Thermal）aerial 数据集与 2D-3D-2D 标签迁移范式，MultiFly 将其扩展至四模态并消除人工修正需求，实现更广模态覆盖与更高自动化程度。
- **UAVScenes [6]**：提供 RGB+LiDAR 与 IMU/GNSS，但仅两模态且无帧级语义标注传播机制，MultiFly 在其基础上补充 Thermal 与 Radar 并提供完整的跨模态语义一致性保证。
- **KITTI-360 [21]**：地面场景的图像-LiDAR 标签迁移方法，MultiFly 借鉴其可见性感知渲染策略并适配至 aerial 四维场景与多模态传播。
- **WildScenes [22]**：多视图图像标注投影至 LiDAR 的方法，MultiFly 将其思想推广至四模态 aerial 场景，并加入雷达特有的预处理（RANSAC+DBSCAN）。
- **nuScenes [13]**：自动驾驶多模态基准，MultiFly 回应其在 aerial 场景的空白，强调 UAV 特有的俯视几何与物体尺度分布差异。
- **SegFly 对比定位**：MultiFly 相比 SegFly 的核心差异在于从两模态扩展至四模态、从需人工验证修正到零人工修正、从 15K 样本扩展至 17K+ 并覆盖 LiDAR 与 Radar 的 3D 语义标注。

## 局限性与未来方向
- **动态物体处理受限**：当前标注迁移依赖静态 3D 重建，对运动物体（如车辆、行人）的语义传播可能存在不一致，文中提及 4D 重建方法 [25] 是潜在改进方向。
- **Radar 标注质量相对偏低**：Radar 平均一致性 85.84%，低于其他模态，源于雷达固有的虚假目标（18.8%）与稀疏性，预处理阈值（RANSAC $\tau=0.2$ m, DBSCAN $\varepsilon=1.5$ m）可能需要场景自适应调优。
- **城郊场景单一性**：仅采集 4 个城郊场景，缺乏城市密集区、机场、乡村等多样化环境，限制了模型的泛化评估。
- **飞行高度与轨迹限制**：仅覆盖 30 m 与 50 m 两个高度，双网格轨迹可能导致视角多样性不足，高空或斜视场景覆盖有限。
- **雷达空间分辨率限制**：雷达点云稀疏且精度较低，语义分割 mIoU 仅约 14.5%，难以支撑精细感知任务，需更高分辨率雷达或融合策略。

## 研究启发与可借鉴点
- **几何驱动的零人工修正标注迁移**：通过共享 3D 几何作为语义接口，将极少手动标注（0.67%）自动传播至多模态，为其他多模态数据集构建提供可扩展范式，可减少 90%+ 的人工标注成本。
- **雷达数据预处理 pipeline**：RANSAC 平面拟合+DBSCAN 聚类去除虚假目标的预处理策略，对 aerial 雷达点云语义分割具有直接可迁移价值，可作为雷达感知的标准预处理流程。
- **跨模态一致性作为几何保真度代理指标**：用跨模态语义一致性（90.94%）同时验证几何配准精度与语义传播质量，为多模态数据集质量评估提供无需额外人工标注的高效评测手段。
- **稀疏卷积对稀疏雷达的优势发现**：benchmark 结果显示 SpUNet（稀疏卷积）在雷达分割上优于 Point Transformer，挑战了"attention 模型更优"的直觉，启发后续研究在稀疏点云场景下重新评估架构选择。
- **与团队方向结合机会**：本团队的 multi-modal fusion 研究可直接利用 MultiFly 的四模态基准验证融合架构；annotation-efficient label transfer 方法可迁移至其他 aerial 或自动驾驶数据集构建；cross-modal consistency evaluation 方法可用于评估自监督多模态预训练的一致性。

## 关键术语表
**MultiFly**：本文提出的真实世界低空无人机四模态（RGB、热成像、LiDAR、雷达）语义分割数据集，含 17,272 个同步样本与 15 个语义类别。
**Annotation-Efficient Label Transfer**：通过仅 115 张手动标注 RGB 图像，利用共享 3D 几何将语义标签传播至所有模态的标注高效方法，无需额外人工修正。
**Cross-Modal Semantic Consistency**：衡量不同模态间同一场景语义标签一致性的指标，本文六种模态对平均达到 90.94%。
**2D-3D-2D Pipeline**：SegFly 提出的标签迁移范式，将 2D 图像标注提升（lift）至 3D 点云再渲染（render）回 2D 图像，本文扩展至四模态。
**Z-Buffered Rendering**：可见性感知的深度缓冲渲染方法，用于将 3D 语义点云投影至 2D 图像时处理遮挡关系。
**GNSS-RTK/IMU**：全球导航卫星系统实时动态定位与惯性测量单元，提供 UAV 的六自由度位姿与导航流（100 Hz）。
**Ghost Targets**：雷达测量中因多径反射或噪声产生的虚假检测点，本文数据中平均占比 18.8%，需通过 RANSAC+DBSCAN 预处理去除。
**Semantic Classes**：本文采用的 15 类语义标签体系，继承自 SegFly，涵盖道路、建筑、植被、车辆、行人等常见 aerial 感知类别。

## 可复现要素
- **数据集**：MultiFly 已公开，获取地址 https://github.com/markus-42/multifly，含 17,272 个同步样本、15 个语义类别、4 个场景、2 个高度（30m/50m）、完整标定参数与 IMU/GNSS-RTK 位姿。
- **代码**：论文未明确提及开源代码仓库，数据集仓库位于上述 GitHub 链接。
- **权重**：benchmark 使用模型官方配置与预训练权重（UPerNet-Swin-S、SegFormer-MiT-B3、Firefly、SpUNet、LitePT、PTv3），未提供本文训练权重。
- **关键超参**：LiDAR 标签迁移阈值 $\tau_{\text{Lid}} = 0.75$；Radar 标签迁移阈值 $\tau_{\text{Rad}} = 0.75$；RANSAC 平面拟合阈值 $\tau = 0.2$ m；DBSCAN $\varepsilon = 1.5$ m, min_points = 25；跨模态一致性评估雷达-LiDAR 邻域阈值 $r = 1$ m。
- **传感器规格**：RGB 2048×1536@10Hz，Thermal 1280×1024@10Hz（raw 20Hz），LiDAR 2048×128@10Hz，Radar 0.22m 范围分辨率，INS 100Hz。
