---
title: "Revisiting-Cross-View-Completion-Self-Supervised-Pre-Trainin"
source: https://arxiv.org/pdf/2609.01530v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 19:12:24"
---

# 论文速读：Revisiting-Cross-View-Completion-Self-Supervised-Pre-Trainin

## 一句话总结
Gekko 提出了一种自监督预训练框架，通过将交叉视图补全（Cross-View Completion）的重建误差与掩码自编码器（MAE）误差进行比较，得到无需任何 3D 标注的共视性代理信号；在相同架构与训练数据下，Gekko 在零样本对应估计、相对位姿估计和点图回归三项任务上均稳定超越 CroCo，最严格阈值下位姿精度提升近 6 倍，且支持直接从原始视频中训练。

## 研究问题与动机
- **交叉视图补全在非共视区域退化为单目信号**：CroCo 在共视区域能有效利用参考视图重建目标图像，但在被遮挡或非共视区域，参考视图几乎无贡献，隐式退化为 MAE 式的单目重建，导致训练信号不足。
- **现有共视性监督依赖昂贵的 3D 标注**：Alligat0R 等前期工作需要 ground-truth 深度图和相机位姿，且依赖繁琐的重叠度预处理 pipeline，限制了可扩展性。
- **CroCo 重建误差本身无法可靠预测共视性**：图 2 左侧直方图显示，共视与非共视区域的 CroCo 误差分布高度重叠（AP 仅 0.57），单靠 $\ell_{\mathrm{CroCo}}$ 难以区分。
- **低重叠图像对未被充分利用**：标准交叉视图补全方法仅从精心筛选的高重叠对中受益，大量低重叠图像对训练信号弱，造成数据浪费。

## 核心贡献（创新点）
- **提出相对改进（Relative Improvement）作为自监督共视性代理**：通过比较 CroCo 与 MAE 的重建误差比值得到 $\mathbf{c}(\mathbf{p})$，AP 达 0.74，显著优于 CroCo 误差 alone 的 0.57；与 CroCo 本质区别在于显式建模了"跨视图增益"这一几何信号，而非仅依赖重建误差绝对值。
- **设计 Gekko 三任务联合训练框架**：网络同时进行交叉视图补全、MAE 重建和相对改进预测三个互补任务，新增仅一个输出通道，与 CroCo 共享 encoder-decoder 架构；与 CroCo 的本质区别是在每像素上引入了 binocular 监督信号，覆盖所有 masked 区域（包括非共视区域）。
- **在严格一致条件下全面超越 CroCo**：相同架构、数据、batch size 和训练步数下，ScanNet-1500 最严格阈值（10°/0.25m）位姿精度近 6 倍提升（29.1% vs 5.5%），ETH3D 零样本对应 AEPE 降低 22%；与已有工作的本质区别是不依赖 3D 标注而获得更强特征。
- **推出原始视频课程训练方案**：使用简单的 stride-based curriculum 直接从未标注原始视频中预训练，消除 3D 预处理依赖，匹配甚至超越经过精心筛选数据训练的模型；这是首个实现 fully self-supervised 的交叉视图补全训练方案。
- **额外通道具备强共视性检测能力**：Gekko 预测的第 4 通道 $\hat{\mathbf{C}}$ 在未见场景上以 0.763 AP 远超 CroCo 的 0.576 AP，且优于其自身训练所用的伪标签（0.555 AP）；这与 Alligat0R 使用 GT 共视掩码的本质不同在于完全自监督。

## 方法详解

### 核心洞察：相对改进作为共视性代理
定义第 $\mathbf{p}$ 像素处的相对改进：
$$\mathbf{c}(\mathbf{p}) = \frac{\ell_{\mathrm{MAE}}(\mathbf{p}) - \ell_{\mathrm{CroCo}}(\mathbf{p})}{\ell_{\mathrm{MAE}}(\mathbf{p})}$$
- 共视区域：参考视图大幅降低重建误差，$\ell_{\mathrm{CroCo}} \ll \ell_{\mathrm{MAE}}$，$\mathbf{c}(\mathbf{p}) \approx 1$
- 非共视区域：两分支重建质量相近，$\mathbf{c}(\mathbf{p}) \approx 0$
- 结构性局限：在均匀区域或重复纹理区域，MAE 本身已能较好重建，$\ell_{\mathrm{MAE}}$ 较小，此时 $\mathbf{c}(\mathbf{p})$ 接近 0 但无法区分是否共视。

### 三阶段前向传播
给定目标图像 $I_{\mathrm{T}}$ 和参考图像 $I_{\mathrm{R}}$，Gekko 执行三个互补 pass：
1. **交叉视图补全 pass**：$\hat{I}_{\mathrm{T|R}} = F_{\mathrm{Gekko}}(\mathbb{M} \odot I_{\mathrm{T}}, I_{\mathrm{R}})$，其中 $\mathbb{M}$ 随机遮盖 90% 目标 patch
2. **MAE pass**：$\hat{I}_{\mathrm{T}} = F_{\mathrm{Gekko}}(\mathbb{M} \odot I_{\mathrm{T}})$，不使用参考视图
3. **相对改进预测 pass**：$\hat{\mathbf{C}} = F_{\mathrm{Gekko}}(I_{\mathrm{T}}, I_{\mathrm{R}})$，对未遮盖图像进行单次额外前向传播

### 网络架构
- Encoder：ViT-based，与 CroCo 完全相同
- Decoder：cross-attention transformer，MAE pass 中退化为 self-attention
- **Pixel head**：仅在 CroCo 基础上增加第 4 个输出通道用于预测 $\hat{\mathbf{C}}$，其余结构不变

### 训练损失
$$\mathcal{L}_{\mathrm{Gekko}} = \sum_{\mathbf{p} \in \Omega_{\mathtt{M}}} \ell_{\mathrm{MAE}}(\mathbf{p}) + \ell_{\mathrm{CroCo}}(\mathbf{p}) + \ell_{\mathrm{RI}}(\mathbf{p})$$

相对改进损失（Eq. 8）：
$$\ell_{\mathrm{RI}}(\mathbf{p}) = \left( \mathrm{sg}[\ell_{\mathrm{MAE}}(\mathbf{p}) - \ell_{\mathrm{CroCo}}(\mathbf{p})] - \mathrm{sg}[\ell_{\mathrm{MAE}}(\mathbf{p})] \cdot \hat{\mathbf{C}}(\mathbf{p}) \right)^2$$

关键设计：
- **Stop-gradient（sg）**：同时阻断 $\ell_{\mathrm{MAE}}$ 和差值的梯度回传到 encoder/decoder，防止 MAE 分支的梯度干扰相对改进预测目标；MAE 和 CroCo 分支独立优化各自重建损失
- **以 $\ell_{\mathrm{MAE}}$ 作为软加权**：自动 down-weight 低 $\ell_{\mathrm{MAE}}$ 像素（均匀/重复纹理区域），避免噪声伪标签干扰训练，消融证明此项提升 18.0 个点
- **相对形式优于绝对形式**：回归 $\ell_{\mathrm{MAE}} - \ell_{\mathrm{CroCo}}$ 的绝对改进在 ScanNet-all 上差 18.3 个点（39.4% vs 57.6%）

### 原始视频课程训练（Stride Curriculum）
- 初始步长 $k_{\max} = 1$，每 $N = 5\mathrm{k}$ 步增加 $\Delta k$（-50 variant: $\Delta k = 2$，-all variant: $\Delta k = 10$）
- 每对从 $[1, k_{\max}]$ 均匀采样步长，逐步引入更低重叠度的图像对
- 完全不需要 3D 标注、SfM、位姿或重叠度估计，12 源原始视频 mix 即可直接训练

## 实验与结果

### 数据集
- **预训练**：Cub3（ScanNet 室内，含 ScanNet-50 和 ScanNet-all 两种重叠度变体）、DL3DV（室外场景）、12 源原始视频 mix（ScanNet、ScanNet++、ARKitScenes、RealEstate10K、EPIC-KITCHENS、Hypersim、CO3Dv2、Objectron、DL3DV、KITTI-360、VirtualKITTI2、TartanAir）
- **评估**：ETH3D（零样本对应估计）、ScanNet-1500（相对度量位姿估计）、ScanNet/DL3DV/ETH3D（点图回归）、7-Scenes（位姿 fine-tuning）、MPI-Sintel（光流）

### 关键结果

**零样本对应估计（ETH3D，AEPE ↓）**：
| 模型 | Encoder | Decoder |
|------|---------|---------|
| CroCo-B | 78.4 | 101.9 |
| Gekko-B | **60.8** (-22.4%) | **80.8** (-21.0%) |

**相对度量位姿估计（ScanNet-1500，10°/0.25m，准确率 ↑）**：
| 模型 | ScanNet-50 | ScanNet-all |
|------|-----------|-------------|
| CroCo-B | 5.5% | 6.6% |
| Gekko-B | **29.1%** (+4.3×) | **43.7%** (+6.6×) |
| CroCo-L | — | 7.5% |
| Gekko-L | — | **35.9%** (+4.8×) |

**点图回归（Chamfer Overall ↓，DL3DV 预训练 + ScanNet-all fine-tune）**：
| 模型 | ScanNet | DL3DV | ETH3D |
|------|---------|-------|-------|
| CroCo-B | 0.126 | 1.430 | 0.399 |
| Gekko-B | **0.113** | **1.359** | **0.373** |

**共视性分类性能（ScanNet-1500，AP ↑）**：
- Gekko $\hat{\mathbf{C}}$：**0.763 AP** / 0.737 ROC-AUC / 0.691 balanced accuracy
- CroCo $\ell_{\mathrm{CroCo}}$：0.576 AP / 0.577 ROC-AUC / 0.539 b.acc.
- 在最低重叠度三分位组中差距最大：Gekko +0.32 AP vs chance，CroCo 仅 +0.04 AP

**与已发布 backbone 对比（Frozen Large，ScanNet-1500 10°/0.25m）**：
- Gekko-L$^{\mathrm{mix}}$：**33.2%** vs CroCo v2-L 19.7%（+13.5pt）、MuM-L 19.9%

### 消融关键结论
- 仅加 MAE 不加 $\ell_{\mathrm{RI}}$：几乎无提升（6.7% vs 5.5%），增益来自相对改进预测而非多任务训练
- 绝对改进（$\ell_{\mathrm{AI}}$）远差于相对形式（23.3% vs 43.7% on ScanNet-all）
- 无 patch normalization 时 ScanNet-all 性能暴跌（14.2% vs 39.4%）
- 公式(8)的 down-weighting 设计贡献 18.0 个点
- 数据缩放实验：Gekko 在 1%~100% 全数据规模下均领先，且 Indoor-only mix 在 indoor benchmark 上优于 full 12-source mix（39.4% vs 34.4%）

## 相关工作脉络
- **CroCo/CroCo v2（Weinzaepfel et al., 2022/2023）**：交叉视图补全奠基工作，将 MAE 扩展到图像对；Gekko 在架构和训练数据上与 CroCo 完全对齐比较，仅增加一个输出通道和 $\ell_{\mathrm{RI}}$ 损失，是对 CroCo 训练信号的直接改进而非新架构。
- **MAE（He et al., 2022）**：单图掩码重建；CroCo 在此基础上引入参考视图，Gekko 进一步显式引入 MAE 作为 binocular 信号的对比基线，将单目/双目重建差异转化为几何监督。
- **DUSt3R / MASt3R（Wang et al., 2024; Leroy et al., 2024）**：基于 CroCo 框架的稠密 3D 重建和匹配方法；Gekko 与这些方法正交——Gekko 改进的是预训练阶段的像素级训练信号，而 DUSt3R/MASt3R 是在 CroCo 之上构建下游任务头。
- **Alligat0R（Loiseau et al., 2025）**：使用 ground-truth 共视掩码进行监督预训练；Gekko 追求同等目标（学习共视性），但完全去除对 3D 标注和重叠度预处理的依赖，实现 fully self-supervised。
- **MuM / Muskie（Nordström et al., 2026; Li et al., 2025）**：将交叉视图补全扩展到多视图；Gekko 的相对改进信号与多视图扩展正交，文中指出多视图场景下需重新定义"相对改进"（相对于一组参考而非单个参考），是未来的结合方向。
- **P-Match（Zhu & Liu, 2023）**：双图均部分遮盖的图像匹配预训练；Gekko 关注的是参考视图对目标 masked 区域的重建增益，与 P-Match 的思路互补而非替代。

## 局限性与未来方向
- **预训练成本增加**：额外一次全分辨率前向传播使激活内存约翻倍，每设备 batch 减半（Base: 48 vs 96），总 GPU 时间约 2.4×（~400 H100 小时）；fine-tuning 和推理阶段无额外开销。
- **相对改进比值在低 MAE 区域不可靠**：均匀区域和重复/自相似结构导致 $\ell_{\mathrm{MAE}}$ 本身就小，$\mathbf{c}(\mathbf{p})$ 接近 0 但与真实共视性无关；当前通过软 down-weight 缓解，显式 masking 可能进一步改善。
- **仅限静态场景**：方法未假设刚性，但未在动态场景上验证；动态环境下的共视性定义需要额外考虑。
- **单图任务收益有限**：NYUv2 深度估计、ADE20K 分割、ImageNet-1K 分类等单图探针上 Gekko 与 CroCo 基本持平，不声称通用表征改进。
- **仅支持两视图**：固定 patch 大小和两视图输入的限制，扩展到多视图需重新设计相对改进的定义。
- **Full fine-tuning 优势缩小**：在 MPI-Sintel 光流和 7-Scenes 位姿的全网 fine-tuning 上，Gekko 的领先优势减弱或不复存在，表明改进主要体现在 frozen feature probing 层面。

## 研究启发与可借鉴点
- **"误差对比"作为几何自监督信号的设计范式**：用单目/双目重建误差的比值/差值替代人工标注的几何标签，这一思路可迁移到其他多视图几何任务（如立体匹配、多视图重建），在缺少 3D 标注的场景中构建 dense 几何监督。
- **Stop-gradient 的巧妙解耦设计**：公式(8)通过 sg 阻断 MAE 误差的反向传播，使 encoder/decoder 同时优化三个目标而互不干扰，是 multi-task self-supervised 训练中值得复用的梯度隔离技巧。
- **以训练目标自身的统计特性（$\ell_{\mathrm{MAE}}$）作为软加权因子**：用"难样本检测器"自动屏蔽不可靠伪标签，这一策略可泛化到其他自监督学习中噪声标签过滤的场景。
- **原始视频课程训练的 stride curriculum**：从相邻帧逐步扩展到远距离帧的课程策略，摆脱对 SfM/位姿/重叠度预处理的依赖，可直接用于其他视频预训练任务。
- **额外输出通道的极低成本高收益设计**：仅增加 1 个输出通道（pixel head 从 3→4）即获得显著性能提升，这种"最小侵入式"改进思路对已有 backbone 的改造具有重要参考价值。

## 关键术语表
**Cross-View Completion（交叉视图补全）**：将 MAE 扩展到图像对，用参考视图 + 未遮盖的目标 patch 重建目标图像中被随机遮盖的 patch，迫使网络学习跨视图几何对应关系。

**Co-visibility（共视性）**：场景中某个 3D 点同时在参考视图和目标视图中可见（未被遮挡且投影在图像边界内），是双目光度/几何一致性的基础概念。

**Relative Improvement（相对改进）**：CroCo 重建误差相对于 MAE 重建误差的归一化提升量 $\mathbf{c}(\mathbf{p}) = (\ell_{\mathrm{MAE}} - \ell_{\mathrm{CroCo}}) / \ell_{\mathrm{MAE}}$，作为自监督共视性代理。

**Stop-gradient（sg）**：阻止梯度沿某条计算路径反向传播的操作符，用于解耦多任务训练中的相互干扰，使某些分支的梯度不影响其他分支的参数更新。

**Stride-based Curriculum（步长课程学习）**：预训练初期使用相邻帧（小 stride）构建图像对，逐步增大最大步长以引入更多样化的重叠度，模拟由易到难的学习过程。

**Pointmap Regression（点图回归）**：在冻结的预训练特征上附加 DPT head，直接回归每个像素在参考坐标系下的 3D 点坐标，用于稠密 3D 重建评估。

**Chamfer Distance（钱弗距离）**：评估点云/点图回归质量的常用指标，计算预测点集与 ground-truth 点集之间双向最近邻距离的平均值，值越小表示重建越精确。

## 可复现要素
- **数据集**：Cub3（ScanNet）、DL3DV、12 源原始视频 mix（ScanNet、ScanNet++、ARKitScenes、RealEstate10K、EPIC-KITCHENS、Hypersim、CO3D
