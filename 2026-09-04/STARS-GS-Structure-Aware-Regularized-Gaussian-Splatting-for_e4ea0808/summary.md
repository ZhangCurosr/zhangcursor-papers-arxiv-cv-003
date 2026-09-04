---
title: "STARS-GS-Structure-Aware-Regularized-Gaussian-Splatting-for"
source: https://arxiv.org/pdf/2609.03447v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 18:41:05"
field: "大规模3D表面重建"
keywords: ["3D Gaussian Splatting", "surface reconstruction", "large-scale aerial photogrammetry", "structure-aware partitioning", "neighborhood geometric constraint", "adaptive regularization"]
innovations: ["结构感知场景划分：融合几何与外观特征进行监督体素聚类与RAG合并，使分区边界对齐场景元素并保持跨区连续性", "邻域感知高斯组织：在几何感知邻域图上联合施加法线一致性与切向分布约束，从单点属性优化扩展至邻域相对排布优化", "自适应表面正则化：根据局部平坦度动态调节不透明度二值化与深度方差约束强度，平衡结构化与非结构化区域"]
benchmarks: ["GauU-Scene", "AIRLY", "UrbanScene3D", "Mill19"]
---

# 论文速读：STARS-GS: Structure-Aware Regularized Gaussian Splatting for Large-Scale Aerial Surface Reconstruction

## 一句话总结
针对大规模航拍场景表面重建中场景划分破坏结构连续性、几何约束仅关注个体高斯属性、均匀正则化难以适配异构几何等问题，提出STARS-GS框架，通过结构感知场景划分、邻域感知高斯组织与自适应表面正则化三阶段协同优化，显著提升几何精度与表面完整性，同时保持良好渲染质量。

## 研究问题与动机
1. **场景分割破坏结构连续性**：现有大规模3DGS方法将场景划分为独立优化的子区域，但划分边界常切割连续结构（如建筑、道路），导致相同地表元素被拆分到不同子区域独立训练，产生跨区域几何不一致与拼接伪影。
2. **几何约束局限于单点属性**：已有表面重建方法引入的深度/法线约束主要针对单个高斯基元的属性优化，而连续表面由多个相邻高斯共同决定，其几何不仅依赖个体属性，更取决于空间相对排布，单点约束难以捕捉局部邻域几何关系。
3. **均匀正则化无法适配异构场景**：大规模航拍场景包含建筑、道路（结构化/高平坦度）与植被、裸土（非结构化/低平坦度）等异构元素，统一强度正则化在结构化区域易过度平滑，在非结构化区域又无法提供足够几何约束。
4. **计算效率与几何精度难以兼得**：随着场景尺度增大，维持重建质量需要更多高斯基元，GPU内存与计算开销剧增；现有分区策略仅控制计算规模，未考虑结构完整性对几何质量的贡献。

## 核心贡献（创新点）
1. **结构感知场景划分策略**：基于空间邻近、法线相似与颜色特征进行监督体素聚类，并通过RAG合并减少过分割，使分区边界对齐场景元素边界；相邻子区域重叠带进行颜色与深度一致性精炼，减少跨区几何错位与拼接伪影。*本质区别：将划分目标从"控制训练规模"升级为"兼顾结构完整性与跨区一致性"，而非仅按相机可见性或点云密度硬性切分。*
2. **邻域感知高斯组织**：构建几何感知邻域图，联合施加法线一致性与切向分布约束，引导相邻高斯中心向局部切平面靠近并抑制法向堆叠，同时通过梯度调制与动态图更新维持训练过程中的局部几何组织。*本质区别：从"单点属性约束"扩展到"邻域相对空间关系约束"，使高斯集合整体贴合局部表面几何而非各自为政。*
3. **自适应表面正则化**：根据局部平坦度动态调整不透明度二值化与深度方差正则化的强度——高平坦度结构化区域增强约束以保证几何一致性，低平坦度非结构化区域减弱约束以保留合理局部变化。*本质区别：替代传统均匀正则化，实现"结构化区域严约束、非结构化区域松约束"的区域自适应机制。*

## 方法详解
**整体流程**：输入多视图航拍影像与SfM稀疏点云→结构感知场景划分（监督体素聚类+RAG合并+训练视角分配+边界精炼）→各子区域独立优化（邻域感知高斯组织+自适应表面正则化）→重叠带边界精炼→合并输出。

**结构感知场景划分（Sec. 3.2）**：
- **有效点筛选**：仅保留SfM track中有效视图数 $\eta_P \geq \tau_{\text{track}}$（默认5）的点，避免错误匹配点干扰特征计算。
- **联合特征构建**：对每个有效点，查询k近邻（$k=20$, $r_{\text{max}}=0.005 d_{\text{scene}}$）计算协方差矩阵，推导平坦度 $c_P$、曲率 $\sigma_P$、线性度 $\ell_P$、球形度 $S_P$；利用Depth Anything V2生成深度图并估计多视图法线，结合投影距离加权融合，再按局部平坦度与几何法线融合得到鲁棒法线 $\mathbf{N}_P$；颜色转换至CIELab空间减少曝光差异。
- **曲率引导种子初始化**：场景体素化后，高曲率体素分配更多种子（$K_i = K_{\text{total}} \frac{1+r_i}{\sum(1+r_j)}$），初始种子移至局部法线与颜色梯度最小位置。
- **结构感知聚类距离**：$D(P,S_k) = \sqrt{d_z^2 + d_n^2 + d_c^2}$，其中 $d_z$ 含垂直惩罚因子 $\beta_z=2.0$ 以区分不同高程表面（如屋顶与地面）。
- **RAG合并**：相邻区域法线夹角小于10°且 $(\mathbf{N}_A \times \mathbf{d}) \cdot (\mathbf{N}_B \times \mathbf{d}) > 0$ 时合并。
- **边界精炼损失**：重叠带 $\Omega_{\text{ov}}$ 上施加颜色+深度一致性 $\mathcal{L}_{\text{ov}}$ 与透明度稀疏正则 $\mathcal{L}_{\text{sparse}}$，合并为 $\mathcal{L}_{\text{boundary}} = \mathcal{L}_{\text{ov}} + \lambda_{\text{sp}} \mathcal{L}_{\text{sparse}}$。

**邻域感知高斯组织（Sec. 3.3）**：
- **几何感知邻域图**：基于空间距离 $r_{\text{max}}$ 与几何一致性得分 $s_{ij} = \frac{1}{2}[|\mathbf{n}_i^\top \mathbf{n}_j| + 1 - 3|\sigma_i - \sigma_j|]$ 选取前50%候选邻居构建无向图，训练过程中随分裂/克隆动态更新。
- **法线一致性约束**：$\mathcal{L}_{\text{nor}} = \sum_i \sum_{j \in N(i)} (1 - |\mathbf{n}_i \cdot \mathbf{n}_j|)$ 促进局部表面朝向连贯。
- **切向分布约束**：$\mathcal{L}_{\text{tan}} = \sum_i \sum_{j \in N(i)} (\mathbf{v}_{ij} \cdot \mathbf{n}_{\text{avg}})^2$ 抑制沿法线方向冗余堆叠。
- **梯度调制**：用临时参数预测法线与邻域平均法线的偏差定义调制权重 $w_i^n = \exp(-\|\hat{\mathbf{n}}_i - \bar{\mathbf{n}}_{N(i)}\|)$，抑制与邻域几何不一致的更新。

**自适应表面正则化（Sec. 3.4）**：
- **不透明度二值化**：$\mathcal{L}_{\text{bin}} = \frac{1}{|S|}\sum_{g \in S} \mathbb{M}_{\text{adapt}}(g) \Phi(\alpha_g)$，其中 $\Phi$ 为二元交叉熵形式，$\mathbb{M}_{\text{adapt}}(g) = \frac{1+c_p(g)}{\frac{1}{|S|}\sum(1+c_p(g))}$ 按局部平坦度归一化加权。
- **深度方差正则**：$\mathcal{L}_{\text{var}} = \frac{1}{|\mathcal{U}|}\sum_{u \in \mathcal{U}} \mathbf{W}_{\text{adapt}}(u) \text{Var}(d)_u$，自适应权重 $\mathbf{W}_{\text{adapt}}(u) = \frac{1+R_u}{\frac{1}{|\mathcal{U}|}\sum(1+R_u)}$，$R_u$ 为像素级平坦度响应（按 $\alpha$-compositing 贡献加权）。

**总体优化目标**：
$$\mathcal{L} = \mathcal{L}_{\text{render}} + \lambda_{\text{nbr}}(\mathcal{L}_{\text{nor}} + \mathcal{L}_{\text{tan}}) + \lambda_{\text{surf}}(\mathcal{L}_{\text{bin}} + \mathcal{L}_{\text{var}}) + \lambda_{\text{bd}} \mathcal{L}_{\text{boundary}}$$
损失逐步激活：先渲染损失建立初始分布→7000步后激活邻域约束（权重线性增）→15000步后激活表面正则→子区域独立训练后3000步边界精炼。

## 实验与结果
**数据集**：主评估使用GauU-Scene（4个场景，共约3675张图像，总面积约5.86 km²）与AIRLY（602张，0.325 km²），两者均含高分辨率航拍影像与LiDAR地面真值；补充评估使用UrbanScene3D（Residence/Sci-Art）与Mill19（Building/Rubble）。

**对比基线**：SuGaR、2DGS、PGSR、CityGaussianV2；AIRLY上额外对比GSDF、RaDe-GS、Trim2DGS（表面重建）以及BlockGaussian、VastGaussian（新视角合成）。所有方法使用2DGS mesh-extraction管线统一提取网格（体素1m，TSDF截断4m）后公平比较。

**主要定量结果**：
- GauU-Scene平均F1-score：STARS-GS 0.693，第二CityGaussianV2为0.651，提升约6.5%。
- AIRLY平均F1-score：STARS-GS 0.719，第二RaDe-GS为0.670，提升约7.3%；若与第二类Gaussian方法（PGSR等）对比，从0.596提升至0.719，提升约20.6%。
- STARS-GS在所有5个测试场景上F1均为最高；新视角合成PSNR/SSIM/LPIPS三指标均最优（平均PSNR 24.79 dB，较最佳基线提升0.60 dB；LPIPS降低17.2%）。
- 训练效率：平均训练时间3.60h，低于SuGaR（4.20h）、PGSR（5.60h）、CityGaussianV2（3.96h）；平均高斯数量24.20M，低于PGSR（28.14M）与CityGaussianV2（59.30M）。

**消融结论**：
- 移除结构感知划分（替换为VastGaussian/BlockGaussian分区）主要损害Recall，证明其对表面完整性贡献显著。
- 移除边界精炼导致Precision与Recall双降，出现明显块状拼接伪影。
- 邻域感知组织贡献最大：移除后平均F1从0.698降至0.627；仅用深度+法线约束仅恢复至0.650，证明"邻域相对几何关系"是核心增量。
- 自适应正则化在低平坦度区域将F1从0.598提升至0.736，验证区域自适应的必要性。
- 超参敏感性：$k=20$、$\lambda_{\text{nbr}}^{\max}=0.01$、$\lambda_{\text{surf}}^{\max}=0.01$ 为最优，参数容忍度较好（F1波动<2.4%）。

## 相关工作脉络
1. **大规模3DGS分区训练**：VastGaussian（按视锥可见性分区）、BlockGaussian（按区域复杂度自适应确定块范围）、CityGaussian（全局先验+自适应训练数据选择）。本文定位差异：上述方法仅控制计算规模，本文首次将"结构完整性"纳入分区目标，并通过边界精炼解决跨区不一致。
2. **几何导向表面重建3DGS**：CityGaussianV2（2D高斯建模+几何优化）、ULSR-GS（多视几何一致性）、GaussianCraft（细粒度几何精炼）。本文定位差异：现有工作聚焦子区域内部几何精炼，未处理分区边界切割连续结构的问题。
3. **直接几何约束方法**：SuGaR（表面对齐正则）、2DGS（2D高斯盘替代3D高斯）、PGSR（平面距离推导无偏深度）。本文定位差异：这些方法约束集中于单点属性（法线对齐、尺度控制），本文扩展至邻域相对排布约束。
4. **SDF/隐式表面结合**：GSDF（联合优化3DGS与SDF分支）。本文定位差异：依赖外部网络或隐式表示可能引入泛化性问题，STARS-GS纯在显式高斯框架内添加邻域与正则化约束，无需额外网络。
5. **大场景NVS方法**：BlockGaussian、VastGaussian。本文定位差异：虽以表面重建为核心目标，但几何约束同时提升新视角合成质量（PSNR/SSIM/LPIPS全优）。

## 局限性与未来方向
1. **依赖TSDF网格提取**：当前统一几何评估流程需经2DGS TSDF提取，有限体素分辨率与多视角平均可能衰减精细几何变化，且稀疏或不一致深度观测会导致表面不完整或定位偏差；未来探索直接从优化后高斯表征中提取表面。
2. **依赖SfM初始化质量**：分区结果与邻域关系受初始SfM点云质量（覆盖度、相机位姿、稀疏点可靠性）影响；未来可通过更强的几何初始化、联合相机位姿优化与跨视几何约束提升鲁棒性。
3. **额外计算开销**：几何特征构建、邻域图构建与动态更新、表面正则化引入额外开销；虽经多GPU并行大幅缩短wall-clock时间，但未来仍需探索增量图更新、稀疏邻域约束与高效并行调度。

## 研究启发与可借鉴点
1. **邻域图约束思想的迁移价值**：将几何约束从"单点属性"升级到"邻域相对排布"的思路可迁移至室内场景重建、稀疏视角重建等任务，避免高斯堆叠与法向混乱。
2. **自适应正则化策略的可复用性**：基于局部几何特征（平坦度/曲率）动态调节约束强度的机制，可推广至其他需要兼顾"规则结构"与"自然细节"的3D重建任务（如医学影像、遥感）。
3. **结构感知划分的分区设计范式**：融合几何特征（法线、曲率）与外观特征（颜色）的超体素聚类+RAG合并方案，可作为其他大场景3DGS工作的通用预处理模块。
4. **边界精炼的轻量化设计**：重叠带颜色/深度一致性+透明度稀疏正则的组合，以较低计算代价解决跨区拼接问题，可作为大规模并行训练的通用后处理步骤。
5. **梯度调制机制**：基于邻域平均法线偏差的动态梯度缩放，可有效抑制优化过程中的几何不一致更新，具有广泛的优化稳定性提升潜力。

## 关键术语表
**3D Gaussian Splatting (3DGS)**：使用可学习3D高斯基元显式表示场景，通过可微分光栅化实现实时照片级渲染的3D重建方法。
**Surface Reconstruction（表面重建）**：从多视角影像或点云恢复场景连续几何表面的任务，区别于纯渲染的新视角合成。
**Supervoxel Clustering（监督体素聚类）**：基于局部几何与外观特征将点云聚合成结构一致的体素单元，常用于点云语义分割与场景理解。
**Region Adjacency Graph (RAG)**：以聚类区域为节点、仅连接空间相邻区域的图结构，用于高效合并过分割区域。
**TSDF (Truncated Signed Distance Function)**：截断符号距离场，广泛用于多视角深度融合与网格提取的隐式表面表示方法。
**Normal Consistency Constraint（法线一致性约束）**：要求邻居高斯主法线方向相近的损失项，促进局部表面朝向连贯性。
**Tangential Distribution Constraint（切向分布约束）**：抑制邻居高斯中心沿法线方向堆叠的损失项，使高斯群更贴合局部切平面。
**Adaptive Surface Regularization（自适应表面正则化）**：根据局部几何平坦度动态调整正则化强度的策略，平衡结构化区域的几何约束与非结构化区域的细节保留。

## 可复现要素
- **数据集**：GauU-Scene、AIRLY、UrbanScene3D、Mill19；论文提及GauU-Scene与AIRLY含LiDAR真值，具体开源状态见原论文声明（GauU-Scene与AIRLY通常在对应引用论文中公开）。
- **代码/权重**：论文未明确声明代码开源状态（截至发表时），需核查项目主页或arXiv源码链接。
- **关键超参**：$k=20$（近邻数）、$r_{\text{max}}=0.005 d_{\text{scene}}$、$K_{\text{total}}=12$（种子数）、$\beta_z=2.0$（垂直惩罚）、$\lambda_{\text{nbr}}^{\max}=0.01$、$\lambda_{\text{surf}}^{\max}=0.01$、$\lambda_{\text{bd}}=0.01$、$\lambda_{\text{sp}}=0.01$、$\tau_{\text{track}}=5$；各子区域优化30000步，每250步densification，邻域约束7000步后激活，表面正则15000步后激活，边界精炼3000步。
- **硬件**：双 NVIDIA RTX A6000 GPU。
- **网格提取**：2DGS mesh-extraction管线，体素大小1m，TSDF截断距离4m。
