---
title: "Stable-and-Scalable-Bundle-Adjustment-of-Holistic-3D-Structu"
source: https://arxiv.org/pdf/2609.04026v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 18:41:59"
field: "三维计算机视觉/多视图几何"
keywords: ["Bundle Adjustment", "3D Reconstruction", "Geometric Primitives", "Structure from Motion", "Wireframe", "Plane Constraints"]
innovations: ["将group约束统一建模为camera-like实体并保持Schur补稀疏性", "Group-induced与cross-feature重投影误差避免3D正则化的尺度歧义", "线框约束reformulation为单特征优化保持Hessian块对角"]
benchmarks: ["1DSfM", "Hypersim", "ETH3D", "ScanNet++", "7Scenes"]
---

# 论文速读：Stable-and-Scalable-Bundle-Adjustment-of-Holistic-3D-Structu

## 一句话总结
本文提出了一种统一的全局Bundle Adjustment框架，通过将高阶几何结构（如平面、消失点、线框等）建模为"groups"并与相机参数一同优化，同时将所有约束统一表述为像素空间的重投影误差，从而在保持经典BA稀疏性和计算效率的前提下，实现了更丰富的3D几何重建。

## 研究问题与动机
- 传统Bundle Adjustment仅优化点和相机参数，无法利用真实场景中丰富的几何结构（平行、共面、线框等）来增强重建精度。
- 引入几何关系约束会导致特征间的耦合，破坏Hessian矩阵的块对角结构，使Schur补消元失效，计算复杂度剧增。
- 直接使用3D距离（如点到平面距离）作为正则项会引入尺度歧义、条件数恶化及权重调参困难等问题，且破坏BA的尺度gauge不变性。
- 现有方法在处理point-line关联（线框）时引入密集耦合，导致Schur补填充严重，难以扩展到大规模场景。

## 核心贡献（创新点）
1. **特征与组的分类学**：首次系统区分具有直接2D测量的"features"（点、线）与编码高阶关系的"groups"（消失点、平面、球体等），并将groups置于与非消除块（相机侧）相同的位置，从而保持features Hessian的块对角结构。
2. **Group-induced重投影误差**：将位置型组约束（如平面）重新表述为像素空间的重投影差异，避免直接3D正则化的尺度问题和条件数退化，同时隐式加权由Fisher信息矩阵决定。
3. **Cross-feature重投影误差**：将点线关联（线框）约束 reformulate 为单一特征参与优化的重投影误差形式，通过另一特征的2D观测作为固定测量值，保持稀疏性同时充分利用线框几何。
4. **统一像素空间优化框架**：所有约束（点/线重投影、组约束、跨特征约束）均在像素空间统一表述，收敛监测更简单，且兼容robust loss（Cauchy）。
5. **端到端SfM系统集成**：将框架集成到COLMAP增量式SfM管线中，从三角化到局部/全局BA均支持结构约束，实现 richer 的3D重建。

## 方法详解
- **特征-组分类学**：Features（点$\mathcal{X}$、线$\mathcal{L}$）具有直接2D观测；Groups（消失点VP、平面、球体等$\mathcal{G}$）编码高阶关系。将groups与cameras合并为$\mathcal{A} = \mathcal{C} \cup \mathcal{G}$，normal equations改写为：
$$\begin{bmatrix} \mathbf{H}_{aa} & \mathbf{H}_{af} \\ \mathbf{H}_{fa} & \mathbf{H}_{ff} \end{bmatrix} \begin{bmatrix} \delta\mathcal{A} \\ \delta\mathcal{F} \end{bmatrix} = -\begin{bmatrix} \mathbf{g}_a \\ \mathbf{g}_f \end{bmatrix}$$
其中$\mathbf{H}_{ff}$保持块对角，可进行Schur补消元：$\mathbf{S} = \mathbf{H}_{aa} - \mathbf{H}_{af}\mathbf{H}_{ff}^{-1}\mathbf{H}_{fa}$。

- **Group-induced重投影误差（位置型）**：对平面上一点$\mathbf{X}_j$，残差为原始重投影与投影到平面后的重投影之差：
$$r_g^{(p)} = \pi_p(C_i, \mathbf{X}_j) - \pi_p\Big(C_i, \text{Proj}_{G_g}(\mathbf{X}_j)\Big) \in \mathbb{R}^2$$
对线的类似形式为两端点重投影距离之差。线性化后等价于Mahalanobis距离$\delta^\top \Sigma_{\mathbf{X}}^{-1}\delta$，隐式加权由特征观测几何决定。

- **Cross-feature重投影误差（线框）**：对于点$\mathbf{X}_j$与线$\mathbf{L}_k$的关联，分两种情况：
  - 线可见但点不可见时：$r_{p\to l} = d(\pi_p(C_i, \mathbf{X}_j), \ell_{ik})$
  - 点可见但线不可见时：$r_{l\to p} = d(\mathbf{x}_{ij}, \pi_l(C_i, \mathbf{L}_k))$
  每个残差仅涉及一个feature和一个camera，不引入feature间耦合，保持$\mathbf{H}_{ff}$块对角。

- **方向型group约束**：VP与线的角向残差$r_v = \|\mathbf{d}_k \times \mathbf{v}_g\|$，scale-free，独立优化。

- **Inter-group约束**：VP对间的正交性/平行性、平面法向量间的约束仅影响$\mathbf{H}_{aa}$，不影响消元结构。

## 实验与结果
- **数据集**：1DSfM（3个户外场景：Tower of London、Madrid Metropolis、Gendarmenmarkt）、Hypersim（8个室内场景）、ETH3D（13个场景）、ScanNet++（20个场景）、7Scenes（7个场景）。
- **基线**：COLMAP（仅点）、Point-Line SfM（点+线）。
- **运行时间**：Holistic SfM比Point baseline慢约1.3×，远低于文献[39]报告的2-4×开销；迭代次数略增（因约束更丰富），但每迭代成本与经典BA相近。图3(a)显示四种配置均以$n^{1.6-1.9}$（SPARSE_SCHUR）或$n^{2.6-2.8}$（DENSE_SCHUR）缩放。
- **几何精度（Hypersim）**：加入group约束后，点@5mm inlier从31.7%提升至35.6%，线recall@5mm从222.5m提升至248.8m；wireframe进一步略有提升。
- **ETH3D点重建**：Accuracy@1cm从45.17%提升至47.05%，Completeness@1cm从0.14%提升至0.15%。
- **相机位姿（AUC）**：ScanNet++ AUC@3°从84.0（COLMAP）提升至87.4，改善最显著；ETH3D AUC@5°从66.4提升至68.1。
- **消融（Tab.5）**：2D重投影group误差无需调参即优于3D距离约束（需精细调节权重w=1~1000），后者权重过大时pose精度下降。
- **线框对稀疏性影响**（Tab.7）：wireframe约束仅使Schur complement密度增加约2-7%，实际影响极小。

## 相关工作脉络
- **经典BA与稀疏求解**：Triggs等（2000）[63]综述BA；Ceres solver [1]、g2o [32]、GTSAM [16]等库利用Schur补。本文在此基础上的扩展而非替代。
- **基于线的BA与SfM**：Bartoli & Sturm [5]提出无穷远线的优化框架；Liu等[39,40]（Line-SfM、3D Line Mapping Revisited）已实现点线联合优化，但未处理高阶group约束与跨特征关联的稀疏性问题。
- **平面约束BA**：早期工作[6,21]基于双目homography；近期[35,58]采用3D point-plane正则化，但存在尺度歧义和权重调参问题。本文以2D重投影差替代3D距离。
- **线框/点线关联优化**：[40]使用3D点线距离正则，引入feature间耦合破坏稀疏性。本文通过cross-feature reprojection reformulation解决。
- **Manhattan/Atlanta世界假设**：[14,52]显式假设正交方向；本文通过VP正交约束与平面法向量约束隐式编码，无需额外world assumption模块。
- **可微BA与学习式BA**：BA-Net [59]、DROID-SLAM [60]将BA融入端到端学习；本文聚焦传统优化框架的结构扩展。

## 局限性与未来方向
- **依赖上游关联质量**：方法假设feature-group和cross-feature关联已给定（precision-oriented策略），若关联错误率高则优化效果受限；模型选择/置信度估计仍是开放问题。
- **退化观测无法恢复**：视线几何退化（如沿光线方向）的特征无法被可靠地拉回到group表面，3D formulation可能在此有优势，但此类特征对group估计贡献本就微弱。
- **未支持所有primitive类型**：当前实现支持VP、平面、球体、圆柱、圆锥、椭球、长方体；splines、parametric surfaces等尚未纳入。
- **可扩展至3D基础模型先验**：作者在讨论中提到可结合monocular depth/normal priors（如从VGGT、Depth Anything 3等获取），但未在实际实验中验证。
- **全局BA扩展待验证**：虽声称sparsity analysis适用于global positioning [47,86]，但实验主要在增量SfM框架下完成，全局BA的实际表现需进一步评估。

## 研究启发与可借鉴点
- **稀疏性保持技巧**：将耦合约束转化为"虚拟相机-like"实体 + 重投影残差，是保持Schur补高效性的通用思路，可迁移到其他多视图优化问题（如SLAM中的平面/骨架约束）。
- **隐式不确定性加权**：重投影差线性化后自然导出Fisher信息矩阵，为不同特征的约束提供了几何一致的自动加权，无需手工调参，可应用于其他多源异构观测融合场景。
- **2D/3D混合约束的统一表述**：将3D metric约束转为pixel-space residual是一种避免尺度歧义的通用策略，尤其适合monocular/弱尺度场景。
- **线框约束的reformulation思路**：cross-feature约束通过固定一个feature的2D观测值、仅优化另一个feature，实现解耦——此技巧可用于其他 feature-pair 约束（如点-曲线、线-曲面）。
- **增量SfM的系统级集成**：从三角化、group匹配、合并到refinement的完整管线设计（含reliability-aware refinement）对构建生产级SfM系统有参考价值。

## 关键术语表
**Bundle Adjustment (BA)**：同时优化相机参数和3D结构，最小化所有观测的重投影误差的非线性优化问题。
**Schur Complement Elimination**：利用Hessian矩阵的块对角结构，先消元3D点变量，将系统压缩为仅含相机变量的 compact system，大幅降低计算复杂度。
**Feature vs. Group**：Features是具有直接2D观测的几何基元（点、线）；Groups是编码高阶关系（平行、共面等）的虚拟实体，无直接2D测量但通过associated features间接约束。
**Group-Induced Reprojection Error**：衡量3D特征经group表面投影前后的像素位移差，将3D结构约束转化为2D像素空间残差。
**Cross-Feature Reprojection Error**：将点线关联约束表述为单一特征的重投影到另一特征2D观测的距离，避免feature间直接耦合。
**Fisher Information Matrix**：重投影Jacobian的外积和，表征3D点位置的估计不确定性；group约束的隐式加权即由此矩阵决定。
**Wireframe**：由点和线在junction处连接构成的框架结构，编码建筑/城市环境中的几何骨架。
**Manhattan World Assumption**：假设场景中主导方向为三个相互正交的轴方向，常用于室内结构化环境重建。

## 可复现要素
- **代码开源**：是，代码发布于 https://github.com/cvg/limap，完全兼容COLMAP ecosystem。
- **数据集**：1DSfM [72]、Hypersim [50]、ETH3D [54]、ScanNet++ [79]、7Scenes [55]均为公开数据集。
- **关键超参**：VP角向约束权重$5\times10^3$，Cauchy loss尺度0.2；group reprojection权重0.5；wireframe reprojection权重0.1；base点/线重投影权重1.0。
- **检测/匹配工具**：ALIKED [82] / DeepLSD [49]用于点/线检测；LightGlue [38] / GlueStick [48]用于匹配；JLinkage [61]用于VP检测；MoGe-2 [68,69]用于平面分割。
- **BA求解器**：Ceres solver [1]，含analytic Jacobians。
- **论文未提及**：GPU加速、分布式BA扩展的具体实现细节。
