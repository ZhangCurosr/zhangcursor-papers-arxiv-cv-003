---
title: "Stochastic-Optimization-of-Tree-Tensor-Networks"
source: https://arxiv.org/pdf/2609.00870v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 11:52:14"
field: "张量网络机器学习优化"
keywords: ["Tree Tensor Networks", "Riemannian Optimization", "Stochastic Optimization", "Tensor Networks for ML", "Distance over Gradients", "Quotient Manifold"]
innovations: ["推导TTN参数流形与商流形上的随机自适应与免学习率黎曼优化算法族", "证明RADAGRAD在TTN商流形上的regret界并揭示规范自由度下的测地距离分解", "提出CNN-TTN混合架构并系统验证无约束优化对下游可压缩性的破坏"]
benchmarks: ["Fashion-MNIST", "CIFAR10", "Imagenette"]
---

# 论文速读：Stochastic-Optimization-of-Tree-Tensor-Networks

## 一句话总结
本文针对张量网络在机器学习中的应用，推导了树张量网络（TTN）上基于流形的随机优化算法，包括自适应与无学习率两类方案；借助混合 CNN–TTN 架构在 Fashion-MNIST、CIFAR10 和 Imagenette 上验证，结果表明所提方法在预测性能上与无约束 ADAM 相当，但能生成数值稳定的可压缩迭代点，使下游压缩任务得以成功执行。

## 研究问题与动机
- **传统优化与 TTN 不兼容**：物理中经典的 DMRG 算法为全批次确定性优化，假定目标函数固定且可精确计算局部环境，这与机器学习中基于海量随机样本的批量训练不匹配。
- **自动微分忽略几何结构**：当前主流做法通过反向传播计算 TTN 各核心张量上的欧式梯度，再喂给标准随机优化器（如 ADAM），这忽略了 TTN 固有的正交约束与商流形结构。
- **无约束优化导致下游任务崩溃**：直接对 TTN 参数做无约束优化会使根张量范数指数级增长，破坏正交性，使得依赖 SVD/正交化的后续任务（模型压缩、可解释性分析）完全失效。
- **缺乏系统的随机黎曼优化框架**：已有 TTN 优化工作多为全批次设置，缺少能在小批量训练下运行、同时尊重商流形结构的自适应与学习率无关方案。

## 核心贡献（创新点）
1. **系统推导 TTN 流形与商流形上的随机黎曼优化算法族**：将梯度投影至 Stiefel 切空间与水平空间，并配套相应的向量输运与重traction，得到可在小批量下逐节点计算的更新规则。
- 与前人基于自动微分+欧式优化器的方案相比，本文保留 TTN 的几何结构，确保迭代轨迹始终落在正交 TTN 流形或其商流形上。

2. **首次将自适应优化（RADAM、RMUON）与距离梯度法（RDOG、RMUONDOG）移植到 TTN 的商流形上**：利用 TTN 参数流形 $\mathcal{T}$ 的笛卡尔积结构以及商流形 $\mathcal{T}/\mathcal{G}$ 与 Grassmann 乘积流形共享测地线喷流这一关键性质，给出节点级别的距离计算与步长自适应。
- 与仅在总空间 $\mathcal{E}$ 上做 Euclidean 优化的方法不同，上述四种算法均明确处理商自由度。

3. **证明 RADAGRAD 在 TTN 商流形上的 regret 界（Appendix B）**：在测地凸性与梯度有界的假设下，推导出 $R_K \leq \left(\frac{D_\infty^2}{2\alpha}+\alpha\right)\sum_{t\in T}\sqrt{\sum_{k=1}^K\|G_k^{(t)}\|^2}$，为后续更复杂算法（RADAM、RMSGRAD）的收敛分析提供模板。
- 这在以往的 TTN 优化文献中是首次给出带商结构约束的随机优化收敛保证。

4. **提出 CNN–TTN 混合架构并应用于中大规模视觉数据集**：将图像划分为 $p\times p$ 块，每块独立经过 CNN 提取 64 维特征，TTN 作为分类头，既保留 TTN 的可解释性/可压缩性，又克服纯 TTN 缺乏平移不变性的缺陷。
- 此前的 TTN 视觉工作（如 [14, 20]）多在小分辨率数据集上做端到端实验，本文首次在 CIFAR10 / Imagenette 级别验证随机黎曼优化。

5. **揭示无约束优化对下游可压缩性的系统性破坏**：通过 Fashion-MNIST、CIFAR10 与 Imagenette 三组实验表明，ADAM 训练后根张量范数在仅 6 个 epoch 后便超出 float32 范围，导致压缩完全失败；而所有 Riemannian 优化器能稳定维持总范数。
- 这一发现补全了 TTN 作为 ML 模型的一个关键算法优势——"可训练即需可压缩"。

## 方法详解
- **TTN 参数流形与正交化**：TTN 参数空间 $\mathcal{T}$ 由 Stiefel 流形的笛卡尔积组成（非根节点为正交等距映射 $B^{(t)}\in\mathrm{St}(\chi_{t_L}\chi_{t_R},\chi_t)$，根节点为普通矩阵）。通过 isometrization 可将任意非正交 TTN 映射到 $\mathcal{T}$，不影响张量收缩结果。
- **商结构与水平空间**：TTN 在每个虚拟键 $\nu_t$ 处存在 $O(\chi_t)$ 规范自由度，构成李群 $\mathcal{G}$。将 $\mathcal{T}$ 对 $\mathcal{G}$ 取商得到 $\mathcal{T}/\mathcal{G}$。水平空间取笛卡尔水平空间 $H_\theta\mathcal{T}=\prod_t H_{B^{(t)}}\mathrm{St}$，满足 $X^TV=0$。
- **欧式梯度与黎曼梯度**：使用反向传播在 TTN 上计算欧式梯度 $\nabla f(\theta)=(\bar{B}^{(t)})_{t\in T}$，其中 $\bar{B}^{(t)}=\bar{\phi}^{(t)}_{\nu_t}\phi^{(t_L)}_{\nu_{t_L}}\phi^{(t_R)}_{\nu_{t_R}}$。黎曼梯度为投影 $\operatorname{grad}f(\theta)=\mathrm{Proj}_\theta(\nabla f(\theta))$；商流形梯度对应水平投影 $\mathrm{Proj}_\theta^H(\nabla f(\theta))$。
- **Retraction 与向量输运**：采用 POGO [26] 近似极分解重traction；向量输运为简单投影 $\nabla_{\theta\to\theta'}^{\mathcal{T}}(\xi)=\mathrm{Proj}_{\theta'}(\xi)$、$\nabla_{\theta\to\theta'}^{H}(\xi)=\mathrm{Proj}_{\theta'}^{H}(\xi)$。
- **距离公式**：
  - 在 $\mathcal{T}$ 上：$d^{\mathcal{T}}(\theta,\theta')^2=\sum_{t\in T^*}d^{\mathrm{St}}(B^{(t)},C^{(t)})^2+\|B^{(N)}-C^{(N)}\|_F^2$，用下界 $q_k(\|X-Y\|_F)=2\sqrt{k}\arcsin(\|X-Y\|_F/(2\sqrt{k}))$ 近似 Stiefel 距离。
  - 在 $\mathcal{T}/\mathcal{G}$ 上：$d^{\mathcal{T}/\mathcal{G}}([ \theta],[\theta'])^2=\sum_{t\in T^*}d^{\mathrm{Gr}}([B^{(t)}],[C^{(t)}])^2+\|B^{(N)}-C^{(N)}\|_F^2$，其中 Grassmann 距离 $d^{\mathrm{Gr}}([X],[Y])=\sqrt{\sum_i\arccos(\sigma_i(X^TY))^2}$。
- **关键定理**（Appendix A）：$\mathcal{T}/\mathcal{G}$ 与 Grassmann 乘积流形 $\mathcal{Z}$ 共享测地线喷流，故距离公式 (17) 成立，指数映射可写成 $\mathrm{Exp}_{[\theta]}=\pi\circ R_\theta^{\mathrm{Exp}}\circ\operatorname{lift}_\theta$。

## 实验与结果
- **数据集**：Fashion-MNIST（60k/10k，压缩至 16×16，256 个输入，$\chi_{\max}=16$）、CIFAR10（32×32，4×4/8×8 分块，16 个 site，$\chi_{\max}=12$）、Imagenette（160×160，8×8 分块，64 个 site，$\chi_{\max}=64$，树深 32）。
- **基线**：标准 Euclidean ADAM；本文 RADAM、RDOG、RMUON、RMUONDOG（均在 $\mathcal{T}$ 和 $\mathcal{T}/\mathcal{G}$ 上实现）。
- **超参**：batch size=256；momentum 参数 $\beta_1=0.9,\beta_2=0.999$ 固定；学习率通过网格搜索选取（Fashion MNIST：$\alpha^{\mathrm{ADAM}}=0.001,\alpha^{\mathrm{RADAM}}=0.03,\alpha^{\mathrm{RMUON}}=0.007,\varepsilon^{\mathrm{RDOG}}=0.1,\varepsilon^{\mathrm{RMUONDOG}}=0.1$）。
- **Fashion-MNIST**：30 个 epoch 后所有优化器测试精度均约 ~89%；RMUONDOG 训练损失最低，其次 RADAM、ADAM、RMUON；RDOG 收敛最慢但最终精度相当；ADAM 在 6 个 epoch 后根张量范数超出 float32 范围，压缩完全失效。
- **CIFAR10 & Imagenette**：所有优化器测试精度均略超/略低于 ~80%；ADAM 速度最快（无需投影/重traction）；Riemannian 优化器压缩后可保持接近原始精度，而 ADAM 模型经正交化后不可压缩（Imagenette 因树更深而问题更严重）。
- **最强结果与提升幅度**：三类数据集的预测精度上 Riemannian 优化器与 ADAM 基本持平（Fashion-MNIST 89%，CIFAR10/Imagenette ~80%）；真正的提升在于下游可压缩性——ADAM 在 Fashion-MNIST 上 3 个 epoch 后已不可压缩，而 Riemannian 方法可压缩至极低参数量且精度几乎无损。

## 相关工作脉络
- **DMRG 系列优化器**：传统物理方法 [9] 为全批次、局部交替优化，假设固定目标函数，无法直接用于 ML 的随机批量设定；本文以随机梯度的视角重写该流程。
- **自动微分+通用优化器**：如 TensorNetwork 库 [19] 及后续工作 [14, 20] 将 TTN 视为普通参数化网络接入 PyTorch/TensorFlow 框架训练，但忽略了几何结构；本文与其对比，强调"几何感知"训练带来的下游可压缩性优势。
- **TTN 黎曼优化（全批次）**：Willner et al. [24] 建立了 TTN 商流形的黎曼几何基础；Hauru et al. [23] 研究了等距张量网络的优化；本文在此基础上扩展到随机小批量训练。
- **MUON 与正交优化**：MUON [6] 在 Euclidean 神经网络隐藏层上用极分解正交化更新；POGO [26] 提出 GPU 友好的近似极分解；本文将这些思想推广至 Stiefel/Grassmann 流形上的 TTN 节点。
- **距离梯度 DOG**：Ivgi et al. [7] 提出免学习率调度；Dodd et al. [33] 将其推广到黎曼流形；本文首次结合 DOG 与 TTN 商流形的测地距离。
- **张量网络机器学习应用**：Felser 等 [16, 17, 18] 已在高能物理、SAR 分类等任务中展示 TTN 模型的可解释性和可压缩性优势，本文从优化器层面为这类应用提供算法保障。

## 局限性与未来方向
- **计算效率差距**：Riemannian 优化器需要投影和 POGO retraction，运行时间显著长于 ADAM，尚未达到"可忽略额外开销"的程度。
- **深层 TTN 数值不稳定加剧**：实验显示树更深（Imagenette 树深 32 vs CIFAR10 树深 12）时 ADAM 的不可压缩问题更严重，根张量范数爆炸机制尚需理论澄清。
- **商流形距离计算的近似性**：Stiefel 测地距离仅用下界近似，Grassmann 距离虽精确但涉及 SVD，批量训练中仍有加速空间。
- **固定键维度假设**：当前算法假设键维度 $\chi_t$ 固定，未考虑类似两站点 DMRG 的自适应键维方法，这是未来一个有吸引力的方向。
- **代码暂未公开**：论文声明代码将在期刊接受后开源，现阶段复现依赖自行实现。

## 研究启发与可借鉴点
- **"几何感知的优化器即下游任务的保险"**：对任何依赖低秩/正交结构的张量网络模型（MPS、TTN、Hierarchical Tucker），随机黎曼优化应作为默认训练范式，避免无约束优化带来的数值灾难。
- **商流形距离与 Grassmann 乘积流形的测地等价性**可作为一类新的理论工具——凡涉及规范自由度的张量网络优化，均可借鉴 Appendix A 的论证思路，将复杂的商距离拆解为节点级 Grassmann 距离的欧氏组合。
- **CNN–TTN 混合架构的设计思路**具有可迁移价值：用 CNN 提取局部特征，TTN 负责跨块高阶交互，兼顾平移不变性与可解释性，可应用于其他需结构化特征融合的视觉或序列任务。
- **POGO 近似极分解 retraction** 作为 GPU 友好的正交投影方案，可直接复用于任何涉及 Stiefel/Grassmann 流形的随机优化场景。
- **节点级 DOG 步长调度**（每核心张量维护独立 $r_k^{(t)}$ 和 $n_k^{(t)}$）比全局步长更灵活，适合多尺度张量网络，值得探索是否可推广至 Matrix Product Operator 等其他 TN 格式。

## 关键术语表
- **Tree Tensor Network (TTN)**：一种层次化张量分解结构，以有根二叉树组织核心张量，逐层合并虚拟索引，常用于高效表示高维张量。
- **Stiefel manifold $\mathrm{St}(n,k)$**：$\mathbb{R}^{n\times k}$ 中所有列标准正交矩阵构成的流形，TTN 非根节点的核心张量受此约束。
- **Grassmann manifold $\mathrm{Gr}(n,k)$**：$\mathbb{R}^n$ 中所有 $k$ 维线性子空间的集合，可表为 $\mathrm{St}(n,k)/O(k)$ 商流形。
- **Quotient manifold $\mathcal{T}/\mathcal{G}$**：TTN 正交参数流形关于规范群 $\mathcal{G}$ 作用的商空间，刻画了 TTN 唯一对应的张量值映射。
- **Horizontal space**：切空间中与规范轨道（垂直空间）正交的子空间，商流形梯度经水平提升后可在参数流形上显式计算。
- **Retraction**：流形上从切向量返回流形点的局部微分同胚映射，是黎曼优化迭代中替代指数映射的实用工具。
- **Distance over Gradients (DOG)**：一种免学习率调度策略，通过累计历史梯度范数与迭代间测地距离动态计算步长。
- **POGO retraction**：Javaloy & Vergari 提出的近似极分解重traction，避免严格正交化带来的计算开销，利于 GPU 加速。

## 可复现要素
- **数据集**：Fashion-MNIST、CIFAR10、Imagenette 均为公开数据集；Imagenette 可通过 https://github.com/fastai/imagenette 获取。
- **代码**：论文声明"full code will be released upon journal acceptance"，截至本文发表时尚未开源。
- **关键超参**：batch size=256；$\beta_1=0.9,\beta_2=0.999$；初始学习率/步长估计见正文表 1；retraction 使用 POGO [26]。
- **硬件**：NVIDIA A30 GPU，单精度。
- **初始化**：采用 [29] 的无监督构造算法确定初值，所有优化器共用同一初始迭代点。
