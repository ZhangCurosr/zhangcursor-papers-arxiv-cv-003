---
title: "Stochastic-Optimization-of-Tree-Tensor-Networks"
source: https://arxiv.org/pdf/2609.00870v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 22:21:53"
field: "张量网络与黎曼优化"
keywords: ["tree tensor network", "stochastic Riemannian optimization", "quotient manifold", "distance over gradients", "MUON", "model compression"]
innovations: ["在 TTN 参数流形与商流形上推导随机黎曼自适应与免超参优化器", "证明 TTN 商流形测地线喷流与 Grassmann 笛卡尔积共享结构以支持距离计算", "提出 CNN-TTN 混合架构并验证黎曼优化对下游压缩的数值稳定性优势"]
benchmarks: ["Fashion-MNIST", "CIFAR10", "Imagenette"]
---

# 论文速读：Stochastic-Optimization-of-Tree-Tensor-Networks

## 一句话总结
本文推导了适用于树张量网络（TTN）参数流形与商流形的随机黎曼优化器（包括自适应与免超参方案），并结合 CNN–TTN 混合架构在 Fashion-MNIST、CIFAR10 与 Imagenette 上验证：所提优化器在预测性能上与无约束 Adam 相当，但保持的迭代点具备更好的数值稳定性，从而显著改善下游压缩等任务。

## 研究问题与动机
- 核心问题：TTN 作为机器学习模型时，传统来自物理的确定性优化器（如 DMRG）难以直接适配基于大批量打乱数据的经验风险最小化场景。
- 现有主流做法通常通过自动微分把 TTN 当作一般参数函数训练，忽略其流形几何结构，容易导致迭代点偏离正交/等距约束。
- 忽略商流形与正交结构会使 TTN 整体范数在训练中爆炸，进而使依赖 SVD/正交化的下游任务（如模型压缩、可解释性度量）失效。
- 需要在 TTN 的参数流形与商流形上建立可小批量计算的随机一阶方法，兼顾收敛性能与后续压缩可用性。

## 核心贡献（创新点）
- 在 TTN 参数流形与商流形上推导了随机黎曼梯度更新规则，支持小批量自适应与无学习率方案。区别在于显式利用 TTN 的几何结构与水平提升，而非将分解视为普通欧氏参数。
- 提出 RADAM、RDOG、RMUON 及 RMUONDOG 等算法，适配 TTN 的可分离块坐标与商流形水平空间。与以往仅在满批次下讨论的黎曼优化相比，本文面向随机 minibatch 训练。
- 证明 TTN 商流形的测地线喷流与 Grassmann 流形的笛卡尔乘积共享几何结构，从而给出商流形上距离与指数映射的可计算表示。该理论支撑了距离过梯度方案的可行性。
- 设计了一种保留 TTN 可解释性的 CNN–TTN 混合架构：CNN 按 patch 独立提取特征，TTN 负责跨 patch 的高阶组合。与直接用 TTN 处理大图像相比，更适合 CIFAR10/Imagenette 等任务。
- 实验表明，黎曼优化在测试集准确率上与 Adam 可比，但能保持 TTN 整体范数稳定；相比之下，Adam 训练的深层 TTN 在压缩时几乎完全失效。

## 方法详解
- 模型与流形：采用正交 TTN，除根节点外所有核心张量为 Stiefel 流形元素，满足等距约束；根张量为欧氏空间。整体参数流形为各核张量流形的笛卡尔积。
- 商流形与规范自由度：在虚拟键上可插入正交矩阵与其转置而不改变展开后的全张量，形成 Lie 群作用；通过水平空间 $H_\theta \mathcal{T}$ 与水平提升将商流形梯度映射回参数流形实现隐式优化。
- 投影与重traction：使用切空间投影 $P_X(V)=V-\frac{1}{2}X(X^TV+V^TX)$ 与水平投影 $P_X^H(V)=(I-XX^T)V)$；重traction 可选极分解的 polar 形式、QR 形式，或实验采用的近似但 GPU 友好的 POGO 方案。
- 梯度计算：前向递归计算各节点特征 $\phi^{(t)}$，反向传播得到欧氏梯度 $\bar{B}^{(t)}=\bar{\phi}^{(t)}\phi^{(t_R)}\phi^{(t_L)}$；Riemannian gradient 为 Euclidean gradient 的切空间投影，商流形方向进一步投影到水平空间。
- 向量传输：采用简单投影式传输 $\nabla_{\theta\to\theta'}^{\mathcal{T}}(\xi)=\mathrm{Proj}_{\theta'}(\xi)$ 与 $\nabla_{\theta\to\theta'}^{\mathcal{T}/\mathcal{G}}(\xi)=\mathrm{Proj}_{\theta'}^H(\xi)$，便于实现。
- RADAM：在每个核张量块上维护一阶矩与二阶矩估计，并按 $\delta B_k^{(t)}=-\alpha m_k^{(t)}/\sqrt{\nu_k^{(t)}}$ 更新，再通过重traction 返回流形。
- RDOG：为每个核张量维护步长估计 $r_k^{(t)}$ 与累积梯度范数 $n_k^{(t)}$，更新为 $\delta B_k^{(t)}=-r_k^{(t)}G_k^{(t)}/\sqrt{\zeta_\kappa(r_k^{(t)})\,n_k^{(t)}}$；根节点用 Frobenius 距离，非根节点在商流形上用 Grassmann 距离、在参数流形上用 Stiefel 下界。
- RMUON/RMUONDOG：将每个块梯度做极分解正交因子 qf 操作以实现 MUON 风格更新，并可叠加动量或 DOG 式动态步长；正交化后的方向仍位于水平空间，兼容商流形优化。

## 实验与结果
- 实现与硬件：基于 PyTorch，单卡 NVIDIA A30，单精度；minibatch=256，损失为平方误差；初始参数由无监督构造算法得到，各优化器共享同一起点。
- Fashion-MNIST：将 $28\times28$ 图像压缩为 $16\times16$，直接使用像素输入（$n=256,\, d_i=2$），$\chi_{\max}=16$。30 epoch 后各优化器测试准确率均约 89%；RMUONDOG 训练损失最低，RDOG 收敛最慢但最终准确率相近。关键差异在于 ADAM 会导致 TTN 整体范数爆炸并难以压缩，而黎曼优化保持范数稳定、压缩精度显著更好。
- CIFAR10 与 Imagenette：采用 CNN 特征提取器 + TTN 分类头，CNN 对每个 patch 独立处理以避免 patch 间信息交换破坏 TTN 可解释性。各优化器在 CIFAR10 上测试准确率略超 80%，在 Imagenette 上略低于 80%，与 Adam 性能可比；ADAM 在更深的 Imagenette 树结构上导致模型不可压缩，而黎曼优化仍可实现压缩。
- 总体结论：黎曼优化在预测指标上与 Adam 相当，但在下游依赖正交结构的任务中具备明显优势；Adam 在深层 TTN 上因范数爆炸会导致压缩完全失效。

## 相关工作脉络
- DMRG 等物理起源的确定性 Sweeping 优化：假设目标固定且可精确构建局部环境；本文针对随机 minibatch 与不可精确求全梯度的机器学习场景重新推导。
- 基于自动微分的端到端 TTN/CNN 训练：将 TTN 视作普通参数化函数，忽略流形约束；本文强调几何一致性对数值稳定与压缩的影响。
- 已有的 TT/TTN 黎曼优化（如满批次框架）：本文将其推广到随机优化，并在商流形与参数流形两个层面给出统一更新形式。
- 正交优化中的 POGO 等快速重traction：本文将其引入 TTN 训练以缓解黎曼优化的计算开销，同时保持接近正交流形的迭代轨迹。
- 自适应与免超参优化（ADAM/MUON/DOG）：本文分别将其黎曼化，利用 TTN 的分块结构与 Grassmann 距离实现适配。
- 可解释与压缩导向的 TTN 应用：本文强调若迭代点破坏正交结构，则 SVD 类压缩与相关可解释性度量会失效，从而突出黎曼优化的必要性。

## 局限性与未来方向
- 黎曼优化仍需投影/重traction/向量传输等额外计算，训练时间仍长于 Adam；效率差距在更深网络中可能更显著。
- 当前工作在固定键维度下进行；自适应键维度可在训练与压缩间取得更好平衡，但会改变流形结构，传统黎曼方法不再直接适用。
- 深层 TTN 对非正交扰动的放大效应更强，Adam 在更深结构上压缩失败更严重；需要更系统的稳定性分析与正则策略。
- 代码目前声明将在期刊接受后开源，短期复现需要等待发布或自行实现。
- 实验主要集中在图像分类任务，尚未广泛验证到高能物理、SAR 等其他已展示的 TTN 应用场景。

## 研究启发与可借鉴点
- 将商流形的水平提升与 Grassmann 距离结合到分层张量模型训练中，是一种“既保留结构又可批量优化”的可迁移思路。
- 通过 CNN 按 patch 独立提取、再由 TTN 做跨 patch 组合的架构，可在可扩展性与可解释性之间取得折中，适合需要溯源的特征分析任务。
- 用 POGO 等近似但 GPU 友好的重traction 替代严格正交化步骤，能在数值稳定性与训练速度之间取得实用平衡。
- 把 MUON 的正交化更新思想引入张量网络各核张量块，可作为无记忆、低开销的替代方案，尤其适合对步长调参敏感的场景。
- 实验设计强调“不仅看测试准确率，还要看下游压缩/可解释性可用性”，为张量网络模型的评估提供了更全面的范式。

## 关键术语表
- **Tree Tensor Network (TTN)**：按 rooted binary tree 结构组装的三阶核张量集合，通过收缩虚拟键得到完整张量。
- **Orthogonal/Isometric TTN**：除根节点外各核张量满足等距约束的 TTN，构成 Stiefel 流形的笛卡尔积子流形。
- **Quotient manifold $\mathcal{T}/\mathcal{G}$**：在虚拟键处去除正交规范自由度后得到的等价类空间，参数映射在此变为单射。
- **Horizontal space**：与规范方向垂直的子空间，用于唯一表示商流形切向量并保证更新方向影响输出张量。
- **Riemannian gradient**：欧氏梯度经切空间（或水平空间）投影后得到的流形下降方向。
- **POGO retraction**：近似极分解的重traction，避免每步严格正交化但仍保持迭代点靠近正交流形。
- **Distance over gradients (DOG)**：基于历史测地线距离与梯度范数的免超参动态步长策略。
- **Feature map $\phi(x)$**：将输入编码为局部张量积形式的映射，使 TTN 可直接作用于扩展后的输入表示。

## 可复现要素
- 数据集：Fashion-MNIST、CIFAR10、Imagenette（均为公开数据集）。
- 代码/权重：论文声明完整代码将在期刊接受后开源；当前未提供预训练权重链接。
- 关键超参：batch size=256；Adam 常用参数 $\alpha=0.001,\beta_1=0.9,\beta_2=0.999$；动量相关参数 $\beta_1=0.9,\beta_2=0.999$；各任务学习率/步长估计通过网格搜索确定（如 Fashion-MNIST 中 $\alpha^{\text{ADAM}}=0.001,\alpha^{\text{RADAM}}=0.03,\varepsilon^{\text{RDOG}}=0.1$ 等，详见文中表格）。
- 初始化和硬件：初始参数由无监督构造算法获得；实验在 NVIDIA A30 GPU 单精度下进行。
