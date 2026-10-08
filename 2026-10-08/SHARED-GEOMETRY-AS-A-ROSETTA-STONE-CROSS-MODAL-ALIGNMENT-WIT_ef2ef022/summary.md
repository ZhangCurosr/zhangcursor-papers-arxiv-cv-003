---
title: "SHARED-GEOMETRY-AS-A-ROSETTA-STONE-CROSS-MODAL-ALIGNMENT-WIT"
source: https://arxiv.org/pdf/2610.09411v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 09:54:20"
---

# 论文速读：Shared Geometry as a Rosetta Stone: Cross-Modal Alignment Without Paired Data

## 一句话总结
本文证明独立训练的单模态嵌入空间若共享足够的几何结构，即可在无一对配对数据的情况下实现跨模态粗略对齐。提出的 Wasserstein Procrustes 算法配合几何初始化，在零配对及极少配对（≤100）场景下显著优于现有方法，且对齐成功率可被 CKA 等几何度量强预测。

## 研究问题与动机
1. **核心问题**：在不观测任何跨模态对应样本的前提下，能否仅凭嵌入空间的几何结构完成跨模态对齐？
2. **现有方法局限**：主流多模态学习严重依赖大规模图文/跨模态配对数据；即便少配对（few-pair）方法仍需若干锚点对作为参考坐标系。
3. **理论动机**：“柏拉图表示假说”指出随规模增长，不同模态的独立模型会自发收敛至共享表示几何；前人已观察到跨模态间存在显著的粗略几何一致性。
4. **现实缺口**：此前完全无配对的方法仅局限于小规模数据集或类聚合嵌入，尚未解决两个完全独立、无共享坐标系的大规模嵌入集对齐问题。

## 核心贡献（创新点）
1. **提出零配对跨模态对齐新范式**：证明仅需嵌入空间的共享几何即可恢复跨模态粗略对应，打破“对齐必须依赖配对数据”的传统假设。
2. **设计带几何初始化的 Wasserstein Procrustes 算法**：通过 k-means 聚类匹配与 QAP 求解粗略对应，再经交替指派与正交 Procrustes 细化，全程无需配对、类别标签或共享参考点。
3. **揭示几何相似度可预测对齐成功率**：CKA 等通用几何度量与无配对对齐质量呈强负相关，为判断哪些模型/模态组合适合无配对对齐提供了实证准则。
4. **自然平滑扩展至极少配对 regime**：已知配对以加权线性项融入初始化与细化阶段，在 ≤100 对时以 14×~28× 优势超越现有 few-pair 基线。
5. **跨领域泛化验证**：在视觉-语言、自然语言、单细胞组学、fMRI 神经科学及 11 个科学模态对上均取得有效对齐，证明方法不依赖特定领域设计。

## 方法详解
- **联合优化目标**：给定中心化的独立嵌入矩阵 $\boldsymbol{X} \in \mathbb{R}^{n \times d_X}$ 与 $\boldsymbol{Y} \in \mathbb{R}^{m \times d_Y}$，联合估计半正交映射 $W \in \mathrm{St}(d_X, d_Y)$ 与运输计划 $T \in \Pi(\boldsymbol{a}, \boldsymbol{b})$：
  $$W^{WP}, T^{WP} = \arg\max_{W, T} \mathrm{Tr}(\boldsymbol{X} W \boldsymbol{Y}^\top T^\top)$$
  固定 $T$ 时 $W$ 有 Procrustes 闭式解，固定 $W$ 时 $T$ 退化为线性指派，但联合问题非凸，随机初始化无效。
- **粗略几何初始化**：借鉴 mini-vec2vec，对两空间独立随机采样批数据并执行 k-means 聚类（$C=30$）。通过最大化线性 CKA 求解簇中心间的二次分配问题（QAP），采用 MPOpt+GRASP 启发式求解器获得低秩近似运输计划 $M = \frac{1}{SC}\sum_s \boldsymbol{A}_s^\top P_s \boldsymbol{B}_s$。
- **读取与细化**：对平均计划执行 Gromov-Wasserstein 目标的一阶条件梯度步，退化为对得分矩阵 $\boldsymbol{X} M \boldsymbol{Y}^\top$ 的线性指派。为避免 $O(n^3)$ 复杂度，采用随机分块策略并行求解匈牙利匹配，再经极分解初始化 $W$。随后在 mini-batch 上交替执行线性指派与正交 Procrustes 更新（$R=100$ 轮）。
- **极少配对扩展**：已知配对 $(\hat{\boldsymbol{X}}, \hat{\boldsymbol{Y}})$ 在 QAP 阶段作为线性项融入 CKA 优化，在 Procrustes 阶段以同等总权重加入更新步，使算法在零配对与少量配对之间连续插值。

## 实验与结果
- **数据集与模型**：视觉模型（iBOT, DINOv2, Franca, DINOv3）；语言模型（all-mpnet-base-v2, Qwen3-Embedding-8B, Qwen3-8B）；评测集含 MS COCO、SPC、DCI、DOCCI、NQ、PBMC、NSD fMRI 及 11 个跨学科模态对。基线包括 vec2vec、mini-vec2vec、SCOT+、Platonic Brain、ASIF、STRUCTURE、SOTAlign 等。
- **零配对对齐**：MS COCO 上平均 FOSCTTM 达 **0.154**（对比 mini-vec2vec 的 0.223），超越 17/21 组合；跨数据集（COCO 图像 + SPC 文本）平均为 **0.236**。零样本分类在 CIFAR-10
