---
title: "MOSPR-HISTOLOGY-TO-GENE-EXPRESSION-PREDICTION-WITHMORPHO-SPA"
source: https://arxiv.org/pdf/2609.34280v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 09:55:25"
field: "计算病理学与多组学整合"
keywords: ["histology-to-gene expression", "whole-slide image", "morpho-spatial macrostates", "low-rank regression", "computational pathology", "TCGA", "multiple instance learning"]
innovations: ["Cohort 级形态邻接聚合构建共享宏观状态字典", "线性框架耦合形态-空间表征与低秩分子基实现精确可解释分解"]
benchmarks: ["TCGA-BRCA", "TCGA-KIRC", "TCGA-LUAD", "Hallmark", "GO-BP", "KEGG"]
---

## 论文速读：MoSPR — Histology-to-Gene Expression Prediction with Morpho-Spatial Macrostates and Low-Rank Molecular Programs

## 一句话总结
MoSPR 提出了一种线性框架，通过将冻结的病理编码器 patch 嵌入聚类为形态微状态、以空间邻接关系构建 cohort 级宏观状态，并以此表征 WSI，再用低秩分子基进行岭回归，从而在 TCGA 乳腺癌、肾透明细胞癌和肺腺癌三个队列上实现了最优的全转录组基因表达预测。

## 研究问题与动机
1. **从 H&E 组织学预测分子谱**是癌症诊断与分层的低成本替代方案，但 WSIs 包含高度异质性且空间组织的组织形态，而基因表达包含数千个高度相关的靶标，直接回归困难。
2. **现有方法的局限**：全局池化和标准 MIL（如 ABMIL）仅聚合 patch 级别的形态，未显式编码空间邻接关系；Transformer/图方法（如 TransMIL、Patch-GCN）依赖患者特异的 patch 序列或图，无法共享跨患者的形态结构。
3. **缺乏可扩展的低秩预测**：直接回归数千个基因的计算成本高，且基因间强相关性未被利用。
4. **低资源场景下的需求**：配对组织学-RNA 数据稀缺，如何在有限标注下保持性能是一大挑战。

## 核心贡献（创新点）
1. **Cohort 级形态字典学习**：通过 patch 邻接矩阵在全队列上学习共享的宏观状态，无需 spatial-omics 监督，也无需构建患者特异的图。
2. **线性框架耦合低秩分子基**：将形态-空间表征与训练派生的低秩基因表达基相耦合，实现从 $G$ 维基因空间到 $q \ll G$ 维系数空间的降维回归。
3. **精确的可解释分解**：线性公式使得每个预测分子谱可精确分解为全局形态贡献与宏观状态特异性贡献，无需后验归因方法。
4. **系统性跨癌种验证**：在三个 TCGA 队列上实现了基因层面和通路层面的 SOTA，且在低资源场景（半数训练数据）下超越最强基线的满数据表现。

## 方法详解

**Patch 特征提取（§3.1）**：使用冻结的 CONCH 病理编码器（Lu et al., 2024），将每个 patch $x_i^{(n)}$ 映射为 $z_i^{(n)} = f_{enc}(x_i^{(n)}) \in \mathbb{R}^{512}$。

**微观状态→宏观状态构建（§3.2）**：
- 对训练 patch 嵌入做 k-means（$J=200$）得到微状态 centroid $\{\mu_j\}_{j=1}^{J}$，每个 patch 分配至最近微状态 $a_i^{(n)}$。
- 对每个患者计算 8-邻域微状态邻接矩阵 $C^{(n)} \in \mathbb{R}^{J \times J}$，按 $(u,v)$ 相邻次数累加（对称）。
- 每个患者的邻接矩阵按其总质量归一化 $\widehat{C}^{(n)}$，再对全训练队列取平均得到全局邻接矩阵 $C_{global}$。
- 对 $C_{global}$ 行归一化得到转移矩阵 $P$，计算其特征分解 $P\psi_\ell = \lambda_\ell\psi_\ell$。
- 取前 $L=20$ 个非平凡谱分量，按特征值加权得到谱嵌入 $\phi_u = [\lambda_1\psi_1(u), \ldots, \lambda_L\psi_L(u)]$，归一化后做 k-means（$K=8$）得到宏观状态映射 $\rho: \{1,\ldots,J\} \to \{1,\ldots,K\}$。

**形态-空间低秩回归（§3.3）**：
- 全局形态向量：$M_n = \frac{1}{N_n}\sum_i z_i^{(n)}$。
- 每个宏观状态 $k$ 的均值特征 $h_{nk}$ 和丰度加权偏差 $s_{nk} = p_{nk}(h_{nk} - M_n)$，拼接为 $S_n \in \mathbb{R}^{K D_0}$。
- 组织学表征 $X_n = [M_n \mid S_n] \in \mathbb{R}^{(K+1)D_0}$。
- 对标准化基因表达 $Y_z$ 做 PCA，取前 $q$ 主成分得基 $U_q \in \mathbb{R}^{q \times G}$，系数矩阵 $A_q = Y_z U_q^\top$。
- 岭回归：$W_q^* = \arg\min_{W_q} \|A_q - X W_q\|_F^2 + \lambda_{ridge}\|W_q\|_F^2$。
- 预测：$\widehat{Y}_z = X W_q^* U_q$。

**宏观状态分子可解释性（§3.4）**：$W_q^*$ 按块分解为全局权重 $W_{q,global}^*$ 和各宏观状态权重 $W_{q,k}^*$，则 $\widehat{Y}_{z,n} = \Gamma_{n,global} + \sum_k \Gamma_{nk}$，其中 $\Gamma_{nk} = s_{nk} W_{q,k}^* U_q$。

## 实验与结果

**数据集**：TCGA-BRCA（1,037 患者/1,467 切片/14,042 基因）、TCGA-KIRC（342 患者/681 切片/14,295 基因）、TCGA-LUAD（482 患者/756 切片/14,514 基因），四折患者级交叉验证。

**评估指标**：基因层面——每个基因的 Pearson/Spearman 相关系数平均；通路层面——Hallmark、GO-BP、KEGG 三条路径集合的通路 SCC 中位数。

**主要结果（表 1）**：
| 队列 | MoSPR PCC | MoSPR SCC | vs 最强基线增益 |
|------|-----------|-----------|-----------------|
| BRCA | **0.413** | **0.411** | +0.043/+0.048（CPNN 0.360/0.356） |
| KIRC | **0.334** | **0.348** | +0.022/+0.029（CPNN 0.293/0.310） |
| LUAD | **0.358** | **0.376** | +0.039/+0.040（MambaMIL 0.319/0.335） |

**通路层面（表 2）**：九次比较中八次通路 SCC 排名最优。BRCA Hallmark 达 **0.551**（vs ABMIL 0.500）。

**消融（表 3，BRCA）**：
- Global Direct → Spatial Direct：+0.047（宏观状态表征贡献）
- Global Direct → Global Low-Rank：+0.032（低秩输出贡献）
- Shuffled-State Low-Rank（0.379）远低于 Spatial Low-Rank（0.411），验证邻接派生宏观状态的价值

**数据效率（图 2）**：BRCA 上仅用约 **44%** 训练数据即可达到 ABMIL 满数据的基因预测水平（SCC ≈ 0.36），约 **48%** 即达 ABMIL 满数据 Hallmark 水平。

**模型效率**：MoSPR 仅 **0.96M 参数**（最低），推理仅需 **2.02 GFLOPs**（CONCH 冻结后）。

## 相关工作脉络
1. **HE2RNA (Schmauch et al., 2020)**：首个 H&E→RNA 预测方法，基于 patch 级 CNN 聚合，未建模空间邻接。
2. **tRNAformer (Alsaafin et al., 2023)**：Transformer 架构预测 RNA 序列表达，使用患者特异的 patch 序列。
3. **MOSBY (Şenbabaoglu et al., 2024)**：多组学推断与空间生物标志物发现，依赖 attention 机制。
4. **SEQUOIA VIS (Pizurica et al., 2024)**：线性化注意力用于基因表达数字分析，仍为深度学习参数化方法。
5. **CPNN (Nishimura et al., 2026)**：引入单细胞衍生原型融入分子结构，参数量 1.51M。
6. **ABMIL (Ilse et al., 2018)** / **MambaMIL (Yang et al., 2024)**：通用 WSI 聚合方法，作为非领域专用基线对比。
7. **MoSPR 的定位差异**：以 cohort 级邻接构建共享形态字典而非患者特异图，以线性回归替代深度参数化网络，兼具高效性与可解释分解能力。

## 局限性与未来方向
1. **宏观状态仅捕获局部邻接模式**而非完整二维组织几何，无法完全还原空间结构。
2. **bulk RNA 无法验证宏观状态关联的分子程序是否在对应组织区域空间局域化**，生物学表征目前仅定性。
3. **仅在 TCGA 内部验证**，未建立跨独立队列/机构的泛化性，外部验证必要。
4. **潜在的批次效应与染色差异**：不同实验室的扫描和染色变异可能影响模型鲁棒性。

## 研究启发与可借鉴点
1. **Cohort 级邻接聚合策略**：将空间邻接从患者特异层面提升到 cohort 层面，为跨样本共享结构化表示提供了简洁且可复现的范式，可迁移到其他多实例学习场景。
2. **低秩输出回归与形态表征解耦**：先用谱聚类学习结构化输入，再用低秩 PCA 基压缩输出，两条独立的降维策略各司其职，可借鉴于高维回归任务。
3. **精确线性分解用于可解释性**：线性模型天然支持贡献分解（全局 vs 状态特异），无需 SHAP/LIME 等事后归因，在需要生物学解释的任务中具有优势。
4. **低资源高效性**：half-data 即超越 full-data 基线的结果，提示邻接先验能有效缓解标注数据稀缺问题。
5. **CONCH 冻结特征的普适性**：证明预训练病理 Foundation Model 的冻结嵌入可与轻量下游模块配合取得最优性能，避免端到端微调的计算成本。

## 关键术语表
**MoSPR（Morpho-Spatial Program Regression）**：本文提出的线性框架，通过形态-空间宏观状态与低秩分子基联合建模实现 H&E→基因表达预测。
**Morphology Microstate（形态微状态）**：通过 k-means 聚类冻结 patch 嵌入得到的细粒度形态原型（$J=200$）。
**Macrostate（宏观状态）**：基于全局微状态邻接矩阵的谱嵌入聚类得到的粗粒度组织状态（$K=8$）。
**Low-Rank Molecular Basis（低秩分子基）**：对标准化基因表达做 PCA 得到的前 $q$ 个主成分方向 $U_q$。
**Spectral Embedding（谱嵌入）**：对归一化邻接矩阵的特征分解结果，用于描述微状态间的邻接结构相似性。
**Hallmark Pathway（Hallmark 通路）**：MSigDB 中的 50 条高度特异的生物学通路集合，用于通路级评估。
**ABMIL（Attention-Based MIL）**：基于注意力的多实例学习基线方法。
**PCC / SCC**：Pearson 相关系数 / Spearman 等级相关系数，分别衡量线性关联与单调关系。

## 可复现要素
- **数据集**：TCGA-BRCA/KIRC/LUAD 为公开数据，患者级四折划分随代码仓库提供（Supplementary Table S1）。
- **代码**：已开源，地址 https://github.com/Radisen-Panthera/MoSPR，含完整训练脚本、预处理流程、超参搜索及表格/图表生成脚本。
- **预训练编码器**：CONCH（Lu et al., 2024），冻结使用。
- **关键超参**：$J=200$（微状态数）、$K=8$（宏观状态数）、$L=20$（谱分量数）；$q \in \{4,8,16,32,48,64,96,128\}$ 和 $\lambda_{ridge} \in \{1,10,10^2,10^3,10^4,10^5\}$ 通过验证集 MSE 搜索。
- **基因预处理**：log1p-transformed CP10K（每样本归一化至 $10^4$ 后 log 变换），再按基因做训练集标准标准化。
