---
title: "Learning-What-to-Trust-in-Multimodal-Learning-under-Noisy-Su"
source: https://arxiv.org/pdf/2610.11057v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 17:20:35"
field: "多模态噪声标签学习"
keywords: ["多模态学习", "噪声标签", "样本选择", "表示学习", "判别分析"]
innovations: ["提出REFINE框架联合融合与单模态表示进行噪声检测", "通过目标-背景判别分析构造判别特征向量扩大清洁与噪声样本的可分性", "设计类自适应的信任空间选择机制"]
benchmarks: ["UPMC-Food101", "N24News", "NWPU-Captions", "VGGSound50", "MIntRec2.0", "BSD"]
---

# 论文速读：Learning-What-to-Trust-in-Multimodal-Learning-under-Noisy-Su

## 一句话总结
本文针对多模态学习中标签噪声下的样本选择问题，提出了REFINE框架，通过融合表示与单模态表示的判别分析、自适应信任空间选择和标签一致性机制，在多种噪声类型和数据集上显著提升了多模态分类性能。

## 研究问题与动机
- **现实痛点**：真实世界多模态数据（图像-文本、音频-视频等）的标注往往存在噪声，而现有方法大多依赖高质量标签进行监督。
- **已有方法不足**：当前噪声标签学习中的样本选择方法主要聚焦于单模态场景，未能充分利用多模态模型提供的多个表示空间（融合空间+各单模态空间）中的互补信息。
- **核心挑战**：不同表示空间对不同类型类别的噪声检测能力存在差异，单一空间无法在所有类别上达到最优选择效果。
- **关键问题**：如何在多模态学习中识别并信任具有更好噪声检测能力的表示空间，从而构建更可靠的标签噪声检测器。

## 核心贡献（创新点）
- **理论分析表示结构与噪声检测的关系**：首次从理论上证明不同表示空间中清洁样本与噪声样本的对齐分数差异与类间夹角相关，且不存在对 все 类别都最优的统一空间。
- **判别特征向量（DE）构造**：受dPCA启发，通过目标类与背景类的判别分析构造加权背景矩阵，生成对目标类结构敏感但对共享背景不敏感的特征向量。
- **信任空间选择（TSS）**：通过两两比较各表示空间的判别能力，为每个类别自适应地选择一组"可信"表示空间，而非依赖单一空间。
- **标签一致性筛选（LA）**：摒弃传统GMM阈值拟合，直接通过目标类对齐分数是否严格大于所有其他类分数来判断样本可靠性。
- **与现有范式的广泛兼容性**：作为即插即用组件，REFINE可无缝集成到Co-teaching、FreeMatch、DSS+及多种噪声容忍损失函数中。

## 方法详解
REFINE框架包含三个核心模块：

**1. 判别分析（Discriminative Analysis）**
- 对每个观察类$k$在空间$r$中，计算非目标类$j$沿目标类FINE主特征向量的平均对齐分数：$d_{kj}^r = (\boldsymbol{u}_{k,\text{FINE}}^r)^\top \boldsymbol{A}_j^r \boldsymbol{u}_{k,\text{FINE}}^r$
- 归一化得到背景权重：$\gamma_{kj}^r = d_{kj}^r / \sum_{\ell \neq k} d_{k\ell}^r$
- 构造加权背景矩阵：$\boldsymbol{B}_k^r = \sum_{j \neq k} \gamma_{kj}^r \boldsymbol{A}_j^r$
- 求解广义特征值问题得到判别特征向量：$\boldsymbol{u}_{k,\text{DE}}^r = \arg\max_{\|\boldsymbol{u}\|=1} \frac{\boldsymbol{u}^\top \boldsymbol{A}_k^r \boldsymbol{u}}{\boldsymbol{u}^\top(\boldsymbol{B}_k^r + \epsilon \boldsymbol{I})\boldsymbol{u}}$
- 实例对齐分数：$s_{i,k}^r = [(\boldsymbol{u}_{k,\text{DE}}^r)^\top \boldsymbol{z}_i^r]^2$

**2. 信任空间选择（Trusted Space Selection）**
- 对每对空间$r,s$和类$k$，构建二维投影表示：$\boldsymbol{p}_{i,k}^{r,s} = [p_{i,k}^r, p_{i,k}^s]^\top$
- 在二维空间中再次执行判别分析，求解广义特征向量：$\boldsymbol{a}_k^{r,s} = (a_k^r, a_k^s)^\top$
- 坐标绝对值较大者获胜：$|a_k^r| > |a_k^s| \iff \Delta_{\text{DE}}^r > \Delta_{\text{DE}}^s$
- 信任空间集合：$\mathcal{R}_k = \{r \in \mathcal{R} : W_k^r \geq L_k^r\}$

**3. 可靠样本过滤（Reliable Sample Filtering）**
- 标签一致性条件：$\mathcal{C}_k^r = \{(x_i, \tilde{y}_i) : \tilde{y}_i = k, s_{i,k}^r > \max_{j \neq k} s_{i,j}^r\}$
- 最终选择集合：$\mathcal{C} = \bigcup_k \bigcup_{r \in \mathcal{R}_k} \mathcal{C}_k^r$

**关键理论定理**：
- Theorem 1：FINE的对齐分数差异$\Delta_{\text{FINE}}^r$随类间夹角$\theta^r$单调递增
- Theorem 2：DE的对齐分数差异$\Delta_{\text{DE}}^r \geq \Delta_{\text{FINE}}^r$，且差距随噪声率$\tau$增大
- Theorem 3：二维空间中坐标大小关系与单空间判别能力排序一致

## 实验与结果
**数据集**：6个合成噪声数据集（UPMC-Food101, N24News, NWPU-Captions, Rakuten France, VGGSound50, MIntRec2.0）+ 1个真实噪声数据集（BSD）。

**噪声类型**：对称噪声（Sym 50%）、非对称噪声（Asym 40%）、全模态实例依赖噪声（F-IDN 40%/50%）、部分模态实例依赖噪声（P-IDN 40%/50%）。

**主要结果**：
- 在全部7个数据集的6种噪声设置下，REFINE均取得最高平均准确率
- UPMC-Food101上：Sym 50%噪声下86.63% vs 次优85.56%（DIST），提升约1.07个百分点
- N24News上：Sym 50%噪声下64.55% vs 次优64.01%（DIST），提升0.54个百分点
- MIntRec2.0（小样本场景）：REFINE在所有设置下均唯一超越Standard基线
- BSD真实噪声数据集：61.03% vs FINE的60.71%，是唯一超越Standard的选择方法

**选择质量评估**（UPMC-Food101, F-IDN 50%）：
- REFINE F1=85.96%，显著高于FINE的84.22%
- 保留高损失干净样本比例达55.93%，其中91.88%仅由单一空间支持

**消融实验**：LA、DE、TSS三个组件均带来显著提升，尤其在Asym 40%设置下LA+DE贡献最大。

## 相关工作脉络
- **FINE [12]**：单模态噪声标签学习的代表性质性选择方法，通过类内对齐分数与GMM拟合选择干净样本；REFINE将其推广至多模态场景并通过DE扩展了判别能力。
- **CRUST [69]、AUM [70]、L2D [71]、DIST [72]**：梯度/训练动态驱动的样本选择方法；REFINE基于表示结构而非训练动态，不依赖噪声率估计。
- **Co-teaching [15]**：双网络小损失交叉更新范式；REFINE可作为即插即用替换组件，实验证明R-Co-teaching在所有设置下均优于原方法。
- **FreeMatch [83]、DSS+ [84]**：半监督噪声学习框架；REFINE的集成版本R-FreeMatch和R-DSS+在多项设置下达到SOTA。
- **TMNR [8]**：多视图噪声学习，依赖跨视图一致性；REFINE无需额外假设，仅利用表示空间间的判别差异。
- **dPCA [60]**：传统判别分析方法；本文借鉴其目标-背景加权思想但扩展至多模态多维空间选择。

## 局限性与未来方向
- **编码器冻结假设**：主要实验在冻结预训练编码器下进行，仅少量实验验证联合微调场景，对端到端训练下表示演化的适应性有待进一步验证。
- **类角度假设简化**：理论分析基于高斯分布假设，实际深度学习表示可能偏离理想化模型。
- **计算开销**： pairwise空间比较增加了额外计算成本，在高维表示或大规模数据下效率待优化。
- **未来方向**：论文提出将方法扩展至部分损坏、缺失和不一致的多模态监督信号，以及构建统一的"可信任多模态学习"框架。

## 研究启发与可借鉴点
- **判别分析范式迁移**：将dPCA的目标-背景加权思想应用于多模态噪声检测，为表示学习中的去噪方向提供了新思路。
- **空间自适应选择策略**：不同类别可依赖不同"最佳"表示空间的发现，对多专家系统、条件计算有借鉴价值。
- **无需阈值的标签一致性**：摒弃GMM拟合的替代方案，简化实现同时保持高选择精度。
- **即插即用设计哲学**：REFINE作为独立模块可兼容多种训练范式（小损失、半监督、鲁棒损失），增强了方法的实用价值。
- **保留困难干净样本**：通过单模态空间支持高损失正确标签的机制，为hard example mining提供了新视角。

## 关键术语表
**REFINE**：Reliable Sample Filtering in Multimodal Learning的缩写，本文提出的多模态噪声标签检测框架。
**Discriminative Eigenvector (DE)**：通过目标类与背景类加权判别分析构造的特征向量，最大化目标类对齐分数与背景类对齐分数的比值。
**Trusted Space Selection (TSS)**：为每个类别自适应选择具有更强噪声检测能力的表示空间集合，通过pairwise比较确定。
**Label Agreement (LA)**：基于判别特征向量的标签一致性筛选准则，要求样本在其观察类的对齐分数严格大于所有其他类。
**Alignment Score**：样本表示沿判别特征向量的投影长度的平方，用于衡量样本与某类别的契合程度。
**Instance-Dependent Noise (IDN)**：实例依赖噪声，噪声模式取决于样本本身的特征而非随机翻转。
**Symmetric/Asymmetric Noise**：对称/非对称噪声，前者均匀随机翻转标签，后者按语义混淆映射替换标签。

## 可复现要素
- **数据集**：UPMC-Food101、N24News、NWPU-Captions、Rakuten France、VGGSound50、MIntRec2.0公开可用；BSD来自DCASE 2026 Challenge。
- **代码**：论文声明源码将公开发布（"The source code will be publicly available"）。
- **超参数**：正则化参数$\epsilon = 10^{-4}$；warm-up 10轮后每5轮执行一次选择；学习率$10^{-4}$；batch size 256；dropout 0.3。
- **编码器**：DINOv2-Base（图像）、XLM-R-Base（文本）、WavLM-Base+（音频）、VideoMAEv2-Base（视频），投影层输出256维。
