---
title: "SAGE-Subpopulation-Aware-Generative-Enhancement-for-Mitigati"
source: https://arxiv.org/pdf/2609.01051v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 19:12:36"
field: "鲁棒机器学习与虚假相关性缓解"
keywords: ["spurious correlations", "generative data augmentation", "worst-group accuracy", "diffusion models", "domain generalization", "robust classification"]
innovations: ["提出无组标签下的两阶段生成增强框架SAGE，通过聚类推断子群体引导定向生成", "设计逆密度采样与token合并策略，解决聚类随机性导致的多样性下降问题", "构建合成子标签平衡验证集替代手动DFR验证集，实现无监督最后层重加权"]
benchmarks: ["Waterbirds", "CelebA", "MetaShift"]
---

# 论文速读：SAGE: Subpopulation-Aware Generative Enhancement for Mitigating Spurious Correlations

## 一句话总结
SAGE 提出了一种两阶段生成增强框架，通过特征聚类推断隐式子群体标签，微调条件扩散模型生成针对欠代表区域的合成数据，并构建平衡的合成验证集用于最后层重加权，在无需真实组标签和人工验证集的实用场景下有效缓解虚假相关性。

## 研究问题与动机
- **核心问题**：深度学习模型常依赖数据集中的虚假相关（spurious correlations，如背景代替主体分类），导致最坏组准确率（Worst-Group Accuracy, WGA）显著下降。
- **现有方法不足**：
  1. 依赖组标签的方法（如 gDRO、DFR）需要手动标注的虚假属性分组或平衡验证集，在实际场景中难以获取。
  2. 无组标签的重采样方法（如 JTT、LfF、MaskTune）重复使用已有样本，降低了有效多样性，易导致过拟合。
  3. 已有的生成式方法往往依赖外部复杂工具（LLM、分割模型）或显式组标注来指导生成，框架复杂且实用性受限。
- **关键洞察**：视觉显著的特征偏差会在预训练特征空间中形成结构化聚类，即使没有组标签，聚类分配也可作为偏差相关子群体的代理信号。

## 核心贡献（创新点）
1. **提出 SAGE 无监督生成增强框架**：在虚假属性不可见、无组平衡验证集的现实设定下，通过聚类推断子标签引导生成，区别于依赖 LLM/分割模型或显式组标注的前作。
2. **设计了基于语义聚类的子标签发现与条件生成策略**：使用 AP 聚类将特征空间划分为子群体，为每个簇分配可学习的 token，并联合类标签微调 Stable Diffusion，使生成过程能捕获子群体特异性视觉因素，与简单文本引导的生成方法本质不同。
3. **提出逆密度采样与 token 合并机制**：通过逆密度权重对欠代表区域进行采样，并在采样前合并高相似度 token 以避免聚类随机性导致的多样性下降，解决了现有生成式方法采样策略粗糙的问题。
4. **构建合成验证集替代手动 DFR 验证集**：利用均匀采样构建子标签平衡的合成验证集，作为 Deep Feature Reweighting (DFR) 的代理重加权集，避免了手动构建组平衡验证集的需求。

## 方法详解
**整体框架**：SAGE 分为两阶段，Stage 1 进行语义聚类与条件生成模型微调，Stage 2 进行定向数据生成与下游模型训练。

**Stage 1: 语义聚类与条件生成器微调**
- **AP 聚类**：使用预训练 CLIP 编码器将训练图像映射到隐空间 $\mathcal{Z}=\{z_1,...,z_N\}$，然后采用 Affinity Propagation (AP) 聚类进行子群体划分。AP 无需预设聚类数，通过迭代消息传递找出 exemplar，对角偏好 $s(k,k)$ 设为所有成对相似度的均值。
- **LoRA 微调扩散模型**：以 Stable Diffusion v1.5 为基础，在 UNet 上应用 LoRA（rank=128, $\alpha$=128），并为每个子标签 $k$ 在文本编码器中注册唯一 token $\langle t_k\rangle$，初始化为 "photo" 的语义先验。
- **提示模板**：使用结构化提示 `"a photo of [class label y], [sub-label token ⟨t_k⟩]"`，联合优化新 token 嵌入与 LoRA 参数，冻结其余权重。损失函数为隐扩散损失：
  $$\mathcal{L}_{\text{joint}} = \mathbb{E}_{z \sim \mathcal{E}(x), \epsilon \sim \mathcal{N}(0,\mathbf{I}), t}[||\epsilon - \epsilon_{\theta+\Delta\theta}(z_t, t, \tau(y, t_k))||_2^2]$$

**Stage 2: 定向数据生成与下游模型训练**
- **逆密度采样策略**：对每个类内簇计算逆密度权重 $P_{\text{train}}(\langle t_k\rangle) = N_k^{-1} / \sum_j N_j^{-1}$，但先进行 token 合并——计算同类内 token 的余弦相似度均值 $\bar{s}_y$，满足 $\cos(\mathbf{e}_i, \mathbf{e}_j) > \beta \cdot \bar{s}_y$ 的 token 被合并为语义组 $G_m$（$\beta=2.5$ 对 Waterbirds，$4.0$ 对 CelebA/MetaShift）。全局归一化的采样概率为：
  $$P_{\text{train}}(\langle t_k\rangle) = \frac{1}{|G_m|} \cdot \frac{N(G_m)^{-1}}{\sum_{G' \in \mathcal{G}} N(G')^{-1}}$$
  合成样本数 $Q=|\mathcal{D}|$，形成 $\mathcal{D}_{\text{syn}}$，与原始数据集混合得到 $\mathcal{D}_{\text{mix}}$。
- **合成验证集构建**：采用确定性 samples-per-token 均匀采样策略，构建全局平衡的 $\mathcal{D}_{\text{val}}$，作为 DFR 的代理重加权集。
- **下游训练与 DFR**：在 $\mathcal{D}_{\text{mix}}$ 上进行标准 ERM 训练（ResNet-50 + ImageNet 预训练），冻结特征提取器，仅用 $\mathcal{D}_{\text{val}}$ 重新训练最后线性层完成 DFR。

## 实验与结果
**数据集**：Waterbirds（背景-鸟类类别虚假相关）、CelebA（性别-发色虚假相关）、MetaShift（动物-背景虚假相关）。

**评估指标**：最坏组准确率（Worst-Group Accuracy, WGA）与平均准确率（Average Accuracy）。

**主要结果（Table 1）**：
| 数据集 | SAGE WGA | 最佳无组标签基线 WGA | 提升幅度 |
|--------|----------|---------------------|----------|
| Waterbirds | **89.5%** | DISC 88.7% | +0.8 pp |
| CelebA | **85.7%** | CnC 88.8%（有验证集）/ MaskTune 78.0%（无验证集） | +7.7 pp vs MaskTune |
| MetaShift | **79.1%** | DaC 78.3% | +0.8 pp |

- SAGE 在所有三个数据集的"无组标签"类别中取得最高 WGA。
- 在 MetaShift 上，SAGE 甚至超越了使用组标签的 gDRO（72.8%）。
- 与生成式增强方法对比（Table 2）：SAGE 超越 ASPIRE+ERM（Waterbirds: 78.7%→89.5%, CelebA: 50.5%→85.7%），虽略逊于依赖外部分割模型的 DDB（93.0%/85.8%），但框架更简洁实用。

**消融实验（Table 3-4）**：
- 通用生成增强（$\mathcal{D}_{\text{sg}}$）效果有限甚至有害（Waterbirds: 70.5% vs D 的 73.2%），而目标定向生成（$\mathcal{D}_{\text{mix}}$）显著提升（+4.3/-5.2/+6.1 pp）。
- 合成验证集在 CelebA（82.6% vs 62.8%）和 MetaShift（73.4% vs 69.5%）上显著优于手动验证集，Waterbirds 上略低于手动验证集（88.6% vs 89.7%）。
- 可视化表明子标签 token 能捕获子群体特异性视觉因素（如 Waterbirds 的背景类型、CelebA 的发色-性别组合）。

## 相关工作脉络
1. **组标签依赖方法（gDRO, DFR）**：利用显式组信息进行训练重加权或验证集选择；SAGE 定位差异在于完全无需组标签和手动验证集，通过聚类推断子群体并生成合成验证集替代。
2. **代理信号方法（JTT, LfF, CnC, DaC）**：从模型行为推断困难/偏差样本；SAGE 定位差异在于不依赖模型训练过程中的代理信号，而是直接从特征空间结构推断子群体，并通过生成扩展数据而非重采样已有样本。
3. **严格无先验方法（MaskTune, DISC）**：MaskTune 通过掩码最显著区域强迫探索替代线索；DISC 通过发现不稳定概念进行干预；SAGE 定位差异在于主动生成新样本填充欠代表区域，而非仅修改已有训练策略。
4. **生成式增强方法（ASPIRE, DDB, Clustered DreamBooth）**：ASPIRE 依赖 LLM 识别虚假属性并个性化文本到图像模型；DDB 结合文本反演、语言分割和扩散生成；Clustered DreamBooth 需在已知组内聚类；SAGE 定位差异在于仅利用目标数据集内部聚类结构，无需外部工具或显式组标注。
5. **扩散模型数据增强（Azizi et al., Dunlap et al.）**：主要关注 ImageNet 分类性能提升；SAGE 将其应用于虚假相关性缓解，引入了子标签条件和逆密度采样等针对性设计。

## 局限性与未来方向
- **聚类纯度非100%**：后验分析显示 Waterbirds/CelebA 聚类纯度超90%，但 MetaShift 仅72.49%，说明聚类与真实虚假属性并非完全对齐，部分低密度簇可能反映非虚假变异（如姿态、光照）。
- **生成质量与多样性权衡**：逆密度采样可能导致相似特征被重复生成，虽有 token 合并缓解，但生成样本的多样性仍受限于基础扩散模型。
- **计算开销较大**：需微调 Stable Diffusion（2×RTX 4090，100 epochs）+ 聚类 + 下游训练，相比无生成方法计算成本显著更高。
- **特征空间依赖**：方法有效性建立在"虚假特征会在预训练特征空间中形成可区分聚类"这一假设上，对于特征空间结构不清晰的模态（如文本、表格）可能需要适配。
- **未来方向**：可扩展至其他模态（视频、多模态）、研究更高效的聚类与生成策略、探索与模型端去偏方法的结合。

## 研究启发与可借鉴点
1. **子标签 Token 的可学习机制**：将聚类分配的离散标签转换为可学习 token 并联合 LoRA 微调扩散模型，是一种将无监督结构信息注入生成过程的优雅范式，可迁移至其他需条件控制的生成任务。
2. **逆密度采样 + 全局归一化策略**：针对长尾分布进行采样加权时，全局归一化优于类内归一化，能有效处理跨类别的不平衡问题，这一设计可推广至其他生成式数据增强场景。
3. **合成验证集作为 DFR 代理**：在缺乏真实组平衡验证集时，利用生成模型构建平衡代理验证集进行最后层重训练，是一种实用的无监督去偏策略，可与其他 reweighting 方法结合。
4. **Token 合并去冗余**：通过余弦相似度合并语义重叠的 token，缓解了聚类随机性导致的多样性下降，这一后处理步骤对任何基于聚类的条件生成系统均有参考价值。
5. **与下游团队的结合机会**：若团队研究方向涉及长尾分类、领域泛化或生成式数据增强，SAGE 的聚类-生成-重加权三阶段框架可直接借鉴；其无监督子群体推断思路也可应用于医疗影像、工业缺陷检测等标签稀缺场景。

## 关键术语表
- **Spurious Correlations（虚假相关性）**：模型依赖的非因果特征与标签之间的统计关联，在测试分布变化时导致性能骤降。
- **Worst-Group Accuracy (WGA)（最坏组准确率）**：在各类子群体中最低的分类准确率，用于衡量模型对少数/困难组的鲁棒性。
- **Affinity Propagation (AP) Clustering（亲和传播聚类）**：无需预设聚类数的迭代消息传递聚类算法，通过 exemplar 选择自动确定聚类数量。
- **LoRA (Low-Rank Adaptation)（低秩自适应）**：通过在预训练模型注意力层注入低秩矩阵微调参数，大幅降低训练计算成本。
- **Deep Feature Reweighting (DFR)（深度特征重加权）**：冻结特征提取器，仅在有组平衡验证集上重新训练最后线性层以缓解虚假相关的方法。
- **Inverse-Density Sampling（逆密度采样）**：对低密度（少数）簇赋予更高采样概率的策略，用于定向生成欠代表区域的数据。
- **Sub-label Token（子标签 Token）**：为每个聚类分配的可学习文本 token，捕获子群体特异性视觉因素并作为条件生成输入。
- **Conditional Diffusion Model（条件扩散模型）**：以额外条件（如类别、子标签）指导去噪过程的扩散模型，用于可控图像生成。

## 可复现要素
- **数据集**：Waterbirds、CelebA、MetaShift（标准benchmark，公开可得）
- **代码**：已开源，GitHub: https://github.com/luoym-lym/SAGE
- **关键超参**：
  - LoRA rank=128, α=128
  - 学习率 1e-5, AdamW, batch size 4/GPU, 100 epochs
  - 图像分辨率 512×512
  - AP 聚类 damping: Waterbirds/MetaShift=0.5, CelebA=0.8; α_pref=1
  - Token 合并阈值系数 β: Waterbirds=2.5, CelebA/MetaShift=4.0
  - CelebA 使用 FPS 降采样至 K=10,000 再运行 AP
- **硬件**：Stable Diffusion 微调用 2×RTX 4090，其余阶段用 1×RTX 4090
- **基础模型**：Stable Diffusion v1.5 (Hugging Face), CLIP, ResNet-50 (ImageNet 预训练)
