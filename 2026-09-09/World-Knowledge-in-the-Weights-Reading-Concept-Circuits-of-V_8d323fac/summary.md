---
title: "World-Knowledge-in-the-Weights-Reading-Concept-Circuits-of-V"
source: https://arxiv.org/pdf/2609.09055v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 04:05:18"
field: "视觉模型可解释性"
keywords: ["Concept Circuit", "Vision Transformer", "Cross-Layer Transcoder", "Interpretability", "Spurious Correlation", "Mechanistic Interpretability", "Sparse Autoencoder"]
innovations: ["首次用CLT从ViT权重直接读取input-invariant全局概念电路", "实例电路faithfulness超越SAE 24.4%并用于自动虚假相关去除", "通过co-activation重加权抑制feature interference获得因果相关的全局边权"]
benchmarks: ["ImageNet validation", "Waterbird worst-group accuracy"]
---

# 论文速读：World-Knowledge-in-the-Weights-Reading-Concept-Circuits-of-V

## 一句话总结
论文将跨层译码器（CLTs）适配到Vision Transformer，首次实现从已训练权重中直接读取**全局概念电路**（input-invariant）与**实例概念电路**（input-dependent），并用于自动发现虚假相关、干预去除虚假相关（Waterbird最弱组+11.0%），以及跨监督范式的内部结构对比。

## 研究问题与动机
- **核心空白**：ViTs在大量视觉任务上表现优异，但内部如何组织"世界知识"仍不清楚——哪些概念被编码、概念间如何在层间交互？
- **现有IML不足**：saliency/grad-CAM等仅定位输入关注区域，将网络内部当黑盒，无法回答"模型如何理解视觉世界"。
- **SAE局限**：稀疏自编码器等分解单层层内特征，主要捕捉**input-dependent**行为；获取全局视图需昂贵数据集级attribution，且结果仍可能受输入分布偏差影响。
- **自动化审计需求**：虚假相关发现多依赖人工检查，不可扩展；缺少无需领域知识的自动化shortcut发现与靶向干预框架。

## 核心贡献（创新点）
1. **CLT首次适配ViT并验证有效性**：将原为语言模型设计的跨层译码器迁移到视觉数据，解决超参配置、CLS token融合、token位置聚合三大挑战。
2. **全局概念电路直接从参数读取**：通过编码器/解码器权重与co-activation重加权，获得input-invariant的概念交互图，避免SAE依赖大数据集attribution的开销与偏差。
3. **实例概念电路比SAE更faithful**：结合attribution patching与token聚合，在CLIP上faithfulness与completeness分别超越SAE 24.4%与22.1%。
4. **自动虚假相关发现无需人工标注**：利用全局边权（模型依赖度）与co-activation频率（共现频率）的张力，自动分类因果/虚假/非判别概念。
5. **跨监督范式内部结构对比**：揭示CLIP深部稠密连接对应强泛化、DINO浅中层强连接对应稠密预测优势、ViT相邻层交互对应渐进特征演化。

## 方法详解
- **CLT架构**：对每层l，用稀疏激活函数φ（JumpReLU）将残差流编码为稀疏隐层z^l = φ(W_enc^l x^l)，再用前序层所有隐层解码重建MLP输出ŷ^l = Σ_{l'≤l} W_dec^{l'→l} z^{l'}。损失：ℒ_CLT = Σ_l ‖y^l - ŷ^l‖² + λℒ_spa + λ₂ℒ_preact。
- **全局边权计算**：两特征s→t的原始权重w_{s→t} = ⟨Σ_{l∈L_st} W_dec^{s,l}, W_enc^t⟩，再经co-activation重加权：ŵ_{s→t} = E[1(a_s>0)·1(a_t>0)]·w_{s→t}，抑制仅通过残差流产生的虚假大边。
- **类子图抽取（Algorithm 1）**：以目标类top激活的最后层特征为种子，迭代回溯上游最强影响节点，构建可解释的小子图。
- **实例电路抽取**：(1) 运行CLT获取稀疏激活；(2) attribution patching计算A_{s→t} = a_s ∇_{a_s} a_t；(3) 跨token位置聚合求和；(4) 按阈值剪枝节点/边。
- **虚假相关发现**：目标类C → 找top影响logit的目标概念t → 在D（ImageNet+LVIS+Visual Genome）上统计t与各源概念的co-activation；低共现+高边权=虚假概念，高共现+高边权=因果概念，高共现+低边权=非判别概念。
- **虚假相关干预**：在Waterbird上分别抽取水面/陆地背景实例电路C_water与C_land，mean-ablate对称差(C_water∪C_land)\(C_water∩C_land)中的背景相关特征，保留鸟种判别特征。

## 实验与结果
- **数据集**：ImageNet（训练/验证）、Waterbird（虚假背景）、LVIS、Visual Genome。
- **基线**：随机ablate、SpLiCE（人工选取背景概念）、SAE、BatchTopKSAE（匹配字典大小与稀疏度）。
- **全局电路有效性**：mean-ablate top-10全局重要概念后，CLIP准确率从0.60降至0.22（下降38pp），显著高于随机（0.66）与仅用原始权重的ablate（0.48）；co-activation重加权有效降低feature interference。
- **实例电路faithfulness**：CLT vs SAE在CLIP上平均faithfulness +24.4%、completeness +22.1%。
- **Waterbird干预**：最弱组（waterbirds on land）准确率原始0.48 → SpLiCE 0.60 → SAE 0.53 → **Ours 0.71**（+23pp，超越所有基线）；landbirds on land保持0.95。
- **模型比较**：CLIP深部稠密连接→强泛化；DINO浅中层强连接→稠密预测优势；ViT相邻层交互→渐进抽象。

## 相关工作脉络
- **SAE+电路发现**（Bricken et al. 2023; Huben et al. 2023; Marks et al. 2025）：单层层内稀疏分解+attribution patching，主要捕捉input-dependent行为，全局视图需大数据集聚合；本文用CLT直接从权重读取全局电路，避免昂贵attribution。
- **SpLiCE**（Bhalla et al. 2024）：线性概念嵌入干预CLIP，但仅作用于最后一层表征， Steering能力受限；本文跨层干预实例电路，提升更大。
- **概念瓶颈模型**（Koh et al. 2020; Yang et al. 2023）：预定义概念空间；本文概念由CLT自动学习，无需人工标注。
- **梯度/扰动类IML**（Selvaraju et al. 2017; Fong & Vedaldi 2019）：定位输入区域重要性；本文进一步揭示层间概念交互机制。
- **虚假相关检测**（Neuhaus et al. 2023; Singla & Feizi 2022）：依赖人工或事后分析；本文全自动基于全局电路统计发现。
- **Mechanistic interpretability**（Olah et al. 2020; Conmy et al. 2023）：电路发现方法论；本文将其扩展至ViT并首次获得全局input-invariant电路。

## 局限性与未来方向
- CLTs主要捕获MLP通路，**注意力结构的反映是间接的**；未来需扩展至attention head级别的电路读取。
- **概念解释依赖人工可视化**，尚未完全自动化；可结合语言模型生成概念标签。
- 全局电路未完全捕捉**因果关系的有向性**，当前基于相关性统计；未来可结合do-calculus或干预实验强化因果解读。
- 仅验证于ViT-B/32等中等规模模型，**大模型（ViT-L/224x224、SigLIP等）的缩放行为**有待检验。
- 实例电路抽取依赖attribution patching，计算开销较高；未来可探索近似或并行化方案。

## 研究启发与可借鉴点
- **权重直接读取全局电路**的思想可迁移至其他架构（CNN、Mamba、扩散模型），无需大数据集attribution即可获得input-invariant概念交互图。
- **co-activation重加权抑制feature interference**的技巧通用性强，可推广至任何基于线性路径求和的电路估计。
- **对称差干预策略**（C_water Δ C_land）简洁有效，可借鉴用于其他成对背景/域偏移场景的shortcut去除。
- **全局+实例双视图**框架可同时支持系统级审计与实例级解释，为可解释性工具链设计提供范式。
- 监督范式对比揭示了**训练目标→电路拓扑→下游行为**的可追溯链条，为模型选择与微调策略提供机制层面依据。

## 关键术语表
- **Cross-Layer Transcoder (CLT)**：稀疏跨层字典，将 Transformer MLP 计算分解为 input-invariant 编码器/解码器权重与 input-dependent 稀疏激活。
- **Global concept circuit**：由 CLT 权重与 co-activation 重加权得到的 input-invariant 概念交互图，反映模型泛化行为。
- **Instance concept circuit**：针对单张输入，经 attribution patching 与 token 聚合得到的 active 概念子图，用于实例级解释。
- **Co-activation reweighting**：用特征共激活概率 E[1(a_s>0)·1(a_t>0)] 对原始全局边权重加权，抑制残差流引入的虚假大边。
- **Attribution patching**：计算特征 a_s 对 a_t 的梯度贡献 a_s ∇_{a_s} a_t，用于实例电路的有向边权重估计。
- **Spurious correlation**：模型依赖但很少与目标类共现的特征关联；本文通过低 co-activation + 高全局边权自动识别。
- **Monosemanticity**：特征表征单一语义的程度；本文用它评估 CLT 学习到的概念可解释性。
- **Class subgraph**：以目标类 top 激活特征为种子，迭代回溯上游最强影响节点形成的可解释小子图。

## 可复现要素
- **代码**：https://github.com/deep-real/VisionCLT（已开源）
- **数据集**：ImageNet（公开）、Waterbird（公开，Sagawa et al. 2020）、LVIS（公开）、Visual Genome（公开）
- **模型**：CLIP-ViT-B/32、CLIP-ViT-B/16、ImageNet-supervised ViT-B/16、DINO-ViT-B/16（权重公开）
- **关键超参**：稀疏度 35、字典大小 6144、学习率 5×10⁻⁵、batch size 4096、训练 token 数 2.5×10⁸、JumpReLU 阈值 τ=0.03、λ₁ sweep 范围 [1,5]、λ₂=3×10⁻⁵
- **实现细节**：Adam 优化器、β₁=0.9、β₂=0.999、最后 20% 步骤线性衰减学习率至 0、sparsity 惩罚线性 ramp-up
