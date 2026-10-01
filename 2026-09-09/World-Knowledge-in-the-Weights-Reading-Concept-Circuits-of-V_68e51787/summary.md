---
title: "World-Knowledge-in-the-Weights-Reading-Concept-Circuits-of-V"
source: https://arxiv.org/pdf/2609.09055v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 04:04:55"
field: "视觉可解释性与 mechanistic interpretability"
keywords: ["vision transformer", "concept circuit", "cross-layer transcoder", "mechanistic interpretability", "spurious correlation", "global circuit", "instance circuit"]
innovations: ["从ViT权重直接读出输入不变的全局概念电路", "CLT实例电路比SAE更忠实", "基于全局权与共激活双指标自动发现并去除虚假相关性"]
benchmarks: ["ImageNet validation", "Waterbird worst-group accuracy", "CLIP/DINO/ViT 模型比较"]
---

# 论文速读：World-Knowledge-in-the-Weights-Reading-Concept-Circuits-of-V

## 一句话总结
本文提出将跨层译码器（CLT）适配到视觉Transformer（ViT），从模型权重中直接读出**全局概念电路**（输入无关的“世界知识”）与**实例概念电路**（输入相关的决策路径），并用于自动发现/去除虚假相关性、比较不同监督范式的内部结构差异。

## 研究问题与动机
1. **核心问题**：ViT在视觉领域取得显著泛化，但其内部如何表征“世界知识”、概念之间如何跨层交互仍不明确，阻碍了可信审计与定向干预。
2. **现有方法局限**：传统可解释方法（显著图、特征重要性）只定位输入区域，将网络内部视为黑箱；基于SAE的管道主要捕获输入相关行为，若要获得全局视图需在大尺度数据集上做归因聚合，计算昂贵且仍可能受噪声干扰。
3. **缺乏可直接读取的全局电路**：现有机制可解释工作多聚焦实例级回路，缺少从参数中直接读出输入不变的概念交互结构的方法。

## 核心贡献（创新点）
1. **将CLT适配到ViT并验证有效性**：针对视觉数据调整超参、融合CLS token、聚合token位置以简化电路，使CLT能在视觉域稳定训练并提取稀疏可解释特征。
2. **首次从ViT读出全局概念电路**：通过编码器/解码器权重直接计算跨层概念边权，并用共激活统计重加权，得到输入无关、可审计的“世界知识”图。
3. **提出实例概念电路并证明其忠实度更高**：基于归因打patch与剪枝，得到单图决策路径；在CLIP/DINO/ViT上实验显示，CLT实例电路比匹配设置下的SAE电路在忠实度与完整性上分别高出约24.4%与22.1%。
4. **三个应用验证效用**：自动发现虚假相关性（无需人工标注）、基于实例电路进行干预去除虚假相关（Waterbird最弱组提升11.0%）、通过全局电路拓扑对比不同预训练范式（CLIP/DINO/ViT）的结构差异。

## 方法详解
- **CLT基础**：用稀疏编码器$\mathbf{W}_{\text{enc}}^l$将第$l$层残差流映射为稀疏隐激活$\mathbf{z}^l=\phi(\mathbf{W}_{\text{enc}}^l \mathbf{x}^l)$，再用跨层解码器$\mathbf{W}_{\text{dec}}^{l'\to l}$从前期所有层特征重建本层MLP输出$\hat{\mathbf{y}}^l=\sum_{l'\le l}\mathbf{W}_{\text{dec}}^{l'\to l}\mathbf{z}^{l'}$；损失为重建MSE加稀疏惩罚与死神经元预防项。
- **全局概念电路**：两特征$s,t$间原始边权为所有中间层线性路径的内积$w_{s\to t}=\langle\sum_{l\in L_{st}}\mathbf{W}_{\text{dec}}^{s,l},\mathbf{W}_{\\text{enc}}^t\rangle$；为抑制残余流带来的伪相关，乘以共激活概率重加权$\hat{w}_{s\to t}=\mathbb{E}[\mathbf{1}(a_s>0)\mathbf{1}(a_t>0)]\,w_{s\to t}$。对某类别提取子图时，以该类激活最强的末层特征为种子，迭代回溯$k$个上游特征。
- **实例概念电路**：对单张图，先用CLT获得稀疏激活，再用归因打patch计算特征间权重$A_{s\to t}=a_s\nabla_{a_s}a_t$；沿token位置求和聚合同一特征；按对logit贡献阈值剪枝节点与高影响力边，保留关键路径。
- **应用方法**：自动发现虚假相关以“高全局权但低共激活”判为捷径概念；去除时比较含/不含背景的两类实例电路，均值消融对称差中的背景特征；模型比较将全局边按层聚合、对比深层/浅层连接密度。

## 实验与结果
- **数据集与模型**：ImageNet验证集（50类随机采样）用于有效性验证；Waterbird用于虚假相关性去除；模型包含CLIP-ViT-B/32、DINO-ViT-B/16、ImageNet监督ViT-B/16。
- **全局电路有效性**：对CLIP/DINO/ViT分别消融全局Top-10重要概念，均值精度下降显著（CLIP由0.60降至0.22），远大于随机消融；使用重加权边比原始边更有效。
- **实例电路忠实度**：CLT在CLIP上平均忠实度较SAE提升24.4%、完整性提升22.1%，在所有图稀疏度下均优。
- **虚假相关性去除**：在Waterbird最弱组（水鸟在陆地背景）上，本文方法达到0.71，较SpLiCE（0.60）和SAE（0.53）分别提升11.0%和18.0%；仅随机消融相近数量概念则降至0.22。
- **模型比较**：CLIP深层连接密集→强泛化；DINO浅中层强连接→适合稠密预测；ViT呈相邻层渐进演化→部分类在深层前已编码类别概念。

## 相关工作脉络
1. **显著图/梯度方法**（Grad-CAM、DeepLIFT等）：定位输入区域，未揭示跨层概念交互结构。
2. **稀疏自编码器**（SAE/JumpReLU SAE等）：分解单层层内表示，主要用于输入相关特征提取；获得全局视图需大规模归因聚合。
3. **概念瓶颈模型**（CBM等）：依赖人工定义概念，难以捕获模型原生复杂概念；本文概念由CLT自学习得。
4. **机械可解释性回路发现**（A*、sparse feature circuits等）：多针对语言模型或依赖归因打patch进行实例级搜索；本文从CLT权重直接读出全局不变回路。
5. **虚假相关性检测**（Spurious features large-scale detection等）：依赖人工检查或单指标；本文以全局权+共激活双指标自动化发现。
6. **CLIP离散/概念白化**：对齐预定义概念方向；本文不需预定义，直接从权重读出可解释概念。

## 局限性与未来方向
- CLT主要捕获MLP路径，对注意力机制的结构仅间接反映，可能遗漏部分跨层交互。
- 概念语义仍需人工可视化校验，尚未完全自动化标签化。
- 全局图基于统计共激活重加权，并不能完全等价于因果结构。
- 扩展到大分辨率图像、视频或多模态内部结构仍待验证。

## 研究启发与可借鉴点
1. **从跨层权重直接读全局电路**：把“输入不变的可解释字典”作为世界知识载体，可迁移到CNN/Mamba等架构。
2. **双视图框架**：全局（可审计/可比较）与实例（可干预）互补，适合构建可信AI审计管线。
3. **双指标发现捷径**：用“连接强度×共激活频率”区分因果/非判别/虚假概念，思路可用于其他偏见诊断任务。
4. **CLT在视觉的调参经验**（稀疏度35、字典6144、JumpReLU阈值0.03、L0/单调性权衡）为后续复现提供基线。

## 关键术语表
**Cross-Layer Transcoder（CLT）**：跨层稀疏字典，编码器输入残差流、解码器从早期层重建本层输出，权重输入不变、激活输入相关。  
**Concept Circuit（概念电路）**：节点为可解释概念、边为概念间跨层影响的有向图。  
**Global Concept Circuit（全局概念电路）**：由CLT固定权重与共激活重加权构成的输入不变图，反映模型重用的一般知识。  
**Instance Concept Circuit（实例概念电路）**：基于单图稀疏激活与归因打patch得到的具体预测路径子图。  
**Spurious Correlation（虚假相关性）**：模型依赖但与目标类别在无偏环境中很少共现的捷径特征。  
**Attribution Patching（归因打patch）**：扰动某特征激活测量其对下游的影响，用于估计概念间因果权重。  
**Monosemanticity（单义性）**：特征仅响应单一语义概念的程度，常用于评估稀疏分解的可解释性。  

## 可复现要素
- **代码**：已开源（https://github.com/deep-real/VisionCLT）。  
- **数据集**：ImageNet、Waterbird、LVIS、Visual Genome均已公开。  
- **关键超参**：稀疏度35、字典大小6144、学习率5e-5、Batch 4096、训练token数2.5e8、JumpReLU阈值0.03、$\lambda_2=3\times10^{-5}$、c=4。  
- **模型**：CLIP-ViT-B/32、CLIP-ViT-B/16、DINO-ViT-B/16、ImageNet监督ViT-B/16。
