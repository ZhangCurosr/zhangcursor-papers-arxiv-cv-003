---
title: "Whole-Body-MRI-Classification-via-Prompt-Based-Clinical-Cond"
source: https://arxiv.org/pdf/2608.30824v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 10:42:50"
---

# 论文速读：Whole-Body-MRI-Classification-via-Prompt-Based-Clinical-Cond

## 一句话总结
本文提出TACTIC，一种将结构化临床变量以提示（prompt）形式条件化全身MRI视觉特征的多模态框架，天然支持任意数量输入与缺失数据，在UK Biobank的五项系统性与肿瘤性疾病分类任务上均取得最优或接近最优的AUROC性能。

## 研究问题与动机
- 核心问题：如何有效融合WB-MRI影像与结构化临床表格数据以提升疾病分类精度，同时应对临床记录在实际中普遍存在的缺失、稀疏与维度不一致问题。
- 传统多模态融合方法通常假设临床输入固定且完整，依赖插补或截断处理，导致在真实临床场景下性能显著退化、泛化能力受限。
- 现有医学提示学习方法多依赖空间线索、自由文本报告或结构化数据的翻译文本，未能以原生形式直接利用表格属性（如检验指标、人口学变量）指导视觉特征学习。
- 动机：借助Prompt-based视觉模型的灵活性，将临床属性直接编码为可替换的提示序列，通过条件化机制实现患者特异性特征聚合，无需固定输入结构即可自适应整合异构、部分缺失的临床上下文。

## 核心贡献（创新点）
- 提出TACTIC框架，将结构化临床表格变量编码为提示序列并通过交叉注意力条件化WB-MRI视觉表征。与特征拼接或早期融合方法不同，该方法在语义层面实现提示与图像的交互，彻底摆脱固定输入维度与插补需求。
- 将多模态融合重新表述为提示条件化问题，支持任意数量的表格属性输入并原生兼容数据缺失。与依赖固定张量结构的传统网络相比，其输入长度随可用临床变量动态变化，无需额外预处理即可适配异构患者记录。
- 设计MoFe与TabMoFe差分损失，在训练中显式模拟临床数据可用性梯度并约束性能单调提升。与常规CE损失或单样本融合策略不同，该设计以多-少属性对比为正则，确保模型在信息不足时不退化、在信息充足时充分吸收互补信号。
- 在UK Biobank五项疾病分类任务上系统验证，证明提示条件化范式的泛化与鲁棒性。与仅在单一模态或特定病种验证的基线不同，本文覆盖代谢、呼吸、肿瘤等多领域，表明该方法对异质性临床任务具有普适适用潜力。

## 方法详解
- **图像编码器（Primus-M）**：基于MAE框架预训练。针对WB-MRI超过40%体素位于体外的特性，先用Otsu阈值与二值形态学生成身体掩码，再采用掩码感知池化（窗口$[2\times2\times4]$）压缩背景并保留解剖信息；仅对包含前景内容的patch执行掩码重建，避免背景bias主导预训练目标。
- **表格编码器（TARTE）**：将每个临床属性的名称与对应值联合编码为embedding，每个属性生成一个独立提示向量，保留结构化信息的原始语义。
- **提示条件化Transformer**：临床提示先经自注意力捕获表格属性间的全局依赖，随后通过交叉注意力与图像特征$x\in\mathbb{R}^{n\times d}$交互，最终由输出token $T_{\mathrm{out}}$ 聚合得到分类预测。
- **单模态训练损失**：图像分支使用MAE重建损失；表格分支引入TabMoFe损失 $\mathcal{L}_{\mathrm{TabMoFe}} = \max(\mathcal{L}_{\mathrm{more}} - \mathcal{L}_{\mathrm{few}}, 0)$，其中$\mathcal{L}_{\mathrm{more}}$与$\mathcal{L}_{\mathrm{few}}$分别为使用更多/更少临床属性子集计算的重置交叉熵，最终 $\mathcal{L}_{\mathrm{tab}} = \mathcal{L}_{\mathrm{TabMoFe}} + \mathcal{L}_{\mathrm{more}} + \mathcal{L}_{\mathrm{few}}$。
- **多模态训练损失**：随机采样$N$个临床属性子集作为提示，计算多模态损失$\mathcal{L}_{\mathrm{mm}}$；同时用空提示集合$\emptyset$计算图像损失$\mathcal{L}_{\mathrm{img}}$；引入MoFe损失 $\mathcal{L}_{\mathrm{MoFe}} = \max(\mathcal{L}_{\mathrm{mm}} - \mathcal{L}_{\mathrm{img}}, 0)$。总损失为 $\mathcal{L} = \mathcal{L}_{\mathrm{MoFe}} + \mathcal{L}_{\mathrm{mm}} + \mathcal{L}_{\mathrm{img}}$。微调时TARTE冻结，仅更新Primus与提示交互模块。

## 实验与结果
- **数据集**：UK Biobank（80,000例含WB-MRI与住院临床记录），提取235项临床属性，基于ICD-10构建5项分类任务（糖尿病、COPD、乳腺癌、前列腺癌、转移瘤）。训练集按疾病类别与性别比例平衡采样，规模分别为：糖尿病4060、COPD 1292、乳腺癌3092、前列腺癌2682、转移瘤2885。
- **评估基线**：单模态（Primus图像、TARTE表格）；多模态融合（Concat Fuse、Max Fuse、Sum Fuse、DAFT、DMoMe、TIP）。
- **评估指标**：AUROC，报告10%至100%临床数据可用率下的均值。
- **主要结果**：TACTIC在糖尿病（89.1）、COPD（8
