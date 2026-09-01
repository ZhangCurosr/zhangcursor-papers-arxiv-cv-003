---
title: "Who-Remains-What-Changes-Identity-Anchored-Composed-Gait-Ret"
source: https://arxiv.org/pdf/2608.26632v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 00:54:45"
field: "步态识别与组合检索"
keywords: ["Composed Gait Retrieval", "Gait Recognition", "Identity Anchoring", "Vision-Language Models", "Composed Retrieval"]
innovations: ["提出CoGR任务与首个步态-语言数据集", "设计PIA模块将多帧肢体感知身份压缩为ID令牌并双向注入共享Q-Former", "构造CoGR对比损失支持多正样本并过滤身份冲突负样本"]
benchmarks: ["Language-Augmented CCPG", "Language-Augmented CASIA-B"]
---

# 论文速读：Who-Remains-What-Changes-Identity-Anchored-Composed-Gait-Ret

## 一句话总结
本文提出步态组合检索（CoGR）新任务，通过自然语言指令在保持身份不变的前提下修改步态序列的属性（如服装、携带物、视角），并构建首个步态-语言数据集与身份锚定框架ComposeGait，实现了指令满足与身份保持的双重目标。

## 研究问题与动机
- **核心问题**：现有步态检索系统受限于刚性的人-to-人匹配范式，无法理解自然语言修改指令（如"找这个人，但现在穿蓝色衬衫"），仅能处理静态的视觉特征比对。
- **身份-语义纠缠瓶颈**：步态序列是高维时空表示，生物识别身份与外观属性深度交织；通用组合检索方法早期融合文本与视觉特征时缺乏对身份线索的显式保护，易导致"身份漂移"——模型正确修改了属性但检索到了错误的人。
- **数据缺失**：现有步态数据集（如CCPG、CASIA-B）仅提供离散字母数字条件标签，缺乏自然语言所需的句法结构与语义丰富性，无法支持指令式检索训练。
- **任务定位差异**：CoGR要求身份必须保持不变的条件下，用户指定的子集条件成为目标的一部分，将步态分析从被动识别扩展为身份保持的语言引导组合检索。

## 核心贡献（创新点）
- **提出CoGR新任务与自动化数据集构建**：首次形式化组合步态检索任务，并开发基于大视觉语言模型（VLM）的自动标注流水线，构建Language-Augmented CCPG和CASIA-B数据集；区别于传统步态识别仅关注身份分类，本文使条件变化成为检索目标。
- **身份锚定组合框架ComposeGait**：设计Part-aware Identity Adapter（PIA）模块，从多帧肢体感知步态特征中提取样本级ID令牌并注入共享Q-Former双分支，防止语义修改时的身份漂移；与通用CIR方法的本质区别在于显式引入生物识别身份锚点而非隐式特征融合。
- **CoGR对比损失设计**：构造支持多正样本并过滤身份冲突负样本的对比学习目标，解决同一身份但条件不匹配的目标不应作为硬负样本的问题；区别于标准InfoNCE或SupConLoss，该损失仅在满足CoGR相关性的正样本间计算相似度。
- **系统性实验验证**：在两个语言增强基准上建立最强基线，CCPG达72.38% R@1，CASIA-B达83.61% R@1，显著提升现有CIR/CoVR/CPR方法的性能。

## 方法详解
**CoGR范式定义**：给定参考步态序列$X^r$、自然语言修改指令$m$和图库$\mathcal{G}$，检索目标需满足$y(X)=y(X^r)$（身份保持）且$c(X)=T_m(c(X^r))$（指定条件改变，未指定条件保持不变），检索公式为内积排序：
$$\hat{X}^t = \arg\max_{X \in \mathcal{G}} f_r(X^r, m)^\top f_t(X)$$

**自动化数据集构建流水线**：使用Qwen3-VL-235B进行三阶段标注——属性提取（填入{upper}/{lower}/{bag}等语义槽）、静态组装（填充语言模板生成外观描述）、动态三元组生成（对比同身份参考与目标序列检测变化，合成相对指令）。CASIA-B额外支持视角变化指令，CCPG专注外观变化。

**ComposeGait框架**：
1. **冻结ViT-G编码**：对采样步态帧（训练最多30帧，推理最多60帧）进行单次编码
2. **Part-aware Identity Adapter（PIA）**：从ViT-G的选定层{8,16,24,39}提取分层特征，经SE块+残差块多路融合，对P=16个身体区域进行时域最大池化与水平池化，映射为$d^{\text{id}} \in \mathbb{R}^{D_i}$（$D_i=768$）
3. **ID令牌构建**：通过投影$a = W_{\text{id}}z^{\text{id}} + b_{\text{id}} + e_{\text{type}}$将身份表示映射到Q-Former隐维度，附加到预训练查询令牌后作为身份锚点
4. **双向共享Q-Former**：查询分支处理预训练查询+参考ID令牌+参考视觉令牌+修改文本；目标分支处理预训练查询+目标ID令牌+目标视觉令牌；仅原始查询令牌输出参与检索嵌入，ID令牌输出被排除

**联合身份与组合学习**：
- 身份损失：$\mathcal{L}_{\text{id}} = \mathcal{L}_{\text{ce}}(h_{\text{id}}(Z^{\text{id}}), Y) + \mathcal{L}_{\text{tri}}(Z^{\text{id}}, Y)$（交叉熵+硬三元组，margin=0.3，label smoothing=0.1）
- CoGR对比损失：$\mathcal{L}_{\text{CoGR}}^{(i)} = -\frac{1}{|\mathcal{P}(i)|}\sum_{p \in \mathcal{P}(i)}\log\frac{\exp(f_{r,i}^\top f_{t,p}/\tau)}{\sum_{j \in \mathcal{D}(i)}\exp(f_{r,i}^\top f_{t,j}/\tau)}$
  - $\mathcal{P}(i)$：满足CoGR相关性的批次内目标集合（多正样本）
  - $\mathcal{D}(i)$：排除同身份但条件不匹配的目标$\mathcal{A}(i)$后的分母集合
- 总损失：$\mathcal{L} = \mathcal{L}_{\text{CoGR}} + \lambda_{\text{id}}\mathcal{L}_{\text{id}}$（$\lambda_{\text{id}}=1$，温度$\tau=0.07$）

## 实验与结果
**数据集**：
- Language-Augmented CCPG：50,000三元组（40,000训练/10,000测试），覆盖多样化服装与携带变化
- Language-Augmented CASIA-B：64,506三元组（55,160训练/9,346测试），覆盖0°–180°视角及normal/bag/coat条件

**评估指标**：R@K（严格成功：身份+未指定条件保持+指定改变）、SC-R@K（仅指令满足）、ID R@1（仅身份正确）、CASIA-B额外报告SC-R_a（属性）、SC-R_v（视角）、SC-R_c（组合）

**主要结果**：
| 方法 | Backbone | CCPG R@1 | CASIA-B R@1 | CASIA-B ID R@1 |
|------|----------|----------|-------------|----------------|
| SPRC | ViT-G/14 | 57.72% | 66.46% | 71.57% |
| FAFA | ViT-G/14 | 68.82% | 43.77% | 84.73% |
| **ComposeGait** | **ViT-G/14** | **72.38%** | **83.61%** | **84.84%** |

- **CCPG**：ComposeGait达72.38% R@1，超越同骨干FAFA +3.56pp，ID R@1达76.56%
- **CASIA-B**：83.61% R@1，超越最强竞争者SPRC +17.15pp；视角召回SC-R_v@1达96.56%，组合召回SC-R_c@1达78.36%
- **消融实验**：Shared QF+PIA联合贡献最大（R@1从61.47%→67.74% +6.27pp）；多帧证据（ViT→QF Multi）贡献+4.64pp；双侧ID令牌注入最优（比单侧+1.26~2.37pp）

## 相关工作脉络
- **Composed Image Retrieval (CIR)**：TIRG、CLIP4CIR、TG-CIR、SPRC等方法在静态图像上实现组合检索，但未考虑步态序列的生物识别身份保持需求；本文将CIR扩展到时序步态且显式保护身份。
- **Composed Video Retrieval (CoVR)**：CoVR-2、FDCA等方法扩展视频检索，但目标为对象/场景级编辑，非生物识别身份 preservation；本文需在整段步态序列中保持身份稳定。
- **Composed Person Retrieval (CPR)**：FAFA等基于静态人物图像，无多帧证据聚合机制；本文PIA模块从多帧聚合肢体感知身份证据，并通过共享Q-Former双分支双向锚定。
- **Gait Recognition**：GaitSet、OpenGait、GaitBase等从集合/序列建模演进至大视觉模型（BigGait/BiggerGait）；本文利用ViT-G分层特征经PIA提取肢体感知身份表示。
- **Cross-Covariate Gait Datasets**：CCPG、CASIA-B等提供离散条件标签；本文通过VLM自动转换为自然语言指令，构建首个步态-语言数据集。

## 局限性与未来方向
- **大规模真实场景数据缺失**：当前数据集基于受控/半受控环境（CASIA-B）或单一场景（CCPG），未来需扩展至in-the-wild大规模步态-语言数据集
- **开放词汇描述支持有限**：当前指令基于预设模板与离散条件标签，未支持自由开放词汇描述
- **颜色-光照解耦不足**：失败案例分析显示，绿色主导照明会使白色服装呈现绿色外观，导致颜色条件检索敏感度不足（相似度仅从0.77降至0.73）
- **未指定条件保持能力待加强**：部分失败案例显示模型可能遗漏未修改条件（如保留bag-carrying状态）
- **未来方向**：开放词汇指令、光照不变表示、更灵活的标注策略、非约束检索场景下的身份感知组合

## 研究启发与可借鉴点
- **身份锚定机制设计**：PIA将多帧肢体感知特征压缩为单一样本级ID令牌并注入共享Q-Former，实现了"身份保持+语义修改"的解耦，该设计可迁移至其他时序生物识别任务（如人脸重识别、步态重识别）
- **多维度对比损失构造**：CoGR损失通过排除同身份条件不匹配目标（$\mathcal{A}(i)$）避免身份-条件梯度冲突，该思路可用于任何需同时优化多个约束的检索任务
- **VLM辅助数据集构建流水线**：三阶段自动化标注（属性提取→静态组装→动态三元组生成）结合数据集特定规则（CASIA-B视角/CCPG服装），为其他模态-语言对齐任务提供可复用范式
- **双侧Token注入策略**：ID令牌同时注入查询与目标分支的共享编码器，实现双向身份感知嵌入空间，相比单侧注入性能提升显著（+1.26~2.37pp）

## 关键术语表
**Composed Gait Retrieval (CoGR)**：基于参考步态序列与自然语言修改指令的组合检索任务，要求检索结果保持参考身份且满足指定属性变化
**Part-aware Identity Adapter (PIA)**：从ViT-G多层特征提取肢体感知身份表示的模块，聚合多帧证据生成样本级ID令牌
**Identity Token (ID Token)**：由PIA输出映射到Q-Former隐空间的样本特定身份锚点，注入共享编码器引导注意力但不直接进入检索嵌入
**Language-Augmented CCPG/CASIA-B**：通过VLM自动标注构建的首个步态-语言组合检索数据集
**Shared Q-Former**：查询与目标分支共享权重的双塔编码结构，确保嵌入空间一致性与参数效率
**CoGR Contrastive Loss**：支持多正样本且排除同身份条件不匹配目标的对比学习目标，避免身份-条件梯度冲突
**Identity Drift**：组合检索中模型正确修改属性但检索到错误身份的错误模式
**Specified-Change Recall (SC-R@K)**：仅检查指定属性/视角变化是否实现的评估指标，区分严格任务成功与指令满足

## 可复现要素
- **数据集**：Language-Augmented CCPG（50,000三元组）与Language-Augmented CASIA-B（64,506三元组）基于公开CCPG与CASIA-B构建；论文声明数据集与代码将公开
- **代码/权重**：论文未明确说明当前是否开源，仅声明"will be made publicly available"；初始化使用BLIP-2 Salesforce/blip2-itm-vit-g checkpoint
- **关键超参**：ViT-G层{8,16,24,39}，P=16身体区域，$D_i=768$，训练采样最多30帧/序列，推理最多60帧；AdamW学习率$2\times10^{-5}$，weight decay 0.05，warmup 500步，cosine decay 20,000迭代；batch含16三元组（每身份4实例）；对比温度$\tau=0.07$，$\lambda_{\text{id}}=1$，梯度裁剪1.0
- **硬件**：单卡NVIDIA RTX 5090
