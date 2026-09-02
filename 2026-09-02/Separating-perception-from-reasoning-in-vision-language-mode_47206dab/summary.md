---
title: "Separating-perception-from-reasoning-in-vision-language-mode"
source: https://arxiv.org/pdf/2609.00663v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 19:13:01"
field: "多模态模型评估与诊断"
keywords: ["vision-language models", "multimodal evaluation", "perception-reasoning separation", "crystal structure", "model-free benchmark", "render ceiling"]
innovations: ["提出无模型render ceiling通过逆变换已知渲染建立certified参考", "证明预言机仅在可枚举phantom set上失败并在2160结构验证为空", "建立R1-R3-R4归因阶梯精确分解感知与推理缺陷贡献"]
benchmarks: ["Materials Project crystal structures", "14 vision-language models zero-shot leaderboard", "ResNet-50/ViT supervised pixel baseline"]
---

# 论文速读：Separating-perception-from-reasoning-in-vision-language-mode

## 一句话总结
本文提出"渲染天花板"(render ceiling)这一无模型参考框架，通过逆变换固定相机并重新求解跨视角对应关系，精确量化视觉-语言模型在晶体结构识别任务中的感知缺陷与推理缺陷，发现感知并非大多数模型的主要瓶颈。

## 研究问题与动机
- 多模态评估无法区分模型是"读错了图像"还是"推理错误"，两者产生相同的错误答案但需要不同 remedy
- 现有分离方法（模型重写描述、ground-truth序列化、最佳模型集成等）都将第二个模型置于循环中，归因结果继承该模型的误差
- 晶体结构通常以可视化图片形式呈现（球棍模型等），需要模型能直接读取这些图像；但测试显示模型表现远低于渲染图允许的水平
- 在材料科学工作流中， fabricated intermediate（模型虚构的坐标列表）会被下游准确率误归因为推理错误，缺乏可审计的中间层验证机制

## 核心贡献（创新点）
- **提出render ceiling概念**：通过逆变换已知对象的渲染建立无模型参考，与已有工作依赖第二个模型作为参考的本质区别在于其准确性能被逐样本证明
- **证明天花板仅在可枚举的投影巧合集合上失败**：定义phantom set并证明其唯一失败模式，在2160个结构上验证该集合为空，与已有假设天花板为1的方法本质不同
- **建立attribution ladder框架**：通过R1(预言机天花板)-R3(提供精确几何文本)-R4(像素渲染)三级阶梯精确分解感知与推理贡献，而非依赖模型间比较
- **发现感知非主要瓶颈**：提供精确几何文本仅关闭13/14模型不到一半的差距，感知份额中位数为0.2901，推翻"感知是主要约束"的常见假设
- **暴露提取阶段虚构问题**：强模型作为提取器生成 syntactically perfect 的坐标列表但median recall为0.0000，105/206结构无一个发射原子在容忍范围内

## 方法详解
- **数据集构建**：从Materials Project抽取1820个结构（1610训练/210评估），采用composition-exclusion split保留13种元素仅用于评估；每个结构通过5个正交相机（3个主轴+2个斜向）渲染为768px图像
- **几何预言机(geometric oracle)**：算法1描述，将已知原子位置通过固定相机正交投影，丢弃跨视图对应关系，通过射线最近接近点(recovering point)重新求解对应，merge容差δ=τ时达到认证设置
- **Identifiability定理**：定义phantom set Φδ(s,C)为超出δ距离的所有same-species原子的交叉投影点；定理3证明预言机仅在Φδ非空时失败，且Φδ非单调依赖于δ
- **Ceiling度量**：R1 = |D|⁻¹ Σ 1[O(s,C) = y(s)]，在δ=τ=0.01Å认证设置下为1.0000；在δ=0.15Å释放设置下为0.9524
- **Attribution ladder**：定义P(m)=R3(m)-R4(m)为感知分量，S(m)=R1-R3(m)为post-perception残差，perception share=P(m)/(R1-R4(m))；三者均在相同结构、配对条件下评分
- **监督像素基线**：ResNet-50和ViT-small/patch16在1610个训练渲染上微调，无augmentation，五视角mean-logit聚合，ResNet-50达到0.8952

## 实验与结果
- **数据集**：210个评估结构（每个晶体系统30个，均匀分布），1950个scale-up样本（含phantom枚举），1610个训练样本
- **评估基线**：14个视觉-语言模型（Gemini 3.6 Flash, GPT-4.1 mini, Grok 4.5, Claude Opus 5, Llama 4 Maverick等），监督视觉基线ResNet-50/ViT-small，cell-metric random forest
- **主要结果**：
  - R1 = 1.0000（认证），R1^0.15 = 0.9524（释放）
  - 最强模型Gemini 3.6 Flash: R4=0.7333, R3=0.8524，感知份额0.4464
  - 最弱模型GPT-4.1 mini: R4=0.3667, R3=0.4143，感知份额0.0752
  - 13/14模型的S>P（推理缺陷大于感知缺陷），中位数感知份额0.2901 [0.1492, 0.3576]
  - ResNet-50达到0.8952（original sample），ViT-small达0.8333，均超过所有VLM
  - 移除图像（仅化学式）使所有模型坍缩至chance附近（mean 0.1487 vs 0.1429）
- **噪声鲁棒性**：frozen protocol容忍2.1×于off-axis perturbation的噪声（σ=0.02Å时分离）
- **相机放置规则**：四相机清空phantom set，第五相机贡献重建保真度而非准确性；camera placement而非view count设置noise budget

## 相关工作脉络
- **感知-推理分离方法**：Wang et al. (2025) [10]将图像替换为模型重写描述重新评分；Wei et al. (2025) [11]序列化ground-truth文本描述；本文区别在于完全不依赖模型，建立certified reference
- **视觉-语言模型benchmark**：Alampara et al. (2025) [2]发现模型处理chemistry图像感知良好但空间推理失败；Roberts et al. (2025) [29]指出benchmark headroom常被假设为1- best_score；本文提供certified ceiling替代假设
- **晶体结构识别**：Ziletti et al. (2018) [22]通过深度学习分类衍射图像；本文区别在于无模型closed-form procedure而非learned network
- **多视图重建**：Tomasi & Kanade (1992) [33]因子化方法估计相机；本文相机已固定，直接逆变换
- **机制探针**：Osaka et al. (2026) [14]定位visual access boundary；本文测量image information content而非网络内部使用位置
- **VLM感知诊断**：Kamoi et al. (2025) [15]VisOnlyQA发现模型仍不能读取基本几何属性；本文提供certified reference验证该结论

## 局限性与未来方向
- **前置条件限制**：需要forward rendering known且invertible，当前仅在正交投影晶体渲染上演示；在calibrated perspective下identifiability theorem仍成立但lattice-vector consequence不成立
- **理想提取假设**：预言机假设ideal extraction（centroid完美可用），实际pipeline的提取误差未被完整度量；R2（learned detector输出）作为diagnostic未进入ladder评分
- **容忍度非单调性**：phantom set非单调依赖δ，未证明growth rate with cell size，独立样本在认证设置下已丢失一个结构
- **残差上界**：post-perception residual S(m)是symmetry reasoning的上界而非精确测量
- **泛化范围**：未在更多domain验证transferability，仅提及可转移至molecular conformers, band structures等

## 研究启发与可借鉴点
- **无模型参考设计范式**：对于任何具有known forward rendering的benchmark（如化学分子图、材料表征图），可构造model-free ceiling进行attribution，避免模型-in-the-loop的误差继承
- **归因阶梯的可迁移框架**：R1（oracle ceiling）-R3（perception transplant）-R4（pixel input）三级分解可直接迁移至其他视觉理解任务，量化感知vs推理的贡献比例
- **对训练策略的启示**：感知份额与模型强度无关（rank correlation ρ=-0.1588, p=0.5877），说明decoupling perception-reasoning的训练策略应针对top模型而非bottom模型
- **提取器设计原则**：camera placement（conditioning constant κ）比view count更重要；四相机已足够empty phantom set，额外相机仅提升reconstruction fidelity
- **Agent pipeline审计工具**：fabrication detection方法（中间层score against model-free reference）可直接应用于materials design agentic workflow的中间验证

## 关键术语表
- **Render ceiling (R1)**：通过几何预言机在给定渲染协议下计算的理论最高准确率，无模型参与，作为模型表现的绝对上界参考
- **Phantom set (Φδ)**：因不同原子投影重合导致的虚假三维点集合，是预言机的唯一失败模式， emptiness证明ceiling可达
- **Attribution ladder**：由R1(oracle)、R3(geometry-as-text)、R4(pixel)三级组成的分解框架，量化感知与推理对总缺陷的贡献
- **Geometric oracle**：将已知原子位置通过固定相机投影、重新求解跨视图对应、应用spglib对称算法的确定性 pipeline
- **Perception share**：P(m)/(R1-R4(m))，感知缺陷占总缺陷的比例，中位数0.2901表明推理缺陷占主导
- **Shape-free baseline**：基于atom count、density、cell volume等shape-free特征的随机森林基线，达到0.5286准确率
- **Composition-exclusion split**：保留13种元素仅用于评估的划分策略，防止模型通过化学组成预测对称性
- **Conditioning constant (κ)**：堆叠投影系统的伪逆条件数，量化相机放置对centroid error的放大程度

## 可复现要素
- **数据集**：Materials Project [39]，CC BY 4.0许可；1820个结构（1610训练/210评估）已公开
- **代码/权重**：开源仓库 https://github.com/KurbanIntelligenceLab/render-ceiling，MIT许可；包含geometric oracle、render protocol、labelling pipeline完整实现
- **关键超参**：τ=0.01Å（symmetry tolerance），δ=0.01Å（certified）/0.15Å（released）；768×768 px分辨率；五相机正交投影；ResNet-50训练8 epochs，batch size 64，lr=3×10⁻⁴
- **随机种子**：split seed=23，训练seed=23
- **评估设置**：K=3 majority vote，temperature=0.7，decode budget共享；13种保留元素：Cd, Ce, Hg, In, Ir, La, Mn, Os, Re, Ru, Tc, Ti, Tl
