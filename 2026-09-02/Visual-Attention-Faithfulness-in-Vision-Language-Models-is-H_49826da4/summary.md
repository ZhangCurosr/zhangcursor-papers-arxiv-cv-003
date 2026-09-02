---
title: "Visual-Attention-Faithfulness-in-Vision-Language-Models-is-H"
source: https://arxiv.org/pdf/2609.00830v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 07:24:47"
field: "多模态模型可解释性"
keywords: ["视觉语言模型", "注意力忠实性", "因果扰动", "可解释性", "VQA", "comprehensiveness", "sufficiency gap"]
innovations: ["首次系统评估VLM视觉注意力忠实性，提出comprehensiveness与sufficiency gap双维度因果评估框架", "识别三种异构处理模式（Faithful-Sufficient/Faithful-Distributed/Non-Focal）揭示注意力忠实性的非均匀性", "发现人类标注ground-truth仅约60%满足comprehensiveness，揭示人-机视觉依赖的系统性分歧"]
benchmarks: ["VQAv2", "VRDU", "ChartQA"]
---

# 论文速读：Visual-Attention-Faithfulness-in-Vision-Language-Models-is-H

## 一句话总结
本文首次系统评估了视觉语言模型（VLMs）中视觉注意力的忠实性，通过因果扰动分析发现视觉注意力忠实性呈现三种异构处理模式，并揭示了模型视觉依赖与人类标注ground-truth之间的系统性分歧。

## 研究问题与动机
1. **注意力是否忠实反映模型推理**：注意力权重是否能忠实反映模型对视觉信息的依赖，这一问题在NLP中已有争议，但在VLM的视觉模态中尚未探索。
2. **现有方法假设的缺陷**：近年工作要么假设低注意力视觉token可安全剪枝以加速推理，要么分析注意力模式以表征内部机制，但未验证视觉注意力排序是否真正对应模型对特定视觉区域的因果依赖。
3. **人类感知与模型依赖的分歧**：模型实际依赖的视觉区域是否与人标注的ground-truth区域一致，现有工作缺乏系统对比。
4. **忠实性的二元误区**：现有工作常将注意力忠实性视为二元属性，本文主张应将其视为graded、干预特定的属性，需按样本评估。

## 核心贡献（创新点）
1. **首个系统评估**：提出首个针对VLMs视觉注意力忠实性的系统评估框架，通过token级因果扰动同时度量comprehensiveness（必要性）与sufficiency gap（充分性），与既往仅停留在热力图可视化的工作形成本质区别。
2. **三种处理模式识别**：发现VLMs视觉注意力忠实性呈异质性，分为Faithful-Sufficient、Faithful-Distributed和Non-Focal三种处理模式，突破了"统一忠实/不忠实"的二元认知。
3. **人与模型依赖的系统性分歧**：揭示人类标注ground-truth区域仅在约60%样本中满足comprehensiveness，模型实际依赖的视觉区域与人类直觉存在系统性偏差。
4. **跨任务与跨架构泛化**：将三种模式扩展至文档信息抽取（VRDU）和文档VQA（ChartQA）任务，并验证在Qwen2.5-VL与InternVL2.5两种不同架构上的一致性。

## 方法详解
**注意力token排序（§3.2）**：聚合所有层与所有头的注意力权重，为每个视觉token $v_i$ 计算聚合分数 $a_i = \sum_{l=1}^{L} \frac{1}{H}\sum_{h=1}^{H} \frac{1}{|\mathbf{y}|}\sum_{s=1}^{|\mathbf{y}|} \alpha_{l,h}(i, y_s)$，按降序排序选取top-k token集合 $S_k$。

**因果扰动（§3.3）**：在解码器第一层入口对选定token集合S进行zero-ablation，即将hidden state置零：$\mathbf{h}_i^{(0)} \leftarrow \mathbf{0}, \forall v_i \in S$，以消除token内容贡献同时保持序列长度与位置结构。

**忠实性度量（§3.4）**：
- **Comprehensiveness（Compressiveness）**：$\text{Comp}(S_k) = \frac{P(\mathbf{y}|\mathbf{V}, \mathbf{T}) - P(\mathbf{y}|\mathbf{V}\setminus S_k, \mathbf{T})}{P(\mathbf{y}|\mathbf{V}, \mathbf{T})}$，衡量top-k token是否为预测所必需。
- **Sufficiency Gap**：$\text{Sgap}(S_k) = \frac{P(\mathbf{y}|\mathbf{V}, \mathbf{T}) - P(\mathbf{y}|S_k, \mathbf{T})}{P(\mathbf{y}|\mathbf{V}, \mathbf{T})}$，衡量top-k token单独是否足以恢复预测。

**模式划分**：对(Comp, Sgap)二维空间应用两阶段Otsu阈值法：先在所有样本上按Comp阈值分割，再在高Comp子集上按Sgap二次分割，得到Mode A/B/C三类，并对边界样本进行人工微调。

## 实验与结果
**数据集**：VQAv2（90个样本，平衡yes/no、number、other三类问题）、VRDU（60个文档IE样本）、ChartQA（100个文档VQA样本）。

**基线策略对比（VQAv2）**：
- Overall Top-k：Mean Comp=0.813，Mean Sgap=0.469（最优）
- GT Object：Mean Comp=0.505，Mean Sgap=0.846
- Non-GT Top-k：Mean Comp=0.659，Mean Sgap=0.653
- Random：Mean Comp=0.502，Mean Sgap=0.599

**三种处理模式分布（VQAv2）**：
- Mode A Faithful-Sufficient：32.9%，Comp=0.978，Sgap=-0.026，Acc=92.9%
- Mode B Faithful-Distributed：47.1%，Comp=0.952，Sgap=0.818，Acc=100%
- Mode C Non-Focal：20.0%，Comp=0.213，Sgap=0.460，Acc=94.1%

**问题类型关联**：yes/no问题集中分布在Mode B（55.6%），open-ended "other"问题Mode C占比最高（34.5%）。

**跨数据集验证**：VRDU与ChartQA上Qwen2.5-VL均呈现Mode B特征（Comp高但Sgap接近1）。

**跨模型泛化**：InternVL2.5-8B上Top-k同样最优（Comp=0.733，Sgap=0.468），三种模式同样存在；InternVL在文档任务上更接近Mode A（Sgap在top-10%时降至0.189），而Qwen保持Mode B。

**模型规模一致性**：3B与7B参数规模下三种模式比例稳定，Top-k始终优于GT。

**人类标注vs模型注意力的分歧**：GT区域满足comprehensiveness的比例仅约60%，在Misaligned子集中GT-Comp低至-0.150，而模型仍能保持94.4%准确率。

## 相关工作脉络
1. **Attention as Explanation**：Jain & Wallace (2019) 认为注意力权重与特征重要性弱相关，Wiegreffe & Pinter (2019) 反驳认为约束注意力会导致模型行为变化，Bastings & Filippova (2020) 指出梯度saliency方法更优——本文将这些文本-only结论推广至视觉模态。
2. **Faithfulness Evaluation**：DeYoung et al. (2020) 提出comprehensiveness与sufficiency度量框架，Jacovi & Goldberg (2020) 形式化忠实性为graded属性——本文将其首次应用于VLM视觉注意力评估。
3. **VLM Interpretability**：Chefer et al. (2021)、Aflalo et al. (2022) 的Grad-CAM/注意力热力图方法仅生成解释性可视化，不验证因果贡献；Golovanevsky et al. (2025)、Jiang et al. (2025) 从机制角度分析注意力——本文直接通过扰动验证token级别的因果必要性。
4. **Attention Sparsity for Efficiency**：Chen et al. (2024a)、Shang et al. (2025)、Zhang et al. (2025) 假设低注意力token可安全剪枝/合并以实现推理加速——本文揭示视觉依赖具有异质性，盲目剪枝可能在Mode B/C样本上导致性能下降。
5. **Uppaal et al. (2026)**：同样研究VLM视觉忠实性，但聚焦于显式推理是否与输入图像 grounding，使用black-box方法——本文聚焦于模型内部注意力机制本身的因果忠实性，属于机制层面的不同问题。

## 局限性与未来方向
1. **限于动态分辨率架构**：本文聚焦采用动态分辨率视觉策略的VLMs，可能无法直接推广至固定分辨率输入模型（如LLaVA类）。
2. **未涉及推理型VLM**：未研究执行多步 deliberation 的reasoning-oriented VLMs，这类模型可能呈现不同的视觉注意力行为，留待未来工作。
3. **样本量有限**：VQA评估仅90个样本，虽通过重复抽样验证了稳定性，但覆盖范围仍有限。

## 研究启发与可借鉴点
1. **因果扰动评估范式**：采用zero-ablation+概率变化的因果干预框架替代纯相关性分析，可作为后续可解释性研究的标准方法；该范式可直接迁移至我们团队的多模态模型解释性研究。
2. **双维度忠实性度量**：同时评估comprehensiveness（必要性）与sufficiency gap（充分性），避免了单维度指标的片面性，值得在多模态归因研究中复用。
3. **人-机视觉依赖差异的实证警示**：人类标注ground-truth仅60%匹配模型因果依赖区域，提示在设计多模态训练数据或评估集时需警惕"人类中心偏差"，避免将人类标注直接作为模型忠实性黄金标准。
4. **三种处理模式的工程指导价值**：不同任务可能落入不同模式，在进行attention-aware token pruning/compression时，应对Mode B/C样本保留更多上下文，而非一刀切策略。
5. **跨架构对比的实验设计**：本文对比Qwen（native-resolution）与InternVL（dynamic-tile）两种主流架构，揭示了架构设计对模式选择的影响，为后续架构-可解释性联合分析提供了参照模板。

## 关键术语表
**Comprehensiveness（Compressiveness）**：衡量被选中的token集合是否为模型预测所必需，通过移除这些token后预测概率的归一化下降程度来量化。

**Sufficiency Gap（Sgap）**：衡量被选中的token集合是否足以独立恢复模型预测，通过仅保留这些token后预测概率的归一化下降程度来量化。

**Faithful-Sufficient（Mode A）**：top-k注意力token既必要又充分的处理模式，占比约32.9%，通常对应可从局部物体证据直接回答的问题。

**Faithful-Distributed（Mode B）**：top-k注意力token必要但不足够的处理模式，占比约47.1%，需要整合局部证据与更广泛的视觉上下文。

**Non-Focal（Mode C）**：无集中注意力区域对预测具有决定性因果作用的模式，占比约20%，视觉信息作为全局context trigger而非局部证据。

**Zero-ablation**：将选定token的hidden state替换为零向量以消除其内容贡献的扰动方法，相比直接删除token可保持序列长度与位置编码不变。

**Two-stage Otsu Thresholding**：先对所有样本按Comprehensiveness做二值分割，再在高Comp子集上按Sgap二次分割，实现三个模式的自动化划分。

## 可复现要素
- **数据集**：VQAv2（公开）、VRDU（公开）、ChartQA（公开）；VQAv2上每样本附带半自动生成的ground-truth region标注（基于COCO实例分割+人工精修）
- **代码/权重**：论文未明确声明开源代码，使用了Qwen2.5-VL-7B和InternVL2.5-8B等开源模型
- **关键超参**：greedy decoding（temperature=0）；token消融层级为language backbone第一解码器层入口；k值等于GT region覆盖的视觉token数量；Otsu阈值法确定模式边界
- **模型尺度实验**：Qwen2.5-VL-3B/7B、InternVL2.5-4B/8B
