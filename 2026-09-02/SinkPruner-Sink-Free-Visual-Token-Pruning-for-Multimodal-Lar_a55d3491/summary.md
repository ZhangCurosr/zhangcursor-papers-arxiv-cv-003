---
title: "SinkPruner-Sink-Free-Visual-Token-Pruning-for-Multimodal-Lar"
source: https://arxiv.org/pdf/2609.01004v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 19:13:11"
field: "多模态大模型高效推理"
keywords: ["视觉Token剪枝", "多模态大语言模型", "注意力汇聚", "无需训练", "高效推理"]
innovations: ["发现高范数异常Token的空间和特征双重冗余性并提出预过滤机制", "提出训练无关的级联剪枝框架SinkPruner实现粗到细的Token压缩", "验证高范数过滤模块的可迁移性作为即插即用组件"]
benchmarks: ["MMBench", "MME", "POPE", "ScienceQA", "VQA-v2", "MMStar", "MMMU", "AI2D", "MM-Vet", "MVBench", "SEED-Bench", "NextQA", "VideoMME"]
---

# 论文速读：SinkPruner: Sink-Free Visual Token Pruning for Multimodal Large Language Models

## 一句话总结
本文提出了一种无需训练的级联视觉Token剪枝框架SinkPruner，通过"视觉净化器"预过滤高范数异常Token来消除注意力汇聚和分散问题，再结合"文本引导剪枝器"保留与查询语义对齐的Token，在LLaVA-1.5和Qwen2.5-VL上分别实现89%和89%剪枝率下保持96.5%和91.8%的原始性能。

## 研究问题与动机
- **MLLM视觉Token序列过长导致推理瓶颈**：Transformer注意力机制的二次计算复杂度与超长视觉序列叠加，造成计算和内存开销巨大，限制边缘设备部署。
- **现有剪枝方法受高范数异常Token误导**：Vision-centric方法依赖CLS注意力评分选择Token，但异常高范数Token（通常来自背景区域）会吸引不成比例的高注意力，导致真正有意义的低范数Token被错误丢弃。
- **LLM解码器存在注意力汇聚和分散问题**：Text-guided方法依赖跨模态注意力选择Token，但LLM解码器中大量激活（attention sinks）使注意力分布不均匀，同时文本-视觉注意力过于分散，难以可靠排序相关Token。
- **高范数Token的双重冗余性未被利用**：这些Token在空间维度上与相邻Patch高度相似（重复背景），在特征维度上内部余弦相似度极高（表示坍塌），却仍占据宝贵的Token预算。

## 核心贡献（创新点）
- **发现并量化高范数异常Token的冗余性**：首次系统证明高范数Token在空间相似性和特征相似度上均高度冗余，且与注意力异常热点空间对齐，区别于以往仅关注注意力分数的剪枝思路。
- **提出训练无关的级联剪枝框架SinkPruner**：采用粗到细的级联设计，视觉净化器先行过滤高范数冗余，文本引导剪枝器后行保留语义相关Token，区别于单阶段剪枝方法。
- **视觉净化器的三重设计**：包含高范数识别（top-ρ规则）、高范数聚合（平均池化为sink Token）和低范数选择（显著性+多样性双池策略），相比直接丢弃高范数Token保留全局信息。
- **无CLS编码器的通用替代方案**：提出基于平均接收自注意力的显著性评分，使SinkPruner可直接适配Qwen2.5-VL等无CLS Token的动态分辨率架构。
- **可视化净化器的可迁移增强能力**：将高范数过滤模块作为即插即用组件集成到VisionZip等现有方法中，一致带来0.9%-2.6%性能提升，证明其正交性。

## 方法详解
**整体流程**：输入图像经Vision Encoder得到视觉Token序列X，先通过Visual Sanitizer净化，再通过Text-Guided Pruner在LLM早期层逐步剪枝。

**Visual Sanitizer**：
1. **高范数识别**：计算每个Token的L₂范数，按top-ρ规则（默认ρ=1%）划分高范数集合X_high和低范数集合X_low，该规则与编码器绝对尺度无关。
2. **高范数聚合**：对X_high做平均池化得到单一sink Token：$x_{sink} = \frac{1}{|X_{high}|}\sum_{x \in X_{high}} x$，保留全局信息而非直接丢弃。
3. **低范数选择**：从X_low中选取两类Token：
   - 显著性池：保留CLS注意力top-k_res个Token（$X_{res}$）
   - 多样性池：通过最远点采样保留与已选集合余弦相似度最低的Token（$X_{div}$），公式为$x^* = \arg\min_{x \in \mathcal{R}}(\max_{s \in S}\text{CosSim}(x,s))$
4. **拼接输出**：最终净化序列$Z = [x_{sink}, X_{res}, X_{div}]$。

**Text-Guided Pruner**：
- 在LLM解码器早期层（如layer 2, 6, 15）计算文本到视觉的跨模态注意力。
- 对每个视觉Token $z_j$，聚合所有文本Token的注意力权重得到全局相关性得分：$\tilde{p}_j = \frac{1}{L_t}\sum_{i=1}^{L_t}\text{Softmax}(Q_{text} \cdot K_{vis}^T)_{i,j}$
- 保留得分top-K的Token，采用渐进式剪枝策略（多阶段逐步缩减）。

**无CLS替代方案**（针对Qwen2.5-VL）：
- 用平均接收自注意力替代CLS注意力：$s_j = \frac{1}{HN}\sum_{h=1}^{H}\sum_{i=1}^{N}A_{ij}^{(h)}$，衡量其他Token对该Token的关注程度。

**Batched Diversity Selection**：
- 为加速GPU计算，采用批量近似：每轮同时选取b=16个最不相似的Token，将串行轮数减少约16倍。

## 实验与结果
**数据集与基准**：
- 图像理解：GQA、MMBench、MMB-CN、MME、POPE、ScienceQA、VQA-v2、TextVQA、MMStar、MMMU、AI2D、MM-Vet
- 视频理解：MVBench、SEED-Bench、NextQA、VideoMME

**主要结果（LLaVA-1.5-7B）**：
- 保留64 Token（剪枝率88.9%）：平均性能保持96.5%，超越VisionZip（93.2%）+3.3%、HoloV（94.7%）+1.8%
- 保留32 Token（剪枝率94.4%）：平均性能保持91.2%，超越VisPruner（87.2%）+4.0%
- 难推理任务（32 Token）：平均保持93.7%，在MM-Vet上达29.36（VisionZip 25.23，FastV 22.66）

**主要结果（Qwen2.5-VL-7B）**：
- 剪枝率66.7%：保持98.6%，超越HoloV（93.6%）+5.0%
- 剪枝率77.8%：保持96.3%，超越VisionZip（93.4%）+2.9%
- 剪枝率88.9%：保持91.8%，超越VisionZip（87.7%）+4.1%

**视频任务**（Qwen2.5-VL-7B，80%剪枝率）：
- 平均保持98.0%，超越DART（96.6%）和DivPrune（96.0%）

**推理效率**（90%剪枝率）：
- SinkPruner：总推理时间减少33.3%，准确率保持97.1%，优于VisionZip（91.8%）和FastV（65.5%）

**消融实验**：
- 移除Visual Sanitizer导致灾难性下降（MMB: -10.2%），验证预过滤的必要性
- 移除High-Norm Removal比移除Deduplication影响更大，确认高范数过滤是核心贡献
- 高范数聚合略优于直接丢弃（MME: 1754.1 vs 1737.0）

**注意力分析**：
- SinkPruner将LLM解码器中attention sink Token比例从14.23%降至3.85%（减少约73%）
- 文本-视觉注意力熵从6.359降至4.851，显著提升选择置信度

**可迁移性**：
- 集成到VisionZip中，64 Token时MMB提升2.0%（92.9%→94.9%），POPE提升0.9%（89.6%→90.5%）

## 相关工作脉络
- **Vision-centric剪枝**：ToMe、VisionZip、Faster-VLM等依赖CLS注意力或特征相似性，但未考虑高范数异常Token的干扰，SinkPruner通过预过滤解决此问题。
- **Text-guided剪枝**：FastV、SparseVLM等利用LLM解码器的跨模态注意力，但受注意力汇聚和分散影响，SinkPruner通过净化输入改善注意力分布。
- **Progressive dropping**：MustDrop、PDrop、HiDrop等采用渐进剪枝策略，SinkPruner同样采用多阶段剪枝但增加了预净化步骤。
- **Token merging**：PruMerge、ToMe等通过合并相似Token压缩序列，SinkPruner则直接丢弃冗余Token并聚合高范数异常。
- **Holistic context retention**：HoloV、VisPruner等关注全局上下文保留，SinkPruner通过显式分离高/低范数Token实现更精确的冗余过滤。
- **Attention sink研究**：Kang et al. (2025) 发现MLLM中存在视觉注意力汇聚现象，SinkPruner从源头减少此类异常Token而非事后修正。

## 局限性与未来方向
- **仅支持离线推理**：当前评估仅限于预录制的固定长度输入，未探索连续视频流等在线场景，未来需适配动态更新的时序上下文。
- **高分辨率文本感知不足**：在LLaVA-NeXT的文本VQA任务上略逊于VisPruner，作者归因于小字体Patch可能携带高范数，建议未来引入文本感知机制（如边缘密度豁免）。
- **超参数敏感性未完全探索**：虽然展示了较宽的鲁棒区间，但未系统性研究不同场景下的最优配置。
- **仅评估CLIP和DINOv2类编码器**：未来需验证在非Transformer架构（如CNN、State Space Model）上的适用性。

## 研究启发与可借鉴点
- **高范数异常Token的量化分析框架**：可通过L₂范数分布的双峰分离、空间邻域相似性、特征空间内聚度三重验证冗余性，该方法可迁移到其他视觉Token压缩场景。
- **级联净化-剪枝设计范式**：先净化再剪枝的两阶段策略有效解耦了冗余过滤和语义选择，可推广到视频Token压缩、3D点云处理等领域。
- **无CLS编码器的通用替代方案**：接收自注意力作为显著性评分的思路不依赖特定架构，可适配各类Vision Transformer变体。
- **注意力汇聚的度量与缓解**：通过sink ratio和注意力熵量化注意力质量，为后续研究提供可复用的评估指标。
- **即插即用的模块设计**：高范数过滤模块的正交性证明简化了与现有方法的集成路径，适合科研团队快速验证假设。

## 关键术语表
**High-norm outlier tokens**：特征L₂范数异常大的视觉Token，通常来自背景区域，在空间上和特征上高度冗余，但会吸引不成比例的高注意力。
**Attention sink**：MLLM解码器中少量Token吸引 disproportionate 大注意力的现象，导致跨模态注意力分布失真。
**Attention dispersion**：文本-视觉注意力过于分散、缺乏 discriminative focus 的问题，使Token选择置信度降低。
**Visual sanitizer**：SinkPruner的前置模块，通过识别和聚合高范数异常Token来净化视觉序列。
**Top-ρ rule**：基于相对排名的尺度无关高范数识别方法，选取Norm最大的ρ fraction Token作为异常集合。
**Salience pool / Diversity pool**：Visual sanitizer中低范数Token选择的两个子池，分别基于显著性（CLS注意力）和多样性（最远点采样）筛选。
**Progressive pruning**：在LLM多层中逐步缩减Token数量的剪枝策略，平衡早期冗余过滤和晚期语义保留。
**CLS-free salience score**：针对无[CLS] Token编码器（如Qwen2.5-VL）的替代显著性评分，基于平均接收自注意力计算。

## 可复现要素
- **数据集**：MMBench、MME、POPE、ScienceQA、GQA、VQA-v2、TextVQA、MMStar、MMMU、AI2D、MM-Vet、MVBench、SEED-Bench、NextQA、VideoMME（公开基准）
- **代码**：已开源，地址https://github.com/LaVi-Lab/SinkPruner
- **模型权重**：使用LLaVA-1.5-7B、Qwen2.5-VL-7B官方权重
- **关键超参**：ρ=1%（高范数阈值）、k_res和k_div匹配目标Token预算、剪枝层(2,6,15)、批量大小b=16
- **硬件**：NVIDIA A800-SXM4-80GB GPU
- **框架**：Python 3.10 + PyTorch 2.1.2 + CUDA 12.1
- **评估工具**：lmms-eval harness
