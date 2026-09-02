---
title: "SinkPruner-Sink-Free-Visual-Token-Pruning-for-Multimodal-Lar"
source: https://arxiv.org/pdf/2609.01004v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 07:22:38"
field: "多模态大模型高效推理"
keywords: ["visual token pruning", "multimodal large language models", "attention sink", "token compression", "efficient inference", "training-free method"]
innovations: ["发现高范数异常token是视觉冗余与注意力汇聚的根源", "提出scale-free top-ρ相对排名规则实现零校准跨架构迁移", "设计即插即用visual sanitizer模块正交增强现有剪枝方法"]
benchmarks: ["LLaVA-1.5-7B", "Qwen2.5-VL-7B", "MMBench", "MME", "POPE", "ScienceQA", "MMMU", "MM-Vet", "MVBench", "SEED-Bench"]
---

# 论文速读：SinkPruner: Sink-Free Visual Token Pruning for Multimodal Large Language Models

## 一句话总结
提出SinkPruner，一种免训练的级联视觉token剪枝框架，通过视觉过滤器预先剔除高范数异常token以抑制LLM解码器中的注意力汇聚和分散现象，实现激进压缩下的高保真多模态大模型推理（88.9%剪枝保留96.5%性能）。

## 研究问题与动机
1. **计算瓶颈**：MLLM将图像/视频投影为大量视觉token后，Transformer注意力的二次复杂度（$O(n^2)$）导致推理开销巨大，尤其在长视频场景下（1小时视频可达200万+ token）。
2. **高范数异常token误导视觉中心剪枝**：现有方法（VisionZip、Faster-VLM等）依赖[CLS]注意力保留重要token，但高范数异常token（主要来自均匀背景区域如天空、墙壁）会吸引异常高的注意力，导致剪枝算法优先保留这些冗余token，浪费预算。
3. **注意力汇聚与分散破坏文本引导剪枝**：LLM解码器中存在massive activations（注意力汇聚，少数token吸收不成比例的大量注意力）和text-visual attention dispersion（注意力分散），使得基于文本-视觉注意力的token重要性估计不可靠，尤其在激进压缩比下。
4. **动态分辨率架构缺乏稳定空间先验**：Qwen2.5-VL等采用动态分辨率的MLLM无法使用固定网格的[CLS]注意力，现有剪枝方法在该设置下性能急剧下降。

## 核心贡献（创新点）
1. **发现并量化高范数异常token的冗余性**：首次系统揭示视觉编码器中high-norm outlier tokens在空间维度（邻居高度相似）和特征维度（对内余 similarity极高、代表 collapse）均高度冗余，且是注意力汇聚的主要根源——区别于仅关注注意力分数的既往工作。
2. **Scale-free top-ρ 高范数识别准则**：提出相对排名而非绝对阈值的高范数检测规则（默认ρ=1%），无需每模型校准，可泛化至不同尺度分布的视觉编码器（如CLIP与DINOv2）。
3. **Visual Sanitizer（视觉过滤器）**：将高范数异常token平均池化为单一sink token以保留全局信息，同时对低范数候选token采用"显著性+多样性"混合选择策略，从源头净化视觉流——区别于直接丢弃的既往方法。
4. **Text-Guided Pruner（文本引导剪枝器）**：在LLM解码器早期层利用累积文本到视觉注意力进行query-aware token选择，得益于上游净化，该方法能更自信地识别真正与文本指令语义对齐的token。
5. **即插即用的可迁移性**：视觉过滤器作为正交增强模块可无缝集成至现有剪枝方法（如VisionZip），在相同token预算下 consistently 提升0.9%~2.6%性能。

## 方法详解
### 3.1 前提与效率瓶颈
- 视觉编码器（如CLIP-ViT）产生token序列 $X_{vis} = [x_{CLS}; x_{img}^1, ..., x_{img}^{n_{vis}-1}]$，计算CLS注意力 $A_{vis}$。
- LLM处理拼接序列 $X = [x_{sys}; X_{vis}; x_{txt}]$，使用RoPE + 下三角因果掩码的自注意力。
- 总FLOPs近似为 $T \times (4nd^2 + 2n^2d + 2ndm)$，其中$n = n_{sys} + n_{vis} + n_{txt}$，视觉token $n_{vis}$ 主导二次项。

### 3.2 高范数token的冗余性分析
- **双峰分布**：token norm直方图显示清晰双峰，大多数token < 60，约1%异常token ∈ [60, 250]。
- **空间冗余**：高范数token与4-近邻的余弦相似度显著高于低范数token，主要源于均匀背景区域。
- **特征冗余**：高范数token子集内部pairwise余弦相似度极高（representation collapse），而低范数token保持多样性。
- **注意力汇聚陷阱**：高范数token持续获得最高[CLS]注意力，成为attention sink，误导基于注意力的剪枝。

### 3.3 SinkPruner框架
**Visual Sanitizer（视觉过滤器）**：
1. 计算每个token的 $L_2$ 范数 $n_i = \|x_i\|_2$，按top-ρ规则划分高范数集 $\mathbf{X}_{high}$（ρ=1%）和低范数候选集 $\mathbf{X}_{low}$。
2. 高范数token聚合为单一sink token：$x_{sink} = \frac{1}{|\mathbf{X}_{high}|} \sum_{x \in \mathbf{X}_{high}} x$（公式1）。
3. 从$\mathbf{X}_{low}$中选择：
   - **显著性子集** $\mathbf{X}_{res}$：取CLS注意力top-$k_{res}$的token（公式2）。
   - **多样性子集** $\mathbf{X}_{div}$：迭代采样与当前已选集合S余弦相似度最低的token（公式3），使用batched近似（每轮选b=16个）加速。
4. 最终 purified 表示：$Z = [x_{sink}, \mathbf{X}_{res}, \mathbf{X}_{div}]$。

**Text-Guided Pruner（文本引导剪枝器）**：
- 在LLM解码器第$L_t$层计算全局相关度得分：$\tilde{p}_j = \frac{1}{L_{txt}} \sum_{i=1}^{L_{txt}} \mathrm{Softmax}(\mathbf{Q}_{text} \cdot \mathbf{K}_{vis}^\top)_{i,j}$（公式4）。
- 保留top-K视觉token，实现query-aware选择性保留。

**无[CLS]编码器的扩展**：对于Qwen2.5-VL等无[CLS] token的架构，使用received visual self-attention作为salience score替代：$s_j = \frac{1}{HN} \sum_{h=1}^H \sum_{i=1}^N A_{ij}^{(h)}$（公式5）。

## 实验与结果
**基准测试**：
- 图像-语言：GQA、MMBench（及中文MMB-CN）、MME、POPE、ScienceQA、VQA-v2、TextVQA、MMStar、MMMU、AI2D、MM-Vet（共12个）。
- 视频-语言：MVBench、SEED-Bench、NextQA、VideoMME（共4个）。

**主要结果（LLaVA-1.5-7B）**：
- **Retain-64（剪枝率88.9%）**：平均性能保留**96.5%**，超越VisionZip（+3.3%）、HoloV（+1.8%）；POPE 83.8 vs 80.3（HoloV）、MME 1754 vs 1715。
- **Retain-32（剪枝率94.4%）**：平均保留**91.2%**，超越VisPruner（+4.0%）、VisionZip（+6.0%）；在MM-Vet（29.36 vs 25.23）和AI2D（53.63 vs 51.81）上优势最大。
- **高难度推理基准**（Tab. 4）：32-token预算下保留93.7%（vs VisionZip 88.7%、FastV 85.9%），证明剪枝不损害多步推理所需的证据保留。

**Qwen2.5-VL-7B（动态分辨率架构）**：
- 剪枝率66.7%：保留98.6%，超越HoloV（+5.0%）、VisionZip（+7.2%）。
- 剪枝率88.9%：保留91.8%，超越HoloV（+5.6%）、VisionZip（+4.1%）。

**视频任务（Tab. 3，80%剪枝）**：平均保留**98.0%**，超越DART（+1.4%）、DivPrune（+2.0%）。

**效率对比（Tab. 5，90%剪枝）**：总推理时间减少33.3%，延迟86.0ms（vs VisionZip 77.4ms），但精度97.1% vs 91.8%。

**消融研究（Tab. 6）**：
- 移除视觉过滤器：MMB从61.6降至51.4（-10.2%），验证上游净化关键性。
- 移除高范数去除：MME从1754降至1706，证明高范数过滤是主要贡献。
- 高范数聚合 vs 直接删除：聚合略优（1737 vs 1705）。

**可迁移性（Tab. 7）**：将高范数过滤器集成至VisionZip，在相同token预算下MMB提升+1.3%~+2.6%，POPE提升+0.9%~+1.7%。

**Sink ratio降低（Fig. 6）**：visual sanitizer将LLM解码器中的注意力汇聚token比例从14.23%降至3.85%（降幅73%）。
**注意力熵降低（Tab. 18）**：text-visual attention entropy从6.359降至4.851，选择更自信。

## 相关工作脉络
1. **Vision-centric剪枝**（VisionZip、Faster-VLM、VisPruner）：依赖[CLS]注意力或特征相似度保留重要token——本文指出这些方法被高范数异常token误导，优先保留背景噪声。
2. **Text-guided剪枝**（FastV、SparseVLM）：利用LLM decoder的text-visual注意力选择token——本文揭示该方法受massive activations和attention dispersion困扰，选择不可靠。
3. **Token合并/聚类**（ToMe、PruMerge）：基于特征相似度合并token——本文聚焦于剔除而非合并冗余token，且引入范数维度。
4. **渐进式剪枝**（MustDrop、PDrop）：逐步剔除视觉token——本文采用coarse-to-fine两级设计（先过滤再文本引导），而非单层渐进。
5. **Hierarchical/Vision-exit策略**（HiDrop、AutoPrune）：控制token进入/退出LLM层的时机——本文聚焦于进入前的上游净化。
6. **Holistic context retention**（HoloV）：激进剪枝时保留全局上下文——本文通过高范数聚合保留全局信息，同时通过多样性选择保留空间覆盖。

## 局限性与未来方向
1. **仅离线推理**：当前评估局限于预记录、固定长度的静态图像和视频片段；不支持在线流式场景（如连续机器人感知）。
2. **无法重访剪枝决策**：流式场景下过去视觉历史动态更新且无法回溯修改剪枝结果，是开放挑战。
3. **TextVQA在高分辨率下略有劣势**：当视觉token经tiled high-resolution编码时，小glyph patch本身可能携带大特征范数，纯范数规则可能将真实文本token与背景异常值一同聚合；作者建议引入边缘密度等文本感知机制作为未来改进。
4. **未探索训练微调**：方法为training-free，未结合轻量微调以进一步提升激进剪枝下的性能。

## 研究启发与可借鉴点
1. **高范数异常token作为普适信号**：发现现象在CLIP和DINOv2中均存在，表明该特性是现代视觉编码器的普遍属性，可推广至其他视觉骨干网络。
2. **Scale-free相对排名准则**：避免绝对阈值依赖，通过top-ρ相对规则实现跨架构零校准迁移，为其他特征分析任务提供范式。
3. **上游净化改善下游注意力分布**：在LLM解码器前预处理视觉流，显著降低massive activations（-73%）和attention entropy，证明coarse-to-fine级联设计的优越性。
4. **正交增强模块的可插拔性**：视觉过滤器可作为即插即用组件集成至现有剪枝框架，为社区提供低成本的性能提升路径。
5. **多样性选择的batched近似实现**：将串行farthest-point采样转化为batch矩阵运算（b=16），在不损失显著性能的前提下实现约16×加速，为实时系统提供工程参考。

## 关键术语表
**High-norm outlier tokens**：特征$L_2$范数异常大的视觉token，主要来自均匀背景区域，在空间与特征维度均高度冗余，却吸引不成比例的高注意力，形成attention sink。
**Attention sink / Massive activations**：LLM解码器中少数token吸收不成比例大量注意力的现象，源于冗余视觉输入，导致token重要性估计失真。
**Attention dispersion**：text-visual注意力分布过于分散、熵值高，模型无法形成对query相关区域的 confident ranking，降低剪枝可靠性。
**Visual sanitizer**：SinkPruner的第一阶段模块，通过高范数过滤、聚合与多样性选择净化视觉序列，为下游文本引导剪枝提供clean input。
**Scale-free top-ρ rule**：基于相对排名而非绝对阈值识别高范数token的规则（默认ρ=1%），无需每模型/每编码器校准，具有跨架构迁移性。
**Salience pool vs Diversity pool**：视觉过滤器中对低范数候选token的两种选择策略——显著性子集基于CLS/received注意力保留突出区域，多样性子集通过余弦相似度去重保留空间覆盖。
**Text-guided pruner**：SinkPruner的第二阶段模块，利用LLM decoder早期层的累积text-to-vision注意力进行query-aware token保留。
**CLS-free salience score**：针对无[CLS] token的编码器（如Qwen2.5-VL）设计的替代方案，使用received visual self-attention平均值作为token显著性度量。

## 可复现要素
- **数据集**：所有基准测试（GQA、MMBench、MME、POPE、ScienceQA、VQA-v2、TextVQA、MMStar、MMMU、AI2D、MM-Vet、MVBench、SEED-Bench、NextQA、VideoMME）均为公开数据集。
- **代码**：论文声明代码已开源，链接为 https://github.com/LaVi-Lab/SinkPruner（注：论文正文未直接列出，但在Abstract末尾提及）。
- **权重**：使用官方LLaVA-1.5-7B、LLaVA-1.5-13B、LLaVA-NeXT-7B、Qwen2.5-VL-7B预训练权重。
- **关键超参**：
  - 高范数比例ρ = 1%（所有backbone和benchmark共享，无需校准）。
  - Salience pool与Diversity pool大小相等，均匹配目标token budget（如Retain-64时各64）。
  - 批处理大小b = 16（batched diversity selection）。
  - 剪枝层：(2, 6, 15)（对LLaVA-1.5）；Qwen2.5-VL采用三层渐进保留。
  - 实验硬件：NVIDIA A800-SXM4-80GB GPU，Python 3.10 + PyTorch 2.1.2 + CUDA 12.1。
