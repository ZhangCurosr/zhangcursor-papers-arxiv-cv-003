---
title: "MEMO-Multi-Level-Entity-Aware-Memory-for-Streaming-Video-Und"
source: https://arxiv.org/pdf/2609.38900v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 01:46:33"
field: "流式视频理解与长程视觉记忆"
keywords: ["streaming video understanding", "multimodal large language models", "structured memory", "entity-aware representation", "online temporal chunking", "query-specific retrieval"]
innovations: ["提出多级实体感知结构化内存框架，解耦轻量索引与高分辨率视觉证据存储", "设计基于全局-实体-空间相似度的自适应在线分块机制，替代固定长度切割", "实现查询特定并行检索，无需训练即可提升多种 MLLM 的流式推理性能"]
benchmarks: ["StreamingBench", "OVO-Bench"]
---

# 论文速读：MEMO-Multi-Level-Entity-Aware-Memory-for-Streaming-Video-Und

## 一句话总结
论文提出 **MEMO**（Multi-Level Entity-Aware Memory）框架，通过**多级实体感知的结构化内存**解决无界流式视频理解中的长程语义保持与精准证据检索问题。该方法无需训练即可即插即用，在 StreamingBench 与 OVO-Bench 上显著超越现有开源基线及专有模型。

## 研究问题与动机
1. **流式视频的长程语义保持难题**：真实场景（如自动驾驶、第一人称视觉）中视频以无界流形式到达，现有模型多假设视频已完整录制，无法有效处理持续到达的视觉流。
2. **现有内存机制的局限性**：早期方法依赖 GPU 内紧凑压缩（如 token/KV cache），容量有限导致细粒度细节丢失；后期方法扩展至 CPU/磁盘存储，但仍以全局或粗粒度表示为主，缺乏对实体动态与空间结构的显式建模。
3. **实体级信息的重要性未被充分挖掘**：对象是场景语义的主要载体，其动态变化可指示事件边界，但现有方法未能将实体级细粒度表示与全局上下文有效结合，导致检索与推理精度受限。

## 核心贡献（创新点）
1. **提出 MEMO 框架**：构建多级实体感知的结构化内存，将无界流切分为语义连贯的块，并解耦轻量索引与高分辨率视觉证据的存储。
2. **设计多级感知与在线分块机制**：联合建模全局语义、实体动态与空间结构，通过自适应相似度阈值实现语义连贯的时间分块，替代固定长度切割。
3. **实现查询特定的证据检索**：基于轻量全局与实体级索引进行并行相似度计算，按需召回高分辨率视觉内容，平衡存储效率与推理精度。
4. **免训练、即插即用的增强模块**：无需更新骨干模型参数，可无缝集成至多种 MLLM（如 LLaVA-OV、Qwen-VL 系列），在多个基准上稳定提升性能。
5. **详细的效率与消融分析**：证明 MEMO 在 GPU 内存占用、延迟与准确率之间取得良好平衡，且多级表征与检索策略具有显著互补性。

## 方法详解
MEMO 由四个核心组件构成，流水线处理流式视频帧序列 $\{x_1, ..., x_t\}$：

1. **多级实体感知（Multi-Level Entity-Aware Perception）**  
   - **全局语义相似度** $S_{\text{global}}^{(t)}$：使用 CLIP 提取相邻帧的全局特征向量，计算 L2 归一化后的余弦相似度。  
   - **局部语义连续性** $S_{\text{local}}^{(t)}$：借助 Grounding DINO 检测与 SAM 分割提取对象，通过 EMA 维护历史语义状态 $\bar{f}_i^{(t)}$，计算当前观测与历史状态的余弦相似度，按对象像素面积加权聚合，并引入新对象出现与消失的惩罚项 $P_{\text{penalty}}^{(t)}$ 以敏感捕捉语义跳跃。  
   - **空间结构连续性** $S_{\text{spatial}}^{(t)}$：对匹配对象计算掩码 IoU 得分 $S_{\text{iou}}^{(t)}$ 与边界框中心位移一致性 $S_{\text{disp}}^{(t)}$，加权融合（$\alpha=0.6$）。  
   - **总相似度** $S_{\text{total}}^{(t)} = \lambda_s S_{\text{spatial}}^{(t)} + \lambda_l S_{\text{local}}^{(t)} + \lambda_g S_{\text{global}}^{(t)}$。

2. **在线时间分块（Online Temporal Chunking）**  
   - 维护活跃段缓冲区，基于滑动窗口内 $S_{\text{total}}$ 的均值与标准差动态计算阈值 $\tau^{(t)} = \text{clip}(\mu^{(t)} - m\cdot\sigma^{(t)}, \tau_{\min}, \tau_{\max})$。  
   - 仅当低相似度持续 $N_{\text{confirm}}$ 帧且满足最小段长约束时，才确认语义边界并输出一个语义连贯的 chunk。

3. **结构化内存构建（Structured Memory Construction）**  
   - **高分辨率视觉证据**：每个 chunk 的原始视频张量 $X^{(i)}$ 存储于 CPU 内存，仅在检索时按需编码。  
   - **轻量结构化内存**：GPU 侧维护每个 chunk 的索引 $M^{(i)} = \langle \mathcal{M}_{\text{global}}^{(i)}, \mathcal{M}_{\text{entity}}^{(i)} \rangle$。  
     - 全局表示 $\mathcal{M}_{\text{global}}^{(i)}$ 为 chunk 内帧 CLIP 特征的加权平均（权重与相似度下降程度相关）。  
     - 实体表示 $\mathcal{M}_{\text{entity}}^{(i)}$ 为 chunk 结束时刻各核心对象的 EMA 语义状态向量堆叠，仅保留未隐藏超过半额阈值的活跃实体。

4. **查询特定证据检索（Query-Specific Evidence Retrieval）**  
   - 使用 CLIP 文本编码器获取查询特征 $T_Q$。  
   - 并行计算每个 chunk 的全局相似度 $\mathrm{Sim}_{\text{global}}^{(i)} = \cos(T_Q, \mathcal{M}_{\text{global}}^{(i)})$ 与最佳局部相似度 $\mathrm{Sim}_{\text{local}}^{(i)} = \max_{e \in \mathcal{M}_{\text{entity}}^{(i)}} \cos(T_Q, e)$。  
   - 检索得分 $\mathrm{Score}(Q, C^{(i)}) = \lambda_{\text{global}} \mathrm{Sim}_{\text{global}}^{(i)} + \lambda_{\text{local}} \mathrm{Sim}_{\text{local}}^{(i)}$，选取 Top-K 个 chunk 召回其视觉证据，并与当前活跃段帧采样至 MLLM 视觉上下文预算上限（如 16 帧）。

## 实验与结果
- **数据集**：StreamingBench（实时流式视频理解）与 OVO-Bench（时间戳锚定任务，含历史检索、实时感知、主动响应）。
- **基线对比**：涵盖专有模型（Gemini 1.5 Pro、GPT-4o）、开源离线 MLLM（LongVA、LLaVA-Video）、训练型在线 MLLM（StreamForest、TimeChat-Online 等）及免训练在线方法（ReKV、LiveVLM、FluxMem 等）。
- **主要结果**：
  - 结合 **Qwen3-VL-8B** 时，MEMO 在 OVO-Bench 提升至 **76.0%**（+5.9%），StreamingBench 提升至 **83.7%**（+10.5%），超越 Gemini 1.5 Pro 与 GPT-4o。
  - 在 LLaVA-OneVision-7B 上，MEMO 优于 ReKV 9.3%（OVO-Bench）和 4.1%（StreamingBench），超越训练型 SOTA StreamForest 14.8% 与 6.4%。
  - 效率分析：基于 Qwen2.5-VL-7B，峰值 GPU 内存 **21.1 GB**，平均延迟 **1.2 秒**，准确率 78.5%，较 TimeChat-Online 延迟降低 67.6%、内存减少 10.9%。
- **消融实验**验证：
  - 感知与记忆模块相互增益（表 2）。
  - 全局与实体索引均必要，单级退化导致性能下降（表 4）。
  - 自适应分块优于固定窗口长度切割（图 4 右）。
  - 查询特定检索优于最近邻、随机检索等策略（图 4 左）。

## 相关工作脉络
1. **内部紧凑内存方法**（如 Flash-VStream、StreamMem、TimeChat-Online）：通过 KV cache 压缩或 token 剪枝维持低延迟，但受 GPU 显存限制，细粒度信息损失严重。MEMO 通过解耦存储突破容量瓶颈。
2. **外部扩展内存方法**（如 ReKV、LiveVLM、Vista）：将历史特征卸载至 CPU/磁盘，但检索单元仍为全局或粗粒度上下文，缺乏实体级结构化索引。MEMO 引入实体 EMA 状态作为细粒度检索键。
3. **训练型在线模型**（如 VideoLLM-online、Dispider、StreamForest）：依赖额外训练数据与优化，成本高且泛化受限。MEMO 为免训练即插即用模块，兼容多种 MLLM。
4. **离线长视频理解模型**（如 LongVA、LLaVA-Video、FluxMem）：假设视频完整可用，无法处理流式到达场景。MEMO 针对在线增量处理设计，支持实时查询响应。
5. **对象轨迹与记忆机制**（如 ObjectStream）：利用潜对象作为记忆锚点，但未显式建模全局-实体多级相似性与自适应分块。MEMO 将对象动态与全局语义统一纳入分块与检索决策。

## 局限性与未来方向
1. **上游感知误差传播**：依赖 Grounding DINO、SAM 等外部检测/分割模型，误差可能影响分块与检索质量。
2. **隐式场景状态表征不足**：当前实体索引难以捕捉隐性属性或对象缺席所隐含的场景信息。
3. **跨 chunk 长期身份关联缺失**：对象跨长时段的重识别与记忆巩固尚未实现，限制小时级/天级流的处理。
4. **未来方向**：显式场景状态内存建模、证据重排序机制、重要性感知的内存整合、更鲁棒的跨帧实体关联算法。

## 研究启发与可借鉴点
1. **多级解耦存储范式**：轻量索引（GPU）与高分辨率证据（CPU）分离的设计，为长视频理解中的存储效率与细粒度保留提供了可复用的架构模式。
2. **实体级记忆索引**：利用 EMA 维护对象语义状态，结合相似度加权检索，可将此思路迁移至动态场景问答、视频摘要等需细粒度实体跟踪的任务。
3. **自适应语义分块替代固定切割**：基于多源相似度动态阈值的时间分块策略，可推广至任意长序列数据的语义边界检测与分段记忆构建。
4. **免训练即插即用增强**：作为独立预处理模块，无需修改骨干网络即可提升性能，适合快速适配不同 MLLM 与下游应用。

## 关键术语表
- **MEMO**：Multi-Level Entity-Aware Memory，一种面向流式视频理解的多级实体感知结构化内存框架。
- **Streaming Video Understanding**：流式视频理解，指视频帧按序连续到达时，模型需在无未来信息条件下进行实时推理。
- **Grounding DINO**：用于开放集目标检测的预训练模型，结合图像-文本对齐与 DINO 检测器，用于视频帧中的对象定位。
- **SAM (Segment Anything Model)**：通用图像分割模型，提供高精度对象掩码，此处用于提取个体对象的视觉区域。
- **EMA (Exponential Moving Average)**：指数移动平均，用于平滑并维护对象跨帧的语义状态表示，降低单帧噪声干扰。
- **Chunking**：分块，将无界视频流按语义边界切割为连贯片段，每个 chunk 对应独立的结构化内存单元。
- **Query-Specific Retrieval**：查询特定检索，根据自然语言查询，从结构化内存中平行匹配全局与实体索引，召回最相关视觉证据。
- **OVO-Bench**：Online Video Understanding Benchmark，包含历史检索、实时感知、主动响应等时间锚定任务的评测基准。
- **StreamingBench**：流式视频理解评测基准，评估模型在连续到达视频流上的实时理解能力。

## 可复现要素
- **数据集**：StreamingBench、OVO-Bench 均已公开。
- **代码/权重**：论文未提及代码与权重开源情况。
- **关键超参数**：
  - 分块权重 $(\lambda_s, \lambda_l, \lambda_g) = (0.35, 0.45, 0.20)$，检索权重 $(\lambda_{\text{global}}, \lambda_{\text{local}}) = (0.6, 0.4)$。
  - 滑动窗口大小 $n=30$，敏感度 $m=1.3$，阈值范围 $(\tau_{\min}, \tau_{\max}) = (0.1, 0.9)$。
  - 边界确认帧数 $N_{\text{confirm}}=1$，最小段长 $L_{\min}=4$，隐藏帧预算 $H_{\text{hid}}=5$。
  - EMA 系数 $\rho=0.3$，空间平衡系数 $\alpha=0.6$。
  - Grounding DINO 阈值：框 0.35、文本 0.25；跨帧关联 IoU 阈值 0.3。
  - 视觉输入预算 $N_{\text{LLM}}=8$（当前段与检索证据各最多 8 帧）。
