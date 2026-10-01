---
title: "Joint-and-Cross-Modal-Video-Audio-Generation-and-Editing-A-U"
source: https://arxiv.org/pdf/2609.34381v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-10-01 04:10:45"
field: "多模态生成"
keywords: ["音视频生成", "跨模态对齐", "视频编辑", "Foley", "多模态生成", "综述", "统一框架"]
innovations: ["提出联合/跨模态/编辑三任务统一公式化框架", "构建五轴设计分类法（生成策略/音频表示/视频表示/对齐机制/预训练复用）"]
benchmarks: ["CAVP", "VATEX", "AudioSet"]
---

# 论文速读：Joint-and-Cross-Modal-Video-Audio-Generation-and-Editing-A-U

## 一句话总结
本论文系统综述了视频-音频联合生成、跨模态生成与联合编辑三大任务的研究进展（截至 2026 年 7 月），提出了统一的三任务公式化框架与五轴设计分类法，并全面评估了各领域方法的代码/权重开源情况。

## 研究问题与动机
- 视频与音频在感知上构成整体，但现有生成模型大多独立处理单模态；简单拼接两模态输出会导致时空/语义不一致的音视频流。
- 现有研究缺乏统一的公式化视角将**联合生成**、**跨模态生成**、**联合编辑**三类任务映射到同一音频-视觉对分布上，难以进行系统比较。
- 已有综述（如 Qin et al., 2026）未覆盖编辑任务，也未以**跨模态一致性**（时间与语义对齐）作为核心组织框架。
- 领域内方法设计差异巨大（不同音频/视频表示、对齐机制、预训练策略），亟需一套分类学来指导研究者定位自身工作。

## 核心贡献（创新点）
1. **统一三任务公式化**：将联合生成 $p_\theta(v,a|c)$、跨模态生成 $p_\theta(a|v)$ / $p_\theta(v|a)$、联合编辑 $p_\theta(v',a'|v,a,e)$ 统一为音视频对分布上的条件建模问题。
2. **五轴设计分类法**：从生成策略、音频表示、视频表示、对齐强制机制、预训练复用五个正交维度刻画所有方法，填补领域系统性分类空白。
3. **全覆盖综述与统计**：梳理 29 种联合生成、20+ 种跨模态生成、7 种编辑方法，并首次公开盘点截至 2026 年 7 月的代码/权重可用性（Table 14–15）。
4. **开放问题提炼**：明确长程一致性、细粒度控制、物理合理性、评估体系四大挑战，指出核心难题为"在保持单模态质量的同时提升跨模态对齐"。

## 方法详解
### 三任务统一公式
- **联合生成（Problem 1）**：$p_\theta(v, a \mid c)$，同时输出视频与音频，条件 $c$ 可为文本/图像等。
- **跨模态生成（Problem 2）**：$p_\theta(a \mid v)$（V2A/Foley）或 $p_\theta(v \mid a)$（A2V），由一模态生成另一模态。
- **联合编辑（Problem 3）**：$p_\theta(v', a' \mid v, a, e)$，编辑指令 $e$ 需在一模态生效的同时传播到另一模态，未编辑区域保持不变。

### 五轴设计分类法（Table 1）
1. **生成策略**：Single-tower / Dual-tower / Cascaded / Unified-token / Guidance-based
2. **音频表示**：Continuous latent（主流，28/29 方法采用）/ Discrete tokens / Mel-spectrogram / Waveform（直接生成波形的方法仍空白）
3. **视频表示**：3D-VAE（主流，19 种方法）/ 2D-VAE + Temporal / Discrete tokens / Pixel
4. **对齐强制机制**：Cross-attention（19 种）/ Shared positional encoding / Discriminator / Classifier guidance / Explicit prior
5. **预训练复用**：From-scratch / Single pretrained / Dual pretrained

### 对齐定义（Sec.8）
- **定义 1（对应对齐）**：$(v,a)$ 对齐当且仅当对比训练编码器 $f_v:\mathcal{V}\to\mathbb{R}^{d_e}$、$f_a:\mathcal{A}\to\mathbb{R}^{d_e}$ 的嵌入在余弦相似度下接近。
- **定义 3（时间同步）**：容忍度 $\delta$ 下，每帧 $v^{(t)}$ 对应的音频事件须落在 $[t-\delta, t+\delta]$。
- **四类对齐**：时间同步（Sec.8.1）、语义对齐（Sec.8.2）、口型同步（Sec.8.3）等。

### 28 类编辑类型（9 大类）
同步 / 联合内容 / 跨模态迁移 / 身份&表演 / 场景&环境 / 叙事 / 生成式 / 质量&修复 / 跨切面。

## 实验与结果
- 本文无独立训练实验，核心结果为**统计性综述结论**：
  - **29 种联合生成方法**中 28/29 采用连续隐空间音频表示；19 种使用 3D-VAE 视频表示。
  - **Unified-token** 与 **Discrete audio tokens** 两类在联合生成中均无人采用（研究空白）。
  - **直接生成波形（Waveform）** 的联合生成方法也为空白。
  - 代码可用性分化明显：联合生成方向开源方法约 11 种（MM-Diffusion、JavisDiT、LTX-2、MOVA 等），跨模态方向约 10 种（Diff-Foley、MMAudio、FoleyCrafter 等），编辑方向 2 种（AvED、JUST-DUB-IT）。
  - 商业/闭源趋势显著：**Wan 2.5**（阿里，仅 API）、**Movie Gen**（Meta，闭源）、**Google V2A**（DeepMind，闭源研究系统）。

## 相关工作脉络
1. **Qin et al. (2026)**：同期综述，但未覆盖编辑任务，也未采用跨模态一致性作为组织框架——本文填补此空白。
2. **ImageBind（Girdhar et al., 2023）**：对比学习音频-视觉编码器，被多个方法复用为对齐评分器，是跨模态对齐的基础设施。
3. **MM-Diffusion（Ruan et al., 2023）**：早期耦合 UNet 联合生成代表，采用 Sequential dual UNet + SC 训练。
4. **Movie Gen（Polyak et al., 2024）**：Cascaded T2V→V2A 工业级方案，体现商业闭源趋势，未开源。
5. **MMAudio（Cheng et al., 2025）**：跨模态联合训练的代表性工作，开源，支持多模态条件。
6. **AV-Edit（Guo et al., 2026a）** / **JUST-DUB-IT（Chen et al., 2026）**：编辑任务代表，前者基于 MM-DiT，后者采用 LoRA 微调。

## 局限性与未来方向
- 论文自述的开放问题：**长程一致性**（long-horizon coherence）、**细粒度控制**、**物理合理性**、**评估体系**。
- 核心挑战仍未解决：在保持单模态质量的同时提升跨模态对齐。
- **Unified-token 音频表示**、**离散音频 token**、**波形直接生成**三类方法在联合生成中完全空白，存在方法论创新空间。
- 部分方法（如 SelVA、Mel-QCD）训练代码 pending，可复现性有待跟进。
- 商业模型（Wan 2.5、Movie Gen）仅 API 可用，限制了社区对标与消融研究。

## 研究启发与可借鉴点
1. **统一公式框架可直接迁移**：将不同任务映射到同一分布建模框架的思路，可推广至其他多模态生成场景（如视频-文本-音频三角生成）。
2. **五轴分类法作为方法定位工具**：后续工作可用该分类快速定位自身在生成策略/表示/对齐机制上的创新贡献。
3. **跨模态对齐定义可作为评测基准**：基于余弦相似度（Definition 1）和时间容忍度 $\delta$（Definition 3）的量化指标可直接用于新方法的评测。
4. **可结合团队方向挖掘空白**：Unified-token 与离散音频表示在联合生成中的空白，是潜在的创新切入点。
5. **代码可用性盘点方法论**：Table 14–15 的开源状态追踪方式，可作为其他领域综述的复现模板。

## 关键术语表
- **联合生成（Joint Generation）**：$p_\theta(v,a|c)$，同时从条件生成视频和音频。
- **跨模态生成（Cross-modal Generation）**：$p_\theta(a|v)$ 或 $p_\theta(v|a)$，由一模态生成另一模态（如 V2A/Foley）。
- **联合编辑（Joint Editing）**：$p_\theta(v',a'|v,a,e)$，编辑指令 $e$ 需同时在一模态生效并传播到另一模态。
- **五轴设计分类法**：从生成策略、音频表示、视频表示、对齐强制机制、预训练复用五维度刻画方法的设计空间。
- **时间同步（Temporal Synchronization）**：容忍度 $\delta$ 下视频帧与音频事件的时间对齐程度。
- **语义对齐（Semantic Alignment）**：弱全局属性层面的音视频内容一致性。
- **口型同步（Lip-sync）**：语音/歌词与面部嘴部运动的精确对齐，属于时间同步的子类。
- **Foley**：影视后期制作中模拟环境音效的跨模态音频生成任务。

## 可复现要素
- **数据集**：论文未统一声明单一数据集，引用各方法自有数据集（如 CAVP 等），未提及标准评测基准。
- **代码/权重**：部分方法已开源（见 Table 15），包括 MM-Diffusion、JavisDiT/JavisDiT++、LTX-2、MOVA、Diff-Foley、MMAudio、FoleyCrafter、AvED、JUST-DUB-IT 等；若干方法仅 API/闭源（Movie Gen、Wan 2.5、Google V2A）。
- **关键超参**：论文未给出统一超参，各方法独立设定。
