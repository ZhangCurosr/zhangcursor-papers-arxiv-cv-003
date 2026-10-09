---
title: "OneSearch-VL-Unified-Multimodal-Deep-Research-Agent-for-Imag"
source: https://arxiv.org/pdf/2610.12419v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 10:09:22"
field: "多模态智能体与深度研究"
keywords: ["multimodal deep research", "vision-language agent", "VGEG", "process reward", "GRPO", "multi-image reasoning", "video grounding"]
innovations: ["提出VGEG统一表征视觉锚点-来源事实-答案操作的端到端证据链", "基于VGEG导出EVGR双维过程奖励（证据溯源+视觉定位）", "联合训练单图/多图/视频三种输入的统一深度研究Agent"]
benchmarks: ["OneSearch-MI-Bench", "OneSearch-Video-Bench", "VideoDR", "SimpleVQA", "VDR", "MMSearch", "LiveVQA", "BrowseComp-VL", "FVQA", "InfoSeek"]
---

# 论文速读：OneSearch-VL: Unified Multimodal Deep Research Agent for Image and Video

## 一句话总结
论文提出 OneSearch-VL，一个基于**视觉化证据图（VGEG）**的统一多模态深度研究智能体，首次将单图、多图和视频三种视觉输入的研究流程统一训练，并通过 EVGR 奖励将证据溯源与视觉定位联合纳入强化学习，在新构建的基准上比 Qwen3-VL-8B 提升 20.2/17.6/27.0 个百分点。

---

## 研究问题与动机
1. **现有方法割裂**：单图检索 Agent（如 OpenSearch-VL）与视频深度研究 Agent（如 VideoDR）各自独立，缺乏统一框架。
2. **多模态依赖链路丢失**：已有证据图（perception–knowledge chains、citation rubrics）无法同时保留"视觉锚点→实体关系→来源支撑事实→答案生成操作"的端到端溯源链；视频管线同样仅把关键帧视为检索种子，未显式保留证据依赖。
3. **多图研究被忽视**：绝大多数工作只关注单图或视频，多图组合检索（跨图像聚合信息、比对、算术运算）尚未被系统研究。
4. **评估粒度不足**：现有视频/多图基准仅报告总准确率，缺乏按研究操作（entity lookup、relation tracing、conditional filtering、fact composition）划分的细粒度分析。

---

## 核心贡献（创新点）
1. **VGEG 中心化数据引擎**：提出 Visually Grounded Evidence Graph 作为统一的任务级参考结构，将视觉锚点、来源支撑事实与答案生成操作绑定，支持多图像/视频的 QA 构造、验证与专家轨迹筛选——现有方法缺乏这种端到端溯源的结构化记录。
2. **统一智能体 + 联合训练**：OneSearch-VL 在同一策略下联合学习单图、多图、视频三种输入的搜索决策，证明跨视觉类型的数据互补而非互斥——区别于前作仅训练单一视觉模态的设定。
3. **EVGR 双维奖励**：从 VGEG 标注导出证据溯源（traceability）与视觉定位（grounding）两个正交维度的过程奖励，作为 GRPO 优化目标——已有工作仅用最终答案或查询质量作监督。
4. **操作导向型基准**：构造 OneSearch-MI-Bench（301 题）与 OneSearch-Video-Bench（307 题），按六种研究操作分类并给出难度划分，使评估聚焦"如何检索与组合证据"而非"输入是什么"——现有基准多以视频内容或来源类别划分。

---

## 方法详解

### 1. 统一问题形式化
给定视觉输入 $X \in \{I,\, \mathcal{T}=\{I_j\}_{j=1}^K,\, V\}$ 与问题 $q$，策略 $\pi_\theta$ 在每步 $t$ 基于交互历史 $h_t$ 生成动作 $a_t=(z_t,c_t)$，其中 $z_t$ 为推理链、$c_t$ 为工具调用或最终答案，轨迹似然 $\pi_\theta(\tau|X,q)=\prod_{t=1}^T\pi_\theta(a_t|h_t)$。三种输入共享同一 action–observation 协议，仅工具集不同。

### 2. 工具集
- **视觉工具 $\mathcal{T}_{vis}$**：CROP、OCR、PerspectiveCorrect、SuperResolution、SHARPEN（单图/多图/视频均使用）。
- **检索工具 $\mathcal{T}_{ret}$**：ImageSearch、TextSearch（三类输入共用）。
- **时序工具 $\mathcal{T}_{temp}$**：SELECTTIMESPAN、SELECTFRAME（仅视频）。

### 3. VGEG 数据引擎（五阶段）
1. **视觉源筛选**：从 2.5M YouTube 视频中经上传日期、元数据评分、类别平衡、Snippet LLM 过滤，保留 70k 候选。
2. **密集视觉锚点发现**：2 FPS 采样→Clip 描述（Seed 2.0 Pro）→关键帧选择→跨 Clip 去重（相似度阈值 0.9/0.8）→事件聚合→关键帧内对象定位，形成 `video→event→keyframe→localized object` 的层级树。
3. **Web 证据图构造**：对象路由至 ImageSearch/OCR+TextSearch，生成候选实体池；通过 Expand Entity Pipeline 迭代扩展关系链，每条事实 $f=(u,r,z,p,\xi)$ 记录头实体、关系、尾实体/属性、来源 URL 与原文引证。
4. **VGEG 任务构造**：对候选图 $\mathcal{G}_X$ 施加任务条件投影 $\Gamma_i=\Phi_i(\mathcal{G}_X)=(\mathcal{A}_i,\mathcal{F}_i,\mathcal{O}_i,\mathcal{R}_i)$，其中 $\mathcal{A}_i$ 为视觉锚点及实体、$\mathcal{F}_i$ 为来源支撑事实、$\mathcal{O}_i$ 为答案生成操作、$\mathcal{R}_i$ 为依赖边。经 QA Verifier 校验、Visual Entity Fuzzing Rewrite 后实例化为多图/视频/单图任务。
5. **专家轨迹合成**：Seed 2.0 Pro 在真实工具环境中求解，经答案正确性（GPT-4o judge）与过程质量双重筛选，构成 OneSearch-VL-SFT-110K（36k 单图 + 37k 多图 + 35k 视频）。

### 4. 训练流程
- **SFT**：屏蔽工具观测 token 的 loss（$M_{t,k}^{pol}=1$ 仅对模型生成 token），冻结视觉编码器与 MM 投影，全参微调 8 epoch，lr=$2\times10^{-5}$，cosine schedule，warmup 0.1，64×H800 约 4 天。
- **RL（GRPO）**：从 SFT 模型起在线 rollout，OneSearch-VL-RL-10K（3.7k 单图 + 2.8k 多图 + 3.6k 视频）；每组 8 条轨迹取相对优势，clip ratio=0.28，KL coef=$1\times10^{-3}$。
- **EVGR 双维奖励**：
  - $r_{trace}$：依据 VGEG 标注 rubric，检查工具观测是否建立答案核心实体与事实跳数，且推理无 unsupported substitution。
  - $r_{ground}$：检查是否正确定位到相关可见实体/区域/帧并用于后续检索。
  - 综合：$R(\tau)=R_{fmt}(\tau)\cdot(\lambda_{acc}R_{acc}+\lambda_{query}R_{query}+\lambda_{EVGR}R_{EVGR})$，权重分别为 0.6/0.2/0.2。

### 5. 基准设计
- **OneSearch-MI-Bench**：301 题，多图（2–8 张，均 3.05 张），每题为至少 2 张图的自包含问题。
- **OneSearch-Video-Bench**：307 题，保留时序结构，锚点绑定 event/keyframe/region。
- 六类操作：Single-anchor lookup、Multi-hop retrieval、Knowledge-conditioned count、Multi-anchor join、Multi-anchor arithmetic、Multi-anchor comparison；难度分 $d_i=10w(s_i)+h_i+0.1|\mathcal{F}_i|$，切分易/中/难三档（≈100/100/101 题）。

---

## 实验与结果

| 数据集 | 设置 | 分数 |
|--------|------|------|
| **OneSearch-MI-Bench** | Qwen3-VL-8B Agent: 35.6 → OneSearch-VL-8B: **55.8** | **+20.2 pp** |
| **OneSearch-Video-Bench** | Qwen3-VL-8B Agent: 17.9 → OneSearch-VL-8B: **35.5** | **+17.6 pp** |
| **VideoDR** | Qwen3-VL-8B Agent: 30.0 → OneSearch-VL-8B: **57.0** | **+27.0 pp** |
| **七单图基准平均** | OpenSearch-VL-8B: 56.6 → OneSearch-VL-8B: **58.3** | **+1.7 pp** |

**关键消融**：
- SFT 数据混合：I+M+V 联合训练达 6-benchmark 平均 55.8，优于任何单一类型（51.7–55.1）。
- RL 奖励组件：Acc 仅 56.3 → +Query 57.3 → +Trace 59.0 → +Ground 59.3 → Full EVGR **61.1**（较 Acc+Query 提升 3.8 pp）。
- 操作层面提升最显著：多图像知识条件计数 +29.8、多锚点算术 +26.3、多锚点对比 +22.5；视频多跳检索 +25.5、多锚点算术 +20.0、多锚点连接 +19.2。

---

## 相关工作脉络
1. **OpenSearch-VL**（Chen et al., 2026）：单图深度研究 Agent，本文在其基础上扩展至多图与视频，并用 VGEG 替代其 perception–knowledge chain。
2. **VideoDR**（Liu et al., 2026）：视频深度研究基准，本文沿用但提出操作导向的 OneSearch-Video-Bench 与之形成对照。
3. **Vision-DeepResearch**（Huang et al., 2026）、**VideoSearcher**（Gao et al., 2026）：视频检索管线，仅把关键帧作为检索种子；本文显式保留 frame↔object↔entity↔source 的全链路依赖。
4. **SearchEyes**（Jiao et al., 2026）、**STAMP**（Xu et al., 2026）：过程监督/引用感知信用分配，本文以 VGEG 作为任务级结构化引用，更细粒度对齐事实跳与视觉位置。
5. **OneThinker**（Feng et al., 2025b）、**LLaVA-ST**（Li et al., 2025a）：统一多模态理解/推理模型，本文聚焦统一**深度研究**而非通用视觉推理。
6. **MMSearch-R1**（Wu et al., 2025）、**DeepMMSearch-R1**（Narayan et al., 2025）：图片搜索 RL 训练，本文在统一框架内同时处理多图聚合与视频时序定位。

---

## 局限性与未来方向
1. **外部搜索依赖**：TextSearch/ImageSearch 受网页更新影响，证据可得性与可复现性波动；需引入检索快照与工具失败鲁棒性评测。
2. **自动标注误差**：VGEG 构造与 EVGR judge 均依赖模型自动化，存在事实偏差与判断偏见；需人工校准与开放多模态过程裁判。
3. **多轮交互成本**：复杂研究轨迹产生大量推理与工具调用开销；未来需自适应预算与成本感知训练。
4. **操作泛化边界**：目前仅覆盖六种预定义操作，复杂跨域长链推理仍未充分探索。

---

## 研究启发与可借鉴点
1. **VGEG 结构化溯源范式**可迁移至任意"视觉定位→外部检索→事实组合"的智能体管线，作为数据构造、过程监督、细粒度评估的统一锚点。
2. **跨模态联合训练**（单图+多图+视频）不牺牲任何单模态表现，且提供互补监督；启示在多视觉输入任务中应避免孤立训练。
3. **EVGR 双维过程奖励设计**（traceability + grounding）有效弥合"答案对但证据错"的假阳性，对需要强可追溯性的 Agent 系统（医疗、法律、科研）有直接参考价值。
4. **操作分类基准**思路：将样本按所需推理操作而非输入类型组织，可在其他 Agent 评测中推广为"操作粒度画像"工具。
5. **专家轨迹筛选的两阶段 judge**（答案正确性 + 过程质量）可直接复用到其他多轮工具使用数据集构建。

---

## 关键术语表
- **VGEG（Visually Grounded Evidence Graph）**：任务级结构化参考，把视觉锚点、来源支撑事实与答案生成操作及其依赖关系绑定在一起。
- **EVGR（Evidence-aware Visual-Grounded Rubric reward）**：从 VGEG 导出的双维过程奖励，分别评估证据溯源完整性与视觉定位准确性。
- **GRPO（Group Relative Policy Optimization）**：以同问题多条轨迹的相对优势进行策略优化的 RL 算法，本文用于 Agent 在线训练。
- **OneSearch-MI-Bench / OneSearch-Video-Bench**：按研究操作分类的多图（301 题）与视频（307 题）深度研究细粒度基准。
- **Research Operation（研究操作）**：六类答案生成操作——单锚点查找、多跳检索、知识条件计数、多锚点连接、多锚点算术、多锚点对比。
- **Dense Visual Anchor**：经视频切片→关键帧→对象定位得到的层级化视觉锚点树（video→event→keyframe→object）。
- **Evidence Traceability vs. Visual Grounding**：前者验证工具观测是否支撑所有事实跳，后者验证是否正确定位到相关可见实体/区域/帧。
- **Fatal-aware Masking**：工具执行失败时只保留失败前合法前缀参与策略优化，避免错误后缀污染训练。

---

## 可复现要素
- **代码与模型**：GitHub https://github.com/appletea233/OneSearch-VL；HuggingFace https://huggingface.co/OneSearch-VL（已公开）。
- **数据集**：OneSearch-VL-SFT-110K、OneSearch-VL-RL-10K、OneSearch-MI-Bench、OneSearch-Video-Bench；论文声明源码与权重开源，未明确说明数据集是否完全独立公开，建议查阅 GitHub。
- **关键超参**：SFT lr=2×10⁻³，epoch=8，warmup=0.1，batch=128（64 GPU×2 accum），cutoff=32768 tokens；RL n=8 rollout/prompt，clip ratio=0.28，γ=0.95，KL coef=1×10⁻³，权重 λ_acc=0.6, λ_query=0.2, λ_EVGR=0.2。
- **训练硬件**：SFT 64×H800 约 4 天；RL 32×H800 约 4 天。
- **基础模型**：Qwen3-VL-8B-Instruct（视觉编码器与 MM 投影冻结）。
- **专家模型**：Seed 2.0 Pro（视觉锚点发现、VGEG 构造、轨迹筛选）。

---
