---
title: "MaLiang-Harness-A-Programmable-Path-to-Image-and-Video-Gener"
source: https://arxiv.org/pdf/2609.34309v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 09:56:40"
field: "可编程视觉生成"
keywords: ["programmatic image generation", "video generation", "MLLM agent", "P2V gap", "executable visual program", "MaLiang-Harness", "revision-aware verification"]
innovations: ["提出 P2V gap 与 PEG/TGP/REV 三模块统一框架，支持图像视频的程序化构建与修订感知验证", "构建 MaLiang-IBench/VBench 双基准并揭示通用 benchmark 得分与可视化程序生成能力的显著解耦", "在统一接口下集成多渲染后端（Canvas/SVG/Three.js/path-tracing）并评估 11 个 MLLM 的系统性差异"]
benchmarks: ["MaLiang-IBench", "MaLiang-VBench", "Artificial Analysis Intelligence Index"]
---

# 论文速读：MaLiang-Harness-A-Programmable-Path-to-Image-and-Video-Gener

## 一句话总结
本文提出 MaLiang-Harness，一个统一的可编程视觉生成框架，通过 MLLM 生成可执行代码驱动图像与视频合成，并以 **PEG（持久可执行生成状态）**、**TGP（可追溯生成过程）** 和 **REV（修订感知编辑与验证）** 三大机制弥合程序级正确性与视觉需求满足之间的 **Program-to-Visual (P2V) Gap**。在 MaLiang-IBench（50 题）与 MaLiang-VBench（13 题）上，GPT-6-Astra 实现 100% 生成成功率，图像/视频分别有 96.0% / 76.9% 满足全部质量阈值。

## 研究问题与动机
1. **P2V Gap 问题**：MLLM 生成的可视化程序可能执行正确，但输出图像/视频不满足用户指定的构图、外观或时序要求，程序级正确性与视觉语义满足之间存在明显鸿沟。
2. **隐式范式的控制缺失**：传统扩散/流匹配模型以隐式方式合成像素，缺乏对空间布局、时序结构、可编辑性和生成过程可追溯性的显式控制。
3. **通用能力不可迁移**：公开的综合 benchmark 得分无法可靠预测模型在可编程视觉生成任务上的表现，同分模型（如 GPT-5.6-Luna 与 GPT-6-Luna，AA 指数同为 37）实际生成质量相差悬殊（44% vs 88% 通过全部质量阈值）。
4. **缺乏统一的状态化反馈循环**：现有工作多关注一次性代码生成，缺少对生成历史、版本回滚、修订关联证据和强制验证的系统化管理机制。

## 核心贡献（创新点）
1. **定义 Program-to-Visual (P2V) Gap**：首次将程序正确性与视觉需求满足之间的不一致性形式化为一个独立问题，并论证状态化迭代框架的必要性。与 prior work 的区别在于不再假设"可运行即达标"，而是将视觉反馈作为闭环修正的核心信号。
2. **提出 MaLiang-Harness 统一框架**：整合 PEG、TGP、REV 三模块，使 MLLM 的规划、执行与视觉评估共享同一修订参照体系。与 VISPROG / Design2Code 等单步生成方法不同，本文提供跨渲染后端的持续修订与版本追溯接口。
3. **构建 MaLiang-IBench 与 MaLiang-VBench 评测基准**：前者含 50 个多样风格文生图任务，后者含 13 个文生视频任务，并同时报告成功率、多维质量阈值通过率与计算开销。这是首个面向"程序→视觉"翻译能力的系统性评测。
4. **揭示通用 benchmark 与可视化程序生成能力的解耦现象**：通过 11 个 MLLM 的横向对比证明 AA Intelligence Index 与 MaLiang-IBench 通过率的 Spearman ρ 仅 0.65，且存在极端反例，为后续研究提出"专用视觉程序生成 benchmark"的必要性的实证支撑。

## 方法详解
### 3.1 统一可编程视觉生成接口
- **PEG 状态**（公式 1）：$S_k = (P_k, A_k, Z_k, C_k, k)$，其中 $P_k$ 为可执行视觉程序及后端标识，$A_k$ 为关联素材，$Z_k$ 为时空结构描述，$C_k$ 为任务上下文（提示 $p$、输出规范 $\omega$、需求列表与当前计划），$k$ 为修订索引。
- **状态演化**（公式 2）：$S_{k+1} = \mathcal{E}(S_k, a_k)$ 提交编辑，$I_k(t) = \mathcal{R}_b(S_k, t; \omega)$ 由选定后端 $b$ 在内容时间点 $t$ 渲染。每次提交获得递增修订号，历史快照保留以支持回退。
- **多后端兼容**：Canvas、SVG、Scene2d、Three.js 及 path-tracing 等均可接入，通过统一协议进行状态检查、代码/素材编辑与需求审查。
- **图像/视频统一**：图像为固定时间点渲染；视频在相同程序表示下采样时序行为。

### 3.2 可追溯生成过程（TGP）
- 操作记录（公式 3）：$\tau_j = (o_j, x_j, y_j, k_j^-, k_j^+)$，$o_j$ 为执行的操作，$x_j/y_j$ 为输入与结果，$k_j^-$/$k_j^+$ 为前后 PEG 修订号。
- 渲染/检查操作不改写 $k$，从而区分"仅查看"与"真正提交"。
- 每次操作与修订级别的视觉证据绑定，便于定位空间/时序偏差来源。

### 3.3 修订感知编辑与验证（REV）
- **验证三元组**：$q_i = (k_i, E_i, v_i)$，$v_i \in \{\text{pass, fail, uncertain}\}$。图像用全帧或局部裁剪；视频在指定区间内取 ≥3 个有序时间戳样本。
- **交付条件**（公式 4）：
$$
\mathrm{Ready}(S_k) = \mathrm{ExportOK}(S_k) \wedge \mathrm{CheckpointOK}(k) \wedge \bigwedge_{h_i \in \mathcal{H}_k} [k_i = k \wedge E_i \neq \emptyset \wedge v_i = \text{pass}]
$$
- **ExportOK** 检查源码/素材存在性、哈希一致性、格式/尺寸/视频时序合规；**CheckpointOK** 要求当前修订的所有强制需求通过验证。
- 每次提交（含仅更新计划）均触发重新评估，防止先前通过的条目因后续修改而失效。
- MLLM 自我判定为结论性证据，但可由外部 judge（本文用 GPT-6-Sol）复核。

## 实验与结果
### 数据集与模型
- **MaLiang-IBench**：50 个文生图任务，覆盖线描/扁平艺术、2D 卡通、像素风、风格化 3D、混合媒体、照片级写实等；评估 11 个 MLLM（DeepSeek V4.1-Flash/V4-Pro、Kimi K2.6/K2.7-Code/K3、GPT 5.6-Luna/Terra/Sol、GPT 6-Luna/Sol/Astra）。
- **MaLiang-VBench**：13 个文生视频任务；评估 4 个模型（DeepSeek-V4.1-Flash、Kimi-K2.6、GPT-5.6-Sol、GPT-6-Astra）。
- **评估工具**：GPT-6-Sol 作为 judge，图像按 prompt 一致性、美学、构图三分（1–5 分）；视频另加运动连贯性四分。

### 主要定量结果
| 基准 | 最强模型 | 成功率 | 全部质量阈值通过 | 次强模型 | 成功率 | 全部通过 |
|---|---|---|---|---|---|---|
| MaLiang-IBench | GPT-6-Astra | **100%** (50/50) | **48/50 (96.0%)** | GPT-5.6-Sol | 96% (48/50) | 43/50 (86.0%) |
| MaLiang-VBench | GPT-6-Astra | **100%** (13/13) | **10/13 (76.9%)** | GPT-5.6-Sol | 53.8% (7/13) | 5/13 (38.5%) |

- **DeepSeek / Kimi**：图像成功率仅 12%–40%，视频 DeepSeek 为 0%，Kimi-K2.6 为 23.1%；失败多为 token 预算耗尽（语义活锁/推理停滞）。
- **成本对比**：GPT-5.6-Sol 每图 3.28 min / 通过图；Astra 3.70 min / 通过图；视频 Time/Success：GPT-5.6-Sol 9.27 min，Astra 10.00 min。
- **质量分布规律**：GPT-6 系列所有成功图像均通过美学与构图阈值，剩余失败集中于 prompt 一致性；视频运动连贯性是最严苛维度（Astra 13 通过美学，仅 10 通过运动）。

### 关键发现
- AA Intelligence Index 与 MaLiang-IBench 全阈值通过率 Spearman ρ = 0.65；GPT-5.6-Luna 与 GPT-6-Luna 同分 37 却通过率 44% vs 88%；Kimi-K3 指数 44 但通过率仅 18%。
- 渲染后端可扩展：path-tracing 后端支持物理材质与光照传输，可在统一 harness 下迭代（Fig. 7）；但仅靠细化笔触无法达到照片级真实（Fig. 8）。

## 相关工作脉络
1. **VISPROG**（Gupta & Kembhavi, 2023）：模块化程序用于视觉推理与图像编辑，暴露中间结果为可审计依据；本文扩展至图像+视频统一框架并引入多版本修订与强制验证。
2. **Design2Code**（Si et al., 2025）：网页截图到可渲染实现的自动转换；本文聚焦创意视觉内容生成而非前端工程还原。
3. **BlenderAlchemy**（Huang et al., 2024）：结合 VLM 与状态评估器在 3D 场景搜索编辑；本文在 2D/视频 Canvas/SVG 等后端建立跨渲染的统一协议与修订历史。
4. **OmniHarness**（Xu et al., 2026b）：从已验证执行中学习可复用符号策略；本文侧重 MLLM 单轮迭代中的状态保持与修订追溯，不依赖策略学习。
5. **Diffusion / Flow Matching 系列**（Ho 2020; Rombach 2022; Lipman 2022）：隐式像素合成；本文与之互补，探索显式程序化路径，强调可编辑性与过程可追溯。
6. **ControlNet**（Zhang et al., 2023）：空间条件引导扩散；本文在"无扩散"路线上通过代码直接控制几何与时间，不需训练额外适配器。

## 局限性与未来方向
1. **超写实渲染仍受限**：仅通过更细笔触或现有 Canvas 后端难以恢复形状–阴影–纹理的协调；需更深入的 path-tracing / 物理材质集成。
2. **修订停滞模式**：推理停滞、语义活锁、无限 agentic 循环三类失败未被框架本身保证消除；目前仅靠 token / 调用 / 时间预算硬性截断。
3. **视频运动连贯性评估粗糙**：judge 仅依据 12 帧有序样本判断，未保证帧间连续性，可能低估/高估实际播放效果。
4. **成本统计口径不一致**：不同模型在 Time/Task、Calls、Token 等指标上使用不同归一化基数（全 50 任务 vs 成功子集），限制跨模型效率直接比较。
5. **评估偏向 GPT-6-Sol judge 主观偏好**：judge 为同一模型族自评判，可能存在系统性偏差。

未来方向包括： richer rendering backends（物理渲染、体积光照）、progress-aware 修订策略（感知改进幅度以提早终止）、更鲁棒的运动连贯性度量、以及与 diffusion 后处理相混合的 hybrid 管线。

## 研究启发与可借鉴点
1. **PEG/TGP/REV 三件套可作为通用"程序→产物"迭代的参考架构**，适用于 3D 场景构建、UI/UX 自动生成、数据可视化叙事等需要版本追溯与强制验收的场景。
2. **修订感知验证的交付条件公式（公式 4）可直接移植**：任何需要"多约束联合通过才能交付"的 Agent 系统均可借鉴 ExportOK ∧ CheckpointOK ∧ 全需求 pass 的模式。
3. **通用 benchmark 与专项能力的解耦分析具有方法论价值**：在团队内部引入同构对比（同 AA 分数不同专项表现）可更准确地识别模型的能力盲区，避免仅凭综合排行榜选型。
4. **多后端统一接口的可扩展设计**：现有 harness 通过 protocol 层屏蔽 Canvas/SVG/Three.js/path-tracing 差异，这一抽象便于后续接入 NeRF、Gaussian Splatting、点云等新一代表示。
5. **操作级追踪（TGP）支持因果归因**：将每次代码变更与渲染前后对比记录，可为"哪一行代码导致哪项需求失败"的诊断提供自动化线索，降低人工调试成本。

## 关键术语表
**Program-to-Visual (P2V) Gap**：程序能正确执行但输出图像/视频未满足用户视觉需求的差距，是本文刻画的核心问题。
**Persistent Executable Generation (PEG) State**：持久化的可执行生成状态 $S_k = (P_k, A_k, Z_k, C_k, k)$，在迭代中保持程序、素材与时空定义不丢失。
**Traceable Generation Process (TGP)**：记录每次操作的输入/输出与前后修订号，使构建历史可追溯并与视觉证据绑定。
**Revision-aware Editing and Verification (REV)**：在每次修订提交后强制重审当前版本的各需求项，并将审查证据锚定到对应修订。
**MaLiang-IBench**：包含 50 个多样性文生图任务的评价基准，衡量成功率、三维质量阈值与计算开销。
**MaLiang-VBench**：包含 13 个文生视频任务的评价基准，额外加入运动连贯性评估。
**ExportOK / CheckpointOK**：REV 中两条交付前置条件，前者校验文件/素材/格式合规，后者校验当前修订所有强制需求均已通过。
**Semantic Livelock**：Agent 反复编辑但从未推进任何需求通过的状态，属本文识别的三大失败模式之一。

## 可复现要素
- **代码**：开源，https://github.com/gulucaptain/MaLiang-Harness。
- **数据集**：MaLiang-IBench（50 prompts）、MaLiang-VBench（13 prompts）；附录 6.4/6.5 列出全部提示词。
- **评估模型**：11 个 MLLM（DeepSeek V4.1-Flash/V4-Pro、Kimi K2.6/K2.7-Code/K3、GPT-5.6-Luna/Terra/Sol、GPT-6-Luna/Sol/Astra）。
- **Judge 模型**：GPT-6-Sol。
- **渲染后端**：Canvas、SVG、Scene2d、Three.js、path-tracing（内置）。
- **关键超参**：图像分辨率与格式在 $\omega$ 中指定（如 1536×864、2048×1152）；seed 控制程序内 PRNG；视频采样 12 帧；token 预算因模型配置而异（DeepSeek-V4.1-Flash 使用 64K 输出上限）。
- **注意事项**：不同模型的 cost 统计口径不完全一致，复现时应统一归一化基准以公平比较。
