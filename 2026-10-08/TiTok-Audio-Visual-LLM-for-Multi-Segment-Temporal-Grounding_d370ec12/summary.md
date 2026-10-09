---
title: "TiTok-Audio-Visual-LLM-for-Multi-Segment-Temporal-Grounding"
source: https://arxiv.org/pdf/2610.09408v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 03:17:50"
field: "多模态时序定位"
keywords: ["Audio-Visual Multi-Segment Temporal Grounding", "Time Token Interleaving", "GDPO", "Count Calibration", "Reinforcement Learning for VTG", "AV-LLM"]
innovations: ["提出 TTI（时间 Token 交织）方法，将专用时间 Token 同时用于 AV-LLM 输入侧与输出侧，实现端到端时间对齐", "设计五维解耦多段导向奖励（global/local/count/precision/format）并以 GDPO 优化，显式纠正计数校准偏差", "建立 UnAV-100 上 AV-MSG 评测协议并引入 CountF1 指标，量化事件计数校准能力"]
benchmarks: ["UnAV-100", "Charades-STA"]
---

# 论文速读：TiTok-Audio-Visual-LLM-for-Multi-Segment-Temporal-Grounding

## 一句话总结
TiTok 是一款针对未裁剪视频 **音频-视觉多段时序定位（AV-MSG）** 的大语言模型，通过引入 **时间 Token 交织（TTI）** 与基于 **GDPO** 的多段导向解耦奖励 RL 训练框架，有效解决了现有 AV-LLM 在重复事件计数上的"计数校准偏差"（count miscalibration）问题，在 UnAV-100 上达到 SOTA（65.7 mIoU、0.58 CountF1）。

## 研究问题与动机
1. **现有 VTG/AV-MSG 方法存在计数校准偏差**：多数视觉中心模型忽略音频线索，而 AV-LLM 往往将多次出现的事件坍缩为单一时段，或过度分割，现有目标函数仅优化时序重叠（IoU），未显式惩罚事件基数（cardinality）错误。
2. **AV-LLM 缺乏显式时间表征**：既有 AV-LLM 通常不内含专用时间 Token，输出侧多依赖文本数字或百分比，无法将输入侧时间感知与输出侧时间预测对齐。
3. **多段场景下多奖励联合优化失效**：不同维度的奖励（全局、局部、计数等）若求和后统一归一化（如 GRPO），会抹平细粒度定位信号，导致训练信号模糊或早期训练失败。
4. **缺乏面向 AV-MSG 的评测基准**：现有基准缺少音频感知的多段标注，且传统 IoU 类指标无法直接量化"计数偏差"，作者由此提出新的 UnAV-100 评测协议与 CountF1 指标。

## 核心贡献（创新点）
1. **提出 TiTok 框架与 TTI 方法**：将专用时间 Token 以交织形式注入音视频流，使输入侧与输出侧共享同一时间表达体系；与既有工作的本质区别在于首次在 AV-LLM 中将绝对时间 Token 同时用于输入和输出，实现端到端时间对齐。
2. **设计五维解耦多段导向奖励（global/local/count/precision/format）**并采用 GDPO 优化：各奖励独立归一化后再加权合并，避免 GRPO 对异构奖励求和后再归一化所导致的信号压制；与既有 VTG RL 工作（如 Time-R1、TempR1）的本质区别在于显式建模了事件计数精度与格式约束，而非仅依赖格式+IoU 二元奖励。
3. **建立 UnAV-100 上的 AV-MSG 评测协议并引入 CountF1 指标**：CountF1 综合衡量欠分割（USA）与过分割（OSA）两类计数校准偏差，其基线值仅依赖数据集标注，可复用于其他基准；这是对现有 IoU 类指标的实质性扩展。

## 方法详解
**整体架构**：以 video-SALMONN 2 plus（7B）为骨干，分两阶段训练。

**Stage 1 — 冷启动 SFT**：仅在 UnAV-100 训练集上做短时监督微调，学习特殊时间 Token 的读写能力。冻结视觉/音频编码器，LoRA 作用于 LLM（r=16, α=16），更新视听对齐器与时间 Token 嵌入/LM head 行。

**Stage 2 — GDPO 强化学习**：
- **TTI（Time Token Interleaving）**：沿时间轴按序交织音视频 Token，每个时间戳编码为 4 个数字 Token + 1 个小数点 Token（如 16.7s → `<t0><t1><t6><tdot><t7>`），插入对应音视频块之前，输入侧标记绝对时间，输出侧生成相同格式的时段边界。
- **五维解耦奖励**（均值化为 [0,1]，权重 w=1）：
  - **全局奖励 $r_{\text{global}}$**：预测段取并集后计算 Prec/Rec，以 F1 为奖励（公式 4）。
  - **局部奖励 $r_{\text{local}}$**：按输出顺序将预测段与最高 GIoU 的未匹配 GT 段配对，以 $\frac{1}{\max(N_{\text{gt}}, N_{\text{pred}})}\sum \text{GIoU}$ 计算（公式 6），未匹配段不贡献分子但计入分母，从而惩罚计数偏差。
  - **计数奖励 $r_{\text{count}}$**：$1 - \frac{|N_{\text{pred}} - N_{\text{gt}}|}{\max(N_{\text{pred}}, N_{\text{gt}})}$；当 $N_{\text{gt}}\geq 2$ 且 $N_{\text{pred}}=1$ 时强制置 0，严惩坍缩。
  - **精度奖励 $r_{\text{prec}}$**：独立 Precision（公式 4 中 Prec），与 $r_{\text{global}}$ 互补。
  - **格式奖励 $r_{\text{format}}$**：严格匹配 `<From tXtXtXtdottX to tXtXtXtdottX>` 格式即得 1，否则 0。
- **GDPO 优势估计**：先对各奖励在组内归一化（公式 1），再加权求和后在批次内归一化（公式 2），避免异构奖励信号相互压制。
- **训练目标**：基于 DAPO 的 Clip-Higher（$\epsilon_{\text{low}}=0.2, \epsilon_{\text{high}}=0.28$）与去除 KL 散度惩罚（公式 3），允许策略更自由地探索。

**RL 阶段配置**：冻结除 LLM LoRA（r=32, α=64）外的所有模块，2000 步，batch size=4，lr=1e-5，每 prompt 采样 $N_{\text{roll}}=8$ 个回复（max output=256），$\mu=2$ 次策略更新。

## 实验与结果
**数据集**：
- **UnAV-100**（主要评测集）：100 类密集标注的 AV 事件数据集，按类别重组为多段 GT。
- **Charades-STA**（补充评测）：纯视觉单段时序定位数据集，用于检验单段场景是否过分割。

**训练数据**：10,358 样本（UnAV-100 7,207，其中多段 4,055；Charades-STA 3,151），单段:多段比 ≈ 6:4。

**主要结果（UnAV-100）**：
| 模型 | mIoU | F1@0.3 | F1@0.5 | F1@0.7 | CountF1 |
|------|------|--------|--------|--------|---------|
| ChronusOmni (FT) | 63.4 | 57.5 | 41.8 | 27.9 | 0.50 |
| **TiTok (Ours, FT)** | **65.7** | **63.8** | **50.3** | **38.6** | **0.58** |

- TiTok 在所有指标上超越最强基线 ChronusOmni（FT），CountF1 提升 **+0.08**，F1@0.7 提升 **+10.7** 个百分点。
- Zero-shot 基线中 ChronusOmni 以 60.9 mIoU 领先，TiTok（FT）超出 **+4.8 mIoU**。

**Charades-STA 结果**：TiTok 56.3 mIoU vs ChronusOmni 48.6，OSA=1.00（无过分割倾向），证明多段训练不影响单段精度。

**消融**（UnAV-100）：
- 去 TTI 交织 → mIoU 降 4.5，证明边界定位依赖输入侧时间注入；
- 去特殊时间 Token（改用纯文本）→ CountF1 降 0.07，证明计数校准依赖 Token 对齐；
- 去音频 → mIoU 降 7.9，但仍高于 FT ChronusOmni；
- 去 RL → mIoU 从 65.7 暴跌至 56.8，CountF1 从 0.58 暴跌至 0.16，RL 是关键组件。

## 相关工作脉络
1. **VTG-LLM [12]**：首创专用绝对时间 Token 用于视觉单段 VTG；TiTok 将其扩展至 AV-LLM 的多段场景，并首次在输入侧交织时间 Token。
2. **ChronusOmni [4] / ARC-Hunyuan [9]**：多模态 VTG 强基线，但原生支持单段或多段时存在严重计数偏差；TiTok 通过 TTI+GDPO 显式纠正此问题。
3. **AVicuna [30]**：支持多段 AV-MSG，但输出计数偏差大（CountF1=0.08）；TiTok 在相同训练数据上大幅超越（0.58 vs 0.08）。
4. **MUSEG [22] / TempR1 [34]**：多段视觉 VTG，但纯视觉输入无法利用音频线索解决声学显著事件（如拍手）的定位与计数。
5. **GDPO [19]**：多奖励解耦归一化 RL 优化器；本文将其引入 VTG 领域，验证了其在异构定位奖励中的有效性，区别于 GRPO 在 VTG 中的直接应用。
6. **UnAV-100 [10]**：密集音视频事件基准；本文首次在其上建立多段定位评测协议并引入 CountF1，填补了 AV-MSG 评测空白。

## 局限性与未来方向
1. **UnAV-100 评测协议仍为 In-domain**：所有评估在同一数据集分布内进行，零样本泛化能力（如跨领域视频）尚需验证。
2. **Charades-STA 为纯视觉单段**：TiTok 在此仅验证了多段模型不损害单段性能，但尚未在纯视觉多段数据集上测试。
3. **RL 训练成本较高**：需 2000 步 GDPO 更新与 8 次 rollout/样本，训练耗时远高于 SFT-only 方法。
4. **CountF1 的基线 b 依赖数据集标注统计**：虽然可复用，但在标注分布差异大的跨数据集迁移时需重新计算。
5. **未来可探索方向**：① 将 TTI 推广至长视频（当前最大输出长度 256 token）；② 探索无需 SFT 冷启动的端到端 RL 训练；③ 将 CountF1 扩展至纯视觉多段 VTG 基准。

## 研究启发与可借鉴点
1. **输入侧时间 Token 交织（TTI）思路可迁移**：任何需要显式时序推理的多模态任务（如视频问答、事件检测）均可借鉴"输入-输出时间 Token 对齐"的设计，降低模型的时间幻觉。
2. **多奖励解耦归一化（GDPO）适用于复杂定位任务**：本文证明异构奖励（全局/局部/计数/精度/格式）在 VTG 中同样适用 GDPO，可为后续视觉定位、分割任务的 RL 训练提供参考范式。
3. **CountF1 指标框架可直接迁移**：其"欠分割/过分割双基准"思想可用于任何需要精确计数的序列预测任务（如音频事件检测、医疗时序标注）。
4. **冷启动 SFT + RL 的两阶段训练策略**：在引入新 Token 体系的 VLM 训练中，先做短时 SFT 建立 Token 读写能力再进 RL，是一个可靠且高效的学习路径，可复用于其他"新增特殊 Token"的场景。
5. **与团队方向结合机会**：若团队涉及多模态时序理解（视频/音频事件定位、跨模态检索），TTI 的时间对齐机制与 GDPO 的多奖励设计可直接复用，大幅缩短从零构建 AV-MSG 系统的路径。

## 关键术语表
**AV-MSG（Audio-Visual Multi-Segment Grounding）**：在未裁剪视频中，根据自然语言查询定位所有匹配事件的多个时间段，需同时利用音视频线索并预测任意数量时段。
**Count Miscalibration（计数校准偏差）**：模型预测的事件段数与真实数量不一致的现象，包括坍缩（多个事件合并为一个时段）和过分割（单一事件被拆分为多个时段）。
**TTI（Time Token Interleaving）**：将专用时间 Token 按绝对时间顺序交织插入音视频 Token 流中，使模型在输入侧感知时间并在输出侧生成同格式时间。
**GDPO（Group reward-Decoupled Normalization Policy Optimization）**：对每个奖励独立归一化后再加权合并并批次归一化的策略优化算法，避免多奖励求和后信号相互压制。
**CountF1**：由欠分割准确率（USA）与过分割准确率（OSA）的调和平均构成的计数校准指标，弥补 IoU 类指标无法量化事件数量的缺陷。
**Clip-Higher**：DAPO 中的非对称裁剪策略，将概率比的上界 $\epsilon_{\text{high}}$ 设为大于下界 $\epsilon_{\text{low}}$，为低概率探索 Token 提供更大增长空间。

## 可复现要素
- **数据集**：UnAV-100（公开）、Charades-STA（公开）；论文提供了数据重组方案（按事件类别整合为多段查询-时段样本）。
- **代码**：已开源，GitHub: https://github.com/AI-Living-Lab/TiTok
- **权重**：模型权重随代码一同开源。
- **关键超参**：SFT 阶段 lr=1e-4，batch=8，250 步，LoRA r=16 α=16；RL 阶段 lr=1e-5，batch=4，2000 步，LoRA r=32 α=64，$N_{\text{roll}}=8$，$\epsilon_{\text{low}}=0.2$，$\epsilon_{\text{high}}=0.28$，$\mu=2$，max output length=256。
- **骨干模型**：video-SALMONN 2 plus（7B）。
