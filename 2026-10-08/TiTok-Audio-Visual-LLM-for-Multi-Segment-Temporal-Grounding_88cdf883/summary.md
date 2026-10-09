---
title: "TiTok-Audio-Visual-LLM-for-Multi-Segment-Temporal-Grounding"
source: https://arxiv.org/pdf/2610.09408v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 03:17:53"
field: "多模态时序定位"
keywords: ["Multi-Segment Temporal Grounding", "Audio-Visual LLM", "Reinforcement Learning", "Count Miscalibration", "Time Token Interleaving", "GDPO"]
innovations: ["提出 TTI 机制，将特殊时间令牌交织入音视频流实现输入输出时间对齐", "设计五解耦多片段导向奖励并采用 GDPO 优化，有效缓解计数失配", "建立 UnAV-100 评估协议并提出 CountF1 指标量化计数校准能力"]
benchmarks: ["UnAV-100", "Charades-STA"]
---

# 论文速读：TiTok-Audio-Visual-LLM-for-Multi-Segment-Temporal-Grounding

## 一句话总结
本文提出 TiTok，一个面向未裁剪视频音频-视觉多片段时序定位（AV-MSG）的音频-视觉大语言模型，通过时间令牌交织（TTI）机制注入显式绝对时间表示，并结合解耦的多片段导向奖励与 GDPO 强化学习，有效缓解现有方法的"计数失配"（count miscalibration）瓶颈，在 UnAV-100 上取得 SOTA。

## 研究问题与动机
1. **任务定义缺失统一基准**：AV-MSG 要求在未裁剪视频中为任意查询定位多个时段片段，但现有工作仅部分覆盖该设定，缺乏统一的评估协议。
2. **视觉中心主义忽略声学线索**：大多数 MLLM-based VTG 方法几乎完全依赖视觉流，而许多事件（如狗叫、枪声）在音频上更易被精确定界且可能多次出现。
3. **计数失配（Count Miscalibration）是核心瓶颈**：已有 AV-LLM（如 AVicuna、ChronusOmni）频繁将多个事件发生合并为一个区间，或将单次事件过度分割为多个片段，现有优化目标（仅 IoU + 格式奖励）未显式惩罚事件基数错误。
4. **多奖励融合导致信号衰减**：GRPO 将多个异构奖励求和后统一归一化，使细粒度定位信号被易学奖励（如格式奖励）压制，难以同时优化多个目标。

## 核心贡献（创新点）
1. **提出 TiTok 框架与 TTI 机制**：首次将特殊时间令牌（special time tokens）交织入音频-视觉流，使模型在输入侧感知与输出侧预测共享同一显式时间轴，与 VTG-LLM 等仅视觉场景下的时间令牌设计形成本质区别。
2. **设计五项解耦多片段导向奖励（global/local/count/precision/format）**：分别覆盖整体重叠、逐片段精度、事件计数、边界紧致性和格式合法性，与仅含 IoU+格式的传统 VTG 奖励体系有本质区别。
3. **采用 GDPO 优化多奖励组合**：对每个奖励独立归一化后再加权组合，避免 GRPO 中多奖励融合导致的梯度混淆，同时结合 DAPO 的 Clip-Higher 与去除 KL 散度以增强探索能力。
4. **建立 UnAV-100 上的 AV-MSG 评估协议并引入 CountF1**：CountF1 通过低估准确度（USA）与过分割准确度（OSA）的调和均值量化计数失配，填补了现有重叠指标无法直接衡量事件基数的空白。
5. **在 UnAV-100 和 Charades-STA 上均取得 SOTA**：TiTok 在 UnAV-100 上达到 65.7 mIoU 和 0.58 CountF1，相比微调后的 ChronusOmni 分别提升 2.3 和 0.08。

## 方法详解
**Time Token Interleaving (TTI)**：在按时间顺序交错的音频-视觉 token 块中，每个块前插入由特殊时间令牌组成的绝对时间戳（格式为 `<tX><tX><tX><tdot><tX>`，如 16.7s → `<t0><t1><t6><tdot><t7>`），使模型能在输入侧感知绝对时间，并在输出侧生成同格式时间戳，实现输入-输出的时间语义对齐。

**两阶段训练**：
- **Stage 1（冷启动 SFT）**：仅在 UnAV-100 训练集上运行 250 步（batch size=8，lr=1e-4），冻结视听编码器，更新视听对齐器和特殊时间令牌嵌入/LM head，对 LLM 应用 LoRA（r=16, α=16），使模型学会读取和生成特殊时间令牌。
- **Stage 2（GDPO-based RL）**：冻结前序模块，仅对 LLM 应用 LoRA（r=32, α=64）运行 2,000 步（batch size=4，lr=1e-5），每 prompt 采样 $N_{roll}=8$ 个响应（最大输出长度 256），执行 μ=2 次策略更新。

**GDPO 优化目标**：对每个奖励 $k$ 在 group 内独立归一化得 $\hat{A}_i^{(k)}$，加权求和得 $\tilde{A}_i$，再在整个 batch 内归一化得 $\hat{A}_i$。结合 Clip-Higher（$\epsilon_{low}=0.2, \epsilon_{high}=0.28$）和去除 KL 散度惩罚，最终目标如公式 (3) 所示。

**五项奖励设计**：
- $r_{global}$：预测并集与 GT 的 F1（精确率+召回率），评估整体重叠。
- $r_{local}$：按输出顺序配对最高 GIoU 的 GT 片段，以 $\max(N_{gt}, N_{pred})$ 为分母，使计数 mismatches 反映在奖励中。
- $r_{count}$：$1 - |N_{pred}-N_{gt}|/\max(N_{pred}, N_{gt})$；当 $N_{gt} \geq 2$ 且 $N_{pred}=1$ 时强制为 0，严厉惩罚合并事件。
- $r_{prec}$：独立精确率（Prec），防止模型通过膨胀边界匹配计数。
- $r_{format}$：输出完全符合时间令牌格式则得 1，否则 0。

## 实验与结果
**数据集**：主评测在 UnAV-100 测试集上进行，另在 Charades-STA 测试集上做单片段对比。训练集含 10,358 样本（UnAV-100 的 7,207 条中 4,055 为多片段 + Charades-STA 的 3,151 条），单/多片段比约 6:4。

**评估指标**：mIoU、F1@{0.3, 0.5, 0.7}、CountF1。

**主要结果（UnAV-100）**：TiTok 在所有指标上领先。mIoU 达 65.7（+2.3 over FT ChronusOmni 的 63.4），CountF1 达 0.58（+0.08），F1@0.7 达 38.6（+10.7）。相比零样本 ChronusOmni（mIoU=60.9, CountF1=0.39）提升显著。

**Charades-STA 结果**：TiTok mIoU=56.3 vs ChronusOmni=48.6，F1@0.7=39.7 vs 24.8，证明多片段训练不损害单片段性能。

**消融关键数字**：去除 RL 后 mIoU 骤降至 56.8、CountF1 降至 0.16；去除音频后 mIoU=57.8；w/o interleaving 比 w/o special tokens 在严格 F1 阈值下降幅更大；w/ GRPO 比 w/ GDPO 在 mIoU 和 F1@0.7 上更差（但 CountF1 略高）。

## 相关工作脉络
1. **ChronusOmni / ARC-Hunyuan**：视觉为主的单片段或多片段 VTG 模型，不支持音频，且缺乏显式时间令牌表示；TiTok 在音频支持、多片段预测和时间令牌注入上全面超越。
2. **MUSEG / TempR1**：支持多片段预测但仅用视觉，且无显式时间令牌；TiTok 引入音频和 TTI 机制实现更精确的计数校准。
3. **VTG-LLM**：首次引入专用时间令牌但仅限视觉场景；TiTok 将其扩展到音频-视觉交错流中，解决跨模态时间对齐问题。
4. **AVicuna**：支持音频-视觉多片段预测，但计数失配严重（CountF1 仅 0.08）；TiTok 的 TTI+解耦奖励机制显著改善了计数校准。
5. **GRPO vs GDPO**：GRPO 将多奖励求和后归一化，导致信号衰减；GDPO 独立归一化每个奖励，保留细粒度定位信号，这是多奖励 RL 训练的关键差异。
6. **Qwen2.5-Omni**：通用音频-视觉多模态模型，零样本 VTG 性能较弱（mIoU=12.6），体现领域适配的必要性。

## 局限性与未来方向
1. **评估仍局限于在域数据**：所有评测均在 UnAV-100 和 Charades-STA 上进行，跨域泛化能力尚未经充分验证。
2. **CountF1 的计算依赖基准注解**：虽然论文称 baseline $b$ 可复用于其他基准，但未在多个不同数据集上验证 CountF1 的稳定性和可比性。
3. **推理效率未讨论**：TTI 增加了输入 token 长度，RL 阶段每 prompt 采样 8 个响应，推理和训练成本可能较高，论文未讨论计算开销。
4. **未探索更复杂的多模态融合策略**：TTI 采用简单的时间令牌交织，未涉及更精细的跨模态对齐机制（如跨注意力、时间感知 cross-modal fusion）。
5. **未来方向**：可扩展至更多音频-视觉 benchmarks、探索更长视频的视频级别计数校准、结合推理链（reasoning chain）进一步改善边界预测。

## 研究启发与可借鉴点
1. **特殊时间令牌的输入-输出对齐思路**可迁移到任意需要时序输出的多模态任务（如视频问答、事件检测），解决 LLM 对时间语义理解不足的问题。
2. **GDPO 的多奖励独立归一化策略**适用于任何需要同时优化多个异构目标的 RL 训练场景（如同时优化准确性和多样性），避免奖励尺度差异导致的信号淹没。
3. **CountF1 的评估理念**——通过针对低估/过分割两个方向分别定义准确度再取调和均值——可推广到任何需要精确计数预测的任务（如目标检测中的 instance counting）。
4. **冷启动 SFT + RL 的两阶段范式**对于引入新 token 类型（如特殊时间令牌）的大型模型微调具有通用参考价值，确保 RL 阶段有可用的 rollout。
5. **Clip-Higher + 去除 KL 散度的训练技巧**来自 DAPO，可直接复用于其他 LLM 强化学习场景以提升探索能力。

## 关键术语表
**Audio-Visual Multi-Segment Temporal Grounding (AV-MSG)**：在未裁剪视频的音视频流中，针对自然语言查询定位所有匹配的时段片段，需同时预测数量和边界。
**Count Miscalibration（计数失配）**：模型预测的事件片段数量与真实数量不一致的现象，包括合并多个事件为一段（under-segmentation）或单个事件被拆成多段（over-segmentation）。
**Time Token Interleaving (TTI)**：将特殊时间令牌（`<t0>`~`<t9>`, `<tdot>`）按绝对时间格式穿插入音视频 token 流中，使输入侧时间感知与输出侧时间预测共享同一表示。
**Special Time Tokens（特殊时间令牌）**：独立于普通数字词表的专用时间 token，避免数字与时间的概念漂移，减少量化误差。
**Group reward-Decoupled Normalization Policy Optimization (GDPO)**：对每个奖励在 group 内独立归一化后加权求和，再在 batch 内统一归一化的多奖励 RL 优化算法，避免 GRPO 的奖励信号衰减。
**CountF1**：由低估准确度（USA）和过分割准确度（OSA）构成的调和均值指标，专门量化多片段时序定位中的计数校准能力。
**Generalized IoU (GIoU)**：在非重叠区间也能提供非零梯度的边界框回归损失/度量，通过最小包围区域归一化。
**Clip-Higher**：将策略优化的 clipping 上界设得比下界更大，允许低概率探索 token 有更大的概率增长空间，防止策略熵坍缩。

## 可复现要素
- **数据集**：UnAV-100（公开）、Charades-STA（公开）；UnAV-100 上的 AV-MSG 评估协议为本文新建立。
- **代码**：已开源，链接为 https://github.com/AI-Living-Lab/TiTok。
- **权重**：论文未明确说明是否开源，代码仓库应包含相关模型权重。
- **关键超参**：SFT 阶段 lr=1e-4、batch size=8、250 steps、LoRA r=16 α=16；RL 阶段 lr=1e-5、batch size=4、2000 steps、LoRA r=32 α=64、$N_{roll}=8$、μ=2、$\epsilon_{low}=0.2$、$\epsilon_{high}=0.28$、最大输出长度 256。
- **骨干模型**：video-SALMONN 2 plus (7B)。
