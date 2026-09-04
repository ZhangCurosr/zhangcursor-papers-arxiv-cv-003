---
title: "SafeRI-Recognition-and-Intervention-for-Token-Level-Safety-I"
source: https://arxiv.org/pdf/2609.03544v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 18:41:28"
field: "多模态大模型安全对齐"
keywords: ["vision-language models", "safety alignment", "token-level intervention", "gated LoRA", "post-alignment hardening", "multimodal safety"]
innovations: ["提出流式风险识别+门控LoRA的按需干预框架，仅在生成轨迹出现不安全迹象时激活安全适配器", "两阶段解耦训练（边界分类器 + LoRA安全重写）避免组件间优化冲突", "粘性可刷新窗口机制实现风险持续跟踪与自动关闭"]
benchmarks: ["SPA-VL-test harm", "AdvBench", "HADES", "XSTest", "MSSBench", "MMBench", "MM-Vet", "BLINK", "HarmBench PAIR"]
---

# 论文速读：SafeRI-Recognition-and-Intervention-for-Token-Level-Safety-I

## 一句话总结
论文提出了 SafeRI 框架，通过在 VLM 自回归生成过程中流式识别潜在不安全的隐藏状态，并选择性激活参数高效的 LoRA 安全适配器来重定向不安全轨迹，实现了对已对齐 VLM 的按需 token 级安全加固，避免了传统全局安全微调对正常生成路径的不必要扰动。

## 研究问题与动机
- 现有 VLM 安全方法（训练时对齐 / 始终运行的安全适配器）一旦启用就会全局修改模型行为，导致即使在安全请求或已安全拒绝的轨迹上也产生不必要的干预，造成"安全对齐税"（过度拒绝、风格漂移、多模态能力下降）。
- 对于已进行安全对齐的 instruction-tuned VLM，残留失败集中在相对较小的生成轨迹子集，是否需额外干预取决于演化的生成状态而非单次 prompt 判断，但现有方法缺乏这种时序敏感性。
- 输入级防御仅在观察生成前决策，输出级护栏仅在 unsafe 响应产生后生效，两者均无法在风险萌芽阶段进行及时干预。
- 风险识别与干预必须在时间上耦合：仅识别不干预只会记录失败，仅干预无识别则可能过早干预或扰动所有响应。

## 核心贡献（创新点）
- 将后对齐 VLM 安全加固形式化为选择性轨迹控制问题，区别于传统全局重学安全行为的范式，强调"仅在风险显现时激活"的按需干预原则。
- 提出 SafeRI 流式识别 + 门控 LoRA 架构：轻量线性风险识别器读取当前前缀的最后一层隐藏状态预测 unsafe 概率，通过阈值触发可更新粘性计数器控制 LoRA 门的开启，实现从冻结基座到安全重定向的动态切换。
- 设计两阶段解耦训练策略：Stage 1 仅训练边界分类器，Stage 2 在冻结识别器和基座后仅训练 LoRA 进行 transition statement + safe continuation 的重写，避免 classifier 被优化为文本生成器、LoRA 隐式学习触发边界。
- 构建四元组训练数据 (q, u⁻, b, s)：从 SPA-VL 中定位 unsafe 轨迹的 pre-risk 前缀，人工指定边界位置，并用强模型重写为"过渡语句 + 安全延续"的目标响应，保留原始 prompt 和 pre-risk prefix 作为监督上下文。
- 提出答案状态验证方法：构造相同 prompt 下安全 / 不安全 paired prefix 的 held-out 集合，证明门控真正追踪的是 assistant 演化轨迹而非仅依赖 prompt 风险。

## 方法详解
**问题形式化**：在解码步骤 t，gate g_t 控制当前前缀 y_{<t} 的前向传播，产生 logits z_t 和 final-layer 隐藏状态序列 H_t。risk score r_t 更新下一时刻 gate g_{t+1}，不影响当前 token y_t 的采样分布。

**Streaming Risk Recognition**：识别器是 backbone 最后一层之后的共享线性头 R_ψ(H_t) = H_t W_ψ + b_ψ ∈ ℝ^{B×L_t×2}，分类 Safe(0) / Unsafe(1)。推理时仅取最后一个 prefix 位置 h_t = H_t[:, -1, :]，计算 r_t = [softmax(R_ψ(h_t))]_1（unsafe 概率）。

**粘性门控机制**：当 r_t > τ 时重置计数器 c_{t+1} = K（粘性窗口），否则 c_{t+1} = max(c_t - 1, 0)；gate g_{t+1} = I[c_{t+1} > 0]。窗口可刷新：识别器在 LoRA 激活期间再次超限则重置 K，确保干预持续到风险消退。

**Gated LoRA Intervention**：对选定 transformer 层中的线性投影 W_l，门控权重为 W_l^{(g_t)} = W_l + g_t · (α/r) B_l A_l；g_t = 0 时禁用所有 adapter，g_t = 1 时启用安全重定向。

**两阶段训练目标**：
- Stage 1（边界分类器）：仅训练 D_ψ，监督 pre-token 状态 h_{t_b}（边界位置）为 ImminentRisk，其余位置 mask，损失为 masked cross-entropy。
- Stage 2（LoRA 安全重写）：冻结识别器和基座，仅训练 LoRA 参数 φ，对 transition statement b 和 safe continuation s 部分施加 causal LM loss，prompt 和 prefix 部分 label 设为 ignore。

**训练数据构造**：从 SPA-VL 数据集 relabel，人工指定 unsafe 轨迹的边界索引 t_b，前缀 y_{<t_b} = u⁻ 保留，原 risky suffix 丢弃，由强模型生成过渡语句 b（如 "I should stop here"）和安全延续 s，形成 u⁻ + b + s 重写目标。

## 实验与结果
- **Backbone 与数据**：主模型 Qwen3.5-9B，另评估 Qwen3.5-2B / 4B、Llama3.2-Vision-11B；训练数据全部来自公开 SPA-VL 数据集重新标注，无额外数据引入。
- **安全基准**：SPA-VL-test harm、AdvBench、HADES、XSTest、MSSBench；通用基准：MMBench、MM-Vet、BLINK。
- **基线对比**：Frozen Base、Always-on LoRA、DPO、SafeWork-R1（外部 72B）、SafeProbing（复现）、DTR（复现）、IMMUNE（适配到 Qwen3.5-9B）。
- **安全–通用 Pareto 最优**：在 Qwen3.5-9B 上，SafeRI 安全分 88.88（Base: 86.56，↑2.32），通用分 67.64（Base: 67.85，↓0.21）；Always-on 安全 89.41 但通用仅 60.44（↓7.41），DPO 安全 88.85 但通用 44.95（↓22.90），证明选择性干预显著降低安全对齐税。
- **跨模型泛化**：Qwen3.5-4B（+0.71 / −1.47）、Qwen3.5-2B（+1.16 / +1.16）、Llama3.2-Vision-11B（+0.78 / −0.52），均在保持通用的同时提升安全。
- **答案状态验证**：在 XSTest paired prefix 测试集上，τ=0.8 时分类器 Precision=0.885、Recall=0.920、F1=0.902，证明 gate 追踪 assistant 轨迹状态而非仅 prompt。
- **消融结论**：LoRA 插入 middle/last 5 层效果最佳（+2.32 安全）；τ=0.8 为最佳阈值（过度激活降低通用分，阈值过高遗漏纠正）；K=8 为最优干预窗口（K=4 通用分降至 65.00，K=12 安全分回落至 86.54）。
- **参数开销**：classifier 仅需 4.1K–8.2K 参数，middle-five-layer LoRA 约 6M，总计 < 6.57M（Qwen3.5-9B）。
- **推理效率**：SafeRI latency 8.929s/sample（Base 7.525s，Always-on 5.023s），throughput 20.543 tokens/s（Base 32.692，Always-on 13.627），相比 Always-on 吞吐量高 50.8%。
- **攻击鲁棒性**：在 HarmBench PAIR 攻击下，SafeRI ASR@1 从 20.00 降至 19.38，ASR@10 从 25.31 降至 24.06。
- **训练稳定性**：三种子随机种子下 Qwen3.5-9B 安全分 88.88±0.12、通用 67.64±0.59，方差极小。

## 相关工作脉络
- **Training-Time Safety Alignment（VLGuard、SPA-VL、DPO 等）**：将安全编码进固定参数集全局生效；SafeRI 保持 backbone 冻结，仅按需激活 safety LoRA。
- **Inference-Time Guardrails（Safety Reminder、CASA、Llama Guard 等）**：输入/输出侧分类或条件解码；SafeRI 在内部 hidden state 层面动态检测并干预。
- **Internal Intervention（CoCA、InferAligner、LoRA-Guard、Activated LoRA 等）**：修改 activation 或 token distribution，但缺乏对 evolving response 的风险感知；SafeRI 通过 recognizer 桥接内部干预与条件适配。
- **Prompt-based Guardrails（AdaShield、UniGuard）**：依赖 prompt engineering 或 soft prompt 重激活安全感知；SafeRI 直接读取 token 级 hidden state 决策。
- **Representation Constraint Methods（token-level data selection、decoupled refusal training）**：在训练阶段约束安全关键 token；SafeRI 属于推理时自适应干预，无需重新训练基座。

## 局限性与未来方向
- 依赖准确的 detector 校准和精细的 unsafe 生成边界标注，边界位置的人工指定可能引入 annotation noise 和对 tokenization 的敏感性。
- 当前 for-off-policy 训练范式使用固定轨迹优化识别与干预，无法捕捉模型演化过程中的协同适应。
- aggregate 多模态能力保持不代表各 benchmark 均匀提升，安全–通用权衡仍是任务相关的。
- 未来可扩展到 on-policy 或 reinforcement-learning 设置，使 recognizer 和 LoRA 联合适应 evolving model 生成的轨迹。
- 当前仅在有害 prompt subset 上评估效率，full-stream 场景下的延迟开销有待进一步分析。

## 研究启发与可借鉴点
- **流式状态监控 + 按需参数激活**的范式可迁移到其他序列生成任务的风险控制（如 hallucination detection、factuality gating），无需重新训练基座。
- **两阶段解耦训练**策略（先训识别器、再训干预器）避免了多目标联合优化的冲突，适用于任何"检测 + 修正"的双模块架构。
- **答案状态验证**（固定 prompt、对比 paired safe/unsafe prefix）的实验设计可有效区分模型决策依据，可复用于其他 VLM 安全 / 对齐论文的可信度验证。
- **transition statement 设计**（如 "I should stop here"）为模型提供自然的安全转向表达，可作为安全的"软拒绝"模板复用。
- 粘性计数器（sticky window with renewal）机制在其它需要持续性干预的任务（如对话安全、代码生成纠错）中具有通用价值。

## 关键术语表
- **SafeRI**：Recognition and Intervention framework，面向 VLM token 级安全的流式识别与门控 LoRA 干预框架。
- **Post-alignment safety hardening**：对已进行安全对齐的 VLM 进行额外的安全加固，而不改变其已学会的安全行为。
- **Gated LoRA**：通过门控信号 g_t 在推理时按需启用/禁用的低秩适配器，g_t=0 时完全等价于冻结基座。
- **Sticky intervention window (K)**：识别器超阈值后 LoRA 持续激活的解码步数，窗口内再次超阈值可刷新重置。
- **Pre-risk prefix (u⁻)**：unsafe 轨迹中首个明显危险 token 之前的 assistant 响应前缀，作为识别器和 LoRA 的输入上下文。
- **Boundary state (h_{t_b})**：人工指定的生成轨迹安全边界位置对应的 pre-token 隐藏状态，用于 Stage 1 分类器监督。
- **Safety-alignment tax**：全局安全微调对正常生成轨迹造成的不必要的隐藏状态扰动和能力退化现象。
- **Imminent Risk vs Ordinary**：识别器二分类标签，分别表示"即将进入不安全区域的当前状态"和"正常生成状态"。

## 可复现要素
- **数据集**：SPA-VL（公开），训练数据为 SPA-VL 重新标注版本，论文未提及独立开源数据集发布。
- **代码/权重**：项目页面 https://safe-vlm.github.io/SafeRI/，论文未明确声明代码 / 权重开源状态。
- **关键超参**：τ=0.8（阈值），K=8（粘性窗口），LoRA rank=16、alpha=32、dropout=0.05，插入 middle 5 layers，lr=1e-5，epochs=1（Stage1）/2（Stage2），effective batch size=16，max length=4096。
- **硬件**：8× NVIDIA H200 GPU（Stage 1），其余未明确。
