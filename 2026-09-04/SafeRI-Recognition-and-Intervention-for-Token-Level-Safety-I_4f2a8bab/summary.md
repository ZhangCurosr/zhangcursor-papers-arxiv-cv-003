---
title: "SafeRI-Recognition-and-Intervention-for-Token-Level-Safety-I"
source: https://arxiv.org/pdf/2609.03544v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 11:54:07"
field: "多模态大模型安全对齐"
keywords: ["视觉语言模型", "安全对齐", "门控LoRA", "token级干预", "后对齐加固", "流式风险识别"]
innovations: ["按需门控LoRA干预：基于流式隐藏状态风险识别在危险轨迹出现后激活安全适配器，安全轨迹保持frozen-backbone", "两阶段解耦训练+过渡语句转向：识别器仅边界监督，LoRA学习从危险前缀经过渡语句到安全续写的局部重定向"]
benchmarks: ["SPA-VL-test harm", "AdvBench", "HADES", "XSTest", "MSSBench", "MMBench", "MM-Vet", "BLINK", "HarmBench PAIR"]
---

# 论文速读：SafeRI-Recognition-and-Intervention-for-Token-Level-Safety-I

## 一句话总结
论文提出 SafeRI 框架，通过在自回归生成过程中流式检测风险隐藏状态，按需激活一个门控 LoRA 将危险轨迹重定向到安全续写，从而在保持 VLM 通用能力的前提下实现后对齐阶段的安全加固。

## 研究问题与动机
- 现有安全对齐方法大多全局修改模型参数（Always-on），会不必要地扰动原本已安全的解码轨迹，造成"安全对齐税"（over-refusal、风格漂移、多模态能力下降）。
- VLM 的安全风险并非仅由输入决定，而是沿着生成轨迹逐步显现；同一有害请求可能在一条轨迹被拒绝、另一条轨迹却给出详细危害信息。
- 后对齐阶段的 VLM 已能安全处理多数请求，残余失败集中在相对少量轨迹上，因此只需针对"即将进入危险区域"的部分响应进行局部纠正，而非全局重训。
- 输入级防御在观察生成前做判断，输出级 Guard 只在有害响应已生成后介入，两者均无法匹配风险在 token 级涌现的时间性。

## 核心贡献（创新点）
- 将后对齐 VLM 安全加固重新表述为"选择性轨迹控制"问题，强调安全适配器应随 evolving response 按需激活，已安全轨迹保持 frozen-backbone 策略。与 Always-on 方法的本质区别在于干预仅在风险浮现时发生。
- 提出 SafeRI 流式识别+门控 LoRA 框架：在每个解码步用轻量线性识别器评分当前 prefix 隐藏状态，触发后进入 K 步可更新激活窗口，随后由 LoRA 重定向后续 token；与 DPO/Always-on LoRA 的本质区别是识别与干预在时间上解耦、因果且非回溯。
- 两阶段解耦训练：Stage 1 只在边界位置（pre-risk prefix 最后一个 token）监督识别器，Stage 2 冻结识别器与骨干，仅训练 LoRA 学习从危险前缀经过渡语句到安全续写的局部重写；与同类训练方法的区别是避免了分类器被生成目标优化、LoRA 隐性学习触发边界的共优化冲突。
- 提出"过渡语句"（transition statement）构造策略，使模型学会在危险意图与安全拒绝之间进行自然的局部转向，而非简单记忆完整拒绝模板；与纯拒答训练的差距在于它学习的是"转向"而非"复制拒答"。
- 系统性 ablation 揭示：中间/靠后 5 层 LoRA 插入能获得更大安全增益，阈值 τ=0.8、窗口 K=8 在安全-通用 Pareto 前沿上最优。

## 方法详解
- **问题形式化**：在解码步 t，gate $g_t$ 配置前向传播；前向输出 logits $\mathbf{z}_t$ 和最终层 hidden states $H_t$。采样 $y_t \sim \text{softmax}(\mathbf{z}_t)$， recognizer 仅基于 $h_t = H_t[:, -1, :]$ 预测风险分数 $r_t$ 并更新下一时刻 gate $g_{t+1}$，不反悔当前已计算的 logits。
- **流式风险识别器**：不在 transformer 层内插入，而是在 backbone 末尾共享一个线性头 $R_\psi(H_t) = H_t W_\psi + b_\psi$，对最后一维输出 2 类 (Safe/Unsafe) softmax。推理只取最后位置 $h_t$。
- **粘性计数器与可更新窗口**：若 $r_t > \tau$ 则 $c_{t+1} = K$，否则 $c_{t+1} = \max(c_t - 1, 0)$；gate $g_{t+1} = \mathbb{I}[c_{t+1} > 0]$。激活期间再次越过阈值会重置到 K，直到连续 K 步未再触发才关闭；这使得干预在风险持续时维持、消退时自动回归 frozen-backbone。
- **门控 LoRA**：在选定 transformer 线性投影 $W_l$ 上执行 $W_l^{(g_t)} = W_l + g_t \frac{\alpha}{r} B_l A_l$；$g_t=0$ 时完全禁用，$g_t=1$ 时启用以改变后续 token 分布。
- **训练数据构建**：从 SPA-VL 等公开数据中定位每条不安全轨迹的第一个明显危险 token 前的 prefix $u^-$，人工指定边界 $t_b$，由更强模型构造过渡语句 $b$（如 "I should stop here"）与安全续写 $s$，目标序列为 $u^- \oplus b \oplus s$；丢弃原始危险后缀。
- **Stage 1（识别器）**：仅更新 $D_\psi$，对边界位置 $h_{t_b}$ 标注 ImminentRisk，其余参与位置标注 Ordinary；损失为掩码交叉熵。骨干 θ 和 LoRA φ 冻结。
- **Stage 2（LoRA 重写）**：冻结 θ 和 $D_\psi$，仅训练 φ；对 $b \oplus s$ 段进行因果语言建模损失，prompt 和 $u^-$ 段 token 设为 ignore。
- **推理流程（Algorithm 1）**：依次执行 forward → 取最后 hidden state 判风险 → 采样下一 token → 按阈值更新粘性计数与 gate → 循环。

## 实验与结果
- **骨干与数据**：主实验 Qwen3.5-9B，另测 Qwen3.5-4B/2B、Llama3.2-Vision-11B；所有训练数据来自 SPA-VL 的重新标注，未引入额外外部数据。
- **安全基准**：SPA-VL-test harm、AdvBench、HADES、XSTest、MSSBench；通用基准：MMBench、MM-Vet、BLINK。
- **主要结果（Qwen3.5-9B）**：Safety Avg 从 Base 86.56 提升到 SafeRI 88.88（+2.32），General Avg 从 67.85 轻微降到 67.64（-0.21）；Always-on 同 LoRA 达 89.41 安全但通用降至 60.44，DPO 达 88.85 但通用仅 44.95，体现 SafeRI 更优的 safety-utility Pareto 前沿。
- **跨模型泛化**：Qwen3.5-4B（+0.71 / -1.47）、Qwen3.5-2B（+1.16 / +1.16）、Llama3.2-Vision-11B（+0.78 / -0.52）。
- **对比基线（Qwen3.5-9B 复现）**：SafeProbing (88.07, 57.23)、DTR (86.14, 66.11)、IMMUNE (82.32, 61.90)；外部参考 SafeWork-R1 72B 达 (90.11, —)，但因骨干与数据不同不可直接比较。
- **Answer-State 验证**：固定 prompt、对比安全/不安全 paired 前缀，τ=0.8 下 Qwen3.5-9B 识别器 Precision=0.885、Recall=0.920、F1=0.902，说明 gate 跟踪的是生成状态而非仅 prompt。
- **Ablation**：LoRA 插入 Middle/Last 5 层安全增益最大（+2.32）；阈值 τ=0.8 最优（+1.46），过低过激、过高漏检；窗口 K=8 最优（Safety 88.88, General 67.64），K=4 通用下降（65.00），K=12 安全回退（86.54）。
- **激活来源分析**：多数为 input-induced，drift-based 仍占一定比例；safe false activation 3.87%–7.95%。
- **HarmBench PAIR 鲁棒性**：Base ASR@10=25.31%，SafeRI 降至 24.06%。
- **开销**： classifier 仅 4.1K–8.2K 参数，Middle-5 LoRA 共 <6.57M；推理延迟 Base 7.525s → SafeRI 8.929s（+18.7%），吞吐 32.69→20.54 tok/s，高于 Always-on 的 13.63 tok/s。

## 相关工作脉络
- **训练时安全对齐**（VLGuard/SPA-VL/DPO/Token-level 数据选择等）：将安全编码进固定参数集；SafeRI 保持骨干冻结、仅按需激活适配器，避免全局重训带来的通用性损失。
- **推理时 Guardrails**（Llama Guard/Safety Reminder/CASA 等）：Safety Reminder 用 soft prompt 重激活，CASA 用条件解码；SafeRI 持续监控 evolving partial answer 并在 token 级触发 LoRA，避免"仅看 prompt"或"仅看最终输出"的时序局限。
- **内部干预与条件适配**（LoRA-Guard/aLoRA/CoCA 等）：单独使用内部 Steering 无法判断何时需要修正；SafeRI 通过隐藏状态风险识别连接二者，实现"安全时才静默、风险时才激活"。
- **Always-on LoRA 安全适配器**：直接比较表明 Always-on 安全略高但通用大幅受损；SafeRI 的可更新粘性窗口在 Pareto 前沿更优。
- **边界标注与过渡语句设计**：与 Refuse Whenever You Feel Unsafe 等"拒绝倾向"训练不同，SafeRI 学习的是从危险 prefix 到过渡语句再到安全续写的局部转向，而非完整拒答模板的 memorization。

## 局限性与未来方向
- 依赖识别器校准精度和细粒度边界标注；手动指定的 onset 位置存在噪声，且受分词器与上下文影响，边界过早起过度干预、过晚则漏检。
- 聚合通用能力保留不代表每个子基准均不下降；不同 perception/reasoning 任务上的安全-通用权衡仍存在差异。
- 当前为 off-policy 训练，识别器和 LoRA 在固定轨迹上优化；作者建议未来探索 on-policy / RL 设定，使两者能随模型演化联合适应并增强协同鲁棒性。
- 少量 safe false activation（最高约 8%）解释了微小 utility 回退，未来可引入更稳健的阈值或不确定性估计。

## 研究启发与可借鉴点
- **按需干预范式可迁移**：对已对齐模型的"选择性微调/增强"（而非全量重训）思路可推广至其他敏感领域（如隐私、偏见、风格一致性），用门控适配器替代 Always-on 路由。
- **两阶段解耦训练**：先训判别器再训生成适配器的分离优化，能有效避免分类器被生成目标带偏；可借鉴于"检测-纠正"耦合系统（如代码补全中的错误检测+修复、对话中的立场切换）。
- **过渡语句（transition statement）设计**：用简短转向语句而非直接拒答，使生成在危险发生后能自然切回安全轨道；该设计可用于任何需要"中途改道"的序列生成任务。
- **可更新粘性窗口**：以计数器替代单次触发，避免对短暂噪声的频繁切换；类似"debounce"机制可移植到任何需要稳定激活阈值的在线决策模块。
- **基于最后 hidden state 的轻量识别头**：不插入 transformer 内部、只做末尾投影，既保推理链路不变又最小化参数开销；适合部署在资源受限的 VLM 管线中。

## 关键术语表
- **SafeRI**：Recognition and Intervention 框架，基于流式风险识别与门控 LoRA 实现 token 级按需安全干预。
- **Streaming risk recognizer**：附加于 backbone 最终层之后的轻量线性分类头，对每步最后 hidden state 预测当前 prefix 为 Unsafe 的概率。
- **Gated LoRA intervention**：在选定 transformer 层中按 gate 开关切换是否注入低秩适配残差，用于将危险轨迹重定向到安全续写。
- **Sticky renewable window (K)**：风险阈值越过后将激活预算重置为 K 步，每步衰减 1，期间再次越阈则重新计 K，决定干预持续时间。
- **Transition statement (b)**：从危险前缀到安全续写之间的简短转向语句（如 "I should stop here"），帮助模型学习局部改道而非完整拒答模板。
- **Boundary-aligned supervision**：仅在危险轨迹的前一个 token 位置（pre-risk prefix 末尾）标注 ImminentRisk 进行识别器监督，其余位置 mask。
- **Two-stage decoupled training**：Stage 1 训识别器（θ, φ 冻结），Stage 2 冻结识别器与 θ、只训 φ（LoRA），避免共优化冲突。
- **Safety-utility trade-off / safety-alignment tax**：安全提升对通用能力的副作用；Always-on 方法 tax 较大，SafeRI 通过选择性激活将其最小化。

## 可复现要素
- **数据集**：SPA-VL（公开），论文基于其 relabeled 训练数据，pipeline 见附录 B；评测使用 SPA-VL-test harm、AdvBench、HADES、XSTest、MSSBench、MMBench、MM-Vet、BLINK、HarmBench PAIR。
- **代码/权重**：论文项目页 https://safe-vlm.github.io/SafeRI/；论文未明确声明代码与权重是否开源。
- **关键超参（默认）**：Stage 1 与 Stage 2 LR=1e-5、cosine warmup 0.1、max_len=4096、bf16；Stage 1 1 epoch、Stage 2 2 epochs、batch per device=2、grad accum=8；LoRA rank=16、alpha=32、dropout=0.05；插入 Middle 5 层；识别阈值 τ=0.8；粘性窗口 K=8；评测 temperature=0，judge 用 gpt-oss-20b。
