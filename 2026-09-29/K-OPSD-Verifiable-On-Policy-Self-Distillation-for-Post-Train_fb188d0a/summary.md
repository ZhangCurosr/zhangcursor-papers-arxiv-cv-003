---
title: "K-OPSD-Verifiable-On-Policy-Self-Distillation-for-Post-Train"
source: https://arxiv.org/pdf/2609.34082v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 04:10:28"
field: "可验证多模态推理与领域适应"
keywords: ["verifiable on-policy self-distillation", "AEC drawing understanding", "cross-entropy inner loss", "process-level verifier", "vision-language model post-training", "architecture engineering construction"]
innovations: ["提出 K-OPSD：无需外部教师、通过过程级验证器从 best-of-N 生成构建自教师的 on-policy 自蒸馏后训练闭环", "用交叉熵内层损失替代有界 token-wise 广义 JSD，解决高置信错误 token 梯度饱和问题", "设计 hint-conditioned rescue + leakage guard 机制，在不引入答案偏置的前提下提升难样本覆盖率"]
benchmarks: ["AECV-Bench", "ArchCAD-400K"]
---

# 论文速读：K-OPSD-Verifiable-On-Policy-Self-Distillation-for-Post-Train

## 一句话总结
论文提出 K-OPSD，一种面向 AEC（建筑、工程、施工）图纸理解的 VLM 后训练方法：通过确定性过程级验证器从模型自身的 best-of-N 生成中筛选正确轨迹构建"自教师"，并以交叉熵内层损失替代有界 JSD 进行 on-policy 蒸馏，在 AECV-Bench 上使 4B 模型达到 0.819 judge 分/0.738 综合准确率，媲美更大开源基线。

## 研究问题与动机
- **AEC 图纸理解对通用 MLLM/VLM 极难**：图纸编码密集符号约定、精确几何与隐含工程语义，常规 SFT 暴露偏差显著，且无法区分"正确推导"与"看似合理的错误"。
- **现有 RLVR 仅提供标量轨迹奖励**，缺乏 token 级监督信号；而 OPSD 的有界 token-wise 广义 JSD 在高置信度错误 token 上梯度饱和，难以有效纠正。
- **特权信息教师会引入偏差**：答案条件化教师易产生"正确答案、错误推理"轨迹，导致 per-token 损失下降但验证准确率停滞甚至恶化。
- **AEC 场景的独特优势被忽略**：图纸提取的几何、语义、代码合规性等正确性可被机器确定性验证，却鲜有工作将这一性质融入训练而非仅用于事后评测。

## 核心贡献（创新点）
- **K-OPSD 验证引导的 on-policy 自蒸馏后训练闭环**：无需外部教师模型，通过过程级验证门持续从模型自身策略构建自教师，覆盖感知、推理与最终答案全链路。
- **用交叉熵内层损失取代有界 JSD 蒸馏**：实验证明验证门控的选择信号远强于软分布匹配；交叉熵在高置信度错误 token 上保持梯度，而广义 JSD 梯度饱和。
- **Hint-conditioned rescue 机制提升难样本覆盖率**：对 best-of-N 全部失败的 prompt，以提示答案的 hint 重新采样，再通过去泄漏守卫 + 重验证后才纳入教师语料，避免"右答案、错推理"偏置。
- **小模型经后训练即可匹敌大模型零样本性能**：FT-Qwen3-VL-4B 在 AECV-Bench 上以 0.819 judge 分/0.738 综合准确率超越 Llama4-17B、Gemma-3-27B 等零样本基线；FT-Qwen3-VL-8B 在跨域 ArchCAD 上以 0.693 judge 分领先所有基线。
- **提出可复用的持续学习与自改进流水线**：验证器套件 + 逐轮 fresh LoRA adapter 拟合机制，形成项目无关的可靠图纸机读范式。

## 方法详解
- **流程总览（6 步）**：(1) On-policy rollout：冻结 SFT 基础模型上接上一轮 adapter，对每 prompt 采样 N=8 条独立完成；(2) 验证门：过程级确定性验证器 V 对所有候选打分，保留 $s(x,c,y)\ge\tau$（$\tau=0.65$）者构成 $\mathcal{C}^+$；(3) Hint-conditioned rescue：$\mathcal{C}^+=\emptyset$ 的难 prompt 以 hint $h(y)$ 条件重采样（best-of-4）；(4) 重验证+去泄漏+规范化：通过的轨迹须 $V=1$ 且未泄漏答案信息，去除 hint 后规范化为 `<think>r</think><answer>ŷ</answer>` 格式；(5) 内循环优化：在新 adapter $\phi_r$ 上对冻结基础 $\theta_0$ 以交叉熵损失 $\mathcal{L}$ 训练，早停选最低 eval loss checkpoint；(6) 同轮 held-out 评估：计算 $\Delta_r = s_r - \text{Eval}(\theta_0\oplus\phi_{r-1})$ 隔离每轮贡献。
- **过程级验证器 V**：复合 gold-type-routed 验证器，对每个完成计算多维度得分（几何维度一致性、BIM schema 符合度、符号识别正确性、规范合规性），聚合为标量 $s\in[0,1]$；硬二元门 $V=1[s\ge\tau]$。关键设计是验证推理链 r 而非仅检查提取答案 $\hat{y}$。
- **内层损失对比**：
  - 基线 OPSD/VS-OPSD 使用 token-wise 广义 Jensen-Shannon 散度（JSD），在高置信错误 token 上梯度被下界 saturate。
  - K-OPSD 使用标准交叉熵 $\mathcal{L}=-\sum_t \log \pi_\theta(c_t|x)$ 在已验证 completion 上优化，保持对错误 token 的有效梯度校正。
- **公式关键项**：
  - Rollout: $\mathcal{C}(x)=\{c_i=(r_i,\hat{y}_i)\sim\pi_r(\cdot|x):i=1,\ldots,N\}$
  - 验证筛选: $\mathcal{C}^+(x)=\{c\in\mathcal{C}(x):V(x,c,y)=1\}$
  - Rescue 采样: $\tilde{c}\sim\pi_r(\cdot|x,h(y))$
  - 规范化: $c=\text{StripHint}(\tilde{c})=\langle\text{think}\rangle r\langle/\text{think}\rangle\langle\text{answer}\rangle\hat{y}\langle/\text{answer}\rangle$
  - 目标 teacher 语料: $\mathcal{T}_r=\bigcup_{(x,y)\in\mathcal{D}}\{(x,c)\}$
  - Adapter 优化: $\phi_r=\arg\min_\phi\frac{1}{|\mathcal{T}_r|}\sum_{(x,c)\in\mathcal{T}_r}\mathcal{L}(\theta_0\oplus\phi;x,c)$

## 实验与结果
- **数据集**：AECV-Bench（训练 150 self-distillation samples + 100 SFT samples；测试 42 items，in-domain）；ArchCAD-400K 子集（测试 40 items，out-of-domain，训练未见）。
- **评估基线**（均 zero-shot，同 prompt/解码设置）：Qwen3-VL-235B、Pixtral-Large-2502、Kimi-K2.5、Gemma-3-27B、Llama4-Maverick-17B、Nemotron-Nano-12B、InternVL3.5-8B、MiniCPM-V-4.6。
- **评估指标**：Combined Accuracy（0.5×lexical token F1 + 0.5×Semantic Similarity via all-MiniLM-L6-v2）；Average Judge Score（0.75×answer accuracy + 0.1×format + 0.15×reasoning，双 judge glm-5/Nova 2 Lite 平均）。
- **主要结果**：
  - **AECV-Bench**：FT-Qwen3-VL-4B 获最高 judge 分 **0.819** 与最高综合准确率 **0.738**（并列超越 Llama4-17B、Gemma-3-27B）；FT-Qwen3-VL-8B 为 0.811/0.690。
  - **ArchCAD**：FT-Qwen3-VL-8B 获最高 judge 分 **0.693**（超越 Kimi-K2.5 的 0.619）；FT-Qwen3-VL-4B 为 0.660/0.390。
- **Ablation 关键数字**：
  - Self-teacher vs External teacher（OPD vs K-OPSD）：AECV 三 scale 上 K-OPSD judge 均超过 OPD（2B: 0.699 vs 0.665；4B: 0.819 vs 0.780；8B: 0.811 vs 0.792）。
  - 验证门效果（OPSD vs VS-OPSD vs K-OPSD）：验证门在 2B 上提升显著（AECV judge 0.588→0.676→0.699）。
  - 内层损失对比（VS-OPSD JSD vs K-OPSD CE）：六组中五组严格更好，唯一例外是 ArchCAD-4B（judge 0.666→0.660，accuracy 0.439→0.390）。

## 相关工作脉络
- **STaR / RFT / ReST 系列**：自教推理模板，保留正确样本、重采样修复失败。本文扩展为持续多轮 on-policy 蒸馏，并用过程级验证器替代简单答案匹配。
- **GKD (On-Policy Distillation)**：使用 token-wise 广义 JSD 蒸馏学生自身样本。本文指出其在高置信错误 token 梯度饱和问题，以交叉熵内层损失改进。
- **Privileged-Information Teacher 偏差分析 (Harne et al. 2026)**：证明答案条件化教师会产生"右答案、错推理"偏置。本文通过 hint stripping + leakage guard + 重验证规避此问题。
- **AEC 领域 LLM 应用 (Text2BIM/BIMgent/FloorplanVLM 等)**：多为 prompt/RAG/agentic 或纯 SFT，未利用机器可验证正确性进行训练。本文首次将 verifiable supervision 嵌入后训练循环。
- **AECV-Bench / ArchCAD-400K 基准**：先前工作仅用作评测工具，本文将其作为训练信号来源，结合 SME-validated ground truth 形成可验证数据闭环。
- **RLVR (Verifiable RL)**：仅提供 trajectory-level 标量奖励，无 per-token 目标。本文在相同 verifiable 假设下提供稠密 token 级监督。

## 局限性与未来方向
- **训练/评测数据规模小**：仅 100 SFT + 150 self-distillation 样本，测试集 42/40 items，统计功效有限；需扩大数据量并做置信区间与显著性检验。
- **梯度饱和机制缺乏实证验证**：理论推导清晰但尚未做 per-token gradient norm 探测与 $\beta$ sweep；$\beta=0.5$ 固定，未做敏感性分析。
- **未做质量-成本分析**：GPU-hours、rollout 消耗、per-round 样本效率等经济性指标缺失，"样本高效"叙事尚待量化。
- **验证器可靠性未经充分刻画**：缺少与 SME 判断的 precision/recall/false-positive 对比，及按题型 coverage 的细粒度分析。
- **未探索 rejected best-of-N 的负学习信号**：当前 pipeline 丢弃所有未通过验证的候选，可能浪费信息。
- **跨模型族泛化未充分验证**：仅在 Qwen3-VL 系（2B/4B/8B）上验证，需扩展到更多基座模型。

## 研究启发与可借鉴点
- **过程级验证器设计可迁移至其他可验证领域**（数学推理、代码生成、科学计算）：将验证从"答案匹配"升级到"推导链校验"，可系统性减少 hallucination。
- **交叉熵内层损失替代有界 JSD 的经验对 OPSD 改进有普适参考价值**：在高置信错误 token 上保持梯度是蒸馏损失设计的关键判据。
- **Hint-conditioned rescue + leakage guard 机制可保护 self-distillation 中的信息污染**：对任何需要答案条件化生成再反推的训练流程均有借鉴意义。
- **Fresh LoRA adapter + frozen base 的逐轮更新策略**避免了灾难性遗忘，同时维持稳定性，适合持续学习场景。
- **与团队方向结合机会**：若团队涉及可验证推理（code/math/science）、领域适应 VLM、或 self-distillation 方法，本工作的验证门控设计、交叉熵蒸馏损失、以及多轮持续学习范式均可直接移植或扩展。

## 关键术语表
- **K-OPSD**：Verifiable On-Policy Self-Distillation，一种无需外部教师、通过过程级验证器筛选模型自身 best-of-N 生成作为蒸馏目标的 VLM 后训练方法。
- **OPSD (On-Policy Self-Distillation)**：用模型当前策略生成的样本进行 token-wise 蒸馏的训练范式，区别于 off-policy SFT 或外部教师蒸馏。
- **Process-level Verifier**：不只看最终答案，而是对推理链 r 的几何一致性、schema 符合性、符号识别、规范合规等多维度进行确定性验证的复合验证器。
- **Generalized Jensen-Shannon Divergence (JSD)**：OPSD 使用的有界 token 级散度损失，在高置信错误 token 上梯度饱和是其已知缺陷。
- **Hint-conditioned Rescue**：对 best-of-N 全部失败的难 prompt，注入答案 hint 重新采样生成纠正轨迹，再经去泄漏守卫和重验证后纳入教师语料的补救机制。
- **Leakage Guard**：防止 hint 信息渗入训练目标的过滤机制，拒绝包含答案引用、复制或直接透露的答案轨迹。
- **Best-of-N (BoN)**：对同一 prompt 采样 N 条候选，经验证器筛选后选取最优通过者的采样策略，本文取 N=8（rescue 时 N=4）。
- **AECV-Bench**：面向建筑与工程图纸理解的 multimodal VQA 评测基准，含 42 个 held-out 测试项。

## 可复现要素
- **数据集**：AECV-Bench（arXiv:2601.04819）、ArchCAD-400K（arXiv:2503.22346），论文未声明自有代码仓库。
- **代码/权重**：论文未提及开源代码或模型权重。
- **关键超参**：LoRA rank=16, $\alpha$=32, dropout=0.05, 4-bit QLoRA + bf16；N=8（rescue N=4）；temperature=1.0, top-p=0.95, max 512 tokens；验证阈值 $\tau$=0.65；内循环最多 10 epochs, patience=3；held-out 比例 15%（round 选择）/ 20%（per-epoch eval）；$\beta$=0.5（JSD 基线）。
- **硬件**：4 GPU 数据并行（rollout/rescue/update 分片），单进程负责候选筛选与指标聚合。
