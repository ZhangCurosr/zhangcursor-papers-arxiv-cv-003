---
title: "INFIMED2-A-GENERALIST-MEDICAL-MULTIMODAL-FOUNDATION-MODEL-FR"
source: https://arxiv.org/pdf/2609.34798v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 04:07:50"
field: "医学多模态基础模型"
keywords: ["medical multimodal LLM", "continued pretraining", "stability-aware supervision", "RLVR", "data curation"]
innovations: ["阶段感知的三阶段CPT数据配比策略", "答案稳定性估计驱动的多档路由重建生成高质量SFT监督", "RLVR结合pass@8难度感知的中等难度优先优化"]
benchmarks: ["MMMU Medical-test", "MMMU-Pro Medical-10", "MedXpertQA-MM", "PMC-VQA-clean", "OmniMedVQA"]
---

# 论文速读：INFIMED2-A-GENERALIST-MEDICAL-MULTIMODAL-FOUNDATION-MODEL-FR

## 一句话总结
InfiMed2 是通过**阶段感知数据设计**（CPT三阶段 + SFT稳定性感知响应重建 + RLVR）训练的 4B/27B 通用医学多模态基础模型家族；其 4B 模型超越更大规模的 Qwen3.5-9B，27B 模型在五个评测基准上取得开放权重模型最高平均分 73.72%。

## 研究问题与动机
- **CPT 数据异质性难以统一处理**：医学来源（电子书、科学文献、网络文档）在结构、粒度、信息密度上差异极大，单一静态数据混合无法同时支撑"广泛知识获取"和"后期收敛巩固"两个阶段的需求。
- **视觉编码器适应存在两难**：固定视觉编码器会限制医学视觉适配，但从训练一开始就全参数联合优化又可能破坏预训练阶段的图文对齐表示。
- **SFT 阶段监督信号薄弱**：医学 VQA 标注多为短实体/类别/判断，仅评估答案正确性，对引导模型产出信息丰富、与视觉证据一致的合理解释缺乏有效监督；直接用教师解释替换短答案仍可能出现 grounding 不足或与已验证答案不一致的问题。
- **教师可靠性未经筛选直接用于训练不可靠**：不同样本的教师生成质量差异大，需建立基于一致性与正确性的可信度度量来决定如何路由再生。

## 核心贡献（创新点）
1. **阶段感知的三阶段 CPT 训练策略**：视觉编码器适配 → 55.68B-token 广泛知识学习 → 学习率衰减阶段的高质 2.5B-token 混合精炼，与已有工作仅关注数据规模不同，本文首次系统验证"数据混合 + 训练阶段 + 可训练组件"三者协同对医学视觉语言能力的增益。
2. **答案稳定性感知的 SFT 响应重建管线**：通过 K=8 次重复采样估计答案稳定性 $S_i$，并据此路由：高稳定性（$6 \leq C_i \leq 8$）走答案掩码重建 + 熵最小化选择；中稳定性（$3 \leq C_i \leq 5$）升级至 Gemini 3.1 Pro 再生；低稳定性直接丢弃。本质区别在于同时保证"答案一致性"（硬约束）与"信息密度"（熵排序），而非单纯追求更长生成。
3. **源特定处理构建互补 CPT 语料库**：医学电子书按学科/体裁重采样（临床学科占比从 55% 提至 75%）、PMC-InterCPT 保留图文上下文关联、医学网页数据通过三级质量分类器过滤并恢复图文交织结构、通用多模态数据作为 replay 防止过度专业化。与 PMC-15M / BIOMEDICA 等仅使用孤立图-文对的数据集相比，保留了figure-caption- context 三元关联。
4. **可验证奖励强化学习（RLVR）+ 难度感知采样**：基于 pass@8 估计样本难度，优先选择 3–5 个正确 rollout 的中等难度样本，避免过早收敛于易题或长期无法学习的难题，从而在 4B 尺度上获得额外泛化提升。
5. **在同等评测协议下取得最优开放权重性能**：InfiMed2-4B 平均 66.73%（超 Qwen3.5-9B 2.02 pts），InfiMed2-27B 平均 73.72%（领先同规模开放模型），证明数据设计可部分替代纯规模堆叠。

## 方法详解
**CPT 数据构建**
- 医学电子书：156k 本经 MinerU 转 Markdown，书籍级去重后按 subject × genre 重采样，最终 60,433 本 / 29.78B token（含视觉 token）。
- 生物医学科学文献：PMC-InterCPT 10.11M 条 / 9.63B token，保留 figure↔context 链接与多图同上下文序列。
- 医学网络数据：从 OmniCorpus-CC / OBELICS / MINT-1T 中检索 33B token，MinHash 去重后经三级质量分类器（98.5% 准确率检索器 + 72.5% macro-F1 质量分类器）过滤，最终 11.27B token。
- 通用多模态 replay：LLaVA-OneVision-1.5 中随机采样 5B token。
- **Vision-encoder adaptation 子集**：1B token（73.7% 为视觉 token，1.71M 张图-文对）。
- **Learning-rate decay 混合**：2.5B token，含医学电子书 40%、BIOMEDICA 20%、通用 replay 15.84%、医学网络文章 3.76%、医学习题/VQA 20.40%。

**稳定性感知响应重建（公式核心）**
1. 教师采样：$r_i^{(k)} \sim p_{\theta_T}(r \mid I_i, q_i),\ k=1,\dots,K\ (K=8)$
2. 答案一致性计数：$C_i = \sum_{k=1}^K \mathbb{I}[\mathcal{M}(\hat{a}_i^{(k)}, a_i^*) = 1]$，$S_i = C_i/K$
3. 高稳定路由（$6 \le C_i \le 8$）：$\tilde{r}_i^{(k)} = \mathrm{MaskAnswer}(r_i^{(k)}, \hat{a}_i^{(k)})$，学生重建 $\bar{r}_i^{(k)} \sim p_{\theta_R}(r \mid I_i, q_i, \tilde{r}_i^{(k)})$，仅在 $\mathcal{R}_i^+$（提取答案匹配 $a_i^*$ 且原教师回答正确）中按平均 token 熵最小化选优：$r_i^* = \arg\min_{r \in \mathcal{R}_i^+} \mathcal{H}(r)$，其中 $\mathcal{H}(\bar{r}_i^{(k)}) = \frac{1}{T_k}\sum_t [-\sum_v p_t(v)\log p_t(v)]$。
4. 中稳定路由（$3 \le C_i \le 5$）：调用 Gemini 3.1 Pro 做 $(I_i, q_i, a_i^*)$ 条件再生；$C_i < 3$ 丢弃。
5. 最终输出 106,183 条稳定性感知示例，占 SFT 混合 44.79%。

**RLVR**
- 数据集：21,529 条（19,529 医学 VQA + 2,000 文本 USMLE）
- 每 prompt 温度 1 采样 8 rollouts，奖励 = 0.1×格式合规 + 0.9×答案准确，KL 正则沿用默认。
- pass@8 难度估计：优先 3–5 个正确样本，过易/过难均降采样。

**训练超参**
- CPT 全部 AdamW，global batch=256；vision 适配 max_len=8192，后两阶段 max_len=12288；LR 分别为 $2\times10^{-6}\to2\times10^{-7}$、$1\times10^{-5}$ 保持、$1\times10^{-5}\to2\times10^{-6}$。
- SFT：batch=64，max_len=9000，5 epochs；LM+projector LR 峰值 $2\times10^{-6}$，VE $1\times10^{-6}$。
- RLVR：batch=128，LR=$1\times10^{-6}$，3 epochs。

## 实验与结果
- **评测基准**（统一协议，禁用 thinking，max_new_tokens=4096）：MMMU Medical-test、MMMU-Pro Medical-10、MedXpertQA-MM、PMC-VQA-clean、OmniMedVQA。
- **InfiMed2-4B（SFT+RLVR）**：MMMU Med. 72.43 / MMMU-Pro 58.39 / MedXQA 46.25 / PMC-VQA 65.80 / OmniMedVQA 90.79，**平均 66.73%**，超 Qwen3.5-4B 6.54 pts、超 Qwen3.5-9B 2.02 pts。
- **InfiMed2-27B（SFT）**：77.85 / 70.98 / 59.40 / 67.50 / 92.93，**平均 73.72%**，居所有评测开放权重模型之首，距 GPT-5（73.94）差 0.22 pts、距 Claude-Opus-4.7（74.25）差 0.53 pts。
- **CPT 最佳 checkpoint**：4B 在 30B warmup + 2.5B decay 最优（64.80）；27B 在 24B warmup + 2.5B decay 最优（73.72），说明更长 CPT 不一定更优（非单调）。
- **RLVR 增益**：4B 全榜提升，最大增幅在 MedXQA（+3.55 pts）与 PMC-VQA（+2.80 pts）。
- **响应目标消融**：Answer only 55.78 / Random response 57.97 / **Stability-aware 58.55**，证明稳定性路由的价值不仅来自"更长解释"，而来自"更可靠且信息丰富的解释"。

## 相关工作脉络
- **LLaVA-Med / HuatuoGPT-Vision / MedGemma**：侧重单一医学术域的视觉-语言对齐或指令微调，缺乏对多源证据的结构化整合与后期监督可信度验证。
- **Lingshu / Hulu-Med / MedXiaoHe**：同代通用医学 MLLM，但本文指出其未系统处理"源异构性→分阶段匹配"与"合成解释的稳定性路由"两个数据设计问题。
- **PMC-15M / BIOMEDICA**：大规模生物医学图-文对，但保留为孤立 pair，丢失 figure↔context 链接；本文 PMC-InterCPT 重构上下文关联。
- **Self-consistency / Semantic uncertainty**：提供重复采样评估可靠性的思想基础，本文将其与答案硬约束 + 熵排序结合形成三档路由。
- **ReasonMed / Synthesis-based VQA**：依赖单一教师生成解释，本文引入多教师一致性校验与正确性门控，降低不可靠合成数据的污染。
- **GMAI-VL / GMAI-VL-5.5M**：面向医疗的大规模数据与模型，但本文在"阶段感知数据配比 + RLVR 难度感知"上的组合策略构成差异。

## 局限性与未来方向
- 评测仍集中于选择题 VQA 与多模态理解，**开放-ended 临床沟通与专家级解释质量评估**尚未覆盖。
- 医学语料虽经清洗但仍可能残留**事实错误、人口/地域偏差与敏感内容**，模型仍存在幻觉与不安全建议风险，当前声明明确限定于研究用途。
- 稳定性路由依赖 Qwen3-VL-32B 与 Gemini 3.1 Pro 作为教师，**成本与可复现性受限**；对低资源场景推广存在障碍。
- CPT 延长在某些任务上呈非单调收益，表明"最佳 checkpoint 随下游能力剖面变化"，需更细粒度的早停或多目标平衡策略。
- RLVR 仅在 4B 模型上应用，27B 版本的 RLVR 增益与可扩展性未验证。

## 研究启发与可借鉴点
1. **阶段感知数据配比**：CPT 早期重多样性 + 晚期重质量的动态 mixture 思路可迁移至其他垂直领域（如法律、金融）的多模态基础模型训练，避免"一锅炖"静态配比。
2. **稳定性路由 + 掩码重建**：将"一致性计数（hard constraint）+ 熵排序（soft ranking）"解耦处理的设计，适用于任何需要"长解释但必须答案一致"的指令微调场景（如科学 QA、代码生成）。
3. **RLVR 难度感知采样**：基于 pass@K 估计难度并优先训练中间难度样本，这一策略在数学推理、代码合成等领域同样有效，可作为通用 post-training 优化范式。
4. **源特定处理流水线**：医学电子书按学科/体裁重采样、科学文献保留多图同上下文、网络数据三级质量分类 + 图文恢复——模块化清洗策略可直接移植至其他多源语料整合。
5. **视觉 replay 防过度专业化**：在 CPT 中保留 5B token 通用多模态数据作为正则项，对任何专精领域模型都能缓解灾难性遗忘，值得在参数预算有限时复用。

## 关键术语表
- **Continued Pretraining (CPT)**：在通用多模态底座上继续预训练以注入领域知识，本文分三阶段（视觉适配、广泛学习、学习率衰减）。
- **Stability-aware Response Regeneration**：通过重复采样估计教师答案一致性 $S_i$，据此路由至掩码重建、模型升级或丢弃，保证 SFT 监督的答案正确性与解释丰富性。
- **Answer-masked Reconstruction**：将教师正确响应的显式最终答案遮蔽后，要求学生模型基于图文 + 掩码解释重新生成，检验解释是否仍蕴含答案信息。
- **RLVR (Reinforcement Learning with Verifiable Rewards)**：以可验证格式合规 + 答案准确为奖励信号、temperature=1 采样 rollouts 进行 RL 优化，避免传统 RLHF 对偏好模型的依赖。
- **Pass@K 难度估计**：同一 prompt 采样 K 次，以正确比例估算样本难度，本文取 K=8 用于筛选 3–5 正确的中等难度样本优先训练。
- **MMCU / MedXpertQA-MM / OmniMedVQA**：本文使用的五大全开源医学多模态评测基准，分别覆盖多学科理解、专家级多步推理与跨模态临床问答。
- **Source-specific Processing**：针对医学电子书、科学文献、网络数据分别设计的去重/分类/质量过滤/图文恢复流水线，以解决异构来源的噪声与分布偏差。
- **Multi-modal Projector**：连接冻结视觉编码器与语言模型的投影层，本文在 CPT 第一阶段与 VE 一同训练。

## 可复现要素
- **数据集**：CPT 语料主要来自 PMC-InterCPT、BIOMEDICA、OmniCorpus-CC、OBELICS、MINT-1T、LLaVA-OneVision-1.5；SFT 语料来自 PMC-VQA、MedXpertQA、OmniMedVQA、RadImageNet 等公开数据及内部整理。论文声明接受后将发布训练模型、允许再分发部分数据与元数据，受限第三方数据提供出处与预处理脚本。
- **代码/权重**：模型权重计划开源（论文未提供具体仓库链接）；部分组件（MinerU、PMC-InterCPT 构建脚本）依赖外部项目。
- **关键超参**：CPT global batch=256、max_len 8192/12288、VE LR $2\times10^{-6}\to2\times10^{-7}$、LM LR $1\times10^{-5}$；SFT batch=64、max_len=9000、5 epochs、LM LR $2\times10^{-6}$；RLVR batch=128、3 epochs、LR $1\times10^{-6}$、rollouts=8、format:accuracy=1:9。
- **硬件**：论文未明确列出算力配置与训练时长。
