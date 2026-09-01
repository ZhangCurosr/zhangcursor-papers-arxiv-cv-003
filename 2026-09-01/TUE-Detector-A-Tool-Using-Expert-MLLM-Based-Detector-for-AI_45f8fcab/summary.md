---
title: "TUE-Detector-A-Tool-Using-Expert-MLLM-Based-Detector-for-AI"
source: https://arxiv.org/pdf/2608.30704v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 16:38:12"
field: "AI生成视频检测与多模态取证"
keywords: ["AI-generated video detection", "MLLM tool use", "influence function", "tool evolution", "evidence-grounded reasoning", "GRPO reinforcement learning", "forensic video analysis"]
innovations: ["首个将通用MLLM训练为面向AI视频检测的工具使用专家检测器", "提出基于影响函数的教师轨迹动态加权策略以解决细粒度过程监督数据稀缺问题", "设计使用广度与证据有效性双维度评估及类遗传操作的用户经验引导工具演化机制"]
benchmarks: ["ViF-Bench", "GenVideo"]
---

# 论文速读：TUE-Detector-A-Tool-Using-Expert-MLLM-Based-Detector-for-AI

## 一句话总结
本文提出 TUE-Detector，首个将通用多模态大语言模型（MLLM）训练为"工具使用专家"检测器的 AI 生成视频检测框架；该框架通过两阶段渐进式训练，使模型学会自适应调用外部分析工具、收集不自然伪迹证据并进行基于证据的推理，从而在 ViF-Bench 和 GenVideo 上达到最优性能。

## 研究问题与动机
- **核心问题**：随着 AI 视频生成模型质量快速提升，明显的生成伪迹（如形变、光影错位）已大幅减少，当前真实/伪造视频的差异主要体现在"微妙的、可测量的不自然伪迹"（subtle-yet-measurable unnatural artifacts），如细微形状变化、纹理扰动、局部光照不一致等，难以直接肉眼识别。
- **现有方法不足**：现有 MLLM 基方法（如 Skyra、VidGuard-R1、BusterX++）主要依赖模型自身视觉知识进行判别，缺乏系统化"证据发现"机制；而仅靠提示词让通用 MLLM 调用工具，模型并不能真正理解何时/为何调用工具以及如何将工具输出融入推理链，导致检测可靠性受限。
- **类比启发**：人类专家在面对微妙视觉线索时，并非仅依赖直觉，而是经过训练学会在合适时机调用合适的分析工具（如轮廓测量、纹理分析、光照探测）采集具体证据，再基于证据做出判断——这一模式可迁移至 AI 视频取证。
- **技术挑战**：① 适合本任务的高质量工具难以预先完备设计，因为 AI 视频伪迹跨类别、场景和生成模型高度多样化；② 即使具备候选工具集，教 MLLM 自适应决定何时/何地/如何调用并整合工具输出仍然困难，固定模式无法覆盖视频的时空多样性。

## 核心贡献（创新点）
1. **首次将通用 MLLM 训练为面向 AI 视频检测的"工具使用专家"检测器**：提出 TUE-Detector 框架，使模型学会自适应选择工具、收集具体证据并完成基于证据的最终判决，区别于仅靠端到端特征聚合的既有 MLLM 检测方法。
2. **基于影响函数的教师知识引导策略（Influence-based Teacher-knowledge Guidance）**：利用 API 可访问的 MLLM 作为"噪声教师"生成候选推理轨迹，通过结果级过滤获得参考集，再基于 influence function 估计每条轨迹对验证集的损失贡献，动态加权引导学生模型学习高质量的工具使用行为，解决了细粒度过程监督数据稀缺的核心难题。
3. **用户经验引导的工具演化策略（User-experience-guided Tool Evolution）**：提出基于"使用广度 U(t)"和"证据有效性 S(t)"的双维度工具评估，并借助选择、突变、交叉/头脑风暴的类进化操作迭代更新工具集，实现模型与工具集的交替协同优化。
4. **两阶段渐进式训练设计**：Stage 1 固定初始启发式工具集训练基础工具使用能力；Stage 2 通过 RL（GRPO）交替优化模型策略与工具集，使两者逐步适配，显著提升泛化性。
5. **全面的实验验证与消融**：在 ViF-Bench（19 个生成器）和 GenVideo（10 个生成器，many-to-many 零样本）上均达到 SOTA；额外 LLM-judge 评估显示其推理轨迹质量显著优于 Skyra-RL 和 VidGuard-R1。

## 方法详解
**整体框架**：TUE-Detector 遵循"工具介导的证据发现"范式，视频输入经均匀采样 16 帧后送入 MLLM，模型在推理链中自适应调用工具，工具返回结构化 XML 证据，模型解释证据并迭代推理直至输出 `<answer>Real|Fake</answer>`。

**Stage 1：基于影响函数的教师知识引导**
- **初始启发式工具集**：由 MLLM 自动构建，含 13 个工具（12 个 code 工具 + toolbox_guide），覆盖光照不一致、形状异常、文本畸变、前景/背景漂移、halo 伪影等类型，每个工具暴露统一 Python 可调接口，封装预训练分割/跟踪/OCR/频谱等后端。
- **参考轨迹生成**：对每训练视频，用 GPT-5.4 作为教师采样生成完整工具增强 CoT 轨迹（工具调用、观察、解释、判决），仅以二值标签做结果级过滤（保留最终判决与 GT 一致的轨迹），形成初始参考集 S_ref。教师 prompt 中不暴露 GT 标签，避免捷径依赖。
- **掩码 token 级 SFT 损失**：借鉴 Tool-Star，对轨迹中工具返回 token 设 mask（label=-100），仅在模型主动生成的 reasoning 和 tool-call token 上计算交叉熵：
  $$\mathcal{L}_j(\vartheta) = -\sum_\eta \log p_\vartheta(\sigma_\eta | v_j, \hat{\mathcal{T}}, \sigma_{<\eta}) \cdot m_\eta$$
- **影响函数轨迹加权**：将标准 influence function 通过 RRInf 的岭回归reformulation近似，避免显式求 Hessian 逆：
  $$\widehat{\chi} = \arg\min_\xi \frac{1}{P_\vartheta}\|\Psi\xi - b\|_2^2 + \rho\|\xi\|_2^2$$
  其中 Ψ 为批次轨迹梯度矩阵，b 为验证集梯度聚合。将影响分数转换为归一化权重 w_j，每隔 γ=3 个 epoch 重算并重加权训练。
- **关键超参**：LoRA r=32, α=64，LR 从 5e-5 衰减至 2e-5；influence 求解 lr=0.01，迭代 1000 次，κ=10。

**Stage 2：交替模型优化与工具演化**
- **模型优化（GRPO RL）**：全参数 GRPO 更新，batch_size=4，rollout=4，LR=5e-7，KL 系数 0.10。奖励函数 R = R_result + R_trajectory，其中 R_result 为判决正确性二元奖励；R_trajectory 包含格式奖励（硬规则 + 工具名漂移惩罚）和 anchor 奖励（鼓励相对于 Stage 1 检查点的改进，尤其奖励对原错误 Real 样本的翻转）。
- **工具演化（每 Δ_evo=20 步触发一次）**：
  - 评估：U(t) = 轨迹缓冲区中被调用的比例（广度）；S(t) = 调用该工具且判决正确的比例（有效性）。
  - 四分法分类：Q_HH（保留）、Q_LL（剪枝）、Q_HL（low-effectiveness → 定向突变 refine）、Q_LH（low-breadth  → 定向突变 generalize）。
  - 探索：Mutation（由教师 MLLM 修改 callable Python 函数，如 first_frame_jump_analyzer 改为归一化至同视频自身 baseline）、Crossover（如将 background_motion_analyzer 与 foreground_burst_analyzer 合并为 complex_scene_consistency_fuser）、Brainstorm（从失败案例生成新工具如 sam_region_stability、ocr_sam_filter）。
  - 准入：候选工具经 trial execution 验证可导入、无异常、输出 XML schema 后方可加入。

**测试阶段**：训练好的模型自适应调用演化后工具集，在推理链中整合多轮工具证据后输出最终判决；推理耗时约 0.5s/视频。

## 实验与结果
- **数据集**：ViF-Bench（≈5000 fake / 163 shared real，19 个 generator，含 Wan2.1/2.2、CogVideoX、HunyuanVideo、LTX-Video、Sora-2 等）和 GenVideo（many-to-many 零样本，10 test generators）。训练数据：ViF-CoT-4K（4034 条带 CoT 标注）和 GenVideo 训练集。
- **评估指标**：ViF-Bench 报告 Acc / Recall（R）/ F1；GenVideo 报告 R / F1。
- **主要结果**：
  - **ViF-Bench**：TUE-Detector 平均 Acc=96.90%、R=94.98%、F1=96.77%，全面超越 Skyra（Acc 91.02 / F1 90.27）、VidGuard-R1（Acc 88.96 / F1 87.25）、BusterX++（Avg Acc 56.90）；对 Sora-2、Pika-V2、Pixverse-V4-5 等最新生成器 Acc 达 99%+，real video Recall 达 100%。
  - **GenVideo**：TUE-Detector 在 10 个测试生成器上 R/F1 均接近或达到 1.00（Avg R=1.00、F1=1.00），远超 VidGuard-R1（Avg F1=0.96）、DeMamba-XCLIP（Avg F1=0.90）。
- **最强提升**：相比最强前作 Skyra，ViF-Bench 平均 Acc 提升 **+5.88 个百分点**（91.02→96.90），F1 提升 **+6.50 个百分点**（90.27→96.77）。
- **LLM-Judge**：TUE-Detector 推理轨迹质量得分 4.8/5，显著高于 Skyra-RL（2.0）和 VidGuard-R1（2.1）。
- **训练效率**：8×NVIDIA H200，总训练耗时 19.3 小时；推理约 0.5s/clip。

## 相关工作脉络
- **Skyra (CVPR 2026)**：训练专用 MLLM 识别人类可感知的视觉伪迹作为 grounded evidence；本文与其本质差异在于 Skyra 依赖模型内部表征直接识别伪迹，TUE-Detector 通过外部工具系统性地采集数值化证据并显式推理。
- **VidGuard-R1 (ICLR 2026)**：基于 GRPO + 专用 reward model 的 RL 框架，鼓励发现时序/物理伪迹；本文与其关联在于同样采用 RL 但额外引入工具演化机制，且_reward 同时涵盖轨迹结构与判决正确性_，不依赖额外 reward model。
- **BusterX / BusterX++**：将 AI 视频检测形式化为视觉推理任务并用 RL 训练 MLLM；本文与之差异在于 BusterX 依赖端到端 MLLM 推理而无显式工具证据收集过程。
- **IVY-FAKE / IVY**：统一图像/视频 AIGC 检测框架；本文聚焦视频并引入工具使用专家范式，提供更强可解释证据链。
- **Tool-LLM / LLaVA-Plus / ToolStar**：通用 LLM 工具使用研究；本文将其思想首次迁移至 AI 视频取证领域，并提出针对任务特化的两阶段训练与工具演化方案。
- **EvoGuard (2026)**：面向 AI 生成图像检测的 agentic RL 框架；本文将其 co-evolution 思想扩展至视频领域，并增加 usage breadth/evidential effectiveness 双维度评估与类遗传操作。

## 局限性与未来方向
- 当前评估聚焦于全 AI 生成（text-to-video / image-to-video）视频，尚未在 face-manipulation 数据集（如 FaceForensics++）上验证；作者指出可通过加入任务专用取证工具扩展。
- 工具集演化依赖离线教师 MLLM（GPT-5.4）生成参考轨迹和执行 mutation/crossover，API 成本约 $260（~$0.064/trajectory），虽可控但对资源受限场景仍有一定门槛。
- 工具输出的数值阈值（如 z > 4.0）在当前任务中表现良好，但跨域/跨分辨率场景下的鲁棒性仍需进一步验证。
- 未来可探索：① 扩展至 face swap / deepfake 检测；② 进一步降低教师模型依赖，探索自蒸馏或零样本工具演化；③ 将工具集扩展到更多模态（音频、时间序列一致性）。

## 研究启发与可借鉴点
- **影响函数用于轨迹质量筛选可迁移**：在缺乏细粒度过程标注的下游任务中，用 influence function 评估"训练样本/参考轨迹"的验证集贡献并自适应加权，是一种无需额外标注的高质量数据筛选策略，可复用于其他工具使用型 agent 训练。
- **"使用广度 × 证据有效性"双维度工具评估框架**：该二维分类（保留/剪枝/refine/generalize）及对应的突变策略具有普适性，可推广至其他需要动态工具集的 agentic 系统（如视觉问答、文档分析、机器人操作）。
- **两阶段渐进式训练设计**：先固定工具集训练基础能力、再交替优化模型与工具，避免了联合优化难题；这一"能力先行、协同演化"范式可借鉴于任何模型-工具耦合的下游任务。
- **Token-level mask 抑制工具输出过拟合**：对工具返回文本设置 label=-100 阻止模型死记硬背，只监督推理和工具调用决策，这一技巧对任何引入外部工具输出的 SFT 训练均有参考价值。
- **与团队方向结合机会**：可将 TUE-Detector 的"证据发现 + 工具演化"范式迁移至 AI 生成音频/多模态内容检测、图像篡改定位，或扩展到 open-domain visual verification 任务。

## 关键术语表
**Subtle-yet-measurable unnatural artifacts**：AI 视频中肉眼难以察觉但可通过数值工具精确量化的异常伪迹，如帧间亮度跳变、纹理统计偏移、非刚性形变等，是当前检测任务的关键区分信号。
**Tool-mediated evidence discovery**：将检测视为通过调用外部分析工具主动采集证据的推理过程，而非端到端直接预测，使模型输出具备可追溯的证据链。
**Influence-based teacher-knowledge guidance**：利用 influence function 估计每条教师生成轨迹对验证性能的影响，据此动态分配训练权重，筛选出真正有助于模型学习的高质量参考轨迹。
**Usage breadth U(t)**：在模型工具调用轨迹缓冲区中，工具 t 被调用的比例，衡量其适用广泛性。
**Evidential effectiveness S(t)**：在调用工具 t 的轨迹中，判决正确的比例，衡量其在实际使用中的判别有效性。
**User-experience-guided tool evolution**：基于模型历史调用经验对工具集进行 selection/mutation/crossover/brainstorming 类进化操作，使工具集随训练自适应演化。
**RRInf ridge-regression reformulation**：将 influence function 中的 Hessian 逆转化为岭回归问题，通过 push-through identity 将计算维度从 P_ϑ×P_ϑ 降至 J_b×J_b，使大规模 MLLM 上的影响估计可行。
**Anchor reward**：Stage 2 RL 中对 Stage 1 检查点预测结果作锚定，奖励模型纠正在 Stage 1 中错误判别的样本（尤其翻转错误 Real 预测），防止 RL 退化。

## 可复现要素
- **数据集**：ViF-Bench / ViF-CoT-4K（开源，CC-BY-NC 4.0）、GenVideo（开源）；代码仓库提供训练数据引用。
- **代码**：已开源，GitHub: github.com/Louis-YW/TUE。
- **模型权重**：基座使用 Qwen2.5-VL-7B-Instruct（开源）；最终模型权重随代码仓库提供。
- **关键超参**：LoRA r=32, α=64, dropout=0；Stage 1 LR 5e-5→2e-5；Stage 2 GRPO batch_size=4, rollout=4, LR=5e-7, KL=0.10；influence 求解 lr=0.01, 1000 iters, κ=10；γ=3（重算间隔），Δ_evo=20（演化间隔）。
- **教师模型**：GPT-5.4（API），也可替换为 Claude Opus 4.7 / Gemini 3.1 Pro（消融显示效果稳健）。
