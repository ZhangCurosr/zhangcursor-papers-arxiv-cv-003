---
title: "TUE-Detector-A-Tool-Using-Expert-MLLM-Based-Detector-for-AI"
source: https://arxiv.org/pdf/2608.30704v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 19:20:27"
field: "AI生成内容检测与取证"
keywords: ["AI生成视频检测", "工具使用MLLM", "多模态大语言模型", "证据发现", "工具演化", "GRPO强化学习", "影响函数"]
innovations: ["首次训练通用MLLM为工具使用专家检测器进行AI生成视频检测", "提出基于影响函数的教师轨迹自适应加权策略", "设计用户经验引导的三阶段工具演化机制（选择-突变-探索）"]
benchmarks: ["ViF-Bench", "GenVideo"]
---

# 论文速读：TUE-Detector-A-Tool-Using-Expert-MLLM-Based-Detector-for-AI

## 一句话总结
本文提出 TUE-Detector，通过将通用多模态大语言模型（MLLM）训练为任务定制的"工具使用专家检测器"，以工具中介的证据发现与证据驱动推理为核心机制，可靠地检测 AI 生成视频中难以察觉的微小异常伪影。

## 研究问题与动机
- **核心问题**：AI 生成视频检测需要从视频中精准识别"细微但可测量的不自然伪影"（subtle-yet-measurable unnatural artifacts），如细微形变、纹理异常、光影不一致等，这些线索肉眼难辨但具有可测量性。
- **现有方法不足**：已有检测方法对显著伪影已较有效，但对多样化、跨类别/场景/生成器的细微伪影泛化能力有限；直接对通用 MLLM 做工具调用提示并不能保证模型理解何时、为何以及如何正确调用工具。
- **解决思路**：将工具调用视为"证据发现"问题，受人类分析师训练后使用合适工具检验不同视觉线索的启发，系统性地训练 MLLM 成为能自适应选择工具、收集证据、并基于证据推理检测的工具使用专家。

## 核心贡献（创新点）
1. **首个工具使用专家 MLLM 检测框架**：首次探索将通用 MLLM 训练为面向 AI 生成视频检测的任务定制化工具使用专家检测器，而非简单拼接预设工具集进行提示。
2. **基于影响函数的教师知识引导策略（Stage 1）**：提出 influence-based teacher-knowledge guidance，利用 API 可调用的强 MLLM 作为噪声教师生成参考轨迹，并引入基于影响函数（Ridge-Regression 近似）的轨迹加权机制，避免对所有教师轨迹一视同仁。
3. **用户经验引导的工具演化策略（Stage 2）**：提出交替优化中基于 MLLM 工具使用经验对工具集进行演化，通过选择（保留高泛用/高效工具、剪除双低工具）、突变（针对不同缺陷改进工具实现）和探索（交叉融合+头脑风暴生成新工具）三阶段操作持续更新工具集。
4. **SOTA 性能**：在 ViF-Bench 和 GenVideo 两大基准上均达到当前最优或接近饱和性能，ViF-Bench 平均准确率达 96.90%。

## 方法详解
**整体框架**：TUE-Detector 采用渐进式两阶段训练。

**Stage 1（基础工具使用能力训练）**：
- 通过 MLLM 自动化流程启发式构建包含 13 个工具的初始工具集，每个工具暴露统一的 Python-callable 接口，后端封装预训练分割/跟踪/OCR/频域分析等模块，输出结构化证据。
- **参考轨迹构建**：使用 GPT-5.4 作为教师，对每个训练视频采样 16 帧后提示其生成完整工具增强推理轨迹（含工具调用、证据解释、最终判决），再以 binary 标签为过滤信号保留最终判决与 GT 一致的轨迹，得到参考数据集 $S_{\text{ref}}$。
- **Token 级掩码损失**：为让模型学习工具使用决策而非记忆工具输出，定义 mask 交叉熵损失 $\mathcal{L}_j(\vartheta) = -\sum_\eta \log p_\vartheta(\sigma_\eta | v_j, \hat{\mathcal{T}}, \sigma_{<\eta}) \cdot m_\eta$，其中 $m_\eta=1$ 对应主动推理/token，$m_\eta=0$ 对应工具返回结果。
- **影响函数轨迹加权**：为避免低质量参考轨迹的干扰，利用 RRInf 框架将影响分数估计转化为脊回归问题 $\widehat{\chi} = \arg\min_\xi \frac{1}{P_\vartheta}\|\Psi\xi - b\|_2^2 + \rho\|\xi\|_2^2$，将影响分数归一化为训练权重 $w_j$，每 $\gamma$ 个 epoch 重新计算并更新权重，实现自适应重加权训练。

**Stage 2（模型与工具集交替优化）**：
- **模型优化（RL）**：采用 GRPO 对 Stage 1 训练好的模型进行强化学习优化，奖励函数 $\mathcal{R} = \mathcal{R}_{\text{result}} + \mathcal{R}_{\text{trajectory}}$，其中 $\mathcal{R}_{\text{result}}$ 为最终判决是否正确（二元），$\mathcal{R}_{\text{trajectory}}$ 包含格式反馈和锚点锚定信号（奖励纠正 Stage 1 错误、惩罚回归）。
- **工具演化（工具优化）**：基于收集的工具使用轨迹统计每个工具的两个指标——**使用广度** $U(t)$（被调用频率）和**证据有效性** $S(t)$（调用后判决正确的比例），将工具划分为四象限，分别执行：①选择：保留高-高工具，剪除低-低工具；②突变：对高泛用低有效工具进行针对性改进，对低泛用高有效工具推广泛化；③探索：通过交叉（融合互补工具）和头脑风暴（从失败样本中提取缺失能力）生成新工具候选，经 trial-execution 验证后纳入工具集。

**测试阶段**：模型自适应调用工具、解释证据、推理并产出最终判决。

## 实验与结果
- **数据集**：ViF-Bench（约 5000 假视频/5000 真视频，覆盖 19 个生成器）和 GenVideo（many-to-many zero-shot，覆盖 10 个测试生成器）。训练数据为 ViF-CoT-4K（4034 条样本）和 GenVideo 训练集。
- **评估指标**：ViF-Bench 报告 ACC/R/F1；GenVideo 报告 R/F1。
- **主要结果**：
  - **ViF-Bench**：TUE-Detector 平均准确率 **96.90%**、Recall **94.98%**、F1 **96.77%**，超越 Skyra（90.27% F1）、VidGuard-R1（87.25% F1）、BusterX++（21.94% F1）等所有对比方法，在 Sora-2 等强生成器上亦表现强劲。
  - **GenVideo**：R=1.00、F1=1.00，达到近饱和性能。
- **最强结果**：ViF-Bench 平均 F1=96.77%，相对次优方法 Skyra 提升约 **6.50 个百分点**。
- **推理速度**：每段视频约 **0.5 秒**（8×NVIDIA H200 GPU）。

## 相关工作脉络
- **Skyra**（CVPR 2026）：训练专用 MLLM 识别肉眼可感知视觉伪影并给出证据化解释，但未系统引入工具调用机制进行可量化证据发现；TUE-Detector 通过工具中介将分析精确到可测量数值信号。
- **VidGuard-R1**（ICLR 2026）：基于 GRPO+RL 的 MLLM 检测框架，强调时空/物理合理性；本文与其同为 MLLM+RL 方向，但核心差异在于本文引入了工具集演化（model-tool coevolution），使工具能力随训练持续进化。
- **BusterX/BusterX++**（2025）：将检测建模为视觉推理任务并用 RL 训练 MLLM；本文在同样 MLLM 范式中进一步引入工具使用与证据发现机制。
- **IVY-FAKE**（2025）：统一图像/视频 AIGC 检测的可解释框架；本文聚焦视频细分任务，强调通过工具调用提取结构化证据。
- **Tool-use for LLM**（ToolLLM、Toolformer、Tool-Star 等）：通用 MLLM 工具调用研究多为开放领域问答或数学推理；本文首次将工具使用能力训练聚焦于 AI 生成视频检测这一特定取证任务。
- **EvoGuard**（2026）：面向 AI 生成图像的 agentic RL 工具演化框架；本文将其思想拓展至视频检测领域，并设计了适配视频特性（时序/空域）的工具演化算子。

## 局限性与未来方向
- 当前评估仅针对全 AI 生成文本到视频/图像到视频内容，未涉及换脸类检测（FaceForensics++ 等），可拓展方向为加入领域特定取证工具。
- 初始工具集由启发式方式构建，虽经第二阶段演化改进，但仍依赖教师模型的生成质量与随机性；工具演化过程中的 trial-execution 可能遗漏某些边缘失败案例。
- 训练需 API 可调用的强 MLLM（GPT-5.4/Claude/Gemini）作为离线教师，推理时无调用，但训练阶段涉及较高 API 成本（约 $260）。
- 工具演化中跨工具交叉和头脑风暴的自动生成质量依赖进化模型能力，缺乏对生成新工具形式正确性与语义合理性的严格保证。

## 研究启发与可借鉴点
1. **影响函数轨迹加权（RRInf 框架迁移）**：利用影响函数对教师生成的参考轨迹进行质量评估并自适应重加权，可有效缓解噪声参考数据的负面影响，该思路可迁移至其他基于教师蒸馏的训练场景。
2. **工具集演化的三阶段策略（选择-突变-探索）**：将工具优化类比为生物演化过程，通过使用广度与有效性双指标对工具诊断并执行差异化演化操作，为多工具系统的持续进化提供了可复用的方法论。
3. **Token 级掩码训练策略**：在工具辅助推理任务中，仅对主动推理 token 计算损失、屏蔽工具输出 token，防止模型"记住"工具输出而非学习推理过程，适用于类似 agentic 系统的训练。
4. **交替优化范式（Model-Tool Coevolution）**：将模型参数优化与离散工具集演化解耦为交替步骤，避免了联合优化的非凸困难，此范式可推广至其他需要协同优化模型与外部工具的 task。
5. **可结合本团队方向**：将工具使用专家范式迁移至 AI 生成图像检测、音频伪造检测或多模态深度伪造联合检测任务，尤其适合需要精细化证据定位的解释性检测场景。

## 关键术语表
- **TUE-Detector**：Tool-Using Expert MLLM-Based Detector for AI-Generated Videos，本文提出的核心框架，将通用 MLLM 训练为工具使用专家检测器。
- **Subtle-yet-measurable unnatural artifacts**：细微但可测量的不自然伪影，指 AI 生成视频中肉眼难辨但在数值尺度上可量化的异常线索（如光影不一致、形变异常等）。
- **Influence-based teacher-knowledge guidance**：基于影响函数的教师知识引导策略，利用影响函数估计教师轨迹对验证损失的影响来自适应加权，指导模型学习高质量参考轨迹。
- **RRInf（Ridge-Regression Influence Function）**：将影响函数估计转化为脊回归问题的快速近似方法，避免显式计算大模型 Hessian 逆矩阵。
- **User-experience-guided tool evolution**：用户经验引导的工具演化策略，通过工具使用广度与证据有效性两个维度诊断工具，执行选择/突变/探索操作持续优化工具集。
- **GRPO（Group Relative Policy Optimization）**：DeepSeekMath 提出的 RL 训练算法，本文用于 Stage 2 中优化 MLLM 的工具使用策略。
- **Token-level masking**：在 SFT 训练中屏蔽工具输出 token 的交叉熵损失，使模型仅学习推理和工具调用决策而非复制工具结果。
- **Many-to-many zero-shot protocol**：GenVideo 数据集的评测协议，每个测试生成器的假视频与共享真实视频池配对评估，衡量跨生成器泛化能力。

## 可复现要素
- **数据集**：ViF-Bench 和 ViF-CoT-4K（公开）、GenVideo（公开）；论文声明遵循各自许可协议。
- **代码**：开源于 github.com/Louis-YW/TUE。
- **模型权重**：基于 Qwen2.5-VL-7B-Instruct，LoRA rank r=32、α=64，仅训练语言侧投影层参数。
- **关键超参**：Stage 1 学习率从 5e-5 衰减至 2e-5；Stage 2 GRPO batch size=4、rollouts per prompt=4、学习率 5e-7、KL 系数 0.10；工具演化间隔 $\Delta_{\text{evo}}=20$；影响函数重评估周期 $\gamma=3$；阻尼系数 $\kappa=10$。
- **教师模型**：主要使用 GPT-5.4（API 调用），训练时不暴露 GT 标签。
