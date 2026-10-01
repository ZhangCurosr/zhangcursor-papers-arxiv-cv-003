---
title: "Multi-Granularity-Language-Guided-Imitation-Learning-via-Ins"
source: https://arxiv.org/pdf/2609.37135v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 21:45:25"
field: "语言引导的机器人模仿学习"
keywords: ["language-guided imitation learning", "multi-granularity language", "diffusion policy", "instruction decomposition", "robot manipulation", "LIBERO benchmark"]
innovations: ["提出多粒度语言指导框架MuGIL，联合利用任务级与子任务级语言指令训练策略", "设计混合语言指令采样策略与子任务感知损失（SAL），提升多阶段任务成功率"]
benchmarks: ["LIBERO-Goal", "LIBERO-Object", "LIBERO-Spatial", "LIBERO-10"]
---

# 论文速读：Multi-Granularity-Language-Guided-Imitation-Learning-via-Instruction-Decomposition

## 一句话总结
论文提出 **MuGIL**（Multi-Granularity language guidance framework for Imitation Learning），通过将整体任务描述分解为细粒度的子任务级语言指令，使策略在训练时接收多粒度语言监督，从而提升多阶段操作任务的学习效率与成功率。

## 研究问题与动机
1. 现有语言指导策略学习方法仅用单一整体指令（如"拿起物体放入容器"）指导整个轨迹，无法区分不同执行阶段所需的子目标。
2. 多阶段任务中，相同语言条件下策略需学习显著不同的行为，导致行为模糊或对不齐，尤其影响长程任务。
3. 大多数数据集仅提供任务级语言标注，中间执行阶段缺乏显式语言监督，限制了策略对过程状态的理解。

## 核心贡献（创新点）
1. **多粒度语言指导框架 MuGIL**：首次系统性地将任务级指令分解为子任务级指令，并在模仿学习中联合利用两种粒度。
2. **混合语言指令策略**：训练时每 timestep 按类别分布采样粗粒度（coarse）、细粒度（fine）或两者组合（both）的语言条件，使策略能从多形式监督中获益。
3. **子任务感知损失（SAL）**：新增辅助分类头，强制编码器表征编码当前子任务阶段信息，提升策略的阶段感知能力，对长程任务尤为有效。
4. **细粒度语言标注流程**：基于机器人状态检测关键帧，利用 GPT-4o 生成子任务描述并经人工校验，构建了多粒度语言标注数据集。

## 方法详解
1. **细粒度语言标注**：基于关节速度与夹爪状态检测关键帧，用 GPT-4o 生成如"Move the gripper towards the object"的细粒度指令，人工校验后按时间区间分配给各 timesteps。
2. **混合语言采样策略**：对每 timestep 按 $m_t \sim \text{Categorical}(p_c, p_f, p_b)$ 采样，默认比例 $p_c:p_f:p_b = 0.6:0.1:0.3$；both 模式为 coarse 与 fine 拼接。
3. **扩散策略架构**：基于 MDT（Multimodal Diffusion Transformer），编码器用 CLIP text encoder 处理语言 token，ResNet-18 提取视觉 token；解码器为扩散 denoiser，训练目标为 Score Matching Loss：
   $\mathcal{L}_{\text{SM}} = \mathbb{E}[\alpha(\sigma_k)\|D_\theta(\bar{a}_i + \epsilon, o_i, \ell_i, \sigma_k) - \bar{a}_i\|_2^2]$
4. **辅助损失**：保留 MDT 原有的 MGF（Masked Generative Foresight）和 CLA（Contrastive Latent Alignment）；新增 SAL（Subtask-Aware Loss）：
   $\mathcal{L}_{\text{SAL}} = -\frac{1}{B}\sum_i \log \mathbf{p}_i(y_i)$，其中 $\mathbf{p}_i = \text{softmax}(h_{\text{SAL}}(\mathbf{z}_i))$，总损失 $\mathcal{L}_{\text{MuGIL}} = \mathcal{L}_{\text{MDT}} + \lambda \mathcal{L}_{\text{SAL}}$（$\lambda=0.2$）。

## 实验与结果
- **数据集**：LIBERO 基准（LIBERO-Goal、LIBERO-Object、LIBERO-Spatial、LIBERO-10），每任务 20 条演示训练，20 次 rollout 评估。
- **基线**：MDT（Multimodal Diffusion Transformer）。
- **主结果**：MDT 平均成功率 68.13%；仅加混合语言策略提升至 72.50%；再加 SAL 提升至 **77.25%**。
- **最强提升**：LIBERO-10 从 48.0% → 46.5%（仅混合语言）→ **63.5%**（+SAL），提升 17.0 个百分点，证明 SAL 对长程任务的关键作用。
- **消融**：语义细粒度语言（77.25%）优于非语义步骤标识（70.50%）；最佳采样比例为 0.6:0.1:0.3。

## 相关工作脉络
1. **Distill-Down / PlayFusion / MDT**：单粒度整体语言指导的扩散策略工作，本文在此基础上引入多粒度监督。
2. **PALO**：用 VLM 将高层任务分解为可重用子任务以实现少样本适应；本文聚焦于训练时的多粒度语言指导而非适应。
3. **CLAP**：将指令分解为逐步指令并引导 3D 关键点预测；本文聚焦于策略直接学习而非中间表示预测。
4. **RACER / STEER**：用丰富语言标注辅助失败恢复或密集语言 grounding；本文强调粗/细粒度的互补联合监督。
5. **Diffusion Policy**：单任务扩散策略基础；本文扩展至多任务语言条件设定。

## 局限性与未来方向
1. 关键帧检测依赖速度和夹爪状态阈值，可能遗漏平滑过渡阶段；细粒度指令由 GPT-4o 生成，依赖大模型且需人工校验。
2. 仅在仿真环境（LIBERO）验证，未在真实机器人上测试泛化性。
3. 未来方向：将细粒度指令用作执行时子任务切换的显式机制，实现在线阶段自动判断与转移。

## 研究启发与可借鉴点
1. **混合多粒度条件采样**思路可迁移至其他多模态条件学习（如视觉+语言+深度），通过动态采样丰富训练信号。
2. **子任务感知辅助损失**设计简洁有效，可借鉴到任何具有隐式阶段结构的多阶段决策任务（如导航、装配）。
3. **基于状态变化的关键帧检测**作为低成本自动化标注方案，可推广至其他机器人数据集的细粒度标注流程。
4. **细粒度语义 vs 非语义**的消融结论表明：语言指导的价值在于语义内容而非仅时间位置指示，对后续语言 grounding 研究有参考意义。

## 关键术语表
**MuGIL**：Multi-Granularity language guidance framework for Imitation Learning，本文提出的多粒度语言指导模仿学习框架。
**混合语言指令策略**：训练时每 timestep 按概率采样 coarse/fine/both 三种语言模式的策略。
**子任务感知损失（SAL）**：辅助 cross-entropy 损失，强制编码器表征识别当前子任务阶段。
**LIBERO**：用于评估终身机器人学习中知识迁移能力的基准测试平台，包含多任务套件。
**MDT**：Multimodal Diffusion Transformer，多模态扩散 Transformer 策略架构，本文以其为基线。
**Score Matching Loss**：扩散策略的核心训练目标，通过去噪噪声动作序列优化策略。
**关键帧检测**：基于关节速度趋零且夹爪状态不变的条件，识别子任务边界的时间点。

## 可复现要素
- 数据集：LIBERO（公开），每任务 20 条演示。
- 代码/权重：论文未明确声明开源。
- 关键超参：epochs=50，batch size=64，learning rate=1e-4，encoder layers=4，decoder layers=6，hidden dim=512，attention heads=8，action chunk size=10，$\lambda=0.2$，$\alpha=\beta=0.1$，采样比例 $p_c:p_f:p_b=0.6:0.1:0.3$，S=4 个子任务类别。
- 硬件：单张 NVIDIA GeForce RTX 4080 GPU。
