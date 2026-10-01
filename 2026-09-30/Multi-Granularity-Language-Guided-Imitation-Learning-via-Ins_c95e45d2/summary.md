---
title: "Multi-Granularity-Language-Guided-Imitation-Learning-via-Ins"
source: https://arxiv.org/pdf/2609.37135v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 21:45:27"
field: "语言引导机器人操作学习"
keywords: ["多粒度语言引导", "模仿学习", "指令分解", "扩散策略", "子任务感知", "LIBERO"]
innovations: ["将整体任务指令分解为细粒度子任务指令并提供多粒度语言监督", "设计混合语言指令训练策略动态组合粗细条件", "引入子任务感知损失增强策略对执行阶段的识别能力"]
benchmarks: ["LIBERO-Goal", "LIBERO-Object", "LIBERO-Spatial", "LIBERO-10"]
---

# 论文速读：Multi-Granularity-Language-Guided-Imitation-Learning-via-Ins

## 一句话总结
本文提出 MuGIL 框架，通过将整体任务指令分解为细粒度子任务指令，并引入混合语言训练策略与子任务感知损失，在多阶段机器人操纵任务中提升了语言引导模仿学习的效率与成功率。

## 研究问题与动机
- 现有语言引导策略学习方法通常使用单一整体任务描述来指导整个演示轨迹，在多阶段操作中难以区分不同子阶段所需的行为。
- 粗糙的语言监督无法提供明确的中间步骤指导，导致策略在长视野或多阶段任务中学习模糊或错位的动作。
- 需要一种机制使策略既能把握全局任务目标，又能感知当前所处的子任务阶段，从而提供更丰富的学习信号。

## 核心贡献（创新点）
- **多粒度语言指导框架**：将单一任务级指令分解为细粒度子任务级指令，使策略能接收全局与局部互补的语言监督；与仅用整体指令的现有方法本质不同。
- **混合语言指令训练策略**：在训练时对每个时间步动态采样粗粒度、细粒度或两者组合的语言条件；与固定使用单一语言条件的策略训练方式本质不同。
- **子任务感知损失（SAL）**：通过辅助分类头鼓励策略表征编码当前子任务阶段信息；与仅依赖动作预测损失的训练目标本质不同。

## 方法详解
- **细粒度语言标注流程**：基于机器人状态（关节速度接近0且夹爪状态稳定）检测关键帧，使用 GPT-4o 生成描述局部子目标的细粒度指令，经人工校验后分配至对应时间区间。
- **混合语言指令策略**：定义三种语言模式——粗粒度（$\ell^c$）、细粒度（$\ell^f$）、两者组合（$\ell^b = \text{Concat}(\ell^c, \ell^f)$），按分类分布采样（实验中 $p_c:p_f:p_b = 0.6:0.1:0.3$）。
- **语言引导策略架构**：以 MDT 为基座，采用编码器-解码器 Transformer，视觉观察经 ResNet-18 编码，语言指令经冻结 CLIP 文本编码器编码。
- **损失函数设计**：总损失 $\mathcal{L}_{\text{MuGIL}} = \mathcal{L}_{\text{MDT}} + \lambda \mathcal{L}_{\text{SAL}}$，其中 $\mathcal{L}_{\text{MDT}} = \mathcal{L}_{\text{SM}} + \alpha \mathcal{L}_{\text{MGF}} + \beta \mathcal{L}_{\text{CLA}}$（$\alpha=\beta=0.1$），$\mathcal{L}_{\text{SAL}}$ 为子任务分类交叉熵（$\lambda=0.2$）。

## 实验与结果
- **数据集**：LIBERO 基准（包含 LIBERO-Goal、LIBERO-Object、LIBERO-Spatial、LIBERO-10 四个任务集），每任务 20 条演示数据用于训练。
- **评估基线**：MDT（Multimodal Diffusion Transformer）。
- **主要结果**：MuGIL（含混合语言 + SAL）平均成功率从 MDT 的 68.13% 提升至 77.25%；在 LIBERO-Goal 达 77.5%，LIBERO-10 提升最显著（从 48.0% 升至 63.5%）。
- **最强结果**：LIBERO-Goal 成功率 77.5%，LIBERO-Object 94.5%。

## 相关工作脉络
- **Distill-Down / PlayFusion / MDT**：单指令语言引导扩散策略学习；本文进一步引入多粒度语言监督。
- **PALO**：利用 VLM 将高层任务分解为可复用子任务，侧重快速适应；本文聚焦于多粒度语言对模仿学习过程的影响。
- **CLAP**：分解指令用于引导语言对齐的 3D 关键点预测；本文关注语言粒度对动作预测策略的直接指导作用。
- **RACER / STEER**：使用丰富语言进行失败恢复或密集重标注；本文强调粗/细粒度语言的组合与采样策略。
- **LIBERO 基准**：常用于评估 lifelong robot learning；本文使用该基准验证多粒度语言指导的有效性。

## 局限性与未来方向
- 在长视野任务（如 LIBERO-10）中，混合语言可能引入歧义，需依赖 SAL 进行阶段感知补偿。
- 当前细粒度指令依赖人工校验与离线分解，未来可探索自动化、在线的子任务切换机制。
- 仅评估了固定任务集，未在更开放或动态环境中验证泛化性。

## 研究启发与可借鉴点
- **关键帧检测策略**：基于机器人状态（关节速度、夹爪状态）推断子任务边界，可为无细粒度标注的数据集提供低成本分段方案。
- **混合采样训练机制**：在条件输入中动态混合不同粒度的监督信号，有助于提升策略对上下文变化的鲁棒性，可迁移至多模态条件学习。
- **辅助分类损失设计**：通过表征分类任务增强模型对执行阶段的内在感知，适用于需要阶段控制或层次决策的长程任务。
- **实验评估维度拓展**：引入“阶段成功数”指标细化分析，可为机器人策略的阶段性性能诊断提供参考。

## 关键术语表
- **MuGIL**：Multi-Granularity Language-guided Imitation Learning，多粒度语言引导模仿学习框架。
- **SAL（Subtask-Aware Loss）**：子任务感知损失，通过分类头鼓励策略表征编码当前子任务阶段信息。
- **LIBERO**：Benchmarking knowledge transfer for lifelong robot learning 基准，包含多类操纵任务套件。
- **MDT**：Multimodal Diffusion Transformer，多模态扩散 Transformer 策略架构。
- **细粒度语言指令**：描述单个子任务局部目标的自然语言（如“移动夹爪靠近牛奶”）。
- **混合语言模式**：训练时在粗粒度、细粒度及两者组合三种语言条件间采样。

## 可复现要素
- **数据集**：LIBERO（公开）
- **代码/权重**：论文未提及
- **关键超参**：epochs=50, batch size=64, action chunk size=10, hidden dimension=512, learning rate=1e-4, $\lambda=0.2$, $\alpha=\beta=0.1$, 采样比例 $p_c:p_f:p_b=0.6:0.1:0.3$
