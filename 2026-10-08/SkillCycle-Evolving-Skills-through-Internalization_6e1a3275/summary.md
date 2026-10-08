---
title: "SkillCycle-Evolving-Skills-through-Internalization"
source: https://arxiv.org/pdf/2610.09430v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 17:20:12"
field: "Language Agent Reinforcement Learning"
keywords: ["Skill Internalization", "Agent Reinforcement Learning", "Skill Distillation", "Policy-Skill Co-evolution", "On-policy Distillation", "Skill Routing"]
innovations: ["策略-技能协同演化框架：交替内部化与修订形成闭环反馈", "蒸馏反馈双重角色：OPD信号同时用于策略监督与规则诊断", "配对环境验证机制：规则级+全库级两段验证确保编辑安全提交"]
benchmarks: ["ALFWorld", "WebShop"]
---

# 论文速读：SkillCycle: Evolving Skills through Internalization

## 一句话总结
论文提出 **SkillCycle** 框架，通过技能内部化与规则修订的反馈循环，使语言智能体将外部技能逐步转化为策略参数能力，最终实现在推理时无需技能输入即可高效完成任务。该方法在 WebShop 基准上以 3B 模型达成 74.74% 成功率，超越现有 SOTA。

## 研究问题与动机
- **核心问题**：技能内部化将外部指导转化为策略参数后，原有技能可能变得冗余、误导或不充分，但现有方法缺乏对技能库本身的动态修订机制。
- **现有方法不足**：
  - Skill0 仅通过 on-policy 有用性指标逐步撤回技能上下文，未考虑规则内容或适用性的演化。
  - 静态技能库（如 Skill-SD）在训练期间固定不变，无法适应策略能力变化后的新需求。
  - 单次刷新（Refresh once）仅提供一次修订机会，无法持续迭代优化技能指导。
  - 蒸馏反馈（如 OPD）仅用于监督策略学习，未被用于诊断和修订技能本身。

## 核心贡献（创新点）
1. **策略-技能协同演化框架**：交替执行固定技能库下的策略学习与固定策略下的规则修订两个阶段，使技能库随策略能力演化持续更新。与 Skill0/Skill-SD 等单次内部化方法不同，本文建立闭环反馈而非单向蒸馏。
2. **状态感知、关键优先的路由机制**：路由器根据可观测事实选择适用技能，优先匹配关键事件规则，当存在冲突时选择弃权而非随意回退到通用工作流。与 OPID 等静态路由相比，本文的路由声明可在周期间修订。
3. **蒸馏反馈的双重角色设计**：将 token 级上下文差异（OPD）既用于策略学习的监督信号，又用于定位需检查的规则；结合交互结果验证候选编辑的有效性，实现"学习反馈→技能修订→更好学习"的闭环。
4. **配对环境验证机制**：候选规则编辑需通过规则级和全库级两段配对对比测试（在冻结策略下），只有同时通过局部利益和整体兼容性检验的编辑才被提交，避免孤立优化导致的冲突。
5. **无推理时技能输入的高性能**：在 WebShop 上以 3B 模型达到 74.74% 成功率/88.37 分，相对 OPID 提升 0.73%/3.96%，证明持续修订可使外部技能完全内化为策略能力。

## 方法详解
**整体框架（Algorithm 1）**：
从初始状态 $(\theta_0, B_0, R_0)$ 出发，交替执行 $C$ 个周期：
- **Phase 1（内部化）**：固定技能库 $H_{c-1}$，Student（无技能）采样轨迹，Teacher（共享权重、含技能上下文）重评分，计算 OPD 信号并训练策略。
- **Phase 2（修订）**：固定策略 $\theta_c$，基于训练证据诊断规则影响，生成 KEEP/REWRITE/DROP/ADD 提案，经配对验证后提交更新 $H_c$。

**关键组件**：

1. **技能内部化损失函数**（Eq. 4）：
$$
A_{t,j}^{\text{mix}} = \text{sg}(A_t^{\text{env}}) + \eta m_{t,j}^o \text{clip}(g_{t,j}, -c_g, c_g)
$$
$$
\mathscr{L} = \mathscr{L}_{\text{PPO}}(A^{\text{mix}}) + \lambda_o \mathscr{L}_{\text{OPD}} + \lambda_d \mathscr{L}_{\text{demo}} + \beta \mathscr{L}_{\text{ref}} - \alpha \mathscr{H}
$$
其中 $g_{t,j} = \text{sg}[\log\pi^T(y_{t,j}|h_t, k_t, y_{t,<j}) - \log\pi^S(y_{t,j}|h_t, y_{t,<j})]$ 为 signed OPD 信号，正表示技能提升该 token 概率，负表示抑制。

2. **规则影响度量**（Eq. 5）：
$$
d_{t,r} = L_c(t; k_t) - L_c(t; k_t \setminus r)
$$
在冻结策略下比较含/不含规则 $r$ 的 action token 对数概率差，隔离规则局部影响。

3. **修订提案筛选**（Eq. 26）：
- $S(q) = \text{HMean}[\ell_t \delta_t(q)]$：标签记录上的均值 outcome-aligned 效应（成功+1，无效-1）
- $A(q) = \text{HMean}[|\delta_t(q)|]$：所有记录上的平均效应幅度
- 通过条件：$A(q) > \varepsilon$ 且 $S(q) \geq 0$（有标签时）

4. **配对验证门控**（Eq. 6, 30-31）：
- 规则级：$\sum_f(G_f - L_f) > 0$ 且每族 $G_f - L_f \geq 0$
- 全库级（WebShop）：$\Delta N > 0 \land \Delta Q \geq 0 \land \bigwedge_a \Delta N_a \geq 0 \land e_{\text{API}}=0$

## 实验与结果
**数据集与基线**：
- **ALFWorld**：6 类 household task families，Qwen3-1.7B / Qwen2.5-3B
- **WebShop**：1000-product subset，Qwen2.5-3B
- 对比基线：GRPO, Skill-GRPO, OPID, Skill-SD, RLSD, SDAR, OPSD 等

**主要结果**：
| 模型 | 任务 | 指标 | SkillCycle | SOTA基线 | 提升 |
|------|------|------|-----------|---------|------|
| 3B | WebShop | SR | **74.74%** | OPID 74.20% | **+0.54pp** |
| 3B | WebShop | Score | **88.37** | OPID 85.00 | **+3.37** |
| 3B | ALFWorld | Macro SR | 82.53% (C4) | SDAR 84.20% | - |
| 1.7B | ALFWorld | Macro SR Δ(C0→C3) | **+8.11pp** | - | - |
| 3B | ALFWorld | Macro SR Δ(C0→C3) | **+15.12pp** | - | - |

**Cycle 3 消融（Table 2）**：
- ALFWorld Student SR：SKILLCYCLE 53.56% vs Static 48.61%（**+4.95pp**）vs Refresh once 52.30%（**+1.26pp**）
- WebShop Student SR：SKILLCYCLE 74.74% vs Static 63.28%（**+11.46pp**）vs Refresh once 72.92%（**+1.82pp**）
- Teacher-Sstudent gap 收缩：WebShop 从 3.90pp（单次刷新）降至 2.08pp（迭代修订），证明内部化更有效

**冻结策略修订实验（Table 3）**：
- ALFWorld：固定 1.7B 策略下，两轮修订使 macro SR 从 49.43% → 52.45%（+3.02pp）
- WebShop：固定 3B 策略下，首轮修订使 SR 从 57.55% → 68.23%（+10.68pp）

**API 模型路由实验（Figure 5）**：
- Skill + Router 在 DeepSeek V4-Pro 上 ALFWorld macro SR 提升 7.13pp，GLM 5.1 上 WebShop SR 提升 17.05pp
- 路由相比 Static General Skill 在所有 8 组比较中均有提升

## 相关工作脉络
1. **Skill0 (Lu et al., 2026b)**：通过 on-policy 有用性逐步撤回技能上下文，目标是推理时无技能依赖；但 skill 库固定，不修订规则内容。
2. **OPID (Yang et al., 2026b)**：提出 on-policy 技能蒸馏机制，提取层次化 hindsight skills 并路由关键步骤指导；本文在此基础上增加技能库修订闭环。
3. **Skill-SD (Wang et al., 2026)**：将轨迹总结为 skill 并在训练时作为 teacher 上下文；skill 库一次性生成，无后续修订。
4. **GRPO/GiGPO (Shao et al., 2024; Feng et al., 2025)**：基于组采样的策略优化方法，提供 outcome-only 基线；本文在此基础上叠加 skill 蒸馏与修订。
5. **Self-Refine / TextGrad**：迭代修订文本/提示的方法；本文的差异在于修订对象是持久训练监督源（技能库），且需环境配对验证而非仅靠模型反馈。
6. **SkillRL / Skill1 (Xia et al., 2026; Shi et al., 2026)**：递归演化分层技能库；本文聚焦于规则级修订与策略内部化的耦合反馈，而非技能生成/层级组织。

## 局限性与未来方向
- **收益转化机制不明确**：技能指导如何转化为持久策略能力尚未清晰建模，缺乏因果分析。
- **修订时机优化缺失**：未研究何时修订最有益，自适应修订调度有待探索。
- **单规则视角局限**：聚焦个体规则修订，未涉及技能组合、层级组织及跨任务迁移。
- **评估环境局限**：仅在 ALFWorld（家庭模拟）和 WebShop（购物）上验证，泛化到物理/实时商业场景需额外安全验证。
- **统计显著性未检验**：验收门控为经验性标准，非严格假设检验。

## 研究启发与可借鉴点
1. **蒸馏反馈的双重角色设计**：将 token 级差异同时用于策略监督和规则诊断，可迁移至其他需要持续迭代的 skill/memory 系统。
2. **配对验证的门控机制**：在冻结策略下进行 rule-level 和 whole-bank-level 两段验证，有效过滤有害编辑；此模式可应用于 prompt engineering、工具选择策略等可编辑组件的迭代优化。
3. **策略-技能协同演化的交替训练范式**：分离学习阶段与监督修订阶段，避免同时进行参数更新和元数据修改导致的耦合干扰，适用于任何"策略+外部知识源"的架构。
4. **状态感知、关键优先的路由设计**：在冲突时选择弃权（abstention）而非强制回退，可推广至多策略/多工具选择场景。
5. **与 GrGenie/GEPA 的结合机会**：本文的 skill revision 可结合 GEPA 的 trajectory reflection 机制，或用 TextGrad 式梯度传播替代手动规则编辑，实现更自动化的技能演化。

## 关键术语表
- **OPD (On-policy Signed Contextual Difference)**：同策略上下文差异，Teacher 与 Student 在相同 token 上的 log-probability 差值，正号表示技能提升该 token 概率，负号表示抑制。
- **Skill Bank (B)**：可编辑的外部技能文档集合，包含带元数据（适用条件、优先级、阻塞规则）的可识别规则。
- **Critical-first Routing**：路由器优先匹配关键事件规则，无匹配时回退到通用工作流，存在冲突时弃权。
- **Paired Verification**：在冻结策略下，比较候选编辑与当前技能库的配对轨迹成功率，通过局部和全局门控后才提交更新。
- **Cycle c**：第 c 个训练周期，包含固定技能库的策略学习阶段和固定策略的技能修订阶段。
- **HMean (Hierarchical Mean)**：先在任务内聚合，再跨任务、跨 seed 取平均的聚合方式。
- **Teacher-Student Gap**：skill-conditioned Teacher 与 no-skill Student 的性能差值，反映外部技能的剩余价值。

## 可复现要素
- **数据集**：ALFWorld（6 类 household tasks）、WebShop（1000-product subset），使用 Yang et al. (2026b) 的基准数据集。
- **代码/权重**：论文声明将发布 training code、configurations、skill banks、evaluation protocols、task manifests、seed records、episode outcomes 及 aggregation scripts（链接至 checkpoint）。
- **关键超参**：
  - WebShop 1.7B Cycle 3：$(\eta, \lambda_o, \lambda_d) = (0, 0.001, 0.1)$
  - WebShop 3B Cycle 3：$(\eta, \lambda_o, \lambda_d) = (0, 0, 0.1)$
  - ALFWorld 3B Cycle 3：$(\eta, \lambda_o, \lambda_d) = (0.1, 0, 0)$
  - Learning rate: $10^{-6}$，PPO clip range $[0.8, 1.2]$，reference coefficient 0.01
  - 训练配置见 Table 5，完整设置在 Appendix E
- **硬件**：WebShop 1.7B 用 2 GPU，3B 用 4 GPU；ALFWorld 3B 用 2 GPU
