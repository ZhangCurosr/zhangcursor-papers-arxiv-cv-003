---
title: "SkillCycle-Evolving-Skills-through-Internalization"
source: https://arxiv.org/pdf/2610.09430v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 22:54:31"
field: "LLM Agent 技能内化与持续学习"
keywords: ["skill internalization", "agentic RL", "self-distillation", "skill bank revision", "teacher-student", "reinforcement learning"]
innovations: ["蒸馏反馈双重角色：同时监督策略学习与定位技能规则修订点", "策略-技能库两段交替共演化（固定库学习↔固定策略修订）", "rule-level + whole-bank 两层成对环境验证确保修订稳健性"]
benchmarks: ["ALFWorld", "WebShop"]
---

# 论文速读：SkillCycle-Evolving-Skills-through-Internalization

## 一句话总结
SkillCycle 提出一个**策略-技能库协同演化**框架：通过技能内化训练无技能 Student，再用蒸馏反馈与交互结果诊断并修订技能规则，经成对环境验证后提交至技能库，再进入下一轮训练，最终实现**推理时不需要外部技能输入**的 agent。

## 研究问题与动机
1. **技能内化后指导仍会失效**：当策略从技能中学习到能力后，原有规则可能变成冗余、误导性甚至对新生状态不足，现有方法（如 Skill0、Skill-SD）仅做"内化→减少依赖"单向操作，未解决"指导内容本身如何随策略演进"的问题。
2. **区分掌握与无效指导困难**：Teacher-Student gap 缩小既可能是成功学习，也可能是某条技能对双方同样无益，需要更细粒度的定位机制。
3. **蒸馏反馈只有一个用途**：既有工作把 token 级上下文差异（OPD）只用于监督策略学习，未利用它来定位并修订技能规则。
4. **静态/单次更新技能库不够**：固定技能库会随策略恶化；只更新一次也无法匹配后续多阶段的训练需求。

## 核心贡献（创新点）
1. **策略-技能库共演化框架**：交替执行"内化学习（固定技能库+路由器）"与"规则修订（冻结策略）"两段，本质区别在于把技能库视为可编辑的训练监督源而非静态提示。
2. **蒸馏反馈的双重角色**：token 级 OPD 差异既用于优势注入监督 Student，又用于定位"受规则影响的关键决策点"；已有工作仅用前者。
3. **规则影响力诊断 + 成对环境验证**：通过冻结策略下"含规则 vs 去规则"的 log-likelihood 差 $d_{t,r}$ 定位 influential rule，再经 rule-level 与 whole-bank 两层配对 rollout 验证才提交，避免仅凭文本质量 commits。
4. **推理时零技能达 SOTA**：3B 模型在 WebShop 无技能 SR 74.74% / Score 88.37，相对 OPID 分别 +0.54pp / +3.37 分，并在 ALFWorld 3B macro SR 82.53% 超过 SDAR/OPID 等。
5. **可解释路由设计**：先优先关键事件规则，其次一般工作流，平局时 abstain，且基于可观测事实判定 eligibility，避免依赖隐藏状态或未来信息。

## 方法详解
**问题设定**：指令 $x \sim \mathcal{D}$，策略 $\pi_\theta$ 基于可观测历史 $h_t$ 生成 $y_t$，环境执行解析动作 $a_t$。技能库 $B$ 含可识别规则与适用性元数据，路由器 $R(h_t; B)$ 选至多一条文档 $k_t$。Student $\pi^S_\theta$ 无技能，Teacher $\pi^T_\theta$ 同参数但上下文含 $k_t$，目标为最大化无技能期望效用 $J^S_\mathcal{D}(\theta_C)$（式 1）。

**交替更新**（式 2）：
- 内化阶段：固定 $H_{c-1}=(B_{c-1}, R_{c-1})$，用 $\omega$ 步 PPO 更新 $\theta$ 得到 $(\theta_c, E_c)$。
- 修订阶段：冻结 $\theta_c$，基于证据 $E_c$ 提议 KEEP/REWRITE/DROP/ADD，成对验证后得 $H_c$。

**技能内化（Sec 2.2）**：
- Token 级 OPD 信号（式 3）：$g_{t,j} = \mathrm{sg}[\log\pi^T(y_{t,j}|h_t,k_t,y_{t,<j}) - \log\pi^S(y_{t,j}|h_t,y_{t,<j})]$，正表示技能提升该 token 概率、负则降低，两者均保留。
- 混合优势（式 4）：$A^{\mathrm{mix}}_{t,j} = \mathrm{sg}(A^{\mathrm{env}}_t) + \eta \cdot \mathrm{clip}(g_{t,j}, -c_g, c_g)$，损失 $\mathscr{L} = \mathscr{L}_{\mathrm{PPO}}(A^{\mathrm{mix}}) + \lambda_o \mathscr{L}_{\mathrm{OPD}} + \lambda_d \mathscr{L}_{\mathrm{demo}} + \beta \mathscr{L}_{\mathrm{ref}} - \alpha \mathscr{H}$。WebShop 3B Cycle 3 取 $(\eta,\lambda_o,\lambda_d)=(0,0,0.1)$ 仅用 demo；ALFWorld 3B 取 $(0.1,0,0)$ 注入符号 gap。

**证据驱动修订（Sec 2.3）**：
- 规则影响力（式 5）：$d_{t,r} = L_c(t;k_t) - L_c(t;k_t \setminus r)$，冻结 $\theta_c$ 下对比含/不含规则 $r$ 的动作 token log-likelihood 差。
- 诊断优先级：$I_t = n_t^{-1}\sum_j m^a_{t,j}|g_{t,j}|$，按绝对 gap 排序；结果标签 $\ell_t \in \{-1,+1,\perp\}$。
- 改写候选筛选：$\delta_t(q) = L_c(t;k^{(q)}_t)-L_c(t;k_t)$，$S(q)=\mathrm{HMean}[\ell_t\delta_t(q)]$（有标签）、$A(q)=\mathrm{HMean}[|\delta_t(q)|]$（全部），通过 $A>\varepsilon=10^{-4}$ 且 $S\ge 0$（有标签时）。
- 四种操作：KEEP / REWRITE / DROP / ADD（失败决策无匹配规则可新增）。

**验证提交（Sec 2.4）**：
- 目标门（式 6）：$\widehat{\Delta}(q_i) = \sum w_{x,\xi}[U(\tau^{\tilde H^{(i)}})-U(\tau^{H^{(i-1)}})]$，固定 $\theta_c$ 配对 rollouts。
- ALFWorld 要求每个受影响 family 净增益 $G_f-L_f\ge 0$ 且总体 $>0$；WebShop 3B 要求严格更多成功数、Score 不降、属性组（color/fit/shape/size）无不降、API 错误为 0。两关均过才提交 $H_c$。

## 实验与结果
**数据集**：ALFWorld（6 个家庭任务族）、WebShop 1000 商品子集；模型 Qwen3-1.7B 与 Qwen2.5-3B/7B。
**基线**：Vanilla GRPO、Skill-GRPO、OPSD、OPID、Skill-SD、RLSD、SDAR、GiGPO、Skill-Prompt。

**主要结果**：
- WebShop 3B 无技能：**SR 74.74% / Score 88.37**，相对 OPID（85.00/74.20%）**+3.37 分 / +0.54pp**，为列表中最高。
- ALFWorld 3B 无技能：macro SR **82.53%**，Pick2=89.36% 超 OPID/SDAR 的 84.20%；Clean/Pick2 分别 +21.16/+31.28pp。
- 1.7B WebShop：Score 84.65，超 Skill-SD 81.80。
- Cycle 3 消融：相对 Static 库 ALFWorld +4.95pp、WebShop +11.46pp；相对 Refresh once 再 +1.26pp / +1.82pp。
- Teacher-Student gap 收窄：WebShop 3B 从 3.90pp（Refresh once）降到 **2.08pp**；ALFWorld 3B Cycle 4 仅 1.19pp。
- API 模型实验：Skill+Router 在 DeepSeek V4-Pro/GLM-5.1/MiniMax-M3/Qwen3.6-Plus 上均优于 No Skill 与 Static General Skill，最大 WebShop +17.05pp（GLM 5.1）。
- 冻结策略修订：ALFWorld macro SR 49.43→52.45%；WebShop 57.55→68.23%（首修）。

## 相关工作脉络
1. **OPID (Yang et al., 2026b)**：同构 teacher-student + OPD 监督框架的底层基础；本文在其上引入**持久、可编辑技能库**与**环境验证修订**。
2. **Skill0/Skill0.5 (Lu 2026b; Zhu 2026)**：Skill0 逐步撤回技能、Skill0.5 结合难度感知路由；二者是"单次内化"，不维护持续修订的技能库。
3. **Skill-SD (Wang et al., 2026)**：用轨迹摘要技能做 teacher conditioning，不做库维护。
4. **SkillRL/Skill1 (Xia 2026; Shi 2026)**：递归/联合共演化技能库；但使用共享任务目标，而 SkillCycle 分离学习与修订阶段，用成对验证保证稳健性。
5. **UCOB (Tu et al., 2026)**：双向蒸馏 + credit-aware 同时更新技能效用与反思 writer；SkillCycle 明确冻结策略后再修库，两段解耦更易验证。
6. **Self-Refine/TextGrad/GEPA**：文本级自修订；SkillCycle 的修订需经**环境成对测试**才提交，区别于纯语言信号优化。

## 局限性与未来方向
1. **收益→能力的转化机制未清**：技能指导如何变成持久策略能力、何时修订最有价值，尚缺因果分析；作者建议结合学习动态做自适应修订调度。
2. **规则粒度单一**：聚焦单条规则及其适用性，**技能组合、分层组织、跨任务迁移**未研究，限制知识复用与终身学习扩展。
3. **验证成本**：每轮修订需固定策略下成对 rollout，计算开销显著；WebShop 3B Cycle 3 仅 10 个候选通过 2 个。
4. **基准限定**：只在 ALFWorld/WebShop 两个仿真环境评估，未覆盖真实部署场景与安全问题（伦理声明明确提及）。
5. **最终测试未评测**：WebShop 官方 test 尚未完成，当前为开发集结果。

## 研究启发与可借鉴点
1. **蒸馏反馈二次利用**：把 token 级 OPD 同时用于"监督学习"和"定位诊断"的思路可迁移到任何 teacher-student 框架（如 RLVR、GRPO+蒸馏组合）。
2. **规则影响力 $d_{t,r}$ 度量**：冻结策略下"含/去规则" log-likelihood 差是一种轻量、不依赖额外模型的规则归因工具，可复用于 prompt/skill 消融。
3. **两段解耦（学习↔修订）**：固定策略做修订验证、固定库做学习，既降低干扰又便于独立调参，是复杂 agent 系统设计的通用模式。
4. **成对验证的两层门禁（rule-level + whole-bank）**：防止局部有效但整体冲突的修改被提交，可借鉴到任何"迭代优化上下文/系统提示"的场景。
5. **State-aware critical-first routing + abstain**：平局时不强制选一般规则，避免噪声注入，对多技能并存的 agent 有参考价值。

## 关键术语表
- **OPD (On-policy Contextual Difference)**：同参数下 skill-conditioned Teacher 与 no-skill Student 在同一 token 上的 log-probability 差，带符号用于监督与诊断。
- **No-skill Student / Skill-conditioned Teacher**：共享参数，区别在于 Teacher 上下文多了一条由路由器选出的技能文档；Student 代表最终目标（推理无技能）。
- **Rule influence $d_{t,r}$**：固定策略下，某响应在含/不含规则 $r$ 的两个上下文中的动作 token log-likelihood 差，用于定位受规则影响的关键决策。
- **Targeted / Whole-bank gate**：两轮环境验证门——前者测单条规则修改的配对增益，后者测全库 assembled 版本是否带来净正向变化。
- **Cycle**：一次"内化学习 + 技能修订"的迭代；Cycle 0 为仅用环境 reward 初始化的 baseline。
- **Router abstention**：当关键事件规则出现平局且无明确可观测优先时，路由器返回空，避免注入任意指导。
- **HMean (Hierarchical Mean)**：先在 task 内平均、再跨 task、再跨 seed 的三层平均，用于 $S(q), A(q)$ 等筛选指标。
- **Verified demonstration**：由成对 branch 回放验证的替代合法动作，作为 Student 的辅助监督信号。

## 可复现要素
- **代码/权重/技能库/评估协议**：论文声明**将开源**（reproducibility statement），提供训练代码、可执行配置、技能库与路由器版本、任务 manifest、seed 记录、接受设置、episode 级结果及聚合脚本，并与 checkpoint 绑定；目前 arXiv 版本尚未附链接。
- **数据集**：ALFWorld、WebShop 1000 商品子集（与 Yang et al. 2026b 相同基准）。
- **关键超参**：PPO ratio clip [0.8, 1.2]、AdamW lr=$10^{-6}$、weight decay=0.01、gradient-norm clip=1、entropy $\beta=0.001$；训练组大小 8、max prompt 4096/response 512（WebShop）或 2048/128（ALFWorld）；正式 actor 更新 8 步；OPD gap clip 取校准集 95 分位数；筛选阈值 $\varepsilon=10^{-4}$。
- **配置**：WebShop 3B $(\eta,\lambda_o,\lambda_d)=(0,0,0.1)$；ALFWorld 3B $(0.1,0,0)$；详细见 Table 5。
