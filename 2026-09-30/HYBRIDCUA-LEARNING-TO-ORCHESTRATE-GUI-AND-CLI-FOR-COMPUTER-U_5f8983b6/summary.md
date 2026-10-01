---
title: "HYBRIDCUA-LEARNING-TO-ORCHESTRATE-GUI-AND-CLI-FOR-COMPUTER-U"
source: https://arxiv.org/pdf/2609.38008v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 21:42:17"
field: "计算机使用智能体"
keywords: ["Computer-Use Agents", "GUI-CLI Orchestration", "RLVR", "Hybrid Action Space", "Reinforcement Learning", "Multimodal Agents"]
innovations: ["统一 bash 动作格式将 GUI 与 CLI 整合进单一动作空间", "CLI-aware 双粒度奖励（任务级路由 + 步级执行）驱动接口编排学习", "GUI-to-CLI 轨迹改写与 b* 偏好标注结合的混合数据生成管线"]
benchmarks: ["OSWorld", "OSWorld-MCP", "WindowsAgentArena"]
---

# 论文速读：HYBRIDCUA-LEARNING-TO-ORCHESTRATE-GUI-AND-CLI-FOR-COMPUTER-U

## 一句话总结
本文提出 HybridCUA，一种让计算机使用智能体（C UA）能够有选择地编排 GUI 交互与命令行（CLI）操作的统一数据与训练框架，通过两阶段训练（SFT + CLI-aware RLVR）使模型学会"何时用 CLI、如何用 CLI"，在 OSWorld 上以 9B 参数规模取得 53.6% 准确率，超越基座模型 14.8 个百分点。

## 研究问题与动机
- **核心问题**：现有 CUA 要么仅依赖 GUI 点击/键盘操作（通用但低效、长序列易累积错误），要么通过应用级 API/工具增强（高效但需大量定制工程、难跨应用扩展）；如何让智能体同时利用 GUI 的通用性与 CLI 的效率是一个开放问题。
- **现有方法不足一**：GUI/CLI/代码语料各自独立、互不互通——CUA 数据集几乎只有 GUI 动作，终端/代码数据缺少 GUI 上下文，导致模型从未见过"混合行为"。
- **现有方法不足二**：当前训练信号对接口选择"失明"——步级模仿只鼓励局部合理的动作，结果奖励只检查任务完成，无法区分"正确选择接口"与"低效完成"。
- **实验观察**：仅暴露 CLI 反而伤害已有模型——OSWorld 上四个主流 CUA 准确率下降 2.5~11.5pp；Qwen3.5-27B 和 EvoCUA-32B 分别仅有 15.0%、0.15% 的步数走 CLI，相当于没有真正学会使用 shell。

## 核心贡献（创新点）
1. **HybridCUA-8K 混合轨迹数据集**：构建了 GUI-only、CLI-only 以及交叉 GUI-CLI 三类轨迹共 5,000 条，以及 3,000 条带 CLI 优势标注的 RLVR 任务，首次为界面选择提供显式监督信号。
   - 与已有工作区别：不同于仅收集 GUI 轨迹或仅做 API 工具调用的数据集，本文首次系统性地将三种接口模式的轨迹与任务级别标签配对，形成可直接用于 SFT + RLVR 的混合训练资源。

2. **统一 bash 动作格式（Unified GUI–CLI Action Schema）**：将 GUI（PyAutoGUI heredoc）和 CLI（直接 shell 命令）封装进单一 `bash(command, timeout)` 动作，消除双工具切换的认知负担。
   - 与已有工作区别：相比 ToolCUA 等使用独立 `computer_use` 和 `cli` 两个工具的方案，统一格式让模型在同一动作空间内学习接口路由，而非依赖分离的工具调用决策。

3. **CLI-aware 在线强化学习奖励设计**：引入任务级奖励 $R_{\text{CLI}}$（教"何时用 CLI"）与步级奖励 $r_t^{\text{exec}}$（教"如何用 CLI"），前者基于预标注的 $b^\star$ 标签，后者惩罚 shell 执行失败。
   - 与已有工作区别：区别于 RecreationWorld 等仅监督任务结果的方案，本文显式将接口选择作为学习信号，并与执行可靠性分离优化。

4. **两阶段训练范式**：在 5,000 条混合轨迹上 SFT 后再在 3,000 条 RLVR 任务上做 GRPO 在线 RL，HybridCUA-9B 在 OSWorld 达 53.6%（+14.8pp）、WindowsAgentArena 达 36.0%（+4.0pp）。
   - 与已有工作区别：结合 SFT 的轨迹覆盖与 RLVR 的任务级验证，比仅做 SFT 或仅做 offline RL 的方法实现更好的精度-效率平衡。

## 方法详解
- **环境建模**：将计算机使用形式化为 POMDP $\mathcal{M} = \langle S, A, P, R, \Omega, O, \gamma \rangle$，观测 $o_t = (I_t, \tilde{y}_{t-1})$，其中 $I_t$ 为当前截图，$\tilde{y}_{t-1}$ 为前一步 CLI 的 stdout/stderr（无 CLI 时为空）。
- **统一动作空间**：
  $$\mathcal{A} = \{ \text{bash}(c, \delta) \mid c \in \mathcal{C}_{\text{CLI}} \cup \mathcal{C}_{\text{GUI}} \} \cup \{\text{wait}, \text{terminate}, \text{answer}\}$$
  其中 $\mathcal{C}_{\text{GUI}}$ 为带 PyAutoGUI 代码的 Python heredoc，$\mathcal{C}_{\text{CLI}}$ 为直接 shell 命令。
- **SFT 损失**：
  $$\mathcal{L}_{\text{SFT}} = -\mathbb{E}_{\tau \sim \mathcal{D}_{\text{SFT}}} \left[ \sum_{t=0}^{T} \log \pi_\theta(a_t \mid x, h_t) \right]$$
  三类轨迹共同提供 GUI 控制、shell 命令与接口路由的互补监督。
- **CLI-aware 任务级奖励**（GRPO 轨迹级）：
  $$R_{\text{CLI}} = \mathbb{I}[\text{Success}(\tau)] \cdot \mathbb{I}[b(\tau) = b^\star]$$
  $b(\tau) = \mathbb{I}[\tau \text{ 包含至少一条直接 CLI 命令}]$，$b^\star \in \{0,1\}$ 为预标注标签。
- **步级执行奖励**：
  $$r_t^{\text{exec}} = \begin{cases} -1, & \text{CLI 执行失败} \\ 0, & \text{否则（或为非 CLI 步）} \end{cases}$$
- **GRPO 归一化优势**（仅在 action $a_t$ 对应 token 上增强）：
  $$\hat{A}_{t,j} = \frac{R(\tau) - \mu_G}{\sigma_G + \epsilon} + \lambda_{\text{exec}} r_t^{\text{exec}}, \quad j \in \text{Tok}(a_t)$$
  总轨迹奖励 $R(\tau) = R_{\text{acc}} + \lambda_{\text{CLI}} R_{\text{CLI}}$，其中 $\lambda_{\text{CLI}} = 0.1$，$\lambda_{\text{exec}} = 0.3$。

## 实验与结果
- **数据集**：HybridCUA-8K，含 5,023 条 SFT 轨迹（GUI-only / CLI-only / Interleaved）与 3,000 条验证 RLVR 任务；覆盖 LibreOffice（Calc/Writer/Impress）、VS Code、Chrome、VLC、GIMP、PDF、OS 等 11 个应用域。
- **基线分类**：
  - GUI-only：Qwen3.5-27B（51.4%）、OpenCUA-7B（28.1%）、EvoCUA-32B（56.7%）
  - GUI+API：ToolCUA-8B（46.8%）、AutoGLM-OS-9B（48.9%）、UltraCUA-32B（43.7%）
  - 自身对照：Qwen3.5-9B 基座仅加 GUI（38.8%）/ 仅暴露 GUI+CLI 不做训练（18.4%）
- **主结果（OSWorld，N=361）**：
  - **HybridCUA-9B：53.6% 准确率，平均 14.0 步**，相比基座 +14.8pp、-17.6 步，为同等规模模型中最高。
  - SFT 后提升至 46.0%（+7.2pp），RL 后再提至 53.6%（+7.6pp），RL 阶段平均节省 5.8 步（-29.3%）。
  - 仅暴露 CLI 不训练的对照组准确率骤降至 18.4%，证明"知道何时/如何用 CLI"是关键瓶颈。
- **OOD 泛化**：
  - OSWorld-MCP：47.1%（+9.1pp over Qwen3.5-9B），与 ToolCUA-8B 相当。
  - WindowsAgentArena：36.0%（+4.0pp over base），超越了 ToolCUA-8B（33.8%），且在 Linux shell 上训练后成功迁移至 PowerShell。
- **消融结论**：
  - SFT 数据混合三类轨迹最优（46.0%），单类均低于混合；
  - 移除 $R_{\text{CLI}}$：CLI 使用率从 64.0% 降至 58.9%，效率提升从 29.3% 降至 18.2%；
  - 移除 $r_t^{\text{exec}}$：执行错误率回升至 16.5%（SFT 水平），说明两奖励分别控制"何时用"和"如何用"；
  - 统一 bash 格式优于分离工具格式（+7.2pp，46.0% vs 38.8%）。

## 相关工作脉络
- **GUI-only CUA**（Wu et al., 2024; OpenCUA; EvoCUA）：仅靠截图+鼠标键盘，序列长、易累积错误，本文在其基础上增加 CLI 通道实现效率提升。
- **GUI+API/Tool 方法**（ToolCUA、UltraCUA、AutoGLM-OS）：通过预定义应用 API 压缩操作，但每个应用需单独开发工具，泛化受限；本文 CLI 无需 per-application 构建，直接共享给所有应用。
- **CUA-Universe**（Shi et al., 2026）：同样合成混合任务，但用 VLM 打分而非程序化验证，反馈过于粗糙，无法支持在线 RL；本文用可执行 verifier + 奖励信号实现精细监督。
- **RecreationWorld**（Bai et al., 2026）：提供可验证环境，但仅监督任务结果，接口选择不作为学习信号；本文显式编码 $b^\star$ 标签实现接口路由监督。
- **OSWorld-MCP / MCPWorld**：测量 GUI+API/MCP 协调，与本文的 GUI+CLI 正交——前者面向结构化 API 调用，后者面向通用 shell，两者可互补。
- **定位差异**：本文填补了"接口选择作为可学习信号"这一空白，将 GUI 通用性与 CLI 可编程性融合为统一训练框架。

## 局限性与未来方向
- **CLI 可用性依赖**：无 CLI 的应用、受限 shell 环境、或 OS 特定命令语义会显著影响效果；当前评估未覆盖真实用户权限设置与长程工作流。
- **应用覆盖有限**：训练数据集中在 OSWorld 内的 11 个应用域，未充分覆盖领域专用应用（如 IDE、CAD、专业设计工具）及 OOD 环境。
- **跨 OS 迁移仍有差距**：WindowsAgentArena 上 36.0% 仍落后最强基线 EvoCUA-32B（56.7%），说明跨平台泛化仍有提升空间。
- **安全边界**：论文明确指出，shell 访问放大单步操作能力，真实部署需要额外安全机制与人工审查，不在本文范围内。
- **未来方向**：扩展数据生成管线至更多专业应用、合成更丰富的 OOD 环境、探索 GUI+CLI+API 三通路联合编排。

## 研究启发与可借鉴点
- **统一动作格式的启示**：将 GUI（heredoc）和 CLI 封装进单一 `bash` 动作，简化了模型的决策空间，避免了"调用哪个工具"的二元选择开销，对多模态 agent 设计有借鉴价值。
- **RLVR + 显式偏好标签的收益**：将任务级偏好标签（$b^\star$）与可执行 verifier 结合，为接口选择提供了细粒度信用分配，可迁移到其他"多工具/多模态选择"场景（如代码编辑 vs 文件操作、搜索 vs 直接访问）。
- **GUI-to-CLI 改写策略**：将 GUI-only 轨迹中有等价 shell 表达式的片段自动改写为 CLI 命令并回放验证，是一种低成本的混合数据合成方式，可复用于其他 CUA 数据集的扩充。
- **双粒度奖励分离设计**：$R_{\text{CLI}}$（任务级，管路由）与 $r_t^{\text{exec}}$（步级，管执行）的解耦设计逻辑清晰，适用于任何需要同时优化"选什么工具"和"怎么用工具"的 RL 场景。
- **交叉平台泛化验证思路**：在 Linux shell 上训练后评估 Windows PowerShell 表现，证明学到的"何时委托给 shell"比具体命令本身更具迁移性，为 agent 跨平台评估提供了新思路。

## 关键术语表
- **Computer-Use Agent (CUA)**：基于多模态大语言模型、通过截图感知与鼠标键盘/命令交互来完成计算机任务的智能体。
- **RLVR (Reinforcement Learning with Verifiable Rewards)**：使用程序化可验证奖励进行在线强化学习的训练范式，区别于传统 RLHF。
- **CLI-aware Reward**：包含任务级接口选择奖励（$R_{\text{CLI}}$）与步级执行奖励（$r_t^{\text{exec}}$）的组合奖励信号，专门监督混合界面编排。
- **GRPO (Group Relative Policy Optimization)**：一种基于组的相对策略优化算法，本文用于 RL 阶段，通过对组内轨迹归一化优势来更新策略。
- **Unified Bash Action Schema**：将 GUI（PyAutoGUI heredoc）与 CLI（shell 命令）统一封装进单一 `bash(command, timeout)` 动作的表示方式。
- **$b^\star$ 标注**：每个 RLVR 任务被标注为 $b^\star=1$（CLI 有明确执行优势）或 $b^\star=0$（无优势），作为接口选择的训练监督信号。
- **CUA-Gym**：一个支持大规模可验证任务生成与在线 RL 的计算机使用训练环境框架，本文用于 CLI-only 轨迹采样与 RLVR 任务构建。
- **PyAutoGUI Heredoc**：将多步 PyAutoGUI 操作封装在 Python 引号 heredoc（`python3 <<'PY' ... PY`）中执行，使 GUI 操作与 CLI 命令共享同一 bash 动作接口。

## 可复现要素
- **数据集**：HybridCUA-8K（5,023 SFT 轨迹 + 3,000 RLVR 任务），论文声明将开源。
- **代码**：论文附 Hugging Face 链接，标注将开源数据生成管线、训练管线与 HybridCUA-9B 权重。
- **基座模型**：Qwen3.5-9B（开源权重）。
- **关键超参**：
  - SFT：2 epochs，global batch size=256，lr=1e-5 cosine warmup 10%，12,000 token 上下文。
  - RL：GRPO，rollouts per prompt=8，$\lambda_{\text{CLI}}=0.1$，$\lambda_{\text{exec}}=0.3$，lr=1e-6 constant，context=3 screenshots（当前高分辨率 2,088,960px，历史各 548,800px），env steps 上限 30（训练）/50（评估）。
- **训练基础设施**：SFT 使用 2 节点 × 8 GPU（NVIDIA H20），RL 使用 3 节点 × 8 GPU；框架 verl/Megatron-LM + slime/SGLang。
- **公开状态**：论文声明将开源模型、数据与代码；基准 OSWorld/WindowsAgentArena 为公开基准。
