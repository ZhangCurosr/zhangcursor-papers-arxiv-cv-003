---
title: "HYBRIDCUA-LEARNING-TO-ORCHESTRATE-GUI-AND-CLI-FOR-COMPUTER-U"
source: https://arxiv.org/pdf/2609.38008v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 21:42:20"
field: "计算机使用智能体"
keywords: ["Computer-Use Agents", "GUI-CLI Orchestration", "Reinforcement Learning with Verifiable Rewards", "Hybrid Action Space", "Unified Action Schema", "CLI-aware Reward"]
innovations: ["构建混合GUI-CLI轨迹数据集HybridCUA-8K并提供可扩展数据生成管线", "设计双层CLI-aware奖励（任务级接口选择+步骤级执行可靠性）的在线RL训练框架", "统一bash action schema替代分离工具设计，提升跨接口路由能力"]
benchmarks: ["OSWorld", "OSWorld-MCP", "WindowsAgentArena"]
---

# 论文速读：HYBRIDCUA: LEARNING TO ORCHESTRATE GUI AND CLI FOR COMPUTER-USE AGENTS

## 一句话总结
本文提出 HybridCUA，构建包含纯 GUI、纯 CLI 及 GUI-CLI 交错轨迹的混合数据集（HybridCUA-8K），并通过两阶段训练（SFT + CLI-aware RLVR）教会计算机使用智能体何时以及如何协调图形界面与命令行接口；HybridCUA-9B 在 OSWorld 上达 53.6% 准确率，较基座模型 Qwen3.5-9B 提升 14.8 个百分点，同时在 WindowsAgentArena 实现跨平台泛化。

## 研究问题与动机
- **GUI-only 智能体效率瓶颈**：现有 CUA 依赖低级点击与键入操作，处理文件编辑、重复操作等任务需长序列 GUI 步骤，易累积级联错误（如 Qwen3.5-27B 在 OSWorld 上 25.7 平均步数）。
- **GUI+API/Tool 方案可扩展性差**：工具化方法需为每个应用单独构建 API，覆盖率和便携性受限于预定义接口，难以跨应用泛化。
- **直接暴露 CLI 反而损害性能**：实验显示（Figure 2a），将 CLI 接入四个代表性智能体后，OSWorld 准确率下降 2.5–11.5 个百分点，因为模型不知何时/如何使用 shell。
- **训练数据与监督信号的双重缺失**：现有 CUA 数据集几乎全为 GUI 动作，终端/代码数据集又缺少 GUI 上下文；步级模仿仅关注局部合理性，结果奖励无法区分"合理接口选择"与"低效路径"。

## 核心贡献（创新点）
1. **构建可扩展的混合数据生成管线与 HybridCUA-8K 数据集**：生成 5K 条 GUI-only、CLI-only、GUI-CLI 交错轨迹及 3K 条经验证的 RLVR 任务；与已有工作（如 CUA-Universe、RecreationWorld）的本质区别在于**所有轨迹均在真实 GUI 环境下执行并附带可执行程序化验证器，且任务标注了 CLI 优势标签 $b^\star$**。
2. **提出两阶段训练框架（SFT + CLI-aware RLVR）**：SFT 学习统一 action schema 下的混合执行，RLVR 通过在线交互优化接口选择与命令可靠性；区别于 ToolCUA/UltraCUA 仅优化 GUI-API 路由，本文**将接口路由本身作为显式学习信号**。
3. **设计双层 CLI-aware 奖励机制**：任务级奖励 $R_{CLI}$ 判断"何时应使用 CLI"，步骤级奖励 $r_t^{exec}$ 惩罚 shell 执行失败；与 CUA-Universe 依赖 VLM 打分相比，**本文使用程序化 verifiable reward 提供细粒度训练信号**。
4. **统一 bash action schema 替代分离工具设计**：将 GUI（pyautogui heredoc）与 CLI 命令封装为同一 `bash(command, timeout)` 动作；消融实验证明该设计较分离工具方案提升 7.2 个百分点准确率（Table 4）。

## 方法详解
**环境建模（POMDP）**：
$$o_t = (I_t, \tilde{y}_{t-1}) \sim O(\cdot \mid s_t)$$
其中 $I_t$ 为当前截图，$\tilde{y}_{t-1}$ 为前一步 CLI 输出的 stdout/stderr（若非 CLI 则为空）。

**统一动作空间**：
$$\mathcal{A} = \{\mathrm{bash}(c, \delta) \mid c \in \mathcal{C}_{\mathrm{CLI}} \cup \mathcal{C}_{\mathrm{GUI}}\} \cup \{\mathrm{wait}, \mathrm{terminate}, \mathrm{answer}\}$$
$\mathcal{C}_{\mathrm{GUI}}$ 为含 pyautogui 调用的 Python heredoc，$\mathcal{C}_{\mathrm{CLI}}$ 为直接 shell 命令。

**数据构建管线**：
- **GUI-only**：将开源 UI-MOPD 轨迹中的 GUI 操作重写为等价的 pyautogui heredoc。
- **CLI-only**：为每个应用从文档/参考代码蒸馏 CLI skill，以 Qwen3.8-27B + Claude Code harness 在 CUA-Gym 上采样成功 rollouts。
- **Interleaved GUI-CLI**：① Qwen3.8-27B 同时访问双接口自主决策；② 对 GUI-only 轨迹进行语义保留的改写（将多步 GUI 折叠为单条 CLI），回放后仅保留成功样本。
- **RLVR 任务与标签**：构建 3000 条带可执行验证器的任务，对每条任务采样 16 次 GUI-only / CLI-only / GUI-CLI rollout，按成功率与步数排名确定 $b^\star \in \{0,1\}$（1 表示 CLI 有明确优势）。

**两阶段训练**：
- **Stage I: SFT**：优化 next-token prediction loss $\mathcal{L}_{\mathrm{SFT}} = -\mathbb{E}[\sum_t \log \pi_\theta(a_t \mid x, h_t)]$。
- **Stage II: CLI-aware RLVR（GRPO）**：
  - 任务级奖励：$R_{\mathrm{CLI}} = \mathbb{I}[\mathrm{Success}(\tau)] \cdot \mathbb{I}[b(\tau) = b^\star]$
  - 步骤级执行奖励：$r_t^{\mathrm{exec}} = -1$（shell 执行失败），否则 0
  - 轨迹总奖励：$R(\tau) = R_{\mathrm{acc}} + \lambda_{\mathrm{CLI}} R_{\mathrm{CLI}}$
  - 归一化优势：$\widehat{A}_{t,j} = \frac{R(\tau) - \mu_G}{\sigma_G + \epsilon} + \lambda_{\mathrm{exec}} r_t^{\mathrm{exec}}$

## 实验与结果
**数据集与基线**：主基准 OSWorld（N=361 任务），对比三类基线：GUI-only（Qwen3.5、OpenCUA、EvoCUA）、GUI-API（ToolCUA、AutoGLM-OS-9B、UltraCUA-32B）、同基座模型的 GUI-only/CLI 配置。另测 OSWorld-MCP（MCP 工具迁移）与 WindowsAgentArena（跨 OS 泛化）。

**主要结果**：
- HybridCUA-9B 在 OSWorld 达 **53.6%** 准确率、**14.0** 平均步数，为同等规模模型最优；较基座 Qwen3.5-9B（38.8%）提升 **14.8 pp**，步数减少 17.6。
- 相较 GUI-API 方法：超越 AutoGLM-OS-9B（+4.7 pp）、ToolCUA-8B（+6.8 pp）、UltraCUA-32B（+9.9 pp，体积为其 1/4）。
- **跨平台泛化**：WindowsAgentArena 36.0%（+4.0 pp vs base）；OSWorld-MCP 47.1%（+9.1 pp vs base），与 ToolCUA-8B 持平。
- **CLI 使用率**：HybridCUA-9B CLI 步占比 64.0%，与最激进基线相当，但准确率更高——说明提升来自"知道何时用 CLI"而非"滥用 CLI"。

**消融**：
- SFT 数据混合：混合轨迹（46.0%）优于单一类型（GUI-only 43.2%、CLI-only 31.7%、Hybrid-only 41.0%）。
- RL 贡献：从 46.0% 提升至 53.6%，步数从 19.8 降至 14.0（-29.3%）。
- 双层奖励各司其职：移除 $R_{CLI}$ 导致 CLI 使用率降至 58.9%、步数缩短幅度减弱；移除 $r_t^{exec}$ 导致执行错误率反弹至 16.5%。
- 统一 action schema 优于分离工具（+7.2 pp）。

## 相关工作脉络
1. **GUI-based CUAs（OS-ATLAS、OpenCUA、EvoCUA、UI-TARS-2）**：仅通过原子 GUI 动作（点击、键入、滚动）操作，长任务需低效序列；HybridCUA 通过引入 CLI 直接压缩重复操作。
2. **GUI-API/Tool hybrid（ToolCUA、UltraCUA、AutoGLM-OS、CUA-Gym）**：扩展至预定义 API 调用，但每应用需单独构建工具，覆盖率有限；HybridCUA 以通用 shell 替代 application-specific 工具。
3. **GUI-CLI 探索（MCPWorld、OSWorld-MCP、CUA-Universe、RecreationWorld）**：前者提供 benchmark 但缺乏程序化验证；后者提供 verifiable 环境但未将接口选择纳入学习信号；HybridCUA 同时提供可执行验证器与显式接口路由奖励。
4. **RL for CUA（ComputerRL、ScaleCUA、Ui-R1）**：在线 RL 通常仅用最终 accuracy reward；HybridCUA 在此基础上注入 step-level 执行惩罚与 interface-choice 任务奖励。
5. **Coding agents（Claude Code、OpenClaw、Codex）**：已证明 shell 可压缩长 GUI 序列；HybridCUA 将该思想迁移至通用桌面场景而非仅限代码编辑。

## 局限性与未来方向
- **依赖 CLI 可用性**：无 shell 接口、受限权限环境或 OS 特定命令语义可能需不同路由策略；当前未在完整多样性场景中验证。
- **应用覆盖有限**：数据构建集中于 OSWorld 的 11 个领域（LibreOffice、Chrome、VS Code 等），未覆盖专业/领域特定应用。
- **真实性能风险**：Shell 访问放大 agent 能力，生产部署需额外安全机制与人工监督（伦理声明已指出）。
- **未来方向**：扩展至更广泛的专业应用、合成更多 OOD 环境以提升对未见应用与工作流的泛化。

## 研究启发与可借鉴点
1. **统一 action schema 设计**：将 GUI（heredoc 封装 pyautogui）与 CLI 合并为单一 `bash` 动作，避免多工具分散上下文；消融证明比分离工具设计提升 7.2 pp，可直接迁移至其他多接口 agent 研究。
2. **双层奖励分工思路**：任务级奖励约束接口路由（when），步骤级奖励约束执行质量（how）；该分离信号设计可推广至任何含多种操作原语的 agent 系统（如 GUI+API+CLI 三元混合）。
3. **数据复用的转换范式**：将开源 GUI 轨迹重写为等价 CLI/代码形式（如 UI-MOPD → pyautogui heredoc）是一种低成本的混合数据生成策略；同类方法可用于构建 GUI+API 混合数据。
4. **CLI advantage labeling 机制**：通过对比三种模式的成功率与步数自动标注 $b^\star$，无需人工标注即可获得接口偏好标签；该自监督信号构建方式可推广至其他多模态决策场景。

## 关键术语表
- **CUA (Computer-Use Agent)**：基于多模态大语言模型的通用桌面操作智能体，通过截图感知与动作执行完成跨应用任务。
- **RLVR (Reinforcement Learning with Verifiable Rewards)**：在具可执行程序化验证器的环境中执行的在线强化学习范式，适用于 agentic 决策。
- **GRPO (Group Relative Policy Optimization)**：将轨迹奖励在采样组内归一化后计算优势值的多 rollout 策略优化算法。
- **CLI-aware Reward**：专为 CLI 接口设计的奖励，含任务级 $R_{CLI}$（接口选择正确性）与步骤级 $r_t^{\mathrm{exec}}$（执行错误惩罚）。
- **Unified Bash Action Schema**：将 GUI（pyautogui heredoc）与 CLI 命令统一封装为 `bash(command, timeout)` 单一动作接口的设计。
- **Hybrid Trajectory**：交替使用 GUI 与 CLI 接口的交互式执行轨迹。
- **Interface Routing**：智能体在每步决策中选择 GUI 或 CLI 接口的路由行为。
- **HybridCUA-8K**：本文构建的数据集，含 5023 条 SFT 轨迹与 3000 条验证 RLVR 任务。

## 可复现要素
- **数据集**：HybridCUA-8K（5023 条轨迹 + 3000 条 RLVR 任务），作者声明将开源。
- **代码/权重**：作者声明将发布数据生成管线、训练管线及 HybridCUA-9B 模型权重。
- **基座模型**：Qwen3.5-9B。
- **SFT 超参**：2 epochs，global batch size=256，~366 optimizer updates，lr=$1\times10^{-5}$（cosine schedule），AdamW，bfloat16，TP=2/PP=1/DP=8，16 张 NVIDIA H20。
- **RL 超参**：GRPO，rollouts per prompt=8，$\lambda_{\mathrm{CLI}}=0.1$（轨迹级），$\lambda_{\mathrm{exec}}=0.3$（步骤级），clip=0.2，KL penalty=0.001，lr=$1\times10^{-6}$（constant），1000 训练任务（从 3000 验证任务采样），24 张 NVIDIA H20（TP=4/DP=4）。
- **框架**：verl + Megatron-LM（SFT）；slime + Megatron-LM + SGLang（RL）。
