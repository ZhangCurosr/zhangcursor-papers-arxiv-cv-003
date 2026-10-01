---
title: "MARATHONER-ULTRA-LONG-HORIZONAUTONOMOUS-INTELLIGENCE"
source: https://arxiv.org/pdf/2609.34378v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 09:53:54"
field: "自主智能体与超长horizon执行"
keywords: ["ultra-long-horizon", "agentic model", "reinforcement learning", "process reward", "code agent", "task synthesis"]
innovations: ["基于Major Release PR与Multi-Task Chaining的超长期任务合成流水线", "Later Stage Bonus Reward过程级奖励以激励长轨迹后半段高价值操作"]
benchmarks: ["FrontierSWE", "SWE-Marathon", "Terminal Bench 2.0", "SWE-Bench Verified", "NL2Repo"]
---

# 论文速读：MARATHONER-ULTRA-LONG-HORIZONAUTONOMOUS-INTELLIGENCE

## 一句话总结
本文提出 Marathoner，一个专为超长期（ultra-long-horizon）自主执行能力训练的智能体模型；通过 GitHub 大型 PR 合成挑战任务、拒绝采样微调与强化学习，配合"后期阶段奖励"机制，使开源模型能在复杂软件工程中持续执行 10+ 小时并完成 1000+ 次工具调用。

## 研究问题与动机
1. **核心问题**：如何让开源大模型获得像 GPT-6-Astra、Claude Fable 5 等私有前沿模型那样的"持续数小时至数天自主执行"能力。
2. **数据稀缺**：现有开源模型缺乏超长期任务训练数据；前沿模型的训练细节未公开，学术社区难以复现。
3. **难度不足**：通用 SFT/RL 训练多基于短链任务，无法驱动 agent 在长时间跨度的规划、调试、回归测试中保持有效推进。
4. **过程信号弱**：传统二值 outcome reward 在长轨迹后期缺乏过程级引导，agent 容易在长时间执行中停滞或偏离目标。

## 核心贡献（创新点）
1. **Ultra-Long-Horizon Task Synthesis 流水线**：从 10,000 个多样化 GitHub 仓库中挖掘 100,000 个 Major Release PR（Hard 级含 1000+ 行新增代码），自动构建含沙箱环境、任务指令与 pass/fail-to-pass 单元测试验证器的 Harbor 格式任务数据；与已有合成工作相比，本文以真实大型 Release PR 为主源并强调多语言/多领域多样性。
2. **Multi-Task Chaining（多任务链式合成）**：将 5 个随机原子任务拼接为单一 Frontier 级高难度任务（代码库、指令、验证器分别串联），从而产生需要跨仓库协调与超长执行的训练样本；有别于单任务生成或简单拼接的 baseline。
3. **Diverse-Harness Trajectory Generation + Rejection Sampling Finetuning (RFT)**：使用 Kimi K3 教师模型配合 Claude Code / Codex / OpenClaw 三种 harness 并行生成轨迹，仅保留 reward=1 的高质量轨迹对 Qwen3.5-9B 做 256k 序列长度的全参数 SFT；区别于单一 harness 生成的做法，避免模型过拟合特定工具接口。
4. **Later Stage Bonus Reward**：在 RL 阶段用 LLM summarizer+judge 将轨迹划分为若干 phase，若后半段（50%–100%）存在"发现并修复关键隐藏 bug / 显著性能优化"等高价值操作，则额外奖励 0.5；这是面向超长期执行的过程奖励设计，不同于仅依赖最终测试正确性的二元奖励。
5. **端到端实证**：在 5 个超长期 benchmark 上，Marathoner-9B 全面超越同规模开源模型并在 FrontierSWE、SWE-Marathon 上超过 Gemini-3.1-Pro 等私有前沿模型。

## 方法详解
**1. Ultra-Long-Horizon Task Synthesis**
- 仓库收集：10,000 个 GitHub 仓库，覆盖前端/后端/CUDA/数据库内核/SciPy/vLLM/SGLang 等，兼顾多语言与多领域。
- PR 挖掘：定义 Major Release PR 为"引入实质性新功能"的 PR；按新增代码行数分层：Easy（100–200 行）、Medium（200–1,000 行）、Hard（1,000+ 行），比例 2:3:5，共 100,000 个 PR。
- 任务构建：克隆仓库至 PR 提交前的最新 commit；指令采用 PR 首条评论→release notes→Qwen-3.8-Max 基于 diff 生成的三级回退策略；验证器包含 PR 引入的 fail-to-pass 测试与仓库原有 pass-to-pass 测试；沙箱为干净 Ubuntu 24.04（4 CPU/10G 内存/40G 存储），无预装包。
- Multi-Task Chaining：随机选取 5 个原子任务拼接为 Frontier 任务，时限 40 小时。

**2. Rejection Sampling Finetuning (RFT)**
- 轨迹生成：Kimi K3 作为教师，通过 Harbor 并发调用 Claude Code / Codex / OpenClaw 三个 harness 在 53,000 个任务（50,000 原始 + 3,000 Frontier）上 rollout；最终保留成功轨迹共 40,820 条（Claude Code 14,159 / Codex 13,583 / OpenClaw 13,078）。
- SFT 目标：
  $$\mathcal{L}_{\mathrm{SFT}} = - \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T_i} \log \pi_\theta(o_{i,t} \mid o_{i,<t}, q_i)$$
  序列长度 256k，全参数微调，5 个 epoch，lr=5e-6，cosine warmup 0.05，DeepSpeed ZeRO-3 + CPU offload。

**3. Reinforcement Learning（GRPO + Later Stage Bonus Reward）**
- 算法：Group Reward Proximal Optimization (GRPO)，batch_size=32，每任务 rollout=8，lr=1e-6，temperature=0.6，top_p=0.95。
- 训练数据：5,000 原始 Hard 任务 + 15,000 用于 Multi-Task Chaining 得到 3,000 Frontier 任务，共 8,000 任务。
- 轨迹阶段化与奖励：
  - 用 Qwen-3.8-Max 将轨迹总结为语义一致的 phase（内容 + 结果）。
  - 用同一模型对每个 phase 打分：是否包含"非平凡高价值操作"（如修复关键隐藏 bug、显著优化）。
  - 后半段（50%–100%）任一 phase 得 flag=1，则 $R_{\mathrm{bonus}} = 0.5$；否则 0。
  - 最终奖励：
    $$R_i = R_{\mathrm{correctness}}(o_i) + R_{\mathrm{bonus}}(o_i)$$
    其中 $R_{\mathrm{correctness}} \in \{0,1\}$ 来自单元测试全部通过。
- harness 多样性：每次 rollout 随机配对 Claude Code / Codex / OpenClaw 之一，避免过拟合。

## 实验与结果
- **数据集与基线**：5 个 benchmark —— FrontierSWE（17 个前沿问题）、NL2Repo、SWE-Marathon（20 个人工构建难题）、Terminal Bench 2.0（89 个终端任务）、SWE-Bench Verified（500 个 GitHub issue 任务）。基线包含 GLM-5.2、Qwen-3.7-Max、Kimi K3、GPT-6-Astra、Claude Fable 5.1、Gemini-3.1-Pro 等私有模型以及 Qwen3.5/3.6 系列开源模型。
- **主要结果**（Table 1，Marathoner-9B + Claude Code harness）：
  - FrontierSWE：26.4（对比基线 Qwen3.5-9B 10.2；超越 Gemini-3.1-Pro 的 25.9）
  - NL2Repo：34.7
  - SWE-Marathon：8.2（对比 Gemini-3.1-Pro 5.9；Kimi K3 35.9 仍领先）
  - Terminal Bench 2.0：57.2
  - SWE-Bench Verified：77.5
- **消融**（Table 3）：vanilla 22.7 → +Multi-Task Chaining 24.9 → +Diverse Harness 24.1 → +Later Stage Bonus 24.6（FrontierSWE）。
- **Multi-Task Chaining 深度**（Tab. 4）：5 个原子任务拼接最优（FrontierSWE 26.4），过多/过少均下降。
- **Bonus 设计**（Tab. 5）：50%–100% 阶段 + 奖励 0.5 最优；0.2 信号弱，0.8 破坏正确性优先级。
- **阶段对比**（Tab. 6）：Base 10.2 → RFT 后 18.2 → RL 后 26.4（FrontierSWE）。
- **执行统计**（Tab. 2）：FrontierSWE 平均执行 3.56h、426.2 步、648.3 次工具调用；单 case 最长 11.8h、872 步、1,273 次调用（附录 F）。
- **结论**：Marathoner-9B 在多个前沿 benchmark 上实现稳定且显著的相对提升，并在 FrontierSWE、SWE-Marathon 上超过 Gemini-3.1-Pro。

## 相关工作脉络
1. **前沿私有模型**（GLM-5.2、Qwen-3.7-Max、Kimi K3、GPT-6-Astra、Claude Fable 5）：均强调超长执行能力，但训练细节不公开；本文以开源基座 + 合成数据 + 公开 harness 路线对齐该类能力。
2. **自主编码智能体**（Claude Fable、CodePlan 等）：多聚焦"任务分解 + 执行"或交互式助手；本文聚焦"持续数小时到数十小时的端到端执行韧性"。
3. **过程奖励模型 (Process Reward Models)**：Discriminative（DreamPRM、PQM）、Generative（ThinkPRM）、Rubric-based（E-GRPO）三类；本文 Later Stage Bonus 属于轻量 rubric + LLM judge 的过程信号，专为超长期阶段设计。
4. **Agent 训练框架**（Harbor、AReaL）：Harbor 提供沙箱+harness 统一 rollout；AReaL 提供异步 GRPO 训练；本文把两者结合以支持 256k 长序列 + 大规模并发 sandbox rollout。
5. **代码合成数据**（SWE-bench、NL2Repo 等）：基于 GitHub issue/PR 构建评估集；本文反其道利用相同来源构造训练数据，强调 hard PR 与多任务链式组合。
6. **长上下文/超长轨迹建模**：256k 序列长度、Flash Attention 2、ZeRO-3 + CPU offload 共同支撑超长轨迹的训练可行性。

## 局限性与未来方向
1. **教师模型依赖**：RFT 阶段高度依赖 Kimi K3 的高质量轨迹；若教师能力不足，拒采样召回率与数据质量将下降。
2. **Harness 多样性有限**：仅使用 3 种主流 harness（Claude Code、Codex、OpenClaw），未见对其他专用 harness 或自定义工具的泛化评估。
3. **Frontier 任务分布偏移**：Multi-Task Chaining 生成的 40 小时级任务可能超出主流评测分布，实验中消融显示 7/9 链式反而降低性能。
4. **奖励设计敏感**：Bonus 值过大（0.8）会削弱对最终正确性的重视；阈值（50%）、分值、judge 准确性均需调参，泛化性待进一步验证。
5. **计算成本**：8,000 任务 × 8 rollout × 256k 序列 × 32 H800，规模较大；对资源受限团队可及性有限。
6. **评估基准仍偏软件工程**：5 个 benchmark 以代码/终端为主，其他超长期领域（科学计算、文档分析、持续研发流程）尚未覆盖。

## 研究启发与可借鉴点
1. **"以难促能"的数据分层策略**：Easy/Medium/Hard 按新增行数 2:3:5 配比，并额外构造 Frontier 链式任务；可迁移到任何需要训练持久执行能力的 agent 任务合成 pipeline。
2. **Diverse-Harness 训练避免过拟合**：同一任务用多种 harness 生成轨迹并在 RL 中随机配对；可作为通用 anti-overfit 技巧用于多工具 agent 训练。
3. **后期阶段过程奖励的 reward shaping**：用 LLM judge 识别轨迹后半段的高价值 phase 并赋予 bonus；这种"持续性推进"信号对任何长 horizon 任务都具参考价值。
4. **Harbor + AReaL 的工程范式**：统一 sandbox rollout 与异步 GRPO 训练的结合；对于需要大规模并发生态交互的 agent 训练项目具有工程复用价值。
5. **从合成数据到前沿基准的对齐思路**：利用 Release PR 作为 oracle（diff 即期望解），以 fail-to-pass + pass-to-pass 双验证器确保任务质量；可推广至其他需要"真实改动验证"的代码类任务。

## 关键术语表
- **Ultra-Long-Horizon**：指 agent 需要连续执行数小时至数十小时才能完成的长跨度任务能力。
- **Major Release PR**：为仓库引入实质性新功能的大型 Pull Request，本文以其作为困难任务合成的主要来源。
- **Multi-Task Chaining**：将多个独立原子任务拼接为单一高难度 Frontier 任务的合成方法。
- **Rejection Sampling Finetuning (RFT)**：利用强教师模型生成轨迹后，仅保留 reward=1 的高质量轨迹对基座模型做 SFT。
- **Later Stage Bonus Reward**：RL 阶段的额外过程奖励，当 agent 在执行轨迹后半段完成高价值 phase 时给予 +0.5。
- **Harbor**：面向 harness-based agent 评估与训练的沙箱统一框架，支持并发 rollout、reward 计算与轨迹收集。
- **GRPO (Group Reward Proximal Optimization)**：基于 group-wise 优势估计的 PPO 变体，本文用于 RL 阶段策略优化。
- **Fail-to-Pass / Pass-to-Pass Test**：分别用于验证新实现功能与回归已有功能的两类单元测试。

## 可复现要素
- **数据集**：论文声明合成 100,000 个任务（50,000 用于 RFT，5,000 + 15,000 用于 RL）；未明确说明是否公开原始 GitHub 仓库列表与合成脚本，但提及使用 Harbor 格式组织（论文未提及是否开源）。
- **代码/权重**：基座为 Qwen3.5-9B；训练使用 LLaMA-Factory 与 AReaL；论文未明确声明 Marathoner 权重或合成数据的开源状态（论文未提及）。
- **关键超参**：RFT——序列长度 256k、lr 5e-6、5 epochs、cosine warmup 0.05、DeepSpeed ZeRO-3 + CPU offload；RL——batch_size 32、rollout 8/task、lr 1e-6、temperature 0.6、top_p 0.95、GRPO、H800×32；教师模型 Kimi K3；harness Claude Code / Codex / OpenClaw；Bonus 阶段 50%–100%、分值 0.5；总结/judge 用 Qwen-3.8-Max。
