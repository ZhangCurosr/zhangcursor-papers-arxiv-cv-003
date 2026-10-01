---
title: "MARATHONER-ULTRA-LONG-HORIZONAUTONOMOUS-INTELLIGENCE"
source: https://arxiv.org/pdf/2609.34378v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 09:53:55"
field: "自主智能体与代码生成"
keywords: ["ultra-long-horizon", "autonomous agent", "reinforcement learning", "code generation", "process reward", "GitHub PR synthesis"]
innovations: ["提出综合后训练管道培养超长期执行能力", "设计多任务链式合成 frontier-level 难度任务", "引入 Later Stage Bonus Reward 鼓励后期有效操作"]
benchmarks: ["FrontierSWE", "NL2Repo", "SWE-Marathon", "Terminal Bench 2.0", "SWE-Bench Verified"]
---

# 论文速读：MARATHONER-ULTRA-LONG-HORIZON-AUTONOMOUS-INTELLIGENCE

## 一句话总结
本文提出 Marathoner，一个通过综合后训练管道（超长期任务合成 + 拒绝采样微调 + 强化学习）注入超长期执行能力的自主智能体模型，可在挑战性任务上持续工作 10+ 小时并完成 1000+ 次工具调用，在多个基准上超越部分强专有模型。

## 研究问题与动机
- **核心问题**：现有开源模型普遍缺乏超长期（ultra-long-horizon）自主执行能力，无法像商业模型（如 GPT-6-Astra、Fable-5）那样持续数小时至数天解决复杂软件工程问题。
- **动机 1**：人类天然具备持久攻克长期目标的能力，这种能力是智能的关键维度，应被赋予开源模型。
- **动机 2**：前沿专有模型的训练细节高度保密，学术界难以复现其超长期能力，导致开源模型在复杂场景下性能严重受限。
- **动机 3**：现有方法缺乏系统性的训练框架来培养"长时间持续有效执行"的能力，而非仅关注单次任务完成。

## 核心贡献（创新点）
1. **Marathoner 超长期智能体模型**：明确提出并系统化训练具备 10+ 小时连续执行、1000+ 工具调用能力的开源智能体，填补了该领域的空白。
2. **超长期任务合成管道 + Multi-Task Chaining**：首次系统地从 GitHub 大型 PR 合成高难度任务级数据，并提出多任务链式技术合成 frontier-level 难度任务，为训练提供难度适配的数据源。
3. **Later Stage Bonus Reward 奖励设计**：在 RL 阶段引入过程级奖励，显式鼓励模型在执行后期仍保持有意义操作，解决长期执行中后期动力衰减问题。
4. **综合后训练框架**：将任务合成、拒绝采样微调（RFT）与强化学习（RL）整合为统一流水线，证明端到端真实环境交互对超长期能力形成的关键作用。

## 方法详解

### 1. 超长期任务合成（Ultra-Long-Horizon Task Synthesis）
- **仓库收集**：采集 10,000 个多样化 GitHub 仓库，覆盖前端、后端、CUDA、数据库内核、SciPy、vLLM、SGLang 等，确保语言与领域多样性。
- **PR 挖掘与难度分级**：挖掘 100,000 个 major release PR，按新增代码行数分为三级：Easy（100-200 行）、Medium（200-1,000 行）、Hard（1,000+ 行），比例 2:3:5。
- **任务构建**：每个任务包含：
  - **Software**：PR 前一个 commit 的代码库
  - **Instruction**：优先使用 PR 首条评论，其次 release notes，最后由 LLM 基于 diff 生成
  - **Reward Verifier**：包含 fail-to-pass（PR 引入的新测试）和 pass-to-pass（原有测试）的完整单元测试套件
  - **Sandbox**：干净 Ubuntu 24.04，4 CPU / 10GB 内存 / 40GB 存储，无预装包，执行时限 5/10/20 小时
- **Multi-Task Chaining**：随机选取 5 个原子任务拼接为 Frontier 任务，软件为多仓库集合，指令拼接，验证器独立作用于各仓库，执行时限 40 小时。

### 2. 拒绝采样微调（Rejection Sampling Finetuning, RFT）
- **教师模型**：使用 Kimi K3 作为 teacher model。
- **多样化 Harness**：同时使用 Claude Code、Codex、OpenClaw 生成轨迹，防止过拟合单一 harness。
- **ReAct Loop**：轨迹格式 $\mathcal{H}_T = (\tau_0, a_0, o_0, \ldots, \tau_T, A)$，其中 $\tau_i$ 为思考、$a_i$ 为行动、$o_i$ 为观测。
- **拒绝采样**：仅保留 reward=1（全部测试通过）的轨迹，最终获得 40,820 条高质量轨迹（Claude Code: 14,159；Codex: 13,583；OpenClaw: 13,078）。
- **SFT 损失函数**：
  $$\mathcal{L}_{\mathrm{SFT}} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{t=1}^{T_i} \log \pi_\theta(o_{i,t} | o_{i,<t}, q_i)$$
  使用 LLaMAFactory 全参数微调，序列长度 256k，学习率 5e-6，5 epochs，cosine warmup 0.05，DeepSpeed ZeRO-3 + CPU offloading。

### 3. 强化学习（Reinforcement Learning）
- **算法**：采用 GRPO（Group Reward Proximal Optimization），目标函数：
  $$J_{\mathrm{GRPO}}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G} \min\left(\frac{\pi_\theta}{\pi_{\theta_{\mathrm{old}}}}A_i, \mathrm{clip}\left(\frac{\pi_\theta}{\pi_{\theta_{\mathrm{old}}}}, 1-\epsilon, 1+\epsilon\right)A_i\right) - \beta D_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{ref}})\right]$$
- **训练设置**：使用 AReaL 框架，每任务 8 次 rollout，batch size=32，学习率 1e-6，temperature=0.6，top_p=0.95。
- **Later Stage Bonus Reward**：
  1. 用 Qwen-3.8-Max 将轨迹总结为多个执行阶段
  2. 对每个阶段判断是否包含"exceptionally valuable operation"（如发现并修复关键隐藏 bug、重大性能优化等），输出 binary flag（0/1）
  3. 若后半段（50%-100%）存在 flag=1 的阶段，给予额外 +0.5 奖励
- **最终奖励**：$R_i = R_{\mathrm{correctness}}(o_i) + R_{\mathrm{bonus}}(o_i)$，其中正确性奖励为二值 0/1，bonus 为 0 或 0.5。

## 实验与结果

### 数据集与基线
- **5 个评测基准**：FrontierSWE（17 题）、NL2Repo、SWE-Marathon（20 题）、Terminal Bench 2.0（89 题）、SWE-Bench Verified（500 题）
- **对比模型**：GLM-5.2、Qwen-3.7-Max、Kimi K3、GPT-6-Astra、Claude Fable 5.1、Gemini-3.1-Pro 及多个开源基线

### 主要结果
| 模型 | FrontierSWE | NL2Repo | SWE-Marathon | Terminal Bench 2.0 | SWE-Bench Verified |
|------|-------------|---------|--------------|-------------------|-------------------|
| Qwen3.5-9B (base) | 10.2 | 17.9 | 0 | 27.3 | 43.8 |
| **Marathoner-9B** | **26.4** | **34.7** | **8.2** | **57.2** | **77.5** |
| GPT-6-Astra | 89.1 | 78.2 | 48.7 | 87.4 | 90.7 |
| Claude Fable 5.1 | 86.2 | 73.2 | 42.8 | 85.9 | 91.8 |

- **提升幅度**：相对于 base 模型，Marathoner-9B 在 FrontierSWE 提升 +16.2pp，Terminal Bench 2.0 提升 +29.9pp，SWE-Marathon 提升 +8.2pp（base 为 0）
- **超越专有模型**：在 FrontierSWE（26.4 vs 25.9）和 SWE-Marathon（8.2 vs 5.9）上超越 Gemini-3.1-Pro

### 执行统计
| 基准 | 平均执行时间 | 平均步骤 | 平均工具调用 |
|------|-------------|---------|-------------|
| FrontierSWE | 3.56h | 426.2 | 648.3 |
| Terminal Bench 2.0 | 0.58h | 104.8 | 239.4 |

### 关键消融
- **Multi-Task Chaining**：vanilla → +2.2pp（FrontierSWE）
- **Diverse-Harness**：vanilla → +1.4pp
- **Later Stage Bonus Reward**：vanilla → +1.9pp
- **环境扩展**：RL 训练任务从 1,000 增至 8,000，性能持续提升
- **链式任务数量**：5 个原子任务链式效果最佳

## 相关工作脉络
1. **GLM-5.2 / Qwen-3.7-Max / Kimi K3 / GPT-6-Astra / Fable-5**：已有超长期能力的专有模型，但训练细节不公开；本文首次系统性地开源复现方案。
2. **交互式代码助手（GitHub Copilot、Cursor）**：聚焦对话式代码补全与编辑，不具备自主长期执行能力。
3. **自主任务导向智能体（Claude Fable 5）**：强调持续工作流，但本文聚焦于开源模型的超长期能力培养框架。
4. **规划中心智能体（CodePlan）**：通过伪代码显式规划；本文强调通过真实环境交互而非显式规划培养长期能力。
5. **判别式过程奖励（DreamPRM、PQM）**：训练专用 PRM 逐步骤评分；本文的 Later Stage Bonus 是轻量级的阶段级奖励，无需额外训练 PRM。
6. **生成式过程奖励（ThinkPRM）**：通过内部思考循环生成 critique；本文直接利用强 LLM judge 进行二值评分，实现更简洁。

## 局限性与未来方向
- **依赖强教师模型**：RFT 阶段依赖 Kimi K3 生成轨迹，自训练闭环尚未实现
- **训练数据规模有限**：仅 100,000 个合成任务，难以覆盖所有领域
- **多任务链式复杂度敏感**：实验显示链式 5 个任务最优，过多或过少均损害性能
- **奖励设计主观性**：Later Stage Bonus 依赖 LLM judge 判断"高价值操作"，可能存在判断偏差
- **未来方向**：探索无教师自训练方案、扩展到更多领域（如数学证明、科学研究）、扩大合成数据规模、改进过程奖励的细粒度设计

## 研究启发与可借鉴点
1. **任务难度递进策略**：通过 Easy/Medium/Hard 三级 PR 规模配比（2:3:5）实现渐进能力培养，可迁移至其他能力培养场景。
2. **多任务链式（Multi-Task Chaining）**：将多个独立任务拼接为复合型挑战任务，是合成高难度训练数据的通用范式，值得在其他 agent 任务中探索。
3. **Later Stage Bonus Reward 的设计思想**：在长程执行中引入过程级奖励弥补稀疏结果奖励的不足，该思路可迁移至数学推理、代码生成等需多步操作的领域。
4. **多样化 harness 生成轨迹**：避免模型过拟合单一工具接口，促进通用执行策略的学习，对多工具 agent 训练具有参考价值。
5. **端到端真实环境交互**：RL 阶段在独立沙箱中真实执行任务，而非仅模拟交互，证明了真实环境对能力形成的必要性。

## 关键术语表
- **Ultra-Long-Horizon Execution**：指智能体能够持续运行数小时至数天、完成复杂多步骤任务的自主执行能力
- **Multi-Task Chaining**：将多个独立任务拼接为一个更高难度的复合任务的合成技术
- **Later Stage Bonus Reward**：在 RL 阶段对执行后半段包含高价值操作的轨迹给予额外奖励的过程级奖励机制
- **Rejection Sampling Finetuning (RFT)**：通过拒绝采样筛选高质量轨迹后进行监督微调的训练阶段
- **Harbor**：基于容器的智能体评估与优化框架，提供统一的沙箱管理、rollout 和奖励计算接口
- **GRPO (Group Reward Proximal Optimization)**：基于组采样的近端策略优化算法，本文用于 RL 训练
- **Fail-to-Pass / Pass-to-Pass Tests**：分别验证新实现功能和回归已有功能的两种单元测试类型

## 可复现要素
- **数据集**：100,000 个合成任务（基于 GitHub PR），论文未提及公开
- **代码**：论文未提及开源
- **权重**：未提及开源
- **关键超参**：
  - SFT：序列长度 256k，学习率 5e-6，5 epochs，cosine warmup 0.05
  - RL：batch size=32，rollouts per task=8，学习率 1e-6，temperature=0.6，top_p=0.95
  - Later Stage Bonus：bonus 值 0.5，奖励阶段 50%-100%
- **训练硬件**：32× H800 GPU
- **框架**：LLaMAFactory（SFT）、AReaL（RL）、Harbor（环境管理）
