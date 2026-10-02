---
title: "LEARNING-NORMAL-DIFFUSION-DYNAMICS-FOR-BACKDOOR-DEFENSE-IN-T"
source: https://arxiv.org/pdf/2609.39548v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 07:48:15"
field: "生成模型安全与鲁棒性"
keywords: ["backdoor defense", "text-to-image diffusion", "normal dynamics learning", "trigger localization", "transition dynamics", "diffusion trajectory"]
innovations: ["提出从转换动力学视角统一建模后门偏移，而非依赖特定异常模式", "构建紧凑多空间轨迹表示并训练 timestep-conditioned 动力学模型实现 attack-agnostic 检测", "设计低语义词替换策略实现无需先验知识的触发器定位"]
benchmarks: ["DiffusionDB", "Stable Diffusion v1.5", "Stable Diffusion XL", "Stable Diffusion v3.5", "Pixart-α"]
---

# 论文速读：LEARNING-NORMAL-DIFFUSION-DYNAMICS-FOR-BACKDOOR-DEFENSE-IN-T

## 一句话总结
本文提出一种基于"正常扩散动态学习"的后门防御框架 NDDL，通过仅使用良性样本建模扩散轨迹的时序转换规律，以动力学不一致性检测和低语义词替换实现后门攻击的泛化检测与触发器定位。

## 研究问题与动机
- **核心问题**：如何对 T2I 扩散模型中未知的后门攻击进行泛化防御，包括后门检测与触发器定位。
- **现有方法不足**：现有防御多依赖特定表征空间（cross-attention、噪声预测、神经元激活）中的异常模式，与攻击机制绑定，面对多样化攻击时泛化性受限。
- **新视角动机**：将后门效应视为对正常扩散轨迹演化规律的偏离，而非某一时刻的局部异常；良性轨迹具有结构化和时序依赖的转换规律，后门会引入扰动导致偏离。
- **理论支撑**：给出"转换动态偏差假设"，证明后门扰动会沿扩散轨迹传播，且可通过正常转换预测的不一致性来检测。

## 核心贡献（创新点）
1. **提出转换动力学视角**：将后门攻击统一建模为对正常扩散轨迹演化的偏离，而非特定空间的异常，提供 attack-agnostic 的防御范式。
2. **设计 NDDL 框架**：构建紧凑的多空间轨迹表示（cross-attention、latent、noise），训练 timestep-conditioned 动力学模型预测正常转换，实现无先验知识的后门检测。
3. **无需先验的触发器定位**：通过低语义词替换策略识别可疑触发令牌，避免单一替换带来的偏差和与嵌入后门交互的风险。
4. **系统实验验证泛化性**：在多种 T2I 架构（U-Net 与 DiT）和多样化攻击上验证，NDDL 在 ACC 和 AUROC 上显著优于基线。

## 方法详解
### 整体框架
NDDL 分为四个阶段：多空间轨迹表示构建 → 正常动态学习 → 后门检测 → 触发器定位。

### 1) 多空间轨迹表示（Stage I）
将原始状态 $x_t = (A_t, z_t, \epsilon_t)$ 映射为紧凑表示 $r_t = \phi(x_t) \in \mathbb{R}^d$：
- **Cross-attention**：提取注意力熵（分布集中程度）、有效秩（结构复杂度）、token 重要性（token 级贡献）、头多样性（多头间差异）。
- **Latent**：提取通道范数、轨迹曲率（二阶时间变化）、频域能量、时间变化（一阶时间变化）。
- **Noise**：提取通道范数、通道方差（空间分散度）、频域能量、时间变化。
- 对三类表示进行稳健归一化（基于良性样本的中位数和 MAD），并按块缩放后拼接。

### 2) 正常扩散动态学习（Stage II）
- 定义转换增量 $\Delta r_t = r_{t+1} - r_t$。
- 训练 timestep-conditioned 动力学模型 $G_\theta$ 预测增量：$\Delta\hat{r}_t = G_\theta(r_t, t)$。
- 预测下一状态：$\hat{r}_{t+1} = r_t + \Delta\hat{r}_t$。
- 损失函数（块加权 MSE）：
  $$\mathcal{L} = \lambda_A \mathcal{L}_A + \lambda_z \mathcal{L}_z + \lambda_\epsilon \mathcal{L}_\epsilon$$
  其中 $\mathcal{L}_m = \frac{1}{T}\sum_{t=0}^{T-1}\|\Delta r_t^m - \Delta\hat{r}_t^m\|_2^2$。
- 模型为 Residual MLP，时间步通过正弦位置编码映射为 32 维向量，隐藏层 512 维。

### 3) 后门检测（Stage III）
- 计算转换不一致性：$E_t = \frac{1}{d}\|r_{t+1} - \hat{r}_{t+1}\|_2^2$。
- 将去噪过程划分为 $n$ 个等长窗口，计算每个窗口的平均不一致性 $S_i(p)$。
- 异常分取最大窗口值：$S(p) = \max_i S_i(p)$。
- 阈值 $\lambda_1$ 通过对良性验证轨迹预测误差做高斯拟合自动确定：$\lambda_1 = \mu_{benign} + m\cdot\sigma_{benign}$。

### 4) 触发器定位（Stage IV）
- 初始候选集 $\mathcal{V}_0$ 包含低语义词（如 a, an, the, this, that 等）。
- 用良性参考 prompt 估计替换偏差 $B(v)$，筛选出变化小的候选集 $\mathcal{V}_c$。
- 对可疑 prompt 中每个 token 进行替换，计算分数：$C_i = S(p_s) - \text{Median}_{v\in\mathcal{V}_c} S(p_s^{(i\to v)})$。
- 若 $C_i > \lambda_2$ 则判定为触发器 token。

## 实验与结果
### 数据集与模型
- 评测基准：DiffusionDB（采样 1,000 良性 prompt + 1,000 后门 prompt）。
- 模型：Stable Diffusion v1.5（主实验）、SDXL、SD v3.5（DiT）、Pixart-α（DiT）。
- 攻击方法：BadT2I、EvilEdit、MasqLoRA、Rickrolling、STEBA。
- 防御基线：UFID、T2IShield、NaviT2I、STEDF。

### 主要结果（SD v1.5）
| 攻击 | 方法 | ACC | AUROC |
|------|------|-----|-------|
| STEBA | STEDF（最优基线） | 89.3% | 89.6% |
| STEBA | **NDDL** | **96.8%** | **97.0%** |
| EvilEdit | **NDDL** | **98.5%** | **98.6%** |
| Rickrolling | **NDDL** | **98.2%** | **98.1%** |
| MasqLoRA | **NDDL** | **98.2%** | **98.3%** |
| BadT2I | NDDL | 99.0% | 99.2% |

- **最强提升**：在 STEBA 攻击上，ACC 提升 **+7.5%**（89.3%→96.8%），AUROC 提升 **+7.4%**。
- **触发器定位**：NDDL 在所有 trigger 类型（单token、多token、特殊字符、句子级）上均达到最优，句子级 ETR 比 NaviT2I 提升 **+7.7%**。
- **DiT 泛化**：在 SD v3.5（DiT）上 NDDL ACC=91.2%，优于 NaviT2I（83.8%）。

### 消融实验要点
- 多空间表示：Full（99.0%/99.2%）优于任一单空间（58.7%~63.5%）。
- 去噪阶段：中间阶段（16-30步）效果最佳，早期弱、晚期略降。
- 动力学建模：学习转换动态优于直接原始差分；加入 timestep 条件显著提升性能。
- 窗口长度：短窗口聚合效果优于单步或全轨迹平均。

## 相关工作脉络
1. **T2IShield（Wang et al., 2024）**：通过 cross-attention 图中的 assimilation 模式检测后门，属于特定空间异常检测，NDDL 从其视角转向正常动力学建模。
2. **NaviT2I（Zhai et al., 2025）**：分析神经元激活变化，需访问内部表征且对触发形式有限制；NDDL 无需先验知识且支持任意触发形式。
3. **STEDF（Pan et al., 2026）**：利用时空特征冗余进行检测，本质仍是异常模式匹配；NDDL 提供 attack-agnostic 的统一框架。
4. **UFID（Guan et al., 2025）**：黑盒方法，仅用图像级相似度，精度较低；NDDL 白盒访问内部状态，性能更高。
5. **Rickrolling（Struppek et al., 2023）**：将后门注入文本编码器，NDDL 能有效检测此类隐蔽攻击（ACC=98.2%）。
6. **BadT2I（Zhai et al., 2023）**：数据投毒型攻击的代表，NDDL 在其单token/句子级触发下均取得最优定位结果。

## 局限性与未来方向
- **仅适用于白盒场景**：需要访问模型内部状态（cross-attention、latent、noise），黑盒部署受限。
- **需要良性 prompt 训练集**：动力学模型需通过良性样本学习，若良性数据本身被污染则效果下降。
- **未讨论对抗性后门**：针对能够自适应绕过动力学检测的高级后门攻击，泛化性待验证。
- **未来方向**：扩展至黑盒/灰盒场景、研究对抗鲁棒性、探索更高效的轨迹压缩表示。

## 研究启发与可借鉴点
1. **正常建模替代异常检测**：从"检测异常"转向"学习正常"，为其他生成模型安全提供新思路，可迁移至音频、视频扩散模型。
2. **紧凑多空间表示设计**：将高维内部状态压缩为互补的特征描述子（熵、范数、频域能量等），兼顾信息保留与计算效率。
3. ** timestep-conditioned 动力学建模**：引入时间步作为条件输入，捕捉扩散过程的非平稳演化规律。
4. **低语义词替换触发器定位**：避免预设替换词，通过一致性校正筛选低影响替换子，减少对嵌入后门的干扰。
5. **窗口化异常聚合策略**：分段聚合而非全轨迹平均，保留阶段特异性异常信号。

## 关键术语表
- **Backdoor Attack（后门攻击）**：向模型注入隐蔽触发机制，使模型在正常输入上表现正常，但在含触发器的输入上执行恶意行为。
- **Diffusion Trajectory（扩散轨迹）**：从纯噪声到目标图像的整个去噪过程所经历的隐状态序列。
- **Transition Dynamics（转换动态）**：相邻时间步之间状态变化的规律，本文指正常扩散过程的演化规则。
- **Normal Diffusion Dynamics Learning（NDDL）**：本文提出的防御框架，通过良性样本学习正常转换动态并检测偏离。
- **Cross-Attention（交叉注意力）**：扩散模型中连接文本条件与图像隐空间的注意力机制层。
- **Dynamics Inconsistency（动力学不一致性）**：观测到的状态转换与预测的正常转换之间的偏差度量。
- **Low-Semantic Word Substitution（低语义词替换）**：用功能性弱义词替换触发器候选词以评估其对异常分的影响。
- **ETR（Exact Trigger Recovery）**：精确触发器恢复率，衡量所有触发 token 均被正确识别的比例。

## 可复现要素
- **数据集**：DiffusionDB（公开），实验使用其采样 prompt。
- **代码**：论文声明匿名仓库已公开代码与实验脚本（Reproducibility Statement）。
- **权重**：使用标准预训练模型（SD v1.5、SDXL、SD v3.5、Pixart-α）。
- **关键超参**：窗口数 n、步长 K=5（主实验）、残差 MLP 维度 512、时间编码维度 32、损失权重 $\lambda_A=\lambda_z=\lambda_\epsilon$。
- **硬件**：论文未明确说明，实验需在 GPU 集群运行。
