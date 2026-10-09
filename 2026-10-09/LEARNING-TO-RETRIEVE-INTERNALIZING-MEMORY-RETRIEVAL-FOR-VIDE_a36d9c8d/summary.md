---
title: "LEARNING-TO-RETRIEVE-INTERNALIZING-MEMORY-RETRIEVAL-FOR-VIDE"
source: https://arxiv.org/pdf/2610.11444v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 10:04:08"
field: "视频生成与3D一致建模"
keywords: ["Video World Models", "Memory Retrieval", "Gated Linear Attention", "Scene Consistency", "Loop Closure"]
innovations: ["将记忆检索内化到视频生成过程的注意力状态更新中", "引入相机姿态条件门控与3D可见性重入监督实现选择性检索"]
benchmarks: ["Echo-Memory", "Revisit-Bench", "VBench"]
---

# 论文速读：LEARNING-TO-RETRIEVE-INTERNALIZING-MEMORY-RETRIEVAL-FOR-VIDE

## 一句话总结
本文提出 **L2R（Learning-to-Retrieve）** 机制，将记忆检索内化到视频世界模型的生成过程中，通过相机姿态条件检索门和3D可见性重入监督的检索触发器，在无需外部记忆库或3D重建管道的情况下显著提升长时序生成的场景一致性。

## 研究问题与动机
1. **长时序场景一致性难题**：视频世界模型需根据相机轨迹生成可探索的3D一致场景视频，但在相机重返已观察区域时，内容需与之前生成保持一致，这是现有方法的薄弱环节。
2. **外部记忆系统的局限性**：已有方法依赖外部记忆模块（如显式3D重建、独立检索通路）辅助生成，检索决策与模型内部生成动力学存在割裂，无法自适应学习何时/何内容需检索。
3. **几何条件依赖的脆弱性**：基于3D重建（深度图、点云、3DGS）的条件注入方法高度依赖重建精度，重建误差会传播到下游生成，损害长时序质量。
4. **线性注意力状态利用不足**：虽然状态空间架构（如GLA、GDN）可累积历史上下文，但缺乏选择性访问机制，无法根据当前视角智能检索相关历史信息。

## 核心贡献（创新点）
1. **检索内化框架**：将记忆检索从外部辅助通路迁移到模型内部生成过程，使检索成为视频世界模型的固有行为，而非外挂组件。
2. **相机姿态条件检索门（Camera-guided Retrieval Gate）**：设计 $\gamma_t$ 门控机制，根据当前相机位姿 $p_t$ 从累积状态 $s_{t-1}$ 中Selective访问历史信息，实现"检索什么"的决策。
3. **检索触发器与3D可见性重入监督**：引入离散触发变量 $r_t$ 决定"何时检索"，并使用基于3D几何的重见监督信号（token对应的3D点离开视锥后重新出现时激活）进行训练，无需推理时的显式3D条件。
4. **即插即用架构扩展**：L2R机制独立于底层状态转移规则，可实例化为 L2R-GLA 和 L2R-GDN 两种变体，兼容不同预训练视频生成模型（Wan2.1、Wan2.2、SANA-WM）。

## 方法详解
**整体流程**：视频世界模型将视频压缩为1D token序列 $\{x_t\}_{t=1}^{N}$，每个token关联Plücker相机射线 $p_t \in \mathbb{R}^{12}$。核心思想是用累积状态 $s_{t-1}$ 替代传统滑动窗口作为历史上下文来源。

**状态更新公式（核心）**：
$$c_t = \begin{cases} \gamma_t s_{t-1}, & \text{if } r_t = 1 \\ s_{t-1}, & \text{if } r_t = 0 \end{cases}, \quad s_t = \varphi(c_t, p_t)$$
其中 $c_t$ 是用于生成 $x_t$ 的历史上下文，$\varphi$ 为状态转移函数（GLA或GDN）。

**相机条件检索门 $\gamma_t$**：在log空间进行乘法门控以保数值稳定：
$$\log \gamma_t \leftarrow \log \gamma_t \odot (1 + \tanh(W_g \phi_t))$$
$\phi_t$ 是相机位姿 $p_t$ 的投影特征，门控值控制状态中哪些历史信息被读取。

**检索触发器 $r_t$**：二值变量，通过sigmoid概率 $\hat{r}_t$ 参数化。训练时使用停止梯度策略强制模型在离散触发上检索，避免可微近似的模糊性。

**3D可见性重入监督**：离线计算三阶段标签：(1) 用前馈几何模型估计逐帧深度；(2) 将token反投影到3D点；(3) 用已知相机轨迹将前一帧3D点重投影到当前帧。若某3D点离开视锥（或被遮挡）后重新出现，则标记 $y_t=1$。损失函数为加权BCE：
$$\mathcal{L}_r = \frac{1}{|\mathcal{V}|} \sum_{t \in \mathcal{V}} \text{BCE}_{w_+}(\hat{r}_t, y_t), \quad w_+ = 3$$

**总损失**：$\mathcal{L} = \mathcal{L}_{\text{denoise}} + \lambda_r \mathcal{L}_r$，其中 $\lambda_r = 0.05$，去噪损失仅在目标帧latents上计算。

**模型集成**：在预训练相机可控视频扩散模型中，每 $n$ 个block替换前 $n-1$ 个自注意力层为L2R层，剩余保留softmax注意力。初始化时将查询/键/值/输出投影从原注意力继承，相机相关路径零初始化。

## 实验与结果
**评测基准**：
- **Echo-Memory**（内存容量评测）：使用 Wan2.1-T2V-1.3B 作为基础模型，含三个子任务：Replay（回放一致性）、In-domain Loop Closure（域内环路闭合）、Open-domain Return (O-V)（开放域返回泛化）。
- **自建 Revisit-Bench**：基于 RealEstate10K 和 DL3DV-10K 构建回环轨迹（palindrome far/near splits），评估视觉质量、相机可控性、重访一致性。

**关键结果**：
| 方法 | Replay PSNR↑ | Loop Closure SSIM↑ | O-V↑ |
|------|-------------|-------------------|------|
| VMem* | 13.71 | 0.352 | 64.73 |
| Context as Memory K=20 | 12.54 | 0.359 | 58.63 |
| **L2R-GLA** | **14.74** (+1.03dB) | **0.375** | **72.57** (+7.84) |
| L2R-GDN | 13.86 | 0.374 | 69.37 |

- **L2R-GLA** 在Echo-Memory全部Replay指标和Loop Closure SSIM、O-V上居首；PSNR较最佳基线VMem提升1.03dB，较Context as Memory K=20提升2.20dB。
- **视频质量**（Wan2.2底座）：L2R-GLA在VBench六维指标全部第一或第二；相机旋转误差从CaR的12.98°降至6.29°，平移误差从0.354降至0.300。
- **跨模型泛化**：在SANA-WM-2.6B上，L2R-GDN将回访PSNR从15.52提升至17.29，SSIM从0.5076提升至0.5597。
- **效率**：L2R-GLA/GDN每步推理耗时固定（~0.5s/step on H100），不随序列长度增长；相比softmax全历史注意力（90s时3.67s）和VMem几何索引重建（3.38s）显著更优。

## 相关工作脉络
1. **Context as Memory (Yu et al., 2025a)**：基于相机重叠选择K个历史帧作为外部上下文；L2R将其替换为模型内部的连续状态检索，无需离线选择固定数量帧。
2. **VMem (Li et al., 2025)**：使用Surfel索引进行几何检索；依赖显式3D重建管线，推理时需维护点云索引；L2R通过内化状态免除此开销。
3. **MemLearner (Yu et al., 2026a)**：用可学习查询token检索历史上下文，但检索通路与生成主干分离；L2R将检索嵌入注意力状态更新本身。
4. **CaR (Peng et al., 2026)**：在压缩上下文上添加专用检索注意力分支；引入额外计算开销且与生成动力学解耦；L2R无需新增分支。
5. **VideoSSM / ARL²**：利用状态空间作为记忆库但缺乏条件检索能力；L2R在此基础上增加相机条件门控和触发机制实现选择性访问。
6. **Wonder (Xu et al., 2026a)**：基于稀疏注意力top-k选择；缺乏几何相关性保证；L2R的相机条件门提供明确的位姿相关性依据。

## 局限性与未来方向
1. **静态场景假设**：当前重见监督依赖静态场景假设（3D点重投影匹配），动态场景中物体运动会导致标签噪声；作者明确将动态场景检索列为未来方向。
2. **线性注意力状态容量瓶颈**：ablation显示过高记忆层比例（7:1）导致信息丢失和性能下降，受限于线性状态的固定容量，需保留部分softmax层作为高带宽回退。
3. **离散触发不可微**：$r_t$ 采用停止梯度策略，无法端到端优化触发器的软决策，可能限制检索时机的精细调节。
4. **重见监督的计算开销**：三阶段3D几何标签计算（深度估计+反投影+重投影）在训练时需离线预处理，对大规模数据pipeline有一定负担。

## 研究启发与可借鉴点
1. **检索-生成内化范式**：将外部记忆检索迁移到模型内部状态更新的思路可推广至其他需要长程一致性的生成任务（如长序列文本生成、多模态时序建模）。
2. **几何监督信号的设计**：利用3D可见性重入作为离散触发监督是一种无需额外标注的自监督信号，可迁移至其他视觉记忆任务（如视觉定位、地图构建）。
3. **Log-space乘法门控**：在log空间进行门控计算保证数值稳定的技巧适用于任何需要 multiplicative gating 的状态空间模型扩展。
4. **相机位姿条件化检索**：将Plücker射线投影后接入门控的设计模式可复用于其他相机可控生成任务（如NEFA、4D生成）。
5. **混合注意力架构**：softmax层与线性注意力层按固定比例（3:1）混合的策略，为平衡长期记忆容量与短期细节保留提供了实用经验。

## 关键术语表
**Video World Model**：根据相机轨迹生成可探索3D一致场景视频的生成分布模型，需同时满足时序连贯性与空间一致性。
**Re-visibility Supervision**：基于3D几何的自监督信号，当历史3D点离开视锥后重新进入当前视野时触发，用于训练检索触发器。
**Gated Linear Attention (GLA)**：带门控机制的线性注意力变体，状态更新形式为 $S_t = \gamma_t S_{t-1} + v_t k_t^\top$，支持线性复杂度长序列建模。
**Gated DeltaNet (GDN)**：另一种状态空间架构，在GLA基础上引入delta修正项实现减法记忆门控，公式为 $S_t = \gamma_t(S_{t-1} - \beta_t S_{t-1} k_t k_t^\top) + \beta_t v_t k_t^\top$。
**Loop Closure**：相机轨迹回到初始或已观察过的位姿，用于评估模型保持长程场景一致性的能力。
**Echo-Memory Benchmark**：控制性记忆评测协议，包含Replay、In-domain Loop Closure、Open-domain Return三个子任务，用于量化视频世界模型的内存容量。
**Plücker Ray**：用6维向量表示的相机射线参数化形式（方向+力矩），用于统一编码相机位姿几何信息。
**State Transition Rule**：定义累积状态如何随新token更新的函数 $\varphi$，GLA和GDN是两种不同实现。

## 可复现要素
- **数据集**：Echo-Memory使用静态场景探索数据集（Yu et al., 2025a）；Revisit-Bench基于RealEstate10K和DL3DV-10K的RealCam-Vid数据构建（论文附录A.2详述构建流程）。
- **代码/权重**：论文声明"计划在接受后开源代码和权重文件"，当前未公开。
- **关键超参**：学习率 $5 \times 10^{-5}$（Wan2.1）/ $8 \times 10^{-5}$（Wan2.2/SANA-WM）；$\lambda_r = 0.05$；正类权重 $w_+ = 3$；层比 3:1（L2R:softmax）；chunk长度81帧；训练5k步，8×A100-80G或64×H20-96G。
