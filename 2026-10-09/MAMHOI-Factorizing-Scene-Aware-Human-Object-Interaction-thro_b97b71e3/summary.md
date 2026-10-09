---
title: "MAMHOI-Factorizing-Scene-Aware-Human-Object-Interaction-thro"
source: https://arxiv.org/pdf/2610.12416v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 17:21:05"
field: "人体-物体交互生成"
keywords: ["Human-Object Interaction", "Scene-aware Motion Generation", "Affordance", "Diffusion Model", "3D Human Motion", "Factorized Generation"]
innovations: ["通过运动affordance解耦场景可行性推理与交互动力学生成，分别利用HSI和HOI数据独立训练", "设计affordance-aware排名损失L_rank，通过正负运动对SDF对比提供密集相对监督", "将局部场景体素规范到人体坐标系，显著提升场景感知运动生成质量"]
benchmarks: ["OMOMO", "LINGO", "TRUMANS"]
---

# 论文速读：MAMHOI: Factorizing Scene-Aware Human-Object Interaction through Affordances

## 一句话总结
MAMHOI 提出了一种通过运动 affordance（可供性）解耦场景理解与人体-物体交互运动生成的框架，将场景-aware 的 HOI 生成分为两个独立学习阶段，分别利用人体-场景数据和纯 HOI 数据，无需配对的人体-物体-场景数据即可生成物理可行且符合场景约束的交互运动。

## 研究问题与动机
1. **场景感知 HOI 生成需要两项互补能力**：一是推理交互在环境中的可行性（场景适应），二是合成真实的人体-物体运动（交互动力学），但当前缺乏大规模联合监督数据。
2. **现有 HOI 数据集与 HSI 数据集存在分布鸿沟**：人体-场景数据集（如 LINGO）提供环境感知运动先验，人体-物体数据集（如 OMOMO）捕捉详细交互动力学，但两者的联合配对数据极其稀缺。
3. **已有方法无法很好衔接两类监督**：现有方法或通过外部规划路点传递场景约束（如 CHOIS），或构建合成数据联合训练（如 UniHM、InfBaGel），但均未解决如何在分布差异较大的两种监督信号之间建立有效接口的核心建模问题。
4. **场景几何适配与交互动力学需要分离学习**：直接将场景约束耦合到 HOI 生成器会导致场景可行性与交互先验之间的权衡，需要一种中间表示来桥接两者。

## 核心贡献（创新点）
1. **提出 affordance 介导的因子化生成框架**：将场景级交互可行性与详细人体-物体运动合成分离，分别利用 HSI 数据和 HOI 数据的互补监督，无需配对的人体-物体-场景数据。
2. **将运动 affordance 建模为场景推理与交互合成的显式接口**：affordance 以 3D 局部体素网格形式编码可行空间区域，传递场景约束但不限制具体轨迹，保持运动生成的多样性。
3. **设计了 affordability-aware 排名损失（L_rank）**：通过正反运动对的 SDF 对比提供密集相对监督，有效区分相似端点但不同中间轨迹的运动模式。
4. **在复杂室内场景中显著降低物体-场景穿透率**：实验表明 MAMHOI 在保持高质量 HOI 动力学的同时，降低了人体-场景和物体-场景穿透，改善了物体路径跟随性和交互接触率。

## 方法详解
**整体框架**：MAMHOI 将场景感知 HOI 生成因子化为 $p(\mathbf{x}|c_{task}, c_{scene}) \approx \int p(\mathbf{x}|\mathbf{A}, c_{task}) p(\mathbf{A}|c'_{task}, c_{scene}) d\mathbf{A}$，其中 $\mathbf{A}$ 为序列级运动 affordance。

**模块一：场景理解模型（Scene Understanding Model）**
- 输入：2D 场景深度图 $\mathbf{S}_{depth}$、规划器提供的物体路径 P、文本指令 $\mathbf{e}_{text}$。
- 结构：残差 2D U-Net + 路径编码器（MLP）+ CLIP 文本编码器，在 U-Net bottleneck 处通过多头交叉注意力融合多模态条件。
- 输出：阈值化后得到二值 2D affordance 掩码 $\hat{\mathbf{A}}$，再提升为 3D 场景坐标表示。
- 训练损失：$\mathcal{L}_{aff} = \mathcal{L}_{x_0} + 0.1\mathcal{L}_{BCE} + 0.2\mathcal{L}_{path}$，其中 BCE 损失对路径附近像素施加距离衰减权重，path loss 显式鼓励规划路径落在 affordance 区域内。

**模块二：Affordance 接口**
- 将预测的 2D affordance 提升为 3D 后，在每个自回归窗口查询三个局部 3D 体素网格：$\mathbf{V}^{pelvis}, \mathbf{V}^{obj}, \mathbf{V}^{goal} \in \{0,1\}^{32\times32\times32}$，分别以人体骨盆、当前物体位置和物体目标为中心。
- 通过 ViT-based Affordance-voxel encoder（6 层 Transformer，8 头注意力）编码为 512 维特征 $\mathbf{e}_{aff}$。
- 体素坐标系采用人体局部规范坐标（canonical human frame），而非全局场景坐标，以更一致地表示邻近几何。

**模块三：Affordance 条件化的运动扩散主干**
- 扩散模型预测干净的人体-物体运动 $\hat{\mathbf{x}}_0$，输入包括噪声运动、稀疏运动条件 $\mathbf{C}_{motion}$、文本、affordance 特征和物体几何（BPS 描述符）。
- 交叉注意力 adapter 将 affordance 特征注入运动骨干网络。
- 自回归生成：30 帧窗口、10 帧重叠，每段生成后更新参考点并重新查询局部 affordance。
- 训练损失：$\mathcal{L}_{total} = 1.0\mathcal{L}_{diff} + 0.5\mathcal{L}_{fk} + \mathcal{L}_{scene}$，其中场景损失 $\mathcal{L}_{scene} = 0.2\mathcal{L}_{sdf} + 0.1\mathcal{L}_{rank}$。

**Affordance-aware 排名损失**：构造正负运动对 $(\mathbf{x}^+, \mathbf{x}^-)$，要求正运动的 SDF 惩罚比负运动低至少 margin $m=0.2$：$\mathcal{L}_{rank} = [m + \mathcal{L}_{sdf}(\mathbf{x}^+, \mathbf{A}^+) - \mathcal{L}_{sdf}(\mathbf{x}^-, \mathbf{A}^+)]_+$。

**测试时引导**：保留 hand-object contact 和 feet-floor guidance；额外提供可选的 scene-SDF guidance $\mathcal{G}_{sdf}$，直接在采样阶段惩罚场景穿透。

## 实验与结果
**数据集**：场景理解模型训练于 LINGO，运动扩散主干训练于 OMOMO；两者均使用宽松的 AABB 区域构建 affordance 监督标签。

**评估设置**：
- 交互-only 评估：OMOMO 验证集（534 对，120 帧，12 种可动物体）。
- 场景感知评估：基于 LINGO 测试集构建的物理测试集（20 个场景 × 5 种物体 × 164 条路径 ≈ 820 次试验）。

**主要结果**（场景感知测试集）：

| 方法 | 人体-场景穿透率 $R^{frm}$ | 物体-场景穿透深度 $\bar{D}^{vtx}$ | 接触率 $R_{con}$ |
|------|--------------------------|----------------------------------|-----------------|
| CHOIS | 29.645% | 0.545 | 90.97% |
| InfBaGel | 31.848% | 3.136 | 92.39% |
| **Ours** | **14.990%** | **0.403** | **94.87%** |
| Ours + G_sdf | 13.745% | 0.398 | 95.69% |

- MAMHOI 无需测试时引导即优于所有基线，物体-场景穿透深度较 CHOIS 降低 26%（0.403 vs 0.545），人体-场景穿透率较 CHOIS 降低 49%。
- 交互-only 评估中 FID=1.8361，R-precision top-1=0.7764，Diversity=8.6923，优于 CHOIS（FID=2.7892）。
- 用户研究（884 票）：MAMHOI 以 86.7% 得票率优于 CHOIS，92.3% 优于 CHOIS+G_sdf。

## 相关工作脉络
1. **CHOIS [16]**：语言/初始状态/物体几何驱动的 HOI 生成，通过外部规划物标路点传递场景约束，未显式建模可行交互区域。MAMHOI 以 affordance 替代稀疏路点，提供更丰富的场景几何感知。
2. **InfBaGel [31]**：混合训练策略结合合成 HOI-场景数据和真实 HSI 数据，存在异构分布间的权衡问题。MAMHOI 避免联合拟合，通过 affordance 接口分离学习。
3. **HOSIG [28]**：分层场景感知 grasp-pose 生成 + 启发式导航，依赖配对 HOI-场景数据，物体覆盖有限。MAMHOI 无需配对数据且支持更多样物体。
4. **UniHM [4]**：通过将已有运动放置到无障碍区域构建合成数据，主要鼓励识别可行区域，但路径条件不保证精确轨迹跟随。MAMHOI 通过 affordance 体素直接编码局部空间约束。
5. **Afford-Motion [23]**：定义连续人体-关节到场景表面的距离场作为 affordance，用于 HSI。MAMHOI 的 affordance 编码的是联合人机-物体的可行空间支撑，而非单纯身体-场景邻近度。
6. **ZeroHSI [14]**：零样本 4D 人-场景交互视频生成。MAMHOI 聚焦 3D 运动生成的物理可行性，两者在任务设定和表征层面各有侧重。

## 局限性与未来方向
1. **依赖规划器提供的物体路径**：当前方法使用 A* 规划生成粗粒度物体级结构线索，若路径规划质量差可能影响 affordance 预测。
2. **affordance 表示仅覆盖 2D 平面扩展**：体素查询范围水平 2.4m，垂直维度有限，对高/低位置交互可能受限。
3. **测试时 SDF 引导虽减少穿透但不提升感知质量**：用户研究揭示后处理几何修正与学习到的场景适配是正交机制，如何更好融合值得探索。
4. **自回归窗口滑动引入累积误差**：30 帧窗口、10 帧重叠的生成策略在长时程交互中可能累积漂移。
5. **未扩展到动态场景**：当前方法仅处理静态室内环境，动态障碍物场景是重要扩展方向。

## 研究启发与可借鉴点
1. **解耦因子化策略具有普适迁移价值**：将"场景可行性推理"与"交互动力学生成"通过中间 affordance 接口分离，可迁移到其他需要多源监督的生成任务（如机器人操作、虚拟角色动画）。
2. **affordance-aware 排名损失的对比学习设计**：通过正负运动对与 SDF 惩罚的排名对比提供密集相对监督，这种设计可推广到其他 motion generation 任务中以增强物理可行性。
3. **规范坐标对齐技巧**：将局部场景体素转换到人体规范坐标系（canonical human frame）而非直接使用全局场景坐标，显著改善场景感知生成质量，这对其他场景条件化生成任务有借鉴意义。
4. **弱 classifier-free guidance 的使用**：对 affordance 条件使用弱引导（s∈[1.0,1.3]）在场景适配与运动分布保真之间取得平衡，这一策略可用于其他多条件扩散模型。
5. **数据构建策略**：通过宽松 AABB 区域构建 affordance 监督标签，避免了精细标注需求，为类似任务的弱监督信号构建提供了范例。

## 关键术语表
**Motion Affordance（运动可供性）**：编码人体-物体交互可在其中 feasible 执行的三维空间支撑区域，作为场景推理与运动生成的中间接口表示。
**Factorization（因子化解耦）**：将联合分布分解为场景到 affordance 和 affordance 到运动两个独立因子，分别利用不同数据源进行监督学习。
**SDF（Signed Distance Field，符号距离场）**：将 affordance 掩码转换为截断符号距离场，作为运动可行性的绝对锚点和 ranking 损失的计算基础。
**Canonical Coordinate Frame（规范坐标系）**：基于初始人体朝向定义的人体局部坐标系，用于消除全局方向歧义并统一运动与场景表示。
**Autoregressive Segmented Generation（自回归分段生成）**：将长时程交互分解为 30 帧窗口逐段生成，每段初始化为下一段的起点。
**Basis Point Set (BPS) Descriptor**：编码物体几何的 1024×3 点集描述符，用于向扩散模型提供物体形状条件。

## 可复现要素
- **数据集**：LINGO [12]（场景理解训练）、OMOMO [15]（运动生成训练）；场景感知测试集为作者基于 LINGO 测试集自行构建（20 场景，164 条路径，约 820 次试验），**非标准公开基准**。
- **代码开源**：项目页面 https://leimingyuan.github.io/MAMHOI-project-page/，论文声明代码将在 acceptance 后开源。
- **权重**：论文未提及预训练权重公开。
- **关键超参**：场景理解学习率 $1\times10^{-4}$，batch=32，500 epoch；运动生成学习率 $2\times10^{-4}$，batch=128，450,000 steps，EMA decay=0.995；扩散步数 1,000；窗口 30 帧/重叠 10 帧；classifier-free guidance strength s∈[1.0, 1.3]。
