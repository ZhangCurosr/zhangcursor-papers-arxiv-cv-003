---
title: "Uncertainty-Aware-Trajectory-Forecasting-from-Imperfect-Trac"
source: https://arxiv.org/pdf/2608.30899v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 22:08:26"
field: "多智能体轨迹预测"
keywords: ["Trajectory Forecasting", "Tracking Uncertainty", "Uncertainty Quantification", "Knowledge Distillation", "Ornstein-Uhlenbeck Process", "Probabilistic Prediction"]
innovations: ["将观测轨迹建模为高斯状态并通过全方差定律融合定位不确定性与关联歧义", "结合时序相关OU噪声注入与响应式NLL蒸馏实现鲁棒训练", "仅替换嵌入层与输出头即可将不确定性感知能力插入SingularTrajectory/VISTA/MART等现有骨干"]
benchmarks: ["Oxford Town Centre", "VIRAT", "ETH/UCY (pseudo-detection protocol)"]
---

# 论文速读：Uncertainty-Aware-Trajectory-Forecasting-from-Imperfect-Tracking

## 一句话总结
本文针对由不完美的多目标跟踪器（MOT）输出的轨迹所进行的预测问题，提出了一种不确定性感知框架：将每个观测状态建模为高斯分布，其协方差通过全方差定律融合定位不确定性和关联歧义；结合Ornstein-Uhlenbeck时序噪声注入与响应式知识蒸馏，使预测模型在真实跟踪噪声下同时提升位移精度与概率可靠性。

## 研究问题与动机
- 现有轨迹预测方法大多假设历史轨迹是干净、完整且确定性的，但实际部署中观测轨迹由上游多目标跟踪器生成，存在定位抖动、漏检/重检、身份切换等问题。
- 传统做法通常通过去噪或删除不可靠观测来"修复"输入，但这会丢弃跟踪器本身携带的可靠性信息（如低置信度、歧义关联反映了遮挡或拥挤等困难场景）。
- 将跟踪不确定性显式编码并传播至预测器，比简单去噪更能利用原始信号中的可靠线索。
- 现有鲁棒预测方法（如NATRA、OosTraj、CaDeT）多将跟踪误差视为需消除的干扰，本文思路与之本质不同：将不确定性作为结构化信号供预测器自适应调节。

## 核心贡献（创新点）
1. **将轨迹预测建模为不确定性传播问题**：观测与预测状态均以高斯分布表示，而非确定性坐标点；与已有方法将轨迹视为清洁点的根本区别在于，输入可靠性本身被显式编码为协方差结构并传入预测骨干。
2. **基于全方差定律的跟踪不确定性分解**：将总协方差拆分为定位不确定性（检测级）与关联歧义（association级），前者由检测框尺寸与置信度构造，后者由多候选位置的空间分散度量，两项通过law of total variance合并；区别于直接用单一置信度或忽略关联来源的做法。
3. **不确定性感知训练策略**：结合由真实跟踪残差估计的Ornstein-Uhlenbeck时序相关噪声注入与响应式知识蒸馏（teacher在clean GT上训练，student在含噪概率输入上学习）；与i.i.d.高斯扰动或直接端到端在tracker输出上训练相比，兼顾了时间相关性与干净动力学目标。
4. **轻量可插拔的架构适配**：仅替换输入嵌入层（将(μx, μy, σx, σy, ρxy)投影到相同维度d）和输出头（二维坐标头改为五维高斯头），核心骨干（SingularTrajectory/VISTA/MART）保持不变；与需要大规模重写或专用去噪模块的方法相比，通用性更强。

## 方法详解
- **不确定性建模（Sec 3.1）**：
  - 对时刻t，跟踪器提供N个候选检测$\mathcal{D}_t=\{d_k\}$，每个候选具有位置$\mu_k$、框尺寸$(W_k, H_k)$、置信度$s_k$。
  - 通过softmax关联成本得到软权重$w_k = \exp(-c_k/\tau)/\sum_j \exp(-c_j/\tau)$（实验中$\tau=1$）。
  - 检测级定位协方差：$\Sigma_k = \text{diag}(W_k^2/(s_k+\epsilon), H_k^2/(s_k+\epsilon))$，低置信度产生更大方差。
  - 总协方差由全方差定律：$\Sigma_{\text{total}} = \underbrace{\sum_k w_k \Sigma_k}_{\Sigma_{\text{pos}}} + \underbrace{\sum_k w_k(\mu_k-\mu_{\text{total}})(\mu_k-\mu_{\text{total}})^\top}_{\Sigma_{\text{assoc}}}$，第一项为加权定位不确定性，第二项为关联歧义。
  - 部署时若MOT为黑盒，可通过Raw-IoU或Projected-IoU重构代理关联成本；图像坐标协方差经雅可比投影至地面平面：$\Sigma_{\text{ground}} = J_h \Sigma_{\text{image}} J_h^\top$。

- **输入嵌入与输出头（Sec 3.2–3.3）**：
  - 每个观测编码为5维$(\mu_x, \mu_y, \sigma_x, \sigma_y, \rho_{xy})$，经$\phi_{\text{prob}}$投影到与原始骨干一致的$d$维，实现即插即用。
  - 输出头将每个未来样本k在每个预测步t映射为$\mathcal{N}(\mu_{\text{pred},t}^{(k)}, \Sigma_{\text{pred},t}^{(k)})$；保留骨干原有的生成机制（如扩散采样）。
  - 监督损失为负对数似然（NLL）：$\mathcal{L}_{\text{NLL}} = \frac{1}{2}\sum_t \left(\log|\Sigma_t| + (y_t-\mu_t)^\top \Sigma_t^{-1}(y_t-\mu_t)\right)$。

- **噪声注入（Sec 3.4）**：
  - 对每行人i从真实跟踪残差KDE估计的分布中采样$\sigma_i$，再用OU过程生成时序相关噪声：$\mathbf{n}_{i,t+1} = \mathbf{n}_{i,t} - \theta(\mathbf{n}_{i,t}-\mu\mathbf{1}_2)\Delta t + \sigma_i\sqrt{\Delta t}\cdot\xi_{i,t}$，其中$\theta=0.15, \mu=0$。

- **知识蒸馏（Sec 3.5）**：
  - Teacher用GT轨迹（加小各向同性噪声$\sigma=0.1$m模拟标注噪声）训练后冻结；Student接收含噪概率输入。
  - 总损失：$\mathcal{L}_{\text{total}} = \lambda_{\text{traj}}\mathcal{L}_{\text{NLL}}(Y_{\text{GT}}|S) + \lambda_{\text{dist}}\mathcal{L}_{\text{consist}}(\mu^T|S)$。
  - 一致性项采用NLL而非KL散度：仅以Teacher均值$\mu^T$为伪目标，避免强迫Student复制Teacher的协方差（Student需额外覆盖输入噪声带来的不确定性）。

## 实验与结果
- **数据集**：Oxford Town Centre (OTC)、VIRAT（使用真实MOT输出，80%/20%时序划分）；ETH/UCY使用补充材料中描述的pseudo-detection协议（非标准benchmark比较）。
- **骨干网络**：SingularTrajectory、MART、VISTA，每个生成$K=20$条样本轨迹。
- **评估配置**（5种）：Baseline（纯GT训练）/Kalman Filtering/Tracker-Specific Adaptation/Noisy GT（OU噪声）/Distillation（完整方案）。
- **关键结果**（Table 1，NLL更优为负值，ADE/FDE越小越好）：
  - OTC + VISTA：Distill配置NLL=2.30（最低），minADE20=0.56，minFDE20=0.54，95%覆盖率86.2%，显著优于Baseline（NLL=41.47, ADE=1.57, FDE=1.77）和Kalman（NLL=14.54, ADE=0.69, FDE=0.82）。
  - OTC + MART：Distill NLL=-15.56，minADE20=0.51，minFDE20=0.77，95%覆盖率97.5%。
  - OTC + SingularTrajectory：Distill NLL=-10.99，minADE20=0.54，minFDE20=0.89，95%覆盖率95.0%。
  - VIRAT + VISTA：Distill NLL=1.39，minADE20=0.52，minFDE20=0.42，95%覆盖率95.9%。
  - ETH/UCY + VISTA：Distill minADE20=0.47，minFDE20=0.83，95%覆盖率92.5%。
- ** Ablation**（Table 2/3）：
  - 输入/输出概率化均带来提升，P/P组合最优（MART: 0.58/0.89/AUC 1.22）。
  - OU噪声优于i.i.d.高斯噪声（MART: 0.58/0.89 vs 0.60/0.94）；NLL蒸馏优于KL蒸馏（VIRAT MART: 0.43/0.70 vs 0.56/0.96）。
  - IoU代理关联成本与BoT-SORT内部成本表现相当，表明无需依赖特定tracker内部信号。

## 相关工作脉络
1. **Clean observation预测**（Social LSTM、AgentFormer、MART、VISTA、SingularTrajectory等）：假设输入轨迹精确完整，本文聚焦其未覆盖的"输入不可靠"维度。
2. **从原始视频/检测直接预测**（Weng et al. [36,37]、Yu & Zhou [41]、Zhang et al. [45]）：绕过显式跟踪，本文仍使用MOT输出但显式传播不确定性，思路不同。
3. **去噪型鲁棒预测**（NATRA [25]、OosTraj [43]、CaDeT [33]）：将跟踪误差视为需消除的干扰；本文将其作为可靠性信号保留并传播。
4. **跟踪不确定性估计**（BoT-SORT [1]、StrongSORT [11]）：提供定位/关联信号来源，本文在其输出之上构建高斯不确定性接口。
5. **LUPI与知识蒸馏**（Vapnik & Vashist [35]、Lopez-Paz et al. [27]）：本文在轨迹预测场景中应用response-based distillation，利用GT作为privileged information指导含噪输入下的学习。

## 局限性与未来方向
- 明确排除对缺失观测（missed detections）的恢复与身份切换（identity switch）后的重建，仅处理已有观测的不确定性传播。
- 定位协方差采用基于框尺寸与置信度的启发式代理（eq. 3），非校准的探测器后验；可替换为更精确的不确定性估计但不影响框架其余部分。
- 仅考虑二阶矩（高斯近似），未建模多峰关联结构；在高度歧义场景下可能过于简化。
- 评估中ETH/UCY采用pseudo-detection协议而非标准clean GT benchmark，限制了与主流文献的直接可比性。
- 未来可扩展至显式处理身份重新识别、跨视角不确定性传播，以及联合检测-关联-预测的端到端训练。

## 研究启发与可借鉴点
1. **不确定性作为信号而非噪声**：将跟踪器的低置信度/多候选作为可靠性 cue显式传入下游任务，而非盲目去噪，这一思想可迁移至任何依赖感知输出的决策管线（如自动驾驶中的意图预测、ROS导航中的避障规划）。
2. **OU过程模拟时序相关感知误差**：用Empirical KDE + Ornstein-Uhlenbeck离散化生成带时间相关性的训练噪声，比i.i.d.扰动更贴近真实滤波器（如Kalman）的行为，适用于各类时序模型的噪声增强。
3. **响应式NLL蒸馏而非KL**：用Teacher均值作伪目标并以NLL计算一致性，避免强制Student匹配Teacher协方差——这对"teacher有clean先验、student需表达额外输入不确定性"的场景通用。
4. **即插即用的5维高斯嵌入**：将$(\mu_x, \mu_y, \sigma_x, \sigma_y, \rho_{xy})$投影到与原骨干同维的隐空间，不改变核心架构即可适配多种现有预测器，便于快速验证不同backbone的鲁棒性。
5. **黑盒部署的代理关联成本**：当MOT不暴露内部cost矩阵时，通过IoU（Raw/Projected）重建软关联权重，使方法在实际系统中仍可用。

## 关键术语表
- **Multi-Object Tracking (MOT)**：在多目标追踪中同时估计场景中多个目标的轨迹与身份。
- **Law of Total Variance**：将总方差分解为组内期望的方差（定位不确定）与组间期望的方差（关联歧义）。
- **Ornstein-Uhlenbeck (OU) Process**：一种具有均值回归特性的随机过程，用于生成时序相关（有色）噪声。
- **Learning Using Privileged Information (LUPI)**：Vapnik提出的学习范式，训练时利用测试时不可用的额外信息（privileged data）指导模型学习。
- **Negative Log-Likelihood (NLL)**：对高斯预测分布下真实标签的对数概率取负，用作回归/不确定性预测的监督损失。
- **MinADE / MinFDE**：在K条采样轨迹中取最佳一条的历史平均欧氏距离（ADE）和末端欧氏距离（FDE）。
- **Response-based Distillation**：以教师模型的输出均值（而非特征或完整分布）作为学生学习的伪目标。
- **Association Ambiguity**：多目标追踪中因多个候选检测相似而难以确定正确匹配的歧义程度。

## 可复现要素
- **数据集**：Oxford Town Centre (OTC)、VIRAT、ETH/UCY（pseudo-detection协议见补充材料）；OTC与VIRAT的跟踪输出来自BoT-SORT（主实验）、ByteTrack（补充）；ETH/UCY无公开bounding box标注，使用伪检测协议。
- **代码/权重**：论文未明确声明开源代码或权重。
- **关键超参**：关联温度$\tau=1$；OU过程$\theta=0.15, \mu=0$；Teacher输入各向同性噪声$\sigma=0.1\text{m}$；输出样本数$K=20$；损失权重$\lambda_{\text{traj}}, \lambda_{\text{dist}}$论文未给出具体数值（"unless otherwise specified"）。
