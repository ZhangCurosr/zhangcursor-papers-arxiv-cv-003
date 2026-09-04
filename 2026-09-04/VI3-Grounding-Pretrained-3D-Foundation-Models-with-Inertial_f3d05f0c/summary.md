---
title: "VI3-Grounding-Pretrained-3D-Foundation-Models-with-Inertial"
source: https://arxiv.org/pdf/2609.03824v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 11:55:17"
field: "3D视觉与多传感器融合"
keywords: ["3D foundation models", "visual-inertial estimation", "test-time refinement", "metric scale recovery", "IMU preintegration"]
innovations: ["提出VI3框架，首次在不依赖GT的情况下将IMU度量信号锚定至任意预训练3DFM", "设计零GT惯性状态初始化方法，结合TLS尺度和重力共识机制", "提出TTR度量注入机制，通过轻量级测试时微调恢复物理一致的重建尺度"]
benchmarks: ["TartanAirV2", "EuRoC MAV", "UZH-FPV"]
---

# 论文速读：VI3: Grounding Pretrained 3D Foundation Models with Inertial Cues

## 一句话总结
VI3是一种模型无关的测试时锚定框架，利用机载IMU数据为预训练3D基础模型（3DFM）恢复真实的度量尺度，无需地面真值监督即可实现物理一致的轨迹和深度重建。

## 研究问题与动机
- **3DFM的度量尺度缺失问题**：DUSt3R、VGGT、π³等3D基础模型从单目图像预测的几何仅在未知全局尺度因子内准确，无法输出真实世界尺寸的重建结果。
- **现有方法的数据驱动局限**：AMB3R、Pi3X、Depth Anything 3等度量感知3DFM通过 learned priors 回归尺度，但这些先验继承自训练分布，在分布外场景（如低纹理、极端运动）下不可靠。
- **输入条件注入的物理信号淹没问题**：MapAnything等模型引入scale token或相机先验条件化，但强数据驱动先验可能覆盖显式度量信号，且依赖精确相机标定。
- **视觉-惯性融合的系统缺口**：现有视觉-惯性系统（如MASt3R-Fusion、VIGS-SLAM）将IMU用于外部优化后端，预训练权重不被适配；VI3填补了"零样本3DFM + 物理度量锚定"的空白。

## 核心贡献（创新点）
1. **首个推理时融合密集3DFM先验与惯性度量锚定的模型无关框架**：区别于依赖优化后端或外部fine-tuning的工作，VI3仅用机载IMU即可物理锚定任意3DFM，无需GT监督。
2. **零GT的惯性状态初始化方法**：通过3DFM预测的相对旋转闭式估计陀螺仪偏置，利用总最小二乘（TLS）约束恢复初始速度和全局尺度，结合视觉/惯性双重种子迭代求解重力方向并建立共识机制。
3. **测试时微调（TTR）度量注入机制**：引入单一可学习标量α缩放平移和深度，配合pose监督、重投影深度一致性损失及深度-初始锚定正则，仅需解冻camera/depth heads和最后L层interframe attention blocks即可适配场景。
4. **适配多样3DFM架构的灵活锚定策略**：对支持可选输入的模型（如Pi3X、DA3）采用IMU-pose conditioning前向注入；对不支持的模型（如VGGT、π³）采用TTR后向适配，验证了不同架构下的泛化能力。

## 方法详解
**整体架构**：VI3包含视觉分支（预训练3DFM前向推理）和惯性分支（IMU预积分），后者生成scene-specific目标轨迹作为监督信号。

**惯性状态初始化**：
- **陀螺仪偏置估计**：基于相对旋转 $\Delta \mathbf{R}_i^{\mathcal{M}}$ 与平均角速度 $\bar{\omega}_i$ 的闭式解，假设小角度旋转：
  $$b_i^g \approx -\frac{1}{\Delta t} \text{Log}\left(\text{Exp}(\bar{\omega}_i \Delta t)^\top \text{Exp}\left(\frac{1}{m}\text{Log} \Delta \bar{\mathbf{R}}_i^{\mathcal{M}}\right)\right)$$
  对 $n-1$ 对帧取平均得 $\hat{b}^g$。
- **初始速度与尺度估计**：由3DFM poses计算up-to-scale速度 $\mathbf{v}_i^{\mathcal{M}} = \Delta \mathbf{p}_i^{\mathcal{M}}/\Delta T$，利用速度增量约束 $s\Delta \mathbf{v}_i^{\mathcal{M}} = \Delta \mathbf{v}_i$ 构建总最小二乘问题：
  $$\hat{s}_{\text{init}} = \text{sign}(\langle \mathbf{V}, \mathbf{V}^{\mathcal{M}} \rangle) \frac{\|\mathbf{V}\|}{\|\mathbf{V}^{\mathcal{M}}\|}$$
  再通过中值反投影细化初始速度幅值 $\mu$。
- **重力估计与共识**：并行使用GeoCalib视觉估计 $\mathbf{g}_{\text{gc}}$ 和惯性粗估计 $\mathbf{g}_{\text{imu}}$，结合运动学最小二乘 $\mathbf{g}_{\text{ls}}$ 通过不动点迭代耦合尺度与重力；当视觉重力偏差超阈值（$\gamma_{\min}=20°$）时切换至纯惯性解。

**测试时微调（TTR）损失函数**：
- **姿态监督**（全局刚性变换不变）：
  $$\mathcal{L}_{\text{pos}} = \frac{1}{n-1}\sum_{i=2}^{n} \|(\mathbf{p}_i^{\mathcal{M}} - \mathbf{p}_{i-1}^{\mathcal{M}}) - \Delta \mathbf{p}_{i-1}\|_1$$
  $$\mathcal{L}_{\text{rot}} = \frac{1}{n-1}\sum_{i=2}^{n} \|\text{Log}((\mathbf{R}_{i-1}^{\mathcal{M}\top}\mathbf{R}_i^{\mathcal{M}})^\top \Delta \mathbf{R}_{i-1})\|_2$$
- **深度一致性损失**（无GT情况下的自监督多视角约束）：
  $$\mathcal{L}_{\text{dcons}} = \frac{1}{n}\sum_{t=1}^{n} \min_{i\neq j} |[\pi_i(\mathcal{X}_j^{\mathcal{M}})]_z - \mathcal{D}_i^{\mathcal{M}}(\pi_i(\mathcal{X}_j^{\mathcal{M}}))|_{\text{valid}}$$
- **锚定正则**：帧1位置锚定 $\mathcal{L}_a^p = \|\hat{\mathbf{p}}_1^{\mathcal{M}}\|_1$ 和深度-初始锚定 $\mathcal{L}_a^D = \frac{1}{n}\sum_i \|\mathcal{D}_i^{\mathcal{M}} - \mathcal{D}_i^{\mathcal{M}(0)}\|_1$ 防止几何失真。
- **总目标**：$\mathcal{L}(\theta) = w_r \mathcal{L}_{\text{rot}} + w_p \mathcal{L}_{\text{pos}} + w_d \mathcal{L}_{\text{dcons}} + w_{a_p} \mathcal{L}_a^p + w_{a_d} \mathcal{L}_a^D$，超参 $w_r=20, w_p=10, w_d=10, w_{a_p}=1000, w_{a_d}=1$，AdamW优化160步。

## 实验与结果
**数据集**：TartanAirV2（合成、高视差）、EuRoC（真实、中度运动）、UZH-FPV（真实、高动态低视差）；每序列采样$n=8$帧窗口、stride=2，取3个clip中位数。

**基线模型**：Up-to-scale模型 VGGT、VGGT-Ω、$\pi^3$；Metric模型 Pi3X、Depth Anything 3 (DA3)。

**关键结果**：
- **Up-to-scale模型TTR效果显著**（Table 1）：VGGT在TartanAirV2上Trans从1.130m降至0.358m，Scale(Traj.)从0.14恢复至1.05，$\delta<1.25$从0.000提升至0.515。
- **Metric模型在OOD场景仍可改善**（Table 2）：Pi3X在UZH-FPV上Rot从0.933°降至0.233°（TTR），Scale(Traj.)从1.11趋近1.00。
- **初始化精度**（Table 3）：重力估计误差在1.05°–6.61°，EuRoC陀螺偏置相对误差11.3%。
- **初始化敏感性**：初始速度方向/幅值误差对最终尺度影响最大，重力/偏置噪声鲁棒性较好。
- **对比输入条件化**：TTR普遍优于IMU-cond，尤其在数据驱动先验淹没输入信号的场景（如DA3在某些配置下IMU-cond反而劣化）。

## 相关工作脉络
- **Feedforward 3DFMs**（DUSt3R、VGGT、π³）：本文扩展对象，这些模型预测up-to-scale几何，缺乏度量锚定能力。
- **Scale-aware 3DFMs**（Pi3X、DA3、MapAnything、AMB3R）：通过学习prior回归尺度，本文论证其OOD鲁棒性不足，物理测量更可靠。
- **Visual-inertial SLAM**（ORB-SLAM3、VINS-Mono）：稀疏重建+状态后端优化，与VI3的密集前向+测试时适配形成对比。
- **视觉-惯性3DFM融合**（MASt3R-Fusion、VIGS-SLAM）：将IMU接入因子图或3D Gaussian优化，预训练权重冻结；VI3通过TTR适配权重。
- **Test-time Training/Adaptation**（ViT³、TTT3R、ZipMap）：本文定位——将TTR用于3DFM的度量尺度恢复而非仅分布偏移鲁棒性。
- **IMU预积分与状态初始化**（Forster et al.、Cerezo et al.）：经典VI初始化理论，本文将其适配于3DFM输出条件。

## 局限性与未来方向
- **运动视差可观测性限制**：静止或纯旋转场景下IMU无法恢复尺度，这是fundamental限制而非估计失败。
- **初始速度估计敏感性**：速度幅值和方向误差直接传导至尺度恢复，低视差帧误差放大。
- **短序列窗口约束**：$n=8$帧的上下文限制了对长序列漂移的校正能力。
- **计算成本**：TTR在RTX 5090上单次clip约48–128秒，难以实时应用。
- **未来方向**：扩展至GNSS、轮式里程计等多源度量信号融合；接入下游任务（前向合成、实时建图）；探索foundation models的物理世界 specialization。

## 研究启发与可借鉴点
1. **总最小二乘（TLS）用于双噪声变量尺度估计**：VI3将视觉up-to-scale速度与惯性速度增量均视为含噪观测，采用TLS替代OLS消除回归衰减偏差，对类似多源度量对齐任务有借鉴价值。
2. **多源重力估计共识机制**：视觉GeoCalib + 惯性粗估计 + 运动学LS的三元共识设计，为单目重力校准提供了鲁棒fallback策略。
3. **测试时微调的轻量适配设计**：仅解冻camera/depth heads和最后L层attention blocks（L=8），配合标量α缩放，实现了"高适应力+低计算开销"的平衡，可迁移至其他foundation模型的领域适配。
4. **无GT深度一致性损失的构造**：通过重投影匹配构建自监督约束，在缺乏3D标注场景下维持几何一致性，适用于任意前向3D模型的测试时改进。
5. **模型无关的尺度注入范式**：区分"支持可选输入的模型"（前向conditioning）和"不支持的模型"（后向TTR），为foundation model的硬件部署提供了通用接口设计思路。

## 关键术语表
- **3D Foundation Models (3DFMs)**：在大规模图像集上预训练的端到端前向3D重建模型（如VGGT、π³），可从无序图像直接预测相机位姿和密集深度。
- **Up-to-scale geometry**：仅恢复几何形状和相对尺度，但全局尺度因子未知的3D重建结果，常见于基于SfM标签训练的模型。
- **IMU Preintegration**：对两帧间IMU测量值进行积分，计算相对旋转、位移和速度变化，避免重复积分计算。
- **Test-Time Refinement (TTR)**：在推理阶段微调预训练模型的部分参数（而非全量fine-tuning），使其适配当前输入分布或满足额外约束。
- **Total Least Squares (TLS)**：当自变量和因变量均含噪声时使用的最小二乘变体，VI3用于同时含噪声的视觉速度和惯性速度尺度估计。
- **Gravity Consensus**：通过视觉GeoCalib、惯性粗估计和运动学LS三种独立方法估计重力方向，利用角度一致性检测异常并fallback。
- **Scale (Traj.) / Scale (Depth)**：评估指标，分别为预测轨迹长度与GT的比值、预测深度与GT的比值，理想值为1。

## 可复现要素
- **数据集**：TartanAirV2（公开）、EuRoC MAV（公开）、UZH-FPV（公开）
- **代码**：项目主页 https://ernestolozano.github.io/vi3（论文未明确说明GitHub链接）
- **预训练权重**：使用官方发布的VGGT、VGGT-Ω、π³、Pi3X、DA3权重
- **关键超参**：窗口帧数$n=8$，stride=2，优化步数$K=160$，解冻block数$L=8$，损失权重 $w_r=20, w_p=10, w_d=10, w_{a_p}=1000, w_{a_d}=1$
- **硬件**：NVIDIA RTX 5090
