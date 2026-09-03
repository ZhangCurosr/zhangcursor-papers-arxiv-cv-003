---
title: "VIPS-Vehicle-Infrastructure-Cooperative-Planning-Benchmark-v"
source: https://arxiv.org/pdf/2609.02462v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 21:35:43"
field: "自动驾驶协同规划评测"
keywords: ["Vehicle-Infrastructure Cooperative Planning", "Pseudo-Simulation", "End-to-End Autonomous Driving", "V2X", "Sparse Representation", "Benchmark"]
innovations: ["将两阶段伪仿真扩展至V2I协同场景，实现可扩展且贴近闭环的规划鲁棒性评估", "提出CoS-V2X稀疏锚实例框架，在降低28.6%通信开销的同时取得最优EPDMS", "构建带矢量地图标注的V2X-Real规划数据集与新颖视图合成管线"]
benchmarks: ["VIPS (Vehicle-Infrastructure Pseudo-Simulation)", "V2X-Real with Vector Map Annotations"]
---

# 论文速读：VIPS: Vehicle-Infrastructure Cooperative Planning Benchmark via Pseudo-Simulation

## 一句话总结
本文提出 **VIPS**，首个面向车路协同（V2I）端到端自动驾驶规划的**伪仿真（pseudo-simulation）评测基准**，通过两阶段真实/合成观测评估弥补了开放循环与封闭循环评估之间的鸿沟；同时设计了基于稀疏锚表示的协同规划方法 **CoS-V2X**，在更低通信开销下实现最优规划性能。

## 研究问题与动机
- 端到端自动驾驶在城市复杂路口面临部分可观测性与多智能体交互挑战，单车系统受限于视野盲区；V2I 基础设施可提供全局视角，但现有工作多聚焦于感知增强而非端到端规划融合。
- 现有开放循环评估（对比专家轨迹）无法捕捉误差累积与恢复行为，与真实表现相关性弱；封闭循环评估（如 CARLA）保真度高但成本高、难以扩展，且存在仿真域偏移问题。
- 现有伪仿真方法（如 NAVSIM v2）主要针对单车场景，无法处理 V2I 中异构观测与车-路信息交互的复杂调度。
- 现有 V2X 评测基准仍受限于单一范式：真实数据集仅支持开放循环，模拟器支持闭环但缺乏真实数据驱动的高保真度。

## 核心贡献（创新点）
1. **提出 VIPS 基准**：将伪仿真扩展至 V2I 协同场景，通过两阶段（真实观测 + 合成观测扰动）评估，在真实数据上实现可扩展且贴近闭环的规划鲁棒性评估。
2. **设计 CoS-V2X 稀疏协同规划框架**：采用锚实例库 + Top-K 选择性通信 + 双向交叉注意力 + 置信度加权融合，仅传输少量高置信度实例，相比 Uni-V2X 降低约 28.6% 通信带宽（2.5M vs 3.5M BPS）。
3. **构建带矢量地图标注的 V2X-Real 规划数据集**：为 V2X-Real 补充车道、交叉口、人行横道等矢量地图标注（共 36 条车道、7 个交叉口、25 个人行横道），并筛选出 12,944 训练帧 / 1,233 测试帧的 V2I 配对场景。
4. **提出面向基础设施视角的新颖视图合成方法**：车端视图采用 3DGS + Difix3D+ 扩散 refine，路端视图采用 SAM3 patch 检索 + inpainting 填充，LPIPS 分别为 0.371 / 0.076，显著优于纯 3DGS 的 0.414 / 0.324。

## 方法详解

### VIPS 两阶段伪仿真评估协议
- **Stage 1**：使用真实车端与路端多视角图像输入规划模型，在 BEV 非反应式仿真器（运动学模型 + 历史记录回放）中执行预测轨迹，以 EPDMS 评分。
- **Stage 2**：在专家轨迹终点周围采样扰动起点——横向 ±2.0m（1.0m 间隔），纵向按可行性加速度生成最多 7 个点（5.0m 间隔），通过 Hermite 样条拼接 Stage 1→Stage 2 轨迹并合成运动历史，共得 7,168 个 Stage 2 样本。
- **新颖视图合成**：车端用 3DGS 重定位 ego 后渲染，再用 Difix3D+ 扩散 refine（以真实无扰动视图为参考）；路端用 SAM3 检索相似朝向的 ego 车辆 patch 进行 patch-and-fill 合成，保证配对一致性。
- **Stage 2 聚合**：以高斯加权平均各样本得分，权重 $w_i = \exp(-\|x_i - \hat{x}\|^2 / 2\sigma^2)$ 鼓励靠近 Stage 1 终点的样本更重要。
- **统一评分**：$\mathrm{EPDMS} = \prod_{n \in M_{pen}} score_n \cdot \frac{\sum_{m \in M_{avg}} w_m \cdot score_m}{\sum w_m}$，其中安全关键指标（NC, DAC, DDC）相乘，性能指标（EP, TTC, LK, HC）加权平均；最终分 $s_{final} = s_1 \cdot s_2$。

### CoS-V2X 稀疏协同感知与规划
- 采用锚实例库（instance bank），车辆与路端共享 N 个 learnable anchors 作为时间先验。
- 每侧输出实例特征 $\mathbf{F}^s$、锚参数 $\mathbf{B}^s$、分类 logit $\mathbf{L}^s$；路端发送 Top-K（K=100）高置信度实例至车端。
- 双向交叉注意力：$\widetilde{\mathbf{F}}^{\text{veh}} = \text{Attn}(\mathbf{F}^{\text{veh}}, \mathbf{F}_\mathcal{K}^{\text{infra}}, \mathbf{F}_\mathcal{K}^{\text{infra}})$，$\widetilde{\mathbf{F}}_\mathcal{K}^{\text{infra}} = \text{Attn}(\mathbf{F}_\mathcal{K}^{\text{infra}}, \widetilde{\mathbf{F}}^{\text{veh}}, \widetilde{\mathbf{F}}^{\text{veh}})$。
- 置信度加权融合（$z_i^s = \max_c \sigma(\mathbf{L}_{i,c}^s)$）：$\mathbf{B}_i^{\text{fuse}} = w_i^{\text{veh}} \mathbf{B}_i^{\text{veh}} + w_i^{\text{infra}} \mathbf{B}_i^{\text{infra}}$，保留未选中的车辆锚。
- 感知输出直接送入基于 SparseDrive 的统一 motion prediction + planning 模块，无需额外改造。

## 实验与结果

- **数据集**：V2X-Real（含本文新增矢量地图标注），训练帧 12,944，测试帧 1,233。
- **对比基线**：单车 E2E 方法（AD-MLP、UniAD、SparseDrive、HiP-AD、MomAD）及 V2X 方法（Uni-V2X）。
- **最强结果**：CoS-V2X 综合得分 **50.88 EPDMS**，显著优于 Uni-V2X 的 43.79（+16.2%）与 SparseDrive 的 43.26；Stage 1 EPDMS 78.69 为所有方法最高。
- **效率优势**：训练峰值显存 8.86 GB（vs Uni-V2X 29.38 GB），推理显存 4.81 GB（vs 6.95 GB），通信带宽 2.5M BPS（vs 3.5M BPS）。
- **合成观质量**：Stage 2 性能下降主要来自扰动分布偏移而非渲染瑕疵；Stage 1 合成数据（原地重构）与真实数据下游性能差距极小（检测 mAP 仅下降 ~1.4）。
- **EPDMS 与人类判断强相关**：Kendall τ-b = 0.84，Pairwise Accuracy = 0.94；与常量速度基线（0.23 / 0.62）和随机（-0.17 / 0.42）形成鲜明对比。
- **附加实验**：IDM 反应式交通代理评估结果趋势一致（CoS-V2X 51.00）；通信延迟（500ms/1000ms）、数据丢包（10%/20%）、位姿误差（8°/0.4m）下均保持稳健且优于无 V2I 基线。

## 相关工作脉络
1. **NAVSIM v2 / 伪仿真（Cao et al., CoRL 2025）**：本工作的评测协议直接扩展自 NAVSIM v2 的两阶段伪仿真框架，但将其从单车扩展至 V2I 异构感知与协同规划场景。
2. **UniAD（Hu et al., CVPR 2023）/ SparseDrive（Sun et al., 2024）等端到端自动驾驶**：作为核心基线对比，证明稀疏表示在 V2I 扩展中的有效性与效率优势。
3. **Uni-V2X（Yu et al., AAAI 2025）**：当前最强的 V2X E2E 规划方法，本文通过稀疏锚表示在保持更高规划性能的同时降低约 28.6% 通信开销，体现了不同的设计哲学（密集 BEV vs 稀疏实例）。
4. **V2X 感知数据集（V2V4Real、DAIR-V2X-C、V2X-Seq、TUMTraf-V2X、UrbanIng-V2X）**：本文在 Table 1 中系统对比，选择 V2X-Real 因其唯一提供全 RGB 车+路多视角；补充矢量地图标注使其适用于规划任务。
5. **单车辆评测基准（nuScenes、Waymo、CARLA、Metadrive）**：本文指出其局限——开放循环无法评估误差累积，封闭循环存在域偏移，从而引出伪仿真作为折中方案的必要性。

## 局限性与未来方向
- 评测使用非反应式（log-replay）交通代理，虽扩展至 IDM 反应式但仍不能完全替代真实封闭循环仿真；后续可探索更复杂的动态交互评估。
- 伪仿真依赖 3DGS + Diffusion refine 合成新颖视图，存在轻微伪影风险；Stage 2 性能下降部分源于合成分布偏移而非纯决策难度。
- V2I 场景假设固定路端相机与已知相对位姿，现实中长基线标定误差与异步通信会带来额外挑战（补充材料已部分讨论位姿噪声鲁棒性）。
- 未探索多车协同（V2V+V2I）的扩展，以及大语言模型/世界模型在 V2I 协同规划中的应用潜力。

## 研究启发与可借鉴点
1. **两阶段伪仿真协议可迁移**：Stage 1 评估名义性能 + Stage 2 评估对扰动/分布偏移的鲁棒性，这一设计可推广至其他多智能体协同决策或具身机器人的离线-在线混合评测。
2. **稀疏锚实例库用于通信受限的协同系统**：Top-K 选择 + 双向交叉注意力 + 置信度加权融合的组合，为带宽受限的多智能体感知通信提供了简洁有效的范式。
3. **Patch-and-fill 基础设施视图合成**：针对固定高位相机场景，利用历史相似朝向帧进行 patch 检索填补，相比全局 3DGS 渲染在路端视角取得更低 LPIPS（0.076 vs 0.324），是低多边形/小目标场景新颖视图合成的实用技巧。
4. **EPDMS 综合指标的人类对齐验证**：通过专家标注者配对准确率（0.94）与 Kendall τ（0.84）建立指标可信度，可作为后续工作评估新指标时的重要参考。
5. **V2X-Real 矢量地图标注方案**：聚合并投影 LiDAR 点云到 BEV + 人工绘制 polyline/polygon 标注管线，可复用至其他缺少高精地图的 V2X 数据集规划任务适配。

## 关键术语表
- **V2I（Vehicle-to-Infrastructure）**：车联网中车辆与固定路侧基础设施（摄像头、雷达等）之间的通信与协作范式，提供超越 ego 视距的全局感知。
- **Pseudo-Simulation（伪仿真）**：在真实数据基础上引入合成状态扰动与新颖视图渲染，模拟 plausible future scenarios 以近似闭环评估的非交互式评测方法。
- **EPDMS（Extended Predictive Driver Model Score）**：整合 NC/DAC/DDC 安全惩罚与 EP/TTC/LK/HC 性能指标的综合评分，取值 [0,1]，与人类驾驶偏好高度一致。
- **Sparse Drive（稀疏驱动）**：基于可学习锚实例库的场景表征范式，以少量高质量实例替代密集 BEV 网格，降低感知与通信开销。
- **3D Gaussian Splatting（3DGS）**：基于 3D 高斯 primitives 的实时神经渲染技术，用于从多视角图像快速合成新颖视角，在动态驾驶场景中表现优异。
- **Instance Bank（实例银行）**：跨帧存储高置信度目标锚及其特征的缓冲区，提供时间一致性先验，支持多帧间对象级关联。
- **BPS（Bytes Per Second）**：衡量 V2X 通信带宽消耗的指标，本文以 10Hz 频率计算每秒传输字节数。
- **Hermite Spline（赫尔米特样条）**：通过端点位置与速度约束构造平滑轨迹的参数曲线，用于连接 Stage 1 与 Stage 2 状态生成一致的运动历史。

## 可复现要素
- **数据集**：V2X-Real（公开），本文补充的矢量地图标注亦随项目发布；代码与数据主页：https://vips2026.github.io
- **代码开源**：是（项目主页链接已给出）
- **基线模型训练**：除 Uni-V2X 因 OOM 将 BEV 尺寸从 (200,200) 降至 (160,160) 外，其余基线均从 scratch 重新训练；硬件为 4×A100 GPU
- **关键超参**：评估 horizon 5s；Anchor 数量（检测 900，地图 100）；Top-K 通信实例数 100；Stage 2 横向采样 ±2.0m / 1.0m 间隔，纵向 7 点 / 5.0m 间隔；传输频率 10Hz
- **未提及**：具体学习率、优化器配置、损失权重细节见 supplementary Sec. S3
