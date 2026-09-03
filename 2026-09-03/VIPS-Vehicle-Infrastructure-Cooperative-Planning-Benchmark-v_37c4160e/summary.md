---
title: "VIPS-Vehicle-Infrastructure-Cooperative-Planning-Benchmark-v"
source: https://arxiv.org/pdf/2609.02462v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 14:59:13"
field: "车路协同自动驾驶规划"
keywords: ["V2I", "cooperative planning", "pseudo-simulation", "benchmark", "sparse representation", "end-to-end autonomous driving", "vehicle-infrastructure"]
innovations: ["扩展伪仿真至V2I协作场景，实现可扩展且高保真的规划鲁棒性评估", "提出CoS-V2X稀疏协作框架，以更低通信开销实现优于密集BEV基线的规划性能"]
benchmarks: ["VIPS", "V2X-Real"]
---

# 论文速读：VIPS-Vehicle-Infrastructure-Cooperative-Planning-Benchmark-v

## 一句话总结
本文提出 VIPS 基准，将伪仿真扩展至 V2I 协作场景，实现可扩展且高保真的自动驾驶规划鲁棒性评估；同时提出 CoS-V2X 稀疏协作规划框架，以低通信开销实现优于密集 BEV 基线的规划性能。

## 研究问题与动机
1. **开放环评估失真**：现有 V2X 协作规划工作普遍采用开放环评估，无法捕捉误差累积与偏离恢复行为，与真实表现相关性弱。
2. **闭环评估成本高**：CARLA 等闭环仿真面临部署风险、计算开销大及域差距（domain gap）等问题，难以大规模扩展。
3. **V2I 端到端规划研究匮乏**：现有 V2I 工作多聚焦感知层增强，端到端协作规划路径尚未充分探索。
4. **伪仿真未覆盖多智能体场景**：NAVSIM v2 等伪仿真方法仅适用于单智能体，将信息交互纳入评估体系仍存挑战。

## 核心贡献（创新点）
1. **提出 VIPS 两阶段伪仿真基准**：首次将伪仿真扩展至 V2I 协作场景，在真实数据上直接评估鲁棒性与误差传播，无需依赖仿真器。
2. **设计 CoS-V2X 稀疏协作规划框架**：基于 Top-K 锚点选择与置信度加权融合，以 2.5×10⁶ BPS 通信成本实现比 Uni-V2X（3.5×10⁶ BPS）更优的规划性能。
3. **构建 V2X-Real 向量地图标注**：通过 LiDAR 点云聚合与人工标注，补全 V2X-Real 缺失的地图信息，使该数据集可直接用于规划导向评估。
4. **提出面向 V2I 的新型视角合成管线**：车辆视角采用 3D Gaussian Splatting + Difix3D+ 细化，基础设施视角采用 patch-and-fill 策略，在 LPIPS 指标上显著优于纯 3DGS。

## 方法详解
**VIPS 两阶段评估协议**：
- **Stage 1（真实观测）**：接收同步的车辆多视图与基础设施多视图，预测 5 秒轨迹，在 BEV 非反应式仿真中执行并计算 EPDMS。
- **Stage 2（合成扰动）**：以专家轨迹终点为中心采样候选起点——横向 ±2m（1m 间隔）、纵向沿行驶方向 5m 间隔（最多 7 点），共 7,168 个样本。通过 Hermite 样条构造运动历史，再用 3DGS 合成车辆新视角（辅以 Difix3D+ 扩散 refine），基础设施视角采用 SAM3 提取 ego 车辆 patch 并检索填充。
- **统一评分**：EPDMS 整合安全惩罚项（NC、DAC、DDC 乘积）与性能加权平均项（EP、TTC、LK、HC），最终分 s_final = s₁ × s₂。

**CoS-V2X 协作感知与规划**：
- 采用 N 个可学习锚点的稀疏实例库，车辆与基础设施各自预测实例特征 F ∈ ℝ^{N×D}、锚点参数 B ∈ ℝ^{N×A}、分类 logits L ∈ ℝ^{N×C}。
- **通信压缩**：仅选 Top-K（K=100）高置信度基础设施实例（检测 900→100，映射 100 全部传输）。
- **双向交叉注意力**：F̃^veh = Attn(F^veh, F_K^infra) 与 F̃_K^infra = Attn(F_K^infra, F̃^veh)。
- **置信度加权融合**：w_i^s = max_c(σ(L_i,c^s)) / (max_c(σ(L_i,c^veh)) + max_c(σ(L_i,c^infra)) + ε)，F^fuse = w^veh·F̃^veh + w^infra·F̃^infra。
- 融合特征直接输入 SparseDrive 风格的并行预测-规划模块，无需额外修改。

## 实验与结果
- **数据集**：V2X-Real（12,944 训练帧、1,233 测试帧），含 36 车道、7 交叉口、25 人行横道标注。
- **基线**：Constant Velocity、AD-MLP、UniAD、SparseDrive、HiP-AD、MomAD、Uni-V2X。
- **核心结果**（Table 2）：
  - CoS-V2X：Stage 1 EPDMS = 78.69，Stage 2 EPDMS = 73.21，综合 = **50.88**（最优）。
  - 相对 SparseDrive（43.26）提升 **+17.7%**；相对 Uni-V2X（43.79）提升 **+16.2%**。
  - 多数方法 Stage 2 相比 Stage 1 均有下降，CoS-V2X 下降幅度最小，体现更强鲁棒性。
- **感知收益**（Table 3）：开启基础设施融合后，3D 检测 NDS 17.78→31.30，mAP 14.63→28.13；地图 mAP 30.46→34.24。
- **效率对比**（Table 4）：CoS-V2X 训练显存 8.86 GB vs Uni-V2X 29.38 GB；传输成本 2.5×10⁶ BPS vs 3.5×10⁶ BPS。
- **最强结果**：CoS-V2X 综合 EPDMS 50.88，在所有对比方法中位居第一。

## 相关工作脉络
1. **单智能体 E2E 自动驾驶**（UniAD、SparseDrive、HiP-AD、MomAD）——本文方法在稀疏表征范式上扩展至 V2I 协作，而非单智能体内部改进。
2. **伪仿真评估**（NAVSIM v2，Cao et al., CoRL 2025）——本文核心基准来源，将其从单智能体扩展至多智能体 V2I 场景，新增异构观测与协同扰动。
3. **V2X 协作感知数据集与方法**（V2X-Real、DAIR-V2X-C、V2V4Real、V2X-ViT、CoBEVT）——本文聚焦规划任务，而非感知层，解决了此前 V2X 规划缺乏真实数据基准的空白。
4. **V2X 协作规划**（Uni-V2X 等）——采用密集 BEV 表征，通信开销大；本文以稀疏锚点替代，实现更低带宽下更优性能。
5. **闭环仿真基准**（CARLA、Metadrive、Bench2Drive）——本文规避仿真域差距，直接在真实数据上完成高保真评估。

## 局限性与未来方向
1. **非反应式交通参与物**：主实验基于 log replay，虽补充了 IDM 反应式实验但未能完全模拟多车实时交互。
2. **合成观测仍存在轻微 artifacts**：Stage 2 性能下降部分源于视角合成误差与扰动难度叠加，难以严格分离。
3. **通信延迟评估受限**：因两阶段间无中间观测，延迟实验仅在 Stage 1 评估，缺少对 Stage 2 恢复能力的量化。
4. **地图标注依赖人工**：V2X-Real 向量地图需人工在 BEV 上逐条标注，跨数据集泛化成本较高。
5. 未来可探索真实闭环多车仿真验证、动态严重遮挡下的鲁棒性分析，以及向多车-多基础设施复杂路网扩展。

## 研究启发与可借鉴点
1. **两阶段伪仿真评估范式可直接迁移**：Stage 1 评估标称性能 + Stage 2 扰动鲁棒性，为其他自动驾驶子领域（如多智能体预测）的基准建设提供了通用模板。
2. **稀疏锚点 + Top-K 传输的通信压缩策略**：在保持协作增益的同时将带宽需求降低约 29%，适合部署在带宽受限的车路协同场景。
3. **基础设施视角的 patch-and-fill 合成策略**：利用固定摄像头特性，通过检索与填充实现高质量新视角，LPIPS 仅 0.076，显著优于 3DGS 的 0.324，可推广至其他静态传感器场景。
4. **EPDMS 与人类专家评估高度一致（Kendall τ-b = 0.84）**：为自动化规划评估提供了可信替代，减少对人工标注的依赖。
5. **数据集扩展路径**：LiDAR 点云聚合 + BEV 人工标注构建向量地图的方案，可复用于其他缺乏地图标注的 V2X 数据集（如 V2V4Real）。

## 关键术语表
- **V2I (Vehicle-to-Infrastructure)**：车路协同通信范式，基础设施传感器提供全局视野以弥补单车感知盲区。
- **Pseudo-Simulation (伪仿真)**：在真实数据基础上施加状态扰动与视角合成，近似闭环评估效果的非交互式评估方法。
- **EPDMS**：Extended Predictive Driver Model Score，整合安全惩罚与性能加权平均的统一评估标量（0~1）。
- **3D Gaussian Splatting**：基于 3D 高斯原语的神经渲染技术，适用于驾驶场景的动态物体新视角快速合成。
- **CoS-V2X**：本文提出的协作稀疏 V2X 端到端自动驾驶框架，以锚点实例替换密集 BEV 表征。
- **Top-K 实例选择**：仅传输基础设施中置信度最高的 K 个实例表征，实现通信带宽的显式控制。
- **Confidence-weighted Fusion**：基于双智能体分类最大 softmax 值的归一化加权融合，自适应整合互补视角信息。
- **Patch-and-fill 视角合成**：针对固定基础设施相机，通过检索相似朝向 ego 车辆 patch 并填充空洞区域，生成高质量合成视图。

## 可复现要素
- **数据集**：V2X-Real（公开），本文额外提供向量地图标注（见 https://vips2026.github.io）。
- **代码**：已开源（https://vips2026.github.io）。
- **关键超参**：评估时间窗 5s；Anchor 数量：检测 900 + 映射 100；Top-K = 100；Stage 2 采样：横向 ±2m（1m 间隔）、纵向 5m 间隔（最多 7 点），共 7,168 样本。
- **硬件**：4×A100 GPU。
- **训练细节**：基线模型从官方源码重新训练；Uni-V2X 因显存限制将 BEV 尺寸从 200×200 调整为 160×160。
