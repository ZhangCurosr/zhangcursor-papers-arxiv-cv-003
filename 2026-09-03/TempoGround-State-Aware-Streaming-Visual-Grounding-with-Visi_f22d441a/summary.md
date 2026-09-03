---
title: "TempoGround-State-Aware-Streaming-Visual-Grounding-with-Visi"
source: https://arxiv.org/pdf/2609.02359v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 10:17:14"
field: "流式视觉定位"
keywords: ["streaming visual grounding", "vision-language models", "cross-frame correspondence", "reinforcement learning", "3D detection", "curriculum prediction"]
innovations: ["状态感知跨帧对应引导的课程预测机制，从2D关联到相机帧3D定位", "流式定位强化(SGR)三重奖励(定位/身份/一致性)对齐几何与轨迹一致性目标"]
benchmarks: ["ARKitScenes", "ScanNet", "ScanNet++", "ASE"]
---

# 论文速读：TempoGround: State-Aware Streaming Visual Grounding with Vision-Language Models

## 一句话总结
本文提出 TempoGround，一个基于 VLM 原生的流式视觉定位框架，通过状态感知的跨帧对应关系与课程预测机制，解决因果流式输入下的身份漂移、跨帧不一致及遮挡定位脆弱问题；并引入流式定位强化（SGR）对齐几何与一致性目标，在多个基准上取得显著超越。

## 研究问题与动机
- **流式输入下的独特挑战**：现有 VLM 定位方法主要针对单帧或离线视频片段，而实际部署中视觉观测以因果方式逐帧到达，要求模型在连续追踪的同时保持稳定定位。
- **三大失效模式**：流式定位常见身份漂移（混淆同类不同实例）、跨帧不一致（相邻帧边界框跳变）、部分遮挡下定位脆弱（间歇可见导致漏检无法恢复）。
- **世界坐标系不适于ego-centric场景**：现有3D场景理解方法依赖全局坐标系预测，需额外对齐步骤，引入额外误差；本文以当前相机帧为目标，更贴合具身操作等应用。
- **Token 级监督不足**：自回归 token 级 SFT 无法直接建模边界框的连续几何属性，也缺少轨迹级监督，难以满足流式定位的一致性需求。

## 核心贡献（创新点）
- **提出 TempoGround 框架**：通过状态感知的跨帧对应关系引导课程预测，从 2D 关联、存在状态估计到相机帧 3D 提升，实现流式视觉定位的稳定与准确；与以往离线或全序列方法本质区别在于因果流式设计。
- **引入流式定位强化（SGR）**：基于 RLVR 框架，使用可验证的定位、身份、一致性三重奖励优化模型；与常规 SFT 不同，SGR 直接对齐流式几何与轨迹一致性目标。
- **设计三阶段渐进训练策略**：从单帧几何能力 → 流式课程预测 → 序列级强化，逐步建立 streaming grounding 所需的空间与时间推理能力；与一次性端到端训练的本质区别在于分阶段能力递进。
- **构造对应关系扰动增强策略**：混合真实闭环误差与合成扰动（抖动、 dropout、幻影、过期 cue、空 cue），覆盖 80% 训练样本，防止模型过度依赖 prior cue 导致误差累积；与仅用干净 GT 监督的方法形成对比。
- **构建相机帧视角的统一流式数据管线**：过滤并规范多源异构数据，重新生成 view-grounded referring 表达，确保类别粒度统一、几何标签可靠；与直接使用公共 scene-level 标注的方法不同。

## 方法详解
- **问题形式化**：给定语言查询 q 与因果到达的图像序列 $I_{1:T}$，在每一步 t 输出匹配查询的实例集合 $\mathcal{Y}_t = \{(b_t^{2D,(k)}, b_t^{3D,(k)})\}$，预测仅依赖当前及历史观测与先前预测。
- **课程预测链**：每个时间步 t，模型依次执行：① 跨帧 2D 实例关联（使用前一帧 label + 2D box 作为轻量 cue $c_{t-1}$，不包含相机帧 3D box 以避免过时几何偏差）；② 存在状态估计（$S = \{new, track, lost\}$）；③ 2D 边界框解码；④ 提升到相机帧 3D 框 $(c_x, c_y, c_z, s_x, s_y, s_z, r, p, y)$。
- **流式定位强化（SGR）**：采用 GRPO 优化目标 $J(\theta) = E_{x,\tau}[R(\tau)]$，总奖励 $R(\tau) = R_g + R_{id} + R_{con} - f_{fmt}$。
  - **定位奖励 $R_g$**：按轨迹内每实例计算平均质量 $q = 0.35 \cdot \mathrm{IoU}_{2D} + 0.65 \cdot \mathrm{IoU}_{3D}$，再减去归一化假阳性惩罚。
  - **身份奖励 $R_{id}$**：对 new/track/lost 状态事件分别评分，鼓励正确进入/持续跟踪/退出决策，惩罚错误状态预测。
  - **一致性奖励 $R_{con}$**：检测连续定位失败链与 phantom（退出后仍有预测）链，以平方长度惩罚长串错误。
- **三阶段训练**：
  - Stage 1：26M 单帧样本 SFT，学习 2D 检测与 2D→3D 提升。
  - Stage 2：5.1M 流式样本 SFT，引入跨帧对应 cue 与存在状态监督。
  - Stage 3：4.1K 流式轨迹 SGR，2 epoch，G=4 rollout，温度 1.0，学习率 $1\times10^{-6}$。
- **对应关系扰动**：Stage 2 中 80% 样本使用非干净 cue（Stage 1 rollout 25%、抖动 20%、Dropout 12%、Phantom 12%、Stale 7%、Empty 4%），20% 用干净 GT cue，提升闭环鲁棒性。

## 实验与结果
- **数据集**：ARKitScenes、ScanNet、ScanNet++（3D）；ASE、ARKitScenes、ScanNet++（2D）；因果流式测试协议，每数据集 24 条测试序列。
- **评估指标**：$\mathrm{F1}_{2D}@0.5$、$\mathrm{F1}_{2D}@0.95$、$\mathrm{F1}_{3D}@0.25$、$\mathrm{AP}_{3D}$（在 $\Theta=\{0.15,0.25,0.50\}$ 均值）。
- **主要结果（3D）**：TempoGround-9B 在 ScanNet $\mathrm{AP}_{3D}$ 达 **45.3**（最强），ARKitScenes $\mathrm{AP}_{3D}$ 30.8，ScanNet++ 30.4；较基线 OVMono3D（28.0 均值）平均提升 7.5。
- **主要结果（2D）**：TempoGround-9B 在 ScanNet++ $\mathrm{F1}_{2D}@0.5$ 达 **79.4**，均值 $\mathrm{F1}_{2D}@0.5$ 58.9、$\mathrm{F1}_{2D}@0.95$ 26.4；较最强基线 LocateAnything（53.2/25.9）平均提升 4.4（@0.5）和 0.5（@0.95）。
- **消融**：去掉跨帧对应 prior $\mathrm{AP}_{3D}$ 降至 37.7（-4.6）；2D+3D prior 仅 41.8（< 纯 2D cue 的 42.3）；去掉 SGR 降至 42.3；各奖励单独移除均有下降。
- **鲁棒性分析**：在 0%-80% cue 扰动强度下，混合扰动训练的最优，纯 GT cue 训练在强扰动下衰减最快。
- **推理延迟**：9B 模型在单张 H200 上平均 0.65s，4B 模型 0.45s。

## 相关工作脉络
- **VLM-based Visual Perception（如 Llava-3D、3D-RFT）**：侧重于离线重建或短 clip 的全局几何推理；TempoGround 聚焦因果流式输入下的当前相机帧定位。
- **Visual Object Grounding（如 GroundingDINO、GLaMM、LocateAnything）**：针对单帧或全序列离线处理；TempoGround 引入跨帧对应与状态估计，处理因果流式场景。
- **3D 场景理解方法（如 3D-LDM、VLM-3R）**：依赖世界/场景坐标系；TempoGround 直接预测相机帧 3D 框，无需额外对齐。
- **Reinforcement Fine-Tuning for 3D（如 3D-RFT）**：针对视频级 3D 场景理解做 RLVR；TempoGround 的 SGR 专为流式定位设计，包含身份与一致性奖励。
- **Ego-centric 3D 感知（如 EgoObjects、EmbodiedScan）**：关注单帧或少量帧的 3D 标注；TempoGround 扩展到时序流式连续定位，处理进入/退出动态。

## 局限性与未来方向
- **训练数据依赖性强**：Stage 1 需 26M 单帧样本 + Stage 2 需 5.1M 流式样本，数据构建成本较高，在低资源场景下难以复现。
- **依赖高质量深度/3D 标注**：流式训练数据来源于多视角 RGB-D 序列（ScanNet 系、ARKitScenes 等），真实场景中深度信息缺失时性能可能下降。
- **未与更多流式感知方法对比**：主要对比了单帧/离线视频方法，缺乏与专门设计的流式 SLAM 或跟踪方法的对比。
- **仅支持轴对齐 3D 框**：未考虑旋转不变的 3D 定位或更复杂的几何形状表达。
- **推理延迟仍有优化空间**：0.65s/帧（9B）对于高帧率应用仍显偏高，未来可探索蒸馏或轻量化。
- **未来方向**：扩展到多模态流（如音频+视觉）、更复杂的遮挡恢复、与下游控制任务联合训练。

## 研究启发与可借鉴点
- **课程预测链设计可迁移**：将"关联→状态估计→定位→提升"的分步推理范式可迁移到其他流式感知任务（如流式语义分割、流式关键点检测）。
- **对应关系扰动增强策略**：混合真实闭环误差与人工合成扰动的做法，对其他依赖 prior 的时序模型有借鉴价值。
- **流式 RLVR 奖励设计**：三重奖励（定位/身份/一致性）的结构可复用于其他需要轨迹级一致性的视觉-语言任务。
- **相机帧 3D 提升策略**：先预测稳定 2D box 再 lift 到 3D 的思路，适用于 monocular 3D grounding 场景。
- **可与团队方向结合**：若团队从事具身交互或 AR/VR 定位，此框架可直接应用于实时物体追踪；SGR 思想也可用于优化其他流式 VLM 应用的 consistency。

## 关键术语表
- **流式视觉定位（Streaming Visual Grounding）**：在因果到达的图像序列中，根据语言查询连续定位并追踪匹配物体的任务。
- **状态感知跨帧对应（State-Aware Cross-Frame Correspondence）**：利用前一帧 2D box 与当前帧匹配，判断物体是 new/track/lost 的跨帧关联机制。
- **课程预测（Curriculum Prediction）**：按固定顺序依次输出状态、2D 框、3D 框的自回归生成策略。
- **流式定位强化（SGR）**：基于 RLVR 框架，使用可验证奖励优化流式定位模型的方法。
- **相机帧 3D 框（Camera-Frame 3D Box）**：以当前相机坐标系为原点的 3D 边界框，参数为 (cx,cy,cz,sx,sy,sz,r,p,y)。
- **对应关系扰动（Correspondence-Cue Perturbation）**：在训练时对前一帧 cue 施加抖动、Dropout、Phantom 等扰动，提升模型鲁棒性。
- **GRPO**：Group Relative Policy Optimization，无 critic 的强化学习策略梯度方法。
- **RLVR**：Reinforcement Learning with Verifiable Rewards，基于可验证奖励的强化学习微调。

## 可复现要素
- **数据集**：训练数据来自 Objects365、Flickr30k Entities、COCO、LVIS、EgoObjects、BDD100K、nuImages、PACO、RefCOCO/+/g、RefL4、ScanNet++、ScanNet、ADT、ASE、ARKitScenes、Hypersim、Objectron、SUN-RGBD、KITTI、nuScenes；测试在 ARKitScenes、ScanNet、ScanNet++、ASE 上进行。论文未声明是否公开训练数据，测试数据多为公开。
- **代码/权重**：论文未明确声明代码是否开源；模型基于 Qwen3.5 微调，权重未说明开源状态。
- **关键超参**：Stage 1 LR $4\times10^{-5}$（vision $2\times10^{-5}$），batch 1024，seq 8192；Stage 2 同上，seq 16384；Stage 3 GRPO LR $1\times10^{-6}$（vision $5\times10^{-7}$），KL $\beta=0.01$，G=4，温度 1.0，最多 15 步。
