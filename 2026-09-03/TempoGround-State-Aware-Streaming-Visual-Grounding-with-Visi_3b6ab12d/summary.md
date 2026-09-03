---
title: "TempoGround-State-Aware-Streaming-Visual-Grounding-with-Visi"
source: https://arxiv.org/pdf/2609.02359v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 18:12:06"
field: "流式视觉定位"
keywords: ["streaming visual grounding", "vision-language models", "cross-frame correspondence", "reinforcement learning", "3D detection", "referring expression"]
innovations: ["State-aware curriculum prediction with cross-frame 2D correspondence and presence estimation", "Streaming Grounding Reinforcement (SGR) with verifiable Grounding, Identity, Consistency rewards", "Three-stage progressive training from single-frame SFT to streaming SGR"]
benchmarks: ["ARKitScenes", "ScanNet", "ScanNet++"]
---

# 论文速读：TempoGround: State-Aware Streaming Visual Grounding with Vision-Language Models

## 一句话总结
本文提出 TempoGround，一个 VLM-native 流式视觉定位框架，通过跨帧对应关系驱动的状态感知课程预测（2D关联→存在估计→2D框解码→相机帧3D提升）与可验证的流式定位强化（SGR）奖励，解决因果流式输入下的身份漂移、跨帧不一致和弱鲁棒性问题。

## 研究问题与动机
- **流式视觉定位的定义与挑战**：语言查询给定后，RGB图像以因果顺序逐帧到达，模型需持续跟踪匹配对象并输出当前相机帧的2D/3D边界框；现有单帧/离线视频方法无法处理时序身份一致性与跨帧预测稳定性。
- **三类典型失效模式**：身份漂移（同类别实例在时间上混淆）、跨帧不一致（相邻帧边界框抖动）、部分遮挡下的弱鲁棒性（间歇可见导致检测遗漏且无法通过因果时序推理纠正）。
- **全局坐标系依赖的局限**：多数3D场景理解方法在世界/场景坐标系中预测物体，不适用于具身操作等 ego-centric 视角；TempoGround 聚焦于当前相机帧的直接预测，避免额外的对齐误差。
- **Token 级监督的不足**：自回归 token 预测无法直接建模连续边界框的几何属性，且缺乏流式轨迹级别的一致性约束，需引入可验证奖励进行强化对齐。

## 核心贡献（创新点）
- **提出 TempoGround 框架**：基于标准 VLM 解码器，实现从2D对应关联到相机帧3D提升的课程预测链，本质区别于以往离线视频定位方法，首次明确面向因果流式输入。
- **设计状态感知跨帧对应机制**：通过轻量2D线索（仅传递标签与2D框）关联前后帧实例并预测存在状态（new/track/lost），与直接使用3D先验的方法相比避免旧视角几何偏差。
- **引入 Streaming Grounding Reinforcement (SGR)**：基于 GRPO 利用可验证的定位、身份、一致性三重奖励进行强化学习，弥补 token 级监督无法捕捉的几何与时序一致性目标。
- **构建三阶段渐进训练策略与大规模流式数据集**：从26M单帧样本（2D检测+2D-to-3D）到5.1M流式课程预测样本，再到 SGR 强化，逐步建立流式定位能力。
- **多基准评测验证有效性**：在 ARKitScenes、ScanNet、ScanNet++ 上因果流式协议下，TempoGround 显著提升 F1_2D、F1_3D 与 AP_3D，提供流式视觉定位的实用基础。

## 方法详解
- **问题形式化**：给定语言查询 q 和因果图像流 I_{1:T}，每步 t 输出当前帧可见匹配对象的集合 V_t = {(b^{2D}, b^{3D})}，其中2D框归一化坐标量化至[0,1000]，3D框以相机帧表示为 (c_x,c_y,c_z,s_x,s_y,s_z,r,p,y)。
- **课程预测链**：每步解码结构化的 token 序列，依次执行：①跨帧对应（利用前一帧标签与2D框构建线索 c_{t-1}，与当前帧区域匹配）→②存在状态估计（预测 new/track/lost）→③2D框解码→④相机帧3D提升（先2D后3D稳定单目估计）。
- **跨帧对应设计**：仅传递2D线索（丢弃3D），因3D框耦合前一视角会引入几何偏差；匹配当前帧对应区域后重新在当前相机帧估计3D。
- **存在状态词汇**：S = {new, track, ⊥ost}，分别表示新进入视野、持续在视野、离开视野；状态决定实例是否参与当前帧框预测，抑制身份漂移与退出后的误预测。
- **流式定位强化 (SGR)**：采用 GRPO 优化轨迹奖励 R(τ) = R_g + R_id + R_con - f_fmt；其中 R_g 衡量轨迹级定位质量（按 track 平均质量减去归一化假阳性惩罚），R_id 鼓励正确进入/退出行为（基于 per-instance 事件评分），R_con 惩罚由线索反馈放大的长失败连续链（含定位失败与退出后幻觉帧的二次惩罚）；f_fmt 惩罚不可解析输出。
- **三阶段训练**：Stage 1 在26M单帧样本上 SFT 学习2D检测与2D-to-3D提升；Stage 2 在5.1M流式样本上 SFT 学习课程预测与存在状态；Stage 3 在约4.1K流式轨迹上执行 SGR（GRPO，2 epochs，KL coef 0.01，G=4 rollouts）。
- **数据与扰动设计**：混合多源公开数据集（Objects365、COCO、RefCOCO、ScanNet++、ARKitScenes等）；Stage 2 对应线索采用80%非干净模式（Stage 1 rollout 25%、Jitter 20%、Dropout 12%、Phantom 12%、Stale 7%、Empty 4%）加20%干净GT线索，增强对错误线索的鲁棒性；引用表达重新生成视图 grounded 版本，避免绝对方向词依赖。

## 实验与结果
- **数据集与评测协议**：在 ARKitScenes、ScanNet、ScanNet++ 上进行因果流式评估，每数据集24条测试序列（检测与引用各半）；报告 F1_2D@0.5/@0.95、F1_3D@0.25、AP_3D（IoU阈值0.15/0.25/0.50均值）。
- **主要结果**：TempoGround-9B 在 ScanNet 上 AP_3D 达 45.3，较次优 DetAny3D（30.6）提升 14.7；平均 F1_3D@0.25 提升 6.2，平均 AP_3D 提升 7.5；2D 方面平均 F1_2D@0.5 提升 4.4，F1_2D@0.95 提升 0.5。TempoGround-4B 在 ScanNet AP_3D 为 38.6，ARKitScenes F1@0.25 为 47.6。
- **对比基线**：对比通用 VLM（Qwen3-VL、Qwen3.5、Cosmos-Reason2）、专用定位器（GroundingDINO-T、GLaMM、RexOmni-3B、LocateAnything、VG-LLM、DetAny3D、OVMono3D），TempoGround 在3D与2D流式评测中均取得最佳或次佳结果。
- **消融实验**：去除跨帧对应先验使 AP_3D 从42.3降至37.7；使用2D+3D先验（41.8）仍低于仅2D先验；去除存在状态预测降至40.5。移除 SGR 使 AP_3D 从45.3降至42.3；单独移除任一奖励（R_g、R_id、R_con）均降低性能，验证三者协同必要性。
- **鲁棒性分析**：推理时注入线索噪声（0%–80%空间抖动/丢弃/标签混合），本文混合扰动训练曲线在全强度范围最优，纯 GT 线索起点高但衰减快，纯 Stage 1 rollout 在轻噪声稳定但重噪声退化，充分扰动则峰值受抑。
- **推理延迟**：单图平均延迟在 NVIDIA H200 上 TempoGround-9B 为 0.65s、4B 为 0.45s，涵盖视觉编码、prefill 与自回归生成。

## 相关工作脉络
- **VLM-based 3D 场景理解**：如 3D-LLM、Chatscene、Llava-3D 等工作将重建几何或多视图注入 LLM，但监督与推理多基于扫描/片段中心，且预测常基于世界/场景坐标系；TempoGround 聚焦当前相机帧因果流，无需全局对齐。
- **开放式视觉定位**：GroundingDINO、GLaMM、RexOmni、LocateAnything 等方法将边界框视为自回归 token 或像素对齐，但主要针对单帧/离线视频；TempoGround 增加跨帧对应与存在状态建模，支持在线连续定位。
- **2D-to-3D 相机帧预测**：LocateAnything3D 等提出分阶段2D到相机帧3D预测；TempoGround 在此基础上引入时序课程，先2D后3D作为空间锚点稳定单目估计，并结合流式强化。
- **可验证奖励强化学习**：3D-RFT 等利用 RLVR 对齐几何生成；TempoGround 的 SGR 专门针对流式定位设计三重可验证奖励（定位质量、身份事件、连续一致性），不同于单帧几何对齐。
- **流式/在线感知**：多数视觉定位工作假设完整 clip 可用；本文明确针对因果流式协议，评估指标（F1 逐帧平均、AP 轨迹聚合）同步反映步骤级质量与流级稳定性。

## 局限性与未来方向
- **对前序线索质量的依赖**：虽经扰动训练提升鲁棒性，但快速 ego-motion 或严重遮挡下错误线索仍可能累积；未来可探索自适应线索置信度或更 robust 的关联机制。
- **计算开销与实时性**：当前推理延迟约 0.5–0.65s/帧，对高帧率具身应用仍显不足；未来可研究蒸馏、并行解码或更轻量的对应匹配模块。
- **数据集覆盖范围**：训练数据主要来自室内 RGB-D 场景（ScanNet、ARKitScenes 等），室外开放场景、极端光照与动态语义变化下的泛化尚未验证。
- **存在状态定义的简化**：lost 状态仅在退出瞬间标记一帧，后续空帧不产生框；未来可研究更细粒度的遮挡/重入建模与长期身份恢复。
- **扩展至更多下游任务**：本文聚焦检测与引用定位，未来可探索将状态感知课程预测推广至流式实例分割、轨迹预测与具身导航控制。

## 研究启发与可借鉴点
- **课程预测链设计**：将复杂时序定位分解为对应→状态→2D→3D的有序 token 序列，可作为通用范式迁移至流式分割、跟踪等任务。
- **SGR 奖励构造思路**：三重可验证奖励（几何质量、身份事件、连续一致性链惩罚）尤其适合需跨步稳定的在线感知任务，可复用到流式目标跟踪、视频理解。
- **对应线索扰动训练**：混合真实 rollout 误差与合成扰动（Jitter/Dropout/Phantom/Stale/Empty）模拟关闭回路错误分布，显著提升推理时对坏先验的鲁棒性，该方法论可广泛用于时序模型训练。
- **视图 grounded 引用表达生成**：摒弃全局方向词，基于当前视野内共视锚点生成唯一可判定的引用句，并辅以 uniqueness 检查，可提升流式引用定位的数据质量与泛化。
- **三阶段渐进训练**：从单帧几何能力到流式课程预测再到序列强化，逐步引入时序依赖，避免端到端训练的不稳定，该策略对任何需要引入时序信息的 VLM 微调均有参考价值。

## 关键术语表
- **Visual Grounding（视觉定位）**：将自然语言指代映射到图像/视频中空间目标的过程。
- **Streaming Visual Grounding（流式视觉定位）**：语言查询给定后，图像因果到达，模型需持续跟踪匹配对象并输出当前帧 2D/3D 边界框的设定。
- **Cross-frame Correspondence（跨帧对应）**：利用前一帧预测的轻量 2D 线索与当前帧区域匹配，维持同一实例的身份连续性。
- **Presence State（存在状态）**：离散标签 new/track/lost，表示实例当前帧是否新进入、持续存在或已离开视野。
- **Curriculum Prediction（课程预测）**：按固定顺序生成的 token 序列，依次为对应线索、存在状态、2D 框、3D 框，与推理顺序一致。
- **Streaming Grounding Reinforcement (SGR)**：基于 GRPO 的强化学习阶段，利用可验证的定位、身份、一致性奖励优化流式轨迹。
- **Camera-frame 3D Box（相机帧3D框）**：以当前相机坐标系表达的 3D 边界框，参数为 (中心坐标, 尺寸, roll/pitch/yaw)，避免世界坐标对齐误差。
- **Correspondence-cue Perturbation（对应线索扰动）**：在训练时对前一帧 2D 线索施加 rollout 误差、抖动、丢弃、幻象、陈旧或清空等扰动，提升模型对错误先验的鲁棒性。

## 可复现要素
- **数据集**：训练使用 Objects365、V3Det、COCO、LVIS、EgoObjects、BDD100K、nuImages、PACO、RefCOCO/+/g、RefL4、ScanNet++、ScanNet、ADT、ASE、ARKitScenes、Hypersim、Objectron、SUN-RGBD、KITTI、nuScenes 等公开数据集；测试在 ARKitScenes、ScanNet、ScanNet++ 上进行。论文未明确声明所有数据是否开源，但来源均为公开数据集。
- **代码与权重**：论文未明确说明代码/权重是否开源（基于提供文本未提及）。
- **关键超参**：Stage 1 学习率 4e-5（vision 2e-5），序列长度 8192，batch size 1024，1 epoch；Stage 2 序列长度 16384，其余相同；Stage 3 GRPO 学习率 1e-6（vision 5e-7），KL coef 0.01，G=4 rollouts，温度 1.0，最多15步流式，2 epochs。基础模型为 Qwen3.5，使用 AdamW、bfloat16、FlashAttention、DeepSpeed ZeRO-3，weight decay 0.05，梯度裁剪 1.0，128 张阿里云 PPU 训练约 109 小时。
