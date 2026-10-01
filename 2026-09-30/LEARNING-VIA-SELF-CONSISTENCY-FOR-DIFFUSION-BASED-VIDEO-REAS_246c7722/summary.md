---
title: "LEARNING-VIA-SELF-CONSISTENCY-FOR-DIFFUSION-BASED-VIDEO-REAS"
source: https://arxiv.org/pdf/2609.36826v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 17:30:23"
field: "视频生成与视觉推理"
keywords: ["video reasoning", "self-consistency", "diffusion models", "test-time scaling", "rejection fine-tuning", "visual search", "maze solving", "referring segmentation"]
innovations: ["首次将自洽性扩展至视频推理的结构化连续输出并设计任务特定聚合规则", "发现视频模型去噪早期锁定任务布局，提出早期读出显著降低推理成本", "提出拒绝微调RFT将多样本共识蒸馏为单生成模型，无需ground-truth监督"]
benchmarks: ["Frozen Lake Maze", "Visual Search Arrays (Campbell et al.)", "gRefCOCO Referring Segmentation"]
---

# 论文速读：LEARNING-VIA-SELF-CONSISTENCY-FOR-DIFFUSION-BASED-VIDEO-REASONING

## 一句话总结
本文首次将大语言模型中的自洽性(self-consistency)思想引入基于扩散的视频推理任务，提出无训练测试时扩展方法（多rollout聚合+早期读出）与拒绝微调(RFT)蒸馏方法，在迷宫求解、视觉搜索、引用分割三类任务上显著提升了视频生成模型的感知与推理能力。

## 研究问题与动机
- **视频生成模型的零样本推理可靠性不足**：即使如Veo 3等先进视频生成器，在视觉推理任务中也存在较大不稳定性，难以直接作为可靠推理器使用。
- **扩散生成的随机性与任务确定性之间的矛盾**：视频外观可以自由变化，但任务相关输出（路径、位置、掩码）必须满足约束，如何保留多样性的同时保证任务解的一致性是关键问题。
- **现有改进方法依赖外部监督**：SFT需要ground-truth视频数据，RLVR需要任务特定的可验证奖励函数，缺乏无需标注的自监督改进路径。

## 核心贡献（创新点）
- **首次系统研究视频推理的自洽性信号**：将自洽性从离散文本推理扩展到结构化连续输出（掩码、点、路径），其中精确匹配投票不再适用，需设计任务特定的聚合规则。
- **测试时扩展的早期读出机制**：发现视频模型在去噪早期即锁定任务布局，可在第20步（共49步）提前读出并聚合，推理时间减少52.4%而质量几乎无损。
- **拒绝微调(RFT)蒸馏多样本共识**：首次将多rollout共识蒸馏进视频生成模型，仅用高支持度的伪标签训练LoRA适配器，无需ground-truth视频或外部验证器，推理时仅需单次生成。
- **跨任务通用框架与显著提升**：在三个差异较大的任务上统一验证，视觉搜索准确率从48.4%→99.0%，迷宫严格成功率从72.0%→84.0%，分割gIoU从0.353→0.513。

## 方法详解
- **结构化预测提取**：对条件$(I, q)$和随机种子$s_i$，视频生成器$F_\theta$输出视频$\bar{V}^i$，任务提取器$L$将其映射为结构化预测$y^i$（迷宫路径为格点序列，视觉搜索为点坐标或否定符号$\bot$，分割为二值掩码）。
- **任务特定共识聚合**：
  - **点（视觉搜索）**：在输入图像坐标系中以半径$r=25$像素聚类，选择最大群，支持数$\geq m$时输出坐标中位数，否则返回$\bot$。
  - **路径（迷宫）**：去除连续重复格点后，相同序列形成同意群，固定支持阈值$m$接受，模态规则返回最大群，平局时按字典序打破。
  - **掩码（引用分割）**：像素级投票$\widehat{M}(u) = \mathbb{I}[\sum M^i(u) \geq m]$，保留至少$m$个样本支持的像素。
- **早期去噪读出**：在第$k=20$步解码预测的clean latent $\widehat{z}_{\mathrm{clean},k}$，提取任务预测并聚合，去噪器调用从$NG$降至$kG$，保留约99.6%的共识质量。
- **拒绝微调(RFT)**：冻结教师模型$F_{\theta_0}$，采样$G$个视频并计算共识$\hat{y}$，仅保留支持度达阈值的高质量伪标签；将共识掩码渲染为视频、VAE编码为clean target latent $z^\star$；学生模型用LoRA适配器训练，损失为flow-matching目标$\mathcal{L}_{\mathrm{SC}} = \mathbb{E}[\|v_\theta(z_t,t,c) - u_t^\star\|_2^2]$，训练阶段包括普通flow matching、marker区域聚焦、teacher state初始化、否定样本加权等。

## 实验与结果
- **数据集与基线**：MiniMax-H3 FL2VA（33B rectified-flow视频Transformer，50层）为主干；Frozen Lake迷宫（4×4/5×5/6×6，三种洞密度）、Campbell等视觉搜索阵列（2D/3D，合取/析取）、gRefCOCO引用分割；基线为Vanilla GenCeption、VR-Bench SFT、Wan-R1/VideoRLVR等。
- **推理自洽性结果**（Table 1）：
  - **视觉搜索**：准确率从48.4%→99.0%，命中率98.0%，特异性100.0%。
  - **迷宫求解**：严格成功率从59.1%→82.2%，目标到达率95.6%，最短路径率82.2%。
  - **引用分割**：gIoU从0.373→0.489，cIoU从0.224→0.359；早期读出20步与完整49步差距仅0.002 gIoU，时间减半。
- **RFT蒸馏结果**（Table 2）：
  - **视觉搜索**：单生成准确率从47.8%→69.4%，特异性从2.5%→38.8%；加入反事实增强后达78.4%/56.9%。
  - **迷宫求解**：4×4严格成功率从72.0%→84.0%，洞进入率从12.7%→6.7%，跳跃率从10.0%→5.3%。
  - **引用分割**：gIoU从0.353→0.513，精度从0.393→0.595，前景面积从0.464→0.200。
- **最强结果**：视觉搜索10-of-10共识准确率达99.0%，RFT蒸馏后单生成78.4%（反事实增强），迷宫RFT严格成功率84.0%。

## 相关工作脉络
- **GenCeption (Wang et al., 2026a)**：将视频生成先验适配视觉任务，本文在其基础上引入自洽性聚合提升零样本推理可靠性。
- **VR-Bench (Yang et al., 2025) / Wan-R1 (Liu et al., 2026) / VideoRLVR (Zhu et al., 2026)**：依赖ground-truth视频或可验证奖励的SFT/RL方法，本文无需外部监督，用内生于多样本共识的信号训练。
- **Self-Consistency (Wang et al., 2023)**：LLM中多数投票改进CoT推理，本文将其扩展到视频生成的结构化连续输出，并解决精确匹配不适用的问题。
- **TTRL (Zuo et al., 2025) / MM-UPT (Wei et al., 2025) / SCRL (Yan et al., 2026)**：多模态LLM的测试时强化学习/自训练，本文首次将多样本共识蒸馏进视频生成模型。
- **Thinking with Video (Tong et al., 2026)**：观察到Sora-2重复生成的多数投票提升可验证谜题准确率，本文系统研究结构化输出（非离散答案）的自洽性并延伸至后训练。
- **Proprio (Hassan et al., 2026) / Self-Refining (Jang et al., 2026)**：单轨迹迭代 refinment 利用物理可塑性信号，本文利用多独立轨迹的共识信号，两者正交互补。

## 局限性与未来方向
- **复杂任务扩展受限**：8×8迷宫等复杂推理中个体预测多样性过高，共识质量下降，方法有效性待验证。
- **后训练仅限RFT**：当前仅研究拒绝微调，共识信号作为reward与GRPO结合可验证奖励的对比尚未探索。
- **解码开销仍存**：早期读出虽减少去噪步数，但每个rollout仍需独立解码，latent selection策略可进一步优化。
- **任务特定提取器依赖**：不同任务需设计专属提取规则（颜色过滤、轨迹追踪等），泛化到任意任务需通用extractor设计。

## 研究启发与可借鉴点
- **自洽性作为内在监督信号**：无需外部标注，仅通过多rollout交叉一致性即可构建高质量伪标签，适用于任何可重复生成的结构化输出任务。
- **早期读出节省推理成本**：视频模型在去噪早期锁定布局这一观察可推广至其他生成式推理任务，实现test-time scaling的效率优化。
- **反事实增强提升否定能力**：视觉搜索中通过重着色共识点对象生成counterfactual样本，显著提升模型 abstention 能力，可借鉴于其他需区分有无目标的任务。
- **像素级高支持投票优于样本选择**：消融显示9-of-10像素共识(gIoU 0.487)远优于选择最一致单个样本(0.385)，说明聚合结构化输出比选样更有效。
- **与团队方向结合机会**：可将此方法迁移至视频轨迹预测、机器人操作规划等需结构化输出的任务，或结合团队已有RLVR工作探索共识reward信号。

## 关键术语表
- **Self-Consistency（自洽性）**：通过多次采样并聚合结果提升推理可靠性的方法，源自LLM的CoT多数投票。
- **Rejection Fine-Tuning (RFT)**：仅保留高共识支持度的伪标签进行微调，避免确认偏差的训练策略。
- **Early Readout（早期读出）**：在去噪轨迹中途解码clean latent并提取预测，减少推理计算量的技术。
- **Flow-Matching**：视频生成模型使用的训练目标，预测从噪声到数据的速度场。
- **Modal Path（模态路径）**：迷宫求解中得票数最多的路径序列，用于共识聚合。
- **Counterfactual Augmentation（反事实增强）**：修改共识点对象颜色生成否定训练样本的数据增强方法。
- **Test-Time Scaling (TTS)**：推理时增加计算预算（如多次采样）以提升性能的策略。
- **Rectified Flow**：MiniMax-H3使用的视频生成模型架构，基于流匹配的正则化轨迹。

## 可复现要素
- **数据集**：Frozen Lake（Newman et al., 2026）、Campbell等视觉搜索阵列（Campbell et al., 2024）、gRefCOCO（Liu et al., 2023a）；论文未声明自研数据集，均使用公开基准。
- **代码/权重**：使用MiniMax-H3 FL2VA（HuggingFace开源），学生模型LoRA适配器未公开；附录提供NumPy包用于复现离线曲线。
- **关键超参**：rollout数$G=10$，支持阈值$m=\lceil 0.9G \rceil$，早期读出步数$k=20$（总$N=49$），LoRA rank=16，scale=16，无dropout，AdamW $\beta=(0.9, 0.99)$，学习率$5\times10^{-5}\to10^{-4}$，训练3 epoch（分割）/2 epoch（迷宫）。
