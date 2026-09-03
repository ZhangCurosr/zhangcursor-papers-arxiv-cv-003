---
title: "Towards-Zero-Shot-Transfer-Across-Embodiments-For-Driving-VL"
source: https://arxiv.org/pdf/2609.02341v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 18:12:31"
field: "自动驾驶视觉-语言-动作模型"
keywords: ["VLA", "zero-shot transfer", "BEV-Forcing", "cross-embodiment", "autonomous driving", "multi-dataset training", "spatial representation"]
innovations: ["提出BEV-Forcing辅助任务，通过轻量BEV头将occupancy map监督注入VLM backbone以提升跨载体零样本迁移", "系统揭示多载体训练与辅助监督的交互规律，指出二者存在重叠收益", "证明低容量BEV头优于全Transformer块，强迫主干特征编码密集空间信息"]
benchmarks: ["WOD-E2E", "KITScenes LongTail", "Physical AI", "nuScenes-QA"]
---

# 论文速读：Towards-Zero-Shot-Transfer-Across-Embodiments-For-Driving-VL

## 一句话总结
论文研究了多数据集训练对驾驶VLAs（Vision-Language-Action）的影响，并提出**BEV-Forcing**辅助任务——通过将专业BEV模型的俯视图occupancy map作为监督信号注入VLM backbone——来提升零样本跨载体迁移能力；当训练载体数量有限时增益显著，但随着数据多样性增加其作用递减。

## 研究问题与动机
1. **缺乏跨载体零样本迁移研究**：现有驾驶VLA主要在单一数据集上训练，极少评估在未见过的相机配置上的零样本迁移能力；虽然多载体预训练在机器人操作中已被证明有效，但自动驾驶领域对此探索不足。
2. **几何迁移缺失**：VLA在微调过程中可能学到特定数据集的空间理解，但在不同相机配置的载体上会出现灾难性失效——语义理解可迁移，3D空间理解难以迁移。
3. **简单堆叠数据不一定有效**：直接将多个数据集加入训练并不必然提升在同一载体上的性能，且部分数据集（如WOD-E2E）缺乏3D标注，限制了显式空间学习。
4. **缺乏统一的评估视角**：新方法往往仅在单一数据集上评估，缺少在多载体、零样本场景下的系统比较。

## 核心贡献（创新点）
1. **提出BEV-Forcing辅助任务**：通过在VLM backbone最后一层提取图像隐状态，经轻量BEV头生成occupancy map并与SimpleBEV输出的teacher occupancy map进行对比监督；与Spatial Forcing不同，BEV-Forcing仅使用摄像头输入、无需额外3D标注，且在推理时可完全移除无额外成本。
2. **系统揭示多载体缩放与辅助任务的交互规律**：在WOD-E2E单载体训练时，BEV-Forcing使Physical AI零样本ADE下降10.1%；但在WOD-E2E + NAVSIM + nuScenes-QA多载体组合下，该辅助任务效果被数据多样性部分替代，揭示二者存在重叠收益。
3. **发现低容量BEV头的必要性**：对比Cross-Attention仅BEV头与全Transformer Block BEV头，前者在在分布（ADE 2.376）和零样本（ADE 1.725）均优于后者，证明强迫图像特征本身编码密集空间信息比让辅助头"代劳"更有效。
4. **提供VQA联合训练的证据**：多数据集训练会导致骨干语言/语义能力退化（nuScenes-QA准确率从29.2%降至11.1%），加入约10% VQA样本后可恢复至60.2%并改善规划性能，凸显保持语言监督的重要性。

## 方法详解
1. **骨干与动作表示**：采用预训练开源VLM **Qwen 3.5 2B**，动作表示沿用文本化waypoint序列（1 Hz，2D坐标对），配合**LoRA**（rank=16）避免灾难性遗忘。
2. **BEV头设计**：
   - 初始化查询向量 $q \in \mathbb{R}^{(N_X \cdot N_Y) \times d}$，$N_X \times N_Y$ 为空间网格分辨率。
   - 从backbone目标层 $\ell$（取最后一层）提取图像嵌入 $f_I^{(\ell)} \in \mathbb{R}^{N_I \times D}$，经MLP投影至维度 $d$。
   - 交叉注意力：$\hat{b} = \text{Attn}(W_Q \cdot q, W_K \cdot \tilde{f}_I, W_V \cdot \tilde{f}_I)$，其中 $\tilde{f}_I = \text{MLP}(f_I^{(\ell)})$。
   - 线性层输出occupancy logits：$\hat{\mathcal{F}}_{\text{BEV}} = \text{Linear}(\hat{b})$。
3. **损失函数**：
   - BEV辅助损失采用加权二分类交叉熵（正样本加权 $w_+$ 缓解类别不平衡）：$\mathcal{L}_{\text{BEV}}$。
   - 总损失：$\mathcal{L} = \mathcal{L}_{\text{NTP}} + \lambda \cdot \mathcal{L}_{\text{BEV}}$。
   - VQA样本采用相同结构：NTP监督答案token，同时BEV头对图像隐状态产生$\mathcal{L}_{\text{BEV}}$。
4. **Teacher模型**：使用 **SimpleBEV** 输出occupancy map作为监督目标，因其即使在未见数据集上也能输出较高质量结果，适合WOD-E2E等无地图标注的数据集。
5. **鲁棒性技巧协同**：图像增强和相机校准参数注入（通过MLP后乘$\alpha$初始化为0，叠加至image patch）两项技术与BEV-Forcing配合可进一步提升零样本迁移。

## 实验与结果
- **数据集**：规划任务使用 **WOD-E2E**、**NAVSIM**、**nuScenes**、**Physical AI**（5k训练+1k验证）；VQA使用 **nuScenes-QA**；零样本评估使用 **KITScenes LongTail** 测试集。
- **指标**：ADE/FDE（越低越好）、RFS（WOD-E2E，越高越好）、MMS（KITScenes LongTail，越高越好）。
- **主要结果**：
  - 单载体WOD-E2E训练：加BEV-Forcing后Physical AI零样本ADE下降10.1%，FDE从5.946降至5.831。
  - KITScenes LongTail测试（WOD-E2E + NAVSIM）：加BEV-Forcing后MMS从4.62提升至**5.15**，超越Gemini 3 Pro（4.61）和Alpamayo 1.5（4.31）。
  - WOD-E2E测试集：最佳配置（W+N+nSQA + BEV）RFS达7.902，ADE 2.938。
  - VQA准确率：基座Qwen 3.5 2B为29.2%，单WOD+E2E训练降至11.1%，加入nuScenes-QA后恢复至**60.2%**。
- **结论**：BEV-Forcing在有限载体训练时作为几何感知正则化器效果显著；随着训练载体增多，数据多样性与辅助监督的收益趋于重叠；多载体+VQA联合训练可保持语言理解能力。

## 相关工作脉络
1. **Spatial Forcing [19]**：用3D模型（VGGT）特征监督图像嵌入，本文采用BEV occupancy map替代，目标更统一且无需3D标注。
2. **VGGT / VGGDrive [36, 37]**：利用视觉几何先验辅助VLAs，本文聚焦于仅用摄像头输入的低成本方案。
3. **DriveMA-2B [44]**：引入可验证meta-actions提升训练信号质量；本文侧重于空间表征学习而非动作表征改进。
4. **Poutine [32]**：基于类似架构的SFT-only基线，本文与其表现接近（RFS 7.902 vs 7.986），验证了方法的有效性。
5. **Gemini 3 Pro / Alpamayo 1.5**：通用模型或复杂推理驱动方案；本文证明经过领域数据微调的小模型在零样本场景下可超越它们。
6. **DriveVLA-W0 [23]**：通过世界模型解锁scaling law；本文聚焦低成本辅助任务，强调在有限数据下如何通过表征学习提升迁移。

## 局限性与未来方向
1. **仅关注图像嵌入层面的空间表征学习**，未深入探索推理链（reasoning trace）、动作表示优化和架构改进对几何迁移的贡献。
2. **Teacher occupancy map质量受限**：使用SimpleBEV生成的map可能不够精细，未来可探索包含可通行区域（drivable area）等更丰富信息的监督信号。
3. **仅评估了有限数量的公开数据集**，真实场景中的载体多样性（光照、天气、地理）尚未充分覆盖。
4. **缺乏闭环仿真验证**：仅在开环轨迹预测上评估，零样本实际驾驶表现有待进一步验证。

## 研究启发与可借鉴点
1. **BEV-Forcing的低容量设计原则**：辅助任务头应"足够轻"以迫使主干特征编码信息，而非"足够强"以绕过特征学习——这对其他辅助任务（如深度估计、语义分割）的设计具有迁移价值。
2. **多任务联合训练的必要性**：多载体训练必须配合语言/VQA监督以防止语义退化，这为多模态预训练的混合策略提供了实证依据。
3. **鲁棒性技巧的协同效应**：图像增强和相机校准注入需配合统一空间表征接口才能发挥最大效果，启示未来研究应关注"技巧组合"而非孤立尝试。
4. **消融需纳入数据缩放变量**：本文强调新方法的效果应随训练数据多样性变化进行评估，否则可能被高估——这为实验设计提供了方法论参考。

## 关键术语表
- **VLA（Vision-Language-Action Model）**：融合视觉、语言和动作输出的端到端自动驾驶模型，具备语义理解和场景推理能力。
- **BEV-Forcing**：本文提出的辅助任务，通过BEV头将专业BEV模型的occupancy map监督信号注入VLM backbone图像隐层，以提升空间感知。
- **Cross-embodiment zero-shot transfer**：模型在未见过的相机配置/载体设备上直接执行规划任务的能力，无需微调。
- **Occupancy map**：BEV空间中的二值网格，表示每个网格是否被车辆等障碍物占据。
- **SimpleBEV**：论文选用的轻量级BEV感知模型，用于生成teacher occupancy map。
- **RFS（Rater Feedback Score）**：WOD-E2E上基于人工评分的轨迹质量指标，综合考虑轨迹平滑性、安全性等。
- **MMS（Multi-Maneuver Score）**：KITScenes LongTail上的综合评分，衡量复杂场景下的多 maneuver 完成能力。
- **NTP（Next-Token Prediction）**：自回归语言模型的标准训练目标，预测下一个token的概率分布。

## 可复现要素
- **数据集**：WOD-E2E、NAVSIM、nuScenes、Physical AI（5k train / 1k val）、nuScenes-QA、KITScenes LongTail（部分）；**Physical AI训练子集已开源**。
- **代码/权重**：代码已开源在 github.com/caiocj1/ad-vla；论文未提及模型权重是否公开。
- **关键超参**：LoRA rank=16，learning rate=$10^{-4}$（cosine decay），global batch size=64，训练1 epoch，4×A100 GPU；16步过去轨迹（4 Hz）；图像输入降分辨率保 aspect ratio；相机校准参数经MLP后以$\alpha$（初始化为0）缩放叠加至image patch。
