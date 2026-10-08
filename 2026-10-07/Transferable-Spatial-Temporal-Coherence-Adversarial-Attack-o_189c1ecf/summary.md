---
title: "Transferable-Spatial-Temporal-Coherence-Adversarial-Attack-o"
source: https://arxiv.org/pdf/2610.08331v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 09:46:20"
field: "多模态安全与对抗鲁棒性"
keywords: ["Vision Language Model", "Adversarial Attack", "Autonomous Driving", "Spatio-Temporal Coherence", "Transferability", "Black-box Attack", "Video Understanding"]
innovations: ["三阶段STCA框架：模态扩展+空间掩码攻击+时序连贯性破坏，首次系统性攻击视频VLM", "Motion-guided时序攻击：利用相邻帧YOLO掩码差值定位动态区域，破坏跨帧时序语义", "揭示域微调VLM（Dolphin）的对抗鲁棒性优势，建立模型专业化与安全性关联"]
benchmarks: ["BDD100K", "nuScenes"]
---

# 论文速读：Transferable-Spatial-Temporal-Coherence-Adversarial-Attack-o

## 一句话总结
本文提出了时空连贯性对抗攻击框架（STCA），针对自动驾驶场景中的视频视觉语言模型（VLM）在**黑盒威胁模型**下实施攻击，通过"模态扩展→空间攻击→时序攻击"三阶段流水线，利用可转移性成功破坏跨帧时序语义连贯性，对Video LLaVA-7B、Qwen2.5-VL-7B、Dolphin等模型实现最高96.5%的攻击成功率（ASR）。

## 研究问题与动机
- **现有VLM对抗攻击局限于静态图像**，未能评估视频数据固有的时序动态性，对自动驾驶场景中跨帧语义连贯性的攻击属于空白。
- 自动驾驶场景的时序理解依赖帧间运动关系（如"行人穿越"与"车辆停止"的因果顺序），仅攻击单帧无法全面破坏模型推理。
- 现有方法扰动均匀作用于整帧，未利用语义关键区域（前景物体/运动区域），导致攻击效率与不可感知性难以兼顾。
- 域特定VLM（如Dolphin）的对抗鲁棒性与通用模型的差异尚未被系统揭示。

## 核心贡献（创新点）
- **三阶段STCA框架**：首次将模态扩展（多caption生成+CLIP语义对齐选帧）、空间掩码攻击（YOLO-guided）、时序连贯性破坏（LanguageBind+motion mask）串联，系统性攻击视频VLM。
- **Caption-guided Frame Selection策略**：利用CLIP计算帧与多caption的余弦相似度，选择语义最相关的关键帧进行扰动，而非均匀扰动所有帧。
- **Motion-guided Temporal Attack**：通过相邻帧YOLO掩码差值生成运动掩码，仅在动态区域施加时序扰动，显著提升攻击效率且保持高SSIM。
- **发现域微调提升鲁棒性**：Dolphin（自动驾驶域微调）相比通用VLM表现出明显更高的对抗鲁棒性，首次揭示这一现象。
- **高可转移的黑盒攻击**：基于TCL和LanguageBind代理模型生成的扰动，成功转移至架构各异的三类目标VLM。

## 方法详解
**Stage 1：Modalities Expansion（模态扩展）**
- 使用Gemini 2.5和Video-LLaVA生成视频的多caption集合 $C = C^G \cup C^L$。
- 利用CLIP ViT-B/32提取帧视觉嵌入 $v_t$ 和caption文本嵌入 $u_k$，计算余弦相似度 $\cos(v_t, u_k)$。
- 对每个caption，对其所有帧相似度取平均得到caption得分 $S(c_k, v_i) = \frac{1}{T}\sum_{t=1}^{T}\cos(v_t, u_k)$，选取Top-3 caption作为候选。
- 将Top-3 caption嵌入取平均得到聚合文本向量 $\bar{u}$，再对每帧计算与$\bar{u}$的相似度，选取Top-K（60帧）作为关键帧集合 $\mathcal{F}^*$。

**Stage 2：Spatial Attack（空间攻击）**
- 对关键帧使用YOLOv8s（置信度阈值0.25，IoU阈值0.45）检测前景物体，生成二值掩码 $M_i$（物体区域为1，背景为0）。
- 基于PMP框架（Fu et al. [26]）优化扰动：在ℓ∞范数约束 $||i'-f_i||_\infty \leq \epsilon$ 下，最大化扰动区域与caption的跨模态错位：
$$i^{adv} = \arg\max_{||i'-f_i||_\infty \leq \epsilon} \mathcal{L}(E_I(i'), \{E_T(C_j)\}_{j=1}^{3}), \quad \mathcal{L} = -\sum_{j=1}^{3} (E_I(i') \cdot E_T(C_j))$$
- 采用多尺度增强 $S=\{0.5, 0.75, 1.0, 1.25, 1.5\}$ 提升可转移性。
- 超参数：$\epsilon=32/255$，$\alpha=1/255$，$T=20$，动量衰减$\lambda=0.9$。

**Stage 3：Temporal Attack（时序攻击）**
- 计算相邻帧运动掩码：$\mathcal{M}_{motion} = |\mathcal{M}_{t+1} - \mathcal{M}_t| > \tau_m$（$\tau_m=0.1$），仅对运动区域施加额外扰动。
- 双重损失：
  - 文本-视觉对齐损失（利用LanguageBind视频编码器）：$\mathcal{L}_{text} = \frac{1}{|\mathcal{T}|}\sum_{j=1}^{|\mathcal{T}|}\cos(E_V(i^{adv}), E_T(t_j))$
  - 时序连贯性损失：$\mathcal{L}_{temporal} = \frac{1}{K-1}\sum_{t=1}^{K-1} \mathcal{M}_{motion} \cdot \cos(E_V(i_t^{adv}), E_V(i_{t+1}))$
- 超参数：$\epsilon=16/255$，$\alpha=2/255$，$T=20$。

## 实验与结果
**数据集**：BDD100K（800条视频用于攻击）和nuScenes（85个场景）。

**目标模型**：Video LLaVA-7B、Qwen2.5-VL-7B、Dolphin（自动驾驶域专用）。

**评估指标**：ASR（词重叠相似度<0.5视为成功）和SSIM（感知质量）。

**主要结果**：

| 数据集 | 模型 | Spatial Attack ASR | Full STCA ASR | SSIM |
|--------|------|--------------------|---------------|------|
| BDD100K | Video LLaVA-7B | 32.6% | **71%** | 0.82 |
| BDD100K | Qwen2.5-VL-7B | 45% | **84.2%** | 0.82 |
| BDD100K | Dolphin | 46.9% | 46.2% | 0.82 |
| nuScenes | Video LLaVA-7B | 57.6% | **83%** | 0.79 |
| nuScenes | Qwen2.5-VL-7B | 64.7% | **96.5%** | 0.79 |
| nuScenes | Dolphin | 37.6% | 47.1% | 0.79 |

**最强结果**：在nuScenes数据集上对Qwen2.5-VL-7B达到**96.5% ASR**，较空间攻击提升31.8个百分点；Dolphin保持最高鲁棒性（STCA仅47.1%）。

**基线对比**：PGD（Video LLaVA: 47.2%）和FGSM（Video LLaVA: 25.2%）远低于本文方法（71%）。

## 相关工作脉络
- **AdvCLIP [14]**：针对CLIP跨模态对齐的通用对抗攻击，但仅评估分类/检索任务，未涉及视频时序，且为白盒设置。
- **VLAttack [15]**：黑白盒VLM攻击，聚焦单模态+多模态层级攻击图像-文本对，未考虑视频的时序维度。
- **ADvLM [27]**：首个针对自动驾驶VLM的攻击，但为**白盒**设置，且仅基于空间扰动，无时序攻击模块。
- **CAD [28]**：首个自动驾驶VLM的**黑盒**攻击，采用决策链破坏策略，但未显式建模跨帧时序连贯性。
- **BTC [24]**：针对视频动作识别的时序一致性攻击，使用图像代理生成视频通用对抗扰动，目标为分类任务而非VLM语义输出。
- **PG-Attack [26]**：采用精确掩码扰动（PMP）+欺骗性文本patch，为本文空间攻击阶段的直接借鉴来源。

## 局限性与未来方向
- **检测器依赖**：YOLO掩码可能遗漏小目标或低置信度物体，影响攻击覆盖范围。
- **域特定性限制**：仅评估了三个模型，缺乏对更多架构（如多模态大模型、视觉-动作模型）的泛化验证。
- **物理可实现性未验证**：当前为数字空间攻击，未考虑物理世界中的光照、视角、距离等变换。
- **黑盒查询次数**：未报告实际查询开销，对实际部署场景的实用性存疑。
- **未来方向**：物理攻击转移、时序防御机制设计、域微调对鲁棒性的系统性影响分析。

## 研究启发与可借鉴点
- **多caption生成+CLIP语义排序策略**可用于视频关键帧选择，提升后续任务的信息密度，可迁移至视频摘要、关键帧检测等任务。
- **Motion-guided Mask**思路（相邻帧掩码差值）可直接复用于视频异常检测、运动目标分割等下游任务。
- **域微调提升鲁棒性**的发现值得深入：可探索在视觉-语言模型微调过程中引入对抗训练，系统性提升安全性。
- **多尺度增强+PMP框架**的组合策略，可作为通用的可转移对抗攻击模板，适配不同VLM架构。
- **评估指标设计**（词重叠相似度+SSIM双指标）为视频VLM安全性评估提供了可复用的评估范式。

## 关键术语表
- **Vision Language Model (VLM)**：融合视觉与语言理解的多模态大模型，如CLIP、LLaVA、Qwen-VL等。
- **Spatio-Temporal Coherence**：视频空间中单帧语义与时间上跨帧连贯性的联合一致性。
- **Attack Success Rate (ASR)**：攻击后模型输出与原始输出词重叠相似度低于阈值的样本占比。
- **Structural Similarity Index (SSIM)**：衡量对抗样本与原帧在亮度、对比度、结构上的感知相似性。
- **Transferability**：在黑盒设置下，白盒代理模型生成的对抗扰动对未知目标模型的有效性。
- **Motion-guided Mask**：由相邻帧目标检测掩码差值生成的掩码，仅标记运动区域。
- **Semantic Entropy**：用于过滤冗余caption的度量，基于文本 Embedding 分布的熵值。
- **Precision Mask Perturbation (PMP)**：仅对掩码标记的关键区域施加对抗扰动的策略。

## 可复现要素
- **数据集**：BDD100K（公开）、nuScenes（公开）。
- **代码/权重**：论文未声明开源，需联系作者获取。
- **关键超参**：
  - 帧选择：CLIP ViT-B/32，Top-3 caption，Top-60帧。
  - 空间攻击：YOLOv8s（conf=0.25, IoU=0.45），$\epsilon=32/255$，$\alpha=1/255$，$T=20$，$\lambda=0.9$，多尺度$\{0.5, 0.75, 1.0, 1.25, 1.5\}$。
  - 时序攻击：$\epsilon=16/255$，$\alpha=2/255$，$T=20$，运动阈值$\tau_m=0.1$。
- **硬件**：NVIDIA A100 GPU（Google Colab Pro）。
- **模型精度**：Video LLaVA（FP16）、Qwen2.5-VL（BF16）、Dolphin（FP16 + LoRA）。
