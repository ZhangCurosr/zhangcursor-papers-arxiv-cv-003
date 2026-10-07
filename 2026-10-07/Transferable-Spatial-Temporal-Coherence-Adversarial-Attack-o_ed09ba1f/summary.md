---
title: "Transferable-Spatial-Temporal-Coherence-Adversarial-Attack-o"
source: https://arxiv.org/pdf/2610.08331v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-07 10:43:40"
field: "多模态AI安全与对抗鲁棒性"
keywords: ["Vision Language Model", "Adversarial Attack", "Temporal Coherence", "Autonomous Driving", "Transferability", "Black-box Attack", "Video Understanding"]
innovations: ["提出三阶段STCA框架，首次系统性从时序维度攻击自动驾驶视频VLM", "Caption引导的语义关键帧选择与YOLO-guided空间掩码结合，实现精准时序破坏", "基于LanguageBind的运动掩码时序连贯性攻击，在黑盒设定下实现高迁移性"]
benchmarks: ["BDD100K", "nuScenes"]
---

# 论文速读：Transferable-Spatial-Temporal-Coherence-Adversarial-Attack-o

## 一句话总结
本文提出了一种面向自动驾驶场景的**时空连贯性对抗攻击框架（STCA）**，通过Caption引导的帧选择、YOLO-guided空间掩码扰动和LanguageBind驱动的时序连贯性破坏，实现了对黑盒视频VLM模型的高效跨模型迁移攻击。

## 研究问题与动机
- **现有VLM对抗攻击缺乏时序维度考量**：当前针对VLM的对抗攻击主要针对静态图像，忽视了视频中连续帧之间的时序连贯性与运动一致性这一关键安全漏洞。
- **自动驾驶VLM时序脆弱性未被系统评估**：自动驾驶场景依赖跨帧时序推理（如车辆-行人因果关系），但现有方法未针对该时序语义一致性进行专门攻击设计。
- **黑盒迁移性不足**：已有方法多在白盒设定下验证，跨不同VLM架构的黑盒迁移攻击效果有限，且未考虑语义相关性约束。
- **均匀扰动导致可感知性高**：对整帧均匀施加扰动会破坏背景区域，降低对抗样本隐蔽性；需针对语义关键区域（目标物体、运动区域）进行精准攻击。

## 核心贡献（创新点）
1. **提出三阶段STCA攻击框架**：结合模态扩展、空间攻击与时序连贯性攻击，首次系统性地从时序维度攻击自动驾驶视频VLM，区别于仅针对单帧的现有方法。
2. **Caption引导的语义关键帧选择策略**：利用CLIP计算帧与多Caption的余弦相似度，筛选语义最相关帧进行攻击，提升攻击效率并保留视频整体相似性，区别于均匀扰动所有帧的常规做法。
3. **YOLO-guided空间掩码与多尺度增强**：采用YOLOv8生成目标对象掩码，将扰动限制在语义关键区域，并结合多尺度数据增强提升跨模型迁移能力，显著优于随机空间扰动。
4. **基于LanguageBind的时序连贯性破坏**：通过相邻帧运动掩码（Motion Mask）定位动态区域，优化时序一致性损失，直接破坏VLM的跨帧时序推理机制。
5. **首次在自动驾驶VLM上验证时序攻击的强迁移性**：在BDD100K和nuScenes数据集上，对Video LLaVA-7B、Qwen2.5-VL-7B、Dolphin三个异构模型实现高ASR（最高96.5%），揭示了领域专用模型的部分鲁棒性。

## 方法详解
**整体流程**：输入视频 → 模态扩展（生成多Caption） → 时空关键帧选择 → 空间攻击（YOLO掩码+PGD扰动） → 时序攻击（Motion Mask+LanguageBind） → 输出对抗视频。

**阶段一：模态扩展（Modalities Expansion）**
- 使用Gemini 2.5和Video-LLaVA生成视频的多条语义等价Caption，构成Caption集合 $C = C^G \cup C^L$
- 利用CLIP计算每帧视觉嵌入 $v_t$ 与每Caption文本嵌入 $u_k$ 的余弦相似度：$\cos(v_t, u_k) = \frac{v_t^\top u_k}{\|v_t\|_2 \|u_k\|_2}$
- 对每个Caption在所有帧上取平均相似度作为评分，选择Top-3最优Caption用于后续帧选择

**阶段二：空间攻击（Spatial Attack）**
- Caption引导帧选择：计算每帧与Caption集合均值嵌入 $\bar{u}$ 的相似度，选取Top-K关键帧 $\mathcal{F}^*$
- YOLOv8生成目标对象二值掩码 $M_i$，仅对检测到的物体区域施加扰动
- 采用PGD优化：$\epsilon = 32/255$，步长 $\alpha = 1/255$，迭代20次，动量衰减 $\lambda = 0.9$
- 损失函数：$\mathcal{L} = -\sum_{j=1}^{3} E_I(i^{adv}) \cdot E_T(C_j)$，最小化视觉-文本嵌入对齐
- 多尺度增强：$S = \{0.5, 0.75, 1.0, 1.25, 1.5\}$ 提升迁移性

**阶段三：时序攻击（Temporal Attack）**
- 基于相邻帧掩码差生成运动掩码：$\mathcal{M}_{motion} = |\mathcal{M}_{t+1} - \mathcal{M}_t| > \mathcal{T}_m$（阈值 $\mathcal{T}_m = 0.1$）
- 使用LanguageBind视频编码器提取时序特征，最大化以下两项损失：
  - 文本-视频对齐损失：$\mathcal{L}_{text} = \frac{1}{|\mathcal{T}|}\sum \cos(E_V(i^{adv}), E_T(t_j))$
  - 时序连贯性损失：$\mathcal{L}_{temporal} = \frac{1}{K-1}\sum \mathcal{M}_t \cdot \cos(E_V(i_t^{adv}), E_V(i_{t+1}))$
- 时序攻击参数：$\epsilon = 16/255$，$\alpha = 2/255$，迭代20次

## 实验与结果
**数据集**：
- BDD100K：1000视频，800用于攻击生成与评估
- nuScenes：85个场景，每场景约40帧

**目标模型**：Video LLaVA-7B、Qwen2.5-VL-7B、Dolphin（自动驾驶专用）

**代理模型**：TCL（空间攻击）、LanguageBind（时序攻击）

**主要结果（BDD100K）**：
| 模型 | 空间攻击ASR | 时空攻击ASR | SSIM |
|------|------------|------------|------|
| Video LLaVA-7B | 32.6% | **71%** | 0.82 |
| Qwen2.5-VL-7B | 45% | **84.2%** | 0.82 |
| Dolphin | 46.9% | 46.2% | 0.82 |

**主要结果（nuScenes）**：
- Video LLaVA：空间57.6% → 时空**83%**
- Qwen2.5-VL：空间64.7% → 时空**96.5%**
- Dolphin：空间37.6% → 时空47.1%
- 最大SSIM达0.98（空间阶段），时空阶段保持0.79

**对比基线**：
- PGD白盒迁移：Video LLaVA 47.2%，Qwen2.5-VL 44.8%，Dolphin 36.2%
- FGSM白盒迁移：Video LLaVA 25.2%，Qwen2.5-VL 35.4%，Dolphin 18.1%
- STCA全面超越，相对PGD提升约24%（Video LLaVA）、39%（Qwen2.5-VL）

**消融实验**：
- ASR随扰动预算 $\epsilon$ 单调递增（4/255→32/255）
- ASR随迭代步数增加而提升（5→40步）

**关键发现**：
- Dolphin作为领域专用模型表现出相对更强的鲁棒性（ASR仅~47%），暗示领域微调可能带来一定对抗防御能力
- Qwen2.5-VL最易受攻击（nuScenes达96.5%），其时序聚合机制对局部运动扰动更敏感
- 时序攻击阶段相比纯空间攻击带来显著增益（Video LLaVA提升38.4%，Qwen2.5提升39.5%）

## 相关工作脉络
- **AdvCLIP (Zhou et al.)**：针对CLIP的通用对抗扰动，关注视觉-文本对齐，但未考虑视频时序维度，攻击对象为静态图像。
- **VLAttack (Yin et al.)**：黑盒VLM攻击，利用预训练模型提取特征，但仅针对单模态输入，缺乏对视频时序一致性的破坏。
- **PG-Attack (Fu et al.)**：针对自动驾驶VLM的精确掩码扰动攻击，采用PMP框架，但未引入时序连贯性攻击，仅聚焦单帧空间语义。
- **ADvLM (Zhang et al.)**：首个针对自动驾驶VLM的攻击框架，关注Prompt语义不变性，但同样局限于静态图像扰动。
- **CAD (Wang et al.)**：黑盒自动驾驶VLM攻击，通过决策链破坏实现攻击，但未显式建模时序特征。
- **Breaking Temporal Consistency (Kim et al.)**：针对视频识别模型的时序攻击，通过降低相邻帧特征相似性实现，但未考虑VLM的跨模态语义对齐。

## 局限性与未来方向
- **Dolphin模型鲁棒性差异原因未深入分析**：领域微调带来的鲁棒性提升缺乏理论解释，未探讨具体是哪些训练机制或架构设计导致了该现象。
- **攻击计算成本较高**：需要调用多个模型（Gemini、Video-LLaVA、CLIP、YOLOv8、LanguageBind）进行多阶段生成，实际部署可行性受限。
- **仅评估了三种VLM模型**：未涵盖更多架构（如具有不同时序聚合机制的模型），结论普适性有待验证。
- **未考虑物理世界可实现性**：攻击在数字域验证，未研究物理场景中光照、视角变化对攻击有效性的影响。
- **缺乏防御机制探索**：仅揭示了时序脆弱性，未提出针对该攻击的防御方法，对实际系统安全指导有限。

## 研究启发与可借鉴点
- **Caption引导的关键帧选择策略可迁移**：利用CLIP跨模态相似度筛选语义重要帧的思路，可复用于视频剪辑、关键事件检测等任务。
- **运动掩码（Motion Mask）设计值得借鉴**：通过相邻帧掩码差定位动态区域，可有效区分目标对象与背景，提升攻击隐蔽性，可推广至其他视频对抗攻击场景。
- **领域专用模型的鲁棒性发现**：Dolphin的结果提示领域微调可能增强对抗防御，可进一步研究训练数据分布、架构设计对时序鲁棒性的影响。
- **多尺度增强的迁移性提升**：在空间攻击阶段应用多尺度扰动可显著提升跨模型迁移效果，适用于其他黑盒VLM攻击任务。
- **时序连贯性损失的构造方式**：$\mathcal{L}_{temporal}$ 通过掩码加权相邻帧特征相似度，为视频VLM的安全性评估提供了可量化的时序脆弱性指标。

## 关键术语表
- **Vision Language Model (VLM)**：融合视觉与语言理解的多模态大模型，可同时处理图像/视频和文本输入。
- **Attack Success Rate (ASR)**：对抗攻击成功率，指模型输出发生语义偏离的样本比例。
- **Structural Similarity Index (SSIM)**：结构相似性指标，衡量对抗样本与原始图像的视觉相似程度。
- **Temporal Coherence**：时序连贯性，指视频相邻帧之间在语义和运动上的连续性。
- **Motion Mask**：运动掩码，通过相邻帧差异定位视频中动态区域的二值掩码。
- **Cross-modal Alignment**：跨模态对齐，指视觉与文本表示在共享嵌入空间中的匹配程度。
- **Black-box Transferability**：黑盒迁移性，指在未知目标模型参数情况下，攻击从代理模型向目标模型的泛化能力。
- **Semantic Divergence**：语义分歧，通过词重叠相似度衡量攻击前后模型输出的语义差异。

## 可复现要素
- **数据集**：BDD100K（公开）、nuScenes（公开）
- **代码/权重**：论文未明确声明开源，但提及使用官方 checkpoint（Dolphin、Video LLaVA、Qwen2.5-VL）
- **关键超参**：
  - 空间攻击：$\epsilon = 32/255$，$\alpha = 1/255$，迭代20次，多尺度 $\{0.5, 0.75, 1.0, 1.25, 1.5\}$
  - 时序攻击：$\epsilon = 16/255$，$\alpha = 2/255$，迭代20次
  - YOLOv8置信度阈值：0.25，IoU阈值：0.45
  - 运动掩码阈值：0.1
  - 关键帧选择数：60帧/视频，Top-3 Caption
- **硬件环境**：NVIDIA A100 GPU，Google Colab Pro
