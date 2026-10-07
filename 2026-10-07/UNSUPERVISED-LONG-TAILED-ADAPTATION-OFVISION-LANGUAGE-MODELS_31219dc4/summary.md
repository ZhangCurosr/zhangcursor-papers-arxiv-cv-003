---
title: "UNSUPERVISED-LONG-TAILED-ADAPTATION-OFVISION-LANGUAGE-MODELS"
source: https://arxiv.org/pdf/2610.07903v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-07 17:53:10"
field: "视觉-语言模型无监督适配"
keywords: ["无监督适配", "长尾学习", "视觉-语言模型", "伪标签", "CLIP", "prompt tuning", "边界侵蚀"]
innovations: ["首次提出 ULTA 场景并揭示无监督长尾适配中头类边界反常侵蚀现象", "设计 BPA 利用 zero-shot CLIP 视觉锚点修正无视觉支持的伪标签概率提升", "提出 MSR 通过类别频率先验与样本边际一致性动态调节边界保留与特征学习"]
benchmarks: ["CUB", "RESISC45", "FGVCAircraft", "Caltech101", "EuroSAT", "Flowers102", "Food101", "OxfordPets", "StanfordCars", "Places-LT", "ImageNet-LT", "iNaturalist2018"]
---

# 论文速读：UNSUPERVISED-LONG-TAILED-ADAPTATION-OF-VISION-LANGUAGE-MODELS

## 一句话总结
本文首次研究视觉-语言模型在长尾无标签数据上的无监督适配（ULTA）任务，发现现有方法因均匀先验假设会导致头类决策边界被侵蚀；为此提出 MARS 框架，通过边界保持对齐（BPA）与边际感知自细化（MSR）两步策略，在九个基准数据集上平均提升 **4.71 个百分点**。

## 研究问题与动机
- **核心问题**：现有无监督 VLM 适配方法隐式假设无标签数据服从均匀分布，生成类别平衡的伪标签；但真实场景数据呈长尾分布，导致适配失败。
- **反常现象**：在监督长尾学习中尾部类表现最差，而在无监督长尾适配中，头部类性能反而急剧下降（Fig. 1(b)）。
- **机制剖析**：均匀约束迫使模型将头类样本错误分配为尾部伪标签，拟合这些错误样本会拉大视觉特征与正确文本嵌入的距离，导致头类 logit margin 显著缩小（Fig. 1(c)），同时放大 CLIP 对易混淆类的固有偏向（Fig. 1(d)）。
- **目标**：在无标签长尾数据上实现跨所有类别（头/中/尾）的鲁棒适配，不依赖类别频率先验。

## 核心贡献（创新点）
1. **首次形式化 ULTA 场景**：将无监督 VLM 适配扩展至真实长尾无标签数据分布，填补了该方向的研究空白。
2. **揭示头类边界侵蚀机制**：从梯度动态角度证明均匀伪标签迫使头类视觉特征向尾部文本嵌入对齐，导致决策边界崩塌；该发现与监督长尾学习中的尾部劣势模式截然不同。
3. **提出 BPA（边界保持对齐）**：以冻结的 zero-shot CLIP 作为视觉参考，通过计算类级视觉锚点 μ_c 并判断概率增加是否具备视觉支持，修正训练目标；与直接拟合模型自身错误预测的方法（如 CPL）本质不同。
4. **提出 MSR（边际感知自细化）**：采用教师-学生架构，利用类别频率先验 r_ŷ 与样本累积边际一致性（CMC）动态调节 KL 保留损失与 CE 学习损失的权重；与 CAP 等仅依赖类别对齐的方法相比，MSR 进一步区分同 classe 内不同样本的质量。
5. **SOTA 性能与强鲁棒性**：在九数据集、三种不平衡比（γ∈{10,20,50}）下平均提升 4.71 pp；在最具挑战的 γ=50 时仍比 zero-shot CLIP 高 8.08 pp，且性能衰减仅 1.27 pp（对比 CAP 的 5.97 pp）。

## 方法详解
MARS 遵循 **"先保护、后细化"** 的两阶段设计，仅训练 prompt 与文本编码器适配器，视觉/文本主干冻结。

### 3.3 Boundary-Preserving Alignment (BPA)
- **视觉锚点构建**：对每个 zero-shot 预测类别 c，聚合属于该类的弱增强样本特征 u_i，L2 归一化得到类级视觉锚点 μ_c = normalize(Σ_{i:c_i=c} u_i)。
- **视觉支持判定**：样本 i 对类别 j 的支持度 s_{ic} = u_i' · μ_c。若 s_{ic_i} > s_{ij}，则认为从 c_i 到 j 的概率提升缺乏视觉证据，记 h_{ij}=1。
- **跨样本聚合**：为避免单样本噪声，按类对累加支持度不一致的概率增量：ρ_{cj} = Σ_{i:c_i=c} g_{ij}h_{ij} / Σ_{i:c_i=c} g_{ij}，其中 g_{ij}=max(p_{ij}-a_{ij},0)。
- **目标修正**：将无视觉支持的增量归还给 zero-shot 类别：q_i = p_i + Σ_{j≠c_i} δ_{ij}(e_{c_i}-e_j)，δ_{ij}=ρ_{c_i j}h_{ij}g_{ij}。
- **损失函数**：L_BPA = -1/B Σ_i Σ_c q_{ic} log p(c|x_i^s)，其中 x_i^s 为 RandAugment 强增强视图。

### 3.4 Margin-Aware Self-Refinement (MSR)
- **教师-学生架构**：冻结 BPA 对齐后的教师模型，引入可学习适配器作为学生。
- **基础损失**：L_refine = α·L_KL(p^T,p^S) + (1-α)·L_CE(ŷ,p^S)，其中 ŷ=argmax_c p_c^T。
- **类别级先验门控**：α = (r_ŷ)^β，r_ŷ = n̂_ŷ / max_c(n̂_c)，n̂_c 为教师软预测累积估计的类别样本数。高频头类 α 大（保守保留边界），低频尾类 α 小（积极学习新知识）。
- **样本级边际一致性（CMC）**：定义边际 M^(t)(x,ŷ)=z_ŷ^(t)(x)-max_{j≠ŷ} z_j^(t)(x)，计算指数移动平均 CMC(x,ŷ)（动量 λ=0.95）。正确伪标签样本边际单调递增，错误样本边际波动或为负。
- **最终门控**：α̂ = 1/2(r_ŷ^β + (1-N(CMC(x,ŷ))))，N(·) 为 min-max 归一化。高质量样本获得更低的 α̂（更多学习），低质量样本获得更高的 α̂（更多保留）。

### 3.5 训练流程
- **Stage 1（BPA）**：10 个 epoch，batch size=64，SGD，lr=0.0035，weight decay=5e-4，warmup+cosine 调度。
- **Stage 2（MSR）**：30 个 epoch，冻结教师与视觉编码器，仅更新文本 prompt 与学生适配器。
- **推理**：丢弃教师适配器，使用 refined 编码器提取视觉特征 v*，与预计算文本嵌入 {t_c} 计算余弦相似度输出预测。

## 实验与结果
- **数据集**：CUB(200类)、RESISC45(45类)、FGVCAircraft(100类)、Caltech101(100类)、EuroSAT(10类)、Flowers102(102类)、Food101(101类)、OxfordPets(37类)、StanfordCars(196类)。长尾设置通过指数衰减下采样实现，γ∈{10,20,50}，每类最大 100 样本、最小 2 样本。
- **基线**：FPL、GRIP、CPL、UEO、TMP、CAP、microCLIP，统一使用 OpenAI CLIP ViT-B/32 作为视觉骨干。
- **主要结果**：
  - **γ=10**：MARS 平均准确率 **68.47%**，较前 SOTA（CAP 64.35%）提升 **4.12 pp**。
  - **γ=20**：MARS 平均 **67.69%**，较前 SOTA（microCLIP 63.19%）提升 **4.50 pp**。
  - **γ=50**：MARS 平均 **67.20%**，较前 SOTA（microCLIP 61.68%）提升 **5.52 pp**；比 zero-shot CLIP（59.12%）高 **8.08 pp**，而 CAP 等四个基线已低于 zero-shot。
  - **统计显著性**：在 189 组对比中，84.7%（160/189）呈现显著改进（p<0.05）。
  - **泛化性**：在 OpenCLIP ViT-B/32 上平均提升 8.79 pp；在 SigLIP ViT-B/16 上亦取得显著优势（如 RESISC45 γ=50 提升 5.10 pp）。
  - **大规模长尾基准**：Places-LT、ImageNet-LT、iNaturalist2018 上均获 SOTA，平均提升 2.51 pp；iNaturalist2018 上提升最佳基线 3.16 pp。
  - **平衡设置**：在均匀分布（γ=1）下 MARS 仍达 86.27% 平均准确率，略超最佳基线 0.30 pp。
- **消融**：BPA 单独贡献 +11.70 pp，MSR 单独 +9.79 pp，联合 +20.11 pp；BPA 对防止头类边界侵蚀起决定性作用。

## 相关工作脉络
1. **Prompt Learning for VLMs**：CoOp、VPT、MaPLe 等方法依赖有标签数据微调 prompt，无法直接应用于无标签场景；MARS 在无监督设定下扩展了 MaPLe 的多模态 prompt 框架。
2. **Pseudo-labeling 无监督适配**：UPL、GRIP、CPL 等通过置信度过滤或候选集生成伪标签，隐式假设均匀先验；CAP 虽识别了 CLIP 的类别混淆偏差并通过投影层校正，但未处理长尾分布；MARS 从根本上解决了均匀假设与长尾现实的冲突。
3. **熵优化与边界正则化**：UEO 使用边际熵统一度量决策边界，TMP 将适配重构为二值验证任务；MARS 通过 BPA 的视觉支持判定与 MSR 的边际一致性机制，在长尾条件下实现更精准的边界保护。
4. **Token Fusion 方法**：microCLIP 结合粗到细 token 融合与显著性感知池化；MARS 从目标修正与动态门控角度切入，两者正交可互补。
5. **长尾学习**：传统监督长尾学习关注尾部类提升；本文揭示无监督适配中长尾的"反转型"退化模式（头类受损更严重），提出了针对性的 preserve-before-refine 策略。

## 局限性与未来方向
- **依赖 zero-shot CLIP 作为参考**：BPA 的有效性部分建立在 zero-shot CLIP 视觉结构可靠的前提下，对于预训练质量较差的 VLM 可能受限。
- **两阶段训练开销**：先 BPA 后 MSR 的顺序设计增加了训练复杂度，实际部署时可能需要权衡效率。
- **仅限图像分类任务**：当前实验集中于分类基准，未来可扩展至检测、分割等更复杂视觉任务。
- **未探索极端长尾比**：γ=50 已具挑战性，但 γ→∞ 的极端场景下方法表现待验证。
- **未来方向**：将 MARS 思想迁移至视频理解、多模态生成等下游任务；探索无需 zero-shot 参考的自适应边界保护机制。

## 研究启发与可借鉴点
1. **以预训练模型为固定视觉参考的目标修正策略**：BPA 利用冻结的 zero-shot 模型提供视觉结构先验，避免自训练中的误差累积；该思想可迁移至其他自蒸馏或自监督适配场景。
2. **双粒度动态门控设计**：MSR 结合类别级频率先验与样本级边际一致性，同时处理类别不平衡与样本质量差异；该分层调制思路可用于其他伪标签细化任务。
3. **"先保护后细化"的 preserve-before-refine 范式**：先稳固可靠决策边界再针对困难样本优化，避免了端到端优化中边界崩塌的风险；该原则可指导其他长尾适配方法的设计。
4. **边际一致性作为伪标签质量代理**：CMC 利用训练过程中边际的指数移动平均判断样本正确性，无需额外标注；可作为通用的伪标签筛选信号。
5. **长尾无监督适配作为独立研究场景**：本文揭示了无监督长尾适配与监督长尾学习的"反向退化"现象，为后续研究提供了新的问题定义与分析框架。

## 关键术语表
- **ULTA（Unsupervised Long-Tailed Adaptation）**：视觉-语言模型在长尾分布无标签数据上的无监督适配新场景。
- **MARS（Margin-Aware Refinement with Structural alignment）**：本文提出的两阶段适配框架，包含 BPA 与 MSR 两个核心模块。
- **BPA（Boundary-Preserving Alignment）**：以 zero-shot CLIP 的类级视觉锚点为参考，修正缺乏视觉支持的概率提升，防止头类边界侵蚀。
- **MSR（Margin-Aware Self-Refinement）**：基于教师-学生架构，通过类别频率先验与样本边际一致性动态平衡边界保留与特征学习。
- **CMC（Cumulative Margin Consistency）**：对分类边际的指数移动平均，用于量化样本级伪标签质量。
- **Logit Margin（边际）**：目标类别 logit 与最高非目标类别 logit 之差，反映决策边界强度；头类边际缩小表明边界侵蚀。
- **Zero-shot CLIP**：未经微调的预训练 CLIP 模型，其视觉-文本对齐结构作为 BPA 的固定参考。
- **Pseudo-labeling（伪标签）**：利用模型自身预测作为监督信号进行自训练，是无监督 VLM 适配的核心技术。

## 可复现要素
- **数据集**：CUB、RESISC45、FGVCAircraft、Caltech101、EuroSAT、Flowers102、Food101、OxfordPets、StanfordCars 均为公开基准，长尾版本通过指数下采样生成；大规模基准 Places-LT、ImageNet-LT、iNaturalist2018 亦公开。
- **代码/权重**：论文未明确声明 MARS 代码开源状态；基线方法代码来自各官方公开仓库（已重新实现以确保公平对比）。视觉骨干为 OpenAI CLIP ViT-B/32、OpenCLIP ViT-B/32、SigLIP ViT-B/16。
- **关键超参**：β=1（所有数据集固定），λ=0.95（CMC 动量），Stage 1 学习率 0.0035、weight decay 5e-4、batch size=64、10 epochs；Stage 2 训练 30 epochs；prompt 深度 9、上下文长度 2，初始化模板 "a photo of a "；强增强使用 RandAugment。
