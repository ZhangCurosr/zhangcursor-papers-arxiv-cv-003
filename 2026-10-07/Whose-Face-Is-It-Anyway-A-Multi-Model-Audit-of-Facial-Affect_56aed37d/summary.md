---
title: "Whose-Face-Is-It-Anyway-A-Multi-Model-Audit-of-Facial-Affect"
source: https://arxiv.org/pdf/2610.08279v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-07 17:53:41"
field: "面部表情识别公平性与偏差审计"
keywords: ["facial affect recognition", "bias audit", "children", "action unit", "linear probe", "head recalibration", "domain shift", "valence-arousal"]
innovations: ["首个五模型共享管道的儿童FER审计，证明儿童差距模型无关且集中在张口脸-AU26机制", "头而非特征的诊断：线性probe证明冻结特征可达0.75-0.91，差距主要位于分类头校准", "仅重训线性头+10-25名儿童数据即可实现+0.13至+0.28的提升且成人代价 negligible"]
benchmarks: ["AffectNet-8", "CAFE", "NIMH-ChEFS", "RaFD-Kid", "ChildEFES", "LIRIS-CSE", "EmoReact", "ABAW"]
---

# 论文速读：Whose-Face-Is-It-Anyway-A-Multi-Model-Audit-of-Facial-Affect

## 一句话总结
本文对五个成人预训练的 facial affect 模型在儿童数据集上进行共享基准测试，发现儿童识别差距具有**模型无关性**、**人口学集中性**（张口脸→惊讶误判、南亚裔儿童偏差），且差距**主要位于分类头校准**而非表征本身；仅需少量儿童标注数据重训线性头即可显著提升性能，但增益不跨数据集迁移。

---

## 研究问题与动机
- 现有 FER 模型几乎全部在成人数据（AffectNet、RAF-DB、VGGFace2 等）上训练，却在教育、健康、发展研究中越来越多地应用于儿童，但缺乏对"成人-儿童"域间差距的系统性评估。
- 已有儿童 FER 评估多为**单模型、单数据集**（如 [34][35]），无法区分特定模型的偶发缺陷与普遍性的域间差距，也无法定位失败机制。
- 关键科学问题：**为什么成人预训练模型在儿童上表现下降？差距在于表征还是分类头？**
- 此外，dimensional valence/arousal（V/A）回归的域迁移问题尚未在儿童场景下被系统研究，且与 categorical 识别之间存在潜在的不平衡性。

---

## 核心贡献（创新点）
- **首个共享 harness 的五模型儿童 FER 审计**：五种 AffectNet 预训练模型（EmoNet、EmotiEffLib、DDAMFN++、OpenFace 3.0、LibreFace）在四个儿童图像数据集、AffectNet-8 验证集及两个儿童视频数据集上通过统一管道评估，消除了单模型单数据集的混淆。
- **建模无关的儿童差距发现**：所有五个模型均从摆拍（RaFD）到自然面部（CAFE）全面退化，且共享恐惧→惊讶误判模式，证明差距源于数据分布而非模型结构。
- **AU26 下颚张开→惊讶误判的机制定位**：首次将儿童表情识别偏差与具体的 facial action unit（AU26 jaw drop）相关联，建立剂量-响应单调关系，并验证了两个独立模型（LibreFace、OpenFace 3.0）的 AU26 预测与惊讶误判的强相关性。
- **"头而非特征"的诊断与低成本修复**：线性 probe 实验证明冻结特征后，线性分类头在未见儿童上可达 0.75–0.91（vs 零样本 0.48–0.66），提出仅重训线性头的固定骨干策略，用约 10–25 名儿童数据即可实现 +0.13–+0.28 的提升，且对成人类别几乎无损。

---

## 方法详解

### 1. 统一审计框架（Shared Harness）
- **公共标签空间**：将所有数据集标签映射到 AffectNet-8 体系（neutral, happy, sad, surprise, fear, disgust, anger, contempt），模型原生输出顺序经软冒烟测试验证后重映射。
- **每模型原生预处理**：各模型保持其训练协议（EmoNet 用 FAN 地标裁剪至 256；EmotiEffLib/DDAMFN++ 用人脸 bbox；OpenFace 3.0/LibreFace 自检测），高分辨率儿童图像上用共享 FAN 裁剪控制检测器方差。
- **指标设计**：分类用 accuracy 和 macro-F1；视频聚合为 clip-level majority vote；V/A 用 concordance correlation coefficient (CCC)。

### 2. 偏见切片与统计检验
- **人口学切片维度**：mouth open/closed（CAFE）、race（CAFE: White/Black/South-Asian/East-Asian/Latino）、gaze（NIMH: direct/averted）、age、gender。
- **统计方法**：每模型用 two-proportion z-test，跨模型用 participant-level cluster bootstrap（重采样儿童而非图像以避免伪重复），并用 sign test 验证跨模型一致性。

### 3. 线性 Probe 分析（Sec. 6）
- 冻结各模型 backbone，提取嵌入向量，在 participant-disjoint 的 CAFE 子集（108 train / 46 test subjects）上训练线性分类器，对比零样本精度。

### 4. 头部重校准方法（Sec. 7）
- **冻结骨干 + 线性头重训**：在 AffectNet-train（成人）与儿童数据集的类平衡池化样本上重训线性表达头，扫描儿童 up-weight w 以追踪成人/儿童权衡前沿。
- **数据效率**：目标儿童数扫描 5–50，发现在 10–25 名儿童（约 190 帧）时增益饱和。
- **与微调对比**：fine-tune last block+head（1.1M 参数）对比冻结头策略，证明后者在两个带 V/A 头的模型上更优。

---

## 实验与结果

### 数据集
- **四个儿童图像数据集**：RaFD-Kid（摆拍，N=80）、ChildEFES（N=158）、NIMH-ChEFS（N=533，含 gaze 标签）、CAFE（N=1192，open/closed-mouth）。
- **两个儿童视频数据集**：LIRIS-CSE（6-class，clip-level majority vote）、EmoReact（multi-label，per-clip ROC-AUC）。
- **成人基准**：AffectNet-8 验证集（raw 4000）、ABAW/AffWild2（valence-arousal track）。

### 图像数据集结果（表 3）
| 模型 | RaFD | ChildEFES | NIMH | CAFE | mean |
|------|------|-----------|------|------|------|
| EmoNet | .963 | .759 | .666 | .634 | .756 |
| EmotiEffLib | .950 | .703 | .690 | .637 | .745 |
| DDAMFN++ | .963 | .741 | .578 | .636 | .730 |
| OpenFace 3.0 | .887 | .601 | .480 | .492 | .615 |
| LibreFace | .662 | .525 | .675 | .575 | .609 |

- 所有模型从 RaFD 到 CAFE 全面退化；EmotiEffLib 在最难数据集（NIMH、CAFE）上最稳健。

### AffectNet-8 成人验证（表 4）
- 现代模型 Acc-8 聚集于 0.44–0.68，EmoNet 原始验证精度 0.594（其声称 0.75 为 cleaned subset + TTA artifact）。

### 视频结果
- **LIRIS-CSE**：五模型 accuracy 0.30–0.42，majority baseline（always "happy"）= 0.342，EmotiEffLib 最佳仅超基线约 0.08。
- **EmoReact**：macro AUC 0.54–0.62，仅 happiness 显著高于 chance。

### 偏见切片结果（表 5）
- **张嘴脸**：hard=0.46 vs control=0.69，5/5 模型下降，p<0.001（participant bootstrap p=0.0002）。
- **南亚裔儿童**：hard=0.53 vs White=0.61/Black=0.60，5/5 模型下降，p=0.02（exploratory）。
- **视线回避**：direct 优于 averted 0.06，5/5 模型方向一致，p≈0.011。
- **闭合嘴、White/Black 儿童、直接视线、性别、年龄**：无显著退化。

### 线性 Probe（表 6）
| 模型 | zero-shot | linear probe | ∆ |
|------|-----------|-------------|----|
| EmotiEffLib | .638 | .910 | +.272 |
| EmoNet | .657 | .878 | +.221 |
| OpenFace 3.0 | .484 | .859 | +.375 |
| DDAMFN++ | .638 | .843 | +.205 |
| LibreFace | .593 | .747 | +.154 |

### 头部重校准（表 7，w=10）
- **CAFE 提升**：+0.21 至 +0.27（所有五模型）；**NIMH 提升**：+0.13 至 +0.28。
- **成人成本**：最大仅 -0.015。
- **机制修复**：EmotiEffLib 张嘴脸精度 0.49→0.85，fear→surprise 误判率 0.38→0.07。
- **跨数据集迁移弱**：训练三个儿童集测试第四个，平均提升仅 +0.05/+0.04，远不如 +0.2 的 in-distribution 增益。

### V/A 回归结果
- EmoNet：AffectNet-val CCC(Val)=0.68，ABAW holdout=0.21，EmoReact=0.22。
- 三种 post-hoc 路由对比（EmotiEffLib）：回归特征=0.218，categorical posterior→V/A=0.250，fine-tune backbone=0.276，均远低于 native joint head=0.375。

---

## 相关工作脉络
- **Gender Shades [3]**：成人性别分类的交叉偏见审计（3 个商业系统），本文将其思路扩展到 FER 儿童场景并增加 AU 机制定位。
- **Xu et al. [37]**：单模型 FER 偏见审计（gender/race/skin tone），本文扩展至多模型 + 儿童 + AU 机制。
- **Domnich & Anbarjafari [7]**：5 模型 FER 性别偏见审计（无 subgroup 深度切片、无机制、无修复），本文补充了人口学切片与可部署修复。
- **Kim et al. [19] / Gaya-Morey et al. [12]**：成人年龄偏见研究，本文关注儿童 vs 成人这一更极端的域间差距。
- **Virgolin et al. [34] / Vivekananthan [35]**：单模型儿童 FER 研究（分别用 contrastive augmentation 和 synthetic augmentation），本文首次在多模型共享管道下定位差距根源。
- **Infant AU 检测器 [10][14][15]**：聚焦婴儿 AU 检测或粗粒度 affect，本文针对实际部署的成人预训练 FER 模型在儿童上的系统性审计。

---

## 局限性与未来方向
- **头重校准仅 in-distribution 有效**：跨儿童集合迁移性弱（不同数据集类别质心方向各异），需从部署群体采集数据。
- **V/A 回归仍是开放问题**：儿童维表数据缺乏 comparable 标签，categorical-to-dimensional 的鸿沟未闭合。
- **视频感知仍为帧级基线**：自发儿童视频需 temporal 架构建模表情动力学。
- **种族偏差归因假设**：南亚裔偏差归因于训练分布不足，尚需更多数据验证；皮肤色调被排除但需更多测量证据。
- **方法泛化性待验证**：head-not-features 框架可扩展至老年人、临床群体、跨文化场景，但未在该论文中验证。

---

## 研究启发与可借鉴点
- **共享 harness 审计范式**：跨模型、跨数据集的统一评估管道是识别真实域间差距而非模型特有偏差的关键，可迁移至其他 under-represented 群体的公平性审计。
- **AU 机制定位偏差**：将性能退化与具体 Action Unit（AU26）关联，提供了可解释的、可操作的故障诊断路径，而不仅是统计层面的 accuracy gap。
- **冻结骨干 + 线性头重训的性价比**：仅需 ~10–25 名目标群体样本（约 190 帧）即可大幅弥合差距，为低资源场景下的领域适配提供了实用的工程方案。
- **Categorical vs Dimensional 的不对称可恢复性**：证明分类通道比 V/A 回归更具域迁移鲁棒性，提示未来系统应优先通过 categorical 通道路由 affect 估计，再映射至 dimensional 空间。
- **participant-level bootstrap 避免伪重复**：在多图来自同一受试者的数据中，重采样受试者而非图像是关键统计规范，值得在类似审计中推广。

---

## 关键术语表
- **Facial Affect Recognition (FER)**：从面部图像/视频中自动识别情绪类别或维度（valence/arousal）的任务。
- **AffectNet**：大规模野外面部表情数据集，包含 8 类表情及连续 valence/arousal 标注，是多个审计模型的主要预训练源。
- **Linear Probe**：冻结 backbone 特征提取器，仅在冻结嵌入上训练线性分类器，用于评估表征质量与下游任务可分离性。
- **Action Unit (AU)**：FACS（面部动作编码系统）定义的独立面部肌肉动作单元；AU26 为 jaw drop（下颚下降），是本文定位的核心机制。
- **Concordance Correlation Coefficient (CCC)**：衡量 valence/arousal 预测值与标注值之间一致性的指标，ABAW 挑战赛的核心评测指标。
- **Head Recalibration**：仅重训分类头（线性层），冻结骨干网络，在少量目标域数据上重新校准决策边界的低代价适配方法。
- **Participant-level Cluster Bootstrap**：以受试者为单位的重采样统计方法，避免同一受试者多图像导致的伪重复和过置信。
- **Class-conditional Domain Shift**：不同数据集在特征空间中同一类别的质心位置发生不同方向偏移的现象，解释了头重校准为何不能跨数据集迁移。

---

## 可复现要素
- **数据集**：四个儿童图像数据集（CAFE、ChildEFES、NIMH-ChEFS、RaFD-Kid）及两个视频数据集（LIRIS-CSE、EmoReact），**数据不重新分发**，受原始 license 约束；AffectNet-8 验证集（raw 4000）为标准公开集。
- **代码**：共享 harness、每样本预测结果、分析代码**将开源发布**。
- **模型权重**：五个模型均使用公开可获取的 AffectNet-pretrained 权重（EmoNet、EmotiEffLib/HSEmotion、DDAMFN++、OpenFace 3.0、LibreFace）。
- **关键超参**：linear probe 训练集 108 subjects/测试集 46 subjects（CAFE）；head recalibration child up-weight w=10；fine-tune 1.1M 参数（last block+head）；AffectNet anchor 8k 帧。

---
