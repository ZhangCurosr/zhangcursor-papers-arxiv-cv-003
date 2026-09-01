---
title: "Real-Time-Video-Anomaly-Detection-Using-YOLO-Pose-Estimation"
source: https://arxiv.org/pdf/2608.31074v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 19:18:46"
field: "视频异常检测与视觉语言模型"
keywords: ["video anomaly detection", "CLIP", "YOLO pose estimation", "real-time surveillance", "zero-shot classification", "vision-language model"]
innovations: ["用CLIP零样本文本相似度替代多特征密度估计，实现3.36倍吞吐提升", "固定文本提示驱动的两阶段轻量管线，无需异常标注训练", "在真实CCTV流上验证5路并发实时部署（10 FPS/路）"]
benchmarks: ["CUHK Avenue", "ShanghaiTech Campus", "CU Indoor Anomaly"]
---

# 论文速读：Real-Time-Video-Anomaly-Detection-Using-YOLO-Pose-Estimation

## 一句话总结
本文提出了一种轻量级两阶段视频异常检测框架，以 YOLO v11n-pose 完成人物检测与骨骼关键点提取，再经 CLIP ViT-B/32 计算图像-文本余弦相似度实现异常语义评分；该方法消除了光流、独立姿态估计器和密度估计模块，在单张 NVIDIA Titan XP GPU 上实现约 51 FPS 的实时吞吐，同时保持与多特征基线相当的帧级 AUROC 性能。

## 研究问题与动机
- **现有 pipeline 算力代价过高**：如 Reiss & Hoshen [6] 的组合式方法（AlphaPose + FlowNet2 + CLIP + GMM/kNN）准确率尚可，但端到端吞吐仅约 15 FPS，无法满足实时监控需求。
- **光流与独立姿态估计的冗余性**：传统视频异常检测依赖 FlowNet2 等光流网络和 AlphaPose 等独立姿态估计器，增加了管线复杂度与延迟。
- **已有视觉-语言方法不适合实时部署**：VadCLIP [9] 等方法虽然利用 CLIP 进行弱监督异常检测，但依赖可学习提示词和离线分析设计，未在实时系统层面进行验证。
- **固定室内场景下人物级异常的在线检测需求**：研究面向固定室内 CCTV 环境，聚焦跌倒、躺地、打斗、坐地四类异常，目标是单张桌面级 GPU 上实现稳定 30+ FPS。

## 核心贡献（创新点）
1. **用直接 CLIP 语义评分替代多特征密度估计**：将 AlphaPose + FlowNet2 + GMM/kNN 的重型 pipeline 压缩为仅 YOLO + CLIP 两模块，吞吐量提升 3.36×，在 CUHK Avenue 和 ShanghaiTech Campus 上与基线持平、在 CU Indoor Anomaly 上超越基线约 2 个百分点。
2. **文本提示驱动的零样本异常分类机制**：通过预定义 4 条自然语言描述（如"A person lying on the floor"），利用 CLIP 零样本能力直接对多种异常类型进行分类，无需异常标注训练数据。
3. **在真实 CCTV 流上的工程化部署验证**：在朱拉隆功大学 12 层电梯口 5 路实时监控流上稳定运行（约 10 FPS/路），包含 GUI 叠加和分数可视化，确认系统级可行性。

## 方法详解
**整体架构**：两阶段串行管线，Stage 1 为人物检测与姿态估计，Stage 2 为基于 CLIP 的语义异常评分。

- **Stage 1：YOLO v11n-pose 检测与姿态回归**
  对输入帧 $I_t \in \mathbb{R}^{H \times W \times 3}$，YOLO v11n-pose 单次前向传播同步输出每个人物的边界框 $b_i = (x_i, y_i, w_i, h_i)$（置信度 $c_i$）以及 17 个 COCO 拓扑关键点 $\mathbf{k}_i = \{(x_j^k, y_j^k, v_j^k)\}_{j=1}^{17}$，其中 $v_j^k$ 为关键点可见性标志。以 padding $\delta = 10$ 像素裁剪人物区域：
  $$p_i = I_t[y_i - \delta : y_i + h_i + \delta,\ x_i - \delta : x_i + w_i + \delta]$$
  该 padding 值经 CU Indoor Anomaly 数据集（640×480）经验选取，约为帧宽的 1.6%。

- **Stage 2：CLIP ViT-B/32 语义异常评分**
  每张 $p_i$ Resize 至 224×224 后，经 CLIP 视觉编码器得到归一化嵌入 $\mathbf{f}_i^{\text{img}} \in \mathbb{R}^{512}$；4 条固定文本提示经 CLIP 文本编码器得到 $\mathbf{f}_j^{\text{txt}} \in \mathbb{R}^{512}$（初始化时计算并缓存）。
  人物级异常分：
  $$s_i^t = \max_{j \in \{1,\dots,M\}} \mathbf{f}_i^{\text{img}} \cdot \mathbf{f}_j^{\text{txt}}$$
  帧级异常分（取所有检测人物的最大值）：
  $$S_t = \max_{i \in \{1,\dots,N_t\}} s_i^t$$
  其中 $M=4$，文本提示集为：
  - "A person lying on the floor"
  - "A person falling down"
  - "A person fighting with another person"
  - "A person sitting on the floor"

- **时序平滑与分类决策**
  原始分数经一维高斯核平滑（$\sigma=5$，半窗口 $w=15$）得 $\hat{S}_t$，与阈值 $\theta$ 比较：$\hat{S}_t \geq \theta$ 判定为异常帧。最优阈值 $\theta^*=0.7$ 通过在验证集上 sweeping $[0.1, 0.9]$ 最大化 AUROC 确定。

- **训练细节**
  YOLO v11n-pose 以 COCO 预训练权重初始化，在 CU Indoor Anomaly 的 2864 张训练图（由 1154 帧人工标注增强）上 fine-tune 500 epochs（640×640，batch=16，SGD，lr=0.01）；CLIP ViT-B/32 直接使用预训练权重，不做 fine-tune。"零样本"一词特指 CLIP 评分模块无需异常标注。

## 实验与结果
**数据集**：
- **CUHK Avenue** [13]：37 个视频，30,652 帧，固定大学通道摄像头。
- **ShanghaiTech Campus** [14]：437 个视频，317,398 帧，13 种校园场景。
- **CU Indoor Anomaly**（自建）：40 个视频，17,798 帧（11,841 正常 / 5,957 异常），覆盖 4 类异常（跌倒、躺地、坐地、打斗）；33 训练 / 7 测试视频划分。

**评估指标**：帧级 AUROC（主指标）；YOLO 检测侧报告 Precision、Recall、mAP@.5、mAP@.5:.95；吞吐率以端到端 FPS 计。

**主要结果**（Table I）：

| 方法 | Avenue | ShanghaiTech | CU Indoor |
|---|---|---|---|
| Multi-feat. baseline [6] | 89.26% | 70.26% | 82.12% |
| Proposed（本文） | **89.26%** | **70.26%** | **84.13%** |
| DSTN [10] | 86.40% | 73.10% | — |
| Attr-Based [6] | 86.00% | 71.10% | — |

- 最强结果：CU Indoor Anomaly 上 **84.13%** AUROC，较重基线提升约 **+2.01 pp**；Avenue 与上海Tech与基线持平（因共享同一 YOLO 检测器，差异仅在评分模块）。
- 相较 DSTN，Avenue 上高出 **2.86 pp**，且吞吐率高出一个数量级。

**YOLO 检测性能**（Table II，CU Indoor 测试集，360 个人物实例）：
- 总体：Precision=0.91，Recall=0.97，mAP@.5=0.92，mAP@.5:.95=0.67
- 正常姿态：Prec=0.95，Rec=0.98，mAP@.5=0.96
- 异常姿态：Prec=0.86，Rec=0.96，mAP@.5=0.88

**吞吐率**（Table III，单张 NVIDIA Titan XP）：
- 基线端到端：**15.32 FPS**
- 本文端到端：**51.39 FPS**（**3.36× 加速**）
- 各组件：YOLO v11n-pose 87.76 FPS，CLIP 评分 124.04 FPS；基线中 FlowNet2、AlphaPose、GMM/kNN 分别构成瓶颈。

**真实部署**：朱拉隆功大学 12 楼电梯口 5 路 CCTV 流，稳定约 **10 FPS/路**（含解码、渲染、GUI 更新开销）。

## 相关工作脉络
1. **Reiss & Hoshen [6]（Attr-Based / Multi-feature baseline）**：结合 AlphaPose、FlowNet2、CLIP 与 GMM/kNN 密度估计的多组件方案；本文与其关系是"同检测器、换评分器"——用 CLIP 余弦相似度替代密度估计，以大幅换取吞吐。
2. **VadCLIP [9]**：利用可学习 prompt tokens + 弱监督训练进行视频异常检测；本文与之定位不同：本文采用**固定文本提示 + 零样本评分**，无需训练即部署，专为实时系统设计。
3. **Moriyama et al. [12]（Language-guided decision override）**：基于 CLIP 的免训练重校准方法；本文共享其 vision-language 思路，但进一步将系统简化为两模块端到端实时管线。
4. **Hasan et al. [3]（Conv-AE）**：经典重构类异常检测方法（训练正常视频自编码器，重构误差作为异常信号）；本文放弃无监督重构范式，直接利用预训练 VLM 的语义对齐能力。
5. **Morais et al. [11]（MPED-RNN）**：基于骨骼轨迹的多时间尺度 RNN 学习正常运动规律；本文用单帧 CLIP 文本相似替代时序骨架建模，显著降低复杂度。
6. **Ganokratanaa et al. [10]（DSTN）**：融合背景消除帧与光流的时空翻译网络做像素级异常定位；本文仅做帧级/人物级分类，不做像素定位，但吞吐量显著更高。

## 局限性与未来方向
- **固定提示词覆盖有限**：上海 Tech Campus AUROC 偏低（70.26%）源于骑行、车辆闯入等异常不在 4 条提示范围内；提示集缺乏可扩展性。
- **仅针对室内人物级异常**：不含人参与的异常（如车辆闯入）无法检测；泛化到室外/开放场景受限。
- **YOLO 检测器仅 indoors 微调**：COCO 预训练 + CU Indoor 微调，跨域迁移到显著不同的监控环境时检测精度可能下降。
- **未来方向**（作者自述）：① 引入 VadCLIP 启发的可学习提示词扩展以拓宽异常覆盖；② 低光照鲁棒预处理；③ 开发面向资源受限边缘设备的统一端到端模型。

## 研究启发与可借鉴点
1. **"重评分替换代替全管线重构"**：对于已有重型多组件基线，仅替换评分模块（用 VLM 零样本语义评分替代密度估计）即可在几乎不牺牲精度的前提下获得数倍吞吐增益——这一"模块化替换"思路可迁移至其他多组件系统的轻量化改造。
2. **Padding 参数经验调优方法**：人物裁剪 padding=10 像素（约帧宽 1.6%）的选取逻辑——在上下文保留与背景噪声之间权衡——可作为类似管线中 crop 策略的参考模板。
3. **阈值 sweeping 替代交叉验证**：在 AUROC 优化中直接 sweep 决策阈值选 $\theta^*$，比网格搜索更直接高效，适合部署阶段快速校准。
4. **VLM + 检测器解耦设计的通用性**：YOLO（检测）与 CLIP（语义评分）完全解耦，可独立替换升级——例如未来 YOLOv12 或更大的 CLIP 变体均可无缝接入，架构具有良好的演进弹性。
5. **与团队方向结合机会**：将"固定提示词"扩展为"LLM 生成候选提示 + 主动学习筛选"的渐进式提示挖掘管线，可在不改变两阶段架构的前提下显著扩展异常类别覆盖，尤其适用于多场景监控系统的持续部署。

## 关键术语表
**YOLO v11n-pose**：Ultralytics 推出的轻量级 YOLOv11 nano 版本，支持单次前向传播同时输出人物边界框与 17 个 COCO 关键点，适合实时姿态感知检测。

**CLIP（ViT-B/32）**：OpenAI 的 Vision-Language 预训练模型，将图像与文本映射到同一 512 维嵌入空间，支持通过余弦相似度进行零样本分类。

**Frame-level AUROC**：以帧为单位计算的 ROC 曲线下面积，衡量异常帧得分高于正常帧的概率，是视频异常检测的主流评估指标。

**Zero-shot anomaly classification**：利用预训练 VLM 的文本-图像对齐能力，在不使用异常标注数据训练的情况下，通过自然语言提示直接分类异常行为。

**GMM/kNN 密度估计**：在特征空间中用高斯混合模型（GMM）建模正常分布，再对测试样本用 k 近邻（kNN）计算偏离度作为异常分；本文方法的核心对比基线。

**Text-prompt-based semantic scoring**：将每张人物裁剪图的 CLIP 嵌入与多条异常行为文本嵌入做余弦相似度比较，取最大相似度作为异常得分的评分策略。

**Temporal smoothing（Gaussian kernel）**：对时序异常分序列施加一维高斯核平滑（σ=5, w=15），抑制瞬时波动，使最终判决更稳定。

**CU Indoor Anomaly**：作者团队在朱拉隆功大学 4 栋建筑内自行采集的室内异常监控数据集，含 40 视频 / 17,798 帧，标注 4 类人物异常。

## 可复现要素
- **数据集**：CUHK Avenue [13]、ShanghaiTech Campus [14] 为公开数据集；**CU Indoor Anomaly 为自建数据集，未提及开源**。
- **代码**：论文未明确声明开源仓库。
- **权重**：YOLO v11n-pose 使用 COCO 预训练权重（公开发布），微调后权重未见开源声明；CLIP ViT-B/32 使用官方预训练权重（公开发布）。
- **关键超参**：YOLO 输入分辨率 640×640，batch=16，lr=0.01，500 epochs；CLIP 输入 224×224；裁剪 padding δ=10 像素；高斯平滑 σ=5，半窗口 w=15；决策阈值 θ*=0.7（通过 sweep [0.1, 0.9] 选取）。
- **硬件**：NVIDIA Titan XP 单卡。
