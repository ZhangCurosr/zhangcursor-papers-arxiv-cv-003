---
title: "Real-Time-Video-Anomaly-Detection-Using-YOLO-Pose-Estimation"
source: https://arxiv.org/pdf/2608.31074v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 10:40:14"
field: "视频异常检测"
keywords: ["video anomaly detection", "CLIP", "YOLO pose estimation", "zero-shot classification", "real-time processing", "vision-language model"]
innovations: ["用CLIP零样本图文余弦相似度替代多特征GMM/kNN密度估计，实现3.36×吞吐提升", "固定4条文本prompt实现无需异常标签训练的多类异常分类", "YOLO v11n-pose与CLIP的两阶段轻量架构在单GPU上实现51 FPS实时检测"]
benchmarks: ["CUHK Avenue", "ShanghaiTech Campus", "CU Indoor Anomaly"]
---

# 论文速读：Real-Time-Video-Anomaly-Detection-Using-YOLO-Pose-Estimation

## 一句话总结
本文提出了一种轻量级两阶段框架，利用 YOLO v11n-pose 进行人体检测与姿态估计，再用 CLIP ViT-B/32 的文本-图像余弦相似度实现异常语义评分，在不依赖光流、独立姿态估计器和密度建模的前提下，在 NVIDIA Titan XP 上实现了约 51 FPS 的实时异常检测。

## 研究问题与动机
- **现有方法吞吐量低**：Reiss & Hoshen [6] 的多特征 pipeline（AlphaPose + FlowNet2 + CLIP + GMM/kNN）虽然精度高，但端到端仅约 15 FPS，难以满足实时监控需求。
- **CLIP 视频异常检测偏向离线**：VadCLIP [9] 等 CLIP-based 方法依赖可学习 prompt 和弱监督训练，设计目标是离线分析，而非实时部署。
- **缺乏兼顾速度与精度的简单方案**：传统重建/预测/属性基方法需要复杂的多组件流水线（光流、骨架估计、密度估计），系统开销大。
- **室内固定摄像头的特定场景需求**：针对高校/机构内 CCTV 中人员跌倒、打斗、躺卧、坐地等人员级异常，需要在单张消费级 GPU 上实现 >30 FPS 的稳定运行。

## 核心贡献（创新点）
1. **用 CLIP 语义评分替代多特征密度估计**：将 AlphaPose + FlowNet2 + GMM/kNN 的多阶段管线精简为 YOLO + CLIP 的两阶段结构，在保持相当精度的同时实现 3.36× 吞吐提升，本质区别在于摒弃了逐组件的特征提取，直接用零样本图文匹配完成评分。
2. **固定文本提示的零样本异常分类机制**：仅通过 4 条人工编写描述性 prompt（如"A person lying on the floor"），无需任何带标签的异常训练数据即可对多类异常进行识别与分类，与 VadCLIP 需可学习 prompt + 弱监督训练形成对比。
3. **面向真实部署的系统集成验证**：在 CUHK Avenue、ShanghaiTech Campus 和自建 CU Indoor Anomaly 三个数据集上评测，并在朱拉隆功大学 5 路真实 CCTV 实时部署（约 10 FPS/路），验证了工程可行性。

## 方法详解
框架分为两个顺序阶段：

**阶段一：YOLO v11n-pose 人体检测与姿态估计**
- 输入帧 $I_t \in \mathbb{R}^{H \times W \times 3}$，YOLO v11n-pose 单次前向传播同时输出每个人体的边界框 $b_i = (x_i, y_i, w_i, h_i)$（置信度 $c_i$）和 17 个 COCO 拓扑关键点 $\mathbf{k}_i = \{(x_j^k, y_j^k, v_j^k)\}_{j=1}^{17}$（含可见性标志 $v_j^k$）。
- 以边界框为中心、 padding $\delta = 10$ 像素裁剪人体区域：$p_i = I_t[y_i-\delta : y_i+h_i+\delta, \; x_i-\delta : x_i+w_i+\delta]$，保留周围上下文信息。

**阶段二：CLIP ViT-B/32 语义异常评分**
- 每个 crop $p_i$ 缩放到 $224 \times 224$，经 CLIP 视觉编码器得到归一化嵌入 $\mathbf{f}_i^{\text{img}} \in \mathbb{R}^{512}$。
- 4 条预定义异常文本 prompt 经 CLIP 文本编码器得到 $\mathbf{f}_j^{\text{txt}} \in \mathbb{R}^{512}$，初始化时计算一次并缓存。
- 单人异常分：$s_i^t = \max_{j \in \{1,\dots,M\}} \mathbf{f}_i^{\text{img}} \cdot \mathbf{f}_j^{\text{txt}}$（余弦相似度）。
- 帧级异常分：$S_t = \max_{i \in \{1,\dots,N_t\}} s_i^t$（取帧内所有人最高分）。

**时序平滑与决策**
- 对 $S_t$ 用一维高斯核（$\sigma = 5$，半窗 $w = 15$）平滑得 $\hat{S}_t$。
- 阈值 $\theta^* = 0.7$（在验证集上 sweep [0.1, 0.9] 以最大化 AUROC 确定），$\hat{S}_t \geq \theta$ 判定为异常帧。

**训练细节**
- YOLO v11n-pose：COCO 预训练权重初始化，在 CU Indoor Anomaly 2864 张增强训练图上 fine-tune 500 轮（640×640，batch=16，SGD lr=0.01）。
- CLIP ViT-B/32：直接使用预训练权重，不 fine-tune。

## 实验与结果
**数据集**
- CUHK Avenue：37 视频，30,652 帧
- ShanghaiTech Campus：437 视频，317,398 帧
- CU Indoor Anomaly（自建）：40 视频，17,798 帧（11,841 正常 / 5,957 异常），4 类异常标注，33 训练 / 7 测试

**评估指标**：帧级 AUROC（主要）；YOLO 检测：Precision、Recall、mAP@.5、mAP@.5:.95；吞吐量：FPS

**主要结果（Table I）**
| 方法 | Avenue | ShanghaiTech | CU Indoor |
|---|---|---|---|
| Conv-AE [3] | 80.00 | 60.85 | — |
| MPED-RNN [11] | 83.50 | 73.40 | — |
| DSTN [10] | 86.40 | 73.10 | — |
| Attr-Based [6] | 86.00 | 71.10 | — |
| 多特征基线† | 89.26 | 70.26 | 82.12 |
| **本文** | **89.26**‡ | **70.26** | **84.13** |

- Avenue 与多特征基线持平（89.26%），因两者共享同一 YOLO 检测器。
- ShanghaiTech 70.26% 受限于 4 条固定 prompt 无法覆盖骑行、车辆入侵等异常。
- CU Indoor 84.13%，超越基线约 2 个百分点，归因于 CLIP 图文对齐与室内四类异常的强语义匹配。

**YOLO 检测性能（Table II，CU Indoor 测试集）**
- 总体：Prec. 91%，Rec. 97%，mAP@.5 = 92%
- 正常姿态：Prec. 95%，Rec. 98%；异常姿态：Prec. 86%，Rec. 96%

**吞吐量（Table III）**
- 基线端到端：15.32 FPS；本文端到端：**51.39 FPS**（3.36× 加速）
- 瓶颈从基线检测（23.41 FPS）转移至 YOLO v11n-pose（87.76 FPS），CLIP 评分达 124.04 FPS
- 实地部署：5 路 CCTV 稳定运行，每路约 10 FPS

## 相关工作脉络
- **Reiss & Hoshen [6]**（Attr-Based 多特征基线）：结合 AlphaPose + FlowNet2 + CLIP + GMM/kNN，精度强但 15 FPS；本文在同一检测器基础上以 CLIP 直接语义评分替代密度估计，获得 3.36× 加速。
- **VadCLIP [9]**：适配 CLIP 进行弱监督视频异常检测，依赖可学习 prompt token 和离线分析；本文用固定手工 prompt 实现零训练部署，定位实时推理。
- **Moriyama et al. [12]**：语言引导的 CLIP 决策覆盖（retraining-free），偏离线自适应；本文强调端到端实时管线集成。
- **DSTN [10]**：深度时空翻译网络 + 光流做像素级定位；本文不用光流，仅做人员级检测与评分，速度远超。
- **MPED-RNN [11]**：基于骨架轨迹的多时态 RNN 学习正常运动规律；本文直接跳过显式运动建模，用 CLIP 图文相似度隐式捕获异常语义。
- **Hasan et al. [3]**（Conv-AE）：重建误差型方法；本文属于分类/匹配范式，在 Avenue 上显著超越（89.26% vs 80.00%）。

## 局限性与未来方向
- **固定 prompt 覆盖有限**：ShanghaiTech 上 70.26% 的较低分数反映了骑行、车辆入侵等未被 4 条 prompt 覆盖的异常类型。
- **仅针对人员级异常**：不含人的异常（如火灾、遗留物）无法检测，需要额外模块补充。
- **检测器域泛化受限**：YOLO 在室内场景 fine-tune，对完全不同的室外/光照环境泛化存疑。
- **未来方向**：可学习 prompt 扩展（借鉴 VadCLIP）、低光照鲁棒预处理、面向资源受限边缘设备的一体化端到端模型。

## 研究启发与可借鉴点
1. **"精简替代复杂"的设计哲学**：用 CLIP 零样本图文匹配直接替代多组件密度估计，说明在某些场景中语义匹配可等价甚至超越手工特征+统计建模，为其他视频理解任务提供了架构简化的思路。
2. **固定 prompt 零训练的部署策略**：适用于快速落地到新场景（只需改写 prompt），可在生产环境初期以零样本方式快速验证可行性，再逐步扩展。
3. **Yolo v11n-pose 的"单 pass 检测+姿态"价值**：将检测与姿态估计统一为一个轻量模型，避免了多模型串行调用的开销，这一设计模式可迁移到其他需要人员级分析的管道（如人群计数、行为识别）。
4. **实时部署的开销分析视角**：论文详细拆解了各组件 FPS 与端到端 FPS 的差异来源（GPU-CPU 传输、动态 batch、渲染等），这种分解方法可作为后续部署工作的参考模板。

## 关键术语表
- **Video Anomaly Detection（视频异常检测）**：在视频流中自动识别偏离正常模式的事件（如跌倒、打斗），无需事先标注所有异常类型。
- **CLIP（Contrastive Language-Image Pre-training）**：通过对比学习将图像和文本映射到共享嵌入空间，支持零样本图像分类。
- **YOLO v11n-pose**：Ultralytics YOLO v11 的轻量姿态估计变体（nano 版），单次前向传播同时输出人体边界框和 17 个关键点。
- **Cosine Similarity Scoring**：计算图像嵌入与文本嵌入的余弦相似度作为异常置信度分数，替代传统的密度估计方法。
- **AUROC（Area Under the Receiver Operating Characteristic Curve）**：衡量异常/正常帧可分离性的指标，值越高表示检测性能越好。
- **Zero-shot Classification**：无需异常标签训练，直接通过自然语言 prompt 对未见过的异常类别进行分类。
- **GMM/kNN Density Estimation**：传统多特征方法中对 pose/flow/CLIP 特征拼接后进行概率密度建模的评分机制。
- **COCO Keypoint Topology**：标准人体骨骼 17 点拓扑结构（鼻、眼、耳、肩、肘、腕、髋、膝、踝）。

## 可复现要素
- **数据集**：CUHK Avenue [13]（公开）、ShanghaiTech Campus [14]（公开）、CU Indoor Anomaly（自建，论文未声明公开）
- **代码**：论文未提及代码开源状态
- **权重**：YOLO v11n-pose（COCO 预训练权重 + CU Indoor 微调权重）；CLIP ViT-B/32（预训练权重，未微调）
- **关键超参**：padding $\delta = 10$ 像素；CLIP 输入尺寸 $224 \times 224$；YOLO 输入 640×640，batch=16，lr=0.01，500 轮；时序平滑高斯核 $\sigma=5$，半窗 $w=15$；决策阈值 $\theta^* = 0.7$
- **硬件**：NVIDIA Titan XP GPU
