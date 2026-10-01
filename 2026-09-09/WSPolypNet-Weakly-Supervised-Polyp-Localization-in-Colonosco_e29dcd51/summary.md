---
title: "WSPolypNet-Weakly-Supervised-Polyp-Localization-in-Colonosco"
source: https://arxiv.org/pdf/2609.08182v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 04:04:51"
field: "弱监督医学视频目标定位"
keywords: ["weakly supervised localization", "polyp detection", "colonoscopy video", "class activation map", "MedSAM2", "3D CNN", "multi-view strategy"]
innovations: ["仅用视频级标签的弱监督息肉定位框架", "多视图CAM融合显著提升小息肉定位", "CAM点提示驱动MedSAM2细化的定位流水线"]
benchmarks: ["LDPolypVideo", "CorLoc@0.3/0.5/0.7"]
---

# 论文速读：WSPolypNet-Weakly-Supervised-Polyp-Localization-in-Colonosco

## 一句话总结
论文提出 **WSPolypNet**，一种仅需**视频级标签**的弱监督息肉定位框架，通过 3D CNN 生成类激活图（CAM）并结合**多视图策略**与 **MedSAM2** 进行边界细化，显著降低结肠镜视频中密集标注成本，对小息肉（CorLoc@0.5 从 16.01% 提升至 30.97%）尤为有效。

## 研究问题与动机
1. **密集空间标注成本高**：传统监督方法依赖每帧的 bounding box 或像素级 mask，标注耗时且受观察者间差异影响。
2. **视频级监督利用不足**：现有弱监督视频目标定位（WSVOL）方法在结肠镜息肉空间定位上尚未充分探索。
3. **CAM 定位粗糙**：传统 CAM 仅关注最具判别性的区域，对**小息肉**定位不完整，且易受肠壁复杂结构干扰激活无关区域。
4. **时序信息缺失**：多数方法将帧视为独立图像，未显式建模连续帧间的时序相关性，导致预测在视频流中不稳定。

## 核心贡献（创新点）
1. **弱监督框架**：仅用视频级二元标签同时预测息肉存在性与单帧空间位置，无需帧级 bounding-box/pixel mask 标注。
2. **3D CNN + CAM 时空表征**：利用 X3D 骨干学习连续帧的时空特征，通过 GAP + 分类权重加权生成 CAM，作为初始定位线索。
3. **多视图（Multi-view）增强策略**：采集 5 个视角（全图 + 4 个角裁切 144×144 → 放大至 224×224），显著提升小息肉 CorLoc@0.5（16.01% → 30.97%）。
4. **MedSAM2 边界细化流水线**：将 CAM 峰值点作为点提示输入冻结的 MedSAM2，双向传播生成分割 track，按 track score 规则选取最终 mask。

## 方法详解
### 整体流程
输入视频 clip → 3D CNN 分类 + 生成 CAM → 多视图 CAM 融合 → 提取候选点 → MedSAM2 提示细化 → 多 track 评估选优 → 输出最终 mask。

### A. ROI 预处理
- 用**时序亮度投票**估计有效内窥镜视野（FOV）凸包，裁剪并掩码外部区域。
- 训练时做**空间随机化**：ROI 随机缩放到 90–100%，随机放置，避免模型依赖背景/边界伪迹。

### B. 时空特征学习与 CAM 生成
- 输入 $\mathbf{X} \in \mathbb{R}^{C \times T \times H \times W}$，经 3D CNN 得特征张量 $\mathbf{F} \in \mathbb{R}^{K \times T' \times H' \times W'}$。
- GAP + BCEWithLogitsLoss 做视频级二分类；推理时用 sigmoid 得息肉存在概率。
- CAM 计算：
$$
M(t, u, v) = \sum_{k=1}^{K} w_k F_k(t, u, v)
$$
其中 $w_k$ 为分类权重。

### C. 多视图 CAM 融合
- 5 个视角：全图 224×224 + 4 个角区域 144×144（覆盖每边 64%，相邻重叠 64 px）。
- 每个视角得分 map：
$$
L_t^{(q)}(u,v) = S_t^{(q)}(u,v) \cdot \sigma\big(A_t^{(q)}(u,v)\big)
$$
其中 $S$ 为空间 softmax 权重，$\sigma$ 为 sigmoid。
- 五视角**逐像素 max 融合**得到 $L_t^{\mathrm{fused}}$，取峰值帧 $t^* = \arg\max_t \max_{u,v} L_t^{\mathrm{fused}}$。
- 对 $t^*$ 的 fused map 做 min-max 归一化 + Gaussian 平滑（$\sigma=4$ px），再进行 NMS（半径 16 px）提取最多 5 个候选点。

### D. MedSAM2 细化
- 每个候选点作为**正点提示**输入冻结的 MedSAM2，生成 seed mask 并双向传播到相邻帧，形成一条 candidate track。
- Track 评分：对每帧计算 mask 外框内 fused-CAM 均值，再跨视频平均。
- 默认选最高分 track；次高分 track 仅在 score 超过当前第一 ≥20% 时替换。
- 退化处理：若某帧 propagated mask 为空 或 置信度 < $\tau_{\mathrm{conf}}=0.755$（前景像素 sigmoid 均值），则回退到该帧自身 CAM 峰值点的独立预测。

## 实验与结果
### 数据集
- **LDPolypVideo**：871 条正样本 clip（来自 100 条源视频，每 clip ≤30 帧，1 FPS）、615 条负样本 clip（来自 60 条无息肉视频），共 1,486 clips。

### 骨干网络对比（Table I）
| Backbone | Recall | CorLoc@0.3 | CorLoc@0.5 | CorLoc@0.7 |
|---|---|---|---|---|
| Slow R50 | 80.22% | 18.94% | 7.87% | 1.05% |
| **X3D** | 74.18% | 21.03% | 9.32% | 1.63% |
| SlowFast R50 | 69.23% | 13.57% | 3.63% | 0.59% |
| R3D-18 | 91.21% | 16.75% | 6.53% | 0.82% |
| R(2+1)D-18 | 94.51% | 12.72% | 2.78% | 0.11% |

→ 选 X3D 作为主干（定位性能最优）。

### 主结果（Table II）
| Method | Recall | CorLoc@0.3 | CorLoc@0.5 | CorLoc@0.7 |
|---|---|---|---|---|
| Single-view | 74.18% | 36.87% | 33.72% | 27.94% |
| **WSPolypNet** | **94.51%** | **47.80%** | **43.68%** | **35.01%** |

### 多视图 vs 单视图按息肉大小（Table III）
- **小息肉**（面积比 ≤5%）：CorLoc@0.5 从 16.01% → **30.97%**（+14.96pp）
- **大息肉**（面积比 >5%）：CorLoc@0.5 从 71.84% → 65.75%（↓6.09pp，存在 trade-off）

### 结论
WSPolypNet 以视频级标签实现 Recall 94.51%、CorLoc@0.5 = 43.68%，小息肉定位提升尤为突出。

## 相关工作脉络
1. **CAM / WSOL 系**：Zhou et al. (CVPR 2016) 提出 CAM；Choe & Shim (CVPR 2019) 提出 attention dropout；Zhang et al. (CVPR 2018) 对抗互补学习 —— 本文与之区别在于首次将 CAM 引入结肠镜**视频级**弱监督定位并接 MedSAM2 细化。
2. **视频级弱监督定位**：Belharbi et al. (WACV 2023) 的 TCAM、Liao et al. (WACV 2025) 的外科视频定位 —— 本文聚焦内窥镜息肉且用 3D CNN + 多视图替代 Transformer。
3. **Promptable 分割**：Kirillov et al. (ICCV 2023) SAM、Ravi et al. (arXiv 2024) SAM 2、Ma et al. (arXiv 2025) MedSAM2 —— 本文把 CAM 点提示作为 MedSAM2 的入口，属"弱监督定位 + 可提示分割"的**互补组合**范式。
4. **3D CNN 骨干**：Slow R50、X3D、R(2+1)D 等 Kinetics 预训练模型 —— 本文系统对比后选用 X3D 作 CAM 提取主干。
5. **Grad-CAM 系**：Selvaraju et al. (ICCV 2017) 基于梯度的解释方法 —— 本文采用原始 CAM 而非 Grad-CAM。

## 局限性与未来方向
- **单息肉假设**：当前仅选取一条高置信度 track，无法处理同帧/同视频多息肉共存。
- **大息肉性能 trade-off**：多视图虽提升小息肉，却使大息肉 CorLoc@0.5 下降约 6pp。
- **视频长度受限**：当前 clip ≤30 帧，对仅在短暂窗口出现的息肉覆盖不足。
- **标注辅助定位而非替代**：作者明确当前更适合"标注辅助"，距离全自动标注仍有差距。

## 研究启发与可借鉴点
1. **弱监督 + 冻结大模型提示的范式**：先用轻量 3D CNN 生成粗定位，再喂给冻结的 MedSAM2 做细化，大幅降低对密集标注的依赖，可迁移到其它内窥镜病变/器官定位任务。
2. **多视图角裁剪策略**：对小目标特别有效（通过放大边界附近区域提升响应），可用于任何 CAM/GAP 类定位模型的改进。
3. **Track 选择 + 退化回退机制**：用 CAM 均值评分选 track，并在 confidence < 阈值时回退到帧级独立预测，提升鲁棒性。
4. **ROI 空间随机化增强**：去除 FOV 边缘偏置，降低模型利用背景伪迹的风险，可推广到多源内镜数据融合场景。

## 关键术语表
- **CorLoc**：Correct Localization，预测 bbox 与 GT bbox 的 IoU 超过阈值即计为正确，是弱监督定位常用指标。
- **CAM (Class Activation Map)**：通过分类权重加权最后卷积层特征图得到的空间注意力图。
- **MedSAM2**：面向医学图像/视频的 Segment Anything 2 适配版本，支持点提示驱动分割。
- **WSVOL (Weakly Supervised Video Object Localization)**：仅用视频级标签在视频中对目标做空间定位的任务设定。
- **X3D**：扩展 3D 卷积网络，在视频动作识别上取得 SOTA，本文用作 3D CNN 主干。
- **GAP (Global Average Pooling)**：对特征图空间/时间维度平均池化，用于生成 CAM 的全局表征。
- **BCEWithLogitsLoss**：带 logit 的二元交叉熵损失，常用于二分类任务的端到端训练。
- **NMS (Non-Maximum Suppression)**：此处用于在 CAM map 上贪婪提取相距 ≥16 px 的多个候选点。

## 可复现要素
- **数据集**：LDPolypVideo（论文说明正样本来自该数据集，但未说明数据集本身是否公开；训练集构造细节完整）。
- **代码/权重**：论文未提供开源链接；3D CNN 骨干使用 **Kinetics-400** 预训练权重（公开可获取）。
- **关键超参**：
  - 优化器：AdamW，lr = 1×10⁻⁴，weight decay = 0.01
  - 前 5 epoch 冻结 backbone，之后解冻端到端微调；总 50 epoch，余弦学习率调度
  - 多视图：5 视角（全图 224×224 + 4 个角 144×144 → resize 224×224）
  - NMS 半径：16 px；Gaussian 平滑 σ = 4 px
  - 置信度阈值 τ_conf = 0.755
  - Track 替换阈值：score 超过第一 ≥20%

---
