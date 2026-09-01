---
title: "SegWave-Wavelet-Driven-Segmentation-of-Tampered-Regions"
source: https://arxiv.org/pdf/2608.30714v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 22:08:18"
field: "图像篡改检测与定位"
keywords: ["Image Forensics", "Tampering Detection", "Discrete Wavelet Transform", "Adaptive Sub-band Attention", "Multi-source Segmentation", "Weak Supervision", "Segment Anything"]
innovations: ["提出 DWT + ASA 的混合 Transformer 框架，利用局部化多尺度频率线索实现压缩无关的篡改定位", "设计自适应子带注意力模块，动态加权 LH/HL/HH 高频子带以提升边界定位与多源分区能力", "在弱二值掩码监督下实现点提示引导的多源分割，无需精细标注即可分区多个伪造源"]
benchmarks: ["CocoGlide", "Columbia", "RealisticTampering", "SafireMS-Expert"]
---

# 论文速读：SegWave-Wavelet-Driven-Segmentation-of-Tampered-Regions

## 一句话总结
提出 SegWave，一种基于离散小波变换（DWT）与自适应子带注意力（ASA）的混合 Transformer 框架，通过融合空间特征与局部化多尺度频率线索，在仅依赖弱二值掩码监督的条件下，实现图像篡改区域的精细分割与多源分区定位，并在多个基准上超越现有 SOTA 方法。

## 研究问题与动机
- **现实需求**：随着生成式 AI 的普及，图像篡改（拼接、复制移动、GAN/扩散模型编辑）日益逼真，新闻、司法、安全等领域需要精确知道"哪里被篡改"以及"由几个不同来源合成"，而不仅是"是否篡改"的二元判定。
- **现有方法局限一（频率基线的压缩依赖）**：当前主流取证方法（如 CAT-Net、SAFIRE）依赖 FFT 或 DCT 捕获的全局频率压缩伪影，在 PNG 等无损格式上信号衰减甚至消失。
- **现有方法局限二（缺乏细粒度源感知分割）**：多数检测器仅输出单一真/假结论，无法分离图像中多个共存的伪造源区域，难以支撑溯源与归因需求。
- **视觉-语言方法的潜在缺陷**：MLLM 驱动的取证方法可能因语义优先而降低定位精度（近年分析表明语义先验反而可能损害真实性判断），凸显低层伪影敏感线索的价值。

## 核心贡献（创新点）
1. **提出 SegWave 混合 Transformer 框架**：将原始图像与 DWT 增强的频率特征分别进行 patch embedding 后融合，送入 Vision Transformer  backbone；与以往依赖 JPEG 压缩痕迹的方法本质区别在于，利用多尺度、空间局部化的频率表示，在无损格式（PNG）上仍能有效检测。
2. **设计自适应子带注意力模块（ASA）**：动态计算 LH、HL、HH 三个高频子带的注意力权重并加权求和，再经逆 DWT 重建增强图像；与均匀加权或全局 DCT 方案本质不同，ASA 能按方向与源分别强化最具判别力的频带，提升边界定位精度。
3. **在弱二值掩码监督下实现点提示引导的多源分割**：从 GT 掩码中采样正/负点提示嵌入 prompt encoder，引导 transformer mask decoder 进行多源分区；与 SAFIRE 等需更多标注的方法本质不同，本文仅在弱监督下完成细粒度、源感知的分割输出。

## 方法详解
**整体架构流程（Algorithm 1）：**
- **输入**：训练图像对 $\{ (I_k, M_k) \}_{k=1}^{K}$，$M_k$ 为二值篡改掩码。
- **步骤 1 — DWT 分解**：对输入 $I_k$ 执行单层 Haar 小波变换，得到近似子带 LL 与三个细节子带 LH（水平）、HL（垂直）、HH（对角）。丢弃 LL，保留三个高频子带。
- **步骤 2 — ASA 加权**：计算 LH、HL、HH 的动态注意力分数，加权求和以突出最有判别力的高频信息。
- **步骤 3 — 逆 DWT 重建**：将加权后的高频分量经逆 DWT 重建为增强图像 $\hat{I}_k$。
- **步骤 4 — Patch Embedding**：对原始图像 $I_k$ 和增强图像 $\hat{I}_k$ 分别做 patch embedding，得到 $E_k$ 和 $\hat{E}_k$。
- **步骤 5 — 特征融合**：对 $\hat{E}_k$ 做线性变换后与 $E_k$ 相加，形成联合表征，自适应注入 Vision Transformer 各层。
- **步骤 6 — 点提示编码**：从 GT 掩码中采样一个篡改区点（label=1）和一个未篡改区点（label=0），经 prompt encoder 嵌入并与图像 embedding 兼容。
- **步骤 7–8 — Transformer 编解码**：Transformer backbone 处理融合特征，mask decoder 融合图像与提示 embedding，输出预测掩码 $\hat{Y}_k$ 及置信度 $\hat{c}_k$。
- **损失函数**：
  - 加权二值交叉熵：$\mathcal{L}_{wBCE} = -w_+ y \log(\hat{y}) - w_- (1-y) \log(1-\hat{y})$
  - MSE 置信度损失：$\mathcal{L}_{MSE} = \frac{1}{N}\sum (\hat{c}_i - c_i)^2$
  - 总损失：$\mathcal{L}_{total} = \mathcal{L}_{wBCE} + \lambda \cdot \mathcal{L}_{MSE}$，其中 $\lambda = 0.1$。
- **推理时多提示策略**：在 $16\times16$ 网格上均匀采样 256 个提示点，经 mask decoder 得到多个预测掩码，按区域均值取代表性特征后进行聚类（K 组对应 K 个源），每簇取最 confident 掩码，最终 soft-max 或多源合并输出。

**SegWave$_{DCT}$ 消融变体**：将 DWT 替换为全局 DCT，因无方向子带故移除 ASA 模块，其余结构不变，用于隔离变换方式的影响。

## 实验与结果
- **训练数据**：CASIA 2.0 与 FantasticReality 各 2,000 张。
- **测试基准**：RealisticTampering（220 张）、CocoGlide（512 张）、Columbia（180 张）、SafireMS-Expert（238 张，含 2/3/4 源分区标注）。
- **评估指标**：$F1_{fixed}$ / $F1_{best}$（阈值 0.5 vs. 图像最优阈值）；多源分区用 p-mIoU 与 p-ARI。

**主要结果（Table 1，二元源定位）：**
| 数据集 | 最强基线 | SegWave | 提升幅度 |
|---|---|---|---|
| CocoGlide | SAFIRE 0.63 / 0.76 | **0.69 / 0.79** | +0.06 / +0.03 |
| Columbia | SAFIRE 0.98 / 0.99 | **0.99 / 0.99** | +0.01 / 持平 |
| RealisticTampering | TruFor 0.43 / 0.53 | 0.36 / **0.51** | −0.07 / −0.02 |

- **多源分区（Table 2，SafireMS-Expert）：**
  - 2 源：SegWave p-mIoU = **0.821**，p-ARI = **0.710**，分别超越 SAFIRE（0.763 / 0.660）+0.058 / +0.050。
  - 3 源：p-mIoU = **0.648**，p-ARI = **0.641**，优于 SAFIRE（0.618 / 0.573）。
  - 4 源：p-mIoU = 0.422，p-ARI = 0.577， margin 缩小，反映多源难度上升。

**消融结论**：ASA 在二元定位上增益 modest（CocoGlide 0.67→0.69），但在多源分区上作用显著（2 源 p-mIoU 0.695→0.821）；SegWave$_{DCT}$ 在 RealisticTampering 上略优（0.40 vs 0.36），归因于该数据集可能含 JPEG 压缩痕迹，而 DWT+ASA 更擅长捕捉空间局部边界不一致性。

## 相关工作脉络
- **CAT-Net v2 [26]**：基于 JPEG 压缩伪影学习的篡改定位方法，依赖 DCT 块级分析；SegWave 与之差异在于利用 DWT 的局部化多尺度频率表示，不依赖压缩痕迹。
- **SAFIRE [25]**：点提示引导的分割型取证方法，但使用全局频率线索；SegWave 在此基础上引入方向感知的小波子带与 ASA，实现更精细的边界定位和多源分区。
- **TruFor [15]**：多线索融合（像素/频域/PRNU）的检测器；SegWave 侧重纯学习式的空间-频率融合架构，在弱监督下达到 comparable 或更强性能。
- **FFT/DCT 频域取证传统**（如 [21,25,26,27]）：关注全局频域压缩伪影；SegWave 转向局部化、多尺度的小波分解，弥补无损格式上的能力缺口。
- **Vision-Language 取证方法**（[40,41]）：依赖大型多模态模型和文本标注；SegWave 采用低层伪影敏感特征，不受语义先验误导，且仅需弱二值掩码监督。
- **传统 DWT 取证**（如 [14,18]）：使用手工特征；SegWave 将 DWT 与端到端 Transformer 学习框架结合，实现端到端优化。

## 局限性与未来方向
- **多源数量增加时性能收敛**：4 源设置下各方法差距缩小，反映当前架构在处理高度重叠、多源复杂场景时仍有提升空间。
- **数据集分布敏感性**：SegWave 在 RealisticTampering 上不及 TruFor，因该数据集可能含 JPEG 压缩痕迹，DWT 局部化优势未能充分发挥，说明模型对数据分布仍有一定依赖。
- **单层 Haar DWT**：仅使用单层分解，可能无法捕获更精细的多尺度特征；未来可探索深层或小波包分解。
- **论文自述的未来方向**：将 DCT（压缩感知优势）与 DWT（局部边界感知优势）统一至单一模型，实现压缩无关与边界敏感的联合检测。

## 研究启发与可借鉴点
- **小波变换 + 注意力融合的可迁移设计**：DWT 分解后对方向子带做自适应加权（ASA），可有效捕获边缘/纹理异常，该思路可迁移至深度伪造检测、视频篡改定位等任务。
- **弱监督下的点提示分割范式**：仅用二值掩码监督，通过正/负点采样引导多源分区，降低了标注成本，适用于缺乏精细标注的取证场景。
- **空间-频率双路径融合策略**：原始图像与频域增强图像各自 embedding 后融合注入 Transformer 各层，兼顾语义上下文与底层伪影，是通用的多模态特征融合范式。
- **消融设计的启发性对比**：通过 SegWave$_{DCT}$ 与 SegWave$_{w/o\ ASA}$ 两组消融，清晰分离了"变换类型"与"子带加权"的贡献，实验设计值得借鉴。
- **推理时多提示聚类合并策略**：网格采样 + 特征聚类 + 每簇取最 confident 预测的融合方案，可有效处理多源场景，类似策略可扩展至其他分割任务。

## 关键术语表
- **Discrete Wavelet Transform (DWT)**：将图像分解为多尺度、多方向的低频近似与高频细节子带的数学工具，支持空间局部化频率分析。
- **Adaptive Sub-band Attention (ASA)**：动态计算各高频小波子带（LH/HL/HH）的注意力权重并加权融合，突出判别性频带、抑制结构噪声的模块。
- **Point Prompt**：从 GT 掩码中采样的单点坐标及其标签（篡改/非篡改），作为引导分割注意力的条件输入。
- **p-mIoU / p-ARI**：多源分区评估指标，分别衡量预测掩码与真值掩码的 IoU（允许标签排列置换）和聚类相似度。
- **SegWave$_{DCT}$**：将 DWT 替换为全局 DCT 的消融变体，无方向子带故不含 ASA，用于隔离变换方式的影响。
- **$F1_{fixed}$ / $F1_{best}$**：前者使用固定阈值 0.5 计算 F1，后者逐图像优化阈值取最优 F1，分别反映一致性和上限性能。
- **SafireMS-Expert**：含 2/3/4 个不同源区域的多源篡改图像数据集，用于评估模型的源分区能力。
- **RealisticTampering**：包含真实世界篡改操作的基准数据集，存储为 TIFF 但可能含前置 JPEG 压缩痕迹。

## 可复现要素
- **训练数据集**：CASIA 2.0、FantasticReality（各 2,000 张），均公开可用。
- **测试数据集**：CocoGlide、Columbia、RealisticTampering、SafireMS-Expert，均公开可用。
- **代码/权重**：论文未明确声明开源，需进一步确认。
- **关键超参**：$\lambda = 0.1$（MSE 损失权重）；推理时 256 个提示点（$16\times16$ 网格）；batch size 128（推理）；Haar 单层小波变换。
