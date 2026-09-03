---
title: "Video-Based-Palm-Vein-Authentication-under-Challenging-Condi"
source: https://arxiv.org/pdf/2609.02776v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 18:13:18"
---

# 论文速读：Video-Based-Palm-Vein-Authentication-under-Challenging-Condi

## 一句话总结
论文提出了首个公开的视频流掌静脉数据集 CUP，揭示现实表面退化会导致主流识别器精度骤降（脏手条件下 EER 约 4 倍恶化），并设计了一个零参数测试时双视图匹配器（全局嵌入 + 显著性引导的区域级最优传输 + 时序共识），在多种恶劣条件下恢复识别性能且可泛化至其他冻结骨干与公开图像数据集。

## 研究问题与动机
- **核心问题**：真实部署中手掌表面噪声（汗液、污渍、温湿度变化）会严重破坏近红外掌静脉信号，现有方法在非理想条件与动态序列维度的性能瓶颈尚不明确。
- **现有数据集局限**：CASIA-MS、VERA、SCUT 等公开掌静脉数据集均为静态清晰图像，缺乏自然表面劣化与视频序列维度，无法反映日常使用应力。
- **标准训练手段失效**：干净数据微调、混合表面训练与帧聚合只能部分缓解误差；在脏手条件下所有基线仍远高于干净条件底线，全局池化 embedding 被局部污染拖垮。
- **区域感知方法未触及匹配层**：现有分区方法（如 RSNet）仍在比较前将特征图池化为单一向量，区域信息在匹配时被淹没；测试时动态路由可靠区域的研究仍匮乏。

## 核心贡献（创新点）
1. **提出首个公开视频掌静脉数据集 CUP**：提供 5,049 条双秒近红外视频、四种受控表面条件（clean/warm/wet/dirty）与配对生理/人口统计元数据，填补动态捕捉与自然退化基准的空白。不同于此前仅含静态清晰图像的集合，CUP 将真实部署四大应力源纳入统一协议。
2. **设计零参数测试时双视图匹配器**：编码阶段同时提取全局嵌入与 7×7 区域特征网格，利用熵正则最优传输在匹配时绕过受损区域，并与全局 cosine 分数经 per-probe 标准化后融合。与现有方法在训练前池化特征的做法不同，该设计将区域对比延后至测试时，不引入任何可训练参数即可动态路由匹配质量。
3. **引入轻量双共识训练策略**：对聚合帧与单帧分别施加共享 ArcFace 分类器，迫使每个单帧 Embedding 自身具备身份判别力，仅用少量均匀采样帧即可抵消瞬态腐蚀。与端到端 3D 视频模型依赖高帧率或复杂时空卷积不同，该方法以极低采集与推理成本换取时序稳定性。
4. **开展数据集驱动的公平性初步审计**：基于 CUP 元数据评估 10 项人口与生理特征，揭示温暖条件下体水含量与性别差异对识别精度的显著影响。不同于以往仅依赖 demographic 标签的公平性报告，本文首次将可测量生理参数（体水、BMI、SpO2 等）与表面退化耦合，为生理导向的公平性研究提供信号。

## 方法详解
- **双视图编码器**：共享冻结骨干 $f$ 逐帧输出 $P \times P$ 特征图；区域视图保留网格形式 $\{\mathbf{r}_j^{(t)}\}$，全局视图经 GAP 得 $\mathbf{f}^{(t)}$，再通过线性投影头 $\phi$ 与 $\ell_2$ 归一化。对 $K$ 帧时序平均并归一化得到聚合描述子 $(\mathbf{u}_p, \{\hat{\mathbf{p}}_j\})$ 与 $(\mathbf{u}_G, \{\hat{\mathbf{g}}_i\})$。
- **双共识损失**：$\mathcal{L} = \mathcal{L}_{\text{arc}}(\phi(\bar{\mathbf{f}}), y) + \alpha \cdot \frac{1}{K}\sum_t \mathcal{L}_{\text{arc}}(\phi(\mathbf{f}^{(t)}), y)$，$\alpha=2.0$。区域视图不单独加监督，因可信区域交由匹配时的最优传输决定，避免冗余。
- **区域路径：熵正则最优传输**：代价矩阵 $\mathbf{M}_{ij}(G) = 1 - \hat{\mathbf{g}}_i(G)^\top \hat{\mathbf{p}}_j$。通过 Sinkhorn 迭代（$L=50$, $\varepsilon=0.1$）求解软分配 $\Gamma^*(G)$，区域得分 $S_{\text{reg}}(G) = \langle \Gamma^*, \mathbf{1} - \mathbf{M} \rangle$，使匹配质量流向存活区域。
- **画廊显著性边缘分布**：$d_i(G) = \frac{1}{|\mathcal{G}_{\text{gal}}|-1}\sum_{G'\neq G}(1-\hat{\mathbf{g}}_i(G)^\top \hat{\mathbf{g}}_i(G'))$ 度量区域跨身份的判别性；边缘权重 $a_i(G) \propto (\max(0,d_i(G))+\delta)^\gamma$（$\gamma=2, \delta=0.05$）引导 OT 优先对齐高信息量区域。
- **互补融合**：全局路径 $S_{\text{glob}}(G) = \mathbf{u}_p^\top \mathbf{u}_G$。两路分数分别做 per-probe Z-score 标准化后加权求和 $S(G) = z[S_{\text{reg}}] + \lambda z[S_{\text{glob}}]$（$\lambda=0.5$），全程无额外训练参数，仅在匹配时计算。

## 实验与结果
- **数据集与基线**：CUP（210 palms / 5,049 clips）及四个公开图像集 SCUT、TJ-PV、CASIA、VERA；21 个基线覆盖掌静脉专用（RSNet、AMPVNet、MDNet
