---
title: "Uncertainty-Guided-Adverse-Weather-Restoration-via-Gated-Tra"
source: https://arxiv.org/pdf/2609.02434v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 10:18:16"
field: "图像恢复与去噪去模糊"
keywords: ["adverse weather restoration", "all-in-one image restoration", "gated transformer", "uncertainty-aware refinement", "balanced multi-scale skip connection", "linear attention"]
innovations: ["GDTB 通过 query-conditioned gating 实现退化感知的自适应全局聚合", "BMSC 以预测-校正多步积分渐进融合多尺度跳连以降低浅层噪声", "URH 与 BAE-Loss 联合训练像素级均值/方差以提高严重退化鲁棒性"]
benchmarks: ["Snow100K-S", "Snow100K-L", "Outdoor-Rain", "Raindrop"]
---

# 论文速读：Uncertainty-Guided-Adverse-Weather-Restoration-via-Gated-Transformer

## 一句话总结
本文提出 UAR-Net，一种面向统一恶劣天气恢复的 gated transformer 框架，通过选择性门控注意力、渐进式多尺度跳连与不确定性感知细化头，解决了现有方法在跨区域/跨天气场景下的特征聚合不均与确定性预测不可靠问题，在多个基准上达到 SOTA。

## 研究问题与动机
- 现有 all-in-one 方法对全局上下文采用统一的 weather-agnostic 聚合方式，难以适应不同天气对长程依赖的不同需求。
- 浅层特征在不同天气退化下可靠性差异显著，简单拼接或相加会放大退化噪声，阻碍有效跨尺度交互。
- 多数 unified 模型采用确定性目标，对严重/模糊退化区域的预测容易过度自信且不稳定。
- 单任务专用模型可扩展性差，难以覆盖雨、雪、雾、雨滴等多种退化并保持一致性能。

## 核心贡献（创新点）
- 提出 GDTB，将 query-conditioned gating 引入线性注意力，实现对全局上下文的自适应、退化敏感的聚合；与 Histoformer 等均匀聚合方法相比，强调“按区域内容选择性利用全局信息”。
- 提出 BMSC，以预测-校正（P–C）多步积分方式渐进融合多级编码器特征，替代一次性拼接/相加，显著降低浅层强噪声传播并提升解码引导的稳定性。
- 提出 URH 与 BAE-Loss，并行预测像素级均值与方差，并以能量评分 + 亮度自适应回归联合训练，缓解严重退化下的过度自信与亮度偏移。
- 在 Snow100K-S/L、Outdoor-Rain、Raindrop 等多个基准上获得一致 SOTA，且 perceptual 指标同步提升，验证方法泛化与稳定性。

## 方法详解
- 整体架构为 U-Net 式 encoder-decoder，由多层 GDTB 堆叠；额外引入 supplementary skip（平均池化 + 1×1 + 3×3 depthwise）以保留低频先验。
- GDTB 由 Selective Gated Attention（SGA）与 Dual-scale Gated Feed-Forward（DGFF）组成：SGA 基于 sinusoidal 重加权增强局部感知的线性注意力，并将 query 分支拆分为注意力与 gating 两路，经 sigmoid 门控调制输出；DGFF 用两个并行 depthwise 卷积（3×3 与 5×5）捕获不同感受野并自适应融合。
- BMSC 将多尺度编码器特征从深到浅进行渐进融合，采用 linear multistep（AB/AM）风格的 predictor–corrector 更新，并在输出前用 vHeat（基于 2D 热传导算子 HCO 的频率域扩散）进行轻量全局细化，得到干净稳定的 skip 表示。
- URH 为四阶段 compact U-Net，并行 mean/variance 头输出恢复图像与像素不确定性图；不确定性不追求严格校准，而是作为反映退化难度的任务信号引导细化。
- 损失函数包括：
  - BAE-Loss = (1 − w_prob(t))·L_GT + w_prob(t)·L_ES，其中 L_ES 为像素 l1 距离的能量评分，L_GT 为含亮度自适应权重 W（基于 Bhattacharyya 距离）的回归项；w_prob(t) 训练过程中退火上升。
  - L_cor 基于 Pearson 相关系数，强制恢复图像与 GT 的全局结构与强度一致性。
  - 总损失 L = L_BAE + L_cor。

## 实验与结果
- 训练数据合并 Snow100K（9K）、Raindrop（1,069）、Outdoor-Rain（9K），在 H100 ×4 上从头训练 300k 步，采用五阶段 progressive learning（patch 128→360，batch 8→1）。
- 主要基线：Histoformer、MODEM、HOGformer、PromptIR、DiffUIR-L、Restormer、WGWSNet、WeatherDiff 等。
- 关键数值（PSNR/SSIM）：
  - Snow100K-S：Ours 38.34 / 0.9697；Snow100K-L：32.77 / 0.9324；Outdoor-Rain：33.40 / 0.9491；Raindrop：33.32 / 0.9487；Avg：34.46 / 0.9500。
  - 相对 Histoformer 平均提升约 +0.78 dB PSNR；相对 MODEM 平均提升约 +0.28 dB PSNR。
  - 任务细分：Snow-S/L、Outdoor-Rain、Raindrop 分别较 MODEM 提升 +0.26~+0.31 dB。
- 感知指标（Table IV）：LPIPS 最低、Q-Align 与 MUSIQ 最高，跨数据集均领先。
- Ablation：组件逐次加入带来稳定增益（GDTB → BMSC → BAE → URH）；gating 对性能贡献大于 sinusoidal reweighting；LMF + vHeat 的组合优于 Avg/CosFormer/Nonlocal；平衡特征尺寸选 H/4×W/4；BAE 退火步越大越好（实验设置 300k）。
- 复杂度：UAR-Net 35.44 GMacs，高于 Histoformer（23.27）与 MODEM（29.08），但在同预算范围内取得更高精度。

## 相关工作脉络
- Histoformer（[19]）：基于 histogram transformer 的 AiO 框架，本文以其为 baseline，改进其均匀全局聚合与浅层跳连。
- MODEM（[21]）： Morton-order 退化估计的统一模型；本文在多个数据集上超越其 PSNR/SSIM。
- HOGformer（[24]）：梯度/退化条件注意力；本文在 Snow 与 Rain 任务上继续领先。
- Restormer / NAFNet：高分辨率恢复经典 backbone；作为通用恢复方法的参照。
- Desnownet / AttentiveGAN：单任务 snow/raindrop 方法；本文在统一设定下仍匹敌或超越部分专项方法。
- vHeat（[39]）、CosFormer（[36]）：分别代表热传导注意力与余弦注意力；本文将其用于 BMSC 细化阶段并进行对比 ablation。

## 局限性与未来方向
- 计算开销偏高（35.44 GMacs），在实时或低算力部署场景受限。
- 数据集以合成为主（Snow100K、Outdoor-Rain）+ 小规模真实 Raindrop，真实复杂场景泛化仍需进一步验证。
- 不确定性以任务信号为主导，未做严格概率校准评估。
- 训练采用五阶段 progressive 策略，调参与训练时长较长；对超参敏感。
- 未来可探索更轻量的 global context 模块、自校准不确定性度量、更多真实气象域与跨下游任务联合评测。

## 研究启发与可借鉴点
- 将 predictor–corrector 思想引入 skip fusion，可有效抑制浅层退化噪声并实现更平滑的跨尺度传播。
- 在统一恢复中引入像素级不确定度估计，并配合非局域能量评分，对严重退化区域更有鲁棒性；该范式可迁移至其他密集预测任务。
- sinusoidal 位置重加权与门控并用的 attention 设计，在保持线性复杂度的同时提升内容自适应能力，适合高分辨率恢复。
- 五阶段 progressive 训练与 annealing 式 loss 调度值得借鉴；可通过 T-SNE 可视化验证特征可分性作为辅助分析手段。
- 使用 Q-Align、MUSIQ 等无参考感知指标与 LPIPS 形成互补，避免仅凭 PSNR/SSIM 判断主观质量。

## 关键术语表
- **UAR-Net**：本文提出的统一恶劣天气图像恢复网络。
- **GDTB**：Gated Dual-scale Transformer Block，含 SGA 与 DGFF 的注意力-前馈组合模块。
- **SGA**：Selective Gated Attention，通过 sinusoidal 重加权与 query 门控实现自适应全局聚合。
- **BMSC**：Balanced Multi-scale Skip Connection，基于 P–C 多步积分的渐进多尺度跳连。
- **vHeat / HCO**：基于二维热传导方程的频率域扩散注意力机制与算子。
- **URH**：Uncertainty-Aware Refinement Head，并行预测均值与方差的细化头。
- **BAE-Loss**：Brightness-Aware Energy Loss，结合能量评分与亮度自适应回归的训练目标。
- **AiO**：All-in-One，单一模型处理多种退化类型的统一恢复范式。

## 可复现要素
- 数据集：Snow100K、Raindrop、Outdoor-Rain（公开基准）；训练集规模论文已给出。
- 代码/权重：论文声明 "codes will open source upon acceptance"，截至来源版本未提供稳定开源链接。
- 关键超参：AdamW，初始 lr 3e-4，cosine 退火至 1e-6；五阶段 patch 128/160/256/320/360，batch 8/5/2/1/1；编码器块数 4/4/6/8，基础通道 36，DGFF 扩张比 2.667，head 数 1/2/4/8；BAE annealing 步 300k。
