---
title: "VeCAS-Vessel-Focused-Contrast-Free-Angiogram-Synthesis-for-V"
source: https://arxiv.org/pdf/2608.22828v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 00:53:01"
---

# 论文速读：VeCAS: Vessel-Focused Contrast-Free Angiogram Synthesis for Vascular Interventions

## 一句话总结
本文针对碘对比剂在血管介入手术中的安全风险，提出两阶段**VeCAS**框架，通过显式解耦血管结构定位与造影外观合成，直接从非增强X光图像生成高真实感的数字减影血管造影（DSA）图像；体模机器人导航实验表明，该方法可将导丝导航时间与操作步数分别降低约41.4%与40.7%，具备作为“meta contrast agent”的临床潜力。

## 研究问题与动机
1. **临床痛点**：X射线血管介入高度依赖碘对比剂以增强血管可视化，但对比剂注射量增加会显著提升急性肾损伤（AKI）与过敏反应风险（如每增加75 mL对比剂，AKI风险上升42%），临床亟需低/无对比剂替代策略。
2. **现有AI方法局限（定位失控）**：主流方法多将任务建模为全局图像到图像翻译（image-to-image translation），缺乏对血管空间位置的显式控制，易生成视觉逼真但解剖位置偏移的伪血管，可能误导术中器械导航。
3. **现有AI方法局限（背景冗余）**：血管仅占图像局部区域，非血管背景无需生成。现有全局生成模型将大量容量浪费于冗余背景重建，削弱了对细微血管结构的建模能力。
4. **方法学空白**：直接从未注射对比剂的X光图像生成近似DSA的造影图像仍属探索盲区，缺乏兼顾结构保真度与计算效率的可靠方案。

## 核心贡献（创新点）
1. **提出两阶段血管聚焦无对比剂造影合成框架VeCAS**，将任务拆分为独立的定位与生成阶段，突破传统单模型全局映射的耦合瓶颈。
2. **设计跨模态隐特征蒸馏策略**，冻结已在造影图像上训练的分割模型参数，仅训练非增强图像编码器与多级预测器，以灵活的隐空间预测桥接对比增强与非增强X光的巨大模态鸿沟。
3. **开发血管聚焦扩散修复模块**，联合血管聚焦去噪损失与修复采样策略，使扩散模型仅在有血管掩码区域内执行去噪生成，非血管背景直接由输入帧保留，显著提升生成效率与结构保真度。
4. **提供多维度临床效用验证**，除定量指标与视觉图灵测试外，首次引入血管体模机器人导丝导航实验，证明VeCAS引导可大幅缩短导航时间并减少操作步数，性能逼近真实造影引导。

## 方法详解
**整体架构**：给定非增强X光图像 $x_N$，Stage I 输出二值血管掩码 $\hat{m} \in \{0,1\}^{1\times H\times W}$；Stage II 以 $\hat{m}$ 与 $x_N$ 为条件，生成合成造影图像 $\hat{x}_A$。

**Stage I：血管结构定位（判别式）**
- 采用U-Net架构，编码器 $\mathcal{E}$ 提取特征后经解码器 $\mathcal{D}$ 预测掩码：$\hat{m} = \mathcal{D}(\mathcal{E}(x_N))$。
- **跨模态隐特征蒸馏**：
  1. 先用造影图像 $x_A$ 与真值掩码 $m$ 训练造影域分割模型 $\mathcal{D}_A(\mathcal{E}_A(\cdot))$，随后冻结其参数。
  2. 训练非增强图像专用编码器 $\mathcal{E}_N(\cdot)$（与 $\mathcal{E}_A$ 共享U-Net架构）及多级预测器 $\mathcal{P}(\cdot)$。
  3. 提取多尺度特征 $f_A^{(i)} = \mathcal{E}_A^{(i)}(x_A)$ 与 $f_N^{(i)} = \mathcal{E}_N^{(i)}(x_N)$，预测器仅作用于高层特征（$i=3,4,5$）：$\hat{f}_A^{(i)} = \mathcal{P}^{(i)}(f_N^{(i)})$。
  4. 将预测特征 $\hat{f}_A^{(i)}$ 与低层特征 $f_N^{(i)}$ 拼接后输入冻结解码器 $\mathcal{D}_A$ 得到 $\hat{m}$。
- **损失函数**：
  - 分割损失：$\mathcal{L}_{seg} = 0.5\mathcal{L}_{BCE} + 0.5\mathcal{L}_{Dice}$
  - 特征预测损失：$\mathcal{L}_{pred} = \sum_{i=3}^{5} \|\hat{f}_A^{(i)} - f_A^{(i)}\|_2^2$
  - 总损失：$\mathcal{L}_{distill} = \mathcal{L}_{seg} + \lambda_{pred}\mathcal{L}_{pred}$

**Stage II：血管聚焦扩散修复**
- 基于DDPM反向扩散过程，输入为造影图像 $x_A$ 及其血管掩码 $m$。
- **血管聚焦去噪损失**：仅对血管区域像素计算噪声预测误差，抑制背景梯度干扰：
  $\mathcal{L}_{denoise} = \mathbb{E}_{x_A, t, \epsilon}[\|m \odot (\epsilon_\theta(x_{A,t}, t) - \epsilon)\|_2^2]$
- **血管聚焦采样（推理）**：从纯高斯噪声 $x_T$ 出发，每步计算去噪输出 $o_{t-1}$ 后，按Stage I预测掩码 $\hat{m}$ 与正向加噪的非增强图像 $x_{N,t-1}$ 融合：
  $\hat{x}_{t-1} = o_{t-1} \odot \hat{m} + x_{N,t-1} \odot (1 - \hat{m})$
  最终输出 $\hat{x}_A = \hat{x}_0$，实现“血管生成+背景冻结”。

## 实验与结果
- **数据集**：北京友谊医院（Capital Medical University）下肢血管介入病例，共599对非增强/造影帧（102名患者），按患者水平进行5折交叉验证。伦理批号：2020-P2-073-02。
- **评估指标**：Stage I 使用 DSC 与 clDice；Stage II 额外评估 PSNR、SSIM；辅以 Visual Turing Test、医师评估（VMR/COR/PU）、体模机器人导航实验。
- **Stage I 定位结果**：VeCAS 取得最佳性能，DSC **52.56%**，clDice **60.62%**，较次优分割模型 UNet++（DSC 51.39%, clDice 59.87%）提升 +1.17% / +0.75%；显著优于各类生成模型（如 Pix2Pix、ControlNet）与跨模态对齐方法（KD-Net、PMKL等）。
- **Stage I + II 合成结果**：VeCAS 综合最优，DSC **42.27%**，clDice **47.70%**，较次优方法 ControlNet（+5.84%）与 RegGAN（+5.92%）提升明显；PSNR 36.78 dB，SSIM 0.9537。全局指标（PSNR/SSIM）受背景保留主导，结构性指标更能反映血管保真度。
- **感知与医师评估**：Visual Turing Test 平均 Sensitivity 77.50%、Specificity 60.50%、Accuracy 69.00%；三位医师对 VMR、COR、PU 三项评分均 >3.00（5分制），Kappa 系数 0.46~0.84，提示生成图像视觉真实且具备手术引导价值。
- **体模导航实验**：机器人导丝导航中，VeCAS 引导较无对比剂引导 **导航时间减少 41.4%**、**操作步数减少 40.7%**，性能与真实造影引导相当。

## 相关工作脉络
1. **低/无对比
