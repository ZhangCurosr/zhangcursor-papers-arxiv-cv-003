---
title: "Uncertainty-Guided-Adverse-Weather-Restoration-via-Gated-Tra"
source: https://arxiv.org/pdf/2609.02434v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 18:12:49"
field: "图像复原与增强"
keywords: ["图像复原", "恶劣天气移除", "全合一网络", "门控Transformer", "不确定性建模", "线性注意力", "多尺度融合"]
innovations: ["提出门控双尺度Transformer块GDTB，通过正弦重加权线性注意力与数据相关门控实现天气条件自适应的全局上下文聚合", "设计基于预测器-校正器数值积分的平衡多尺度跳跃连接BMSC，渐进式抑制浅层含噪特征的传播", "提出不确定性感知细化头URH与亮度感知能量损失BAE-Loss，联合优化像素级重建与不确定性校准"]
benchmarks: ["Snow100K-S", "Snow100K-L", "Outdoor-Rain", "Raindrop"]
---

# 论文速读：Uncertainty-Guided-Adverse-Weather-Restoration-via-Gated-Tra

## 一句话总结
本文提出 UAR-Net，一个面向多云天气图像的统一全合一（All-in-One）复原网络，通过门控双尺度 Transformer 块（GDTB）、平衡多尺度跳跃连接（BMSC）和不确定性感知细化头（URH）协同建模，结合亮度感知能量损失（BAE-Loss）联合优化像素级重建与不确定性估计，在多个恶劣天气基准上实现 SOTA 性能。

## 研究问题与动机
1. **空间异质性退化挑战统一建模**：雨、雪、雾等不同天气退化在空间分布上高度不均匀，现有方法对全局上下文进行天气无关的均匀聚合，导致严重退化区域的处理能力受限。
2. **跳跃连接直接拼接传播噪声**：多数全合一方法采用简单的 concat 或 add 跳跃连接，浅层含噪特征会直接传播到解码端，放大天气特异性噪声并削弱跨尺度交互。
3. **确定性目标在高不确定性区域过自信**：统一场景下存在多样性天气条件和退化程度，忽略预测不确定性会导致模型在模糊或严重退化区域输出过自信且不稳定。
4. **现有全合一方法上下文建模容量有限**：早期共享表征方法缺乏全局建模能力；近期基于 Transformer 的方法虽有提升，但未能针对不同天气的条件自适应特征选择。

## 核心贡献（创新点）
1. **提出 UAR-Net 统一框架，集成 GDTB、BMSC 和 URH**：区别于 Histoformer 等基线中 Uniform global aggregation，UAR-Net 通过查询条件门控实现天气感知的自适应特征建模。
2. **引入 Gated Dual-Scale Transformer Block（GDTB）**：将查询投影拆分为注意力查询和数据相关门控分支，结合正弦调制线性注意力实现选择性全局聚合；与 CosFormer 等基线相比，本质区别在于引入门控信号实现内容自适应的上下文抑制与强调。
3. **设计 Balanced Multi-scale Skip Connection（BMSC）**：用预测器-校正器（P-C）方案替换直接跳跃拼接，从深层到浅层渐进式集成编码器特征，通过线性多步积分框架（LMF）稳定跨尺度信息传递，避免浅层含噪特征直接传播。
4. **提出 Uncertainty-Aware Refinement Head（URH）及 BAE-Loss**：URH 通过并行均值/方差头输出概率预测，BAE-Loss 结合能量分数（Energy Score）与亮度感知回归项，联合监督重建与不确定性估计；区别于传统 $\ell_1$/$\ell_2$ 点态损失，该损失鼓励预测分布在真值附近而非单点匹配。
5. **在多个基准上实现 SOTA**：平均 PSNR 达 34.46 dB / SSIM 0.9500，相对 Histoformer 提升 +0.78 dB，相对 MODEM 提升 +0.28 dB。

## 方法详解

### 整体架构
UAR-Net 采用 U-Net 式编码器-解码器结构，由堆叠的 GDTB 构成层次化特征提取与重建。每级均含 SGA（Selective Gated Attention）+ DGFF（Dual-scale Gated Feed-Forward），并通过 BMSC 实现跨尺度特征融合。最终由 URH 进行精细修正并输出不确定图。另有补充跳跃连接（平均池化 + $1\times1$ 卷积 + $3\times3$ 深度卷积）保留低频先验。

### GDTB：门控双尺度 Transformer 块
- **从 Softmax 注意力到线性注意力**：标准自注意力复杂度 $\mathcal{O}(N^2)$，通过核函数分解 $\mathrm{sim}(q,k)=\phi(q)^\top\phi(k)$ 转化为线性注意力，复杂度降至 $\mathcal{O}(N)$，常用 $\phi(x)=\mathrm{ELU}(x)+1$。
- **正弦调制（Sinusoidal Reweighting）**：对 Q、K 施加位置相关的余弦/正弦重加权：
  $$s(Q_i, K_j)=Q_i'(K_j')^\top\cos\!\left(\frac{\pi(i-j)}{2M}\right)$$
  利用三角恒等式分解为 cos/sin 两个互补通道，编码相对位置关系，增强局部结构保持能力，同时保持线性复杂度。
- **门控机制（Gating）**：查询投影展开为 $[Q_i, G_i]=X_iW_q$，其中 $G_i$ 经 sigmoid 生成数据相关门控信号，对线性注意力输出进行逐元素调制：
  $$\tilde{O}_i=\sigma(G_i)\odot O_i$$
  门控引入数据依赖的非线性，打破纯线性变换，充当选择性过滤器。
- **DGFF 局部增强**：采用两路并行深度可分离卷积（$3\times3$ 与 $5\times5$ 感受野）捕获多尺度局部模式，经门控自适应融合。

### BMSC：平衡多尺度跳跃连接
- **问题动机**：不同天气条件下浅层特征受污染程度不同（如雨痕、雪花、雾），直接拼接会传播不可靠特征。
- **预测器-校正器（P-C）融合**：将编码特征 $\{X_n\}$ 从深到浅渐进集成：
  - 连续动力学建模：$\dot{Y}(t)=f(Y(t)+g(X(t)))-Y(t)$，其中 $-Y(t)$ 为稳定衰减项防止累积噪声。
  - 离散化采用两步 Adams-Moulton 校正：先 Euler 显式预测 $\bar{Y}_{n+1}=Y_n+\delta F(Y_n,X_n)$，再用隐式校正 $Y_{n+1}=Y_n+\frac{\delta}{2}(F(Y_n,X_n)+F(\bar{Y}_{n+1},X_{n+1}))$。
  - 一般化为线性多步融合（LMF）：$Y_{n+1}=Y_n+\delta\sum_{j=0}^{K-1}\beta_j F(Y_{n-j},X_{n-j})$，本文取 $K=4$。
- **vHeat 精炼**：采用 Physics-inspired Heat Conduction Operator（HCO），将特征全局聚合视为热扩散过程：
  $$U_t=\mathrm{IDCT2D}\!\left(\mathrm{DCT2D}(U_0)\cdot\exp(-k(\omega_x^2+\omega_y^2)t)\right)$$
  通过可学习的频率值嵌入（FVEs）动态预测扩散强度 $k$，实现内容自适应的全局精炼，复杂度 $\mathcal{O}(N^{1.5})$。
- **平衡特征尺寸**：默认取 $H/4\times W/4$，兼顾空间细节与语义鲁棒性，且计算开销最低。

### URH：不确定性感知细化头
- 采用紧凑四阶段 U-Net 结构（各阶段含 GDTB 块），保留标准跳跃连接（因跨尺度融合已由 BMSC 完成）。
- 并行均值头 $\mu(x)$ 和方差头 $\sigma^2(x)$ 输出像素级高斯分布，方差作为任务驱动信号反映空间恢复难度（严重退化区域方差高，清晰区域方差低）。

### BAE-Loss：亮度感知能量损失
- **能量分数损失（Energy Score）**：从预测高斯分布采样 $M=1000$ 次，评估概率质量分布：
  $$\mathcal{L}_{\mathrm{ES}}=\frac{1}{N}\sum_{n}\left(\frac{1}{M}\sum_i\|z_{n,i}-z_n\|_1-\frac{1}{2(M-1)}\sum_{i}\|z_{n,i}-z_{n,i+1}\|_1\right)$$
  非局部度量，鼓励预测分布将概率质量集中于真值附近。
- **亮度感知回归损失**：
  $$\mathcal{L}_{\mathrm{GT}}=W\|f(x)-y\|_1+(1-W)\left\|\frac{\mu_y}{\mu_{f(x)}}f(x)-y\right\|_1$$
  权重 $W$ 由预测与真值的 Bhattacharyya 距离自适应计算，平衡原始预测与亮度对齐版本。
- **总损失**：$\mathcal{L}_{\mathrm{BAE}}=(1-w_{\mathrm{prob}}(t))\mathcal{L}_{\mathrm{GT}}+w_{\mathrm{prob}}(t)\mathcal{L}_{\mathrm{ES}}$，其中 $w_{\mathrm{prob}}(t)$ 为训练过程中逐步增大的退火权重。
- **关联损失**：$\mathcal{L}_{\mathrm{cor}}=\frac{1}{2}(1-\rho(I_{\mathrm{HQ}},I_{\mathrm{GT}}))$，鼓励全局结构与强度一致性。
- **最终训练目标**：$\mathcal{L}=\mathcal{L}_{\mathrm{BAE}}+\mathcal{L}_{\mathrm{cor}}$。

## 实验与结果

### 数据集
- **Snow100K**：10 万张合成雪景图，训练集 9,000 张；测试集包括 Snow100K-S（小雪）、Snow100K-L（大雪）和 Snow100K-Real（真实雪景）。
- **Raindrop**：1,319 对真实雨滴退化图像，训练 1,069 对，测试 249 对。
- **Outdoor-Rain**：9,000 张合成雨雾混合退化图；测试用 Test1 split。
- 训练时三数据集合并为统一多天气训练集。

### 评估基线
- **全合一基线**：All-in-One [4]、TransWeather [50]、Restormer [40]、Chen et al. [51]、WGWSNet [52]、WeatherDiff64 [45]、PromptIR [53]、DiffUIR-L [54]、Histoformer [19]、MODEM [21]、HOGformer [24]。
- **任务特定基线**：SPANet、JSTASR、RESCAN、DesnowNet、DDMSNet、ConvIR（除雪）；CycleGAN、pix2pix、HRGAN、PCNet、MPRNet、NAFNet（去雨）；pix2pix、DuRN、RaindropAttn、AttentiveGAN、MAXIM、AST（去雨滴）。

### 主要结果
| 数据集 | Ours PSNR/SSIM | 次优 MODEM PSNR/SSIM | 提升 |
|---|---|---|---|
| Snow100K-S | **38.34 / 0.9697** | 38.08 / 0.9673 | **+0.26 / +0.0024** |
| Snow100K-L | **32.77 / 0.9324** | 32.52 / 0.9292 | **+0.25 / +0.0032** |
| Outdoor-Rain | **33.40 / 0.9491** | 33.10 / 0.9410 | **+0.30 / +0.0081** |
| Raindrop | **33.32 / 0.9487** | 33.01 / 0.9434 | **+0.31 / +0.0053** |
| **平均** | **34.46 / 0.9500** | **34.18 / 0.9452** | **+0.28 / +0.0048** |

- 相对 Histoformer：平均 **+0.78 dB PSNR**。
- **感知质量**（Table IV）：UAR-Net 在所有数据集上均取得最低 LPIPS 和最高 Q-Align / MUSIQ 分数。
- **T-SNE 可视化**：UAR-Net 特征聚类更紧凑、类间分离更清晰，表明门控与 BMSC 实现了更好的天气条件感知特征解耦。
- **复杂度**：35.44 GMac 的 FLOPs，在同等计算预算下精度最优（Fig. 9）。

### 消融实验关键结论
1. **组件贡献**（Table V）：GDTB +0.09 dB → +BMSC +0.15 dB → +BAE +0.21 dB → +URH +0.25 dB，各组件正交互补。
2. **正弦重加权**（Table VI-a）：平均 +0.05 dB，主要增强局部感知交互。
3. **门控机制**（Table VI-b）：平均 **+0.27 dB**，是最关键组件。
4. **BMSC 内部设计**（Table VII）：LMF 积分 vs 平均池化 +0.18 dB；vHeat 精炼略优于 CosFormer 和 Nonlocal。
5. **平衡特征尺寸**（Table VIII）：$H/4\times W/4$ 最优且计算最省。
6. **BAE-Loss 退火步数**（Table IX）：300k 步（全程保留回归损失）最优。

## 相关工作脉络
1. **Histoformer [19]**：UAR-Net 的核心基线，采用直方图 Transformer 实现强度感知的多尺度特征聚合；本文定位为在其基础上引入门控选择性和不确定性建模，解决其天气无关的均匀聚合和确定性预测缺陷。
2. **MODEM [21]**：Morton 排序退化估计机制，当前最强全合一方法之一；本文在平均 PSNR 上超越 MODEM +0.28 dB，差异在于 UAR-Net 显式建模不确定性而 MODEM 未考虑。
3. **HOGformer [24]**：梯度条件注意力 + 显式先验的全合一框架；本文相较其在 BMSC 渐进融合和 BAE-Loss 不确定性监督方面各有侧重，综合性能更优。
4. **任务特定方法**（DesnowNet [29]、AttentiveGAN [31]、PCNet [60] 等）：针对单一天气设计的专用模型；本文定位为统一全合一方案，在相同统一设定下超越多数任务特定方法。
5. **Linear Attention 系列**（CosFormer [36]、Restormer [40]）：以线性复杂度建模全局依赖；本文在其基础上引入正弦重加权与门控机制，实现更细粒度的选择性上下文聚合。
6. **Uncertainty-aware Restoration**（如 UrNet [42]）：事件相机立体深度估计中的不确定性建模；本文将不确定性建模迁移至图像复原领域，采用能量分数而非标准方差损失进行监督。

## 局限性与未来方向
1. **高计算复杂度**：35.44 GMac 高于多数对比方法（Histoformer 23.27、MODEM 29.08），在实际部署中仍需效率优化。
2. **Monte Carlo 采样开销**：BAE-Loss 需 1000 次采样估计能量分数，仅训练阶段使用，推理时无额外开销，但训练时间成本较高。
3. **训练数据依赖合成与少量真实数据**：Snow100K 为合成数据，Raindrop 仅 249 对真实测试图像，泛化到极端真实场景的能力有待进一步验证。
4. **未涉及视频或多帧场景**：当前为单帧图像复原，未来可探索时序一致性扩展。
5. **退化类型有限**：仅覆盖雨、雪、雾、雨滴四种常见退化，未涉及沙尘、冰霜等其他恶劣天气。

## 研究启发与可借鉴点
1. **门控线性注意力用于条件自适应聚合**：将查询投影拆分为注意力 + 门控两个分支，可迁移至其他需要条件自适应的全局聚合任务（如低光照增强、去模糊），实现"选择性上下文"而非均匀聚合。
2. **BMSC 的 P-C 数值积分视角**：将跳跃连接建模为 Adams-Bashforth/Moulton 多步积分过程，提供了跨尺度特征融合的新理论框架，可推广至分割、分割等密集预测任务的特征融合设计。
3. **BAE-Loss 的能量分数监督范式**：在非局部密度估计中，用 Energy Score 替代 MSE/MAE 进行概率预测监督的思路，可迁移至超分、去噪等密集回归任务的不确定性建模。
4. **vHeat 热扩散精炼模块**：以热传导方程为启发的全局注意力近似，复杂度 $\mathcal{O}(N^{1.5})$ 且保证全局感受野，可作为轻量级全局建模组件嵌入其他 Transformer 架构。
5. **退火式损失融合策略**：BAE-Loss 中回归损失与能量损失的比例随训练逐步切换，这种"先稳后精"的训练策略对多损失联合优化的鲁棒性有参考价值。

## 关键术语表
**All-in-One (AiO) 全合一复原**：在单个统一网络中同时处理多种恶劣天气退化的图像复原范式，避免为每种退化单独训练模型。
**Gated Dual-Scale Transformer Block (GDTB)**：UAR-Net 的核心块，集成门控线性注意力（SGA）与双尺度门控前馈（DGFF），实现选择性全局聚合与多尺度局部增强。
**Selective Gated Attention (SGA)**：将查询投影拆分为注意力查询和数据相关门控分支，通过 sigmoid 门控对线性注意力输出进行内容自适应的加权调制。
**Balanced Multi-scale Skip Connection (BMSC)**：基于预测器-校正器（P-C）方案的渐进式跨尺度特征融合模块，通过线性多步积分从深到浅集成编码器特征，抑制浅层噪声传播。
**Uncertainty-Aware Refinement Head (URH)**：输出像素级均值与方差的精细修正头，通过并行头估计恢复图像的分布参数，方差反映空间恢复难度。
**Brightness-Aware Energy Loss (BAE-Loss)**：结合能量分数（Energy Score）概率评估与亮度感知回归的损失函数，通过退火权重在训练中逐步引入不确定性监督。
**Energy Score**：一种严格正当的多元概率预测评估准则，通过蒙特卡洛采样衡量预测分布与真值的距离，鼓励概率质量集中于真值附近。
**vHeat**：基于热传导方程的注意力机制，通过离散余弦变换（DCT）在频域实现全局特征扩散，复杂度 $\mathcal{O}(N^{1.5})$。
**Linear Multistep Fusion (LMF)**：BMSC 中的渐进特征融合策略，采用 Adams-Bashforth/Moulton 格式的预测器-校正器更新规则整合多尺度编码器特征。
**Sinusoidal Reweighting**：对线性注意力的查询和键施加基于相对位置的余弦/正弦重加权，增强局部结构保持能力同时维持线性计算复杂度。

## 可复现要素
- **数据集**：Snow100K（公开）、Raindrop（公开）、Outdoor-Rain（公开）；训练集为三者合并，论文未提供预处理脚本细节。
- **代码/权重**：论文声明"codes will open source upon acceptance"，目前（2025）尚未开源；模型权重未公开。
- **训练环境**：PyTorch，4 × NVIDIA H100 GPU，共 300,000 次迭代。
- **关键超参**：
  - 渐进学习：5 阶段，patch size = {128, 160, 256, 320, 360}，batch size = {8, 5, 2, 1, 1}，迭代数 = {92k, 84k, 56k, 36k, 32k}。
  - 优化器：AdamW，初始 lr=$3\times10^{-4}$，前 92k 步不变，之后 cosine annealing 至 $1\times10^{-6}$。
  - 架构：编码器各阶段 GDTB 块数 {4, 4, 6, 8}，基础通道 C=36，DGFF 扩展因子 r=2.667，注意力头数 {1, 2, 4, 8}。
  - BAE-Loss 退火：$w_{\mathrm{prob}}(t)$ 在 300k 步内逐步增大。
  - 能量分数采样：M=1000 次蒙特卡洛采样。
  - 数据增强：随机水平与垂直翻转。
