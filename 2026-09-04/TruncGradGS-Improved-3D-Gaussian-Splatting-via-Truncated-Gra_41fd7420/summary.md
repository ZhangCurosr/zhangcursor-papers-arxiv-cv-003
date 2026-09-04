---
title: "TruncGradGS-Improved-3D-Gaussian-Splatting-via-Truncated-Gra"
source: https://arxiv.org/pdf/2609.03534v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 21:31:24"
---

# 论文速读：TruncGradGS-Improved-3D-Gaussian-Splatting-via-Truncated-Gra

## 一句话总结
本文针对 3D Gaussian Splatting 中因分块光栅化与高斯导数指数衰减导致的梯度消失问题，提出分段截断梯度（piecewise truncated gradient）优化框架，在保持 tiled rasterization 效率的同时显著扩大原语的有效感受野；该方法可作为即插即用模块稳定提升静态与动态 3DGS 的重建质量，并同步发布了一个面向大时空位移的合成动态评测基准。

## 研究问题与动机
1. **分块光栅化的局部性瓶颈**：图像被划分为 16×16 像素 tile，仅 tile 内像素参与反向传播；远离 Gaussian 投影中心的像素无法提供梯度，导致原语难以向关键区域迁移。
2. **高斯导数尾区梯度消失**：2D 高斯偏导数均含 $G(\Delta)$ 乘积项，在 Mahalanobis 距离较大时指数衰减至近零，链式法则断裂，远端像素即使未被 tile 裁切也无法驱动有效更新。
3. **随机/动态初始化下的探索困境**：现有自适应密度控制（ADC）依赖克隆/分裂逐步扩展覆盖，但在随机初始化或大帧位移动态场景中，初始原语往往错过关键像素，造成 floaters、模糊或结构缺失。
4. **动态场景评测标准偏低**：现有基准（如 N3DV、ENeRF-Outdoor）普遍序列短、形变简单或缺乏工业级光照/物理真实度，难以反映方法在复杂非刚性运动下的极限能力。

## 核心贡献（创新点）
1. **形式化揭示梯度消失的数学根源**：推导 2D 高斯对均值的偏导公式，明确指出 $G(\Delta)$ 乘法项与 tile 裁切共同导致远端梯度归零；区别于以往仅改进初始化或密度控制的路线，本文直接从梯度场本身入手。
2. **提出分段截断梯度场**：在高斯等值线轮廓内保留真实偏导，外部用连续线性函数替代近零尾区；本质区别在于无需改动 tiled rasterization 架构，仅通过梯度替换与边界连续性保证实现“远距离拉力”。
3. **设计稳定训练策略**：仅对不透明度低于剪枝阈值的“死亡 Gaussian”启用截断梯度，配合 radius padding、延迟剪枝与负 alpha 过滤，防止长程梯度引发高质量原语偏移或原语间相互排斥。
4. **发布高难度动态合成基准**：构建 6 个含流体、半透明、大非刚性运动的 Blender 渲染序列，填补现有动态 3DGS 评测在时空发散性与工业真实度方面的空白。

## 方法详解
- **梯度分析**：对投影后 2D 高斯密度 $G^{2D}(\Delta) = \exp(-\frac{1}{2}\Delta^T \Sigma'^{-1} \Delta)$ 求偏导，得到 $\frac{\partial G}{\partial \Delta_x} = -G(\Delta)(\Delta_x a + \Delta_y c)$ 与 $\frac{\partial G}{\partial \Delta_y} = -G(\Delta)(\Delta_y b + \Delta_x c)$，表明梯度过快衰减源于 $G(\Delta)$ 自身。
- **分段截断公式**：
  $$
  \widetilde{\nabla G} = \begin{cases}
  \frac{\partial G}{\partial \Delta_x} & \text{if } -2\log G < -2\log \tau \\
  \min(m \cdot x + b,\, \frac{\partial G}{\partial \Delta_x}) & \text{if } \frac{\partial}{\partial \Delta_x}G(b) < 0 \\
  \max(-m \cdot x + b,\, \frac{\partial G}{\partial \Delta_x}) & \text{if } \frac{\partial}{\partial \Delta_x}G(b) > 0
  \end{cases}
  $$
  其中密度阈值 $\tau$ 设为 alpha-blending discard threshold（$1/255$），斜率 $m$ 为可调超参，截距 $b$ 由边界点 $x_b$ 处的连续性 $G^{2D}(x_b)=\tau$ 解出。
- **三元梯度路径**：
  - **Normal path**：$\sigma\alpha \geq 1/255$，使用真实梯度。
  - **Truncated path**：$\sigma\alpha < 1/255$ 且 Mahalanobis 距离超过阈值，使用线性截断梯度。
  - **Skip path**：贡献极低且距离在阈值内，无梯度更新。
- **防发散机制**：仅对 $\sigma_i < \tau_{\text{pruning}}$ 的死亡 Gaussian 应用截断梯度；采用截断梯度优化与 ADC 开启的正常优化交错进行；剪枝操作延迟至训练末尾。
- **Tile 覆盖扩展**：为死亡 Gaussian 扩大投影 bounding box 半径（radius padding），使其分配至更多 tile，使截断梯度能作用于更远的像素误差。
- **动态场景加速**：通过帧差与图像处理生成运动掩码，忽略静态像素的截断梯度计算，训练耗时控制在基线的 1–10 倍。

## 实验与结果
- **数据集**：Mip-NeRF360、自建静态子集、Neural 3D Video (N3DV)、自建 6 场景动态基准（alley, windy tree, water cup, neon city, bouncy balls, underwater）。
- **评估指标**：PSNR ↑、SSIM ↑、LPIPS ↓、#GS ↓。
- **静态结果**（Table 2）：
  - 随机初始化：3DGS+Ours 在 Our dataset 上 PSNR 24.74→25.16，SSIM 0.8351→0.8511，#GS 1.126M→0.799M；Mip-NeRF360 上 PSNR 20.92→22.18。2DGS+Ours 同样一致提升。
  - COLMAP 初始化：3DGS+Ours PSNR 30.07→30.86，SSIM 0.9058→0.9097，#GS 1.217M→0.768M。
- **动态结果**（Table 3, 4）：
  - N3DV：4DGS+Ours PSNR 30.22→31.70（+1.48 dB），SSIM 0.9443→0.9493。
  - 自建难例基准：4DGS(θ)+Ours PSNR 达 27.56，显著优于 4DGS(φ)+Ours 的 26.08；CEM-4DGS+Ours 亦取得正向增益。
- **消融**（Table 5）：移除截断梯度下降最大；移除死亡 Gaussian 过滤导致 PSNR 暴跌至 21.87；半径填充与延迟剪枝贡献次之；负 alpha 过滤防止长程排斥。

## 相关工作脉络
1. **SfM-free 初始化**：RAIN-GS、Librated-GS 通过稀疏大方差初始化或单目深度对齐降低 SfM 依赖；本文聚焦梯度场改进，两者正交可叠加。
2. **密度控制/剪枝**：Taming 3DGS、ConeGS、Pixel-GS、LP-3DGS 改进克隆/分裂/剪枝策略；本文不改变 ADC 宏观逻辑，而是为其提供更有效的长程梯度信号。
3. **PAPR [ZPML23]**：同样关注点渲染中的梯度消失，但针对点云辐射场与随机采样传播；本文公式推导与截断策略专为 Gaussian 投影与 tiled rasterization 设计。
4. **动态 3DGS**：4DGS、CEM-4DGS、4D-Scaffold 等建模时空变形；本文作为即插即用优化模块，显著缓解大帧位移下的梯度衰减，兼容主流动态框架。
5. **前馈/免优化方法**：LGM、Flash3D 用神经网络直接预测 Gaussian；本文坚持迭代优化范式，在稀疏/难初始化与长序列一致性上仍具灵活优势。
6. **动态评测基准**：N3DV、ENeRF-Outdoor 等偏短序列或低复杂度；本文提出含流体、半透明与复杂非刚性运动的合成基准，推动评测标准升级。

## 局限性与未来方向
- **训练开销增加**：更多 Gaussian-tile 配对参与反向传播与截断梯度计算，训练时间约为基线的 1–10 倍，存在效率与鲁棒性的权衡。
- **超参未联合调优**：除延迟剪枝外未重新调整基线方法的 ADC/剪枝超参，保守集成策略在 SSIM 等敏感指标上仍有提升空间。
- **线性近似局限**：极端长距离下线性代理可能引入优化偏差，斜率 $m$ 与阈值 $\tau$ 的选择依赖经验。
- **未来方向**：与专用密度控制/剪枝调度联合优化；探索自适应斜率或可学习截断边界；扩展至极稀疏视图/单目初始化场景；验证于真实采集动态数据。

## 研究启发与可借鉴点
1. **从梯度场本身破局**：跳出“调初始化/调密度控制”的常规思路，直接修改可微渲染的底层偏导表达式，用连续线性代理替换指数尾区，方法简洁且可迁移至其他基于核函数的光栅化方法。
2. **死/活原语区分策略**：仅对低透明度原语施加长程修正，既激活“沉睡”原语的探索能力，又保护已收敛结构不被扰动，
