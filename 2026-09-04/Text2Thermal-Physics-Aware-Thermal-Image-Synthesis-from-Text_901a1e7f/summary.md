---
title: "Text2Thermal-Physics-Aware-Thermal-Image-Synthesis-from-Text"
source: https://arxiv.org/pdf/2609.03585v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 18:42:28"
---

# 论文速读：Text2Thermal: Physics-Aware Thermal Image Synthesis from Textual Priors

## 一句话总结
针对 RGB 到热红外（TIR）图像映射固有的辐射度学病态性，提出 Text2Thermal 框架，通过物理感知结构化文本先验驱动参数高效微调的 Stable Diffusion，实现无需配准可见光图像的纯文本热图生成，并可附加 ControlNet 分支提供空间结构引导。

## 研究问题与动机
- 热红外成像在夜间、低照度及恶劣天气下具有不可替代的感知优势，但高质量热图数据集稀缺且采集/标注成本高昂。
- 现有 RGB-to-TIR 翻译方法存在根本性物理缺陷：热图像强度是物体自身发射辐射、环境反射辐射与大气辐射的叠加，其中决定发射项的表面发射率与绝对温度在可见光谱中不可观测，导致单张 RGB 图像对应多个合法热分布（一对多病态映射）。
- 通用 T2I 模型缺乏热域先验，且标准 caption 仅描述可见外观，无法编码热物理属性；直接迁移会导致生成结果停留在可见光谱统计分布上（FID 高达 287）。
- 需要一种机制绕过从可见光推断热属性的不可行路径，转而显式注入辐射度学先验，同时保留对场景几何结构的可控性。

## 核心贡献（创新点）
- **物理感知文本到热图生成框架**：首次系统化将热物理属性（材质、天气、时段、发热状态）通过结构化 caption 注入生成过程，从源头化解 RGB-to-TIR 的病态歧义，与“从 RGB 推断热图”的翻译范式形成本质区别。
- **热域参数高效适配**：仅对 SD 1.5 的 attention 投影注入 LoRA（rank=16， trainable params 0.3%），冻结 VAE 与文本编码器，以极低代价将大规模可见光语义先验重定向至热域分布。
- **热域感知的空间控制分支**：将 ControlNet 的可训练分支初始化为热适配后骨干的编码器副本（而非原始 RGB 权重），使控制信号与生成先验处于同一热域特征空间，实现结构对齐的同时不干扰 prompt 决定的辐射度学特征。
- **SOTA 性能与细粒度诊断**：在 FLIR、M3FD、FMB 三个基准上取得最优 FID（FLIR 72.67，较 TherA 提升 >11 分），并提供 caption 属性贡献度、空间模态对比、零样本泛化等系统性消融。

## 方法详解
- **物理感知结构化 Caption**：沿用 TherA 的规范化 schema，每条 prompt 为 $y = \{y_{scene}, y_{object}, y_{material}, y_{heat}\}$，token 直接对应黑体辐射公式中的发射率、绝对温度与环境辐射项，避免可见外观描述污染热域先验。
- **Unconditional Text2Thermal（无空间条件）**：冻结 SD 1.5 骨干，仅在 Q/K/V/O 投影上应用 LoRA：$\mathbf{W}' = \mathbf{W} + \frac{\gamma}{r}\mathbf{B}\mathbf{A}$（$r=16, \gamma=1.0$）。训练目标为标准噪声预测损失 $\mathcal{L}_{LDM}$，使模型学会将“warm engine block”等文本映射为热图高亮辐射区。
- **Conditional Text2Thermal（含空间条件）**：ControlNet 分支克隆自热适配后的 UNet 编码器，通过零初始化 $1\times1$ 卷积与冻结骨干相连。输入可为配准 RGB 帧，或经 G-SAM / Canny / Depth Anything 提取的分割、边缘、深度图。训练仅优化控制分支，损失为 $\mathcal{L}_{ctrl}$，训练时以 0.5 概率空出文本 prompt 以强化分支对空间信号的依赖。
- **推理与引导**：无条件采样仅依赖文本 classifier-free guidance（$s_y=7.5$）；有条件采样同时施加文本引导与空间引导（$s_s=7.5$），采样步数 50（PNDM/UniPC），分辨率 256×256。

## 实验与结果
- **数据集**：预训练使用 R2T2（10 万 RGB-Thermal-Text 三元组）；评测在 FLIR、M3FD、FMB 公开基准上进行。基准 caption 由 InternVL2.5-14B 按 TherA 协议双模态联合生成。
- **评估指标**：FID（分布保真度）、CLIP Score（文本对齐）、Reconstructed BERTScore（语义/物理属性保真度，通过重 caption 闭环计算）、有条件时的 PSNR/SSIM/LPIPS。
- **核心结果**：
  - FLIR：FID **72.67**（SOTA，超越 TherA 83.78 约 11 分，PID 84.26）。
  - M3FD：FID **78.18**（超越 TherA 87.08、DiffV2IR 92
