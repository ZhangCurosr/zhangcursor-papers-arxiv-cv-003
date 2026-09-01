---
title: "Zero-Shot-Video-Restoration-and-Enhancement-with-Text-to-Ima"
source: https://arxiv.org/pdf/2608.26476v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 00:55:07"
field: "视频修复与增强"
keywords: ["零样本视频修复", "潜在扩散模型", "时间一致性", "多模态参考", "Token合并", "双重提示调优"]
innovations: ["提出首个支持文本+图像多模态参考的零样本视频修复框架ZVRM", "双重提示调优反转与采样将推理步数从1000降至360步（约1/3）", "纹理感知视频token合并自适应平衡时间一致性与细节保留"]
benchmarks: ["REDS4", "UDM10", "DAVIS", "DID", "RealMCVSR"]
---

# 论文速读：Zero-Shot Video Restoration and Enhancement with Text-to-Image Latent Diffusion Models and Multi-Modal References

## 一句话总结
论文提出ZVRM框架，利用预训练的文本到图像潜在扩散模型（T2I LDM）实现零样本视频修复与增强，通过双重提示调优加速推理、纹理感知token合并保持时间一致性，并支持多模态参考（文本/图像），显著提升了修复质量和时序一致性。

## 研究问题与动机
- **核心问题**：将预训练的文本到图像扩散模型直接应用于视频修复时，由于缺乏时间建模机制，会产生严重的帧间闪烁（temporal flickering）
- **现有方法不足**：
  - PSLD等零样本图像修复方法专为图像设计，无时间维度建模，直接扩展至视频会产生剧烈时间不一致
  - 已有零样本视频修复方法（ZVRD、DiffIR2VR、VISION-XL）均未考虑引入额外多模态参考（如文本描述、参考图像）来辅助修复
  - DDIM inversion对退化图像不可用——预测噪声偏离标准正态分布，导致初始化偏移，生成不真实结果
  - 现有视频编辑方法的token merging使用统一合并比例，无法兼顾平滑区域与纹理区域的修复难度差异

## 核心贡献（创新点）
- **提出首个支持多模态参考的零样本视频修复框架**：同时支持无参考、文本参考、图像参考及二者组合，与PSLD/VISION-XL等方法本质区别在于引入外部参考信息辅助生成
- **双重提示调优反转与采样（Dual Prompt Tuning Inversion/Sampling）**：同时优化条件嵌入C和无条件嵌入∅，与仅做图像编辑的Null-text inversion本质不同，专为修复任务设计以更好地编码退化图像信息并加速推理（1000步→360步）
- **纹理感知视频token合并（Texture-Aware Video Token Merging）**：根据图像区域类型（平滑/纹理丰富）分配不同合并比例，与VidToMe/TokenFlow的统一合并策略本质不同，更好地平衡时间一致性与纹理保留
- **引用自注意力与引用token合并（Referenced Self-Attention & Referenced Token Merging）**：首次将参考图像纹理跨模态传递给视频帧，与现有无参考方法本质不同，利用参考图像的空间先验增强修复细节

## 方法详解
- **整体框架**：给定含N帧的退化视频，利用预训练T2I LDM（SDv1.5/SDXL）进行修复。将U-Net中的3×3 2D卷积替换为1×3×3膨胀3D卷积以处理视频。
- **双重提示调优反转（DPTI）**：
  - 使用Distortion Adaptive (DA) Inversion将退化latent $z_0 = \mathcal{E}(A^T y)$ 转换为$z_{T'}$（$T'=15$）作为良好初始化
  - 在inversion阶段，依次优化条件嵌入$\mathcal{C}_t$（式8）和无条件嵌入$\mathcal{O}_t$（式9），跨帧一致性项$||\cdot^i - \cdot^{i-1}||_2^2$保证时序稳定
  - 推理步数从1000降至250（60次inversion + 250次sampling）
- **双重提示调优采样（DPTS）**：
  - 在采样阶段继续交替优化$\mathcal{C}_t$和$\mathcal{O}_t$（式10-11），约束各帧嵌入一致
  - 仅当$t < T_{dpts}=5$时进行N=5次嵌入优化，平衡计算开销与性能
- **纹理感知空间token合并（TSTM）**：使用中低频分类器将帧内像素分为平滑区（$\mathbf{T}_s$）和纹理区（$\mathbf{T}_t$），平滑区使用更高合并比例$r_s \gg r_t$，每$N_c=25$步更新分类
- **纹理感知时间token合并（TTTM）**：以第一帧为关键帧，对不同帧间同样按平滑/纹理区域分别合并token，提升时间一致性
- **引用自注意力（RSA）**：用$K'=[v_i, v_{ref}]$和$V'=[v_i, v_{ref}]$替代原始self-attention的K、V，使当前帧可同时关注自身与参考图像
- **引用token合并（RTM）**：以参考图像为key，将当前帧token按相似度合并至参考图像，再unmerge回原帧，实现跨帧纹理传递

## 实验与结果
- **数据集**：REDS4、UDM10、DAVIS（视频修复）；DID（低光视频增强，10对）；RealMCVSR（不同参考图像消融）；所有测试帧resize至512×512
- **评估指标**：PSNR↑、SSIM↑、Warping Error WE↓（时间一致性）
- **主要结果**（SDv1.5 backbone，4×视频超分）：
  - ZVRM（no Ref.）：PSNR 25.16 / SSIM 0.6599 / WE 0.5003，较PSLD提升0.31 dB，WE降至约1/4
  - ZVRM（text Ref.+image Ref. 1）：PSNR 25.61 / SSIM 0.6785 / WE 0.3871
  - 低光增强：ZVRM（no Ref.）PSNR 18.40 / WE 0.2170，较PSLD提升0.23 dB，WE降至约1/3
  - 盲超分（DAVIS）：DiffBIR+ZVRM（text+image Ref.）PSNR 24.96 / SSIM 0.6971 / WE 0.4288，最优
- **推理加速**：总采样步数从1000降至360（约1/3），单步时间从1.424s降至1.184s，A40 GPU上总耗时从23min44s降至7min6s
- **消融验证**：DPTI+DPTS+TSTM+TTTM+RSA+RTM完整模块组合达到PSNR 25.49 / SSIM 0.6726 / WE 0.4057

## 相关工作脉络
- **PSLD [10]**：首个基于T2I LDM的零样本图像修复框架，本文以其为backbone，扩展至视频并引入多模态参考
- **VISION-XL [23]**：利用SDXL处理高清视频逆问题，本文在相同backbone下对比，WE指标更优（约一半）
- **DiffIR2VR [21] / ZVRD [20]**：流引导/短长程时间注意力的零样本视频修复，本文在无参考场景下全面超越
- **VidToMe [19] / TokenFlow [18]**：视频编辑中的token合并方法，本文改进为纹理感知版本以适配修复任务
- **Rerender-A-Video [17]**：分层跨帧约束的零样本视频编辑，启发了本文的时间一致性设计思路
- **RefIR [40]**：检索增强图像修复，本文借鉴其参考图像思路，首次引入视频修复场景

## 局限性与未来方向
- **推理速度仍有限**：即使在A40 GPU上仍需约7分钟处理一个视频，采样加速空间较大
- **参考图像质量依赖**：性能受参考图像与目标内容的相似度影响，极端情况下可能引入错误纹理
- **线性退化假设**：修复任务基于线性退化模型$y = Ax + n$，对复杂非线性退化（如运动模糊、压缩伪影）泛化需验证
- **未来方向**：进一步加速采样过程；探索更高效的参考图像检索策略；扩展至更多实际应用场景（监控、摄影）

## 研究启发与可借鉴点
- **多模态参考机制的设计范式**：首次将文本+图像参考统一纳入零样本视频修复框架，可迁移至图像修复、去噪等任务
- **纹理感知自适应合并策略**：根据图像内容复杂度分配不同处理强度，可有效平衡质量与效率，值得推广至其他扩散模型加速方法
- **嵌入联合优化的时序约束设计**：跨帧一致性正则项$||\cdot^i - \cdot^{i-1}||_2^2$简单有效，可复用于其他基于扩散模型的视频生成/编辑任务
- **双阶段提示调优（Inversion+Sampling）思路**：将嵌入优化分为初始化和精细化两阶段，避免单阶段优化的局部最优，启发后续工作改进零样本扩散模型的采样策略
- **3D卷积替换的轻量扩展**：用1×3×3膨胀卷积替换2D卷积以支持视频处理，相比全参数微调节省资源，可作为T2I LDM视频化的通用技巧

## 关键术语表
- **Zero-shot Restoration**：无需针对特定任务训练，利用预训练生成模型直接完成图像/视频修复
- **Latent Diffusion Model (LDM)**：在压缩潜空间中进行扩散过程的生成模型，以Stable Diffusion为代表
- **Dual Prompt Tuning**：同时在inversion和sampling阶段优化条件嵌入和无条件嵌入
- **Token Merging**：将相似visual token合并以降低计算量，常用于扩散模型加速
- **Warping Error (WE)**：衡量视频帧间时间一致性的指标，值越小表示时序越稳定
- **Classifier-Free Guidance**：通过组合条件/无条件预测来放大条件信号的技术，公式为$w \cdot \varepsilon_\theta(z_t, t, C) + (1-w) \cdot \varepsilon_\theta(z_t, t, \emptyset)$
- **Distortion Adaptive Inversion**：以退化程度自适应控制噪声注入强度的inversion策略
- **Multi-modal Reference**：除退化视频外的额外参考信息，包括文本描述和参考图像

## 可复现要素
- **数据集**：REDS4、UDM10、DAVIS、DID（低光）、RealMCVSR——部分公开，DID和RealMCVSR需申请或下载
- **代码**：论文未提及代码开源情况，需关注作者后续发布
- **权重**：基于SDv1.5和SDXL预训练权重（HuggingFace可获取）
- **关键超参**：$T'=15$（inversion步数）、$T_{dpts}=5$（采样阶段开始优化嵌入的步数阈值）、$N=5$（每步优化迭代次数）、$N_c=25$（每25步更新纹理分类）、$w=7.5$（classifier-free guidance权重）、$\gamma_1/\gamma_2$（嵌入一致性正则系数，论文未给出具体值）
- **硬件要求**：实验在48G A40 GPU上进行
