---
title: "Training-Free-Pseudo-Fusion-for-Composed-Image-Retrieval-wit"
source: https://arxiv.org/pdf/2608.23102v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 00:52:22"
field: "多模态检索与生成"
keywords: ["Composed Image Retrieval", "Zero-shot Retrieval", "Diffusion Models", "Multimodal Large Language Models", "Training-free", "Cross-modal"]
innovations: ["提出无需训练的伪融合框架PeFuse，将CIR转换为单模态检索任务", "系统性地比较单向/双向四种转换策略（T→I, I→I, T→T, I→T）", "揭示T→I转换效果最优，扩散伪像损害检索性能的规律"]
benchmarks: ["Fashion-IQ", "CIRR", "CIRCO", "GeneCIS"]
---

# 论文速读：Training-Free Pseudo-Fusion for Composed Image Retrieval with Diffusion Models and Multimodal Large Language Models

## 一句话总结
本文提出 PeFuse，一种无需训练的伪融合框架，利用预训练扩散模型和 MLLM 将组合图像检索（CIR）中的多模态查询转换为单模态表示，实现零样本 CIR。实验表明将 CIR 重构为文本到图像检索任务最为有效。

## 研究问题与动机
- **现有方法依赖训练**：传统 CIR 方法依赖大规模标注数据训练，泛化能力受限，难以适应领域迁移。
- **跨模态对齐挑战**：如何将图像参考与文本修改的有效组合语义传递给检索系统是一个核心难题。
- **零样本需求**：现有训练方法需要合成三元组或大规模 image-text 数据，缺乏灵活性和适应性。
- **成本与效率**：多模型级联方案（如 ImageScope）导致累积计算开销和推理时间增加。

## 核心贡献（创新点）
1. **提出 PeFuse 训练自由伪融合策略**：通过单向/双向转换将 CIR 转换为单模态检索任务，无需任何任务特定微调。
2. **系统性地探索四种转换范式**：首次全面基准测试基于扩散模型和 MLLM 的 T→I、I→I、T→T、I→T 四种转换策略。
3. **揭示转换路径有效性差异**：发现文本到图像转换（PeFuse T→I）效果最佳，扩散生成图像的伪像会损害检索性能。
4. **超参数敏感性与缩放分析**：量化了 MLLM 和扩散模型超参数对性能的影响，并分析了模型规模扩展规律。

## 方法详解
**核心思想**：将多模态组合查询 $(I_{ref}, T)$ 通过生成模型转换为单模态表示，再利用预训练检索模型进行相似度匹配。

**两种转换策略**：

**单向转换（Uni-directional Conversion）**：
- **MLLM 路径**：使用 MLLM $h(\cdot)$ 生成描述文本 $T_h = h(I_{ref}^i, T_i, p_d)$，将 CIR 转化为文本到图像检索（T→I）
- **扩散模型路径**：使用扩散模型 $g(\cdot)$ 生成目标图像 $I_g = g(I_{ref}^i, T_h^i)$，将 CIR 转化为图像到图像检索（I→I）

**双向转换（Bi-directional Conversion）**：
- 在单向基础上，额外使用 MLLM 将候选图像转换为文本 $T_h^j = h(I_{tar}^j, q_d)$
- 实现文本到文本检索（T→T）和图像到文本检索（I→T）

**相似度计算**：
$$s_{ij} = \cos(\Psi(q_i), \Psi(I_{tar}^j)) = \frac{\Psi(q_i)^\top \Psi(I_{tar}^j)}{\|\Psi(q_i)\|_2 \|\Psi(I_{tar}^j)\|_2}$$
其中 $\Psi(\cdot)$ 为预训练检索模型（如 CLIP、OpenCLIP、SigLIP2）。

**算法流程**（Algorithm 1）：根据选择的检索模式（m ∈ {T→I, T→T, I→I, I→T}），先生成查询表示，再对候选池计算余弦相似度并排序返回 top-k。

## 实验与结果
**数据集**：Fashion-IQ、CIRR、CIRCO、GeneCIS（均为标准 CIR 基准）

**关键模型**：
- 检索模型：CLIP (ViT-B/32)、OpenCLIP (ViT-B/32)、SigLIP2 (ViT-B-16)
- 生成模型：Qwen2.5-VL-7B-Instruct（MLLM）、SDXL-InstructPix2Pix（扩散模型）

**主要结果**：

| 数据集 | 最优方法 | 关键指标 | 提升幅度 |
|--------|----------|----------|----------|
| Fashion-IQ | PeFuse (T→I) + OpenCLIP | R@10=28.27, R@50=48.08 | 超越 CIReVL/LDRE 等零样本方法 |
| CIRR | PeFuse (T→I) + OpenCLIP | R@1=34.00, R@10=77.47 | 接近 SOTA 训练方法 |
| CIRCO | PeFuse (T→I) + SigLIP2 | mAP@50=22.62 | 显著优于多数基线 |
| GeneCIS | PeFuse (T→I) + SigLIP2 | Avg R@1=17.37 | 各子任务均表现稳定 |

**重要发现**：
- T→I 转换在所有数据集上普遍最优，I→I 转换效果最差
- SigLIP2 在 T→I 任务上表现弱，但在 I→I 任务上最强，体现模型能力差异
- OpenCLIP 作为检索模型整体表现最佳
- 模型规模扩展（ViT-L→ViT-H→ViT-bigG）带来性能提升，但存在波动

## 相关工作脉络
1. **CIReVL**：使用 LLM 细化 VLM 生成的图像描述进行 T2I 检索，PeFuse 直接使用 MLLM 端到端生成更简洁
2. **ImageScope**：多阶段 MLLM 级联生成描述，计算开销大；PeFuse 仅用单个 MLLM 或扩散模型
3. **LDRE**：利用 LLM 生成多个多样化描述并加权，PeFuse 可与其结合但更轻量化
4. **WeiMoCIR**：用 MLLM 为候选图生成描述，PeFuse 聚焦于查询侧的转换
5. **CLIP4CIR**：引入可学习融合算子捕获组合语义，需要训练；PeFuse 无需训练
6. **Textual Inversion 系列**（SEARLE、Pic2Word、LinCIR）：将图像映射为 token，需训练；PeFuse 直接生成自然语言或图像

## 局限性与未来方向
**局限性**：
- 多模型级联可能导致概念漂移和误差传播
- 用户原始文本修改直接输入 MLLM 可能影响效果
- 扩散模型生成的图像含有伪像，损害检索性能
- 超参数调优劳动密集且难以泛化到不同数据集
- 推理延迟较高，不适合实时应用

**未来方向**：
- 开发鲁棒的集成技术以减少多模型流水线的误差传播
- 探索迭代多轮融合以逐步精炼描述
- 结合复杂场景：多图输入、长文本叙述、视频等多模态
- 开发更轻量级模型以实现可扩展部署

## 研究启发与可借鉴点
1. **查询侧生成优于候选侧生成**：将复杂多模态查询转换为高质量单模态表示（尤其是文本描述），比生成目标图像更有效
2. **提示工程的关键作用**：数据集特定的 prompt 设计能显著提升 MLLM 生成质量，值得深入探索
3. **模型选择敏感性**：检索模型的能力差异（CLIP vs OpenCLIP vs SigLIP2）对最终性能影响巨大，需仔细匹配任务
4. **扩散模型辅助价值**：即使 I→I 效果不佳，MLLM 生成的描述作为扩散模型提示仍能提升图像质量（Figure 4）
5. **解耦生成与检索**：离线预处理候选图像索引，在线仅处理查询，可实现高效部署（Section 6）

## 关键术语表
- **Composed Image Retrieval (CIR)**：组合图像检索，用户通过参考图像+文本修改来检索相似目标图像
- **Pseudo-Fusion（伪融合）**：通过生成模型隐式融合多模态信息，而非显式拼接特征向量
- **Uni-directional Conversion（单向转换）**：仅将查询转换为单模态，保持候选图像不变
- **Bi-directional Conversion（双向转换）**：查询和候选图像均转换为同一模态进行匹配
- **Text-to-Image Retrieval (T→I)**：将 CIR 重构为文本查询检索图像的任务
- **Image-to-Image Retrieval (I→I)**：将 CIR 重构为图像查询检索图像的任务
- **Zero-Shot CIR (ZS-CIR)**：零样本组合图像检索，无需任务特定训练

## 可复现要素
- **数据集**：Fashion-IQ、CIRR、CIRCO、GeneCIS（公开可用）
- **代码开源**：是，https://github.com/StevenXuf/PeFuse4CIR
- **关键超参数**：
  - 扩散模型：guidance scale=7.5，image guidance scale=3.0，denoising steps=30
  - MLLM：temperature=0.1，top-p=0.9，top-k=50，max tokens=128
- **检索模型输入分辨率**：224×224（CLIP/OpenCLIP），768×768（扩散模型处理）
