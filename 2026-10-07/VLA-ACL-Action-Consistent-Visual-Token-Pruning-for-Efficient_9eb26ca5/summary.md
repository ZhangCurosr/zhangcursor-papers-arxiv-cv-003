---
title: "VLA-ACL-Action-Consistent-Visual-Token-Pruning-for-Efficient"
source: https://arxiv.org/pdf/2610.08133v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 09:47:16"
---

# 论文速读：VLA-ACL-Action-Consistent-Visual-Token-Pruning-for-Efficient

## 一句话总结
VLA-ACL 提出了一种针对冻结态 VLA 模型的轻量级视觉 Token 剪枝框架，通过在动作级别施加一致性监督（Action Consistency Learning），仅训练一个小型 Patch Scorer 即可实现高达 87.5% 的视觉 Token 剪枝率，在几乎不损失任务成功率的前提下显著降低计算量与推理延迟。

## 研究问题与动机
- **核心问题**：VLA 模型在每一步控制中需处理由相机观测主导的超长视觉 Token 序列，导致计算开销大、延迟高，难以满足实时具身部署需求。
- **训练-free 方法局限**：现有启发式剪枝/缓存方法（如基于注意力分数、运动阈值或时序缓存）仅间接估计 Token 重要性，未直接关联剪枝对下游控制动作的影响；且常依赖逐层/逐时间步的手动调参，与 FlashAttention、torch.compile 等融合加速技术不兼容。
- **训练-based 方法局限**：LightVLA、Grid-S 等需端到端联合微调整个 VLA 主干，训练成本高（多 GPU 数天），且导致剪枝策略与特定 VLA 权重强耦合，部署灵活性差。
- **本文动机**：能否在保持基座 VLA 完全冻结的前提下，从剪枝对下游动作的实际影响中学习有效的视觉 Token 选择策略？

## 核心贡献（创新点）
- **动作一致性学习（ACL）目标**：以冻结 VLA 的全上下文动作输出作为教师信号，结合 Ground Truth 动作辅助监督，直接优化 Patch Scorer，使剪枝后的动作与完整上下文动作保持一致。
- **黑图嵌入可微门控机制**：引入基于 Black-Patch Embedding 的软 Top-K 门控，将低重要性 Token 替换为黑图编码器输出而非零向量或随机噪声，平滑梯度回传至 Patch Scorer，有效缩小训练-推理 gap。
- **冻结基座的高效剪枝框架**：仅训练一个 5 层双向 Transformer 构成的轻量 Patch Scorer，无需修改或微调 VLA 主干，单卡 GPU 数小时即可完成训练，且完全兼容现有推理加速技术。
- **跨架构泛化验证**：同一套超参数与单一 Patch Scorer 成功迁移至 π_0.5 等不同 VLA 架构，在 LIBERO 全 suite 上实现 67% TFLOPs 削减与 2.2× 预填充加速，验证了方法的一般性。

## 方法详解
- **Patch Scorer 结构**：一个 5 层双向 Transformer，接收冻结视觉编码器输出的 $X \in \mathbb{R}^{N_e \times d}$，为每个视觉 Token 生成 logit $l \in \mathbb{R}^{N_e}$。多视角输入时拼接各视角嵌入联合打分，但每视角独立保留 Top-K。
- **推理阶段硬剪枝**：在语言骨干网输入前，按 logit 降序选取 $K$ 个最高分 Token（$\hat{X} = \text{Top-K}(X; l)$），丢弃其余。保留原始 RoPE 位置索引，避免位置关系错乱。
- **训练阶段可微软剪枝**：为突破 Hard Top-K 不可微限制，采用 Soft Top-K 松弛（LapSum 算子），通过拉普拉斯 CDF 确定自适应阈值 $\theta_K$，使 $\sum s_i = K$，输出连续分数 $s_i \in [0,1]$。温度参数 $\alpha$ 从 2.0 余弦衰减至 0.1，逐步逼近硬剪枝。
- **黑图门控公式**：构造替换向量 $\tilde{X}_i = X_i \cdot s_i + b_i \cdot \sqrt{1 - s_i^2}$，其中 $b_i = \mathcal{E}(I_{\mathrm{black}})$ 为黑图经冻结编码器产生的嵌入。该设计使低分 Token 被分布内（in-distribution）特征平滑替代，保障梯度有效传播。
- **动作一致性损失**：$\mathcal{L}_{\mathrm{acl}} = \lambda \| \tilde{\boldsymbol{a}} - \mathbf{a} \|_1 + \mu \| \tilde{\boldsymbol{a}} - \mathbf{a}_{\mathrm{gt}} \|_1$。其中 $\tilde{\boldsymbol{a}}$ 为软剪枝上下文经冻结 LLM 输出的动作，$\mathbf{a}$ 为全上下文教师动作，$\mathbf{a}_{\mathrm{gt}}$ 为演示 GT 动作。$\lambda=\mu=0.5$ 等权，梯度仅更新 Patch Scorer，VLA 主干完全冻结。

## 实验与结果
- **数据集**：LIBERO（Spatial, Object, Goal, Long 四大 suite）及真实世界 ALOHA-style 双臂操作平台（AgileX PiPER-X，含 4 项抓取/放置任务）。
- **基线模型**：OpenVLA-OFT（主实验）、π_0.5（泛化实验）；对比基线包括 VLA-Cache、SAFE-Pruner、SpecPrune-VLA、VLA-Pruner、VLA-ADP 等冻结态剪枝/缓存方法。
- **LIBERO 仿真结果（OpenVLA-OFT 基座）**：
  - K=32（剪枝 87.5%）：FLOPs 降低 75%（4.013→0.991 TFLOPs），延迟降至 42.30 ms（基准 63.49 ms），速度提升 1.5×，平均成功率 95.4%（基准 96.8%，-1.4%）。
  - K=64（剪枝 75%）：FLOPs 降低 65%，速度提升 1.4×，平均成功率 97.0%（略高于基准 +0.2%），Long 任务与基准持平（94.2%）。
  - VLA-ACL 在同等 Pruning Rate 下 Action Loss 最低，且全程保持恒定剪枝率，无需时序控制器或首帧全量预填。
- **真实世界结果**：K=64 下平均成功率 72.5%（基准 73.75%），FLOPs 降低超 50%（6.257→2.953 TFLOPs），延迟降至 62.1 ms（基准 91.5
