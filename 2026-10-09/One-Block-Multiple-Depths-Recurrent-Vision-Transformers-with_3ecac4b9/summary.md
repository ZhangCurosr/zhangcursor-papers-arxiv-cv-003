---
title: "One-Block-Multiple-Depths-Recurrent-Vision-Transformers-with"
source: https://arxiv.org/pdf/2610.12448v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 10:07:47"
field: "视觉Transformer高效架构"
keywords: ["Vision Transformer", "Recurrent Network", "Mixture of Experts", "Weight Sharing", "Elastic Depth", "Model Compression"]
innovations: ["用单个循环Transformer块+深度程序化专家替代全深度堆叠，73%参数减少匹配DeiT III精度", "归一化深度坐标作为唯一路由信号实现弹性深度推理", "在统一测试床下对比七种MoE机制，证明权重空间合并是one-FFN预算下的最优方案"]
benchmarks: ["ImageNet-1k", "ADE20k", "NYUv2", "DINOv2-B/14 distillation"]
---

# 论文速读：One-Block-Multiple-Depths-Recurrent-Vision-Transformers-with

## 一句话总结
本文提出 reViT，将 ViT 的深度堆叠替换为单个 Transformer 块的循环复用，通过归一化深度坐标程序化混合专家 FFN 银行，在相同推理 FLOPs 下匹配全深度编码器精度，同时将存储参数减少约 70%，并支持弹性深度推理。

## 研究问题与动机
1. **核心问题**：标准 ViT 使用 L 个独立参数块堆叠，能否用一个共享的 Transformer 块循环 L 次来替代，同时保持深度相关的计算能力？
2. **ViT 层间相似性过高**：相比 CNN，ViT 层间表征相似度更高（Raghu et al., 2021），表明直接权重共享会损失表达能力。
3. **已有工作不足**：Raptor（Jacobs et al., 2026）使用 k 个循环块模板（k=4），仍需中间特征蒸馏且性能仍低于原始网络，未探索完全单块共享的情形。
4. **MoE 适配空白**：现有 MoE 变体主要在 token 层面做路由，缺乏在"单块全深度复用"设定下对 FFN 参数空间连续轨迹的有效探索。

## 核心贡献（创新点）
1. **连续深度程序的循环 ViT**：提出 reViT，用一个循环 Transformer 模块替代深度堆叠，通过归一化深度坐标在共享 FFN 专家银行上形成连续参数轨迹；与已有方法本质区别在于路由信号是"深度坐标"而非"图像内容"，实现存储容量与执行深度的解耦。
2. **统一测试床下的 MoE 机制对比**：在相同循环骨干、任务和训练配方下，系统比较了七种 MoE 方案（含权重合并、token 分发、输出混合），证明权重空间合并是匹配 one-FFN 预算下的最优方案；与其他工作本质区别在于控制了计算预算和骨干一致性，而非在不同架构中横向比较。
3. **弹性深度训练与固定深度导出**：同一 checkpoint 可通过重新采样归一化坐标区间支持多种推理深度；固定部署时可将深度特定 FFN 折叠为传统稠密图；与其他方法本质区别在于"单次训练 → 多深度复用"能力，无需为每个深度单独训练。
4. **深度程序可解释性分析**：揭示了路由如何学习主导专家切换点和过渡混合，且 gate-expert 分配本身对教师对齐至关重要；与 Raptor 等方法的本质区别在于仅用最终层蒸馏就涌现出了类 Raptor 的深度分段结构。

## 方法详解
1. **循环块结构**：将标准 ViT 的 L 个独立块替换为共享 attention、LayerNorm 和路由器，仅 FFN 部分在每步 t 通过归一化深度坐标 $s_t^{(L)} = t/(L-1)$ 程序化生成。
2. **深度程序路由器**：路由器 $\psi$ 为两层 MLP（16 隐藏单元，SiLU 激活），将标量深度坐标映射到 E 维 logit，经 softmax 得到门控权重 $g_t$，τ=1；输出层零初始化使所有专家初始权重相等。
3. **权重空间合并（Weight-space Merging）**：在 FFN 执行前，将 E 个专家 FFN 的投影矩阵和偏置做加权和：$\bar{\theta}_t^{\text{fin}} = \sum_e g_{t,e} \theta_e^{\text{fin}}$，得到单一稠密 FFN 应用于所有 token；仅评估合并后的 FFN，不单独执行各专家。
4. **循环步骤公式**：$H^t = X^t + \text{MHSA}(\text{LN}_{\text{attn}}(X^t))$，$X^{t+1} = H^t + \text{FFN}_t(\text{LN}_{\text{fn}}(H^t))$，所有步共享 attention 和 norm，仅 FFN 随深度变化。
5. **辅助正则化损失**（E>1 时）：
   - **使用平衡损失** $\mathcal{L}_{\text{bal}} = \frac{1}{L}\sum_t [1 - H(g_t)/\log E]$ 鼓励软混合；
   - **Router z-loss** $\mathcal{L}_z$ 控制 logit 尺度；
   - **专家多样性损失** $\mathcal{L}_{\text{div}}$ 惩罚专家参数间的余弦相似度。
6. **弹性深度训练**：使用 per-image DropPath 随机丢弃完整循环步（作为恒等映射），保留步重新编号并从 0 到 1 重新均匀分布坐标，暴露路由器到多种有效深度。
7. **弹性深度推理**：推理时选定全局深度 L，重新计算归一化深度网格；同一模型可在不同 L 下运行而无需重新训练。
8. **蒸馏设置**：仅使用教师最终层特征进行 L2 蒸馏 $\mathcal{L}_{\text{dist}} = \frac{1}{Td}\|\text{LN}(X^L) - Y\|_F^2$，不使用中间特征蒸馏。

## 实验与结果
1. **监督 ImageNet-1k 训练**（Table 1）：
   - reViT-B/16（E=4）在 ~34 GFLOPs 下达到 **83.0% Top-1**，超越 DeiT III-B/16（82.8%），存储参数 **23.6M vs 86.6M（减少 73%）**。
   - reViT-L/16（E=4）达到 **83.8% vs 84.1%**，参数 **40.9M vs 304.4M（约 7× 更少）**。
   - 超越 Raptor（k=4）0.1–0.2 个百分点在所有尺度上。

2. **DINOv2 蒸馏**（Table 2）：
   - reViT-B/14（E=8）ImageNet 线性探针达到 **83.9%**，接近教师 DINOv2-B/14 的 84.5%。
   - ADE20k mIoU 达到 **45.8%**（教师 47.5%），NYUv2 线性 RMSE **0.554m**、MLP RMSE **0.483m**。

3. **MoE 机制对比**（Table 3 & 4）：
   - 在 U=1（单 dense FFN 预算）下，**reViT 权重合并以 78.2% 居首**，显著优于 token 分发（最高 70.6%）和输出混合（最高 70.1%）。
   - SMEAR 在蒸馏中 MLP-head RMSE 0.485 略优于 reViT 的 0.491，但其余指标 reViT 占优。
   - 扩大预算至 U=4 时，MoEUT（77.6%）和 ReMoE（83.2% on distillation）追平或超越，但改变了专家数量和宽度。

4. **弹性深度**（Fig. 2）：
   - 深度程序门控在 L=8/12/16/24 均保持良好性能；ADE20k 从 L=8 的 40.3 提升至 L=16 的 44.9，而特征-only 门控在 L>12 后基本停滞甚至在 NYUv2 上恶化。

5. **专家数量影响**（Table 5）：
   - S/16：E=1→E=8 提升 **+9.6 点**（70.6→80.2%）；B/16：+6.8 点；L/16：+5.7 点。最大提升均出现在 E=1→E=2。

6. **推理效率**（Table 7）：
   - 动态 reViT-S/16 部署图小 **3.6×**，batch=1 时内存降低 **48%**，latency 增加 13%；batch=64 时 latency 增加 39%。

## 相关工作脉络
1. **Universal Transformer（Dehghani et al., 2019）**：显式循环共享单块；reViT 在此基础上引入深度程序化专家路由，实现深度相关计算而非全等权重重复。
2. **ALBERT（Lan et al., 2020）/ MiniViT（Zhang et al., 2022）**：层间权重共享用于压缩；reViT 通过专家混合恢复被共享损失的表达能力，而非简单复制。
3. **Raptor（Jacobs et al., 2026）**：用 k=4 个循环块模板近似全深度 ViT，仍需中间特征蒸馏；reViT 仅用 1 个块且仅依赖最终层蒸馏即达到接近性能。
4. **MoEUT（Csordás et al., 2024）/ DeepSeek-V3 / Qwen3**：token 级别稀疏路由；reViT 在深度维度做全局混合，路由信号是归一化坐标而非 token 内容。
5. **Lory（Zhong et al., 2024）/ SMEAR（Muqeeth et al., 2024）**：图像特征条件化参数合成；reViT 以归一化深度坐标替代图像特征作为路由信号，支持弹性深度。
6. **CondConv（Yang et al., 2019）/ FiLM（Perez et al., 2018）**：条件化卷积/Film 调制；reViT 扩展到完整 FFN 参数空间的凸组合而非仅激活调制。

## 局限性与未来方向
1. **弹性深度的推理延迟开销**：动态路由和合并带来额外计算（论文报告 batch=64 时 latency 增加 39%），需定制 kernel 优化。
2. **E=1 时性能缺口仍显著**：无专家时 S/16 仅 70.6%（vs DeiT III 79.9%），说明完全共享仍有表达瓶颈，需更多专家或正则化改进。
3. **蒸馏仅用最终层特征**：虽避免了 Raptor 的中间特征依赖，但可能限制了更小模型的蒸馏效率；探索是否有轻量级中间监督可进一步提升。
4. **有效秩分析显示 L/16 利用率低**：更大模型从增加专家中获益更少（S/16 +9.6 vs L/16 +5.7），可能受限于骨干表征空间的低秩特性。
5. **未探索跨任务通用性**：当前仅在图像分类、分割、深度预测上验证，对生成任务、多模态任务的有效性尚待检验。

## 研究启发与可借鉴点
1. **深度程序化路由范式可迁移**：将"归一化深度坐标 → 专家混合"的设计可用于其他循环架构（如语言模型 Universal Transformer、状态空间模型），为参数高效循环网络提供新思路。
2. **弹性深度训练技巧**：per-image DropPath 配合坐标重分布实现弹性深度，该方法无需额外调度器即可训练多深度兼容模型，可借鉴到 recurrent Mamba/SSM 等架构中。
3. **MoE 机制统一对比基准**：在固定骨干和任务下系统对比七种 MoE 方案的方法论，为后续工作提供了可复用的实验协议，建议团队沿此框架扩展比较新提出的 MoE 变体。
4. **专家分配敏感性分析**：通过 CKA 消融和 gate 列置换验证路由-专家配对的重要性，这种分析方法可用于诊断其他共享权重架构中的表征退化问题。
5. **参数量压缩潜力**：73% 参数减少的同时保持甚至超越全深度模型，对边缘设备部署 ViT 具有重要参考价值，可结合 quantization/pruning 进一步压缩。

## 关键术语表
**reViT**：本文提出的循环 Vision Transformer，用一个共享块循环 L 次并用深度程序化专家替代独立 FFN。

**Depth-programmed Experts**：以归一化深度坐标为唯一路由信号，在共享专家 FFN 银行上进行凸组合，生成每步专用的稠密 FFN。

**Elastic-depth Training/Inference**：通过循环深度 Dropout 训练，使同一 checkpoint 可在不同推理深度（L 值）下通过重新采样深度坐标正常运行。

**Weight-space Merging**：在参数空间（而非 token 空间或输出空间）对专家 FFN 权重做加权和，每步只评估一个稠密 FFN，保持 U=1 计算预算。

**Normalized Depth Coordinate**：$s_t^{(L)} = t/(L-1)$，将循环步 t 映射到 [0,1] 区间，使路由函数与具体深度 L 解耦。

**Effective Rank**：基于奇异值熵计算的矩阵有效秩，用于衡量专家银行、门控矩阵和合并权重的多样性程度。

**CKA（Centered Kernel Alignment）**：线性核对齐度量，用于量化学生模型各步特征与教师模型各层特征的相似性。

**MoE Formulation（U 值）**：名义 FFN 计算预算，U=1 对应每步一个标准稠密 FFN 的 FLOPs，用于公平比较不同 MoE 机制。

## 可复现要素
- **数据集**：ImageNet-1k（监督训练）、DINOv2-B/14 蒸馏、ADE20k（分割）、NYUv2（深度预测）；均为公开数据集。
- **代码开源**：论文未明确声明代码开源，但提到所有模型在 PyTorch 中实现。
- **超参数**：
  - 路由器：两层 MLP，16 隐藏单元，SiLU 激活，τ=1，输出层零初始化
  - 辅助损失系数：λ_bal=10⁻²，λ_z=10⁻³，λ_div=10⁻³
  - 优化器：AdamW，峰值学习率 S/B 为 4×10⁻³/L 为 3×10⁻³
  - 训练轮数：300 epochs，warmup 5 epochs
  - Batch size：2048
  - Elastic-depth dropout rate：0.3
- **硬件**：8× NVIDIA H100 GPU，每模型训练约 1-2 天（192-384 H100 GPU-hours）
