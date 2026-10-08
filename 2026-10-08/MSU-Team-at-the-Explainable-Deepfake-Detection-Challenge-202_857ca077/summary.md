---
title: "MSU-Team-at-the-Explainable-Deepfake-Detection-Challenge-202"
source: https://arxiv.org/pdf/2610.09952v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 09:49:17"
---

# 论文速读：MSU-Team-at-the-Explainable-Deepfake-Detection-Challenge-2026

## 一句话总结
本文针对可解释深度伪造检测挑战赛，提出模块化“检测+解释”框架：通过 Grounding-DINO 将训练解释文本转化为弱补丁级伪掩码，结合 DINOv3/Mesorch 多骨干特征与局部对比学习提升取证检测精度，并采用类条件 Qwen3-VL 与 GRPO 优化生成复杂/简单双级解释，在 XPlainVerse 测试集上取得 0.9349 检测准确率与 0.7456 综合挑战分数。

## 研究问题与动机
- 现代生成模型产出的篡改图像高度逼真，传统检测器仅输出二分类标签，无法满足审计、溯源与公众科普对视觉/文本证据的需求。
- XPlainVerse 挑战赛要求系统同时完成真假判定、生成与技术向/大众向两类解释，并依据实体对齐与视觉证据 grounding 得分评判，单一分类模型难以直接适配。
- 像素级伪造掩码标注成本极高，且实际篡改往往仅涉及局部区域，需探索弱监督或自监督的定位线索获取路径。
- 直接将大 VLM 用于检测易引发类别泄露与真图幻觉伪影，将检测决策与解释生成解耦并按预测类别路由成为提升证据可信度的可行方向。

## 核心贡献（创新点）
- **文本解释驱动的伪掩码生成流水线**：利用 Qwen3-VL 提取伪影短语并按局/全局划分，经 Grounding DINO 框选合并为补丁覆盖比例目标，使检测器可在无像素标注条件下学习文本-视觉对齐的取证区域。
- **类条件双分支解释生成模块**：依据检测器预测结果分别调用 fake/real 专属 Qwen3-VL-8B 生成复杂解释，阻断跨类别幻觉；再通过纯文本简化器与 GRPO 直接优化官方简易解释指标。
- **置信度状态驱动的局部补丁对比学习（LCL）**：在无需配对图像或像素级掩码的情况下，依据当前补丁预测概率划分一致/不确定/矛盾状态，设计拉紧/推开规则矩阵并组织 stop-gradient 防干扰，显著改善补丁嵌入空间的可分性。
- **多骨干统一取证特征架构**：融合 Transformer/CNN 双系 DINOv3 与 Mesorch 频域特征，构建 UFFM 并通过最大池化将局部证据放大后参与全局分类，兼顾细节敏感性与鲁棒性。

## 方法详解
- **伪掩码构建**：对假图解释 $e_i$ 使用 Qwen3-VL-32B 提取并聚类得词汇库 $\mathcal{V}=\mathcal{V}_{\text{loc}}\cup\mathcal{V}_{\text{glob}}$，仅对 $\mathcal{V}_{\text{loc}}$ 执行图像级 Open-vocabulary Grounding。保留置信度阈值以上的框集合 $\mathcal{B}_i$，取并集 $U_i$ 覆盖补丁网格，计算 $t_{iuv}=|R_{uv}\cap U_i|/|R_{uv}|$ 作为软目标 $T_i\in[0,1]^{H_p\times W_p}$；真图设 $T_i\equiv 0$。
- **多骨干特征提取与融合**：各 DINOv3 骨干输出全局嵌入 $f_i^{(k)}$ 与稠密图 $H_i^{(k)}$，经投影与 MLP 拼接触发 $H_i^{\text{DINO}}$ 与 $f_i^{\text{DINO}}$；Mesorch 特征对齐至相同网格得 $H_i^{\text{Mes}}$。两者拼接过 $F_{\text{UFFM}}$ 得到 $U_i\in\mathbb{R}^{H_p\times W_p\times D_U}$。
- **图像级分类分支**：对 $U_i$ 做 max pooling 得 $u_i^{\text{pool}}$，与 $f_i^{\text{DINO}}$ 拼接后经 $F_{\text{img}}$ 与二元分类器输出 $z_i$，采用 Focal Loss（$\alpha=0.5,\gamma=2.0$）训练。
- **伪影证据图分支**：共享补丁头预测 logits $a_i[u,v]$ 与概率 $q_i[u,v]$，以 BCE 配合 $T_i$ 监督；真实图像目标恒为 0。
- **局部对比损失（LCL）**：批次内假-假、真-真、假-真配对时，按余弦相似度双向匹配补丁。依据 $q_i$ 距 0.5 的距离引入不确定性带 $\delta$，将补丁划分为 CF/UF/WF 与 CR/UR/WR 状态；对应关系决定执行 $\ell_{\text{pull}}=1-\cos(a,b)$、$\ell_{\text{push}}
