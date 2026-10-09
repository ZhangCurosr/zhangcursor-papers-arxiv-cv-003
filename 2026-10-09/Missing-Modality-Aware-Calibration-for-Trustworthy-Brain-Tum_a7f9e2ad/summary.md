---
title: "Missing-Modality-Aware-Calibration-for-Trustworthy-Brain-Tum"
source: https://arxiv.org/pdf/2610.11419v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 10:06:39"
---

# 论文速读：Missing-Modality-Aware-Calibration-for-Trustworthy-Brain-Tum

## 一句话总结
本文提出了MMA-LTS，一种用于模态缺失条件下脑肿瘤MRI分割的后验体素级置信度校准方法。该方法通过条件化模态可用性与逐体素难度，修正组合特异性与空间异质性的校准误差，在保持SOTA分割精度的同时显著提升预测可靠性。

## 研究问题与动机
1. **临床模态缺失普遍但可靠性被忽视**：多模态脑肿瘤分割严重依赖FLAIR/T1/T1ce/T2等多序列MRI，但临床实践中常因协议差异或扫描故障导致部分模态缺失；现有方法多聚焦分割精度，忽略了缺失模态下的预测置信度可靠性。
2. **现有校准技术的核心假设不成立**：主流校准方法（联合预测-校准或后验概率修正）通常假设输入完整固定，或认为“模态越多预测越容易、置信度单调上升”；但脑肿瘤中不同模态贡献不均（如T1ce对增强肿瘤更关键），导致预测难度取决于“缺失哪些模态”而非“缺失多少”，且误差在空间上高度异质。
3. **
