---
title: "The-Blind-Spot-in-2D-Infants-Pose-Estimation-Robust-Learning"
source: https://arxiv.org/pdf/2609.04009v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 15:35:54"
---

# 论文速读：The-Blind-Spot-in-2D-Infants-Pose-Estimation-Robust-Learning

## 一句话总结
针对早产儿2D姿态估计中标注噪声严重且现有鲁棒学习方法不适配的问题，本文提出REMIND策略，通过建模关键点级别的训练动态轨迹（损失下降幅度与收敛时序），在无先验噪声分布假设下自动聚类识别噪声标注，将含噪训练的mAP性能衰减从22.5%压缩至1.8%，实现了免阈值调参的鲁棒姿态学习。

## 研究问题与动机
1. **临床标注质量先天不足**：早产儿姿态估计用于神经发育评估，但图像常伴随肢体自遮挡、医疗设备干扰及护理人员手部遮挡，人工关键点标注 inherently prone to error，导致监督信号严重污染。
2. **现有噪声学习方法的领域错位**：深度学习的标签噪声研究主要集中于图像分类，面向关键点回归（PE）的工作极少；唯一相关方法ScarceNet依赖预设阈值的小损失（Small-Loss）过滤，阈值需根据假设噪声比例人工调节，临床实用性受限。
3. **瞬时
