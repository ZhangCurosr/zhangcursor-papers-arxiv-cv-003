---
title: "VLA-ACL-Action-Consistent-Visual-Token-Pruning-for-Efficient"
source: https://arxiv.org/pdf/2610.08133v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-07 10:45:14"
---

# 论文速读：VLA-ACL-Action-Consistent-Visual-Token-Pruning-for-Efficient

## 一句话总结
本文提出 VLA-ACL，一种针对视觉-语言-动作（VLA）模型的视觉 token 剪枝框架；在完全冻结基座 VLA 的前提下，通过动作级一致性学习训练轻量级 patch 评分器，实现最高 87.5% 的视觉 token 裁剪，将计算量降低 75% 并带来约 1.5× 推理加速，在 LIBERO 仿真与真实机器人任务上均取得优于现有冻结型剪枝/缓存基线的方法-效率权衡。

## 研究问题与动机
- VLA 模型在每一步控制步均需处理由相机观测主导的超长多模态序列，视觉 patch 占据序列主体且冗余度高，直接导致高昂的计算开销与延迟，阻碍实时部署。
- 现有免训练（training-free）剪枝方法依赖注意力分数、时序运动阈值等间接启发式信号，信号与下游控制性能脱钩，且需逐层/逐时步精细调参，常因访问内部状态而与 FlashAttention、图编译等加速技术不兼容。
- 现有基于训练（training-based）的方法需联合微调完整 VLA 骨干，计算成本极高（多卡数天），且导致模型 checkpoint 与剪枝机制强耦合，丧失跨场景部署灵活性。
- 核心科学问题：能否让 token 选择决策直接受下游动作预测效果驱动，在保持基座 VLA 完全冻结的同时，学习到高效、静态且通用的视觉 token 筛选策略？

## 核心贡献（创新点）
- **动作一致性学习（Action Consistency Learning）框架**：构建以冻结全上下文 VLA 输出为教师信号、地面真实动作为辅助监督的损失函数，将 token 选择目标直接绑定至最终控制行为，彻底摆脱对注意力等中间代理信号的依赖。
- **可微软剪枝门控机制**：设计基于 soft Top-K 松弛与黑色图像嵌入（black-patch embedding）的可微门控模块，使梯度能穿透冻结 LLM 反传至轻量评分器，并通过余弦调度逐渐锐化
