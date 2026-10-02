---
title: "INLINE-MEMORY-MEETS-REUSABLE-SKILLS-MEMORY-CENTRIC-FRAMEWORK"
source: https://arxiv.org/pdf/2609.39794v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 01:44:15"
---

# 论文速读：INLINE-MEMORY-MEETS-REUSABLE-SKILLS-MEMORY-CENTRIC-FRAMEWORK

## 一句话总结
提出 Optimus-R，一种以记忆为核心的视觉-语言-动作（VLA）适配框架，将机器人技能获取与跨域迁移显式化为查询-技能记忆调优；通过内联记忆接口、可复用记忆库与轻量桥接适配机制，在低数据与持续学习场景下显著提升样本效率并有效缓解灾难性遗忘。

## 研究问题与动机
1. 现有 VLA 模型适配新任务依赖参数微调，每次新技能都会隐式修改全局权重，导致适配成本高且易干扰已学行为。
2. 传统外部记忆检索方法仅作为松散上下文，与 VLA 的 action-conditioning 路径耦合薄弱，难以编码与运动控制直接相关的表征。
3. 参数化技能存储不具备可检索性与可扩展性，持续学习中新旧技能相互干扰，引发灾难性遗忘。
4. 域偏移（如 sim-to-real）会使目标域观测偏离源域记忆空间，导致查询与存储技能匹配错位，传统微调难以快速对齐。

## 核心贡献（创新点）
1. 提出记忆中心化的 VLA 适配范式，将技能获取与更新转化为显式查询-技能记忆调优，大幅降低对全局参数微调的依赖；与参数中心化微调不同，新技能以可检索原型外部存储，避免权重覆盖。
2. 设计内联记忆接口，将可学习记忆 token 插入 VLA prefix stream，通过双分支注意力池化同步生成控制感知的查询向量与技能向量；与 detached 上下文记忆不同，该接口在策略原生前向路径中直接提取可执行表征。
3. 构建查询-技能记忆库，将技能外化为解耦的查询原型与技能值对，支持基于运动感知采样与方差约束聚类的原型构建、残差更新与按需扩展；与 episodic 记忆仅缓存轨迹不同，该库维护语义-行为对齐的可复用条目。
4. 开发 Bridge-and-Adapt 三阶段训练机制，通过线性查询对齐矩阵与可选低秩 Skill Adapter 实现跨域表征匹配，仅调整局部记忆坐标；与直接 fine-tune backbone 相比，该方法在保留基础原型的同时完成目标域迁移。

## 方法详解
- **内联记忆接口**：给定观测 $O_t$ 与语言指令 $L$，将 $m$ 个可学习记忆 token $E^{\mathrm{mem}}$ 拼接至 VLA prefix 输入骨干网络 $F_\theta$，得到多模态状态 $H_t^{\mathrm{vl}}$ 与记忆状态 $M_t$。通过两个独立注意力池化器 $\mathrm{AttnPool}_q$ 与 $\mathrm{AttnPool}_s$ 分别提取查询与技能汇总表征，经 Query Encoder 与 Skill Encoder 得到 $q_t \in \mathbb{R}^{d_q}$ 和 $s_t \in \mathbb{R}^{d_s}$。训练阶段使用轻量 Skill Decoder 以 Huber 损失监督 $s_t$ 重建未来动作 chunk，使技能空间与可执行行为对齐；Token Projector 将技能映射为 VLA 兼容的 skill tokens 用于策略 conditioning。
- **查询-技能记忆库**：初始库由 Stage A 提取的 $(q_t, s_t)$ 对构建，先按动作变化分数 $\nu_i$ 进行运动感知降采样，再按归一化轨迹进度分层采样以避免单阶段主导。随后在查询空间聚类，对技能方差超阈值的子簇进行二次分割，生成查询原型 $p_j^q$ 与技能原型 $p_j^s$。推理时通过线性对齐 $q'_t = A_{\mathrm{mem}} q_t$ 计算与 Top-$K_r$ 原型的相似度，温度加权聚合得到检索技能 $\hat{s}_t$；当技能空间错位较大时，经零初始化的低秩 Skill Adapter $A_s(\hat{s}_t) = \hat{s}_t + U_s V_s^\top \hat{s}_t$ 进行校正。
- **三阶段训练机制**：
  - **Stage A（接口预训练）**：联合优化记忆 token、双池化器、双编码器、Skill Decoder、Token
