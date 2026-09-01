---
title: "ScenePilot-Grow-and-Repair-Policy-for-Text-Driven-3D-Indoor"
source: https://arxiv.org/pdf/2608.30307v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 16:37:45"
---

# 论文速读：ScenePilot-Grow-and-Repair-Policy-for-Text-Driven-3D-Indoor

## 一句话总结
提出 ScenePilot，一种检索增强的“生长-修复”框架，将文本驱动的 3D 室内场景生成建模为基于分层先验引导的增量构建与学习校正过程，通过功能组顺序插入与多模态 move-rotate-scale 修复策略，显著提升了生成场景的物理有效性、功能连贯性与可控性。

## 研究问题与动机
1. **提示先验不足**：简短或信息稀疏的文本提示缺乏物体关系、功能分区与房间布局规律，从零推断极不稳定。
2. **一次性生成的级联错误**：现有单步预测或纯事后优化方法中，早期锚点物体（如床、沙发）的错放会级联影响后续布局，导致全局结构与功能失效，且事后全场景重优化成本高、易漂移。
3. **过程监督数据空白**：标准 3D 场景数据集仅提供干净的最终布局，缺乏“中间损坏状态 ↔ 可执行修复动作”的配对数据，限制了显式步级修复策略的学习。
4. **核心诉求**：室内场景构建具有路径依赖性，需将中间状态视为规划、观察、校正与监督的一等目标，而非最终布局的副产品。

## 核心贡献（创新点）
1. **分层检索增强规划 (HRAP)**：从离线空间记忆中检索房间级、功能组级与锚点级布局先验，以软上下文形式预置给规划器，使生成器能按功能单元而非孤立物体进行有序布局。
2. **多模态强化修复策略 (RMR)**：在每次功能组插入后执行轻量级局部修复，融合顶视/斜视渲染图、结构化场景 JSON 与动作历史预测 move-rotate-scale 动作，并通过紧凑质量分数 $Q$ 进行接受/拒绝过滤，避免有害漂移。
3. **SceneReverse-17k 数据集**：基于 3D-FRONT 构建的面向过程修复轨迹数据集，通过对干净场景施加位置/旋转/尺度扰动并取逆操作作为可执行修正目标，实现从静态干净场景到多模态中间状态监督的转化。
4. **Grow-and-Repair 推理范式**：将场景生成从一次性预测重构为“规划→分组生长→局部修复→全局修复”的序列决策过程，兼顾物理合理性、功能连贯性与计算效率。

## 方法详解
- **过程感知目标设定**：将场景表示为 $\mathcal{S} = \{(o_i, p_i, r_i, s_i)\}_{i=1}^N$，生成过程建模为序列 $S^{(0)} \to S^{(1)} \to \cdots \to S^{(M)}$，每步插入一个功能组 $g_m$。定义质量分数 $Q(S) = -\lambda_{\mathrm{pbl}} \mathrm{PBL}(S) - \lambda_{\mathrm{rel}} \mathrm{REL}(S) - \lambda_{\mathrm{func}} \mathrm{FUNC}(S)$，其中 PBL 度量越界与碰撞，REL 度量高置信关系违规，FUNC 度量可达性与通行可用性；推理时仅接受使 $Q$ 提升的动作。
- **HRAP 先验构造与检索**：从干净场景 JSON 中解析物体类别、位姿与房间类型，识别地板支撑的主导锚点物体，按类别兼容性、平面距离、表面间隙与支撑关系将其分配为锚点中心化功能组 $(a, \mathcal{N}_a)$。提取锚点-成员的局部坐标偏移、距离分布、朝向差与组合签名 $\sigma(r,a)$，转换为 overview/signature/member-prior 三类自然语言文档。推理时通过文本嵌入与余弦相似度检索 Top-5 先验，作为软提示prepend 给规划器，输出有序组计划 $\mathcal{G}$。
- **RMR 训练流程**：观测状态 $o_t = (I_t^{\mathrm{top}}, I_t^{\mathrm{diag}}, I_t^{\mathrm{ann}}, J_t, H_t)$。Stage I 基于 SceneReverse-17k 的逆修复轨迹进行 SFT，最大化目标动作序列的似然 $\mathcal{L}_{\mathrm{SFT}}$，学习 JSON 格式规范与粗粒度空间校正；Stage II 采用 GRPO 策略优化，采样 $N_{\mathrm{cand}}$ 个候选动作列表，奖励 $R = 0.15 R_{\mathrm{format}} + 0.15 R_{\mathrm{apply}} + 0.5 R_{\mathrm{phys}} + 0.2 R_{\mathrm{vlm}}$，计算组内相对优势 $A_i$ 后以裁剪代理目标更新策略 $\pi_\theta$。
- **分组生长与修复推理**：第 $m$ 组插入后定义局部作用域 $\Omega_m = \mathrm{Obj}(g_m) \cup \mathrm{NeighborAnchors}(g_m, S^{(m-1)})$，RMR 仅在此范围内提案；结合小规模候选集（微平移、微调角、贴墙对齐、锚点附着移动）与确定性几何清理，按 $Q$ 阈值接受最优候选；所有组插入完成后执行全局修复 $\Pi_{\mathrm{global}}$ 处理跨组通行瓶颈与累积拥挤。

## 实验与结果
- **数据集与评估设置**：100 个跨 5 种房间类型（living/bedroom/dining/library/laundry）的场景，包含 75 条 GPT-4o 生成的长提示与 25 条仅含房间类型与物体数量的短提示。基线：Reason-3D、ReSpace、ReSpace + Fine-tuned Qwen3-VL-8B。评估使用 GPT-5.4 VLM Judge 与 50 人人工评分（LC/SPA/FC，1-10 分制）。
- **核心定量结果**：ScenePilot
