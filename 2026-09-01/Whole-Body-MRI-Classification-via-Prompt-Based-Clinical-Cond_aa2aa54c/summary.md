---
title: "Whole-Body-MRI-Classification-via-Prompt-Based-Clinical-Cond"
source: https://arxiv.org/pdf/2608.30824v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 19:20:55"
---

# 论文速读：Whole-Body-MRI-Classification-via-Prompt-Based-Clinical-Cond

## 一句话总结
提出TACTIC框架，将结构化临床变量以Prompt序列形式直接条件化全身体磁共振（WB-MRI）视觉特征，无需插补即可兼容任意数量的表格输入，在5项系统性与肿瘤类疾病分类任务上显著优于单模态基线与主流多模态融合方法，并在临床数据缺失场景下保持高度稳健。

## 研究问题与动机
1. **临床数据缺失的现实瓶颈**：传统多模态融合方法假设结构化临床变量完整且维度固定，但真实医疗记录普遍存在缺失、稀疏或异构问题，导致模型性能在实际部署中急剧退化。
2. **现有Prompt方法的模态局限**：当前医学领域的Prompt机制多依赖空间提示、自由文本放射报告或需额外对齐的跨模态映射，未能直接利用原生结构化表格数据，预处理成本高且语义对齐困难。
3. **WB-MRI多模态融合潜力未释放**：全身体MRI具备无辐射、多器官同步成像的优势，但在自动化诊断管线中尚未充分结合实验室指标、病史等不可见临床上下文，限制了其在精准医学中的落地。

## 核心贡献（创新点）
1. 提出TACTIC（Tabular-Attribute Conditioned Transformer for Image Classification），将临床属性名-值对编码为Prompt序列，通过Cross-Attention条件化视觉特征；与Concat/Max/Sum等固定维度拼接或门控融合策略本质不同，原生支持变长输入与任意比例缺失。
2. 将多模态融合重新定义为条件化问题，使预训练视觉主干（Primus-M）无需修改架构即可接入临床上下文，避免了传统方法对固定表格维度的强假设。
3. 引入MoFe（More vs. Fewer）与TabMoFe损失，强制模型在完整与部分临床数据条件下均实现性能单调增益，从根本上抑制缺失属性引入的表征污染。
4. 针对WB-MRI背景体素占比超40%的问题，设计掩码感知池化与仅对解剖区域进行MAE重建的预训练策略，显著提升3D医学影像的表征质量。
5. 在UK Biobank的5项独立疾病分类任务上系统验证，TACTIC在4/5任务取得最优AUROC，且在仅含少量或低重要性临床属性时仍显著优于所有融合基线。

## 方法详解
- **图像编码器（Primus-M）**：采用MAE框架预训练。先用Otsu阈值+二元形态学生成身体掩码，执行掩码感知池化（窗口 `[2×2×4]`）压缩背景token；MAE重建损失仅作用于含解剖信息的patch，避免背景偏置主导预训练目标。
- **表格编码器（TARTE）**：联合编码属性名（如 `sex`）与属性值（如 `female`），为每个临床变量生成统一维度的嵌入向量 `c ∈ R^(K×d)`。
- **Prompt条件化Transformer**：图像特征 `x ∈ R^(n×d)` 与临床嵌入并行处理。训练中随机采样N个子集 `{c_1,...,c_N}`（每子集含k个属性），与输出Token `T_out` 拼接为Prompt序列。先对临床Prompt做Self-Attention捕捉全局依赖，再通过Cross-Attention引导图像特征聚合，输出多模态预测。
- **损失函数设计**：
  - 多模态损失：`L_mm = CE((X, C), y)`，取N个子集预测均值。
  - 图像-only损失：`L_img = CE((X, ∅), y)`。
  - MoFe损失：`L_MoFe = max(L_mm - L_img, 0)`，确保引入临床Prompt后性能不劣于纯图像分支。
  - 总损失：`L = L_MoFe + L_mm + L_img`。
  - 表格编码器预训练阶段采用TabMoFe损失 `L_TabMoFe = max(L_more - L_few, 0)`，鼓励增加临床信息时性能持续提升。
- **训练配置**：多模态微调时冻结TARTE，从单模态微调权重初始化TACTIC；图像masking率设为40%以去除背景
