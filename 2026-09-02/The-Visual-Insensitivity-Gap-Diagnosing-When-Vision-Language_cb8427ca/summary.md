---
title: "The-Visual-Insensitivity-Gap-Diagnosing-When-Vision-Language"
source: https://arxiv.org/pdf/2609.00868v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 15:51:46"
---

# 论文速读：The-Visual-Insensitivity-Gap-Diagnosing-When-Vision-Language

## 一句话总结
本文提出“视觉不敏感差距”（Visual Insensitivity Gap）与逐样本视觉敏感度指数（VSI），通过问题相关区域的反事实模糊测量 VLM 是否真正使用了视觉输入；发现该现象具有跨模型一致的样本内在性，并证明 VSI 在多选题推理场景下能有效诊断“无视视觉证据的自信错误”，但更适合作为条件集成组件而非通用置信度替代信号。

## 研究问题与动机
- 现有 VLM 评估依赖多模态基准的聚合准确率，隐含假设模型已利用视觉输入，但聚合指标无法区分“自信且无视图像的错误”与“谨慎且正确使用视觉的回答”。
- 对问题相关视觉区域施加高斯模糊后，6 个主流 VLM 在 40%–97% 的样本上输出分布几乎不变，表明视觉编码器已感知扰动，但语言头未将其传播至最终词元。
- 内部状态探测（注意力图、线性探针）只能证明信息存在于中间表示，无法建立输入扰动与输出因果变化之间的实证联系。
- 需要一种逐样本、跨架构可迁移的输入级干预指标，以定位视觉利用失败的具体模式并指导下游 abstention 决策。

## 核心贡献（创新点）
1. 形式化 Visual Insensitivity Gap 并提出 VSI 指标，基于 KL 散度量化局部视觉扰动对 next-token 分布的全分布漂移。→ 与已有工作的本质区别：从描述性内部探针转向输入级反事实因果测量，直接回答“若该视觉证据消失，答案是否会改变”。
2. 证实视觉不敏感性是样本内在属性：15 对模型间的 VSI 排名 grand-mean Spearman ρ=+0.40（permutation p<10^-3），跨架构差异极大的模型仍对同一批样本一致忽略。→ 与已有工作的本质区别：将失败归因从“特定模型架构缺陷”转移至“样本特性与主流对比预训练范式”的共性问题。
3. 揭示 encoder–LLM 路由断层机制：低 VSI 样本上自有视觉塔线性探针准确率 0.72–0.79，但 Δargmax 变化率仅 2%–11%，Gap 稳定 >0.65。→ 与已有工作的本质区别：同时结合表征可分性证明与输出端反事实验证，明确故障位于跨模态路由而非编码器容量不足。
4. 系统绘制 VSI 诊断效用边界：MMStar 数学/科学子类在 Qwen2.5-VL-32B 上 AUROC_VSI 达 0.85/0.87，但在已良好校准的事实性基准上 max-prob 仍占优（18 个单元格中 softmax 胜 10、混合信号胜 7、纯 VSI 仅 1）。→ 与已有工作的本质区别：拒绝单一最优置信度信号的假设，提出 VSI 应作为条件性集成组件按需启用。

## 方法详解
- **VSI 计算**：给定 $f(\cdot|x,q)$，先用轻量 noun phrase 解析器提取问题主名词，经 Grounding-DINO 获取 open-vocabulary 框再经 SAM 细化为像素掩码；对该区域施加强度 $\sigma=20$ 的高斯模糊得到 $x_\sigma$，计算 $VSI(x,q;f)=\text{KL}(f(\cdot|x,q)\|f(\cdot|x_\sigma,q))$，取 top-50 词元分布对齐求 KL。无框样本回退至全图模糊。
- **样本分层与分组**：按 VSI 值五等分；重点分析低 VSI 子群，并以 high-VSI 匹配样本为对照验证探针容量。
- **Encoder–LLM 断层测量**：在 Qwen3-VL 标定的 83 个低 VSI 共享池上，对每个模型自有视觉塔末层 $\ell_2$-归一化特征做 $L_2$ 正则逻辑回归探针（$C=1.0$，分组 5-fold CV，同图正负样本同 fold）；同步记录扰动后 Δargmax 比例，Gap = probe acc − Δargmax。
- **诊断与集成评估**：以每样本错误为 positive class、−VSI 为 ranking score 计算 AUROC；构建等权 z-score 混合信号 $\text{hyb}(r+w)$（区域 VSI + 全图 VSI），并与 max-prob、verbalised confidence 联合比较 PRR@80。
- **跨模型一致性检验**：两两计算 Spearman 排名相关，辅以 $10^5$ 次置换检验；对 σ 取值与阈值截断做敏感度扫描验证非调参依赖。

## 实验与结果
- **数据集与模型**：6 个 VLM（LLaVA-1.5-7B, LLaVA-NeXT-Mistral-7B, Idefics3-8B, Qwen3-VL-8B, Qwen2.5-VL-7B, Qwen2.5-VL-32B）；4 个基准（POPE, MMVP, HallusionBench, MMStar）。
- **VSI 分布**：6 模型 × 3 感知基准中 VSI<0.05 样本占比 40%–97%，HallusionBench 最高、POPE 最低；Hartigan’s dip test 未拒绝单峰性，属重左尾分布。全图模糊变体 insensitive fraction 仅 ~6%，证实局部重尾非任意扰动 artefact。
- **跨模型一致性**：grand-mean Spearman $\rho=+0.40$；同类家族平均 0.51，跨家族 0.37。POPE 最强（0.55），MMVP（0.34），HallusionBench 最弱（0.32，受图表地区检测失败干扰）。
- **断层机制**：低 VSI 池上各模型自有塔探针 0.72–0.79，独立 frozen CLIP 参考 0.82±0.04；Δargmax 2%–11%，Gap 0.65–0.71 在所有模型上一致。high-VSI 对照组探针升至 0.86–0.91 且 Δargmax 同步上升，排除探针容量天花板。
- **诊断效用**：MMStar math AUROC=0.851、science AUROC=0.867（Qwen2.5-VL-32B）；POPE/HallusionBench 整体 AUROC_VSI 多在 [0.45, 0.60]。18 个单元格中 max-prob 胜 10、含 VSI 混合信号胜 7、纯 VSI 胜 1；hyb(r+w) 在 Qwen3-VL POPE 实现 AUROC 0.636 vs max-prob 0.544（paired bootstrap p<0.01）。
- **鲁棒性**：σ∈{10,20,40} 切换下 AUROC 波动 ≤0.05，跨 σ 排名 Spearman 0.76–0.97；阈值 {0.01,0.05,0.10,0.20} 扫描下低/高 VSI 错误率方向不变，极差比波动 ≤1.3×。

## 相关工作脉络
- **VLM
