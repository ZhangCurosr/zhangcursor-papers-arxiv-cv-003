---
title: "The-Diagnosis-a-Reporter-Leaves-Unspoken-Surfacing-Frozen-Tu"
source: https://arxiv.org/pdf/2609.02411v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 10:18:06"
---

# 论文速读：The-Diagnosis-a-Reporter-Leaves-Unspoken-Surfacing-Frozen-Tu

## 一句话总结
论文提出 NeuroFusion，一种通过冻结分割特征上的轻量判别分类头“浮现”潜在诊断信号的条件化报告生成框架；该方法在相同 Mistral-7B 骨干上恢复了对脑膜瘤与转移瘤的正确诊断，以 5–6× 更低延迟输出机器可核查的结构化报告，并在双盲神经科医生评测中实现零关键错误。

## 研究问题与动机
1. **诊断抑制（Diagnostic Suppression）现象**：同骨干的强 CoT 报告器在冻结特征上对三类肿瘤的线性探针可达 0.82 macro-F₁，但实际生成时诊断召回率仅 0.44（脑膜瘤）/ 0.07（转移瘤），证明诊断信号存在于模型内部却未被 verbalize。
2. **现有生成器缺乏可核查架构**：主流 3D 医学 VLM 与脑报告生成器（如 AutoRG-Brain、BrainGemma3D）默认输出自由文本且倾向最常见诊断，无法保证 schema 合法性与句子级语义一致性。
3. **规则/掩码读取路径的天花板**：直接读取分割掩码字段可填满结构化模板，但只能生成固定鉴别诊断，且受限于掩码覆盖质量，无法处理分布外病例。
4. **临床工作流自动化缺口**：全球约三分之二人口缺乏诊断影像访问，放射科医生读片压力大（单例脑 MRI 约 11 分钟）、 burnout 率高，亟需既能提速又能承诺诊断且可审计的辅助工具。

## 核心贡献（创新点）
1. **实证诊断抑制现象**：揭示冻结特征线性可分性与 CoT 语言模型输出间的巨大落差，为“模型知道但不说”提供定量证据。
2. **判别性头条件化（Discriminative-head conditioning）**：在冻结特征上部署轻量枚举分类头预测结构化字段，并将 argmax 注入解码器作为 committed conditioning，而非直接覆盖 LM 输出。
3. **受控负向结果（Diagnosis Pin）**：证明若用诊断头硬钉覆盖解码器，跨分布召回会完全坍缩（MET 从 0.75 跌至 0.03），阐明“浮现引导”优于“强制覆盖”的本质。
4. **机器可核查审计层**：结合 XGrammar 约束解码与 record-verbalizer 变体，将逐句矛盾率从 36.8% 降至 7.5%，实现每句 entailment-checkable。
5. **首个神经科医生双盲 reader study**：在九案例盲评中，NeuroFusion 每类肿瘤均获最高评分，且是唯一零关键错误的系统，填补脑肿瘤报告生成器的临床可信度验证空白。

## 方法详解
1. **冻结骨干与病灶路由**：预训练 MedNeXt-L（foreground Dice 0.89）完成肿瘤分割；连通分量分析保留最多 4 个病灶，按体积排序并以 3D 质心位置编码标记，保证输入有界。
2. **每病灶 Q-Former 压缩**：每个病灶的池化特征经独立 Q-Former 交叉注意力压缩为固定 32 个视觉 token，形成按体积顺序排列的视觉前缀，避免输入长度随病灶数线性增长。
3. **字段分类头（Head-surfacing）**：在 Q-Former 输出上接轻量线性枚举分类头，直接预测 composition、enhancement pattern、mass effect 等结构化字段；**诊断字段故意不设为头**，交由 LLM 自行恢复以保持分布外泛化。
4. **单遍 Draft-then-Review 解码**：字段 argmax 转为文本提示（如 “temporal; heterogeneous; ring;”）注入 LLaVA-Med v1.5 的 Mistral-7B（QLoRA 微调）；XGrammar 掩码强制每步仅采样 schema 允许 token，单次 pass 即生成完整 JSON 报告。
5. **Record-verbalizer 变体**：关闭 LoRA adapter，仅用 head 输出驱动文本生成，用于最大化 faithfulness 与语义核查实验。
6. **校准与模态归因**：采用 temperature scaling（NLL 拟合 50 校准集）将 15-bin ECE 从 0.128 降至 0.095；精确 Shapley 分解量化各 MRI 序列贡献（如 FLAIR 主导 edema 字段，贡献 54%）。

## 实验与结果
- **数据集**：BraTS-2020（121 份带结构化报告，39 in-dist 测试）；RadGenome GLI（n=60）/ MEN（n=60）；BraTS-MET（n=60，reporter 训练阶段零接触）。
- **评估协议**：Schema validity、per-field union-class macro-F₁、RadGraph-F₁、RaTEScore、GREEN、Clin-O（1-5 临床量尺）；预注册 Holm 校正 + TOST 等价界 ±0.03，20000 次 BCa 配对 bootstrap。
- **同骨干对比**：NeuroFusion 在 9 项 prose-content 比较中赢 8 项（MET RadGraph 不显著），无倒退；主指标 RaTEScore-MEN 提升 +0.117；延迟 80s vs 457s/case（5-6× 加速，可部署于 L4）。
- **诊断恢复**：MEN 召回 0.92、MET 召回 0.75（CoT 仅 0.44/0.07）；Field-conditioned LM 自身诊断能力优于 head（GLI 0.98/0.92 vs 0.85/0.77）。
- **负向对照**：Diagnosis pin 变体在 MET 召回坍缩至 0.03（2/60），验证硬钉不可行。
- **外部基线**：在 MET 上显著领先所有 learned baselines（RaTEScore 0.596，Clin-O 3.17）；AutoRG-Brain 仅结构化指标略优但完全不输出诊断（ddx=1.02）。
- **Reader study**：两位认证神经科医生独立盲评，NeuroFusion 每类肿瘤评分最高（GLI 4.47 / MEN 4.00 / MET 3.47），零关键错误，8/9 案例获首选签字（sign-of），且两名阅读者在全部 255 个轴项上完全一致。

## 相关工作脉络
1. **3D 医学 VLM 报告生成**（M3D-LaMed、LLaVA-Med）：端到端自由文本生成，无结构化 schema 与诊断承诺；本文定位为其诊断盲区提供判别性头条件化解法。
2. **图像区域 grounding 报告器**（AutoRG-Brain、BrainGemma3D）：基于分割掩码生成 findings，但诊断仍交由 LM
