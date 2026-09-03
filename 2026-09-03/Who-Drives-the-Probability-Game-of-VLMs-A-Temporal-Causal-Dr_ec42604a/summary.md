---
title: "Who-Drives-the-Probability-Game-of-VLMs-A-Temporal-Causal-Dr"
source: https://arxiv.org/pdf/2609.02000v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 18:13:04"
field: "多模态大模型评估与可解释性"
keywords: ["Vision-Language Models", "Causal Inference", "Evaluation Framework", "Autoregressive Generation", "Process-Oriented Evaluation", "Structural Causal Model"]
innovations: ["基于SCM的无参考时序因果驱动指标（PCD/QCD/VCD）", "通过do-算子与后背调整剥离混杂实现源级干预估计", "揭示早期视觉/问题引导→后期前缀依赖的一致性动态规律"]
benchmarks: ["MAVIS", "LLaVA-Video-178K", "MiraData", "VLMBias"]
---

# 论文速读：Who-Drives-the-Probability-Game-of-VLMs-A-Temporal-Causal-Dr

## 一句话总结
论文提出一种基于结构因果模型（SCM）的时序因果评估框架，通过前缀因果驱动（PCD）、问题因果驱动（QCD）和视觉因果驱动（VCD）三个无参考答案指标，追踪多模态大模型在自回归生成过程中不同信息源的动态支配关系，揭示"早期问题/视觉引导→后期前缀依赖"的一致演变规律。

## 研究问题与动机
1. **现有评估偏结果导向，缺乏过程溯源**：BLEU、CIDEr、BERTScore等常规指标仅衡量生成文本与参考答案的表层一致性，无法诊断错误根源（视觉接地不足、问题误解或前缀误差累积）。
2. **模态贡献估计静态化**：现有模态贡献方法（信息分解、相对熵等）仅提供全局重要性分数，无法刻画自回归生成过程中各信息源影响力的时间演化。
3. **因果混淆难以剥离**：视觉输入V与问题Q存在依赖（V→Q），且两者共同影响答案A，导致观测相关性无法直接解释为因果效应，需借助do-算子与后背调整剥离混杂路径。
4. **多源协同诊断缺失**：缺乏可同时区分"前缀过度依赖"、"视觉抑制"与"多源协调生成"的细粒度过程级诊断工具。

## 核心贡献（创新点）
1. **提出时序因果驱动框架**：将VLM答案生成形式化为受问题、视觉输入和历史前缀共同影响的动态自回归过程，构建三个步骤索引的因果驱动指标（PCD/QCD/VCD），无需参考答案即可实现源级诊断。
2. **推导可计算的后背调整公式**：针对PCD和QCD存在混杂路径的问题，基于SCM引入do-算子进行后背调整，分别对视觉V和问题Q的经验分布求和，得到无偏的干预概率估计；VCD通过固定空问题直接识别（V为根节点）。
3. **揭示一致的生成动态规律**：实验发现Qwen3-VL-8B-Instruct与InternVL2-8B在多数据集上均呈现"早期QCD/VCD较高→后期PCD持续上升"的共性轨迹，为模型行为解释提供统一视角。
4. **外部有效性验证**：在VLMBias上，前缀-视觉不平衡分数$S_{PV}=\overline{PCD}-\overline{VCD}$以0.767 AUROC/0.873 AUPRC区分先验驱动与视觉基础生成；随机干预恢复实验显示QCD/PCD较观察性PMI基线误差降低34.8%/47.1%。

## 方法详解
**结构因果模型设定**：
- 变量：视觉输入$V$（根节点）、问题$Q$（受$V$影响，边$V→Q$）、答案序列$A_{\leq t}$（$A_t$为当前token）
- 因果路径：直接路径$A_{<t}→A_t$、$V→A_t$、$Q→A_t$；混杂路径$A_{<t}←Q→A_t$、$A_{<t}←V→A_t$、$Q←V→A_t$

**共享空源参考**：
$$P_0(A_{\leq t}=a_{\leq t})=P(A_{\leq t}=a_{\leq t}\mid V=\emptyset, Q=\emptyset)$$
作为三个指标的公共校准基线。

**PCD（前缀因果驱动）**：
对$do(A_{<t}=a_{<t})$做后背调整，边际化$(V,Q)$：
$$PCD_t=\log_2\frac{\sum_{v,q}P(V=v)P(Q=q\mid V=v)P(A_t=a_t\mid A_{<t}=a_{<t}, V=v, Q=q)}{P_0(A_{\leq t}=a_{\leq t})}$$

**QCD（问题因果驱动）**：
对$do(Q=q)$做后背调整，边际化$V$：
$$QCD=\log_2\frac{\sum_v P(V=v)P(A_{\leq t}=a_{\leq t}\mid V=v, Q=q)}{P_0(A_{\leq t}=a_{\leq t})}$$
分子按自回归链式法则展开为逐token概率连乘。

**VCD（视觉因果驱动）**：
固定$Q=\emptyset$，直接计算（无需后背调整，V为根节点）：
$$VCD=\log_2\frac{P(A_1=a_1\mid V=v, Q=\emptyset)\cdot\prod_{i=2}^{t}P(A_i=a_i\mid A_{<i}=a_{<i}, V=v, Q=\emptyset)}{P_0(A_{\leq t}=a_{\leq t})}$$

**计算细节**：边际权重$P(V)$与$P(Q\mid V)$从评估数据经验估计；支持集大小$K=20$（默认），PCD复杂度$O(|V|\cdot|Q|\cdot T)$，QCD为$O(|V|\cdot T)$，VCD最轻为$O(T)$。

## 实验与结果
**实验设置**：
- 模型：Qwen3-VL-8B-Instruct（主实验）、InternVL2-8B（跨模型验证）
- 数据集：MAVIS（500样本，图像VQA）、LLaVA-Video-178K（100样本，视频QA）、MiraData（500样本，开放式视频QA）
- GPU：8×NVIDIA RTX 4090（24GB）；大规模随机干预验证用8×NVIDIA H200

**主要结果**：
- **时序动态**：三个模型/数据集上均观察到一致的"早期QCD/VCD主导→后期PCD上升"过渡模式；MAVIS上转变最快，视频任务上QCD持续性更强；InternVL2-8B的VCD/QCD在早中期更强。
- **聚类分析**（$k=3$）：Cluster 1（74%）前缀驱动+视觉弱化；Cluster 2（6%）视觉抑制（VCD全程为负）；Cluster 3（20%）多源协调生成（VCD高且增长）。
- **随机干预恢复**（表5）：QCD平均恢复误差从PMI_Q的7.822 bits降至5.434 bits（↓34.8%，98%样本更优）；PCD从3.660 bits降至2.019 bits（↓47.1%，100%样本更优）。
- **VLMBias外部验证**（表4）：Answer-level VCD → AUROC 0.747/AUPRC 0.842；$S_{PV}$ → AUROC 0.767/AUPRC 0.873，有效区分先验驱动vs视觉基础生成。
- **与传统指标互补性**：QCD/VCD与BERT-F1/ BLEU弱正相关（Spearman~0.2），PCD与传统指标近乎不相关，证明捕捉正交信号。

## 相关工作脉络
1. **结果导向评估**（BLEU/CIDEr/BERTScore等）：仅衡量生成与参考答案的表层一致性，本文转向过程级源溯源，弥补"答案正确但证据缺失"的盲区。
2. **过程导向评估**（ReCEval/THINK-Bench/Temporalizing Confidence）：关注推理链正确性、效率与置信度轨迹，但未分解多模态来源的贡献动态，本文首次引入因果干预实现源级动态追踪。
3. **模态贡献量化**（Partial Information Decomposition/相对熵方法）：提供表征级或全局静态贡献估计，无法刻画自回归生成的时序演化，本文的step-indexed设计弥补此缺口。
4. **视觉幻觉与接地诊断**（Grounded CoT/Multi-modal Hallucination Control）：侧重检测幻觉或验证显式推理链有效性，本文通过因果驱动轨迹直接定位"何时何地"模型偏离视觉证据。
5. **因果推断在NLP中的应用**（CausalML等）：传统方法依赖昂贵的反事实解码，本文利用后背调整+teacher-forced评分实现高效无偏估计，避免逐样本重生成。

## 局限性与未来方向
1. **因果驱动≠正确性保证**：强视觉驱动仅表示概率支持，不代表视觉理解正确；需与答案级指标联合使用。
2. **SCM假设限制**：因果解释依赖所设定的结构因果图，未考虑未被调整的样本级共同原因；$P_0$参考项的选择（空prompt/black image）可能引入敏感性。
3. **计算开销**：PCD需遍历$|V|\cdot|Q|$对，复杂度最高，虽可用$K=20$近似但仍有扩展瓶颈。
4. **实验规模有限**：仅验证8B/32B两档模型，未覆盖更大规模（如65B+）及长链式推理（CoT）场景，后者是明确提到的未来方向。
5. **静态参数假设**：因果效应估计基于固定模型参数，未考虑训练动态或在线学习场景。

## 研究启发与可借鉴点
1. **SCM+后背调整的因果驱动度量范式**可直接迁移至LLM推理链评估（追踪"问题→中间步→最终答案"的因果贡献），为Chain-of-Thought归因提供新工具。
2. **步索引轨迹聚类**（30维表示+kmeans）可用于大模型生成行为的无监督类型学挖掘，发现稀有模式（如视觉抑制簇仅6%但具诊断价值）。
3. **随机干预恢复误差**可作为因果度量有效性的通用验证协议，本文的34.8%/47.1%提升幅度为后续方法比较提供了基准参考。
4. **前缀-视觉不平衡分数$S_{PV}$**设计简洁且具解释性，可推广至视频/多轮对话场景诊断"上下文依赖 vs 内容接地"的失衡。
5. **无参考答案设计**（依赖$P_0$共享基线）对低资源/开放域任务极具价值，避免了人工标注依赖，可与现有benchmark无缝集成。

## 关键术语表
**Structural Causal Model (SCM)**：用有向图表示变量间因果关系的数学框架，支持do-算子干预与后背调整以识别因果效应。
**do-operator**：Caesar Pearl提出的干预算子，$do(X=x)$表示强制将变量X设为x并切断其所有入边，用于估计因果效应而非相关性。
**Backdoor Adjustment（后背调整）**：通过条件化混杂变量并边际化，阻断从处理变量到结果变量的"后门路径"，从而识别因果效应的数学方法。
**Prefix Causal Drive (PCD)**：固定历史前缀后当前token概率相对于空源参考的对数比值，刻画自回归惯性/历史依赖强度。
**Question Causal Drive (QCD)**：对问题做干预后经后背调整的答案概率相对提升，衡量问题文本对生成语义目标的引导力度。
**Visual Causal Drive (VCD)**：固定视觉输入且问题为空时的答案概率相对提升，隔离视觉证据对生成内容的独立支撑作用。
**Randomized Intervention Recovery（随机干预恢复）**：在 held-out 样本上主动重分配信息源（如随机配对图像-问题），检验因果驱动指标能否准确恢复模型的实际行为响应。
**Prefix-Visual Imbalance Score ($S_{PV}$)**：$\overline{PCD}-\overline{VCD}$，综合衡量生成过程中前缀历史依赖相对于视觉证据的压倒性程度。

## 可复现要素
- **数据集**：MAVIS、LLaVA-Video-178K、MiraData、VLMBias（均为公开benchmark）
- **模型权重**：Qwen3-VL-8B-Instruct、InternVL2-8B（开源模型）
- **代码**：论文未提及代码开源（arxiv链接未包含code availability声明）
- **关键超参**：支持集大小$K=20$（默认）；聚类$k=3$，10个bin分箱；随机干预采样$M=100$；GPU配置8×RTX 4090（24GB）或8×H200
- **复现难点**：PCD需遍历$K^2$对$(v,q)$组合做teacher-forced打分，显存与时间开销显著；$P_0$中$Q=\emptyset$的实现方式（空prompt vs任务无关prompt）需确认
