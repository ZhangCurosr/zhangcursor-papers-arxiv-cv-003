---
title: "It-s-Not-What-the-Image-Shows-Irrelevant-Context-Destabilise"
source: https://arxiv.org/pdf/2609.37863v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 17:30:17"
field: "多模态NLP评估"
keywords: ["VLM judge", "替代性评估", "无关上下文", "MIST", "invariance test", "习语理解", "prompt敏感性"]
innovations: ["提出MIST基准，首次以标签保持扰动检验VLM judge对无关图像的鲁棒性", "发现图像存在本身动摇judge而非图像内容，且动摇不提升human一致性", "将invariance test从task model迁移至judge model，揭示替代性协议的条件依赖性"]
benchmarks: ["MIST", "AdMIRe (SemEval-2025 Task 1)", "ID10M-JAM"]
---

# 论文速读：It's-Not-What-the-Image-Shows-Irrelevant-Context-Destabilise

## 一句话总结
本文提出 **MIST（Misleading-Image Stress Test）**，通过200个带歧义图片的英文习语句子测试13个VLM judge对无关上下文的鲁棒性，发现图像**存在本身**而非内容会动摇模型判定，但这种动摇并不提升与人类标注者的一致性。

## 研究问题与动机
1. **替代性测试的脆弱性**：现有L作为human annotator的替代性协议（如alt-test）只基于单一输入配置给出 verdict，无法揭示结论是否依赖于特定评估条件。
2. **真实场景中的无关上下文**：真实标注任务中项目常附带无关图像或前置消息，人类可被指示忽略，但VLM可能无法做到。
3. **误差传播风险**：一旦用模型替代人工面板，其标签成为评测集和训练数据，其中的系统性误差会传播至后续研究并导致错误结论。
4. **既有基准的方向性偏差**：现有VLM相关基准（如IRFL、AdMIRe）以图像为决策对象，改变图像会合法改变答案；而本文旨在构造**标签不变、仅扰动judge**的实验设置。

## 核心贡献（创新点）
1. **提出MIST基准**：构建200条英文句子+4种图像条件的对照测试集，其中图像是无关上下文，正确标签由句子本身决定，实现了标签保持的扰动。
2. **发现"存在效应"而非"内容效应"**：证明VLM judge被无关图像动摇的程度与图像内容无关（aligned和misleading图像引起相似的标签改变率），动摇来自图像**存在**本身。
3. **揭示替代性评估的盲点**：aggregate agreement指标掩盖了instance-level不稳定性，证明一个judge可以是"可替代的"，但仍对无关上下文极度不稳定。
4. **建立行为测试框架应用于judge而非task model**：将invariance test从任务模型转移到评估模型，并在第二模态方向上进行双向扰动，为评估协议的鲁棒性分析提供了方法论参考。

## 方法详解
**MIST数据集构建：**
- 基于AdMIRe共享任务的551个英语潜在习语表达，抽样200个表达式，保留每个表达式的一条句子（100条figurative + 100条literal）。
- 由于被丢弃行的图像仍可使用，每条保留句子均可与任意图像配对，形成4种条件（各50条）：
  - T1：figurative句子 + literal图像（misleading）
  - T2：figurative句子 + figurative图像（aligned）
  - T3：literal句子 + literal图像（aligned）
  - T4：literal句子 + figurative图像（misleading）

**标签体系（四级，半有序）：**
- FF（Fully Figurative）：短语完全 figurative，字面含义不起作用
- WF（Weak Figurative）：figurative使用，但字面与比喻之间仍有语义联系
- FL（Figurative and Literal）：两种解读同时活跃
- LL（Fully Literal）：仅字面意义，无 figurative 解读激活

**实验设置：**
- 13个VLM judge（7个通过alt-test + 6个不通过），覆盖GPT-5.2、Gemini系列、Gemma-3、Mistral、Qwen等，参数量3B~32B。
- 4种prompt策略：zero-shot、few-shot、CoT、few-shot+CoT。
- 每个judge在每种策略下分explicit arm（明确告知忽略图像）和silent arm（不提及图像），共20个标签/项，总计52,000标签。

**评估指标：**
- **Change rate**：标签变化的比例（从纯文本到添加图像，以及删除"忽略图像"指令的对照）
- **Direction**：变化标签中移向图像所示含义的比例（chance=50%）
- **Agreement**：与人类多数投票标注的exact match准确率
- **Substitutability**：alt-test（ε=0.15或0.20）

## 实验与结果
**主要数据（ pooled over 4 prompts）：**

| 指标 | Aligned图像 | Misleading图像 | Prompt编辑控制 |
|------|------------|----------------|----------------|
| 平均change rate（7个通过judge） | 15.9% | 14.7% | 8.8% |
| 平均change rate（全部13个judge） | 20.5% | 19.4% | 11.6% |
| Direction（移向图像的比例） | 34%（全13个） | — | — |
| Agreement with humans | 53.7% | 54.7% | 54.4%（无图） |

**核心结论：**
1. **图像存在动摇judge**：无论图像内容如何，添加图像均显著改变标签（均高于prompt编辑对照）。
2. **内容不影响方向**：仅37%的变化移向图像所示含义（低于50%随机），表明模型未被图像"inform"，只是被"destabilise"。
3. **准确率不变**：图像不降低整体agreement，因为变化在aggregate层面相互抵消。
4. **通过alt-test的judge仍受影响**：7个通过alt-test的judge的change rate为15.9%，而6个不通过的为25.8%，但所有judge均受影响。
5. **71%的变化不跨越figurative-literal边界**：图像主要在同侧内部洗牌，而非改变根本解读。
6. **部分judge呈现active suppression**：6个judge的F/S比率<0.4，表明它们实际在主动抵抗图像影响，而非忽视。

## 相关工作脉络
1. **Alt-test替代性评估**（Calderon et al., 2025）：本文对比的核心理论基准，指出其单次配置评估的局限，MIST提供了同一judge在扰动条件下的行为分析。
2. **LLM-as-a-judge的prompt敏感性**（Li et al., 2025）：已有工作发现judge分数随prompt变化，但本文将其从"需优化的设计选择"重新定位为"威胁替代性决策的漏洞"。
3. **ID10M-JAM**（Hashiloni et al., 2026）：同样保持标签不变的习语压力测试，但扰动文本而非图像，且以识别准确率为目标而非与人工标注者的一致性。
4. **无关上下文干扰LLM**（Shi et al., 2023; Gonen et al., 2025）：测量任务准确率退化，本文关注的是与人类annotator agreement的偏离。
5. **VLM中text优先于image**（Deng et al., 2025）：发现VLM在文本-图像冲突时偏向文本，但其设置是文本被corrupt，本文是图像为无关干扰。
6. **AdMIRe / IRFL / V-FLUTE**：现有多模态习语基准，但图像是决策对象；本文反转方向，让图像成为无关上下文，句子解读才是目标。

## 局限性与未来方向
1. **两种图像均与短语相关**：缺乏真正无关图像（与短语毫无关联）的条件对比，无法判断相关图像的影响是否更大。
2. **标签尺度的部分有序性**：FF↔WF和FL↔LL是两个不同性质的判断边界，一步移动在不同位置的语义量不同。
3. **无人类change rate**：每位人类标注者仅看到一种条件，无法测量人类自身标签因无关原因变化的频率，只能依赖prompt编辑作为within-judge对照。
4. **仅英语单一语言**：结论能否推广至其他语言未经验证。
5. **未测试句子真正改变含义时的情形**：MIST仅做invariance测试，缺少complementary directional测试（当句子本身确实改变解读时，模型是否会正确跟随）。

## 研究启发与可借鉴点
1. **invariance test应用于judge而非task model**：可将相同思路迁移到任何"模型替代人工评估"的场景，检验judge在多配置下的稳定性，作为替代性评估的必要补充。
2. **双向扰动设计**：同时测试aligned和misleading两种方向的扰动，能够分离"内容效应"与"存在效应"，此设计可用于检验模型对其他无关上下文（如前置消息、无关图表）的鲁棒性。
3. **label-preserving perturbation思想**：保持正确标签不变、仅扰动输入条件，是检验模型决策是否基于目标信号的经典方法，可广泛复用于评测各类AI系统的context handling能力。
4. **prompt sensitivity作为系统漏洞而非设计缺陷**：本文框架将prompt变化从"需工程优化的噪声"重新定义为"评估协议有效性的威胁"，这一视角转换值得在团队后续的评估研究中借鉴。
5. **与alt-test的互补关系**：alt-test回答"模型是否可替代人类"，MIST回答"替代性结论对配置有多敏感"，两者结合可构建更全面的替代性验证pipeline。

## 关键术语表
**MIST（Misleading-Image Stress Test）**：本文提出的压力测试集，包含200个带歧义图像的习语句子，用于检验VLM judge对无关上下文的鲁棒性。

**Alt-test（Alternative Annotator Test）**：Calderon等人提出的L替代性评估协议，检验模型是否至少与一名被 withholding 的人类标注者同样一致。

**Figurative / Literal reading**：习语的比喻义解读 vs. 字面义解读；同一短语在不同句子中可能有不同解读。

**Label-preserving perturbation**：保持正确标签不变、仅扰动输入条件的实验设计，变化量即为模型的错误率。

**Invariance test**：Ribeiro等人提出的行为测试方法，检验模型在标签保持扰动下的预测一致性。

**Change rate**：两个输入条件下judge标签发生变化的比例，衡量模型对扰动的敏感度。

**Direction**：在发生变化的标签中，移向扰动所指示方向的比例，衡量扰动是否有方向性地影响judge。

**Explicit arm / Silent arm**：prompt中明确告知judge忽略图像（explicit）与完全不提及（silent）的两种实验条件。

## 可复现要素
- **数据集**：MIST（200个英语项目），来源于AdMIRe共享任务（SemEval-2025 Task 1）的551条候选表达式，已公开发布（论文声明release MIST with all human annotator and VLM judge labels）
- **代码**：论文未明确提供代码仓库链接，但发布了全部标注数据和judge标签
- **权重**：评测的13个VLM均为公开或可访问模型（GPT-5.2、Gemini系列、Gemma-3、Mistral、Qwen系列等），其中openweight模型参数范围3B~32B
- **关键超参**：alt-test slack ε=0.15（blind trio）/ 0.20（informed trio）；greedy decoding；推理模式统一禁用
