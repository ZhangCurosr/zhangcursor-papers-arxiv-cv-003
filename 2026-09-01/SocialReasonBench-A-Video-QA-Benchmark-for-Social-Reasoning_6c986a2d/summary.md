---
title: "SocialReasonBench-A-Video-QA-Benchmark-for-Social-Reasoning"
source: https://arxiv.org/pdf/2608.30716v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 10:42:36"
field: "多模态社会推理评测"
keywords: ["视频理解", "社会推理", "反事实推理", "多模态大模型", "基准评测", "互动叙事游戏"]
innovations: ["提出 SocialReasonBench，基于互动叙事游戏分支结构构建可验证的社会推理视频QA基准（532实例，7维度）", "设计多智能体自动化策展管线，将游戏内状态信号作为可核查的答案锚定源", "构建五类诊断性干扰项体系（视觉/逻辑/解释/知识/归因陷阱），支持细粒度模型错误归因"]
benchmarks: ["SocialReasonBench"]
---

# 论文速读：SocialReasonBench: A Video-QA Benchmark for Social Reasoning with Counterfactual Narrative Videos

## 一句话总结
本文提出 SocialReasonBench，一个基于互动叙事游戏《底特律：变人》视频片段的视频问答评测基准，利用分支剧情结构提供可验证的反事实与社会因果推理测试，共含 532 个视频-QA 实例，覆盖 7 个细粒度社会推理维度。实验表明当前主流 LMM 在基础社会理解任务上表现尚可，但在反事实推理与因果 antecedent 推理上显著掉点，且模型易依赖视觉捷径与外部先验。

## 研究问题与动机
- **核心问题**：现有大型多模态模型（LMM）在视频理解上已取得显著进展，但在以人为中心的社会情境中推断潜在社会动态（信念、意图、情感、人际动机）的能力仍然有限，缺乏对深层社会推理的可靠评测手段。
- **现有基准的局限（线性叙事）**：多数现有社会视频推理基准（如 TVQA、VLEP、Social-IQ）来源于电影或短视频，叙事沿单一固定轨迹展开，无法判断模型是真正理解社会动态，还是仅利用了重复的叙事模式。
- **缺乏可验证的真相标注**：现有基准的真实答案往往依赖事后人工标注，成本高且可能引入解释偏差；而本文利用游戏内流图、角色亲和度等游戏状态信号直接锚定答案标签，实现可核查的 ground truth。
- **反事实与社会因果推理的评测空白**：当前反事实推理基准多关注视觉变化或一般事件结果，极少探索替代决策如何影响人际关系与社会后果，本文通过互动叙事的分支结构提供自然配对的反事实结局。

## 核心贡献（创新点）
1. **提出 SocialReasonBench 视频 QA 基准**：基于互动叙事游戏《Detroit: Become Human》构建，利用分支剧情提供可观测的替代社会结果，覆盖 7 个社会推理维度（intent recognition、behavior prediction、emotional empathy、relationship dynamics、moral dilemma、counterfactual reasoning、causal antecedent），共 532 个实例。与已有工作本质区别：首次在游戏分支结构中构建社会推理评测，使反事实推理具备真实可实现的世界线，而非仅凭假设文本。
2. **设计多智能体自动化数据构建管线**：开发 Director Agent、Tracker Agent、Generator Agent 三阶段协作框架，实现从长视频到带诊断性干扰项 QA 对的自动抽取与标注；与人工标注基准的本质区别：答案标签由游戏状态信号（流图、UI 指标、脚本分支）直接锚定，无需依赖人工解释。
3. **构建基于社会认知理论的六陷阱诊断干扰项体系**：针对五种推理失败模式（视觉陷阱、逻辑陷阱、解释陷阱、知识陷阱、归因陷阱）设计诊断性错误选项，使模型错误可归因；与已有工作的本质区别：首次将社会认知理论中的具体推理错误类型化为结构化干扰项，支持细粒度误差诊断。
4. **发布全面的 LMM 评测与诊断分析**：在 8 个主流闭源/开源 LMM 上开展评测，揭示模型在基本社会理解与深层因果/反事实推理之间的系统性差距，并给出模态消融（音频缺失导致约 6% 准确率下降）和陷阱落点率（TFR）分析；本质区别：同时关注性能落差分布与错误归因机制，为模型缺陷分析提供细粒度工具。

## 方法详解
**任务形式化**：每个基准实例定义为 $\boldsymbol{x} = (v, b, q, \mathcal{C}, y)$，其中 $v$ 为视频片段，$b$ 为匿名化背景文本，$q$ 为社会推理问题，$\mathcal{C} = \{c_1, \ldots, c_6\}$ 为六个候选选项，$y$ 为正确答案。模型预测 $\hat{y} = f(v, b, q, \mathcal{C})$，以是否 $\hat{y} = y$ 评估正确率。正确答案 $y$ 由隐藏的游戏信号 $z$ 确定，模型不可见。

**游戏环境抽象**：将每章游戏流程图抽象为分支叙事图 $\mathcal{G} = (\mathcal{S}, \mathcal{A}, \mathcal{T})$，其中 $\mathcal{S}$ 为叙事状态、$\mathcal{A}$ 为可选动作、$\mathcal{T}: \mathcal{S} \times \mathcal{A} \to \mathcal{S}$ 为确定性转移函数。已实现轨迹 $\tau = (s_1, a_1, s_2, \ldots, a_{T-1}, s_T)$，未实现分支边构成反事实对比依据。

**多智能体合成框架**：
- **Director Agent（导演智能体）**：基于叙事引导的视频定位策略，识别关键叙事节点（分支选择点、游戏变量显式变化时刻如亲和度转移），为每段候选片段打上二维标签（社交交互类型 $\in \mathcal{I}$ × 认知推理能力 $\in \mathcal{R}$），并在目标节点前后各加 15 秒上下文缓冲生成最终片段 $v_i$。
- **Tracker Agent（追踪智能体）**：提取隐藏游戏信号 $z_i$，包括显式信号（UI 界面可见的亲和度/关系变化）和隐式信号（通过匹配已执行动作与流图预条件推导出的状态，如"低信任"）。
- **Generator Agent（生成智能体）**：根据覆盖标签 $(i, r) \in \mathcal{W}$ 合成问题 $q_i$、正确答案 $y_i$ 及候选选项集 $\mathcal{C}_i$。采用实体匿名化策略（如 "Connor" → "Agent Alpha"，"CyberLife" → "OmniCorp"）防止模型利用外部记忆先验。

**标签锚定规则**：
- 对于道德困境类问题，参考游戏内 "World's Stats" 特征中各选项的人类玩家选择比例，以最大比例选项作为参考标签（描述性事实而非规范性判断）。
- 其余问题的正确答案必须被游戏内信号 $z_i$ 所蕴含。
- 64.4% 的标签可通过直接查询 UI 证据或官方流图验证，35.6% 需要基于同一文档源做短程推导。

**诊断干扰项设计**：五种陷阱类型及其对应的推理失败模式——视觉陷阱（依赖显著视觉线索）、逻辑陷阱（因果幻觉/虚构未发生事件）、解释陷阱（夸大社会判断）、知识陷阱（引入外部规则先验）、归因陷阱（根本归因错误，将情境行为归因于内在人格特质）。

**问题分类 taxonomy**：二维正交框架 $\mathcal{W} = \mathcal{I} \times \mathcal{R}$，$\mathcal{I}$ 包含亲社会/对抗/策略/规范四类社交交互，$\mathcal{R}$ 包含七类推理能力，每类均有对应社会认知理论支撑（如 counterfactual reasoning 基于 Simulation Theory + Structural Causal Models）。

## 实验与结果
**数据集**：SocialReasonBench，共 532 个视频-QA 实例，每实例为一段中位时长 85 秒（IQR 70–115 秒）的游戏视频片段，总计约 15.5 小时；每问题六选项单选。100% 的视频实例同时附带去音频版本，支持模态消融实验。

**评测模型**：8 个代表模型——闭源：Gemini 3 Pro、Gemini 3 Flash、GPT-4o（2024-08）、Claude 3.5 Sonnet、GLM-4.6V-Flash；开源：Qwen3-Omni-30B、MiniCPM-o 2.6、ARC-Hunyuan-Video。全部 zero-shot 评测。

**评估指标**：主指标为准确率（Accuracy）；辅助指标为陷阱落点率 $\mathrm{TFR}(k) = \frac{\sum_i \mathbb{I}[\hat{y}^{(i)} \neq y^{(i)}] \cdot \mathbb{I}[\hat{y}^{(i)} \in \mathcal{T}_k^{(i)}]}{\sum_i \mathbb{I}[\hat{y}^{(i)} \neq y^{(i)}]}$，衡量模型在各类诊断陷阱上的错误分布。

**主要结果**（取自 Table 3，准确率 %）：

| 推理维度 | HE（人类） | Gemini 3 Pro | GPT-4o | Claude 3.5 | Gemini 3 Flash | GLM-4.6V | Qwen3-Omni | MiniCPM-o | ARC-Hunyuan |
|---|---|---|---|---|---|---|---|---|---|
| Intent Recognition | 93.24 | 83.78 | 85.14 | 82.43 | 79.73 | 75.68 | 72.97 | 60.81 | 24.32 |
| Behavior Prediction | 88.46 | 66.67 | 74.36 | 71.79 | 62.82 | 64.10 | 58.97 | 52.56 | 41.03 |
| Emotional Empathy | 90.83 | 70.00 | 76.67 | 73.33 | 66.67 | 68.33 | 61.67 | 55.83 | 36.67 |
| Relationship Dynamics | 89.36 | 82.98 | 55.32 | 52.13 | 73.40 | 38.30 | 32.98 | 23.40 | 25.53 |
| Moral Dilemma | 85.71 | 66.67 | 28.57 | 33.33 | 47.62 | 28.57 | 28.57 | 19.05 | 28.57 |
| Counterfactual Reasoning | 88.00 | 62.00 | 50.00 | 56.00 | 46.00 | 42.00 | 36.00 | 26.00 | 22.00 |
| Causal Antecedent | 86.32 | 70.53 | 55.79 | 58.95 | 64.21 | 38.95 | 37.89 | 27.37 | 30.53 |

- **最强模型**：Gemini 3 Pro 整体表现最优，在所有维度上均领先；GPT-4o 在 Intent Recognition 上达 85.14%，但在 Counterfactual Reasoning 上仅 50.00%，Moral Dilemma 仅 28.57%。
- **核心发现**：所有模型均呈现一致规律——基础社会理解维度（intent recognition、behavior prediction、emotional empathy）表现较好，而因果/反事实推理维度显著下滑；经 Holm 校正的单侧 Fisher 精确检验，六款模型的感知级 vs 因果/反事实推理性能差异显著（最大 adjusted $p = 0.045$）。
- **模态消融**：在 Gemini 3 Pro 上去除音频导致整体准确率下降约 6%，在 Causal Antecedent 和 Emotional Empathy 上下降最明显。
- **陷阱分析**：强模型（GPT-4o、Claude 3.5）更多落入知识陷阱和视觉陷阱（依赖外部常识/显著视觉线索）；小型开源模型更多落入逻辑陷阱（难以追踪时序依赖和分支状态转移）。

## 相关工作脉络
1. **视频理解基准**（MVBench、Video-MME、EgoSchema、LongVideoBench）：侧重动作识别、事件理解和时序推理等通用视频理解能力；SocialReasonBench 在此基础上引入社会情境的深层推理需求，聚焦意图、情感、道德和因果等隐性社会状态。
2. **多模态社会推理基准**（Social-IQ、Social-IQ 2.0、TVQA、VLEP、VisualCOMET、SIV-Bench、Social Genome）：均基于线性叙事视频，无法评测反事实社会后果；SocialReasonBench 的独特性在于利用互动叙事的分支结构，使替代社会结果成为可验证的事实。
3. **反事实推理基准**（CounterBench、What-if TV was off?、ACQUIRED、video generalization benchmark）：现有工作多关注视觉层面的反事实变化或泛化事件结果；SocialReasonBench 关注替代决策对人际关系和社会后果的影响，且反事实分支在游戏世界中真实发生过。
4. **拟人化/社交机器人社会推理**（HumanVideo-MME、MoMentS）：聚焦 theory-of-mind 和人类中心关系推理；SocialReasonBench 通过 android 主角间接考察社会推理能力，其游戏内可验证机制提供了更强的标签可靠性。
5. **视频游戏 AI 评测**（VideoGameBench、SIMA）：侧重于任务完成和 long-horizon 交互；SocialReasonBench 侧重社会推理质量而非任务绩效，关注模型能否遵循分支特定的因果证据而非视觉捷径。
6. **多智能体数据构建管线**：本文的多智能体策展框架（Director/Tracker/Generator）将理论指导、信号锚定和诊断性干扰项生成相结合，为高质量社会推理基准构建提供了可复用的方法论参考。

## 局限性与未来方向
- **单一来源游戏的局限**：基准仅基于《Detroit: Become Human》一款英语互动叙事游戏构建，反映特定文化、风格和叙事特征；android 主角与人类社会代理的推理仍存在一层间接性。未来可扩展至更多叙事驱动游戏以丰富社交场景。
- **策展模型家族优势**：三个策展智能体使用 Gemini 家族模型，而该家族模型同时参与评测，可能引入 generator-family 风格偏好；作者已验证排除 Gemini 后主要结论不变，但未来发布版本可进一步多样化策展模型或引入完全人工编写子集。
- **道德困境子集规模过小**：仅 21 个实例，作者明确表态仅作为参考性数字，不作能力结论。
- **输入模态不一致**：评测中不同模型接受的输入格式不同（原生视频 vs 32 帧抽帧），虽不影响模型内部跨维度比较的可靠性，但限制了模型间的绝对能力排序。

## 研究启发与可借鉴点
1. **多智能体数据构建管线的设计范式**：Director（场景选择）→ Tracker（信号锚定）→ Generator（QA 合成）的三段式分工，结合理论 taxonomy 引导，可作为构建高质量推理基准的通用工程框架，尤其适用于需要从复杂交互环境（游戏/仿真器/交互式媒体）中提取 QA 数据的场景。
2. **诊断性干扰项（Trap Design）的方法论价值**：将模型常见失败模式（视觉捷径、知识先验污染、因果幻觉、归因错误等）形式化为结构化干扰项体系，使错误分析可归因；这一思路可迁移至其他推理基准的评测设计，将单纯 accuracy 升级为多粒度误差诊断。
3. **游戏内信号作为可验证 ground truth 的标注策略**：利用交互环境内部的确定性状态变量（流图分支、UI 指标、脚本预条件）作为答案锚定源，而非纯人工标注，可大幅提升标签可靠性和可扩展性；该策略适用于其他具有良好状态追踪能力的模拟环境。
4. **实体匿名化防止数据污染**：将游戏特定实体映射为中性占位符（人物/组织/地点/物种概念），在保留语义结构的同时切断模型对原作内容的记忆先验；可借鉴于其他受版权保护的叙事内容基准构建。
5. **反事实推理的"已实现分支"范式**：与虚构反事实不同，利用游戏中实际发生过的替代剧情分支作为反事实对比，使反事实推理测试具备真实因果约束而非纯想象；这一思路可应用于其他支持多结局的叙事系统或可交互模拟环境。

## 关键术语表
- **SocialReasonBench**：本文提出的视频多项选择 QA 基准，用于评估 LMM 在互动叙事游戏分支剧情中的社会推理能力，含 532 个实例覆盖 7 个推理维度。
- **Counterfactual Reasoning（反事实推理）**：推断若采取替代行动或分支，社会结果将如何变化；本文通过游戏实际存在的替代分支提供可验证的反事实对照。
- **Causal Antecedent（因果前件推理）**：从已观察到的社会结果反向推断促成该结果的前置条件、决策或隐藏状态。
- **Diagnostic Distractor（诊断性干扰项）**：针对特定推理失败模式设计的 plausible 但错误的选项，用于将模型错误归因到具体认知漏洞类型。
- **Trap Fall Rate（TFR，陷阱落点率）**：衡量模型在错误预测中落入某类诊断陷阱的比例，公式为 $\mathrm{TFR}(k) = \frac{\sum_i \mathbb{I}[\hat{y}^{(i)} \neq y^{(i)}] \cdot \mathbb{I}[\hat{y}^{(i)} \in \mathcal{T}_k^{(i)}]}{\sum_i \mathbb{I}[\hat{y}^{(i)} \neq y^{(i)}]}$。
- **Multi-Agent Curation Pipeline（多智能体策展管线）**：由 Director Agent、Tracker Agent、Generator Agent 组成的自动化基准构建流程，分别负责视频定位、游戏信号锚定和 QA 合成。
- **Branching Narrative Graph（分支叙事图）**：抽象为 $\mathcal{G} = (\mathcal{S}, \mathcal{A}, \mathcal{T})$ 的图结构，描述游戏叙事状态、可选动作和确定性转移函数，支撑反事实推理的 formal 定义。
- **Two-Axis Task Taxonomy（二维任务分类法）**：$\mathcal{W} = \mathcal{I} \times \mathcal{R}$，横轴为社交交互类型（亲社会/对抗/策略/规范），纵轴为推理能力（7 类），每实例标注为 $(i, r)$ 坐标以实现覆盖率控制。

## 可复现要素
- **数据集**：SocialReasonBench 已公开发布，视频部分受游戏 ToU 约束（非商业研究用途），原始标注以 CC BY-NC-SA 4.0 发布。数据来源为 YouTube 公开游戏视频。
- **代码**：附录声明多智能体管线 prompt 全文随代码库一同发布；结构化 prompt 卡片见附录 L，完整 prompt 随 codebase 开源。
- **权重**：不涉及自训练模型，仅对现有 LMM 进行评测。
- **关键超参**：Director/Tracker Agent（Gemini 3.1 Pro Preview，temperature 0.1–0.2，结构化输出解码）；Generator Agent（Gemini 2.5 Pro，temperature 0.3，多模态视频输入）；评测时所有模型 temperature=0.0（GLM-4.6V-Flash 为 0.01），max output tokens=2048；关键帧输入模型采用 FFmpeg 均匀采样 32 帧。
- **环境**：游戏为《Detroit: Become Human》（Quantic Dream 开发，Sony Interactive Entertainment 发行）。
