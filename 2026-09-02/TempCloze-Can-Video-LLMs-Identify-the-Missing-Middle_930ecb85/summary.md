---
title: "TempCloze-Can-Video-LLMs-Identify-the-Missing-Middle"
source: https://arxiv.org/pdf/2609.01515v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 11:52:10"
field: "视频多模态大模型评测"
keywords: ["Video-LLM", "Temporal Reasoning", "Benchmark", "Visual Cloze", "Alignment", "Video Understanding"]
innovations: ["提出视频完形填空范式 TempCloze 以视觉片段比较替代文本选项，削弱语言捷径", "在同源视频内按 Semantic/Alignment/Progression 三维度构造干扰项并诊断视觉时间推理瓶颈", "系统性揭示 Alignment 为当前 Video-LLM 主要短板并提出候选顺序与上下文方向敏感性评测协议"]
benchmarks: ["TempCloze", "TempCloze-Mixed", "TempCloze-Hard"]
---

# 论文速读：TempCloze-Can-Video-LLMs-Identify-the-Missing-Middle

## 一句话总结
本文提出 TempCloze，一个以"视觉完形填空"形式评估 Video-LLM 视觉时间推理能力的基准：给定视频首尾片段，模型需从四个候选中间片段中选出正确的那个。基于对 31 个主流模型（10 个闭源 + 21 个开源）的评测，论文发现当前 Video-LLM 在**语义识别（Semantic）**和**事件进展（Progression）**上表现尚可，但**时间对齐（Alignment）**仍是主要瓶颈。

## 研究问题与动机
- **现有 benchmark 受语言干扰严重**：TempCompass、TVBench、VideoMME 等主流时间推理评测仍以文本选项或多选问答为主，模型可通过选项措辞、答案相关性或语言先验走捷径（如纯预训练语言模型在无视觉输入时即可超随机基线 25% 以上）。
- **缺少视觉化的时间推理评估**：缺乏让模型直接对比视觉片段而非文本描述的评测设置，难以区分"真正的视觉时间理解"与"语言模式匹配"。
- **三维度分解缺失**：已有工作很少将时间推理拆分为语义内容（what）、时间对齐（when）、进展顺序（how）三个独立维度进行诊断。
- **长镜头/连续动作场景覆盖不足**：现有基准多采用短片段剪辑或场景切换频繁的视频，缺乏针对长连续拍摄（long-take）、第一人称视角（egocentric）和细粒度动作（fine-grained motion）视频的时间推理评测。

## 核心贡献（创新点）
- **提出视频完形填空评测范式**：以"首尾可见 + 候选中间"的视觉完形设置替代文本问答，从根本上削弱语言捷径的影响。与 MovieFIB、FIBER 等文本完形工作的本质区别在于评估对象由自然语言描述改为视频片段。
- **三维度混淆项构造**：在同源视频内构造 Semantic、Alignment、Progression 三个维度的干扰项，迫使模型在保留场景与物体线索的前提下专注于时间证据判断；区别于仅靠随机裁剪或跨视频检索的噪声候选。
- **系统评测 31 个模型并定位瓶颈**：覆盖 10 个闭源 + 21 个开源 Video-LLM，首次量化指出 Alignment 是当前视觉时间推理的最主要短板（商业平均 48.13%，开源平均 26.54%），而语义（70.73%/34.00%）和进展（67.72%/36.97%）相对可控。
- **误差模式与行为敏感性分析**：在 TempCloze-Mixed 和 TempCloze-Hard 子集上展开候选顺序扰动、上下文方向消融、可见跨度/帧密度缩放及测试时 scaling（Pass@k）等分析，揭示稳定准确率可能掩盖不稳定选择、模型更依赖开头而非结尾等深层现象。

## 方法详解
- **任务形式化**：将视频 $V$ 划分为 $V=[C_{0,s} \mid C_{s,e} \mid C_{e,T}]=[B \mid M \mid E]$，模型观察 $B$ 与 $E$，从候选集 $\mathcal{V}_d=\{M, D_d^1, D_d^2, D_d^3\}$ 中选出填补 $(s,e)$ 的正确片段。
- **三维度混淆项定义**：
  - **Semantic（S）**：与目标等长但不重叠的片段 $D_S=C_{u,u+\ell}$，$(u,u+\ell)\cap(s,e)=\emptyset$，考查"该发生什么事件"。
  - **Alignment（A）**：在原区间附近偏移或扩展的片段 $D_A$（Advanced/Deferred 平移 $\ell/2$，Expanded 向两侧各扩 $\ell/2$），考查"事件应在何时发生"。
  - **Progression（P）**：对目标区间内部子事件顺序做扰动——Reversed（倒放）、Reordered（重排子段）、Repeated（重复短子段），考查"事件如何展开"。
- **数据构建流水线**：从 CaReBench、Daily-Omni、EgoLife、FAVOR-Bench、LVD-2M、MiraData、Video-TT 七源起步 → 时长 12–90s 过滤 → 使用 GPT-o3 评估 caption 是否适合完形 → 远心 50% 区域采样 gap（占总长 20%–40%） → Farnebäck 光流验证（平均幅度>1.0，至多重采 3 次） → 质量阈值（码率>200 kbps、清晰度>30） → 最终 1,521 个视频。
- **帧采样与防捷径校验**：采用 Bin-Centered Sampling（每段均匀分 n 个区间取中心点）避免端点泄露；验证 18,252 个候选片段的首尾 1–3 帧与相邻上下文精确匹配数为 0；Edge Baseline（DINOv2 相似度）均接近均匀分布 25%，证明不靠边界帧捷径。
- **辅助子集**：TempCloze-Mixed（300 视频、跨维度随机混合干扰项）与 TempCloze-Hard（150 误判最多的样本）。

## 实验与结果
- **数据集规模**：1,521 视频，来源分布 LVD-2M（515）、EgoLife（437）、MiraData（198）、FAVOR（145）、CaReBench（94）、Video-TT（89）、Daily-Omni（43）。
- **评测模型**：10 个闭源（Seed1.8、Qwen3.5-Plus、Gemini2.5-Pro/Flash、GPT5.4、Claude4.6-Sonnet/Opus、Gemini3-Flash、Seed1.6、Grok4.1）+ 21 个开源（Qwen3.5-397B-A17B、Qwen3.5-35B-A3B、KimiK2.5、Qwen3VL 系列、InternVL3.5/3 系列、KimiVL-A3B、LLaVA-CriticR1、GLM4.6V-Flash、ThinkLiteVL、MiMoVL、Molmo2 等）。
- **主要数字**：闭源均值 S=70.73%、A=48.13%、P=67.72%、Mean=62.19%；开源均值 S=34.00%、A=26.54%、P=36.97%、Mean=32.51%；人类基准 S=96%、A=98%、P=97%、Mean=97%；随机基线 25%。
- **最强结果**：闭源 Seed1.8 三维度 S=96.25%、A=61.93%、P=92.11%、3/3=56.94%（thinking 版）；开源 Qwen3.5-397B-A17B S=75.94%、A=50.62%、P=78.24%、3/3=35.77%。
- **关键结论**：Alignment 显著低于 S 与 P，且 S+A/A+P 联合准确率远低于 S+P（闭源 S+P 均值 53.67%，S+A 均值 40.54%，A+P 均值 37.39%）；Open-source 在 3/3 上差距尤为悬殊（7.15% vs. 33.24%）。
- **误差模式**：Alignment 错误以 Expanded 占主导（如 Seed1.8-I 的 75% Alignment 错误为 Expanded）；Progression 错误以 Reversed 为主（GPT-5.4 的 67%）。
- **敏感性结论**：候选顺序扰动下 Mean 稳定（方差<4%）但 CFR 高达 32%–61%；仅看开头优于仅看结尾（Seed1.6 Progression 80% vs. 46%）；扩展可见跨度/增加帧密度反而使 Alignment 下降；测试时 Pass@k 提升约 30 个百分点但不改变维度间排序。

## 相关工作脉络
- **Temporal Reasoning Benchmarks（TemporalBench、TVBench、TempCompass、LongVideoBench、MVBench、VideoMME）**：这些基准以视频 QA 为主，仍依赖文本选项或生成型描述，存在语言捷径风险；TempCloze 转向纯视觉片段比较，定位差异在于"去语言化"与"时间对齐精度检验"。
- **Cloze for Video Understanding（MovieFIB、FIBER）**：使用文本填空评估叙事理解；TempCloze 将其移植到视频完形，目标是评测视觉时间推理而非文本补全。
- **Self-supervised Video Pre-training using Cloze（VideoBERT、VCP、MaskFeat、VideoMAE）**：cloze 作为预训练目标学习表征；TempCloze 将其作为评测目标，关注下游推理能力而非表示学习。
- **VideoQA 与视觉 QA 偏差研究（TGIF-QA、NExT-QA、ActivityNet-QA、Xiao et al. 2024、Balepur et al. 2024）**：揭示了基于语言的选项偏差；TempCloze 通过同源四选一视频片段设计试图规避此类偏差。
- **Fine-grained Temporal / Motion Understanding（FAVOR-Bench、CaReBench、Video-TT）**：侧重动作时序与细粒度识别；TempCloze 在此基础上引入"首尾推理中间"的设置，强调时间连续性推断。

## 局限性与未来方向
- **评测范围受限**：专注"缺失中间识别"的诊断性任务，未覆盖开放式生成、对话式叙事理解、音频 grounding、无约束未来预测等更广泛的视频理解能力。
- **固定候选格式**：只能相对于设计好的干扰项作答，不代表自由-form 视频推理的泛化水平。
- **同源设计的严格性**：同场景同物体虽能压制外观捷径，但使任务偏向精确时间兼容而非广义事件合理性，部分 Alignment 干扰项仍含相关事件内容，属有意设计但仍可能高估人类实际表现。
- **领域覆盖不均**：来源集中于 long-take、egocentric、细粒度动作类视频，未包含所有域或镜头风格；不同 API/解码策略会随时间变化，结果为当前 snapshot。
- **未来方向**：引入视频原生架构以更好地利用密集时序证据；发展同时融合双向时间线索（前向续接+后向回溯）的机制；扩展数据来源与模态（如加入音频）；在更长视频中保持视觉时间推理的稳定性。

## 研究启发与可借鉴点
- **去语言化评测设计**：用视觉片段比较替代文本选项，可有效剥离语言先验/选项措辞带来的虚假高分，该方法可迁移至其他多模态推理 benchmark 的设计。
- **三维度分解诊断思路**：将时间推理拆分为 Semantic/Alignment/Progression 并进行独立计分与联合分析，为模型短板定位提供清晰的拆解框架；可借鉴用于其他时序理解任务的能力剖面绘制。
- **同源同场景混淆项构造**：在同一视频中截取等长/近邻/重排片段作为干扰，既保留外观一致性又放大时间差异，是一种高质量、低歧义的干扰项生成策略。
- **行为敏感性测试套件**：候选顺序扰动（FR/CFR）、上下文方向消融（仅开头/仅结尾/双向）、可见跨度与帧密度缩放、Pass@k 测试时扩展等实验设计，可作为模型稳健性评测的标准流程。
- **防捷径 sanity check 规范**：端点帧精确匹配检测、Edge Baseline（DINOv2 相似度归一化）、单帧对比基线等方法，可作为视频 benchmark 中排除"边界泄露"和"单帧捷径"的参考模板。

## 关键术语表
- **TempCloze**：以"视频完形填空"形式评估 Video-LLM 视觉时间推理能力的基准，给定首尾片段要求从四个候选中间片段中选真 middle。
- **Semantic Distractor**：与目标等长但时间区间不重叠的同源片段，用于测试模型对"应发生何事件"的判断。
- **Alignment Distractor**：在目标区间附近偏移或扩展的片段（Advanced/Deferred/Expanded），用于测试模型对"事件应在何时发生"的敏感度。
- **Progression Distractor**：对目标区间内部做倒放、重排子段或重复片段的变体，用于测试模型对"事件如何展开"的细粒度感知。
- **Bin-Centered Sampling**：将视频片段等分区间后取每个区间中心采样帧，避免首尾端点泄露的帧采样策略。
- **Pass@k**：在温度 0.7 下多次采样，只要有一次选对即视为正确，用于衡量测试时扩展带来的性能上界。
- **Temporal Cloze Suitability**：使用 reasoning LLM（GPT-o3）基于 caption 判断视频是否具备清晰可视动作序列、可被首尾约束出一个合理中间段的质量筛选标准。
- **Cumulative Accuracy**：统计每个视频上至少正确 1/2/3 个维度的比例，用于衡量视频级整体理解程度。

## 可复现要素
- **数据集**：1,521 视频来自七个公开数据源（CaReBench、Daily-Omni、EgoLife、FAVOR-Bench、LVD-2M、MiraData、Video-TT），论文未声明 TempCloze 本身作为独立数据集的开源状态；辅助子集 TempCloze-Mixed（300）与 TempCloze-Hard（150）亦未单独发布声明。
- **代码/权重**：评测脚本与提示词（Appendix G）给出；模型权重均为已有开源/闭源模型（Qwen3.5、KimiK2.5、Qwen3VL、InternVL3.5、LLaVA-CriticR1、GLM4.6V、MiMoVL、Molmo2 等），未新增模型权重。
- **关键超参**：均匀采样 16 帧/片段（每维度 6 个片段共 96 帧）；视频时长 12–90s；gap 长度占总长 20%–40%、位于中段 50%；Farnebäck 光流平均幅度>1.0；码率>200 kbps、清晰度>30；temperature 0（主实验）/0.7（Pass@k）；Pass@k 最多 k=5。
