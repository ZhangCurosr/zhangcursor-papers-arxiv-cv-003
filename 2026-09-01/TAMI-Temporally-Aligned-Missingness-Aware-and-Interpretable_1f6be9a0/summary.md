---
title: "TAMI-Temporally-Aligned-Missingness-Aware-and-Interpretable"
source: https://arxiv.org/pdf/2608.30857v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 16:37:50"
field: "多模态心理健康数字生物标志物"
keywords: ["multimodal fusion", "temporal alignment", "mental health", "mild cognitive impairment", "missingness-aware", "interpretable AI", "depression detection", "anxiety screening"]
innovations: ["自适应时间分箱实现跨模态细粒度时间对齐，消除序列索引配对导致的时间错位", "模态-时间步缺失掩码显式编码遥测缺失，区分零填充与有效测量", "问题上下文在每时间分箱联合所有模态条件化融合"]
benchmarks: ["AUROC (depression/anxiety)", "AUPRC", "Macro-F1"]
---

# 论文速读：TAMI-Temporally-Aligned-Missingness-Aware-and-Interpretable

## 一句话总结
本文提出 **TAMI** 框架，通过对远程临床访谈中的多模态特征进行**时间对齐、缺失感知与问题上下文条件化**，在伴有轻度认知障碍（MCI）的老年人中检测抑郁与焦虑；其中**细粒度时间对齐**是性能提升的最大来源，仅用开放式问题片段（中位时长 5.1 分钟）即可实现与全访谈相当（AUROC = 0.67）的抑郁筛查效果。

## 研究问题与动机
1. **现有方法存在跨模态时间错位**：多模态特征在不同时间分辨率下提取（帧级 vs. 窗级 vs. 词级），已有工作采用序列索引配对（sequence-index pairing）将其对齐，导致跨模态行为关联在时间上错位最高可达 22 秒，远超人类感知跨模态同步的时间窗口（200 ms）。
2. **远程录制中 modality 级随时间缺失严重，零填充无法区分真缺失与有效近零值**：参与者设备/网络条件差异大，部分时间步特征完全缺失，而已有方法多将缺失置零，使模型无法辨别"无信号"与"低信号行为"。
3. **缺乏对预测结果的多粒度可解释归因**：现有工作仅关注模态级重要性，未联合归因到"哪些模态 → 哪些问题 → 哪些时刻"，限制了临床信任与访谈协议优化。
4. **MCI 老年人抑郁/焦虑的远程多模态筛查研究相对匮乏**：已有相关研究多针对普通成年人群（如 DAIC-WOZ），在 MCI 老年群体中同时利用面部、声学、语言与 rPPG 生理特征的工作尚未充分探索。

## 核心贡献（创新点）
1. **自适应时间分箱（Adaptive Temporal Binning）实现跨模态细粒度时间对齐**：以问-答片段为基础构建共享时间轴，帧级特征按时间戳分配、窗级特征覆盖重叠分箱后平均，消除序列索引配对造成的时间错位；与既有工作的本质区别在于对齐基准是原始录音时间而非序列位置。
2. **模态-时间步缺失掩码（Modality-Timestep Missingness Mask）**：每个分箱记录每模态的二进制可用性标志，将零填充缺失与有效近零测量显式区分；与已有零填充策略的本质区别是缺失信息被编码为独立信号而非隐式处理。
3. **问题上下文条件化融合（Question-Conditioned Fusion）**：将 Sentence-BERT 编码的问题嵌入加至每个时间分箱的融合 token 上，使模型在每个回答时刻都能结合提问语义；与已有工作仅对文本模态单独条件化或仅在整体级别条件化的本质区别是在细粒度时间分辨率上与全部模态联合条件化。
4. **基于 Integrated Gradients 的多级可解释性分析框架**：从特征级归因聚合到模态级、问题级和全访谈时间轴级三个临床可解释维度；与已有工作仅做模态级或单一时间窗可视化的本质区别是实现"模态 × 问题 × 时间"三维联合归因。

## 方法详解
**问题形式化**：每位参与者 $i$ 的访谈表示为问-答片段序列 $\mathcal{X}_i = \{(q_{ij}, a_{ij})\}_{j=1}^{J_i}$，第 $j$ 个回答时长 $L_{ij}$ 被划分为 $B_{ij}$ 个时间分箱，构建共享时间轴。每模态-特征 $m$ 在每个分箱 $k$ 对应特征向量 $\mathbf{x}_{ijk}^{(m)} \in \mathbb{R}^{D_m}$ 和二进制可用性掩码 $M_{ijk}^{(m)} \in \{0,1\}$。

**时间对齐表示**：
- 特征提取（共 8 路模态-特征）：帧级（1 fps）包括 AUs & LMs（$D=155$）、head pose（$D=3$）、eyegaze（$D=2$）；窗级（2s 窗/1s hop）包括 eGeMAPS（$D=88$）、ComParE（$D=6373$→PCA 降至 512）、Wav2Vec2（$D=1024$）；rPPG 心率（6s 窗/1s hop，$D=1$）；语言用 WhisperX 词级转写 + RoBERTa 编码为 bin-level 向量（$D=768$）。
- 自适应时间分箱：以 1s 为默认箱宽，$B_{ij} = \text{clamp}(\text{round}(L_{ij}/w) + 1, B_{\min}=1, B_{\max}=64)$，超长回答按比例加宽箱宽而不截断。
- 特征分配：帧级按时间戳落入对应分箱；窗级落入所有与之重叠的分箱，同一分箱内多窗口向量平均；词级按 WhisperX 时间戳分配后合并编码。
- 缺失掩码：当某分箱内无该模态有效观测时设 $M=0$，特征向量零填充；$M=1$ 表示有效观测（即使值为零）。

**问题条件化融合与预测**：
- 两种投影骨干网络：Linear 直接将每模态特征投影到维度 $d$（$d=128$）并乘以掩码门控；Non-linear 使用带时间自注意力（掩码作为 attention mask）的单层 Transformer $f_m$ 处理。
- 融合：将投影后的 $\hat{\mathbf{x}}_{i,s}^{(m)}$ 与各模态掩码拼接后投影为融合 token $\mathbf{z}_{i,s} = W_f[\hat{\mathbf{x}}_{i,s}^{(1)};\dots;\hat{\mathbf{x}}_{i,s}^{(N_m)}; M_{i,s}^{(1)};\dots;M_{i,s}^{(N_m)}]$。
- 问题条件化：对第 $j$ 个问题回答内的每个分箱 $s$，$\tilde{\mathbf{z}}_{i,s} = \mathbf{z}_{i,s} + W_q \mathbf{q}_{ij}$，其中 $\mathbf{q}_{ij}$ 为 Sentence-BERT 编码的问题嵌入。
- 预测：融合序列经 [CLS] token + 双层 Transformer Encoder 编码，线性头输出 logits，二元交叉熵损失，类平衡 batch sampler，Adam(lr=0.001)，10 epochs。

**多级可解释归因**：采用 IG（Integrated Gradients）计算每个特征在每个分箱的归因，聚合到四个临床可解释维度：模态重要性 $I_m$、问题重要性 $I_q$、问题内模态重要性 $H_{q,m}^{\text{rel}}$、访谈时间轴重要性 $R_u$（将访谈归一化为 20 个等宽桶）。

## 实验与结果
**数据集**：Emory University CEP 项目招募的 49 名 MCI 老年人（平均年龄 73.4±8.1 岁，47% 女性），通过 Zoom 完成半结构化访谈（参与者侧 21–67 分钟，中位 31 分钟）；抑郁定义为 GDS≥10（阳性 21 例），焦虑定义为 GAD-7≥5（阳性 17 例/48 人）。

**评估协议**：参与者无关的 5-fold 交叉验证 × 3 次重复（共 15 次运行），主指标 AUROC + 95% CI，配对 t 检验（$p<0.05$）。

**主要结果（Table I）**：

| 变体 | 抑郁 AUROC | 焦虑 AUROC |
|---|---|---|
| Linear baseline | 0.51±0.06 | 0.58±0.03 |
| Non-linear baseline | 0.58±0.11 | 0.56±0.06 |
| **Linear + T** | 0.64±0.08* | **0.69±0.09**¶ |
| **Non-linear + T** | **0.68±0.04** | 0.58±0.07 |
| Linear + Q+T | 0.67±0.10† | 0.67±0.07‡ |
| Non-linear + Q+T | 0.67±0.07‡ | 0.61±0.01 |

- 时间对齐（T）带来最大提升：抑郁 Δ=+0.10（Non-linear），焦虑 Δ=+0.11（Linear, $p≈0.05$）。
- 添加问题条件化（Q）在已对齐基础上无显著增益（$p>0.05$）。
- 缺失掩码对焦虑有提升（Linear: 0.69 vs. 0.64），对抑郁影响较小且方向不一致（Table II）。

**可解释性发现**：
- 抑郁：eyegaze 贡献 66%，ComParE 11.2%；焦虑：eyegaze 44%，head pose 37.2%。
- GDS/GAD-7 筛选题组并非归因最高的问题群，最高归因集中在开放式问题、社会支持与自评健康相关题目。
- 抑郁归因集中在访谈早期（开放式问题段），焦虑归因均匀分布。
- **开放式问题 alone（中位 5.1 min）抑郁 AUROC=0.67，与全访谈 0.68 无显著差异（$p>0.05$）**；焦虑则显著下降至 0.59（$p≈0.02$）。

## 相关工作脉络
1. **Zhang & Poellabauer [25] (Mitigating Interviewer Bias, EMNLP 2025)**：将声学响应时序和文本时序 temporal pooling 为单向量后与问题做 gated addition 融合——丢弃了细粒度时序信息；TAMI 保留每分箱的细粒度时序并联合所有模态进行条件化。
2. **Niu et al. [34] (HCAG, ICASSP 2021)**：用 additive attention 分别对音频和文本序列做问题条件化后再融合——条件化仅作用于部分模态且不在融合后联合进行；TAMI 将问题嵌入加到所有模态融合后的每时间步 token 上。
3. **Guo et al. [23]**：仅对文本模态用 RoBERTa 拼接问题进行条件化，声学响应独立编码后融合；TAMI 对所有八路模态-特征联合条件化。
4. **Fan et al. [26] / Tao et al. [27] (DepMSTAT)**：采用序列索引配对对齐多模态特征——导致跨模态时间错位；TAMI 的自适应时间分箱按原始录音时间对齐，消除该问题。
5. **Gimeno-Gomez et al. [28]**：用 per-frame presence mask 作为 attention mask 进行抑郁预测，但未纳入问题结构且仅限单个短时间窗可视化；TAMI 将 mask 置于 answer-aligned 时间分箱上，并在完整访谈上做全量 IG 归因。
6. **Jiang et al. [20] / Mu et al. [21]**：MCI 老年群体远程多模态分析的前置工作，但未引入时间对齐、缺失感知掩码及问题条件化融合；TAMI 在此基础上补足三项关键设计。

## 局限性与未来方向
1. **样本量小（N=49）**，统计检验力有限，部分差异未达显著性，需在更大多样本队列中验证。
2. **面部与声学特征提取器基于普通人群预训练**，在 MCI 老年人远程录制的低质量数据上可靠性可能下降。
3. 访谈中问-答时间戳由人工标注，部署时需依赖 WhisperX 自动对齐；目前仅验证了手动标注版本。
4. 开放式问题可单独用于抑郁筛查的发现**需要在大样本中进一步验证**，且不适用于焦虑（焦虑需全访谈信息）。
5. 未来可在 neuropsychiatric 共病人群（如痴呆、精神分裂症）中拓展验证，并探索自动化问-答分割。

## 研究启发与可借鉴点
1. **时间对齐优先于特征复杂度**：在跨模态时间序列任务中，将不同采样率的特征按原始时间对齐（而非序列位置对齐）本身即可带来最大性能提升；这对语音-视觉-生理信号融合任务具有直接迁移价值。
2. **缺失掩码的有效性取决于缺失是否与标签相关**：本文通过 Pearson 相关分析发现 eyegaze/head pose 缺失与焦虑标签存在弱相关（$|r|≈0.18–0.19$），而与抑郁标签几乎无关（$|r|=0.014$），解释了掩码效果的差异性——提示后续工作应显式检验缺失信息的预测价值。
3. **开放式问题作为高效筛查窗口的发现**：仅用 5.1 分钟开放式回答即可达到与全访谈无显著差异的抑郁检测性能，为临床访谈协议设计提供了实证依据，可迁移至其他精神健康远程筛查场景。
4. **多级 IG 归因聚合方案的设计细节值得复用**：论文在 Supplementary 中给出了完整的归因归一化与聚合公式（按模态观测数归一化、按回答长度归一化、按相对访谈位置分桶），解决了跨参与者访谈长度不一带来的偏差问题，可直接应用于其他纵向多模态解释性研究。
5. **Linear vs. Non-linear 投影骨干的交替表现**：Depression 最佳为 Non-linear+T，Anxiety 最佳为 Linear+T，说明两种架构各有适用场景，未来研究可探索任务自适应或 ensemble 策略。

## 关键术语表
**Mild Cognitive Impairment (MCI)**：介于正常老化与痴呆之间的临床状态，以超出预期的认知下降为特征，但日常功能基本保留，是抑郁症和焦虑症高发人群。

**Temporal Alignment / Adaptive Temporal Binning**：将不同时序分辨率的多模态特征映射到基于问-答片段的共享时间分箱上，以原始录音时间而非序列索引对齐特征。

**Modality-Timestep Missingness Mask**：每个分箱中每路特征的 0/1 可用性标志，显式区分"无信号"与"有效近零测量"，拼接入融合 token。

**Question-Conditioned Fusion**：将问题嵌入在每个时间分箱的融合 token 上进行加法注入，使模型在回答的每个时刻结合提问语义进行联合建模。

**Integrated Gradients (IG)**：基于公理化的深度网络归因方法，通过沿零基线到输入的积分路径计算各输入特征对预测的贡献值。

**Sequence-Index Pairing**：已有工作将多模态特征按序列位置一一配对的朴素对齐方式，不反映原始录音时间关系，导致跨模态时间错位。

**rPPG (remote Photoplethysmography)**：通过视频面部区域提取心跳等生理信号的远程光容积脉搏波技术。

**WhisperX**：支持词级时间戳标注的自动语音识别系统，用于生成分箱级别的语言特征。

## 可复现要素
- **数据集**：来自 Emory University CEP 项目的 49 名 MCI 老年人远程访谈视频；**论文未声明公开**（含 IRB 伦理审批号 #2025P012652）。
- **代码**：论文未声明开源。
- **关键超参**：hidden dimension $d=128$；$B_{\min}=1, B_{\max}=64$；箱宽默认 $w=1\text{s}$；Transformer Encoder 2 层；Non-linear backbone 每模态 1 层 Transformer；Adam lr=0.001；batch size=4；10 epochs；类平衡采样器。
- **评估协议**：5-fold × 3 次重复的参与者无关交叉验证，AUROC + 95% CI，one-sided paired t-test。
