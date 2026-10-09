---
title: "Mid-Training-Language-Models-on-Raw-Video"
source: https://arxiv.org/pdf/2610.11019v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 10:05:42"
field: "多模态学习"
keywords: ["多模态大模型", "中期训练", "自监督学习", "视频理解", "原始视频", "视觉token预测", "语言模型"]
innovations: ["提出原始视频无标注中期训练框架，通过预测连续视觉token提升多模态能力", "证明纯自监督视觉预测优于字幕预测，消除昂贵标注管道", "揭示视觉增益在30%训练量后plateau的高效训练规律"]
benchmarks: ["EgoSchema", "NExT-QA", "VideoMME", "TempCompass", "ChartQA", "DocVQA", "MMBench", "AI2D", "MathVista", "MMStar"]
---

# 论文速读：Mid-Training-Language-Models-on-Raw-Video

## 一句话总结
本文首次探索用原始网络视频（无字幕、无文本损失）对预训练语言模型进行中阶段训练，通过将视频帧编码为连续视觉token序列并让模型预测下一个视觉token，在仅消耗64.4B视觉token的情况下，模型视频理解平均提升2.9分、图像理解提升5.1分，且文本能力得以保持。

## 研究问题与动机
1. **人类文本储量趋近瓶颈**：公共人类文本预计将在十年内耗尽，重复训练文本收益递减，需要探索非文本模态（如视频）来支撑持续Scaling。
2. **视频是最大数据来源但未被充分利用**：网络视频存量约1350万亿文本token等价物，是已索引网页数据的2.6倍，且帧序列天然适合自回归建模。
3. **现有方法依赖昂贵的字幕管道**：当前多模态模型主要通过对帧生成字幕或QA标注来利用视频，但逐帧字幕成本高且引入标注噪声。
4. **预训练语言模型能否直接从原始视频学习**：尚未有工作系统研究用无标注原始视频对中阶段训练已有LLM的有效性与代价。

## 核心贡献（创新点）
1. **提出原始视频中期训练框架**：通过预测连续视觉token实现自监督视频学习，无需任何字幕或人工标注。与Cambrian-S等需额外预测头或离散codebook的方法本质不同，该方法直接复用LLM因果架构。
2. **证明视频中期训练可显著提升跨模态能力**：在4个视频基准上平均+2.9分（EgoSchema +4.8分），在10个图像基准上平均+5.1分（ChartQA +8.2分、DocVQA +6.4分）。与仅做instruction tuning的基线形成鲜明对比。
3. **揭示纯自监督视觉预测优于字幕预测**：预测下一帧字幕反而使文本性能下降1.5分，而直接预测视觉token在保持文本能力（48.9 vs 48.0）的同时获得更好视觉性能，消除了昂贵字幕生成管道的必要性。
4. **验证视觉能力在30%训练量后趋于饱和**：图像和视频增益在前0.3 epoch（19.5B token）即出现，后续变化不足0.5分，为高效训练提供了理论依据。

## 方法详解
**架构设计**：
- 使用SigLIP SO400M/14视觉编码器（384×384分辨率）提取每帧特征
- 通过两层GELU MLP投影器将视觉token映射到Qwen3-1.7B的2048维语言embedding空间
- 每帧内2D patch grid按行主序序列化，行间插入学习到的换行embedding保持空间结构可恢复
- 16帧序列 temporal concatenation 形成连续token流 $V \in \mathbb{R}^{L \times d}$

**自监督目标**：
- 在语言模型hidden state $h_i = f_\theta(v_{1:i})$ 上接两层MLP预测头 $r_\omega$
- 预测目标 $\hat{v}_{i+1} = r_\omega(h_i)$ 与真实token $v_{i+1}$ 的余弦距离
- 对目标应用stop-gradient：$\mathcal{L} = \frac{1}{|L-1|}\sum_{i=1}^{L-1}(1 - \frac{h_i^\top v_{i+1}}{\|h_i\|\|v_{i+1}\|})$
- 梯度仅通过因果预测路径传播，目标作为固定回归基准

**训练配置**：
- 1 epoch（27,300 steps），effective batch size=256 clips，context window=16k tokens
- 语言模型学习率1e-5，视觉编码器2e-6，cosine decay + 3% warmup
- 128× NVIDIA H100 GPUs，端到端微调所有参数 $(\phi, \psi, \theta, \omega)$

## 实验与结果
**数据集**：
- 中期训练：YT-Temporal-1B子集，6.99M clips，64.4B visual tokens
- 指令微调：LLaVA-OneVision-Data，3.9M images，3.2B tokens
- 基线模型：Qwen3-1.7B

**视频理解基准**（Table 2）：
| 基准 | No Mid-Training | Video Mid-Training | 提升 |
|------|-----------------|-------------------|------|
| EgoSchema | 45.20% | **50.00%** | +4.8 |
| NExT-QA | 59.04% | 62.18% | +3.14 |
| VideoMME | 42.85% | 45.15% | +2.3 |
| TempCompass | 51.58% | 53.04% | +1.46 |
| **Average** | 49.67% | **52.59%** | **+2.92** |

**图像理解基准**（Table 1）：
| 基准 | No Mid-Training | Video Mid-Training | 提升 |
|------|-----------------|-------------------|------|
| ChartQA | 43.76% | **51.96%** | +8.20 |
| DocVQA | 32.13% | 38.57% | +6.44 |
| AI2D | 66.39% | 72.51% | +6.12 |
| MMBench | 76.00% | 79.46% | +3.46 |
| MathVista | 58.15% | 62.59% | +4.44 |
| **Average** | 48.72% | **53.80%** | **+5.08** |

**文本基准**（Table 3）：
- 视频中期训练后文本平均48.9 vs 基线48.0，仅+0.9分
- GSM8K下降最小（54.77% vs 49.92%），BBH下降9.5分（vs 17.7分），表明视频训练缓解了multimodal instruction tuning对推理能力的损害

**关键发现**：
- 最强结果：EgoSchema +4.8分（长程第一人称理解），ChartQA +8.2分（文本密集图像理解）
- 增益在0.3 epoch（19.5B token）时已捕获90%以上，后续plateau

## 相关工作脉络
1. **自监督视频表示学习**（VideoMAE, V-JEPA 2）：从头在视觉域训练，语言仅post-hoc接入；本文研究已有LLM直接从连续视频流学习的机制差异。
2. **端到端多模态预训练**（Emu3, Cambrian-S）：从零训练统一transformer或使用离散codebook；本文仅对已有LLM做中期训练，无需重新设计架构。
3. **视频作为LLM预训练数据**（Tong et al. 2026）：联合text+video从头预训练，保留显式text LM loss；本文在纯自监督下仅预测visual token，无text supervision。
4. **世界模型**（Genie, NVIDIA Cosmos）：学习未来帧预测用于规划/控制；本文聚焦于提升LLM的理解能力而非生成能力。
5. **多模态中期训练**（LLaVA-OneVision-2等）：依赖合成clip字幕监督；本文证明字幕非必需，纯视觉预测已足够。

## 局限性与未来方向
1. **时序结构的因果性未验证**：无法区分增益来自时序结构还是单纯更多视觉token exposure，需future frame-shuffling实验验证。
2. **模型规模限制**：仅在1.7B参数模型上验证，更大模型（如7B+）的收益未知。
3. **训练效率未深入优化**：虽30%训练即plateau，但 denser temporal sampling（>0.2 fps）或更长序列horizon是否有益未探索。
4. **仅覆盖YouTube视频**：YT-Temporal-1B来源单一，不同领域视频（如教育、新闻）的泛化性待验证。
5. **未探索与其他中期训练策略的组合**：如同时加入text reconstruction或mixed modality objectives的效果。

## 研究启发与可借鉴点
1. **自监督视觉预测作为通用中期训练策略**：该方法可迁移到其他预训练语言模型（如Llama、DeepSeek），无需重构整个训练pipeline。
2. **Stop-gradient在对比学习中的应用**：借鉴V-JEPA 2设计，目标表示不反传梯度可防止representation collapse，值得在多模态训练中复用。
3. **缓解multimodal instruction tuning的认知损伤**：视频中期训练使GSM8K/BBH下降减小约50%，为多模态训练中保持推理能力提供新途径。
4. **早期增益plateau现象的资源优化价值**：证实30%训练量即捕获大部分收益，可指导高效训练调度（如early stopping、curriculum design）。
5. **无需字幕的大规模数据利用范式**：方法论可推广到其他高成本标注场景（如医疗视频、工业质检），降低多模态数据获取门槛。

## 关键术语表
**Mid-Training**：介于预训练与指令微调之间的中间训练阶段，用于桥接领域特定表示与下游任务。
**Visual Token**：通过视觉编码器提取的连续特征向量，直接嵌入语言模型空间，无需离散化。
**Stop-Gradient (sg)**：阻止梯度反向传播的操作符，在对比学习中用于稳定训练并防止表示坍缩。
**Causal Language Model**：仅依赖历史信息的自回归模型架构，适用于序列预测任务。
**Yt-Temporal-1B**：包含约2000万YouTube视频的时态理解数据集，本文采样其6.99M clips作为中期训练源。
**EgoSchema**：长程第一人称视频理解诊断基准，测试模型对日常活动中物体交互与动作序列的理解。
**Cosine Distance Loss**：$\mathcal{D}(u,v)=1-\frac{u^\top v}{\|u\|\|v\|}$，衡量预测与目标向量方向相似性的距离度量。
**Instruction Tuning**：在多模态配对数据上微调模型以遵循自然语言指令的训练范式。

## 可复现要素
- **数据集**：YT-Temporal-1B（公开）+ LLaVA-OneVision-Data（公开）
- **代码**：项目页面 https://jd730.github.io/projects/RawVideoMidTraining，论文未提供GitHub链接
- **权重**：未开源，仅报告性能数字
- **关键超参**：batch size=256 clips，lr_lm=1e-5，lr_vision=2e-6，epochs=1（video mid-training）+1（SFT），H100×128
- **框架依赖**：PyTorch（隐含），Qwen3-1.7B，SigLIP SO400M/14
