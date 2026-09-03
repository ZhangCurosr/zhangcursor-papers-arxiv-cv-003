---
title: "The-Visual-Insensitivity-Gap-Diagnosing-When-Vision-Language"
source: https://arxiv.org/pdf/2609.00868v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 00:49:12"
field: "多模态大模型可解释性与评测"
keywords: ["Vision-Language Models", "Visual Insensitivity", "Hallucination Diagnosis", "Selective Generation", "Input Intervention", "Encoder-Decoder Disconnect"]
innovations: ["提出VSI指标量化per-sample视觉使用情况并发现40%-97%样本存在视觉忽略", "证明视觉不敏感性是样本固有属性而非模型特有（跨6 VLM，Spearman ρ=+0.40）", "建立编码器-LLM断开连接的因果证据（probe acc 0.72-0.79 vs ∆argmax 2%-11%, gap>0.65）"]
benchmarks: ["POPE", "MMVP", "HallusionBench", "MMStar"]
---

# 论文速读：The-Visual-Insensitivity-Gap-Diagnosing-When-Vision-Language

## 一句话总结
论文提出"视觉不敏感性差距"（Visual Insensitivity Gap）概念，通过视觉敏感性指数（VSI）量化每个样本的视觉使用情况，发现40%–97%的VLM样本存在模型忽略视觉证据却输出高置信度答案的现象；VSI在多选推理任务上可达AUROC=0.85–0.87，但并非万能信号，最佳用法是与softmax置信度联合的条件集成组件。

## 研究问题与动机
- 现有VLM评测依赖多模态基准的聚合准确率，隐式假设模型会利用视觉输入，但实际上大规模样本中模型忽略视觉证据
- 40%–97%的感知基准样本在被扰动问题相关视觉区域后，模型softmax分布几乎不变，说明模型"自信地无视视觉证据"
- "自信错误但忽略视觉"比"谨慎但正确利用视觉"更具隐蔽性，在临床分诊、文档问答、具身代理等决策关键场景中危害更大
- 现有选择性生成信号（softmax max-prob、verbalised confidence）仅基于输出分布，无法区分"利用视觉证据的高置信度预测"与"忽略视觉证据的高置信度预测"

## 核心贡献（创新点）
- 提出视觉敏感性指数（VSI），以KL散度衡量问题相关区域模糊前后next-token分布的变化，实现per-sample级别的视觉使用量化；与现有全图扰动或单一argmax变更指标不同，VSI关注的是问题定位后的分布级扰动响应。
- 证明视觉不敏感性是样本固有属性而非模型特有：跨6个VLM（含三个不同架构族）的VSI排名Spearman ρ=+0.40（permutation p<10⁻³），跨族对共享的只有对比预训练视觉塔；这与模型内省式诊断的本质差异在于，它揭示的是样本层面的结构性盲点而非架构缺陷。
- 建立编码器–LLM断开连接的机制证据：在低VSI样本上，每模型自有视觉塔的线性探针准确率0.72–0.79，但argmax变化率仅2%–11%，形成>0.65的gap；这一输入–输出层面的因果诊断比仅做内部探针或注意力可视化更深入一层。
- 映射VSI的诊断效用细胞图谱：多选推理（MMStar math/science）AUROC=0.85–0.87最强，感知类事实性基准（POPE、HallusionBench）提升有限；VSI不是通用最优信号，而是有条件适用的集成组件。
- 构建18个(model, benchmark)单元格的系统性对照实验网格（6 VLM × 3感知基准 + MMStar），并公开全部per-sample数据与交叉模型Spearman排列零分布。

## 方法详解
- **VSI定义**：给定模型f、图像x和问题q，定位问题相关视觉区域（解析名词短语→Grounding-DINO框→SAM掩码），施加高斯模糊σ=20得到x_σ，计算KL散度：VSI(x, q; f) = KL(f(·|x,q) ‖ f(·|x_σ,q))，使用top-50 token对齐分布；鲁棒性测试覆盖σ∈{10,20,40}。
- **编码器–LLM gap测量**：对每模型自有视觉塔最后层特征做L2正则化逻辑回归线性探针（C=1.0），5-fold分组交叉验证，标签为扰动(+1)/未扰动(-1)；同时统计同一子集上argmax token变化率；Gap = probe准确率 − ∆argmax。
- **参考CLIP探针**：使用独立于六个VLM的frozen openai/clip-vit-large-patch14（224px）作为外部见证者，在同一样本池上验证表示可用性与模型是否使用无关。
- **四象限分解**：按底VSI五分化样本，再按正确/错误与softmax置信度高于/低于单元格中位数划分，提取"自信错误+忽略视觉"危险象限。
- **信号集成**：对region-VSI与whole-image-VSI做z-score等权相加得hyb(r+w)，并与max-prob、verbalised confidence组成完整信号集比较AUROC与PRR@80。

## 实验与结果
- **模型与基准**：6个VLM（LLaVA-1.5-7B、LLaVA-NeXT-Mistral-7B、Idefics3-8B、Qwen3-VL-8B、Qwen2.5-VL-7B、Qwen2.5-VL-32B）；3个感知基准（POPE、MMVP、HallusionBench）+ 1个多选推理基准（MMStar）。
- **VSI分布**：每单元格heavy left tail，VSI<0.05样本占比40%–97%（HallusionBench最高、POPE最低）；全图模糊变体仅~6% insensitive，排除任意扰动假阳性。
- **跨模型一致性**：15对模型两两极Spearman grand-mean ρ=+0.40；同族均值ρ=0.51 vs 跨族0.37；最弱对(Qwen2.5-VL-32B×Idefics3 on MMVP) ρ=+0.20，p=3.4×10⁻⁴。
- **Encoder–LLM gap**：低VSI池(n=83)上，own-tower probe acc 0.72–0.79（LLaVA-NeXT最高0.789），∆argmax仅2%–11%（Qwen3-VL最低2.4%）；Gap 0.65–0.71全模型一致；高VSI对照区probe 0.86–0.91、gap闭合，排除探针能力不足。
- **MMStar强信号**：Qwen2.5-VL-32B math AUROC=0.851[0.73,0.94], science AUROC=0.867[0.78,0.95]；底/顶五分化错误率比4.05×；LLaVA-1.5-7B同类反而反转(AUROC 0.29/0.33)。
- **18单元格选择性生成**：max-prob赢10/18单元格（校准良好模型），hybrid含VSI赢7/18（校准差或幻觉主导），VSI单独仅赢1/18(LLaVA-1.5 Hallu)；hyb(r+w)在Qwen3-VL POPE AUROC=0.636 vs max-prob 0.544 (+0.09, p<0.01)，Qwen2.5-VL-7B MMVP +0.18；hyb最差单元格AUROC最高。
- **σ鲁棒性**：AUROC across σ∈{10,20,40}差异≤0.05，per-sample rank Spearman 0.76–0.97(mean 0.89)；insensitive fraction阈值在{0.01,0.05,0.10,0.20}间方向不变。
- **VSI与verbalised confidence独立性**：Pearson |r|<0.10 across all models on POPE，联合五分化矩阵错误集中在bottom-left但off-diagonal仍有显著误差质量。

## 相关工作脉络
- **多模态幻觉评测基准**（POPE、MMVP、HallusionBench、MMStar、MMBench、MMMU）：本文不提出新基准，而是把这些基准的聚合分数在per-sample层面拆解，使重尾VSI分布可视化；与这些工作的差异在于从"构造评测"转向"诊断已构造评测中的样本异质性"。
- **选择性生成与校准**（Geifman & El-Yaniv 2017; Hendrycks & Gimpel 2017; Tian et al. 2023; Lin et al. 2022）：softmax max-prob与verbalised confidence均基于输出分布，无法捕获"忽略视觉的高置信度"；本文证明这两者与VSI几乎不相关(|r|<0.10)，并展示VSI-containing ensemble在校准差或幻觉主导单元格胜出。
- **输入级干预 vs 内部探针**（Alain & Bengio 2017; Neo et al. 2025; Ben Melech Stan et al. 2024; deletion-based attribution如Zeiler & Fergus 2014、Petsiuk et al. 2018、Ribeiro et al. 2016）：内部探针回答"模型中间表示里有什么信息"，本文的区域扰动回答"模型输出实际使用了什么信息"；两者结合才完整刻画encoder–decoder routing failure。
- **VQA语言先验文献**（Agrawal et al. 2018 "Don't Just Assume; Look and Answer"）：早期已发现模型可不用图像回答；本文将其从整体观察升级为per-sample、跨模型、输入输出因果测量。
- **VLM诊断与探针工作**（Yuksekgonul et al. 2023 bag-of-words; Tong et al. 2024b CLIP-blindness; Darcet et al. 2024 ViT registers; Laurenccon et al. 2024b cross-modal projection ablation）：这些是模型内禀性诊断；本文强调"property of samples, not models"，样本排序跨architecture transfer。
- **并行/近期相关**（Liu et al. 2025 "Seeing but Not Believing"; Raghu & Pandey 2026 "Evidence Collapse"）：分别聚焦内部状态证据与自回归轨迹，本文定位在同一encoder–decoder routing问题的另一切片。

## 局限性与未来方向
- 分析仅作用于next-token softmax分布，未测量中间层激活、注意力模式或KV-cache，故"routing failure"解释停留在输入–输出因果层面，未定位语言头内部结构。
- 跨模型一致性因基准而异（POPE ρ=0.55、MMVP 0.34、HallusionBench 0.32），后者因图表题破坏区域检测管线；VSI–错误关系在个别单元格出现反向（如Qwen3-VL MMVP perceptual subtypes p=0.043），部署阈值需逐单元格校准。
- 线性探针准确率是对编码器信息内容的下界估计；最后一层特征探针可能错过中间层更丰富的扰动敏感表征。
- verbalised confidence受prompt措辞影响（同一模型尝试两种template AUROC差异可达0.06），本文仅报告单一模板结果。
- 未来方向：(1) activation patching across cross-modal projection定位失败位置；(2) targeted retraining up-weight insensitive sub-population；(3) 扩展到intermediate-layer probing与KV-cache intervention以细化因果链。

## 研究启发与可借鉴点
- **区域扰动+KL散度的per-sample因果测量范式**可迁移到任何需要验证"模型是否真正使用某输入模态"的研究场景（音频、文本、结构化数据），比单纯accuracy或attention visualization更有因果说服力。
- **编码器探针+输出扰动双重验证**（Table 2的gap构造）是验证"模型表征了但未使用某信息"的标准套路，后续工作可直接套用：先证表征可用，再证输出不动，最后证gap集中分布于低敏感度样本。
- **跨模型一致性检验（Spearman排列测试）**可作为证明某现象"非架构特化"的通用统计工具，本文6 VLM × 15 pairs的网格设计值得在对比新信号时复用。
- **条件集成而非单一信号**的思路：VSI不是universal best abstention signal，而是"在特定failure mode（vision-ignoring confident errors）上补充softmax盲区"；团队后续研究应摒弃追求单一最强信号，转向cell-conditional signal selection。
- **四象限分解与danger quadrant提取**用于刻画"危险子类"（wrong+high conf且low VSI）具有通用性，可迁移至安全关键应用（医疗、自动驾驶）的failure mode taxonomy构建。

## 关键术语表
**Visual Insensitivity Gap**：视觉编码器所表征内容与语言头最终输出之间的不匹配差距，表现为输入扰动不被输出反映。
**Visual Sensitivity Index (VSI)**：per-sample指标，度量问题相关视觉区域被模糊前后next-token分布的KL散度，值越低表示模型越忽略该视觉证据。
**Encoder–LLM Disconnect**：感知到扰动（探针准确率高）但argmax几乎不变（2%–11%）的输入–输出断裂，量化值为0.66–0.71。
**Sample-intrinsic**：VSI排名在跨模型间保持一致（ρ=+0.40），表明视觉不敏感性是样本属性而非模型架构属性。
**Selective Generation**：允许模型在置信度不足时拒绝回答的任务设定，VSI作为该设定下的一个诊断信号。
**Danger Quadrant**：低VSI × 高softmax置信度 × 错误答案的象限，代表"自信地忽略视觉证据"的幻觉失败模式。
**hyb(r+w)**：region-VSI与whole-image-VSI的等权z-score和，在18单元格中作为默认hybrid baseline。
**PRR@80**：在保留80%预测的前提下，信号实现的风险降低与oracle风险降低之比，衡量选择性生成的有效性。

## 可复现要素
- **数据集**：POPE、MMVP、HallusionBench、MMStar（均为公开基准）；论文声明每基准原始样本池相同，answer-parseable子集因解码退化有差异；per-sample CSV、per-cell统计量、排列零分布均随代码公开。
- **代码/权重**：代码开源（arXiv:2609.00868），checkpoint标识与revision hashes已pin；HuggingFace Transformers仓库中各模型权重均可下载。
- **关键超参**：高斯模糊σ=20（主实验），核大小4σ+1 pixels；地区检测：Grounding-DINO base、confidence threshold 0.3、SAM-ViT-base、mask dilation 4 pixels；KL计算：top-50 tokens对齐；探针：L2 logistic regression、C=1.0、5-fold grouped CV；解码：greedy (do_sample=False)、max_new_tokens=8 (binary/multi-choice) / 64 (open-ended)；生成VSI时temperature不适用。
- **硬件**：NVIDIA A100-80GB GPU，BF16；单卡运行（除Qwen2.5-VL-32B双卡naive model parallelism）；总计算约32 A100-hours；Wall-clock约0.7–1.1s/样本（7B）/ 2.3s/样本（32B）。
- **统计与复现细节**：bootstrap B=1000，permutation N=10⁵（seed 42）；scikit-learn 1.8.0 pin（因GroupKFold版本差异可能使参考探针均值波动±0.01）；所有随机种子固定seed=42。
