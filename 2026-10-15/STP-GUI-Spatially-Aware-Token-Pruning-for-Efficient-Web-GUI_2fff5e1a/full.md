# STP-GUI: Spatially-Aware Token Pruning for Efficient Web-GUI Agents

Shuo Wang<sup>1†</sup>, Sifan Zhou<sup>1,2†‡</sup>, Xiaobo Wang<sup>3</sup>, Tianyu Fu<sup>1</sup>, Chenxu Zhao<sup>1</sup>, Anyang Su<sup>1</sup>, Hanning Wang<sup>1</sup>, Ping Jiang\*<sup>1</sup>, Minghui Wu\*<sup>1</sup> <sup>1</sup>Mininglamp, <sup>2</sup>Southeast University, <sup>3</sup>Shenzhen University of Advanced Technology

## Abstract

With the rapid development of vision-language models (VLMs), Web-GUI agents have achieved promising performance in interacting with real-world interfaces through screenshotbased grounding. However, high-resolution screenshots and multi-turn histories introduce excessive visual tokens, causing substantial inference overhead. Existing token pruning methods, mainly designed for natural images, often select tokens independently based on importance scores, which can fragment the local visual structures required for recognizing small GUI elements such as buttons, icons, and hyperlinks. In this work, we propose STP-GUI (Spatially-aware Token Pruning for GUI agents), a query-guided token pruning framework that preserves finegrained instruction-conditioned token scoring while performing spatially coherent windowlevel selection under aggressive token reduction. A lightweight selector is optimized through policy learning while the VLM backbone remains frozen. Experiments on Online-Mind2Web show that STP-GUI achieves a favorable efficiency-performance tradeoff under low token budgets. At 21% token retention, it preserves 77.7% of full-token performance while reducing inference latency by 31% and FLOPs by 57%. Results on Android in the Wild further suggest that the method transfers to mobile GUI environments.

## 1 Introduction

With the rapid development of vision-language models (VLMs), Web-GUI agents that navigate real websites by grounding actions in screenshots and text instructions have seen growing adoption (Fu et al., 2025; Wang et al., 2025; Bai et al., 2025a,b; Wang et al., 2024a). This setting requires high-resolution visual inputs to resolve small interactive elements such as hyperlinks, icons, and drop-down menus. The cost is a large visual token sequence: for instance, Qwen3-VL (Bai et al., 2025a) encodes a single 1920×1080 screenshot into approximately 2,040 visual tokens. Furthermore, as is a common practice in Web-GUI agent systems (Fu et al., 2025; Qin et al., 2025; Wang et al., 2025; Bai et al., 2024), agents receive multiple interaction history—prior screenshots, actions, and observations—at each step, which can easily push the total visual token count beyond 10K. These large prefill token counts severely bottleneck inference, yet users are sensitive to agent response latency. Reducing visual token load is therefore a practical challenge.

![](images/7876d6f3ad1f87ffe0d270799f160f6c09ed9de9ea4ce67958af37aa54a28618.jpg)  
Figure 1: Compared to existing VLM token pruning methods, our framework for Web-GUI agents aggregates fine-grained scores into non-overlapping spatial windows and uses a Plackett–Luce policy during training to select complete, contiguous patches. This helps suppress background noise while preserving local semantic and geometric structure, enabling the frozen VLM backbone to retain action-relevant evidence for downstream decisions.

A straightforward strategy is to prune visual tokens before input the LLM modules, as has been studied for VLM (Shen et al., 2025; Shang et al., 2025; Yang et al., 2025). Web-GUI screenshots, however, pose distinct challenges that make standard pruning strategies unreliable. First, informative regions are sparse, localized, and often tiny—a single icon or hyperlink may occupy only a handful of visual tokens—while most of the page consists of blank background, repeated layout blocks, or decorative structure. Second, the spatial continuity of these small regions is essential: losing a few adjacent patches can destroy the visual evidence needed to identify a clickable element. These characteristics make pruning strategies designed for natural images (Jizhihui et al., 2026; Xing et al., 2024), where foreground objects are large and continuous, a poor fit for web interfaces.

Despite these challenges, prior pruning paradigms in the VLM literature—attention-based selection (Jizhihui et al., 2026; Xing et al., 2024; Shang et al., 2025), similarity-based filtering (Cai et al., 2025; Yang et al., 2025; Jeddi et al., 2025), and learnable summarization (Cha et al., 2024) may remain suboptimal for GUI screenshots. Existing pruning methods (Jizhihui et al., 2026; Sun et al., 2026) involves two related decisions: estimating instruction relevance and organizing the retained evidence. Token-wise selection offers fine-grained relevance ranking, but may retain disconnected fragments of interface elements. Window-wise selection imposes local spatial structure, yet its effectiveness also depends on whether the underlying scores identify useful regions. We therefore study scoring and selection granularity jointly. Our matched-budget comparison shows that pooling Attention scores into windows yields modest improvements, while window-wise selection provides larger gains with the learned STP-GUI scorer. This motivates a lightweight instruction-conditioned selector that estimates token relevance and retains complete spatial windows before LLM processing.

As illustrated in Fig. 1, we propose STP-GUI (Spatially-aware Token Pruning for GUI agents), a query–guided token pruning method that selects spatially contiguous windows rather than isolated tokens. Concretely, STP-GUI first computes token-wise importance scores using a lightweight query–conditioned scoring module, aggregates the scores within W×W windows, and then uses a Plackett–Luce policy during training to learn which windows to retain. At inference time, windows are selected greedily by score rank under a fixed token budget, while the VLM backbone remains frozen throughout. This design preserves fine-grained token scoring while reducing spatial fragmentation of the retained set, thereby helping the model preserve local evidence around GUI elements such as buttons, icons, and label-field pairs. Experiments on Online-Mind2Web (300 tasks across 136 live websites) (Xue et al., 2025) show that STP-GUI is particularly competitive at low retention budgets. By evaluating on Android in the Wild (AitW) (Rawles et al., 2023), we further show that the same design transfers beyond web screenshots to mobile GUI settings. At 21% token retention, it achieves the strongest success rate among the compared methods reported at their evaluated operating points while reducing inference latency by 31% and FLOPs by 57% relative to the dense baseline. Our main contributions are summarized as:

• We identify a previously overlooked failure mode of GUI-agent token pruning: coupling importance estimation and selection granularity causes spatial fragmentation of actionrelevant visual evidence.

• We introduce STP-GUI, a granularitydecoupled selection paradigm that scores tokens precisely while selecting contiguous windows to preserve spatial coherence.

• We formulate GUI visual token retention as a budget-constrained discrete selection problem and design a lightweight query-conditioned selector trained with policy learning, while keeping the VLM backbone frozen.

• Experiments on Online-Mind2Web show that STP-GUI provides a favorable tradeoff between accuracy and efficiency under token pruning, reducing inference latency by 31% and FLOPs by 57%. Results on AitW also show the generalization for mobile GUI environments.

## 2 Related Work

Web-GUI Agents. Early GUI agents relied on modular pipelines combining text extraction, reasoning, and memory components (Hurst et al., 2024; Xie et al., 2024), which can be brittle under changing environments (Xia et al., 2024). Recent VLM-based agents instead directly ground actions from screenshots and achieve stronger generalization on benchmarks such as Mind2Web and AitW (Wang et al., 2024b; Hong et al., 2024; Chen et al., 2025). Meanwhile, mobile GUI agents further improve grounding through specialized training pipelines (Tang et al., 2025). However, regardless of agent architecture, high-resolution visual inputs introduce substantial token overhead, making efficient visual token selection an important problem for deployment.

Token Pruning in VLMs. Token pruning reduces the inference cost of VLMs by removing redundant visual tokens. Existing approaches can be broadly categorized into training-based and training-free methods (Bolya et al., 2022). Training-based methods, such as ATP-LLaVA (Ye et al., 2025), learn adaptive pruning policies or ratios, while STP-GUI instead freezes the VLM backbone and optimizes a lightweight queryconditioned selector for spatially coherent pruning. Training-free methods mainly rely on importancebased scoring (e.g., FastV (Chen et al., 2024), SparseVLM (Zhang et al., 2024), MustDrop (Liu et al., 2024), PDrop (Xing et al., 2024)) or redundancybased filtering (e.g., DART (Wen et al., 2025), DivPrune (Alvar et al., 2025)). However, these methods are primarily designed for natural images and select tokens independently, which may fragment the spatial evidence required for GUI grounding. STP-GUI addresses this limitation by decoupling token relevance estimation from evidence retention granularity, enabling spatially coherent token selection under constrained budgets.

Visual Grounding for GUI Agents. Recent works improve GUI grounding through richer supervision or region-aware representations. For example, Gao et al. (Gao et al., 2024) enhance visionlanguage alignment with region-level supervision, YORO (Ho et al., 2022) develops efficient visual grounding models, and R-VLM (Park et al., 2025) introduces region-aware visual representations for GUI grounding. Different from these approaches, STP-GUI addresses visual token reduction for a frozen GUI agent rather than modifying the grounding backbone. Its selector uses action-level supervision and/or full-token model feedback, without requiring token-retention masks or UI-element bounding boxes.

## 3 Methodology

## 3.1 Preliminary

A VLM processes visual tokens $\mathbf { V } \in \mathbb { R } ^ { L _ { v } \times d }$ concatenated with text tokens $\mathbf { T } \in \mathbb { R } ^ { L _ { t } \times d } .$ . Since $L _ { v } \gg L _ { t }$ , token pruning selects a subset of V via a scoring policy $\mathcal { P }$ :

$$
\mathbf { S } = \mathcal { P } ( \mathbf { V } , \mathbf { T } ) , \quad \mathbf { M } _ { i } = \mathbf { 1 } [ \mathbf { S } _ { i } \in \mathrm { T o p } { \mathbf { - } } N ( \mathbf { S } ) ]\tag{1}
$$

where $L _ { v }$ and $L _ { t }$ denote the number of visual and text tokens respectively, and d is the feature dimension. $\mathbf { M } \in \{ 0 , 1 \} ^ { L _ { v } }$ is the binary retention mask and $N = \lfloor \beta L _ { v } \rfloor$ is a fixed retention budget. We refer to $\beta$ as the keep ratio, i.e., the fraction of visual tokens retained after pruning. For multi-turn GUI agent interaction where input history concatenated with current observation yields multiple image frames, STP-GUI applies the window selection mechanism frame-wise under a uniform keep ratio $\beta .$ This ensures that each temporal visual step retains a spatially contiguous and coherent local representation before entering the LLM.

## 3.2 Method Overview

STP-GUI prunes adapter-projected visual tokens before they enter the LLM. Given visual tokens and an instruction, a lightweight scorer produces tokenlevel relevance scores. These scores are aggregated within non-overlapping spatial windows. During training, windows are sampled under a fixed budget and the selector is optimized using action-level feedback from the frozen VLM. At inference, the highest-scoring windows are retained deterministically.

## 3.3 Instruction-Guided Token Scoring

We define a lightweight instruction-conditioned scorer $f _ { \theta } ( \mathbf { V } , \mathbf { T } ) : \mathbb { R } ^ { L _ { v } \times d } \times \mathbb { R } ^ { L _ { t } \times d } \to \mathbb { R } ^ { L _ { v } }$ that maps visual tokens V and text tokens T to pertoken importance scores $s _ { i } = f _ { \theta } ( { \bf V } , { \bf T } ) _ { i }$ <sub>i</sub>. V denotes the adapter-processed visual token sequence from the vision encoder, and T denotes the embedding sequence of the current instruction. Following (Feng et al., 2025), T is restricted to the action-relevant portion of the instruction, excluding chain-of-thought reasoning steps.

The scorer performs lightweight cross-modal interaction between projected visual and textual features, conditioning each visual token on the instruction to estimate its task relevance. The resulting per-token scores are subsequently aggregated for window-level selection.

![](images/7b58bc92c0f220ef2d2fcf2733d597ca3b42bc1838d0fb2776f059f4bba3d64f.jpg)  
Figure 2: Overview of the STP-GUI pipeline. Given the task instruction and visual tokens, STP-GUI scores token relevance and performs window-level selection under a token budget before the retained tokens are fed into the LLM. The resulting compact visual representation is then used for downstream action prediction.

STP-GUI shifts the selection unit from individual tokens to spatial windows while keeping tokenlevel scoring for fine-grained importance estimation. The two operations are decoupled: token scores provide local precision, and window-level selection provides spatial coherence.

Specifically, given visual tokens on a spatial grid of size $H _ { g } \times W _ { g }$ (where $H _ { g }$ and $W _ { g }$ are the grid height and width in tokens, respectively, with $H _ { g } \cdot W _ { g } = L _ { v } )$ , we partition the grid into nonoverlapping spatial windows of size $W \times W$ , indexed by $w \in \{ 1 , . . . , \lfloor H _ { g } / W \rfloor \cdot \lfloor W _ { g } / W \rfloor \}$ . The per-token scores within each window are aggregated into a window-level score:

$$
s _ { w } = \phi ( \{ s _ { i } \} _ { i \in w } )\tag{2}
$$

where $\phi ( \cdot )$ aggregates the token importance scores within window $w .$ These aggregated window scores parameterize a Plackett–Luce distribution over candidate windows. During training, we sample window subsets from this distribution so that the expected token budget matches the target keep ratio $\beta ,$ and optimize the resulting policy with PPO. At inference time, no policy optimization is involved: windows are ranked by the aggregated scores produced by the trained scorer $f _ { \theta } ,$ and the top-ranked windows are selected under the same budget. Because every retained region is a complete $W \times W$ patch, the downstream VLM receives local evidence instead of isolated token fragments.

## 3.4 Training the Window Selector

We do not have ground-truth token-retention masks. We therefore optimize the discrete windowselection policy using action-level feedback from the frozen VLM. We therefore sample window subsets with a Plackett–Luce policy induced by the aggregated window scores, and use PPO to train the selector toward window choices that produce action responses consistent with the full-token model while respecting the token budget.

Reward design. We use two complementary reward signals: $r _ { \mathrm { g t } }$ , which encourages actions consistent with task-level supervision when available, and $r _ { \mathrm { f u l l } }$ , which encourages the pruned-input policy to remain consistent with the action preference induced by the same model under the full, unpruned input. In other words, $r _ { \mathrm { f u l l } }$ pushes the selector to retain window subsets that elicit action responses close to those of the full-token model.

$$
r _ { \mathrm { s t e p } } = \mathrm { T } _ { \mathrm { m t } } \big ( \lambda _ { \mathrm { t y p e } } + ( 1 - \lambda _ { \mathrm { t y p e } } ) \cdot \sin \big )\tag{3}
$$

where ${ \mathrm { T } } _ { \mathrm { m t } } \ \in \ \{ 0 , 1 \}$ indicates the binary type match and sim ∈ [0, 1] is the similarity score measured by edit distance for text arguments and tolerance-based matching for spatial arguments. For actions without parameters, we set $\lambda _ { \mathrm { t y p e } } = 1$

The final reward is a weighted combination of the two, allowing the selector to balance task supervision and full-context consistency:

$$
r = \lambda _ { \mathrm { g t } } \cdot r _ { \mathrm { g t } } + \lambda _ { \mathrm { f u l l } } \cdot r _ { \mathrm { f u l l } }\tag{4}
$$

where $\lambda _ { \mathrm { g t } } , \lambda _ { \mathrm { f u l l } } \in [ 0 , 1 ]$ . When task-level supervision is unavailable, training can still rely on $r _ { \mathrm { f u l l } }$ alone. As shown in the appendix, either reward alone yields competitive results, the combined reward achieves the best overall performance.

Optimization. We maintain per-action-type EMA baselines for variance reduction and train the selector using a PPO-style clipped objective with entropy regularization:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { c l i p } } = - \mathbb { E } [ \operatorname* { m i n } ( \rho A , \ \mathrm { c l i p } ( \rho , 1 { - } \epsilon , 1 { + } \epsilon ) A ) ] , } \end{array}\tag{5}
$$

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { c l i p } } - \beta _ { H } \mathcal { H } ( \pi _ { \theta } \mid \{ s _ { w _ { k } } \} ) ,\tag{6}
$$

where $\rho ~ = ~ \pi _ { \theta } / \pi _ { \mathrm { o l d } }$ is the policy ratio, $A =$ $r \mathrm { ~ - ~ } b _ { \mathrm { { e m a } } }$ is the advantage with $b _ { \mathrm { e m a } }$ being the per-action-type EMA baseline, ϵ is the clipping threshold, and $\beta _ { H }$ controls the entropy regularization strength. $\mathcal { H } ( \pi _ { \theta } \mid \{ s _ { w _ { k } } \} )$ denotes the entropy of the resulting window-selection policy. The entropy term is used to encourage exploration during training. The VLM backbone is frozen and only the lightweight selector is updated. We optimize the discrete selection policy using PPO with entropy regularization. PPO is an optimization choice rather than a separate novel algorithmic component. Nevertheless, the optimizer affects the achieved performance, as evaluated in Table 6.

## 4 Experiments

Benchmarks. Online-Mind2Web (Xue et al., 2025) is a live web agent benchmark comprising 300 tasks across 136 real websites, using task-level success rate (SR) as the primary metric. Tasks require end-to-end interaction with live pages, making them sensitive to spatial-localization errors introduced by aggressive token pruning. Android in the Wild (AitW) (Rawles et al., 2023) is a largescale mobile GUI dataset spanning diverse app categories. Following (Bai et al., 2024), we evaluate on the General and Web Shopping test splits to assess cross-domain transferability of STP-GUI.

Baseline model construction. We used Qwen3- VL as the backbone model for our experiments. We construct the Web-GUI supervised training data following prior work (Qin et al., 2025; Wang et al., 2025; Fu et al., 2025). For Online-Mind2Web and AitW, we separately obtain the corresponding backbones by supervised fine-tuning(SFT) Qwen3-VL on their training data. STP-GUI and all pruning baselines are evaluated on top of these SFT backbones. We only performed SFT as the baseline and developing advanced RL method to achieve a higher accuracy is beyond the scope of this paper.

Implementation details. We fine-tune Qwen3-VL as the base model using LLaMA-Factory (Zheng et al., 2024) on 8×A100 (80G)

GPUs for 3 epochs with an initial learning rate of 1e-5 and a cosine decay schedule. During SFT, the ViT encoder is frozen, while the adapter and LLM parameters are updated. After SFT, the VLM backbone is completely frozen for token pruning. STP-GUI only optimizes a lightweight 2M-parameter selector for window retention, without updating any VLM parameters. Unlike pruning methods that adapt VLM parameters during optimization, STP-GUI introduces only a small selection module that can be trained offline and deployed on top of the frozen backbone. The detailed tuning cost comparison with adapted pruning baselines is provided in appendix. Training the selector takes approximately 10 hours on 8 GPUs. This additional cost is incurred only once offline, while the learned selector reduces inference latency by 31% and FLOPs by 57% during deployment.

with entropy regularization. PPO is an optimization choice rather than a separate novel algorithmic component. Nevertheless, the optimizer affects the achieved performance, as evaluated in Table 6.

For comparison methods, we re-implement representative pruning approaches (Xing et al., 2024; Jizhihui et al., 2026; Sun et al., 2026; Cai et al., 2025) on Qwen3-VL using their official implementations. To ensure fair comparison, we evaluate both their original training-free settings and fine-tuned variants adapted on the same SFT data. Layer-sensitive baselines are evaluated using their best-performing configurations, with detailed sensitivity analyses provided in the appendix. Since Online-Mind2Web is evaluated on live websites, we repeat experiments 3 times under the same protocol and report the mean SR.

Comparison with Other Pruning Methods. Tab. 1 compares STP-GUI against representative visual token pruning methods on Online-Mind2Web. We evaluate 2 keep ratios (21% and 50%) on Qwen3-VL-2B, and 21% and 50% on Qwen3-VL-4B where reported. “Attention” selects tokens by prompt-to-image cross-attention scores averaged over LLM decoder layers. Following the layer sensitivity analysis in the appendix, we adopt layer 16 for Attention, which yields the strongest results within that method family. At 21% retention, STP-GUI reaches SR = 0.290, where it can be directly compared against HiPrune, HiPrune\*, FlashVLM, FlashVLM\*, and Attention at the same keep ratio. Notably, this result already exceeds PDrop and PDrop\*, which are reported at their native operating point of about 41%, as well as IVC and IVC\*, which are reported at 50% in our reproduction. At 50%, STP-GUI further reaches 0.350, remaining ahead of the other available methods at the same operating point. Overall, the table shows that STP-GUI remains competitive under substantially tighter token budgets, while FlashVLM and HiPrune further incur substantial inference-time memory overhead. STP-GUI stays much closer to the dense baseline while being more memory efficient and lower latency.

Table 1: Task success rate (SR) on Online-Mind2Web. Keep ratio (KR) denotes the fraction of visual tokens retained. The baseline means without token pruning. Best results are in bold. \* denotes our re-training implementation for training-free methods. ACC represents the ratio relative to the full-token model. Partial baselines are reported at their native setting, while direct same-ratio comparisons are available where matched keep ratios are reported.
<table><tr><td>Model</td><td>Method</td><td>KR</td><td>SR (↑)</td><td>ACC(↑)</td></tr><tr><td>OV--2B</td><td>Baseline</td><td>100%</td><td>0.373</td><td>100%</td></tr><tr><td>HiPrune</td><td>21% 50%</td><td>0.100 0.183</td><td>26.8% 49.1%</td></tr><tr><td>HiPrune*</td><td>21% 50%</td><td>0.140 0.193</td><td>37.5% 51.7%</td></tr><tr><td>PDrop</td><td>41%</td><td>0.120</td><td>32.2%</td></tr><tr><td>PDrop* IVC</td><td>41%</td><td>0.143</td><td>38.3%</td></tr><tr><td>IVC*</td><td>50% 50%</td><td>0.206 0.217</td><td>55.2% 58.2%</td></tr><tr><td rowspan="2">FlashVLM</td><td></td><td>0.153</td><td>41.0%</td></tr><tr><td>21%</td><td>0.177</td><td>47.5%</td></tr><tr><td rowspan="2">FlashVLM*</td><td>50% 21%</td><td>0.157</td><td>42.1%</td></tr><tr><td>50%</td><td>0.263</td><td>70.5%</td></tr><tr><td>Attention</td><td>21%</td><td>0.220</td><td>58.9%</td></tr><tr><td rowspan="2">STP-GUI</td><td>50%</td><td>0.280</td><td>75.1%</td></tr><tr><td>21%</td><td>0.290</td><td>77.7%</td></tr><tr><td rowspan="5">Ow---4B</td><td>Baseline</td><td>50 %</td><td>0.350</td><td>93.8%</td></tr><tr><td>IVC*</td><td>100%</td><td>0.403</td><td>100% 61.0%</td></tr><tr><td>FlashVLM</td><td>50%</td><td>0.246 0.213</td><td>52.9%</td></tr><tr><td>FlashVLM*</td><td>50% 50%</td><td>0.27</td><td>72.4%</td></tr><tr><td>Attention</td><td>21%</td><td>0.227</td><td>56.3%</td></tr><tr><td rowspan="2"></td><td>50%</td><td>0.303</td><td></td><td>75.0%</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">STP-GUI</td><td></td><td>21% 50%</td><td>0.307</td><td>76.2%</td></tr><tr><td></td><td></td><td>0.383</td><td>95.0%</td></tr></table>

Generalization to AitW Dataset. To examine whether the benefit of STP-GUI extends beyond Online-Mind2Web, we additionally evaluate it on AitW, a mobile UI grounding benchmark with different interface layouts, visual styles, and action distributions. Following prior work (Bai et al., 2024; Feng et al., 2025), we report results on the

Table 2: Results on AitW. Following (Bai et al., 2024; Feng et al., 2025), we report General and Web Shopping splits.
<table><tr><td>Method</td><td>KR</td><td>General</td><td>Web Shopping</td></tr><tr><td>Baseline</td><td>100%</td><td>0.782</td><td>0.525</td></tr><tr><td>FlashVLM*</td><td>50%</td><td>0.176</td><td>0.081</td></tr><tr><td>PDrop*</td><td>41%</td><td>0.178</td><td>0.131</td></tr><tr><td>HiPrune*</td><td>50%</td><td>0.252</td><td>0.214</td></tr><tr><td>Attention</td><td>50%</td><td>0.338</td><td>0.338</td></tr><tr><td>STP-GUI</td><td>21%</td><td>0.718</td><td>0.436</td></tr><tr><td></td><td>50%</td><td>0.741</td><td>0.492</td></tr></table>

General and Web Shopping splits. As shown in Tab. 2, all compared methods are evaluated on top of backbones separately fine-tuned on the AitW training data, and we keep the pruning configurations consistent with those used in the main experiments. For PDrop, the retained token ratio is determined by its progressive halving design across 3 layers. Under the same 50% budget, STP-GUI outperforms Attention by a large margin on both splits (0.741 vs. 0.338 on General, 0.492 vs. 0.338 on Web Shopping), showing that the advantage of STP-GUI is not limited to the web setting. At the more aggressive 21% keep ratio, STP-GUI still maintains strong transfer performance, achieving 0.718 on General and 0.436 on Web Shopping. Because not all baselines are evaluated at exactly the same keep ratio, we do not claim a strict iso-budget comparison for every entry in Tab. 2. Rather, these results show that STP-GUI remains competitive when the token budget is tightened substantially below the operating points used by several competing methods.

Note that the different keep ratios reported in Tab.1 and Tab.2 follow the original evaluation setting of the compared pruning methods, rather than being intentionally chosen to favor STP-GUI. Different pruning approaches adopt different native pruning schedules. For example, PDrop uses a progressive halving strategy, leading to a default retention ratio of approximately 41%, whereas FlashVLM, HiPrune, IVC and STP-GUI are evaluated under explicitly specified retention budgets. Following the original implementations allows us to faithfully reproduce each method.

To further check whether the observed trend is not limited to the base backbone used in our main experiments, we additionally evaluate STP-GUI on a larger backbone, Qwen3.5-9B (Qwen Team, 2026). We keep these results in the appendix, where they show a similar efficiency–performance trend on both Online-Mind2Web and AitW at a

50% token budget.

Analysis of Spatial Coherence. Beyond task success rate, we further analyze the spatial structure of the retained tokens. This analysis is motivated by the window-size ablation in 4: when $W { = } 1$ degenerates to token-wise selection, performance drops at the same keep ratio, suggesting that how retained tokens are organized in space may matter. Ideally, one would also measure whether the retained tokens continue to cover the action-critical target region at each interaction step. However, such a dataset-level evaluation is not directly available in either Online-Mind2Web or AitW: these benchmarks do not provide dense perstep UI-element annotations for complete trajectories, and successful interaction paths are not unique. We therefore focus on a trajectory-level analysis of the binary retention masks produced by different pruning methods at the same 21% budget, using 4 connected-component statistics (8-connectedcomponent statistics are shown in appendix) as a quantitative measure of spatial fragmentation, and complement this with manually curated case studies in the appendix, where we annotate steplevel targets for representative successful Online-Mind2Web episodes and compare them with the retained token regions.

$N _ { c c }$ is the number of connected components. A smaller value means less fragmentation. $R _ { \mathrm { l a r g e s t } }$ is the fraction of retained tokens that belong to the largest connected component. A larger value means that most retained evidence stays in one dominant region. $A _ { \mathrm { m e a n } }$ is the mean area of connected components (in tokens). A larger value means the retained regions are larger and more complete. $H _ { c c }$ is the entropy of component sizes. A smaller value means the retained tokens are less evenly scattered across many small regions. In short, better spatial coherence corresponds to lower $N _ { c c }$ and $H _ { c c }$ , and higher $R _ { \mathrm { l a r g e s t } }$ and $A _ { \mathrm { m e a n } } .$

Table 3: Spatial coherence of retained token on Mind2Web at 21% budget. $N _ { c c } { : }$ #connected components; $R _ { \mathrm { l a r g e s t } } { \mathrm { : } }$ ratio of tokens in the largest component; $A _ { \mathrm { m e a n } } { \mathrm { . } }$ mean component area; $H _ { c c } \mathrm { : }$ component-size entropy. Lower $N _ { c c } / H _ { c c }$ and higher $R _ { \mathrm { l a r g e s t } } / A _ { \mathrm { m e a n } }$ indicate more coherent retention.
<table><tr><td>Method</td><td> $N _ { c c \downarrow }$ </td><td> $R _ { \mathrm { l a r g e s t } } \uparrow$ </td><td> $A _ { \mathrm { m e a n } } \uparrow$ </td><td> $H _ { c c \ \downarrow }$ </td></tr><tr><td>FlashVLM</td><td> $1 2 4 . 5 6 \pm 1 6 . 5 5$ </td><td> $0 . 1 7 6 \pm 0 . 0 5 9$ </td><td> $3 . 2 4 \pm 0 . 4 9$ </td><td> $3 . 8 8 5 \pm 0 . 3 4 3$ </td></tr><tr><td>HiPrune</td><td> $1 2 8 . 9 5 \pm 1 1 . 8 0$ </td><td> $0 . 0 8 3 \pm 0 . 0 3 0$ </td><td> $3 . 1 0 \pm 0 . 2 7$ </td><td> $4 . 2 9 3 \pm 0 . 1 7 9$ </td></tr><tr><td>Attention</td><td> $9 4 . 9 5 \pm 1 7 . 6 8$ </td><td> $0 . 2 7 2 \pm 0 . 1 1 8$ </td><td> $4 . 4 7 \pm 0 . 8 9$ </td><td> $3 . 2 3 4 \pm 0 . 4 7 5$ </td></tr><tr><td>STP-GUI</td><td> ${ \bf 5 . 7 1 \pm 2 . 2 9 }$ </td><td> ${ \bf 0 . 7 8 0 \pm 0 . 1 8 0 }$ </td><td> ${ \bf 1 2 5 . 1 \pm 7 6 . 1 }$ </td><td> $\mathbf { 0 . 7 2 4 \ : \pm 0 . 4 6 1 }$ </td></tr></table>

Tab. 3 shows that STP-GUI produces only 5.71 connected components on average, compared to 94.95 for Attention and over 120 for FlashVLM and HiPrune. More than three quarters of its retained tokens (78%) fall within a single dominant region, and the mean component area is $2 8 { - } 4 0 \times$ larger. In other words, competing methods scatter retained tokens into many small islands, while STP-GUI keeps them in one or a few coherent patches. These statistics describe the spatial organization of retained masks. Lower fragmentation is partly expected from whole-window selection and does not independently establish improved action-target coverage. We interpret these results alongside the task-performance comparisons in Table 4. The qualitative examples illustrate retained local evidence rather than providing a dataset-wide measure of target coverage.

## 4.1 Ablation Studies

Table 4: Effect of selection granularity on Online-Mind2Web with Qwen3-VL-2B. KR denotes the visual token keep ratio. We report task success rate (SR) for token-wise selection $( W = 1 )$ and window-wise selection $( W = 4 )$ . $\Delta$ denotes the SR improvement in percentage points. Attention uses scores from layer 16. The best SR at each keep ratio is bolded.

<table><tr><td>KR</td><td>Method</td><td> $W = 1$ </td><td> $W = 4$ </td><td> $\Delta \left( \mathsf { p p } \right)$ </td></tr><tr><td rowspan="2">21%</td><td>Attention</td><td>0.220</td><td>0.230</td><td>+1.0</td></tr><tr><td>STP-GUI</td><td>0.247</td><td>0.290</td><td>+4.3</td></tr><tr><td>50%</td><td>Attention STP-GUI</td><td>0.280 0.313</td><td>0.297 0.350</td><td>+1.7 +3.7</td></tr></table>

Scoring and selection granularity. To examine whether simple window aggregation is sufficient to explain the performance of STP-GUI, we compare token-wise selection $( W = 1 )$ and window-wise selection $( W = 4 )$ for Attention and STP-GUI at matched keep ratios. For Attention, scores from layer 16 are aggregated within non-overlapping $4 \times 4$ windows, followed by greedy selection under the token budget. Table 4 shows that window aggregation improves Attention SR by 1.0 and 1.7 percentage points at 21% and 50% retention, respectively. STP-GUI achieves larger improvements of 4.3 and 3.7 percentage points when switching from token-wise to window-wise selection. With the same window size, STP-GUI exceeds Attention by 6.0 and 5.3 percentage points. These results suggest that window aggregation alone is insufficient to explain the performance of STP-GUI, and support the complementary roles of learned scoring and spatially structured retention.

![](images/a3d480f425274df62805fea1a54ee6fca601f1d61e4ddc18181e1869ff17a4ff.jpg)  
Figure 3: SR vs. KR on Online-Mind2Web. The dashed red line denotes the full-token baseline. STP-GUI achieves 77.7% of baseline performance at $\mathrm { K R } = 0 . 2 1$

Effect of keep ratio. Fig. 3 shows SR across KR from 0.10 to 0.56, with the full-token baseline as reference. SR rises monotonically with the budget but exhibits diminishing returns: gains are steep at low ratios and flatten as the ratio approaches the baseline. At $\mathrm { K R } = 0 . 2 1$ , STP-GUI achieves $\mathrm { S R } = 0 . 2 9 0 .$ , recovering 77.7% full-token setting.

Table 5: Effect of window overlap on task success rate.
<table><tr><td>Window strategy</td><td>Mind2Web</td><td>AitW General</td><td>AitW WebShop</td></tr><tr><td>Overlap (W/2)</td><td>0.283</td><td>0.711</td><td>0.418</td></tr><tr><td>Non-overlap</td><td>0.290</td><td>0.718</td><td>0.436</td></tr></table>

Effect of Window Overlap. We compare nonoverlapping windows with an overlapping variant using stride $W / 2$ at $W = 4$ and 21% retention. The overlapping variant does not improve the reported scores on the three evaluation splits. We therefore use non-overlapping windows as the default. The small differences do not establish that boundary coverage is unimportant.

Table 6: Comparison between PPO and naive REIN-FORCE with EMA baseline on Online-Mind2Web.
<table><tr><td>Optimizer</td><td>SR (↑)</td></tr><tr><td>Naive REINFORCE</td><td>0.247</td></tr><tr><td>PPO (clipped objective)</td><td>0.290</td></tr></table>

PPO vs. Simple REINFORCE. The method section adopts a PPO-style clipped objective for training the window selector. A natural question is whether the clipping mechanism is strictly necessary or whether a simpler policy-gradient method (naive REINFORCE) would suffice. We compare the 2 optimization strategies on Online-Mind2Web with Qwen3-VL-2B at $\mathrm { K R } = 0 . 2 1$ . As shown in Tab. 6, PPO improves SR from 0.247 to 0.290, a gain of 4.3 percentage points. This result shows that the optimization strategy affects the achieved performance. Additional training-seed statistics are needed to assess optimization stability.

Table 7: Inference efficiency on an H20 GPU (3-image)
<table><tr><td>Method</td><td>KR</td><td>Lat. (ms)↓</td><td>Mem. (MB)↓</td><td>FLOPs (G)↓</td></tr><tr><td>Baseline</td><td>100%</td><td>714.1</td><td>1914</td><td>48216.0</td></tr><tr><td>FlashVLM</td><td>21%</td><td>895.2</td><td>9174</td><td>22207.4</td></tr><tr><td>HiPrune</td><td>21%</td><td>789.2</td><td>15270</td><td>22207.4</td></tr><tr><td>PDrop</td><td>41%</td><td>557.1</td><td>956</td><td>26430.7</td></tr><tr><td>IVC</td><td>50%</td><td>723.5</td><td>984</td><td>39337.8</td></tr><tr><td>Attention</td><td>21%</td><td>707.7</td><td>6840</td><td>35328.4</td></tr><tr><td>STP-GUI</td><td>21%</td><td>494.5</td><td>982</td><td>20746.4</td></tr></table>

Efficiency Analysis. We benchmark inference on an H20 GPU with 1, 2 and 3 frames (input lengths are 2420 / 4468 / 6517 tokens). Since realworld web-agent trajectories typically involve at least 3 steps, Tab. 7 reports the 3-frame setting, while 1- and 2-frame results are provided in the Appendix. We compare STP-GUI at KR ≈ 0.21 against competing methods in terms of latency, memory and FLOPs. At 21% retention, STP-GUI reduces latency by 31% and FLOPs by 57% relative to the dense baseline. STP-GUI achieves the lowest absolute latency (494.5ms) among all evaluated methods. Competing methods like FlashVLM and HiPrune require dynamic LLM attention masks during inference, incurring 9.3× and 15.5× higher memory overhead than STP-GUI, respectively (and about 4.8× and $8 . 0 \times$ more than the dense baseline). In contrast, STP-GUI operates on ViT features before LLM prefill, maintaining a minimal memory footprint while delivering superior end-toend speedups.

## 5 Conclusion

In this paper, we present STP-GUI, a spatiallyaware visual token pruning method for Web-GUI agents. By combining query-guided scoring with window-level selection, STP-GUI preserves actionrelevant local evidence as contiguous patches under constrained token budgets. On Online-Mind2Web, it retains 93.8% of full-token performance at 50% retention and 77.7% at 21% retention, while reducing latency by 31% and FLOPs by 57%. These results show that lightweight pre-LLM pruning is a practical design for GUI agents operating under aggressive token constraints.

Limitations. Our current study focuses on Web-GUI benchmarks and model scales up to Qwen3.5- 9B. Additionally, STP-GUI uses a fixed token budget and predefined non-overlapping window granularity, which may be less suitable for interface elements that span window boundaries. Nevertheless, high-resolution encodings preserve sufficient token signals, and our overlap ablation (Tab. 5) suggests that coarse locality outweighs strict boundary alignment in practice.

## References

Saeed Ranjbar Alvar, Gursimran Singh, Mohammad Akbari, and Yong Zhang. 2025. Divprune: Diversitybased visual token pruning for large multimodal models. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pages 9392–9401.

Hao Bai, Yifei Zhou, Mert Cemri, Jiayi Pan, Alane Suhr, Sergey Levine, and Aviral Kumar. 2024. Digirl: Training in-the-wild device-control agents with autonomous reinforcement learning. Advances in Neural Information Processing Systems, 37:12461– 12495.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, and 1 others. 2025a. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, and 8 others. 2025b. Qwen2.5-vl technical report. Preprint, arXiv:2502.13923.

Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. 2022. Token merging: Your vit but faster. arXiv preprint arXiv:2210.09461.

Kaitong Cai, Jusheng Zhang, Jing Yang, Yijia Fan, Pengtao Xie, Jian Wang, and Keze Wang. 2025. Flashvlm: Text-guided visual token selection for large multimodal models. arXiv preprint arXiv:2512.20561.

Junbum Cha, Wooyoung Kang, Jonghwan Mun, and Byungseok Roh. 2024. Honeybee: Localityenhanced projector for multimodal llm. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13817–13827.

Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. 2024. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models. In European Conference on Computer Vision, pages 19–35. Springer.

Wentong Chen, Junbo Cui, Jinyi Hu, Yujia Qin, Junjie Fang, Yue Zhao, Chongyi Wang, Jun Liu, Guirong Chen, Yupeng Huo, and 1 others. 2025. Guicourse: From general vision language model to versatile gui agent. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 21936–21959.

Lang Feng, Weihao Tan, Zhiyi Lyu, Longtao Zheng, Haiyang Xu, Ming Yan, Fei Huang, and Bo An. 2025. Towards efficient online tuning of vlm agents via counterfactual soft reinforcement learning. In International Conference on Machine Learning, pages 16884–16903. PMLR.

Tianyu Fu, Anyang Su, Chenxu Zhao, Hanning Wang, Minghui Wu, Zhe Yu, Fei Hu, Mingjia Shi, Wei Dong, Jiayao Wang, and 1 others. 2025. Mano technical report. arXiv preprint arXiv:2509.17336.

Yuan Gao, Kunyu Shi, Pengkai Zhu, Edouard Belval, Oren Nuriel, Srikar Appalaraju, Shabnam Ghadar, Zhuowen Tu, Vijay Mahadevan, and Stefano Soatto. 2024. Enhancing vision-language pre-training with rich supervisions. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 13480–13491.

Chih-Hui Ho, Srikar Appalaraju, Bhavan Jasani, R. Manmatha, and Nuno Vasconcelos. 2022. YORO: Lightweight end to end visual grounding. In ECCV Workshop on International Challenge on Compositional and Multimodal Perception.

Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, and 1 others. 2024. Cogagent: A visual language model for gui agents. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14281–14290.

Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, and 1 others. 2024. Gpt-4o system card. arXiv preprint arXiv:2410.21276.

Ahmadreza Jeddi, Negin Baghbanzadeh, Elham Dolatabadi, and Babak Taati. 2025. Similarity-aware token pruning: Your vlm but faster. arXiv preprint arXiv:2503.11549.

Liu Jizhihui, Zhu Guangdao, Du Feiyi, Lian Niu, Li Jun, Chen Bin, Guan Weili, and Wang Yaowei. 2026. Hiprune: Hierarchical attention for efficient token pruning in vision-language models. In The 64th Annual Meeting ofthe Associationfor Computational Linguistics.

Ting Liu, Liangtao Shi, Richang Hong, Yue Hu, Quanjun Yin, and Linfeng Zhang. 2024. Multistage vision token dropping: Towards efficient multimodal large language model. arXiv preprint arXiv:2411.10803.

Joonhyung Park, Peng Tang, Sagnik Das, Srikar Appalaraju, Kunwar Yashraj Singh, R. Manmatha, and Shabnam Ghadar. 2025. R-VLM: Region-aware vision language model for precise GUI grounding. In Proceedings ofthe 63rd Annual Meeting ofthe Association for Computational Linguistics.

Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li,

Yunxin Li, Shijue Huang, Wanjun Zhong, Kuanye Li, Jiale Yang, Yu Miao, Woyu Lin, Longxiang Liu, Xu Jiang, Qianli Ma, Jingyu Li, and 16 others. 2025. Ui-tars: Pioneering automated gui interaction with native agents. Preprint, arXiv:2501.12326.

Qwen Team. 2026. Qwen3.5: Towards native multimodal agents.

Christopher Rawles, Alice Li, Daniel Rodriguez, Oriana Riva, and Timothy Lillicrap. 2023. Android in the wild: A large-scale dataset for android device control. Preprint, arXiv:2307.10088.

Yuzhang Shang, Mu Cai, Bingxin Xu, Yong Jae Lee, and Yan Yan. 2025. Llava-prumerge: Adaptive token reduction for efficient large multimodal models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22857–22867.

Leqi Shen, Guoqiang Gong, Tao He, Yifeng Zhang, pengzhang liu, Sicheng Zhao, and Guiguang Ding. 2025. FastVID: Dynamic density pruning for fast video large language models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Zhichao Sun, Yidong Ma, Gang Liu, Yibo Chen, Xu Tang, Yao Hu, and Yongchao Xu. 2026. Ivcprune: Revealing the implicit visual coordinates in lvlms for vision token pruning. Preprint, arXiv:2602.03060.

Liujian Tang, Shaokang Dong, Yijia Huang, Minqi Xiang, Hongtao Ruan, Bin Wang, Shuo Li, Zhiheng Xi, Zhihui Cao, Hailiang Pang, and 1 others. 2025. Magicgui: A foundational mobile gui agent with scalable data pipeline and reinforcement fine-tuning. arXiv preprint arXiv:2508.03700.

Haoming Wang, Haoyang Zou, Huatong Song, Jiazhan Feng, Junjie Fang, Junting Lu, Longxiang Liu, Qinyu Luo, Shihao Liang, Shijue Huang, and 1 others. 2025. Ui-tars-2 technical report: Advancing gui agent with multi-turn reinforcement learning. arXiv preprint arXiv:2509.02544.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, and 1 others. 2024a. Qwen2- vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191.

Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Song XiXuan, and 1 others. 2024b. Cogvlm: Visual expert for pretrained language models. Advances in Neural Information Processing Systems, 37:121475–121499.

Zichen Wen, Yifeng Gao, Shaobo Wang, Junyuan Zhang, Qintong Zhang, Weijia Li, Conghui He, and Linfeng Zhang. 2025. Stop looking for “important tokens” in multimodal language models: Duplication

matters more. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 9972–9991.

Chunqiu Steven Xia, Yinlin Deng, Soren Dunn, and Lingming Zhang. 2024. Agentless: Demystifying llm-based software engineering agents. arXiv preprint arXiv:2407.01489.

Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh J Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, and 1 others. 2024. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments. Advances in Neural Information Processing Systems, 37:52040–52094.

Long Xing, Qidong Huang, Xiaoyi Dong, Jiajie Lu, Pan Zhang, Yuhang Zang, Yuhang Cao, Conghui He, Jiaqi Wang, Feng Wu, and 1 others. 2024. Pyramiddrop: Accelerating your large vision-language models via pyramid visual redundancy reduction. arXiv preprint arXiv:2410.17247.

Tianci Xue, Weijian Qi, Tianneng Shi, Chan Hee Song, Boyu Gou, Dawn Song, Huan Sun, and Yu Su. 2025. An illusion of progress? assessing the current state of web agents. arXiv preprint arXiv:2504.01382.

Senqiao Yang, Yukang Chen, Zhuotao Tian, Chengyao Wang, Jingyao Li, Bei Yu, and Jiaya Jia. 2025. Visionzip: Longer is better but not necessary in vision language models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19792–19802.

Xubing Ye, Yukang Gan, Yixiao Ge, Xiao-Ping Zhang, and Yansong Tang. 2025. Atp-llava: Adaptive token pruning for large vision language models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24972–24982.

Yuan Zhang, Chun-Kai Fan, Junpeng Ma, Wenzhao Zheng, Tao Huang, Kuan Cheng, Denis Gudovskiy, Tomoyuki Okuno, Yohei Nakata, Kurt Keutzer, and 1 others. 2024. Sparsevlm: Visual token sparsification for efficient vision-language model inference. arXiv preprint arXiv:2410.04417.

Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, and Zheyan Luo. 2024. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd annual meeting of the association for computational linguistics (volume 3: system demonstrations), pages 400–410.

## A Appendix

## A.1 More Implementation Details

For comparison methods, we re-implement each method (Xing et al., 2024; Jizhihui et al., 2026; Sun et al., 2026; Cai et al., 2025) on top of Qwen3- VL using their official codebases. To ensure a fair comparison, we evaluate both their original training-free configurations and fine-tuned variants adapted on the same SFT data. Layer-sensitive baselines are evaluated using their best-performing configurations. Detailed sensitivity analyses are provided in appendix. Because Online-Mind2Web is conducted on live websites, we further assess evaluation stability by repeating all experiments 3 times under the same protocol and report the mean success rate in the main results. Across runs, the difference between the highest and lowest success rates never exceeds 5 successful tasks out of 300 (0.017 absolute SR). Since the performance gap between STP-GUI and the strongest competing baseline is larger than this fluctuation range, the main conclusions are unlikely to be explained by run-to-run evaluation noise alone. We use the action space defined in the agent prompt, which includes discrete action types with optional text or spatial arguments, such as clicking, hotkey invocation, scrolling, and termination.

## A.2 Layer Selection Sensitivity

For layer-dependent methods (IVC and Attention), we evaluate three layer configurations: shallow (layer 4), middle (layer 8), and deep (layer 16). Table 8 reports SR across configurations for both methods. Deep layers (layer 16) consistently yield the best results, consistent with the view that taskrelevant localization signals emerge in later transformer layers after the model has processed both visual context and instruction semantics. We use the best-performing configuration for each method in the main experiments.

## A.3 Training Cost

Table 9 reports the tuning cost of STP-GUI and the adapted training-based baselines under the same Qwen3-VL-2B setting. All adapted methods are trained with their official implementations on the same SFT data used to train STP-GUI. The adapted methods require comparable or larger optimization budgets than STP-GUI. For example, HiPrune requires 10.3–14.1 hours and Attention requires 9.7–11.0 hours, while STP-GUI takes 4.8 hours when trained with ground-truth rewards only and 10.6 hours with the combined reward. Unlike the baselines, which update a subset of VLM parameters during training, STP-GUI only optimizes a lightweight 2M-parameter selector while keeping the VLM backbone fully frozen. When groundtruth action supervision is available, STP-GUI can be trained without the additional cost of offline full-model rollouts, making the GT-only variant a practical option for budget-constrained settings.

Table 8: SR across layer configurations on Qwen3-VL-2B. Best result per group in bold.
<table><tr><td>Method</td><td>KR</td><td>Layer</td><td>SR (↑)</td></tr><tr><td rowspan="3">IVC</td><td rowspan="3">0.50</td><td>4</td><td>0.130</td></tr><tr><td>8</td><td>0.177</td></tr><tr><td>16</td><td>0.217</td></tr><tr><td rowspan="9">Attention</td><td rowspan="3">0.10</td><td>4</td><td>0.060</td></tr><tr><td>8</td><td>0.100</td></tr><tr><td>16</td><td>0.130</td></tr><tr><td rowspan="3">0.21</td><td>4</td><td>0.090</td></tr><tr><td>8</td><td>0.147</td></tr><tr><td>16</td><td>0.220</td></tr><tr><td rowspan="3">0.28</td><td>4</td><td>0.127</td></tr><tr><td>8</td><td>0.170</td></tr><tr><td>16</td><td>0.250</td></tr><tr><td rowspan="3">0.49</td><td>4</td><td>0.130</td></tr><tr><td>8</td><td>0.173</td></tr><tr><td>16</td><td>0.280</td></tr></table>

Table 9: Tuning cost on Qwen3-VL-2B (61,427 samples, 8×A100 GPUs). All times are wall-clock hours. STP-GUI optimizes only the 2M-parameter selector with the VLM backbone frozen.
<table><tr><td>Method</td><td>KR</td><td>Training Time (h)</td></tr><tr><td>FlashVLM</td><td>0.21 0.50</td><td>5.3 6.2</td></tr><tr><td>HiPrune</td><td>0.21 0.50</td><td>10.3 14.1</td></tr><tr><td>PDrop IVC</td><td>0.41</td><td>2.1</td></tr><tr><td>Attention</td><td>0.50 0.21</td><td>7.0 9.7</td></tr><tr><td>STP-GUI (GT reward only) STP-GUI (GT + full reward)</td><td>0.50 0.21 0.21</td><td>11.0 4.8</td></tr></table>

## A.4 8-Connectivity Connected-Component Analysis

The main paper reports connected-component statistics under 4-connectivity to quantify spatial fragmentation. We additionally report the same statistics under 8-connectivity. The 4 metrics— number of connected components $( N _ { \mathrm { c c } } , \downarrow )$ , largest component ratio $( R _ { \mathrm { l a r g e s t } } , \uparrow )$ , mean component area $( A _ { \mathrm { m e a n } } , \uparrow )$ , and component-size entropy $( H _ { \mathrm { c c } } ,$ ↓)—are identical to those in the main paper; the only difference is the adjacency rule.

Table 10: Connected-component statistics under 8- connectivity on Online-Mind2Web (Qwen3-VL-2B, keep ratio 0.21). Mean±std. over 300 trajectories.
<table><tr><td>Method</td><td> $N _ { \mathrm { c c \downarrow } }$ </td><td> $R _ { \mathrm { l a r g e s t } } \uparrow$ </td><td> $A _ { \mathrm { m e a n } } \uparrow$ </td><td> $H _ { \mathrm { c c } \downarrow }$ </td></tr><tr><td>STP-GUI</td><td> ${ \pm } \mathbf { \delta } . 2 9 { \pm } 2 . 1 6$ </td><td>0.794±0.180</td><td> ${ \pm } 7 5 3 . 4 \pm 7 5 . 4$ </td><td> $\mathbf { 0 . 6 6 5 \pm 0 . 4 4 8 }$ </td></tr><tr><td>FlashVLM</td><td> $8 2 . 8 2 \pm 1 2 . 4 9$ </td><td> $0 . 2 1 5 { \scriptstyle \pm 0 . 0 7 7 }$ </td><td> $4 . 9 1 \pm 0 . 8 6 $ </td><td> $3 . 3 8 3 { \pm } 0 . 3 9 9$ </td></tr><tr><td>Attention</td><td> $7 1 . 4 5 { \pm } 1 1 . 6 3$ </td><td> $0 . 3 1 1 { \scriptstyle \pm 0 . 1 2 7 }$ </td><td> $5 . 8 6 \pm 1 . 0 3$ </td><td> $2 . 8 9 4 { \scriptstyle \pm 0 . 4 3 9 }$ </td></tr><tr><td>HiPrune</td><td> $9 7 . 8 8 \pm 9 . 4 1 $ </td><td>0.104±0.038</td><td>4.08±0.39</td><td> $3 . 9 6 1 { \scriptstyle \pm 0 . 1 7 1 }$ </td></tr></table>

The trends are similar to those under 4- connectivity. Under 8-connectivity, STP-GUI produces 5.29 connected components on average, with 79.4% of retained tokens concentrated in the largest component. In contrast, FlashVLM, Attention, and HiPrune produce 71–98 components, with the largest component accounting for only 10–31% of retained tokens. The component-size entropy of STP-GUI (0.665) is also substantially lower than that of the baselines (2.89–3.96), confirming that the retained tokens are concentrated in a small number of coherent regions rather than scattered across many isolated patches. These results indicate that the spatial coherence advantage of STP-GUI is robust to the choice of connectivity rule.

## A.5 Results on Qwen3.5-9B

To examine whether STP-GUI generalizes to larger and more recent VLM backbones, we additionally evaluate it on Qwen3.5-9B. The backbone is first fine-tuned on the GUI training data following the same protocol as the Qwen3-VL models, after which the STP-GUI selector is trained on top of the frozen backbone. We evaluate on AitW-General and Online-Mind2Web.

At 50% retention, STP-GUI maintains 90.31% of the full-token performance on AitW-General (0.7546 vs. 0.8356) and 92.14% on Online-Mind2Web (0.387 vs. 0.42). The performance retention rate is comparable to that observed on Qwen3-VL-2B (93.8%) and Qwen3-VL-4B (95.0%), suggesting that the effectiveness of spatially coherent token selection is not tied to a specific model scale or architecture generation.

Table 11: Results on Qwen3.5-9B. Baseline uses all visual tokens; ACC denotes the ratio relative to the fulltoken baseline.
<table><tr><td>Dataset</td><td>KR</td><td>SR (↑)</td><td>ACC (↑)</td></tr><tr><td rowspan="3">Mind2Web</td><td>100%</td><td>0.42</td><td>100%</td></tr><tr><td>21%</td><td>0.319</td><td>75.95%</td></tr><tr><td>50%</td><td>0.387</td><td>92.14%</td></tr><tr><td rowspan="3">AitW-General</td><td>100%</td><td>0.8356</td><td>100%</td></tr><tr><td>21%</td><td>0.7083</td><td>84.76%</td></tr><tr><td>50%</td><td>0.7546</td><td>90.31%</td></tr></table>

## A.6 Efficiency Analysis: 1-Frame and 2-Frame Settings

Table 12 and Table 13 report inference efficiency for the 1-frame (2420 tokens) and 2-frame (4468 tokens) scenarios on an H20 GPU, complementing the 3-frame results in Tab. 7 of the main paper. All settings use keep ratio $\approx 0 . 2 1$ with STP-GUI at $W = 4 .$

Table 12: Inference efficiency at keep ratio ≈ 0.21 on an H20 GPU (1-image scenario, 2420 tokens). STP-GUI uses $W = 4$ . Lower is better for all metrics.
<table><tr><td>Method</td><td>KR</td><td>Lat. (ms)↓</td><td>Mem. (MB)↓</td><td>FLOPs (G)↓</td></tr><tr><td>Baseline</td><td>100%</td><td>244.6</td><td>712</td><td>15016.8</td></tr><tr><td>FlashVLM</td><td>21%</td><td>298.5</td><td>4759</td><td>8196.3</td></tr><tr><td>HiPrune</td><td>21%</td><td>286.6</td><td>6791</td><td>8196.3</td></tr><tr><td>PDrop</td><td>41%</td><td>204.1</td><td>318</td><td>9302.0</td></tr><tr><td>IVC</td><td>50%</td><td>236.4</td><td>388</td><td>12601.9</td></tr><tr><td>Attention</td><td>21%</td><td>232.2</td><td>1005</td><td>11513.3</td></tr><tr><td>STP-GUI</td><td>21%</td><td>189.3</td><td>328</td><td>8110.6</td></tr></table>

## A.7 Effect of window size.

Window size W controls the spatial granularity of token selection. We study its effect by evaluating $W \in \{ 1 , 2 , 4 , 6 , 8 \}$ across training iterations on Online-Mind2Web (Qwen3-VL-2B, KR = 0.21), with all other hyperparameters fixed. Fig. 4 reports SR at each checkpoint. When $W = 1$ , the selector degenerates to token-wise selection, exhibiting high variance throughout training and the lowest final SR. To better understand the effect of selection granularity, we further visualize the retained visual tokens of several pruning methods in appendix. Under the similar token budget, these methods tend to keep visually scattered patches, while STP-GUI preserves more spatially contiguous regions around the target element. The difference helps explain the window-size ablation results: when $W { = } 1$ , the selector degenerates to token-wise selection and is more likely to fragment the local evidence needed for grounding, whereas moderately sized windows better preserve action-relevant local structure. The training curves cross across checkpoints, while $W = 4$ achieves the highest reported final SR among the tested settings. Small windows impose weaker local grouping, whereas large windows may allocate more tokens to irrelevant regions. This tradeoff motivates the default window size, although the experiment does not separate target-boundary coverage from optimization effects.

Table 13: Inference efficiency at keep ratio ≈ 0.21 on an H20 GPU (2-image scenario, 4468 tokens). STP-GUI uses W = 4. Lower is better for all metrics.
<table><tr><td>Method</td><td>KR</td><td> $\operatorname { L a t } .$  (ms)↓</td><td> $\mathrm { { \bf { M e m } . } }$  (MB)↓</td><td>FLOPs (G)↓</td></tr><tr><td>Baseline</td><td>100%</td><td>476.8</td><td>1312</td><td>30649.2</td></tr><tr><td>FlashVLM</td><td>21%</td><td>576.0</td><td>6966</td><td>15160.0</td></tr><tr><td>HiPrune</td><td>21%</td><td>538.0</td><td>11031</td><td>15160.0</td></tr><tr><td>PDrop</td><td>41%</td><td>379.1</td><td>636</td><td>17668.6</td></tr><tr><td>IVC</td><td>50%</td><td>472.1</td><td>683</td><td>25294.4</td></tr><tr><td>Attention</td><td>21%</td><td>458.5</td><td>3275</td><td>22850.2</td></tr><tr><td>STP-GUI</td><td>21%</td><td>341.1</td><td>656</td><td>14256.5</td></tr></table>

![](images/07b36105799e69a2bed44ee90915f095466dd0d7d8fa83ad19a73bbfadb3f259.jpg)  
Figure 4: SR vs. training iterations for different window sizes on Online-Mind2Web. $W \ : = \ : 4$ achieves best-performing window granularity under our GUI settings (0.290).

## A.8 Effect of Reward Composition.

Tab. 14 ablates the weighting between the groundtruth reward $r _ { \mathrm { g t } }$ and the full-context reward $r _ { \mathrm { f u l l } }$ in Eq. 4. Using $r _ { \mathrm { g t } }$ alone (0.280) or $r _ { \mathrm { f u l l } }$ alone (0.283) already yields competitive results, confirming that the selector can be trained without ground-truth

Table 14: Ablation on reward composition $( \lambda _ { \mathrm { g t } } , \lambda _ { \mathrm { f u l l } } )$ Qwen3-VL-2B with W = 4 and ratio = 0.21.
<table><tr><td> $\lambda _ { \mathrm { g t } }$ </td><td> $\lambda _ { \mathrm { f u l l } }$ </td><td>SR(↑)</td><td>ACC(↑)</td></tr><tr><td colspan="2">Baseline</td><td>0.373</td><td>100%</td></tr><tr><td>1.0</td><td>0.0</td><td>0.280</td><td>75.1%</td></tr><tr><td>0.0</td><td>1.0</td><td>0.283</td><td>75.9%</td></tr><tr><td>0.7</td><td>0.3</td><td>0.290</td><td>77.7%</td></tr><tr><td>0.3</td><td>0.7</td><td>0.287</td><td>76.9%</td></tr><tr><td>0.5</td><td>0.5</td><td>0.256</td><td>68.6%</td></tr></table>

supervision. Combining both signals $( \lambda _ { \mathrm { g t } } = 0 . 7 ,$ $\lambda _ { \mathrm { f u l l } } = 0 . 3 )$ achieves the best result $_ { ( 0 . 2 9 0 ) }$ , indicating that the two rewards provide complementary learning signal. More detailed comparison in $\mathsf { A p - }$ pendix.

Algorithm 1: SQTS Training Procedure   
Input :Visual tokens $\overline { { \mathbf { V } } } ;$ query $\overline { { \mathbf { T } } } ;$ window   
size $W ;$ keep ratio $\beta ;$ selector   
parameters $\theta ;$ moving scale $\gamma .$   
Output :Updated selector parameters $\theta .$   
$b _ { e m a } \gets 0$   
$\{ s _ { w _ { k } } \}  \phi _ { W \times W } ( f _ { \theta } ( \mathbf { V } , \mathbf { T } ) )$   
$\mathbf { w } \sim \mathrm { P l a c k e t t L u c e } ( \{ s _ { w _ { k } } \} , \ \lfloor \beta L _ { v } / W ^ { 2 } \rfloor )$   
$/ /$ Frozen VLM rollout   
$\hat { a } \gets \mathrm { V L M } ( \mathbf { V } [ \mathbf { w } ] , \mathbf { T } )$   
a ← parse\_action(ˆa)   
$r \gets \lambda _ { \mathrm { g t } } r _ { \mathrm { g t } } ( \hat { a } ) + \lambda _ { \mathrm { f u l l } } r _ { \mathrm { f u l l } } ( \hat { a } ) ;$   
$A  r - b _ { e m a } ( a ) ;$   
for k optimization steps do   
$\left\{ s _ { w _ { k } } ^ { \prime } \right\} \gets \phi _ { W \times W } ( f _ { \theta } ( \mathbf { V } , \mathbf { T } ) ) ;$   
$\rho \gets \pi _ { \theta } ( \mathbf { w } \mid \{ s _ { w _ { k } } ^ { \prime } \} ) / \pi _ { \theta _ { \mathrm { o l d } } } ( \mathbf { w } \mid \{ s _ { w _ { k } } \} ) ;$   
$\mathcal { L } _ { \mathrm { c l i p } } $   
$\scriptstyle - \mathbb { E } [ \operatorname* { m i n } ( \rho A , \ \mathrm { c l i p } ( \rho , 1 - \epsilon , 1 + \epsilon ) A ) ] \colon$   
θ ←   
$\theta - \eta \nabla _ { \theta } \left( \mathcal { L } _ { \mathrm { c l i p } } - \beta _ { H } \mathcal { H } ( \pi _ { \theta } \mid \{ s _ { w _ { k } } ^ { \prime } \} ) \right)$   
$b _ { e m a } ( a ) \gets \gamma b _ { e m a } ( a ) + ( 1 - \gamma ) r ;$

## A.9 Step-wise Visualization of Fragmentation Patterns

Fig. 5 shows a qualitative visualization of STP-GUI’s token selection across 4 sequential steps of a representative Online-Mind2Web task. Because Online-Mind2Web and AitW do not provide dense per-step target annotations for complete trajectories, and successful interaction paths are not unique, we do not treat such examples as a dataset-level metric. Instead, we manually curate several successful Online-Mind2Web episodes as case studies, annotate the step-level ground-truth targets for those representative steps, and compare them with the retained token regions. In these cases, the selector concentrates the token budget on spatially coherent regions (navigation menus, product cards, and pricing tables), while discarding large uniform backgrounds. Token-wise pruning risks fragmenting contiguous GUI elements across the retention boundary; window-level selection instead preserves the local structure that accurate action grounding requires.

To further investigate how different pruning strategies affect the spatial structure of retained visual evidence, we provide additional visualizations for 5 representative methods: FlashVLM, PDrop, HiPrune, Attention, and IVC. Figures 6, 10, 7, 8, and 9 show the retained-token patterns of these methods across all execution steps of the same GUI interaction episode(In fact, this example task can be completed in just 4 steps.). Instead of presenting only a single-step snapshot, each large figure visualizes the full step-by-step pruning behavior of one method throughout the entire interaction process.

These full-process visualizations are useful for examining whether a pruning strategy preserves spatially coherent local evidence or gradually degenerates into fragmented token retention over time. This distinction is particularly important for GUI grounding, where accurate action prediction often depends not only on the target region itself, but also on nearby contextual cues such as surrounding text, icons, and local layout structure. As a result, methods that retain visually scattered fragments may fail to preserve the structured local evidence needed for reliable grounding, even when using a similar token budget.

Overall, these visualizations provide complementary qualitative evidence to the quantitative results in the main paper. By comparing the retainedtoken patterns across multiple execution steps, we can more clearly observe the different fragmentation behaviors induced by different pruning strategies and better understand how preserving local spatial coherence may benefit GUI agents.

![](images/4ef7d37d0e0b9dc256e948ea165cd1dc0a9cb5996fd3705e99eb487e7ee61913.jpg)  
Think: To compare different plans, you first need to go to the plan-related page. "Products" usually contains information on various products and plans; clicking this option allows you to view and compare the available AeroAPI plans on Flightaware Action: click Products | click(start\_box='(101,26),(101,26)')

![](images/43ee56252aee22aac6ad466e671fbd963769f7d78fa60076c4816ada2ea8c16b.jpg)  
Think: The current page displays several product options, including AeroAPl, FlightAware Aviation. and Premium Subscriptions. To view further details about AeroAPl or compare it with other products, click on AeroAPl to access its details page. This will provide information on the product' specific features, pricing, and available plans Action: click AeroAPI. | click(start\_ box='(219,138),(219,138)'

![](images/6cb1a2e57b8f277b76d17da110335c3cea736bd859f7ae1f11ab9aa47c7923ed.jpg)  
Think: To compare different AeroAPl plans, you first need to view their pricing structure and details There's a "View pricing tiers" button on the page; clicking it will take you to a details page for each pricing tier, providing specific information and prices for all available plans, offering direct data for subsequent comparisons Action: Click View pricing tiers. | click(start box='(71,507),(71,507)')

![](images/219debc88a3ff62daa4882d2919f068ca407413e8c2416d794bccda12dd204cc.jpg)  
Think: All available pricing tiers of AeroAPI have been reviewed, and the functionality and applicable scenarios of each tier have been understood. All key information has been displayed, and no further action is required. The task is now complete. Action: Finish. | finish()

Figure 5: Case-study visualization of STP-GUI token selection across four steps of a successful Online-Mind2Web task (“Compare AeroAPI’s available plans”). Shaded and highlighted areas indicate pruned and retained tokens, respectively, while red boxes mark manually annotated step-level targets. Although such annotations are not available benchmark-wide, this representative example shows that STP-GUI preserves spatially coherent regions around key UI elements (e.g., menus, pricing tables) despite significant token reduction.

FLASHVLM | Task: "Compare AeroAPI available plans on Flightaware"  
![](images/57d852485554a37d67ec6511cb1f42e63f0f951d3ad677417c668c4da7518c1b.jpg)  
Figure 6: Step-wise visualization of retained visual tokens for FlashVLM over a complete GUI interaction episode at 50 % keep ratio. Each panel corresponds to one execution step in the same task. The retained-token pattern appears spatially fragmented across steps, which may weaken the local structural and contextual evidence needed for GUI grounding. In this example, the method does not complete the task within 10 steps.

HIPRUNE | Task: "Compare AeroAPI available plans on Flightaware  
![](images/dbb9c497ab204e6e6f4592bb64e32cd4b028e60fd7425cbb0f889de9a6926673.jpg)

![](images/990525a22a916177f137904c485f8898da065975d1c039b311fdbfc00aec77e9.jpg)

![](images/98fe810b23e81f5c3ab7dcbc4bb244752695e3cf0c7a2df10437c90ff331f98a.jpg)

![](images/1e3a5f2e04036f216a4305febb425409f6ab683827baa7429af9a0aafec24d32.jpg)

![](images/cc566da23b9aaa50ab361fb570e978b619f116da1570b7e9d822f039a7cc6acf.jpg)

![](images/269e9b2c30fab58a0e28c00d8bb396d26b6b2ada05979818fd4a0d424ddc76d5.jpg)

![](images/2d1dc82d2bb6ca81ea05bfd4e55551036fc73a4d29ea16d75dfa57400949ebac.jpg)  
artlen dmp-SrelDwmo/artln dmspn nartien sorollemn(e

![](images/a5d7e801c14b977016a604372a164cbe26807f010f170a782fd5609d8eca926e.jpg)  
Figure 7: Step-wise visualization of retained visual tokens for HiPrune over a complete GUI interaction episode under a fixed token budget. Each panel corresponds to one execution step in the same task. In this example, the trajectory includes multiple scrolling steps and does not reach task completion.

VTATTN | Task: "Compare AeroAPI available plans on Flightaware'  
![](images/ae53c0293bc1083c741caab7aa128c0300f6c4044cfd047de136294d8a63185c.jpg)

![](images/b47bbbe01213bb9d774e31bcc2a0f301676f73f997c818cb214e65159aab8a81.jpg)

![](images/e3831164931a72a93eb4bc8c9c9305b10db61f9ce9d4f4109540d9977d1ba53b.jpg)  
<action>click(start box='(71.507).(71.507))</action>

![](images/b93a98f85bcc99ea21383f48df13e7ca48b30eb321287bfe9728e64ce1932ea1.jpg)  
Figure 8: Step-wise visualization of retained visual tokens for Attention over a complete GUI interaction episode at 50 % keep ratio. Each panel corresponds to one execution step in the same task. This method completes the task in this example, but its retained-token pattern remains more fragmented than that of STP-GUI, while appearing more continuous than some other baselines.

![](images/bfad45e117a3ebed575a06efbf18c60924d2869643c12bb6d0587c00d8f01b05.jpg)

IVCP | Task: "Compare AeroAPI available plans on Flightaware  
![](images/10d78ebeff6cf739e1abc08e5291acb18e7eb3d54a44cd0532130c594a37eb4a.jpg)  
<actice\_deg>cdick Prodects<laction espactioes <actioe\_deng>cik Pvoducte</action,

![](images/b550aa658304a4d39e79ce4f2d2f5ca66294e869a8e261d455d37b7536e36bcc.jpg)

![](images/d3bc2184f797a296d82617c314b9a43274a2fa4daa47895c13f405e9090d6466.jpg)

![](images/50c462d57defa0f97f073af69057c6e11f5a30f6d2f4287e96dc5ef92031af46.jpg)

![](images/4cd09eb6380c1ace456b8ecbcb794be0ebb37cef39bc57ce372bfc5679af5755.jpg)

![](images/70fdbc007529333d658f7b5c5bc459477fdd7b9f68353500ae13602e4d29ade7.jpg)  
Figure 9: Step-wise visualization of retained visual tokens for IVC over a complete GUI interaction episode at 50 % keep ratio. Each panel corresponds to one execution step in the same task. In this example, the method executes the first two steps correctly, but later steps exhibit less continuous retained evidence over the webpage, and the task is not completed.

PDROP | Task: "Compare AeroAPI available plans on Flightaware'  
![](images/978ce59af706e5b004da4e6e511d46700addfe0782b8b51d5ca9d217db2cc0f2.jpg)

![](images/be0d4554b3ae4ee918b691b2c77da2887008061a518774f386306803ab220951.jpg)

![](images/76f6bb729b093a904d7cddab94ef284fd5c5f2dc7034d06be885deb593b68783.jpg)

![](images/eac87a2c14947dd3bbc0d591958821e2f14e4f2ff28ca1a6776f676e85b3bf09.jpg)

![](images/5cc4ec5a8526510d566f1c8c7eed02ae3809aa3ca5ea26e8a83d70d7648e4d2f.jpg)  
<action\_esp>-icraitowmn-</action\_ep> <actian>scrttew(</actiono

![](images/0b3fca3a773ae276e023bb43f0d753e13bb30088a9c940c58be28fbccd6a0b95.jpg)  
<actice deseiScralportrciection dete> cacthano-scrolitemn-c/action

![](images/dd08f052d7908732da9e299e0e02e614a85f57329ddfe54006c7654c68ae6b11.jpg)

![](images/35c70d54b63a2398cd7499a4f68316e31e6ffaaf061d407945fa4466312c5121.jpg)  
Figure 10: Step-wise visualization of retained visual tokens for PDrop over a complete GUI interaction episode at 41 % keep ratio. Each panel corresponds to one execution step in the same task. The retained-token pattern appears spatially fragmented across steps, which may weaken the local structural and contextual evidence needed for GUI grounding. In this example, the trajectory returns to the initial page and does not complete the task.