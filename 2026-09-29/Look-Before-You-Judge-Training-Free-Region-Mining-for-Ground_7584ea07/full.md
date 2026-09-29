# Look Before You Judge: Training-Free Region Mining for Grounded and Explainable Deepfake Detection

Chia-Ling Chen<sup>1∗</sup>, Yu-Ting Ta<sup>1∗</sup>, Jian-Yu Jiang-Lin<sup>1</sup>, Tai-Ming Huang<sup>1</sup>, Ling Lo<sup>2</sup>, Po-Ching Chen<sup>1</sup>, Yan-Tsung Wang<sup>1</sup>, Pei-Heng Li<sup>1</sup>, Ling Zou<sup>1</sup>, Hong-Han Shuai<sup>3</sup>, Wen-Huang Cheng<sup>1†</sup>

<sup>1</sup>National Taiwan University <sup>2</sup>National Tsing Hua University <sup>3</sup>National Yang Ming Chiao Tung University wenhuang@csie.ntu.edu.tw

## Abstract

Multimodal large language models (MLLMs) can explain deepfake verdicts in natural language, but such explanations are not necessarily visually grounded in the visual evidence underlying the prediction. A model may describe plausible artifacts inferred from language priors rather than from image evidence. Existing grounding methods improve visual reliance through decoding or attention interventions, but they generally strengthen grounding over the entire image, making them illsuited for forensic artifacts that are subtle, spatially localized, and image-dependent. We propose Look Before You Judge, a training-free framework that formulates explainable deepfake detection as a sequential evidence acquisition process. Instead of directly predicting image authenticity from holistic visual reasoning, our framework first identifies image-specific candidate evidence regions by contrasting the MLLM’s decoder-tovisual attention between an original image and its Gaussianblurred counterpart. The identified regions are then inspected individually, and the resulting local evidence is integrated with the global image context before reaching a final verdict. The framework operates without manipulation masks, external forensic models, or parameter updates, making it directly applicable to of-the-shelf MLLMs. Across five open-source MLLMs on TriDF and MMTD-Set, our framework improves detection accuracy by up to 12.8%, reduces CHAIR by up to 33.4% and hallucination rate by up to 21.3%, and outperforms representative training-free decoding and attention methods.

## Introduction

Deep generative models have rapidly improved the realism of synthetic imagery, raising growing concerns about media integrity, identity verification, and digital forensics. Conventional deepfake detectors formulate deepfake detection as an image-level binary classification task, predicting whether an input is real or manipulated (Chen et al. 2022; Dong et al. 2023; Haliassos et al. 2022; Han et al. 2025; Sun et al. 2026). However, a reliable forensic analysis requires more than a binary verdict. To support trustworthy decision making, a detector should provide concrete visual evidence that justifies its prediction. Motivated by this need, recent multimodal large language model (MLLM)-based forensic methods augment image-level prediction with natural-language explanations (Jia et al. 2024; Zou et al. 2025; Shi et al. 2025). Despite their impressive fluency, the explanations often fail to faithfully reflect the visual evidence underlying the prediction. As illustrated in Figure 1, a state-of-the-art MLLM may generate a plausible forensic explanation while overlooking annotated manipulation artifacts, ultimately producing an incorrect verdict. The key challenge of trustworthy deepfake detection is not generating explanations after prediction, but explicitly acquiring image-specific forensic evidence before making a forensic judgment.

![](images/651888807dbb27f9df02e33d817a994b9b4390d51934c56d246524789f7fcbaa.jpg)  
Figure 1: Look Before You Judge. A general MLLM produces an ungrounded explanation and an incorrect authentic verdict. Our framework instead inspects localized candidate regions and aggregates the resulting evidence to reach the correct manipulated verdict.

Acquiring forensic evidence, however, is fundamentally more challenging than reasoning over semantic content. Unlike high-level semantic concepts, forensic evidence is typically weak, spatially localized, and expressed through subtle visual inconsistencies. Depending on the manipulation, the diagnostically useful cues may occur around facial structures in DeepFakes or around edited objects and regenerated regions in broader AIGC images. Under holistic image reasoning, these subtle forensic signals are easily overwhelmed by semantically dominant image content. Current MLLMbased forensic systems nevertheless attempt to localize evidence, perform reasoning, generate explanations, and predict authenticity within a unified inference process (Xu et al. 2025; Huang et al. 2025; Kang et al. 2025), making evidence acquisition an implicit of reasoning rather than an explicit objective. Although recent grounding methods improve visual reliance through decoding or attention interventions (An et al. 2025; Yin, Si, and Wang 2025; Tang et al. 2025; Zhuang et al. 2025; Kim et al. 2025), they generally strengthen grounding over the entire image or assume informative regions are already available (Wu et al. 2024). Consequently, existing approaches do not enable an of-the-shelf MLLM to autonomously identify image-specific forensic evidence before making a forensic judgment.

In this work, we formulate explainable deepfake detection as a sequential, spatially grounded evidence acquisition problem rather than a single-pass explanation task. Instead of directly predicting image authenticity from holistic visual reasoning, we advocate a look-before-you-judge paradigm in which the model first identifies image-specific regions that are likely to contain diagnostically useful evidence, inspects these regions to collect manipulation-related observations, and finally integrates the resulting local evidence with the global image context before reaching a verdict. This formulation explicitly separates evidence acquisition from decision making, making local evidence inspection a prerequisite for reliable forensic reasoning rather than a by-product of explanation generation.

Following the formulation, we propose a training-free evidence acquisition framework for of-the-shelf MLLMs. Our key insight is that image regions containing forensic evidence exhibit distinctive attention responses to fine-grained visual perturbations. We exploit this property by contrasting the MLLM’s decoder-to-visual attention between the origina and blurred images, enabling the model to identify candidate evidence regions without additional supervision. The identified regions are subsequently inspected individually, and the resulting local observations are integrated with the global image context for final prediction. Since the entire pipeline relies solely on the deployed MLLM’s test-time attention responses, our framework is self-contained and requires neither manipulation masks, predefined facial priors, external forensic encoders, learned localization modules, nor task-specific training.

Our contributions are summarized as follows:

• We introduce a look-before-you-judge formulation for explainable deepfake detection, which explicitly separates region discovery, local evidence inspection, and final image-level judgment. This formulation addresses the missing pre-decision inspection stage in existing MLLMbased forensic reasoning.

• Based on this formulation, we develop a fully training-free test-time evidence acquisition framework that identifies detail-sensitive forensic regions through blur-contrastive attention analysis and guides the MLLM to inspect them without manipulation masks, task-specific training, or learned localization modules.

• Extensive experiments across five open-source MLLM backbones and two benchmarks covering diverse deepfake manipulations and AIGC editing demonstrate consistent improvements in both detection accuracy and explanation grounding, validating the efectiveness and generalizability of the proposed framework.

## Related Work

## From Detection to Local Evidence Inspection

Early deepfake detectors formulate forgery detection as image-level binary classification, typically trained on largescale datasets such as FaceForensics++ (Rossler et al. 2019) and DFDC (Dolhansky et al. 2020). Although efective in controlled settings, these models often rely on datasetspecific artifacts and may generalize poorly to unseen generators (Rossler et al. 2019; Dolhansky et al. 2020). Later methods incorporate forgery-aware priors to improve robustness (Liu et al. 2024; Yang et al. 2025), but still mainly provide image-level predictions without exposing local evidence. Recent MLLM-based image forensic systems, such as FakeShield (Xu et al. 2025), SIDA (Huang et al. 2025), and LEGION (Kang et al. 2025), extend detection toward interpretable forgery analysis by producing textual explanations and, in some cases, localization outputs. However, their localization and explanation are typically obtained through supervised training on annotated masks, artifact labels, or curated explanation data, and are generated jointly with the final decision. Our work neither learns localization from annotated data nor directly asks the MLLM for a global explanation. Instead, it first mines candidate forensic regions from the model’s own test-time attention responses, enabling a look-before-you-judge inspection of local evidence prior to the final judgment.

## Visual Grounding and Forensic Localization

Recent studies have linked MLLM hallucination to insuficient visual grounding and an excessive reliance on language priors, motivating a growing body of training-free methods that intervene in attention or decoding at inference time (Lu et al. 2026; Basile et al. 2026; Tu et al. 2026; Huang et al. 2024; Leng et al. 2024). While these approaches generally improve the model’s reliance on visual information, they predominantly operate at the whole-image level. Such global interventions may be suboptimal for forensic analysis, where discriminative artifacts are often sparse and confined to small regions of the image. ControlMLLM (Wu et al. 2024) enables attention steering toward designated image regions, but assumes that the regions of interest are specified in advance.

In parallel, existing forensic approaches extract localized evidence using frequency-domain priors (Qian et al. 2020), gradient-based saliency (Selvaraju et al. 2017), or dedicated localization networks (Li et al. 2020; Xu et al. 2025; Kang et al. 2025). LAA-Net (Nguyen et al. 2024) further reduces the dependence on explicit manipulation masks by learning attention over artifact-prone regions. Nevertheless, its localization behavior remains governed by task-specific training objectives and the artifact distributions represented in the training data.

![](images/ac2566abf00811a04a392cc8a7c888771d0a829ccd25bece12de9bbbd22c4d00.jpg)  
Figure 2: Overview of Look Before You Judge. Stage 1 derives a blur-sensitive visual prior by contrasting output-to-visual attention between the input image and its blurred counterpart. Stage 2 constructs candidate forensic regions from high-response tokens. Stage 3 performs region-wise inspection with attention steering and aggregates the filtered evidence to produce the final prediction and explanation. The entire framework is training-free and requires only frozen MLLMs.

These two lines of research therefore address complementary, yet incomplete, aspects of explainable forensic reasoning. Training-free grounding methods strengthen visual reliance without explicitly identifying where forensic evidence is located, whereas forensic localization methods typically depend on task-specific supervision or dedicated localization components. Our method bridges this gap by automatically deriving image-specific forensic regions from an MLLM’s inference-time attention responses. This enables region-wise evidence inspection without localization supervision, auxiliary localization modules, or parameter updates.

## Method

We introduce a training-free framework that uses an MLLM’s output-to-visual attention to determine where to inspect before making an image-level judgment. As shown in Figure $^ { 2 , }$ it discovers and inspects detail-sensitive regions and verifies the evidence retained under an undirected global view.

The mined visual prior is authenticity-agnostic: it serves only as a proposal signal for subsequent forensic inspection. Because all candidate regions and steering distributions are derived from test-time attention, the framework requires no manipulation masks, predefined facial regions, auxiliary modules, or parameter updates.

## Stage 1: Blur-Sensitive Visual Prior Mining

Stage 1 mines a token-level blur-sensitive prior that highlights visual tokens whose attention responses depend on fine-grained image detail. Because Gaussian blur suppresses such detail while largely preserving coarse image structure, reduced output-to-visual attention under blur provides a useful signal for forensic region mining.

We analyze decoder self-attention from generated response tokens to visual tokens, rather than the vision encoder’s internal patch attention, as it reflects the visual evidence consulted while producing the forensic response.

Given a raw image I, we construct a Gaussian-blurred counterpart I<sup>blur</sup>. To avoid confounding visual changes with diferent textual continuations, we generate a response from I and reuse its token sequence as a fixed, teacher-forced continuation for both inputs.

Let $l = 1 , \ldots , L$ and $h = 1 , \ldots , H$ index decoder layers and attention heads. For condition $c \in \{ \mathrm { r a w } , \mathrm { b l u r } \}$ , let $\alpha _ { t , v } ^ { c , ( l , h ) }$ denote the post-softmax attention from responsetoken position t to visual token v.

We average the output-to-visual attention over generated response positions:

$$
\bar { A } _ { l , h } ^ { c , \mathrm { a b s } } ( v ) = \frac { 1 } { | \mathcal { T } _ { \mathrm { o u t } } | } \sum _ { t \in \mathcal { T } _ { \mathrm { o u t } } } \alpha _ { t , v } ^ { c , ( l , h ) } , \quad v = 1 , \ldots , N _ { v } ,\tag{1}
$$

where $N _ { v }$ denotes the number of visual tokens and $\tau _ { \mathrm { { o u t } } }$ the generated response-token positions. The superscript abs de-

notes the original post-softmax attention mass before renormalization over visual tokens.

From the aligned raw and blurred attention maps, we compute the positive raw–blur delta

$$
\Delta _ { l , h } ^ { + } ( v ) = \mathrm { R e L U } \left( \bar { A } _ { l , h } ^ { \mathrm { r a w , a b s } } ( v ) - \bar { A } _ { l , h } ^ { \mathrm { b l u r , a b s } } ( v ) \right) ,\tag{2}
$$

The reverse delta $\Delta _ { l , h } ^ { - }$ is defined analogously by exchanging the raw and blurred terms. $\Delta _ { l , h } ^ { + }$ provides the token-level signal used to construct the visual prior, while $\Delta _ { l , h } ^ { - }$ is used only to assess the directionality of each head’s attention shift. Head scoring. Not every head provides a reliable blursensitive response. We therefore score heads by directiona change, spatial redistribution, and spatial reliability.

We first measure the fraction of directional change occurring in the desired raw > blur direction:

$$
G _ { l , h } = \frac { \sum _ { v = 1 } ^ { N _ { v } } \Delta _ { l , h } ^ { + } ( v ) } { \sum _ { v = 1 } ^ { N _ { v } } \Delta _ { l , h } ^ { + } ( v ) + \sum _ { v = 1 } ^ { N _ { v } } \Delta _ { l , h } ^ { - } ( v ) + \epsilon } ,\tag{3}
$$

where $\epsilon > 0$ ensures numerical stability.

Let $\mathcal { M } _ { \mathrm { i n } }$ denote a valid-region mask excluding border tokens. For each head, $R _ { l , h } ^ { \mathrm { i n } }$ denotes the fraction of positivedelta mass within the valid region, while $R _ { l , h } ^ { \mathrm { p k } }$ denotes the fraction concentrated on its highest-response token.

To measure spatial redistribution, we normalize each attention map over visual tokens:

$$
A _ { l , h } ^ { c } ( v ) = \frac { \bar { A } _ { l , h } ^ { c , \mathrm { a b s } } ( v ) } { \sum _ { u = 1 } ^ { N _ { v } } \bar { A } _ { l , h } ^ { c , \mathrm { a b s } } ( u ) + \epsilon } .\tag{4}
$$

We combine these factors into the head score

$$
S _ { l , h } = D _ { \mathrm { J S } } \left( A _ { l , h } ^ { \mathrm { r a w } } , A _ { l , h } ^ { \mathrm { b l u r } } \right) \cdot G _ { l , h } \cdot R _ { l , h } ^ { \mathrm { i n } } \cdot \left( 1 - R _ { l , h } ^ { \mathrm { p k } } \right)\tag{5}
$$

where $D _ { \mathrm { J S } }$ denotes Jensen–Shannon divergence. This score favors heads with predominantly raw-over-blur change, meaningful spatial redistribution, and responses that are neither border-dominated nor concentrated on a single token.

We discard heads that fail fixed nondegeneracy checks and retain up to $K _ { H }$ high-scoring heads under fixed diversity constraints, yielding the image-specific set H.

We normalize each selected positive-delta map to unit mass and average the resulting maps:

$$
\bar { P } ( v ) = \frac { 1 } { | \mathcal { H } | } \sum _ { ( l , h ) \in \mathcal { H } } \tilde { \Delta } _ { l , h } ^ { + } , ~ \tilde { \Delta } _ { l , h } ^ { + } ( v ) = \frac { \Delta _ { l , h } ^ { + } ( v ) } { \sum _ { u = 1 } ^ { N _ { v } } \Delta _ { l , h } ^ { + } ( u ) + \epsilon } .\tag{6}
$$

Per-head normalization prevents attention-scale diferences from dominating the pooled prior. After applying the validregion mask, we renormalize $\bar { P }$ to obtain the Stage 1 prior $\boldsymbol { P } ^ { \check { \mathbf { \Phi } } } \in \mathbb { R } ^ { N _ { v } }$

## Stage 2: Local Forensic Region Construction

Stage 1 produces a token-level blur-sensitive prior P, whose high-response tokens may occupy multiple spatially separated locations. Stage 2 groups these responses into a small set of spatially coherent regions for independent inspection.

For each input, we map the one-dimensional prior onto a unified two-dimensional visual-token grid using the spatial layout provided by the corresponding backbone:

$$
P  P _ { \mathrm { g r i d } } \in \mathbb R ^ { H _ { v } \times W _ { v } } , \quad N _ { v } = H _ { v } W _ { v } ,\tag{7}
$$

where $H _ { v }$ and $W _ { v }$ are the input-specific grid dimensions. This backbone-aware mapping preserves the spatial ordering of the visual tokens for subsequent region construction.

We retain the top-p fraction of tokens, group spatially adjacent responses using connected-component analysis, discard degenerate components, and slightly expand the survivors for local context, yielding the candidate components $\{ \tilde { C } _ { j } \}$

For each candidate component, we compute its total prior mass and the corresponding steering distribution:

$$
\begin{array} { r l r l } { M _ { j } = \displaystyle \sum _ { v \in \tilde { C } _ { j } } P ( v ) , } & { \quad P _ { j } ( v ) } & { = \left\{ \frac { P ( v ) } { M _ { j } } , \begin{array} { l l } { v \in \tilde { C } _ { j } , } \\ { 0 , } \end{array} \right. } & { } & { v \in \tilde { C } _ { j } . } \end{array}\tag{8}
$$

Here $M _ { j }$ measures the component’s contribution to the image-level blur-sensitive prior and is used for region rank ing, while $P _ { j }$ preserves its relative prior weights and satisfies $\begin{array} { r } { \sum _ { v } P _ { j } ( v ) = \mathbf { \dot { 1 } } } \end{array}$ for Stage 3 steering.

We rank the candidate components by $M _ { j }$ and retain up to K regions:

$$
\mathcal { C } = \mathrm { T o p } { \cdot } K _ { j } ( M _ { j } ) = \{ C _ { 1 } , C _ { 2 } , \dots , C _ { K ^ { \prime } } \} , \quad K ^ { \prime } \le K ,\tag{9}
$$

where $K ^ { \prime } < K$ when fewer than K valid components remain. After ranking, the retained components and their steering distributions are reindexed as $\{ ( C _ { i } , P _ { i } ) \} _ { i = 1 } ^ { K ^ { \prime } }$ for regionwise inspection in Stage 3.

## Stage 3: Region-Wise Evidence Inspection and Global Aggregation

Stage 3 converts the retained regions into structured forensic evidence and aggregates the filtered evidence into an imagelevel verdict. It comprises region-wise inspection (Stage 3.a), then evidence filtering and global aggregation (Stage 3.b).

Stage 3.a: Region-Wise Evidence Inspection For each candidate region $C _ { i } \in \mathcal { C }$ , we steer generation using its distribution $P _ { i }$ from Stage 2. Following AttnReal (Tu et al. 2026), we recycle attention from generated-response positions, but redistribute it according to $P _ { i }$ rather than uniformly over all visual tokens.

Let $\mathcal { H } ^ { \star }$ denote all attention heads within a model-specific decoder-layer range $\mathcal { L } ^ { \star }$ . This intervention set is distinct from the mined set ${ \mathcal { H } } ,$ used only to construct the Stage 1 prior.

Following the sink-selection rule of AttnReal, let $S _ { t } ^ { ( l , h ) }$ denote positions selected from the generated-response history. With retention coeficient $\rho \in [ 0 , 1 )$ , the recycled mass is

$$
m _ { t } ^ { ( l , h ) } = ( 1 - \rho ) \sum _ { k \in S _ { t } ^ { ( l , h ) } } \alpha _ { t , k } ^ { ( l , h ) } .\tag{10}
$$

Let $\nu$ denote the visual-token key positions, with $P _ { i } ( k )$ denoting the weight of the visual token at position $k \in { \mathcal { V } } .$

<table><tr><td rowspan="2">Method</td><td colspan="5">TriDF</td><td rowspan="2"></td><td colspan="3">MMTD-Set</td></tr><tr><td>ACC↑</td><td>Cover ↑</td><td>CHAIR↓</td><td>Hal↓</td><td> $F ^ { 0 . 5 } \uparrow$ </td><td>DeepFake</td><td>F1↑</td><td>AIGC-Editing ACC↑ F1↑</td></tr><tr><td colspan="9">ACC ↑</td></tr><tr><td></td><td></td><td>0.0270</td><td>Open-source models</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>InternVL-3.5-8B InternVL-3.5-8B + Ours</td><td>0.4176 0.5458</td><td>0.2239</td><td>0.9745 0.6407</td><td>1.0000 0.7875</td><td>0.0296 0.2564</td><td>0.5235 0.5591</td><td>0.4936 0.6278</td><td>0.5200 0.5260</td><td>0.4419 0.5917</td></tr><tr><td></td><td></td><td>0.0395</td><td>0.9763</td><td>0.9980</td><td>0.0217</td><td>0.5300</td><td>0.5488</td><td>0.5105</td><td>0.4872</td></tr><tr><td>InternVL-3.5-14B InternVL-3.5-14B + Ours</td><td>0.4213 0.5324</td><td>0.2045</td><td>0.7963</td><td>0.9449</td><td>0.1587</td><td>0.5776</td><td>0.5496</td><td>0.5525</td><td>0.6193</td></tr><tr><td>MiMo-VL-7B</td><td>0.5650</td><td>0.2280</td><td>0.6539</td><td>0.8739</td><td>0.2914</td><td>0.6046</td><td>0.5423</td><td></td><td></td></tr><tr><td>MiMo-VL-7B + Ours</td><td>0.6086</td><td>0.1110</td><td>0.6536</td><td>0.7271</td><td>0.2144</td><td>0.6326</td><td>0.6011</td><td>0.5935 0.5655</td><td>0.4389 0.4749</td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td>0.6207</td><td>0.2557</td><td>0.8073</td><td>0.9993</td><td>0.2022</td><td>0.6141</td><td>0.5952</td><td>0.4940</td><td></td></tr><tr><td>Qwen3-VL-8B-Instruct + Ours</td><td>0.6408</td><td>0.1875</td><td>0.7566</td><td>0.9349</td><td>0.2020</td><td>0.6181</td><td>0.6294</td><td>0.5725</td><td>0.4960 0.5479</td></tr><tr><td></td><td>0.6969</td><td>0.0700</td><td>0.9381</td><td>0.9889</td><td>0.0552</td><td>0.5195</td><td>0.6524</td><td></td><td></td></tr><tr><td>Qwen3.5-9B Qwen3.5-9B + Ours</td><td>0.6972</td><td>0.1088</td><td>0.8722</td><td>0.9564</td><td>0.1108</td><td>0.6066</td><td>0.6851</td><td>0.4885 0.6130</td><td>0.6207 0.6387</td></tr><tr><td colspan="10">Commercial models</td></tr><tr><td></td><td>0.6573</td><td>0.2714</td><td>0.6982</td><td>0.9651</td><td>0.2919</td><td>0.7727</td><td>0.7471</td><td>0.5786</td><td>0.2936</td></tr><tr><td>GPT-5 Gemini 2.5-Pro</td><td>0.7311</td><td>0.4208</td><td>0.5571</td><td>0.9332</td><td>0.4258</td><td>0.5998</td><td>0.6664</td><td>0.7063</td><td>0.7302</td></tr><tr><td>Claude Sonnet 4.5</td><td>0.6240</td><td>0.3988</td><td>0.7235</td><td>0.9980</td><td>0.2908</td><td>0.5509</td><td>0.6513</td><td>0.5804</td><td>0.5508</td></tr></table>

Table 1: Overall Quantitative Comparison. Results across five open-source MLLMs on TriDF and MMTD-Set, with commercial MLLMs reported as performance references.

For $( l , h ) \in { \mathcal { H } } ^ { \star }$ , we update

$$
\begin{array} { r } { \tilde { \alpha } _ { t , k } ^ { ( l , h ) } = \left\{ \begin{array} { l l } { \rho \alpha _ { t , k } ^ { ( l , h ) } , } & { k \in S _ { t } ^ { ( l , h ) } , } \\ { \alpha _ { t , k } ^ { ( l , h ) } + m _ { t } ^ { ( l , h ) } P _ { i } ( k ) , } & { k \in \mathcal { V } , } \\ { \alpha _ { t , k } ^ { ( l , h ) } , } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}\tag{11}
$$

Because the sink and visual positions are disjoint and $\begin{array} { r } { \sum _ { v } P _ { i } ( v ) = 1 } \end{array}$ , the removed and added masses are equal, preserving the attention-row sum while redistributing attention toward $C _ { i }$

The same region-agnostic prompt $q _ { \mathrm { l o c } }$ is used for every region. It asks the model to report a local irregularity and assess its evidential strength without issuing an image-level verdict. The structured response is

$$
E _ { i } = ( o _ { i } , e _ { i } , \gamma _ { i } , a _ { i } , n _ { i } ) ,\tag{12}
$$

consisting of a localized observation, manipulation-evidence label, confidence, artifact category, and optional natural explanation. The responses form $\mathcal { E } _ { \mathrm { r a w } } = \{ E _ { i } \} _ { i = 1 } ^ { K ^ { \prime } }$

Stage 3.b: Evidence Filtering and Global Aggregation Evidence Filtering. Region-wise inspection may produce unreliable or redundant observations. We process ${ \mathcal { E } } _ { \mathrm { r a w } }$ in descending region-prior order, filtering such responses and retaining the higher-priority observation when duplicates occur. The remaining observations form $\mathcal { E } _ { \mathrm { c a n d } }$ , with their contribution calibrated according to evidential strength before global aggregation.

Final Aggregation. We construct an aggregation prompt $q _ { \mathrm { a g g } } ( \mathcal { E } _ { \mathrm { c a n d } } )$ from the authenticity instruction and filtered observations, presenting them as tentative cues to be verified against the full image. The final prediction is

$$
\begin{array} { r } { \hat { y } = g \big ( I , q _ { \mathrm { a g g } } ( \mathcal { E } _ { \mathrm { c a n d } } ) \big ) , } \end{array}\tag{13}
$$

where $g$ denotes standard inference with the same model but without attention steering, thereby restoring the global context needed to verify and integrate the local observations.

## Experiments

## Experimental Setup

Datasets and Evaluation Metrics. We evaluate our framework on TriDF Type-B <OEQ> (Jiang-Lin et al. 2026) and MMTD-Set (Xu et al. 2025). TriDF provides artifact-level annotations, enabling joint evaluation of authenticity prediction and explanation grounding. We report accuracy (ACC), artifact coverage (Cover), CHAIR, hallucination rate (Hal), and $F ^ { 0 . 5 }$ on TriDF. Cover measures the recall of annotated artifacts, whereas CHAIR and Hal quantify unsupported artifact claims. The $F ^ { 0 . 5 }$ score summarizes the trade-of between artifact precision and coverage, with greater emphasis on precision. On MMTD-Set, we report ACC and F1 on the DeepFake and AIGC-Editing subsets.

Backbone Models. We evaluate five of-the-shelf MLLMs: InternVL-3.5-8B, InternVL-3.5-14B (Wang et al. 2025), Qwen3-VL-8B-Instruct (Bai et al. 2025), Qwen3.5- 9B (Qwen Team 2026), and MiMo-VL-7B (Xiaomi 2025). For each backbone, we compare vanilla inference with the proposed test-time evidence acquisition framework.

Implementation Details. All experiments are conducted in a fully training-free setting. The same three-stage inference procedure is applied across all MLLM backbones. Additional implementation details and hyperparameter settings are provided in the supplementary material.

<table><tr><td rowspan="2">Method</td><td colspan="2">TriDF</td></tr><tr><td>ACC ↑ Cover ↑</td><td>CHAIR↓ Hal↓  $F ^ { 0 . 5 } \uparrow$ </td></tr><tr><td>Vanilla</td><td>0.4176 0.0270</td><td>0.9745 1.0000 0.0296</td></tr><tr><td>+ VCD</td><td>0.4241 0.0372</td><td>0.9777 1.00000.0212</td></tr><tr><td>+ AttnReal 0.4206</td><td>0.0719</td><td>0.9750 0.9993 0.0246</td></tr><tr><td>+ Ours</td><td>0.5458 0.2239</td><td>0.6407 0.7875 0.2564</td></tr></table>

Table 2: Comparison with Training-Free Grounding. Results on TriDF using InternVL-3.5-8B.

## Overall Quantitative Comparison

We first evaluate the generalizability of the proposed trainingfree evidence acquisition framework across diferent MLLM backbones, benchmark datasets, and model scales. Table 1 summarizes the overall quantitative results.

Generalization across MLLM Backbones. Our framework consistently improves image-level accuracy while substantially reducing CHAIR and hallucination rates across all five evaluated open-source MLLM backbones, demonstrating robust generalization across diverse architectures. For example, InternVL-3.5-8B improves accuracy from 0.418 to 0.546 while reducing CHAIR from 0.975 to 0.641. Although artifact coverage decreases for some models, these reductions are consistently accompanied by lower CHAIR and hallucination rates, indicating a shift toward more selective and better-supported artifact descriptions rather than exhaustive but potentially unsupported enumeration.

Generalization across Benchmarks. We further evaluate our framework on MMTD-Set. Our framework improves both ACC and F1 across nearly all model–subset combinations on the DeepFake and AIGC-Editing subsets. Together with the results on TriDF, which also covers diverse manipulation types, these consistent gains demonstrate that the framework generalizes across benchmark datasets and heterogeneous manipulation scenarios.

Comparison with Commercial MLLMs. For reference, we also compare our framework against several commercial MLLMs evaluated under vanilla inference. While these proprietary models generally outperform open-source MLLMs under vanilla inference, applying our framework substantially narrows the gap. In several cases, the enhanced opensource models achieve comparable or even superior explanation grounding despite requiring neither additional training nor proprietary components. These results suggest that efective test-time evidence acquisition can be as critical as mode scaling for explainable forensic reasoning.

## Comparison with Training-Free Visual Grounding

We further compare our framework with VCD (Leng et al. 2024) and AttnReal (Tu et al. 2026), two representative training-free visual grounding methods that improve visual reliance during inference without additional training. As shown in Table 2, all methods are evaluated using the same InternVL-3.5-8B backbone and TriDF setting. VCD and AttnReal yield only marginal gains over vanilla inference: improvements in ACC and Cover are limited, CHAIR and Hal remain nearly unchanged, and $F ^ { 0 . 5 }$ decreases. In contrast, our framework improves all five metrics, reducing CHAIR from 0.9745 to 0.6407 together with a marked decrease in Hal. These results indicate that globally strengthening visual reliance is insuficient for deepfake forensics; efective grounding instead requires spatially selective region discovery and inspection before the final judgment.

<table><tr><td rowspan="2">Method</td><td colspan="4">TriDF</td></tr><tr><td>ACC↑ Cover ↑</td><td>CHAIR↓</td><td>Hal↓</td><td>F0.5 ↑</td></tr><tr><td>Vanilla</td><td>0.4176 0.0270</td><td>0.9745</td><td>1.0000</td><td>0.0296</td></tr><tr><td>Prompt-only</td><td>0.5113 0.1369</td><td>0.8545</td><td>0.9927</td><td>0.1340</td></tr><tr><td>w/o blur contrast</td><td>0.4867 0.1920</td><td>0.8600</td><td>0.95680.1132</td><td></td></tr><tr><td>w/o component steering</td><td>0.4884 0.1890</td><td>0.8574</td><td>0.95480.1182</td><td></td></tr><tr><td>Ours</td><td>0.5458 0.2239</td><td>0.6407</td><td>0.7875 0.2564</td><td></td></tr></table>

Table 3: Ablation Study. Efects of evidence-beforejudgment prompting, blur-contrastive region mining, and component-wise steering on TriDF.

![](images/bf2d23790165e25494a6c6693468f6722895e1bddbed42dc5558fcb974aa6c39.jpg)  
Figure 3: Qualitative Spatial Grounding. Candidate regions discovered by Stages 1 and 2 align with annotated tampered areas without localization supervision.

## Ablation Study

We conduct an ablation study on TriDF using InternVL-3.5- 8B to evaluate the look-before-you-judge procedure, blurcontrastive region mining, and connected-component steering. Table 3 compares Vanilla, Prompt-only, w/o blur contrast, w/o component steering, and the full framework.

Prompted evidence is not necessarily grounded. The prompt-only variant improves detection and artifact coverage over Vanilla, but leaves hallucination metrics largely unchanged. This shows that eliciting more evidence does not ensure that the reported observations are visually supported. Region discovery matters. Using raw attention instead of the blur-contrastive prior underperforms even the promptonly variant, suggesting that globally salient responses do not reliably expose subtle forensic cues. Efective inspection therefore depends on first discovering informative, detailsensitive regions.

Separate inspection improves grounding. Without component-level steering, a single global prior mixes spatially distinct cues and weakens focused inspection. In contrast, the full framework achieves the best overall performance, validating the complete look-before-you-judge pipeline of region discovery, separate inspection, and evidence aggregation.

![](images/34e332edcecbdb3f4942c24bdc747ccaaa0e9f71f4959b56c8f0352734925184.jpg)  
Figure 4: From Global Error to Local Evidence. Top: Vanilla inference overlooks the annotated blending and texture artifact and incorrectly predicts the image as authentic. Bottom: Our framework discovers three candidate regions, inspects them independently, and aggregates the retained local evidence under a global view to produce the correct manipulated verdict.

## Qualitative Spatial Grounding

We qualitatively examine whether regions discovered before judgment correspond to actual manipulation areas. Figure 3 visualizes the top candidate regions produced by blurcontrastive region mining and connected-component grouping on fake images from the DeepFake subset of MMTD-Set. Although the framework is not trained for localization and does not access ground-truth masks during inference, the discovered regions frequently cover manipulated facial boundaries and locally edited structures. These examples suggest that the proposed procedure produces meaningful inspection targets for acquiring local evidence before final judgment.

## Qualitative Analysis

Figure 4 shows how the proposed look-before-you-judge procedure corrects an erroneous global judgment. Vanilla inference predicts the manipulated image as authentic based on broad claims of consistent skin tone and seamless facial integration. In contrast, our framework identifies candidate regions around the eyes, facial boundary, and jawline, where region-wise inspection reveals blending artifacts, abnormal smoothness, and texture inconsistencies. Aggregating these locally grounded observations under the global view leads to the correct manipulated verdict. Additional examples and failure cases are provided in the supplementary material.

## Conclusion

We introduce Look Before You Judge, a formulation that casts explainable deepfake detection as a sequential evidence acquisition problem rather than a single-pass prediction task. Based on this formulation, we develop a fully trainingfree test-time evidence acquisition framework that identifies detail-sensitive candidate regions from blur-contrastive attention responses, inspects them individually, and integrates the resulting local evidence into a final image-level judgment. The framework operates entirely at inference time on frozen MLLMs, requiring neither localization supervision, auxiliary modules, nor parameter updates.

Experiments across five open-source MLLMs and two benchmarks demonstrate consistent improvements in detection accuracy and explanation grounding. Together with the ablation and qualitative analyses, these results support the central premise of Look Before You Judge: trustworthy forensic reasoning should first identify where potential evidence resides, then examine it locally, and only afterward reach a global conclusion. We hope this work motivates future research on evidence-centric DeepFake detection, where explicit evidence acquisition becomes a fundamental component oftrustworthy and interpretable decision making beyond deepfake detection.

## References

An, W.; Tian, F.; Leng, S.; Nie, J.; Lin, H.; Wang, Q.; Chen, P.; Zhang, X.; and Lu, S. 2025. Mitigating object hallucinations in large vision-language models with assembly of global and local attention. In Proceedings of the Computer Vision and Pattern Recognition Conference, 29915–29926.

Bai, S.; Cai, Y.; Chen, R.; Chen, K.; Chen, X.; et al. 2025. Qwen3-VL Technical Report. arXiv preprint arXiv:2511.21631.

Basile, L.; Maiorca, V.; Doimo, D.; Locatello, F.; and Cazzaniga, A. 2026. Head pursuit: Probing attention specialization in multimodal transformers. Advances in Neural Information Processing Systems, 38: 112351–112383.

Chaubey, A.; Guan, X.; and Soleymani, M. 2026. Face-LLaVA: Facial expression and attribute understanding through instruction tuning. In 2026 IEEE/CVF Winter Conference on Applications ofComputer Vision (WACV), 2648– 2660. IEEE.

Chen, L.; Zhang, Y.; Song, Y.; Liu, L.; and Wang, J. 2022. Self-supervised learning of adversarial example: Towards good generalizations for deepfake detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 18710–18719.

Dolhansky, B.; Bitton, J.; Pflaum, B.; Lu, J.; Howes, R.; Wang, M.; and Ferrer, C. C. 2020. The deepfake detection challenge (dfdc) dataset. arXiv preprint arXiv:2006.07397.

Dong, S.; Wang, J.; Ji, R.; Liang, J.; Fan, H.; and Ge, Z. 2023. Implicit identity leakage: The stumbling block to improving deepfake detection generalization. In Proceedings ofthe IEEE/CVF conference on computer vision andpattern recognition, 3994–4004.

Haliassos, A.; Mira, R.; Petridis, S.; and Pantic, M. 2022. Leveraging real talking faces via self-supervision for robust forgery detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 14950– 14962.

Han, Y.-H.; Huang, T.-M.; Hua, K.-L.; and Chen, J.-C. 2025. Towards more general video-based deepfake detection through facial component guided adaptation for foundation model. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, 22995–23005.

Huang, Q.; Dong, X.; Zhang, P.; Wang, B.; He, C.; Wang, J.; Lin, D.; Zhang, W.; and Yu, N. 2024. Opera: Alleviating hallucination in multi-modal large language models via overtrust penalty and retrospection-allocation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 13418–13427.

Huang, Z.; Hu, J.; Li, X.; He, Y.; Zhao, X.; Peng, B.; Wu, B.; Huang, X.; and Cheng, G. 2025. Sida: Social media image deepfake detection, localization and explanation with large multimodal model. In Proceedings of the Computer Vision and Pattern Recognition Conference, 28831–28841.

Jia, S.; Lyu, R.; Zhao, K.; Chen, Y.; Yan, Z.; Ju, Y.; Hu, C.; Li, X.; Wu, B.; and Lyu, S. 2024. Can ChatGPT Detect DeepFakes? A Study of Using Multimodal Large Language Models for Media Forensics. arXiv:2403.14077.

Jiang-Lin, J.-Y.; Huang, K.-Y.; Zou, L.; Lo, L.; Yang, S.-P.; Tseng, Y.-W.; Lin, K.-H.; Chen, C.-L.; Ta, Y.-T.; Wang, Y.- T.; et al. 2026. TriDF: Evaluating Perception, Detection, and Hallucination for Interpretable DeepFake Detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 17087–17098.

Kang, H.; Wen, S.; Wen, Z.; Ye, J.; Li, W.; Feng, P.; Zhou, B.; Wang, B.; Lin, D.; Zhang, L.; et al. 2025. Legion: Learning to ground and explain for synthetic image detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 18937–18947.

Kaul, P.; Li, Z.; Yang, H.; Dukler, Y.; Swaminathan, A.; Taylor, C.; and Soatto, S. 2024. Throne: An object-based hallucination benchmark for the free-form generations of large vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 27228–27238.

Kim, J.; Kim, J.; Kim, Y.; and Cho, S.-B. 2025. Fuzzy Contrastive Decoding to Alleviate Object Hallucination in Large Vision-Language Models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 20572–20581.

Leng, S.; Zhang, H.; Chen, G.; Li, X.; Lu, S.; Miao, C.; and Bing, L. 2024. Mitigating object hallucinations in large vision-language models through visual contrastive decoding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 13872–13882.

Li, L.; Bao, J.; Zhang, T.; Yang, H.; Chen, D.; Wen, F.; and Guo, B. 2020. Face x-ray for more general face forgery detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 5001–5010.

Liu, H.; Tan, Z.; Tan, C.; Wei, Y.; Wang, J.; and Zhao, Y. 2024. Forgery-aware Adaptive Transformer for Generalizable Synthetic Image Detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Lu, H.; Chu, B.; Fu, W.; Nan, G.; Liu, J.; Pan, M.; Li, Q.; Yu, Y.; Wang, H.; and Wang, K. 2026. Reallocating Attention Across Layers to Reduce Multimodal Hallucination. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 4157–4167.

Nguyen, D.; Mejri, N.; Singh, I. P.; Kuleshova, P.; Astrid, M.; Kacem, A.; Ghorbel, E.; and Aouada, D. 2024. Laanet: Localized artifact attention network for quality-agnostic and generalizable deepfake detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 17395–17405.

Qian, Y.; Yin, G.; Sheng, L.; Chen, Z.; and Shao, J. 2020. Thinking in frequency: Face forgery detection by mining frequency-aware clues. In European conference on computer vision, 86–103. Springer.

Qwen Team. 2026. Qwen3.5: Towards Native Multimodal Agents.

Rohrbach, A.; Hendricks, L. A.; Burns, K.; Darrell, T.; and Saenko, K. 2018. Object hallucination in image captioning. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, 4035–4045.

Rossler, A.; Cozzolino, D.; Verdoliva, L.; Riess, C.; Thies, J.; and Nießner, M. 2019. Faceforensics++: Learning to detect manipulated facial images. In Proceedings ofthe IEEE/CVF international conference on computer vision, 1–11.

Selvaraju, R. R.; Cogswell, M.; Das, A.; Vedantam, R.; Parikh, D.; and Batra, D. 2017. Grad-cam: Visual explanations from deep networks via gradient-based localization. In Proceedings of the IEEE international conference on computer vision, 618–626.

Shi, Y.; Gao, Y.; Lai, Y.; Wang, H.; Feng, J.; He, L.; Wan, J.; Chen, C.; Yu, Z.; and Cao, X. 2025. SHIELD: an evaluation benchmark for face spoofing and forgery detection with multimodal large language models. Visual Intelligence.

Sun, J.; Yan, Z.; Zhang, K.-Y.; Yao, T.; and Ding, S. 2026. DFD-HR: Generalizable Deepfake Detection via Hierarchical Routing Learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 13984–13995.

Tang, F.; Liu, C.; Xu, Z.; Hu, M.; Huang, Z.; Xue, H.; Chen, Z.; Peng, Z.; Yang, Z.; Zhou, S.; et al. 2025. Seeing far and clearly: Mitigating hallucinations in mllms with attention causal decoding. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, 26147–26159.

Tu, C.; Ye, P.; Zhou, D.; Bai, L.; Yu, G.; Chen, T.; and Ouyang, W. 2026. Attention reallocation: Towards zero-cost and controllable hallucination mitigation of mllms. International Journal ofComputer Vision, 134(1): 22.

Wang, W.; Gao, Z.; Gu, L.; Pu, H.; Cui, L.; Wei, X.; Liu, Z.; Jing, L.; Ye, S.; Shao, J.; et al. 2025. InternVL3. 5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Eficiency. arXiv preprint arXiv:2508.18265.

Wu, M.; Cai, X.; Ji, J.; Li, J.; Huang, O.; Fei, H.; Jiang, G.; Sun, X.; and Ji, R. 2024. Controlmllm: Training-free visual prompt learning for multimodal large language models. Advances in Neural Information Processing Systems, 37: 45206–45234.

Xiaomi, L.-C.-T. 2025. MiMo-VL Technical Report. arXiv:2506.03569.

Xu, Z.; Zhang, X.; Li, R.; Tang, Z.; Huang, Q.; and Zhang, J. 2025. Fakeshield: Explainable image forgery detection and localization via multi-modal large language models. In International Conference on Learning Representations, volume 2025, 31186–31216.

Yang, Y.; Qian, Z.; Zhu, Y.; Russakovsky, O.; and Wu, Y. 2025. D3: Scaling Up Deepfake Detection by Learning from Discrepancy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Yin, H.; Si, G.; and Wang, Z. 2025. Clearsight: Visual signal enhancement for object hallucination mitigation in multimodal large language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, 14625– 14634.

Zhuang, X.; Zhu, Z.; Xie, Y.; Liang, L.; and Zou, Y. 2025. Vasparse: Towards eficient visual hallucination mitigation via visual-aware token sparsification. In Proceedings of the Computer Vision and Pattern Recognition Conference, 4189–4199.

Zou, Y.; Li, P.; Li, Z.; Huang, H.; Cui, X.; Liu, X.; Zhang, C.; and He, R. 2025. Survey on AI-Generated Media Detection: From Non-MLLM to MLLM. ArXiv.

## Supplementary Material

This supplementary material is organized as follows:

A. Discussion and Limitations explores the boundary of our approach.

B. Evaluation Metrics defines how each reported number is computed, covering the artifact-level Cover, CHAIR, Hal, and $F ^ { 0 . 5 }$ protocol used in TriDF and the ACC/F1 protocol used in MMTD-Set.

C. Self-Proposed Regions experiments with language-level guidance in regional examination instead of the blurcontrastive attention mining in Look Before You Judge.

D. Alternative Region Source replaces the mined regions with fixed landmark-defined facial components corresponding to the eyes, nose, and mouth, while leaving Stage 3 unchanged.

E. Qualitative Results shows additional side-by-side comparisons between vanilla inference and our framework.

F. Implementation Details specifies the model configuration, attention-extraction procedure, per-stage hyperparameters, and evaluation hardware used across all evaluated backbones.

G. Computational Cost & Runtime quantifies the overhead our framework introduces, measured over 100 images by the average number of model invocations, peak GPU memory, and end-to-end inference time per image.

H. Prompt Templates lists the Stage-3 region-inspection and aggregation prompts.

## A. Discussion and Limitations

The blur-contrastive attention prior in our framework should be interpreted as a proposal mechanism for acquiring evidence rather than as a causal attribution of the model’s decision. Our framework does not imply that the selected regions are the sole causes of the final prediction. Instead, they serve as probes for identifying the MLLM’s evidencesensitive attention head, which are subsequently leveraged to recognize candidate forensic evidence.

On the other hand, our framework inherits several limitations from its training-free design. First, because candidate evidence regions are proposed through blur-contrastive attention responses, the framework naturally emphasizes artifacts that are sensitive to local high-frequency perturbations, potentially making it less efective for manipulations characterized by more subtle low-frequency inconsistencies. Future work could explore richer perturbation strategies or adaptive mechanisms to improve coverage across diverse manipulation types. Second, the framework requires whitebox access to decoder attention, limiting its applicability to open-weight MLLMs. Extending evidence acquisition to black-box models remains an important direction for future research. Finally, the quality of the acquired evidence depends on the preservation of fine-grained visual details, and performance may degrade under severe compression, downsampling, or coarse visual-token representations. Incorporating multi-scale visual representations or resolution-aware evidence acquisition may further improve robustness in such scenarios.

Despite these limitations, we believe the proposed evidence acquisition formulation, look-before-you-judge, provides a flexible foundation that can accommodate more advanced proposal mechanisms, perturbation strategies, and visual representations as future MLLMs continue to evolve.

## B. Evaluation Metrics

We adopt the evaluation protocol of TriDF (Jiang-Lin et al. 2026) without modification, and report standard binary classification metrics on MMTD-Set (Xu et al. 2025). Because the artifact-level metrics are central to our claims about explanation grounding, we restate their definitions here so that our numbers can be interpreted and reproduced without reference to the original paper.

## Artifact-Level Metrics on TriDF

Artifact mapping. All artifact-level metrics operate on artifact lists rather than on raw text. Given a model’s free-form response $R ^ { \mathrm { D F } }$ to a Type-B <OEQ> query, an external LLM θ maps the response onto TriDF’s predefined artifact taxonomy $A r t = \{ a r t _ { 1 } , \ldots , a r t _ { n } \}$ , producing a mapped artifact list

$$
R _ { \mathrm { a r t } } ^ { \mathrm { D F } } = \theta \left( R ^ { \mathrm { D F } } \right) ,\tag{S1}
$$

where each entry records whether the response asserts the presence of the corresponding artifact. The reference list $\bar { Y } _ { \mathrm { a r t } } ^ { \mathrm { D F } }$ records the artifacts actually annotated for that sample. This mapping step is what makes the metrics robust to surface wording: two responses describing the same artifact with diferent phrasing map to the same taxonomy entry. It also means that any claim the mapper cannot align with the taxonomy is discarded before scoring.

Cover measures the recall of annotated artifacts, i.e. how much of the ground-truth evidence the explanation recovers:

$$
C o v e r ( R ) = \frac { \left| R _ { \mathrm { a r t } } ^ { \mathrm { D F } } \cap Y _ { \mathrm { a r t } } ^ { \mathrm { D F } } \right| } { \left| Y _ { \mathrm { a r t } } ^ { \mathrm { D F } } \right| } .\tag{S2}
$$

CHAIR (Rohrbach et al. 2018) measures the proportion of asserted artifacts that are not supported by the annotation, and is therefore one minus the precision of the artifact claims:

$$
C H A I R ( R ) = 1 - \frac { \left| R _ { \mathrm { a r t } } ^ { \mathrm { D F } } \cap Y _ { \mathrm { a r t } } ^ { \mathrm { D F } } \right| } { \left| R _ { \mathrm { a r t } } ^ { \mathrm { D F } } \right| } .\tag{S3}
$$

Hal converts CHAIR into a per-sample indicator, so that its average over a dataset is the percentage of responses containing at least one unsupported artifact claim:

$$
H a l ( R ) = { \left\{ \begin{array} { l l } { 1 } & { { \mathrm { i f ~ } } C H A I R ( R ) \neq 0 , } \\ { 0 } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }\tag{S4}
$$

Hal is thus considerably coarser than CHAIR: a response with a single unsupported claim and a response that is entirely fabricated both receive $H a l = 1$ . Values close to 1 should accordingly be read as “almost every response contains some unsupported claim” rather than as a severity measure.

$F ^ { \beta }$ -score combines the two directions into a single indicator. Following THRONE (Kaul et al. 2024), TriDF treats precision as twice as important as recall, since false positives in forensic explanations are typically hallucinationdriven and more damaging than incomplete coverage. Writing $P r e c ( R ) = 1 - \bar { C } \bar { H A } I R ( R )$ for the precision of the artifact claims,

$$
F ^ { \beta } ( R ) = \frac { ( 1 + \beta ^ { 2 } ) P r e c ( R ) C o v e r ( R ) } { \beta ^ { 2 } P r e c ( R ) + C o v e r ( R ) } ,\tag{S5}
$$

where $\beta = 0 . 5$

Penalty conventions. Two cases receive the maximal penalty $C H A I R = 1$ , and consequently $H a l = 1$ and $F ^ { 0 . 5 } = 0 \mathrm { : }$ an empty mapped artifact list, $| R _ { \mathrm { a r t } } ^ { \mathrm { \check { D } F } } | = 0$ , and a manipulated sample classified as authentic. Since the artifact-level metrics require annotated artifacts, they are defined on manipulated samples, whereas accuracy (ACC) is computed over the full real–fake set.

## Detection Metrics on MMTD-Set

On MMTD-Set, we report accuracy (ACC) and F1 on the DeepFake and AIGC-Editing subsets, treating “manipulated” as the positive class. We report both metrics because ACC summarizes overall classification performance, whereas F1 focuses on the detection of manipulated samples and better exposes models that are biased toward predicting “authentic.”

Reporting precision. Scores for FakeShield (Xu et al. 2025) are reported to two decimal places in the original publication. We zero-pad them to three decimal places for column alignment with our own measurements and introduce no additional precision.

## C. Self-Proposed Regions

We further examine whether the gains of our framework can be explained merely by multi-stage prompting or textual region decomposition. In this baseline, the MLLM first proposes up to three small localized regions for close inspection. The same region-level inspection and final verification procedure is then applied, but without blur-contrastive region mining or attention reallocation.

As shown in Table S1, Self-Proposed Region Inspection improves over vanilla inference, indicating that explicitly decomposing the task into local inspection and final aggregation can provide a useful reasoning structure. However, it does not outperform Prompt-only on ACC, Cover, CHAIR, or $F ^ { 0 . 5 }$ , and remains substantially behind our full framework. This suggests that asking the model to propose and inspect its own regions does not reliably identify manipulation-relevant evidence; the selected regions may still be driven by semantic saliency or language priors rather than image-specific forensic cues.

In contrast, our full framework achieves the strongest overall performance across detection, artifact grounding, and hallucination reduction. These results show that efective lookbefore-you-judge reasoning requires more than textual region decomposition: the model must first discover detailsensitive, image-specific regions and inspect them under region-specific attention steering.

<table><tr><td>Method</td><td colspan="3">TriDF</td></tr><tr><td></td><td>ACC ↑ Cover ↑</td><td>CHAIR↓</td><td> $H a l \downarrow$   $F ^ { 0 . 5 } \uparrow$ </td></tr><tr><td>Vanilla</td><td>0.4176 0.0270</td><td>0.9745</td><td>1.0000 0.0296</td></tr><tr><td>Prompt-only</td><td>0.5113 0.1369</td><td>0.8545</td><td>0.9927 0.1340</td></tr><tr><td>Self-Proposed Region Inspection 0.5076</td><td>0.0919</td><td>0.8870</td><td>0.9495 0.0836</td></tr><tr><td>Ours</td><td>0.5458 0.2239</td><td>0.6407</td><td>0.7875 0.2564</td></tr></table>

Table S1: Comparison with Self-Proposed Region Inspection. Results on TriDF using InternVL-3.5-8B.

## D. Alternative Region Source

We further examine whether the gains of our framework can be reproduced by steering the model toward predefined facial components. Such anatomical priors are a wellestablished source of localized supervision in face forensics: DFD-FCG (Han et al. 2025) adapts a foundation model for video-based deepfake detection through facial-component guidance, and Face-LLaVA (Chaubey, Guan, and Soleymani 2026) injects face-region priors into an MLLM through faceregion guided cross-attention. Both indicate that directing a model’s capacity toward facial parts is beneficial, which makes a facial-component prior the natural non-learned alternative to our mined regions. We therefore construct a facialregion variant that replaces Stages 1–2 with landmark-based eye, nose, and mouth regions, and distributes the recycled attention uniformly within each region. All remaining settings, including the region-inspection prompt, intervention layers, evidence filtering, and final aggregation, are kept identical to the full method.

The facial-region variant substantially improves over vanilla inference across both detection and explanationgrounding metrics. This result shows that decomposing the image into local inspection regions and reallocating attention during region-wise evidence acquisition is itself efective, even when the candidate regions are defined by a simple face-centric prior. In other words, part of the improvement comes from explicitly directing the model to inspect localized visual evidence before making the final judgment.

Nevertheless, the full method consistently performs best across all metrics. Since the downstream inspection and aggregation procedures are unchanged, this remaining gap indicates that the efectiveness of regional attention reallocation depends not only on applying the intervention, but also on where the redistributed attention is directed. Predefined facial components provide useful inspection targets for face-related artifacts, but they may spend the limited region budget on less informative facial areas or fail to cover manipulations located elsewhere in the image. In contrast, our blur-contrastive prior derives image-specific regions from the input itself, allowing the inspection targets to adapt across facial manipulations, edited objects, and regenerated image regions. This adaptability is particularly important for settings such as image editing, where the face may occupy only a small portion of the image and the diagnostically useful evidence may instead appear in non-facial content.

<table><tr><td rowspan="2">Method</td><td colspan="2">TriDF</td></tr><tr><td>ACC↑ Cover ↑</td><td>CHAIR↓ Hal ↓  $F ^ { 0 . 5 } \uparrow$ </td></tr><tr><td>Vanilla</td><td>0.4176 0.0270</td><td>0.9745 1.00000.0296</td></tr><tr><td>+ Facial Regions 0.5176</td><td>0.2096</td><td>0.6727 0.80140.2343</td></tr><tr><td>+ Ours</td><td>0.5458 0.2239</td><td>0.6407 0.7875 0.2564</td></tr></table>

Table S2: Comparison with Alternative Region Selection Method. Results on TriDF using InternVL-3.5-8B.

## E. Qualitative Results

We provide three additional examples of successful cases and one representative failure case. These examples show how region mining and region-wise inspection can correct erroneous global judgments, improve explanation grounding, and calibrate ambiguous local evidence.

Correcting erroneous global judgments. Figures S1 and S2 show complementary cases in which our framework corrects vanilla inference. In the manipulated image generation example, vanilla relies on broad claims of natural appearance and overlooks localized blur and texture inconsistencies. Our framework inspects the discovered regions and integrates the resulting evidence into the correct manipulated verdict.

In the authentic-face example, vanilla instead hallucinates manipulation cues around the hair, skin, and eyes. Although our framework also selects locally ambiguous regions, it describes them with low evidential strength and retains a negative observation. Global aggregation rejects the unsupported cues and produces the correct authentic prediction. Together, these examples show that the method calibrates local evidence rather than simply increasing sensitivity to manipulation.

Improving explanation grounding. Figure S3 shows that a correct verdict does not necessarily imply a grounded explanation. Vanilla correctly predicts manipulation but describes generic hair and background anomalies. Our framework instead links its findings to inspected regions around the hairline and forehead, where the observed blending and texture irregularities more closely match the annotated artifacts. The final explanation is therefore more localized and visually traceable.

Failure case. Figure S4 presents a manipulated image that both methods classify as authentic. Vanilla supports its decision using broad claims about natural scene appearance. Our framework provides a more localized observation by identifying blur in the railway background, but this cue remains ambiguous because it may also result from natural focus variation. Moreover, the selected regions do not reveal sufficiently strong complementary evidence of blending, color inconsistency, or object-integrity flaws.

This example suggests that the current framework may be less efective when manipulation evidence is spatially difuse, background-dominated, or naturally explainable in isolation. Nevertheless, its intermediate reasoning remains better calibrated than vanilla inference, as it identifies a relevant local cue without treating it as conclusive evidence.

## F. Implementation Details

We instantiate the three-stage framework identically across all evaluated backbones, changing only the quantities that are inherently architecture-dependent: the decoder layer/head counts $L , H ,$ the visual-token grid $H _ { v } \times W _ { v }$ , and the steering layer range $\mathcal { L } ^ { \star }$ . All other hyperparameters are shared, apart from two adjustments required by the hybrid attention design of Qwen3.5-9B, which we state explicitly below. This section gives the concrete instantiation for InternVL-3.5-8B, the backbone used for the ablation studies, and describes how the remaining backbones difer.

Model and inference environment. InternVL-3.5-8B pairs a 448 × 448 InternViT vision encoder (patch size 14, 0.5× pixel-shufle downsampling) with a 36-layer, 32-head Qwen3 decoder (8 key–value heads under grouped-query attention). We disable dynamic image tiling (max\_num=1), so every image is encoded as a single tile of $H _ { v } \mathrm { ~ \times ~ }$ $W _ { v } = \bar { 1 } 6 \times \bar { 1 6 } = 2 5 6$ visual tokens. All backbones run in bfloat16 on a single GPU. Because Stage 1 and Stage 3 both require materialized per-head attention weights, we load the model with FlashAttention2 disabled and request output\_attentions=True; region-wise steering (Stage 3.a) is implemented as a forward-pass patch over the decoder’s self-attention module rather than a modification to the generation config. All generation calls – the Stage 1 fixed continuation, Stage 3.a region inspection, and Stage 3.b final aggregation – use greedy decoding (do\_sample=False, one beam), so outputs are deterministic given fixed weights and inputs.

Per-stage hyperparameters. We instantiate Stage 1 with $L \times H ^ { - } = 3 6 \times 3 2$ decoder layers and heads, a Gaussian blur radius of 3.0px defining $I ^ { \mathrm { b l u r } }$ , and a fixed continuation capped at 64 generated tokens. Head selection retains up to $K _ { H } = 2 0$ heads under fixed diversity constraints (at most 4 heads per layer, minimum grid-distance 2 between selected peaks) and fixed nondegeneracy checks (bounds on border mass, peak mass, efective tokens, and $\Delta ^ { + }$ mass); heads failing any check are excluded from H regardless of score. Stage 2 operates on the $H _ { v } \times W _ { v } = 1 6 \times 1 6$ visual-token grid, retains the top-p fraction of grid cells with $p = 0 . 1 5 ,$ , groups them with 8-connectivity, discards components smaller than 2 grid cells, dilates survivors by one grid step, and keeps up to $K = 3$ ranked regions. Stage 3 uses retention coeficient $\rho = 0 . 5$ with sink threshold 1.0, steers all heads within decoder layers $\mathcal { L } ^ { \star } = [ 1 2 , 3 1 ]$ , and caps region inspection and final aggregation at 96 and 384 generated tokens, respectively. During evidence filtering (Stage 3.b), an observation is admitted only at medium confidence or higher, and it is dropped as a near-duplicate of an already-kept observation when it shares the same evidence label and artifact type and its normalized text similarity exceeds 0.82. Observations labeled uncertain are, by default, retained as weak cues rather than discarded outright, provided they name a concrete artifact type and meet the confidence floor.

Adapting to other backbones. The steering layer range $\mathcal { L } ^ { \star }$ and the visual-token grid $H _ { v } \times W _ { v }$ are the two hyperparameters above that are inherently architecture-dependent (Section A); we identify each once per backbone family by inspecting its transformers implementation and hold it fixed across all of that backbone’s evaluations, rather than tuning it per sample. We use $\mathcal { L } ^ { \star } = [ 1 2 , 3 1 ]$ (of $L = 3 6$ layers, $H = 3 2 )$ for InternVL-3.5-8B, $\mathcal { \bar { L } } ^ { \star } = [ \mathrm { i } 9 , 3 9 ]$ (of $L = 4 0$ layers, $H = 4 0 )$ for InternVL-3.5-14B, $\mathcal { L } ^ { \star } = [ 1 8 , 3 5 ]$ (of $L = 3 6 , H = 3 2 )$ for Qwen3-VL-8B-Instruct, and the full decoder depth for MiMo-VL-7B. All remaining hyperparameters above are reused unchanged across backbones, with the single exception noted below for Qwen3.5-9B.

![](images/01a566f3fefd962cdd453b6993c708f6c3092ef3c11ace3ecc9eeb77fd7e0b9d.jpg)  
Figure S1: Qualitative result on a manipulated image generation example. Vanilla inference overlooks the localized artifacts and predicts the image as authentic. Our framework identifies blur and texture/color inconsistencies in the selected regions and produces the correct manipulated verdict.

![](images/0dc60f06507b4138c718e3aca2ec600a4bad12daf1bb7b1e84409247205c1368.jpg)  
Figure S2: Qualitative result on an authentic face. Vanilla inference hallucinates facial artifacts and produces a false positive. Our framework assigns only weak, uncertain, or negative evidence to the inspected regions and recovers the correct authentic verdict.

![](images/25b18e33daf501418cc944b78f52ec38ff1e8bcbb3e3c48c356bb3c2479f654d.jpg)  
Figure S3: Qualitative comparison when both methods correctly predict manipulation. Vanilla inference provides generic and weakly supported claims, whereas our framework grounds its explanation in blending and texture irregularities around the hairline and forehead.

![](images/56ee64bcb5bae8f06d74768e12623b37f06e471b81856ecaa1ce575aa88a13c2.jpg)  
Figure S4: Representative failure case on a manipulated outdoor scene. Our framework identifies a relevant background-blur cue, but the cue is also compatible with natural focus variation and lacks suficient complementary evidence. Both method therefore predict the image as authentic.

<table><tr><td rowspan="2">Method</td><td colspan="4">Inference Efficiency</td></tr><tr><td> $A \nu g .$  Invoc.↓</td><td>Time (s/img)↓</td><td>Rel. Latency↓</td><td>Peak GPU Mem. (GB)↓</td></tr><tr><td>Vanilla</td><td>1.000</td><td>13.480</td><td>1.000×</td><td>16.129</td></tr><tr><td>Ours</td><td>6.970</td><td>19.951</td><td>1.480×</td><td>16.624</td></tr></table>

Table S3: Inference eficiency comparison. inference eficiency on a balanced 100-image subset of TriDF. Both methods use the same InternVL-3.5-8B backbone under identical single-GPU settings. A model invocation denotes either an autoregressive generation or an explicit attention forward evaluation.

Hybrid-attention backbones: Qwen3.5-9B. Qwen3.5-9B is architecturally hybrid, and is the one backbone that required an adjustment beyond $\mathcal { L } ^ { \star }$ and the grid. Of its $L = 3 2$ decoder layers, only eight use full self-attention – layers $\{ 3 , 7 , 1 1 , 1 \dot { 5 } , 1 9 , 2 3 , 2 7 , \dot { 3 } 1 \}$ , with $H ~ = ~ 1 6$ heads each – while the remaining 24 use linear (GatedDeltaNet) attention and are therefore not steerable under our formulation. We accordingly set $\mathcal { L } ^ { \star }$ to this fixed eight-layer subset and steer all 16 heads of any layer it selects. This leaves only 128 candidate layer–head cells, an order of magnitude fewer than the other backbones, so the shared minimum-peak-distance gate (default 2 grid cells) admits only about 9 of the intended $K _ { H } = 2 0$ heads on this backbone; we therefore lower it to 1 for Qwen3.5-9B alone, which restores selection to 15–17 heads at a modest cost in spatial diversity among the selected heads. Its visual-token grid $H _ { v } \times W _ { v }$ is likewise smaller than the other backbones’ and varies slightly across images because preprocessing preserves the input aspect ratio. Finally, because its responses are longer, we raise its Stage 3 region-inspection and final-aggregation budgets to 256 and 512 generated tokens (versus 96 and 384 elsewhere) to avoid truncation. All remaining hyperparameters are unchanged.

Hardware. All models are evaluated on a single GPU without model or tensor parallelism. InternVL-3.5-8B, Qwen3- VL-8B-Instruct, Qwen3.5-9B, and MiMo-VL-7B use an NVIDIA RTX 4090 (24 GB), while InternVL-3.5-14B uses an NVIDIA RTX PRO 6000 Blackwell (96 GB) to support its larger eager-attention memory footprint. Section G compares vanilla inference and our framework on the same device.

## G. Computational Cost & Runtime

We evaluate inference eficiency on a balanced 100-image subset of TriDF, covering five image manipulation tasks with 10 of each real and fake samples per task. Vanilla inference and our method use the same InternVL-3.5-8B backbone, BF16 precision, 448 × 448 inputs, and single-GPU execution. Model loading, warm-up, and result serialization are excluded, while the same per-sample memory cleanup procedure is retained for both methods. We report the average number of high-level model invocations, inference time per image, relative latency, and peak allocated GPU memory. An invocation denotes either an autoregressive generation or an explicit attention forward evaluation, rather than an individual decoding step.

As shown in Table S3, vanilla inference requires one invocation and takes 13.480 seconds per image, with 16.129 GB peak memory. Our method averages 6.970 invocations and takes 19.951 seconds per image, corresponding to 1.480× relative latency. Peak memory increases by only 0.495 GB, from 16.129 GB to 16.624 GB, or approximately 3.1%. The invocation count is slightly below seven because the number of candidate regions is input-dependent. Stage 1 uses three fixed invocations, Stage 2 introduces no model invocation, and Stage 3 uses one inspection call per retained region followed by final aggregation. The total is therefore $4 + K$ where $\dot { K } \le 3 ;$ the observed average corresponds to 2.970 regions per image.

The increase in invocation count does not produce a proportional increase in runtime because the additional calls are substantially cheaper than a full vanilla generation. Our method distributes a comparable overall generation budget across short intermediate outputs and final aggregation, rather than repeating a full-length response at every stage. In addition, Stage 1 generates the continuation only once and reuses it for the raw and blurred attention evaluations, avoiding an additional autoregressive generation. Stage 2 consists only of lightweight thresholding and connected-component processing, while the regional inspections produce short, structured observations.

The limited memory increase also follows from the sequential design. All stages reuse the same backbone, and the additional attention maps and steering states are transient rather than maintained as parallel model branches. Overall, the proposed framework introduces a moderate 48.0% latency overhead and approximately 3.1% additional peak GPU memory despite requiring nearly seven high-level invocations. This indicates that its computational cost grows substantially more slowly than invocation count while supporting multi-stage region mining and evidence aggregation.

## H. Prompt Templates

All evaluated backbones use the same Stage 3 prompting protocol: they inspect only the attention-guided local region, compare it with nearby visual context, and produce structured observations instead of image-level decisions. The retained observations are then verified under a full-image view before generating the final prediction and explanation.

To accommodate diferences in instruction following, output-format compliance, and confidence calibration across MLLM backbones, we apply only minor wording adjustments to the shared prompts. These changes do not afect the region proposals, attention steering, evidence schema, filtering rules, or evidence-before-judgment workflow.

## Prompt S1: Region-Wise Evidence Inspection.

You are a forensic media authenticity inspector performing a local region-level inspection. Inspect only the attention-guided local region. Do not make the final image-level authentic/manipulated decision at this stage. Examine the strongest visible property within the guided region and compare the local structure with its nearest relevant visual context, such as an adjacent boundary, surface, body part, object, shadow, reflection, or repeated pattern. Focus on concrete and observable visual evidence rather than speculation. Classify the local observation as:

• yes: a specific localized contradiction consistent with editing or synthesis is clearly visible;

• uncertain: a visible irregularity exists, but a plausible natural explanation remains;

• no: the local structure appears visually coherent.

Generic blur, smoothness, compression, focus variation, lighting variation, or image quality alone is not suficient manipulation evidence. The artifact type should describe the concrete visible cue being evaluated rather than an inferred manipulation process. Return exactly:

• Observation: <one concise sentence describing the local comparison and visible result>

• Manipulation Evidence: yes / no / uncertain

• Confidence: high / medium / low

• Artifact Type: <short visual artifact or none>

• Natural Explanation: <brief explanation for no or uncertain; otherwise none>

## Prompt S2: Global Evidence Aggregation.

You are a forensic media authenticity inspector. Determine whether the provided image is authentic or manipulated. The region-level observations below are tentative candidate cues rather than verified findings.

First verify whether each retained cue is visibly supported by the full image and consistent with its surrounding visual context. Give greater weight to specific localized contradictions in boundary, texture, geometry, lighting, shadow, reflection, object structure, occlusion, or physical contact than to generic image properties.

Weak or ambiguous observations should be treated as lowerconfidence cues when a plausible natural explanation remains. Repeated observations describing the same underlying visual effect should be consolidated rather than counted as independent evidence. A mostly natural-looking image does not invalidate a concrete local contradiction, but normal global context may weaken generic or unsupported cues.

Retained region-level observations:

## Filtered Region Observations

Base the final decision on the specificity, visual support, consistency, and overall strength of the verified evidence.

Return exactly:

## Forensic Analysis:

<In one or two sentences, summarize which local cues remain visually supported after the full-image verification and how strongly they support the manipulation hypothesis.>

Final Decision: Likely Authentic or Likely Manipulated.

## Artifact Findings:

\- Title of artifact: <short visual artifact or none>

\- Reason: <brief technical rationale grounded in visible evidence>