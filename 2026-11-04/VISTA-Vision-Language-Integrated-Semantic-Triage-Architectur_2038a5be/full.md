# VISTA: Vision-Language Integrated Semantic Triage Architecture for Online Face Clustering in Long Videos

Jiashu Liao University of Glasgow Glasgow, United Kingdom jiashu.liao@glasgow.ac.uk

Sanjay Sharma The University of Warwick Coventry, United Kingdom sanjay.sharma@warwick.ac.uk

Tanaya Guha University of Glasgow   
Glasgow, United Kingdom   
tanaya.guha@glasgow.ac.uk

## Abstract

Clustering faces by identity in long, continuous video streams is a challenging task that serves as a building block for human-centric video analysis. The majority of existing face clustering models are ofline, i.e., they assume the entire data to be available altogether. In many practical scenarios, however, the entire data is either unavailable or poses computational challenges to be processed in its entirety. A handful of online methods that exist rely on geometric visual features and simple thresholding techniques, which fail when the faces exhibit significant appearance variations in long, unconstrained videos. To this end, we present VISTA, the first vision language framework for online face clustering in long videos. VISTA leverages semantic reasoning in a controlled, uncertainty aware fashion to maintain both high cluster purity and long range identity continuity. This makes it robust to the variations in illumination, pose, and costume changes that the faces and characters exhibit in long real world videos. Extensive experiments on the VPCD benchmark dataset (with 15 full episodes from The Big Bang Theory [2007-2019], Bufy the Vampire Slayer [1997- 2003], and Sherlock [2010-2017]) demonstrate that VISTA achieves a robust precision and recall balance, consistently improving over the strongest matched online unsupervised baseline (COFC) under identical detector, embeddings, and evaluation protocol. Code and supplementary material: https://github.com/SkyJiashu/VISTA.

## CCS Concepts

• Computing methodologies → Video segmentation; Object identification; Tracking; Cluster analysis.

## Keywords

Face Clustering, Online Clustering, Vision Language Models, Multimodal Learning, Long Videos

## ACM Reference Format:

Joanne Garde-Hansen   
University of Leeds   
Leeds, United Kingdom   
j.garde-hansen@leeds.ac.uk

Jiashu Liao, Joanne Garde-Hansen, Sanjay Sharma, and Tanaya Guha. 2026. VISTA: Vision-Language Integrated Semantic Triage Architecture for Online

![](images/24b81351952f1d39b3fecde435b408a8c6ed45231a95512b38d1e2d6a9e886b8.jpg)

Face Clustering in Long Videos. In Proceedings of the 34th ACM International Conference on Multimedia (MM ’26), November 10–14, 2026, Rio de Janeiro, Brazil. ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/3767308. 3835422

## 1 Introduction

The explosive growth of long form video content demands robust algorithms to facilitate a number of human centric video understanding tasks [37], such as automated character identification and narrative analysis [3, 11, 31] and personalised video browsing [15]. At the core of these applications lies video face clustering i.e., grouping faces detected over a video stream based on identity, where each face cluster represents a unique person. The majority of existing video face clustering methods are ofline methods i.e., they process all data points simultaneously through global optimization [3, 33, 35]; however, the scale of modern video streams demands online, single pass processing algorithms that can maintain low clustering latency and remain compatible with streaming.

Strictly online face clustering algorithms are scarce in the literature due to the constraints of single pass, low latency processing. To the best of our knowledge, Constrained Online Face Clustering (COFC) [16] remains the state of the art for online, unsupervised face clustering. It operates by generating them incrementally using geometric embeddings (e.g., FaceNet [25]) and spatio temporal must link/cannot link constraints. Despite the eficiency, such pure visual methods fail under extreme variations in pose and illumination, thereby often confusing identities. Several works have attempted to address this via improved face representations [7] or graph based linkage technique [36], but the fundamental limitation is still relying solely on visual face similarity.

Recently, Vision Language Models (VLMs) like CLIP [22] have been successful in improving ofline clustering [29] by extracting zero shot attributes (e.g., gender, age, clothing). However, applying VLM attributes directly as hard constraints introduces a critical prediction instability problem; i.e., in unconstrained videos, large variation in illumination causes VLM ‘hallucinations’ [19] (e.g., a red jacket identified as black in the dark) [29]. Furthermore, as characters in long videos frequently change costumes, hard constraints based on hair or clothing will force the algorithm to reject valid faces. Because continuous temporal sequences are broken when intermediate faces are rejected, this leads to the artificial fragmentation of face trajectories. This over segmentation artificially

This work is licensed under a Creative Commons Attribution 4.0 International License. MM ’26, Rio de Janeiro, Brazil   
© 2026 Copyright held by the owner/author(s).   
ACM ISBN 979-8-4007-2213-4/2026/11   
https://doi.org/10.1145/3767308.3835422

![](images/d8f088eccb9663653c46ba361e7ca5c6514998509ce19d239d164dec82db1ac5.jpg)  
Figure 1: The VISTA pipeline. Stage 1 extracts geometric and VLM features. Stage 2 forms intra shot tracks. Stage 3 performs inter shot association via SSR mapping and dynamic three zone semantic triage.

inflates Weighted Cluster Purity (WCP) while degrading global identification accuracy [9].

The basic idea of online face clustering is to group face tracks based on identity by processing short video segments (1 sec) as they come. If an identity has already been discovered, the new face track will be merged with that cluster, otherwise, a new cluster will be formed considering a ’new’ identity. To harness VLM reasoning within this paradigm (without falling into the over segmentation trap), we propose the Vision language Integrated Semantic Triage Architecture (VISTA). VISTA operates as a fully unsupervised, online clustering pipeline enriched by three semantic modules. (1) First, we introduce the idea of Symbolic Semantic Representation: Instead of comparing raw VLM predictions, SSR maps noisy coarse age attributes (e.g., oscillating age predictions like "20-29" vs. "30-39") to a compact set of stable discrete tokens (e.g., <age\_adult>) via a predefined deterministic mapping table (which we term a symbol registry), acting as a temporal denoising layer. (2) We propose a three zone semantic triage mechanism to im prove clustering accuracy for the dificult samples. By maintaining a sliding window of the most recent 150 geometric similarity scores between track cluster pairs, we dynamically fit a Gaussian Mixture Model (GMM) [23] to partition these track to cluster association candidates into three zones: high confidence (merge), reject, and ambiguity zone. The ambiguity zone requires VLM arbitration. (3) To resolve the ambiguity we introduce spatio temporal decoupled semantic arbitration. VISTA decouples attributes into persistent (gender, age) and transient (clothing, hair) attributes. By applying a low confidence semantic masking mechanism during poor illumination to filter out VLM hallucinations, and a temporal decay for transient attributes across long temporal gaps (bypassing costume changes), VISTA achieves more robust global identity tracking. To summarise, our main contributions are:

• We present VISTA, an online vision language framework that fuses geometric embeddings with zero shot semantic attributes. By designing the semantic integration to be lightweight, our core clustering module operates at over 1800 FPS (clustering module only; the end to end pipeline including detection and feature extraction runs at 7–8 FPS, Section 4.6), closing the semantic gap between low level geometric features and human interpretable identity without compromising online eficiency.

• We introduce the idea of a Symbolic Semantic Representation and a Three Zone Semantic Triage scheme with GMM based dy namic thresholding to adapt to the high temporal variability in facial/personal attributes in long videos. Since COFC is already online and unsupervised, the novelty of VISTA lies specifically in this selective online semantic control layer.

• We identify and address the VLM induced over segmentation problem through a spatio temporal decoupled semantic arbitration strategy. Rather than naively enforcing semantic constraints, VISTA uses semantics selectively to reduce cross identity confusion while maintaining a favorable trade of between cluster purity and identity completeness across challenging long video benchmarks.

## 2 Related Work

Face and person clustering in videos. Video face clustering has traditionally been framed as an ofline task. Standard approaches extract deep CNN features and perform global optimization such as graph linkage [38] or test time finetuning [33]. Beyond pure face features, person level representations incorporating body and clothing cues have been explored to disambiguate visually similar faces [28]. Sarfraz et al. proposed FINCH [24], a parameter free first nearest neighbour clustering method that achieves competitive results with pre trained features. Building on this, Kalogeiton and Zisserman introduced C1C [12], which combines first nearest neighbour clustering with spatio temporal cannot link constraints derived from the video structure, improving cluster purity. For streaming scenarios, Kulshreshtha et al. proposed COFC [16], utilizing temporal overlap within shots as a strict cannot link constraint (� matrix) to guide incremental updates; this online face clustering paradigm was later extended to construct dynamic character graphs for movie analysis [17]. Supervised methods such as Ball Cluster Learning (BCL) [33] learn to map features into pre defined metric balls; on the VPCD benchmark, BCL results are reported under the AT (unknown number of clusters) and OC (oracle number of clusters) protocols [3], and BCL is neither unsupervised nor streaming. Multi modal approaches like MuHPC [3] hierarchically fuse face, voice, and body features to bridge otherwise unmergeable clusters, achieving strong ofline results on the VPCD benchmark but remaining inherently ofline and requiring multiple modalities. More recently, VideoClusterNet [35] proposed self supervised refinement of face embeddings during inference, but requires multiple passes over the data. Because recent advancements have predom inantly focused on ofline or supervised paradigms, COFC [16] remains the most representative baseline for fully online, unsupervised settings. We note that existing ofline methods cannot be trivially adapted to online settings: MuHPC requires global hierarchical merging across the full dataset, VideoClusterNet performs multi pass self supervised refinement, and C1C relies on a complete first-nearest-neighbour graph, all fundamentally requiring full data access. VISTA builds upon this online tracking paradigm but expands its inter shot capability by incorporating dynamic semantic triage while remaining fully unsupervised and single pass.

VLMs for long video understanding. The advent of VLMs like CLIP [22] has revolutionized zero shot video understanding. For face analysis, Shen et al. introduced CLIP-Cluster [29], showing that VLMs can extract semantic attributes to assist in face grouping. Subsequent large scale vision language models such as BLIP2 [18] and LLaVA [20] further improved zero shot recognition capabilities, enabling richer attribute extraction from video frames. However, VLM outputs are susceptible to motion blur and illumination drops in the wild. DeepFace [26] ofers a complementary source of demographic predictions but similarly degrades under low image quality. Earlier attempts at naming TV characters also leveraged multimodal cues from subtitles and dialog [11], yet these require structured metadata not always available in general video streams. VISTA repurposes VLM predictions as a source of symbolic supervision, employing a spatio temporal decoupling mechanism to dynamically penalize or ignore semantic contradictions based on environmental context, thereby bridging the gap between powerful zero shot capabilities and the practical instability of real world deployment.

## 3 Methodology

The VISTA framework processes streaming video through three cascaded stages: (1) Feature Extraction, (2) Intra Shot Tracking, and (3) Inter Shot Semantic Association. Figure 1 illustrates the overall pipeline. Pseudocode for the complete online clustering procedure is provided in the Supplementary Material. Geometric embeddings and CLIP/DeepFace attributes are extracted up front for every face crop in Stage 1; triage controls only how often semantic arbitration fires (Section 3.3).

## 3.1 Problem Formulation

Given a continuous video stream $\mathcal { V } = \{ f _ { 1 } , f _ { 2 } , . . . \}$ , we detect face instances and seek to assign each detection to an identity cluster in an online fashion, without revisiting past frames. Let $\mathcal { D } _ { t } ~ =$ $\{ d _ { 1 } ^ { t } , d _ { 2 } ^ { t } , \ldots \}$ denote detected faces at frame �, each associated with a geometric embedding $\boldsymbol { v } \in \mathbb { R } ^ { 5 1 2 }$ and a set ofraw attribute predictions $\mathcal { A } = \{ a _ { \mathrm { g e n d e r } } , a _ { \mathrm { a g e } } , a _ { \mathrm { c l o t h i n g } } , a _ { \mathrm { h a i r } } \}$ . The goal is to maintain a set of global identity clusters $\overset { \triangledown } { \boldsymbol { C } } = \left\{ \boldsymbol { C } _ { 1 } , \boldsymbol { C } _ { 2 } , \ldots , \boldsymbol { C } _ { K } \right\}$ where � is unknown and grows dynamically. Each incoming face track must be either merged into an existing cluster or instantiated as a new identity, using only information available up to the current time step.

## 3.2 Feature Extraction and Intra Shot Tracking

Faces are detected using RetinaFace [8] and represented via FaceNet [25] embeddings. While our architecture is modular and supports alternative backends (e.g., InsightFace [10]), we employ this specific setup by default to maintain a strict, fair comparison with classical online baselines like COFC. Within a camera shot, we construct short term face tracks � based on bounding box IoU and $L _ { 2 }$ feature distance. Specifically, a detection $d _ { i } ^ { t }$ is appended to an existing track $T _ { j }$ if IoU $\lceil ( d _ { i } ^ { t } , d _ { j } ^ { t - 1 } ) \ > \ \theta _ { \mathrm { i o u } }$ and $\| v _ { i } ^ { t } - \bar { v } _ { j } \| _ { 2 } < \theta _ { \mathrm { f e a t } }$ , where $\bar { v } _ { j }$ is the running average embedding $\operatorname { o f } T _ { j }$ . Each track accumulates up to $L _ { \mathrm { m a x } }$ frames (default 24) to form a stable representation. To enforce physical realism, we construct a temporal constraint matrix $Q .$ . If two tracks co-occur in the same frame, $Q _ { A , B } = 0$ (cannot link constraint); otherwise $Q _ { A , B } = 1$

## 3.3 Inter Shot Clustering: Semantic Association

Symbolic Semantic Representation (SSR). Applying raw VLM textual predictions $( \mathrm { e . g . , \dot { A } g e : 2 0 - 2 9 ^ { \circ } } )$ directly is fragile due to frame level noise. VISTA defines a Symbol Registry, consisting of deterministic mapping functions $\phi _ { d } : L _ { d } \to S _ { d }$ that convert coarse age grouping labels into discrete symbols. For example, both ‘Age: 20-29’ and ‘Age: $3 0 – 3 9 '$ map to the symbol <age\_adult>, absorbing boundary oscillations. We define four attribute dimensions $d \in \{ \mathrm { g e n d e r }$ , age, clothing, hair}, each with a compact symbol vocabulary $| S _ { d } | \in \{ 2 , 5 , 7 , 5 \}$ } respectively. The exact mapping schemes and an empirical comparison between diferent age symbol configurations are provided in the Supplementary Material. Overall, the coarse grouping used in our unified setting provides a robust compromise across datasets by reducing boundary oscillation while retaining more discriminative power than broader adult collapsing.

Over the duration of a track, VISTA accumulates these symbols. The track’s final SSR profile is generated via a temporal majority vote, acting as a robust denoising mechanism. The confidence $p ( s )$ of a symbol � in a track is defined as its frequency of occurrence across all valid frames in the track:

$$
p _ { T } ( s _ { d } ) = \frac { | \{ f \in T : \phi _ { d } ( a _ { d } ^ { f } ) = s _ { d } \} | } { | \{ f \in T : a _ { d } ^ { f } \neq \mathsf { n u l l } \} | } .\tag{1}
$$

Only frames where the attribute prediction passes a minimum detection quality threshold contribute to the vote, filtering out corrupted observations.

Three zone triage with GMM dynamic thresholding. To evaluate a merge between a query track � and a global cluster $C ,$ we compute the cosine similarity sim $( v _ { T } , v _ { C } )$ . Using thresholds �<sub>high</sub> and $\tau _ { \mathrm { l o w } } .$ , VISTA triages pairs into three zones, each activating a diferent mode of semantic arbitration: (1) High confidence zone (sim $> \tau _ { \mathrm { h i g h } } )$ : merge is accepted; semantic cues contribute bonus only adjustments (no penalties) to reinforce the decision; (2) Ambiguity zone $( \tau _ { \mathrm { l o w } } \leq \mathrm { s i m } \leq \tau _ { \mathrm { h i g h } } ) ;$ full semantic arbitration with both rewards and penalties; (3) Reject zone $( \sin < \tau _ { \mathrm { l o w } } ) ;$ merge is rejected by default, but a rescue mechanism can override the rejection when persistent semantic cues (gender + age) strongly agree.

![](images/1ed8d6d15a614ae99fb16438767ad60e7b724a52da500117840144566247a48b.jpg)

![](images/07acd69ee860015ca6acf360951ac9754b0272d795ef89be12717bbfb0f4fad9.jpg)

![](images/51c1d6cbb7e728633bf1e7a6d0d9449da1a1f3a809287c5993cb4fc1858affee.jpg)  
Figure 2: Detailed mechanisms of VISTA. (a) Three Zone Semantic Triage with GMM dynamic thresholding. (b) SSR maps raw predictions to stable tokens. (c) Semantic Arbitration via temporal voting compares symbolic profiles.

Table 1: Face clustering performance. Top: controlled comparison, VISTA vs. COFC under identical detector (RetinaFace), embeddings (FaceNet), and protocol (WCP / NMI / ID\_Acc %). Bottom: literature reference numbers with SENet-50 features<sup>†</sup> (WCP / NMI only); feature/protocol diferences make these reference points, not controlled comparisons. BCL (AT)/(OC) denote the VPCD protocols with unknown/oracle cluster counts as reported by Brown et al. [3]; BCL is supervised and not streaming. Bold: best in controlled comparison.
<table><tr><td>Method</td><td>Online?</td><td>Unsup.?</td><td>Core Mechanism</td><td>TBBT</td><td>Buffy</td><td>Sherlock</td></tr><tr><td colspan="7">Controlled comparison: online, unsupervised, identical features and protocol (WCP / NMI / ID_Acc)</td></tr><tr><td>COFC [16]</td><td>√</td><td>√</td><td>Constrained Incremental</td><td>92.13 / 86.97  / 89.74</td><td>64.60 / 66.28 / 64.31</td><td>76.22 / 50.56 / 71.27</td></tr><tr><td>VISTA (Full Framework)</td><td>√</td><td>√</td><td>Spatio Temporal Semantic Triage</td><td>96.97 / 91.25 / 93.94</td><td>87.39 / 79.30 / 82.82</td><td>80.45 / 51.19 / 68.10</td></tr><tr><td colspan="7">Literature reference: SENet-50 features†, reported numbers (WCP / NMI)</td></tr><tr><td>C1C† [12]</td><td>x</td><td>√</td><td>Constrained 1NN</td><td>89.20 / 87.40</td><td>66.30 / 68.80</td><td>76.30 / 50.30</td></tr><tr><td>BCL† (AT) [3, 33]</td><td>x</td><td>x</td><td>Ball Cluster Learning</td><td>90.80 / 85.70</td><td>85.00 / 78.80</td><td>- / -</td></tr><tr><td>BCL† (OC) [3, 33]</td><td>x</td><td>x</td><td>Ball Cluster Learning</td><td>94.00 / 85.00</td><td>86.50 / 77.60</td><td>- / -</td></tr><tr><td>MuHPC† (Face) [3]</td><td>x</td><td>x</td><td>Global Hierarchical 1NN</td><td>99.40 / 97.80</td><td>96.10 / 92.80</td><td>86.70 / 60.30</td></tr></table>

Rather than relying exclusively on static boundaries, VISTA maintains a sliding window queue Q of recent similarity scores and fits a bimodal Gaussian Mixture Model (GMM) online to capture the distribution of same identity and diferent identity pairs. During the initial warm up phase (i.e., before Q is fully populated), the system relies on static initial thresholds $\tau _ { \mathrm { h i g h } } ^ { ( 0 ) }$ and $\tau _ { \mathrm { l o w } } ^ { ( 0 ) }$ . The sensitivity of these initial parameters is extensively evaluated in the Supplementary Material. The GMM is formalized as:

$$
p ( x ) = \pi _ { 1 } N ( x \mid \mu _ { 1 } , \sigma _ { 1 } ^ { 2 } ) + \pi _ { 2 } N ( x \mid \mu _ { 2 } , \sigma _ { 2 } ^ { 2 } ) ,\tag{2}
$$

where the lower component $( \mu _ { 1 } < \mu _ { 2 } )$ models diferent identity similarities and the upper component models same identity similarities. We dynamically estimate the threshold proposals as

$$
\tau _ { \mathrm { l o w } } ^ { \star } = \mathrm { c l i p } ( \mu _ { 1 } + \sigma _ { 1 } , 0 . 1 5 , 0 . 5 0 ) , \qquad \tau _ { \mathrm { h i g h } } ^ { \star } = \mu _ { 2 } - 0 . 5 \sigma _ { 2 } .\tag{3}
$$

The 0.50 cap on $\tau _ { \mathrm { l o w } } ^ { \star }$ is a deliberate design choice: it preserves a suficiently wide ambiguity zone for semantic arbitration in visually compressed windows. A cap sweep over all 15 episodes (Supplementary Material) shows that relaxing it to 0.70 or 1.00 leaves all metrics unchanged after rounding, i.e., it is a stability safeguard rather than a tuned knob. To ensure temporal stability, we apply an exponential moving average update [34]:

$$
\tau _ { \mathrm { n e w } } = \eta _ { \mathrm { g m m } } \tau ^ { \star } + \big ( 1 - \eta _ { \mathrm { g m m } } \big ) \tau _ { \mathrm { o l d } } ,\tag{4}
$$

where $\eta _ { \mathrm { g m m } } = 0 . 3$ is the GMM threshold smoothing coeficient, distinct from the cluster representation momentum ema\_rep (Section 4.2). This mechanism allows VISTA to adapt to evolving visual distributions across diferent scenes while preventing abrupt threshold oscillations. An empirical visualization confirming the bimodal structure of the similarity distributions across all 15 episodes is provided in the Supplementary Material.

Spatio temporal decoupled arbitration. For pairs in the ambiguity zone $( \tau _ { \mathrm { l o w } } ~ \leq ~ \mathrm { s i m } ~ \leq ~ \tau _ { \mathrm { h i g h } } )$ , VISTA compares their SSR profiles to compute an adjustment score adj. We categorize attributes into persistent traits $\mathcal { P } ~ = ~ \{ \mathrm { g e n d e r } , \mathrm { a g e } \}$ and transient traits T = {clothing, hair}. We define a joint confidence weight � = min $( p _ { T } ( s ) , p _ { C } ( s ) )$ ). The adjustment combines reward and penalty terms:

$$
\mathbf { a d j } = \underbrace { \sum _ { r } \boldsymbol { \lambda } _ { r } \cdot \boldsymbol { w } _ { r } } _ { \mathrm { r e w a r d s } } - \underbrace { \boldsymbol { \gamma } \cdot \sum _ { p } \boldsymbol { \lambda } _ { p } \cdot \boldsymbol { w } _ { \hat { p } } } _ { \mathrm { p e n a l t i e s } } ,\tag{5}
$$

where $\gamma$ is a global penalty scaling factor. Reward terms include gender+age agreement $( \lambda _ { \mathrm { g a } } )$ and clothing consistency $( \lambda _ { \mathrm { c l o t h } } )$ , while penalties include gender mismatch $( \lambda _ { \mathrm { g m } } ) _ { \mathrm { : } }$ , age mismatch $\left( \lambda _ { \mathrm { a m } } \right)$ , and hair conflict $( \lambda _ { \mathrm { h c } } )$ . The final updated similarity is sim<sup>′</sup> = sim + adj. A merge is accepted if sim<sup>′</sup> $\geq \tau _ { \mathrm { m e r g e } }$ . We set $\tau _ { \mathrm { m e r g e } } = 0 . 3 2 5 { \mathrm { : } }$ it is the final acceptance gate for ambiguity zone pairs after arbitration, not a standalone visual threshold; a pair must be the best current assignment and its post arbitration score, after possible penalties, must pass the gate, whereas high confidence pairs are routed directly by $\tau _ { \mathrm { h i g h } }$ . A sweep over {0.280–0.475} (Supplemen tary Material) confirms the expected purity–continuity trade of. To prevent over segmentation, VISTA decouples the spatial and temporal dimensions as follows:

Low Confidence Semantic Masking. When the mean face detection confidence $\bar { c } _ { T }$ of a track drops below a quality threshold $\theta _ { \mathrm { c l i f f } }$ (indicating blur or darkness), VISTA continuously attenuates all semantic penalties via a linear scaling factor:

$$
\begin{array} { r } { \gamma _ { \mathrm { e f f } } = \mathrm { m i n } \Bigl ( 1 . 0 , ~ \mathrm { m a x } \bigl ( 0 . 1 , ~ \frac { \bar { c } _ { T } - \theta _ { \mathrm { f l o o r } } } { \theta _ { \mathrm { c l i f f } } - \theta _ { \mathrm { f l o o r } } } \bigr ) \Bigr ) , } \end{array}\tag{6}
$$

where $\theta _ { \mathrm { f l o o r } }$ is a lower bound below which attenuation saturates. This soft masking ensures that as visual quality degrades, the influence of potentially hallucinated VLM predictions is progressively suppressed rather than abruptly removed, avoiding discontinuous clustering behaviour at the quality boundary.

Temporal Decay for Transient Attributes. If the frame gap $\Delta f$ between a track and a cluster exceeds a long range threshold $\Delta _ { \mathrm { m a x } } ,$ VISTA masks transient traits $( \mathrm { e . g . }$ , clothing, hair) as unknown, bypassing costume driven conflicts:

$$
\begin{array} { r } { w _ { d } ^ { \mathrm { e f f } } = \left\{ \begin{array} { l l } { w _ { d } } & { \mathrm { i f } \ d \in \mathcal { P } \ \mathrm { o r } \Delta f \leq \Delta _ { \mathrm { m a x } } , } \\ { 0 } & { \mathrm { i f } \ d \in \mathcal { T } \ \mathrm { a n d } \Delta f > \Delta _ { \mathrm { m a x } } . } \end{array} \right. } \end{array}\tag{7}
$$

This mechanism ensures that VISTA relies more on persistent traits for long range temporal association, while still leveraging transient cues for temporally adjacent merges where costume stability can be assumed.

Perplexity based confidence weighting. To further modulate the influence of semantic arbitration, VISTA maintains a short history window of � recent per cluster entropy observations. When the perplexity ofa cluster’s embedding distribution exceeds a threshold, indicating high intra cluster visual diversity, the semantic adjustment magnitude is down weighted by a factor

$$
\omega = \operatorname* { m a x } ( \omega _ { \mathrm { m i n } } , \ e ^ { - \beta \cdot \mathrm { p e r p } } ) ,\tag{8}
$$

where $\beta$ controls the decay rate. This prevents aggressive semantic penalties for clusters that already exhibit high visual variance.

## 4 Experiments

## 4.1 Datasets and Evaluation Metrics

We evaluate on the VPCD benchmarks: TBBT (6 episodes, well lit sitcom), Sherlock (3 episodes, thriller with costume changes), and Bufy (6 episodes, dark action, which feature rich narrative and character arcs [14]). We report Weighted Cluster Purity (WCP), Normalized Mutual Information (NMI), and Identification Accuracy (ID\_Acc). WCP focuses on cluster homogeneity, while ID\_Acc penalizes trajectory fragmentation. VISTA seeks a favorable balance across these metrics.

## 4.2 Implementation Details

VISTA is implemented in Python using PyTorch. As described in Section 3.2, we utilize RetinaFace [8] for face detection and FaceNet [25] for identity embedding. CLIP ViT-B/32 [22] performs zero shot attribute classification with text prompts for clothing and hair, and DeepFace [26] is used for age analysis. The complete CLIP text prompt templates, label vocabularies, and the full Symbol Registry mappings are provided in the Supplementary Material.

For intra shot tracking, we use $\theta _ { \mathrm { i o u } } = 0 . 3$ and $\theta _ { \mathrm { f e a t } } = 1 . 0$ . In the final full framework evaluation, we use a single unified configuration across all three datasets to avoid dataset specific tuning and demonstrate robust generalization: RetinaFace for detection, FaceNet for embedding, coarse age grouping age symbols, $\tau _ { \mathrm { h i g h } } ^ { ( 0 ) } = 0 . 8 5 , \tau _ { \mathrm { l o w } } ^ { ( 0 ) } = 0 . 2 5$ , cluster representation smoothing momentum ema\_ $\mathsf { r e p } = 0 . 9 5$ , GMM threshold smoothing $\eta _ { \mathrm { g m m } } = 0 . 3$ arbitration gate $\tau _ { \mathrm { m e r g e } } = 0 . 3 2 5 , \mathrm { G M M }$ queue size $| Q | = 1 5 0 $ , perplexity weighting enabled, and fixed length scene segmentation with length 24. Dynamic thresholding and semantic arbitration are enabled. Perplexity weighting uses a history window of $H = 2 0$ observations with $\omega _ { \mathrm { m i n } } = 0 . 1 0$ and $\beta = 1 . 2 .$ . For semantic arbitration $( \operatorname { E q . 5 } ) _ { \mathrm { : } }$ , the adjustment weights are empirically set as $\lambda _ { g a } = 0 . 0 8 ,$ $\lambda _ { c l o t h } = 0 . 0 4 , \lambda _ { g m } = 0 . 1 2 , \lambda _ { a m } = 0 . 0 2 ,$ and $\lambda _ { h c } = 0 . 0 4 . \mathrm { A }$ grid search over 243 � configurations (detailed in the Supplementary Material) shows that VISTA’s performance is stable across a wide range of settings: WCP standard deviation is only 0.31% on TBBT, 1.09% on Bufy, and 0.78% on Sherlock, confirming that the method is not driven by fragile hyperparameter tuning.

Table 2: Ablation study of semantic components (WCP / NMI / ID\_Acc %).
<table><tr><td>Configuration</td><td>TBBT</td><td> $B u f f y$ </td><td>Sherlock</td></tr><tr><td>VISTA (Full Framework)</td><td>96.97 / 91.25 / 93.94</td><td>87.39 / 79.30 / 82.82</td><td>80.45 / 51.19 / 68.10</td></tr><tr><td>w/o Semantic Arbitration</td><td>96.71 / 89.97 / 91.27</td><td>80.54 / 77.62 / 78.35</td><td>79.06 / 52.55 / 69.56</td></tr><tr><td>w/o SSR</td><td>94.83 / 89.29 / 91.60</td><td>74.75 / 74.47 / 73.31</td><td>78.36 / 52.62 / 71.43</td></tr><tr><td>w/o Three Zone Gating</td><td>96.78 / 90.80 / 92.32</td><td>80.17 / 77.03 / 76.58</td><td>80.27 / 51.49 / 64.63</td></tr><tr><td>w/o Temporal Smoothing</td><td> $9 6 . 2 8 \ : / \ : 9 0 . 0 9 \ : / \ : 9 1 . 9 5$ </td><td>82.07 / 77.44 / 77.05</td><td>84.11 / 52.54 / 62.23</td></tr></table>

To study robustness, we additionally perform a parameter sweep over $\tau _ { \mathrm { h i g h } } ^ { ( 0 ) } , \tau _ { \mathrm { l o w } } ^ { ( 0 ) } ,$ , age symbol configuration, ema\_rep momentum, and GMM queue size. The detailed dataset specific parameter efect tables are reported in the Supplementary Material and should be interpreted as marginal robustness studies rather than the source of the final unified full framework setting; key trends there: coarse age grouping and strong ema\_rep smoothing yield consistently best results, and darker scenes $( B u f f y )$ favor larger GMM queues for stable threshold estimation. In addition, ${ \mathrm { ~ a ~ } } 3 \times 3$ grid bracketing the warm up initialisation over $\tau _ { \mathrm { h i g h } } ^ { ( 0 ) } ~ \in ~ \{ 0 . 7 5 , \stackrel { - } { 0 . 8 2 5 } , 0 . 9 0 \}$ and $\tau _ { \mathrm { l o w } } ^ { ( 0 ) } \in \left\{ 0 . 2 0 , 0 . 2 7 5 , 0 . 3 5 \right\}$ (135 episode level evaluations, all else fixed) yields max−min variation of at most 0.69 pp across all metrics and datasets (Supplementary Material); $\tau _ { \mathrm { h i g h } }$ sensitivity is limited precisely because GMM dynamic thresholding overrides the static initialisation after warm up.

All experiments are conducted on a single NVIDIA RTX A6000 GPU. A detailed breakdown of the computational eficiency and throughput of our pipeline is discussed in Section 4.6.

## 4.3 Quantitative Results

Table 1 reports a controlled comparison between VISTA and COFC under identical detector, embeddings, and protocol, together with literature reported results as reference points. We note that strictly online unsupervised face clustering methods are rare; COFC [16] remains the only published method in this category with results on VPCD. We additionally include ofline unsupervised (C1C [12]), supervised (BCL [33]), and multi-modal (MuHPC [3]) methods as reference points. Methods marked with † use SENet-50 features, which are generally more discriminative than the FaceNet embeddings used by VISTA and COFC [7].

VISTA outperforms COFC across all three datasets under identical features and evaluation. On TBBT, WCP and NMI improve from 92.13%/86.97% to 96.97%/91.25%, and ID\_Acc from 89.74% to

93.94%. On Bufy, VISTA raises WCP and NMI from 64.60%/66.28% to 87.39%/79.30%, with ID\_Acc improving from 64.31% to 82.82% (+18.51 pp), also far surpassing the ofline C1C (66.30%/68.80%) that uses stronger features. As literature reference points, VISTA’s TBBT results (96.97%/91.25%) are also higher than the reported supervised BCL (AT) numbers (90.80%/85.70%), and its Bufy NMI is comparable (79.30% vs. 78.80%), although feature and protocol diferences preclude a strict comparison.

On Sherlock, VISTA improves over both COFC (76.22%/50.56%) and C1C (76.30%/50.30%), reaching 80.45%/51.19%. However, ID\_Acc decreases from 71.27% (COFC) to 68.10% (−3.17 pp): under Sherlock’s heavy disguises and long gap appearance variation, VISTA prioritises cluster purity over identity continuity, yielding cleaner clusters at the cost of additional fragmentation. We regard this as a limitation rather than a universal all metric gain; Sections 4.4 and 4.5 analyse this purity–continuity trade of in detail. Although the gains are more modest than on TBBT and Bufy, this confirms that selective semantic integration provides consistent WCP/NMI improvements even on the most volatile dataset, with room for future refinement.

Additional online baselines. To isolate the source of the gains, we evaluate two further online baselines under the same features and protocol (full tables in the Supplementary Material). (i) Naive VLM hard constraints: raw attributes applied as must link/cannot link rules (no SSR, no triage) over fragment severely, producing 4.0– 11.3× more clusters than ground truth; the inflated cluster count artificially raises WCP while ID\_Acc collapses to 63.95/60.64/40.65 on TBBT/Bufy/Sherlock, far below VISTA Full and below our weakest geometry driven ablation (Table 2). This confirms that VISTA’s improvements stem from selective semantic triage, not from VLM attributes alone. (ii) Sequential online HAC attains very high WCP (99.73/97.62/91.19) under severe fragmentation (#Pred/#GT = 4.5/ 9.5/39.8×) but substantially lower NMI/ID\_Acc (84.93/81.29, 76.78/ 63.49, 49.28/40.56), illustrating the over segmentation trap that three zone triage is designed to avoid.

## 4.4 Ablation Studies

We ablate key semantic components in Table 2. Overall, the results confirm that the semantic modules afect the final precision recall trade of, with the clearest gains appearing on TBBT and Bufy.

Semantic Components. As shown in Table 2, disabling Semantic Arbitration or SSR degrades performance (especially on TBBT), confirming that geometric similarity alone struggles with appearance variations and that SSR is vital for stabilizing noisy predictions. Similarly, removing Three Zone Gating or Temporal Smoothing weakens the precision recall trade of, leading to brittle merges. Interestingly, on the volatile Sherlock dataset, some ablated variants achieve higher individual metric scores. In particular, removing Temporal Smoothing raises WCP to 84.11% (vs. 80.45%). We attribute this to Sherlock’s frequent costume changes and disguises: temporal smoothing carries stale transient attribute profiles into subsequent decisions, occasionally penalising valid same identity merges. Conversely, removing SSR yields a higher NMI (52.62%) and ID\_Acc (71.43%) but at the cost of lower WCP (78.36%), reflecting the fundamental purity completeness trade of. This dataset specific tension motivates future work on adaptive smoothing decay rates.

![](images/82b3c80022abb1fd670c28f4416f80d0b2ec95a50314681dcb6e5153d9f34318.jpg)  
Figure 3: Timeline level qualitative comparison on Sherlock. Top: ground truth; Middle: COFC; Bottom: VISTA. Each row represents a character, with colors indicating predicted clusters. Compared to COFC’s cross identity confusion, VISTA maintain cleaner global identity organization and better separates visually similar characters (e.g., Lestrade and Mycroft).

Table 3: Throughput analysis of the VISTA pipeline
<table><tr><td>Dataset</td><td>Frames</td><td>Total Time (s)</td><td>End to End FPS</td><td>Clustering FPS</td></tr><tr><td>TBBT (s01e01)</td><td>32,990</td><td>4,655</td><td>7.09</td><td>2122.45</td></tr><tr><td>Buffy (s05e01)</td><td>62,620</td><td>7,978</td><td>7.85</td><td>2288.42</td></tr><tr><td>Sherlock (e01)</td><td>132,150</td><td>16,393</td><td>8.06</td><td>1842.30</td></tr></table>

Table 4: Semantic arbitration invocation rate. CLIP/DeepFace attributes are extracted up front for every crop (Stage 1); this table measures how often the arbitration step fires during association.
<table><tr><td rowspan="2">Dataset</td><td colspan="2">Frame level</td><td colspan="2">Tracklet level</td></tr><tr><td>Arb. / Total</td><td>Rate</td><td>Arb. / Total</td><td>Rate</td></tr><tr><td>TBBT</td><td>29,389 / 319,646</td><td>9.19%</td><td>3,685 / 20,737</td><td>17.77%</td></tr><tr><td>Buffy</td><td>64,222 / 474,210</td><td>13.54%</td><td>8,198 / 35,079</td><td>23.37%</td></tr><tr><td>Sherlock</td><td>47,919 / 421,899</td><td>11.36%</td><td>5,797 / 30,448</td><td>19.04%</td></tr><tr><td>Overall</td><td>141,530 / 1,215,755</td><td>11.64%</td><td>17,680 / 86,264</td><td>20.50%</td></tr></table>

Efect of Tracklet Length. We ablate intra shot tracklet length averaged over all 15 episodes. Increasing the length from 12 to 48 improves ID\_Acc (77.88% → 81.05%) and NMI (77.39% → 78.72%), but extending to 96 slightly degrades both metrics due to accumulated appearance noise. WCP peaks at length 24 (92.83%). We adopt $L _ { \operatorname* { m a x } } = 2 4$ as the unified default, balancing purity and the early decision requirement of online clustering. The full table is in the Supplementary Material.

## 4.5 Qualitative Analysis

Figure 3 and the crop level comparison in the Supplementary Material provide qualitative comparisons on Sherlock, the most challenging dataset. These visualizations reveal that the main failure mode of classical online clustering is not uniform: some characters primarily sufer from cross identity confusion, while others are more afected by trajectory fragmentation.

Figure 3 shows that COFC’s identity errors extend beyond local boundary noise: predicted clusters intrude into other characters’ timelines for extended intervals. A representative case is COFC’s Lestrade timeline, which contains repeated intrusions from a Mycroft dominant cluster, indicating substantial cross identity confusion rather than mere short term fragmentation. VISTA preserves cleaner separation between these visually similar but semantically distinct characters, suggesting that semantic triage prevents overly aggressive merging.

The crop level comparison (Supplementary Material) complements this: under challenging pose, illumination, and blur conditions, COFC more frequently assigns samples to competing clusters, whereas VISTA keeps a larger portion within identity consistent clusters. Importantly, VISTA’s remaining errors on Sherlock stem from overly conservative semantic decisions that occasionally fragment the same identity into multiple segments, rather than cross identity confusion. This is precisely the mechanism behind the ID\_Acc decrease relative to COFC on Sherlock (Table 1): purity is defended at the cost of identity continuity. The qualitative gain is thus better understood as a shift in error profile, from large scale identity mixing toward more structured, localized fragmentation, consistent with the quantitative trends in Table 2. This also explains why removing Temporal Smoothing can raise WCP on Sherlock: without stale attribute profiles, fewer valid merges are rejected, at the cost of reduced cross identity discrimination.

## 4.6 Computational Eficiency and Throughput

A critical requirement for online clustering is computational efficiency. To demonstrate that VISTA’s semantic triage and spatio temporal arbitration do not introduce prohibitive overhead, we profile the throughput of our pipeline. As shown in Table 3, the performance exhibits a clear dichotomy between the heavy feature extraction front end and the eficient clustering back end.

The end to end throughput (Total FPS) stabilizes at approximately 7 to 8 frames per second. Profiling reveals that over 99% of this time is consumed by Stage 1 (Feature Extraction), which involves decoding high resolution video frames and executing dense face detection alongside multi-modal feature encoding (FaceNet, CLIP, and DeepFace attribute extraction).

In contrast, VISTA’s online clustering module (Stages 2 and 3) operates at 1800 to 2300 FPS. This highlights a major advantage of our design: by abstracting raw VLM outputs into compact SSR and employing dynamic GMM thresholding, VISTA restricts semantic arbitration to ambiguous boundary cases. Consequently, the proposed vision language integration achieves robust performance gains while keeping the online clustering phase itself eficient; in our implementation, the dominant runtime bottleneck remains face detection and feature extraction and not the clustering.

The Spatio Temporal Semantic Triage acts as an efective decision filter (Table 4). Across 1.2M+ frames, semantic arbitration is invoked for only 20.50% of tracklets (covering 11.64% of total frames). TBBT requires the least intervention (9.19% of frames) due to cleaner visual conditions, while Bufy’s darker scenes trigger more ambiguity (13.54%). Since CLIP/DeepFace attributes are extracted up front for every crop in Stage 1 (always included in the end to end cost), Table 4 measures how often arbitration fires during association, not how many VLM forward passes are avoided; triage keeps the clustering stage lightweight by restricting arbitration to ambiguous boundary cases.

## 4.7 Analysis of Demographic Disparities

![](images/55d945be8092b73fd244b4c51d0ba4eeefb36bdb368c40c101c01d1438158175.jpg)  
(a) By Gender

![](images/ce47bc66b64495838d041c4f9f68d32d320b4c95e72cefc55f11fd730356d488.jpg)  
(b) By Age Group  
Figure 4: Distribution of VLM intervention and rejection rates across diferent demographic groups.

Although VISTA reduces global identity confusion, VLM arbitration is not invoked uniformly across predicted demographic groups. We therefore report both triage rate diagnostics (Figure 4) and subgroup clustering-quality gaps in the Supplementary Material.

Female characters show higher intervention and ambiguity-zone rejection rates than male characters (13.48% vs. 10.76%, and 2.38% vs. 1.62%), suggesting that more disambiguation burden is ofloaded to semantic arbitration in unconstrained settings. This observation is consistent with known gender accuracy disparities in commercial vision models [4], although our analysis is diagnostic rather than causal.

Age Disparities: Figure 4b reveals distinct efects at age extremes. Children (0–9 years) trigger the highest VLM intervention rate (24.03%), while older adults (50–59 years) face the highest rejection rate (5.93%). Conversely, the core 30–39 demographic, which is common in face model training distributions [13], requires the least intervention (8.55%). These trends suggest that skewed training data and dificult capture conditions can jointly afect representation robustness for age extremes [32].

Subgroup Quality and Safeguards: Beyond triage rates, the Supplementary Material reports WCP/NMI/ID\_Acc gaps on the IoU matched subset using the same VLM predicted gender/age groups. Because VPCD lacks ground truth demographics, these groups do not constitute a full fairness audit; nevertheless, the gap signs vary across datasets rather than showing one directional degradation. As an ethical safeguard, VISTA can disable gender/age cues and fall back to geometry plus transient attributes.

Discussion: These preliminary findings show clear algorithmic disparities, but demographic variables in videos often entangle with environmental factors. Screen industry tropes, visual ageism [5, 6, 21], and historical underrepresentation [2, 30] often dictate lighting, camera angles, and action sequences for specific groups. For instance, low illumination scenes common in action genres can disproportionately degrade geometric feature extraction for darker skin tones, compounding existing gender and age ambiguities. Thus, attributing these disparities solely to intrinsic model bias requires further disentanglement. Future work will systematically analyze these issues by cross referencing scene level factors (illumination, pose, focal presence) with face model training distributions, while extending to the intersectional impacts of digital racialization [1, 27]. Addressing these entangled biases is essential for deploying equitable face clustering systems in real world, unconstrained video streaming applications.

## 5 Conclusion

We presented VISTA, a vision language semantic triage architecture for fully online face clustering in long videos. By combining Symbolic Semantic Representation, dynamic GMM thresholding, and spatio temporal decoupled arbitration, VISTA mitigates over segmentation and improves the purity–continuity trade of on TBBT, Bufy, and Sherlock. Under identical detector, embeddings, and evaluation protocol, VISTA consistently outperforms the strongest matched online unsupervised baseline while preserving streaming compatible eficiency. Limitations and Future Work: VISTA still relies on fixed symbolic vocabularies, a bimodal GMM prior, and approximate confidence attenuation; cross domain validation beyond scripted TV remains open. On Sherlock, VISTA trades identity continuity for purity (ID\_Acc −3.17 pp vs. COFC), motivating adaptive smoothing, calibrated VLM uncertainty, and confounder aware fairness analysis.

## Acknowledgments

This work was funded by the Leverhulme Trust (RPG-2023-091).

## References

[1] Ramon Amaro. 2022. The Black Technical Object: On Machine Learning and the Aspiration ofBlack Being. Steinberg Press, London.

[2] David Bamman, Rachel Samberg, Richard Jean So, and Nianlong Zhou. 2024. Measuring diversity in Hollywood through the large-scale computational analysis of film. Proceedings ofthe National Academy ofSciences 121, 46 (2024), e2409770121.

[3] Andrew Brown, Vicky Kalogeiton, and Andrew Zisserman. 2021. Face, Body, Voice: Video Person-Clustering with Multiple Modalities. In Proc. IEEE/CVF Int. Conf. on ComputerVision (ICCV) Workshops. 3178–3187. doi:10.1109/ICCVW54120. 2021.00313

[4] Joy Buolamwini and Timnit Gebru. 2018. Gender shades: Intersectional accuracy disparities in commercial gender classification. In Conference on fairness, accountability and transparency. PMLR, 77–91.

[5] Toni Calasanti, Kathleen F. Slevin, and Neal King. 2006. Ageism and feminism: From “et cetera” to center. NWSA journal 18, 1 (2006), 13–30.

[6] Kate Crawford and Trevor Paglen. 2019. Excavating AI: The politics of images in machine learning training sets. The AI Now Institute, New York University (2019).

[7] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. 2019. ArcFace: Additive Angular Margin Loss for Deep Face Recognition. In Proc. IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR). 4690–4699.

[8] Jiankang Deng, Jia Guo, Yuxiang Zhou, Jinke Yu, Irene Kotsia, and Stefanos Zafeiriou. 2020. RetinaFace: Single-Stage Dense Face Localisation in the Wild. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

[9] Matthias Grundmann, Vivek Kwatra, Mei Han, and Irfan Essa. 2010. Eficient hi erarchical graph-based video segmentation. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 1374–1381.

[10] Jia Guo, Jiankang Deng, Alexandros Lattas, and Stefanos Zafeiriou. 2021. In sightFace: 2D and 3D Face Analysis Project. https://github.com/deepinsight/ insightface. GitHub repository.

[11] Monica Laura Haurilet, Makarand Tapaswi, Ziad Al-Halah, and Rainer Stiefelhagen. 2016. Naming TV Characters by Watching and Analyzing Dialogs. In Proc. IEEE Winter Conf. on Applications ofComputer Vision (WACV). 1–9.

[12] Vicky Kalogeiton and Andrew Zisserman. 2020. Constrained Video Face Clustering Using 1NN Relations. In Proc. British Machine Vision Conference (BMVC).

[13] Kimmo Kärkkäinen and Jungseock Joo. 2021. FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age for Bias Measurement and Mitigation. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). 1548–1558.

[14] Roz Kaveney. 2004. Reading the Vampire Slayer: The New, Updated, Unoficial Guide to Bufy and Angel. Tauris Parke Paperbacks.

[15] Atif Khurshid, Bostan Khan, Muhammad Shahzad, and Muhammad Moazam Fraz. 2025. TVFace: towards large-scale unsupervised face recognition in video streams. Pattern Analysis and Applications 28, 2 (2025), 88.

[16] Prakhar Kulshreshtha and Tanaya Guha. 2018. An Online Algorithm for Constrained Face Clustering in Videos. In Proc. IEEE Int. Conf. on Image Processing (ICIP). 1889–1893. doi:10.1109/ICIP.2018.8451658

[17] Prakhar Kulshreshtha and Tanaya Guha. 2020. Dynamic character graph via online face clustering for movie analysis. Multimedia Tools and Applications 79 (2020), 33103–33118. doi:10.1007/s11042-020-09449-6

[18] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023. BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. In Proc. Int. Conf. on Machine Learning (ICML). 19730–19742.

[19] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. 2023. Evaluating object hallucination in large vision-language models. In

Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP). 292–305.

[20] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2024. Visual Instruction Tuning. In Proc. Advances in Neural Information Processing Systems (NeurIPS).

[21] Anna Maria Ozimek. 2020. Equality, Diversity and Inclusion in the Screen Industries. Technical Report. Screen Industries Growth Network, University of York

[22] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning Transferable Visual Models From Natural Language Supervision. In Proc. Int. Conf. on Machine Learning (ICML). 8748–8763.

[23] Douglas A Reynolds. 2009. Gaussian mixture models. In Encyclopedia of Biometrics. Springer, 707–709.

[24] Saquib Sarfraz, Vivek Sharma, and Rainer Stiefelhagen. 2019. Eficient Parameter-Free Clustering Using First Neighbor Relations. In Proc. IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR). 8934–8943.

[25] Florian Schrof, Dmitry Kalenichenko, andJames Philbin. 2015. FaceNet: A Unified Embedding for Face Recognition and Clustering. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR). 815–823. doi:10.1109/CVPR.2015.7298682

[26] Sefik Ilkin Serengil and Alper Ozpinar. 2024. A Benchmark of Facial Recognition Pipelines and Co-Usability Performances of Modules. In Journal ofInformation Technologies, Vol. 17. 95–107.

[27] Sanjay Sharma. 2023. Understanding Digital Racism: Networks, Algorithms, Scale. Rowman & Littlefield, London.

[28] Vivek Sharma, Makarand Tapaswi, M. Saquib Sarfraz, and Rainer Stiefelhagen. 2019. Video Person Clustering with Self-Supervised Representation Learning. In IEEE Trans. on Multimedia, Vol. 22. 696–708.

[29] Shuai Shen, Wanhua Li, Xiaobing Wang, Dafeng Zhang, Zhezhu Jin, Jie Zhou, and Jiwen Lu. 2023. CLIP-Cluster: CLIP-Guided Attribute Hallucination for Face Clustering. In Proc. IEEE/CVF Int. Conf. on Computer Vision (ICCV). 18935–18945. doi:10.1109/ICCV48922.2023.01830

[30] Stacy L Smith, Marc Choueiti, Katherine Pieper, Kevin Yao, Ariana Case, and Angel Choi. 2020. Inequality in 1,300 Popular Films: Examining Portrayals of Gender, Race/Ethnicity, LGBTQ & Disability from 2007 to 2019. Technical Report. Annenberg Inclusion Initiative, University of Southern California.

[31] Krishna Somandepalli, Tanaya Guha, Victor R Martinez, Naveen Kumar, Hartwig Adam, and Shrikanth Narayanan. 2021. Computational media intelligence: Human-centered machine analysis of media. Proc. IEEE 109, 5 (2021), 891–910.

[32] Harini Suresh and John Guttag. 2021. A Framework for Understanding Sources of Harm throughout the Machine Learning Life Cycle. In Proceedings ofthe 1st ACM Conference on Equity and Access in Algorithms, Mechanisms, and Optimization (EAAMO ’21).

[33] Makarand Tapaswi, Marc T. Law, and Sanja Fidler. 2019. Video Face Clustering with Unknown Number of Clusters. In Proc. IEEE/CVF Int. Conf. on Computer Vision (ICCV). 5027–5036.

[34] Antti Tarvainen and Harri Valpola. 2017. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. In Advances in Neural Information Processing Systems (NeurIPS), Vol. 30.

[35] Devesh Walawalkar and Pablo Garrido. 2024. VideoClusterNet: Self-Supervised and Adaptive Face Clustering for Videos. In European Conference on Computer Vision (ECCV).

[36] Zhongdao Wang, Liang Zheng, Yali Li, and Shengjin Wang. 2019. Linkage Based Face Clustering via Graph Convolution Network. In Proc. IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR). 1117–1125.

[37] Chao-Yuan Wu and Philipp Krähenbühl. 2021. Towards long-form video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 1884–1894.

[38] Zhanpeng Zhang, Ping Luo, Chen Change Loy, and Xiaoou Tang. 2016. Joint Face Representation Adaptation and Clustering in Videos. In European Conference on Computer Vision (ECCV).

# Supplementary Material: VISTA: Vision-Language Integrated Semantic Triage Architecture for Online Face Clustering in Long Videos

Jiashu Liao University of Glasgow Glasgow, United Kingdom jiashu.liao@glasgow.ac.uk

Sanjay Sharma The University of Warwick Coventry, United Kingdom sanjay.sharma@warwick.ac.uk

## ACM Reference Format:

Jiashu Liao, Joanne Garde-Hansen, Sanjay Sharma, and Tanaya Guha. 2026. Supplementary Material: VISTA: Vision-Language Integrated Semantic Triage Architecture for Online Face Clustering in Long Videos. In Proceedings ofthe 34th ACM International Conference on Multimedia (MM ’26), November 10–14, 2026, Rio de Janeiro, Brazil. ACM, New York, NY, USA, 8 pages. https://doi.org/10.1145/3767308.3835422

## S1 Supplementary Video Demonstrations

We provide two supplementary video demonstrations, demo\_mycroft .mp4 and demo\_lestrade.mp4, to illustrate the temporal stability of our framework across diferent levels of scene complexity. Both videos have been compressed to meet the file size limit of the submission system, but still clearly show a side-by-side comparison between the COFC baseline and VISTA in Sherlock (Season 1, Episode 1).

Video Layout & Evaluation Metric. Both videos are rendered in a condensed highlight format, with the left panel showing COFC and the right panel showing VISTA. We color the bounding boxes dynamically based on a purity-based metric:

• Cyan Box (Correct): The predicted cluster’s majority identity matches the ground-truth target character, and the cluster purity exceeds our threshold (≥ 60%).

• Red Box (Wrong): The cluster is either misclassified (wrong majority identity) or heavily contaminated (purity < 60%). A contaminated cluster is treated as a failure even if it contains the target character.

Demo 1: Resolving IdentityOver-merging. This video (demo\_mycroft .mp4) focuses on Mycroft in relatively stable conversational scenes.

• Baseline: COFC over-merges Mycroft and Lestrade into a single large cluster. Although the assigned ID appears temporally continuous, it is incorrect, producing a persistent red box throughout.

![](images/dc1eb54c52b8d8cfeb32f69254a74ea89149bea0d5be376621eacf0ae4df1f8e.jpg)

Joanne Garde-Hansen   
University of Leeds   
Leeds, United Kingdom   
j.garde-hansen@leeds.ac.uk   
Tanaya Guha   
University of Glasgow   
Glasgow, United Kingdom   
tanaya.guha@glasgow.ac.uk

• VISTA: By leveraging vision-language semantic arbitration, VISTA distinguishes the two characters and isolates Mycroft into a separate high-purity cluster, as indicated by the consistent cyan box.

Demo 2: Purity vs. Continuity in Dynamic Scenes. This video (demo\_lestrade.mp4) tracks Lestrade through scenes with frequent occlusions, motion blur, and illumination changes.

• Baseline: COFC merges Lestrade, Mycroft, and several background extras into a single contaminated cluster (purity ∼55%), which fails our metric and remains red throughout.

• VISTA: To preserve cluster purity under these conditions, VISTA’s dynamic triage deliberately over-splits the track into a few separate but pure clusters (e.g., C22 and C26, both with purity >60%). This trade-of ensures that downstream ofline aggregation receives clean tracklets rather than contaminated ones. The cyan boxes confirm that each resulting cluster passes our threshold.

## S2 VISTA Algorithm Pseudocode

To provide a clearer understanding of the online clustering process, we detail the step by step execution of VISTA in Algorithm S1. The algorithm processes the video stream in a single pass. For each camera shot, it extracts dense visual features and zero shot VLM attributes. Short term temporal tracking is performed to construct reliable local tracklets. Subsequently, the inter shot association is guided by our dynamic Gaussian Mixture Model (GMM) thresholding and spatio temporal decoupled semantic arbitration. This pipeline ensures that low confidence semantic noise is filtered out while valuable temporal constraints and geometric similarities are preserved. We emphasise that geometric embeddings and CLIP/DeepFace attributes are extracted up front for every face crop at shot ingestion (line 3) and voted into tracklet level SSR profiles; the three zone triage controls only whether semantic arbitration is invoked during association, not whether VLM inference is performed. The final acceptance gate for ambiguity zone pairs is �<sub>merge</sub> = 0.325 (see Section 3.3 of the main paper).

Algorithm S1 VISTA Online Clustering Pipeline   
Require: Video stream V, initial thresholds $\tau _ { \mathrm { h i g h } } ^ { ( 0 ) } , \tau _ { \mathrm { l o w } } ^ { ( 0 ) } , \tau _ { \mathrm { m e r g e } } =$   
0.325   
Ensure: Global identity clusters C   
1: Initialize $C \gets \emptyset ,$ similarity queue $Q  \emptyset$   
2: for each shot � in $_ \mathrm { ~  ~ }$ do   
3: Extract faces, compute geometric embeddings � and VLM   
attributes A (for every face crop)   
4: Form intra shot tracks $\{ T _ { 1 } , \dots , T _ { m } \}$ ; build temporal con  
straint $Q \cdot$ -matrix   
5: for each track $T _ { i }$ do   
6: Compute SSR profile via temporal majority vote   
7: Find best matching cluster: $C ^ { * } = \arg \operatorname* { m a x } _ { C _ { j } } \sin ( v _ { T _ { i } } , v _ { C _ { j } } )$   
s.t. $Q _ { T _ { i } , C _ { j } } = 1$   
8: � ← sim( $v _ { T _ { i } } , v _ { C ^ { * } } )$   
9: if $s > \tau _ { \mathrm { h i g h } }$ then   
10: Apply bonus only semantic adjustment; merge �<sub>�</sub> into   
$C ^ { * }$   
11: else if $\dot { \mathbf { \eta } } _ { s } \geq \tau _ { \mathrm { l o w } }$ then   
12: Compute adjustment adj via Spatio Temporal Decou  
pled Arbitration   
13: if $s + \mathrm { a d j } \geq \tau _ { \mathrm { m e r g e } }$ then   
14: Merge $T _ { i }$ into $\ b { C } ^ { * }$ (Resolved ambiguity)   
15: else   
16: Create new cluster $C _ { \mathrm { n e w } } \gets \{ T _ { i } \} ; C \gets C \cup \{ C _ { \mathrm { n e w } } \}$   
17: end if   
18: else   
19: if gender + age strongly agree (persistent trait rescue)   
then   
20: Merge �<sub>�</sub> into $C ^ { * }$ (Rescued from reject zone)   
21: else   
22: Create new cluster $C _ { \mathrm { n e w } } \gets \{ T _ { i } \} ; C \gets C \cup \{ C _ { \mathrm { n e w } } \}$   
23: end if   
24: end if   
25: Push � to Q; update GMM thresholds $( \tau _ { \mathrm { h i g h } } , \tau _ { \mathrm { l o w } } )$ via EMA   
$( \eta _ { \mathrm { g m m } } = 0 . 3 )$   
26: end for   
27: end for   
28: return C

## S3 Age Symbol Configurations

As discussed in Section 3.3 of the main paper, applying raw VLM textual predictions directly introduces severe frame level noise. To mitigate this, VISTA maps raw age strings to a discrete vocabulary. We tested three diferent age grouping strategies, detailed in Table S1.

The fine\_grained strategy preserves decade level precision for adults, which can occasionally diferentiate close identities but is highly susceptible to VLM oscillation (e.g., predicting “20-29” in one frame and “30-39” in the next). The broad\_adult strategy collapses a massive 40-year span into a single token, which guarantees stability but loses discriminative power. The coarse\_grouping strategy provides the most robust compromise for the unified setting by reducing age boundary oscillation without collapsing the entire adult range into a single token.

Table S1: Age symbol configurations. Raw DeepFace age predictions are mapped to discrete tokens via one of three schemes. fine\_grained preserves decade level distinctions for adults; coarse\_grouping merges the 30s and 40s into a single <age\_adult> token; broad\_adult further collapses the full 20-59 range.
<table><tr><td>Raw age range</td><td>fine_grained</td><td>coarse_grouping</td><td>broad_adult</td></tr><tr><td>0-9</td><td>&lt;age_child&gt;</td><td>&lt;age_child&gt;</td><td>&lt;age_child&gt;</td></tr><tr><td>10-19</td><td>&lt;age_teen&gt;</td><td>&lt;age_teen&gt;</td><td>&lt;age_teen&gt;</td></tr><tr><td>20-29</td><td>&lt;age_20s&gt;</td><td>&lt;age_young_adult&gt;</td><td>&lt;age_adult&gt;</td></tr><tr><td>30-39</td><td>&lt;age_30s&gt;</td><td>&lt;age_adult&gt;</td><td>&lt;age_adult&gt;</td></tr><tr><td>40-49</td><td>&lt;age_40s&gt;</td><td>&lt;age_adult&gt;</td><td>&lt;age_adult&gt;</td></tr><tr><td>50-59</td><td>&lt;age_50s&gt;</td><td>&lt;age_senior&gt;</td><td>&lt;age_adult&gt;</td></tr><tr><td>60+</td><td>&lt;age_senior&gt;</td><td>&lt;age_senior&gt;</td><td>&lt;age_senior&gt;</td></tr><tr><td>Vocabulary size</td><td>7</td><td>5</td><td>4</td></tr></table>

## S4 CLIP Prompt Templates and Symbol Registry

VISTA uses CLIP ViT-B/32 for zero shot attribute classification. For each detected face crop, logits are computed against all prompt candidates simultaneously in a single forward pass. Table S2 lists the complete prompt templates and label vocabularies for each attribute dimension. DeepFace is used separately for age estimation (not via CLIP prompts) and its raw integer predictions are mapped through the age symbol configurations in Table S1.

Table S2: CLIP zero shot classification prompts and label vocabularies.
<table><tr><td>Attribute</td><td>Prompt Template</td><td>Labels</td></tr><tr><td>Gender</td><td>&quot;A face of a {label}&quot;</td><td>Male, Female</td></tr><tr><td>Clothing</td><td>&quot;A photo of a person wearing a {label}</td><td>T-shirt, Jacket, Shirt, Sweater, Dress, Coat, Vest</td></tr><tr><td>Hair</td><td>“A close-up photo of a person with {label}&quot;</td><td>Curly Hair, Straight Hair, Bald, Short Hair, Long Hair</td></tr></table>

Table S3 provides the complete Symbol Registry mappings that convert raw CLIP predictions into stable symbolic tokens (SSR). Gender and hair labels are mapped one to one, while clothing labels undergo a many to one compression (e.g., both “T-shirt” and “Shirt” map to distinct tokens, but the vocabulary is reduced from open ended text to a fixed set of 7 symbols). Age mappings are handled separately via the configurations in Table S1.

For age estimation, VISTA uses DeepFace [1] with detector\_backend=’skip’ (since face crops are already provided by RetinaFace) and actions=[’age’]. The raw integer age prediction is binned into decade ranges (e.g., 25 → “20-29”) and then mapped through the Symbol Registry age configuration (Table S1). In the unified evaluation, we use the coarse\_grouping preset, yielding a 5-symbol age vocabulary.

## S5 Dataset Specific Parameter Efects

While the main paper reports VISTA’s performance using a unified parameter setting for fair benchmarking, we present a marginal sensitivity analysis here to demonstrate the framework’s robustness across diferent visual environments. Tables S4, S5, and S6 display the marginal efects of individual hyperparameters on TBBT, Bufy, and Sherlock, respectively.

Table S3: Symbol Registry: raw label to symbolic token mappings.
<table><tr><td>Attribute</td><td>Raw Label</td><td>Symbolic Token</td></tr><tr><td>Gender</td><td>Male Female</td><td>&lt;gender_male&gt; &lt;gender_female&gt;</td></tr><tr><td>Clothing</td><td>T-shirt Jacket Shirt Sweater Dress Coat Vest</td><td>&lt;clothing_tshirt&gt; &lt;clothing_jacket&gt; &lt;clothing_shirt&gt; &lt;clothing_sweater&gt; &lt;clothing_dress&gt; &lt;clothing_coat&gt;</td></tr><tr><td>Hair</td><td>Curly Hair Straight Hair Bald Short Hair Long Hair</td><td>&lt;clothing_vest&gt; &lt;hair_curly&gt; &lt;hair_straight&gt; &lt;hair_bald&gt; &lt;hair_short&gt; &lt;hair_long&gt;</td></tr></table>

Overall, the standard deviation of metrics across these reasonable hyperparameter ranges remains small. We consistently observe that a higher cluster representation smoothing momentum (ema\_ $. \mathsf { r e p } = 0 . 9 5 ;$ named ema in an earlier version, distinct from the GMM threshold smoothing coeficient $\eta _ { \mathrm { g m m } } )$ is beneficial for preventing abrupt threshold oscillation in online scenarios. Furthermore, challenging datasets with darker scenes (e.g., Bufy) favor a larger GMM sliding window (queue = 300) to gather enough diverse samples for stable threshold estimation, whereas visually stable datasets (e.g., TBBT) can adapt faster with a smaller queue. Note that the $\tau _ { \mathrm { h i g h } }$ spread in these tables reflects simultaneous covariation of ema\_rep and queue size; these marginal tables are not a one factor test of $\tau _ { \mathrm { h i g h } }$ alone. A dedicated 3 × 3 warm up threshold grid with all other settings fixed is reported in Section S12.

## S6 Semantic Arbitration Invocation Rate Visualization

Figure S1 visually reinforces the high eficiency of VISTA, comple menting the raw statistics reported in the main paper. Across all three datasets, the triage mechanism successfully limits semantic arbitration by routing the vast majority of frames (86.5%-90.8%) through the fast, visual only pipeline. As clarified in the main paper (Section 4.6), CLIP/DeepFace attributes are extracted up front for every face crop; these rates measure how often the arbitration step fires during association, not how many VLM forward passes are avoided. Notably, TBBT requires the least arbitration (9.2%), consistent with its well lit studio conditions. Conversely, Bufy triggers the highest rate (13.5%) due to frequent low illumination scenes and fast motion that push more track cluster pairs into the ambiguity zone. Sherlock falls between the two (11.4%) despite being the hardest dataset overall, suggesting that its dificulty stems more from extreme long range appearance variation (e.g., character disguises) than from persistent local visual ambiguity.

Table S4: TBBT parameter efects under dataset specific marginal analysis. Values indicate average performance when the specific parameter is fixed while others vary.
<table><tr><td>Parameter age</td><td>Value broad_adult</td><td>WCP / NMI / ID_Acc 96.66 / 87.57 / 89.40</td><td>Bal. Avg. 91.21</td></tr><tr><td>age</td><td>coarse_grouping fine_grained</td><td>96.70 / 87.15 / 88.58 96.60 / 87.01 / 88.68</td><td>90.81 90.76</td></tr><tr><td>age ema_rep</td><td>0.95</td><td>97.47 / 89.43 / 91.32</td><td>92.74</td></tr><tr><td>ema_rep</td><td>0.75</td><td>95.84 / 85.05 / 86.46</td><td>89.11</td></tr><tr><td>queue</td><td>300.0</td><td>96.97 / 87.65 / 89.17</td><td>91.26</td></tr><tr><td>queue</td><td>150.0</td><td>96.34 / 86.83 / 88.60</td><td>90.59</td></tr><tr><td>vishigh</td><td>0.75</td><td>96.65 / 87.24 / 88.89</td><td>90.93</td></tr><tr><td>vishigh</td><td>0.825</td><td>96.65 / 87.24 / 88.89</td><td>90.93</td></tr><tr><td>vishigh</td><td>0.9</td><td>96.65 / 87.24 / 88.89</td><td>90.93</td></tr><tr><td>vislow</td><td></td><td></td><td></td></tr><tr><td>vislow</td><td>0.275</td><td>96.64 / 87.30 / 88.97</td><td>90.97</td></tr><tr><td></td><td>0.2</td><td>96.65 / 87.21 / 88.88</td><td>90.91</td></tr><tr><td>vislow</td><td>0.35</td><td>96.67  / 87.20 / 88.82</td><td>90.90</td></tr></table>

Table S5: Bufy parameter efects under dataset specific marginal analysis.
<table><tr><td>Parameter</td><td>Value coarse_grouping</td><td>WCP / NMI / ID_Acc 84.85 / 75.24 / 76.98</td><td>Bal. Avg. 79.02</td></tr><tr><td>age age</td><td>broad_adult fine_grained</td><td>84.18 / 74.50 / 76.02 81.62 / 72.17 / 73.49</td><td>78.23 75.76</td></tr><tr><td>age</td><td>0.95</td><td>85.18 / 77.22 / 79.68</td><td>80.69</td></tr><tr><td>ema_rep ema_rep</td><td>0.75</td><td>81.87 / 70.67 / 71.24</td><td>74.59</td></tr><tr><td></td><td>150.0</td><td>84.70 / 75.01 / 76.31</td><td>78.67</td></tr><tr><td>queue queue</td><td>300.0</td><td>82.39 / 72.93 / 74.67</td><td>76.66</td></tr><tr><td>vishigh</td><td></td><td>84.67 / 75.04 / 76.57</td><td>78.76</td></tr><tr><td>vishigh</td><td>0.825 0.9</td><td>84.67 / 75.04 / 76.57</td><td>78.76</td></tr><tr><td>vishigh</td><td>0.75</td><td>81.31 / 71.85 / 73.36</td><td>75.51</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>vislow</td><td>0.35</td><td>84.60 / 75.08 / 76.72</td><td>78.80</td></tr><tr><td>vislow</td><td>0.2</td><td>84.86 / 75.05 / 76.48</td><td>78.80</td></tr><tr><td>vislow</td><td>0.275</td><td>81.20 / 71.79 / 73.30</td><td>75.43</td></tr></table>

## S7 Semantic Arbitration Weight Sensitivity

To systematically assess the robustness of the semantic arbitration function introduced in the main paper, we perform a comprehensive grid search over 243 configurations covering all five semantic weight dimensions:

$$
\lambda _ { \mathrm { g a } } \in \{ 0 . 0 4 , 0 . 0 8 , 0 . 1 2 \} , \qquad \lambda _ { \mathrm { c l o t h } } \in \{ 0 . 0 2 , 0 . 0 5 , 0 . 0 8 \} ,
$$

$$
\lambda _ { \mathrm { g m } } \in \{ 0 . 0 8 , 0 . 1 2 , 0 . 1 6 \} ,
$$

$$
\lambda _ { \mathrm { a m } } \in \{ 0 . 0 1 , 0 . 0 3 , 0 . 0 5 \} ,
$$

$$
\lambda _ { \mathrm { h c } } \in \{ 0 . 0 2 , 0 . 0 4 , 0 . 0 6 \} .
$$

![](images/980ac3b573e799939f1a794ea1971e3a8e1bf3cc20b9111f71fb9b6077345632.jpg)

![](images/847e23ceeb94c9f63bb73620f001cad67339a8a0d257a2da2a86116fe5bd76e2.jpg)

![](images/cd9f139a0b3706b46abb1e776d198dbff8bdba7c870ab9d11dae16792e4f1416.jpg)

![](images/2dddb8b23327e2be1a0700a83cfc625b234ab19d6902c8388bdaea64c559ceae.jpg)  
Figure S1: Frame level semantic arbitration invocation rate per dataset and overall. The dynamic triage mechanism restricts arbitration to a small fraction of ambiguous frames, preserving near real time processing speeds.

Table S6: Sherlock parameter efects under dataset specific marginal analysis.
<table><tr><td>Parameter age</td><td>Value broad_adult</td><td>WCP / NMI / ID_Acc 80.16 / 46.69 / 58.51</td><td>Bal. Avg. 61.79</td></tr><tr><td>age</td><td>fine_grained</td><td>80.23 / 46.68 / 58.08 80.38 / 46.67 / 55.58</td><td>61.66 60.87</td></tr><tr><td>age</td><td>coarse_grouping</td><td></td><td>64.26</td></tr><tr><td>ema_rep ema_rep</td><td>0.95 0.75</td><td>81.88 / 49.36 / 61.54 78.63 / 44.00 / 53.24</td><td>58.62</td></tr><tr><td></td><td></td><td>80.04 / 46.72 / 58.56</td><td>61.77</td></tr><tr><td>queue queue</td><td>150.0 300.0</td><td>80.47  / 46.64 / 56.22</td><td>61.11</td></tr><tr><td>vishigh</td><td>0.75</td><td>80.25 / 46.68 / 57.39</td><td>61.44</td></tr><tr><td></td><td>0.825</td><td>80.25 / 46.68 / 57.39</td><td>61.44</td></tr><tr><td>vishigh vishigh</td><td>0.9</td><td></td><td>61.44</td></tr><tr><td></td><td></td><td>80.25 / 46.68 / 57.39</td><td></td></tr><tr><td>vislow</td><td>0.2</td><td>80.41 / 46.82 / 57.69</td><td>61.64</td></tr><tr><td>vislow</td><td>0.275</td><td>80.27 / 46.74 / 57.57</td><td>61.53</td></tr><tr><td>vislow</td><td>0.35</td><td>80.08 / 46.48 / 56.92</td><td>61.16</td></tr></table>

Overall stability. Table S7 reports the minimum, maximum, and standard deviation of WCP and NMI across all 243 configurations for each dataset. The remarkably low standard deviations (e.g., only 0.31 for WCP on TBBT) confirm that VISTA’s clustering gains are not the product of fragile weight tuning.

Table S7: Performance variance across 243 semantic weight (�) configurations.
<table><tr><td>Dataset</td><td>Metric</td><td>Min</td><td>Max</td><td>Std</td></tr><tr><td rowspan="2">TBBT</td><td>WCP</td><td>96.08</td><td>97.59</td><td>0.31</td></tr><tr><td>NMI</td><td>90.20</td><td>91.79</td><td>0.27</td></tr><tr><td rowspan="2">Buffy</td><td>WCP</td><td>84.71</td><td>90.34</td><td>1.09</td></tr><tr><td>NMI</td><td>76.86</td><td>80.96</td><td>0.74</td></tr><tr><td rowspan="2">Sherlock</td><td>WCP</td><td>79.22</td><td>83.44</td><td>0.78</td></tr><tr><td>NMI</td><td>48.90</td><td>53.37</td><td>0.78</td></tr></table>

Marginal efects. Table S8 isolates the marginal efect of each specific semantic trait. We observe that persistent traits (gender and age) heavily dictate performance. Specifically, the gender mismatch penalty $\lambda _ { \mathrm { g m } }$ and gender+age reward $\lambda _ { \mathrm { g a } }$ exhibit the largest influence on the Balanced Average. In contrast, transient cues like clothing reward $\lambda _ { \mathrm { c l o t h } }$ and hair conflict $\lambda _ { \mathrm { h c } }$ are comparatively insensitive, validating VISTA’s decision to decouple transient and persistent traits.

Table S8: Cross dataset marginal efects ofarbitration weights. Each row averages over 81 configurations sharing that specific weight value. Bal. Avg. = (WCP + NMI + ID\_Acc) / 3.
<table><tr><td rowspan="2">Weight</td><td rowspan="2">Value</td><td colspan="3">Bal. Avg. (%)</td></tr><tr><td>TBBT</td><td>Buffy</td><td>Sherlock</td></tr><tr><td rowspan="3"> $\lambda _ { \mathrm { g a } }$ </td><td>0.04</td><td>93.88</td><td>83.53</td><td>65.22</td></tr><tr><td>0.08</td><td>93.81</td><td>82.42</td><td>65.70</td></tr><tr><td>0.12</td><td>94.01</td><td>82.40</td><td>66.29</td></tr><tr><td rowspan="3"> $\lambda _ { \mathrm { c l o t h } }$ </td><td>0.02</td><td>93.93</td><td>82.82</td><td>65.63</td></tr><tr><td>0.05</td><td>93.88</td><td>82.81</td><td>65.76</td></tr><tr><td>0.08</td><td>93.89</td><td>82.72</td><td>65.82</td></tr><tr><td rowspan="3"> $\lambda _ { \mathrm { g m } }$ </td><td>0.08</td><td>93.95</td><td>82.74</td><td>65.64</td></tr><tr><td>0.12</td><td>93.93</td><td>82.77</td><td>65.63</td></tr><tr><td>0.16</td><td>93.82</td><td>82.84</td><td>65.95</td></tr><tr><td rowspan="3"> $\lambda _ { \mathrm { a m } }$ </td><td>0.01</td><td>93.94</td><td>82.68</td><td>65.77</td></tr><tr><td>0.03</td><td>93.88</td><td>82.76</td><td>65.72</td></tr><tr><td>0.05</td><td>93.88</td><td>82.90</td><td>65.73</td></tr><tr><td rowspan="3"> $\lambda _ { \mathrm { h c } }$ </td><td>0.02</td><td>93.93</td><td>82.90</td><td>65.73</td></tr><tr><td>0.04</td><td>93.88</td><td>82.73</td><td>65.72</td></tr><tr><td>0.06</td><td>93.89</td><td>82.72</td><td>65.77</td></tr></table>

Robustness over absolute peaks. It is worth noting that the default weights used in the main paper’s unified evaluation $( \mathrm { e . g . , } \lambda _ { \mathrm { g m } } =$ $0 . 1 2 , \lambda _ { \mathrm { c l o t h } } ~ = ~ 0 . 0 4 )$ were deliberately selected as a robust global compromise rather than absolute dataset specific peaks. While Table S8 shows that tuning these weights per dataset could yield marginally higher scores (e.g., Bufy peaks at $\lambda _ { \mathrm { g a } } = 0 . 0 4$ while Sherlock peaks at 0.12), the unified setting ensures strong out of the box generalization. The minimal performance variance across these settings ultimately proves that VISTA’s efectiveness stems from its core architectural design symbolic denoising and spatio temporal triage rather than fragile hyperparameter overfitting.

## S8 GMM Bimodal Fitting Visualization

To empirically validate the bimodal assumption underlying VISTA’s dynamic thresholding (Section 3.3 of the main paper), we visualize

## GMM Bimodal Fitting of Track-to-Cluster Cosine Similarity

![](images/e9144774d954a4b122f5a317edb1cdc3b4065d86035328262e6c1b417003cd91.jpg)  
Figure S2: GMM bimodal fitting of track to cluster cosine similarity distributions across all 15 episodes (86,230 total scores). Each panel shows the empirical histogram overlaid with the fitted two component GMM. The dashed red and solid blue curves represent the diferent identity and same identity components, respectively. Vertical dotted lines indicate the derived $\tau _ { \mathrm { l o w } }$ and $\tau _ { \mathbf { h i g h } }$ thresholds after clipping.

the GMM fit across all 15 episodes in Figure S2. The cosine similarity scores between track cluster pairs are aggregated per dataset and fitted with a two component GMM.

All three datasets exhibit clear bimodal structure, with the same identity component concentrated at $\mu \approx 0 . 8 2 – 0 . 8 6$ and the diferent identity component at $\mu \approx 0 . 5 8 – 0 . 6 0$ . TBBT shows the tightest same identity distribution $\scriptstyle ( \sigma = 0 . 0 6 ) ,$ , consistent with its controlled studio lighting. Bufy and Sherlock exhibit wider spread $( \sigma { = } 0 . 0 8 ) ;$ reflecting greater illumination and appearance variation. The estimated $\tau _ { \mathrm { l o w } }$ and $\tau _ { \mathrm { h i g h } }$ thresholds consistently partition the score space into well separated reject, ambiguity, and merge zones. The raw GMM derived $\tau _ { \mathrm { l o w } }$ values $( \mu _ { 1 } + \sigma _ { 1 } \approx 0 . 7 0  – 0 . 7 5 )$ are bounded via the online proposal rule $\tau _ { \mathrm { l o w } } ^ { \star } = \mathrm { c l i p } ( \mu _ { 1 } + \sigma _ { 1 } , 0 . 1 5 , 0 . 5 0 )$ , followed by EMA smoothing with $\eta _ { \mathrm { g m m } } = 0 . 3$ (Section 3.3 of the main paper). The 0.50 cap prevents the lower boundary from drifting too high in visually compressed windows, preserving an ambiguity zone in which semantic arbitration can operate; it is an intentional design choice rather than a fitting artifact. In an isolated online cap sweep over all 15 episodes, relaxing the cap to 0.70 or 1.00 yields identical WCP/NMI/ID\_Acc after rounding to two decimals: once the ambiguity zone is preserved, downstream best assignment selection and the arbitration gate $\tau _ { \mathrm { m e r g e } }$ dominate the final accept/reject decisions. The cap therefore acts as a declared stability safeguard rather than a tuned performance knob.

## S9 Intra Shot Tracklet Length Ablation

Table S9 expands upon the tracklet length ablation discussed in Section 4.4 of the main paper. As shown, increasing the maximum intra shot tracklet length from 12 to 48 frames improves Identity Accuracy $( \mathrm { I D \_ A c c } )$ and NMI by accumulating richer temporal context and producing more stable running average embeddings.

However, over extending the window to 96 frames begins to degrade performance due to intra track appearance noise (e.g., characters turning completely away from the camera). Note that

Precision, Recall, and F1 remain perfectly constant across these variations. This is expected, as the upstream face detection bounding boxes and initial track candidate generation are fixed; the tracklet length parameter only dictates downstream cluster aggregation and inter shot tracking logic. Based on this trade of, we adopted $L _ { \mathrm { m a x } } = 2 4$ as our unified default to maintain high Weighted Cluster Purity (WCP) without sacrificing latency.

Table S9: Intra shot tracklet length ablation, averaged over all 15 episodes (WCP / NMI / ID\_Acc / Precision / Recall / F1 %). Upstream detection metrics (Precision, Recall, F1) are naturally unafected by downstream aggregation logic.
<table><tr><td>Length</td><td>WCP</td><td>NMI</td><td>ID_Acc</td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td>12</td><td>92.27</td><td>77.39</td><td>77.88</td><td>81.16</td><td>96.19</td><td>87.84</td></tr><tr><td>24</td><td>92.83</td><td>77.92</td><td>79.14</td><td>81.16</td><td>96.19</td><td>87.84</td></tr><tr><td>48</td><td>91.89</td><td>78.72</td><td>81.05</td><td>81.16</td><td>96.19</td><td>87.84</td></tr><tr><td>96</td><td>91.23</td><td>78.50</td><td>80.79</td><td>81.16</td><td>96.19</td><td>87.84</td></tr></table>

## S10 Additional Qualitative Results

Figure S3 provides the crop level qualitative comparison on Sherlock referenced in Section 4.5 of the main paper (moved here for space). Under challenging pose, illumination, and motion blur variations, VISTA maintains higher local purity than COFC and minimizes identity confusion between visually similar characters.

## S11 Additional Online Baselines: Full Results

This section provides the full per dataset results for the two additional online baselines summarised in Section 4.3 of the main paper. Both baselines use the same detector (RetinaFace), embeddings (FaceNet), tracks, and evaluation protocol as VISTA and COFC.

Naive VLM hard constraint baseline. Raw VLM attributes (no SSR, no triage) are applied as must link/cannot link rules. Table S10 shows severe over fragmentation: the number of predicted clusters exceeds ground truth by 4.0–11.3×. This inflates WCP while ID\_Acc collapses far below VISTA Full (93.94/82.82/68.10 on TBBT/Bufy/Sherlock) and below the weakest geometry driven ablation in the main paper, confirming that VISTA’s gains stem from selective semantic triage rather than from VLM attributes alone.

Table S10: Naive VLM hard constraint baseline (%). #Pred/ep and #GT/ep are per episode averages by series.
<table><tr><td>Hard-VLM</td><td>WCP</td><td>NMI</td><td>ID_Acc</td><td>#Pred/ep</td><td>#GT/ep</td></tr><tr><td>TBBT</td><td>97.98</td><td>76.57</td><td>63.95</td><td>83.8</td><td>20.5</td></tr><tr><td>Buffy</td><td>91.01</td><td>72.99</td><td>60.64</td><td>108.5</td><td>27.0</td></tr><tr><td>Sherlock</td><td>84.67</td><td>44.98</td><td>40.65</td><td>136.0</td><td>12.0</td></tr></table>

Sequential online HAC baseline. A sequential online HAC variant on the same tracks reaches very high WCP, as expected under fragmentation, but substantially lower NMI and ID\_Acc, with severe over fragmentation (Table S11).

Table S11: Sequential online HAC baseline (%). The last column is the ratio of predicted clusters to ground truth clusters per episode.
<table><tr><td>Online HAC</td><td>WCP</td><td>NMI</td><td>ID_Acc</td><td>#Pred/#GT</td></tr><tr><td>TBBT</td><td>99.73</td><td>84.93</td><td>81.29</td><td>4.5×</td></tr><tr><td>Buffy</td><td>97.62</td><td>76.78</td><td>63.49</td><td>9.5×</td></tr><tr><td>Sherlock</td><td>91.19</td><td>49.28</td><td>40.56</td><td>39.8X</td></tr></table>

## S12 Additional Threshold Sensitivity Analyses

Warm up threshold grid. Complementing the marginal tables in Section S5, we run a dedicated 3×3 grid bracketing the warm up initialisation over $\tau _ { \mathrm { h i g h } } ^ { ( 0 ) } \in \{ 0 . 7 5 , 0 . 8 2 5 , 0 . 9 0 \}$ and $\tau _ { \mathrm { l o w } } ^ { ( 0 ) } \in \{ 0 . 2 0 , 0 . 2 7 5 , 0 . 3 5 \}$ with all other settings fixed (135 episode level evaluations). Table S12 reports the per dataset max−min variation, which remains below 0.7 pp for all metrics. �<sub>h h</sub> shows limited sensitivity in this grid precisely because GMM dynamic thresholding overrides the static initialisation after warm up.

Table S12: Max−min variation (pp) across the $3 \times 3$ warm up threshold grid (all other settings fixed).
<table><tr><td>Metric</td><td>TBBT</td><td>Buffy</td><td>Sherlock</td></tr><tr><td>WCP</td><td>0.18</td><td>0.15</td><td>0.67</td></tr><tr><td>NMI</td><td>0.41</td><td>0.15</td><td>0.30</td></tr><tr><td>ID_Acc</td><td>0.67</td><td>0.37</td><td>0.69</td></tr></table>

�<sub>merge</sub> sensitivity. We sweep the arbitration acceptance gate over $\tau _ { \mathrm { m e r g e } } \in \{ 0 . 2 8 0 , 0 . 3 2 5 , 0 . 3 7 5 , 0 . 4 2 5 , 0 . 4 7 5 \}$ . The sweep confirms the expected purity–continuity trade of: at the most conservative setting (0.475), Sherlock ID\_Acc drops by 8.37 pp relative to the 0.325 default, reflecting conservative fragmentation rather than a standalone visual threshold failure.

�<sub>low</sub> cap sweep. As reported in Section S8, an isolated online cap sweep over all 15 episodes shows that relaxing the 0.50 cap on $\tau _ { \mathrm { l o w } } ^ { \star }$ to 0.70 or 1.00 yields identical WCP/NMI/ID\_Acc after rounding to two decimals, confirming that the cap is a declared stability safeguard rather than a tuned performance knob.

## S13 Subgroup Clustering Quality

Complementing the triage rate analysis in the main paper (Section 4.7), Table S13 reports subgroup clustering quality gaps on the IoU matched face subset, using the same VLM predicted gender/age groups. VPCD provides no ground truth demographic annotations, so these figures should be read as a diagnostic rather than a full fairness audit. The signs of the gaps vary across datasets, indicating character and dataset dependent efects rather than a one directional degradation. As an ethical safeguard, our implementation provides an option to disable the gender/age attributes entirely, in which case arbitration falls back to transient cues and geometry only.

COFC  
VISTA  
![](images/b9af6d17de51aadfa35e6da69cdfd25be8ffc030a3379371df907d0a8703e5aa.jpg)  
Figure S3: Local clustering purity for Mycroft and Sherlock (Green: true positives, Red: false positives). VISTA maintains higher purity than COFC across challenging pose, illumination, and motion blur variations, minimizing identity confusion.

Table S13: Subgroup clustering quality gaps (pp) on the IoU matched subset using VLM predicted demographic groups. Positive values indicate the first group scores higher.
<table><tr><td>Gap</td><td>Dataset</td><td>WCP</td><td>NMI</td><td>ID_Acc</td></tr><tr><td>Gender</td><td>TBBT</td><td>-0.5</td><td>+13.1</td><td>+1.7</td></tr><tr><td>(M−F)</td><td>Buffy Sherlock</td><td>+2.1 -6.6</td><td>+14.7 -19.8</td><td>+5.0 -1.4</td></tr><tr><td></td><td>TBBT</td><td>+2.0</td><td>+12.0</td><td>+13.4</td></tr><tr><td>Age</td><td></td><td></td><td></td><td></td></tr><tr><td>(adult-senior)</td><td>Buffy</td><td>-2.3</td><td>+4.7</td><td>-0.6</td></tr><tr><td></td><td>Sherlock</td><td>-6.6</td><td>+13.4</td><td>+1.1</td></tr></table>

## References

[1] Sefik Ilkin Serengil and Alper Ozpinar. 2024. A Benchmark of Facial Recognition Pipelines and Co-Usability Performances of Modules. In Journal ofInformation Technologies, Vol. 17. 95–107.