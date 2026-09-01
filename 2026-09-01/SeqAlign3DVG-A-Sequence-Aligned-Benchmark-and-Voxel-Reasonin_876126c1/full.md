![](images/9baa816896d085711b1ad98a5371e8ab5f20c94670fc4b4536640f8187b044e6.jpg)

# SeqAlign3DVG: A Sequence-Aligned Benchmark and Voxel Reasoning Framework for 3D Visual Grounding

Yi Zhang   
yi-eee.zhang@connect.polyu.hk   
The Hong Kong Polytechnic   
University   
Hong Kong, China

Kaiyue Yang sheesage36@sjtu.edu.cn Shanghai Jiao Tong University Shanghai, China

Yi Wang ✉   
yi-eie.wang@polyu.edu.hk   
The Hong Kong Polytechnic   
University   
Hong Kong, China   
Yuejiao Su   
yuejiao.su@connect.polyu.hk   
The Hong Kong Polytechnic   
University   
Hong Kong, China   
Yueting Wu   
lisala.wu@connect.polyu.hk   
The Hong Kong Polytechnic   
University   
Hong Kong, China   
Lap-Pui Chau<sup>✉</sup>   
lap-pui.chau@polyu.edu.hk   
The Hong Kong Polytechnic   
University   
Hong Kong, China

Single-view grounding Query directly paired with one RGB-D image

![](images/c9467617d0e5b938a4b33de8012634dcddb7bd680d2f740058dc38d3cf646f66.jpg)  
Query: there is a brown wooden dresser beside the mirror.  
SUNRefer  
Full-scene 3D grounding Query directly paired with the reconstructed point cloud  
Scene-object grounding with supporting views Query directly paired with a 3D target/relation;  
Query: the bag that is far away from the bathtub.

![](images/8e974e3212e61efa6371f678393edc8a77f6c37d84271626f849d1238edb89b6.jpg)  
Query: there is a dark brown chair. brown leather and placed in the kitchen table.  
ScanRefer

![](images/e45e940d06afead871f850715162c5372990e64f0f7b79b3bf58b403483f3bab.jpg)  
MMScan & EmbodiedScan

Figure 1: Benchmark taxonomy by what the query is directly paired with. Unlike prior works relying on global scenes or unordered views, our SeqAlign3DVG ensures strict alignment between queries and observation sequences or single views.

## Abstract

Image-based 3D visual grounding is critical for embodied agents, yet existing benchmarks sufer from loose text-observation align ment and neglect temporal ordering. We introduce SeqAlign3DVG, a novel benchmark dedicated to temporally ordered and strictly observation-aligned image-based 3D visual grounding. Unlike prior works using order-agnostic views or global point clouds, SeqAlign3DVG ensures all expressions are human-verified and strictly grounded in the provided RGB observations (single frames or ordered observation sequences). It comprises 9,622 single-view and 14,493 sequence samples featuring rich descriptions, complex relations, and multi-instance ambiguities. To tackle this benchmark, we propose a unified voxel-based pipeline featuring Relevance-Ordered

![](images/f5de1e09069e9215df89502b454b552ec2fd832be2c3fa9d97563311f92190f6.jpg)

Voxel Memory (ROVM) and Progressive Language-Voxel Fusion (PLVF). ROVM dynamically ranks and aggregates multi-view evidence via a conservative memory to mitigate noisy observations, while PLVF performs coarse-to-fine spatial-linguistic reasoning for precise disambiguation. Our approach achieves state-of-the-art performance under the depth-free protocol, significantly improving localization for targets defined by complex relations and appearance cues.

## CCS Concepts

• Computing methodologies → Scene understanding; Object detection; Vision for robotics.

## Keywords

3D Visual Grounding; Embodied Perception; Vision-Language; Scene Understanding; Voxel Memory; Benchmark

ACM Reference Format: Yi Zhang, Yi Wang, Yueting Wu, Kaiyue Yang, Yuejiao Su, and Lap-Pui Chau. 2026. SeqAlign3DVG: A Sequence-Aligned Benchmark and Voxel

Reasoning Framework for 3D Visual Grounding. In Proceedings ofthe 34th ACM International Conference on Multimedia (MM ’26), November 10–14, 2026, Rio de Janeiro, Brazil. ACM, New York, NY, USA, 9 pages. https://doi. org/10.1145/3767308.3835644

## 1 Introduction

Grounding free-form referring expressions to 3D objects is essential for embodied agents and AR systems to interact with the physical world, facilitating downstream tasks like navigation and manipulation. However, a critical gap exists between current 3D visual grounding benchmarks and realistic embodied perception. Most existing benchmarks [2, 4, 11, 27] assume access to fully reconstructed global 3D scenes (e.g., complete point clouds or meshes). In practical indoor deployments, agents perceive environments through egocentric motion trajectories characterized by limited fields of view, partial observability, and viewpoint-dependent evidence. This discrepancy necessitates image-based 3D visual grounding: systems must infer a target’s 3D location directly from sequential visual observations available during the perceptual process, rather than relying on a priori global reconstructions.

Existing image-based benchmarks, however, fall short of capturing these realistic dynamics. Early datasets are largely restricted to single-view grounding [16] or outdoor driving scenarios [12, 26]. While recent indoor embodied benchmarks, such as EmbodiedScan [22] and MMScan [19], have moved toward multi-view grounding, their paradigm for attaching language supervision introduces two fundamental limitations. In these datasets, expressions are typically associated with a scene-level target or a target-anchor relation, and the supporting views are subsequently selected through 3Dto-2D projection and visibility checks. Consequently, the language is not directly paired with the exact observations used for evaluation: described appearance cues or spatial evidence may be heavily occluded, ambiguous, or entirely absent in the provided frames, leading to a severe observability mismatch. Moreover, the retrieved views are treated as an unordered "bag of views" rather than a trajectory-preserving, temporally ordered observation sequence. This formulation neglects the temporal dynamics inherent in embodied perception, bypassing how agents progressively resolve visual ambiguities over time.

To address these critical issues, we introduce SeqAlign3DVG, an observation-aligned indoor benchmark for 3D visual grounding from both single views and observation sequences. SeqAlign3DVG supports two settings under a unified protocol: single-view grounding from an isolated RGB observation, and observation-sequence grounding from a temporally ordered RGB sequence. All expressions are human-verified against the exact evaluation observation(s), enforcing strict observation alignment and ensuring that the target can be uniquely identified from the provided visual evidence. Beyond alignment, SeqAlign3DVG is deliberately challenging: its expressions combine appearance cues and spatial relations, and a large fraction of the samples involve same-class distractors that require fine-grained disambiguation.

SeqAlign3DVG also highlights critical modeling challenges for image-based grounding. First, in realistic indoor trajectories, view quality is highly uneven: some frames capture clear target evidence, while others are heavily occluded or only observe supporting anchors. However, existing multi-view aggregation strategies typically treat inputs as order-agnostic sets and rely on uniform feature fusion. This inevitably blurs sharp geometric cues when integrating noisy or partial observations. Second, indoor queries often involve complex spatial relations and same-class distractors, requiring fine-grained token-level disambiguation. Standard 3Dlanguage fusion mechanisms [12, 22] typically apply dense global cross-attention between text and full 3D representations. This is computationally expensive and struggles to isolate subtle visual attributes in cluttered spaces. Therefore, a robust pipeline must intelligently filter uneven temporal evidence and perform focused, noise-reduced spatial-linguistic reasoning.

Motivated by this, we build a voxel-based grounding pipeline that supports both single-view and observation sequence inputs. Our pipeline lifts per-view 2D features into 3D voxel volumes using camera projection and regresses 3D bounding boxes directly. To efectively aggregate temporal sequences, we introduce a Relevance-Ordered Voxel Memory (ROVM) that ranks views by query relevance and updates a conservative memory to preserve high-quality evidence while suppressing noise. For fine-grained spatial-linguistic disambiguation, we propose Progressive Language-Voxel Fusion (PLVF), which establishes a coarse target hypothesis before applying sparse token-level cross-attention exclusively on the most promising voxels. We further stabilize training with an auxiliary occupancy supervision signal derived from ground-truth 3D boxes.

Empirically, our method achieves state-of-the-art performance under the depth-free protocol. On the observation-sequence split, it obtains 51.30%/22.39% Acc@0.25/0.50, exceeding BIP3D [15] without ground-truth depth supervision but with detector initialization by 1.25/2.27 percentage points, while using neither target phrasetoken supervision nor a pretrained 3D detector prior.

In summary, we make three contributions:

• We introduce SeqAlign3DVG, an observation-aligned indoor benchmark for 3D visual grounding from single views and observation sequences, with human-verified expressions tied to the exact evaluation observation(s).

• We establish a unified voxel-based grounding pipeline and propose Relevance-Ordered Voxel Memory (ROVM) for queryaware sequence aggregation, alongside Progressive Language-Voxel Fusion (PLVF) for eficient spatial-linguistic disambiguation.

• We achieve state-of-the-art performance on SeqAlign3DVG under the depth-free protocol.

## 2 Related Work

## 2.1 3D Visual Grounding Benchmarks

Reconstructed indoor 3D-scene benchmarks. Most established 3D visual grounding (3DVG) benchmarks are built on reconstructed indoor scans, evaluating grounding with explicit 3D scene representations (e.g., point clouds or meshes). ScanRefer [4] and ReferIt3D [2] pioneered this field by providing free-form natural language and template-based relational utterances, respectively. Subsequent works like Multi3DRefer [27], ScanEnts3D [1], and ViGiL3D [21] further expanded the task to multi-target localization, phrase-toobject correspondences, and linguistically diverse diagnostics.

Image-based 3D grounding benchmarks. Complementary to full-scan inputs, image-based benchmarks study grounding from posed camera views, where partial observability becomes a central challenge. Early datasets focused on single-view RGB-D grounding (SUNRefer [16]) or outdoor driving scenarios (Mono3DRefer [26], NuGrounding [12]). Recently, embodied benchmarks like Embodied-Scan [22] and MMScan [19] pushed toward ego-centric multi-view streams. However, their language is not necessarily written and verified against the exact evaluation observations, and their supporting frames are treated as order-agnostic sets. SeqAlign3DVG provides human-verified expressions strictly conditioned on the exact single-view or temporally ordered sequence inputs used for evaluation.

Table 1: Comparison with representative indoor 3D visual grounding benchmarks. Among the listed benchmarks, SeqAlign3DVG alone combines strict observation alignment, ordered sequences, and a unified setting for single-view and sequence grounding.
<table><tr><td>Dataset</td><td>Evaluation input</td><td>Exact evaluation- observation alignment†</td><td>Ordered sequence</td><td>Free-form language</td><td>Single-view + sequence</td><td>Instances</td></tr><tr><td>ScanRefer [4]</td><td>Global 3D scene</td><td>一</td><td>x</td><td>√</td><td>x</td><td>11k</td></tr><tr><td>Nr3D [2]</td><td>Global 3D scene</td><td>一</td><td>x</td><td>√</td><td>x</td><td>6k</td></tr><tr><td>SUNRefer [16]</td><td>Single RGB-D view</td><td>√</td><td>x</td><td>√</td><td>x</td><td>7k</td></tr><tr><td>EmbodiedScan [22]</td><td>Multi-view set</td><td>x</td><td>x</td><td>x</td><td>x</td><td>27k</td></tr><tr><td>MMScan [19]</td><td>Multi-view set</td><td>x</td><td>x</td><td>√</td><td>x</td><td>61k</td></tr><tr><td>SeqAlign3DVG (ours)</td><td>Single view + sequence</td><td>√</td><td>√</td><td>√</td><td>√</td><td>~10k</td></tr></table>

Instances denote the number of unique target instances. <sup>†</sup> For image-based inputs, exact evaluation-observation alignment means that the released query is written and checked against the exact observation unit paired with that sample; “–” denotes that this view-level property is not applicable to a global 3D-scene input.

## 2.2 3D Visual Grounding with Explicit 3D Geometry

A broad line of work assumes access to explicit 3D geometry (e.g., point clouds with instance candidates). Many follow a grounding by-matching paradigm, utilizing instance-level contextual modeling and relation reasoning [5, 7, 25]. Other works enhance robustness via multi-view projections [8] or adopt detection-transformer-style formulations [10, 18]. Beyond closed-set supervised 3DVG, recent work explores open-world zero-shot 3D grounding by leveraging large foundation models [9, 13]. Distinct from pre-scanned point clouds, another stream derives geometry from RGB-D inputs. Methods like SUNRefer [16], DenseGrounding [31], and DEGround [30] bridge 2D features with 3D spatial perception using depth maps. Since these approaches rely on explicit geometric reconstruction priors or depth inputs, they are not directly comparable to RGB-only grounding methods under depth-free settings.

## 2.3 RGB-only Image-based 3D Visual Grounding

Compared with geometry-based 3DVG, RGB-only image-based 3D grounding avoids depth input at inference time and must infer 3D cues from visual observations. In monocular driving scenarios, Mono3DVG-TR [26] proposes single-stage transformer-based monocular 3D grounding. Mono3DVG-EnSD [14] further improves performance by decoupling dimension-specific textual features and dynamically masking high-certainty keywords to enforce spatial reasoning. For multi-view autonomous driving, NuGrounding [12] fine-tunes a Multimodal LLM to aggregate 3D geometric priors from BEV features with linguistic instructions for precise localization. However, these methods are predominantly tailored for outdoor driving, whereas BIP3D [15] supports indoor posed-image grounding with optional depth-map input through structured 2D features and explicit 3D position encoding. In our evaluation, its strongest depth-free variant uses target phrase-token supervision and detector initialization, and removing the latter causes a sharp performance drop; our depth-free framework instead uses shared voxel lifting, ROVM, and PLVF for strictly aligned sequences without either additional supervision or prior.

## 3 The SeqAlign3DVG Benchmark

SeqAlign3DVG is an indoor benchmark for image-based 3D visual grounding from both single RGB observations and temporally ordered observation sequences. Its central design principle is strict observation alignment: each released expression is written and verified against the exact observation unit paired with that sample, whether a single view or an ordered sequence. This difers from prior benchmarks, where language is attached to scene-level target/anchor objects or relations, while the associated views are selected afterward according to visibility or benchmark protocol. As a result, exact text–observation alignment is not guaranteed at the sample level. SeqAlign3DVG also explicitly models temporal ordering and evaluates single-view and sequence grounding under a unified setting. Table 1 summarizes these distinctions.

## 3.1 Construction and Quality Control

SeqAlign3DVG is built on ScanNetV2 [6] scenes with registered camera trajectories and aligned 3D instance annotations. We first mine candidate targets together with observation units from the original RGB streams. For the single-view split, an observation unit is one RGB frame in which the target is suficiently visible and can in principle be uniquely identified from the selected view. For the sequence split, an observation unit is a temporally contiguous clip sampled along the original trajectory and preserved in its natural order. Crucially, language is attached to the selected observation unit itself, rather than inherited from a scene-level annotation or post-hoc support-view retrieval.

We use a scalable generate-then-verify pipeline. An automatic stage proposes candidate descriptions conditioned on the selected frame or sequence. Human annotators then verify each sample against the exact evaluation observation(s). A sample is accepted only if (1) the target is uniquely identifiable from the provided input, (2) every mentioned appearance cue, anchor object, and spatial relation is visually supported, and (3) the description does not rely on evidence unobservable anywhere in the provided input or on frames outside the evaluation input. Samples violating any of these criteria are rewritten. Figure 2a illustrates this pipeline and the strict verification criteria.

![](images/a9c579893bf1cc86639b110deca11ae04f4103e50f8d43606980ba329cc166bf.jpg)  
Figure 2: (a) Construction pipeline. Candidate views/sequences are paired with draft descriptions and human-verified against strict observability criteria. (b) Dataset statistics, illustrating the distribution of dificulty levels and anchor categories.

For a sequence sample, each mentioned target cue, anchor, and relation must be supported somewhere in the 15-frame input, but need not appear in every frame. Draft descriptions are generated with Qwen3-VL-32B [3] using structured prompts. Three annotators conduct independent, blind verification, with at least two annotators checking each sample; approximately 59% of sequence drafts and 41% of single-view drafts are rewritten. The train/validation split follows ScanNetV2’s scene allocation, and the core sequence setting uses �=15 ordered keyframes.

## 3.2 Dataset Characteristics and Diagnostic Subsets

The current core benchmark contains 9,622 single-view samples and 14,493 observation-sequence samples. These correspond to roughly 10k unique target instances. SeqAlign3DVG is intentionally designed to be diagnostic rather than merely large: many queries combine appearance cues with anchor objects and spatial relations, and a substantial fraction of samples involve same-class distractors that require fine-grained disambiguation.

To support detailed analysis, we organize the benchmark into diagnostic subsets along two axes, as summarized in Figure 2b. Dificulty is defined by the number of same-class distractors in the corresponding ScanNet scene, yielding Easy (0), Medium (1–2), and Hard (≥ 3) subsets. Anchor usage is characterized by the number of distinct anchor categories mentioned in each expression (0, 1, 2, or 3+). Descriptions with zero anchors rely primarily on intrinsic target cues. These subsets enable fine-grained evaluation of whether a model improves by better local appearance grounding, stronger relation reasoning, or more efective use of temporally accumulated evidence.

## 4 Methodology

## 4.1 Problem Definition and Overall Pipeline

We address image-based 3D visual grounding under a unified setting that supports both single-view and observation-sequence inputs. Given a natural-language description $L ,$ a sequence of RGB images $\boldsymbol { \mathcal { T } } = \{ I _ { t } \} _ { t = 1 } ^ { T }$ , camera intrinsics $\mathcal { K } = \{ \mathbf { K } _ { t } \} _ { t = 1 } ^ { T }$ , and camera extrinsics $\mathcal { T } = \{ \mathbf { T } _ { t } \} _ { t = 1 } ^ { T }$ , our goal is to localize the referred object with a 3D bounding box b $\in \mathbb { R } ^ { 7 }$ . The sequence length is $T = 1$ for single-view grounding and $T > 1$ for observation-sequence grounding.

As shown in Figure 3, our pipeline bridges 1D language, 2D observations, and 3D geometric reasoning through a four-stage workflow: (1) 2D-to-3D Voxel Feature Lifting, which lifts each observation into a per-view 3D voxel volume using camera projection; (2) Relevance-Ordered Voxel Memory (ROVM), which is activated for observation sequences and consolidates the lifted per-view volumes into a query-relevant 3D representation; (3) Progressive Language–Voxel Fusion (PLVF), which performs coarseto-fine language conditioning on the fused 3D volume; and (4) a 3D Grounding Head for final prediction. For the single-view setting, Stage (2) reduces to the identity mapping.

## 4.2 2D-to-3D Voxel Feature Lifting

We construct a unified 3D voxel grid V of resolution $X \times Y \times Z ,$ centered at the scene origin. A shared visual backbone extracts a 2D feature map $\mathbf { F } _ { t } \in \mathbb { R } ^ { C \times H \times W }$ from each image $I _ { t } .$

For each voxel center $\mathbf { v } \in \mathcal { N }$ , we obtain its projected 2D coordinate ${ \mathbf u } _ { t } \in \mathbb { R } ^ { 2 }$ on the �-th image plane via standard camera matrix projection using intrinsics $\mathbf { K } _ { t }$ and extrinsics T<sub>�</sub>. If u<sub>�</sub> lies inside the image bounds and the voxel is in front of the camera, we sample the 2D feature $\mathbf { F } _ { t } ( \mathbf { u } _ { t } )$ by bilinear interpolation.

Instead of collapsing the sequence at this stage, we retain a lifted voxel volume for each observation:

$$
\mathbf { V } _ { t } ( \mathbf { v } ) = \mathbf { m } _ { t } ( \mathbf { v } ) \mathbf { F } _ { t } ( \mathbf { u } _ { t } ) ,\tag{1}
$$

where $\mathbf { m } _ { t } ( \mathbf { v } ) \in \{ 0 , 1 \}$ indicates whether voxel v is projectable into view �, i.e., its projection lies inside the image bounds and the voxel is in front of the camera. This yields a set of per-view lifted volumes $\{ \mathbf { V } _ { t } , \mathbf { m } _ { t } \} _ { t = 1 } ^ { T } ;$ where $\mathbf { V } _ { t } \in \mathbb { R } ^ { X \times \check { Y } \times Z \times C }$ and m<sub>�</sub> $\in \{ 0 , 1 \} ^ { X \times Y \times Z }$

![](images/f04d647dca2270d51a1afdfb46bbc7ef1e1ae1e45690abc9ad3af27af4d10be0.jpg)  
Figure 3: Overall pipeline of our proposed method. It consists of four stages: (1) 2D-to-3D voxel lifting that maps RGB features into per-view volumes using camera projection; (2) Relevance-Ordered Voxel Memory (ROVM) for query-aware sequence aggregation; (3) Progressive Language-Voxel Fusion (PLVF) for fine-grained spatial-linguistic disambiguation; and (4) a 3D grounding head for final bounding box prediction.

For single-view grounding $( T \ = \ 1 )$ , we directly set $\mathbf { V } _ { r } \ = \mathbf { V } _ { 1 }$ and ${ \bf m } _ { r } = { \bf m } _ { 1 }$ . For observation-sequence grounding $( T > 1 ) :$ , the set $\left\{ \mathbf { V } _ { t } , \mathbf { m } _ { t } \right\} _ { t = 1 } ^ { T }$ is passed to the ROVM module for query-aware aggregation.

## 4.3 Relevance-Ordered Voxel Memory

Given lifted per-view voxel volumes $\left\{ \mathbf { V } _ { t } , \mathbf { m } _ { t } \right\} _ { t = 1 } ^ { T }$ and a language description $L ,$ our goal is to aggregate the observation sequence in a query-aware manner. A key dificulty is that view usefulness is highly uneven: some views contain clear target evidence, whereas others are redundant, partially occluded, or only weakly related to the query. This issue is particularly severe for anchor-rich and relation-heavy expressions, where one view may clearly observe the target but not its supporting anchor, while another observes the anchor but only partial target evidence. Uniform fusion therefore tends to blur the cues needed for grounding. To address this, ROVM first estimates a target prior for each view, then ranks views by query relevance, and finally refines the sequence representation through a conservative memory written in relevance order.

Target-Slot-Guided Coarse Prior Estimation. To compare views meaningfully, we first need a coarse estimate of where the referred target may lie in each lifted volume. The description � is encoded into token embeddings $\mathbf { E } \in \mathbb { R } ^ { N \times C }$ and a pooled sentence embedding $\mathbf { e } \in \mathbb { R } ^ { C }$ . Rather than matching every voxel against all tokens at this stage, we extract a compact target slot and use it as a stable targetoriented query for prior estimation. We first compute a pooled voxel context across the whole observation sequence:

$$
\mathbf { c } _ { \mathrm { g l o b } } = \frac { \sum _ { t = 1 } ^ { T } \sum _ { \mathbf { v } \in \mathcal { V } } \mathbf { m } _ { t } ( \mathbf { v } ) \mathbf { V } _ { t } ( \mathbf { v } ) } { \epsilon + \sum _ { t = 1 } ^ { T } \sum _ { \mathbf { v } \in \mathcal { V } } \mathbf { m } _ { t } ( \mathbf { v } ) } .\tag{2}
$$

We then obtain a context-aware target slot

$$
{ \pmb s } _ { \mathrm { t g t } } = g _ { \mathrm { t g t } } \big ( { \bf E } , { \bf e } , \phi _ { \mathrm { c t x } } ( { \bf c } _ { \mathrm { g l o b } } ) \big ) ,\tag{3}
$$

where ${ { g } _ { \mathrm { t g t } } }$ is a lightweight slot extractor that uses a query conditioned on sentence-level text and pooled voxel context to attend over token embeddings and summarize them into a single targetfocused text vector, and $\phi _ { \mathrm { c t x } }$ is a linear projection from voxel to text space.

For each view �, we estimate a voxel-wise target prior by matching projected voxel features against $\begin{array} { r l } { \mathbf { \cal { s } } _ { \mathrm { { t g t } } } . } & { { } \quad } \end{array}$

$$
\begin{array} { r l } & { \mathbf { Z } _ { t } ( \mathbf { v } ) = \beta \operatorname { S i m } \left( \boldsymbol { \Psi } _ { \mathrm { v 2 t } } ( \mathbf { V } _ { t } ( \mathbf { v } ) ) , \mathbf { s } _ { \mathrm { t g t } } \right) , } \\ & { \mathbf { A } _ { t } ( \mathbf { v } ) = \sigma ( \mathbf { Z } _ { t } ( \mathbf { v } ) ) \mathbf { m } _ { t } ( \mathbf { v } ) , } \end{array}\tag{4}
$$

where $\Psi _ { \mathrm { v 2 t } }$ maps voxel features to the text space, $\mathrm { S i m } ( \cdot , \cdot )$ denotes similarity in the text space, $\beta$ is a learnable logit scale, and $\sigma ( \cdot )$ is the sigmoid function.

Query-Aware View Scoring and Selection. To rank views, we summarize each lifted volume with a target-aware descriptor

$$
\mathbf { d } _ { t } = \frac { \sum _ { \mathbf { v } \in \mathcal { V } } \big ( \mathbf { A } _ { t } ( \mathbf { v } ) + \eta \mathbf { m } _ { t } ( \mathbf { v } ) \big ) \mathbf { V } _ { t } ( \mathbf { v } ) } { \epsilon + \sum _ { \mathbf { v } \in \mathcal { V } } \big ( \mathbf { A } _ { t } ( \mathbf { v } ) + \eta \mathbf { m } _ { t } ( \mathbf { v } ) \big ) } , \qquad \mathbf { d } _ { t } \in \mathbb { R } ^ { C } ,\tag{5}
$$

where $\eta > 0$ provides a small valid-region fallback when the prior is sparse. Thus, $\mathbf { d } _ { t }$ compactly summarizes the target-related content of view �.

We additionally use two scalar reliability cues: a projectioncoverage cue $r _ { t } = \mathrm { M e a n } ( { \mathbf { m } _ { t } } )$ , and a focus cue $f _ { t } = \mathrm { T o p K M e a n } ( \mathbf { A } _ { t } \odot$ $\mathbf { m } _ { t } , K _ { f } )$ , where $K _ { f }$ is a small constant. Here $r _ { t }$ measures how much of the voxel grid is projectable into the current view, whereas $f _ { t }$ reflects whether the target evidence is concentrated rather than difuse. We also use a simple sequence-position cue $\pmb { \tau } _ { t } = \left[ \hat { p } _ { t } , 1 - \hat { p } _ { t } \right]$ where $p _ { t } = \left( t - 1 \right) / \left( T - 1 \right)$ ). These features are fed into a lightweight scorer:

$$
\ell _ { t } = g _ { \mathrm { v i e w } } ( [ \mathbf { d } _ { t } , \mathbf { s } _ { \mathrm { t g t } } , r _ { t } , f _ { t } , \tau _ { t } ] ) , \qquad w _ { t } = \left[ \mathrm { M a s k e d S o f t m a x } ( \{ \ell _ { i } \} _ { i = 1 } ^ { T } ) \right] _ { t } ,\tag{6}
$$

where $g _ { \mathrm { v i e w } }$ separately projects the target-aware view descriptor and the target slot into a shared hidden space, encodes the lowdimensional cues with a small MLP, and outputs a scalar relevance logit. The masked softmax is applied only over views with valid observations.

The resulting weights are also used to form a robust all-view baseline:

$$
\mathbf { V } _ { \mathrm { a v g } } ( \mathbf { v } ) = \frac { \sum _ { t = 1 } ^ { T } w _ { t } \mathbf { m } _ { t } ( \mathbf { v } ) \mathbf { V } _ { t } ( \mathbf { v } ) } { \epsilon + \sum _ { t = 1 } ^ { T } w _ { t } \mathbf { m } _ { t } ( \mathbf { v } ) } .\tag{7}
$$

Relevance-Ordered Conservative Memory Refinement. View scoring identifies promising observations, but weighted averaging alone can blur sharp target regions by mixing uneven evidence. To address this, we refine the sequence using a conservative memory updated in descending order of view relevance, rather than the standard chronological order. This ensures the most confident views establish a strong initial target state, while subsequent, less confident views only selectively fill in missing details without overwriting reliable evidence or deviating far from the robust all-view baseline.

Let $\boldsymbol { S } = \left( t _ { 1 } , t _ { 2 } , \dots , t _ { K } \right)$ denote the ordered sequence of indices for the top-� valid views, sorted in descending order by their relevance scores $\ell _ { t }$ . Within $s ,$ we renormalize the view weights as $\tilde { w } _ { t _ { k } } ~ =$ $w _ { t _ { k } } / \sum _ { i = 1 } ^ { K } w _ { t _ { i } }$ . The resulting consensus prior is then computed as:

$$
\bar { \mathbf { A } } = \sum _ { k = 1 } ^ { K } \tilde { w } _ { t _ { k } } \mathbf { A } _ { t _ { k } } .
$$

We initialize an empty memory volume, an accumulated observation mask, and a memory confidence map:

$$
{ \bf V } _ { \mathrm { m e m } } ^ { ( 0 ) } = { \bf 0 } , \qquad { \bf m } _ { \mathrm { o b s } } ^ { ( 0 ) } = { \bf 0 } , \qquad { \bf C } _ { \mathrm { m e m } } ^ { ( 0 ) } = { \bf 0 } .
$$

The memory is then updated recurrently for $k = 1 , \ldots , K$ . For the �-th update step, directly using the raw prior $\mathbf { A } _ { t _ { k } }$ as write support is risky, because broad low-confidence activations may overwrite the memory. We therefore derive a stricter write-support map:

$$
\begin{array} { r } { \mathbf { S } _ { t _ { k } } = \mathbf { m } _ { t _ { k } } \odot \mathbf { A } _ { t _ { k } } ^ { \gamma } \odot \big ( 0 . 5 + 0 . 5 \bar { \mathbf { A } } \big ) , } \end{array}
$$

where $\gamma \geq 1$ is a sharpening exponent and ${ \bf A } _ { t _ { k } } ^ { \gamma }$ denotes element-wise exponentiation. This operation suppresses medium/low responses and retains compact high-confidence regions, while the consensus term favors regions repeatedly supported across selected views.

A lightweight gating network then predicts a voxel-wise update gate:

$$
\mathbf { U } _ { t _ { k } } = \Phi _ { \mathrm { u p d } } ( G _ { t _ { k } } ) ,
$$

where $\mathcal { G } _ { t _ { k } }$ concatenates the current memory $\mathbf { V } _ { \mathrm { m e m } } ^ { ( k - 1 ) }$ and currentview features $\mathbf { V } _ { t _ { k } }$ , the target-support signals $\mathsf { S } _ { t _ { k } }$ , the accumulated support state $( \mathbf { m } _ { \mathrm { o b s } } ^ { ( k - 1 ) } , \mathbf { C } _ { \mathrm { m e m } } ^ { ( \bar { k } - 1 ) } )$ ), and simple update cues such as the update stage $k / K .$ . The update is designed to favor reliable and previously unsupported target evidence.

To keep ROVM focused on selection and conservative correction rather than full language reasoning, we form a candidate write representation by adding only a weak target-aligned bias inside the supported region:

$$
\mathbf { H } _ { t _ { k } } = \mathbf { V } _ { t _ { k } } + \rho _ { \mathrm { t x t } } \mathbf { S } _ { t _ { k } } \odot \mathrm { E x p a n d } ( \mathbf { W } _ { \mathrm { t v } } \mathbf { s } _ { \mathrm { t g t } } ) ,
$$

where $\mathbf { W } _ { \mathrm { t \downarrow } }$ projects the target slot to the voxel feature space, $\rho _ { \mathrm { t x t } }$ is a learnable scalar gate, and Expand(·) broadcasts a vector to the voxel grid. Because this residual is restricted by $\mathbf { S } _ { t _ { k } }$ and scaled by a small learnable gate, it only nudges voxels already supported as target evidence instead of rewriting the whole volume.

The memory states are then updated by:

$$
\mathbf { V } _ { \mathrm { m e m } } ^ { ( k ) } = \mathbf { V } _ { \mathrm { m e m } } ^ { ( k - 1 ) } + \mathbf { U } _ { t _ { k } } \odot \big ( \mathbf { H } _ { t _ { k } } - \mathbf { V } _ { \mathrm { m e m } } ^ { ( k - 1 ) } \big ) ,\tag{8}
$$

$$
\begin{array} { r } { \mathbf { m } _ { \mathrm { o b s } } ^ { ( k ) } = \operatorname* { m a x } ( \mathbf { m } _ { \mathrm { o b s } } ^ { ( k - 1 ) } , \mathbf { m } _ { t _ { k } } ) , } \end{array}\tag{9}
$$

$$
\mathbf { C } _ { \mathrm { m e m } } ^ { ( k ) } = \operatorname* { m a x } \Bigl ( \mathbf { C } _ { \mathrm { m e m } } ^ { ( k - 1 ) } , \mathbf { S } _ { t _ { k } } \odot \left( 0 . 5 + 0 . 5 \mathbf { U } _ { t _ { k } } \right) \Bigr ) .\tag{10}
$$

Here $\mathbf { V } _ { \mathrm { m e m } } ^ { ( k ) }$ stores the running refined memory, $\mathbf { n } _ { \mathrm { o b s } } ^ { ( k ) }$ records which voxels have been covered by selected views up to step $k ,$ and ${ \bf C } _ { \mathrm { m e n } } ^ { ( k ) }$ accumulates reliable target support for the final blending step.

After all � updates, we blend the refined memory with the robust all-view baseline:

$$
\mathbf { V } _ { r } = \mathbf { V } _ { \mathrm { a v g } } + \sigma ( \alpha _ { \mathrm { b l e n d } } ) \mathbf { C } _ { \mathrm { m e m } } ^ { ( K ) } \odot \big ( \mathbf { V } _ { \mathrm { m e m } } ^ { ( K ) } - \mathbf { V } _ { \mathrm { a v g } } \big ) ,
$$

where $\alpha _ { \mathrm { b l e n d } }$ is a learnable scalar. Consequently, confident target regions are sharpened by relevance-ordered memory refinement, while uncertain regions remain close to the robust all-view average.

We also keep the fused validity mask m<sub>�</sub> which will be used by the subsequent PLVF module. ROVM thus outputs a query-relevant but only coarsely language-conditioned 3D volume, leaving finer token-level disambiguation to PLVF.

## 4.4 Progressive Language–Voxel Fusion

ROVM efectively filters the observation sequence into a unified 3D volume with coarse spatial support. However, precise 3D visual grounding often hinges on fine-grained token-level cues, such as subtle visual attributes or complex target-anchor relations among nearby candidates. To bridge this gap, we introduce Progressive Language–Voxel Fusion (PLVF), a dedicated module that performs deep spatial-linguistic reasoning on the fused volume $( \mathbf { V } _ { r } , \mathbf { m } _ { r } )$ PLVF operates in a progressive two-stage manner: it first establishes a coarse target hypothesis using a condensed language slot, and subsequently revisits the full text tokens exclusively on the most promising voxels for fine-grained disambiguation.

Slot-Guided Coarse Target Hypothesis. Following the slot-based prior estimation formulation introduced in Sec. 4.3, but applied to the fused volume $\left( \mathbf { V } _ { r } , \mathbf { m } _ { r } \right)$ , PLVF derives its own fused target slot $\mathbf { s } _ { r } ,$ prior logits $\scriptstyle { \mathbf { Z } } _ { r } $ , and prior map $\mathbf { A } _ { r }$

To explicitly emphasize candidate target regions, we use the fused prior to gate a lightweight residual:

$$
\mathbf { V } _ { \mathrm { m i d } } = \mathbf { V } _ { r } + \sigma ( \gamma _ { 1 } ) \Phi _ { \mathrm { g a t e } } ( [ \mathbf { V } _ { r } , \mathbf { A } _ { r } , \mathbf { V } _ { r } \odot \mathbf { A } _ { r } ] ) ,\tag{11}
$$

where $\Phi _ { \mathrm { g a t e } }$ is a lightweight voxel fusion block and $\gamma _ { 1 }$ is a learnable scalar gate. This stage is computationally eficient, preserves the full 3D context, and supplies a coarse language-aware hypothesis for subsequent sparse reasoning.

Sparse Fine-Grained Token Refinement. The initial slot-guided stage relies on a compressed target slot $\mathbf { s } _ { r } ,$ which may miss finegrained linguistic distinctions. We therefore bring back the full token sequence E, but only for a sparse set of candidate voxels. Specifically, after masking invalid voxels by ${ \bf m } _ { r }$ , we select the top-� voxel indices $\mathcal { I } _ { \mathrm { t o p } }$ according to the fused prior logits $\scriptstyle { Z _ { r } } =$ , and gather their features:

$$
\begin{array} { r } { \mathbf { Q } _ { \mathrm { s p a r s e } } = \operatorname { G a t h e r } ( \mathbf { V } _ { \mathrm { m i d } } , \mathcal { I } _ { \mathrm { t o p } } ) , \qquad \mathbf { Q } _ { \mathrm { s p a r s e } } \in \mathbb { R } ^ { M \times C } . } \end{array}\tag{12}
$$

To preserve geometric structure, we project these voxel features to the text space and add 3D positional embeddings derived from normalized voxel coordinates:

$$
\widetilde { \bf Q } _ { \mathrm { s p a r s e } } = { \bf W } _ { q } { \bf Q } _ { \mathrm { s p a r s e } } + { \bf P } _ { \mathrm { p o s } } ( \mathcal { I } _ { \mathrm { t o p } } ) .\tag{13}
$$

We then perform multi-head cross-attention,

$$
\mathbf { F } _ { \mathrm { a t t n } } = \mathrm { M C A } ( Q = \widetilde { \mathbf { Q } } _ { \mathrm { s p a r s e } } , K = \mathbf { E } , V = \mathbf { E } ) ,\tag{14}
$$

so that only the most plausible target voxels re-examine the full text tokens. The attended features are mapped back to the voxel space and written to their original locations:

$$
\mathbf { V } _ { f } = \mathrm { S c a t t e r } \big ( \mathbf { V } _ { \mathrm { m i d } } , \mathbf { Q } _ { \mathrm { s p a r s e } } + \sigma ( \gamma _ { 2 } ) \mathbf { W } _ { o } \mathbf { F } _ { \mathrm { a t t n } } , { \mathcal { I } } _ { \mathrm { t o p } } \big ) ,\tag{15}
$$

where $\mathbf { W } _ { o }$ maps the attended text features back to the voxel feature space and �<sub>2</sub> is a learnable scalar gate. This progressive design keeps expensive token-level reasoning of the full voxel grid and focuses it on a much cleaner candidate set produced by ROVM and the coarse prior-guided hypothesis.

## 4.5 3D Grounding Head and Training Objectives

The final fused volume $\mathbf { V } _ { f }$ is processed by a 3D feature pyramid network (FPN) to produce multi-scale representations. We employ a lightweight, anchor-free detection head shared across scales.

Head Architecture. The head comprises two parallel branches: a score branch predicting a 3D confidence heatmap $\hat { \mathbf { Y } } \in [ 0 , 1 ] ^ { X \times Y \times Z }$ and a regression branch predicting per-voxel box ofsets representing the distances to the six faces of the target box.

Loss Functions. The network is trained end-to-end with a composite objective $\mathcal { L } = \mathcal { L } _ { \mathrm { s c o r e } } + \mathcal { L } _ { \mathrm { b b o x } } + \lambda _ { \mathrm { p r i o r } } \mathcal { L } _ { \mathrm { p r i o r } } + \lambda _ { \mathrm { v i e w } } \mathcal { L } _ { \mathrm { v i e w } } .$ Specifically, the score loss $( \mathcal { L } _ { \mathrm { s c o r e } } )$ applies Gaussian Focal Loss to supervise the predicted heatmap ${ \hat { \mathbf { Y } } } ,$ where the ground-truth heatmap is obtained by splatting a Gaussian kernel at the target center. The bounding box loss $( \mathcal { L } _ { \mathrm { b b o x } } )$ applies an IoU loss to predicted boxes at positive voxel locations. To encourage meaningful coarse local ization, the prior supervision $( \mathcal { L } _ { \mathrm { p r i o r } } )$ minimizes the binary crossentropy between the fused prior map A<sub>�</sub> and an auxiliary binary occupancy grid $\mathbf { O } _ { \mathrm { g t } } \in \{ 0 , 1 \} ^ { X \times Y \times Z }$ . Notably, $0 _ { \mathrm { g t } }$ is directly derived from the ground-truth 3D bounding box without requiring any additional signals. Finally, for observation-sequence grounding, the view-relevance supervision $( \mathcal { L } _ { \mathrm { v i e w } } )$ applies a weighted soft binary cross-entropy to the view logit $\ell _ { t }$ in Eq. (6). It is supervised by a soft target label $y _ { t } \in [ 0 , 1 ]$ , defined as the fraction of the ground-truth target volume covered by view �. We assign larger weights to views with larger target coverage, and this loss is applied exclusively in the observation-sequence setting.

## 5 Experiments

## 5.1 Experimental Settings

Datasets and Metrics. We evaluate our method on the proposed SeqAlign3DVG benchmark, reporting results on both the singleview and observation-sequence splits. Following standard protocols, we adopt Acc@0.25 and Acc@0.50 as our primary metrics, which measure the percentage of predicted bounding boxes having an Intersection-over-Union (IoU) with the ground truth strictly greater than 0.25 and 0.50, respectively.

Implementation Details. We implement our pipeline using the MMDetection3D framework. The 3D voxel grid covers a spatial range of6.4 m×6.4 m×2.56 m with a voxel resolution of0.16 m. The model is trained for 12 epochs using the AdamW optimizer with an initial learning rate of $5 \times 1 0 ^ { - 5 }$ , which is decayed by a factor of 10 at epochs 8 and 11. The loss weights for the auxiliary objectives are set to $\lambda _ { f r i o r } = 0 . 5$ and $\lambda _ { v i e w } = 0 . 0 5$ . We set $M = 5 1 2$ and $K = 5$ All experiments are conducted on 3 NVIDIA RTX 6000 Ada GPUs.

## 5.2 Baselines

For the observation-sequence split, SeqAlign3DVG evaluates strictly observation-aligned grounding from ordered posed-RGB inputs. Among the evaluated methods, BIP3D provides the closest endto-end posed-image comparison. However, the methods difer in depth input or supervision, target phrase-token supervision, and 3D detector priors, which are made explicit in Table 2.

Detector-based re-ranking. We use MVSDet [24], pretrained on the ScanNetV2 training split, to generate candidate 3D proposals and rank them by detector objectness (Top-Score), CLIP image–text similarity [20], or MiniLM-L6-v2 class–text similarity. This detector-agnostic pipeline could in principle be instantiated with other image-based 3D detectors, such as 3DGeoDet [28] and GVSynergy-Det [29].

Other grounding baselines. Grounding DINO + Back-project [17] and VLM-Grounder [23] use ground-truth depth maps at inference to recover 3D boxes; our VLM-Grounder adaptation uses GPT-5.4- mini. In our experiments, all three BIP3D [15] variants receive posed RGB images without depth-map input, while ground-truth depth supervision and detector initialization are independently varied.

## 5.3 Main Results

Tables 2 and 3 present results for the observation-sequence and single-view settings, respectively. Our method attains the best performance under the depth-free protocol in both settings; methods using ground-truth depth input or depth supervision are included with explicit protocol labels for transparency.

Observation-sequence Grounding. As shown in Table 2, our method achieves state-of-the-art performance under the depthfree protocol, reaching 51.30%/22.39% Acc@0.25/0.50. It outperforms BIP3D without depth supervision but with detector initialization (50.05%/20.12%) by 1.25/2.27 percentage points, while using neither target phrase-token supervision nor a pretrained 3D detector prior. Without detector initialization, BIP3D drops to 5.69%/0.24%, indicating a strong reliance on this prior on SeqAlign3DVG. VLM-Grounder obtains 47.28%/21.18% using groundtruth depth maps at inference, whereas BIP3D with depth supervision reaches 59.60%/24.55% without depth-map input; both settings lie outside the depth-free protocol. Grounding DINO + Backproject likewise uses ground-truth depth input but reaches only 26.61%/5.62%, indicating that independent frame-wise grounding followed by direct back-projection remains insuficient for accurate sequence-level 3D localization.

Table 2: Observation-sequence Performance. Protocol-aware comparison on the SeqAlign3DVG observation-sequence split.
<table><tr><td>Method</td><td>GT Depth Input</td><td>GT Depth Sup.</td><td>Phrase Sup.</td><td>3D Det. Prior</td><td>Acc@0.25</td><td>Acc@0.50</td></tr><tr><td>Detector + Top-Score</td><td>x</td><td>x</td><td>x</td><td>√</td><td>21.26</td><td>14.30</td></tr><tr><td>Detector + CLIP</td><td>x</td><td>x</td><td>x</td><td>√</td><td>31.86</td><td>18.79</td></tr><tr><td>Detector + MiniLM-L6-v2</td><td>x</td><td>x</td><td>x</td><td>√</td><td>30.01</td><td>17.49</td></tr><tr><td>Grounding DINO + Back-project</td><td>√</td><td>x</td><td>x</td><td>x</td><td>26.61</td><td>5.62</td></tr><tr><td>VLM-Grounder*</td><td>√</td><td>x</td><td>x</td><td>x</td><td>47.28</td><td>21.18</td></tr><tr><td>BIP3D (w/o Depth Sup. &amp; Det. Init.)</td><td>x</td><td>x</td><td>√</td><td>x</td><td>5.69</td><td>0.24</td></tr><tr><td>BIP3D (w/o Depth Sup.)</td><td>x</td><td>x</td><td>√</td><td>√</td><td>50.05</td><td>20.12</td></tr><tr><td>BIP3D (w/ Depth Sup.)</td><td>x</td><td>√</td><td>√</td><td>√</td><td>59.60</td><td>24.55</td></tr><tr><td>Ours</td><td>x</td><td>x</td><td>x</td><td>x</td><td>51.30</td><td>22.39</td></tr></table>

GT Depth Input: ground-truth depth maps used at inference. GT Depth Sup.: ground-truth depth used as training supervision without depth-map input at inference. Phrase Sup.: target phrase-token supervision. 3D Det. Prior: proposals from a pretrained 3D detector or initialization from a pretrained 3D detection checkpoint. <sup>∗</sup> Our VLM-Grounder adaptation uses GPT-5.4-mini. Bold denotes the best result under the depth-free protocol, i.e., without ground-truth depth input or depth supervision.

Table 3: Single-view Performance. Overall comparison on the SeqAlign3DVG single-view split.
<table><tr><td>Method</td><td></td><td>Acc@0.25 Acc@0.50</td></tr><tr><td>Detector + Top-Score</td><td>22.97</td><td>9.87</td></tr><tr><td>Detector + CLIP</td><td>29.44</td><td>11.28</td></tr><tr><td>Detector + MiniLM-L6-v2</td><td>27.73</td><td>10.69</td></tr><tr><td>Grounding DINO + Back-project†</td><td>38.25</td><td>8.52</td></tr><tr><td>Ours</td><td>44.54</td><td>16.86</td></tr></table>

<sup>†</sup> Uses ground-truth depth maps at inference.

Single-view Grounding. Table 3 reports the overall single-view comparison. Without ground-truth depth input or depth supervision, our method achieves 44.54% Acc@0.25 and 16.86% Acc@0.50. Grounding DINO + Back-project uses ground-truth depth maps at inference but reaches 38.25%/8.52%, indicating that direct 2D grounding and depth-based back-projection remain insuficient for precise 3D localization.

## 5.4 Ablation Study

To validate that the architecture is supported by controlled design choices rather than a generic stack of layers, we conduct incremental component ablations in Table 4. The Baseline removes both ROVM and PLVF.

Efectiveness of PLVF. Integrating PLVF improves Acc@0.25 by 6.29 points on single-view and 4.63 points on sequence grounding, confirming the value of progressive coarse-to-fine voxel–token reasoning. Compared with dense full-grid cross-attention, PLVF reduces single-view computation by 35.7% (158.69 to 101.99 GFLOPs), while lowering latency from 63 to 59 ms and peak memory from 3026 to 2810 MB.

Table 4: Component Ablation. Incremental contributions of PLVF and ROVM.
<table><tr><td rowspan="2">Model</td><td colspan="2">Single-view</td><td colspan="2">Sequence</td></tr><tr><td>Acc@0.25</td><td>Acc@0.50</td><td>Acc@0.25</td><td>Acc@0.50</td></tr><tr><td>Baseline</td><td>38.25</td><td>13.40</td><td>41.46</td><td>17.73</td></tr><tr><td>+ PLVF</td><td>44.54</td><td>16.86</td><td>46.09</td><td>20.40</td></tr><tr><td>+ ROVM (Full)</td><td>一</td><td></td><td>51.30</td><td>22.39</td></tr></table>

Efectiveness of ROVM. Adding ROVM raises sequence performance from 46.09%/20.40% to 51.30%/22.39%. In an additional write-order comparison, relevance-ordered writing outperforms the strongest alternative, reverse writing (50.34%/21.87%), by 0.96/0.52 points, supporting the deliberate ordering and conservative memory design.

## 6 Conclusion

We introduce SeqAlign3DVG, a strictly observation-aligned indoor benchmark for 3D grounding from single views and temporally ordered observation sequences. We further propose a unified voxelbased framework with ROVM for query-aware sequence aggregation and PLVF for eficient coarse-to-fine language–voxel reasoning. Extensive experiments demonstrate state-of-the-art performance under the depth-free protocol.

## Acknowledgments

The research work described in this paper was conducted in the JC STEM Lab of Machine Learning and Computer Vision funded by The Hong Kong Jockey Club Charities Trust. This research was partially supported by a grant from the Research Grant Council, Hong Kong Special Administrative Region, China, under Project PolyU 15215824 and received partially support from the Global STEM

Professorship Scheme from the Hong Kong Special Administrative Region.

## References

[1] Ahmed Abdelreheem, Kyle Olszewski, Hsin-Ying Lee, Peter Wonka, and Panos Achlioptas. 2024. ScanEnts3D: Exploiting Phrase-to-3D-Object Correspondences for Improved Visio-Linguistic Models in 3D Scenes. In 2024 IEEE/CVF Winter Conference on Applications ofComputer Vision (WACV). 3512–3522. doi:10.1109/ WACV57701.2024.00349

[2] Panos Achlioptas, Ahmed Abdelreheem, F. Xia, Mohamed Elhoseiny, and Leonidas J. Guibas. 2020. ReferIt3D: Neural Listeners for Fine-Grained 3D Object Identification in Real-World Scenes. In European Conference on Computer Vision. 422–440. doi:10.1007/978-3-030-58452-8\_2

[3] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. 2025. Qwen3-VL Technical Report. arXiv:2511.21631 [cs.CV] https://arxiv.org/abs/2511.21631

[4] Dave Zhenyu Chen, Angel X Chang, and Matthias Nießner. 2020. ScanRefer: 3D Object Localization in RGB-D Scans Using Natural Language. In European Conference on Computer Vision. Springer, 202–221. doi:10.1007/978-3-030-58565- 5\_13

[5] Shizhe Chen, Pierre-Louis Guhur, Makarand Tapaswi, Cordelia Schmid, and Ivan Laptev. 2022. Language conditioned spatial relation reasoning for 3D ob ject grounding. In Proceedings of the 36th International Conference on Neural Information Processing Systems (NIPS ’22). Curran Associates Inc., Article 1492, 14 pages.

[6] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. 2017. ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 5828–5839.

[7] Dailan He, Yusheng Zhao, Junyu Luo, Tianrui Hui, Shaofei Huang, Aixi Zhang, and Si Liu. 2021. TransRefer3D: Entity-and-Relation Aware Transformer for Fine-Grained 3D Visual Grounding. Proceedings ofthe 29th ACM International Conference on Multimedia (2021), 2344–2352. doi:10.1145/3474085.3475397

[8] Shijia Huang, Yilun Chen, Jiaya Jia, and Liwei Wang. 2022. Multi-View Trans former for 3D Visual Grounding. In 2022 IEEE/CVFConference on Computer Vision and Pattern Recognition (CVPR). 15503–15512. doi:10.1109/CVPR52688.2022.01508

[9] Wenyuan Huang, Zhao Wang, Zhou Wei, Ting Huang, Fang Zhao, Jian Yang, and Zhenyu Zhang. 2025. OpenGround: Active Cognition-based Reasoning for Open-World 3D Visual Grounding. arXiv:2512.23020 [cs.CV] https://arxiv.org/ abs/2512.23020

[10] Ayush Jain, Nikolaos Gkanatsios, Ishita Mediratta, and Katerina Fragkiadaki. 2022. Bottom Up Top Down Detection Transformers for Language Grounding in Images and Point Clouds. In European Conference on Computer Vision (Tel Aviv, Israel). Springer-Verlag, Berlin, Heidelberg, 417–433. doi:10.1007/978-3-031-20059-5\_24

[11] Shunya Kato, Shuhei Kurita, Chenhui Chu, and Sadao Kurohashi. 2023. ARKitSceneRefer: Text-based Localization of Small Objects in Diverse Real-World 3D Indoor Scenes. In Findings ofthe Association for Computational Linguistics: EMNLP 2023. 784–799. doi:10.18653/v1/2023.findings-emnlp.56

[12] Fuhao Li, Huan Jin, Bin Gao, Liaoyuan Fan, Lihui Jiang, and Long Zeng. 2025. NuGrounding: A Multi-View 3D Visual Grounding Framework in Autonomous Driving. arXiv:2503.22436 [cs.CV] https://arxiv.org/abs/2503.22436

[13] Rong Li, Shijie Li, Lingdong Kong, Xulei Yang, andJunwei Liang. 2025. SeeGround: See and Ground for Zero-Shot Open-Vocabulary 3D Visual Grounding. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 3707– 3717. doi:10.1109/CVPR52734.2025.00351

[14] Yuzhen Li, Min Liu, Zhaoyang Li, Yuan Bian, Xueping Wang, Erbo Zhai, and Yaonan Wang. 2026. Mono3DVG-EnSD: Enhanced Spatial-aware and Dimensiondecoupled Text Encoding for Monocular 3D Visual Grounding. Proceedings ofthe AAAI Conference on Artificial Intelligence 40, 8 (2026), 6726–6734. doi:10.1609/ aaai.v40i8.37604

[15] Xuewu Lin, Tianwei Lin, Lichao Huang, Hongyu Xie, and Zhizhong Su. 2025. BIP3D: Bridging 2D Images and 3D Perception for Embodied Intelligence. In

2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 9007–9016. doi:10.1109/CVPR52734.2025.00842

[16] Haolin Liu, Anran Lin, Xiaoguang Han, Lei Yang, Yizhou Yu, and Shuguang Cui. 2021. Refer-it-in-RGBD: A Bottom-up Approach for 3D Visual Grounding in RGBD Images. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 6028–6037. doi:10.1109/CVPR46437.2021.00597

[17] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, and Lei Zhang. 2024. Grounding DINO: Marrying DINO with Grounded Pre-training for Open-Set Object Detection. In European Conference on Computer Vision. 38–55. doi:10.1007/ 978-3-031-72970-6\_3

[18] Junyu Luo, Jiahui Fu, Xianghao Kong, Chen Gao, Haibing Ren, Hao Shen, Huaxia Xia, and Si Liu. 2022. 3D-SPS: Single-Stage 3D Visual Grounding via Referred Point Progressive Selection. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 16433–16442. doi:10.1109/CVPR52688.2022.01596

[19] Ruiyuan Lyu, Jingli Lin, Tai Wang, Shuai Yang, Xiaohan Mao, Yilun Chen, Runsen Xu, Haifeng Huang, Chenming Zhu, Dahua Lin, and Jiangmiao Pang. 2024. MMScan: a multi-modal 3D scene dataset with hierarchical grounded language annotations. In Proceedings ofthe 38th International Conference on Neural Information Processing Systems (Vancouver, BC, Canada) (NIPS ’24). Curran Associates Inc., Article 1611, 27 pages. doi:10.52202/079017-1611

[20] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. Learning Transferable Visual Models From Natural Language Supervision. In International Conference on Machine Learning, Vol. 139. PMLR, 8748–8763.

[21] Austin Wang, ZeMing Gong, and Angel X Chang. 2025. ViGiL3D: A Linguistically Diverse Dataset for 3D Visual Grounding. In Proceedings ofthe 63rd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (Eds.). Association for Computational Linguistics, Vienna, Austria, 30453–30475. doi:10.18653/v1/2025.acl-long.1470

[22] Tai Wang, Xiaohan Mao, Chenming Zhu, Runsen Xu, Ruiyuan Lyu, Peisen Li, Xiao Chen, Wenwei Zhang, Kai Chen, Tianfan Xue, Xihui Liu, Cewu Lu, Dahua Lin, and Jiangmiao Pang. 2024. EmbodiedScan: A Holistic Multi-Modal 3D Perception Suite Towards Embodied AI. In 2024 IEEE/CVFConference on Computer Vision and Pattern Recognition (CVPR). 19757–19767. doi:10.1109/CVPR52733.2024.01868

[23] Runsen Xu, Zhiwei Huang, Tai Wang, Yilun Chen, Jiangmiao Pang, and Dahua Lin. 2025. VLM-Grounder: A VLM Agent for Zero-Shot 3D Visual Grounding. In Proceedings ofThe 8th Conference on Robot Learning (Proceedings ofMachine Learning Research, Vol. 270). PMLR, 3961–3985.

[24] Yating Xu, Chen Li, and Gim Hee Lee. 2024. MVSDet: multi-view indoor 3D object detection via eficient plane sweeps. In Proceedings ofthe 38th International Conference on Neural Information Processing Systems (NIPS ’24). Curran Associates Inc., Article 4222, 19 pages. doi:10.52202/079017-4222

[25] Zhihao Yuan, Xu Yan, Yinghong Liao, Ruimao Zhang, Sheng Wang, Zhen Li, and Shuguang Cui. 2021. InstanceRefer: Cooperative Holistic Understanding for Visual Grounding on Point Clouds through Instance Multi-level Contextual Referring. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV). 1771–1780. doi:10.1109/ICCV48922.2021.00181

[26] Yang Zhan, Yuan Yuan, and Zhitong Xiong. 2024. Mono3DVG: 3D Visual Ground ing in Monocular Images. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 38. 6988–6996. doi:10.1609/aaai.v38i7.28525

[27] Yiming Zhang, ZeMing Gong, and Angel X. Chang. 2023. Multi3DRefer: Grounding Text Description to Multiple 3D Objects. 2023 IEEE/CVF International Conference on Computer Vision (ICCV) (2023), 15225–15236. doi:10.1109/ICCV51070. 2023.01397

[28] Yi Zhang, Yi Wang, Yawen Cui, and Lap-Pui Chau. 2025. 3DGeoDet: General-Purpose Geometry-Aware Image-Based 3D Object Detection. IEEE Transactions on Multimedia 27 (2025), 6235–6247. doi:10.1109/TMM.2025.3581780

[29] Yi Zhang, Yi Wang, Lei Yao, and Lap-Pui Chau. 2025. GVSynergy-Det: Synergistic Gaussian-Voxel Representations for Multi-View 3D Object Detection. arXiv:2512.23176 [cs.CV] https://arxiv.org/abs/2512.23176

[30] Yani Zhang, Dongming Wu, Hao Shi, Yingfei Liu, Tiancai Wang, and Xingping Dong. 2026. DEGround: An Efective Baseline for Ego-centric 3D Visual Grounding with a Homogeneous Framework. arXiv:2506.05199 [cs.CV] https://arxiv.org/abs/2506.05199

[31] Henry Zheng, Hao Shi, Qihang Peng, Yong Xien Chng, Rui Huang, Yepeng Weng, Zhongchao Shi, and Gao Huang. 2025. DenseGrounding: Improving Dense Language-Vision Semantics for Ego-centric 3D Visual Grounding. In The Thirteenth International Conference on Learning Representations.