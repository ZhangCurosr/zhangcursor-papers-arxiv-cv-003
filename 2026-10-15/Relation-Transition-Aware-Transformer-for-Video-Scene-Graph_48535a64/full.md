# Relation-Transition-Aware Transformer for Video Scene Graph Generation

Jaeyeon Heo , Geuntaek Lim , and Yukyung Choi<sup>†</sup>

Sejong University, Seoul, Republic of Korea {jyheo,gtlim,ykchoi}@rcv.sejong.ac.kr

Abstract. Video scene graph generation (VidSGG) aims to capture objects and their evolving relationships in videos. This requires accounting for both the persistence of object identities and abrupt changes in relationships. Existing methods aggregate information across frames to exploit temporal context, but an emphasis on consistency can obscure these changes. As a result, a model may retain its previous prediction even after the underlying relation has changed. We refer to this failure mode as relational semantic inertia. To address this issue, we propose Relation-Transition-Aware Transformer (RTAformer), a one-stage framework that separately models entity persistence and relation changes. It injects calibrated diferences between adjacent-frame relation queries to guide predicate updates, while cross-frame instance alignment during training helps reduce cross-frame representational drift. Experiments on Action Genome show improvements in recall and mean recall over existing methods, with ablation studies supporting the contributions of the proposed components and comparisons using matched backbones showing larger gains on relation-transition frames than on non-transition frames. Code is available at https://github.com/sejong-rcv/RTAformer.

Keywords: Scene Graph Generation · Spatio-Temporal Modeling · Video Understanding

## 1 Introduction

Video scene graph generation (VidSGG) represents objects and their relationships in videos as structured ⟨subject, predicate, object⟩ triplets, with entities as nodes and relationships as edges. By explicitly representing who interacts with what and how these relationships evolve, video scene graphs provide structured information for action selection in embodied navigation and planning [9, 24, 34] and for interpreting events in video-language reasoning and question answering [3, 21, 27]. Approaches to VidSGG have evolved from multi-stage pipelines for object detection, temporal association, and relation classification [4,14,20] to one-stage frameworks that integrate instance and relation prediction, reducing reliance on external tracking [12, 29, 31].

![](images/d3a50a3e3825868f78a5897c7dc330799c6313ed9b2969b54c346f1c90d4ab18.jpg)  
(a) Relational Semantic Inertia

![](images/4128cb5af665cb343940255e4be60ebf77b03f9802fad4bafe5bb15c16b4f4d8.jpg)  
(b) Relation-Transition Challenge  
Fig. 1: (a) Relational semantic inertia: the ‘person–cup/glass/bottle’ pair persists across frames, but its relation changes from looking at to not looking at, while the baseline [29] retains the previous predicate in the target frame. (b) Our reimplementation of the baseline exhibits an 11.6-point gap in SGDET R@10 between the non-transition and relation-transition frame subsets.

Despite these advances, capturing abrupt relation changes remains challenging when the same entities persist across frames. While existing approaches aggregate information across frames to improve predictions [4, 8, 29, 31, 38], explicitly modeling relation changes for persistent entity pairs has received comparatively less attention. In such cases, a model may retain its previous prediction even after the underlying relation has changed. Figure 1(a) illustrates this behavior: the same subject–object pair remains visible, but its relation changes from looking at to not looking at, while a baseline model [29] continues to predict looking at. Our reimplementation of the baseline [29] achieves an R@10 of 34.3 on the relation-transition frame subset, compared with 45.9 on the nontransition frame subset under the with constraint setting (Fig. 1(b)). We refer to this failure mode as relational semantic inertia.

These observations motivate temporal modeling that preserves stable entity information while explicitly accounting for changes in relationships. To this end, we propose Relation-Transition-Aware Transformer (RTAformer), a one-stage VidSGG framework that uses diferences between adjacent-frame relation queries as explicit cues for predicate updates. RTAformer refines instance and relation queries in separate temporal streams to aggregate entity information and incorporate relation changes into predicate prediction. In the relation stream, adjacent-frame query diferences are calibrated and injected before temporal aggregation. These diferences are more informative when the queries remain grounded to consistent entity pairs across frames. We therefore regularize adjacent-frame instance-pair queries during training to encourage consistent grounding and help reduce representational drift in the transition cues. Our main contributions are as follows:

1. We characterize relational semantic inertia, a failure mode in VidSGG where models retain previous predicate predictions despite relation changes between persistent entities.

2. We propose RTAformer, a one-stage framework that separates entity and relation temporal refinement and uses calibrated relation-query diferences as explicit cues for predicate updates, supported by cross-frame instance alignment that encourages consistent query grounding.

3. Experiments on Action Genome [11] demonstrate improved scene graph generation performance over state-of-the-art methods, with a backbone-controlled comparison against OED [29] showing larger gains on the relation-transition frame subset than on the non-transition frame subset.

## 2 Related Work

Video Scene Graph Generation. Video scene graph generation (VidSGG) extends image-based scene graph generation [5,10,13,25,33,35] by modeling objects and their relationships across video frames. Earlier methods typically separate object detection from subsequent temporal and relational reasoning [4, 26, 32], whereas recent one-stage frameworks integrate instance and relation prediction [12, 29, 31]. OED [29] and ARN [31] use learnable queries to predict instances and relations from visual features, while UNO [12] adopts an objectcentric slot-attention framework. Our work builds on the one-stage formulation by introducing explicit relation-change cues into temporal refinement.

Temporal Aggregation and Relation Transitions. Temporal aggregation uses cross-frame context to refine instance and relation predictions. STTran [4] captures inter-frame dependencies with a temporal transformer, APT [14] introduces anticipatory learning objectives, and DSG-DETR [8] uses object tracklets to exploit long-term dependencies. Within one-stage frameworks, OED [29] aggregates temporal information through cross-attention over instance and relation queries, while ARN [31] combines temporal feature modeling with entitypredicate association reasoning. Complementary eforts address prediction reliability, including memory-guided debiasing in TEMPURA [20] and selective spatio-temporal denoising in $\mathrm { T D ^ { 2 } \mathrm { - N e t } }$ [15].

More directly related to our motivation, $\mathrm { T R ^ { 2 } }$ [30] addresses time-varying relations through text-guided supervision of inter-frame relation-feature diferences. It also uses a short-term message token to modulate the contribution of preceding-frame features during relation prediction. RTAformer shares the motivation of capturing relation changes but uses adjacent-frame diferences as a direct input to relation-query refinement. Specifically, it calibrates and injects these diferences before temporal aggregation within a one-stage framework, while separate instance and relation streams accommodate entity persistence and relation changes.

Query-Based Temporal Modeling. DETR [1] formulates object detection as set prediction using learnable object queries, while subsequent methods incorporate explicit positional priors into query design [16,37]. For video understanding, queries also provide a means of incorporating temporal information. FAQ [6] aggregates neighboring-frame evidence into queries for video object detection, while TrackFormer [18] and MOTR [36] propagate query states to maintain object tracks. In VidSGG, TPT [38] uses temporal propagation to connect predictions across frames. RTAformer focuses on diferences between adjacent-frame relation queries as cues for predicate updates. Because these diferences can also reflect inconsistent entity-pair grounding, training-time instance alignment encourages cross-frame consistency to help reduce representational drift in the transition cues.

![](images/5cb0d8ededa414d7f4397316ab10e287d215b86c33ec69233daf93da1666c654.jpg)  
Fig. 2: Overall framework of RTAformer. (a) Frame-wise spatial triplet parsing produces instance-pair and relation queries for target and reference frames. (b) Disentangled Temporal Aggregation (DTA) refines the target queries with separate instance and relation streams, where TADI injects relation-transition cues. FFN heads produce the instance and relation predictions that form the output triplets. (c) Spatio-Semantic Instance Alignment (SSIA) aligns adjacent-frame instance-pair queries during training.

## 3 Method

## 3.1 Problem Formulation

Given a video sequence $\mathcal { V } = \{ I _ { t } \} _ { t = 1 } ^ { T }$ consisting of T temporally ordered frames, video scene graph generation (VidSGG) aims to construct a dynamic spatiotemporal representation ${ \mathcal G } = \{ { \mathcal G } _ { t } \} _ { t = 1 } ^ { T }$ that captures both the presence of physical entities and their evolving interactions. Formally, each frame-wise graph is defined as $\mathcal { G } _ { t } ~ = ~ ( \mathcal { O } _ { t } , \mathcal { R } _ { t } )$ , where $\mathcal { O } _ { t } ~ = ~ \{ o _ { t , i } \} _ { i = 1 } ^ { N _ { t } }$ is the set of detected objects, each characterized by a spatial bounding box $\bar { b } _ { t , i } \in \mathbb { R } ^ { 4 }$ and a categorical labe $c _ { t , i } \in \mathcal { V } _ { o b j }$ , where $\mathcal { \mathrm { V } } _ { o b j }$ denotes the set of object categories. The relation set $\mathcal { R } _ { t }$ comprises interaction triplets $\left. o _ { s } , p , o _ { o } \right.$ , where $o _ { s } , o _ { o } \in \mathcal { O } _ { t }$ are the subject and object, and $p \in \{ 0 , 1 \} ^ { | { \mathcal { C } } _ { p r e d } | }$ is a multi-label predicate vector, where $\mathcal { C } _ { p r e d }$ denotes the set of predicate categories. The objective is to learn a mapping $\nu \mapsto \mathcal G$ that maximizes the posterior $P ( \mathcal G \mid \mathcal V )$ .

## 3.2 Framework Overview

As illustrated in Fig. 2, RTAformer is a one-stage set-prediction framework that models the distinct temporal behaviors of entities and predicates in VidSGG. The framework consists of three key components.

(1) Frame-Wise Spatial Triplet Parsing. Given a target frame $I _ { t a r }$ and $N _ { r e f }$ reference frames ${ \{ I _ { r e f } ^ { 1 } , . . . , I _ { r e f } ^ { N _ { r e f } } \} }$ , a vision backbone [7, 23] followed by a Transformer encoder [28] produces encoded visual features for each frame. These features are processed by two cascaded decoders, the Instance-Pair Decoder and the Relation Context Decoder, which transform $N _ { q }$ learnable triplet queries Q into frame-wise instance-pair queries $Q ^ { i n s }$ and relation queries $Q ^ { r e l }$

(2) Disentangled Temporal Aggregation (DTA). To mitigate relational semantic inertia, DTA refines the target queries $Q _ { t a r } ^ { i n s }$ and $Q _ { t a r } ^ { r e l }$ in separate instance and relation streams. The instance stream aggregates entity information across frames, while the relation stream applies Transition-Aware Delta Injection (TADI) to inject inter-frame relation-change cues into predicate decoding.

(3) Spatio-Semantic Instance Alignment (SSIA). During training, SSIA applies contrastive regularization to adjacent-frame instance-pair queries to improve cross-frame query consistency and reduce representational drift in the displacements used by TADI.

## 3.3 Frame-Wise Spatial Triplet Parsing

Following the DETR-like set prediction paradigm [1], we associate $N _ { q }$ learnable triplet queries $Q \in \mathbb { R } ^ { N _ { q } \times D }$ with potential subject–object interactions, where D denotes the hidden dimension. Unlike standard object detection queries that represent individual entities, each triplet query represents a candidate subject–object interaction. The cascaded decoders first produce instance-pair representations and then condition relation decoding on them. For notational simplicity, we describe the parsing process for a generic frame $I _ { t } .$ . The same spatial parser applies identically whether t denotes the target frame (t=tar) or a reference frame.

For each frame $I _ { t } ,$ a vision backbone (e.g., ViT [7, 23]) and Transformer encoder [28] produce encoded visual features $F _ { t }$ . The Instance-Pair Decoder $\left( \mathrm { D e c o d e r } _ { i n s } \right)$ first transforms the initial queries $Q$ into instance-pair representations by aggregating spatial context from $F _ { t }$ . Since relational semantics are inherently conditioned on the identity and spatial configuration of the underlying subject–object pair, the Relation Context Decoder $\left( \mathrm { D e c o d e r } _ { r e l } \right)$ subsequently takes $Q _ { t } ^ { i n s }$ as its query input to produce relation-aware representations:

$$
\begin{array} { r l } & { Q _ { t } ^ { i n s } = \mathrm { D e c o d e r } _ { i n s } ( Q , \ F _ { t } ) , } \\ & { Q _ { t } ^ { r e l } = \mathrm { D e c o d e r } _ { r e l } ( Q _ { t } ^ { i n s } , \ F _ { t } ) . } \end{array}\tag{1}
$$

Each decoder layer follows the standard DETR decoder design [1], using selfattention to exchange contextual information among triplet queries and crossattention to retrieve spatial evidence from $F _ { t } \left[ 1 , 1 6 , 1 9 \right]$ , followed by a feed-forward network.

The resulting $Q _ { t } ^ { i n s }$ and $Q _ { t } ^ { r e l }$ encode frame-level instance and relation contexts, respectively, and serve as inputs to the Disentangled Temporal Aggregation module described in ${ \mathrm { S e c . ~ 3 . 4 } }$

## 3.4 Disentangled Temporal Aggregation

Entity identities tend to persist across frames, whereas predicates can change abruptly for the same subject–object pair. Disentangled Temporal Aggregation (DTA) therefore processes instance and relation queries in separate temporal streams. The instance stream aggregates persistent entity evidence, while the relation stream incorporates relation-transition cues before temporal aggregation.

Given $N _ { r e f }$ reference frames, we apply the spatial parser to each and concatenate the resulting instance and relation queries into reference query sets $Q _ { r e f } ^ { i n s } , Q _ { r e f } ^ { r e l } \in \mathbb { R } ^ { N _ { r e f } \tilde { N _ { q } } \times D }$ , where the superscript denotes the stream. The preceding frame in the input sequence, indexed by tar−1 and included in the reference set, provides $Q _ { p r e v } ^ { r e l } : = Q _ { t a r - 1 } ^ { r e l }$ and $Q _ { p r e v } ^ { i n s } : = Q _ { t a r - 1 } ^ { i n s }$ for computing the relation-query displacement and confidence gate in TADI, respectively.

Each stream contains $N _ { d } { = } 3$ Temporal Aggregation Decoders. At stage $d ,$ the decoder applies self-attention to the target queries, cross-attention to the selected reference queries, and a feed-forward network [28]. The two streams use independently scored reference query sets:

$$
\begin{array} { r } { Q _ { t a r } ^ { u , ( d ) } = \mathrm { D e c o d e r } _ { t e m p } ^ { u , ( d ) } \Big ( Q _ { t a r } ^ { u , ( d - 1 ) } , ~ Q _ { r e f } ^ { u , ( d ) } \Big ) , \quad u \in \{ i n s , r e l \} , } \end{array}\tag{2}
$$

where $Q _ { r e f } ^ { u , ( d ) } = \mathrm { T o p K } ( Q _ { r e f } ^ { u } , K _ { d } )$ selects the $\mathrm { t o p } { - } K _ { d }$ reference queries according to the stream-specific scores defined below. We set $K _ { d } = \alpha _ { d } N _ { r e f } N _ { q }$ with $( \alpha _ { 1 } , \alpha _ { 2 } , \alpha _ { 3 } ) = ( 0 . 8 , 0 . 5 , 0 . 3 )$ , progressively retaining the highest-scoring temporal candidates. The instance stream is initialized with $Q _ { t a r } ^ { i n s , ( 0 ) } = Q _ { t a r } ^ { i n s }$ , while the relation stream is initialized with $Q _ { t a r } ^ { r e l , ( 0 ) } = \widehat { Q } _ { t a r } ^ { r e l }$ , where $\widehat { Q } _ { t a r } ^ { r e l }$ denotes the TADI-injected queries defined in Eq. (7).

Instance stream. The instance stream prioritizes high-confidence reference queries to aggregate persistent entity information. Each of the $N _ { r e f } N _ { q }$ pooled reference instance queries, indexed by $j ,$ is scored via the object classification head $\phi _ { o b j }$ as:

$$
s _ { j } ^ { i n s } = \log ( P _ { j } ^ { o b j } + \epsilon ) , \quad P _ { j } ^ { o b j } = \operatorname* { m a x } _ { c \in \mathcal { V } _ { o b j } } \mathrm { s o f t m a x } ( \phi _ { o b j } ( Q _ { r e f , j } ^ { i n s } ) ) _ { c } ,\tag{3}
$$

where ϵ is a small constant for numerical stability. The top- $K _ { d }$ candidates form $Q _ { r e f } ^ { i n s , ( d ) }$ . The final instance queries $Q _ { t a r } ^ { i n s , * } = Q _ { t a r } ^ { i n \bar { s } , ( N _ { d } ) }$ are obtained via $\operatorname { E q . } \ ( 2 )$ Relation stream with TADI. The relation stream applies Transition-Aware Delta Injection (TADI) to the target relation queries before temporal aggregation. TADI uses diferences between same-index relation queries across adjacent frames as candidate cues for predicate updates. As illustrated in Fig. 3, for each query index $q ,$ the displacement is computed as:

$$
\Delta _ { q } = Q _ { t a r , q } ^ { r e l } - Q _ { p r e v , q } ^ { r e l } ,\tag{4}
$$

![](images/8f50b6ef3ef35fcafd4de796623e0a75bdbd7de3c774abe0f550cf7995be12f3.jpg)  
Fig. 3: Transition-Aware Delta Injection (TADI). TADI uses the adjacent-frame relation-query displacement $\Delta _ { q } = Q _ { t a r , q } ^ { r e l } - Q _ { p r e v , q } ^ { r e l }$ as a transition cue. The displacement is decomposed into its numerically stabilized root-mean-square (RMS) magnitude $r _ { q }$ and RMS-normalized direction $\Delta _ { q } ^ { d i r }$ , while the scale gate $g _ { q }$ and foreground confidence $c _ { q }$ modulate the injection strength.

where $Q _ { t a r } ^ { r e l }$ denotes the target-frame relation queries from Eq. (1). To support these same-index comparisons, SSIA encourages cross-frame consistency of the underlying instance-pair representations during spatial training (Sec. 3.5). Rather than injecting raw $\Delta _ { q } ,$ TADI separates its magnitude and direction and computes a magnitude-dependent scale gate:

$$
\begin{array} { c } { r _ { q } = \sqrt { \frac { 1 } { D } | | \Delta _ { q } | | _ { 2 } ^ { 2 } + \epsilon } , } \\ { \Delta _ { q } ^ { d i r } = \Delta _ { q } / r _ { q } , } \\ { g _ { q } = 1 + \operatorname { t a n h } ( f _ { g } ( \log ( 1 + r _ { q } ) ) ) , } \end{array}\tag{5}
$$

where $r _ { q }$ is the numerically stabilized root-mean-square (RMS) magnitude of $\Delta _ { q } ,$ $\Delta _ { q } ^ { d i r }$ is the corresponding RMS-normalized direction, and $g _ { q }$ is a bounded scale gate predicted by a two-layer MLP $f _ { g }$

We modulate the update using foreground object confidence from both frames, reducing its contribution when either query has low confidence:

$$
c _ { q } = \sqrt { P _ { t a r , q } ^ { o b j } \cdot P _ { p r e v , q } ^ { o b j } } ,\tag{6}
$$

where $P _ { t a r , q } ^ { o b j }$ and $P _ { p r e v , q } ^ { o b j }$ are computed analogously to Eq. (3). The transitionaware target relation query is then:

$$
\widehat { Q } _ { t a r , q } ^ { r e l } = Q _ { t a r , q } ^ { r e l } + \gamma c _ { q } r _ { q } g _ { q } f _ { \Delta } ( \Delta _ { q } ^ { d i r } ) ,\tag{7}
$$

where $f _ { \varDelta }$ is a learnable MLP projector and $\gamma = \gamma _ { \mathrm { m a x } } \sigma ( \rho _ { \gamma } )$ is a learnable global injection strength bounded by $\gamma _ { \mathrm { m a x } }$ . Here, $f _ { \varDelta } ( \Delta _ { q } ^ { d i r } )$ maps the normalized displacement direction to an update vector, while the scalar $c _ { q } r _ { q } g _ { q }$ scales this update according to foreground confidence and displacement magnitude. The resulting set of TADI-injected queries is $\widehat { Q } _ { t a r } ^ { r e l } = \{ \widehat { Q } _ { t a r , q } ^ { r e l } \} _ { q = 1 } ^ { N _ { q } }$

Reference relation queries are ranked using their object and predicate confidence, together with a displacement-saliency term computed from the target and preceding frames:

$$
s _ { j } ^ { r e l } = \log \left( \operatorname* { m a x } ( P _ { j } ^ { o b j } , \epsilon ) \right) + \log \left( \operatorname* { m a x } ( P _ { j } ^ { r e l } , \epsilon ) \right) + \log \left( 1 + \eta \bar { a } _ { q ( j ) } \right) ,\tag{8}
$$

where $P _ { j } ^ { r e l }$ is the product of the maximum predicate probabilities from the attention, spatial, and contacting families, and $q ( j )$ maps reference candidate $j$ to its query index. The saliency $\bar { a } _ { q ( j ) }$ is the clipped, query-mean-normalized form of $P _ { t a r , q ( j ) } ^ { o b j } r _ { q ( j ) } g _ { q ( j ) }$ , and η controls its contribution. The top- $K _ { d }$ candidates form $Q _ { r e f } ^ { r e l , ( d ) }$ , and the final relation queries $Q _ { t a r } ^ { r e l , * } = Q _ { t a r } ^ { r e l , ( N _ { d } ) }$ are obtained via Eq. (2). The refined $Q _ { t a r } ^ { i n s , * }$ and $Q _ { t a r } ^ { r e l , * }$ are passed to temporal prediction heads for object classification, box regression, and predicate classification.

## 3.5 Spatio-Semantic Instance Alignment

Because TADI computes displacement between same-index queries across adjacent frames, cross-frame query consistency is important for reducing representational drift in the displacement signal. Following prior work on same-index contrastive alignment for temporal representation learning [17], Spatio-Semantic Instance Alignment (SSIA) applies contrastive regularization to adjacent-frame instance-pair queries during spatial training. This encourages cross-frame consistency of instance-pair queries and introduces no inference overhead.

Anchor construction. Bipartite matching between the predicted triplet queries and ground-truth triplets at frame t yields M matched query indices $\{ q _ { m } \} _ { m = 1 } ^ { M }$ Here, $^ { c _ { t , m } }$ and $b _ { t , m }$ denote the category and bounding box of the object entity in the ground-truth triplet matched to query $q _ { m } \ \mathrm { ( S e c . \ 3 . 1 ) }$ . For each $q _ { m }$ , we pair $Q _ { t , q _ { m } } ^ { i n s }$ with the same-index embedding $Q _ { t - 1 , q _ { m } } ^ { i n s }$ from frame t−1 as a candidate positive, without a separate matching step. We use ground-truth object persistence to select anchors for same-index contrastive alignment. Specifically, an anchor is retained when a ground-truth object $j$ in frame t−1 shares the category of the object in the matched triplet and suficiently overlaps its box:

$$
\mathcal { M } _ { m } = \mathbb { 1 } \binom { \operatorname* { m a x } _ { j : c _ { t - 1 , j } = c _ { t , m } } \operatorname { I o U } ( b _ { t , m } , b _ { t - 1 , j } ) > \tau \bigg ) . }\tag{9}
$$

We set $\mathcal { M } _ { m } = 0$ when no object of category $c _ { t , m }$ exists in frame $t { - } 1$

Contrastive objective. A two-layer MLP projection head $f _ { p r o j }$ maps instancepair queries $Q _ { t , q _ { m } } ^ { i n s } , Q _ { t - 1 , q _ { m } } ^ { i n s }$ to ℓ -normalized embeddings $z _ { t , m } , \bar { z _ { t - 1 , m } } \in \mathbb { R } ^ { D }$ . Let $\mathcal { A } = \{ m \ | \ M _ { m } ^ { ' } = \ 1 \}$ be the valid anchor set. The concatenated embedding bank of size 2M is $\mathcal { Z } = \{ z _ { t , 1 } , \ldots , z _ { t , M } , z _ { t - 1 , 1 } , \ldots , z _ { t - 1 , M } \}$ , indexed such that $z _ { m } = z _ { t , m }$ for $m \le M$ and $z _ { m + M } = z _ { t - 1 , m }$ for $m \le M$ . The validity mask restricts the anchor set, while all 2M embeddings remain in the contrastive bank. SSIA applies a symmetric InfoNCE loss [22] over valid anchors:

$$
\mathcal { L } _ { S S I A } = - \frac { 1 } { 2 | \boldsymbol { A } | } \sum _ { m \in \boldsymbol { A } } \left[ \ell ( \boldsymbol { z } _ { t , m } , \boldsymbol { z } _ { t - 1 , m } , m ) + \ell ( \boldsymbol { z } _ { t - 1 , m } , \boldsymbol { z } _ { t , m } , m + M ) \right] ,\tag{10}
$$

where $\begin{array} { r } { \ell ( a , b , i ) = \log { \frac { \exp ( a ^ { \top } b / \kappa ) } { \sum _ { k \neq i } \exp ( a ^ { \top } z _ { k } / \kappa ) } } } \end{array}$ and κ is the temperature. If $A = \emptyset$ , we set $\mathcal { L } _ { S S I A } = 0$

## 3.6 Optimization and Inference

Set-prediction objective. We assign triplet queries to ground-truth annotations using Hungarian matching [1, 29], with a weighted cost combining object and predicate classification with $\ell _ { 1 }$ and GIoU terms for subject and object boxes. The training loss is

$$
\mathcal { L } _ { S G G } = \lambda _ { o b j } \mathcal { L } _ { o b j } + \lambda _ { r e l } \mathcal { L } _ { r e l } + \lambda _ { L 1 } \mathcal { L } _ { L 1 } ^ { b o x } + \lambda _ { g i o u } \mathcal { L } _ { g i o u } ^ { b o x } ,\tag{11}
$$

where $\lambda$ denotes the loss weights. Object classification supervises all queries, with unmatched queries assigned to the no-object class. Predicate and box losses apply only to matched queries. The box losses supervise both subject and object boxes, while $\mathcal { L } _ { r e l }$ sums the classification losses for the attention, spatial, and contacting families.

Decoupled optimization and one-stage inference. Following OED [29], we train RTAformer in two phases, first learning the spatial representations and then optimizing temporal refinement. In Phase 1, the spatial parser is trained with the backbone frozen, using frame-level supervision and SSIA regularization:

$$
\mathcal { L } _ { s t a g e 1 } = \mathcal { L } _ { S G G } ^ { s p a t i a l } + \lambda _ { s s i a } \mathcal { L } _ { S S I A } ,\tag{12}
$$

where $\mathcal { L } _ { S G G } ^ { s p a t i a l }$ denotes $\mathcal { L } _ { S G G }$ computed from the frame-wise spatial parsing outputs. In Phase 2, the backbone and spatial modules are frozen, and only the DTA module and temporal prediction heads are optimized on sequence-level clips:

$$
\mathcal { L } _ { s t a g e 2 } = \mathcal { L } _ { S G G } ^ { t e m p o r a l } ,\tag{13}
$$

where $\mathcal { L } _ { S G G } ^ { t e m p o r a l }$ denotes $\mathcal { L } _ { S G G }$ computed from the temporally refined DTA outputs. This two-phase schedule is used only during training. At inference, RTAformer performs spatial triplet parsing, DTA, and final prediction without external trackers or temporal post-processing.

## 4 Experiments

Dataset. Following prior VidSGG work [4,14,15,20,29,31,39], we evaluate our method on Action Genome (AG) [11], the standard benchmark for video scene graph generation. AG contains 234,253 annotated frames sampled from 9,848 videos, with 476,229 object bounding boxes and 1,715,568 relation annotations across 35 object categories excluding person and 25 predicate categories. The predicates are grouped into three semantic types: attention, spatial, and contacting.

Evaluation Metrics. We evaluate our model under the standard scene graph detection (SGDET) protocol. SGDET requires the model to jointly localize object instances, classify their object categories, and predict the predicates between subject–object pairs from video frames. We report recall@K (R@K) with $K \in \{ 1 0 , 2 0 , 5 0 \}$ under two graph construction settings: With Constraint and No Constraint. The With Constraint setting restricts each subject–object pair to its highest-scoring predicate prediction, whereas the No Constraint setting allows multiple predicate predictions for the same pair. We additionally report mean recall@K (mR@K), which averages recall across predicate categories to give each category equal weight.

Table 1: Experimental results on Action Genome under the SGDET protocol. We report Recall@K (R@K) and mean Recall@K (mR@K) under With Constraint and No Constraint settings. ‘–’ indicates that the metric is not reported.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Backbone</td><td colspan="6">With Constraint</td><td colspan="6">No Constraint</td></tr><tr><td colspan="8">R@10 R@20 R@50 |mR@10 mR@20 mR@50| |R@10 R@20</td><td colspan="4">R@50 |mR@10 mR@20 mR@50</td></tr><tr><td colspan="10">Multi-stage methods</td><td></td><td></td><td></td><td></td></tr><tr><td>STTran [4]</td><td>ResNet-101</td><td>25.2</td><td>34.1</td><td>37.0</td><td>16.6</td><td>20.8</td><td>22.2</td><td>24.6</td><td>36.2</td><td>48.8</td><td>20.9</td><td>29.7</td><td>39.2</td></tr><tr><td>APT [14]</td><td>ResNet-101</td><td>26.3</td><td>36.1</td><td>38.3</td><td></td><td></td><td></td><td>25.7</td><td>37.9</td><td>50.1</td><td></td><td></td><td></td></tr><tr><td>TR2 [30]</td><td>ResNet-101</td><td>26.8</td><td>35.5</td><td>38.3</td><td></td><td></td><td></td><td>27.8</td><td>39.2</td><td>50.0</td><td></td><td></td><td></td></tr><tr><td>TEMPURA [20]</td><td>ResNet-101</td><td>28.1 30.3</td><td>33.4</td><td>34.9</td><td>18.5</td><td>22.6</td><td>23.7</td><td>29.8</td><td>38.1</td><td>46.4</td><td>24.7</td><td>33.9</td><td>43.7</td></tr><tr><td>DSG-DETR [8]</td><td>ResNet-101</td><td>28.7</td><td>34.8</td><td>36.1</td><td></td><td></td><td></td><td>32.1</td><td>40.9</td><td>48.3</td><td></td><td></td><td></td></tr><tr><td>TD2-Net [15]</td><td>ResNet-101 ResNet-101</td><td>29.9</td><td>34.2</td><td>37.1 35.0</td><td>20.4</td><td></td><td>26.1</td><td>30.5</td><td></td><td>49.3</td><td>27.9</td><td></td><td>46.3</td></tr><tr><td>STABILE [39] DIFFVSGG [2]</td><td>ResNet-50+DETR</td><td>34.7</td><td>41.9</td><td>47.3</td><td>22.8 24.7</td><td>25.6 27.3</td><td>26.5 28.4</td><td>33.7 37.4</td><td>41.2 45.1</td><td>47.7 53.1</td><td>29.1 29.7</td><td>36.5 37.9</td><td>43.9 46.4</td></tr><tr><td colspan="10">One-stage methods</td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10"></td><td></td><td></td><td></td><td></td></tr><tr><td>OED [29]</td><td>ResNet-50</td><td>33.5</td><td>40.9</td><td>48.9</td><td>21.5</td><td>27.4</td><td>33.4</td><td>35.3</td><td>44.0</td><td>51.8</td><td>27.2</td><td>40.5</td><td>50.4</td></tr><tr><td>ARN [31]</td><td>ResNet-50</td><td>35.1 35.4</td><td>41.8</td><td>47.2</td><td>23.2</td><td>29.0</td><td>33.6</td><td>37.6 36.6</td><td>46.8 46.1</td><td>54.1</td><td>29.7</td><td>41.9</td><td>51.9</td></tr><tr><td>UNO [12] UNO [12]</td><td>ResNet-50 ViT-S/14</td><td>36.7</td><td>42.2 43.1</td><td>49.5 50.2</td><td></td><td></td><td></td><td>37.5</td><td>47.5</td><td>53.9 54.5</td><td></td><td></td><td></td></tr><tr><td>UNO [12]</td><td>ViT-B/14</td><td>38.2</td><td>44.7</td><td>51.9</td><td></td><td></td><td></td><td>39.9</td><td>48.2</td><td>56.3</td><td></td><td></td><td></td></tr><tr><td>UNO [12]</td><td>ViT-L/14</td><td>39.3</td><td>45.2</td><td>53.8</td><td></td><td></td><td></td><td>40.8</td><td>49.7</td><td>57.1</td><td></td><td></td><td></td></tr><tr><td>Ours</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Ours</td><td>ViT-S/14</td><td>37.7 42.2</td><td>45.1 50.1</td><td>52.3 57.6</td><td>24.6 28.8</td><td>30.9 35.6</td><td>36.2 41.3</td><td>40.0 44.8</td><td>49.7 55.0</td><td>57.5 63.2</td><td>30.5 35.3</td><td>45.0 50.7</td><td>56.0 61.9</td></tr><tr><td>Ours</td><td>ViT-B/14</td><td>44.6</td><td>52.9</td><td>60.2</td><td>30.5</td><td>38.3</td><td>44.2</td><td>47.2</td><td>57.8</td><td>65.9</td><td>37.2</td><td>53.5</td><td>64.4</td></tr><tr><td></td><td>ViT-L/14</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Implementation Details. Our framework follows recent query-based one-stage VidSGG architectures [29, 31]. Following UNO [12], we use the same frozen DINOv2 vision transformer (ViT) backbone [7, 23]. The transformer encoder, instance-pair decoder, and relation decoder each contain six transformer layers, and the model uses $N _ { q } = 1 0 0$ learnable triplet queries. In DTA, the instance and relation streams each contain three temporal aggregation decoders, each implemented as a single transformer decoder layer. For SSIA, we set $\kappa = 0 . 1$ $\tau = 0 . 3$ , and $\lambda _ { \mathrm { S S I A } } ~ = ~ 0 . 1$ . We train the model using the AdamW optimizer on four NVIDIA RTX A6000 GPUs. Unless otherwise specified, ablation studies use ${ \mathrm { V i T } } { \mathrm { - B } } / 1 4 .$ , while the main comparisons report ViT-S/14, ViT-B/14, and $\mathrm { V i T - L } / 1 4$ variants. Additional implementation details are provided in the supplementary material.

## 4.1 Comparison with State-of-the-Art Methods

We compare RTAformer with recent VidSGG methods, including both multistage pipelines and one-stage frameworks. Table 1 reports SGDET R@K and class-balanced mR@K under both With Constraint and No Constraint settings. For mR@K, methods without reported values [8,12,14,30] are marked as ‘–’ and are not considered in the mR@K comparison.

With matched frozen DINOv2 backbones [23], RTAformer consistently outperforms UNO [12] across all backbone scales and both constraint settings. With ${ \mathrm { V i T - L } } / 1 4$ , RTAformer achieves R@10/20/50 of 44.6/52.9/60.2 under with constraint and 47.2/57.8/65.9 under no constraint, improving R@10 over UNO by 5.3 and 6.4 percentage points, respectively. The R@10 margin increases with backbone scale under both settings, from 1.0/2.5 percentage points with ViT-

Table 2: Module ablation. Starting from the frame-wise spatial baseline, we evaluate SSIA and DTA individually and jointly. The spatial baseline performs frame-wise triplet parsing without temporal refinement. DTA denotes the full temporal module with TADI. All variants use the ViT-B/14 backbone.  
Table 3: Ablation on relation-stream transition injection design. Starting from the spatial+SSIA baseline, we compare temporal aggregation without transition injection and three relation-displacement injection variants. All variants use the same spatial branch and ViT-B/14 backbone.
<table><tr><td rowspan="3">Spatial SSIA DTA</td><td rowspan="3"></td><td colspan="3">With Constraint</td><td colspan="3">No Constraint</td></tr><tr><td>R@10</td><td>R@20</td><td>R@50</td><td>R@10</td><td>R@20</td><td>R@50</td></tr><tr><td>√</td><td></td><td></td><td>39.7 47.5</td><td>53.7</td><td>42.0</td><td>52.6</td><td>61.2</td></tr><tr><td>√</td><td>√</td><td>40.8</td><td>48.1</td><td>53.9</td><td>43.3</td><td>53.6</td><td>61.4</td></tr><tr><td>√</td><td></td><td>√</td><td>41.2 49.4</td><td>56.8</td><td>43.8</td><td>54.3</td><td>62.8</td></tr><tr><td>√</td><td>√</td><td>√</td><td>42.2 50.1</td><td>57.6</td><td>44.8</td><td>55.0</td><td>63.2</td></tr></table>

<table><tr><td rowspan="2">Variant</td><td colspan="3">With Constraint</td><td colspan="3">No Constraint</td></tr><tr><td>R@10 R@20 R@50 R@10 R@20 R@50</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Spatial + SSIA</td><td>40.8</td><td>48.1</td><td>53.9</td><td>43.3</td><td>53.6</td><td>61.4</td></tr><tr><td>+ Temporal Agg. (no inj.)</td><td>41.3</td><td>49.0</td><td>56.1</td><td>43.9</td><td>54.1</td><td>61.7</td></tr><tr><td>+ Direction-only Inj.</td><td>41.7</td><td>49.5</td><td>56.9</td><td>44.2</td><td>54.1</td><td>62.0</td></tr><tr><td>+ Raw Delta Inj.</td><td>41.9</td><td>49.8</td><td>57.2</td><td>44.4</td><td>54.4</td><td>62.4</td></tr><tr><td>+ TADI</td><td>42.2</td><td>50.1</td><td>57.6</td><td>44.8</td><td>55.0</td><td>63.2</td></tr></table>

S/14 to 5.3/6.4 percentage points with ViT-L/14. The ViT-L/14 variant also achieves the highest R@K among all methods reported in Table 1.

We further compare mR@K among methods that report it. With ViT-L/14, RTAformer attains the highest reported mR@K across all evaluated cutofs and settings, reaching 30.5/38.3/44.2 under with constraint and 37.2/53.5/64.4 under no constraint, outperforming both ARN [31] and DIFFVSGG [2]. The improvements extend to class-averaged recall, complementing the gains in overall R@K.

## 4.2 Ablation Studies

Efect of SSIA and DTA. Table 2 reports module-level contributions. Both SSIA and DTA improve over the spatial baseline, and their combination achieves the highest recall across all evaluated cutofs and settings. SSIA remains beneficial when DTA is enabled, increasing R@10 from 41.2 to 42.2 under with constraint.

Efect of Transition-Aware Delta Injection. Table 3 compares transition cue injection designs within the relation stream. Temporal aggregation without injection already improves over the spatial+SSIA baseline, supporting the benefit of temporal context. Injecting relation-query displacement further improves performance, with raw delta consistently outperforming direction-only injection. TADI achieves the highest recall across all evaluated cutofs and settings, supporting the efectiveness of the complete calibrated injection design.

## 4.3 Analysis of Relation Transitions

We examine two questions: (i) whether relation-transition frames are particularly challenging for a baseline model, and (ii) whether adjacent-frame relation-query diferences provide useful cues for these transitions.

Relation-transition subset construction. To focus on relation changes involving persistent entities, we construct evaluation subsets from adjacent annotated frames in the AG test split. We retain a subject–object pair across frames when the entities satisfy an IoU threshold of 0.3 and the object category remains consistent. A pair adjacency is labeled as a transition when at least one relation family (attention, spatial, or contacting) has diferent, disjoint predicate sets across the two frames. Changes with partially overlapping predicate sets do not satisfy this criterion. Among 105,356 persistent pair adjacencies, 35,523 (33.7%) satisfy the transition criterion. Among target frames containing at least one persistent subject–object pair, a frame is classified as relation-transition if any such pair satisfies the transition criterion, and as non-transition otherwise. This yields 25,883 relation-transition frames and 21,474 non-transition frames.

Table 4: Controlled comparison with OED using the same frozen DINOv2 ViT-B/14 backbone. WC and NC denote the with constraint and no constraint settings, respectively. Recall entries report R@10/R@20.
<table><tr><td rowspan="2">Method</td><td colspan="2">Full AG</td><td colspan="2">Transition Subset</td><td colspan="2">Non-transition Subset Parameters (M)</td><td colspan="2"></td><td rowspan="2">GFLOPs FPS</td><td rowspan="2"></td></tr><tr><td>WC</td><td>NC</td><td>WC</td><td>NC</td><td>WC</td><td>NC</td><td></td><td>Total Temporal</td></tr><tr><td>OED [29]</td><td>38.7/46.6</td><td>41.1/51.4</td><td>34.3/42.7</td><td>37.4/48.0</td><td>45.9/53.7</td><td>48.9/58.9</td><td>126.8</td><td>12.9</td><td>1041.6</td><td>31.43</td></tr><tr><td>Ours</td><td></td><td></td><td>42.2/50.1 44.8/55.0 39.1/47.741.5/52.9 47.6/55.2 50.8/60.5123.8</td><td></td><td></td><td></td><td></td><td>9.9</td><td>1041.0</td><td>31.35</td></tr></table>

![](images/5a350d25e85e6b7c3c5083df50a3e686958e8fe6607048179f34840dda7f94c9.jpg)  
(a) Recall@K on relation-transition frames.

![](images/b635946583d04b5e0cc66db4d86c7d433e1cd7f748c06b24b347bc26679b8474.jpg)  
Fig. 4: Relation-transition analysis. (a) Recall on the relation-transition frame subset, comparing RTAformer with ViT-B/14 against OED and TEMPURA using their released SGDET configurations. (b) Empirical transition rates across equal-count deciles of relation-query displacement magnitude $r _ { q }$ and foreground confidence $c _ { q } ,$ computed separately for each score over persistent pair adjacencies. Deciles are ordered from low to high score values. The dashed line indicates the overall transition rate of 33.7%.

Backbone-controlled comparison. We reimplement OED [29] with the same frozen DINOv2 ViT-B/14 backbone as RTAformer to compare performance across the two frame subsets. The baseline’s R@10 is 11.6 percentage points lower on relation-transition frames than on non-transition frames under with constraint, highlighting the dificulty of the transition subset (Table 4). RTAformer improves R@10 over this baseline by 4.8 percentage points on transition frames, compared with 1.7 on non-transition frames. The same pattern holds under no constraint, with gains of 4.1 and 1.9 percentage points, respectively. The larger gains on transition frames are consistent with our motivation to make temporal relation modeling more responsive to relation changes. RTAformer also uses fewer total and temporal-module parameters while maintaining comparable GFLOPs and inference speed.

![](images/752806d042cb105f6c2c6b91a6ceb026bb6fe5eae6ccd55e0f518b0ebc036d88.jpg)  
Fig. 5: Qualitative comparison around relation transition boundaries. The participating entities remain visible across adjacent frames, while predicates for ‘person–broom’ and ‘person–door’ change. RTAformer updates the changed predicates, e.g., ‘person– holding–broom’ and ‘person–not contacting–door’, whereas OED preserves previousframe relations such as ‘person–not contacting–broom’ and ‘person–touching–door’.

Comparison with released baseline configurations. As a complementary evaluation, we compare RTAformer with ViT-B/14 against OED [29] and TEM-PURA [20] using their released SGDET configurations. RTAformer achieves higher recall on the relation-transition frame subset than both baselines at all evaluated cutofs under both constraint settings (Fig. 4(a)). In particular, its R@10 exceeds OED’s by 8.0 and 8.6 percentage points under with constraint and no constraint, respectively.

Relation-query displacement as a transition cue. We next examine whether larger changes in relation-query representations are associated with actual predicate transitions. We evaluate the displacement magnitude $r _ { q }$ from Eq. (5) over the 105,356 persistent pair adjacencies. As $r _ { q }$ increases across equal-count deciles, transition prevalence rises monotonically from 20.9% to 55.4%, compared with the overall rate of 33.7% (Fig. 4(b)). Transition cases also have a larger mean $r _ { q }$ than non-transition cases (0.396 vs. 0.289), and using $r _ { q }$ directly as a transition score yields an AUROC of 0.621 without fitting a classifier. Similar AUROCs are observed within attention, spatial, and contacting relations (0.633/0.636/0.632). These findings support using relation-query displacement as an informative but imperfect cue for predicate updates.

For comparison, the foreground confidence $c _ { q }$ used to gate injection in Eq. (6) is lower for transition cases than for non-transition cases (0.517 vs. 0.591), with an AUROC of 0.442 when larger values are treated as indicating transitions.

Qualitative Analysis at Transition Boundaries. Fig. 5 shows representative relation-transition cases. The same entities remain visible across adjacent frames, while the predicates for ‘person–broom’ and ‘person–door’ change. For example, ‘person–broom’ changes from not contacting to holding, and ‘person– door’ changes from touching to not contacting. OED preserves the previous-frame predicates at these transition boundaries, whereas RTAformer updates the relation labels according to the current frame. This qualitatively illustrates relational semantic inertia and complements the transition-subset results in Fig. 4(a).

## 5 Limitations

Although SSIA improves same-index temporal consistency and qualitative observations suggest that same-index queries can maintain consistent subject–object grounding across adjacent frames, DETR-style decoding does not guarantee exact cross-frame query correspondence. Our evaluation is limited to Action Genome under the SGDET setting, and broader evaluation across VidSGG datasets and protocols would strengthen evidence for generalization.

## 6 Conclusion

Reliable video scene graph generation requires models to recognize when a previously valid relation no longer describes the observed interaction. We examine this challenge through relational semantic inertia, drawing attention to outdated predicate predictions that persist even as relationships between visible entities change. RTAformer uses calibrated relation-query changes to guide predicate updates, and experiments on Action Genome show improved SGDET performance, with a backbone-controlled comparison revealing larger gains on relationtransition frames than on non-transition frames. Our analyses connect these performance observations with evidence that relation-query displacement carries information about predicate transitions. Together, the characterization of this failure mode and the transition-focused evaluation provide a concrete basis for future work to examine responsiveness to relation changes alongside overall prediction accuracy.

## Acknowledgements

This work was partly supported by the Institute of Information & Communications Technology Planning & Evaluation (IITP) under the Leading Generative AI Human Resources Development (IITP-2026-RS-2026-25544647, 40%) grant funded by the Korean government (MSIT), the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (RS-2025-00553785, 30%), and the Technology Innovation Program (K-On-Device AI Semiconductor Technology Development) (RS-2026-25617922, 30%, Development of On-Device AI Solution and Service Technology Utilizing Task-Specialized Foundation Model for Next-Generation Cobot) funded by the Ministry of Trade, Industry and Resources (MOTIR, Korea).

## References

1. Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., Zagoruyko, S.: End-to-end object detection with transformers. In: ECCV. pp. 213–229 (2020). https://doi.org/10.1007/978-3-030-58452-8\_13

2. Chen, M., Li, L., Wang, W., Yang, Y.: DIFFVSGG: Difusion-driven online video scene graph generation. In: CVPR. pp. 29161–29172 (2025)

3. Cherian, A., Hori, C., Marks, T.K., Le Roux, J.: (2.5+1)D spatio-temporal scene graphs for video question answering. In: AAAI. vol. 36, pp. 444–453 (2022)

4. Cong, Y., Liao, W., Ackermann, H., Rosenhahn, B., Yang, M.Y.: Spatial-temporal transformer for dynamic scene graph generation. In: ICCV. pp. 16372–16382 (2021)

5. Cong, Y., Yang, M.Y., Rosenhahn, B.: RelTR: Relation transformer for scene graph generation. IEEE TPAMI 45(9), 11169–11183 (2023). https://doi.org/10.1109/ TPAMI.2023.3268066

6. Cui, Y.: Feature aggregated queries for transformer-based video object detectors. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 6365–6376 (2023)

7. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N.: An image is worth 16x16 words: Transformers for image recognition at scale. In: ICLR (2021)

8. Feng, S., Mostafa, H., Nassar, M., Majumdar, S., Tripathi, S.: Exploiting longterm dependencies for generating dynamic scene graphs. In: WACV. pp. 5130–5139 (2023)

9. Huang, J., Sethi, A., Kuo, M., Keoliya, M., Velingker, N., Jung, J., Lim, S.N., Li, Z., Naik, M.: ESCA: Contextualizing embodied agents via scene-graph generation. In: NeurIPS (2025)

10. Im, J., Nam, J., Park, N., Lee, H., Park, S.: EGTR: Extracting graph from transformer for scene graph generation. In: CVPR. pp. 24229–24238 (2024)

11. Ji, J., Krishna, R., Fei-Fei, L., Niebles, J.C.: Action genome: Actions as compositions of spatio-temporal scene graphs. In: CVPR. pp. 10236–10247 (2020)

12. Le, H., Chung, N., Kieu, T., Yang, J., Le, N.: UNO: Unifying one-stage video scene graph generation via object-centric visual representation learning. In: WACV (2026)

13. Li, R., Zhang, S., He, X.: SGTR+: End-to-end scene graph generation with transformer. IEEE TPAMI 46(4), 2191–2205 (2024). https://doi.org/10.1109/ TPAMI.2023.3332246

14. Li, Y., Yang, X., Xu, C.: Dynamic scene graph generation via anticipatory pretraining. In: CVPR. pp. 13874–13883 (2022)

15. Lin, X., Shi, C., Zhan, Y., Yang, Z., Wu, Y., Tao, D.: TD<sup>2</sup>-Net: Toward denoising and debiasing for video scene graph generation. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 38, pp. 3495–3503 (2024). https: //doi.org/10.1609/aaai.v38i4.28137

16. Liu, S., Li, F., Zhang, H., Yang, X., Qi, X., Su, H., Zhu, J., Zhang, L.: DAB-DETR: Dynamic anchor boxes are better queries for DETR. In: ICLR (2022)

17. Manasyan, A., Seitzer, M., Radovic, F., Martius, G., Zadaianchuk, A.: Temporally consistent object-centric learning by contrasting slots. In: Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR). pp. 5401–5411 (June 2025)

18. Meinhardt, T., Kirillov, A., Leal-Taixé, L., Feichtenhofer, C.: TrackFormer: Multiobject tracking with transformers. In: CVPR. pp. 8844–8854 (2022)

19. Meng, D., Chen, X., Fan, Z., Zeng, G., Li, H., Yuan, Y., Sun, L., Wang, J.: Conditional DETR for fast training convergence. In: ICCV. pp. 3651–3660 (2021)

20. Nag, S., Min, K., Tripathi, S., Roy-Chowdhury, A.K.: Unbiased scene graph generation in videos. In: CVPR. pp. 22803–22813 (2023)

21. Nguyen, T.T., Nguyen, P., Cothren, J., Yilmaz, A., Luu, K.: HyperGLM: Hypergraph for video scene graph generation and anticipation. In: CVPR. pp. 29150– 29160 (2025)

22. van den Oord, A., Li, Y., Vinyals, O.: Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748 (2018)

23. Oquab, M., Darcet, T., Moutakanni, T., Vo, H.V., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al.: DINOv2: Learning robust visual features without supervision. TMLR (2024)

24. Rosinol, A., Gupta, A., Abate, M., Shi, J., Carlone, L.: 3D dynamic scene graphs: Actionable spatial perception with places, objects, and humans. In: Robotics: Science and Systems (RSS) (2020). https://doi.org/10.15607/RSS.2020.XVI.079

25. Tang, K., Zhang, H., Wu, B., Luo, W., Liu, W.: Learning to compose dynamic tree structures for visual contexts. In: CVPR. pp. 6619–6628 (2019)

26. Teng, Y., Wang, L., Li, Z., Wu, G.: Target adaptive context aggregation for video scene graph generation. In: ICCV. pp. 13688–13697 (2021)

27. Urooj, A., Kuehne, H., Wu, B., Chheu, K., Bousselham, W., Gan, C., Lobo, N., Shah, M.: Learning situation hyper-graphs for video question answering. In: CVPR. pp. 14879–14889 (2023)

28. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., Polosukhin, I.: Attention is all you need. In: NeurIPS. pp. 5998–6008 (2017)

29. Wang, G., Li, Z., Chen, Q., Liu, Y.: OED: Towards one-stage end-to-end dynamic scene graph generation. In: CVPR. pp. 27938–27947 (2024)

30. Wang, J., Huang, J., Zhang, C., Deng, Z.: Cross-modality time-variant relation learning for generating dynamic scene graphs. In: Proceedings of the IEEE International Conference on Robotics and Automation (ICRA) (2023)

31. Wang, L., Zhang, Y., Lu, T., Liu, F., Zhang, H., Ma, J., Zhou, H.: End-to-end entity-predicate association reasoning for dynamic scene graph generation. In: ICCV. pp. 17729–17738 (2025)

32. Wang, S., Gao, L., Lyu, X., Guo, Y., Zeng, P., Song, J.: Dynamic scene graph generation via temporal prior inference. In: ACM MM. pp. 5793–5801 (2022)

33. Yang, J., Lu, J., Lee, S., Batra, D., Parikh, D.: Graph R-CNN for scene graph generation. In: ECCV. pp. 670–685 (2018)

34. Yin, H., Xu, X., Wu, Z., Zhou, J., Lu, J.: SG-Nav: Online 3D scene graph prompting for LLM-based zero-shot object navigation. In: NeurIPS. vol. 37, pp. 5285–5307 (2024)

35. Zellers, R., Yatskar, M., Thomson, S., Choi, Y.: Neural motifs: Scene graph parsing with global context. In: CVPR. pp. 5831–5840 (2018)

36. Zeng, F., Dong, B., Zhang, Y., Wang, T., Zhang, X., Wei, Y.: MOTR: End-to-end multiple-object tracking with transformer. In: ECCV. pp. 659–675 (2022)

37. Zhang, H., Li, F., Liu, S., Zhang, L., Su, H., Zhu, J., Ni, L., Shum, H.Y.: DINO: DETR with improved denoising anchor boxes for end-to-end object detection. In: ICLR (2023)

38. Zhang, Y., Pan, Y., Yao, T., Huang, R., Mei, T., Chen, C.W.: End-to-end video scene graph generation with temporal propagation transformer. IEEE TMM 26, 1613–1625 (2024). https://doi.org/10.1109/TMM.2023.3283879

39. Zhuang, W., Dong, B., Zhu, Z., Li, Z., Liu, J., Wang, Y., Hong, X., Li, X., Zuo, W.: Spatial-temporal saliency guided unbiased contrastive learning for video scene graph generation. IEEE TMM 27, 3092–3104 (2025). https://doi.org/10.1109/ TMM.2025.3557688

# Relation-Transition-Aware Transformer for Video Scene Graph Generation (Supplementary Material)

## S1 Additional Implementation and Evaluation Details

In this section, we provide the implementation details omitted from the main paper due to space constraints.

## S1.1 Architecture Details

Table S1: Architecture configuration of RTAformer.
<table><tr><td>Component</td><td>Configuration</td></tr><tr><td>Visual backbone</td><td>Frozen DINOv2 [7] ViT-S/14, ViT-B/14, ViT-L/14</td></tr><tr><td>Input resolution</td><td>518 × 518</td></tr><tr><td>Transformer encoder</td><td>6 layers</td></tr><tr><td>Instance-pair decoder</td><td>6 layers</td></tr><tr><td>Relation context decoder</td><td>6 layers</td></tr><tr><td>Temporal aggregation decoder</td><td>1 layer</td></tr><tr><td>Temporal aggregation depth  $N _ { d }$ </td><td>3 blocks per DTA stream</td></tr><tr><td>Hidden dimension  $d _ { \mathrm { m o d e l } }$ </td><td>256</td></tr><tr><td>Feed-forward dimension  $d _ { \mathrm { f f } }$ </td><td>2048</td></tr><tr><td>Number of attention heads</td><td>8</td></tr><tr><td>Dropout rate</td><td>0.1</td></tr><tr><td>Number of triplet queries  $N _ { q }$ </td><td>100</td></tr></table>

Following the recent VidSGG baseline UNO [5], we adopt the same frozen DI-NOv2 [7] visual backbones under the ViT-based [3] setting. Keeping the backbone frozen and identical across all compared variants ensures that performance diferences arise from the model design rather than from backbone adaptation, enabling a controlled comparison under a shared visual representation. For the Transformer modules, we adopt a standard DETR-style configuration with $d _ { \mathrm { m o d e l } } = 2 5 6 , d _ { \mathrm { f f } } = 2 0 4 8$ , 8 attention heads, and $N _ { q } = 1 0 0$ triplet queries [1]. Here, $N _ { d }$ denotes the number of stacked Temporal Aggregation Decoder blocks in each DTA stream. Each block is a single standard Transformer decoder layer consisting of self-attention, temporal cross-attention, and a feed-forward network [8].

## S1.2 Training and Evaluation Details

Training is performed in two stages. In the first stage, we train the frame-wise spatial parser with frame-level SGG supervision and the SSIA objective L<sub>SSIA</sub>. In the second stage, we initialize from the trained spatial checkpoint, freeze both the visual backbone and the spatial parser, and optimize only the DTA module and temporal prediction heads on sequence-level clips. Thus, the SSIA loss is applied only during spatial module training, while the temporal training stage focuses on learning transition-aware temporal aggregation.

Table S2: Training details. All RTAformer training runs use a fixed random seed of 42.
<table><tr><td>Hyperparameter</td><td>Spatial module training</td><td>Temporal module training</td></tr><tr><td>Trainable modules</td><td>Spatial parser + SSIA</td><td>DTA + temporal heads</td></tr><tr><td>Frozen modules</td><td>Visual backbone</td><td>Visual backbone + spatial parser</td></tr><tr><td>Per-GPU batch size</td><td>20</td><td>32</td></tr><tr><td>Number of GPUs</td><td>4 NVIDIA RTX A6000</td><td>4 NVIDIA RTX A6000</td></tr><tr><td>Epochs</td><td>31</td><td>10</td></tr><tr><td>Optimizer</td><td>AdamW</td><td> $\mathrm { A d a m W }$ </td></tr><tr><td>Weight decay</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Base learning rate</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>LR schedule</td><td>Step</td><td>Step</td></tr><tr><td>LR decay epoch</td><td>27</td><td>7</td></tr><tr><td>Random seed</td><td>42</td><td>42</td></tr><tr><td>SSIA IoU validity threshold τ</td><td>0.3</td><td></td></tr><tr><td>Stability constant €</td><td></td><td> $1 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>TADI maximum injection strength  $\gamma _ { \mathrm { m a x } }$ </td><td></td><td>0.2</td></tr><tr><td>Relation-stream delta saliency weight η</td><td></td><td>0.05</td></tr></table>

Table S3: Loss coeficients.
<table><tr><td>Loss term  $/$  hyperparameter</td><td>Coefficient</td></tr><tr><td>Object classification loss  $\lambda _ { \mathrm { o b j } }$ </td><td>1.0</td></tr><tr><td>Predicate classification loss  $\lambda _ { \mathrm { { r e l } } }$ </td><td>2.0</td></tr><tr><td>Box  $\ell _ { 1 }$  loss  $\lambda _ { \ell _ { 1 } }$ </td><td>2.5</td></tr><tr><td>Box GIoU loss  $\lambda _ { \mathrm { g i o u } }$ </td><td>1.0</td></tr><tr><td>SSIA loss λssiA</td><td>0.1</td></tr><tr><td>SSIA temperature κ</td><td>0.1</td></tr></table>

For SSIA loss, the IoU validity threshold $\tau = 0 . 3$ in Eq. (9) of the main paper is used only for mining valid positive anchors in the SSIA objective, not for SGDET evaluation. We use a relatively relaxed threshold because Action Genome annotates sparsely sampled frames rather than every video frame [4]. Thus, two adjacent annotated frames may correspond to a non-trivial temporal gap in the original video, and a persistent object can undergo substantial spatial displacement. To avoid discarding such valid cross-frame anchors, we combine this IoU threshold with the category-consistency constraint in Eq. (9) of the main paper.

## S1.3 Relation-Transition Subset Construction

This section provides additional details on the construction of the relationtransition subset used in the main paper’s relation-transition frame subset evaluation (Sec. 4.3 and Fig. 4). This analysis is designed to evaluate model robustness under challenging relation-transition cases, where the same subject–object pair remains visible across adjacent annotated frames but its predicate label changes.

We therefore construct an evaluation-only subset from the Action Genome test split using ground-truth annotations. For each video, we sort annotated frames by frame index and consider adjacent annotated frames $( t - 1 , t )$ , which are not necessarily consecutive frames in the original video.

Table S4: Statistics of the relation-transition subset used for evaluation on the Action Genome test split. Frame-level counts denote unique target frames containing at least one transition pair, while pair-adjacency-level counts denote individual persistent subject–object pairs matched across adjacent annotated frames.
<table><tr><td>Quantity</td><td>Count</td></tr><tr><td>Test videos</td><td>1,750</td></tr><tr><td>Annotated test frames</td><td>56,880</td></tr><tr><td>Relation-transition event frames</td><td>25,883</td></tr><tr><td>Attention-family transition frames</td><td>16,567</td></tr><tr><td>Spatial-family transition frames</td><td>7,177</td></tr><tr><td>Contacting-family transition frames</td><td>11,835</td></tr><tr><td>Persistent subject-object pair-adjacencies</td><td>105,356</td></tr><tr><td>Transition pair-adjacencies</td><td>35,523</td></tr><tr><td>Attention-family transition pair-adjacencies</td><td>20,349</td></tr><tr><td>Spatial-family transition pair-adjacencies</td><td>7,878</td></tr><tr><td>Contacting-family transition pair-adjacencies</td><td>14,093</td></tr></table>

Persistent subject–object pairs. Since Action Genome provides person-centric triplet annotations, we use the annotated person box as the subject and nonperson boxes as objects. A subject–object pair-adjacency is retained only if the subject box persists with $\mathrm { I o U ~ > ~ 0 . 3 }$ and an object of the same category is matched across the two frames by class-wise IoU matching with $\mathrm { I o U > 0 . 3 }$

Transition criterion. For each persistent subject–object pair $p = \left( s , o \right)$ matched between frames t − 1 and t, let $\mathcal { R } _ { t } ^ { f } ( \boldsymbol { p } )$ denote its predicate label set for relation family $f \in$ {attention, spatial, contacting} at frame t. We count a family-wise transition when the predicate sets in the two frames difer and have no overlap:

$$
I _ { \mathrm { t r a n s } } ^ { f } ( p , t ) = \Im \big [ \mathcal { R } _ { t - 1 } ^ { f } ( p ) \neq \mathcal { R } _ { t } ^ { f } ( p ) \land \mathcal { R } _ { t - 1 } ^ { f } ( p ) \cap \mathcal { R } _ { t } ^ { f } ( p ) = \emptyset \big ] .\tag{S1}
$$

A persistent pair p is included in the transition subset if $I _ { \mathrm { t r a n s } } ^ { f } ( p , t ) = 1$ for at least one relation family f. Partially overlapping predicate sets are excluded from the transition subset, i.e., not counted as transition cases, to construct a conservative subset with unambiguous multi-label changes. For frame-level evaluation, we consider only target frames containing at least one persistent subject–object pair with the preceding annotated frame. A target frame belongs to the relationtransition subset if at least one persistent pair satisfies the transition criterion in any relation family, and to the non-transition subset otherwise. These subsets partition the 47,357 eligible target frames into 25,883 relation-transition frames and 21,474 non-transition frames. Non-transition therefore denotes the absence of transitions under our criterion, rather than necessarily unchanged predicate sets. Pair-adjacency-level statistics count individual persistent subject–object pairs across adjacent annotated frames.

Scope of baseline comparison. The relation-transition subset is newly constructed by filtering the Action Genome test split for persistent subject–object pairs with predicate changes. As a result, previously reported full-test-set results are not directly comparable to this subset evaluation, and baselines must be re-evaluated under the same subset construction and evaluation protocol. Evaluating a method on this subset requires its pretrained checkpoint, exact evaluation hyperparameters, and a compatible SGDET evaluation pipeline. We therefore report transition-subset results only for TEMPURA [6] and OED [9], for which compatible public resources allowed controlled evaluation. For UNO [5], DIFFVSGG [2], and ARN [10], we were unable to obtain the combination of pretrained checkpoints, exact evaluation hyperparameters, and compatible SGDET evaluation code required for controlled subset evaluation at the time of our experiments.

## S2 Additional Analysis of Spatio-Semantic Instance Alignment

We first clarify the index correspondence used throughout this analysis. RTAformer represents each candidate interaction by a triplet query indexed by $q \in \{ 1 , \ldots , N _ { q } \}$ . At frame t, the instance-pair query $Q _ { t , q } ^ { i n s }$ predicts the subject and object of the q-th candidate, while the relation query $\dot { Q } _ { t , q } ^ { r e l }$ predicts the predicate for the same candidate pair. This correspondence follows from the cascaded decoding design: $Q _ { t } ^ { r e l }$ is produced by applying the Relation Context Decoder to ${ Q } _ { t } ^ { i n s }$ , with no sorting, reassignment, or re-indexing between the instance-pair and relation stages. Therefore, the predictions at index $q$ are combined to form the $q \mathrm { . }$ -th triplet ⟨subject, predicat $\operatorname { e , o b j e c t } \rangle _ { q } .$ During training, Hungarian matching supervises the subject, object, and predicate predictions at the assigned query index as a single triplet, using a joint cost over object classification, predicate classification, and box localization. Thus, within each frame, index q defines one triplet candidate shared by the instance-pair and relation predictions.

SSIA is designed to stabilize this index-level instance-pair grounding across adjacent frames. This property is important for RTAformer because TADI computes same-index relation-query displacements between adjacent frames. If the same-index queries were not grounded to consistent subject–object pairs, the resulting displacement could reflect representational drift rather than a genuine relation transition. Therefore, this section provides additional analysis showing that SSIA induces temporally consistent instance-pair query representations.

## S2.1 Illustration of SSIA Validity Masking

Before analyzing the trained query representations, we first illustrate the validity masking used in SSIA. SSIA does not force every same-index candidate pair $( Q _ { t , q _ { m } } ^ { i n s } , Q _ { t - 1 , q _ { m } } ^ { i n s } )$ constructed from a GT-matched target query to be close. Instead, a candidate pair is used as a positive anchor only when the target-frame object matched to query $q _ { m }$ has a same-category object with suficient spatial overlap in the previous frame. Otherwise, the pair is masked out from the SSIA contrastive objective.

This masking prevents SSIA from incorrectly pulling together query embeddings when the corresponding object-side entity is not verified to persist across adjacent annotated frames. Thus, SSIA encourages temporal consistency only for reliable cross-frame anchors, rather than imposing same-index alignment unconditionally.

![](images/70c112943ebb4abb4f26f301172fb7374d906635c4e942468753d358c3487d53.jpg)  
Fig. S1: Example of SSIA validity masking. The target frame contains an annotated cup/glass/bottle, while the corresponding object is not annotated in the previous frame. Therefore, the same-index pair $( Q _ { t , q _ { m } } ^ { i n s } , Q _ { t - 1 , q _ { m } } ^ { i n s } )$ fails the spatio-semantic validity check and is not used as a positive anchor in the SSIA loss.

## S2.2 Representation-Space Visualization of Same-Index Query Consistency

Using the same spatio-semantic persistence criterion for sample selection, we qualitatively examine the representation space of same-index instance-pair queries. We visualize instance-pair query representations extracted from the Action Genome test split using t-SNE. We compare two spatial parsers trained with the same configuration, with and without SSIA. For both variants, we collect GT-matched and class-consistent cross-frame pairs whose bounding boxes satisfy $\mathrm { I o U > 0 . 3 }$ across adjacent annotated frames. We then select the 15 most represented query slots and randomly sample 60 temporal pairs per slot, yielding 900 temporal instance pairs per model. For each variant, embeddings from frame t and frame t − 1 are projected into a shared two-dimensional t-SNE space.

As shown in Fig. S2, SSIA produces a more organized query representation space. Same-index query embeddings from adjacent frames remain closer to each other, whereas the model trained without SSIA shows more dispersed embeddings and longer cross-frame links. This supports the role of SSIA as a training-time regularizer that reduces instance-level representation drift before relation-query displacement is used by TADI.

![](images/134f15b9e2456fb380df199d4c5869991c393a8ad5906c4b254013562283773a.jpg)  
Fig. S2: t-SNE visualization of cross-frame instance-pair query representations. Circles and triangles denote embeddings from frame t and frame t − 1, respectively, and colors indicate query slot indices. A line segment connects the same-index temporal query pair $( Q _ { t , q } ^ { i n s } , Q _ { t - 1 , q } ^ { i n s } )$ . With SSIA, same-slot embeddings form more compact clusters with shorter cross-frame segments, indicating improved temporal consistency of instance-pair query anchors. Without SSIA, same-slot embeddings are more dispersed and connected by longer, crossing segments, suggesting stronger cross-frame representational drift.

## S2.3 Temporal Grounding Consistency of Same-Index Queries

We next examine whether the representation-space alignment in Fig. S2 has a corresponding image-space efect. We use the SSIA-trained frame-wise spatial parser from Phase 1, before applying DTA, and run it independently on four consecutive annotated frames from each clip. No reference frame, temporal aggregation, tracking, IoU matching, or other cross-frame association is used. Thus, boxes with the same color in Fig. S3 are direct frame-wise predictions from the same query index. Following the person-centric annotations of Action Genome, we omit person boxes and visualize only object-side predictions for clarity. Since the subject side is the person in these visualized examples, object-side consistency corresponds to stable person–object candidate grounding.

Fig. S3 shows that the visualized same-index queries maintain object-side grounding across annotated frames under purely frame-wise inference. In the first clip, the same query indices consistently localize the sandwich and dish despite hand motion and object displacement. In the second clip, they remain grounded to clothes and the floor region across the frame sequence. This qualitative check complements Fig. S2 and supports the role of SSIA in stabilizing instance-pair query anchors before TADI computes same-index relation-query displacements.

![](images/61740e963f54fd7867a572381df18dc426db6f9d142c7cae229234dcd0457a2b.jpg)  
0J1BT.mp4  
Fig. S3: Temporal grounding of same-index queries. The SSIA-trained frame-wise parser is applied independently to each annotated frame, without temporal aggregation. Within each clip, the same color denotes the same query index. Person boxes are omitted for clarity.

## Supplementary References

1. Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., Zagoruyko, S.: End-to-end object detection with transformers. In: ECCV. pp. 213–229 (2020). https://doi.org/10.1007/978-3-030-58452-8\_13

2. Chen, M., Li, L., Wang, W., Yang, Y.: DIFFVSGG: Difusion-driven online video scene graph generation. In: CVPR. pp. 29161–29172 (2025)

3. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N.: An image is worth 16x16 words: Transformers for image recognition at scale. In: ICLR (2021)

4. Ji, J., Krishna, R., Fei-Fei, L., Niebles, J.C.: Action genome: Actions as compositions of spatio-temporal scene graphs. In: CVPR. pp. 10236–10247 (2020)

5. Le, H., Chung, N., Kieu, T., Yang, J., Le, N.: UNO: Unifying one-stage video scene graph generation via object-centric visual representation learning. In: WACV (2026)

6. Nag, S., Min, K., Tripathi, S., Roy-Chowdhury, A.K.: Unbiased scene graph generation in videos. In: CVPR. pp. 22803–22813 (2023)

7. Oquab, M., Darcet, T., Moutakanni, T., Vo, H.V., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al.: DINOv2: Learning robust visual features without supervision. TMLR (2024)

8. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., Polosukhin, I.: Attention is all you need. In: NeurIPS. pp. 5998–6008 (2017)

9. Wang, G., Li, Z., Chen, Q., Liu, Y.: OED: Towards one-stage end-to-end dynamic scene graph generation. In: CVPR. pp. 27938–27947 (2024)

10. Wang, L., Zhang, Y., Lu, T., Liu, F., Zhang, H., Ma, J., Zhou, H.: End-to-end entity-predicate association reasoning for dynamic scene graph generation. In: ICCV. pp. 17729–17738 (2025)