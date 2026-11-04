# TAQ-MOTR: Tracking-Aware Detection Query Formation for End-to-End Multiple Object Tracking

Hanyang Liu<sup>a</sup>, Hong Zhang<sup>a,c</sup>, Ding Yuan<sup>a,d</sup>, Yang Han<sup>b</sup>, Yifan Yang<sup>a,d,</sup>∗

<sup>a</sup>School of Astronautics, Beihang University, 37 Xueyuan Road, Haidian District, Beijing 100191, China

<sup>b</sup>School of Software, Beihang University, 37 Xueyuan Road, Haidian District, Beijing 100191, China <sup>c</sup>Key Laboratory of Spacecraft Design Optimization and Dynamic Simulation Technology, Ministry of Education

<sup>d</sup>State Key Laboratory of High-Eficiency Reusable Aerospace Transportation Technology

## Abstract

Tracking-by-propagation (TBP) methods perform end-to-end multi-object tracking by propagating track queries across frames, jointly handling detection and identity association without heuristic post-processing while naturally modeling temporal information. However, as detection and association are tightly coupled, detection queries in TBP are expected to discover currently untracked objects and provide complementary current-frame evidence for challenging tracks whose queries and identities are retained. Existing methods largely inherit detectionquery formation from single-frame detection and construct such queries before the decoder with little awareness of trajectory states. Consequently, currently untracked objects may be missed, while challenging tracked objects with reduced visibility under occlusion may lack complementary evidence, causing association failures or duplicate responses. To address these issues, we propose TAQ-MOTR, a tracking-aware detection query formation framework that employs a pre-decoder Hybrid Query Generator (HQG) to fuse current-frame image features, track memory, and learnable memory. By conditioning detection queries on trajectory states, TAQ-MOTR adapts them to the discovery of currently untracked objects and tracked-object complementation. We further introduce a training-only auxiliary detection branch whose one-to-one label assignment is decoupled from that of the main tracking branch, providing additional detection supervision to

strengthen the shared HQG and decoder for the discovery of currently untracked objects. TAQ-MOTR improves performance on DanceTrack, SportsMOT, and BFT, achieving 72.0 HOTA, 63.1 AssA, and 76.6 IDF1 on the DanceTrack test set. Code is available at https://github.com/Magicmiya/TAQ-MOTR.

Keywords: Multi-object tracking, End-to-end, Tracking-by-propagation, Query formation, Tracking-aware

## 1. Introduction

Multi-object tracking (MOT) aims to localize objects of interest in a video and preserve their identities across frames. It supports applications in autonomous driving, intelligent transportation, video surveillance [1, 2], and sports analytics [3]. A central challenge in MOT is to jointly handle object discovery and identity preservation under complex motion, occlusion, and appearance variation. Traditional methods usually follow the Tracking-by-Detection (TBD) paradigm, which first detects objects frame by frame and then associates detections over time. Although representative methods such as ByteTrack [4] are efective, their detectors are optimized mainly for frame-level recognition and are not explicitly conditioned on trajectory states, making detection errors directly afect association.

Inspired by DETR-style detectors [5], recent end-to-end MOT methods extend query-based set prediction to videos by propagating identity-preserving track queries across frames [6, 7, 8]. We refer to this family as Tracking-by-Propagation (TBP). Unlike DETR object queries, which detect all objects in a single image, TBP detection queries operate alongside propagated track queries that maintain existing trajectories. They therefore serve two roles: discovering currently untracked objects and supplying complementary current-frame evidence to challenging retained tracks. Here, currently untracked objects are those without a maintained track query, including newly appearing objects and previously observed objects that have not been initialized or whose tracks have been terminated. Temporarily lost objects with retained track queries remain tracked, even if no tracking result is output in the current frame. Under identity-preserving one-to-one assignment, ground-truth objects associated with propagated track queries are reserved for those queries, leaving detection queries with positive labels only from currently untracked objects. This imbalanced label assignment leaves the detection component insuficiently trained, contributing to the conflict between detection and association [9].

![](images/9232080dcd024a9c72271d23e365106cab51fe3ebc076b8a57553a6c012df884.jpg)  
Figure 1: Detection query formation before decoding. (a) MOTR: learnable queries; (b) MOTRv2: detector-guided queries; (c) TAQ-MOTR: tracking-aware queries. The comparison highlights currentframe information, trajectory-state priors, and learnable embeddings.

Existing TBP methods use diferent query-initialization strategies to support object discovery. MOTR [8] initializes detection queries with learnable content and positional embeddings (Fig. 1(a)). Building on learnable query embeddings, MOTRv2 [10] uses boxes and confidence scores from a pretrained external detector to initialize detection queries (Fig. 1(b)). This introduces current-frame information indirectly and reduces the tracker’s detection burden, but requires an additional detector. Neither strategy conditions detection query formation on existing trajectory states; distinguishing tracked from currently untracked objects still relies on subsequent interaction with track queries in the decoder.

Motivated by visual search studies showing that human visual attention is guided by task demands toward informative regions [11, 12], we seek to align detection query formation with current tracking needs. We therefore propose TAQ-MOTR, a tracking-aware detection query formation framework for end-toend MOT (Fig. 1(c)). Its Hybrid Query Generator (HQG) initializes proposals from image features and refines them through attention to learnable memory acquired from training data and track memory from previous-frame tracks. After scoring and Top-K selection, the resulting detection queries are aware of already tracked objects before bidirectional interaction with track queries in the decoder, helping focus detection on currently untracked objects.

Furthermore, we add a training-only auxiliary detection branch to improve detection-side learning under sparse supervision. It reuses the image-based detection proposals and shares HQG and decoder parameters with the main branch, but excludes track memory and blocks interaction with the main queries through attention masks. Its proposal and decoder predictions are independently matched to all current-frame ground-truth objects, preserving one-to-one assignment within each branch and supervising HQG and the decoder through their respective losses. This design supports joint optimization within the end-to-end tracker without an external detector or additional trainable parameters, and the auxiliary branch is removed at inference.

We evaluate TAQ-MOTR on DanceTrack [13], SportsMOT [3], and BFT [14], achieving the highest HOTA among the compared TBP methods on all three benchmarks. The association gains and competitive detection performance support the benefit of forming detection queries with awareness of existing track states.

The main contributions of this work are summarized as follows:

• We propose TAQ-MOTR to make detection query formation tracking-aware in TBP-based MOT. It introduces trajectory-state conditioning before decoder interaction, enabling detection queries to adapt to their shifted role in query propagation while retaining the unified TBP decoding framework.

• We design a Hybrid Query Generator (HQG) that integrates image feature proposals, track memory, and learnable memory to form tracking-aware detection queries before decoding, enabling them to focus on tracking-relevant regions for the discovery of currently untracked objects and complementary evidence provision.

• We introduce a training-only auxiliary detection branch with a stage-wise training strategy to provide additional detection supervision through decoupled label assignments between the auxiliary and main tracking branches. By strengthening detection query learning without competing with track queries for labels, this branch improves detection quality and further benefits identity association.

## 2. Related Work

## 2.1. Multi-object Tracking Paradigms

Multi-object tracking (MOT) methods can be broadly categorized by how object detection and temporal association are coupled. Tracking-by-Detection (TBD) first performs frame-wise detection and then associates detection results across frames. Representative trackers, such as ByteTrack [4], OC-SORT [15] and Hybrid-SORT [16], improve this pipeline by recovering low-confidence detections, refining motion modeling, or combining complementary association cues. Despite their efectiveness, the separated pipeline makes track initialization and identity continuity rely heavily on detector quality and hand-crafted matching. Missed or inaccurate detections may therefore lead to target loss, trajectory fragmentation, and identity switches.

Inspired by DETR [17], recent end-to-end MOT methods couple detection and association through query-based set prediction. TransTrack [6] uses target features from the previous frame as tracking queries, while TrackFormer [7] and MOTR [8] further develop query propagation for identity-preserving tracking. MeMOTR [18] further incorporates long-term memory into track-query updates to maintain identity consistency. In the TBP paradigm, propagated track queries and detection queries are jointly decoded in a shared decoder, enabling unified optimization of object discovery and identity association. However, this tight coupling also changes the role of detection queries. Beyond the discovery of currently untracked objects, detection queries are expected to provide complementary current-frame evidence for challenging tracked objects under occlusion, crossing, or overlap. This dual role makes detection query formation before shared decoding crucial for TBP-based MOT.

Another line of end-to-end MOT revisits the separation between object detection and identity association. LA-MOTR [19] decouples object detection and trajectory association into dedicated modules, while MOTIP [20] formulates association as in-context ID prediction over frame-level object representations. Although these methods reduce the entanglement between detection and association, their detection stage is largely driven by single-frame object evidence before identity reasoning. As a result, detection is less directly conditioned on propagated track states, leaving limited room for track memory to guide current-frame object discovery and complement challenging tracked objects. TAQ-MOTR therefore focuses on a diferent setting: it preserves the TBP formulation with a unified querypropagation decoder, and improves detection query formation before decoding by making detection queries tracking-aware.

## 2.2. Detection Query Formation in End-to-End MOT

End-to-end MOT methods such as MOTR [8] build on DETR-style object detectors [17]. In MOTR, detection queries are initialized with a fixed-size set of learnable content and positional embeddings. These embeddings acquire shared detection priors from the training data, providing a consistent initialization across frames. However, they are not conditioned on current-frame observations or existing track states. Detection queries therefore rely on subsequent interactions with image features and track queries in the decoder to localize objects and distinguish currently untracked objects from those already tracked. This lack of guidance at initialization can leave early decoder attention insuficiently focused on relevant object regions, limiting how efectively detection queries extract features for tracking. To incorporate current-frame object information, MOTRv2 [10] uses proposals from a pretrained external detector to initialize detection queries, but its proposal generation does not incorporate existing track states or undergo joint optimization with the tracker.

Contextual conditioning in related vision tasks provides further inspiration for query formation. SAM 2 [21] conditions current-frame image representations on object-specific memory for segmentation, with memory attention and mask decoding performed separately for each object. SAM2MOT [22], for example, adds a separate cross-object interaction module to coordinate memory updates under occlusion while retaining per-object segmentation. In object detection, RT-DETRv2 [23] selects scored image-based proposals to initialize decoder queries and jointly trains proposal generation with all-object supervision. Extending image-based proposal initialization to TBP also requires conditioning on existing trajectory states.

HQG extends image-based proposal initialization with track-state conditioning before joint MOT decoding, while preserving the unified TBP framework.

## 2.3. Query Competition and Detection-side Supervision

Supervision strategies in end-to-end MOT difer in how they balance detection learning with identity-preserving tracking.

MOTRv3 [9] redistributes labels between detection and track queries through release-fetch supervision, while CO-MOT [24] extends detection supervision to tracked objects in intermediate decoder layers and introduces shadow query sets. These strategies modify supervision within the main decoder. Our approach instead uses a separate auxiliary query group with independent all-object matching, drawing on query grouping and attention masking in DN-DETR [25]. This supplies additional detection supervision while preserving the main branch’s identitypreserving assignments.

## 3. Our Approach

## 3.1. Overall Architecture

TAQ-MOTR is an end-to-end MOT framework that forms tracking-aware detection queries before decoder interaction. As shown in Fig. 2, given an input frame $I _ { t } ,$ a backbone and an RT-DETR-style hybrid encoder extract multi-scale image features [26]. These features serve as visual context for the DAB-Deformable DETR decoder [27, 5], while the lowest-resolution feature map is projected into image-based detection proposals for the Hybrid Query Generator (HQG). In parallel, track queries $T _ { t - 1 }$ propagated from the previous frame are updated by the track-query updater. The updated propagated track queries are used as decoder queries for identity-preserving tracking and are also converted into track memory for HQG.

The main tracking branch is active during both training and inference. HQG takes the detection proposals as input queries and allows them to interact with two memory sources: track memory, which encodes existing trajectory states, and learnable memory, which stores trajectory-agnostic detection priors. The resulting detection queries therefore combine current-frame object evidence, track-state priors, and learned detection priors before entering the decoder. They are concatenated with the updated track queries and jointly decoded. High-confidence detection-query outputs are initialized as new track queries, while active propagated track queries are updated to continue existing trajectories. Following the missing-frame tolerance principle in MOTR [8] and MeMOTR [18], established tracks are retained during temporary loss until the configured missing-frame tolerance is reached. Retained track queries continue to participate in decoding; once a retained query again produces a reliable prediction, tracking resumes with the same identity. These objects remain tracked even when no current-frame output box is produced. Exclusion from HQG’s conditioning memory does not itself delete a retained track query or its identity. Once the tolerance is reached, the track query is deleted, and its ID is permanently retired and will not be restored or reassigned. A later detection of the same physical object therefore initializes a new track with a new identity. The newly initialized and retained track queries together form T<sub>t</sub> for propagation to the next frame.

![](images/8f5e30d3c699c26c095276ea56c53aceba5825c70a19af3007ca4523b520c0a8.jpg)  
Figure 2: Overall architecture of TAQ-MOTR. TAQ-MOTR forms tracking-aware detection queries before decoding. In the main tracking branch, HQG uses image-based detection proposals as queries and reads track memory and learnable memory as keys/values. The generated detection queries are concatenated with updated track queries and jointly decoded. During training, an auxiliary detection branch shares the detection proposals and HQG layers but removes track memory, producing a separate detection-query group supervised by all current-frame objects. Attention masks isolate this auxiliary group from the main query groups. The right panel shows an HQG layer.

For tracked-object complementation, HQG conditions detection queries on current-frame image evidence, track memory, and learnable memory. Following MOTR [8], retained track queries read detection-query features through decoder self-attention (SA) to complement the representations used for their predictions, while the existing identities remain assigned to the track queries.

The training-only auxiliary branch strengthens detection-side learning without changing the inference pipeline. It reuses the same detection proposals and HQGlayer weights as the main branch, but excludes track memory and reads only learnable memory. The auxiliary queries are decoded together with the main queries, isolated by attention masks, and supervised by all ground-truth objects in the current frame. They are not used for tracking prediction and are not propagated to subsequent frames.

![](images/d0d82f3c1438fa5f59929a1d6872f1b3df836f0574d7f5046da249fed576844b.jpg)  
Figure 3: Prior carriers used in HQG. Learnable memory provides learnable detection priors that are independent of specific trajectories. Track memory encodes existing trajectory states from propagated track queries, including content, position, and disappearance-time information. Detection proposals provide current-frame object evidence and coarse spatial priors.

## 3.2. Hybrid Query Generator

As discussed in Section 2.2, detection queries in TBP-based MOT do not simply inherit DETR’s full-image discovery role. Existing targets are mainly maintained by propagated track queries, whereas detection queries should discover currently untracked objects and provide complementary current-frame evidence for tracked targets under occlusion, crossing, or overlap. HQG addresses this trackingspecific role by organizing three prior carriers before decoding: learnable memory, track memory, and image-based detection proposals. Its position in the overall architecture is shown in Fig. 2, and the three prior carriers are decomposed row by row in Fig. 3.

Learnable memory. As shown in the top row of Fig. 3, learnable memory is a single set of trainable tokens shared across all HQG layers and the main and auxiliary branches. It learns trajectory-independent detection priors from the training dataset, preserving the prior role of learnable queries in DETR/MOTR-style models. The tokens provide cross-attention keys and values alongside track memory, with image-based detection proposals as queries. They remain available when track memory is empty and are reused across frames, but are neither directly decoded nor propagated as track states. We denote this shared memory as

$$
\mathbf { M } ^ { \mathrm { l e a r n } } \in \mathbb { R } ^ { N _ { l } \times d } ,\tag{1}
$$

where $N _ { l }$ is the number of tokens in the shared memory set and d is the embedding dimension. The same $\mathbf { M } ^ { \mathrm { l e a r n } }$ is used in every HQG layer.

Track memory. As shown in the middle row of Fig. 3, track memory carries trackstate priors from previous-frame trajectories. In the overall architecture of Fig. 2, it is derived from the updated propagated track queries and then used as part of the HQG key/value memory. Each track-memory token contains a content prior from the track-query feature, a spatial prior from the reference box, and a temporal-state prior from the disappearance-time index. Let $C _ { t } ^ { \mathrm { t r k } } , B _ { t } ^ { \mathrm { t r k } }$ , and $\Delta _ { t } ^ { \mathrm { t r k } }$ denote the track content embedding, reference box, and disappearance-time index, respectively. Track memory is formulated as

$$
\mathbf { M } _ { t } ^ { \mathrm { t r k } } = C _ { t } ^ { \mathrm { t r k } } + \mathrm { F o P E } ( B _ { t } ^ { \mathrm { t r k } } ) + \mathbf { E } ^ { \mathrm { t i m e } } [ \Delta _ { t } ^ { \mathrm { t r k } } ] ,\tag{2}
$$

where FoPE( ) is Fourier position embedding and $\mathbf { E } ^ { \mathrm { t i m e } }$ is a learnable temporal embedding table. Through $\mathbf { M } _ { t } ^ { \mathrm { t r k } }$ , detection proposals can condition query formation on existing trajectory states.

Detection proposals. As shown in the bottom row of Fig. 3, detection proposals provide current-frame object evidence and coarse spatial priors. In Fig. 2, they are generated from the lowest-resolution encoder feature map, which provides strong semantic cues with a low token resolution. The feature map is flattened and projected into proposal content embeddings $F _ { t } ^ { \mathrm { l o w } }$ . Regular grid anchors $A _ { t } ^ { 0 }$ are generated on the same feature map and encoded by Fourier position embedding. The initial detection proposals are defined as

$$
\widehat {  Ḋ \mathbf Ḋ \Gamma Ḍ Ḍ } ^ { \mathrm { d e t } } = F _ { t } ^ { \mathrm { l o w } } + \mathrm { F o P E } ( A _ { t } ^ { 0 } ) \in \mathbb { R } ^ { N _ { p } \times d } ,\tag{3}
$$

where $N _ { p } ~ = ~ H _ { \mathrm { l o w } } W _ { \mathrm { l o w } } , ~ H _ { \mathrm { l o w } }$ and $W _ { \mathrm { l o w } }$ are the height and width of the lowestresolution encoder feature map, and d is the embedding dimension. We define

$A _ { t } ^ { 0 } \in ( 0 , 1 ) ^ { N _ { p } \times 4 }$ as grid anchors in normalized $( c _ { x } , c _ { y } , w , h )$ coordinates; FoPE maps them to $N _ { p } \times d$ positional embeddings. $\widehat { \mathbf { D } } _ { t } ^ { \mathrm { d e t } }$ carries current-frame object evidence and coarse spatial priors and serves as the input query of HQG.

Tracking-aware query formation. In the main branch of Fig. 2, HQG uses $\widehat { \mathbf { D } } _ { t } ^ { \mathrm { d e t } }$ as queries and the concatenation of track memory and learnable memory as keys/values. Let $H _ { t } ^ { 0 } = \widehat { \mathbf { D } } _ { t } ^ { \mathrm { d e t } }$ and $\mathbf { M } _ { t } ^ { \mathrm { h y b } } = [ \mathbf { M } _ { t } ^ { \mathrm { t r k } } , \mathbf { M } ^ { \mathrm { l e a r n } } ]$ . Each HQG layer, illustrated in the right panel of Fig. 2, contains self-attention, cross-attention, and an FFN:

$$
\begin{array} { r l } & { \widetilde { H } _ { t } ^ { l } = \mathrm { S A } ( H _ { t } ^ { l } ) , } \\ & { } \\ & { H _ { t } ^ { l + 1 } = \mathrm { F F N } ( \mathrm { C A } ( \widetilde { H } _ { t } ^ { l } , \mathbf { M } _ { t } ^ { \mathrm { h y b } } ) ) , } \end{array}\tag{4}
$$

where $\mathbf { M } _ { t } ^ { \mathrm { h y b } }$ provides the cross-attention keys and values. Residual connections and normalization are omitted for clarity. After L HQG layers, the output is projected and scored:

$$
\begin{array} { r l } & { E _ { t } = \mathrm { P r o j } ( H _ { t } ^ { L } ) , } \\ & { S _ { t } = \psi _ { \mathrm { c l s } } ( E _ { t } ) , } \\ & { B _ { t } = \sigma ( \psi _ { \mathrm { b o x } } ( E _ { t } ) + \sigma ^ { - 1 } ( A _ { t } ^ { 0 } ) ) , } \\ & { Q _ { t } ^ { \mathrm { d e t } } = \mathrm { T o p K } ( E _ { t } , S _ { t } , B _ { t } ) . } \end{array}\tag{5}
$$

Here, $E _ { t }$ and $S _ { t }$ denote candidate embeddings and classification scores, and $B _ { t }$ contains normalized predicted boxes. $\sigma$ is the sigmoid function, and $\psi _ { \mathrm { { b o x } } }$ predicts residuals in inverse-sigmoid coordinates. TopK ranks candidates by $S _ { i }$ and returns K pairs of content embeddings and corresponding reference boxes, of shapes $K \times d$ and $K \times 4 .$ respectively. Conditioning before selection makes these queries tracking-aware.

Following RT-DETRv2 [23], only the selected K candidates participate in matching and direct supervision for the Top-K proposal loss; the remaining candidates are excluded from this loss and decoder initialization. The selected content embeddings and reference boxes are detached from the computation graph before decoder initialization, preventing decoder losses from backpropagating to HQG through these tensors.

## 3.3. Detection-only Auxiliary Learning

Although HQG conditions detection queries on track states, one-to-one assignment still limits their positive supervision. Ground-truth identities held by maintained track queries, including temporarily lost queries, are reserved for those queries. Main detection queries are matched only to currently untracked objects, whose identities are not held by any maintained track query. At clip initialization, they can match all current-frame objects. Auxiliary queries are independently matched to all current-frame objects at supervised layers when their branch is active on non-initial clip frames. This supervision imbalance weakens detectionside learning, making detection queries less efective in discovering currently untracked objects and in providing complementary current-frame evidence for challenging tracked targets.

To quantify this supervision imbalance, we analyze GT assignments throughout the 30-epoch DanceTrack training run. The track-query, main detection, auxiliary detection, and combined detection ratios are T/G, D/G, A/G, and $( D +$ $A ) / G$ , where G counts all sampled GT occurrences, T and A are logged positiveassignment counts, and $D \ = \ G - \ T$ reconstructs main detection supervision. The auxiliary branch is disabled on the first frame of every training clip, where $T = A = 0$ and main detection queries receive all GT labels; these frames remain in G.

In Fig. 4, main detection receives 51.80% and 27.59% of GT labels at epochs 0 and 14; excluding clip-initial frames gives 3.58% and 3.45%. Auxiliary Detection/GT stays below 100% because initial frames contribute to G but not A. On non-initial frames alone, auxiliary assignments cover 99.94%–99.98% of GT during epochs 0–23. Longer clips reduce the initialization-frame share, explaining the auxiliary curve’s stepwise increases. The combined ratio averages 102.01% over these epochs because labels are reused across branches, supplementing detection supervision without reallocating main-branch labels. These counts come from one run, not separately trained models.

To provide stable detection-side supervision, we introduce a training-only auxiliary detection branch, shown by the dotted auxiliary path in Fig. 2. It follows the query grouping and attention masking strategy of DN-DETR [25], but does not construct denoising queries from ground-truth annotations. Instead, it reuses the same image-based detection proposals as the main branch to form a separate detection-query group. The auxiliary branch shares the initial proposals $\widehat { \mathbf { D } } _ { t } ^ { \mathrm { d e t } }$ and HQG-layer weights with the main branch, but removes track memory from cross-

![](images/2cc58734246106b7fafeec9454ab4f17cb5fc9d777980b6f534acdb9435e133e.jpg)  
Figure 4: Positive label assignments during training on DanceTrack. All ratios use the total GT count per epoch, including clip-initial frames. The auxiliary branch is disabled on each clip’s first frame, so Auxiliary Detection/GT remains below 100% while enabled. Dashed lines mark clip-length changes; shading marks branch shutdown from epoch 24. The auxiliary-only ratio is then zero and omitted. Epoch numbering starts at 0.

attention:

$$
\begin{array} { r l } & { H _ { t , \mathrm { a u x } } ^ { 0 } = \widehat { \mathbf { D } } _ { t } ^ { \mathrm { d e t } } , } \\ & { H _ { t , \mathrm { a u x } } ^ { l + 1 } = \mathrm { H Q G B l o c k } ( H _ { t , \mathrm { a u x } } ^ { l } , \mathbf { M } ^ { \mathrm { l e a r n } } ) . } \end{array}\tag{6}
$$

Thus, the auxiliary queries retain current-frame image evidence and learnable detection priors while remaining independent of propagated track states.

As shown in Fig. 2, the auxiliary queries are decoded together with the main detection queries and track queries, but attention masks block information exchange between the auxiliary group and the main query groups. The main branch performs tracking prediction, track-query update, and query propagation. The auxiliary branch is used only for detection losses on non-initial clip frames when enabled by the training schedule: its outputs are independently matched with all ground-truth objects in the current frame, and its queries are discarded after training. Both branches share HQG and decoder parameters, including their prediction heads, without adding trainable parameters or competing with track queries for positive labels. This design strengthens current-frame object discovery and complementary evidence learning without increasing inference cost or disrupting the

unified TBP decoder.

Selected Top-K proposals receive Varifocal Loss (VFL) for classification and L1 and generalized IoU (GIoU) losses for box regression. Main-branch proposals are matched to currently untracked ground-truth objects; active auxiliary proposals are independently matched to all current-frame ground-truth objects. These losses train the proposal heads and preceding HQG layers without diferentiating through the ranking indices. The selected-proposal losses directly supervise HQG, whereas decoder-level losses optimize the decoder without backpropagating through the detached query-initialization tensors. Proposal supervision is retained throughout training.

## 4. Experiments

## 4.1. Benchmarks and Metrics

We use three benchmarks: DanceTrack [13], SportsMOT [3], and Bird Flock Tracking (BFT) [14]. DanceTrack features similar appearances, crossings, and occlusions; SportsMOT introduces fast motion, viewpoint changes, and small targets in wide-view scenes. BFT extends evaluation to non-human targets with rapid motion, deformation, and occlusion. We report HOTA as the primary overall metric, DetA and AssA for detection and association quality [28], IDF1 for identity preservation [29], and MOTA for frame-level tracking errors [30]. Best and second-best results are highlighted among Tracking-by-Propagation (TBP) methods, which propagate track queries across frames.

## 4.2. Implementation Details

TAQ-MOTR follows the MeMOTR framework [18] for query propagation and long-term memory. The model uses RT-DETRv2 weights pretrained on COCO [23, 31], a PResNet-50vd backbone, and the RT-DETRv2 Hybrid Encoder. The decoder follows DAB-Deformable DETR [27, 5] with six layers and takes three feature levels with strides of 8, 16, and 32. HQG constructs imagebased detection proposals from the stride-32 feature level. The first decoder layer updates only detection-side queries, while the remaining layers jointly decode detection queries and propagated track queries. Unless otherwise specified, HQG uses 200 main detection queries and 200 detection-only auxiliary queries. HQG contains three layers that share a set of 32 learnable memory tokens.

Training uses NVIDIA GeForce RTX 4090 GPUs. DanceTrack is the main benchmark for selecting the default configuration and conducting ablations, and its default model is trained for 30 epochs on four GPUs. We adopt a progressive training curriculum based on MeMOTR [18]: training clips are sampled with random temporal intervals of up to 10 frames, and the clip length gradually increases from 2 to 5. Within this curriculum, we remove intermediate-layer auxiliary decoder losses at epoch 14 and disable the auxiliary branch at epoch 24. SportsMOT uses the same architecture with a 42-epoch schedule; the entire auxiliary branch is disabled at epoch 36, the beginning of the final training stage. BFT uses the same architecture with a 50-epoch schedule, with training stages and auxiliarysupervision transitions proportionally mapped from the DanceTrack schedule.

For tracker training in the main comparisons, TAQ-MOTR uses only the oficial training split of each benchmark; COCO detector pretraining is specified above and is distinct from additional tracking-training data. We additionally report a SportsMOT variant trained with additional CrowdHuman data, marked with <sup>‡</sup> and excluded from the main ranking. TAQ-MOTR performs online tracking without ofline trajectory linking, smoothing, or interpolation. Outputs undergo per-frame confidence filtering.

## 4.3. Comparison with State-of-the-Art Methods

Accuracy results for other methods follow the cited publications or oficial releases. We distinguish additional data for tracking-model training (<sup>‡</sup>) from additional models (<sup>†</sup>) and identify train+val variants explicitly. TBDQ-Net is grouped with TBP because it propagates track queries, despite using an external detector. The MOTIP rows use the published CVPR 2025 version, and SambaMOTR uses the published Deformable-DETR variant. Training stages and optimization schedules follow each method’s architecture and learning objective; inference eficiency is measured under common conditions specified below.

Results on DanceTrack. In Table 1, TAQ-MOTR achieves 72.0 HOTA, 82.4 DetA, 63.1 AssA, 91.7 MOTA, and 76.6 IDF1, improving over MeMOTR by 3.5, 1.9, 4.7, 1.8, and 5.4 points, respectively. It also exceeds CO-MOT by 2.6 HOTA and

Table 1: Comparison with state-of-the-art methods on the DanceTrack test set. Best and second-best results among Tracking-by-Propagation (TBP) methods are marked in bold and underlined, respectively. Tracking-by-Detection-like methods are listed for reference only. <sup>‡</sup> denotes additional data for tracking-model training, and <sup>†</sup> denotes additional models. Train+val results are excluded from ranking.
<table><tr><td colspan="6">Method HOTA↑ DetA↑ AssA ↑ MOTA↑ IDF1 ↑</td></tr><tr><td colspan="6">Tracking-by-Detection-like</td></tr><tr><td>Hybrid-SORT-ReID† [16] (AAAI 2024)</td><td>65.7</td><td></td><td></td><td>91.8</td><td>67.4</td></tr><tr><td>DiffMOT† [32] (CVPR 2024)</td><td>62.3</td><td>82.5</td><td>47.2</td><td>92.8</td><td>63.0</td></tr><tr><td>PlugTrack† [33] (AAAI 2026)</td><td>63.3</td><td>82.5</td><td>48.4</td><td>92.4</td><td>64.1</td></tr><tr><td>MOTIP [20] (CVPR 2025)</td><td>69.6</td><td>80.4</td><td>60.4</td><td>90.6</td><td></td></tr><tr><td>OA-SORT† [34] (CVPR 2026)</td><td>63.1</td><td>一</td><td>48.5</td><td>91.7</td><td>74.7 64.2</td></tr><tr><td colspan="6">Tracking-by-Propagation (TBP)</td></tr><tr><td>MOTR [8] (ECCV 2022)</td><td>54.2</td><td>73.5</td><td>40.2</td><td>79.7</td><td>51.5</td></tr><tr><td>MOTRv2*†‡ [10] (CVPR 2023)</td><td>69.9</td><td>83.0</td><td>59.0</td><td>91.9</td><td>71.7</td></tr><tr><td>MeMOTR [18] (ICCV 2023)</td><td>68.5</td><td>80.5</td><td>58.4</td><td>89.9</td><td>71.2</td></tr><tr><td>SambaMOTR [35] (ICLR 2025)</td><td>67.2</td><td>78.8</td><td>57.5</td><td>88.1</td><td>70.5</td></tr><tr><td>TBDQ-Net† [36] (PR 2026)</td><td>69.4</td><td>80.8</td><td>59.8</td><td>89.5</td><td>73.4</td></tr><tr><td>CO-MOT‡ [24] (ICLR 2025)</td><td>69.4</td><td>82.1</td><td>58.9</td><td>91.2</td><td>71.9</td></tr><tr><td>TGFormer [37] (AAAI 2025)</td><td>69.1</td><td>81.9</td><td>58.3</td><td>91.3</td><td>71.9</td></tr><tr><td>DecoderTracker‡ [38] (PR 2026,</td><td>60.8</td><td>81.2</td><td>47.1</td><td>91.1</td><td>62.9</td></tr><tr><td>train+val)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TAQ-MOTR (Ours)</td><td>72.0</td><td>82.4</td><td>63.1</td><td>91.7</td><td>76.6</td></tr></table>

4.2 AssA, supporting the benefit of tracking-aware query formation alongside auxiliary supervision. Compared with MOTRv2, gains of 2.1 HOTA, 4.1 AssA, and 4.9 IDF1 accompany slightly lower DetA and MOTA, indicating stronger association with competitive detection performance.

MOTRv2 and CO-MOT use CrowdHuman in addition to DanceTrack train; the listed DecoderTracker variant also uses DanceTrack val and is excluded from ranking. MOTRv2 obtains query priors from an external YOLOX detector, while TAQ-MOTR uses only DanceTrack train for tracker training without an additional detector. We report MOTRv2’s 69.9-HOTA setting; its 73.4-HOTA variant additionally uses association post-processing, train+val training, and test-time ensembling [10].

Table 2: Comparison with state-of-the-art methods on the SportsMOT test set. Best and second-best results among Tracking-by-Propagation (TBP) methods are marked in bold and underlined, respectively. Tracking-by-Detection-like methods are listed for reference only. <sup>‡</sup> denotes additional data for tracking-model training, and <sup>†</sup> denotes additional models. Our mixed variant uses additional Crowd-Human data and is excluded from ranking.
<table><tr><td colspan="6">Method HOTA↑ DetA↑ AssA↑ MOTA↑ IDF1 ↑</td></tr><tr><td>Tracking-by-Detection-like</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DiffMOT† [32] (CVPR 2024)</td><td>72.1</td><td>86.0</td><td>60.5</td><td>94.5</td><td>72.8</td></tr><tr><td>MOTIP [20] (CVPR 2025)</td><td>72.6</td><td>83.5</td><td>63.2</td><td>92.4</td><td>77.1</td></tr><tr><td>ETTrack† [39] (AI 2025)</td><td>72.2</td><td>86.9</td><td>60.1</td><td>94.9</td><td>72.5</td></tr><tr><td>FDTA† [40] (CVPR 2026)</td><td>74.2</td><td>84.1</td><td>65.5</td><td>93.0</td><td>78.5</td></tr><tr><td colspan="6">Tracking-by-Propagation (TBP)</td></tr><tr><td>MeMOTR [18] (ICCV 2023)</td><td>70.0</td><td>83.1</td><td>59.1</td><td>91.5</td><td>71.4</td></tr><tr><td>SambaMOTR [35] (ICLR 2025)</td><td>69.8</td><td>82.2</td><td>59.4</td><td>90.3</td><td>71.9</td></tr><tr><td>TBDQ-Net‡ [36] (PR 2026)</td><td>72.1</td><td>82.8</td><td>62.8</td><td>89.5</td><td>72.9</td></tr><tr><td>TAQ-MOTR (Ours, train)</td><td>72.3</td><td>82.3</td><td>63.6</td><td>92.0</td><td>75.6</td></tr><tr><td>TAQ-MOTR‡ (Ours, mixed)</td><td>74.6</td><td>84.9</td><td>65.6</td><td>94.7</td><td>77.8</td></tr></table>

Results on SportsMOT. As shown in Table 2, TAQ-MOTR achieves 72.3 HOTA, 82.3 DetA, 63.6 AssA, 92.0 MOTA, and 75.6 IDF1. Compared with MeMOTR, it improves HOTA, AssA, and IDF1 by 2.3, 4.5, and 4.2 points, respectively, while

DetA is 0.8 points lower. These results indicate stronger identity association under rapid motion and dense interactions, with room to improve detection.

DifMOT, ETTrack, and TBDQ-Net use YOLOX detections or proposal priors; FDTA uses Video Depth Anything for pseudo-depth supervision during training. Our main result uses only SportsMOT train and no additional model for detection or supervision. For reference, we also report a variant trained with additional CrowdHuman data, reaching 74.6 HOTA and 77.8 IDF1. This variant is excluded from the main ranking.

Results on BFT. To align our benchmark coverage with recent methods such as MOTIP [20] and SambaMOTR [35], we additionally evaluate TAQ-MOTR on BFT. Table 3 shows that TAQ-MOTR achieves the best results on all five metrics among the compared TBP methods, exceeding SambaMOTR by 3.8 HOTA, 4.7 AssA, and 3.7 IDF1. These gains indicate stronger association and identity preservation for non-human targets with complex motion. The SORT, ByteTrack, OC-SORT, and TrackFormer results are reported by NetTrack [14]; the first three use YOLOX detections. FDTA uses additional depth supervision.

Table 3: Comparison with state-of-the-art methods on the BFT test set. Best and second-best results among Tracking-by-Propagation (TBP) methods are marked in bold and underlined, respectively. Tracking-by-Detection-like methods are listed for reference only. <sup>‡</sup> denotes additional data for trackingmodel training, and <sup>†</sup> denotes additional models.
<table><tr><td colspan="6">Method HOTA↑ DetA↑ AssA↑ MOTA↑ IDF1 ↑</td></tr><tr><td>Tracking-by-Detection-like</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SORT† [41] (ICIP 2016)</td><td>61.2</td><td>60.6</td><td>62.3</td><td>75.5</td><td>77.2</td></tr><tr><td>ByteTrack† [4] (ECCV 2022)</td><td>62.5</td><td>61.2</td><td>64.1</td><td>77.2</td><td>82.3</td></tr><tr><td>OC-SORT‡ [15] (CVPR 2023)</td><td>66.8</td><td>65.4</td><td>68.7</td><td>77.1</td><td>79.3</td></tr><tr><td>MOTIP [20] (CVPR 2025)</td><td>70.5</td><td>69.6</td><td>71.8</td><td>77.1</td><td>82.1</td></tr><tr><td>FDTA† [40] (CVPR 2026)</td><td>72.2</td><td>70.1</td><td>74.5</td><td>78.2</td><td>84.2</td></tr><tr><td colspan="6">Tracking-by-Propagation (TBP)</td></tr><tr><td>TrackFormer [7] (CVPR 2022)</td><td>63.3</td><td>66.0</td><td>61.1</td><td>74.1</td><td>72.4</td></tr><tr><td>SambaMOTR [35] (ICLR 2025)</td><td>69.6</td><td>66.0</td><td>73.6</td><td>72.0</td><td>81.9</td></tr><tr><td>TAQ-MOTR (Ours)</td><td>73.4</td><td>69.0</td><td>78.3</td><td>76.7</td><td>85.6</td></tr></table>

Eficiency Analysis. Eficiency is measured using the open-source implementations on a single NVIDIA GeForce RTX 4090 GPU, with sequential frame-byframe inference and batch size 1. All methods use the same input resizing protocol: a target short side of 768 pixels and a maximum long side of 1536 pixels. FPS measures model inference time only, excluding preprocessing and post-processing. MOTRv2’s reported 16.1 FPS includes both YOLOX and tracker inference.

Table 4: Eficiency comparison with open-source TBP methods on DanceTrack. All methods follow the same measurement protocol. MOTRv2 includes YOLOX in both its parameter count and inference time.
<table><tr><td>Method</td><td>Parameters (M)↓</td><td>FPS↑</td></tr><tr><td>MOTR [8]</td><td>43.9</td><td>21.0</td></tr><tr><td>MOTRv2 [10] (with YOLOX)</td><td>140.6</td><td>16.1</td></tr><tr><td>CO-MOT [24]</td><td>41.7</td><td>25.1</td></tr><tr><td>MeMOTR [18]</td><td>42.9</td><td>22.5</td></tr><tr><td>TAQ-MOTR</td><td>48.9</td><td>41.1</td></tr></table>

As shown in Table 4, TAQ-MOTR reaches 41.1 FPS with 48.9M parameters. It runs faster than the compared TBP implementations. Its parameter count is slightly higher than those of MOTR, CO-MOT, and MeMOTR, but lower than that of MOTRv2 with YOLOX. The compact hybrid encoder contributes to this eficiency.

![](images/9e03e321eb9ecadec65230864fa5d6022ceb61a4ad9b0428cc614bbdec8c9494.jpg)  
Figure 5: Module-level eficiency analysis of TAQ-MOTR. Rounded parameter counts and reported per-frame runtimes are shown for the backbone, hybrid encoder, decoder, HQG, and query updater.

Fig. 5 shows that HQG accounts for 3.4M parameters and 3.1 ms per frame. Most computation remains in feature extraction and decoder processing, indicating that tracking-aware query formation adds limited overhead.

## 4.4. Ablation Study and Analysis

We conduct ablation studies on the DanceTrack validation set to validate the main design choices of TAQ-MOTR. HOTA is used as the primary overall metric, while DetA, AssA, and IDF1 are reported to analyze detection accuracy, association quality, and identity preservation, respectively. Unless otherwise specified, all variants use the same training schedule and inference protocol as the default model, and the best result of each metric is shown in bold.

Table 5: Ablation of HQG priors on the full DanceTrack validation set. DP, TM, and LM denote detection proposals, track memory, and learnable memory, respectively. Original reports the published MeMOTR results [18]. Unified basic model adopts the same backbone, encoder, and pretrained weights as TAQ-MOTR while retaining the rest of MeMOTR’s tracking design. The best result for each metric is shown in bold, and unreported values are indicated by a dash.
<table><tr><td>DP TM LM HOTA↑ DetA↑ AssA↑ IDF1↑ DetRe ↑ IDSW↓</td></tr><tr><td></td></tr><tr><td>Baselines 55.0 67.1</td></tr><tr><td>Original 63.9 74.6 Unified basic model 62.9 76.0</td></tr><tr><td>52.3 66.1 85.4 1530</td></tr><tr><td>HQG variants</td></tr><tr><td>V 58.1 71.9 47.1 61.9 一 一</td></tr><tr><td>√√ 63.3 74.1 54.2 68.0 83.7</td></tr><tr><td>一 996 √ √ 一 64.4 77.1 53.9 69.1 一</td></tr><tr><td>√√ V 65.3 76.5 56.0 70.3 86.1 1038</td></tr><tr><td></td></tr></table>

Prior carriers in HQG. Relative to DP+LM, introducing track memory improves HOTA, AssA, and IDF1 by 0.9, 2.1, and 1.2 points, respectively, with a 0.6-point decrease in DetA. These results highlight the detection–association tradeof introduced by track-state conditioning.

In the Baselines section of Table 5, Original reports the published MeMOTR results [18], whereas Unified basic model reports the results of a MeMOTR control using the same backbone, encoder, and pretrained weights as TAQ-MOTR. The underlying feature-extraction architecture and pretraining are aligned, with corresponding adaptations to the training settings for the MeMOTR baseline. The comparison therefore provides an additional reference for assessing HQG rather than fully isolating its contribution. Compared with Original, Unified basic model improves DetA from 74.6 to 76.0 but reduces AssA from 55.0 to 52.3 and IDF1 from 67.1 to 66.1. As discussed in FDTA [40], object embeddings optimized solely for detection tend to have weaker discriminative ability for association. In our setting, the backbone and encoder initialized with RT-DETRv2 detection weights may favor detection cues, potentially contributing to the observed improvement in DetA and decline in association.

We additionally report detection recall (DetRe) and identity switches (IDSW) to assess missed detections and identity consistency. Relative to Unified basic model, the full HQG configuration improves HOTA, DetA, AssA, and IDF1 by 2.4, 0.5, 3.7, and 4.2 points, respectively. It also increases DetRe from 85.4 to 86.1 and reduces IDSW from 1,530 to 1,038. These results support improved overall target recall and identity association, without separately establishing a gain in recall for currently untracked objects. However, compared with DP+TM, the full configuration improves DetRe and IDF1 while increasing IDSW from 996 to 1,038, indicating a tradeof between recall and identity switches.

Table 6: Ablation of feature levels for constructing image-based detection proposals on the DanceTrack validation set.
<table><tr><td colspan="5">Level HOTA↑ DetA ↑ AssA ↑ IDF1↑</td></tr><tr><td>Stride 8</td><td>59.5</td><td>66.7</td><td>53.2</td><td>68.2</td></tr><tr><td>Stride 16</td><td>61.2</td><td>68.6</td><td>54.7</td><td>68.9</td></tr><tr><td>Stride 16+32</td><td>64.8</td><td>74.7</td><td>56.4</td><td>70.0</td></tr><tr><td>Stride 32 (Ours)</td><td>65.3</td><td>76.5</td><td>56.0</td><td>70.3</td></tr></table>

Image-based detection proposals. Table 6 compares proposal feature levels. The stride-32 setting has the highest reported HOTA, DetA, and IDF1; stride-16+32 has the highest AssA but requires additional computation. We use the compact stride-32 representation as the default. The diferences suggest that denser proposal features do not necessarily improve overall tracking.

Track memory. Table 7 compares active-only HQG memory with memory that also retains recently lost tracks. The active-only default gives the highest HOTA,

Table 7: Ablation of retaining recently lost tracks in HQG track memory.
<table><tr><td rowspan="2">Retain lost tracks</td><td colspan="3"></td><td rowspan="2">IDF1 ↑</td></tr><tr><td>HOTA↑</td><td>DetA ↑</td><td>AssA ↑</td></tr><tr><td>0 frames (Ours)</td><td>65.3</td><td>76.5</td><td>56.0</td><td>70.3</td></tr><tr><td>≤1 frame</td><td>61.7</td><td>78.3</td><td>48.8</td><td>63.1</td></tr><tr><td>≤3 frames</td><td>63.9</td><td>77.9</td><td>52.5</td><td>66.3</td></tr><tr><td>≤5 frames</td><td>63.7</td><td>77.5</td><td>52.5</td><td>66.4</td></tr></table>

AssA, and IDF1, while including lost tracks improves DetA. Temporarily lost queries still remain in the decoder and retain their identities. These results favor active track states for HQG conditioning; the additional historical cues may aid detection while reducing association quality.

Table 8: Efect of memory type and shared-set token count in HQG on the DanceTrack validation set. All HQG layers use the same memory set. Fixed random memory uses frozen, randomly initialized embeddings.
<table><tr><td>Tokens (shared set)</td><td>HOTA↑</td><td>DetA ↑</td><td>AssA↑</td><td>IDF1 ↑</td></tr><tr><td>0 (None)</td><td>63.3</td><td>74.1</td><td>54.2</td><td>68.0</td></tr><tr><td>4 (Learnable)</td><td>64.2</td><td>74.6</td><td>55.3</td><td>70.2</td></tr><tr><td>32 (Fixed random)</td><td>64.6</td><td>74.7</td><td>56.0</td><td>69.6</td></tr><tr><td>32 (Learnable, Ours)</td><td>65.3</td><td>76.5</td><td>56.0</td><td>70.3</td></tr><tr><td>64 (Learnable)</td><td>65.7</td><td>76.2</td><td>56.6</td><td>70.8</td></tr></table>

Learnable memory. Inspired by the learnable detection queries in MeMOTR [18], HQG uses a shared set of trainable memory tokens across all layers to encode dataset-level priors. Table 8 shows that using a shared set of 32 tokens improves HOTA from 63.3 to 65.3. Increasing the count to 64 yields further gains of 0.4 HOTA, 0.6 AssA, and 0.5 IDF1, but reduces DetA by 0.3. We select 32 as an eficiency/performance compromise, retaining competitive accuracy with fewer tokens participating in cross-attention.

To examine memory learnability, we add a fixed random memory control with the same shared set of 32 token slots, embedding dimensions, and attention operations as the learnable variant. Compared with this control, learnable memory achieves gains of 0.7 HOTA, 1.8 DetA, and 0.7 IDF1, while both obtain 56.0 AssA. These results support the contribution of learned detection priors. Freezing the embeddings reduces the number of trainable parameters while keeping the memory representation size unchanged. Fixed memory also improves HOTA and IDF1 by 1.3 and 1.6 points over using no memory, suggesting that the network can learn to use these tokens even when their embeddings are fixed. Learning the embeddings further allows the memory itself to adapt to the dataset. More efective memory representations and learning strategies remain directions for future work.

Table 9: Ablation of the number of HQG layers.
<table><tr><td>Layers</td><td>HOTA ↑</td><td>DetA ↑</td><td>AssA ↑</td><td>IDF1↑</td></tr><tr><td>1</td><td>63.7</td><td>76.8</td><td>52.9</td><td>68.4</td></tr><tr><td>2</td><td>64.1</td><td>76.1</td><td>54.3</td><td>69.0</td></tr><tr><td>3 (Ours)</td><td>65.3</td><td>76.5</td><td>56.0</td><td>70.3</td></tr><tr><td>6</td><td>63.7</td><td>75.2</td><td>54.1</td><td>68.2</td></tr></table>

Number of HQG layers. Table 9 compares HQG depth. Three layers yield the highest HOTA, AssA, and IDF1, whereas one layer gives the highest DetA. Six layers reduce overall performance. We therefore use three layers as the default.

Table 10: Auxiliary supervision schedules on the DanceTrack validation set. Parenthesized epochs indicate when intermediate-layer auxiliary decoder losses are removed and when the entire auxiliary branch is disabled, respectively. None denotes training without auxiliary supervision.
<table><tr><td>Auxiliary schedule</td><td>HOTA↑</td><td>DetA ↑</td><td>AssA ↑</td><td>IDF1↑</td></tr><tr><td>None (0, 0)</td><td>64.2</td><td>74.8</td><td>55.2</td><td>69.3</td></tr><tr><td>Staged (8, 24)</td><td>64.6</td><td>75.3</td><td>55.7</td><td>69.0</td></tr><tr><td>Staged (14, 24) (Ours)</td><td>65.3</td><td>76.5</td><td>56.0</td><td>70.3</td></tr><tr><td>Staged (21, 24)</td><td>64.8</td><td>76.7</td><td>55.6</td><td>69.1</td></tr><tr><td>Staged (24, 24)</td><td>65.0</td><td>77.0</td><td>54.9</td><td>68.6</td></tr><tr><td>Staged (14, 21)</td><td>64.4</td><td>75.2</td><td>55.3</td><td>70.1</td></tr><tr><td>Staged (14, 27)</td><td>64.9</td><td>76.7</td><td>55.4</td><td>67.7</td></tr></table>

Detection-only auxiliary learning. The supervision imbalance in Fig. 4 motivates additional detection supervision without changing the main branch’s association label assignment. Table 10 shows that our default auxiliary schedule improves HOTA, DetA, AssA, and IDF1 over training without the branch by 1.1, 1.7, 0.8, and 1.0 points, respectively. The timing of intermediate-layer loss removal and complete auxiliary-branch shutdown also afects the detection–association balance. Relative to the default, removing intermediate-layer losses earlier reduces DetA by 1.2 points, whereas delaying their removal increases DetA by 0.2–0.5 points but lowers AssA by 0.4–1.1 points. Advancing complete branch shutdown reduces DetA by 1.3 points; postponing it yields only a 0.2-point DetA gain while lowering IDF1 by 2.6 points. These results support suficient auxiliary supervision during early and middle training, followed by its reduction and removal to focus late optimization on the main MOT objective.

Tracked-object complementation. To examine the role of detection-query information in existing-track predictions, we evaluate Unified basic model and TAQ-MOTR on the DanceTrack validation set. After normal initialization in the first frame, two branches receive identical current-frame inputs and incoming tracking states: one retains normal SA, while the other blocks track queries from attending to detection-query keys and values. Predictions for incoming track queries are collected before lifecycle management and combined with the normal branch’s new-track initializations for evaluation. Only the normal branch updates and propagates track queries, memory, and lifecycle states, including track initialization, retention, and termination. The blocked branch does not modify this state, ensuring identical tracking histories for both branches at every frame. Normal-branch results are reevaluated under this protocol, separately from the HQG ablation in Table 5.

Table 11: Self-attention masking analysis of tracked-object complementation. T D SA denotes track queries attending to detection-query keys and values: $\checkmark$ retains this interaction, whereas – blocks it. Both branches share the normal branch’s tracking history and new-track initializations.
<table><tr><td colspan="5">Model  $T \gets D \operatorname { S A }$  HOTA↑ DetA ↑ AssA ↑ IDF1 ↑</td></tr><tr><td>Unified basic model</td><td>46.4</td><td>51.5</td><td>42.3</td><td>55.8</td></tr><tr><td>Unified basic model √</td><td>63.3</td><td>76.2</td><td>52.8</td><td>66.7</td></tr><tr><td>TAQ-MOTR</td><td>47.7</td><td>51.7</td><td>43.9</td><td>56.5</td></tr><tr><td>TAQ-MOTR</td><td>√ 65.4</td><td>76.5</td><td>56.0</td><td>70.3</td></tr></table>

Table 11 provides diagnostic evidence that detection-query information supports existing-track predictions under the fixed-history protocol. Blocking this interaction reduces TAQ-MOTR’s DetA from 76.5 to 51.7, alongside drops of 12.1 AssA and 13.8 IDF1 points, reflecting changes in both detection quality and identity consistency. Unified basic model also degrades. Because inferencetime masking alters the attention-input distribution, these reductions do not isolate HQG’s independent contribution. Fig. 7 illustrates where detection queries gather image evidence.

## 4.5. Visual Analysis

Tracking Results Visualization. Fig. 6 compares detection-query formation strategies on crowded DanceTrack clips with occlusion, crossing, and target overlap. These conditions challenge propagated track queries and motivate complementary current-frame evidence. Image-based detection queries use RT-DETR-style selection from current-frame encoder features without tracking-aware interaction.

In the first row, heavy occlusion around frame #2046 causes image-based detection queries to lose the target, while learned queries assign a diferent identity. Image-based detection queries subsequently recover the more visible target with an inconsistent ID. This sequence highlights the dificulty of maintaining identity from current-frame visual evidence alone when the local region remains ambiguous. The second row shows identity switches between interacting targets, with learned queries losing Target 4 during crossing. TAQ-MOTR maintains the queried targets in the tracked state across these frames. These examples illustrate more stable association and are consistent with the intended role of tracking-aware query formation, but do not isolate the mechanism responsible for the improvement.

![](images/0e9a5fbeb2760b18a39282884b545e8ec7af79854e0808e75804426f3ed89321.jpg)  
Figure 6: Qualitative comparison of detection-query formation strategies on DanceTrack. Green boxes denote ground truth. Yellow, red, and blue boxes denote tracking outputs from three queryformation variants: image-based detection queries without HQG, learned queries, and HQG-generated queries (TAQ-MOTR), respectively. The status tables report matched IDs and tracking states: “Lost” denotes no matched result, “IDSW” an identity switch, and “track” correct tracking. In the illustrated frames, TAQ-MOTR preserves the queried identities through occlusion, crossing, and overlap; this comparison provides qualitative evidence of association behavior.

Query Focus Visualization. Fig. 7 complements the quantitative SA masking experiment in Table 11 by showing where detection queries gather current-frame image evidence during initialization, discovery of currently untracked objects, and tracking under occlusion or interaction.

In Fig. 7(a)–(b), the first two video frames, detection queries initially provide broad object evidence before reliable track states are established. Learned queries retain similar focus after propagated track queries take over targets, while imagebased detection queries mainly follow visual saliency. Once track states become available, TAQ-MOTR focuses more on the crowded central region with overlap and occlusion while retaining coverage of visible targets. In the dense scene of Fig. 7(c)–(e), learned queries remain difuse, and image-based detection queries follow visual saliency. TAQ-MOTR attends to a locally ambiguous region, shifts toward the red-highlighted target in (d), and refocuses on other crowded regions after that target acquires an established output in (e). These patterns are consistent with detection-query focus adapting to current-frame evidence and existing track states, supporting discovery of currently untracked objects and complementary cues for retained tracks. The maps show where image cues are gathered; the SA masking experiment examines how existing-track predictions depend on that information under a fixed history. Neither identifies individual restored detections. The recall claim remains based on the overall DetRe evaluation.

<sup></sup> Image-based detection queries (w/o HQG) <sup></sup> Learned queries (Baseline)  
![](images/1ad52b093480bff63642827b3a4e8ce84ae89a34fb429024f3afc9c905d44580.jpg)  
<sup></sup> TAQ-MOTR (Ours)  
Figure 7: Query-focus visualization of detection queries on DanceTrack. The heatmaps aggregate cross-attention between detection queries and multi-scale image features in the first decoder layer; each panel is independently normalized, so heatmap colors indicate relative spatial focus rather than absolute attention magnitudes. Green boxes mark established tracking outputs and red boxes missing or unstable outputs; these colors do not indicate track-query retention. Panels (a)–(b) show video initialization, and (c)–(e) show target discovery and challenging tracking in a crowded scene. TAQ-MOTR attends to missing or unstable output regions and crowded regions where existing tracks may require additional current-frame evidence, complementing the SA masking analysis in Table 11.

## 5. Limitations

HQG does not explicitly estimate or gate prior reliability. Under dense overlap and similar appearance, image proposals may become ambiguous and propagated tracks may drift or confuse identities. The lost-track ablation in Table 7 shows that adding uncertain track states can improve detection but degrade association and identity preservation. Improving robustness to unreliable priors remains an open direction.

The compact stride-32 proposals in Table 6 may also limit sensitivity to small targets, local overlaps, and abrupt shape changes. Selective use of finer multi-scale evidence could address this limitation while preserving eficient query formation.

Finally, auxiliary supervision relies on a coarse, manually calibrated schedule whose transition points influence the balance between detection and association performance. Learning to adapt the strength and duration of auxiliary supervision to the model’s training progress remains an important direction for future work.

## 6. Conclusion

TAQ-MOTR forms detection queries from current-frame image evidence, track memory, and learnable priors before shared decoding. A training-only auxiliary branch strengthens detection learning through independent label assignments while preserving identity-based assignment in the main branch. Experiments on DanceTrack, SportsMOT, and BFT show improved association and identity preservation, with the highest HOTA among the compared TBP methods. Ablations support tracking-aware query formation and staged auxiliary supervision.

The results motivate conditioning detection queries on evolving track states while improving the robustness of memory priors and auxiliary scheduling.

Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work, the authors used Codex (OpenAI) to refine the academic phrasing, improve the overall grammatical structure of the manuscript, ensure correct spelling and formatting, and assist in organizing the public code repository and its accompanying documentation. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.

## CRediT authorship contribution statement

Hanyang Liu: Writing – review & editing, Writing – original draft, Software, Visualization, Validation, Formal analysis, Data curation, Conceptualization; Hong Zhang: Resources, Methodology, Formal analysis, Funding acquisition; Ding Yuan: Writing – review & editing, Resources, Formal analysis; Yang Han: Data curation, Investigation; Yifan Yang: Supervision, Project administration, Funding acquisition.

## Data availability

The source code, trained checkpoints, training logs, and tracking results are publicly available at https://github.com/Magicmiya/TAQ-MOTR.

## Acknowledgements

This work was supported by the National Natural Science Foundation of China (Grant Nos. 62433003 and 62476017).

## References

[1] G. Ciaparrone, F. Luque Sánchez, S. Tabik, L. Troiano, R. Tagliaferri, F. Herrera, Deep learning in video multi-object tracking: A survey, Neurocomputing 381 (2020) 61–88. doi:10.1016/j.neucom.2019.11.023.

[2] W. Luo, J. Xing, A. Milan, X. Zhang, W. Liu, T.-K. Kim, Multiple object tracking: A literature review, Artificial Intelligence 293 (2021) 103448. doi: 10.1016/j.artint.2020.103448.

[3] Y. Cui, C. Zeng, X. Zhao, Y. Yang, G. Wu, L. Wang, SportsMOT: A large multi-object tracking dataset in multiple sports scenes, in: Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 9921– 9931.

[4] Y. Zhang, P. Sun, Y. Jiang, D. Yu, F. Weng, Z. Yuan, P. Luo, W. Liu, X. Wang, ByteTrack: Multi-object tracking by associating every detection box, in: European conference on computer vision, Springer, 2022, pp. 1–21.

[5] X. Zhu, W. Su, L. Lu, B. Li, X. Wang, J. Dai, Deformable DETR: Deformable transformers for end-to-end object detection, in: International Conference on Learning Representations, 2021.

[6] P. Sun, J. Cao, Y. Jiang, R. Zhang, E. Xie, Z. Yuan, C. Wang, P. Luo, TransTrack: Multiple object tracking with transformer, arXiv preprint arXiv:2012.15460 (2020).

[7] T. Meinhardt, A. Kirillov, L. Leal-Taixé, C. Feichtenhofer, TrackFormer: Multi-object tracking with transformers, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 8844– 8854.

[8] F. Zeng, B. Dong, Y. Zhang, T. Wang, X. Zhang, Y. Wei, MOTR: End-toend multiple-object tracking with transformer, in: European conference on computer vision, Springer, 2022, pp. 659–675.

[9] E. Yu, T. Wang, Z. Li, Y. Zhang, X. Zhang, W. Tao, MOTRv3: Releasefetch supervision for end-to-end multi-object tracking, arXiv preprint arXiv:2305.14298 (2023).

[10] Y. Zhang, T. Wang, X. Zhang, MOTRv2: Bootstrapping end-to-end multiobject tracking by pretrained object detectors, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 22056–22065.

[11] M. Hayhoe, D. Ballard, Eye movements in natural behavior, Trends in cognitive sciences 9 (4) (2005) 188–194.

[12] J. M. Wolfe, Guided search 2.0 a revised model of visual search, Psychonomic bulletin & review 1 (2) (1994) 202–238.

[13] P. Sun, J. Cao, Y. Jiang, Z. Yuan, S. Bai, K. Kitani, P. Luo, DanceTrack: Multi-object tracking in uniform appearance and diverse motion, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 20993–21002.

[14] G. Zheng, S. Lin, H. Zuo, C. Fu, J. Pan, NetTrack: Tracking highly dynamic objects with a net, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 19145–19155.

[15] J. Cao, J. Pang, X. Weng, R. Khirodkar, K. Kitani, Observation-centric SORT: Rethinking SORT for robust multi-object tracking, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 9686–9696.

[16] M. Yang, G. Han, B. Yan, W. Zhang, J. Qi, H. Lu, D. Wang, Hybrid-SORT: Weak cues matter for online multi-object tracking, in: Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 38, 2024, pp. 6504–6512. doi:10.1609/aaai.v38i7.28471.

[17] N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, S. Zagoruyko, End-to-end object detection with transformers, in: European conference on computer vision, Springer, 2020, pp. 213–229.

[18] R. Gao, L. Wang, MeMOTR: Long-term memory-augmented transformer for multi-object tracking, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 9901–9910.

[19] P. Wang, Y. Wang, H. Cao, W. Chen, D. Li, LA-MOTR: End-to-end multiobject tracking by learnable association, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 12438–12448.

[20] R. Gao, J. Qi, L. Wang, Multiple object tracking as ID prediction, in: Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 27883–27893.

[21] N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. Rädle, C. Rolland, L. Gustafson, et al., SAM 2: Segment anything in images and videos, in: International Conference on Learning Representations, Vol. 2025, 2025, pp. 28085–28128.

[22] J. Jiang, Z. Wang, M. Zhao, Y. Li, D. Jiang, SAM2MOT: A novel paradigm of multi-object tracking by segmentation, in: Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40, 2026, pp. 5388–5396.

[23] W. Lv, Y. Zhao, Q. Chang, K. Huang, G. Wang, Y. Liu, RT-DETRv2: Improved baseline with bag-of-freebies for real-time detection transformer, arXiv preprint arXiv:2407.17140 (2024).

[24] F. Yan, W. Luo, Y. Zhong, Y. Gan, L. Ma, CO-MOT: Boosting end-to-end transformer-based multi-object tracking via coopetition label assignment and shadow sets, in: The Thirteenth International Conference on Learning Representations, 2025.

[25] F. Li, H. Zhang, S. Liu, J. Guo, L. M. Ni, L. Zhang, DN-DETR: Accelerate DETR training by introducing query denoising, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 13619–13627.

[26] Y. Zhao, W. Lv, S. Xu, J. Wei, G. Wang, Q. Dang, Y. Liu, J. Chen, DETRs beat YOLOs on real-time object detection, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 16965– 16974.

[27] S. Liu, F. Li, H. Zhang, X. Yang, X. Qi, H. Su, J. Zhu, L. Zhang, DAB-DETR: Dynamic anchor boxes are better queries for DETR, in: International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=oMI9PjOb9Jl

[28] J. Luiten, A. Osep, P. Dendorfer, P. Torr, A. Geiger, L. Leal-Taixé, B. Leibe, HOTA: A higher order metric for evaluating multi-object tracking, International Journal of Computer Vision 129 (2021) 548–578. doi:10.1007/ s11263-020-01375-2.

[29] E. Ristani, F. Solera, R. S. Zou, R. Cucchiara, C. Tomasi, Performance measures and a data set for multi-target, multi-camera tracking, in: Computer Vision – ECCV 2016 Workshops, 2016, pp. 17–35. doi:10.1007/ 978-3-319-48881-3\_2.

[30] K. Bernardin, R. Stiefelhagen, Evaluating multiple object tracking performance: The CLEAR MOT metrics, EURASIP Journal on Image and Video Processing 2008 (2008) 246309. doi:10.1155/2008/246309.

[31] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollaŕ, C. L. Zitnick, Microsoft COCO: Common objects in context, in: European conference on computer vision, Springer, 2014, pp. 740–755.

[32] W. Lv, Y. Huang, N. Zhang, R.-S. Lin, M. Han, D. Zeng, DifMOT: A realtime difusion-based multiple object tracker with non-linear prediction, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 19321–19330.

[33] S. Kim, S. Lee, M. Cho, PlugTrack: Multi-perceptive motion analysis for adaptive fusion in multi-object tracking, in: Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40, 2026, pp. 5709–5716. doi: 10.1609/aaai.v40i7.37491.

[34] C. Li, J. Ma, L. Shen, Y. Chen, L. Chen, Occlusion-aware SORT: Observing occlusion for robust multi-object tracking, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 42560– 42570.

[35] M. Segu, L. Piccinelli, S. Li, Y.-H. Yang, L. Van Gool, B. Schiele, Samba: Synchronized set-of-sequences modeling for multiple object tracking, in: International Conference on Learning Representations, Vol. 2025, 2025, pp. 30057–30070.

[36] S. Jia, S. Hu, Y. Cao, F. Yang, X. Lu, X. Lu, Tracking by detection and query: An eficient end-to-end framework for multi-object tracking, Pattern Recognition 179 (2026) 113552. doi:10.1016/j.patcog.2026.113552.

[37] R. Zeng, Y. Huang, S. Pei, TGFormer: Transformer with track query group for multi-object tracking, in: Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 39, 2025, pp. 9824–9832.

[38] P. Liao, F. Yang, D. Wu, W. Zhao, J. Yu, D. Zhang, DecoderTracker: Decoder-only end-to-end method for multiple-object tracking, Pattern Recognition 177 (2026) 113242. doi:10.1016/j.patcog.2026.113242.

[39] X. Han, N. Oishi, Y. Tian, E. Ucurum, R. Young, C. Chatwin, P. Birch, ET-Track: Enhanced temporal motion predictor for multi-object tracking, Applied Intelligence 55 (2025) 33. doi:10.1007/s10489-024-05866-4.

[40] Y. Shao, Y. Yang, R. Yu, W. Li, X. Guo, H. Yan, W. Wang, X. Sun, From detection to association: Learning discriminative object embeddings for multiobject tracking, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 6878–6888.

[41] A. Bewley, Z. Ge, L. Ott, F. Ramos, B. Upcroft, Simple online and realtime tracking, in: Proceedings of the IEEE International Conference on Image Processing, 2016, pp. 3464–3468.