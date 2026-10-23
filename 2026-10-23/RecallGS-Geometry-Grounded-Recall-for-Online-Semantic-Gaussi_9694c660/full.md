# RecallGS: Geometry-Grounded Recall for Online Semantic Gaussian Splatting

Thanh-Khoi Nguyen<sup>1,2</sup> , Thien-Phuc Tran<sup>1,2</sup> , and Minh-Triet Tran <sup>⋆1,2</sup>

<sup>1</sup> University of Science, Ho Chi Minh City, Vietnam2 Viet Nam National University, Ho Chi Minh City, Vietnam23120009@student.hcmus.edu.vn, ttphuc@selab.hcmus.edu.vn,tmtriet@fit.hcmus.edu.vn

Abstract. Online semantic 3D Gaussian Splatting (3DGS) enables openvocabulary scene understanding from streaming observations. However, existing methods recover geometric context missing from image-derived semantic features using an additionally learned 3D network, incurring extra training and runtime overhead. We observe that such geometricsemantic evidence may already exist in the agent’s observation history and can be retrieved rather than learned. Based on this insight, we introduce RecallGS (Geometry-Grounded Recall for Online Semantic Gaussian Splatting), a lightweight, training-free mechanism that treats past observations as a non-parametric geometric-semantic prior. RecallGS follows a recall-ground-fuse pipeline: CLIP retrieves relevant observations, alpha-compositing weights identify geometrically supported views, and aggregate their semantic features over each Gaussian’s rendered footprint. On ScanNet and ScanNet++, RecallGS achieves strong semantic segmentation performance at 2.48 FPS, demonstrating that geometric-semantic context can be efectively recovered from observation history without an additional learned 3D branch.

Keywords: 3D Gaussian Splatting, 3D Segmentation, Scene Understanding

## 1 Introduction

3D Gaussian Splatting (3DGS) provides an eficient explicit representation for real-time novel-view synthesis[11]. Recent works extend 3DGS with openvocabulary semantics by distilling vision-language features into Gaussians, either through rendering-based supervision[19,22,30] or direct 3D feature lifting and referring[26,12]. This paradigm has further expanded to online embodied settings, using either language-aware 3DGS-SLAM[9,31] or generalizable feed-forward reconstruction[14], alongside online 3D instance segmentation methods that fuse 2D foundation-model predictions into persistent 3D representations[27,24].

Despite this progress, existing online semantic 3DGS pipelines recover the geometric context missing from image-derived language features through an additional learned 3D branch, rather than directly reusing geometric-semantic evidence from past observations. EmbodiedSplat[14], for instance, complements its 2D CLIP features [21] with geometry-aware 3D features produced by passing the local Gaussian point cloud through a sparse 3D U-Net[6] equipped with a temporal memory adapter[28]. This 3D branch, however, must be trained on large-scale posed 3D indoor scene data, which is costly to collect and introduces additional GPU-memory consumption and latency that directly reduce the online reconstruction speed. EmbodiedSplat-fast removes the 3D U-Net and memory adapter, increasing throughput from 0.75 FPS to 5.18 FPS at the cost of geometric-semantic context. This highlights the training and inference overhead of recovering explicit 3D context through an additional learned branch. We ask whether such context must be learned at all, or can instead be recovered from the agent’s observation history.

We observe that this trade-of is a consequence of treating each observation independently. An embodied agent operates in a persistent environment: a Gaussian that is geometrically or semantically ambiguous from the current viewpoint may correspond to a physical surface already observed earlier under a more favorable viewpoint, illumination, or occlusion pattern. This suggests that part of the geometric prior EmbodiedSplat’s 3D branch must learn from parameters could instead be recovered from the agent’s own trajectory, treating its accumulated observation history as a non-parametric geometric-semantic prior queried on demand rather than compressed once and discarded. However, efectively reusing past observations introduces several challenges. First, semantic similarity does not imply physical correspondence: a CLIP-based retrieval system[21] may return frames containing a visually similar object-another chair, table, or door-at an entirely diferent location. Second, a Gaussian may project inside a historical image but still be occluded by closer geometry. Sampling the projected pixel would then transfer features from the occluder rather than from the queried surface. Finally, exhaustively verifying every historical camera against every incoming frame would itself undermine the eficiency this history is meant to provide, as a long-running agent may accumulate hundreds to thousands of observations. The problem is therefore not retrieval of similar-looking memories, but memory routing: determining, without prior semantic knowledge of the current Gaussians, which past observations should be allowed to influence each one.

To this end, we propose RecallGS (Geometry-Grounded Recall), a lightweight mechanism requiring no additional training that treats an agent’s observation history as a queryable source of geometric-semantic evidence, routed to the current Gaussians through a recall–ground–replay pipeline. Our contributions are threefold: (1) semantic recall, a cheap CLIP-based retrieval stage that narrows an unbounded observation history to a small candidate pool of appearanceplausible frames; (2) geometric grounding, which reranks these candidates using alpha-compositing weights to identify views that geometrically support the current Gaussians; and (3) compositing fusion, which reuses the same alphacompositing weights to aggregate historical features over each Gaussian’s full rendered footprint. Our experiments show that each component contributes independently to semantic accuracy, and that their combination enables efective geometric-semantic reasoning from observation history while maintaining eficient online inference

Our contributions are as follows:

We show that an agent’s observation history can serve as a non-parametric geometric-semantic prior, providing complementary semantic evidence without additional 3D training or per-scene optimization.

– We propose RecallGS, a training-free recall-ground-fuse framework that eficiently retrieves relevant past observations, grounds them against the current geometry, and fuses their semantic evidence into the current Gaussians.

– We validate the contribution of each component and show that exploiting observation history consistently improves online semantic 3DGS while maintaining eficient inference.

## 2 Related Work

## 2.1 Language-Embedded Gaussian Splatting

Building on 3D Gaussian Splatting[11], one line of work embeds open-vocabulary semantics onto Gaussians via per-scene optimization, either by distilling 2D vision-language features through diferentiable rendering supervision[19,22,30] or by directly associating semantic features with 3D structure through clustering, feature lifting, or compact Gaussian-level representations [26,12,16,5,1,20,23]. These methods establish 3DGS as an efective carrier of language features and improve their spatial consistency, instance awareness, or representation eficiency. However, most construct semantic fields specifically for each scene, which limits their applicability when observations arrive continuously.

To remove scene-specific optimization, a class of feed-forward methods predicts 3D Gaussian representations directly from input images without per-scene training. This paradigm was established by geometric backbones such as pixelSplat[2] and MVSplat[4], which regress RGB Gaussians from sparse image pairs. Building on this feed-forward formulation, subsequent work adds an open-vocabulary semantic branch: LSM [8] jointly regresses geometry, appearance, and semantics from unposed image pairs by lifting a pre-trained 2D language-based segmentation model into a 3D-consistent feature field decoded into semantic Gaussians, SLGaussian [3] targets the sparse-view regime with a multi-view language memory bank that maps high-dimensional CLIP features to compact per-object IDs for fast openvocabulary querying, and SemanticSplat [17] further improves language-aware feed-forward reconstruction by fusing SAM [13] and CLIP-LSeg [15] features into a cost-volume representation for cross-view semantic aggregation. These approaches avoid per-scene semantic optimization and generalize across scenes, but still operate on a fixed, small set of input views rather than maintaining a persistent semantic representation under a streaming observation sequence.

## 2.2 Online Gaussian Splatting Reconstruction and Segmentation

A separate line of work extends 3DGS to the online setting required by embodied perception. Within visual SLAM, SplaTAM [10] and Gaussian Splatting SLAM [18] use explicit Gaussians for incremental tracking, mapping, and rendering, demonstrating that 3DGS can serve as a persistent scene representation under streaming observations. Building on this formulation, Online-LangSplat [9] and EA3D [31] inject language or foundation-model features into the online reconstruction loop. Online-LangSplat’s language module itself runs in milliseconds, but the overall SLAM-GS pipeline remains bottlenecked by per-frame camerapose optimization, yielding well under 2 FPS end-to-end; EA3D similarly relies on a recurrent joint optimization module to align geometric and semantic features, at a comparable throughput cost. In parallel, online 3D instance segmentation methods incrementally fuse 2D foundation-model predictions into temporally consistent 3D instances [27,24], building on adapter-based caching mechanisms that aggregate RGB-D features frame-by-frame to endow otherwise-ofline back bones with temporal learning ability [28]; these methods, however, operate on point-based representations rather than a diferentiable and renderable Gaussian semantic field.

Closest to our setting are online, generalizable feed-forward pipelines that avoid per-scene optimization entirely. FreeSplat++ [25] targets eficient whole-scene indoor reconstruction from long image sequences, using a low-cost cross-view aggregation module and a pixel-wise triplet fusion strategy to incrementally merge overlapping Gaussian primitives across views, together with a weighted floater-removal step that exploits the accumulated blending weights from this fusion. EmbodiedSplat [14] extends this backbone with open-vocabulary semantics through a sparse CLIP codebook and a learned 3D U-Net branch that injects geometric context into image-derived semantic features. This improves geometricsemantic awareness but requires additional 3D training and incurs non-negligible inference overhead. In contrast, we recover complementary geometric-semantic evidence directly from the agent’s own observation history and route it to the current Gaussians through a lightweight recall-ground-fuse mechanism without an additional learned 3D semantic branch.

## 3 Method

## 3.1 Preliminaries

3D Gaussian Splatting 3DGS [11] represents a scene as Gaussian primitives $\boldsymbol \theta = \{ \mu _ { i } , { \Sigma } _ { i } , { \alpha } _ { i } , { \mathbf { c } } _ { i } \} _ { i = 1 } ^ { M }$ , rendered along each ray by

$$
\begin{array} { r } { \hat { \mathbf { c } } = \displaystyle \sum _ { i = 1 } ^ { N _ { r } } T _ { i } \tilde { \alpha } _ { i } \mathbf { c } _ { i } , \qquad \tilde { \alpha } _ { i } = \alpha _ { i } \exp \left( - \frac { 1 } { 2 } \mathbf { d } ^ { \top } \boldsymbol \Sigma _ { 2 D } ^ { - 1 } \mathbf { d } \right) , } \end{array}\tag{1}
$$

where $N _ { r }$ is the number of contributing Gaussians, $T _ { i }$ the transmittance, $\Sigma _ { 2 D }$ the projected covariance, and d the displacement from the projected mean. For pixel $p$ in camera $j ,$ we use the transmittance $T _ { i p } ^ { ( j ) }$ and efective opacity $\tilde { \alpha } _ { i p } ^ { ( j ) }$ . Their product, $w _ { i p } ^ { ( j ) } = T _ { i p } ^ { ( j ) } \tilde { \alpha } _ { i p } ^ { ( j ) }$ , defines the rendering responsibility used for visibility reasoning in Sec. 3.5.

FreeSplat++ Following EmbodiedSplat [14], we adopt FreeSplat++ [25] as the feed-forward reconstruction backbone. At timestep t, FreeSplat++ extracts multi-view image features, aggregates them through its cross-view reconstruction pipeline, and predicts pixel-aligned local Gaussian triplets $\boldsymbol { \Theta } _ { l } ^ { t } = \{ \mu _ { l } ^ { t } , \gamma _ { l } ^ { t } , \mathbf { f } _ { l } ^ { t } \}$ , where $\mu _ { l } ^ { t } , \gamma _ { l } ^ { t }$ , and $\mathbf { f } _ { l } ^ { t }$ denote the Gaussian centers, fusion confidences, and latent features, respectively. These triplets form a latent Gaussian representation rather than the full rendering parameters $\{ \mu , \Sigma , \alpha , \mathbf { c } \}$ . Overlapping local and global triplets are incrementally fused through confidence-weighted center updates and GRU-based latent fusion. FreeSplat++ decodes the fused latent representation into rendering parameters after aggregation; for geometric grounding, we apply the same decoder to the current local triplets at every timestep to obtain covariance, opacity, and color. We denote the resulting renderable local Gaussians by $\hat { \boldsymbol { \Theta } } _ { l } ^ { t }$ , which preserve the centers and indices of $\boldsymbol { \Theta } _ { l } ^ { t }$

EmbodiedSplat EmbodiedSplat [14] provides open-vocabulary semantics for this online backbone. Since storing a dense CLIP feature for every Gaussian is prohibitive at scene scale (M often exceeds $1 0 ^ { 6 } )$ , it maintains a $C L I P$ Global Codebook $\mathbf { C } ^ { t }$ of instance-level features, ${ \bf C } ^ { t } = \mathrm { c o n c a t } ( { \bf C } ^ { t - 1 } , \hat { \bf s } ^ { t } )$ , together with a fixed-length per-Gaussian sparse coeficient field $\bigl \{ \mathrm { I d x } _ { g } ^ { t } , \varOmega _ { g } ^ { t } \bigr \}$ that stores $L$ codebook indices and coeficients instead of a dense D-dimensional feature. The semantic feature of Gaussian i is reconstructed as

$$
\mathbf { S } _ { g } ^ { T _ { \mathrm { e n d } } } \bigl ( i \bigr ) = \sum _ { k = 1 } ^ { L } \varOmega _ { g } ^ { T _ { \mathrm { e n d } } } \bigl ( i , k \bigr ) { \mathbf { C } } ^ { T _ { \mathrm { e n d } } } \bigl ( \mathrm { I d } { \mathbf { x } } _ { g } ^ { T _ { \mathrm { e n d } } } \bigl ( i , k \bigr ) \bigr ) , \qquad \sum _ { k = 1 } ^ { L } \varOmega _ { g } ^ { T _ { \mathrm { e n d } } } \bigl ( i , k \bigr ) = 1 .\tag{2}
$$

Because the codebook stores unmodified CLIP vectors rather than a learned bottleneck, it requires neither per-scene optimization nor pretraining, while retaining CLIP’s open-vocabulary generalization and supporting online updates through fixed-length caches. For semantic replay, we additionally cache a dense pixel-aligned CLIP feature map $F _ { j } ( p ) \in \mathbb { R } ^ { D }$ for each historical frame $I ^ { j } { } _ { ; }$ , distinct from the instance-pooled features $\hat { \mathbf { s } } ^ { j }$

## 3.2 Problem Formulation

At timestep $t ,$ the system receives a posed RGB observation $\mathcal { O } ^ { t }$ and maintains a scene state $S ^ { t }$ :

$$
\mathcal { O } ^ { t } = \left( I ^ { t } , \mathbf { K } ^ { t } , \mathbf { E } ^ { t } \right) , \qquad \mathcal { S } ^ { t } = \left( \hat { \theta } _ { g } ^ { t } , \mathbf { C } ^ { t } , \mathrm { I d x } _ { g } ^ { t } , \varOmega _ { g } ^ { t } \right) .\tag{3}
$$

Here, $I ^ { t }$ is the current RGB frame, $\mathbf { K } ^ { t }$ the camera intrinsics, and $\mathbf { E } ^ { t }$ the camerato-world pose. Following EmbodiedSplat, $\hat { \theta } _ { g } ^ { t }$ is the global Gaussian map, incrementally updated from the current local prediction $\hat { \boldsymbol { \Theta } } _ { l } ^ { t }$ , while the semantic state comprises the CLIP Global Codebook $\mathbf { C } ^ { t }$ and the per-Gaussian sparse coeficient field $\{ \mathrm { I d x } _ { g } ^ { t } , \varOmega _ { g } ^ { t } \}$ . We denote by $g _ { i } ^ { t } \in \hat { \theta } _ { l } ^ { t }$ the i-th local Gaussian at timestep $t ,$ represented by its decoded parameters when rasterized. Given the streaming sequence $\mathcal { O } ^ { 1 : t }$ , our goal is to construct this persistent geometric and open-vocabulary semantic representation online. RecallGS additionally maintains a fixed-size episodic memory

![](images/2505d8ba57810a9d9a12cc7ea7511d0e4c4a2416c43219e0610fcf113ebbdc10.jpg)  
Fig. 1. Overview of RecallGS. Given a posed observation and the observation history, RecallGS recalls candidate views via $\mathrm { C L I P , }$ verifies their geometric support with rendering responsibility, and replays the verified features to build an online 3DGS reconstruction with a semantic field

$$
\mathcal { H } _ { W } ^ { t - 1 } = \left\{ \left( I ^ { j } , \mathbf { K } ^ { j } , \mathbf { E } ^ { j } , q ^ { j } , F _ { j } \right) \right\} _ { j = \operatorname* { m a x } ( 1 , t - W ) } ^ { t - 1 } ,\tag{4}
$$

containing at most W recent observations. Each entry stores the RGB frame, camera calibration and pose, image-level retrieval descriptor $q ^ { j }$ , and pixel-aligned feature map $F _ { j }$ . Once the memory reaches capacity, the oldest observation is discarded. The online state update is therefore

$$
{ \cal S } ^ { t } = \mathcal { U } \left( \mathcal { O } ^ { t } , { \cal S } ^ { t - 1 } , \mathcal { H } _ { W } ^ { t - 1 } \right) .\tag{5}
$$

The model must produce $S ^ { t }$ using only $\mathcal { O } ^ { 1 : t }$ , without access to future frames or ofline per-scene optimization.

## 3.3 Overview

As shown in Fig 1, RecallGS equips a lightweight online semantic mapping pipeline with episodic memory. For each incoming observation, we retrieve relevant past views, verify their geometric compatibility with the current scene, and reuse their semantic evidence to refine the current representation. This enables the model to exploit previously observed information while preserving an eficient online pipeline.

## 3.4 Semantic Recall over Episodic Memory

EmbodiedSplat grounds created Gaussians $g _ { i } ^ { t } \in \hat { \theta } _ { l } ^ { t }$ using semantic features from the current observation, while its global map and sparse coeficient field propagate fused semantics across timesteps. However, it does not explicitly reuse per-frame semantic features from past observations for newly introduced Gaussians. RecallGS addresses this limitation by querying its episodic memory $\mathcal { H } _ { W } ^ { t - 1 }$ . A newly introduced Gaussian may correspond to a surface observed earlier under a more favorable viewpoint, illumination, or occlusion pattern; retrieving such observations provides complementary semantic evidence for the current local Gaussians.

Although the episodic memory is bounded by $W$ , only a small subset of its observations is typically relevant to the current local Gaussians. Geometrically evaluating all cached observations would therefore require unnecessary rasterization from camera poses unlikely to provide useful evidence. We adopt a coarse-to-fine strategy that first uses inexpensive appearance-based retrieval to select a small set of plausible observations from $\mathcal { H } _ { W } ^ { \dot { t } - 1 }$ , and applies geometrydependent verification only to the recalled candidates (Sec. 3.5).

Concretely, in addition to the per-instance CLIP feature $\hat { \mathbf { s } } ^ { t }$ that Embodied-Splat extracts from $I ^ { t }$ to populate the codebook $\mathbf { C } ^ { t }$ (Sec. 3.1), we compute a global image-level CLIP embedding

$$
q ^ { t } = \mathrm { C L I P } ( I ^ { t } ) \in \mathbb { R } ^ { D } .\tag{6}
$$

The two features serve diferent purposes: $\hat { \mathbf { s } } ^ { t }$ is an instance-level feature associated with a segmentation mask and used to construct per-Gaussian semantics, whereas $q ^ { t }$ is a single image-level descriptor used only for retrieving observations from the episodic memory. Each $q ^ { j }$ is computed once when its observation enters the memory and remains fixed while that observation is retained.

Given the current embedding $q ^ { t }$ , we compute its cosine similarity with each observation retained in $\mathcal { H } _ { W } ^ { t - 1 }$ and retrieve the $B _ { \mathrm { r e c } }$ highest-scoring observations:

$$
\rho _ { j } ^ { \mathrm { c l i p } } = \frac { ( q ^ { t } ) ^ { \top } q ^ { j } } { \Vert q ^ { t } \Vert _ { 2 } \Vert q ^ { j } \Vert _ { 2 } } , \qquad \mathcal { R } ^ { t } = \underbrace { \mathrm { T o p } . \mathrm { B } _ { \mathrm { r e c } } } _ { \mathrm { m a x } ( 1 , t - W ) \leq j < t } \rho _ { j } ^ { \mathrm { c l i p } } .\tag{7}
$$

Here, $B _ { \mathrm { r e c } } \ll W$ is the recall budget, and Top-B returns all available observations when fewer than $B$ are stored. The resulting candidate set $\mathcal { R } ^ { t }$ restricts the more expensive geometric verification to at most $B _ { \mathrm { r e c } }$ observations, as described next.

## 3.5 Geometric Grounding of Recalled Memories

The candidate pool $\mathcal { R } ^ { t }$ obtained in Sec. 3.4 is constructed entirely from imagelevel appearance and carries no guarantee about the underlying 3D scene. Two frames may be assigned a high similarity score $\rho _ { j } ^ { \mathrm { c l i p } }$ simply because they share dominant visual attributes, such as similar wall color or furniture style, without observing any common physical surface; conversely, a historical camera $\mathbf { E } ^ { j }$ that does observe the region corresponding to $\hat { \boldsymbol { \Theta } } _ { l } ^ { t }$ may receive only a modest score if the remainder of $I ^ { j }$ is dominated by unrelated content. Using $\mathcal { R } ^ { t }$ directly as historical evidence would therefore risk contaminating the current Gaussians with semantics harvested from frames that never actually depicted the queried geometry. We resolve this ambiguity by explicitly verifying, for every candidate $j \in \mathcal { R } ^ { t }$ , how much of $\hat { \boldsymbol { \Theta } } _ { l } ^ { t }$ it renders once projected through the true camera geometry, rather than relying on appearance alone.

A naive verification would rasterize $\hat { \boldsymbol { \Theta } } _ { l } ^ { t }$ from camera $j$ and treat a Gaussian as visible whenever its projected center falls within the image bounds, or equivalently whenever its screen-space radius is non-zero. Such a criterion, however, only certifies that Gaussian i possesses a footprint in the image plane; it says nothing about whether that footprint survives occlusion by other Gaussians nearer to camera j along the same ray. Under the alpha-compositing model of Sec. 3.1, a Gaussian may lie well within the frustum of camera $j$ and still contribute negligibly to the rendered image if its transmittance $T _ { i p } ^ { ( j ) }$ has already been suppressed by closer geometry, in which case treating it as ${ \mathfrak { n } } _ { \mathbf { \lambda } _ { \mathbf { Y } } }$ visible" would inject semantic evidence from a view that, for that Gaussian, is efectively occluded. We therefore reject binary, projected-center visibility in favor of a criterion grounded directly in the rendering weight $w _ { i p } ^ { ( j ) } = T _ { i p } ^ { ( j ) } \tilde { \alpha } _ { i p } ^ { ( j ) }$ , which already accounts for occlusion among the rasterized Gaussians, partial coverage, and the soft fallof of the Gaussian kernel.

Concretely, for each candidate camera $j \in \mathcal { R } ^ { t }$ , we rasterize the current local Gaussian set $\hat { \boldsymbol { \theta } } _ { l } ^ { t }$ through $( \mathbf { K } ^ { j } , \mathbf { E } ^ { j } )$ and accumulate, for every Gaussian $g _ { i } ^ { t } \in \hat { \theta } _ { l } ^ { t }$ 2 its total rendering contribution across the image plane,

$$
A _ { i } ^ { ( j ) } = \sum _ { p } w _ { i p } ^ { ( j ) } .\tag{8}
$$

A Gaussian is considered geometrically supported by camera $j$ if its accumulated contribution exceeds a fixed threshold $\tau ,$ and we define the coverage of camera $j$ with respect to the current local Gaussian set as the fraction of Gaussians satisfying this criterion,

$$
\kappa _ { j } = \frac { 1 } { N _ { t } } \sum _ { i = 1 } ^ { N _ { t } } \mathbf { 1 } \Big [ A _ { i } ^ { ( j ) } > \tau \Big ] , \qquad N _ { t } = | \hat { \theta } _ { l } ^ { t } | .\tag{9}
$$

Intuitively, $\kappa _ { j }$ answers a question that $\rho _ { j } ^ { \mathrm { c l i p } }$ cannot: not merely whether camera $j$ depicts similar content to $I ^ { t }$ , but what fraction of the geometry newly introduced at timestep t was actually, and non-negligibly, observed by camera $j .$ . Because $A _ { i } ^ { ( j ) }$ is an unnormalized sum over pixels, the fixed threshold $\tau$ depends on image resolution and projected footprint size. It therefore measures visible rendering mass jointly with footprint size, rather than visibility alone.

With a fixed grounding budget $B _ { \mathrm { g e o } }$ , we select the candidates with the highest coverage as the final historical evidence set,

$$
\mathcal { V } ^ { t } = \operatorname * { T o p - } _ { j \in \mathcal { R } ^ { t } } \kappa _ { j } , \qquad B _ { \mathrm { g e o } } \ll B _ { \mathrm { r e c } } .\tag{10}
$$

Combining Eqs. (7) and (10), the selection process is

$$
\mathcal { H } _ { W } ^ { t - 1 } \xrightarrow { \mathrm { ~ S e m a n t i c ~ R e c a l l ~ } } \mathcal { R } ^ { t } \xrightarrow { \mathrm { ~ G e o m e t r i c ~ G r o u n d i n g ~ } } \mathcal { V } ^ { t } .\tag{11}
$$

Semantic recall first reduces the full history to at most $\boldsymbol { B } _ { \mathrm { r e c } }$ plausible candidates, after which geometric grounding retains only the views that provide verified support for the current Gaussians. This avoids rasterizing the entire history while ensuring that the selected views are geometrically relevant to $\hat { \boldsymbol { \theta } } _ { l } ^ { t }$

## 3.6 Compositing Fusion into the Global Codebook

We realize semantic replay via responsibility-weighted feature aggregation over the per-Gaussian historical view set $\mathcal { V } _ { i } ^ { t }$ defined below. Rendering-weighted multiview feature aggregation has also been explored in language Gaussian splatting[5]; our use of the same rendering responsibility is instead applied to causal historical replay in an online feed-forward system. The output of the two modules above, $\mathcal { V } ^ { t }$ identifies historical cameras verified to observe $\hat { \boldsymbol { \Theta } } _ { l } ^ { t }$ , but must still be reintroduced at the point where EmbodiedSplat grounds each Gaussian from I<sup>t</sup> alone. For each Gaussian $g _ { i } ^ { t }$ , we restrict evidence to views satisfying the same coverage criterion, $\mathcal { V } _ { i } ^ { t } = \{ j \in \mathcal { V } ^ { t } : A _ { i } ^ { ( j ) } > \tau \}$ , and sample the cached dense feature map $F _ { j } ( \cdot )$ over its rendered footprint, weighted by $w _ { i p } ^ { ( j ) }$

$$
\mathbf { s } _ { i , j } ^ { \mathrm { h i s t } } = \frac { \sum _ { p } w _ { i p } ^ { ( j ) } F _ { j } ( p ) } { A _ { i } ^ { ( j ) } + \epsilon } , \qquad \mathbf { s } _ { i } ^ { \mathrm { h i s t } } = \frac { \sum _ { j \in \mathcal { V } _ { i } ^ { t } } A _ { i } ^ { ( j ) } \mathbf { s } _ { i , j } ^ { \mathrm { h i s t } } } { \sum _ { j \in \mathcal { V } _ { i } ^ { t } } W _ { i } ^ { ( j ) } + \epsilon } .\tag{12}
$$

Here $\epsilon > 0$ stabilizes the denominators. The current-frame feature ${ \bf s } _ { i } ^ { \mathrm { c u r } } \in \mathbb { R } ^ { D }$ is obtained by the baseline current-frame grounding: the per-Gaussian sparse coeficients weight the current instance-level CLIP entries $\hat { \mathbf { s } } ^ { t }$ assigned to Gaussian i, and the resulting weighted sum is $\ell _ { 2 } \cdot$ -normalized. It lies in the same CLIP feature space as the codebook entries and $\mathbf { s } _ { g } ^ { T _ { \mathrm { e n d } } } ( i )$ , but precedes global fusion. We fuse the historical and current-frame features as

$$
\mathbf { s } _ { i } ^ { \mathrm { f i n a l } } = \mathrm { N o r m } \bigl ( \mathbf { s } _ { i } ^ { \mathrm { c u r } } + \lambda \mathbf { s } _ { i } ^ { \mathrm { h i s t } } \bigr ) ,\tag{13}
$$

where λ controls the historical contribution and Norm denotes $\ell _ { 2 }$ normalization. Since ${ \mathbf { s } } _ { i } ^ { \mathrm { c u r } }$ is unit-normalized, this reduces to ${ \bf s } _ { i } ^ { \mathrm { f i n a l } } = { \bf s } _ { i } ^ { \mathrm { c u r } }$ when $\mathcal { V } _ { i } ^ { t } = \mathcal { O } . \ \mathbf { s } _ { i } ^ { \mathrm { f i n a l } }$ is then passed to the sparse coeficient field and global codebook unchanged, so our two modules act purely as an upstream semantic evidence source, requiring no additional training.

## 4 Experiment

Dataset: We follow the same dataset and evaluation setup as EmbodiedSplat[14]. Specifically, we evaluate on the indoor ScanNet[7] and ScanNet++[29] benchmarks using the same preprocessed test scenes and ground-truth semantic annotations provided by the oficial EmbodiedSplat implementation. For ScanNet, we report 3D semantic segmentation under the 10, 15, and 19 class evaluation protocols, while ScanNet++ is evaluated under the 20-class setting. Unless otherwise stated, all dataset preprocessing, scene splits, and evaluation protocols are kept identical to EmbodiedSplat to ensure direct comparability.

Baselines. We compare RecallGS against representative open-vocabulary semantic 3DGS methods, including OpenGaussian[26], Occam’s LGS[5], Dr.Splat[12], and InstanceGaussian[16], as well as both EmbodiedSplat and EmbodiedSplatfast[14]. Among them, EmbodiedSplat is the most relevant reference, as it augments the feed-forward reconstruction backbone with an additional geometry aware 3D semantic branch, while EmbodiedSplat-fast removes this branch and relies only on the lightweight 2D semantic pathway. Since RecallGS is built upon EmbodiedSplat-fast, we use it as the primary baseline to isolate the benefit of historical semantic replay, while comparison with the full EmbodiedSplat evaluates whether historical evidence can recover geometry-aware semantic context without the additional 3D U-Net.

Table 1. Main comparison on ScanNet and ScanNet++. We report mIoU, mAcc, and online FPS; a dash denotes ofline methods. Best and second-best results are bold and underlined, respectively.
<table><tr><td>Method</td><td colspan="2"> $\mathrm { S c a n N e t { - } 1 0 }$  mIoU mAcc</td><td colspan="2">ScanNet-15 mIoU mAcc</td><td colspan="2"> $\mathrm { S c a n N e t { - } 1 9 }$  mIoU mAcc</td><td colspan="2">ScanNet++-20 mIoU mAcc</td><td> $\operatorname { A v g } \operatorname { F P S } |$ </td><td>Scope</td><td>Mode</td></tr><tr><td>LangSplat LEGaussians</td><td>6.52</td><td>20.11</td><td>3.25</td><td>13.16</td><td>1.34</td><td>7.44</td><td>2.21</td><td>10.18</td><td>一</td><td></td><td>Offline</td></tr><tr><td>Online-LangSplat</td><td>6.79 7.13</td><td>18.13 21.56</td><td>4.13 3.89</td><td>15.49 14.52</td><td>2.53 3.45</td><td>5.67 8.97</td><td>2.93 4.51</td><td>9.34 11.34</td><td>- 1.12</td><td>Per-Scene</td><td>Online</td></tr><tr><td>OpenGaussian</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>29.50</td><td>44.61</td><td>23.74</td><td>39.14</td><td>22.52</td><td>35.02</td><td>25.65</td><td>37.03</td><td>1 -</td><td></td><td></td></tr><tr><td>Occam&#x27;s LGS</td><td>42.14</td><td>70.28</td><td>35.04</td><td>63.71</td><td>30.49</td><td>57.91</td><td>34.08</td><td>61.19</td><td>1</td><td>Per-Scene</td><td>Offline</td></tr><tr><td>Dr. Splat</td><td>39.21</td><td>66.66</td><td>31.84</td><td>60.58</td><td>28.38</td><td>55.85</td><td>39.85</td><td>58.34</td><td></td><td></td><td></td></tr><tr><td>InstanceGaussian</td><td>29.77</td><td>52.32</td><td>28.79</td><td>50.07</td><td>26.57</td><td>48.63</td><td>29.98</td><td>47.47</td><td></td><td></td><td></td></tr><tr><td>EmbodiedSplat (RGB)</td><td></td><td>76.13</td><td></td><td>75.47</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>EmbodiedSplat-fast (RGB)</td><td>49.81 47.86</td><td></td><td>49.23</td><td></td><td>46.22</td><td>70.37</td><td>41.93</td><td>61.50</td><td>0.75</td><td>Generalizable</td><td>Online</td></tr><tr><td></td><td></td><td>77.62</td><td>43.21</td><td>73.85</td><td>41.03</td><td>70.12</td><td>45.53</td><td>71.42</td><td>5.18</td><td></td><td></td></tr><tr><td>RecallGS</td><td>50.96</td><td>81.55</td><td>46.40</td><td>76.97</td><td>44.41</td><td>73.26</td><td>48.75</td><td>75.12</td><td>2.48</td><td></td><td></td></tr></table>

Implementation Details. We implement RecallGS on top of the oficial EmbodiedSplat-fast, keeping the FreeSplat++ backbone and semantic codebook unchanged. Unless otherwise specified, we use $B _ { \mathrm { r e c } } = 1 6 , B _ { \mathrm { g e o } } = 3 , \tau = 0 . 9$ , and λ = 0.75, following the replay procedure described in Sec. 3. We evaluate semantic segmentation using mIoU and mAcc on ScanNet-10/15/19 and ScanNet++-20, and report FPS under the same hardware and evaluation setting. All experiments are conducted in a single NVIDIA RTX5090 GPU.

Quantitative result Table 1 shows that RecallGS improves EmbodiedSplatfast by 3.10, 3.19, 3.38, and 3.22 mIoU on ScanNet-10/15/19 and ScanNet++-20, respectively, with corresponding mAcc gains. It remains competitive with the full model while running at 2.48 FPS versus 0.75 FPS (≈ 3× faster). Although slower than EmbodiedSplat-fast (5.18 FPS), RecallGS provides a favorable accuracy– eficiency trade-of and compares favorably with per-scene methods while retaining online inference.

Qualitative result: Fig. 2 demonstrates that RecallGS produces substantially more coherent semantic reconstructions, with cleaner region continuity and fewer fragmented or streak-like artifacts across large surfaces and object boundaries. In contrast, EmbodiedSplat exhibits pronounced local inconsistency, particularly in cluttered regions and partially observed structures. These results indicate that geometrically grounded historical replay provides complementary evidence that stabilizes semantic predictions across viewpoints and improves the consistency of the reconstructed semantic field.

## 5 Ablation study

## 5.1 Historical View Selection

Replacing temporally recent observations with CLIP-based retrieval improves performance across the four evaluation protocols in Table 2, yielding gains of approximately 0.44-0.63 mIoU. This indicates that temporal proximity alone is insuficient for identifying informative historical evidence, whereas semantic similarity provides a stronger prior for memory selection. Nevertheless, directly replaying the top- $B _ { \mathrm { g e o } }$ CLIP-retrieved views remains suboptimal. Incorporating Gaussian-based geometric reranking further improves performance by 0.54-0.88 mIoU, demonstrating that semantic relevance does not necessarily imply geometric compatibility with the current reconstruction. Overall, the full recalland-grounding strategy outperforms recent-history selection by approximately 1.14-1.51 mIoU, validating the complementary roles of semantic retrieval and geometric verification in this study.

![](images/067aacf2a957074755b0b5638f4071879547776d69b1f1d6094cea09fc1df6b5.jpg)  
Fig. 2. Qualitative comparison on ScanNet and ScanNet++. RecallGS produces more coherent semantic regions and fewer artifacts than EmbodiedSplat-fast, particularly on large surfaces and partially observed boundaries.

Table 2. Ablation of historical-view retrieval
<table><tr><td>Variant</td><td></td><td></td><td></td><td>ScanNet-10 ScanNet-15 ScanNet-19 ScanNet++-20</td></tr><tr><td>Recent History</td><td>49.74</td><td>44.89</td><td>43.07</td><td>47.61</td></tr><tr><td>CLIP Only</td><td>50.18</td><td>45.52</td><td>43.57</td><td>48.21</td></tr><tr><td>Full (RecallGS)</td><td>50.96</td><td>46.40</td><td>44.41</td><td>48.75</td></tr></table>

## 5.2 Compositing vs. Center-Based Fusion

Table 3 studies how Gaussian geometry is used during historical replay. Projected Center uses only the projected Gaussian center for both historical-view ranking and semantic feature assignment. Center Lifting retains our T α-based coverage for selecting historical views, but assigns each Gaussian the semantic feature sampled only at its projected center. In contrast, the full model uses T α throughout: historical cameras are ranked using the accumulated Gaussian contribution, and semantic features are aggregated over the complete projected footprint using the same rendering weights. The full formulation improves over Projected Center by approximately 1.22-1.69 mIoU and over Center Lifting by 0.69-0.91 mIoU, indicating the expected benefit of exploiting the complete Gaussian footprint for both historical observability and semantic transfer. These variants do not isolate the transmittance factor T from opacity α; therefore, the occlusion-specific contribution of T is not separately measured by this ablation.

Table 3. Ablation on compositing vs. center-based fusion.
<table><tr><td>Variant</td><td></td><td></td><td></td><td>ScanNet-10 ScanNet-15 ScanNet-19 ScanNet++-20</td></tr><tr><td>Projected Center</td><td>49.27</td><td>45.18</td><td>42.86</td><td>47.34</td></tr><tr><td>Center Lifting</td><td>50.05</td><td>45.62</td><td>43.58</td><td>48.06</td></tr><tr><td>Full Tα (RecallGS)</td><td>50.96</td><td>46.40</td><td>44.41</td><td>48.75</td></tr></table>

Table 4. Retrieval and replay budget sensitivity (default: $B _ { \mathrm { r e c } } = 1 6 , B _ { \mathrm { g e o } } = 3 )$

<table><tr><td colspan="4">Varying  $B _ { \mathrm { r e c } } \ \left( B _ { \mathrm { g e o } } = 3 \right)$ </td><td colspan="4">Varying  $B _ { \mathrm { g e o } } \ ( B _ { \mathrm { r e c } } = 1 6 )$ </td></tr><tr><td> $B _ { \mathrm { r e c } }$ </td><td>ScanNet-10</td><td> $\mathrm { S c a n N e t { - } 1 5 }$ </td><td>ScanNet-19</td><td> $B _ { \mathrm { g e o } }$ </td><td>ScanNet-10</td><td> $\mathrm { S c a n N e t { - } 1 5 }$ </td><td>ScanNet-19</td></tr><tr><td>8</td><td>50.82</td><td>46.12</td><td>44.02</td><td>2</td><td>50.71</td><td>46.08</td><td>43.91</td></tr><tr><td>16</td><td>50.96</td><td>46.40</td><td>44.41</td><td>3</td><td>50.96</td><td>46.40</td><td>44.41</td></tr><tr><td>32</td><td>50.77</td><td>46.05</td><td>43.96</td><td>5</td><td>50.84</td><td>46.17</td><td>44.05</td></tr></table>

## 5.3 Recall and Grounding Budget Sensitivity

Table 4 varies the recall budget $B _ { \mathrm { r e c } }$ and the grounding budget $B _ { \mathrm { g e o } }$ . Performance is moderately sensitive to both: moving away from the default $( B _ { \mathrm { r e c } } ~ = ~ 1 6$ $B _ { \mathrm { g e o } } = 3 )$ lowers mIoU by 0.12-0.50 across the three ScanNet protocols, and every setting remains above the CLIP-only and Center Lifting variants in Tables 3 and 2. A smaller $B _ { \mathrm { r e c } }$ may exclude informative views from the candidate pool, whereas a larger one admits more appearance-similar but geometrically unsupported candidates. Likewise, a smaller $B _ { \mathrm { g e o } }$ provides less complementary evidence, while a larger one adds redundant or weakly supported views. We therefore use $B _ { \mathrm { r e c } } = 1 6$ and $B _ { \mathrm { g e o } } = 3$ as the default; note that this setting was chosen on the same evaluation scenes

## 6 Conclusion

We propose RecallGS, a lightweight mechanism requiring no additional training that recasts an embodied agent’s observation history as a non-parametric geometric-semantic prior via a recall-ground-fuse pipeline: retrieving plausible historical frames, verifying their geometric support through alpha-compositing responsibility, and fusing only this verified evidence into the current Gaussians. On ScanNet and ScanNet++, RecallGS narrows the gap to EmbodiedSplat’s full 3D branch with higher speed than the full model.

We observed that RecallGS’s quality is bounded by the past observations it relies on, ofering limited benefit when historical views are sparse or lowquality and additionally, grounding rasterizes only the current local Gaussians, so occluders in the global map are not modeled and visibility may be overestimated. Future work may extend retrieval with a certainty or assessment gate that evaluates the reliability of a recalled observation, for instance based on viewpoint quality, illumination, or depth confidence, before allowing it to influence the current Gaussians.

## References

1. Cen, J., Fang, J., Yang, C., Xie, L., Zhang, X., Shen, W., Tian, Q.: Segment any 3d gaussians. In: Proceedings of the AAAI Conference on Artificial Intelligence. pp. 1971–1979 (2025)

2. Charatan, D., Li, S.L., Tagliasacchi, A., Sitzmann, V.: pixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 19457–19467 (2024)

3. Chen, K., Dai, B., Qin, M., Zhang, D., Li, P., Zou, Y., Wang, H.: Slgaussian: Fast language gaussian splatting in sparse views. In: Proceedings of the 33rd ACM International Conference on Multimedia (ACM MM) (2025)

4. Chen, Y., Xu, H., Zheng, C., Zhuang, B., Pollefeys, M., Geiger, A., Cham, T.J., Cai, J.: Mvsplat: Eficient 3d gaussian splatting from sparse multi-view images. In: European Conference on Computer Vision (ECCV) (2024)

5. Cheng, J., Zaech, J.N., Van Gool, L., Paudel, D.P.: Occam’s lgs: An eficient approach for language gaussian splatting. In: 36th British Machine Vision Conference (BMVC). BMVA (2025), https://bmva-archive.org.uk/bmvc/2025/assets/ papers/Paper\_694/paper.pdf

6. Choy, C., Gwak, J.Y., Savarese, S.: 4d spatio-temporal convnets: Minkowski convolutional neural networks. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3075–3084 (2019)

7. Dai, A., Chang, A.X., Savva, M., Halber, M., Funkhouser, T., Nießner, M.: Scannet: Richly-annotated 3d reconstructions of indoor scenes. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp. 5828–5839 (2017)

8. Fan, Z., Zhang, J., Cong, W., Wang, P., Li, R., Wen, K., Zhou, S., Kadambi, A., Wang, Z., Xu, D., Ivanovic, B., Pavone, M., Wang, Y.: Large spatial model: End-to-end unposed images to semantic 3d. In: Advances in Neural Information Processing Systems (NeurIPS) (2024)

9. Katragadda, S., Wu, C.Y., Guo, Y., Huang, X., Huang, G., Ren, L.: Online language splatting. arXiv preprint arXiv:2503.09447 (2025)

10. Keetha, N., Karhade, J., Jatavallabhula, K.M., Yang, G., Scherer, S., Ramanan, D., Luiten, J.: Splatam: Splat, track & map 3d gaussians for dense rgb-d slam. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 21357–21366 (2024)

11. Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G.: 3d gaussian splatting for real time radiance field rendering. ACM Transactions on Graphics 42(4), 139:1–139:14 (2023)

12. Kim, J.S., Kim, G., Kim, Y.J., Wang, Y.C.F., Choe, J., Oh, T.H.: Dr. splat: Directly referring 3d gaussian splatting via direct language embedding registration. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 14137–14146 (2025)

13. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., et al.: Segment anything. In: 2023 IEEE/CVF international conference on computer vision (ICCV). pp. 3992–4003. IEEE (2023)

14. Lee, S., Wang, Z., Wang, Y., Lee, G.H.: Embodiedsplat: Online feed-forward semantic 3dgs for open-vocabulary 3d scene understanding. arXiv preprint arXiv:2603.04254 (2026)

15. Li, B., Weinberger, K.Q., Belongie, S., Koltun, V., Ranftl, R.: Language-driven semantic segmentation. arXiv preprint arXiv:2201.03546 (2022)

16. Li, H., Wu, Y., Meng, J., Gao, Q., Zhang, Z., Wang, R., Zhang, J.: Instancegaussian: Appearance-semantic joint gaussian representation for 3d instance-level perception. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 14078–14088 (2025)

17. Li, Q., Sun, J., An, L., Su, Z., Zhang, H., Liu, Y.: Semanticsplat: Feed-forward 3d scene understanding with language-aware gaussian fields. arXiv preprint arXiv:2506.09565 (2025)

18. Matsuki, H., Murai, R., Kelly, P.H., Davison, A.J.: Gaussian splatting slam. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 18039–18048 (2024)

19. Qin, M., Li, W., Zhou, J., Wang, H., Pfister, H.: Langsplat: 3d language gaussian splatting. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 20051–20060 (2024)

20. Qiu, J., Liu, L., Su, Z., Lin, T.: Gls: Geometry-aware 3d language gaussian splatting. arXiv preprint arXiv:2411.18066 (2024)

21. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International Conference on Machine Learning. pp. 8748–8763 (2021)

22. Shi, J.C., Wang, M., Duan, H.B., Guan, S.H.: Language embedded 3d gaussians for open-vocabulary scene understanding. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5333–5343 (2024)

23. Shorinwa, O., Sun, J., Schwager, M.: Fastsplat: Fast, ambiguity-free semantics transfer in gaussian splatting. arXiv preprint arXiv:2411.13753 (2024)

24. Tang, Y., Zhang, J., Lan, Y., Guo, Y., Dong, D., Zhu, C., Xu, K.: Onlineanyseg: Online zero-shot 3d segmentation by visual foundation model guided 2d mask merging. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3676–3685 (2025)

25. Wang, Y., Huang, T., Chen, H., Lee, G.H.: Freesplat++: Generalizable 3d gaussian splatting for eficient indoor scene reconstruction. arXiv preprint arXiv:2503.22986 (2025)

26. Wu, Y., Meng, J., Li, H., Wu, C., Shi, Y., Cheng, X., Zhao, C., Feng, H., Ding, E., Wang, J., et al.: Opengaussian: Towards point-level 3d gaussian-based open vocabulary understanding. Advances in Neural Information Processing Systems 37, 19114–19138 (2024)

27. Xu, X., Chen, H., Zhao, L., Wang, Z., Zhou, J., Lu, J.: Embodiedsam: Online segment any 3d thing in real time. arXiv preprint arXiv:2408.11811 (2024)

28. Xu, X., Xia, C., Wang, Z., Zhao, L., Duan, Y., Zhou, J., Lu, J.: Memory-based adapters for online 3d scene perception. In: Proceedings of the IEEE/CVF Confer ence on Computer Vision and Pattern Recognition. pp. 21604–21613 (2024)

29. Yeshwanth, C., Liu, Y.C., Nießner, M., Dai, A.: Scannet++: A high-fidelity dataset of 3d indoor scenes. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 12–22 (2023)

30. Zhou, S., Chang, H., Jiang, S., Fan, Z., Zhu, Z., Xu, D., Chari, P., You, S., Wang, Z., Kadambi, A.: Feature 3dgs: Supercharging 3d gaussian splatting to enable distilled feature fields. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 21676–21685 (2024)

31. Zhou, X., Wang, J., Jia, Y., Wang, Y., Sun, D., Yang, M.H.: Ea3d: Online openworld 3d object extraction from streaming videos. arXiv preprint arXiv:2510.25146 (2025)