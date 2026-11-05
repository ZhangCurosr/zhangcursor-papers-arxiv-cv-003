# PixelSpline: Continuous Space-Time Video Super-Resolution via Endpoint-Anchored Shared Pixel Trajectories

Wanqi Yuan

Nianyi Li

Clemson University

Clemson, South Carolina, United States

wanqiy@clemson.edu

nianyil@clemson.edu

## Abstract

Existing continuous space-time video super-resolution methods often infer motion from only two endpoint frames and model the two motion directions independently, making them vulnerable to ambiguous correspondence and inconsistent trajectories. We introduce a continuous STVSR framework that combines long-context encoding, continuous HR feature lifting, and an endpoint-anchored Pixel-Spline. Long-range observations provide more reliable endpoint representations, while continuous upsampling constructs detailed appearance features directly on the requested HR grid. PixelSpline explicitly models each pixel’s motion as a single continuous trajectory shared by both endpoints. Using this trajectory, our method warps endpoint colors into a coarse intermediate frame and aligns the HR features for residual reconstruction. Experiments across spatial and temporal upsampling settings demonstrate that the proposed framework offers a favorable quality–efficiency trade-off. Our project page is: https: //wanqiyuan.github.io/PixelSpline/

## 1. Introduction

Video super-resolution (VSR) aims to recover highresolution (HR) videos from low-resolution (LR) observations. It is important for video capture, display, editing, and analysis, where spatial resolution is often limited by camera hardware, transmission bandwidth, or storage. However, improving spatial resolution alone does not resolve temporal undersampling: a sharp video can still exhibit discontinuous motion when recorded at a low frame rate. Video frame interpolation (VFI) addresses this limitation by synthesizing frames at unobserved time steps, but it cannot recover spatial details already lost in the LR inputs. Spacetime video super-resolution (STVSR) therefore combines VSR and VFI to reconstruct videos with both higher spatial resolution and higher frame rate. More recently, continuous

![](images/0ebcc46ea01766392798d4d0388cfdb7e05df1b8aef0df4583213b7517a77038.jpg)  
Figure 1. Quality–latency trade-off on Adobe240 Average under S×4 spatial and T×8 temporal upsampling. Our method is 2.24× faster than ${ \mathrm { V } } ^ { 3 } .$ , with a 0.90 dB PSNR difference. Latency is measured end-to-end on a single NVIDIA H100 GPU after warm-up.

STVSR (C-STVSR) has generalized this objective further, allowing a video to be queried at arbitrary spatial scales and continuous time coordinates.

Early STVSR approaches either connect VFI and VSR models with two-stage settings or jointly optimize the two tasks for predefined integer upsampling factors. One-stage methods such as Zooming Slow-Mo [27] and TMNet [28] demonstrate the benefit of joint spatial and temporal reconstruction. Continuous methods instead model video as a function over space and time. VideoINR [6] uses implicit functions for spatial appearance and time-dependent motion. MoTIF [5] models forward motion trajectories and splats endpoint features to a queried time, while BF-STVSR [14] introduces B-spline motion modeling together with a Fourier-based spatial representation. V<sup>3</sup> [3] recently explores a unified 3D Fourier field that avoids explicit warping.

Despite this progress, three closely related challenges remain. First, a correspondence inferred from only two endpoint frames can be ambiguous in occluded or disoccluded regions, and can be corrupted by motion blur, repetitive texture, weak texture, or large displacement. A mistaken endpoint match then affects every intermediate time queried between the pair, often producing ghosting, temporal jumps, or unstable structures. A longer temporal context can reduce this ambiguity: content hidden at one endpoint may be visible in an earlier or later frame, repeated observations provide additional evidence for motion direction, and trajectories observed over several frames help distinguish persistent motion from pairwise matching noise. Second, existing continuous methods generally evaluate the motion fields from the two endpoints to the target time separately. Even when the two directions share network parameters, their predictions are conditioned on different endpoint features or motion latents, without an explicit constraint that corresponding endpoint pixels follow the same physical curve. The two warps can therefore transport corresponding content toward inconsistent target locations. Third, arbitraryscale temporal synthesis requires endpoint appearance features on the requested HR grid. Fixed-scale upsamplers cannot serve arbitrary spatial queries, while warping only LR features postpones the construction of spatial detail until the final decoder. It is therefore desirable to lift the endpoint features continuously before temporal transport.

To address these challenges, we propose a continuous STVSR framework composed of four stages. First, a longcontext encoder aggregates information across multiple LR observations to produce target-pair-aware endpoint representations, improving correspondence estimation in ambiguous regions. Second, a CUF-based continuous feature upsampling module transforms the two LR endpoint representations into HR appearance features on the requested output grid, providing spatially detailed evidence before temporal synthesis. Third, we establish dense bidirectional endpoint correspondences and use them to construct an endpoint-constrained PixelSpline for every source pixel. Rather than predicting two endpoint-to-target trajectories, we construct one curve on the first endpoint’s correspondence grid and use the endpoint correspondence to express the same curve from the other endpoint. The two endpoint warps therefore traverse one shared path from opposite ends. At any target time, this trajectory is used twice: bicubic-upsampled endpoint colors are forward-warped and fused into a coarse, potentially blurry HR intermediate frame, while the lifted endpoint features are transported to the same target locations to form an aligned HR feature representation. Finally, a lightweight pointwise decoder converts the aligned representation into an HR residual, which is added to the coarse frame to obtain the final reconstruction. This design integrates geometry and appearance with distinct roles: the shared PixelSpline models how observed content moves, whereas the decoder restores details that cannot be recovered through direct pixel transport.

Extensive experiments on standard STVSR benchmarks show that the proposed framework occupies a favorable point on the quality–efficiency trade-off. As shown in Fig. 1, our method runs 2.24× faster than the qualityleading ${ \mathrm { V } } ^ { 3 }$ baseline, with a PSNR that is only 0.90 dB lower.

Our main contributions are summarized as follows:

• We introduce long-context endpoint encoding that uses observations beyond the queried pair to obtain more stable and globally informed endpoint representations.

• We propose an endpoint-anchored shared PixelSpline that explicitly models per-pixel motion as one continuous trajectory evaluated from both endpoints.

• We develop a CUF-based continuous feature lifting module that adapts continuous upsampling filters to map context-enhanced LR endpoint features onto arbitrary HR grids before trajectory-guided temporal transport.

## 2. Related Work

Arbitrary-scale super-resolution. Early arbitrary-scale image super-resolution predicts scale-conditioned reconstruction filters, as in Meta-SR [12]. Coordinate-based methods subsequently model an image as a continuous function: LIIF [4] decodes local features at arbitrary queries, while LTE [16] and Thera [2] use sinusoidal representations to better preserve high-frequency content and control aliasing across scales. SAVSR [17] and ST-AVSR [22] extend arbitrary spatial magnification to video by exploiting neighboring frames, but retain the original temporal grid. These developments motivate lifting video features continuously in space, but do not address interpolation at unseen times by themselves.

Space-time video super-resolution. Classical STVSR pipelines combine frame interpolation [1, 13] with image or video super-resolution [25, 30]. One-stage methods instead learn spatial and temporal reconstruction jointly. Zooming Slow-Mo [27] employs bidirectional deformable ConvLSTMs, TMNet [28] introduces temporal modulation, and subsequent architectures improve cross-frame propagation and aggregation [8–11]. These models are effective at their trained integer factors, but do not provide continuous space-time queries. Recently, Time Without Time [7] improves their pretraining through masked pseudo-temporal sequences, while retaining the same fixed-scale formulation.

Continuous STVSR removes this restriction by treating space and time as query coordinates. VideoINR [6] combines an implicit spatial representation with backward motion, and MoTIF [5] models forward trajectories and splats endpoint features to the target time. BF-STVSR [14] augments this formulation with Fourier features and B-spline motion, whereas ${ \mathrm { V } } ^ { 3 }$ [3] represents the video as a unified 3D Fourier field without explicit warping. Pairwise trajectory methods, however, construct a query from two anchorattached representations and model the two transport directions separately; ${ \mathrm { V } } ^ { 3 }$ uses broader temporal context but does not ensure explicit correspondence-consistent geometry. Our method brings these complementary ideas together by using long-context endpoint features to construct one endpoint-anchored PixelSpline shared by both transport directions.

## 3. Method

## 3.1. Overview

Let $\mathcal { T } ^ { L } = \{ I _ { n } ^ { L } \} _ { n = 1 } ^ { N }$ be a window of LR frames, and let $I _ { 0 } ^ { L }$ and $I _ { 1 } ^ { L }$ be the two anchor frames surrounding a query time $t \in [ 0 , 1 ]$ . Given an arbitrary spatial scale s, our goal is to reconstruct the HR intermediate frame $I _ { t } ^ { H }$

Our framework contains four main components. First, a long-context encoder uses all frames in $\mathcal { T } ^ { L }$ to produce context-enhanced features for the two anchors. Second, a continuous feature lifting module maps these LR anchor features to the requested HR grid. Third, PixelSpline constructs one explicit trajectory for every matched point from the first anchor to the second. At any query time, the trajectory has a unique position on the intermediate frame. We use this position to transport both anchor pixels and their lifted HR features, producing a coarse intermediate frame and an aligned HR feature. Finally, a lightweight decoder predicts an image residual from the aligned feature and adds it to the coarse frame.

We denote the encoder output by

$$
( \mathbf { F } _ { 0 } ^ { L } , \mathbf { F } _ { 0 1 } ^ { L } , \mathbf { F } _ { 1 } ^ { L } ) = E ( \mathcal { T } ^ { L } ) ,\tag{1}
$$

where $\mathbf { F } _ { 0 } ^ { L }$ and $\mathbf { F } _ { 1 } ^ { L }$ are the two anchor features, and $\mathbf { F } _ { 0 1 } ^ { L }$ describes their shared interval. The upsampling module produces $\mathbf { F } _ { 0 } ^ { H } , \mathbf { F } _ { 1 } ^ { H }$ and $\mathbf { F } _ { 0 1 } ^ { H }$ at scale s. PixelSpline then transports the anchor RGB values and HR features to time $t ,$ yielding a coarse frame $\widetilde { I } _ { t } ^ { H }$ and an aligned feature $\mathbf { F } _ { t } ^ { H }$ . The final result is

$$
\begin{array} { r } { \widehat { I } _ { t } ^ { H } = \widetilde { I } _ { t } ^ { H } + G _ { \theta } ( \mathbf { F } _ { t } ^ { H } , \mathbf { F } _ { 0 1 } ^ { H } , \widetilde { I } _ { t } ^ { H } , t , s ) , } \end{array}\tag{2}
$$

where $G _ { \theta }$ is the residual decoder. Fig. 2 illustrates the pipeline of our method.

## 3.2. Target-Pair-Aware Context Encoding

We adopt the native multi-frame encoder of Zooming Slow-Mo [27] to extract temporally aggregated features from the LR input sequence. We augment its output with a targetpair attention module. This module enables the two anchor frames to exchange features with relevant frames across the entire encoded sequence.

Multi-frame sequence encoding. Given N LR frames, the encoder first extracts a three-level feature pyramid for each frame, we then align the features of adjacent frames. The two aligned features are fused by a 1 × 1 convolution to obtain an interval feature. The original frame features and the interval features are arranged into an interleaved sequence,

$$
\mathcal { H } = \{ \mathbf { H } _ { 0 } , \mathbf { H } _ { 0 , 1 } , \mathbf { H } _ { 1 } , \mathbf { H } _ { 1 , 2 } , \ldots , \mathbf { H } _ { N - 1 } \} ,\tag{3}
$$

which contains 2N − 1 elements. A bidirectional deformable ConvLSTM propagates information through this sequence in both temporal directions. Consequently, every element in H contains multi-frame context while retaining its role as either a frame feature or an interval feature.

Target-pair dual-scope attention. The multi-frame encoder treats all temporal positions uniformly, whereas our subsequent reconstruction concerns one particular interval. Let a denote the position of the queried pair in the input window. We take its left-anchor, interval, and right-anchor features as the initial query,

$$
{ \bf Q } _ { a } ^ { 0 } = \left[ { \bf H } _ { a } , { \bf H } _ { a , a + 1 } , { \bf H } _ { a + 1 } \right] ,\tag{4}
$$

while the complete sequence H serves as key–value memory.

We refine this triplet at both local and global spatial scopes. The local branch applies cross-attention within regular non-overlapping windows and then within shifted windows. The first pass retrieves spatially corresponding evidence from all temporal positions, while the shifted pass enables information exchange across window boundaries. The global branch pools every feature map to a compact spatial grid and performs cross-attention over the pooled query and memory tokens. Its output is upsampled and added to the local features, providing scene-level context at a low computational cost. We also add temporal-offset embeddings to identify the distance of each memory element from the queried interval, and type embeddings to distinguish frame features from interval features.

The attention module returns the refined target triplet

$$
( \mathbf { F } _ { 0 } ^ { L } , \mathbf { F } _ { 0 1 } ^ { L } , \mathbf { F } _ { 1 } ^ { L } ) = A _ { \mathrm { c t x } } ( \mathbf { Q } _ { a } ^ { 0 } , \mathcal { H } ) ,\tag{5}
$$

where $\mathbf { F } _ { 0 } ^ { L }$ and $\mathbf { F } _ { 1 } ^ { L }$ are the context-enhanced anchor features and $\mathbf { F } _ { 0 1 } ^ { L }$ represents their interval.

## 3.3. Continuous Feature Upsampling

Our lifting module follows continuous upsampling filters (CUF) and the hyper-upsampling design used in ST-AVSR [22]. We adapt these image-reconstruction modules to our feature-transport setting: instead of producing RGB values, the module maps a C-channel LR anchor feature to a C-channel HR feature that will subsequently be warped by PixelSpline.

![](images/23e4154e37b5c5f240083c4f71a8a4f88c0f3c4d2d664de18e818f504c751423.jpg)  
Figure 2. Overview of our continuous STVSR framework. A bidirectional long-context encoder aggregates information from multiple LR frames to obtain context-enhanced endpoint representations (Sec.3.2). Continuous feature upsampling maps the endpoint features onto the requested HR grid (Sec.3.3), while an endpoint-anchored PixelSpline constructs a shared trajectory for RGB and feature transport at the queried time (Sec. 3.4). Finally, a lightweight decoder predicts an HR residual that is added to the coarse warped frame to produce the reconstruction.

Given an LR anchor feature $\mathbf { F } _ { i } ^ { L } , i \in \{ 0 , 1 \}$ , and an HR query coordinate x, we first obtain a bilinearly upsampled base feature

$$
\mathbf { B } _ { i } ^ { H } ( \mathbf { x } ) = \mathrm { B i l i n e a r } ( \mathbf { F } _ { i } ^ { L } , \mathbf { x } ) .\tag{6}
$$

The second branch predicts a continuous, depth-wise $3 \times 3$ filter for each HR query. Let $\mathbf { x } _ { r }$ be the nearest LR feature coordinate, $\Delta { \bf x } = { \bf x } - { \bf x } ,$ its relative offset, a the output cell size, and $\mathbf { o } _ { k }$ the spatial index of the k-th location in a $3 \times 3$ kernel. After sinusoidal encoding, these three geometric quantities are passed to a SIREN-based[23] kernel predictor:

$$
\mathbf { w } _ { k } ( \mathbf { x } , \mathbf { a } ) = K _ { \phi } ( \gamma _ { r } ( \Delta \mathbf { x } ) , \gamma _ { a } ( \mathbf { a } ) , \gamma _ { k } ( \mathbf { o } _ { k } ) ) , \qquad k = 1 , \ldots , 9 ,\tag{7}
$$

where $\gamma _ { r } , \gamma _ { a } .$ , and $\gamma _ { k }$ denote the three sinusoidal encoders. Each ${ \mathbf w } _ { k } \in \mathbb { R } ^ { C }$ contains one weight per feature channel. The filter depends only on the query geometry and output scale, not on the input feature values. Thus, the same filters can be shared by both anchors when they are lifted to the same target grid.

To apply the predicted filters, we unfold each LR feature into its $3 \times 3$ neighborhoods and bilinearly sample the unfolded tensor at all HR queries. Writing the resulting tensor as $\overline { { \mathcal { U } } } _ { i }$ and the collection of predicted filters as W, the CUF

reconstruction is

$$
{ \bf R } _ { i } ^ { H } = \mathrm { F o l d } \big ( { \bf W } \odot \overline { { \mathcal { U } } } _ { i } \big ) .\tag{8}
$$

$H _ { \psi }$ transforms $\mathbf { R } _ { i } ^ { H }$ into a feature residual with $\mathbf { a \ 1 \times 1 }$ convolution, a LeakyReLU, and a $3 \times 3$ convolution. The final lifted feature is

$$
\mathbf { F } _ { i } ^ { H } = \mathbf { B } _ { i } ^ { H } + H _ { \psi } ( \mathbf { R } _ { i } ^ { H } ) .\tag{9}
$$

This residual formulation retains the bilinearly lifted anchor content while allowing the continuous filters to recover scale-dependent feature detail. The same lifting module is applied to both anchors.

## 3.4. Shared PixelSpline Modeling and Rendering

Endpoint correspondence. Inspired by SGM-VFI[19] and GMFlow[29] We first estimate a dense correspondence between the two anchor frames. A local flow estimator (SEA-RAFT[26] in our pipeline) supplies an initial match and its confidence. To correct unreliable regions, we measure the reconstruction error after warping one anchor to the other and perform global matching with frozen GMFlow at locations with large errors. The local and global candidates are merged and refined using the context-enhanced anchor features. This produces bidirectional endpoint flows ${ \bf { M } } _ { 0  1 }$ and $\bf { M } _ { 1  0 }$ together with their confidence maps. These endpoint correspondences determine which two pixels should be connected by PixelSpline.

A unique trajectory between two matched pixels. Consider a pixel at position p in the first anchor and its matched position

$$
\mathbf { q } = \mathbf { p } + \mathbf { M } _ { 0  1 } ( \mathbf { p } )\tag{10}
$$

in the second anchor. Instead of predicting two independent motions from p and q to the target frame, we explicitly connect them with one cubic curve. The motion head predicts two offsets $\delta _ { 1 }$ and $\delta _ { 2 }$ from the anchor features. We define four control points

$$
\begin{array} { l l } { { { \bf { C } } _ { 0 } = { \bf { p } } , } } & { { { \bf { C } } _ { 1 } = { \bf { p } } + \frac { 1 } { 3 } ( { \bf { q } } - { \bf { p } } ) + \delta _ { 1 } , } } \\ { { { \bf { C } } _ { 2 } = { \bf { p } } + \frac { 2 } { 3 } ( { \bf { q } } - { \bf { p } } ) + \delta _ { 2 } , } } & { { { \bf { C } } _ { 3 } = { \bf { q } } . } } \end{array}\tag{11}
$$

The resulting clamped cubic B-spline is equivalently a cubic Bezier segment:´

$$
\mathbf { P _ { p } } ( t ) = \sum _ { k = 0 } ^ { 3 } { \binom { 3 } { k } } ( 1 - t ) ^ { 3 - k } t ^ { k } \mathbf { C } _ { k } .\tag{12}
$$

Because $\mathbf { C } _ { 0 } = \mathbf { p }$ and $\mathbf { C } _ { 3 } = \mathbf { q } .$ , the curve passes through the two matched endpoints exactly.

The geometric meaning of Eq. (12) is simple: the time-t frame can be viewed as a plane that intersects the trajectory at one point, $\mathbf { p } _ { t } = \mathbf { P _ { p } } ( t )$ . We therefore obtain the two forward motions by measuring displacement to the same intersection:

$$
\mathbf { M } _ { 0  t } ( \mathbf { p } ) = \mathbf { p } _ { t } - \mathbf { p } , \qquad \mathbf { M } _ { 1  t } ( \mathbf { q } ) = \mathbf { p } _ { t } - \mathbf { q } .\tag{13}
$$

It immediately follows that

$$
\mathbf { p } + \mathbf { M } _ { 0  t } ( \mathbf { p } ) = \mathbf { q } + \mathbf { M } _ { 1  t } ( \mathbf { q } ) = \mathbf { p } _ { t } .\tag{14}
$$

Thus, corresponding pixels transported from the two anchors must meet at the same location in every intermediate frame, ensuring consistent motion trajectories between the two anchors. Visibility weights are predicted separately to determine how much each endpoint contributes when a point is occluded or multiple points arrive at the same target location.

Pixel and feature warping. The shared trajectory is used for two forward-warping operations. First, we bicubically upsample the anchor RGB frames and forward-warp them with ${ \bf { M } } _ { 0  t }$ and $\mathbf { M } _ { 1  t }$ . Confidence-weighted soft splatting[21] combines the two contributions into a coarse frame $\tilde { I } _ { t } ^ { H }$ . This frame provides reliable color and lowfrequency structure, although it may be blurry around motion boundaries or uncovered regions.

Second, we apply the same two motions to the lifted features $\mathbf { F } _ { 0 } ^ { H }$ and $\mathbf { F } _ { 1 } ^ { \hat { H } }$ :

$$
\mathbf { F } _ { t } ^ { H } = \operatorname { S o f t S p l a t } ( \mathbf { F } _ { 0 } ^ { H } , \mathbf { F } _ { 1 } ^ { H } ; \mathbf { M } _ { 0  t } , \mathbf { M } _ { 1  t } ) .\tag{15}
$$

Pixel warping and feature warping therefore share exactly the same geometry. The former gives an interpretable image-space estimate, while the latter brings detailed HR appearance evidence to the same target locations.

Residual reconstruction. Conditioned on (t, s), a lightweight pointwise decoder combines the warped appearance and feature representation with the upsampled interval-context feature $\mathbf { F } _ { 0 1 } ^ { H }$ to predict a bounded RGB residual. Adding this residual to $\widetilde { I } _ { t } ^ { H }$ gives the final output in Eq. (2). In this way, explicit warping handles motion and color transport, while the decoder focuses on disocclusions, splatting artifacts, and fine details.

## 3.5. Training Objective

We train the network with four objectives. First, the final reconstruction is supervised by a Charbonnier loss

$$
\mathcal { L } _ { \mathrm { i m g } } = \rho ( \widehat { I } _ { t } ^ { H } - I _ { t } ^ { H } ) , \qquad \rho ( x ) = \sqrt { x ^ { 2 } + \epsilon ^ { 2 } } .\tag{16}
$$

Second, when several target times are sampled from the same interval, we match their first-order temporal changes:

$$
\mathcal { L } _ { \mathrm { t e m p } } = \rho \Big ( \big ( \widehat { I } _ { t _ { j + 1 } } ^ { H } - \widehat { I } _ { t _ { j } } ^ { H } \big ) - \big ( I _ { t _ { j + 1 } } ^ { H } - I _ { t _ { j } } ^ { H } \big ) \Big ) .\tag{17}
$$

Third, we directly supervise the motions predicted by PixelSpline using pseudo-ground-truth optical flows generated by a frozen SEA-RAFT model. Let $\mathbf { M } _ { i  t } ^ { \ast }$ be the optical flow estimated by SEA-RAFT from anchor i to time t. The flow loss is

$$
\mathcal { L } _ { \mathrm { { f l o w } } } = \frac { 1 } { 2 } \sum _ { i \in \{ 0 , 1 \} } \rho \biggl ( \frac { \mathbf { M } _ { i \to t } - \mathbf { M } _ { i \to t } ^ { * } } { ( W _ { s } , H _ { s } ) } \biggr ) ,\tag{18}
$$

where the horizontal and vertical errors are normalized by the output width and height, respectively, so that the loss is comparable across spatial scales. Finally, the coarse warped frame is encouraged to match a blurry target obtained by downsampling and then upsampling the HR ground truth:

$$
\mathcal { L } _ { \mathrm { w a r p } } = \rho ( \widetilde { I } _ { t } ^ { H } - I _ { t } ^ { \mathrm { b l u r } } ) .\tag{19}
$$

This term supervises the explicit pixel-transport path without forcing it to reconstruct details that should be recovered by the residual decoder.

The complete objective is

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { i m g } } + \lambda _ { \mathrm { t e m p } } \mathcal { L } _ { \mathrm { t e m p } } + \lambda _ { \mathrm { f l o w } } \mathcal { L } _ { \mathrm { f l o w } } + \lambda _ { \mathrm { w a r p } } \mathcal { L } _ { \mathrm { w a r p } } .\tag{20}
$$

## 4. Experiments

We evaluate our method on the full continuous spacetime video super-resolution (C-STVSR) task, as well as on spatial-only and temporal-only reconstruction. We report reconstruction accuracy, temporal consistency, and end-toend runtime to assess the quality achieved by each method relative to its inference cost.

Table 1. Quantitative comparison of C-STVSR reconstruction quality under S×4 spatial and T×8 temporal upsampling, reported as PSNR ↑ / SSIM ↑. Best and second-best results are highlighted in bold and underlined, respectively.
<table><tr><td>Method</td><td>GoPro Center</td><td>GoPro Average</td><td>Adobe Center</td><td>Adobe Average</td><td>REDS Center</td><td>REDS Average</td></tr><tr><td>Zooming SlowMo [27]</td><td>30.69 / 0.885</td><td></td><td>30.26 / 0.882</td><td></td><td></td><td></td></tr><tr><td>TMNet [28]</td><td>24.94 / 0.840</td><td>24.76 / 0.824</td><td>27.25 / 0.843</td><td>26.65 / 0.827</td><td>22.64 / 0.667</td><td>19.84 / 0.573</td></tr><tr><td>VideoINR [6]</td><td>30.26 / 0.879</td><td>29.41 / 0.867</td><td>29.92 / 0.875</td><td>29.27 / 0.865</td><td>23.25 / 0.650</td><td>19.85 / 0.558</td></tr><tr><td>MoTIF [5]</td><td>31.12 / 0.889</td><td>30.14 / 0.879</td><td>30.63 / 0.884</td><td>29.82 / 0.875</td><td>23.62 / 0.654</td><td>20.01 / 0.556</td></tr><tr><td>BF-STVSR [14]</td><td>31.22 / 0.890</td><td>30.26 / 0.881</td><td>30.83 / 0.888</td><td>30.12 / 0.880</td><td>23.55 / 0.646</td><td>19.95 / 0.549</td></tr><tr><td>V3 [3]</td><td>32.98 / 0.923</td><td>32.27 / 0.919</td><td>32.91 / 0.922</td><td>32.29 / 0.917</td><td>24.00 / 0.671</td><td>20.36 / 0.573</td></tr><tr><td>Ours</td><td>32.21 / 0.912</td><td>31.47 / 0.906</td><td>31.99 / 0.913</td><td>31.39 / 0.907</td><td>23.78 / 0.651</td><td>20.04 / 0.549</td></tr></table>

## 4.1. Experimental Setup

Implementation and training details. We train our model in PyTorch on four NVIDIA H100 GPUs with a total batch size of 8. Each sample contains eight temporally subsampled context frames. For supervision, the first and last frames of a nine-frame high-frame-rate sequence define the endpoint pair, and three timestamps are randomly sampled from the sequence as targets. Training uses 64 × 64 lowresolution crops with random flipping and rotation. We train the model with Adam optimizer $( \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 )$ , using an initial learning rate of $1 0 ^ { - 4 }$ and cosine annealing to $1 0 ^ { - 7 }$ . The pretrained SEA-RAFT [26] and GMFlow [29] components in Shared PixelSpline Modeling remain frozen, while all other components are trained from scratch.

Datasets. Following the standard C-STVSR protocol [3, 5, 6, 14], we train on the training split of Adobe240 [24]. Adobe240 contains 133 high-frame-rate videos captured at 240 fps with a spatial resolution of $1 2 8 0 \times 7 2 0$ . We construct the low-resolution inputs by bicubic spatial downsampling and temporally subsample the videos by a factor of 8, producing 30-fps input sequences. The remaining high-frame-rate frames provide supervision at continuous temporal coordinates. We report in-distribution results on the Adobe240 test split and cross-dataset results on Go-Pro [15] and REDS [20]. We additionally use Vid4 [18] to evaluate the spatial-only and temporal-only boundary cases. No dataset-specific fine-tuning is performed for Go-Pro, REDS, or Vid4.

Baseline methods. We compare against six representative STVSR methods. Zooming Slow-Mo [27] and TMNet [28] are fixed-scale one-stage models, whereas VideoINR [6], MoTIF [5], BF-STVSR [14], and ${ \mathrm { V } } ^ { 3 }$ [3] support continuous space-time reconstruction. Fixed-scale methods are evaluated only at scaling factors supported by their pretrained models; continuous methods are queried directly at the requested spatial and temporal coordinates.

Evaluation metrics. We evaluate reconstruction quality using PSNR and SSIM. In the joint C-STVSR evaluation, “Center” measures the input keyframes and the center frame of each interval, whereas “Average” includes every reconstructed output frame. The Adobe240, GoPro, and REDS experiments use $\mathrm { S } \times 4$ spatial and $\mathrm { T } \times \mathrm { 8 }$ temporal upsampling. For the boundary-case evaluation on Vid4, we separately set the temporal factor to ×1 for spatial-only SR and the spatial factor to ×1 for temporal-only SR. We further report temporal optical-flow consistency (tOF), end-to-end latency, and FPS.

## 4.2. Quality–Efficiency Trade-off

Tab. 1 compares reconstruction accuracy on the full C-STVSR task. ${ \mathrm { V } } ^ { 3 }$ achieves the highest PSNR, and our method ranks second in all six dataset–protocol combinations. The PSNR difference between the two methods is 0.77/0.80 dB on GoPro Center/Average, 0.92/0.90 dB on Adobe240 Center/Average, and 0.22/0.32 dB on REDS Center/Average. Our method also outperforms every other baseline in PSNR across all reported columns.

Tab. 2 reports end-to-end latency and throughput under the timing protocol described in the caption. As shown in Fig. 1, our method requires 406.41 ms per output frame, compared with 912.21 ms for ${ \mathrm { V } } ^ { 3 }$ . This corresponds to a 2.24× speedup, with a 0.90 dB lower PSNR on Adobe240 Average. Methods with lower latency than ours operate at lower reconstruction accuracy, while MoTIF and BF-STVSR are both slower and less accurate. Our method therefore provides a practical balance: it approaches the reconstruction quality of ${ \mathrm { V } } ^ { 3 }$ while requiring less than half of its end-to-end inference time.

Qualitative comparison. Fig. 3 presents qualitative comparisons under $\mathrm { S } \times 4$ spatial and $\mathrm { ~ T ~ } \times \mathrm { ~ 8 ~ }$ temporal upsampling. The first three rows show results on GoPro, while the last two rows show results on the more challenging REDS dataset. Across both datasets, our method better preserves fine textures and high-frequency structures. In the GoPro examples, it reconstructs sharper characters, whereas ${ \mathrm { V } } ^ { 3 }$ and BF-STVSR produce noticeably blurred text. This advantage remains evident on REDS: our method faithfully recovers the fine stripe patterns on the tiger and preserves the folded structure of the pedestrian’s shirt collar with smooth, well-defined boundaries. In comparison, ${ \mathrm { V } } ^ { 3 }$ and

![](images/87c5bf2d2dfd725e56c3f86b965f5a156cd61e6fe006a7db1fdd3e5775c949aa.jpg)  
GT  
Ours  
v³  
BF-STVSR  
VideoINR  
MoTIF  
Figure 3. Qualitative comparison of C-STVSR methods under S×4 spatial and T×8 temporal upsampling. The first three rows show results on GoPro, and the last two rows show results on REDS. Our method preserves fine details such as characters, texture patterns, and clothing structures while producing sharp and smooth boundaries. Best viewed zoomed in.

Table 2. End-to-end inference efficiency on Adobe240 after one full-resolution warm-up pass. Latency includes input loading and preprocessing, device transfers, all method-specific inference operations, image conversion, PNG encoding, and disk writing.
<table><tr><td>Method</td><td>Latency (ms/frame) ↓ FPS ↑</td></tr><tr><td>Zooming SlowMo [27] 354.72</td><td>2.82</td></tr><tr><td>TMNet [28] 81.44</td><td>12.28</td></tr><tr><td>VideoINR [6] 360.58</td><td>2.77</td></tr><tr><td>MoTIF [5] 494.65</td><td>2.02</td></tr><tr><td>BF-STVSR [14] 417.51</td><td>2.40</td></tr><tr><td>v³ [3] 912.21</td><td>1.10</td></tr><tr><td>Ours 406.41</td><td>2.46</td></tr></table>

VideoINR over-smooth the collar and fail to separate its structural components, while BF-STVSR and MoTIF recover only its coarse shape and introduce jagged artifacts along the edges. These results demonstrate the ability of our method to preserve fine appearance details and coherent object boundaries across different data distributions.

## 4.3. Temporal Consistency

Table 3. Temporal consistency on REDS under S×4 spatial and T×8 temporal upsampling, measured by tOF ↓.
<table><tr><td>Method</td><td>TMNet</td><td>VideoINR</td><td>MoTIF</td><td>BF-STVSR</td><td>V3</td><td>Ours</td></tr><tr><td>tOF↓</td><td>9.944</td><td>9.495</td><td>9.391</td><td>9.237</td><td>8.923</td><td>8.171</td></tr></table>

Following prior C-STVSR evaluation [3], we compute tOF as the endpoint error between the optical flow of adjacent reconstructed frames and the corresponding groundtruth flow. Lower tOF indicates that the reconstructed motion is closer to the reference and is therefore less affected by temporal artifacts such as flicker or irregular motion. Tab. 3 shows that our method achieves the lowest tOF on

![](images/4bc18ebf2d3d8af15d10e08cfdcc3ae2e191632c424c4ad3638a89f57f7973d9.jpg)  
Figure 4. Qualitative comparison of temporal consistency. The red dashed line marks the vertical pixel column sampled from each frame to construct the temporal profile, while the red arrows indicate regions with noticeable deviations from the ground truth. Our method faithfully reconstructs fine details along nonlinear motion trajectories while suppressing motion-induced artifacts.

REDS, reducing the error from 8.923 for V<sup>3</sup> to 8.171. Fig. 4 illustrates qualitative results of temporal consistency, our method successfully reconstructs temporal motion while suppressing motion artifacts.

## 4.4. Decoupled Spatial and Temporal Reconstruction

Although our model is designed for joint space–time superresolution, its continuous formulation supports both boundary cases without architectural changes. Setting the temporal factor to ×1 isolates spatial super-resolution, while setting the spatial factor to ×1 isolates temporal interpolation.

Table 4. Decoupled spatial-only and temporal-only superresolution results on Vid4 (PSNR ↑ / SSIM ↑). Temporal-only results are evaluated on center frames. Best and second-best results are highlighted in bold and underlined, respectively.
<table><tr><td>Setting</td><td>VideoINR [6]</td><td>MoTIF [5]</td><td>BF-STVSR [14]</td><td>V3 [3]</td><td>Ours</td></tr><tr><td>S×4, T×1</td><td>25.97 / 0.775</td><td>26.21 / 0.780</td><td>26.28 / 0.782</td><td>27.15 / 0.819</td><td>26.78 / 0.808</td></tr><tr><td>S×1, T×8</td><td>19.76 / 0.462</td><td>28.86 / 0.864</td><td>29.42 / 0.810</td><td>31.15 / 0.862</td><td>29.01 / 0.890</td></tr></table>

Tab. 4 reports both settings on Vid4. For spatial-only S× 4, T×1 reconstruction, our method obtains 26.78 dB/0.808, ranking second behind V<sup>3</sup>. This result shows that the continuous feature-lifting module remains effective when temporal synthesis is disabled. For temporal-only S×1, T×8 interpolation, our method achieves the highest SSIM (0.890), although its PSNR (29.01 dB) performs below V<sup>3</sup> and BF-STVSR. Overall, the two boundary cases show that the model remains competitive when spatial and temporal reconstruction are evaluated separately.

## 4.5. Ablation study.

We evaluate all variants using their checkpoints at 10,000 iterations on the same randomly selected subset of the Adobe240 test set, comprising 3,858 frames under S×4 spatial and T×8 temporal upsampling. The full model achieves 32.49 dB PSNR and 0.930 SSIM. Replacing the shared bidirectional trajectory controls with independently predicted forward and backward controls, while retaining the longcontext architecture, reduces performance to 32.25 dB and 0.924, corresponding to a 0.24 dB PSNR drop. Removing the additional context frames and using only the two anchor frames further decreases performance to 31.80 dB and 0.918, a 0.69 dB drop relative to the full model. These results verify that both endpoint-consistent control sharing and long-range temporal context contribute to reconstruction quality.

## 5. Conclusion

We presented a continuous space-time video superresolution framework that combines long-context encoding, continuous HR feature upsampling, and an endpoint-anchored PixelSpline. By constructing a single correspondence-consistent trajectory shared by both endpoints, our method aligns pixel and feature transport while exploiting observations beyond the queried frame pair. Experiments across joint, spatial-only, and temporalonly super-resolution settings demonstrate competitive reconstruction quality, improved temporal consistency, and a favorable quality–efficiency trade-off.

Despite these results, the reconstruction performance in terms of PSNR and SSIM can be further improved, particularly compared with the quality-leading method in several settings. Moreover, although our method offers practical inference efficiency, its training process remains computationally demanding and time-consuming. Improving reconstruction accuracy while developing more efficient training strategies constitutes a promising direction for future work.

## 6. Acknowledgment

This work was supported in part by the U.S. National Science Foundation (NSF) under Awards IIS-2232298, IIS-2442792, and OIA-2242812.

## References

[1] Wenbo Bao, Wei-Sheng Lai, Chao Ma, Xiaoyun Zhang, Zhiyong Gao, and Ming-Hsuan Yang. Depth-aware video frame interpolation. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3698– 3707. IEEE, 2019. 2

[2] Alexander Becker, Rodrigo Caye Daudt, Dominik Narnhofer, Torben Peters, Nando Metzger, Jan Dirk Wegner, and Konrad Schindler. Thera: Aliasing-free arbitrary-scale super-resolution with neural heat fields. arXiv preprint arXiv:2311.17643, 2023. 2

[3] Alexander Becker, Julius Erbach, Dominik Narnhofer, and Konrad Schindler. Continuous space-time video super-resolution with 3d fourier fields. arXiv preprint arXiv:2509.26325, 2025. 1, 3, 6, 7, 8

[4] Yinbo Chen, Sifei Liu, and Xiaolong Wang. Learning continuous image representation with local implicit image function. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8628–8638, 2021. 2

[5] Yi-Hsin Chen, Si-Cun Chen, Yen-Yu Lin, and Wen-Hsiao Peng. Motif: Learning motion trajectories with local implicit neural functions for continuous space-time video superresolution. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 23074–23084. IEEE, 2023. 1, 2, 6, 7, 8

[6] Zeyuan Chen, Yinbo Chen, Jingwen Liu, Xingqian Xu, Vidit Goel, Zhangyang Wang, Humphrey Shi, and Xiaolong Wang. Videoinr: Learning video implicit neural representation for continuous space-time super-resolution. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2037–2047. IEEE, 2022. 1, 2, 6, 7, 8

[7] Hee Min Choi, Hyoa Kang, Suji Kim, Dokwan Oh, and Nam Ik Cho. Time without time: Pseudo-temporal representation for space-time super-resolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6812–6822, 2026. 2

[8] Zhicheng Geng, Luming Liang, Tianyu Ding, and Ilya Zharkov. Rstt: Real-time spatial temporal transformer for space-time video super-resolution. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 17420–17430. IEEE, 2022. 2

[9] Muhammad Haris, Greg Shakhnarovich, and Norimichi Ukita. Space-time-aware multi-resolution video enhancement. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2856–2865. IEEE, 2020.

[10] Mengshun Hu, Kui Jiang, Zhixiang Nie, and Zheng Wang. You only align once: Bidirectional interaction for spatialtemporal video super-resolution. In Proceedings of the 30th ACM International Conference on Multimedia, pages 847– 855, 2022.

[11] Mengshun Hu, Kui Jiang, Zhixiang Nie, Jiahuan Zhou, and Zheng Wang. Store and fetch immediately: Everything is all you need for space-time video super-resolution. In Proceed-

ings of the AAAI Conference on Artificial Intelligence, pages 863–871, 2023. 2

[12] Xuecai Hu, Haoyuan Mu, Xiangyu Zhang, Zilei Wang, Tieniu Tan, and Jian Sun. Meta-sr: A magnification-arbitrary network for super-resolution. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1575–1584. IEEE, 2019. 2

[13] Huaizu Jiang, Deqing Sun, Varun Jampani, Ming-Hsuan Yang, Erik Learned-Miller, and Jan Kautz. Super slomo: High quality estimation of multiple intermediate frames for video interpolation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 9000– 9008, 2018. 2

[14] Eunjin Kim, Hyeonjin Kim, Kyong Hwan Jin, and Jaejun Yoo. Bf-stvsr: B-splines and fourier—best friends for high fidelity spatial-temporal video super-resolution. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 28009–28018. IEEE, 2025. 1, 2, 6, 7, 8

[15] Tae Hyun Kim, Seungjun Nah, and Kyoung Mu Lee. Deep multi-scale convolutional neural network for dynamic scene deblurring. In Conference on Computer Vision and Pattern Recognition, pages 1–21. IEEE, 2017. 6

[16] Jaewon Lee and Kyong Hwan Jin. Local texture estimator for implicit representation function. In 2022 IEEE/CVF Confer ence on Computer Vision and Pattern Recognition (CVPR), pages 1919–1928. IEEE, 2022. 2

[17] Zekun Li, Hongying Liu, Fanhua Shang, Yuanyuan Liu, Liang Wan, and Wei Feng. Savsr: Arbitrary-scale video super-resolution via a learned scale-adaptive network. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 3288–3296, 2024. 2

[18] Ce Liu and Deqing Sun. A bayesian approach to adap tive video super resolution. In CVPR 2011, pages 209–216. IEEE, 2011. 6

[19] Chunxu Liu, Guozhen Zhang, Rui Zhao, and Limin Wang. Sparse global matching for video frame interpolation with large motion. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 19125– 19134. IEEE, 2024. 4

[20] Seungjun Nah, Sungyong Baik, Seokil Hong, Gyeongsik Moon, Sanghyun Son, Radu Timofte, and Kyoung Mu Lee. Ntire 2019 challenge on video deblurring and superresolution: Dataset and study. In CVPR Workshops, 2019. 6

[21] Simon Niklaus and Feng Liu. Softmax splatting for video frame interpolation. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5436– 5445. IEEE, 2020. 5

[22] Wei Shang, Dongwei Ren, Wanying Zhang, Yuming Fang, Wangmeng Zuo, and Kede Ma. Arbitrary-scale video super resolution with structural and textural priors. In European Conference on Computer Vision, pages 73–90. Springer, 2024. 2, 3

[23] Vincent Sitzmann, Julien Martel, Alexander Bergman, David Lindell, and Gordon Wetzstein. Implicit neural representations with periodic activation functions. Advances in neural information processing systems, 33:7462–7473, 2020. 4

[24] Shuochen Su, Mauricio Delbracio, Jue Wang, Guillermo Sapiro, Wolfgang Heidrich, and Oliver Wang. Deep video deblurring for hand-held cameras. In 2017 IEEE Conference on computer vision and pattern recognition (CVPR), pages 237–246. IEEE, 2017. 6

[25] Xintao Wang, Kelvin CK Chan, Ke Yu, Chao Dong, and Chen Change Loy. Edvr: Video restoration with enhanced deformable convolutional networks. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 1954–1963. IEEE, 2019. 2

[26] Yihan Wang, Lahav Lipson, and Jia Deng. Sea-raft: Simple, efficient, accurate raft for optical flow. In European Conference on Computer Vision, pages 36–54. Springer, 2024. 4, 6

[27] Xiaoyu Xiang, Yapeng Tian, Yulun Zhang, Yun Fu, Jan P Allebach, and Chenliang Xu. Zooming slow-mo: Fast and accurate one-stage space-time video super-resolution. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3367–3376. IEEE, 2020. 1, 2, 3, 6, 7

[28] Gang Xu, Jun Xu, Zhen Li, Liang Wang, Xing Sun, and Ming-Ming Cheng. Temporal modulation network for controllable space-time video super-resolution. arXiv preprint arXiv:2104.10642, 2021. 1, 2, 6, 7

[29] Haofei Xu, Jing Zhang, Jianfei Cai, Hamid Rezatofighi, and Dacheng Tao. Gmflow: Learning optical flow via global matching. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8111–8120. IEEE, 2022. 4, 6

[30] Yulun Zhang, Kunpeng Li, Kai Li, Lichen Wang, Bineng Zhong, and Yun Fu. Image super-resolution using very deep residual channel attention networks. In European conference on computer vision, pages 294–310. Springer, 2018. 2