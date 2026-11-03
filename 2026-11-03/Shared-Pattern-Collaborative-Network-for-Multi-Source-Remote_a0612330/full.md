# Shared-Pattern Collaborative Network for Multi-Source Remote Sensing Data Classification

Chenfang Yang, Chuanzheng Gong, Feng Gao, Member, IEEE, Junyu Dong, Member, IEEE, and Qian Du, Fellow, IEEE

Abstract—Multi-source remote sensing data classification has received growing attention recently. However, existing methods still struggle to jointly model long-range contextual dependencies and shared spatial patterns across heterogeneous modalities. To address this issue, we propose a Shared-Pattern Collaborative Network (SPCNet) for HSI and SAR/LiDAR joint classification. To enhance cross-modal collaborative learning, Shared-Pattern Collaborative Block (SPCB) is designed by introducing a shared large-kernel operator and a dynamic convolution branch, enabling the network to capture common spatial structures and content-adaptive local features. Furthermore, Region-Adaptive Modality Allocator (RAMA) is developed to assign spatially varying modality weights for reliable feature fusion. Experiments on Houston2013, Augsburg, and Houston2018 datasets demonstrate that SPCNet achieves superior classification performance over several state-of-the-art methods. The code will be made publicly available at https://github.com/oucailab/SPCNet.

Index Terms—Multi-source data fusion; Hyperspectral image; Synthetic aperture radar; Light detection and ranging; Sharedpattern collaborative learning.

## I. INTRODUCTION

R <sup>EMOTE</sup> <sup>sensing</sup> <sup>image</sup> <sup>classification</sup> <sup>has</sup> <sup>become</sup> <sup>a</sup> <sup>fun-</sup> damental task in Earth observation [1]. It has attracted considerable attention in recent years owing to its wideranging applications in environmental monitoring [2], urban planning [3], and disaster management [4]. With the increasing availability of heterogeneous Earth observation data, the task has expanded from single-source pixel recognition toward multimodal scene understanding. However, different sensors measure distinct physical properties and exhibit different spatial, spectral, and noise characteristics, making reliable joint representation learning a persistent challenge.

Land-cover classification is not merely the recognition of isolated pixels. Natural and urban scenes contain objects at markedly different spatial scales, including compact buildings, elongated roads, fragmented vegetation, and mixed transition regions. Their spectra may exhibit high intra-class variability and strong inter-class similarity because of illumination changes, material mixtures, seasonal conditions, and sensor noise [5]. An effective classifier must therefore jointly exploit diagnostic spectral signatures, local structures, and broader spatial context [6], [7].

![](images/9595c7dd00d40d65316e4243d0406a271c6ea84238c9e9193397cd58a7ed7e65.jpg)  
Fig. 1. Comparison between existing multi-source data classification models and the proposed SPCNet. SPCNet shares a large-kernel operator across the two modality branches to collaboratively learn common spatial patterns while retaining modality-specific features for adaptive fusion.

Hyperspectral images (HSIs) acquire hundreds of contiguous spectral bands across the electromagnetic spectrum. Such rich spectral information provides strong capability for identifying complex objects [8]–[11], but neighboring bands are highly correlated and limited spatial resolution may blur object boundaries. The classification performance of HSI is therefore often constrained by spectral similarity among different land-cover classes, noise contamination, and spectral redundancy. Integrating HSI with other modalities has consequently become an effective strategy for improving classification performance [12]. Synthetic aperture radar (SAR) and Light Detection and Ranging (LiDAR) data are particularly attractive because of their distinct imaging mechanisms and rich structural information [13]–[15]. SAR captures geometric and textural characteristics under all-weather and allday conditions, while LiDAR provides precise elevation and three-dimensional structural information. Their combination with HSI enables complementary spectral, scattering, and structural cues to be exploited jointly, as demonstrated by large multimodal benchmark studies [16]. Motivated by these advantages, this paper investigates the joint classification of HSI and SAR/LiDAR data.

The benefit of multi-source observation does not simply arise from stacking additional channels. HSI records reflectance, SAR characterizes backscattering related to surface roughness and imaging geometry, whereas LiDAR directly measures elevation and three-dimensional structure. Their relationship is therefore established through land-cover semantics rather than homogeneous signal statistics. Moreover, source reliability is spatially nonuniform: HSI is affected by shadows and spectral mixing, SAR by speckle and layover, and LiDAR may be less informative over locally flat surfaces. Differences in spatial resolution, noise distribution, and physical response consequently create a substantial cross-modal gap [17]–[20]. A successful fusion model should preserve modality-specific evidence while discovering structural regularities shared across sensors, rather than propagating redundant or unreliable responses through indiscriminate aggregation.

Multi-source classification methods have consequently evolved from decision-level fusion toward feature-level collaborative learning. Decision-level approaches combine predictions produced by independently trained classifiers and are relatively robust to heterogeneous feature dimensions, but their late interaction cannot fully exploit intermediate correlations between spectral, scattering, and elevation cues. Feature-level methods use parallel encoders and exchange information during representation learning, allowing complementary cues to participate directly in the construction of class-discriminative features [21], [22]. CNN-based architectures provide strong local inductive biases, while attention mechanisms and Transformer-based models improve global interaction and long-range dependency modeling [23]–[25]. Shared-and-specific representation learning and masked modeling further demonstrate that cross-source consistency and modality individuality should be considered jointly [15], [18]. Nevertheless, increasing network depth or introducing global attention does not automatically guarantee that heterogeneous branches learn compatible spatial structures, nor does it determine which source should dominate at every location.

As illustrated in Fig. 1, existing multi-source models typically extract HSI and SAR/LiDAR features using independent operators before combining the resulting representations. Such separate processing can overlook the common spatial structures shared by co-registered observations. SPCB selectively shares a patch-spanning spatial operator across co-registered modalities and block depths, while modality-specific stems, identity channels, and adaptive local filtering retain sourcedependent features. This design complements shared-specific and cross-modal learning [18].

Deep learning-based methods have become the mainstream solution for multi-source remote sensing classification. Existing approaches are primarily based on convolutional neural networks (CNNs) [21], [26], Transformers [27], [28], or their hybrid architectures to exploit complementary information. Despite their promising performance, two challenges remain insufficiently addressed: 1) Difficulty in simultaneously capturing long-range dependencies and shared spatial patterns between heterogeneous modalities. Many existing methods process different modalities independently or simply fuse them at a late stage, which limits their ability to discover common spatial structures across modalities in feature learning. Consequently, the shared contextual patterns and complementary structural cues across heterogeneous modalities remain insufficiently explored. 2) Limited region-adaptive feature fusion. Most methods adopt fixed or globally learned fusion schemes, which fail to dynamically adjust the contribution of each modality across different regions. Consequently, regionspecific complementary information may be insufficiently exploited, while redundant or unreliable modality features can be introduced into the fused representation.

These two challenges are closely coupled. If the feature extractors of different sources model context using unrelated spatial operators, subsequent fusion must align not only heterogeneous physical measurements but also incompatible spatial responses. Conversely, even well-aligned representations can be degraded when a globally fixed fusion rule assigns excessive importance to an unreliable modality in a particular region. Co-registered HSI and SAR/LiDAR observations share object boundaries, neighborhood organization, and scene-level geometry, suggesting that common spatial patterns can serve as a stable inductive prior for cross-modal learning. At the same time, the appearance of these patterns remains content dependent, and the discriminative contribution of each source changes with land-cover type and local observation conditions. Therefore, the desired framework should combine shared contextual modeling with content-adaptive local representation, and should perform fusion according to spatially varying source reliability rather than a single scene-level preference.

To overcome these limitations, this paper proposes a Shared-Pattern Collaborative Network (SPCNet). To promote effective contextual feature learning, we design Shared-Pattern Collaborative Block (SPCB), in which a shared large-kernel operator is applied to both branches to capture common spatial patterns and enhance long-range contextual modeling. Meanwhile, an input-adaptive dynamic convolution is introduced to strengthen content-aware local feature representation. Furthermore, we introduce Region-Adaptive Modality Allocator (RAMA) to dynamically estimate the contribution of each modality in different spatial regions. Instead of using fixed fusion weights, RAMA adaptively assigns region-specific modality importance, allowing the network to emphasize the more reliable modality according to local scene characteristics.

The main contributions of this work are threefold:

• We design SPCB to enhance cross-modal collaborative contextual feature learning. By introducing a shared large-kernel operator, SPCB encourages different modalities to learn common spatial patterns, while the dynamic convolution branch provides content-adaptive feature modeling.

• We introduce RAMA for adaptive multi-source feature fusion. It dynamically assigns modality-specific weights to different spatial regions, allowing the network to emphasize the more reliable modality according to local scene characteristics.

![](images/a4a9717d86729cc2352a12108293f64975b7479ebeb05a8ccaacabf3811fd1fa.jpg)  
Fig. 2. Overview of the proposed Shared-Pattern Collaborative Network (SPCNet) for multi-source remote sensing data classification. The input multi source data are processed by stacked Shared-Pattern Collaborative Blocks (SPCB) that exchange and learn common spatial-spectral representations through shared large kernels. Region-Adaptive Modality Allocator (RAMA) dynamically integrates complementary modality information to generate the classification map.

• Comprehensive experiments conducted on three benchmark datasets verify the effectiveness of the proposed SPCNet. In addition, we will publicly release the source code to facilitate the development of multi-source data fusion methods.

## II. RELATED WORKS

## A. Multi-Source Remote Sensing Image Classification

Multi-source remote sensing classification methods can be broadly grouped according to the stage at which heterogeneous information is integrated. Decision-level fusion trains separate classifiers for different sources and combines their outputs by voting, probability aggregation, or graph-based reasoning. This strategy avoids directly reconciling heterogeneous feature dimensions and can remain effective when one modality is missing or severely degraded. However, because the sources interact only after independent inference, the learned representations cannot use complementary information to resolve ambiguous samples during feature extraction. Feature-level fusion has therefore become the dominant paradigm. It usually employs modality-specific encoders followed by summation, concatenation, attention, or learned interaction modules, enabling spectral, spatial, scattering, and elevation cues to be optimized jointly [21], [22], [29].

CNN-based methods established the basic dual-stream architecture for HSI and LiDAR/SAR joint classification. Coupled CNNs exchange information between modality branches, while spatial-spectral enhancement and recurrent attention models strengthen local feature interaction at multiple scales [22], [26], [30]. FusAtNet introduces dual attention to emphasize informative spectral-spatial responses [23], and asymmetric or collaborative fusion networks model unequal source characteristics instead of treating the modalities identically [31], [32]. Depthwise feature interaction, fractional Gabor filtering, nearest-neighbor contrastive learning, and cross-scale attention further improve multiscale and cross-source discrimination [33]–[36]. These methods provide effective local representation and relatively stable optimization. Nevertheless, standard convolutions operate within limited receptive fields. Enlarging the network depth can propagate information over wider regions, but indirect long-range modeling may weaken boundary details and increase redundant computation.

Transformer and hybrid architectures address this limitation by modeling global token relationships. Multimodal Transformer networks and ExViT establish long-range dependencies through self-attention and cross-modal attention [24], [25], [37], whereas hierarchical CNN–Transformer models combine convolutional locality with global interaction [38], [39]. Deep symmetric fusion, local information interaction, multiscale mixed CNN–Transformer modeling, and head selection further tailor global representation learning to multimodal remote sensing data [40]–[43]. Quaternion, high-order, and progressive cross-fusion mechanisms further enrich the interaction between heterogeneous representations [27], [44], [45]. These architectures are effective for modeling nonlocal context, but dense attention may introduce considerable computation and can establish correlations between semantically unrelated positions. More importantly, global interaction alone does not ensure that different branches respond to the same object structures in a consistent manner.

Recent studies have therefore investigated shared-andspecific representations, masked pretraining, mutual guidance, and iterative cross-source refinement. S2FL learns shared and modality-specific subspace projections [18]. SS-MAE uses spatial-spectral masked modeling to learn multi-source representations [15]. Mutual-guidance or spectral-correlation networks progressively refine one source with information from another [2], [46]. Style separation, frequency-domain fusion, metric learning, and multiscale state-space modeling address cross-source discrepancies from complementary perspectives [47]–[50]. These approaches demonstrate the importance of cross-source consistency.

## B. Context Modeling and Adaptive Multi-Source Fusion

Context modeling determines whether a classifier can distinguish objects using both local appearance and surrounding structure. Conventional small-kernel CNNs are effective at preserving local details, but their effective receptive fields grow slowly with network depth. Large-kernel CNNs broaden spatial support, while selective large-kernel, dynamic, and cross-guided filters adapt contextual responses to input content or another modality [46], [51]–[54]. Cross-modal interaction and selected parameter sharing also promote source consistency [22], [32]. These directions leave a joint challenge: integrating long-range dependencies with a spatial prior shared across registered modalities and network depths, while retaining input-sensitive local filtering. SPCB addresses this by embedding the shared prior in feature extraction and pairing it with filtering that adapts to each input. The shared prior captures recurring scene geometry across modalities and network depths; adaptive filtering models local patterns that vary by source and input.

The need for shared contextual modeling follows from the spatial correspondence of registered observations. Although HSI, SAR, and LiDAR describe different physical attributes, roads, buildings, vegetation parcels, and object boundaries occupy approximately the same spatial locations. Their low-level intensity patterns differ, but their neighborhood organization and larger geometric structures are semantically related.

Fusion strategy is equally important because complementary information is not uniformly reliable. Addition and concatenation assume fixed source roles and rely on subsequent convolutions to suppress irrelevant responses. Channel attention and cross-attention learn data-dependent interactions [23], [27], [32], while intensity constraints, mutual guidance, and global weighting further improve source selection [28], [46]. GMU gates modality contributions at the representation level [55]. Recent remote-sensing methods further use cross-modal attention and semantic guidance, mutual guidance, or graphbased interaction with feature-dependent fusion [32], [46], [56]. These mechanisms support adaptive interaction, but spatial saliency within one representation does not itself compare the reliability of competing sources, and feature-wise modality gates do not explicitly allocate source contributions by location. Because shared scene geometry does not make modality evidence equally discriminative in every region, fusion must retain this spatial variation. RAMA addresses this requirement by deriving spatially varying modality weights from joint HSI-auxiliary features after shared-pattern extraction, so that source contributions can adapt to local content. This regional variation is reflected in the complementary cues provided by each source: LiDAR can distinguish spectrally similar objects with different elevations, HSI can discriminate materials in regions with similar geometry, and SAR can provide structural cues when optical observations are unreliable.

Accordingly, robust multi-source fusion requires two complementary capabilities: cross-source feature extractors should learn compatible contextual patterns, and the fusion module should assign source contributions according to regional content.

## III. METHODOLOGY

The overall framework of the proposed Shared-Pattern Collaborative Network (SPCNet) is illustrated in Fig. 2. SPCNet contains two parallel modality-specific branches: the HSI branch uses PCA followed by a 3D convolutional stem, whereas the SAR/LiDAR branch uses a 2D convolutional stem. Their features are subsequently refined by stacked SPCBs and integrated by RAMA.

![](images/20369aefe5a579217a6d6630d635852afe947b46f031fdb1258c4e8ab8a01ddd.jpg)  
Fig. 3. Details of the Shared-Pattern Collaborative Block (SPCB).

Let $\mathbf { X } _ { h } \in \mathbb { R } ^ { \tilde { H } \times W \times B }$ and $\mathbf { X } _ { x } \in \mathbb { R } ^ { H \times W \times C _ { x } }$ denote the spatially aligned HSI and auxiliary-source patches, respectively, where B is the number of original hyperspectral bands and $C _ { x }$ is the number of SAR/LiDAR channels. PCA retains the leading $N _ { \mathrm { P C } }$ components of the HSI before 3D convolution extracts joint spectral-spatial features, whereas the 2D auxiliary branch extracts structural, scattering, or elevation features. The two stems retain modality-specific low-level statistics while projecting both sources into compatible feature spaces.

The encoded features are successively processed by stacked Shared-Pattern Collaborative Blocks (SPCBs) and the Region-Adaptive Modality Allocator (RAMA). The internal structure of SPCB is shown in Fig. 3. SPCB learns common spatial patterns while preserving source-dependent features, and RAMA produces the fused representation $\mathbf { F } _ { f u s }$ for pixelwise classification. The two modules and their end-to-end optimization are described below.

## A. Shared-Pattern Collaborative Block (SPCB)

Given an input feature $\mathbf { X } \in \mathbb { R } ^ { H \times W \times C }$ , SPCB first applies a $3 \times 3$ convolution to obtain feature F. Then, F is split along the channel dimension into $\mathbf { F } ^ { a } \in \mathbb { R } ^ { H \times W \times 1 6 }$ and $\mathbf { F } ^ { i } \in \overline { { \mathbb { R } } } ^ { H \times W \times ( C - 1 6 ) }$ . Here, $\mathbf { F } ^ { a }$ is used as the active branch for shared-pattern learning and dynamic kernel generation, while $\mathbf { F } ^ { i }$ is preserved as an identity branch and later concatenated with the enhanced feature. Because these channels bypass the shared and dynamic filters before rejoining the output projection, they carry shallow responses into later mixing. Crossdepth reuse therefore acts on progressively refined features that retain earlier cues, while the dynamic branch adapts local filtering to each input.

Restricting the active branch to 16 channels keeps the largekernel and dynamic operations lightweight. The remaining $C -$ 16 channels form a direct information path, reducing the risk that repeated adaptive filtering removes fine source-specific details. The channel split thus provides a practical balance between contextual enhancement and feature preservation.

Shared Large-Kernel Pattern. The upper branch uses a shared large convolution kernel $K _ { s } \in \mathbb { R } ^ { P \times \bar { P } \times 1 6 \times 1 6 }$ to model broad spatial context, where $P$ denotes the side length of the input image patch. Since the same kernel $K _ { s }$ is applied to both HSI and SAR/LiDAR features, the two branches are encouraged to extract spatial structures under a unified convolutional pattern. Through kernel sharing, SPCB establishes an implicit interaction mechanism between the two modalities, allowing the network to align and compare their spatial responses in a common feature space. The shared-pattern response is computed as:

$$
\mathbf { F } _ { s } ^ { a } = K _ { s } * \mathbf { F } ^ { a } ,\tag{1}
$$

where ∗ denotes convolution, and $\mathbf { F } _ { s } ^ { a } \in \mathbb { R } ^ { H \times W \times 1 6 }$ . The large kernel helps the network capture long-range context and shared spatial structures. With stride one, the operator acts on the patch grid. Reflection padding of $( P - 1 ) / 2$ pixels per side preserves the output size, while the support centered on the target pixel spans the unpadded input patch. The kernel side length equals the patch side length P.

Compared with repeatedly stacking small convolutions to approximate a large receptive field, the $P \ \times \ P$ operator directly aggregates the spatial context of the input patch within one collaborative unit. Sharing its parameters constrains the two source branches to analyze that neighborhood with the same spatial template. Reusing the operator at different depths further maintains a consistent structural prior as the features evolve from low-level responses to more semantic representations.

The shared operator does not require the two modalities to produce identical feature values. Instead, it constrains the spatial relation used to process them, while the identity and dynamic branches retain modality-dependent evidence and local appearance variation. This selective sharing is important for aligned multi-source observations, whose geometry is common but whose radiometric responses remain heterogeneous.

Dynamic Kernel Generation. SPCB introduces a dynamic kernel branch to generate an adaptive convolution kernel according to the input feature. First, global average pooling is applied to $\mathbf { F } ^ { a }$ , and MLP is used to predict the dynamic kernel as:

$$
K _ { d } = \mathrm { M L P } ( \operatorname { G A P } ( \mathbf { F } ^ { a } ) ) ,\tag{2}
$$

where the generated lightweight dynamic kernel is $K _ { d } \in$ $\mathbb { R } ^ { 3 \times 3 \times 1 \times 1 6 }$ . For each channel of $\mathbf { F } ^ { a }$ , a corresponding $3 \times 3$ dynamic filter is used as:

$$
{ \bf F } _ { d } ^ { a } = \mathrm { D y D C o n v } ( { \bf F } ^ { a } ; { \cal K } _ { d } ) ,\tag{3}
$$

where DyDConv denotes dynamic depth-wise convolution. The dynamic depth-wise convolution is more efficient while still providing input-adaptive feature modeling capability.

The global descriptor produced by GAP summarizes the current sample and allows the MLP to generate a different group of channel-wise filters for different inputs. Because each $3 \times 3$ filter operates depth-wise, the dynamic branch avoids the parameter and computation cost of a dense input-dependent convolution. It therefore complements the fixed shared large kernel with efficient local adaptation rather than duplicating its long-range modeling role.

![](images/80e778bae5d0b9840abc13fa9b9c368b9661bb70b1df2f4d78c7f02de7bd726e.jpg)  
Fig. 4. Details of the Region-Adaptive Modality Allocator (RAMA).

Collaborative Fusion. The outputs of the shared largekernel branch and the dynamic kernel branch are then combined by element-wise addition, and then concatenated with $\mathbf { F } ^ { i }$ as:

$$
\mathbf { F } ^ { c } = \operatorname { C o n c a t } ( \mathbf { F } _ { s } ^ { a } + \mathbf { F } _ { d } ^ { a } , \ \mathbf { F } ^ { i } ) .\tag{4}
$$

Element-wise addition first combines the common-context response and adaptive local response in the same 16-channel space. Concatenation with $\mathbf { F } ^ { i }$ then restores the preserved identity information. Finally, a $1 \times 1$ convolution mixes the enhanced and identity channels, and a $3 \times 3$ convolution refines their local spatial consistency to generate the output feature:

$$
\begin{array} { r } { \mathbf { Y } = \operatorname { C o n v } _ { 3 \times 3 } \left( \operatorname { C o n v } _ { 1 \times 1 } ( \mathbf { F } ^ { c } ) \right) . } \end{array}\tag{5}
$$

This two-step projection enables information exchange across channels before local spatial refinement and prepares a compatible feature representation for the next SPCB or the subsequent fusion stage.

## B. Region-Adaptive Modality Allocator (RAMA)

The Region-Adaptive Modality Allocator (RAMA), illustrated in Fig. 4, is designed to adaptively allocate the contributions of multi-source features. RAMA learns spatially varying modality weights and performs region-aware fusion. Instead of assigning a fixed global importance to each modality, RAMA dynamically determines which modality should contribute more at each spatial location.

For a mini-batch of M co-registered patch pairs, the input features $\mathbf { F } _ { h } , \mathbf { F } _ { x } \in \mathbb { R } ^ { M \times C \times H \times W }$ are independently processed by $3 { \times } 3$ convolutions to obtain $\hat { { \mathbf { F } } } _ { h }$ and $\hat { \mathbf { F } } _ { x }$ . Their concatenation has shape $M \times 2 C \times H \times W$ , and a $1 \times 1$ projection produces $\mathbf { F } _ { r }$ of the same shape.

$$
\mathbf { F } _ { c a t } = \mathrm { C o n c a t } ( \hat { \mathbf { F } } _ { h } , \hat { \mathbf { F } } _ { x } ) ,\tag{6}
$$

$$
\mathbf F _ { r } = \operatorname { R e s h a p e } ( \operatorname { C o n v } _ { 1 \times 1 } ( \mathbf F _ { c a t } ) ) .\tag{7}
$$

Flattening the spatial axes gives $M \times 2 C \times N _ { s }$ , where $N _ { s } =$ HW. An MLP produces two logits per feature pixel, with grid shape $M \times 2 \times N _ { s }$ . Softmax over modalities and reshaping yield pixelwise maps $[ \mathbf { A } _ { h } , \mathbf { A } _ { x } ] \in \mathbb { R } ^ { M \times 2 \times H \times W }$ , each of shape $M \times 1 \times H \times W ;$

$$
\begin{array} { r l } & { [ \mathbf { A } _ { h } ( i , j ) , \mathbf { A } _ { x } ( i , j ) ] = \operatorname { S o f t m a x } _ { m } \big ( \mathrm { M L P } ( \mathbf { F } _ { r } ( i , j ) ) \big ) , } \\ & { \mathbf { A } _ { h } ( i , j ) + \mathbf { A } _ { x } ( i , j ) = 1 . } \end{array}\tag{8}
$$

Each map is broadcast over $C$ channels to weight its refined feature. The maps allocate contributions at each patch feature

pixel, while input-independent global coefficients α and $\beta$ calibrate their overall contributions.

$$
\hat { \mathbf { F } } _ { h } ^ { \prime } = \mathbf { A } _ { h } \odot \hat { \mathbf { F } } _ { h } , \quad \hat { \mathbf { F } } _ { x } ^ { \prime } = \mathbf { A } _ { x } \odot \hat { \mathbf { F } } _ { x } ,\tag{9}
$$

$$
\mathbf { F } _ { f u s } = \alpha \hat { \mathbf { F } } _ { h } ^ { \prime } + \beta \hat { \mathbf { F } } _ { x } ^ { \prime } , \qquad \alpha + \beta = 1 .\tag{10}
$$

## C. Optimization Objective

For an input sample with K land-cover categories, the classification head converts $\mathbf { F } _ { f u s }$ into the predicted classprobability vector $\mathbf { p } _ { n }$ . SPCNet is trained end-to-end using the standard cross-entropy objective:

$$
\mathcal { L } _ { \mathrm { c e } } = - \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \sum _ { k = 1 } ^ { K } y _ { n , k } \log p _ { n , k } ,\tag{11}
$$

where N is the number of labeled training samples in a mini-batch and $y _ { n , k }$ is the one-hot ground-truth indicator. The classification loss jointly supervises the modality-specific encoders, SPCBs, dynamic kernel generator, RAMA, and final prediction head, so the shared patterns and regional allocations are learned specifically for discriminative land-cover recog nition. During back-propagation, the shared spatial operator, dynamic local filters, and allocation maps are optimized jointly rather than through separate auxiliary objectives, ensuring that each mechanism is directly guided by the final classification error.

## IV. EXPERIMENTAL RESULTS AND ANALYSIS

## A. Experimental Setup and Dataset Description

To evaluate the effectiveness and generality of SPCNet, experiments were conducted on three benchmark multi-source remote sensing datasets: Houston2013, Augsburg, and Houston2018. All methods were implemented in PyTorch and trained on an NVIDIA H100 GPU with 80 GB of memory. The same training and testing partitions were used for all compared methods to ensure a fair evaluation. The Houston datasets pair HSI with LiDAR-derived structural information, whereas Augsburg combines HSI with polarimetric SAR (PolSAR) observations. These complementary settings allow the proposed network to be evaluated under both spectral–elevation and spectral–scattering fusion conditions.

Houston2013 Dataset. The Houston2013 dataset [17] was released for the 2013 IEEE GRSS Data Fusion Contest and covers the University of Houston campus and its surrounding urban area. The airborne HSI has a spatial size of 349 × 1905 pixels and 144 spectral bands spanning 380–1050 nm, with a ground sample distance of 2.5 m. The co-registered LiDAR data provide complementary elevation cues for buildings, vegetation, and transportation infrastructure. The scene contains 15 land-cover classes, including several visually similar paved and built-up categories as well as narrow road and railway structures. It therefore provides a representative urban setting for assessing whether cross-modal contextual modeling can preserve object continuity while separating classes with similar spectral responses.

TABLE I  
NUMBER OF TRAINING AND TEST SAMPLES FOR THE HOUSTON2013 DATASET.
<table><tr><td>No.</td><td>Name</td><td>Color</td><td>Train</td><td>Test</td></tr><tr><td></td><td>Healthy grass</td><td></td><td>198</td><td>1053</td></tr><tr><td></td><td>Stressed grass</td><td></td><td>190</td><td>1064</td></tr><tr><td></td><td>Synthetic grass</td><td></td><td>192</td><td>505</td></tr><tr><td></td><td>Trees</td><td></td><td>188</td><td>1056</td></tr><tr><td>123456</td><td>Soil</td><td></td><td>186</td><td>1056</td></tr><tr><td></td><td>Water</td><td></td><td>182</td><td>143</td></tr><tr><td></td><td>Residential</td><td></td><td>196</td><td>1072</td></tr><tr><td>78</td><td>Commercial</td><td></td><td>191</td><td>1053</td></tr><tr><td>9</td><td>Road</td><td></td><td>193</td><td>1059</td></tr><tr><td>10</td><td>Highway</td><td></td><td>191</td><td>1036</td></tr><tr><td>11</td><td>Railway</td><td></td><td>181</td><td>1054</td></tr><tr><td>12</td><td>Parking lot 1</td><td></td><td>192</td><td>1041</td></tr><tr><td>13</td><td>Parking lot 2</td><td></td><td>184</td><td>285</td></tr><tr><td>14</td><td>Tennis court</td><td></td><td>181</td><td>247</td></tr><tr><td>15</td><td>Running track</td><td></td><td>187</td><td>473</td></tr><tr><td colspan="3">Total</td><td>2832</td><td>12197</td></tr></table>

TABLE II

NUMBER OF TRAINING AND TEST SAMPLES FOR THE AUGSBURG DATASET.
<table><tr><td> $\mathrm { N o . }$ </td><td>Name</td><td>Color</td><td>Train</td><td>Test</td></tr><tr><td></td><td>Forest</td><td></td><td>146</td><td>13361</td></tr><tr><td>１２３４５６７</td><td>Residential area</td><td></td><td>264</td><td>30065</td></tr><tr><td></td><td>Industrial area</td><td></td><td>21</td><td>3830</td></tr><tr><td></td><td>Low plants</td><td></td><td>248</td><td>26609</td></tr><tr><td></td><td>Allotment</td><td></td><td>52</td><td>523</td></tr><tr><td></td><td>Commercial area</td><td></td><td>7</td><td>1638</td></tr><tr><td></td><td>Water</td><td></td><td>23</td><td>1507</td></tr><tr><td colspan="2">Total</td><td colspan="2"></td><td>77533</td></tr></table>

Augsburg Dataset. The Augsburg dataset [18] covers the vicinity of Augsburg, Germany, and contains HySpex HSI together with Sentinel-1 PolSAR data. Both modalities are provided at a 30-m ground sample distance with a spatial size of 332 × 485 pixels. The HSI contains 180 bands from 0.4 to 2.5 µm, while the SAR modality includes VV and VH intensity together with the real and imaginary components of the off-diagonal PolSAR covariance term. Its ground-truth labels are derived from OpenStreetMap and describe seven urban and peri-urban classes. The coarse spatial resolution, heterogeneous land use, and highly uneven class distribution make this dataset particularly useful for testing whether a fusion model can exploit scattering information without amplifying source-specific noise or minority-class ambiguity.

Houston2018 Dataset. The Houston2018 dataset [18] was introduced in the 2018 IEEE GRSS Data Fusion Contest and covers the University of Houston campus together with neighboring urban areas. The original release provides HSI, LiDAR, and optical observations; following the multi-source HSI–LiDAR setting, only the HSI and LiDAR modalities are used in this work. The HSI contains 48 bands over the 380– 1050 nm range at 1-m spatial resolution, while the LiDAR data have a finer native spatial resolution and are spatially aligned with the HSI for classification. This larger and more diverse scene includes 20 land-cover categories spanning vegetation, buildings, roads, parking areas, railways, and small objects such as cars and trains. Its extensive and strongly imbalanced labeled set provides a demanding test of generalization across complex urban spatial layouts.

(a)

![](images/3f8521fd33964cbcf80c79da3b8d6355efef28cb47b861448d79caa6991a69c2.jpg)

![](images/9fd4bb4d7aec89e8cc128fa6680f33c02242ef0d9267e59050a1c2c90a53f24f.jpg)  
(b)

![](images/99f7e96156d8190fff0389cdcf1d013ed5bdfa14cb1648fb1063e1c7f91c0692.jpg)  
(c)  
Fig. 5. Visualization of the three multi-source datasets. (a) Houston2013 dataset. (b) Augsburg dataset. (c) Houston2018 dataset.

TABLE III  
NUMBER OF TRAINING AND TEST SAMPLES FOR THE HOUSTON2018 DATASET.
<table><tr><td>No.</td><td>Name</td><td>Color</td><td>Train</td><td>Test</td></tr><tr><td>1</td><td>Healthy grass</td><td></td><td>1000</td><td>38196</td></tr><tr><td></td><td>Stressed grass</td><td></td><td>1000</td><td>129008</td></tr><tr><td></td><td>Artificial turf</td><td></td><td>1000</td><td>1736</td></tr><tr><td></td><td>Evergreen trees</td><td></td><td>1000</td><td>53322</td></tr><tr><td></td><td>Deciduous trees</td><td></td><td>1000</td><td>19172</td></tr><tr><td>23456</td><td>Bare earth</td><td></td><td>1000</td><td>17064</td></tr><tr><td>789</td><td>Water</td><td></td><td>500</td><td>564</td></tr><tr><td></td><td>Residential buildings</td><td></td><td>1000</td><td>157995</td></tr><tr><td></td><td>Non-residential buildings</td><td></td><td>1000</td><td>893769</td></tr><tr><td>10</td><td>Roads</td><td></td><td>1000</td><td>182283</td></tr><tr><td>11</td><td>Sidewalks</td><td></td><td>1000</td><td>135035</td></tr><tr><td>12</td><td>Crosswalks</td><td></td><td>1000</td><td>5059</td></tr><tr><td>13</td><td>Major thoroughfares</td><td></td><td>1000</td><td>184438</td></tr><tr><td>14</td><td>Highways</td><td></td><td>1000</td><td>38438</td></tr><tr><td>15</td><td>Railways</td><td></td><td>1000</td><td>26748</td></tr><tr><td>16</td><td>Paved parking lots</td><td></td><td>1000</td><td>44932</td></tr><tr><td>17</td><td>Unpaved parking lots</td><td></td><td>250</td><td>337</td></tr><tr><td>18</td><td>Cars</td><td></td><td>1000</td><td>25289</td></tr><tr><td>19</td><td>Trains</td><td></td><td>1000</td><td>20479</td></tr><tr><td>20</td><td>Stadium seats</td><td></td><td>1000</td><td>26296</td></tr><tr><td colspan="2">Total</td><td></td><td>18750</td><td>2000160</td></tr></table>

The class-wise training and testing samples of Houston2013, Augsburg, and Houston2018 are listed in Tables I, II, and III, respectively. Fig. 5 presents the false-color HSI, the corresponding SAR/LiDAR data, and the ground-truth maps of the three datasets.

![](images/dae1b518b4814da01ff3669d9bdcf59b0c3c93a2adde0dbbcd1c1e59194a33c1.jpg)

the overall accuracy (OA), average accuracy (AA), and Kappa coefficient are employed for evaluation.

After normalization, HSI is projected by PCA and each labeled target pixel defines a co-registered centered patch pair supervised by its center label. Reflection padding of $( P { - } 1 ) / 2$ pixels per side preserves output size. Both modalities receive the same horizontal or vertical flip with probability 0.5. All methods use these input settings and the same train/test partitions and optimization settings (Adam, initial learning rate $2 \times 1 0 ^ { - 4 }$ , weight decay 0, batch size 128, 35 epochs, dropout 0.05, and seed 6). We compare our SPCNet with ten state-of-the-art methods, including FusAtNet [23], S2ENet [26], AsyFFNet [31], ExViT [24], HCT [38], MACN [39], MICF-Net [32], GCCQTNet [44], CHNet [45], and MGMNet [46]. To quantitatively assess the classification performance,

Fig. 6. Depth of SPCB Layers.  
![](images/c28d84bdb948b06b3167c3c8ccf781388010bf296520fedbd3af4c56b1aa7da9.jpg)  
Fig. 7. Number of Principal Components for HSI.

## B. Analysis of Important Parameters

Depth of SPCB Layers. Fig. 6 shows the OA variation as the number of SPCB layers increases from 2 to 10. The best results are obtained with eight layers on Houston2013 (93.53%) and Houston2018 (93.44%), whereas Augsburg reaches its highest OA of 92.34% with six layers. Increasing the depth initially improves the representation of shared spatial patterns and cross-source interactions. However, further stacking does not provide a consistent benefit, because redundant feature interactions can increase optimization difficulty and computation without contributing discriminative context. The selected depths are 8, 6, and 8 for Houston2013, Augsburg, and Houston2018, respectively.

![](images/d5da157ecb012ee7664a16941a1d7663caf23f9bdae6da66e2c58486dd170868.jpg)  
Fig. 8. Size of the Input Image Patches.

Number of Principal Components for HSI. Fig. 7 evaluates the number of retained principal components. On Houston2013, OA increases sharply to 93.53% when 30 components are retained, while 40 and 50 components introduce redundant spectral information and reduce performance. Augsburg and Houston2018 achieve their best OAs of 92.35% and 93.65%, respectively, with 20 components. The different optima reflect the distinct spectral redundancy and scene composition of the three datasets. Accordingly, 30 principal components are retained for Houston2013, whereas 20 components are retained for Augsburg and Houston2018.

Size of the Input Image Patches. Fig. 8 studies the input patch size. Houston2013 reaches 93.53% OA with an 11-pixel patch, Augsburg reaches 92.68% with a 9-pixel patch, and Houston2018 reaches 93.77% with a 13-pixel patch. Small patches provide insufficient contextual evidence for neighboring land-cover categories, whereas excessively large patches can include unrelated pixels and weaken boundary discrimination. The different optima therefore reflect differences in object scale and spatial heterogeneity among the three scenes.

## C. Performance Comparison

We compare SPCNet with ten representative state-of-the-art multi-source classification methods, following the comparison protocol commonly adopted in recent multi-source remote sensing studies. These baselines cover complementary design routes from attention-based CNNs to hybrid Transformers and high-order adaptive fusion. FusAtNet [23] applies selfattention to HSI spectral features and uses attention cues derived from LiDAR to enhance spatial representations. S2ENet [26] introduces a spatial-spectral cross-modal enhancement module to exchange information between HSI and LiDAR branches. AsyFFNet [31] adopts an asymmetric fusion strategy to account for the unequal information contributions of HSI and SAR. ExViT [24] extends Vision Transformer with separable convolutions and a cross-modality attention module, while HCT [38] combines hierarchical CNNs with a crosstoken attention encoder for joint classification. MACN [39] mixes self-attention and convolution so that local structures and global dependencies can be modeled in a unified layer. MICF-Net [32] employs dual-branch cross-source attention to extract global context and spatial relationship similarities, together with adaptive modulation to balance the optimization of different sources. GCCQTNet [44] uses an independent squeeze-expansion-like fusion structure and a cross-memory quaternion Transformer to reduce multimodal heterogeneity and model complex intra- and inter-modality relationships. CHNet [45] progressively removes spectral-spatial redundancy and combines dynamic feature selection with factorized highorder learning to improve feature discriminability. MGMNet [46] embeds mutual guidance in a two-stream network, where context-aware filters generated from one source are used to refine the features of the other. These methods provide a representative coverage of local convolutional modeling, cross-attention, global token interaction, high-order fusion, and mutual-guidance strategies. Tables IV–VI report the classwise and overall performance, while Figs. 9–11 show the corresponding classification maps. In addition to OA, AA, and Kappa, we examine confusing categories, boundary regions, and spatially fragmented objects to evaluate both numerical accuracy and spatial consistency.

Results on Houston2013 Dataset. The quantitative results in Table IV show that SPCNet achieves the best OA, AA, and Kappa values of 93.53%, 94.85%, and 92.98%, respectively. Compared with the strongest competing OA of 90.98% obtained by MGMNet, SPCNet provides an absolute gain of 2.55 percentage points. The improvement is particularly evident for difficult transportation-related classes: the accuracy for Highway reaches 94.79%, exceeding the second-best result by 9.65 percentage points, while Railway and Parking lot 2 reach 99.53% and 96.49%, respectively. These results indicate that the shared large-kernel patterns capture extended spatial structures and that RAMA effectively exploits the complementary elevation information from LiDAR. As shown in Fig. 9, several competing methods produce fragmented predictions and blurred boundaries in the enlarged region. SPCNet generates a map that is more consistent with the ground truth, with clearer object contours and fewer isolated errors. The agreement between the quantitative and visual results demonstrates strong discriminative capability in complex urban scenes.

Results on Augsburg Dataset. Table V shows that SPC-Net obtains the best OA and Kappa values of 92.68% and 89.37%, outperforming the second-best results by 1.50 and 2.14 percentage points, respectively. Its AA of 64.84% remains competitive with the best value of 65.51%. At the category level, SPCNet maintains strong results for Low plants, Forest, and Residential area, although the best score in each individual category is sometimes achieved by a competing method. The 2.32% Commercial-area accuracy is consistent with severe class imbalance (only seven training pixels) and the coarse 30- m resolution of Augsburg’s OpenStreetMap-derived functional labels [18], which may group mixed surfaces whose spectral and scattering responses overlap with Residential or Industrial areas. This sparse supervision and within-class variation depress minority-class recall and AA, while the common classes largely determine OA. The visual results in Fig. 10 further support this observation. SPCNet preserves more continuous forest and residential structures in the enlarged region and suppresses isolated errors that are visible in several competing maps. This demonstrates that collaborative HSI-SAR representation learning improves both regional consistency and robustness to scattering variations.

![](images/849bc750ded033c011fec930b9d55bb0ced1e3a9456dd2824a2c188b03db3d3c.jpg)  
Fig. 9. Visualized classification results of different methods for the Houston2013 dataset. (a) Ground truth. (b) FusAtNet. (c) S2ENet. (d) AsyFFNet. (e) ExViT. (f) HCT. (g) MACN. (h) MICF-Net. (i) GCCQTNet. (j) CHNet. (k) MGMNet. (l) SPCNet.

TABLE IV  
CLASSIFICATION PERFORMANCE OF DIFFERENT METHODS ON THE HOUSTON2013 DATASET. THE BOLD ENTRIES DENOTE THE BEST RESULTS.
<table><tr><td>Class</td><td>|FusAtNet</td><td>S2ENet</td><td>AsyFFNet</td><td>ExViT</td><td>HCT</td><td>MACN</td><td></td><td>MICF-Net GCCQTNet</td><td>CHNet MGMNet</td><td></td><td>SPCNet</td></tr><tr><td>Healthy grass</td><td>82.53</td><td>81.48</td><td>82.72</td><td>82.72</td><td>81.10</td><td>82.90</td><td>82.91</td><td>82.05</td><td>83.10</td><td>83.00</td><td>82.91</td></tr><tr><td>Stressed grass</td><td>85.15</td><td>84.68</td><td>83.93</td><td>85.15</td><td>86.09</td><td>100.0</td><td>83.46</td><td>96.33</td><td>85.15</td><td>85.15</td><td>99.72</td></tr><tr><td>Synthetic grass</td><td>98.81</td><td>99.01</td><td>100.0</td><td>99.21</td><td>99.80</td><td>99.80</td><td>99.80</td><td>99.41</td><td>99.60</td><td>99.60</td><td>100.0</td></tr><tr><td>Trees</td><td>92.90</td><td>92.14</td><td>91.95</td><td>91.95</td><td>98.01</td><td>98.11</td><td>97.63</td><td>91.00</td><td>92.61</td><td>87.59</td><td>93.37</td></tr><tr><td>Soil</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>99.52</td><td>99.90</td><td>100.0</td><td>100.0</td></tr><tr><td>Water</td><td>98.60</td><td>100.0</td><td>95.80</td><td>98.60</td><td>95.10</td><td>95.80</td><td>95.80</td><td>95.80</td><td>95.80</td><td>89.51</td><td>100.0</td></tr><tr><td>Residential</td><td>85.45</td><td>89.93</td><td>96.18</td><td>86.66</td><td>83.49</td><td>78.17</td><td>83.49</td><td>90.95</td><td>88.25</td><td>95.62</td><td>85.63</td></tr><tr><td>Commercial</td><td>81.81</td><td>90.18</td><td>81.14</td><td>91.05</td><td>86.43</td><td>80.46</td><td>81.91</td><td>95.67</td><td>81.62</td><td>76.42</td><td>88.41</td></tr><tr><td>Road</td><td>83.76</td><td>92.63</td><td>86.12</td><td>91.12</td><td>86.69</td><td>90.93</td><td>82.15</td><td>87.54</td><td>89.61</td><td>93.96</td><td>82.34</td></tr><tr><td>Highway</td><td>53.67</td><td>64.86</td><td>64.58</td><td>64.86</td><td>79.05</td><td>55.88</td><td>66.22</td><td>65.15</td><td>73.84</td><td>85.14</td><td>94.79</td></tr><tr><td>Railway</td><td>79.65</td><td>97.60</td><td>86.76</td><td>77.06</td><td>95.97</td><td>99.33</td><td>95.59</td><td>86.85</td><td>97.41</td><td>94.91</td><td>99.53</td></tr><tr><td>Parking lot 1</td><td>91.26</td><td>88.09</td><td>90.59</td><td>88.76</td><td>98.46</td><td>99.71</td><td>94.62</td><td>99.14</td><td>99.90</td><td>98.37</td><td>99.62</td></tr><tr><td>Parking lot 2</td><td>81.05</td><td>91.93</td><td>92.28</td><td>87.72</td><td>90.88</td><td>88.77</td><td>81.40</td><td>82.81</td><td>89.47</td><td>89.12</td><td>96.49</td></tr><tr><td>Tennis court</td><td>100.0</td><td>100.0</td><td>95.14</td><td>97.17</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>99.60</td><td>100.0</td></tr><tr><td>Running track</td><td>98.73</td><td>100.0</td><td>100.0</td><td>98.52</td><td>100.0</td><td>100.0</td><td>99.79</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>OA</td><td>85.32</td><td>89.56</td><td>87.95</td><td>87.42</td><td>90.66</td><td>89.81</td><td>88.09</td><td>90.40</td><td>90.30</td><td>90.98</td><td>93.53</td></tr><tr><td>AA</td><td>87.56</td><td>91.50</td><td>89.81</td><td>89.37</td><td>92.07</td><td>91.33</td><td>89.65</td><td>91.48</td><td>91.75</td><td>91.87</td><td>94.85</td></tr><tr><td>Kappa</td><td>84.12</td><td>88.69</td><td>86.96</td><td>86.38</td><td>89.86</td><td>88.93</td><td>87.12</td><td>89.59</td><td>89.47</td><td>90.20</td><td>92.98</td></tr></table>

TABLE V  
CLASSIFICATION PERFORMANCE OF DIFFERENT METHODS ON THE AUGSBURG DATASET. THE BOLD ENTRIES DENOTE THE BEST RESULTS.
<table><tr><td>Class</td><td>|FusAtNet S2ENet AsyFFNet ExViT</td><td></td><td></td><td></td><td>HCT MACN</td><td></td><td>MICF-Net GCCQTNet</td><td></td><td>CHNet MGMNet</td><td></td><td>SPCNet</td></tr><tr><td>Forest</td><td>93.78</td><td>98.10</td><td>96.01</td><td>90.04</td><td>96.92</td><td>97.95</td><td>95.98</td><td>95.26</td><td>96.24</td><td>96.28</td><td>96.84</td></tr><tr><td>Residential area</td><td>97.58</td><td>99.08</td><td>97.96</td><td>95.44</td><td>98.96</td><td>98.68</td><td>98.46</td><td>96.80</td><td>99.22</td><td>97.76</td><td>98.71</td></tr><tr><td>Industrial area</td><td>26.48</td><td>12.19</td><td>53.29</td><td>34.58</td><td>33.05</td><td>45.30</td><td>58.38</td><td>68.20</td><td>44.23</td><td>49.61</td><td>64.10</td></tr><tr><td>Low plants</td><td>97.67</td><td>91.78</td><td>91.74</td><td>90.68</td><td>92.36</td><td>94.24</td><td>91.75</td><td>91.52</td><td>93.83</td><td>92.32</td><td>97.08</td></tr><tr><td>Allotment</td><td>52.77</td><td>45.12</td><td>65.39</td><td>51.82</td><td>54.11</td><td>51.24</td><td>40.73</td><td>41.68</td><td>54.68</td><td>56.60</td><td>52.39</td></tr><tr><td>Commercial area</td><td>24.66</td><td>1.22</td><td>4.58</td><td>28.63</td><td>6.62</td><td>6.74</td><td>23.53</td><td>13.97</td><td>19.85</td><td>5.15</td><td>2.32</td></tr><tr><td>Water</td><td>47.51</td><td>24.09</td><td>49.64</td><td>17.65</td><td>47.40</td><td>49.27</td><td>44.81</td><td>46.80</td><td>46.74</td><td>49.93</td><td>42.47</td></tr><tr><td>OA</td><td>90.62</td><td>88.22</td><td>90.15</td><td>86.65</td><td>89.83</td><td>91.18</td><td>90.74</td><td>90.22</td><td>91.14</td><td>90.10</td><td>92.68</td></tr><tr><td>AA</td><td>62.92</td><td>53.08</td><td>65.51</td><td>58.40</td><td>61.34</td><td>63.35</td><td>64.81</td><td>64.89</td><td>64.97</td><td>63.95</td><td>64.84</td></tr><tr><td>Kappa</td><td>86.33</td><td>86.47</td><td>85.74</td><td>80.79</td><td>85.28</td><td>87.23</td><td>86.65</td><td>86.00</td><td>87.10</td><td>85.75</td><td>89.37</td></tr></table>

![](images/7b4e4b7ba71e024c4ef6e43dbb8c7909b20a90709106b3d16e66c6c2171f4959.jpg)  
Fig. 10. Visualized classification results of different methods for the Augsburg dataset. (a) Ground truth. (b) FusAtNet. (c) S2ENet. (d) AsyFFNet. (e) ExViT. (f) HCT. (g) MACN. (h) MICF-Net. (i) GCCQTNet. (j) CHNet. (k) MGMNet. (l) SPCNet.

Results on Houston2018 Dataset. As reported in Table VI, SPCNet achieves the best OA, AA, and Kappa values of 93.77%, 96.19%, and 91.93%, respectively. Relative to the second-best results, the gains are 1.14 percentage points in OA, 0.26 percentage points in AA, and 1.45 percentage points in Kappa. SPCNet is especially effective for Stressed grass, Sidewalks, and Major thoroughfares, obtaining accuracies of 94.92%, 85.47%, and 90.96%, respectively. The consistently high scores across vegetation, building, and transportation categories show that the proposed network balances spectral discrimination with structural modeling. In Fig. 11, SPCNet produces cleaner class regions and more complete linear structures than the competing methods. Roads, railways, and the boundaries between vegetation and buildings are better retained in the enlarged region, while scattered prediction noise is reduced. This consistency between visual quality and quantitative performance confirms the generalization capability of SPCNet on a larger and more diverse urban scene.

## D. Ablation Study

Effectiveness of Essential Components. To evaluate the contributions of SPCB and RAMA, Table VII compares the baseline, the two single-component variants, and the complete SPCNet using OA on all three datasets. The baseline obtains 88.56%, 89.75%, and 89.66% on Houston2013, Augsburg, and Houston2018, respectively. The variant without SPCB, corresponding to the RAMA-only configuration, improves these results by 1.77, 1.67, and 2.09 percentage points. The variant without RAMA, corresponding to the SPCB-only configuration, improves OA by 2.36, 0.44, and 1.47 percentage points on Houston2013, Augsburg, and Houston2018, respectively. Thus, both modules improve the baseline, although their individual gains vary across datasets. When SPCB and RAMA are jointly incorporated, OA reaches 93.53%, 92.68%, and 93.77%, which are 4.97, 2.93, and 4.11 percentage points above the baseline. The full model further exceeds the RAMAonly configuration by 3.20, 1.26, and 2.02 percentage points, and the SPCB-only configuration by 2.61, 2.49, and 2.64 percentage points. These results indicate that SPCB and RAMA provide complementary benefits under the same evaluation protocol.

TABLE VI  
CLASSIFICATION PERFORMANCE OF DIFFERENT METHODS ON THE HOUSTON2018 DATASET. THE BOLD ENTRIES DENOTE THE BEST RESULTS.
<table><tr><td>Class</td><td colspan="4">|FusAtNet S2ENet AsyFFNet ExViT</td><td>HCT MACN</td><td colspan="7">MICF-Net GCCQTNet CHNet MGMNet</td></tr><tr><td>Healthy grass</td><td>60.69</td><td>99.15</td><td>96.90</td><td>93.34</td><td>97.38</td><td>89.53</td><td>94.40</td><td>91.71</td><td>91.24</td><td></td><td>92.83</td><td>92.26</td></tr><tr><td>Stressed grass</td><td>93.85</td><td>84.35</td><td>91.82</td><td>93.44</td><td>89.40</td><td>93.50</td><td>89.79</td><td>94.22</td><td></td><td>93.94</td><td>93.51</td><td>94.92</td></tr><tr><td>Artificial turf</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td></td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>Evergreen trees</td><td>99.02</td><td>98.13</td><td>99.29</td><td>95.87</td><td>98.90</td><td>99.10</td><td>99.24</td><td></td><td>98.58</td><td>98.50</td><td>99.63</td><td>99.40</td></tr><tr><td>Deciduous trees</td><td>94.73</td><td>86.39</td><td>99.52</td><td>96.85</td><td>98.89</td><td>98.49</td><td>98.85</td><td></td><td>98.71</td><td>99.17</td><td>99.34</td><td>99.04</td></tr><tr><td>Bare earth</td><td>99.87</td><td>99.33</td><td>99.98</td><td>100.0</td><td>100.0</td><td>99.96</td><td>99.84</td><td></td><td>99.96</td><td>99.98</td><td>99.96</td><td>99.99</td></tr><tr><td>Water</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td></td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>Residential buildings</td><td>95.51</td><td>97.17</td><td>95.16</td><td>97.39</td><td>95.30</td><td>95.31</td><td>94.83</td><td></td><td>96.41</td><td>94.81</td><td>97.22</td><td>97.13</td></tr><tr><td>Non-residential buildings</td><td>93.64</td><td>93.93</td><td>95.02</td><td>93.00</td><td>95.25</td><td>93.86</td><td>95.39</td><td></td><td>94.16</td><td>94.39</td><td>95.64</td><td>96.41</td></tr><tr><td>Roads</td><td>70.94</td><td>70.17</td><td>72.72</td><td>72.18</td><td>72.77</td><td>72.11</td><td>73.33</td><td></td><td>77.79</td><td>82.42</td><td>72.04</td><td>78.13</td></tr><tr><td>Sidewalks</td><td>79.11</td><td>72.11</td><td>77.18</td><td>69.69</td><td>73.27</td><td>80.69</td><td>78.10</td><td></td><td>82.55</td><td>77.99</td><td>84.50</td><td>85.47</td></tr><tr><td>Crosswalks</td><td>86.06</td><td>82.47</td><td>93.14</td><td>88.12</td><td>98.95</td><td>97.69</td><td>97.63</td><td></td><td>97.11</td><td>97.61</td><td>97.98</td><td>94.15</td></tr><tr><td>Major thoroughfares</td><td>76.12</td><td>89.44</td><td>84.67</td><td>86.12</td><td>80.90</td><td>79.05</td><td>77.73</td><td></td><td>88.03</td><td>83.84</td><td>89.64</td><td>90.96</td></tr><tr><td>Highways</td><td>98.89</td><td>98.53</td><td>99.53</td><td>99.50</td><td>98.01</td><td>96.86</td><td>97.04</td><td></td><td>98.43</td><td>96.80</td><td>99.41</td><td>99.18</td></tr><tr><td>Railways</td><td>99.88</td><td>99.21</td><td>99.90</td><td>99.87</td><td>99.74</td><td>99.90</td><td>99.76</td><td></td><td>99.77</td><td>99.57</td><td>99.90</td><td>99.76</td></tr><tr><td>Paved parking lots</td><td>95.88</td><td>95.62</td><td>99.18</td><td>97.02</td><td>97.07</td><td>97.74</td><td>98.72</td><td></td><td>99.35</td><td>97.22</td><td>97.77</td><td>98.07</td></tr><tr><td>Unpaved parking lots</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td></td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>Cars</td><td>94.25</td><td>96.77</td><td>96.64</td><td>98.13</td><td>99.01</td><td>98.74</td><td>98.18</td><td></td><td>97.44</td><td>99.08</td><td>99.27</td><td>98.97</td></tr><tr><td>Trains</td><td>99.84</td><td>100.0</td><td>100.0</td><td>99.98</td><td>99.97</td><td>99.74</td><td>99.88</td><td></td><td>99.89</td><td>99.97</td><td>99.99</td><td>99.98</td></tr><tr><td>Stadium seats</td><td>100.0</td><td>100.0</td><td>99.89</td><td>99.99</td><td>99.98</td><td>99.74</td><td>99.98</td><td></td><td>99.84</td><td>99.95</td><td>100.0</td><td>100.0</td></tr><tr><td>OA</td><td>89.09</td><td>90.05</td><td>91.24</td><td>89.98</td><td>90.56</td><td>90.31</td><td></td><td>90.65</td><td>92.13</td><td>91.75</td><td>92.63</td><td>93.77</td></tr><tr><td>AA</td><td>91.91</td><td>93.14</td><td>95.02</td><td>94.02</td><td>94.74</td><td>94.61</td><td></td><td>94.63</td><td>95.70</td><td>95.32</td><td>95.93</td><td>96.19</td></tr><tr><td>Kappa</td><td>85.96</td><td>87.20</td><td>88.71</td><td>87.14</td><td>87.81</td><td>87.55</td><td>87.94</td><td></td><td>89.86</td><td>89.37</td><td>90.48</td><td>91.93</td></tr></table>

![](images/f406090b4d0118718200278db8d592d6ab2143c3981b62772ae47c546532128b.jpg)

(a) Ground truth  
![](images/a4b1cf728a19f3c4b0e678ec3f5c500adbad127ab272a7ad63ef0b244270d935.jpg)

(c) S2ENet  
![](images/ac3ca0508b0bf9b1d2502b7af8929ffdfeba532e94aebfb56a2d31a1d084bbd3.jpg)

(e) ExViT  
![](images/2b5cab10b5041c2e8ae4a63f29b34fa22924324bcc236e40d530cc14c36bec2f.jpg)

(g) MACN  
![](images/b2b6fdcac2ae2b1759c7330116c237ec4d7f83b02d8ea9b0df35e02c18f3e0f0.jpg)

(i) GCCQTNet  
![](images/ecbf460cf28c85889a853efcd034fc330993f07523a1dfec75a2b2c2ae262c56.jpg)  
(k) MGMNet

![](images/0a2c7480d50b2d298d8064e1cafe624f434f27055bae8ae31a8963b260f98112.jpg)  
Fig. 11. Visualized classification results of different methods for the Houston2018 dataset. (a) Ground truth. (b) FusAtNet. (c) S2ENet. (d) AsyFFNet. (e) ExViT. (f) HCT. (g) MACN. (h) MICF-Net. (i) GCCQTNet. (j) CHNet. (k) MGMNet. (l) SPCNet.

TABLE VII  
ABLATION STUDY OF THE PROPOSED SPCNET IN TERMS OF OA (%).
<table><tr><td>Model Variant</td><td>Houston 2013</td><td>Augsburg</td><td>Houston 2018</td></tr><tr><td>Baseline</td><td>88.56</td><td>89.75</td><td>89.66</td></tr><tr><td>w/o SPCB</td><td>90.33</td><td>91.42</td><td>91.75</td></tr><tr><td>w/o RAMA</td><td>90.92</td><td>90.19</td><td>91.13</td></tr><tr><td>Full SPCNet</td><td>93.53</td><td>92.68</td><td>93.77</td></tr></table>

TABLE VIII  
ABLATION STUDY OF SPCB IN TERMS OF OA (%).
<table><tr><td>Model Variant</td><td>Houston 2013</td><td>Augsburg</td><td>Houston 2018</td></tr><tr><td>Shared-LK only</td><td>92.67</td><td>92.19</td><td>93.24</td></tr><tr><td>DK only</td><td>91.13</td><td>91.62</td><td>92.12</td></tr><tr><td>Static-DW (generator)</td><td>93.21</td><td>91.78</td><td>93.33</td></tr><tr><td>Static-DW (pointwise)</td><td>92.66</td><td>91.90</td><td>93.26</td></tr><tr><td>Full SPCB</td><td>93.53</td><td>92.68</td><td>93.77</td></tr></table>

TABLE IX  
ABLATION STUDY OF RAMA IN TERMS OF OA (%).
<table><tr><td>Fusion Strategy</td><td>Houston 2013</td><td>Augsburg</td><td>Houston 2018</td></tr><tr><td>1 × 1 convolution</td><td>91.19</td><td>90.74</td><td>92.58</td></tr><tr><td>Global allocator</td><td>93.53</td><td>92.26</td><td>93.53</td></tr><tr><td>Regional allocation</td><td>93.29</td><td>92.31</td><td>93.63</td></tr><tr><td>RAMA</td><td>93.53</td><td>92.68</td><td>93.77</td></tr></table>

TABLE X

ABLATION STUDY OF THE LARGE-KERNEL SHARING STRATEGIES IN TERMS OF OA (%).
<table><tr><td>Modality Shared</td><td>Depth Shared</td><td>Houston 2013</td><td>Augsburg</td><td>Houston 2018</td></tr><tr><td>X</td><td>X</td><td>90.46</td><td>91.42</td><td>92.28</td></tr><tr><td>√</td><td>X</td><td>90.52</td><td>92.26</td><td>93.63</td></tr><tr><td>X</td><td>√</td><td>91.34</td><td>91.76</td><td>93.33</td></tr><tr><td>√</td><td>√</td><td>93.53</td><td>92.68</td><td>93.77</td></tr></table>

Shared-Pattern Collaborative Block (SPCB). Table VIII isolates the Shared Large-Kernel Pattern and Dynamic Kernel Generation while keeping the remaining network, including RAMA, unchanged. The Shared-LK-only and DK-only variants obtain 92.67%, 92.19%, and 93.24%, and 91.13%, 91.62%, and 92.12%, respectively, on Houston2013, Augsburg, and Houston2018. The Shared-LK-only variant is higher than the DK-only variant by 1.54, 0.57, and 1.12 percentage points, indicating a stronger individual contribution of the shared long-range pattern in these experiments. The complete SPCB reaches 93.53%, 92.68%, and 93.77%, outperforming the Shared-LK-only variant by 0.86, 0.49, and 0.53 percentage points and the DK-only variant by 2.40, 1.06, and 1.65 percentage points. To assess input-conditioned filtering, Table VIII replaces the dynamic branch with two static controls of comparable capacity while retaining Shared-LK and the remaining network. Static-DW (generator) feeds a learned constant into the original generator; Static-DW (pointwise) uses a fixed depth-wise kernel with pointwise expansion and projection. Neither consistently improves on Shared-LK only: on Augsburg, OA drops by 0.41 and 0.29 percentage points, respectively, and both remain below Full SPCB on all three datasets. This may reflect the inability of fixed filters to adapt to input structures and their redundant responses with Shared-LK.

Region-Adaptive Modality Allocator (RAMA). Table IX compares four fusion strategies with different levels of sourceselection adaptivity. 1×1 convolution concatenates the features and learns a linear channel mixture, but the same convolutional rule is applied at every spatial location. Global allocator predicts one pair of modality-importance weights from a globally aggregated representation and applies the weights to the entire scene. Regional allocation generates spatially varying allocation maps so that different regions can favor different sources; to isolate the contribution of regional allocation, its final fusion coefficients are fixed to $\alpha ~ = ~ \beta ~ = ~ 0 . 5$ . Full RAMA combines the regional allocation maps with learnable fusion coefficients, enabling both spatial source selection and adaptive final aggregation.

The 1 × 1 convolution produces OA values of 91.19%, 90.74%, and 92.58% on Houston2013, Augsburg, and Houston2018, respectively. The global allocator improves the results to 93.53%, 92.26%, and 93.53%, demonstrating the benefit of learning source reliability, although one global weight pair cannot describe all local regions. Regional allocation obtains 93.29%, 92.31%, and 93.63%, showing that spatially varying source selection is useful in Augsburg and Houston2018, although it is slightly lower than the global allocator on Houston2013. Full RAMA achieves 93.53%, 92.68%, and 93.77%, tying the global allocator on Houston2013 and giving the best OA on the other two datasets. Compared with the global allocator, Full RAMA improves OA by 0, 0.42, and 0.24 percentage points, respectively; compared with regional allocation, it improves OA by 0.24, 0.37, and 0.14 percentage points. These results indicate that regional allocation and learnable final aggregation are complementary, yielding the best or tied-best OA across the three datasets.

Shared-Pattern Collaborative Block (SPCB) Sharing Strategy. Table X analyzes the effect of sharing the largekernel operator across modalities and network depths. Its first row corresponds to an independently parameterized setting in which the large-kernel operators are not shared across modalities or network depths, rather than a setting that removes the large-kernel branch. Without either form of sharing, OA is 90.46%, 91.42%, and 92.28% on Houston2013, Augsburg, and Houston2018, respectively. Cross-modal sharing alone changes these results to 90.52%, 92.26%, and 93.63%, giving gains of 0.06, 0.84, and 1.35 percentage points. Cross-depth sharing alone obtains 91.34%, 91.76%, and 93.33%, improving OA by 0.88, 0.34, and 1.05 percentage points. When both sharing strategies are enabled, OA reaches 93.53%, 92.68%, and 93.77%, exceeding the independently parameterized variant by 3.07, 1.26, and 1.49 percentage points. The combined strategy achieves the highest OA on all three datasets. These results support the complementarity of cross-modal and crossdepth sharing, while the dataset-dependent gains suggest that their relative contributions are not uniform across scenes. On Houston2013, the gain from cross-modal sharing increases from 0.06 to 2.19 percentage points when cross-depth reuse is enabled. Similar paved/built-up classes and narrow roads and railways make spatial structure informative [17]. Reusing the shared operator across modalities and depths helps retain this geometry as features evolve, while identity channels and dynamic filters preserve early and source-specific cues.

TABLE XI  
COMPARISON OF MODEL PARAMETERS, FLOATING-POINT OPERATIONS (FLOPS).
<table><tr><td>Metrics</td><td>FusAtNet</td><td>S2ENet</td><td>AsyFFNet</td><td>ExViT</td><td>HCT</td><td>MACN</td><td>MICF-Net</td><td>GCCQTNet</td><td>CHNet</td><td>MGMNet</td><td>SPCNet</td></tr><tr><td>Params (M)</td><td>36.5760</td><td>1.1752</td><td>1.0654</td><td>0.2339</td><td>5.9186</td><td>2.3472</td><td>0.1332</td><td>1.2322</td><td>20.8156</td><td>4.3010</td><td>1.0893</td></tr><tr><td>FLOPs (G)</td><td>6.6950</td><td>0.2708</td><td>0.1747</td><td>0.0642</td><td>0.0849</td><td>0.3338</td><td>0.0377</td><td>0.0493</td><td>0.3214</td><td>0.1455</td><td>0.2470</td></tr></table>

## E. Computational Complexity Analysis

Table XI compares the number of parameters and floatingpoint operations (FLOPs) of different models. SPCNet contains 1.0893 M parameters and requires 0.2470 GFLOPs. Its parameter count is approximately 97.02% and 94.77% lower than those of FusAtNet and CHNet, respectively, indicating that the proposed modules do not rely on a heavyweight backbone. Although several highly lightweight competitors require less computation, SPCNet still has substantially lower FLOPs than FusAtNet, MACN, and CHNet. With controllable parameter and computational costs, the model achieves the highest overall accuracy (OA) on all three datasets. These results demonstrate that SPCNet achieves a favorable tradeoff between accuracy and efficiency.

## F. Discussion

Current experiments use registered multimodal imagery and do not quantify the effect of residual misregistration on crossmodal sharing. Future work can introduce controlled spatial offsets to measure this sensitivity and investigate registration aware feature alignment to improve robustness.

RAMA learns location-wise weights under the classification objective to allocate branch contributions, but the correspondence between these weights and sensor quality remains to be assessed. Future work can introduce controlled modalityspecific degradation to examine how allocations respond to changing input quality and develop quality-aware fusion. Although experiments cover three datasets and HSI–LiDAR and HSI–SAR combinations, they are evaluated using datasetspecific training and test splits, leaving their ability to generalize to unseen geographic scenes unexamined. Future work can combine cross-scene evaluation with domain generalization or lightweight adaptation to assess and improve transfer across acquisition conditions.

## V. CONCLUSION AND FUTURE WORK

We proposed SPCNet for multi-source remote sensing data classification. Specifically, the SPCB introduces a shared large-kernel mechanism and dynamic convolution to enhance common spatial-pattern learning while preserving modalityspecific representations. Moreover, the RAMA adaptively assigns modality weights for different spatial regions, enabling more effective and reliable cross-modal feature fusion. Experimental results on the Houston2013, Augsburg, and Houston2018 datasets demonstrate that SPCNet achieves superior classification performance compared with several state-ofthe-art methods. Future work will investigate more efficient dynamic-kernel generation and evaluate the proposed collaborative learning strategy in broader multi-source remote sensing scenarios.

## REFERENCES

[1] M. Ahmad, S. Shabbir, S. K. Roy, D. Hong, X. Wu, J. Yao, A. M. Khan, M. Mazzara, S. Distefano, and J. Chanussot, “Hyperspectral image classification—traditional to deep models: A survey for future prospects,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 15, pp. 968–999, 2022.

[2] C. Gong, F. Gao, J. Lin, J. Dong, and Q. Du, “Representative spectral correlation network for multisource remote sensing image classification,” IEEE Transactions on Geoscience and Remote Sensing, vol. 64, pp. 1– 14, 2026.

[3] Q. Lv, Z. Qi, H. Li, W. Song, D. Huang, Z. Ding, and Q. Shi, “Evaluating the potential of multi-source remote sensing and urban big data for urban land use classification,” International Journal of Digital Earth, vol. 18, no. 1, p. 2546519, 2025.

[4] X. Shu, C. Ye, Z. Xu, R. Liao, and S. Zhang, “A multiscale physicallybased approach to urban flood risk assessment using abm and multisource remote sensing data,” International Journal of Disaster Risk Reduction, vol. 119, p. 105332, 2025.

[5] Y. Gao, J. Li, Y. Wu, L. Zhang, D. Lei, L. Kong, and L. Yue, “PyS<sup>2</sup>CF-Mamba: A pyramid spatial–spectral competitive fusion Mamba network for hyperspectral image classification,” IEEE Geoscience and Remote Sensing Letters, vol. 23, pp. 1–5, 2026.

[6] S. Hao, W. Wang, and M. Salzmann, “Geometry-aware deep recurrent neural networks for hyperspectral image classification,” IEEE Transactions on Geoscience and Remote Sensing, vol. 59, no. 3, pp. 2448–2460, 2021.

[7] S. K. Roy, S. Manna, T. Song, and L. Bruzzone, “Attention-based adaptive spectral–spatial kernel resnet for hyperspectral image classification,” IEEE Transactions on Geoscience and Remote Sensing, vol. 59, no. 9, pp. 7831–7843, 2021.

[8] D. Hong, C. Li, N. Yokoya, B. Zhang, X. Jia, A. Plaza, P. Gamba, J. A. Benediktsson, and J. Chanussot, “Hyperspectral imaging,” Nature Reviews Methods Primers, vol. 6, no. 1, p. 19, 2026.

[9] S. Li, W. Song, L. Fang, Y. Chen, P. Ghamisi, and J. A. Benediktsson, “Deep learning for hyperspectral image classification: An overview,” IEEE Transactions on Geoscience and Remote Sensing, vol. 57, no. 9, pp. 6690–6709, 2019.

[10] B. Rasti, D. Hong, R. Hang, P. Ghamisi, X. Kang, J. Chanussot, and J. A. Benediktsson, “Feature extraction for hyperspectral imagery: The evolution from shallow to deep: Overview and toolbox,” IEEE Geoscience and Remote Sensing Magazine, vol. 8, no. 4, pp. 60–88, 2020.

[11] J. Li, K. Zheng, L. Gao, Z. Han, Z. Li, and J. Chanussot, “Enhanced deep image prior for unsupervised hyperspectral image super-resolution,” IEEE Transactions on Geoscience and Remote Sensing, vol. 63, pp. 1–18, 2025.

[12] J. Li, K. Zheng, L. Gao, L. Ni, M. Huang, and J. Chanussot, “Modelinformed multistage unsupervised network for hyperspectral image super-resolution,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–17, 2024.

[13] P. Ghamisi, R. Gloaguen, P. M. Atkinson, J. A. Benediktsson, B. Rasti, N. Yokoya, Q. Wang, B. Hofle, L. Bruzzone, F. Bovolo, M. Chi, and K. Anders, “Multisource and multitemporal data fusion in remote sensing: A comprehensive review of the state of the art,” IEEE Geoscience and Remote Sensing Magazine, vol. 7, no. 1, pp. 6–39, 2019.

[14] Z. Xue, X. Yu, A. Yu, B. Liu, P. Zhang, and S. Wu, “Self-supervised feature learning for multimodal remote sensing image land cover classification,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–15, 2022.

[15] J. Lin, F. Gao, X. Shi, J. Dong, and Q. Du, “SS-MAE: Spatial– spectral masked autoencoder for multisource remote sensing image classification,” IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1–14, 2023.

[16] J. Hu, R. Liu, D. Hong, A. Camero, J. Yao, M. Schneider, F. Kurz, K. Segl, and X. X. Zhu, “MDAS: A new multimodal benchmark dataset for remote sensing,” Earth System Science Data, vol. 15, no. 1, pp. 113–131, 2023.

[17] C. Debes, A. Merentitis, R. Heremans, J. Hahn, N. Frangiadakis, T. van Kasteren, W. Liao, R. Bellens, A. Pizurica, S. Gautama, W. Philips, S. Prasad, Q. Du, and F. Pacifici, “Hyperspectral and LiDAR data fusion: Outcome of the 2013 GRSS data fusion contest,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 7, no. 6, pp. 2405–2418, 2014.

[18] D. Hong, J. Hu, J. Yao, J. Chanussot, and X. X. Zhu, “Multimodal remote sensing benchmark datasets for land cover classification with a shared and specific feature learning model,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 178, pp. 68–80, 2021.

[19] K. Ding, T. Lu, W. Fu, S. Li, and F. Ma, “Global–local transformer network for HSI and LiDAR data joint classification,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–13, 2022.

[20] D. Hong, L. Gao, N. Yokoya, J. Yao, J. Chanussot, Q. Du, and B. Zhang, “More diverse means better: Multimodal deep learning meets remotesensing imagery classification,” IEEE Transactions on Geoscience and Remote Sensing, vol. 59, no. 5, pp. 4340–4354, 2021.

[21] X. Xu, W. Li, Q. Ran, Q. Du, L. Gao, and B. Zhang, “Multisource remote sensing data classification based on convolutional neural network,” IEEE Transactions on Geoscience and Remote Sensing, vol. 56, no. 2, pp. 937–949, 2018.

[22] R. Hang, Z. Li, P. Ghamisi, D. Hong, G. Xia, and Q. Liu, “Classification of hyperspectral and LiDAR data using coupled CNNs,” IEEE Transactions on Geoscience and Remote Sensing, vol. 58, no. 7, pp. 4939–4950, 2020.

[23] S. Mohla, S. Pande, B. Banerjee, and S. Chaudhuri, “FusAtNet: Dual attention based spectrospatial multimodal fusion network for hyperspectral and LiDAR classification,” in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2020, pp. 416– 425.

[24] J. Yao, B. Zhang, C. Li, D. Hong, and J. Chanussot, “Extended vision transformer (ExViT) for land use and land cover classification: A multimodal deep learning framework,” IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1–15, 2023.

[25] Y. Zhang, S. Xu, D. Hong, H. Gao, C. Zhang, M. Bi, and C. Li, “Multimodal transformer network for hyperspectral and LiDAR classification,” IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1– 17, 2023.

[26] S. Fang, K. Li, and Z. Li, “S2ENet: Spatial–spectral cross-modal enhancement network for classification of hyperspectral and LiDAR data,” IEEE Geoscience and Remote Sensing Letters, vol. 19, pp. 1– 5, 2022.

[27] H. Pan, Q. Zhang, H. Ge, M. Liu, and C. Shi, “PyionNet: Pyramid progressive cross-fusion network for joint classification of hyperspectral and LiDAR data,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 18, pp. 12 042–12 058, 2025.

[28] C. Liang, Y. Zhao, Y. Song, and K. Ni, “IDNet: Intensity-constrained detail-enhanced network for hyperspectral and LiDAR collaborative classification,” IEEE Transactions on Geoscience and Remote Sensing, vol. 63, pp. 1–15, 2025.

[29] X. Wu, D. Hong, and J. Chanussot, “Convolutional neural networks for multimodal remote sensing data classification,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–10, 2022.

[30] H.-C. Li, W.-S. Hu, W. Li, J. Li, Q. Du, and A. Plaza, “A3CLNN: Spatial, spectral and multiscale attention ConvLSTM neural network for multisource remote sensing data classification,” IEEE Transactions

on Neural Networks and Learning Systems, vol. 33, no. 2, pp. 747–761, 2022.

[31] W. Li, Y. Gao, M. Zhang, R. Tao, and Q. Du, “Asymmetric feature fusion network for hyperspectral and SAR image classification,” IEEE Transactions on Neural Networks and Learning Systems, vol. 34, no. 10, pp. 8057–8070, 2023.

[32] X. Tang, Y. Zou, J. Ma, X. Zhang, F. Liu, and L. Jiao, “Multiple information collaborative fusion network for joint classification of hyperspectral and LiDAR data,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–16, 2024.

[33] Y. Gao, W. Li, M. Zhang, J. Wang, W. Sun, R. Tao, and Q. Du, “Hyperspectral and multispectral classification for coastal wetland using depthwise feature interaction network,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–15, 2022.

[34] X. Zhao, R. Tao, W. Li, W. Philips, and W. Liao, “Fractional gabor convolutional network for multisource remote sensing data classification,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–18, 2022.

[35] M. Wang, F. Gao, J. Dong, H.-C. Li, and Q. Du, “Nearest neighbor-based contrastive learning for hyperspectral and LiDAR data classification,” IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1– 16, 2023.

[36] Y. Gao, M. Zhang, J. Wang, and W. Li, “Cross-scale mixing attention for multisource remote sensing data fusion and classification,” IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1–15, 2023.

[37] S. K. Roy, A. Deria, D. Hong, B. Rasti, A. Plaza, and J. Chanussot, “Multimodal fusion transformer for remote sensing image classification,” IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1– 20, 2023.

[38] G. Zhao, Q. Ye, L. Sun, Z. Wu, C. Pan, and B. Jeon, “Joint classification of hyperspectral and LiDAR data using a hierarchical CNN and transformer,” IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1–16, 2023.

[39] K. Li, D. Wang, X. Wang, G. Liu, Z. Wu, and Q. Wang, “Mixing self-attention and convolution: A unified framework for multi-source remote sensing data classification,” IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1–16, 2023.

[40] H. Chang, H. Bi, F. Li, C. Xu, J. Chanussot, and D. Hong, “Deep symmetric fusion transformer for multimodal remote sensing data classification,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–15, 2024.

[41] Y. Zhang, Y. Peng, B. Tu, and Y. Liu, “Local information interaction transformer for hyperspectral and LiDAR data classification,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 16, pp. 1130–1143, 2023.

[42] L. Sun, X. Wang, Y. Zheng, Z. Wu, and L. Fu, “Multiscale 3-d–2-d mixed CNN and lightweight attention-free transformer for hyperspectral and LiDAR classification,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–16, 2024.

[43] K. Ni, D. Wang, Z. Zheng, and P. Wang, “MHST: Multiscale head selection transformer for hyperspectral and LiDAR classification,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 17, pp. 5470–5483, 2024.

[44] W.-S. Hu, W. Li, H.-C. Li, F.-H. Huang, and R. Tao, “Global clue-guided cross-memory quaternion transformer network for multisource remote sensing data classification,” IEEE Transactions on Neural Networks and Learning Systems, vol. 36, no. 4, pp. 7357–7371, 2025.

[45] K. Ni, Y. Xie, G. Zhao, Z. Zheng, P. Wang, and T. Lu, “Coarse-to-fine high-order network for hyperspectral and LiDAR classification,” IEEE Transactions on Geoscience and Remote Sensing, vol. 63, pp. 1–16, 2025.

[46] W. Zhang, W. Song, J. Wang, and W. Gao, “MGMNet: Mutual-guidance mechanism for joint classification of multisource remote sensing data,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 18, pp. 1085–1097, 2025.

[47] Y. Han, H. Zhu, L. Jiao, X. Yi, X. Li, B. Hou, W. Ma, and S. Wang, “SSMU-Net: A style separation and mode unification network for multimodal remote sensing image classification,” IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1–15, 2023.

[48] K. Ni, D. Wang, G. Zhao, Z. Zheng, and P. Wang, “Hyperspectral and LiDAR classification via frequency domain-based network,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–17, 2024.

[49] W. Song, Y. Dai, Z. Gao, L. Fang, and Y. Zhang, “Hashing-based deep metric learning for the classification of hyperspectral and LiDAR data,”

IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1– 13, 2023.

[50] F. Gao, X. Jin, X. Zhou, J. Dong, and Q. Du, “MSFMamba: Multiscale feature fusion state space model for multisource remote sensing image classification,” IEEE Transactions on Geoscience and Remote Sensing, vol. 63, pp. 1–16, 2025.

[51] X. Ding, X. Zhang, J. Han, and G. Ding, “Scaling up your kernels to 31x31: Revisiting large kernel design in CNNs,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 11 963–11 975.

[52] Y. Li, X. Li, Y. Dai, Q. Hou, L. Liu, Y. Liu, M.-M. Cheng, and J. Yang, “LSKNet: A foundation lightweight backbone for remote sensing,” International Journal of Computer Vision, vol. 133, no. 3, pp. 1410– 1431, 2025.

[53] Y. Chen, X. Dai, M. Liu, D. Chen, L. Yuan, and Z. Liu, “Dynamic convolution: Attention over convolution kernels,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 11 030–11 039.

[54] B. Yang, G. Bender, Q. V. Le, and J. Ngiam, “CondConv: Conditionally parameterized convolutions for efficient inference,” in Advances in Neural Information Processing Systems, vol. 32, 2019.

[55] J. Arevalo, T. Solorio, M. Montes-y Gomez, and F. A. Gonz´ alez,´ “Gated multimodal units for information fusion,” arXiv preprint arXiv:1702.01992, 2017.

[56] H. Gao, H. Feng, Y. Zhang, S. Fei, R. Shen, S. Xu, and B. Zhang, “Interactive enhanced network based on multihead self-attention and graph convolution for classification of hyperspectral and LiDAR data,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1– 16, 2024.