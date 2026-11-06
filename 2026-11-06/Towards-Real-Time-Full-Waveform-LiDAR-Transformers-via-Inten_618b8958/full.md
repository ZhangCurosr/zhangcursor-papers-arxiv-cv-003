# Towards Real-Time Full-Waveform LiDAR Transformers via Intensity-Guided Token Reduction and Physics-Aware Augmentation

Kotaro Oishi<sup>∗</sup> koutarou00140805@keio.jp Keio University

Ryo Yoshida ryo-yoshida@keio.jp Keio University

Kazuma Ikeda<sup>∗</sup> kazu2080@keio.jp Keio University

Mariko Isogawa<sup>†</sup> mariko.isogawa@keio.jp Keio University

Ryosei Hara ryosei\_hara@keio.jp Keio University

Kentaro Yoshioka<sup>†</sup> kyoshioka47@keio.jp Keio University

## Abstract

LiDAR is a critical sensor for autonomous driving and robotics, yet it remains vulnerable to adverse conditions such as fog, rain, transparent objects, and lowreflectance surfaces. Full-waveform LiDAR (FWL) addresses these limitations by capturing the complete return waveform as a histogram, preserving rich reflection characteristics including multi-path echoes and weak signals. While recent Transformer-based approaches have demonstrated promise on FWL data, two fundamental bottlenecks remain: high computational cost from redundant tokens in low-intensity background regions, and limited generalization due to scarce annotated training data. We address both bottlenecks by exploiting the physical structure inherent to FWL data. First, we propose FWL-ToPM, a token pruning and merging mechanism that leverages intensity distributions to selectively reduce tokens while retaining critical waveform content. Second, we introduce FWLAug, a data augmentation framework designed as a regularization method that preserves waveform structure, retaining the Time-of-Flight waveform of each ray and the frustum geometry, to improve generalization to unseen scenes. Evaluated on the Ghost-FWL benchmark, FWL-ToPM achieves over 8× speedup at real-time throughput alongside a nearly 9-point F1 gain over the prior state of the art, establishing the first real-time FWL Transformer that simultaneously improves accuracy. FWLAug further improves performance on both the standard benchmark and an out-of-distribution test set of unseen scenes. Together, these methods push FWL-based perception toward practical deployment. The code and the OOD dataset are publicly available and can be accessed via the project page: https://keio-csg.github.io/FWL-ToPM

## 1 Introduction

LiDAR is a key sensor for autonomous driving, robotics, and mapping, measuring 3D geometry via Time-of-Flight (ToF) of reflected laser pulses. Conventional LiDAR, however, records only the peak intensity of each pulse, and therefore cannot capture complex reflection characteristics such as multi-path returns from glass or backscatter from fog and rain. These limitations cause spurious point generation (ghosts and noise), degrading object detection and SLAM performance and introducing safety risks.

Full-waveform LiDAR (FWL) addresses these limitations by recording the complete temporal intensity waveform of each pulse as a histogram, thereby preserving rich reflection characteristics including multi-path echoes and weak-signal components. Transformer-based FWL models (FWLTs) exploit this richer representation to enable tasks fundamentally beyond conventional LiDAR, including ghost detection [10] and fog-robust point cloud reconstruction [31]. Despite this promise, two fundamental challenges remain unresolved for practical FWLT deployment.

The first challenge is real-time inference. Because FWL records a complete histogram for every pixel, its data volume is far larger than that of a point cloud, yet the vast majority consists of low-intensity background regions carrying no object information. This redundancy severely inflates computational cost, limiting existing FWLTs to inference rates well below real-time requirements even on high-end GPUs [10]. Existing token reduction methods [2, 30], designed for camera images and videos, assume informative content is broadly distributed across 2D space and thus indiscriminately discard the sparse high-intensity tokens that encode object-relevant reflections in FWL data. Exploiting FWL’s physical structure for token reduction requires a dedicated mechanism.

The second challenge is generalization. FWL observations vary substantially across acquisition environments, object configurations, and reflectance conditions, yet publicly available data remains scarce. Minority classes such as glass and water lack sufficient samples to learn their characteristic reflection signatures, pushing models to overfit dominant classes. Standard image and point cloud augmentations ignore FWL’s ToF waveform structure and frustum geometry and thus distort the waveform of each ray. A dedicated augmentation framework that respects these constraints is essential.

We address both challenges with two methods designed around the physical structure of FWL data. FWL-ToPM reduces computational cost by pruning redundant background tokens based on intensity distributions and merging the remainder, while preserving critical waveform content. FWLAug expands training diversity while respecting FWL’s ToF waveform structure and frustum geometry. Together, these methods advance FWLTs toward practical deployment by simultaneously achieving real-time operation and improved generalization to unseen environments. Our contributions are as follows:

• We propose FWL-ToPM, a token pruning and merging mechanism tailored to FWL data. By leveraging intensity distributions to prune low-intensity background tokens and merge the remainder, FWL-ToPM reduces computational cost while retaining critical waveform content. It achieves an 8× speedup (53.2 FPS) and an approximately 9-point F1 improvement (0.503→0.592), establishing the first real-time FWLT.

• We find that, unlike image-based Transformers where aggressive pruning destabilizes training [16, 30], even pruning up to 70% of tokens can consistently improve FWLT accuracy. Removing low-intensity background tokens allows the model to focus on object-relevant signals that are otherwise diluted by uninformative waveform regions.

• We propose FWLAug, a data augmentation framework tailored to FWL data. FWLAug expands training diversity while preserving the ToF waveform structure and frustum geometry that standard image and point cloud augmentations violate.

• We construct an out-of-distribution (OOD) test set captured entirely outside the Ghost-FWL [10] recording locations, on which FWLAug achieves the best F1-mean among all compared configurations.

## 2 Related Work

Deep Learning Methods for Full-waveform LiDAR. Conventional LiDAR internally selects several high-intensity peak positions for each laser pulse and stores data in a point cloud format $( x \times y \times z )$ , recording only the 3D coordinates of detected points. In contrast, Full-waveform LiDAR (FWL) captures complete temporal light intensity information at each pixel of the receiver, which is stored in the FWL data format $( u \times v \times t )$ , where u and v denote the azimuth and elevation of each ray and t denotes the ToF bin. From the complete histogram at each pixel, arbitrary peaks can be selected and converted into 3D coordinates, enabling the use of all waveform information and providing richer representations. In recent years, various FWL-based studies have attracted attention, taking advantage of its characteristics, such as Classification [5, 17, 32–34, 45], 3D Object Detection [6], Jamming Recovery [41], Fog Removal [31, 38], Ghost Removal [10].

While early FWL-based methods mainly used CNN-based networks, recent advancements in Transformers in the vision field [7, 8, 28] have led to the adoption of Transformer-based networks [10, 31]. Scheuble et al. [31] proposed a Neural Digital Signal Processor (Neural DSP) using U-Net spatiotemporal transformers with FWL data as input, enabling robust point cloud reconstruction in low SNR environments such as fog. However, the Neural DSP partitions tokens only along the time axis while treating spatial dimensions as 1×1, causing computational cost to scale poorly with spatial resolution and limiting throughput to 2 FPS. Ikeda et al. [10] proposed Ghost-FWL, a large-scale annotated FWL dataset for ghost detection and FWL-MAE, a self-supervised pre-training method for FWL data. FWL-MAE treats the time axis as a single tube and applies a 16 × 16 patch division in the spatial direction to reduce computational complexity. Despite this, FWL-MAE reaches only 6 FPS, falling well short of the 10 Hz required for real-time operation.

Computational cost thus remains a fundamental bottleneck for FWLTs. We address this by introducing intensity-guided token pruning and merging tailored to FWL’s sparse structure, targeting the ghost detection task in Ghost-FWL [10] as a representative real-time perception challenge.

Token Pruning and Merging. LiDAR-based systems require real-time performance, whereas deep learning–based methods tend to have high computational costs and low throughput. There is a trade-off between the accuracy and inference time of deep learning models. In particular, for Transformers, efficient vision transformers [4, 20, 28] and token reduction methods such as pruning [9, 13, 14, 19, 30] and merging [1, 2, 15, 22, 23, 27, 29, 37] have been widely studied to reduce computational cost while maintaining accuracy. Recently, methods that combine pruning and merging have also been proposed [3, 12, 24]. However, most of these studies focus on token reduction for images or videos, and their application to FWL data remains unexplored. These methods assume that significant information is widely distributed in 2D space, whereas FWL data has a sparse structure in 3D space where peaks appear at only a few locations and the rest are almost zero, making them unsuitable for FWL data.

In FWL data, regions around peaks contain more important information, while the majority of the data consists of intensities close to zero, making regions outside the peaks redundant and increasing computational cost. Therefore, we propose FWL-ToPM, a token pruning and merging method tailored to FWL data that enables faster inference while maintaining high classification performance.

Augmentation for Full-waveform LiDAR. Augmentation is widely used to improve generalization by introducing diversity into limited training data, with extensive work spanning image recognition and 3D point clouds. For images, geometric and intensity transforms such as CutMix [42] have been proposed, while 3D point cloud methods account for the geometric structure of the data. Mix3D [26] generates new training samples by blending multiple 3D scenes, and PolarMix [40] exploits the polar coordinate structure of LiDAR scans for rotation and region swapping.

No augmentation method has been designed specifically for FWL data. FWL data retains a continuous temporal waveform along each ray direction and forms a frustum-shaped observation structure with the LiDAR origin as its apex. Applying existing image or point cloud augmentations directly disrupts the ToF waveform structure and frustum geometry, causing unnatural distortions in object shape and distance. Augmentations proposed for ToF-based transient imaging [11, 35] are limited to basic operations such as noise addition and rotation, which are insufficient to improve generalization for FWL data.

## 3 Method

We propose a fast and robust framework for estimating the class logits $\pmb { y } \in \mathbb { R } ^ { H \times W \times T \times C }$ of each voxel from FWL data $V \in \mathbb { R } ^ { H \times W \times T }$ , even for scenes that are not included in the training data. Here, H, W, and T denote the spatial and temporal resolutions, respectively; H and W correspond to the elevation (v) and azimuth (u) axes of the $( u \times v \times t )$ format, and T to the ToF (t) axis, and C denotes the number of target classes, such as Noise, Object, Glass, and Ghost.

The proposed framework consists of FWL-ToPM, which enables fast and accurate FWL recognition, and FWLAug, which improves generalization performance for unseen scenes. FWL-ToPM focuses on the spatial sparsity of signal intensity specific to FWL data and achieves efficient feature extraction by reducing redundant tokens. FWLAug generates diverse training samples of waveform structures, object arrangements, and minority classes while preserving the ToF-direction waveform structure specific to FWL data and the frustum geometry based on the LiDAR viewpoint. This increases the diversity of the training data and enables the model to learn more robust representations for unseen scenes. The details of FWL-ToPM and FWLAug are described in §3.1 and §3.2, respectively.

![](images/680effb08a8f78346be2c7c49ff50c777c75ec19ca751158f921b348dac36767.jpg)

![](images/87b42e96bb9fb1319562ac8fa5bea667bbc733d2f102c4cd88b29b8269b33560.jpg)

![](images/6de96168eaf9c20af64f5f7aba8795c1489adf51467200f46194204aed17f06f.jpg)  
Figure 1: Overview of the proposed framework. (Top) During training, FWLAug augments FWL samples while preserving ToF waveform structure and frustum geometry. High-intensity patches are selected by pruning, spatially reordered along a space-filling curve, and processed by the FWL-ToPM Encoder, which merges redundant tokens before self-attention and unmerges afterward across L Transformer blocks. (Bottom Left) Pruning removes low-intensity background tokens, and merging aggregates the remainder, concentrating computation on object-relevant waveform content. (Bottom Right) Proportion of pixel classes across intensity bins in FWL data. Most regions in FWL data correspond to noisy background, and our pruning strategy removes these dominant low-intensity regions.

## 3.1 FWL-ToPM

Recently, the effectiveness of Transformer architectures has been reported for the analysis of FWL data [10, 31]. However, existing methods are largely based on Vision Transformers (ViTs), which were originally designed for images, and therefore do not sufficiently account for the characteristics of FWL data. Unlike images, FWL data are characterized by spatially sparse peaks with high signal intensity, while most regions are occupied by low-intensity signals that contribute little to the classification task. As a result, applying uniform ViT-based tokenization generates many uninformative tokens, which can obscure important high-intensity regions. Consequently, a large portion of computational resources is spent on ineffective tokens. Moreover, since low-intensity tokens dominate the representation, an imbalance in feature representations occurs, leading to degraded recognition performance. These issues cannot be resolved simply by applying token merging methods for 2D images [1, 2], which are designed under the assumption that information is relatively homogeneous across pixels.

To address these issues, we propose FWL-ToPM, which explicitly exploits the signal characteristics of FWL data to optimize token representations specifically for FWL Transformers. Specifically, FWL-ToPM combines (1) signal-intensity-based token pruning and (2) spatial-structure-aware token merging to achieve both computational efficiency and recognition performance.

Model Architecture. FWL-ToPM uses a Vision Transformer (ViT) [7] as the backbone for voxelwise classification. However, the main contribution of our method lies in its token selection and aggregation mechanisms, which are applicable to any token-based models. First, the input frame $\breve { V } \in \breve { \mathbb R } ^ { H \times W \times T }$ is divided into 3D patches of patch size $\pmb { p } = \left( p _ { H } , p _ { W } , p _ { T } \right)$ . Then, each patch is projected into an E-dimensional patch token by a 3D convolution with both kernel size and stride set to p. A 3D positional embedding corresponding to the position of each patch is added to these tokens, yielding the token sequence used as the input to the encoder. Next, these tokens are input into a Transformer encoder composed of $N _ { L }$ blocks. Each Transformer block consists of a multi-head selfattention layer with $N _ { h }$ attention heads and an MLP. When pruning and merging are not introduced, the encoder processes all patch tokens as they are, and E-dimensional features are extracted for each patch at the final layer. Finally, a single-layer linear decoder is applied to each token to predict class logits for the voxels within the corresponding patch. These logits are rearranged according to the original patch layout to obtain the final output $\mathbf { \boldsymbol { y } } ^ { \perp } \in \mathbb { R } ^ { H \times W \times T \times C }$ . The following describes the details of the proposed token pruning and token merging.

Intensity-Based Pruning. In FWL data, peaks in the LiDAR histogram with high signal intensity are sparsely distributed, while most low signal intensity regions correspond to noise that does not contribute to the classification task. Therefore, our method evaluates the importance of each patch based on its maximum signal intensity and performs token pruning to remove low signal intensity patches in advance. Specifically, the procedure is as follows: First, we store the maximum signal intensity within the i-th patch as $m _ { i }$ . Next, all patches are sorted in ascending order according to $m _ { i } .$ , and the bottom $\rho _ { \mathrm { l o w } }$ proportion of patches are assigned to $P _ { \mathrm { l o w } }$ as removal targets, while the remaining patches are assigned to $P _ { \mathrm { h i g h } }$ . Only the patches belonging to $P _ { \mathrm { h i g h } }$ are tokenized and fed into the encoder. Since the removed patches do not pass through the encoder, the decoder cannot predict class logits for the corresponding patch regions. Therefore, during training, the loss is computed by excluding the losses for the pruned patch regions, while during inference, all voxels in those patch regions are predicted as the noise label.

Our proposed pruning method consists only of patch selection and modification of the token pathway. It does not modify the weights of the patch embedding, encoder, and decoder. Therefore, this pruning method can be applied to a pretrained model at inference time without additional training.

Sparse-Aware Token Merging. Although pruning can remove most highly redundant tokens, redundant tokens may still remain among the patches in $P _ { \mathrm { h i g h } }$ after pruning. Therefore, we apply token merging to the tokens in $P _ { \mathrm { h i g h } }$ to further reduce the number of tokens used in attention computation. Token merging is a method that selects representative destination tokens (dst tokens) from a token sequence and aggregates the remaining source tokens (src tokens) into the dst tokens based on feature similarity. To make token merging work effectively, it is necessary to appropriately sample dst tokens in a way that reflects the characteristics of the data.

In the case of FWL data, spatially neighboring patches are likely to contain similar waveform information originating from the same object or reflection region. Therefore, the dst tokens used for merging should be distributed across the original 3D space rather than being biased toward specific local regions. A key challenge in applying token merging after pruning is that $P _ { \mathrm { h i g h } }$ forms a sparse and irregular subset of the original 3D patch grid. Existing dst token selection methods divide the patch grid into fixed-size regions and sample one dst token per region [1, 36, 43, 44]; these methods assume a dense, uniform token distribution and fail when applied to a post-pruning sparse set, as regions may contain no tokens at all.

We resolve this by serializing $P _ { \mathrm { h i g h } }$ along a space-filling curve prior to dst selection. A space-filling curve maps the 3D patch grid to a 1D sequence such that patches that are close in 3D space are placed near each other in the sequence. Using this property, we compute the position along the curve of each patch token in $P _ { \mathrm { h i g h } }$ from its coordinates in the 3D patch grid before pruning, and reorder the tokens accordingly. This constructs a token sequence that preserves spatial relationships even when gaps are introduced into the token sequence by pruning. We then select dst tokens at equal intervals from the serialized token sequence according to the dst token ratio $\rho _ { d } .$ , and treat the remaining tokens as src tokens. Next, we compute the cosine similarity between the features of each src token and each dst token, and assign each src token to the dst token with the highest similarity. Then, based on the merge ratio $\rho _ { m } ,$ src tokens with high similarity are merged by averaging them with their corresponding dst tokens.

![](images/10057c6d24bb3a60745e3300d184d5187c0032280eb4e5723da6ef2446947e45.jpg)  
Figure 2: Illustration of FWLAug. FWLShiftMix (top): an object in Sample 1 is shifted away from the LiDAR, creating a blank region $\Omega _ { \mathrm { b l a n k } }$ . A glass-containing region $\Omega _ { \mathrm { s r c } }$ is then selected from Sample 2 and pasted into the blank, producing an augmented scene in which the waveform structure of each ray is preserved. FWLCutMix (bottom): a crop region $\Omega _ { \mathrm { s r c } }$ containing minority class (glass) is selected from a source frame and pasted into a destination frame at a random location. Both methods prioritize glass-containing regions to mitigate class imbalance.

This serialization does not depend on a particular curve, as long as the curve preserves spatial proximity in the 1D sequence. In our experiments, we adopt the Morton order [25], which is obtained by interleaving the coordinate bits of the three axes for each patch. Serializing 3D data with spacefilling curves has also been adopted in Point Transformer V3 (PTv3) [39], which orders point clouds along Z-order and Hilbert curves to partition them into patches for local attention. Our approach shares the same principle of preserving spatial proximity in a 1D sequence, but differs in its input and purpose: we serialize the sparse and irregular set of 3D waveform patches that remain after intensity-based pruning, in order to select spatially distributed dst tokens for merging.

Since token merging is performed to reduce the computational cost of attention, after the attention computation, the number of tokens must be restored by reassigning the merged destination features to each token, thereby maintaining consistency with the residual connections. Therefore, after the attention computation, for each merged src token, we copy the feature of the dst token into which it was merged and redistribute it back to the original token position. Our merging method also does not change the parameter weights, and can be applied to pre-trained models without additional training.

## 3.2 FWLAug

We propose FWLAug, an augmentation framework designed around the physical structure of FWL data (Figure 2). In FWL data $\mathbf { \bar { \boldsymbol { V } } } \in \mathbb { R } ^ { H \times W \times T }$ , each per-pixel histogram $Q _ { h , w }$ encodes the temporal intensity response of a laser pulse, preserving time-of-flight delays, reflectance, and multi-path waveform characteristics. Since each $Q _ { h , w }$ corresponds to a ray from the LiDAR origin, the full FWL frame forms a frustum-shaped observation structure. These properties impose two requirements on any FWL augmentation: (1) the T-axis waveform structure of each $Q _ { h , w }$ must be preserved, and (2) consistency with frustum geometry must be maintained. While FWLAug is not intended to serve as a strict optical simulator that reproduces the complete light transport of the augmented scene, it is strategically designed as an effective regularization method. By intentionally focusing on preserving the ray-wise waveform structure, it efficiently increases training diversity and the representation of minority classes while satisfying the two requirements above.

FWLAug comprises two augmentations that address independent challenges. FWLCutMix alleviates minority-class under-representation by prioritizing glass-containing regions during patch pasting, while FWLShiftMix expands diversity in observation distance and position by translating objects in 3D space. Used together, they improve generalization to unseen scenes while preserving in-domain accuracy.

FWLShiftMix. Shifting objects away from the LiDAR increases distance diversity but creates blank regions due to the compression effect. Zero-filling these regions introduces unrealistic data distributions absent from real FWL scenes, causing the model to learn spurious artifacts. FWLShiftMix instead fills the blank regions with FWL data from another scene, producing synthetic samples that preserve the waveform structure of each ray.

Concretely, FWLShiftMix proceeds in three steps (Figure 2). First, each pixel’s peak position $t _ { h , w } ^ { * } = \arg \operatorname* { m a x } _ { t } Q _ { h , w } ( t )$ is converted to a 3D point $\mathbf { \Delta } r _ { h , w }$ via ToF geometry. Second, we translate each point along its LiDAR ray direction. Let $\pmb { u } _ { h , w } = \pmb { r } _ { h , w } / \Vert \pmb { r } _ { h , w } \Vert _ { 2 }$ denote the unit ray direction and let $\Delta d \sim \mathcal { U } ( - 3 , 1 5 )$ denote the sampled radial displacement in meters. The translated point is computed as $\pmb { r } _ { h , w } ^ { \prime } = \pmb { r } _ { h , w } + \Delta d \pmb { u } _ { h , w }$ and is then reprojected back into the range view. The waveform at each reprojected pixel is shifted along the $T$ axis according to the corresponding range change, so that the histogram remains consistent with the new object position. Unlike a direct tensor shift, this 3D-space manipulation preserves object geometry. Third, the blank region $\Omega _ { \mathrm { b l a n k } }$ left by the shifted object is filled with waveform data from another scene $V ^ { \mathrm { s r c } }$ . We select a glass-containing source region $\Omega _ { \mathrm { s r c } }$ from $V ^ { \mathrm { s r c } }$ and define a mapping $\phi : \Omega _ { \mathrm { b l a n k } }  \Omega _ { \mathrm { s r c } }$ that assigns each blank pixel to a source pixel. For each pixel in $\Omega _ { \mathrm { b l a n k } }$ , we copy the entire temporal waveform from its mapped source pixel in $\Omega _ { \mathrm { s r c } } .$ This fills the unrealistic blank region while preserving the ToF waveform structure of the pasted FWL data. Prioritizing glass-containing regions further increases minority-class frequency during training.

FWLCutMix. FWLCutMix extends CutMix [42] to FWL data by combining a destination sample $V _ { \mathrm { d s t } } \in \mathbb { R } ^ { H \times W \times T }$ and a source sample $V _ { \mathrm { s r c } } \in \bar { \mathbb { R } } ^ { H \times W \times T }$ to generate a synthetic scene. The crop size is determined by a mixing ratio λ sampled randomly: $e l l _ { H } = \lfloor H \sqrt { 1 - \lambda } \rfloor , e l l _ { W } = \lfloor W \sqrt { 1 - \lambda } \rfloor$ A minority-class region $\bar { \Omega } _ { \mathrm { s r c } }$ is preferentially selected from $V _ { \mathrm { s r c } }$ and pasted at a random location in $V _ { \mathrm { d s t } }$ to mitigate class imbalance. The waveform $Q _ { h , w }$ is pasted in its entirety along the T axis, preserving the temporal waveform structure.

## 4 Experimental Settings

Implementation details. All methods are trained using the Ghost-FWL dataset [10], the only publicly available FWL dataset with peak-level annotations to date. The raw FWL data in Ghost-FWL has a size of $( H \times W \times T ) \stackrel { - } { = } ( 5 1 2 \times 4 0 0 \times 7 0 0 )$ First, the top and bottom 88 bins corresponding to reflections from the ceiling and floor and the front 25 bins containing noise from internal sensor reflections were removed. We also removed the rear 375 bins, since the data contains only a few object peaks in the rear region, resulting in a full-frame size of $( H \times W \times T ) =$ $( 3 3 \dot { 6 } \times 4 0 0 \times \dot { 3 } 0 0 )$ . Furthermore, due to GPU memory limitations in the comparison methods, we use a crop corresponding to one quarter of the full frame, with size $( H \times W \times \dot { T } ) = ( 1 6 8 \times 2 0 0 \times 3 0 0 )$ as the input for all methods in order to keep the input size consistent. For the Vision Transformer, the patch size is set to $( p _ { H } \times p _ { W } \times p _ { T } ) = ( 8 \times 4 \times 2 0 )$ , the number of Transformer blocks $N _ { L }$ is set to $^ { 6 , }$ and the number of attention heads $N _ { h }$ is set to 6. The feature dimension E is set to 240, and the ratio of dst tokens $\rho _ { d }$ is set to 0.1. Following Ghost-FWL [10], we use the focal loss [18] as the training objective. We use AdamW [21] as the optimizer, set the learning rate to $1 . 0 \times 1 0 ^ { - 4 }$ , and train the model for 50 epochs. All training and evaluation were performed on a computer equipped with an Intel Core i9-14900K CPU and a single RTX 6000 Ada Generation GPU. Further details are provided in the supplemental material.

Metrics. Following several prior studies [33, 34], we evaluate performance using the mean F1 score over all classes excluding Noise (F1-mean). To verify the inference speedup achieved by token pruning and merging, we report the number of full-size frames that can be inferred per second (FPS).

Baseline Methods. To investigate the effectiveness of the proposed method, we compare it with three FWL-based methods: (1) the proposed method (Ours), (2) a Transformer-based model using pretraining, which is the state-of-the-art model for FWL-based ghost detection (Ikeda et al.) [10], and (3) a model obtained by attaching a classification head for ghost detection to the backbone of a spatio-temporal Transformer model with a U-Net architecture for digital signal processing using FWL (Scheuble et al.) [31]. We also compare our method with pruning and merging methods used for image-based models: (1) DynamicViT [30], which learns whether each token should be pruned using an MLP and then performs pruning, and (2) ToMe [2], which divides tokens into two groups and merges them based on cosine similarity between tokens.

Table 1: Comparison with FWL-based methods and ablation of token pruning and merging components. For FWL-ToPM, we set $\rho _ { \mathrm { l o w } } = 0 . 7$ and $\rho _ { m } = 0 . 9$ . FWL-ToPM achieves the highest F1 and improves throughput by 8× over the prior state of the art. Our pruning and merging components each outperform their image-based counterparts (DynamicViT and ToMe) and their combination achieves the best F1–FPS trade-off. All methods in this table are trained without FWLAug.
<table><tr><td>Method</td><td>Pruning</td><td>Merging</td><td>F1-mean</td><td>FPS</td></tr><tr><td>Scheuble et al. [31]</td><td></td><td></td><td>0.497</td><td>1.97</td></tr><tr><td>Ikeda et al. [10]</td><td></td><td></td><td>0.503 0.555</td><td>6.54</td></tr><tr><td>Ours w/o FWL-ToPM DynamicViT [30]</td><td></td><td>一</td><td>0.480</td><td>2.01 20.0</td></tr><tr><td>Ours w/Pruning</td><td></td><td></td><td>0.591</td><td>20.3</td></tr><tr><td>ToMe [2]</td><td></td><td></td><td>0.543</td><td>5.24</td></tr><tr><td>Ours w/Merging</td><td></td><td></td><td>0.565</td><td>18.0</td></tr><tr><td>Ours w/FWL-ToPM</td><td></td><td></td><td>0.592</td><td>53.2</td></tr></table>

![](images/8f64c6e2cf4100d5cfbffc87b594f0d64b86678cd8f622f7fbeefdbe3974bcee.jpg)  
Figure 3: (Left) F1-mean vs. FPS trade-off for all compared methods. (Right) Comparison of F1-mean and FPS without additional training. For pruning and merging, $\rho _ { \mathrm { l o w } }$ and $\rho _ { m }$ are varied from 0 to 0.9, respectively. For pruning+merging, $\rho _ { \mathrm { l o w } }$ is fixed to 0.7 while $\rho _ { m }$ is varied. All models are trained without FWLAug.

Augmentation. At each training iteration, either FWLCutMix or FWLShiftMix is selected with equal probability (0.5). For FWLCutMix, 1–3 glass regions are sampled from a randomly selected training frame and pasted into the current frame. For FWLShiftMix, the shift magnitude is sampled uniformly from [−3, 15] meters.

OOD Test Set. To evaluate generalization, we construct an OOD test set at a location entirely separate from Ghost-FWL [10] (see supplementary). The data were collected with the same FWL sensor platform and acquisition configuration as Ghost-FWL and annotated following the same protocol, so that the evaluation isolates changes in scene distribution rather than changes in sensor hardware or annotation procedure. The 500-frame set contains over 300 glass instances and scenes with multiple spatially separated glass regions not present in Ghost-FWL, providing a challenging benchmark for FWLT out-of-distribution evaluation.

## 5 Experiments and Results

In this section, we conducted the following experiments. To evaluate the effectiveness of the FWL-ToPM architecture, we conducted (1) a comparison with existing FWL-based recognition methods and (2) a comparison with image-based token reduction baselines. We then ablate (3) the pruning and merging components and the (4) FWL augmentations. We also (5) analyze the training-free applicability of token reduction, and (6) discuss limitations.

Comparison with existing FWL-based recognition methods. To evaluate the effectiveness of the FWL-ToPM architecture, we compared its accuracy and computational cost with those of existing FWL-based recognition methods. The results are presented in Table 1 and the left part of Figure 3.

![](images/9abe43468b2001598c9d3a055acbc10e4804a6b1280824030856b2c637fe7d1f.jpg)  
Figure 4: Visualization of token merging on two real Ghost-FWL scenes. Color indicates patch intensity (red: high, white: low). After pruning low-intensity background tokens, the remaining high-intensity tokens are merged by aggregating similar-feature source tokens into destination tokens, reducing the total token count while preserving object-relevant content. The zoomed-in regions show that merged tokens retain the spatial structure of reflected objects across diverse scenes.

Our FWL-ToPM simultaneously achieved a high F1-mean score of 0.592 and an inference speed of 53.2 FPS. Compared with the baselines of Scheuble et al. [31] and Ikeda et al. [10], this corresponds to an approximately 9-point improvement in the F1 score and a speedup of more than 8× in inference speed, demonstrating that FWL-ToPM achieves both high accuracy and high efficiency. A qualitative evaluation of the classification results for the comparison with existing FWL-based methods i provided in the supplementary material.

Comparison with existing pruning and merging methods. To verify the effectiveness of the proposed pruning and merging methods, we compared them with existing methods designed for image-based Transformers. The results are presented in Table 1. As a result, both our pruning and merging methods individually outperformed the existing methods, confirming that each component i independently effective.

Ablation study on pruning and merging. To clarify the contribution of each component in the proposed method, we conducted an ablation study on pruning and merging. The results are presented in Table 1 and the left part of Figure 3. As shown in Table 1, the F1-mean score was 0.555 when neither component was applied, whereas it increased to 0.591 when only token pruning was applied and to 0.565 when only token merging was applied. These results confirm that both components contribute to improving accuracy.

Moreover, as shown in the left part of Figure 3, token pruning alone maintains high accuracy up to around 20 FPS, with an F1-mean of approximately 0.59. However, when the token pruning ratio is increased to achieve a speed of around 80 FPS, the accuracy drops substantially. This is likely because token pruning also removes high-intensity information that is essential for classification. On the other hand, token merging alone suppresses the degradation in accuracy, but its ability to reduce computational cost is limited, resulting in a speed of only around 18 FPS. By combining pruning and merging, the proposed method removes unnecessary background regions through pruning while aggregating the remaining effective waveform information through merging. This complementary design mitigates the limitations of each individual component and enables the method to simultaneously achieve high accuracy and real-time performance exceeding 50 FPS.

Ablation Study of FWLAug. Table 2 shows the contribution of each FWLAug component to both in-distribution and OOD performance. The full FWLAug achieves the largest OOD F1 gain (4.6-point over baseline), particularly on the minority Glass class (8.4-point), confirming improved generalization to unseen scenes. FWLShiftMix contributes more to in-distribution accuracy, while the combination of both drives the OOD improvement, suggesting complementary effects.

Applicability to pretrained models. To evaluate the behavior of token reduction without additional training, we analyzed the performance when pruning and merging were directly applied to pretrained models. The results are presented in the right part of Figure 3. When token pruning was applied without additional training, the accuracy dropped sharply as the reduction ratio increased. This is likely because the model, which was trained using all tokens, also relies on information from low-intensity patches. In contrast, when token merging was applied without additional training, the F1-mean remained above 0.5 even when the inference speed was increased to around 20 FPS, indicating that severe accuracy degradation did not occur. This result indicates that the proposed merging method can effectively reduce computational cost without substantially degrading feature representations, even without additional training.

Table 2: Ablation study of FWLAug. To evaluate out-of-distribution generalization, we construct an OOD test set captured at a location entirely separate from Ghost-FWL [10]. The full FWLAug achieves the best F1 on the OOD test set, particularly for the minority Glass class, confirming improved generalization to unseen scenes.
<table><tr><td>Method</td><td>FWLCutMix</td><td>FWLShiftMix</td><td colspan="2">Ghost-FWL [10]</td><td colspan="2">OOD Test</td></tr><tr><td>Ours w/o Aug</td><td></td><td></td><td>F1-mean</td><td>Glass</td><td>F1-mean</td><td>Glass</td></tr><tr><td rowspan="4">Ours w/ FWLAug</td><td>x</td><td>X √</td><td>0.592</td><td>0.352</td><td>0.406</td><td>0.491</td></tr><tr><td>x</td><td></td><td>0.622</td><td>0.376</td><td>0.425 0.452</td><td>0.494</td></tr><tr><td>√</td><td>√</td><td>0.617</td><td>0.385</td><td></td><td>0.575</td></tr><tr><td>√</td><td>X</td><td>0.586</td><td>0.315</td><td>0.434</td><td>0.606</td></tr></table>

Limitations. While our proposed framework demonstrates significant improvements, we acknowledge several limitations that provide directions for future work:

• Risk of removing weak signals: FWL-ToPM prunes low-intensity regions by regarding them as background or noise, and therefore may incorrectly remove useful weak signals, such as distant objects, low-reflectance objects, or weak reflections under adverse weather conditions. Since pruned patches are treated as noise during inference, such erroneous removal can directly lead to false negatives. Thus, adaptively determining $\rho _ { \mathrm { l o w } } , \rho _ { m } ,$ , and ρ<sub>d</sub> according to the scene and signal distribution remains an important direction for future work.

• Single sensor platform: Our evaluation relies on a single FWL sensor platform: generalization is evaluated only on unseen scenes acquired with the same sensor platform, acquisition configuration, and annotation protocol as Ghost-FWL, and cross-sensor evaluation has not yet been conducted. Broader validation across different sensors, resolutions, and weather conditions remains necessary.

• Scope of physical consistency in FWLAug: FWLAug is a regularization method that preserves waveform structure rather than an optical simulator. Specifically, FWLShiftMix does not model distance-dependent intensity attenuation or recompute the optical interactions between the pasted glass regions and the shifted objects, and FWLCutMix does not guarantee consistent reflection paths across the entire scene because of spatial discontinuities at the paste boundaries.

• System-level efficiency evaluation: Our current efficiency evaluation is limited to modellevel throughput. Sensor-to-decision latency, including sensor input, preprocessing, and system integration, as well as deployment on actual embedded robotic platforms, are important directions for future work.

## 6 Conclusion

We presented FWL-ToPM and FWLAug, two complementary methods that address the core bottlenecks of real-time inference and limited generalization for Full-Waveform LiDAR Transformers. FWL-ToPM exploits the intensity sparsity inherent to FWL data to prune redundant background tokens and merge the remainder, achieving 53.2 FPS with an 8× speedup and a 9-point F1 gain over the prior state-of-the-art — establishing the first real-time FWLT. A notable finding is that pruning up to 70% of tokens consistently improves accuracy, suggesting that low-intensity tokens act as noise that dilutes object-relevant signals. FWLAug improves generalization through FWLCutMix and FWLShiftMix, both preserving FWL’s ToF waveform structure and frustum geometry, with gains confirmed on an OOD test set of unseen scenes. Together, these advances bring real-time FWL perception a step closer to deployment in autonomous driving and robotics.

Acknowledgements and disclosures. We gratefully acknowledge Takahiro Kado, Ibuki Fujioka, and Taro Beppu at Sony Semiconductor Solutions for their generous support and for providing the Sony LiDAR sensor used in this work. The Sony LiDAR system played an essential role in enabling this research. This research was supported in part by JST Next-generation Edge AI Semiconductor JPMJES2515, JST CREST JPMJCR21D2, JPMJCR23M4, JST PRESTO JPMJPR22PA, JST ASPIRE JPMJAP2515, JST FOREST JPMJFR242I, JSPS KAKENHI 24K02940, and 25H01159.

## References

[1] Daniel Bolya and Judy Hoffman. Token Merging for Fast Stable Diffusion. In IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 4599–4603, 2023.

[2] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token Merging: Your ViT But Faster. In International Conference on Learning Representations (ICLR), 2023.

[3] Maxim Bonnaerens and Joni Dambre. Learned Thresholds Token Merging and Pruning for Vision Transformers. Transactions on Machine Learning Research, 2023.

[4] Han Cai, Junyan Li, Muyan Hu, Chuang Gan, and Song Han. EfficientViT: Lightweight Multi-Scale Attention for High-Resolution Dense Prediction. In IEEE/CVF International Conference on Computer Vision (ICCV), pages 17302–17313, 2023.

[5] Nesrine Chehata, Li Guo, and Clément Mallet. Airborne LIDAR feature selection for urban classification using random forests. In Laser Scanning, 2009.

[6] Jean-Luc Déziel, Pierre Merriaux, Francis Tremblay, Dave Lessard, Dominique Plourde, Julien Stanguennec, Pierre Goulet, and Pierre Olivier. Pixset: An Opportunity for 3D Computer Vision to Go Beyond Point Clouds with a Full-Waveform LiDAR Dataset. In IEEE International Intelligent Transportation Systems Conference (ITSC), pages 2987–2993, 2021.

[7] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. In International Conference on Learning Representations (ICLR), 2021.

[8] Han, Kai and Wang, Yunhe and Chen, Hanting and Chen, Xinghao and Guo, Jianyuan and Liu, Zhenhua and Tang, Yehui and Xiao, An and Xu, Chunjing and Xu, Yixing and Yang, Zhaohui and Zhang, Yiman and Tao, Dacheng. A Survey on Vision Transformer. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 45(1):87–110, 2023.

[9] Joakim Bruslund Haurum, Sergio Escalera, Graham W. Taylor, and Thomas B. Moeslund. Which Tokens to Use? Investigating Token Reduction in Vision Transformers. In IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), pages 773–783, 2023.

[10] Kazuma Ikeda, Ryosei Hara, Rokuto Nagata, Ozora Sako, Zihao Ding, Takahiro Kado, Ibuki Fujioka, Taro Beppu, Mariko Isogawa, and Kentaro Yoshioka. Ghost-FWL: A Large-Scale Full-Waveform LiDAR Dataset for Ghost Detection and Removal. arXiv preprint arXiv:2603.28224, 2026.

[11] Mariko Isogawa, Ye Yuan, Matthew O’Toole, and Kris M Kitani. Optical non-line-of-sight physics-based 3d human pose estimation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 7013–7022, 2020.

[12] Minchul Kim, Shangqian Gao, Yen-Chang Hsu, Yilin Shen, and Hongxia Jin. Token fusion: Bridging the gap between token pruning and token merging. In Proceedings of the IEEE/CVF Winter Conference on Applications ofComputer Vision (WACV), pages 1383–1392, 2024.

[13] Zhenglun Kong, Peiyan Dong, Xiaolong Ma, Xin Meng, Wei Niu, Mengshu Sun, Xuan Shen, Geng Yuan, Bin Ren, Hao Tang, Minghai Qin, and Yanzhi Wang. SPViT: Enabling Faster Vision Transformers via Latency-Aware Soft Token Pruning. In European Conference on Computer Vision (ECCV), pages 620–640, 2022.

[14] Sheng Li, Qitao Tan, Yue Dai, Zhenglun Kong, Tianyu Wang, Jun Liu, Ao Li, Ninghao Liu, Yufei Ding, Xulong Tang, and Geng Yuan. Mutual Effort for Efficiency: A Similarity-based Token Pruning for Vision Transformers in Self-Supervised Learning. In International Conference on Learning Representations (ICLR), 2025.

[15] Xirui Li, Chao Ma, Xiaokang Yang, and Ming-Hsuan Yang. VidToMe: Video Token Merging for Zero-Shot Video Editing. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 7486–7495, 2024.

[16] Youwei Liang, Chongjian Ge, Zhan Tong, Yibing Song, Jue Wang, and Pengtao Xie. Not all patches are what you need: Expediting vision transformers via token reorganizations. arXiv preprint arXiv:2202.07800, 2022.

[17] Wenzhi Liao, Frieke Van Coillie, Lianru Gao, Liwei Li, Bing Zhang, and Jocelyn Chanussot. Deep Learning for Fusion of APEX Hyperspectral and Full-Waveform LiDAR Remote Sensing Data for Tree Species Mapping. IEEE Access, 6:68716–68729, 2018.

[18] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollar. Focal Loss for Dense Object Detection. In IEEE/CVF International Conference on Computer Vision (ICCV), pages 2999–3007, 2017.

[19] Yifei Liu, Mathias Gehrig, Nico Messikommer, Marco Cannici, and Davide Scaramuzza. Revisiting Token Pruning for Object Detection and Instance Segmentation. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 2658–2668, 2024.

[20] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows. In IEEE/CVF International Conference on Computer Vision (ICCV), pages 10012–10022, 2021.

[21] Ilya Loshchilov and Frank Hutter. Decoupled Weight Decay Regularization. In International Conference on Learning Representations (ICLR), 2017.

[22] Chenyang Lu, Daan de Geus, and Gijs Dubbelman. Content-aware Token Sharing for Efficient Semantic Segmentation with Vision Transformers . In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 23631–23640, 2023.

[23] Wenbo Lu, Shaoyi Zheng, Yuxuan Xia, and Shengjie Wang. ToMA: Token Merge with Attention for Diffusion Models. In International Conference on Machine Learning (ICML), 2025.

[24] Junzhu Mao, Yang Shen, Jinyang Guo, Yazhou Yao, Xiansheng Hua, and Hengtao Shen. Prune and Merge: Efficient Token Compression for Vision Transformer With Spatial Information Preserved. IEEE Transactions on Multimedia, 27:4670–4683, 2025.

[25] G.M. Morton. A Computer Oriented Geodetic Data Base and a New Technique in File Sequencing. International Business Machines Company, 1966.

[26] Alexey Nekrasov, Jonas Schult, Or Litany, Bastian Leibe, and Francis Engelmann. Mix3d: Out-of-context data augmentation for 3d scenes. In International Conference on 3D Vision, pages 116–125, 2021.

[27] Narges Norouzi, Svetlana Orlova, Daan De Geus, and Gijs Dubbelman. ALGM: Adaptive Local-then-Global Token Merging for Efficient Semantic Segmentation with Plain Vision Transformers . In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15773–15782, 2024.

[28] Lorenzo Papa, Paolo Russo, Irene Amerini, and Luping Zhou. A Survey on Efficient Vision Transformers: Algorithms, Techniques, and Performance Benchmarking . IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 46(12):7682–7700, 2024.

[29] Sam Pollard and Michael Wray. Video, How do Your Tokens Merge? In IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 3338–3347, 2025.

[30] Yongming Rao, Wenliang Zhao, Benlin Liu, Jiwen Lu, Jie Zhou, and Cho-Jui Hsieh. DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification. In Conference on Neural Information Processing Systems (NeurIPS), 2021.

[31] Dominik Scheuble, Hanno Holzhüter, Steven Peters, Mario Bijelic, and Felix Heide. Lidar Waveforms are Worth 40x128x33 Words. In IEEE/CVF International Conference on Computer Vision (ICCV), pages 28913–28924, 2025.

[32] Takayuki Shinohara, Haoyi Xiu, and Masashi Matsuoka. FWNetAE: Spatial Representation Learning for Full Waveform Data Using Deep Learning. In IEEE International Symposium on Multimedia (ISM), pages 259–2597, 2019.

[33] Takayuki Shinohara, Haoyi Xiu, and Masashi Matsuoka. FWNet: Semantic Segmentation for Full-Waveform LiDAR Data Using Deep Learning. Sensors, 20(12), 2020.

[34] Takayuki Shinohara, Haoyi Xiu, and Masashi Matsuoka. Semantic Segmentation for Full-Waveform LiDAR Data Using Local and Hierarchical Global Feature Extraction. In Proceedings ofthe 28th International Conference on Advances in Geographic Information Systems, page 640–650, 2020.

[35] Shuochen Su, Felix Heide, Gordon Wetzstein, and Wolfgang Heidrich. Deep end-to-end time-of-flight imaging. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6383–6392, 2018.

[36] Wenhao Sun, Rong-Cheng Tu, Jingyi Liao, Zhao Jin, and Dacheng Tao. AsymRnR: Video Diffusion Transformers Acceleration with Asymmetric Reduction and Restoration. In International Conference on Machine Learning (ICML), 2025.

[37] Tuan Anh Tran, Duy Minh Ho Nguyen, Hoai-Chau Tran, Michael Barz, Khoa D. Doan, Roger Wattenhofer, Vien Anh Ngo, Mathias Niepert, Daniel Sonntag, and Paul Swoboda. How Many Tokens Do 3D Point Cloud Transformer Architectures Really Need? In Conference on Neural Information Processing Systems (NeurIPS), 2025.

[38] Andrew M. Wallace, Abderrahim Halimi, and Gerald S. Buller. Full Waveform LiDAR for Adverse Weather Conditions. IEEE Transactions on Vehicular Technology, 69(7):7064–7077, 2020.

[39] Xiaoyang Wu, Li Jiang, Peng-Shuai Wang, Zhijian Liu, Xihui Liu, Yu Qiao, Wanli Ouyang, Tong He, and Hengshuang Zhao. Point Transformer V3: Simpler Faster Stronger. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

[40] Aoran Xiao, Jiaxing Huang, Dayan Guan, Kaiwen Cui, Shijian Lu, and Ling Shao. Polarmix: A genera data augmentation technique for lidar point clouds. Conference on Neural Information Processing Systems (NeurIPS), 35:11035–11048, 2022.

[41] Ryo Yoshida, Takami Sato, Wenlun Zhang, Yuki Hayakawa, Shota Nagai, Takahiro Kado, Taro Beppu, Ibuki Fujioka, Yunshan Zhong, and Kentaro Yoshioka. Neural Reconstruction of LiDAR Point Clouds under Jamming Attacks via Full-Waveform Representation and Simultaneous Laser Sensing. arXiv preprint arXiv:2604.00371, 2026.

[42] Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, and Youngjoon Yoo. Cutmix: Regularization strategy to train strong classifiers with localizable features. In IEEE/CVF International Conference on Computer Vision (ICCV), pages 6023–6032, 2019.

[43] Evelyn Zhang, Bang Xiao, Jiayi Tang, Qianli Ma, Chang Zou, Xuefei Ning, Xuming Hu, and Linfeng Zhang. Token Pruning for Caching Better: 9× Acceleration on Stable Diffusion for Free. arXiv preprint arXiv:2501.00375, 2024.

[44] Evelyn Zhang, Jiayi Tang, Xuefei Ning, and Linfeng Zhang. Training-Free and Hardware-Friendly Acceleration for Diffusion Models via Similarity-based Token Pruning. AAAI Conference on Artificial Intelligence (AAAI), 39(9):9878–9886, 2025.

[45] Stefano Zorzi, Eleonora Maset, Andrea Fusiello, and Fabio Crosilla. Full-Waveform Airborne LiDAR Data Classification Using Convolutional Neural Networks. IEEE Transactions on Geoscience and Remote Sensing, 57(10):8255–8261, 2019.

## Supplementary Material

## Contents

1 Implementation Details 1   
1.1 Training 1   
1.2 Inference 1   
1.3 Baseline Methods . 2   
2 FWL Generalization Benchmark 2   
3 Additional Experiments and Results 2   
3.1 Statistical Significance of FWL-Based Comparison 2   
3.2 Comparison with Top-K Peak Extraction 2   
3.3 Ablation on Space-Filling Curves . 3   
3.4 Qualitative Evaluation 4

## 1 Implementation Details

## 1.1 Training

We evaluated our method on the Ghost-FWL dataset [2]. Following the original data split, we assigned Scenes 001, 003, 004, 005, 006, 008, and 010 to the training and validation set, and Scenes 002, 007, and 009 to the test set. The original dataset provided by Ikeda et al. [2] contains 13,853 training frames and 2,994 validation frames. To reduce computational cost, we randomly sampled one-third of the data from both sets while preserving the scene distribution. Consequently, our final subsets consist of 4,617 training frames and 998 validation frames. For the test set, we did not perform any sampling and used all 1,427 original frames for evaluation.

To mitigate overfitting in the Transformer encoder, we applied a dropout rate of 0.2 to the positional embeddings, the attention weights, and the output of each block. We also set the drop path rate for the residual connections to 0.2.

The model was trained using the AdamW optimizer with a batch size of 4. The optimizer hyperparameters were set to $\beta _ { 1 } \mathrm { = } 0 . 9 , \bar { \beta _ { 2 } } \mathrm { = } 0 . 9 9 9 , \epsilon = 1 \mathrm { \bar { \times } } 1 0 ^ { - 8 }$ , and weight decay $\lambda = 1 \times \mathsf { \bar { 1 0 ^ { - 2 } } }$ . We employed CosineLRScheduler, a cosine learning rate scheduler with a linear warmup, setting the total number of epochs to 50, the minimum learning rate to 0, the warmup duration to 3 epochs, and the initial warmup learning rate to $1 . 0 \times 1 0 ^ { - 5 }$

For the loss function, we utilized the focal loss [3] with the class weights set to $\alpha _ { \mathrm { g l a s s } } = 1 . 5 6$ $\alpha _ { \mathrm { g h o s t } } = 1 . 1 3 , \alpha _ { \mathrm { o b j e c t } } = 0 . 5 3 6$ , and $\alpha _ { \mathrm { n o i s e } } = 0 . 0 6 2 5$ , and the focusing parameter γ set to 2.0.

All training and processing were performed on a computer equipped with an Intel Core i9-14900K CPU and a single RTX 6000 Ada Generation GPU, running the Ubuntu 20.04 operating system. In this environment, the training time was approximately 5.5 minutes per epoch.

## 1.2 Inference

Because the model input is limited to one-quarter of the full frame, processing an entire frame in a single forward pass is not feasible. To evaluate the full-frame performance, we followed Ikeda et al. [2] by sequentially applying a non-overlapping sliding window to infer local regions. The resulting outputs were then spatially concatenated to reconstruct the full-frame predictions. Furthermore, following Ikeda et al. [2] for label assignment, a voxel was classified into the category with a predicted probability exceeding 0.5; if no class probability exceeded this threshold, it was labeled as Undefined.

## 1.3 Baseline Methods

The baseline methods evaluated in our experiments are primarily based on their official implementations, with appropriate modifications to accommodate our specific task and the 3D data structure. The specific adaptations and settings for each method are detailed below.

Scheuble et al. [5]: As the original model targets a different task, we appended a new prediction head for voxel-wise classification. After upsampling the decoder output features to the original spatio-temporal resolution, we applied $1 \times 1 \times 1$ 3D convolution to generate the class logits for each voxel.

Ikeda et al. [2]: The architecture and training procedure strictly follow the official implementation. However, because this model divides the spatial dimensions into $1 6 \times 1 6$ patches, the input spatial resolution must be divisible by 16. Therefore, we adjusted the input size from $( H \times \bar { W } \times \bar { T } ) =$ $( 1 6 8 \times 2 0 0 \times 3 0 0 )$ used in our method to $( H \times W \times \bar { T } ) = ( 1 7 6 \times 2 0 8 \times 3 0 0 )$ for evaluation.

DynamicViT [4]: We extended this 2D image-based method to accommodate 3D FWL data. To align the conditions with our proposed pruning strategy, we configured the model to perform token selection at the input stage and set the keeping ratio to 0.3. This ensures that 70% of the input tokens are pruned, matching our pruning ratio.

ToMe [1]: Similar to DynamicViT, we extended this method for 3D data. The original algorithm alternately assigns tokens as dst tokens and src tokens on a 2D grid; we adapted this mechanism to alternate across a 3D token grid. This structural constraint limits the maximum merge ratio to 0.5. Because our proposed method utilizes a merge ratio of 0.9, we set the merge ratio in ToMe to its maximum allowable value of 0.5 to approximate our experimental conditions. To align with our setup, we applied the merging process to the attention layers. Additionally, mirroring our merging strategy, we incorporated a step to copy the features of the dst tokens back to the merged src tokens after the attention computation to maintain consistency with the residual connections.

## 2 FWL Generalization Benchmark

To evaluate generalization to unseen scenes acquired with the same sensor platform, we construct a new FWL test set distinct from Ghost-FWL [2]. Data were collected using the same FWL sensor as Ghost-FWL, ensuring that any performance difference reflects scene distribution rather than sensor variation. The set consists of 500 frames collected from 10 viewpoints (50 frames each), annotated following the procedure of Ikeda et al. [2]. All scenes were captured outdoors during daytime at a seven-story building clad in glass on all four sides, with additional glass installations around elevators and connecting corridors, totaling over 300 glass windows. Such a glass-dense environment does not appear in Ghost-FWL [2], making this test set a challenging out-of-distribution benchmark (Figure 1).

## 3 Additional Experiments and Results

## 3.1 Statistical Significance of FWL-Based Comparison

Table 1 presents the comparison with existing FWL-based methods under different random seeds. To avoid drawing conclusions from a single trial, we evaluated each method over 3 independent runs and report the mean and standard deviation of F1-mean. This evaluation captures seed-dependent variability and demonstrates that our method achieves superior performance robustly, rather than only under a particular random seed.

The results on the OOD test set further suggest the applicability of FWLAug to unseen data. While Ours fails to detect glass regions located behind other glass surfaces, Ours + FWLAug partially detects such regions, albeit with some inaccuracies. This indicates that FWLAug improves robustness to configurations not observed in the training data.

## 3.2 Comparison with Top-K Peak Extraction

Conventional multi-return ToF LiDAR keeps only the Top-K highest-intensity peaks of each ray. In contrast, FWL-ToPM ranks 3D patches by their maximum intensity over the entire frame, removes a fixed proportion $( \rho _ { \mathrm { l o w } } = 0 . 7 )$ of low-ranked patches, and retains the complete temporal waveform within each remaining patch. Figure 2 shows waveforms along single rays from two Ghost-FWL scenes. In Scene 001, the Top-3 peaks consist of a Glass return and two Ghost returns, so the Object return around depth bin 60 is discarded. In Scene 007, two Object returns and a Ghost return occupy the Top-3 peaks, so the Glass return around depth bin 90 is discarded. Since the Top-2 peaks are a subset of the Top-3 peaks, these returns are also discarded by Top-2 extraction. In both cases, the discarded returns lie well above the pruning threshold and are therefore retained by our pruning. These examples illustrate that a fixed per-ray peak budget can drop important Object and Glass returns when strong Ghost returns are present along the same ray, whereas our patch-level pruning preserves them together with their full waveforms.

![](images/f0cfe7c9ebab9e52b7d64617580039ec05f5919e66a855bdcfe042004b63d423.jpg)  
Figure 1: Our OOD Test Set. Based on the dense 3D maps as shown in Scene, we annotated FWL data with 4 labels: Ghost (red), Object (green), Glass (blue), Noise. Blue crosses indicate the LiDAR positions during data acquisition. The RGB images show the scenery of the capture locations.

Table 1: Statistical significance of the main comparison with FWL-based methods. F1-mean is reported as mean ± standard deviation over 3 independent runs with different random seeds. All methods are trained without FWLAug.
<table><tr><td>Method</td><td>F1-mean</td><td>Object-F1</td><td>Glass-F1</td><td>Ghost-F1</td><td>FPS</td></tr><tr><td>Scheuble et al. [5]</td><td> $\overline { { 0 . 4 9 7 \pm 0 . 0 4 3 } }$ </td><td>0.648</td><td>0.304</td><td>0.541</td><td>1.97</td></tr><tr><td>Ikeda et al. [2]</td><td> $0 . 5 0 3 \pm 0 . 0 0 5$ </td><td>0.680</td><td>0.268</td><td>0.561</td><td>6.54</td></tr><tr><td>Ours</td><td> $\mathbf { 0 . 5 9 2 \pm 0 . 0 0 5 }$ </td><td>0.741</td><td>0.358</td><td>0.678</td><td>53.2</td></tr></table>

## 3.3 Ablation on Space-Filling Curves

In FWL-ToPM, the tokens remaining after pruning are serialized along a space-filling curve before dst token selection, and our experiments adopt the Morton order as this curve. To examine whether the choice of curve affects performance, we replace the Morton order with the Hilbert curve [? ] while keeping all other settings unchanged, and evaluate both the merging-only and the pruning + merging settings. As shown in Table 2, the two curves yield nearly identical F1-mean scores in both settings, with the Morton order slightly higher: 0.565 versus 0.564 for merging only, and 0.592 versus

![](images/a8b48522e29640a50fc770c7c17d39f51474f571dbde7f33925347dc2ab7f035.jpg)  
Figure 2: Waveforms along single rays in Ghost-FWL Scene 001 (left) and Scene 007 (right). Markers indicate annotated Object (green), Glass (blue), and Ghost (red) bins, labels indicate the Top-3 highest-intensity peaks of each ray, and the dashed line indicates the intensity threshold below which patches are pruned at $\rho _ { \mathrm { l o w } } = 0 . 7$ . The photographs show the corresponding scenes, with the red marker indicating the observed ray. $\mathrm { T o p } { \bar { - } } 3$ peak extraction discards the Object and the Glass return, whereas our intensity-based pruning retains both because they lie above the pruning threshold.

Table 2: Comparison of space-filling curves used to serialize tokens for merging. F1 scores on the Ghost-FWL test set are reported. We set $\rho _ { \mathrm { l o w } } = 0 . 7$ (pruning + merging only), $\rho _ { m } = 0 . 9$ , and $\rho _ { d } = 0 . 1$ . All models are trained without FWLAug.
<table><tr><td>Setting</td><td>Curve</td><td>Object-F1</td><td>Glass-F1</td><td>Ghost-F1</td><td>F1-mean</td></tr><tr><td>Merging</td><td>Hilbert Morton (Ours)</td><td>0.690 0.702</td><td>0.349 0.369</td><td>0.652 0.624</td><td>0.564 0.565</td></tr><tr><td>Pruning + Merging</td><td>Hilbert Morton (Ours)</td><td>0.740 0.741</td><td>0.355 0.358</td><td>0.672 0.678</td><td>0.589 0.592</td></tr></table>

0.589 for pruning + merging. This indicates that the benefit comes from serializing the sparse tokens in an order that preserves spatial proximity, rather than from a particular choice of curve.

## 3.4 Qualitative Evaluation

Figure 3 shows a qualitative comparison of the classification results with existing FWL-based recognition methods. We evaluate the methods on scenes from the Ghost-FWL test split and on additional scenes not included in Ghost-FWL [2]. Across these scenes, Ours classifies Glass and Ghost regions more accurately than the existing methods. On the additional test scene, Ours also improves the classification of Objects located behind Glass regions, although some Glass regions tend to be partially confused with Object.

Nevertheless, failure cases remain. In particular, Glass regions located behind other Glass surfaces are still difficult to classify accurately. We consider this to be partly due to the limited number of such configurations in Ghost-FWL [2]. By increasing the occurrence of Glass-containing regions during training, FWLAug provides additional waveform and spatial variations associated with Glass. As a result, Ours+FWLAug shows partial improvements in detecting Glass regions in such challenging configurations, although the predictions remain imperfect. This suggests that FWLAug can improve robustness to glass configurations that are under-represented in the original training data.

![](images/835a6437dfc2490f86d1fc5b55bd526f358712c1c2b28a6523992c6b82b93094.jpg)  
Figure 3: Qualitative comparison of classification results with existing FWL-based recognition methods.

## References

[1] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token Merging: Your ViT But Faster. In International Conference on Learning Representations (ICLR), 2023.

[2] Kazuma Ikeda, Ryosei Hara, Rokuto Nagata, Ozora Sako, Zihao Ding, Takahiro Kado, Ibuki Fujioka, Taro Beppu, Mariko Isogawa, and Kentaro Yoshioka. Ghost-FWL: A Large-Scale Full-Waveform LiDAR Dataset for Ghost Detection and Removal. arXiv preprint arXiv:2603.28224, 2026.

[3] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollar. Focal Loss for Dense Object Detection. In IEEE/CVF International Conference on Computer Vision (ICCV), pages 2999–3007, 2017.

[4] Yongming Rao, Wenliang Zhao, Benlin Liu, Jiwen Lu, Jie Zhou, and Cho-Jui Hsieh. DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification. In Conference on Neural Information Processing Systems (NeurIPS), 2021.

[5] Dominik Scheuble, Hanno Holzhüter, Steven Peters, Mario Bijelic, and Felix Heide. Lidar Waveforms are Worth 40x128x33 Words. In IEEE/CVF International Conference on Computer Vision (ICCV), pages 28913–28924, 2025.