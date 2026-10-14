# Multimodal Fusion of RGB and Complementary Modalities for Semantic Segmentation

Hu Cao<sup>1</sup>, Yifeng Cheng<sup>2</sup>, Mu He<sup>2</sup>, Mengyu Li<sup>2</sup>, Yinlong Liu<sup>2</sup>, Alois Knoll<sup>2</sup>

<sup>1</sup>School of Automation, Southeast University, Sipailou 2, Nanjing, 210096, China. <sup>2</sup>Chair of Robotics, Artificial Intelligence and Real-time Systems, Technical University of Munich, Boltzmannstraße 15, Munich, 85748, Germany.

## Abstract

Multi-modal semantic segmentation adds an auxiliary sensing stream to RGB images (RGB-X), such as thermal, LiDAR, depth, event, polarization, or light field. This can improve robustness in poor lighting and under motion blur. But diferent sensors often produce features that do not match well, and fusion can become unstable. To alleviate these issues, we propose a bidirectional polarity-aware cross-modality fusion (BPCF) module that efectively captures complementary cues while enhancing feature alignment. We evaluate the framework on six modality pairings: RGB-Thermal, RGB-LiDAR, RGB-Depth, RGB-Event, RGB-Polarization, and RGB-Light Field. Our model achieve state-of-the-art (SOTA) results on eight public datasets, including MFNet, KITTI-360, DELIVER, DDD17, DSEC, MCubeS, ZJU, and UrbanLF. The codes and trained models will be released upon acceptance.

Keywords: Semantic Segmentation, Multi-modal Fusion, Polarity-aware Cross-attention, Cross-modal Feature Alignment

## 1 Introduction

In autonomous driving and robotic perception, RGB-only semantic segmentation often sufers from performance degradation under low illumination, fast motion, and adverse weather conditions [1, 2]. Fortunately, non-RGB modalities provide complementary cues that can mitigate these limitations. For instance, incorporating thermal, LiDAR, depth, event, light field, and polarization modality data can enhance robustness against low-light conditions, motion blur, and adverse weather conditions [1, 3]. Nevertheless, achieving reliable multi-modal fusion that scales to many sensors and consistently leverages each modality’s strengths remains an open challenge.

The problem can be considered from two perspectives. The first is the eficient utilization of comple mentary information. In multi-modal scenarios such as RGB–Thermal or RGB–Event, the same object may exhibit reversed attention responses across modalities. Consequently, cross-modal interactions in RGB-X semantic segmentation are not purely positively correlated. Depending on the sensor physics and scene conditions, the same semantic structure can elicit aligned responses across modalities or reversed responses. Importantly, these negative correlations are not necessarily noise. Our key observation is that these contrastive responses can be indicative: they can reveal complementary cues that are hard to capture while the fusion mechanism assumes only non-negative similarity. Recognizing that such negative correlations can also provide valuable cues, we propose a bidirectional polarity-aware linear cross-attention (BPLCA) mechanism. BPLCA decomposes features into positive and negative components and performs cross-modal attention across both branches. This design enables the model to capture not only shared but also complementary signals across modalities, leading to more complete and balanced feature integration.

A second challenge in RGB-X semantic segmentation is cross-modal feature alignment. Heterogeneous modalities (e.g., RGB, thermal, LiDAR, depth, events, and polarization) difer substantially in their statistics and appearance, making naive fusion prone to inconsistent representations or over-reliance on a single modality [2, 4, 5]. To address this, we introduce a Dual Feature Consistency Constraint (DFCC) that explicitly encourages cross-modal features to become semantically coherent, thereby narrowing the modality gap and stabilizing fusion. DFCC aligns RGB and auxiliary features by first reducing distribution mismatch, and then applying scene-adaptive afine modulation whose parameters are predicted from fused features. During fusion, it further regularizes cross-modal semantics by constructing spatial attention maps at each stage and penalizing their discrepancies, encouraging consistent attention and representations while retaining complementary cues.

With BPLCA and DFCC integrated into our framework (together with the proposed designs above), we

![](images/f9f6346b072a68c84aabc8f9be49ee349e289b1fd4ad50a9206cc99a3dad08ff.jpg)  
Fig. 1 Performance comparison between our proposed method (Ours) and CMNeXt [2] across diferent modality combinations, including RGB-Thermal (MFNet), RGB-LiDAR (KITTI360), RGB-Depth (DELIVER), RGB-Event (DSEC and DDD17), RGB-Polarization (MCubeS), and RGB-Light Field (UrbanLF-Real and UrbanLF-Syn).

achieve consistent improvements and outperform recent SOTA methods on eight multi-modal datasets (Fig. 1). The main contributions of this work are as follows:

• We propose a multi-modal semantic segmentation framework that supports flexible modality combinations. It enables efective representation learning across diverse RGB-X configurations, including RGB-Thermal, RGB-LiDAR, RGB-Depth, RGB-Event, RGB-Polarization, and RGB-Light Field.

• We develop a bidirectional polarity-aware cross-modality fusion (BPCF) module, which integrates bidirectional polarity-aware linear cross-attention (BPLCA) with a dual feature consistency constraint (DFCC) to efectively fuse heterogeneous modalities.

• We introduce a stage-wise constraint loss that progressively enforces consistent cross-modal feature alignment.

• We conduct extensive experiments on eight multi-modal datasets covering six auxiliary modalities. The results show that our method consistently outperforms SOTA approaches across all datasets.

## 2 Related Work

## 2.1 Semantic segmentation

Semantic segmentation has progressed from CNN-based architectures such as FCN [6], U-Net [7], and SegNet [8], to encoder–decoder designs like DeepLabv3+ [9]. More recently, Transformer architectures have become dominant, with models including ViT [10], Swin Transformer [11], SegFormer [12], SDPT [13], serving as fundamental baselines for RGB-only dense prediction. Despite this progress, performance degrades under motion blur, low light, or rain/snow. This motivates multi-modal RGB+X architectures for robust scene understanding, where X denotes auxiliary modalities such as thermal, LiDAR, depth, event cameras, polarization, and light field.

## 2.2 Multi-modal semantic segmentation

Prior work recognizes the complementary value of non-RGB modalities for semantic segmentation. It explores fusion methods from pairwise RGB–X integration to modality-agnostic frameworks that unify multiple sensor types. For RGB+event combination, [14–19] integrate events over time and use eventcount or activity-rate cues, then use dual-branch backbones together with bidirectional cross-attention, to synchronize features across modalities and scales. Several studies also evaluate fusion with auxiliary modalities beyond events. Their results indicate that non-RGB modalities provide powerful complementary cues for scene understanding [20–22]. KTBNet [23] proposes a parameter-eficient symmetric framework that balances the contributions between RGB and an additional modality such as thermal, thereby preventing the dominance of a single branch. Moreover, better-known works on RGB–P segmentation has shown only limited performance with these dual branch architecture [2, 24]. A possible reason is that polarization cues can be inconsistent with RGB appearance, so cross-modal evidence is not always positively correlated and may even be negatively correlated. Recently, PolaFormer [25] introduces a learnable mechanism with power-based scaling to explicitly model both positive and negative query-key interactions. PolaFormer targets single-modality linear self-attention, so its polarity design mainly repairs attention expressiveness by recovering missing negative interactions. In contrast, our $\mathrm { B P L C A / P L C A }$ targets cross-modality fusion, so its polarity design mainly structures inter-modal evidence by separating shared cues from complementary cues to improve fusion. Nevertheless, the above designs lack modality agnostic generalization, as they are tied to fixed modality combinations. CMX [4] employs cross-modal feature rectification and fusion modules for long-range context exchange across modalities. Its successor CMNeXt [2] extends scalability through a self-query hub and a parallel pooling mixer, maintaining compactness while adapting to diverse sensor types. Similarly, Zheng et al. [26] propose a modalityagnostic pipeline that dynamically selects the most informative modalities, while their follow-up [18] introduces a modality-agnostic Feature Fusion (MFF) module that synthesizes heterogeneous sensor streams into a unified representation. GeminiFusion [27] further refines cross-modal interactions using pixel-wise intra- and inter-modal attention, and OmniVec2 [28] explores a large-scale shared representation space for multi-modal and multi-task learning. Although these works have advanced the field, challenges still remain in fully exploiting complementary information across modalities and in achieving consistent feature alignment. In this work, we propose a bidirectional polarity-aware cross-modality fusion (BPCF) module to alleviate these issues.

![](images/7dc0f7e235653ae9dc80a3dbfb965c28b80ef9d2f91c240ba4bef25b5c8859c7.jpg)  
Fig. 2 Overview of the proposed multi-modal semantic segmentation framework. Each modality is independently encoded using a dedicated Transformer backbone. RGB and auxiliary modality features are then progressively fused through the proposed fusion module in a stage-by-stage manner. Finally, the fused multi-scale features are passed to the segmentation head to generate predictions.

## 3 Method

As shown in Fig. 2, our model adopts an encoder–decoder architecture with stage-wise fusion modules. Each modality is first encoded by a dedicated Transformer backbone [12]. Following prior work [2, 4], we incorporate ScoreNet to flexibly integrate RGB with one or more auxiliary modalities. For multiple auxiliary inputs, ScoreNet selects the most informative feature at each spatial location. The stage-wise BPCF fusion modules then combine the hierarchical feature representations from the two modality branches. Finally, the fused multi-scale features are aggregated by an MLP-based segmentation head to produce final predictions.

## 3.1 Feature Extraction

## 3.1.1 ScoreNet

To extract the most informative modality features, we employ ScoreNet with a dynamic selection mechanism. Specifically, ScoreNet processes each patch in each auxiliary modality as follows:

$$
S _ { i , j } = \mathrm { S o f t m a x } ( \operatorname { L i n e a r } ( \operatorname { G E L U } ( \operatorname { L i n e a r } ( \operatorname { N o r m } ( X _ { i , j } ) ) ) ) ) ,\tag{1}
$$

for $i \in [ 1 , M ]$ and $j \in [ 1 , N ]$ , where $S _ { i , j }$ denotes the informative score for the i-th input modality on the j-th patch token, M refers to the number of modalities and $N = H \times W$ refers to the number of patches. For each patch token, the embedding from the modality with the highest score is dynamically selected and fused into a single auxiliary input to complement the primary RGB data:

$$
\boldsymbol { x } _ { \mathrm { a } } ^ { ( 0 ) } = \{ X _ { m , j } | j \in H \times W , m = \arg \operatorname* { m a x } _ { i } ( S _ { i , j } ) \} .\tag{2}
$$

## 3.1.2 Backbone

Subsequently, the primary RGB data and the dynamically selected auxiliary modality are encoded by parallel Transformer backbones (MiT) [12]. The extracted multi-level features are then passed to the proposed stage-wise fusion modules. The process can be formulated as follows:

$$
x _ { \mathrm { r } } = \mathrm { M i T } ( x _ { \mathrm { r } } ^ { ( 0 ) } ) , x _ { \mathrm { a } } = \mathrm { M i T } ( x _ { \mathrm { a } } ^ { ( 0 ) } ) .\tag{3}
$$

## 3.2 Bidirectional Polarity-Aware Cross-Modality Fusion Module

To efectively integrate heterogeneous cues from RGB and auxiliary modalities, we introduce a bidirectional polarity-aware cross-modality fusion (BPCF) module, as illustrated in Fig. 3. The BPCF is composed of two key parts: bidirectional polarity-aware linear crossattention (BPLCA) and dual feature consistency constraint (DFCC). The BPLCA facilitates comprehensive cross-modal interaction by employing symmetric crossgating and polarity-aware cross-linear attention. Meanwhile, the DFCC enhances feature alignment by exploiting statistical correlations and applying the consistency constraint loss $\mathcal { L } _ { \mathrm { s t a g e } } ,$ complemented by a refinement block to generate the final fused output.

![](images/9f6cbdf4ec463a828e6fa014d82ff29969c6809d8c457254365b01db0289f816.jpg)  
Fig. 3 The overall structure of the BPCF module. It consists of a bidirectional polarity-aware linear crossattention (BPLCA) part and a dual feature consistency constraint (DFCC) part.

## 3.2.1 Bidirectional polarity-aware linear cross-attention (BPLCA)

To comprehensively fuse complementary cues from the RGB stream and the auxiliary modality stream, our BPLCA is designed with dual cross-modal fusion branches. First, an eficient cross-gating mechanism is applied to activate relevant features. Second, a symmetric polarity-aware linear cross-attention (PLCA) is introduced (see Fig. 4), which preserves informative negative evidence while maintaining the lineartime and linear-memory eficiency of kernelized attention. Formally, let $\boldsymbol { x } _ { \mathrm { r } } , \boldsymbol { x } _ { \mathrm { a } } \in \mathbb { R } ^ { N \times C }$ represent the extracted features from the RGB and auxiliary branches, reshaped into sequences of $N { = } H W$ tokens with C channels. Two 1×1 convolutions are then used to perform per-location linear projections:

$$
\begin{array} { r } { x _ { \mathrm { r } } ^ { \mathrm { p r o j } } = \mathrm { C o n v } _ { 1 \times 1 } ( x _ { \mathrm { r } } ) , x _ { \mathrm { a } } ^ { \mathrm { p r o j } } = \mathrm { C o n v } _ { 1 \times 1 } ( x _ { \mathrm { a } } ) . } \end{array}\tag{4}
$$

We then realize bidirectional interaction via symmetric cross-gating using the element-wise (Hadamard) product ◦:

$$
\begin{array} { r l } & { x _ { \mathrm { r } } ^ { \prime } = x _ { \mathrm { r } } ^ { \mathrm { p r o j } } + ( x _ { \mathrm { r } } ^ { \mathrm { p r o j } } \circ x _ { \mathrm { a } } ^ { \mathrm { p r o j } } ) , } \\ & { x _ { \mathrm { a } } ^ { \prime } = x _ { \mathrm { a } } ^ { \mathrm { p r o j } } + ( x _ { \mathrm { r } } ^ { \mathrm { p r o j } } \circ x _ { \mathrm { a } } ^ { \mathrm { p r o j } } ) . } \end{array}\tag{5}
$$

To further enhance cross-modal interaction, the fused features are passed into the PLCA, as illustrated in Fig. 4. Within PLCA, the input features are projected into four distinct matrices: query (Q), key (K), value (V ), and gating (G). Among these, the gating matrix G plays a pivotal role in realizing polarity-aware attention [25], whose mechanism will be detailed in the following section.

$$
\begin{array} { r l r } & { } & { Q _ { \mathrm { r } } = x _ { \mathrm { r } } ^ { \prime } q _ { \mathrm { r } } , K _ { \mathrm { r } } = x _ { \mathrm { r } } ^ { \prime } k _ { \mathrm { r } } , V _ { \mathrm { r } } = x _ { \mathrm { r } } ^ { \prime } v _ { \mathrm { r } } , G _ { \mathrm { r } } = x _ { \mathrm { r } } ^ { \prime } g _ { \mathrm { r } } ; } \\ & { } & { Q _ { \mathrm { a } } = x _ { \mathrm { a } } ^ { \prime } q _ { \mathrm { a } } , K _ { \mathrm { a } } = x _ { \mathrm { a } } ^ { \prime } k _ { \mathrm { a } } , V _ { \mathrm { a } } = x _ { \mathrm { a } } ^ { \prime } v _ { \mathrm { a } } , G _ { \mathrm { a } } = x _ { \mathrm { a } } ^ { \prime } g _ { \mathrm { a } } . } \end{array}\tag{6}
$$

where $q _ { \mathrm { r } } , k _ { \mathrm { r } } , v _ { \mathrm { r } } , g _ { \mathrm { r } } , q _ { \mathrm { a } } , k _ { \mathrm { a } } , v _ { \mathrm { a } } , g _ { \mathrm { a } }$ denote the learnable linear projections for the RGB and auxiliary branches, respectively. Following the kernelization strategy in [29], vanilla attention can be reformulated using a similarity function defined by a feature map $\phi ( \cdot )$ . The i-th row of the attention result is formulated as:

$$
{ \mathrm { A t t e n t i o n } } ( Q , K , V ) _ { i } = { \frac { \sum _ { j } \sin ( Q _ { i } , K _ { j } ) V _ { j } } { \sum _ { j } \sin ( Q _ { i } , K _ { j } ) } } ,\tag{7}
$$

$$
\mathrm { s i m } ( x , y ) \stackrel { } { = } \kappa ( x , y ) = \phi ( x ) ^ { T } \phi ( y ) .\tag{8}
$$

This kernelization renders the similarity function decomposable. By exploiting distributive and associative properties, the final attention computation becomes:

$$
\operatorname { A t t e n t i o n } ( Q , K , V ) = { \frac { \phi ( Q ) ^ { T } \Bigl ( \sum _ { j } \phi ( K _ { j } ) V _ { j } \Bigr ) } { \phi ( Q ) ^ { T } \Bigl ( \sum _ { j } \phi ( K _ { j } ) \Bigr ) } } .\tag{9}
$$

Notably, Eq. 9 avoids the $\mathcal { O } ( N ^ { 2 } D )$ pairwise computations $Q _ { i } ^ { \top } K _ { j }$ by precomputing the terms $\textstyle \sum _ { j } \phi ( K _ { j } ) V _ { j }$ and $\textstyle \sum _ { j } \phi ( K _ { j } )$ , thereby reducing the overall time complexity to $\mathcal { O } \big ( N D ^ { 2 } \big )$ . Standard non-negative feature maps like ReLU discard negative information. This information is retained by decomposing queries and keys into positive $( Q ^ { + } )$ and negative $( Q ^ { - } )$ parts via $Q = Q ^ { + } - Q ^ { - }$ where $Q ^ { + } = \operatorname { R e L U } ( Q )$ and $Q ^ { - } = \mathrm { R e L U } ( - Q )$ . The same decomposition is applied to the key matrix K. Then the inner product can be decomposed into:

$$
\begin{array} { c } { { \langle q _ { i } , k _ { j } \rangle = \langle q _ { i } ^ { + } , k _ { j } ^ { + } \rangle + \langle q _ { i } ^ { - } , k _ { j } ^ { - } \rangle } } \\ { { - \langle q _ { i } ^ { - } , k _ { j } ^ { + } \rangle - \langle q _ { i } ^ { + } , k _ { j } ^ { - } \rangle . } } \end{array}\tag{10}
$$

Then the linear-decomposable similarity function in Eq. 8 can be derived as

![](images/4787905f71ddce98787a32676a1d4b1e6d6a91afc21c716539520ee8f03e3d13.jpg)

$$
\begin{array} { c } { \sin ( x , y ) = \phi ( { q _ { i } } ) ^ { T } \phi ( k _ { j } ) } \\ { = ( ( \phi ( q _ { i } ) ^ { + } ) ^ { T } \phi ( k _ { j } ) ^ { + } } \\ { + ( \phi ( q _ { i } ) ^ { - } ) ^ { T } \phi ( k _ { j } ) ^ { - } ) } \\ { - ( ( \phi ( q _ { i } ) ^ { + } ) ^ { T } \phi ( k _ { j } ) ^ { - } } \\ { + ( \phi ( q _ { i } ) ^ { - } ) ^ { T } \phi ( k _ { j } ) ^ { + } ) . } \end{array}\tag{11}
$$

Fig. 4 Illustration of BPLCA. Features from RGB and auxiliary modality are deeply fused through the BPLCA component.

Thus, we can accept a non-positive feature map $\phi$ in the kernel. To process these non-negative components, we employ a channel-wise learnable feature map, $\phi ( x )$ , defined as:

$$
\phi ( x ) = x ^ { P } , { \mathrm { w h e r e ~ } } P = 1 + \alpha \cdot { \mathrm { s i g m o i d } } ( W ) .\tag{12}
$$

where P is a dynamically computed exponent, determined by a hyperparameter α and a d-dimensional learnable parameter vector W, which learns to weigh the relative importance of diferent channels.

To explicitly model the interactions between polarities, we first concatenate the positive and negative components along the channel dimension (denoted by ∥), then capture the interactions between $Q$ and K for same-signed $( + + , -- )$ and opposite-signed $( + - , - + )$ pairs separately while ensuring all kernelized terms remain non-negative. The value (V) is partitioned accordingly with $V _ { \mathrm { r } } = V _ { \mathrm { r } } ^ { \mathrm { s } } \parallel V _ { \mathrm { r } } ^ { \mathrm { o } }$ and $V _ { \mathrm { a } } = V _ { \mathrm { a } } ^ { \mathrm { s } } \parallel V _ { \mathrm { a } } ^ { \mathrm { o } }$ while both split tensors have half of the channels from the original one. Based on these decompositions, we then compute the intermediate fusion results for the RGB and auxiliary modalities, distinguishing between same-signed and opposite-signed correlations, substituting Eq. 11 into Eq. 9:

$$
\begin{array} { r } { x _ { \mathrm { r } } ^ { \mathrm { s } } = \frac { \phi ( Q _ { \mathrm { a } } ^ { + } \parallel Q _ { \mathrm { a } } ^ { - } ) ^ { T } \sum _ { j } \phi ( K _ { \mathrm { r } } ^ { + } \parallel K _ { \mathrm { r } } ^ { - } ) _ { j } ( V _ { \mathrm { r } } ^ { \mathrm { s } } ) _ { j } } { \phi ( Q _ { \mathrm { a } } ^ { + } \parallel Q _ { \mathrm { a } } ^ { - } ) ^ { T } \sum _ { j } \phi ( K _ { \mathrm { r } } ^ { + } \parallel K _ { \mathrm { r } } ^ { - } ) _ { j } } , } \\ { x _ { \mathrm { r } } ^ { \mathrm { s } } = \frac { \phi ( Q _ { \mathrm { a } } ^ { - } \parallel Q _ { \mathrm { a } } ^ { + } ) ^ { T } \sum _ { j } \phi ( K _ { \mathrm { r } } ^ { + } \parallel K _ { \mathrm { r } } ^ { - } ) _ { j } ( V _ { \mathrm { r } } ^ { \mathrm { s } } ) _ { j } } { \phi ( Q _ { \mathrm { a } } ^ { - } \parallel Q _ { \mathrm { a } } ^ { + } ) ^ { T } \sum _ { j } \phi ( K _ { \mathrm { r } } ^ { + } \parallel K _ { \mathrm { r } } ^ { - } ) _ { j } } ; } \\ { x _ { \mathrm { a } } ^ { \mathrm { s } } = \frac { \phi ( Q _ { \mathrm { r } } ^ { + } \parallel Q _ { \mathrm { r } } ^ { - } ) ^ { T } \sum _ { j } \phi ( K _ { \mathrm { a } } ^ { + } \parallel K _ { \mathrm { a } } ^ { - } ) _ { j } ( V _ { \mathrm { a } } ^ { \mathrm { s } } ) _ { j } } { \phi ( Q _ { \mathrm { r } } ^ { + } \parallel Q _ { \mathrm { r } } ^ { - } ) ^ { T } \sum _ { j } \phi ( K _ { \mathrm { a } } ^ { + } \parallel K _ { \mathrm { a } } ^ { - } ) _ { j } } , } \\  x _ { \mathrm { a } } ^ { \mathrm { s } } = \frac  \phi ( Q _  \mathrm  r  \end{array}\tag{13}
$$

To capture sophisticated relations between same-signed and opposite-signed parts, we employ a learnable, element-wise mixing that weights the same-signed and opposite-signed streams via gating tensors, $G =$ $G ^ { \mathrm { s } } \parallel G ^ { \mathrm { o } }$ . The final polarity-aware outputs are then produced through gated fusion:

$$
\begin{array} { r } { x _ { \mathrm { r } } ^ { \prime \prime } = ( x _ { \mathrm { r } } ^ { \mathrm { s } } \circ G _ { \mathrm { a } } ^ { \mathrm { s } } ) \parallel ( x _ { \mathrm { r } } ^ { \mathrm { o } } \circ G _ { \mathrm { a } } ^ { \mathrm { o } } ) , } \\ { x _ { \mathrm { a } } ^ { \prime \prime } = ( x _ { \mathrm { a } } ^ { \mathrm { s } } \circ G _ { \mathrm { r } } ^ { \mathrm { s } } ) \parallel ( x _ { \mathrm { a } } ^ { \mathrm { o } } \circ G _ { \mathrm { r } } ^ { \mathrm { o } } ) . } \end{array}\tag{14}
$$

Other than the self attention for enhanced uni-modal image perception originated from PolaFormer [25], we emphasize the high-sensitivity cross-modal information aggregation of our polarity-aware cross attention. Especially, PLCA incorporates a bidirectional architecture where the Query (Q) and Gating (G) matrices of one branch are explicitly designed to interact with the Key (K) and Value (V ) of the other modality, both achieving linear complexity and capturing universal signals. This mechanism preserves the integrity of the original content alignment by mitigating excessive influence from the complementary modality, yet still allows for the infusion of cross-modal information.

## 3.2.2 Dual feature consistency constraint (DFCC)

To strengthen cross-modal feature alignment while maintaining consistency, we introduce the DFCC. As shown in Fig. 5, DFCC integrates a feature consistency constraint with a refinement block in a unified design, jointly enabling semantic alignment and feature enhancement. Specifically, given enhanced RGB and auxiliary features $x _ { \mathrm { r } } ^ { \prime } , x _ { \mathrm { a } } ^ { \prime } \in \mathbb { R } ^ { B \times C \times H \times W }$ and fused features $x _ { \mathrm { r } } ^ { \prime \prime } , x _ { \mathrm { a } } ^ { \prime \prime } \in$ $\mathbf { \mathbb { R } } ^ { \mathbf { \bar { B } } \times C \times H \times W }$ , we first compute feature statistics to align the feature spaces across both streams:

$$
\begin{array} { r } { \hat { x } _ { \mathrm { r } } = \displaystyle \frac { x _ { \mathrm { r } } ^ { \prime } - \mu ( x _ { \mathrm { r } } ^ { \prime } ) } { \sigma ( x _ { \mathrm { r } } ^ { \prime } ) } , } \\ { \hat { x } _ { \mathrm { a } } = \displaystyle \frac { x _ { \mathrm { a } } ^ { \prime } - \mu ( x _ { \mathrm { a } } ^ { \prime } ) } { \sigma ( x _ { \mathrm { a } } ^ { \prime } ) } ; } \end{array}\tag{15}
$$

where,

![](images/a0362df00a21a5dc1aeaff34fdacf732281d1da3d9d3ffd953735a9f65045b63.jpg)  
Fig. 5 Illustration of DFCC. The features from BPLCA are further refined to obtain the final fused output.

$$
\mu ( x ) = \frac { 1 } { H W } \sum _ { h = 1 } ^ { H } \sum _ { w = 1 } ^ { W } x , ~ \sigma ( x ) = \sqrt { \frac { 1 } { H W } \sum _ { h = 1 } ^ { H } \sum _ { w = 1 } ^ { W } \left( x - \mu ( x ) \right) ^ { 2 } } .\tag{16}
$$

where $\mu ( x )$ and $\sigma ( x )$ denote the channel-wise mean and standard deviation, respectively.

DFCC then learns adaptive afine modulation parameters from the fused features $x _ { \mathrm { r } } ^ { \prime \prime }$ and $x _ { \mathrm { a } } ^ { \prime \prime }$ . Specifically, the global average pooling operations are first applied:

$$
z _ { \mathrm { r } } = \mathrm { A v g P o o l } ( x _ { \mathrm { r } } ^ { \prime \prime } ) , z _ { \mathrm { a } } = \mathrm { A v g P o o l } ( x _ { \mathrm { a } } ^ { \prime \prime } ) .\tag{17}
$$

Subsequently, MLPs are employed to generate branch-specific modulation parameters:

$$
\begin{array} { r } { \gamma _ { \mathrm { r } } = \mathrm { M L P } ( z _ { \mathrm { r } } ) , \beta _ { \mathrm { r } } = \mathrm { M L P } ( z _ { \mathrm { r } } ) , } \\ { \gamma _ { \mathrm { a } } = \mathrm { M L P } ( z _ { \mathrm { a } } ) , \beta _ { \mathrm { a } } = \mathrm { M L P } ( z _ { \mathrm { a } } ) . } \end{array}\tag{18}
$$

The normalized features are then modulated as:

$$
y _ { \mathrm { r } } = \gamma _ { \mathrm { r } } \cdot \hat { x } _ { \mathrm { r } } + \beta _ { \mathrm { r } } , y _ { \mathrm { a } } = \gamma _ { \mathrm { a } } \cdot \hat { x } _ { \mathrm { a } } + \beta _ { \mathrm { a } } .\tag{19}
$$

Finally, the two streams are concatenated and fed into the refinement block, which comprises two parallel paths: $\mathrm { ~ a ~ } 1 \times 1$ convolutional shortcut that preserves identity information, and a nonlinear transformation branch that enhances feature fusion. The nonlinear branch sequentially applies a 1 × 1 convolution for channel reduction, a $3 \times 3$ depthwise convolution for spatial interaction, a ReLU activation, and another $1 \times 1$ convolution to restore the channel dimension, followed by normalization. The final output is then obtained by summing the two paths:

$$
y _ { \mathrm { f } } = { \mathrm { R e f i n e m e n t } } ( y _ { \mathrm { r } } \parallel y _ { \mathrm { a } } ) .\tag{20}
$$

Moreover, to further enhance cross-modal consistency, we incorporate a stage-wise constraint loss within DFCC. At each fusion stage, the features $y _ { \mathrm { r } }$ and $y _ { \mathrm { a } }$ are computed by averaging over the channel dimension:

$$
\begin{array} { c l } { \alpha _ { \mathrm { r } } = \mathrm { N o r m } ( \mathrm { M e a n } ( y _ { \mathrm { r } } , \mathrm { d i m } = 1 ) ) , } \\ { \alpha _ { \mathrm { a } } = \mathrm { N o r m } ( \mathrm { M e a n } ( y _ { \mathrm { a } } , \mathrm { d i m } = 1 ) ) . } \end{array}\tag{21}
$$

The $\begin{array} { r } { \mathrm { N o r m } ( a ) = \frac { a - \operatorname* { m i n } ( a ) } { \operatorname* { m a x } ( a ) - \operatorname* { m i n } ( a ) + \epsilon } } \end{array}$ here is applied pixel-wise. The stage-wise constraint loss is then defined as the mean squared error (MSE) between the normalized features:

$$
{ \mathcal { L } } _ { \mathrm { s t a g e } } = \| \alpha _ { \mathrm { r } } - \alpha _ { \mathrm { a } } \| _ { 2 } ^ { 2 } .\tag{22}
$$

This objective encourages both modalities to produce consistent responses, thereby improving feature alignment.

## 3.3 Loss Function

Our proposed model is trained using a combination of segmentation and alignment objectives. For the primary segmentation task, we employ the standard pixel-wise cross-entropy loss, commonly used in dense prediction tasks:

$$
\mathcal { L } _ { \mathrm { C E } } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \log \frac { \exp ( p _ { i , y _ { i } } ) } { \sum _ { j = 1 } ^ { C } \exp ( p _ { i , j } ) } .\tag{23}
$$

Table 1 Summary of multi-modal datasets used in our experiments.
<table><tr><td>Dataset</td><td>Modality</td><td>Classes</td><td>Train Test</td><td>Resolution</td></tr><tr><td>MFNet [30]</td><td>RGB-Thermal</td><td>8</td><td>784  /  392</td><td> $4 8 0 \times 6 4 0$ </td></tr><tr><td>KITTI-360 [31]</td><td>RGB-LiDAR</td><td>19</td><td>49004 / 12276</td><td> $1 4 0 8 \times 3 7 6$ </td></tr><tr><td>DELIVER [2]</td><td>RGB-Depth-Event-LiDAR</td><td>25</td><td>3893 /1897</td><td> $1 0 2 4 \times 1 0 2 4$ </td></tr><tr><td>DDD17 [32]</td><td>RGB-Event</td><td>5</td><td>15950 / 3890</td><td> $2 0 0 \times 3 4 6$ </td></tr><tr><td>DSEC [33]</td><td>RGB-Event</td><td>11</td><td>8082 /2809</td><td> $4 4 0 \times 6 4 0$ </td></tr><tr><td>MCubeS [24]</td><td> $\mathrm { R G B - A o L P - D o L P - N I R }$ </td><td>20</td><td>302  /  102</td><td> $1 2 2 4 \times 1 0 2 4$ </td></tr><tr><td>ZJU [35]</td><td>RGB-Polarization</td><td>8</td><td>344 / 50</td><td> $1 2 2 4 \times 1 0 2 4$ </td></tr><tr><td>UrbanLF [34]</td><td>RGB-Light Field</td><td>14</td><td>580 / 80 (Real), 172 / 28 (Syn) 623 × 432, 640 × 480</td><td></td></tr></table>

where $N = H \times W$ denotes the number of valid pixels, $C$ is the number of classes, $p _ { i , j }$ represents the logit for class j at pixel $i ,$ and $y _ { i }$ is the corresponding ground-truth label. The overall training objective combines the segmentation loss with the stage-wise constraint loss as follows:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \lambda _ { 1 } \cdot \mathcal { L } _ { \mathrm { C E } } + \lambda _ { 2 } \cdot \sum _ { i = 1 } ^ { 4 } \mathcal { L } _ { \mathrm { s t a g e } } ^ { ( i ) } .\tag{24}
$$

where $\lambda _ { 1 }$ and $\lambda _ { 2 }$ balance the segmentation and stage-wise constraint losses. In our experiments, we set $\lambda _ { 1 } = 1$ and $\lambda _ { 2 } = 0 . 1$

## 4 Experiments

## 4.1 Experimental Setup

## 4.1.1 Datasets

We evaluate our approach on eight widely used multi-modal datasets: MFNet [30], KITTI-360 [31], DDD17 [32], DSEC [33], DELIVER [2], UrbanLF [34], MCubeS [24], and ZJU [35]. An overview of the datasets is provided in Tab. 1; additional dataset details and common configuration parameters are provided in Sec. 4.1.1 and Sec. 4.1.2.

An overview of the datasets used in our experiments is provided in Tab. 1. The details are introduced below:

RGB-T MFNet [30] is a driving-scene dataset with 1,569 aligned RGB-thermal pairs across 8 semantic classes. It provides 784/392/392 images for training, validation, and testing, respectively, covering both daytime and nighttime conditions.

RGB-L KITTI-360 [31] is a suburban driving dataset, providing 49,004 training and 12,276 validation images with a resolution of 1408 × 376. Following the Cityscapes dataset [36], it defines 19 semantic classes.

RGB-X DELIVER [2] is a large-scale synthetic benchmark built in the CARLA simulator, supporting multi-modal segmentation across RGB, depth, event, and LiDAR streams. It contains over 47k annotated frames from six camera views under diverse conditions, including challenging weather (fog, rain, night) and five sensor degradation types (e.g., motion blur, exposure imbalance, LiDAR jitter). The dataset defines 25 semantic classes covering urban elements such as vehicles, roads, pedestrians, and vegetation.

RGB-E DDD17 [32] includes over 12 hours of driving data collected with a DAVIS346B sensor, providing event streams and grayscale images at $2 0 0 \times 3 4 6$ resolution. We generate pseudo-labels using EV-SegNet [37], resulting in 5 semantic classes and splits of 15,950/3,890 samples.

RGB-E DSEC [33] contains over 10k RGB-event frames with 11 categories, captured from stereo cameras in urban and rural environments. We follow [16] for preprocessing and dataset splits (8082/2809 for training/testing).

RGB-P MCubeS [24] consists of 500 samples with aligned RGB, NIR, and polarization cues (DoLP, AoLP), annotated with 20 material categories. The dataset is split into 302/96/102 for training, validation, and testing.

RGB-P ZJU [35] is a polarization semantic segmentation dataset collected in outdoor scenes. Each sample contains four polarized RGB images at $0 ^ { \circ } , 4 5 ^ { \circ } , 9 0 ^ { \circ }$ , and 135<sup>◦</sup>, from which AoLP and DoLP are derived using Stokes parameters. It provides 344 training and 50 validation samples across 8 semantic classes, with image resolution of 1224 × 1024.

RGB-LF UrbanLF [34] is a light-field dataset for urban-scene segmentation, including both real (580/80/164) and synthetic (172/28/50) subsets, each with 14 semantic classes. Central-view annotations are used for supervision.

## 4.1.2 Implementation Details

Table 2 Dataset-specific training configuration parameters.
<table><tr><td>Dataset</td><td></td><td></td><td></td><td></td><td>Shrink Shrink Scale Backbone Learning Rate Training Epochs</td></tr><tr><td>MFNet [30]</td><td>True</td><td>0.9</td><td>mit_b4</td><td>6e-5</td><td>500</td></tr><tr><td>KITTI-360 [31]</td><td>False</td><td></td><td>mit_b2</td><td>6e-5</td><td>60</td></tr><tr><td>DELIVER [2]</td><td>True</td><td>0.5</td><td>mit_b2</td><td>6e-5</td><td>60</td></tr><tr><td>DDD17 [32]</td><td>False</td><td>1</td><td>mit_b4</td><td>2e-4</td><td>40</td></tr><tr><td>DSEC [33]</td><td>False</td><td></td><td>mit_b4</td><td>6e-5</td><td>60</td></tr><tr><td>MCubeS [24]</td><td>False</td><td></td><td>mit_b4</td><td>6e-5</td><td>500</td></tr><tr><td>ZJU [35]</td><td>False</td><td></td><td>mit_b4</td><td>6e-5</td><td>500</td></tr><tr><td>UrbanLF [34]</td><td>False</td><td></td><td>mit_b4</td><td>6e-5</td><td>500</td></tr></table>

Table 3 Common configuration hyperparameters.

Except for the high-resolution DELIVER experiments in Sec. ??, models are trained on a single NVIDIA RTX 3090 GPU with a batch size of 2. We adopt AdamW [38] as the optimizer, with an initial learning rate ranging from $6 \times 1 0 ^ { - 5 } \mathrm { ~ t o ~ 2 ~ } \times 1 0 ^ { - 4 }$ depending on the dataset, momentum of 0.9, weight decay of 0.01, and epsilon of $1 \times 1 0 ^ { - 8 }$ . The learning rate is scheduled using polynomial decay with a power of 0.9, preceded by a 10-epoch linear warm-up. Input images are resized to dataset-specific resolutions, and standard data augmentations are applied, including random resizing within [0.5, 2.0], horizontal flipping, color jitter, Gaussian blur, and random cropping. Following prior works such as CMX [4] and CMNeXt [2], we adopt dataset-specific training configurations to ensure fair comparisons, where backbone variants (e.g., MiT-B2 or MiT-B4), learning rates, and training epochs are selected according to commonly used settings for each dataset. This practice is widely adopted in RGB-X semantic segmentation due to diferences in dataset scale, resolution, and modality characteristics. More details can be found in Tab. 2 and Tab. 3.

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Background</td><td>255</td></tr><tr><td>Learning Rate (LR) Power 0.9</td><td></td></tr><tr><td>Momentum</td><td>0.9</td></tr><tr><td>Weight Decay</td><td>0.01</td></tr><tr><td>Batch Size</td><td>2</td></tr><tr><td>Number of Workers</td><td>2</td></tr><tr><td>Train Scale Array</td><td>[0.5, 0.75, 1, 1.25, 1.5, 1.75]</td></tr><tr><td>Warm-up Epochs</td><td>10</td></tr><tr><td>Fix Bias</td><td>True</td></tr><tr><td>Batch Norm Epsilon</td><td>1e-3</td></tr><tr><td>Batch Norm Momentum</td><td>0.1</td></tr><tr><td>Evaluation Iteration</td><td>25</td></tr><tr><td>Evaluation Stride Rate</td><td>2/3</td></tr><tr><td>Evaluation Scale Array</td><td>[i]</td></tr><tr><td>Evaluation Flip</td><td>False</td></tr></table>

## 4.1.3 Evaluation Metrics

We evaluate model performance using a set of standard metrics for semantic segmentation, including per-class Intersection over Union (IoU), mean IoU (mIoU), mean IoU excluding the background class. These metrics jointly provide a comprehensive assessment of segmentation quality across categories and pixels, while also accounting for class imbalance. In addition, we report computational eficiency in terms of FLOPs, number of parameters, and inference speed (FPS) to evaluate the practicality of our models for real-world deployment. Detailed metric definitions are provided below.

## Intersection over Union (IoU):

IoU quantifies the overlap between the predicted and ground truth segmentation masks for each class. It is defined as:

$$
\mathrm { I o U } _ { c } = \frac { \left| A _ { c } \cap B _ { c } \right| } { \left| A _ { c } \cup B _ { c } \right| } ,\tag{25}
$$

where $A _ { c }$ and $B _ { c }$ denote the predicted and ground truth pixels for class $^ { c , }$ respectively. A higher IoU indicates better boundary delineation and region matching.

## Mean IoU (mIoU):

The primary evaluation metric is mIoU, which averages IoU scores across all classes, ensuring a balanced evaluation regardless of class frequency. Formally,

$$
\mathrm { m I o U } = \frac { 1 } { C } \sum _ { c = 1 } ^ { C } \frac { | A _ { c } \cap B _ { c } | } { | A _ { c } \cup B _ { c } | } ,\tag{26}
$$

where $C$ is the total number of classes. This metric is particularly valuable in datasets with diverse categories, such as autonomous driving, where rare classes (e.g., pedestrians) must be treated with equal importance as dominant ones (e.g., road).

## Computational Eficiency Metrics:

To assess eficiency, we additionally report the computational cost of each model, including floatingpoint operations (FLOPs), number of parameters, and inference speed measured in frames per second (FPS). FLOPs estimate the theoretical workload of a forward pass, parameter count reflects model size and memory footprint, and FPS measures practical deployment speed under fixed input resolution. By jointly analyzing these metrics, we achieve a comprehensive evaluation of both segmentation quality and computational eficiency.

## 4.2 Quantitative results

## 4.2.1 Results on MFNet

We evaluate our method on the MFNet dataset, a popular benchmark for RGB-thermal semantic segmentation. As shown in Tab. 4, our model achieves 76.9% mIoU, significantly outperforming prior methods such as OmniSegmentor [44] (60.6%), KTBNet [23] (59.9%), and CAFR [5] (67.2%). In particular, our approach yields a 9.7% absolute improvement over CAFR, highlighting its superior capability for RGB-thermal segmentation. We attribute this strong gain to the fact that RGBthermal fusion often contains both shared semantics and complementary / inverse responses across modalities, especially under adverse illumination where RGB cues degrade but thermal remains informative.

Table 4 Semantic segmentation results on the MFNet RGB-Thermal dataset.
<table><tr><td rowspan=1 colspan=3>Method</td><td rowspan=1 colspan=1>Venue</td><td rowspan=1 colspan=1>Backbone</td><td rowspan=1 colspan=1>mIoU (%)</td></tr><tr><td rowspan=3 colspan=3>PAP [39]ABMDRNet [40]GMNet [41]</td><td rowspan=1 colspan=1>CVPR&#x27;19</td><td rowspan=1 colspan=1>ResNet-18</td><td rowspan=1 colspan=1>50.5</td></tr><tr><td rowspan=1 colspan=1>CVPR&#x27;21</td><td rowspan=1 colspan=1>ResNet-18</td><td rowspan=1 colspan=1>54.8</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TIP&#x27;21</td><td rowspan=1 colspan=1>ResNet-50</td><td rowspan=1 colspan=1>57.3</td></tr><tr><td rowspan=3 colspan=1>EGFNet [42]DooDLeNet [43]CMX [4]</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>AAAI&#x27;22</td><td rowspan=1 colspan=1>ResNet-152</td></tr><tr><td></td><td></td><td rowspan=1 colspan=1>CVPR&#x27;22</td><td rowspan=1 colspan=1>ResNet-101</td><td rowspan=1 colspan=1>57.3</td></tr><tr><td></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TITS&#x27;23</td><td rowspan=1 colspan=1>MiT-B4</td><td rowspan=1 colspan=1>59.7</td></tr><tr><td rowspan=1 colspan=1>CMNeXt [2]</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>CVPR&#x27;23</td><td rowspan=1 colspan=1>MiT-B4</td></tr><tr><td rowspan=1 colspan=1>CAFR [5]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>ECCV&#x27;24</td><td rowspan=1 colspan=1>MiT-B4</td><td rowspan=1 colspan=1>67.2</td></tr><tr><td rowspan=1 colspan=3>KTBNet [23]</td><td rowspan=1 colspan=1>CVPR&#x27;25</td><td rowspan=1 colspan=1>Swin-B</td><td rowspan=1 colspan=1>59.9</td></tr><tr><td rowspan=1 colspan=3>OmniSegmentor [44]</td><td rowspan=1 colspan=1>NeurIPS&#x27;25</td><td rowspan=1 colspan=1>DFormer-L</td><td rowspan=1 colspan=1>60.6</td></tr><tr><td rowspan=1 colspan=3>Ours</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>MiT-B4</td><td rowspan=1 colspan=1>76.9</td></tr></table>

Our BPLCA explicitly models both same-signed and opposite-signed cross-modal correlations by decomposing the query/key features into positive and negative components and fusing them in a bidirectional manner, which allows the network to preserve informative “negative evidence” rather than discarding it via non-negative attention mappings. In addition, DFCC and the stage-wise constraint loss $L _ { \mathrm { s t a g e } }$ promote cross-modal feature alignment and stabilize stage-wise fusion, alleviating sensor heterogeneity and reducing the risk of over-relying on a single modality. These mechanisms are consistent with our ablation findings that the fusion and alignment components are complementary and lead to the best MFNet performance when combined.

## 4.2.2 Results on KITTI-360

As shown in Tab. 5, our model achieves 66.8% mIoU, outperforming all competing fusion methods. In particular, PMF [45], TransFuser [20], TokenFusion [46], and HRFuser [47] lag significantly behind. Moreover, our approach surpasses DFormer [48] (66.3%), CAFR [5] (64.5%), and CMNeXt [2] (65.3%), further demonstrating the efectiveness of the proposed fusion mechanism. These results highlight the superiority of our method for RGB-LiDAR segmentation, particularly in large-scale outdoor scenes. We attribute the improvement on RGB-LiDAR segmentation to two aspects of our design. First, LiDAR provides complementary geometric structure while

Table 5 Semantic segmentation results on the KITTI-360 RGB-LiDAR dataset.
<table><tr><td rowspan=1 colspan=2>Method</td><td rowspan=1 colspan=1>Venue</td><td rowspan=1 colspan=1>Backbone</td><td rowspan=1 colspan=1>mIoU (%)</td></tr><tr><td rowspan=1 colspan=2>PMF [45]</td><td rowspan=1 colspan=1>ICCV&#x27;21</td><td rowspan=1 colspan=1>SalsaNext</td><td rowspan=1 colspan=1>54.5</td></tr><tr><td rowspan=2 colspan=2>TransFuser [20]TokenFusion [46]</td><td rowspan=1 colspan=1>CVPR&#x27;21</td><td rowspan=1 colspan=1>RegNetY</td><td rowspan=1 colspan=1>56.6</td></tr><tr><td rowspan=1 colspan=1>CVPR&#x27;22</td><td rowspan=1 colspan=1>MiT-B2</td><td rowspan=1 colspan=1>54.6</td></tr><tr><td rowspan=3 colspan=2>HRFuser [47]CMX [4]CMNeXt [2]</td><td rowspan=1 colspan=1>ITSC&#x27;23</td><td rowspan=1 colspan=1>HRFormer-T</td><td rowspan=1 colspan=1>48.7</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TITS&#x27;23</td><td rowspan=1 colspan=1>MiT-B2</td><td rowspan=1 colspan=1>64.3</td></tr><tr><td rowspan=1 colspan=1>CVPR&#x27;23</td><td rowspan=1 colspan=1>MiT-B2</td><td rowspan=1 colspan=1>65.3</td></tr><tr><td rowspan=1 colspan=2>CAFR [5]</td><td rowspan=1 colspan=1>ECCV&#x27;24</td><td rowspan=1 colspan=1>MiT-B2</td><td rowspan=1 colspan=1>64.5</td></tr><tr><td rowspan=1 colspan=2>DFormer[48]</td><td rowspan=1 colspan=1>ICLR&#x27;24</td><td rowspan=1 colspan=1>DFormer-L</td><td rowspan=1 colspan=1>66.3</td></tr><tr><td rowspan=1 colspan=2>Ours</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>MiT-B2</td><td rowspan=1 colspan=1>66.8</td></tr></table>

RGB provides dense texture/appearance cues; their feature statistics and spatial characteristics difer substantially, which makes naive fusion prone to mismatch. Our DFCC explicitly aligns feature distributions via normalization and learnable afine modulation, and further enforces stage-wise consistency through $L _ { \mathrm { s t a g e } }$ , which encourages the two streams to produce coherent semantic responses across stages. Second, our fusion method performs stage-by-stage fusion, enabling progressive interaction between hierarchical representations rather than relying on a single late fusion point. Together, these components yield more stable cross-modal integration and improve segmentation quality in large-scale outdoor scenes.

Table 6 Semantic segmentation results on the DELIVER dataset.
<table><tr><td rowspan=1 colspan=6>Method</td><td rowspan=1 colspan=1>Venue</td><td rowspan=1 colspan=1>Modal</td><td rowspan=1 colspan=1>Backbone</td><td rowspan=1 colspan=1>mIoU</td></tr><tr><td rowspan=1 colspan=6>HRFuser [47]</td><td rowspan=1 colspan=1>ITSC&#x27;23</td><td rowspan=1 colspan=1>RGB+D</td><td rowspan=1 colspan=1>HRFormer-T</td><td rowspan=1 colspan=1>51.9</td></tr><tr><td rowspan=5 colspan=6>TokenFusion [46]CMX [4]CMNeXt [2]MemorySAM [49]Ours</td><td rowspan=1 colspan=1>CVPR&#x27;22</td><td rowspan=1 colspan=1>RGB+D</td><td rowspan=1 colspan=1>MiT-B2</td><td rowspan=1 colspan=1>60.3</td></tr><tr><td rowspan=1 colspan=1>TITS&#x27;23</td><td rowspan=1 colspan=1>RGB+D</td><td rowspan=1 colspan=1>MiT-B2</td><td rowspan=1 colspan=1>62.7</td></tr><tr><td rowspan=1 colspan=1>CVPR&#x27;23</td><td rowspan=1 colspan=1>RGB+D</td><td rowspan=1 colspan=1>MiT-B2</td><td rowspan=1 colspan=1>63.6</td></tr><tr><td rowspan=2 colspan=1>Arxiv&#x27;25</td><td rowspan=1 colspan=1>RGB+D</td><td rowspan=1 colspan=1>Hiera-B</td><td rowspan=1 colspan=1>63.5</td></tr><tr><td rowspan=1 colspan=1>RGB+D</td><td rowspan=1 colspan=1>MiT-B2</td><td rowspan=1 colspan=1>64.2</td></tr><tr><td rowspan=1 colspan=6>HRFuser [47]</td><td rowspan=1 colspan=1>ITSC&#x27;23</td><td rowspan=1 colspan=1>RGB+E</td><td rowspan=1 colspan=1>HRFormer-T</td><td rowspan=1 colspan=1>42.2</td></tr><tr><td rowspan=2 colspan=6>TokenFusionCMX [4]</td><td rowspan=1 colspan=1>CVPR&#x27;22</td><td rowspan=1 colspan=1>RGB+E</td><td rowspan=1 colspan=1>MiT-B2</td><td rowspan=1 colspan=1>45.6</td></tr><tr><td rowspan=1 colspan=1>TITS&#x27;23</td><td rowspan=1 colspan=1>RGB+E</td><td rowspan=1 colspan=1>MiT-B2</td><td rowspan=1 colspan=1>56.5</td></tr><tr><td rowspan=1 colspan=6>CMNeXt [2]</td><td rowspan=1 colspan=1>CVPR&#x27;23</td><td rowspan=1 colspan=1>RGB+E</td><td rowspan=1 colspan=1>MiT-B2</td><td rowspan=1 colspan=1>57.5</td></tr><tr><td rowspan=1 colspan=6>Any2Seg [18]</td><td rowspan=1 colspan=1>ECCV&#x27;24</td><td rowspan=1 colspan=1>RGB+E</td><td rowspan=1 colspan=1>MiT-B2</td><td rowspan=1 colspan=1>57.8</td></tr><tr><td rowspan=1 colspan=5>KTBNet [23]</td><td rowspan=1 colspan=1>K</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>CVPR&#x27;25</td><td rowspan=1 colspan=1>RGB+E</td><td rowspan=1 colspan=1>Swin-B</td></tr><tr><td rowspan=2 colspan=6>OmniSegmentor [44]Ours</td><td rowspan=1 colspan=1>aniSe</td><td rowspan=1 colspan=1>Segmentor [44]</td><td rowspan=1 colspan=1>NeurIPS&#x27;25</td><td rowspan=1 colspan=1>RGB+E</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>RGB+E</td><td rowspan=1 colspan=1>MiT-B2</td><td rowspan=1 colspan=1>60.3</td></tr><tr><td rowspan=7 colspan=6>HRFuser [47]TokenFusion [46]CMX [4]CMNeXt [2]Any2Seg [18]OmniSegmentor [44]Ours</td><td rowspan=1 colspan=1>ITSC&#x27;23</td><td rowspan=1 colspan=1>RGB+L</td><td rowspan=1 colspan=1>HRFormer-T</td><td rowspan=1 colspan=1>43.1</td></tr><tr><td rowspan=1 colspan=1>CVPR&#x27;22</td><td rowspan=1 colspan=1>RGB+L</td><td rowspan=1 colspan=1>MiT-B2</td><td rowspan=1 colspan=1>53.0</td></tr><tr><td rowspan=1 colspan=1>TITS&#x27;23</td><td rowspan=1 colspan=1>RGB+L</td><td rowspan=1 colspan=1>MiT-B2</td><td rowspan=1 colspan=1>56.4</td></tr><tr><td rowspan=1 colspan=1>CVPR&#x27;23</td><td rowspan=1 colspan=1>RGB+L</td><td rowspan=1 colspan=1>MiT-B2</td><td rowspan=1 colspan=1>58.0</td></tr><tr><td rowspan=3 colspan=1>ECCV&#x27;24NeurIPS&#x27;25</td><td rowspan=2 colspan=1>RGB+LRGB+L</td><td rowspan=1 colspan=1>MiT-B2</td><td rowspan=1 colspan=1>57.9</td></tr><tr><td rowspan=1 colspan=1>MiT-B2</td><td rowspan=1 colspan=1>59.0</td></tr><tr><td rowspan=1 colspan=1>RGB+L</td><td rowspan=1 colspan=1>MiT-B2</td><td rowspan=1 colspan=1>60.7</td></tr><tr><td rowspan=2 colspan=6>HRFuser [47]CMNeXt [2]Ours</td><td rowspan=2 colspan=1>ITSC&#x27;23CVPR&#x27;23</td><td rowspan=2 colspan=1>RGB+D+LRGB+D+LRGB+D+L</td><td rowspan=2 colspan=1>HRFormer-TMiT-B2MiT-B2</td><td rowspan=1 colspan=1>52.765.5</td></tr><tr><td rowspan=1 colspan=1>65.6</td></tr><tr><td rowspan=3 colspan=6>HRFuser [47]CMNeXt [2]MemorySÀM [49]Ours</td><td rowspan=3 colspan=1>ITSC&#x27;23CVPR&#x27;23Arxiv&#x27;25</td><td rowspan=3 colspan=1>RGB+D+E+LRGB+D+E+LRGB+D+E+LRGB+D+E+L</td><td rowspan=1 colspan=1>HRFormer-TMiT-B2</td><td rowspan=1 colspan=1>53.066.3</td></tr><tr><td rowspan=1 colspan=1>Hiera-B</td><td rowspan=1 colspan=1>65.4</td></tr><tr><td rowspan=1 colspan=1>MiT-B2</td><td rowspan=1 colspan=1>67.4</td></tr></table>

## 4.2.3 Results on DELIVER

We further evaluate our approach on the DELIVER dataset, which provides high-resolution RGB-event pairs for semantic segmentation. We trained our model for a total of 60 epochs. As shown in Tab. 6, our method achieves excellent performance compared to other competitive methods. Even at half the original resolution and fewer training epochs, it surpasses recent SOTA methods. This result indicates that our fusion strategy remains efective under memory-constrained settings and is robust to resolution changes, because the proposed BPCF is applied in a stage-wise manner on multi-level features rather than depending on extremely high-resolution feature maps at a single stage.

## 4.2.4 Results on DDD17 and DSEC

Tab. 7 summarizes the results on the DDD17 and DSEC datasets. Our model achieves 77.85% mIoU and 97.10% Pixel Acc on DDD17 dataset and 74.77% mIoU and 95.60% Pixel Acc on DSEC dataset. Compared with the second-best RGB+Event fusion method, EISNet [16], our approach consistently outperforms it in both mIoU and Pixel Acc. In contrast, event-only models such as ESEG-L [50], KWYAF [51], and OpenESS [52] achieve considerably lower scores, underscoring the importance of multi-modal fusion. These results validate the efectiveness of our method and set a new SOTA on the DDD17 and DSEC datasets. We attribute the improvements on event-based driving datasets to the physical and statistical complementarity between RGB and events: events encode intensity changes and often exhibit response patterns that are not strictly positively correlated with RGB appearance. Our BPLCA is designed precisely to capture both same-signed and opposite-signed cross-modal correlations via polarity decomposition, enabling the model to exploit informative signals that would otherwise be suppressed by standard nonnegative attention mappings. Meanwhile, DFCC and the stage-wise consistency loss encourage the two modalities to maintain aligned semantic responses across stages, which helps mitigate heterogeneous feature distributions and stabilizes optimization. Overall, the combination of polarity-aware interaction (BPLCA) and alignment-oriented regularization $\left( \mathrm { D F C C } + L _ { \mathrm { s t a g e } } \right)$ leads to more reliable RGB-event fusion and consistent gains on both DDD17 and DSEC.

Table 7 Semantic segmentation results on the DDD17 and DSEC RGB-Event datasets.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Venue</td><td rowspan="2">Modal</td><td colspan="2">DDD17</td><td colspan="2">DSEC</td></tr><tr><td>mIoU (%)</td><td>Acc. (%)</td><td>|mIoU (%)</td><td>Acc. (%)</td></tr><tr><td>Ev-SegNet [37]</td><td>CVPRW&#x27;19</td><td>Event</td><td>54.81</td><td>89.76</td><td>51.76</td><td>88.61</td></tr><tr><td>E2VID [53]</td><td>TRAMI&#x27;19</td><td>Event</td><td>48.47</td><td>85.84</td><td>44.08</td><td>80.06</td></tr><tr><td>SegFormer [12]</td><td>NeurIPS&#x27;20</td><td>Event</td><td>52.76</td><td>89.34</td><td>54.19</td><td>90.12</td></tr><tr><td>Vid2E [54]</td><td>CVPR&#x27;20</td><td>Event</td><td>56.01</td><td>90.19</td><td></td><td></td></tr><tr><td>EvDistill [55]</td><td>CVPR&#x27;21</td><td>Event</td><td>58.02</td><td></td><td></td><td></td></tr><tr><td>DTL [56]</td><td>ICCV&#x27;21</td><td>Event</td><td>58.80</td><td></td><td></td><td></td></tr><tr><td>ESS [57]</td><td>ECCV&#x27;22</td><td>Event</td><td>61.37</td><td>91.08</td><td>51.57</td><td>89.25</td></tr><tr><td>EvSegFormer [58]</td><td>TIP&#x27;23</td><td>Event</td><td>54.41</td><td>94.72</td><td>56.04</td><td>90.38</td></tr><tr><td>OpenESS [52]</td><td>CVPR&#x27;24</td><td>Event</td><td>63.00</td><td>91.05</td><td>57.21</td><td>90.21</td></tr><tr><td>KWYAF [51]</td><td>AAAI&#x27;25</td><td>Event</td><td>57.69</td><td>90.04</td><td>57.75</td><td>90.87</td></tr><tr><td>ESEG-B [50]</td><td>AAAI&#x27;25</td><td>Event</td><td>57.01</td><td>89.64</td><td>55.93</td><td>90.22</td></tr><tr><td>ESEG-L [50]</td><td>AAAI&#x27;25</td><td>Event</td><td>59.97</td><td>90.68</td><td>57.55</td><td>91.47</td></tr><tr><td>EDCNet-S2D [59]</td><td>TITS’21</td><td>RGB+Event</td><td>61.99</td><td>93.80</td><td>56.75</td><td>92.39</td></tr><tr><td>EV-Transfer [60]</td><td>RA-L’22</td><td>RGB+Event</td><td></td><td></td><td>23.20</td><td>60.50</td></tr><tr><td>CMX [4]</td><td>TITS&#x27;23</td><td>RGB+Event</td><td>71.88</td><td>95.64</td><td>72.42</td><td>95.07</td></tr><tr><td>CMNeXt [2]</td><td>CVPR&#x27;23</td><td>RGB+Event</td><td>72.67</td><td>95.74</td><td>72.54</td><td>95.10</td></tr><tr><td>HMNet-B [61]</td><td>CVPR&#x27;23</td><td>RGB+Event</td><td></td><td></td><td>51.20</td><td>88.70</td></tr><tr><td>HMNet-L [61]</td><td>CVPR&#x27;23</td><td>RGB+Event</td><td></td><td></td><td>55.00</td><td>89.80</td></tr><tr><td>HALSIE [62]</td><td>WACV&#x27;24</td><td>RGB+Event</td><td>60.66</td><td>92.50</td><td>52.43</td><td>89.01</td></tr><tr><td>EISNet [16]</td><td>TMM&#x27;24</td><td>RGB+Event</td><td>75.03</td><td>96.04</td><td>73.07</td><td>95.12</td></tr><tr><td>Hybrid-Segmentation [19]</td><td>AAAI&#x27;25</td><td>RGB+Event</td><td>67.31</td><td>95.07</td><td>66.57</td><td>94.27</td></tr><tr><td>Ours</td><td></td><td>RGB+Event</td><td>77.85</td><td>97.10</td><td>74.77</td><td>95.60</td></tr></table>

Table 8 Semantic segmentation results on the MCubeS dataset with diferent modality combinations. “A” denotes AoLP, “D” denotes DoLP, and “N” denotes NIR.
<table><tr><td rowspan=1 colspan=2>Method</td><td rowspan=1 colspan=1>Venue</td><td rowspan=1 colspan=1>Modal</td><td rowspan=1 colspan=1>mIoU (%)</td></tr><tr><td rowspan=2 colspan=2>MCubeSNet [24]CMNeXt [2]Any2Seg [18]Ours</td><td rowspan=2 colspan=1>CVPR&#x27;22CVPR&#x27;23ECCV&#x27;24一</td><td rowspan=2 colspan=1>RGB+ARGB+ARGB+ARGB+A</td><td rowspan=1 colspan=1>39.10</td></tr><tr><td rowspan=1 colspan=1>48.4248.9550.09</td></tr><tr><td rowspan=4 colspan=2>MCubeSNet [24]CMNeXt [2]ShareCMP [63]Any2Seg [18]Ours</td><td rowspan=4 colspan=1>CVPR’22CVPR&#x27;23TCSVT&#x27;25ECCV&#x27;24</td><td rowspan=4 colspan=1>RGB+A+DRGB+A+DRGB+A+DRGB+A +DRGB+A+D</td><td rowspan=1 colspan=1>42.00</td></tr><tr><td rowspan=1 colspan=1>49.48</td></tr><tr><td rowspan=1 colspan=1>50.99</td></tr><tr><td rowspan=1 colspan=1>[18]</td><td rowspan=1 colspan=1>50.6751.14</td></tr><tr><td rowspan=1 colspan=2>FuseNet [22]</td><td rowspan=1 colspan=1>ACCV&#x27;16</td><td rowspan=1 colspan=1>RGB+A+D+N</td><td rowspan=1 colspan=1>40.58</td></tr><tr><td rowspan=1 colspan=2>MMTM [21]</td><td rowspan=1 colspan=1>CVPR&#x27;20</td><td rowspan=1 colspan=1>RGB+A+D+N</td><td rowspan=1 colspan=1>39.71</td></tr><tr><td rowspan=1 colspan=2>DRConv [64]</td><td rowspan=1 colspan=1>CVPR&#x27;21</td><td rowspan=1 colspan=1>RGB+A+D+N</td><td rowspan=1 colspan=1>34.63</td></tr><tr><td rowspan=1 colspan=2>DDF [65]</td><td rowspan=1 colspan=1>CVPR&#x27;21</td><td rowspan=1 colspan=1>RGB+A+D+N</td><td rowspan=1 colspan=1>36.16</td></tr><tr><td rowspan=1 colspan=2>TransFuser [20]</td><td rowspan=1 colspan=1>CVPR&#x27;21</td><td rowspan=1 colspan=1>RGB+A+D+N</td><td rowspan=1 colspan=1>37.66</td></tr><tr><td rowspan=1 colspan=2>MCubeSNet [24]</td><td rowspan=1 colspan=1>CVPR’22</td><td rowspan=1 colspan=1>RGB+A+D+N</td><td rowspan=1 colspan=1>42.86</td></tr><tr><td rowspan=1 colspan=2>CMNeXt [2]</td><td rowspan=1 colspan=1>CVPR&#x27;23</td><td rowspan=1 colspan=1>RGB+A+D+N</td><td rowspan=1 colspan=1>51.54</td></tr><tr><td rowspan=1 colspan=2>U3M [66]</td><td rowspan=1 colspan=1>PR&#x27;25</td><td rowspan=1 colspan=1>RGB+A+D+N</td><td rowspan=1 colspan=1>51.69</td></tr><tr><td rowspan=2 colspan=2>Any2Seg [18]Ours</td><td rowspan=1 colspan=1>ECCV&#x27;24</td><td rowspan=1 colspan=1>RGB+A+D+N</td><td rowspan=1 colspan=1>51.91</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>RGB+A+D+N</td><td rowspan=1 colspan=1>52.22</td></tr></table>

Table 9 Semantic segmentation results on the ZJU RGB-P dataset. “P(A+D)” denotes using both AoLP and DoLP as auxiliary input.
<table><tr><td>Method</td><td>Venue</td><td>Backbone</td><td>Modal</td><td>|mIoU (%)</td></tr><tr><td>SwiftNet [67]</td><td>CVPR&#x27;19</td><td></td><td>RGB</td><td>80.3</td></tr><tr><td>MultiSenseSeg [68]</td><td>TGRS&#x27;24</td><td></td><td>RGB</td><td>85.7</td></tr><tr><td>SegFormer [12]</td><td>NeurIPS&#x27;21</td><td>MiT-B2</td><td>RGB</td><td>89.6</td></tr><tr><td>CMX [4]</td><td>TITS’23</td><td>MiT-B4</td><td>RGB+A</td><td>92.6</td></tr><tr><td>Ours</td><td>一</td><td>MiT-B4</td><td>RGB+A</td><td>92.8</td></tr><tr><td>CMX [4]</td><td>TITS&#x27;23</td><td>MiT-B4</td><td>RGB+D</td><td>92.5</td></tr><tr><td>MultiSenseSeg [68]</td><td>TGRS&#x27;24</td><td></td><td>RGB+D</td><td>86.9</td></tr><tr><td>Ours</td><td></td><td>MiT-B4</td><td>RGB+D</td><td>93.1</td></tr><tr><td>RPLENet [69]</td><td>TIV&#x27;24</td><td></td><td>|RGB+P(A+D)</td><td>93.1</td></tr><tr><td>NLFNet [70]</td><td>ROBIO&#x27;21</td><td></td><td>RGB+P(A+D)</td><td>84.4</td></tr><tr><td>EAFNet [35]</td><td>OE&#x27;21</td><td></td><td>RGB+P(A+D)</td><td>85.7</td></tr><tr><td>RoadFormer+ [71]</td><td>TIV&#x27;24</td><td></td><td>RGB+P(A+D)</td><td>93.0</td></tr><tr><td>CAFR [5]</td><td>ECCV&#x27;24</td><td>MiT-B4</td><td>RGB+P(A+D)</td><td>83.2</td></tr><tr><td></td><td></td><td>MiT-B4</td><td></td><td></td></tr><tr><td>Ours</td><td>1</td><td></td><td>RGB+P(A+D)</td><td>93.8</td></tr></table>

## 4.2.5 Results on MCubeS

Tab. 8 presents the results on the MCubeS dataset. Using all four modalities (RGB+A+D+N), our method outperforms all compared approaches, including Any2Seg [18], U3M [66], CMNeXt [2] and MCubeSNet [24]. Experiments with diferent modality combinations further demonstrate that each additional modality (A, D, and N) contributes to improved segmentation performance. Overall, fusing all four modalities yields the best results.

Table 10 Semantic segmentation results on the UrbanLF-real and UrbanLF-syn datasets. “LF80”, “LF33”, and “LF8” denote using 80, 33, and 8 sub-aperture views as auxiliary modalities.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Venue</td><td rowspan=1 colspan=2>Modal</td><td rowspan=1 colspan=3>RealSyn|mIoU|mIoU(%) (%)</td></tr><tr><td rowspan=1 colspan=1>PSPNet [72]SegFormer [12]OCR [73]</td><td rowspan=1 colspan=1>CVPR&#x27;17NeurIPS&#x27;21TCSVT’22</td><td rowspan=1 colspan=2>RGBRGBRGB</td><td rowspan=1 colspan=2>76.3482.2078.60</td><td rowspan=1 colspan=1>75.7878.5379.36</td></tr><tr><td rowspan=1 colspan=1>DAVSS [74]TMANet [75]</td><td rowspan=1 colspan=1>TCSVT’20ICIP&#x27; 21</td><td rowspan=1 colspan=2>VideoVideo</td><td rowspan=1 colspan=2>75.9177.14</td><td rowspan=1 colspan=1>74.2776.41</td></tr><tr><td rowspan=1 colspan=1>ESANet [76]SA-Gate[77]</td><td rowspan=1 colspan=1>ICRA&#x27;21ECCV&#x27; 20</td><td rowspan=1 colspan=2>RGB-DRGB-D</td><td rowspan=1 colspan=2>-–</td><td rowspan=1 colspan=1>79.4379.53</td></tr><tr><td rowspan=4 colspan=1>PSPNet-LF [72]OCR-LF [34]CMNeXt[2]CMNeXt[2]CMNeXt[2]</td><td rowspan=1 colspan=1>CVPR&#x27;17TCSVT’22CVPR&#x27;23</td><td rowspan=1 colspan=2>|RGB+LF33|RGB+LF33RGB+LF8</td><td rowspan=2 colspan=2>78.1079.3283.2282.62</td><td rowspan=2 colspan=1>77.8880.4380.7480.98</td></tr><tr><td rowspan=3 colspan=1>CVPR’23CVPR&#x27;23</td><td rowspan=3 colspan=2>RGB+LF33RGB+LF80</td></tr><tr><td rowspan=2 colspan=2>83.11</td><td></td></tr><tr><td rowspan=1 colspan=1>81.02</td></tr><tr><td rowspan=3 colspan=1>OursOursOurs</td><td rowspan=3 colspan=1>一I</td><td rowspan=3 colspan=2>RGB+LF8RGB+LF33RGB+LF80</td><td rowspan=1 colspan=2>82.59</td><td rowspan=1 colspan=1>80.98</td></tr><tr><td rowspan=2 colspan=1></td><td></td><td></td></tr><tr><td rowspan=1 colspan=2>83.3083.99</td><td rowspan=1 colspan=1>81.2281.91</td></tr></table>

## 4.2.6 Results on ZJU

On the ZJU dataset, our method achieves 93.8% mIoU with both A and D inputs, surpassing all competing methods (see Tab. 9). Under diferent modality settings, it attains 92.8% with RGB+A and 93.1% with RGB+D, outperforming CMX [4] in both cases. These results highlight the complementary role of polarization cues and demonstrate the efectiveness of our fusion strategy.

## 4.2.7 Results on UrbanLF

As shown in Tab. 10, our model achieves 83.99% mIoU on UrbanLF-real and 81.91% mIoU on UrbanLFsyn, outperforming CMNeXt [2] (83.22%/80.74%). Even under diferent sub-aperture configurations, our framework maintains strong performance, achieving 83.30%/81.22% with LF33 and 82.59%/80.98% with LF8. Compared with RGB-only and video-based methods, our approach delivers substantial improvements, underscoring the advantage of integrating sub-aperture views with RGB guidance.

## 4.3 Ablation Studies

## 4.3.1 PLCA Branch Ablation

To provide quantitative evidence for the PLCA design, we conduct mechanism-level ablations by isolating diferent branches of BPLCA. In addition, We further introduce an SS-Duplicated setting to separate the efect of complementary interactions from increased model capacity. The results on MFNet (RGB+Thermal) are summarized in Tab. 11. Unibranch settings lead to a clear performance drop, while combining both branches achieves the best performance, demonstrating that opposite-signed interactions provide complementary information. Qualitative results is provided in Fig. 7.

Table 11 Ablation of PLCA branch settings. ‘SS‘ and ‘OS‘ denote the number of Same-signed and Opposite-signed branches used. Single: sets opposite to zero; Duplicated: applies two identical branches.
<table><tr><td>Method</td><td>SS</td><td>OS</td><td>mIoU (%)</td></tr><tr><td>Baseline</td><td>0</td><td>0</td><td>59.7</td></tr><tr><td>SS Single</td><td>1</td><td>0</td><td>74.5</td></tr><tr><td>OS Single</td><td>0</td><td>1</td><td>76.5</td></tr><tr><td>SS Duplicated</td><td>2</td><td>0</td><td>76.0</td></tr><tr><td>Full BPLCA</td><td>1</td><td>1</td><td>76.9</td></tr></table>

## 4.3.2 Component Ablation

Table 12 Ablation study of diferent architectures on the MFNet and DELIVER (RGB+Depth) datasets.
<table><tr><td>Architecture</td><td>MFNet mIoU (%)</td><td>DELIVER mIoU (%)</td></tr><tr><td>Baseline</td><td>59.70</td><td>59.55</td></tr><tr><td>+ DFCC</td><td>65.40</td><td>63.54</td></tr><tr><td>+ BPLCA</td><td>73.66</td><td>63.70</td></tr><tr><td> $+ \mathrm { B P L C A } + \mathrm { D F C C }$ </td><td>76.36</td><td>64.12</td></tr><tr><td>十  $\mathrm { B P L C A } + { \mathcal { L } } _ { \mathrm { s t a g e } }$ </td><td>76.57</td><td>63.76</td></tr><tr><td>+ BPLCA + DFČC  $+ \mathcal { L } _ { \mathrm { s t a g e } }$ </td><td>76.94</td><td>64.18</td></tr></table>

Table 13 Ablation study of diferent architectures on the DSEC, and DDD17 datasets.
<table><tr><td>Architecture</td><td>DSEC mIoU (%)</td><td>DDD17 mIoU (%)</td></tr><tr><td>Baseline</td><td>72.42</td><td>71.88</td></tr><tr><td> $+ \mathrm { \ D F C C }$ </td><td>72.91</td><td>74.32</td></tr><tr><td>+ BPLCA</td><td>74.04</td><td>77.57</td></tr><tr><td> $+ \mathrm { B P L C A } + \mathrm { D F C C }$ </td><td>74.29</td><td>77.73</td></tr><tr><td> $+ \ \mathrm { B P L C A } + \mathcal { L } _ { \mathrm { s t a g e } }$ </td><td>74.37</td><td>77.79</td></tr><tr><td> $+ \mathrm { B P L C A } + \mathrm { D F C C }$   $+ \mathcal { L } _ { \mathrm { s t a g e } }$ </td><td>74.77</td><td>77.85</td></tr></table>

Tab. 12 summarizes the ablation results on the MFNet and DELIVER (RGB+Depth) datasets, highlighting the contributions of each proposed component in terms of mIoU. Specifically, introducing BPLCA yields substantial gains, demonstrating that BPLCA efectively enhances cross-modal interaction. Incorporating DFCC further boosts performance by refining feature consistency. The dense stage-wise loss $\mathcal { L } _ { s t a g e }$ provides additional supervision and stabilizes optimization, leading to further improvements across all datasets. Finally, combining BPLCA, DFCC, and $\mathcal { L } _ { s t a g e }$ achieves the best performance. Overall, these results confirm that the three components are complementary: BPLCA strengthens modality fusion, DFCC enforces feature consistency, and $\mathcal { L } _ { \mathrm { s t a g e } }$ supplies multi-level supervision—together delivering significant segmentation improvements.

We further provide additional ablation results on RGB+Event datasets including DSEC and DDD17 in Tab. 13. Certain improvements are observed while combining diferent components of the proposed method.

## 4.4 Eficiency Analysis

Table 14 GFLOPs and parameter counts of diferent models using MiT-B2 and MiT-B4 backbones. Lower values indicate better eficiency.
<table><tr><td rowspan="2">Model 一</td><td rowspan="2">| Input Size</td><td colspan="2">MiT-B2</td><td colspan="2">MiT-B4</td></tr><tr><td></td><td>GFLOPs ↓ Params (M) ↓ | GFLOPs ↓ Params (M) ↓</td><td></td><td></td></tr><tr><td rowspan="2">Baseline [5] Ours</td><td>3×480×640</td><td>184.8</td><td>63.1</td><td>333.9</td><td>136.3</td></tr><tr><td>3×480×640</td><td>162.4</td><td>66.9</td><td>315.1</td><td>139.8</td></tr><tr><td rowspan="2">Baseline [5] Ours</td><td> $3 { \times } 5 1 2 { \times } 5 1 2$ </td><td>152.5</td><td>63.1</td><td>275.6</td><td>136.3</td></tr><tr><td> $3 { \times } 5 1 2 { \times } 5 1 2$ </td><td>138.3</td><td>66.9</td><td>264.2</td><td>139.8</td></tr></table>

Table 15 Inference speed comparison using MiT-B2 backbone. Higher FPS and lower runtime indicate better eficiency.
<table><tr><td>Method</td><td>|Baseline</td><td>[5] |CMX [4]</td><td>|CMNeXt [2]</td><td>Ours</td></tr><tr><td>FPS ↑</td><td>19.04</td><td>28.27</td><td>30.32</td><td>16.90</td></tr><tr><td>Avg. Time (s) ↓</td><td>0.052</td><td>0.035</td><td>0.033</td><td>0.059</td></tr></table>

We analyze model eficiency in terms of computational complexity and inference speed. As shown in Tab. 14, our model introduces a modest increase in parameters compared with the cross-self-attention [5] (from 63.1M to 66.9M), mainly due to the additional fusion and alignment components. Despite this increase, the proposed design reduces overall FLOPs (from 184.8 to 162.4 GFLOPs at $4 8 0 \times 6 4 0 )$ , indicating that the polarity-aware linear attention and stage-wise fusion avoid the heavy quadratic cost typical of standard attention mechanisms.

In terms of runtime eficiency (Tab. 15), our model achieves 16.9 FPS, only slightly lower than the cross-self-attention (19.0 FPS), with a small increase in average inference time (from 0.052 s to 0.059 s per image). This demonstrates that the added fusion modules incur limited practical overhead while enabling consistent performance gains across datasets. Overall, the proposed method achieves a favorable trade-of between computational eficiency and segmentation accuracy, remaining practical for real-time applications.

## 4.5 Qualitative Analysis

In Fig. 6, we present qualitative comparisons on five representative datasets: MFNet [30] (RGB-Thermal), KITTI-360 [31] (RGB-LiDAR), DSEC [33] (RGB-Event), ZJU-RGBP [35] (RGB-Polarization), and

![](images/20d008ce071015f20af599405f48b972b18f99d9e5412cbbb8a95cadee8c5cb0.jpg)  
Fig. 6 Qualitative results across multiple modalities. Our model consistently produces sharper boundaries and more accurate object recognition under challenging conditions.

UrbanLF-real [34] (RGB-LightField). On MFNet, our method more clearly distinguishes pedestrians—particularly small and distant ones—that the baseline often fails to separate from the background. On KITTI-360, it delineates roadside trees with sharper boundaries and smoother transitions between vegetation and surrounding regions. On DSEC, our model suppresses spurious responses in the background sky, yielding more stable predictions under fast-motion conditions. On ZJU-RGBP, improvements are evident around reflective surfaces, where pedestrians and vehicle windows are predicted with higher accuracy and fewer artifacts. On UrbanLF-real, our approach provides more consistent segmentation of riders, pedestrians, and background elements, maintaining coherent parsing even under occlusion. Across all benchmarks, these qualitative visualizations show that our fusion strategy consistently produces more reliable results, underscoring its superiority across diverse sensing scenarios.

In Fig. 7 we present visualization of the BPLCA component by computing the afinity of the $x ^ { s }$ and $x ^ { o }$ vector on specific points on the image. We can observe a certain complementary information focus and collaboration between the separated branches.

## 4.6 Analysis of ScoreNet Dynamic Modality Selection

To further investigate whether ScoreNet performs meaningful dynamic modality selection, we visualize its decision heatmaps at diferent encoder stages. ScoreNet assigns an informativeness score to each auxiliary modality at every spatial patch and selects the modality with the highest score to construct a unified auxiliary representation before stage-wise RGB-auxiliary fusion. Therefore, the routing decision is spatially adaptive rather than image-wise or modality-wise fixed.

As shown in Fig.8, the ScoreNet responses exhibit clear stage-dependent and modality-dependent patterns. In shallow stages, the score maps contain more fine-grained and high-frequency responses, indicating that early routing mainly focuses on local texture, boundary, and low-level appearance variations. In contrast, the deeper stages show smoother and more semantically organized activation regions, where high-score responses are aligned with object-level structures such as vehicles, road boundaries, and background regions. This suggests that ScoreNet progressively shifts from low-level local selection to high-level semantic modality routing.

Moreover, the decision maps of diferent auxiliary modalities are not identical. For example, AoLP and DoLP exhibit complementary high-response regions across the scene, indicating that ScoreNet does not collapse to a static preference for a single modality. Instead, it selects diferent auxiliary cues according to local scene characteristics and feature hierarchy. These observations support our design motivation: ScoreNet first constructs an informative auxiliary feature stream through patch-wise dynamic modality selection, and the subsequent BPCF modules further fuse this selected auxiliary representation with RGB features in a stage-wise manner. The resulting predictions are well aligned with the ground-truth segmentation, demonstrating that the proposed selection mechanism provides meaningful complementary cues for robust semantic segmentation.

![](images/384080e250fa6d50f9c9913ecfb3d795cd75aea33af2c9899b9c1aa5786d9598.jpg)  
Fig. 7 BPLCA component visualization. RGB-Thermal sample selected from MFNet dataset [30] while RGB-Event sample from DELIVER dataset [2].

![](images/6f03dc97f77741c5455ef392e37f050713a03a3463674d1dcd2c80a620a5fffd.jpg)  
Fig. 8 Here is the visualization of ScoreNet on ZJU dataset [35]. Red regions indicate higher modality selection scores, while blue regions indicate lower scores.

## 5 Conclusion

In this work, we presented a multi-modal semantic segmentation framework that flexibly integrates RGB with diverse auxiliary modalities, including thermal, LiDAR, depth, event, light field, and polarization data. Our approach features a selective fusion mechanism that dynamically activates the most informative auxiliary modality at each spatial location, a bidirectional polarity-aware linear cross-attention (BPLCA) combined with a dual feature consistency constraint (DFCC) for feature-aligned fusion, and a stage-wise supervision loss that progressively enforces cross-modal consistency. Extensive experiments on eight public benchmarks, including MFNet, KITTI-360, DELIVER, DDD17, DSEC, MCubeS, ZJU, and UrbanLF, demonstrate that our method consistently outperforms SOTA approaches.

## Funding Information

This work was supported by the National Natural Science Foundation of China (Grant No. 62603145). The funding body had no role in the design of the study; in the collection, analysis, or interpretation of data; in the writing of the manuscript; or in the decision to publish the results.

## Conflict of Interest Statement

The authors declare that they have no conflict of interest regarding the publication of this paper.

## Data Availability

This study uses publicly available datasets. No new data were created or analyzed in this study. Data sharing is not applicable to this article.

## References

[1] Bijelic, M., Gruber, T., Mannan, F., Kraus, F., Ritter, W., Dietmayer, K., Heide, F.: Seeing through fog without seeing fog: Deep multimodal sensor fusion in unseen adverse weather. In: 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11679–11689 (2020)

[2] Zhang, J., Liu, R., Shi, H., Yang, K., Reiß, S., Peng, K., Fu, H., Wang, K., Stiefelhagen, R.: Delivering arbitrarymodal semantic segmentation. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1136–1147 (2023)

[3] Chen, G., Cao, H., Conradt, J., Tang, H., Röhrbein, F., Knoll, A.C.: Event-based neuromorphic vision for autonomous driving: A paradigm shift for bio-inspired visual sensing and perception. IEEE Signal Process. Mag. 37(4), 34–49 (2020)

[4] Zhang, J., Liu, H., Yang, K., Hu, X., Liu, R., Stiefelhagen, R.: CMX: cross-modal fusion for RGB-X semantic segmentation with transformers. IEEE Trans. Intell. Transp. Syst. 24(12), 14679–14694 (2023)

[5] Cao, H., Zhang, Z., Xia, Y., Li, X., Xia, J., Chen, G., Knoll, A.: Embracing events and frames with hierarchical feature refinement network for object detection. In: European Conference on Computer Vision (ECCV), vol. 15142, pp. 161–177 (2024)

[6] Long, J., Shelhamer, E., Darrell, T.: Fully convolutional networks for semantic segmentation. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3431–3440 (2015)

[7] Ronneberger, O., Fischer, P., Brox, T.: U-net: Convolutional networks for biomedical image segmentation. In: Navab, N., Hornegger, J., III, W.M.W., Frangi, A.F. (eds.) Medical Image Computing and Computer-Assisted Intervention (MICCAI), vol. 9351, pp. 234–241 (2015)

[8] Badrinarayanan, V., Kendall, A., Cipolla, R.: Segnet: A deep convolutional encoder-decoder architecture for image segmentation. IEEE Trans. Pattern Anal. Mach. Intell. 39(12), 2481–2495 (2017)

[9] Chen, L., Zhu, Y., Papandreou, G., Schrof, F., Adam, H.: Encoder-decoder with atrous separable convolution for semantic image segmentation. In: Ferrari, V., Hebert, M., Sminchisescu, C., Weiss, Y. (eds.) European Conference on Computer Vision (ECCV), vol. 11211, pp. 833–851 (2018)

[10] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N.: An image is worth 16x16 words: Transformers for image recognition at scale. In: 9th International Conference on Learning Representations (ICLR) (2021)

[11] Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B.: Swin transformer: Hierarchical vision transformer using shifted windows. In: 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 9992–10002 (2021)

[12] Xie, E., Wang, W., Yu, Z., Anandkumar, A., Álvarez, J.M., Luo, P.: Segformer: Simple and eficient design for semantic segmentation with transformers. In: Annual Conference on Neural Information Processing Systems (NeurIPS), pp. 12077–12090 (2021)

[13] Cao, H., Chen, G., Zhao, H., Jiang, D., Zhang, X., Tian, Q., Knoll, A.: SDPT: semantic-aware dimension-pooling transformer for image segmentation. IEEE Trans. Intell. Transp. Syst. 25(11), 15934–15946 (2024)

[14] Sun, L., Sakaridis, C., Liang, J., Jiang, Q., Yang, K., Sun, P., Ye, Y., Wang, K., Gool, L.V.: Event-based fusion for motion deblurring with cross-modal attention. In: European Conference on Computer Vision (ECCV), vol. 13678, pp. 412–428 (2022)

[15] Zhou, Z., Wu, Z., Boutteau, R., Yang, F., Demonceaux, C., Ginhac, D.: Rgb-event fusion for moving object detection in autonomous driving. In: IEEE International Conference on Robotics and Automation (ICRA), pp. 7808–7815 (2023)

[16] Xie, B., Deng, Y., Shao, Z., Li, Y.: Eisnet: A multi-modal fusion network for semantic segmentation with events and images. IEEE Trans. Multim. 26, 8639–8650 (2024)

[17] Cao, H., Chen, G., Li, Z., Hu, Y., Knoll, A.: Neurograsp: Multimodal neural network with euler region regression for neuromorphic vision-based grasp pose estimation. IEEE Transactions on Instrumentation and Measurement 71, 1–11 (2022)

[18] Zheng, X., Lyu, Y., Wang, L.: Learning modality-agnostic representation for semantic segmentation from any modalities. In: Leonardis, A., Ricci, E., Roth, S., Russakovsky, O., Sattler, T., Varol, G. (eds.) European Conference on Computer Vision (ECCV), vol. 15092, pp. 146–165 (2024)

[19] Li, H., Peng, Y., Yuan, J., Wu, P., Wang, J., Zhang, Y., Sun, X.: Eficient event-based semantic segmentation via exploiting frame-event fusion: A hybrid neural network approach. In: Walsh, T., Shah, J., Kolter, Z. (eds.) Association for the Advancement of Artificial Intelligence (AAAI), pp. 18296–18304 (2025)

[20] Prakash, A., Chitta, K., Geiger, A.: Multi-modal fusion transformer for end-to-end autonomous driving. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 7077–7087 (2021)

[21] Joze, H.R.V., Shaban, A., Iuzzolino, M.L., Koishida, K.: MMTM: multimodal transfer module for CNN fusion. In: 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 13286–13296 (2020)

[22] Hazirbas, C., Ma, L., Domokos, C., Cremers, D.: Fusenet: Incorporating depth into semantic segmentation via fusionbased CNN architecture. In: Lai, S., Lepetit, V., Nishino, K., Sato, Y. (eds.) 13th Asian Conference on Computer Vision (ACCV), vol. 10111, pp. 213–228 (2016)

[23] Cai, J., Su, J., Li, Q., Yang, W., Wang, S., Zhao, T., He, S., Liu, W.: Keep the balance: A parameter-eficient symmetrical framework for RGB+X semantic segmentation. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10587–10598 (2025)

[24] Liang, Y., Wakaki, R., Nobuhara, S., Nishino, K.: Multimodal material segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 19800–19808 (2022)

[25] Meng, W., Luo, Y., Li, X., Jiang, D., Zhang, Z.: Polaformer: Polarity-aware linear attention for vision transformers. In: The Thirteenth International Conference on Learning Representations (ICLR) (2025)

[26] Zheng, X., Lyu, Y., Zhou, J., Wang, L.: Centering the value of every modality: Towards eficient and resilient modalityagnostic semantic segmentation. In: European Conference on Computer Vision (ECCV), vol. 15127, pp. 192–212 (2024)

[27] Jia, D., Guo, J., Han, K., Wu, H., Zhang, C., Xu, C., Chen, X.: Geminifusion: Eficient pixel-wise multimodal fusion for vision transformer. In: Forty-first International Conference on Machine Learning (ICML) (2024)

[28] Srivastava, S., Sharma, G.: Omnivec2-a novel transformer based network for large scale multimodal and multitask learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 27412–27424 (2024)

[29] Katharopoulos, A., Vyas, A., Pappas, N., Fleuret, F.: Transformers are rnns: Fast autoregressive transformers with linear attention. In: Proceedings of the 37th International Conference on Machine Learning (ICML), vol. 119, pp. 5156–5165 (2020)

[30] Ha, Q., Watanabe, K., Karasawa, T., Ushiku, Y., Harada, T.: Mfnet: Towards real-time semantic segmentation for autonomous vehicles with multi-spectral scenes. In: 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 5108–5115 (2017)

[31] Liao, Y., Xie, J., Geiger, A.: KITTI-360: A novel dataset and benchmarks for urban scene understanding in 2d and 3d. IEEE Trans. Pattern Anal. Mach. Intell. 45(3), 3292–3310 (2023)

[32] Binas, J., Neil, D., Liu, S.-C., Delbruck, T.: Ddd17: End-to-end davis driving dataset. arXiv preprint arXiv:1711.01458 (2017)

[33] Gehrig, M., Aarents, W., Gehrig, D., Scaramuzza, D.: DSEC: A stereo event camera dataset for driving scenarios. IEEE Robotics Autom. Lett. 6(3), 4947–4954 (2021)

[34] Sheng, H., Cong, R., Yang, D., Chen, R., Wang, S., Cui, Z.: Urbanlf: A comprehensive light field dataset for semantic segmentation of urban scenes. IEEE Trans. Circuits Syst. Video Technol. 32(11), 7880–7893 (2022)

[35] Xiang, K., Yang, K., Wang, K.: Polarization-driven semantic segmentation via eficient attention-bridged fusion. Optics Express 29(4), 4802–4820 (2021)

[36] Cordts, M., Omran, M., Ramos, S., Rehfeld, T., Enzweiler, M., Benenson, R., Franke, U., Roth, S., Schiele, B.: The cityscapes dataset for semantic urban scene understanding. In: 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3213–3223 (2016)

[37] Alonso, I., Murillo, A.C.: Ev-segnet: Semantic segmentation for event-based cameras. In: IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pp. 1624–1633 (2019)

[38] Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: 7th International Conference on Learning Representations (ICLR) (2019)

[39] Zhang, Z., Cui, Z., Xu, C., Yan, Y., Sebe, N., Yang, J.: Pattern-afinitive propagation across depth, surface normal and semantic segmentation. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4106–4115 (2019)

[40] Zhang, Q., Zhao, S., Luo, Y., Zhang, D., Huang, N., Han, J.: Abmdrnet: Adaptive-weighted bi-directional modality diference reduction network for RGB-T semantic segmentation. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2633–2642 (2021)

[41] Zhou, W., Liu, J., Lei, J., Yu, L., Hwang, J.: Gmnet: Graded-feature multilabel-learning network for rgb-thermal urban scene semantic segmentation. IEEE Trans. Image Process. 30, 7790–7802 (2021)

[42] Zhou, W., Dong, S., Xu, C., Qian, Y.: Edge-aware guidance fusion network for rgb-thermal scene parsing. In: Thirty-Sixth AAAI Conference on Artificial Intelligence (AAAI), pp. 3571–3579 (2022)

[43] Frigo, O., Martin-Gafé, L., Wacongne, C.: Doodlenet: Double deeplab enhanced feature fusion for thermal-color semantic segmentation. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pp. 3020–3028 (2022)

[44] Yin, B.-W., Cao, J.-L., Zhang, X., Chen, Y., Cheng, M.-M., Hou, Q.: Omnisegmentor: A flexible multi-modal learning framework for semantic segmentation. The Thirty-ninth Annual Conference on Neural Information Processing Systems (NeurIPS) (2025)

[45] Zhuang, Z., Li, R., Jia, K., Wang, Q., Li, Y., Tan, M.: Perception-aware multi-sensor fusion for 3d lidar semantic segmentation. In: 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 16260–16270 (2021)

[46] Wang, Y., Chen, X., Cao, L., Huang, W., Sun, F., Wang, Y.: Multimodal token fusion for vision transformers. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 12176–12185 (2022)

[47] Brödermann, T., Sakaridis, C., Dai, D., Gool, L.V.: Hrfuser: A multi-resolution sensor fusion architecture for 2d object detection. In: 26th IEEE International Conference on Intelligent Transportation Systems (ITSC), pp. 4159–4166 (2023)

[48] Yin, B., Zhang, X., Li, Z., Liu, L., Cheng, M., Hou, Q.: Dformer: Rethinking RGBD representation learning for semantic segmentation. In: The Twelfth International Conference on Learning Representations (ICLR) (2024)

[49] Liao, C., Zheng, X., Lyu, Y., Xue, H., Cao, Y., Wang, J., Yang, K., Hu, X.: Memorysam: Memorize modalities and semantics with segment anything model 2 for multi-modal semantic segmentation. arXiv preprint arXiv:2503.06700 (2025)

[50] Zhao, Y., Lyu, G., Li, K., Wang, Z., Chen, H., Yang, Z., Deng, Y.: ESEG: event-based segmentation boosted by explicit edge-semantic guidance. In: Walsh, T., Shah, J., Kolter, Z. (eds.) Association for the Advancement of Artificial Intelligence (AAAI), pp. 10510–10518 (2025)

[51] Li, K., Lyu, G., Chen, H., Xie, B., Yang, Z., Li, Y., Deng, Y.: Know where you are from: Event-based segmentation via spatio-temporal propagation. In: Walsh, T., Shah, J., Kolter, Z. (eds.) Association for the Advancement of Artificial Intelligence (AAAI), pp. 4806–4814 (2025)

[52] Kong, L., Liu, Y., Ng, L.X., Cottereau, B.R., Ooi, W.T.: Openess: Event-based semantic scene understanding with open vocabularies. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 15686–15698 (2024)

[53] Rebecq, H., Ranftl, R., Koltun, V., Scaramuzza, D.: Events-to-video: Bringing modern computer vision to event cameras. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3857–3866 (2019)

[54] Gehrig, D., Gehrig, M., Hidalgo-Carrió, J., Scaramuzza, D.: Video to events: Recycling video datasets for event cameras. In: 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3583–3592 (2020)

[55] Wang, L., Chae, Y., Yoon, S., Kim, T., Yoon, K.: Evdistill: Asynchronous events to end-task learning via bidirectional reconstruction-guided cross-modal knowledge distillation. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 608–619 (2021)

[56] Wang, L., Chae, Y., Yoon, K.: Dual transfer learning for event-based end-task prediction via pluggable event to image translation. In: 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 2115–2125 (2021)

[57] Sun, Z., Messikommer, N., Gehrig, D., Scaramuzza, D.: ESS: learning event-based semantic segmentation from still images. In: Avidan, S., Brostow, G.J., Cissé, M., Farinella, G.M., Hassner, T. (eds.) European Conference on Computer Vision (ECCV), vol. 13694, pp. 341–357. Springer, ??? (2022)

[58] Jia, Z., You, K., He, W., Tian, Y., Feng, Y., Wang, Y., Jia, X., Lou, Y., Zhang, J., Li, G., Zhang, Z.: Event-based semantic segmentation with posterior attention. IEEE Trans. Image Process. 32, 1829–1842 (2023)

[59] Zhang, J., Yang, K., Stiefelhagen, R.: Exploring event-driven dynamic context for accident scene segmentation. IEEE Trans. Intell. Transp. Syst. 23(3), 2606–2622 (2022)

[60] Messikommer, N., Gehrig, D., Gehrig, M., Scaramuzza, D.: Bridging the gap between events and frames through unsupervised domain adaptation. IEEE Robotics Autom. Lett. 7(2), 3515–3522 (2022)

[61] Hamaguchi, R., Furukawa, Y., Onishi, M., Sakurada, K.: Hierarchical neural memory network for low latency event processing. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 22867–22876 (2023)

[62] Biswas, S.D., Kosta, A., Liyanagedera, C.M., Apolinario, M.P.E., Roy, K.: HALSIE: hybrid approach to learning segmentation by simultaneously exploiting image and event modalities. In: IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pp. 5952–5962. IEEE, ??? (2024)

[63] Liu, Z., Wang, B., Wang, L., Mao, C., Li, Y.: Sharecmp: Polarization-aware RGB-P semantic segmentation. IEEE Trans. Circuits Syst. Video Technol. 35(10), 10316–10329 (2025)

[64] Chen, J., Wang, X., Guo, Z., Zhang, X., Sun, J.: Dynamic region-aware convolution. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 8064–8073 (2021)

[65] Zhou, J., Jampani, V., Pi, Z., Liu, Q., Yang, M.: Decoupled dynamic filter networks. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 6647–6656 (2021)

[66] Li, B., Zhang, D., Zhao, Z., Gao, J., Li, X.: U3M: unbiased multiscale modal fusion model for multimodal semantic segmentation. Pattern Recognit. 168, 111801 (2025)

[67] Orsic, M., Kreso, I., Bevandic, P., Segvic, S.: In defense of pre-trained imagenet architectures for real-time semantic segmentation of road-driving images. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 12607–12616 (2019)

[68] Wang, Q., Chen, W., Huang, Z., Tang, H., Yang, L.: Multisenseseg: A cost-efective unified multimodal semantic segmentation model for remote sensing. IEEE Trans. Geosci. Remote. Sens. 62, 1–24 (2024)

[69] Lu, Y., Tian, J., Su, Y., Liu, J., Hao, C., Yue, C., Yang, X.: Polarization-aware low-light image enhancement for nighttime intelligent vehicles. IEEE Trans. Intell. Veh. 10(3), 1944–1958 (2025)

[70] Yan, R., Yang, K., Wang, K.: Nlfnet: Non-local fusion towards generalized multimodal semantic segmentation across rgb-depth, polarization, and thermal images. In: IEEE International Conference on Robotics and Biomimetics (ROBIO), pp. 1129–1135 (2021)

[71] Huang, J., Li, J., Jia, N., Sun, Y., Liu, C., Chen, Q., Fan, R.: Roadformer+: Delivering RGB-X scene parsing through scale-aware information decoupling and advanced heterogeneous feature fusion. IEEE Trans. Intell. Veh. 10(5), 3156–3165 (2025)

[72] Zhao, H., Shi, J., Qi, X., Wang, X., Jia, J.: Pyramid scene parsing network. In: 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 6230–6239 (2017)

[73] Yuan, Y., Chen, X., Wang, J.: Object-contextual representations for semantic segmentation. In: European Conference on Computer Vision (ECCV), vol. 12351, pp. 173–190 (2020)

[74] Zhuang, J., Wang, Z., Wang, B.: Video semantic segmentation with distortion-aware feature correction. IEEE Trans. Circuits Syst. Video Technol. 31(8), 3128–3139 (2021)

[75] Wang, H., Wang, W., Liu, J.: Temporal memory attention for video semantic segmentation. In: 2021 IEEE International Conference on Image Processing (ICIP), pp. 2254–2258 (2021)

[76] Seichter, D., Köhler, M., Lewandowski, B., Wengefeld, T., Gross, H.: Eficient RGB-D semantic segmentation for indoor scene analysis. In: IEEE International Conference on Robotics and Automation (ICRA), pp. 13525–13531 (2021)

[77] Chen, X., Lin, K.-Y., Wang, J., Wu, W., Qian, C., Li, H., Zeng, G.: Bi-directional cross-modality feature propagation with separation-and-aggregation gate for rgb-d semantic segmentation. In: European Conference on Computer Vision (ECCV) (2020)