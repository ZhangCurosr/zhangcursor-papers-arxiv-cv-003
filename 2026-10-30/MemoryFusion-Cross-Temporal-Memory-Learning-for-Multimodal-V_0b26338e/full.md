# MemoryFusion: Cross-Temporal Memory Learning for Multimodal Video Fusion

Meiqi Gong   
Electronic Information School   
Wuhan University   
Wuhan, China   
meiqigong@whu.edu.cn   
Hao Zhang<sup>∗</sup>   
Electronic Information School   
Wuhan University   
Wuhan, China   
zhpersonalbox@gmail.com   
Jiayi Ma<sup>∗</sup>   
School of Robotics   
Wuhan University   
Wuhan, China   
jyma2010@gmail.com

## Abstract

Multimodal videos contain rich temporal cues that can facilitate faithful fusion. However, existing video fusion methods mainly rely on short-range temporal interactions, leaving valuable long-term information underexplored. Moreover, inconsistent fusion representations can cause flickering artifacts, undermining temporal coherence. To address these coupled challenges, we propose Memo ryFusion, a two-stage framework integrating cross-temporal memory learning with progressive temporal refinement. Considering the complementary roles of adjacent and distant frames, the first stage combines a Short-Term Memory Bank for local temporal guidance with a Long-Term Memory Bank that preserves reliable historical information through a fusion-oriented update strategy. To mitigate temporal fluctuations, a Temporal Smoothing Module leverages soft motion cues for coarse background stabilization. Building upon the resulting fusion prior, The second stage further employs a lightweight 3D residual Temporal Refinement Module to correct remaining temporal inconsistencies while enhancing structural details. Extensive experiments demonstrate that MemoryFusion achieves superior fusion quality and temporal consistency over existing framebased and video-based fusion methods. The source code is publicly available at https://github.com/Meiqi-Gong/MemoryFusion.

## 1 Introduction

Due to the inherent limitations of sensing mechanisms and hardware, single-modal images often fail to provide reliable scene understanding, especially in complex environments [5, 2, 24]. To address this issue, multimodal image fusion has emerged as a crucial technique that integrates complementary information from different modalities to produce more comprehensive and reliable representations [19, 8]. Benefiting from its enhanced representational capacity, multimodal fusion has been successfully applied in a wide range of fields, such as autonomous driving [18, 9], intelligent surveillance [23], and medical diagnosis [3], among others [10, 11].

Early research on multi-modal fusion has primarily focused on frame-based design, where framebased modality-specific features are typically combined to enhance detail and contrast. For instance, Ma et al. introduced SwinFusion [12], which employs a Swin Transformer backbone to jointly capture intra-modal and inter-modal spatial dependencies within a single frame, enabling more adaptive and content-aware frame-based fusion. After that, CDDFuse [25] employs Restormer blocks to extract shallow cross-modal frame-wise features, followed by a dual-branch Transformer-CNN extractor that captures global and local representations within each frame independently. Further, Text-DiFuse [22] introduces a text-modulated diffusion model into the frame-based fusion paradigm.

![](images/9268c864bc7971715723dd8059e9fbe0d8d9d9a81a1f1f67db4c07f6801f872e.jpg)  
(a) Existing video fusion methods  
(b) Our proposed MemoryFusion  
(c) Results Comparison with TemCoCo  
Figure 1: Comparison of existing video fusion methods and our proposed MemoryFusion. (a) and (b) illustrate the differences in how temporal information is exploited. (c) shows the corresponding visual comparison results, where the top-right region displays the results of stacked 40 consecutive frames in the same row for temporal consistency evaluation. Higher VIF means better fusion quality and lower flowD means better temporal consistency.

It embeds feature-level integration directly into the single-frame generative diffusion process, enabling simultaneous degradation removal and cross-modal information fusion within each frame. Recently, FC-Former [16] uses a fully-connected self-attention mechanism to capture local and non-local correlations within each single frame. It embeds a multi-source image representation as a non-local prior to enable effective single-frame feature fusion across modalities. Despite their remarkable success in image fusion, they are designed under single frame, whereas in practical scenarios, multimodal data are typically acquired in video form. This discrepancy prevents such methods from effectively modeling and leveraging inter-frame dependencies, thereby resulting in limited fusion performance and undesired temporal flickering when applied to practical video sequences.

Due to the growing recognition of the limitations of frame-based fusion, researchers have recently shifted their attention toward video fusion, which explicitly models and exploits temporal information across consecutive frames. By leveraging temporal correlations, video fusion enables a more comprehensive and stable fusion of multimodal data, resulting in significantly improved visual quality and reduced inter-frame flickering. For example, Gong et al. proposed TemCoCo [6], which aligns features from adjacent frames through deformable convolution (DCN) to capture local short-term correlations, while incorporating a temporal modeling module and a dedicated temporal consistency loss to ensure temporal consistency. In addition, Tang et al. introduced VideoFusion [15], which employs an attention mechanism to associate neighboring frames with the current one and maintains temporal coherence through explicit consistency constraints. Besides that, Zhao et al. proposed UniVF [27], leveraging neighboring-frame learning and optical flow-based feature warping to fuse adjacent frames. Also, they integrate a Transformer-based encoder-decoder with a temporal consistency loss to enhance temporal coherence. Overall, the development of video fusion methods is still at a preliminary stage. Existing video fusion methods primarily focus on short-range correlations between adjacent frames, as concluded in Figure 1 (a), where information of adjacent frames is transferred through various strategies to assist current-frame fusion. This short-term dependency limits the full exploitation of accessible temporal context, thus constraining fusion fidelity. Moreover, their temporal consistency modeling is confined to complex networks with merely simple temporal constraints and no incorporation of temporal or spatial prior, which often results in limited smoothness and the loss of fine spatial details.

Beyond the video fusion literature, memory mechanisms have also been investigated in broader video processing tasks. For example, memory banks are commonly used in video object segmentation to establish long-term correspondence for mask propagation [28, 4], while motion-adaptive temporal filtering and 3D convolution have been explored to suppress temporal fluctuations and capture local spatio-temporal dependencies [13, 21]. These studies demonstrate the potential of temporal memory and temporal filtering for video understanding. Inspired by these temporal modeling paradigms, we redesign them under the unique characteristics of multimodal video fusion.

To address the above limitation, we propose MemoryFusion, a two-stage multimodal video fusion method to explicitly tackle the challenges of underutilized temporal context and inadequate temporal consistency. On the one hand, we fully exploit cross-temporal learning through dual memory banks in the first stage. Specifically, the Short-Term Memory Bank (STMB) aggregates informative cues from recent frames to provide local temporal guidance, while the Long-Term Memory Bank (LTMB) adaptively captures and preserves key representations from long-term frames to supply global temporal guidance. As shown in Figure 1 (b), unlike conventional methods that rely solely on short-range temporal cues, our MemoryFusion also explicitly maintains long-term temporal information through dedicated memory banks, enabling the network to retrieve more comprehensive information. On the other hand, to alleviate temporal fluctuations introduced during cross-frame feature aggregation, a Temporal Smoothing Module (TSM) is incorporated at the end of the first stage to perform region-adaptive coarse temporal smoothing by selectively preserving background information from the preceding frame. Building upon the resulting fusion prior, the second stage incorporates a Temporal Refinement Module (TRM) to guarantee fine temporal coherence. Treating the first-stage output as the initial results, the TRM leverages a 3D convolutional network to explicitly model and enforce temporal constraints, achieving a holistic optimization. Through this progressive coarse-to-fine design, MemoryFusion effectively integrates historical information while maintaining temporal stability. As illustrated in Figure 1(c), our MemoryFusion not only restores structural details that remain blurred in results from other video fusion methods such as TemCoCo, but also preserves temporal coherence across consecutive frames, effectively suppressing flickering artifacts and ensuring stable visual dynamics.

![](images/a964248c09b394bf03c87f686990f6f4f15dfe0ab9e8f9dbcee3d3c939216362.jpg)  
Figure 2: The overall framework of the proposed MemoryFusion.

Our main contributions of this work can be summarized as:

• We propose MemoryFusion, a cross-temporal memory learning for multimodal video fusion that jointly enhances visual quality and temporal consistency.

• Dual memory banks are designed with a redundancy-aware and fusion-quality-aware LTMB update strategy to preserve diverse and reliable historical information.

• We develop a two-fold progressive temporal modeling approach, a TSM alleviating coarse background transitions, and a TRM enforcing fine consistency while refining details.

• Extensive experiments on public datasets demonstrate that MemoryFusion achieves superior performance in terms of fusion quality and temporal consistency.

## 2 Method

This section details the proposed two-stage MemoryFusion, as illustrated in Figure 2. The first stage performs memory-guided fusion with coarse temporal consistency modeling, while the second stage refines temporal consistency based on the fusion prior from the first stage.

## 2.1 Notation and Overview

In the first stage, dual memory banks are designed to store information from multiple temporal scales, and the framework primarily focuses on exploiting historical information stored in the memory bank to facilitate enhanced video fusion. Taking the t-th frame as an example, the visible frame $I _ { t } ^ { \mathrm { v i s } } \in \mathbb { R } ^ { H \times W \times 3 }$ and the infrared frame $I _ { t } ^ { \mathrm { i r } } \in \breve { \mathbb R ^ { H \times W \times 1 } }$ are first fed into two Multiscale Feature Encoders $\mathcal { E } _ { \mathrm { v i s } } , \mathcal { E } _ { \mathrm { i r } }$ to extract hierarchical representations by:

$$
\mathcal { F } _ { t , l } ^ { \mathrm { v i s } } , \mathcal { F } _ { t , m } ^ { \mathrm { v i s } } , \mathcal { F } _ { t , h } ^ { \mathrm { v i s } } = \mathcal { E } _ { \mathrm { v i s } } \big ( I _ { t } ^ { \mathrm { v i s } } \big ) , \quad \mathcal { F } _ { t , l } ^ { \mathrm { i r } } , \mathcal { F } _ { t , m } ^ { \mathrm { i r } } , \mathcal { F } _ { t , h } ^ { \mathrm { i r } } = \mathcal { E } _ { \mathrm { i r } } \big ( I _ { t } ^ { \mathrm { i r } } \big ) ,\tag{1}
$$

![](images/eb5c4b8e0e3c443262657c01c089454058af26626948f598591a430fa307aca6.jpg)  
Figure 3: Implementation of memory banks.

where $\mathcal { F } _ { t , l } ^ { \mathrm { v i s } } , \mathcal { F } _ { t , l } ^ { \mathrm { i r } } \ \in \ \mathbb { R } ^ { \frac { H } { 4 } \times \frac { W } { 4 } \times C _ { 1 } } , \mathcal { F } _ { t , m } ^ { \mathrm { v i s } } , \mathcal { F } _ { t , m } ^ { \mathrm { i r } } \ \in \ \mathbb { R } ^ { \frac { H } { 2 } \times \frac { W } { 2 } \times C _ { 2 } }$ and $\mathcal { F } _ { t , h } ^ { \mathrm { v i s } } , \mathcal { F } _ { t , h } ^ { \mathrm { i r } } ~ \in ~ \mathbb { R } ^ { H \times W \times C _ { 3 } }$ . The corresponding multi-modal features are then fused through multiple feature fusion layers to obtain preliminary fused features $\mathcal { F } _ { t , l } ^ { f } , \mathcal { F } _ { t , m } ^ { f }$ and $\mathcal { F } _ { t , h } ^ { f }$ . These fused features interact with the memory banks through a retrieval process, where the network selectively extracts the relevant information from the memory banks to support the current fusion. The memory banks are dynamically updated following distinct strategies: the Short-term Memory Bank (STMB) is refreshed at every frame, while the Long-term Memory Bank (LTMB) is updated only when its capacity limit is reached, following a specific update rule that will be detailed in Section 2.2. Subsequently, the features are passed to the Temporal Smoothing Module (TSM), which integrates temporal information by incorporating the previous features $\mathcal { F } ^ { \mathrm { p r e v } }$ at corresponding network layers to get the coarse temporal-smoothed feature $\mathcal { F } _ { t } ^ { s m }$ . Finally, the memory-enhanced features $\mathcal { F } _ { t } ^ { \mathrm { e n } }$ and the smoothed fusion features $\mathcal { F } _ { t } ^ { \mathrm { s m } }$ are fed into a Multiscale Decoder D to reconstruct the fused frame $I _ { t } ^ { f }$

$$
I _ { t } ^ { f } = \mathcal { D } ( \mathcal { F } _ { t } ^ { \mathrm { e n } } , \mathcal { F } _ { t } ^ { \mathrm { s m } } ) .\tag{2}
$$

Except for the first frame $( \mathcal { F } ^ { p r e v }$ is absent), all subsequent frames follow the same processing pipeline. After the first stage, the entire fused video sequence $V _ { f }$ , together with the original input video sequences $V _ { v i s }$ and $V _ { i r } ,$ is fed into the second-stage Temporal Refinement Module (TRM) to further enhance global temporal consistency across the video.

## 2.2 Memory Banks

For each frame $t ,$ after obtaining the preliminary fused feature $\mathcal { F } _ { t }$ by fusion layers, we retrieve relevant information from a Short-Term Memory Bank (STMB) and a Long-Term Memory Bank (LTMB) to implement cross-temporal enhancement, as illustrated in Figure 3 (a).

Short-term Memory Bank (STMB) The STMB stores enhanced features from recent frames to provide short-term temporal context for the current fused feature. Since recent frames usually contain locally correlated structures and textures, the short-term memory can provide complementary cues for enhancing dynamic details. Different from directly adding past-frame features to the current representation, our STMB performs current-anchored memory retrieval. Specifically, the current fused feature $\mathcal { F } _ { t }$ provides the query $\mathbf { Q } _ { c } ,$ as well as the current key and value $( \mathbf { K } _ { c } , \mathbf { V } _ { c } )$ , while the STMB provides memory keys and values $\mathbf { \Pi } ( \mathbf { K } _ { s } , \mathbf { V } _ { s } )$ . Then the short-term enhanced output is computed via multi-head attention:

$$
\mathbf { t g t } _ { s } = \operatorname { A t t e n t i o n } ( \mathbf { Q } _ { c } , \mathbf { K } _ { c } + \mathbf { K } _ { s } , \mathbf { V } _ { c } + \mathbf { V } _ { s } ) .\tag{3}
$$

Long-term Memory Bank (LTMB) The LTMB stores representative features from wider-range frames to provide long-term contextual guidance for the current frame. Similar to STMB, the LTMB also adopts current-anchored memory retrieval, where the current fused feature provides the query $\mathbf { Q } _ { c } ,$ and the LTMB provides memory keys and values $( \mathbf { K } _ { l } , \mathbf { V } _ { l } )$ . Different from STMB, the LTMB focuses on retrieving representative long-range contextual cues from temporally distant frames, and is updated with a selective strategy to avoid being dominated by redundant or low-quality historical features. The long-term enhanced output is computed as:

$$
\mathbf { t g t } _ { l } = \mathrm { A t t e n t i o n } ( \mathbf { Q } _ { c } , \mathbf { K } _ { l } , \mathbf { V } _ { l } ) .\tag{4}
$$

After both short-term and long-term enhancement, the final enhanced feature is $\mathcal { F } _ { t } ^ { \mathrm { e n } } = \mathcal { F } _ { t } + \mathbf { t g } \mathbf { t } _ { s } +$ $\mathbf { t g t } _ { l }$ . The enhanced feature $\mathcal { F } _ { t } ^ { \mathrm { e n } }$ is stored in STMB for the next frame enhancement. For LTMB, new features are directly added until the maximum length is reached, and we follow previous work to set the maximum length as 8 [28]. Otherwise, LTMB follows update rules based on two criteria when receiving a new frame, detailed as follows.

Update Rules Although the LTMB effectively stores historical cues for enhancement, its capacity must be restricted to avoid redundant memory accumulation and outdated information. As illustrated in Figure 3 (b), we introduce a dual-criterion pruning mechanism that evaluates each memory entry from two complementary perspectives: information redundancy, which measures whether a memory entry can be well reconstructed by the remaining entries, andfusion quality, which reflects its ability to perform enhancement and fusion. This design encourages the LTMB to retain diverse and reliable memories, while removing entries that are either redundant or associated with poor fusion quality.

Specifically, when the number of long-term memory entries exceeds the predefined capacity, we perform memory pruning for each batch independently. Given N long-term memory entries $\{ \mathbf { M } _ { i } \} _ { i = 1 } ^ { N } .$ we first reshape and normalize them along the feature dimension to obtain $\mathbf { M } \in \mathbb { R } ^ { N \times d }$ . For each memory entry $\mathbf { M } _ { i } ,$ we use the remaining entries ${ { \bf { M } } _ { - i } }$ to linearly reconstruct it by solving a leastsquares problem:

$$
\mathbf { w } _ { i } ^ { * } = \arg \operatorname* { m i n } _ { \mathbf { w } _ { i } } \left\| \mathbf { M } _ { - i } ^ { \top } \mathbf { w } _ { i } - \mathbf { M } _ { i } \right\| _ { 2 } ^ { 2 } ,\tag{5}
$$

where ${ { \bf { M } } _ { - i } }$ denotes all memory entries except $\mathbf { M } _ { i }$ . The reconstructed memory is then obtained as $\hat { \mathbf { M } } _ { i } = \mathbf { M } _ { - i } ^ { \top } \mathbf { w } _ { i } ^ { * }$ . The reconstruction error is calculated by $e _ { i } = \left\| \hat { \mathbf { M } } _ { i } - \mathbf { M } _ { i } \right\| _ { 2 } ^ { 2 }$ . A smaller reconstruction error indicates that $\mathbf { M } _ { i }$ can be well represented by other memory entries and is therefore more redundant. Accordingly, we define the reconstruction-based redundancy score as:

$$
r _ { i } = 1 - \frac { e _ { i } } { \sum _ { j = 1 } ^ { N } e _ { j } + \epsilon } .\tag{6}
$$

In addition to redundancy, we also consider the enhancement and fusion quality. Given the image-level loss $\mathcal { L } _ { i }$ corresponding to the frame stored in the LTMB, we define the quality score as:

$$
q _ { i } = \frac { \mathcal { L } _ { i } } { \sum _ { j = 1 } ^ { N } \mathcal { L } _ { j } + \epsilon } ,\tag{7}
$$

where a larger $q _ { i }$ indicates poorer fusion quality and thus a higher priority for removal. Finally, the overall removal score is computed as

$$
s _ { i } = r _ { i } + q _ { i } .\tag{8}
$$

The memory entry with the maximum score $i ^ { * } = \arg \operatorname* { m a x } _ { i } s _ { i }$ is removed, and the LTMB is updated by discarding $\mathbf { M } _ { i ^ { * } }$ while keeping the remaining memory entries. In this way, the LTMB preserves memory entries that are both unique and beneficial to fusion quality, thereby maintaining a compact yet informative long-term representation.

## 2.3 Temporal Smoothing Module (TSM)

To prevent obvious background variations between consecutive frames caused by the integration of differing memory bank features, we introduce a soft adaptive Temporal Smoothing Module (TSM) to implement coarse temporal consistency, which adaptively balances the reliance on current and previous frame information across motion and static regions. Specifically, TSM first estimates a soft motion mask by analyzing inter-frame variations in the original video sequence. Since the infrared modality is often contaminated by noise, which may be erroneously interpreted as motion, the motion estimation is instead performed using the visible video features. Given two consecutive visible-frame features, the difference map is computed as: $D _ { t } = \Vert \mathcal { F } _ { t } ^ { \mathrm { v i s } } - \mathcal { F } _ { t - 1 } ^ { \mathrm { v i s } } \Vert _ { 1 }$ <sub>1</sub>, where $\| \cdot \|$ <sub>1</sub> denotes the $l _ { 1 }$ norm. After normalization, a soft motion probability map $M _ { t }$ is obtained via a sigmoid activation: $M _ { t } = \sigma ( n o r m ( D _ { t } ) )$ . To further suppress boundary artifacts and enhance spatial smoothness, the motion map is refined through mean-kernel blurring: $\tilde { M } _ { t } = K * M _ { t }$ , where K denotes a mean filter.

Using the resulting soft mask $\tilde { M _ { t } }$ , the adaptive smoothing coefficient is constructed as:

$$
\alpha _ { t } = \alpha _ { \mathrm { m o t i o n } } \cdot \tilde { M } _ { t } + \alpha _ { \mathrm { s t a t i c } } \cdot ( 1 - \tilde { M } _ { t } ) ,
$$

where $\alpha _ { \mathrm { { m o t i o n } } }$ and $\alpha _ { \mathrm { s t a t i c } }$ control the degree of temporal filtering for motion and static regions, respectively. Finally, the feature output of the current frame is given by:

$$
\tilde { \mathcal { F } } _ { t } = \alpha _ { t } \odot \mathcal { F } _ { t } + \left( 1 - \alpha _ { t } \right) \odot \mathcal { F } ^ { \mathrm { p r e v } } ,
$$

where $\tilde { \mathcal { F } } _ { t }$ will serve as the previous frame feature ${ \mathcal { F } } ^ { \mathrm { p r e v } }$ in the computation of the next frame.

The whole first stage is constrained by temporal-level consistency loss ${ \mathcal { L } } _ { \mathrm { t } }$ , and image-level quality loss $\mathcal { L } _ { \mathrm { g r a d } } , \mathcal { L } _ { \mathrm { c o l o r } }$ and $\mathcal { L } _ { \mathrm { { c o n t e n t } } }$ following [6], concluded as:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { t } } + \lambda _ { 1 } \mathcal { L } _ { \mathrm { g r a d } } + \lambda _ { 2 } \mathcal { L } _ { \mathrm { c o l o r } } + \lambda _ { 3 } \mathcal { L } _ { \mathrm { c o n t e n t } } . } \end{array}\tag{9}
$$

## 2.4 Temporal Refinement Module (TRM)

Although the previous TSM coarsely enhances temporal consistency, the overall temporal coherence of the network remains largely constrained by the 2D convolution-based encoder-decoder and other spatially oriented structures, which are inherently limited in modeling temporal dynamics. Therefore, we design the second-stage Temporal Refinement Module (TRM) to further refine temporal consistency in a fine-grained and efficient manner.

We assume that the model has already learned high-quality fused feature representations in the first stage. With this strong fusion prior, the TRM is deliberately kept lightweight, consisting of only several stacked 3D separable convolutions to capture subtle temporal variations without excessive computational overhead. To further uncover the informative details overlooked in the first stage, both the fused outputs $V _ { f }$ and the original visible and infrared sequences $( V _ { \mathrm { v i s } } , V _ { \mathrm { i r } } )$ are fed into this module. A residual learning strategy is employed to inherit and refine the performance of the first stage, leading to a stable and progressive improvement in temporal coherence.

This process can be mathematically formulated as:

$$
\begin{array} { r } { \tilde { V } _ { f } = T R M ( V _ { f } , V _ { v i s } , V _ { i r } ) + V _ { f } , } \end{array}\tag{10}
$$

where $T R M ( \cdot )$ denotes the 3D convolution-based refinement network. In this stage, we adopt an optimization objective composed of a gradient loss $\mathcal { L } _ { \mathrm { g r a d } }$ to preserve fine spatial textures and a temporal loss $\mathcal { L } _ { \mathrm { t } }$ to enforce inter-frame consistency. This supervision enables the TRM to recover spatio-temporal dynamics overlooked by the first-stage fusion process.

## 3 Experiments

In this section, we first introduce the experimental setup, including the comparison methods, evaluation metrics, hyperparameters, and other implementation details. Subsequently, we present the quantitative and qualitative results on various benchmark datasets. Finally, we perform ablation studies to demonstrate the effectiveness of each designed module in our method.

## 3.1 Implementation Details

Datasets and Metrics We conduct experiments on two infrared and visible video datasets, namely M3SVD [15] and HDO [17], both of which contain pre-registered video pairs. The M3SVD dataset consists of 220 pairs of video sequences with frame counts ranging from 100 to 1000, while the HDO dataset includes 26 pairs of sequences, each containing 150 to 200 frames. Our method is trained only on the M3SVD dataset and tested on both datasets. To comprehensively evaluate the model’s performance, we employ MI, CC, SCD and VIF to assess the fusion quality, while flowD and feaCD [6] are used to measure the temporal consistency of the fused videos.

Comparison Methods Since video fusion methods are still in their early stage of development, we select five state-of-the-art image fusion algorithms and three video fusion methods for comparison. The image fusion algorithms include LRRNet [7], Diff-IF [20], MMIF-EMMA [26], Text-DiFuse [22], and TC-MoA [29], while the video fusion methods include TemCoCo [6], VideoFusion [15] and UniVF [27]. To ensure a fair comparison, for those fusion models lacking integrated de-degradation modules, we apply DSTNet [14] to deblur the visible videos and MDIVDNet [1] to denoise the infrared videos before fusion. Both preprocessing works have been fully retrained on our datasets. All comparative video fusion methods are evaluated using the standard number of frames specified in their official implementations.

![](images/e206fbfec4268004a6f43ef5bfed991f89132a45a207f537d3f33f8b53530f84.jpg)  
Figure 4: Qualitative results on M3SVD dataset under degradation.

Table 1: Quantitative results on the M3SVD and HDO datasets. Bold indicates the best result.
<table><tr><td rowspan="2">Method</td><td colspan="5">M3SVD</td><td colspan="5">HDO</td></tr><tr><td>MI↑</td><td>CC↑ SCD↑</td><td>VIF↑</td><td>flowD↓</td><td> feaCD↓</td><td>MI↑</td><td>CC↑ SCD↑</td><td>VIF↑</td><td></td><td>flowD↓ feaCD↓</td></tr><tr><td>LRRNet</td><td></td><td>1.78890.45841.4884 0.3906</td><td></td><td>5.5065</td><td>1.9309</td><td>1.0616 0.6725 0.7250 0.24167.4598</td><td></td><td></td><td></td><td>1.9523</td></tr><tr><td>Diff-IF</td><td>2.0692 0.4912 1.6068 0.48474.0890</td><td></td><td></td><td></td><td>1.8210</td><td>1.1387 0.68511.1483 0.30398.0450</td><td></td><td></td><td></td><td>1.9587</td></tr><tr><td>MMIF-EMMA</td><td>2.11750.47581.64430.47584.6592</td><td></td><td></td><td></td><td>1.8412</td><td>1.1668 0.69451.2582 0.30717.9598 1.9565</td><td></td><td></td><td></td><td></td></tr><tr><td>Text-DiFuse</td><td>1.6253 0.45761.4989 0.35148.6578</td><td></td><td></td><td></td><td>1.9744</td><td>1.1682 0.70241.4096 0.29409.72621.9732</td><td></td><td></td><td></td><td></td></tr><tr><td>TC-MoA</td><td>2.01780.46601.5276 0.40544.1954</td><td></td><td></td><td></td><td>1.8118</td><td>1.0942 0.70231.1639 0.31817.81251.9570</td><td></td><td></td><td></td><td></td></tr><tr><td>TemCoCo</td><td>2.1717 0.48001.59610.47703.1882</td><td></td><td></td><td></td><td>1.8058</td><td>1.3212 0.71921.3689 0.38235.11471.7336</td><td></td><td></td><td></td><td></td></tr><tr><td>VideoFusion</td><td>2.03530.46121.52990.41263.4161</td><td></td><td></td><td></td><td>1.8154</td><td>1.14280.71061.31520.32386.0564</td><td></td><td></td><td></td><td>1.8643</td></tr><tr><td>UniVF</td><td>2.1316 0.48481.6122 0.46774.0271</td><td></td><td></td><td></td><td>1.8138</td><td>1.1854 0.72051.3266 0.34276.7987</td><td></td><td></td><td></td><td>1.8680</td></tr><tr><td>Ours-stage1</td><td>2.16310.48721.5975 0.48773.6135</td><td></td><td></td><td></td><td>1.8088</td><td>1.3261 0.73191.4002 0.37336.6011</td><td></td><td></td><td></td><td>1.7747</td></tr><tr><td>MemoryFusion</td><td>2.2350 0.4952 1.6152 0.49262.67591.7640</td><td></td><td></td><td></td><td></td><td></td><td>1.3264 0.73471.4450 0.39474.91881.7293</td><td></td><td></td><td></td></tr></table>

Experimental Settings We empirically set the hyperparameters as $\alpha _ { m o t i o n } ~ = ~ 0 . 7 , \alpha _ { s t a t i c } ~ =$ $0 . 3 , \lambda _ { 1 } = 5 , \lambda _ { 2 } = 1 0$ and $\lambda _ { 3 } ~ = ~ 5$ . During training, the batch size is set to 2, and each batch contains 30 frames. During testing, 200 frames are used for each video sequence. All experiments are conducted on the PyTorch platform using a 2.0 GHz Intel Xeon Gold 6330 CPU and an NVIDIA A800 80 GB PCIe GPU.

## 3.2 Video Fusion

In this part, we present the experimental results on M3SVD and HDO datasets, including both quali tative and quantitative results, to comprehensively demonstrate the effectiveness of MemoryFusion.

Results on M3SVD Figure 4 compares different methods on the M3SVD dataset. As shown, the proposed MemoryFusion exhibits clear advantages in both detail restoration and contrast preservation. Specifically, frame-based fusion methods fail to preserve complete and realistic human structures or suffer from noticeable detail blurring. Among video-based methods, TemCoCo and VideoFusion remain limited by low contrast or misalignment artifacts, while UniVF, despite sharper structures, is still inferior. In the yellow box at the lower right, our method reconstructs window structures with the highest contrast among all competitors. Table 1 reports the quantitative results on M3SVD. Our method achieves the best performance on all image quality metrics (MI, CC, SCD, VIF) as well as temporal consistency metrics (flowD, feaCD), demonstrating superior spatial fidelity and temporal coherence. While TemCoCo and VideoFusion show relatively strong temporal consistency due to video-sequence modeling, UniVF exhibits weaker temporal coherence without such mechanisms. Notably, our first-stage model already surpasses image-based methods due to temporal consistency loss and TSM, though frame-wise processing limits its temporal performance. With the lightweight second stage, temporal coherence is effectively restored and image quality is further improved.

Results on HDO Figure 5 presents the qualitative results on the HDO dataset. In the enlarged yellow box at the upper right, similar to the previous results, all compared methods except TemCoCo and our proposed MemoryFusion fail to preserve the complete human structure, where TC-MoA and UniVF exhibit noticeable color distortion, and Diff-IF, MMIF-EMMA and VideoFusion produce blurred or fragmented contours. In contrast, both TemCoCo and our approach maintain more coherent human outlines, yet TemCoCo still suffers from residual blur and local detail loss. Beyond the structural regions, the enlarged blue box at the lower right highlights differences in fine texture reconstruction. Our MemoryFusion yields sharper edges and more faithful contrast, effectively restoring spatial details that are heavily degraded in other methods. The quantitative results summarized in Table 1 further confirm these observations, showing that MemoryFusion consistently achieves superior scores across all metrics, validating its robustness and effectiveness under severe degradation conditions.

![](images/ef9938723edd579ddf7303e190c67adf4a497683edc1277ddf989f808c3900a6.jpg)  
Figure 5: Qualitative results on HDO dataset under degradation.

Temporal Consistency Comparison To evaluate temporal consistency, we concatenate the same pixel row from 60 consecutive frames for visualization, as shown in Figure 6. Existing methods suffer from temporal artifacts. For instance, LRRNet and Text-DiFuse show ghosting and tearing effects, while Diff-IF and MMIF-EMMA exhibit structural discontinuities. Although TemCoCo, VideoFusion, and UniVF reduce some distortions via frame alignment, they still struggle with smooth temporal transitions. In contrast, MemoryFusion pro-

![](images/b99f69fddd97595eb3ff9bbebd046370f76f0157cb1dcf77c8a0d170391c6e21.jpg)  
Figure 6: Temporal consistency comparison.

duces temporally coherent and visually stable results with well-preserved structures, demonstrating its effectiveness in modeling temporal dependencies and maintaining spatial-temporal stability.

## 3.3 Ablation Studies

In this section, we conduct comprehensive ablation studies to validate the effectiveness of each designed component in MemoryFusion. Specifically, we provide qualitative evaluations from both image quality and temporal consistency perspectives, as shown in Figure 7, and report the correspond ing quantitative results in Table 2. It is worth noting that, except for the two variants related to the second refinement stage (i.e., w/o stage2 tloss and w/o stage2 Conv3D), all other ablation settings are evaluated only within the first stage. Therefore, these variants are compared with the stage-1 baseline rather than the complete MemoryFusion model, so as to more accurately reflect the contribution of each stage-1 component.

w/o TSM This setting removes the Temporal Smoothing Module (TSM). The TSM leverages the previous frame to smooth the current frame, reducing obvious inconsistency between consecutive frames. As shown in Figure 7 and Table 2, omitting this module has little effect on image quality but results in temporal flickering, confirming its role in enhancing temporal smoothness. This indicate that TSM mainly contributes to inter-frame stability.

w/o STMB This experiment disables the Short-Term Memory Bank (STMB). Without it, the model cannot effectively reuse short-term contextual cues from recent frames to guide current-frame fusion.

![](images/f7a2bf22502b38dbaf44eb19df257b5ab764271f5862faa2e45e871d1ebb167c.jpg)  
Figure 7: Qualitative results of ablation experiments.

As shown in the figure, the result shows blurred textures and weakened local contrast. These results confirm that STMB plays a vital role in reinforcing details during enhancement and fusion. This suggests that recent temporal information is beneficial for preserving local structures and fine textures.

Random LTMB In this configuration, the Long-Term Memory Bank (LTMB) is updated randomly rather than by the proposed dual-criterion rule. Such stochastic replacement disrupts the accumulation of informative long-term cues, leading to unstable global guidance. The results highlight the importance of an effective update mechanism for maintaining meaningful long-term memory. This further shows that the benefit of LTMB comes not only from memory storage, but also from selective memory updating.

w/o LTMB This variant removes the LTMB to examine the contribution of global memory aggregation. Without access to long-term frame information, the model is limited to short-range temporal cues from the STMB and fails to capture global contextual relations and scene structures. As a result, the fused results exhibit degraded structural integrity and weakened detail representation, especially in complex or spatially diverse regions, demonstrating the effectiveness of LTMB. This verifies that long-term memory provides complementary global cues beyond short-term temporal memory.

w/o stage2 tloss & w/o stage2 Conv3D These two variants evaluate the temporal loss and 3D convolution in the second refinement stage. Removing the temporal loss weakens the explicit supervision on inter-frame consistency, leading to less stable transitions. For the w/o stage2 Conv3D variant, we replace the 3D convolution with 2D convolution, which preserves spatial refinement but removes explicit

Table 2: Quantitative results of ablation experiments.
<table><tr><td>M3SVD</td><td>MI↑</td><td>CC↑</td><td>SCD↑</td><td>VIF↑</td><td>flowD↓ feaCD↓</td><td></td></tr><tr><td>w/o TSM</td><td></td><td></td><td>2.01730.48211.57340.49434.2172</td><td></td><td></td><td>1.8038</td></tr><tr><td>w/o STMB</td><td></td><td></td><td>2.02390.47671.55380.48173.65861.8061</td><td></td><td></td><td></td></tr><tr><td>random LTMB</td><td></td><td></td><td>2.01990.47551.5580 0.48253.70101.8036</td><td></td><td></td><td></td></tr><tr><td>w/o LTMB</td><td></td><td></td><td>2.00310.46461.55090.47703.77681.8078</td><td></td><td></td><td></td></tr><tr><td>w/o stage2 tloss</td><td></td><td></td><td>2.19870.48841.59610.49323.20651.7985</td><td></td><td></td><td></td></tr><tr><td>w/o stage2 Conv3D</td><td></td><td></td><td>2.10070.49421.59940.49383.5290</td><td></td><td></td><td>1.7956</td></tr><tr><td>Ours-stage1</td><td></td><td></td><td>2.16310.48721.59750.48773.6135</td><td></td><td></td><td>1.8088</td></tr><tr><td>MemoryFusion</td><td></td><td></td><td>2.2350 0.49521.6152 0.49262.6759 1.7640</td><td></td><td></td><td></td></tr></table>

temporal modeling across adjacent frames. As shown in Table 2, 3D convolution plays a more decisive role in maintaining temporal stability, demonstrating that the joint spatio-temporal refinement in the second stage is crucial for stabilizing video fusion results.

## 3.4 Plug-and-Play Spatiotemporal Refinement with TRM

To further investigate the generalization and effectiveness of the proposed TRM, we apply it to two existing fusion methods, MMIF-EMMA and UniVF. Specifically, the original fusion models are kept fixed, and their fused outputs are directly used as fusion priors for TRM. This allows TRM to perform spatiotemporal refinement without modifying the original fusion networks, demonstrating its flexibility in enhancing existing fusion methods.

As shown in Table 3, introducing TRM significantly improves the temporal consistency of both methods, achieving relative reductions in flowD of 36.6% and 30.4% for MMIF-EMMA and UniVF, respectively. In addition to these substantial improvements in temporal consistency, TRM also enhances fusion quality in terms of CC and SCD for both methods, while maintaining comparable performance on other fusion metrics. These results demonstrate that TRM can effectively enhance temporal consistency while preserving or improving spatial fusion quality, even when applied to fusion models without memory mechanisms.

Furthermore, the complete MemoryFusion achieves the best temporal consistency and superior overall fusion performance, suggesting that the memory-enhanced Stage-1 fusion prior provides a favorable foundation for subsequent refinement. More importantly, TRM achieves substantial improvements when directly applied to other fusion methods without requiring additional training or finetuning, highlighting its plug-and-play

Table 3: Quantitative comparison of applying TRM to different fusion methods. The best results are highlighted in bold.
<table><tr><td>Method</td><td>MI↑</td><td>CC↑</td><td>SCD↑</td><td>VIF↑</td><td>flowD↓</td><td>feaCD↓</td></tr><tr><td>MMIF-EMMA</td><td>2.1175</td><td>0.4758</td><td>1.6443</td><td>0.4758</td><td>4.6592</td><td>1.8412</td></tr><tr><td>MMIF-EMMA+TRM</td><td>2.0641</td><td>0.4932</td><td>1.6624</td><td>0.4762</td><td>2.9549</td><td>1.8125</td></tr><tr><td>UniVF</td><td>2.1316</td><td>0.4848</td><td>1.6122</td><td>0.4677</td><td>4.0271</td><td>1.8138</td></tr><tr><td>UniVF+TRM</td><td>2.0337</td><td>0.4921</td><td>1.6325</td><td>0.4673</td><td>2.8025</td><td>1.8122</td></tr><tr><td>Ours-stage1</td><td>2.1631</td><td>0.4872</td><td>1.5975</td><td>0.4877</td><td>3.6135</td><td>1.8088</td></tr><tr><td>MemoryFusion</td><td>2.2350</td><td>0.4952</td><td>1.6152</td><td>0.4926</td><td>2.6759</td><td>1.7640</td></tr></table>

capability. These results demonstrate the general applicability of TRM as a flexible spatiotemporal refinement module that can effectively enhance fusion quality and, more importantly, temporal consistency, with the potential for further improvements through finetuning.

## 3.5 Computation Efficiency Comparison

We compare the computational efficiency in terms of FLOPs, parameter count, and runtime in Table 4. Although MemoryFusion introduces an additional second stage, the overall computational cost remains moderate. Specifically, the first-stage model achieves competitive efficiency compared with existing video-based fusion methods, while the lightweight second stage incurs only negligible overhead. For preprocessing-based methods, the reported values only account for the computational cost of the fusion networks, excluding additional pretrained deblurring or denoising modules. Therefore, the comparison reflects the intrinsic efficiency of the fusion networks rather than the computational cost of the complete preprocessing-and-fusion pipelines. Overall, our full model strikes a favorable balance between fusion quality and computational efficiency, outperforming heavy diffusion-based or large-scale temporal models while maintaining comparable inference speed.

Table 4: Comparison of computational efficiency.
<table><tr><td colspan="10">LRRNet Diff-IF MMIF-EMMA Text-DiFuse TC-MoA TemCoCo VideoFusion UniVF|Stage1 Stage2</td></tr><tr><td>FLOPs (G)</td><td>14.2</td><td>86.9</td><td>41.5</td><td>1268</td><td>4833</td><td>505.6</td><td>301.1</td><td>740.7</td><td>213.6 1.65</td></tr><tr><td>Param (M)</td><td>0.047</td><td>22.36</td><td>1.52</td><td>119.4</td><td>340</td><td>19.18</td><td>6.74</td><td>9.15</td><td>17.2 0.033</td></tr><tr><td>Time (s)</td><td>0.162</td><td>4.23</td><td>0.056</td><td>24.2</td><td>0.185</td><td>0.109</td><td>0.105</td><td>0.245</td><td>0.099 0.001</td></tr></table>

## 4 Conclusion

This paper proposes MemoryFusion, a two-stage cross-temporal multimodal video fusion framework designed to achieve high-fidelity fusion and superior temporal consistency. In the first stage, dual memory banks dynamically update and effectively retrieve informative cues from historical frames to enhance current-frame fusion. Additionally, a TSM is incorporated to adaptively mitigate coarse frame variations and establish a stable temporal foundation. In the second stage, guided by the powerful fusion prior from the first stage, a lightweight TRM further enforces fine-grained temporal coherence and enhances spatial fidelity through residual refinement. Extensive experiments on multiple public datasets demonstrate that MemoryFusion not only achieves state-of-the-art performance in both fusion quality and temporal consistency but also exhibits strong generalization across diverse scenes.

## Acknowledgments and Disclosure of Funding

This work was supported by the National Natural Science Foundation of China under Grant no. 62676289, and the China Postdoctoral Science Foundation under no. 2026T190426. The authors declare no competing interests.

## References

[1] Lijing Cai, Xiangyu Dong, Kailai Zhou, and Xun Cao. Exploring video denoising in thermal infrared imaging: Physics-inspired noise generator, dataset and model. IEEE Transactions on Image Processing, 33:3839–3854, 2024.

[2] Bing Cao, Yiming Sun, Pengfei Zhu, and Qinghua Hu. Multi-modal gated mixture of local-to-global experts for dynamic image fusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 23555–23564, 2023.

[3] Wanyi Chen, Zihua Zhao, Jiangchao Yao, Ya Zhang, Jiajun Bu, and Haishuai Wang. Multi-modal medical diagnosis via large-small model collaboration. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 30763–30773, 2025.

[4] Ho Kei Cheng and Alexander G Schwing. Xmem: Long-term video object segmentation with an atkinson shiffrin memory model. In European conference on computer vision, pages 640–658. Springer, 2022.

[5] Meiqi Gong, Hao Zhang, Hebaixu Wang, Jun Chen, Jun Huang, Xin Tian, and Jiayi Ma. Cross-scale domain adaptation with comprehensive information for pansharpening. In Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, pages 785–793, 2024.

[6] Meiqi Gong, Hao Zhang, Xunpeng Yi, Linfeng Tang, and Jiayi Ma. Temcoco: Temporally consistent multimodal video fusion with visual-semantic collaboration. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 14326–14335, 2025.

[7] Hui Li, Tianyang Xu, Xiao-Jun Wu, Jiwen Lu, and Josef Kittler. Lrrnet: A novel representation learning guided fusion network for infrared and visible images. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(9):11040–11052, 2023.

[8] Xuan Li, Rongfu Chen, Jie Wang, Lei Ma, Li Cheng, and Haiwen Yuan. Dstcfuse: A method based on dual-cycled cross-awareness of structure tensor for semantic segmentation via infrared and visible image fusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5558–5567, 2024.

[9] Jinyuan Liu, Zhu Liu, Guanyao Wu, Long Ma, Risheng Liu, Wei Zhong, Zhongxuan Luo, and Xin Fan. Multi-interactive feature learning and a full-time multi-modality benchmark for image fusion and segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8115–8124, 2023.

[10] Jinyuan Liu, Runjia Lin, Guanyao Wu, Risheng Liu, Zhongxuan Luo, and Xin Fan. Coconet: Coupled contrastive learning network with multi-level feature ensemble for multi-modality image fusion. International Journal ofComputer Vision, 132(5):1748–1775, 2024.

[11] Fuya Luo, Yunhan Li, Guang Zeng, Peng Peng, Gang Wang, and Yongjie Li. Thermal infrared image colorization for nighttime driving scenes with top-down guided attention. IEEE Transactions on Intelligent Transportation Systems, 23(9):15808–15823, 2022.

[12] Jiayi Ma, Linfeng Tang, Fan Fan, Jun Huang, Xiaoguang Mei, and Yong Ma. Swinfusion: Cross-domain long-range learning for general image fusion via swin transformer. IEEE/CAA Journal ofAutomatica Sinica, 9(7):1200–1217, 2022.

[13] Zhong-Zheng Peng, Yi-Xin Yang, Jin-Hui Tang, and Jin-Shan Pan. Video colorization: A survey. Journal ofComputer Science and Technology, 39(3):487–508, 2024.

[14] Chen Rao, Guangyuan Li, Zehua Lan, Jiakai Sun, Junsheng Luan, Wei Xing, Lei Zhao, Huaizhong Lin, Jianfeng Dong, and Dalong Zhang. Rethinking video deblurring with wavelet-aware dynamic transformer and diffusion model. In Proceedings of the European Conference on Computer Vision, pages 421–437, 2024.

[15] Linfeng Tang, Yeda Wang, Meiqi Gong, Zizhuo Li, Yuxin Deng, Xunpeng Yi, Chunyu Li, Han Xu, Hao Zhang, and Jiayi Ma. Videofusion: A spatio-temporal collaborative network for multi-modal video fusion and restoration. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, 2026.

[16] Xiao Wu, Zi-Han Cao, Ting-Zhu Huang, Liang-Jian Deng, Jocelyn Chanussot, and Gemine Vivone. Fully-connected transformer for multi-source image fusion. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(3):2071–2088, 2025.

[17] Housheng Xie, Meng Sang, Yukuan Zhang, Yang Yang, Shan Zhao, and Jianbo Zhong. Rcvs: A unified registration and fusion framework for video streams. IEEE Transactions on Multimedia, 26:11031–11043, 2024.

[18] Jun Xie, Jiazhen Dou, Liyun Zhong, Jianglei Di, and Yuwen Qin. A dual-mode intensity and polarized imaging system for assisting autonomous driving. IEEE Transactions on Instrumentation and Measurement, 73:1–13, 2024.

[19] Han Xu, Meiqi Gong, Xin Tian, Jun Huang, and Jiayi Ma. Cufd: An encoder–decoder network for visible and infrared image fusion based on common and unique feature decomposition. Computer Vision and Image Understanding, 218:103407, 2022.

[20] Xunpeng Yi, Linfeng Tang, Hao Zhang, Han Xu, and Jiayi Ma. Diff-if: Multi-modality image fusion via diffusion model with fusion knowledge prior. Information Fusion, 110:102450, 2024.

[21] Xinyi Ying, Longguang Wang, Yingqian Wang, Weidong Sheng, Wei An, and Yulan Guo. Deformable 3d convolution for video super-resolution. IEEE Signal Processing Letters, 27:1500–1504, 2020.

[22] Hao Zhang, Lei Cao, and Jiayi Ma. Text-difuse: An interactive multi-modal image fusion framework based on text-modulated diffusion model. Advances in Neural Information Processing Systems, 37:39552–39572, 2024.

[23] Hao Zhang, Linfeng Tang, Xinyu Xiang, Xuhui Zuo, and Jiayi Ma. Dispel darkness for better fusion: A controllable visual enhancer based on cross-modal conditional adversarial learning. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26487–26496, 2024.

[24] Hao Zhang, Lei Cao, Xuhui Zuo, Zhenfeng Shao, and Jiayi Ma. Omnifuse: Composite degradation-robust image fusion with language-driven semantics. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(9):7577–7595, 2025.

[25] Zixiang Zhao, Haowen Bai, Jiangshe Zhang, Yulun Zhang, Shuang Xu, Zudi Lin, Radu Timofte, and Luc Van Gool. Cddfuse: Correlation-driven dual-branch feature decomposition for multi-modality image fusion. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5906–5916, 2023.

[26] Zixiang Zhao, Haowen Bai, Jiangshe Zhang, Yulun Zhang, Kai Zhang, Shuang Xu, Dongdong Chen, Radu Timofte, and Luc Van Gool. Equivariant multi-modality image fusion. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 25912–25921, 2024.

[27] Zixiang Zhao, Haowen Bai, Bingxin Ke, Yukun Cui, Lilun Deng, Yulun Zhang, Kai Zhang, and Konrad Schindler. A unified solution to video fusion: From multi-frame learning to benchmarking. Advances in Neural Information Processing Systems, 2025.

[28] Junbao Zhou, Ziqi Pang, and Yu-Xiong Wang. Rmem: Restricted memory banks improve video object segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18602–18611, 2024.

[29] Pengfei Zhu, Yang Sun, Bing Cao, and Qinghua Hu. Task-customized mixture of adapters for general image fusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7099–7108, 2024.

## A Supplemental Material

## A.1 Experimental Details

MemoryFusion directly takes the raw infrared-visible video sequences as input without using external de-degradation preprocessing. For baseline methods that do not contain degradation-removal components, we additionally apply DSTNet/MDIVDNet as preprocessing only for robustness evaluation under degraded inputs. Unless otherwise stated, the FLOPs, parameters, and inference time reported for these baseline fusion networks do not include the additional computational cost of DSTNet/MDIVDNet. Therefore, the efficiency comparison is favorable to the baselines, while MemoryFusion remains competitive without relying on external preprocessing.

## A.2 Extension to Medical Scenes

We further extend the proposed model to medical sequence fusion to evaluate its generalization capability to medical scenes. Specifically, experiments are conducted on the Harvard medical imaging dataset<sup>0</sup>, where the MRI and PET sequences consist of a series of consecutive slice images acquired from cross-sectional scans of the same organ. Owing to the strong anatomical correlation between adjacent slices, the spatial continuity among them is analogous to the temporal coherence between consecutive frames in a video. Hence, we treat each MRI or PET slice as a video frame to evaluate the model’s performance in medical sequence fusion. Since medical scenes are relatively homogeneous, this experiment primarily focuses on assessing the fusion performance. As shown in Figure s1, methods such as LRRNet, Text-DiFuse and VideoFusion tend to blur fine structural boundaries or over-smooth functional details, while Diff-IF and MMIF-EMMA exhibit noticeable inconsistencies between structural and metabolic regions. Despite their competence in detail preservation, TC-MoA and UniVF fail to retain the color information of PET images, resulting in severe spectral distortion. In contrast, our MemoryFusion framework effectively integrates structural information from MRI with metabolic patterns from PET, achieving both high structural fidelity and functional contrast. The highlighted regions further demonstrate that our approach preserves sharp anatomical edges and detailed texture in gray matter areas while maintaining accurate PET intensity distributions, thereby providing more clinically interpretable fused results.

![](images/a7e00637e3a81bb15544966278ec5e96a82391f9d02e367bfbff4ed8be575bc5.jpg)  
Figure s1: Results of medical sequence fusion on Harvard dataset.

## A.3 Results of Non-degraded Scenes

Figure s2 shows the qualitative results on non-degraded scenes. MemoryFusion enhances visual contrast and salient structural cues. In the enlarged yellow box at the upper left, most methods fail to clearly separate the human from the background, while TemCoCo and VideoFusion introduce boundary artifacts due to over-enhancement. In contrast, MemoryFusion effectively incorporates thermal cues from the infrared modality, producing clearer human boundaries with well-preserved texture continuity. The enlarged green box at the lower right further compares texture reconstruction around the window region. Obviously, our MemoryFusion delivers sharper and more consistent structures with higher contrast. These results demonstrate that the proposed memory-enhanced modeling not only preserves global scene semantics but also improves fine-detail reconstruction, even under non-degraded conditions where other methods tend to over-smooth or distort details.

![](images/fef4b9ad52b96a55ccdf830c37d2b864cb053dfd9c8ca47ae15f14be71974ac1.jpg)  
Figure s2: Qualitative results of non-degraded scene.

## A.4 Limitation

Although MemoryFusion achieves strong fusion quality and temporal consistency, it is not explicitly designed as a degradation-specific restoration framework. Its robustness thus depends partly on the degradation coverage of the training data, and may decrease under severely unseen degradations. Moreover, since TSM uses visible-frame differences only as a lightweight temporal smoothing cue, when motion is mainly reflected in the infrared modality, the first-stage smoothing may provide limited guidance, placing greater pressure on the second-stage Temporal Refinement Module to restore temporal consistency. Nevertheless, these limitations are orthogonal to our core contribution, and degradation-aware modeling or reliability-aware bi-modal motion estimation can be naturally incorporated into MemoryFusion as additional modules in future work.