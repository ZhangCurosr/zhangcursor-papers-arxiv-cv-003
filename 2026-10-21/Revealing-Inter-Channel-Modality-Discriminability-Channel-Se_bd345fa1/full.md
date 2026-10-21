# Revealing Inter-Channel Modality Discriminability: Channel Selective Interaction Fusion Network for Optical-SAR Object Detection

Haojie Li, Yuanfei Huang, Member, IEEE, Lingfei Song, Jia Li, Lizhi Wang, Member, IEEE

Abstract—Synthetic aperture radar (SAR) and optical imagery provide complementary structural and appearance information for remote sensing object detection. However, substantial cross-modal discrepancies and indiscriminate channel interaction inevitably introduce redundant or conflicting features. Since different feature channels encode heterogeneous semantic information and contribute unequally to cross-modal fusion, selectively highlighting task-relevant channels while suppressing redundant ones is essential. To address this issue, we propose the Channel Selective Interaction Fusion Network (CSIF-Net), which performs selective channel-level cross-modal interaction. CSIF-Net first employs a frequency decomposition module to construct frequency-aware SAR inputs, facilitating the separation of structural information from noise-related responses. It then models cross-modal channel relationships through a crossattention channel selection module and dynamically partitions the features into shared and modality-specific components. Finally, a dual-branch fusion module performs cross-modal interaction and channel realignment for shared features while independently enhancing SAR-specific features, thereby exploiting complementary information and suppressing modality interference. Extensive experiments on the OGSOD 1.0 and M4-SAR datasets demonstrate that CSIF-Net achieves state-of-the-art performance in both horizontal and oriented object detection, surpassing the existing methods by 4.01% and 3.18% gains, respectively.

Index Terms—optical-SAR fusion, multimodal object detection, cross-modal attention, modality feature partitioning.

## I. INTRODUCTION

YNTHETIC aperture radar (SAR) imagery is sensitive to S the electromagnetic properties of targets, enabling reliable imaging under diverse illumination and weather conditions but providing limited visual appearance information. In contrast, optical imagery contains clearer visual representations, but its quality is more susceptible to illumination and weather conditions [1], [2]. Therefore, SAR and optical imagery are highly complementary, and provide more comprehensive target representations [3]–[7]. Nevertheless, effective optical-SAR fusion remains challenging because of the substantial discrepancies between the two modalities, including differences in imaging mechanisms and visual appearance [3], [4], [8]–[11]. These discrepancies may hinder the extraction of consistent target representations and introduce irrelevant or conflicting information during feature fusion.

Existing studies mainly explore two directions of modality translation and cross-modal fusion. For modality translation, image translation and generative approaches are widely adopted [8], [9], [12], [13]. These methods employ generative models to transform heterogeneous data into a unified modality space, thereby facilitating knowledge transfer and reducing feature discrepancies caused by different imaging mechanisms. However, the generated representations are often optimized for visual consistency rather than downstream detection objectives, limiting their ability to fully exploit complementary cross-modal information. For cross-modal fusion, researchers have developed various strategies to enhance feature interaction and alignment across modalities, developing domain adaptation and cross-modal alignment methods [5], [10], [14]–[16]. Dual-stream backbone architectures [17], [18] further enable hierarchical feature extraction and interaction between SAR and optical modalities. Besides, attention-based methods [11], [19]–[25] adaptively model cross-modal feature dependencies to enhance complementary information while suppressing conflicting responses. More recently, Mamba-based architectures [26] have shown promising capability in capturing long-range cross-modal dependencies, while gated fusion mechanisms [3], [16], [21] dynamically regulate information flow during feature aggregation.

Although existing cross-modal feature alignment and fusion methods have partially alleviated modality interference, they typically model feature interactions mainly in the spatial domain and treat channel-wise features with either uniform importance or unimodal-driven attention. They lack an explicit cross-modal view of which channels carry shared, complementary, or modality-specific information [27], [28]. However, as illustrated in Fig. 1, different feature channels often encode heterogeneous semantic information, exhibit varying cross-modal attention responses, and contribute unequally to downstream detection. Indiscriminately aggregating all channels may introduce redundant representations and modalityspecific interference, thereby weakening the effectiveness of cross-modal interaction [29]. Therefore, selectively filtering and dynamically interacting with features at the channel level is crucial for emphasizing task-relevant information and fully exploiting the complementary characteristics of SAR and optical modalities.

To address the above issues and serve the SAR and optical cross-modal object detection task, we propose a feature-level fusion framework, called Channel Selective Interaction Fusion Network (CSIF-Net). First, a Fast Fourier Transform (FFT)- based Frequency Decomposition Module (FDM) separates the SAR input into multiple frequency components, facilitating the subsequent learning of structural and noise-related patterns. Next, the Cross-Attention Channel Selection Module (CACS) performs the sole channel embedding operation and constructs a cross-modal attention matrix to partition the channels into shared and specific sets. A straight-through estimator (STE) enables this discrete selection process to be optimized end-to-end. Finally, the Dual-Branch Fusion Module (DBF) applies differentiated processing to the indexed components by sharing features that undergo cross-modal interaction and channel alignment, while the SAR specific features are enhanced by self-attention before reconstruction. By jointly leveraging CACS and DBF, CSIF-Net identifies cross-modal channel relationships and performs differentiated feature interaction, thereby enhancing complementary information while suppressing channel redundancy and modalityspecific interference. The main contributions of this work are summarized as follows,

![](images/ac22246ec65aaf2957ba0c0d99a058ac358fd96f8b229e8dcf6a8804b9715976.jpg)  
(a) Averaged accumulated gradients during the training process

![](images/9f0555ddd4416b440e39fb5ec7b312ddc5e52c8a5c51dcd7120a665065719370.jpg)  
(b) Cross-modal channel correlation matrix  
Fig. 1. We employ a simple 1×1 convolutional network for fusion and object detection, and track the accumulated gradients of each channel during training. The top-64 channels with the largest accumulated gradients are selected from the OPT and SAR modalities respectively to compute their channel correlation matrix. As can be observed, the accumulated gradients vary substantially across channels. Moreover, the cross-modal channel correlation matrix exhibits a pronounced banded pattern, with significant differences in correlation across channels.

• We analyze the channel discrepancy in cross-modal fusion, revealing that different feature channels exhibit highly imbalanced cross-modal attention responses and contribute unequally to downstream detection, emphasizing the necessity of channel-level selective interaction.

• We propose a cross-modal selective fusion framework for optical-SAR object detection, which achieves channellevel cross-modal feature disentanglement and dynamic interaction through a cross-modal attention matrix.

• We design distinct strategies for shared and specific features, where shared features are fused via cross-attention with channel realignment, while the CACS embeddings of SAR specific features undergo self-attention and reconstruction.

## II. RELATED WORK

## A. SAR and Optical Cross-Modal Fusion Strategies

Recent surveys have summarized multimodal remote sensing fusion from the perspectives of fusion levels, protocols, datasets, and downstream applications [6], [7]. Owing to their complementarity, optical-SAR fusion has been widely studied in land cover classification, semantic segmentation, building extraction, and heterogeneous change detection. Existing studies have explored different strategies to bridge the modality gap between SAR and optical images. Liu et al. [4] reviewed deep feature fusion methods for optical and SAR semantic segmentation, emphasizing the importance of multimodal feature representation and cross-modal alignment. For land cover and land use classification, Ren et al. [17] proposed DDHRNet, which adopts a dual-stream high-resolution architecture to fuse GF-2 optical and GF-3 SAR features and improves classification performance under degraded optical conditions. MCANet [30] further introduces pseudo-Siamese feature extraction, multimodal cross-attention, and multi-level feature fusion for joint optical-SAR land use classification. Optical-SAR fusion has also been investigated in building extraction and change detection. Li et al. [10] proposed a progressive fusion learning framework for building extraction, where shared modal-invariant features are extracted as an information bridge for multistage optical-SAR fusion. For heterogeneous change detection, Jiang et al. [31] introduced deep homogeneous feature fusion to transform optical and SAR images into a comparable feature space while preserving semantic content. Liu et al. [14] proposed commonality feature representation learning to extract latent comparable features from multimodal bi-temporal images for unsupervised change detection.

These methods demonstrate the effectiveness of optical-SAR fusion, but they mainly target scene-level dense prediction tasks such as segmentation, classification, and change detection. Object detection additionally demands precise localization, instance-level discrimination, and robust multi-scale representation, so detection-oriented optical-SAR fusion that suppresses modality interference and exploits complementarity remains an open problem.

## B. Cross-Modal Image Fusion for Object Detection

Recent multimodal remote sensing detectors further explore correlation-guided progressive interaction, gated fusion, diffusion deformation, and knowledge distillation to improve optical-SAR feature complementarity and detection robustness [21], [32]. Cross-modal object detection aims to improve detection robustness by exploiting complementary information from heterogeneous modalities. In optical-SAR remote sensing, related studies are still relatively limited and mainly focus on cross-domain transfer, few-shot learning, and benchmark construction. Zhou et al. [5] proposed a domain-adaptive fewshot SAR ship detection method that transfers knowledge from optical images to SAR images through feature distance measurement and dual-stream feature alignment. Huang et al. [8] investigated domain adaptive oriented object detection from optical to SAR images, where pixel-level and instance-level information transfer are jointly used to bridge the modality gap. Yang et al. [33] further introduced cross-domain feature interaction and data contribution balance for unsupervised SAR ship detection. Recently, Wang et al. [34] constructed M4-SAR, a large-scale optical-SAR fusion object detection benchmark, and proposed a Mamba-based optical and SAR fusion algorithm which demonstrated good performance. Since optical-SAR object detection remains underexplored, infraredvisible and multispectral object detection methods provide useful references for general cross-modal detection [35]– [37]. Fang et al. [38] proposed a Cross-Modality Fusion Transformer to model intra- and inter-modality interactions between RGB and thermal features. Shen et al. [19] proposed ICAFusion, which uses iterative dual cross-attention transformers to capture complementary information across modalities. Helvig et al. [39] introduced CAFF-DINO by integrating cross-attention feature fusion into a Transformerbased detector.

Nevertheless, these detectors rely mainly on spatial interaction, global attention, or domain adaptation and seldom model channel-level cross-modal relationships, motivating more selective feature interaction.

## C. Channel-Level Feature Differential Processing

Channel-level feature modeling has been widely used to improve representational ability. Hu et al. [40] proposed the squeeze-and-excitation block, which models channel interdependencies to adaptively recalibrate channel responses; Li et al. [41] introduced Selective Kernel Networks to adaptively select receptive fields; Woo et al. [42] proposed CBAM, which sequentially infers channel and spatial attention; and Wang et al. [43] proposed ECA-Net, which uses efficient local cross-channel interaction without dimensionality reduction. These methods show that channels contribute unequally and that channel-wise recalibration can improve downstream performance. In multimodal fusion tasks, channel-level operations have also been explored to enhance complementary information and reduce computational cost. Cao et al. [29] proposed a lightweight channel switching and spatial attention module for multimodal object detection, which efficiently exchanges information between infrared and visible modalities while emphasizing informative spatial regions. Yuan et al. [44] designed C<sup>2</sup>Former for RGB-infrared object detection, where intermodality cross-attention is used to learn calibrated and complementary features across modalities. These methods indicate that channel-level or attention-based interaction is effective for improving cross-modal detection performance.

However, existing channel attention methods estimate importance from a single modality or globally recalibrate features without distinguishing shared from modality-specific channels in cross-modal fusion. In optical-SAR fusion this distinction is especially important, since SAR features contain both useful scattering structures and noise-related responses. Modeling channel-level cross-modal relationships and differentially processing shared and modality-specific features is therefore necessary for more selective and robust fusion.

## III. PROPOSED METHOD

The objective of cross-modal fusion is to exploit complementary information across modalities while suppressing redundant features. However, due to the substantial differences in imaging mechanisms, directly fusing all feature channels often introduces low-contribution or conflicting information that degrades the fused representation. To address this, we propose a cross-modal fusion framework based on frequencyaware prior enhancement and shared-specific feature disentanglement.

## A. Overall Framework

As illustrated in Fig. 2, the proposed framework consists of three major components: a Frequency Decomposition Module (FDM), a Cross-Attention Channel Selection Module (CACS), and a Dual-Branch Fusion Module (DBF).

Given the optical features $F _ { o p t }$ and SAR features $F _ { s a r }$ extracted by the backbone, the overall fusion process can be formulated as

$$
F _ { o u t } = \mathcal { F } _ { D B F } \big ( F _ { o p t } , F _ { s a r } ; \mathcal { F } _ { C A C S } ( F _ { o p t } , F _ { s a r } ) \big ) ,\tag{1}
$$

where $\mathcal { F } _ { C A C S } ( \cdot )$ returns the channel embeddings and selection indices I, while $\mathcal { F } _ { D B F } ( \cdot )$ retrieves the indexed components and performs dual-branch fusion.

The FDM is placed before the backbone to enrich the frequency representation of SAR inputs, while cross-modal fusion is performed between the backbone and the neck, so that shared information is deeply fused and modality-specific information is preserved before multi-scale aggregation. The core idea is to explicitly distinguish the roles of channels in cross-modal fusion: some channels encode shared semantics across modalities, while others carry modality-specific observations. Accordingly, CSIF-Net partitions channels into shared and specific sets via a cross-modal attention matrix and processes them differently: shared channels undergo cross-modal interaction and channel alignment, while specific channels are enhanced or retained within their own modality. In the following, we detail the FDM, CACS, and DBF in sequence.

## B. Frequency Decomposition Module

Due to the unique imaging mechanism of SAR, SAR images contain abundant speckle noise, with noise-related responses and structural information highly entangled in the spatial domain. To this end, the Frequency Decomposition Module (FDM) decomposes the SAR input in the frequency domain. A two-dimensional FFT is first applied to the SAR image X, and the spectrum is partitioned into low-, middle-, and highfrequency bands using two radial thresholds $r _ { 1 } < r _ { 2 }$ and the corresponding binary masks $M _ { l o w } , M _ { m i d } .$ , and $M _ { h i g h }$ . Each band is then transformed back to the spatial domain via the inverse FFT, and the resulting components are concatenated with the original image along the channel dimension:

![](images/69bdb0c0ba84686c00f197fbd8a1ce2ca516c904987fb90b96732cd4eb300fbf.jpg)  
Fig. 2. Overall architecture of the proposed CSIF-Net. FDM decomposes the SAR input into multiple frequency components to provide frequency-aware priors for feature extraction. CACS then models channel-wise correlations between the optical and SAR features and partitions them into shared and specific components. Finally, DBF enhances the shared components through cross-modal interaction while preserving modality-specific information in the specific components before multi-scale feature aggregation.

$$
X _ { o u t } = \mathrm { C o n c a t } \left( X _ { l o w } , X _ { m i d } , X _ { h i g h } , X \right) ,\tag{2}
$$

where $X _ { i } = \mathcal { F } ^ { - 1 } ( \mathcal { F } ( X ) \odot M _ { i } )$ for $i \in \{ l o w , m i d , h i g h \}$ . The low-, middle-, and high-frequency components preserve global structure, object contours and texture, and fine details together with noise, respectively. By presenting these components as separate inputs, FDM enables the backbone to learn their relative contributions and distinguish useful structures from noise without an explicit denoising operation.

## C. Cross-Attention Channel Selection Module

The relationships between channels from different modalities are inherently heterogeneous. While some channels exhibit clear counterparts across modalities, others primarily describe modality-dependent observations. Therefore, establishing channel-level correspondences before cross-modal interaction is essential for identifying the roles of different channels in the joint representation space. To this end, we propose a Cross-Attention Channel Selection Module (CACS), which constructs a cross-modal attention matrix between optical and SAR features and partitions channels according to their attention responses.

Each channel is first encoded into a compact semantic descriptor. This is the only embedding operation in the proposed framework. Considering the varying spatial resolutions across network stages, high-resolution features are compressed using depthwise convolution before linear projection, whereas low-resolution features are directly projected. The channel embeddings are

$$
\begin{array} { r } { E _ { m } = \Phi ( F _ { m } ) \in \mathbb { R } ^ { C _ { m } \times d } , \quad m \in \{ o p t , s a r \} . } \end{array}\tag{3}
$$

where Φ(·) denotes the channel embedding function and d is the embedding dimension. The embeddings are then used to construct a cross-modal attention matrix:

$$
A = \frac { Q _ { o p t } K _ { s a r } ^ { \top } } { \sqrt { d } } , \quad A \in \mathbb { R } ^ { C _ { o p t } \times C _ { s a r } } ,\tag{4}
$$

where $Q _ { o p t }$ and $K _ { s a r }$ are linear projections of $E _ { o p t }$ and $E _ { s a r } ,$ respectively, and the Softmax operation is performed over the SAR-channel dimension. Each element of A represents the cross-modal attention weight of an optical-SAR channel pair. For optical-channel scoring, the attention vectors are obtained row-wise from $A ;$ for SAR-channel scoring, they are obtained column-wise from the same matrix.

To characterize the cross-modal attention distributions, statistical descriptors are extracted from the attention responses:

$$
\begin{array} { r } { A _ { i } ^ { m } = \left\{ { A } _ { i , : } , \quad m = o p t , \right. } \\ { A _ { : , i } , \quad m = s a r , } \end{array}\tag{5}
$$

$$
\begin{array} { r } { s _ { i } ^ { m } = \left[ \operatorname* { m a x } ( A _ { i } ^ { m } ) , \mu ( A _ { i } ^ { m } ) , \mathrm { V a r } ( A _ { i } ^ { m } ) \right] . } \end{array}\tag{6}
$$

where $A _ { i } ^ { m }$ is the attention response vector of the i-th channel in modality m, and $\mu ( \cdot )$ and $\mathrm { V a r } ( \cdot )$ denote its mean and variance. The maximum captures the strongest cross-modal response, the mean provides a scale-dependent baseline (since the opposite-modality channel count varies across scales), and the variance reflects whether attention is concentrated on a few counterparts or spread uniformly. These statistics compactly describe the strength and concentration of crossmodal attention.

![](images/5dc951b4fff4dc210e7981ccc5efb367c54cffb3543251ab4a57f81254b2e451.jpg)  
Fig. 3. Detailed architectures and data flows of CACS and DBF. Blue, red, and purple denote the optical stream, SAR stream, and fused information or module flow, respectively. CACS constructs the cross-modal attention matrix and selects shared and specific channels through statistical scoring and Top-K selection. DBF performs cross-modal interaction and aligned grouped fusion on shared features, enhances SAR-specific features through self-attention, preserves optical-specific features, and concatenates the three outputs.

The extracted statistics are subsequently mapped to channel scores via

$$
p _ { i } ^ { m } = g ( s _ { i } ^ { m } ) ,\tag{7}
$$

where $g ( \cdot )$ denotes a lightweight scoring function and $p ^ { m } =$ $[ p _ { 1 } ^ { m } , \ldots , p _ { C _ { m } } ^ { m } ]$ contains the channel scores. The shared and specific indices are obtained by Top-K selection:

$$
\mathcal { T } _ { m } ^ { b } = \mathrm { T o p K } ( p ^ { m } , k _ { b } ) , \qquad b \in \{ s h , s p \}\tag{8}
$$

Following our experimental setting, $k _ { s h } ~ = ~ 0 . 2 5 C _ { m }$ and $k _ { s p } ~ = ~ 0 . 7 5 C _ { m }$ . A STE is adopted: the hard indices are used for branch routing in the forward pass, while gradients are approximated through the continuous scores $p ^ { m }$ during backpropagation.

## D. Dual-Branch Fusion Module

After obtaining the shared and specific indices, how to exploit the corresponding components becomes critical. Existing methods typically fuse all channels uniformly, which may introduce redundant information and neglect modality-specific characteristics. To address this, the Dual-Branch Fusion Module (DBF) applies differentiated processing to shared and specific features, enabling deep integration of shared information while preserving modality-specific representations.

For each modality $m \in \{ o p t , s a r \}$ and branch $b \in \{ s h , s p \}$ DBF obtains the corresponding components by

$$
\begin{array} { r } { F _ { m } ^ { b } = \mathrm { G a t h e r } _ { c } ( F _ { m } ; \mathcal { T } _ { m } ^ { b } ) , } \\ { E _ { m } ^ { b } = \mathrm { G a t h e r } _ { c } ( E _ { m } ; \mathcal { T } _ { m } ^ { b } ) . } \end{array}\tag{9}
$$

This operation produces the shared and specific spatial features together with their embedded counterparts without additional embedding.

The proposed DBF consists of two parallel branches: a Shared Feature Fusion Branch and a Specific Feature Preservation Branch, as illustrated in Fig. 3.

Shared features correspond to channels exhibiting strong cross-modal attention responses, making them suitable for cross-modal interaction. A Transformer-based interaction module therefore operates directly on the shared embeddings selected above. Specifically, the SAR shared embeddings are treated as Query, while the optical shared embeddings serve as Key and Value:

$$
E _ { s a r } ^ { s h , * } = \mathrm { A t t e n t i o n } ( Q _ { s a r } ^ { s h } , K _ { o p t } ^ { s h } , V _ { o p t } ^ { s h } ) ,\tag{10}
$$

This lets SAR features selectively aggregate semantically relevant information from the optical modality, enhancing crossmodal representation learning and exploiting complementarity.

The interacted features are then projected back to the spatial domain through a decoder:

$$
\hat { F } _ { s a r } ^ { s h } = \Psi ( E _ { s a r } ^ { s h , * } ) ,\tag{11}
$$

where $\Psi ( \cdot )$ denotes the inverse embedding decoder.

To preserve low-level structural details contained in the original shared features, a learnable residual connection is introduced:

$$
\tilde { F } _ { s a r } ^ { s h } = \lambda _ { 1 } \hat { F } _ { s a r } ^ { s h } + F _ { s a r } ^ { s h } ,\tag{12}
$$

where $\lambda _ { 1 }$ is a learnable parameter.

Although the shared channels have been identified, potential channel misalignment between modalities may still mix semantically unmatched channels. $\mathrm { ~ A ~ } 1 \times 1$ convolution-based channel alignment is therefore applied before fusion:

$$
\begin{array} { r } { F _ { s a r } ^ { a l i g n } = \mathrm { C o n v } _ { 1 \times 1 } \left( \tilde { F } _ { s a r } ^ { s h } \right) . } \end{array}\tag{13}
$$

The aligned SAR shared features are then fused with the optical shared features in a pairwise manner:

$$
\begin{array} { r } { F ^ { s h a r e d } = \Gamma \left( F _ { o p t } ^ { s h } , F _ { s a r } ^ { a l i g n } \right) , } \end{array}\tag{14}
$$

where $\Gamma ( \cdot )$ denotes the shared feature fusion module. Specifically, optical-SAR channel pairs are first constructed. Grouped convolutions are then employed to perform local spatial fusion within each channel pair.

TABLE I  
COMPARISON WITH STATE-OF-THE-ART METHODS FOR HORIZONTAL OBJECT DETECTION ON OGSOD 1.0.
<table><tr><td>Method</td><td>Params (M)</td><td>Precision</td><td>Recall</td><td> $\overline { { \mathrm { A P } _ { 5 0 } } }$ </td><td> $\overline { { \mathrm { A P } _ { 7 5 } } }$ </td><td> $\overline { { \mathrm { A P } _ { \mathrm { b r i d g e } } } }$ </td><td> $\overline { { \mathrm { A P } _ { \mathrm { h a r b o r } } } }$ </td><td>AP storage tank</td><td>mAP</td></tr><tr><td>VSFF [3]</td><td>34.1</td><td>86.40</td><td>73.03</td><td>92.83</td><td>71.52</td><td>52.16</td><td>83.64</td><td>61.46</td><td>65.75</td></tr><tr><td>CFT [38]</td><td>186.4</td><td>88.36</td><td>75.69</td><td>93.58</td><td>74.83</td><td>55.68</td><td>86.76</td><td>63.57</td><td>68.67</td></tr><tr><td>CSSA [29]</td><td>55.0</td><td>88.52</td><td>76.92</td><td>93.43</td><td>75.89</td><td>56.18</td><td>87.05</td><td>66.42</td><td>69.89</td></tr><tr><td>ICAFusion [19]</td><td>99.3</td><td>87.95</td><td>76.01</td><td>93.52</td><td>75.37</td><td>56.10</td><td>87.65</td><td>65.05</td><td>69.60</td></tr><tr><td>MMIDet [45]</td><td>186.4</td><td>88.13</td><td>75.57</td><td>93.44</td><td>74.40</td><td>55.75</td><td>86.67</td><td>64.41</td><td>68.94</td></tr><tr><td>E2E-OSDet [34]</td><td>56.9</td><td>88.27</td><td>76.41</td><td>93.20</td><td>75.83</td><td>53.44</td><td>88.87</td><td>67.15</td><td>69.94</td></tr><tr><td>CSIF-Net</td><td>62.4</td><td>89.66</td><td>79.32</td><td>93.33</td><td>78.59</td><td>58.29</td><td>89.74</td><td>73.82</td><td>73.95</td></tr></table>

Specific features characterize modality-specific observations that generally lack reliable counterparts, so cross-modal fusion on them may cause information dilution or semantic distortion. Self-attention is therefore applied to the selected SAR-specific embedding before reconstruction:

$$
E _ { s a r } ^ { s p , * } = \mathrm { S e l f A t t n } ( E _ { s a r } ^ { s p } ) .\tag{15}
$$

This operation models dependencies among SAR-specific channels and enhances informative responses without introducing cross-modal interference. The enhanced embedding is then decoded back into the spatial feature domain:

$$
\hat { F } _ { s a r } ^ { s p } = \Psi \big ( E _ { s a r } ^ { s p , * } \big ) .\tag{16}
$$

A residual connection is further introduced to preserve the original observations:

$$
F _ { s a r } ^ { s p e c } = \lambda _ { 2 } \hat { F } _ { s a r } ^ { s p } + F _ { s a r } ^ { s p } ,\tag{17}
$$

where $\lambda _ { 2 }$ denotes a learnable weighting parameter.

Since optical images provide richer texture and structure, their specific features are inherently reliable and are directly retained:

$$
F _ { o p t } ^ { s p e c } = F _ { o p t } ^ { s p } .\tag{18}
$$

This design avoids unnecessary cross-modal interference while maximizing the preservation of modality-specific discriminative information.

Finally, the fused shared representation, reconstructed SAR specific features, and preserved optical specific features are concatenated along the channel dimension:

$$
\begin{array} { r } { F _ { o u t } = \mathrm { C o n c a t } \left( F ^ { s h a r e d } , F _ { s a r } ^ { s p e c } , F _ { o p t } ^ { s p e c } \right) . } \end{array}\tag{19}
$$

The resulting representation contains cross-modal shared semantics from deep interaction, modality-specific scattering and structure from SAR, and modality-specific texture and appearance from optical imagery.

Compared with directly concatenating all channels, the proposed strategy retains a single shared representation while preserving all modality-specific information, thereby reducing redundancy and balancing compactness with completeness.

## IV. EXPERIMENTS

## A. Datasets and Implementation Details

We evaluate the proposed method on two public optical-SAR object detection datasets, including OGSOD 1.0 and M4- SAR. These two datasets differ in image resolution, object categories, annotation format, and detection protocol, allowing us to evaluate the proposed fusion strategy under both horizontal and oriented object detection settings.

OGSOD 1.0. OGSOD 1.0 contains paired SAR and optical images with a spatial resolution of 10 m and an image size of 256 × 256. It includes three object categories: bridge, harbor, and storage tank. The dataset is divided into 14,665 training pairs and 3,666 testing pairs.

M4-SAR. M4-SAR provides paired optical-SAR images with two spatial resolutions and different SAR polarization modes. The 10 m resolution subset corresponds to VH polarization and contains the first 56,087 image pairs, while the 60 m resolution subset corresponds to VV polarization and contains the remaining 56,087 image pairs. The dataset includes six categories: airport, harbor, bridge, playground, wind turbine, and oil tank. All images are cropped to $5 1 2 \times 5 1 2$

Only the 10 m VH-polarized subset is used, divided into 28,058 training, 11,056 validation, and 16,973 testing pairs. The 10 m and 60 m subsets differ in both resolution and polarization, so mixing them would introduce distribution shifts that obscure the contribution of the proposed fusion module; the 10 m subset thus ensures a consistent distribution and fair comparison.

Experimental Settings. The experiments are organized into horizontal object detection on OGSOD 1.0 and oriented object detection on M4-SAR. For the horizontal setting, ResNet-50, FPN, and TOOD are adopted as the backbone, neck, and detector. Training uses AdamW for 100 epochs with a batch size of 32, a learning rate of $1 \times 1 0 ^ { - 4 }$ , a weight decay of $1 \times 1 0 ^ { - 5 }$ , and a step schedule with an interval of 10 epochs and a decay factor of 0.6. For the oriented setting, YOLO11n is used with its default backbone and neck. Training uses SGD for 300 epochs with a batch size of 32, an initial learning rate of $1 \times 1 0 ^ { - 2 }$ , a weight decay of $5 \times 1 0 ^ { - 4 }$ , and a cosine schedule that reduces the learning rate to $1 \times 1 0 ^ { - 4 }$

## B. Horizontal Object Detection

To evaluate the effectiveness of the proposed CSIF-Net under the horizontal object detection setting, we compare it with several representative multimodal fusion methods on the OGSOD 1.0 dataset, including CFT, CSSA, ICAFusion, MMIDet, E2E-OSDet, and VSFF. For a fair comparison, all methods are implemented under the same detection framework consisting of TOOD with a ResNet-50 backbone and an FPN neck, so that the observed performance differences mainly reflect the effectiveness of different fusion strategies rather than the detector architecture.

TABLE II  
COMPARISON WITH STATE-OF-THE-ART METHODS FOR ORIENTED OBJECT DETECTION ON THE M4-SAR 10 M SUBSET.
<table><tr><td>Method</td><td>Params (M)</td><td>Prec.</td><td> $\overline { { \mathrm { R e c . } } }$ </td><td> $\overline { { \mathrm { A P } _ { 5 0 } } }$ </td><td> $\overline { { \mathrm { A P } _ { 7 5 } } }$ </td><td>mAP</td></tr><tr><td>VSFF [3]</td><td>9.3</td><td>81.65</td><td>81.93</td><td>85.31</td><td>72.43</td><td>61.39</td></tr><tr><td>CFT [38]</td><td>53.8</td><td>83.45</td><td>82.64</td><td>88.40</td><td>76.39</td><td>65.43</td></tr><tr><td>CSSA [29]</td><td>13.6</td><td>82.91</td><td>82.13</td><td>87.25</td><td>75.30</td><td>63.07</td></tr><tr><td>ICAFusion [19]</td><td>29.1</td><td>82.68</td><td>82.44</td><td>87.70</td><td>76.19</td><td>64.63</td></tr><tr><td>MMIDet [45]</td><td>53.8</td><td>83.16</td><td>82.53</td><td>88.65</td><td>76.62</td><td>64.54</td></tr><tr><td>E2E-OSDet [34]</td><td>24.6</td><td>87.09</td><td>83.75</td><td>89.91</td><td>79.70</td><td>68.12</td></tr><tr><td>CSIF-Net</td><td>31.4</td><td>94.7</td><td>82.19</td><td>91.82</td><td>83.67</td><td>71.30</td></tr></table>

VSFF is a pixel-level fusion method that fuses the SAR and optical images before feeding them into a single detection backbone, whereas the remaining methods are feature-level approaches that interact with the two modalities within dualstream backbones.

Table I reports the quantitative comparison on the OGSOD 1.0 dataset, where the results are analyzed from the perspectives of detection accuracy, localization reliability, cross-scale robustness, and model complexity. In terms of overall detection accuracy, CSIF-Net achieves the best mAP of 73.95%, surpassing E2E-OSDet (69.94%), the strongest optical-SAR fusion method, by 4.01 percentage points. Regarding detection reliability, CSIF-Net simultaneously achieves the highest Precision (89.66%) and Recall (79.32%), surpassing the best competing results by 1.14 and 2.40 percentage points, respectively, which suggests that the proposed method reduces false-positive detections while recovering more true targets without a precision-recall trade-off. Although CFT attains the highest $\mathrm { { A P } _ { 5 0 } }$ (93.58%), its notably lower $\mathsf { A P } _ { 7 5 }$ (74.83%) reflects weaker localization under stricter IoU, whereas CSIF-Net achieves the highest $\mathsf { A P } _ { 7 5 }$ (78.59%), confirming more precise object localization. Regarding object scale, CSIF-Net consistently achieves the highest category-wise AP across all three categories. For large-scale objects, CSIF-Net surpasses the best competing results by 2.11 and 0.87 percentage points, respectively. For the compact storage tank category, whose features are subtle and easily overwhelmed by noise, CSIF-Net reaches 73.82% AP, exceeding the best competing result (67.15% from E2E-OSDet) by $6 . 6 7$ percentage points. This demonstrates that the proposed shared-specific channel partitioning is particularly beneficial for small objects, while still improving large-object detection through more discriminative feature interaction. In terms of model complexity, CSIF-Net has only 62.4M parameters, which is substantially fewer than CFT and MMIDet (186.4M) and comparable to CSSA (55.0M) and E2E-OSDet (56.9M). Despite this moderate parameter scale, CSIF-Net achieves the best mAP, indicating a favorable trade-off between model efficiency and detection accuracy.

## C. Oriented Object Detection

To further evaluate the effectiveness and generalization capability of the proposed CSIF-Net under the oriented object detection setting, experiments are conducted on the M4-SAR dataset. All compared methods are implemented using the same YOLO11n detection framework with identical training settings.

TABLE III  
ABLATION STUDY ON THE OGSOD 1.0 DATASET.
<table><tr><td colspan="3">Components</td><td colspan="6">Params (M) Recall  $\mathrm { A P _ { 5 0 } }$   $\mathrm { A P } _ { 7 5 }$  mAP</td></tr><tr><td>FDM CACS PB FB</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td> $\checkmark$ </td><td> $\checkmark$   $\checkmark$ </td><td>62.4</td><td></td><td>77.2</td><td>93.4</td><td>77.9</td><td>72.99</td></tr><tr><td> $\checkmark$ </td><td></td><td> $\checkmark$  √</td><td></td><td>61.3 62.1</td><td>78.1</td><td>92.9</td><td>77.2</td><td>71.68</td></tr><tr><td> $\checkmark$ </td><td> $\checkmark$ </td><td> $\checkmark$ </td><td></td><td>62.1</td><td>77.3 77.9</td><td>93.5 93.1</td><td>77.9 77.0</td><td>72.46</td></tr><tr><td> $\checkmark$ </td><td> $\checkmark$ </td><td> $\checkmark$ </td><td></td><td>59.8</td><td>75.8</td><td>93.2</td><td>74.5</td><td>71.21</td></tr><tr><td> $\checkmark$ </td><td> $\checkmark$ </td><td></td><td></td><td>56.9</td><td>75.1</td><td>93.1</td><td>74.2</td><td>68.20</td></tr><tr><td> $\checkmark$ </td><td></td><td></td><td></td><td>56.9</td><td>75.4</td><td>93.1</td><td>74.3</td><td>67.09</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>67.13</td></tr><tr><td> $\checkmark$ </td><td> $\checkmark$ </td><td> $\checkmark$ </td><td> $\checkmark$ </td><td>62.4</td><td>79.3</td><td>93.3</td><td>78.6</td><td>73.95</td></tr></table>

Table II reports the comparison results, where CSIF-Net again achieves the best detection performance in several metrics. In terms of overall detection accuracy, CSIF-Net achieves the best mAP of 71.30%, surpassing E2E-OSDet (68.12%), the strongest optical-SAR fusion method, by 3.18 percentage points. Regarding detection reliability, the gain of CSIF-Net is concentrated in precision rather than recall. CSIF-Net achieves the highest Precision of 94.70%, surpassing E2E-OSDet (87.09%) by 7.61 percentage points, whereas its Recall (82.19%) is slightly lower than that of E2E-OSDet (83.75%). This profile differs from the horizontal setting, where both metrics improved simultaneously, and indicates that CSIF-Net suppresses false-positive detections at a marginal cost of recall, which is particularly valuable for oriented detection where precise bounding-box regression matters more. Correspondingly, CSIF-Net attains the highest $\mathrm { { A P } _ { 5 0 } }$ (91.82%) and $\mathsf { A P } _ { 7 5 }$ (83.67%), improving over E2E-OSDet by 1.91 and 3.97 percentage points, respectively. In terms of model complexity, CSIF-Net uses 31.4M parameters. Notably, CSIF-Net has a parameter count close to that of ICAFusion (29.1M) yet attains a substantially higher mAP (71.30% vs. 64.63%), indicating that the performance gain stems from the channel-selective fusion strategy rather than from the parameter scale.

## D. Ablation Study

To evaluate the contribution of each component, we conduct ablation experiments on the OGSOD 1.0 dataset, as shown in Table III. The ablation follows a two-stage strategy. In the first stage, we remove each module individually, i.e., FDM, CACS, PB, and FB, to isolate the contribution of a single component. However, since these modules cooperate with one another, their effects are not simply additive, and the removal of one module may be partially compensated by the remaining ones. We therefore further perform a cumulative combination ablation that progressively removes the modules from the later stages back to the earlier ones, i.e., from the full model down to the baseline, so as to verify how the components jointly contribute to the overall framework. Here, FDM, CACS, PB, and FB denote the Frequency Decomposition Module, Cross-Attention Channel Selection Module, Preservation Branch, and Fusion Branch, respectively.

In the single-module ablation, removing any component degrades mAP. Removing the shared branch causes the largest

![](images/083f16ebfe2a446be24e97c0072c345d9b991eddc653ea7c304206e0c2eea841.jpg)  
(a) Feature responses before and after cross-modal fusion.  
(b) Full-channel versus selected-channel fusion.

Fig. 4. Feature-level analysis of CSIF-Net: (a) single-channel responses before and after fusion; (b) full-channel versus selected-channel fusion.

drop of 2.74 points (from 73.95% to 71.21%), followed by removing CACS (2.27 points) and the specific branch (1.49 points), while removing FDM causes the smallest drop of 0.96 points (to 72.99%). This suggests that the shared-branch cross-modal interaction and channel selection are the primary contributors to detection accuracy, whereas the specific branch and FDM provide smaller but complementary gains by preserving modality-specific information and enriching the frequency representation, respectively. In terms of localization, the complete model attains the highest $\mathrm { A P _ { 7 5 } }$ of 78.6%, and removing any single module lowers $\mathrm { A P _ { 7 5 } }$ , indicating that all components collectively contribute to precise localization.

The cumulative combination ablation further confirms the non-additive cooperation among the modules. Removing both branches while keeping FDM and CACS reduces mAP to 68.20%, and further removing either CACS or FDM drops it to about 67.1%, close to the baseline, indicating that these modules contribute little in isolation but become effective through joint cooperation. Overall, while the $\mathrm { A P _ { 5 0 } }$ values remain close across all variants, the complete model consistently attains the best Recall (79.3%), $\mathrm { A P _ { 7 5 } }$ (78.6%), and mAP (73.95%), demonstrating that the four components complement one another and collectively improve detection.

## E. Model Analysis and Visualization

To provide an intuitive understanding of CSIF-Net, we visualize intermediate features from two perspectives: singlechannel features before and after fusion, and multi-channel features under different fusion strategies.

Fig. 4a shows representative single-channel features before and after fusion. Before fusion, the optical features provide rich appearance and boundary cues, while the SAR features capture structural scattering patterns; the two modalities are thus complementary in content. After channel-selective fusion, the fused features retain the key target information extracted from both modalities while suppressing modalityspecific interference: the speckle noise inherent to the SAR modality becomes less salient, so the target-related structures are preserved and the irrelevant background responses are weakened. Moreover, when the optical modality fails to provide reliable cues (for example, when the target color is close to the background or the contours are unclear), the SAR modality can compensate in time by supplying robust structural features, ensuring stable target representation under degraded optical conditions. These observations confirm that CSIF-Net effectively exploits cross-modal complementarity to produce more discriminative features.

Fig. 4b further compares the feature patterns produced by full-channel fusion and the proposed selected-channel fusion, where the maps are obtained by averaging the ten channels with the largest accumulated gradients. In full-channel fusion, all channels participate in cross-modal interaction, so channels with weak semantic correspondence not only introduce redundant background responses but also bring modalityspecific noise and mutual interference from the other modality, diluting the target and contaminating the fused representation. By contrast, selected-channel fusion restricts the cross-modal interaction to the shared channels, preventing the noise-prone and modality-specific channels from disturbing one another, while the remaining channels are preserved in their own branches. As a result, the target responses become more compact and the background is more strongly suppressed. This indicates that the proposed channel selection effectively suppresses both noise and intermodality interference, yielding more discriminative and reliable representations.

## F. Qualitative Results

Fig. 5 presents qualitative comparisons under both the horizontal and oriented object detection settings. The top two rows show horizontal bounding-box results on representative OGSOD 1.0 scenes, and the bottom two rows show oriented bounding-box results on representative M4-SAR scenes. In each row, the paired optical and SAR ground-truth views are shown side by side, followed by the optical-view detection results of the compared methods.

Under the horizontal setting, the main problem is the missed detection of targets with non-salient features or small scales. For such targets, whose responses in the optical modality are weak and easily confused with the background, the compared methods frequently fail to detect them. By exploiting the complementary scattering information from the SAR modality, CSIF-Net recovers these easily missed targets and localizes them more precisely, thereby reducing missed detections.

OPT GT  
SAR GT  
CFT  
CSSA  
MMIDet  
E2E-OSDet  
![](images/57c376d2096a74b657f5780cf81193954f59075bb64c11af36c30af65252b632.jpg)  
Fig. 5. Qualitative comparison of CSIF-Net against existing methods. The top two rows show horizontal bounding-box detection results on representative OGSOD 1.0 scenes, and the bottom two rows show oriented bounding-box detection results on representative M4-SAR scenes. Each row is one paired optical-SAR sample; the first two columns show the optical and SAR ground-truth annotations, and the remaining columns show the optical-view detection results of different methods. Red dashed boxes denote ground truth, yellow solid boxes denote incorrect predictions and green solid boxes denote correct predictions.

Under the oriented setting, CSIF-Net mainly benefits from more accurate box orientation and fewer false positives. Methods such as CFT, CSSA, and MMIDet tend to output misaligned boxes or redundant predictions for objects with ambiguous boundaries, whereas CSIF-Net suppresses modalityspecific noise and interference through selective channel interaction, yielding clean and well-aligned oriented boxes.

## V. CONCLUSION

This paper presented CSIF-Net, a channel-selective interaction fusion network for optical-SAR object detection. By combining frequency decomposition, cross-modal channel selection, and dual-branch feature fusion, the framework models shared and modality-specific information separately, enabling effective cross-modal interaction while suppressing redundant interference. Experiments on OGSOD 1.0 and M4-SAR show that CSIF-Net outperforms state-of-the-art methods under both horizontal and oriented detection settings, especially for small objects and targets with weak responses in one modality, which is further confirmed by ablation studies and qualitative comparisons.

Despite these gains, CSIF-Net still has two limitations. First, it encodes the raw feature maps directly, which precludes random cropping and other spatial augmentations and thus limits its generalization. Second, the Top-K-based hard selection used to separate the two types of features is non-differentiable and cannot be optimized directly, so it is trained only through a STE. Future work will investigate more adaptive channel interaction mechanisms and extend the framework to more challenging multimodal remote sensing scenarios with additional sensing modalities and larger-scale datasets.

## REFERENCES

[1] J. Liu, G. Wu, Z. Liu, D. Wang, Z. Jiang, L. Ma, W. Zhong, X. Fan, and R. Liu, “Infrared and visible image fusion: From data compatibility to task adaption,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 47, no. 4, pp. 2349–2369, 2025.

[2] Q. Zhang, Y. Wei, Z. Han, H. Fu, X. Peng, Q. Hu, C. Deng, C. Xu, J. Wen, D. Hu, and C. Zhang, “Multimodal fusion on low-quality data: A comprehensive survey,” Inf. Fusion, vol. 135, p. 104437, 2026.

[3] Y. Ye, J. Zhang, L. Zhou, J. Li, X. Ren, and J. Fan, “Optical and SAR image fusion based on complementary feature decomposition and visual saliency features,” IEEE Trans. Geosci. Remote Sens., vol. 62, pp. 1–15, 2024.

[4] C. Liu, Y. Sun, Y. Xu, Z. Sun, X. Zhang, L. Lei, and G. Kuang, “A review of optical and SAR image deep feature fusion in semantic segmentation,” IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens., vol. 17, pp. 12 910–12 930, 2024.

[5] Z. Zhou, L. Zhao, K. Ji, and G. Kuang, “A domain-adaptive few-shot SAR ship detection algorithm driven by the latent similarity between optical and SAR images,” IEEE Trans. Geosci. Remote Sens., vol. 62, pp. 1–18, 2024.

[6] J. Li, D. Hong, L. Gao, J. Yao, K. Zheng, B. Zhang, and J. Chanussot, “Deep learning in multimodal remote sensing data fusion: A comprehensive review,” Int. J. Appl. Earth Obs. Geoinf., vol. 112, p. 102926, 2022.

[7] G. Vivone, L.-J. Deng, S. Deng, D. Hong, M. Jiang, C. Li, W. Li, H. Shen, X. Wu, J.-L. Xiao, J. Yao, M. Zhang, J. Chanussot, S. Garc´ıa, and A. Plaza, “Deep learning in remote sensing image fusion: Methods, protocols, data, and future perspectives,” IEEE Geosci. Remote Sens. Mag., vol. 13, no. 1, pp. 269–310, 2025.

[8] H. Huang, J. Guo, H. Lin, Y. Huang, and X. Ding, “Domain adaptive oriented object detection from optical to SAR images,” IEEE Trans. Geosci. Remote Sens., vol. 63, pp. 1–14, 2025.

[9] X. Li, Z. Du, Y. Huang, and Z. Tan, “A deep translation (GAN) based change detection network for optical and SAR remote sensing images,” ISPRS J. Photogramm. Remote Sens., vol. 179, pp. 14–34, 2021.

[10] X. Li, G. Zhang, H. Cui, S. Hou, Y. Chen, Z. Li, H. Li, and H. Wang, “Progressive fusion learning: A multimodal joint segmentation framework for building extraction from optical and SAR images,” ISPRS J. Photogramm. Remote Sens., vol. 195, pp. 178–191, 2023.

[11] G. Gao, M. Wang, X. Zhang, and G. Li, “DEN: A new method for SAR and optical image fusion and intelligent classification,” IEEE Trans. Geosci. Remote Sens., vol. 63, pp. 1–18, 2025.

[12] J. Li, H. Huo, C. Li, R. Wang, and Q. Feng, “AttentionFGAN: Infrared and visible image fusion using attention-based generative adversarial networks,” IEEE Trans. Multimedia, vol. 23, pp. 1383–1396, 2021.

[13] J. Zhang, L. Jiao, W. Ma, F. Liu, X. Liu, L. Li, P. Chen, and S. Yang, “Transformer based conditional GAN for multimodal image fusion,” IEEE Trans. Multimedia, vol. 25, pp. 8988–9001, 2023.

[14] T. Liu, M. Zhang, M. Gong, Q. Zhang, F. Jiang, H. Zheng, and D. Lu, “Commonality feature representation learning for unsupervised multimodal change detection,” IEEE Trans. Image Process., vol. 34, pp. 1219–1233, 2025.

[15] H. Li, Z. Yang, Y. Zhang, W. Jia, Z. Yu, and Y. Liu, “MulFS-CAP: Multimodal fusion-supervised cross-modality alignment perception for unregistered infrared-visible image fusion,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 47, no. 5, pp. 3673–3690, 2025.

[16] B. Ren, B. Liu, Q. Wang, B. Hou, C. Yang, and L. Jiao, “DCIFNet: Cross-modal fusion with correction and interaction for optical–SAR land cover classification,” IEEE Trans. Geosci. Remote Sens., vol. 63, pp. 1– 18, 2025.

[17] B. Ren, S. Ma, B. Hou, D. Hong, J. Chanussot, J. Wang, and L. Jiao, “A dual-stream high resolution network: Deep fusion of GF-2 and GF-3 data for land cover classification,” Int. J. Appl. Earth Obs. Geoinf., vol. 112, p. 102896, 2022.

[18] X. Ma, X. Zhang, M.-O. Pun, and M. Liu, “A multilevel multimodal fusion transformer for remote sensing semantic segmentation,” IEEE Trans. Geosci. Remote Sens., vol. 62, pp. 1–15, 2024.

[19] J. Shen, Y. Chen, Y. Liu, X. Zuo, H. Fan, and W. Yang, “ICAFusion: Iterative cross-attention guided feature fusion for multispectral object detection,” Pattern Recognit., vol. 145, p. 109913, 2024.

[20] W. Tang, F. He, and Y. Liu, “YDTR: Infrared and visible image fusion via y-shape dynamic transformer,” IEEE Trans. Multimedia, vol. 25, pp. 5413–5428, 2023.

[21] X.-N. Jiang, Z.-X. Huang, F.-L. Wu, Y. Fu, S.-J. Fan, X. Gao, and A. Han, “Correlation-guided progressive interaction and gated fusion for multimodal optical and SAR remote sensing image object detection,” IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens., vol. 19, pp. 21 055–21 069, 2026.

[22] R. Li, J. Xiang, F. Sun, Y. Yuan, L. Yuan, and S. Gou, “Multiscale cross-modal homogeneity enhancement and confidence-aware fusion for multispectral pedestrian detection,” IEEE Trans. Multimedia, vol. 26, pp. 852–863, 2024.

[23] Q. Li, C. Zhang, Q. Hu, H. Fu, and P. Zhu, “Confidence-aware fusion using dempster-shafer theory for multispectral pedestrian detection,” IEEE Trans. Multimedia, vol. 25, pp. 3420–3431, 2023.

[24] J. Chen, J. Ding, and J. Ma, “HitFusion: Infrared and visible image fusion for high-level vision tasks using transformer,” IEEE Trans. Multimedia, vol. 26, pp. 10 145–10 159, 2024.

[25] Z. Wang, W. Shao, Y. Chen, J. Xu, and X. Zhang, “Infrared and visible image fusion via interactive compensatory attention adversarial learning,” IEEE Trans. Multimedia, vol. 25, pp. 7800–7813, 2023.

[26] W. Dong, H. Zhu, S. Lin, X. Luo, Y. Shen, G. Guo, and B. Zhang, “Fusion-Mamba for cross-modality object detection,” IEEE Trans. Multimedia, vol. 27, pp. 7392–7406, 2025.

[27] F. Yang, W. Li, L. Li, M. Yang, and J. Zhang, “DWSF-Net: A dynamic wavelet-based spatial-frequency fusion network for multispectral object detection,” IEEE Trans. Multimedia, vol. 28, pp. 7061–7073, 2026.

[28] J. Shi, P. Duan, X. Ma, J. Chi, and Y. Dai, “Frefusion: Frequency domain transformer for infrared and visible image fusion,” IEEE Trans. Multimedia, vol. 27, pp. 5722–5730, 2025.

[29] Y. Cao, J. Bin, J. Hamari, E. Blasch, and Z. Liu, “Multimodal object detection by channel switching and spatial attention,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2023, pp. 403–411.

[30] X. Li, G. Zhang, H. Cui, S. Hou, S. Wang, X. Li, Y. Chen, Z. Li, and L. Zhang, “MCANet: A joint semantic segmentation framework of optical and SAR images for land use classification,” Int. J. Appl. Earth Obs. Geoinf., vol. 106, p. 102638, 2022.

[31] X. Jiang, G. Li, Y. Liu, X.-P. Zhang, and Y. He, “Change detection in heterogeneous optical and SAR remote sensing images via deep homogeneous feature fusion,” IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens., vol. 13, pp. 1551–1566, 2020.

[32] C. Yue, Y. Zhang, J. Yan, Z. Luo, Y. Liu, and P. Guo, “Diffusion mechanism and knowledge distillation object detection in multimodal remote sensing imagery,” IEEE Trans. Geosci. Remote Sens., vol. 63, pp. 1–14, 2025.

[33] Y. Yang, J. Chen, L. Sun, Z. Zhou, Z. Huang, and B. Wu, “Unsupervised domain-adaptive SAR ship detection based on cross-domain feature interaction and data contribution balance,” Remote Sens., vol. 16, p. 420, Jan. 2024.

[34] C. Wang, W. Lu, X. Li, J. Yang, and L. Luo, “M4-SAR: A multiresolution, multi-polarization, multi-scene, multi-source dataset and benchmark for optical-SAR object detection,” 2026, arXiv:2505.10931. [Online]. Available: https://arxiv.org/abs/2505.10931

[35] M. Zhou, Y. Li, G. Yang, X. Wei, H. Pu, J. Luo, and W. Jia, “COFNet: Contrastive object-aware fusion using box-level masks for multispectral object detection,” IEEE Trans. Multimedia, vol. 27, pp. 7444–7458, 2025.

[36] X. Wang, X. Chen, H. Fan, W. Ren, S. Wang, Y. Tang, L. Liu, and Z. Han, “Seeing only the focus: RGB-T object-aware region enhancement for object detection in harsh environments,” IEEE Trans. Multimedia, pp. 1–15, 2026.

[37] B. Wan, X. Zhou, Y. Sun, T. Wang, C. Lv, S. Wang, H. Yin, and C. Yan, “MFFNet: Multi-modal feature fusion network for V-D-T salient object detection,” IEEE Trans. Multimedia, vol. 26, pp. 2069–2081, 2024.

[38] Q. Fang and Z. Wang, “Cross-modality attentive feature fusion for object detection in multispectral remote sensing imagery,” Pattern Recognit., vol. 130, p. 108786, 2022.

[39] K. Helvig, B. Abeloos, and P. Trouve-Peloux, “CAFF-DINO: Multi-´ spectral object detection transformers with cross-attention features fusion,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2024, pp. 3037–3046.

[40] J. Hu, L. Shen, and G. Sun, “Squeeze-and-excitation networks,” in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018, pp. 7132–7141.

[41] X. Li, W. Wang, X. Hu, and J. Yang, “Selective kernel networks,” in 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 510–519.

[42] S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, “CBAM: Convolutional block attention module,” in Computer Vision – ECCV 2018, V. Ferrari, M. Hebert, C. Sminchisescu, and Y. Weiss, Eds. Cham: Springer International Publishing, 2018, pp. 3–19.

[43] Q. Wang, B. Wu, P. Zhu, P. Li, W. Zuo, and Q. Hu, “ECA-Net: Efficient channel attention for deep convolutional neural networks,” in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 11 531–11 539.

[44] M. Yuan and X. Wei, “C²Former: Calibrated and complementary transformer for RGB-infrared object detection,” IEEE Trans. Geosci. Remote Sens., vol. 62, pp. 1–12, 2024.

[45] Y. Zeng, T. Liang, Y. Jin, and Y. Li, “MMI-Det: Exploring multimodal integration for visible and infrared object detection,” IEEE Trans. Circuits Syst. Video Technol., vol. 34, no. 11, pp. 11 198–11 213, 2024.