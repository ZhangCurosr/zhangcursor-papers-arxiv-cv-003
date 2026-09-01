# SegWave: Wavelet-Driven Segmentation of Tampered Regions

Siddhi Pravin Lipare<sup>1†</sup>, Vishesh Kumar<sup>2</sup>, and Akshay Agarwal<sup>2</sup>

<sup>1</sup> IIIT-Hyderabad, India

siddhi.l@research.iiit.ac.in

2 Trustworthy BiometraVision Lab, IISER-Bhopal, India {vishesh22, akagarwal}@iiserb.ac.in

Abstract. Verifying image authenticity is increasingly dificult, posing serious risks across journalism, law enforcement, and political domains. Most existing forensic methods rely on high-level visual artifacts and treat frame detection as a simple binary task. To address this, we propose SegWave, a hybrid framework that jointly leverages spatial and frequency-domain cues for image tampering detection. SegWave integrates a transformer-based architecture with the Discrete Wavelet Transform (DWT) to capture localized, multi-scale frequency inconsistencies indicative of manipulation. To further improve localization efectiveness, we introduce an Adaptive Sub-band Attention module (ASA) that dynamically highlights the informative high-frequency wavelet components. Extensive experiments on multiple benchmark datasets demonstrate that SegWave consistently outperforms state-of-the-art tampering detection methods in challenging evaluation settings.

Keywords: Image Forensics, Trustworthy AI, Tampering Localization

## 1 Introduction

Verifying whether an image is authentic has become central to the modern information ecosystem - from journalism and legal evidence to public discourse on social media. With the rapid progress of generative AI [2, 24, 34], images can now be synthesized and edited with a realism that is dificult to spot by eye. The same tools that enable creative work let malicious actors fabricate evidence, alter surveillance footage, or spread convincing visual misinformation. As manipulations become more sophisticated, the demand for tampering detection algorithms that can keep pace grows just as quickly.

Consider an image circulating online that shows a public figure at an event they never attended. The face may be spliced from one photograph, the background lifted from another, and a small object inserted by a difusion model - three distinct manipulations composited into a single frame. For a journalist verifying the image, an analyst preparing it for court, or an automated moderation system, a verdict of simply “tampered” is of little practical use. What these users actually need is to know where the image is altered and how many distinct sources are stitched together, so that the manipulation can be traced, attributed, and explained.

![](images/d15381f0f379da1aa9d07db43c6073c9c41a7e435ebd0974a65afccda4b8a3a3.jpg)  
Fig. 1. SegWave enables efective segmentation of tampered regions. The input image is decomposed into frequency subbands (LL, LH, HL, HH) via the Haar Wavelet Transform. The high-frequency subbands are reweighted by an Adaptive Sub-band Attention module and recombined into an enriched representation for the image encoder. In parallel, point prompts sampled region-wise from the ground-truth mask are embedded by a prompt encoder. The image and prompt embeddings feed a transformer encoder-decoder whose mask decoder predicts the tampered regions, supervised by a weighted BCE and MSE loss. This lets SegWave fuse frequency and spatial cues for fine-grained, multi-source localization.

Yet most existing forensic models stop at exactly this binary verdict. Stateof-the-art methods such as CAT-Net [26], SAFIRE [25], and DocTamper [33] perform well under controlled conditions but struggle to generalize across the diversity of real manipulations such as object splicing, copy-move operations, and GAN- or difusion-based edits [24, 34]. A deeper limitation lies in what they rely on: much of their evidence comes from global frequency artifacts left behind by JPEG compression, captured through the Fast Fourier Transform (FFT) [21, 25] or the Discrete Cosine Transform (DCT) [7, 26, 27]. These cues are powerful for compressed media but fade on uncompressed formats such as PNG, where no lossy footprint exists, and they ofer little help in separating the multiple forgery sources that coexist within a single composite image. In high-stakes settings, where traceability and accountability matter as much as accuracy, this absence of fine-grained, source-aware localization is a serious gap.

To address this, we present SegWave (Segment any Manipulated Image using Wavelet Transforms), a hybrid prompt-guided framework that unifies spatial and localized frequency cues for tampering detection and localization. Rather than depending on compression artifacts, SegWave employs the Discrete Wavelet

Transform (DWT) [11, 36] to capture multi-scale, spatially localized frequency variations that remain informative even on uncompressed images. An Adaptive Subband Attention Module then dynamically weights the most informative high-frequency subbands while suppressing structural noise, sharpening the boundaries of tampered regions. Beyond a binary verdict, SegWave integrates a point-prompt-based segmentation mechanism [22, 25] that traces and partitions a composite image by its source regions—and crucially, it learns to do so under only weak binary-mask supervision, bridging fine-grained forensic interpretability with real-world scalability. Our key contributions are summarized as follows:

1. We propose SegWave, a hybrid transformer framework that fuses spatial features with localized DWT-based frequency cues, exposing multi-scale anomalies that survive in uncompressed (PNG) images and sidestepping the JPEG dependence of prior baselines.

2. We introduce an Adaptive Subband Attention Module that dynamically weights the most informative high-frequency wavelet subbands, sharpening boundary localization and improving robustness to GAN- and difusionbased edits.

3. We conduct extensive experiments across splicing, copy-move, and generative benchmarks, showing that SegWave consistently outperforms strong baselines such as SAFIRE, CAT-Net V2, and TruFor under challenging evaluation settings.

The rest of this paper is organized as follows: Section 2 reviews related work in tampering detection. Section 3 details the SegWave architecture. Section 4 presents experimental results and analysis. Finally, Section 6 concludes with key insights and future directions.

## 2 Related Works

Spatial forensic detectors. Early methods searched for inconsistencies directly in pixel space, relying on cues such as edge and boundary anomalies [12, 28]. The shift to learned representations brought CNN-based detectors [9,32,39], later strengthened with self-attention [30] and hierarchical modeling [17, 38] to sharpen localization. While efective on visible splices, spatial cues alone remain fragile against subtle and well-blended edits.

Frequency-domain forensics. To expose manipulations invisible in pixel space, a parallel line of work analyzes spectral artifacts, often fusing FFT or DCT features with spatial evidence [22, 25–27, 34, 38]. These approaches excel on compressed media, but much of their signal originates from JPEG-specific compression footprints, which weakens their reliability on uncompressed formats.

Multi-source partitioning. Most detectors return a single tampered/authentic decision and cannot separate the distinct sources composited into one image. Kwon et al. [25] address this with point-prompt-based segmentation, recovering multiple source regions even when trained only on binary-labeled data, and ofering the kind of source-aware output that high-stakes forensics demands.

Wavelet-based forensics. The Discrete Wavelet Transform ofers a compressionagnostic alternative whose localized, multi-scale decomposition is well suited to spatially confined edits [1,4–6]. Gadhiya et al. [14] use a hash-based DWT scheme for medical image tampering, and Hayat and Qazi [18] combine DWT and DCT for copy-move detection, though both depend on handcrafted features that limit robustness to complex forgeries. Closer to our setting, Chen et al. [11] pair DWT with transformer-based attention for text tampering; motivated by this, we explore DWT for object splicing and manipulation in natural images, where its localized frequency representation can be learned end-to-end.

Vision-language forensics. More recent work has moved toward MLLM-based explainable localization [40, 41], which produces interpretable, text-grounded forgery explanations but relies on large multimodal backbones and text-annotated training data. Notably, recent analysis finds that vision-language semantic priors can even degrade localization by favoring semantic plausibility over authenticity [16], reafirming the value of the low-level, artifact-sensitive cues that Seg-Wave is built around. Positioned within this landscape, SegWave instead pursues precise localization and multi-source partitioning learned purely from binarymask supervision, similar to weakly-supervised approaches [35].

## 3 Proposed SegWave

We introduce SegWave (Segment any Manipulated Image using Wavelet Transforms), a transformer-based architecture designed to efectively integrate both frequency and spatial domain features for enhanced tampering detection, which has demonstrated significant improvements across several benchmarks.

## 3.1 Method

A single-level Haar wavelet transform decomposes the input image into four subbands by applying low-pass (L) and high-pass (H) filtering successively along the rows and columns. This yields an approximation subband (LL) that retains the coarse image content, and three detail subbands: LH, HL, and HH, that capture high-frequency variations along the horizontal, vertical, and diagonal orientations, respectively. Since tampering artifacts such as splicing boundaries and local texture inconsistencies manifest as high-frequency discontinuities, we retain only the three detail subbands (LH, HL, HH) and discard the low-frequency approximation.

An Adaptive Sub-band Attention (ASA) module then computes dynamic attention scores for these subbands, emphasizing the most relevant high-frequency information. These weighted subbands are combined and reconstructed using inverse DWT, generating an enhanced image that accentuates high-frequency inconsistencies essential for detecting tampering. Subsequently, both the original and enhanced images pass independently through patch embedding layers. The extracted features undergo linear transformations, are tuned, and then fused into a unified representation. This joint representation is adaptively integrated into each layer of a vision transformer backbone, allowing simultaneous learning of low-level manipulation indicators and high-level semantic features.

Algorithm 1: Proposed SegWave Training Pipeline   
Input: Training images $\{ ( I _ { k } , M _ { k } ) \} _ { k = 1 } ^ { K }$ where $I _ { k }$ is the image and $M _ { k }$ is the mask.   
Weighting factor $\lambda { = } 0 . 1 .$   
Positive weight $w _ { + } ,$ negative weight w\_- .   
for each mini-batch $\{ ( I _ { k } , M _ { k } ) \} _ { k = 1 } ^ { B }$ do   
for each $( I _ { k } , M _ { k } )$ in the mini-batch do   
1. Wavelet Transform: Perform single-level Haar DWT on $I _ { k }$ and extract {LH,   
HL, HH}.   
2. Adaptive Sub-band Attention: Compute attention scores over LH, ${ \mathrm { H L } } ,$ HH   
and obtain a weighted sum of high-frequency clues.   
3. Inverse DWT: Reconstruct enhanced image $\hat { I } _ { k }$ from attended high-frequency   
components.   
4. Patch Embedding: Apply patch embedding to both $I _ { k }$ and $\hat { I } _ { k }$ to get $E _ { k }$ and   
$\hat { E } _ { k }$   
5. Feature Fusion: Tune and add $E _ { k } + \hat { E } _ { k }$ to form fused representation.   
6. Prompt Encoding: Sample point prompts from $M _ { k }$ (one tampered, one   
untampered) and encode them.   
7. Transformer Backbone: Process $E _ { k }$ and inject adapted features from the   
fusion path into each block.   
8. Mask Decoder: Fuse features and predict $( \hat { Y } _ { k } , \hat { c } _ { k } )$   
9. Compute Losses:   
$\mathcal { L } _ { w B C E } = - w _ { + } M _ { k } \log ( \hat { Y } _ { k } ) ~ - ~ w _ { - } \left( 1 - M _ { k } \right) \log \left( 1 - \hat { Y } _ { k } \right)$   
$\mathcal { L } _ { M S E } = \frac { 1 } { B } \sum ( \hat { c } _ { k } - c _ { k } ) ^ { 2 }$   
$\mathcal { L } _ { t o t a l } = \mathcal { L } _ { w B C E } + \lambda \mathcal { L } .$ MSE   
10. Update Parameters: Backpropagate $\mathcal { L } _ { t o t a l }$ and update \theta .   
end   
end

Simultaneously, region-wise paired sampling is applied to the ground truth tampering mask, where manipulated regions are labeled as 1 and authentic regions as 0. One point from each region is sampled and encoded by a prompt encoder, embedding them in a space compatible with image embeddings. These point prompts guide the model’s attention toward relevant regions, enhancing multisource awareness. The transformer-based mask decoder then fuses embeddings from the image and prompt encoders, leveraging learned mask and IoU tokens to generate a segmentation mask that precisely localizes the tampered regions. Training is guided by a composite loss function: the Weighted Binary Cross Entropy loss $( \mathcal { L } _ { w B C E } )$ is used for accurate mask prediction, while the Mean Squared Error loss $\left( \mathcal { L } _ { M S E } \right)$ estimates prediction confidence. The overall objective function is defined as $\mathcal { L } _ { t o t a l } = \mathcal { L } _ { w B C E } + \lambda \mathcal { L } _ { M S E }$ . The complete training pipeline is summarized in Algorithm 1.

Weighted Binary Cross Entropy Loss:

$$
\mathcal { L } _ { w B C E } = - w _ { + } y \log ( \hat { y } ) - w _ { - } ( 1 - y ) \log ( 1 - \hat { y } )\tag{1}
$$

where $y$ is the probability of the ground truth label being tampered, $\hat { y }$ is the predicted probability of a pixel being tampered, and $w _ { + }$ and $w _ { - }$ are the weights

![](images/6d1494a0738317b2422dc738c9c5979d6196ecaeb192ab1ce73551e4c12b2641.jpg)  
Fig. 2. SegWave<sub>DCT</sub> as an ablation variant for efective segmentation of tampered regions. Unlike SegWave which uses DWT, this variant applies a Discrete Cosine Transform (DCT) to the input image to capture global frequency components and block-wise structural information.

for positive and negative samples, ensuring equal importance to both source regions.

Mean Squared Error Loss:

$$
\mathcal { L } _ { M S E } = \frac { 1 } { N } \sum _ { I = 1 } ^ { N } \left( \hat { c } _ { i } - c _ { i } \right) ^ { 2 }\tag{2}
$$

where $c _ { i }$ represents the ground truth confidence scores, $\hat { c } _ { i }$ represents the predicted confidence scores, and N is the number of samples.

The total loss for training is given by:

$$
\mathcal { L } _ { t o t a l } = \mathcal { L } _ { w B C E } + \lambda \mathcal { L } _ { M S E }\tag{3}
$$

where \lambda is the weighting factor to balance the contributions of both losses.

## 3.2 SegWave<sub>DCT</sub> Variant

As an ablation, we replace the wavelet decomposition with a global Discrete Cosine Transform, yielding the SegWave<sub>DCT</sub> variant. Because the global DCT produces a single global frequency representation rather than orientation-specific subbands, there are no subbands for the Adaptive Sub-band Attention module to weight, and it is therefore omitted in this variant. The DCT-transformed input is otherwise passed through the same encoder-decoder pipeline as SegWave, isolating the efect of the transform choice while holding the rest of the architecture fixed.

## 3.3 Inference with Multiple Point Prompts

For inference, multiple-point prompts have been used to arrange in a grid pattern. Given an input image I and a set of points $Q _ { 1 } , \cdots , Q _ { N }$ , the model first extracts the image embedding $\mathcal { E } = E ( I )$ and prompt embeddings $\mathcal { H } _ { i } = H ( Q _ { i } )$

<table><tr><td colspan="2">Model</td><td>CocoGlide</td><td>Columbia</td><td>RealTamper</td></tr><tr><td rowspan="3">Pror</td><td>CAT-Net v2 [26]</td><td>0.43 / 0.60</td><td>0.86 / 0.92</td><td>0.14 / 0.24</td></tr><tr><td>TruFor [15]</td><td>0.52 / 0.72</td><td>0.86 / 0.91</td><td>0.43 / 0.53</td></tr><tr><td>SAFIRE [25]</td><td>0.63 / 0.76</td><td>0.98 / 0.99</td><td>0.39 / 0.50</td></tr><tr><td rowspan="2">1.</td><td>SegWaveDCT</td><td>0.65 / 0.76</td><td>0.95 / 0.99</td><td>0.40 / 0.49</td></tr><tr><td>SegWave w/o ASA</td><td>0.67 / 0.78</td><td>0.97 / 0.99</td><td>0.39 / 0.47</td></tr><tr><td>Ours</td><td>SegWave</td><td>0.69 / 0.79</td><td>0.99 / 0.99</td><td>0.36 / 0.51</td></tr></table>

Table 1. Binary-source localization on CocoGlide [15, 29], Columbia [19], and RealisticTampering [24]. Each cell reports $F 1 _ { f i x e d } \ / \ F 1 _ { b e s t } ;$ best per column in bold. Shaded green rows are ablations of our full model SegWave: SegWave w/o ASA removes the Adaptive Sub-band Attention module, while SegWave<sub>DCT</sub> additionally replaces the DWT with a DCT, which operates globally and therefore has no sub-bands to attend over. Epochs: SegWave<sub>DCT</sub> 35, SegWave w/o ASA 100, SegWave 46.

for all I. Since $\mathcal { E }$ is computed once, the process remains eficient despite multiple points.

Next, the mask decoder $D ( \cdot , \cdot )$ generates prediction maps:

$$
( \{ Y _ { 1 } , \cdot \cdot \cdot , Y _ { N } \} , \{ c _ { 1 } , \cdot \cdot \cdot , c _ { N } \} ) = D ( \mathcal { E } , \{ \mathcal { H } _ { 1 } , \cdot \cdot \cdot , \mathcal { H } _ { N } \} ) ,\tag{4}
$$

where $Y _ { i }$ represents the predicted mask and $c _ { i }$ its confidence score.

To refine predictions, we have computed representative features $\tau _ { i }$ by averaging image embeddings over predicted regions:

$$
\mathcal { T } _ { i } = \frac { 1 } { \vert \mathcal { R } ^ { b i n ( \mathcal { V } ) , \{ 1 \} } \vert } \sum _ { ( m , n ) \in \mathcal { R } ^ { b i n ( \mathcal { V } ) , \{ 1 \} } } \mathcal { E } [ m , n ] .\tag{5}
$$

The representative features are clustered into M groups, ensuring that predictions from the same source region are grouped. From each cluster, the most confident mask is selected as the best representation of that region. For the final prediction, selected masks are merged, using softmax for multi-source segmentation and averaging for binary-source segmentation to refine the output.

## 3.4 Metrics for Multi-Source Partitioning

To evaluate the performance of multi-source partitioning, two key metrics are used: the mean Intersection over Union (mIoU) and the Adjusted Rand Index (ARI). These metrics have been utilized in SOTA, i.e., SAFIRE [25], and thus, adopted for comparison with SegWave and its ablation variants. The mIoU is a widely used metric in semantic segmentation that quantifies how well the model’s predicted segmentation aligns with the ground truth. SAFIRE [25] extends the concept of permuted metrics to mIoU in multi-source partitioning, defining the permuted metric $p _ { - } m e t ( \cdot , \cdot )$ as follows:

<table><tr><td>Dataset</td><td>Model</td><td>p_mIoU</td><td>P-ARI</td></tr><tr><td rowspan="3">SAFIRE-MS-Expert-2</td><td>SAFIRE</td><td>0.763</td><td>0.660</td></tr><tr><td>SegWaveDCT</td><td>0.570</td><td>0.514</td></tr><tr><td>SegWave w/o ASA SegWave</td><td>0.695 0.821</td><td>0.630 0.710</td></tr><tr><td rowspan="4">SAFIRE-MS-Expert-3</td><td>SAFIRE</td><td>0.618</td><td>0.573</td></tr><tr><td>SegWaveDCT</td><td>0.570</td><td>0.514</td></tr><tr><td>SegWave w/o ASA</td><td>0.648</td><td>0.637</td></tr><tr><td>SegWave</td><td>0.648</td><td>0.641</td></tr><tr><td rowspan="4">SAFIRE-MS-Expert-4</td><td>SAFIRE</td><td>0.421</td><td>0.568</td></tr><tr><td>SegWaveDCT</td><td>0.393</td><td>0.513</td></tr><tr><td>SegWave w/o ASA</td><td>0.419</td><td>0.587</td></tr><tr><td>SegWave</td><td>0.422</td><td>0.577</td></tr></table>

Table 2. Multi-source partitioning on SafireMS-Expert [25] with 2, 3, and 4 source regions. p mIoU (Permuted mean IoU) and p ARI (Adjusted Rand Index), per Kwon et al. [25]. Green rows are ablations of SegWave (blue); best per block in bold.

$$
p _ { - } m e t ( Y , X ^ { * } ) = \operatorname* { m a x } _ { \tilde { X } ^ { * } \in \mathrm { P e r m } ( X ^ { * } ) } m e t ( Y , \tilde { X } ^ { * } ) ,\tag{6}
$$

where Perm(·) denotes the set of all possible permutations along the label dimension. ARI measures the similarity between two clusters, which ranges from -1 (misclassified clusters) to 1 (perfect clustering). It handles label permutations to make it easier to evaluate label-agnostic segmentation in multi-source partitioning.

## 4 Implementation Details

As defined by Guillaro et al. [15], we evaluate performance using the permuted F1-Score [20, 26], considering both $F 1 _ { f i x e d }$ and $F 1 _ { b e s t } . \ F 1 _ { f i x e d }$ computes the F1-score using a threshold of 0.5 on the predicted localization map, classifying pixels with probability $\geq 0 . 5$ as tampered, ensuring consistency across images. $F 1 _ { b e s t }$ optimizes the threshold per image to maximize the F1-score, serving as an upper-bound performance measure. We trained SegWave<sub>DCT</sub> and SegWave until the loss converged. During testing, we uniformly sample 256 prompt points arranged in a $1 6 \times 1 6$ grid and process them in batches of 128 to avoid GPU memory overflow during inference. The loss weighting factor λ was set to 0.1, keeping the auxiliary MSE confidence term as a minor regularizer relative to the primary segmentation objective.

CocoGlide Ground Truth  
TruFor  
CAT-Net V2  
SAFIRE  
SegWave\_DCT SegWave\_DWT  
![](images/471aea3b50d841f5df1c1e04bb36d8fc5b27ae9568885d8ac1239122992258fc.jpg)  
Fig. 3. SegWave test results on CocoGlide [15] dataset images demonstrate their superior ability to segment tampered regions compared to SegWave<sub>DCT</sub>, SAFIRE [25], TruFor [15], and CAT-Net V2 [26].

Rather than a block-based approach, we apply Global DCT and single-level Haar Wavelet Transform for SegWave<sub>DCT</sub> and SegWave, respectively; we assert that it is suficient for natural image tampering, where manipulated regions tend to be relatively large compared to document text forgeries [33, 34]. Our findings show that this strategy enables robust detection, making SegWave efective across diverse manipulation scenarios. The choice of Haar has been motivated by its efectiveness in prior forensic literature and its compatibility with lightweight attention mechanisms [3, 8, 10, 37].

## 4.1 Datasets

We train SegWave on a subset of two datasets, CASIA 2.0 [13] and FantasticReality [23], contributing 2,000 images each. We evaluate on four benchmarks held out from training: RealisticTampering [24] (220 images), CocoGlide [15, 29, 31] (512 images), Columbia [19] (180 images), and SafireMS-Expert [25] (238 images). SafireMS-Expert additionally provides images with 2, 3, and 4 distinct source regions, which we use for the multi-source partitioning evaluation.

## 5 Experimental Results & Analysis

Binary-source localization. As shown in Table 1, SegWave achieves the strongest results on CocoGlide and Columbia. On CocoGlide, it improves $F 1 _ { \mathrm { f i x e d } }$ from 0.43

![](images/c2324a0e8ee391478d3acc751d89d981b59602b0dc801fa046836c18f8dc1e81.jpg)  
Fig. 4. SegWave and SegWave<sub>DCT</sub> test results on RealisticTampering [24] dataset images demonstrate their superior ability to segment tampered regions compared to SAFIRE [25] and CAT-Net V2 [26].

(CAT-Net v2) and 0.52 (TruFor) to 0.69 - gains of 0.26 and 0.17, and improves on SAFIRE’s 0.63 by 0.06, with a similar ordering on $F 1 _ { \mathrm { b e s t } }$ . On Columbia, Seg-Wave reaches near-saturated performance, matching or exceeding all baselines. SegWave does not win everywhere: on RealisticTampering it trails TruFor’s 0.43 by 0.07 on $F 1 _ { \mathrm { f i x e d } }$ , a gap we analyze below.

The role of Adaptive Sub-band Attention (ASA). The ablation rows isolate the contribution of the attention module. Removing it (SegWave w/o ASA) leaves the DWT-enhanced input intact but weights all high-frequency subbands equally, while replacing the transform entirely (SegWave ) removes the notion of orientation-specific subbands altogether. On binary localization, the gain from ASA is modest $( 0 . 6 7  0 . 6 9 ~ F 1 _ { \mathrm { f i x e d } }$ on CocoGlide), but its value becomes pronounced on the harder multi-source task (Table 2): on SafireMS-Expert-2, ASA lifts p mIoU from 0.695 to 0.821 and p ARI from 0.630 to 0.710 over the equally-weighted variant. This matches the intuition behind the module: when multiple manipulated sources coexist, their boundary signatures reside in diferent orientation subbands, and adaptively reweighting LH, HL, and HH lets the model attend to the discriminative subband per source rather than diluting all three. SegWave<sub>DCT</sub>, lacking subbands to reweight, trails the full model across every multi-source setting, confirming that the gain stems from oriented, adaptively weighted decomposition rather than from frequency information alone.

When DCT helps. The one setting where SegWave<sub>DCT</sub> is competitive is RealisticTampering, where it edges out SegWave on $F 1 _ { \mathrm { f i x e d } }$ (0.40 vs 0.36). We attribute this to the dataset: although stored as TIFF, these images have likely undergone prior lossy JPEG compression, leaving DCT-aligned periodic artifacts that a DCT model captures directly. This is consistent with the broader pattern we observe: global DCT is better matched to compression-induced cues, whereas oriented DWT with ASA is better suited to spatially confined boundary inconsistencies of splicing and generative edits.

![](images/9218af1ab94d1734988ecd61743d3ed2e7cc651b72ae2552469276bdbd6d2fd7.jpg)  
Fig. 5. SegWave outperforms all other models on the SafireMS-Expert dataset [25], achieving the best performance across all evaluation metrics.

Multi-source partitioning. Table 2 compares SAFIRE, SegWave<sub>DCT</sub>, Seg-Wave w/o ASA, and the full SegWave across 2-, 3-, and 4-source settings. Seg-Wave attains the best p mIoU in every setting and the best p ARI on the 2- and 3-source splits, with its clearest advantage on the 2-source case (0.821 p mIoU vs SAFIRE’s 0.763). As the number of sources grows to four, all methods converge and the margins narrow, reflecting the increasing dificulty of separating many co-located manipulations. Qualitatively, Figures 3–5 show that SegWave produces sharper, better-separated source regions than SAFIRE, TruFor, and CAT-Net v2, consistent with the quantitative trends.

## 6 Conclusion

We presented SegWave, a prompt-guided framework that combines a Haar wavelet decomposition with an Adaptive Sub-band Attention module for image tampering localization and multi-source partitioning. By operating on oriented, spatially localized frequency subbands rather than global-frequency artifacts, Seg-

Wave remains efective on uncompressed formats where compression-dependent cues fade. It efectively localizes the spatially-confined boundary inconsistencies characteristic of splicing and generative edits. Our ablations isolate the source of these gains: rather than weighting the LH, HL, and HH subbands equally, adaptively reweighting them prevents informative cues from being diluted. This efect is most pronounced on the harder multi-source task, where SegWave clearly improves over both the equally-weighted variant and the DCT variant. The comparison between the two transforms also reveals a complementary structure: global DCT better captures compression-induced artifacts, while oriented DWT better captures localized boundary cues. This suggests a natural direction for future work: unifying both transforms so that a single model can draw on the compression-awareness of DCT and the localized sensitivity of wavelets, potentially yielding robust and generalizable tampering detection.

## Acknowledgements

V. Kumar is partially supported through the Visvesvaraya PhD Fellowship, implemented by the Ministry of Electronics and Information Technology (MeitY), Government of India.

## References

1. Agarwal, A., Agarwal, A., Sinha, S., Vatsa, M., Singh, R.: Md-csdnetwork: Multidomain cross stitched network for deepfake detection. In: IEEE International Conference on Automatic Face and Gesture Recognition. pp. 1–8 (2021)

2. Agarwal, A., Ratha, N., Vatsa, M., Singh, R.: Crafting adversarial perturbations via transformed image component swapping. IEEE Transactions on Image Processing 31, 7338–7349 (2022)

3. Agarwal, A., Singh, R., Vatsa, M.: Face anti-spoofing using haralick features. In: IEEE International Conference on Biometrics Theory, Applications and Systems. pp. 1–6 (2016)

4. Agarwal, A., Singh, R., Vatsa, M., Noore, A.: Boosting face presentation attack detection in multi-spectral videos through score fusion of wavelet partition images. Frontiers in Big Data 5, 836749 (2022)

5. Agarwal, A., Singh, R., Vatsa, M., Ratha, N.: Image transformation-based defense against adversarial perturbation on deep learning models. IEEE Transactions on Dependable and Secure Computing 18(5), 2106–2121 (2021)

6. Agarwal, A., Vatsa, M., Singh, R., Ratha, N.: Parameter agnostic stacked wavelet transformer for detecting singularities. Information Fusion 95, 415–425 (2023)

7. Ahmed, N., Natarajan, T., Rao, K.: Discrete cosine transform. IEEE Transactions on Computers C-23(1), 90–93 (1974). https://doi.org/10.1109/T-C.1974.223784

8. Anshumaan, D., Agarwal, A., Vatsa, M., Singh, R.: Wavetransform: Crafting adversarial examples via input decomposition. In: European Conference on Computer Vision. pp. 152–168. Springer (2020)

9. Barni, M., Bondi, L., Bonettini, N., Bestagini, P., Costanzo, A., Maggini, M., Tondi, B., Tubaro, S.: Aligned and non-aligned double jpeg detection using convolutional neural networks. Journal of Visual Communication and Image Representation 49, 153–163 (2017)

10. Bhattacharjee, A., Islam, K., Anan, K., Intesher, A., Fuad, A.A., Saha, U., Imtiaz, H.: Cae-net: Generalized deepfake image detection using convolution and attention mechanisms with spatial and frequency domain features. Journal of Visual Communication and Image Representation p. 104679 (2025)

11. Chen, Z., Chen, S., Yao, T., Sun, K., Ding, S., Lin, X., Cao, L., Ji, R.: Enhancing tampered text detection through frequency feature fusion and decomposition. In: European Conference on Computer Vision. pp. 200–217. Springer (2024)

12. Dong, C., Chen, X., Hu, R., Cao, J., Li, X.: Mvss-net: Multi-view multiscale supervised networks for image manipulation detection. IEEE Transactions on Pattern Analysis and Machine Intelligence PP (06 2022). https://doi.org/10.1109/TPAMI.2022.3180556

13. Dong, J., Wang, W., Tan, T.: Casia image tampering detection evaluation database. In: 2013 IEEE China Summit and International Conference on Signal and Information Processing. pp. 422–426 (2013). https://doi.org/10.1109/ChinaSIP.2013.6625374

14. Gadhiya, T.D., Roy, A.K., Mitra, S.K., Mall, V.: Use of discrete wavelet transform method for detection and localization of tampering in a digital medical image. In: 2017 IEEE Region 10 Symposium (TENSYMP). pp. 1–5 (2017). https://doi.org/10.1109/TENCONSpring.2017.8070082

15. Guillaro, F., Cozzolino, D., Sud, A., Dufour, N., Verdoliva, L.: Trufor: Leveraging all-round clues for trustworthy image forgery detection and localization (2023), https://arxiv.org/abs/2212.10957

16. Guo, S., Cui, J., Hong, R.: Rethinking vlms for image forgery detection and localization (2026), https://arxiv.org/abs/2603.12930

17. Hao, J., Zhang, Z., Yang, S., Xie, D., Pu, S.: Transforensics: Image forgery localization with dense self-attention (2021), https://arxiv.org/abs/2108.03871

18. Hayat, K., Qazi, T.: Forgery detection in digital images via discrete wavelet and discrete cosine transforms. Computers & Electrical Engineering 62, 448– 458 (2017). https://doi.org/https://doi.org/10.1016/j.compeleceng.2017.03.013, https://www.sciencedirect.com/science/article/pii/S0045790617305785

19. Hsu, Y.F., Chang, S.F.: Detecting image splicing using geometry invariants and camera characteristics consistency. In: International Conference on Multimedia and Expo (2006)

20. Huh, M., Liu, A., Owens, A., Efros, A.A.: Fighting fake news: Image splice detection via learned self-consistency (2018), https://arxiv.org/abs/1805.04096

21. Kanwal, N., Girdhar, A., Kaur, L., Bhullar, J.S.: Detection of digital image forgery using fast fourier transform and local features. In: 2019 International Conference on Automation, Computational and Technology Management (ICACTM). pp. 262– 267 (2019). https://doi.org/10.1109/ICACTM.2019.8776709

22. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., Doll´ar, P., Girshick, R.: Segment anything. arXiv:2304.02643 (2023)

23. Kniaz, V.V., Knyaz, V., Remondino, F.: The point where reality meets fantasy: Mixed adversarial generators for image splice detection. Advances in neural information processing systems 32 (2019)

24. Korus, P., Huang, J.: Multi-scale analysis strategies in prnu-based tampering localization. IEEE Transactions on Information Forensics and Security 12(4), 809–824 (2017). https://doi.org/10.1109/TIFS.2016.2636089

25. Kwon, M.J., Lee, W., Nam, S.H., Son, M., Kim, C.: Safire: Segment any forged image region. arXiv preprint arXiv:2412.08197 (2024)

26. Kwon, M.J., Nam, S.H., Yu, I.J., Lee, H.K., Kim, C.: Learning jpeg compression artifacts for image manipulation detection and localization. International Journal of Computer Vision 130(8), 1875–1895 (aug 2022). https://doi.org/10.1007/s11263- 022-01617-5

27. Kwon, M.J., Yu, I.J., Nam, S.H., Lee, H.K.: Cat-net: Compression artifact tracing network for detection and localization of image splicing. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. pp. 375–384 (2021)

28. Li, D., Zhu, J., Wang, M., Liu, J., Fu, X., Zha, Z.J.: Edge-aware regional message passing controller for image forgery localization. pp. 8222–8232 (06 2023). https://doi.org/10.1109/CVPR52729.2023.00795

29. Lin, T.Y., Maire, M., Belongie, S., Bourdev, L., Girshick, R., Hays, J., Perona, P., Ramanan, D., Zitnick, C.L., Doll´ar, P.: Microsoft coco: Common objects in context (2015), https://arxiv.org/abs/1405.0312

30. Liu, X., Liu, Y., Chen, J., Liu, X.: Pscc-net: Progressive spatio-channel correlation network for image manipulation detection and localization. IEEE Transactions on Circuits and Systems for Video Technology 32(11), 7505–7517 (2022). https://doi.org/10.1109/TCSVT.2022.3189545

31. Nichol, A., Dhariwal, P., Ramesh, A., Shyam, P., Mishkin, P., McGrew, B., Sutskever, I., Chen, M.: Glide: Towards photorealistic image generation and editing with text-guided difusion models (2022), https://arxiv.org/abs/2112.10741

32. Park, J., Cho, D., Ahn, W., Lee, H.K.: Double jpeg detection in mixed jpeg quality factors using deep convolutional neural network. In: Proceedings of the European Conference on Computer Vision (ECCV). pp. 636–652 (2018)

33. Qu, C., Liu, C., Liu, Y., Chen, X., Peng, D., Guo, F., Jin, L.: Towards robust tampered text detection in document image: New dataset and new solution. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5937–5946 (2023)

34. Qu, C., Zhong, Y., Guo, F., Jin, L.: Revisiting tampered scene text detection in the era of generative ai (2025), https://arxiv.org/abs/2407.21422

35. Sheng, Z., Wu, J., Lu, W., Zhou, J.: Weakly-supervised image forgery localization via vision-language collaborative reasoning framework (2025), https://arxiv.org/abs/2508.01338

36. Skodras, A.: Discrete wavelet transform: An introduction (12 2015)

37. Sun, R., Zhang, Y., Yu, X., Li, M., Gao, J.: Dual-tree complex wavelet driven hierarchical spatial-frequency fusion learning for robust deepfake detection. IEEE Transactions on Dependable and Secure Computing (2026)

38. Wang, J., Wu, Z., Chen, J., Han, X., Shrivastava, A., Lim, S.N., Jiang, Y.G.: Objectformer for image manipulation detection and localization (2022), https://arxiv.org/abs/2203.14681

39. Wang, Q., Zhang, R.: Double jpeg compression forensics based on a convolutional neural network. EURASIP Journal on Information Security 2016(1), 23 (2016)

40. Xu, Z., Zhang, X., Li, R., Tang, Z., Huang, Q., Zhang, J.: Fakeshield: Explainable image forgery detection and localization via multi-modal large language models (2025), https://arxiv.org/abs/2410.02761

41. Zhang, F., Liu, J., Zhu, J., Sun, E., Li, D., Zhang, Q., Zha, Z.J.: Forgerygpt: A multimodal llm for interpretable image forgery detection and localization (2026), https://arxiv.org/abs/2410.10238