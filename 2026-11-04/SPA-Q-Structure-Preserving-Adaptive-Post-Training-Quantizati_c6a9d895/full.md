# SPA-Q: Structure-Preserving Adaptive Post-Training Quantization for Monocular Depth Estimation

Jaemin Choi<sup>1∗</sup> Jincheol Yang<sup>1∗</sup> Nahyun Lim<sup>1∗</sup> Yun-Seong Jeong<sup>1∗</sup> Matti Zinke<sup>2∗</sup> Hyunwoo Yu<sup>1∗</sup> Suk-Ju Kang<sup>1†</sup>

<sup>1</sup>Department of Electronic Engineering, Sogang University

<sup>2</sup>Department of Computer Science, Sogang University

{jam0225, yjc3232, julialim5, jus8077, mattizinke, hyunwoo137, sjkang}@sogang.ac.kr

## Abstract

Monocular depth estimation (MDE) has advanced rapidly with the emergence of foundation models such as Depth Anything. Their transformer-based architectures provide strong generalization across diverse scenes and domains, but also incur high computational and memory cost, making efficient deployment challenging. Post-Training Quantization (PTQ) provides an efficient and practical solution for model compression, yet low-bit PTQ remains challenging for MDE models. When applying PTQ to Depth Anything, we identify two key challenges: (i) independently minimizing the quantization error of the query and key projections fails to preserve the attention maps induced by their interaction, and (ii) quantization errors accumulate across layers, resulting in intermediate feature distribution shifts. To address these issues, we propose SPA-Q, a Structure-Preserving Adaptive PTQ framework for MDE, consisting of two main components. First, Attention-Preserving Calibration (APC) determines query and key quantization parameters by matching the full-precision attention distribution. Second, Channel-Wise Distribution Alignment (CWDA) learns channel-wise affine transformations to mitigate quantization-induced distribution shifts, and the learned parameters are absorbed into the weights after training. Experimental results show that SPA-Q consistently outperforms existing PTQ methods under 4-bit quantization, achieving an average 25.9% reduction in AbsRel and a 17.3% improvement in $\delta _ { 1 }$ across NYUv2 and KITTI datasets.

## 1 Introduction

Monocular depth estimation (MDE) is a fundamental task in computer vision, with applications in robotics [1, 2], autonomous driving [3, 4], and 3D scene understanding [5–7]. Recent advances in Vision Transformers (ViTs) [8–12] have significantly improved MDE, leading to strong foundation models such as MiDaS [13] and Depth Anything [14, 15]. Depth Anything achieves strong zero-shot generalization through large-scale training with labeled and pseudo-labeled data, while leveraging multi-scale intermediate features for dense depth prediction. However, its large model size and high computational cost hinder real-world deployment in resource-constrained environments.

Model quantization [16–21] is an effective approach for reducing the memory and computational cost of deep neural networks. Existing quantization methods can be broadly divided into quantizationaware training (QAT) [22–24] and post-training quantization (PTQ) [25–28]. QAT incorporates quantization effects during training and often achieves higher accuracy, but it requires additional training and access to training data. In contrast, PTQ applies quantization to a pre-trained model without full retraining, making it suitable for practical deployment. Although PTQ has been widely studied for recognition models, its application to MDE remains underexplored. Unlike recognition tasks, depth estimation requires preserving fine-grained geometric and structural information in intermediate features, making it more sensitive to quantization-induced distortions.

In our study, we identify two key limitations of low-bit PTQ for transformer-based MDE models. (i) Independent quantization error minimization for query and key projections is not well aligned with preserving the resulting attention distribution. Existing PTQ methods [25–28] typically calibrate each quantizer by minimizing the discrepancy between full-precision and quantized outputs of each projection. However, in self-attention, the attention map is determined by the query-key interaction passed through the softmax operation, rather than by the fidelity of each projection. Therefore, independently minimizing the quantization error of the query and key projections does not guarantee preservation of the resulting attention map. This issue is especially critical for MDE, as distorted attention maps can corrupt features aggregated by the decoder, thereby degrading scene geometry and object boundaries in depth predictions. (ii) Quantization errors accumulate across layers and induce distribution shifts in intermediate features. Since the DPT decoder fuses features from multiple encoder stages, even small shifts in feature distributions can propagate through the decoder and degrade depth prediction. In practice, low-bit quantization induces noticeable distribution shifts in intermediate features, making them increasingly mismatched with full-precision counterparts.

To address these limitations, we propose SPA-Q, a novel Structure-Preserving Adaptive PTQ framework for transformer-based MDE models. SPA-Q is designed for architectures whose decoders aggregate multi-stage encoder features to produce dense depth predictions. It consists of two complementary components. Attention-Preserving Calibration (APC) determines the quantization parameters of the query and key projections by directly preserving the full-precision attention distribution instead of minimizing projection errors independently. In particular, APC minimizes the KL divergence between the full-precision and quantized attention distributions, thereby better capturing their discrepancy after softmax. Channel-Wise Distribution Alignment (CWDA) addresses distribution shifts in intermediate features through learnable channel-wise affine transformations, which are optimized with feature-level and depth-level objectives in an end-to-end manner. The learned affine parameters are then fused into the weights, introducing no additional inference overhead. Our contributions are summarized as follows:

• We identify two limitations of low-bit PTQ for transformer-based MDE models. First, independent quantization error minimization for query-key projections fails to preserve the resulting attention map. Second, accumulated quantization errors induce distribution shift in intermediate features.

• We propose SPA-Q, a Structure-Preserving Adaptive PTQ method for MDE. SPA-Q introduces Attention-Preserving Calibration (APC), which calibrates query and key quantizers with an attention-level objective, and Channel-Wise Distribution Alignment (CWDA), a channel-wise alignment scheme for correcting intermediate feature distribution shifts.

• Extensive experiments on relative depth estimation benchmarks show that SPA-Q outperforms existing PTQ methods under 4-bit quantization across all evaluated settings.

## 2 Related Work

## 2.1 Monocular Depth Estimation Models

MDE [29, 30] refers to the task of predicting a dense depth map from a single image, where each pixel reflects the 3D structure of the scene. Recent advances driven by large-scale data and transformerbased architectures have significantly improved depth prediction, leading to models such as MiDaS [13] and Depth Anything [14, 15]. MiDaS improves generalization by training on multiple datasets with varying scales and distributions, reducing dependence on a single dataset. Building on this, Depth Anything adopts a DINOv2 [11]-pretrained ViT encoder. Depth Anything v1 leverages both labeled and unlabeled data, while v2 further improves performance by incorporating large-scale synthetic and pseudo-labeled data. Both MiDaS and Depth Anything are typically built on the DPT [31] architecture, which aggregates multi-scale features using a transformer encoder and reconstructs them with a multi-scale decoder. We focus on Depth Anything v1 and v2 for evaluation, as they represent state-of-the-art MDE models with strong generalization performance.

## 2.2 Post-Training Quantization

PTQ [16–18] has been widely studied as an efficient approach for compressing neural networks without requiring full retraining. Most existing PTQ methods are designed for recognition tasks such as image classification. BRECQ [25] minimizes quantization error via block-wise reconstruction using a second-order approximation. By randomly dropping activation quantization during reconstruction, QDrop [26] mitigates overfitting and improves generalization. RepQ-ViT [27] further reduces inter-channel variation through scale reparameterization, enabling channel-wise calibration with tensor-wise inference. However, these methods have been primarily designed for and evaluated on classification tasks [8, 9, 32], and do not adequately address dense prediction tasks. In MDE, predic tions rely on multi-scale intermediate features, making the task more sensitive to quantization errors. To address this limitation, QSCA [33] introduces lightweight auxiliary modules for residual error compensation. Despite its effectiveness, this approach incurs additional parameters and computational overhead, which is undesirable for efficient deployment. In this work, we propose a PTQ method for MDE that preserves intermediate features required for dense depth prediction by maintaining attention maps and mitigating feature distribution shifts.

## 3 Method

## 3.1 Preliminaries

Uniform quantizer. A uniform quantizer [16–18] maps full-precision values to evenly spaced discrete levels. Given an input x and bit-width b, the quantized integer x¯ and the dequantized value xˆ are computed as:

$$
\bar { x } = \mathrm { c l i p } \left( \left\lfloor \frac { x } { s } \right\rceil + z , 0 , 2 ^ { b } - 1 \right) , \quad \hat { x } = s \cdot ( \bar { x } - z ) ,\tag{1}
$$

where the step size s and zero-point z are

$$
s = { \frac { \operatorname* { m a x } ( x ) - \operatorname* { m i n } ( x ) } { 2 ^ { b } - 1 } } , \quad z = \left\lfloor { \frac { - \operatorname* { m i n } ( x ) } { s } } \right\rfloor ,\tag{2}
$$

and $\lfloor \cdot \rceil$ denotes rounding to the nearest integer, while ${ \mathrm { c l i p } } ( \cdot , 0 , 2 ^ { b } - 1 )$ constrains the quantized integer to the valid range $[ 0 , 2 ^ { b } - 1 ]$

Log2 quantizer. As a basic form of logarithmic quantizer, the log2 quantizer [21] maps inputs to quantization levels defined on a base-2 logarithmic scale. Compared to uniform quantization, it allocates more quantization levels to small values, making it better suited for highly skewed or long-tailed distributions such as post-Softmax activations. Given an input x and step size s, the quantized value xˆ is computed as:

$$
\bar { x } = \mathrm { c l i p } \left( \left\lfloor - \log _ { 2 } \frac { x } { s } \right\rceil , 0 , 2 ^ { b } - 1 \right) , \quad \hat { x } = s \cdot 2 ^ { - \bar { x } } .\tag{3}
$$

The log2 quantizer defines quantization levels on an exponential scale, making it more effective for values concentrated near zero. Accordingly, we adopt the log2 quantizer for post-Softmax activations.

## 3.2 Motivation

To effectively quantize transformer-based MDE models, we analyze how low-bit quantization degrades representations used for dense depth prediction. In particular, we examine two sources of distortion: attention maps resulting from query-key interaction and intermediate feature distributions.

Mismatch between local quantization error minimization and attention preservation. Existing PTQ methods typically calibrate the query and key projections by minimizing local quantization error. Let $\mathbf { \bar { \boldsymbol { X } } } _ { Q } \in \mathbb { R } ^ { N _ { q } ^ { \bullet } \times D }$ and $\boldsymbol { X } _ { K } \in \mathbb { R } ^ { N _ { k } \times D }$ denote the input features to the query and key projections, respectively, where $N _ { q }$ and $N _ { k }$ are the numbers of query and key tokens, and $\dot { D }$ is the input channel dimension. The query, key, and attention map are then given by:

$$
\mathbf { Q } = X _ { Q } W _ { Q } , \qquad \mathbf { K } = X _ { K } W _ { K } , \qquad \mathbf { A } = \mathrm { s o f t m a x } \left( { \frac { \mathbf { Q } \mathbf { K } ^ { \top } } { \sqrt { d _ { h } } } } \right) ,\tag{4}
$$

where $W _ { Q } , W _ { K } \in \mathbb { R } ^ { D \times d _ { h } }$ are the corresponding projection weights, and $d _ { h }$ denotes the head dimension. This formulation shows that the attention map is determined by the query-key interaction through

![](images/8bcc0a3c9fdec43560154eaaa4f029568b3032e012309dc4bc9519adb60427c8.jpg)  
(a)

![](images/0eca38f20b2da7d85cd42e2ef764fc5ad749e26d99a8876970df8362ee843c59.jpg)  
(b)

![](images/18268dd6df2b038a993b558c893128340886eebf4ac196a135ba943a81fd1613.jpg)  
(c)

Figure 1: Comparison of calibration objectives for query-key quantization, where all improvements are measured relative to conventional local quantization error minimization. (a) MSE improvement from logits-level calibration. (b) Attention-level metric improvement from logits-level calibration. (c) Attention-level metric improvement from the proposed APC.  
![](images/df1f5b84507d631ac57f5415d60d648c800a32fa84aa5cb63b167c56b2f5736e.jpg)

![](images/1aa29d942a5b8398e92d6513578e994549daeb6c71029d7b01f67184c3b45e86.jpg)

![](images/94a1bdcb0e6b29cea8fd4dc89085c46d2a4731b9fb613d2a42d6beb12ad94ed6.jpg)  
(a)

![](images/477020c60cc8b90d931550d1b99a2fc28a5e8a1a56fc8d6bf1a9790cf62911a5.jpg)

![](images/adc860c0007d1c7def382959e4aa64703c98efb1cde74c8d40eea563db106565.jpg)

(b)  
![](images/0ac4437d2ab0a4222e148440c2f5fda9a94df74a151c88a6a21821d29c39d3a8.jpg)  
Figure 2: Quantization-induced distortion in intermediate features under 4-bit quantization (W4A4). (a) Distribution shift in intermediate features caused by accumulated quantization errors across layers. (b) Distortion of intermediate feature maps under W4A4, as illustrated by comparisons of patch norms with the full-precision model.

the pre-softmax logits $\mathbf { Q K } ^ { \top } / \sqrt { d _ { h } } .$ , rather than by the fidelity of each projection alone. Nevertheless, conventional PTQ minimizes the local quantization error of the two projections independently:

$$
\boldsymbol { \theta } _ { Q } ^ { \star } = \arg \operatorname* { m i n } _ { \boldsymbol { \theta } _ { Q } } \left\| \mathbf { Q } - \hat { \mathbf { Q } } _ { \boldsymbol { \theta } _ { Q } } \right\| _ { F } ^ { 2 } , \qquad \boldsymbol { \theta } _ { K } ^ { \star } = \arg \operatorname* { m i n } _ { \boldsymbol { \theta } _ { K } } \left\| \mathbf { K } - \hat { \mathbf { K } } _ { \boldsymbol { \theta } _ { K } } \right\| _ { F } ^ { 2 } ,\tag{5}
$$

where $\theta _ { Q } = \{ s _ { Q } , z _ { Q } \}$ and $\theta _ { K } = \{ s _ { K } , z _ { K } \}$ denote the corresponding quantization parameter sets. This objective implicitly assumes that preserving Q and K individually is sufficient for preserving the resulting attention map, even though the attention map is produced by their interaction.

To verify this mismatch empirically, we replace the local quantization error minimization in Eq. (5) with a calibration objective defined on the pre-softmax attention logits:

$$
\theta _ { Q } ^ { \star } , \theta _ { K } ^ { \star } = \arg \operatorname* { m i n } _ { \theta _ { Q } , \theta _ { K } } \left\| \frac { \mathbf { Q } \mathbf { K } ^ { \top } } { \sqrt { d _ { h } } } - \frac { \hat { \mathbf { Q } } _ { \theta _ { Q } } \hat { \mathbf { K } } _ { \theta _ { K } } ^ { \top } } { \sqrt { d _ { h } } } \right\| _ { F } ^ { 2 } .\tag{6}
$$

As shown in Figure 1 (a), although this logits-level objective often degrades the MSE of Q and K individually, it consistently improves the MSE of the attention logits $\mathbf { Q } \mathbf { \breve { K } } ^ { \top }$ . This observation reveals a fundamental mismatch between local quantization error minimization and query-key interaction: optimizing the two projections individually does not necessarily preserve their interaction. However, reducing the MSE of the attention logits does not consistently improve the attention map under MSE and KL divergence, as shown in Figure 1 (b). This gap arises from the nonlinearity of the softmax. The attention logits are unnormalized real-valued scores, whereas the attention map is a normalized probability distribution after softmax. Consequently, minimizing the Euclidean error of the attention logits is not equivalent to minimizing the discrepancy between the corresponding attention maps.

Feature distortion from accumulated quantization errors. Low-bit quantization progressively shifts intermediate feature distributions away from their full-precision counterparts, as illustrated in Figure 2 (a). This distortion arises because the quantized output of each layer is fed into the next layer, causing quantization errors to accumulate across layers. Under tensor-wise quantization, this accumulation can become more severe because a single quantization scale is shared across all channels, making the quantization error more sensitive to inter-channel variation [27, 34, 35]. As a result, the distribution shift can be further amplified, distorting not only feature statistics but also the corresponding feature maps, as shown in Figure 2 (b). This issue is especially critical in transformer-based MDE models, where the decoder relies on stage-wise features from multiple encoder blocks. Once these distorted features are propagated to the decoder, the quality of depth prediction deteriorates.

![](images/de9104ac7949370fe048039b93cd24f987fc1bfc0a15fc2887bf7d83d9940803.jpg)  
Figure 3: Overview of the SPA-Q framework. SPA-Q first applies Attention-Preserving Calibration (APC) to determine query and key quantization parameters by minimizing the row-wise KL divergence between full-precision and quantized attention maps. It then applies Channel-Wise Distribution Alignment (CWDA), where learnable affine parameters align quantized intermediate features with their full-precision counterparts using stage-wise feature maps from the transformer encoder and the depth prediction from the decoder. After optimization, the learned affine parameters are fused into the weights, introducing no additional inference overhead.

## 3.3 Attention-Preserving Calibration

As discussed in Sec. 3.2, local quantization error minimization for query and key projections is not well aligned with preserving the resulting attention map, and logits-level calibration is still insufficient for matching the attention distribution after softmax. Based on this observation, we propose Attention-Preserving Calibration (APC), which determines the query and key quantization parameters by directly minimizing the discrepancy between the full-precision and quantized attention distributions. Unlike objectives defined on individual projections or pre-softmax attention logits, APC defines the calibration objective on the attention map. Since each row of the attention map forms a normalized probability distribution over tokens, KL divergence is more appropriate than Euclidean error for measuring discrepancies in probability mass and capturing distortions in the attention distribution. Accordingly, APC minimizes the row-wise KL divergence between the full-precision attention map A and the quantized attention map A<sup>q</sup>:

$$
\theta _ { Q } ^ { \star } , \theta _ { K } ^ { \star } = \arg \operatorname* { m i n } _ { \theta _ { Q } , \theta _ { K } } \frac { 1 } { N _ { q } } \sum _ { i = 1 } ^ { N _ { q } } D _ { \mathrm { K L } } \bigl ( \mathbf { A } _ { i , : } \bigr | \big | \mathbf { A } _ { i , : } ^ { q } \bigr ) ,\tag{7}
$$

where $\mathbf { A } ^ { q } = \mathrm { s o f t m a x } ( \hat { \mathbf { Q } } _ { \theta _ { Q } } \hat { \mathbf { K } } _ { \theta _ { K } } ^ { \top } / \sqrt { d _ { h } } )$ . As shown in Figure 1 (c), the proposed APC objective consistently reduces both the MSE and KL divergence between the full-precision and quantized attention maps across most blocks. This demonstrates that attention-level calibration better aligns the quantized attention map with its full-precision counterpart than projection-wise or logits-level objectives. As a result, APC better maintains the structural information used to construct intermediate features for depth prediction.

## 3.4 Channel-Wise Distribution Alignment

To address the intermediate feature distribution shifts identified in Sec. 3.2, we propose Channel-Wise Distribution Alignment (CWDA), which aligns dequantized output activations with their full-precision counterparts through learnable channel-wise affine transformations. CWDA can be applied to output activations from both fully connected and convolutional layers. For notational simplicity, we represent the dequantized output activation as $\hat { \mathbf { Y } } \in \mathbb { R } ^ { N \times D }$ , where N denotes the number of tokens or flattened spatial positions, and $D$ denotes the number of channels. The aligned output $\hat { \mathbf { Y } } ^ { \star }$ is then given by:

$$
\hat { \mathbf { Y } } _ { : , d } ^ { \star } = \alpha _ { d } \hat { \mathbf { Y } } _ { : , d } + \beta _ { d } , \quad d = 1 , \ldots , D ,\tag{8}
$$

where $\alpha _ { d }$ and $\beta _ { d }$ are learnable scaling and shifting parameters for the d-th channel, optimized on training set. Defining $\pmb { \alpha } = [ \alpha _ { 1 } , \ldots , \alpha _ { D } ] ^ { \top }$ and $\beta = [ \bar { \beta } _ { 1 } , \ldots , \beta _ { D } ] ^ { \top }$ , Eq. (8) becomes

$$
\hat { \mathbf { Y } } ^ { \star } = \hat { \mathbf { Y } } \operatorname { D i a g } ( \alpha ) + \beta .\tag{9}
$$

Since the affine transformation is applied independently to each output channel, it can be absorbed into the weight and bias of the preceding transformation:

$$
\hat { \mathbf { Y } } ^ { \star } = \hat { \mathbf { X } } \hat { \mathbf { W } } \operatorname { D i a g } ( \alpha ) + \mathbf { b } \odot \alpha + \beta = \hat { \mathbf { X } } \hat { \mathbf { W } } ^ { \star } + \mathbf { b } ^ { \star } ,\tag{10}
$$

where $\hat { \mathbf { W } } ^ { \star } = \hat { \mathbf { W } } \mathrm { D i a g } ( \alpha )$ and $\mathbf { b } ^ { \star } = \mathbf { b } \odot \pmb { \alpha } + \pmb { \beta }$ . Therefore, CWDA introduces no additional inference overhead after training.

End-to-end optimization. The DINOv2 [11] encoder and DPT [31] decoder jointly determine the final depth prediction in MDE. Quantization-induced distortion in intermediate features propagates through the network and ultimately affects the predicted depth map. To account for both representationlevel and output-level effects, we optimize all CWDA parameters in an end-to-end manner using feature-level and depth-level objectives. To preserve the stage-wise features delivered to the decoder, we minimize the $\ell _ { 1 }$ distance between the full-precision and quantized stage features:

$$
\mathcal { L } _ { \mathrm { f e a t } } = \frac { 1 } { \vert \boldsymbol { S } \vert } \sum _ { s \in \boldsymbol { S } } \left. \mathbf { F } _ { s } - \hat { \mathbf { F } } _ { s } \right. _ { 1 } ,\tag{11}
$$

where S denotes the set of stage indices whose outputs are fed to the decoder, and $\mathbf { F } _ { s } , \hat { \mathbf { F } } _ { s } \in \mathbb { R } ^ { N \times C }$ denote the full-precision and quantized feature maps at stage s, respectively. To preserve the output depth structure, we minimize the SILog loss [13] between full-precision and quantized depth maps:

$$
\mathcal { L } _ { \mathrm { S I L o g } } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } z _ { i } ^ { 2 } - \frac { 1 } { n ^ { 2 } } \left( \sum _ { i = 1 } ^ { n } z _ { i } \right) ^ { 2 } , \qquad z _ { i } = \log \hat { d } _ { i } - \log d _ { i } ,\tag{12}
$$

where $d _ { i }$ and $\hat { d } _ { i }$ denote the full-precision and quantized depth maps at pixel i, respectively, and n is the number of valid pixels. The final objective is defined as:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { f e a t } } + \lambda _ { \mathrm { S I L o g } } \mathcal { L } _ { \mathrm { S I L o g } } , } \end{array}\tag{13}
$$

which is used to jointly optimize all CWDA parameters in the encoder and decoder.

## 4 Experiments

## 4.1 Experimental Setup

Models, datasets, and evaluation. We use Depth Anything v1 [14] and v2 [15], built on the DINOv2 [11] encoder with ViT-S and ViT-B backbones, as representative foundation models for monocular depth estimation. We evaluate SPA-Q on five MDE benchmarks: NYUv2 [36], KITTI [37], ETH3D [38], DIODE [39], and Sintel [40]. These datasets correspond to the official zero-shot relative depth evaluation benchmarks used in Depth Anything. For evaluation, we report standard depth estimation metrics, including the absolute relative error (AbsRel) and threshold accuracy $\delta _ { 1 }$ AbsRel measures the mean relative difference between predicted and reference depth values, while $\delta _ { 1 }$ measures the percentage of predictions satisfying a threshold ratio of 1.25.

Table 1: Comparison of PTQ methods on NYUv2 [36], KITTI [37], ETH3D [38], DIODE [39], and Sintel [40] for zero-shot relative depth estimation under 4-bit quantization. E. denotes the encoder backbone used in the MDE architectures.
<table><tr><td rowspan="3">Dataset</td><td rowspan="3">Method</td><td rowspan="3">W/A</td><td colspan="4">Depth Anything v1</td><td colspan="4">Depth Anything v2</td></tr><tr><td colspan="2">E. ViT-S</td><td colspan="2">E. ViT-B</td><td colspan="2">E. ViT-S</td><td colspan="2">E. ViT-B</td></tr><tr><td>AbsRel ↓</td><td>δ1↑</td><td>AbsRel ↓</td><td>δ1↑</td><td>AbsRel ↓</td><td>δ1↑</td><td>AbsRel ↓</td><td>δ1↑</td></tr><tr><td rowspan="6">NYUv2</td><td>FP</td><td>32/32</td><td>0.0525</td><td>0.9720</td><td>0.0459</td><td>0.9791</td><td>0.0513</td><td>0.9736</td><td>0.0460</td><td>0.9770</td></tr><tr><td>BRECQ [25]</td><td>4/4</td><td>0.2535</td><td>0.5395</td><td>0.2886</td><td>0.4692</td><td>0.2714</td><td>0.5042</td><td>0.2910</td><td>0.4646</td></tr><tr><td>QDrop [26]</td><td>4/4</td><td>0.1742</td><td>0.7166</td><td>0.2334</td><td>0.5785</td><td>0.1773</td><td>0.7115</td><td>0.2332</td><td>0.5794</td></tr><tr><td>RepQ-ViT [27]</td><td>4/4</td><td>0.1959</td><td>0.6639</td><td>0.2539</td><td>0.5410</td><td>0.2053</td><td>0.6385</td><td>0.2417</td><td>0.5634</td></tr><tr><td>QSCA [33]</td><td>4/4</td><td>0.1377</td><td>0.8097</td><td>0.1875</td><td>0.6845</td><td>0.1361</td><td>0.8151</td><td>0.1875</td><td>0.6845</td></tr><tr><td>SPA-Q (Ours)</td><td>4/4</td><td>0.1090</td><td>0.8771</td><td>0.1021</td><td>0.8991</td><td>0.1081</td><td>0.8808</td><td>0.1001</td><td>0.9060</td></tr><tr><td rowspan="6">KITTI</td><td>FP</td><td>32/32</td><td>0.0818</td><td>0.9369</td><td>0.0804</td><td>0.9396</td><td>0.0832</td><td>0.9340</td><td>0.0814</td><td>0.9389</td></tr><tr><td>BRECQ [25]</td><td>4/4</td><td>0.3719</td><td>0.3522</td><td>0.3989</td><td>0.3160</td><td>0.3906</td><td>0.3344</td><td>0.3990</td><td>0.3175</td></tr><tr><td>QDrop [26]</td><td>4/4</td><td>0.3934</td><td>0.3234</td><td>0.3855</td><td>0.3338</td><td>0.3620</td><td>0.3748</td><td>0.3412</td><td>0.4082</td></tr><tr><td>RepQ-ViT [27]</td><td>4/4</td><td>0.3434</td><td>0.4159</td><td>0.3540</td><td>0.3915</td><td>0.3634</td><td>0.3897</td><td>0.3906</td><td>0.3287</td></tr><tr><td>QSCA [33]</td><td>4/4</td><td>0.1874</td><td>0.7273</td><td>0.2365</td><td>0.6203</td><td>0.2067</td><td>0.6794</td><td>0.2296</td><td>0.6174</td></tr><tr><td>SPA-Q (Ours)</td><td>4/4</td><td>0.1622</td><td>0.7826</td><td>0.1840</td><td>0.7230</td><td>0.1807</td><td>0.7449</td><td>0.1696</td><td>0.7702</td></tr><tr><td rowspan="7">ETH3D</td><td>FP</td><td>32/32</td><td>0.0584</td><td>0.9652</td><td>0.0513</td><td>0.9741</td><td>0.0548</td><td>0.9701</td><td>0.0467</td><td>0.9791</td></tr><tr><td>BRECQ [25]</td><td>4/4</td><td>0.2829</td><td>0.5082</td><td>0.2958</td><td>0.4874</td><td>0.2920</td><td>0.4962</td><td>0.2963</td><td>0.4870</td></tr><tr><td>QDrop [26]</td><td>4/4</td><td>0.2351</td><td>0.6069</td><td>0.2706</td><td>0.5373</td><td>0.2288</td><td>0.6186</td><td>0.2697</td><td>0.5341</td></tr><tr><td>RepQ-ViT [27]</td><td>4/4</td><td>0.2195</td><td>0.6286</td><td>0.2491</td><td>0.5670</td><td>0.2215</td><td>0.6299</td><td>0.2606</td><td>0.5538</td></tr><tr><td>QSCA [33]</td><td>4/4</td><td>0.1791</td><td>0.7332</td><td>0.2241</td><td>0.6309</td><td>0.1983</td><td>0.6791</td><td>0.2197</td><td>0.6298</td></tr><tr><td>SPA-Q (Ours)</td><td>4/4</td><td>0.1485</td><td>0.8071</td><td>0.1512</td><td>0.7943</td><td>0.1533</td><td>0.7958</td><td>0.1444</td><td>0.8140</td></tr><tr><td>FP</td><td>32/32</td><td>0.0753</td><td>0.9413</td><td>0.0745</td><td>0.9474</td><td>0.0721</td><td>0.9426</td><td>0.0701</td><td>0.9498</td></tr><tr><td rowspan="6">DIODE</td><td>BRECQ [25]</td><td>4/4</td><td>0.2079</td><td>0.6755</td><td>0.2201</td><td>0.6464</td><td>0.2173</td><td>0.6538</td><td>0.2211</td><td>0.6441</td></tr><tr><td>QDrop [26]</td><td>4/4</td><td>0.1864</td><td>0.7275</td><td>0.2058</td><td>0.6819</td><td>0.1848</td><td>0.7279</td><td>0.2062</td><td>0.6777</td></tr><tr><td>RepQ-ViT [27]</td><td>4/4</td><td>0.1746</td><td>0.7538</td><td>0.2016</td><td>0.6891</td><td>0.1800</td><td>0.7552</td><td>0.1999</td><td>0.6945</td></tr><tr><td>QSCA [33]</td><td>4/4</td><td>0.1513</td><td>0.8033</td><td>0.1997</td><td>0.7099</td><td>0.1463</td><td>0.8135</td><td>0.1947</td><td>0.7094</td></tr><tr><td>SPA-Q (Ours)</td><td>4/4</td><td>0.1240</td><td>0.8653</td><td>0.1314</td><td>0.8595</td><td>0.1342</td><td>0.8524</td><td>0.1184</td><td>0.8901</td></tr><tr><td>FP</td><td>32/32</td><td>0.2296</td><td>0.7304</td><td>0.2281</td><td>0.7539</td><td>0.2680</td><td>0.6974</td><td>0.2576</td><td>0.7111</td></tr><tr><td rowspan="6">Sintel</td><td>BRECQ [25]</td><td></td><td>0.4787</td><td>0.3123</td><td></td><td>0.3085</td><td>0.4816</td><td>0.3055</td><td>0.4821</td><td>0.3008</td></tr><tr><td>QDrop [26]</td><td>4/4</td><td></td><td></td><td>0.4820 0.4867</td><td></td><td>0.4948</td><td>0.3184</td><td>0.4882</td><td>0.3151</td></tr><tr><td>RepQ-ViT [27]</td><td>4/4 4/4</td><td>0.4991 0.4722</td><td>0.3181 0.3619</td><td>0.4642</td><td>0.3106 0.3629</td><td>0.4693</td><td>0.3829</td><td>0.4838</td><td>0.3237</td></tr><tr><td>QSCA [33]</td><td>4/4</td><td>0.4531</td><td>0.3884</td><td>0.4781</td><td>0.4059</td><td>0.4474</td><td>0.3919</td><td>0.4679</td><td>0.4070</td></tr><tr><td>SPA-Q (Ours)</td><td>4/4</td><td>0.4377</td><td>0.4318</td><td>0.3949</td><td>0.5382</td><td>0.4211</td><td>0.4225</td><td>0.3811</td><td>0.4892</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Implementation details. For fair comparison, we follow the experimental settings of prior work [33]. We adopt the widely used percentile method for calibration, using channel-wise quantization for weights and tensor-wise quantization for activations. The calibration set is constructed by randomly sampling 16 images from the training set of NYUv2 for indoor scenes or KITTI for outdoor scenes. The CWDA parameters are initialized such that all elements of α are set to 1 and all elements of $\beta$ are set to 0. We optimize all CWDA parameters in an end-to-end manner for one epoch using 5% of the training images without ground-truth depth, with $\lambda _ { \mathrm { S I L o g } } = 0 . 5$ . We use the Adam optimizer with a learning rate of $1 \times 1 0 ^ { - 4 }$ and no weight decay. All experiments are conducted with a batch size of 1 on a single RTX 6000 Ada Generation GPU.

## 4.2 Experimental Results

Quantitative results. As shown in Table 1, we compare SPA-Q with existing PTQ methods, including BRECQ [25], QDrop [26], RepQ-ViT [27], and QSCA [33], under 4-bit quantization on five zero-shot relative depth estimation benchmarks. Overall, our framework consistently achieves the best performance in both AbsRel and $\delta _ { 1 }$ across all evaluated datasets, outperforming all compared PTQ methods. For Depth Anything v1 with the ViT-B backbone, the improvement is substantial on the two representative benchmarks, NYUv2 [36] and KITTI [37]. On NYUv2, our framework reduces AbsRel from 0.1875 to 0.1021 and improves $\delta _ { 1 }$ from 0.6845 to 0.8991. On KITTI, AbsRel is further reduced from 0.2365 to 0.1840 and $\delta _ { 1 }$ is improved from 0.6203 to 0.7230. These results show that the proposed framework effectively preserves depth estimation performance under severe low-bit quantization in both indoor and outdoor settings. Consistent improvements are also observed on ETH3D [38], DIODE [39], and Sintel [40], where SPA-Q achieves the best results among the compared PTQ methods. In particular, the improvement remains consistent on Sintel, a challenging synthetic benchmark, indicating that the proposed framework generalizes well not only to indoor and outdoor scenes but also to synthetic domains under 4-bit quantization.

![](images/bde76d8928fda524451fc9d86bf111c6d5fa4c7a744cf6ec572b72203a8efbf5.jpg)  
Figure 4: Qualitative comparison of 4-bit quantization results for Depth Anything with a ViT-S backbone on benchmarks: NYUv2 [36], KITTI [37], ETH3D [38], DIODE [39], and Sintel [40].

Qualitative results. To validate the effectiveness of the proposed framework, we present qualitative comparisons of predicted depth maps in Figure 4. Compared with QSCA [33], SPA-Q produces qual itatively better depth maps with improved structural consistency and sharper object boundaries across diverse indoor, outdoor, and synthetic scenes. For NYUv2 [36], it better captures the indoor layout, including the desk and chair boundaries, while providing clearer separation between foreground objects and distant regions. On KITTI [37], it produces smoother depth transitions across the road scene, with better separation between nearby vehicles and the background. Similarly, on ETH3D [38] and DIODE [39], thin structures such as wooden poles and ropes, as well as the boundary of the wall opening and object silhouettes, are well recovered, whereas QSCA largely fails to preserve them. Sintel [40] is particularly challenging due to its domain gap from real-world imagery and rendering effects such as atmospheric haze and motion blur. While QSCA collapses into a nearly radial depth pattern that loses the character entirely, SPA-Q recovers its contours, including the raised hand, consistent with the full-precision prediction. These results demonstrate that the proposed framework consistently preserves depth structure across diverse domains, including indoor, outdoor, and synthetic scenes.

## 4.3 Ablation Study

Comparison of calibration objectives. We conduct an ablation study on NYUv2 [36] and KITTI [37] to examine the effect of the proposed calibration objective under 4-bit quantization for Depth Anything v1 with a ViT-B backbone. As shown in Table 2, on NYUv2, replacing local quantization error minimization with a logits-level objective yields only a marginal improvement, with AbsRel changing from 0.1908 to 0.1839 and $\delta _ { 1 }$ from 0.6836 to 0.7013. This result suggests that directly considering query-key interaction at the logits level is still insufficient to consistently improve final depth performance. In contrast, replacing the logits-level objective with APC further improves performance, reducing AbsRel to 0.1444 and improving $\delta _ { 1 }$ to 0.7990. Finally, combining APC with

![](images/69e7c05060d985954b8dac54cff56a8b8b6a5480bbb9f7befc7b4134cb0c11a1.jpg)  
(a)

![](images/a9bb0aadc79a0c659cbb9c0c28ae249d73b6d355c89a355a3b0becdc176702a3.jpg)  
(b)  
Figure 5: Block-wise feature analysis under 4-bit quantization for Depth Anything v1 [14] with a ViT-B backbone across transformer blocks. (a) Mean squared error (MSE) between full-precision and quantized features. (b) Cosine similarity between full-precision and quantized features.

Table 2: Ablation study of calibration objectives under 4-bit quantization for Depth Anything v1 with a ViT-B backbone on NYUv2 [36] and KITTI [37].
<table><tr><td>Dataset</td><td>Method</td><td>AbsRel ↓</td><td> $\delta _ { 1 } \uparrow$ </td></tr><tr><td rowspan="3">NYUv2</td><td>Local</td><td>0.1908</td><td>0.6836</td></tr><tr><td>Logits-level</td><td>0.1839</td><td>0.7013</td></tr><tr><td>APC SPA-Q</td><td>0.1444 0.1021</td><td>0.7990 0.8991</td></tr><tr><td rowspan="3">KITTI</td><td>Local</td><td>0.3241</td><td>0.4366</td></tr><tr><td>Logits-level</td><td>0.3002</td><td>0.4828</td></tr><tr><td>APC</td><td>0.2490</td><td>0.5789</td></tr><tr><td rowspan="2"></td><td>SPA-Q</td><td>0.1840</td><td>0.7230</td></tr><tr><td></td><td></td><td></td></tr></table>

Table 3: Comparison of model size, parameter count, and bit operations (BOPs) under 4-bit quantization. SPA-Q\* denotes the model before fusing the learned CWDA parameters into the weights.
<table><tr><td></td><td>Encoder Method</td><td>Model size (MB) Params (M)</td><td></td><td>BOPs (G)</td></tr><tr><td rowspan="4">ViT-S</td><td>FP</td><td>94.55</td><td>24.79</td><td>85742</td></tr><tr><td>QSCA [33]</td><td>16.27</td><td>25.23</td><td>1544</td></tr><tr><td>SPA-Q*</td><td>14.50</td><td>24.87</td><td>1457</td></tr><tr><td>SPA-Q</td><td>14.16</td><td>24.79</td><td>1340</td></tr><tr><td rowspan="4">ViT-B</td><td>FP</td><td>371.82</td><td>97.47</td><td>273280</td></tr><tr><td>QSCA [33]</td><td>60.02</td><td>99.24</td><td>5092</td></tr><tr><td>SPA-Q*</td><td>51.84</td><td>97.65</td><td>4493</td></tr><tr><td>SPA-Q</td><td>51.16</td><td>97.47</td><td>4270</td></tr></table>

CWDA yields the best performance, reducing AbsRel to 0.1021 and improving $\delta _ { 1 }$ to 0.8991. KITTI shows a similar trend, where logits-level calibration provides only limited gains, while APC and the full SPA-Q framework lead to clear performance improvements.

Block-wise comparison of feature similarity. We further evaluate whether SPA-Q preserves encoder features for dense depth prediction across transformer blocks. Since the decoder reconstructs depth maps from encoder features, preserving their fidelity is important for maintaining depth structure under quantization. As shown in Figure 5 (a), SPA-Q consistently achieves the lowest MSE across most blocks, with the advantage becoming more pronounced in deeper blocks. Figure 5 (b) shows that SPA-Q yields the highest cosine similarity to the full-precision features throughout the encoder, indicating better block-wise feature preservation. These results demonstrate that APC and CWDA jointly preserve the encoder features required for depth prediction by improving attention-aware calibration and mitigating accumulated feature distortion.

Efficiency analysis. We further compare model size, parameter count, and bit operations (BOPs) of SPA-Q and QSCA [33] under 4-bit quantization for Depth Anything v1 with ViT-S and ViT-B backbones. The efficiency results in Table 3 are reported on the NYUv2 dataset. As shown in Table 3, SPA-Q is more efficient than QSCA in both model size and computational cost. For ViT-S, the fused SPA-Q model reduces model size from 16.27 MB to 14.16 MB and BOPs from 1544 G to 1340 G. For ViT-B, SPA-Q also yields a smaller model size than QSCA, decreasing it from 60.02 MB to 51.16 MB after fusion. Although SPA-Q before fusion introduces additional channel-wise affine parameters, these parameters are absorbed into the weights after training, so the fused model restores the parameter count to that of the original quantized network. These results show that SPA-Q improves depth estimation accuracy without introducing additional inference overhead.

## 5 Conclusion

In this paper, we addressed low-bit post-training quantization for MDE. We identified two key limitations of existing PTQ methods. Independently minimizing quantization error for query and key projections does not adequately preserve the resulting attention distribution, while accumulated quantization errors induce distribution shifts in intermediate features. To address these issues, we proposed SPA-Q, a structure-preserving adaptive PTQ framework that combines APC and CWDA to preserve attention distributions and mitigate feature distortion. Experimental results on multiple zero-shot depth estimation benchmarks showed that SPA-Q consistently outperforms existing PTQ methods under 4-bit quantization.

Limitations & Future Works. Although the proposed method consistently outperforms existing PTQ baselines under 4-bit quantization and yields improved qualitative results, the quantized depth maps remain visibly less clean than their full-precision counterparts. We conjecture that this limitation stems from the restricted set of depth values representable under low-bit quantization, which is more discrete than in full precision and can reduce the smoothness of predicted depth maps. Our study is limited to MDE with Depth Anything v1 and v2, partly due to GPU memory constraints for larger foundation models. Extending SPA-Q to larger foundation models and other depth estimation settings, such as stereo and video depth estimation, remains an important direction for future work.

## Acknowledgements

This work was supported by the Institute of Information & Communications Technology Planning & Evaluation(IITP) grant funded by the Korea government(MSIT) (No.RS-2025-02263706, Development of an analog-digital mixed ultra-low power neuromorphic edge SoC, 25%) and the IITP(Institute of Information & Communications Technology Planning & Evaluation)-ITRC(Information Technology Research Center) grant funded by the Korea government(Ministry of Science and ICT) (IITP-2026-RS-2023-00260091, 25%) and Institute of Information communications Technology Planning Evaluation (IITP) under the Artificial Intelligence Innovation Human Resources Development (IITP-RS-2026-25547954, 25%) grant funded by the Korea government(MSIT) and the K-On-Device AI Semiconductor RD Program (RS-2026-25620710, Development of On-Device AI Semiconductor-Based AI Algorithms, Intelligence Interconnection Framework, and Software Platforms for AI Operations, 25%) funded By the Ministry of Trade, Industry and Energy(MOTIE, Korea).

## References

[1] Xingshuai Dong, Matthew A Garratt, Sreenatha G Anavatti, and Hussein A Abbass. Towards real-time monocular depth estimation for robotics: A survey. IEEE Transactions on Intelligent Transportation Systems, 23(10):16940–16961, 2022.

[2] Diana Wofk, Fangchang Ma, Tien-Ju Yang, Sertac Karaman, and Vivienne Sze. Fastdepth: Fast monocular depth estimation on embedded systems. In 2019 International Conference on Robotics and Automation (ICRA), pages 6101–6108. IEEE, 2019.

[3] Xinshuo Weng and Kris Kitani. Monocular 3d object detection with pseudo-lidar point cloud. In Proceedings of the IEEE/CVF international conference on computer vision workshops, 2019.

[4] Yan Wang, Wei-Lun Chao, Divyansh Garg, Bharath Hariharan, Mark Campbell, and Kilian Q Weinberger. Pseudo-lidar from visual depth estimation: Bridging the gap in 3d object detection for autonomous driving. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 8445–8453, 2019.

[5] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications ofthe ACM, 65 (1):99–106, 2021.

[6] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis, et al. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139:1–139:14, 2023.

[7] Jincheol Yang, Matti Zinke, Beoungwoo Kang, Hyung Uk Cho, Hyunyoung Choi, SangGu Lee, and Suk-Ju Kang. Efficient monocular depth-based physical distance measurement for low-depth scales. IEEE Transactions on Industrial Informatics, 2025.

[8] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

[9] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou. Training data-efficient image transformers & distillation through attention. In International conference on machine learning, pages 10347–10357. PMLR, 2021.

[10] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021.

[11] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

[12] Hyunwoo Yu, Yubin Cho, Beoungwoo Kang, Seunghun Moon, Kyeongbo Kong, and Suk-Ju Kang. Embedding-free transformer with inference spatial reduction for efficient semantic segmentation. In European Conference on Computer Vision, pages 92–110. Springer, 2024.

[13] René Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE transactions on pattern analysis and machine intelligence, 44(3):1623–1637, 2022.

[14] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10371–10381, 2024.

[15] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. Advances in Neural Information Processing Systems, 37:21875–21911, 2024.

[16] Amir Gholami, Sehoon Kim, Zhen Dong, Zhewei Yao, Michael W Mahoney, and Kurt Keutzer. A survey of quantization methods for efficient neural network inference. In Low-power computer vision, pages 291–326. Chapman and Hall/CRC, 2022.

[17] Markus Nagel, Marios Fournarakis, Rana Ali Amjad, Yelysei Bondarenko, Mart Van Baalen, and Tijmen Blankevoort. A white paper on neural network quantization. arXiv preprint arXiv:2106.08295, 2021.

[18] Hao Wu, Patrick Judd, Xiaojie Zhang, Mikhail Isaev, and Paulius Micikevicius. Integer quantization for deep learning inference: Principles and empirical evaluation. arXiv preprint arXiv:2004.09602, 2020.

[19] Yoni Choukroun, Eli Kravchik, Fan Yang, and Pavel Kisilev. Low-bit quantization of neural networks for efficient inference. In 2019 IEEE/CVF International Conference on Computer Vision Workshop (ICCVW), pages 3009–3018. IEEE, 2019.

[20] Markus Nagel, Rana Ali Amjad, Mart Van Baalen, Christos Louizos, and Tijmen Blankevoort. Up or down? adaptive rounding for post-training quantization. In International conference on machine learning, pages 7197–7206. PMLR, 2020.

[21] Yang Lin, Tianyu Zhang, Peiqin Sun, Zheng Li, and Shuchang Zhou. Fq-vit: Post-training quantization for fully quantized vision transformer. arXiv preprint arXiv:2111.13824, 2021.

[22] Steven K Esser, Jeffrey L McKinstry, Deepika Bablani, Rathinakumar Appuswamy, and Dharmendra S Modha. Learned step size quantization. arXiv preprint arXiv:1902.08153, 2019.

[23] Yash Bhalgat, Jinwon Lee, Markus Nagel, Tijmen Blankevoort, and Nojun Kwak. Lsq+: Improving low-bit quantization through learnable offsets and better initialization. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition workshops, pages 696–697, 2020.

[24] Jungwook Choi, Zhuo Wang, Swagath Venkataramani, Pierce I-Jen Chuang, Vijayalakshmi Srinivasan, and Kailash Gopalakrishnan. Pact: Parameterized clipping activation for quantized neural networks. arXiv preprint arXiv:1805.06085, 2018.

[25] Yuhang Li, Ruihao Gong, Xu Tan, Yang Yang, Peng Hu, Qi Zhang, Fengwei Yu, Wei Wang, and Shi Gu. Brecq: Pushing the limit of post-training quantization by block reconstruction. arXiv preprint arXiv:2102.05426, 2021.

[26] Xiuying Wei, Ruihao Gong, Yuhang Li, Xianglong Liu, and Fengwei Yu. Qdrop: Randomly dropping quantization for extremely low-bit post-training quantization. arXiv preprint arXiv:2203.05740, 2022.

[27] Zhikai Li, Junrui Xiao, Lianwei Yang, and Qingyi Gu. Repq-vit: Scale reparameterization for posttraining quantization of vision transformers. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 17227–17236, 2023.

[28] Yunshan Zhong, Jiawei Hu, You Huang, Yuxin Zhang, and Rongrong Ji. Erq: Error reduction for posttraining quantization of vision transformers. In Forty-First International Conference on Machine Learning, 2024.

[29] Jin Han Lee, Myung-Kyu Han, Dong Wook Ko, and Il Hong Suh. From big to small: Multi-scale local planar guidance for monocular depth estimation. arXiv preprint arXiv:1907.10326, 2019.

[30] Shariq Farooq Bhat, Reiner Birkl, Diana Wofk, Peter Wonka, and Matthias Müller. Zoedepth: Zero-shot transfer by combining relative and metric depth. arXiv preprint arXiv:2302.12288, 2023.

[31] René Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction. In Proceedings ofthe IEEE/CVF international conference on computer vision, pages 12179–12188, 2021.

[32] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10012–10022, 2021.

[33] Jincheol Yang, Jaemin Choi, Matti Zinke, and Suk-Ju Kang. Qsca: Quantization with self-compensating auxiliary for monocular depth estimation. Advances in Neural Information Processing Systems, 38: 31444–31465, 2026.

[34] Lianwei Yang, Haisong Gong, Haokun Lin, Yichen Wu, Zhenan Sun, and Qingyi Gu. Dopq-vit: Towards distribution-friendly and outlier-aware post-training quantization for vision transformers. arXiv preprint arXiv:2408.03291, 2024.

[35] Tao Jiang, Yucheng Jiang, Xiwen Yao, Gong Cheng, and Junwei Han. Uq-vit: Harmonizing extreme activations with hardware-friendly uniform quantization in vision transformers. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 22354–22362, 2026.

[36] Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from rgbd images. In European conference on computer vision, pages 746–760. Springer, 2012.

[37] Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Vision meets robotics: The kitti dataset. The international journal ofrobotics research, 32(11):1231–1237, 2013.

[38] Thomas Schops, Johannes L Schonberger, Silvano Galliani, Torsten Sattler, Konrad Schindler, Marc Pollefeys, and Andreas Geiger. A multi-view stereo benchmark with high-resolution images and multicamera videos. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 3260–3269, 2017.

[39] Igor Vasiljevic, Nick Kolkin, Shanyi Zhang, Ruotian Luo, Haochen Wang, Falcon Z Dai, Andrea F Daniele, Mohammadreza Mostajabi, Steven Basart, Matthew R Walter, et al. Diode: A dense indoor and outdoor depth dataset. arXiv preprint arXiv:1908.00463, 2019.

[40] Daniel J Butler, Jonas Wulff, Garrett B Stanley, and Michael J Black. A naturalistic open source movie for optical flow evaluation. In European conference on computer vision, pages 611–625. Springer, 2012.

[41] Rundong Li, Yan Wang, Feng Liang, Hongwei Qin, Junjie Yan, and Rui Fan. Fully quantized network for object detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2810–2819, 2019.

[42] Zhihang Yuan, Chenhao Xue, Yiqi Chen, Qiang Wu, and Guangyu Sun. Ptq4vit: Post-training quantization for vision transformers with twin uniform quantization. In European conference on computer vision, pages 191–207. Springer, 2022.

[43] Minghao Fu, Hao Yu, Jie Shao, Junjie Zhou, Ke Zhu, and Jianxin Wu. Quantization without tears. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pages 4462–4472, 2025.

[44] Junjie Hu, Mete Ozay, Yan Zhang, and Takayuki Okatani. Revisiting single image depth estimation: Toward higher resolution maps with accurate object boundaries. In 2019 IEEE winter conference on applications of computer vision (WACV), pages 1043–1051. IEEE, 2019.

[45] Alexey Bochkovskiy, Amaël Delaunoy, Hugo Germain, Marcel Santos, Yichao Zhou, Stephan Richter, and Vladlen Koltun. Depth pro: Sharp monocular metric depth in less than a second. In International Conference on Learning Representations, volume 2025, pages 75602–75637, 2025.

[46] Ke Xian, Jianming Zhang, Oliver Wang, Long Mai, Zhe Lin, and Zhiguo Cao. Structure-guided ranking loss for single image depth prediction. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 608–617. IEEE, 2020.

[47] Tobias Koch, Lukas Liebel, Friedrich Fraundorfer, and Marco Körner. Evaluation of cnn-based singleimage depth estimation methods. In European Conference on Computer Vision, pages 331–348. Springer, 2018.

## Appendix

## A Extended Comparison with Additional PTQ Baselines

Table 4: Extended comparison with additional PTQ baselines on NYUv2 [36] and KITTI [37] for zero-shot relative depth estimation under 4-bit quantization. E. denotes the encoder backbone used in the MDE architectures.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td rowspan="2">W/A</td><td colspan="4">Depth Anything v1</td><td colspan="4">Depth Anything v2</td></tr><tr><td colspan="2">E. ViT-S</td><td colspan="2">E. ViT-B</td><td colspan="2">E. ViT-S</td><td colspan="2">E. ViT-B</td></tr><tr><td rowspan="10">NYUv2</td><td></td><td></td><td>AbsRel ↓</td><td>δ1↑</td><td>AbsRel ↓</td><td>δ1↑</td><td>AbsRel ↓</td><td>δ1↑</td><td>AbsRel ↓</td><td>δ1↑</td></tr><tr><td>FP</td><td>32/32</td><td>0.0525</td><td>0.9720</td><td>0.0459</td><td>0.9791</td><td>0.0513</td><td>0.9736</td><td>0.0460</td><td>0.9770</td></tr><tr><td>MinMax [16]</td><td>4/4</td><td>0.2728</td><td>0.5024</td><td>1.5735</td><td>0.1972</td><td>0.2815</td><td>0.4873</td><td>2.3712</td><td>0.1102</td></tr><tr><td>Percentile [41]</td><td>4/4</td><td>0.2006</td><td>0.6542</td><td>0.2522</td><td>0.5430</td><td>0.1935</td><td>0.6675</td><td>0.2543</td><td>0.5379</td></tr><tr><td>BRECQ [25]</td><td>4/4</td><td>0.2535</td><td>0.5395 0.7166</td><td>0.2886 0.2334</td><td>0.4692 0.5785</td><td>0.2714 0.1773</td><td>0.5042 0.7115</td><td>0.2910</td><td>0.4646</td></tr><tr><td>QDrop [26]</td><td>4/4</td><td>0.1742 0.2393</td><td>0.5693</td><td>0.2294</td><td>0.5895</td><td>0.2735</td><td>0.5034</td><td>0.2332</td><td>0.5794</td></tr><tr><td>PTQ4ViT [42]</td><td>4/4</td><td>0.1959</td><td>0.6639</td><td>0.2539</td><td>0.5410</td><td>0.2053</td><td></td><td>0.2694</td><td>0.5183</td></tr><tr><td>RepQ-ViT [27] ERQ [28]</td><td>4/4</td><td>0.1785</td><td>0.7062</td><td>0.2182</td><td>0.6126</td><td>0.1751</td><td>0.6385 0.7140</td><td>0.2417</td><td>0.5634</td></tr><tr><td>QwT [43]</td><td>4/4 4/4</td><td>0.1407</td><td>0.8007</td><td>0.2024</td><td>0.6486</td><td>0.1400</td><td>0.8050</td><td>0.2883</td><td>0.4705</td></tr><tr><td>QSCA [33]</td><td>4/4</td><td>0.1377</td><td>0.8097</td><td>0.1875</td><td>0.6845</td><td>0.1361</td><td>0.8151</td><td>0.1992 0.1875</td><td>0.6589</td></tr><tr><td>SPA-Q (Ours)</td><td>4/4</td><td>0.1090</td><td>0.8771</td><td>0.1021</td><td>0.8991</td><td>0.1081</td><td>0.8808</td><td>0.1001</td><td>0.6845 0.9060</td></tr><tr><td rowspan="10">FP</td><td></td><td>32/32</td><td>0.0818</td><td>0.9369</td><td>0.0804</td><td>0.9396</td><td>0.0832</td><td>0.9340</td><td>0.0814</td><td>0.9389</td></tr><tr><td>MinMax [16]</td><td>4/4</td><td>0.3770</td><td>0.3441</td><td>1.9612</td><td>0.2058</td><td>0.3938</td><td>0.3423</td><td>4.4358</td><td>0.0832</td></tr><tr><td>Percentile [41]</td><td>4/4</td><td>0.3418</td><td>0.4099</td><td>0.3876</td><td>0.3327</td><td>0.3668</td><td>0.3780</td><td>0.3932</td><td>0.3275</td></tr><tr><td>BRECQ [25]</td><td>4/4</td><td>0.3719</td><td>0.3522</td><td>0.3989</td><td>0.3160</td><td>0.3906</td><td>0.3344</td><td>0.3990</td><td>0.3175</td></tr><tr><td>QDrop [26]</td><td>4/4</td><td>0.3934</td><td>0.3234</td><td>0.3855</td><td>0.3338</td><td>0.3620</td><td>0.3748</td><td>0.3412</td><td>0.4082</td></tr><tr><td>PTQ4ViT [42]</td><td>4/4</td><td>0.3439</td><td>0.4106</td><td>0.3923</td><td>0.3251</td><td>0.3308</td><td>0.4187</td><td>0.3972</td><td>0.3200</td></tr><tr><td>RepQ-ViT [27]</td><td>4/4</td><td>0.3434</td><td>0.4159</td><td>0.2539</td><td>0.5410</td><td>0.3634</td><td>0.3897</td><td>0.3906</td><td>0.3287</td></tr><tr><td>ERQ [28]</td><td>4/4</td><td>0.3178</td><td>0.4847</td><td>0.3528</td><td>0.4241</td><td>0.3176</td><td>0.4616</td><td>0.3490</td><td>0.4066</td></tr><tr><td>QwT [43]</td><td>4/4</td><td>0.1802</td><td>0.6862</td><td>0.2417</td><td>0.5867</td><td>0.1951</td><td>0.6941</td><td>0.2539</td><td>0.5346</td></tr><tr><td></td><td>4/4</td><td>0.1874</td><td>0.7273</td><td>0.2365</td><td>0.6203</td><td>0.2067</td><td>0.6794</td><td>0.2296</td><td>0.6174</td></tr><tr><td>QSCA [33]</td><td>SPA-Q (Ours)</td><td>4/4</td><td>0.1622</td><td>0.7826</td><td>0.1840</td><td>0.7230</td><td>0.1807</td><td>0.7449</td><td>0.1696</td><td>0.7702</td></tr></table>

Table 4 provides an extended comparison with additional PTQ baselines on NYUv2 [36] and KITTI [37] under 4-bit quantization. In addition to the baselines reported in the main paper, we include MinMax [16], Percentile [41], PTQ4ViT [42], ERQ [28], and QwT [43] to further evaluate the robustness of SPA-Q across different calibration and reconstruction strategies. Across Depth Anything v1 [14] and v2 [15] with both ViT-S and ViT-B backbones, SPA-Q achieves the best performance in all evaluated settings. On NYUv2, SPA-Q consistently improves both AbsRel and $\delta _ { 1 }$ over all compared methods. The improvement is more pronounced for ViT-B backbones, where preserving attention distributions and aligning intermediate feature distributions are especially important under low-bit quantization. On KITTI, SPA-Q also achieves the best results across all model variants, showing that the proposed framework remains effective in outdoor driving scenes. Although QwT provides a strong baseline in some KITTI settings, SPA-Q still achieves lower AbsRel and higher $\delta _ { 1 }$ demonstrating the benefit of explicitly preserving depth-relevant intermediate representations.

Overall, these extended results confirm that the gains of SPA-Q are not limited to the main PTQ baselines. Instead, SPA-Q remains consistently effective against a broader set of PTQ methods, further supporting the importance of attention-preserving calibration and channel-wise distribution alignment for quantized monocular depth estimation.

Table 5: Extended comparison with additional PTQ baselines on NYUv2 [36] and KITTI [37] for zero-shot relative depth estimation under W4A6 and W4A8 quantization. E. denotes the encoder backbone used in the MDE architectures.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td rowspan="2">W/A</td><td colspan="4">Depth Anything v1</td><td colspan="4">Depth Anything v2</td></tr><tr><td>E. ViT-S</td><td></td><td>E. ViT-B</td><td></td><td>E. ViT-S</td><td></td><td>E. ViT-B</td><td></td></tr><tr><td></td><td></td><td></td><td>AbsRel ↓</td><td>δ1↑</td><td>AbsRel ↓</td><td>δ1↑</td><td>AbsRel ↓</td><td>δ1↑</td><td>AbsRel ↓</td><td>δ1↑</td></tr><tr><td></td><td>FP</td><td>32/32</td><td>0.0525</td><td>0.9720</td><td>0.0459</td><td>0.9791</td><td>0.0513</td><td>0.9736</td><td>0.0460</td><td>0.9770</td></tr><tr><td rowspan="20"></td><td>MinMax [16]</td><td>4/6</td><td>0.2417</td><td>0.5632</td><td>0.2748</td><td>0.4973</td><td>0.2599</td><td>0.5324</td><td>0.2866</td><td>0.4738</td></tr><tr><td>Percentile [41]</td><td>4/6</td><td>0.1050</td><td>0.8837</td><td>0.0958</td><td>0.9071</td><td>0.0891</td><td>0.9196</td><td>0.0831</td><td>0.9355</td></tr><tr><td>BRECQ [25]</td><td>4/6</td><td>0.2337</td><td>0.5786</td><td>0.2462</td><td>0.5542</td><td>0.2611</td><td>0.5269</td><td>0.2697</td><td>0.5084</td></tr><tr><td>QDrop [26]</td><td>4/6</td><td>0.2071</td><td>0.6369</td><td>0.1823</td><td>0.6987</td><td>0.1700</td><td>0.7285</td><td>0.1560</td><td>0.7619</td></tr><tr><td>PTQ4ViT [42]</td><td>4/6</td><td>0.2166</td><td>0.6187</td><td>0.2201</td><td>0.6122</td><td>0.2642</td><td>0.5159</td><td>0.2275</td><td>0.6095</td></tr><tr><td>RepQ-ViT [27]</td><td>4/6</td><td>0.1304</td><td>0.8256</td><td>0.1058</td><td>0.8867</td><td>0.1131</td><td>0.8657</td><td>0.0918</td><td>0.9158</td></tr><tr><td>ERQ [28]</td><td>4/6</td><td>0.0945</td><td>0.9075</td><td>0.0766</td><td>0.9390</td><td>0.0874</td><td>0.9245</td><td>0.0914</td><td>0.9152</td></tr><tr><td>QwT [43]</td><td>4/6</td><td>0.0884</td><td>0.9189</td><td>0.0928</td><td>0.9089</td><td>0.0833</td><td>0.9323</td><td>0.1000</td><td>0.8944</td></tr><tr><td>QSCA [33]</td><td>4/6</td><td>0.0810 0.0732</td><td>0.9333</td><td>0.0739</td><td>0.9441</td><td>0.0757</td><td>0.9450</td><td>0.0726</td><td>0.9468</td></tr><tr><td>NYUv2 SPA-Q (Ours)</td><td>4/6</td><td></td><td>0.9473</td><td>0.0623</td><td>0.9597</td><td>0.0735</td><td>0.9485</td><td>0.0647</td><td>0.9603</td></tr><tr><td>MinMax [16]</td><td>4/8</td><td>0.1522</td><td>0.7740</td><td>0.2253</td><td>0.5934</td><td>0.1345</td><td>0.8187</td><td>0.2507</td><td>0.5465</td></tr><tr><td>Percentile [41]</td><td>4/8</td><td>0.0861</td><td>0.9233</td><td>0.0648</td><td>0.9588</td><td>0.0798</td><td>0.9375</td><td>0.0604</td><td>0.9621</td></tr><tr><td>BRECQ [25]</td><td>4/8</td><td>0.1182</td><td>0.8533</td><td>0.2152</td><td>0.6166</td><td>0.0999</td><td>0.8927</td><td>0.2193</td><td>0.6095</td></tr><tr><td>QDrop [26]</td><td>4/8</td><td>0.0700</td><td>0.9504</td><td>0.0599</td><td>0.9627</td><td>0.0651</td><td>0.9592</td><td>0.0601</td><td>0.9621</td></tr><tr><td>PTQ4ViT [42]</td><td>4/8</td><td>0.2001 0.1066</td><td>0.6544</td><td>0.1877</td><td>0.6837</td><td>0.2596</td><td>0.5246</td><td>0.2190</td><td>0.6328</td></tr><tr><td>RepQ-ViT [27]</td><td></td><td>4/8</td><td>0.8833</td><td>0.0864</td><td>0.9259</td><td>0.1066</td><td>0.8792</td><td>0.0788</td><td>0.9367</td></tr><tr><td>ERQ [28]</td><td>4/8</td><td>0.0812</td><td>0.9333</td><td>0.0608</td><td>0.9632</td><td>0.0809</td><td>0.9355</td><td>0.0597</td><td>0.9641</td></tr><tr><td>QwT [43]</td><td>4/8</td><td>0.0769</td><td>0.9393</td><td>0.0789</td><td>0.9348</td><td>0.0663</td><td>0.9564</td><td>0.0808</td><td>0.9300</td></tr><tr><td>QSCA [33] SPA-Q (Ours)</td><td>4/8</td><td>0.0682 0.0664</td><td>0.9551</td><td>0.0657</td><td>0.9542</td><td>0.0679</td><td>0.9549</td><td>0.0599</td><td>0.9635</td></tr><tr><td></td><td>4/8</td><td></td><td>0.9574</td><td>0.0560</td><td>0.9672</td><td>0.0651</td><td>0.9591</td><td>0.0562</td><td>0.9682</td></tr><tr><td></td><td>FP</td><td>32/32</td><td>0.0818</td><td>0.9369</td><td>0.0804</td><td>0.9396</td><td>0.0832</td><td>0.9340</td><td>0.0814</td><td>0.9389</td></tr><tr><td rowspan="15">KITTI</td><td>MinMax [16]</td><td>4/6</td><td>0.3290</td><td>0.4467</td><td>0.3769</td><td>0.3586</td><td>0.3744</td><td>0.3609</td><td>0.3982</td><td>0.3161</td></tr><tr><td>Percentile [41]</td><td>4/6</td><td>0.1269</td><td>0.8558</td><td>0.1687</td><td>0.7580</td><td>0.1283</td><td>0.8613</td><td>0.1866</td><td>0.7219</td></tr><tr><td>BRECQ [25]</td><td>4/6</td><td>0.2851</td><td>0.5223</td><td>0.3529</td><td>0.4035</td><td>0.3332</td><td>0.4320</td><td>0.3497</td><td>0.4093</td></tr><tr><td>QDrop [26]</td><td>4/6</td><td>0.2408</td><td>0.6052</td><td>0.2611</td><td>0.5897</td><td>0.2790</td><td>0.5351</td><td>0.3037</td><td>0.5268</td></tr><tr><td>PTQ4ViT [42]</td><td>4/6</td><td>0.3230</td><td>0.4446</td><td>0.3848</td><td>0.3488</td><td>0.3614</td><td>0.3931</td><td>0.3917</td><td>0.3308</td></tr><tr><td>RepQ-ViT [27]</td><td>4/6</td><td>0.1502</td><td>0.7888</td><td>0.2602</td><td>0.5610</td><td>0.1292</td><td>0.8293</td><td>0.1458</td><td>0.8109</td></tr><tr><td>ERQ [28]</td><td>4/6</td><td>0.1244</td><td>0.8629</td><td>0.1133</td><td>0.8705</td><td>0.1144</td><td>0.8770</td><td>0.1115</td><td>0.8685</td></tr><tr><td>QwT [43]</td><td>4/6</td><td>0.1152</td><td>0.8844</td><td>0.1759</td><td>0.7292</td><td>0.1169</td><td>0.8872</td><td>0.1550</td><td>0.7700</td></tr><tr><td>QSCA [33]</td><td>4/6</td><td>0.1161</td><td>0.8857</td><td>0.1174</td><td>0.8722</td><td>0.1124</td><td>0.8893</td><td>0.1067</td><td>0.8873</td></tr><tr><td>SPA-Q (Ours)</td><td>4/6</td><td>0.1247</td><td>0.8849</td><td>0.1023</td><td>0.8907</td><td>0.1684</td><td>0.7892</td><td>0.0994</td><td>0.9003</td></tr><tr><td>MinMax [16]</td><td>4/8</td><td>0.3486</td><td>0.4215</td><td>0.3275</td><td>0.4845</td><td>0.3529</td><td>0.4237</td><td>0.3327</td><td>0.4376</td></tr><tr><td>Percentile [41]</td><td>4/8</td><td>0.1111</td><td>0.8740</td><td>0.1263</td><td>0.8564</td><td>0.1051</td><td>0.8892</td><td>0.1045</td><td>0.9003</td></tr><tr><td>BRECQ [25]</td><td>4/8</td><td>0.3532</td><td>0.4070</td><td>0.3317</td><td>0.4513</td><td>0.3194</td><td>0.4575</td><td>0.3239</td><td>0.4493</td></tr><tr><td>QDrop [26]</td><td>4/8</td><td>0.3071</td><td>0.4883</td><td>0.2754</td><td>0.5422</td><td>0.2713</td><td>0.5368</td><td>0.2688</td><td>0.5616</td></tr><tr><td>PTQ4ViT [42]</td><td>4/8 4/8</td><td>0.3224 0.1375</td><td>0.4580 0.8145</td><td>0.3813 0.2227</td><td>0.3550 0.6436</td><td>0.3538 0.1212</td><td>0.3909 0.8431</td><td>0.3969 0.1136</td><td>0.3205 0.8683</td></tr><tr><td>RepQ-ViT [27] ERQ [28]</td></table>

We further evaluate SPA-Q on additional MDE models, MiDaS-BEiT-B-384 [13], DPT-L-384 [31], and ZoeDepth [30], as shown in Table 6. Models with BEiT backbones, namely MiDaS and ZoeDepth, suffer more severe degradation under 4-bit quantization than DPT-L. Nevertheless, SPA-Q consistently outperforms the state-of-the-art QSCA [33] across all models and datasets.

## B Generalization Across MDE Models

Table 6: Comparison of PTQ methods on NYUv2 [36] and KITTI [37] with additional MDE architectures, MiDaS-BEiT-B-384 [13], DPT-L-384 [31], and ZoeDepth [30], under 4-bit quantization.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td rowspan="2">W/A</td><td colspan="2">MiDaS-BEiT-B-384</td><td colspan="2">DPT-L-384</td><td colspan="2">ZoeDepth (Metric)</td></tr><tr><td>AbsRel ↓</td><td> $\delta _ { 1 } \uparrow$ </td><td>AbsRel ↓</td><td> $\delta _ { 1 } \uparrow$ </td><td>AbsRel ↓</td><td> $\delta _ { 1 } \uparrow$ </td></tr><tr><td rowspan="4">NYUv2</td><td>FP</td><td>32/32</td><td>0.0663</td><td>0.9568</td><td>0.0947</td><td>0.9112</td><td>0.0755</td><td>0.9544</td></tr><tr><td>RepQ-ViT [27]</td><td>4/4</td><td>0.2603</td><td>0.5254</td><td>0.1394</td><td>0.8139</td><td>0.4949</td><td>0.1033</td></tr><tr><td>QSCA [33]</td><td>4/4</td><td>0.2253</td><td>0.5998</td><td>0.1380</td><td>0.8192</td><td>0.3969</td><td>0.2054</td></tr><tr><td>SPA-Q (Ours)</td><td>4/4</td><td>0.1803</td><td>0.7081</td><td>0.1276</td><td>0.8425</td><td>0.3051</td><td>0.3894</td></tr><tr><td rowspan="4">KITTI</td><td>FP</td><td>32/32</td><td>0.1928</td><td>0.7076</td><td>0.1079</td><td>0.8858</td><td>0.0589</td><td>0.9664</td></tr><tr><td>RepQ-ViT [27]</td><td>4/4</td><td>0.3761</td><td>0.3580</td><td>0.1661</td><td>0.7824</td><td>0.4650</td><td>0.1529</td></tr><tr><td>QSCA [33]</td><td>4/4</td><td>0.3740</td><td>0.3628</td><td>0.1650</td><td>0.7810</td><td>0.2701</td><td>0.5237</td></tr><tr><td>SPA-Q (Ours)</td><td>4/4</td><td>0.3250</td><td>0.4383</td><td>0.1427</td><td>0.8379</td><td>0.2551</td><td>0.6171</td></tr></table>

To examine whether SPA-Q generalizes beyond Depth Anything, we further evaluate it on three additional MDE models with different backbones and prediction targets: MiDaS-BEiT-B-384 [13] and DPT-L-384 [31] for relative depth estimation, and ZoeDepth [30] for metric depth estimation. As shown in Table 6, SPA-Q achieves the best AbsRel and $\delta _ { 1 }$ for all three models on both NYUv2 [36] and KITTI [37] under 4-bit quantization, consistently outperforming RepQ-ViT [27] and QSCA [33]. These results indicate that SPA-Q is not tailored to a specific architecture and is applicable to a broad range of MDE models, including those for metric depth estimation.

## C Evaluation on Structural Metrics

Table 7: Comparison of edge, boundary, and planarity metrics for Depth Anything v1 [14] under W4A4 and W4A6 quantization. Edge F1 [44], Boundary F1 [45], and ORD [46] are evaluated on NYUv2 [36], and planarity errors (PE) are evaluated on iBims-1 [47].
<table><tr><td rowspan="2">Encoder</td><td rowspan="2">Method</td><td rowspan="2">W/A</td><td colspan="4">Depth Anything v1</td></tr><tr><td colspan="3">Boundary &amp; Ordinal Accuracy</td><td colspan="2">Planarity Errors (cm / °)</td></tr><tr><td rowspan="7">ViT-S</td><td></td><td></td><td>Edge F1@0.5 ↑</td><td>Boundary F1 ↑</td><td>ORD↓</td><td>plan EPE ↓</td><td>epri e ↓</td></tr><tr><td>FP</td><td>32/32</td><td>0.1862</td><td>0.0438</td><td>0.1177</td><td>1.3337</td><td>4.4868</td></tr><tr><td>RepQ-ViT [27] QSCA [33]</td><td>4/4 4/4</td><td>0.1025 0.1292</td><td>0.0018 0.0126</td><td>0.5405 0.2956</td><td>11.4530 10.5740</td><td>35.0064 24.1218</td></tr><tr><td>SPA-Q (Ours)</td><td>4/4</td><td>0.1286</td><td>0.0161</td><td>0.2423</td><td>13.1100</td><td>19.8219</td></tr><tr><td>RepQ-ViT [27]</td><td>4/6</td><td>0.1251</td><td>0.0174</td><td>0.2795</td><td>5.4846</td><td>14.5692</td></tr><tr><td>QSCA [33]</td><td>4/6</td><td>0.1915</td><td>0.0475</td><td>0.1627</td><td>4.0663</td><td>9.8937</td></tr><tr><td>SPA-Q (Ours)</td><td>4/6</td><td>0.1907</td><td>0.0436</td><td>0.1627</td><td>4.2679</td><td>9.0219</td></tr><tr><td rowspan="7">ViT-B</td><td>FP</td><td>32/32</td><td>0.1991</td><td>0.0477</td><td>0.1005</td><td>1.0963</td><td>4.1992</td></tr><tr><td>RepQ-ViT [27]</td><td>4/4</td><td>0.1011</td><td>0.0011</td><td>0.7412</td><td>14.1688</td><td>44.1632</td></tr><tr><td>QSCA [33]</td><td>4/4</td><td>0.1069</td><td>0.0018</td><td>0.4732</td><td>14.0001</td><td>34.3514</td></tr><tr><td>SPA-Q (Ours)</td><td>4/4</td><td>0.1111</td><td>0.0091</td><td>0.2441</td><td>13.6755</td><td>20.1182</td></tr><tr><td>RepQ-ViT [27]</td><td>4/6</td><td>0.1172</td><td>0.0180</td><td>0.2418</td><td>4.1241</td><td>15.4422</td></tr><tr><td>QSCA [33]</td><td>4/6</td><td>0.1658</td><td>0.0328</td><td>0.1672</td><td>5.6829</td><td>11.1425</td></tr><tr><td>SPA-Q (Ours)</td><td>4/6</td><td>0.1938</td><td>0.0404</td><td>0.1377</td><td>3.7100</td><td>7.7550</td></tr></table>

We further report structural metrics to evaluate how well each method preserves the geometric structure of depth maps, including boundary and ordinal accuracy metrics (Edge F1, Boundary F1, and ORD) and planarity errors, as shown in Table 7. SPA-Q achieves the best performance on most metrics under both W4A4 and W4A6 quantization, indicating that it effectively preserves fine depth structures such as object boundaries, relative depth ordering, and planar surfaces.

## D Block-wise Feature Similarity Analysis

Figure 6 and Figure 7 provide an extended block-wise feature similarity analysis under 4-bit quantization across Depth Anything v1 [14] and v2 [15] on NYUv2 [36] and KITTI [37]. For each setting, we compare the MSE and cosine similarity between full-precision and quantized encoder features across transformer blocks. Across all settings, SPA-Q consistently yields lower feature MSE and higher cosine similarity than the compared calibration objectives, especially in deeper blocks.

![](images/7f9b42f82e91503d30ae46cdeec6b7c47838cf53cb0cd6e810c6488d35d668d5.jpg)

![](images/4bfe29e7ef19c9aa2efdfa95abb55afee4dc47e6ef8ad61e9bda3a9e76691232.jpg)  
(a)

![](images/47ec27557c41f918c1a06403f56246d28163cfd2bf683f45d1e42493cd46459a.jpg)

![](images/fc2b92d065a9476329eee20b74b8f719e153dcaa19aac8ba54bf6af6e5d03186.jpg)  
(b)  
Figure 6: Block-wise feature similarity analysis under 4-bit quantization on Depth Anything v1. (a) and (b) show results on NYUv2 [36] and KITTI [37]. We compare the MSE and cosine similarity between full-precision and quantized encoder features across transformer blocks.

![](images/a2fb5116fddb1b311b2f704e8c43b30d2ffe0b2708f2f1af5c1c9cb2f0eb665d.jpg)

![](images/329e9378a85533fca3cd7661e4375d16e22cf48f6cb7114202a817c4e432ff41.jpg)  
(a)

![](images/4950f3f8bb20d015a6d7c7c4562481d1c5e0b52413869cb6f729d2766ed9a518.jpg)

![](images/00c459ebb884727404a59aa0c93684faf146cd1fde832370feba17d6b3a2f21b.jpg)  
(b)  
Figure 7: Block-wise feature similarity analysis under 4-bit quantization on Depth Anything v2. (a) and (b) show results on NYUv2 [36] and KITTI [37]. We compare the MSE and cosine similarity between full-precision and quantized encoder features across transformer blocks.

## E Qualitative Analysis of Feature Maps

Figures 8, 9 visualize feature maps on NYUv2 [36], while Figures 10 and 11 show the corresponding results on KITTI [37]. We compare the full-precision model, naive W4A4 quantization, and SPA-Q. Naive W4A4 quantization substantially distorts the spatial activation patterns of intermediate features, which the decoder relies on for dense depth prediction. In contrast, SPA-Q yields feature maps with higher fidelity to their full-precision counterparts, preserving spatial layouts, object regions, and boundary-related responses. These results show that SPA-Q mitigates feature distortion caused by low-bit quantization across different Depth Anything models, backbones, and datasets.

![](images/3b69da92e6c54ff8201ef5c43a7eb8613b7d6858c441df7d8ea82a1b50e0df70.jpg)  
Figure 8: Feature map visualization for Depth Anything v2 with a ViT-S backbone on NYUv2 [36] under 4-bit quantization. We compare the full-precision model, W4A4 quantization, and SPA-Q.

![](images/c0fbab7a885983e367aaa662db3920efa4aca2970467b71580641a03c67b7409.jpg)  
Figure 9: Feature map visualization for Depth Anything v1 with a ViT-B backbone on NYUv2 [36] under 4-bit quantization. We compare the full-precision model, W4A4 quantization, and SPA-Q.

![](images/feb1c3aa39e961f268730f7521de6d2be5b56705a33076a4acb9bba865689c01.jpg)  
Figure 10: Feature map visualization for Depth Anything v2 with a ViT-S backbone on KITTI [37] under 4-bit quantization. We compare the full-precision model, W4A4 quantization, and SPA-Q.

![](images/9ec320983cf80cd1e428b95da5e2746bd0e0366d89c08df5beba06ff0f3c8e18.jpg)  
Figure 11: Feature map visualization for Depth Anything v1 with a ViT-B backbone on KITTI [37] under 4-bit quantization. We compare the full-precision model, W4A4 quantization, and SPA-Q.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

## Answer: [Yes]

Justification: The main claims in the abstract and introduction match the paper’s contributions and experimental results. The paper claims that SPA-Q addresses attention-preservation and intermediate feature distribution shift in low-bit PTQ for MDE, and these claims are supported by the method description and experiments in Secs. 1, 3, and 4.

Guidelines:

• The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.

• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.

• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification:

Justification: The paper explicitly discusses limitations in the Limitations & Future Works section. In particular, it notes that a noticeable gap to full precision still remains under severe low-bit quantization, that quantized depth maps can appear less smooth due to the restricted set of representable depth values, and that the current study is limited to monocular depth estimation rather than stereo or video depth estimation.

Guidelines:

• The answer [N/A] means that the paper has no limitation while the answer [No] means that the paper has limitations, but those are not discussed in the paper.

• The authors are encouraged to create a separate “Limitations” section in their paper.

• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [N/A]

Justification: The paper does not include formal theoretical results such as theorems, lemmas, propositions, or proofs. It presents objective formulations and empirical analyses, but no claims requiring formal proof.

Guidelines:

• The answer [N/A] means that the paper does not include theoretical results.

• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

• All assumptions should be clearly stated or referenced in the statement of any theorems.

• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

• Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: The main experiments are conducted on publicly available models and datasets, and the paper specifies the model variants, datasets, evaluation metrics, calibration protocol, optimization objectives, and training details needed for reproduction. Since the evaluation is performed on open benchmarks with standard settings, the reported results can be reproduced by other researchers.

## Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• If the paper includes experiments, a [No] answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [No]

Justification: The paper evaluates the proposed method on publicly available datasets and provides the experimental details needed for reproduction, but it does not release implementation code, reproduction scripts, or an anonymized repository link in the current submission.

Guidelines:

• The answer [N/A] means that paper does not include experiments requiring code.

• Please see the NeurIPS code and data submission guidelines (https://neurips.cc/ public/guides/CodeSubmissionPolicy) for more details.

• While we encourage the release of code and data, we understand that this might not be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //neurips.cc/public/guides/CodeSubmissionPolicy) for more details.

• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [Yes]

Justification: The paper specifies the main experimental details needed to understand the re ported results, including the evaluated model variants and backbones, datasets and evaluation metrics, calibration set construction, quantization setting, CWDA initialization, optimization strategy, loss weighting, optimizer, learning rate, batch size, and hardware configuration.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

• The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

## Answer: [No]

Justification: The paper does not report formal measures of statistical significance. However, the experiments follow the standard PTQ evaluation protocol used in prior work, and all methods are compared under the same quantization setting, calibration procedure, and evaluation benchmarks.

## Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The authors should answer [Yes] if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)

• The assumptions made should be given (e.g., Normally distributed errors).

• It should be clear whether the error bar is the standard deviation or the standard error of the mean.

• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g., negative error rates).

• If error bars are reported in tables or plots, the authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: The paper reports that all experiments are conducted on a single RTX 6000 Ada Generation GPU, and also provides model size, parameter count, and BOPs for the evaluated models.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: We have reviewed the NeurIPS Code of Ethics, and this work conforms to it. The paper studies model quantization for monocular depth estimation using standard public benchmarks and does not involve human subjects, personal data collection, or high-risk model release.

Guidelines:

• The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.

• If the authors answer [No], they should explain the special circumstances that require a deviation from the Code of Ethics.

• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [No]

Justification: The manuscript does not include a dedicated broader impacts discussion because this work focuses on a PTQ method for existing MDE models, rather than introducing a new dataset, deployed system, or application. Its societal impacts are mainly indirect and tied to the underlying MDE applications, while the paper focuses on model compression and evaluation under standard benchmarks.

Guidelines:

• The answer [N/A] means that there is no societal impact of the work performed.

• If the authors answer [N/A] or [No], they should explain why their work has no societal impact or why the paper does not address societal impact.

• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

Answer: [N/A]

Justification: This paper does not release new datasets, models, or other assets that require special safeguards for responsible use. Therefore, this question is not applicable to the current work.

• The answer [N/A] means that the paper poses no such risks.

• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: The paper uses publicly available datasets, models, and prior methods, and properly credits their creators through citations. We respect the corresponding licenses and terms of use for all existing assets used in this work.

Guidelines:

• The answer [N/A] means that the paper does not use existing assets.

• The authors should cite the original paper that produced the code package or dataset.

• The authors should state which version of the asset is used and, if possible, include a URL.

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.

• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

• If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [N/A]

Justification: The paper does not introduce or release new datasets, code, models, or other assets. Therefore, this question is not applicable to the current work.

Guidelines:

• The answer [N/A] means that the paper does not release new assets.

• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

• The paper should discuss whether and how consent was obtained from people whose asset is used.

• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [N/A]

Justification: The paper does not involve crowdsourcing or research with human subjects.   
Therefore, this question is not applicable to the current work.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A]

Justification: The paper does not involve human subjects research or crowdsourcing. Therefore, IRB approval or an equivalent review is not applicable to this work.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

Answer: [N/A]

Justification: Large language models were used only for writing assistance in preparing the manuscript. They were not used in the core methodology, experiments, or analysis, and therefore did not affect the scientific content or conclusions of the paper.

## Guidelines:

• The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.

• Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.