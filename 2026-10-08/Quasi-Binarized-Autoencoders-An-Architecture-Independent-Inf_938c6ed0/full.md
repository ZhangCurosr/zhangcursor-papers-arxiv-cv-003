# Quasi-Binarized Autoencoders: An Architecture-Independent Information Bottleneck for Medical Image Anomaly Detection

Shouhei Hanaoka<sup>a,∗</sup>, Takahiro Nakao<sup>b</sup>, Atsushi Takamatsu<sup>b,c</sup>, Takeharu Yoshikawa<sup>b</sup> Osamu Abe<sup>a</sup>

<sup>a</sup>Department of Radiology, The University of Tokyo Hospital, 7-3-1 Hongo, Bunkyo-ku, Tokyo, 113-8655, Japan

<sup>b</sup>Department of Computational Diagnostic Radiology and Preventive Medicine, The University of Tokyo Hospital, 7-3-1 Hongo, Bunkyo-ku, Tokyo, 113-8655, Japan <sup>c</sup>Department of Radiology, Kanazawa University Hospital, 13-1 Takara-machi, Kanazawa, Ishikawa, 920-8641, Japan

## Abstract

Unsupervised anomaly detection, which learns only from normal images, is a central task in medical image analysis and remains an open problem. Reconstruction-based methods pass an image through an encoder–decoder network trained on normal data and detect anomalies from the residual between the image and its reconstruction. This works only if the information passed from the encoder to the decoder is limited; otherwise the network learns an identity mapping and reconstructs anomalies too. This limit is usually imposed through architectural choices, tuned per dataset, that cannot be stated in bits. We introduce the quasi-binarizing (QB) layer, which squashes each latent element into [0, 1] and adds Laplace noise of scale 1/ε. Each element is then ε-locally diferentially private, and the mutual information between an image and its reconstruction is bounded by a quantity that depends only on ε and the number of QB elements, whatever the encoder and decoder. Placing a QB layer on every encoder–decoder path, including all skip connections, we build QBAE, a seven-level attention U-Net with 32,768 QB elements. On the seven datasets of the MedIAnomaly benchmark, QBAE with one architecture and one configuration reaches a mean image-level AUROC of 0.828, the highest among methods that do not adapt to each dataset, and the best reported results on BraTS2021 (AUROC 0.911, pixel-level AP 0.838). The noise is kept at test time, so that every reconstruction satisfies the bound. Without input corruption, the bottleneck alone prevents identity collapse (mean AUROC 0.805 vs. 0.590). Code is available at https://github.com/hanaokalog/MedIAnomalyQB.

Keywords: anomaly detection, autoencoder, information bottleneck, diferential privacy, medical imaging

## 1. Introduction

Unsupervised anomaly detection aims to flag images that deviate from a model of normal anatomy, using only normal training data. This setting fits medical imaging well. Normal studies are plentiful, abnormal ones are diverse and costly to annotate, and the full range of pathologies can never be listed in advance. Many paradigms have been proposed, including image reconstruction, self-supervised pretext tasks, and feature embedding. Yet the recent MedIAnomaly benchmark (Cai et al., 2025) showed that reconstruction-based autoencoders (AEs) remain among the most competitive and versatile approaches across seven datasets and five imaging modalities. They are especially strong for pixel-level localization, where feature-embedding methods built on natural-image backbones struggle.

An AE detects anomalies only if it reconstructs normal images well and abnormal images poorly. This works only when the information that passes from the encoder to the decoder is limited. Otherwise the network learns a near-identity mapping and reconstructs lesions as faithfully as normal tissue (Cai et al., 2024). In practice this limit is imposed through the architecture. Designers shrink the latent dimension, remove skip connections, or reduce depth, and then tune these choices per dataset. This practice has two drawbacks.

• Expressiveness and information limit are coupled. Components that make reconstructions sharper, such as U-Net skip connections (Ronneberger et al., 2015), attention, and modern up/down-sampling blocks (Chen et al., 2025), also widen the path through which identity can leak. Anomaly-detection AEs therefore stay deliberately small and shallow, while the rest of deep learning has moved on.

• The limit is not measured. A real-valued latent vector of any dimension can in principle carry unbounded information. The latent dimension is only an indirect proxy, and the amount of information an AE actually transmits can neither be stated in bits nor compared across architectures.

Existing remedies address these issues only in part.

• Variational AEs (Kingma and Welling, 2014) quantify information through a KL term, but they tie the bottleneck to a prior and tend to produce blurry reconstructions.

• Memory-augmented (Gong et al., 2019) and vector-quantized (van den Oord et al., 2017) AEs restrict the latent to a learned codebook. However, the codebook size is again an architectural choice, and skip connections bypass it.

• Denoising AEs (Vincent et al., 2008; Kascenas et al., 2022) take a complementary route. They corrupt the input so that the network must learn to map corrupted images back onto the normal manifold. This yields the strongest image-level results on brain MRI in MedIAnomaly. Yet input corruption shapes what the network learns, not how much information it can transmit.

We propose to separate the information bottleneck from the architecture. We introduce the quasi-binarizing (QB) layer. A QB layer squashes each element of the latent tensor into [0, 1] with a sigmoid and adds Laplace noise of scale 1/ε. To remain readable despite this noise, the outputs are driven toward 0 or 1 during training, so that the layer acts as a soft, noisy binarizer; hence its name (Section 3.2.1). The construction comes from diferential privacy (DP), a framework for releasing the output of a computation while provably limiting what that output reveals about its input (Dwork et al., 2006). In DP terms, each element has sensitivity one, i.e. its value can change by at most one whatever the input (a property of the function, unrelated to the sensitivity of a diagnostic test), and adding Laplace noise of scale 1/ε is exactly the Laplace mechanism. Every element is therefore ε-locally diferentially private (Duchi et al., 2013). This yields two bounds on the information an element carries about the input image.

• Privacy bound. The mutual information is at most min $\{ \varepsilon , \varepsilon ^ { 2 } \}$ nats (Cuf and Yu, 2016).

• Amplitude bound. Because each element is confined to [0, 1], the mutual information is also at most $\frac { 1 } { 2 }$ ln $( 2 \pi \mathrm { e } ( { \frac { 1 } { 4 } } + 2 / \varepsilon ^ { 2 } ) ) - \ln ( 2 \mathrm { e } / \varepsilon )$ nats (Cover and Thomas, 2006). This bound grows only logarithmically in $\varepsilon ,$ for example to about 2 bits per element at $\varepsilon = 1 0$

The noise is drawn independently for each element. The information the decoder receives is therefore bounded by the sum of these per-element bounds, whatever the depth, width, or connectivity of the encoder and decoder. We place a QB layer on every path from the encoder to the decoder, including all skip connections. The resulting budget depends only on ε and the number of QB elements. We keep the noise at test time, so that every reconstruction used for detection is obtained under this budget, and the bottleneck can be stated, and audited, in bits.

Once the information limit no longer depends on the architecture, the encoder and decoder can be made as expressive as reconstruction quality requires. Our Quasi-Binarized Autoencoder (QBAE) is a seven-level U-Net with full-resolution skip connections. It uses DC-AE residual autoencoding blocks (Chen et al., 2025), cross-attention gates that mix each skip with the upsampled path, and attention blocks in the decoder. Every skip connection passes through a QB layer, with the number of QB elements halved at each level, and a fully connected layer at the $2 \times 2$ bottom feeds a final QB layer. In total, 32,768 QB elements feed the decoder. Classical design rules would call this latent far too large for anomaly detection. Under the QB bottleneck it is not: at $\varepsilon = 1 0$ , these elements carry at most $6 . 5 \times 1 0 ^ { 4 }$ bits, half the 131,072 bits of a raw 8-bit $1 2 8 \times 1 2 8$ input image.

Because the budget is set by ε alone, one fixed network can be adapted to datasets with very diferent anomalies by changing a single scalar with a direct meaning, the number of bits allowed through, instead of redesigning the latent space or the skip connections.

Our contributions are:

1. The QB layer. It is a bounded-input Laplace noise layer that gives every encoder– decoder path an information budget controlled by $\varepsilon ,$ with an $\varepsilon { \mathrm { - L D P } }$ guarantee and closed-form bounds on the mutual information per element.

2. QBAE. It is an anomaly-detection U-Net in which every encoder–decoder path, including all skip connections, passes through a QB layer. This allows modern high-capacity components without identity leakage.

3. A systematic evaluation on all seven MedIAnomaly datasets at $1 2 8 \times 1 2 8$ resolution, with one network and a sweep of ε over almost two orders of magnitude. With a single configuration for all datasets, QBAE reaches the highest mean AU-ROC (0.828) among methods that do not adapt to each dataset, and an image-level AUROC of 0.911 and a pixel-level AP of 0.838 on BraTS2021, exceeding the best reported results (0.859 and 0.755). We show that the best $\varepsilon$ difers systematically between datasets, that performance is insensitive to ε near its optimum, and that without input corruption the QB bottleneck alone prevents the identity collapse of the network.

4. Open-source code. The implementation, training scripts and configurations are available at https://github.com/hanaokalog/MedIAnomalyQB.

## 2. Related Work

Reconstruction-based anomaly detection in medical imaging. AEs and their variants are the most widely used family for unsupervised anomaly detection in medical images (Baur et al., 2021). Examples include adversarial models such as f-AnoGAN (Schlegl et al., 2019), context-encoding VAEs (Zimmerer et al., 2018), AEs with uncertainty estimation (Mao et al., 2020), and perceptual AEs (Shvetsova et al., 2021). Denoising AEs (Kascenas et al., 2022) corrupt brain MR images with coarse, spatially correlated noise, and this simple scheme rivals far more complex models. More recently, difusion models have been used to “heal” anomalies (Wolleb et al., 2022; Wyatt et al., 2022; Pinaya et al., 2022), at a much higher inference cost.

Two benchmarks now enable a fair comparison across modalities. BMAD (Bao et al., 2024) covers six datasets, and MedIAnomaly (Cai et al., 2025) evaluates more than 30 methods on seven datasets under a unified protocol. MedIAnomaly found that well-tuned AEs remain highly competitive. Its companion analysis (Cai et al., 2024) argued, from an information-theoretic viewpoint, that the latent size, and hence the information passed to the decoder, is the key factor in AE-based detection.

Training-free methods that score densities of pretrained features have also been benchmarked on MedIAnomaly. MSDE (Kar et al., 2026) refines embeddings from a pretrained network (ResNet-18 or a self-supervised AnatPaste encoder, chosen per dataset) by mean shift before Gaussian density scoring. It reaches high image-level AUROC on chest X-ray and brain-tumor MRI, but it falls well behind reconstruction-based methods on BraTS2021 (0.736 vs. 0.859 for DAE).

In industrial anomaly detection, the failure mode in which an over-capable reconstruction network copies anomalies to its output is known as the “identical shortcut” (You et al., 2022). The reconstruction-based works above all control the information that reaches the decoder indirectly, through architectural choices such as latent size, depth, or the absence of skip connections. QBAE instead controls it with an explicit, architecture-independent budget.

Limiting and measuring information in autoencoders. The information bottleneck (Tishby et al., 1999) and its variational form (Alemi et al., 2017) formalize the trade-of between compressing the input and keeping what is relevant. In VAEs (Kingma and Welling, 2014) and β-VAEs (Higgins et al., 2017), the KL term is an upper bound on the rate. However, this rate is measured relative to a learned prior, and skip connections fall outside it (Alemi et al., 2018). Discrete bottlenecks such as vector quantization (van den Oord et al., 2017) and memory modules (Gong et al., 2019) limit the latent to a codebook, so their capacity follows from the codebook size, which is again an architectural choice. Sparse AEs penalize mean activations with a KL term (Ng, 2011), but they bound no information-theoretic quantity.

The idea closest to the QB layer comes from learned hashing and compression. Semantic hashing (Salakhutdinov and Hinton, 2009; Kaiser and Bengio, 2018) adds noise before a sigmoid to push the codes toward binary values. Learned image compression (Ballé et al., 2017) adds uniform noise as a diferentiable proxy for quantization and measures the rate in bits. Neural joint source–channel coding (Choi et al., 2019) trains codes to survive a noisy binary channel. The QB layer difers from all of these in three ways:

• It adds noise after a bounded squashing. This is what yields a formal diferentialprivacy guarantee and closed-form bounds on the information per element. Because the noise has the same scale everywhere, values are easiest to tell apart when they lie at the two ends of [0, 1]; training therefore drives the outputs toward 0 or 1, and the output of the QB layer becomes nearly binary.

• It is placed on every encoder–decoder path, including the skips.

• Its purpose is to withhold information, to prevent identity mapping, rather than to transmit it eficiently.

Diferential privacy and information. Diferential privacy (DP) is a formal framework for releasing the output of a computation, called a mechanism, while limiting how much that output can reveal about any single input record: a mechanism is $\varepsilon { \mathrm { - D P } }$ if changing one record changes the probability of any output by at most a factor $e ^ { \varepsilon }$ (Dwork et al., 2006). Smaller ε thus means less can be inferred about the input from the output. The sensitivity of a function is the largest change in its value that a change of one record can cause. The Laplace mechanism (Dwork et al., 2006) makes a function with bounded sensitivity ε-diferentially private by adding Laplace noise scaled to that sensitivity. In the local model, each record is privatized before it leaves the data owner (Duchi et al., 2013). ε-DP implies a bound on mutual information (McGregor et al., 2010; Cuf and Yu, 2016). In deep learning, DP is mostly applied to training data through gradient perturbation (Abadi et al., 2016). Noise injected into intermediate activations has been studied to protect inference-time privacy (Mireshghallah et al., 2020). To our knowledge, the Laplace mechanism has not been used as an information bottleneck for anomaly detection. In QBAE, the privacy parameter ε directly sets how much of each image the decoder may “see”.

Architectural components and pretrained features. QBAE builds on the U-Net (Ronneberger et al., 2015) with attention gating of skip connections (Oktay et al., 2018). Its gates follow CBAM (Woo et al., 2018), extended to cross-attention between the skip and the upsampled path. The encoder and decoder use the residual autoencoding blocks of DC-AE (Chen et al., 2025), which pair space-to-channel shortcuts without parameters with learned residuals. The decoder and the bottom of the network use channel attention in the style of Restormer (Zamir et al., 2022) and spatial self-attention. None of these components is specific to anomaly detection. The point of QBAE is that, because every path to the decoder passes through a QB layer, they can be adopted without re-introducing an identity shortcut.

Like perceptual AEs (Shvetsova et al., 2021) and the AE-Perceptual baseline of MedIAnomaly, QBAE trains with, and scores by, a relative perceptual loss computed on ImageNet-pretrained VGG19 features (Simonyan and Zisserman, 2015; Johnson et al., 2016). It therefore also relies on external pretrained weights. In contrast to featuredensity methods such as MSDE, however, these features enter only the loss and the anomaly score. The representation that crosses the bottleneck is learned from normal medical images alone.

## 3. Method

## 3.1. Problem setting and overview

We consider the standard unsupervised setting. A training set $\mathcal { D } = \{ x ^ { ( m ) } \} _ { m = 1 } ^ { M }$ contains only normal images $\boldsymbol { x } \in \mathbb { R } ^ { C \times H \times W }$ . At test time, the model must produce a pixel-wise anomaly map $\bar { A ( x ) } \in \mathbb { R } ^ { H \times W }$ and an image-level score $s ( x ) \in \mathbb { R }$ for images that may contain unseen abnormalities. Like other reconstruction-based methods, QBAE trains a network F to reconstruct normal images and derives A and s from the discrepancy between x and ${ \hat { x } } = F ( x )$

QBAE (Fig. 1) is a seven-level U-Net-type autoencoder in which every tensor passed from the encoder to the decoder first goes through a quasi-binarizing (QB) layer. At each encoder level $l \in \{ 1 , \ldots , 6 \}$ (resolutions $1 2 8 ^ { 2 }$ down to $4 ^ { 2 } )$ , the skip connection is projected to $k _ { l } = 2 ^ { l - 1 }$ channels per pixel and passed through a QB layer, so that the number of QB elements halves from level to level: 16,384, 8,192, . . . , 512. At the deepest level, the $2 \times 2$ feature map is flattened and mapped by a fully connected layer to a further 512 QB elements. This gives

$$
N = \sum _ { l = 1 } ^ { 6 } 2 ^ { l - 1 } \frac { H W } { 4 ^ { l - 1 } } + 5 1 2 = 1 6 , 3 8 4 + 8 , 1 9 2 + 4 , 0 9 6 + 2 , 0 4 8 + 1 , 0 2 4 + 5 1 2 + 5 1 2 = 3 2 , 7 6 8 = 2 ^ { 1 5 }
$$

QB elements in total, for $H = W = 1 2 8$ . The same network, with the same $N .$ , is used for all datasets in this work; only ε is changed. Let $h \in \mathbb { R } ^ { N }$ denote the concatenated pre-activations produced by the encoder E. Let $\boldsymbol { \tilde { z } } \in \mathbb { R } ^ { N }$ denote the corresponding QB outputs. The decoder D produces $\hat { x } = D ( \tilde { z } )$ . The encoder and decoder are otherwise unconstrained. Section 3.2 shows that the QB layers alone bound how much information about x can reach xˆ. Section 3.3 describes the architecture and shows that no path from x to xˆ avoids a QB layer. During training, the encoder receives a corrupted image $\mathcal C ( x )$ and the network is trained to reconstruct the clean x (Section 3.4). Section 3.5 defines the anomaly scores.

## 3.2. The quasi-binarizing layer

## 3.2.1. Definition

A QB layer with privacy parameter $\varepsilon > 0$ acts element-wise on a pre-activation vector $h \in \mathbb { R } ^ { N }$ . It first squashes each element into the unit interval,

$$
z _ { i } = \sigma ( h _ { i } ) \in [ 0 , 1 ] ,
$$

and, during training, adds independent Laplace noise,

$$
\tilde { z } _ { i } = z _ { i } + n _ { i } , \qquad n _ { i } \stackrel { \mathrm { i . i . d . } } { \sim } \mathrm { L a p l a c e } ( 0 , b ) , \quad b = 1 / \varepsilon .\tag{1}
$$

The noise is resampled for every sample, every element, and every forward pass, independently of the input. Since the noise is additive, gradients pass through the layer unchanged, and no straight-through estimator is required.

The noise is kept at test time, so that the network used for detection is the same noisy mechanism as in training (Section 3.5).

Two limits help with interpretation. As $\varepsilon  \infty$ the noise vanishes, and the QB layer reduces to a sigmoid. In our experiments this case is approximated by $\varepsilon = 1 0 ^ { 8 }$ and serves as the no-bottleneck control. $\mathrm { A s } \ \varepsilon  0$ the noise swamps the signal, and no information is transmitted.

The name reflects the efect of the noise. To survive noise of scale b, the encoder is driven to place each $z _ { i }$ near the ends of $[ 0 , 1 ]$ , so that its value can still be read after noise is added. The learned code thus becomes quasi-binary (Appendix D).

## 3.2.2. Information budget

In short: the amount of information that can pass through QBAE has a rigorous upper bound, which depends only on ε and on the number of QB elements and not on the architecture of the encoder or the decoder.

(a) Overall architecture (7 levels, 128 × 128 input)  
![](images/1317a35798ed18c1bee17db6e6373ae8d1ed592a13db716229834e246f778a2f.jpg)  
Fig. 1. Overview of QBAE. (a) A seven-level U-Net in which each of the six skip connections and the fully connected bottom passes through a QB layer; the decoder receives only QB outputs (32,768 elements). (b) QB layer. (c), (d) DC-AE residual down- and upsampling. Details are given in Appendix B.

The intuition is simple. Whatever the encoder computes, each QB element can only send a value between 0 and 1, and during training this value reaches the decoder blurred by noise of scale $1 / \varepsilon .$ A receiver can therefore tell apart only a limited number of levels, roughly of the order of ε, no matter how cleverly the encoder chooses them. Each element thus behaves like a noisy dial with a fixed resolution. The decoder sees the image only through N such dials, so its view of the image is limited by the number of dials and their resolution, not by how deep or wide the network around them is. This is the same mechanism that diferential privacy uses to limit what a released statistic reveals about an individual (the Laplace mechanism; Dwork et al., 2006). Here, the “individual” is the input image.

This intuition can be made exact. We state the result informally here; the formal statements and proofs are given in Appendix A.

Theorem 1 (information budget, informal). Assume that every path from the input to the output passes through a QB layer (Section 3.3 shows that QBAE satisfies this). Then, during training and at test time, the mutual information between the input image X and the reconstruction $\hat { X }$ satisfies

$$
I ( X ; { \hat { X } } ) ~ \leq ~ N \cdot c ( \varepsilon ) ,
$$

where $c ( \varepsilon )$ is an explicit per-element bound that depends on ε only (Appendix A, Eq.   
(A.4)).

The bound holds for any encoder and decoder, for normal and abnormal images alike, and with or without input corruption. For large $\varepsilon , c ( \varepsilon )$ grows only logarithmically, roughly as $\log _ { 2 } \varepsilon$ bits. Table 1 gives the budgets for the settings used in this work (ε on a $\sqrt { 1 0 }$ grid). To put them in perspective, a raw 8-bit $1 2 8 \times 1 2 8$ single-channel image contains 131,072 bits.

Table 1. Upper bounds on the information transmitted through QBAE $( N = 3 2 , 7 6 8$ QB elements).
<table><tr><td>Setting</td><td>Per element</td><td>Total budget</td><td>Relative to a raw 8-bit  $1 2 8 ^ { 2 }$  image</td></tr><tr><td> $\varepsilon = 0 . 3$ </td><td>0.11 bits</td><td> $3 . 7 \times 1 0 ^ { 3 }$  bits</td><td>0.03</td></tr><tr><td>ε = 1</td><td>0.19 bits</td><td> $6 . 2 \times 1 0 ^ { 3 }$  bits</td><td>0.05</td></tr><tr><td>ε = 3</td><td>0.65 bits</td><td> $2 . 1 \times 1 0 ^ { 4 }$  bits</td><td>0.16</td></tr><tr><td>ε = 10</td><td>1.98 bits</td><td> $6 . 5 \times 1 0 ^ { 4 }$  bits</td><td>0.50</td></tr><tr><td>ε = 30</td><td>3.52 bits</td><td> $1 . 1 5 \times 1 0 ^ { 5 }$  bits</td><td>0.88</td></tr><tr><td> $\varepsilon = 1 0 0$ </td><td>5.25 bits</td><td> $1 . 7 2 \times 1 0 ^ { 5 }$  bits</td><td>1.31</td></tr></table>

Because the budget is set by a single scalar, it can be adapted to a dataset without changing the network: across the range of Table 1 it spans almost two orders of magnitude.

Theorem 1 applies to the noisy QB layers. Removing the noise at test time would make the layers noiseless and their capacity unbounded in principle. We therefore keep the noise at test time and average the anomaly scores over several noise draws (Section 3.5), so that every reported result is obtained under the stated budget.

## 3.2.3. Practical remarks

Saturation. A sigmoid that saturates early would make the layer hard to train. At the bottom QB layer, we therefore apply a learnable parametrized arctangent, $\mathrm { P A } ( h ) =$ α arctan $( \beta h + \gamma ) + \delta$ , before σ. Here $\alpha , \beta , \gamma , \delta$ are learnable scalars, and $\beta$ is initialized to 0.01, so that $\sigma ( \mathrm { P A } ( h ) )$ starts close to ${ \frac { 1 } { 2 } } .$ . The QB layers of the skip connections take the output of a $1 \times 1$ convolution directly.

Sparsity. Optionally, a sparsity penalty can push the mean activation of each element toward a target $\rho$ (Section 3.4). The bounds above do not rely on this penalty.

Normalization. All normalization layers in QBAE operate within each sample (GroupNorm). Batch statistics therefore cannot carry information from one sample to another around the QB layers.

## 3.3. Network architecture

The architecture of QBAE is not where its bottleneck comes from, and we therefore keep its description short. The network is chosen to reconstruct normal images well; how much information it may pass is fixed separately by the QB layers (Section 3.2). All experiments use the same network, shown in Fig. 1. Layer-level details are given in Appendix B.

Encoder and decoder. QBAE is a seven-level U-Net operating on $1 2 8 \times 1 2 8$ inputs, with feature widths 16, 32, . . . , 1024 at resolutions $1 2 8 ^ { 2 }$ down to $2 ^ { 2 }$ . Each level uses a residual convolution block (two 3×3 convolutions with Swish activations (Ramachandran et al., 2017) and GroupNorm (Wu and He, 2018)). Down- and upsampling follow the residual autoencoding of DC-AE (Chen et al., 2025), in which a parameter-free spaceto-channel rearrangement is added to a learned convolution. In the decoder, each skip connection is merged with the upsampled path by a cross-attention gate (Woo et al., 2018), and the decoder levels at $6 4 ^ { 2 }$ and below contain an attention block (Zamir et al., 2022).

QB layers. Every tensor that the decoder receives from the encoder passes through a QB layer. At each of the six skip connections, a $1 \times 1$ convolution projects the encoder features to $2 ^ { l - 1 }$ channels per pixel at level l, a QB layer is applied, and a second $1 \times 1$ convolution maps the result back to the decoder width. Because the number of pixels falls by a factor of four per level while the number of channels per pixel doubles, the number of QB elements halves from one level to the next (Table 2). At the bottom, the $2 \times 2$ feature map is flattened and passed through a fully connected layer to a final QB layer of 512 elements, the same number as the deepest skip connection. The fully connected layer gives the bottom path position-specific weights, which the convolutional levels lack. In total, the decoder receives $N = 3 2 , 7 6 8$ QB elements.

Table 2. QB elements per level (128 × 128 input).
<table><tr><td>Level</td><td>Resolution</td><td>Channels per pixel</td><td>QB elements</td><td>Share of N</td></tr><tr><td>1</td><td> $1 2 8 ^ { 2 }$ </td><td>1</td><td>16,384</td><td>50.0%</td></tr><tr><td>2</td><td> $6 4 ^ { 2 }$ </td><td>2</td><td>8,192</td><td>25.0%</td></tr><tr><td>3</td><td> $3 2 ^ { 2 }$ </td><td>4</td><td>4,096</td><td>12.5%</td></tr><tr><td>4</td><td> $1 6 ^ { 2 }$ </td><td>8</td><td>2,048</td><td>6.3%</td></tr><tr><td>5</td><td> $8 ^ { 2 }$ </td><td>16</td><td>1,024</td><td>3.1%</td></tr><tr><td>6</td><td> $4 ^ { 2 }$ </td><td>32</td><td>512</td><td>1.6%</td></tr><tr><td>bottom total</td><td> $2 ^ { 2 }$  (fully connected)</td><td></td><td>512 32,768</td><td>1.6% 100%</td></tr></table>

No path avoids a QB layer. Theorem 1 requires that the reconstruction depends on the input only through the QB outputs. QBAE satisfies this by construction. The residual shortcuts of the convolution blocks and of the DC-AE resampling layers stay inside the encoder or inside the decoder. The cross-attention gates combine only QB outputs with decoder features. All normalization layers act within a single sample, so no information can pass from one image to another through batch statistics. Hence every path from x to $\hat { x }$ in the computation graph contains a QB layer, and the bound of Section 3.2.2 applies to the network as a whole, whatever the capacity of the encoder and the decoder.

## 3.4. Training

Input corruption. During training, the encoder receives a corrupted image ${ \mathcal { C } } ( x ) =$ x+η and the network is trained to reconstruct the clean $x ,$ as in the denoising autoencoder (DAE) of Kascenas et al. (2022). Our corruption is a modified version of their coarse noise (Kascenas et al., 2022, 2023), which is low-resolution Gaussian noise upsampled to the image size. Instead of covering the whole image, our η consists of smooth, localized blobs of random size. It is the product of two Gaussian-smoothed white-noise fields with random smoothing radii: one field sets the texture, and the other is thresholded and raised to a random power so that only a few patches remain non-zero. The product is scaled by the standard deviation of the image. The exact generator is given in Appendix C. Input corruption and the QB layers play diferent roles: the corruption shapes what the network learns, namely to map locally altered images back to normal ones, whereas the QB layers limit how much information about the input can reach the decoder. Section 5 shows that the two are complementary.

Loss. The network is trained with

$$
\mathcal { L } = \frac { 1 } { H W } \left\| x - \hat { x } \right\| _ { 2 } ^ { 2 } + \lambda _ { p } \mathcal { L } _ { \mathrm { p e r c } } ( x , \hat { x } ) + \lambda _ { \mathrm { K L } } \sum _ { i = 1 } ^ { N } \mathrm { K L } ( \rho \| \bar { z } _ { i } ) ,\tag{2}
$$

where xˆ is the reconstruction of $\mathcal C ( x )$ . The perceptual term $\mathcal { L } _ { \mathrm { p e r c } }$ is the relative-perceptual-$L _ { 1 }$ loss used by the perceptual autoencoder (AE-PL) of Shvetsova et al. (2021): an $L _ { 1 }$ distance between VGG-19 (Simonyan and Zisserman, 2015) relu4\_2 features with ImageNet (Deng et al., 2009) weights of x and xˆ, with the features standardized per channel and the distance divided by the mean feature magnitude of x. The only change is that, during training, x and xˆ are jointly shifted by up to 8 pixels before the feature extraction. The last term is an optional sparsity penalty (Ng, 2011): $\bar { z } _ { i }$ is the mean of $z _ { i } = \sigma ( h _ { i } )$ over the mini-batch, and $\mathrm { K L } ( \rho \| \bar { z } _ { i } )$ is the Kullback–Leibler divergence between Bernoulli distributions with means $\rho$ and $\bar { z } _ { i }$ . It encourages each element to be active for only a fraction $\rho$ of the images, so that the code becomes sparse as well as quasi-binary. We report results with and without this term. None of the bounds in Section 3.2 depends on it.

## 3.5. Anomaly scores

At test time, the corruption is switched of and the QB layers keep their noise (Section 3.2.1). The pixel-wise anomaly map combines the squared reconstruction error with the perceptual error map,

$$
A ( x ) = ( x - \hat { x } ) ^ { 2 } + \lambda _ { p } \mathcal { U } \big ( D _ { \mathrm { p e r c } } ( x , \hat { x } ) \big ) ,\tag{3}
$$

where $D _ { \mathrm { { p e r c } } }$ is the channel-averaged relative $L _ { 1 }$ feature diference before spatial averaging and $\mathcal { U }$ denotes bilinear upsampling to the image size. The second term is exactly the anomaly map of AE-PL (Shvetsova et al., 2021; Cai et al., 2025). The map $A ( x )$ is used for pixel-level evaluation. The image-level score $s ( x )$ is the spatial mean of this perceptual term alone, $\begin{array} { r } { s ( x ) = \lambda _ { p } \overline { { \mathcal { U } ( D _ { \mathrm { p e r c } } ( x , \hat { x } ) ) } } } \end{array}$ , i.e. the image-level score of AE-PL.

Because the noise is kept, the score depends on the noise draw. We therefore average $A ( x )$ and $s ( x )$ over $K = 8$ independent draws; K was fixed in advance and is the same for all datasets. Every test-time reconstruction is produced by exactly the noisy network analyzed in Theorem 1 and is thus obtained under the stated budget. Because the averaged score combines K independent passes, Proposition 3 bounds the information it carries about x by K times this budget.

## 4. Experimental setup

## 4.1. Datasets

We use all seven datasets of the MedIAnomaly benchmark (Cai et al., 2025) with its oficial training and test splits (Table 3). The training sets contain only normal images. The datasets cover five modalities and difer widely in the size and visibility of the abnormalities, from large tumors in brain MRI to subtle changes in retinal fundus photographs and histopathology patches. BraTS2021 is the only dataset with pixel-level annotations. All images are resized to $1 2 8 \times 1 2 8$ . We follow the benchmark protocol of downsampled inputs so that our results are comparable with the reported baselines, but use twice its default size of $6 4 \times 6 4$

Table 3. Datasets (MedIAnomaly splits).
<table><tr><td>Dataset</td><td>Source</td><td>Modality</td><td>Train (normal)</td><td>Test normal</td><td>Test abnormal</td><td>Pixel labels</td></tr><tr><td>RSNA</td><td>Shih et al. (2019)ª</td><td>Chest X-ray</td><td>3,851</td><td>1,000</td><td>1,000</td><td></td></tr><tr><td>VinDr-CXR</td><td>Nguyen et al. (2022)</td><td>Chest X-ray</td><td>4,000</td><td>1,000</td><td>1,000</td><td></td></tr><tr><td>Brain Tumor</td><td>Hamada (2025); Saleh et al. (2020); Cheng et</td><td>Brain MRI</td><td>1,000</td><td>600</td><td>600</td><td></td></tr><tr><td>LAG</td><td>al. (2015)b Li et al. (2019)</td><td>Fundus photograph</td><td>1,500</td><td>811</td><td>811</td><td></td></tr><tr><td>ISIC2018</td><td>Codella et al. (2019)</td><td>Dermoscopy</td><td>6,705</td><td>909</td><td>603</td><td></td></tr><tr><td>Camelyon16</td><td>Ehteshami Bejnordi et al. (2017); Bao et</td><td>Histopathology</td><td>5,088</td><td>1,120</td><td>1,113</td><td></td></tr><tr><td>BraTS2021</td><td>al. (2024)c Baid et al. (2021)</td><td>Brain MRI (FLAIR) 4,211</td><td></td><td>828</td><td>1,948</td><td>yes</td></tr></table>

<sup>a</sup> RSNA Pneumonia Detection Challenge (2018), https://www.rsna.org/education /ai-resources-and-training/ai-image-challenge/RSNA-Pneumonia-Detection-C hallenge-2018. <sup>b</sup> Normal images from Br35H (Hamada, 2025) and Saleh et al. (2020); glioma and meningioma images from Saleh et al. (2020) and Cheng et al. (2015). <sup>c</sup> Preprocessed patches provided by Bao et al. (2024).

## 4.2. Implementation details

QBAE is implemented in PyTorch on top of the MedIAnomaly codebase (Cai et al., 2025); our code is available at https://github.com/hanaokalog/MedIAnomalyQB. All experiments use the network of Section 3.3 without modification (N = 32,768 QB elements). Networks are trained with AdamW (Loshchilov and Hutter, 2019; learning rate $1 0 ^ { - 3 }$ , weight decay $1 0 ^ { - 3 }$ , batch size 128). The learning rate is increased linearly over the first 5 epochs and then decreased to $1 0 ^ { - 5 }$ with a cosine schedule (Loshchilov and Hutter, 2017); the global gradient norm is clipped at 1 (Pascanu et al., 2013). Training lasts 250 epochs (600 for Brain Tumor, following the benchmark). The perceptual weight is $\lambda _ { p } =$ 0.1. When the sparsity penalty is used, $\rho = 0 . 0 5$ and $\lambda _ { \mathrm { K L } } = 1 0 ^ { - 6 }$ . The input corruption has unit strength relative to the image standard deviation. The VGG-19 network of the perceptual loss is evaluated in bfloat16 during training and in single precision at test time. No hyper-parameter other than ε, the input corruption $\left( \mathrm { o n } / \mathrm { o f f } \right)$ and the sparsity penalty $( \mathrm { o n / o f f } )$ is varied between experiments. Each run takes about 10–45 min on a single NVIDIA GH200 GPU, depending on the dataset.

<sub>We</sub> <sub>vary</sub> <sub>ε</sub> <sub>on</sub> <sub>a</sub> √<sub>10</sub> <sub>grid,</sub> $\varepsilon \in \{ 0 . 3 , 1 , 3 , 1 0 , 3 0 , 1 0 0 \}$ , which spans information budgets from $3 . 7 \times 1 0 ^ { 3 }$ to $1 . 7 \times 1 0 ^ { 5 }$ bits (Table 1). As a control without a bottleneck, we use $\varepsilon = 1 0 ^ { 8 }$ : the QB layers then reduce to a sigmoid, and the Laplace noise is negligible.

## 4.3. Evaluation protocol

We report the image-level AUROC as the main metric and the image-level AP as a secondary one. $\mathrm { O n }$ BraTS2021 we also report the pixel-level AP and the best achievable Dice score over all thresholds, computed from the anomaly map $A ( x )$ . All scores are averaged over $K = 8$ noise draws (Section 3.5). Every configuration is trained five times with diferent random seeds on the same split, and we report the mean and standard deviation.

MedIAnomaly provides no validation set, and choosing hyper-parameters on the test set would bias the comparison. We therefore report one common configuration for all datasets in the main comparison: $\varepsilon ~ = ~ 1 0$ , input corruption and sparsity penalty on, which gave the best mean AUROC over the seven datasets. This single choice for all datasets limits, but does not remove, the optimism of selecting on test data; Section 5 shows that the mean AUROC changes by at most 0.012 for ε between 3 and 30. Results at the best ε for each dataset are given separately and are marked as an optimistic upper bound. The full ε sweep is shown in Section 5.

For other methods we quote the results reported by Cai et al. (2025) and by Kar et al. (2026, MSDE). These were obtained at the benchmark’s input size, which may difer from ours, and in several cases with dataset-specific settings.

## 5. Results

## 5.1. Comparison with existing methods

Table 4 shows QBAE in the common setting, i.e. one network and one configuration for all seven datasets. Its mean AUROC is 0.828. This is the highest of all methods in the MedIAnomaly benchmark; the best of these, AE-PL, reaches 0.820. MSDE (0.826) is on par with QBAE. However, MSDE chooses its backbone per dataset and refines the features with the test images, so the two results were not obtained under the same conditions. The diference of 0.002 is also smaller than the seed-to-seed variation of our mean $\mathrm { ( S D \approx 0 . 0 0 4 ) }$ QBAE is clearly best on BraTS2021 (0.911, vs. 0.859 for DAE and 0.736 for MSDE). On ISIC2018 (0.751) it exceeds AE-PL, DAE and MSDE, but not the best self-supervised method (0.807). It is weakest on VinDr-CXR and Camelyon16, where methods based on pretrained feature densities perform better. AE-PL uses the same perceptual anomaly score as QBAE, and QBAE improves on it on four of the seven datasets. The largest gains are on ISIC2018 (+0.067) and BraTS2021 $( + 0 . 0 6 0 )$ . Note that our input resolution $( 1 2 8 \times 1 2 8 )$ is higher than that of the benchmark $( 6 4 \times 6 4 )$ On BraTS2021, the pixel-level AP is 0.838 and the best Dice score 0.773, compared with 0.755 and 0.711 for DAE (Table 5). Fig. 2 shows examples from the run with the median Dice score. The tumors are removed from the reconstructions, and the anomaly map follows the lesions. The examples include two dificult cases. The third image labeled normal contains white-matter hyperintensities consistent with age-related change; they are flagged, which is correct for an anomaly detector but counts as a false positive under the tumor labels. The last abnormal image has small, multifocal lesions, which give only a weak response.

Table 4. Image-level results in the common setting (one configuration for all datasets). AE-PL, DAE and the per-dataset best: MedIAnomaly (Cai et al., 2025, Table 6; AUROC, 64×64 input, mean of repeated runs). MSDE: Kar et al. (2026), Table 3 (fixed hyperparameters; AUROC with AP in parentheses). Methods: AE-PL (Shvetsova et al., 2021), DAE (Kascenas et al., 2022), FAE-SSIM (Meissen et al., 2023), AE-U (Mao et al., 2020), MSC (Reiss and Hoshen, 2023), AnatPaste (Sato et al., 2023), AutoDDPM (Bercea et al., 2023). AE-PL uses the same perceptual term as our image score; DAE uses the input corruption that ours modifies. <sup>†</sup>ImageNet pre-trained backbone. The last column picks the best method separately for each dataset; no single method attains all of these values. Among the methods in MedIAnomaly, AE-PL has the highest mean AUROC over the seven datasets (0.820).
<table><tr><td>Dataset</td><td>QBAE AUROC</td><td>QBAE AP</td><td>AE-PL</td><td>DAE</td><td>MSDE AUROC</td><td>Best in C (AP) MedIAnomaly</td></tr><tr><td>RSNA</td><td> $0 . 8 8 1 \pm 0 . 0 0 2$ </td><td> $0 . 8 5 8 \pm 0 . 0 0 5$ </td><td>0.875</td><td>0.861</td><td>0.918 (0.906)</td><td>0.911 (FAE-SSIM†)</td></tr><tr><td>VinDr-CXR</td><td> $0 . 7 2 3 \pm 0 . 0 0 9$ </td><td> $0 . 7 3 0 \pm 0 . 0 1 0$ </td><td>0.753</td><td>0.686</td><td>0.819 (0.797)</td><td>0.769 (AE-U)</td></tr><tr><td>Brain Tumor</td><td> $0 . 9 5 9 \pm 0 . 0 0 3$ </td><td> $0 . 9 2 0 \pm 0 . 0 0 6$ </td><td>0.957</td><td>0.832</td><td>0.981 (0.981)</td><td>0.973 (MSCt)</td></tr><tr><td>LAG</td><td> $0 . 8 4 4 \pm 0 . 0 1 0$ </td><td> $0 . 7 7 3 \pm 0 . 0 1 1$ </td><td>0.856</td><td>0.716</td><td>0.810 ) (0.831)</td><td>0.856 (AE-PL†)</td></tr><tr><td>ISIC2018</td><td> $0 . 7 5 1 \pm 0 . 0 0 9$ </td><td> $0 . 6 5 6 \pm 0 . 0 0 9$ </td><td>0.684</td><td>0.700</td><td>0.705 (0.638)</td><td>0.807 (AnatPaste)</td></tr><tr><td>Camelyon16</td><td> $0 . 7 2 6 \pm 0 . 0 0 5$ </td><td> $0 . 6 4 7 \pm 0 . 0 0 6$ </td><td>0.761</td><td>0.654</td><td>0.812 (0.820)</td><td>0.807 (AutoDDPM)</td></tr><tr><td>BraTS2021</td><td> $0 . 9 1 1 \pm 0 . 0 1 7$ </td><td> $0 . 9 5 6 \pm 0 . 0 1 0$ </td><td>0.851</td><td>0.859</td><td>0.736 (0.867)</td><td>0.859 (DAE)</td></tr><tr><td>Mean</td><td>0.828</td><td>0.792</td><td>0.820</td><td>0.758</td><td>0.826 (0.834)</td><td></td></tr></table>

MSDE scores ImageNet- or AnatPaste-pretrained features (backbone chosen per dataset), refines them by mean shift over the union of the training and test images, and reports a single run.

Table 5. BraTS2021: image- and pixel-level results. Pixel AP and best Dice are computed from the map A(x) averaged over 8 noise draws.
<table><tr><td>Setting</td><td>Image AUROC Pixel AP</td><td></td><td>Best Dice</td></tr><tr><td>Common setting  $( \varepsilon = 1 0 , \mathrm { { K L } \ o n ) }$ </td><td> $0 . 9 1 1 \pm 0 . 0 1 7$ </td><td> $0 . 8 3 8 \pm 0 . 0 1 6$ </td><td> $0 . 7 7 3 \pm 0 . 0 1 7$ </td></tr><tr><td>ε = 30, KL on</td><td> $0 . 9 3 9 \pm 0 . 0 0 3$ </td><td> $0 . 8 4 9 \pm 0 . 0 0 8$ </td><td> $0 . 7 8 6 \pm 0 . 0 0 6$ </td></tr><tr><td> $\varepsilon = 3 0 ,$  , KL off (best on BraTS)</td><td> $0 . 9 4 2 \pm 0 . 0 0 3$ </td><td> $0 . 8 5 5 \pm 0 . 0 0 2$ </td><td> $0 . 7 9 2 \pm 0 . 0 0 3$ </td></tr><tr><td>DAE (best in MedIAnomaly)</td><td>0.859</td><td>0.755</td><td>0.711</td></tr></table>

## 5.2. Efect of the information budget

Fig. 3 shows how the AUROC depends on ε. With input corruption, the mean AUROC changes by at most 0.012 for ε between 3 and 30 (0.816–0.828, about $2 \times 1 0 ^ { 4 }$ to $1 . 2 \times 1 0 ^ { 5 }$ bits). It drops for smaller budgets (0.751 at $\varepsilon = 0 . 3 )$ . Without the bottleneck $( \varepsilon = 1 0 ^ { 8 } )$ ， the mean AUROC falls to 0.797. Without input corruption, the bottleneck is essential. The mean AUROC peaks at ε = 1 (0.805) and falls to 0.590 without the bottleneck. On BraTS2021 and LAG, the AUROC then drops below 0.5, i.e. the network reconstructs abnormal images at least as well as normal ones. With corruption, the bottleneck has little efect on BraTS2021, Brain Tumor and VinDr-CXR, but adds 0.04–0.13 on Camelyon16, RSNA and LAG (Table 6). On BraTS2021, most of the improvement over the uncorrupted network thus comes from the input corruption.

![](images/224ab7a2d49618f266374a444e1de52b96eddf782cff2d925624c343f334c3d6.jpg)  
Fig. 2. BraTS2021 examples from the common setting (run with the median Dice of five seeds). Left: four test images labeled normal; right: four abnormal images. Rows: input; reconstruction (one noise draw); anomaly map A(x) (shown from a noise-free pass of the same network; one square-root color scale for all eight images, without clipping); input with the ground-truth tumor outline.

The best budget depends on the dataset (Table 6). It is large (ε = 30) for BraTS2021, Brain Tumor and VinDr-CXR, intermediate (ε = 10) for LAG and ISIC2018, and small (ε = 3 and 1) for RSNA and Camelyon16. Choosing ε per dataset, with the architecture and all other settings fixed, raises the mean AUROC to 0.847. Because this choice is made on the test set, 0.847 is an optimistic upper bound. Still, it shows that the single parameter ε, which sets the information budget, is enough to adapt one fixed network to datasets of very diferent kinds.

Table 6. Efect of the QB bottleneck and of choosing ε per dataset (AUROC; corruption and KL on unless stated). $\mathrm { ^ { 4 \cdot } Q B \ o f f ^ { \prime \prime } } = \varepsilon = 1 0 ^ { 8 }$ . “Best $\varepsilon ^ { \gamma } \colon$ : only ε chosen per dataset. “Best of all settings”: ε, corruption and KL all chosen. Both “best” columns are chosen on the test set and are optimistic upper bounds. Results without input corruption are shown in Fig. 3.

<table><tr><td>Dataset</td><td></td><td>QB off ε = 10 (common)</td><td>Best ε</td><td>Best of all settings</td></tr><tr><td>RSNA</td><td>0.832</td><td>0.881</td><td>0.890 (ε = 3)</td><td>0.893 (ε = 3, corr. off, KL on)</td></tr><tr><td>VinDr-CXR</td><td>0.734</td><td>0.723</td><td>0.748 (ε = 30)</td><td>0.753 (ε = 100, corr. on, KL off)</td></tr><tr><td>Brain Tumor</td><td>0.961</td><td>0.959</td><td>0.965 (ε = 30)</td><td>0.968 (ε = off, corr. on, KL off)</td></tr><tr><td>LAG</td><td>0.802</td><td>0.844</td><td>0.844 (ε = 10)</td><td>0.860 (ε = 3, corr. on, KL off)</td></tr><tr><td>ISIC2018</td><td>0.738</td><td>0.751</td><td>0.751 (ε = 10)</td><td>0.752 (ε = 30, corr. on, KL off)</td></tr><tr><td>Camelyon16</td><td>0.593</td><td>0.726</td><td>0.794 (ε = 1)</td><td>0.803 (ε = 1, corr. off, KL on)</td></tr><tr><td>BraTS2021</td><td>0.919</td><td>0.911</td><td>0.939 (ε = 30)</td><td>0.942 (ε = 30, corr. on, KL off)</td></tr><tr><td>Mean</td><td>0.797</td><td>0.828</td><td>0.847</td><td>0.853</td></tr></table>

## 6. Discussion

QBAE achieves three things. First, it uses a single architecture and a single configuration for all seven MedIAnomaly datasets and reaches a mean image-level AUROC of 0.828. To our knowledge, this is the highest mean AUROC reported for a method that does not adapt its architecture or settings to each dataset; the best previous such method is AE-PL, at 0.820. It is also on par with MSDE (0.826), which selects its feature extractor for each dataset and uses the test images. Second, it is the best reported method on BraTS2021, both at the image level (AUROC 0.911 vs. 0.859) and at the pixel level (AP 0.838 vs. 0.755). Third, it obtains these results with the noise kept at test time, so that every reconstruction is produced through QB layers whose information is provably bounded. A single parameter, ε, sets this bound and adapts the same network to datasets of very diferent kinds.

![](images/ba29999ecfa1706336be04eaa6e7f8377843d671f29232a1910d7af01f324855.jpg)  
Fig. 3. Image-level AUROC as a function of ε (KL on). Blue: input corruption on; orange: of. “of” = no bottleneck $( \varepsilon = 1 0 ^ { 8 } )$ . Shaded bands: ±1 SD over 5 seeds; rings: best ε per curve; dashed line: common setting ε = 10. The top axis of the mean panel gives the information budget N·c(ε).

## 6.1. One parameter for the information budget

In QBAE, the information that reaches the decoder is set by ε alone. Over the range we tested, the bound spans about 4 $\times 1 0 ^ { 3 }$ to $1 . 7 \times 1 0 ^ { 5 }$ bits, and the network never changes. Other reconstruction-based methods tune the same quantity only indirectly, through the latent size, the depth or the skip connections. The mean AUROC is insensitive to the budget near its optimum: it varies by at most 0.012 for ε between 3 and 30, a factor of five in the budget. Hence a single default works across datasets. The best budget still depends on the dataset. It is large for BraTS2021, Brain Tumor and VinDr-CXR, and small for RSNA and Camelyon16. A plausible reading is that large, conspicuous lesions are best detected when the decoder receives enough detail to reconstruct normal tissue sharply, whereas subtle or difuse changes require a tighter budget, so that they are not reconstructed. We did not test this interpretation directly.

## 6.2. Bottleneck and input corruption are complementary

The two mechanisms address the identity mapping in diferent ways. Without input corruption and without the bottleneck, the network copies its input: the mean AUROC falls to 0.590, and it drops below 0.5 on BraTS2021 and LAG. The bottleneck alone prevents this collapse and raises the mean to 0.805. With input corruption, the network can no longer simply copy, and the bottleneck adds less on average (0.797 to 0.828). Its gain, however, is concentrated where the corruption fits the anomalies least. On

BraTS2021 and Brain Tumor, whose focal lesions resemble the blob-shaped corruption, the bottleneck adds little. On Camelyon16, RSNA and LAG, whose abnormalities are difuse or subtle, it adds 0.04–0.13. The corruption thus tells the network what to remove, and the budget limits how much it can pass where the corruption gives no guidance.

## 6.3. A provable budget at test time

Because the noise is kept at test time, the network used for detection is the noisy network analyzed in Theorem 1, and every reported result is obtained under a stated information budget. Restricting the decoder to a provably limited view of the image did not prevent QBAE from reaching the results above. A budget stated in bits also opens uses beyond anomaly detection. For example, it bounds how much of an image could be recovered from what crosses the bottleneck, which may matter when encoder and decoder run on diferent sites.

## 6.4. Relation to feature-density methods

QBAE and MSDE reach the same mean AUROC, but their strengths difer. QBAE is clearly better on BraTS2021 and better on ISIC2018 and LAG. MSDE is better on VinDr-CXR, Camelyon16 and Brain Tumor. The two also rest on diferent resources. MSDE scores features of a pretrained network, chosen per dataset, and refines them with the test images. QBAE uses pretrained VGG-19 features only in its loss and its anomaly score; the representation that crosses the bottleneck is learned from normal training images alone. The two approaches are therefore complementary, and combining them is a natural next step.

## 6.5. Limitations

Input corruption as a prior. The blob-shaped corruption encodes a prior that anomalies are local and smooth. This prior matches large brain lesions well. On BraTS2021, most of the improvement comes from the corruption rather than from the bottleneck. Without corruption, the best AUROC is 0.802; with corruption but without the bottleneck, it is 0.919 (Fig. 3, Table 6). Part of our advantage on this dataset may therefore reflect how closely the synthetic perturbation resembles real tumors. The same criticism applies to DAE and to methods based on synthetic anomalies. The corruption is not specific to brain MRI, however. It uses random textures, does not model hyperintensity and is identical for all datasets. It also improves LAG, VinDr-CXR and ISIC2018 (by 0.03–0.07 at ε = 10), while it does not help on RSNA and slightly lowers the AUROC on Camelyon16.

Selection on the test set. MedIAnomaly has no validation set. The common setting was chosen because it gave the best mean AUROC on the test sets. A single setting for all datasets and the insensitivity to ε limit this optimism, but do not remove it. The per-dataset results in Table 6 are upper bounds. Selecting ε without labels, for example from the reconstruction error on normal validation images, remains open.

Scope of the bound. Theorem 1 bounds the mutual information between the input and each reconstruction. It is an upper bound that need not be tight, and it does not by itself guarantee detection performance. It holds only because the noise is kept at test time. The averaged noisy score combines K = 8 independent passes; by Proposition 3, it can carry at most K times the per-reconstruction budget.

Other. QBAE depends on ImageNet-pretrained VGG-19 features for its loss and score. We use 128×128 inputs, whereas the reported baselines use 64×64; like the benchmark, we work on downsampled two-dimensional images rather than on full-resolution images or three-dimensional volumes. Pixel-level evaluation is possible only on BraTS2021. In BraTS2021, slices without tumor are labeled normal even when they show other abnormalities such as age-related white-matter hyperintensities (Fig. 2), so some detections count as false positives.

## 6.6. Conclusion

QBAE separates the information bottleneck of an autoencoder from its architecture. A quasi-binarizing layer on every encoder–decoder path bounds the information that reaches the decoder, and this bound depends only on ε and the number of QB elements. The encoder and decoder can therefore be as expressive as reconstruction requires. One network with one configuration reaches the highest mean AUROC on MedIAnomaly among methods that do not adapt to each dataset, and the best reported results on BraTS2021. Its single parameter ε adapts the network to datasets of very diferent kinds.

## CRediT authorship contribution statement

Shouhei Hanaoka: Conceptualization, Methodology, Software, Formal analysis, Investigation, Visualization, Writing – original draft, Funding acquisition. Takahiro Nakao: Validation, Writing – review & editing. Atsushi Takamatsu: Validation, Writing – review & editing. Takeharu Yoshikawa: Resources, Writing – review & editing. Osamu Abe: Supervision, Resources, Writing – review & editing.

## Declaration of competing interest

The Department of Computational Diagnostic Radiology and Preventive Medicine, The University of Tokyo Hospital, is sponsored by HIMEDIC Inc. and Siemens Healthcare K.K. The sponsors had no role in the design of the study, the analysis, the writing of the manuscript or the decision to publish. The authors declare no other competing interests.

Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work the authors used Claude (Anthropic) in order to assist with implementing and debugging code, analyzing experimental logs, preparing figures and tables, and drafting and editing the text. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the publication.

## Ethics statement

This study used only publicly available, de-identified datasets (Section 4.1). No new data were collected, and no additional ethical approval was required.

## Code and data availability

The source code of QBAE, including the training and evaluation scripts and the configurations used in this work, is available at https://github.com/hanaokalog/MedIAn omalyQB. All datasets are publicly available through the MedIAnomaly benchmark (Cai et al., 2025) and the original sources listed in Table 3.

## Acknowledgments

This work was supported by the Japan Society for the Promotion of Science (JSPS) KAKENHI Grant Number JP25K03479 and the Japan Science and Technology Agency (JST) CREST Grant Number JPMJCR21M2. The Department of Computational Diagnostic Radiology and Preventive Medicine, The University of Tokyo Hospital, is sponsored by HIMEDIC Inc. and Siemens Healthcare K.K. This research was conducted using the Supermicro ARS-111GL-DNHR-LCC and FUJITSU Server PRIMERGY CX2550 M7 (Miyabi) at Joint Center for Advanced High Performance Computing (JCAHPC).

## Appendix A. Information-theoretic analysis of the QB layer

This appendix gives the formal statements behind Theorem 1 (Section 3.2.2) and their proofs.

Setting. Throughout, U denotes the encoder input together with any randomness used by the encoder; during training this is $U = \left( X , \xi \right)$ , with $\xi$ the input corruption. Let $z _ { i } = g _ { i } ( U ) = \sigma ( h _ { i } ( U ) ) \in [ 0 , 1 ]$ denote the pre-noise value of QB element i, where $g _ { i }$ is a deterministic (measurable) function. The noisy output is $\tilde { Z } _ { i } = g _ { i } ( U ) + N _ { i }$ . The noise variables $N _ { 1 } , \ldots , N _ { N }$ are i.i.d. Laplace(0, b) with $b = 1 / \varepsilon$ , independent of $U$ . Natural logarithms are used, so information is measured in nats; to convert to bits, divide by ln 2.

## Appendix A.1. Formal statements

Proposition 1 (privacy bound). Each element $\tilde { Z } _ { i }$ is an ε-locally diferentially private release of U (Duchi et al., 2013). Consequently,

$$
I ( U ; \tilde { Z } _ { i } ) \ \leq \ \operatorname* { m i n } \{ \varepsilon , \varepsilon ^ { 2 } \} \ \mathrm { n a t s } .\tag{A.1}
$$

Proposition 2 (amplitude bound). For every element,

$$
I ( U ; \tilde { Z } _ { i } ) \ \leq \ C ( \varepsilon ) \ : = \ \frac { 1 } { 2 } \ln \biggl ( 2 \pi e \Bigl ( \textstyle { \frac { 1 } { 4 } } + \frac { 2 } { \varepsilon ^ { 2 } } \Bigr ) \biggr ) - \ln \frac { 2 e } { \varepsilon } \ \mathrm { n a t s . }\tag{A.2}
$$

Proposition 3 (additivity and data processing). Suppose $\hat { X }$ depends on $U$ only through $\tilde { Z } = ( \tilde { Z } _ { 1 } , \dots , \tilde { Z } _ { N } )$ . Then

$$
I ( X ; { \hat { X } } ) \leq I ( U ; { \tilde { Z } } ) \leq \sum _ { i = 1 } ^ { N } I ( U ; { \tilde { Z } } _ { i } ) .\tag{A.3}
$$

Theorem 1 (formal). Combining Propositions 1–3,

$$
I ( X ; { \hat { X } } ) \ \leq \ N { c ( \varepsilon ) } , \qquad c ( \varepsilon ) : = \operatorname* { m i n } { \big \{ } \varepsilon , \ \varepsilon ^ { 2 } , \ C ( \varepsilon ) { \big \} } \ \operatorname { n a t s } .\tag{A.4}
$$

The privacy bound is the tighter one for $\varepsilon \lesssim 0 . 2 8$ ; the amplitude bound is tighter for all larger ε, including every setting used in this work.

## Appendix A.2. Proof of Proposition 1

The conditional density of $\tilde { Z } _ { i }$ given $U = u { \mathrm { ~ i s ~ } } p ( y \mid u ) = { \textstyle { \frac { 1 } { 2 b } } } \exp \Bigl ( - | y - g _ { i } ( u ) | / b \Bigr )$ . For any two inputs $u , u ^ { \prime }$ and any $y \in \mathbb { R }$ , the triangle inequality gives

$$
\frac { p ( y \mid u ) } { p ( y \mid u ^ { \prime } ) } = \exp \Bigl ( \frac { | y - g _ { i } ( u ^ { \prime } ) | - | y - g _ { i } ( u ) | } { b } \Bigr ) \le \exp \Bigl ( \frac { | g _ { i } ( u ) - g _ { i } ( u ^ { \prime } ) | } { b } \Bigr ) \le e ^ { \varepsilon } ,\tag{A.5}
$$

because $g _ { i } ( u ) , g _ { i } ( u ^ { \prime } ) \in [ 0 , 1 ]$ and $1 / b = \varepsilon$ . Hence the mechanism $u \mapsto \tilde { Z } _ { i }$ is ε-diferentially private with respect to arbitrary changes of its single input. This is ε-local diferential privacy (Dwork et al., 2006; Duchi et al., 2013).

Bound ε. Let $p ( y ) ~ = ~ \mathbb { E } _ { U ^ { \prime } } [ p ( y ~ \vert ~ U ^ { \prime } ) ]$ be the marginal density of $\tilde { Z } _ { i }$ . By (A.5), $p ( y ) \geq e ^ { - \varepsilon } p ( y \mid u )$ for every u. Therefore

$$
I ( U ; \tilde { Z } _ { i } ) = \mathbb { E } \Big [ \ln \frac { p ( \tilde { Z } _ { i } \mid U ) } { p ( \tilde { Z } _ { i } ) } \Big ] \leq \varepsilon .\tag{A.6}
$$

Bound $\varepsilon ^ { 2 }$ . For $\varepsilon < 1$ , the tighter bound $I ( U ; \tilde { Z } _ { i } ) \le \varepsilon ^ { 2 }$ follows from Cuf and Yu (2016, Lemma 1), who show that ε-diferential privacy implies mutual-information diferential privacy with parameter min $\{ \varepsilon , \varepsilon ^ { 2 } \}$ nats. Our setting is the special case of a single record. Combining the two bounds gives Proposition 1. □

## Appendix A.3. Proof of Proposition 2

Reduction to $Z _ { i } .$ . Since $Z _ { i } = g _ { i } ( U )$ and $N _ { i }$ is independent of U, $U  Z _ { i }  \tilde { Z } _ { i }$ is a Markov chain. By the data-processing inequality, $I ( U ; \bar { Z } _ { i } ) \le I ( Z _ { i } ; \tilde { Z } _ { i } )$ . (In fact, equality holds.)

Decomposition. For an additive channel with noise independent of the input,

$$
I ( Z _ { i } ; \tilde { Z } _ { i } ) = h ( \tilde { Z } _ { i } ) - h ( \tilde { Z } _ { i } \mid Z _ { i } ) = h ( \tilde { Z } _ { i } ) - h ( N _ { i } ) .
$$

Noise entropy. The diferential entropy of Laplace(0, b) is $h ( N _ { i } ) = 1 + \ln ( 2 b )$ Output entropy. Since $Z _ { i } \in [ 0 , 1 ]$ , Popoviciu’s inequality gives $\begin{array} { r } { \mathrm { V a r } ( Z _ { i } ) \ \leq \ \frac { 1 } { 4 } } \end{array}$ . Together with $\mathrm { V a r } ( N _ { i } ) = 2 b ^ { 2 }$ and independence, $\begin{array} { r } { \mathrm { V a r } ( \tilde { Z } _ { i } ) \leq \frac { 1 } { 4 } + 2 b ^ { 2 } } \end{array}$ . Among all densities with a given variance, the Gaussian maximizes diferential entropy (Cover and Thomas, 2006, Thm. 8.6.5). Hence

$$
\begin{array} { r } { h ( \tilde { Z } _ { i } ) \leq \frac { 1 } { 2 } \ln \Bigl ( 2 \pi e \bigl ( \frac { 1 } { 4 } + 2 b ^ { 2 } \bigr ) \Bigr ) . } \end{array}
$$

Conclusion. Combining these and substituting $b = 1 / \varepsilon$

$$
\begin{array} { r } { I ( U ; \tilde { Z } _ { i } ) \le \frac { 1 } { 2 } \ln \Big ( 2 \pi e \Big ( \frac { 1 } { 4 } + \frac { 2 } { \varepsilon ^ { 2 } } \Big ) \Big ) - 1 - \ln \frac { 2 } { \varepsilon } = C ( \varepsilon ) . \qquad \big \sqcup } \end{array}
$$

Remark. $C ( \varepsilon )$ is a (generally loose) upper bound on the capacity of the amplitudeconstrained additive Laplace channel. It is used here only because it is explicit. For large ε it behaves as ln $\begin{array} { r } { \varepsilon + \frac { 1 } { 2 } \ln ( \pi / ( 8 e ) ) + o ( 1 ) } \end{array}$ , i.e. it grows logarithmically; as $\varepsilon \to 0$ it tends to the constant $\textstyle \frac { 1 } { 2 } \ln ( \pi / e ) \approx 0 . 0 7 2$ nats rather than to zero, which is why the privacy bound is the tighter one for small ε.

## Appendix A.4. Proof of Proposition 3

First inequality. By assumption, $\hat { X }$ is a (possibly randomized) function of $\tilde { Z }$ alone, with any randomness independent of everything else. Hence $X  U  { \tilde { Z } }  { \hat { X } }$ is a Markov chain, since $X$ is a component of $U$ . Two applications of the data-processing inequality give

$$
I ( X ; \hat { X } ) \leq I ( X ; \tilde { Z } ) \leq I ( U ; \tilde { Z } ) .
$$

Second inequality. Conditionally on $U _ { : }$ , the outputs $\tilde { Z } _ { i } = g _ { i } ( U ) + N _ { i }$ are independent, because the $N _ { i }$ are independent of each other and of U. Hence $\begin{array} { r } { h ( \tilde { Z } \mid U ) = \sum _ { i } h ( N _ { i } ) = } \end{array}$ $\textstyle \sum _ { i } h \big ( \tilde { Z } _ { i } \mid U \big )$ . By subadditivity of diferential entropy, $\begin{array} { r } { h ( \tilde { Z } ) \leq \sum _ { i } { h ( \tilde { Z } _ { i } ) } } \end{array}$ . Therefore

$$
I ( U ; \tilde { Z } ) = h ( \tilde { Z } ) - h ( \tilde { Z } \mid U ) \le \sum _ { i = 1 } ^ { N } \left[ h ( \tilde { Z } _ { i } ) - h ( \tilde { Z } _ { i } \mid U ) \right] = \sum _ { i = 1 } ^ { N } I ( U ; \tilde { Z } _ { i } ) .
$$

Third inequality. Each summand is bounded by Propositions 1 and 2. □

Remark. The sum of the privacy bounds alone could also be obtained from the basic composition theorem of diferential privacy: the joint release is $N { \varepsilon } – \mathrm { D P }$ , so $\left( \mathrm { A . 6 } \right)$ gives $I ( U ; \tilde { Z } ) \le N \varepsilon$ . The argument above is needed to sum the amplitude bounds.

## Appendix A.5. Numerical values

The per-element values in Table 1 are $c ( \varepsilon ) /$ ln 2, i.e. the smaller of min $\{ \varepsilon , \varepsilon ^ { 2 } \} /$ ln 2 (privacy) and $C ( \varepsilon ) /$ ln 2 (amplitude). The totals multiply this by $N = 3 2 , 7 6 8$

<table><tr><td>ε min</td><td> $\{ \varepsilon , \varepsilon ^ { 2 } \}$  [nats]</td><td>C(ε) [nats]</td><td>Bound per element [bits]</td><td>Total [bits]</td></tr><tr><td>0.3</td><td>0.09</td><td>0.078</td><td>0.112</td><td>3,685</td></tr><tr><td>1</td><td>1</td><td>0.131</td><td>0.189</td><td>6,205</td></tr><tr><td>3</td><td>3</td><td>0.449</td><td>0.648</td><td>21,238</td></tr><tr><td>10</td><td>10</td><td>1.374</td><td>1.982</td><td>64,941</td></tr><tr><td>30</td><td>30</td><td>2.438</td><td>3.518</td><td>115,267</td></tr><tr><td>100</td><td>100</td><td>3.638</td><td>5.249</td><td>171,994</td></tr></table>

At $\varepsilon = 0 . 3$ the two bounds are close (0.090 vs 0.078 nats), consistent with the crossover at $\varepsilon \approx 0 . 2 8$ stated after (A.4).

## Appendix B. Network details

Fig. B.1 shows every layer of the network described in Section 3.3; Table B.1 lists the levels. The network has 60.7 M parameters. Most of them sit in the two deepest decoder levels (Table B.1), which operate on $4 \times 4$ and $8 \times 8$ feature maps and therefore add little computation; a forward pass costs 4.2 GFLOPs per $1 2 8 \times 1 2 8$ image (2.1 G multiply–accumulates).

Table B.1. Levels of the network (128 × 128 input).
<table><tr><td>Level</td><td>Resolution</td><td>Width</td><td>QB elements (channels per pixel)</td><td>Decoder attention</td><td>Parameters  $( \mathrm { e n c . } ~ / ~ \mathrm { d e c . } )$ </td></tr><tr><td>1</td><td> $1 2 8 ^ { 2 }$ </td><td>16</td><td>16,384 (1)</td><td></td><td>0.003M / 0.03M</td></tr><tr><td>2</td><td> $6 4 ^ { 2 }$ </td><td>32</td><td>8,192 (2)</td><td>channel attention + FFN</td><td>0.015 M / 0.12 M</td></tr><tr><td>3</td><td> $3 2 ^ { 2 }$ </td><td>64</td><td>4,096 (4)</td><td>spatial attention + FFN</td><td>0.06M / 0.46M</td></tr><tr><td>4</td><td> $1 6 ^ { 2 }$ </td><td>128</td><td>2,048 (8)</td><td>spatial attention + FFN</td><td>0.23 M / 1.84M</td></tr><tr><td>5</td><td> $8 ^ { 2 }$ </td><td>256</td><td>1,024 (16)</td><td>spatial  $\mathrm { a t t e n t i o n + F F N }$ </td><td>0.92 M / 7.35 M</td></tr><tr><td>6</td><td> $4 ^ { 2 }$ </td><td>512</td><td>512 (32)</td><td> $\mathrm { s p a t i a l ~ a t t e n t i o n + F F N }$ </td><td>3.67 M / 29.39 M</td></tr><tr><td>7 (bottom)</td><td> $2 ^ { 2 }$ </td><td>1024</td><td>512 (fully connected)</td><td></td><td>14.69M / −</td></tr></table>

Not listed: DC-AE downsampling layers (0.79 M), $1 \times 1$ convolutions of the skip connections (0.05 M), fully connected bottom layers (1.05 M) and output head (0.005 M). Decoder parameters include the upsampling layer, cross-attention gate, convolution block and attention block of each level.

Components. Convolution blocks use two weight-standardized (Qiao et al., 2019) $3 \times 3$ convolutions with reflection padding, each followed by Swish and GroupNorm, and a residual connection (1 × 1 convolution when the width changes). DC-AE downsampling adds a learned path (convolution followed by pixel unshufle; Shi et al., 2016) to a parameter-free path (pixel unshufle followed by averaging over channel groups); upsampling mirrors this with pixel shufle and channel duplication. The cross-attention gate projects the upsampled decoder features u and the post-QB skip features sˆ to $F _ { \mathrm { i n t } }$ channels (half the level width), applies CBAM (Woo et al., 2018) channel and spatial attention to their sum, and outputs a map $a \in [ 0 , 1 ]$ ; the decoder then concatenates $u \odot ( 1 - a )$ and $\hat { s } \odot a$ . Spatial attention uses 8 heads of softmax self-attention over all positions; channel attention follows Restormer (Zamir et al., 2022; 4 heads, transposed attention); the feedforward block is a gated $1 \times 1 ~ /$ depth-wise $3 \times 3$ network. All attention and feed-forward blocks are pre-normalized residual blocks whose output projections are initialized to zero. At the bottom, a 1 × 1 convolution reduces the 1024 $\times ~ 2 \times 2$ map to $1 2 8 \times 2 \times 2$ , which is flattened and passed through a fully connected layer $5 1 2  5 1 2$ with GroupNorm and a 1 × 1-convolution shortcut, the learnable arctangent of Section 3.2.3, the QB layer, and a mirrored fully connected layer back to $1 0 2 4 \times 2 \times 2$

![](images/9963cb3b8547133263181735c4e5c87ada5a821ca3850e3db2c70f76287f2ba5.jpg)  
Fig. B.1. Detailed architecture of QBAE. (a) Overall network. (b) QB layer. (c), (d) DC-AE residual down- and upsampling. (e) Convolution block. (f) Decoder (up) block. (g) Cross-attention gate. (h) Attention and feed-forward blocks. (i) Fully connected layers around the bottom QB layer. (j) Training objective and anomaly score.

## Appendix C. Input corruption

For an image $\boldsymbol { x } \in \mathbb { R } ^ { C \times H \times W }$ , the corruption η of Section 3.4 is generated as follows. All random variables are drawn independently for each image and each training step.

1. Draw the smoothing scales $r _ { 1 } , r _ { 2 } \sim \mathcal { U } ( 1 , 1 7 )$ pixels, a threshold $t \sim \mathcal { U } ( 0 , 2 )$ and an exponent $p \sim \mathcal { U } ( 0 . 0 1 , 2 . 0 1 )$ .

2. Texture field: draw white Gaussian noise $w _ { 1 } \in \mathbb { R } ^ { C \times H \times W }$ , convolve each channel with a Gaussian kernel of standard deviation $r _ { 1 }$ , and normalize the result $g _ { 1 }$ to unit standard deviation over the image.

3. Mask field: draw $w _ { 2 } \in \mathbb { R } ^ { H \times W }$ , smooth it with standard deviation $r _ { 2 }$ and normalize it to $g _ { 2 }$ in the same way. The blob mask is

$$
m = \operatorname* { m a x } ( g _ { 2 } - t , 0 ) ^ { p } ,
$$

which is zero outside a few smooth patches; larger t gives fewer and smaller blobs, and $p$ controls how sharp their edges are.

4. The corruption is

$$
\eta = \sigma \mathrm { s t d } ( x ) g _ { 1 } \odot m ,
$$

with the mask shared by all channels, std(x) the standard deviation of the image and $\sigma = 1$ in all experiments.

Compared with the coarse noise of Kascenas et al. (2022), which perturbs the whole image with upsampled low-resolution Gaussian noise, this corruption alters only parts of the image, at scales that vary from image to image, and leaves the rest unchanged. In our implementation the Gaussian smoothing is done on the GPU in the Fourier domain, on a canvas of twice the image size that is then cropped, so that the periodic boundary of the FFT has no visible efect.

## Appendix D. Other test-time readouts

Instead of keeping the noise, the QB layers can be read out without noise $( \tilde { z } _ { i } = z _ { i } )$ or binarized $\begin{array} { r } { ( \tilde { z } _ { i } = \mathcal { H } [ z _ { i } > \frac { 1 } { 2 } ] ) } \end{array}$ . Theorem 1 does not cover the noise-free readout; the binarized readout limits the information to N bits, since each element takes only two values. In the common setting, the mean AUROC over the seven datasets is 0.817 without noise and 0.812 with binarization, compared with 0.828 with the noise kept and averaged over $K = 8$ draws. Binarization costs little except on BraTS2021 (0.852 vs. 0.911), which indicates that the learned code is close to binary. The information carried by the binarized code, estimated as $\textstyle \sum _ { i } h _ { 2 } ( p _ { i } )$ from the firing frequencies $p _ { i }$ of the elements on the test set $( h _ { 2 } { : }$ binary entropy), is on average 14,800 bits per image with the sparsity penalty and 26,300 bits without it.

Abadi, M., Chu, A., Goodfellow, I., McMahan, H.B., Mironov, I., Talwar, K., Zhang, L., 2016. Deep learning with diferential privacy, in: Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security (CCS), pp. 308–318.

Alemi, A.A., Fischer, I., Dillon, J.V., Murphy, K., 2017. Deep variational information bottleneck, in: International Conference on Learning Representations (ICLR).

Alemi, A., Poole, B., Fischer, I., Dillon, J., Saurous, R.A., Murphy, K., 2018. Fixing a broken ELBO, in: Proceedings of the 35th International Conference on Machine Learning (ICML), PMLR 80, pp. 159–168.

Baid, U., Ghodasara, S., Mohan, S., Bilello, M., Calabrese, E., Colak, E., Farahani, K., Kalpathy-Cramer, J., Kitamura, F.C., Pati, S., Prevedello, L.M., Rudie, J.D., Sako, C., Shinohara, R.T., Bergquist, T., Chai, R., Eddy, J., Elliott, J., Reade, W., Schafter, T., Yu, T., Zheng, J., Moawad, A.W., Coelho, L.O., McDonnell, O., Miller, E., Moron, F.E., Oswood, M.C., Shih, R.Y., Siakallis, L., Bronstein, Y., Mason, J.R., Miller, A.F., Choudhary, G., Agarwal, A., Besada, C.H., Derakhshan, J.J., Diogo, M.C., Do-Dai, D.D., Farage, L., Go, J.L., Hadi, M., Hill, V.B., Iv, M., Joyner, D., Lincoln, C., Lotan, E., Miyakoshi, A., Sanchez-Montano, M., Nath, J., Nguyen, X.V., Nicolas-Jilwan, M., Jimenez, J.O., Ozturk, K., Petrovic, B.D., Shah, C., Shah, L.M., Sharma, M., Simsek, O., Singh, A.K., Soman, S., Statsevych, V., Weinberg, B.D., Young, R.J., Ikuta, I., Agarwal, A.K., Cambron, S.C., Silbergleit, R., Dusoi, A., Postma, A.A., Letourneau-Guillon, L., Guzman Perez-Carrillo, G.J., Saha, A., Soni, N., Zaharchuk, G., Zohrabian, V.M., Chen, Y., Cekic, M.M., Rahman, A., Small, J.E., Sethi, V., Davatzikos, C., Mongan, J., Hess, C., Cha, S., Villanueva-Meyer, J., Freymann, J.B., Kirby, J.S., Wiestler, B., Crivellaro, P., Colen, R.R., Kotrotsou, A., Marcus, D., Milchenko, M., Nazeri, A., Fathallah-Shaykh, H., Wiest, R., Jakab, A., Weber, M.-A., Mahajan, A., Menze, B., Flanders, A.E., Bakas, S., 2021. The RSNA-ASNR-MICCAI BraTS 2021 benchmark on brain tumor segmentation and radiogenomic classification. arXiv:2107.02314.

Ballé, J., Laparra, V., Simoncelli, E.P., 2017. End-to-end optimized image compression, in: International Conference on Learning Representations (ICLR).

Bao, J., Sun, H., Deng, H., He, Y., Zhang, Z., Li, X., 2024. BMAD: Benchmarks for medical anomaly detection, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pp. 4042–4053.

Baur, C., Denner, S., Wiestler, B., Navab, N., Albarqouni, S., 2021. Autoencoders for unsupervised anomaly segmentation in brain MR images: A comparative study. Medical Image Analysis 69, 101952. https://doi.org/10.1016/j.media.2020.101952

Bercea, C.I., Neumayr, M., Rueckert, D., Schnabel, J.A., 2023. Mask, stitch, and re-sample: Enhancing robustness and generalizability in anomaly detection through automatic difusion models, in: ICML 2023 Workshop on Interpretable Machine Learning in Healthcare. arXiv:2305.19643.

Cai, Y., Chen, H., Cheng, K.-T., 2024. Rethinking autoencoders for medical anomaly detection from a theoretical perspective, in: Medical Image Computing and Computer Assisted Intervention (MICCAI), LNCS 15011, pp. 544–554. https://doi.org/10.1007/978-3-031-7 2120-5\_51

Cai, Y., Zhang, W., Chen, H., Cheng, K.-T., 2025. MedIAnomaly: A comparative study of anomaly detection in medical images. Medical Image Analysis 102, 103500. https: //doi.org/10.1016/j.media.2025.103500

Chen, J., Cai, H., Chen, J., Xie, E., Yang, S., Tang, H., Li, M., Lu, Y., Han, S., 2025. Deep compression autoencoder for eficient high-resolution difusion models, in: International Conference on Learning Representations (ICLR).

Cheng, J., Huang, W., Cao, S., Yang, R., Yang, W., Yun, Z., Wang, Z., Feng, Q., 2015. Enhanced performance of brain tumor classification via tumor region augmentation and partition. PLoS ONE 10, e0140381.

Choi, K., Tatwawadi, K., Grover, A., Weissman, T., Ermon, S., 2019. Neural joint sourcechannel coding, in: Proceedings of the 36th International Conference on Machine Learning (ICML), PMLR 97, pp. 1182–1192.

Codella, N., Rotemberg, V., Tschandl, P., Celebi, M.E., Dusza, S., Gutman, D., Helba, B., Kalloo, A., Liopyris, K., Marchetti, M., Kittler, H., Halpern, A., 2019. Skin lesion analysis toward melanoma detection 2018: A challenge hosted by the International Skin Imaging Collaboration (ISIC). arXiv:1902.03368.

Cover, T.M., Thomas, J.A., 2006. Elements of Information Theory, 2nd ed. Wiley, Hoboken, NJ.

Cuf, P., Yu, L., 2016. Diferential privacy as a mutual information constraint, in: Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security (CCS), pp. 43–54. https://doi.org/10.1145/2976749.2978308

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., Fei-Fei, L., 2009. ImageNet: A large-scale hierarchical image database, in: IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 248–255.

Duchi, J.C., Jordan, M.I., Wainwright, M.J., 2013. Local privacy and statistical minimax rates, in: IEEE 54th Annual Symposium on Foundations of Computer Science (FOCS), pp. 429– 438. https://doi.org/10.1109/FOCS.2013.53

Dwork, C., McSherry, F., Nissim, K., Smith, A., 2006. Calibrating noise to sensitivity in private data analysis, in: Theory of Cryptography Conference (TCC), LNCS 3876, pp. 265–284.

Ehteshami Bejnordi, B., Veta, M., van Diest, P.J., van Ginneken, B., Karssemeijer, N., Litjens, G., van der Laak, J.A.W.M., the CAMELYON16 Consortium, Hermsen, M., Manson, Q.F., Balkenhol, M., Geessink, O., Stathonikos, N., van Dijk, M.C.R.F., Bult, P., Beca, F., Beck, A.H., Wang, D., Khosla, A., Gargeya, R., Irshad, H., Zhong, A., Dou, Q., Li, Q., Chen, H., Lin, H.-J., Heng, P.-A., Haß, C., Bruni, E., Wong, Q., Halici, U., Öner, M.Ü., Cetin-Atalay, R., Berseth, M., Khvatkov, V., Vylegzhanin, A., Kraus, O., Shaban, M., Rajpoot, N., Awan, R., Sirinukunwattana, K., Qaiser, T., Tsang, Y.-W., Tellez, D., Annuscheit, J., Hufnagl, P., Valkonen, M., Kartasalo, K., Latonen, L., Ruusuvuori, P., Liimatainen, K., Albarqouni, S., Mungal, B., George, A., Demirci, S., Navab, N., Watanabe, S., Seno, S., Takenaka, Y., Matsuda, H., Ahmady Phoulady, H., Kovalev, V., Kalinovsky, A., Liauchuk, V., Bueno, G., Fernandez-Carrobles, M.M., Serrano, I., Deniz, O., Racoceanu, D., Venâncio, R., 2017. Diagnostic assessment of deep learning algorithms for detection of lymph node metastases in women with breast cancer. JAMA 318, 2199–2210.

Gong, D., Liu, L., Le, V., Saha, B., Mansour, M.R., Venkatesh, S., van den Hengel, A., 2019. Memorizing normality to detect anomaly: Memory-augmented deep autoencoder for unsupervised anomaly detection, in: IEEE/CVF International Conference on Computer Vision (ICCV), pp. 1705–1714.

Hamada, A., 2025. Br35H :: Brain tumor detection 2020. IEEE Dataport. https://doi.org/ 10.21227/tbkk-q937

Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick, M., Mohamed, S., Lerchner, A., 2017. β-VAE: Learning basic visual concepts with a constrained variational framework, in: International Conference on Learning Representations (ICLR).

Johnson, J., Alahi, A., Fei-Fei, L., 2016. Perceptual losses for real-time style transfer and superresolution, in: European Conference on Computer Vision (ECCV), LNCS 9906, pp. 694–711.

Kaiser, Ł., Bengio, S., 2018. Discrete autoencoders for sequence models. arXiv:1801.09797.

Kar, P., Lakshmi S, G., Bej, S., 2026. Improved anomaly detection in medical images via mean shift density enhancement. arXiv:2604.19191v1.

Kascenas, A., Pugeault, N., O’Neil, A.Q., 2022. Denoising autoencoders for unsupervised anomaly detection in brain MRI, in: Proceedings of the 5th International Conference on Medical Imaging with Deep Learning (MIDL), PMLR 172, pp. 653–664.

Kascenas, A., Sanchez, P., Schrempf, P., Wang, C., Clackett, W., Mikhael, S.S., Voisey, J.P., Goatman, K., Weir, A., Pugeault, N., Tsaftaris, S.A., O’Neil, A.Q., 2023. The role of noise in denoising models for anomaly detection in medical images. Medical Image Analysis 90, 102963. https://doi.org/10.1016/j.media.2023.102963

Kingma, D.P., Welling, M., 2014. Auto-encoding variational Bayes, in: International Conference on Learning Representations (ICLR).

Li, L., Xu, M., Wang, X., Jiang, L., Liu, H., 2019. Attention based glaucoma detection: A large-scale database and CNN model, in: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10571–10580.

Loshchilov, I., Hutter, F., 2017. SGDR: Stochastic gradient descent with warm restarts, in: International Conference on Learning Representations (ICLR).

Loshchilov, I., Hutter, F., 2019. Decoupled weight decay regularization, in: International Conference on Learning Representations (ICLR).

Mao, Y., Xue, F.-F., Wang, R., Zhang, J., Zheng, W.-S., Liu, H., 2020. Abnormality detection in chest X-ray images using uncertainty prediction autoencoders, in: Medical Image Computing and Computer Assisted Intervention (MICCAI), LNCS 12266, pp. 529–538. https://doi. org/10.1007/978-3-030-59725-2\_51

McGregor, A., Mironov, I., Pitassi, T., Reingold, O., Talwar, K., Vadhan, S., 2010. The limits of two-party diferential privacy, in: IEEE 51st Annual Symposium on Foundations of Computer Science (FOCS), pp. 81–90.

Meissen, F., Paetzold, J., Kaissis, G., Rueckert, D., 2023. Unsupervised anomaly localization with structural feature-autoencoders, in: Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries (BrainLes 2022), LNCS 13769, pp. 14–24. https://doi.org/10 .1007/978-3-031-33842-7\_2

Mireshghallah, F., Taram, M., Ramrakhyani, P., Jalali, A., Tullsen, D., Esmaeilzadeh, H., 2020. Shredder: Learning noise distributions to protect inference privacy, in: Proceedings of the 25th International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS). https://doi.org/10.1145/3373376.3378522

Ng, A., 2011. Sparse autoencoder. CS294A Lecture notes, Stanford University.

Nguyen, H.Q., Lam, K., Le, L.T., Pham, H.H., Tran, D.Q., Nguyen, D.B., Le, D.D., Pham, C.M., Tong, H.T.T., Dinh, D.H., Do, C.D., Doan, L.T., Nguyen, C.N., Nguyen, B.T., Nguyen, Q.V., Hoang, A.D., Phan, H.N., Nguyen, A.T., Ho, P.H., Ngo, D.T., Nguyen, N.T., Nguyen, N.T., Dao, M., Vu, V., 2022. VinDr-CXR: An open dataset of chest X-rays with radiologist’s annotations. Scientific Data 9, 429.

Oktay, O., Schlemper, J., Le Folgoc, L., Lee, M., Heinrich, M., Misawa, K., Mori, K., Mc-Donagh, S., Hammerla, N.Y., Kainz, B., Glocker, B., Rueckert, D., 2018. Attention U-Net: Learning where to look for the pancreas, in: Medical Imaging with Deep Learning (MIDL). arXiv:1804.03999.

Pascanu, R., Mikolov, T., Bengio, Y., 2013. On the dificulty of training recurrent neural networks, in: Proceedings of the 30th International Conference on Machine Learning (ICML), PMLR 28, pp. 1310–1318.

Pinaya, W.H.L., Graham, M.S., Gray, R., da Costa, P.F., Tudosiu, P.-D., Wright, P., Mah, Y.H., MacKinnon, A.D., Teo, J.T., Jager, R., Werring, D., Rees, G., Nachev, P., Ourselin, S., Cardoso, M.J., 2022. Fast unsupervised brain anomaly detection and segmentation with difusion models, in: Medical Image Computing and Computer Assisted Intervention (MIC-CAI), pp. 705–714. https://doi.org/10.1007/978-3-031-16452-1\_67

Qiao, S., Wang, H., Liu, C., Shen, W., Yuille, A., 2019. Micro-batch training with batch-channel normalization and weight standardization. arXiv:1903.10520.

Ramachandran, P., Zoph, B., Le, Q.V., 2017. Searching for activation functions. arXiv:1710.05941.

Reiss, T., Hoshen, Y., 2023. Mean-shifted contrastive loss for anomaly detection, in: Proceedings of the AAAI Conference on Artificial Intelligence 37(2), pp. 2155–2162. https://doi.org/ 10.1609/aaai.v37i2.25309

Ronneberger, O., Fischer, P., Brox, T., 2015. U-Net: Convolutional networks for biomedical image segmentation, in: Medical Image Computing and Computer-Assisted Intervention (MICCAI), LNCS 9351, pp. 234–241.

Salakhutdinov, R., Hinton, G., 2009. Semantic hashing. International Journal of Approximate Reasoning 50, 969–978.

Saleh, A., Sukaik, R., Abu-Naser, S.S., 2020. Brain tumor classification using deep learning, in: International Conference on Assistive and Rehabilitation Technologies (iCareTech), IEEE, pp. 131–136.

Sato, J., Suzuki, Y., Wataya, T., Nishigaki, D., Kita, K., Yamagata, K., Tomiyama, N., Kido, S., 2023. Anatomy-aware self-supervised learning for anomaly detection in chest radiographs. iScience 26, 107086. https://doi.org/10.1016/j.isci.2023.107086

Schlegl, T., Seeböck, P., Waldstein, S.M., Langs, G., Schmidt-Erfurth, U., 2019. f-AnoGAN: Fast unsupervised anomaly detection with generative adversarial networks. Medical Image Analysis 54, 30–44.

Shi, W., Caballero, J., Huszár, F., Totz, J., Aitken, A.P., Bishop, R., Rueckert, D., Wang, Z., 2016. Real-time single image and video super-resolution using an eficient sub-pixel convolutional neural network, in: IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1874–1883.

Shih, G., Wu, C.C., Halabi, S.S., Kohli, M.D., Prevedello, L.M., Cook, T.S., Sharma, A., Amorosa, J.K., Arteaga, V., Galperin-Aizenberg, M., Gill, R.R., Godoy, M.C.B., Hobbs, S., Jeudy, J., Laroia, A., Shah, P.N., Vummidi, D., Yaddanapudi, K., Stein, A., 2019. Augmenting the National Institutes of Health chest radiograph dataset with expert annotations of possible pneumonia. Radiology: Artificial Intelligence 1, e180041. https: //doi.org/10.1148/ryai.2019180041

Shvetsova, N., Bakker, B., Fedulova, I., Schulz, H., Dylov, D.V., 2021. Anomaly detection in medical imaging with deep perceptual autoencoders. IEEE Access 9, 118571–118583. https://doi.org/10.1109/ACCESS.2021.3107163

Simonyan, K., Zisserman, A., 2015. Very deep convolutional networks for large-scale image recognition, in: International Conference on Learning Representations (ICLR).

Tishby, N., Pereira, F.C., Bialek, W., 1999. The information bottleneck method, in: Proceedings of the 37th Annual Allerton Conference on Communication, Control and Computing, pp. 368–377.

van den Oord, A., Vinyals, O., Kavukcuoglu, K., 2017. Neural discrete representation learning, in: Advances in Neural Information Processing Systems (NeurIPS) 30.

Vincent, P., Larochelle, H., Bengio, Y., Manzagol, P.-A., 2008. Extracting and composing robust features with denoising autoencoders, in: Proceedings of the 25th International Conference on Machine Learning (ICML), pp. 1096–1103.

Wolleb, J., Bieder, F., Sandkühler, R., Cattin, P.C., 2022. Difusion models for medical anomaly detection, in: Medical Image Computing and Computer Assisted Intervention (MICCAI), LNCS 13438, pp. 35–45. https://doi.org/10.1007/978-3-031-16452-1\_4

Woo, S., Park, J., Lee, J.-Y., Kweon, I.S., 2018. CBAM: Convolutional block attention module, in: European Conference on Computer Vision (ECCV), LNCS 11211, pp. 3–19.

Wu, Y., He, K., 2018. Group normalization, in: European Conference on Computer Vision (ECCV), LNCS 11217, pp. 3–19.

Wyatt, J., Leach, A., Schmon, S.M., Willcocks, C.G., 2022. AnoDDPM: Anomaly detection with denoising difusion probabilistic models using simplex noise, in: IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pp. 650–656.

You, Z., Cui, L., Shen, Y., Yang, K., Lu, X., Zheng, Y., Le, X., 2022. A unified model for multiclass anomaly detection, in: Advances in Neural Information Processing Systems (NeurIPS) 35, pp. 4571–4584.

Zamir, S.W., Arora, A., Khan, S., Hayat, M., Khan, F.S., Yang, M.-H., 2022. Restormer: Eficient transformer for high-resolution image restoration, in: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5728–5739.

Zimmerer, D., Kohl, S.A.A., Petersen, J., Isensee, F., Maier-Hein, K.H., 2018. Context-encoding variational autoencoder for unsupervised anomaly detection. arXiv:1812.05941.