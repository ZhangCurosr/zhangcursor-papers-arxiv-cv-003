# Seeing through Denoising: Frequency-guided One-step Denoising for Infrared Small Target Detection

Xuelin Qian, Zixuan Wang, Chang Xu, Dingwen Zhang and Junwei Han

Abstract—Infrared small target detection (IRSTD) is a challenging task due to the extremely small size of targets, low signalto-noise ratios, and heavy background clutter in infrared imagery. While the progress of deep learning has advanced this field, existing methods primarily adopt discriminative paradigms that directly map images to target masks, overlooking the potential of generative modeling. In this paper, we propose DiffIRSTD, a novel generative framework that reformulates infrared small target detection as a background reconstruction problem. Instead of explicitly predicting target regions, we reinterpret small targets as noise-like signals and leverage a diffusion-based model to recover a clean, target-free background from a perturbed input image. The residual between the reconstructed background and the original image is then exploited to highlight candidate target regions, providing explicit cues for subsequent detection. To address the computational cost typically associated with diffusion sampling, we introduce a one-step denoising strategy that enables efficient background reconstruction while maintaining practical inference speed. Furthermore, we present a frequency-guided perception loss to preserve high-frequency background details and focus more on the reconstruction quality of target regions. More importantly, our DiffIRSTD operates as a plug-and-play framework, compatible with existing IRSTD models without increasing the computational complexity of detectors. Extensive experiments on IRSTD-1K, NUAA-SIRST, and NUDT-SIRST datasets demonstrate the superiority of our method, achieving state-of-the-art performance across three benchmarks.

Index Terms—Infrared small target detection, one-step denoising, background reconstruction, frequency-guided perceptual loss.

## I. INTRODUCTION

NFRARED small target detection (IRSTD) aims to auto-I matically localize and segment small targets in infrared imagery, where targets often occupy only a few pixels and lack clear structural or texture cues [1]. Such targets typically appear as dim and point-like targets, such as distant aircraft, missiles, or drones, in complex and cluttered backgrounds. IRSTD is critical in numerous military and civilian scenarios, including early warning systems, long-range surveillance, and autonomous sensing. However, the long imaging distance, low signal-to-noise ratio (SNR), and strong background interference often cause targets to be submerged in noise and clutter, making robust and accurate detection highly challenging.

Recent advances in deep learning have significantly improved IRSTD performance. Early approaches mainly rely on U-Net architectures, which enhance small target representation through multi-scale feature interaction [2]–[5]. Transformerbased models are later introduced to capture long-range dependencies and global contextual information, further boosting detection accuracy [6]–[8]. More recently, large vision foundation models, such as Mamba [9]–[11] and SAM [12]– [14], have been explored to enrich feature representations and improve generalization. Despite these developments, most existing methods remain fundamentally discriminative, focusing on learning direct mappings from images to target masks, while the potential of generative modeling in IRSTD remains underexplored.

Diffusion models [15] have recently demonstrated remarkable capability in generative modeling and have achieved impressive success in image synthesis, restoration, and inpainting. Existing diffusion-based studies primarily explore two directions of either using diffusion models to generate additional training samples to alleviate data scarcity [16], or employing diffusion-based networks as iterative decoders to progressively refine predicted masks [17]–[19]. Nevertheless, data generation struggles to faithfully synthesize small infrared targets due to their low intensity and ambiguous boundaries, whereas iterative diffusion decoders may introduce additional computational overhead, which limits deployment in resourceconstrained scenarios.

From an imaging perspective, infrared small targets typically exhibit intensity values that differ from their surrounding background, appearing as local “anomalies” against the smooth background (see Fig. 1(a)). However, factors such as blurry target boundaries, complex background textures, and low SNR frequently obscure these “anomalies”. Motivated by this analysis, we propose a novel generative paradigm for infrared small target detection, termed DiffIRSTD. Instead of directly predicting targets from raw images, we reinterpret small targets as noise-like signals and leverage diffusion models to denoise them from the image, as illustrated in Fig. 1(b)-(c). By reconstructing a clean background image and comparing it with the original input, the target regions can be explicitly highlighted (Fig. 1(d)). This counterintuitive perspective transforms the detection problem into a background reconstruction problem. Reconstructing the low-rank background is expected to be easier than directly generating small target masks, and this pipeline is not required to alter the architecture or increase the complexity of the downstream detector.

Specifically, our DiffIRSTD framework consists of a reconstruction network for generating target-free background images and a detection network for localizing small targets. Given an input infrared image, we first perturb it by adding Gaussian noise, where the target regions are implicitly treated as part of the noise. The noisy image is then processed by a diffusion model to reconstruct a clean background image in which the targets are removed. Diffusion models typically require dozens or even hundreds of sampling steps to generate high-quality images. Such heavy iterative sampling is computationally expensive and conflicts with the efficiency requirements of infrared small target detection. To address this issue, we propose a one-step denoising strategy, which performs only a single denoising step to directly reconstruct the target-free background from the noisy observation. Although one-step denoising can greatly improve inference speed, it tends to produce overly smooth images and may lose fine details, which can lead to false alarms or mislead the subsequent detection stage. Consequently, we further introduce a frequency-guided perception loss that enforces consistency between the reconstructed image and the ground-truth background in the highfrequency domain. This constraint helps preserve background details such as edges and textures, while also strengthening the supervision signal in target regions.

![](images/33443352593c02cfc75aa52bcdb78c97b8aedee3f24781e1ac86cabced565818.jpg)  
Fig. 1. Illustration of our motivation. From a new perspective of denoising, (a) we treat small targets as noise-like anomalies, and (b-c) then denoise the perturbed image to obtain a target-free background image. (d) The difference between the reconstructed background and the original image can be viewed as informative clues to highlight candidate target regions, promoting the subsequent detection accuracy.

Finally, the reconstructed object-free background image is concatenated with the original input image along the channel dimension and fed into the detection network. This dualchannel input provides explicit background prior information, enabling the detector to more accurately localize small targets by comparing the differences between the two images. Importantly, DiffIRSTD is designed as a plug-and-play framework that can be integrated with existing infrared small target detection models without increasing the computational complexity of the detection network, and it can be flexibly adapted to different deployment constraints. In summary, our contributions can be organized as follows,

• We propose a novel diffusion-based framework for infrared small target detection, which offers a new perspective for denoising. We treat the small targets as a part of noise perturbation and use the diffusion reverse process to denoise them, thereby enhancing the contrast between the foreground and background.

• To improve the efficiency of diffusion sampling, we present a one-step denoising strategy and additionally introduce a frequency-guided perceptual loss. The constraints from the high-frequency domain can effectively guide the denoising process to preserve the background details while focusing on the reconstruction quality of the target area.

• Extensive experiments on three IRSTD benchmarks demonstrate that our proposed method outperforms stateof-the-art approaches, achieving superior detection per-

formance.

## II. RELATED WORK

## A. Infrared Small Target Detection

In the field of deep learning-based IRSTD methods, the research utilizing CNNs has mainly focused on structural fine-tuning and loss function optimization to break through the bottleneck that small target features are prone to being submerged. DNA-Net [20] proposes a dense nested interaction module to avoid small target loss during deep downsampling. UIU-Net [3] adopts a U-Net nested structure to strengthen multi-scale information retention. ISNet [5] emphasizes the crucial role of target shape information by designing an edge block. MSHNet [4] proposes a scale and location-sensitive loss to improve the model’s discriminative ability for targets of different scales and spatial positions. Transformer-based methods like SCTransNet [6] deploy spatial-channel cross Transformer blocks at the skip connections of U-Net to bridge the semantic gap between the encoder and decoder. RKformer [8] introduces ViTs to balance local details and global dependencies. DATransNet [7] proposes a DATrans to accurately extract target edge gradients. Recently, research on IRSTD has been extended to Mamba and SAM to enrich feature representations. MiM-ISTD [9] proposes a ”Mambain-Mamba” nested design, which achieves a leap in accuracy while maintaining linear computational overhead. IR-Mamba [10] introduces PDMamba to enhance target contour perception. In the SAM-based research, IRSAM [12] is the first work to adapt SAM to the IRSTD task, utilizing WPMD and a GAD to finely reconstruct masks. Meanwhile, SAM-SPL [13] leverages the general semantic knowledge of SAM to propose a unified self-prompt learning framework.

Different from the aforementioned architectures that focus on the explicit learning of target features, some low-rank decomposition studies [21]–[25] return to the physical essence of infrared imaging and formulate infrared small target detection as a background-target decoupling problem. Given an infrared image (I), the scene is typically modeled as,

$$
\mathbf { I } = \mathbf { B } + \mathbf { T } + \mathbf { N }\tag{1}
$$

where B, T, and N denote the background, target, and noise components, respectively. These methods estimate the background component via the deep model $\mathcal { F } ( * )$ and subsequently detect targets through residual extraction,

$$
\hat { \mathbf { T } } = \mathbf { I } - \hat { \mathbf { B } } = \mathbf { I } - \mathcal { F } ( \mathbf { I } )\tag{2}
$$

For example, RPCANet [24] unfolds the iterative solution of Robust Principal Component Analysis (RPCA) into a network, combining low-rank priors with deep features to improve interpretability. LRRNet [25] adopts a compression-reconstructionsubtraction paradigm, implicitly learning and generating lowrank background representations with structure-aware capabilities in a data-driven manner. Although LRRNet and our work both involve background estimation, the motivations and mechanisms are fundamentally different. LRRNet assumes that targets in the infrared image are sparse components over a low-rank background and performs target detection through low-rank decomposition and residual extraction. In contrast, our method formulates target removal as a generative denoising problem, where diffusion models can be leveraged to learn the underlying background distribution and generate a target-free image. The generated background is not used to directly locate targets but is instead provided as complementary contextual information to enhance target perception within the detector. Briefly, LRRNet attempts to suppress the background, whereas our method attempts to understand the background. Furthermore, our generated background serves as an auxiliary cue rather than a detection result, enabling the proposed framework to be seamlessly integrated into various existing detectors in a plug-and-play manner.

## B. Diffusion-based Object Detection Models

As a class of generative models, Diffusion Models have demonstrated powerful capabilities in the field of computer vision. Their applications have transcended image generation and gradually expanded to discriminative tasks such as detection and segmentation. The pioneering work that first introduced diffusion models into object detection is DiffusionDet [26], which regards the generation process of target bounding boxes as a denoising process from a random noise distribution to real target boxes. DiffusionAD [27] proposes a noise normalization paradigm and a single-step denoising mechanism for anomaly detection. It achieves unbiased reconstruction and accurate localization of abnormal regions through a dual-branch network. Although both DiffusionAD and our method employ diffusion-based reconstruction, their underlying assumptions and objectives are fundamentally different. DiffusionAD assumes that anomalies lie outside the normal data distribution and reconstructs normal samples by removing semantic defects. In contrast, infrared small targets are not semantic anomalies but extremely weak signals embedded in cluttered backgrounds and sensor noise, often exhibiting substantial distribution overlap with the background. Consequently, our technical designs aim to efficiently purify the backgrounds to enhance target-background contrast rather than eliminating semantic abnormalities.

In recent years, diffusion models have been gradually introduced into IRSTD tasks, and relevant research has mainly developed along two paths: data augmentation and detection architecture. In the data augmentation paradigm, aiming at the lack of realism in traditional stitching methods, Diff-Mosaic [28] designs a pixel coordination network combined with latent diffusion priors for resampling. Gaussian Agnostic Representation Learning [29] addresses the challenge of data scarcity by proposing a Gaussian-driven non-uniform quantization strategy and a two-stage diffusion generation framework. In the detection and feature modeling paradigm, DCFR-Net [30] innovatively embeds implicit neural representation into conditional diffusion models, learning continuous high-resolution feature representations in a self-supervised manner. ISTD-diff [31] models IRSTD as a conditional mask generation task, directly outputting high-precision target masks through an iterative denoising process. Unlike these methods that use diffusion models to generate images with various foreground and background, our method treats targets as noiselike signals and utilizes diffusion models to remove them, thereby generating target-free background images for the subsequent detection. In addition, our reconstruction and detection networks are not tightly coupled but are independently separated. Therefore, it can be easily adapted to different detectors according to the requirements of deployment scenarios.

## III. METHODOLOGY

## A. Overview

As aforesaid, the key idea of our DiffIRSTD is to enhance the contrast of small targets in infrared images using generative models, thereby improving the detector’s ability to perceive them. Unlike previous approaches that use diffusion models to either generate background-free target masks or as the detector to localize targets, we take a different perspective of treating small targets as noise-like perturbations in infrared images $( \mathbf { T } \ \approx \ \mathbf { N } )$ . Based on this assumption, we thus exploit the denoising property of diffusion models to learn the conditional distribution of target-free backgrounds,

$$
\mathbf { B } _ { \mathrm { c l e a n } } \sim p ( \mathbf { B } | \mathbf { I } )\tag{3}
$$

where $\mathbf { B } _ { \mathrm { c l e a n } }$ denotes the generated target-free background conditioned on the observed image (I). The generated background is not used for residual computation, but is instead provided as contextual information to a detector (compared with Eq. 2),

$$
\hat { \mathbf { T } } = \mathcal { D } ( \mathbf { I } , \mathbf { B } _ { \mathrm { c l e a n } } )\tag{4}
$$

where D(·) denotes the target detector. Consequently, existing methods regard the background as interference to be suppressed, whereas our method explicitly models the background distribution and exploits the generated background as informative scene context. By comparing the original image with the reconstructed target-free image, the detector can more easily identify the location and precise boundaries of the target. Figure 2 illustrates the overall architecture of our proposed diffusion-based framework for infrared small target detection.

In the remainder of this section, we first explain the basic principles of diffusion models, including the diffusion and denoising processes (in Sec. III-B). Next, we describe how diffusion models are utilized to denoise infrared images in Sec. III-C, focusing on our one-step denoising strategy and the proposed frequency-guided perceptual loss, which together improve both the efficiency and quality of reconstructing background images without targets. We then discuss how to leverage the generated target-free infrared images to strengthen the detector’s ability to perceive small targets (see Sec. III-D). Subsequently, we present the workflow of obtaining synthetic background images using morphological operations, which serve as pseudo-labels for training the diffusion model (in Sec. III-E). Finally, we detail the training and inference procedures of the proposed DiffIRSTD framework in Sec. III-F.

## B. Preliminary

Denoising Diffusion Probabilistic Models (DDPM) [32] are a class of latent-variable generative models that define the data generation process through two Markov chains: a forward diffusion process and a reverse denoising process.

![](images/1a92a031944a44339d3a508a85bbfa0f549d9f0eb396d636a996f8c68fb29b55.jpg)  
Fig. 2. Schematic of our proposed DiffIRSTD, which consists of a reconstruction network and a detection network. The reconstruction network is designed to generate a target-free background image x<sub>B</sub> from the input infrared image x. We further incorporate a one-step denoising strategy together with a frequency-guided perceptual loss. This design not only accelerates the sampling process but also preserves the denoising quality of the reconstructed background. The detection network takes as input a dual-channel representation formed by concatenating the original infrared image and the reconstructed image. $\boldsymbol { \mathrm { B y } }$ leveraging the explicit background prior, the network can effectively perceive target regions and predict the corresponding target mask.

Forward Diffusion Process. In the forward process, Gaussian noise is gradually added to the data. Given a real data sample $x _ { 0 } \ \sim \ q ( x _ { 0 } )$ , a Markov chain is defined in which noise is incrementally injected over $T$ time steps,

$$
q ( x _ { t } | x _ { t - 1 } ) = \mathcal { N } ( x _ { t } ; \sqrt { 1 - \beta _ { t } } x _ { t - 1 } , \beta _ { t } \mathbf { I } )\tag{5}
$$

where $\beta _ { t } ~ \in ~ ( 0 , 1 )$ denotes a predefined noise schedule that controls the noise level at each step. By applying the reparameterization trick, a noisy sample at any timestep t can be directly obtained from the original data $x _ { 0 }$

$$
x _ { t } = \sqrt { \bar { \alpha } _ { t } } x _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon\tag{6}
$$

where $\epsilon \sim \mathcal { N } ( 0 , \mathbf { I } ) , \alpha _ { t } = 1 - \beta _ { t }$ , and $\begin{array} { r } { \bar { \alpha } _ { t } = \prod _ { i = 1 } ^ { t } \alpha _ { i } . } \end{array}$

Reverse Denoising Process. The reverse process aims to recover clean data from pure Gaussian noise $x _ { T } \sim$ $\mathcal { N } ( 0 , \bf { I } )$ through iterative denoising. A parameterized model $p _ { \theta } ( x _ { t - 1 } | x _ { t } )$ is trained to approximate the posterior distribution,

$$
p _ { \theta } ( x _ { t - 1 } | x _ { t } ) = \mathcal { N } ( x _ { t - 1 } ; \mu _ { \theta } ( x _ { t } , t ) , \Sigma _ { \theta } ( x _ { t } , t ) )\tag{7}
$$

In practice, the variance is often fixed as $\Sigma _ { \theta } ( x _ { t } , t ) = \sigma _ { t } ^ { 2 } \mathbf { I } .$ and a neural network $\epsilon _ { \theta } ( x _ { t } , t )$ is trained to predict the noise component ϵ added at step t. The denoising step from $x _ { t }$ to $x _ { t - 1 }$ can be computed directly as,

$$
x _ { t - 1 } = \frac { 1 } { \sqrt { \alpha _ { t } } } ( x _ { t } - \frac { 1 - \alpha _ { t } } { \sqrt { 1 - \bar { \alpha } _ { t } } } \epsilon _ { \theta } ( x _ { t } , t ) ) + \sigma _ { t } z\tag{8}
$$

where $z \sim \mathcal { N } ( 0 , \mathbf { I } )$ is standard Gaussian noise. The training objective of diffusion models is to minimize the mean squared error between the predicted and ground-truth noise,

$$
\mathcal { L } _ { m s e } = \mathbb { E } _ { t , x _ { 0 } , \epsilon } \left[ \| \epsilon - \epsilon _ { \theta } ( x _ { t } , t ) \| ^ { 2 } \right]\tag{9}
$$

## C. Reconstruction Network

Revisiting the imaging characteristics of infrared small targets, we observe that although infrared targets are small and lack distinct structural features, they often exhibit subtle local intensity deviations from their surrounding background. These deviations may be visually difficult to perceive due to complex scenes, illumination conditions, and sensor noise. From a statistical perspective, infrared small targets can therefore be regarded as noise-like signals relative to the smooth background. Generating a target-free image is thus analogous to the denoising process of a diffusion model, which progressively removes noise (e.g., target) to recover a clean signal (e.g., background). Therefore, we adopt a diffusion model as the reconstruction network to generate the background image.

Given an infrared image $\boldsymbol { x } ~ \in ~ \mathbb { R } ^ { H \times W }$ containing small targets, we first apply Eq. 6 to gradually add Gaussian noise, so that after t steps, the image becomes increasingly corrupted and the targets are overwhelmed by noise. The resulting noisy image is then fed into the reverse diffusion process, which iteratively denoises the observation according to Eq. 8. During this reverse process, the model is encouraged to reconstruct a clean background image while suppressing target-related “noise”. As a result, we obtain a reconstructed image $x _ { B } \in \mathbb { R } ^ { H \times W }$ that approximates the scene without targets.

However, according to Eq. 8, the reverse diffusion process progressively denoises the perturbed image from $x _ { t }$ to x<sub>0</sub> through multiple iterative steps. Each iteration corresponds to a single denoising transition, and performing all t steps during inference leads to substantial computational overhead. This iterative sampling procedure significantly increases the inference time and limits the practicality of diffusion models in real-time infrared small target detection. Is it possible to directly predict the background image $x _ { 0 }$ from a noisy observation $x _ { t }$ in a single step?

One-step Denoising Strategy. During the theoretical derivation of diffusion models [32], it has been proven that the clean image $x _ { 0 }$ can be estimated directly from a noisy sample $x _ { t }$ by rearranging the forward formulation Eq. 8. However, this direct estimation generally produces images of lower quality compared to multi-step iterative sampling. The primary reason is that when $x _ { t }$ approaches pure Gaussian noise, the prediction errors in ϵ may be amplified, leading to distortions and artifacts. Differently, our purpose is not to synthesize images from random noise, but to remove small target “noise”. Since the background of infrared images is typically smooth, recovering a target-free background image from a noisy observation is easier than reconstructing a natural image from pure noise. Therefore, given an input image x and its noisy version $x _ { t }$ obtained at diffusion step t, we directly employ the one-step denoising process to estimate the background image,

$$
x _ { 0 } = \frac { 1 } { \sqrt { \bar { \alpha } _ { t } } } ( x _ { t } - \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon _ { \theta } ( x _ { t } , t ) ) .\tag{10}
$$

where $x _ { B } = x _ { 0 }$ . By skipping intermediate iterative steps, the one-step denoising strategy substantially accelerates inference and reduces computational cost. Intuitively, if t is too large, one-step denoising encounters the above limitation again and may degrade image quality. Conversely, a small t makes the noise-like target signals difficult to remove by one step. Therefore, selecting an appropriate noise level balances the target denoising and background fidelity, which will be discussed in the experimental section.

Frequency-guided Perceptual Loss. Infrared small targets typically occupy only a tiny fraction of the image area, which leads to highly imbalanced supervision during the reconstruction process. Moreover, even minor reconstruction errors in the background may introduce spurious high-frequency artifacts, which can mislead the subsequent detection network. These high-frequency components, especially edges and textures, provide important cues for distinguishing true targets from complex background clutter.

To explicitly preserve these structural details, we introduce a high-frequency perceptual constraint in the frequency domain. Specifically, the predicted background image $x _ { B }$ and the ground-truth background $\hat { x } _ { B }$ are first transformed into the frequency domain using two-dimensional discrete Fourier transform (2D DFT),

$$
{ \mathcal { F } } ( X ) ( u , v ) = \sum _ { m = 0 } ^ { H - 1 } \sum _ { n = 0 } ^ { W - 1 } X ( m , n ) e ^ { - j 2 \pi \left( { \frac { u m } { H } } + { \frac { v n } { W } } \right) }\tag{11}
$$

where $X \in [ x _ { B } , \hat { x } _ { B } ]$ , and $( u , v )$ denote the frequency-domain coordinates. The zero-frequency component is then shifted to the center of the spectrum to obtain $\mathcal { F } _ { \mathrm { s h i f t } } ( X )$ ). To further

reduce the dynamic range of spectral magnitudes, we compute the logarithmic amplitude spectrum and normalize it spatially,

$$
\hat { \mathcal { F } } _ { \mathrm { s h i f t } } ( X ) = \mathrm { N o r m } \left[ \log \left( 1 + | \mathcal { F } _ { \mathrm { s h i f t } } ( X ) | \right) \right]\tag{12}
$$

Next, a binary high-pass mask $M ( u , v ) \ \in \ \{ 0 , 1 \} ^ { H \times W }$ is constructed to retain only frequency components whose radial distance from the spectrum center exceeds a predefined cutoff frequency $r _ { c } .$ . Thereby, the high-frequency representation $\mathcal H ( X )$ can be obtained by,

$$
M ( u , v ) = \left\{ \begin{array} { l l } { 1 , } & { \sqrt { ( u - u _ { c } ) ^ { 2 } + ( v - v _ { c } ) ^ { 2 } } > r _ { c } } \\ { 0 , } & { \mathrm { o t h e r w i s e } } \end{array} \right.\tag{13}
$$

$$
{ \mathcal { H } } ( X ) = { \hat { \mathcal { F } } } _ { \mathrm { s h i f t } } ( X ) \cdot M\tag{14}
$$

where $( u _ { c } , v _ { c } )$ means the coordinates of spectrum center. Finally, the high-frequency perceptual loss is defined as the mean squared error between the masked spectra of the reconstructed and ground-truth backgrounds,

$$
\mathcal { L } _ { h f } = | | \mathcal { H } ( x _ { B } ) - \mathcal { H } ( \hat { x } _ { B } ) | | ^ { 2 }\tag{15}
$$

This frequency-guided perceptual loss encourages the reconstruction network to pay closer attention to high-frequency discrepancies introduced by target regions while simultaneously preserving intrinsic background structures during onestep denoising. As a result, the model is able to suppress targetinduced noise without sacrificing the sharpness and texture of the reconstructed background, leading to higher-quality targetfree images.

## D. Detection Network

Similar to previous approaches [2], [3], [6], [10], the objective of our detection network is to accurately predict the location mask of small targets. But, differently, benefiting from the background prior provided by our reconstruction network, the detection network can additionally receive the corresponding target-free background image, so as to highlight target regions through comparison analysis.

Specifically, we first take as input the input infrared image $x \in \mathbb { R } ^ { H \times W }$ and the reconstructed background image $x _ { B } \in$ R $\updownarrow ^ { H \times W }$ , and concatenate them along the channel dimension to form a dual-channel input,

$$
\boldsymbol { x } _ { d u a l } = \operatorname { C o n c a t } ( \boldsymbol { x } , \boldsymbol { x } _ { B } ) \in \mathbb { R } ^ { 2 \times H \times W } .\tag{16}
$$

This dual-channel representation is thus fed into the detection network to localize target regions by learning the discrepancy between the two channels.

The detection network follows a standard U-Net architecture [33], consisting of an encoder, a decoder, and skip connections. The encoder extracts multi-scale semantic features through progressive down-sampling, while the decoder gradually restores spatial resolution via up-sampling. Skip connections fuse features from corresponding encoder and decoder stages, helping preserve fine-grained spatial details that are important for small target localization.

To further enhance feature discriminability, we adopt the channel attention module, which is inserted after each encoder block. The attention module aggregates contextual information along spatial and channel dimensions, and then calculates the channel weights to rescale the importance of feature maps. Formally, given an input feature map $f ^ { i n } \ \in \ \mathbb { R } ^ { c \times h \times w }$ , the attention-enhanced output is computed as,

![](images/0dff58a435803639c506bd8f4ac310ded2c2c59253b8c50d0c2681860a3a9e3a.jpg)  
Fig. 3. The workflow of acquiring the pseudo-label target-free background images. Best viewed in color and zoom in.

$$
f ^ { \mathrm { o u t } } = f ^ { \mathrm { i n } } \cdot \sigma \left( \mathrm { M L P } ( \mathrm { G M P } ( f ^ { \mathrm { i n } } ) + \mathrm { G A P } ( f ^ { \mathrm { i n } } ) ) \right)\tag{17}
$$

where $\sigma$ is the sigmoid activation, MLP means Multilayer Perceptron, and GMP and GAP represent global max pooling and global average pooling, respectively. In this way, the network can adaptively emphasize channels that are more relevant to small targets while suppressing those dominated by background interference. The refined feature maps are then forwarded to the next encoder stage and to the corresponding skip connections. The decoder finally gradually interprets and outputs the predicted target mask.

Discussion. In this paper, we intentionally adopt a simple U-Net as the baseline detection network. This design choice allows us to more clearly evaluate the effectiveness of the proposed reconstruction network and directly support our motivation. Importantly, our DiffIRSTD framework does not depend on any specific detector architecture. The reconstruction network is designed as a plug-and-play component that generates a clean background image. This reconstructed background can be concatenated with the original infrared image to form a dual-channel representation, which can be readily used by existing infrared small target detection models. As shown in Tab. IX, integrating the reconstruction network with various state-of-the-art detectors consistently improves their performance. Furthermore, different from previous diffusion-based approaches [17]–[19], the reconstruction and detection networks in our framework are loosely connected. This decoupled design provides greater flexibility in practical deployment. Depending on computational constraints, the detection network can be replaced with either lighter or more powerful models without modifying the reconstruction network. Alternatively, the reconstruction network can be deployed independently on the cloud as a preprocessing step during inference.

## E. Background Acquisition Strategy

Existing infrared small target datasets provide only infrared images and their corresponding target masks, while groundtruth background images are not available. However, such background images are necessary to supervise the reconstruction process described in Sec. III-C. To address this issue, we propose an inpainting-based background acquisition strategy that removes target regions using the provided masks and fills them to synthesize high-quality background images for training. The workflow of this strategy is illustrated in Fig. 3.

More specifically, given an infrared image x and its target mask $\hat { M }$ , we first apply a threshold of 0.5 to ensure that the mask is strictly binary. Next, a morphological dilation operation with a 5×5 kernel is applied to the mask to slightly enlarge the target region. This step ensures that the dilated mask $\hat { M } _ { d i l a t e d }$ fully covers the target boundaries and nearby pixels, preventing residual target artifacts after inpainting. Finally, we employ the OpenCV implementation of the Telea inpainting algorithm [34] to fill the masked regions, with the inpainting radius set to 3 pixels. The resulting image is thus used as the pseudo-label background image, denoted by ${ \hat { x } } _ { B }$ . This algorithm preserves structural continuity and texture consistency in the surrounding background, making it suitable for the reconstruction of small regions.

## F. Training and Inference

Training. Our framework is trained in an end-to-end manner. For the reconstruction network, it learns to generate highquality backgrounds under the joint supervisions of the pixellevel reconstruction loss and frequency-guided perceptual loss,

$$
\mathcal { L } _ { r e c o n } = \mathcal { L } _ { m s e } ( x _ { B } , \hat { x } _ { B } ) + \lambda \cdot \mathcal { L } _ { h f } ( x _ { B } , \hat { x } _ { B } )\tag{18}
$$

where λ is the balancing coefficient to control the importance of the frequency-guided perceptual loss. $\mathcal { L } _ { m s e }$ denotes the mean squared error between the predicted and ground-truth images, and we empirically find it is better than the noise supervision in Eq. 9. Here, the high-frequency loss $\mathcal { L } _ { h f }$ plays a more similar role to that of the classifier-based guidance, guiding the denoising process to generate high-quality targetfree background images.

Meanwhile, the detection network learns to locate targets and output segmentation masks under the supervision of SoftIoU Loss and BCE Loss,

$$
\mathcal { L } _ { d e t e c t } = \mathcal { L } _ { s o f t i o u } ( M , \hat { M } ) + \mathcal { L } _ { b c e } ( M , \hat { M } )\tag{19}
$$

where M and $\hat { M }$ are the predicted and ground-truth masks. The SoftIoU loss $\mathcal { L } _ { s o f t i o u }$ optimizes the intersection over union between target regions, while the BCE loss $\mathcal { L } _ { b c e }$ ensures the pixel-level classification accuracy. The collaboration between two functions improves the quality of target localization. Inference. During inference, the input image x first passes through the reconstruction network to get the denoised targetfree image $x _ { B }$ , and then both of them are fed into the detection network to predict the target mask M.

## IV. EXPERIMENTS

## A. Experimental Setups

Datasets. Three public datasets of IRSTD-1K [47], NUDT-SIRST [20], and NUAA-SIRST [48] are used for evaluation. Following common practice in prior work [20], [48], we split

Implementation Details. Our framework is implemented on the PyTorch platform using one NVIDIA GeForce RTX 4090 GPU. The reconstruction network is based on the U-Net architecture [27], [49]. During training, all input images are resized to 256 × 256 with the augmentation of random cropping, padding, and normalization. Both reconstruction and detection models are jointly trained with a batch size of 8 for a total of 800 epochs. We use the Adam optimizer with an initial learning rate of $1 \times 1 0 ^ { - 4 }$ , and the learning rate is adjusted using a cosine annealing schedule. We evaluate models using three widely-used metrics to provide a balanced assessment, where Intersection over Union $( \mathrm { I o U } \times 1 0 ^ { - 2 } )$ evaluates geometric fidelity, detection probability $( F _ { d } \times 1 0 ^ { - 2 } )$ ensures high recall of targets, and false-alarm rate $( F _ { a } \times 1 0 ^ { - 6 } )$ penalizes false activations. These indicators jointly quantify localization fidelity, recall, and specificity, offering a holistic assessment of detection accuracy, precision, and reliability.

TABLE I  
COMPARISON WITH EXISTING IRSTD APPROACHES ON IRSTD-1K, NUDT-SIRST, AND NUAA-SIRST DATASETS. THE EVALUATION METRICS ARE IOU $( 1 0 ^ { - 2 } ) , P _ { d } ( 1 0 ^ { - 2 } )$ , AND $F _ { a } \ ( 1 0 ^ { - 6 } )$ . THE FIRST AND SECOND BEST RESULTS ARE HIGHLIGHTED IN BOLD AND UNDERLINED, RESPECTIVELY. † DENOTES METHODS THAT USE DIFFUSION MODELS FOR EITHER DATA AUGMENTATION, FEATURE LEARNING, OR DIFFUSION-BASED DETECTORS.
<table><tr><td rowspan="2">METHODS</td><td rowspan="2">BACKBONE</td><td colspan="3">IRSTD-1K</td><td colspan="3">NUDT-SIRST</td><td colspan="3">NUAA-SIRST</td></tr><tr><td>IoU ↑</td><td> $\mathrm { P } _ { d } \uparrow$ </td><td> $\operatorname { F } _ { a \mathrm { ~ \downarrow ~ } }$ </td><td>IoU ↑</td><td> $\mathrm { P } _ { d } \uparrow$ </td><td> $\operatorname { F } _ { a \mathrm { ~ \downarrow ~ } }$ </td><td>IoU ↑</td><td> $\mathrm { P } _ { d } \uparrow$ </td><td> $\operatorname { F } _ { a \mathrm { ~ \downarrow ~ } }$ </td></tr><tr><td>Top-Hat [21]</td><td>Traditional</td><td>10.06</td><td>75.11</td><td>1432</td><td>20.72</td><td>78.41</td><td>166.7</td><td>7.143</td><td>79.84</td><td>1012</td></tr><tr><td>IPI [22]</td><td>Traditional</td><td>27.92</td><td>81.37</td><td>16.18</td><td>17.76</td><td>74.49</td><td>41.23</td><td>25.67</td><td>84.63</td><td>16.67</td></tr><tr><td>PSTNN [23]</td><td>Traditional</td><td>24.57</td><td>71.99</td><td>35.26</td><td>14.85</td><td>66.13</td><td>44.17</td><td>30.30</td><td>72.80</td><td>48.99</td></tr><tr><td>MSLSTIPT [35]</td><td>Traditional</td><td>11.43</td><td>79.03</td><td>1524</td><td>8.342</td><td>47.40</td><td>888.1</td><td>10.30</td><td>82.13</td><td>1131</td></tr><tr><td>MTU-Net [36]</td><td>CNN + Trans.</td><td>66.11</td><td>93.27</td><td>36.80</td><td>74.85</td><td>93.97</td><td>46.95</td><td>74.78</td><td>93.54</td><td>22.36</td></tr><tr><td>SCTransNet [6]</td><td>CNN + Trans.</td><td>68.03</td><td>93.27</td><td>10.74</td><td>94.09</td><td>98.62</td><td>4.290</td><td>77.50</td><td>96.95</td><td>13.92</td></tr><tr><td>HSTNet [37]</td><td>CNN + Trans.</td><td>69.79</td><td>91.16</td><td>10.70</td><td></td><td></td><td></td><td>78.14</td><td>100</td><td>14.72</td></tr><tr><td>ISTD-Diff† [31]</td><td>Diffusion</td><td></td><td></td><td></td><td></td><td></td><td></td><td>76.09</td><td>97.25</td><td>11.91</td></tr><tr><td>DCDNet† [17]</td><td>Diffusion</td><td></td><td></td><td></td><td>90.34</td><td>98.62</td><td></td><td></td><td></td><td></td></tr><tr><td>IRMamba [10]</td><td>Mamba</td><td>70.04</td><td>95.81</td><td>5.920</td><td>95.18</td><td>99.26</td><td>1.309</td><td>80.20</td><td>99.36</td><td>0.887</td></tr><tr><td>FC3-Net [38]</td><td>CNN</td><td>65.07</td><td>91.54</td><td>15.55</td><td>78.56</td><td>93.86</td><td>23.92</td><td>72.44</td><td>98.14</td><td>10.85</td></tr><tr><td>DNA-Net [39]</td><td>CNN</td><td>65.90</td><td>90.91</td><td>12.24</td><td>88.19</td><td>98.83</td><td>9.000</td><td>75.80</td><td>95.82</td><td>8.780</td></tr><tr><td>UIU-Net [3]</td><td>CNN</td><td>66.15</td><td>93.98</td><td>22.07</td><td>93.48</td><td>98.31</td><td>7.790</td><td>76.91</td><td>95.82</td><td>14.13</td></tr><tr><td>ISTDU [40]</td><td>CNN</td><td>66.36</td><td>93.60</td><td>53.10</td><td>89.55</td><td>97.67</td><td>13.44</td><td>75.52</td><td>96.58</td><td>14.54</td></tr><tr><td>AGPCNet [41]</td><td>CNN</td><td>66.29</td><td>92.83</td><td>13.12</td><td>88.87</td><td>97.20</td><td>10.02</td><td>75.69</td><td>96.48</td><td>14.99</td></tr><tr><td>Dim2Clear [42]</td><td>CNN</td><td>66.34</td><td>93.75</td><td>20.93</td><td>81.37</td><td>96.23</td><td>9.170</td><td>77.29</td><td>99.10</td><td>6.72</td></tr><tr><td>MMLNet [43]</td><td>CNN</td><td>67.21</td><td>94.28</td><td>14.00</td><td>81.81</td><td>98.43</td><td>11.77</td><td>78.71</td><td>98.88</td><td>25.71</td></tr><tr><td>PConv [44]</td><td>CNN</td><td>67.93</td><td>94.22</td><td>9.19</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>L2SKNet [45]</td><td>CNN</td><td>67.81</td><td>90.24</td><td>17.46</td><td>93.58</td><td>97.57</td><td>5.330</td><td>73.43</td><td>98.17</td><td>20.82</td></tr><tr><td>Diff-Mosaic† [28]</td><td>CNN</td><td></td><td></td><td></td><td>91.18</td><td>99.47</td><td>1.92</td><td></td><td></td><td></td></tr><tr><td>DCFRNet† [30]</td><td>CNN</td><td>65.41</td><td>93.60</td><td>7.340</td><td>86.93</td><td>98.26</td><td>2.48</td><td>76.23</td><td>99.08</td><td>6.52</td></tr><tr><td>FAA-Net [46]</td><td>CNN</td><td>68.63</td><td>93.88</td><td>14.65</td><td>93.53</td><td>99.15</td><td>2.37</td><td>77.57</td><td>100</td><td>17.39</td></tr><tr><td>Upper Bound</td><td></td><td>94.21</td><td>98.99</td><td>0.000</td><td>97.20</td><td>100.0</td><td>0.48</td><td>85.96</td><td>99.84</td><td>3.11</td></tr><tr><td>U-Net Baseline</td><td>CNN</td><td>67.13</td><td>93.93</td><td>26.64</td><td>93.97</td><td>99.04</td><td>3.44</td><td>78.35</td><td>94.29</td><td>15.36</td></tr><tr><td>DiffIRSTD (Ours)</td><td>CNN</td><td>72.13</td><td>91.91</td><td>13.64</td><td>95.91</td><td>99.49</td><td>1.72</td><td>80.73</td><td>96.95</td><td>7.55</td></tr></table>

both NUDT-SIRST and NUAA-SIRST evenly into training and testing sets (5:5). For IRSTD-1K, we use 80% of the samples for training and the remaining for testing (8:2), same as the official protocol for fair comparisons.

## B. Comparisons with the State-of-the-arts

We first compare our method with previous state-of-theart approaches on the IRSTD-1K, NUDT-SIRST, and NUAA-SIRST datasets. All results of these competitors are taken from the reported results in the published papers. As shown in Tab. I, our DiffIRSTD achieves consistent improvements across all three datasets, demonstrating its effectiveness for infrared small target detection. Specifically, (1) our method attains the best IoU scores, reaching 72.13 on IRSTD-1K, 95.91 on NUDT-SIRST, and 80.73 on NUAA-SIRST, indicating improved localization accuracy. In terms of false-alarm rate $( F _ { a } ) .$ , our method achieves competitive performance, particularly on NUDT-SIRST, while maintaining a reasonable balance between detection accuracy and false positives across datasets. (2) Compared with existing diffusion-based approaches (e.g., ISTD-Diff, DCDNet, Diff-Mosaic, and DCFRNet), we beat them by a clear margin across datasets, suggesting that the proposed reconstruction strategy provides more effective background priors for detection. (3) In addition, compared with hybrid backbone methods such as IRMamba and HSTNet, our method also achieves slightly higher IoU on all three datasets. Although IRMamba reports lower $F _ { a }$ in some cases, we offer a competitive trade-off between localization accuracy and falsealarm suppression, despite using a relatively simple CNNbased detector. (4) The “Upper Bound” results are obtained by bypassing the generation module and directly training the detector with the pseudo-label background image $\hat { x } _ { B }$ This setting provides an estimate of the maximum achievable performance of the detection stage under ideal background reconstruction. As shown in Tab. I, the upper bound substantially outperforms existing methods and achieves remarkably strong results on IRSTD-1K and NUDT-SIRST. This observation strongly validates the effectiveness of our motivation and the quality of our background acquisition strategy. Moreover, the remaining gap between DiffIRSTD and the upper bound suggests considerable room for improvement, indicating that more accurate target-free background generation could further enhance detection performance.

![](images/65d3f7eb60127bf4a5b8210bea44fc9ad27afa21211bbcbe8a6a99a5c49cf362.jpg)  
Fig. 4. Visualization of different IRSTD methods. The green boxes indicate correctly detected targets, red boxes denote false alarms, and orange circles represent missed targets. Additionally, dashed boxes show close-up views of the targets for clearer visualization.

Figure 4 presents visualization comparisons of the inference results between our method and other state-of-theart approaches, such as DNANet [20], SCTransNet [6], L2SKNet [45], MMLNet [43]. As illustrated by the objects highlighted with green bounding boxes in the first and second rows, the predictions generated by our method exhibit superior alignment with the ground truth, demonstrating its enhanced capacity for precise boundary delineation. Furthermore, across all five visualized samples, the comparative methods are prone to false alarms and missed detections, whereas our approach effectively suppresses these prevalent segmentation errors.

TABLE II  
ABLATION STUDY ON KEY COMPONENTS OF DIFFIRSTD.
<table><tr><td></td><td>Detection Reconstruction Frequency-guided</td><td></td><td>IRSTD-1K</td></tr><tr><td>Network</td><td>Network</td><td>Perceptual Loss</td><td>IoU↑Pd↑ Fa↓</td></tr><tr><td>√</td><td></td><td></td><td>67.1393.93 26.64</td></tr><tr><td>√ √</td><td>√ √</td><td></td><td>69.9992.92 25.92</td></tr><tr><td></td><td></td><td></td><td>72.1391.9113.64</td></tr></table>

## C. Ablation Study

Next, we provide in-depth studies to dissect the efficacy of our designs. Without loss of generality, we conduct ablation studies on IRSTD-1K and use the same hyperparameters for other datasets. More quantitative analysis and qualitative results can be found in the supplementary materials.

Efficacy of Designs in Reconstruction Network. Table II presents the results with different components. Our baseline is to solely activate the detection network with U-Net architecture. It obtains an IoU of 67.13 with a relatively high falsealarm rate of 26.64. After incorporating with the reconstruction network, the IoU increases to 69.99, indicating that the reconstructed background provides useful prior information for enhancing target localization. Meanwhile, a slight reduction in $F _ { a }$ is also observed, suggesting improved suppression of background interference. Finally, we attach the proposed frequency-guided perceptual loss, and the performance is significantly improved. It strongly demonstrates the efficacy of our design, allowing the detector to locate target boundaries more accurately while avoiding false detections caused by blurred backgrounds.

TABLE III  
ABLATION STUDY OF RECONSTRUCTION LOSS ON IRSTD-1K.
<table><tr><td>LOSS VARIANTS</td><td>IoU ↑</td><td> $\mathrm { P } _ { d } \uparrow$ </td><td> $\operatorname { F } _ { a \downarrow }$ </td></tr><tr><td> $\mathcal { L } _ { m s e } ( \epsilon _ { \theta } , \epsilon ) ~ ( \mathrm { E q . } ~ 9 )$ </td><td>67.14</td><td>91.24</td><td>32.05</td></tr><tr><td> $\mathcal { L } _ { m s e } ( x _ { B } , \hat { x } _ { B } ) \ ( \mathrm { i n \ E q . \ 1 8 } )$ </td><td>69.99</td><td>92.92</td><td>25.92</td></tr><tr><td> $\mathcal { L } _ { m s e } ( x _ { B } , \hat { x } _ { B } ) + \mathcal { L } _ { h f }$  (Eq. 18)</td><td>72.13</td><td>91.91</td><td>13.64</td></tr></table>

TABLE IV

ABLATION STUDY OF DIFFUSION STEP t IN EQ. 10.
<table><tr><td>TIME STEPS</td><td>IoU ↑</td><td> $\mathrm { P } _ { d } \uparrow$ </td><td> $\operatorname { F } _ { a \mathrm { ~ \downarrow ~ } }$ </td></tr><tr><td>30</td><td>72.16</td><td>91.58</td><td>18.95</td></tr><tr><td>50</td><td>72.13</td><td>91.91</td><td>13.64</td></tr><tr><td>70</td><td>70.69</td><td>91.58</td><td>12.63</td></tr><tr><td>90</td><td>71.27</td><td>93.93</td><td>14.45</td></tr><tr><td>120</td><td>68.94</td><td>90.90</td><td>16.13</td></tr></table>

We also investigate in Tab. III the impact of different reconstruction losses on the detection performance. Specifically, we observe that replacing the standard noise supervision of the diffusion model with image supervision can effectively improve the performance. We explain that small targets are treated as slight noise perturbations in the background images, so the image supervision offers a smoothing loss bias, and it is task-aligned with the subsequent detection network.

Analysis of Diffusion Steps t. As discussed in Sec. III-C, the diffusion step t plays an important role in reconstruction quality. We therefore analyze the effect of different time steps in Tab. IV. Two main observations can be drawn: (1) within the range of $3 0 \sim 9 0$ steps, the overall performance variation is relatively small; (2) moderate step values $( e . g . , t = 5 0 \mathrm { o r } 7 0 )$ tend to yield better results. In addition, the qualitative results in Fig. 5 show that as t increases, the removal of targets does not improve significantly, while the background details become progressively blurred. This is because introducing excessive noise makes one-step denoising more prone to distortions. In this paper, we set $t = 5 0$ for all experiments.

Designs of Dual-channel Inputs To effectively exploit the reconstructed background image $x _ { B } ,$ , we adopt a dual-channel input paradigm, where the original image x is combined with auxiliary information derived from $x _ { B }$ . The key idea is that $x _ { B }$ provides a purified background prior, which can implicitly highlight the foreground–background discrepancy and thus enhance the perception of small targets. Therefore, we conduct a series of ablation studies with different input designs, as summarized in Tab. V.

First, among all the two-channel configurations, the simplest design Conca $\mathbf { \Psi } : ( x , x _ { B } )$ achieves the best overall performance, yielding the highest IoU of 72.13 and a competitive false alarm rate. This suggests that directly feeding both the original image and the reconstructed background allows the network to implicitly learn the discrepancy between foreground and background, without requiring manually engineered transformations. Such a design preserves better information and avoids introducing additional distortions.

TABLE V  
ABLATION STUDY OF DUAL-CHANNEL INPUTS ON IRSTD-1K. EXPERIMENTS VERIFY THE EFFECTS OF BACKGROUND INFORMATION UTILIZATION, WITH EVALUATION METRICS INCLUDING $\mathrm { I o U } ( 1 0 ^ { - 2 } ) , P _ { d }$ $( 1 0 ^ { - 2 } )$ , AND $F _ { a } \ ( 1 0 ^ { - 6 } )$ $\mathrm { ^ { * } C o N C A T } ^ { \mathrm { 3 } }$ MEANS THE CONCATENATION OPERATION, AND $\mathbf { \ddot { \rho } } _ { \mathcal { H P } } \mathbf { \vec { \rho } }$ DENOTES THE HIGH-PASS FILTERING OPERATION.
<table><tr><td>IN-CHANNELS</td><td>INPUT DESIGN</td><td>IoU ↑  $\mathrm { P } _ { d } \uparrow$   $\operatorname { F } _ { a \downarrow }$ </td></tr><tr><td>2</td><td> $\mathrm { C o n c a t } ( x , \mathcal { H P } ( x ) )$ </td><td>65.47 89.22 11.06</td></tr><tr><td>2</td><td> $\mathbf { C o n c a t } ( x , x _ { B } )$ </td><td>72.13 91.91 13.64</td></tr><tr><td>2</td><td> $\mathrm { C o n c a t } ( x , | x - x _ { B } | )$ </td><td>70.85 94.61 15.90</td></tr><tr><td>2</td><td> $\mathrm { C o n c a t } ( x , ( x - x _ { B } ) ^ { 2 } )$ </td><td>71.77 90.90 13.98</td></tr><tr><td>3</td><td> $\mathrm { C o n c a t } ( x , x _ { B } , | x - x _ { B } | )$ </td><td>69.53 93.93 21.12</td></tr><tr><td>3</td><td> $\mathrm { C o n c a t } ( x , x _ { B } , ( x - x _ { B } ) ^ { 2 } )$ </td><td>69.45 93.26 21.92</td></tr></table>

TABLE VI

ABLATION STUDY OF FREQUENCY-GUIDED PERCEPTUAL LOSS COEFFICIENT λ ON IRSTD-1K. THE EVALUATION METRICS ARE IOU $( 1 0 ^ { - 2 } ) , P _ { d } ( 1 0 ^ { - 2 } )$ , AND $F _ { a } \ ( 1 0 ^ { - 6 } )$
<table><tr><td>λ</td><td>IoU ↑</td><td> $\mathrm { P } _ { d } \uparrow$ </td><td> $\operatorname { F } _ { a \mathrm { ~ \downarrow ~ } }$ </td></tr><tr><td>0.01</td><td>69.62</td><td>92.92</td><td>25.46</td></tr><tr><td>0.05</td><td>70.21</td><td>94.61</td><td>18.40</td></tr><tr><td>0.1</td><td>72.13</td><td>91.91</td><td>13.64</td></tr><tr><td>0.3</td><td>70.91</td><td>92.25</td><td>23.17</td></tr><tr><td>0.5</td><td>70.60</td><td>91.24</td><td>16.75</td></tr><tr><td>0.7</td><td>70.04</td><td>93.60</td><td>23.40</td></tr></table>

We also compare with the commonly used high-pass filtering pre-processing, i.e., $\mathrm { C o n c a t } ( x , \mathcal { H P } ( x ) )$ , which is designed to emphasize high-frequency components and suppress smooth background regions in infrared images. Although this strategy can highlight potential target regions, it leads to noticeably inferior performance. This indicates that handcrafted high-frequency enhancement is less effective than our reconstruction-based approach, and further demonstrates that our DiffIRSTD framework provides a more accurate and adaptive way to separate foreground from background.

In addition, we explicitly construct difference-based inputs, such as $\left| x \mathrm { ~ - ~ } x _ { B } \right|$ and $( x \mathrm { ~ - ~ } x _ { B } ) ^ { 2 }$ , to directly model the discrepancy. However, these variants result in slight performance degradation compared to the simple concatenation. One possible explanation is that the explicit difference map, while providing the potential target region prior, may also amplify noise or reconstruction errors, thereby misleading the detection network. This phenomenon is also observed in the studies of triple-channel inputs, and significantly increases $\mathrm { F } _ { a }$

Coefficients in Frequency-guided Perceptual Loss. As discussed in the main paper, the frequency-guided perceptual loss is introduced to explicitly preserve the structural details of reconstructed images in the frequency domain. Results of varying loss coefficients are reported in Tab VI. We can observe that (1) When λ is too small, the high-frequency constraint becomes relatively weak, which may lead to insufficient preservation of structural details in the reconstructed background, slightly degrading detection performance. (2) Conversely, when λ is too large, the high-frequency loss tends to dominate the optimization process, encouraging the network to overemphasize detail reconstruction, which may introduce noise or high-frequency artifacts and thus slightly increase false alarms. (3) The best comprehensive performance is achieved at $\lambda = 0 . 1$ , where the model obtains the highest IoU (72.13) and the lowest false alarm rate. Overall, although the performance varies with different λ, the fluctuations are not significant, and the model consistently achieves competitive results. This suggests that the proposed frequency-guided loss is inherently stable, while a moderate setting provides the best trade-off between structural consistency and detail fidelity, leading to optimal detection performance.

![](images/0605e64e029a0cec30314ca17798d8f5dcd8a37e471b4a0a427075e276fb896c.jpg)  
Fig. 5. Visualizations of reconstructed background images with different diffusion time steps t. The red and yellow boxes denote close-ups of the foreground target and background details, respectively. Best viewed zoomed in.

TABLE VII  
ABLATION STUDY OF HIGH-PASS CUTOFF FREQUENCY r<sub>c</sub> ON IRSTD-1K. THE EVALUATION METRICS ARE IOU (10<sup>−2</sup>), $\mathring { P } _ { d } ( 1 0 ^ { - 2 } ) ,$ , AND $F _ { a } \left( 1 0 ^ { - 6 } \right)$ ).
<table><tr><td> $r _ { c }$ </td><td>IoU ↑</td><td> $\mathrm { P } _ { d } \uparrow$ </td><td> $\operatorname { F } _ { a \downarrow }$ </td></tr><tr><td>0.1</td><td>67.74</td><td>92.92</td><td>41.50</td></tr><tr><td>0.3</td><td>69.87</td><td>94.27</td><td>24.50</td></tr><tr><td>0.5</td><td>70.26</td><td>94.94</td><td>15.88</td></tr><tr><td>0.7</td><td>72.13</td><td>91.91</td><td>13.64</td></tr><tr><td>0.9</td><td>68.40</td><td>94.27</td><td>26.64</td></tr></table>

Choice of High-pass Cutoff Frequency $r _ { c } .$ It can be observed from the experimental results in Tab VII that the cutoff radius coefficient $r _ { c }$ of the high-pass filter exerts a crucial impact on the detection performance. The parameter $r _ { c }$ controls the retention range of high-frequency components in the frequency domain. If $r _ { c }$ is too small, the high-frequency constraint becomes excessively lenient, causing the generated background to lose a large number of edge details, which in turn leads to a low IoU and a high false alarm rate. If $r _ { c }$ is too large, low-frequency information is overly suppressed, and the generated background may produce texture artifacts or structural distortion, also resulting in performance degradation. The experimental results demonstrate that $r _ { c } ~ = ~ 0 . 7$ can achieve the optimal balance between retaining necessary highfrequency details and maintaining the overall structure of the background, thereby maximizing the detection performance. Generalization to Other Detectors. Table IX evaluates the generalization ability of our DiffIRSTD framework when combined with different detection networks. We reproduce several representative works [6], [39], [50] with the official<sup>1</sup> or popular<sup>2</sup> codebases. It can be observed that our method can be seamlessly integrated with various detectors, including ACM, DNANet, and SCTransNet, demonstrating good compatibility and flexibility. With the introduction of the reconstruction module, all detectors show consistent performance improvements. Specifically, ACM achieves an IoU gain of +3.73, DNANet improves by +4.03, and SCTransNet gains +2.76. Similar trends can also be observed in the false-alarm rate $F _ { a }$ , where detectors benefit from noticeable reductions. This suggests the plug-and-play advantage of our DiffIRSTD framework, which provides effective background priors that can be readily exploited by different detectors, leading to consistent and stable performance improvements.

Discussion on Reconstruction Methods. To validate the effectiveness of the reconstruction network, we compare two representative denoising paradigms, $i . e .$ , the classical autoencoder and our one-step diffusion model, as shown in Tab. VIII. Auto-encoders are widely adopted for image reconstruction and denoising due to their simple architecture and efficient inference, making them a strong baseline for background modeling. In contrast, diffusion models demonstrate superior generative capabilities, particularly in modeling complex data distributions and producing high-fidelity reconstructions. In this work, we further adopt a one-step denoising strategy to retain the strong generative prior of diffusion models while ensuring computational efficiency.

TABLE VIII  
COMPARISONS WITH DIFFERENT RECONSTRUCTION METHODS. THE EVALUATION METRICS ARE IOU $( 1 0 ^ { - 2 } ) , P _ { d } ( 1 0 ^ { - 2 } )$ , AND $F _ { a } \ ( 1 0 ^ { - 6 } )$
<table><tr><td rowspan="2">RECONSTRUCTION NETWORK</td><td colspan="3">IRSTD-1K</td><td colspan="3">NUDT-SIRST</td><td colspan="3">NUAA-SIRST</td></tr><tr><td>IoU ↑</td><td> $\mathrm { P } _ { d } \uparrow$ </td><td> $\mathrm { F } _ { a \downarrow }$ </td><td>IoU ↑</td><td> $\mathrm { P } _ { d } \uparrow$ </td><td> $\operatorname { F } _ { a \mathrm { ~ \downarrow ~ } }$ </td><td>IoU ↑</td><td> $\mathrm { P } _ { d } \uparrow$ </td><td> $\operatorname { F } _ { a \mathrm { ~ \downarrow ~ } }$ </td></tr><tr><td>Auto-Encoder</td><td>67.01</td><td>90.23</td><td>15.97</td><td>94.95</td><td>99.36</td><td>2.57</td><td>78.99</td><td>94.29</td><td>10.44</td></tr><tr><td>One-Step Diffusion</td><td>72.13</td><td>91.91</td><td>13.64</td><td>95.91</td><td>99.49</td><td>1.72</td><td>80.73</td><td>96.95</td><td>7.55</td></tr></table>

TABLE IX

GENERALIZATION OF OUR DIFFIRSTD WITH OTHER DETECTORS. THE RED NUMBER MEANS THE PERFORMANCE GAIN ON IRSTD-1K.
<table><tr><td>METHODS</td><td>IoU ↑</td><td> $\mathrm { P } _ { d } \uparrow$ </td><td> $\operatorname { F } _ { a \mathrm { ~ \downarrow ~ } }$ </td></tr><tr><td>ACM [50]</td><td>60.98</td><td>89.89</td><td>38.26</td></tr><tr><td>+ DiffIRSTD (Ours)</td><td> $6 4 . 7 1 _ { + 3 . 7 3 }$ </td><td> $9 0 . 9 0 _ { + 1 . 0 1 }$ </td><td> $3 6 . 0 0 _ { - 2 . 2 6 }$ </td></tr><tr><td>DNANet [39]</td><td>65.50</td><td>90.23</td><td>18.80</td></tr><tr><td>+ DiffIRSTD (Ours)</td><td> $6 9 . 5 3 _ { + 4 . 0 3 }$ </td><td> $9 1 . 2 4 _ { + 1 . 0 1 }$ </td><td> $1 4 . 5 9 _ { - 4 . 2 1 }$ </td></tr><tr><td>SCTransNet [6]</td><td>67.28</td><td>95.28</td><td>19.49</td></tr><tr><td>+ DiffIRSTD  $( O u r s )$ </td><td> $7 0 . 0 4 _ { + 2 . 7 6 }$ </td><td> $9 5 . 5 8 _ { + 0 . 3 0 }$ </td><td> $1 8 . 9 5 _ { - 0 . 5 4 }$ </td></tr></table>

From the quantitative results, the one-step diffusion model consistently outperforms the auto-encoder across all three datasets. Specifically, it achieves higher IoU and $P _ { d } ,$ while reducing the false alarm rate $F _ { a } .$ For example, on IRSTD-1K, the IoU improves from 67.01 to 72.13, accompanied by a noticeable reduction in false alarms. Similar performance gains can also be observed on NUDT-SIRST and NUAA-SIRST, demonstrating the robustness of the improvement. These results indicate that the diffusion-based reconstruction provides a cleaner background reconstruction, which is crucial for highlighting foreground–background discrepancies in infrared small target detection. Compared with the auto-encoder, the stronger generative capacity of the diffusion model enables better denoising of target-related signals during reconstruction, thereby facilitating subsequent detection.

## V. BROADER IMPACT AND LIMITATION

## A. Broader Impact

In the field of infrared small target detection, this work proposes an explicit decomposition paradigm of “background reconstruction + target detection”. In critical scenarios such as maritime search and rescue and missile early warning, this work could improve the reliability of the system with a lower false alarm rate and higher detection accuracy. Furthermore, this work promotes the application of generative models in the remote sensing field, utilizing diffusion models to achieve “target erasure”. This idea can be migrated to other remote sensing tasks (e.g., cloud removal, occluder elimination), expanding a new perspective for the role of generative models in visual tasks.

## B. Limitations and Future Work

Although the framework proposed in this paper achieves significant performance improvements in the infrared small target detection task, it still has some limitations. In this work, diffusion sampling is compressed to a single step, which greatly improves the inference speed and reduces the inference overhead, but in terms of computational cost, the reconstruction network still contains a large number of parameters. For practical applications, a viable solution is to deploy the reconstruction network independently in the cloud, serving as a preprocessing step during the inference process. However, if the entire framework needs to be deployed on edge devices, further lightweight optimization is still required. An interesting future work is to investigate more computationally efficient reconstruction mechanisms and explore the applicability of this framework to broader small-object detection scenarios.

## VI. CONCLUSION

This paper presents DiffIRSTD, a diffusion-driven framework for infrared small target detection that approaches the problem from a reconstruction-oriented perspective. Rather than directly inferring target regions, we first recover a target-free background representation by suppressing targetrelated signals through a diffusion-based denoising process. The comparison differences between the reconstructed background and the original image thus naturally reveal candidate target regions, providing informative guidance for detection. Experiments on multiple benchmarks verify the effectiveness and robustness of the proposed framework. Furthermore, the modular design of the reconstruction component allows it to be readily integrated with different detection architectures, demonstrating strong adaptability and practical value.

## REFERENCES

[1] M. Zhao, W. Li, L. Li, J. Hu, P. Ma, and R. Tao, “Single-frame infrared small-target detection: A survey,” IEEE geoscience and remote sensing magazine, vol. 10, no. 2, pp. 87–119, 2022.

[2] D. Ren, J. Li, M. Han, and M. Shu, “Dnanet: Dense nested attention network for single image dehazing,” in ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2021, pp. 2035–2039.

[3] X. Wu, D. Hong, and J. Chanussot, “Uiu-net: U-net in u-net for infrared small object detection,” IEEE Transactions on Image Processing, vol. 32, pp. 364–376, 2022.

[4] Q. Liu, R. Liu, B. Zheng, H. Wang, and Y. Fu, “Infrared small target detection with scale and location sensitivity,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 17 490–17 499.

[5] M. Zhang, R. Zhang, Y. Yang, H. Bai, J. Zhang, and J. Guo, “Isnet: Shape matters for infrared small target detection,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 877–886.

[6] S. Yuan, H. Qin, X. Yan, N. Akhtar, and A. Mian, “Sctransnet: Spatialchannel cross transformer network for infrared small target detection,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1– 15, 2024.

[7] C. Hu, Y. Huang, K. Li, L. Zhang, C. Long, Y. Zhu, T. Pu, and Z. Peng, “Datransnet: Dynamic attention transformer network for infrared small target detection,” IEEE Geoscience and Remote Sensing Letters, 2025.

[8] M. Zhang, H. Bai, J. Zhang, R. Zhang, C. Wang, J. Guo, and X. Gao, “Rkformer: Runge-kutta transformer with random-connection attention for infrared small target detection,” in Proceedings of the 30th ACM International Conference on Multimedia, 2022, pp. 1730–1738.

[9] T. Chen, Z. Ye, Z. Tan, T. Gong, Y. Wu, Q. Chu, B. Liu, N. Yu, and J. Ye, “Mim-istd: Mamba-in-mamba for efficient infrared smalltarget detection,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–13, 2024.

[10] M. Zhang, X. Li, F. Gao, and J. Guo, “Irmamba: Pixel difference mamba with layer restoration for infrared small target detection,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 9, 2025, pp. 10 003–10 011.

[11] J. Jiang, S. Liao, X. Yang, and K. Shen, “Eamnet: Efficient adaptive mamba network for infrared small target detection,” IEEE Transactions on Geoscience and Remote Sensing, 2025.

[12] M. Zhang, Y. Wang, J. Guo, Y. Li, X. Gao, and J. Zhang, “Irsam: Advancing segment anything model for infrared small target detection,” in European Conference on Computer Vision. Springer, 2024, pp. 233– 249.

[13] Y. Fu, J. Lyu, P. Ma, Z. Liu, and M. K. Ng, “A unified sam-guided selfprompt learning framework for infrared small target detection,” IEEE Transactions on Geoscience and Remote Sensing, 2025.

[14] M. Zhang, X. Li, F. Gao, J. Guo, X. Gao, and J. Zhang, “Saist: Segment any infrared small target model guided by contrastive language-image pretraining,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 9549–9558.

[15] F.-A. Croitoru, V. Hondru, R. T. Ionescu, and M. Shah, “Diffusion models in vision: A survey,” IEEE transactions on pattern analysis and machine intelligence, vol. 45, no. 9, pp. 10 850–10 869, 2023.

[16] H. Ding, N. Huang, Y. Wu, and X. Cui, “Infrared small target detection improvement via hybrid data augmentation using diffusion models and gan,” IEEE Transactions on Aerospace and Electronic Systems, vol. 61, no. 2, pp. 3031–3045, 2024.

[17] L. Li, Y. Song, T. Tian, and J. Tian, “Dense condition-driven diffusion network for infrared small target detection,” IEEE Transactions on Instrumentation and Measurement, vol. 73, pp. 1–13, 2024.

[18] S.-h. Rong, Z.-m. Wang, X.-z. Gao, W.-f. Zhao, X.-p. Wu, and T. Zhang, “A novel diffusion-based background estimation for infrared dim small target detection,” Infrared Physics & Technology, p. 106384, 2026.

[19] T. Chenhao, Y. Wenya, D. Nini, Z. Binhao, and X. Sheng, “Diffusion model for infrared small target detection,” Infrared Technology, vol. 47, no. 6, pp. 757–764, 2025.

[20] B. Li, C. Xiao, L. Wang, Y. Wang, Z. Lin, M. Li, W. An, and Y. Guo, “Dense nested attention network for infrared small target detection,” IEEE Transactions on Image Processing, vol. 32, pp. 1745–1758, 2023.

[21] X. Bai and F. Zhou, “Analysis of new top-hat transformation and the application for infrared dim small target detection,” Pattern Recognition, vol. 43, no. 6, pp. 2145–2156, 2010.

[22] C. Gao, D. Meng, Y. Yang, Y. Wang, X. Zhou, and A. G. Hauptmann, “Infrared patch-image model for small target detection in a single image,” IEEE transactions on image processing, vol. 22, no. 12, pp. 4996–5009, 2013.

[23] L. Zhang and Z. Peng, “Infrared small target detection based on partial sum of the tensor nuclear norm,” Remote Sensing, vol. 11, no. 4, p. 382, 2019.

[24] F. Wu, T. Zhang, L. Li, Y. Huang, and Z. Peng, “Rpcanet: Deep unfolding rpca based infrared small target detection,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2024, pp. 4809–4818.

[25] G. Zhang, G. Xu, S. Chen, H. Wang, and X. Zhang, “It’s not the target, it’s the background: Rethinking infrared small target detection via deep

patch-free low-rank representations,” IEEE Transactions on Geoscience and Remote Sensing, 2025.

[26] S. Chen, P. Sun, Y. Song, and P. Luo, “Diffusiondet: Diffusion model for object detection,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 19 830–19 843.

[27] H. Zhang, Z. Wang, D. Zeng, Z. Wu, and Y.-G. Jiang, “Diffusionad: Norm-guided one-step denoising diffusion for anomaly detection,” IEEE transactions on pattern analysis and machine intelligence, 2025.

[28] Y. Shi, Y. Lin, P. Wei, X. Xian, T. Chen, and L. Lin, “Diff-mosaic: augmenting realistic representations in infrared small target detection via diffusion prior,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–11, 2024.

[29] J. Li, Y. Lu, X. Guo, X. Xian, T. Wang, and Y. Shi, “Exploiting gaussian agnostic representation learning with diffusion priors for enhanced infrared small target detection,” Neural Networks, p. 108185, 2025.

[30] L. Fan, Y. Wang, G. Hu, F. Li, Y. Dong, H. Zheng, C. Lin, Y. Huang, and X. Ding, “Diffusion-based continuous feature representation for infrared small-dim target detection,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–17, 2024.

[31] N. Du, X. Gong, and Y. Liu, “Istd-diff: Infrared small target detection via conditional diffusion models,” IEEE Geoscience and Remote Sensing Letters, vol. 21, pp. 1–5, 2024.

[32] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in neural information processing systems, vol. 33, pp. 6840– 6851, 2020.

[33] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in International Conference on Medical image computing and computer-assisted intervention. Springer, 2015, pp. 234–241.

[34] A. Telea, “An image inpainting technique based on the fast marching method,” Journal of Graphics Tools, vol. 9, 01 2004.

[35] Y. Sun, J. Yang, and W. An, “Infrared dim and small target detection via multiple subspace learning and spatial-temporal patch-tensor model,” IEEE Transactions on Geoscience and Remote Sensing, vol. 59, no. 5, pp. 3737–3752, 2021.

[36] T. Wu, B. Li, Y. Luo, Y. Wang, C. Xiao, T. Liu, J. Yang, W. An, and Y. Guo, “Mtu-net: Multilevel transunet for space-based infrared tiny ship detection,” IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1–15, 2023.

[37] K. Li, Y. Wang, F. Han, H. Wang, Z. Xiong, and Y. Tian, “Hstnet: A hybrid spatial-channel sparse transformer network for infrared small target detection,” IEEE Transactions on Geoscience and Remote Sensing, 2025.

[38] M. Zhang, K. Yue, J. Zhang, Y. Li, and X. Gao, “Exploring feature compensation and cross-level correlation for infrared small target detection,” in Proceedings of the 30th ACM International Conference on Multimedia, 2022, pp. 1857–1865.

[39] B. Li, C. Xiao, L. Wang, Y. Wang, Z. Lin, M. Li, W. An, and Y. Guo, “Dense nested attention network for infrared small target detection,” IEEE Transactions on Image Processing, vol. 32, pp. 1745–1758, 2022.

[40] Q. Hou, L. Zhang, F. Tan, Y. Xi, H. Zheng, and N. Li, “Istdu-net: Infrared small-target detection u-net,” IEEE Geoscience and Remote Sensing Letters, vol. 19, pp. 1–5, 2022.

[41] T. Zhang, S. Cao, T. Pu, and Z. Peng, “Agpcnet: Attention-guided pyramid context networks for infrared small target detection,” arXiv preprint arXiv:2111.03580, 2021.

[42] M. Zhang, R. Zhang, J. Zhang, J. Guo, Y. Li, and X. Gao, “Dim2clear network for infrared small target detection,” IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1–14, 2023.

[43] Q. Li, W. Zhang, W. Lu, and Q. Wang, “Multi-branch mutual-guiding learning for infrared small target detection,” IEEE Transactions on Geoscience and Remote Sensing, 2025.

[44] J. Yang, S. Liu, J. Wu, X. Su, N. Hai, and X. Huang, “Pinwheelshaped convolution and scale-based dynamic loss for infrared small target detection,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 9, 2025, pp. 9202–9210.

[45] F. Wu, A. Liu, T. Zhang, L. Zhang, J. Luo, and Z. Peng, “Saliency at the helm: Steering infrared small target detection with learnable kernels,” IEEE Transactions on Geoscience and Remote Sensing, 2024.

[46] S. Zhuang, Y. Hou, M. Qi, and D. Wang, “Faa-net: A frequency-aware attention network for single-frame infrared small target detection,” IEEE Transactions on Instrumentation and Measurement, 2025.

[47] M. Zhang, R. Zhang, Y. Yang, H. Bai, J. Zhang, and J. Guo, “Isnet: Shape matters for infrared small target detection,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 877–886.

[48] Y. Dai, Y. Wu, F. Zhou, and K. Barnard, “Asymmetric contextual modulation for infrared small target detection,” in Proceedings of the IEEE/CVF winter conference on applications of computer vision, 2021, pp. 950–959.

[49] P. Dhariwal and A. Nichol, “Diffusion models beat gans on image synthesis,” Advances in neural information processing systems, vol. 34, pp. 8780–8794, 2021.

[50] Y. Dai, Y. Wu, F. Zhou, and K. Barnard, “Asymmetric contextual modulation for infrared small target detection,” in Proceedings of the IEEE/CVF winter conference on applications of computer vision, 2021, pp. 950–959.