# PolUR: Polarization Aware Multimodal Degradation Perception for Underwater Image Restoration

Wenshuo Zhang, Xiangyue Zhang\*, Junlin Li, Wenjie Tian, Yubo Zheng, Hongli Xu, Jingyu Ru, Xinde Li\*\*, Senior Member, IEEE

Abstract—Recovering high-quality images from degraded underwater observations is crucial for underwater exploration. Polarization observations show promise as an effective source of complementary physical cues for recovering target radiance under clear-water conditions. However, existing methods have yet to jointly model raw intensity information and polarization cues to enable image restoration driven by an understanding of mixed degradations (color casts, haze, and low light). To overcome these limitations, we propose PolUR, a unified multimodal language-driven framework that jointly uses polarization and RGB information to identify degradation types and severity levels through structured semantic instructions and enhances joint restoration representations under the guidance of degradationaware semantic features to improve restoration robustness under mixed degradations. We further construct a new benchmark named MixPolar, which contains underwater images exhibiting real-world degradations and their corresponding degradation descriptions. MixPolar supports systematic evaluation of degradation perception, image restoration, and performance on downstream tasks. Extensive experiments demonstrate that PolUR effectively integrates polarization and RGB information, achieving competitive performance in degradation perception and image restoration, while also demonstrating generalization capability and utility for downstream tasks.

Index Terms—Polarimetric imaging, Vision-Language Models (VLM), underwater imaging.

## I. INTRODUCTION

U <sup>NDERWATER</sup> <sup>imaging</sup> <sup>is</sup> <sup>significantly</sup> <sup>affected</sup> <sup>by</sup> <sup>light</sup> absorption and scattering, which attenuate target radiance and introduce scattered-light interference, resulting in color casts, low contrast, and poor visibility [1]. Optical differences between water and air limit the direct transfer of restoration methods to underwater environments [26], [28], [29], motivating dedicated underwater image restoration (UIR) techniques that enhance visual quality and make scene information more discernible, thereby providing more reliable observations for downstream vision tasks.

Traditional UIR methods rely on statistical priors [3], [5] or physical models [2], [4]. However, their reliance on predefined assumptions and sensitivity to parameter estimation accuracy can limit their generalization to complex underwater scenes.

Existing RGB-based methods for underwater image restoration typically enhance feature representations through multidomain modeling [16], [57] and prior guidance [17], [30]–[33], [54]. Multi-domain approaches combine domain transformations with cross-domain feature fusion, with their adaptability to degradations relying primarily on architecture-driven adaptive feature learning. To enable more targeted control over restoration, recent studies increasingly incorporate priors from generative models and vision–language models for conditionguided feature modulation (as shown in Fig. 1(a)). However, the loss of scene information caused by absorption and scattering in water is difficult to fully compensate for through feature modeling and prior-based inference alone, leaving their performance constrained by the original observations.

![](images/06a0bd89f0977a59e218da28ac22c0d67e20427cd14762484f86794d90bf0c57.jpg)  
Fig. 1. Comparison between proposed PolUR and previous approaches to underwater image restoration. (a) RGB-based UIR methods rely primarily on degraded intensity observations, with degradation priors inferred from the same information-limited RGB inputs, which restricts reliable degradation perception under complex underwater conditions. (b) Polarization-based UIR methods exploit physical cues related to light propagation and scattering, but incorporating these cues alone does not ensure adequate characterization of water conditions and compound degradation states. (c) PolUR unifies physical observation, degradation perception, and restoration decision-making within a framework, jointly using RGB and polarization cues to reason about coexisting degradations and guide restoration.

In contrast, polarization imaging captures polarization variations induced during underwater light propagation and scattering, thereby characterizing the differences between scattered light and target radiance and providing a physical basis for separating and suppressing scattering components [6]. Meanwhile, the polarization responses of target reflections vary with the material and surface characteristics of underwater objects, revealing additional intrinsic information about the targets. Such characteristics provide a more informative basis for underwater image restoration.

Existing polarization-based UIR methods mainly rely on scattering models, using multi-angle polarization observations to estimate backscattering and transmission map for physically constrained radiance recovery [6], [7], [58]. However, their reliance on idealized imaging assumptions limits robustness and generalization under complex real-world underwater conditions. With advances in deep learning, data-driven restoration methods that exploit the physical relationship between target radiance and polarization information have attracted increasing attention. Specifically, as shown in 1(b), these methods jointly use RGB and polarization information to learn a mapping from degraded observations to target radiance through improvements in network architectures or restoration pipelines [12], [13], [22]. However, imaging characteristics vary across aquatic environments, as do the relative contributions of the two modalities. Focusing solely on incorporating polarization information without adequately representing water conditions limits the model’s ability to adapt to varying scene conditions.

To address these challenges, we propose PolUR, an instruction-driven framework for polarization-based underwater image restoration that unifies physical perception and language understanding, as illustrated in Fig. 1(c). The framework takes RGB and polarization information together with textual instructions as input and performs degradation diagnosis and instruction-guided image restoration.

The framework comprises two components: a Polarizationaware Degradation Understanding Model (PolarDU) and a Unified Response-Guided Restoration Network (UR<sup>2</sup>Net). PolarDU combines the complementary degradation representations of RGB and polarization observations with structured text to enable explicit perception and understanding of complex degradations. Guided by degradation-aware hidden states extracted from PolarDU, UR<sup>2</sup>Net adapts the restoration process to different degradation conditions. We further introduce an RGB–polarization alignment (RPA) module and a progressive training strategy to narrow the representation gap between the two modalities and facilitate cross-modal feature fusion. To support systematic joint degradation analysis and restoration evaluation, we construct MixPolar, an RGB–polarization benchmark covering diverse underwater degradation types and severity levels.

The main contributions of this work are summarized as follows:

1) We introduce MixPolar, an RGB–polarization benchmark covering diverse types and severity levels of mixed underwater degradations, with a unified evaluation pipeline spanning degradation analysis, image restoration, and downstream vision tasks to assess both restoration quality and the utility of restored images.

2) We propose PolUR, the first unified framework that combines RGB and polarization observations with structured textual instructions for joint degradation understanding and uses the resulting degradation-aware representations to guide restoration under mixed underwater degradations.

3) We propose a progressive training strategy and RPA module to gradually align semantic and physical representations while preserving the model’s visual understanding and image restoration capabilities.

## II. RELATED WORK

This section reviews RGB-based and polarization-based underwater image restoration, as well as all-in-one image restoration in both general and underwater scenes.

## A. RGB-based Underwater Image Restoration

RGB-based underwater image restoration methods primarily learn enhanced representations from degraded observations to establish a stable restoration mapping [14] [15].

Some methods construct enhanced representations through multi-domain feature modeling based on degraded observations. Peng et al. [16] jointly model degradation features in the spatial and frequency domains, while Wang et al. fuse complementary features from the RGB and XYZ color spaces to improve color restoration and detail enhancement.

Other methods derive vision or language priors from degraded RGB observations to guide representation enhancement. Wu et al. [17] leverage generative models to jointly restore the clear scene and estimate transmission and backscattering from a single RGB image. Cao et al. [31] use high-level semantic representations from CLIP to construct perceptual supervision constraints for training. Yao et al. [32] identify degradation categories in RGB inputs through CLIP and use the corresponding representations to guide restoration. Wang et al. [54] combine CLIP-derived degradation representations with RAM-generated semantic tags to guide adaptive restoration. Fan et al. [33] use LLaVA to extract the textual descriptions from input images to refine diffusion conditions.

Despite improvements in restoration performance, these methods still rely primarily on degraded RGB observations for scene-specific restoration, leaving their reliability constrained by information loss caused by absorption and scattering.

## B. Polarization-based Underwater Image Restoration

Polarization-based underwater image restoration leverages polarization cues associated with underwater light propagation and scattering to facilitate scene radiance recovery. Existing approaches can be broadly divided into physics-driven and learning-based methods, with the latter further characterized by advances in network architecture and restoration paradigm.

The Jaffe–McGlamery model [20], [21] established the physical basis for modeling underwater image formation. Schechner et al. [6] [7] exploited multi-angle polarization observations to characterize the polarization differences between target radiance and scattered light, thereby enabling scattering suppression under general polarization assumptions. Li et al. [24] further developed a polarization imaging optimization model and employed local parameter optimization to improve the stability of parameter estimation.

With the development of deep learning, data-driven polarization representation learning has gradually become an important direction in underwater polarization restoration.

At the architectural level, Hu et al. [22] employed a densely connected network to fuse multi-level polarization features and learn the nonlinear mapping between polarization observations in turbid water and clear-scene radiance. Lin et al. [23] introduced CNN and Transformer modules into U-Net to model local details and long-range dependencies, respectively, thereby enhancing target reconstruction under turbid conditions. Fan et al. [25] further combined deformable convolution with meta-learning to improve adaptation under limited-data settings.

At the restoration paradigm level, Shen et al. [11] adopted a dual-branch network to estimate the transmission map and background light, respectively, and incorporated no-reference constraints to enable unsupervised polarization-based image restoration. Liu et al. [12] introduced diffusion models into underwater polarization restoration and employed a dualstream generation process to separately model image intensity and polarization information, enabling their joint restoration under turbid conditions. Ju et al. [13] designed a lightweight polarization feature extraction module and injected polarization information as auxiliary features into existing RGBbased underwater enhancement networks, thereby extending the integration of polarization cues into general restoration frameworks.

These studies have advanced the use of polarization information in underwater image restoration. Building on these advances, further research is warranted on representing compound degradation states in complex underwater environments and using these representations to guide RGB–polarization fusion and image restoration.

## C. All-in-One Image Restoration

To address unknown degradation types, dynamically varying degradation severity, and the coexistence of multiple degradations in real-world scenarios, All-in-One image restoration aims to adaptively handle diverse degradations within a unified framework. Li et al. [26] employed contrastive learning to construct a degradation encoder and used degradation-guided restoration to accommodate unknown degradations of varying types and severity levels. Wang et al. [28] constructed a retrieval-based degradation bank to dynamically retrieve degradation priors matched to the input, allowing external degradation knowledge to participate in the restoration process. Fan et al. [29] leveraged a pretrained vision-language model to capture high-level degradation semantics and combined degradation-aware prompting with feature compensation to adaptively guide restoration under diverse degradation conditions.

However, the distinct degradation characteristics caused by underwater absorption and scattering limit the direct transfer of general all-in-one restoration methods to underwater scenes.

In underwater scenes, Zhang et al. [30] employed a mixtureof-experts mechanism to model different degradation patterns, enabling adaptive restoration of heterogeneous and spatially varying underwater degradations. Nevertheless, its performance remains constrained by limited coverage of degradation conditions. Moreover, its depth priors are inferred solely from degraded RGB images, the pretrained depth estimator faces a domain gap when applied to underwater scenes.

## III. PROPOSED METHOD

In the following sections, we first introduce the construction of polarization representations (Section III-A) and provide an overview of the proposed framework (Section III-B). We then elaborate on the two components of our method, including UR<sup>2</sup>Net (Section III-C) and PolarDU (Section III-D). Finally, we present the training Strategy (Section III-E).

## A. Polarization Representation Construction

Given the observations captured by a division-of-focal-plane (DoFP) polarization camera at four analyzer orientations $0 ^ { \circ }$ 45<sup>◦</sup>, 90<sup>◦</sup>, 135<sup>◦</sup>, denoted by $I _ { 0 } , I _ { 4 5 } , I _ { 9 0 } , I _ { 1 3 5 }$ , the linear Stokes components are computed as [34], [35]:

$$
\left[ S _ { 0 } \right] = \left[ \begin{array} { c } { { I _ { 0 } + I _ { 4 5 } + I _ { 9 0 } + I _ { 1 3 5 } } } \\ { { 2 } } \\ { { I _ { 0 } - I _ { 9 0 } } } \\ { { I _ { 4 5 } - I _ { 1 3 5 } } } \end{array} \right] .\tag{1}
$$

where $S _ { 0 }$ represents the total intensity, while $S _ { 1 }$ and $S _ { 2 }$ characterize the polarization differences between the horizontal– vertical and diagonal analyzer pairs, respectively.

![](images/0c9b67b8ad38d7ab3647f49c429c8bf1a78bec92f9543d8ce883a80701d66a8f.jpg)  
Fig. 2. Polarization Observation-to-Representation Mapping.

As shown in Fig. 2, we represent the normalized linear polarization state by jointly encoding the polarization magnitude with DoLP and the polarization orientation with the unit-circle coordinates $( \hat { S } _ { 1 } , \hat { S } _ { 2 } )$ . The resulting polarization representation $\mathbf { P } = [ \mathrm { D o L P } , \hat { S } _ { 1 } , \hat { S } _ { 2 } ]$ can be directly derived from the Stokes components as

$$
\mathbf { P } = \left[ \frac { \sqrt { S _ { 1 } ^ { 2 } + S _ { 2 } ^ { 2 } } } { S _ { 0 } } , \mathcal { M } _ { \varepsilon } \left[ \frac { S _ { 1 } } { S _ { 0 } \mathrm { D o L P } } \right] , \mathcal { M } _ { \varepsilon } \left[ \frac { S _ { 2 } } { S _ { 0 } \mathrm { D o L P } } \right] \right]\tag{2}
$$

Here, $\mathcal { M } _ { \varepsilon } [ \cdot ]$ sets the two directional components to zero wherever $\sqrt { S _ { 1 } ^ { 2 } + S _ { 2 } ^ { 2 } } \le \varepsilon$ to suppress unreliable orientation cues from weak linearly polarized signals. Constructed directly from the Stokes components, this representation retains informative polarization cues for subsequent feature learning without explicit estimation of the angle of polarization (AoP).

## B. Overall Framework

As illustrated in Fig. 3, PolUR comprises two main components: (a) UR<sup>2</sup>Net and (b) PolarDU. Given the degraded RGB image I, the polarization representation P, and a degradationassessment question Q, PolarDU generates a degradation description A<sup>ˆ</sup> that identifies the color cast and specifies the scattering severity, illumination level, and visibility level. The contextual representations of its answer tokens, denoted by $f ^ { e }$ provide semantic guidance for restoration. $\mathrm { { U R ^ { 2 } } }$ Net receives I, P, and the degradation-aware semantic feature $f ^ { e }$ as inputs. The inputs I and P are encoded into latent features $f ^ { R }$ and $f ^ { p }$ by two independent VAE encoders [40]. To bridge the modality gap, $\scriptstyle { \dot { f } } ^ { R }$ and $f ^ { p }$ are mapped into a shared representation space and fused via RPA to obtain a unified enhanced feature $f ^ { a }$ . The resulting feature $f ^ { a }$ is then fed into the denoising U-Net, whose operation is modulated by $f ^ { G }$ after transformation through a semantic adapter, yielding the restored image <sup>ˆ</sup>I.

![](images/488d2ec1dfe30452a374d1f2fd6787058bf3b1e9f676ba34a0c1062ec957f6e4.jpg)  
Fig. 3. Overview of the proposed PolUR framework. (a) and (b) jointly constitute the inference pipeline. Given an RGB image together with P, PolUR produces both an explicit degradation response and a restored image. Specifically, (a) represents $\mathrm { U R ^ { 2 } N e t }$ , which performs response-guided image restoration, while (b) denotes the proposed PolarDU for multimodal degradation understanding. (c) illustrates the overall training procedure of PolUR. (d) presents the RPA module embedded in $\mathrm { U R ^ { 2 } N e t }$ , which aligns and adaptively integrates the polarization and RGB latent representations.

## C. Unified Response-Guided Restoration Network

RGB and polarization information differ in their ability to characterize underwater degradations, while compound degradations complicate the selection of targeted restoration strategies. To address these challenges, we propose $\mathrm { U R ^ { 2 } N e t }$ a single-step latent diffusion framework that adaptively fuses the latent representations of both modalities and incorporates degradation semantics encoded in the hidden states of VLM responses to guide restoration, enabling unified restoration across diverse underwater scenes without iterative reversediffusion sampling.

As shown in Fig. 3(a), I and P are encoded into $f ^ { R }$ and $f ^ { p }$ , respectively, by two independent VAE encoders initialized with the pretrained encoder weights of the Stable Diffusion 2 VAE. These representations are then fused by the RPA module shown in Fig. 3(d) as follows:

$$
f ^ { A } = \widetilde { f } ^ { r } + \mathrm { C A } ( \widetilde { f } ^ { r } , \widetilde { f } ^ { p } ) + \mathrm { F F N } ( \mathrm { C A } ( \widetilde { f } ^ { r } , \widetilde { f } ^ { p } ) ) ,\tag{3}
$$

where $\widetilde { f } ^ { r }$ and $\widetilde { f } ^ { p }$ are obtained by projecting $f ^ { r }$ and $f ^ { p }$ through $3 \times 3$ convolutions. CA denotes cross-attention over

$3 \times 3$ neighborhoods, and FFN denotes a $1 \times 1$ feed-forward projection.

The fused features $f ^ { A }$ are fed into a restoration network based on the U-Net architecture of Stable Diffusion 2 (SD2) [39]. Within the U-Net, the projected semantic features $f ^ { E }$ interact with image features through cross-attention layers in the downsampling, middle, and upsampling paths, providing degradation-aware guidance for reconstruction. Together, $f ^ { A }$ and $f ^ { E }$ serve as the spatial and semantic conditions for learning a direct restoration mapping. A single-step formulation is adopted, in which the U-Net is evaluated once at a fixed terminal timestep $\tau ,$ conditioned on these features. Its output is then converted into the restored latent representation using the v-prediction parameterization with $z _ { \tau } = 0 ,$ , followed by VAE decoding to reconstruct the restored image.

## D. Polarization-aware Degradation Understanding Model

Directly expanding the RGB input channels to incorporate physical information may disrupt the semantic representations learned during pretraining. To introduce physical awareness while preserving the base reasoning capabilities, a degradation-aware architecture built on LLaVA-1.5 [55] is proposed, as shown in Fig. 3(b). Feature extraction from intensity and physical information is decoupled, with the intensity branch providing a stable semantic reference and the physical branch supplying complementary degradation cues. Given I and P, feature extraction and projection into the language model’s embedding space are jointly expressed as

$$
\widetilde { T } _ { R } = \mathcal { F } _ { R } ( I ) , \qquad \widetilde { T } _ { P } = \mathcal { F } _ { P } ( { \bf P } ) ,\tag{4}
$$

where $\mathcal { F } _ { R }$ and $\mathcal { F } _ { P }$ denote modality-specific encoding and projection mappings, and $\widetilde { T } _ { R }$ and $\widetilde { T } _ { P }$ are the RGB and polarization tokens in the embedding space.

As $\smash { \widetilde { T } } _ { R }$ primarily encodes appearance-related information while $\bar { T } _ { P }$ provides complementary polarization-aware physical cues, we preserve them as separate token sequences for degradation understanding. Different from the restoration branch, where local spatial correspondences between the two modalities are explicitly modeled through cross-attention, the understanding branch concatenates $\widetilde { T } _ { R }$ and $\smash { \widetilde { T } } _ { P }$ with the text embeddings of $Q _ { \ l }$ , allowing the LLM to selectively aggregate complementary cross-modal evidence through self-attention and autoregressively generate the degradation description ${ \hat { a } } .$ To transfer this degradation understanding to the restoration branch, we directly use the contextualized hidden states associated with aˆ. Passing only the discrete response text to restoration would omit the internal representations formed during multimodal reasoning. These hidden states retain response semantics conditioned on the visual observations and the query, providing semantic guidance for restoration. Specifically, the final-layer hidden states corresponding to valid generated answer tokens are stacked to construct the semantic representation:

$$
f ^ { e } = \left[ \mathbf { h } _ { t } ^ { ( L ) } \right] _ { t \in \mathcal { A } _ { e } } \in \mathbb { R } ^ { T _ { e } \times d _ { \ell } } ,\tag{5}
$$

where ${ \bf h } _ { t } ^ { ( L ) }$ denotes the final-layer hidden state at position $t ,$ $A _ { e }$ is the set of valid answer-token positions, and $T _ { e } = | \mathcal { A } _ { e } |$ Prompt, end-of-sequence, and padding positions are excluded.

## E. Training Strategy and Objectives

We develop a three-stage progressive training strategy, as illustrated in Fig. 3(c).In the first stage, the RGB branch is bypassed, and polarization-only alignment is performed to associate polarization cues with degradation semantics. In the second stage, RGB observations are introduced for joint RGB–polarization training. LoRA adapters [38] are employed to adapt the polarization encoder and language model for cross-modal degradation understanding while keeping their pretrained base weights frozen. Both recognition stages are trained using the cross-entropy loss:

$$
\mathcal { L } _ { \mathrm { r e c } } = - \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \log p _ { \theta } ( a _ { t } \mid a _ { < t } , Q , \mathcal { V } ) ,\tag{6}
$$

where $a _ { t }$ is the t-th token of the target response $^ { a , }$ and $a _ { < t }$ denotes the preceding answer tokens. The target response length T includes the end-of-sequence token. The visual context V consists of $\smash { \widetilde { T } } _ { P }$ in the first stage and both $\widetilde { T } _ { R }$ and $\smash { \widetilde { T } } _ { P }$ in the second stag.

In the third stage, the degradation recognizer is frozen, and only $f ^ { E }$ is used to provide semantic guidance for image restoration. The RGB VAE encoder and decoder also remain frozen, while the remaining restoration components are trained. Restoration losses are computed between the restored image $\hat { I }$ and the paired reference image $I _ { \mathrm { g t } }$ , with subject masks used for structural supervision:

$$
\mathcal { L } _ { \mathrm { r e s t } } \lambda _ { 1 } \mathcal { L } 1 + \lambda s \mathcal { L } \mathrm { S S I M } ^ { \mathrm { s u b j } } + \lambda p \mathcal { L } \mathrm { L P I P S } ,\tag{7}
$$

where $\lambda 1 , \lambda _ { s } ,$ , and $\lambda _ { p }$ weight the three loss terms. $\mathcal { L } _ { 1 } , \mathcal { L } _ { \mathrm { S S I M } } ^ { \mathrm { s u b j } }$ and $\mathcal { L } _ { \mathrm { L P I P S } }$ denote the pixel-wise $\ell _ { 1 }$ loss, subject-maskweighted SSIM loss, and LPIPS perceptual loss.

## IV. EXPERIMENT

## A. Implementation Details

AdamW [46] is used as the optimizer for all three training stages. In the first stage, we set the learning rates for the polarization input layer and projector to $5 \times 1 0 ^ { - 5 }$ and $2 \times 1 0 ^ { - 4 }$ and train for 10 epochs with a batch size of 4 and 4 gradient accumulation steps. In the second stage, the learning rates for the LoRA adapters, Vicuna LoRA adapters, and projector are set to $2 \times 1 0 ^ { - 5 } , 2 \times 1 0 ^ { - 4 }$ , and $5 \times 1 0 ^ { - 5 }$ , respectively. Training lasts for 20 epochs with a batch size of 4 and 2 gradient accumulation steps. In the final stage, we set the learning rate to $5 \times 1 0 ^ { - 6 }$ for both the polarization VAE and language adapter module, $5 \times 1 0 ^ { - 5 }$ for the fusion module, and $1 \times 1 0 ^ { - 5 }$ for the U-Net. This stage lasts for 15 epochs with a batch size of 1 and 4 gradient accumulation steps. All training is implemented in PyTorch and conducted on an NVIDIA A40 GPU. The input image resolution in the final stage is set to $5 1 2 \times 5 1 2$

## B. Experiment Settings

![](images/7b8702d412d7fc3fe63a3c5ddcf1fc321bd13bd7c78c6e5a4400b8daf7c3cc1c.jpg)  
Fig. 4. The distribution of the Mixpolar.

1) Details on MixPolar: To support the training and evaluation of PolUR, we construct MixPolar, a multi-task polarization dataset, by integrating and reprocessing raw polarization measurements from existing datasets [37], [41], [42] and providing task-specific annotations. As shown in Fig. 4, MixPolar supports degradation perception, image restoration, and downstream task evaluation. For degradation perception, we annotate four attributes: color cast, scattering severity, illumination conditions, and overall visibility, and convert the resulting labels into standardized textual descriptions. Scattering severity is categorized as low, medium, or high based on the corresponding optical thickness values. The remaining labels are assigned or verified through manual visual assessment. Specifically, color cast is categorized as none, red, green, or blue; illumination conditions as low-light or normal; and overall visibility as low, medium, or high. For image restoration, we establish an Indoor subset with paired ground-truth images acquired under laboratory conditions and an Outdoor subset without paired reference images. For downstream task evaluation, we use images annotated with binary segmentation masks and compare segmentation performance on the original and restored images to assess the impact of restoration on downstream visual tasks.

2) Evaluation Metrics: We evaluate degradation perception using mean macro-F1 and joint accuracy. Mean macro-F1 provides a balanced assessment of recognition performance across degradation attributes and classes, while joint accuracy assesses the reliability of complete degradation diagnoses. For indoor image restoration, we use PSNR, SSIM [50], LPIPS [51], and DISTS [52] to assess reconstruction fidelity, structural similarity, and perceptual similarity. For outdoor evaluation, where paired references are unavailable, we use the no-reference metrics UIQM [47], URanker [48], and UCIQE [49] to assess image quality.

## C. Effectiveness of Polarization Information

we investigate the response of P to scattering and its ability to distinguish color-cast types by computing PSNR and SSIM between the corresponding representations of each degraded observation and its paired clean reference.

As shown in Fig. 5, the scene-averaged PSNR and SSIM generally decrease as scattering increases at both acquisition distances. The qualitative comparisons in Fig. 6 further support this trend, showing an overall decline in the similarity between the polarization representation and the reference images as scattering increases. Together, these quantitative and qualitative results support the sensitivity of the polarization representation to scattering-related changes, although the quantitative metrics do not decrease strictly monotonically at intermediate scattering levels.

![](images/70ba708f31a90dbb9e0b44802c0146970883394e0a807940ef3389b1039312a4.jpg)

![](images/9abe192105c70da7487161177dacb5ee8cdff68091ba6893bd43290190d7ffda.jpg)

![](images/b5189fa518bb4e1b60bcfcb1889491d62410440c292efc50401957767a252bf6.jpg)

![](images/6d1b17a2c53fd9b0ad28f5fa92d1144a804b5f14af3f4cf5cf12f74f8912c067.jpg)  
Fig. 5. Similarity between degraded polarization representations and paired clean references on PURB. Panels (a) and (b) show PSNR and SSIM at an acquisition distance of 45 cm; panels (c) and (d) show the results at 60 cm. Curves represent normal, red-cast, green-cast, and blue-cast conditions. Shaded regions indicate pointwise 95% confidence intervals obtained through scene-level bootstrap resampling.

The separation between color-cast types is less pronounced. The condition without a color cast yields higher mean similarity to the clean reference, whereas the red-cast and greencast conditions follow closely spaced curves. Under stronger scattering, the PSNR curves for the three color-cast conditions become increasingly close. These observations suggest that the scalar similarity measures provide limited discrimination among specific color-cast types in this analysis, motivating the joint use of polarization information and RGB appearance cues for degradation identification.

![](images/96c14cb7f074f7aa59a4013ce6eb79c4be2c0f4114e7645c3cf59613ddcb5b4b.jpg)  
Fig. 6. Variation in polarization characteristics with scattering severity

## D. Degradation Analysis on Mixpolar

To characterize the empirical capability boundaries of current open-source VLMs in underwater degradation perception and the use of polarization information, we compare PolarDU with three representative models on MixPolar: Qwen3-VL-8B [45], InternVL3.5-8B [43], and MiniCPM-V 4.5 [44]. Each baseline is directly evaluated with RGB-only and RGB-Polar inputs under the same evaluation protocol and inference settings as PolarDU to assess the correctness of joint degradation diagnoses and whether adding polarization information improves attribute identification.

As shown in Table I, PolarDU outperforms all compared VLMs in both joint accuracy and macro-F1, reflecting more reliable joint identification of degradation attributes and stronger class-balanced recognition performance. Adding polarization inputs yields inconsistent benefits: InternVL3.5-8B improves across the reported metrics, whereas Qwen3-VL-8B shows mixed changes and MiniCPM-V 4.5 performs worse. These results suggest limitations in the use of polarization information by the evaluated VLMs under the tested input settings, motivating further exploration of targeted fine-tuning and cross-modal alignment to improve their understanding of polarization cues and degradation attributes.

![](images/18ae15ed4ac156d3489f33535e18bab28cc781332fa8a77cd585f5dedb075f36.jpg)  
Fig. 7. Qualitative comparison of degradation identification using RGB-only and RGB-Polar inputs. Green and red text indicate correct and incorrect attribute predictions.

![](images/0414021ba2c8a8d68bdd8edb46dcdba1b39ae6ae45e47a0258ed5ddc0ed6445b.jpg)  
Fig. 8. Visual comparisons on the MixPolar indoor subset and RGBP. Input and reference images (GT) are shown in the first and last columns, respectively.

TABLE I  
PERFORMANCE COMPARISON OF DIFFERENT METHODS
<table><tr><td>Model</td><td>Input</td><td>Accuracy (%) F1 (%)</td><td>Score</td></tr><tr><td rowspan="2">Qwen3-VL-8B</td><td>RGB-only</td><td>4.40</td><td>26.01 15.20</td></tr><tr><td>RGB-Polar</td><td>1.70</td><td>26.96 14.33</td></tr><tr><td rowspan="2">InternVL3.5-8B</td><td>RGB-only</td><td>1.31</td><td>32.62 16.97</td></tr><tr><td>RGB-Polar</td><td>7.33</td><td>37.69 22.51</td></tr><tr><td rowspan="2">MiniCPM-V 4.5</td><td>RGB-only</td><td>6.56</td><td>30.28 18.42</td></tr><tr><td>RGB-Polar</td><td>6.02</td><td>24.45 15.23</td></tr><tr><td>PolarDU</td><td>RGB-Polar</td><td>76.23</td><td>90.07 83.15</td></tr></table>

To further illustrate the differences in joint degradation identification, we qualitatively compare representative RGBonly and RGB-Polar predictions, as shown in Fig. 7. All three compared VLMs correctly identify the red color cast but misclassify cloudiness: InternVL3.5-8B underestimates it, whereas Qwen3-VL-8B and MiniCPM-V 4.5 overestimate it. Although adding polarization inputs corrects the visibility prediction of InternVL3.5-8B, it introduces an illumination error. The other two models retain the same diagnoses under both input settings and continue to underestimate visibility. By comparison, PolarDU correctly identifies all four attributes, providing a complete and accurate degradation diagnosis for this example.

## E. Restoration Performance in Controlled Scenes

We compare PolUR with nine representative approaches, all evaluated using their official implementations and grouped into three categories: (1) polarization-based underwater restoration and enhancement methods, including UIRPIOM [24], UPI2Diff [12], and UPGD [13], (2) underwater image restoration methods, including SS-UIE [16], SLURPP [17], and DIE2UIE [54], and (3) all-in-one image restoration methods, including UniUIR [30], DCMPC [29], and R2R [28]. We use their official implementations, pretrained checkpoints, and default configurations without further training or tuning.

As shown in Table II, PolUR achieves the best or joint-best results on most metrics across both datasets, demonstrating strong overall restoration performance. Specifically, it leads in PSNR and LPIPS while attaining the best or joint-best SSIM, reflecting advantages in reconstruction fidelity, structural consistency, and perceptual similarity to the reference images. Although its DISTS score on the MixPolar indoor subset is slightly higher than those of DIE-UIE and SLURPP, the absolute gaps remain small (0.007 and 0.006, respectively). We further conduct qualitative comparisons, as shown in Fig. 8(b). Polarization-based optimization methods are omitted from the figure due to their poor restoration quality. Other polarizationbased restoration methods recover some structural details but still exhibit noticeable color casts and blurring. Among the image restoration methods, SS-UIE exhibits incomplete detail recovery and grainy artifacts. SLURPP preserves clear overall structures, although the colors in some regions still differ from the reference images. DIE-UIE produces local overexposure in some examples, reducing detail visibility. In examples with pronounced haze, all-in-one restoration methods, including UniUIR, DCMPC, and R2R, still exhibit blurred details, residual haze, or color casts. By comparison, PolUR preserves clearer textures and object contours, achieving a good balance between detail recovery and color cast suppression.

TABLE II  
QUANTITATIVE COMPARISON ON THE MIXPOLAR INDOOR AND RGBP. THE BEST AND SECOND-BEST DISTINCT VALUES ARE MARKED IN BOLD AND UNDERLINED, RESPECTIVELY, TIED BEST RESULTS ARE BOTH MARKED IN BOLD. ↑ AND ↓ INDICATE THAT HIGHER AND LOWER VALUES ARE BETTER, RESPECTIVELY.
<table><tr><td>Method</td><td colspan="4">MixPolar Indoor</td><td colspan="4">RGBP</td></tr><tr><td></td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>DISTS ↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>DISTS ↓</td></tr><tr><td>UIRPIOM</td><td>6.6200</td><td>0.3030</td><td>0.7930</td><td>0.6570</td><td>9.3662</td><td>0.2467</td><td>0.6636</td><td>0.4265</td></tr><tr><td>UPI2Diff</td><td>18.9040</td><td>0.4720</td><td>0.4236</td><td>0.3550</td><td>17.0288</td><td>0.3937</td><td>0.7035</td><td>0.4924</td></tr><tr><td>UPGD</td><td>14.2305</td><td>0.5081</td><td>0.3387</td><td>0.2879</td><td>15.8287</td><td>0.5195</td><td>0.4854</td><td>0.4589</td></tr><tr><td>SS-UIE</td><td>16.8250</td><td>0.4300</td><td>0.5101</td><td>0.3910</td><td>16.9519</td><td>0.1958</td><td>0.4805</td><td>0.3270</td></tr><tr><td>SLURPP</td><td>20.1621</td><td>0.7340</td><td>0.1760</td><td>0.1620</td><td>19.4897</td><td>0.6037</td><td>0.2214</td><td>0.1800</td></tr><tr><td>DIE-UIE</td><td>17.3200</td><td>0.7350</td><td>0.1831</td><td>0.1610</td><td>18.6579</td><td>0.5225</td><td>0.3525</td><td>0.3235</td></tr><tr><td>UniUIR</td><td>19.1524</td><td>0.6290</td><td>0.2804</td><td>0.2390</td><td>18.1337</td><td>0.5898</td><td>0.4544</td><td>0.3583</td></tr><tr><td>DCMPC</td><td>19.7740</td><td>0.5620</td><td>0.2120</td><td>0.2100</td><td>18.3306</td><td>0.4982</td><td>0.4860</td><td>0.3890</td></tr><tr><td>R2R</td><td>18.5353</td><td>0.4120</td><td>0.2605</td><td>0.2470</td><td>16.2180</td><td>0.1822</td><td>0.4706</td><td>0.3870</td></tr><tr><td>PolUR</td><td>20.8262</td><td>0.7350</td><td>0.1684</td><td>0.1680</td><td>19.7495</td><td>0.6181</td><td>0.2095</td><td>0.1609</td></tr></table>

TABLE III

QUANTITATIVE COMPARISON ON MIXPOLAR OUTDOOR. THE BEST AND SECOND-BEST RESULTS ARE MARKED IN BOLD AND UNDERLINED, RESPECTIVELY.
<table><tr><td>Method</td><td>UIQM ↑</td><td>URanker ↑</td><td>UCIQE ↑</td></tr><tr><td>UIRPIOM</td><td>0.6370</td><td>-0.7039</td><td>0.3187</td></tr><tr><td>UPI2Diff</td><td>2.9569</td><td>-0.0010</td><td>0.3451</td></tr><tr><td>UPGD</td><td>1.8900</td><td>0.7613</td><td>0.2763</td></tr><tr><td>SS-UIE</td><td>2.2645</td><td>0.7897</td><td>0.3108</td></tr><tr><td>SLURPP</td><td>2.5210</td><td>0.3099</td><td>0.2473</td></tr><tr><td>DIE-UIE</td><td>2.7413</td><td>0.7794</td><td>0.3472</td></tr><tr><td>UniUIR</td><td>2.7480</td><td>1.0174</td><td>0.3370</td></tr><tr><td>DCMPC</td><td>2.0294</td><td>0.1379</td><td>0.3362</td></tr><tr><td>R2R</td><td>2.2145</td><td>0.1666</td><td>0.2912</td></tr><tr><td>PolUR</td><td>2.8934</td><td>1.2767</td><td>0.3529</td></tr></table>

## F. Generalization to Real-World Underwater Scenes

To assess PolUR’s generalization ability, we directly evaluate it on the MixPolar outdoor subset, which was excluded from both its degradation identification and image restoration training. Given the distribution differences between indoor laboratory environments and real-world outdoor scenes, underwater restoration baselines use their official pretrained checkpoints for a fair assessment of outdoor generalization, while baselines requiring adaptation to underwater restoration are trained on the MixPolar indoor subset. All methods are evaluated directly on the outdoor subset without further training or fine-tuning.

As shown in Table III, PolUR achieves the highest URanker and UCIQE scores, performing well under both learned quality ranking and statistical assessment of color and contrast. Although it ranks second in UIQM, the absolute gap to UPI2Diff is only 0.0635. These complementary results support PolUR’s competitive restoration performance in the tested real-water conditions without additional training.

The visual results in Fig. 9 complement these scores. UPI2Diff retains residual color casts and leaves some details in dark regions indistinct, while UPGD reduces the casts but produces relatively smooth surface details in the first example. Among the image restoration methods, SS-UIE produces excessively bright regions, SLURPP retains yellowish or grayish appearances, and DIE-UIE exhibits yellow amplification and local overexposure. All-in-one restoration methods, including UniUIR, DCMPC, and R2R, retain residual haze or much of the original color cast, limiting scene visibility. In comparison, PolUR reduces the dominant green and blue casts and makes the fish, vegetation, and surrounding surfaces more distinguishable, although details in some shadow regions remain limited.

## G. Ablation Studies

We conduct ablation studies on degradation identification and image restoration to assess the contributions of the main design choices in PolUR.

1) Degradation Identification: To assess the contributions of the input modalities and the polarization alignment stage, we conduct ablation studies. We compare RGB-only and polarization-only variants with the full model. The variant without stage 1 skips polarization alignment and proceeds directly to joint fine-tuning.

As shown in Table IV, the full model achieves the highest joint accuracy and macro-F1, outperforming both singlemodality variants and supporting the benefit of jointly using RGB and polarization information. Although the RGB-only variant performs better than the polarization-only variant, its lower performance than the full model suggests that polarization information provides complementary cues for degradation identification. Removing stage 1 also reduces both metrics despite retaining joint fine-tuning, supporting the contribution of polarization alignment to complete attribute diagnosis and class-balanced recognition.

2) Image Restoration: To assess the contributions of the restoration components, we construct three ablation variants: removing RPA, omitting degradation guidance from PolarDU (w/o recognition), and replacing the adapter with direct pooling to match the required feature dimensions (w/o adapter). All variants are evaluated on the MixPolar indoor subset.

![](images/d4df6b34747dba120ea2dbf7d5e35d5974978f12cf465cc5f9eec173ad91e3b0.jpg)  
Fig. 9. Visual comparisons on real-world outdoor scenes from MixPolar under direct evaluation without target-domain retraining or fine-tuning. The first and last columns show the input images and PolUR results, respectively.

TABLE IV  
ABLATION OF INPUT MODALITIES AND THE TRAINING STRATEGY FOR DEGRADATION IDENTIFICATION ON MIXPOLAR. ACC. DENOTES JOINT ACCURACY, AND SCORE IS THE ARITHMETIC MEAN OF ACC. AND MACRO-F1. BEST AND SECOND-BEST RESULTS ARE HIGHLIGHTED IN BOLD AND UNDERLINED.
<table><tr><td>Variant</td><td>Acc. (%) ↑</td><td>Macro-F1(%)↑</td><td>Score ↑</td></tr><tr><td>RGB only</td><td>70.14</td><td>87.46</td><td>78.80</td></tr><tr><td>Polarization only</td><td>56.02</td><td>79.59</td><td>67.80</td></tr><tr><td>w/o stage 1</td><td>72.07</td><td>88.23</td><td>80.15</td></tr><tr><td>Full model</td><td>76.23</td><td>90.07</td><td>83.15</td></tr></table>

TABLE V

ABLATION OF IMAGE RESTORATION COMPONENTS ON THE MIXPOLAR INDOOR SUBSET. BEST RESULTS ARE SHOWN IN BOLD, AND SECOND-BEST RESULTS ARE UNDERLINED.
<table><tr><td>Variant</td><td>PSNR (dB) ↑</td><td>SSIM ↑</td></tr><tr><td>w/o recognition</td><td>20.3990</td><td>0.728</td></tr><tr><td>w/o RPA</td><td>20.4408</td><td>0.725</td></tr><tr><td>w/o adapter</td><td>20.5936</td><td>0.746</td></tr><tr><td>Full model</td><td>20.8262</td><td>0.735</td></tr></table>

As shown in Table V, omitting the hidden states from PolarDU or removing RPA reduces both PSNR and SSIM, supporting the use of degradation guidance and the alignment and fusion of VAE-encoded polarization and RGB features for restoration. The adapter learns to compress the hidden states into compact conditioning features and achieves higher PSNR than direct pooling, although its lower SSIM reveals a tradeoff between reconstruction fidelity and structural similarity.

## H. Downstream Segmentation Evaluation

To assess whether PolUR enhancement improves downstream binary semantic segmentation, we train two LR-ASPP [56] models on raw underwater images and their enhanced counterparts, respectively. Both models use identical initialization, data splits, reference masks, and training settings. As shown in Table VI, LR-ASPP achieves higher

TABLE VI  
DOWNSTREAM SEGMENTATION PERFORMANCE OF SEGFORMER TRAINED AND EVALUATED ON RAW AND POLUR-ENHANCED IMAGES. BEST RESULTS ARE SHOWN IN BOLD.
<table><tr><td>Images</td><td>IoU (%) ↑</td><td>Dice (%) ↑</td><td>MAE↓</td></tr><tr><td>Raw</td><td>83.10</td><td>90.16</td><td>0.006005</td></tr><tr><td>Enhanced</td><td>83.22</td><td>90.23</td><td>0.005870</td></tr></table>

IoU and Dice and lower MAE than its counterpart using raw images, reflecting better agreement with the reference masks and reduced pixel-wise prediction error.

![](images/8d71def18462be6409bb6415884c3ae4cb7c263063f04f742a5dd10cefc59902.jpg)  
Fig. 10. Visual comparison of downstream segmentation results. From left to right: raw images, PolUR-restored images, ground-truth masks (GT), and predictions from SegFormer trained and evaluated on raw and restored images, respectively.

We further analyze representative examples in Fig. 10. In the first example, the restored-image setting produces a more complete foreground mask with fewer internal gaps. In the second example, the predicted contour more closely matches the reference mask, while in the third, isolated false-positive regions in the upper-right background are removed. These examples complement the quantitative results by illustrating improved mask completeness and reduced background false positives, although some boundary discrepancies remain.

## V. CONCLUSION

In this paper, we present PolUR, a unified framework for underwater degradation understanding and image restoration. Guided by textual instructions, PolUR uses complementary RGB and polarization information to understand mixed degradations and restore degraded images. We further developed a progressive training strategy and an RPA module to reduce cross-modal representation discrepancies during degradation perception and restoration. Meanwhile, we also constructed MixPolar, a multi-task benchmark supporting degradation understanding, image restoration, and downstream segmentation evaluation. Extensive experiments demonstrate that PolUR achieves competitive performance in both degradation understanding and image restoration.

## REFERENCES

[1] D. Akkaynak, T. Treibitz, T. Shlesinger, Y. Loya, R. Tamir, and D. Iluz, “What is the space of attenuation coefficients in underwater computer vision?” IEEE, 2017, pp. 568–577.

[2] Y.-T. Peng and P. C. Cosman, “Underwater image restoration based on image blurriness and light absorption,” vol. 26, no. 4, pp. 1579–1594, 2017.

[3] C. O. Ancuti, C. Ancuti, C. De Vleeschouwer, and P. Bekaert, “Color balance and fusion for underwater image enhancement,” vol. 27, no. 1, pp. 379–393, 2017.

[4] D. Akkaynak and T. Treibitz, “Sea-thru: A method for removing water from underwater images.” IEEE, 2019, pp. 1682–1691.

[5] D. Berman, D. Levy, S. Avidan, and T. Treibitz, “Underwater single image color restoration using haze-lines and a new quantitative dataset,” vol. 43, no. 8, pp. 2822–2837, 2020.

[6] Y. Y. Schechner and N. Karpel, “Recovery of underwater visibility and structure by polarization analysis,” vol. 30, no. 3, pp. 570–587, 2005.

[7] T. Treibitz and Y. Y. Schechner, “Active polarization descattering,” vol. 31, no. 3, pp. 385–399, 2008.

[8] J. Xu, J. Fan, J. Gao, C. Liao, and Y. Lin, “Msg-net: Structureguided enhancement for underwater images based on multi-view feature interaction,” 2026.

[9] D. Du, L. Si, F. Xu, J. Niu, and F. Sun, “A physical model-guided framework for underwater image enhancement and depth estimation,” 2026.

[10] X. Guo, X. Chen, S. Wang, and C.-M. Pun, “Underwater image restoration through a prior guided hybrid sense approach and extensive benchmark analysis,” vol. 35, no. 5, pp. 4784–4800, 2025.

[11] L. Shen, H. Xia, X. Zhang, Y. Zhao, N. Li, S. G. Kong, B. Wang, and Z. Li, “U<sup>2</sup>pnet: An unsupervised underwater image-restoration network using polarization,” vol. 54, no. 9, pp. 5164–5177, 2024.

[12] H. Liu, M. Qiu, H. Cao, Q. Ma, X. Li, and H. Hu, “Upi 2 diff: Restoring underwater polarization image and information in turbid conditions via polarization guided diffusion model,” 2026.

[13] G. Ju, J. Zhang, J. Zang, Y. Li, Y. Li, Z. Mi, X. Yang, H. Wang, J. Fan, and X. Fu, “A lightweight polarization-guided plug-in for underwater image enhancement,” 2025.

[14] C. Li, C. Guo, W. Ren, R. Cong, J. Hou, S. Kwong, and D. Tao, “An underwater image enhancement benchmark dataset and beyond,” vol. 29, pp. 4376–4389, 2019.

[15] M. J. Islam, Y. Xia, and J. Sattar, “Fast underwater image enhancement for improved visual perception,” vol. 5, no. 2, pp. 3227–3234, 2020.

[16] L. Peng and L. Bian, “Adaptive dual-domain learning for underwater image enhancement,” vol. 39, no. 6, 2025, pp. 6461–6469.

[17] J. Wu, T. Wang, M. A. B. Siddique, M. J. Islam, C. Fermuller, Y. Aloimonos, and C. A. Metzler, “Single-step latent diffusion for underwater image restoration,” 2025.

[18] Z. Zhang, Z. Jiang, L. Ma, J. Liu, X. Fan, and R. Liu, “Hupe: Heuristic underwater perceptual enhancement with semantic collaborative learning,” vol. 133, no. 6, pp. 3259–3277, 2025.

[19] S. Huang, K. Wang, H. Liu, J. Chen, and Y. Li, “Contrastive semisupervised learning for underwater image restoration via reliable bank.” IEEE, 2023, pp. 18 145–18 155.

[20] B. McGlamery, “A computer model for underwater camera systems,” in Ocean Opt. VI, vol. 208. SPIE, 1980, pp. 221–231.

[21] J. S. Jaffe, “Computer modeling and the design of optimal underwater imaging systems,” vol. 15, no. 2, pp. 101–111, 1990.

[22] H. Hu, Y. Zhang, X. Li, Y. Lin, Z. Cheng, and T. Liu, “Polarimetric underwater image recovery via deep learning,” vol. 133, p. 106152, 2020.

[23] B. Lin, W. Chen, X. Fan, P. Peng, and Z. Guo, “Transformer-based improved u-net for high-performance underwater polarization imaging,” vol. 181, p. 111664, 2025.

[24] Y. Li, Y. Chen, J. Zhang, Y. Li, and X. Fu, “An underwater image restoration method with polarization imaging optimization model for poor visible conditions,” vol. 35, no. 5, pp. 3924–3939, 2024.

[25] X. Fan, M. Ding, T. Lv, X. Sun, B. Lin, and Z. Guo, “Meta-dnetupi: Efficient underwater polarization imaging combining deformable convolutional networks and meta-learning,” vol. 187, p. 112900, 2025.

[26] B. Li, X. Liu, P. Hu, Z. Wu, J. Lv, and X. Peng, “All-in-one image restoration for unknown corruption.” IEEE, 2022, pp. 17 431–17 441.

[27] V. Potlapalli, S. W. Zamir, S. H. Khan, and F. Shahbaz Khan, “Promptir: Prompting for all-in-one image restoration,” vol. 36, pp. 71 275–71 293, 2023.

[28] C. Wang, K. Zhang, and J. Yang, “Retrieve-to-restore: Efficient all-inone image restoration with a retrieval-based degradation bank,” 2026, pp. 1277–1287.

[29] W. Fan, Y. Zhang, Y. Shen, L. Wang, J. Qin, K.-M. Lam, C. Wang, and J. Pan, “Degradation-aware prompt learning with cross-modal compensation for adverse weather removal,” 2026.

[30] X. Zhang, H. Zhang, G. Wang, Q. Zhang, L. Zhang, and B. Du, “Uniuir: Considering underwater image restoration as an all-in-one learner,” 2025.

[31] J. Cao, Z. Zeng, X. Zhang, H. Zhang, C. Fan, G. Jiang, and W. Lin, “Unveiling the underwater world: Clip perception model-guided underwater image enhancement,” Pattern Recognition, vol. 162, p. 111395, 2025.

[32] M. Yao, Z. Jiang, X. Fu, and H. Wang, “Conditional prompt learning via degradation perception for underwater image enhancement,” vol. 40, no. 14, 2026, pp. 11 892–11 900.

[33] G. Fan, S. Zhou, Z. Hua, J. Li, and J. Zhou, “Llava-based semantic feature modulation diffusion model for underwater image enhancement,” Information Fusion, p. 103566, 2025.

[34] H. Mei, B. Dong, W. Dong, J. Yang, S.-H. Baek, F. Heide, P. Peers, X. Wei, and X. Yang, “Glass segmentation using intensity and spectral polarization cues.” IEEE, 2022, pp. 12 612–12 621.

[35] J. Tang, R. Wu, X. Xu, S. Hu, and Y.-C. Chen, “Learning to remove wrinkled transparent film with polarized prior.” IEEE, 2024, pp. 24 987– 24 996.

[36] Y. Li, C. Zhou, H. Guo, B. Shi, I. Sato, and Z. Ma, “Polarvlm: Bridging the semantic-physical gap in vision-language models,” arXiv preprint arXiv:2605.07574, 2026.

[37] Q. Ma, X. Li, B. Li, Z. Zhu, J. Wu, F. Huang, and H. Hu, “Stamf: Synergistic transformer and mamba fusion network for rgb-polarization based underwater salient object detection,” Information Fusion, vol. 122, p. 103182, 2025.

[38] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “Lora: Low-rank adaptation of large language models,” arXiv preprint arXiv:2106.09685, 2021.

[39] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “Highresolution image synthesis with latent diffusion models.” ieee, 2022, pp. 10 674–10 685.

[40] D. P. Kingma and M. Welling, “Auto-encoding variational bayes,” arXiv preprint arXiv:1312.6114, 2013.

[41] Y. Li, “RGBP-UIE,” GitHub repository, 2025, rGB-polarization dataset for underwater image enhancement. [Online]. Available: https://github.com/yudongLi-dlmu/RGBP-UIE

[42] X. Zhang, W. Sun, Y. Zheng, J. Li, X. Qu, and H. Xu, “P 2 dnet: A physics-constrained dual-branch network for underwater polarimetric image restoration,” 2026.

[43] W. Wang, Z. Gao, L. Gu, H. Pu, L. Cui, X. Wei, Z. Liu, L. Jing, S. Ye, J. Shao et al., “Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency,” arXiv preprint arXiv:2508.18265, 2025.

[44] T. Yu, Z. Wang, C. Wang, F. Huang, W. Ma, Z. He, T. Cai, W. Chen, Y. Huang, R. Zhao et al., “Minicpm-v 4.5: Cooking efficient mllms via architecture, data, and training recipe,” 2026, pp. 11 704–11 715.

[45] S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge et al., “Qwen3-vl technical report,” arXiv preprint arXiv:2511.21631, 2025.

[46] I. Loshchilov, F. Hutter et al., “Fixing weight decay regularization in adam,” arXiv preprint arXiv:1711.05101, vol. 5, no. 5, p. 5, 2017.

[47] K. Panetta, C. Gao, and S. Agaian, “Human-visual-system-inspired underwater image quality measures,” vol. 41, no. 3, pp. 541–551, 2015.

[48] C. Guo, R. Wu, X. Jin, L. Han, W. Zhang, Z. Chai, and C. Li, “Underwater ranker: Learn which is better and how to be better,” vol. 37, no. 1, 2023, pp. 702–709.

[49] M. Yang and A. Sowmya, “An underwater color image quality evaluation metric,” vol. 24, no. 12, pp. 6062–6071, 2015.

[50] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: from error visibility to structural similarity,” vol. 13, no. 4, pp. 600–612, 2004.

[51] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric.” IEEE, 2018, pp. 586–595.

[52] K. Ding, K. Ma, S. Wang, and E. P. Simoncelli, “Image quality assessment: Unifying structure and texture similarity,” vol. 44, no. 5, pp. 2567–2581, 2020.

[53] E. Xie, W. Wang, Z. Yu, A. Anandkumar, J. M. Alvarez, and P. Luo, “Segformer: Simple and efficient design for semantic segmentation with transformers,” vol. 34, pp. 12 077–12 090, 2021.

[54] F. Wang, F. Li, C. Cen, Z. Li, and Q. Duan, “Underwater image enhancement via degradation information extraction and guidance,” Pattern Recognition, p. 113121, 2026.

[55] H. Liu, C. Li, Y. Li, and Y. J. Lee, “Improved baselines with visual instruction tuning.” IEEE, 2024, pp. 26 286–26 296.

[56] A. Howard, M. Sandler, G. Chu, L.-C. Chen, B. Chen, M. Tan, W. Wang, Y. Zhu, R. Pang, V. Vasudevan et al., “Searching for mobilenetv3,” in Proceedings of the IEEE/CVF international conference on computer vision, 2019, pp. 1314–1324.

[57] Y. Wang, P. Zhuang, Z. Fu, J. Yan, and Z. Li, “Mcsf-net: A multicolor space fusion network for underwater image enhancement,” IEEE Transactions on Multimedia, 2025.

[58] Y. Y. Schechner and N. Karpel, “Clear underwater vision,” in Proceedings of the 2004 IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2004. CVPR 2004., vol. 1. IEEE, 2004, pp. I–I.