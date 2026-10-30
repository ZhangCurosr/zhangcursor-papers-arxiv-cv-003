# Multi-level Constrained Learning for Underwater Image Enhancement

Wangtao Bao, Liping Xie, Member, IEEE, Kanjian Zhang, and Huimin Lu, Senior Member, IEEE

Abstract—Underwater image enhancement aims to correct color and recover detail degraded by wavelength-dependent attenuation, scattering, and noise. Paired-reference supervision constrains output similarity, but does not by itself specify the color mapping, the strength of local detail corrections, or consistency with the degraded observation. We propose MCL-UIE, a paired-supervised framework that addresses these requirements by constraining restoration operations and the training objective. Monotonic Chromatic Transform (MCT) restricts global color mapping in log-LMS space through positive diagonal entries and bounded cross-channel coefficients. Uncertainty-Gated Wavelet Reconstruction (UGWR) modulates high-frequency residuals with a learned confidence gate calibrated against normalized wavelet reconstruction errors. During training, Reference-Anchored Degradation Replay (RADR) renders the enhanced image and its paired reference using shared latent degradation proxies, encouraging both to reproduce the input. The three components share a restoration network that combines structured color correction, locally controlled detail recovery, and reference-anchored replay. MCL-UIE ranks first in thirteen of sixteen full-reference dataset-metric comparisons and produces competitive visual results on noreference benchmarks. Objective ablations indicate that confidence calibration is most effective when combined with replay supervision. RADR is used only during training and adds no inference-time computation. Our code is available at https://github.com/Wangtao-Bao/MCL-UIE.

Index Terms—Underwater Image Enhancement, Multi-level Constrained Learning, Color Correction, Wavelet Reconstruction, Degradation Consistency.

## I. INTRODUCTION

<sup>UE</sup> <sup>to</sup> <sup>light</sup> <sup>absorption,</sup> <sup>scattering,</sup> <sup>and</sup> <sup>wavelength-</sup>D selective attenuation in water, underwater images often suffer severe degradation, including color shifts, reduced contrast, uneven brightness, and loss of texture and structural detail. These degradations make underwater scenes difficult to analyze and interpret [1]. Underwater images are widely used in marine engineering, ecological monitoring, and detection by underwater robots [2], [3], [4]. Image enhancement can improve visibility for human observers and provide more reliable inputs for downstream tasks such as object detection and semantic segmentation [5], [6], [7], [8], [9]. Underwater image enhancement (UIE) thus supports both scene perception and the practical reliability of underwater vision systems.

Early UIE studies explored traditional image enhancement and model-based restoration [10], [11], [12], [13]. These methods improve color and contrast through image-domain processing or by estimating degradation parameters. However, handcrafted rules and assumed imaging conditions can limit their ability to adapt to different water types, lighting conditions, and scene content.

Deep learning has substantially advanced UIE [14], [15], [16], [17]. Trained on large datasets, these methods learn mappings from degraded observations to clear images and outperform traditional approaches in various underwater scenes. Their architectures, priors, and supervision terms differ, but supervision of the enhanced output alone does not specify how the network should transform color channels or regulate individual detail corrections. A low reconstruction error also does not establish whether the enhanced image can be re-degraded consistently with the input. These gaps motivate explicit constraints on intermediate restoration operations, used alongside output supervision.

Fig. 1 illustrates three considerations for controlling these operations. An unrestricted RGB mapping leaves the direction of channel transformations and cross-channel coupling unconstrained, which may destabilize image tones. Highfrequency corrections may benefit from an explicit reliability estimate, as their responses can also contain noise, suspended particles, and weak scene edges. The enhanced result can also be evaluated for consistency with the degraded input, a relation that direct comparison with a paired reference does not test. These considerations concern global color transformation, local detail reconstruction, and the relation between enhancement and degradation, respectively. Accordingly, we consider constraints at three complementary levels: global chromatic transformation, local high-frequency reconstruction, and degradation consistency. Together, they provide structured control over different stages of the restoration process while retaining a shared reconstruction objective.

![](images/980c58bf46150e583fa22f25f1629299982cc458cc6d5d0c4571267e033f1285.jpg)  
Fig. 1. Illustration of the challenges motivating MCL-UIE: unconstrained global color mapping, unreliable high-frequency enhancement that may amplify noise or artifacts, and ambiguous interpretation of degradation causes under paired supervision.

We propose Multi-level Constrained Learning for Underwater Image Enhancement (MCL-UIE), which combines these constraints in one restoration pipeline. Monotonic Chromatic Transform (MCT) first predicts a restricted global log-LMS mapping from robust image-level statistics and supplies the color-corrected input for spatial restoration. Uncertainty-Gated Wavelet Reconstruction (UGWR) predicts wavelet detail residuals and controls their magnitude through confidence calibrated against normalized reconstruction errors. During training, Reference-Anchored Degradation Replay (RADR) applies a shared renderer to the enhanced image and its paired reference, encouraging both paths to reproduce the degraded input. MCT and UGWR constrain restoration operations directly, while RADR constrains the output through a shared degradation model. A common backbone provides spatial features for reconstruction and degradation-proxy estimation. We train the three components jointly and evaluate their contributions through progressive component and training-objective ablations.

The main contributions are as follows:

1. We propose MCL-UIE to extend supervision beyond output fidelity by regulating key restoration operations. MCT, UGWR, and RADR respectively target global color transformation, local high-frequency correction, and degradation consistency.

2. We design MCT to predict a lower-triangular transform from robust log-LMS statistics, with positive diagonal entries and bounded cross-channel coefficients. For fixed transform parameters, this parameterization preserves positive selfchannel responses in the linear log-LMS core and limits crosschannel influence.

3. We develop UGWR to regulate wavelet detail residuals with a bounded confidence gate. Calibration against detached, normalized reconstruction errors gives the gate an explicit local reliability target for controlling high-frequency corrections.

4. We introduce RADR, which re-renders the enhanced image and its reference during training using shared wavelength-dependent attenuation, backscattering, and distance proxies. Reference replay anchors the degradation parameters to the paired observation; prediction replay couples the enhanced output to those same parameters.

## II. RELATED WORK

## A. Traditional and Physics-Based UIE Methods

Traditional UIE methods either enhance images without a physical model or restore them using a physical model. The first group uses white balance, color compensation, histogram adjustment, Retinex decomposition, or multi-scale fusion to improve color distribution and local contrast directly [11], [18], [19]. These methods generally require less computation because they do not estimate scene depth or water optical parameters, and they can produce relatively natural results in some scenes. However, their reliance on statistical priors or empirical rules makes it difficult to distinguish color attenuation, nonuniform illumination, and backscattering during light propagation through water. Fixed enhancement strategies can consequently overcompensate colors, overexpose local regions, or amplify noise when water type, lighting, or imaging distance changes substantially.

Physical model-based methods describe underwater light propagation and recover clear images by estimating transmittance, background light, attenuation coefficients, and scene distance. Early approaches adapted atmospheric scattering models, using dark-channel, red-channel, color-line, or wavelength-attenuation priors to estimate degradation parameters [20], [21], [22]. These methods offer physical interpretability, but their restoration quality depends strongly on the underlying assumptions and parameter estimates. Coupling between transmittance and background light can bias the estimates, particularly under artificial lighting, in nonuniform water, or in low-light conditions.

## B. Deep Learning-Based UIE Methods

Deep learning-based UIE has evolved from convolutional restoration to diverse representations and learning strategies. Early methods include UWCNN [14], which learns residual mappings from synthesized underwater images, and WaterNet [15], which fuses differently preprocessed inputs. FUnIE-GAN [23] introduces adversarial and perceptual constraints, while domain-adaptive UIE reduces distribution discrepancies across underwater environments [24]. UIEC²-Net [25] and Ucolor [26] exploit complementary color spaces, whereas PUIE-Net [27] models enhancement through conditional distribution learning and consensus estimation.

![](images/5ae966195f4d63eee3bc05e791f62275ef634a7ea38bf316b6133bae862422e6.jpg)  
Fig. 2. Overall architecture of the proposed MCL-UIE framework. The framework comprises Monotonic Chromatic Transform (MCT), a U-shaped spatial restoration backbone, Uncertainty-Gated Wavelet Reconstruction (UGWR), and Reference-Anchored Degradation Replay (RADR). RADR is used only during training and incurs no computational overhead during inference.

Recent studies further explore degradation-aware representations and improved supervision. SS-UIE [28] combines spatial selective scanning with spectral attention and frequency-wise supervision. PSPL [29] addresses imperfect pseudo-references through quality-guided comparisons and contrastive learning. IA2U [30] adapts pretrained in-air models using degradation-related priors, while PGMamba [31] incorporates underwater imaging priors into state-space modeling.

These methods advance feature representation and supervision, but output fidelity alone does not explicitly regulate global color mappings, individual high-frequency corrections, or consistency with the degraded observation. MCL-UIE addresses these complementary aspects through MCT, UGWR, and RADR, respectively, introducing constraints on restoration operations and degradation consistency alongside paired-reference supervision.

## C. Frequency-Domain Image Restoration

Spatial-domain networks effectively extract local features but lack explicit frequency separation. Frequency-domain methods address this limitation by decomposing and integrating complementary representations. Wavelet-based Dual-stream Networks [32] separately process low- and highfrequency components, while UHD-SFNet [33] combines spatial and Fourier features for color and texture restoration. Subsequent methods strengthen cross-domain interactions: SFGNet [34] employs gradient-aware spatial-frequency fusion, FDCE-Net [35] integrates frequency-spatial residual modeling with dual-color representations, and CDF-UIE [36] explores cross-domain feature interaction. More recent approaches incorporate wavelet modeling into advanced architectures. USWformer [37] introduces sparse wavelet attention, WWE-UIE [38] combines wavelet enhancement with white balancing and gradient-aware gating, and WaveletFormer [39] integrates learnable wavelet filters with multi-scale Transformers.

Despite these advances, high-frequency components contain both genuine scene details and degradation-induced responses, making the reliability of individual corrections important. UGWR addresses this issue by adaptively controlling predicted wavelet residuals through confidence gates calibrated against normalized reconstruction errors. This design explicitly regulates local corrections across color channels and wavelet directions, complementing existing frequency representation and fusion strategies.

## III. METHOD

## A. Overall Architecture

Fig. 2 shows the MCL-UIE architecture. MCT applies a global chromatic transform, after which a U-shaped backbone extracts spatial features. UGWR uses these features to predict wavelet residuals and control high-frequency updates, yielding the enhanced image. During training, RADR predicts degradation parameters from the backbone bottleneck and uses the same parameters in the enhanced-image and referenceimage replay paths. The network directly constrains the input color mapping and final detail corrections, while the training objective evaluates the output against both its paired reference and the degraded observation.

Given a degraded underwater image

$$
I \in [ 0 , 1 ] ^ { 3 \times H \times W } .\tag{1}
$$

MCT first corrects the image-level color. The spatial backbone then extracts full-resolution restoration features and bottleneck features, and UGWR uses the restoration features to reconstruct the image in the frequency domain:

$$
\left\{ \begin{array} { l l } { \ I _ { c } = \mathcal { M } ( I ) , } \\ { ( F , Z ) = \mathcal { B } ( I _ { c } ) , } \\ { \hat { J } = \mathcal { U } ( I _ { c } , F ) . } \end{array} \right.\tag{2}
$$

Here, $I _ { c }$ denotes the color-corrected MCT output, � and � denote the full-resolution and bottleneck features, respectively, and �<sup>̂</sup> is the final enhanced image.

During paired training, RADR estimates a shared set of latent degradation variables from � and uses them to re-render the enhanced image �<sup>̂</sup> and reference image � in the degradation domain:

$$
\left\{ \begin{array} { c } { \theta _ { d } = g ( Z ) = \{ \beta , A , d \} , } \\ { \tilde { I } _ { p } = \mathcal { R } \big ( \hat { J } ; \theta _ { d } \big ) , } \\ { \tilde { I } _ { r } = \mathcal { R } ( J ; \theta _ { d } ) . } \end{array} \right.\tag{3}
$$

MCT controls the global color mapping supplied to the backbone, UGWR controls the local corrections that form the output, and RADR constrains the relation between that output and the observation. We jointly optimize all three components under an objective that includes paired-reference fidelity. At inference, we retain only the restoration path in Eq. (2), so the replay branch adds no computation.

## B. Monotonic Chromatic Transform

Underwater color distortion often appears as a pronounced global imbalance between channels. Convolutional and attention networks can learn flexible RGB corrections, but need not restrict the direction or strength of cross-channel coupling. This flexibility may lead to overcompensation or tone shifts as water conditions change. MCT applies a restricted global transform before spatial restoration, reducing the need for the backbone to learn the entire color mapping implicitly. Image-level statistics determine the transform parameters. The parameterization controls self-channel responses and cross-channel influence. The backbone then reconstructs spatial content from the color-corrected input.

## 1) Statistical Representation

We map the input RGB image to LMS space and take its logarithm:

$$
L = l o g [ m a x ( M _ { R G B  L M S } \times I , 1 0 ^ { - 4 } ) ] .\tag{4}
$$

The fixed matrix $M _ { R G B  L M S }$ performs the RGB-to-LMS transformation. Taking the logarithm converts multiplicative changes in channel intensities into additive changes that are easier to model.

MCT uses a 14-dimensional statistical descriptor as input to the parameter predictor in place of high-dimensional image features:

$$
\begin{array} { r } { s = [ q _ { 0 . 0 5 } , q _ { 0 . 5 0 } , q _ { 0 . 9 5 } , \mu _ { R G B } , \bar { s } , \bar { g } ] \in \mathbb { R } ^ { 1 4 } . } \end{array}\tag{5}
$$

where $q _ { 0 . 0 5 } , q _ { 0 . 5 0 } .$ , and $q _ { 0 . 9 5 } \in \mathbb { R } ^ { 3 }$ are the 5%, 50%, and 95% quantiles of the three log-LMS channels, respectively; $\mu _ { R G B } \in \mathrm { R } ^ { 3 }$ contains the RGB-channel means; and �¯ and �¯ are the mean saturation and mean brightness gradient. The quantiles describe each channel's distribution range more robustly than a single mean, reducing sensitivity to isolated bright or dark regions. Together, the nine quantile statistics, three RGB means, and two global statistics form the descriptor. A lightweight statistical MLP maps this descriptor to the nine parameters of the constrained chromatic transform:

$$
\left\{ \begin{array} { r l } { p = \mathcal { F } _ { s t a t } ( s ) \in \mathbb { R } ^ { 9 } , } \\ { \mathcal { F } _ { s t a t } \colon \mathbb { R } ^ { 1 4 } \to \mathbb { R } ^ { 6 4 } \to \mathbb { R } ^ { 6 4 } \to \mathbb { R } ^ { 9 } . } \end{array} \right.\tag{6}
$$

## 2) Constrained Chromatic Mapping

To constrain the 3×3 transform, MCT splits the predicted vector p into three diagonal elements, three cross-channel elements, and three biases, with separate constraints for each group:

$$
\left\{ \begin{array} { l l } { d _ { i } = S o f t p l u s ( p _ { i } ) , } \\ { l _ { i } = 0 . 1 5 t a n h ( p _ { i + 3 } ) , } \\ { b _ { i } = 0 . 2 5 t a n h ( p _ { i + 6 } ) . } \end{array} \right.\tag{7}
$$

For $i = 1 , 2 , 3$ , these parameters form the lower-triangular color transform

$$
T = \left[ l _ { 1 } ^ { d _ { 1 } } \quad ^ { 0 } \quad ^ { 0 } \right] , d _ { i } > 0 , | l _ { i } | \leq 0 . 1 5 .\tag{8}
$$

This parameterization bounds cross-channel compensation and restricts arbitrary coupling. With the transform parameters held fixed, positive diagonal entries ensure positive selfchannel derivatives in the linear log-LMS core and a nonsingular transformation matrix:

$$
d e t ( T ) = d _ { 1 } d _ { 2 } d _ { 3 } > 0 ,\tag{9}
$$

We apply the transform in log-LMS space and clip the resulting RGB output to [0,1]:

$$
\begin{array} { r } { L _ { c } = T L + b , \qquad } \\ { \Bigl \{ I _ { c } = c l i p _ { [ 0 , 1 ] } \bigl ( { \cal M } _ { L M S \to R G B } e x p \bigl [ c l i p _ { [ - 9 , 4 ] } ( L _ { c } ) \bigr ] \bigr ) . } \end{array}\tag{10}
$$

Here, $b = ( b _ { 1 } , b _ { 2 } , b _ { 3 } ) ^ { T }$ is shared across spatial locations, and clipping is element-wise. The fixed RGB-to-LMS matrix operates on normalized RGB values without inverse-gamma correction; $M _ { L M S  R G B }$ is its numerical inverse. We initialize the final MLP layer to produce an identity transform and zero bias, so MCT starts with minimal changes to the input colors before learning image-dependent corrections. Monotonicity applies to the forward self-channel response of the linear core in log-LMS space when its parameters are held fixed.

## C. U-Shaped Spatial Restoration Backbone

MCT primarily addresses global color imbalance and does not explicitly model spatial restoration associated with scattering and blur. A compact U-shaped encoder-decoder, with three downsampling stages and four spatial scales, handles spatial restoration. It extracts multi-scale features from the color-corrected image and produces full-resolution features for UGWR and bottleneck features for RADR.

![](images/35e175077993025b539ef3c632c0cac018fc02f41c50c50c4eec67b7b30a3e19.jpg)  
Fig. 3. Structure of the UGWR module. The color-corrected image $I _ { c }$ is decomposed into low- and high-frequency wavelet subbands, while the pooled full-resolution feature $\bar { \boldsymbol { F } }$ predicts low-frequency residuals, highfrequency residuals, and confidence scores. The confidence gates highfrequency residual corrections and is calibrated during training using reference-detail errors.

The default encoder channel widths are �, 2�, 4� and 8�, with $C = 3 2$ . Their spatial resolutions are

$$
H \times W , \frac { H } { 2 } \times \frac { W } { 2 } , \frac { H } { 4 } \times \frac { W } { 4 } , \frac { H } { 8 } \times \frac { W } { 8 } .\tag{11}
$$

Each stage comprises Residual Depthwise Blocks. For an input feature $X ,$ , a restoration block computes

$$
Y = X + \alpha \odot P \left[ \phi \left( D w \left( \phi \left( E _ { \exp \left( N ( X ) \right) } \right) \right) \right) \right] .\tag{12}
$$

where � is the output feature, and ⊙ denotes element-wise multiplication. �(·) denotes Group Normalization; �\_���(·), $D w ( \cdot )$ , and $P ( \cdot )$ denote the 1×1 channel expansion, 3×3 depthwise convolution, and 1×1 channel projection, respectively; �(·) is GELU; and α is a learnable per-channel residual scale initialized to 0.1.

The encoder produces multi-scale features $E _ { 1 } , E _ { 2 }$ , and $E _ { 3 }$ and the bottleneck feature �. Let $D _ { 4 } = Z$ . For $i = 3 , 2 , 1$ , the decoder upsamples the lower-resolution feature to the corresponding scale, concatenates it with encoder feature $E _ { i } ,$ and applies a 1×1 convolution followed by a residual restoration stage:

$$
D _ { i } = \mathcal { B } _ { i } ( \mathrm { C o n v } _ { 1 \times 1 } [ U p ( D _ { i + 1 } ) , E _ { i } ] ) .\tag{13}
$$

��(·) denotes bilinear upsampling followed by a 3×3 convolution, whereas $\mathcal { B } _ { i }$ denotes the residual restoration stage at scale �.

The backbone therefore produces two features with distinct roles:

$$
F \in \mathbb { R } ^ { C \times H \times W } , \quad Z \in \mathbb { R } ^ { 8 C \times \frac { H } { 8 } \times \frac { W } { 8 } } .\tag{14}
$$

UGWR uses the full-resolution feature � to predict wavelet corrections, while RADR estimates shared degradation parameters from the bottleneck feature � . Both spatial recovery and degradation estimation therefore use the same backbone, without an additional encoder.

## D. Uncertainty-Gated Wavelet Reconstruction

High-frequency components in underwater images contain both scene details and responses to noise or suspended particles. UGWR predicts candidate wavelet corrections, then controls the fraction applied at each spatial location and wavelet direction (Fig. 3). The learned gate is calibrated against normalized detail-coefficient reconstruction errors, defining confidence in terms of local reconstruction quality. The network thus separates candidate prediction from magnitude control, while training both operations jointly.

## 1) Wavelet Decomposition

A one-level reversible Haar transform decomposes the MCT output $I _ { c }$ into one low-frequency and three directional high-frequency subbands. For each color channel, a, b, c, and d denote the top-left, top-right, bottom-left, and bottom-right values of a 2×2 patch, respectively.

$$
\left\{ \begin{array} { c } { L L = \frac { a + b + c + d } { 2 } , } \\ { L H = \frac { - a - b + c + d } { 2 } , } \\ { H L = \frac { - a + b - c + d } { 2 } , } \\ { H H = \frac { a - b - c + d } { 2 } . } \end{array} \right.\tag{15}
$$

We collect the three high-frequency subbands as

$$
\begin{array} { r } { D = [ L H , H L , H H ] , ( L L , D ) = \mathcal { W } ( I _ { c } ) . } \end{array}\tag{16}
$$

Here, �� mainly represents low-frequency color and largescale structure, while � retains directional edges and local detail.

## 2) Confidence-Gated Reconstruction

Adaptive average pooling ����(·) reduces the fullresolution backbone feature to the wavelet-subband resolution:

$$
F _ { w } = p o o l ( F ) .\tag{17}
$$

Three independent 3×3 convolutional heads predict lowfrequency residuals, high-frequency residuals, and surrogate gate scores, respectively:

$$
\left\{ \begin{array} { c } { \varDelta L L = \mathcal { H } _ { L } ( F _ { w } ) , } \\ { \varDelta D = \mathcal { H } _ { D } ( F _ { w } ) , } \\ { U = c l i p _ { [ - 6 , 6 ] } \bigl ( \mathcal { H } _ { U } ( F _ { w } ) \bigr ) . } \end{array} \right.\tag{18}
$$

where $\varDelta L L \in \mathbb { R } ^ { 3 \times h \times w }$ and $D , \Delta D , U , Q \in \mathbb { R } ^ { 3 \times 3 \times h \times w }$ We arrange the high-frequency residuals and gate scores along the RGB-channel and wavelet-direction axes. The predicted scores are clipped to [-6,6] to obtain �, which controls the strength of each high-frequency correction. The corresponding confidence � is

$$
Q = \sigma ( - U ) .\tag{19}
$$

where � denotes the sigmoid function. Clipping limits � to approximately [0.0025, 0.9975], allowing high-frequency residuals to be strongly suppressed or almost fully retained. Lower gate scores increase �; higher scores decrease � and suppress the candidate residual at that location. We then update the low- and high-frequency components:

$$
\widehat { L L } = L L + \varDelta L L , \widehat { D } = D + Q \odot \varDelta D .\tag{20}
$$

We apply the low-frequency residual directly and gate the high-frequency residual by confidence. Here, �� is the candidate correction, and � determines the fraction added to each detail coefficient.

Inverse Haar reconstruction produces the enhanced image, which is cropped to the original spatial size and clipped to [0,1]:

$$
\hat { J } = c l i p _ { [ 0 , 1 ] } \big ( C r o p _ { H , W } \big [ { \mathcal { W } } ^ { - 1 } \big ( \widehat { L L } , \widehat { D } \big ) \big ] \big ) .\tag{21}
$$

All three UGWR prediction heads are zero-initialized, giving $\varDelta L L = 0$ and $\varDelta D = 0$ at initialization. The network therefore initially preserves the MCT output approximately and learns the required wavelet corrections during training.

## E. Reference-Anchored Degradation Replay

Paired supervision encourages the enhanced image to approximate its reference, but does not directly test consistency with the degraded observation. Re-degrading the enhanced image with a learned renderer provides an additional constraint. Joint optimization may allow the restoration network and renderer to partially compensate for each other. RADR is designed to reduce this ambiguity by rendering the enhanced image and its paired reference with the same degradation parameters. A predictor maps the bottleneck feature to latent attenuation, background-light, and range proxies. We then use the observed input as the reconstruction target for both rendered images. The reference path constrains the renderer through the paired target, and the prediction path ties the enhanced output to the same estimated degradation.

## 1) Shared Degradation Estimation

Given the bottleneck feature �, a global branch predicts 6- dimensional water parameters and a spatial branch estimates the distance proxy map. The global prediction head computes

$$
\omega = { \mathcal { H } } _ { w a t e r } { \big ( } G A P ( Z ) { \big ) } .\tag{22}
$$

Here, ��� represents global average pooling over the spatial dimensions. Softplus makes the three channel-wise attenuation parameters positive, after which we normalize them by their mean:

$$
\beta = \frac { S o f t p l u s ( \omega _ { 1 : 3 } ) } { M e a n [ S o f t p l u s ( \omega _ { 1 : 3 } ) ] } .\tag{23}
$$

Here, ���� averages the three channel-wise attenuation values. We define the background light and distance proxy as

$$
A = \sigma ( \omega _ { 4 : 6 } ) , \quad d = R e s i z e _ { H , W } \big [ \sigma \big ( \mathcal { H } _ { d } ( Z ) \big ) \big ] .\tag{24}
$$

The operator $\mathcal { H } _ { d }$ represents the spatial range-prediction head. � and � are image-level three-channel vectors, and � is a single-channel spatial map. Resize denotes bilinear interpolation to the input resolution. Thus, � represents relative attenuation across the three RGB channels, while � acts as a spatially varying distance/degradation proxy. These variables are learned through reconstruction consistency and parameterize the replay model. They do not represent independently measured optical properties or metric scene depth.

## 2) Reference-Anchored Replay

These parameters define the transmittance:

$$
t _ { c } ( x , y ) = \exp \bigl ( - \beta _ { c } d ( x , y ) \bigr ) .\tag{25}
$$

where � indexes the RGB channel and (�, �) denotes a spatial location. The range map d is shared across channels. A simplified underwater image formation model then re-renders the image:

$$
\begin{array} { r } { \mathcal { R } ( X ) = X \odot t + A \odot ( 1 - t ) . } \end{array}\tag{26}
$$

Both the enhanced image �<sup>̂</sup> and reference image � are rendered with the same parameter set {�, �, �}, giving two replay paths:

$$
\begin{array} { r } { \tilde { I } _ { p } = \mathcal { R } \big ( \hat { J } ; \beta , A , d \big ) , \tilde { I } _ { r } = \mathcal { R } ( J ; \beta , A , d ) . } \end{array}\tag{27}
$$

Reference replay anchors the degradation predictor to the paired observation; prediction replay links the enhanced output to the same parameters. Both terms update the predictor and its upstream features. The two replay paths therefore share the same degradation parameters. RADR provides auxiliary consistency supervision during training and is omitted at inference.

## F. Training Objective

The training objective combines output fidelity, UGWR confidence calibration, and RADR replay consistency.

## 1) Fidelity and Replay Objectives

We combine Charbonnier loss and SSIM to constrain pixel reconstruction and local structure:

$$
\mathcal { L } _ { \mathrm { f i d } } = \mathcal { L } _ { \mathrm { c h a r } } \big ( \hat { J } , J \big ) + \lambda _ { s } \big [ 1 - \mathrm { S S I M } \big ( \hat { J } , J \big ) \big ] .\tag{28}
$$

where $\lambda _ { s } = 0 . 2$ . We clip the SSIM score in the fidelity loss to [0,1] and define the Charbonnier loss as

$$
\mathcal { L } _ { \mathrm { c h a r } } ( X , Y ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sqrt { ( X _ { i } - Y _ { i } ) ^ { 2 } + \varepsilon ^ { 2 } } , \varepsilon = 1 0 ^ { - 3 } .\tag{29}
$$

where � is the total number of image elements. RADR requires both the prediction and reference replay paths to reconstruct the original observation �:

$$
\mathcal { L } _ { \mathrm { { r e p } } } = \frac { 1 } { 2 } \big [ \mathcal { L } _ { \mathrm { c h a r } } \big ( \tilde { I } _ { p } , I \big ) + \mathcal { L } _ { \mathrm { c h a r } } ( \tilde { I } _ { r } , I ) \big ] .\tag{30}
$$

We give the two terms equal weights, allowing prediction replay and the reference anchor to optimize the degradation variables jointly.

## 2) Confidence Calibration

Eq. (19) bounds �, but the sigmoid parameterization alone does not define a reconstruction-reliability target for the gate. We calibrate the gate against the wavelet-detail reconstruction error relative to the reference image. Under this definition, higher confidence corresponds to smaller errors relative to the adaptive normalization scale.

TABLE I  
QUANTITATIVE COMPARISON OF VARIOUS UIE METHODS ON FOUR FULL-REFERENCE TESTING DATASETS. THE BEST, SECOND-BEST, AND THIRD-BEST RESULTS ARE HIGHLIGHTED IN RED, BLUE, AND GREEN, RESPECTIVELY.
<table><tr><td rowspan="2">Methods</td><td colspan="4">Test-E515</td><td colspan="4">Test-L400</td><td colspan="4">Test-U90</td><td colspan="4">Test-UFO120</td></tr><tr><td colspan="4">PSNR↑ SSIM↑ LPIPS↓ FID↓</td><td colspan="4">PSNR↑ SSIM↑ LPIPS↓ FID↓</td><td colspan="4">PSNR↑ SSIM↑ LPIPS↓ FID↓</td><td colspan="4">PSNR↑ SSIM↑ LPIPS↓ FID↓</td></tr><tr><td>UWCNN (2020 PR)</td><td>17.27</td><td>0.732</td><td>0.381</td><td>68.57</td><td>16.95</td><td>0.744 0.397</td><td></td><td>75.58</td><td>14.94</td><td>0.699</td><td>0.334</td><td>103.09 17.11</td><td>0.701</td><td>0.383</td><td>94.82</td></tr><tr><td>WaterNet (2020 TIP)</td><td>20.25</td><td>0.805</td><td>0.276</td><td>33.42</td><td>21.91</td><td>0.855 0.237</td><td>40.47</td><td></td><td>22.78 0.881</td><td>0.113</td><td>29.76</td><td>19.70</td><td>0.772</td><td>0.314</td><td>55.59</td></tr><tr><td>Ucolor (2021 TIP)</td><td>19.98</td><td>0.813</td><td>0.272</td><td>34.14</td><td>20.55</td><td>0.835 0.256</td><td>41.61</td><td>22.34</td><td>0.868</td><td>0.118</td><td>33.66</td><td>19.95</td><td>0.780</td><td>0.311</td><td>53.80</td></tr><tr><td>SGUIE-Net (2022 TIP)</td><td>18.65</td><td>0.773</td><td>0.302</td><td>38.64</td><td>20.70</td><td>0.824 0.239</td><td>43.42</td><td>23.52</td><td>0.900</td><td>0.106</td><td>31.03</td><td>19.17</td><td>0.763</td><td>0.288</td><td>54.42</td></tr><tr><td>TACL (2022 TIP)</td><td>18.93</td><td>0.767</td><td>0.264</td><td>36.57</td><td>20.64</td><td>0.835 0.222</td><td>40.64</td><td>23.41</td><td>0.883</td><td>0.114</td><td>31.72</td><td>19.14</td><td>0.760</td><td>0.271</td><td>54.35</td></tr><tr><td>U-shape (2023 TIP)</td><td>27.78</td><td>0.861</td><td>0.176</td><td>28.74</td><td>23.94</td><td>0.836 0.197</td><td>46.06</td><td>20.51</td><td>0.778</td><td>0.205</td><td>65.24</td><td>23.44</td><td>0.799</td><td>0.221</td><td>54.14</td></tr><tr><td>TUDA (2023 TIP)</td><td>16.84</td><td>0.754</td><td>0.340</td><td>53.57</td><td>18.28</td><td>0.809 0.288</td><td>63.30</td><td>18.93</td><td>0.856</td><td>0.193</td><td>72.58</td><td>16.12</td><td>0.714</td><td>0.370</td><td>79.31</td></tr><tr><td>NU²Net (2023 AAAI)</td><td>19.50</td><td>0.800</td><td>0.290</td><td>37.41</td><td>21.35</td><td>0.848 0.243</td><td>43.87</td><td>23.08</td><td>0.904</td><td>0.114</td><td>31.23</td><td>24.39</td><td>0.816</td><td>0.273</td><td>55.47</td></tr><tr><td>P2CNet (2024 TCSVT)</td><td>17.47</td><td>0.747</td><td>0.332</td><td>43.03</td><td>17.90</td><td>0.774 0.299</td><td>51.26</td><td></td><td>19.51 0.823</td><td>0.193</td><td>55.23</td><td>17.48</td><td>0.725</td><td>0.341</td><td>68.47</td></tr><tr><td>CCL-Net (2025 TMM)</td><td>24.57</td><td>0.822</td><td>0.161</td><td>33.08</td><td>24.84</td><td>0.872 0.142</td><td>35.51</td><td>21.80</td><td>0.886</td><td>0.143</td><td>40.65</td><td>24.85</td><td>0.814</td><td>0.139</td><td>50.93</td></tr><tr><td>SS-UIE (2025 AAAI)</td><td>26.56</td><td>0.858</td><td>0.188</td><td>27.04</td><td>27.52</td><td>0.878</td><td>0.163 39.67</td><td></td><td>21.36 0.858</td><td>0.156</td><td>44.32</td><td>26.23</td><td>0.842</td><td>0.174</td><td>41.61</td></tr><tr><td>PSPL (2025 TIP)</td><td>16.36</td><td>0.666</td><td>0.373</td><td>82.24</td><td>18.27</td><td>0.743</td><td>0.312 76.43</td><td></td><td>19.10 0.821</td><td>0.237</td><td>75.74</td><td>16.62</td><td>0.660</td><td>0.341</td><td>68.47</td></tr><tr><td>GHS-UIR (2025 TCSVT)</td><td>24.63</td><td>0.849</td><td>0.149</td><td>30.18</td><td>25.45</td><td>0.872</td><td>0.119 38.73</td><td></td><td>23.48 0.893</td><td>0.109</td><td>30.50</td><td>24.54</td><td>0.818</td><td>0.149</td><td>52.64</td></tr><tr><td>SUCode (2026 TGRS)</td><td>18.69</td><td>0.795</td><td>0.271</td><td>35.08</td><td>19.86</td><td>0.822</td><td>0.230</td><td>41.76</td><td>18.49</td><td>0.762 0.262</td><td>56.14</td><td>21.14</td><td>0.844</td><td>0.158</td><td>39.90</td></tr><tr><td>HRMamba (2026 TIP)</td><td>27.11</td><td>0.868</td><td>0.192</td><td>27.38</td><td>28.04</td><td>0.909</td><td>0.139</td><td>28.92</td><td>21.85</td><td>0.878 0.126</td><td>34.34</td><td>26.93</td><td>0.844</td><td>0.204</td><td>46.84</td></tr><tr><td>MCL-UIE (Ours)</td><td>28.53</td><td>0.891</td><td>0.143</td><td>24.26</td><td>28.09</td><td>0.917</td><td>0.112</td><td>25.23</td><td>23.37</td><td>0.911</td><td>0.100 25.16</td><td>28.02</td><td>0.868</td><td>0.148</td><td>42.08</td></tr></table>

TABLE II

NO-REFERENCE QUALITY AND COMPUTATIONAL EFFICIENCY COMPARISON OF VARIOUS UIE METHODS. UCCS (AVG.) DENOTES THE AVERAGE PERFORMANCE OVER THE UCCS-BLUE, UCCS-BLUE-GREEN, AND UCCS-GREEN SUBSETS. THE BEST, SECOND-BEST, AND THIRD-BEST RESULTS ARE HIGHLIGHTED IN RED, BLUE, AND GREEN, RESPECTIVELY.
<table><tr><td rowspan="2">Methods</td><td colspan="2">Test-C60</td><td colspan="2">Test-U45</td><td colspan="2">UCCS (Avg.)</td><td colspan="3">Efficiency</td></tr><tr><td></td><td>UIQM↑ UCIQE↑</td><td></td><td>UIQM↑ UCIQE↑</td><td>UIQM↑</td><td>UCIQE↑</td><td>FLOPs (G)</td><td>Params (M)</td><td>GPU Runtime (ms)</td></tr><tr><td>UWCNN (2020 PR) [14]</td><td>2.270</td><td>0.544</td><td>2.755</td><td>0.577</td><td>2.777</td><td>0.507</td><td>2.615</td><td>0.040</td><td>2.927</td></tr><tr><td>WaterNet (2020 TIP) [15]</td><td>2.697</td><td>0.580</td><td>3.187</td><td>0.590</td><td>3.099</td><td>0.569</td><td>71.452</td><td>1.091</td><td>8.827</td></tr><tr><td>Ucolor (2021 TIP) [26]</td><td>2.627</td><td>0.554</td><td>3.196</td><td>0.588</td><td>2.933</td><td>0.536</td><td>443.855</td><td>157.421</td><td>89.310</td></tr><tr><td>SGUIE-Net (2022 TIP) [43]</td><td>2.715</td><td>0.586</td><td>3.188</td><td>0.613</td><td>3.083</td><td>0.550</td><td>70.786</td><td>1.095</td><td>7.174</td></tr><tr><td>TACL (2022 TIP) [44]</td><td>2.745</td><td>0.579</td><td>3.214</td><td>0.625</td><td>3.119</td><td>0.584</td><td>56.864</td><td>11.378</td><td>4.316</td></tr><tr><td>U-shape (2023 TIP) [16]</td><td>2.583</td><td>0.558</td><td>2.978</td><td>0.586</td><td>3.004</td><td>0.556</td><td>66.204</td><td>65.617</td><td>39.275</td></tr><tr><td>TUDA (2023 TIP) [24]</td><td>2.788</td><td>0.582</td><td>3.324</td><td>0.590</td><td>3.149</td><td>0.563</td><td>175.039</td><td>29.876</td><td>26.539</td></tr><tr><td>NU2Net (2023 AAAI) [45]</td><td>2.761</td><td>0.571</td><td>3.243</td><td>0.593</td><td>3.056</td><td>0.551</td><td>10.486</td><td>3.146</td><td>1.550</td></tr><tr><td>P2CNet (2024 TCSVT) [46]</td><td>2.769</td><td>0.598</td><td>3.320</td><td>0.621</td><td>3.085</td><td>0.577</td><td>5.180</td><td>2.068</td><td>2.840</td></tr><tr><td>CCL-Net (2025 TMM) [17]</td><td>2.683</td><td>0.581</td><td>3.310</td><td>0.608</td><td>3.048</td><td>0.560</td><td>108.145</td><td>0.553</td><td>6.346</td></tr><tr><td>SS-UIE (2025 AAAI) [28]</td><td>2.777</td><td>0.579</td><td>3.103</td><td>0.600</td><td>2.993</td><td>0.554</td><td>46.764</td><td>20.634</td><td>51.291</td></tr><tr><td>PSPL (2025 TIP) [29]</td><td>2.752</td><td>0.586</td><td>3.425</td><td>0.597</td><td>3.194</td><td>0.569</td><td>0.976</td><td>0.101</td><td>0.923</td></tr><tr><td>GHS-UIR (2025 TCSVT) [47]</td><td>2.599</td><td>0.574</td><td>3.184</td><td>0.603</td><td>2.979</td><td>0.553</td><td>10.053</td><td>1.145</td><td>55.628</td></tr><tr><td>SUCode (2026 TGRS) [48]</td><td>2.661</td><td>0.568</td><td>3.206</td><td>0.589</td><td>3.013</td><td>0.555</td><td>333.758</td><td>68.220</td><td>93.563</td></tr><tr><td>HRMamba (2026 TIP) [49]</td><td>2.651</td><td>0.566</td><td>3.049</td><td>0.592</td><td>3.013</td><td>0.564</td><td>0.594</td><td>1.067</td><td>15.875</td></tr><tr><td>MCL-UIE (Ours)</td><td>2.521</td><td>0.571</td><td>3.111</td><td>0.602</td><td>2.998</td><td>0.565</td><td>10.098</td><td>2.487</td><td>6.278</td></tr></table>

Applying the Haar transform to the reference image � gives its detail coefficients $D _ { J }$ . We define the current high-frequency reconstruction error as

$$
{ \cal E } = \big | \widehat { D } - D _ { J } \big | .\tag{31}
$$

We detach the error before constructing the calibration target. For each image � and color channel � , the adaptive error scale averages this detached error over the three wavelet directions � and all spatial locations, with a lower bound of 10⁻³:

$$
\begin{array} { c } { \displaystyle \tau _ { n , c } = \displaystyle \operatorname* { m a x } \bigl ( M e a n _ { k , x , y } \bigl [ s g \bigl ( E _ { n , c , k , x , y } \bigr ) \bigr ] , 1 0 ^ { - 3 } \bigr ) , } \\ { \displaystyle Q ^ { * } = \exp \left( - \frac { \mathrm { s g } ( E ) } { \tau } \right) . } \end{array}\tag{32}
$$

where $Q ^ { * }$ is the calibration target and $\tau _ { n , c }$ is broadcast over wavelet directions and spatial locations. ��(·) denotes stopgradient; � is a data-dependent normalization scale, not a learnable parameter. We calibrate the predicted confidence with Smooth L1 loss:

$$
\mathcal { L } _ { \mathrm { c a l } } = \mathrm { S m o o t h L 1 } ( Q , Q ^ { * } ) .\tag{33}
$$

Smaller normalized errors $E / \tau$ yield higher target confidence. The calibration loss trains the gate to follow this error-based target, while the fidelity loss optimizes the enhanced image. Stop-gradient blocks gradients from the calibration target from propagating through the reconstructed detail coefficients. Confidence is therefore a learned weight calibrated to normalized reconstruction error, rather than a probability that a detail coefficient is correct. The objective ablation tests how calibration affects final image quality, both with and without replay supervision.

![](images/0e291a082d27388f9771db19cf678a84fbda6f1cfccca18045dec7eedbdb3ecd.jpg)  
Fig. 4. Full-reference qualitative comparisons on Test-E515, Test-L400, Test-U90, and Test-UFO120. Each representative example is accompanied by an error map relative to the ground truth to visualize restoration deviations.

## 3) Overall Objective

The final objective combines fidelity, calibration, and replay:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { f i d } } + 0 . 1 \big ( \mathcal { L } _ { \mathrm { r e p } } + \mathcal { L } _ { \mathrm { c a l } } \big ) .\tag{34}
$$

## IV. EXPERIMENTS

## A. Experimental Details

We implement MCL-UIE in PyTorch and train it on an NVIDIA GeForce RTX 4090D GPU using AdamW, with an initial learning rate of $2 \times 1 0 ^ { - 4 }$ and weight decay of $1 \times 1 0 ^ { - 4 }$ Training runs for 300 epochs with a batch size of 8, and cosine annealing reduces the learning rate to $2 \times 1 0 ^ { - 6 }$ . We apply the same augmentations to each input-reference pair: random cropping to 256×256, horizontal and vertical flips, and rotations of 90, 180, and 270 degrees. Images smaller than the crop size are resized by bicubic interpolation.

## B. Experimental Settings

## 1) Datasets

We train on two publicly available paired datasets: UIEB [15] and LSUI [16]. UIEB contains 890 underwater images with corresponding references. Following the commonly used split, we use 800 pairs for training and 90 for testing. LSUI contains 4279 input-reference pairs, of which 3879 are used for training and 400 for testing.

We evaluate generalization across underwater environments using paired and no-reference datasets. The paired test sets contain 515 image pairs from EUVP [23], 400 from LSUI, 90 from UIEB, and 120 from UFO-120 [40], denoted as Test-E515, Test-L400, Test-U90, and Test-UFO120, respectively. The no-reference sets comprise 60 challenge images from UIEB and 45 images from U45 [41], denoted as Test-C60 and Test-U45. We also select 300 images from the UCCS subset of RUIE [42]: 100 each from UCCS-blue, UCCS-green, and UCCS-blue-green, forming Test-UCCS-blue, Test-UCCSgreen, and Test-UCCS-blue-green. These datasets cover varied water conditions, color casts, and degradation levels for testing generalization.

![](images/6e94e60f16c6b5f41d1b266d35b476775a036660e7d9b828c869dcd945ea370d.jpg)  
Fig. 5. No-reference qualitative comparisons on Test-C60, Test-U45, and the three RUIE-UCCS subsets (blue, blue-green, and green). The examples illustrate the generalization of MCL-UIE under diverse color casts and real-world underwater degradations.

TABLE III  
COMPONENT-WISE ABLATION ANALYSIS OF MCT, UGWR, AND RADR ON THE TEST-E515, TEST-L400, TEST-U90, AND TEST-UFO120 DATASETS.
<table><tr><td colspan="4">Model</td><td colspan="3">Test-E515</td><td colspan="3">Test-L400</td><td colspan="3">Test-U90</td><td colspan="3">Test-UFO120</td></tr><tr><td>Baseline</td><td>MCT</td><td>UGWR</td><td>RADR</td><td>PSNR↑</td><td></td><td>SSIM↑ LPIPS↓</td><td></td><td>PSNR↑ SSIM↑ LPIPS↓</td><td></td><td></td><td>PSNR↑ SSIM↑ LPIPS↓</td><td></td><td>PSNR↑ SSIM↑ LPIPS↓</td><td></td><td></td></tr><tr><td>√</td><td>X</td><td>X</td><td>X</td><td></td><td>27.64</td><td>0.885</td><td>0.155</td><td>26.85 0.909</td><td>0.126</td><td>22.25</td><td>0.902</td><td>0.116</td><td>26.78</td><td>0.859</td><td>0.167</td></tr><tr><td>√</td><td>√</td><td>X</td><td></td><td>X</td><td>28.14</td><td>0.890</td><td>0.149</td><td>27.15 0.912</td><td>0.120</td><td>23.01</td><td>0.906</td><td>0.106</td><td>27.29</td><td>0.859</td><td>0.159</td></tr><tr><td>√</td><td>√</td><td>√</td><td></td><td>X</td><td>28.34</td><td>0.889</td><td>0.150</td><td>27.32 0.912</td><td>0.121</td><td>23.15</td><td>0.908</td><td>0.104</td><td>27.71</td><td>0.863</td><td>0.155</td></tr><tr><td>√</td><td>√</td><td>√</td><td></td><td>√</td><td>28.53</td><td>0.891</td><td>0.143</td><td>28.09 0.917</td><td>0.112</td><td>23.37</td><td>0.911</td><td>0.100</td><td>28.02</td><td>0.868</td><td>0.148</td></tr></table>

For a compact quantitative comparison, Table II averages the results on UCCS-blue, UCCS-blue-green, and UCCSgreen under UCCS (Avg.). The qualitative comparison presents representative samples from each subset separately.

## 2) Comparative Methods

We compare MCL-UIE with fifteen UIE methods covering different architectures and learning paradigms: UWCNN [14], WaterNet [15], Ucolor [26], SGUIE-Net [43], TACL [44], Ushape [16], TUDA [24], NU<sup>2</sup>Net [45], P2CNet [46], CCL-Net [17], SS-UIE [28], PSPL [29], GHS-UIR [47], SUCode [48], and HRMamba [49]. We use public implementations and official pretrained weights with their original inference settings, while applying a common preprocessing and evaluation protocol to improve comparability across methods.

## 3) Evaluation Metrics

For paired datasets, PSNR measures pixel fidelity, SSIM [50] measures structural fidelity, and LPIPS [51] measures deep perceptual distance. FID [52] evaluates distribution-level similarity to the reference set. For no-reference datasets, we report UIQM [53] and UCIQE [54] as complementary indicators of color, contrast, and sharpness. These metrics do not fully capture perceptual naturalness.

## C. Quantitative Results

## 1) Full-Reference Comparison

Table I compares full-reference results on four paired datasets. MCL-UIE ranks first in 13 of the 16 dataset-metric combinations, including all four metrics on Test-E515 and Test-L400. Compared with the second-best method, it improves PSNR by 0.75 dB on Test-E515 and reduces FID by 3.69 on Test-L400. On Test-U90, it achieves the best SSIM, LPIPS, and FID. Its PSNR of 23.37 dB is 0.15 dB below the best result. On Test-UFO120, it ranks first in PSNR and SSIM, second in LPIPS, and third in FID. These gains span pixel, structural, perceptual, and distribution-level measures, although rankings vary by dataset and metric. Test-UFO120 illustrates this distinction: the best pixel and structural fidelity need not coincide with the best perceptual distance or distributional match.

## 2) No-Reference Comparison

Table II reports no-reference results on Test-C60 and Test-U45, along with averages over UCCS-blue, UCCS-blue-green, and UCCS-green. MCL-UIE is competitive across these conditions, though it does not consistently achieve the highest handcrafted metric scores. It obtains a UCIQE of 0.602 on Test-U45 and an average UIQM/UCIQE of 2.998/0.565 across the three UCCS subsets. UIQM and UCIQE emphasize colorfulness, contrast, and sharpness, so stronger enhancement can increase the scores without improving naturalness. We interpret these results alongside the visual comparisons. They support performance across the evaluated color casts, but the relative quality of individual outputs depends on the metric and visual criterion used.

![](images/cbccb5601dbdf0647f4a2c31695b83be6d2e35989061a702a15d36194adac861.jpg)

![](images/9b369a822f3af40aa21fc92938c180f354f07a44ee78fd362ff638a4708b9e40.jpg)

![](images/08c6d88efd47640ef39eb464c8508a94c8bb02a7e0df46534db1a72472dc25d0.jpg)

![](images/ca2c48821b28d48cbf3b2d64c3c547182d87e94f746b937ac58f60d1382452f2.jpg)  
Input

![](images/f729d1e02866b0b20543aeb48a291fd7a60a7686479c615ad9d120d8b08a8091.jpg)  
Baseline

![](images/93f54c9a19997fa20b725825d8e88a81af231a175a13ca69d0423a41d2aae0e6.jpg)

![](images/566e09085e2ccb43dcf29abcddbdc75ff3e93536a67d3d5e51cd1edc0530b41f.jpg)  
+MCT

![](images/6cc2c8a301a27f182848bee34f6e172366f1c89b3735eeb89999d002b8e3c725.jpg)

![](images/5dd6adeeefb9ddf32674b5ad1234e4207643959f635492c485ff85ffc15a9409.jpg)  
+MCT+UGWR

![](images/c7df55e61b2ef868fffd7a71adec4e84e6c1bcc0be2c54064e9d7fa2327289e6.jpg)

![](images/da96000b4f57a496adfb8e66607bf52b789a9bc32f1463789a68fc623754e0eb.jpg)  
Full Model

![](images/381133d8aaece2cfb4d4c00db8795e9b4fc5d077c4b9bb20d146d8092c05e46b.jpg)  
Ground Truth  
Fig. 6. Component-wise qualitative ablation and RGB distribution analysis. Outputs of the baseline and progressive MCL-UIE variants are compared with the ground truth to illustrate the effects of MCT, UGWR, and RADR on restoration quality and global color distribution.

## D. Model Complexity Analysis

Table II also compares FLOPs, parameter counts, and GPU inference latency with representative UIE methods. For a 256 × 256 input, MCL-UIE has 2.487M parameters, requires 10.098 GFLOPs, and takes 6.278 ms on an RTX 4090D GPU. Several lightweight methods cost less to run. At its reported computational cost, MCL-UIE achieves the highest PSNR and SSIM on Test-E515, Test-L400, and Test-UFO120, and the best SSIM, LPIPS, and FID on Test-U90. These results describe the trade-off between restoration fidelity and computation. RADR contributes its consistency constraint during training and is absent at inference, so the reported costs cover only the restoration path.

## E. Qualitative Results

## 1) Full-Reference Comparison

Fig. 4 presents one representative example from each paired test set with the corresponding error maps. Deeper red or blue indicates a larger positive or negative deviation from the reference images. In the selected examples, MCL-UIE exhibits relatively small color deviation, and its error map is closer to white than those of other methods. It also achieves the highest PSNR and SSIM in all four examples. On Test-E515 and Test-L400, it restores fish and coral colors while preserving a natural water background and balanced lighting. On Test-U90, it retains the contours and fine textures of the diver and fish without noticeable color artifacts. On Test-UFO120, it recovers foreground objects and low-contrast background detail, producing results close to the reference. Competing methods show residual blue or green casts, excessive red enhancement, uneven brightness, or lost local detail. The examples show improvements in both global color and local structure. Component ablations examine the contribution of each constraint to these changes.

## 2) No-Reference Comparison

Fig. 5 compares results on Test-C60, Test-U45, and representative samples from UCCS-blue, UCCS-blue-green, and UCCS-green. We show the UCCS subsets separately to examine different color casts, although their scores are averaged in Table II. MCL-UIE improves visibility in the kelp, shipwreck, starfish, and seabed scenes while retaining relatively natural color and contrast. Some competing methods produce stronger saturation or contrast and higher UIQM or UCIQE scores, but also introduce warm color biases or unnatural local tones in these examples. MCL-UIE yields more restrained results across blue, green, and mixed casts. These comparisons show where higher metric scores can diverge from visual naturalness. Because paired references are unavailable, the observations describe visible restoration behavior rather than fidelity to an unknown clean image.

## F. Ablation Study

We evaluate MCT, UGWR, and RADR through progressive component-wise ablations. The baseline in Table III uses the same U-shaped restoration backbone with all three constraints disabled. Sequentially adding MCT, UGWR, and RADR gives four configurations, trained under identical settings and evaluated on the same four test sets. Each comparison measures the gain over the preceding configuration. The backbone remains fixed throughout, but each measured gain depends on the components already present. We also ablate the training objectives to examine the separate and combined effects of calibration and replay.

## 1) Effectiveness of MCT

Adding MCT to the baseline improves PSNR by 0.30 to 0.76 dB across the four test sets, with an average gain of 0.52 dB. SSIM increases by 0.003 on average, and LPIPS decreases by 0.008. In Fig. 6, MCT shifts the RGB intensity distributions closer to those of the reference images and reduces the channel imbalance in the baseline outputs. These distribution changes agree with the reconstruction gains, supporting explicit global color correction before spatial restoration. The gains apply to MCT as a complete module, including its statistical descriptor, triangular parameterization, and coefficient bounds.

TABLE IV  
ABLATION OF AUXILIARY TRAINING OBJECTIVES ON FOUR FULL-REFERENCE TEST SETS. THE FIDELITY LOSS IS RETAINED IN ALL VARIANTS.
<table><tr><td colspan="3">Model</td><td colspan="3">Test-E515</td><td colspan="3">Test-L400</td><td colspan="3">Test-U90</td><td colspan="3">Test-UFO120</td></tr><tr><td> $L _ { f i d }$ </td><td> $L _ { c a l }$ </td><td> $L _ { r e p }$ </td><td></td><td>PSNR↑ SSIM↑ LPIPS↓</td><td></td><td></td><td>PSNR↑ SSIM↑ LPIPS↓</td><td></td><td>PSNR↑ SSIM↑ LPIPS↓</td><td></td><td></td><td>PSNR↑ SSIM↑ LPIPS↓</td><td></td><td></td></tr><tr><td> $\surd$ </td><td> $\times$ </td><td>X</td><td>27.93</td><td>0.887</td><td>0.149</td><td>28.04</td><td>0.916</td><td>0.109</td><td>23.35</td><td>0.911</td><td>0.100</td><td></td><td>27.71</td><td>0.865 0.151</td></tr><tr><td> $\surd$ </td><td>√</td><td>X</td><td></td><td>26.95 0.878</td><td>0.163</td><td>27.59</td><td>0.913</td><td>0.117</td><td>23.17</td><td>0.907</td><td>0.103</td><td>27.03</td><td>0.858</td><td>0.161</td></tr><tr><td> $\surd$ </td><td>X</td><td> $\surd$ </td><td></td><td>28.47 0.891</td><td>0.144</td><td>27.88</td><td>0.917</td><td>0.109</td><td>23.26</td><td>0.911</td><td>0.101</td><td>27.97</td><td>0.867</td><td>0.149</td></tr><tr><td> $\surd$ </td><td>√</td><td></td><td>√</td><td>28.53 0.891</td><td>0.143</td><td>28.09</td><td>0.917</td><td>0.112</td><td>23.37</td><td>0.911</td><td>0.100</td><td>28.02</td><td>0.868</td><td>0.148</td></tr></table>

(a) Input  
![](images/1016b0fe9df8fdc95f410a1775ca5ece7c39a4fca8129df069dfebd485905027.jpg)

![](images/d7cf83f0a323dbe4e323633d6a237ce02e8bd580a0ee31e4dba224294109a5cc.jpg)  
(b) Enhanced

![](images/3954eaf6ef8dd714e9fdf0d0c6e9daeb4173b4f6b591b1bca50b9930c49aeeb9.jpg)  
(c) Ground truth

![](images/28d5e466df15f18dfc31b0bd8667f8114323e94bed916ad34dcb84bdf40a2a5c.jpg)  
(d) UGWR confidence

![](images/c93eed60ecab8d452fba3f4b36e940dd6fdb178f5d40ad76c3aa6dfbff83f84e.jpg)  
(e) Gated HF correction

![](images/1f36588dd9134f500faa41c2a16556aeb9fbf8c09483f23b359f64f6835f785b.jpg)  
0.131  
(f) HF reconstruction error  
Mean absolute Haar coefficient difference

Mean confidence Q  
Fig. 7. UGWR confidence-gated high-frequency reconstruction in a selected Test-L400 crop. (a–c) Input, enhanced output, and paired reference. (d) Mean confidence �. (e) Mean magnitude of the applied high-frequency correction |� ⊙ ��|. (f) Mean absolute wavelet-detail error relative to the reference. Maps are averaged over RGB channels and Haar directions.  
![](images/75e632d5be52b4ab231052b2e88b810a021729c9b31f7dd25a0b928a09161da5.jpg)

![](images/f7700346c42fb6df8475674245a3d2069788572b387c7b29d07732976a47550c.jpg)

![](images/d183b1d695388d178bd3ca2fb4ceb70a037baa7b542de8e6cadbeca47a07fdd1.jpg)

![](images/7bf516aa560ce9fbcd660ff6b537f6e2f2314da724610009e17eba6553e6aa10.jpg)

![](images/fead100a9d04211b877d8053990b99b5485f4b15c277f3407bdb9be04a2986be.jpg)

![](images/9816d9592b701c121f7bc9f19071b1cd6f04e242ec9c706ae3c5791192ef9dfc.jpg)

![](images/d87988076af62a2a0d05602a845c34d56cdf146c2bfc8bf027a8ec6f66ca58b5.jpg)

![](images/8fc067a48590c7b6f42a42512ac9f40d2fba67b05814cfefde659133e027937f.jpg)

![](images/40be9373e824f33888022a4adb0ace72449f4c2e29dfa1e7bb723a8a59e03f14.jpg)

![](images/7b82949c34f032c5550fdba96439855d3458ee3fe7f7e6755efbc8a81a55120b.jpg)

![](images/3b41a015b140437df50eb38b7aaf6b6600cefb6faa033033a1032171b0604775.jpg)

![](images/9c5e5d1e11def267a5a5fe0ae8df228e4e1f52f810a48796bc3e235dca43217b.jpg)

![](images/06444ecc91f75efdc112b523bfd1f578e658f5e82bc1a69e4e48f7f281e31e07.jpg)  
Input

![](images/60f10b5ab6089e1ec3a58ff730c97829bf2001de6bd6841172ae5c7acf6578d5.jpg)  
LFid

![](images/de9ada402de0468567d5a80c186dd9a65e1ef1f395e8a13b0151e859ecf616a6.jpg)  
LFid + Lcal

![](images/234d373c6511a57a712da9d759e21c2ba875ed92c44b8f9b1c5106447c2dd3e2.jpg)  
LFid + Lrep

![](images/7b39ae56cd17ae0b59e99bba2d61440f6a0facd29fe6395582fc4965fd067752.jpg)  
Full Loss

![](images/cae1f5681ce4e4dd65451c26b75196f73c8b1660b04b2a58e844011f8bf7df28.jpg)  
GT  
Fig. 8. Qualitative analysis of different training objectives on a representative LSUI sample. The first row shows restoration results under different objective configurations, the second row presents absolute RGB error maps relative to the ground truth, and the third row compares the corresponding RGB intensit distributions.

## 2) Effectiveness of UGWR

Adding UGWR to the MCT configuration further improves PSNR on all four test sets by 0.14 to 0.42 dB. The largest improvement occurs on Test-UFO120, where SSIM rises from 0.859 to 0.863 and LPIPS falls from 0.159 to 0.155. The RGB box plots in Fig. 6 change only moderately with UGWR. This pattern is consistent with better local reconstruction and little change in global color statistics. However, marginal RGB distributions cannot locate the wavelet corrections. The local visualizations therefore give a more direct view of where UGWR changes the image.

Fig. 7 illustrates UGWR’s spatial response in a Test-L400 crop. Compared with the input, the enhanced image renders the coral-like structures more clearly and appears closer to the paired reference. The confidence map varies across the crop, while the gated high-frequency correction is concentrated around textured structures. The wavelet-detail error map shows where discrepancies from the reference remain.

## 3) Overall Effect and Qualitative Analysis

The complete MCL-UIE configuration yields the strongest aggregate performance among the evaluated variants in Table

III. Relative to the baseline, it improves PSNR by 1.12 dB and SSIM by 0.008 on average, while reducing LPIPS by 0.015. Fig. 6 shows natural tones and color distributions close to the references, without noticeable oversaturation. Along the evaluated sequence, global color correction improves the backbone input, wavelet reconstruction adds local corrections, and degradation replay provides additional training supervision. Their combination yields the strongest overall results among the evaluated variants, suggesting that the three constraints are complementary within this architecture.

## 4) Effectiveness of Different Training Objectives

To assess the training objectives, we retain the fidelity loss in every variant and add confidence calibration and degradation replay separately or together. Table IV shows that calibration alone reduces performance under the evaluated settings, whereas replay provides clear gains on Test-E515 and Test-UFO120 and mixed effects on Test-L400 and Test-U90. Combining both objectives yields the highest PSNR on all four datasets: 28.53, 28.09, 23.37, and 28.02 dB on Test-E515, Test-L400, Test-U90, and Test-UFO120, respectively. The combination also achieves the best or joint-best SSIM. LPIPS increases slightly on Test-L400, but reaches the best or jointbest value on the other datasets. Calibration improves reconstruction when combined with replay. Used alone, it reduces performance under this protocol. The two objectives constrain different aspects of restoration: calibration trains a local error-based gate, while replay enforces consistency with the observed image through shared degradation parameters. Their combined improvement supports supervising both local corrections and observation consistency.

Fig. 8 compares the training objectives on a representative Test-L400 sample. Replay supervision reduces RGB MAE from 0.051 to 0.041, while calibration alone raises it to 0.054. The complete objective gives the lowest MAE of 0.027, accompanied by a lower-intensity error map and an RGB distribution closer to the reference images. The sample follows the dataset-level pattern, in which replay improves reconstruction on its own and calibration provides an additional benefit when combined with replay. The error map and RGB distribution show how this quantitative improvement appears in the reconstructed image.

## V. CONCLUSION

We present MCL-UIE, a multi-level constrained framework for underwater image enhancement that extends pairedreference supervision from the final output to intermediate restoration operations and degradation consistency. The proposed MCT constrains global chromatic transformation, UGWR regulates local high-frequency reconstruction through calibrated confidence, and RADR introduces referenceanchored degradation consistency during training. Together, these components provide complementary constraints on global color correction, local detail restoration, and the relation between the enhanced result and the observed underwater image. Experiments on paired and no-reference benchmarks demonstrate competitive restoration performance with a compact inference architecture, while progressive component and objective ablations indicate complementary contributions from the three constraints. In particular, the results suggest that confidence calibration is more effective when combined with replay supervision, highlighting the interaction between local correction control and observationlevel consistency. Since RADR is used only during training, the additional degradation-consistency supervision does not increase inference-time computation. MCL-UIE currently relies on paired references during training, and the latent attenuation, background-light, and range variables in RADR are learned consistency proxies rather than directly measured physical quantities. Future work may investigate weaker supervision, more flexible degradation models, and adaptation to diverse real-world underwater environments.

## REFERENCES

[1] S.-B. Gao, M. Zhang, Q. Zhao, X.-S. Zhang, and Y.-J. Li, “Underwater image enhancement using adaptive retinal mechanisms,” IEEE Trans. Image Process., vol. 28, no. 11, pp. 5580–5595, Nov. 2019.

[2] Q. Jiang, Y. Gu, Z. Wu, C. Li, H. Xiong, F. Shao, and Z. Wang, “Deep underwater image quality assessment with explicit degradation awareness embedding,” IEEE Trans. Image Process., vol. 34, pp. 1297–1310, 2025.

[3] R. Cong, W. Yang, W. Zhang, C. Li, C.-L. Guo, Q. Huang, and S. Kwong, “PUGAN: Physical model-guided underwater image enhancement using GAN with dual-discriminators,” IEEE Trans. Image Process., vol. 32, pp. 4472–4485, 2023.

[4] Z. Liang, X. Ding, Y. Wang, X. Yan, and X. Fu, “GUDCP: Generalization of underwater dark channel prior for underwater image restoration,” IEEE Trans. Circuits Syst. Video Technol., vol. 32, no. 7, pp. 4879–4884, Jul. 2022.

[5] B. Lin, F. Gao, Y. Yu, J. Dong, and Q. Du, “Downstream task-inspired underwater image enhancement: A perception-aware study from dataset construction to network design,” IEEE Trans. Image Process., vol. 35, pp. 2873–2888, 2026.

[6] Y. Zhang, W. Bao, Y. Yang, W. Wan, Q. Xiao, Z. Zhang, and X. Zou, “FSGNet: A frequency-aware and semantic guidance network for infrared small target detection,” Expert Syst. Appl., vol. 328, Art. no. 132925, Oct. 2026.

[7] Y. Zhang, W. Bao, Y. Yang, W. Wan, Q. Xiao, and X. Zou, “MPCNet: Multiscale perception and cross-attention feature fusion network for infrared small target detection,” IEEE Trans. Geosci. Remote Sens., vol. 64, pp. 1–15, 2026.

[8] Y. Zhang, W. Bao, Y. Yang, W. Wan, Q. Xiao, and X. Zou, “HAFNet: Hierarchical attention fusion network for infrared small target detection,” IEEE Trans. Geosci. Remote Sens., vol. 63, pp. 1–16, 2025.

[9] Y. Zhang, W. Bao, W. Wan, Q. Xiao, Y. Tang, X. Zou, L. Huang, K. Zhong, and Y. Lan, “APTNet: Adaptive partial transformer network for infrared small target detection,” IEEE Sensors J., vol. 25, no. 10, pp. 17960–17974, May 2025.

[10] C.-Y. Li, J.-C. Guo, R.-M. Cong, Y.-W. Pang, and B. Wang, “Underwater image enhancement by dehazing with minimum information loss and histogram distribution prior,” IEEE Trans. Image Process., vol. 25, no. 12, pp. 5664–5677, Dec. 2016.

[11] C. O. Ancuti, C. Ancuti, C. De Vleeschouwer, and P. Bekaert, “Color balance and fusion for underwater image enhancement,” IEEE Trans. Image Process., vol. 27, no. 1, pp. 379–393, Jan. 2018.

[12] W. Zhang, P. Zhuang, H.-H. Sun, G. Li, S. Kwong, and C. Li, “Underwater image enhancement via minimal color loss and locally adaptive contrast enhancement,” IEEE Trans. Image Process., vol. 31, pp. 3997–4010, 2022.

[13] J. Yuan, W. Cao, Z. Cai, and B. Su, “An underwater image vision enhancement algorithm based on contour bougie morphology,” IEEE Trans. Geosci. Remote Sens., vol. 59, no. 10, pp. 8117–8128, Oct. 2021.

[14] C. Li, S. Anwar, and F. Porikli, “Underwater scene prior inspired deep underwater image and video enhancement,” Pattern Recognit., vol. 98, Art. no. 107038, Feb. 2020.

[15] C. Li, C. Guo, W. Ren, R. Cong, J. Hou, S. Kwong, and D. Tao, “An underwater image enhancement benchmark dataset and beyond,” IEEE Trans. Image Process., vol. 29, pp. 4376–4389, 2020.

[16] L. Peng, C. Zhu, and L. Bian, “U-shape transformer for underwater image enhancement,” IEEE Trans. Image Process., vol. 32, pp. 3066– 3079, 2023.

[17] Y. Liu, Q. Jiang, X. Wang, T. Luo, and J. Zhou, “Underwater image enhancement with cascaded contrastive learning,” IEEE Trans. Multimedia, vol. 27, pp. 1512–1525, 2025.

[18] C. Ancuti, C. O. Ancuti, T. Haber, and P. Bekaert, “Enhancing underwater images and videos by fusion,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2012, pp. 81–88.

[19] X. Fu, P. Zhuang, Y. Huang, Y. Liao, X.-P. Zhang, and X. Ding, “A Retinex-based enhancing approach for single underwater image,” in Proc. IEEE Int. Conf. Image Process. (ICIP), 2014, pp. 4572–4576.

[20] P. Drews, Jr., E. R. Nascimento, F. Moraes, S. S. C. Botelho, and M. F. M. Campos, “Transmission estimation in underwater single images,” in Proc. IEEE Int. Conf. Comput. Vis. Workshops (ICCVW), 2013, pp. 825–830.

[21] A. Galdran, D. Pardo, A. Picón, and A. Alvarez-Gila, “Automatic redchannel underwater image restoration,” J. Vis. Commun. Image Represent., vol. 26, pp. 132–145, Jan. 2015.

[22] Y.-T. Peng, K. Cao, and P. C. Cosman, “Generalization of the dark channel prior for single image restoration,” IEEE Trans. Image Process., vol. 27, no. 6, pp. 2856–2868, Jun. 2018.

[23] M. J. Islam, Y. Xia, and J. Sattar, “Fast underwater image enhancement for improved visual perception,” IEEE Robot. Autom. Lett., vol. 5, no. 2, pp. 3227–3234, Apr. 2020.

[24] Z. Wang, L. Shen, M. Xu, M. Yu, K. Wang, and Y. Lin, “Domain adaptation for underwater image enhancement,” IEEE Trans. Image Process., vol. 32, pp. 1442–1457, 2023.

[25] Y. Wang, J. Guo, H. Gao, and H. Yue, “UIEC²-Net: CNN-based underwater image enhancement using two color space,” Signal Process. Image Commun., vol. 96, Art. no. 116250, Aug. 2021.

[26] C. Li, S. Anwar, J. Hou, R. Cong, C. Guo, and W. Ren, “Underwater image enhancement via medium transmission-guided multi-color space embedding,” IEEE Trans. Image Process., vol. 30, pp. 4985–5000, 2021.

[27] Z. Fu, W. Wang, Y. Huang, X. Ding, and K.-K. Ma, “Uncertainty inspired underwater image enhancement,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2022, pp. 465–482.

[28] L. Peng and L. Bian, “Adaptive dual-domain learning for underwater image enhancement,” in Proc. AAAI Conf. Artif. Intell. (AAAI), vol. 39, no. 6, 2025, pp. 6461–6469.

[29] Y. Liu, Q. Jiang, X. Li, T. Luo, and W. Ren, “Toward better than pseudo-reference in underwater image enhancement,” IEEE Trans. Image Process., vol. 34, pp. 6168–6179, 2025.

[30] J. Zhou, D. Zhang, Z. He, Q. Gai, and Q. Jiang, “Multi-prior fusion transfer plugin for adapting in-air models to underwater image enhancement and detection,” IEEE Trans. Image Process., vol. 34, pp. 7773–7785, 2025.

[31] Z. Tan, C. Fu, T. Guo, Z. Nan, P. Zhou, X. Peng, and F. Luo, “PGMamba: A physical model-guided global Mamba for underwater image enhancement,” in Proc. AAAI Conf. Artif. Intell. (AAAI), vol. 40, no. 11, 2026, pp. 9359–9367.

[32] Z. Ma and C. Oh, “A wavelet-based dual-stream network for underwater image enhancement,” in Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP), 2022, pp. 2769–2773.

[33] Y. Wei, Z. Zheng, and X. Jia, “UHD underwater image enhancement via frequency-spatial domain aware network,” in Proc. Asian Conf. Comput. Vis. (ACCV), 2022, pp. 299–314.

[34] C. Zhao, W. Cai, C. Dong, and Z. Zeng, “Toward sufficient spatialfrequency interaction for gradient-aware underwater image enhancement,” in Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP), 2024, pp. 3220–3224.

[35] Z. Cheng, G. Fan, J. Zhou, M. Gan, and C. L. P. Chen, “FDCE-Net: Underwater image enhancement with embedding frequency and dual color encoder,” IEEE Trans. Circuits Syst. Video Technol., vol. 35, no. 2, pp. 1728–1744, Feb. 2025.

[36] H. Zhang, H. Xu, X. Yu, X. Zhang, X. Gao, and C. Wu, “CDF-UIE: Leveraging cross-domain fusion for underwater image enhancement,” IEEE Trans. Geosci. Remote Sens., vol. 63, pp. 1–15, 2025.

[37] P. Mishra, N. Mehta, S. K. Vipparthi, and S. Murala, “USWformer: Efficient sparse wavelet transformer for underwater image enhancement,” in Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV), 2025, pp. 3372–3382.

[38] C.-H. Cheng, J.-W. Lee, C.-M. Lee, and C.-C. Hsu, “WWE-UIE: A wavelet & white balance efficient network for underwater image enhancement,” in Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV), 2026, pp. 2135–2145.

[39] N. Li, X. Li, S. Wang, and F. Cui, “Underwater image enhancement by combining wavelet transform and multiscale transformer networks,” Digit. Signal Process., vol. 168, pt. E, Art. no. 105671, Jan. 2026.

[40] M. J. Islam, P. Luo, and J. Sattar, “Simultaneous enhancement and super-resolution of underwater imagery for improved visual perception,” in Proc. Robot. Sci. Syst. (RSS), 2020, doi: 10.15607/RSS.2020.XVI.018.

[41] H. Li and P. Zhuang, “DewaterNet: A fusion adversarial real underwater image enhancement network,” Signal Process. Image Commun., vol. 95, Art. no. 116248, Jul. 2021.

[42] R. Liu, X. Fan, M. Zhu, M. Hou, and Z. Luo, “Real-world underwater enhancement: Challenges, benchmarks, and solutions under natural light,” IEEE Trans. Circuits Syst. Video Technol., vol. 30, no. 12, pp. 4861–4875, Dec. 2020.

[43] Q. Qi, K. Li, H. Zheng, X. Gao, G. Hou, and K. Sun, “SGUIE-Net: Semantic attention guided underwater image enhancement with multiscale perception,” IEEE Trans. Image Process., vol. 31, pp. 6816–6830, 2022.

[44] R. Liu, Z. Jiang, S. Yang, and X. Fan, “Twin adversarial contrastive learning for underwater image enhancement and beyond,” IEEE Trans. Image Process., vol. 31, pp. 4922–4936, 2022.

[45] C. Guo, R. Wu, X. Jin, L. Han, W. Zhang, Z. Chai, and C. Li, “Underwater Ranker: Learn which is better and how to be better,” in Proc. AAAI Conf. Artif. Intell. (AAAI), vol. 37, no. 1, 2023, pp. 702– 709.

[46] Y. Rao, W. Liu, K. Li, H. Fan, S. Wang, and J. Dong, “Deep color compensation for generalized underwater image enhancement,” IEEE Trans. Circuits Syst. Video Technol., vol. 34, no. 4, pp. 2577–2590, Apr. 2024.

[47] X. Guo, X. Chen, S. Wang, and C.-M. Pun, “Underwater image restoration through a prior guided hybrid sense approach and extensive benchmark analysis,” IEEE Trans. Circuits Syst. Video Technol., vol. 35, no. 5, pp. 4784–4800, May 2025.

[48] B. Lin, F. Gao, Y. Yu, J. Dong, and Q. Du, “Enhancing underwater images via adaptive semantic-aware codebook learning,” IEEE Trans. Geosci. Remote Sens., vol. 64, pp. 1–14, 2026.

[49] Y. Fan, L. Gao, F. Zhou, N. Li, and Y. Huang, “HRMamba: A hybrid Retinex and state-space model for underwater image enhancement,” IEEE Trans. Image Process., vol. 35, pp. 5064–5079, 2026.

[50] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: From error visibility to structural similarity,” IEEE Trans. Image Process., vol. 13, no. 4, pp. 600–612, Apr. 2004.

[51] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018, pp. 586–595.

[52] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, “GANs trained by a two time-scale update rule converge to a local Nash equilibrium,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 30, 2017, pp. 6626–6637.

[53] K. Panetta, C. Gao, and S. Agaian, “Human-visual-system-inspired underwater image quality measures,” IEEE J. Ocean. Eng., vol. 41, no. 3, pp. 541–551, Jul. 2016.

[54] M. Yang and A. Sowmya, “An underwater color image quality evaluation metric,” IEEE Trans. Image Process., vol. 24, no. 12, pp. 6062–6071, Dec. 2015.