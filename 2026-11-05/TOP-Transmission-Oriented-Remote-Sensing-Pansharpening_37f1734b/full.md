# TOP: Transmission-Oriented Remote Sensing Pansharpening

Zi-Han Cao, Jing-Ze Liang, Shao-Jie Wu, Jin-Cheng Di, Liang-Jian Deng, Senior Member, IEEE, and Gemine Vivone, Senior Member, IEEE

Abstract—Pansharpening has been extensively studied as a fundamental image fusion task in remote sensing. However, most existing methods focus on improving fusion quality under the implicit assumption that high-resolution observations have already been fully downlinked to the ground. As a result, existing models are highly vulnerable to perturbations introduced by lossy compression used for downlink transmission. To address this limitation, we present, to the best of our knowledge, the first transmission-oriented pansharpening (TOP) framework, which unifies compact latent coding, onboard pansharpening, and efficient downlink transmission within a single satellite-toground pipeline. TOP encodes remote-sensing observations into a compact binary latent space, where a lightweight onboard fusion module can efficiently predict the fused binary latent code. Since only the compact fused code is downlinked, the transmission payload is substantially reduced. Overall, TOP enables low-payload downlink and lightweight onboard processing while maintaining high fusion quality under severe bandwidth constraints. Experiments across multispectral and hyperspectral settings show that TOP consistently achieves superior fusion quality at higher compression ratios than representative pansharpening methods paired with standard lossy codecs. For example, on GaoFen-2, TOP attains a fusion PSNR of 31.4 dB at a downlink compression ratio of 50×. This design also points toward transmissionoriented processing for other remote-sensing tasks. The code will be made available at https://github.com/JingZe-Liang/TOP.git.

Index Terms—Pansharpening, Image fusion, Hyperspectral imaging, Image compression, Onboard processing, Binary quantization

## I. INTRODUCTION

ANSHARPENING reconstructs a high-spatial-resolution multispectral or hyperspectral image (HR-MSI/HR-HSI)   
by combining the spatial structures of a panchromatic (PAN)   
observation with the spectral information of a low-spatial  
resolution multispectral or hyperspectral image (LR-MSI/LR-  
HSI), and underpins remote-sensing image interpretation [1],   
[2]. From the explicit detail-injection rules of component  
substitution and multi-resolution-analysis methods [3]–[5] to   
the nonlinear mappings learned by deep networks [6]–[8],

these methods share an implicit assumption: the input observations have already been fully delivered to the ground station and are directly available to the fusion model.

In practice, the PAN and LR-MSI/LR-HSI observations are acquired on board and must pass through bandwidth-limited downlink channels [9]. A conventional deployment therefore compresses and transmits them independently and fuses the decompressed images on the ground (first row of Fig. 1(a)). Under restrictive bitrate budgets, lossy compression introduces spatial blurring, spectral distortion, and quantization artifacts, whereas pansharpening models are commonly trained on clean observations [6]–[8]. This training–deployment mismatch can severely degrade the fusion result (Fig. 1(b)), and the codecbased baselines in Sec. IV-C indeed degrade sharply under constrained downlink budgets. Recent work on degradationrobust fusion [10] further highlights the importance of realistic input perturbations.

Fusing on board (second row of Fig. 1(a)) does not resolve this issue. The satellite must then generate a full HR-MSI/HR-HSI, which is computationally expensive on resource-constrained platforms, and the fused image must still be compressed with a standard codec such as JPEG2000 [11] or High Efficiency Video Coding (HEVC) [12], whose rate– distortion performance then bounds the final quality [9], [13].

These limitations motivate a different question: rather than deciding whether to compress images before or after pansharpening, can transmission and pansharpening, which have so far been handled separately, benefit from each other? To this end, we propose, to the best of our knowledge, the first transmission-oriented pansharpening (TOP) framework, which jointly designs compact latent coding, onboard pansharpening, and downlink transmission. TOP maps the PAN and LR-MSI/LR-HSI observations into binary latent codes and employs a lightweight latent fusion module to predict the fused HR-MSI/HR-HSI code on board (third row of Fig. 1(a)). Only this code is downlinked; the ground station decodes it into the final HR-MSI/HR-HSI or archives it for later reconstruction. TOP thus shifts the communication target from source images to a compact fused code.

TOP is built on a transmission-oriented binary tokenizer (TOBT), which maps PAN, LR-MSI/LR-HSI, and HR-MSI/HR-HSI observations into a shared latent space through an adaptive embedding for different numbers of bands and a Swin-based encoder [14]. Binary spherical quantization (BSQ) [15] then maps each latent entry to one bit. Besides reducing the payload, this bottleneck turns continuous-latent regression into an easier per-entry binary classification for the latent fusion module (Sec. V-B). The binary code is transmitted directly, without an entropy model [13] or range asymmetric numeral systems (rANS) coding [16] (Sec. V-C), and bit-flip

## a Where fusion happens and what is downlinked

## b Delivered image

![](images/bc4efa3f5654a02f00fecc6b7408e6051a12b337ac9df770fe8ad5199a8a65b1.jpg)

![](images/4d26304a489e1ee9a8484f260ed3d439ae749e382eea01e93f631a48d4124992.jpg)

![](images/973dc9030ea30d66ca0383b3871d95b602b81883cc1179f0796c89a8e3468918.jpg)

![](images/8f5e96072b557e59691cb7837416f329eb2bf2b511a756bc0e4857faf01884e6.jpg)  
Fig. 1. TOP compared with existing deployment paths for pansharpening. (a) Three ways of delivering a pansharpened image to the ground, with compression ratios and PSNR ranges from WorldView-3 (WV3). (b) Ground truth, the best compress-then-fuse baseline under JPEG2000, and TOP on WV3, GaoFen-2 (GF2), and Pavia Center. (c) PSNR versus downlink compression ratio for methods under HEVC or JPEG2000; onboard fusion followed by compression is evaluated on WV3 only (Sec. S9 of the Supplementary Material).

augmentation makes the ground-side decoder robust to latent prediction and transmission errors.

We compare TOP with representative pansharpening methods whose inputs are compressed by HEVC or JPEG2000 before ground-side fusion, on the WorldView-3 (WV3) and GaoFen-2 (GF2) multispectral datasets and the Pavia Center hyperspectral dataset (hereafter Pavia). On all three datasets, TOP delivers a higher PSNR than every compressed baseline while operating at a higher compression ratio (Fig. 1(c)); on WV3, for example, it reaches 30.32 dB at 33×, whereas the strongest baseline reaches 27.70 dB at a lower ratio. At these ratios, 1 TB of source observations corresponds to roughly 15.2 GB of downlinked data on WV3 (at 66×), 20.0 GB on GF2, and 41.7 GB on Pavia. Our contributions are summarized as follows:

• We firstly introduce and formalize transmission-oriented pansharpening, a system-level design that jointly considers compact binary coding, onboard fusion, downlink communication, and ground-side reconstruction instead of assuming fully available lossless observations.

• We build TOP, which performs pansharpening in a compact binary latent space and transmits only the fused HR-MSI/HR-HSI code; its BSQ bottleneck yields a code that is reconstructable, directly transmissible, and predictable for the latent fusion module.

• We conduct experiments on multispectral and hyperspectral datasets, showing that TOP achieves superior fusion quality at substantially higher compression ratios than representative pansharpening methods paired with codecs.

## II. RELATED WORK

## A. Pansharpening

Pansharpening fuses the spatial details of a PAN image with the spectral information of an LR-MSI/LR-HSI image [1], [2] and belongs to the broader family of image fusion tasks, which also includes infrared–visible, multi-exposure, and multi-focus fusion [17]–[19]. Traditional methods are based on component substitution, such as GS [3], BDSD [4], GSA [20], and PRACS [21]; multi-resolution analysis, such as GLP [5], AWLP [22], and morphological filters [23]; or variational, Bayesian, and low-rank formulations [1], [2], [24], [25]. They are efficient and interpretable, but predefined models may cause spatial or spectral distortions. Deep multispectral models have introduced detail injection [6], dual-branch spatial– spectral modeling [7], frequency-aware or adaptive feature extraction [8], [26], [27], degradation-aware modeling [28], and zero-shot adaptation [29]. Hyperspectral pansharpening must further handle high spectral dimensionality and sensor variability, and relies on Transformer-based fusion and equivariant imaging [30], [31], image-specific priors and bandwise propagation [32]–[34], and diffusion-based zero-shot guidance [35], [36]. Degradation-robust fusion methods [10] handle generic low-quality inputs rather than downlink compression artifacts. All these methods assume that the source observations are fully available at the processing platform, whereas TOP performs pansharpening before downlink in a compact binary latent space.

## B. Remote-Sensing Image Compression and Transmission

The growing volume of remote-sensing imagery strains onboard storage and satellite-to-ground communication [9]. Standard codecs such as JPEG2000 [11] and HEVC [12] reduce payloads through transform coding, quantization, and entropy coding. Lossless and near-lossless remote-sensing codecs such as CCSDS 123.0-B-2 [37] target low-complexity, distortionbounded compression rather than the high-compression regime studied here, and are therefore not included in our comparison. Neural image compression instead learns nonlinear transforms with quantized latents under rate–distortion optimization [13]. Existing pipelines combine compression and pansharpening in one of two orders, either compressing the source observations and fusing on the ground or fusing on board and compressing the fused image, and in both cases treat fusion and compression as separate stages. TOP instead fuses within a compact binary latent space and downlinks the fused code as the payload.

## C. Discrete Visual Representation and Quantization

Transmitting a learned latent requires discretizing it. VQ-VAE [38] assigns latent vectors to a learned codebook whose capacity and utilization are hard to control. Finite scalar quantization [39] maps each dimension to fixed scalar levels, lookup-free quantization [40] derives token indices directly from binary states, and BSQ [15] projects latents onto a unit hypersphere before sign quantization. Such binary codes are highly compact and thus well suited to transmission, so TOP adopts BSQ as a shared latent code for all three modalities. Unlike generative models, which predict discrete tokens with autoregressive or masked-token objectives [40], TOP predicts the target binary code conditionally from the PAN and LR-MSI/LR-HSI codes, and this prediction serves as both the downlink payload and the input to the ground-side decoder.

## III. METHODOLOGY

Sec. III-A introduces the notation and the deployment-time pipeline. We then detail the components of TOBT, namely the transmission-oriented binary encoder (TOBE; Sec. III-B), BSQ (Sec. III-C), and the transmission-oriented binary decoder (TOBD; Sec. III-E), together with the latent fusion module (Sec. III-D), the training strategy (Sec. III-F), and the overall formulation of TOP (Sec. III-G). Implementation details are provided in Sec. S3 of the Supplementary Material.

Algorithm 1: Deployment-time inference of TOP.   
Input: PAN observation P; upsampled LR-MSI/LR-HSI observation   
M; trained encoder $\mathcal { E } _ { \boldsymbol { \theta } } ;$ quantizer Q; latent fusion module   
$\mathcal { F } _ { \psi } ;$ ground-side decoder $\mathcal { D } _ { \phi }$   
Output: Reconstructed HR-MSI/HR-HSI image Hb   
▷ Onboard Encoding, Fusion, and Downlink   
1 $\mathbf { B } _ { \mathrm { P } }  \mathcal { Q } ( \mathcal { E } _ { \theta } ( \mathbf { P } ) )$ ▷ Encode PAN   
2 ${ \bf B } _ { \mathrm { M } }  \dot { \mathcal { Q } } ( \mathcal { E } _ { \theta } ( { \bf M } ) )$ ▷ Encode spectral image   
3 $\mathbf { O }  \mathcal { F } _ { \psi } ( \mathrm { C o n c a t } _ { c } ^ { \prime } [ \mathbf { B } _ { \mathrm { M } } , \mathbf { B } _ { \mathrm { P } } ] )$ ▷ Predict binary-state logits   
4 $\widehat { \mathbf { B } } _ { \mathrm { H } } \gets ( 2$ arg max $\mathbf { O } _ { a } - \mathbf { 1 } ) / \sqrt { C _ { \mathrm { l a t } } } \triangleright$ Normalize the fused code   
a∈{0,1}   
5 $\mathbf { S } _ { \mathrm { d o w n } }  \mathrm { B i t P a c k } ( \widehat { \mathbf { B } } _ { \mathrm { H } } )$ ▷ Pack the binary payload   
6 Downlink $\mathbf { S } _ { \mathrm { d o w n } }$ to the ground station   
▷ Ground-Side Reconstruction   
7 Be ← BitUnpack $\mathbf { \Gamma } ( \mathbf { S } _ { \mathrm { d o w n } } )$ ▷ Recover the received fused code   
8 $\widehat { \mathbf { H } }  \mathcal { D } _ { \phi } ( \widetilde { \mathbf { B } } _ { \mathrm { H } } )$ ▷ Decode the fused HR image   
9 return Hb

## A. System Overview

Let $\mathbf { P } \in \mathbb { R } ^ { 1 \times H \times W }$ $\mathbf { M } \in \mathbb { R } ^ { C \times H \times W }$ , and $\mathbf { H } \in \mathbb { R } ^ { C \times H \times W }$ denote the PAN observation, the upsampled LR-MSI/LR-HSI observation, and the target HR-MSI/HR-HSI image, respectively. During deployment, TOP divides pansharpening into onboard latent fusion and ground-side reconstruction (Fig. 2). On board, the shared TOBE ${ \mathcal { E } } _ { \theta }$ and the BSQ quantizer Q encode P and M into normalized binary latent codes $\mathbf { B } _ { \mathrm { P } }$ and ${ \bf { B } } _ { \mathrm { { M } } } .$ , which are concatenated along the channel dimension and processed by the lightweight latent fusion module $\mathcal { F } _ { \psi }$ to predict the fused code $\tilde { \widehat { \mathbf { B } } } _ { \mathrm { H } }$ . During training, this prediction is supervised by $\mathbf { B } _ { \mathrm { H } }$ , the code obtained by encoding H; during inference, $\widehat { \mathbf { B } } _ { \mathrm { H } }$ is directly bit-packed and transmitted. At the ground station, the received code is unpacked as $ { \widetilde { \mathbf { B } } } _ { \mathrm { H } }$ , and the decoder $\mathcal { D } _ { \phi }$ reconstructs the fused image $\widehat { \mathbf { H } } \in \mathbb { R } ^ { C \times H \times W } .$ which approximates the ground-truth (GT) image H. Algorithm 1 summarizes the deployment-time inference. The variables H and $\mathbf { B } _ { \mathrm { H } }$ define only the training target; neither is available on board during deployment. The fused code is inferred solely from P and M, so the payload is a predicted fused representation rather than a compressed copy of a preexisting HR-MSI/HR-HSI.

## B. Transmission-Oriented Binary Encoder

TOBE maps observations with different numbers of spectral bands into a shared, compact latent feature space before binary quantization (middle-left panel of Fig. 2).

1) Band-Adaptive Embedding: TOBE embeds all modalities with an adaptive convolution that maintains a learnable base kernel $\mathbf { W } _ { 0 }$ with $C _ { 0 }$ input channels. For an incoming modality with $C _ { \mathrm { i n } }$ spectral bands, the kernel is resampled along its input-channel dimension:

$$
\mathbf { W } _ { \mathrm { i n p u t } } ^ { ( c ) } = \sum _ { j = 1 } ^ { C _ { 0 } } \alpha _ { c , j } \mathbf { W } _ { 0 } ^ { ( j ) } ,\tag{1}
$$

where the superscripts (c) and (j) select input-channel slices of the adapted and base kernels, $c = 1 , \ldots , C _ { \mathrm { i n } } ,$ , and $\alpha _ { c , j }$ is the bicubic resampling weight from base channel j to adapted channel c. PAN and spectral inputs are thus mapped to the same feature dimension with shared weights. TOBD applies the same adaptation along the output-channel dimension.

![](images/7436d1b695a149a34cb211cb8c8c51ea502bc8ee6d7251e83247e0a41e840a0f.jpg)  
Fig. 2. Overall pipeline of TOP. On board, the observations are encoded by TOBE, binarized, and fused; the fused code is bit-packed, downlinked, and decoded by TOBD or stored on the ground. Downsampling is omitted for clarity; configurations are listed in Tab. X of the Supplementary Material. GELU: Gaussian error linear unit.

TABLE I  
COMPLEXITY OF CANDIDATE TOBE BACKBONE BLOCKS $( N _ { \mathrm { t o k } }$ TOKENS,WIDTH d, KERNEL SIZE k, WINDOW SIZE M)
<table><tr><td>Block</td><td>Modeling behavior</td><td>Main complexity ↓</td></tr><tr><td>CNN</td><td>Fixed, local</td><td> $\mathcal { O } ( N _ { \mathrm { t o k } } k ^ { 2 } d ^ { 2 } )$ </td></tr><tr><td>Global attention</td><td>Adaptive, all tokens</td><td> $\mathcal { O } ( N _ { \mathrm { t o k } } ^ { 2 } d + \dot { N } _ { \mathrm { t o k } } d ^ { 2 } )$ </td></tr><tr><td>Swin</td><td>Adaptive, windowed</td><td> $\mathcal { O } ( N _ { \mathrm { t o k } } ^ { \mathrm { w a x } } M ^ { 2 } d + N _ { \mathrm { t o k } } d ^ { 2 } )$ </td></tr></table>

2) Swin-Based Multi-Stage Feature Extraction: The embedded features are processed by a Swin Transformer encoder [14], whose windowed self-attention provides stronger content-adaptive spatial–spectral modeling than CNNs at a much lower cost than global self-attention [41] (Tab. I); the ablation in Sec. V-A verifies this choice. With $\mathbf { F } _ { 0 }$ denoting the embedded feature, the L encoder stages compute

$$
{ { \bf { F } } _ { \ell } } = { { S } _ { \ell } } \left( { { \mathcal { R } } _ { \ell } } \left( { { \bf { F } } _ { \ell - 1 } } \right) \right) , \qquad \ell = 1 , \dots , L ,\tag{2}
$$

where $\mathcal { R } _ { \ell }$ is a downsampling operator, applied only until the target factor s is reached, and S<sub>ℓ</sub> is a stack of Swin blocks whose depth varies with ℓ (Tab. X of the Supplementary Material). The output $\mathbf { F } _ { L }$ has $C _ { \mathrm { l a t } }$ channels and is spatially reduced by a factor of s along each dimension.

## C. Binary Spherical Quantization

Although spatially compressed, $\mathbf { F } _ { L }$ is still continuousvalued. TOBT therefore adopts BSQ [15], a binary tokenization technique from generative visual representation learning that maps the latent vector at each location to a binary code on the unit hypersphere. BSQ reduces the payload to one bit per latent entry and provides the latent fusion module with a compact and explicit discrete space that is well suited to fusion (Sec. V-B). Let $\mathbf { f } _ { i } \in \mathbb { R } ^ { C _ { \mathrm { l a t } } }$ denote the channel vector of $\mathbf { F } _ { L }$ at latent location $i = 1 , \ldots , N$ , with $N = H W / s ^ { 2 }$ . BSQ performs deterministic sign quantization

$$
\mathbf { b } _ { i } = \mathrm { s i g n } \left( \mathbf { f } _ { i } \right) \in \{ - 1 , + 1 \} ^ { C _ { \mathrm { l a t } } } ,\tag{3}
$$

where sign(·) acts elementwise, maps zero to +1, and is trained with a straight-through estimator. Collecting the codes over all locations gives the normalized binary latent code

$$
\begin{array} { l } { { \displaystyle { \bf B } = \frac { 1 } { \sqrt { C _ { \mathrm { l a t } } } } \mathrm { r e s h a p e } \left( \left[ { \bf b } _ { 1 } , \ldots , { \bf b } _ { N } \right] \right) } \ ~ } \\ { { \displaystyle ~ \in \left\{ - \frac { 1 } { \sqrt { C _ { \mathrm { l a t } } } } , + \frac { 1 } { \sqrt { C _ { \mathrm { l a t } } } } \right\} ^ { C _ { \mathrm { l a t } } \times H / s \times W / s } } , } \end{array}\tag{4}
$$

where ‘reshape(·)’ arranges the $C _ { \mathrm { l a t } } \times N$ matrix of column vectors into a tensor in a fixed raster order, and B denotes the code of any of the three modalities. During training, soft assignment and grouped entropy regularization encourage effective code utilization [15]. In practice, the latent channels are split into several groups and the entropy terms are computed per group; for simplicity, the main text treats all channels as a single group. Its details are given in Sec. S1 of the Supplementary Material.

## D. Latent Fusion Module

From the concatenated codes ${ \bf { B } } _ { \mathrm { { M } } }$ and $\mathbf { B } _ { \mathrm { P } }$ , the latent fusion module $\mathcal { F } _ { \psi }$ predicts two logits for each latent entry:

$$
\mathbf { O } = \mathcal { F } _ { \psi } \left( \operatorname { C o n c a t } _ { c } \left[ \mathbf { B } _ { \mathrm { M } } , \mathbf { B } _ { \mathrm { P } } \right] \right) \in \mathbb { R } ^ { 2 \times C _ { \mathrm { l a t } } \times H / s \times W / s } .\tag{5}
$$

and the fused code is obtained by selecting a binary state independently for each entry:

$$
\widehat { \bf B } _ { \mathrm { H } } = \frac { 1 } { \sqrt { C _ { \mathrm { l a t } } } } \left( 2 \arg \operatorname* { m a x } _ { a \in \{ 0 , 1 \} } { \bf O } _ { a } - { \bf 1 } \right) ,\tag{6}
$$

![](images/49d857c5d8438bb6a1a968f525dbb208a70a40aaaa727637ba8edf0274ac71a0.jpg)  
Fig. 3. Training pipeline of the proposed method. (a) Three-stage TOBT training: stage 1 trains TOBE and TOBD jointly (40k iterations); stages 2 and 3 refine TOBD with TOBE frozen (20k iterations each), stage 3 with random bit flips. (b) Latent fusion training with TOBT frozen. Bold outlines: trainable; hatched: frozen; gray: parameter-free; dashed arrows: loss targets.

where ${ \bf O } _ { a }$ is the slice of O for state $a \in \{ 0 , 1 \}$ , the arg max is taken elementwise, and 1 is an all-ones tensor of the latent shape.

## E. Binary Downlink and TOBD

The satellite downlinks the bitstream $\mathbf { S } _ { \mathrm { d o w n } }$ = BitPack(Bb <sub>H</sub>), where ‘BitPack(·)’ maps negative entries to 0 and nonnegative entries to 1 and serializes them in a fixed order, together with the latent raster shape, whose negligible size is omitted from the compression ratio in Eq. (18). Unlike the original BSQ, which further compresses the codes with an autoregressive prior and arithmetic coding [15], TOP uses no entropy model or rANS codec [16], as supported by Sec. V-C.

On the ground side, the received bitstream is unpacked and decoded as

$$
\begin{array} { r } { \widetilde { \bf B } _ { \mathrm { H } } = \mathrm { B i t U n p a c k } ( { \bf S } _ { \mathrm { d o w n } } ) , } \\ { \widehat { \bf H } = \mathcal { D } _ { \phi } ( \widetilde { \bf B } _ { \mathrm { H } } ) , \qquad } \end{array}\tag{7}
$$

where ‘BitUnpack(·)’ inverts ‘BitPack(·)’ by mapping each bit $z \in \{ 0 , 1 \}$ to $( 2 z - 1 ) / \sqrt { C _ { \mathrm { l a t } } } ;$ the received code $ { \widetilde { \mathbf { B } } } _ { \mathrm { H } }$ may differ from $\hat { \bf B } _ { \mathrm { H } }$ if transmission errors occur. $\mathcal { D } _ { \phi }$ follows the stage layout of ${ \mathcal { E } } _ { \theta }$ in reverse order, possibly with different numbers of Swin blocks per stage (lower-left panel of Fig. 2), and outputs the final HR-MSI/HR-HSI reconstruction Hb . The binary code can also be archived for deferred decoding, which yields considerable storage savings.

## F. Optimization Objectives

We first train TOBT, including its TOBE and TOBD, and then train the latent fusion module with the frozen tokenizer (Fig. 3).

1) Three-Stage TOBT Training:

a) Loss Components: For an image-domain target X and its reconstruction $\hat { \mathbf { X } } .$ , the mean squared error (MSE) and Sobelgradient losses are

$$
\mathcal { L } _ { \mathrm { M S E } } = \frac { \| \widehat { \mathbf { X } } - \mathbf { X } \| _ { 2 } ^ { 2 } } { C H W } , \qquad \mathcal { L } _ { \mathrm { g r a d } } = \frac { \| \nabla _ { \mathrm { S } } \widehat { \mathbf { X } } - \nabla _ { \mathrm { S } } \mathbf { X } \| _ { 1 } } { C H W } ,\tag{8}
$$

where the norms are computed over vectorized entries and $\nabla _ { \mathrm { { S } } }$ is the channel-wise Sobel operator; $\mathcal { L } _ { \mathrm { g r a d } }$ penalizes edge discrepancies when refining the ground-side decoder. For spectral fidelity, the spectral angle mapper (SAM) loss [42] averages the angle, in radians, between the reconstructed and target spectral vectors $\widehat { \mathbf { x } } _ { i } , \mathbf { x } _ { i } \in \mathbb { R } ^ { C }$ over all spatial locations:

$$
\mathcal { L } _ { \mathrm { S A M } } = \frac { 1 } { H W } \sum _ { i = 1 } ^ { H W } \operatorname { a r c c o s } \left( \frac { \langle \widehat { \mathbf { x } } _ { i } , \mathbf { x } _ { i } \rangle } { \| \widehat { \mathbf { x } } _ { i } \| _ { 2 } \| \mathbf { x } _ { i } \| _ { 2 } } \right) .\tag{9}
$$

Unlike MSE, SAM is insensitive to uniform spectral scaling. The quantization loss follows BSQ [15]: a commitment term pulls the normalized continuous latent toward its binary code, and an entropy term prevents the encoder from collapsing onto a few binary patterns:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { Q } } = \beta \mathcal { L } _ { \mathrm { c o m m i t } } + \mathcal { L } _ { \mathrm { e n t r o p y } } , } \end{array}\tag{10}
$$

where $\beta$ weights the commitment term (Sec. S1 of the Supplementary Material).

b) Training Stages: In stage 1 (Fig. 3(a)), TOBE and TOBD are jointly trained on all three modalities to stabilize the shared binary latent space before the decoder is refined, using

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { T O B T } } = \lambda _ { \mathrm { M S E } } \mathcal { L } _ { \mathrm { M S E } } + \lambda _ { \mathrm { Q } } \mathcal { L } _ { \mathrm { Q } } , } \end{array}\tag{11}
$$

where $\lambda _ { \mathrm { M S E } }$ and $\lambda _ { \mathrm { Q } }$ are weighting coefficients. Stage 2 targets the full TOP setting, in which only the HR code is transmitted: TOBE is frozen, so the quantization loss is dropped, and TOBD is fine-tuned on the HR modality with

$$
\mathcal { L } _ { \mathrm { T O B D } } = \mathcal { L } _ { \mathrm { M S E } } + \lambda _ { \mathrm { g r a d } } \mathcal { L } _ { \mathrm { g r a d } } + \lambda _ { \mathrm { S A M } } \mathcal { L } _ { \mathrm { S A M } } ,\tag{12}
$$

Decoded image under 5% random bit flips

Trained with bit flips  
![](images/9e0451c4bf9e0a37a1ea1ff6f7df48975d314cdd7147fd4883781c992198d1d0.jpg)  
Fig. 4. WV3 reduced-resolution MSI reconstruction under 5% bit flips. Left: TOBD after stage 3 (with bit-flip augmentation). Right: TOBD after stage 2 (without). White boxes mark the close-ups below.

where the weights $\lambda _ { \mathrm { g r a d } }$ and $\lambda _ { \mathrm { S A M } }$ are listed in Sec. S3 of the Supplementary Material.

Stage 3 copes with binary latent errors caused by imperfect latent fusion or by transmission. It keeps the stage-2 setting but injects random bit flips into the HR code before decoding:

$$
\mathbf { B } _ { \mathrm { H } } ^ { \mathrm { f i p } } = ( \mathbf { 1 } - 2 \mathbf { R } ) \odot \mathbf { B } _ { \mathrm { H } } ,\tag{13}
$$

where the entries of the mask $\mathbf { R } \in \{ 0 , 1 \} ^ { C _ { \mathrm { l a t } } \times H / s \times W / s }$ are independently drawn from Bernoulli(p(t)), and an entry equal to 1 reverses the sign of the corresponding latent entry. The flip probability is linearly ramped from $p _ { \mathrm { s t a r t } } = 1 0 ^ { - 5 }$ to $p _ { \mathrm { e n d } } =$ $3 \times 1 0 ^ { - 3 }$ over the first 2000 iterations:

$$
p ( t ) = p _ { \mathrm { s t a r t } } + \operatorname* { m i n } \left( \frac { t } { 2 0 0 0 } , 1 \right) \left( p _ { \mathrm { e n d } } - p _ { \mathrm { s t a r t } } \right) ,\tag{14}
$$

where t is the stage-3 iteration index. TOBD thereby learns to correct occasional bit errors by exploiting the redundancy of the code. Under 5% bit flips (Fig. 4), the stage-3 decoder suppresses the speckle-like artifacts of the stage-2 decoder and better preserves spatial structures and colors; Sec. V-D quantifies this effect.

2) Latent Fusion Module Training: With all tokenizer parameters frozen, the PAN, upsampled LR-MSI/LR-HSI, and HR-MSI/HR-HSI images of each training sample are encoded into B<sub>P</sub>, ${ \bf { B } } _ { \mathrm { { M } } }$ , and $\mathbf { B } _ { \mathrm { H } }$ . The latent fusion module predicts the logits O from ${ \bf { B } } _ { \mathrm { { M } } }$ and $\mathbf { B } _ { \mathrm { P } }$ as in Eq. (5), and the fixed code $\mathbf { B } _ { \mathrm { H } }$ supervises it, so the training target uses the same binary representation as the transmitted payload. The module is trained with a binary cross-entropy (BCE) loss on the logits:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { f u s e } } = f _ { \mathrm { B C E W i t h L o g i t s } } ( \mathbf { O } , \mathbf { T } ) , } \end{array}\tag{15}
$$

where the target $\mathbf { T } ~ \in ~ \{ 0 , 1 \} ^ { 2 \times C _ { \mathrm { l a t } } \times H / s \times W / s }$ has the same shape as O, with state slices $\mathbf { T } _ { 1 } \ = \ \mathbb { I } [ \mathbf { B } _ { \mathrm { H } } \ > \ 0 ]$ and ${ \bf T } _ { 0 } = { \bf 1 } - { \bf T } _ { 1 }$ (elementwise), and the loss averages the scalar

BCE terms over both states and all latent entries (Sec. S2 of the Supplementary Material). No pixel-domain loss is backpropagated through TOBD during this training.

## G. Transmission-Oriented Pansharpening Formulation

At the system level, TOP can be summarized as

$$
\begin{array} { r l } { \displaystyle \operatorname* { m i n } _ { \boldsymbol { \theta } , \boldsymbol { \psi } , \boldsymbol { \phi } } } & { \int _ { \mathrm { f i d } } \Big ( \widehat { \mathbf { H } } , \mathbf { H } \Big ) } \\ { \mathrm { s . t . } } & { \widehat { \mathbf { H } } = \mathcal { D } _ { \boldsymbol { \phi } } ( \widetilde { \mathbf { B } } _ { \mathrm { H } } ) , \big | \mathbf { S } _ { \mathrm { d o w n } } \big | \le B _ { \mathrm { m a x } } , } \end{array}\tag{16}
$$

where the downlinked bitstream $\mathbf { S } _ { \mathrm { d o w n } }$ and the received code $ { \widetilde { \mathbf { B } } } _ { \mathrm { H } }$ are obtained from (P, M) through the encoding, latentfusion, and bit-packing steps of Alg. $1 ~ ( \mathrm { E q s . } ~ ( 5 ) \mathrm { - } ( 7 ) ) , \mathcal { L } _ { \mathrm { f i d } } ( \cdot )$ collectively represents the fidelity objectives used throughout TOP, and $B _ { \mathrm { m a x } }$ denotes the maximum allowable downlink budget.

In contrast, conventional pansharpening mainly considers

$$
\operatorname* { m i n } _ { \omega } \quad \mathcal { L } _ { \mathrm { f u s } } \left( \mathbf { H } , \mathcal { G } _ { \omega } ( \mathbf { P } , \mathbf { M } ) \right) ,\tag{17}
$$

where $\mathcal { G } _ { \omega }$ is a conventional pansharpening model with learnable parameters ω and ${ \mathcal { L } } _ { \mathrm { f u s } }$ is its fusion objective. Unlike this formulation, TOP explicitly incorporates the downlink constraint into the pansharpening problem.

## IV. EXPERIMENTS

## A. Experimental Settings

1) Datasets: We use the WV3 (8 bands) and GF2 (4 bands) multispectral datasets from PanCollection [43] and the 102-band Pavia Center hyperspectral dataset distributed through HyperPanCollection and used in Hyper-DSNet [44], all with a spatial scale factor of four. WV3, GF2, and Pavia contain 9,714, 19,809, and 1,512 training samples and 1,080, 2,201, and 168 validation samples, respectively; the validation samples are held-out training patches distinct from the test scenes. Training samples consist of 64 × 64 PAN, upsampled LR-MSI/LR-HSI, and HR-MSI/HR-HSI patches together with $1 6 \times 1 6$ native LR-MSI/LR-HSI patches. For WV3 and GF2, we test on 20 reduced-resolution scenes of size $2 5 6 \times 2 5 6$ simulated following Wald’s protocol [45], and on 20 fullresolution scenes of size $5 1 2 \times 5 1 2$ without GT. For Pavia, the Full test split contains two 400 × 400 scenes with HR-HSI references. Details are given in Sec. S4 of the Supplementary Material.

2) Compared Methods: The multispectral baselines are GS [3], BDSD [4], GLP [5], and MF [23] (traditional); FusionNet [6], U2Net [7], WFANet [8], ADWM [26], and ARConv [27] (supervised); and ZS-PAN [29] (zero-shot). The hyperspectral baselines are AWLP [22], BDSD-PC [46], BT-H [47], GSA [20], MF [23], and PRACS [21] (traditional); HyperTransformer [30] (supervised); and DIP-HyperKite [32], R-PNN [33], uTDSP [35], DM-ZS [36], and ρ-PNN [34] (zero-shot).

WV3 (33×)  
WV3 (66×)  
GF2 (50×)  
Pavia (23.97×)  
![](images/5884ca05cf4848026292a9034af78bb0c0676de35654e56df4bd0c761e7cf007.jpg)  
Fig. 5. TOBT multi-modality reconstruction results on WV3, GF2, and Pavia. Upper rows: original test crops; lower rows: independent TOBT reconstructions. Ratios are overall TOP compression ratios (Eq. (18)).

3) Transmission-Oriented Protocol and Communication Metrics: All baselines follow the ‘compress first, fuse later pipeline with JPEG2000 [11] (OpenJPEG, with a search over quality layers to match the target ratio) or HEVC [12] (libx265 with constant rate factor control to match the target ratio). The compression ratio R divides the raw sensor-side data volume by the size $B _ { \mathrm { t r a n s m i t t e d } }$ of the transmitted bitstream. To avoid inflating R, the raw volume uses the native, non-upsampled LR-MSI/LR-HSI:

$$
R = { \frac { b H W + b C \left( { \frac { H } { r } } \right) \left( { \frac { W } { r } } \right) } { B _ { \mathrm { t r a n s m i t t e d } } } } ,\tag{18}
$$

where the PAN image has size $H \times W .$ , the LR-MSI/LR-HSI has $C$ bands and size $\textstyle { \frac { H } { r } } \times { \frac { W } { r } }$ , r is the spatial scale factor, and b is the native sensor bit depth (11 for WV3, 10 for GF2, and 13 for Pavia, based on the official specifications). For a codecbased baseline, $B _ { \mathrm { t r a n s m i t t e d } }$ is the total size of the separately encoded PAN and native LR-MSI/LR-HSI streams; for TOP, it is the size of the single bit-packed fused latent. Using the native spectral image in the numerator is essential because interpolation creates samples without adding sensor information and would otherwise overstate the apparent compression ratio. The bits per pixel per band (bppb) normalizes the payload by the size of the target HR-MSI/HR-HSI:

$$
\mathrm { b p p b } = { \frac { B _ { \mathrm { t r a n s m i t t e d } } } { C H W } } .\tag{19}
$$

4) Evaluation Metrics: On scenes with GT, we report PSNR, SSIM [48], SAM (in degrees) [42], and ERGAS [49]. We also report the no-reference HQNR index and its spectral and spatial distortion components $D _ { \lambda }$ and $D _ { s }$ [50]. In the reduced-resolution tables, they compare each fused image with the LR-MSI/LR-HSI and PAN of the same scene and thus measure consistency with the source observations rather than full-resolution quality. For the lossy baselines, $D _ { \lambda }$ and $D _ { s }$ are computed against the original lossless observations rather than their decompressed counterparts. Otherwise, a codec could distort both the fusion input and the metric anchor in a similar direction and receive an artificially favorable consistency score. Retaining the common lossless anchors makes HQNR terms comparable across codecs and TOP, while PSNR, SSIM, SAM, and ERGAS independently measure agreement with GT. In the result tables, the best and second-best results are shown in boldface and underlined, respectively.

5) Implementation Details: TOP is implemented in Py-Torch with bfloat16 mixed precision on an NVIDIA RTX 4090 GPU and optimized with AdamW at an initial learning rate of $2 \times 1 0 ^ { - 4 }$ , with weight decay 0.05 for TOBT and 0.01 for the latent fusion module. Architectures and training schedules are listed in Sec. S3 of the Supplementary Material.

## B. Multimodal Reconstruction of TOBT

Table III reports validation-set reconstruction of PAN, upsampled LR-MSI/LR-HSI, and HR-MSI/HR-HSI images, and Fig. 5 shows examples. These modality-wise results isolate the representation capacity available before latent fusion. On WV3, halving the HR-MSI payload from 0.063 to 0.031 bppb lowers reconstruction PSNR from 30.17 to 28.22 dB, exposing the expected rate–fidelity trade-off. LR-MSI reconstruction nevertheless remains above 37 dB and 0.95 SSIM. TOBT also reaches 33.46 dB on GF2 HR-MSI and 34.85 dB on Pavia HR-HSI, supporting the shared design across sensors and spectral dimensionalities. Fine textures become smoother as the ratio increases, while the main structures and spectral appearance remain stable. These results define the tokenizer’s reconstruction ceiling; the end-to-end experiments test how closely the predicted fused codes approach it.

## C. Transmission-Oriented Pansharpening

Having established that TOBT preserves the three modalities under a constrained binary capacity, we evaluate whether the complete TOP pipeline can turn that representation into a useful downlink product. The comparison therefore considers fusion fidelity and communication cost jointly: a method is favorable only if it preserves spatial–spectral information while reducing the transmitted payload relative to the codecbased pipelines.

1) Multispectral Pansharpening on WV3: Table II reports the WV3 reduced-resolution results under JPEG2000. The best baseline reaches 27.70 dB at 21.26×, whereas TOP achieves 30.32 dB at 33× and 28.88 dB at 66×; relative to the best baseline value of each metric, TOP also reduces ERGAS from 8.36 to 6.00 and SAM from 10.39 to 6.00 degrees. Both TOP configurations outperform all baselines on every metric. Fig. 6(a) shows that TOP preserves sharper local structures and more stable spectral appearance, whereas the codecbased baselines suffer from edge degradation and texture loss. Comparisons with all baselines are given in Sec. S5 of the Supplementary Material.

2) Multispectral Pansharpening on GF2: On GF2 (Tab. IV), TOP achieves 31.37 dB PSNR, 0.83 SSIM, 2.63 ERGAS, and 2.46 degrees SAM at 50×. The strongest baseline, U2Net, reaches 28.53 dB and 4.29 degrees at 20.33×, so TOP improves PSNR by 2.84 dB at about 2.5 times its compression ratio; the zero-shot ZS-PAN reaches 24.52 dB. Fig. 6(b) shows that TOP avoids the codec artifacts that affect the baselines.

![](images/552b432880405b96ef722a29d45912732ae86b8f803172bb87d02b1e0bb8c341.jpg)  
Fig. 6. Qualitative comparison under the JPEG2000 transmission-oriented protocol on (a) WV3, (b) GF2, and (c) Pavia. TOP is compared with th six baselines of highest PSNR (averaged over WV3 and GF2 for the multispectral data). Rows show the predictions with GT and the error maps (dark blue: lower error). All baselines are compared in Sec. S5 of the Supplementary Material.

TABLE II  
WV3 REDUCED-RESOLUTION RESULTS UNDER JPEG2000 (MEAN ± STD.)
<table><tr><td>Method</td><td>Ratio ↑</td><td> $\mathrm { P S N R } \pm \mathrm { S t d . } \ 1$ </td><td> $\mathrm { S S I M } \pm \mathrm { S t d . } \uparrow$ </td><td> $\mathrm { E R G A S } \pm \mathrm { S t d . } \downarrow$ </td><td> $\mathrm { H Q N R } \pm \mathrm { S t d . ~ } \uparrow$ </td><td> $D _ { \lambda } \pm \mathrm { S t d . ~ \downarrow ~ }$ </td><td> $D _ { s } \pm \mathrm { S t d . ~ } \downarrow$ </td><td> $\mathbf { S A M } \ ( \mathrm { d e g } ) \pm \mathbf { S t d } . \downarrow$ </td></tr><tr><td>BDSD+JPEG2000 [4]</td><td rowspan="9"></td><td> $2 5 . 4 4 \pm 3 . 8 5$ </td><td> $0 . 5 5 \pm 0 . 1 6$ </td><td> $1 0 . 8 5 \pm 2 . 3 1$ </td><td> $0 . 1 7 \pm 0 . 0 6$ </td><td> $0 . 6 8 \pm 0 . 0 6$ </td><td> $0 . 4 8 \pm 0 . 1 2$ </td><td> $1 1 . 2 0 \pm 2 . 4 4$ </td></tr><tr><td>FusionNet+JPEG2000 [6]</td><td> $2 6 . 5 0 \pm 3 . 6 9$ </td><td> $0 . 7 3 \pm 0 . 0 9$ </td><td> $9 . 3 9 \pm 2 . 3 1$ </td><td> $0 . 3 6 \pm 0 . 0 8$ </td><td> $0 . 5 4 \pm 0 . 0 8$ </td><td> $0 . 2 3 \pm 0 . 0 9$ </td><td> $1 0 . 5 1 \pm 2 . 3 4$ </td></tr><tr><td>GLP+JPEG2000 [5]</td><td> $2 6 . 5 3 \pm 3 . 8 4$ </td><td> $0 . 6 8 \pm 0 . 1 1$ </td><td> $9 . 5 6 \pm 2 . 0 4$ </td><td> $0 . 2 7 \pm 0 . 0 6$ </td><td> $0 . 6 3 \pm 0 . 0 5$ </td><td> $0 . 2 9 \pm 0 . 1 1$ </td><td> $1 0 . 6 4 \pm 2 . 4 2$ </td></tr><tr><td>GS+JPEG2000 [3]</td><td> $2 7 . 7 0 \pm 3 . 7 4$ </td><td> $0 . 7 1 \pm 0 . 1 1$ </td><td> $8 . 3 6 \pm 1 . 7 8$ </td><td> $0 . 4 7 \pm 0 . 0 9$ </td><td> $0 . 4 3 \pm 0 . 0 8$ </td><td> $0 . 1 8 \pm 0 . 0 9$ </td><td> $1 0 . 5 2 \pm 2 . 4 9$ </td></tr><tr><td>MF+JPEG2000 [23]</td><td> $2 6 . 5 9 \pm 3 . 8 8$ </td><td> $0 . 6 7 \pm 0 . 1 2$ </td><td> $9 . 4 9 \pm 2 . 0 8$ </td><td> $0 . 2 7 \pm 0 . 0 8$ </td><td> $0 . 6 2 \pm 0 . 0 7$ </td><td> $0 . 2 9 \pm 0 . 1 2$ </td><td> $1 0 . 4 4 \pm 2 . 3 4$ </td></tr><tr><td>U2Net+JPEG2000 [7]</td><td> $2 6 . 6 2 \pm 3 . 6 3$ </td><td> $0 . 7 0 \pm 0 . 1 0$ </td><td> $9 . 3 4 \pm 2 . 0 5$ </td><td> $0 . 3 3 \pm 0 . 0 8$ </td><td> $0 . 5 6 \pm 0 . 0 8$ </td><td> $0 . 2 6 \pm 0 . 0 9$ </td><td> $1 0 . 3 9 \pm 2 . 2 3$ </td></tr><tr><td>WFANet+JPEG2000 [8]</td><td> $2 5 . 6 4 \pm 3 . 9 2$ </td><td> $0 . 6 4 \pm 0 . 1 3$ </td><td> $1 0 . 5 4 \pm 2 . 4 5$ </td><td> $0 . 2 0 \pm 0 . 0 8$ </td><td> $0 . 6 8 \pm 0 . 0 7$ </td><td> $0 . 3 8 \pm 0 . 1 1$ </td><td> $1 0 . 6 2 \pm 2 . 3 4$ </td></tr><tr><td>ZS-PAN+JPEG2000 [29]</td><td> $2 6 . 0 2 \pm 3 . 5 4$ </td><td> $0 . 6 2 \pm 0 . 1 4$ </td><td> $9 . 9 2 \pm 1 . 7 0$ </td><td> $0 . 2 3 \pm 0 . 0 9$ </td><td> $0 . 6 5 \pm 0 . 0 8$ </td><td> $0 . 3 7 \pm 0 . 1 4$ </td><td> $1 1 . 0 0 \pm 2 . 4 4$ </td></tr><tr><td>ADWM+JPEG2000 [26]</td><td> $2 6 . 0 2 \pm 5 . 5 2$ </td><td> $0 . 7 1 \pm 0 . 1 4$ </td><td> $1 1 . 2 3 \pm 5 . 6 6$ </td><td> $0 . 3 8 \pm 0 . 1 2$ </td><td> $0 . 5 3 \pm 0 . 1 0$ </td><td> $0 . 2 1 \pm 0 . 1 2$ </td><td> $1 5 . 1 9 \pm 7 . 3 7$ </td></tr><tr><td>ARConv+JPEG2000 [27]</td><td></td><td> $2 6 . 7 0 \pm 3 . 8 4$ </td><td> $0 . 7 0 \pm 0 . 1 1$ </td><td> $9 . 3 3 \pm 1 . 9 5$ </td><td> $0 . 2 9 \pm 0 . 0 8$ </td><td> $0 . 6 0 \pm 0 . 0 7$ </td><td> $0 . 2 9 \pm 0 . 1 0$ </td><td> $1 0 . 4 3 \pm 2 . 3 0$ </td></tr><tr><td>TOP (ours)</td><td>33×</td><td> ${ \bf 3 0 . 3 2 \pm 2 . 8 3 }$ </td><td> ${ \bf 0 . 8 2 \pm 0 . 0 4 }$ </td><td> ${ \bf 6 . 0 0 \pm 1 . 2 3 }$ </td><td> ${ \bf 0 . 7 6 \pm 0 . 1 4 }$ </td><td> ${ \bf 0 . 1 9 \pm 0 . 1 1 }$ </td><td> $\mathbf { 0 . 0 8 \pm 0 . 0 6 }$ </td><td> ${ \bf 6 . 0 0 \pm 1 . 3 8 }$ </td></tr><tr><td>TOP (ours)</td><td>66×</td><td> $2 8 . 8 8 \pm 3 . 0 6$ </td><td> $\underline { { 0 . 7 5 } } \pm 0 . 0 7$ </td><td> ${ \underline { { 7 . 1 3 } } } \pm 1 . 4 8$ </td><td> $\underline { { 0 . 6 9 } } \pm 0 . 1 3$ </td><td> $\underline { { 0 . 2 3 } } \pm 0 . 1 1$ </td><td> $\underline { { 0 . 1 1 } } \pm 0 . 0 5$ </td><td> $\underline { { 6 . 6 9 } } \pm 1 . 5 6$ </td></tr></table>

Tables V and VI report the corresponding reducedresolution comparisons under HEVC. The TOP rows are identical to those in the JPEG2000 tables because neither codec is part of TOP: HEVC and JPEG2000 are applied only to the source observations transmitted by the pixel-domain baselines, whereas TOP always bit-packs and transmits the same fused binary latent. Repeating the TOP operating point in each table provides a direct comparison against the baseline pipeline induced by that codec. Although the HEVC baselines operate at only 6.69× on WV3 and 8.94× on GF2, TOP exceeds their strongest PSNR values (27.22 and 29.13 dB) by 3.10 and 2.24 dB at 33× and 50×, respectively, and leads on every metric. The persistence of these gains across two sensors and two codecs indicates that the advantage comes from changing the transmitted representation rather than adapting to the artifacts of one particular codec. The corresponding visual comparisons are provided in Sec. S6, and full-resolution results, where TOP attains the highest HQNR and lowest $D _ { \lambda }$ under both codecs, are given in Sec. S7 of the Supplementary Material.

TABLE III  
TOBT RECONSTRUCTION ON THE VALIDATION SETS (MEAN ± STD.; BPPB: MODALITY-SPECIFIC PAYLOAD, EQ. (19))
<table><tr><td>Dataset</td><td>Modality</td><td>bppb↓</td><td> $\mathrm { P S N R } \pm \mathrm { S t d . } \uparrow$ </td><td> $\mathrm { S S I M } \pm \mathrm { S t d . } \uparrow$ </td></tr><tr><td></td><td>HR-MSI</td><td>0.063</td><td> $3 0 . 1 7 \pm 4 . 4 2$ </td><td> $0 . 8 3 \pm 0 . 0 8$ </td></tr><tr><td>WV3 (33×)</td><td>LR-MSI</td><td>0.063</td><td> $3 9 . 6 1 \pm 3 . 1 8$ </td><td> $0 . 9 7 \pm 0 . 0 1$ </td></tr><tr><td></td><td>PAN</td><td>0.500</td><td> $3 4 . 8 8 \pm 5 . 0 1$ </td><td> $0 . 9 1 \pm 0 . 0 4$ </td></tr><tr><td></td><td>HR-MSI</td><td>0.031</td><td> $2 8 . 2 2 \pm 4 . 5 9$ </td><td> $0 . 7 4 \pm 0 . 1 2$ </td></tr><tr><td>WV3 (66×)</td><td>LR-MSI</td><td>0.031</td><td> $3 7 . 1 4 \pm 3 . 7 3$ </td><td> $0 . 9 5 \pm 0 . 0 2$ </td></tr><tr><td></td><td>PAN</td><td>0.250</td><td> $3 1 . 9 4 \pm 5 . 4 3$ </td><td> $0 . 8 3 \pm 0 . 0 8$ </td></tr><tr><td></td><td>HR-MSI</td><td>0.063</td><td> $3 3 . 4 6 \pm 3 . 7 5$ </td><td> $0 . 8 7 \pm 0 . 0 6$ </td></tr><tr><td>GF2</td><td>LR-MSI</td><td>0.063</td><td> $4 0 . 3 1 \pm 2 . 8 5$ </td><td> $0 . 9 7 \pm 0 . 0 2$ </td></tr><tr><td></td><td>PAN</td><td>0.250</td><td> $3 3 . 4 6 \pm 3 . 9 4$ </td><td> $0 . 8 7 \pm 0 . 0 5$ </td></tr><tr><td></td><td>HR-HSI</td><td>0.039</td><td> $3 4 . 8 5 \pm 3 . 5 0$ </td><td> $0 . 9 2 \pm 0 . 0 2$ </td></tr><tr><td>Pavia</td><td>LR-HSI</td><td>0.039</td><td> $4 3 . 9 0 \pm 2 . 5 7$ </td><td> $0 . 9 9 \pm 0 . 0 0$ </td></tr><tr><td></td><td>PAN</td><td>4.000</td><td> $4 0 . 8 6 \pm 2 . 9 3$ </td><td> $0 . 9 7 \pm 0 . 0 1$ </td></tr></table>

3) Hyperspectral Pansharpening on Pavia: Table VII reports the Pavia results under JPEG2000. The baselines operate at 19.62×, with the best PSNR and HQNR reaching 28.09 dB and 0.83. At the higher ratio of 23.97×, TOP achieves 29.72 dB PSNR, 0.83 SSIM, 5.91 ERGAS, 0.85 HQNR, 6.39 degrees SAM, $D _ { \lambda } = 0 . 1 4$ , and $D _ { s } = 0 . 0 2$ , the best result on every metric. Fig. 6(c) shows that TOP also preserves hyperspectral spatial structures under a constrained downlink. The HEVC results, where TOP again outperforms all baselines, are given in Sec. S6 of the Supplementary Material. Across WV3, GF2, and Pavia, the same pattern holds for 8-, 4-, and 102-band data: TOP improves both GT-based fidelity and consistency with the source observations while using a higher compression ratio. The simultaneous gains in PSNR, SAM, $D _ { \lambda }$ , and $D _ { s }$ are important because a higher PSNR alone could mask band-dependent spectral distortion. The following profile analysis examines this balance directly at the band level.

4) Spectral Profile Analysis: Image-level metrics summarize the final reconstruction but do not show how compression and fusion redistribute errors across spectral bands. Fig. 7 therefore compares the normalized band-wise mean over a fixed ROI with the GT profile, with all baselines paired with JPEG2000. In all four settings, the TOP curve follows the GT profile most closely and its residual strip remains nearer to zero, whereas the codec-based baselines show larger banddependent amplitude errors. This trend persists for WV3 at both 33× and 66× and extends to GF2 and the 102-band Pavia data. The spectral-profile evidence thus complements SAM and $D _ { \lambda } \colon$ the quantitative gains are not produced by spatial sharpening at the expense of spectral consistency.

## V. ABLATION STUDIES

We ablate four design choices of TOP on the WV3 reducedresolution data: the tokenizer backbone, BSQ, an additional entropy model, and stage-3 bit-flip training. These components address different system constraints, so they cannot be judged by reconstruction accuracy alone: the backbone controls representation quality and computation, BSQ affects payload and latent predictability, entropy coding trades bitrate for senderside overhead, and stage 3 controls sensitivity to erroneous bits. Fig. 8 summarizes these accuracy–payload–complexity trade-offs, and the numerical values are listed in panels (a)– (e) of Tab. XVI in the Supplementary Material.

## A. Architecture Choice

We compare CNN, Vision Transformer (ViT) [41], and Swin [14] backbones with 66.721M, 62.709M, and 65.609M parameters, respectively, keeping the latent configuration, quantization, and training fixed; this Swin variant is matched to the parameter budget and differs from the dataset-specific tokenizer configurations listed in Tab. X of the Supplementary Material. As shown in Fig. 8(a), Swin achieves the best HR-MSI reconstruction at a computational cost close to that of the CNN and lower than that of ViT, whereas ViT adds computation without a corresponding gain. Swin is therefore adopted for TOBE and TOBD.

## B. Effectiveness of BSQ

BSQ reduces the payload to one bit per latent entry (Fig. 8(b)). To test whether it also makes the target latent easier to predict, we compare the fusion PSNR with the PSNR of directly encoding and decoding the GT image with the same tokenizer; their gap indicates how closely latent fusion approaches the reconstruction capability of the latent space. Both PSNRs are computed on the 20 WV3 reducedresolution test scenes, so the fusion PSNR with BSQ is the 30.32 dB of TOP at 33× in Tab. II, whereas Tab. III reports reconstruction on the validation set. Removing BSQ improves reconstruction, but even with a deliberately enlarged and fully trained fusion module, the continuous-latent variant leaves a 2.59 dB gap between reconstruction and fusion. BSQ reduces this gap to 1.27 dB while shrinking the fusion module from 19.33M to 0.61M parameters and from 9.88 to 0.31 giga floating-point operations (GFLOPs). BSQ thus trades part of the reconstruction capacity for a compact payload and a more predictable fusion target.

## C. Trade-Off of an Additional Entropy Model

A neural entropy model, which drives rANS [16] or arithmetic coding with estimated symbol probabilities [13], is most effective when the latent retains considerable statistical redundancy, whereas TOBT already compresses the HR information

TABLE IV

GF2 REDUCED-RESOLUTION RESULTS UNDER JPEG2000 (MEAN ± STD.)

$$
\mathrm { P S N R } \pm \mathrm { S t d . } \uparrow
$$

$$
\mathrm { S S I M } \pm \mathrm { S t d . } \uparrow
$$

$$
\mathrm { E R G A S } \pm \mathrm { S t d . } \downarrow
$$

$$
\mathrm { H Q N R } \pm \mathrm { S t d . ~ } \uparrow
$$

$$
D _ { \lambda } \pm \mathrm { S t d . ~ \downarrow ~ }
$$

$$
D _ { s } \pm \mathrm { S t d . ~ \downarrow ~ }
$$

$$
\mathbf { S A M } \ ( \mathrm { d e g } ) \pm \mathbf { S t d } . \downarrow
$$

$$
2 4 . 3 3 \pm 2 . 1 9
$$

$$
0 . 6 3 \pm 0 . 0 8
$$

$$
5 . 7 8 \pm 0 . 9 9
$$

$$
2 6 . 7 3 \pm 2 . 1 6
$$

$$
\underline { { 0 . 8 1 } } \pm 0 . 0 3
$$

$$
4 . 3 6 \pm 0 . 7 4
$$

$$
0 . 0 9 \pm 0 . 0 2
$$

$$
0 . 8 0 \pm 0 . 0 2
$$

$$
0 . 5 7 \pm 0 . 0 6
$$

$$
5 . 3 2 \pm 0 . 7 6
$$

$$
0 . 4 6 \pm 0 . 0 6
$$

$$
0 . 6 9 \pm 0 . 0 8
$$

$$
2 4 . 5 6 \pm 2 . 1 3
$$

$$
5 . 6 2 \pm 0 . 9 6
$$

$$
0 . 5 0 \pm 0 . 0 6
$$

$$
\underline { { 0 . 0 8 } } \pm 0 . 0 3
$$

$$
4 . 9 7 \pm 0 . 8 0
$$

$$
0 . 1 2 \pm 0 . 0 3
$$

$$
0 . 7 8 \pm 0 . 0 5
$$

$$
2 6 . 9 9 \pm 1 . 7 7
$$

$$
0 . 4 6 \pm 0 . 0 8
$$

$$
0 . 7 9 \pm 0 . 0 3
$$

$$
4 . 1 6 \pm 0 . 6 4
$$

$$
5 . 2 8 \pm 0 . 7 7
$$

$$
0 . 5 3 \pm 0 . 0 7
$$

$$
0 . 3 6 \pm 0 . 0 6
$$

$$
0 . 1 8 \pm 0 . 0 5
$$

$$
5 . 3 6 \pm 1 . 1 4
$$

$$
2 5 . 0 4 \pm 2 . 1 8
$$

$$
5 . 3 2 \pm 0 . 9 3
$$

$$
0 . 7 4 \pm 0 . 0 6
$$

$$
0 . 2 0 \pm 0 . 0 4
$$

$$
0 . 7 3 \pm 0 . 0 4
$$

$$
5 . 2 0 \pm 0 . 7 7
$$

$$
2 8 . 5 3 \pm 1 . 6 5
$$

$$
0 . 8 1 \pm 0 . 0 3
$$

$$
\underline { { 3 . 4 1 } } \pm 0 . 4 5
$$

$$
\underline { { 0 . 6 3 } } \pm 0 . 0 7
$$

$$
0 . 7 6 \pm 0 . 0 4
$$

$$
\underline { { 0 . 3 1 } } \pm 0 . 0 6
$$

$$
\overline { { 2 5 . 6 5 } } \pm 2 . 2 3
$$

$$
4 . 8 8 \pm 0 . 8 5
$$

$$
0 . 0 9 \pm 0 . 0 4
$$

$$
0 . 2 7 \pm 0 . 0 3
$$

$$
\underline { { 4 . 2 9 } } \pm 0 . 6 0
$$

$$
0 . 6 5 \pm 0 . 0 4
$$

$$
0 . 7 9 \pm 0 . 0 4
$$

$$
2 5 . 9 5 \pm 2 . 2 9
$$

$$
4 . 7 4 \pm 0 . 8 9
$$

$$
0 . 2 3 \pm 0 . 0 5
$$

$$
0 . 3 8 \pm 0 . 0 5
$$

$$
\underline { { 0 . 5 6 } } \pm 0 . 0 5
$$

$$
2 4 . 8 2 \pm 2 . 2 4
$$

$$
\overline { { 4 . 9 9 } } \pm 0 . 7 6
$$

$$
0 . 7 2 \pm 0 . 0 6
$$

$$
5 . 4 1 \pm 0 . 9 6
$$

$$
0 . 1 3 \pm 0 . 0 5
$$

$$
0 . 1 4 \pm 0 . 0 2
$$

$$
0 . 7 7 \pm 0 . 0 2
$$

$$
2 4 . 5 2 \pm 2 . 0 5
$$

$$
5 . 1 0 \pm 0 . 8 4
$$

$$
0 . 6 7 \pm 0 . 0 8
$$

$$
0 . 3 7 \pm 0 . 0 6
$$

$$
5 . 6 3 \pm 0 . 9 0
$$

$$
0 . 1 2 \pm 0 . 0 3
$$

$$
0 . 7 8 \pm 0 . 0 3
$$

$$
0 . 4 7 \pm 0 . 0 8
$$

$$
5 . 1 0 \pm 0 . 7 5
$$

$$
{ \bf 3 1 . 3 7 \pm 1 . 9 0 }
$$

$$
{ \bf 0 . 8 3 \pm 0 . 0 5 }
$$

$$
{ \bf 2 . 6 3 \pm 0 . 5 0 }
$$

$$
5 . 3 4 \pm 0 . 7 7
$$

$$
{ \bf 0 . 8 3 \pm 0 . 0 3 }
$$

$$
{ \bf 0 . 1 1 \pm 0 . 0 3 }
$$

$$
{ \bf 0 . 0 7 \pm 0 . 0 2 }
$$

$$
{ \bf 2 . 4 6 \pm 0 . 4 2 }
$$

WV3 REDUCED-RESOLUTION RESULTS UNDER HEVC (MEAN ± STD.)
<table><tr><td>Method</td><td>Ratio ↑</td><td> $\mathrm { P S N R } \pm \mathrm { S t d . } \uparrow$ </td><td> $\mathrm { S S I M } \pm \mathrm { S t d . } \uparrow$ </td><td> $\mathrm { E R G A S } \pm \mathrm { S t d . } \downarrow$ </td><td> $\mathrm { H Q N R } \pm \mathrm { S t d . ~ } \uparrow$ </td><td> $D _ { \lambda } \pm { \mathrm { S t d . ~ } } \downarrow$ </td><td> $D _ { s } \pm \mathrm { S t d . } \downarrow$ </td><td> $\mathbf { S A M } \ ( \mathrm { d e g } ) \pm \mathbf { S t d } . \downarrow$ </td></tr><tr><td>BDSD+HEVC</td><td></td><td> $2 6 . 5 3 \pm 3 . 6 5$ </td><td> $0 . 6 0 \pm 0 . 1 4$ </td><td> $9 . 5 6 \pm 2 . 0 5$ </td><td> $0 . 3 1 \pm 0 . 0 8$ </td><td> $0 . 5 0 \pm 0 . 1 0$ </td><td> $0 . 3 9 \pm 0 . 0 8$ </td><td> $1 0 . 7 2 \pm 2 . 5 2$ </td></tr><tr><td>FusionNet+HEVC</td><td></td><td> $2 6 . 5 9 \pm 3 . 5 5$ </td><td> $0 . 6 2 \pm 0 . 1 2$ </td><td> $9 . 3 1 \pm 2 . 0 7$ </td><td> $0 . 3 4 \pm 0 . 0 9$ </td><td> $0 . 4 9 \pm 0 . 1 1$ </td><td> $0 . 3 4 \pm 0 . 0 7$ </td><td>9.71 ± 2.36</td></tr><tr><td>GLP+HEVC</td><td></td><td> $2 6 . 7 9 \pm 3 . 5 8$ </td><td> $0 . 6 2 \pm 0 . 1 3$ </td><td> $9 . 2 4 \pm 1 . 9 7$ </td><td> $0 . 3 1 \pm 0 . 0 7$ </td><td> $0 . 5 1 \pm 0 . 0 9$ </td><td> $0 . 3 7 \pm 0 . 0 7$ </td><td> $9 . 5 2 \pm 2 . 3 0$ </td></tr><tr><td>GS+HEVC</td><td></td><td> $2 7 . 2 2 \pm 3 . 5 3$ </td><td> $0 . 6 4 \pm 0 . 1 2$ </td><td> $8 . 8 2 \pm 1 . 9 1$ </td><td> $0 . 3 8 \pm 0 . 1 0$ </td><td> $0 . 4 6 \pm 0 . 1 1$ </td><td> $0 . 3 1 \pm 0 . 0 6$ </td><td> $9 . 7 9 \pm 2 . 3 6$ </td></tr><tr><td>MF+HEVC</td><td>6.69×</td><td> $2 6 . 9 2 \pm 3 . 5 8$ </td><td> $0 . 6 2 \pm 0 . 1 3$ </td><td> $9 . 1 3 \pm 2 . 0 0$ </td><td> $0 . 3 2 \pm 0 . 0 9$ </td><td> $0 . 5 1 \pm 0 . 1 0$ </td><td> $0 . 3 5 \pm 0 . 0 7$ </td><td>9.35 ± 2.26</td></tr><tr><td>U2Net+HEVC</td><td></td><td> $2 6 . 6 5 \pm 3 . 5 4$ </td><td> $0 . 6 1 \pm 0 . 1 3$ </td><td> $9 . 3 6 \pm 2 . 0 5$ </td><td> $0 . 3 3 \pm 0 . 0 9$ </td><td> $0 . 4 9 \pm 0 . 1 1$ </td><td> $0 . 3 6 \pm 0 . 0 7$ </td><td>9.48 ± 2.29</td></tr><tr><td>WFANet+HEVC</td><td></td><td> $2 6 . 2 4 \pm 3 . 7 0$ </td><td> $0 . 5 9 \pm 0 . 1 5$ </td><td> $9 . 8 2 \pm 2 . 1 7$ </td><td> $0 . 2 7 \pm 0 . 0 7$ </td><td>0.55 ± 0.09</td><td> $0 . 4 1 \pm 0 . 0 8$ </td><td>9.65 ± 2.35</td></tr><tr><td>ZS-PAN+HEVC</td><td></td><td> $2 6 . 5 7 \pm 3 . 4 \bar { 4 }$ </td><td> $0 . 6 0 \pm 0 . 1 4$ </td><td> $9 . { \overset { . } { 4 } } 0 \overset { - } \pm 1 . 8 5$ </td><td> $0 . 2 8 \pm 0 . 0 9$ </td><td> $0 . 5 4 \stackrel { - } { \pm } 0 . 1 1$ </td><td> $0 . 4 0 \pm 0 . 0 8$ </td><td> $9 . 6 0 \pm 2 . 3 2$ </td></tr><tr><td></td><td></td><td> $2 6 . 8 3 \pm 3 . 7 0$ </td><td> $0 . 6 3 \pm 0 . 1 3$ </td><td>9.18 ± 2.19</td><td> $\bar { 0 } . 3 \bar { 7 } \pm 0 . \bar { 0 9 }$ </td><td> $0 . 4 6 \pm 0 . 1 1$ </td><td> $0 . 3 2 \pm 0 . 0 7$ </td><td> $1 0 . 1 3 \pm 3 . 1 2$ </td></tr><tr><td>ADWM+HEVC ARConv+HEVC</td><td></td><td> $2 6 . 6 4 \pm 3 . 5 9$ </td><td> $0 . 6 1 \pm 0 . 1 4$ </td><td> $9 . 3 9 \pm 2 . 0 0$ </td><td> $0 . 3 0 \pm 0 . 0 8$ </td><td> $0 . 5 2 \pm 0 . 1 0$ </td><td> $0 . 3 8 \pm 0 . 0 8$ </td><td> $9 . 4 9 \pm 2 . 3 1$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TOP (ours)</td><td>33×</td><td> ${ \bf 3 0 . 3 2 \pm 2 . 8 3 }$ </td><td> ${ \bf 0 . 8 2 \pm 0 . 0 4 }$ </td><td> ${ \bf 6 . 0 0 \pm 1 . 2 3 }$ </td><td> ${ \bf 0 . 7 6 \pm 0 . 1 4 }$ </td><td> ${ \bf 0 . 1 9 \pm 0 . 1 1 }$ </td><td> $\mathbf { 0 . 0 8 \pm 0 . 0 6 }$ </td><td> ${ \bf 6 . 0 0 \pm 1 . 3 8 }$ </td></tr><tr><td>TOP (ours)</td><td>66×</td><td> $2 8 . 8 8 \pm 3 . 0 6$ </td><td>0.75 ± 0.07</td><td> ${ \underline { { 7 . 1 3 } } } \pm 1 . 4 8$ </td><td> $\underline { { 0 . 6 9 } } \pm 0 . 1 3$ </td><td> $0 . 2 3 \pm 0 . 1 1$ </td><td> $\underline { { 0 . 1 1 } } \pm 0 . 0 5$ </td><td>6.69 ± 1.56</td></tr></table>

TABLE VI

GF2 REDUCED-RESOLUTION RESULTS UNDER HEVC (MEAN ± STD.)
<table><tr><td>Method</td><td>Ratio ↑</td><td> $\mathrm { P S N R } \pm \mathrm { S t d . } \uparrow$ </td><td> $\mathrm { S S I M } \pm \mathrm { S t d . } \uparrow$ </td><td> $\mathrm { E R G A S } \pm \mathrm { S t d . } \downarrow$ </td><td> $\mathrm { H Q N R } \pm \mathrm { S t d . ~ } \uparrow$ </td><td> $D _ { \lambda } \pm { \mathrm { S t d . ~ } } \downarrow$ </td><td> $D _ { s } \pm \mathsf { S t d . } \downarrow$ </td><td> $\overline { { \mathrm { S A M ~ ( d e g ) } \pm \mathrm { S t d . ~ } \downarrow } }$ </td></tr><tr><td>BDSD+HEVC</td><td rowspan="9"></td><td> $2 8 . 0 3 \pm 1 . 9 1$ </td><td> $0 . 7 0 \pm 0 . 0 7$ </td><td> $3 . 7 7 \pm 0 . 6 5$ </td><td> $0 . 4 3 \pm 0 . 0 6$ </td><td>0.40 ± 0.05</td><td> $0 . 2 9 \pm 0 . 0 6$ </td><td> $3 . 9 5 \pm 0 . 5 9$ </td></tr><tr><td>FusionNet+HEVC</td><td>28.68 ± 1.84</td><td> $0 . 7 3 \pm 0 . 0 6$ </td><td> $3 . 4 7 \pm 0 . 5 6$ </td><td> $0 . 5 0 \pm 0 . 0 6$ </td><td> $0 . 3 7 \pm 0 . 0 5$ </td><td>0.21 ± 0.04</td><td>3.85 ± 0.57</td></tr><tr><td>GLP+HEVC</td><td> $2 7 . 7 8 \pm 1 . 8 6$ </td><td> $\overline { { 0 . 6 9 } } \pm 0 . 0 7$ </td><td> $3 . 8 8 \pm 0 . 6 5$ </td><td> $0 . 3 7 \pm 0 . 0 6$ </td><td> $0 . 4 6 \pm 0 . 0 5$ </td><td> $0 . 3 2 \pm 0 . 0 6$ </td><td> $3 . 9 1 \pm 0 . 5 7$ </td></tr><tr><td>GS+HEVC</td><td> $2 8 . 8 9 \pm 1 . 7 1$ </td><td> $0 . 7 3 \pm 0 . 0 6$ </td><td> $3 . 3 6 \pm 0 . 5 4$ </td><td> $\underline { { 0 . 6 1 } } \pm 0 . 0 5$ </td><td> $0 . 3 1 \pm 0 . 0 4$ </td><td> $\underline { { 0 . 1 1 } } \pm 0 . 0 5$ </td><td> $3 . 8 1 \pm 0 . 6 0$ </td></tr><tr><td>MF+HEVC</td><td> $2 8 . 1 6 \pm 1 . 8 9$ </td><td> $0 . 7 1 \pm 0 . 0 7$ </td><td> $3 . 7 2 \pm 0 . 6 2$ </td><td> $\overline { { 0 . 4 3 } } \pm 0 . 0 6$ </td><td> $0 . 4 2 \pm 0 . 0 5$ </td><td> $0 . 2 6 \pm 0 . 0 6$ </td><td> $3 . 8 7 \pm 0 . 5 7$ </td></tr><tr><td>U2Net+HEVC</td><td> $\underline { { 2 9 . 1 3 } } \pm 1 . 7 2$ </td><td> $0 . 7 2 \pm 0 . 0 6$ </td><td> $\underline { { 3 . 2 5 } } \pm 0 . 5 2$ </td><td> $0 . 6 1 \pm 0 . 0 5$ </td><td> $\underline { { 0 . 2 9 } } \pm 0 . 0 4$ </td><td> $0 . 1 4 \pm 0 . 0 4$ </td><td> $\underline { { 3 . 5 5 } } \pm 0 . 4 9$ </td></tr><tr><td>WFANet+HEVC</td><td> $\overline { { 2 8 . 0 5 } } \pm 1 . 9 1$ </td><td> $0 . 6 9 \pm 0 . 0 7$ </td><td> $\overline { { 3 . 7 2 } } \pm 0 . 6 1$ </td><td> $0 . 4 1 \pm 0 . 0 5$ </td><td> $\overline { { 0 . 4 2 } } \pm 0 . 0 5$ </td><td> $0 . 2 9 \pm 0 . 0 5$ </td><td> $\overline { { 3 . 9 2 } } \pm 0 . 5 6$ </td></tr><tr><td>ADWM+HEVC</td><td> $2 8 . 5 2 \pm 1 . 8 7$ </td><td> $0 . 7 2 \pm 0 . 0 6$ </td><td> $3 . 5 5 \pm 0 . 5 8$ </td><td> $0 . 4 7 \pm 0 . 0 5$ </td><td> $0 . 3 8 \pm 0 . 0 5$ </td><td> $0 . 2 4 \pm 0 . 0 4$ </td><td> $3 . 8 6 \pm 0 . 5 7$ </td></tr><tr><td>ARConv+HEVC</td><td> $2 7 . 5 0 \pm 1 . 9 8$ </td><td> $0 . 6 7 \pm 0 . 0 8$ </td><td> $3 . 9 8 \pm 0 . 6 8$ </td><td> $0 . 3 4 \pm 0 . 0 6$ </td><td> $0 . 4 7 \pm 0 . 0 5$ </td><td> $0 . 3 6 \pm 0 . 0 6$ </td><td> $3 . 9 1 \pm 0 . 5 7$ </td></tr><tr><td>ZS-PAN+HEVC</td><td></td><td>27.72 ± 1.85</td><td>0.69 ± 0.07</td><td> $3 . 8 8 \pm 0 . 6 4$ </td><td> $0 . 3 7 \pm 0 . 0 5$ </td><td> $0 . 4 5 \pm 0 . 0 5$ </td><td> $0 . 3 3 \pm 0 . 0 5$ </td><td>3.93 ± 0.57</td></tr><tr><td>TOP (ours)</td><td>50×</td><td> ${ \bf 3 1 . 3 7 \pm 1 . 9 0 }$ </td><td> ${ \bf 0 . 8 3 \pm 0 . 0 5 }$ </td><td> ${ \bf 2 . 6 3 \pm 0 . 5 0 }$ </td><td> ${ \bf 0 . 8 3 \pm 0 . 0 3 }$ </td><td>0.11 ± 0.03</td><td> ${ \bf 0 . 0 7 \pm 0 . 0 2 }$ </td><td> ${ \bf 2 . 4 6 \pm 0 . 4 2 }$ </td></tr></table>

TABLE VII

PAVIA RESULTS UNDER JPEG2000 (MEAN OVER TWO FULL TEST SCENES; STD. OMITTED)
<table><tr><td>Method</td><td>Ratio ↑</td><td>PSNR ↑</td><td>SSIM ↑</td><td>ERGAS↓</td><td>HQNR ↑</td><td> $D _ { \lambda } \downarrow$ </td><td> $D _ { s } \downarrow$ </td><td>SAM (deg) ↓</td></tr><tr><td>AWLP+JPEG2000 [22]</td><td rowspan="10"></td><td>26.15</td><td>0.65</td><td>8.94</td><td>0.76</td><td>0.16</td><td>0.09</td><td>10.91</td></tr><tr><td>BDSD-PC+JPEG2000 [46]</td><td>27.02</td><td>0.65</td><td>8.15</td><td>0.83</td><td>0.15</td><td>0.03</td><td>9.69</td></tr><tr><td>BT-H+JPEG2000 [47]</td><td>27.25</td><td>0.72</td><td>8.60</td><td>0.66</td><td>0.30</td><td>0.06</td><td>10.16</td></tr><tr><td>GSA+JPEG2000 [20]</td><td>28.09</td><td>0.69</td><td>7.54</td><td>0.78</td><td>0.16</td><td>0.07</td><td>10.20</td></tr><tr><td>MF+JPEG2000 [23]</td><td>26.72</td><td>0.67</td><td>8.39</td><td>0.78</td><td>0.16</td><td>0.07</td><td>10.00</td></tr><tr><td>PRACS+JPEG2000 [21]</td><td>26.58</td><td>0.62</td><td>8.59</td><td>0.81</td><td>0.15</td><td>0.05</td><td>9.73</td></tr><tr><td>uTDSP+JPEG2000 [35]</td><td>25.26</td><td>0.48</td><td>10.28</td><td>0.69</td><td>0.15</td><td>0.19</td><td>12.40</td></tr><tr><td>DIP-HyperKite+JPEG2000 [32]</td><td>27.40</td><td>0.69</td><td>7.80</td><td>0.82</td><td>0.15</td><td>0.03</td><td>9.68</td></tr><tr><td>HyperTransformer+JPEG2000 [30]</td><td>19.19 26.62</td><td>0.21 0.64</td><td>20.50 8.46</td><td>0.44 0.79</td><td>0.25 0.15</td><td>0.42 0.06</td><td>34.43 10.34</td></tr><tr><td>R-PNN+JPEG2000 [33] DM-ZS+JPEG2000 [36]</td><td>26.18</td><td>0.67</td><td>9.67</td><td>0.75</td><td>0.20</td><td>0.06</td><td>11.70</td></tr><tr><td>ρ-PNN+JPEG2000 [34]</td><td>25.14</td><td>0.58</td><td>10.00</td><td>0.75</td><td>0.14</td><td>0.13</td><td>11.31</td></tr><tr><td>TOP (ours)</td><td>23.97×</td><td>29.72</td><td>0.83</td><td>5.91</td><td>0.85</td><td>0.14</td><td>0.02</td><td>6.39</td></tr></table>

into a tight binary bottleneck. We therefore code the fused binary latent that TOP downlinks for WV3, which uses 0.0625 bppb with direct bit packing, with a recent 10.329096Mparameter dictionary-based entropy model [51] and rANS. As shown in Fig. 8(c), the payload decreases only from 0.062500 to 0.060260 bppb (about 3.58%), at the cost of 74.769797 additional GFLOPs/sample for probability prediction alone, plus GPU synchronization, data transfer, CPU rANS coding, and packet assembly on the sender side. This overhead is disproportionate to the gain, which supports the entropycoding-free design of TOP.

![](images/efcd878560d96452bfbefe511d826d44c551b69b86a6a60035b260a9828b78c7.jpg)

![](images/98709fcc9cf21f66dd285abe74b76d8ccdbef616acffa1566025976856013547.jpg)

![](images/6e216ba4564223730de921698734da95bbb46546497a80c59f56e3a8669d501f.jpg)

![](images/58037343f71643c1cf13ca1a3f7b6f7e35972ce09c6f077d3a1ca80b37a3f9eb.jpg)  
Fig. 7. Spectral profile comparison on WV3, GF2, and Pavia. (a, b) WV3 with TOP at 33× and 66×, (c) GF2 at 50×, and (d) Pavia at 23.97×. Upper plots: normalized band-wise mean over a fixed ROI; lower strips: deviation from GT; best-ranked baseline of each method family.

![](images/ed4f9e93c5cf3fd7092498fc23772a3e706d5a46afb1011736473004c7ddb63f.jpg)

![](images/442d47930029e81a9572aa6f57b974d7896be17cb460f2f75bd00a257d83aadd.jpg)  
Fig. 8. Ablation studies of the key design choices of TOP. WV3 reduced-resolution data; blue: design adopted in TOP, gray: alternative. (a) Backbone (HR-MSI reconstruction). (b) BSQ: reconstruction–fusion PSNR gap (left), payload and fusion-module complexity (right). (c) Additional entropy model with rANS. (d) Degradation under random bit flips after stages 2 and 3. Bars in (b) and (c) are scaled per pair. Values are listed in Sec. S8 of the Supplementar Material.

## D. Stage-3 Training Strategy on Robustness

The fused code is both the output of onboard fusion and the downlink payload, so a wrong bit may come from imperfect latent prediction or from transmission errors. We randomly flip 3% or 5% of the signs of the fused code on the 20 WV3 reduced-resolution test scenes and decode it with TOBD after stage 2 or stage 3, keeping the encoder, latent fusion module, and unperturbed code unchanged. As shown in Fig. 8(d), stage-3 training reduces the PSNR drop from 3.24 to 0.76 dB at 3% flips and from 5.17 to 1.61 dB at 5% flips. Under the same flips, stage 2 increases SAM by 0.36 and 0.64 degrees, whereas after stage 3 SAM changes by only −0.11 and −0.07 degrees. The small negative values can be interpreted as negligible sample-level variation rather than a benefit from corruption; the relevant result is that spectral fidelity no longer deteriorates systematically. Bit-flip training therefore makes TOBD markedly less sensitive to bit errors.

## VI. DISCUSSION

## A. Mechanism of Latent-Space Fusion

TOP works because one binary latent space serves reconstruction, fusion, and transmission. In stage 1, the shared tokenizer must reconstruct PAN, LR-MSI/LR-HSI, and HR-MSI/HR-HSI images under the same binary capacity, so it keeps the structures needed for both spatial detail and spectral recovery and discards pixel-level redundancy that is costly to transmit. In this space, the latent fusion module only has to predict a binary state at each aligned latent location, instead of synthesizing every output pixel and band.

The BSQ ablation shows the cost and the benefit of this design (Sec. V-B). Without BSQ, the tokenizer reconstructs better on its own, but the fusion target is continuous. It then needs a much larger fusion module and still leaves a larger reconstruction–fusion gap. BSQ gives up part of the standalone reconstruction capacity and obtains a target that is compact and easy to predict. The training stages build on this. Stage 2 fine-tunes TOBD on the HR code, and stage 3 adds bit-flip augmentation so that TOBD tolerates wrong bits from imperfect latent prediction or from the channel. With TOBT frozen, latent-fusion training then makes the onboard prediction match the HR code. The fused code is therefore already the payload for the channel, not an intermediate image that must be compressed again.

## B. Improvements Over Existing Pipelines

Ground-side fusion after compression and onboard pixeldomain fusion before compression are both limited by an objective mismatch. In the first, the codec discards information before fusion, and the fusion model is usually trained on clean inputs, so a better fusion model cannot recover what the downlink has already removed. In the second, the fusion model produces a full-resolution image without knowing how a generic codec will quantize it, and the codec minimizes image distortion without knowing which structures matter for pansharpening. Either way, two stages with different objectives are concatenated.

TOP removes this mismatch by making the fused latent code the payload. Tokenization sets what can be represented within the bit budget, onboard fusion predicts that representation, and the ground decoder reconstructs the image from the same code that was optimized during training. This agrees with the results against compress-then-fuse baselines in the main text and against onboard fusion followed by JPEG2000 or HEVC in Sec. S9 of the Supplementary Material. The gain therefore comes from coordinating representation, fusion, and transmission, not from one fusion network or one codec alone.

## C. Deployment Flexibility

The binary payload keeps onboard processing simple. It ends with sign selection and bit packing, and it needs neither full-resolution reconstruction nor a learned entropy model (Sec. V-C). The fixed payload size is known before transmission, which helps buffer allocation and link scheduling, and decoding stays on the ground.

The shared encoder supports two transmission policies. By default, TOP sends the fused latent. If downstream tasks need the source observations, the satellite can instead send the PAN and spectral latents, and the ground station reconstructs them and applies an existing fusion method. Table VIII shows that this policy remains viable: on TOBT-decoded inputs, FusionNet reaches 29.95 dB on GF2 at 25.0× and 29.32 dB on WV3 at 16.5×. These ratios are about half those of TOP because two equal-sized source latents are sent. TOP sends only the fused latent, at 50× and 33×, while improving PSNR by 1.42 dB on GF2 and 1.00 dB on WV3. The choice is therefore explicit: transmit the source latents for reuse, or transmit the fused latent for the best rate–quality trade-off.

TOP also has two practical limitations: (1) the tokenizer is trained per sensor, and (2) training takes several stages. Sec. S10 of the Supplementary Material discusses them.

TABLE VIII  
GROUND-SIDE FUSION AFTER TOBT-ONLY TRANSMISSION (MEAN ± STD.)
<table><tr><td>Dataset</td><td>Method</td><td>Ratio ↑</td><td>PSNR ↑</td><td>SAM (deg) ↓</td><td>HQNR ↑</td></tr><tr><td rowspan="3">GF2</td><td>FusionNet [6]</td><td>25.0×</td><td> $2 9 . 9 5 \pm 1 . 6 3$ </td><td> $3 . 3 3 \pm 0 . 4 8$ </td><td> $0 . 7 2 \pm 0 . 0 4$ </td></tr><tr><td>WFANet [8]</td><td>25.0×</td><td> $2 9 . 1 9 \pm 1 . 7 7$ </td><td> $3 . 4 8 \pm 0 . 4 7$ </td><td> $0 . 6 6 \pm 0 . 0 3$ </td></tr><tr><td>TOP (ours)</td><td>50×</td><td> ${ \bf 3 1 . 3 7 \pm 1 . 9 0 }$ </td><td> ${ \bf 2 . 4 6 \pm 0 . 4 2 }$ </td><td> ${ \bf 0 . 8 3 \pm 0 . 0 3 }$ </td></tr><tr><td rowspan="3">WV3</td><td>FusionNet [6]</td><td>16.5×</td><td> $2 9 . 3 2 \pm 2 . 6 2$ </td><td> $7 . 9 3 \pm 1 . 8 6$ </td><td> $0 . 6 5 \pm 0 . 1 5$ </td></tr><tr><td>WFANet [8]</td><td>16.5×</td><td> $2 7 . 6 5 \pm 3 . 2 4$ </td><td> $8 . 2 4 \pm 1 . 9 4$ </td><td> $0 . 4 9 \pm 0 . 1 1$ </td></tr><tr><td>TOP (ours)</td><td>33×</td><td> ${ \bf 3 0 . 3 2 \pm 2 . 8 3 }$ </td><td> ${ \bf 6 . 0 0 \pm 1 . 3 8 }$ </td><td> ${ \bf 0 . 7 6 \pm 0 . 1 4 }$ </td></tr></table>

## VII. CONCLUSION

This work presents TOP, which treats pansharpening for satellite downlink as a joint design of compact latent coding, onboard fusion, and transmission. TOBE with a BSQ bottleneck produces a binary latent code that can be reconstructed, predicted, and bit-packed directly. TOBD decodes the fused code on the ground and can tolerate occasional bit errors. On WV3, GF2, and Pavia datasets, TOP produces higher fusion quality than pansharpening methods paired with common codecs, at compression ratios 1.2 to 9.9 times those of these baselines. For example, on WV3 it reaches 30.32 dB PSNR at 33×, whereas the best JPEG2000 baseline reaches 27.70 dB at 21.26×. These findings suggest that future remote-sensing systems could benefit from co-designing task processing and transmission, rather than treating downlink transmission as a separate stage. Future work will simplify the training procedure and improve generalization across sensors.

## REFERENCES

[1] L.-J. Deng, G. Vivone, W. Guo, M. Dalla Mura, and J. Chanussot, “A variational pansharpening approach based on reproducible kernel Hilbert space and Heaviside function,” IEEE Transactions on Image Processing, vol. 27, no. 9, pp. 4330–4344, 2018.

[2] Z.-C. Wu, T.-Z. Huang, L.-J. Deng, J. Huang, J. Chanussot, and G. Vivone, “LRTCFPan: Low-rank tensor completion based framework for pansharpening,” IEEE Transactions on Image Processing, vol. 32, pp. 1640–1655, 2023.

[3] C. A. Laben and B. V. Brower, “Process for enhancing the spatial resolution of multispectral imagery using pan-sharpening,” U.S. Patent 6,011,875, 2000.

[4] A. Garzelli, F. Nencini, and L. Capobianco, “Optimal MMSE pan sharpening of very high resolution multispectral images,” IEEE Transactions on Geoscience and Remote Sensing, vol. 46, no. 1, pp. 228–236, 2008.

[5] B. Aiazzi, L. Alparone, S. Baronti, and A. Garzelli, “Context-driven fusion of high spatial and spectral resolution images based on oversampled multiresolution analysis,” IEEE Transactions on Geoscience and Remote Sensing, vol. 40, no. 10, pp. 2300–2312, 2002.

[6] L.-J. Deng, G. Vivone, C. Jin, and J. Chanussot, “Detail injectionbased deep convolutional neural networks for pansharpening,” IEEE Transactions on Geoscience and Remote Sensing, vol. 59, no. 8, pp. 6995–7010, 2021.

[7] S. Peng, C. Guo, X. Wu, and L.-J. Deng, “U2Net: A general framework with spatial-spectral-integrated double U-net for image fusion,” in Proceedings of the ACM International Conference on Multimedia (ACM MM), 2023, pp. 3219–3227.

[8] J. Huang, R. Huang, J. Xu, S. Peng, Y. Duan, and L.-J. Deng, “Wavelet-assisted multi-frequency attention network for pansharpening,” in Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), vol. 39, 2025, pp. 3662–3670.

[9] S. Xiang, Q. Liang, and P. Tang, “Task-oriented compression framework for remote sensing satellite data transmission,” IEEE Transactions on Industrial Informatics, vol. 20, no. 3, pp. 3487–3496, 2024.

[10] T. Lv, H. Jiang, and Y. Liu, “STAFuse: Scene-text aggregation-guided composite degradation-robust infrared and visible image fusion,” IEEE Transactions on Image Processing, vol. 35, pp. 8603–8618, 2026.

[11] D. S. Taubman and M. W. Marcellin, “JPEG2000: Standard for interactive imaging,” Proceedings of the IEEE, vol. 90, no. 8, pp. 1336–1357, 2002.

[12] G. J. Sullivan, J.-R. Ohm, W.-J. Han, and T. Wiegand, “Overview of the high efficiency video coding (HEVC) standard,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 22, no. 12, pp. 1649– 1668, 2012.

[13] J. Balle, D. Minnen, S. Singh, S. J. Hwang, and N. Johnston, “Variational´ image compression with a scale hyperprior,” in International Conference on Learning Representations (ICLR), 2018.

[14] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, “Swin transformer: Hierarchical vision transformer using shifted windows,” in Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 10 012–10 022.

[15] Y. Zhao, Y. Xiong, and P. Krahenb¨ uhl, “Image and video tokenization¨ with binary spherical quantization,” in International Conference on Learning Representations (ICLR), 2025.

[16] J. Duda, “Asymmetric numeral systems: Entropy coding combining speed of Huffman coding with compression rate of arithmetic coding,” arXiv preprint arXiv:1311.2540, 2013.

[17] X. Li, X. Li, T. Tan, H. Li, and T. Ye, “UMCFuse: A unified multiple complex scenes infrared and visible image fusion framework,” IEEE Transactions on Image Processing, vol. 34, pp. 6231–6245, 2025.

[18] J. Luo, W. Ren, X. Gao, and X. Cao, “Multi-exposure image fusion via deformable self-attention,” IEEE Transactions on Image Processing, vol. 32, pp. 1529–1540, 2023.

[19] J. Li, X. Guo, G. Lu, B. Zhang, Y. Xu, F. Wu, and D. Zhang, “DRPL: Deep regression pair learning for multi-focus image fusion,” IEEE Transactions on Image Processing, vol. 29, pp. 4816–4831, 2020.

[20] B. Aiazzi, S. Baronti, and M. Selva, “Improving component substitution pansharpening through multivariate regression of MS+Pan data,” IEEE Transactions on Geoscience and Remote Sensing, vol. 45, no. 10, pp. 3230–3239, 2007.

[21] J. Choi, K. Yu, and Y. Kim, “A new adaptive component-substitutionbased satellite image fusion by using partial replacement,” IEEE Transactions on Geoscience and Remote Sensing, vol. 49, no. 1, pp. 295–309, 2011.

[22] X. Otazu, M. Gonzalez-Audicana, O. Fors, and J. Nunez, “Introduction of sensor spectral response into image fusion methods: Application to wavelet-based methods,” IEEE Transactions on Geoscience and Remote Sensing, vol. 43, no. 10, pp. 2376–2385, 2005.

[23] R. Restaino, G. Vivone, M. Dalla Mura, and J. Chanussot, “Fusion of multispectral and panchromatic images based on morphological operators,” IEEE Transactions on Image Processing, vol. 25, no. 6, pp. 2882–2895, 2016.

[24] T. Wang, F. Fang, F. Li, and G. Zhang, “High-quality bayesian pansharpening,” IEEE Transactions on Image Processing, vol. 28, no. 1, pp. 227–239, 2019.

[25] H. Lu, Y. Yang, S. Huang, W. Tu, and W. Wan, “A unified pansharpening model based on band-adaptive gradient and detail correction,” IEEE Transactions on Image Processing, vol. 31, pp. 918–933, 2022.

[26] J. Huang, H. Chen, J. Ren, S. Peng, and L.-J. Deng, “A general adaptive dual-level weighting mechanism for remote sensing pansharpening,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 7447–7456.

[27] X. Wang, Z. Zheng, J. Shao, Y. Duan, and L.-J. Deng, “Adaptive rectangular convolution for remote sensing pansharpening,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 17 872–17 881.

[28] B. Xu, Y. Zheng, L. Wang, S. Mazhar, Z. Huang, and J. Liu, “DTEA: Degradation-aware taylor expansion approximation network for pansharpening,” IEEE Transactions on Image Processing, vol. 35, pp. 7581– 7596, 2026.

[29] Q. Cao, L.-J. Deng, W. Wang, J. Hou, and G. Vivone, “Zero-shot semisupervised learning for pansharpening,” Information Fusion, vol. 101, p. 102001, 2024.

[30] W. G. C. Bandara and V. M. Patel, “HyperTransformer: A textural and spectral feature fusion transformer for pansharpening,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 1767–1777.

[31] N. Wang, A. Guo, R. Dian, and S. Li, “Equivariant high-resolution hyperspectral imaging via mosaiced and PAN image fusion,” IEEE Transactions on Image Processing, vol. 35, pp. 1246–1260, 2026.

[32] W. G. C. Bandara, J. M. J. Valanarasu, and V. M. Patel, “Hyperspectral pansharpening based on improved deep image prior and residual reconstruction,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–16, 2022.

[33] G. Guarino, M. Ciotola, G. Vivone, and G. Scarpa, “Band-wise hyperspectral image pansharpening using CNN model propagation,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–18, 2024.

[34] G. Guarino, M. Ciotola, G. Vivone, G. Poggi, and G. Scarpa, “Zero-shot hyperspectral pansharpening using hysteresis-based tuning for spectral quality control,” IEEE Transactions on Geoscience and Remote Sensing, vol. 63, pp. 1–19, 2025.

[35] H. Jiang and Z. Chen, “Hyperspectral pansharpening with transformerbased spectral diffusion priors,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops, 2025, pp. 581–590.

[36] J.-L. Xiao, T.-Z. Huang, L.-J. Deng, G. Lin, Z. Cao, C. Li, and Q. Zhao, “Hyperspectral pansharpening via diffusion models with iteratively zero-shot guidance,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 12 669– 12 678.

[37] M. Hernandez-Cabronero, A. B. Kiely, M. Klimesh, I. Blanes, J. Ligo,´ E. Magli, and J. Serra-Sagrista, “The CCSDS 123.0-B-2 “Low-\` complexity lossless and near-lossless multispectral and hyperspectral image compression” standard: A comprehensive review,” IEEE Geoscience and Remote Sensing Magazine, vol. 9, no. 4, pp. 102–119, 2021.

[38] A. van den Oord, O. Vinyals, and K. Kavukcuoglu, “Neural discrete representation learning,” in Advances in Neural Information Processing Systems, vol. 30, 2017.

[39] F. Mentzer, D. Minnen, E. Agustsson, and M. Tschannen, “Finite scalar quantization: VQ-VAE made simple,” in International Conference on Learning Representations (ICLR), 2024.

[40] L. Yu, J. Lezama, N. B. Gundavarapu, L. Versari, K. Sohn, D. Minnen, Y. Cheng, A. Gupta, X. Gu, A. G. Hauptmann, B. Gong, M.-H. Yang, I. Essa, D. A. Ross, and L. Jiang, “Language model beats diffusion— tokenizer is key to visual generation,” in International Conference on Learning Representations (ICLR), 2024.

[41] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An image is worth 16x16 words: Transformers for image recognition at scale,” in International Conference on Learning Representations (ICLR), 2021.

[42] F. A. Kruse, A. B. Lefkoff, J. W. Boardman, K. B. Heidebrecht, A. T. Shapiro, P. J. Barloon, and A. F. H. Goetz, “The spectral image processing system (SIPS)—interactive visualization and analysis of imaging spectrometer data,” Remote Sensing of Environment, vol. 44, no. 2–3, pp. 145–163, 1993.

[43] L.-J. Deng, G. Vivone, M. E. Paoletti, G. Scarpa, J. He, Y. Zhang, J. Chanussot, and A. Plaza, “Machine learning in pansharpening: A benchmark, from shallow to deep networks,” IEEE Geoscience and Remote Sensing Magazine, vol. 10, no. 3, pp. 279–315, 2022.

[44] Y.-W. Zhuo, T.-J. Zhang, J.-F. Hu, H.-X. Dou, T.-Z. Huang, and L.-J. Deng, “A deep-shallow fusion network with multidetail extractor and spectral attention for hyperspectral pansharpening,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 15, pp. 7539–7555, 2022.

[45] L. Wald, T. Ranchin, and M. Mangolini, “Fusion of satellite images of different spatial resolutions: Assessing the quality of resulting images,” Photogrammetric Engineering & Remote Sensing, vol. 63, no. 6, pp. 691–699, 1997.

[46] G. Vivone, “Robust band-dependent spatial-detail approaches for panchromatic sharpening,” IEEE Transactions on Geoscience and Remote Sensing, vol. 57, no. 9, pp. 6421–6433, 2019.

[47] S. Lolli, L. Alparone, A. Garzelli, and G. Vivone, “Haze correction for contrast-based multispectral pansharpening,” IEEE Geoscience and Remote Sensing Letters, vol. 14, no. 12, pp. 2255–2259, 2017.

[48] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: From error visibility to structural similarity,” IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600–612, 2004.

[49] G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. A. Licciardi, R. Restaino, and L. Wald, “A critical comparison among pansharpening algorithms,” IEEE Transactions on Geoscience and Remote Sensing, vol. 53, no. 5, pp. 2565–2586, 2015.

[50] B. Aiazzi, L. Alparone, S. Baronti, R. Carla, A. Garzelli, and L. Santurri,\` “Full scale assessment of pansharpening methods and data products,” in Image and Signal Processing for Remote Sensing XX, vol. 9244. SPIE, 2014, p. 924402.

[51] J. Lu, L. Zhang, X. Zhou, M. Li, W. Li, and S. Gu, “Learned image compression with dictionary-based entropy model,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 12 850–12 859.

[52] D. Hendrycks and K. Gimpel, “Gaussian error linear units (GELUs),” arXiv preprint arXiv:1606.08415, 2016.

# Supplementary Material for

# TOP: Transmission-Oriented Remote Sensing Pansharpening

Zi-Han Cao, Jing-Ze Liang, Shao-Jie Wu, Jin-Cheng Di, Liang-Jian Deng, and Gemine Vivone

This supplement details BSQ and the latent-fusion loss (Secs. S1–S2), implementation and evaluation protocols (Secs. S3–S4), and additional JPEG2000, HEVC, fullresolution, ablation, and onboard-fusion results (Secs. S5–S9). Sec. S10 discusses limitations.

In all tables, bold and underlined entries are the best and second best; repeated-sample results are mean ± standard deviation. Baseline references appear in the main manuscript. In reduced-resolution figures, prediction and error-map rows alternate; darker blue indicates lower error, and green boxes mark enlarged regions.

## S1. DETAILS OF BINARY SPHERICAL QUANTIZATION

Following BSQ [15], inference and downlink use only the hard normalized tensor B; the soft assignments below are training auxiliaries.

a) Hard Spherical Binary Code: For $\mathbf { f } _ { i } \in \mathbb { R } ^ { C _ { \mathrm { l a t } } }$ at latent location i,

$$
\begin{array} { r l } & { { \bf { b } } _ { i } = \mathrm { s i g n } ( { \bf { f } } _ { i } ) \in \{ - 1 , + 1 \} ^ { C _ { \mathrm { l a t } } } , } \\ & { { \bf { B } } = \frac { 1 } { \sqrt { C _ { \mathrm { l a t } } } } \mathrm { r e s h a p e } \big ( [ { \bf { b } } _ { 1 } , \ldots , { \bf { b } } _ { N } ] \big ) } \\ & { \phantom { { \bf { b } } _ { i } = \mathrm { s i g n } } \in \{ - 1 / \sqrt { C _ { \mathrm { l a t } } } , + 1 / \sqrt { C _ { \mathrm { l a t } } } \} ^ { C _ { \mathrm { l a t } } \times H / s \times W / s } , } \end{array}\tag{20}
$$

where $N = H W / s ^ { 2 }$ and reshape maps the $C _ { \mathrm { l a t } } \times N$ rasterordered vectors to the latent grid. For each of the three modalities, define

$$
\bar { \bf f } _ { i } = \frac { { \bf f } _ { i } } { \| { \bf f } _ { i } \| _ { 2 } + \epsilon } , \qquad { \bf q } _ { i } = \frac { { \bf b } _ { i } } { \sqrt { C _ { \mathrm { l a t } } } } ,\tag{21}
$$

with numerical constant $\epsilon > 0$ . Positive normalization preserves the sign, and the straight-through estimator (STE) is

$$
{ \mathcal { Q } } ( { \bar { \mathbf { f } } } _ { i } ) = { \bar { \mathbf { f } } } _ { i } + \mathrm { s g } ( \mathbf { q } _ { i } - { \bar { \mathbf { f } } } _ { i } ) , \qquad { \frac { \partial { \mathcal { Q } } } { \partial { \bar { \mathbf { f } } } _ { i } } } \ { \stackrel { \mathrm { s T E } } { = } } \ \mathbf { I } _ { C _ { \mathrm { l a t } } } .\tag{22}
$$

Thus, the forward value is $\mathbf { q } _ { i }$ , while the identity surrogate Jacobian is used only for back-propagation; $\mathrm { s g }$ denotes stopgradient. Bit packing stores signs, and unpacking restores the two normalized values. The corresponding implicit, unlearned codebook is

$$
\mathcal { C } _ { C _ { \mathrm { l a t } } } = \left\{ \frac { \mathbf { c } _ { n } } { \sqrt { C _ { \mathrm { l a t } } } } \biggm | \begin{array} { l } { \mathbf { c } _ { n } \in \{ - 1 , + 1 \} ^ { C _ { \mathrm { l a t } } } , } \\ { n = 1 , \ldots , 2 ^ { C _ { \mathrm { l a t } } } } \end{array} \right\} .\tag{23}
$$

b) Geometry and Packed Payload: The spherical normalization makes binary disagreement directly measurable. For two codewords, with Hamming distance $h ( \mathbf { b } _ { i } , \mathbf { b } _ { i } ^ { \prime } )$

$$
\mathbf { q } _ { i } ^ { \top } \mathbf { q } _ { i } ^ { \prime } = 1 - \frac { 2 h ( \mathbf { b } _ { i } , \mathbf { b } _ { i } ^ { \prime } ) } { C _ { \mathrm { l a t } } } , \qquad \| \mathbf { q } _ { i } - \mathbf { q } _ { i } ^ { \prime } \| _ { 2 } ^ { 2 } = \frac { 4 h ( \mathbf { b } _ { i } , \mathbf { b } _ { i } ^ { \prime } ) } { C _ { \mathrm { l a t } } } .\tag{24}
$$

A sign error therefore causes a fixed, bounded latent displacement, and latent prediction errors have the same form as transmission bit errors. The packed tensor contains $N C _ { \mathrm { l a t } } =$ $C _ { \mathrm { l a t } } H W / s ^ { 2 }$ bits, giving bppb $= \ C _ { \mathrm { l a t } } / ( C s ^ { 2 } )$ for C target bands. Its payload is fixed by $( C _ { \mathrm { l a t } } , s )$ and requires no codebook indices, floating-point magnitudes, or probability tables. The dot-product identity also keeps similarity in the fixed interval [−1, 1] across latent dimensions, so changing the rate configuration does not change the scale presented to the decoder.

c) Soft Assignment for Entropy Regularization: During training, inverse temperature $\eta > 0$ defines the nearest-code distribution

$$
\begin{array} { r l } & { p _ { i , n } = \frac { \exp ( - \eta \| \bar { \mathbf { f } } _ { i } - \widehat { \mathbf { c } } _ { n } \| _ { 2 } ^ { 2 } ) } { \sum _ { j = 1 } ^ { 2 ^ { C _ { \mathrm { l a t } } } } \exp ( - \eta \| \bar { \mathbf { f } } _ { i } - \widehat { \mathbf { c } } _ { j } \| _ { 2 } ^ { 2 } ) } } \\ & { \qquad = \frac { \exp ( 2 \eta \widehat { \mathbf { c } } _ { n } ^ { \top } \bar { \mathbf { f } } _ { i } ) } { \sum _ { j = 1 } ^ { 2 ^ { C _ { \mathrm { l a t } } } } \exp ( 2 \eta \widehat { \mathbf { c } } _ { j } ^ { \top } \bar { \mathbf { f } } _ { i } ) } , \quad \widehat { \mathbf { c } } _ { n } = \mathbf { c } _ { n } / \sqrt { C _ { \mathrm { l a t } } } . } \end{array}\tag{25}
$$

The second form follows because codeword-independent squared-norm terms cancel. The Cartesian binary codebook factorizes by channel:

$$
\begin{array} { l l } { { q _ { i , d } ^ { - } = \sigma \left( - \frac { 4 \eta } { \sqrt { C _ { \mathrm { l a t } } } } \bar { f } _ { i , d } \right) , } } & { { q _ { i , d } ^ { + } = 1 - q _ { i , d } ^ { - } , } } \\ { { \sigma ( z ) = ( 1 + e ^ { - z } ) ^ { - 1 } . } } & { { } } \end{array}\tag{26}
$$

In practice, split the channels into $G = C _ { \mathrm { l a t } } / g$ disjoint groups $\mathcal { T } _ { h }$ of $g$ channels. Let $\bar { \mathbf { f } } _ { i } ^ { ( h ) } = ( \bar { f } _ { i , d } ) _ { d \in \mathcal { T } _ { h } } , \mathbf { c } _ { h , n } \in \{ - 1 , + 1 \} ^ { g }$ for $n = 1 , \ldots , 2 ^ { g }$ , and $\widehat { \mathbf { c } } _ { h , n } = \mathbf { c } _ { h , n } / \sqrt { C _ { \mathrm { l a t } } }$ . Then

$$
p _ { i , h , n } = \frac { \exp ( 2 \eta \widehat { \mathbf { c } } _ { h , n } ^ { \top } \bar { \mathbf { f } } _ { i } ^ { ( h ) } ) } { \sum _ { j = 1 } ^ { 2 ^ { g } } \exp ( 2 \eta \widehat { \mathbf { c } } _ { h , j } ^ { \top } \bar { \mathbf { f } } _ { i } ^ { ( h ) } ) } , \qquad \mathbf { p } _ { i , h } = ( p _ { i , h , n } ) _ { n = 1 } ^ { 2 ^ { g } } .\tag{27}
$$

This marginal preserves the global $1 / \sqrt { C _ { \mathrm { l a t } } }$ scaling and reduces explicit assignment from $2 ^ { C _ { \mathrm { l a t } } }$ states to $G 2 ^ { g }$ grouped states.

d) Entropy and Commitment Terms: For the batch and latent locations $\Omega _ { \mathrm { e } } ,$

$$
\begin{array} { r } { \mathcal { H } _ { \mathrm { s a m p l e } } = \displaystyle \frac { 1 } { | \Omega _ { \mathrm { e } } | } \sum _ { i \in \Omega _ { \mathrm { e } } } \sum _ { h = 1 } ^ { G } \mathcal { H } ( \mathbf { p } _ { i , h } ) , } \\ { \mathcal { H } ( \mathbf { p } ) = - \sum _ { n } p _ { n } \log ( p _ { n } + \epsilon ) . } \end{array}\tag{28}
$$

while the entropy of average code usage is

$$
\bar { p } _ { h , n } = \frac { 1 } { | \Omega _ { \mathrm { e } } | } \sum _ { i \in \Omega _ { \mathrm { e } } } p _ { i , h , n } , \qquad \mathcal { H } _ { \mathrm { c o d e } } = \sum _ { h = 1 } ^ { G } \mathcal { H } ( \bar { \mathbf { p } } _ { h } ) ,\tag{29}
$$

where $\bar { \mathbf p } _ { h } = ( \bar { p } _ { h , n } ) _ { n = 1 } ^ { 2 ^ { g } }$ . The first term favors decisive local assignments; the second prevents collapse through broad groupcode usage. TOP minimizes

$$
\mathcal { L } _ { \mathrm { e n t r o p y } } = \frac { \zeta } { \eta } ( \gamma _ { 0 } \mathcal { H } _ { \mathrm { s a m p l e } } - \gamma \mathcal { H } _ { \mathrm { c o d e } } )\tag{30}
$$

and the commitment term

$$
\mathcal { L } _ { \mathrm { c o m m i t } } = \frac { 1 } { | \Omega _ { \mathrm { q } } | } \sum _ { i \in \Omega _ { \mathrm { q } } } \| \mathrm { s g } ( \mathbf { q } _ { i } ) - \bar { \mathbf { f } } _ { i } \| _ { 2 } ^ { 2 } ,\tag{31}
$$

where $\Omega _ { \mathrm { q } }$ is the full batch and latent grid. This moves the continuous feature toward the fixed hard code. Their combination is Eq. (10) of the main manuscript. Tab. IX gives all training coefficients; they do not change the deterministic hard code. The effective coefficients of $\mathcal { L } _ { \mathrm { c o m m i t } }$ $\mathcal { H } _ { \mathrm { s a m p l e } } ,$ , and $\mathcal { H } _ { \mathrm { c o d e } }$ are $( 5 \times 1 0 ^ { - 3 } , 5 \times 1 0 ^ { - 4 } , - 5 \times 1 0 ^ { - 4 } )$ for WV3 33×, $( 1 0 ^ { - 2 } , 1 0 ^ { - 3 } , - 1 0 ^ { - 3 } )$ for WV3 66×, and $( 0 . 1 , 0 . 1 , - 0 . 1 )$ for GF2 and Pavia.

TABLE IX  
BSQ TRAINING COEFFICIENTS; TRAINING USES HARD BSQ WITH STE,WHEREAS INFERENCE USES HARD BSQ ONLY.
<table><tr><td>Coefficient</td><td>WV3 33×</td><td>WV366×</td><td>GF2 50× Pavia</td><td> $2 3 . 9 7 \times$ </td></tr><tr><td> $\beta , \gamma _ { 0 } , \gamma , \zeta$ </td><td>0.1</td><td>0.1</td><td>1</td><td>1</td></tr><tr><td>η</td><td>1</td><td>1</td><td>1</td><td>1</td></tr><tr><td> $\lambda _ { \mathrm { Q } }$ </td><td>0.05</td><td>0.1</td><td>0.1</td><td>0.1</td></tr></table>

e) Boundary With Entropy Coding: The entropy regularizer is used only for optimization; it produces no probability model, side information, or arithmetic-coded stream. Deployment discards $p _ { i , n } , \ q _ { i , d } ^ { \pm } ,$ , and $p _ { i , h , n }$ and directly bit-packs the signs of $\widehat { \mathbf { B } } _ { \mathrm { H } }$ . A neural entropy model would additionally require probability inference, packet coding, and decoding.

## S2. BINARY CROSS-ENTROPY LOSS OF THE LATENT FUSION MODULE

Sec. III-D defines the single logit tensor O. Its first axis $a \in$ {0, 1} represents the negative or positive state, and $O _ { a , d , u , v }$ is one logit. For each target entry $B _ { \mathrm { H } , d , u , v } ,$ define

$$
y _ { d , u , v } = \mathbb { I } [ B _ { \mathrm { H } , d , u , v } > 0 ] \in \{ 0 , 1 \}\tag{32}
$$

and lift it to two complementary channels:

$$
T _ { a , d , u , v } = \mathbb { I } [ a = y _ { d , u , v } ] , \qquad a \in \{ 0 , 1 \} .\tag{33}
$$

Thus, $T _ { 0 , d , u , v } \quad = \quad 1 - y _ { d , u , v }$ and $\begin{array} { r l r } { T _ { 1 , d , u , v } } & { { } = } & { y _ { d , u , v } . } \end{array}$ The resulting $\mathbf { T } ~ \in ~ \{ 0 , 1 \} ^ { 2 \times C _ { \mathrm { l a t } } \times H / s \times W / s }$ matches O, so $f _ { \mathrm { B C E W i t h L o g i t s } } ( \mathbf { O } , \mathbf { T } )$ requires no broadcasting. For scalar logit o and target t,

$$
\ell ( o , t ) = \operatorname* { m a x } ( o , 0 ) - o t + \log ( 1 + \exp ( - | o | ) ) ,\tag{34}
$$

which evaluates the exponential only at a nonpositive argument and avoids an explicit sigmoid. The latent-fusion objective is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { f u s e } } = \displaystyle \frac { 1 } { 2 C _ { \mathrm { l a t } } N } \sum _ { a = 0 } ^ { 1 } \sum _ { d = 1 } ^ { C _ { \mathrm { l a t } } } \sum _ { u = 1 } ^ { H / s } \sum _ { v = 1 } ^ { W / s } \ell ( O _ { a , d , u , v } , T _ { a , d , u , v } ) , } \\ { N = H W / s ^ { 2 } . \qquad } \end{array}\tag{35}
$$

This is complementary one-vs-rest BCE, rather than softmax cross-entropy. The frozen values $- 1 / \sqrt { C _ { \mathrm { l a t } } }$ and $+ 1 / \sqrt { C _ { \mathrm { l a t } } }$ map to $ { \boldsymbol { a } } \quad = \quad 0$ and $ { \boldsymbol { a } } \ = \ 1$ . At inference, $\begin{array} { r l } { a _ { d , u , v } ^ { * } } & { { } = } \end{array}$ arg $\mathrm { m a x } _ { a \in \{ 0 , 1 \} } O _ { a , d , u , v }$ and $\widehat { B } _ { \mathrm { H } , d , u , v } = ( 2 a _ { d , u , v } ^ { * } - 1 ) / \sqrt { C _ { \mathrm { l a t } } }$ preserving the training state convention; no sigmoid precedes f<sub>BCEWithLogits</sub>.

## S3. IMPLEMENTATION DETAILS

All configurations use Swin-based encoders/decoders, grouped BSQ with eight channels per group, and a bandadaptive $2 \times 2$ stride-2 embedding. The kernel layout is output×input×height×width and is interpolated along the input-channel dimension. The factor s applies to both spatial axes, and stage counts include stages without scale changes. Tab. X retains all architecture and configuration-specific training values.

TABLE X  
TOP ARCHITECTURE AND CONFIGURATION-SPECIFIC TRAINING SETTINGS.
<table><tr><td>Configuration</td><td>WV3 33×</td><td>WV3 66×</td><td>GF2 50×</td><td>Pavia 23.97×</td></tr><tr><td> $\mathrm { { B a n d s } } / C _ { \mathrm { { l a t } } } / s$ </td><td>8/8/4</td><td>8/16/8</td><td>4/16/8</td><td>102/64/4</td></tr><tr><td>Stages/d0</td><td>4/64</td><td>3/128</td><td>3/128</td><td>4/64</td></tr><tr><td>Width multipliers</td><td> $[ 1 , 2 , 4 , 4 ]$ </td><td>[2, 4, 4]</td><td>[2, 4, 4]</td><td>[1, 2, 4, 4]</td></tr><tr><td>Encoder/decoder blocks</td><td>[2,4,4,2]/[3,6,6,3]</td><td>[2, 4, 2]/[2, 4, 2]</td><td>[2, 4, 2]/[2, 4, 2]</td><td> $[ 2 , 4 , 4 , 2 ] / [ 2 , 4 , 4 , 2 ]$ </td></tr><tr><td>Embedding kernel</td><td>64×8×2×2</td><td>128×8×2×2</td><td>128×4×2×2</td><td>64×102×2×2</td></tr><tr><td>BSQ groups</td><td>1</td><td>2</td><td>2</td><td>8</td></tr><tr><td>Fusion blocks/input channels</td><td>8/16</td><td>12/32</td><td>8/32</td><td>10/128</td></tr><tr><td>Stage-1 batch</td><td>8</td><td>16</td><td>16</td><td>8</td></tr><tr><td>Stage-1 MSE/BSQ weights</td><td>1/0.05</td><td>4/0.1</td><td>4/0.1</td><td>1/0.1</td></tr><tr><td>Fusion batch</td><td>2</td><td>8</td><td>8</td><td>8</td></tr></table>

The fusion module concatenates spectral and PAN codes, maps $2 C _ { \mathrm { l a t } }$ channels to two logits per target bit, and uses 64- channel residual blocks with two $3 \times 3$ convolutions, residual addition, and GELU [52]. With the tokenizer frozen, it is trained to convergence using only Sec. S2 (weight 1; imagedomain losses disabled), seed 2025.

All tokenizers use EMA weights and three stages. Stage 1 jointly trains encoder and decoder for 40,000 iterations on mixed PAN, upsampled LR spectral, and HR reference images. Stages 2 and 3 initialize from the preceding EMA, freeze the encoder, and refine the decoder for 20,000 iterations each with batch 16 and MSE/Sobel-gradient/SAM weights $1 / 0 . 1 / 0 . 1 ;$ quantizer loss is disabled. Stage 3 ramps bit-flip probability from $1 0 ^ { - 5 }$ to $3 \times 1 0 ^ { - 3 }$ over 2,000 iterations and holds it fixed. The stage-3 EMA is final. AdamW uses learning rate $2 \times 1 0 ^ { - 4 }$ 500 tokenizer warm-up iterations, and weight decay 0.05/0.01 for tokenizer/fusion. No adversarial, perceptual, or LeCAM loss is used. Training uses PyTorch, bfloat16, and one NVIDIA RTX 4090.

Payload is $C _ { \mathrm { l a t } } / s ^ { 2 } ;$ : WV3 33× uses 0.5 bit/pixel (0.0625 bppb); WV3 66× and GF2 50× use 0.25 bit/pixel (0.03125 and 0.0625 bppb); Pavia uses 4 bits/pixel, i.e., 64/(16 × $1 0 2 ) \ = \ 0 . 0 3 9 2 1 6$ bppb and 23.9688×. WV3 33× sets z\_channels=512. Pavia pads to at most 128 channels while using 102 spectral and 64 latent channels.

## S4. DATASETS AND EVALUATION PROTOCOLS

WV3 (8 bands) and GF2 (4 bands) come from PanCollection [43],<sup>1</sup> and the 102-band ROSIS Pavia Center data come from HyperPanCollection/HISI and Hyper-DSNet [44].<sup>2</sup> All use scale factor four. Tab. XI gives every split used in training and evaluation; HR denotes the common spatial size of the reference or upsampled spectral image and PAN, and LR denotes the native spectral size.

Arrays are channel-first (N, C, H, W). The keys gt/GT, lms/LMS, ms/MS, and pan/PAN denote the HR reference, upsampled LR spectral image, native LR spectral image, and single-band PAN; WV3/GF2 use lowercase and Pavia uppercase. Communication accounting uses the native ms/MS. WV3/GF2 reduced-resolution tests follow Wald’s protocol [45]; their GT supports PSNR, SSIM, ERGAS, and SAM, whereas full-resolution tests lack GT. Whenever HQNR, $D _ { \lambda }$ , and $D _ { s }$ are reported, their anchors are the original spectral and PAN observations. Pavia Full retains GT, so both metric groups are available and are averaged over its two scenes. Its MAT variable S has shape (400, 400, 102) and is permuted to (102, 400, 400) before adding the sample dimension.

TABLE XI  
DATASET SPLITS AND SPATIAL DIMENSIONS.
<table><tr><td>Dataset</td><td>Split</td><td>N</td><td>C</td><td>HR</td><td>LR</td><td>GT</td></tr><tr><td rowspan="4">WV3</td><td>Train</td><td>9714</td><td>8</td><td>64</td><td>16</td><td>Yes</td></tr><tr><td>Val.</td><td>1080</td><td>8</td><td>64</td><td>16</td><td>Yes</td></tr><tr><td>Reduced</td><td>20</td><td>8</td><td>256</td><td>64</td><td>Yes</td></tr><tr><td>Full</td><td>20</td><td>8</td><td>512</td><td>128</td><td>No</td></tr><tr><td rowspan="4">GF2</td><td>Train</td><td>19809</td><td>4</td><td>64</td><td>16</td><td>Yes</td></tr><tr><td>Val.</td><td>2201</td><td>4</td><td>64</td><td>16</td><td>Yes</td></tr><tr><td>Reduced</td><td>20</td><td>4</td><td>256</td><td>64</td><td>Yes</td></tr><tr><td>Full</td><td>20</td><td>4</td><td>512</td><td>128</td><td>No</td></tr><tr><td rowspan="3">Pavia</td><td>Train</td><td>1512</td><td>102</td><td>64</td><td>16</td><td>Yes</td></tr><tr><td>Val.</td><td>168</td><td>102</td><td>64</td><td>16</td><td>Yes</td></tr><tr><td>Full</td><td>2</td><td>102</td><td>400</td><td>100</td><td>Yes</td></tr></table>

![](images/af93ef6867b441a2effd486c707792f4f2888d9dfd8e0cb092f1ff8eaed9576e.jpg)  
Fig. 9. WV3 reduced-resolution JPEG2000 comparison. Baselines: 21.26×; TOP: 33×.

## S5. QUALITATIVE COMPARISONS WITH ALL BASELINESUNDER JPEG2000

For clarity, Fig. 6 of the main manuscript shows only the best-performing baselines of each dataset. Figs. 9, 10, and 11 provide the comparisons with all baselines under the JPEG2000 protocol for the same WV3, GF2, and Pavia scenes.

## S6. COMPARISONS UNDER THE HEVC PROTOCOL

The main manuscript reports the reduced-resolution WV3 and GF2 HEVC results in Tabs. V and VI. Tab. XII gives the Pavia results, while Figs. 12–14 provide the corresponding visual comparisons. The pixel-domain baselines operate at 6.69×, 8.94×, and 7.79× on WV3, GF2, and Pavia, whereas TOP uses 33×, 50×, and 23.97×, respectively. Metric anchors follow Sec. S4; across all three datasets, TOP provides the strongest quantitative results and preserves clearer structures and more stable spectral appearance.

## S7. FULL-RESOLUTION VALIDATION ON WV3 AND GF2

Full-resolution HR-MSI references are unavailable, so WV3/GF2 evaluation uses HQNR, $D _ { \lambda }$ , and $D _ { s }$ against the original lossless PAN and LR-MSI. Pixel-domain baselines receive codec-decoded observations, TOP transmits its fused binary latent, and ZS-PAN follows its per-sample full-resolution protocol. Across both codecs and datasets, TOP attains the highest HQNR and lowest $D _ { \lambda }$ at a higher ratio, and also the lowest $D _ { s }$ under HEVC. Some JPEG2000 baselines have slightly lower $D _ { s } ,$ , but lower HQNR and weaker spectral consistency. Figs. 15–18 show the corresponding qualitative trend.

![](images/86da3e260505bdddbe25d015af6e24fac02abde82ac3795f03e64dd4824647b0.jpg)

Fig. 10. GF2 reduced-resolution JPEG2000 comparison. Baselines: 20.33×; TOP: 50×.  
![](images/02dfff6cfdc58dee498cc648f3c6a8f5899d2634719b836d55837341dca8e835.jpg)  
Fig. 11. Pavia Full JPEG2000 comparison. Baselines: 19.62×; TOP: 23.97×.

![](images/1d419b20373f9aa4b5b3b3ea169fef848d0a5e3e5c19e8853b1e1c95bec03de0.jpg)  
Fig. 12. WV3 reduced-resolution HEVC comparison. Baselines: 6.69×; TOP: 33×.

TABLE XII  
PAVIA FULL HEVC RESULTS AVERAGED OVER TWO SCENES.
<table><tr><td>Method</td><td>Ratio ↑</td><td>PSNR ↑</td><td>SSIM ↑</td><td>ERGAS↓</td><td>HQNR ↑</td><td> $D _ { \lambda } \downarrow$ </td><td> $D _ { s } \downarrow$ </td><td>SAM (deg) ↓</td></tr><tr><td>AWLP+HEVC</td><td rowspan="10"></td><td>24.47</td><td>0.57</td><td>10.79</td><td>0.67</td><td>0.16</td><td>0.20</td><td>13.36</td></tr><tr><td>BDSD-PC+HEVC</td><td>24.28</td><td>0.49</td><td>11.05</td><td>0.64</td><td>0.16</td><td>0.24</td><td>12.32</td></tr><tr><td>BT-H+HEVC</td><td>25.11</td><td>0.67</td><td>10.38</td><td>0.60</td><td>0.33</td><td>0.11</td><td>12.65</td></tr><tr><td>GSA+HEVC</td><td>26.64</td><td>0.64</td><td>8.74</td><td>0.74</td><td>0.17</td><td>0.11</td><td>12.22</td></tr><tr><td>MF+HEVC</td><td>24.77</td><td>0.59</td><td>10.43</td><td>0.70</td><td>0.16</td><td>0.16</td><td>12.67</td></tr><tr><td>PRACS+HEVC</td><td>25.10</td><td>0.56</td><td>10.07</td><td>0.73</td><td>0.17</td><td>0.12</td><td>12.33</td></tr><tr><td>uTDSP+HEVC</td><td>24.50</td><td>0.44</td><td>11.45</td><td>0.58</td><td>0.16</td><td>0.31</td><td>13.88</td></tr><tr><td>DIP-HyperKite+HEVC</td><td>25.23</td><td>0.61</td><td>9.94</td><td>0.77</td><td>0.17</td><td>0.08</td><td>12.61</td></tr><tr><td>HyperTransformer+HEVC</td><td>18.71</td><td>0.17</td><td>21.71</td><td>0.40</td><td>0.19</td><td>0.50</td><td>35.16</td></tr><tr><td>R-PNN+HEVC</td><td>24.67 24.56</td><td>0.57 0.64</td><td>10.59 11.95</td><td>0.71</td><td>0.16</td><td>0.16</td><td>12.87</td></tr><tr><td>DM-ZS+HEVC</td><td>23.86</td><td>0.50</td><td>11.63</td><td>0.70 0.62</td><td>0.23 0.16</td><td>0.09 0.27</td><td>13.60 13.36</td></tr><tr><td>ρ-PNN+HEVC</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TOP (ours)</td><td>23.97×</td><td>29.72</td><td>0.83</td><td>5.91</td><td>0.85</td><td>0.14</td><td>0.02</td><td>6.39</td></tr></table>

![](images/4f198291697f1bc1a7c813ba47610e73ad499d177ec8854c6c49a2c218a7c44e.jpg)

Fig. 13. GF2 reduced-resolution HEVC comparison. Baselines: 8.94×; TOP: 50×.  
![](images/ed2dd9c2d37322702883339973f52f5b79b855907aff5b8d1156ace17bfc21c6.jpg)  
Fig. 14. Pavia Full HEVC comparison. Baselines: 7.79× (displayed as 8×); TOP: 23.97× (displayed as 24×).

## S8. NUMERICAL RESULTS OF THE ABLATION STUDIES

All ablations use WV3 reduced-resolution data. Table XVI reports (a) backbone, (b)–(c) BSQ payload/quality/complexity, (d) direct packing versus entropy model+rANS on the same latent, and (e) stage-3 robustness. The continuous-latent model is fully retrained with a larger fusion module; its entries use 32-bit latents. Both PSNR columns in (c) use the same 20 test scenes, unlike the validation value in Tab. III. In (e), ∆PSNR is no-flip minus perturbed PSNR and ∆SAM is perturbed minus no-flip SAM; lower is better.

TABLE XIII  
WV3 FULL-RESOLUTION RESULTS UNDER BOTH CODECS.
<table><tr><td>Method Ratio ↑1 HQNR ↑</td></tr><tr><td>Dλ ↓  $D _ { s } \downarrow$  JPEG2000</td></tr><tr><td>BDSD+JPEG2000 0.7925 0.1636 0.0524</td></tr><tr><td>FusionNet+JPEG2000 0.7336 0.1839 0.1011</td></tr><tr><td>GLP+JPEG2000 0.7342 0.2043 0.0769</td></tr><tr><td>GS+JPEG2000 0.7983 0.1756 0.0322</td></tr><tr><td>MF+JPEG2000 0.7508 0.1938 0.0686</td></tr><tr><td>28.43× U2Net+JPEG2000 0.7143 0.1880 0.1204</td></tr><tr><td>WFANet+JPEG2000 0.6447 0.2216 0.1723</td></tr><tr><td>ADWM+JPEG2000 0.7386 0.1842 0.0947</td></tr><tr><td>ARConv+JPEG2000 0.6718 0.2158 0.1436</td></tr><tr><td>ZS-PAN+JPEG2000 0.7082 0.2049 0.1101</td></tr><tr><td></td></tr><tr><td>TOP (ours, 33×) 33× 0.8724 0.0984 0.0329</td></tr><tr><td>TOP (ours, 66×) 66× 0.8481 0.1184 0.0389</td></tr></table>

<table><tr><td rowspan="7">BDSD+HEVC FusionNet+HEVC GLP+HEVC GS+HEVC</td><td rowspan="7"></td><td>0.3821</td><td>0.4182</td><td>0.3502</td></tr><tr><td>0.3937 0.3791</td><td>0.4198</td><td>0.3236</td></tr><tr><td>0.5420</td><td>0.4466 0.3300</td><td>0.3175</td></tr><tr><td>MF+HEVC U2Net+HEVC</td><td>0.4286</td><td>0.1939</td></tr><tr><td>0.4007 23.64×</td><td></td><td>0.3033</td></tr><tr><td>0.4108 0.3340</td><td>0.4066 0.4637</td><td>0.3111</td></tr><tr><td>0.4383</td><td>0.3835</td><td>0.3826 0.2900</td></tr><tr><td>ADWM+HEVC ARConv+HEVC</td><td>0.3550</td><td>0.4510</td><td>0.3583</td></tr><tr><td>ZS-PAN+HEVC</td><td></td><td>0.3795 0.4382</td><td>0.3309</td></tr><tr><td>TOP (ours, 33×)</td><td>33×</td><td>0.8724 0.0984</td><td>0.0329</td></tr><tr><td>TOP (ours, 66 ×)</td><td>66× 0.8481</td><td>0.1184</td><td>0.0389</td></tr><tr><td></td><td></td><td></td><td></td></tr></table>

![](images/cf7c14790e7489b9a51caed5ea303e913b6a90ad42bb5cf0bf2d258e7fe3b3a0.jpg)  
Fig. 15. WV3 full-resolution JPEG2000 comparison. Baselines: 28.43×; TOP: 33× and 66×.

TABLE XIV  
GF2 FULL-RESOLUTION RESULTS UNDER BOTH CODECS.
<table><tr><td colspan="4">Method Ratio ↑ HQNR ↑  $D _ { \lambda }$  ↓  $D _ { s } \downarrow$ </td></tr><tr><td colspan="4">JPEG2000</td></tr><tr><td>BDSD+JPEG2000</td><td>0.8278</td><td>0.1366</td><td>0.0416</td></tr><tr><td>FusionNet+JPEG2000</td><td>0.8304 0.7959</td><td>0.1366</td><td>0.0382</td></tr><tr><td>GLP+JPEG2000</td><td></td><td>0.1726</td><td>0.0383</td></tr><tr><td>GS+JPEG2000</td><td>0.7112</td><td>0.1787</td><td>0.1344</td></tr><tr><td>MF+JPEG2000 27.46×</td><td>0.7977</td><td>0.1689</td><td>0.0406</td></tr><tr><td>U2Net+JPEG2000</td><td>0.7999</td><td>0.1301</td><td>0.0800</td></tr><tr><td>WFANet+JPEG2000</td><td>0.8150</td><td>0.1535</td><td>0.0374</td></tr><tr><td>ADWM+JPEG2000</td><td>0.8311</td><td>0.1382</td><td>0.0355</td></tr><tr><td>ARConv+JPEG2000</td><td>0.7298</td><td>0.1846</td><td>0.1048</td></tr><tr><td>ZS-PAN+JPEG2000</td><td>0.8094</td><td>0.1597</td><td>0.0367</td></tr><tr><td>TOP (ours) 50×</td><td>0.8511</td><td>0.1031</td><td>0.0521</td></tr><tr><td colspan="4">HEVC</td></tr><tr><td>BDSD+HEVC FusionNet+HEVC</td><td>0.5165 0.5704</td><td>0.3248</td><td>0.2365</td></tr><tr><td></td><td></td><td>0.3048</td><td>0.1801</td></tr><tr><td>GLP+HEVC</td><td>0.4497</td><td>0.3841</td><td>0.2709</td></tr><tr><td>GS+HEVC</td><td>0.6606</td><td>0.2905</td><td>0.0694</td></tr><tr><td>MF+HEVC</td><td>0.4947</td><td>0.3616</td><td>0.2258</td></tr><tr><td>U2Net+HEVC</td><td>31.69× 0.6819</td><td>0.2453</td><td>0.0971</td></tr><tr><td>WFANet+HEVC</td><td>0.4861</td><td>0.3494</td><td>0.2536</td></tr><tr><td>ADWM+HEVC</td><td>0.5499</td><td>0.3164</td><td>0.1964</td></tr><tr><td>ARConv+HEVC</td><td>0.4152</td><td>0.3978</td><td>0.3116</td></tr><tr><td>ZS-PAN+HEVC</td><td>0.4748</td><td>0.3592</td><td>0.2600</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>TOP (ours)</td><td>50× 0.8511</td><td>0.1031</td><td>0.0521</td></tr></table>

![](images/6b49b6fa42842be7680d267d60dc3255df350746e74f3ddadc6be594161a6d02.jpg)  
Fig. 16. WV3 full-resolution HEVC comparison. Baselines: 23.64×; TOP: 33× and 66×.

![](images/fd954e48b97d104d216ce3803e674518a880fb723da33617b90d0063ebd27967.jpg)  
Fig. 17. GF2 full-resolution JPEG2000 comparison. Baselines: 27.46×; TOP: 50×.

Panel (a) isolates the backbone choice. Swin improves reconstruction PSNR by 1.50 dB over CNN and 1.87 dB over ViT, while its 171.72 GFLOPs remain close to CNN and below ViT; the SSIM increase from 0.75 to 0.83 confirms better structural preservation. It therefore provides the strongest representation without using the most expensive candidate.

![](images/dcbce7f96df83a8e8227f526472e9c4082c082377a2e5280867ae1dd99fb9983.jpg)  
Fig. 18. GF2 full-resolution HEVC comparison. Baselines: 31.69×; TOP: 50×.

Panels (b)–(c) clarify the BSQ trade-off. Removing BSQ raises reconstruction and fusion PSNR, but changes each latent entry from one to 32 bits and requires about 31.7× the fusion parameters and 31.9× the GFLOPs. Its reconstruction–fusion gap also grows from 1.27 to 2.59 dB. BSQ therefore sacrifices some reconstruction ceiling for a target that is much smaller and easier for the onboard module to predict.

Panels (d)–(e) address the remaining communication risks. Entropy modeling saves only 3.58% bppb while adding 10.33M parameters and 74.77 GFLOPs per sample, supporting direct packing. Stage-3 training reduces the PSNR loss under 3% and 5% flips by 76.5% and 68.9%, and leaves SAM nearly unchanged. The small negative ∆SAM values reflect sample-level variation rather than a benefit from corruption; the meaningful result is the absence of systematic spectral degradation. The three components consequently have distinct roles: BSQ controls payload and prediction complexity, direct packing avoids disproportionate sender overhead, and decoder refinement limits residual bit errors.

## S9. COMPARISON WITH ONBOARD PANSHARPENING FOLLOWED BY DOWNLINK COMPRESSION

This comparison tests whether moving conventional pansharpening on board is sufficient without redesigning the transmitted representation. Each baseline fuses the uncompressed PAN and LR-MSI into a full-resolution WV3 MSI and then applies JPEG2000 or HEVC; TOP predicts and bit-packs one fused latent. All methods use the same 20 scenes and ratio definition, so their payloads are measured against the same native observations.

Table XV reports every quality metric and measured fusion time. Against JPEG2000, TOP improves the strongest baseline PSNR by 1.08 dB at 1.65× its compression ratio, while raising the best HQNR from 0.6356 to 0.7552. Against HEVC, it improves the strongest PSNR by 1.66 dB at about 4.98× the ratio. Its lower $D _ { \lambda }$ and $D _ { s }$ under both codecs show that the rate gain does not trade spectral consistency for spatial sharpness.

The payload difference follows from where compression occurs. After pixel-domain fusion, a baseline must represent all CHW samples of the full-resolution MSI, and its final size depends on how strongly a generic codec can quantize that image. TOP instead transmits $C _ { \mathrm { l a t } } H W / s ^ { 2 }$ binary decisions, whose size is fixed before observing image content. For and Pavia setting in Tab. X therefore uses its own tokenizer and fusion module. A new sensor, or a ratio requiring a different latent configuration, requires one-time ground-side retraining without additional onboard computation. A sensor-agnostic tokenizer remains open.

TABLE XV  
ONBOARD PANSHARPENING FOLLOWED BY JPEG2000/HEVC ON WV3. FULL-REFERENCE METRICS USE GT; HQNR TERMS USE THE ORIGINAL LR-MSI/PAN. DASHES DENOTE OMITTED CPU-ONLY FUSION TIMES.
<table><tr><td>Method</td><td>Ratio ↑</td><td>Fusion time (ms/img) ↓</td><td>HQNR ↑</td><td>Dλ↓</td><td>Ds↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>ERGAS↓</td><td>SAM (deg) ↓</td></tr><tr><td colspan="10">JPEG2000 Codec</td></tr><tr><td colspan="10">185.624 0.5847 ± 0.0989 0.2622 ± 0.1046</td></tr><tr><td>WFANet U2Net</td><td>20.00× 20.03×</td><td>114.427</td><td>0.5830 ± 0.0999</td><td>0.2658 ± 0.1057</td><td>0.2103 ± 0.0365 0.2088 ± 0.0367</td><td>29.2451 ± 3.9892 29.2254 ± 3.9889</td><td>0.6964 ± 0.1230 0.6960 ± 0.1230</td><td>7.1451 ± 1.4832 7.1554 ± 1.4812</td><td>8.0026 ± 1.8765 8.0102 ± 1.8692</td></tr><tr><td>FusionNet</td><td>19.99×</td><td>1.444</td><td>0.5824 ± 0.0994</td><td> $0 . 2 6 9 1 \pm 0 . 1 0 6 4$ </td><td>0.2057 ± 0.0367</td><td>29.2050 ± 3.9294</td><td> $0 . 6 9 6 1 \pm 0 . 1 2 2 3$ </td><td>7.1534 ± 1.4757</td><td>7.9752 ± 1.9140</td></tr><tr><td>ADWM</td><td>19.97×</td><td>47.665</td><td>0.5794 ± 0.0982</td><td> $0 . 2 6 7 6 \pm 0 . 1 0 3 5$ </td><td>0.2116 ± 0.0398</td><td>29.1645 ± 3.9383</td><td>0.6947 ± 0.1238</td><td>7.1916 ± 1.4841</td><td>8.1475 ± 1.8269</td></tr><tr><td>ARConv</td><td>20.01×</td><td>80.710</td><td>0.5879 ± 0.1001</td><td> $0 . 2 6 5 5 \pm 0 . 1 0 6 5$ </td><td>0.2024 ± 0.0360</td><td>29.1799 ± 3.9359</td><td> $0 . 6 9 5 2 \pm 0 . 1 2 3 1$ </td><td>7.1642 ± 1.4738</td><td>7.8361 ± 1.8401</td></tr><tr><td>GS</td><td>20.05×</td><td></td><td>0.5464 ± 0.0867</td><td> $0 . 2 9 9 0 \stackrel {  } { \pm } 0 . 1 0 6 4$ </td><td>0.2197 ± 0.0475</td><td>28.4811 3.7461</td><td> $0 . 6 6 5 8 \stackrel { \_ } { \pm } 0 . 1 3 0 6$ </td><td>7.7017 7 ± 1.5986</td><td>7.6836 ± 1.9582</td></tr><tr><td>BDSD</td><td>19.98×</td><td></td><td>0.5857 ± 0.0975</td><td> $0 . 2 6 7 0 \overset { - } { \pm } 0 . 1 1 3 0$ </td><td>0.2020 ± 0.0308</td><td>28.6829 ± 3.7466</td><td> $0 . 6 8 3 0 \pm 0 . 1 1 9 2$ </td><td>7.5294 ± 1.6008</td><td>7.9723 ± 1.8917</td></tr><tr><td>GLP</td><td>19.97×</td><td></td><td> $0 . 6 2 1 2 \pm 0 . 0 9 3 9$ </td><td> $0 . 2 4 8 3 \pm 0 . 1 0 1 2$ </td><td>0.1747 ± 0.0399</td><td>28.9107 ± 3.7803</td><td>0.6869 ± 0.1196</td><td> $7 . 3 4 7 6 \pm 1 . 5 4 0 6$ </td><td>7.6171 ± 1.8623</td></tr><tr><td>MF</td><td>20.02×</td><td></td><td>0.6356 ± 0.0922</td><td>0.2277 ± 0.0945</td><td>0.1786 ± 0.0435</td><td>28.7851 X 3.7437</td><td>0.6848 ± 0.1226</td><td>7.4207 ± 1.5290</td><td>7.6305 ± 1.8427</td></tr><tr><td>ZS-PAN</td><td>20.07×</td><td>36863.683</td><td>0.5629 ± 0.1090</td><td>0.3016 ± 0.1267</td><td>0.1942 ± 0.0393</td><td>28.5151 ± 3.5605</td><td>0.6838 ± 0.1216</td><td>7.4629 ± 1.4911</td><td>8.0665 ± 1.8173</td></tr><tr><td>TOP</td><td>33×</td><td>32.511</td><td>0.7552 ± 0.1422</td><td>0.1863 ± 0.1145</td><td>0.0789 ± 0.0564</td><td>30.3237 ± 2.8339</td><td>0.8193 ± 0.0387</td><td>6.0036 ± 1.2298</td><td>5.9989 ± 1.3813</td></tr><tr><td colspan="10">HEVC Codec</td></tr><tr><td></td><td></td><td>185.624</td><td>0.5355 ± 0.1369</td><td>0.3004 ± 0.1321</td><td>0.2437 ± 0.0586</td><td>28.6397 ± 3.4463</td><td></td><td>7.3943 ± 1.4707</td><td></td></tr><tr><td>WFANet U2Net</td><td>6.65× 6.68×</td><td>114.427</td><td>0.5267 ± 0.1348</td><td>0.3073 ± 0.1322</td><td>0.2484 ± 0.0575</td><td>28.5693 ± 3.4832</td><td>0.6983 ± 0.0965</td><td>7.4309 ± 1.4827</td><td>8.4511 ± 1.8924 ± 1.9079</td></tr><tr><td>FusionNet</td><td>6.66×</td><td>1.444</td><td>0.5281 ± 0.1415</td><td>0.3115 ± 0.1406</td><td>0.2423 ± 0.0577</td><td>28.5932 ± 3.4080</td><td>0.6952 ± 0.0992 0.6971 ± 0.0965</td><td>7.4091 ± 1.4595</td><td>8.5067 8.4481 ± 1.8467</td></tr><tr><td>ADWM</td><td>6.68×</td><td>47.665</td><td>0.5232 ± 0.1290</td><td>0.3083 ± 0.1269</td><td>0.2515 ± 0.0584</td><td>28.5146 ± 3.4834</td><td>0.6923 ± 0.1028</td><td>7.4620 ± 1.4281</td><td>8.6357 ± 1.8846</td></tr><tr><td>ARConv</td><td>6.63×</td><td>80.710</td><td>0.5424 ± 0.1423</td><td>0.3012 ± 0.1366</td><td>0.2332 ± 0.0596</td><td>28.6597 ± 3.3866</td><td>0.6995 ± 0.0964</td><td> $\underline { { 7 . 3 6 8 2 } } \pm 1 . 4 5 3 9$ </td><td>8.2661 ± 1.7866</td></tr><tr><td>GS</td><td>6.65×</td><td></td><td>0.5099 ± 0.1286</td><td>0.3311 ± 0.1399</td><td>0.2440 ± 0.0530</td><td>28.1831 ± 3.4005</td><td>0.6690 ± 0.1047</td><td>7.8659 ± 1.6736</td><td>8.0256 ± 1.9401</td></tr><tr><td>BDSD</td><td>6.60×</td><td></td><td>0.5598 ± 0.1453</td><td>0.2814 ± 0.1393</td><td>0.2311 ± 0.0611</td><td>28.4264 X 3.3264</td><td>0.6910 ± 0.0900</td><td>7.6359 ± 1.6836</td><td>8.3336 ± 1.8504</td></tr><tr><td>GLP</td><td>6.60× 6.62×</td><td></td><td>0.5822 ± 0.1529 ± 0.1475</td><td>0.2744 ± 0.1399</td><td>0.2085 ± 0.0643</td><td>28.6502 ± 3.3278</td><td>0.6950 ± 0.0904</td><td> $7 . 4 5 3 6 \pm 1 . 6 0 2 6$ </td><td>8.0500 ± 1.8544</td></tr><tr><td>MF ZS-PAN</td><td>6.66×</td><td></td><td>0.5861 ± 0.1424</td><td> $\underline { { 0 . 2 6 3 5 } } \pm 0 . 1 3 3 9$ </td><td>0.2146 ± 0.0667</td><td>28.4843 ± 3.3915 28.2744 ± 3.2000</td><td>0.6895 ± 0.0984</td><td>7.5802 ± 1.6303</td><td>8.1932 ± 1.8319</td></tr><tr><td>TOP</td><td>33×</td><td>36863.683</td><td>0.5327 0.7552 ± 0.1422</td><td> $\overline { { 0 . 3 1 3 7 } } \pm 0 . 1 3 8 4$ </td><td>0.2329 ± 0.0634</td><td></td><td>0.6888 ± 0.0956</td><td> $7 . 6 0 5 7 \pm 1 . 5 6 3 0$ </td><td>8.5449 ± 1.8477</td></tr></table>

TABLE XVI  
NUMERICAL ABLATIONS ON WV3 REDUCED-RESOLUTION DATA.
<table><tr><td colspan="2">(a) Architecture reconstruction Architecture Params (M) ↓</td><td>GFLOPs ↓</td><td>PSNR ± Std. ↑</td><td></td><td> $\mathrm { S S I M } \pm \mathrm { S t d . } \uparrow$ </td></tr><tr><td colspan="2">CNN</td><td>66.721</td><td>169.96</td><td> $2 7 . 4 2 \pm 2 . 0 2$ </td><td> $\underline { { 0 . 7 5 } } \pm 0 . 0 6$ </td></tr><tr><td>ViT</td><td></td><td>62.709</td><td>219.46</td><td> $\overline { { 2 7 . 0 5 } } \pm 1 . \dot { 8 } 2$ </td><td> $\overline { { 0 . 7 5 } } \pm 0 . 0 6$ </td></tr><tr><td>Swin</td><td></td><td>65.609</td><td>171.72</td><td> $\mathbf { 2 8 . 9 2 \bar { \pm } 2 . 0 7 }$ </td><td> $\pm { \bf 0 . 8 3 } \pm { \bf 0 . 0 4 }$ </td></tr><tr><td colspan="6">(b) BSQ payload</td></tr><tr><td>Method</td><td></td><td></td><td>bppb ↓ Payload/latent entry ↓</td><td></td><td></td></tr><tr><td>TOBT TÓBT w/o BSQ</td><td></td><td>0.0625 2.000</td><td>1 bit 32 bits</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="6">(c) BSQ reconstruction, fusion, and latent-module complexity</td></tr><tr><td>Method</td><td></td><td>Recon. PSNR ↑</td><td>Fusion PSNR ↑</td><td>Params (M) ↓</td><td>GFLOPs↓</td></tr><tr><td>TOBT</td><td></td><td>31.59</td><td>30.32</td><td>0.61</td><td>0.31</td></tr><tr><td></td><td>TOBT w/o BSQ</td><td>34.70</td><td>32.11</td><td>19.33</td><td>9.88</td></tr><tr><td colspan="6">(d) Direct packing versuș entropy coding Method Add. params ↓ Add. GFLOPs/sample ↓</td></tr><tr><td>TOP</td><td></td><td>bppb ↓</td><td></td><td></td><td></td></tr><tr><td>TOPEM+rANS</td><td></td><td>0.062500 0.060260</td><td>0.00M 10.329096M</td><td>0.00 74.769797</td><td></td></tr><tr><td colspan="6">(e) Bit-flip robustness on 20 scenes</td></tr><tr><td>Stage</td><td></td><td>Flip 3%</td><td>∆PSNR (dB) ↓</td><td>∆SAM (deg) ↓</td><td></td></tr><tr><td>Stage 2 Stage 3</td><td></td><td></td><td>3.24 0.76</td><td>0.36 -0.11</td><td></td></tr><tr><td>Stage 2</td><td></td><td>3% 5%</td><td>5.17</td><td>0.64</td><td></td></tr><tr><td>Stage 3</td><td></td><td>5%</td><td>1.61</td><td>-0.07</td><td></td></tr></table>

b) Multi-Stage Training: TOBT uses 40,000 joint encoder–decoder iterations, two 20,000-iteration decoderrefinement stages (the last with bit flips), and separate fusion training (Tab. X). These stages learn the latent space, adapt decoding to the transmitted HR code, and improve bit-error robustness, as supported by Sec. V-D. The added training and tuning effort motivates a simpler future procedure.

WV3 at 33×, this is 0.5 bit per spatial pixel, or 0.0625 bppb across eight bands. The comparison therefore evaluates two deployment representations, rather than only two fusion networks.

TOP has the same row in both codec blocks because it uses neither codec. The changing baseline rows show that onboard pixel-domain fusion still exposes the final product to a separately optimized lossy stage. The timing column measures fusion inference rather than complete encode–transmit–decode latency; among the reported neural baselines, only FusionNet is faster than TOP, while traditional CPU timings are omitted. Thus, the table supports the rate–quality advantage and moderate onboard fusion cost, but does not claim hardwareindependent end-to-end latency or energy.

## S10. LIMITATIONS

a) Sensor-Specific Training: The band-adaptive embedding accepts different band counts, but dimensions and learned parameters remain configuration-specific. Each WV3, GF2,