# Unified RAW Image Exposure Correction and Restoration via Linear Illumination and Frequency-Adaptive Enhancement

1<sup>st</sup> Yingkai Zhang

2<sup>nd</sup> Yunhao Zou

3<sup>rd</sup> Ying Fu<sup>\*</sup>

School of Computer Science

School of Computer Science

School of Computer Science

and Technology

and Technology

and Technology

Beijing Institute of Technology

Beijing Institute of Technology

Beijing, China

Beijing Institute of Technology

Beijing, China

zhangyingkai@bit.edu.cn

yunhaozou@outlook.com

Beijing, China

fuying@bit.edu.cn

Abstract—Incorrect exposure degrades both image brightness and recoverable scene detail. Existing sRGB-based correction methods are limited by the non-linear image signal processing pipeline, whereas RAW-based methods are usually developed for a single degradation type, such as low-light denoising or high-light recovery. In this paper, we focus on joint exposure correction and restoration directly in the RAW domain and propose a unified RAW Image Exposure Correction and Restoration (RIECR) framework for both under- and over-exposed images. RIECR decomposes the problem into two coupled subproblems that follow the RAW imaging model: linear illumination correction and residual degradation restoration. First, a Linear Illumination Correction Network (LICN) estimates exposureaware linear gains by using channel-specific RAW cues. It exploits the high signal-to-noise ratio of Green channels for robust brightening and the relatively unsaturated Red/Blue responses for high-light darkening. Second, a dual-branch Frequency-based Enhancement Network (FEN) restores the remaining degradation with a Frequency Dynamic Convolution (FDC) module. FDC decomposes convolutional responses into low- and highfrequency components and adaptively suppresses amplified noise or recovers clipped textures according to the exposure context. We further build a real paired RAW benchmark containing 223 scenes and 1,784 pairs covering complex multi-exposure errors. Experiments against sixteen state-of-the-art methods show that RIECR achieves state-of-the-art performance on real RAW exposure correction.

Index Terms—RAW image restoration, exposure correction, linear illumination adjustment, frequency enhancement, RAW image dataset

## I. INTRODUCTION

Exposure correction improves visibility and preserves scene detail for both photography and downstream tasks such as segmentation and detection [1]–[5]. Yet exposure errors are not a single degradation [6]. Limited photons in under-exposed scenes produce photon shot noise, sensor noise [7], and color distortion [8]; limited sensor range in over-exposed scenes saturates and irreversibly clips bright structures [9]. A practical model must therefore restore exposure-dependent content as well as brightness.

Learning-based sRGB correction has progressed rapidly [10]–[17]. However, sRGB is produced by a nonlinear ISP involving demosaicking, white balance, tone mapping, and quantization [18]; amplified noise or clipped highlights are then difficult to recover. RAW data, as shown in Fig. 1, retains higher bit depth and a near-linear radiance response [9], [19], [20], making it a better domain for physical exposure correction. Nevertheless, RAW methods remain task-specific, separating low-light denoising [18], [21] from high-light recovery [9]. A unified model must estimate a valid linear gain and restore the direction-dependent residual within one network.

To this end, we propose RAW Image Exposure Correction and Restoration (RIECR), which explicitly decomposes the task into two coupled sub-problems. First, the Linear Illumination Correction Network (LICN) predicts the exposure direction and estimates linear gains from complementary RAW channels: Green offers robust high-SNR cues for low-light brightening, while Red/Blue often remain less saturated and guide high-light darkening. Second, the Frequency-based Enhancement Network (FEN) restores the remaining degradation through local convolution and global attention. Its Frequency Dynamic Convolution (FDC) decomposes kernel responses and adaptively suppresses amplified high-frequency noise or emphasizes missing texture according to the exposure context.

Because existing RAW datasets usually cover only one exposure subtask [9], [18], [22], we collect Complex Multi-Exposure RAW (CMER), with 223 real scenes and 1,784 pairs spanning both directions, and synthesize additional pretraining pairs from MIT-Adobe FiveK [23]. CMER supplies pixel-aligned supervision for unified evaluation, while the synthetic data tests whether broader exposure coverage benefits real-data fine-tuning. Extensive experiments show consistent improvements under both training protocols and exposure directions. In summary, the main contributions are summarized as follows:

![](images/3fa82b425658d348f8667f47b9a4daa72cd03a6bf7ef2a662e5bad1b4b5937d6.jpg)

![](images/430f9f68164ad0275816e6935175448fc971b204657f3b4276ac9e1fa34117bb.jpg)  
Reference (High)

![](images/484edf4dbec4c369b45bee50cad42dee1a817538d5ba4cfe30876c1b3778a4c5.jpg)  
RAW image

![](images/a197e4db895cd6df6ae1c6ff77dd9184c7210a456a243928d0322678efb73da8.jpg)  
Camera output

![](images/bdd5c5a91363ce70605acabafa91a04736ccd6f5dafa6c3b14f26fc879c95de8.jpg)  
Reference (Low)

![](images/d0f3b15158520533f86b128ff8450cbab17b96d21ac61b35f5b4a7a6ad068ac6.jpg)  
RAW image

![](images/e8c6e0a815dda3795d2a8c90df88d61c33694778ecc7b55a382b7e66bb787499.jpg)  
Camera output  
Fig. 1. The RAW/sRGB images of indoor/outdoor scenes under different lighting conditions. Due to the limited dynamic range of the camera, the 8-bit sRGB output loses significant information from the scene under high-light conditions. The stamens of the white flowers are almost invisible in the sRGB image but are visible in the 14-bit RAW image. Due to the low signal-to-noise (SNR) and low signal intensity under low-light conditions, the sRGB image loses considerable detail. The tail of Pikachu and the edge of the box are essentially unrecognizable in the sRGB image, whereas they are discernible in the RAW image (Zoom in for better details).

• We establish a unified RAW-domain formulation for simultaneous exposure correction and restoration. By addressing divergent degradations, our framework provides a holistic solution to the RAW-domain multi-exposure errors within a single model.

• We propose two physically motivated components: a linear illumination correction network for exposure-aware linear gain estimation from channel-specific RAW cues, and a frequency-based enhancement network with dynamic convolution for adaptive restoration of divergent frequency degradations.

• We construct CMER, a paired real RAW benchmark with 223 scenes and 1,784 multi-exposure pairs for unified RAW-based exposure correction and restoration tasks under diverse lighting conditions.

## II. RELATED WORK

Single Image High Dynamic Range Reconstruction: SI-HDR expands local dynamic range through inverse tone mapping [24], direct CNN/Transform reconstruction [25]– [27], exposure-stack synthesis [28], or inverse-camera modeling [29], [30]. RawHDR further exploits high-bit-depth RAW measurements [31]. These methods recover localized dynamicrange loss from a nominally exposed input and usually assume a usable global tone, whereas exposure correction must infer the global rescaling.

Low-Light Image Enhancement: Classical enhancement adjusts histograms, suppresses noise, or decomposes illumination and reflectance [32]–[34]. Learning-based methods employ illumination decomposition, color spaces, frequency embedding, transformers, and Mamba architectures [35]–[43]. RAW restoration additionally models sensor noise for synthesis and removal [7], [21], [44]–[46]. These methods are effective for photon-limited inputs but typically assume a known exposure ratio and do not address clipping-dominated over-exposure.

Image Exposure Correction: Classical exposure correction uses histogram, tone, or Retinex adjustment [32], [47], [48], while high-light-specific networks recover brightness and clipped content [9], [49], [50]. Deep learning has seen rapid development recently [51]–[54], unified sRGB methods learn coarse-to-fine correction, exposure consistency, luminance, color-space, or task-shared mappings [10]–[17], [55]. Guided reconstruction can exploit frequency cues or crossmodal structure [56]–[62], but does not connect them to RAW exposure formation. Existing RAW methods still separate lowlight restoration [18], [19], [21] and high-light recovery [9]; RIECR instead couples RAW-linear gain estimation with exposure-adaptive residual restoration.

![](images/babe596050bacd670dcc58037f8b44c998bf047285068c7a5ddfe889de942ac7.jpg)

![](images/3c1d3b8687ca2a28e41f106bde2a45ae74d9e618745f76744dbdd71d2cb46dd2.jpg)

(d) Reference HF  
![](images/67b461c2c94574ed0e76cae57ca530f07afe4f6097b8a0843ccb2b9b1f476a5a.jpg)  
(e) High Light HF

![](images/03676d7081ca519befd062ff9b0c95255b03a0ac849f94835717a525241f0b4d.jpg)

![](images/3b70da5a3d0caefabc959934acb686bb94150554485e660734b321580f472bdf.jpg)  
(g) Reference LF

(f) Low Light HF  
![](images/09987c2834acb6627449ef37ab340f998528ea21e3f0c4f755a519cdace14fe7.jpg)  
(h) High Light LF

![](images/07664bfdc5e035a4618f31ad6dafb7b64e480f4034bb9fe4972e1de5e0457c41.jpg)  
(i) Low Light LF  
Fig. 2. Frequency analysis after linear scaling. (a–c) Scaled images, (d–f) high-frequency (HF) components, and (g–i) low-frequency (LF) components. Over-exposure removes edges, whereas under-exposure amplifies noise. Under different lighting conditions, the low-frequency information is similar to that in the reference image, indicating that information such as illumination remains essentially unchanged after linear adjustment.

## III. THE PROPOSED METHOD

## A. Problem Formulation

Let X be the clean normal-exposure RAW image and Y an observed RAW image. The linear sensor response:

$$
\mathbf { Y } = \left\{ \begin{array} { l l } { r \mathbf { X } + \mathbf { N } = r ( \mathbf { X } + \tilde { \mathbf { N } } ) , } & { \mathrm { i f ~ } 0 < r < 1 } \\ { \operatorname* { m i n } ( r \mathbf { X } , 1 ) = r \cdot \operatorname* { m i n } ( \mathbf { X } , \frac { 1 } { r } ) , } & { \mathrm { i f ~ } r \geq 1 } \end{array} \right.\tag{1}
$$

where r is the exposure ratio, N contains photon and sensor noise, and min(·, 1) denotes saturation. Noise and clipping make inverse scaling insufficient.

![](images/7cd3e6c8d940eb5ba4634e3a144300a0397aa8c9ba9913618041aa9ab06e2aa4.jpg)  
Fig. 3. The overview framework of the proposed method (RIECR), including Linear Illumination Correction Network (LICN) to adjust the illumination and Frequency-based Enhancement Network (FEN) to address the divergent degradation.

For low-light inputs $( 0 < r < 1 )$ , inverse scaling amplifies the normalized noise $\tilde { \bf N } \mathrm { : }$

$$
\mathbf { X } = \frac { \mathbf { Y } } { r } - \tilde { \mathbf { N } } .\tag{2}
$$

For high-light inputs $( r \geq 1 )$ , let S denote the structural and textural signal lost by clipping. Restoration becomes:

$$
\begin{array} { l } { \displaystyle \mathbf { Y } = r ( \mathbf { X } - \mathbf { S } ) , } \\ { \displaystyle \mathbf { X } = \frac { \mathbf { Y } } { r } + \mathbf { S } . } \end{array}\tag{3}
$$

Amplified noise and missing structure have different causes but mainly affect high-frequency residuals, while illumination are more stable. Both cases can be written as:

$$
\mathbf { X } = \frac { \mathbf { Y } } { r } + \mathbf { D } ,\tag{4}
$$

where r is the gain and D is $- \tilde { \mathbf { N } }$ for low-light denoising or S for high-light recovery.

## B. Motivation and Framework Architecture

Fig. 2 illustrates the frequency behavior. It motivates two coupled modules. LICN estimates r using channel-dependent RAW responses: Green has higher sensitivity and SNR [31], [63], [64], while Red/Blue often retain highlight responses. FEN restores D by adapting low-/high-frequency responses to amplified noise or missing texture.

The architecture of the proposed RAW Image Exposure Correction and Restoration (RIECR) framework is illustrated in Fig. 3. A normalized Bayer image $\mathbf { Y } \in \mathbb { R } ^ { 2 H \times 2 W }$ is packed as $\mathbf { I } _ { r g b g } \in \mathbb { R } ^ { H \times W \times 4 }$ and mapped by:

$$
\mathbf { I } = f ( \mathbf { I } _ { r g b g } ; \theta ) ,\tag{5}
$$

where $f$ and θ denote RIECR and its parameters. LICN produces a linearly corrected <sup>˜</sup>I; local and global FEN branches then predict the residual and output I.

## C. Linear Illumination Correction Network (LICN)

LICN contains an exposure-label estimator and channelaware gain guidance (Fig. 3(b)).

Illumination Label Estimation. A $1 \times 1$ convolution mixes packed channels, a $5 \times 5$ depth-wise convolution aggregates local illumination, and global pooling with a linear projection:

$$
\mathbf { L } = \mathrm { S i g m o i d } ( \mathrm { L i n e a r } ( \operatorname { A v g . } ( \mathrm { C o n v s . } ( \mathbf { I } _ { r g b g } ) ) ) ) ,\tag{6}
$$

where $\begin{array} { r } { \mathbf { L } \ = \ [ \mathbf { L } _ { h i g h } , \mathbf { L } _ { l o w } ] . } \end{array}$ . The exposure-time ratio $f _ { r a t i o }$ and maximum factors $f _ { h } , f _ { l }$ define severity-aware targets $\hat { \bf L } = [ f _ { r a t i o } / f _ { h } , 0 ]$ for high-light and $[ 0 , f _ { r a t i o } / f _ { l } ]$ for lowlight, supervised by:

$$
\mathcal { L } _ { l a b e l } = - \hat { \mathbf { L } } \cdot \log ( \mathbf { L } ) - ( 1 - \hat { \mathbf { L } } ) \cdot \log ( 1 - \mathbf { L } ) ,\tag{7}
$$

which supervises exposure direction and magnitude.

Channel-Aware Guidance. We split $\mathbf { I } _ { \mathit { r g b g } }$ into Green $\mathbf { I } _ { g }$ and Red/Blue $\mathbf { I } _ { r b }$ . AdaIN [65] normalizes each cue before fusion with the shared feature $\mathbf { F } _ { r g b g }$

$$
\mathbf { F } _ { t } = { \mathrm { C o n v s } } ( { \mathrm { C o n c a t } } [ \mathbf { F } _ { r g b g } , { \mathrm { A d a I N } } ( \mathbf { I } _ { t } ) ] ) ,\tag{8}
$$

where $t \in \{ r b , g \}$ . The label-weighted gains are:

$$
R _ { r b } = \operatorname* { m a x } ( { \mathrm { S i g m o i d } } ( \operatorname { A v g . } ( \mathbf { F } _ { r b } \cdot \mathbf { L } _ { h i g h } ) { \big ) } \cdot f _ { h } , 1 ) ,\tag{9}
$$

$$
R _ { g } = \operatorname* { m a x } ( { \mathrm { S i g m o i d } } ( \operatorname { A v g . } ( \mathbf { F } _ { g } \cdot \mathbf { L } _ { l o w } ) ) \cdot f _ { l } , 1 ) ,\tag{10}
$$

where $R _ { r b }$ and $R _ { g }$ are darkening and brightening ratios. The adjusted image is:

$$
\tilde { \mathbf { I } } = \left\{ \begin{array} { l l } { \mathbf { I } _ { r g b g } / R _ { r b } , } & { \mathrm { i f } ~ \mathbf { L } _ { h i g h } > \mathbf { L } _ { l o w } } \\ { \mathbf { I } _ { r g b g } \cdot R _ { g } , } & { \mathrm { i f } ~ \mathbf { L } _ { h i g h } \le \mathbf { L } _ { l o w } } \end{array} \right.\tag{11}
$$

so the predicted state selects the RAW cue and linear operator before residual restoration.

## D. Frequency-Based Enhancement Network (FEN)

FEN uses local and global branches with FDC (Fig. 3(c)). The local branch exploits spatial self-similarity, while the global branch supplies context for severely degraded regions.

TABLE I  
COMPARISON OF DATASETS FOR EXPOSURE CORRECTION AND RESTORATION TASKS.
<table><tr><td>Task</td><td>Dataset</td><td>Format</td><td> $\mathrm { T y p e }$ </td><td>Size</td><td> $\mathrm { H i g h }$ </td><td>High-light Exposure Value</td><td>Low</td><td>Low-light Exposure Value</td><td>Normal-light Ground-Truth</td></tr><tr><td rowspan="3">Under-Exposure Correction</td><td>LOL [66]</td><td>sRGB</td><td>Real</td><td>789</td><td>××</td><td>N/A</td><td>√</td><td>N/A</td><td>√</td></tr><tr><td>LOL-v2 [66]</td><td>sRGB</td><td>Real</td><td>500</td><td></td><td>N/A</td><td>√</td><td>N/A</td><td>√</td></tr><tr><td>SID [18]</td><td>RAW</td><td>Real</td><td>2,697</td><td>x</td><td>N/A</td><td>√</td><td>Ratio= 1 1 100，250，300</td><td>√</td></tr><tr><td>Over-Exposure Correction</td><td>RPO [9]</td><td>RAW</td><td>Real</td><td>2,600</td><td>√</td><td>Ratio=3, 5, 8, 10</td><td>x</td><td>N/A</td><td>√</td></tr><tr><td>HDR Reconstruction</td><td>SIHDR [22]</td><td>RAW</td><td>Real</td><td>905</td><td>√</td><td>N/A</td><td>√</td><td>N/A</td><td>x</td></tr><tr><td rowspan="3">Multiple-Exposure Correction</td><td>SICE [67]</td><td>sRGB</td><td>Real</td><td>4,413</td><td>√</td><td>EV=0.5,0.7,1,2,3</td><td>√</td><td> $\mathrm { E V } { = } { - } 0 . 5 { , } { - } 0 . 7 { , } { - } 1 { , } { - } 2 { , } { - } 3$ </td><td>√</td></tr><tr><td>ME [10]</td><td>sRGB</td><td>Syn.</td><td>24,330</td><td>√</td><td>EV=1,1.5</td><td>√</td><td>EV=-1,-1.5</td><td>√</td></tr><tr><td>CMER (Ours)</td><td>RAW</td><td>Real</td><td>1,784</td><td>√</td><td>Ratio=2,4,8</td><td>√</td><td>Ratio= 1 1 1 1 1 2,4,8,16,32</td><td>√</td></tr></table>

Frequency Dynamic Convolution. Fixed convolution cannot simultaneously denoise and recover detail across spatially varying degradation [56], [57]. FDC decomposes a base kernel ${ \bf K } _ { s }$ in the spatial domain:

$$
\begin{array} { r } { \mathbf { K } _ { l } = \mathbf { A v g } ( \mathbf { K } _ { s } ) , } \\ { \mathbf { K } _ { h } = \mathbf { K } _ { s } - \mathbf { K } _ { l } , } \end{array}\tag{12}
$$

where $\mathrm { { A v g } ( \cdot ) }$ is $3 \times 3$ pooling. $\mathbf { K } _ { l }$ captures stable structure and ${ \bf K } _ { h }$ captures residual detail. Pixel-wise gates ${ \bf M } _ { l } , { \bf M } _ { h }$

$$
\hat { \mathbf { F } } = \mathbf { M } _ { h } \odot \mathbf { M } \mathbf { C } ( \mathbf { K } _ { h } , \mathbf { F } ) + \mathbf { M } _ { l } \odot \mathbf { M } \mathbf { C } ( \mathbf { K } _ { l } , \mathbf { A } \mathbf { v } \mathbf { g } ( \mathbf { F } ) ) ,\tag{13}
$$

where MC is the primitive convolution and ⊙ denotes multiplication. The gates attenuate noise-exposed high frequencies or strengthen contextual texture near saturated structures.

Local Convolution Branch. A U-shaped CNN uses residual blocks:

$$
\hat { \mathbf { F } } = \{ \mathrm { L R e L U } ( \mathrm { C o n v } _ { 3 \times 3 } ( \mathbf { F } ) ) \} _ { \times 2 } + \mathrm { C o n v } _ { 1 \times 1 } ( \mathbf { F } ) ,\tag{14}
$$

where LReLU is the activation. Down/up-sampling and two skip connections predict a residual added to <sup>˜</sup>I, yielding $\mathbf { I } _ { l o c a l }$ Global Attention Branch. A U-shaped transformer supplies long-range context [36]. Its i-th attention block is:

$$
\begin{array} { r l } & { \hat { { \bf F } } _ { i } = \mathrm { C M C A } ( \mathrm { L N } ( { \bf F } _ { i - 1 } ) ) + { \bf F } _ { i - 1 } , } \\ & { { \bf F } _ { i } = \mathrm { F F N } ( \mathrm { L N } ( \hat { { \bf F } } _ { i } ) ) + \hat { { \bf F } } _ { i } , } \end{array}\tag{15}
$$

where CMCA, LN [68], and FFN denote channel attention, layer normalization, and feed-forward layers. Multi-scale FDC blocks let reliable regions guide distant degradation; convolution fuses $\mathbf { I } _ { l o c a l }$ and $\mathbf { I } _ { g l o b a l }$ into I.

## E. Learning Details

We scale each reference by its exposure ratio to obtain the LICN target $\mathbf { I } _ { l } \colon$

$$
\begin{array} { r } { \mathcal { L } _ { i l l u } = \| \tilde { \mathbf { I } } - \mathbf { I } _ { l } \| _ { 1 } . } \end{array}
$$

The total loss is:

(16)

$$
\mathcal { L } = \mathcal { L } _ { r } + \gamma _ { 1 } \mathcal { L } _ { i l l u } + \gamma _ { 2 } \mathcal { L } _ { l a b e l } ,\tag{17}
$$

where $\mathcal { L } _ { r } = \| \mathbf { I } - \mathbf { X } \| _ { 1 }$ and $\gamma _ { 1 } , \gamma _ { 2 }$ balance the illumination and label terms.

## IV. MULTIPLE EXPOSURE RAW DATASET

Existing RAW datasets target low-light denoising, highlight recovery, or HDR reconstruction, while multi-exposure datasets are mainly sRGB [9], [10], [18], [22], [69]. CMER provides paired RAW data for both directions (Table I).

![](images/23503ecaa61ac0217afc942725c2e222bc1560e7ff539ef8e6f72abed318d82c.jpg)  
(a) Examples of our CMER dataset

![](images/0c890bd58ba4edeb114b0cbea43d362cf81c71c00f9dccb4bfe9c63a734e3897.jpg)

![](images/4758c47bb8e1c5d9b0e1fa7b05b0078aade81e379744d7da5f5e419360928a10.jpg)  
(b) Comparison of input data range (RAW vs. sRGB)  
(c) Comparison of dataset range (Our CMER vs. SIHDR)  
Fig. 4. Overview and analysis of the proposed CMER dataset. (a) Paired Samples: Representative scenes demonstrating rigorous alignment across extreme exposure ratios (from 1/32 to 8) and their normal-light ground truth. (b) Data Integrity: Histogram comparison confirming that the RAW data preserves the original linear response compared to processed sRGB formats [10]. (c) Dataset Diversity: Statistical comparison showing CMER provides broader scene coverage than the existing SIHDR [22].

Complex Multiple Exposure RAW (CMER) Dataset: We capture 223 static indoor/outdoor scenes with a Canon EOS 5D Mark IV and save 14-bit RAW images. The scenes include dark smooth regions, strong light sources, reflective materials, and fine structures. A tripod and remote control fix the viewpoint; after acquiring a well-exposed reference, we fix aperture and vary only exposure time. Captures with motion, illumination variation, or misalignment are discarded, preserving pixel alignment. Daytime images use ISO 100; nighttime inputs additionally use ISO 1,600–6,400 while references remain at ISO 100. We record over-exposure ratios {2,4,8} and under-exposure ratios $\{ 1 / 2 , 1 / 4 , 1 / 8 , 1 / 1 6 , 1 / 3 2 \}$ [70]. CMER contains 1,784 pairs at 6720 × 4480: 669 over-exposed and 1,115 under-exposed. A scene-disjoint split uses 213 scenes (1,704 pairs) for training and 10 scenes (80 pairs) for testing. Data samples and comparative data analysis are shown in Fig. 4.

Synthetic RAW Image Dataset: The synthetic dataset, named SMIT, applies the same eight ratios to 1,500 MIT-Adobe FiveK RAW references [23]. Linear amplification plus clipping simulates over-exposure; attenuation plus sensor noise simulates under-exposure [7], [9], [10], yielding 12,000 pretraining pairs.

RawHDR [31]  
ADU [76]  
SMG [75]  
TABLE II  
RESULTS ON THE DISJOINT CMER TEST SPLIT UNDER TWO TRAINING PROTOCOLS. PARAMETERS AND MACS ARE REPORTED IN M AND G, RESPECTIVELY, WITH MACS CALCULATED ON A 256 × 256 RAW PATCH. BEST AND SECOND-BEST RESTORATION SCORES ARE BOLD AND UNDERLINED.
<table><tr><td rowspan="2">Methods</td><td colspan="6">CMER (train) → CMER (test)</td><td colspan="6">SMIT(pre) → CMER(ft) → CMER(test)</td><td rowspan="2">Params. MACs (M)</td><td rowspan="2">(G)</td></tr><tr><td>High-light</td><td>PSNR↑ SSIM↑ PSNR↑</td><td></td><td>Low-light</td><td>SSIM↑ PSNR↑ SSIM↑</td><td>Average</td><td>High-light PSNR↑ SSIM↑ PSNR↑ SSIM↑ PSNR↑ SSIM↑</td><td></td><td>Low-light</td><td></td><td>Average</td><td></td></tr><tr><td>sRGB-based Model</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MSEC [10][CVPR&#x27;21]</td><td>26.72</td><td>0.875</td><td>25.23</td><td>0.824</td><td>25.79</td><td>0.843</td><td>25.72</td><td>0.857</td><td>26.46</td><td>0.836</td><td>26.18</td><td>0.844</td><td>7.02</td><td>10.57</td></tr><tr><td>ECCM [71][BMVC&#x27;21]</td><td>28.81</td><td>0.913</td><td>31.84</td><td>0.920</td><td>30.07</td><td>0.917</td><td>30.92</td><td>0.943</td><td>31.65</td><td>0.926</td><td>31.38</td><td>0.933</td><td>5.39</td><td>9.55</td></tr><tr><td>ECLNet [12][MM&#x27;22]</td><td>26.76</td><td>0.899</td><td>26.89</td><td>0.876</td><td>26.76</td><td>0.885</td><td>29.08</td><td>0.929</td><td>26.99</td><td>0.890</td><td>27.77</td><td>0.905</td><td>0.02</td><td>1.67</td></tr><tr><td>ENC [72][CVPR&#x27;22]</td><td>30.27</td><td>0.940</td><td>31.30</td><td>0.910</td><td>30.91</td><td>0.921</td><td>30.74</td><td>0.956</td><td>31.38</td><td>0.924</td><td>31.14</td><td>0.936</td><td>0.62</td><td>14.24</td></tr><tr><td>FECNet [11][ECCV&#x27;22]</td><td>30.47</td><td>0.908</td><td>31.35</td><td>0.915</td><td>31.02</td><td>0.912</td><td>30.62</td><td>0.923</td><td>31.84</td><td>0.917</td><td>31.38</td><td>0.919</td><td>0.15</td><td>5.82</td></tr><tr><td>RF [36][ICCV&#x27;23]</td><td>29.94</td><td>0.939</td><td>30.09</td><td>0.906</td><td>30.03</td><td>0.919</td><td>32.69</td><td>0.950</td><td>32.51</td><td>0.924</td><td>32.58</td><td>0.933</td><td>1.61</td><td>20.38</td></tr><tr><td>CSEC [16][CVPR&#x27;24]</td><td>28.92</td><td>0.908</td><td>29.80</td><td>0.850</td><td>29.47</td><td>0.872</td><td>29.41</td><td>0.927</td><td>30.43</td><td>0.864</td><td>30.05</td><td>0.887</td><td>6.06</td><td>30.62</td></tr><tr><td>ES [15][CVPR&#x27;25]</td><td>28.87</td><td>0.853</td><td>28.92</td><td>0.888</td><td>28.91</td><td>0.875</td><td>30.75</td><td>0.923</td><td>30.62</td><td>0.903</td><td>30.67</td><td>0.911</td><td>1.91</td><td>14.34</td></tr><tr><td colspan="9">RAW-based Model</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SID [18][CVPR&#x27;18]</td><td>25.09</td><td>0.805</td><td>25.32</td><td>0.824</td><td>25.23</td><td>0.817</td><td>25.18</td><td>0.818</td><td>25.79</td><td>0.844</td><td>25.65</td><td>0.834</td><td>7.76</td><td>13.73</td></tr><tr><td>RED [73][CVPR&#x27;21]</td><td>25.43</td><td>0.859</td><td>24.37</td><td>0.798</td><td>24.77</td><td>0.806</td><td>26.06</td><td>0.865</td><td>24.61</td><td>0.813</td><td>25.16</td><td>0.832</td><td>0.78</td><td>0.97</td></tr><tr><td>ABF [74][CVPR&#x27;22]</td><td>25.74</td><td>0.819</td><td>30.94</td><td>0.920</td><td>29.00</td><td>0.898</td><td>29.00</td><td>0.910</td><td>30.88</td><td>0.922</td><td>30.18</td><td>0.917</td><td>15.01</td><td>22.56</td></tr><tr><td>CGNet [9][TCSVT&#x27;23]</td><td>25.46</td><td>0.858</td><td>27.80</td><td>0.867</td><td>26.92</td><td>0.863</td><td>27.99</td><td>0.852</td><td>29.65</td><td>0.834</td><td>29.03</td><td>0.841</td><td>17.67</td><td>46.24</td></tr><tr><td>RawHDR [31][ICCV&#x27;23]</td><td>27.03</td><td>0.899</td><td>26.90</td><td>0.879</td><td>26.94</td><td>0.886</td><td>27.81</td><td>0.913</td><td>27.72</td><td>0.890</td><td>27.76</td><td>0.898</td><td>10.49</td><td>50.81</td></tr><tr><td>SMG [75][CVPR&#x27;23]</td><td>29.07</td><td>0.867</td><td>30.07</td><td>0.898</td><td>29.69</td><td>0.886</td><td>28.01</td><td>0.860</td><td>30.82</td><td>0.909</td><td>29.77</td><td>0.891</td><td>15.42</td><td>42.64</td></tr><tr><td>ADU [76][TOMM&#x27;24]</td><td>28.16 29.74</td><td>0.893 0.941</td><td>27.33 30.22</td><td>0.839</td><td>27.64</td><td>0.859</td><td>28.92 30.49</td><td>0.895 0.942</td><td>28.49 32.08</td><td>0.861</td><td>28.65</td><td>0.874</td><td>0.64</td><td>0.71</td></tr><tr><td>RRM [39][TCSVT&#x27;25] RIECR (Ours)</td><td>31.23</td><td>0.950</td><td>33.45</td><td>0.933 0.950</td><td>30.05 32.63</td><td>0.936 0.950</td><td>33.65</td><td>0.965</td><td>33.98</td><td>0.935 0.941</td><td>31.48 33.85</td><td>0.938 0.951</td><td>7.65</td><td>32.06</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>2.56</td><td>31.91</td></tr></table>

![](images/965fa2f5e3173fd69dda727e6d1e63907c67b716e0b3c1c907376c7febc6527b.jpg)  
ES [15]  
RED [73]  
ABF [74]  
Fig. 5. CMER visual comparison for over-exposure (top) and under-exposure (bottom), displayed using the same RAW-to-sRGB ISP.

## V. EXPERIMENTS

We evaluate RIECR against representative sRGB- and RAW-domain methods, followed by component, domain, efficiency, and cross-camera analyses.

## A. Experimental Settings

Compared Methods. We compare eight sRGB methods: MSEC [10], ECCM [71], ECLNet [12], ENC [72], FEC-

Net [11], Retinexformer (RF) [36], CSEC [16], and ES [15]; and eight RAW methods: SID [18], RED [73], ABF [74], CGNet [9], RawHDR [31], SMG [75], ADU [76], and RRM [39]. We adapt sRGB input/output layers to packed fourchannel RAW and retrain all methods on identical splits and protocols using official implementations when available.

Evaluation Metrics. We report PSNR and SSIM [77] on normalized RAW outputs for high-light, low-light, and their average; higher values indicate better restoration. All visualizations use the same RAW-to-sRGB ISP.

Implementation Details. We train on random 256×256 crops using PyTorch [78] and Adam [79] $( \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 )$ with batch size 4 and learning rate 10<sup>−4</sup>. LICN is initialized for 50 epochs with $( \gamma _ { 1 } , \gamma _ { 2 } ) = ( 1 , 0 . 1 )$ to learn linear correction. Joint training uses (0.5, 0.01) and halves the rate every 50 epochs. Training lasts 50 epochs on SMIT and 500 on CMER using an NVIDIA GeForce 4090.

![](images/547873c17564a2d7ce10691da2bcee0d5a16cbf0d550e5ee1278e31fedfac657.jpg)  
Input

![](images/7678c320a68d3eb36cec6232c440fbd69b6a5192c44e11caecd7fe3ada501aad.jpg)  
w/o Guidance

![](images/09ddaeb58de1ccd4f1a3597db67c5467f9dca895ab9d6858a5eadbab8a415939.jpg)  
w/o Label

![](images/d8c06b7fd809fb9ac1c759f9ec0edb83a8cf7aa7afde99ce10082a98bb9f6e59.jpg)  
w/o Linear

![](images/af44cb8397d88e3ec295432a0d7a9a3312fc48bf921a29787763fd6b9c112c19.jpg)  
Ours

![](images/5afc97d619abff650c15f570b6384f7df3bc32be3a4bfd3e7fb51651aab7ad5e.jpg)  
Reference  
Fig. 6. Comparison of illumination correction by the LICN in our model. The term w/o Guidance indicates adjustment without separating the color channels w/o Label means adjustment without estimating the illumination label, w/o Linear means adjustment without linear operation, instead of an illumination map that is added to the original image.

TABLE III  
ABLATION OF THE PROPOSED RIECR FRAMEWORK.
<table><tr><td rowspan="2">LICN</td><td rowspan="2">FEN</td><td colspan="2">High-light</td><td colspan="2">Low-light</td><td colspan="2">Average</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>PSNR↑</td><td>SSIM↑</td><td>PSNR↑</td><td>SSIM↑</td></tr><tr><td>x</td><td>x</td><td>26.35</td><td>0.894</td><td>27.61</td><td>0.825</td><td>27.13</td><td>0.851</td></tr><tr><td>√</td><td>x</td><td>29.90</td><td>0.950</td><td>30.67</td><td>0.808</td><td>30.38</td><td>0.861</td></tr><tr><td>x</td><td>√</td><td>30.07</td><td>0.939</td><td>31.37</td><td>0.931</td><td>30.88</td><td>0.934</td></tr><tr><td>√</td><td>√</td><td>31.23</td><td>0.950</td><td>33.45</td><td>0.950</td><td>32.63</td><td>0.950</td></tr></table>

## B. Comparison with State-of-the-Art Methods

Quantitative Results. Table II reports CMER-only training and SMIT pre-training followed by CMER fine-tuning under identical real-data supervision. RIECR achieves the best average PSNR/SSIM under both protocols and exposure directions: 32.63 dB/0.950 and 33.85 dB/0.951, exceeding the respective second-best averages by 1.61/1.27 dB and 0.014/0.013 SSIM. The gain from pre-training shows that SMIT complements rather than replaces real paired supervision.

(1) Physics-Aware vs. Generic Architectures. Generic sRGB architectures such as FECNet [11] and RF [36] learn strong mappings but do not explicitly model RAW-linear exposure. LICN estimates illumination from channel-specific RAW cues, while FEN handles the exposure-dependent residual, producing consistent gains over both baselines.

(2) Unified vs. Task-Specific RAW Restoration. SID [18] and SMG [75] emphasize low-light recovery, whereas CGNet [9] and RawHDR [31] target high-light/HDR restoration. Against the strongest average RAW baseline, RRM [39], RIECR gains 2.58/2.37 dB under the two protocols. This supports sharing linear correction while adapting residual restoration to the exposure direction.

(3) Computational Efficiency. RIECR uses 2.56 M parameters and 31.91 G MACs. Relative to CGNet/RawHDR, it reduces parameters by 85.5%/75.6% and MACs by 31.0%/37.2%; versus RRM it uses 66.5% fewer parameters at comparable MACs. Lightweight models such as ECLNet, FECNet, RED, and ADU require fewer MACs but restore substantially worse, indicating that RIECR offers a favorable accuracy–complexity trade-off without relying on an excessively large parameter count.

TABLE IV  
COMPONENT ABLATION OF LICN AND FEN.
<table><tr><td rowspan="2">Methods</td><td colspan="2">High-light</td><td colspan="2">Low-light</td><td colspan="2">Average</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>PSNR↑</td><td>SSIM↑</td><td>PSNR↑</td><td>SSIM↑</td></tr><tr><td colspan="7">LICN variants</td></tr><tr><td>w/o Linear</td><td>30.47</td><td>0.924</td><td>30.22</td><td>0.913</td><td>30.31</td><td>0.917</td></tr><tr><td>w/o Label</td><td>31.68</td><td>0.944</td><td>32.43</td><td>0.920</td><td>32.15</td><td>0.929</td></tr><tr><td>w/o Guidance</td><td>30.10</td><td>0.940</td><td>31.88</td><td>0.937</td><td>31.21</td><td>0.938</td></tr><tr><td colspan="7">FEN variants</td></tr><tr><td>w/o Local</td><td>29.41</td><td>0.944</td><td>31.39</td><td>0.929</td><td>30.65</td><td>0.935</td></tr><tr><td>w/o Global</td><td>30.65</td><td>0.950</td><td>30.33</td><td>0.926</td><td>30.45</td><td>0.935</td></tr><tr><td>w/o FDC</td><td>29.08</td><td>0.932</td><td>32.08</td><td>0.926</td><td>30.96</td><td>0.929</td></tr><tr><td>RIECR (Ours)</td><td>31.23</td><td>0.950</td><td>33.45</td><td>0.950</td><td>32.63</td><td>0.950</td></tr></table>

![](images/42b328152365394497a0952d70dce7a4f3a17b2c12dd716c7ba644604baf00be.jpg)

![](images/f0dbfa775020f736a3a6e3fe052f7f4519ccef1c152d7b881cbf8ec96a8f398e.jpg)

![](images/65b0fbef9d41fb34bb00e0b53ac866b8b6f2ba50211633ab52c55a4444df0f5c.jpg)  
w/o Local  
Ours  
Reference  
Fig. 7. Visual quality comparison of FEN. Local and Global mean the local convolution branch and global attention branch, respectively.

Perceptual Quality. Fig. 5 compares both exposure directions after the same ISP. For over-exposure, several methods leave color casts or miss clipped window structure, whereas RIECR preserves tone and highlight edges. For under-exposure, ENC [72] and FECNet [11] retain artifacts or over-smooth details. RIECR suppresses amplified noise while maintaining boundaries, consistent with the LICN/FDC design.

## C. Ablation Studies

Unified Framework. LICN and FEN raise the 27.13 dB baseline to 30.38 and 30.88 dB, respectively; together they reach 32.63 dB/0.950 and perform best on both subsets (Table III). This verifies that physically constrained illumination correction and learned residual restoration are complementary rather than interchangeable.

Linear Illumination Correction Network. In Table IV, replacing linear gain with residual addition loses 2.32 dB;

![](images/e4d8b87c5ca280edd2fda0c2f3b78c1e6a676e443f55919ca529993f80f92aa9.jpg)  
(a) FDC Input

![](images/15cda5f02150f414b37c8c34396868c2dfa82ad74f15a3f7f8e757fb035e82c1.jpg)  
(b) Low-frequency

![](images/32e37958d747a56b190011e05f41fbbb449276f3a70df14d8f16d0711fbc71d7.jpg)  
(c) Standard Input

![](images/e5cdd27ec5db8500555e5e379129d93cfa34a65617fa2327478bdefaf336bd0f.jpg)  
(d) FDC Output

![](images/360ce8536af31613abdb1191260c9de84ec49319d0f79528a32e45e0c526c6ac.jpg)  
(e) High-frequency

![](images/bc2b614300d0761b94964e944e5cbb7130987506d04a0451b0da3f5903c19c55.jpg)  
(f) Standard Output  
Fig. 8. FDC features in the first global-branch layer: (a,d) FDC input/output, (b,e) decomposed low/high frequencies, and (c,f) standard-convolution input/output. FDC retains structure while isolating residual detail.

TABLE V  
SRGB VERSUS RAW INPUT; METRICS ARE EVALUATED IN RGB.
<table><tr><td rowspan="2">Setting</td><td>High-light</td><td>Low-light</td><td></td><td>Average</td><td></td></tr><tr><td>PSNR↑ SSIM↑</td><td>PSNR↑</td><td>SSIM↑</td><td>PSNR↑</td><td>SSIM↑</td></tr><tr><td>sRGB</td><td>27.14 0.911</td><td>29.00</td><td>0.902</td><td>28.35</td><td>0.905</td></tr><tr><td>RAW</td><td>29.43 0.925</td><td>29.73</td><td>0.915</td><td>29.62</td><td>0.919</td></tr></table>

removing the exposure label or channel guidance also degrades performance. The variants confirm the need for exposureaware gating and complementary Green and Red/Blue cues. As visualized in Fig. 6, our LICN achieves a result close to the reference image, validating the necessity of our channelspecific design.

Frequency-Based Enhancement Network. Removing either FEN branch reduces performance in Table IV. Fig. 7 confirms that local processing suppresses noise while global attention supplies context; their combination best preserves structure.

Frequency Dynamic Convolution. The “w/o FDC” row in Table IV drops average PSNR from 32.63 to 30.96 dB. Fig. 8 shows that FDC retains low-frequency structure while isolating high-frequency residuals for exposure-adaptive modulation.

## D. Discussion

Evaluation between RAW and sRGB. We train RIECR on CMER RAW data and sRGB counterparts generated by the same ISP [21], and evaluate both through that ISP. Table V shows that RAW improves average PSNR/SSIM from 28.35 dB/0.905 to 29.62 dB/0.919. The gain occurs in both subsets because linear high-bit-depth RAW preserves highlights before tone mapping and enables physically consistent low-light scaling and noise restoration. As shown in Fig. 9, the visual example further shows that RAW input preserves recoverable high-light details that are already compressed or clipped in sRGB. This confirms that the rich detail, higher bit depth, and linear response of RAW data are crucial for joint exposure correction and restoration.

Evaluation on Cross-Camera Setting. Trained on Canon 5D IV, RIECR transfers without retraining to the Sony IMX586 LRD dataset [80]. Fig. 10 shows corrected global tone and recovered structure despite sensor and pipeline changes, suggesting that the channel-aware decomposition is less devicespecific than a learned sRGB mapping.

![](images/cc7ae47ef0a1b9f0e4fd83c2f8b804a0569b999700b2ce56e60ed2b985d20f65.jpg)  
(a) Input

![](images/e6a5bf919426fe61c6d5be4b73ddd37e08ef90415f2a4f6bcf19d6d746f7c2d9.jpg)  
(b) sRGB

![](images/9fa7d9445ef20ba4d4731c0878dfa43c62fadc04d24966384aaf58f5484bc060.jpg)  
(c) RAW

(d) Reference  
![](images/dbd42cdc86e10e3386c6c6d4d671d626d145e443669d1b839befc19b762896e2.jpg)

Fig. 9. The visualized comparison between the sRGB and RAW domains.  
![](images/056afec2a385af54529a66f6d9f1cb4f0bd0c70b035a6a8c38f7d39aa7e24922.jpg)

![](images/716d48d1569bd4143be0ccebdda03ec28d555c0f5772b159236b78aadfb1f99a.jpg)

![](images/b7df202f55d7db1257cbe14a24f26b5f76eda712bcdf28787feeacb39009f156.jpg)  
(a) Input

![](images/0f1e8c0edc48b06c9ca348476649c15890e4a811036b2d32353f5c0477ee516c.jpg)  
(c) Ours  
(d) Reference

(b) ES [15]  
Fig. 10. The visualized results for the cross-camera setting. The model is trained on Canon 5D IV and evaluated on the Sony IMX586 sensor.  
![](images/1e9699e21e009656f19a742518c504a50dc882f2062ab2a4786e46b25810e273.jpg)

![](images/6691572545cf9471a2dcaa7113aec5bcb774d34c2723c3ab36fd418f9088aa81.jpg)  
(a) Reference  
(b) Ours

![](images/25876176bd21fec0bc27780dcca1c90a2f62993b02ad614a44b3d22b8017fdf0.jpg)  
(c) Input

![](images/bc07ac45028279f92f6b00fb183d57a8b91d5654a2722e47569e7e5806b2af89.jpg)  
(d) Input data range  
Fig. 11. Failure example for over-exposure areas. The histogram of the input data range indicates that some pixels are saturated.

Failure Case and Limitation. Table V shows better detail recovery from RAW than from sRGB. However, RIECR remains limited in large, fully saturated highlight regions (Fig. 11). When all RAW channels are clipped, the original signal cannot be uniquely recovered from a single observation. Contextual inference may produce plausible textures but cannot guarantee recovery of the original details or colors. Future work could incorporate generative priors to improve visual plausibility in these severely degraded regions.

## VI. CONCLUSION

In this paper, we proposed RIECR, a unified RAW-domain framework that bridges the information loss of sRGB correction and the single-degradation focus of task-specific RAW restoration. Following the physical RAW formation model, RIECR decomposes joint under- and over-exposure restoration into linear illumination correction and residual enhancement. LICN estimates exposure-aware gains through channelspecific guidance, using the high SNR of Green channels for under-exposure and the less saturated Red/Blue responses for over-exposure. FEN then addresses the divergent residual degradations, while FDC separates low- and high-frequency responses to adaptively suppress amplified noise and recover weakened high-light structures. We further constructed CMER with 223 real scenes and 1,784 paired RAW images and built a synthetic dataset for pre-training. Comparisons with sixteen methods, component ablations, RAW/sRGB evaluation, efficiency analysis, and cross-camera tests validate the unified formulation and its components. RIECR achieves 33.85 dB/0.951 after pre-training and fine-tuning with 2.56 M parameters. Future work will extend CMER to dynamic scenes and broader sensors and explore metadata-aware correction and stronger priors for severely clipped regions.

[1] L. Chen, Y. Fu, K. Wei, D. Zheng, and F. Heide, “Instance segmentation in the dark,” International Journal of Computer Vision, vol. 131, no. 8, pp. 2198–2218, 2023.

[2] P. Oza, V. A. Sindagi, V. V. Sharmini, and V. M. Patel, “Unsupervised domain adaptation of object detectors: A survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023.

[3] L. Chen, Y. Fu, J. Shangguan, J. Xu, and Y. Peng, “A survey on degraded image segmentation,” Chinese Journal of Electronics, vol. 35, no. 5, pp. 1–21, 2026.

[4] Q. Liu, R. Liu, B. Zheng, H. Wang, and Y. Fu, “Infrared small target detection with scale and location sensitivity,” in Proc. of Computer Vision and Pattern Recognition. IEEE, 2024, pp. 17 490–17 499.

[5] R. Liu, J. Nie, and Y. Fu, “Rdanet: Relative degradation aware network for infrared small target detection,” IEEE Transactions on Geoscience and Remote Sensing, 2026.

[6] Y. Fu, Y. Li, S. You, B. Shi, L. Chen, Y. Zou, Z. Wang, Y. Li, Y. Han, Y. Zhang et al., “Technique report of cvpr 2024 pbdl challenges,” arXiv preprint arXiv:2406.10744, 2024.

[7] K. Wei, Y. Fu, J. Yang, and H. Huang, “A physics-based noise formation model for extreme low-light raw denoising,” in Proc. ofComputer Vision and Pattern Recognition, 2020, pp. 2758–2767.

[8] X. Guo and Q. Hu, “Low-light image enhancement via breaking down the darkness,” International Journal of Computer Vision, vol. 131, no. 1, pp. 48–66, 2023.

[9] Y. Fu, Y. Hong, Y. Zou, Q. Liu, Y. Zhang, N. Liu, and C. Yan, “Raw image based over-exposure correction using channel-guidance strategy,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 34, no. 4, pp. 2749–2762, 2023.

[10] M. Afifi, K. G. Derpanis, B. Ommer, and M. S. Brown, “Learning multi-scale photo exposure correction,” in Proc. of Computer Vision and Pattern Recognition, 2021, pp. 9157–9167.

[11] J. Huang, Y. Liu, F. Zhao, K. Yan, J. Zhang, Y. Huang, M. Zhou, and Z. Xiong, “Deep fourier-based exposure correction network with spatialfrequency interaction,” in Proc. of European Conference on Computer Vision. Springer, 2022, pp. 163–180.

[12] J. Huang, M. Zhou, Y. Liu, M. Yao, F. Zhao, and Z. Xiong, “Exposureconsistency representation learning for exposure correction,” in Proc. of the ACM International Conference on Multimedia, 2022, pp. 6309–6317.

[13] J. Huang, F. Zhao, M. Zhou, J. Xiao, N. Zheng, K. Zheng, and Z. Xiong, “Learning sample relationship for exposure correction,” in Proc. of Computer Vision and Pattern Recognition, 2023, pp. 9904–9913.

[14] J.-H. Baek, D. Kim, S.-M. Choi, H.-j. Lee, H. Kim, and Y. J. Koh, “Luminance-aware color transform for multiple exposure correction,” in Proc. of International Conference on Computer Vision, 2023, pp. 6156– 6165.

[15] D. Jung, D. Kim, G. Wang, and T. H. Kim, “Exposure-slot: Exposurecentric representations learning with slot-in-slot attention for regionaware exposure correction,” in Proc. of Computer Vision and Pattern Recognition, 2025, pp. 17 892–17 901.

[16] Y. Li, K. Xu, G. P. Hancke, and R. W. Lau, “Color shift estimation-andcorrection for image enhancement,” in Proc. of Computer Vision and Pattern Recognition, 2024, pp. 25 389–25 398.

[17] K.-F. Yang, C. Cheng, S.-X. Zhao, H.-M. Yan, X.-S. Zhang, and Y.-J. Li, “Learning to adapt to light,” International Journal of Computer Vision, vol. 131, no. 4, pp. 1022–1041, 2023.

[18] C. Chen, Q. Chen, J. Xu, and V. Koltun, “Learning to see in the dark,” in Proc. of Computer Vision and Pattern Recognition, 2018, pp. 3291– 3300.

[19] H. Huang, W. Yang, Y. Hu, J. Liu, and L.-Y. Duan, “Towards low light enhancement with raw images,” IEEE Transactions on Image Processing, vol. 31, pp. 1391–1405, 2022.

[20] J. Liu, D. Xu, W. Yang, M. Fan, and H. Huang, “Benchmarking low-light image enhancement and beyond,” International Journal of Computer Vision, vol. 129, pp. 1153–1184, 2021.

[21] K. Wei, Y. Fu, Y. Zheng, and J. Yang, “Physics-based noise modeling for extreme low-light photography,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 11, pp. 8520–8537, 2021.

[22] P. Hanji, R. K. Mantiuk, G. Eilertsen, S. Hajisharif, and J. Unger, “Comparison of single image hdr reconstruction methods-the caveats of quality assessment,” in Proc. of the special interest group on Computer Graphics and Interactive Techniques, 2022.

[23] V. Bychkovsky, S. Paris, E. Chan, and F. Durand, “Learning photographic global tonal adjustment with a database of input/output image pairs,” in Proc. of Computer Vision and Pattern Recognition, 2011, pp. 97–104.

[24] F. Banterle, K. Debattista, A. Artusi, S. Pattanaik, K. Myszkowski, P. Ledda, and A. Chalmers, “High dynamic range imaging and low dynamic range expansion for generating hdr content,” in Computer Graphics Forum, vol. 28, no. 8, 2009, pp. 2343–2367.

[25] G. Eilertsen, J. Kronander, G. Denes, R. K. Mantiuk, and J. Unger, “Hdr image reconstruction from a single exposure using deep cnns,” ACM Transactions on Graphics, vol. 36, no. 6, pp. 1–15, 2017.

[26] D. Marnerides, T. Bashford-Rogers, J. Hatchett, and K. Debattista, “Expandnet: A deep convolutional neural network for high dynamic range expansion from low dynamic range content,” in Computer Graphics Forum, vol. 37, no. 2, 2018, pp. 37–49.

[27] R. Chen, B. Zheng, H. Zhang, Q. Chen, C. Yan, G. Slabaugh, and S. Yuan, “Improving dynamic hdr imaging with fusion transformer,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, no. 1, 2023, pp. 340–349.

[28] S. Lee, G. H. An, and S.-J. Kang, “Deep recursive hdri: Inverse tone mapping using generative adversarial networks,” in Proc. of European Conference on Computer Vision, 2018, pp. 596–611.

[29] Y.-L. Liu, W.-S. Lai, Y.-S. Chen, Y.-L. Kao, M.-H. Yang, Y.-Y. Chuang, and J.-B. Huang, “Single-image hdr reconstruction by learning to reverse the camera pipeline,” in Proc. of Computer Vision and Pattern Recognition, 2020, pp. 1651–1660.

[30] X. Chen, Y. Liu, Z. Zhang, Y. Qiao, and C. Dong, “Hdrunet: Single image hdr reconstruction with denoising and dequantization,” in Proc. of Computer Vision and Pattern Recognition, 2021, pp. 354–363.

[31] Y. Zou, C. Yan, and Y. Fu, “Rawhdr: High dynamic range image reconstruction from a single raw image,” in Proc. of International Conference on Computer Vision, 2023, pp. 12 334–12 344.

[32] Q.-C. Tian and L. D. Cohen, “Global and local contrast adaptive enhancement for non-uniform illumination color images,” in Proc. of International Conference on Computer Vision workshops, 2017, pp. 3023–3030.

[33] M. R. Hajiaboli, “An anisotropic fourth-order diffusion filter for image noise removal,” International Journal of Computer Vision, vol. 92, pp. 177–191, 2011.

[34] Q. Zhang, G. Yuan, C. Xiao, L. Zhu, and W.-S. Zheng, “High-quality exposure correction of underexposed photos,” in Proc. of the ACM International Conference on Multimedia, 2018, pp. 582–590.

[35] X. Li, M. Liu, and Q. Ling, “Pixel-wise gamma correction mapping for low-light image enhancement,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 34, no. 2, pp. 681–694, 2023.

[36] Y. Cai, H. Bian, J. Lin, H. Wang, R. Timofte, and Y. Zhang, “Retinexformer: One-stage retinex-based transformer for low-light image enhancement,” in Proc. of International Conference on Computer Vision, 2023, pp. 12 504–12 513.

[37] Q. Yan, Y. Feng, C. Zhang, G. Pang, K. Shi, P. Wu, W. Dong, J. Sun, and Y. Zhang, “Hvi: A new color space for low-light image enhancement,” in Proc. of Computer Vision and Pattern Recognition, 2025, pp. 5678– 5687.

[38] P. Huang, X. Chen, X. Feng, Q. Wang, D. Zhang, L. Han, and J. Han, “Decouplenet: Domain-specific task decoupling network for low-light image enhancement,” Pattern Recognition, p. 112203, 2025.

[39] X. Chen, L. Han, P. Huang, X. Feng, D. Zhang, and J. Han, “Retinexrawmamba: Bridging demosaicing and denoising for low-light raw image enhancement,” IEEE Transactions on Circuits and Systems for Video Technology, 2025.

[40] H. Li, Z. Wu, R. Shao, T. Zhang, and Y. Fu, “Noise calibration and spatial-frequency interactive network for stem image enhancement,” in Proc. of Computer Vision and Pattern Recognition, 2025, pp. 21 287– 21 296.

[41] H. Li, Z. Wu, R. Shao, and Y. Fu, “Statistical characteristic-guided denoising for rapid high-resolution transmission electron microscopy imaging,” in Proc. of Computer Vision and Pattern Recognition, 2026, pp. 34 050–34 060.

[42] B. Zheng, H. Li, Q. Chen, T. Wang, X. Zhou, Z. Hu, and C. Yan, “Quad bayer joint demosaicing and denoising based on dual encoder network with joint residual learning,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 7, 2024, pp. 7552–7561.

[43] Z. Liu, B. Zheng, Q. Chen, Q. Zhang, X. Jia, J. Zhang, and C. Yan, “Pyramid learnable bandpass filters for ultra-high-definition image

demoireing,”´ IEEE Transactions on Circuits and Systems for Video Technology, vol. 35, no. 8, pp. 7821–7834, 2025.

[44] Y. Zou, C. Yan, and Y. Fu, “Iterative denoiser and noise estimator for self-supervised image denoising,” in Proc. of International Conference on Computer Vision, October 2023, pp. 13 265–13 274.

[45] Y. Zou and Y. Fu, “Estimating fine-grained noise model via contrastive learning,” in Proc. of Computer Vision and Pattern Recognition, June 2022, pp. 12 682–12 691.

[46] Y. Zhang, T. Zhang, J. Nie, and Y. Fu, “Real noise decoupling for hyperspectral image denoising,” in Proc. of the AAAI Conference on Artificial Intelligence, vol. 40, no. 15, 2026, pp. 12 925–12 933.

[47] W.-M. Ke, C.-R. Chen, and C.-T. Chiu, “Bita/swce: Image enhancement with bilateral tone adjustment and saliency weighted contrast enhancement,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 21, no. 3, pp. 360–364, 2010.

[48] X. Fu, D. Zeng, Y. Huang, X.-P. Zhang, and X. Ding, “A weighted variational model for simultaneous reflectance and illumination estimation,” in Proc. of Computer Vision and Pattern Recognition, 2016, pp. 2782–2790.

[49] J. Liu, J. Shang, R. Liu, and X. Fan, “Attention-guided global-local adversarial learning for detail-preserving multi-exposure image fusion,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 32, no. 8, pp. 5026–5040, 2022.

[50] Y. Cao, Y. Ren, T. H. Li, and G. Li, “Over-exposure correction via exposure and scene information disentanglement,” in Proc. of the Asian Conference on Computer Vision, 2020.

[51] Y. Tian, Y. Fu, and J. Zhang, “Transformer-based under-sampled singlepixel imaging,” Chinese Journal ofElectronics, vol. 32, no. 5, pp. 1151– 1159, 2023.

[52] Y. Fu and Y. Zhang, “Reinforcement learning based plug-and-play method for hyperspectral image reconstruction,” in CAAI International Conference on Artificial Intelligence. Springer, 2022, pp. 466–477.

[53] Z. Fang, Y. Fu, and L. Liu, “A dual of transformer features-related mapintelligent generation method,” Journal of Image and Graphics, vol. 28, no. 10, pp. 3281–3294, 2023.

[54] T. Zhang, B. Wang, Y. Fu, S. Liu, J. Ye, P. Shan, and C. Yan, “Deep learning-based spectral image super-resolution: A survey,” Journal of Image and Graphics, vol. 29, no. 8, pp. 2113–2136, 2024.

[55] Z. Li, Y. Shao, F. Zhang, J. Zhang, Y. Wang, and N. Sang, “Difficultyaware dynamic network for lightweight exposure correction,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 34, no. 6, pp. 5033–5048, 2023.

[56] X. Hu, X. Chen, B. Ni, T. Li, and Y. Liu, “Bi-volution: A static and dynamic coupled filter,” in Proc. of the AAAI Conference on Artificial Intelligence, vol. 36, no. 1, 2022, pp. 960–968.

[57] L. Chen, L. Gu, D. Zheng, and Y. Fu, “Frequency-adaptive dilated convolution for semantic segmentation,” in Proc. of Computer Vision and Pattern Recognition, 2024, pp. 3414–3425.

[58] Y. Zhang, Z. Lai, T. Zhang, Y. Fu, and C. Zhou, “Unaligned rgb guided hyperspectral image super-resolution with spatial-spectral concordance,” International Journal of Computer Vision, vol. 133, no. 9, pp. 6590– 6610, 2025.

[59] H. Li and Y. Fu, “Fcdfusion: A fast, low color deviation method for fusing visible and infrared image pairs,” Computational Visual Media, vol. 11, no. 1, pp. 195–211, 2025.

[60] Y. Zhang, T. Zhang, J. Nie, and Y. Fu, “Enhancing unregistered hyperspectral image super-resolution via unmixing-based abundance fusion learning,” in Proc. of Computer Vision and Pattern Recognition, 2026, pp. 41 573–41 583.

[61] M. Li and Y. Fu, “Joint self-attention transformer for multispectral and hyperspectral image fusion,” Journal of Image and Graphics, vol. 28, no. 12, pp. 3922–3934, 2023.

[62] Y. Zhang, S. Chen, Y. Tian, Y. Gao, J. Jiang, and Y. Fu, “Superviseassisted multi-modality fusion diffusion model for pet restoration,” Tsinghua Science and Technology, 2026.

[63] J. Yang, Q. Min, W. Lu, W. Yao, Y. Ma, J. Du, T. Lu, and G. Liu, “An automated cloud detection method based on the green channel of total-sky visible images,” Atmospheric Measurement Techniques, vol. 8, no. 11, pp. 4671–4679, 2015.

[64] T. Zhang, Y. Fu, J. Zhang, and C. Yan, “Deep guided attention network for joint denoising and demosaicing in real image,” Chinese Journal of Electronics, vol. 33, no. 1, pp. 303–312, 2024.

[65] X. Huang and S. Belongie, “Arbitrary style transfer in real-time with adaptive instance normalization,” in Proc. of International Conference on Computer Vision, Oct 2017.

[66] C. Wei, W. Wang, W. Yang, and J. Liu, “Deep retinex decomposition for low-light enhancement,” Proc. ofthe British Machine Vision Conference, 2018.

[67] J. Cai, S. Gu, and L. Zhang, “Learning a deep single image contrast enhancer from multi-exposure images,” IEEE Transactions on Image Processing, vol. 27, no. 4, pp. 2049–2062, 2018.

[68] J. L. Ba, J. R. Kiros, and G. E. Hinton, “Layer normalization,” arXiv preprint arXiv:1607.06450, 2016.

[69] T. Zhu, H. Li, and Y. Fu, “Trim-sod: A multi-modal, multi-task, and multi-scale spacecraft optical dataset,” Space: Science & Technology, vol. 5, p. 0299, 2025.

[70] E. Reinhard, W. Heidrich, P. Debevec, S. Pattanaik, G. Ward, and K. Myszkowski, High dynamic range imaging: acquisition, display, and image-based lighting. Morgan Kaufmann, 2010.

[71] N. E. Nsampi, Z. Hu, and Q. Wang, “Learning exposure correction via consistency modeling,” in Proc. of the British Machine Vision Conference, 2021.

[72] J. Huang, Y. Liu, X. Fu, M. Zhou, Y. Wang, F. Zhao, and Z. Xiong, “Exposure normalization and compensation for multiple-exposure correction,” in Proc. of Computer Vision and Pattern Recognition, 2022, pp. 6043–6052.

[73] M. Lamba and K. Mitra, “Restoring extremely dark images in real time,” in Proc. of Computer Vision and Pattern Recognition, 2021, pp. 3487– 3497.

[74] X. Dong, W. Xu, Z. Miao, L. Ma, C. Zhang, J. Yang, Z. Jin, A. B. J. Teoh, and J. Shen, “Abandoning the bayer-filter to see in the dark,” in Proc. of Computer Vision and Pattern Recognition, 2022, pp. 17 431– 17 440.

[75] X. Xu, R. Wang, and J. Lu, “Low-light image enhancement via structure modeling and guidance,” in Proc. of Computer Vision and Pattern Recognition, 2023, pp. 9893–9903.

[76] J. Huang, H. Ren, S. Liu, Y. Liu, C. Lv, J. Lu, C. Xie, and H. Lu, “Realtime attentive dilated u-net for extremely dark image enhancement,” ACM Transactions on Multimedia Computing, Communications and Applications, vol. 20, no. 8, pp. 1–19, 2024.

[77] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: from error visibility to structural similarity,” IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600–612, 2004.

[78] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga et al., “Pytorch: An imperative style, high-performance deep learning library,” Proc. of the Advances in Neural Information Processing Systems, vol. 32, 2019.

[79] D. Kinga, J. B. Adam et al., “A method for stochastic optimization,” in Proc. of the International Conference on Learning Representations, vol. 5, 2015, p. 6.

[80] F. Zhang, B. Xu, Z. Li, X. Liu, Q. Lu, C. Gao, and N. Sang, “Towards general low-light raw noise synthesis and modeling,” in Proc. of International Conference on Computer Vision, 2023, pp. 10 820– 10 830.