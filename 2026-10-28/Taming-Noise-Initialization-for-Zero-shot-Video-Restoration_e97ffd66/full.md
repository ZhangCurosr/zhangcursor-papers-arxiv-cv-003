# Taming Noise Initialization for Zero-shot Video Restoration with Image Diffusion Priors

Chong Wang, Zixuan Fu, Hao Cheng Member, IEEE, Siyuan Yang Member, IEEE, Alex C. Kot, Life Fellow, IEEE, Bihan Wen, Senior Member, IEEE

Abstract—Recent advances in diffusion models have achieved remarkable success in restoring high-quality images in a zeroshot manner. Unfortunately, extending this success to video restoration is challenging due to the substantial computational resources required to pre-train video diffusion models. A rising trend seeks to address video restoration by utilizing a pretrained image diffusion model, initiating restoration from either frame-wise independent or identical Gaussian noise. Yet, these approaches often fall short, as such simple noise initiations fail to capture and utilize the inherent temporal correlations within degraded video frames. In this work, we propose taming noise initialization for zero-shot video restoration, which preserves the intrinsic temporal correlations by generating initial noises directly using DDIM inversion on the corrupted video frames themselves. Building on this, we introduce an adaptive noise blending strategy that suppresses degradation patterns encoded during DDIM inversion while gradually steering the noise toward a Gaussian distribution. To further emphasize the temporal consistency, we propose measurement-guided trajectory attention to better align features across frames. Extensive experiments on multiple video restoration tasks demonstrate that our method outperforms existing zero-shot approaches, consistently achieving superior results on the DAVIS, REDS4, and Set8 datasets.

Index Terms—Diffusion models, video restoration, zero-shot restoration.

## I. INTRODUCTION

D <sup>ENOISING</sup> <sup>diffusion</sup> <sup>models</sup> <sup>[2]</sup> <sup>have</sup> <sup>demonstrated</sup> <sup>ex-</sup>ceptional performance in synthesizing images and videos ceptional performance in synthesizing images and videos with promising perceptual quality [3]–[9]. Recently, such effectiveness in modeling complex distributions has motivated a wide application in both image and video-based restoration tasks [10]–[17]. While task-specific diffusion models achieve impressive results on their targeted problems, they frequently struggle with generalizability across various restoration tasks. Instead, another line of research tends to exploit the generative power inherent in a pre-trained diffusion network to address various image restoration problems in a zero-shot manner [18]–[22].

Despite the success of zero-shot approaches in image restoration, extending these achievements to video restoration poses significant challenges. Pre-training video diffusion models from scratch requires substantial computational resources, making it impractical for various applications. Consequently, how to effectively address zero-shot video restoration remains largely underexplored. Very recently, researchers have attempted to bridge this gap by adapting pre-trained image diffusion models for video restoration without retraining [1], [23]. In [23], the authors proposed to utilize the temporal consistency guidance to warp the information between frames and prevent temporal misalignment in the restored video. Alternatively, another work [1] adopts a simple strategy that enforces identical initial noise across frames, thereby ensuring temporal consistency.

![](images/9fde9f924887af9f23bc93dbf11104dcf597deae976e28c4e0202a20b591eed2.jpg)  
Fig. 1. (a) Existing method [1] utilizes a simple identical Gaussian noise initialization but fails to ensure frame consistency. (b) Our method tames the DDIM inversion noise with adaptive filtering while capturing the inherent temporal correlation in the degraded video, achieving results with guaranteed temporal consistency.

Video, known for its unique characteristics, contains timecorrelated scenes and exhibits varying motion dynamics. However, existing approaches directly start restoration with a simple noise initiation, completely neglecting the temporal correlation across frames. Specifically, [1] initializes restoration with identical noise for all frames, imposes an overly strong restriction between frames, potentially constraining the natural dynamics of the restored frames. Conversely, [23] adopts frame-wise completely independent Gaussian noises that may allow excessive freedom for each frame, thereby violating the temporal correlation across frames. Therefore, determining how to appropriately impose temporal correlations during noise initialization is crucial for achieving promising video restoration results. To be more specific, the initialization determines the starting point of the sampling trajectory and strongly affects (i) how well the reverse process stays within the diffusion model learned data manifold, and (ii) how effectively the subsequent steps can correct the trajectory to satisfy the physical degradation model. To this end, we argue that the degraded videos inherently contain sufficient temporally correlated information, which can be effectively utilized to initialize the “right” noise for the diffusion restoration process.

In this work, we propose a novel zero-shot framework for video restoration, called TINDER, which only necessitates a pre-trained image diffusion prior. Motivated by the success of DDIM inversion in video editing [24]–[27], which establish a deterministic mapping between clean image and the corresponding noise sample. Our method starts by performing DDIM inversion on the degraded video frames to obtain temporally correlated initial noise, effectively preserving temporal correlations and providing essential temporal guidance for the restoration process. Unfortunately, directly applying DDIM inversion in video restoration will produce sub-optimal results since the degradation pattern in the input frames can also be encoded into the inverted noise. To this end, the main contribution of this paper is introducing an adaptive filtering strategy to tame noise initialization for video restoration. Since video sequences inherently exhibit dynamic content, with significant variations in frequency distribution even across consecutive frames. Thus, our filtering strategy dynamically tailors filters based on the energy distribution of each frame, which is further utilized to blend the DDIM inversion results with Gaussian noise. This strategy not only enables varying yet adaptive filtering on each DDIM inverted noise, but also compensates for the missing high-frequency energy caused by degradation in the inverted noise. Consequently, it steers the DDIM inversion noise towards a Gaussian distribution that better aligns with the diffusion process. To further emphasize temporal consistency, we propose a measurement guided trajectory attention during the diffusion sampling process. More in detail, motion trajectories are estimated from adjacent input frames and incorporated into the attention layer of the diffusion network, ensuring temporal consistency between consecutive frames. Finally, the proposed method offers a flexible framework for video restoration, where data consistency can be guaranteed by plugging any existing imagebased diffusion sampling strategy. Extensive experiments show that the proposed method consistently achieves superior performances in various video restoration tasks on the DAVIS, REDS4, and Set8 datasets.

The main contributions of this work are as follows:

• We propose TINDER, a novel zero-shot video restoration framework with image diffusion priors, where the initial noise is obtained via DDIM inversion to maintain the temporal relation in the degraded video frames.

• We propose an adaptive noise blending strategy to eliminate the degradation pattern in inversion noise for better restoration, followed by a measurement-guided trajectory attention to further guarantee the temporal consistency.

• Extensive experiments on multiple video restoration tasks demonstrate our methods outperform existing competing approaches on the DAVIS, REDS4, and Set8 datasets.

## II. RELATED WORK

## A. Conditional Diffusion for Video Restoration

Denoising diffusion probabilistic models [2]–[4], [28]–[31] have been proven to be a powerful class of unconditional generative models for high-quality image synthesis. This iterative denoising mechanism not only results in high-fidelity and di verse image synthesis but also establishes a flexible framework that can be extended to various conditional tasks. Building on their remarkable success in unconditional image generation, researchers have adapted these models for conditional settings by incorporating additional inputs such as degraded images. This adaptation allows the model to generate out puts that are not only photorealistic but also context-specific, making them particularly well-suited for video restoration tasks where preserving spatial detail and temporal coherence is critical [15], [16], [32]. Traditional video restoration methods [33]–[41] typically employ deterministic mappings that learn a fixed transformation from degraded to restored frames. Recent representative approaches employ Transformer architectures to capture long-range spatio-temporal dependencies while explicitly addressing motion and alignment across frames. For instance, VRT [40] presents a Transformerbased framework with spatio-temporal attention and alignment mechanisms for high-quality video restoration, and RVRT [41] further improves temporal modeling and efficiency via a recurrent formulation with guided deformable attention for correspondence-aware feature aggregation under complex motion. Benefiting from task-specific supervision, these end-to end methods typically achieve strong performance and fast feed-forward inference when the target degradation and data distribution match the training setting. In contrast, diffusion based approaches embrace a probabilistic framework that models the full underlying distribution of clean video frames. This probabilistic nature enables the model to capture intrinsic variability and uncertainty, leading to more robust restoration outcomes even in challenging conditions. Several studie have explored the application of diffusion models in video restoration. For instance, Zhou et al. [15] introduced Upscale A-Video, a text-guided latent diffusion framework for video upscaling that ensures temporal coherence through integrated temporal layers and a flow-guided recurrent latent propagation module. Similarly, SATeCo [32] leverages spatial-temporal guidance extracted from low-resolution videos to calibrate the high-resolution video denoising and restoration.

## B. Diffusion Models of Zero-shot Restoration

Although these task-specific diffusion models excel in narrowly defined problems, they frequently struggle with generalizability across different restoration tasks. Instead, another line of research tends to leverage pre-trained diffusion networks as generative priors to solve restoration tasks in a zero-shot manner [18]–[22], [42].

Zero-shot Image Restoration. Existing zero-shot approaches typically apply constraints on the intermediate estimation, guiding the generative process to align with the degradation process of each specific image restoration task. For example, DDNM [21] uses a range-null space decomposition on the clean image prediction, restricting diffusion to null-space components. A relaxed constraint is introduced in [18] by incorporating a least-squares penalty on the data term to ensure data consistency. In [22], a latent diffusion model enforces data consistency by optimizing the intermediate clean image prediction to minimize the data term during reverse sampling. DPS [19] further extends diffusion models to general nonlinear inverse problems by introducing a manifoldconstrained gradient approximation, enabling wide applicability to noisy cases. Additionally, DDPG [42] proposes an iterative-preconditioned guidance approach that smoothly transitions from back-projection to least-squares gradient steps, enhancing noise robustness in image restoration tasks.

Zero-shot Video Restoration. Building on the successes in image restoration, recent studies have explored extending these techniques to video data. Zero-shot video restoration presents unique challenges due to the need for maintaining temporal consistency across frames where simple frame-by-frame restoration methods often fail to address, leading to noticeable temporal flickering. To overcome this, Cao et al. [23] modify a pre-trained image diffusion model by incorporating crossprevious-frame attention layers. This adaptation explicitly models inter-frame dependencies, thereby mitigating flickering and ensuring that temporal features are coherently maintained across the sequence. Yeh et al. [43] introduce hierarchical latent warping to leverage pre-trained image restoration diffusion models for video restoration. Kwon et al. [1] proposed a video inverse solver that utilizes image diffusion models, introducing a batch-consistent diffusion sampling strategy to ensure consistency across batches. However, existing works only adopt a simple noise initiation, struggling to achieve temporal consistent results. In this paper, we propose to initiate obtaining a temporal-correlated noise by exploiting the temporal correlation inherent in the degraded video, which provides crucial guidance for the restoration process.

## III. PRELIMINARY

## A. DDIM Inversion

We adopt the most popular formulation of diffusion models as described in [2], where there are two main processes. The diffusion process begins with a clean image $\pmb { x } _ { 0 } \sim p ( \pmb { x } )$ from a desired distribution, which is incrementally perturbed by adding small Gaussian noise $\epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ as $\begin{array} { r l } { \mathbf { \boldsymbol { x } } _ { t } } & { { } = } \end{array}$ $\sqrt { 1 - \beta _ { t } } \pmb { x } _ { t - 1 } + \sqrt { \beta _ { t } } \pmb { \epsilon }$ , where $\{ \beta _ { t } \} _ { t = 1 } ^ { T }$ defines a noise schedule in ascending order. Through reparameterization, a perturbed sample $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ could be directly calculated from $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ in a closed form: $\begin{array} { r } { \pmb { x } _ { t } = \sqrt { \bar { \alpha } _ { t } } \pmb { x } _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \pmb { \epsilon } } \end{array}$ where $\alpha _ { t } ~ = ~ 1 - \beta _ { t }$ and $\textstyle { \bar { \alpha } } _ { t } = \prod _ { i = 1 } ^ { t } \alpha _ { i }$ . The diffusion model $\epsilon _ { \theta } ( \cdot )$ is trained to approximate the reverse diffusion process, transitioning from a pure Gaussian noise $\mathbf { \sigma } _ { \mathbf { X } _ { T } } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ to a clean natural image $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ The reverse process in [2] necessitates thousands of iterations, resulting in slow image generation. As a remedy, [44] proposed Denoising Diffusion Implicit Models (DDIM) which define a non-Markovian process as:

$$
{ \pmb x } _ { t - 1 } = \sqrt { \alpha _ { t - 1 } } \frac { { \pmb x } _ { t } - \sqrt { 1 - \alpha _ { t } } { \pmb \epsilon } _ { \theta } ( { \pmb x } _ { t } ) } { \sqrt { \alpha _ { t } } } + \sqrt { 1 - \alpha _ { t - 1 } } { \pmb \epsilon } _ { \theta } ( { \pmb x } _ { t } ) .\tag{1}
$$

As DDIM sampling in (1) is the deterministic process, it can be approximately reversed to map a clean image $x _ { 0 }$ to a noise $x _ { T }$ bijectively:

$$
\pmb { x } _ { t } = \sqrt { \alpha _ { t } } \frac { \pmb { x } _ { t - 1 } - \sqrt { 1 - \alpha _ { t - 1 } } \pmb { \epsilon } _ { \theta } \big ( \pmb { x } _ { t - 1 } \big ) } { \sqrt { \alpha _ { t - 1 } } } + \sqrt { 1 - \alpha _ { t } } \pmb { \epsilon } _ { \theta } \big ( \pmb { x } _ { t - 1 } \big ) .\tag{2}
$$

Thus we can establish a bijective link that bridges the noise with clean images.

## B. Diffusion Sampling with Data Consistency

In both image and video restoration, ensuring data consistency is essential for achieving reliable restoration results. Existing methods [18], [19], [21], [45] strive to apply constraints on the intermediate outputs of the diffusion sampling, iteratively guiding the restoration to align with the physics prior. Notably, Chung et al. [45] introduced a decomposed diffusion sampling approach that effectively integrates diffusion sampling with Krylov subspace methods. They demonstrate that if the tangent space at a denoised sample ${ \mathbf { } } x _ { 0 \mid t } ~ =$ $( { \pmb x } _ { t } \mathrm { ~ - ~ } \sqrt { 1 - \bar { \alpha } _ { t } } { \pmb \epsilon } _ { \theta } ( { \pmb x } _ { t } ) ) / \sqrt { \bar { \alpha } _ { t } }$ constitutes a Krylov subspace, data consistency can be guaranteed by Conjugate Gradient method to solve the optimization as

$$
\pmb { \hat { x } } _ { 0 \mid t } = \underset { { \pmb { x } } \in { \pmb { x } } _ { 0 \mid t } + { \mathcal { K } } _ { l } } { \mathrm { a r g m i n } } \ \Vert \pmb { y } - A { \pmb { x } } \Vert ^ { 2 } ,\tag{3}
$$

where y is the degraded measurement, A denotes the degradation process and $\kappa _ { l }$ is the l-dimension Kyrlov subspace. Inspired by [45], such strategy has been extended to video restoration in [1]. Another research [18] incorporates data consistency by utilizing a weighted sum of least-squares as

$$
\hat { \pmb { x } } _ { 0 \vert t } = \underset { \pmb { x } } { \mathrm { a r g m i n } } \Vert \pmb { y } - \pmb { A x } \Vert ^ { 2 } + \rho \Vert \pmb { x } - \pmb { x } _ { 0 \vert t } \Vert ^ { 2 } ,\tag{4}
$$

where $\rho$ is the weighting parameter.

## IV. METHOD

We start with an introduction to video restoration problem formulation. Given a degraded video with N frames $\boldsymbol { \mathcal { Y } } = \{ \boldsymbol { \mathsf { y } } ^ { i } \} _ { i = 1 } ^ { N }$ , we aim to recover the underlying clean video $\mathcal { X } = \{ \pmb { x } ^ { i } \} _ { i = 1 } ^ { N }$ . The degradation process can be modeled as performing degradation on each frame independently:

$$
\pmb { y } ^ { i } = A \pmb { x } ^ { i } + \pmb { n } ^ { i } ,\tag{5}
$$

where A denotes the forward degradation process on each video frame, and n is the additive Gaussian noise.

![](images/b564863f7d2aa0881c0737fb98f86ce488f7f2cefdd9967764e2c7ec9b41fa18.jpg)

Fig. 2. Overview of the proposed TINDER framework, where the initial noise is first obtained by performing DDIM inversion on the degraded video to capture the inherent temporal correlation. Subsequently, an adaptive noise blending strategy is employed to mitigate the degradation pattern in the inversion noise X<sub>T</sub>, while directing the noise toward a Gaussian distribution. After that, the temporal correlation is further exploited by measurement-guided trajectory attention during the sampling process.  
![](images/f9a003ee6257aa79ae9af7224cc1f263aa7b335c0014e13066a83c735a991d13.jpg)  
Fig. 3. Visual comparison of restoration using (a) DDIM inversion noise and (b) DDIM inversion with adaptive noise blending. Directly using inversion noise for restoration encodes the degradation pattern into restored results, as shown in red boxes in (a).

## A. Overall Framework

The proposed framework aims to solve the video restoration problem while only necessitating a pre-trained image diffusion model without re-tuning for each specific task. As shown in Figure 2, our framework can be divided into three main components: 1) DDIM inversion for initial noise generation, 2) adaptive noise blending for degradation pattern mitigation, and 3) sampling with trajectory attention for temporal consistency enhancement. In contrast to the existing works [1], [23] where the Gaussian noise is adopted as the starting point of diffusion sampling, we initiate the restoration process by obtaining temporal-correlated noises via performing DDIM inversion on the input video frames, denoted as $\mathcal { X } _ { T } ~ = ~ \{ \pmb { x } _ { T } ^ { i } \} _ { i = 1 } ^ { N } =$ $\{ D D I M _ { i n v } ( { \pmb y } ^ { i } ) \} _ { i = 1 } ^ { N }$ . With the merit of DDIM inversion, the temporal correlation across each frame in the input video Y can be encoded into the initial inversion noise $\mathcal { X } _ { T }$ , thus offering effective guidance for the subsequent restoration process. However, as the initial noise $\mathcal { X } _ { T }$ is derived from the input degraded video, the undesired degradation pattern can also be preserved and encoded through the DDIM inversion. Thus, directly performing diffusion sampling starting from X<sub>T</sub> could cause significant artifacts and hinder the restored results, as shown in Figure 3. To address this issue, we propose an adaptive noise blending strategy to mitigate the encoded degradation in initial inversion noise (Section IV-B). Lastly, we further propose a measurement-guided trajectory attention mechanism (Section IV-C) to achieve temporal consistent restoration during diffusion sampling.

## B. Adaptive Noise Blending

In the context of video restoration, the noise is initialized via performing DDIM inversion on the degraded video frames. The inverted noise contains significant degradation patterns and directly adopting it for sampling will lead to sub-optimal restoration results. Figure 4 (a) and (c) present the image and frequency domain of the inverted noises using a degraded frame and corresponding clean frame, respectively. The initial inversion noise contains noticeable jagged patterns with details being smeared out compared to the corresponding ground truth (GT) inversion noise. This discrepancy is particularly pronounced in the frequency domain, where high frequencies are either completely lost or distorted, compelling the restoration process to infer or amplify the missing information. We can notice that the low frequency components of initial inversion noise are relatively accurate compared to the high frequency. Based on this observation, one may trivially adopt a predefined low-pass filter (LPF) to perform filtering on the initial inversion noise, which is common sense. However, video is known for its dynamic content, with significant variations in frequency distribution even between frames of the same video.

![](images/d9e319f25366fd8e7e5150cb858914d93c8b9de6ed7c34ed5fdfca629394fdf3.jpg)  
Fig. 4. A visual example of (a) the initial inversion noise obtained directly from the degraded frame, which exhibits significant degradation patterns and distorted frequency components. With adaptive noise blending, (b) the blended noise increasingly resembles (c) the ground truth inverted noise, both in the image domain and in terms of energy distribution across the frequency domain.

Therefore, directly applying a uniform pre-defined LPF across different frames would yield suboptimal results.

To address this challenge, we introduce an adaptive noise blending strategy that tailors the LPF based on the energy distribution of each degraded frame. Specifically, for a given a degraded frame ${ \pmb x } _ { T } ^ { i }$ , a Gaussian LPF $f ( c ^ { i } )$ is determined by optimizing the cutoff frequency $c ^ { i }$ that retains a fragment of energy of ${ \pmb x } _ { T } ^ { i }$ as follows:

$$
c ^ { i } = \underset { c } { \mathrm { a r g m i n } } \left| \frac { \| D F T _ { 2 D } ( \pmb x _ { T } ^ { i } ) * \pmb f ( c ) \| _ { F } ^ { 2 } } { \| D F T _ { 2 D } ( \pmb x _ { T } ^ { i } ) \| _ { F } ^ { 2 } } - \eta \right| ,\tag{6}
$$

where $D F T _ { 2 D }$ denotes the 2D discrete Fourier transform, ∗ is the Hadarmard product, and $\eta$ represents the proportion of retained energy. This problem can be simply solved as $\| D F T _ { 2 D } ( { \pmb x } _ { T } ^ { i } ) * \pmb { f } ( c ) \| _ { F } ^ { 2 }$ monotonically increasing with c. Consequently, we can progressively increase the value of $c$ and halt the process once c exceeds the energy coefficient $\eta .$ Once the LPF is optimized according to each frame $i ,$ the initial inversion noise is blended with a Gaussian noise as:

$$
\hat { \pmb x } _ { T } ^ { i } = { \pmb f } ( c ^ { i } ) * { \pmb x } _ { T } ^ { i } + ( { \bf 1 } - { \pmb f } ( c ^ { i } ) ) * { \pmb \epsilon } ^ { i } .\tag{7}
$$

Figure 4 (b) provides a visualization of the blended inversion noise, demonstrating greater similarity to the ground truth inversion noise, particularly in terms of energy distribution in the frequency domain.

Why it works? The proposed noise blending strategy aims to adaptively remove the degradation pattern encoded in the initial DDIM inversion noise and provide reliable information for the subsequent restoration process. The merits of this strategy can be categorized into two aspects:

• First, the adaptive noise blending strategy enables a dynamic filtering according to the energy distribution of each frame with a single energy parameter η.

• Besides, this adaptive filtering mitigates artifacts in the inversion noise by preserving the relatively accurate lowfrequency and restoring high-frequency components via Gaussian blending, which yields a more balanced initialization that better matches the diffusion prior.

Algorithm 1 TINDER   
Input: degraded video $\boldsymbol { \ y } ~ = ~ \{ \boldsymbol { y } ^ { i } \} _ { i = 1 } ^ { N }$ , pre-trained image   
diffusion model $\epsilon _ { \theta } .$ , stochasticity strength parameter ζ.   
Initialization: $\pmb { \mathscr { C } } = \{ \mathcal { C } ^ { i } \} _ { i = 1 } ^ { N - 1 } \xleftarrow { - } \{ \pmb { y } ^ { i } , \pmb { y } ^ { i + 1 } \} _ { i = 1 } ^ { N - 1 }$ via (8)   
# Initial noise with DDIM inversion   
$\{ \pmb { x } _ { T } ^ { i } \} _ { i = 1 } ^ { N } = D D I M _ { i n v } ( \{ \pmb { y } ^ { i } \} _ { i = 1 } ^ { N } )$   
# Adaptive noise blending   
$\pmb { f } ( c ^ { i } )  \pmb { x } _ { T } ^ { i }$ via (6)   
$\epsilon ^ { i } \sim \mathcal { N } ( \mathbf { 0 } , \bar { \cal I } )$   
$\hat { \pmb { x } } _ { T } ^ { i } = \pmb { f } ( c ^ { i } ) \ast \pmb { x } _ { T } ^ { i } + ( \mathbf { 1 } - \pmb { f } ( c ^ { i } ) ) \ast \epsilon ^ { i }$   
$\tilde { \mathcal { X } _ { T } } = \{ \hat { \pmb { x } } _ { T } ^ { i } \} _ { i = 1 } ^ { N }$   
# Perform diffusion sampling for restoration   
for $t = T , \dots , 1$ do   
# Tweedie denoising with trajectory guidance   
$\hat { \mathcal { X } } _ { 0 \mid t } = ( \hat { \mathcal { X } } _ { t } - \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon _ { \theta } ( \hat { \mathcal { X } } _ { t } , \pmb { \mathcal { C } } ; t ) ) / \sqrt { \bar { \alpha } _ { t } }$   
# Data consistency   
Update $\hat { \mathcal { X } } _ { 0 \mid t }$ with Eq. (3) or Eq. (4)   
# Renoising   
$\mathbf { \epsilon } \epsilon _ { t } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$   
$\hat { \mathcal { X } } _ { t - 1 } =$   
$\sqrt { \bar { \alpha } _ { t - 1 } } \hat { \mathcal X } _ { 0 \mid t } + \sqrt { 1 - \bar { \alpha } _ { t - 1 } - \zeta ^ { 2 } } \epsilon _ { \theta } ( \hat { \mathcal X } _ { t } , \mathcal C ; t ) + \zeta \epsilon _ { t }$   
end for   
return $\hat { \mathcal { X } } _ { 0 }$

## C. Measurement-guided Trajectory Attention

As pre-trained image diffusion models address the restoration on a frame-by-frame basis, it is crucial to impose temporal correlation during the restoration process. To further guarantee temporal consistency, we propose a measurementguided trajectory attention (MGTA) mechanism that can be seamlessly integrated into pre-trained image diffusion models. The process begins by extracting temporal information via estimating the correspondence maps between adjacent frames in the degraded video. Given a pair of adjacent frames $y ^ { i }$ and $y ^ { i + 1 }$ , let $\phi ( \boldsymbol { y } ^ { i } [ \boldsymbol { p } ^ { i } ] )$ denotes the patch centered at the pixel $p ^ { i }$ in the frame $y ^ { i }$ , the corresponding matching point $q ^ { i + \bar { 1 } }$ in the next frame $y ^ { i + 1 }$ is searched as:

$$
q ^ { i + 1 } = \underset { q } { \operatorname { a r g m i n } } \| \phi ( \pmb { y } ^ { i + 1 } [ q ] ) - \phi ( \pmb { y } ^ { i } [ p ^ { i } ] ) \| _ { F } ^ { 2 } .\tag{8}
$$

Therefore, the correspondence map $\begin{array} { l c l } { { { \mathcal C } ^ { i } } } & { { = } } & { { \{ p ^ { i } , q ^ { i + 1 } \} _ { i = 1 } ^ { N - 1 } } } \end{array}$ is compiled by collecting these matched points. To utilize the correspondence map to guide the attention module, we downsampled it to the resolution of each latent space.<sup>1</sup> Inspired by the previous works [24], [25], [50], we extend the spatial self-attention module in the diffusion model to the spatialtemporal domain. Besides, temporal consistency is reinforced with the aid of correspondence map ${ \mathcal { C } } ^ { i }$ as:

$$
\begin{array} { c } { { Q = W ^ { Q } z ^ { i } , } } \\ { { K = W ^ { K } [ z ^ { i } , { \mathcal T } ( z ^ { i + 1 } ; { \mathcal C } ^ { i } ) ] , V = W ^ { V } [ z ^ { i } ; { \mathcal T } ( z ^ { i + 1 } ; { \mathcal C } ^ { i } ) ] , } } \\ { { \mathrm { A t t n } ( Q , K , V ) = \mathrm { S o f t m a x } ( \frac { Q K ^ { \top } } { \sqrt { d } } ) V , } } \end{array}\tag{9}
$$

TABLE I  
QUANTITATIVE EVALUATION RESULTS (PSNR, SSIM, LPIPS, AND FVD) OF VIDEO RESTORATION TASKS: Gaussian DEBLUR, SUPER-RESOLUTION (2× AND 4×) AND 70% INPAINTING ON DAVIS [46], REDS4 [47], AND SET8 [48], RESPECTIVELY. THE BEST RESULTS ARE MARKED IN BOLD.
<table><tr><td></td><td colspan="3">DAVIS [46]</td><td colspan="3">REDS4 [47]</td><td colspan="4">Set8 [48]</td></tr><tr><td>Method</td><td>PSNR↑</td><td>SSIM↑ LPIPS↓</td><td>FVD↓</td><td>PSNR↑</td><td>SSIM↑ LPIPS↓</td><td>FVD↓</td><td>PSNR↑</td><td>SSIM↑ LPIPS↓</td><td></td><td>FVD↓</td></tr><tr><td>DDS [45] + independent</td><td>27.93</td><td>0.8198 0.1870</td><td>69.08</td><td>25.29</td><td>0.7789 0.2218</td><td>59.33</td><td>27.06</td><td>0.7949</td><td>0.2102</td><td>83.60</td></tr><tr><td>DDS [45] + mix [49]</td><td>27.99</td><td>0.8226 0.1862</td><td>69.25</td><td>25.27</td><td>0.7790 0.2215</td><td>60.79</td><td>27.07</td><td>0.7951</td><td>0.2107</td><td>83.52</td></tr><tr><td>Dbblur DDS [45] + prog [49]</td><td>27.99</td><td>0.8220 0.1869</td><td>70.38</td><td>25.21</td><td>0.7768 0.2219</td><td>62.49</td><td>27.19</td><td>0.8000</td><td>0.2073</td><td>82.92</td></tr><tr><td>Kwon et al. [1] (DDS+identical)</td><td>27.90</td><td>0.8185 0.1873</td><td>72.26</td><td>25.25</td><td>0.7787 0.2195</td><td>65.31</td><td>26.91</td><td>0.7849</td><td>0.2129</td><td>88.22</td></tr><tr><td>Ours (DDS)</td><td>29.43</td><td>0.8608 0.1763</td><td>63.42</td><td>27.53</td><td>0.8357 0.2142</td><td>58.85</td><td>29.10</td><td>0.8547</td><td>0.1909</td><td>81.06</td></tr><tr><td>DDS [45] + independent</td><td>29.74</td><td>0.8790 0.0996</td><td>21.13</td><td>26.66</td><td>0.8456 0.1265</td><td>16.18</td><td>29.45</td><td>0.8670</td><td>0.1045</td><td>22.36</td></tr><tr><td>(×2) DDS [45] + mix [49]</td><td>29.75</td><td>0.8788 0.0994</td><td>22.64</td><td>26.64</td><td>0.8454 0.1263</td><td>17.63</td><td>29.42</td><td>0.8670</td><td>0.1048</td><td>21.45</td></tr><tr><td>DDS [45] + prog [49]</td><td>29.78</td><td>0.8799 0.0990</td><td>23.95</td><td>26.62</td><td>0.8446 0.1269</td><td>20.59</td><td>29.47</td><td>0.8682</td><td>0.1039</td><td>27.25</td></tr><tr><td>SR Kwon et al. [1] (DDS+identical)</td><td>29.76</td><td>0.8789 0.0998</td><td>25.21</td><td>26.64</td><td>0.8444 0.1274</td><td>22.05</td><td>29.46</td><td>0.8688</td><td>0.1045</td><td>25.53</td></tr><tr><td>Ours (DDS)</td><td>31.92</td><td>0.9204 0.1032</td><td>18.48</td><td>29.45</td><td>0.9044 0.1363</td><td>14.29</td><td>31.87</td><td>0.9193</td><td>0.1147</td><td>20.63</td></tr><tr><td>DDS [45] + independent</td><td>26.89</td><td>0.7810 0.2537</td><td>167.12</td><td>24.61</td><td>0.7383 0.3076</td><td>149.21</td><td>26.07</td><td>0.7612</td><td>0.2914</td><td>179.26</td></tr><tr><td>(×t) DDS [45] + mix [49]</td><td>26.90</td><td>0.7811 0.2538</td><td>164.53</td><td>24.62</td><td>0.7386 0.3073</td><td>144.76</td><td>26.05</td><td>0.7613</td><td>0.2911</td><td>182.51</td></tr><tr><td>DDS [45] + prog [49]</td><td>26.88</td><td>0.7806 0.2533</td><td>160.62</td><td>24.61</td><td>0.7383 0.3077</td><td>138.26</td><td>26.07</td><td>0.7614</td><td>0.2913</td><td>170.55</td></tr><tr><td>SR Kwon et al. [1] (DDS+identical)</td><td>26.89</td><td>0.7806 0.2538</td><td>155.58</td><td>24.62</td><td>0.7384 0.3079</td><td>137.58</td><td>26.08</td><td>0.7617</td><td>0.2913</td><td>176.50</td></tr><tr><td>Ours (DDS)</td><td>27.09</td><td>0.7872 0.2701</td><td>258.18</td><td>24.91</td><td>0.7468 0.3325</td><td>233.21</td><td>26.27</td><td>0.7688</td><td>0.3055</td><td>294.58</td></tr><tr><td>DDS [45] + independent</td><td>25.86</td><td>0.7806 0.2338</td><td>240.72</td><td>23.77</td><td>0.7473 0.2557</td><td>287.74</td><td></td><td>24.34</td><td>0.7388 0.2794</td><td>541.28</td></tr><tr><td>DDS [45] + mix [49]</td><td>25.85</td><td>0.7804 0.2339</td><td>257.64</td><td>23.80</td><td>0.7481 0.2552</td><td>293.65</td><td></td><td>24.33</td><td>0.7388 0.2795</td><td>552.04</td></tr><tr><td>Inpnng DDS [45] + prog [49]</td><td>25.82</td><td>0.7798 0.2347</td><td>206.10</td><td>23.78</td><td>0.7474 0.2553</td><td>205.76</td><td></td><td>24.33</td><td>0.7388 0.2788</td><td>322.22</td></tr><tr><td>Kwon et al. [1] (DDS+identical)</td><td>25.85</td><td>0.7799 0.2348</td><td>191.36</td><td>23.76</td><td>0.7466 0.2563</td><td>187.25</td><td></td><td>24.34</td><td>0.7389 0.2791</td><td>258.08</td></tr><tr><td>Ours (DDS)</td><td>27.08</td><td>0.8075 0.2349</td><td>270.70</td><td>25.46</td><td>0.7880</td><td>0.2551</td><td>267.00</td><td>25.77</td><td>0.7743</td><td>0.2772530.15</td></tr></table>

where [·] denotes the concatenation, $z ^ { i }$ is the feature of the ith frame and $\tau$ transform that leverages trajectory maps ${ \mathcal { C } } ^ { i }$ to align the feature $z ^ { i + 1 }$ with those of the current frame. The latent features z are refined through MGTA to mitigate the adverse effects of aggregating irrelevant patches in dense spatial-temporal attention. Following [24], we perform MGTA in the decoder layers of the UNet during sampling.

Overall, our method provides a flexible framework for video restoration where the data consistency can be directly guaranteed by plug-in existing image-based diffusion sampling strategies as discussed in Section III-B. The overall steps of our framework are described in Algorithm 1.

## V. EXPERIMENTS

## A. Experimental Setup

Datasets and tasks. To evaluate the efficacy of the proposed method, we conducted comprehensive experiments across three video datasets: DAVIS [46], REDS4 [47], and Set8 [48]. For each dataset, we sampled 100 16-frame video clips as the ground truth for evaluation. To demonstrate the generalization capability of our method, we conduct three representative video restoration tasks including Gaussian deblur, super-resolution (SR), and inpainting. Following [1], [18], the degradation is performed using $2 \times$ and 4× bicubic downsampling for SR. For Gaussian deblurring, a blur kernel of sized 13×13 with a standard deviation of 2.0 is employed. For video inpainting, we randomly drop 70% pixels in each video frame.

Metrics. For video restoration tasks, evaluating both distortion and perceptual quality is crucial. Thus, we employ the Peak

TABLE II  
HYPER-PARAMTERS SETTING FOR EACH TASK.
<table><tr><td>Datasets</td><td></td><td>DAVIS</td><td>REDS4</td><td>Set8</td></tr><tr><td>Task</td><td> $T _ { i n v }$  T</td><td>η</td><td>η</td><td>η</td></tr><tr><td>Deblur (Gaussian)</td><td>20 20</td><td>0.5</td><td>0.7</td><td>0.7</td></tr><tr><td>SR (2×)</td><td>20 20</td><td>0.7</td><td>0.7</td><td>0.7</td></tr><tr><td>SR (4×)</td><td>20 50</td><td>0.1</td><td>0.1</td><td>0.1</td></tr><tr><td>Inpaint (10%)</td><td>20 50</td><td>0.1</td><td>0.1</td><td>0.1</td></tr></table>

Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM) to assess the fidelity of the restored videos. For perceptual quality, we use the Learned Perceptual Image Patch Similarity (LPIPS) distance [51]. Additionally, a video-specific metric, namely Frechet Video Distance (FVD) [52] is utilized´ for comprehensive evaluation.

## B. Implementation Details

The proposed method is implemented on PyTorch using NVIDIA RTX A5000. The total iterations for the DDIM inversion are fixed at $T _ { i n v } ~ = ~ 2 0$ for each task and implementation. For tasks such as Gaussian deblurring and $2 \times \mathrm { ~ S R ~ }$ the sampling iteration step is set to $T = 2 0$ . For the more challenging tasks of 4× SR and 70% inpainting, a higher iteration of $T = 5 0$ is required.

The proposed method and all competing baselines necessitate a pre-trained image diffusion as the prior. To conduct a fair comparison, we adopt the same pre-trained diffusion model sourced from [30] for all experiments. The hyper-parameter setting is listed in Table II. The parameter η controls the energy partition preserved during the adaptive noise blending. For a task with less degradation severity, the η can be relatively high, e.g., around 0.7, indicating that we can preserve more energy in the DDIM inversion noise to maintain the correlation across each frame. In contrast, for those tasks with severe degradation, e.g., 4× SR and 70% inpainting, the η should be relatively low (η = 0.1) since the inverted noise contains significant degradation. Therefore, a smaller η indicating more aggressive filtering is required to remove the severe degradation pattern encoded in the inversion noises.

TABLE III  
QUANTITATIVE EVALUATION RESULTS (PSNR, SSIM, LPIPS, AND FVD) OF VIDEO Gaussian DEBLUR ON DAVIS [46], REDS4 [47], AND SET8 [48], RESPECTIVELY. THE BEST RESULTS ARE MARKED IN BOLD.
<table><tr><td>Method PSNR↑ SSIM↑ LPIPS↓ FVD↓</td></tr><tr><td>[46] DiffPIR [18] + independent 30.51 0.8706 0.1742 37.53 DiffPIR [18] + mix [49] 30.52 0.8707 0.1741 36.84 DiffPIR [18] + prog [49] 30.50 0.8705 0.1744 39.74</td></tr><tr><td>DiffPIR [18] + identical [1] 30.52 0.8709 0.1739 38.70 Ours (DiffPIR) 30.54 0.8757 0.1621 32.07</td></tr><tr><td>DiffPIR [18] + independent 28.61 0.8662 0.1939 23.83 14 DiffPIR [18] + mix [49] 28.62 0.8661 0.1941 24.47 REEDS4 DiffPIR [18] + prog [49] 28.61 0.8663 0.1939 23.87 DiffPIR [18] + identical [1] 28.60 0.8661 0.1941 23.21</td></tr><tr><td>Ours (DiffPIR) 28.59 0.8653 0.1887 22.96</td></tr><tr><td>DiffPIR [18] + independent 30.33 0.8811 0.1725 33.85 DiffPIR [18] + mix [49] 30.32 0.8810 0.1725 34.46</td></tr><tr><td>[Se] t8] DiffPIR [18] + prog [49] 30.32 0.8809 0.1727 34.21 DiffPIR [18] + identical [1] 30.32 0.8809 0.1723 33.74 Ours (DiffPIR) 30.17 0.8787 0.1616 33.64</td></tr></table>

## C. Comparison with State-of-the-Art

To evaluate the performance of our proposed method, we conduct a comparison with several state-of-the-art zero-shot video restoration methods. Since zero-shot video restoration is an under-explored yet challenging area, there are limited works available to compare <sup>2</sup>. Kwon et al. [1] is the most related work in this task, where the method adopts an identical Gaussian noise as the initiation and utilizes DDS [45] to ensure data consistency during the diffusion sampling process. To conduct a comprehensive evaluation, we constructed various competing baselines that utilized different noise initial strategies, namely frame-wise independent noise, mix noise model [49], progressive noise model [49]. To conduct a fair comparison, all of these baselines including our method adopt the same sampling strategy for data consistency, namely DDS [45].

Table I presents the quantitative evaluation results of our proposed method compared to several state-of-the-art approaches, including Gaussian deblur, super-resolution (2× and 4×), and inpainting tasks on the DAVIS, REDS4, and Set8 datasets. For Gaussian deblurring, our approach achieves the highest PSNR and SSIM values with lower LPIPS and FVD compared to other methods, indicating better detail preservation and perceptual quality. In the super-resolution tasks, our method shows significant improvement for the 2× scaling factor, particularly in PSNR and SSIM. However, for the more challenging 4× super-resolution, some results are less consistent, though our method still maintains competitive perceptual quality. For video inpainting, our method achieves superior results in PSNR and SSIM on all datasets. Overall, the results in Table 1 demonstrate that our method generally outperforms or is highly competitive with existing zero-shot video restoration techniques, highlighting its robustness and generalization ability. To further demonstrate the advantage of our method against other competing baselines, Figure 5 provides qualitative comparisons on various video restoration tasks. Existing methods only adopt a simple noise initiation strategy, leading to significant speckling artifacts across restored frames. With the merits of the proposed framework, our method consistently aligns closely with the ground truth across all tasks, suggesting robustness that handles various types of image restoration effectively.

TABLE IV  
QUANTITATIVE COMPARISON OF THE RESTORATION STABILITY.
<table><tr><td>Method</td><td>PSNR↑ SSIM↑</td><td>LPIPS↓</td></tr><tr><td></td><td>Kwon et al. [1] 27.92±0.0912 0.8194±0.0038 0.1875±0.0017</td><td></td></tr><tr><td>DVVIS Ours</td><td>29.43±0.0020 0.8608±0.0001 0.1764±0.0001</td><td></td></tr><tr><td></td><td>Kwon et al. [1] 25.31±0.1002 0.7801±0.0038 0.2211±0.0014</td><td></td></tr><tr><td>REDS4 Ours</td><td>27.53±0.0012 0.8357±0.0001 0.2141±0.0001</td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td>Set8 Ours</td><td>Kwon et al. [1] 27.10±0.1267 0.7957±0.0056 0.2092±0.0034</td><td>29.10±0.0013 0.8547±0.0001 0.1909±0.0001</td></tr></table>

## D. Extension to Other Sampling Strategies

As described in Algorithm 1, our framework can easily extend existing image-based diffusion sampling strategies to solve video restoration problems with data consistency. To further demonstrate the generalizability and robustness of the proposed method, we conduct extensive experiments by plugin DiffPIR [18] into our framework on various video datasets. Following the previous section, we construct competing baseline using DiffPIR combined with frame-wise independent noise, mix noise model [49], progressive noise model [49] and frame-wise identical noise [1]. Table III presents the quantitative results using DiffPIR sampling. The proposed method consistently outperforms or remains highly competitive with other approaches across various restoration tasks on all datasets, with particular strength in the LPIPS and FVD.

## E. Restoration Stability

To evaluate the robustness of our proposed method, we analyze the restoration stability of our method compared to competing method [1] across three datasets. As shown in Table IV, our method shows superior stability across all datasets, with lower standard deviations in all metrics. For example, on the DAVIS dataset, we not only improve the PSNR from 27.92 to 29.43 but also reduce its standard deviation from ±0.0912 to ±0.0020. Similar trends are observed for SSIM and LPIPS on the DAVIS dataset, as well as for all three metrics on the REDS4 and Set8 datasets. These results validate that our noise initialization strategy not only enhances restoration quality but also ensures more reliable and consistent performance.

![](images/eafe7f7f468c330135bacc30aa5e4a6780a2f9489f27d0e7963a687902c4b042.jpg)  
Fig. 5. Example of various video restoration results including Gaussian deblur, super-resolution, and inpainting. Please zoom in for better visual details

TABLE V  
ABLATION STUDY OF EACH PROPOSED COMPONENT ON Gaussian DEBLUR ON THE REDS4 [48] DATASET.
<table><tr><td>DDIM Inv.</td><td>ANB</td><td>MGTA</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>FVD↓</td></tr><tr><td>x</td><td>x</td><td>x</td><td>25.29</td><td>0.7789</td><td>0.2218</td><td>59.33</td></tr><tr><td>√</td><td>x</td><td>x</td><td>27.07</td><td>0.8372</td><td>0.2679</td><td>71.64</td></tr><tr><td>√</td><td>x</td><td>√</td><td>27.47</td><td>0.8487</td><td>0.2516</td><td>82.09</td></tr><tr><td>√</td><td>√</td><td>x</td><td>26.98</td><td>0.8196</td><td>0.2367</td><td>93.11</td></tr><tr><td></td><td>√</td><td>√</td><td>27.53</td><td>0.8357</td><td>0.2142</td><td>58.85</td></tr></table>

## F. Ablation Study

To investigate the efficacy of each key component of the proposed method, we perform experiments on several variants on the REDS4 dataset. The ablation comparison starts with a base model employing independent noise initiation alongside DDS sampling. We incrementally integrated additional components into the framework, specifically DDIM Inversion, Adaptive Noise Blending (ANB), and Measurement-Guided Trajectory Attention (MGTA). The results in Table V indicate that using all components together provides the best balance across all metrics, achieving notably low LPIPS and FVD scores, indicative of superior perceptual quality and visual accuracy. Specifically, utilizing DDIM inversion for noise initiation can effectively improve the PSNR while also introducing unnatural patterns that distort the perceptual quality. However, with the aid of the proposed modules, such drawbacks can be significantly mitigated, achieving promising performance in both fidelity and visual quality. Figure 6 illustrates a visual comparison on Gaussian deblur on the DAIVIS dataset where each key component of our method is removed. It clearly demonstrates that without these components, the restoration results suffer from pronounced degradation and noticeable blurring artifacts.

![](images/e35a9a0d65dded044e7727d569524a5dee49f129b95924fd4434c6fc6f113588.jpg)  
Fig. 6. Visualization of ablation study by removing each key component of Gaussian deblur on the DAVIS dataset.

TABLE VI  
COMPARISON OF USING A FIXED GAUSSIAN LOW-PASS FILTER FOR NOISE BLENDING. THE BEST RESULTS ARE MARKED IN BOLD.
<table><tr><td></td><td colspan="3">Fixed Gaussian Low-pass Filter</td><td>Ours</td></tr><tr><td>cut-off freq.</td><td>0.1 0.3</td><td>0.5 0.7</td><td>0.9</td><td>|adaptive</td></tr><tr><td>PSNR↑</td><td>27.50 29.39</td><td>28.81 29.32</td><td>29.37</td><td>29.43</td></tr><tr><td>SSIM↑</td><td></td><td>0.8039 0.8400 0.7996 0.8442 0.8567</td><td></td><td>0.8609</td></tr><tr><td>LPIPS↓</td><td></td><td>0.2093 0.1847 0.2022 0.1839 0.1978</td><td></td><td>0.1764</td></tr></table>

To validate the superiority of our proposed adaptive noise blending strategy, we construct a comparison using a set of fixed Gaussian LPFs with different cut-off frequencies and use them for noise blending. Concretely, for each frame, we replace the adaptive filter f(c<sub>i</sub>) with a frame-invariant f(c), where $c \in \{ 0 . 1 , 0 . 3 , 0 . 5 , 0 . 7 , 0 . 9 \}$ , and keep all other settings (DDIM inversion, sampling steps, and data-consistency updates) unchanged. Table VI reports results on video Gaussian deblurring. We observe that while fixed LPFs can improve over directly using the raw inversion noise, our adaptive filtering achieves the best overall fidelity and perceptual quality (PSNR/SSIM and LPIPS). This is consistent with our motivation: applying a single pre-defined cut-off frequency to all frames is suboptimal when the frequency distribution varies across frames, since overly aggressive filtering can suppress useful structures for detail-rich frames, whereas overly mild filtering retains more degradation-corrupted components for severely degraded frames.

We further analyze the key hyper-parameter η, which controls the proportion of frequency energy retained by the adaptive filtering. We evaluate $\eta \in \{ 0 . 1 , 0 . 3 , 0 . 5 , 0 . 7 , 0 . 9 \}$ on three datasets (DAVIS, REDS4, and Set8) under Gaussian deblurring, using the same experimental settings as the main experiments. As shown in Table VII, moderate values of η (e.g., 0.5–0.7) generally provide the best performance, while extreme choices lead to degradation: small η tends to overfilter and removes too much informative content from the initialization, making subsequent diffusion sampling harder to recover details, while large η retains more high-frequency components from the inversion noise, which may include distorted or degradation-encoded patterns, reducing perceptual quality.

TABLE VII  
ABLATION STUDY OF THE RETAINED ENERGY PROPORTION (η) IN ADAPTIVE FILTERING STRATEGY ON THE VIDEO Gaussian DEBLUR ON THREE DIFFERENT DATASETS DAVIS, REDS4, AND SET8. THE BEST RESULTS ARE MARKED IN BOLD. THE GRAY CELLS INDICATES THE SELECTED PARAMETER FOR THE MAIN EXPERIMENTS.
<table><tr><td></td><td>DAVIS</td><td>REDS4</td><td>Set8</td></tr><tr><td>η</td><td>PSNR↑ SSIM↑ LPIPS↓</td><td></td><td>|PSNR↑ SSIM↑ LPIPS↓PSNR↑ SSIM↑ LPIPS↓</td></tr><tr><td>0.1</td><td>27.97 0.8242 0.1995</td><td>25.65 0.7868 0.2412</td><td>27.46 0.8215 0.2133</td></tr><tr><td>0.3</td><td>29.04 0.8578 0.1877</td><td>26.83 0.8329 0.2308</td><td>28.67 0.8583 0.2057</td></tr><tr><td>0.5</td><td>29.43 0.8607 0.1765</td><td>27.32 0.8431 0.2221</td><td>29.02 0.8632 0.1985</td></tr><tr><td>0.7</td><td>29.48 0.84670.1788</td><td>27.53 0.8358 0.2139</td><td>29.10 0.8547 0.1907</td></tr><tr><td>0.9</td><td>28.85 0.8018 0.2018</td><td>27.24 0.7958 0.2352</td><td>28.43 0.8092 0.2031</td></tr></table>

To further explore the effectiveness of the proposed measurement-guided trajectory attention mechanism, we conducted additional ablation studies. Unlike existing methods in video editing [24]–[26] which apply spatial-temporal attention during both the inversion and sampling phases, our studies suggest that employing the proposed attention mechanism exclusively during the sampling process enhances restoration performance. We conduct an ablation study where the DDIM inversion is performed with and without the proposed attention mechanism. Table VIII shows the results of the aforementioned two variants on Gaussian deblur on three different datasets. These findings indicate that using the measurementguided trajectory attention during the inversion process actually hinder the performance, as it reinforces degradation patterns within the frames. As for video editing, the DDIM inversion is performed on the clean video frames, thus leveraging the spatial-temporal attention will benefit the temporal correlation. In contrast, in the context of video restoration, the inversion process is performed on the degraded frames. Consequently, imposing such a temporal-related attention mechanism during the DDIM inverse process will also exploit the correlation across the degradation pattern in each frame, thus hindering restoration performance.

TABLE VIII  
ABLATION STUDY OF INCORPORATING MGTA DURING THE INVERSION PROCESS.
<table><tr><td>Method</td><td></td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>FVD↓</td></tr><tr><td>DAVIS</td><td>Inversion w/ MGTA</td><td>29.36</td><td>0.8594</td><td>0.1794</td><td>69.15</td></tr><tr><td></td><td>Inversion w/o MGTA</td><td>29.43</td><td>0.8608</td><td>0.1763</td><td>63.42</td></tr><tr><td>REDSA4</td><td>Inversion w/ MGTA</td><td>27.29</td><td>0.8250</td><td>0.2302</td><td>74.05</td></tr><tr><td></td><td>Inversion w/o MGTA</td><td>27.53</td><td>0.8357</td><td>0.2142</td><td>58.85</td></tr><tr><td rowspan="2">Set8</td><td>Inversion w/ MGTA</td><td>28.69</td><td>0.8389</td><td>0.2108</td><td>98.29</td></tr><tr><td>Inversion w/o MGTA</td><td>29.10</td><td>0.8547</td><td>0.1909</td><td>81.06</td></tr></table>

## G. Limitations

Our method improves zero-shot restoration quality and stability through enhanced noise initialization and trajectoryguided temporal attention. However, it also introduces additional computational overhead. Compared with the previous method [45], our current implementation is slower in wallclock time under the same experimental setting. The previous method [45] takes 9.67 s per video, whereas our full pipeline takes 20.18 s per video. This increase is mainly due to the proposed components required by our framework, including DDIM inversion for obtaining the initial noise, as well as extra operations such as adaptive noise blending and trajectory attention. Specifically, the measurement-guided trajectory attention introduces a modest overhead in both runtime and memory, with around +1.94 s runtime and +500 MB memory overhead. Future work includes accelerating the pipeline by reducing the number of sampling steps, adopting faster initialization strategies, caching or reusing trajectories, and exploring acceleration techniques to obtain a faster feed-forward variant while preserving the proposed stability benefits.

## VI. CONCLUSION

In this paper, we present a novel zero-shot video restoration framework that only leverages a pre-trained image diffusion model. We initiate by obtaining correlated noise via DDIM inversion, which can effectively capture and utilize the inherent temporal correlation in the degraded video frames. Based on that, we propose an adaptive noise blending strategy to remove the degradation pattern encoded during the inversion process while steering the noise toward the Gaussian distribution to enhance the generation ability. We further propose a measurement-guided trajectory attention to emphasize the temporal consistency during the restoration process.

Finally, extensive experiments on various video restoration tasks demonstrate the superiority of the proposed methods on DAVIS, REDS4 and Set8 datasets.

## REFERENCES

[1] T. Kwon and J. C. Ye, “Solving video inverse problems using image diffusion models,” arXiv preprint arXiv:2409.02574, 2024.

[2] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in neural information processing systems, vol. 33, pp. 6840– 6851, 2020.

[3] J. Ho, C. Saharia, W. Chan, D. J. Fleet, M. Norouzi, and T. Salimans, “Cascaded diffusion models for high fidelity image generation.” J. Mach. Learn. Res., vol. 23, no. 47, pp. 1–33, 2022.

[4] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “Highresolution image synthesis with latent diffusion models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 10 684–10 695.

[5] Z. Xing, Q. Feng, H. Chen, Q. Dai, H. Hu, H. Xu, Z. Wu, and Y.-G. Jiang, “A survey on video diffusion models,” ACM Computing Surveys, 2023.

[6] W. Harvey, S. Naderiparizi, V. Masrani, C. Weilbach, and F. Wood, “Flexible diffusion modeling of long videos,” Advances in Neural Information Processing Systems, vol. 35, pp. 27 953–27 965, 2022.

[7] J. Ho, W. Chan, C. Saharia, J. Whang, R. Gao, A. Gritsenko, D. P. Kingma, B. Poole, M. Norouzi, D. J. Fleet et al., “Imagen video: High definition video generation with diffusion models,” arXiv preprint arXiv:2210.02303, 2022.

[8] Y. Jiang, Q. Liu, D. Chen, L. Yuan, and Y. Fu, “Animediff: Customized image generation of anime characters using diffusion model,” IEEE Transactions on Multimedia, 2024.

[9] Y. Xu, X. Xu, H. Gao, and F. Xiao, “Sgdm: An adaptive styleguided diffusion model for personalized text to image generation,” IEEE Transactions on Multimedia, 2024.

[10] S. Gao, X. Liu, B. Zeng, S. Xu, Y. Li, X. Luo, J. Liu, X. Zhen, and B. Zhang, “Implicit diffusion models for continuous super-resolution,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 10 021–10 030.

[11] M. Ren, M. Delbracio, H. Talebi, G. Gerig, and P. Milanfar, “Multiscale structure guided diffusion for image deblurring,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 10 721–10 733.

[12] L. Guo, C. Wang, W. Yang, Y. Wang, and B. Wen, “Boundary-aware divide and conquer: A diffusion-based solution for unsupervised shadow removal,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 13 045–13 054.

[13] L. Guo, C. Wang, W. Yang, S. Huang, Y. Wang, H. Pfister, and B. Wen, “Shadowdiffusion: When degradation prior meets diffusion model for shadow removal,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 14 049–14 058.

[14] A. Lugmayr, M. Danelljan, A. Romero, F. Yu, R. Timofte, and L. Van Gool, “Repaint: Inpainting using denoising diffusion probabilistic models,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 11 461–11 471.

[15] S. Zhou, P. Yang, J. Wang, Y. Luo, and C. C. Loy, “Upscale-avideo: Temporal-consistent diffusion model for real-world video superresolution,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 2535–2545.

[16] X. Yang, C. He, J. Ma, and L. Zhang, “Motion-guided latent diffusion for temporally consistent real-world video super-resolution,” in European Conference on Computer Vision. Springer, 2025, pp. 224–242.

[17] C. Zhang, W. Yang, X. Li, and H. Han, “Mmginpainting: Multi-modality guided image inpainting based on diffusion models,” IEEE Transactions on Multimedia, 2024.

[18] Y. Zhu, K. Zhang, J. Liang, J. Cao, B. Wen, R. Timofte, and L. Van Gool, “Denoising diffusion models for plug-and-play image restoration,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 1219–1229.

[19] H. Chung, J. Kim, M. T. Mccann, M. L. Klasky, and J. C. Ye, “Diffusion posterior sampling for general noisy inverse problems,” in The Eleventh International Conference on Learning Representations, ICLR 2023. The International Conference on Learning Representations, 2023.

[20] H. Chung, B. Sim, D. Ryu, and J. C. Ye, “Improving diffusion models for inverse problems using manifold constraints,” Advances in Neural Information Processing Systems, vol. 35, pp. 25 683–25 696, 2022.

[21] Y. Wang, J. Yu, and J. Zhang, “Zero-shot image restoration using denoising diffusion null-space model,” The Eleventh International Conference on Learning Representations, 2023.

[22] B. Song, S. M. Kwon, Z. Zhang, X. Hu, Q. Qu, and L. Shen, “Solving inverse problems with latent diffusion models via hard data consistency,” in The Twelfth International Conference on Learning Representations, 2024. [Online]. Available: https://openreview.net/ forum?id=j8hdRqOUhN

[23] C. Cao, H. Yue, X. Liu, and J. Yang, “Zero-shot video restoration and enhancement using pre-trained image diffusion model,” arXiv preprint arXiv:2407.01960, 2024.

[24] D. Ceylan, C.-H. P. Huang, and N. J. Mitra, “Pix2video: Video editing using image diffusion,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 23 206–23 217.

[25] C. Qi, X. Cun, Y. Zhang, C. Lei, X. Wang, Y. Shan, and Q. Chen, “Fatezero: Fusing attentions for zero-shot text-based video editing,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 15 932–15 942.

[26] M. Geyer, O. Bar-Tal, S. Bagon, and T. Dekel, “Tokenflow: Consistent diffusion features for consistent video editing,” arXiv preprint arXiv:2307.10373, 2023.

[27] J. Wang, Y. Ma, J. Guo, Y. Xiao, G. Huang, and X. Li, “Cove: Unleashing the diffusion feature correspondence for consistent video editing,” arXiv preprint arXiv:2406.08850, 2024.

[28] J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli, “Deep unsupervised learning using nonequilibrium thermodynamics,” in International Conference on Machine Learning. PMLR, 2015, pp. 2256–2265.

[29] A. Q. Nichol and P. Dhariwal, “Improved denoising diffusion probabilistic models,” in International Conference on Machine Learning. PMLR, 2021, pp. 8162–8171.

[30] P. Dhariwal and A. Nichol, “Diffusion models beat gans on image synthesis,” Advances in Neural Information Processing Systems, vol. 34, pp. 8780–8794, 2021.

[31] J. Ho, T. Salimans, A. Gritsenko, W. Chan, M. Norouzi, and D. J. Fleet, “Video diffusion models,” Advances in Neural Information Processing Systems, vol. 35, pp. 8633–8646, 2022.

[32] Z. Chen, F. Long, Z. Qiu, T. Yao, W. Zhou, J. Luo, and T. Mei, “Learning spatial adaptation and temporal coherence in diffusion models for video super-resolution,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 9232–9241.

[33] X. Wang, K. C. Chan, K. Yu, C. Dong, and C. Change Loy, “Edvr: Video restoration with enhanced deformable convolutional networks,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops, 2019, pp. 0–0.

[34] K. Zhou, W. Li, L. Lu, X. Han, and J. Lu, “Revisiting temporal alignment for video restoration,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 6053–6062.

[35] T. H. Kim, M. S. Sajjadi, M. Hirsch, and B. Scholkopf, “Spatiotemporal transformer network for video restoration,” in Proceedings of the European conference on computer vision (ECCV), 2018, pp. 106– 122.

[36] H. Cheng, Y. Guo, J. Yin, H. Chen, J. Wang, and L. Nie, “Audio-driven talking video frame restoration,” IEEE Transactions on Multimedia, 2021.

[37] Z. Chen, H. Wang, and C. W. Chen, “Self-supervised video representation learning by serial restoration with elastic complexity,” IEEE Transactions on Multimedia, 2023.

[38] A. A. Baniya, T.-K. Lee, P. W. Eklund, and S. Aryal, “Omnidirectional video super-resolution using deep learning,” IEEE Transactions on Multimedia, vol. 26, pp. 540–554, 2023.

[39] Q. Ding, L. Shen, L. Yu, H. Yang, and M. Xu, “Blind quality enhancement for compressed video,” IEEE Transactions on Multimedia, 2023.

[40] J. Liang, J. Cao, Y. Fan, K. Zhang, R. Ranjan, Y. Li, R. Timofte, and L. Van Gool, “Vrt: A video restoration transformer,” IEEE Transactions on Image Processing, vol. 33, pp. 2171–2182, 2024.

[41] J. Liang, Y. Fan, X. Xiang, R. Ranjan, E. Ilg, S. Green, J. Cao, K. Zhang, R. Timofte, and L. V. Gool, “Recurrent video restoration transformer with guided deformable attention,” Advances in Neural Information Processing Systems, vol. 35, pp. 378–393, 2022.

[42] T. Garber and T. Tirer, “Image restoration by denoising diffusion models with iteratively preconditioned guidance,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 25 245–25 254.

[43] C.-H. Yeh, C.-Y. Lin, Z. Wang, C.-W. Hsiao, T.-H. Chen, H.-S. Shiu, and Y.-L. Liu, “Diffir2vr-zero: Zero-shot video restoration with diffusion-

based image restoration models,” arXiv preprint arXiv:2407.01519, 2024.

[44] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” arXiv preprint arXiv:2010.02502, 2020.

[45] H. Chung, S. Lee, and J. C. Ye, “Decomposed diffusion sampler for accelerating large-scale inverse problems,” arXiv preprint arXiv:2303.05754, 2023.

[46] J. Pont-Tuset, F. Perazzi, S. Caelles, P. Arbelaez, A. Sorkine-Hornung,´ and L. Van Gool, “The 2017 davis challenge on video object segmentation,” arXiv preprint arXiv:1704.00675, 2017.

[47] S. Nah, S. Baik, S. Hong, G. Moon, S. Son, R. Timofte, and K. Mu Lee, “Ntire 2019 challenge on video deblurring and super-resolution: Dataset and study,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops, 2019, pp. 0–0.

[48] M. Tassano, J. Delon, and T. Veit, “Fastdvdnet: Towards real-time deep video denoising without flow estimation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 1354–1363.

[49] S. Ge, S. Nah, G. Liu, T. Poon, A. Tao, B. Catanzaro, D. Jacobs, J.- B. Huang, M.-Y. Liu, and Y. Balaji, “Preserve your own correlation: A noise prior for video diffusion models,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 22 930–22 941.

[50] J. Z. Wu, Y. Ge, X. Wang, S. W. Lei, Y. Gu, Y. Shi, W. Hsu, Y. Shan, X. Qie, and M. Z. Shou, “Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 7623–7633.

[51] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 586–595.

[52] T. Unterthiner, S. van Steenkiste, K. Kurach, R. Marinier, M. Michalski, and S. Gelly, “Fvd: A new metric for video generation,” 2019.

[53] G. Daras, W. Nie, K. Kreis, A. Dimakis, M. Mardani, N. Kovachki, and A. Vahdat, “Warped diffusion: Solving video inverse problems with image diffusion models,” Advances in Neural Information Processing Systems, vol. 37, pp. 101 116–101 143, 2024.