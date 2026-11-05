# Unaligned Image Guided Denoising via Cross-modal Conditional Flow Matching

Runmin Zhang<sup>1∗</sup> Linshan Li<sup>1∗</sup> Si-Yuan Cao<sup>1</sup> Zhu Yu<sup>1</sup> Lingyu Zhu<sup>2</sup> Huaqi Zhang<sup>3</sup> Hui-Liang Shen<sup>1†</sup>

<sup>1</sup>Zhejiang University <sup>2</sup>City University of Hong Kong <sup>3</sup>vivo Mobile Communication Co., Ltd.

{runmin\_zhang, linshan\_li, cao\_siyuan, yu\_zhu}@zju.edu.cn

lingyzhu-c@my.cityu.edu.hk, zhanghuaqi@vivo.com, shenhl@zju.edu.cn

## Abstract

This work presents UGD-FM, a novel unaligned image guided denoising framework based on cross-modal conditional flow matching. Unlike previous approaches that assume spatially aligned inputs or only handle small-baseline rectified pairs, we address a more challenging setting where severe target noise and large-range twodimensional cross-modal misalignment coexist. Specifically, we formulate guided denoising as a progressive flow matching process, allowing image restoration and aligned guidance aggregation to mutually reinforce each other. To support efficient few-step inference, we introduce a stage-focused and inference-consistent training strategy that focuses on task-critical denoising stages, and mitigates the traininginference mismatch. We further propose adaptive sparse guidance propagation for efficient and reliable large-range matching, regularized by perceptual and geometric auxiliary losses. Experiments demonstrate that UGD-FM achieves stateof-the-art denoising performance with only one or two inference steps, and is compatible with modern single-image restoration backbones. Code is available at https://github.com/RM-Zhang/UGD-FM.

## 1 Introduction

Image denoising is a fundamental problem in low-level vision. Although modern restoration networks [47, 39, 46, 4, 48] have substantially improved denoising performance, faithfully recovering fine structures from heavily corrupted observations without over-smoothing remains challenging. To alleviate this limitation, recent works [6, 10, 30, 44, 36, 31, 51, 37] have leveraged auxiliary guidance images, such as near-infrared (NIR) images, which provide cleaner structural cues for denoising.

However, most guided denoising approaches [6, 30, 44, 36, 37] assume that the noisy RGB image and the guidance image are spatially aligned. This assumption is difficult to satisfy in practical mobile imaging systems, where cross-modal images are typically captured by different cameras [53, 33]. Recent approaches [31, 51] have extended guided denoising to cross-modal stereo pairs, but remain limited to horizontally rectified pairs with small baselines. In contrast, we study unaligned guided denoising under more severe spatial misalignment than the small-baseline rectified stereo setting considered in prior work. This expands the correspondence search from a one-dimensional horizontal search to a two-dimensional search space [54, 28, 50], making reliable cross-modal matching and guidance extraction significantly more difficult. This challenge is further amplified by severe noise in the target RGB image and inherent cross-modal appearance differences.

To address these challenges, we propose UGD-FM, a novel framework for Unaligned image Guided Denoising via cross-modal conditional Flow Matching. Specifically, we formulate guided denoising as a conditional flow matching process that progressively transports the noisy RGB image distribution toward the clean RGB image distribution, conditioned on unaligned cross-modal guidance. As the transport process progresses, the target image becomes cleaner, making cross-modal correspondence easier to establish and enabling more effective use of the guidance information. As illustrated in Fig. 1, this creates a mutually reinforcing process between guidance extraction and image restoration, leading to more robust denoising under severe spatial misalignment and noise.

![](images/b67b10e57a8d90c1c1aea820bdc5cc5c4f1b8f6699f7de458c5d2a427b4852a6.jpg)  
Figure 1: Illustration of the cross-modal conditional flow matching trajectory of UGD-FM. Given a noisy RGB image and an unaligned NIR guidance image, UGD-FM progressively restores the RGB state toward the clean target, forming a mutual reinforcing process between image restoration and guidance aggregation. For clearer visualization, the brightness of the zoom-in patches is enhanced.

To enable efficient and effective few-step guided denoising, we further design a stage-focused and inference-consistent training strategy. Our key observation is that the denoising trajectory contains two critical stages, i.e., a highly corrupted stage for coarse RGB structure recovery, and a relatively cleaner stage for reliable cross-modal matching and fine-detail restoration. Based on this observation, we place the inference timesteps around these stages, and align training with this schedule through stage-focused time sampling and inference-consistent state construction. With this design, UGD-FM delivers strong performance with only a single inference step, and achieves state-of-the-art results with two-step inference.

As for cross-modal matching and guidance extraction, a straightforward solution is to perform dense correspondence search over a large displacement range [31, 51]. However, such a dense matching strategy incurs prohibitive computational cost. To mitigate this issue, we introduce an adaptive sparse guidance propagation mechanism. Specifically, our method progressively focuses on a small set of high-confidence correspondence candidates, and propagates reliable guidance cues in a coarse-to-fine manner. To stabilize sparse guidance aggregation, we further introduce two auxiliary losses that provide perceptual and geometric regularization.

Importantly, we highlight that both our adaptive sparse guidance propagation mechanism and conditional flow matching formulation are compatible with a broad range of modern single-image restoration backbones, such as NAFNet [4], Uformer [39], and Restormer [46]. Our contributions are summarized as follows:

• We propose UGD-FM, a general cross-modal conditional flow matching framework for unaligned image guided denoising. UGD-FM establishes a mutually reinforcing process between image restoration and guidance aggregation, enabling robust restoration under severe noise and large-range spatial misalignment.

• We design a stage-focused and inference-consistent training strategy for few-step inference. It focuses inference and training on two critical denoising stages, and mitigates the training inference mismatch, enabling effective one-step and two-step guided denoising.

• We introduce adaptive sparse guidance propagation for efficient large-range cross-modal matching and aligned guidance aggregation, regularized by perceptual and geometric auxiliary losses to stabilize sparse correspondence learning.

• Comprehensive experiments demonstrate that UGD-FM achieves state-of-the-art denoising performance, and is compatible with modern single-image restoration backbones.

## 2 Related Work

Single Image Restoration. Single image restoration aims to recover clean images from degraded observations. Recently, deep convolutional neural networks have become a dominant paradigm by learning mappings from degraded inputs to clean targets. Representative designs include residual learning [49], multi-stage refinement [47], and inter-/intra-block architectures [4]. Vision transformers and their variants [3, 19, 39, 46, 48] further improve restoration performance by modeling long-range context. Despite these advances, single image restoration remains fundamentally ill-posed under severe degradation, often leading to over-smoothed results and missing structural details.

Guided Image Restoration. Guided image restoration leverages guidance images from auxiliary modalities to provide complementary structural information, with NIR-guided RGB denoising being a representative application. Early methods [9, 18, 26] employ joint filtering to model structural relationships between target and guidance images. Further approaches improve denoising performance with convolutional sparse coding [6], ADMM-based optimization [44], frequency decomposition [30, 23, 37], and multi-modal inconsistency priors [10, 36, 43]. Nevertheless, most of these approaches assume that the target and guidance images are strictly aligned. This requirement usually depends on specialized imaging systems or careful calibration [8, 31, 12], making it difficult to satisfy in practical mobile imaging systems. Recent approaches [32, 42, 52, 5, 31, 51] start to relax the strict alignment assumption. However, they either depend on external optical flow networks for pre-alignment, or handle limited misalignment. Other approaches [20, 21, 13, 16, 17] handle infrared and visible image fusion, in which SFRF [17] introduces multi-scale iterative registration. In contrast, we address a more challenging setting, where severe target noise and large-range cross-modal displacement must be addressed jointly.

Generative Models for Image Restoration. Generative priors, particularly diffusion [5, 55, 34, 15] and flow matching [24, 40, 27] models, have demonstrated strong capabilities in image restoration by learning to transport a simple base distribution to the data manifold. Recent studies [2, 45, 7] further show that initializing from low-quality observations, rather than Gaussian noise, reduces the transport distance and improves structural fidelity by anchoring the generation process to the input.

## 3 Method

Given a noisy RGB image $\mathbf { I } _ { \mathrm { N } } \in \mathbb { R } ^ { H \times W \times 3 }$ and an unaligned NIR guidance image $\mathbf { I } _ { \mathrm { G } } \in \mathbb { R } ^ { H \times W }$ captured by different cameras, our goal is to recover the clean RGB image $\mathbf { I } _ { \mathrm { T } } \in \mathbf { \mathbb { R } } ^ { H \times W \times 3 }$ . The spatial misalignment induces cross-modal displacements spanning a 2D range $[ - D , D ] ^ { 2 }$ , requiring joint image restoration and correspondence estimation under severe noise.

To tackle this challenge, we propose UGD-FM, an unaligned image guided denoising framework based on cross-modal conditional flow matching. We first introduce the conditional flow formulation and overall network architecture in Sec. 3.1. We then describe the stage-focused and inferenceconsistent training strategy for effective few-step flow matching in Sec. 3.2, followed by adaptive sparse guidance propagation for efficient guidance aggregation in Sec. 3.3. Finally, we present the training losses in Sec. 3.4.

## 3.1 Overview

Conditional Flow Formulation. Previous guided denoising approaches [10, 44, 31, 36, 43] typically learn a direct mapping from the noisy target and guidance images to the clean target image. However, this paradigm is less effective under severe target noise and spatial misalignment, since reliable cross-modal correspondences must be estimated from a highly corrupted target state. To address this issue, we formulate unaligned guided denoising as a conditional flow matching process that progressively transports the noisy RGB distribution to the clean RGB distribution.

For each training triplet $\left( \mathbf { I } _ { \mathrm { N } } , \mathbf { I } _ { \mathrm { G } } , \mathbf { I } _ { \mathrm { T } } \right)$ , we define a linear path [22, 7] that satisfies

$$
{ \bf z } _ { t } = ( 1 - t ) { \bf I } _ { \mathrm { N } } + t { \bf I } _ { \mathrm { T } } , \quad t \in [ 0 , 1 ] ,\tag{1}
$$

where ${ \bf z } _ { 0 } = { \bf I } _ { \mathrm { N } }$ and $\mathbf { z } _ { \mathrm { 1 } } = \mathbf { I } _ { \mathrm { T } }$ . The corresponding target velocity is

$$
\mathbf { u } _ { t } = \frac { d \mathbf { z } _ { t } } { d t } = \mathbf { I } _ { \mathrm { T } } - \mathbf { I } _ { \mathrm { N } } .\tag{2}
$$

![](images/896e4df54bcc4e025d8aa610f61b0785d1d8d95b905759e8f9693fac7b96c4e2.jpg)  
Figure 2: Network architecture of UGD-FM. (a) UGD-FM adopts a two-branch framework for unaligned RGB-NIR guided denoising. (b) The decoder block aggregates aligned guidance features and fuses them with target features. (c) Adaptive sparse guidance propagation transfers reliable coarse candidates to finer scales for efficient sparse matching and guidance aggregation.

We train a conditional velocity network $v _ { \theta }$ to predict this velocity from the current state $\mathbf { z } _ { t } .$ the unaligned guidance $\mathbf { I } _ { \mathrm { G } }$ , and the time step t. The training objective is

$$
\mathcal { L } _ { \mathrm { f m } } = \mathbb { E } _ { ( \mathbf { I } _ { \mathrm { N } } , \mathbf { I } _ { \mathrm { G } } , \mathbf { I } _ { \mathrm { T } } ) , t } \left. v _ { \theta } ( \mathbf { z } _ { t } , \mathbf { I } _ { \mathrm { G } } , t ) - ( \mathbf { I } _ { \mathrm { T } } - \mathbf { I } _ { \mathrm { N } } ) \right. _ { 1 } .\tag{3}
$$

During inference, we initialize ${ \bf z } _ { 0 } = { \bf I } _ { \mathrm { N } }$ , and solve $d \mathbf { z } _ { t } / d t = v _ { \theta } ( \mathbf { z } _ { t } , \mathbf { I } _ { \mathrm { G } } , t )$ with a few-step Euler solver. Unlike direct regression, this formulation provides a progressive denoising trajectory. Along this trajectory, the target state $\mathbf { z } _ { t }$ is encouraged to become cleaner and more structurally reliable as t increases, which facilitates cross-modal correspondence estimation. The aggregated guidance further improves velocity prediction, forming a mutually reinforcing process between guidance aggregation and image restoration.

Overall Network Architecture. As illustrated in Fig. 2(a), we implement the velocity network with a two-branch U-Net [29]. We first extract multi-scale RGB and NIR features from the current denoising state $\mathbf { z } _ { t }$ and the guidance image $\mathbf { I } _ { \mathrm { G } } .$ , respectively. The decoder then performs cross-modal guidance aggregation and feature fusion in a coarse-to-fine manner. As shown in Fig. 2(b), at each decoding scale, we first perform cross-modal guidance aggregation to obtain aligned NIR features, and then inject them into the target features through spatially variant feature fusion [51]. Across decoding scales, we further introduce adaptive sparse guidance propagation, as shown in Fig. 2(c), to avoid redundant large-range feature matching. Following feature alignment, a gated feature fusion module is added to adaptively adjust the role of the guiding information, preventing the injection of erroneous information in case of misalignment.

We highlight that both the conditional flow formulation and the guidance aggregation design are compatible with modern single-image restoration backbones, such as NAFNet [4], Uformer [39], and Restormer [46]. Thus, our UGD-FM provides a general framework for adapting existing single-image denoisers to unaligned cross-modal guided denoising.

## 3.2 Stage-Focused and Inference-Consistent Training

Although the conditional flow formulation improves over direct regression, directly applying standard flow matching is still suboptimal for efficient image restoration. Increasing solver steps introduces additional computation, and may accumulate prediction errors along solver-induced inference trajectories. Therefore, we aim to learn an effective and efficient few-step model, e.g., with one-step or two-step inference, for high-quality guided denoising.

To this end, we revisit the denoising trajectory of unaligned guided denoising, and identify two taskcritical stages. The highly corrupted stage with small t focuses on coarse RGB structure recovery, while the relatively cleaner stage with larger t enables more reliable cross-modal matching and fine-detail restoration. Based on this observation, we place the few-step inference timesteps around these two stages, such that the solver directly visits the states most relevant to guided denoising. We then align training with this stage-focused inference schedule at two levels, i.e., stage-focused time sampling aligns the training timesteps with the inference anchors, while inference-consistent state construction aligns the training states with solver-induced inference states.

Stage-Focused Time Sampling. We first sample t uniformly from $[ 0 , 1 ]$ for stable velocity learning, and then switch to stage-focused sampling to match the few-step inference schedule. Specifically, we use two inference anchors $\tau _ { \mathrm { h c } }$ and $\tau _ { \mathrm { r c } }$ for the highly corrupted and relatively cleaner stages, respectively, where $\tau _ { \mathrm { h c } } = 0$ and $0 < \tau _ { \mathrm { r c } } < 1$ . With training iteration n and warm-up length $N _ { \mathrm { w a r m } } ,$ the training time distribution is defined as

$$
p _ { \mathrm { t r a i n } } ^ { ( n ) } = \left\{ \begin{array} { l l } { \mathcal { U } ( 0 , 1 ) , } & { n < N _ { \mathrm { w a r m } } , } \\ { p _ { \mathrm { s f } } , } & { n \ge N _ { \mathrm { w a r m } } , } \end{array} \right.\tag{4}
$$

where $\mathcal { U } ( 0 , 1 )$ denotes the uniform distribution over $[ 0 , 1 ]$ . The stage-focused distribution $p _ { \mathrm { s f } }$ is defined as a mixture of two truncated Gaussian distributions as

$$
p _ { \mathrm { s f } } = \lambda _ { \mathrm { h c } } q _ { \mathrm { h c } } + \lambda _ { \mathrm { r c } } q _ { \mathrm { r c } } , \quad \lambda _ { \mathrm { h c } } + \lambda _ { \mathrm { r c } } = 1 ,\tag{5}
$$

with

$$
q _ { s } = \mathcal { N } _ { [ 0 , 1 ] } ( \tau _ { s } , \sigma _ { s } ^ { 2 } ) , \quad s \in \{ \mathrm { h c } , \mathrm { r c } \} .\tag{6}
$$

Here, $\sigma _ { s }$ controls the sampling concentration around the anchor, and $\mathcal { N } _ { [ 0 , 1 ] }$ denotes a Gaussian distribution truncated and normalized on [0, 1].

Inference-Consistent State Construction. Although stage-focused sampling aligns training and inference at the timestep level, the relatively cleaner stage still has a state mismatch. It is reached after the first solver update during inference, whereas the network is trained on the interpolation state $\mathbf { z } _ { t }$ in Eq. 1. We therefore replace the interpolation state with a solver-induced state for samples from $q _ { \mathrm { r c } }$ . For $t \sim q _ { \mathrm { r c } } ,$ , we first predict the initial velocity as $\mathbf { v } _ { 0 } = v _ { \theta } \left( \mathbf { I } _ { \mathrm { N } } , \mathbf { I } _ { \mathrm { G } } , 0 \right)$ . The inference-consistent training state is then constructed by a local Euler update as $\tilde { \mathbf { z } } _ { t } = \mathbf { I } _ { \mathrm { N } } + t \mathbf { v } _ { 0 }$ . Finally, the corresponding velocity target can be defined as the residual direction to the clean target as $\tilde { \mathbf { u } } _ { t } = ( \mathbf { I } _ { \mathrm { T } } - \tilde { \mathbf { z } } _ { t } ) \bar { / } ( 1 - t \bar { ) }$

By combining stage-focused time sampling with inference-consistent state construction, UGD-FM better aligns training with few-step inference at both the timestep and state levels, enabling highquality guided denoising with only one or two solver steps.

## 3.3 Adaptive Sparse Guidance Propagation

Large-range cross-modal matching is essential for unaligned guided denoising. A straightforward solution is to perform dense large-range matching [31, 51, 5] at every decoding scale. However, it is computationally expensive, especially at high resolutions. Moreover, effective guidance aggregation mainly depends on a sparse set of reliable correspondences rather than exhaustive dense relations. We therefore propose adaptive sparse guidance propagation, which discovers reliable candidates at a coarse scale and propagates them to finer scales for sparse refinement, as illustrated in Fig. 2(c).

At scale l, let $s _ { i } ^ { l } ( \mathbf { p } )$ denote the similarity score between target location $\mathbf { p } _ { i } ^ { l }$ and guidance location p, and let $\mathbf { v } ^ { l } ( \mathbf { p } )$ denote the guidance value feature at p. A larger l indicates a coarser scale.

Coarse Candidate Discovery. At the coarsest scale L, we search within the range $\mathcal { R } _ { i } ^ { L } = \{ \mathbf { p } \ |$ $\mathbf p - \mathbf p _ { i } ^ { L } \in [ - D / 2 ^ { L - 1 } , D / 2 ^ { L - 1 } ] ^ { 2 } \big \}$ for each target location $\mathbf { p } _ { i } ^ { L }$ , and retain the top-K guidance candidates as

$$
\mathcal { C } _ { i } ^ { L } = \left\{ ( \mathbf { p } , \alpha _ { i } ^ { L } ( \mathbf { p } ) ) \mid \mathbf { p } \in \mathrm { T o p K } _ { \mathbf { p } \in \mathcal { R } _ { i } ^ { L } } \left( s _ { i } ^ { L } ( \mathbf { p } ) \right) \right\} ,\tag{7}
$$

where $\alpha _ { i } ^ { L } ( \mathbf { p } )$ is obtained by applying softmax over the selected top-K candidates. The coordinates are propagated to finer scales, while the weights are used for aligned guidance aggregation. Thus, $\mathcal { C } _ { i } ^ { L }$ preserves reliable large-displacement candidates, and discards most irrelevant regions.

Sparse Candidate Propagation. At a finer scale $l < L ,$ we refine the candidates propagated from the previous coarser scale. Let $\pi ( i )$ denote the parent location of target location i at scale $l + 1$ . We

expand the coordinates in ${ \mathcal { C } } _ { \pi ( i ) } ^ { l + 1 }$ to their $2 \times 2$ child locations at scale l, forming a sparse search set as

$$
\mathcal { S } _ { i } ^ { l } = \mathrm { C h i l d } \left( \{ \mathbf { p } \ | \ ( \mathbf { p } , \alpha ) \in \mathcal { C } _ { \pi ( i ) } ^ { l + 1 } \} \right) ,\tag{8}
$$

where Child(·) maps each coarse candidate coordinate to its $2 \times 2$ child locations at the current scale. We then select the top- $K$ candidates only within this sparse set as

$$
\mathcal { C } _ { i } ^ { l } = \left\{ \left( \mathbf { p } , \alpha _ { i } ^ { l } ( \mathbf { p } ) \right) | \mathbf { p } \in \mathrm { T o p K } _ { \mathbf { p } \in S _ { i } ^ { l } } \left( s _ { i } ^ { l } ( \mathbf { p } ) \right) \right\} ,\tag{9}
$$

where $\alpha _ { i } ^ { l } ( \mathbf { p } )$ is normalized over the selected top-K candidates. The refined candidates are used for guidance aggregation at scale l, and further propagated to the next finer scale.

Aligned Guidance Aggregation. Given $\mathcal { C } _ { i } ^ { l }$ , we perform aligned guidance aggregation only over the selected sparse candidates as $\begin{array} { r } { \mathbf { F } _ { \mathrm { G , a l i g n } } ^ { l } ( \mathbf { p } _ { i } ^ { l } ) = \sum _ { ( \mathbf { p } , \boldsymbol { \alpha } _ { \mathrm { + } } ^ { l } ( \mathbf { p } ) ) \in \mathcal { C } _ { \mathrm { + } } ^ { l } } \boldsymbol { \alpha } _ { i } ^ { l } ( \mathbf { p } ) \mathbf { v } ^ { l } ( \mathbf { p } ) } \end{array}$ . Following SGDFormer [51], we further fuse the aligned guidance and target features using spatially variant weights, which adaptively modulate each location and mitigate artifacts from imprecise matches. In this way, we avoid redundant dense matching, and aggregate only the most useful cross-modal cues for guided denoising.

## 3.4 Loss Functions

Although adaptive sparse guidance propagation avoids exhaustive dense matching at finer scales, learning reliable sparse candidates remains challenging. To regularize training, we introduce two auxiliary losses, namely perceptual consistency loss and flow consistency loss, which provide auxiliary perceptual regularization and explicit displacement supervision, respectively. These losses encourage the model to select reliable guidance information for restoration.

Perceptual Consistency Loss. To encourage feature-space compatibility between the aggregated NIR guidance and the RGB target, we apply the learned aggregation weights to the guidance image $\mathbf { I } _ { \mathrm { G } }$ and obtain an aligned guidance image at scale l as $\begin{array} { r } { \mathbf { I } _ { \mathrm { G , a l i g n } } ^ { l ^ { - } } ( \mathbf { p } _ { i } ^ { l } ) = \breve { \sum } _ { ( \mathbf { p } , \alpha _ { i } ^ { l } ( \mathbf { p } ) ) \in \mathcal { C } _ { i } ^ { l } } \alpha _ { i } ^ { l } ( \mathbf { p } ) \mathbf { I } _ { \mathrm { G } } ^ { l } ( \mathbf { \bar { p } } ) } \end{array}$ where $\mathbf { I } _ { \mathrm { G } } ^ { l }$ is the downsampled NIR image at scale l. We then compute the perceptual consistency loss between the aligned guidance image and the clean RGB target as

$$
\mathcal { L } _ { \mathrm { p c } } = \sum _ { l } \mathcal { P } \left( \mathbf { I } _ { \mathrm { G , a l i g n } } ^ { l } , \mathbf { I } _ { \mathrm { T } } ^ { l } \right) ,\tag{10}
$$

where $\mathcal { P } ( \cdot , \cdot )$ denotes the LPIPS [11] loss, and $\mathbf { I } _ { \mathrm { T } } ^ { l }$ is the clean RGB target downsampled to scale l.

Flow Consistency Loss. To further constrain sparse correspondence learning, we supervise the displacement induced by the selected candidates with pseudo flow annotations. The pseudo annotations are precomputed by an existing flow estimation network [54] between the clean RGB target $\mathbf { I } _ { \mathrm { T } }$ and the guidance image $\mathbf { I } _ { \mathrm { G } }$ , and are used as geometric supervision during training. Specifically, we compute a soft displacement field from the aggregation weights as $\begin{array} { r } { \mathbf { f } ^ { l } ( \mathbf { p } _ { i } ^ { l } ) = \sum _ { ( \mathbf { p } , \alpha _ { i } ^ { l } ( \mathbf { p } ) ) \in \mathcal { C } _ { i } ^ { l } } \alpha _ { i } ^ { \bar { l } } ( \mathbf { p } ) \left( \mathbf { p } - \mathbf { \bar { p } } _ { i } ^ { l } \right) } \end{array}$ Let $\mathbf { f } _ { \mathrm { p s e u d o } } ^ { l }$ denote the pseudo displacement field downsampled to scale l. The flow consistency loss is defined as

$$
\mathcal { L } _ { \mathrm { f c } } = \sum _ { l } \sum _ { i } \left. \mathbf { f } ^ { l } ( \mathbf { p } _ { i } ^ { l } ) - \mathbf { f } _ { \mathrm { p s e u d o } } ^ { l } ( \mathbf { p } _ { i } ^ { l } ) \right. _ { 1 } .\tag{11}
$$

This loss provides geometric supervision for sparse candidate selection, and further encourages reliable cross-modal guidance aggregation.

Overall Objective. The final training objective is

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { f m } } + \lambda _ { \mathrm { p c } } \mathcal { L } _ { \mathrm { p c } } + \lambda _ { \mathrm { f c } } \mathcal { L } _ { \mathrm { f c } } ,\tag{12}
$$

where ${ \mathcal L } _ { \mathrm { f m } }$ is the flow matching loss in Eq. 3, and $\lambda _ { \mathrm { p c } }$ and $\lambda _ { \mathrm { f c } }$ balance the auxiliary losses. We note that ${ \mathcal { L } } _ { \mathrm { p c } }$ and ${ \mathcal L } _ { \mathrm { f c } }$ are used only during training to provide perceptual regularization and geometric supervision, respectively, and introduce no additional inference cost.

## 4 Experiments

## 4.1 Experimental Setting

Datasets. We evaluate UGD-FM on FANVID [8], RGB-NIR Scene [1], and MS2 [33]. Dataset details are given in Appendix A.1.

Table 1: Quantitative comparison on the FANVID [8] dataset under Gaussian noise $( \sigma = 0 . 3 )$ Poisson-Gaussian noise $( \alpha = 0 . 0 2 , \sigma = 0 . 2 )$ , and Poisson-Gaussian noise (α = 0.03, σ = 0.3). The best and second-best results are highlighted in bold and underlined, respectively.
<table><tr><td rowspan="2">Method</td><td colspan="3">σ = 0.3</td><td colspan="3"> $\alpha = 0 . 0 2 , \sigma = 0 . 2$ </td><td colspan="3"> $\alpha = 0 . 0 3 , \sigma = 0 . 3$ </td></tr><tr><td>PSNR↑</td><td>SSIM ↑ LPIPS↓</td><td></td><td>PSNR ↑</td><td>SSIM ↑ LPIPS ↓</td><td></td><td>PSNR ↑</td><td></td><td>SSIM ↑ LPIPS ↓</td></tr><tr><td>MPRNet [47]</td><td>29.48</td><td>0.8039</td><td>0.3480</td><td>30.82</td><td>0.8358</td><td>0.3040</td><td>29.25</td><td>0.7995</td><td>0.3520</td></tr><tr><td>NAFNet [4]</td><td>29.05</td><td>0.7932</td><td>0.3541</td><td>30.28</td><td>0.8239</td><td>0.3165</td><td>28.84</td><td>0.7891</td><td>0.3583</td></tr><tr><td>Uformer [39]</td><td>29.44</td><td>0.8037</td><td>0.3437</td><td>30.76</td><td>0.8358</td><td>0.3003</td><td>29.22</td><td>0.7995</td><td>0.3464</td></tr><tr><td>Restormer [46]</td><td>29.42</td><td>0.8024</td><td>0.3459</td><td>30.75</td><td>0.8339</td><td>0.3044</td><td>29.20</td><td>0.7981</td><td>0.3497</td></tr><tr><td>Xformer [48]</td><td>29.36</td><td>0.8008</td><td>0.3514</td><td>30.69</td><td>0.8332</td><td>0.3078</td><td>29.14</td><td>0.7967</td><td>0.3548</td></tr><tr><td>CUNet [6]</td><td>28.16</td><td>0.7618</td><td>0.4097</td><td>29.55</td><td>0.8023</td><td>0.3711</td><td>27.87</td><td>0.7537</td><td>0.4154</td></tr><tr><td>DVN [10]</td><td>29.27</td><td>0.7985</td><td>0.3592</td><td>30.60</td><td>0.8311</td><td>0.3152</td><td>29.04</td><td>0.7938</td><td>0.3631</td></tr><tr><td>RFFNet [36]</td><td>29.34</td><td>0.8013</td><td>0.3506</td><td>30.67</td><td>0.8332</td><td>0.3074</td><td>29.12</td><td>0.7968</td><td>0.3546</td></tr><tr><td>NAID [43]</td><td>29.21</td><td>0.7976</td><td>0.3504</td><td>30.49</td><td>0.8297</td><td>0.3098</td><td>28.99</td><td>0.7933</td><td>0.3540</td></tr><tr><td>SANet [31]</td><td>29.15</td><td>0.8072</td><td>0.3611</td><td>30.45</td><td>0.8388</td><td>0.3194</td><td>28.92</td><td>0.8029</td><td>0.3651</td></tr><tr><td>UGD-FM (NAFNet, 1-step)</td><td>30.34</td><td>0.8384</td><td>0.2313</td><td>31.55</td><td>0.8648</td><td>0.2015</td><td>30.12</td><td>0.8351</td><td>0.2351</td></tr><tr><td>UGD-FM (NAFNet, 2-step)</td><td>30.62</td><td>0.8486</td><td>0.2210</td><td>31.79</td><td>0.8722</td><td>0.1964</td><td>30.42</td><td>0.8458</td><td>0.2237</td></tr><tr><td>UGD-FM (Uformer, 1-step)</td><td>30.32</td><td>0.8342</td><td>0.2308</td><td>31.54</td><td>0.8652</td><td>0.2010</td><td>30.09</td><td>0.8343</td><td>0.2351</td></tr><tr><td>UGD-FM (Uformer, 2-step)</td><td>30.66</td><td>0.8499</td><td>0.2201</td><td>31.81</td><td>0.8732</td><td>0.1962</td><td>30.45</td><td>0.8467</td><td>0.2229</td></tr><tr><td>UGD-FM (Restormer, 1-step)</td><td>30.50</td><td>0.8436</td><td>0.2179</td><td>31.71</td><td>0.8694</td><td>0.1909</td><td>30.29</td><td>0.8401</td><td>0.2215</td></tr><tr><td>UGD-FM (Restormer, 2-step)</td><td>30.77</td><td>0.8535</td><td>0.2098</td><td>31.93</td><td>0.8764</td><td>0.1870</td><td>30.58</td><td>0.8506</td><td>0.2121</td></tr></table>

Table 2: Quantitative comparison on the RGB-NIR Scene dataset [1] under Gaussian noise $( \sigma = 0 . 3 )$ Poisson-Gaussian noise $( \alpha = 0 . 0 2 , \sigma = 0 . 2 )$ , and Poisson-Gaussian noise $( \alpha = 0 . 0 3 , \sigma = 0 . 3 )$ . The best and second-best results are highlighted in bold and underlined, respectively.
<table><tr><td rowspan="2">Method</td><td colspan="3">σ = 0.3</td><td colspan="3"> $\alpha = 0 . 0 2 , \sigma = 0 . 2$ </td><td colspan="3"> $\alpha = 0 . 0 3 , \sigma = 0 . 3$ </td></tr><tr><td>PSNR ↑</td><td>SSIM ↑ LPIPS↓</td><td></td><td>PSNR ↑</td><td>SSIM↑</td><td>LPIPS ↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>MPRNet [47]</td><td>29.37</td><td>0.7887</td><td>0.3428</td><td>30.55</td><td>0.8189</td><td>0.3069</td><td>29.12</td><td>0.7831</td><td>0.3480</td></tr><tr><td>NAFNet [4]</td><td>29.07</td><td>0.7851</td><td>0.3409</td><td>30.23</td><td>0.8152</td><td>0.3068</td><td>28.81</td><td>0.7794</td><td>0.3466</td></tr><tr><td>Uformer [39]</td><td>29.35</td><td>0.7890</td><td>0.3370</td><td>30.54</td><td>0.8201</td><td>0.3010</td><td>29.09</td><td>0.7833</td><td>0.3407</td></tr><tr><td>Restormer [46]</td><td>29.38</td><td>0.7897</td><td>0.3365</td><td>30.56</td><td>0.8198</td><td>0.3014</td><td>29.12</td><td>0.7841</td><td>0.3417</td></tr><tr><td>Xformer [48]</td><td>29.46</td><td>0.7912</td><td>0.3349</td><td>30.65</td><td>0.8216</td><td>0.2978</td><td>29.19</td><td>0.7859</td><td>0.3394</td></tr><tr><td>CUNet [6]</td><td>27.37</td><td>0.7248</td><td>0.4484</td><td>28.70</td><td>0.7709</td><td>0.4046</td><td>27.00</td><td>0.7115</td><td>0.4612</td></tr><tr><td>DVN [10]</td><td>29.14</td><td>0.7831</td><td>0.3528</td><td>30.32</td><td>0.8142</td><td>0.3168</td><td>28.86</td><td>0.7771</td><td>0.3587</td></tr><tr><td>RFFNet [36]</td><td>29.06</td><td>0.7817</td><td>0.3540</td><td>30.25</td><td>0.8130</td><td>0.3175</td><td>28.79</td><td>0.7759</td><td>0.3603</td></tr><tr><td>NAID [43]</td><td>29.10</td><td>0.7856</td><td>0.3399</td><td>30.25</td><td>0.8155</td><td>0.3065</td><td>28.53</td><td>0.7801</td><td>0.3456</td></tr><tr><td>SANet [31]</td><td>28.23</td><td>0.7540</td><td>0.3850</td><td>29.27</td><td>0.7878</td><td>0.3477</td><td>28.01</td><td>0.7477</td><td>0.3917</td></tr><tr><td>UGD-FM (NAFNet, 1-step)</td><td>29.54</td><td>0.8077</td><td>0.2697</td><td>30.65</td><td>0.8346</td><td>0.2413</td><td>29.29</td><td>0.8024</td><td>0.2753</td></tr><tr><td>UGD-FM (NAFNet, 2-step)</td><td>29.77</td><td>0.8163</td><td>0.2600</td><td>30.83</td><td>0.8407</td><td>0.2363</td><td>29.54</td><td>0.8117</td><td>0.2642</td></tr><tr><td>UGD-FM (Uformer, 1-step)</td><td>29.56</td><td>0.8074</td><td>0.2706</td><td>30.70</td><td>0.8350</td><td>0.2408</td><td>29.29</td><td>0.8018</td><td>0.2768</td></tr><tr><td>UGD-FM (Uformer, 2-step)</td><td>29.84</td><td>0.8172</td><td>0.2598</td><td>30.94</td><td>0.8421</td><td>0.2351</td><td>29.60</td><td>0.8123</td><td>0.2636</td></tr><tr><td>UGD-FM (Restormer, 1-step)</td><td>29.62</td><td>0.8095</td><td>0.2599</td><td>30.77</td><td>0.8369</td><td>0.2322</td><td>29.35</td><td>0.8040</td><td>0.2651</td></tr><tr><td>UGD-FM (Restormer, 2-step)</td><td>29.87</td><td>0.8188</td><td>0.2551</td><td>30.98</td><td>0.8437</td><td>0.2315</td><td>29.63</td><td>0.8141</td><td>0.2585</td></tr></table>

Implementation Details. We implement UGD-FM with the basic blocks of NAFNet [4], Uformer [39], and Restormer [46], yielding three model variants. The displacement search range is $[ - D , D ] ^ { 2 }$ with radius $D = 8 0 .$ . During training, degraded observations are synthesized by randomly adding Poisson-Gaussian noise to clean RGB images. Specifically, the Poisson noise parameter α is sampled from [0, 0.03], and the Gaussian noise level σ is sampled from [0, 0.3]. More implementation and training details are provided in Appendix A.2.

Metrics. We quantitatively evaluate the denoising performance using peak signal-to-noise ratio (PSNR), structural similarity (SSIM) [38], and learned perceptual image patch similarity (LPIPS) [11]. Higher PSNR and SSIM, and lower LPIPS, indicate better performance.

Table 3: Quantitative comparison on the MS2 [33] dataset under Gaussian noise $( \sigma = 0 . 3 )$ , Poisson-Gaussian noise $( \alpha = 0 . 0 2 , \sigma = 0 . 2 )$ , and Poisson-Gaussian noise (α = 0.03, σ = 0.3). The best and second-best results are highlighted in bold and underlined, respectively.
<table><tr><td rowspan="2">Method</td><td colspan="3">σ = 0.3</td><td colspan="3"> $\alpha = 0 . 0 2 , \sigma = 0 . 2$ </td><td colspan="3"> $\alpha = 0 . 0 3 , \sigma = 0 . 3$ </td></tr><tr><td>PSNR↑</td><td>SSIM ↑ LPIPS↓</td><td></td><td>PSNR ↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS ↓</td></tr><tr><td>MPRNet [47]</td><td>30.89</td><td>0.8836</td><td>0.3201</td><td>32.53</td><td>0.9091</td><td>0.2785</td><td>30.57</td><td>0.8797</td><td>0.3253</td></tr><tr><td>NAFNet [4]</td><td>30.76</td><td>0.8838</td><td>0.3095</td><td>32.36</td><td>0.9087</td><td>0.2718</td><td>30.48</td><td>0.8803</td><td>0.3133</td></tr><tr><td>Uformer [39]</td><td>30.10</td><td>0.8837</td><td>0.3188</td><td>31.42</td><td>0.9095</td><td>0.2799</td><td>29.84</td><td>0.8799</td><td>0.3234</td></tr><tr><td>Restormer [46]</td><td>30.98</td><td>0.8857</td><td>0.3098</td><td>32.62</td><td>0.9106</td><td>0.2707</td><td>30.69</td><td>0.8821</td><td>0.3145</td></tr><tr><td>Xformer [48]</td><td>30.34</td><td>0.8623</td><td>0.2966</td><td>31.98</td><td>0.8930</td><td>0.2538</td><td>30.01</td><td>0.8573</td><td>0.3033</td></tr><tr><td>CUNet [6]</td><td>29.61</td><td>0.8542</td><td>0.3887</td><td>31.27</td><td>0.8877</td><td>0.3444</td><td>29.29</td><td>0.8487</td><td>0.3959</td></tr><tr><td>DVN [10]</td><td>31.78</td><td>0.9096</td><td>0.2552</td><td>33.29</td><td>0.9272</td><td>0.2283</td><td>31.64</td><td>0.9077</td><td>0.2574</td></tr><tr><td>RFFNet [36]</td><td>30.08</td><td>0.8685</td><td>0.3666</td><td>31.78</td><td>0.8971</td><td>0.3219</td><td>29.77</td><td>0.8643</td><td>0.3728</td></tr><tr><td>NAID [43]</td><td>30.33</td><td>0.8732</td><td>0.3362</td><td>31.91</td><td>0.9000</td><td>0.2965</td><td>30.04</td><td>0.8693</td><td>0.3412</td></tr><tr><td>SANet [31]</td><td>29.68</td><td>0.8466</td><td>0.3489</td><td>31.27</td><td>0.8806</td><td>0.3021</td><td>29.33</td><td>0.8402</td><td>0.3579</td></tr><tr><td>UGD-FM (NAFNet, 1-step)</td><td>32.42</td><td>0.9237</td><td>0.1752</td><td>33.79</td><td>0.9379</td><td>0.1550</td><td>32.14</td><td>0.9219</td><td>0.1779</td></tr><tr><td>UGD-FM (NAFNet, 2-step)</td><td>32.64</td><td>0.9285</td><td>0.1685</td><td>33.96</td><td>0.9409</td><td>0.1524</td><td>32.39</td><td>0.9270</td><td>0.1704</td></tr><tr><td>UGD-FM (Uformer, 1-step)</td><td>32.65</td><td>0.9282</td><td>0.1794</td><td>33.99</td><td>0.9411</td><td>0.1589</td><td>32.36</td><td>0.9261</td><td>0.1827</td></tr><tr><td>UGD-FM (Uformer, 2-step)</td><td>32.97</td><td>0.9330</td><td>0.1648</td><td>34.24</td><td>0.9441</td><td>0.1490</td><td>32.71</td><td>0.9313</td><td>0.1667</td></tr><tr><td>UGD-FM (Restormer, 1-step)</td><td>32.85</td><td>0.9303</td><td>0.1635</td><td>34.18</td><td>0.9425</td><td>0.1458</td><td>32.59</td><td>0.9285</td><td>0.1658</td></tr><tr><td>UGD-FM (Restormer, 2-step)</td><td>33.07</td><td>0.9343</td><td>0.1592</td><td>34.34</td><td>0.9452</td><td>0.1444</td><td>32.82</td><td>0.9329</td><td>0.1609</td></tr></table>

## 4.2 Comparisons with Existing Approaches

We compare UGD-FM with ten representative methods, including five single-image denoisers, i.e., MPRNet [47], NAFNet [4], Uformer [39], Restormer [46], and Xformer [48], and five guided denoising methods, i.e., CUNet [6], DVN [10], RFFNet [36], NAID [43], and SANet [31]. We evaluate UGD-FM with three backbone blocks, including NAFNet, Uformer, and Restormer. The denoising performance is evaluated under three noise settings: Gaussian noise with $\sigma = 0 . 3 \mathrm { . }$ , Poisson-Gaussian noise with $\alpha = 0 . 0 2 , \sigma = 0 . 2$ , and Poisson-Gaussian noise with $\alpha = 0 . 0 3 , \sigma = 0 . 3$

Tables 1, 2, and 3 report quantitative results on FANVID [8], RGB-NIR Scene [1], and MS2 [33], respectively. UGD-FM consistently outperforms both single-image and guided denoising baselines across all datasets and noise settings. The gains over single-image denoisers demonstrate the benefit of exploiting NIR guidance, while the improvements over guided denoising methods show the effectiveness of UGD-FM under cross-modal misalignment. Among different variants, two-step inference further improves over one-step inference, and UGD-FM with the Restormer block achieves the best overall performance. Fig. 3 presents qualitative comparisons of different approaches. Single image denoising methods tend to oversmooth fine structures, while existing guided methods struggle to handle unaligned cross-modal inputs. In contrast, UGD-FM better exploits unaligned NIR guidance and recovers sharper textures and more faithful structural details.

Diffusion and Flow Matching Baselines. We further compare our UGD-FM with more diffusion and flow matching baselines like RDBM [35], 3M-TI [5] and RFfusion [40]. As shown in Table 4, UGD-FM outperforms these baselines in PSNR, SSIM, and LPIPS.

RGB-Guided Thermal Image Super-Resolution. We extend UGD-FM to RGB-guided thermal image super-resolution following the 3M-TI protocol [5], and list results in Table 5. UGD-FM achieves the best SSIM and MUSIQ, while obtaining competitive PSNR and perceptual quality, validating the generalizability and extensibility of our framework beyond guided denoising.

## 4.3 Ablation Study

We conduct ablation studies on FANVID [8] under Poisson-Gaussian noise with $\alpha = 0 . 0 3$ and $\sigma = 0 . 3$ . We use UGD-FM with the Restormer block as the default setting, and list results in Table 6.

Flow Matching Formulation and Training Strategy. We first examine the effect of the flow formulation. As shown in Table 6a, compared with direct regression, base flow matching achieves comparable or slightly better one-step performance. More importantly, the flow formulation naturally supports multi-step inference, improving PSNR from 30.28 to 30.34. This demonstrates the benefit of modeling guided denoising as a progressive transport process. Cumulatively adding stage-focused sampling and inference-consistent states increases two-step PSNR to 30.42 and 30.58, respectively. Stage-focused sampling aligns the training-time distribution with the fixed inference anchors. Inference-consistent state construction further aligns the training and inference states by replacing the interpolation state with a solver-induced state.

Table 4: Comparison with more diffusion and flow matching methods on FANVID [8] under Poisson-Gaussian noise (α = 0.03, σ = 0.3). The best and second-best results are highlighted in bold and underlined, respectively.
<table><tr><td>Method</td><td>PSNR ↑ SSIM ↑</td><td>LPIPS ↓</td></tr><tr><td>RDBM [35]</td><td>28.80 0.7828</td><td>0.3535</td></tr><tr><td>3M-TI [5]</td><td>28.89 0.7872</td><td>0.3221</td></tr><tr><td>RFfusion [40]</td><td>28.53 0.7702</td><td>0.3251</td></tr><tr><td>UGD-FM (NAFNet, 1-step)</td><td>30.12 0.8351</td><td>0.2351</td></tr><tr><td>UGD-FM (NAFNet, 2-step)</td><td>30.42 0.8458</td><td>0.2237</td></tr></table>

Table 5: Comparison of RGB-guided thermal image super-resolution following the 3M-TI [5] protocol. The best and second-best results are highlighted in bold and underlined, respectively.
<table><tr><td>Method</td><td>PSNR ↑</td><td></td><td>SSIM ↑ LPIPS ↓</td><td>MANIQA ↑ MUSIQ ↑</td><td></td></tr><tr><td>CoRPLE [14]</td><td>30.47</td><td>0.8642</td><td>0.3206</td><td>0.2833</td><td>30.46</td></tr><tr><td>SeeSR [41]</td><td>29.41</td><td>0.8495</td><td>0.1828</td><td>0.4278</td><td>35.22</td></tr><tr><td>3M-TI [5]</td><td>30.09</td><td>0.8610</td><td>0.1787</td><td>0.4443</td><td>36.66</td></tr><tr><td>UGD-FM</td><td>30.36</td><td>0.8710</td><td>0.2012</td><td>0.4008</td><td>36.94</td></tr></table>

![](images/85a75e829145182d345cfe004f61abb4d6a6c1806bf774c71e33469137adb781.jpg)  
Figure 3: Qualitative comparison of image denoising results across different approaches.

Table 6: Ablation studies on FANVID [8] under Poisson-Gaussian noise $( \alpha = 0 . 0 3 , \sigma = 0 . 3 )$ (a) Flow matching formulation and training strategy.
<table><tr><td rowspan="2">Setting</td><td colspan="3">1-step</td><td colspan="3">2-step</td></tr><tr><td>PSNR↑</td><td>SSIM ↑</td><td>LPIPS↓</td><td>PSNR ↑</td><td>SSIM↑</td><td>LPIPS ↓</td></tr><tr><td>Direct regression</td><td>30.27</td><td>0.8367</td><td>0.2376</td><td></td><td></td><td></td></tr><tr><td>Standard FM</td><td>30.28</td><td>0.8389</td><td>0.2357</td><td>30.34</td><td>0.8401</td><td>0.2312</td></tr><tr><td>+ Stage-focused sampling</td><td>30.31</td><td>0.8396</td><td>0.2238</td><td>30.42</td><td>0.8421</td><td>0.2224</td></tr><tr><td>+ Inference-consistent state construction</td><td>30.29</td><td>0.8401</td><td>0.2215</td><td>30.58</td><td>0.8506</td><td>0.2121</td></tr></table>

(b) Guidance aggregation.
<table><tr><td rowspan="2">Setting</td><td colspan="3">1-step</td><td colspan="3">2-step</td></tr><tr><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td></tr><tr><td>No attention</td><td>28.97</td><td>0.7920</td><td>0.3247</td><td>29.10</td><td>0.7976</td><td>0.3244</td></tr><tr><td>Global attention</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>00M</td></tr><tr><td>Window attention</td><td>29.64</td><td>0.8188</td><td>0.2512</td><td>30.03</td><td>0.8370</td><td>0.2369</td></tr><tr><td>Sparse propagation</td><td>30.29</td><td>0.8401</td><td>0.2215</td><td>30.58</td><td>0.8506</td><td>0.2121</td></tr></table>

(c) Auxiliary losses.
<table><tr><td rowspan="2"> $\mathcal { L } _ { \mathrm { { p c } } }$ </td><td rowspan="2"> ${ \mathcal L } _ { \mathrm { f c } }$ </td><td colspan="3">1-step</td><td colspan="3">2-step</td></tr><tr><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>PSNR ↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>x</td><td>x</td><td>29.48</td><td>0.8062</td><td>0.2862</td><td>29.78</td><td>0.8213</td><td>0.2660</td></tr><tr><td>x</td><td>√</td><td>30.13</td><td>0.8355</td><td>0.2306</td><td>30.45</td><td>0.8465</td><td>0.2217</td></tr><tr><td>√</td><td>x</td><td>30.01</td><td>0.8324</td><td>0.2345</td><td>30.27</td><td>0.8445</td><td>0.2234</td></tr><tr><td>√</td><td>√</td><td>30.29</td><td>0.8401</td><td>0.2215</td><td>30.58</td><td>0.8506</td><td>0.2121</td></tr></table>

Adaptive Sparse Guidance Propagation. We evaluate the efficiency of the guidance aggregation design in Table 6b. Without attention, the model cannot handle spatial misalignment, leading to a large performance drop. Global attention leads to out-of-memory errors, showing that dense large-range matching is impractical at high resolutions. Window attention provides a non-local receptive field but is less flexible. In contrast, our sparse propagation avoids exhaustive matching while retaining reliable candidates for guidance aggregation, enabling efficient inference.

Auxiliary Losses. Table 6c shows complementary benefits from perceptual regularization and displacement supervision. Removing either $\mathcal { L } _ { \mathrm { p c } } \ : \mathrm { o r } \ : \mathcal { L } _ { \mathrm { f c } }$ degrades the restoration quality, and removing both losses causes a clear performance drop, with the two-step PSNR decreasing from 30.58 to 29.78. This suggests that perceptual and geometric regularization are complementary and important for stabilizing sparse correspondence learning and improving aligned guidance aggregation.

## 5 Conclusions

We have presented UGD-FM, a general cross-modal conditional flow matching framework for unaligned RGB-NIR guided denoising. To support efficient few-step inference, we design a stagefocused and inference-consistent training strategy that aligns training with task-critical solver states. We also introduce adaptive sparse guidance propagation with perceptual and geometric auxiliary losses for efficient and reliable large-range cross-modal matching. Extensive experiments demonstrate that UGD-FM achieves state-of-the-art denoising performance with only one or two inference steps.

## Acknowledgments and Disclosure of Funding

This work was supported by the National Natural Science Foundation of China under Grant 62571478.

## References

[1] Matthew Brown and Sabine Süsstrunk. Multi-spectral SIFT for scene category recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 177–184, 2011.

[2] Clément Chadebec, Onur Tasar, Sanjeev Sreetharan, and Benjamin Aubin. LBM: Latent bridge matching for fast image-to-image translation. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 29086–29098, 2025.

[3] Hanting Chen, Yunhe Wang, Tianyu Guo, Chang Xu, Yiping Deng, Zhenhua Liu, Siwei Ma, Chunjing Xu, Chao Xu, and Wen Gao. Pre-trained image processing transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12299–12310, 2021.

[4] Liangyu Chen, Xiaojie Chu, Xiangyu Zhang, and Jian Sun. Simple baselines for image restoration. In European Conference on Computer Vision, pages 17–33, 2022.

[5] Minchong Chen, Xiaoyun Yuan, Junzhe Wan, Jianing Zhang, and Jun Zhang. 3M-TI: Highquality mobile thermal imaging via calibration-free multi-camera cross-modal diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5659–5669, 2026.

[6] Xin Deng and Pier Luigi Dragotti. Deep convolutional neural network for multi-modal image restoration and fusion. IEEE Transactions on Pattern Analysis and Machine Intelligence, 43(10):3333–3348, 2020.

[7] Tobias Fischer, Samuel Rota Bulò, Yung-Hsu Yang, Nikhil Keetha, Lorenzo Porzi, Norman Müller, Katja Schwarz, Jonathon Luiten, Marc Pollefeys, and Peter Kontschieder. FlowR: Flowing from sparse to dense 3D reconstructions. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 27702–27712, 2025.

[8] Yunyi Gao, Lin Gu, Qiankun Liu, and Ying Fu. Object-aware NIR-to-visible translation. In European Conference on Computer Vision, pages 93–109, 2024.

[9] Kaiming He, Jian Sun, and Xiaoou Tang. Guided image filtering. IEEE Transactions on Pattern Analysis and Machine Intelligence, 35(6):1397–1409, 2012.

[10] Shuangping Jin, Bingbing Yu, Minhao Jing, Yi Zhou, Jiajun Liang, and Renhe Ji. DarkVisionNet: Low-light imaging via RGB-NIR fusion with deep inconsistency prior. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 1104–1112, 2022.

[11] Justin Johnson, Alexandre Alahi, and Li Fei-Fei. Perceptual losses for real-time style transfer and super-resolution. In European Conference on Computer Vision, pages 694–711, 2016.

[12] Jinnyeong Kim and Seung-Hwan Baek. Pixel-aligned RGB-NIR stereo imaging and dataset for robot vision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11482–11492, 2025.

[13] Xingyuan Li, Songcheng Du, Yang Zou, HaoYuan Xu, Zhiying Jiang, and Jinyuan Liu. UniFusion: A unified image fusion framework with robust representation and source-aware preservation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 33869–33880, 2026.

[14] Xingyuan Li, Jinyuan Liu, Zhixin Chen, Yang Zou, Long Ma, Xin Fan, and Risheng Liu. Contourlet residual for prompt learning enhanced infrared image super-resolution. In European Conference on Computer Vision, pages 270–288, 2024.

[15] Xingyuan Li, Zirui Wang, Yang Zou, Zhixin Chen, Jun Ma, Zhiying Jiang, Long Ma, and Jinyuan Liu. DifIISR: A diffusion model with gradient guidance for infrared image superresolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7534–7544, 2025.

[16] Xingyuan Li, Haoyuan Xu, Shulin Li, Xiang Chen, Zhiying Jiang, and Jinyuan Liu. DR-Fusion: Drift-resilient temporally consistent infrared-visible video fusion. arXiv preprint arXiv:2605.25775, 2026.

[17] Xingyuan Li, Haoyuan Xu, Xingyue Zhu, Jun Ma, Yang Zou, Zhiying Jiang, and Jinyuan Liu. Uncertainty-aware spatial-frequency registration and fusion for infrared and visible images. arXiv preprint arXiv:2605.13049, 2026.

[18] Yijun Li, Jia-Bin Huang, Narendra Ahuja, and Ming-Hsuan Yang. Deep joint image filtering. In European conference on computer vision, pages 154–169, 2016.

[19] Jingyun Liang, Jiezhang Cao, Guolei Sun, Kai Zhang, Luc Van Gool, and Radu Timofte. SwinIR: Image restoration using Swin transformer. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1833–1844, 2021.

[20] Jinyuan Liu, Xingyuan Li, Zirui Wang, Zhiying Jiang, Wei Zhong, Wei Fan, and Bin Xu. PromptFusion: Harmonized semantic prompt learning for infrared and visible image fusion.

IEEE/CAA Journal ofAutomatica Sinica, 12(3):502–515, 2024.

[21] Jinyuan Liu, Bowei Zhang, Qingyun Mei, Xingyuan Li, Yang Zou, Zhiying Jiang, Long Ma, Risheng Liu, and Xin Fan. DCEvo: Discriminative cross-dimensional evolutionary learning for infrared and visible image fusion. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2226–2235, 2025.

[22] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.

[23] Xiongwei Liu, Zehua Sheng, and Hui-Liang Shen. Frequency-relevant residual learning for multi-modal image denoising. In International Conference on Image Processing, pages 86–90, 2022.

[24] Zhen Liu, Diedong Feng, Hai Jiang, Liaoyuan Zeng, Hao Wang, Chaoyu Feng, Lei Lei, Bing Zeng, and Shuaicheng Liu. RAW-Flow: Advancing RGB-to-RAW image reconstruction with deterministic latent flow matching. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 7431–7439, 2026.

[25] Lars Mescheder, Wei Dong, Shiwei Li, Xuyang Bai, Marcel Santos, Peiyun Hu, Bruno Lecouat, Mingmin Zhen, Amaël Delaunoy, Tian Fang, et al. Sharp monocular view synthesis in less than a second. In International Conference on Learning Representations, pages 24192–24230, 2026.

[26] Jinshan Pan, Jiangxin Dong, Jimmy S Ren, Liang Lin, Jinhui Tang, and Ming-Hsuan Yang. Spatially variant linear representation models for joint filtering. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1702–1711, 2019.

[27] Haina Qin, Wenyang Luo, Libin Wang, Dandan Zheng, Jingdong Chen, Ming Yang, Bing Li, and Weiming Hu. Reversing flow for image restoration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7545–7558, 2025.

[28] Jiangwei Ren, Xingyu Jiang, Zizhuo Li, Dingkang Liang, Xin Zhou, and Xiang Bai. MINIMA: Modality invariant image matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23059–23068, 2025.

[29] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, pages 234–241, 2015.

[30] Zehua Sheng, Xiongwei Liu, Si-Yuan Cao, Hui-Liang Shen, and Huaqi Zhang. Frequencydomain deep guided image denoising. IEEE Transactions on Multimedia, 25:6767–6781, 2022.

[31] Zehua Sheng, Zhu Yu, Xiongwei Liu, Si-Yuan Cao, Yuqi Liu, Hui-Liang Shen, and Huaqi Zhang. Structure aggregation for cross-spectral stereo image guided denoising. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13997–14006, 2023.

[32] Takashi Shibata, Masayuki Tanaka, and Masatoshi Okutomi. Misalignment-robust joint filter for cross-modal image pairs. In Proceedings of the IEEE International Conference on Computer Vision, pages 3295–3304, 2017.

[33] Ukcheol Shin, Jinsun Park, and In So Kweon. Deep depth estimation from thermal image. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1043–1053, 2023.

[34] Darshan Thaker, Abhishek Goyal, and René Vidal. Frequency-guided posterior sampling for diffusion-based image restoration. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12873–12882, 2025.

[35] Hebaixu Wang, Jing Zhang, Haoyang Chen, Haonan Guo, Di Wang, Jiayi Ma, and Bo Du. Residual diffusion bridge model for image restoration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8375–8386, 2026.

[36] Qiang Wang, Yuning Cui, Yawen Li, Yaping Ruan, Ben Zhu, and Wenqi Ren. RFFNet: Towards robust and flexible fusion for low-light image denoising. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 836–845, 2024.

[37] Yuchen Wang, Hongyuan Wang, Lizhi Wang, Xin Wang, Lin Zhu, Wanxuan Lu, and Hua Huang. Complementary advantages: Exploiting cross-field frequency correlation for NIRassisted image denoising. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12679–12689, 2025.

[38] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: From error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4):600– 612, 2004.

[39] Zhendong Wang, Xiaodong Cun, Jianmin Bao, Wengang Zhou, Jianzhuang Liu, and Houqiang Li. Uformer: A general U-shaped transformer for image restoration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17683–17693, 2022.

[40] Zirui Wang, Jiayi Zhang, Tianwei Guan, Yuhan Zhou, Xingyuan Li, Minjing Dong, and Jinyuan Liu. Efficient rectified flow for image fusion. Advances in Neural Information Processing Systems, 2025.

[41] Rongyuan Wu, Tao Yang, Lingchen Sun, Zhengqiang Zhang, Shuai Li, and Lei Zhang. SeeSR: Towards semantics-aware real-world image super-resolution. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 25456–25467, 2024.

[42] Jinhui Xiong, Jian Wang, Wolfgang Heidrich, and Shree Nayar. Seeing in extra darkness using a deep-red flash. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10000–10009, 2021.

[43] Rongjian Xu, Zhilu Zhang, Renlong Wu, and Wangmeng Zuo. NIR-assisted image denoising: A selective fusion approach and a real-world benchmark dataset. IEEE Transactions on Multimedia, 27:2543–2555, 2024.

[44] Shuang Xu, Jiangshe Zhang, Jialin Wang, Kai Sun, Chunxia Zhang, Junmin Liu, and Junying Hu. A model-driven network for guided image denoising. Information Fusion, 85:60–71, 2022.

[45] Zongsheng Yue, Jianyi Wang, and Chen Change Loy. ResShift: Efficient diffusion model for image super-resolution by residual shifting. Advances in neural information processing systems, 36:13294–13307, 2023.

[46] Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang. Restormer: Efficient transformer for high-resolution image restoration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5728–5739, 2022.

[47] Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao. Multi-stage progressive image restoration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14821–14831, 2021.

[48] Jiale Zhang, Yulun Zhang, Jinjin Gu, Jiahua Dong, Linghe Kong, and Xiaokang Yang. Xformer: Hybrid X-shaped transformer for image denoising. In International Conference on Learning Representations, 2024.

[49] Kai Zhang, Wangmeng Zuo, Yunjin Chen, Deyu Meng, and Lei Zhang. Beyond a Gaussian denoiser: Residual learning of deep CNN for image denoising. IEEE Transactions on Image Processing, 26(7):3142–3155, 2017.

[50] Runmin Zhang, Jialiang Wang, Si-Yuan Cao, Zhu Yu, Junchen Yu, Guangyi Zhang, and Hui-Liang Shen. Rethinking unsupervised cross-modal flow estimation: Learning from decoupled optimization and consistency constraint. In International Conference on Learning Representations, 2026.

[51] Runmin Zhang, Zhu Yu, Zehua Sheng, Jiacheng Ying, Si-Yuan Cao, Shu-Jie Chen, Bailin Yang, Junwei Li, and Hui-Liang Shen. SGDFormer: One-stage transformer-based architecture for cross-spectral stereo image guided denoising. Information Fusion, 113:102603, 2025.

[52] Yingkai Zhang, Zeqiang Lai, Tao Zhang, Ying Fu, and Chenghu Zhou. Unaligned RGB guided hyperspectral image super-resolution with spatial-spectral concordance. International Journal ofComputer Vision, 133(9):6590–6610, 2025.

[53] Tiancheng Zhi, Bernardo R Pires, Martial Hebert, and Srinivasa G Narasimhan. Deep materialaware cross-spectral stereo matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1916–1925, 2018.

[54] Shili Zhou, Weimin Tan, and Bo Yan. Promoting single-modal optical flow network for diverse cross-modal flow estimation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 3562–3570, 2022.

[55] Yuanzhi Zhu, Kai Zhang, Jingyun Liang, Jiezhang Cao, Bihan Wen, Radu Timofte, and Luc Van Gool. Denoising diffusion models for plug-and-play image restoration. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1219–1229, 2023.

## Appendix

## A Additional Experimental Details

## A.1 Details of Datasets

We evaluate UGD-FM on the RGB-NIR Scene [1], FANVID [8], and MS2 [33] datasets. A statistical summary of these datasets is provided in Table 7. Specifically, FANVID contains 453 RGB images and 3624 RGB-NIR pairs with an original image resolution of 576 × 768 (H × W), while RGB-NIR Scene provides 434 RGB images and 3472 RGB-NIR pairs with original resolutions ranging from $6 2 0 \times 5 0 2$ to $7 6 8 \times 1 0 2 4$ . MS2 consists of 1229 RGB-NIR pairs captured from real-world multicamera imaging systems with an original resolution of 320 × 1022. We use 3008/616, 3112/360, and 981/248 image pairs for training/testing on FANVID, RGB-NIR Scene, and MS2, respectively.

RGB-NIR Scene and FANVID originally provide spatially aligned RGB-NIR image pairs, whereas MS2 is captured using real multi-camera imaging systems and naturally contains viewpoint discrepancies between modalities. Therefore, for RGB-NIR Scene and FANVID, we further construct misaligned RGB-NIR pairs through a controlled parallax simulation strategy.

Specifically, starting from the aligned RGB-NIR pairs, the RGB image is retained as the reference target view, while the NIR image is independently reconstructed into a 3D Gaussian representation using SHARP [25]. Based on the reconstructed representation, eight neighboring virtual camera viewpoints are sampled around the original camera pose to render new NIR views, as illustrated in Fig. 4. We place the virtual cameras on a circular trajectory whose radius is normalized so that the nominal translational parallax at a representative near-scene depth is approximately $x \%$ of the image diagonal. We use $x = 6$ for FANVID and $x = 8$ for RGB-NIR Scene. This simulation protocol is designed to approximate practical dual-camera acquisition settings, where RGB and NIR sensors observe the same scene from different viewpoints. The rendered NIR images are subsequently paired with the original RGB images to form misaligned RGB-NIR image pairs with realistic 3D parallax.

Table 7 summarizes 2D displacement magnitudes. For the synthetic FANVID and RGB-NIR Scene datasets, we obtain ground-truth flow from 3DGS rendering. For the real-world MS2 dataset, we estimate flow using CrossModalRAFT [54]. The P95 displacements are 45.25, 76.25, and 50.75 pixels, respectively.

Table 7: Dataset characteristics and statistics of 2D displacement magnitudes.
<table><tr><td>Characteristic</td><td>FANVID [8]</td><td>RGB-NIR Scene [1]</td><td>MS2 [33]</td></tr><tr><td># RGB images # RGB-NIR pairs Resolution  $( H \times W )$  # Training pairs</td><td>453 3624 576 × 768 3008</td><td>434 3472  $6 2 0 \times 5 0 2 { - } 7 6 8 \times 1 0 2 4$  3112</td><td>1229 1229 320 × 1022 981</td></tr><tr><td># Testing pairs Mean displacement (px) Std. displacement (px) P75 displacement (px)</td><td>616 22.10 13.46 32.25</td><td>360 36.70 23.16 54.25 69.25</td><td>248 29.64 12.28 36.25 43.75</td></tr></table>

## A.2 Implementation and Experimental Details

Different variants instantiate Uformer, Restormer, or NAFNet as backbone blocks while sharing the same overall architecture. The base channel width is set to $C = 3 2$ , and only a single block is employed at each scale. The maximum network depth is set to $L = 4$ . The Top-K parameter is fixed to $K = 5$ . Inference under the flow matching framework is performed using two steps at $t = \tau _ { \mathrm { h c } }$ and $t = \tau _ { \mathrm { r c } } .$ , where $\tau _ { \mathrm { h c } } = 0$ and $\tau _ { \mathrm { r c } } = 0 . 9$

We optimize the network using the AdamW optimizer with a weight decay of 0.01, an initial learning rate of $3 \times 1 0 ^ { - 4 }$ , and a OneCycle learning rate scheduler. Training is performed for 80k iterations using a batch size of 8. The warm-up stage is conducted for $N _ { \mathrm { w a r m } } = 5 0 \mathrm { k \Omega }$ iterations. The auxiliary loss weights are set to $\lambda _ { \mathrm { p c } } = 0 . 0 5$ and $\lambda _ { \mathrm { f c } } = 0 . 0 5$ , while the truncated Gaussian distribution weights are set to $\lambda _ { \mathrm { h c } } = 0 . 5$ and $\lambda _ { \mathrm { { r c } } } = 0 . 5$ . In addition, the parameter $\sigma _ { s }$ is set to 0.05.

![](images/4f21b2a36622c344725ca2b4720a9c9c6fe8a66cee38c07386875a8d6dd286df.jpg)  
Figure 4: Visualization of the simulation process for constructing unaligned RGB-NIR datasets.

For fair comparison, all approaches are trained using the same training set, and the training iterations are also the same. All experiments are conducted on NVIDIA RTX 4090 GPUs.

## B More Quantitative and Qualitative Results

## B.1 Ablation on Hyperparameters

In Table 8, we provide detailed ablations on hyperparameters including the displacement search radius D, Top-K candidate selection, inference time anchors $\tau _ { \mathrm { h c } }$ and $\tau _ { \mathrm { r c } }$ , time sampling standard deviation $\sigma _ { s } .$ , and warm-up length $N _ { \mathrm { w a r m } }$

Displacement Search Radius D. We vary the displacement search radius from 32 to 96. A small search range may fail to cover the true displacement range, causing a performance drop, while increasing $D$ generally improves PSNR by providing a broader search range.

Top-K Candidate Selection. We vary K from 1 to 7. A small K may discard plausible correspondences, causing a clear performance drop. Increasing K improves performance, with only a small additional gain from 5 to 7. We use $K = 5$ as the default.

Inference Time Anchors $\tau _ { \mathrm { h c } }$ and $\tau _ { \mathrm { r c } } .$ Moving $\tau _ { \mathrm { h c } }$ from 0 to 0.1 reduces performance, supporting the use of the initial noisy state as the highly corrupted-stage anchor. For $\tau _ { \mathrm { r c } } .$ , an early anchor provides limited separation between the highly corrupted and relatively cleaner stages, resulting in smaller two-step gains. When the anchor clearly separates these stages, as with $\tau _ { \mathrm { r c } } = 0 . 7$ or 0.9, two-step inference yields clear improvements. Their comparable results indicate that the improvement does not require a precisely tuned anchor. We use $\tau _ { \mathrm { r c } } = 0 . 9$ as the default.

Time Sampling Standard Deviation $\sigma _ { s }$ . The parameter $\sigma _ { s }$ controls the sampling concentration around the time anchors. Similar results at 0.025 and 0.05 show that the method does not require careful tuning within this range. However, an excessively large value such as 0.2 disperses the training distribution away from the anchors and degrades performance.

Warm-up Length $N _ { \mathrm { w a r m } }$ . We vary the warm-up length from 40k to 60k iterations. The results remain similar across these settings, indicating limited sensitivity to the warm-up length within the tested range. We use $N _ { \mathrm { w a r m } } = 5 0 \mathrm { k \Omega }$ as the default.

## B.2 Sensitivity to Flow-Label Quality

To evaluate the relationship between denoising performance and the quality of flow labels, we compare three types of supervision including synthetic ground-truth flow, estimated flow, and estimated flow with synthetic perturbations on the FANVID dataset. The ground-truth flow is derived from the known geometry and camera transformations used in 3DGS rendering. For the perturbed flow, we apply Gaussian blur and additive Gaussian noise to the estimated flow, increasing its mean endpoint error (EPE) by up to 15 pixels. As shown in Table 9, replacing estimated flow with ground-truth flow improves the 2-step results by only 0.03 dB PSNR, 0.0007 SSIM, and 0.0036 LPIPS. Conversely, increasing the mean EPE by 15 pixels decreases performance by only 0.05 dB PSNR, 0.0020 SSIM, and 0.0052 LPIPS. These results indicate that denoising performance is not highly sensitive to flow-label accuracy. Flow supervision mainly serves as an auxiliary signal for learning sparse correspondence attention, rather than being a key determinant of restoration quality.

Table 8: Hyperparameter ablations on FANVID [8] under Poisson-Gaussian noise $( \alpha = 0 . 0 3 ,$ σ = 0.3).
<table><tr><td rowspan="2">Hyperparameter</td><td rowspan="2">Setting</td><td colspan="3">1-step</td><td colspan="3">2-step</td></tr><tr><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM ↑</td><td>LPIPS ↓</td></tr><tr><td rowspan="5">Search range D</td><td>32</td><td>30.19</td><td>0.8377</td><td>0.2241</td><td>30.42</td><td>0.8424</td><td>0.2179</td></tr><tr><td>48</td><td>30.29</td><td>0.8396</td><td>0.2167</td><td>30.49</td><td>0.8471</td><td>0.2105</td></tr><tr><td>64</td><td>30.32</td><td>0.8413</td><td>0.2179</td><td>30.56</td><td>0.8527</td><td>0.2097</td></tr><tr><td>80 (default)</td><td>30.29</td><td>0.8401</td><td>0.2215</td><td>30.58</td><td>0.8506</td><td>0.2121</td></tr><tr><td>96</td><td>30.31</td><td>0.8403</td><td>0.2221</td><td>30.61</td><td>0.8507</td><td>0.2128</td></tr><tr><td rowspan="4">Top-K selection</td><td>1</td><td>29.89</td><td>0.8248</td><td>0.2449</td><td>30.19</td><td>0.8364</td><td>0.2345</td></tr><tr><td> $^ 3$ </td><td>30.26</td><td>0.8382</td><td>0.2264</td><td>30.55</td><td>0.8492</td><td>0.2159</td></tr><tr><td>5 (default)</td><td>30.29</td><td>0.8401</td><td>0.2215</td><td>30.58</td><td>0.8506</td><td>0.2121</td></tr><tr><td> $^ 7$ </td><td>30.30</td><td>0.8417</td><td>0.2185</td><td>30.59</td><td>0.8520</td><td>0.2110</td></tr><tr><td rowspan="2">Time anchor  $\tau _ { \mathrm { h c } }$ </td><td>0 (default)</td><td>30.29</td><td>0.8401</td><td>0.2215</td><td>30.58</td><td>0.8506</td><td>0.2121</td></tr><tr><td>0.1</td><td>29.85</td><td>0.8266</td><td>0.2509</td><td>30.21</td><td>0.8391</td><td>0.2358</td></tr><tr><td rowspan="4">Time anchor  $\tau _ { \mathrm { r c } }$ </td><td>0.3</td><td>30.34</td><td>0.8432</td><td>0.2235</td><td>30.39</td><td>0.8457</td><td>0.2216</td></tr><tr><td>0.5</td><td>30.33</td><td>0.8412</td><td>0.2192</td><td>30.49</td><td>0.8459</td><td>0.2191</td></tr><tr><td>0.7</td><td>30.33</td><td>0.8409</td><td>0.2237</td><td>30.56</td><td>0.8495</td><td>0.2133</td></tr><tr><td>0.9 (default)</td><td>30.29</td><td>0.8401</td><td>0.2215</td><td>30.58</td><td>0.8506</td><td>0.2121</td></tr><tr><td rowspan="3">Sampling std.  $\sigma _ { s }$ </td><td>0.025</td><td>30.33</td><td>0.8414</td><td>0.2220</td><td>30.62</td><td>0.8524</td><td>0.2139</td></tr><tr><td>0.05 (default)</td><td>30.29</td><td>0.8401</td><td>0.2215</td><td>30.58</td><td>0.8506</td><td>0.2121</td></tr><tr><td>0.2</td><td>30.27</td><td>0.8384</td><td>0.2257</td><td>30.46</td><td>0.8453</td><td>0.2192</td></tr><tr><td rowspan="3">Warm-up  $N _ { \mathrm { w a r m } }$ </td><td>40k</td><td>30.31</td><td>0.8406</td><td>0.2211</td><td>30.56</td><td>0.8512</td><td>0.2110</td></tr><tr><td>50k (default)</td><td>30.29</td><td>0.8401</td><td>0.2215</td><td>30.58</td><td>0.8506</td><td>0.2121</td></tr><tr><td>60k</td><td>30.29</td><td>0.8392</td><td>0.2224</td><td>30.55</td><td>0.8497</td><td>0.2123</td></tr></table>

Table 9: Effect of flow-label quality on FANVID [8] under Poisson-Gaussian noise $( \alpha = 0 . 0 3 \mathrm { { . } }$ $\sigma = 0 . 3 )$
<table><tr><td rowspan="2">Flow label</td><td rowspan="2">Added mean EPE</td><td colspan="3">1-step</td><td colspan="3">2-step</td></tr><tr><td>PSNR↑</td><td>SSIM ↑</td><td>LPIPS↓</td><td>PSNR ↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td rowspan="2">Ground-truth flow Estimated flow (default)</td><td>一</td><td>30.31</td><td>0.8422</td><td>0.2199</td><td>30.61</td><td>0.8513</td><td>0.2085</td></tr><tr><td>一</td><td>30.29</td><td>0.8401</td><td>0.2215</td><td>30.58</td><td>0.8506</td><td>0.2121</td></tr><tr><td rowspan="3">Perturbed estimated flow</td><td>5px</td><td>30.27</td><td>0.8393</td><td>0.2232</td><td>30.56</td><td>0.8497</td><td>0.2140</td></tr><tr><td>10 px</td><td>30.25</td><td>0.8382</td><td>0.2249</td><td>30.53</td><td>0.8487</td><td>0.2164</td></tr><tr><td>15 px</td><td>30.25</td><td>0.8383</td><td>0.2261</td><td>30.53</td><td>0.8486</td><td>0.2173</td></tr></table>

## B.3 Number of Inference Steps

Table 10 compares standard FM with 1, 2, 4, and 8 solver steps and UGD-FM with one-step or two-step inference. Standard FM yields consistent gains from 1 to 4 solver steps. However, at 8 steps, PSNR and SSIM drop, while LPIPS improves, indicating that additional steps do not consistently improve restoration quality. We attribute the fidelity degradation to error accumulation during iterative forward-Euler updates, which may cause global blur and detail loss. In contrast, UGD-FM employs stage-focused and inference-consistent training for efficient inference, achieving SOTA performance with only 2 steps.

Table 10: Effect of the number of inference steps on FANVID [8] under Poisson-Gaussian noise $( \alpha = 0 . 0 3 , \sigma = 0 . 3 )$
<table><tr><td>Method</td><td>Steps</td><td>PSNR ↑</td><td>SSIM ↑ LPIPS ↓</td></tr><tr><td rowspan="5">Standard FM</td><td>1</td><td>30.28</td><td>0.8389 0.2357</td></tr><tr><td>2</td><td>30.34 0.8401</td><td>0.2312</td></tr><tr><td>4</td><td>30.36 0.8422</td><td>0.2291</td></tr><tr><td>8</td><td>30.02</td><td>0.8340 0.2265</td></tr><tr><td>1</td><td>30.29</td><td>0.8401 0.2215</td></tr><tr><td>UGD-FM</td><td>2</td><td>30.58</td><td>0.8506 0.2121</td></tr></table>

Table 11: Comparison under different misalignment levels on FANVID [8] with Poisson–Gaussian noise $( \alpha = 0 . 0 { \overset { \cdot } { 3 } } , \sigma = 0 . 3 ) .$
<table><tr><td>Misalignment</td><td>Method</td><td>PSNR ↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td rowspan="4">Aligned</td><td>Restormer [46]</td><td>29.20</td><td>0.7981</td><td>0.3497</td></tr><tr><td>NAID [43]</td><td>31.93</td><td>0.8961</td><td>0.1616</td></tr><tr><td>UGD-FM (NAFNet, 1-step)</td><td>31.75</td><td>0.8898</td><td>0.1771</td></tr><tr><td>UGD-FM (NAFNet, 2-step)</td><td>32.06</td><td>0.8984</td><td>0.1535</td></tr><tr><td rowspan="3"> $x = 2$ </td><td>NAID [43]</td><td>29.22</td><td>0.8071</td><td>0.2919</td></tr><tr><td>UGD-FM (NAFNet, 1-step)</td><td>30.50</td><td>0.8478</td><td>0.2129</td></tr><tr><td>UGD-FM (NAFNet, 2-step)</td><td>30.86</td><td>0.8597</td><td>0.1988</td></tr><tr><td rowspan="3">x = 4</td><td>NAID [43]</td><td>29.05</td><td>0.7970</td><td>0.3324</td></tr><tr><td>UGD-FM (NAFNet, 1-step)</td><td>30.29</td><td>0.8409</td><td>0.2250</td></tr><tr><td>UGD-FM (NAFNet, 2-step)</td><td>30.62</td><td>0.8523</td><td>0.2119</td></tr><tr><td rowspan="3"> $x = 6$ </td><td>NAID [43]</td><td>28.99</td><td>0.7933</td><td>0.3540</td></tr><tr><td>UGD-FM (NAFNet, 1-step)</td><td>30.12</td><td>0.8351</td><td>0.2351</td></tr><tr><td>UGD-FM (NAFNet, 2-step)</td><td>30.42</td><td>0.8458</td><td>0.2237</td></tr><tr><td rowspan="3"> $x = 8$ </td><td>NAID [43]</td><td>28.97</td><td>0.7922</td><td>0.3673</td></tr><tr><td>UGD-FM (NAFNet, 1-step)</td><td>29.97</td><td>0.8297</td><td>0.2448</td></tr><tr><td>UGD-FM (NAFNet, 2-step)</td><td>30.24</td><td>0.8394</td><td>0.2352</td></tr></table>

## B.4 Experiments on Misalignment Magnitudes

We vary the misalignment magnitudes on FANVID from aligned guidance to $x = 2 , 4 , 6 , 8 .$ , using the same virtual-camera displacement definition as in Appendix A.1. Larger x indicates stronger parallax. As shown in Table 11, guided denoising methods like NAID [43] perform well only with aligned guidance or mild misalignment. In contrast, our UGD-FM maintains strong performance as the misalignment increases, demonstrating its robustness to unaligned guidance.

## B.5 More Evaluation on Real-World Datasets

We conduct additional experiments on RNS [12], which is a real-world dataset captured using an RGB-NIR dual-camera system. We use the original images without epipolar rectification, with RGB images from the right camera as targets and NIR images from the left camera as guidance. Table 12 reports results under Poisson-Gaussian noise $( \alpha = 0 . 0 3 , \sigma = 0 . 3 )$ . UGD-FM achieves the best performance, demonstrating its generalizability to real-world data.

Table 12: Quantitative comparison on the RNS [12] dataset under Poisson-Gaussian noise (α = $0 . 0 3 , \sigma = 0 . 3 )$ . The best and second-best results are highlighted in bold and underlined, respectively.
<table><tr><td>Method</td><td>PSNR ↑</td><td>SSIM↑</td><td>LPIPS ↓</td></tr><tr><td>Restormer [46] RDBM [35]</td><td>28.20</td><td>0.7761 0.7678</td><td>0.3793 0.3377</td></tr><tr><td>NAID [43]</td><td>28.38 28.56</td><td>0.7997</td><td>0.3137</td></tr><tr><td>SGDFormer [51] RFfusion [40]</td><td>28.39 27.76</td><td>0.7863 0.7626</td><td>0.3307 0.3144</td></tr><tr><td>3M-TI [5]</td><td>28.62</td><td></td><td></td></tr><tr><td></td><td></td><td>0.8011</td><td>0.3024</td></tr><tr><td>UGD-FM (NAFNet, 1-step) UGD-FM (NAFNet, 2-step)</td><td>28.96 29.10</td><td>0.8110</td><td>0.2505</td></tr></table>

Table 13: Inference efficiency at input resolution 768 × 1024 on an NVIDIA RTX 4090 GPU.
<table><tr><td>Method</td><td>Time (ms)</td><td>Peak memory (GiB)</td></tr><tr><td>MPRNet [47]</td><td>437</td><td>6.3</td></tr><tr><td>NAFNet [4]</td><td>76</td><td>0.9</td></tr><tr><td>Uformer [39]</td><td>171</td><td>4.7</td></tr><tr><td>Restormer [46] Xformer [48]</td><td>676 1390</td><td>6.7 7.8</td></tr><tr><td>RDBM [35]</td><td>3431</td><td>9.3</td></tr><tr><td>CUNet [6]</td><td>162</td><td>1.9</td></tr><tr><td>DVN [10]</td><td>159</td><td>3.9</td></tr><tr><td>RFFNet [36]</td><td>164</td><td>11.4</td></tr><tr><td>NAID [43]</td><td>120</td><td>1.5</td></tr><tr><td>SANet [31]</td><td>194</td><td>4.4</td></tr><tr><td>RFfusion [40]</td><td>841</td><td>20.7</td></tr><tr><td>3M-TI [5]</td><td>654</td><td>9.7</td></tr><tr><td>UGD-FM (NAFNet, 1-step)</td><td></td><td></td></tr><tr><td></td><td>250</td><td>5.1</td></tr><tr><td>UGD-FM (NAFNet, 2-step)</td><td>487</td><td>5.1</td></tr><tr><td>UGD-FM (Uformer, 1-step)</td><td>441</td><td>5.0</td></tr><tr><td>UGD-FM (Uformer, 2-step)</td><td>829</td><td>5.0</td></tr><tr><td>UGD-FM (Restormer, 1-step)</td><td>368</td><td>5.0</td></tr><tr><td>UGD-FM (Restormer, 2-step)</td><td>694</td><td>5.0</td></tr></table>

## B.6 Inference Efficiency

Table 13 reports inference time and peak memory usage at an input resolution of 768 × 1024 on a single NVIDIA RTX 4090 GPU. UGD-FM achieves state-of-the-art performance without incurring prohibitive memory or inference-time overhead.

## B.7 Visualization of the Mutually Reinforcing Process

Figure 5 demonstrates the mutually reinforcing process between image restoration and guidance aggregation. The visualized features correspond to the guidance features of the finest level of decoding blocks. As the flow matching trajectory evolves, the guidance features become more structured and coherent.

## B.8 More Qualitative Comparisons

We provide more qualitative comparisons of different approaches in Fig. 6. Single image denoising approaches generally cannot preserve structures and textures well. Among all guided denoising methods, our UGD-FM produces denoised images with more salient structures and details.

![](images/0b5d08b15952142af2a3dca8647dc8d6e4d1a86d49d46278382631daba284050.jpg)  
Figure 5: Visualization of the mutually reinforcing process between image restoration and guidance aggregation.

## C Limitations

UGD-FM remains challenged by severe 3D parallax, where occlusions and incomplete cross-view observations can make local guidance ambiguous or unreliable, as shown in Fig. 7. Joint restoration and cross-modal alignment also introduce computational overhead. Despite these limitations, we believe UGD-FM provides a practical solution for unaligned guided image denoising.

## D Broader Impacts

UGD-FM can benefit low-light photography through image restoration using cross-modal guidance. However, its use in unauthorized surveillance may raise privacy concerns, and restored images may contain plausible but inaccurate details. Deployment in safety-critical applications therefore requires careful validation and further assessment of uncertainty and reliability.

Restormer  
![](images/11e60df8cb7e6ae65c03d78817a149823e76b54b651ced63cfd25f8b78d70af0.jpg)  
Noisy target image (RGB)

![](images/4f3d71e91bf5d640b0114f3004c87adbf980750c214c0a78d5f7d80bd419a240.jpg)  
Guidance image (NIR)

![](images/b71285de95ed9d828887d354e28d29eaf4817a4e88e61f20fc44e1b19273885a.jpg)

![](images/82404da5ea21adb7ceeabe6e2760f72414111e1d7cf9a07a5652c7f116f1e011.jpg)  
DVN  
RFFNet

![](images/1b4a804ff082d1a7c8554fce340b28703c52f322d2c698f72f3873a23a722998.jpg)

![](images/490a14246d7591f1612c4026b283d29b89ef8508e72ab27b02396c12d0b7297b.jpg)  
NAID

Restormer  
![](images/7ece1439991f78ec98fbd952420076cce309313fe768bb6d490bc0f517120b62.jpg)  
Xformer

![](images/2443a81d1e2eba7b63a4efde80e3ea7078213e6e21d7b97421d471842dc3f894.jpg)  
UGD-FM (Restormer, 1-step)

![](images/0078794ea6701b214a45838afab5f6ab6105d5dd6b789e9a36bbf3c766b3bb18.jpg)  
UGD-FM (Restormer, 2-step)

![](images/86db12f173f39de03c992825c0089a2d333285bf6776ca2471960048ebe03368.jpg)  
Ground truth  
Noisy target image (RGB)

![](images/0392361befd0695f129cf5a78e71c321731554d657ef7df5f2f05eb87cdcf292.jpg)  
Guidance image (NIR)

![](images/84f8c3e29c24280241cceff9ba563e012193c91b95909d23a5337bb5b6906064.jpg)

![](images/362c0f4b9e061265aa62a78f51c7972eff3256cf4f0d82427d1a8f77b21e929b.jpg)

![](images/bc67a271b9c61e68e61cc59f18c20be6795653aaa1e7c549059c2408acfa100a.jpg)

![](images/07b5230c5a2c61d098adf0a498e6b878e395db5f977eebf370f48bc42ead538d.jpg)  
DVN

![](images/624a575c1d5c8e6bad2d50e4861ef27a65c07353de6e67c81fb90f72c96aec33.jpg)

RFFNet  
![](images/deabe956748e5eec6c1608595661f500b2817938bfed2dcc429776ae80798ebb.jpg)  
Xformer  
UGD-FM (Restormer, 1-step)

![](images/901c226302b4feed0b3d325d07a161f9b19d5d22a3ab0642ea96367cb586d4aa.jpg)  
UGD-FM (Restormer, 2-step)

NAID  
![](images/77705ec0daea6aad5177dd4276e5c74978f8a38f3e0f5b979001eb56f5908928.jpg)

![](images/ab6d9eb711dcc75b7acf16d454822143942b7de1975dbed94834c7bd6a412c21.jpg)  
Ground truth  
Noisy target image (RGB)

![](images/ce275f201be94f6e8a0e9a0660b2ca4b1df044219b5f4c937e06df559a6445a2.jpg)

![](images/33e697ed24ea57cc4b2f2afec35d1579cb285b8d58756ff25ae60caa3f9bbdf3.jpg)  
Guidance image (NIR)

![](images/0eac9042e8cc774b11bce8cc22eed3e1721f315ef343c442f7dd6e2f4680b267.jpg)  
Restormer

![](images/e5b546492e8aa43de07f7d1d0f6eaaff6dea4a01c7ebaf3f0cbbb154982775df.jpg)

![](images/546cf83f928088fc559b859754ac7e62f9730893bbb3720ec9dd456d9367802e.jpg)  
DVN  
Xformer

RFFNet  
![](images/801ecb5c612b200f326c74140ffcc1d993127445d91dbb74a1f39794c1dabaae.jpg)

![](images/38dc825bfa1c49ca8db0544dfa9ce1cfbf3d234590ac464c6b7ad78690dd29ee.jpg)  
UGD-FM (Restormer, 1-step)

![](images/ebd44ddaf1d3ac1d686aeb9e099711f606c40bacb11a93426f6aeea312d4b7d2.jpg)  
NAID  
UGD-FM (Restormer, 2-step)

![](images/0aff279a8b77346f6ecadba10ad1a0cf58b8aa7b3fe77ed904d9fc28cfd4b6a2.jpg)  
Ground truth

Figure 6: More qualitative comparison of image denoising results across different approaches.  
![](images/83344d481561c6f19ddb9da3e79030bb496f8059e83adcfd170c9ab5a358b984.jpg)  
Noisy target image (RGB)

![](images/7cb11d7f9dbc15c85c9cf40f390ea18208d10e8e0d50a8201029251b7868536c.jpg)  
Ground truth

![](images/cf43ced01a23bb6871bc8a29edf4317b4568e8b5f37a9f8235782df17eaa6906.jpg)  
Guidance image view 1 (NIR)

![](images/4a8d22ffc64ac30ea85d312e844de5004b970e095f98b6bfaa316bcc37c24c76.jpg)  
Guidance image view 2 (NIR)

![](images/9310dfaad62bc58be1bed12fe4e8dfab38cf528b69c5447252cd8f1341487a08.jpg)  
Aligned guidance feature 1

![](images/cdd7ec516d0bf81cce6f679486fc398cef02e328c3239c9d287b4710f459d6b2.jpg)  
Aligned guidance feature 2

![](images/e68780a7ef065ae7280bb66e961fdabf2f3f79f7ddf5233083321e214edb9a12.jpg)

Restored image 1  
![](images/f299d78b146c026e09e4b1a6f49cd99cd024a256e0cf3793572bb8533fb5c07a.jpg)  
Restored image 2

Figure 7: Impact of parallax-induced occlusion on guided denoising. The top and bottom rows present visualizations of the same sample under different parallax conditions. Red and green boxes highlight how occlusion affects feature alignment and reconstruction quality.