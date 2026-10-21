# Test-time Adaptation for Point Cloud Denoising under Unseen Noise

Kun Jiang<sup>a</sup>, Shida Fu<sup>a</sup>, Fanhao Yi<sup>a</sup>, Weiqing Li<sup>b</sup>, Zhiyong Su<sup>a,∗</sup>

<sup>a</sup>School ofAutomation, Nanjing University ofScience and Technology, Nanjing, Jiangsu Province, 210094, P.R. China <sup>b</sup>School of Computer Science and Engineering, Nanjing University of Science and Technology, Nanjing, Jiangsu Province, 210094, P.R. China

## Abstract

Although deep learning-based point cloud denoising methods have demonstrated impressive results in recent years, their efectiveness usually degrades substantially in real-world scenarios where the noise distribution of test point clouds deviates from that of the training set. To address this challenge, we propose a test-time adaptive denoising (TTAD) framework that enables existing denoising approaches to adapt to unknown noise patterns at test time without relying on additional supervision or retraining from source data. Taking a pre-trained denoising model and a test noisy point cloud as input, TTAD first employs a residual-driven noise modeling (RDNM) module to estimate the sample-specific noise distribution through analyzing the deviation between the noisy input and its pseudo-clean version denoised by the initial model. Then, a noise-aware contrastive adaptation (NACA) module is designed to build<sup>c</sup> sample-specific noisy–noisier test-time training pair from the denoised point cloud, and update the model parameters over a few adaptation steps per sample at test time through contrastive learning. Furthermore, TTAD naturally extends to a continual TTAD (CTTAD) setting, facilitating eficient progressive refinement across sequential test samples with consistent noise distributions, while reducing the computational cost per sample. Extensive experiments on both synthetic and real-world datasets demonstrate that our approach significantly improves the generalization ability of existing denoising models under unseen noise, achieving superior performance across multiple evaluation metrics. The source code is released at https://github.com/zuowenzhitiao/TTAD.

Keywords: Point cloud denoising, Test-time adaptation, Domain shift, Contrastive learning<sup>.</sup>

## 1. Introduction

Point cloud is a fundamental representation of 3D geometric data, with widespread applications in autonomous driving   
[1, 2], industrial inspection [3], digital twins [4], etc. How-4   
ever, in practice, they are often corrupted by noise due to sensor<sup>7</sup>   
limitations, environmental interference, or pose estimation er-<sup>2</sup>   
rors, significantly impairing downstream tasks such as surface0 reconstruction, registration, and object recognition [5, 6, 7].   
Recent advances in deep learning-based point cloud denoising<sup>6</sup> have demonstrated promising results [8, 9, 10]. These methods are typically trained on synthetic datasets, either in a su-<sup>v</sup>   
pervised [9, 10, 11, 12, 13, 14, 15] or an unsupervised fashion [16, 17, 18]. They achieve state-of-the-art performance when   
the test noisy point clouds (target) and the training point clouds<sup>a</sup> (source) belong to the same distribution. However, the performance of these methods is often substantially degraded at test time when there exists a noise distribution gap between the training and test point clouds, which is termed "domain shift" problem, as illustrated in Figure 1.

Test-time adaptation (TTA) [19] has emerged as a promising solution to improve model generalization against domain shift through acclimating a trained model from the source domain to unlabeled test data during the test phase. Recently, some pioneering studies have explored the TTA technique for image denoising [20, 21, 22]. These methods typically introduce auxiliary self-supervised objectives at test time, enabling adaptation using only unlabeled samples. For example, Lidia [21] performs test-time adaptation by injecting synthetic noise into the denoised output and optimizing a denoising loss. SS-TTA [22] employs Gaussian noise to create sample pairs for contrastive learning. However, they all rely on sample-agnostic and synthetic noises (e.g., Gaussian noise), which may not accurately reflect the corruption patterns present in real-world test data. This mismatch often leads to suboptimal adaptation, especially in denoising scenarios where noise characteristics can be highly diverse, sample-specific, and structurally complex. Therefore, exploring the application of TTA to point cloud denoising represents a critical yet challenging research direction for improving generalization ability of current point cloud denoising methods, particularly in real-world noisy point cloud denoising scenarios.

![](images/39d26ac4f14c76973537c960482bd00dd82a01977c84a63454aa48152163296c.jpg)  
(a)

![](images/7a75ab9982e5b7cf4d9a7180bb22a2e30c99661368eaa54930cac8fe011c0fac.jpg)

![](images/bd461a25cc70aa0d66ac2be189256083c0c0f40454e8235f816b2c7d8f43b1fc.jpg)  
(b)  
(c)  
Figure 1: Illustration of existing point cloud denoising methods and the proposed test-time adaptive denoising (TTAD) framework. The source model (PointFilter [13]) f<sub>θ</sub> is trained on synthetic datasets with Gaussian noise. f<sub>θ</sub> performs well on noisy point clouds with seen (Gaussian) noise (a), but fails under unseen noise (b). The proposed TTAD framework significantly enhances the denoising performance of f<sub>θ</sub> by adapting it to unseen noise distributions during test time (c).

In this paper, we propose a test-time adaptation framework termed test-time adaptive denoising (TTAD), which empowers a pre-trained point cloud denoising model to dynamically adapt to unseen noise during the test phase, without access to clean supervision or source-domain data. Given a pre-trained point cloud denoising model, TTAD first introduces a residual-driven noise modeling (RDNM) module to capture sample-specific noise pattern in an unsupervised manner by estimating the error field between the noisy input and its pseudo-clean prediction. Then, a noise-aware contrastive adaptation (NACA) module is designed to update the parameters of the pre-trained denoising model over a few adaptation steps per sample at test time in a self-supervised manner. NACA achieves adaptation by constructing sample-specific noisy–noisier input pair and enforcing consistency via contrastive learning. To further stabilize the adaptation process, exponential moving average (EMA) is incorporated to prevent overfitting to individual noise pattern. Finally, our TTAD framework allows the pre-trained denoising model to achieve sample-wise test-time adaptation without network modifications. Furthermore, for scenarios where point clouds encounter noise distributions with consistent characteristics, we extend our TTAD to a continual TTAD (CTTAD) setting, where the denoising model incrementally evolves across test samples by reusing previously adapted parameters. This formulation not only reduces computational overhead by avoiding repeated adaptation for each input, but also improves stability through gradual knowledge accumulation across test samples. The key contributions of this paper are summarized as follows:

• We present a systematic study of test-time adaptation for point cloud denoising under unseen noise. To this end, we propose a general framework, termed TTAD, which allows pre-trained denoising models to achieve sample-wise testtime adaptation under unseen noise during the test phase. The TTAD framework can be seamlessly integrated with a wide range of existing point cloud denoising methods, without altering their original network architectures.

• We propose a noise-aware contrastive adaptation module that performs self-supervised parameter updates on the pre-trained denoising model during test time via contrastive learning. Specifically, it leverages a residualdriven noise modeling mechanism to construct samplespecific noisy–noisier training pair, enabling the model to iteratively refine its parameters in response to the noise characteristics of the current sample.

• We further extend TTAD to continual TTAD, facilitating eficient and scalable adaptation across sequential test noisy point clouds with consistent noise characteristics, while reducing per-sample computational cost, which makes the method suitable for real-world deployment in streaming applications.

• Extensive experiments conducted on both synthetic and real-world datasets demonstrate that TTAD and CTTAD consistently improve the performance of state-of-the-art denoising models under diverse unseen noise conditions, achieving notable gains across multiple evaluation metrics.

## 2. Related Works

## 2.1. Deep Learning-based Point Cloud Denoising

Deep learning-based point cloud denoising methods can be broadly categorized into supervised and unsupervised approaches, depending on whether clean ground-truth data is used during training.

## 2.1.1. Supervised Point Cloud Denoising

Supervised point cloud denoising methods are typically trained on synthetic clean-noisy point cloud pairs, where degradation is commonly simulated by adding Gaussian noise to clean point cloud. These approaches can be broadly classified into three categories: 1) resampling, 2) probabilistic modeling, and 3) displacement prediction. In resampling-based methods, denoising is achieved through downsampling and upsampling reconstruction on low-noise surfaces, yet geometric details may be lost in high-curvature regions. For example, DMRDenoise [12] introduces diferentiable pooling to downsample points to a low-noise manifold and then upsamples them back to a dense output. To address the computational cost of dense grids, modern methods leverage sparse convolutional backbones via frameworks like Minkowski Engine [23] and TorchSparse [24]. These approaches typically utilize 3D U-Net architectures [25] to extract multi-scale representations from sparse voxels for effective resampling. For instance, Choe et al. [26] employed a sparse stacked hourglass network for voxel-space denoising and densification. Probabilistic modeling methods, such as ScoreDenoise [11] and its extension DeepPSR [27], formulate denoising as an iterative inference process using score-based generative modeling, often requiring computationally intensive Langevin sampling. Displacement prediction approaches (e.g., PointCleanNet [15], which is based on PointNet, and PointFil ter [13]) learn point-wise ofsets toward clean surfaces. Recent advances, such as IterativePFN [14], adopt iterative refinement modules to jointly optimize displacement prediction and confidence estimation. Moreover, recent methods explore denoising by disentangling noise in the latent space. PDFlow [28] leverages normalizing flows to decouple noise from the underlying structure in a high-dimensional latent space. Similarly, PD-LTS [29] employs graph convolution and invertible neural networks for latent-space denoising.

However, supervised denoising methods often sufer from significant performance degradation when the noise distribution at test time deviates from the synthetic noise used during training. This distribution gap limits their generalization to unseen noise types, particularly in test-time scenarios where the noise characteristics are unknown and vary considerably across domains.

## 2.1.2. Unsupervised Point Cloud Denoising

Unsupervised point cloud denoising methods aim to train denoising models without access to clean ground-truth data, typically by exploiting the internal structure of point clouds or constructing surrogate supervision signals. TotalDenoising [16] leverages intrinsic structural information of point clouds to provide self-supervision for model training. In contrast, methods such as SITF [17] and Noise4Denoise [18] construct noisynoisier training pairs by injecting Gaussian noise into input point clouds.

While these approaches benefit from the unsupervised paradigm and generally exhibit good cross-domain generalization, their performance is often limited by the absence of clean supervision, resulting in suboptimal denoising quality. Moreover, methods that construct noisy-noisier pairs (e.g., SITF [17] and Noise4Denoise [18]) typically rely on synthetic Gaussian noise injection, which does not account for the actual distribution of test-time noise.

## 2.2. Test-Time Adaptation and Contrastive Learning

## 2.2.1. Test-Time Adaptationfor Point Clouds

Test-time adaptation (TTA) [19] has emerged as a promising solution to improve model generalization against domain shift by adapting a trained model to unlabeled test data during inference. While TTA has been extensively studied in image restoration [20, 21, 22], its application to 3D point clouds remains relatively underexplored. Existing point-cloud TTA studies mainly focus on high-level tasks such as classification, segmentation, and open-vocabulary analysis. For example, Wang et al. [30] propose a backpropagation-free two-stream framework for eficient 3D test-time adaptation, and Zou et al. [31] introduce hierarchical geometry learning to exploit local, global, and temporal geometric cues for point cloud segmentation. More recently, difusion-based methods have also been explored for pointcloud TTA. CloudFixer [32] performs test-time input adaptation via difusion-guided geometric transformations, while Dastmalchi et al. [33] propose a denoising-difusion-based adaptation framework in latent space. LaPDA [34] improves point cloud denoising through adaptive latent-space modeling, providing a more structured representation for separating geometry from noise.

## 2.2.2. Contrastive Learning for Point Clouds

Contrastive learning has shown strong potential for selfsupervised 3D representation learning. PointContrast [35] highlights the efectiveness of enforcing feature consistency across transformed views for point cloud understanding. Later works further extend contrastive learning to cross-modal alignment [36], eficient point cloud representation learning [37], and large-scale scene-level pre-training [38]. Inspired by these advances, we introduce a noise-aware contrastive adaptation strategy for test-time denoising. Unlike prior methods that mainly target ofline pre-training for high-level 3D tasks, our method enforces consistency across sample-specific noisynoisier augmentations at test time, enabling the model to learn noise-invariant representations without architectural modifications.

## 3. Method

## 3.1. Problem Formulation

In the training phase, we consider a point cloud denoising model, $f _ { \theta } : X \to Y .$ , designed to map a noisy point cloud X to its clean counterpart Y. The model’s parameters θ are optimized over a source-domain dataset $\mathcal { D } _ { s } = \{ ( X _ { s } ^ { i } , Y ^ { i } ) \} _ { i = 1 } ^ { N }$ by minimizing a supervised loss:

$$
\operatorname* { m i n } _ { \theta } \mathbb { E } _ { ( X _ { s } , Y ) \sim \mathcal { D } _ { s } } \left[ \mathcal { L } _ { \operatorname* { s u p } } ( f _ { \theta } ( X _ { s } ) , Y ) \right] ,\tag{1}
$$

where $X _ { s } ~ = ~ Y + \epsilon _ { s }$ is the input corrupted by synthetic noise $\epsilon _ { s } \sim p _ { s }$ , and $\mathcal { L } _ { \mathrm { s u p } } ( \cdot )$ is a task-specific loss, typically measuring geometric discrepancy (e.g., point-wise Euclidean distance).

In the test phase, the pre-trained model $f _ { \theta }$ encounters a new noisy input $X _ { t } ~ = ~ Y + \epsilon _ { t } ,$ where the noise $\epsilon _ { t }$ is drawn from an unseen distribution $p _ { t }$ that difers from the source distribution $( p _ { t } \neq p _ { s } )$ . This distribution shift introduces a significant domain gap, leading to a clear drop in performance when the model faces noise characteristics for which it was not trained.

Our goal is to adapt the pre-trained model $f _ { \theta }$ to a new noisy point cloud $X _ { t }$ with unseen noise, without access to any groundtruth Y during the test phase. To accomplish this, we formulate this adaptation as an optimization problem, guided by a selfsupervised loss on the test input:

$$
\theta ^ { * } = \arg \operatorname* { m i n } _ { \theta } \mathbb { E } \left[ \mathcal { L } _ { \mathrm { s e l f } } ( f _ { \theta } ( X _ { t } ) ) \right] ,\tag{2}
$$

where $\mathcal { L } _ { \mathrm { s e l f } } ( \cdot )$ is a self-supervised objective that leverages only the information contained within the test data itself to guide the parameter updates.

## 3.2. Overview

The proposed TTAD framework facilitates a pre-trained point cloud denoising model to adapt to unseen noise during inference, without requiring clean supervision or retraining from source-domain data, as illustrated in Fig. 2. It consists of two main components: residual-driven noise modeling (RDNM) and noise-aware contrastive adaptation (NACA). Specifically, given a test noisy point cloud $X _ { t }$ and a pre-trained denoising model $f _ { \theta } ,$ firstly, the RDNM estimates a sample-specific noise distribution $p ( { \bf n } )$ by analyzing the deviation between $X _ { t }$ and its initial denoised prediction ${ \bar { \tilde { Y } } } .$ Then, NACA creates samplespecific noisy-noisier test-time training pair from ${ \tilde { Y } } _ { \cdot }$ , and adapts $f _ { \theta }$ towards improved performance within a few adaptation steps using contrastive learning at test time. To stabilize the optimization and prevent overfitting to individual test samples, TTAD employs an exponential moving average (EMA) strategy to update model parameters $f _ { \theta } .$ . Furthermore, TTAD naturally extends to a continual TTAD (CTTAD) setting, where the model parameters are progressively updated across sequential noisy point clouds characterized by a consistent noise distribution. Overall, the proposed TTAD is architecture-agnostic and can be seamlessly integrated into various existing denoising models, enhancing their generalization ability under noise distribution shift at deployment.

![](images/8432fd241433a156993531f7f4aaf47aa0f48a54eb8ade1e2becc29daf2ebf54.jpg)  
Figure 2: Overview of the proposed TTAD framework. (a) The input noisy point cloud $X _ { t }$ is first denoised by a pre-trained model $f _ { \theta }$ to produce a pseudo-clean output Y<sup>˜</sup> . (b) The residual-driven noise modeling (RDNM) module utilizes the residual noise $\hat { N } _ { t } = X _ { t } - \tilde { Y }$ to estimate the test-time noise distribution p(n) of X via a lightweight Gaussian mixture model (GMM). (c) The noise-aware contrastive adaptation (NACA) module generates a sample-specific noisy-noisier pair $( \tilde { X } _ { t } ^ { ( 1 ) } , \tilde { X } _ { t } ^ { ( 2 ) } )$ of Y<sup>˜</sup> . Within each test-time adaptation step, we first use the reconstruction loss $\scriptstyle { \mathcal { L } } _ { \mathrm { r e c o n } } .$ , identity loss $\mathcal { L } _ { \mathrm { i d } } .$ and consistency loss $\mathcal { L } _ { \mathrm { c o n s i s t } }$ to update the online model $f _ { \theta _ { \mathrm { o n l i n e } } }$ via gradient descent at each iteration step. After that, the target model’s parameters $\theta _ { \mathrm { t a r g e t } }$ are updated as an exponential moving average (EMA) of the online model’s parameters $\theta _ { \mathrm { o n l i n e } }$ . Finally, we can use the final $\theta _ { \mathrm { t a r g e t } }$ to obtain the adapted denoised result $\bar { Y _ { c } } = f _ { \theta _ { \mathrm { t a r g e t } } } ( X _ { t } )$

## 3.3. Residual-driven Noise Modeling

The residual-driven noise modeling (RDNM) aims to capture the complex and diverse noise characteristics of the input noisy point cloud $X _ { t }$ based on a Gaussian mixture model at testtime. Unlike existing denoising methods that assume sampleagnostic and synthetic noise in TTA-based image denoising [21, 22] and self-supervised point cloud denoising [18, 17], our approach estimates the latent noise distribution directly from the input data in a sample-specific manner, resulting in more accurate and adaptive perturbation construction for test-time adaptation.

Firstly, given a test noisy sample $X _ { t } ~ \in ~ \mathbb { R } ^ { 3 \times N }$ , we employ the pre-trained denoising model $f _ { \theta } ( \cdot )$ to produce a pseudo-clean point cloud ${ \tilde { Y } } ,$ and compute the residual noise $\hat { N } _ { t } \mathrm { : }$

$$
\hat { N } _ { t } = X _ { t } - \tilde { Y } ,\tag{3}
$$

where $\begin{array} { r c l } { \tilde { Y } } & { = } & { f _ { \theta } ( X _ { t } ) } \end{array}$ , and $\hat { N } _ { t } ~ \in ~ \mathbb { R } ^ { 3 \times N }$ consists of N three dimensional noise vectors $\{ { \mathbf { n } _ { i } } \} _ { i = 1 } ^ { N }$

Then, we fit a M-component Gaussian mixture model (GMM) to approximate the noise distribution of $X _ { t } \colon$

$$
p ( { \bf n } ) = \sum _ { m = 1 } ^ { M } \pi _ { m } N ( { \bf n } ; \mu _ { m } , \Sigma _ { m } ) ,\tag{4}
$$

where $\pi _ { m }$ are the mixture weights satisfying $\sum \pi _ { m } = 1 , \mu _ { m }$ and $\pmb { \Sigma } _ { m }$ denote the mean and covariance of the m-th Gaussian component, respectively.

The RDNM strategy allows the generation of sample-specific noise perturbations $p ( { \bf n } )$ that align with the actual noise distribution of the test sample, thereby providing reliable support for contrastive adaptation in downstream learning.

## 3.4. Noise-Aware Contrastive Adaptation

The noise-aware contrastive adaptation (NACA) enables $f _ { \theta }$ to adapt in a self-supervised manner by creating sample-specific noisy–noisier test-time training pair and enforcing output consistency using contrastive learning. Unlike existing TTA-based image denoising approaches which rely on synthetic or predefined noise assumptions (e.g., Gaussian noise) [22, 17], the proposed NACA can capture sample-specific noise distributions encountered in real-world scenarios, improving the generalization performance under unseen noise at test time. To this end, firstly, a sample-specific noisy-noisier pairing mechanism produces noisy-noisier pair $( \tilde { X } _ { t } ^ { ( 1 ) } , \tilde { X } _ { t } ^ { ( 2 ) } )$ that reflect the test sample’s noise characteristics. Then, an exponential moving average (EMA) update strategy is employed to stabilize the online adaptation process while preserving the pre-trained model’s performance on the source domain.

## 3.4.1. Sample-specific Noisy–Noisier Pairing

For efective contrastive adaptation, we adopt a samplespecific noisy–noisier pairing strategy to create noisy–noisier test-time training pair $( \tilde { X } _ { t } ^ { ( 1 ) } , \tilde { X } _ { t } ^ { ( 2 ) } )$ based on the noise distribution $p ( { \bf n } )$ of $X _ { t }$ estimated by RDNM.

Given the noise distribution p(n) and pseudo-clean point cloud Y<sup>˜</sup> , we sample two independent noise vectors $\mathbf { n } ^ { ( 1 ) } \sim p ( \mathbf { n } )$ , $\begin{array} { l } { \mathbf { n } ^ { ( 2 ) } ~ \sim ~ p ( \mathbf { n } ) } \end{array}$ from $p ( { \bf n } )$ , and construct a noisy–noisier pair $( \tilde { X } _ { t } ^ { ( 1 ) } , \tilde { X } _ { t } ^ { ( 2 ) } )$ as:

$$
\begin{array} { r l r } & { } & { \tilde { X } _ { t } ^ { ( 1 ) } = \tilde { Y } + \alpha _ { 1 } \cdot \mathbf { n } ^ { ( 1 ) } , } \\ & { } & { \tilde { X } _ { t } ^ { ( 2 ) } = \tilde { Y } + \alpha _ { 2 } \cdot \mathbf { n } ^ { ( 2 ) } , } \end{array}\tag{5}
$$

where $0 ~ < ~ \alpha _ { 1 } ~ < ~ \alpha _ { 2 } ~ \leq ~ 1$ This pairing strategy enforces output consistency across varying noise levels, compelling the denoising model to learn noise-invariant feature representations. Moreover, the deliberately structured diversity in the input space enhances the model’s robustness to distribution shifts, leading to superior generalization capability.

## 3.4.2. Parameter Update

Updating model parameters θ during test-time adaptation must balance two competing objectives: adapting to unseen noise while preserving the denoising capability learned from the source domain. This becomes particularly challenging when the noisy-noisier pair introduces large variations in the input, potentially leading to unstable optimization and catastrophic forgetting. To mitigate these issues, we employ an exponential moving average (EMA) strategy to regularize the adaptation process and stabilize the learning dynamics throughout test-time inference.

Specifically, we introduce the online model $f _ { \theta _ { \mathrm { o n l i n e } } }$ and the target model $f _ { \theta _ { \mathrm { t a r g c t } } }$ , both initialized from the pre-trained denoising model $f _ { \theta } . f _ { \theta _ { \mathrm { o n l i n e } } }$ and $f _ { \theta _ { \mathrm { t a r g e t } } }$ share identical architectures but difer in their parameters. Given the initial denoised point cloud $\tilde { Y }$ and its noisy–noisier pair $( \tilde { X } _ { t } ^ { ( 1 ) } , \tilde { X } _ { t } ^ { ( 2 ) } )$ , for the k-th adaptation step, we first set $\theta _ { \mathrm { o n l i n e } } ^ { k } = \theta _ { \mathrm { t a r g e t } } ^ { k - 1 } \cdot$ . Then, we employ the online model $f _ { \theta _ { \mathrm { o n l i n e } } ^ { k } }$ to update its parameters $\theta _ { \mathrm { o n l i n e } } ^ { k }$ via gradient descent at each iteration, according to the loss function $\mathcal { L } _ { \mathrm { t o t a l } }$ discussed in Section 3.4.3. After that, the target model’s parameters $\theta _ { \mathrm { t a r g e t } } ^ { k }$ are updated as an exponential moving average of the online model’s parameters $\theta _ { \mathrm { o n l i n e } } ^ { k } \mathrm { : }$

$$
\begin{array} { r } { \theta _ { \mathrm { t a r g e t } } ^ { k } = \beta \cdot \theta _ { \mathrm { t a r g e t } } ^ { k - 1 } + ( 1 - \beta ) \cdot \theta _ { \mathrm { o n l i n e } } ^ { k } , } \end{array}\tag{6}
$$

where $\beta ~ \in ~ [ 0 , 1 )$ is the decay factor controlling the update smoothness. The final denoised output $Y _ { c }$ can be obtained by:

$$
Y _ { c } = f _ { \theta _ { \mathrm { t a r g e t } } ^ { k } } ( X _ { t } ) .\tag{7}
$$

This strategy ofers two primary benefits. First, it suppresses abrupt parameter fluctuations caused by noisy gradients, improving stability in the absence of supervision. Second, by maintaining a trajectory-smoothed version of the model, EMA helps retain prior knowledge from the source domain and acts as a safeguard against overfitting to individual test samples.

## 3.4.3. Loss Function

A key challenge in test-time adaptive denoising lies in designing an optimization objective that facilitates efective adaptation without access to clean supervision. When an unseen noisy point cloud $X _ { t }$ arrives at test time, we update the parameters of the online denoising model $f _ { \theta _ { \mathrm { o n l i n e } } }$ by minimizing a self-supervised loss, as ground-truth labels are unavailable in this setting. Concretely, our objective combines three complementary terms: reconstruction loss, noise consistency loss, and identity constraint.

Reconstruction Loss $\scriptstyle { \mathcal { L } } _ { \mathrm { r e c o n } }$ This component encourages the denoising model to remain close to its pre-trained behavior by aligning its output $f _ { \theta _ { \mathrm { o n l i n e } } } ( \tilde { X } _ { t } ^ { ( 1 ) } )$ on the noisy input $\tilde { X } _ { t } ^ { ( 1 ) }$ with the pseudo-clean prediction Y<sup>˜</sup> , which serves as a soft supervision signal. To maintain compatibility with diferent denoising methods, we directly use their predefined reconstruction metrics as the form of $\scriptstyle { \mathcal { L } } _ { \mathrm { r e c o n } }$

Noise Consistency Loss $\mathcal { L } _ { \mathrm { c o n s i s t } } .$ To promote robustness against varying noise intensities, we enforce consistency between the denoised outputs of the noisy–noisier input pair $( \tilde { X } _ { t } ^ { ( 1 ) } , \tilde { X } _ { t } ^ { ( 2 ) } ,$ ). We use the symmetric Chamfer Distance (CD) [39] to measure geometric alignment:

$$
\mathcal { L } _ { \mathrm { c o n s i s t } } ( A , B ) = \frac { 1 } { | A | } \sum _ { a \in A } \operatorname* { m i n } _ { b \in B } \| a - b \| ^ { 2 } + \frac { 1 } { | B | } \sum _ { b \in B } \operatorname* { m i n } _ { a \in A } \| b - a \| ^ { 2 } ,\tag{8}
$$

where $A \ = \ f _ { \theta _ { \mathrm { o n l i n c } } } ( \tilde { X } _ { t } ^ { ( 1 ) } ) , B \ = \ f _ { \theta _ { \mathrm { o n l i n c } } } ( \tilde { X } _ { t } ^ { ( 2 ) } )$ . This regularization encourages the network to learn noise-invariant representations and generalize across a broader range of test-time perturbations.

Identity Constraint $\mathcal { L } _ { \mathrm { i d } }$ . To mitigate catastrophic forgetting during adaptation, we introduce a regularization term that preserves the model’s behavior on its own predictions. Specifically, the adapted model $f _ { \theta _ { \mathrm { o n l i n e } } }$ is encouraged to act as an identity mapping when applied to the pseudo-clean input:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { i d } } = \| f _ { \theta _ { \mathrm { o n l i n e } } } ( \tilde { Y } ) - \tilde { Y } \| _ { 2 } ^ { 2 } . } \end{array}\tag{9}
$$

This constraint maintains structural integrity and prevents the model from overfitting to perturbations introduced during adaptation.

The final loss function $\mathcal { L } _ { \mathrm { t o t a l } }$ is defined as:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { r e c o n } } + \lambda _ { 1 } \mathcal { L } _ { \mathrm { c o n s i s t } } + \lambda _ { 2 } \mathcal { L } _ { \mathrm { i d } } , } \end{array}\tag{10}
$$

where $\lambda _ { 1 }$ and $\lambda _ { 2 }$ are hyperparameters controlling the relative importance of the consistency and identity terms, respectively. The complete adaptation procedure of TTAD is summarized in Algorithm 1.

## 3.5. Continual TTAD

The CTTAD is proposed to improve adaptation eficiency for scenarios where point clouds encounter noise distributions with consistent characteristic. In the standard TTAD setup, the adaptation for each test sample $X _ { t }$ is performed independently by solving the optimization problem defined in Equation (2), with model parameters always initialized from the pre-trained source model θ. After adaptation, the updated parameters $\theta ^ { * }$ are discarded, and the next test sample is adapted from the same initialization. For sequential test samples with shared noise distributions, CTTAD achieves continual adaptation by progressively updating model parameters, eliminating the need for persample resets.

Specifically, for each test sample $X _ { t } ^ { i }$ from a target-domain dataset $\mathcal { D } _ { t }$ , the denoising model is initialized with the parameters adapted from the previous sample $X _ { t } ^ { i - 1 } \colon$

$$
\theta _ { i }  \theta _ { i - 1 } .\tag{11}
$$

Algorithm 1 Test-Time Adaptive Denoising for Point Cloud   
Input: Pre-trained denoising model $f _ { \theta } ,$ input noisy point cloud   
$X _ { t } ,$ learning rate $\eta ,$ EMA decay factor $\beta ,$ adaptation step $K$   
Output: Adapted denoised output $Y _ { c }$   
1: Initialize ${ \bar { \theta } } _ { \mathrm { t a r g e t } } ^ { 0 }  \theta$   
2: Pseudo-clean point cloud: $\tilde { Y } = f _ { \theta } ( X _ { t } )$   
3: Estimate residual noise: $\hat { N } _ { t } = \dot { X } _ { t } - \tilde { Y }$   
4: Fit Gaussian Mixture Model $p ( { \bf n } )$ to $\hat { N } _ { t }$   
5: Sample $n ^ { ( 1 ) } \sim p ( { \bf n } ) , n ^ { ( 2 ) } \sim p ( { \bf n } )$   
6: Generate sample-specific noisy–noisier pair:   
$\tilde { X } _ { t } ^ { ( 1 ) } = \tilde { Y } + \alpha _ { 1 } \dot { n } ^ { ( 1 ) } , ~ \tilde { X } _ { t } ^ { ( 2 ) } = \tilde { Y } + \alpha _ { 2 } \dot { n ^ { ( 2 ) } }$   
7: for $k = 1$ to $K$ do   
8: Initialize online weights $\theta _ { \mathrm { o n l i n e } } ^ { k }  \theta _ { \mathrm { t a r g e t } } ^ { k - 1 }$   
9: Compute total loss: $\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { r e c o n } } + \breve { \lambda } _ { 1 } \mathcal { L } _ { \mathrm { c o n s i s t } } + \lambda _ { 2 } \mathcal { L } _ { \mathrm { i d } }$   
10: Update $\theta _ { \mathrm { o n l i n e } } ^ { k }$ via gradient descent:   
$\theta _ { \mathrm { o n l i n e } } ^ { k }  \theta _ { \mathrm { o n l i n e } } ^ { k } - \eta \cdot \nabla _ { \theta } \mathcal { L } _ { \mathrm { t o t a l } }$   
11: Update ${ \ddot { \theta } _ { \mathrm { t a r g e t } } ^ { \bar { k } } }$ with EMA:   
$\theta _ { \mathrm { t a r g e t } } ^ { k }  \beta \theta _ { \mathrm { t a r g e t } } ^ { k - 1 } + ( 1 - \beta ) \theta _ { \mathrm { o n l i n e } } ^ { k }$   
12: end for   
13: Final output: $Y _ { c } = f _ { \theta _ { \mathrm { t a r g e t } } ^ { k } } ( X _ { t } )$

Algorithm 2 Continual Test-Time Adaptive Denoising for   
Point Cloud   
Input: Initial model parameters $\theta ;$ adaptation step $K ;$ sequence   
of test point clouds $X _ { t } ^ { i }$ from the target-domain dataset $\mathcal { D } _ { t }$   
Output: Adapted predictions $\{ Y _ { c , i } \}$ and parameter trajectory   
$\{ \theta _ { i } \}$   
1: Initialize: $\theta _ { \mathrm { o n l i n e , 0 } }  \theta , \theta _ { \mathrm { t a r g e t , 0 } }  \theta$   
2: for each input noisy point cloud $X _ { t } ^ { i } \in \mathcal { D } _ { t }$ do   
3: Inherit parameters from the previous sample $X _ { t } ^ { i - 1 } \colon$   
$\theta _ { \mathrm { o n l i n e } , i }  \theta _ { \mathrm { o n l i n e } , i - 1 } , \theta _ { \mathrm { t a r g e t } , i }  \theta _ { \mathrm { t a r g e t } , i - 1 }$   
4: Determine adaptation step: $K _ { i } =$ anneal(i K)   
5: Perform TTAD on $X _ { t } ^ { i }$ for $K _ { i }$ steps   
6: Output denoised result $Y _ { c , i } = f _ { \theta _ { \mathrm { t a r g e t } , i } } ( X _ { t } ^ { i } )$ and $\theta _ { i } = \theta _ { \mathrm { t a r g e t } , i }$   
7: end for

This permits the model to accumulate knowledge progressively from past samples $\{ X _ { t } ^ { 1 } , X _ { t } ^ { 2 } , \ldots , X _ { t } ^ { i - 1 } \}$ , thereby improving its generalization to future inputs. To balance accuracy and eficiency, the number of adaptation steps $K _ { i }$ is determined by an annealing schedule:

$$
K _ { i } = \mathsf { a n n e a l } ( i , K ) = \mathsf { m a x } ( 0 , \lfloor \gamma ^ { i - 1 } \cdot K \rfloor ) ,\tag{12}
$$

where $\gamma \in ( 0 , 1 )$ controls the decay rate. The denoising model then performs TTAD on $X _ { t } ^ { i }$ for $K _ { i }$ adaptation steps using the self-supervised objective in Equation (10). Finally, the denoised point cloud is obtained by $Y _ { c , i } = f _ { \theta _ { \mathrm { t a r g e t } , i } } ( X _ { t } ^ { i } )$ . The CTTAD gradually reduces the number of adaptation steps over time, mirroring a test-time annealing process, which helps stabilize the model as it processes more test samples. The procedure of CTTAD is elaborated in Algorithm 2.

## 4. Experiment

## 4.1. Experimental Setup

## 4.1.1. Datasets

We evaluate the proposed TTAD framework on both syn thetic and real-world datasets to comprehensively assess its performance under Gaussian and diverse unseen noise conditions.

For experiments on synthetic datasets, we follow standard protocols and adopt the widely used PUNet dataset [40], originally designed for point cloud upsampling and denoising. To evaluate generalization under distribution shift, five types of unseen noise are introduced at test time: (1) Pulse noise, which randomly replaces a subset of points with large displacements; (2) Exponential noise, which introduces long-tailed, asymmetric perturbations; (3) Laplace noise, which produces sharp and heavy-tailed perturbations; (4) Uniform noise, which samples perturbations from a bounded uniform distribution; and (5) Mixed noise, which combines Gaussian, Laplace, and Uniform perturbations to simulate more complex hybrid corruptions. These noise types difer significantly from the Gaussian distribution assumed during training and are designed to challenge the generalization ability of denoising models under diverse distribution shifts. For this setting, we sample 20 unseen CAD meshes from the PUNet dataset [40] and generate noisy point clouds at 50K resolution. Each model is evaluated both with and without the TTAD framework to quantify its efectiveness in improving generalization under domain shift. This setup ensures suficient noise diversity while providing a more comprehensive evaluation of robustness to unseen noise distributions.

For real-world evaluation, we employ the Paris-Rue-Madame dataset [41], which consists of LiDAR scans of outdoor urban street scenes in Paris. This dataset features various realworld sensor noise patterns, including motion artifacts, occlusions, and multipath interference, making it a suitable benchmark for evaluating generalization under naturally occurring corruptions. As the ground-truth geometry is not available, we conduct qualitative comparisons by visually inspecting the denoised outputs.

## 4.1.2. Denoising Models

To verify the generality of our approach, we apply TTAD to three representative point cloud denoising backbones:

• ScoreDenoise (Score) [11]: A score-based denoising model that applies score matching to approximate the gradient of the data distribution for sampling-based restoration.

• PointFilter (PF) [13]: A spatial filtering-based method that predicts residual corrections for noisy inputs by aggregating local neighborhood information.

• IterativePFN (IPFN) [14]: An iterative refinement network that leverages feature fusion and confidence weighting to progressively restore clean point clouds.

To provide a stronger and more comprehensive evaluation, we further include two recent and closely related denoising methods:

• LaPDA [34]: A recent point cloud denoising method that enhances denoising performance through input adaptation at test time together with adaptive geometry-aware feature modeling.

• PD-LTS [29]: A latent-space point cloud denoising method that leverages graph convolution and invertible neural networks to disentangle noise from geometric structure.

We use oficial pre-trained weights for ScoreDenoise and IterativePFN. For PointFilter, we retrain the denoising model using its public implementation and recommended hyperparameters on the PUNet training set [40]. For LaPDA and PD-LTS, we use their public implementations and report the corresponding comparison results under our experimental setting. For a fair comparison, all iterative baseline methods are configured to use 2 inference iterations unless otherwise specified. Specifically, 20 CAD meshes are selected, and Poisson disk sampling is applied to generate uniformly distributed point clouds at three resolutions: 10K, 30K, and 50K. To construct noisy inputs for training, we add zero-mean Gaussian noise with standard deviations randomly sampled from the range of 0.5% to 2.0% of the object’s bounding sphere radius.

## 4.1.3. Adaptation Details

For all TTAD and CTTAD experiments, we use the Adam optimizer with learning rate of $1 \times 1 0 ^ { - 4 }$ . The number of component M of GMM is set as 5. The loss weights are set as $\lambda _ { 1 } = 0 . 2$ for the consistency term and $\lambda _ { 2 } ~ = ~ 0 . 1$ for the identity regularization. The EMA decay factor is fixed at $\beta = 0 . 9 9$ In the standard TTAD setting, each test sample is adapted for $K \ = \ 1 0 0$ iterations. For noisy-noisier perturbations, we use $\alpha _ { 1 } = 0 . 3$ and $\alpha _ { 2 } = 0 . 8$ to generate contrastive input pair with distinct noise intensities. In the CTTAD setting, γ is set as 0.7. All experiments are performed on an NVIDIA RTX 4090 GPU.

## 4.1.4. Evaluation Protocol

We evaluate the denoising performance under both synthetic and real-world noise settings. Chamfer Distance (CD) [39] and Point-to-Mesh distance (P2M) [42] are adopted as quantitative metrics. For real world validation, we conduct qualitative analysis on the Paris-Rue-Madame (PRM) dataset [41]. As ground truth is unavailable, visual comparisons are used to assess the perceptual quality of denoised outputs.

## 4.2. Comparison on Synthetic Dataset

## 4.2.1. Quantitative Results

Table 1 reports the in-distribution results under Gaussian noise, where the test corruption is consistent with the sourcedomain noise used during training. In this setting, the three baseline denoising backbones already achieve relatively strong performance, and the improvements brought by TTAD and

Table 1: Performance comparison of denoising methods on the PUNet dataset under in-distribution Gaussian noise. “Train/Test” denotes the noise type used for training and testing, respectively. Bold and underlined values indicate the best and second-best performance, respectively. CD and P2M values are multiplied by 10<sup>4</sup>.
<table><tr><td>Method</td><td>1% Gaussian/Gaussian CD↓</td><td>P2M↓</td><td>2% Gaussian/Gaussian CD↓</td><td>P2M↓</td></tr><tr><td>Score</td><td>0.86</td><td>0.55</td><td>2.10</td><td>1.68</td></tr><tr><td>+ TTAD + CTTAD</td><td>0.90</td><td>0.58</td><td>1.66</td><td>1.26</td></tr><tr><td></td><td>0.89</td><td>0.57</td><td>1.74</td><td>1.28</td></tr><tr><td>PF + TTAD</td><td>0.80</td><td>0.42</td><td>1.49</td><td>1.07</td></tr><tr><td>+ CTTAD</td><td>0.82 0.81</td><td>0.44</td><td>1.27</td><td>0.91</td></tr><tr><td>IPFN</td><td></td><td>0.43</td><td>1.19</td><td>0.84</td></tr><tr><td>+ TTAD</td><td>0.61</td><td>0.32</td><td>1.31</td><td>0.88</td></tr><tr><td>+ CTTAD</td><td>0.62</td><td>0.31</td><td>1.15</td><td>0.81</td></tr><tr><td></td><td>0.62</td><td>0.29</td><td>1.08</td><td>0.75</td></tr><tr><td>LaPDA</td><td>0.57</td><td>0.35</td><td>1.71</td><td>1.26</td></tr><tr><td>PD-LTS</td><td>0.54</td><td>0.32</td><td>0.97</td><td>0.76</td></tr></table>

CTTAD are generally smaller than those under unseen noise distributions. This observation is expected, since test-time adaptation is primarily designed to handle distribution shifts rather than further optimize already matched source-target conditions. Nevertheless, TTAD and CTTAD usually maintain performance comparable to their corresponding backbones under Gaussian noise, while still yielding moderate gains at the higher noise level. For example, under 2% Gaussian noise, PF-CTTAD reduces CD from 1.49 to 1.19 and P2M from 1.07 to 0.84, while IPFN-CTTAD reduces CD from 1.31 to 1.08 and P2M from 0.88 to 0.75.

Compared with LaPDA and PD-LTS, our TTAD-enhanced models remain competitive under Gaussian noise, although the relative advantage is less pronounced than that under unseen settings. Overall, these results indicate that the proposed adaptation framework does not significantly compromise the original denoising ability when no distribution shift is present.

Table 2 reports the quantitative results under four single unseen noise distributions, including Pulse, Exponential, Laplace, and Uniform noise. When the test noise deviates from the Gaussian corruption used during training, all three baseline denoising backbones sufer from clear performance degradation, confirming the severity of domain shift in point cloud denoising. Among these unseen settings, Pulse and Exponential noise generally lead to the most pronounced deterioration, while Laplace and Uniform noise reveal diferent sensitivities of the competing methods to heavy-tailed and bounded perturbations, respectively.

Incorporating TTAD or CTTAD consistently improves the robustness of the three backbones under most single unseen noise conditions, with the gains being particularly evident at the higher noise level of 2%. For example, under 2% Pulse noise, PF-CTTAD reduces CD from 3.42 to 1.49 and P2M from 3.26 to 2.04, while under 2% Exponential noise, IPFN-CTTAD reduces CD from 5.47 to 2.98 and P2M from 4.24 to 2.68.

Compared with LaPDA and PD-LTS, our method remains competitive and often achieves better results when combined with suitable backbones. In particular, PD-LTS attains the best performance under several relatively mild 1% settings, such as

Table 2: Performance comparison of denoising methods on the PUNet dataset under unseen synthetic noise. “Train/Test” denotes the noise type used for training and testing, respectively. Bold and underlined values indicate the best and second-best performance, respectively. CD and P2M values are multiplied by 10<sup>4</sup>.
<table><tr><td rowspan="3">Method</td><td colspan="8">1%</td><td colspan="8">2%</td></tr><tr><td colspan="2">Gaussian/Pulse</td><td colspan="2">Gaussian/Exponential</td><td colspan="2">Gaussian/Laplace</td><td colspan="2">Gaussian/Uniform</td><td colspan="2">Gaussian/Pulse</td><td colspan="2">Gaussian/Exponential</td><td colspan="2">Gaussian/Laplace</td><td colspan="2">Gaussian/Uniform</td></tr><tr><td>CD↓</td><td>P2M↓</td><td>CD↓</td><td>P2M↓</td><td>CD↓</td><td>P2M↓</td><td>CD↓</td><td>P2M↓</td><td>CD↓</td><td>P2M↓</td><td>CD↓</td><td>P2M↓</td><td>CD↓</td><td>P2M↓</td><td>CD↓</td><td>P2M↓</td></tr><tr><td>Score</td><td>1.49</td><td>1.13</td><td>2.80</td><td>2.03</td><td>0.96</td><td>0.60</td><td>1.98</td><td>1.33</td><td>4.60</td><td>4.40</td><td>7.71</td><td>6.67</td><td>2.23</td><td>1.79</td><td>10.45</td><td>9.31</td></tr><tr><td>+ TTAD</td><td>1.10</td><td>0.74</td><td>2.55</td><td>1.90</td><td>0.95</td><td>0.62</td><td>1.32</td><td>0.86</td><td>3.72</td><td>3.50</td><td>6.20</td><td>5.62</td><td>1.46</td><td>1.10</td><td>8.15</td><td>6.66</td></tr><tr><td>+ CTTAD</td><td>1.12</td><td>0.77</td><td>2.37</td><td>1.77</td><td>0.95</td><td>0.62</td><td>1.33</td><td>0.89</td><td>3.66</td><td>3.52</td><td>5.64</td><td>5.17</td><td>1.53</td><td>1.12</td><td>7.41</td><td>6.13</td></tr><tr><td>PF</td><td>1.12</td><td>0.80</td><td>2.40</td><td>1.84</td><td>1.06</td><td>0.65</td><td>1.49</td><td>0.93</td><td>3.42</td><td>3.26</td><td>6.15</td><td>4.99</td><td>2.44</td><td>2.06</td><td>5.03</td><td>4.33</td></tr><tr><td>+ TTAD</td><td>0.96</td><td>0.64</td><td>2.26</td><td>1.79</td><td>1.10</td><td>0.69</td><td>1.21</td><td>0.74</td><td>1.60</td><td>2.22</td><td>4.10</td><td>3.34</td><td>2.37</td><td>1.97</td><td>3.38</td><td>2.73</td></tr><tr><td>+ CTTAD</td><td>0.90</td><td>0.59</td><td>2.12</td><td>1.68</td><td>1.03</td><td>0.64</td><td>1.13</td><td>0.68</td><td>1.49</td><td>2.04</td><td>3.77</td><td>3.07</td><td>2.20</td><td>1.81</td><td>3.12</td><td>2.51</td></tr><tr><td>IPFN</td><td>1.40</td><td>0.88</td><td>2.66</td><td>1.95</td><td>0.70</td><td>0.43</td><td>0.96</td><td>0.61</td><td>3.98</td><td>3.71</td><td>5.47</td><td>4.24</td><td>1.22</td><td>0.88</td><td>15.84</td><td>13.52</td></tr><tr><td>+ TTAD</td><td>1.12</td><td>0.75</td><td>2.41</td><td>1.90</td><td>0.74</td><td>0.46</td><td>0.97</td><td>0.61</td><td>2.01</td><td>1.77</td><td>3.13</td><td>2.86</td><td>1.18</td><td>0.85</td><td>15.56</td><td>13.24</td></tr><tr><td>+ CTTAD</td><td>1.09</td><td>0.70</td><td>2.27</td><td>1.75</td><td>0.74</td><td>0.43</td><td>0.95</td><td>0.57</td><td>1.95</td><td>1.63</td><td>2.98</td><td>2.68</td><td>1.11</td><td>0.79</td><td>14.81</td><td>12.41</td></tr><tr><td>LaPDA</td><td>0.94</td><td>0.67</td><td>2.07</td><td>1.69</td><td>0.70</td><td>0.47</td><td>1.81</td><td>1.18</td><td>5.25</td><td>5.24</td><td>6.38</td><td>5.44</td><td>2.59</td><td>2.17</td><td>11.38</td><td>10.01</td></tr><tr><td>PD-LTS</td><td>0.80</td><td>0.54</td><td>2.40</td><td>1.95</td><td>0.66</td><td>0.43</td><td>1.42</td><td>0.93</td><td>3.68</td><td>3.57</td><td>7.10</td><td>5.96</td><td>1.33</td><td>1.05</td><td>13.07</td><td>10.99</td></tr></table>

Pulse and Laplace noise, indicating the advantage of structured feature modeling when the corruption level is moderate. However, as the noise level increases, backbones with TTAD enhancement, especially PF-CTTAD and IPFN-CTTAD, exhibit stronger robustness and achieve the best overall performance in most 2% single unseen-noise settings. These results suggest that the proposed test-time adaptation strategy becomes increasingly efective as the test corruption departs further from the source-domain Gaussian distribution.

Table 3 further reports the results under the mixed noise setting, where Gaussian, Laplace, and Uniform perturbations are combined to simulate more complex hybrid corruptions. Compared with a single synthetic noise type, this setting is more challenging because the corruption no longer follows a homogeneous distribution and instead contains multiple perturbation patterns simultaneously. Under 1% mixed noise, PD-LTS achieves the best CD, while IPFN and PD-LTS obtain the best P2M, suggesting that structured representation learning is effective when the hybrid corruption remains relatively moderate. However, under the more challenging 2% mixed noise, IPFN-CTTAD achieves the best performance, with CD and P2M reduced to 1.73 and 1.24, respectively, outperforming both the original IPFN and the additional comparison baselines. This result demonstrates that the proposed continual adaptation strategy can better handle complex hybrid corruptions by progressively adjusting the denoiser to the sample-specific noise characteristics observed at test time. Overall, the mixed-noise results provide further evidence that our method is not limited to a single unseen distribution, but can also generalize efectively to more realistic settings where diferent perturbation types coexist.

## 4.2.2. Qualitative Results

Figure 3 and Figure 4 present visual comparisons between baseline models and their TTAD-enhanced counterparts under five unseen 2% noise conditions, including Pulse, Exponential, Laplace, Uniform, and Mixed noise. Without adaptation, the original denoising backbones often exhibit noticeable artifacts, such as residual outliers, clustered noise, oversmoothed local structures, and incomplete geometric details, especially as the noise level increases. After incorporating TTAD, these artifacts are visibly alleviated, resulting in cleaner point distributions, smoother surfaces, and more complete structural boundaries.

Table 3: Performance comparison of denoising methods on the PUNet dataset under mixed unseen noise. “Train/Test” denotes the noise type used for training and testing, respectively. Bold and underlined values indicate the best and second-best performance, respectively. CD and P2M values are multiplied by 10<sup>4</sup>.
<table><tr><td rowspan="2">Method</td><td colspan="2">1% Gaussian/Mixed</td><td colspan="2">2% Gaussian/Mixed</td></tr><tr><td>CD↓</td><td>P2M↓</td><td>CD↓</td><td>P2M↓</td></tr><tr><td>Score + TTAD</td><td>1.04</td><td>0.64</td><td>3.36</td><td>2.80</td></tr><tr><td>+ CTTAD</td><td>0.99 0.97</td><td>0.63 0.63</td><td>1.84 1.80</td><td>1.40 1.36</td></tr><tr><td rowspan="3">PF + TTAD</td><td></td><td>0.59</td><td>2.26</td><td>1.84</td></tr><tr><td>1.01 1.03</td><td>0.62</td><td>2.03</td><td>1.34</td></tr><tr><td>0.98</td><td>0.58</td><td>1.88</td><td>1.49</td></tr><tr><td>IPFN</td><td>0.70</td><td>0.42</td><td>1.98</td><td>1.48</td></tr><tr><td rowspan="2">+ TTAD + CTTAD</td><td>0.75</td><td>0.46</td><td>1.82</td><td>1.34</td></tr><tr><td>0.73</td><td>0.43</td><td>1.73</td><td>1.24</td></tr><tr><td>LaPDA</td><td>0.72</td><td>0.46</td><td>3.44</td><td>2.86</td></tr><tr><td>PD-LTS</td><td>0.66</td><td>0.42</td><td>1.89</td><td>1.48</td></tr></table>

More specifically, under Pulse noise, PF with TTAD effectively removes sparse outliers and yields a cleaner overall shape, while under Exponential noise, IPFN with TTAD shows improved preservation of geometric continuity and surface smoothness. Under Laplace and Uniform noise, TTADenhanced models generally retain sharper contours and more stable local geometry than their corresponding original backbones. For the more challenging Mixed noise, the benefit of test-time adaptation becomes even more evident, as the TTAD enhanced results exhibit stronger robustness to hybrid perturbations and better preserve the overall object structure.

Compared with the additional baselines LaPDA and PD LTS, our TTAD-enhanced models achieve competitive visual quality across all five noise types. LaPDA shows favorable preservation of geometric features, while PD-LTS, as a flow-based latent-space method, generally produces smooth and coherent denoised surfaces. This comparison reflects the trade-of between representation capacity and adaptation flexibility: latent-space methods provide structured representations for geometry–noise disentanglement, whereas TTAD es-

Noisy

Score

Score-TTAD

PF

PF-TTAD

IPFN

IPFN-TTAD

LaPDA

PD-LTS

![](images/d59a12d7ea978b5777a7964c3dd0761cede1bf12ef624089fbb1a182006b5aef.jpg)  
(b) Qualitative comparison under Exponential noise.

Figure 3: Visualization of denoising results on diferent unseen 2% noise conditions. All baseline models (Score [11], PF [13], IPFN [14], LaPDA [34], and PD-LTS [29]) are trained on Gaussian noise and tested on inputs corrupted by (a) Pulse noise and (b) Exponential noise.

timates sample-specific residual perturbations to adapt existing Euclidean-space denoisers at test time. As shown in the visual results, this plug-and-play adaptation strategy efectively improves the original backbones under diverse unseen noise conditions while maintaining competitive visual quality against recent latent-space baselines. These observations qualitatively support the quantitative findings.

## 4.3. Comparison on Real-world Dataset

We further evaluate the TTAD framework on the Paris-Rue-Madame (PRM) dataset [41], which contains LiDAR scans of urban street scenes with diverse real-world noise patterns, including motion-induced distortions, ghosting artifacts, background clutter, and non-uniform sampling. Since clean groundtruth point clouds are unavailable, we provide qualitative comparisons in Figure 5, where two representative regions are further enlarged for detailed inspection.

As shown in Figure 5, the selected scene includes a roadside sign afected by surrounding clutter and ghosting artifacts. Non-adaptive baseline models such as PF [13] and ScoreDenoise [11] exhibit noticeable surface roughness and residual scattered noise, especially in the highlighted regions. IPFN [14] produces relatively smooth outputs, but local structural details can still be further refined.

In contrast, incorporating TTAD consistently improves the visual quality across the three backbones. The TTAD-enhanced results show cleaner point distributions, reduced noise accumulation, and more clearly defined structural boundaries. In particular, PF-TTAD efectively suppresses scattered outliers, while

IPFN-TTAD better preserves the continuity of geometric structures, as evidenced by the smoother contours and more coherent surfaces in the zoomed-in regions.

The comparison methods, LaPDA [34] and PD-LTS [29], also produce visually competitive denoising results. LaPDA shows favorable preservation of geometric features, while PD-LTS generates smooth and coherent surfaces through latentspace modeling. Compared with these recent baselines, the TTAD-enhanced models achieve comparable visual quality while maintaining strong adaptability to the sample-specific real-world noise patterns in the test scene.

These improvements can be attributed to the two key components of our framework. The RDNM module simulates realistic test-time perturbations to expose the model to samplespecific noise patterns, while the NACA strategy enforces consistency across these perturbations. Together, they guide the model to learn noise-invariant geometric features, enabling effective adaptation to complex real-world noise without requiring clean supervision.

## 4.4. Computational Eficiency

We first evaluate the computational eficiency by measuring the average processing time per point cloud under the pulse noise setting, where the reported values are averaged over three TTAD-compatible denoising backbones. As shown in Figure 6, the analysis includes three methods: the original network without adaptation (Baseline), TTAD (TTAD), and continual TTAD (CTTAD).

Noisy

Score

Score-TTAD

PF

IPFN

PF-TTAD

IPFN-TTAD

LaPDA

PD-LTS

![](images/b501b7d2fc36c880113885199f6634a95a831db6311a6458636ed51cc42fd06f.jpg)  
(c) Qualitative comparison under Mixed noise.

Figure 4: Visualization of denoising results on additional unseen 2% noise conditions. All baseline models (Score [11], PF [13], IPFN [14], LaPDA [34], and PD-LTS [29]) are trained on Gaussian noise and tested on inputs corrupted by (a) Laplace noise, (b) Uniform noise, and (c) Mixed noise.

The baseline model maintains a low and constant processing time, as it only performs inference. In contrast, the standard TTAD incurs a high and constant computational cost, as it performs a full, independent adaptation from scratch for each test sample. The CTTAD, while initially incurring a similar cost to standard TTAD, progressively reduces its average processing time. This significant eficiency gain is due to the parameter reuse mechanism: instead of re-initializing for each sample, the model reuses and incrementally updates the parameters learned from previous instances. This makes CTTAD particularly wellsuited for online or streaming scenarios where test samples arrive continuously.

Beyond eficiency, we then analyze the evolution of the average denoising error (Chamfer Distance and P2M Distance) as the adaptation process unfolds over a sequence of test samples under the unseen 2% Pulse noise condition. The performance dynamics, illustrated in Figure 7, reveal the core advantage of our continual learning approach by tracking the average error of the three methods over a sequence of test samples. The original network exhibits consistently high error, serving as a perfor mance lower bound. The TTAD achieves a strong, stable de noising performance. It acts as a powerful baseline for singleinstance adaptation but shows no improvement over time as it treats each sample independently. During the initial phase (processing the first few samples), CTTAD incurs a minor perfor mance cost, with its error being slightly higher than that of the standard TTAD. This reflects a transient adaptation phase where the model builds its long-term memory. However, as more samples are processed, the accumulated knowledge allows CTTAD to rapidly improve, eventually surpassing the standard TTAD and achieving the best overall performance.

Average Processing Time vs. Number of Point Clouds  
![](images/5dd9f4612f073f0eb4d7d8d934c97fd94c7145ba8f6011730d85e0fa4795d5d8.jpg)  
Figure 5: Qualitative results on the Paris-Rue-Madame dataset [41]. TTAD improves denoising quality under real-world noise, removing scattered outliers, suppressing LiDAR scan-line artifacts, and restoring smooth and coherent surface structures.

![](images/2ba484b36cac797396fd7d8912bdbe75e5e56afc4d362478120f13aae77cf9ba.jpg)  
Figure 6: Average processing time comparison. The baseline model (blue) has a constant low runtime. Standard TTAD (green) incurs a constant high cost due to per-sample re-initialization. In contrast, CTTAD (orange) becomes increasingly eficient by reusing and incrementally updating parameters across samples.

In conclusion, these results validate our central hypothesis: by integrating knowledge from previous samples, CTTAD not only achieves superior amortized eficiency but also reaches a higher final denoising accuracy, proving its efectiveness for real-world, continuous data streams.

Table 4: Ablation study on components in TTAD. Results are averaged over three selected baseline denoising methods. "NNP" denotes the noisy-noisier pairing strategy. A checkmark (✓) indicates the module is enabled, and a cross (✗) indicates it is disabled. CD and P2M values are multiplied by 10<sup>4</sup>. The best results under each noise type are highlighted in bold.
<table><tr><td rowspan="2">RDNM</td><td rowspan="2">NNP</td><td colspan="2">2%Gaussian</td><td colspan="2">2%Pulse</td><td colspan="2">2%Exponential</td></tr><tr><td>CD</td><td>P2M</td><td>CD</td><td>P2M</td><td>CD</td><td>P2M</td></tr><tr><td>x</td><td>x</td><td>1.63</td><td>1.21</td><td>4.00</td><td>2.84</td><td>6.44</td><td>5.31</td></tr><tr><td>x</td><td>√</td><td>1.42</td><td>1.25</td><td>4.12</td><td>4.57</td><td>5.56</td><td>6.00</td></tr><tr><td>√</td><td>x</td><td>1.72</td><td>1.33</td><td>3.01</td><td>2.55</td><td>5.16</td><td>4.97</td></tr><tr><td>√</td><td>√</td><td>1.36</td><td>0.99</td><td>2.44</td><td>2.50</td><td>4.47</td><td>3.94</td></tr></table>

## 4.5. Ablation Study

## 4.5.1. Residual-driven Noise Modeling

We first assess the efectiveness of the RDNM module by comparing it against a simplified alternative that employs Gaussian noise to create noisy-noisier training pair during adapta tion.

As shown in Table 4, our RDNM consistently outperforms the Gaussian baseline across all synthetic noise types. The performance gap is particularly pronounced under unseen noise conditions (e.g., exponential noise), where the simplistic Gaussian assumption fails to capture the specific characteristics of test-time corruption. From the noise modeling perspective, Gaussian noise introduces spatially uniform and unimodal perturbations, limiting the diversity and realism of augmented inputs. In contrast, our approach estimates the actual noise distribution from residuals, which captures the multi-modal patterns critical for generating realistic test-time perturbations.

(b) Average P2M distanc  
![](images/8c510e1d6f6508a0f0c4273721c54b03eb5f71136ebabd66cbbcf5cbdc49c50d.jpg)  
(a) Average Chamfer distance

![](images/edcde74ff66948e5956cbf002bc88a01d5c41562a0c3b5ffeb8d12d51362aba1.jpg)

Figure 7: Performance comparison under unseen 2% Pulse noise, showing the average error tracked over a sequence of point clouds. While TTAD (green) shows stable, strong performance, CTTAD (orange) initially trades a minor performance dip for long-term gain, ultimately outperforming both baselines.  
![](images/3ae7f43a7156f64b33d2a235f03b07ccd4ae14be8ea0701a111b2624309e7d35.jpg)  
Figure 8: Visualization of noise direction distributions using spherical-to-polar projection. Left: Ground-truth input noise shows a structured pattern. Middle: Our RDNM captures the dominant directional structures efectively. Right: Standard Gaussian noise appears spatially uniform and lacks structure. Each dot represents a normalized noise direction projected to 2D; the denser areas indicate dominant noise directions.

To better illustrate the diferences in noise modeling, we visualize the directional distributions of three types of noise: the ground-truth input noise, our residual-driven noise sampled from the learned GMM, and standard Gaussian noise. We compute the residual vectors for each point, normalize them to obtain unit vectors, and map them to a 2D polar plane using a spherical projection. As shown in Figure 8, the input noise exhibits a highly structured distribution, reflecting complex and directional corruption patterns. Our modeled noise captures these directional structures with high fidelity, showing concentrated patterns that align closely with the ground-truth distribution. In contrast, the Gaussian noise is spatially uniform and lacks the directional specificity of real test-time noise. This confirms that our RDNM provides a more realistic and informative estimation of the actual noise distribution, which is critical for efective and robust test-time adaptation.

## 4.5.2. Noisy–Noisier Pairing

To evaluate the impact of sample-specific noisy–noisier pair with varying perturbation intensities, we design a baseline method which perturbs each input sample only once with a fixed-strength noise for comparison. Specifically, the consistency loss $\mathcal { L } _ { \mathrm { c o n s i s t } } .$ , which relies on a pair of augmented inputs, is consequently deactivated.

As reported in Table 4, our noisy-noisier pairing strategy consistently yields lower errors and improves generalization across all noise types compared to this baseline. The mechanism’s success can be attributed to the structured contrastive learning it enables. The less-corrupted noisy sample serves to anchor the optimization around a reliable pseudo-clean reference. Simultaneously, the more heavily-corrupted noisier counterpart introduces marked contrastive variation. By enforcing output consistency between these two versions via $\mathcal { L } _ { \mathrm { c o n s i s t } } ,$ , the model is compelled to learn features that are robust to varying noise intensities. Essentially, our pairing mechanism enhances the adaptation process by providing structured data diversity, guiding the model toward learning truly noise-invariant representations.

## 4.5.3. Parameter Update Strategy

We further assess the impact of our parameter update mechanism by comparing full gradient-based updates with the EMA strategy under 2% noise setting, where the reported values represent the average performance over three TTAD-compatible denoising backbones. As reported in Figure 9, although both methods improve performance over the original pre-trained denoising models, EMA yields more robust results, particularly under unseen noise types such as pulse and exponential noise. In addition, we test both variants under Gaussian noise, which is the same distribution used during training, to examine potential forgetting efects. The EMA version retains significantly better performance, suggesting that it helps preserve the sourcedomain knowledge while adapting to new noise. This supports our hypothesis that EMA acts as a regularizer, mitigating catastrophic forgetting by gradually incorporating new information without abruptly overwriting the learned priors. Moreover, EMA behaves as a temporal ensemble over the online network trajectory, smoothing noisy gradients caused by imperfect pseudo clean references. This stabilizes the adaptation process, especially when the pseudo-clean reference is inaccurate.

![](images/e63756c6c8c6126c0dc95edbdd63de485ff2db3dac5568ae6bc6923c41b88961.jpg)  
Figure 9: Performance comparison of diferent parameter update strategies on point clouds under Gaussian and unseen (Pulse, Exponential) synthetic noise. Results are averaged over three selected baseline denoising methods. Each group compares three strategies: Baseline (blue), Fully Update (orange), and EMA (green), evaluated by Chamfer Distance (top row) and Point-to-Mesh Distance (bottom row). Across all noise types, the adaptation strategy with EMA consistently improves generalization to unseen noise while maintaining source-domain performance on Gaussian noise.

## 4.5.4. Component Number of GMM

We further investigate the sensitivity of our residual-driven noise modeling to the number of Gaussian mixture components M. To this end, we evaluate the distribution Fitting Error (Chamfer Distance between the estimated GMM noise and the actual unseen mixed noise) and the corresponding Fit Time under varying M ∈ [1 15]. As illustrated in Figure 10, when $M \ < \ 5 .$ , the GMM struggles to capture the complex, multimodal nature of the mixed noise, yielding a large fitting error. Notably, at $M = 5$ , the fitting error experiences a significant drop and stabilizes. Beyond $M = 5 .$ , the fitting error plateaus with only marginal improvements, while the computational cost (Fit Time) grows substantially, particularly for $M > 1 0 .$ . Consequently, we set $M = 5$ as the optimal trade-of between noise modeling accuracy and test-time computational eficiency.

## 5. Conclusion and Future Work

We propose a Test-Time Adaptive Denoising (TTAD) framework to address the domain shift issue in point cloud denoising in this paper. By leveraging residual-driven noise estimation, noise-aware contrastive adaptation, and EMA strategies, TTAD enables pre-trained denoising models to adapt to unseen noise distributions at test time, without requiring additional supervision or retraining from source data. Extensive experiments demonstrate that TTAD consistently enhances the performance of state-of-the-art denoising models under diverse unseen noise conditions, achieving significant reductions in both Chamfer Distance and Point-to-Mesh metrics.

![](images/605a3ecda9b4d58789c035e7b6e567ca8741b882c61736981e91fa8591c9999a.jpg)  
Figure 10: Efect of GMM component number M on the distribution fitting error (blue curve) and computational time (red curve) under complex mixed noise.

Additionally, we show that the EMA update strategy outperforms full parameter updates, preserving performance of denoising models across domains. Beyond standard adaptation, our framework also supports a CTTAD mode that facilitates eficient online adaptation over data streams. This extension further enhances its practicality for real-time point cloud processing tasks.

Future work will primarily focus on enhancing the long-term stability of CTTAD, a central challenge in Online Test-Time Adaptation (OTTA). Our analysis revealed that when the underlying denoising model generates less reliable pseudo-clean references (an issue observed particularly with weaker baselines like ScoreDenoise [11]), the continual adaptation process can be susceptible to error accumulation over long data streams. To mitigate this, several avenues can be explored. One promising direction is to introduce uncertainty-aware update mechanisms, where the model adapts more cautiously when its confidence in the estimated clean reference is low. Another direction involves investigating strategic parameter reset policies or partial rollbacks to the original pre-trained state, preventing the model from drifting too far from its robust initial knowledge.

## Acknowledgments

The authors would like to acknowledge the helpful comments and kindly suggestions provided by anonymous referees.

## References

[1] C. Luo, X. Yang, A. Yuille, Self-supervised pillar motion learning for autonomous driving, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 3183–3192. doi:10.1109/ CVPR46437.2021.00320.

[2] Y. Jiang, H. Zhang, L. Li, D. Liu, Z. Li, Sparse point clouds assisted learned image compression, IEEE Transactions on Circuits and Systems for Video Technology 35 (5) (2025) 4000–4010. doi:10.1109/TCSVT.2024. 3522621.

[3] C. H. P. Nguyen, Y. Choi, Comparison of point cloud data and 3d CAD data for on-site dimensional inspection of industrial plant piping systems, Automation in Construction 91 (2018) 44–52. doi:10.1016/j.autcon.2018.03. 011.

[4] F. Xue, W. Lu, Z. Chen, C. J. Webster, From LiDAR point cloud towards digital twin city: Clustering city objects based on gestalt principles, ISPRS Journal of Photogrammetry and Remote Sensing 167 (2020) 418–431. doi:10.1016/j.isprsjprs.2020.07.005.

[5] J. Yu, C. Yu, C. Lin, F. Wei, Improved iterative closest point (ICP) point cloud registration algorithm based on matching point pair quadratic filtering, in: 2021 International Conference on Computer, Internet of Things and Control Engineering (CITCE), 2021, pp. 1–5. doi: 10.1109/CITCE54390.2021.00008.

[6] X. Cao, X. Gong, Q. Xie, J. Huang, Y. Xu, Y. Liu, J. Wang, Raw scanned point cloud registration with repetition for aircraft fuel tank inspection, Computer-Aided Design 144 (2022) 103164. doi:10.1016/j.cad.2021.103164.

[7] U. Fugacci, C. Romanengo, B. Falcidieno, S. Biasotti, Reconstruction and preservation of feature curves in 3d point cloud processing, Computer-Aided Design 167 (2024) 103649. doi:10.1016/j.cad.2023.103649.

[8] L. Zhou, G. Sun, Y. Li, W. Li, Z. Su, Point cloud denoising review: From classical to deep learning-based approaches, Graphical Models 121 (2022) 101140. doi: 10.1016/j.gmod.2022.101140.

[9] X. Wang, X. Fan, D. Zhao, PointFilterNet: A filtering network for point cloud denoising, IEEE Transactions on Circuits and Systems for Video Technology 33 (3) (2023) 1276–1290. doi:10.1109/TCSVT.2022.3207789.

[10] X. Wang, W. Cui, R. Xiong, X. Fan, D. Zhao, FCNet: Learning noise-free features for point cloud denoising, IEEE Transactions on Circuits and Systems for Video Technology 33 (11) (2023) 6288–6301. doi:10.1109/ TCSVT.2023.3266458.

[11] S. Luo, W. Hu, Score-based point cloud denoising, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 4583–4592. doi:10. 1109/ICCV48922.2021.00454.

[12] S. Luo, W. Hu, Diferentiable manifold reconstruction for point cloud denoising, in: Proceedings of the 28th ACM International Conference on Multimedia, 2020, pp. 1330– 1338. doi:10.1145/3394171.3413727.

[13] D. Zhang, X. Lu, H. Qin, Y. He, PointFilter: Point cloud filtering via encoder-decoder modeling, IEEE Transactions on Visualization and Computer Graphics 27 (2021) 2015–2027. doi:10.1109/TVCG.2020.3027069.

[14] D. de Silva Edirimuni, X. Lu, Z. Shao, G. Li, A. Robles-Kelly, Y. He, IterativePFN: True iterative point cloud filtering, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 13530–13539. doi:10.1109/CVPR52729.2023.01300.

[15] M.-J. Rakotosaona, V. La Barbera, P. Guerrero, N. J. Mitra, M. Ovsjanikov, PointCleanNet: Learning to denoise and remove outliers from dense point clouds, Computer Graphics Forum 39 (2020) 185–203. doi:10.1111/ cgf.13753.

[16] P. Hermosilla, T. Ritschel, T. Ropinski, Total denoising: Unsupervised learning of 3d point cloud cleaning, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019, pp. 52–60. doi:10.1109/ ICCV.2019.00014.

[17] Z. Su, C. Wang, K. Jiang, K. Jiang, W. Li, SITF: A selfsupervised iterative training framework for point cloud denoising, Computer-Aided Design 179 (2025) 103812. doi:10.1016/j.cad.2024.103812.

[18] W. Wang, X. Liu, H. Zhou, L. Wei, Z. Deng, M. Murshed, X. Lu, Noise4Denoise: Leveraging noise for unsupervised point cloud denoising, Computational Visual Media 10 (4) (2024) 659–669. doi:10.1007/s41095-024-0423-3.

[19] J. Liang, R. He, T. Tan, A comprehensive survey on test-time adaptation under distribution shifts, International Journal of Computer Vision 133 (1) (2025) 31–64. doi: 10.1007/s11263-024-02181-w.

[20] S. Mohan, J. L. Vincent, R. Manzorro, P. Crozier, C. Fernandez-Granda, E. Simoncelli, Adaptive denoising via gaintuning, Advances in Neural Information Processing Systems 34 (2021) 23727–23740. doi:10.5555/ 3540261.3542078.

[21] G. Vaksman, M. Elad, P. Milanfar, LIDIA: Lightweight learned image denoising with instance adaptation, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, 2020, pp. 524– 525. doi:10.1109/CVPR42600.2020.01356.

[22] M. A.-N. I. Fahim, J. Boutellier, SS-TTA: Test-time adaptation for self-supervised denoising methods, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 1178–1187. doi: 10.1109/CVPRW59228.2023.00125.

[23] C. Choy, J. Gwak, S. Savarese, 4d spatio-temporal convnets: Minkowski convolutional neural networks, in: 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 3070–3079. doi: 10.1109/CVPR.2019.00319.

[24] H. Tang, Z. Liu, X. Li, Y. Lin, S. Han, TorchSparse: Eficient point cloud inference engine, in: Proceedings of Machine Learning and Systems, Vol. 4, 2022. doi: 10.48550/arXiv.2204.10319.

[25] Ö. Çiçek, A. Abdulkadir, S. S. Lienkamp, T. Brox, O. Ronneberger, 3d U-Net: Learning dense volumetric segmentation from sparse annotation, in: Medical Image Computing and Computer-Assisted Intervention (MICCAI), Vol. 9901 of Lecture Notes in Computer Science, 2016, pp. 424–432. doi:10.1007/ 978-3-319-46723-8\_49.

[26] J. Choe, B. Joung, F. Rameau, J. Park, I. S. Kweon, Deep point cloud reconstruction, in: International Conference on Learning Representations, 2022. doi:10.48550/ arXiv.2111.11704.

[27] H. Chen, B. Du, S. Luo, W. Hu, Deep point set resampling via gradient fields, IEEE Transactions on Pattern Analysis and Machine Intelligence 45 (3) (2022) 2913–2930. doi: 10.1109/TPAMI.2022.3175183.

[28] A. Mao, Z. Du, Y.-H. Wen, J. Xuan, Y.-J. Liu, PD-Flow: A point cloud denoising framework with normalizing flows, in: European Conference on Computer Vision, Springer, 2022, pp. 398–415. doi:10.1007/ 978-3-031-20062-5\_23.

[29] A. Mao, B. Yan, Z. Ma, Y. He, Denoising point clouds in latent space via graph convolution and invertible neural network, in: Proceedings of the IEEE/CVF

Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 5768–5777. doi:10.1109/ CVPR52733.2024.00551.

[30] Y. Wang, A. Cheraghian, Z. Hayder, J. Hong, S. Ramasinghe, S. Rahman, D. Ahmedt-Aristizabal, X. Li, L. Petersson, M. Harandi, Backpropagation-free network for 3d test-time adaptation, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 23231–23241. doi:10.1109/ CVPR52733.2024.02192.

[31] T. Zou, S. Qu, Z. Li, A. Knoll, L. He, G. Chen, C. Jiang, HGL: Hierarchical geometry learning for test-time adaptation in 3d point cloud segmentation, in: Computer Vision – ECCV 2024, Vol. 15113 of Lecture Notes in Computer Science, Springer, Cham, 2024, pp. 19–36. doi: 10.1007/978-3-031-73001-6\_2.

[32] H. Shim, C. Kim, E. Yang, CloudFixer: Test-time adaptation for 3d point clouds via difusion-guided geometric transformation, in: Computer Vision – ECCV 2024, Vol. 15089 of Lecture Notes in Computer Science, Springer, Cham, 2024, pp. 454–471. doi:10.1007/ 978-3-031-72751-1\_26.

[33] H. Dastmalchi, A. An, A. Cheraghian, S. Rahman, S. Ramasinghe, Test-time adaptation of 3d point clouds via denoising difusion models, in: Proceedings of the Winter Conference on Applications of Computer Vision (WACV), 2025, pp. 1566–1576. doi:10.1109/ WACV61041.2025.00160.

[34] P. Du, X. Wang, Z. Wu, X. Ru, X. Granier, Y. He, LaPDA: Latent-space point cloud denoising with adaptivity, IEEE Transactions on Visualization and Computer Graphics (2025) 1–15doi:10.1109/TVCG.2025.3635138.

[35] S. Xie, J. Gu, D. Guo, C. R. Qi, L. Guibas, O. Litany, PointContrast: Unsupervised pre-training for 3d point cloud understanding, in: Computer Vision – ECCV 2020, Vol. 12348 of Lecture Notes in Computer Science, Springer, Cham, 2020, pp. 574–591. doi:10.1007/ 978-3-030-58580-8\_34.

[36] M. Afham, I. Dissanayake, D. Dissanayake, A. Dharmasiri, K. Thilakarathna, R. Rodrigo, CrossPoint: Selfsupervised cross-modal contrastive learning for 3d point cloud understanding, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 9902–9912. doi:10.1109/ CVPR52688.2022.00967.

[37] X. Li, J. Chen, J. Ouyang, H. Deng, C. He, X. Yang, A. Yuille, S. Soatto, X. Xue, Q. Tian, ToThePoint: Eficient contrastive learning of 3d point clouds via recycling, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 21781–21790. doi:10.1109/CVPR52729.2023.02086.

[38] X. Wu, X. Wen, X. Liu, H. Zhao, Masked scene contrast: A scalable framework for unsupervised 3d representation learning, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 9642–9651. doi:10.1109/ CVPR52729.2023.00908.

[39] H. Fan, H. Su, L. J. Guibas, A point set generation network for 3d object reconstruction from a single image, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017, pp. 605–613. doi: 10.1109/CVPR.2017.264.

[40] L. Yu, X. Li, C.-W. Fu, D. Cohen-Or, P.-A. Heng, PU-Net: Point cloud upsampling network, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp. 2790–2799. doi:10.1109/ CVPR.2018.00295.

[41] A. Serna, B. Marcotegui, F. Goulette, J.-E. Deschaud, Paris-rue-madame database: A 3d mobile laser scanner dataset for benchmarking urban detection, segmentation and classification methods, in: 4th International Conference on Pattern Recognition, Applications and Methods (ICPRAM 2014), 2014. doi:10.5220/ 0004934808190824.

[42] R. Li, X. Li, P.-A. Heng, C.-W. Fu, Point cloud upsampling via disentangled refinement, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 344–353. doi:10.1109/ CVPR46437.2021.00041.