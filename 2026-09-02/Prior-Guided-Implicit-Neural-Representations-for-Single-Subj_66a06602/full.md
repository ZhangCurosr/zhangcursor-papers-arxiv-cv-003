# Prior-Guided Implicit Neural Representations for Single-Subject Diffusion MRI Super-Resolution

Abdulkader Ghandoura<sup>∗1,2</sup>   
aghandoura@bwh.harvard.edu   
Marsil Zakour<sup>†2</sup>   
marsil.zakour@tum.de   
William Consagra<sup>‡1,3</sup>   
consagra@mailbox.sc.edu   
Yogesh Rathi<sup>‡1</sup>   
yogesh@bwh.harvard.edu

<sup>1</sup> Psychiatry Neuroimaging Laboratory Brigham and Women’s Hospital Harvard Medical School Boston, MA, USA

<sup>2</sup> School of Computation, Information and Technology Technical University of Munich Munich, Germany

<sup>3</sup> Department of Statistics University of South Carolina Columbia, SC, USA

## Abstract

Resolving complex fiber geometries in brain white matter requires high-resolution diffusion MRI at the cost of long acquisition times. This leads many clinical protocols to opt for low-resolution scans, making downstream microstructure estimation and tractography challenging. Implicit neural representations (INRs) can model the diffusion signal continuously, enabling native single-subject super-resolution by querying the network at arbitrary spatial coordinates, yet existing methods often suffer from long training times and lack a mechanism to incorporate anatomical priors to regularize superresolution by constraining the space of plausible reconstructions. To address these limitations, we propose a novel transfer-learning framework that pre-trains an INR on a high-resolution template and then adapts it to subject-specific scans via registration and fine-tuning. For 4× through-plane super-resolution from 5mm to 1.25mm on Human Connectome Project (HCP) data, our method reduces NRMSE by 36–49% and increases FSIM by 24–43% over a recent baseline with 6× faster training, outperforming competing INR-based methods across both image quality and domain-specific metrics. Code is available on the project page at https://abdulkaderghandoura.github.io/ research/msc-thesis/.

## 1 Introduction

Diffusion MRI (dMRI) is a non-invasive imaging technique that is used to probe the structure of brain tissue in vivo. The diffusion signal is related to the direction and magnitude of local water molecule diffusion, enabling the in vivo characterization of complex microstructural features and the tracing of large-scale white matter fiber pathways via tractography. Due to these capabilities, there is significant interest in using dMRI in the study of neurodegenerative diseases, neuropsychiatric disorders, and cognitive impairment [14, 18, 29].

High-spatial-resolution diffusion MRI requires long acquisition times to achieve reasonable signal-to-noise ratios (SNR), pushing many clinical protocols toward thick-slice acquisitions to maintain SNR within feasible scanning durations, at the expense of through-plane spatial resolution. This loss in resolution causes standard reconstruction techniques to fail in recovering the fine-grained anatomical details needed for downstream analysis, such as tractography or microstructure estimation.

To address the limitations of microstructure inference and tractography in clinical acquisitions, we propose a super-resolution framework that reconstructs high-spatial-resolution dMRI signals from low-resolution clinical-grade scans. Building on recent advances in implicit neural representations (INRs) for dMRI [7, 9, 13], we parameterize the diffusion signal with a resolution-agnostic, orientation-consistent neural representation. The network is defined continuously over the image domain, naturally supporting super-resolution at arbitrary scales. Since super-resolution from very thick slices is severely ill-posed due to irreversible loss of detail, our proposed framework leverages transferable anatomical priors from a high-resolution template via registration-guided fine-tuning to estimate subjectspecific INRs. This has the added benefit of amortizing computational cost through template pre-training, enabling fast super-resolution. Our main contributions are as follows.

• A novel template-based transfer-learning framework that leverages anatomical priors via registration-guided fine-tuning for dMRI super-resolution.

• A demonstration of 6× faster training and superior perceptual quality compared to recent INR approaches on 4× through-plane super-resolution.

## 2 Related Work

## 2.1 Implicit Neural Representations

Implicit neural representations (INRs), or neural fields, are neural network-based parameterizations of continuous functions that map spatial or spatiotemporal coordinates to corresponding field values, such as distance, color, or density. While INRs have been widely used in computer vision for 3D shape representation [20], novel view synthesis [17], and image super-resolution [6], their adoption in dMRI is relatively limited; [7] employs a global SIREN-based [21] INR to model single-shell dMRI and proposes an approach for predictive uncertainty quantification; [13] integrates multi-shell constrained spherical deconvolution into an INR framework to continuously predict fiber orientation distribution functions (fODFs); [9] employs multi-resolution hash encoding [19], a hierarchy of discrete grids with a compact MLP head, demonstrating significantly faster training compared to global parameterizations, but at the cost of gradient updates primarily affecting grid entries at supervised positions. All current INR-based methods train from scratch per subject and lack a mechanism for incorporating anatomical prior information.

![](images/870e0678a15f95bce5bf0e0fba7a61a8b2f7a3da55efa6525f52dd619b7a5b55.jpg)  
Figure 1: Three-stage framework: (1) Pre-training an INR on a high-resolution dMRI template. (2) Subject-template registration via a composite transformation T. (3) Fine-tuning to subject anatomy using low-resolution data.

## 2.2 Diffusion MRI Super Resolution

Diffusion MRI data is collected as a 4D array $n _ { x } \times n _ { y } \times n _ { z } \times M$ , where $n _ { x } , n _ { y } , n _ { z }$ are the number of grid points in the X −Y −Z spatial dimensions, and M is the number of diffusion-weighted volumes, each corresponding to a b-vector in q-space $p \in S ^ { 2 }$ , indicating the direction of the applied magnetic gradient, and diffusion weighting b-value $b \geq 0$ related to the amplitude, duration, and separation of the applied diffusion-encoding gradient pulses. Therefore, dMRI super-resolution targets spatial or angular resolution, or both.

Most existing work focuses either on spatial super-resolution of parameter maps estimated from the raw dMRI signal, such as the diffusion tensor [1, 24], or on angular superresolution of the raw dMRI signal [10, 15]. For downstream analyses, such as tractography, arbitrary-scale spatial super-resolution on the dMRI signal is desirable, since it enables streamline propagation at any chosen resolution.

Because INRs represent images as continuous functions of space, they natively enable “zero-shot” super-resolution via evaluation at arbitrary spatial coordinates, making them a natural choice for this task. INR-based approaches developed specifically for spatial dMRI super-resolution have so far been limited to supervised approaches [23, 28], in which the INR is trained as a decoder alongside a traditional deep encoder. However, this reliance on supervision can be problematic in clinical settings, where pathology may induce a substantial distribution shift relative to the training data.

## 3 Methodology

Figure 1 illustrates our prior-guided INR-based approach to single-subject dMRI superresolution. In Stage 1 (Section 3.1), an INR is fit once to a high-resolution dMRI scan from a healthy subject, which will then serve as an anatomical template space prior for superresolution on low-resolution subjects. In Stage 2 (Section 3.2), a non-rigid registration is performed to map spatial coordinates from the subject space to corresponding locations in the high-resolution prior template space. The subject-specific super-resolution is performed in Stage 3 (Section 3.3), where the pre-trained INR is injected into the subject-specific anatomy as a high-resolution prior and then fine-tuned to the new low-resolution data.

## 3.1 High Resolution Template Space INR Prior

Spherical harmonics provide a complete orthonormal basis for square-integrable functions on $S ^ { 2 }$ . The complex spherical harmonics are defined as

$$
Y _ { l } ^ { m } ( \theta , \varphi ) = { \sqrt { \frac { ( 2 l + 1 ) ( l - m ) ! } { 4 \pi ( l + m ) ! } } } P _ { l } ^ { m } ( \cos { \theta } ) e ^ { i m \varphi }\tag{1}
$$

for all non-negative integer orders $l \geq 0$ and integer phase factors $\begin{array} { r } { m = - l , \ldots , l , } \end{array}$ , where $P _ { l } ^ { m }$ are the associated Legendre polynomials. Due to the antipodal symmetry of diffusion, where water molecules diffuse equally in opposite directions, the ODF satisfies $g _ { \nu } ( - p ) = g _ { \nu } ( p )$ for all directions $p \in S ^ { 2 }$ . This symmetry property ensures that only even-order harmonics contribute to the ODF representation.

The real-symmetric spherical harmonics form a set of basis functions $\{ \phi _ { k } \} _ { k = 1 } ^ { K }$ for antipodally symmetric functions, constructed from the complex harmonics as

$$
\phi _ { k } ( p ) = { \left\{ \begin{array} { l l } { { \sqrt { 2 } } \mathrm { R e } ( Y _ { l } ^ { m } ( p ) ) } & { { \mathrm { i f } } m < 0 } \\ { Y _ { l } ^ { 0 } ( p ) } & { { \mathrm { i f } } m = 0 } \\ { { \sqrt { 2 } } \mathrm { I m } ( Y _ { l } ^ { m } ( p ) ) } & { { \mathrm { i f } } m > 0 } \end{array} \right. }\tag{2}
$$

for even orders $l = 0 , 2 , 4 , \ldots , L$ and phase factors $m = - l , \ldots , 0 , \ldots , l ,$ where the index k is given by $k = ( l ^ { 2 } + l + 2 ) / 2 + m$ . A key property is that these basis functions are non-zero eigenfunctions of the Funk-Radon transform, which relates the ODF to the diffusion signal, with eigenvalue $2 \pi P _ { l } ( 0 )$ , where $P _ { l }$ is the ordinary Legendre polynomial of order l.

We use the NODF model proposed in [7] to parameterize the diffusion MRI signal, which is modeled as a continuous function of spatial location $\nu \in \mathbb { R } ^ { 3 }$ and q-space orientation $p \in S ^ { 2 }$ using the finite series expansion

$$
g ( \nu , p ) = \sum _ { k = 1 } ^ { K } c _ { \theta , k } ( \nu ) \phi _ { k } ( p ) ,\tag{3}
$$

where $\{ \phi _ { k } \} _ { k = 1 } ^ { K }$ are the first K real-symmetric spherical harmonic basis functions [8], and $\boldsymbol { c } _ { \theta } ( \nu ) = ( c _ { \theta , 1 } \sp { \circ } ( \nu ) , \ldots , c _ { \theta , K } ( \nu ) ) ^ { \intercal }$ is the corresponding spherical harmonic spatial coefficient field, parameterized by a SIREN-based INR [21].

Let $\Phi \in \mathbb { R } ^ { M \times K }$ be the real-symmetric spherical harmonic basis evaluation matrix at the M gradient directions with $\Phi _ { m k } = \phi _ { k } ( p _ { m } )$ , mapping the K-dimensional SH coefficient vector to the M-dimensional predicted diffusion signal along each gradient direction. We model the observed diffusion signal $\mathbf { y } _ { i } \in \mathbb { R } ^ { M }$ at voxel i with center spatial coordinate $\nu _ { i } \in \mathbb { R } ^ { 3 }$ using a Gaussian measurement noise model with $\pmb { \varepsilon } _ { i } \sim \mathcal { N } ( 0 , \sigma _ { e } ^ { 2 } I _ { M } )$ as

$$
\mathbf { y } _ { i } = \Phi c _ { \theta } \big ( \nu _ { i } \big ) + \pmb { \varepsilon } _ { i } .\tag{4}
$$

Training minimizes, over mini-batches of voxels of size N, a loss combining data fidelity (term 1) and an angular smoothness prior (term 2):

$$
\widehat { \theta } _ { H R } = \underset { \theta } { \mathrm { a r g m i n } } \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( \| \mathbf { y } _ { i } - \Phi c _ { \theta } ( \nu _ { i } ) \| _ { 2 } ^ { 2 } + \lambda _ { c } c _ { \theta } ( \nu _ { i } ) ^ { \top } R _ { \gamma } c _ { \theta } ( \nu _ { i } ) \right) ,\tag{5}
$$

where $R _ { \gamma } \in \mathbb { R } ^ { K \times K }$ is the diagonal precision matrix encoding the angular prior (Section 3.1 of [7]), with $[ R _ { \gamma } ^ { - 1 } ] _ { k k } = s _ { \gamma } \big ( \sqrt { l _ { k } ( l _ { k } + 1 ) } \big )$ for $s _ { \gamma }$ the spectral density of the spherical Matérn prior with parameters γ and for $l _ { k }$ the harmonic order of $\phi _ { k }$ , and $\lambda _ { c }$ controls the strength of the angular prior regularization.

## 3.2 High-Resolution Template Prior Injection to Subject Space

To inject the high-resolution INR-based prior into the low-resolution subject space for superresolution, we first register the high-resolution template (moving) to the low-resolution subject (fixed) using registration foundation model uniGradICON [25] with the local normalized cross-correlation (LNCC) similarity metric. This establishes a subject-to-template mapping, following the ITK conventions [2]. Briefly, uniGradICON first resamples both volumes to $1 7 5 ^ { 3 }$ voxels via trilinear interpolation, and then refines the alignment by minimizing 1 − LNCC. The resulting transformation maps fixed-space point $\tilde { \nu } \in \mathbb { R } ^ { 3 }$ to its moving-space correspondence v through three stages. First, an affine $\mathcal { T } _ { 0 }$ transforms to network space: $\nu _ { \mathrm { n e t } } = \mathbf { M } _ { 0 } \tilde { \nu } + \mathbf { t } _ { 0 }$ . Second, the displacement vector field $\mathcal { T } _ { 1 }$ is applied: $\nu _ { \mathrm { n e t } } ^ { \prime } = \nu _ { \mathrm { n e t } } + \delta ( \nu _ { \mathrm { n e t } } )$ where $\delta : \mathbb { R } ^ { 3 }  \mathbb { R } ^ { 3 }$ is interpolated in network space. Third, an affine $\mathcal { T } _ { 2 }$ maps to moving space: $\nu = \mathbf { M } _ { 2 } \nu _ { \mathrm { n e t } } ^ { \prime } + \mathbf { t } _ { 2 }$ . The complete transform is $\mathcal { T } = \mathcal { T } _ { 2 } \circ \mathcal { T } _ { 1 } \circ \mathcal { T } _ { 0 }$

Before registration, we used [7] with the hyperparameters proposed in Section 4 to resample the subject scan at high resolution. We then obtained the generalized fractional anisotropy (GFA) for registration to focus on white-matter structures, though alternatives, such as non-diffusion-weighted images $( b = 0 \ s / m m ^ { 2 } )$ and fractional anisotropy (FA), are also suitable. Figure 2 illustrates the resulting deformation for a randomly selected subject: T displaces points within and across the axial plane, reflecting the full three-dimensional anatomical realignment required between the subject and the template coordinate systems.

## 3.3 Fine Tuning on Low-Resolution Subject

We now introduce a transfer-learning formulation that uses the high-resolution template INR from Section 3.1 together with the foundation model from Section 3.2 to form a prior for single-subject super-resolution. The model is fine-tuned to enforce data consistency with the measured low-resolution signal, enabling single-subject super-resolution at arbitrary target voxel sizes. Specifically, for subject space voxel i, denote its spatial footprint $\tilde { \Omega } _ { i }$ . Mapping points to template space via the deformable transform, we approximate the downsampling operator by Monte-Carlo integration

$$
\tilde { c } _ { i } = \frac { 1 } { | \tilde { \Omega } _ { i } | } \sum _ { \tilde { \nu } ^ { ( i ) } \in \tilde { \Omega } _ { i } } c _ { \theta } \left( \mathcal { T } ( \tilde { \nu } ^ { ( i ) } ) \right) \in \mathbb { R } ^ { K } ,\tag{6}
$$

where $\tilde { \nu } ^ { ( i ) }$ are sampled uniformly over $\tilde { \Omega } _ { i }$ . Although our experiments use uniform slice averaging, the operator extends to any slice-excitation profile as a non-uniform weighting over the voxel footprint, with uniform averaging as the special case. For a mini-batch of

![](images/26d073985e4ab1c8cb5afcaa86ffa9c97912a67abb77fbe4f49500ccb09642b3.jpg)  
Figure 2: Deformed grid of an axial slice after applying the displacement field. A regular 128 × 128 grid created in the fixed image space was transformed through T, resulting in curved grid lines that visualize the spatial deformation. Grid line colors encode the z-axis displacement of voxels, revealing a through-plane deformation pattern for a random subject.

N voxels with S samples per voxel, evaluating (6) requires NS forward passes through c , i.e., S× the per-iteration cost of single-point evaluation at transformed voxel centers. To bound this overhead, we precompute a dense set of deformed footprint points $\mathcal { T } ( \tilde { \nu } ^ { ( i ) } )$ once at initialization and randomly draw S of these points per voxel at each iteration, bounding GPU memory while preserving accuracy (Section 5.3). Let $\tilde { \mathbf { y } } _ { i } \in \mathbb { R } ^ { M }$ denote the observed lowresolution diffusion signal at voxel i. We fine-tune the network parameters θ by gradient updates according to the loss over mini-batches of N voxels:

$$
\mathcal { L } ( \pmb { \theta } ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \Big ( \| \tilde { \mathbf { y } } _ { i } - \Phi \tilde { c } _ { i } \| _ { 2 } ^ { 2 } + \lambda _ { c } \tilde { c } _ { i } ^ { \top } R _ { \gamma } \tilde { c } _ { i } + \lambda _ { \mathrm { t v } } \| \nabla _ { \mathbb { R } ^ { 3 } } c _ { \pmb { \theta } } \| _ { 1 } \Big ) ,\tag{7}
$$

where $\lambda _ { \mathrm { t v } }$ controls the strength of the spatial total-variation (TV) penalty, which is included here to regularize the spatial downsampling. Optimization of (7) is performed with Adam initialized at $\widehat { \theta } _ { H R }$ , thereby injecting the high-resolution template space prior. Arbitrary-scale spatio-angular super-resolution at any subject location ˜v is then obtained via (3) using the fine-tuned network parameters.

## 4 Experimental Setup

Data: We used the publicly available HCP-YA dataset [22, 27] with $1 . 2 5 ^ { 3 }$ mm isotropic voxels acquired in a three-shell scheme with 90 directions per shell and 18 non-diffusionweighted images. The dMRI data were processed with the HCP minimal preprocessing pipeline [12, 22]. For all experiments, we used only the $b = 1 , 0 0 0 ~ \mathrm { s / m m } ^ { 2 }$ shell for high SNR. We randomly selected one subject as the high-resolution scan for training the templatespace INR as outlined in Section 3.1. To mimic common clinical protocols with low out-ofplane resolution, we randomly selected 10 test HCP subjects and generated anisotropic lowresolution by averaging every 4 consecutive axial slices, yielding $1 . 2 5 \times 1 . 2 5 \times 5$ mm input voxels (4× through-plane reduction). The original high-resolution isotropic test volumes served as ground truth (GT) for evaluation.

Competing Methods: To evaluate single-subject dMRI spatial super-resolution, we compare our method with two recent representative INR-based baselines: NODF [7] and NODF-HashEnc [9]. Both are trained using the default configuration reported in the latter. Because these models represent the signal as a continuous function of space, they support arbitraryscale spatial super-resolution by querying the trained network at any spatial coordinate. Supervised CNN-based methods are excluded as spatial super-resolution methods for the dMRI signal remain scarce [1, 24], and none support arbitrary-scale upsampling, which is essential for tractography at any chosen resolution. Moreover, such multi-subject supervision is problematic under the clinical distribution shift discussed in Section 2.2. In contrast, our template is a single anatomical prior, injected by registration and fully adapted to the subject’s own data, involving no cross-subject supervision.

Evaluation Metrics: We use the ground-truth high-resolution data to compare the methods’ recovery of microstructural features and orientation content from the low-resolution test images. For microstructure, we compare spatial feature maps of GFA and fractional anisotropy (FA), using standard image quality metrics: NRMSE, FSIM, PSNR, and SSIM. For orientational content, we compare the ODF $L _ { 2 }$ error and the angular error of the principal eigenvector of the diffusion tensor. We also perform qualitative comparisons of tractography from the super-resolved estimates vs. tractography on the ground truth data.

Implementation Details: Our template space INR uses a SIREN architecture with the same configuration as outlined above for the NODF competitor. We set 10 fully-connected layers, uniform width $r = 1 { , } 0 2 4$ , mini-batch $N = 1 2 8$ , learning rate $1 0 ^ { - 6 } , \lambda _ { c } = 3 . 7 6 \times 1 0 ^ { - 7 }$ , and $\omega _ { 0 } = 6 0$ for both template training (5) and subject-specific fine-tuning (7). For the latter, we apply total variation regularization $\lambda _ { t \nu } = 2 \times 1 0 ^ { - 3 }$ and use $S = 6$ Monte Carlo samples per voxel in (6). Training for all models was performed on an NVIDIA GeForce GTX 1080 Ti GPU with 11 GB VRAM. Image registration was carried out on a separate NVIDIA RTX A6000 GPU with 48 GB VRAM. The network has 10.5 million trainable parameters, yielding 42 MB of model weights. As each method requires independent optimization for each test subject, the training time directly reflects the per-subject deployment cost. The templatespace INR converges to a faithful reconstruction of the high-resolution input signal. At completion of training, the FA and GFA maps derived from the fitted INR are visually indistinguishable from those computed directly on the raw HCP template using DIPY [11], with an NRMSE of 0.0349 on FA and 0.0782 on GFA, confirming that Stage 1 (Section 3.1) provides a high-fidelity anatomical prior for subsequent registration and fine-tuning.

Table 1: Quantitative results across training time on 10 randomly selected HCP subjects. Our method achieves the lowest NRMSE and the highest FSIM with consistent improvements for PSNR and SSIM, particularly on FA maps. Mean ± standard deviation.
<table><tr><td>Metric</td><td>Map</td><td>Method</td><td>5 minutes</td><td>10 minutes</td><td>30 minutes</td></tr><tr><td rowspan="5">NRMSE↓</td><td rowspan="4">FA</td><td>Baseline</td><td>0.283 ±0.0031</td><td>0.279 ±0.0034</td><td>0.216 ±0.0114</td></tr><tr><td>HashEnc</td><td>0.283 ±0.0046</td><td>0.228 ±0.0309</td><td>0.182 ±0.0044</td></tr><tr><td>Proposed</td><td>0.149 ±0.0028</td><td>0.143 ±0.0031</td><td>0.138 ±0.0029</td></tr><tr><td>Baseline</td><td>0.193 ±0.0398</td><td>0.194 ±0.0394</td><td>0.150 ±0.0280</td></tr><tr><td rowspan="3">GFA</td><td>HashEnc</td><td>0.206 ±0.0408</td><td>0.161 ±0.0272</td><td>0.131 ±0.0254</td></tr><tr><td>Proposed</td><td>0.104 ±0.0187</td><td>0.100 ±0.0180</td><td>0.096 ±0.0168</td></tr><tr><td>Baseline</td><td>0.430 ±0.0090</td><td>0.432 ±0.0105</td><td>0.508 ±0.0139</td></tr><tr><td rowspan="4">FSIM↑</td><td>FA</td><td>HashEnc</td><td>0.473 ±0.0250</td><td>0.551 ±0.0351</td><td>0.612 ±0.0038</td></tr><tr><td></td><td>Proposed</td><td>0.606 ±0.0046</td><td>0.617 ±0.0053</td><td>0.630 ±0.0045</td></tr><tr><td rowspan="3">GFA</td><td>Baseline</td><td>0.442 ±0.0084</td><td>0.437 ±0.0167</td><td>0.468 ±0.0225</td></tr><tr><td>HashEnc</td><td>0.436 ±0.0172</td><td>0.517 ±0.0481</td><td>0.581 ±0.0183</td></tr><tr><td>Proposed</td><td>0.579 ±0.0149</td><td>0.588 ±0.0116</td><td>0.598 ±0.0130</td></tr><tr><td rowspan="6">PSNR (dB) ↑</td><td rowspan="3">FA</td><td>Baseline</td><td>20.453 ±0.8323</td><td>19.939 ±1.2143</td><td>21.267 ±0.5898</td></tr><tr><td>HashEnc</td><td>19.502 ±1.8340</td><td>21.159 ±1.0218</td><td>22.338 ±0.5027</td></tr><tr><td>Proposed</td><td>22.963 ±0.3626</td><td>23.278 ±0.3682</td><td>23.627 ±0.3835</td></tr><tr><td rowspan="3">GFA</td><td>Baseline</td><td>23.793 ±1.5542</td><td>22.454 ±2.2521</td><td>22.255 ±2.0966</td></tr><tr><td>HashEnc</td><td>22.364 ±1.9013</td><td>22.779 ±1.6999</td><td>23.208 ±1.8943</td></tr><tr><td>Proposed</td><td>23.151 ±2.6809</td><td>23.019 ±2.3827</td><td>23.268 ±2.1518</td></tr><tr><td rowspan="6">SSIM ↑</td><td rowspan="3">FA</td><td>Baseline</td><td>0.737 ±0.0181</td><td>0.734 ±0.0206</td><td>0.751 ±0.0147</td></tr><tr><td>HashEnc</td><td>0.745 ±0.0190</td><td>0.769 ±0.0175</td><td>0.803 ±0.0112</td></tr><tr><td>Proposed</td><td>0.807 ±0.0116</td><td>0.816 ±0.0105</td><td>0.830 ±0.0103</td></tr><tr><td rowspan="3">GFA</td><td>Baseline</td><td>0.773 ±0.0244</td><td>0.747 ±0.0351</td><td>0.739 ±0.0230</td></tr><tr><td>HashEnc</td><td>0.740 ±0.0259</td><td>0.748 ±0.0237</td><td>0.763 ±0.0283</td></tr><tr><td>Proposed</td><td>0.761 ±0.0399</td><td>0.760 ±0.0321</td><td>0.767 ±0.0302</td></tr></table>

## 5 Results

## 5.1 Quantitative and Qualitative Evaluation

Table 1 reports the results for the FA and GFA errors averaged over 10 test subjects for all methods as a function of training time, reflecting the practical per-subject deployment cost. Strikingly, our method achieves better performance in 5 minutes than the competing methods NODF and NODF-HashEnc even after 30 minutes of training, reducing NRMSE by 36–49% and increasing FSIM by 24–43% compared to NODF, while improving PSNR by 11–17% and SSIM by 9–11% for FA, with modest GFA gains at longer training budgets (see [5, 16] for well-documented PSNR and SSIM limitations in medical imaging). We attribute this to the template-space prior introduced by our transfer-learning mechanism. Figure 3 summarizes the distributions of subject-wise median errors across three diffusion metrics on the test subjects using violin plots. Our method achieves lower ODF reconstruction error and smaller leading-diffusion-eigenvector angular error than both NODF and NODF-HashEnc, while converging substantially faster. The low inter-subject variability observed across all metrics (Table 1 and Figure 3) indicates that these gains are consistent across subjects. To formally assess significance, we ran a two-sided paired Wilcoxon signed-rank test across the 10 subjects, Holm-corrected within each training-time budget. Our method significantly outperforms both baselines on all seven primary metrics (FA/GFA NRMSE, FA/GFA FSIM, ODF-L2, GFA absolute error, and angular error) at every budget $( \alpha = 0 . 0 5 )$ , and FA PSNR and FA SSIM are likewise significant.

![](images/b1737de85879733dcce6c8432de060c83ce8e0732f7287442eda1e3f3f149946.jpg)

![](images/afa733ff3ed91fe6750a190f29f7be4a775d74886f539ee8dbbad289e62c751e.jpg)

![](images/c22c04f8adbf712cb565b2ea9891fcd17bb3126f3b577a317901022798e37a05.jpg)  
Figure 3: Distribution of subject-wise median on 10 HCP subjects at 5, 10, and 30 minutes for ODF L2 norm error (Proposed 0.08–0.09 vs. Baseline (NODF) 0.11–0.23, HashEnc 0.12–0.13), GFA absolute error (0.023–0.026 vs. Baseline/HashEnc 0.04–0.08), and eigenvector angular error $( 1 7 - 2 0 ^ { \circ }$ vs. Baseline $2 8 { - } 4 7 ^ { \circ }$ , HashEnc 19–52<sup>◦</sup>).

The top rows of Figure 4 compare the spatial super-resolution quality using DTI-derived images. After 1 hour of training, NODF produces noticeably blurry reconstructions, while NODF-HashEnc produces sharper reconstructions but introduces blocking artifacts consistent with the hash-grid parameterization. In contrast, our method achieves reconstructions that much more closely match the GT images after only 5 minutes of fine-tuning. The middle rows of Figure 4 show the fODF fields obtained by running CSD [26] on the super-resolved dMRI. Echoing the DTI results, we see that our method more closely reconstructs the spatioangular content in the GT. The bottom row of Figure 4 shows the tractography results obtained from the reconstructed fODF fields. While NODF is able to reconstruct most of the major pathways, it misses some tracts near the white-gray matter interface, likely due to the oversmoothing effect observed in the DTI images. NODF-HashEnc is able to reconstruct more tracts near the white-gray interface compared to NODF, but suffers from an overall reduction in streamlines, likely due to early termination from the blocking artifacts. Our method produces tracts that much more closely resemble those obtained from the GT, with higher streamline density and better coverage near the white-gray matter interface.

## 5.2 Generalization to Clinical Data

To assess robustness beyond the HCP training distribution, we apply our method to a real clinical dMRI acquisition at $0 . 9 3 7 5 \times 0 . 9 3 7 5 \times 6$ mm, for which appropriate informed consent was obtained. The clinical data were processed with luigi-pnlpipe [3], a publicly available pipeline. As isotropic ground-truth acquisitions are unavailable for real clinical scans, this evaluation is necessarily qualitative.

Compared to the HCP experiments $( 1 . 2 5 \times 1 . 2 5 \times 5$ mm input with 4× through-plane super-resolution), this case presents a non-integer 4.8× through-plane upsampling factor, higher in-plane resolution, and realistic acquisition noise. The super-resolution target is set to $0 . 9 3 7 5 \times 0 . 9 3 7 5 \times 1 . 2 5$ mm, with the HCP template’s resolution in the through-plane direction and the subject’s higher in-plane resolution. We do not include the template in this comparison because anatomical differences with the clinical subject make a direct visual comparison uninformative. The axial slice shown in the fifth column of Figure 5 corresponds approximately to the same anatomical level as the low-resolution input, with minor positional differences arising from the shift to the finer 1.25 mm through-plane resampling grid.

![](images/a77e9ac6ae355a0f5a028437cd0fb0f65633c2cc4f61a5b2dd673fe7edb03c22.jpg)  
Figure 4: Reconstruction and runtime. The figure shows DTI (sagittal, zoomed cerebellum), fODFs overlaid on GFA (coronal, arrow indicates bundles), and tractography (axial) for a missing slice from anisotropic data. Our method recovers fine details in 5 min (plus ≈7 min initialization), closer to GT than both baselines trained for 1 h.

Figure 5 shows DTI color maps in three anatomical views. The fourth column (INR fitting and registration without fine-tuning) is discussed in Section 5.3. Across all views, the proposed method recovers the right-anterior lesion, demonstrating that fine-tuning adapts the template prior to subject-specific anatomy even under more severe through-plane information loss and outside the training distribution. Reconstruction is driven by the per-subject data-consistency loss in (7), with the template acting only as initialization and regularization. Because the template is healthy and, unlike supervised multi-subject models, there is no lesion-bearing training corpus from which anatomy could leak, recovered pathology can only originate from the subject’s own signal. This remains a single qualitative case, however, and quantitative and expert validation on pathological cohorts is important future work.

Raw Signal  
Raw DTI  
Interpolation  
Registration  
Proposed  
![](images/60b1ad71d91ce91b0e6cdae7e964e5b70136e6ca8bf20b33c535a9d85ac27d1a.jpg)  
Figure 5: Qualitative evaluation on a real clinical dMRI acquisition. Yellow circles mark the lesion in the right anterior region of the brain. The third column shows trilinear interpolation. The lesion is absent in the fourth column (Stages 1–2 of Section 3), where the healthy template prior has not yet been adapted to subject-specific anatomy via fine-tuning. Non-brain regions (e.g., eyes) have been masked in columns 2–5. Appropriate informed consent was obtained for this acquisition.

## 5.3 Ablation Study

We validate the three key components of the proposed framework.

Registration and template prior: Removing the template initialization is equivalent to training the INR from scratch on the subject’s low-resolution data alone. Table 1 provides this comparison, where NODF and NODF-HashEnc are both optimized from scratch.

Fine-tuning: The fourth column in Figure 5 shows the reconstruction obtained by querying the registered template INR in the subject coordinates without performing Stage 3 (Section 3.3). The absence of the subject-specific lesion in that column and its recovery in the fifth column of the same figure after fine-tuning demonstrate that Stage 3 is essential for adapting the prior to individual anatomy.

Downsampling operator: To assess the Monte Carlo integration over the deformed voxel footprint $\tilde { \Omega } _ { i }$ in Equation (6), we compare it against a variant that replaces the integration with single-point evaluation at each transformed voxel center during fine-tuning. Table 2 reports results on a randomly selected HCP scan using the same template as in Section 5.1. Removing the downsampling operator consistently degrades all metrics across checkpoints, with NRMSE increasing by up to 8.6% and FSIM decreasing by up to 4.5% on GFA at 30 minutes, confirming that explicitly modeling the spatial extent of each thick slice is important for accurate reconstruction.

Table 2: Quantitative comparison of the proposed downsampling operator versus singlepoint evaluation at transformed voxel centers across different fine-tuning checkpoint durations. Percentages indicate relative change from the proposed method.
<table><tr><td>Metric</td><td>Map</td><td>Method</td><td>5 min</td><td>10 min</td><td>30 min</td></tr><tr><td rowspan="3">NRMSE↓</td><td>FA</td><td>Proposed Ablation</td><td>0.146 0.151 (+3.39%)</td><td>0.139 0.145 (+4.68%)</td><td>0.135 0.145 (+8.05%)</td></tr><tr><td></td><td>Proposed</td><td>0.105</td><td>0.100</td><td>0.098</td></tr><tr><td>GFA</td><td>Ablation</td><td>0.106 (+0.70%)</td><td>0.106 (+5.59%)</td><td>0.106 (+8.61%)</td></tr><tr><td rowspan="3">FSIM ↑</td><td>FA</td><td>Proposed Ablation</td><td>0.602</td><td>0.613</td><td>0.625</td></tr><tr><td></td><td>Proposed</td><td>0.596 (-1.00%) 0.574</td><td>0.607 (-0.96%)</td><td>0.616 (-1.41%)</td></tr><tr><td>GFA</td><td>Ablation</td><td>0.559 (-2.63%)</td><td>0.583 0.561 (-3.82%)</td><td>0.591 0.564 (-4.48%)</td></tr><tr><td rowspan="3">PSNR (dB) ↑</td><td>FA</td><td>Proposed</td><td>22.767</td><td>23.256</td><td>23.604</td></tr><tr><td></td><td>Ablation</td><td>22.535 (-1.02%)</td><td>22.900 (-1.53%)</td><td>22.863 (-3.14%)</td></tr><tr><td>GFA</td><td>Proposed</td><td>22.265</td><td>22.341</td><td>22.409</td></tr><tr><td rowspan="3">SSIM ↑</td><td></td><td>Ablation</td><td>21.960 (-1.37%)</td><td>22.043 (-1.34%)</td><td>22.015 (-1.75%)</td></tr><tr><td>FA</td><td>Proposed</td><td>0.803</td><td>0.814</td><td>0.826</td></tr><tr><td></td><td>Ablation</td><td>0.796 (-0.83%)</td><td>0.805 (-1.12%)</td><td>0.805 (-2.50%)</td></tr><tr><td></td><td>GFA</td><td>Proposed Ablation</td><td>0.737 0.729 (-1.13%)</td><td>0.739 0.730 (-1.13%)</td><td>0.743 0.727 (-2.04%)</td></tr></table>

## 6 Discussion

The experiments indicate that the hash-grid encodings [19] adopted in NODF-HashEnc [9] limit the retrieval of missing through-plane details in thick slices. Gradient updates primarily affect grid entries at supervised positions. However, at inference time, off-grid positions rely on trilinear interpolation of local embeddings that, lacking encoding of global spatial relationships, fail to maintain coherent microstructural features across wide interslice gaps, yielding diminished anisotropy and directional information in the reconstructed slices and degraded tractography with reduced streamline density and early tract termination. In contrast, NODF [7] with SIREN [21] activations, although slow to train, learns a globally coherent, continuous spatial basis that captures spatial correlations across the full volume. This global coherence, combined with the proposed prior, enables effective super-resolution under extreme anisotropy.

The quantitative results further expose a discrepancy between some image quality metrics and perceptual quality: NODF initially achieves nominally better PSNR and SSIM scores despite producing oversmoothed images, and the metrics counterintuitively decrease or fluctuate as training progresses, a behavior explained by the perception-distortion tradeoff [4]. Qualitative inspection confirms NODF’s reconstruction remains blurry even after 1 hour, whereas our method produces sharper outputs with more anatomical detail. PSNR and SSIM, developed for natural images, have well-documented limitations in medical imaging: they systematically favor blurred reconstructions over those that preserve fine anatomical detail [5] and correlate poorly with the assessments of expert radiologists [16]. On domain-specific metrics directly relevant to diffusion MRI analysis, including ODF reconstruction error, eigenvector angular error, and tractography fidelity, our method outperforms both baselines across all training times considered.

## 7 Conclusion

We propose a template-guided implicit neural representation (INR) framework for diffusion MRI super-resolution. Using transfer learning, we pre-train an INR on a high-resolution template and map it to new subjects via deformable registration, serving as a prior for subsequent subject-specific fine-tuning. Evaluated against two recent INR-based methods, our framework achieves significantly faster training and higher-quality reconstructions on Human Connectome Project diffusion data, recovering fine anatomical structures that neither baseline could resolve at 4× through-plane super-resolution. Future work could adopt registration models trained specifically for dMRI-derived scalar images, where improved spatial mapping fidelity is expected to benefit reconstruction quality; extend to multiple b-values and angular super-resolution; incorporate measured slice-excitation profiles into the downsampling operator; and pursue quantitative and expert validation on pathological cohorts.

## Acknowledgements

Research reported in this manuscript was supported by the National Institute of Neurological Disorders and Stroke, the National Institute of Mental Health, and the National Institute of Biomedical Imaging and Bioengineering of the National Institutes of Health under Award Numbers R01NS125307, R01MH132610, R01MH125860, R01EB032378, and R01MH119222. The content is solely the responsibility of the authors and does not necessarily represent the official views of the National Institutes of Health. Data were provided by the Human Connectome Project, WU-Minn Consortium (Principal Investigators: David Van Essen and Kamil Ugurbil; 1U54MH091657), funded by the 16 NIH Institutes and Centers that support the NIH Blueprint for Neuroscience Research; and by the McDonnell Center for Systems Neuroscience at Washington University.

## References

[1] Daniel C Alexander, Darko Zikic, Aurobrata Ghosh, Ryutaro Tanno, Viktor Wottschel, Jiaying Zhang, Enrico Kaden, Tim B Dyrby, Stamatios N Sotiropoulos, Hui Zhang, et al. Image quality transfer and applications in diffusion mri. NeuroImage, 152:283– 298, 2017.

[2] Brian B Avants, Nicholas J Tustison, Michael Stauffer, Gang Song, Baohua Wu, and James C Gee. The insight toolkit image registration framework. Frontiers in neuroinformatics, 8:44, 2014.

[3] Tashrif Billah and Sylvain Bouix. A Luigi workflow joining individual modules of an MRI processing pipeline. https://github.com/pnlbwh/luigi-pnlpipe, 2020. DOI: 10.5281/zenodo.3666802.

[4] Yochai Blau and Tomer Michaeli. The perception-distortion tradeoff. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 6228–6237, 2018.

[5] Anna Breger, Ander Biguri, Malena Sabaté Landman, Ian Selby, Nicole Amberg, Elisabeth Brunner, Janek Gröhl, Sepideh Hatamikia, Clemens Karner, Lipeng Ning, et al. A study of why we need to reassess full reference image quality assessment with medical images. Journal ofImaging Informatics in Medicine, pages 1–26, 2025.

[6] Yinbo Chen, Sifei Liu, and Xiaolong Wang. Learning continuous image representation with local implicit image function. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8628–8638, 2021.

[7] William Consagra, Lipeng Ning, and Yogesh Rathi. Neural orientation distribution fields for estimation and uncertainty quantification in diffusion mri. Medical Image Analysis, 93:103105, 2024.

[8] Maxime Descoteaux, Elaine Angelino, Shaun Fitzgibbons, and Rachid Deriche. Regularized, fast, and robust analytical q-ball imaging. Magnetic Resonance in Medicine: An Official Journal of the International Society for Magnetic Resonance in Medicine, 58(3):497–510, 2007.

[9] Mohammed Munzer Dwedari, William Consagra, Philip Müller, Özgün Turgut, Daniel Rueckert, and Yogesh Rathi. Estimating neural orientation distribution fields on high resolution diffusion mri scans. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 307–317. Springer, 2024.

[10] Axel Elaldi, Neel Dey, Heejong Kim, and Guido Gerig. Equivariant spherical deconvolution: Learning sparse orientation distribution functions from spherical data. In International Conference on Information Processing in Medical Imaging, pages 267– 278. Springer, 2021.

[11] Eleftherios Garyfallidis, Matthew Brett, Bagrat Amirbekian, Ariel Rokem, Stefan Van Der Walt, Maxime Descoteaux, Ian Nimmo-Smith, and Dipy Contributors. Dipy, a library for the analysis of diffusion mri data. Frontiers in neuroinformatics, 8:8, 2014.

[12] Matthew F Glasser, Stamatios N Sotiropoulos, J Anthony Wilson, Timothy S Coalson, Bruce Fischl, Jesper L Andersson, Junqian Xu, Saad Jbabdi, Matthew Webster, Jonathan R Polimeni, et al. The minimal preprocessing pipelines for the human connectome project. Neuroimage, 80:105–124, 2013.

[13] Tom Hendriks, Anna Vilanova, and Maxime Chamberland. Implicit neural representation of multi-shell constrained spherical deconvolution for continuous modeling of diffusion mri. Imaging Neuroscience, 3:imag\_a\_00501, 2025.

[14] Young Tak Jo, Sung Woo Joo, Woohyeok Choi, Soohyun Joe, and Jungsun Lee. White matter tract alterations in schizophrenia identified by dti-based probabilistic tractography: a multisite harmonisation study. Acta Neuropsychiatrica, 37:e47, 2025.

[15] Matthew Lyon, Paul Armitage, and Mauricio A Álvarez. Spatio-angular convolutions for super-resolution in diffusion mri. Advances in Neural Information Processing Systems, 36:12457–12475, 2023.

[16] Allister Mason, James Rioux, Sharon E Clarke, Andreu Costa, Matthias Schmidt, Valerie Keough, Thien Huynh, and Steven Beyea. Comparison of objective image quality metrics to expert radiologists’ scoring of diagnostic quality of mr images. IEEE transactions on medical imaging, 39(4):1064–1072, 2019.

[17] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications ofthe ACM, 65(1):99–106, 2021.

[18] Hans-Peter Müller and Jan Kassubek. Toward diffusion tensor imaging as a biomarker in neurodegenerative diseases: technical considerations to optimize recordings and data processing. Frontiers in Human Neuroscience, 18:1378896, 2024.

[19] Thomas Müller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM transactions on graphics (TOG), 41(4):1–15, 2022.

[21] Vincent Sitzmann, Julien Martel, Alexander Bergman, David Lindell, and Gordon Wetzstein. Implicit neural representations with periodic activation functions. Advances in neural information processing systems, 33:7462–7473, 2020.

[22] Stamatios N Sotiropoulos, Saad Jbabdi, Junqian Xu, Jesper L Andersson, Steen Moeller, Edward J Auerbach, Matthew F Glasser, Moises Hernandez, Guillermo Sapiro, Mark Jenkinson, et al. Advances in diffusion mri acquisition and processing in the human connectome project. Neuroimage, 80:125–143, 2013.

[23] Tyler Spears and P Thomas Fletcher. Learning spatially-continuous fiber orientation functions. In 2024 IEEE International Symposium on Biomedical Imaging (ISBI), pages 1–5. IEEE, 2024.

[24] Ryutaro Tanno, Daniel E Worrall, Aurobrata Ghosh, Enrico Kaden, Stamatios N Sotiropoulos, Antonio Criminisi, and Daniel C Alexander. Bayesian image quality transfer with cnns: exploring uncertainty in dmri super-resolution. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 611–619. Springer, 2017.

[25] Lin Tian, Hastings Greer, Roland Kwitt, Francois-Xavier Vialard, Raúl San José Esté- par, Sylvain Bouix, Richard Rushmore, and Marc Niethammer. unigradicon: A foundation model for medical image registration. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 749–760. Springer, 2024.

[26] J-Donald Tournier, Chun-Hung Yeh, Fernando Calamante, Kuan-Hung Cho, Alan Connelly, and Ching-Po Lin. Resolving crossing fibres using constrained spherical deconvolution: validation using diffusion-weighted imaging phantom data. Neuroimage, 42 (2):617–625, 2008.

[27] David C Van Essen, Stephen M Smith, Deanna M Barch, Timothy EJ Behrens, Essa Yacoub, Kamil Ugurbil, Wu-Minn HCP Consortium, et al. The wu-minn human con nectome project: an overview. Neuroimage, 80:62–79, 2013.

[28] Ruoyou Wu, Jian Cheng, Cheng Li, Juan Zou, Jing Yang, Wenxin Fan, Yong Liang, and Shanshan Wang. Csr-dmri: Continuous super-resolution of diffusion mri with anatomical structure-assisted implicit neural representation learning. In International Workshop on Machine Learning in Medical Imaging, pages 114–123. Springer, 2024.

[29] Xinle Zhao, Mengyue You, Wenyu Ren, Lixin Ji, Yongbo Liu, and Meng Lu. The application of diffusion tensor imaging in patients with mild cognitive impairment: a systematic review and meta-analysis. Frontiers in Neurology, 16:1467578, 2025.