# NEURAL BLIND DECONVOLUTION FOR IMAGE SCANNING MICROSCOPY

Eliot Guez<sup>1</sup>, Nathan Buskulic<sup>1</sup>, Lisa Cuneo<sup>2,3</sup>, Luca Calatroni<sup>1,3</sup>, Giuseppe Vicidomini<sup>3</sup>

<sup>1</sup> MaLGa Center, DIBRIS, University of Genoa, Italy,

<sup>2</sup> CSLM, Italian Institute of Technology, Italy, <sup>3</sup> MMS, Italian Institute of Technology, Italy

## ABSTRACT

Image scanning microscopy (ISM) acquires one scanned image per detector element, which are then fused into a single image by multi-image deconvolution to obtain a superresolved image. Its accuracy, however, depends on point spread function (PSF) models that sample- and systeminduced aberrations often invalidate. Since differences between the scanned images encode aberration information, we propose a training-free blind Deep Image Prior (DIP) framework that jointly estimates the specimen and the detectordependent PSFs. The PSFs are parametrised through the detector geometry and shared excitation and emission PSFs, reducing the unknowns while preserving the physical ISM model, and a Gaussian pre-initialisation stabilises the blind optimisation. On experimental tubulin data, blind DIP recovers finer filamentous structures than multi-image deconvolution and DIP with fixed theoretical PSFs, while adapting the PSFs to the measurements.

Index Terms— Image Scanning Microscopy, Deep Image Prior, Blind deconvolution

## 1. INTRODUCTION

Image scanning microscopy (ISM) replaces the single-element detector of confocal laser-scanning microscopy with a detector array to produce super-resolved images with a high signal-to-noise ratio (SNR) [1]. The array elements simultaneously record scanned images, each containing complementary spatial information owing to the different positions of the elements in the detection plane. Each element acts as a small pinhole, providing the resolution benefits of narrow-pinhole confocal detection, while neighboring elements collect otherwise rejected photons, preserving a high photon budget. Multi-image deconvolution (MID) combines these images using their detector-dependent PSFs [2]; extending its forward model to account for a three-dimensional specimen also enables optical sectioning [3]. These PSFs are often derived from theoretical models that may not capture experimental aberrations. Crucially, differences among the scanned images encode information about these aberrations [4], providing additional constraints for PSF estimation and motivating blind deconvolution.

We consider here a two-dimensional blind deconvolution setting where the unknown sample $\mathbf { x } \in \mathbb { R } _ { + } ^ { n }$ gives rise to $N _ { c h } ~ = ~ 2 5$ observations $\{ \mathbf { y } _ { i } \} _ { i = 1 } ^ { N _ { c h } } , \mathbf { y } _ { i } \in \mathbb { N } _ { 0 } ^ { n }$ , through related but distinct convolution kernels $( \mathrm { i . e . }$ , detector-dependent PSFs) $\{ \mathbf { h } _ { i } \} _ { i = 1 } ^ { N _ { c h } } , \mathbf { h } _ { i } \in \mathbb { R } _ { + } ^ { k }$ , where n and k denote the number of pixels in the image and kernel domains, respectively. Since the SPAD array detects individual photons without readout noise, we model the observations with Poisson statistics, i.e. $\mathbf { y } _ { i } \sim \mathrm { P o i s s o n } ( \mathbf { h } _ { i } * \mathbf { x } )$ . As unknown aberrations can cause the experimental PSFs to differ from their theoretical models, both x and $\{ \mathbf { h } _ { i } \} _ { i = 1 } ^ { N _ { c h } }$ have to be recovered from $\{ \mathbf { y } _ { i } \} _ { i = 1 } ^ { N _ { c h } }$ leading to a blind inverse problem. Its set of solutions can, however, be restricted by exploiting the (possibly estimated) detector geometry, which determines how the kernels are related to one another. Furthermore, a real dataset pairing ISM measurements with ground-truth images and PSFs is not easy to obtain for training. We therefore turn to unsupervised reconstruction and, more precisely, to SelfDeblur [5], a blind extension of the Deep Image Prior (DIP) [6]. In the non-blind setting, DIP has proved flexible and effective in practice [7, 8], and in some regimes it comes with theoretical convergence and recovery guarantees [9, 10]. Blind problems, however, raise additional challenges [11]. In deconvolution, even without noise, the factorisation $\mathbf { y } = \mathbf { x } * \mathbf { h }$ is not unique: any rescaling $( c ^ { - 1 } \mathbf { x } , c \mathbf { h } )$ with $c > 0$ or opposite translations $( T ( \mathbf { x } , \nu ) , T ( \mathbf { h } , - \nu ) )$ , where $T ( \cdot , \nu )$ shifts its input by $\nu ,$ yields the same observation. A classical example in single-image blind deconvolution is the trivial no-blur solution $( \mathbf { y } , \delta )$ , with $\delta \textbf { a }$ Dirac mass. A key issue when adapting DIP to the blind setting is therefore to select robust solutions among these equivalent ones.

In this work, we develop a DIP-based approach for blind ISM reconstruction that jointly estimates the underlying image and the set of convolution kernels. The optical ISM model expresses each kernel $\mathbf { h } _ { i }$ as the product of a shared excitation PSF and a detection PSF derived from the shared emission PSF, shifted according to the position of detector element i. We incorporate this physical relationship into the optimisation, reconstructing all kernels $\{ \mathbf { h } _ { i } \} _ { i = 1 } ^ { N _ { c h } }$ from the two shared excitation and emission PSFs. This reduces the number of unknowns and restricts the set of admissible solutions. We further introduce a pre-initialisation strategy for the kernelgenerating networks, which stabilises the blind optimisation and makes it robust to translation ambiguities. Experiments on real ISM data show promising qualitative results.

## 2. IMAGE SCANNING MICROSCOPY

To describe precisely the underlying ISM image acquisition model, we set all detector elements to share the same geometry, described by a common binary spatial detection response $p ,$ assuming uniform detection efficiency across the active area. Each element’s response is therefore a shifted $\mathrm { c o p y } , p _ { i } ( \mathbf { r } ) = p ( \mathbf { r } - \mathbf { r } _ { i } )$ , where $\mathbf { r } _ { i }$ is the shift vector describing the position of element i in the image plane. The corresponding detection and effective PSFs read

$$
\mathbf { h } _ { \mathrm { d e t } , i } = \mathbf { h } _ { \mathrm { e m } } * \mathbf { p } _ { i } , \qquad \mathbf { h } _ { i } = \mathbf { h } _ { \mathrm { e x c } } \cdot \mathbf { h } _ { \mathrm { d e t } , i } ,\tag{1}
$$

where $\mathbf { h } _ { \mathrm { e x c } }$ and $\mathbf { h } _ { \mathrm { e m } }$ are the excitation and emission PSFs. Each effective PSF $\mathbf { h } _ { i }$ is narrower than $\mathbf { h } _ { \mathrm { e m } } .$ , the PSF of a diffraction-limited widefield system, providing the basis for ISM’s super-resolution capability [1].

Importantly, once the detector geometry is fixed, all $N _ { c h }$ effective PSFs are determined by only two functions, $\mathbf { h } _ { \mathrm { e x c } }$ and $\mathbf { h } _ { \mathrm { e m } }$ . In practice, all quantities are sampled on the kernel grid. Denoting by $\mathbf { h } _ { \mathrm { e x c } } , \mathbf { h } _ { \mathrm { e m } } , \mathbf { p } _ { i } \in \mathbb { R } _ { + } ^ { k }$ the discretised PSFs and apertures, (1) becomes $\mathbf { h } _ { i } = \mathbf { h } _ { \mathrm { e x c } } { \odot } \big ( \mathbf { h } _ { \mathrm { e m } } { * } \mathbf { p } _ { i } \big )$ ), with ⊙ the entrywise product. Since $\mathbf { p } _ { i } = T ( \mathbf { p } , \mathbf { r } _ { i } )$ is a shifted copy of the discretised aperture p of a single element, the set $\{ \mathbf { p } _ { i } \} _ { i = 1 } ^ { N _ { c h } }$ is fully determined by the element shape and the positions $\mathbf { r } _ { i } .$ expressed in pixel units.

When $\{ \mathbf { h } _ { i } \} _ { i = 1 } ^ { N _ { c h } }$ are known, an estimate of the desired object x can be found by solving

$$
\operatorname* { m i n } _ { \mathbf { x } \geq 0 } \sum _ { i = 1 } ^ { N _ { c h } } \mathrm { K L } \big ( \mathbf { y } _ { i } , \mathbf { h } _ { i } * \mathbf { x } \big ) , \quad \mathrm { K L } ( \mathbf { y } , \mathbf { z } ) = \Big \langle \mathbf { y } \log \frac { \mathbf { y } } { \mathbf { z } } \mathbf { - y } \mathbf { + z } , \mathbf { 1 } \Big \rangle ,\tag{2}
$$

where the $\operatorname { K L } ( \cdot , \cdot )$ denotes the Kullback-Leibler divergence. Problem (2) is typically solved via Richardson–Lucy scheme extended to multiple images [2], with $\mathbf { h } _ { \mathrm { e x c } }$ and $\mathbf { h } _ { \mathrm { e m } }$ calculated using vectorial diffraction theory. In the blind setting, these two PSFs become unknowns, and the discretised model above provides the physical structure on which our method builds in the next section.

## 3. DEEP PARAMETRISATION FOR BLIND ISM RECONSTRUCTION

Deep Image Prior (DIP) [6] solves inverse problems without training data by relying on the implicit regularisation of a neural network rather than on an explicit regulariser. The unknown is parametrised as the output $\widehat { \mathbf { x } } = \mathbf { g } _ { \mathbf { x } } ( \mathbf { z } _ { \mathbf { x } } , \pmb { \theta } _ { \mathbf { x } } )$ of a network $\mathbf { g _ { x } } : \mathbb { R } ^ { d } \times \mathbb { R } ^ { p } \to \mathbb { R } ^ { n }$ , with fixed random input $\mathbf { z } _ { \mathbf { x } } \in \mathbb { R } ^ { d }$ and weights $\pmb { \theta _ { x } } \in \mathbb { R } ^ { p }$ . For a known kernel h and a data-fidelity term $\mathcal { L } : \mathbb { R } ^ { n } \times \mathbb { R } ^ { n } \to \mathbb { R } _ { + }$ , the weights are

computed by solving

$$
\operatorname* { m i n } _ { \pmb { \theta } _ { \mathbf { x } } } \mathcal { L } \big ( \mathbf { y } , \mathbf { h } * \mathbf { g } _ { \mathbf { x } } \big ( \mathbf { z } _ { \mathbf { x } } , \pmb { \theta } _ { \mathbf { x } } \big ) \big ) .\tag{3}
$$

Several extensions of DIP to blind problems exist [5, 12, 13]. The closest to our purposes is SelfDeblur [5], which parametrises the kernel by a second network, $\widehat { \textbf { h } } =$ $\mathbf { g _ { h } } ( \mathbf { z _ { h } } , \pmb { \theta } _ { \mathbf { h } } )$ , and jointly solves

$$
\operatorname* { m i n } _ { \pmb { \theta } _ { \mathbf { x } } , \pmb { \theta } _ { \mathbf { h } } } \ \mathcal { L } \big ( \mathbf { y } , \mathbf { g } _ { \mathbf { h } } ( \mathbf { z } _ { \mathbf { h } } , \pmb { \theta } _ { \mathbf { h } } ) * \mathbf { g } _ { \mathbf { x } } ( \mathbf { z } _ { \mathbf { x } } , \pmb { \theta } _ { \mathbf { x } } ) \big ) .\tag{4}
$$

In ISM, the $N _ { c h }$ deconvolution problems are coupled through three unknowns: the object x and the PSFs $\mathbf { h } _ { \mathrm { e x c } } ,$ $\mathbf { h } _ { \mathrm { e m } }$ . We therefore adapt SelfDeblur by parametrising each unknown with its own network, $\begin{array} { r } { \widehat { { \textbf { x } } } = \ \mathbf { g } _ { \mathbf { x } } ( \pmb \theta _ { \mathbf { x } } ) , \ \widehat { \mathbf { h } } _ { \mathrm { e x c } } = } \end{array}$ $\mathbf { g } _ { \mathrm { e x c } } ( \pmb { \theta } _ { \mathrm { e x c } } )$ and $\widehat { \mathbf { h } } _ { \mathrm { e m } } = \mathbf { g } _ { \mathrm { e m } } ( \pmb { \theta } _ { \mathrm { e m } } )$ , omitting the fixed inputs for simplicity. The object network has a non-negative output layer, while the two kernel networks share the same architecture and map onto the simplex $\Delta _ { k } = \left\{ \mathbf { h } \in \mathbb { R } _ { + } ^ { k } : \left. \mathbf { h } , \mathbf { 1 } \right. = 1 \right\}$ through a final softplus followed by $\ell _ { 1 }$ normalisation. With the spatial detection response $\mathbf { p } _ { i }$ fixed by the detector geometry (Section 2), we solve

$$
\begin{array} { r l } & { \displaystyle \operatorname* { m i n } _ { \theta _ { \mathrm { x } } , \theta _ { \mathrm { e x c } } , \theta _ { \mathrm { e m } } } \sum _ { i = 1 } ^ { N _ { c h } } \mathrm { K L } \big ( \mathbf { y } _ { i } , \widehat { \mathbf { h } } _ { i } * \mathbf { g } _ { \mathbf { x } } ( \pmb { \theta } _ { \mathrm { x } } ) + \varepsilon { \mathbf { 1 } } \big ) + R ( \widehat { \mathbf { h } } _ { \mathrm { e x c } } ) + R ( \widehat { \mathbf { h } } _ { \mathrm { e m } } ) , } \\ & { \mathrm { s . t . } \quad \widehat { \mathbf { h } } _ { i } = \widehat { \mathbf { h } } _ { \mathrm { e x c } } \odot \big ( \widehat { \mathbf { h } } _ { \mathrm { e m } } * \mathbf { p } _ { i } \big ) , } \end{array}\tag{DIP-ISM}
$$

where KL is defined in (2), $R ( { \bf h } ) = \frac { \lambda } { 2 } \| \nabla { \bf h } \| _ { 2 } ^ { 2 }$ is a Tikhonov regulariser stabilising the kernel estimation, and $\varepsilon > 0$ is a small offset. Non-negativity and normalisation hold by construction, so (DIP-ISM) is unconstrained in the weights. Note that such normalisation removes the scale ambiguity problem, while preserving the relative intensities of the channels, which carry information on the detector geometry.

The objective of (DIP-ISM) can be differentiated with respect to the weights by backpropagation, and we minimise it with Adam. We stress that this only guarantees well-defined, stable gradients: since (DIP-ISM) is non-convex, no convergence guarantee is claimed, and the analyses of [9, 10] do not directly cover the blind setting.

Note that the proposed approach couples the $N _ { c h }$ kernels through the detector geometry, thus retaining physical knowledge on the excitation and emission PSFs that would be lost by estimating each $\widehat { \mathbf { h } } _ { i }$ independently, e.g. by means of $N _ { c h }$ separate kernel networks.

## 3.1. On the choice of the initialisation

As mentioned in Section 1, blind problems are faced with sets of ambiguous solutions. Notably, SelfDeblur, and vanilla (DIP-ISM) are very sensitive to finding translated solutions, which depend on the initialization. To mitigate this, before starting the optimization of (DIP-ISM), we train for a few steps $\mathbf { g _ { h } }$ to output a centered isotropic Gaussian kernel of chosen variance $\sigma .$ The goal is simply to give the optimization process a physically plausible starting point and, we hope, stay close to. This simple procedure proves to be very effective to guarantee centered solutions in practice, for various noise levels, initialisations and choices of σ.

Contrary to other existing stabilization methods in the litterature [12, 13], this method does not necessitate any training set and works directly at the output level of the network making it very simple to use. Furthermore, pretraining the weights to meet an output also changes the Jacobian of $\mathbf { g _ { h } }$ to make a few of its singular values much more important than others, which does help at the beginning of the optimization to keep the solution close to its pretrained state.

## 4. NUMERICAL RESULTS

We evaluated the proposed method from data obtained from a custom ISM system equipped with a $5 \times 5$ SPAD detector array [14]. We chose for (DIP-ISM) to reconstruct the specimen x using $\mathbf { g _ { x } }$ as a convolutional skip network, following the architecture used in SelfDeblur $[ 5 ] ,$ while $\mathbf { g _ { h } }$ was taken as a two-layer fully connected network with ReLU6 activation, followed by Softplus and $L _ { 1 }$ normalization. We optimize jointly the three networks by doing $1 0 ^ { 3 }$ Adam steps, with image and kernel learning rates of $1 0 ^ { - 2 }$ and $1 0 ^ { - 4 }$ respectively, to minimize $\mathcal { L } .$ For comparison, we used MID [2], which reconstructs the image through a multi-image Richardson–Lucy scheme using the 25 detector-dependent PSFs h constructed from excitation and emission PSFs calculated using vectorial diffraction theory. We also evaluated a non-blind DIP approach using the same fixed theoretical PSFs, providing a baseline for assessing the benefit of joint image and PSF estimation.

## 4.1. Qualitative comparison

Figure 1 compares the conventional confocal image (obtained by summing all the scanned images), the theoretical and estimated effective PSFs, and the three image reconstructions. All three reconstruction methods resolve the main network of tubulin filaments. Compared with MID and DIP with fixed theoretical PSFs, blind DIP produces visually more pronounced fine filament structures, which may reflect its ability to account for optical aberrations affecting the acquisition.

The recovered effective PSFs differ from the theoretical ones in their apparent extent and spatial distribution (fig. 1). A main advantage of our (DIP-ISM) method is that we recover the underlying excitation and emission PSFs which allow a much finer analysis of the optical system. For example, in fig. 2 the estimated excitation PSF has a diagonally elongated central lobe, consistent with astigmatism, a typical optical aberration that could then be corrected. In these estimated PSFs, the peripheral intensity on the border is not optically plausible and probably reflects an artifact of the blind estimation. However, these peripheral artefacts are largely suppressed in h by the pointwise multiplication of the spatially localised excitation and detection PSFs.

## 5. CONCLUSION

We presented a blind DIP approach for jointly estimating the specimen image and the excitation and emission PSFs for ISM imaging. By constructing each detector-dependent effective PSF from the shared excitation and emission PSFs and the detector geometry, the method incorporates the physical relationships among the kernels. On experimental tubulin data, the resulting reconstruction reveals fine filament details and estimates PSFs that differ from the theoretical model. These qualitative results support further investigation of blind DIP for ISM, including the stability and physical interpretation of the recovered PSFs and quantitative validation on suitable reference measurements.

## 6. ACKNOWLEDGMENTS

The work of EG, NB, Lu C was supported by the funding received from the European Research Council (ERC) Starting project MALIN under the European Union’s Horizon Europe programme (grant 101117133). The work of Li C was partially supported by the Teaching Science to Computer Flagship program of the Italian Institute of Technology. This work represents only the view of the authors. The European Commission and the other organisations are not responsible for any use that may be made of the information it contains.

## 7. REFERENCES

[1] Colin J. R. Sheppard, “Super-resolution in confocal imaging,” Optik, 1988.

[2] Alessandro Zunino et al., “Reconstructing the image scanning microscopy dataset: an inverse problem,” Inverse Probl., 2023.

[3] Alessandro Zunino et al., “Structured detection for simultaneous super-resolution and optical sectioning in laser scanning microscopy,” Nat. Photon., 2025.

[4] Francesco Fersini et al., “Wavefront estimation through structured detection in laser scanning microscopy,” Biomed. Opt. Express, 2025.

[5] Dongwei Ren et al., “Neural blind deconvolution using deep priors,” in CVPR, 2020.

[6] Dmitry Ulyanov et al., “Deep image prior,” in CVPR, 2018.

[7] Daniel Otero Baguer et al., “Computed tomography reconstruction using deep image prior and learned reconstruction methods,” Inverse Probl., 2020.

[8] Mohammad Zalbagi Darestani and Reinhard Heckel, “Accelerated MRI with un-trained neural networks,” IEEE Trans. Comput. Imaging, 2021.

![](images/6d202a32c41cc6b10bf5437daacba31de7df457801fba222d6a8f179696cb5b2.jpg)  
Fig. 1. Comparison on experimental tubulin ISM data. Top row, from left to right: sum of the 25 detector observations, theoretical detector-dependent PSFs, and PSFs estimated by blind DIP. Bottom row: multi-image deconvolution (MID) with the full theoretical PSF stack, DIP with fixed theoretical PSFs, and the proposed blind DIP reconstruction. Image intensities are displayed with separate color scales.

![](images/1754797b407d4302cde6cf8a34bfbbfbdf625bfddedd3a01955dc7437fbdd283.jpg)

![](images/648d945a9be2bd17a856baf874493228be347dea09bd1e8360ad1a5befe06a7c.jpg)  
Fig. 2. Latent excitation (left) and emission (right) PSFs estimated jointly with the specimen by the proposed blind DIP approach on experimental tubulin data. Both exhibit a localized central peak, with additional diffuse and peripheral intensity.

[9] Nathan Buskulic et al., “Convergence and recovery guarantees of unsupervised neural networks for inverse problems,” J. Math. Imaging Vis., 2024.

[10] Nathan Buskulic et al., “Recovery guarantees of unsupervised neural networks for inverse problems trained with gradient descent,” in EUSIPCO, 2024.

[11] Patrizio Campisi and Karen Egiazarian, Blind image deconvolution: theory and applications, CRC Press, 2017.

[12] Gustav Bredell et al., “Wiener guided DIP for unsupervised blind image deconvolution,” in WACV, 2023.

[13] Jiangtao Zhang et al., “Blind image deconvolution by generative-based kernel prior and initializer via latent encoding,” in ECCV, 2024.

[14] Laurent Le et al., “Brighteyes-scope: A reference implementation for image scanning microscopy with spad array detection,” bioRxiv, 2026.