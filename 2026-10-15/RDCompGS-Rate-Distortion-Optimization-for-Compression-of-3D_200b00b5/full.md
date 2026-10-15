# RDCompGS: Rate–Distortion Optimization for Compression of 3D Gaussian Splatting with Image Codecs

Huiping Ren, Shuai Wan, Luqian Bai, Zhang Chen, and Fuzheng Yang

Abstract—3D Gaussian Splatting (3DGS) enables high-quality, real-time novel view synthesis but requires significant storage and transmission resources. Mapping Gaussian attributes to 2D representations enables efficient compression using mature image codecs. However, Gaussian representations that achieve comparable rendering quality are generally treated similarly during optimization, despite substantial differences in coding rate, leading to limited rate-distortion (R-D) performance after compression. To address this limitation, we propose RDCompGS, a 3DGS compression method that enables overall R-D optimization of Gaussian representations with respect to subsequent coding. A differentiable estimate of the coding rate is incorporated into the R-D optimization with rendering distortion. The rate constraint adjustment model further regulates the R-D trade-off across different compression conditions. Extensive experiments on three benchmark datasets demonstrate that RDCompGS achieves average Bjøntegaard Delta Rate reductions of 37.84% and 49.31% with H.265/HEVC and JPEG XL, respectively. The estimated rate also shows a strong correlation with the actual bit-rate under different compression conditions.

Index Terms—3D Gaussian Splatting, 3DGS compression, ratedistortion optimization, image codecs.

## I. INTRODUCTION

D Gaussian Splatting (3DGS) [1] is an important approach explicit anisotropic 3D Gaussians and rendering them through α-blending, 3DGS achieves high rendering quality and realtime performance. 3DGS is an active research topic and is relevant to areas such as autonomous driving, multimedia interaction, virtual reality, cultural heritage digitization, and digital twinning [2]–[6]. However, representing a 3D scene with 3DGS typically requires millions of Gaussian primitives, leading to significant storage and memory overhead [7], [8]. Existing studies on 3DGS compression mainly focus on two aspects.

The first aspect directly compresses the native 3D Gaussian representation. We broadly classify current approaches into Gaussian parameter and attribute compression [9]–[17], anchor-based compression [18]–[22], and Gaussian primitive compactification [3], [17], [23]–[26]. These strategies are not mutually exclusive and are often combined to reduce redundancy.

![](images/46028d640c4f3e1edfdc7a1e02a410023de2134ab5221465356020ef4e16663f.jpg)  
Fig. 1. General pipeline for converting 3DGS into 2D representations and compressing them using image codecs.

The second aspect converts 3DGS into 2D representations and compresses them using mature image codecs [27]–[32]. This direction enables 3DGS compression to directly leverage established coding standards and their existing software and hardware support, providing a practical solution for 3DGS storage and transmission. As shown in Fig. 1, the general pipeline consists of 3DGS generation and compression. The scene is first represented by Gaussian primitives and then mapped into 2D representations for coding.

However, in 3DGS compression through image codecs, Gaussian optimization is usually separated from subsequent coding. Specifically, Gaussian optimization evaluates attributes according to their contribution to rendered views, while subsequent compression is performed considering both the coding distortion and the corresponding bit-rate as the cost. As the result, Gaussian representations with comparable rendering quality are generally treated similarly during optimization even when their coding rates differ substantially. Therefore, the rate-distortion (R-D) performance cannot be optimized for compression if reducing the rendering distortion is the optimization goal. Since the actual bit-rate produced by the codec is also not differentiable, it is not feasible to be directly incorporated into Gaussian optimization.

Moreover, the balance between coding rate and rendering distortion changes across compression conditions. Therefore, applying a single rate constraint to all compression conditions cannot maintain an appropriate R-D trade-off, which limits the R-D performance after compression.

To address the above issues, we propose RDCompGS, a 3DGS compression method that enables overall R-D optimization in which Gaussian generation and subsequent coding are treated as a whole. During Gaussian optimization, rendering distortion and the coding rate associated with subsequent compression are jointly optimized to improve the R-D performance after compression. To incorporate the coding rate into Gaussian optimization, RDCompGS introduces differentiable rate estimation based on the statistical distribution of transformed coefficients and their quantization probabilities. Different from prior approaches that optimize under a prescribed rate or distortion setting, the rate constraint adjustment model characterizes the relative strength of the estimated rate and rendering distortion and adjusts their balance across different compression conditions. Extensive experiments on three benchmark datasets demonstrate that RDCompGS achieves average BD-Rate reductions of 37.84% and 49.31% with H.265/HEVC and JPEG XL, respectively, while maintaining comparable rendering quality.

The main contributions of this work are summarized as follows.

• We consider Gaussian generation and subsequent coding as a whole for 3DGS compression and formulate the compression of Gaussian representations as an overall R-D optimization with respect to subsequent coding.

• Based on this formulation, we propose RDCompGS, which derives a differentiable estimate of the coding rate from transformed Gaussian attributes and introduces the rate constraint adjustment model to regulate the rate constraint under different coding conditions.

• We validate RDCompGS on three benchmark datasets using both H.265/HEVC and JPEG XL, showing consistent R-D improvements across different coding conditions. Additional experiments combining Gaussian pruning with image codec compression further confirm the effectiveness of RDCompGS.

The rest of this paper is organized as follows. Section II presents the related work. Section III introduces the proposed method. Section IV reports the experimental results and analysis. Finally, Section V concludes this paper.

## II. RELATED WORK

Based on the Gaussian representation used during compression, we divide existing 3DGS compression methods into two types: compression of 3D Gaussian representations and compression of 3DGS with 2D representations.

## A. Compression of 3D Gaussian Representations

We broadly classify methods for compressing 3D Gaussian representations into Gaussian parameter and attribute compression, anchor-based compression, and Gaussian primitive compactification.

Gaussian parameter and attribute compression reduces storage by making the representation of individual Gaussians more compact while retaining the native 3D Gaussian structure. Representative approaches include quantization, which reduces storage by lowering precision [9]–[13], entropy coding based on symbol probabilities to reduce the coded length [14], [15], and spherical harmonics (SH) reduction [16], [17].

Anchor-based methods reduce 3DGS redundancy by organizing local Gaussians around sparse anchors while exploiting relationships among anchors and their associated attributes. Representative methods include Scaffold-GS [18], HAC [19], HAC++ [20], ContextGS [21], and CAT-3DGS [22], which exploit attribute prediction, contextual dependencies, hierarchical relationships, and spatial correlations for compression.

Gaussian primitive compactification reduces redundancy by controlling the number or distribution of Gaussians. Representative methods include PUP 3D-GS [23], LightGaussian [17], APNet [3], visibility-based geometry pruning [24], REFINE [25], and SPARE-GS [26]. These methods exploit spatial sensitivity scores, global significance scores, viewdependent masks and Rasterization Weight Sampling, multiview visibility, rendering-aware importance, and global budget allocation to reduce redundant Gaussians or regulate their spatial distribution.

Although these methods effectively reduce redundancy in native 3D Gaussian representations, they mainly compress Gaussian primitives in their original representation form. Another direction incorporates 2D representations into 3DGS compression, enabling the use of established image codecs for compression.

## B. Compression of 3DGS with 2D Representations

We broadly divide existing 3DGS compression methods based on 2D representations according to when the 2D representation is introduced: during or after the Gaussian optimization process.

Methods that introduce 2D representations after Gaussian optimization adopt different mapping strategies. LGSCV [27] first applies a 3D Morton scan to order Gaussian primitives and then uses a 2D Morton scan to map them into blockwise 2D maps, which are compressed using H.265/HEVC. To improve performance at medium and low bit-rates, it further applies Principal Component Analysis to SH coefficients and introduces MiniPLAS for local primitive permutation. Zhu et al. [28] propose a Dual-Hilbert scan algorithm that uses a 3D Hilbert curve to preserve spatial locality and a 2D Hilbert curve to map the ordered Gaussian primitives into 2D grids for subsequent H.265/HEVC compression. 3DGCoding [29] sorts Gaussian primitives in z-order according to their positions, maps the quantized attributes onto 2D planes, and compresses the attribute planes of static Gaussians using lossless PNG and those of dynamic Gaussians using H.265/HEVC. V<sup>3</sup> [32] sorts Gaussian primitives using Morton order, maps Gaussian attributes into video frames, and compresses them using H.264/AVC.

For methods that introduce 2D representations during Gaussian optimization, CodecGS [30] redesigns the explicit Gaussian representation using progressive tri-plane features and employs a multilayer perceptron to predict Gaussian attributes. The feature plane channels are then concatenated into frames and compressed using H.265/HEVC. Compared with CodecGS, SOG [31] retains explicit Gaussian attributes and directly organizes them into aligned 2D grids using Parallel Linear Assignment Sorting (PLAS).

![](images/6c90a70db470ad7cff409d22be6bfb83248baa6c630a7c471fded168d39793d3.jpg)  
Fig. 2. Overview of the proposed RDCompGS. To enable overall R-D optimization of Gaussian representations with respect to subsequent coding, RDCompGS first introduces differentiable rate estimation based on the distribution of transformed coefficients and their quantization probabilities. Then, the rate constraint adjustment model regulates the rate constraint across different compression conditions. Finally, the estimated rate and rendering distortion jointly guide Gaussian optimization for subsequent coding.

Specifically, PLAS improves the local smoothness of the grids by constructing a smoothed target with a Gaussian filter and performing parallel blockwise assignments to update the arrangement of Gaussian primitives. The optimized Gaussian parameters are then quantized and compressed using JPEG XL.

To summarize, organizing Gaussian parameters into 2D representations provides a structured form that can be compressed using image codecs. However, Gaussian optimization remains separated from subsequent coding, which limits the overall R-D performance.

## III. METHODOLOGY

To address the mismatch between Gaussian attribute optimization and subsequent coding and the limitation of using a single rate constraint across different coding conditions, we propose RDCompGS, as shown in Fig. 2. RDCompGS introduces two main components. (a) Differentiable rate estimation constructs a differentiable rate term for Gaussian optimization (Section III-A). (b) The rate constraint adjustment model regulates the influence of the estimated rate under different coding conditions (Section III-B). The weighted rate term is then combined with the rendering distortion for Gaussian optimization (Section III-C). The optimized Gaussian attributes are subsequently compressed using image codecs.

## A. Differentiable Rate Estimation Model

Natural images generally exhibit strong spatial correlation, which is exploited by image codecs to reduce spatial redundancy. PLAS organizes Gaussian attributes into 2D grids and places Gaussians with similar attribute values at nearby positions. As shown in Fig. 3, PLAS improves the spatial correlation of the Gaussian attribute grids, but a difference from natural images still remains. To further reduce redundancy for subsequent compression while maintaining rendering quality, RDCompGS performs overall R-D optimization of Gaussian representations with respect to subsequent coding.

To incorporate the coding rate into the overall R-D optimization, we introduce differentiable rate estimation for the non-position Gaussian attributes. The non-position attributes are first normalized and transformed using an 8×8 block-based Discrete Cosine Transform (DCT), while the position attributes are excluded to preserve scene geometry. The statistical analysis in [33] motivates modeling the transformed coefficients using a zero-mean Laplace distribution. Fig. 4 further shows that the transformed coefficients exhibit distributions that are consistent with the Laplace model.

Let $x _ { i }$ denote the i-th DCT coefficient and $b _ { i } > 0$ its scale parameter. The Laplace probability density function is

$$
p ( x _ { i } \mid b _ { i } ) = { \frac { 1 } { 2 b _ { i } } } \exp \left( - { \frac { | x _ { i } | } { b _ { i } } } \right) .\tag{1}
$$

For a uniform scalar quantizer with step size $\Delta ,$ the quantized coefficient is

$$
q _ { i } = \mathrm { r o u n d } \left( \frac { x _ { i } } { \Delta } \right) .\tag{2}
$$

The probability of obtaining $q _ { i }$ is

$$
P _ { i } = \int _ { ( q _ { i } - \frac { 1 } { 2 } ) \Delta } ^ { ( q _ { i } + \frac { 1 } { 2 } ) \Delta } p ( t | b _ { i } ) \mathrm { d } t .\tag{3}
$$

Under the high-resolution scalar quantization approximation, the probability density is approximately constant within each quantization interval. Therefore, Eq. (3) can be approximated as

$$
P _ { i } \approx \Delta p ( x _ { i } \mid b _ { i } ) .\tag{4}
$$

![](images/b2cfc74080c6ea3e9ab3125d52509f9285fc941eb612571fd0b3b7d188ec19c5.jpg)  
Fig. 3. Spatial correlation comparison between a natural image and Gaussian attribute grids on the Truck scene. (a) Natural image; (b) and (c) scale attribute grids without and with RDCompGS, respectively; (d), (e), and (f) spatial correlation maps of (a), (b), and (c), respectively. Spatial correlation is calculated within a $2 1 \times 2 1$ window by averaging the horizontal and vertical Pearson correlations, with gray to blue indicating increasing correlation. The scale attribute shows lower spatial correlation than the natural image, while higher correlation is observed with RDCompGS.

Based on self-information, we define $r _ { i } = - \log _ { 2 } P _ { i }$ . Using Eqs. (4) and (1), we obtain

$$
r _ { i } \approx \log _ { 2 } \left( \frac { 2 b _ { i } } { \Delta } \right) + \frac { | x _ { i } | } { b _ { i } \ln 2 } .\tag{5}
$$

Since the scale parameter $b _ { i }$ is unknown during Gaussian optimization, we eliminate it by minimizing Eq. (5) with respect to $b _ { i }$ . For $x _ { i } \neq 0$ , the minimizing solution is

$$
b _ { i } ^ { \star } = | x _ { i } | .\tag{6}
$$

Substituting Eq. (6) into Eq. (5) yields

$$
r _ { i } \approx \log _ { 2 } \left( \frac { 2 e | x _ { i } | } { \Delta } \right) .\tag{7}
$$

To make the rate term differentiable and nonnegative during Gaussian optimization, we define

$$
r _ { i } = \log _ { 2 } \left( 1 + \frac { 2 e \sqrt { x _ { i } ^ { 2 } + \epsilon } } { \Delta } \right) , \qquad \epsilon = 1 0 ^ { - 6 } .\tag{8}
$$

The estimated rate is obtained by accumulating $r _ { i }$ over the DCT coefficients of the non-position Gaussian attributes:

$$
\mathcal { L } _ { \mathrm { r a t e } } = k \sum _ { i = 1 } ^ { N } r _ { i } = k \sum _ { i = 1 } ^ { N } \log _ { 2 } \left( 1 + \frac { 2 e \sqrt { x _ { i } ^ { 2 } + \epsilon } } { \Delta } \right) ,\tag{9}
$$

where N denotes the total number of DCT coefficients included in the rate term, and k is a conversion coefficient that scales the accumulated rate to kbps.

## B. The Rate Constraint Adjustment Model

During Gaussian optimization, the rendering distortion and the weighted estimated rate are jointly used to update the Gaussian parameters. Since the gradient magnitude of the estimated rate varies with the quantization step, a single rate constraint cannot maintain an appropriate balance between the estimated rate and rendering distortion across different compression conditions. As illustrated in Fig. 5, using a single rate constraint provides only limited improvements in the high-bit-rate region. RDCompGS therefore introduces the rate constraint adjustment model to adjust the rate constraint according to the quantization step.

![](images/0baae8aaa40276db6dc2c500ed0a2a8d4980f06beadc20492e4e7754bc1b6670.jpg)

![](images/780beda21117587fe7881c70bbc8a03493fbca6584fd7078b287fd4e78948161.jpg)  
Fig. 4. Laplace distribution fitting of DCT coefficients for opacity and SH DC component during training on the Truck scene.

We first analyze how the quantization step affects the rate gradient that is backpropagated to the Gaussian parameters. According to Eq. (9), the gradient of the estimated rate with respect to the i-th DCT coefficient is

$$
\frac { \partial \mathcal { L } _ { \mathrm { r a t e } } } { \partial x _ { i } } = \frac { 2 e k x _ { i } } { \ln 2 \sqrt { x _ { i } ^ { 2 } + \epsilon } \left( \Delta + 2 e \sqrt { x _ { i } ^ { 2 } + \epsilon } \right) } .\tag{10}
$$

For a fixed DCT coefficient $x _ { i } ,$ , Eq. (10) shows that increasing the quantization step reduces the gradient magnitude. When

$$
2 e \sqrt { x _ { i } ^ { 2 } + \epsilon } \ll \Delta ,\tag{11}
$$

the dependence on $\Delta$ can be approximated as

$$
\left| { \frac { \partial { \mathcal { L } } _ { \mathrm { r a t e } } } { \partial x _ { i } } } \right| \propto \Delta ^ { - 1 } .\tag{12}
$$

Let θ denote the Gaussian parameters and let ${ \bf x } = F ( \pmb \theta )$ denote the DCT coefficients obtained from the non-position Gaussian parameters. By the chain rule,

$$
\nabla _ { \theta } \mathcal { L } _ { \mathrm { r a t e } } = \mathbf { J } _ { F } ^ { T } \nabla _ { \mathbf { x } } \mathcal { L } _ { \mathrm { r a t e } } ,\tag{13}
$$

where $\mathbf { J } _ { F }$ denotes the Jacobian of $F ( \pmb \theta )$ with respect to $\pmb { \theta } .$ Since the mapping from the Gaussian parameters to the DCT coefficients does not explicitly depend on $\Delta ,$ the dependence of $\nabla _ { \theta } \mathcal { L } _ { \mathrm { r a t e } }$ on the quantization step is introduced through $\nabla _ { \mathbf { x } } \mathcal { L } _ { \mathrm { r a t e } }$ . According to Eq. (12), the rate gradient with respect to the Gaussian parameters therefore approximately satisfies

$$
\nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { r a t e } } ( \boldsymbol { \theta } , \Delta ) \approx \frac { \Delta _ { \mathrm { r e f } } } { \Delta } \nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { r a t e } } ( \boldsymbol { \theta } , \Delta _ { \mathrm { r e f } } ) ,\tag{14}
$$

where $\Delta _ { \mathrm { r e f } }$ denotes a reference quantization step.

To characterize the magnitude of the rate gradient, we define

$$
G _ { R } ( \pmb { \theta } , \Delta ) = \| \nabla _ { \pmb { \theta } } \mathcal { L } _ { \mathrm { r a t e } } ( \pmb { \theta } , \Delta ) \| _ { 2 } .\tag{15}
$$

The corresponding magnitude at $\Delta _ { \mathrm { r e f } }$ is denoted by

$$
\begin{array} { r } { G _ { R , \mathrm { r e f } } = G _ { R } ( \pmb \theta , \Delta _ { \mathrm { r e f } } ) . } \end{array}\tag{16}
$$

Taking the $L _ { 2 }$ norm on both sides of Eq. (14) gives

$$
\frac { G _ { R } ( \theta , \Delta ) } { G _ { R , \mathrm { r e f } } } \approx \frac { \Delta _ { \mathrm { r e f } } } { \Delta } .\tag{17}
$$

The same Gaussian parameters are also updated by the rendering distortion. We therefore define

$$
G _ { D } ( \pmb \theta ) = \| \nabla _ { \pmb \theta } \mathcal { L } _ { D } ( \pmb \theta ) \| _ { 2 } ,\tag{18}
$$

![](images/15ced9d225a8bdfc0bf4f7f58119aae3b335c55aaf00fe0e305224a94e92e594.jpg)

![](images/64a956991d2c458b4ef0fd75b21b343fe0af58dc5e08bebe87e4fe027fb775bd.jpg)

![](images/f7f2adf0e26405f1efed4d2cdab454d89454bf687d0b0f38451499ff5a431f4f.jpg)  
Fig. 5. R-D performance comparison on the Truck scene under H.265/HEVC compression between Gaussian optimization without the rate constraint and with the rate constraint using $\dot { \lambda _ { \mathrm { { r a t e } } } } = 1 \times 1 0 ^ { - 7 }$ at QP values of 4, 10, 16, and 22.

where $G _ { D }$ denotes the magnitude of the rendering distortion gradient with respect to the Gaussian parameters.

To characterize the influence of the estimated rate relative to the rendering distortion, we define

$$
\rho ( \Delta ) = \frac { \lambda _ { \mathrm { { r a t e } } } ( \Delta ) G _ { R } ( \pmb \theta , \Delta ) } { G _ { D } ( \pmb \theta ) } ,\tag{19}
$$

where $\lambda _ { \mathrm { { r a t e } } } ( \Delta )$ denotes the rate constraint at $\Delta ,$ , and $\rho ( \Delta )$ represents the relative rate constraint strength with respect to the rendering distortion. The reference value of the relative rate constraint strength is defined as

$$
\rho _ { \mathrm { r e f } } = \frac { \lambda _ { \mathrm { r e f } } G _ { R , \mathrm { r e f } } } { G _ { D } ( { \pmb \theta } ) } ,\tag{20}
$$

where $\lambda _ { \mathrm { { r e f } } }$ denotes the rate constraint at $\Delta _ { \mathrm { r e f } }$

Combining Eqs. (19), (20), and (17) gives

$$
\frac { \rho ( \Delta ) } { \rho _ { \mathrm { r e f } } } \approx \frac { \lambda _ { \mathrm { r a t e } } ( \Delta ) } { \lambda _ { \mathrm { r e f } } } \frac { \Delta _ { \mathrm { r e f } } } { \Delta } .\tag{21}
$$

If the rate constraint is adjusted only to compensate for the reduction in the rate gradient magnitude as $\Delta$ increases, $\rho ( \Delta )$ should remain approximately unchanged. However, this does not reflect the different compression requirements across quantization steps. A smaller $\Delta$ places more emphasis on rendering quality, whereas a larger $\Delta$ requires greater emphasis on reducing redundancy in Gaussian attributes. Accordingly, we let $\rho ( \Delta )$ increase with $\Delta$ as

$$
\frac { \rho ( \Delta ) } { \rho _ { \mathrm { r e f } } } = \left( \frac { \Delta } { \Delta _ { \mathrm { r e f } } } \right) ^ { \beta } , \qquad \beta > 0 ,\tag{22}
$$

where $\beta$ controls the sensitivity of $\rho ( \Delta )$ to the quantization step.

Combining Eq. (22) with Eq. (21), the rate constraint is defined as

$$
\lambda _ { \mathrm { r a t e } } ( \Delta ) = \lambda _ { \mathrm { r e f } } \left( \frac { \Delta } { \Delta _ { \mathrm { r e f } } } \right) ^ { 1 + \beta } .\tag{23}
$$

## C. Total Training Loss

Following 3DGS [1], the rendering distortion is defined as

$$
\mathcal { L } _ { D } = ( 1 - \lambda _ { \mathrm { d s s i m } } ) \mathcal { L } _ { L 1 } + \lambda _ { \mathrm { d s s i m } } \mathcal { L } _ { \mathrm { D - S S I M } } ,\tag{24}
$$

where $\lambda _ { \mathrm { d s s i m } } = 0 . 2 .$

Algorithm 1 R-D Optimization in RDCompGS   
Input: Multi-view images, camera parameters, quantization   
step $\Delta .$   
Output: Optimized Gaussian parameters $\pmb { \theta } .$   
1: Initialize the Gaussian parameters θ using multi-view   
images and camera parameters.   
2: Apply PLAS to organize the Gaussian attributes into 2D   
grids.   
3: for each training iteration do   
4: Obtain the rendered image through differentiable Gaus  
sian rasterization.   
5: Compute the rendering distortion $\mathcal { L } _ { D }$ using Eq. (24).   
6: if the densification stage has ended then   
7: Part 1: Differentiable Rate Estimation (Section III-A)   
8: Exclude the position attributes and normalize the   
remaining 2D representations.   
9: Apply 8 × 8 block-based DCT to obtain the coeffi  
cients $\{ x _ { i } \} _ { i = 1 } ^ { N }$   
10: Compute the estimated rate $\mathcal { L } _ { \mathrm { r a t e } }$ using Eq. (9).   
11: Part 2: The Rate Constraint Adjustment Model (Sec  
tion III-B)   
12: Compute the rate constraint $\lambda _ { \mathrm { { r a t e } } } ( \Delta )$ using Eq. (23).   
13: Part 3: Total Training Loss (Section III-C)   
14: Compute the total training loss $\mathcal { L } _ { \mathrm { t o t a l } }$ using Eq. (25).   
15: else   
16: Set $\mathcal { L } _ { \mathrm { t o t a l } }  \mathcal { L } _ { D } .$   
17: end if   
18: Backpropagate $\mathcal { L } _ { \mathrm { t o t a l } }$ and update θ.   
19: if a densification step is performed then   
20: Reapply PLAS to restore the local structure of the   
2D representations.   
21: end if   
22: end for   
23: return Optimized Gaussian parameters θ.

The distortion considered in RDCompGS refers to the rendering distortion defined above, while the distortion introduced by compression is not explicitly modeled in the optimization process.

Before densification ends, the Gaussian representation is optimized using only $\mathcal { L } _ { D }$ . After densification, the estimated rate in Eq. (9) is incorporated into Gaussian optimization using $\lambda _ { \mathrm { { r a t e } } } ( \Delta )$ in Eq. (23). The total training loss is

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { D } + \lambda _ { \mathrm { r a t e } } ( \Delta ) \mathcal { L } _ { \mathrm { r a t e } } .\tag{25}
$$

By jointly incorporating the estimated rate and rendering distortion, RDCompGS connects Gaussian optimization with subsequent coding to achieve overall R-D optimization for 3DGS compression. The R-D optimization procedure in RD-CompGS is summarized in Algorithm 1.

## IV. EXPERIMENTS

## A. Experimental Setup

1) Datasets and Evaluation Metrics: We conduct experiments on three datasets: Tanks and Temples [34], Deep Blending [35], and Mip-NeRF 360 [36]. These datasets contain diverse indoor and outdoor scenes with complex geometry and texture details. We evaluate rendering quality using Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index Measure (SSIM) [37], and Learned Perceptual Image Patch Similarity (LPIPS) [38]. The encoded data sizes generated by H.265/HEVC and JPEG XL are used for compression evaluation. The coding performance is evaluated in terms of the Bjøntegaard delta rate (BD-Rate) [39].

2) Implementation Details: The proposed method is implemented in PyTorch. Experiments are conducted on an NVIDIA GeForce RTX 5070 Ti GPU. During training, PLAS is applied after Gaussian initialization and after each densification step to organize the Gaussian parameters into 2D representations. After the densification stage, differentiable rate estimation is introduced into Gaussian optimization.

Following the settings in ${ \mathrm { V } } ^ { 3 }$ [32] and SOG [31], Gaussian attributes are first quantized and then compressed using image codecs. The position attributes are quantized to 16-bit values and split into high and low 8-bit components, which are represented as two three-channel RGB images. The remaining attributes are quantized to 8-bit values. Specifically, the opacity attribute and the w component of the rotation quaternion are represented as single-channel grayscale images, whereas the scale attribute, the xyz components of the rotation quaternion, and the SH coefficients are represented as three-channel RGB images. These attribute images are then compressed using H.265/HEVC [40] (all intra coding) or JPEG XL [41], where the position attributes are encoded losslessly and the remaining attributes are encoded using lossy compression.

For H.265/HEVC compression, the quantization parameter (QP) values are set to 4, 10, 16, and 22. The corresponding quantization step size $\Delta$ is calculated following [42]. Higher QP values are not considered because they cause severe degradation in rendering quality. To determine the JPEG XL distance value, we select 260 3DGS attribute images from the three datasets. The position attribute images are excluded. For each H.265/HEVC QP, we use binary search to determine the JPEG XL distance value that produces the same bits per pixel. The resulting distance values are 0.1005, 0.2157, 0.4511, and 0.9363 for QP values of 4, 10, 16, and 22, respectively. For SOG, the SH attributes are compressed using JPEG XL with quality levels of 95, 90, 70, and 50, while the default compression settings are used for all other attributes. The coefficient $k = 1 / 4 0$ is derived from the encoder frame rate of 25 frames per second.

Based on the parameter analysis conducted on the Truck scene from Tanks and Temples, we set $\Delta _ { \mathrm { r e f } } = 8 , \lambda _ { \mathrm { r e f } } = 1 \times$ $1 0 ^ { - 7 }$ , and $\beta = 0 . 2$ . Since the Truck scene is used to determine the reference parameters, it is excluded from all subsequent R-D evaluations to avoid potential bias. The exceptions are the validation of the rate estimation model in Section IV-C and the analysis of the rate constraint adjustment model in Section IV-D, where the Truck scene is retained.

## B. Compression Performance

We evaluate RDCompGS by compressing the optimized Gaussian representations using JPEG XL and H.265/HEVC, and compare the resulting R-D performance with SOG. Tables I and II report the results with JPEG XL and H.265/HEVC, respectively. The BD-Rate results show consistent R-D improvements across the three datasets. With JPEG XL, RDCompGS obtains BD-Rate reductions of 30.44%, 60.94%, and 56.55% on Tanks & Temples, Deep Blending, and Mip-NeRF 360, respectively, corresponding to an average reduction of 49.31%. Similar improvements are observed with H.265/HEVC, where the corresponding BD-Rate reductions are 15.44%, 50.48%, and 47.61%, with an average reduction of 37.84%.

Fig. 6 further compares the R-D performance of SOG and RDCompGS compressed using JPEG XL in terms of PSNR, SSIM, and LPIPS. For PSNR and SSIM, the RDCompGS curves are generally located above and to the left of those of SOG, indicating improved R-D performance across the three datasets, although SOG achieves slightly lower LPIPS in the high-bit-rate region on Tanks & Temples and Mip-NeRF 360.

Fig. 7 presents visual comparisons on the Train, Playroom, and Bonsai scenes. RDCompGS achieves higher PSNR than SOG with smaller compressed sizes across the three scenes. The enlarged regions further show that RDCompGS better preserves the sky appearance in Train, the paper box details in Playroom, and the floor textures in Bonsai.

## C. Validation of Rate Estimation Model

Having established the R-D improvements, we further evaluate the correlation between the estimated rate and the actual bit-rate. We use 260 non-position Gaussian attribute images generated after PLAS sorting. For each image, the estimated rate is compared with the actual bit-rate obtained by encoding the same image with H.265/HEVC.

As shown in Fig. 8, the estimated rate exhibits a strong linear correlation with the actual bit-rate across all tested QP values. At QP values of 4, 10, 16, and 22, the Pearson correlation coefficients are 0.997, 0.996, 0.990, and 0.973, respectively.

## D. Analysis of the Rate Constraint Adjustment Model

The parameter settings of the rate constraint adjustment model are analyzed in this section. The reference quantization step is set to $\Delta _ { \mathrm { r e f } } \ = \ 8 .$ According to Eq. (23), using a relatively large $\Delta _ { \mathrm { r e f } }$ keeps the rate constraint small at small quantization steps, avoiding an excessive rate constraint in the high-bit-rate region. The effects of $\lambda _ { \mathrm { r e f } }$ and $\beta$ are evaluated on the Truck scene, and the selected values are used in the subsequent experiments.

R-D PERFORMANCE OF RDCOMPGS USING JPEG XL COMPARED WITH SOG. SIZE IS MEASURED IN MB AND PSNR IN DB.  
TABLE I
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td colspan="2">Distance = 0.1005</td><td colspan="2">Distance = 0.2157</td><td colspan="2">Distance = 0.4511</td><td colspan="2">Distance = 0.9363</td><td rowspan="2">BD-Rate (%) ↓</td></tr><tr><td>Size PSNR</td><td>SSIM LPIPS</td><td>Size PSNR</td><td>SSIM LPIPS Size</td><td>PSNR SSIM LPIPS</td><td>Size</td><td>PSNR SSIM LPIPS</td><td></td></tr><tr><td>Tanks &amp; Temples</td><td>SOG</td><td>14.828 21.54 0.794</td><td>0.231 11.343</td><td>21.36 0.791</td><td>0.238 7.779</td><td>20.78</td><td>0.778 0.250</td><td>6.896 20.25</td><td>0.766 0.257</td><td>-30.44</td></tr><tr><td rowspan="2">(Train)</td><td>RDCompGS</td><td>16.194 21.73 0.796</td><td>0.235 10.488</td><td>21.50 0.791</td><td>0.241</td><td>6.321 21.06</td><td>0.779 0.252</td><td>4.079 20.70</td><td>0.758 0.274</td><td rowspan="2"></td></tr><tr><td>SOG 12.175</td><td>29.22 0.891</td><td>0.276 8.969</td><td>28.93 0.889</td><td>0.280 6.338</td><td>28.15 0.882</td><td>0.288 5.845</td><td>27.68 0.879</td><td>0.292</td></tr><tr><td rowspan="2">Deep Blending</td><td>RDCompGS</td><td>11.623 29.52</td><td>0.898 0.271</td><td>7.185 29.32</td><td>0.895 0.273</td><td>4.562 29.20</td><td>0.889 0.282</td><td>3.136 28.63</td><td>0.879 0.299</td><td>-60.94</td></tr><tr><td>SOG</td><td>37.140 26.82</td><td>0.797 0.235</td><td>28.136 26.53</td><td>0.793 0.244</td><td>19.073 25.62</td><td>0.776 0.261</td><td>16.933 24.88</td><td>0.762 0.271</td><td>-56.55</td></tr><tr><td rowspan="2">Mip-NeRF 360</td><td>RDCompGS</td><td>27.284 26.91 0.799</td><td>0.243 16.545</td><td>26.70 0.792</td><td>0.250 10.608</td><td>26.30</td><td>0.782 0.261</td><td>7.459 25.45</td><td>0.755 0.285</td><td rowspan="2"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

TABLE II  
R-D PERFORMANCE OF RDCOMPGS USING H.265/HEVC COMPARED WITH SOG. SIZE IS MEASURED IN MB AND PSNR IN DB.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td colspan="4">QP = 4</td><td colspan="4">QP = 10</td><td colspan="3">QP = 16</td><td colspan="3">QP = 22</td><td rowspan="2">BD-Rate (%) ↓</td></tr><tr><td>Size</td><td>PSNR</td><td>SSIM</td><td>LPIPS</td><td>Size</td><td>PSNR</td><td>SSIM LPIPS</td><td></td><td>Size</td><td>PSNR SSIM</td><td>LPIPS</td><td>Size PSNR</td><td>SSIM LPIPS</td></tr><tr><td>Tanks &amp; Temples</td><td>SOG</td><td>14.828</td><td>21.54</td><td>0.794</td><td>0.231</td><td>11.343</td><td>21.36</td><td>0.791</td><td>0.238</td><td>7.779</td><td>20.78 0.778</td><td>0.250</td><td>6.896</td><td>20.25 0.766</td><td>0.257</td><td>-15.44</td></tr><tr><td>(Train)</td><td>RDCompGS</td><td>16.451</td><td>21.57</td><td>0.795</td><td>0.237</td><td>11.180</td><td>21.24</td><td>0.785</td><td>0.244</td><td>6.886</td><td>21.04 0.776</td><td>0.255</td><td>4.645</td><td>20.51 0.753</td><td>0.279</td><td></td></tr><tr><td>Deep Blending</td><td>SOG</td><td>12.175</td><td>29.22</td><td>0.891</td><td>0.276</td><td>8.969</td><td>28.93</td><td>0.889</td><td>0.280</td><td>6.338</td><td>28.15 0.882</td><td>0.288</td><td>5.845</td><td>27.68</td><td>0.879 0.292</td><td>-50.48</td></tr><tr><td></td><td>RDCompGS</td><td>11.554</td><td>29.46</td><td>0.897</td><td>0.271</td><td>7.603</td><td>29.25</td><td>0.895</td><td>0.275</td><td>4.846</td><td>29.04 0.888</td><td>0.283</td><td>3.417</td><td>28.36</td><td>0.876 0.307</td><td></td></tr><tr><td>Mip-NeRF 360</td><td>SOG</td><td>37.140</td><td>26.82</td><td>0.797</td><td>0.235</td><td>28.136</td><td>26.53</td><td>0.793</td><td>0.244</td><td>19.073</td><td>25.62 0.776</td><td>0.261</td><td>16.933</td><td>24.88</td><td>0.762 0.271</td><td></td></tr><tr><td></td><td>RDCompGS</td><td>27.351</td><td>26.90</td><td>0.798</td><td>0.244</td><td>17.282</td><td>26.61</td><td>0.791</td><td>0.251</td><td>11.322</td><td>26.02 0.777</td><td>0.264</td><td>8.204</td><td>25.12</td><td>0.748 0.291</td><td>-47.61</td></tr></table>

![](images/5ac92ddb0a68ed7893dcf67bebef637f7497c6722d8f7dc4f2ac506125c91580.jpg)  
Fig. 6. R-D performance comparison between SOG and RDCompGS compressed using JPEG XL on three benchmark datasets in terms of PSNR, SSIM, and LPIPS.

1) Selection of $\lambda _ { \mathrm { { r e f } } } .$ Fig. 9 presents the R-D curves obtained with different $\lambda _ { \mathrm { { r e f } } }$ values. As $\lambda _ { \mathrm { r e f } }$ increases, the compressed size generally decreases, showing that a larger rate coefficient places a stronger constraint on the estimated rate during Gaussian optimization. For relatively small $\lambda _ { \mathrm { { r e f } } }$ values, the reduction in compressed size is limited, while the rendering quality remains relatively stable. Increasing $\lambda _ { \mathrm { r e f } }$ to $1 \times 1 0 ^ { - 7 }$ provides a more evident size reduction without noticeable degradation in the three quality metrics. Further increasing $\lambda _ { \mathrm { r e f } }$ results in only limited additional size reduction, while PSNR and SSIM decrease and LPIPS increases more noticeably. Considering these trends, $\lambda _ { \mathrm { { r e f } } } = 1 \times 1 0 ^ { - 7 }$ provides a favorable trade-off between compression efficiency and rendering quality and is therefore adopted in the subsequent experiments.

![](images/98d6ca446d46e3a46cf7d39e31cee428325a004f3f3bc44cb55308d5bcb7095e.jpg)  
Fig. 7. Visual comparison on the Train, Playroom, and Bonsai scenes, where RDCompGS is compressed using JPEG XL. Each row shows GT, 3DGS, SOG, and RDCompGS from left to right, with the same local region enlarged for comparison.

![](images/01f14cbb3d24f60642742f351ae823b8e9c70f4e774c10454955ed1feb0d35f2.jpg)  
Fig. 8. Estimated rate and actual bit-rate produced by H.265/HEVC at different $\mathrm { Q P }$ values. Each point corresponds to one Gaussian attribute image.

![](images/ffb30ac95aece30fe74260d15734751c1beccb5a605649bfc70ff8896bae8672.jpg)  
Fig. 9. R-D curves on the Truck scene for different $\lambda _ { \mathrm { r e f } }$ values. The selected value $\lambda _ { \mathrm { r e f } } = 1 \times 1 0 ^ { - 7 }$ achieves a favorable R-D trade-off according to the observed trend of the curves.

2) Selection of $\beta \colon$ Fig. 10 evaluates the sensitivity of the rate constraint adjustment model to different $\beta$ values, using Gaussian optimization without the rate constraint as the reference. Clear BD-Rate reductions are observed for all tested settings, while $\beta = 0 . 2$ achieves the lowest BD-Rate of −62.31%. Therefore, $\beta = 0 . 2$ is adopted in the subsequent experiments.

## E. Combination with Gaussian Pruning

Gaussian pruning can be incorporated into the 3DGS compression pipeline using image codecs. To evaluate RDCompGS under this setting, we first adopt the pruning procedure in LightGaussian and optimize the Gaussian representation using RDCompGS. In this process, pruning reduces the number of Gaussian primitives, while RDCompGS performs R-D optimization of the remaining Gaussian representation with respect to subsequent coding.

As shown in Table III, the combination of RDCompGS with Gaussian pruning achieves consistent R-D improvements over

TABLE III  
R-D PERFORMANCE COMPARISON WITH LIGHTGAUSSIAN UNDER H.265/HEVC COMPRESSION. HERE, OURS DENOTES RDCOMPGS WITH THE PRUNING STRATEGY PROPOSED IN LIGHTGAUSSIAN. SIZE IS MEASURED IN MB AND PSNR IN DB.
<table><tr><td>Dataset</td><td>Method</td><td colspan="2"> $\mathrm { Q P } = 4$ </td><td colspan="4"> $\mathrm { Q P = 1 0 }$ </td><td colspan="4"> $\mathrm { Q P } = 1 6$ </td><td colspan="2"> $\mathrm { Q P } = 2 2$ </td><td>BD-Rate (%) ↓</td></tr><tr><td></td><td></td><td>Size</td><td>PSNR SSIM LPIPS</td><td>Size</td><td>PSNR</td><td></td><td>SSIM LPIPS</td><td> $\mathrm { s i z e }$ </td><td>PSNR</td><td>SSIM LPIPS</td><td>Size</td><td>PSNR</td><td>SSIM LPIPS</td><td></td></tr><tr><td>Tanks &amp; Temples LightGaussian</td><td></td><td>15.367</td><td>21.52 0.780</td><td>0.238 12.431</td><td>21.46</td><td>0.776</td><td>0.241</td><td>9.692</td><td>21.24</td><td>0.767</td><td>0.251</td><td>7.088 20.67</td><td>0.736 0.283</td><td></td></tr><tr><td>(Train)</td><td>Ours</td><td>14.657</td><td>21.52 0.781</td><td>0.237 10.983</td><td>21.47</td><td>0.778</td><td>0.240</td><td>7.012</td><td>21.34</td><td>0.768 0.251</td><td>3.868</td><td>20.76</td><td>0.743 0.278</td><td>-38.83</td></tr><tr><td>Deep Blending</td><td>LightGaussian</td><td>36.044</td><td>29.76 0.898</td><td>0.256 29.350</td><td>29.72</td><td>0.897</td><td>0.257</td><td>23.069</td><td>29.57</td><td>0.894 0.262</td><td>17.042</td><td>29.06</td><td>0.883 0.277</td><td></td></tr><tr><td></td><td>Ours</td><td>25.096</td><td>29.95 0.902</td><td>0.254 13.073</td><td>29.86</td><td>0.901</td><td>0.257</td><td>7.861</td><td>29.63</td><td>0.897 0.265</td><td>5.731</td><td>28.96</td><td>0.885 0.285</td><td>-67.19</td></tr><tr><td>Mip-NeRF 360</td><td>LightGaussian</td><td>39.069</td><td>26.65 0.789</td><td>0.232 31.674</td><td>26.54</td><td>0.786</td><td>0.236</td><td>24.776</td><td>26.21</td><td>0.776 0.247</td><td>18.223</td><td>25.38</td><td>0.749 0.276</td><td></td></tr><tr><td></td><td>Ours</td><td>32.578</td><td>26.77 0.791</td><td>0.233 21.254</td><td>26.64</td><td>0.787</td><td>0.239</td><td>11.891</td><td>26.22</td><td>0.776</td><td>0.254 7.085</td><td>25.35</td><td>0.750 0.282</td><td>-54.54</td></tr></table>

![](images/34377bbe6ee2486cfeda9b9de6f470355ba9700006c926fe37bbfc91ba24e18c.jpg)  
Fig. 10. BD-Rate comparison of different $\beta$ values on the Truck scene under H.265/HEVC compression.

LightGaussian on all three datasets. The BD-Rate reductions are 38.83%, 67.19%, and 54.54% on Tanks & Temples, Deep Blending, and Mip-NeRF 360, respectively, corresponding to an average reduction of 53.52%. Across different QPs, RDCompGS substantially reduces the compressed size while maintaining comparable or slightly higher PSNR and SSIM, with LPIPS remaining generally comparable to that of Light-Gaussian. These results demonstrate that RDCompGS remains effective when combined with Gaussian pruning, providing additional R-D gains for the pruned Gaussian representation.

## F. Ablation Studies

We conduct ablation studies to evaluate the two main components of RDCompGS, including differentiable rate estimation and the rate constraint adjustment model. Their effects are evaluated separately under H.265/HEVC compression.

1) Effectiveness of Differentiable Rate Estimation: To evaluate the effectiveness of differentiable rate estimation, we remove the estimated rate term from Gaussian optimization while keeping the other settings unchanged. As shown in Table IV, incorporating rate estimation consistently improves the R-D performance across all three datasets, with BD-Rate reductions exceeding 50%. In particular, the reduction reaches 62.32% on Deep Blending, while positive BD-PSNR gains are obtained on all three datasets. These results demonstrate that differentiable rate estimation effectively connects Gaussian representation optimization with subsequent coding, leading to substantial R-D improvements.

TABLE IV  
ABLATION OF DIFFERENTIABLE RATE ESTIMATION UNDER H.265/HEVC COMPRESSION, USING GAUSSIAN OPTIMIZATION WITHOUT THE ESTIMATED RATE TERM AS THE REFERENCE.
<table><tr><td>Dataset</td><td>BD-Rate (%) ↓</td><td>BD-PSNR (dB) ↑</td></tr><tr><td>Tanks &amp; Temples (Train)</td><td>-53.28</td><td>0.63</td></tr><tr><td>Deep Blending</td><td>-62.32</td><td>0.88</td></tr><tr><td>Mip-NeRF 360</td><td>-56.09</td><td>1.18</td></tr></table>

TABLE V

COMPARISON OF THE RATE CONSTRAINT ADJUSTMENT MODEL WITH THEUSE OF A SINGLE RATE CONSTRAINT UNDER H.265/HEVCCOMPRESSION, USING THE LATTER AS THE REFERENCE.
<table><tr><td>Dataset</td><td>BD-Rate (%)↓</td><td>BD-PSNR (dB) ↑</td></tr><tr><td>Tanks &amp; Temples (Train)</td><td>0.77</td><td>0.00</td></tr><tr><td>Deep Blending</td><td>-0.59</td><td>0.02</td></tr><tr><td>Mip-NeRF 360</td><td>-5.46</td><td>0.25</td></tr></table>

2) Effectiveness of the Rate Constraint Adjustment Model: We further evaluate the rate constraint adjustment model while retaining differentiable rate estimation. Specifically, a single rate constraint with $\lambda _ { \mathrm { { r a t e } } } ~ = ~ 1 \times 1 0 ^ { - 7 }$ is used across all compression conditions instead of the adjustment model, while the remaining settings are kept unchanged.

As shown in Table V, the rate constraint adjustment model provides non-negative BD-PSNR gains on all three datasets and improves the overall BD-Rate performance. Although the BD-Rate increases slightly by 0.77% on the Train scene, reductions are obtained on Deep Blending and Mip-NeRF 360, with the latter reaching 5.46%. Averaged over the three datasets, the adjustment model achieves a BD-Rate reduction of 1.76%. These results indicate that the adjustment model provides a more appropriate rate constraint under different compression conditions, leading to a more favorable overall R-D trade-off than using a single rate constraint.

## V. CONCLUSION

In this work, we revisit 3DGS compression with 2D representations from the perspective of R-D optimization, with the aim of better aligning Gaussian attribute optimization with subsequent coding. We introduce RDCompGS, which combines rate estimation with the rate constraint adjustment model to balance compression and rendering quality under different coding conditions. Extensive experiments on three benchmark datasets demonstrate consistent R-D improvements across different codec settings, and RDCompGS remains effective when Gaussian pruning is incorporated into compression with image codecs.

Future work will further explore joint modeling of coding rate and compression distortion. Extending RDCompGS to dynamic scenes is also an important direction for future research.

## REFERENCES

[1] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, “3d gaussian¨ splatting for real-time radiance field rendering,” ACM Trans. Graph., vol. 42, no. 4, pp. 139:1–139:14, 2023.

[2] Y. Tian, W. Chu, W. Zhou, X. Tang, and K. Li, “Hfrs: A high-fidelity framework for autonomous driving scene reconstruction and simulation based on gaussian splatting,” IEEE Trans. Multimedia, pp. 1–12, 2026.

[3] H. Shuai, Y. Shi, Y. Sun, and Q. Liu, “Adversarial pruning networks for compact 3d gaussian splatting,” IEEE Trans. Multimedia, vol. 28, pp. 1080–1089, 2026.

[4] Z. Liang et al., “4dgstream: Variable bitrate dynamic gaussian splatting streaming,” IEEE Trans. Multimedia, vol. 28, pp. 6877–6891, 2026.

[5] Y. Zhao, G. Chen, B. Wu, Q. Jin, and T. Zeng, “Msa-splatting: Multiscale adaptive gaussian splatting for high-fidelity view synthesis,” IEEE Trans. Multimedia, vol. 28, pp. 5357–5367, 2026.

[6] D. Wu et al., “Fieldgs: Unsigned distance fields in gaussian splatting for enhanced surface reconstruction,” IEEE Trans. Multimedia, pp. 1–11, 2026.

[7] M. S. Ali, C. Zhang, M. Cagnazzo, G. Valenzise, E. Tartaglione, and S.-H. Bae, “Compression in 3d gaussian splatting: A survey of methods, trends, and future directions,” IEEE Trans. Circuits Syst. Video Technol., vol. 36, no. 7, pp. 9291–9308, 2026.

[8] G. Chen and W. Wang, “A survey on 3d gaussian splatting,” ACM Comput. Surv., vol. 58, no. 12, 2026.

[9] J. C. Lee, D. Rho, X. Sun, J. H. Ko, and E. Park, “Compact 3d gaussian representation for radiance field,” in Proc. IEEE/CVF Conf. Comput. Vis Pattern Recognit., 2024, pp. 21 719–21 728.

[10] S. Niedermayr, J. Stumpfegger, and R. Westermann, “Compressed 3d gaussian splatting for accelerated novel view synthesis,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024, pp. 10 349– 10 358.

[11] H. Wang et al., “End-to-end rate-distortion optimized 3d gaussian representation,” in Proc. Eur. Conf. Comput. Vis., 2024, pp. 76–92.

[12] S. Girish, K. Gupta, and A. Shrivastava, “Eagles: Efficient accelerated 3d gaussians with lightweight encodings,” in Proc. Eur. Conf. Comput. Vis., 2024, pp. 54–71.

[13] Y. Chen, M. Li, Q. Wu, W. Lin, M. Harandi, and J. Cai, “Pcgs: Progressive compression of 3d gaussian splatting,” in Proc. AAAI Conf. Artif. Intell., vol. 40, no. 4, 2026, pp. 3111–3119.

[14] Y. Chen, Q. Wu, M. Li, W. Lin, M. Harandi, and J. Cai, “Fast feedforward 3d gaussian splatting compression,” in Proc. Int. Conf. Learn. Represent., vol. 2025, 2025, pp. 74 859–74 872.

[15] Y. Huang, J. Pang, F. Zhu, and D. Tian, “Entropygs: an efficient entropy coding on 3d gaussian splatting,” in Proc. IEEE Int. Conf. Acoust. Speech Signal Process., 2026, pp. 11 732–11 736.

[16] P. Papantonakis, G. Kopanas, B. Kerbl, A. Lanvin, and G. Drettakis, “Reducing the memory footprint of 3d gaussian splatting,” Proc. ACM Comput. Graph. Interact. Tech., vol. 7, no. 1, pp. 1–17, 2024.

[17] Z. Fan, K. Wang, K. Wen, Z. Zhu, D. Xu, and Z. Wang, “Lightgaussian: Unbounded 3d gaussian compression with 15x reduction and 200+ fps,” in Adv. Neural Inf. Process. Syst., vol. 37, 2024, pp. 140 138–140 158.

[18] T. Lu et al., “Scaffold-gs: Structured 3d gaussians for view-adaptive rendering,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024, pp. 20 654–20 664.

[19] Y. Chen, Q. Wu, W. Lin, M. Harandi, and J. Cai, “Hac: Hash-grid assisted context for 3d gaussian splatting compression,” in Proc. Eur. Conf. Comput. Vis., 2024, pp. 422–438.

[20] ——, “Hac++: Towards 100x compression of 3d gaussian splatting,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 47, no. 11, pp. 10 210– 10 226, 2025.

[21] Y. Wang, Z. Li, L. Guo, W. Yang, A. C. Kot, and B. Wen, “Contextgs: Compact 3d gaussian splatting with anchor level context model,” in Adv. Neural Inf. Process. Syst., vol. 37, 2024, pp. 51 532–51 551.

[22] Y.-T. Zhan et al., “Cat-3dgs: A context-adaptive triplane approach to rate-distortion-optimized 3dgs compression,” in Proc. Int. Conf. Learn Represent., vol. 2025, 2025, pp. 26 312–26 334.

[23] A. Hanson, A. Tu, V. Singla, M. Jayawardhana, M. Zwicker, and T. Goldstein, “Pup 3d-gs: Principled uncertainty pruning for 3d gaussian splatting,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2025, pp. 5949–5958.

[24] D. R. Freitas, I. Tabus, and C. Guillemot, “Visibility-based geometry pruning of neural plenoptic scene representations,” IEEE Trans. Multimedia, vol. 28, pp. 1–16, 2026.

[25] Z. Chen, S. Wan, M. Yu, F. Yang, and J. Hou, “Refine: Super-efficient 3d gaussian splatting pruning via rendering-free primitive importance,” in Proc. Eur. Conf. Comput. Vis., 2026, pp. 131–147.

[26] Z. Chen, S. Wan, F. Yang, J. Xia, W. Lin, and J. Hou, “Spare-gs: Structural parsimony and resource efficiency for 3d gaussian splatting,” arXiv preprint arXiv:2607.16624, 2026.

[27] Q. Yang, G. Van Der Auwera, and Z. Li, “Lightweight 3d gaussian splatting compression via video codec,” in Proc. Data Compression Conf., 2026, pp. 362–371.

[28] Z. Zhu, Y. Liao, and L. Yu, “Dual-hilbert scan for efficient video-based gaussian splatting compression,” in Proc. IEEE Int. Symp. Circuits Syst., 2026, pp. 4824–4828.

[29] P. Wang et al., “3dgcoding: Novel framework for 3d gaussian video incremental training and coding,” in Proc. IEEE Int. Conf. Multimedia Expo, 2025, pp. 1–6.

[30] S. Lee, F. Shu, Y. Sanchez, T. Schierl, and C. Hellge, “Compression of 3d gaussian splatting with optimized feature planes and standard video codecs,” in Proc. IEEE/CVF Int. Conf. Comput. Vis., 2025, pp. 25 496– 25 505.

[31] W. Morgenstern, F. Barthel, A. Hilsmann, and P. Eisert, “Compact 3d scene representation via self-organizing gaussian grids,” in Proc. Eur. Conf. Comput. Vis., 2024, pp. 18–34.

[32] P. Wang et al., “V<sup>3</sup>: Viewing volumetric videos on mobiles via streamable 2d dynamic gaussians,” ACM Trans. Graph., vol. 43, no. 6, pp. 1–13, 2024.

[33] E. Y. Lam and J. W. Goodman, “A mathematical analysis of the dct coefficient distributions for images,” IEEE Trans. Image Process., vol. 9, no. 10, pp. 1661–1666, 2000.

[34] A. Knapitsch, J. Park, Q.-Y. Zhou, and V. Koltun, “Tanks and temples: Benchmarking large-scale scene reconstruction,” ACM Trans. Graph., vol. 36, no. 4, pp. 1–13, 2017.

[35] P. Hedman, J. Philip, T. Price, J.-M. Frahm, G. Drettakis, and G. Brostow, “Deep blending for free-viewpoint image-based rendering,” ACM Trans. Graph., vol. 37, no. 6, pp. 1–15, 2018.

[36] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman, “Mip-nerf 360: Unbounded anti-aliased neural radiance fields,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2022, pp. 5470–5479.

[37] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: from error visibility to structural similarity,” IEEE Trans. Image Process., vol. 13, no. 4, pp. 600–612, 2004.

[38] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2018, pp. 586– 595.

[39] G. Bjontegaard, “Calculation of average psnr differences between rdcurves,” ITU SG16 Doc. VCEG-M33, 2001.

[40] G. J. Sullivan, J.-R. Ohm, W.-J. Han, and T. Wiegand, “Overview of the high efficiency video coding (hevc) standard,” IEEE Trans. Circuits Syst. Video Technol., vol. 22, no. 12, pp. 1649–1668, 2012.

[41] J. Alakuijala et al., “Jpeg xl next-generation image compression architecture and coding tools,” in Applications of digital image processing XLII, vol. 11137. SPIE, 2019, pp. 112–124.

[42] M. Budagavi, A. Fuldseth, G. Bjøntegaard, V. Sze, and M. Sadafale, “Core transform design in the high efficiency video coding (hevc) standard,” IEEE J. Sel. Topics Signal Process., vol. 7, no. 6, pp. 1029– 1041, 2013.