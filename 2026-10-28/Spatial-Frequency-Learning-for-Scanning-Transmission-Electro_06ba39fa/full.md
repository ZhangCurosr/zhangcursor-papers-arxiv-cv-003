# Spatial-Frequency Learning for Scanning Transmission Electron Microscopy Image Enhancement

Hesong Li, Ying Fu, Senior Member, IEEE, Ziqi Wu, Ruiwen Shao, and Tao Zhang

Abstract—Scanning Transmission Electron Microscopy (STEM) enables the observation of atomic arrangements at sub-angstrom resolution and plays a key role in the discovery and characterization of advanced materials. Raw STEM images are often noisy and blurred due to multiple degradations. Although recent deep learning methods improve STEM images, their performance remains limited under severe degradations. In this work, we propose a spatial-frequency learning framework for STEM image enhancement. Specifically, we design a spatial-frequency fusion network that exploits periodic atomic arrangements in the frequency domain and restores local atomic details in the spatial domain. Spatial- and frequency-domain features are enhanced separately and then fused. We also present a STEM noise calibration method for synthesizing more realistic STEM images. The parameters of scan noise, pointwise noise, and background noise are estimated through statistical analysis and fitting of real STEM images. Using these calibrated parameters, we construct a diverse dataset containing a wider range of atomic arrangements, imaging modes, and degradations. Random masks combine different atomic arrangements to simulate interfaces between multiple materials. Experimental results demonstrate that our method achieves state-of-the-art enhancement performance on synthetic datasets and real images of advanced materials while improving atomic localization accuracy.

Index Terms—Scanning transmission electron microscopy, image enhancement, spatial-frequency learning, noise calibration, data generation, atomic localization.

## I. INTRODUCTION

Scanning Transmission Electron Microscopy (STEM) enables scientific observation at the atomic scale and plays a crucial role in the discovery and study of materials. Recent studies have discovered new properties of advanced materials using STEM, including semiconductor materials [1], [2], superconducting materials [3], new energy materials [4]– [6], functional materials [7], [8], neuromorphic computing materials [9], and nanodevices [10].

Although STEM provides atomic-scale resolution, its images are susceptible to multiple types of noise [11], [12]. These include scan noise caused by probe jitter, pointwise noise introduced during signal readout, and background interference caused by impurities. These types of noise can overwhelm useful signals and hinder the observation of material structures. To avoid sample damage, both the exposure time and electronbeam intensity are constrained, further increasing the noise level. Moreover, spherical aberration blurs atomic contours and makes different atoms difficult to distinguish. Therefore, STEM image enhancement is needed to improve the visualization and analysis of material structures.

![](images/4cda3b6a675228195d15df9874efa9532aeb0caab1513d0f87a00ceeb9641a35.jpg)  
Fig. 1. PSNR results of different methods on STEM image enhancement. Our spatial-frequency fusion network (SFFN) outperforms AtomSegNet [12], FCN [13], AST [14], GAN [15], FADC [16], and SFIN [17] on three datasets (i.e., TEMImageNet [12], SFIN dataset [17], and our AtoMix). HAADF and BF denote two imaging modes, namely high-angle annular dark-field imaging and bright-field imaging, respectively.

Simple image filtering methods (e.g., Gaussian filter, Wiener filter [18], and bilateral filter [19]) can partially remove noise, but their effectiveness is limited, and they cannot eliminate background interference and spherical aberration. Recently, some deep learning-based methods [12], [13], [15], [20] have been designed for electron microscope image enhancement. Unsupervised methods (e.g., UDVD [20]) typically utilize continuous multi-frame STEM images as training data. Since adjacent frames are used as supervision signals, they can only remove noise that varies across frames, but struggle to eliminate background and spherical aberration, which persist in every frame. By synthesizing simulated noisy images and ground truth images, supervised methods (e.g., AtomSegNet [12], FCN [13], and GAN [15]) can simultaneously remove noise, background, and spherical aberration, and generate output images suitable for other tasks such as atomic localization.

Therefore, the performance of neural networks and the quality of synthetic data determine the effectiveness of the STEM image enhancement methods.

One challenge is that STEM image signals are often overwhelmed by noise, making accurate feature extraction difficult. Existing networks [12], [13] mainly process spatial-domain features and neglect the strong prior induced by periodic atomic arrangements in the frequency domain. To address this issue, we design a spatial-frequency fusion network for STEM image enhancement. Our network restores atomic contours in the spatial domain while exploiting frequency-domain priors to suppress noise and enhance periodic structures. It processes features in both domains and adaptively fuses their complementary information. As shown in Fig. 1, our network outperforms existing state-of-the-art methods.

The second challenge is the lack of realistic data for training deep learning models. On one hand, the prolonged exposure can damage the material sample and does not eliminate background and spherical aberration, so the ideal clean STEM images as supervised labels cannot be obtained through actual imaging. On the other hand, existing synthetic datasets [11], [12], [21] lack reality in noise distribution. STEM images have unique scan noise related to image gradients, making traditional calibration methods requiring uniform calibration frames [22], [23] unavailable. Thus, existing datasets usually use random noise parameters, which are different from those in real STEM images. In this work, we design a STEM image calibration method capable of getting various noise parameters from STEM images with atomic patterns, which does not require uniform calibration frames. We calibrate the noise parameters in real STEM images and apply them to synthesize noisy images, thereby ensuring the reality of the simulated STEM images in our dataset.

Existing datasets also lack variety, and their scope of application is limited. First, they only include regular atomic patterns and exclude random arrangements, which often appear at material interfaces and are key parts in observing material changes. They also lack images with multiple arrangements. Second, they only include High-Angle Annular Dark-Field (HAADF) mode images, missing Bright-Field (BF) mode images, which are more suitable for light atoms observation [24]–[26]. Third, most datasets lack simulations of more complex degradations, including spherical aberration, lower atomic brightness, and lower resolution. To address these issues, we construct a new dataset called AtoMix, which contains images with mixed atomic arrangements. It includes simulations of the various imaging modes, atom arrangements, and degradations mentioned earlier, ensuring data diversity. Our dataset includes complex degraded images that are closer to real-world data.

Experimental results show that both our spatial-frequency fusion network and our AtoMix dataset are effective for STEM image enhancement. Our method makes STEM images clearer and atomic localization more accurate. The application results on various advanced materials demonstrate the effectiveness of our method and its role in material research.

In summary, our main contributions are that we

• Present a spatial-frequency fusion network, which can flexibly combine spatial-domain and frequency-domain information to enhance the features of STEM images.

TABLE I  
COMPARISON OF MAIN EXTENSIONS FROM THE CONFERENCE VERSION TO THE JOURNAL VERSION.
<table><tr><td colspan="2">Category</td><td>Conference version</td><td>Journal version Learnable</td></tr><tr><td rowspan="2">Network</td><td>Spatial-frequency interaction</td><td>Fixed addition</td><td>spatial-frequency fusion</td></tr><tr><td>Frequency-position guidance</td><td>Not included</td><td>Included</td></tr><tr><td rowspan="3">Dataset</td><td>Atom patterns</td><td>15 patterns from 5 crystal systems</td><td>46 patterns from 7 crystal systems Regular, random</td></tr><tr><td>Atom arrangements</td><td>Regular, random addition, and random missing</td><td>addition, random missing, and mixed arrangements</td></tr><tr><td>Degradation</td><td>Noise (background noise  $\begin{array} { r } { \dot { N } _ { \mathrm { b a c k g r o u n d } } , } \end{array}$  scan noise  $\bar { N } _ { \mathrm { s c a n } } ,$  and pointwise noise  $\mathbf { \dot { N } } _ { \mathrm { p o i n t w i s e } } )$ </td><td>Noise (Nbackground,  $N _ { \mathrm { { s c a n } } } , \mathrm { { a \bar { n } d } }$   $N _ { \mathrm { p o i n t w i s e } } ) ,$  spherical aberration, lower brightness, and lower resolution</td></tr></table>

• Design a STEM noise calibration method, which can be used to synthesize more realistic STEM images.

• Build a more diverse dataset that contains more types of atom arrangements, imaging modes, and degradations.

This paper is an extension of a preliminary conference paper SFIN [17]. Table I summarizes the main extensions from the conference version to this journal version. In terms of network architecture, we design the Spatial-Frequency Fusion Network (SFFN), which replaces the fixed spatial-frequency addition in SFIN with learnable spatial-frequency fusion and further introduces frequency-position guidance for fine-grained frequency-domain processing. In terms of dataset, we build AtoMix by expanding the atom patterns from 15 patterns from 5 crystal systems to 46 patterns from 7 crystal systems, introducing mixed atomic arrangements for simulating material interfaces, and incorporating stronger degradations, including spherical aberration, lower brightness, and lower resolution. These extensions enable our method to achieve stronger enhancement performance on synthetic datasets and real data of advanced materials. In addition, we add comparisons with more methods, including transformers, unsupervised methods, and other frequency-domain processing networks, as well as more visual analyses of frequency-domain processing and a detailed introduction to our data synthesis method.

## II. RELATED WORK

In this section, we provide an overview of the related work, including STEM image enhancement, frequency domain networks, and STEM image synthesis.

STEM image enhancement. Natural-image enhancement and recognition [27]–[45] have been widely studied. Transformerbased methods [46]–[50] and diffusion models [51]–[53] lever age long-range dependencies and degradation modeling to achieve superior image restoration performance. For STEM images, three commonly used traditional enhancement methods are the Gaussian filter, Wiener filter [18], and bilateral filter [19], which have been integrated into electron microscopy data processing software such as Gatan Microscopy

Suite. These methods can suppress noise but cannot remove background interference or spherical aberration. Recently, deep learning-based methods [12], [13], [15], [17], [54] have achieved better enhancement performance. A fully convolutional network [13] enhances images and identifies atomic defects, including vacancies and substitutions. A generative adversarial network [15] is pretrained on the large-scale cell electron microscopy dataset CEM500k [55] and fine-tuned for downstream tasks such as STEM image enhancement using the TEMImageNet dataset [12]. AtomSegNet [12] is a convolutional neural network based on the U-Net architecture [56]; it is trained on TEMImageNet [12] for STEM image enhancement and atomic localization. AtomSegNet has played a key role in studies of new battery materials [4]–[6]. Several unsupervised methods (e.g., UDVD [20], 3D-B2U [57], and ZeroIG [58]) have been proposed for denoising electron microscopy or natural images. Although these methods do not require synthetic training data, the absence of ideal groundtruth supervision limits their ability to remove background interference and correct atomic-contour distortions. Existing STEM image enhancement methods [12], [13], [15] operate primarily in the spatial domain and therefore fail to exploit the strong frequency-domain information inherent in periodic atomic arrangements.

Frequency-domain networks. Image signals and noise typically lie in different frequency ranges. Recent studies in natural image processing show that incorporating frequencydomain processing into networks helps distinguish image signals from noise and recover image details. DeepRFT [59] employs Fourier transformation for image deblurring, FSNet [60] dynamically decomposes and selects frequency-domain features, and FADC [16] and FDConv [61] use frequency analysis to guide convolutional feature processing. These existing frequency-domain methods mainly use frequency information to guide spatial-domain processing or perform frequency selection, rather than directly modeling individual frequency locations. In contrast, our SFFN directly processes features in the Fourier domain and introduces frequencyposition guidance to distinguish different diffraction spots. Existing coordinate-guided methods, such as CoordConv [62] and Coordinate Attention [63], inject spatial coordinates into spatial-domain features, whereas our SFFN embeds frequency coordinates into Fourier-domain features to distinguish diffraction spots with different directions and periods. Existing methods either use fixed addition for spatial-frequency interaction (SFIN [17] and DeepRFT [59]) or lack explicit cross-domain interaction (FSNet [60], FADC [16], and FDConv [61]), whereas our SFFN uses learnable spatial-frequency fusion to adaptively combine periodic frequency-domain structures with local spatial anomalies for fine-grained denoising of atomicscale STEM images.

STEM image synthesis. Due to the difficulty of obtaining ideal supervised labels in actual experiments, STEM image synthesis provides data support for deep learning-based STEM image enhancement methods [64], [65]. The abTEM code library [11] provides a noise-free STEM image rendering method that supports multiple imaging modes, but is slow and requires atomic structure files. TEMImageNet dataset [12] contains simulated STEM images in HAADF mode and label images for STEM image enhancement and atomic localization tasks. It includes several fixed shooting field angles and rotation angles and incorporates scan noise, pointwise noise, and background with random parameters. Cycle-GAN [21] is used to learn the mapping relationship between noiseless simulated STEM images and noisy real STEM images and generate more realistic STEM noise. This method relies entirely on reference STEM images and lacks flexibility in controlling various parameters in the data synthesis program. Existing synthetic datasets [12], [21] lack realistic STEM image noise and diverse atomic arrangement patterns, imaging modes, and image degradation types. This results in poor performance of the trained networks when applied to complex real-world data.

## III. METHOD

In this section, we first introduce the task formulation and our motivation. Then, we present our spatial-frequency fusion network. Finally, we introduce our STEM image data synthesis method, which includes clean image generation, noise calibration, and noise generation.

## A. Formulation and Motivation

A STEM image with noise can be represented as the sum of a distorted clean image and various types of noise, i.e.,

$$
I _ { \mathrm { n o i s y } } = \mathrm { S A } ( I _ { \mathrm { c l e a n } } ) + N _ { \mathrm { b a c k g r o u n d } } + N _ { \mathrm { s c a n } } + N _ { \mathrm { p o i n t w i s e } } ,\tag{1}
$$

where $I _ { \mathrm { n o i s y } }$ and $\pmb { I } _ { \mathrm { c l e a n } }$ denote the noisy and ideal clean images, respectively; SA(·) denotes the spherical-aberration degradation operator; and $N _ { \mathrm { b a c k g r o u n d } } , \ N _ { \mathrm { s c a n } }$ , and $N _ { \mathrm { p o i n t w i s e } }$ denote background, scan, and pointwise noise, respectively.

The additive formulation in Eq. (1) is grounded in the physical mechanisms of STEM imaging. $N _ { \mathrm { b a c k g r o u n d } }$ arises from sample contamination, which is independent of the atomic signal and is therefore modeled as additive noise. For $N _ { \mathrm { s c a n } } .$ , arising from probe jitter, and $N _ { \mathrm { p o i n t w i s e } } ,$ , arising from the detector’s readout electronics and dark current, we retain additive formulations for computational tractability but explicitly account for their signal dependence by modeling their variances as functions of the local image gradient and pixel intensity, respectively. These heteroscedastic additive formulations reflect the empirical noise behavior while enabling independent calibration and stable deep learning optimization.

STEM image enhancement aims to derive $\pmb { I } _ { \mathrm { c l e a n } }$ from $I _ { \mathrm { n o i s y } } .$ For HAADF-mode images, the ideal image background is pure black, while the brightest atomic centers are pure white. In contrast, BF-mode images have a pure white background with the most significant atomic centers appearing as pure black. STEM image enhancement requires preserving the relative brightness and size of different atoms in an image, which enables the distinction of various elements from the image.

Guided by the challenges outlined in the Introduction, our methodology is driven by two core motivations. First, while the periodic atomic lattice yields strong frequency signatures that can be leveraged for noise suppression, aperiodic features (e.g., defects, interfaces, or ion embeddings) remain spatially localized and cannot be fully characterized in the frequency domain alone. This necessitates a spatial-frequency fusion strategy that jointly enhances complementary information from both domains. To leverage the distinctive spot patterns of STEM images in the frequency domain, we embed coordinate information into the frequency-domain processing to enable fine-grained handling of information across different frequency bands. Second, robust model generalization requires training data that faithfully replicates real-world degradation characteristics. Since STEM scanning noise is intrinsically gradient-dependent, conventional flat-field calibration is inapplicable, motivating our novel parameter extraction scheme. Furthermore, to bridge the simulation-to-experiment gap, our synthesized dataset explicitly incorporates mixed atomic arrangements, dual imaging modes (HAADF/BF), and complex degradations (e.g., spherical aberration and intensity loss). The detailed network architecture and data synthesis pipeline are elaborated in Sections III-B and III-C, respectively.

![](images/34f43d3742ff71fca62449351674a60b421669c2df727c7698a53cf4e96ddd37.jpg)  
Fig. 2. Motivation of our spatial-frequency fusion network. FFT and IFFT stand for Fast Fourier Transform and Inverse Fast Fourier Transform, respectively. A simple filter in the frequency domain can achieve good denoising effects for STEM images.

## B. Spatial-Frequency Fusion Network

As discussed in Section III-A, jointly modeling spatial- and frequency-domain information enables the network to capture both periodic atomic arrangements and local irregularities, thereby improving image enhancement. We next describe the overall network pipeline, the spatial-frequency fusion block, and the coordinate-guided frequency convolution used in each block.

Overall pipeline. Our network follows the basic architecture of common image restoration networks [14], [66], [67], including two convolutions at the head and tail, and multiple blocks stacked in the middle, as shown in Fig. 3(a). First, a 3×3 convolution extracts shallow features from the single-channel input image to form a feature map with C channels. Multiple subsequent stacked spatial-frequency fusion blocks gradually enhance these features, where each block uses residual connections to stabilize training. Finally, a 3×3 convolution fuses the enhanced features into a single-channel output image.

Spatial-frequency fusion block. As shown in Fig. 3(b), we design the spatial-frequency fusion block as the basic module of our network, which can simultaneously enhance the spatial and frequency domain information in the feature maps. First, the feature map with C channels is evenly divided into two parts along the channel dimension, each with C/2 channels. These two parts of features are then directed to two separate branches for enhancement in the spatial and frequency domains, respectively. The spatial domain branch employs a common convolutional module for enhancement, consisting of two 3×3 convolutions, two batch normalizations, and two ReLU (Rectified Linear Unit) activation functions. In the frequency domain branch, the 3×3 convolution is replaced by a specially designed coordinate-guided frequency convolution (Fig. 3(c)) that transforms features to the frequency domain for enhancement. It will be introduced later.

![](images/a4aa25601dd8dfc2064ac0a72c53e2c916345eace9103a35458578caf8cbb9c8.jpg)  
Fig. 3. Framework of our Spatial-Frequency Fusion Network (SFFN). In each Spatial-Frequency Fusion Block (SFFB), the input features are processed by spatial- and frequency-domain branches, whose outputs are fused through a convolution. The Coordinate-Guided Frequency Convolution (CGFC) transforms features into the frequency domain using a real-valued Fast Fourier Transform (RFFT) and processes different frequency components with two coordinate channels. “//” denotes floor division.

The spatial- and frequency-domain branches process complementary information. Periodic atomic arrangements produce strong frequency-domain signatures that can be enhanced by the frequency-domain branch. Moreover, because signal and noise occupy different frequency ranges, this branch can enhance atomic signals while suppressing noise. The spatialdomain branch processes localized aperiodic structures that are difficult to characterize solely in the frequency domain, including material interfaces, atomic vacancies, and intercalated ions. Fusing the outputs of the two branches enables more comprehensive feature enhancement.

Unlike the fixed cross-domain addition used in SFIN [17], we adopt a convolutional operation to fuse features from the two domains. Specifically, the outputs of the spatialand frequency-domain branches are concatenated along the channel dimension and then fused by a 3×3 convolution. Because the complementary information from the two domains does not necessarily follow an additive relationship, convolutional fusion provides greater flexibility. This change does not increase computational complexity or the parameter count relative to SFIN [17].

Coordinate-guided frequency convolution. We design the coordinate-guided frequency convolution to enhance features in the frequency domain. It distinguishes signal from noise by processing frequency-domain features augmented with frequency-position coordinates.

First, the real-valued fast Fourier transform (i.e., the “rfftn” function in PyTorch [68]) is used to convert each channel of the input feature map into the frequency domain separately. Since the input is a real-valued matrix, the transformed matrix exhibits conjugate symmetry, so only half of the information in the resulting feature map needs to be retained. As shown in Fig. 3(c), the width of the resulting frequency-domain feature map is approximately halved. Due to the complexvalued result, the number of channels doubles, including both real and imaginary parts.

![](images/490350abc8e07a623a581257a8ef8a151a59a0ae7c87b8ff1f24cbb50f2d6d96.jpg)  
Fig. 4. Our STEM image synthesis framework, including (a) clean image generation, (b) noise calibration, and (c) noise generation. Clean image generation obtains ground-truth labels and clean images without noise by mixing different atomic arrangement patterns via masks. Noise generation progressively adds various noises to the clean images, where the noise parameters are derived from the noise calibration of real images. Image noise parameters include: $\beta _ { \mathrm { m i n } }$ and $\beta _ { \mathrm { r a n g e } }$ are the minimum background brightness and brightness range, respectively; $\sigma _ { \mathrm { s c a n } }$ is the intensity of scan noise; $\sigma _ { \mathrm { p o i n t } }$ and $\lambda _ { \mathrm { p o i n t } }$ are the intensity and the distribution shape parameter of pointwise noise, respectively.

To enable the convolution to distinguish different frequency locations, we append two coordinate maps to the frequencydomain features, representing the x and y coordinates of each location. A 1×1 convolution then processes the frequency components while considering their positions, with inputs comprising the two coordinate channels and the remaining 2C channels of complex-valued features. Following a standard convolutional-module design, batch normalization and a ReLU activation are applied after the 1×1 convolution. Finally, the enhanced frequency-domain features are transformed back to the spatial domain using the inverse real-valued fast Fourier transform (i.e., the “irfftn” function in PyTorch [68]).

## C. STEM Image Synthesis

It is difficult to obtain ideal supervising labels for image enhancement from STEM imaging. To train the network using paired data, it is necessary to first synthesize ideal clean images and then add various types of degradations to simulate real noisy STEM images. As shown in Fig. 4, our STEM image synthesis framework includes clean image generation, noise calibration, and noise generation. Clean image generation obtains ground-truth labels and clean images without noise by mixing different atomic arrangement patterns via masks. Noise generation progressively adds various noises to the clean images, where the noise parameters are derived from the noise calibration of real images. The following are the details of each step in the order of the data synthesis process.

Atom pattern generation from units. Materials observed by STEM are mostly crystalline structures, whose atomic arrangements exhibit significant periodicity. Therefore, the entire image can be formed by tiling repeated pattern units.

First, we generate 46 pattern units, each representing a periodic unit of a complete image, as shown on the left side of Fig. 4(a). These pattern units are rendered from 37 crystal structures viewed from four orientations. Because of crystal symmetry, some structures produce the same pattern when viewed from different orientations. Our dataset covers the seven major crystal systems defined by the International Union of Crystallography (IUCr), more than previous datasets [12], [21], thereby supporting a wider range of common material structures.

Theoretically, a clean STEM image consists of numerous circular spots, each corresponding to an atomic column or a single atom and modeled using a 2D Gaussian distribution. The atomic species and column thickness determine the size and brightness of these spots. In HAADF images, atomic columns appear bright because the detector collects high-angle scattered electrons. In BF images, atomic columns appear dark because scattering reduces the intensity collected by the brightfield detector [24]–[26]. The relative brightness and size of different atoms are fixed and can be simulated by the abTEM library [11]. Note that the abTEM library cannot be directly used to synthesize the STEM image dataset, as its physicsbased rendering is slow and fails to flexibly synthesize images with different pattern combinations and various noises. In our data synthesis, abTEM is only used to obtain the arrangement structures of different pattern units, as well as the relative brightness and size parameters of atoms. These parameters are then used to render each atom with a 2D normal distribution.

![](images/3a54ae7d3cb9ba64ba26de73a31bda60e7596bd40868e216aa1f8c9b6b515ec1.jpg)  
(a) Noisy Input Image

![](images/4a2b926881aca631a733b114fc0d8c3c8d730d3ab246c4ce6b50920ece75ca85.jpg)

![](images/06b499b1368d10af29170412b82aa1b4f30aa8df8e5d0b88c5d3fc0049fe2b64.jpg)

![](images/5b7431898fead0b08be428e592aabaae78dbb2b5eb8702f2d69d0f93d96c5606.jpg)

(b) Pointwise Noise  
![](images/8036b42fe4b03966938d99a47c985e587c9b2249b78ec76299dbbf8b9cff04fc.jpg)  
(d) Scan Noise

(c) Pointwise Noise Parameters  
![](images/aa87903b7aba0c676d1a4fd49099ca0cf47d4b14d243d18624969502cc8b1442.jpg)  
(e) Scan Noise Parameters  
Fig. 5. Pointwise noise (b) and scan noise (d) extracted from the noisy real input (a), together with their parameter variations in (c) and (e), respectively. $\mathrm { \mathrm { ^ { * } S D ^ { \mathrm { * } } } }$ denotes standard deviation. The light-red rectangular region in (e) indicates fluctuations caused by limited data and is excluded during fitting.

After randomly selecting a pattern unit, we periodically tile it to form an image with a consistent atomic arrangement. To diversify the orientations and scales of the synthesized data, the pattern unit is randomly rotated and scaled for each image. The resulting interatomic spacing ranges from approximately 7 to 30 pixels, based on measurements of real STEM images. The pattern unit is then tiled along two lattice directions, $\alpha _ { 1 }$ and $\alpha _ { 2 } ,$ to fill the image. Different pattern units have different $\alpha _ { 1 }$ and $\alpha _ { 2 }$ , which are not necessarily orthogonal. By expressing the four image corners in the coordinate system defined by $\alpha _ { 1 }$ and $\alpha _ { 2 }$ , we determine the number of repetitions required along each direction to cover the entire image.

Mask-based pattern mixing. Observing the interface between two atomic arrangements is important for analyzing interactions between materials and material growth. Therefore, the synthetic data should include combinations of multiple atomic arrangements with relatively random structures at their interfaces. Unlike previous datasets that simulate only one atomic arrangement per image, we use binarized Perlin noise [69] as a mask to mix two different arrangements. We first generate an image of one arrangement and replace the region where the mask equals 1 with another arrangement. The second arrangement is set to an empty arrangement with a probability of 0.1 to simulate the vacuum outside a material boundary.

We then generate ground-truth and degraded images from the mixed atomic arrangements. To facilitate the adjustment of atomic brightness and size, each arrangement is represented as a list of per-atom attributes, including coordinates, brightness, and size, rather than being directly rendered as an image. To simulate atomic defects and ion intercalation, atoms are randomly removed from and added to the mixed arrangement. We approximate this process by relocating about 10% of the atoms to random positions. When generating ground-truth images for enhancement, the atomic sizes remain unchanged, while the brightness values are proportionally scaled so that the brightest atom has an intensity of 255 in HAADF mode. When generating ground truth for atomic localization, each atom is rendered as a disk with a radius of 1 pixel and an intensity of 255.

When generating a degraded image, the radius of each atom is doubled to approximate spherical-aberration-induced probe broadening. Spherical aberration in STEM originates from lens imperfections that broaden the probe and effectively widen its point spread function. Because the point spread function is commonly approximated by a 2D Gaussian distribution, increasing the Gaussian radius mimics this probe-broadening effect, which appears as blurred and enlarged atomic columns in the spatial domain. In addition, the intensity of the entire image is multiplied by a random value between 0.2 and 1 to simulate signal attenuation.

Next, background, scan noise, and pointwise noise are sequentially added to the degraded image to form the final noisy image (Fig. 4(c)). The noise parameters are obtained by calibrating real images (Fig. 4(b)).

Background calibration and generation. Background $N _ { \mathrm { b a c k g r o u n d } }$ comes from non-uniform sample thickness or impurities such as carbon compounds after long-term electron beam exposure. It can be simulated using 2D Perlin noise [69]. Since the background varies smoothly at different spatial positions, for a small region in a noiseless HAADF image, the minimum pixel value of the region can represent the local background brightness, while the maximum value represents the atomic brightness. Therefore, by statistically analyzing the minimum values of different regions in real images after Gaussian blurring, the variation range of background brightness in the HAADF mode can be obtained. In contrast, for the BF mode, the background brightness is the maximum value of the pixels in the region.

As shown in Fig. 4(b), small-sized patches $( e . g . , 3 2 { \times } 3 2 )$ I<sup>′</sup> are cropped from the blurred image to calculate the maximum and minimum values $\pmb { I } _ { \mathrm { m a x } }$ and ${ \bf { I } } _ { \mathrm { { m i n } } }$ for each location. Then, the background brightness parameters $\beta _ { \mathrm { m i n } }$ and $\beta _ { \mathrm { r a n g e } }$ can be calculated as

$$
\beta _ { \mathrm { m i n } } = \left\{ \begin{array} { l l } { \displaystyle \operatorname* { m i n } _ { I ^ { \prime } } I _ { \mathrm { m i n } } } & { \mathrm { H A A D F ~ m o d e } , } \\ { 2 5 5 - \displaystyle \operatorname* { m a x } _ { I ^ { \prime } } I _ { \mathrm { m a x } } } & { \mathrm { B F ~ m o d e } , } \end{array} \right.\tag{2}
$$

$$
\beta _ { \mathrm { m a x } } = \left\{ \begin{array} { l l } { \displaystyle \operatorname* { m a x } _ { I ^ { \prime } } I _ { \mathrm { m i n } } } & { \mathrm { H A A D F ~ m o d e } , } \\ { \displaystyle 2 5 5 - \operatorname* { m i n } _ { I ^ { \prime } } I _ { \mathrm { m a x } } } & { \mathrm { B F ~ m o d e } , } \end{array} \right.\tag{3}
$$

$$
\beta _ { \mathrm { r a n g e } } = \beta _ { \mathrm { m a x } } - \beta _ { \mathrm { m i n } } ,\tag{4}
$$

where $\pmb { I ^ { \prime } }$ represents a small block in the blurred image. The final synthetic background is

$$
N _ { \mathrm { b a c k g r o u n d } } = \left\{ \begin{array} { l l } { \beta _ { \mathrm { m i n } } { \bf 1 } + r \beta _ { \mathrm { r a n g e } } N _ { \mathrm { p } } \quad } & { \mathrm { H A A D F ~ m o d e } , } \\ { - \beta _ { \mathrm { m i n } } { \bf 1 } - r \beta _ { \mathrm { r a n g e } } N _ { \mathrm { p } } \quad \mathrm { B F ~ m o d e } , } \end{array} \right.\tag{5}
$$

where 1 is an all-ones matrix of the same size as the image and $N _ { \mathfrak { p } }$ is a 2D Perlin noise [69] between 0 and 1. r is a random number generated for each image between 0 and 1, which controls the intensity of the background.

Scan noise calibration and generation. Scan noise $N _ { \mathrm { s c a n } }$ is primarily due to the jitter of the scanning position. This results in the obtained pixel capturing random areas near the ideal imaging location. When there is significant contrast variation in these areas (such as at the edge of an atom), the pixel value differences caused by the position shift become more pronounced. Therefore, the amplitude of scan noise is approximately proportional to the amplitude of the image gradient at that location. To calibrate its standard deviation $\sigma _ { \mathrm { s c a n } } ,$ it is first necessary to isolate the scan noise from the noisy STEM image. As illustrated in Fig. 4(b), each row of the original image is blurred using a one-dimensional Gaussian filter, resulting in a row-wise blurred image. In it, the pointwise noise is approximately removed by the rowwise blurring, leaving only background and scan noise. Then, the difference between the row-wise blurred image and the blurred image represents the scan noise image, as shown in Fig. 5(d). Fig. 5(e) presents the standard deviation of scan noise as a function of pixel brightness and image gradient amplitude, respectively. It can be observed that the standard deviation of scan noise is proportional to the amplitude of the image gradient. Thus, a linear fit can be applied to determine the slope and intercept of $\sigma _ { \mathrm { s c a n } }$ related to image gradient amplitude, which can then be used to simulate the scan noise.

Pointwise noise calibration and generation. Pointwise noise $N _ { \mathrm { p o i n t w i s e } }$ refers to noise that is independent between individual pixels, originating from the detector’s signal processing system [22], [23], such as dark current and readout noise. Related studies on the calibration of noise in natural images [22], [23] have confirmed that Tukey’s lambda distribution with a shape parameter provides a more accurate estimation of pointwise noise, i.e.,

$$
N _ { \mathrm { p o i n t w i s e } } \sim \mathrm { T L } ( 0 , \sigma _ { \mathrm { p o i n t } } , \lambda _ { \mathrm { p o i n t } } ) ,\tag{6}
$$

where $\sigma _ { \mathrm { p o i n t } }$ is the standard deviation and $\lambda _ { \mathrm { p o i n t } }$ is the shape parameter. Pointwise noise can be obtained by calculating the difference between the original image and the row-wise blurred image, as shown in Fig. 5(b). Fig. 5(c) illustrates the variation of two parameters with pixel values. The standard deviation is proportional to the pixel value, while the shape parameter is a small positive constant number. Therefore, the slope and intercept of the standard deviation $\sigma _ { \mathrm { p o i n t } } ,$ along with the constant shape parameter $\lambda _ { \mathrm { p o i n t } }$ , can be calculated from real images and used for the simulation of pointwise noise.

Our calibration framework is mode-agnostic and applicable to common STEM modes $( e . g .$ , HAADF, BF). While scan noise, pointwise noise, and background share consistent physical origins across modes, the contrast inversion between HAADF (bright atoms) and BF (dark atoms) requires simple sign adjustments in parameter extraction (Eqs. (2)(3)(4)(5)). For other imaging modes, the same calibration pipeline can be directly applied by fitting mode-specific parameters from rep resentative real images. The SNR of our synthesized AtoMix dataset ranges from −5.09 dB to 10.58 dB across BF and HAADF modes, where negative values reflect the extremely low-dose conditions targeted by our work (i.e., noise power exceeding signal power in beam-sensitive imaging), ensuring the practical relevance of our dataset for training enhancement models under real-world low-SNR STEM scenarios.

## IV. EXPERIMENTS

In this section, we first introduce the experimental setup. Then, we compare the enhancement results of different methods. Finally, we analyze in detail the effectiveness of the proposed spatial-frequency learning and our AtoMix dataset.

## A. Experimental Settings

We first provide the datasets and metrics. Then, we introduce the comparison methods. Finally, we provide implementation details of our spatial-frequency fusion network.

Datasets and metrics. We validate the effectiveness of our method on both synthetic datasets and real-world data. The synthetic datasets include TEMImageNet [12], SFIN dataset [17], and our AtoMix. The real-world data comprises publicly available images of four types of advanced materials (i.e., superconducting material [3], all-solid-state battery [4], nanodevice [10], and neuromorphic computing material [9]), as well as STEM images of high-entropy alloys captured in our laboratory. Following SFIN [17], we use Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index Measure (SSIM) [70] to evaluate enhancement performance.

Comparison methods. Our Spatial-Frequency Fusion Network (SFFN) is compared with two traditional methods $( i . e . ,$ , Gaussian filter and Wiener filter [18]), ten supervised deep learning methods (i.e., AtomSegNet [12], FCN [13], Restormer [66], SwinIR [67], AST [14], GAN [15], FS-Net [60], FADC [16], FDConv [61], SFIN [17]), and two unsupervised deep learning methods (i.e., UDVD [20] and ZeroIG [58]). AtomSegNet, FCN, Restormer, SwinIR, AST, and GAN are spatial domain networks, while FSNet, FADC, FDConv, SFIN, and our SFFN are spatial and frequency domain networks. Due to the requirement for multi-frame input images, UDVD and ZeroIG are only compared on the multi-frame real-world data captured in our laboratory.

Implementation details of our network. Our SFFN is trained on AtoMix for 500 epochs using the Adam optimizer [71] $( \beta _ { 1 }$ = 0.9 and $\beta _ { 2 } = 0 . 9 9 9 )$ and an $L _ { 1 }$ loss. For each of the HAADF and BF imaging modes, AtoMix contains 1,000 groups of training images and 100 groups of test images. More data can be easily synthesized if needed. The code is implemented in PyTorch [68]. The initial learning rate is set to $2 \times 1 0 ^ { - 4 }$ and reduced by half at epochs 250, 400, 450, and 475, following SFIN [17]. The mini-batch size is set to 8, allowing training to be completed on a single NVIDIA 3090 GPU.

## B. Main Results

We evaluate the effectiveness of our spatial-frequency fusion network on both synthetic datasets and real data. Specifically, the results on synthetic datasets focus on validating the effectiveness of our network architecture under unified training/evaluation protocols. The results on real data further compare the denoising effects of different combinations of network architectures and training datasets.

Results on synthetic datasets. We compare our SFFN with existing methods on TEMImageNet [12], SFIN dataset [17],

![](images/d1eae53f6e388920338cc67e5659c2919be3e629a81d9e700329438fcd73a3b2.jpg)  
Fig. 6. Enhancement results on different synthetic datasets. The training data used corresponds to the training set associated with the test data.

![](images/5ac95de808067b0a29efc3660659e495b9db23370674d2428ba5b1c1910c21c6.jpg)  
Fig. 7. Results on a real STEM image sequence. The results on the right compare different networks trained on different datasets.

TABLE II  
ENHANCEMENT RESULTS ON DIFFERENT SYNTHETIC DATASETS. DEEP LEARNING METHODS ARE TRAINED AND TESTED ON THE CORRESPONDING DATASETS. HAADF AND BF ARE IMAGING MODES. BEST AND SECOND BEST VALUES ARE BOLD AND UNDERLINED, RESPECTIVELY.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Category</td><td colspan="2">TEMImageNet [12]</td><td colspan="4">SFIN Dataset [17]</td><td colspan="4">AtoMix (Ours)</td></tr><tr><td>HAADF Mode</td><td></td><td>HAADF Mode</td><td></td><td></td><td>BF Mode</td><td></td><td>HAADF Mode</td><td></td><td>BF Mode</td></tr><tr><td></td><td></td><td>PSNR ↑</td><td>SSIM ↑</td><td>PSNR ↑</td><td>SSIM ↑</td><td></td><td>PSNR ↑ SSIM ↑</td><td>PSNR ↑</td><td>SSIM ↑</td><td>PSNR ↑</td><td>SSIM ↑</td></tr><tr><td>Gaussian Filter [19] Wiener Filter [18]</td><td>Traditional</td><td>16.78 16.08</td><td>0.4961 0.4991</td><td>16.33 16.43</td><td>0.3303 0.3267</td><td>10.30 10.23</td><td>0.6867 0.8081</td><td>10.57 10.59</td><td>0.1053 0.0803</td><td>10.17 10.01</td><td>0.5291 0.4405</td></tr><tr><td>AtomSegNet [12]</td><td></td><td>22.34</td><td>0.8008</td><td>34.33</td><td>0.8994</td><td>30.26</td><td>0.9673</td><td>27.79</td><td>0.8240</td><td>28.70</td><td>0.8934</td></tr><tr><td>FCN [13]</td><td></td><td>26.18</td><td>0.8842</td><td>32.82</td><td>0.8830</td><td>28.77</td><td>0.9623</td><td>26.38</td><td>0.7711</td><td>27.67</td><td>0.8765</td></tr><tr><td>Restormer [66]</td><td>Spatial</td><td>28.14</td><td>0.9113</td><td>34.82</td><td>0.9099</td><td>30.76</td><td>0.9745</td><td>28.28</td><td>0.8373</td><td>29.06</td><td>0.8945</td></tr><tr><td>SwinIR [67]</td><td>Network</td><td>28.33</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>AST [14]</td><td></td><td></td><td>0.9066</td><td>35.00</td><td>0.9085</td><td>30.81</td><td>0.9696</td><td>28.39</td><td>0.8387</td><td>29.18</td><td>0.8984</td></tr><tr><td></td><td></td><td>28.61</td><td>0.9071</td><td>35.24</td><td>0.9076</td><td>31.05</td><td>0.9721</td><td>28.64</td><td>0.8431</td><td>29.27</td><td>0.9025</td></tr><tr><td>GAN [15]</td><td></td><td>30.28</td><td>0.9311</td><td>36.97</td><td>0.9347</td><td>32.80</td><td>0.9759</td><td>30.22</td><td>0.868</td><td>30.36</td><td>0.9197</td></tr><tr><td>FSNet [60]</td><td></td><td>28.11</td><td>0.9056</td><td>34.80</td><td>0.9067</td><td>30.83</td><td>0.9686</td><td>28.29</td><td>0.8305</td><td>29.09</td><td>0.8991</td></tr><tr><td>FADC [16]</td><td>Spatial and Frequency</td><td>29.49</td><td>0.9269</td><td>36.19</td><td>0.9276</td><td>32.10</td><td>0.9808</td><td>29.56</td><td>0.8573</td><td>30.03</td><td>0.9202</td></tr><tr><td>FDConv [61]</td><td rowspan="3">Network</td><td>28.86</td><td>0.9160</td><td>35.58</td><td>0.9153</td><td>31.51</td><td>0.9702</td><td>28.94</td><td>0.8484</td><td>29.55</td><td>0.9048</td></tr><tr><td>SFIN [17]</td><td>31.76</td><td>0.9543</td><td>38.48</td><td>0.9582</td><td>33.24</td><td>0.9885</td><td>31.95</td><td>0.9028</td><td>30.77</td><td>0.9333</td></tr><tr><td>SFFN (Ours)</td><td>32.83</td><td>0.9649</td><td>39.53</td><td>0.9666</td><td>35.42</td><td>0.9930</td><td>32.74</td><td>0.9214</td><td>32.32</td><td>0.9540</td></tr></table>

## TABLE III

ABLATION RESULTS OF OUR NETWORK. MODELS ARE TRAINED AND TESTED ON OUR ATOMIX DATASET FOR HAADF MODE IMAGE   
ENHANCEMENT. VARIANTS V1 TO V5 FEATURE DIFFERENT NETWORK STRUCTURES, INCLUDING WHETHER SPATIAL-DOMAIN AND   
FREQUENCY-DOMAIN CONVOLUTIONS ARE USED, HOW INFORMATION   
INTERACTS BETWEEN THE TWO DOMAINS, AND WHETHER COORDINATE   
INFORMATION IS EMPLOYED IN THE COORDINATE-GUIDED FREQUENCY CONVOLUTIONS. BEST AND SECOND BEST VALUES ARE BOLD AND UNDERLINED, RESPECTIVELY.

<table><tr><td>Method</td><td>Spatial</td><td>Frequency</td><td>Interaction</td><td>Coordinates</td><td>PSNR ↑</td><td>SSIM ↑</td></tr><tr><td>SFFN-V1</td><td></td><td>X</td><td>X</td><td>一</td><td>27.80</td><td>0.8605</td></tr><tr><td>SFFN-V2</td><td>X</td><td>√</td><td>X</td><td>√</td><td>29.40</td><td>0.8880</td></tr><tr><td>SFFN-V3</td><td>√</td><td>√</td><td>X</td><td>√</td><td>31.60</td><td>0.9060</td></tr><tr><td>SFFN-V4</td><td>√</td><td>√</td><td>Addition</td><td>√</td><td>31.73</td><td>0.9072</td></tr><tr><td>SFFN-V5</td><td>√</td><td>√</td><td>Fusion</td><td>X</td><td>32.49</td><td>0.9193</td></tr><tr><td>SFFN (Ours)</td><td>√</td><td></td><td>Fusion</td><td>√</td><td>32.74</td><td>0.9214</td></tr></table>

and our AtoMix dataset. Fig. 6 shows the visual comparisons. The first row presents the results on TEMImageNet. The outputs of Gaussian filter and Wiener filter [18] are blurred, while distinct artifacts are observed in the upper-left regions of the results generated by AtomSegNet [12], FCN [13], Restormer [66], SwinIR [67], and AST [14]. In contrast, GAN [15], FSNet [60], FADC [16], FDConv [61], SFIN [17], and our SFFN achieve clear enhancement results, among which the brightness distribution of our SFFN’s output is most consistent with the ground truth. The second row presents the results on the SFIN dataset. Only SFIN and our SFFN can effectively distinguish between two closely spaced atoms, while other methods tend to incorrectly merge them into one. The bottom three rows present the results on our AtoMix dataset. Only our SFFN recovers all atoms, including dark and small ones.

Table II shows the quantitative results. It can be observed that the spatial and frequency-domain networks achieve better performance compared to pure spatial-domain networks, as they can exploit the periodicity of atomic arrangements in STEM images. Among the spatial and frequency-domain networks, our SFFN achieves the best performance across different datasets, with an approximate increase of 1 to 2 dB in PSNR compared to the second-best method SFIN [17]. This demonstrates the effectiveness of fusing information from the two domains.

![](images/54a3581b097eb3b06fb0c8d87fba0e1e7cfa09d95d6c181516ec5548fd5dcc94.jpg)  
Fig. 8. Comparison of different methods on real data.

Results on real data. We collect a sequence of HAADF-mode images of a high-entropy alloy to comprehensively compare all methods, including traditional methods, supervised deep learning methods, and unsupervised deep learning methods.

The visual results are shown in Fig. 7. On the left part of Fig. 7, it can be seen that the atomic shapes in the enhanced results of Gaussian and Wiener filters [18], UDVD [20], and ZeroIG [58] are distorted (i.e., not a perfect circle). Meanwhile, the atomic sizes are relatively large, which is not conducive to analyzing atomic positions. This is because traditional and unsupervised methods are not trained with direct ground-truth constraints; although they can remove noise, they struggle to eliminate spherical aberration and background interference, as well as recover the shapes and sizes of atoms.

On the right part of Fig. 7 are the results of different supervised methods trained on different datasets. The atoms in the results of the last row are significantly clearer than those in the first two rows, indicating that our AtoMix dataset achieves better training effectiveness. In the last row, the atom located in the upper region exhibits artifacts or leftward positional deviations in the results of AtomSegNet [12], FCN [13], Restormer [66], SwinIR [67], AST [14], FSNet [60], and FADC [16]. These errors stem from the misclassification of noise as atomic contours. For GAN [15], FDConv [61], and SFIN [17], the upper atom is correctly recovered, yet a small artifact appears in the upper-left corner of the lower atom. In contrast, the result of our SFFN is free from artifacts, and the positions of all atoms are consistent with the material structure.

![](images/9111f0f9957ff07756f219fb7eb7538e0b352226fc85a5c1e31890015e4da032.jpg)  
(c) Clean Image  
(d) Features after Coordinate-Guided Frequency Counvolution  
Fig. 9. Features before and after coordinate-guided frequency convolution. The features shown in (b) and (d) come from the sixth coordinate-guided frequency convolution in our spatial-frequency fusion network. The periodic signals of atoms are significantly enhanced.  
TABLE IV

STEM IMAGE ENHANCEMENT RESULTS OF DIFFERENT FREQUENCY-DOMAIN PROCESSING NETWORKS. MODELS ARE TRAINED AND TESTED ON OUR DATASET (HAADF MODE). BEST AND SECOND BEST VALUES ARE BOLD AND UNDERLINED, RESPECTIVELY. OUR SFFN ACHIEVES THE BEST PERFORMANCE WHILE TYING SFIN FOR THE FEWEST PARAMETERS.
<table><tr><td>Method</td><td>Frequency- position guidance</td><td>Spatial- frequency interaction</td><td>Per-frequency- location processing</td><td>FLOPs Params [G]</td><td>[M]</td><td>PSNR ↑ SSIM ↑</td></tr><tr><td>DeepRFT [59]</td><td>X</td><td>Fixed (Addition)</td><td>√</td><td>45.7</td><td>9.5</td><td>30.62 0.8833</td></tr><tr><td>FSNet [60]</td><td>X</td><td>X</td><td>X</td><td>59.6</td><td>7</td><td>28.29 0.8305</td></tr><tr><td>FADC [16]</td><td>x</td><td>X</td><td>x</td><td>62.5</td><td>1.2</td><td>29.56 0.8573</td></tr><tr><td>FDConv [61]</td><td>X</td><td>x</td><td>X</td><td>59.4</td><td>1.1</td><td>28.94 0.8484</td></tr><tr><td>SFIN [17]</td><td>X</td><td>Fixed (Addition)</td><td>√</td><td>60.9</td><td>0.9</td><td>31.95 0.9028</td></tr><tr><td>SFFN (Ours)</td><td>√</td><td>Learnable</td><td>√</td><td>60.9</td><td>0.9</td><td>32.74 0.9214</td></tr></table>

Fig. 8 presents more complex real-world scenarios to compare different supervised learning methods. In Fig. 8(a), most methods fail to detect theoretically present atoms (e.g., at the red arrows). Although FADC recovers these atoms, it introduces significant shape distortions. Only our SFFN produces a complete and distortion-free image. In Fig. 8(b), only our SFFN yields undistorted atomic images. AtomSegNet, Restormer, and AST inaccurately recover bright atoms, while other competing methods exhibit noticeable distortions in dark atoms.

## C. Discussion

We first discuss the individual effectiveness of the core components of our SFFN, including coordinate-guided frequency convolution, frequency-coordinate embedding, and cross-domain feature fusion. Then, we discuss the effectiveness of our AtoMix dataset.

Effectiveness of our coordinate-guided frequency convolution. We use coordinate-guided frequency convolution in each spatial-frequency fusion block to extract and enhance frequency-domain features. We first visualize its effect on enhancing periodic features and then compare it with other frequency-domain networks or modules.

![](images/4d768e458a23a8173f8898510632e073a05dc49f327bd9392223b1447b331ffe.jpg)  
SFFN (Ours)  
Fig. 10. Comparison of frequency processing networks on AtoMix dataset.

Fig. 9 visualizes the enhancement effect of our coordinateguided frequency convolution on the periodic information in STEM images. Fig. 9(b) shows the feature map before enhancement, where the image signals are masked by noise, making it impossible to distinguish different atoms. After enhancement, each feature map in Fig. 9(d) exhibits the same periodic variation as the real atomic arrangement in Fig. 9(c). It indicates that our coordinate-guided frequency convolution can enhance the periodic information of atomic arrangements in the image, thereby helping to recover the atomic contours from the noise. Furthermore, the features in Fig. 9(d) exhibit richer colors (such as blue and green) compared to those in Fig. 9(b). This indicates that the values of the features have been equalized to distinguish different parts of each atom. Although Fig. 9(d) exhibits clearer atomic periodicity, periodic responses also appear in the background regions. This indicates that frequency-domain processing alone cannot distinguish different spatial locations, motivating the fusion of frequency-domain and spatial-domain features. Table III shows that, compared with the spatial-only variant SFFN-V1, introducing the coordinate-guided frequency branch in SFFN-V3 increases the PSNR from 27.80 dB to 31.60 dB, demonstrating the benefit of frequency-domain processing.

Table IV compares our coordinate-guided frequency convolution with other state-of-the-art frequency-domain networks or modules, including DeepRFT [59], FSNet [60], FADC [16], and FDConv [61]. Although these methods incorporate frequency-domain analysis, they fundamentally differ from our approach in three key aspects. First, instead of using frequency analysis merely to guide spatial convolutions (e.g., FADC) or performing coarse frequency selection (e.g., FSNet and FDConv), our method explicitly transforms features into the

![](images/8956c31cb2b60aa82f8b9b6f2b92c879fe11ca405e7d7f2fd5cf1788645407ab.jpg)  
Fig. 11. Importance of coordinate information in frequency-domain processing. The heat map shows the relative contributions of different input channels to the real and imaginary output channels of the 1×1 convolution in CGFC. Each value is the mean absolute magnitude of the corresponding kernel weights across all 16 CGFC modules in SFFN. The x- and y-coordinate channels contribute substantially to frequency-domain processing.

Fourier domain and applies convolution directly on frequencydomain features. This enables fine-grained modeling of distinct frequency locations, which is crucial for separating and enhancing individual diffraction spots in STEM images. Second, we introduce explicit frequency-position guidance by appending the x/y coordinates of the frequency map to the frequencydomain features. This makes the network explicitly aware of the positional context of each frequency component, enhancing its sensitivity to diffraction spot locations and thereby improving atomic structure recovery under heavy noise. Third, for explicit spatial-frequency feature fusion, rather than omitting cross-domain interaction (e.g., FSNet, FADC, FDConv) or relying on simple addition (e.g., SFIN and DeepRFT), we design parallel spatial and frequency branches fused via concatenation followed by a convolution. This learnable fusion provides greater flexibility than fixed addition and enables the network to combine complementary spatial- and frequency-domain information based on the input features. Consequently, the model can jointly leverage periodic structural priors from the frequency domain and local spatial anomalies, such as atomic vacancies, material interfaces, and embedded ions. As shown in Table IV, our SFFN achieves higher enhancement accuracy with only 0.9M parameters while maintaining computational complexity comparable to existing methods. Fig. 10 shows that our frequency-domain processing method helps restore atomic structures more accurately.

Effectiveness of using frequency coordinates. Fig. 11 shows the relative contributions of different input channels in the coordinate-guided frequency convolution, including the coordinate channels and the real and imaginary feature channels. Both coordinate channels contribute substantially to the real and imaginary outputs, indicating that frequency-position information helps the convolution distinguish different frequency locations and thereby separate signal from noise. The contribution of the $y$ coordinate is slightly larger than that of the x coordinate because the height of the frequency-domain feature map is approximately twice its width (Fig. 3(c)), allowing the y coordinate to encode more detailed positional information. Table III shows that using coordinate information (our SFFN), compared with omitting it (SFFN-V5), increases the PSNR by 0.25 dB.

Effectiveness of our cross-domain feature fusion. Fig. 12 compares the relative contributions of spatial- and frequencydomain features across different spatial-frequency fusion blocks. The two curves have similar magnitudes, indicating that both domains play important roles in STEM image enhancement. The contribution of frequency-domain features is consistently slightly greater than that of spatial-domain features, highlighting the importance of frequency-domain processing for periodic STEM images. In addition, the substantially larger mean absolute kernel weights in the final block indicate stronger cross-domain fusion near the network output.

![](images/330ce055407eded8919b09c6043a53a77489a00c15b477f8bc274f6272a42f8e.jpg)

Fig. 12. Importance of spatial- and frequency-domain features in crossdomain fusion. The curves show their relative contributions in each spatialfrequency fusion block, measured by the mean absolute kernel weights associated with the corresponding input channels in the final 3×3 fusion convolution. Their similar magnitudes indicate that spatial- and frequencydomain features have comparable importance.  
![](images/f0a63cfdadb13bc8263980a31ef15c6de73c5ee9749eb079d0420efd0e6f51f0.jpg)  
Fig. 13. Result of our SFFN using different training datasets. The input image is real data, and the synthetic datasets used for training include TEMImageNet [12], SFIN dataset [17], and our AtoMix dataset. Training with our AtoMix yields the best result.

The ablation results in Table III indicate that compared with the simultaneous use of spatial-domain and frequencydomain information (SFFN-V3), the use of only spatialdomain information (SFFN-V1) or only frequency-domain information (SFFN-V2) both leads to performance degradation. This demonstrates that the information from both domains is important and complementary. Compared with no information interaction between the two domains (SFFN-V3), using addition for cross-domain information interaction (SFFN-V4) can increase the PSNR by 0.13 dB, while using the domain fusion operation in our SFFN can increase the PSNR by 1.14 dB. This indicates that domain fusion can more flexibly utilize information from both spatial and frequency domains to achieve better feature-enhancement results.

Effectiveness of our dataset. Our new dataset AtoMix adopts the noise parameters calibrated in SFIN [17]. Compared with the previous datasets where noise parameters were manually set, such as TEMImageNet [12] and GAN dataset [21], the noise calibration makes the noise distribution in the simulated images closer to the real situation. The effectiveness of our noise calibration and simulation method has been verified in SFIN [17].

Table V compares the differences among various datasets. Unlike the SFIN dataset [17], our AtoMix dataset simulates more complex image degradations, including lower spatial resolution, lower atomic brightness, spherical aberration, and combinations of multiple arrangement patterns. These improvements enable our SFFN to handle more complex atomic arrangements and STEM image degradations.

TABLE V  
COMPARISON OF STEM IMAGE ENHANCEMENT DATASETS. OUR ATOMIX IS MORE DIVERSE AND INCLUDES MORE COMPLEX DEGRADATIONS.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Random Scale</td><td rowspan="2">Random Rotation</td><td rowspan="2">Random Arrangements Arrangements</td><td rowspan="2">Mixed</td><td rowspan="2">No. of Patterns ↑</td><td rowspan="2">No. of Crystal Systems ↑</td><td rowspan="2">BF Mode</td><td rowspan="2">Spherical Aberration</td><td rowspan="2">Minimum</td><td rowspan="2">Minimum Atom Brightness ↓ Diameter (pixels) ↓</td></tr><tr><td></td></tr><tr><td>TEMImageNet [12]</td><td>Discrete</td><td>Discrete</td><td>X</td><td>X</td><td>13</td><td>4</td><td>X</td><td>V</td><td>76</td><td>14</td></tr><tr><td>SFIN Dataset [17]</td><td>Continuous</td><td>Continuous</td><td>√</td><td>x</td><td>15</td><td>5</td><td>√</td><td>x</td><td>119</td><td>15</td></tr><tr><td>AtoMix (Ours)</td><td>Continuous</td><td>Continuous</td><td>√</td><td>√</td><td>46</td><td>7</td><td>√</td><td>√</td><td>70</td><td>7</td></tr></table>

The right part of Fig. 7 demonstrates that the network trained on our AtoMix dataset better recovers atoms in real STEM images. Fig. 13 further compares the results of our SFFN trained on different datasets. Training on AtoMix produces clear results for all atoms. In contrast, the results obtained using the other datasets are blurred and fail to recover relatively dark atoms.

## V. APPLICATION

In this section, we evaluate the practical utility of our method, including its denoising performance on real images of advanced materials and its effectiveness in the downstream atomic localization task.

## A. Enhancement Results on Advanced Materials

We evaluate the practical effectiveness of jointly using our SFFN architecture and AtoMix training dataset for observing advanced materials in real images. Specifically, we apply our Spatial-Frequency Fusion Network (SFFN) to images of superconducting materials, all-solid-state batteries, nanodevices, and neuromorphic computing materials. We compare it with representative STEM image enhancement methods, i.e., Gaussian filtering, Wiener filtering [18], AtomSegNet [12], and SFIN [17].

Superconducting material. The nucleation of binary compounds into superconducting materials can be induced by altering their stoichiometry, and the analysis and control of this process depend on observation at the atomic scale [3]. We apply our enhancement method SFFN to observe the nucleation process from $\mathrm { P d _ { 1 0 } T e _ { 3 } }$ to $\mathrm { P d T e _ { 2 } }$ [3], as shown in Fig. 14(a). To ensure a large field of view, each atom in the image has a diameter of only 7 pixels, which poses significant challenges for image enhancement. The results of Gaussian filtering fail to distinguish different atoms, while those from Wiener filtering [18], AtomSegNet [12], and SFIN [17] cannot even visualize atoms. In contrast, our SFFN clearly reveals the atomic structures of the nucleus and intermediate state, thanks to its more advanced network architecture and more diverse training data. It can be seen that compared with the perfectly regular $\mathrm { P d _ { 1 0 } T e _ { 3 } }$ nuclear structure, the intermediate state structure has atoms that are not completely aligned in the same straight line. These characterizations help to clearly demonstrate the atomic structures required for different superconducting properties.

All-solid-state battery. The chemomechanical failure of layered cathodes in all-solid-state batteries affects the performance of the batteries, and its atomic origin has recently been revealed by STEM images [4]. We use our SFFN to characterize the structure of NMC-811 cathode after longterm contact with $\mathrm { L i _ { 6 } P S _ { 5 } C l }$ [4], as shown in Fig. 14(b). Our results show that only slight cation mixing (showing some darker atoms caused by oxygen loss) occurred at the edge region. This corroborates the observations in the paper [4]. Our SFFN provides a method for evaluating the structure of allsolid-state batteries, which is difficult to achieve through direct observation of noisy images or the use of other enhancement methods [12], [17], [18].

Nanodevice. Precise thermal management represents a critical bottleneck for the efficiency and service life of nanodevices [10]. We apply our SFFN to enhance high-resolution transmission electron microscopy (HRTEM) images of stressed and bent silicon nanoribbons [10], as shown in Fig. 14(c). Unlike point-by-point scanned STEM images, HRTEM images are globally acquired and are therefore free from scan noise. The results demonstrate that our STEM image enhancement method is also applicable to other types of atomic-scale images. Specifically, although the input images exhibit no significant noise, atomic shape distortions arise from phase contrast and angular issues. Gaussian and Wiener filtering fail to remove the distortions, while AtomSegNet’s enhancement results miss some atoms. SFIN can eliminate atomic distortions but remains blurry. Our SFFN clearly presents the central positions of atoms, which facilitates the measurement of distances between atoms and enables analysis of force at the atomic scale and the properties of nanodevices [10].

Neuromorphic computing material. The performance of neuromorphic computing requires an in-depth understanding of ion transport at material interfaces, which is often hindered by noise in STEM images [9]. We apply our enhancement method SFFN to observe structural changes in $\mathrm { S b _ { 4 } S e _ { 6 } }$ during potassium-ion insertion [9], as shown in Fig. 14(d). The cyan arrows show that the atomic pattern units far from and near the interface (corresponding to the first and second rows, respectively) exhibit different arrangements, as indicated by the arrow in the lower-left corner. This difference arises because potassium-ion intercalation induces atomic rearrangement at the edge. Compared with other methods that suffer from spherical aberration, our SFFN removes spherical-aberration effects and produces more distinguishable atomic patterns.

Quantitative analysis. We further quantitatively compare different methods using a real image with a known crystal structure, as shown in Fig. 15. The input is an experimentally acquired image of $\mathrm { P d _ { 1 0 } T e _ { 3 } }$ . Since the crystal structure is known, we obtain the corresponding ground-truth image through atomistic simulation and rendering. We determine the lattice periodicity and phase using line-profile and FFT analyses, and align the lattice arrangement of the ground-truth image with that of the input image. In Fig. 15(a), we compare the normalized brightness profiles extracted by averaging pixel intensities within the same narrow strip region. The line profile of our SFFN is the closest to the ground truth, indicating that our method better preserves the brightness variation and peak positions of atomic columns while suppressing noise and background interference. In Fig. 15(b), we further report the coefficient of determination $( R ^ { 2 } )$ between the intensity profile of each denoising result and that of the ground truth. Our SFFN achieves the highest $R ^ { 2 }$ value of 0.9937, quantitatively demonstrating that its denoising result is most consistent with the simulated crystal structure. Our SFFN also achieves the highest PSNR of 24.32 dB and SSIM of 0.7798. In Fig. 15(c), we evaluate the atomic localization results using IoU. Our SFFN obtains the highest IoU of 0.5098, showing that the enhanced image produced by our method better supports downstream atomic localization.

![](images/4f96d073bbec613a298cf5b8753301c895fea1944725adfae6ed6386d9815a5a.jpg)  
Fig. 14. Enhancement results of real images. (a) Superconducting material [3]; (b) All-solid-state battery [4]; (c) Nanodevice [10]; (d) Neuromorphic computing material [9]. Gaussian and Wiener [18] filters are traditional methods. AtomSegNet [12], SFIN [17], and our SFFN are trained on their corresponding datasets Our SFFN demonstrates better enhancement effects on these advanced materials.

## B. Localization Results

We compare the influence of different enhancement network architectures on downstream atomic localization under the same localization protocol.

After a noisy image is enhanced, atomic positions can be automatically detected to analyze the material’s atomic arrangement [12], [72]. Following the protocols of AtomSeg-Net [12] and SFIN [17], we compare the effects of different enhancement methods on atomic localization. Specifically, we train each enhancement network using data designed for atomic localization, in which atoms of varying sizes and brightness are represented by small white disks so that each atom can be identified clearly. The network output is then binarized into a mask. The geometric center of each connected white region gives an atomic position.

![](images/c8432bfab05ecadfd1d7ea97d3ff7b55faeaddbb365cb26484eeece053b8f230.jpg)

TABLE VI  
LOCALIZATION RESULTS (IOU) ON DIFFERENT SYNTHETIC DATASETS. DEEP LEARNING METHODS ARE TRAINED AND TESTED ON THE CORRESPONDING DATASETS. HAADF AND BF ARE IMAGING MODES.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Category</td><td>TEMImageNet [12]</td><td>SFIN Dataset [17]</td><td colspan="2">AtoMix (Ours)</td></tr><tr><td>HAADF Mode</td><td>HAADF Mode BF Mode</td><td></td><td>HAADF Mode BF Mode</td></tr><tr><td>Gaussian Filter [19]</td><td>Traditional</td><td>0.1251 0.1144</td><td>0.0965</td><td>0.0443</td><td>0.1655 0.1804</td></tr><tr><td>Wiener Filter [18] AtomSegNet [12]</td><td rowspan="6">Spatial</td><td>0.5636</td><td>0.0896 0.4507</td><td>0.0439 0.3581</td><td>0.1238 0.1397 0.3601 0.3902</td></tr><tr><td>FCN [13]</td><td>0.6061</td><td>0.4822 0.4096</td><td>0.2880</td><td>0.3451</td></tr><tr><td>Restormer [66]</td><td>0.6650</td><td>0.5206 0.4513</td><td>0.4104</td><td>0.4001</td></tr><tr><td>SwinIR [67] Network</td><td>0.6523</td><td>0.5150 0.4411</td><td>0.4084</td><td>0.4120</td></tr><tr><td>AST [14]</td><td>0.6462</td><td>0.5035 0.4369</td><td>0.3325</td><td>0.3664</td></tr><tr><td>GAN [15]</td><td>0.7089</td><td>0.5496 0.4481</td><td>0.4208</td><td>0.4778</td></tr><tr><td>FSNet [60]</td><td rowspan="4">Spatial and Frequency Network</td><td>0.6511</td><td>0.5187</td><td>0.4329</td><td>0.4671 0.4147</td></tr><tr><td>FADC [16]</td><td>0.6922</td><td>0.5428</td><td>0.4701 0.4398</td><td>0.4846</td></tr><tr><td>FDConv [61]</td><td>0.6766</td><td>0.5271 0.4377</td><td>0.4239</td><td>0.4382</td></tr><tr><td>SFIN [17]</td><td>0.5987</td><td>0.5694</td><td>0.4809</td><td>0.5236</td></tr><tr><td>SFFN (Ours)</td><td></td><td>0.7765</td><td>0.5950</td><td>0.4814 0.5224</td><td>0.5586 0.5911</td></tr></table>

![](images/602179a954e1e49f98dc97986a2e403308ee20c4f1cf3e0c34ff067e4fce59c5.jpg)  
Fig. 15. Quantitative analysis on a real image of Pd<sub>10</sub>Te<sub>3</sub>. GT is obtained through simulation and rendering of the crystal structure. (a) Line profile extracted from a narrow strip region. Our denoising results exhibit the closest brightness variation curve to the ground truth. (b) Denoising results. (c) Atomic localization results.

Fig. 16 presents the atomic localization results of different methods on various datasets. Gaussian filter and Wiener filter [18] fail to correctly identify most atoms. Among existing deep learning-based methods, SFIN [17] achieves relatively satisfactory atomic localization performance but still suffers from missed detections (e.g., its results on TEMImageNet [12]) and false detections (e.g., its results on the SFIN dataset [17] and our AtoMix). Our SFFN fuses and leverages both frequency-domain and spatial-domain features, which facilitates distinguishing weak signals from noise, thereby yielding results consistent with the ground truth. Table VI also demonstrates that our SFFN achieves the best Intersection over Union (IoU) results on the three atomic localization datasets.

Fig. 17 shows the atomic localization results of traditional methods, unsupervised learning methods, and supervised learning methods on a sequence of real STEM images, where the material is a high-entropy alloy. SFIN [17] and our SFFN achieve the most complete localization results, which are consistent with the theoretical structure of the material.

## VI. CONCLUSION

Noise and aberrations in STEM images hinder in situ atomic-scale observation of new-material structures. We present a spatial-frequency fusion network and develop a realistic and diverse AtoMix dataset for STEM image enhancement. Our spatial-frequency fusion network exploits frequency-domain information from atomic periodicity and enhances and fuses spatial- and frequency-domain features. Specifically, the coordinate-guided frequency convolution processes features according to their frequency positions, thereby enhancing periodic signals. Furthermore, we develop a STEM noise calibration method that derives noise parameters from real images to synthesize realistic STEM images. AtoMix contains diverse atomic arrangements, imaging modes, and degradations and uses masks to simulate multimaterial interfaces. This diversity improves robustness in real-world STEM image applications. Experiments demonstrate that our method achieves state-of-the-art performance on both synthetic datasets and real images of advanced materials, supporting STEM-based materials research.

The experimental results demonstrate the significant enhancement effect of frequency-domain processing on STEM images. Since atomic periodic signals and noise have distinct distributions in the frequency domain, future research can combine frequency-domain processing with attention mechanisms, dynamic convolutions, etc., enabling the network to distinguish and treat effective information and noise differently in the frequency domain. In addition, it is worthwhile to extend our method to higher-dimensional data, such as 4D-STEM data, which provide a two-dimensional diffraction pattern at each two-dimensional probe position and are important for analyzing local structures, strain, and electromagnetic fields.

## REFERENCES

[1] G. Sánchez-Santolino, V. Rouco, S. Puebla, H. Aramberri, V. Zamora, M. Cabero, F. A. Cuellar, C. Munuera, F. Mompean, M. Garcia-Hernandez, A. Castellanos-Gomez, J. Íñiguez, C. Leon, and J. Santa-

Input from our AtoMix

![](images/10ab8b708052b748851880d06f1181405c3de3268fa267c5de25174c2eff8050.jpg)  
Fig. 16. Localization results on different synthetic datasets. The training data used corresponds to the training set associated with the test data.

![](images/714dd3214fc86da24a676b4434a0723a9c140228b78257ff5502ed3f8a113e70.jpg)  
Fig. 17. Localization results on real data. All the supervised deep-learning methods are trained on our AtoMix.

maria, “A 2d ferroelectric vortex pattern in twisted batio3 freestanding layers,” Nature, vol. 626, no. 7999, pp. 529–534, 2024.

[2] Y.-J. Wang, Y.-P. Feng, Y.-L. Tang, Y.-L. Zhu, Y. Cao, M.-J. Zou, W.-R. Geng, and X.-L. Ma, “Polar bloch points in strained ferroelectric films,” Nat. Commun., vol. 15, no. 1, p. 3949, 2024.

[3] M. Huang, Z. Hua, R. Guzman, Z. Ren, P. Gu, S. Yang, H. Chen, D. Zhang, Y. Ding, Y. Ye et al., “Stoichiometry-engineered phase transition in a two-dimensional binary compound,” Nat. Commun., vol. 16, no. 1, p. 4162, 2025.

[4] C. Wang, Y. Jing, D. Zhu, and H. L. Xin, “Atomic origin of chemomechanical failure of layered cathodes in all-solid-state batteries,” J. Am. Chem. Soc., vol. 146, no. 26, pp. 17 712–17 718, 2024.

[5] C. Wang, X. Wang, R. Zhang, T. Lei, K. Kisslinger, and H. L. Xin, “Resolving complex intralayer transition motifs in high-ni-content layered cathode materials for lithium-ion batteries,” Nat. Mater., vol. 22, no. 2, pp. 235–241, 2023.

[6] R. Zhang, C. Wang, P. Zou, R. Lin, L. Ma, T. Li, I.-h. Hwang, W. Xu, C. Sun, S. Trask, and H. L. Xin, “Long-life lithium-ion batteries realized by low-ni, co-free cathode chemistry,” Nat. Energy, vol. 8, no. 7, pp. 695–702, 2023.

[7] C. Yang, B. Zhang, L. Fu, Z. Wang, J. Teng, R. Shao, Z. Wu, X. Chang, J. Ding, L. Wang, and X. Han, “Chemical inhomogeneity–induced profuse nanotwinning and phase transformation in aucu nanowires,” Nat.

Commun., vol. 14, no. 1, p. 5705, 2023.

[8] X. Zhou, P. Min, Y. Liu, M. Jin, Z.-Z. Yu, and H.-B. Zhang, “Insulating electromagnetic-shielding silicone compound enables direct potting electronics,” Science, vol. 385, no. 6714, pp. 1205–1210, 2024.

[9] W. Dong, Y.-C. Wang, C. Yang, C. Sun, H. Li, Z. Hua, Z. Wu, X. Chang, L. Bao, S. Qu et al., “Deep learning enhanced in situ atomic imaging of ion migration at crystalline–amorphous interfaces,” Nano Lett., vol. 24, no. 45, pp. 14 445–14 452, 2024.

[10] L. Yang, S. Yue, Y. Tao, S. Qiao, H. Li, Z. Dai, B. Song, Y. Chen, J. Du, D. Li et al., “Suppressed thermal transport in silicon nanoribbons by inhomogeneous strain,” Nature, vol. 629, no. 8014, pp. 1021–1026, 2024.

[11] J. Madsen and T. Susi, “The abTEM code: transmission electron microscopy from first principles,” Open Res. Eur., vol. 1, p. 24, 2021.

[12] R. Lin, R. Zhang, C. Wang, X.-Q. Yang, and H. L. Xin, “Temimagenet training library and atomsegnet deep-learning models for high-precision atom segmentation, localization, denoising, and deblurring of atomicresolution images,” Sci. Rep., vol. 11, no. 1, p. 5386, 2021.

[13] C.-H. Lee, A. Khan, D. Luo, T. P. Santos, C. Shi, B. E. Janicek, S. Kang, W. Zhu, N. A. Sobh, A. Schleife et al., “Deep learning enabled strain mapping of single-atom defects in two-dimensional transition metal dichalcogenides with sub-picometer precision,” Nano Lett., vol. 20, no. 5, pp. 3369–3377, 2020.

[14] S. Zhou, D. Chen, J. Pan, J. Shi, and J. Yang, “Adapt or perish: Adaptive sparse transformer with attentive feature refinement for image restoration,” in Proc. IEEE Conf. Comput. Vis. Pattern Recog., 2024, pp. 2952–2963.

[15] B. Kazimi, K. Ruzaeva, and S. Sandfeld, “Self-supervised learning with generative adversarial networks for electron microscopy,” in Proc. IEEE Conf. Comput. Vis. Pattern Recog. Worksh., 2024, pp. 71–81.

[16] L. Chen, L. Gu, D. Zheng, and Y. Fu, “Frequency-adaptive dilated convolution for semantic segmentation,” in Proc. IEEE Conf. Comput. Vis. Pattern Recog., 2024, pp. 3414–3425.

[17] H. Li, Z. Wu, R. Shao, T. Zhang, and Y. Fu, “Noise calibration and spatial-frequency interactive network for STEM image enhancement,” in Proc. IEEE Conf. Comput. Vis. Pattern Recog., June 2025, pp. 21 287– 21 296.

[18] N. Wiener, Extrapolation, interpolation, and smoothing of stationary time series: with engineering applications. The MIT press, 1949.

[19] C. Tomasi and R. Manduchi, “Bilateral filtering for gray and color images,” in Proc. Int. Conf. Comput. Vis., 1998, pp. 839–846.

[20] P. A. Crozier, M. Leibovich, P. Haluai, M. Tan, A. M. Thomas, J. Vincent, S. Mohan, A. Marcos Morales, S. A. Kulkarni, D. S. Matteson et al., “Visualizing nanoparticle surface dynamics and instabilities enabled by deep denoising,” Science, vol. 387, no. 6737, pp. 949–954, 2025.

[21] A. Khan, C.-H. Lee, P. Y. Huang, and B. K. Clark, “Leveraging generative adversarial networks to create realistic scanning transmission electron microscopy images,” npj Comput. Mater., vol. 9, no. 1, p. 85, 2023.

[22] K. Wei, Y. Fu, J. Yang, and H. Huang, “A physics-based noise formation model for extreme low-light raw denoising,” in Proc. IEEE Conf. Comput. Vis. Pattern Recog., 2020, pp. 2758–2767.

[23] K. Wei, Y. Fu, Y. Zheng, and J. Yang, “Physics-based noise modeling for extreme low-light photography,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 44, no. 11, pp. 8520–8537, 2021.

[24] S. Pennycook, D. Jesson, M. Chisholm, N. Browning, A. McGibbon, and M. McGibbon, “Z-contrast imaging in the scanning transmission electron microscope,” Microsc. Microanal., vol. 1, no. 6, pp. 231–251, 2003.

[25] S. Pennycook and D. Jesson, “High-resolution z-contrast imaging of crystals,” Ultramicrosc., vol. 37, no. 1, pp. 14–38, 1991.

[26] B. Fultz and J. Howe, Diffraction Contrast in TEM Images. Springer Berlin Heidelberg, 2013, ch. Transmission Electron Microscopy and Diffractometry of Materials, pp. 349–427.

[27] H. Li and Y. Fu, “FCDFusion: A fast, low color deviation method for fusing visible and infrared image pairs,” Comput. Vis. Media, vol. 11, no. 1, pp. 195–211, 2025.

[28] H. Li, F. Dai, Q. Zhao, Y. Ma, J. Cao, and Y. Zhang, “Non-uniform compressive sensing imaging based on image saliency,” Chin. J. Electron., vol. 32, no. 1, pp. 159–165, 2023.

[29] L. Yanshan, C. Shifu, L. Wenhan, Z. Li, and X. Weixin, “Hyperspectral image super-resolution based on spatial-spectral feature extraction network,” Chin. J. Electron., vol. 32, no. 3, pp. 415–428, 2023.

[30] Y. Tian, Y. Fu, and J. Zhang, “Transformer-based under-sampled singlepixel imaging,” Chin. J. Electron., vol. 32, no. 5, pp. 1151–1159, 2023.

[31] T. Zhang, Y. Fu, J. Zhang, and C. Yan, “Deep guided attention network for joint denoising and demosaicing in real image,” Chin. J. Electron., vol. 33, no. 1, pp. 303–312, 2024.

[32] L. Chen, Y. Fu, J. Shangguan, J. Xu, and Y. Peng, “A survey on degraded image segmentation,” Chin. J. Electron., vol. 1, no. 1, pp. 1–20, 2025.

[33] C. He, Y. Shen, C. Fang, F. Xiao, L. Tang, Y. Zhang, W. Zuo, Z. Guo, and X. Li, “Diffusion models in low-level vision: A survey,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 47, no. 6, pp. 4630–4651, 2025.

[34] C. He, K. Li, Y. Zhang, Z. Yang, L. Tang, Y. Zhang, L. Kong, and S. Farsiu, “Segment concealed object with incomplete supervision,” IEEE Trans. Pattern Anal. Mach. Intell., pp. 1–18, 2025.

[35] M. Li, H. Sun, Y. Lei, X. Zhang, Y. Dong, Y. Zhou, Z. Li, and X. Chen, “High-fidelity document stain removal via a large-scale realworld dataset and a memory-augmented transformer,” in IEEE Winter Conf. Appl. Comput. Vis., 2024, pp. 7614–7624.

[36] X. Guo, X. Chen, S. Wang, and C.-M. Pun, “Underwater image restoration through a prior guided hybrid sense approach and extensive benchmark analysis,” IEEE Trans. Circuit Syst. Video Technol., vol. 35, no. 5, pp. 4784–4800, 2025.

[37] J. Bai, Y. Yin, Y. Dong, X. Zhang, C.-M. Pun, and X. Chen, “Lensnet: An end-to-end learning framework for empirical point spread function modeling and lensless imaging reconstruction,” arXiv preprint, 2025, arXiv:2505.01755.

[38] L. Peng, Y. Wang, X. Di, X. Fu, Y. Cao, Z.-J. Zha et al., “Boosting image de-raining via central-surrounding synergistic convolution,” in Proc. AAAI, vol. 39, no. 6, 2025, pp. 6470–6478.

[39] L. Peng, Y. Cao, Y. Sun, and Y. Wang, “Lightweight adaptive feature dedrifting for compressed image classification,” IEEE Trans. Multimedia, vol. 26, pp. 6424–6436, 2024.

[40] X. Di, L. Peng, P. Xia, W. Li, R. Pei, Y. Cao, Y. Wang, and Z.-J. Zha, “Qmambabsr: Burst image super-resolution with query state space model,” in Proc. IEEE Conf. Comput. Vis. Pattern Recog., 2025, pp. 23 080–23 090.

[41] Z. Feng, L. Peng, X. Di, Y. Guo, W. Li, Y. Zhang, R. Pei, Y. Wang, Y. Cao, and Z.-J. Zha, “Pmq-ve: Progressive multi-frame quantization for video enhancement,” arXiv preprint, 2025, arXiv:2505.12266.

[42] Y. Zhang, T. Zhang, J. Nie, and Y. Fu, “Real noise decoupling for hyperspectral image denoising,” in Proc. AAAI, vol. 40, no. 15, 2026, pp. 12 925–12 933.

[43] Y. Liang, Y. Fu, Y. Hu, W. Shao, J. Liu, and D. Zhang, “Flow-anything: Learning real-world optical flow estimation from large-scale single-view images,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 47, no. 10, pp. 8435–8452, 2025.

[44] Y. Liang and Y. Fu, “Relation-guided adversarial learning for data-free knowledge transfer,” Int. J. Comput. Vis., vol. 133, no. 5, pp. 2868–2885, 2025.

[45] T. Zhu, H. Li, and Y. Fu, “Trim-sod: A multi-modal, multi-task, and multi-scale spacecraft optical dataset,” Space Sci. Technol., vol. 5, p. 0299, 2025.

[46] J. Xiao, X. Fu, Y. Zhu, and Z.-J. Zha, “Bayesian window transformer for image restoration,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 48, no. 3, pp. 2431–2444, 2026.

[47] J. Xiao, X. Fu, A. Liu, F. Wu, and Z.-J. Zha, “Image de-raining transformer,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 45, no. 11, pp. 12 978–12 995, 2023.

[48] M. Li and Y. Fu, “Joint self-attention transformer for multispectral and hyperspectral image fusion,” J. Image Graph., vol. 28, no. 12, pp. 3922– 3934, 2023.

[49] Y. Zhang, Z. Lai, T. Zhang, Y. Fu, and C. Zhou, “Unaligned rgb guided hyperspectral image super-resolution with spatial-spectral concordance,” Int. J. Comput. Vis., vol. 133, no. 9, pp. 6590–6610, 2025.

[50] Y. Zhang, T. Zhang, J. Nie, and Y. Fu, “Enhancing unregistered hyperspectral image super-resolution via unmixing-based abundance fusion learning,” arXiv preprint, 2026, arXiv:2603.07918.

[51] C. He, R. Zhang, Z. Chen, B. Yang, C. Fang, Y. Lin, Y. Zhang, F. Xiao, and S. Farsiu, “Unfoldldm: Deep unfolding-based blind image restoration with latent diffusion priors,” arXiv preprint, 2025, arXiv:2511.18152.

[52] Y. Zhang, S. Chen, Y. Tian, Y. Gao, J. Jiang, and Y. Fu, “Superviseassisted multi-modality fusion diffusion model for pet restoration,” Tsinghua Sci. Technol., 2026.

[53] M. Li, Y. Fu, T. Zhang, J. Liu, D. Dou, C. Yan, and Y. Zhang, “Latent diffusion enhanced rectangle transformer for hyperspectral image restoration,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 47, no. 1, pp. 549–564, 2025.

[54] Z. Hua, S. Qu, L. Yan, W. Dong, Y. Zhou, H. Li, X. Chang, L. Bao, Y. Wang, Y. Fu et al., “Deep-learning aided atomic-scale observation of anisotropic melting of the charge density wave in tas2,” Small, vol. 21, no. 45, p. e07496, 2025.

[55] R. Conrad and K. Narayan, “Cem500k, a large-scale heterogeneous unlabeled cellular electron microscopy image dataset for deep learning,” eLife, vol. 10, p. e65894, 2021.

[56] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in Proc. Med. Image Comput. Comput.-Assisted Interv., 2015, pp. 234–241.

[57] J. Wang, H. Li, X. Wang, and Y. Fu, “3d-b2u: Self-supervised fluorescent image sequences denoising,” in Artif. Intell., 2024, pp. 130–142.

[58] Y. Shi, D. Liu, L. Zhang, Y. Tian, X. Xia, and X. Fu, “Zero-ig: zero-shot illumination-guided joint denoising and adaptive enhancement for lowlight images,” in Proc. IEEE Conf. Comput. Vis. Pattern Recog., 2024, pp. 3015–3024.

[59] X. Mao, Y. Liu, W. lei Shen, Q. Li, and Y. Wang, “Deep residual Fourier transformation for single image deblurring,” arXiv preprint, 2021, arXiv:2111.11745.

[60] Y. Cui, W. Ren, X. Cao, and A. Knoll, “Image restoration via frequency selection,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 46, no. 2, pp. 1093–1108, 2024.

[61] L. Chen, L. Gu, L. Li, C. Yan, and Y. Fu, “Frequency dynamic convolution for dense image prediction,” in Proc. IEEE Conf. Comput. Vis. Pattern Recog., 2025, pp. 30 178–30 188.

[62] R. Liu, J. Lehman, P. Molino, F. P. Such, E. Frank, A. Sergeev, and J. Yosinski, “An intriguing failing of convolutional neural networks and the CoordConv solution,” in Proc. Adv. Neural Inform. Process. Syst., vol. 31, 2018.

[63] Q. Hou, D. Zhou, and J. Feng, “Coordinate attention for efficient mobile network design,” in Proc. IEEE Conf. Comput. Vis. Pattern Recog., 2021, pp. 13 713–13 722.

[64] Y. Zou, Y. Fu, Y. Zhang, T. Zhang, C. Yan, and R. Timofte, “Calibrationfree raw image denoising via fine-grained noise estimation,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 47, no. 7, pp. 5368–5384, 2025.

[65] Y. Jiang, Y. Fu, Q. Liu, and J. Zhang, “Msfa image denoising using physics-based noise model and noise-decoupled network,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 48, no. 1, pp. 859–875, 2026.

[66] S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, and M.-H. Yang, “Restormer: Efficient transformer for high-resolution image restoration,” in Proc. IEEE Conf. Comput. Vis. Pattern Recog., 2022, pp. 5728–5739.

[67] J. Liang, J. Cao, G. Sun, K. Zhang, L. Van Gool, and R. Timofte, “Swinir: Image restoration using swin transformer,” in IEEE Int. Conf. Comput. Vis. Worksh., 2021, pp. 1833–1844.

[68] A. Paszke, S. Gross, F. Massa, A. Lerer, and S. Chintala, “PyTorch: An imperative style, high-performance deep learning library,” arXiv preprint, 2019, arXiv:1912.01703.

[69] K. Perlin, “An image synthesizer,” in Proc. Conf. Comput. Graph. Interact. Tech., 1985, pp. 287–296.

[70] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: from error visibility to structural similarity,” IEEE Trans. Image Process., vol. 13, no. 4, pp. 600–612, 2004.

[71] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” arXiv preprint, 2014, arXiv:1412.6980.

[72] M. Ziatdinov, A. Ghosh, C. Y. Wong, and S. V. Kalinin, “Atomai framework for deep learning analysis of image and spectroscopy data in electron and scanning probe microscopy,” Nat. Mach. Intell., vol. 4, no. 12, pp. 1101–1112, 2022.

![](images/2969eacb99769224f9079dab0126b646a7c37592703ffb09e2aadc53e2ded729.jpg)

Ziqi Wu received a B.E. degree in automation from Chang’an University, Xi’an, China, in 2022, and an M.E. degree in mechanical engineering from Beijing Institute of Technology, Beijing, China, in 2025. He is currently a Ph.D. candidate at the School of Medical Technology, Beijing Institute of Technology, Beijing, China. His research interests include oxide cathode materials and intelligent electron microscopy-based precise characterization techniques.

![](images/419543e6a3565eadc2b203a84b5a202e4dc41504104cd52a9224a9b7df4fc2f0.jpg)

Ruiwen Shao received a B.E. degree in mechanical engineering from Tianjin University, Tianjin, China, in 2010, and a Ph.D. degree in engineering from Beijing University of Technology, Beijing, China, in 2016. He worked as a Postdoctoral Fellow at the Electron Microscopy Laboratory, Peking University, from 2016 to 2018, and as an Assistant Researcher at the Technical Institute of Physics and Chemistry, Chinese Academy of Sciences, from 2018 to 2019. He is currently an Associate Professor at the School of Medical Technology, Beijing Institute of

Technology. His research interests include micro-nano robots, micro-nano manipulation, and electron microscopy.

![](images/4a598b8fd9eae3b1b9607ac5a65cc9777f9d3d1c6a613b14b507ae086c7af591.jpg)  
Hesong Li received a B.S. degree in mathematics and applied mathematics from Dalian Maritime University, Dalian, China, in 2021. He is currently a Ph.D. candidate at the School of Computer Science and Technology, Beijing Institute of Technology, Beijing, China. His research interests include artificial intelligence and image processing.

![](images/9f42c7c61d3c0de413bb56113b20febed4527b7daf32aaa040f079ff2afd7370.jpg)

Ying Fu received a B.S. degree in electronic engineering from Xidian University, Xi’an, China, in 2009, an M.S. degree in automation from Tsinghua University, Beijing, China, in 2012, and a Ph.D. degree in information science and technology from the University of Tokyo, Japan, in 2015. She is currently a Professor at the School of Computer Science and Technology, Beijing Institute of Technology. Her research interests include computer vision, image and video processing, and computational photography.

![](images/8dcf5bc80a050270efb80b61b93588246870f0745e5b6f5c38d4644fc7d89773.jpg)

Tao Zhang received B.S. and Ph.D. degrees from the School of Computer Science and Technology, Beijing Institute of Technology, China, in 2017 and 2023, respectively. He is currently an associate professor at Hangzhou Dianzi University. His research interests include intelligent information processing, computational photography, and image processing.