# UnCapsTSR: An Unsupervised Transformer-based Image Super-Resolution Approach for Capsule Endoscopy Images

Anjali Sarvaiya<sup>a</sup>, Shubh Kawa<sup>a</sup>, Lalit Agrawal<sup>a</sup>, Jagrit Joshi<sup>a</sup>, Kishor Upla<sup>a</sup>, Kiran Raja<sup>b</sup>

<sup>a</sup>Department ofElectronics Engineering, Sardar Vallabhbhai National Institute ofTechnology, Surat, India <sup>b</sup>Norwegian University of Science and Technology (NTNU), Norway

## Abstract

Wireless Capsule Endoscopy (WCE) captures and streams video while passing through patient’s Gastrointestinal (GI) tract which is then used to examine its irregularities. While this technology is advantageous over conventional endoscopy, WCE sufers from capsule’s size and wireless transmission that result in images with coarser resolution. Improving the resolution of WCE images in of-line mode (i.e., Super-Resolution (SR)) can help expert physicians to visualize details and help in diagnosis. This work presents a new approach referred as UnCapsTSR to improve the spatial resolution of Low-Resolution (LR) WCE images making use of unsupervised training in transformer-based Generative Adversarial Network (GAN) framework. The novel design of the proposed method accomplishes the SR task without the explicit use of degradation estimation of real-world LR data and eliminates the need for true LR-HR pair. UnCapsTSR employs a new Bilateral Total Variation (BTV) loss to ensure the spatial continuity in the SR images. By combining this loss with other GAN losses, the perceptual and quantitative qualities of the SR results is ensured. In addition, a newly curated dataset from Kvasir capsule set is also presented in this work which is used to train the SR models of WCE images. Consequently, the generalizability of the proposed method is validated by conducting numerous experiments on diferent WCE datasets i.e., KID and GIANA datasets whose samples are not present in the training process. A new non-reference metric called Endoscopy Quality Metric (EndoQM) is also presented to ensure the quantitative performance of SR results of domain specific WCE data. The experimental analysis demonstrates the consistent improvement over the other state-of-the-art unsupervised SR approaches, both in terms of qualitative and quantitative evaluations on diferent no-reference quality measures such as NIQE, BRISQUE and PIQE in addition to EndoQM. Along with statistical evaluation the UnCapsTSR gains 40 to 80 % significant quantitative improvement in terms of EndoQM from LR to SR for all above datasets. Code link:https://github.com/KawaShubh/UnCapsTSR/tree/main

Keywords: Wireless Capsule Endoscopy, Super Resolution, Transformer, Unsupervised, Kvasir capsule dataset

## 1. Introduction

Computerized Tomography (CT), Positron Emission Tomography (PET), Magnetic Resonance Imaging (MRI), ultrasound imaging, endoscopy ad alike imaging approaches provide visual representation of interior parts of human body that are essential for clinical & medical analysis and intervention. Wireless Capsule Endoscopy (WCE) is one such imaging approach that is used to examine Gastrointestinal (GI) abnormalities. This technology is comparatively painless compared to the conventional endoscope based examinations which is often painful and stressful (Karargyris and Bourbakis, 2010). It makes use of a pill size imaging device typically in length and diameter of 26 mm and 11 mm, respectively equipped with an optical dome, illuminator, imaging sensor, battery, and RF transmitter that can be easily ingested by patients and help in diagnosis of GI tract abnormalities (Iddan et al., 2000). Due to the limited size of hardware with limited battery and low field of view of CMOS camera, WCE often hinders acquiring High-Resolution (HR) images/videos. While HR images are invariably preferred in all vision-driven applications, LR images are obtained in WCE with limited high-frequency details making diagnosis challenging (Karargyris and Bourbakis, 2010). Additionally, given the limited battery size of pill camera, WCE captures images with coarser resolution with compression. The motion of pill camera also results in images subjected to diferent artifacts such as motion blur, bubbles, specular reflections, floating objects and pixel saturation which need to be fixed by post-procedural corrections to avoid performance deviation in diagnosis (Swain, 2003). While one can enhance resolution by increasing the size of the optics and sensor array, it is not always an easy solution due to reasonable cost and pill size considerations. To circumvent the aforementioned limitations in other applications, software-driven techniques referred as Super-Resolution (SR) is used for the reconstruction of missing details in the LR observations by posing it as an inverse problem (Wang et al., 2015). Thus, the Medical SR (MedSR) is used to improve visual appearance of anatomical structures to help precise and reliable medical diagnosis.

![](images/e26884bee6beeddb3f5d243d5c4344d6c1f05da49617a7576b60d35f418e8c4c.jpg)  
(a) LR (9.4735)

![](images/93514c0f789f9f113f5666bf8c194c8dd665ec2eef17f4062263d83183754860.jpg)  
(b) SRResCGAN (Umer et al., 2020) (10.0728)

![](images/505b5362121e2b39d0f80f6e2474a16c5038deccfcc497216500d61b07d355d4.jpg)  
(c) DUSGAN (Prajapati et al., 2021) (14.0524)

![](images/179a8f87e4a0a5673ec91e41767c6f9ed70523954601e2550892082a6b0d0278.jpg)  
(d) dSRVAE (Liu et al., 2020) (7.5483)

![](images/62f186b2f8268d13e61810d0d10678ac65d398502c74fa42734122eac46898dc.jpg)  
(e) ZSSR (Shocher et al., 2018) (12.9962)

![](images/3bdae78463bb5a5fca70d071a703140649f462906aaf56f33eeb49bd58bf6c82.jpg)  
(f) DASR (Wang et al., 2021) (14.3810)

![](images/b313167de4b46b7ae2a2551ac664bd62874d00ab8dcd7853c17e10e64d5b6148.jpg)  
(g) MDASR (Liu et al., 2023) (7.8093)

![](images/1311b1a80e638e3a08e8f06d54d1225ce20daf65278d9ba55278febe8c027ce3.jpg)  
(h) Proposed  
(EndoQM↓ = 6.8817)  
Figure 1: The illustration of SR performance of the proposed method (i.e., UnCapsTSR) on WCE LR image in comparison to other state-of-the-art methods. The values of newly proposed EndoQM measure is also indicated alongside of each SR result in the bracket.

Several SR techniques have been proposed for traditional RGB images (Chang et al., 2004;

Smith, 1981; Nuño-Maganda and Arias-Estrada, 2005; Dong et al., 2014; Kim et al., 2016; Zhang et al., 2018). Early methods using CNNs (e.g., SRCNN (Dong et al., 2014)) have been improved with Transformer-based methods (e.g., TTSR (Yang et al., 2020)) that leverage self-attention mechanisms to model long-range dependencies more efectively. Unlike CNNs, which are biased toward local spatial structures, Transformers excel in capturing global context, making them particularly useful for medical imaging. Their scalability, robustness to corruption, and ability to handle large datasets have driven significant interest in Transformer-based models for medical imaging applications.

Despite significant progress in SR algorithms, their application to WCE images remains relatively underexplored. Further, traditional SR approaches for regular RGB images employ supervised training, where models learn to upscale LR images by leveraging paired LR and HR images. The availability of HR endoscopy images is limited, and acquiring well-aligned LR-HR image pairs in the medical domain is particularly challenging. In some cases, synthetic generation of these LR-HR pairs is considered, but this approach is only feasible when the image acquisition process is meticulously controlled and well-defined. Typically, only an approximate understanding of the acquisition process is available, and LR images are often simulated using known downsampling techniques, such as bicubic downsampling. Consequently, supervised SR models are trained on these synthetic LR-HR pairs, where the LR images do not correspond to true camera-captured observations. As a result, such models tend to learn the degradation patterns of simulated LR samples, limiting their ability to generalize to real-world, clinically obtained LR images. The challenge is further exacerbated in medical imaging, where generating true LR-HR pairs by manipulating camera settings is impractical and often infeasible. For these reasons, we propose a deep learning architecture that is trained in an unsupervised manner, eliminating the need for precisely aligned LR-HR image pairs. This unsupervised training approach allows the model to learn directly from the available data without relying on synthetic pairings, thus improving its generalization performance when applied to real-world WCE images. Further, to gain deeper insights into the distributional disparities among diferent modalities such as natural, conventional endoscopy and WCE images we employ t-distributed Stochastic Neighbor Embedding (t-SNE) visualization to examine their feature space in Fig. 1(a) and Kernel Density Estimation (KDE) plots of WCE images and natural images to investigate the pixel intensity distributions in Fig. 1(b) in supplementary material.

This work proposes UnCapsTSR, an unsupervised approach for image super-resolution of WCE images using Transformer without explicit need for true LR-HR pair as it is impossible to obtain such data in WCE. The proposed SR model i.e., UnCapsTSR employs a Generative Adversarial Network (GAN) architecture with a transformer-based generator and dual discriminators. Unlike traditional GAN based SR approaches, the proposed model leverages the power of transformers to capture global and local dependencies efectively, while the dual-discriminator framework ensures enhanced reconstruction quality and realism. The incorporated Bilateral Total Variation (BTV) loss preserves spatial continuity while penalizing inconsistencies. It smoothens SR image across multiple spatially shifted pixel neighborhoods and hence, efectively reduces artifacts enhancing texture fidelity leading to improved perceptual quality in the reconstructed SR images. A sample of SR performance obtained using the proposed method (i.e., UnCapsTSR) is highlighted in Fig. 1 in comparison to other state-of-the-art methods on an LR image of Kvasir dataset along EndoQM score for each SR image. While eliminating the need for explicit LR-HR pair in a true sense of unsupervised setting, this work results in following key contributions:

• Unsupervised training: Unlike the supervised SR methods exhibiting inefective generalizability to the real WCE samples, the proposed model i.e., UnCapsTSR employs transformerbased GAN architecture for SR of WCE images for upscaling factor ×4 which is trained in unsupervised manner, addressing the unavailability of paired LR-HR clinical datasets.

• Transformer-based architecture design: In UnCapsTSR, a novel GAN architecture with a transformer-based generator and a dual-discriminator framework enhances SR for WCE images without explicit degradation estimation. The generator employs self-attention to capture global and local dependencies, while the dual discriminators refine perceptual quality. One discriminator focuses on high-frequency details like edges and textures, preserving diagnostically critical features, while the other ensures structural and perceptual consistency. This balanced approach improves both local and global quality, producing SR outputs that are visually coherent and clinically reliable for endoscopic imaging.

• New BTV Loss: Further, a Bilateral Total Variation (BTV) loss is proposed, integrated with adversarial and texture losses to preserve spatial continuity while improving perceptual quality in SR images. Unlike conventional Total Variation (TV) loss, this loss ensures the spatial smoothness balanced with the retention of edge sharpness, which is critical for maintaining significant details in medical imaging. Thus, by combining BTV loss with adversarial and texture losses, is particularly beneficial in medical applications, where preserving the continuity of anatomical structures alongside detailed textures is essential for accurate diagnosis. Hence, the proposed BTV loss guarantees that the SR outputs not only achieve high visual fidelity but also comply with the rigorous clinical standards.

• New non-reference metric: A reference-less metric Endoscopy Quality Metric (EndoQM) is proposed to gauge the SR performance of the diferent methods on WCE images. This metric is exclusively trained on conventional endoscopic images to assess the preservation and enhancement of edge details essential for diagnostic accuracy in SR images. Unlike other nonreference metrics such as BRISQUE, NIQE, and PIQE, which primarily assess the perceptual quality or statistical characteristics of natural images, this new metric is designed to address the unique medical imaging requirements. Thus, by prioritizing edge preservation, this metric reflects the ability of SR methods to reconstruct fine-grained details such as mucosal patterns, vascular structures, and lesions, which are essential for reliable clinical assessments in WCE images.

• New SR Dataset: An another significant contribution through this work is the creation of a new derivative dataset specifically designed for SR task. It is curated from the publicly available Kvasir Capsule Endoscopy Dataset (Smedsrud et al., 2021). However, the original dataset is not directly suitable for SR tasks due to the presence of redundant and low-quality images, as well as unwanted border pixels that do not contribute meaningful information for training. The original Kvasir dataset is therefore curated through a meticulous pre-processing pipeline. Redundant and non-informative images were identified and removed. Additionally, unwanted border pixels were manually cropped out to improve the overall quality and relevance of the dataset for SR model training. Thus, this newly created derivative dataset provides a unique and valuable resource for advancing SR research in the medical domain, particularly for WCE images and its improved quality and relevance make it well-suited for training and evaluating SR models.

• Generalizability: Finally, the proposed model i.e., UnCapsTSR is trained on the newly derived Kvasir dataset for SR task, and for fair comparison with other state-of-the-art SR methods, we have trained all those methods on the above dataset. Further, to show the generalizability of the proposed method, all SR methods extensively evaluated on other datasets which are not part of training i.e., KID (Koulaouzidis et al., 2017) and GIANA (Bernal and Aymeric, 2017) datasets. It is evident from those experiments that, for real LR samples, the proposed SR method performs better both perceptually and quantitatively than the other SR methods in terms of numerous non-reference metrics, including BRISQUE, NIQE, PIQE, and EndoQM.

## 2. Background and Related Works

## 2.1. Supervised SR approaches

Prior to the emergence of deep learning, Single Image Super-Resolution (SISR) techniques predominantly relied on traditional methods, including interpolation- and reconstruction-based approaches. However, these methods exhibited limitations in handling noise, aliasing, and blurring efects, and were inefective at higher scale factors, often failing to preserve spectral information (Singh and Singh, 2016). Learning-based techniques addressed some of these challenges by utilizing external data to model the relationship between low- and high-resolution images, but their performance remained constrained. The advent of deep learning has revolutionized SISR, enabling the development of advanced methods ranging from early CNNs to state-of-theart GANs and Transformer-based approaches, which achieve significantly superior results by capturing complex image features and long-range dependencies. A comprehensive survey of supervised SR approaches is presented in the Section II in supplementary material.

## 2.2. Unsupervised SR approaches

CNN and GAN based SR models typically follow supervised training where LR sampled are synthesized with known downsampling (i.e., bicubic) where a pair of artificial LR sample along with true HR data is used for the training SR models. Hence, a network trained with such data is prone to learn the distribution characteristics of synthetic LR samples far from that of true LR observations with diferent degradations making them less generalizable. While using real LR-HR pairs for training is a potential solution, it is impractical in WCE as patients cannot undergo multiple investigations for the same condition.

Lugmayr et al. (2019) introduced an unsupervised learning framework for training CNNbased super-resolution models on natural images without requiring paired LR-HR data. They employed CycleGAN to model the degradation process from HR to LR, enabling the SR network to be trained in a supervised fashion using synthetically degraded LR images. SRResCGAN (Umer et al., 2020) was further proposed as an extension of the same idea using diferent losses such as L1, Total-Variation (TV) and VGG losses. dSRVAE (Liu et al., 2020) implements sequential denoising followed by SR task with an encoder-decoder architecture to eliminate noise from the LR images. Further, Shocher et al. (2018) used zero shot learning making use of internal statistics of a single image for super-resolution, eliminating the need for external training data. However, its reliance on internal image redundancy may limit performance on images with minimal recurring structures, making it less efective for highly textured or complex medical images. Wang et al. (2021) proposed another approach that learns degradation representations in an unsupervised manner, allowing the model to adapt to various unknown degradation patterns. However, despite its efectiveness, the method may struggle with extreme degradations and requires careful optimization to generalize well across diferent imaging conditions. Furthermore, USISResNet (Prajapati et al., 2020) employed adversarial learning within an unsupervised domain adaptation framework to transform real noisy LR images into high-quality SR outputs. Similarly, Prajapati et al. (2021) leveraged unpaired LR-HR datasets for direct unsupervised learning, demonstrating improved SR performance without the need for paired supervision. GAN-based technique with one generator and two discriminators was used by Prajapati et al. (2021) and SR network was trained with quality, total variation and content losses. While a number of approaches can be seen for traditional computer vision applications, a limited number of works are noted for medical imaging (CT-(Li et al., 2023), MRI (Liu et al., 2024)). Despite the noted works targeting diferent modalities, the MDASR model (Liu et al., 2023) is the only work focusing on WCE using unsupervised learning with domain adaptation techniques. The summary of diferent supervised and unsupervised SR methods is highlighted in Table I in supplementary material due to space constraint. We further note the following limitations from existing works, specifically for WCE:

• In the medical domain, most of the SR techniques (Mahapatra et al., 2019; Almalioglu et al., 2020; Vaghela et al., 2023) rely on supervised training which demands a true pair of LR-HR images. However, it is often impossible to acquire such pairs which compels alternate strategies to improve the spatial resolution of medical data.

• Further, in the absence of a genuine LR-HR pair, most of the supervised SR techniques (Tong et al., 2017; Zhang et al., 2018; Ledig et al., 2017) simulate the LR observation with known degradation (i.e., bicubic downsampling). This idea of supervised training of deep models results in poor learning capabilities to the real-world data as one can note a significant distribution shift for synthetically degraded images versus naturally observed data (see Fig. 1 in supplementary material).

• Additionally, the successful SR models pre-trained on natural images (Wang et al., 2021; Prajapati et al., 2021; Liu et al., 2020) often fail when applied to medical images; thus, requiring domain-specific adaptation and transfer learning (Li et al., 2021) (see Fig. 1 in supplementary material)

• Moreover, unlike abundant works on to natural images as noted in Table I of supplementary material, unsupervised SR models for medical images, specifically for WCE data are still in their early stages with fewer established benchmarks and datasets i.e., Liu et al. (2023).

While noting the existing works for SR, it can be seen that Transformers have not been well explored for SR in medical imaging. Transformers capture global dependencies through selfattention across all inputs, whereas CNNs primarily model local spatial relationships. Hence, several methods have been proposed for SR of natural images using Transformer (Vaswani et al., 2017; Chen et al., 2024). To restrict the scope of self-attention, some researchers use local (window) self-attention, which splits the feature maps into smaller sections (Dosovitskiy et al., 2020). In the meanwhile, they improve the interaction between windows by using the shift mechanism (Liu et al., 2021), overlapping windows (Chen et al., 2023), or the cross-aggregation operation (Chen et al., 2022b). In comparison to earlier CNN-based techniques, these methods achieve linear complexity with regard to image size. The local design must stack a lot of blocks to create global dependencies, nevertheless, in contrast to global attention. The previously stated techniques implicitly capture global information, but they make it more dificult to model spatial dependencies - a critical component of SR. Therefore, we employ Transformers to create an image SR technique that can eficiently capture global spatial information on high-resolution images at a cheap computing cost. By providing a tailored approach for the unique challenges of WCE images, including domain discrepancies and the absence of paired high-resolution data, this paper bridges this gap for obtaining superior SR images.

![](images/a44c4d92844f6ed1b14dee5dbe25fa7fa08eb920a63310b6beb977852e9bd0d2.jpg)  
Figure 2: The block schematic of the proposed unsupervised SR model-UnCapsTSR for upscaling factor of ×4. Here, $I _ { H }$ indicates a kernel of High Pass Filter (HPF).

![](images/ac69d251a520c5a7e4c492ae25ffd7593bd10f03f28b4def9ec028977c459a49.jpg)  
Figure 3: The architecture of the Transformer-based Generator network (G) used in the proposed method-UnCapsTSR.

## 3. Proposed Framework: UnCapsTSR

To address the aforementioned limitations of current SR methodologies for WCE data, we propose a novel approach that utilizes a transformer model for the super-resolution employing unsupervised training for an upscaling factor of ×4. The block diagram of the proposed method UnCapsTSR is depicted in Fig. 2. Our proposed approach employs transformer-based generative adversarial learning in UnCapsTSR which incorporates three networks: a Generator (i.e., G), and two Discriminators $( \mathrm { i } . \mathrm { e } . , D _ { I }$ and $D _ { I I } )$ inspired by DUSGAN model (Prajapati et al., 2021) designed for natural images. However, the key diferences between DUSGAN model (Prajapati et al., 2021) and the proposed model are listed below:

• Generator Architecture: The block diagram mentioned in Prajapati et al. (2021) utilizes a CNN-based generator with residual-in-residual dense blocks to perform upscaling in the spatial domain. While the proposed model i.e., UnCapsTSR replaces the CNN backbone with a transformer-based generator incorporating recursive-generalization self-attention modules to better capture long-range dependencies in WCE images.

• Loss Function Integration: The loss function introduced in Prajapati et al. (2021) is primarily uses color loss, adversarial losses and Quality Assessment (QA) loss. While in the proposed model we incorporate color loss, adversarial losses, texture loss and new introduced Bilateral Total Variation (BTV) loss for optimization of proposed model.

• Domain Adaptability: The DUSGAN Prajapati et al. (2021) was originally designed for visible image data. Hence, the performance of such model on the medical endoscopy data exhibits degradation due to domain gap. The proposed model i.e., UnCapsTSR explicitly addresses this limitation by leveraging domain adaptation with unpaired LR WCE and HR conventional endoscopy images, ensuring superior performance across modalities.

• Multi-Modality Validation: Unlike DUSGAN, which is limited to visible natural imagery, the UnCapsTSR framework has been validated across multiple medical benchmarks (KID, GIANA), demonstrating its robustness and generalizability to diverse acquisition settings and clinical scenarios. Additionally, the proposed method is also validated on the other medical modality such as Retinal images. However, such finding are not presented in the DUSGAN Prajapati et al. (2021) work.

• Quality Assessment: The DUSGAN Prajapati et al. (2021) reports quantitaive evaluation primarily using standard no-reference metrics such as BRISQUE, PIQE, NIQE. However, the proposed method introduces domain specific metric, Endoscopy Image Quality Metric (EndoQM), specifically trained on WCE data, to better reflect diagnostic relevance in addition to above no-reference measures. Such quantitative evaluation is not missing in the DUSGAN Prajapati et al. (2021) paper.

The input LR image is first processed by the transformer-based generator network G to produce an SR output. This generated SR image is then evaluated against an unpaired HR image by Discriminator-II $( D _ { I I } )$ , which operates directly in the SR image space. As depicted in Fig. 2, Discriminator-I $( D _ { I } )$ is employed to refine the perceptual quality of the SR image by focusing on high-frequency components from both SR and unpaired HR images. To ensure stable training of both discriminators, the framework adopts the Least Squares GAN (LSGAN) loss. To integrate unsupervised learning into the proposed framework, we introduce an additional colorconsistency loss that ensures the color distribution in the super-resolved image remains consistent with that of the corresponding low-resolution input. This loss mitigates color distortions introduced during the reconstruction process and preserves the original chromatic information while enhancing spatial resolution. Thus, an $L _ { 1 }$ loss between SR and bi-cubically upsampled LR image is employed for this assignment to maintain the authenticity of color space in the SR image. In the following sub-sections, we present the detailed descriptions of each of components in above network. 8

## 3.1. Generator Network (G)

The architecture of generator network is depicted in Fig. 3. It is based on transformer, comprising following three key elements: (i) Low level Feature Extractor (LFE) (ii) High level Feature Extractor (HFE) (iii) Image Reconstruction (IREC).

A Low level Feature Extractor (LFE) module is utilized to capture low-frequency structural details from the Low-Resolution (LR) WCE input, denoted as $I _ { L R }$ . Initially, a convolutional layer with kernel of size $3 \times 3$ is applied to extract low-level feature representations. We use 64 number of channels to extract features from the given LR observation $( \mathrm { i } . \mathrm { e } . , I _ { L R } )$ to maintain complexity of the network inspired from Prajapati et al. (2021). Thus, the shallow features extracted by the LFE module, i.e., $F _ { 0 }$ can be formulated as:

$$
F _ { 0 } = \Phi _ { L F E } ( I _ { L R } ) ,\tag{1}
$$

where, $\Phi _ { L F E }$ represents the function of LFE module. Following this, a High level Feature Extractor (HFE) module is employed to extract high-frequency components essential for fine-detail reconstruction. This module composes a sequence of Residual Group (RG), which progressively refine feature representations, followed by a dedicated convolutional layer to enhance feature abstraction with 64 channels. Thus, the deep feature extracted by the HFE module, i.e. $F _ { d }$ can be formulated as,

$$
F _ { d } = \Phi _ { H F E } ( F _ { 0 } ) ,\tag{2}
$$

where, $\Phi _ { H F E }$ represents the function of HFE module. Further, to retain both low-frequency and high-frequency components, the features extracted from the LFE and HFE modules are fused via a residual connection (see Fig. 3), ensuring efective feature propagation:

$$
F _ { f u s i o n } = F _ { 0 } + F _ { d } ,\tag{3}
$$

where, $F _ { f u s i o n }$ represents the enhanced feature map obtained after residual connection. This integration preserves low-level details while enriching high-level feature representations. The fused features are then passed to the Image Reconstruction (IREC) module, which utilizes pixelshufle operations along with additional convolutional layers with kernel size $3 \times 3$ and with 64 features (i.e., channel = 64) to upsample and reconstruct the SR image. The final output is obtained as follows:

$$
I _ { S R } = \Phi _ { I R E C } ( F _ { f u s i o n } ) .\tag{4}
$$

where, $\Phi _ { I R E C }$ represents the function of Image Reconstruction module. Such hierarchical design in UnCapsTSR efectively combines low-level and high-level features, leading to enhanced SR results in the output.

Further, each Residual Group (RG) used in HFE module (see Fig. 3) comprises series of transformer blocks incorporated to enhance SR performance. As depicted in Fig. 3, the transformer blocks are categorized in two blocks: Local Self-Attention (L-SA) and Residual Group Self-Attention (RG-SA) blocks. These are interleaved in an alternating manner to preserve the topological structure of the module. Each transformer block consists of two Layer Normalization (LN) layers, a self-attention mechanism, and a Multilayer Perceptron (MLP) module, following the design principles of the transformer architecture.

Such a structured configuration enables efective modeling of both local and global feature representations. The Rectangle-Window Self-Attention (Rwin-SA) proposed in Chen et al. (2022b) is applied as the Local Self-Attention (L-SA) mechanism within the transformer-based architecture. The input feature maps are partitioned into non-overlapping rectangular windows, enabling localized attention computation while maintaining a balance between computational eficiency and detailed feature representation. This mechanism allows fine-grained spatial dependencies to be captured within each window, which is particularly beneficial for preserving intricate details in WCE images.

Further, the use of Recursive-Generalization Self-Attention (RG-SA) mechanism proposed in Chen et al. (2024) is to capture global contextual information eficiently with linear computational complexity. This employs a Recursive Generalization Module (RGM) to abstract the input features, irrespective of their resolution, into representative feature maps of fixed and small dimensions. This abstraction intuitively aggregates global information into compact, representative maps. Subsequently, a cross-attention mechanism is applied between the input features and representative maps, facilitating an eficient exchange of global contextual information. Given the significantly smaller size of the representative maps compared to the input features, this approach substantially reduces computational overhead. Additionally, as in the RGT (Chen et al., 2024), positional encoding is not explicitly added; instead, the model relies on the windowing mechanism and recursive generalization for positional awareness. Furthermore, RG-SA incorporates an adaptive adjustment of the channel dimensions of the query, key, and value matrices within the self-attention mechanism. This adjustment mitigates redundancy in the channel domain, optimizing computational eficiency while preserving the representational power of the attention mechanism. The combination of the RG-SA with the Local Self-Attention (L-SA) in an alternate arrangement utilizes the global context in a better manner.

## 3.2. Discriminator Networks (i.e., Discriminator-I & -II)

The proposed approach i.e.,UnCapsTSR consists of two discriminator networks (Discriminator-I and Discriminator-II) to improve quality of SR image. Both discriminators follow the same design as presented in Prajapati et al. (2021). Due to page limit we have described discriminators in Section III in supplementary material.

## 3.3. Loss functions

We elaborate the distinct loss functions that are associated with diferent blocks within the proposed unsupervised SR framework-UnCapsTSR.

## 3.3.1. Generator:

To improve the efectiveness of the generator network, the framework integrates a weighted combination of five distinct loss functions, as illustrated in Fig. 2. Mathematically, the total loss $( \mathrm { i } . \mathrm { e } . , \mathrm { L } _ { G } )$ ) can be provided as,

$$
\mathrm { L } _ { G } = \beta _ { 1 } L _ { \mathrm { c o l o r } } + \beta _ { 2 } L _ { \mathrm { G A N - I } } + \beta _ { 3 } L _ { \mathrm { G A N - I I } } + \beta _ { 4 } L _ { \mathrm { T e x t u r e } } + \beta _ { 5 } L _ { \mathrm { B T V } } ,\tag{5}
$$

where $L _ { c o l o r } , L _ { G A N - I } , L _ { G A N - I I } , L _ { T e x t u r e } , L _ { \mathrm { B T V } }$ reveal, color loss, generator adversarial loss with discriminator-I, generator adversarial loss with discriminator-II, texture loss and Bilateral Total Variation (BTV) loss, respectively. The weights of each loss are denoted by $\beta _ { i } , i = 1 , \ldots , 5$ . In UnCapsTSR, the use of unpaired LR-HR image pairs limits the applicability of traditional pixelwise losses such as $L _ { 1 }$ or $L _ { 2 } .$ , as they fail to preserve the structural and color consistency of the LR observation in the resulting SR image. To address this limitation, a color loss (Equation (6))

is introduced, which computes the absolute diference between the bicubically upsampled LR image and the generated SR image.

$$
L _ { \mathrm { c o l o r } } = \frac { 1 } { P } \sum ^ { P } \| I _ { S R } - B ( I _ { L R } ) \| .\tag{6}
$$

Here, || · || denotes the sum of absolute diferences over all pixels in the image. This explicitly clarifies that the loss is computed across all pixel locations in the vectorized domain. P stands for the size of the training batch, while $B ( \cdot )$ stands for the bicubically upsampling operation. As mentioned earlier, we apply Least Squares GAN (LSGAN) loss, to enhance training stability with both discriminators. Hence, the generator loss from Discriminator-I can be represented as,

$$
L _ { G A N - I } ^ { G } = \frac { 1 } { P } \sum ^ { P } \left. 1 - D _ { I } \left( I _ { H } \left( I _ { S R } \right) \right) \right. .\tag{7}
$$

In the above equation, $D _ { I } \left( \cdot \right)$ denotes the function of discriminator-I, and $\mathrm { I } _ { H }$ specifies the kernel weights for a High-Pass Filter (HPF). Similarly, the equation of the adversarial generator loss from Discriminator-II is,

$$
L _ { G A N - I I } ^ { G } = \frac { 1 } { P } \sum ^ { P } \left. 1 - D _ { I I } \left( I _ { S R } \right) \right. ,\tag{8}
$$

where $D _ { I I } ( \cdot )$ indicates the function of discriminator-II.

Importantly, to reconstruct the texture of SR image, we additionally propose to add texture loss (Sajjadi et al., 2017). This loss utilizes the gram matrix, where each element represents the inner product between the vectorized feature maps $\phi _ { i }$ and $\phi _ { j }$ at a specific layer of a pre-trained deep network. The gram matrix efectively captures the spatial correlations of feature maps, representing the global texture statistics of the image. Building upon this, the texture loss is defined as the MSE of the diference between the gram matrices of the generated super-resolved image $\left( I _ { \mathrm { S R } } \right)$ and the high-resolution image $\left( I _ { \mathrm { H R } } \right)$ . The texture loss is stated as (Shi et al., 2019),

$$
L _ { \mathrm { T e x t u r e } } = \left. G \left( \phi ( I _ { \mathrm { S R } } ) \right) - G \left( \phi ( I _ { \mathrm { H R } } ) \right) \right. _ { 2 } ^ { 2 } .\tag{9}
$$

Since LR and HR images in our setup are unpaired, standard SR approaches struggle to map low-resolution inputs to a plausible high-resolution output. By enforcing similarity between the texture distributions of generated SR images and real HR images, texture loss helps in learning a domain-consistent super-resolution model that produces realistic images suitable for clinical use. We extend the applicability of the above formulation by taking the Gram-matrix statistics of HR images as domain-level priors rather than image-specific targets. Specifically, the proposed method enforces that the texture distributions of generated SR images remain consistent with the global distribution of textures observed in high-quality endoscopic images. Thus, enforcing similarity between texture distributions refers to aligning the second-order feature correlations of generated SR outputs with those typical of the HR domain. This alignment ensures that the reconstructed SR images exhibit the global texture statistics characteristic of high-quality endoscopic images, including mucosal folds, vascular textures, and surface continuity, even in the absence of exact LR-HR pairs. In support of this, authors in Hu (2021); Chen et al. (2022a) utilized this concept, where multi-scale texture statistics in feature space are matched across domains to capture realistic textures without paired supervision. Inspired by this, our unpaired texture loss leverages Gram-matrix–based distribution alignment to reduce the domain gap between LR

WCE images and HR conventional endoscopy images, thereby producing SR outputs that are perceptually and diagnostically reliable. Further, we propose the Bilateral Total Variation (BTV) Loss as a customized loss function specifically designed for SR of capsule endoscopy images, capturing both local and contextual information. This loss incorporates a weighted summation of pixel-level diferences within a predefined spatial neighborhood also efectively models bilateral relationships between a central pixel and its surrounding pixels.

$$
{ \mathrm { L } } _ { B T V } ( { \cal I } _ { S R } , n , \epsilon ) \qquad = \qquad { \frac { \lambda } { C \cdot H \cdot W } } \sum _ { k = - n } ^ { n } \sum _ { l = - n } ^ { n } \sqrt { ( { \cal I } _ { S R } ( i , j ) - { \cal I } _ { S R } ( i + k , j + l ) ) ^ { 2 } + \epsilon }\tag{10}
$$

In the above equation, λ is weighing factor which is set to $1 \times e ^ { - 2 }$ . C, H and W are number of channels, height and width of the image, respectively. ϵ is the regularization term for numeric stability which is set to $1 \times e ^ { - 8 }$ . Although the squared diference term under the square root is mathematically non-negative, in practice, very small values often occur in uniform regions of WCE images (e.g., lumen or flat mucosa). Under mixed-precision training, such small values may lead to unstable gradient computations, including division-by-zero or undefined values (NaNs). The addition of ϵ guarantees that the square root and subsequent derivatives remain well-behaved across all regions of the image, thus improving training robustness without afecting the overall loss behavior. This loss ensures the preservation of fine-grained textures and structural details which are critical for medical analysis. By summing over neighboring pixels, the loss penalizes inconsistencies in spatial continuity, enhancing the structural integrity of the reconstructed SR images. Additionally, the inclusion of a small regularization term ϵ ensures numerical stability, while the normalization factor balances the contribution of the loss across diferent image scales and channels. Thus, the BTV loss is tailored to address the unique chal lenges of capsule endoscopy imaging, such as noise, motion artifacts, and low-resolution inputs, ultimately improving the quality and diagnostic utility of the SR images.

## 3.3.2. Discriminator-I

To distinguish high-frequency features between SR and HR images, Discriminator-I is employed. As it aims to align the high-frequency components of the SR image with those of the unpaired HR images, the associated LSGAN loss for Discriminator-I can be formulated as follows:

$$
L _ { G A N - I } ^ { D } = \frac { 1 } { P } \sum ^ { P } \left( \frac { | D _ { I } ( I _ { H } ( I _ { S R } ) ) | + \left| 1 - D _ { I } ( I _ { H } ( I _ { H R } ) ) \right| } { 2 } \right) .\tag{11}
$$

Here, $D _ { I } ( \cdot )$ represents operation of Discriminator-I.

## 3.3.3. Discriminator-II

Here, Discriminator-II is trained in an adversarial manner to distinguish the generated SR image from the original unpaired HR image. To stabilize the adversarial training process, the Least Squares GAN (LSGAN) loss is utilized for Discriminator-II, which is formulated as:

$$
L _ { G A N - I I } ^ { D } = \frac { 1 } { P } \sum ^ { P } \left( \frac { | D _ { I I } ( I _ { S R } ) | + \left| 1 - D _ { I I } ( I _ { H R } ) \right| } { 2 } \right) ,\tag{12}
$$

where, $D _ { I I } ( \cdot )$ denotes the function of Discriminator-II and $I _ { H R }$ is the unpaired HR image in the dataset.

## 4. Proposed Quality Metric - EndoQM

Another novel contribution of this work is the introduction of a new quality metric termed Endoscopy Quality Metric (EndoQM), which is specifically tailored for evaluating the quality of super-resolved endoscopic images. The design of EndoQM is motivated by the limitations of the widely used Natural Image Quality Evaluator (NIQE) proposed by Mittal et al. (2012). NIQE is a completely blind, no-reference quality assessment measure that does not require subjective opinion scores. It operates by modeling Natural Scene Statistics (NSS), extracted from high-quality natural images, and fitting them to a Multivariate Gaussian (MVG) distribution. The quality of a test image is then quantified as the distance between its NSS features and this pristine naturalimage distribution. While NIQE has proven efective for generic natural image assessment, its statistical assumptions are based on natural scenes, which exhibit very diferent textures, illumination patterns, and structural distributions compared to medical imagery such as Wireless Capsule Endoscopy (WCE). As shown in Fig. 1(b) in supplementary material, the statistical distribution of WCE images is narrower and more constrained, making NIQE less reliable in this domain. To overcome this limitation, EndoQM adopts the NIQE framework but re-estimates the MVG distribution using a curated set of high-quality WCE images instead of natural images. By learning the statistical model from WCE data, EndoQM captures the modality-specific features such as mucosal textures, vascular patterns, and illumination artifacts that are diagnostically relevant. Thus, while the feature extraction and mathematical pipeline remain identical to NIQE, the reference distribution is domain-specific. This simple but crucial adaptation ensures that EndoQM aligns more closely with the clinical requirements for endoscopy, providing reliable quality evaluation even in the absence of ground-truth high-resolution references. Unlike BRISQUE, which depends on predefined natural scene statistics, or PIQE, which is overly sensitive to pixel-wise distortions, NIQE’s Multivariate Gaussian (MVG)-based feature representation allows EndoQM to efectively model the unique structural and textural attributes of endoscopic images. This adaptability ensures that EndoQM aligns more closely with the clinical requirements for assessing the perceptual quality of WCE images, providing a reliable metric for evaluating super-resolution performance in medical imaging applications.

## 5. Experimental Evaluations

A comprehensive analysis evaluating the efectiveness of the proposed model (i.e., UnCapsTSR) by comparing it with other leading SR techniques for an upscaling factor of ×4 is presented in this section using qualitative and quantitative measures. We examine representative patches from the output images of all competing models, ofering a visual comparison of their reconstruction quality. On the quantitative side, performance is evaluated using widely recognized non-reference SR metrics, such as the Blind/Referenceless Image Spatial Quality Evaluator (BRISQUE) (Mittal et al., 2011), Natural Image Quality Evaluator (NIQE) (Mittal et al., 2012), and Perception-based Image Quality Evaluator (PIQE) (Venkatanath et al., 2015), along with a newly introduced metric EndoQM, specifically designed to evaluate SR results of capsule endoscopy image, which are crucial for diagnostic accuracy in medical imaging. Additionally, we present an exhaustive ablation study on the diferent aspects of the proposed model such as loss functions and network configuration, demonstrating their efectiveness in the proposed model. To assess generalization, the proposed model is tested on diferent benchmark datasets, including the KID (Koulaouzidis et al., 2017) and GIANA (Bernal and Aymeric, 2017) datasets, showcasing its generalizability and applicability across various clinical scenarios. Additionally, we perform an Analysis of Variance (ANOVA) test, two tile Z-test and Kolmogorov–Smirnov (K–S) test to evaluate the statistical significance of the performance diferences between the proposed method and other state-of-the-art approaches. The detailed description related to diferent experiments and its associated analysis are discussed further in the following sections.

Table 1: The details of endoscopic image datasets.
<table><tr><td rowspan="2">Details</td><td colspan="5">Dataset Name</td></tr><tr><td>Kvasir - Capsule (Smedsrud et al., 2021)</td><td>KID</td><td>GIANA</td><td>Kvasir - Conventional (Pogorelov et al., 2017)</td><td>Newly edited Kvasir capsule dataset for SR task</td></tr><tr><td>Year of publication</td><td>2021</td><td>(Koulaouzidis et al., 2017) (Bernal and Aymeric, 2017) 2017</td><td>2017</td><td>2017</td><td></td></tr><tr><td>Sensor</td><td>Olympus EC-S10 endocapsule</td><td>MiroCam® capsule</td><td>PillCam® SB3</td><td>digital high-definition endoscopes</td><td></td></tr><tr><td>Image types</td><td>Capsule</td><td>Capsule</td><td>Colonoscopy and Capsule</td><td>Gastrointestinal (conventional)</td><td>Capsule</td></tr><tr><td>No. of Images</td><td>47,238</td><td>2448</td><td>&gt;2000</td><td>&gt;8000</td><td>11550</td></tr><tr><td>Image size</td><td>336×336</td><td>360× 360</td><td>576× 576</td><td>720 × 574 to 1920 × 1072</td><td>280 × 280</td></tr><tr><td>Source</td><td>https://osf.io/dv2ag/</td><td>https://mdss.uth.gr/ datasets/endoscopy/kid/</td><td>https://endovissub2017- giana.grand-challenge.org/</td><td>https://datasets.simula.no/kvasir/</td><td></td></tr></table>

## 5.1. Datasets and training details

The proposed model i.e., UnCapsTSR has been trained in unsupervised manner, utilizing unpaired LR and HR images. A significant contribution of this research is the creation of new dataset derived specifically for the SR task, sourced from the existing Kvasir Capsule Endoscopy Dataset (Smedsrud et al., 2021), which comprises only Wireless Capsule Endoscopy (WCE) samples. As depicted in Table 1, the original Kvasir dataset contains 47 236 RGB images, each sized at $3 3 6 \times 3 3 6$ pixels, categorized based on various medical anomalies. We meticulously curated the dataset for the SR task by eliminating the unwanted portions such as redundant images and extraneous border pixels. Consequently, the newly edited SR dataset includes 10 000 training images, 550 validation images, and 1 000 testing images, with each image resized to $2 8 0 \times 2 8 0$ pixels. The UnCapsTSR, along with other models, has been evaluated on this new dataset for SR results. Further, these images serve as input in the LR block, as illustrated in Fig 2. Given that the model relies on unpaired LR-HR images, we incorporated an additional 10 000 conventional endoscopy images (Pogorelov et al., 2017) with a resolution of $1 0 2 4 \times 1 0 2 4$ pixels as the HR block in $\mathrm { F i g } 2 .$ . The details of this conventional Kvasir dataset is mentioned in Table 1. Thus, it is important to note that the training dataset in the proposed method does not consist of true LR-HR pairs; rather, the LR images are derived from WCE images, while the HR dataset comprises of images from conventional endoscopy resulting in a truly unpaired training dataset. Further, to assess the generalization capability of the proposed model, we also conducted tests on two additional datasets, namely KID (Koulaouzidis et al., 2017) and GIANA (Bernal and Aymeric, 2017) which are fully disjoint from training dataset (see Table 1), alongside the newly curated Kvasir dataset.

All SR methods including the proposed method-UnCapsTSR are trained on newly edited Kvasir dataset with an upscaling factor of ×4. The proposed approach is trained for batch size 32, where each input image is randomly cropped to 64 × 64 size, and the total training iterations are kept to 100K. The training patches are augmented using random horizontal flips and rotations with $9 0 ^ { \circ } , 1 8 0 ^ { \circ }$ and 270<sup>◦</sup>. The Adam optimizer is employed and the learning rate is set to $1 \times 1 0 ^ { - 4 }$ for generator, and $1 \times 1 0 ^ { - 3 }$ for discriminator which decays by half to twenty percent, forty percent, sixty percentage and eighty percentage of the total number of iterations within this training set. Additionally, the weightage to diferent losses in the total loss function (Equation (5)) $\mathrm { i . e . , } \beta _ { i } ,$ $i = 1 , \ldots , 5$ are set empirically to 1, 1, 1, $1 0 ^ { - 4 }$ and $1 0 ^ { - 2 }$ , respectively. The neighbor pixel in BTV loss is set to 5 in the proposed method. The number of Residual Groups (RGs) used in generator is 4 $( \mathrm { i . e . , } N = 4 )$ . Similarly, the number of L-SA and RG-SA in each RG is also kept to $4 \ : ( \mathrm { i . e . , } M = 4 )$ . In UnCapsTSR to kernel of HPF i.e., $I _ { H } ,$ , a Gaussian low-pass filter (LPF) with a

![](images/73dc1074982dc9fa7850763adb760eb69d0ebd95b3aab5652f6d275861903811.jpg)  
(a) EndoQM ↓ = 3.1946

![](images/07bd43acabb4e32316915d48b56f61fc434959fc802c7d85bbd0ab5a9d5c0cdc.jpg)  
(b) 9.6054

![](images/b2bec4bbe195b4808514cd7e84b1c88adde9d07231e5ce72c8b7d81685f20baf.jpg)  
(c) 10.5889

![](images/f39f66080f05a4ed82351d7221d7609f75c328e265c081f6a8599048f2efa0b8.jpg)  
(d) 10.6183  
Figure 4: The experimental justification showing the consistency of the newly proposed EndoQM measure through the degradation pipeline from HR to LR on a sample of conventional Kvasir dataset (Pogorelov et al., 2017). Below to each result, we depict value of EndoQM metric. (a) Original image of size 832 × 832, (b) degraded version of (a) with LPF, (c) bicubically downsampled versions of (b) with dimensions 208 × 208 and (d) noisy version of (c) with size 208 × 208.

![](images/d420386964013fc0cfb102e7daaac973e49d67faf2834e837785b658a4b07a99.jpg)  
(a) EndoQM↓ = 7.6746

![](images/b7f8eaa4451bf7cc16376d01686328d2ac6f43515887a2477334c0b157c1ee39.jpg)  
(b) 15.5934

![](images/8df951a89cdfb732eb0a8e9c0be04db76574f357c44e613a7f6f9827ae35a2da.jpg)  
(c) 20.7203

![](images/0d14ff31b61cb88abe0b9b44eb6c8840adecb83a7041af1a8c9ddeae1c658b88.jpg)  
(d) 20.7239  
Figure 5: The experimental justification showing the consistency of the newly proposed EndoQM measure through the degradation pipeline from HR to LR on a sample of WCE Kvasir dataset (Smedsrud et al., 2021). Below to each result, we depict value of EndoQM metric. (a) Original image of size 280 × 280, (b) degraded version of (a) with LPF, (c) bicubically downsampled versions of (b) with dimensions 70 × 70 and (d) noisy version of (c) with size 70 × 70.

9 × 9 kernel, zero mean, and a variance of 0.8 is initially constructed and subsequently inverted to obtain the high-pass filter (HPF) weights, which are empirically determined.

## 5.2. Usability of Endoscopy Quality Metric (EndoQM)

To show the usability of domain specific EndoQM measure for endoscopic images, we carried a series of experiments on conventional and capsule images to measure their quantitative performance. The objective of this analysis is to assess whether EndoQM captures distortions in endoscopic images subjected to various degradations unlike the standard no-reference metrics designed for general purpose images and thereby establish its suitability as a domain-specific image quality metric. Fig. 4 and Fig. 5 show the results of these experiments conducted on conventional and capsule images, respectively. In Fig. 4(a) and Fig. 5(a), the sample of conventional HR endoscopic and WCE images are displayed along with their EndoQM measure. The original images (i.e., Fig. 4(a) and Fig. 5(a)) are passed through a Gaussian Low-Pass Filter (LPF) with kernel size of 5 × 5 to attenuate high-frequency components, thereby simulating the degradation of fine texture and edge details commonly observed in low-quality endoscopic images (See Fig. 4(b) and Fig. 5(b)). One can see the values of EndoQM measure degrade and thus, ensuring the loss of high-frequency details in terms of quantitative value. Additionally, to show the degradation associated to downsampling operation, the filtered images in Fig. 4(b) and Fig. 5(b) are further downsampled by a factor of ×4, mimicking real-world LR endoscopic observations. The results of these experiments are depicted in Fig. 4(c) and Fig. 5(c). By observing the values of EndoQM measure in these cases, it can be deduced that the EndoQM score of these downsampled images are higher, showing downsampling degradation in images. Finally, sensor noise and transmission artifacts are also introduced by adding Gaussian noise with mean 0 and standard deviation 10 into Fig. 4(c) and Fig. 5(c) which further reduces perceptual quality (see Fig. 4(d) and Fig. 5(d)), where values of EndoQM measure are also increasing. Thus, the EndoQM is specifically designed to assess endoscopic image quality, an ideal metric should yield higher degradation scores for images with reduced perceptual fidelity and if EndoQM is a robust evaluator for endoscopic imaging, its score should reflect a progressive decline in image quality across the diferent degradation levels, i.e., the LPF-processed image, the downsampled image, and the noisy image should exhibit significantly higher EndoQM scores than the original endoscopic images as displayed in Fig. 4(a) and Fig. 5(a).

This experimental validation underscores the efectiveness of EndoQM in capturing clinically relevant quality degradations, distinguishing it from generic reference-less image quality metrics that may not fully align with endoscopic imaging characteristics. By demonstrating its sensitivity to key degradations in endoscopic images, EndoQM establishes itself as a reliable metric for evaluating image quality in real-world medical imaging applications, particularly in WCE and conventional endoscopy-based tasks. Consequently, the EndoQM metric assesses the quality of SR endoscopic images by examining local image statistics and comparing them with the established pristine quality model, thus providing a robust, domain-specific evaluation of image fidelity and perceptual realism. By overcoming the limitations of NIQE in the context of medical imaging, EndoQM delivers a more dependable and precise evaluation framework for endoscopic image quality, ensuring that the improvements brought about by SR techniques meet clinical diagnostic requirements. Therefore, this metric signifies a pivotal element in the assessment of image quality within the realm of medical imaging.

## 5.3. Performance Analysis on Newly Derived Kvasir Dataset

This section presents the qualitative and quantitative analysis of the proposed method on the newly derived Kvasir dataset, comparing its performance against state-of-the-art approaches. The dataset curated from the original Kvasir dataset for SR task consists of 1 000 LR samples for testing purposes with a resolution of 280 × 280 pixels. The SR results obtained using the proposed unsupervised transformer-based SR model are evaluated against the existing unsupervised models, including SRResCGAN (Umer et al., 2020), DUSGAN (Prajapati et al., 2021), dSR-VAE (Liu et al., 2020), ZSSR (Shocher et al., 2018), DASR (Wang et al., 2021), and MDASR (Liu et al., 2023) for upscaling factor ×4. Notably, all models, except MDASR, were originally trained in an unsupervised manner on natural images, whereas MDASR is specifically designed for training and evaluation on WCE images. Thus, to ensure fair comparison, we re-trained the natural image-trained models (SRResCGAN (Umer et al., 2020), DUSGAN (Prajapati et al., 2021), dSRVAE (Liu et al., 2020), ZSSR (Shocher et al., 2018) and DASR (Wang et al., 2021)) on WCE images and conducted a comprehensive qualitative and quantitative evaluation against the proposed method.

## 5.3.1. Qualitative Analysis

The qualitative results demonstrate the ability of the proposed approach to significantly enhance the quality of WCE images, a crucial factor for accurate clinical diagnostics. The Fig. 6 showcases a qualitative comparison between the proposed method and six other SR techniques, including SRResCGAN (Umer et al., 2020), DUSGAN (Prajapati et al., 2021), dSRVAE (Liu et al., 2020), ZSSR (Shocher et al., 2018), DASR (Wang et al., 2021), and MDASR (Liu et al., 2023), for the diferent endoscopic LR images. EndoQM score is noted in Fig. 6 to demonstrate the quantitative performance obtained using each SR method<sup>1</sup>. One can note that the LR input images (Fig. 6(a)) show severe degradation, including blurring and the loss of structural details, making it challenging to discern the fine features necessary for medical interpretation. This is also supported with the quantitative values of EndoQM measure obtained for LR samples, high values indicate poor preservation of high-frequency details.

Table 2: The quantitative analysis of proposed and other methods. Here, first and second highest values are highlighted with red and blue colors, respectively.
<table><tr><td rowspan="2">Method (×4)</td><td colspan="4">Newly edited Kvasir Dataset</td><td colspan="4">KID dataset</td><td colspan="4">GIANA Dataset</td></tr><tr><td>BRISQUE↓</td><td>PIQE↓</td><td>NIQE↓</td><td>EndoQM ↓</td><td></td><td>BRISQUE↓ PIQE↓</td><td>NIQE↓</td><td>EndoQM ↓</td><td>BRISQUE↓</td><td>PIQE↓</td><td>NIQE↓</td><td>EndoQM↓</td></tr><tr><td>SRResCGAN (Umer et al., 2020)</td><td>66.1341</td><td>58.1722</td><td>4.4847</td><td></td><td>10.3484</td><td>88.7592</td><td>75.7123 5.1225</td><td>10.7286</td><td>86.5817</td><td></td><td>82.1627 4.9225</td><td>9.7258</td></tr><tr><td>DUSGAN (Prajapati et al., 2021)</td><td>72.1658</td><td>87.5359</td><td>7.3791</td><td></td><td>13.8334</td><td>59.9603</td><td>54.9143 6.6590</td><td>17.3653</td><td>58.0079</td><td></td><td>51.3542 5.9432</td><td>13.4425</td></tr><tr><td>dSRVAE (Liu et al., 2020)</td><td>56.4932</td><td>64.9396</td><td>4.8916</td><td></td><td>10.8352</td><td>56.6771</td><td>47.8164 5.9949</td><td>13.5611</td><td>46.6371</td><td>43.2417</td><td>5.6909</td><td>10.0163</td></tr><tr><td>DASR(Wang et al., 2021)</td><td>63.7744</td><td>91.8829</td><td>6.0622</td><td></td><td>14.2492</td><td>71.3107</td><td>42.6297 5.4863</td><td>13.5945</td><td>63.1576</td><td></td><td>53.8924 4.2092</td><td>10.0339</td></tr><tr><td>ZSSR (Shocher et al., 2018)</td><td>64.7164</td><td>85.2007</td><td>6.0950</td><td></td><td>13.1060</td><td>83.7318</td><td>81.1143 4.9983</td><td>11.3095</td><td>86.9541</td><td>81.2241</td><td>5.0458</td><td>8.4063</td></tr><tr><td>MDASR (Liu et al., 2023)</td><td>54.9760</td><td>55.9510</td><td>4.9506</td><td></td><td>8.9283</td><td>76.7706</td><td>47.45504.8297</td><td>7.6567</td><td>71.0476</td><td></td><td>68.4324 4.8809</td><td>7.7718</td></tr><tr><td>Proposed</td><td>42.0358</td><td>41.8822</td><td>4.5346</td><td></td><td>7.0914</td><td>44.5833</td><td>29.3739 4.0198</td><td>6.7696</td><td>53.2019</td><td>29.7511</td><td>3.7547</td><td>6.9723</td></tr></table>

In Fig. 6(b, c), the SR results obtained using SRResCGAN and DUSGAN SR methods are presented where poor reconstruction of details with lack in high-frequency contents can be seen. Similarly, the SR image obtained using dSRVAE SR method in Fig. 6(d) generates smoother outputs, while the SR images obtained using ZSSR and DASR methods (see Fig. 6(e, f)) reconstruct better output compared to the earlier methods, however, they fail to preserve intricate details adequately. The noisy pixels in the reconstruction in the SR images obtained can be seen for ZSSR and DASR methods. The Fig. 6(g) displays the SR image obtained using MDASR SR method which is trained on domain-specific endoscopic images which generates over-smooth image and unrefined high-frequency textures. The SR output obtained using the proposed method (i.e., (Un-CapsTSR) in Fig. 6(h) demonstrates better clarity and details in reconstruction, preserving fine structural elements and textures making it visually appealing. The high-frequency details associated to the bordering pixels of diferent parts and regions are preserved well compared to other SR techniques. Further, the value of EndoQM measure for the proposed method also supporting the visual observation while comparing to all other SR methods.

## 5.3.2. Quantitative analysis

The SR results obtained using the proposed and other SR methods are quantitatively evaluated in terms of no-reference image quality metrics such as BRISQUE, NIQE, PIQE, and the newly introduced EndoQM in Table 2. These metrics provide insights into the perceptual quality of the super-resolved WCE images without relying on ground-truth HR images. The lower values of BRISQUE, PIQE, NIQE and EndoQM scores indicate better quality in the SR observations. In Table 2, the top two values are highlighted with red and blue colored texts, respectively, for the convenience of the reader. It can be deduced that the proposed model achieves the lowest BRISQUE score among all methods, indicating the highest quality and minimal artifacts. The PIQE evaluates image quality based on local distortion detection and the proposed model attains lowest scores among all, indicating minimal perceptual distortions and superior visual quality compared to the other SR methods. The SR methods such as ZSSR and DASR perform reasonably well in qualitative results despite introducing minor distortions in a few regions as captured by their higher PIQE scores. Further, the quantitative score in terms of NIQE measure is also best using the proposed model than the other competing SR methods, demonstrating its ability to restore perceptually natural textures in WCE images. However, the performance gap highlights the limitations of NIQE when applied to medical images. Finally, the EndoQM metric, specifically tailored for endoscopic images, provide the most domain-relevant evaluation. The proposed model consistently outperforms all other methods, achieving the lowest EndoQM scores highlighting superior ability of the proposed model to reconstruct high-frequency details and preserving structural and textural integrity, crucial for endoscopic diagnostics. Overall, the proposed model achieves the better quantitative performance across all metrics, along with newly proposed EndoQM results validating its suitability for clinical applications by aligning closely with the domain-specific requirements of endoscopic imaging for newly curated Kvasir dataset.

![](images/75acecc9200702f81113163319d8ae7dea3ce75a6e7ec9e15ce805afd98bbe8f.jpg)  
(a) LR (11.5602)

![](images/587169829af95058dba8df7c7d5ce68ffffca7172e7d003571d6fc9e854cb1ec.jpg)  
(b) SRResCGAN (Umer et al., 2020) (8.6431)

![](images/687d4b86152356f2ab18df4a857849514b2f0bc0d6a3e41f843c1a0406381755.jpg)  
(c) DUSGAN (Prajapati et al., 2021) (13.6732)

![](images/26022d2a8fac8ba06615ca3ffd7fa4e7b9fbd8c24d71554772390b923b112098.jpg)  
(d) dSRVAE (Liu et al., 2020) (9.2218)

![](images/ff8ae6df3ee7ba13eafbc0ddfd37ebfcdc625126c36ffdec659c2c1ce8c5a081.jpg)  
(e) ZSSR (Shocher et al., 2018) (12.1514)

![](images/ae4d3ac3cf44770b3c14e5b1543b10d8a1d1e2d8645c0b4fa9f68dc6dcb6e58e.jpg)  
(f) DASR (Wang et al., 2021) (14.1511)

![](images/93d8a1ae8dc683971c2a8e9adf6ed1ea7c633e41d385742e0962cdb287097a8a.jpg)  
(g) MDASR (Liu et al., 2023) (8.7219)

![](images/f16f349167b29f3eecebafa166fa8d22332fbd2f2325cbf7b255e91a3f0c168f.jpg)  
(h) Proposed  
(EndoQM↓ = 7.0383)  
Figure 6: The qualitative comparison of proposed and other existing unsupervised SR methods for upsampling factor ×4 on newly edited Kvasir dataset.

## 5.4. Performance analysis on KID dataset

Further, to evaluate the generalization capability of the proposed model, additional experiments are conducted on the KID dataset (Koulaouzidis et al., 2017), an external dataset containing endoscopic images with distinct characteristics (Details in Table 1). It is important to note here that the images of this dataset are not presented in the training of any algorithm.

![](images/78eff38e04f53603ec1eab4e98c3a7264941c61c5dd2251649f92b3e449009d1.jpg)  
(e) ZSSR (Shocher et al., 2018) (11.8243)

![](images/e2ccdf054eced80650ae56ba74e5edf97f6e37e3455ccc594af7d7db5fb83635.jpg)  
(f) DASR (Wang et al., 2021) (13.6794)

![](images/16d0ce3ccab3af9a9ebfc41017737696976d45a1454326b148dd812763d857fb.jpg)  
(g) MDASR (Liu et al., 2023) (7.6872)

![](images/226000bf4568dbcb05aba4bd89324d1e70074e686e7089199d923c422d64e053.jpg)  
(h) Proposed (EndoQM ↓ = 6.7876)  
Figure 7: The qualitative comparison of proposed and other existing unsupervised SR methods for upsampling factor ×4 on KID Dataset.

## 5.4.1. Qualitative Analysis

The qualitative comparison of the proposed model with state-of-the-art methods on the KID dataset for an upscaling factor of ×4 is shown in Fig. 7<sup>2</sup>. The LR images in Fig. 7(a) exhibit significant degradation with blurred lesion boundaries and loss of high-frequency details. Existing SR methods like SRResCGAN and DUSGAN (see Fig. 7(b, c)) enhance lesion shape and texture clarity to some extent but often introduce artifacts or over-smoothing, obscuring vascular details. Similarly, the dSRVAE method (see Fig. 7(d)) produces a smoother SR image but blurs the red region. The DASR and ZSSR methods (see Fig. 7(e, f)) improve visibility in broader lesion areas yet struggle to reconstruct intricate features like capillary networks. While MDASR (see Fig. 7(g)) generates a clean and smooth output, it over-smooths critical regions, leading to a loss of capillary networks and textural details needed for early-stage lesion identification and complex vascular pattern analysis. In contrast, the proposed model (see Fig. 7(h)) excels in preserving vascular details and reconstructing lesion textures, minimizing artifacts and enabling detailed clinical diagnostics.

## 5.4.2. Quantitative Analysis

The quantitative analysis to evaluate the generalization performance of the proposed model on the KID dataset is displayed in Table 2. Here, the proposed model demonstrates significant improvements across all metrics compared to other state-of-the-art methods. The SRResCGAN and ZSSR SR methods obtain high PIQE and BRISQUE scores, reflecting their inability to recover fine details as they tend to generate noisy artifacts in SR output. The MDASR shows moderate improvements leveraging its endoscopy-specific training, yet it fails to fully restore high-frequency details. In contrast, the proposed model attains the lowest (better) PIQE and BRISQUE scores, indicating minimal perceptual distortions and superior visual quality with restored structural coherence. Similarly, the NIQE scores further validated the perceptual quality of the proposed model, with significantly better results. However, the most notable performance is observed in the EndoQM metric, specifically tailored for endoscopic images. The proposed model obtains lowest EndoQM scores, reflecting its ability to efectively restore subtle textures, intricate vascular patterns, and diagnostically relevant details crucial for clinical applications. This superior performance of the proposed method can be attributed to the model’s transformerbased architecture and its domain adaptation strategies, which enable it to capture long-range dependencies and align with the unique statistical properties of endoscopic images.

![](images/87f0cdb950872d1ac60e27fb0986bd1622da90e8a7601a2c511f1c749c9c4b65.jpg)  
(a) LR (11.8082)

![](images/c27b01d1f55d839551d256f54256753cde825e6ec460c6365f8afa9e50cda002.jpg)  
(b) SRResCGAN (Umer et al., 2020) (10.4429)

![](images/58d32baed899e5361fee5a34c1d004a51860a93e50f491e2c37627ae99cf6525.jpg)  
(c) DUSGAN (Prajapati et al., 2021) (13.4488)

![](images/13b36d245f44664a8647cec2a049f261a52236cb9387aeba665cc4399e805dd4.jpg)  
(d) dSRVAE (Liu et al., 2020) (8.5677)

![](images/bee07d4397609e13b0ef58493c53094030882765378c5366a4eda13dfe7c3ab1.jpg)  
(e) ZSSR (Shocher et al., 2018) (9.6327)

![](images/18723ee44b2cadccbb9a22f24fc33c43ef445aae6fb19af16185207ec1420271.jpg)  
(f) DASR (Wang et al., 2021) (8.6178)

![](images/3049de4fceda2310093850e5812f11c4596b93634e69fca4bc741549aae37571.jpg)  
(g) MDASR (Liu et al., 2023) (7.1949)

![](images/10dfddc7ab826ea7250760b94d32c90cb5233e3ae09d4d0b5e530d5d8d4f23e0.jpg)  
(h) Proposed  
(EndoQM↓ = 6.5410)  
Figure 8: The qualitative comparison of proposed and other existing unsupervised SR methods for upsampling factor ×4 on GIANA Dataset.

## 5.5. Performance analysis on GIANA dataset

Finally, the performance of the proposed model is also evaluated on the GIANA (Bernal and Aymeric, 2017) dataset (Refer Table 1) to assess its generalization capability beyond the original training set.

## 5.5.1. Qualitative Analysis

The visual SR results of the proposed model are compared with other state-of-the-art models in Fig. 8<sup>3</sup>. Here, the LR input image exhibits significant blurring and loss of fine structures. The SR images generated by SRResCGAN and DUSGAN (see Fig. 8(b, c)) enhance the lesion area but fail to preserve vein information crucial for accurate diagnosis. Similarly, the ZSSR method (see Fig. 8(f)) provides moderate lesion sharpness but lacks suficient texture detail for clinical reliability. The dSRVAE and MDASR methods (see Fig. 8(d, g)) ofer improved lesion boundary sharpness but sufer from over-smoothing, resulting in texture loss. In contrast, the proposed model (see Fig. 8(h)) significantly outperforms other methods by reconstructing sharper lesion boundaries while preserving natural texture patterns, delivering enhanced clarity and diagnostic reliability suitable for clinical applications.

## 5.5.2. Quantitative Analysis

Table 2 depicts the quantitative analysis on GIANA dataset with the other state-of-the-art models. By visualizing this table, one can note that the results of the proposed model are significant than the other existing methods across all evaluated metrics. For instance, the SR models such as SRResCGAN (Umer et al., 2020) and ZSSR (Shocher et al., 2018) exhibit high PIQE and BRISQUE scores, highlighting their limitations in recovering fine details and their propensity to generate outputs with noticeable noisy artifacts. While MDASR demonstrates moderate improvements by leveraging endoscopy-specific training, it fails to fully reconstruct high-frequency details and thus, introduces minor distortions in regions with complex textures. In contrast, the proposed model achieves the lowest (i.e., optimal) PIQE and BRISQUE scores, signifying minimal perceptual distortions and superior structural fidelity in its reconstructions. Additionally, the NIQE scores further corroborated the model’s enhanced perceptual quality, showcasing a marked improvement over the competing SR methods. Notably, the proposed model excels in the EndoQM metric. Quantitative metric reflects an exceptional ability to recover subtle textures, intricate patterns, and diagnostically critical details.

## 5.6. Statistical Analysis

To assess the reliability of the quantitative results obtained using the proposed (i.e. Un-CapsTSR) and state-of-the-art SR methods, we conducted a statistical evaluation employing the Analysis of Variance (ANOVA) test. In this test, the quantitative results for BRISQUE, PIQE, NIQE, and EndoQM metrics computed at a 95% Confidence Interval (CI) are presented in Table II of supplementary material, while the corresponding box-plots illustrating the distribution of these quality metrics across diferent SR methods on the newly curated Kvasir Capsule test dataset are depicted in Fig. 9 of supplementary material. To further strengthen the reliability and robustness of the proposed UnCapsTSR framework, we performed a statistical validation analysis using a two-tailed Z-test (Pokhriyal and Jain, 2025a, 2023). The purpose of this test is to determine whether the performance improvements observed with UnCapsTSR over baseline models are statistically significant or could be attributed to random variation. The distribution plots for the Z-tests across all baselines and metrics is shown in Fig. 10 in supplementary material. To further establish the robustness of the proposed UnCapsTSR framework, we have conducted a Kolmogorov–Smirnov (K–S) statistical analysis (Pokhriyal and Jain, 2025b, 2024) on the noreference quality assessment scores of WCE images. The K–S test is a non-parametric method that measures the maximum vertical distance (D) between two Cumulative Distribution Functions (CDFs), thereby quantifying whether two samples are drawn from the same distribution. In this study, we applied two-sample K–S tests between the UnCapsTSR model and each baseline method across non reference metrics: BRISQUE, PIQE, NIQE, and EndoQM. The Empirical Cumulative Distribution Functions (ECDFs) of the scores for all models is shown in Fig. 11 in supplementary material.

## 5.7. Ablation Study

To comprehensively analyze the efectiveness of the proposed model (i.e., UnCapsTSR), an ablation study is carried out. This study evaluates the impact of diferent loss functions and network configurations on the performance across multiple datasets, including the newly edited Kvasir, KID, and GIANA datasets. The evaluation is performed using diferent reference-less quality metrics: BRISQUE, PIQE, NIQE, and EndoQM which are summarized in Table III in the supplementary material. The SR results obtained using these experiments are also depicted in Fig. 12 and Fig. 13 in the supplementary material to see their performance in visual aspects.

## 6. Conclusion

The WCE is a preferred over traditional endoscope in the medical field to inspect patient’s GI track. Despite many benefits of this technology, its hardware limitations lead to sense data with coarser resolution. This work presented an approach named as UnCapsTSR to enhance the spatial resolution of the LR WCE data using unsupervised training eliminating the need for genuine LR-HR paired datasets for supervised training. The approach makes use of transformer in GAN framework with direct domain transfer from LR to SR without the explicit use of degradation learning. Along with perceptual GAN loss, the UnCapsTSR also employs the novel Bilateral Total Variation (BTV) loss to improve the spatial continuity in the final SR image. To check the generalizability of the proposed method, it has experimented on the number of other datasets whose samples are not part of the training and their SR results are checked with many unsupervised SR approaches. Through this study, we also introduced domain-specific quantitative reference-less index designed to gauge to quality of WCE SR models. Additionally, we also introduced the curated version of the Kvasir dataset to train diferent SR algorithms. The visual and quantitative analysis of the proposed method demonstrates its efectively over the other state-of-the-art SR methods alongside the statistical evaluation in terms of ANOVA test, Z-test and Kolmogorov–Smirnov (K–S) test. Future works can also extend the proposed approach for super resolution by taking into account pathology specific details to have better quality images while improving the diagnostic quality.

## Acknowledgements

The authors are thankful to the CapsNetwork – International Network for Capsule Imaging in Endoscopy (project no. 322600) funded by the Research Council of Norway.

## References

Almalioglu, Y., Bengisu Ozyoruk, K., Gokce, A., Incetan, K., Irem Gokceler, G., Ali Simsek, M., Ararat, K., Chen, R.J., Durr, N.J., Mahmood, F., Turan, M., 2020. Endol2h: Deep super-resolution for capsule endoscopy. IEEE TMI 39, 4297–4309. doi:10.1109/TMI.2020. 3016744.

Bernal, J., Aymeric, H., 2017. Gastrointestinal image analysis (giana) angiodysplasia d&l challenge. Web-page of the 2017 Endoscopic Vision Challenge .

Chang, H., Yeung, D.Y., Xiong, Y., 2004. Super-resolution through neighbor embedding, in: Proceedings of the 2004 IEEE Computer Society Conference on CVPR 2004., pp. I–I. doi:10. 1109/CVPR.2004.1315043.

Chen, C., Shi, X., Qin, Y., Li, X., Han, X., Yang, T., Guo, S., 2022a. Real-world blind superresolution via feature matching with implicit high-resolution priors, in: Proceedings of the 30th ACM International Conference on Multimedia, pp. 1329–1338.

Chen, X., Wang, X., Zhou, J., Qiao, Y., Dong, C., 2023. Activating more pixels in image superresolution transformer, in: Proceedings of the IEEE/CVF conference on CVPR, pp. 22367– 22377.

Chen, Z., Zhang, Y., Gu, J., Kong, L., Yang, X., 2024. Recursive generalization transformer for image super-resolution, in: The Twelfth International Conference on Learning Representations.

Chen, Z., Zhang, Y., Gu, J., Kong, L., Yuan, X., et al., 2022b. Cross aggregation transformer for image restoration. Advances in Neural Information Processing Systems 35, 25478–25490.

Dong, C., Loy, C.C., He, K., Tang, X., 2014. Learning a deep convolutional network for image super-resolution, in: Fleet, D., Pajdla, T., Schiele, B., Tuytelaars, T. (Eds.), Computer Vision – ECCV 2014, Springer International Publishing, Cham. pp. 184–199.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al., 2020. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 .

Hu, X., 2021. Multi-texture gan: exploring the multi-scale texture translation for brain mr images. arXiv preprint arXiv:2102.07225 .

Iddan, G., Meron, G., Glukhovsky, A., Swain, P., 2000. Wireless capsule endoscopy. Nature 405, 417. doi:10.1038/35013140.

Karargyris, A., Bourbakis, N., 2010. Wireless capsule endoscopy and endoscopic imaging: A survey on various methodologies presented. IEEE Engineering in Medicine and Biology Magazine 29, 72–83. doi:10.1109/MEMB.2009.935466.

Kim, J., Lee, J.K., Lee, K.M., 2016. Accurate image super-resolution using very deep convolutional networks, in: Proc. of the IEEE conference on CVPR, pp. 1646–1654.

Koulaouzidis, A., Iakovidis, D.K., Yung, D.E., Rondonotti, E., Kopylov, U., Plevris, J.N., Toth, E., Eliakim, A., Wurm Johansson, G., Marlicz, W., Mavrogenis, G., Nemeth, A., Thorlacius, H., Tontini, G.E., 2017. KID Project: an internet-based digital video atlas of capsule endoscopy for research purposes. Endosc Int Open 5, E477–E483.

Ledig, C., Theis, L., Huszár, F., Caballero, J., Cunningham, A., Acosta, A., Aitken, A., Tejani, A., Totz, J., Wang, Z., et al., 2017. Photo-realistic single image super-resolution using a generative adversarial network, in: Proc. of the IEEE conf. on CVPR, pp. 4681–4690.

Li, Y., Chen, L., Li, B., Zhao, H., 2023. 4× super-resolution of unsupervised ct images based on gan. IET Image Processing 17, 2362–2374.

Li, Y., Sixou, B., Peyrin, F., 2021. A review of the deep learning methods for medical images super resolution problems. IRBM 42, 120–133. URL: https://www.sciencedirect. com/science/article/pii/S1959031820301408, doi:https://doi.org/10.1016/j. irbm.2020.08.004.

Liu, J., Li, H., Huang, T., Ahn, E., Han, K., Razi, A., Xiang, W., Kim, J., Feng, D.D., 2024. Unsupervised representation learning for 3-d magnetic resonance imaging superresolution with degradation adaptation. IEEE Trans. on AI 5, 4660–4674. doi:10.1109/TAI.2024.3397292.

Liu, T., Chen, Z., Li, Q., Wang, Y., Zhou, K., Xie, W., Fang, Y., Zheng, K., Zhao, Z., Liu, S., et al., 2023. Mda-sr: Multi-level domain adaptation super-resolution for wireless capsule endoscopy images, in: International Conference on MICCAI, Springer. pp. 518–527.

Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B., 2021. Swin transformer: Hierarchical vision transformer using shifted windows, in: Proc. of the IEEE/CVF ICCV, pp. 10012–10022.

Liu, Z.S., Siu, W.C., Wang, L.W., Li, C.T., Cani, M.P., 2020. Unsupervised real image superresolution via generative variational autoencoder, in: Proc. of the IEEE/CVF CVPR workshops, pp. 442–443.

Lugmayr, A., Danelljan, M., Timofte, R., 2019. Unsupervised learning for real-world superresolution, in: 2019 IEEE/CVF ICCVW, IEEE. pp. 3408–3416.

Mahapatra, D., Bozorgtabar, B., Garnavi, R., 2019. Image super-resolution using progressive generative adversarial networks for medical image analysis. Computerized Medical Imaging and Graphics 71, 30–39.

Mittal, A., Moorthy, A.K., Bovik, A.C., 2011. Blind/referenceless image spatial quality evaluator, in: 2011 conference record of the forty fifth asilomar conference on signals, systems and computers (ASILOMAR), IEEE. pp. 723–727.

Mittal, A., Soundararajan, R., Bovik, A.C., 2012. Making a “completely blind” image quality analyzer. IEEE Signal processing letters 20, 209–212.

Nuño-Maganda, M.A., Arias-Estrada, M.O., 2005. Real-time fpga-based architecture for bicubic interpolation: an application for digital image scaling, in: 2005 International Conference on Reconfigurable Computing and FPGAs (ReConFig’05), IEEE. pp. 8–pp.

Pogorelov, K., Randel, K.R., Griwodz, C., Eskeland, S.L., de Lange, T., Johansen, D., Spampinato, C., Dang-Nguyen, D.T., Lux, M., Schmidt, P.T., Riegler, M., Halvorsen, P., 2017. Kvasir: A multi-class image dataset for computer aided gastrointestinal disease detection, in: Proceedings of the 8th ACM on Multimedia Systems Conference, ACM, New York, NY, USA. pp. 164–169. doi:10.1145/3083187.3083212.

Pokhriyal, H., Jain, G., 2023. Sarcasm detection via sentiment and emotion analysis of news headlines using optimized simulation of weibull entropy distribution, in: 2023 Second International Conference on Informatics (ICI), pp. 1–6. doi:10.1109/ICI60088.2023.10421084.

Pokhriyal, H., Jain, G., 2024. Implicit and explicit sarcasm detection via target of ofensive language and cognition of logistic tsallis entropy distribution in social networks, in: 2024 IEEE International Conference on Interdisciplinary Approaches in Technology and Management for Social Innovation (IATMSI), pp. 1–6. doi:10.1109/IATMSI60426.2024.10503150.

Pokhriyal, H., Jain, G., 2025a. Sarcasm detection with induced sentimental cues using heuristic search based on unconstrained optimisation learning quantifying callousness on social media. Neurocomputing 645, 130499. URL: https://www.sciencedirect.com/science/ article/pii/S0925231225011713, doi:https://doi.org/10.1016/j.neucom.2025. 130499.

Pokhriyal, H., Jain, G., 2025b. Supposititious sarcasm detection and sentiment analysis coping hindi language in social networks harnessing zipf- mandelbrot probabilistic optimisation and perplexity entropy learning. ACM Trans. Asian Low-Resour. Lang. Inf. Process. 24. URL: https://doi.org/10.1145/3712061, doi:10.1145/3712061.

Prajapati, K., Chudasama, V., Patel, H., Upla, K., Raja, K., Ramachandra, R., Busch, C., 2021. Direct unsupervised super-resolution using generative adversarial network (dus-gan) for realworld data. IEEE TIP 30, 8251–8264. doi:10.1109/TIP.2021.3113783.

Prajapati, K., Chudasama, V., Patel, H., Upla, K., Ramachandra, R., Raja, K., Busch, C., 2020. Unsupervised single image super-resolution network (USISResNet) for real-world data using generative adversarial network, in: Proc. of the IEEE/CVF Conf on CVPRW, pp. 464–465.

Sajjadi, M.S., Scholkopf, B., Hirsch, M., 2017. Enhancenet: Single image super-resolution through automated texture synthesis, in: Proceedings of the IEEE ICCV, pp. 4491–4500.

Shi, Y., Li, B., Wang, B., Qi, Z., Liu, J., 2019. Unsupervised single-image super-resolution with multi-gram loss. Electronics 8. URL: https://www.mdpi.com/2079-9292/8/8/833, doi:10.3390/electronics8080833.

Shocher, A., Cohen, N., Irani, M., 2018. “zero-shot” super-resolution using deep internal learning, in: Proc. of the IEEE conf. on CVPR, pp. 3118–3126.

Singh, A., Singh, J., 2016. Super resolution applications in modern digital image processing. International Journal of Computer Applications 150, 6–8. doi:10.5120/ijca2016911458.

Smedsrud, P.H., Thambawita, V., Hicks, S.A., Gjestang, H., Nedrejord, O.O., Espen Næss, H.B., Jha, D., Berstad, T.J.D., Eskeland, S.L., Lux, M., Espeland, H., Petlund, A., Nguyen, D.T.D., Garcia-Ceja, E., Johansen, D., Schmidt, P.T., Toth, E., Hammer, H.L., de Lange, T., Riegler, M.A., Halvorsen, P., 2021. Kvasir-capsule, a video capsule endoscopy dataset. Scientific Data 8 doi:https://doi.org/10.1038/s41597-021-00920-z.

Smith, P., 1981. Bilinear interpolation of digital images. Ultramicroscopy 6, 201–204.

Swain, P., 2003. Wireless capsule endoscopy. Gut 52, iv48–iv50. URL: https://gut.bmj. com/content/52/suppl\_4/iv48, doi:10.1136/gut.52.suppl\_4.iv48.

Tong, T., Li, G., Liu, X., Gao, Q., 2017. Image super-resolution using dense skip connections, in: 2017 IEEE ICCV, pp. 4809–4817. doi:10.1109/ICCV.2017.514.

Umer, R.M., Foresti, G.L., Micheloni, C., 2020. Deep generative adversarial residual convolutional networks for real-world super-resolution, in: Proc. of the IEEE/CVF conf. on CVPR Workshops, pp. 438–439.

Vaghela, H., Sarvaiya, A., Premlani, P., Agarwal, A., Upla, K., Raja, K., Pedersen, M., 2023. Dcan:densenet with channel attention network for super-resolution of wireless capsule endoscopy, in: 2023 11th European Workshop on Visual Information Processing (EUVIP), pp. 1–6. doi:10.1109/EUVIP58404.2023.10323037.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., Polosukhin, I., 2017. Attention is all you need. Advances in neural information processing systems 30.

Venkatanath, N., Praneeth, D., Bh, M.C., Channappayya, S.S., Medasani, S.S., 2015. Blind image quality evaluation using perception based features, in: 2015 twenty first national conference on communications (NCC), IEEE. pp. 1–6.

Wang, L., Wang, Y., Dong, X., Xu, Q., Yang, J., An, W., Guo, Y., 2021. Unsupervised degradation representation learning for blind super-resolution, in: Proc. of the IEEE/CVF conf. on CVPR, pp. 10581–10590.

Wang, Y., Cai, C., Zou, Y.X., 2015. Single image super-resolution via adaptive dictionary pair learning for wireless capsule endoscopy image, in: 2015 IEEE International Conference on DSP, pp. 595–599. doi:10.1109/ICDSP.2015.7251943.

Yang, F., Yang, H., Fu, J., Lu, H., Guo, B., 2020. Learning texture transformer network for image super-resolution, in: Proceedings of the IEEE/CVF conference on CVPR, pp. 5791–5800.

Zhang, Y., Li, K., Li, K., Wang, L., Zhong, B., Fu, Y., 2018. Image super-resolution using very deep residual channel attention networks, in: Proceedings of the ECCV, pp. 286–301.