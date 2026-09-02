# StainPresetNet: Stain Preset Network for Fast Multi-to-Multi Stain Normalization

Hongtao Kang, Die Luo, Li Chen, Jing Cai, Junbo Hu, Xiuli Liu, Shenghua Cheng

Abstract— Stain normalization reduces color variations caused by variations in staining protocols and imaging conditions, thereby enhancing computer-aided diagnostic system performance. Traditional methods derive mapping relationships from individual or limited reference images through pixel-wise transformation, offering style flexibility but suffering from inaccurate color mapping extraction. While existing deep-learning-based approaches achieve accurate dataset-wide color mapping through complex neural networks, they face challenges including computational inefficiency, artifact generation, and fixed normalization directions requiring model retraining for directional changes. To address these limitations, we propose StainPresetNet - a novel framework that combines structural preservation with dataset-level color mapping while maintaining computational efficiency. Our method implements pixelwise normalization guided by preset reference images, enabling multi-directional adaptability without retraining. Evaluations on cytopathology and histopathology datasets demonstrate that StainPresetNet achieves superior color mapping accuracy compared to conventional methods, effectively improves classifier generalization in diagnostic tasks, and reduces computational overhead by 90% versus existing deep learning approaches. The proposed presetguided mechanism facilitates flexible adjustment of normalization directions through simple reference image replacement, overcoming the directional rigidity of current deep-learning-based solutions.

Index Terms— Stain Normalization, Cytopathology, Histopathology, Convolutional Neural Network, Generative Adversarial Network.

## I. INTRODUCTION

D <sup>UE</sup> <sup>to</sup> <sup>various</sup> <sup>factors,</sup> <sup>pathological</sup> <sup>images</sup> <sup>often</sup> <sup>exhibit</sup>significant color variations [1]. The significant color significant color variations [1]. The significant color variations in pathological images hinder the application of a unified algorithm for image analysis, thereby severely impacting the robustness of computer-aided diagnosis (CAD) systems [1]–[3]. Stain normalization can standardize patholog ical images obtained from different sources and under varying conditions to a consistent color distribution, thereby achieving uniform or near-uniform staining effects [1]–[3]. Moreover, stain normalization can adjust the background brightness, contrast of pathological images to a unified and appropriate range, thus enhancing image quality. This improvement not only benefits the performance of CAD systems but also aids pathologists in making more accurate and consistent diagnoses.

Although stain normalization methods have a rich research history, several issues need to be addressed before they can be considered truly practical [2], [3]. Conventional stain normalization methods, whether based on global matching or stain separation, rely on reference images to extract mapping relationships [2]. However, these conventional methods struggle to extract accurate mapping relationships from a single reference image [3].

As general methods, the deep learning-based staining normalization methods are not limited by pathological image types, staining methods, reagents, or imaging techniques, and can be applied to various datasets [3]. Given that pathological images serve as critical evidence for disease diagnosis, stain normalization methods must objectively and faithfully preserve the information of the source images, making only color adjustments without altering the image content (such as introducing distortions or artifacts). Most deep learning-based stain normalization methods employ complex deep networks to perform the normalization [4], [5]. Due to the complexity of deep networks and the instability of adversarial training, normalized images often suffer from distortions and artifacts, which is unacceptable in pathological images [4], [5].

Although some researchers have tried to reduce anomalies by adding additional branches and losses [6]–[8], these improvements often complicate the training process and fail to fundamentally eliminate anomalies. In addition, the normalization direction of existing deep learning-based methods is usually fixed, and when the normalization direction needs to be changed, retraining is needed to learn a new color-mapping relationship [9], [10]. However, training deep learning-based stain normalization methods is a complex and time-consuming task, as it usually involves manually adjusting dozens of hyperparameters, which may take several hours to days to complete.

To address these problems, we propose a fast multi-to-multi stain normalization network, StainPresetNet, and its training framework. Our method has the following features:

1. Structure Preservation: StainPresetNet performs normalization using a fully 1×1 convolutional sub-network, ensuring that only the colors are adjusted without altering the image’s texture or structure.

2. Accurate Mapping Relationships: StainPresetNet employs a backbone network that directly controls the fully 1×1 convolutional color mapping sub-network, enabling the network to handle complex color mapping relationships accurately.

3. Multi-domain to Multi-domain Applicability: Stain-PresetNet uses reference images to guide the direction of stain normalization. This allows it to normalize multiple different styles of images to a single style or transform a single image to multiple different styles by changing the reference image.

4. High Computational Efficiency: The backbone network in StainPresetNet runs at low resolution, and the color mapping sub-network has only 59 parameters, resulting in high computational efficiency.

## II. RELATED WORKS

## A. Conventional Stain Normalization methods

Conventional stain normalization methods can be categorized into two types: global normalization methods [11], [12] and stain separation-based methods [13]–[17].

Global normalization methods typically process images in the LAB color space and mainly include histogram specification methods [11] and color matching methods [12]. Histogram specification achieves normalization by mapping the histogram of the source image to the histogram of the target image [11]. The color matching method normalizes the means and standard deviations of the source image’s LAB color space to those of the target image [12]. However, normalizing only the mean and standard deviation does not eliminate the impact of local differences within the images [2].

The principle behind stain separation-based methods is that the concentration of stains and their intensity in an image are linearly related in the Optical Density (OD) space, allowing for the separation of different stains into independent channels, which can then be normalized individually [2]. These methods can be divided into supervised and unsupervised stain separation methods. Supervised stain separation methods use manual measurements or pre-trained classifiers to separate pixels stained by different stains, thereby obtaining a stain matrix [13], [14]. Unsupervised stain normalization methods use techniques such as Independent Component Analysis [15], Non-Negative Matrix Factorization [17], and Singular Value Decomposition [16] to extract the stain matrix.

## B. Deep Learning Based Methods

Since aligned source and target images are difficult to obtain, the methods based on pix2pix gan often train by performing color transformation and reconstruction on the target image [18]–[21]. Common color transformations include grayscale [19], HSV [20], CIEL \*a\*b [21], etc. In addition, some researchers have tried to directly use unaligned source and target images for training. These methods often add additional branches to constrain the content of the generated images, such as using diagnostic task losses or using pretrained VGG19 [22] to constrain the content and style of the generated images [23].

CycleGAN-based stain normalization methods use cycle consistency loss to constrain image content and are trained using real source and target images, allowing the learning of accurate mapping relationships. Shaban et al. used Cycle-GAN [24] for stain normalization and named their method StainGAN [25]. Subsequently, Runz et al. and Lo et al. adopted similar approaches [26]. Cai et al.replaced the original generator with a U-Net generator with residual structures [27]. Mahapatra et al. introduced self-supervised semantic guidance into CycleGAN training to reduce artifacts [7]. Shrivastava et al. incorporated attention mechanisms and structural similarity loss [8].

CycleGAN [24] is well-suited for transformation between two domains. However, for the problem of normalizing images from multiple domains to a single domain (multi-to-one transformation), it is uncertain to which specific domain the single domain images should be transformed, making the direction of mapping indeterminate. Although CycleGAN [24] can be directly used by treating multiple different domains as a single domain, this approach is likely to result in performance loss. To apply CycleGAN [24] to multi-to-one stain normalization tasks, researchers have proposed several different solutions. Zhou et al. suggested using a stain color matrix to control the mapping between the single domain and multiple domains [6]. Nazki et al. proposed a solution similar to StarGAN [28], using a single generator for cycle consistency training [29].

StainNet [5] is a full 1×1 convolutional network. Due to network capacity limitations, it is only suitable for one-to-one color normalization . Based on StainNet, ParamNet [30] uses a backbone network to predict the weights and biases of the full 1×1 convolutional network, solving the problem of insufficient network capacity of StainNet. Our StainPresetNet proposed in this paper uses a reference image to guide the normalization direction on the basis of ParamNet, thereby having the ability to perform stain normalization transformation in multiple styles.

## III. METHODS

In this section, we introduce the network structure of ParamNet and its adversarial training framework.

## A. Network Structure ofStainPresetNet

As illustrated in Fig. 1, StainPresetNet contains two encoders (a source encoder and a reference encoder), a fusion layer, and a color mapping sub-network. The source encoder extracts color distribution feature vectors from the source image, and the reference encoder extracts color distribution feature vectors from the reference image. The fusion layer calculates all parameters (weights and biases) of the color mapping sub-network based on the color distribution feature vectors from the source and reference images. The color mapping sub-network, a residual structure network with fully 1×1 convolutions, directly normalizes the source image.

![](images/6f2c3339ef513b7fb14a275c1cacc04da5b7f7b47ecc626e23c2f63af76581fe.jpg)  
Fig. 1: The network structure of StainPresetNet. The source encoder and reference encoder are used to extract color feature vectors, and the fusion layer is used to calculate the parameters of the color mapping sub-network, which is used to normalize the source image.

In our implementation, both the source encoder and the reference encoder use a modified ResNet18 [31], in which all batch normalization layers are removed and all convolutional layers are set as biased convolutional layers. The fusion layer contains two linear layers with the activation function of Leaky Rectified Linear Unit (LeakyReLU). For the stability of the network, the outputs of the fusion layer will be normalized by the tanh function, and then multiplied by a learnable parameter. The learnable parameter represents the numerical range of the parameters (weights and biases of convolutional layers) in the color mapping sub-network and its initial value is set to 4.5. The input images of the source encoder and the reference encoder are downsampled to 128×128 by default.

During the running stage, the weights and biases of the color mapping sub-network are not fixed, but are determined by the style of the input image and the reference image, which enables our network to handle complex color mapping relationships. Meanwhile, the color mapping sub-network, which is a fully 1×1 convolutional network, guarantees structural consistency between the input and output images. The network design follows the principle of conducting complex computations at low resolutions and simple computations at high resolutions. Specifically, the source encoder, reference encoder, and fusion layer operate at low resolution, but the color mapping sub-network operates at the original resolution. During the running stage, the reference image is pre-set, and the color distribution feature vector of the reference image does not need to be repeatedly computed. These features endow StainPresetNet with high computational efficiency.

![](images/bb3df84547bf7c33ddee9c4d79c540dd74f82d8f29a328f9b5ea0b51256dd332.jpg)  
Fig. 2: Mapping of different training frameworks. (a) Mapping of the CycleGAN-based method. Since it is not possible to determine where a single domain maps to, the mapping from a single domain to multiple domains is uncertain. (b) Mapping of StainPresetNet. In our training framework, the mapping relationship is clear and stable.

## B. Training Framework ofStainPresetNet

Among stain normalization methods, CycleGAN is the mainstream training framework. However, CycleGAN is designed for one-to-one transformation tasks and cannot be applied to multi-to-one and multi-to-multi transformation tasks. The training framework of CycleGAN consists of two transformations: from domain A to domain B and from domain B to domain A. When the CycleGAN training framework is applied to multi-to-one domain transformation, the transformation from many to one is fine, but the transformation from one to many is uncertain as shown in Fig. 2 (a). The uncertainty of the transformation makes the network training process ambiguous and confusing, resulting in reduced performance.

To solve this problem, a new training framework is proposed in this study, in which each domain is equipped with an independent texture module and discriminator, and the texture module and discriminator to be used are determined according to the domain of the reference image. The transformation of StainPresetNet in the multi-to-multi color normalization task is shown in Figure 3 (b). For each pair of source image and reference image, the training framework of StainPresetNet performs the transformation between the two domains, with a clear mapping direction and more stable training.

![](images/9b71bc5182aa34778008b155c5d0f0bec69b4acb139fa4903d6821c482720048.jpg)  
Fig. 3: The training framework of StainPresetNet. A real imgae $x _ { i n } \notin$ $D o m a i n _ { i } , i = 1 , 2 , . . . , N$ is mapped to Domain under the reference of the real image $x _ { r e f } \in D o m a i n _ { i }$ , the texture module $T _ { i } \in D o m a i n _ { i }$ adds some details, and finally the real imgae $x _ { i n } \notin D o m a i n _ { i }$ is reconstructed under the reference of itself. N is the number of domains.

The proposed training framework is shown in Fig. 3, which contains a StainPresetNet as the generator (G), N texture modules $( T _ { i } , i = 1 , 2 , 3 , . . . , N )$ and N discriminators $( D _ { i } , i =$ $1 , 2 , 3 , . . . , N )$ , where N is the number of domains in the training dataset. $G$ can transform the input source image $x _ { i n }$ to the reference image’s under the reference of the reference image $x _ { r e f }$ . The texture module $T _ { i }$ is used to add some texture details belonging to Domain<sub>i</sub> to the output image of $G ,$ so as to deceive the discriminator $D _ { i }$ . The discriminator $D _ { i }$ is used to distinguish the difference between the real reference image $x _ { r e f }$ and the output image of the texture module $T _ { i }$ . Next, the output image of the texture module $T _ { i }$ will be transformed to the domain of $x _ { i n }$ under the reference of $x _ { i n }$ by G.

The proposed framework includes five loss functions: adversarial loss, cycle consistency loss, domain consistency loss, domain structure loss, and identity loss. The overall objective function can be expressed as:

$$
\begin{array} { r l } {  { \mathcal { L } = \sum _ { i = 1 } ^ { N } \Big [ \mathcal { L } _ { G A N } ( G , T _ { i } , D _ { i } ) + \lambda _ { c y c } \mathcal { L } _ { c y c } ( G , T _ { i } ) } } \\ & { \quad + \lambda _ { d o m } \mathcal { L } _ { d o m } ( G , T _ { i } ) + \lambda _ { s t r u c t } \mathcal { L } _ { s t r u c t } ( G , T _ { i } ) } \\ & { \quad + \lambda _ { i d e } \mathcal { L } _ { i d e } ( G , T _ { i } ) \Big ] } \end{array}\tag{1}
$$

where, $\mathcal { L } _ { G A N } ( G , T _ { i } )$ is the adversarial loss, $\mathcal { L } _ { c y c } ( G , T _ { i } )$ is the cycle consistency loss, $\mathcal { L } _ { d o m } ( G , T _ { i } )$ is the domain consistency loss, $\mathcal { L } _ { s t r u c t } ( G , T _ { i } )$ is the domain structure loss, $\mathcal { L } _ { i d e } ( G , T _ { i } )$ is the identity loss, N is the number of domains in the dataset, $\lambda _ { c y c } , \lambda _ { d o m } , \lambda _ { s t r u c t }$ and $\lambda _ { i d e }$ are the weight coefficients of the cycle consistency loss, the domain consistency loss, the domain structure loss and the identity loss, respectively.

The adversarial loss is aimed to ensure that the output of the texture module $T _ { i }$ have the same distribution as the real reference image $x _ { r e f }$ . The objective function is defined as:

$$
\begin{array} { r l } & { \mathcal { L } _ { G A N } ( G , T _ { i } , D _ { i } ) = } \\ & { \mathbb { E } _ { x _ { r e f } \sim p \rho _ { d a t a } ( x _ { r e f } ) } [ \| D _ { i } ( x _ { r e f } ) \| _ { 2 } ] } \\ & { + \mathbb { E } _ { x _ { i n } \sim p _ { d a t a } ( x _ { i n } ) } [ \| 1 - D _ { i } ( T _ { i } ( G ( x _ { i n } | x _ { r e f } ) ) ) \| _ { 2 } ] } \\ & { \mathrm { w h e r e } } \\ & { x _ { r e f } \in \mathrm { D o m a i n } _ { i } , x _ { i n } \notin \mathrm { D o m a i n } _ { i } } \\ & { \| \cdot \| _ { 2 } \mathrm { i s ~ t h e ~  { \ell _ { 2 } } ~ n o r m } } \\ & { x _ { i n } \mathrm { ~ i s ~ t h e ~ i n p u t ~ i m a g e } } \\ & { x _ { r e f } \mathrm { ~ i s ~ t h e ~ r e f e r e n c e ~ i m a g e } } \end{array}\tag{2}
$$

The cycle consistency loss constrains the consistency of image content by reconstructing the transformed image into the original image. The objective function can be expressed as:

$$
\begin{array} { r l } & { \mathcal { L } _ { c y c } ( G , T _ { i } ) = } \\ & { \mathbb { E } _ { x _ { i n } \sim p _ { d a t a } ( x _ { i n } ) } [ | | x _ { i n } - G ( T _ { i } ( G ( x _ { i n } | x _ { r e f } ) ) | x _ { i n } ) | | _ { 1 } ] } \\ & { \mathrm { w h e r e } } \\ & { x _ { r e f } \in \operatorname { D o m a i n } _ { i } , x _ { i n } \not \in \operatorname { D o m a i n } _ { i } } \\ & { \| \cdot \| _ { 1 } \mathrm { ~ i s ~ t h e ~ } \ell _ { 1 } \mathrm { ~ n o r m } } \end{array}\tag{3}
$$

The domain consistency loss is designed that the output image of the generator G and the output image of the texture module $T _ { i }$ are as consistent as possible. The domain consistency loss uses the L1 distance to constrain the difference between the generator output image $G ( x _ { i n } | x _ { r e f } )$ and the texture module output image $T _ { i } ( G ( x _ { i n } | x _ { r e f } ) )$ . The objective function can be expressed as:

$$
\begin{array} { r l } & { \mathcal { L } _ { d o m } ( G , T _ { i } ) = } \\ & { \mathbb { E } _ { { x _ { i n } } \sim { p _ { d a t a } } ( { x _ { i n } } ) } [ | | G ( { x _ { i n } } | { x _ { r e f } } ) - T _ { i } ( G ( { x _ { i n } } | { x _ { r e f } } ) ) | | _ { 1 } ] } \\ & { \mathrm { w h e r e } } \end{array}\tag{4}
$$

$$
x _ { r e f } \in \mathrm { D o m a i n } _ { i } , x _ { i n } \not \in \mathrm { D o m a i n } _ { i }
$$

The domain structure loss is designed that the output image of the generator G and the output image of the texture module $T _ { i }$ are as consistent as possible in image structure. The domain structure loss uses SSIM to constrain the difference between the generator output image $G ( x _ { i n } | x _ { r e f } )$ and the texture module output image $T _ { i } ( G ( x _ { i n } | x _ { r e f } ) )$ . The objective function can be expressed as:

$$
\begin{array} { r l } & { \mathcal { L } _ { d o m } ( G , T _ { i } ) = } \\ & { \mathbb { E } _ { x _ { i n } \sim p _ { d a t a } ( x _ { i n } ) } [ 1 - S S I M ( G ( x _ { i n } | x _ { r e f } ) , T _ { i } ( G ( x _ { i n } | x _ { r e f } ) ) ] } \\ & { \mathrm { w h e r e } } \\ & { x _ { r e f } \in \mathrm { D o m a i n } _ { i } , x _ { i n } \not \in \mathrm { D o m a i n } _ { i } } \end{array}\tag{5}
$$

The identity loss is designed that when the input image comes from the domain of the reference image $x _ { r e f }$ , the generator can maintain the identity mapping without making any changes to the image. Specifically, in this study, when the input of $G$ and $T _ { i }$ is the reference image $x _ { r e f } , G$ and $T _ { i }$ should not make any changes to the input. The objective function can be expressed as:

$$
\begin{array} { r l } & { \mathcal { L } _ { i d e } ( G , T _ { i } ) = \mathbb { E } _ { { x } _ { r e f } \sim p _ { d a t a } ( { x } _ { r e f } ) } [ | | x _ { r e f } - G ( { x } _ { r e f } | { x } _ { r e f } ) | | _ { 1 } ] } \\ & { \phantom { \mathcal { L } } + \mathbb { E } _ { { x } _ { r e f } \sim p _ { d a t a } ( { x } _ { r e f } ) } [ | | x _ { r e f } - T _ { i } ( { x } _ { r e f } ) | | _ { 1 } ] } \\ & { \phantom { \mathcal { L } } \mathrm { w h e r e } } \\ & { \quad \qquad x _ { r e f } \in \mathrm { D o m a i n } _ { i } , { x } _ { i n } \not \in \mathrm { D o m a i n } _ { i } } \end{array}\tag{6}
$$

In our implementation, the network structure of the texture module and discriminator use the generator and discriminator in ParamNet.

## IV. EXPERIMENTS AND RESULTS

## A. Datasets

In this study, we used three different datasets to verify the performance of different methods, including one public datasets and two private datasets. This study was approved by the Ethics Committee of Tongji Medical College, Huazhong University of Science and Technology. The related descriptions are below:

1) Aligned Cytopathology Dataset: In this dataset, the same slides (Thinprep cytologic test slides from Hubei Maternal and Child Health Hospital) were scanned using three different scanners. The first scanner was equipped with a 20x objective lens with a pixel size of 0.2930 um. The second scanner called was equipped with a 20x objective lens with a pixel size of 0.2436 um. The third scanner was equipped with a 40x objective lens with a pixel size of 0.1803 um. The images captured by these three scanners had three different styles, namely S1, S2, and S3. In this dataset, the images from different scanners are carefully aligned. There are 3,028 image pairs in the training set and 1,472 image pairs in the test set. The aligned cytopathology dataset was used to test the performance of the classifier with and without normalization

TABLE I: The number of image patches on the cytopathology and histopathology classification dataset. The classification set was used to train or test the classifier with and without normalization. The normalization set was used to train stain normalization methods.
<table><tr><td>Styles</td><td>Classification Normalization</td><td></td><td>Centers</td><td></td><td>Classification Normalization</td></tr><tr><td>D1</td><td>244,341</td><td>2,500</td><td>Uni16</td><td>40,000</td><td>2,500</td></tr><tr><td>D2</td><td>10,000</td><td>2,500</td><td>C1</td><td>13,613</td><td>2,500</td></tr><tr><td>D3</td><td>10,000</td><td>2,500</td><td>C2</td><td>12,355</td><td>2,500</td></tr><tr><td>D4</td><td>10,000</td><td>2,500</td><td>C3</td><td>14,076</td><td>2,500</td></tr><tr><td>D5</td><td>10,000</td><td>2,500</td><td>C5</td><td>9,696</td><td>2,500</td></tr><tr><td></td><td></td><td></td><td>C5</td><td>9,696</td><td>2,500</td></tr></table>

2) Cytopathology Classification Dataset: The cytopathology classification dataset from ParamNet was used to verify the performance of the stain normalization algorithm on a multidomain cytopathology classification diagnostic task. The slides in the cytopathology classification dataset are from Hubei Provincial Maternal and Child Health Hospital and Tongji Hospital. There are five different styles (D1-D5), among which D1-D4 come from Hubei Maternal and Child Health Hospital, and D5 comes from Tongji Hospital. In this dataset, the image patches that contain abnormal cells are labeled as abnormal patches, and the image patches that do not contain any abnormal cells are labeled as normal patches. There are 244,341 image patches in D1 and 40,000 image patches in D2- D5 (10,000 image patches in each style) as shown in Table I. We randomly picked 2,500 image patches in each style as the normalization set for training the stain normalization algorithm. For the classifier, we used D1 to train the classifier, and used D2-D5 to test the performance of the classifier with and without normalization.

3) Histopathology Classification Dataset: The histopathology classification datasets used the publicly available Camelyon16 [32] (399 whole slide images from two centers) and Camelyon17 [33] datasets (1,000 whole slide images from five centers). In our experiments, 100 WSIs from University Medical Center Utrecht (Uni16) in Camelyon16 training part were used to extract the training patches, and 500 WSIs from the five centers (C1-C5) in Camelyon17 training part were used to extract the test patches. In this dataset, the image patches that contain tumor cells are labeled as abnormal patches, and the image patches that do not contain any tumor cells are labeled as normal patches. There are 40,000 image patches from Uni16 in the training set of this dataset. And there are 58,975 image patches in the test set of this dataset as shown in Table I. We randomly picked 2,500 image patches in each center as the normalization set for training stain normalization algorithms. For the classifier, we used the training set to train the classifier, and used the test set to test the performance of the classifier with and without normalization.

## B. Evaluation Metrics

In order to evaluate different methods fairly and effectively, we considered in four aspects: similarity with target image, preservation of source image information, computational efficiency, and classifier accuracy.

In order to effectively measure the similarity to the target image and preserve the information of the source image, we used two similarity matrices, Structural Similarity index (SSIM) and Peak Signal-to-Noise Ratio (PSNR). The similarity between the normalized image and the target image was evaluated using SSIM and PSNR, denoted as SSIM-T and PSNR-T. The preservation of source image information was evaluated using SSIM between the normalized image and the source image, denoted as SSIM-S. SSIM-T and PSNR-T were calculated using raw RGB values. SSIM-S was calculated using grayscale images.

In order to evaluate the computational efficiency of different methods, the number of frames per second (FPS) was calculated on the system with 6-core Intel(R) Core (TM) i7- 6850K CPU and NVidia GeForce GTX 1080Ti. The input and output (IO) time was not included. The FPS results of different methods are shown in Table III when the input image size is 512×512 pixels.

The accuracy was used to evaluate the classifier performance. The statistic results of accuracy on the cytopathology and histopathology classification datasets are shown in Tables IV and V, as mean ± standard deviation.

## C. Implementation

For conventional methods (Reinhard [12], Macenko [16] and Vahadane [17], one professionally selected image was used as the reference image. For the GAN based methods (StainGAN [25], StainNet [5] and ParamNet [30]), we used their default training settings.

For StainPresetNet, the proposed framework included five loss functions: adversarial loss, cycle consistency loss, domain consistency loss, domain structure loss and identity loss, and the weights of these loss functions were 1.0, 10.0, 10.0, 1.0, and 2.0 respectively. During training, we randomly transformed image resolutions (0.5∼1.0) to stabilize learning. We used Adam to optimize the network, and the learning rate is linearly increased from 0 to 1e-4 in the first 1,000 iterations, and then linearly decreased to 0 until the end of training during 2000k iterations.

For the classifier, we used SqueezeNet [34] pre-trained on ImageNet [35] as the classification network. The classifier was trained using cross entropy loss and Adam optimizer. The initial learning rate was set to 0.0002, and it reduced by a factor of 0.1 at the 40th and 50th epoch. The training was stopped at the 60th epoch, which was chosen experimentally. The experiment was repeated 10 times in order to enhance reliability.

## D. Results

In this section, we compared our method with state-ofart normalization methods, including Reinhard, Macenko, Vahadane, StainGAN, StainNet, and ParamNet.

1) Image Similarity Evaluation: We compared our StainPresetNet with state-of-art normalization methods on the aligned cytopathology dataset. We reported the following results: (1) Visual comparison among different methods. (2) Quantitative comparison of similarities among different methods. (3) The impact of reference images. (4) Quantitative comparison of computational efficiency among different methods.

## Visual comparison

Firstly, we compared our StainPresetNet with other stain normalization methods in the visual appearance. The visual comparison among different methods is shown in Fig. 4. The aligned cytopathology dataset contains three different styles S1, S2, and S3, so we performed normalization in three different directions. The normalized images by the conventional methods (Reinhard, Macenko and Vahadane) has a large gap with the corresponding real target images in brightness, contrast and color. For StainGAN, StainNet, and ParamNet, these methods cannot change the direction of normalization during inference. In order to achieve transformations among multiple styles, these methods were trained multiple times. Even if each style is learned, these methods still cannot achieve the best performance. Especially on S3→S1, the images normalized by these methods were darker than the real S1 images. Our StainPresetNet only needs to be trained once to achieve the transformation between multiple styles, and it is visually closest to the corresponding target image.

## Quantitative comparison of similarities

Secondly, we quantitatively compared our StainPresetNet with other stain normalization methods in the similarity. The results on the aligned cytopathology dataset are shown in Table II, which contains quantitative evaluations in three different normalization directions. The conventional methods (Reinhard, Macenko, and Vahadane) have low SSIM-T and PSNR-T, which means that there is a large difference between the images normalized by the conventional methods and the target images. Our StainPresetNet have highest SSIM-T and PSNR-T in all normalization directions, which means the images normalized by our StainPresetNet is closest to the corresponding target image. In Table II, among the deep learning-based stain normalization methods, StainPresetNet has a relatively high SSIM-S, and ParamNet has the highest SSIM-S. StainGAN and StainNet do not obtain the best matrices in any style, even though they learn each style.

## The impact of reference images

Conventional methods (Reinhard, Macenko, and Vahadane) and our StainPresetNet extract color information from the reference image to determine the direction of color normalization. We verified the impact of reference images on the performance of different methods on the aligned cytopathology dataset. Specifically, for each style, we randomly selected an image as the reference image, and then used this reference image to normalize images of other styles and calculate the PSNR with the target image. This experiment was repeated 10 times, and the results are shown in Fig. 5.

As shown in Fig. 5, the reference image has a great impact on the performance of the conventional methods (Reinhard, Macenko, and Vahadane), which shows that the conventional methods is very dependent on tthe reference image. For our method, the reference image has little impact and the performance is significantly better than the conventional method, which shows that our method can extract accurate color distribution information from a single reference image.

## Computational efficiency

S2→S1

S1→S2

![](images/6d0f6abf667b42998852252abd977e9fe3fef5d298f67049449fd84dcfe6cd62.jpg)  
S3→S1  
S3→S2  
S1→S3  
S2→S3

Fig. 4: Visual comparison among different methods on the aligned cytopathology dataset. To achieve transformations among multiple styles, StainGAN, StainNet, and ParamNet were trained multiple times, but our StainPresetNet only was trained once.  
TABLE II: Different evaluation metrics are reported for various stain normalization methods on the aligned cytopathology dataset.To achieve transformations among multiple styles, StainGAN, StainNet, and ParamNet were trained multiple times, but our StainPresetNet only was trained once.
<table><tr><td rowspan="2">Methods</td><td colspan="4">To S1</td><td colspan="4">To S2 To S3</td></tr><tr><td>SSIM-T</td><td>PSNR-T</td><td>SSIM-S</td><td>SSIM-T</td><td>PSNR-T</td><td>SSIM-S</td><td>SSIM-T</td><td>PSNR-T</td><td>SSIM-S</td></tr><tr><td>Reinhard</td><td>0.901±0.031</td><td>24.0±3.6</td><td>0.920±0.049</td><td>0.908±0.033</td><td>24.5±5.1</td><td>0.931±0.038</td><td>0.844±0.032</td><td>23.5±4.8</td><td>0.970±0.019</td></tr><tr><td>Macenko</td><td>0.896±0.030</td><td>23.6±2.6</td><td>0.946±0.035</td><td>0.884±0.036</td><td>22.8±2.4</td><td>0.956±0.029</td><td>0.830±0.039</td><td>22.5±2.1</td><td>0.933±0.032</td></tr><tr><td>Vahadane</td><td>0.896±0.034</td><td>23.8±2.8</td><td>0.945±0.036</td><td>0.888±0.034</td><td>23.0±2.7</td><td>0.944±0.036</td><td>0.831±0.043</td><td>22.6±2.1</td><td>0.939±0.038</td></tr><tr><td>StainGAN</td><td>0.891±0.023</td><td>28.4±2.5</td><td>0.910±0.021</td><td>0.901±0.026</td><td>28.5±2.8</td><td>0.915±0.017</td><td>0.830±0.021</td><td>27.6±2.2</td><td>0.911±0.022</td></tr><tr><td>StainNet</td><td>0.916±0.022</td><td>28.4±2.7</td><td>0.925±0.021</td><td>0.924±0.023</td><td>29.4±3.1</td><td>0.941±0.011</td><td>0.870±0.022</td><td>27.6±2.3</td><td>0.959±0.013</td></tr><tr><td>ParamNet</td><td>0.919±0.020</td><td>27.9±2.8</td><td>0.960±0.014</td><td>0.922±0.023</td><td>28.8±3.2</td><td>0.972±0.010</td><td>0.871±0.022</td><td>29.3±2.8</td><td>0.970±0.014</td></tr><tr><td>StainPresetNet</td><td>0.922±0.020</td><td>30.1±2.5</td><td>0.952±0.012</td><td>0.929±0.022</td><td>30.4±3.3</td><td>0.950±0.007</td><td>0.874±0.021</td><td>29.8±2.9</td><td>0.962±0.013</td></tr></table>

![](images/09f160aed9227b10c8549aaa3ce72e7bb12073e9161f0f0fbc84207c27dc06ef.jpg)  
(a) S2 and S3 to S1

![](images/6510f626c32b987210a70cec01ddbde1447d7cddd1b83935ae95617cae9c1ed4.jpg)  
(b) S1 and S3 to S2

![](images/e8e1adfb385802b73c3279f34ab0a866abf0d75947bf8bbcde267db712ebcbad.jpg)  
(c) S1 and S2 to S3  
Fig. 5: The impact of reference images on the performance of different methods. Ten images of each style were randomly selected as reference images on the aligned cytopathology dataset.

TABLE III: Quantitative comparison of computational efficiency among different methods. The input and output (IO) time is not included. The input image size is 512×512 pixels.
<table><tr><td>Methods</td><td>Reinhard</td><td>Macenko</td><td>Vahadane</td><td>StainGAN</td><td>StainNet</td><td>ParamNet</td><td>StainPresetNet</td></tr><tr><td>FPS</td><td>54.8</td><td>4.0</td><td>0.5</td><td>19.6</td><td>881.8</td><td>1,605.2</td><td>1,601.5</td></tr></table>

In this section, the computational efficiency of the proposed method is quantitatively evaluated compared with other color normalization methods. The FPS of different methods is tested when the input image size is 512×512, and the input and output time of the data is not included. The results are calculated on a system equipped with an Intel i7-6850K CPU and an NVIDIA GeForce GTX 1080Ti graphics card and are shown in Table III. As shown in Table III, the proposed StainPresetNet (1,601.5 FPS) is almost as fast as ParamNet (1,605.2 FPS). This is because the color distribution feature vector of the reference image is pre-computed and does not take up computing time. Compared with ParamNet, the additional computation comes from the fusion layer. In our StainPresetNet, the fusion layer has only two fully connected layers, one of which has a 118-dimensional feature vector as input and output, and the other is a fully connected layer with a 118-dimensional input and a 59-dimensional output. Therefore, the fusion layer does not introduce too much computation, and the StainPresetNet proposed in this paper is almost as fast as ParamNet.

## E. Patch-Level Classification Evaluation

In this section, the proposed StainPresetNet and other stain normalization methods are evaluated on patch-level classification tasks on datasets from multiple sources. SqueezeNet pre-trained on ImageNet is used as the classifier because of its small size and relatively high accuracy. The accuracy of the classifier without normalization and with various normalization methods is shown in Table IV and Table V.

The accuracy of the classifier without normalization and with various normalization methods on the cytopathology classification dataset is shown in Table IV. As shown in Table IV, the performance of the classifier on the original images of D2-D5 is poor, and stain normalization can effectively improve the performance of the classifier. The classifier performs better on images normalized by deep learning-based methods than conventional methods. Among deep learning-based methods, the classifier has the highest accuracy on the images normalized by StainPresetNet, with an average of 0.951.

The accuracy without normalization and with various normalization methods on the histopathology classification dataset is shown in Table V. As shown in Table V, the classifier performs poorly on the original images of C1-C5. Using stain normalization can significantly improve the accuracy of the classifier, among which the deep learning-based methods are significantly better than the conventional methods. In Table V, our StainPresetNet has the best average accuracy of 0.907. The images in C3 and Uni16 are from the same medical center but at different times. For the classifier trained on Uni16, the accuracy is 0.907 on the original images of C3. And the accuracy can be increased to 0.919 on the normalized images by our StainPresetNet, which shows that our method can improve the accuracy of the classifier, even on images from the same center.

## F. WSI-Level Classification Evaluation

In this section, we evaluate the proposed StainPresetNet and other stain normalization methods on the whole slide image level (WSI level) classification task on Camelyon16 and Camelyon17 datasets. The training dataset contained 60 negative whole slide images and 40 positive whole slide images from Camelyon16, University Medical Center Utrecht (Uni16). The testing dataset contained 50 positive whole slide images from Camelyon17 (10 whole slide images from each of the 5 centers) to test the performance of the classifier with and without normalization. The classifier predicts the tissue area of the whole slide image in a sliding window manner to obtain a probability heat map.

Quantitative evaluations on C1-C5 using different normalization methods are shown in Table VI. As can be seen from Table VI, the classifier performs very poorly on unnormalized WSIs, with an F1 score of 0.000 at C1, C2, and C4. Classifier performance can be significantly improved using normalization methods. Among the normalization methods in TableVI, StainGAN failed to obtain the best performance at any center, StainNet could obtain the highest F1 score on C1, ParamNet could obtain the best F1 score on C3, and StainPresetNet Best F1 scores were achieved on C2, C4 and C5. In addition, for the average metrics on the C1-C5, StainPresetNet has obtained all the best metrics, which shows that StainPresetNet is more stable and has best performance. In particular, the F1 score of StainPresetNet is 8.0% higher than the second place.

TABLE IV: The accuracy for various stain normalization methods on the cytopathology classification dataset.
<table><tr><td>Accuracy</td><td>D2</td><td>D3</td><td>D4</td><td>D5</td><td>Average</td></tr><tr><td>Original</td><td>0.853±0.031</td><td>0.915±0.022</td><td>0.873±0.029</td><td>0.688±0.033</td><td>0.832±0.029</td></tr><tr><td>Reinhard</td><td>0.862±0.015</td><td>0.867±0.025</td><td>0.796±0.026</td><td>0.728±0.025</td><td>0.813±0.023</td></tr><tr><td>Macenko</td><td>0.929±0.014</td><td>0.937±0.023</td><td>0.915±0.010</td><td>0.818±0.034</td><td>0.900±0.020</td></tr><tr><td>Vahadane</td><td>0.905±0.022</td><td>0.940±0.031</td><td>0.907±0.019</td><td>0.745±0.037</td><td>0.875±0.026</td></tr><tr><td>StainGAN</td><td>0.918±0.007</td><td>0.908±0.011</td><td>0.867±0.006</td><td>0.799±0.018</td><td>0.873±0.011</td></tr><tr><td>StainNet</td><td>0.928±0.034</td><td>0.963±0.014</td><td>0.924±0.032</td><td>0.820±0.059</td><td>0.909±0.035</td></tr><tr><td>ParamNet</td><td>0.968±0.013</td><td>0.965±0.017</td><td>0.943±0.006</td><td>0.867±0.039</td><td>0.936±0.019</td></tr><tr><td>StainPresetNet</td><td>0.971±0.010</td><td>0.976±0.011</td><td>0.950±0.003</td><td>0.907±0.035</td><td>0.951±0.015</td></tr></table>

TABLE V: The accuracy for various stain normalization methods on the histopathology classification dataset. C3 and Uni16 are from the same medical center, but at different times.
<table><tr><td>Accuracy</td><td>C1</td><td>C2</td><td>C3</td><td>C4</td><td>C5</td><td>Average</td></tr><tr><td>Original</td><td>0.500±0.016</td><td>0.561±0.030</td><td>0.907±0.003</td><td>0.528±0.014</td><td>0.765±0.041</td><td>0.652±0.021</td></tr><tr><td>Reinhard</td><td>0.871±0.014</td><td>0.855±0.008</td><td>0.815±0.006</td><td>0.823±0.027</td><td>0.808±0.015</td><td>0.834±0.014</td></tr><tr><td>Macenko</td><td>0.814±0.009</td><td>0.832±0.009</td><td>0.816±0.007</td><td>0.836±0.012</td><td>0.789±0.010</td><td>0.817±0.009</td></tr><tr><td>Vahadane</td><td>0.817±0.005</td><td>0.826±0.005</td><td>0.762±0.007</td><td>0.890±0.004</td><td>0.805±0.003</td><td>0.820±0.005</td></tr><tr><td>StainGAN</td><td>0.898±0.004</td><td>0.871±0.001</td><td>0.921±0.003</td><td>0.903±0.004</td><td>0.881±0.005</td><td>0.895±0.003</td></tr><tr><td>StainNet</td><td>0.898±0.006</td><td>0.836±0.006</td><td>0.895±0.005</td><td>0.862±0.019</td><td>0.877±0.005</td><td>0.874±0.008</td></tr><tr><td>ParamNet</td><td>0.920±0.003</td><td>0.857±0.007</td><td>0.924±0.003</td><td>0.899±0.004</td><td>0.884±0.003</td><td>0.897±0.004</td></tr><tr><td>StainPresetNet</td><td>0.924±0.002</td><td>0.884±0.005</td><td>0.919±0.003</td><td>0.914±0.003</td><td>0.895±0.001</td><td>0.907±0.003</td></tr></table>

TABLE VI: Comparison of different stain normalization methods on the histopathological WSIs from different medical centers. C3 and Uni16 are from the same medical center, but at different times.
<table><tr><td>Centers</td><td>Methods</td><td>Precision</td><td>Sensitivity</td><td>Specificity</td><td>Accuracy</td><td>F1 score</td><td>AUC</td></tr><tr><td>C1</td><td>Original</td><td>0.000</td><td>0.000</td><td>1.000</td><td>0.998</td><td>0.000</td><td>0.500</td></tr><tr><td>Cl</td><td>StainGAN</td><td>0.300</td><td>0.840</td><td>0.995</td><td>0.995</td><td>0.442</td><td>0.932</td></tr><tr><td>C1</td><td>StainNet</td><td>0.709</td><td>0.822</td><td>0.999</td><td>0.999</td><td>0.761</td><td>0.905</td></tr><tr><td>Cl</td><td>ParamNet</td><td>0.372</td><td>0.858</td><td>0.997</td><td>0.996</td><td>0.519</td><td>0.929</td></tr><tr><td>Cl</td><td>StainPresetNet</td><td>0.591</td><td>0.842</td><td>0.999</td><td>0.998</td><td>0.695</td><td>0.917</td></tr><tr><td>C2</td><td>Original</td><td>0.000</td><td>0.000</td><td>1.000</td><td>0.998</td><td>0.000</td><td>0.500</td></tr><tr><td>C2</td><td>StainGAN</td><td>0.050</td><td>0.753</td><td>0.977</td><td>0.977</td><td>0.094</td><td>0.875</td></tr><tr><td>C2</td><td>StainNet</td><td>0.111</td><td>0.477</td><td>0.994</td><td>0.993</td><td>0.180</td><td>0.747</td></tr><tr><td>C2</td><td>ParamNet</td><td>0.128</td><td>0.627</td><td>0.993</td><td>0.993</td><td>0.213</td><td>0.814</td></tr><tr><td>C2</td><td>StainPresetNet</td><td>0.196</td><td>0.668</td><td>0.996</td><td>0.995</td><td>0.303</td><td>0.834</td></tr><tr><td>C3</td><td>Original</td><td>0.770</td><td>0.819</td><td>0.999</td><td>0.998</td><td>0.794</td><td>0.929</td></tr><tr><td>C3</td><td>StainGAN</td><td>0.767</td><td>0.894</td><td>0.998</td><td>0.998</td><td>0.825</td><td>0.952</td></tr><tr><td>C3</td><td>StainNet</td><td>0.630</td><td>0.863</td><td>0.997</td><td>0.996</td><td>0.728</td><td>0.936</td></tr><tr><td>C3</td><td>ParamNet</td><td>0.832</td><td>0.854</td><td>0.999</td><td>0.998</td><td>0.843</td><td>0.933</td></tr><tr><td>C3</td><td>StainPresetNet</td><td>0.817</td><td>0.861</td><td>0.999</td><td>0.998</td><td>0.839</td><td>0.939</td></tr><tr><td>C4</td><td>Original</td><td>0.007</td><td>0.000</td><td>1.000</td><td>0.994</td><td>0.000</td><td>0.500</td></tr><tr><td>C4</td><td>StainGAN</td><td>0.348</td><td>0.812</td><td>0.991</td><td>0.990</td><td>0.487</td><td>0.932</td></tr><tr><td>C4</td><td>StainNet</td><td>0.793</td><td>0.122</td><td>1.000</td><td>0.995</td><td>0.212</td><td>0.593</td></tr><tr><td>C4</td><td>ParamNet</td><td>0.672</td><td>0.838</td><td>0.998</td><td>0.997</td><td>0.746</td><td>0.925</td></tr><tr><td>C4</td><td>StainPresetNet</td><td>0.713</td><td>0.861</td><td>0.998</td><td>0.997</td><td>0.780</td><td>0.930</td></tr><tr><td>C5</td><td>Original</td><td>0.465</td><td>0.874</td><td>0.982</td><td>0.980</td><td>0.607</td><td>0.953</td></tr><tr><td>C5</td><td>StainGAN</td><td>0.606</td><td>0.662</td><td>0.992</td><td>0.987</td><td>0.633</td><td>0.907</td></tr><tr><td>C5</td><td>StainNet</td><td>0.663</td><td>0.836</td><td>0.992</td><td>0.990</td><td>0.739</td><td>0.949</td></tr><tr><td>C5</td><td>ParamNet</td><td>0.552</td><td>0.913</td><td>0.987</td><td>0.985</td><td>0.688</td><td>0.977</td></tr><tr><td>C5</td><td>StainPresetNet</td><td>0.837</td><td>0.935</td><td>0.997</td><td>0.996</td><td>0.883</td><td>0.983</td></tr><tr><td>Average</td><td>Original</td><td>0.209</td><td>0.282</td><td>0.997</td><td>0.994</td><td>0.234</td><td>0.647</td></tr><tr><td>Average</td><td>StainGAN</td><td>0.397</td><td>0.739</td><td>0.991</td><td>0.989</td><td>0.476</td><td>0.904</td></tr><tr><td>Average</td><td>StainNet</td><td>0.565</td><td>0.562</td><td>0.997</td><td>0.994</td><td>0.492</td><td>0.801</td></tr><tr><td>Average</td><td>ParamNet</td><td>0.495</td><td>0.773</td><td>0.995</td><td>0.993</td><td>0.580</td><td>0.901</td></tr><tr><td>Average</td><td>StainPresetNet</td><td>0.588</td><td>0.795</td><td>0.997</td><td>0.996</td><td>0.660</td><td>0.906</td></tr></table>

## V. DISCUSSION AND CONCLUSION

In this paper, we proposed a novel stain normalization method named StainPresetNet, which achieves accurate mapping relationships, high computational efficiency, structure preservation, and multi-domain applicability. StainPresetNet extracts color mapping relationships from entire datasets, normalizes on a pixel-by-pixel manner, and utilizes a preset reference image to guide the normalization direction, all while maintaining high computational efficiency.

Our proposed StainPresetNet can achieve the state-of-art performance on multiple style mutual transformation tasks, multi-center image patch-level classification tasks and multicenter WSI-level classification tasks. Our StainPresetNet can process 1,601.5 512×512 images per second using a single 1080ti graphics card, which means that for a 100,000×100,000 whole slide image, it only takes about 25 seconds to process. These results prove that StainPresetNet has the ability to be applied in CAD systems.

## REFERENCES

[1] M. Salvi, F. Molinari, U. R. Acharya, L. Molinaro, and K. M. Meiburger, “Impact of stain normalization and patch selection on the performance of convolutional neural networks in histological breast and prostate cancer classification,” Computer Methods and Programs in Biomedicine Update, vol. 1, p. 100004, 2021.

[2] S. Roy, A. kumar Jain, S. Lal, and J. Kini, “A study about color normalization methods for histopathology images,” Micron, vol. 114, pp. 42–61, 2018.

[3] T. A. A. Tosta, P. R. de Faria, L. A. Neves, and M. Z. do Nascimento, “Computational normalization of hande-stained histological images: progress, challenges and future potential,” Artificial Intelligence in Medicine, vol. 95, pp. 118–132, 2019.

[4] Y. Zheng, Z. Jiang, H. Zhang, F. Xie, D. Hu, S. Sun, J. Shi, and C. Xue, “Stain standardization capsule for application-driven histopathological image normalization,” IEEE Journal of Biomedical and Health Informatics, vol. 25, no. 2, pp. 337–347, 2020.

[5] H. Kang, D. Luo, W. Feng, S. Zeng, T. Quan, J. Hu, and X. Liu, “Stainnet: a fast and robust stain normalization network,” Frontiers in Medicine, vol. 8, p. 746307, 2021.

[6] N. Zhou, D. Cai, X. Han, and J. Yao, “Enhanced cycle-consistent generative adversarial network for color normalization of hande stained images,” in Medical Image Computing and Computer Assisted Intervention (MICCAI), (Shenzhen, China, 13-17 Oct. 2019), pp. 694–702, Springer, 2019.

[7] D. Mahapatra, B. Bozorgtabar, J. Thiran, and L. Shao, “Structure preserving stain normalization of histopathology images using self supervised semantic guidance,” in Medical Image Computing and Computer Assisted Intervention (MICCAI), (Virtual Conference, 4-8 Oct. 2020), pp. 309–319, Springer, 2020.

[8] A. Shrivastava, W. Adorno, Y. Sharma, L. Ehsan, S. A. Ali, S. R. Moore, B. Amadi, P. Kelly, S. Syed, and D. E. Brown, “Self-attentive adversarial stain normalization,” in International Conference on Pattern Recognition (ICPR), (Milan, Italy, 10-15 Jan. 2021), pp. 120–140, Springer, 2021.

[9] Y. Han, G. Huang, S. Song, L. Yang, H. Wang, and Y. Wang, “Dynamic neural networks: a survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 11, pp. 7436–7456, 2021.

[10] J. Zhou, V. Jampani, Z. Pi, Q. Liu, and M. Yang, “Decoupled dynamic filter networks,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), (Nashville, TN, USA, 20-25 Jun. 2021), pp. 6647–6656, IEEE, 2021.

[11] D. Coltuc, P. Bolon, and J. Chassery, “Exact histogram specification,” IEEE Transactions on Image Processing, vol. 15, no. 5, pp. 1143–1152, 2006.

[12] E. Reinhard, M. Adhikhmin, B. Gooch, and P. Shirley, “Color transfer between images,” IEEE Computer Graphics and Applications, vol. 21, no. 5, pp. 34–41, 2001.

[13] A. C. Ruifrok, D. A. Johnston, et al., “Quantification of histochemical staining by color deconvolution,” Analytical and Quantitative Cytology and Histology, vol. 23, no. 4, pp. 291–299, 2001.

[14] A. M. Khan, N. Rajpoot, D. Treanor, and D. Magee, “A nonlinear mapping approach to stain normalization in digital histopathology images using image-specific color deconvolution,” IEEE transactions on Biomedical Engineering, vol. 61, no. 6, pp. 1729–1738, 2014.

[15] N. Alsubaie, N. Trahearn, S. E. A. Raza, D. Snead, and N. M. Rajpoot, “Stain deconvolution using statistical analysis of multi-resolution stain colour representation,” Plos One, vol. 12, no. 1, p. e0169875, 2017.

[16] M. Macenko, M. Niethammer, J. S. Marron, D. Borland, J. T. Woosley, X. Guan, C. Schmitt, and N. E. Thomas, “A method for normalizing histology slides for quantitative analysis,” in International Symposium on Biomedical Imaging (ISBI), (Boston, MA, USA, 28 Jun-1 Jul 2009), pp. 1107–1110, IEEE, 2009.

[17] A. Vahadane, T. Peng, A. Sethi, S. Albarqouni, L. Wang, M. Baust, K. Steiger, A. M. Schlitter, I. Esposito, and N. Navab, “Structurepreserving color normalization and sparse stain separation for histological images,” IEEE Transactions on Medical Imaging, vol. 35, no. 8, pp. 1962–1971, 2016.

[18] H. Nishar, N. Chavanke, and N. Singhal, “Histopathological stain transfer using style transfer network with adversarial loss,” in Medical Image Computing and Computer Assisted Intervention (MICCAI), (Virtual Conference, 4-8 Oct. 2020), pp. 330–340, Springer, 2020.

[19] B. Zhao, C. Han, X. Pan, J. Lin, Z. Yi, C. Liang, X. Chen, B. Li, W. Qiu, D. Li, et al., “Restainnet: a self-supervised digital re-stainer for stain normalization,” Computers and Electrical Engineering, vol. 103, p. 108304, 2022.

[20] D. Tellez, G. Litjens, P. Bandi, W. Bulten, J. Bokhorst, F. Ciompi, and´ J. Van Der Laak, “Quantifying the effects of data augmentation and stain color normalization in convolutional neural networks for computational pathology,” Medical Image Analysis, vol. 58, p. 101544, 2019.

[21] F. G. Zanjani, S. Zinger, B. E. Bejnordi, J. A. van der Laak, and P. H. de With, “Stain normalization of histopathology images using generative adversarial networks,” in International Symposium on Biomedical Imaging (ISBI), (Washington, DC, USA, 4-7 Apr. 2018), pp. 573–577, IEEE, 2018.

[22] K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” ArXiv, vol. 1409, p. e1556, 2014.

[23] A. BenTaieb and G. Hamarneh, “Adversarial stain transfer for histopathology image analysis,” IEEE Transactions on Medical Imaging, vol. 37, no. 3, pp. 792–802, 2017.

[24] J. Zhu, T. Park, P. Isola, and A. A. Efros, “Unpaired image-to-image translation using cycle-consistent adversarial networks,” in Proceedings of the IEEE International Conference on Computer Vision (ICCV), (Venice, Italy, 22-29 Oct. 2017), pp. 2223–2232, IEEE, 2017.

[25] M. T. Shaban, C. Baur, N. Navab, and S. Albarqouni, “Staingan: stain style transfer for digital histological images,” in International Symposium on Biomedical Imaging (ISBI), (Venice, Italy, 8-11 Apr. 2019), pp. 953–956, IEEE, 2019.

[26] M. Runz, D. Rusche, S. Schmidt, M. R. Weihrauch, J. Hesser, and C. Weis, “Normalization of he-stained histological images using cycle consistent generative adversarial networks,” Diagnostic Pathology, vol. 16, no. 1, pp. 1–10, 2021.

[27] S. Cai, Y. Xue, Q. Gao, M. Du, G. Chen, H. Zhang, and T. Tong, “Stain style transfer using transitive adversarial networks,” in Machine Learning for Medical Image Reconstruction (MLMIR) (F. Knoll, A. Maier, D. Rueckert, and J. C. Ye, eds.), (Shenzhen, China, 13-17 Oct. 2019), pp. 163–172, Springer, 2019.

[28] Y. Choi, M. Choi, M. Kim, J. Ha, S. Kim, and J. Choo, “Stargan: unified generative adversarial networks for multi-domain image-toimage translation,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), (Salt Lake City, USA, 18-22 Jun. 2018), pp. 8789–8797, IEEE, 2018.

[29] H. Nazki, O. Arandjelovic, I. Um, and D. Harrison, “Multipath-´ gan: structure preserving stain normalization using unsupervised multidomain adversarial network with perception loss,” ArXiv, vol. 2204, p. e09782, 2022.

[30] H. Kang, D. Luo, L. Chen, J. Hu, S. Cheng, T. Quan, S. Zeng, and X. Liu, “Paramnet: a parameter-variable network for fast stain normalization,” arXiv preprint arXiv:2305.06511, 2023.

[31] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition (CVPR), (Las Vegas, NV, USA, 27-30 Jun. 2016), pp. 770–778, IEEE, 2016.

[32] B. E. Bejnordi, M. Veta, P. J. Van Diest, B. Van Ginneken, N. Karssemeijer, G. Litjens, J. A. Van Der Laak, M. Hermsen, Q. F. Manson, M. Balkenhol, et al., “Diagnostic assessment of deep learning algorithms for detection of lymph node metastases in women with breast cancer,” Jama, vol. 318, no. 22, pp. 2199–2210, 2017.

[33] P. Bandi, O. Geessink, Q. Manson, M. Van Dijk, M. Balkenhol, M. Hermsen, B. E. Bejnordi, B. Lee, K. Paeng, A. Zhong, et al., “From detection of individual metastases to classification of lymph node status at the patient level: the camelyon17 challenge,” IEEE Transactions on Medical Imaging, vol. 38, no. 2, pp. 550–560, 2018.

[34] F. N. Iandola, S. Han, M. W. Moskewicz, K. Ashraf, W. J. Dally, and K. Keutzer, “Squeezenet: alexnet-level accuracy with 50x fewer parameters and¡ 0.5 mb model size,” ArXiv, vol. 1602, p. e07360, 2016.

[35] J. Deng, W. Dong, R. Socher, L. Li, K. Li, and L. FeiFei, “Imagenet: a large-scale hierarchical image database,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), (Miami, FL, 20-25 Jun. 2009), pp. 248–255, IEEE, 2009.