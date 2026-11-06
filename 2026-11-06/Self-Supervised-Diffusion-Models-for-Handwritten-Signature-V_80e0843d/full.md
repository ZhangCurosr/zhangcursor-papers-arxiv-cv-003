# Self-Supervised Diffusion Models for Handwritten Signature Verification via Ruin Reconstruction

Peirong Zhang, Zhenhua Yang, Jiaxin Zhang, Moises Diaz, Miguel A. Ferrer,

Umapada Pal, Senior Member, IEEE, Wei Zhou, Senior Member, IEEE, Lianwen Jin<sup>∗</sup> Member, IEEE

Abstract—Handwritten signature verification (HSV) is a critical biometric technique in legal and forensic authentication, whose efficacy hinges on the discriminability of signature representations. Current HSV methods predominantly adopt supervised learning to enhance representation quality, leaving selfsupervised learning largely unexplored. We propose SigDiffuser, a diffusion-based self-supervised framework for signature representation and verification. At its core, SigDiffuser introduces a novel pretext task termed Reconstruction from Ruins (RFR), which deliberately “ruins” signatures through severe crumpling, obliterating their structural integrity and writing styles. A visual encoder is employed to capture crucial handwriting patterns from intact signature regions, guiding the denoiser to reconstruct the “ruined” portions and learning meaningful representations. Additionally, we introduce a pyramidal feature aggregation module (PFAM) to bridge the visual encoder and denoiser, enhancing representation quality via hierarchical aggregation and modulation. Extensive experiments on multilingual signature datasets demonstrate that SigDiffuser significantly outperforms both stateof-the-art SSL methods and HSV models. We further show that increasing the crumpling severity on signatures effectively improves representation quality. Notably, our work demonstrates that extreme destruction, coupled with self-supervised diffusion models, can yield superior signature representation compared to traditional mild deformations, paving new avenues for HSV and other biometric domains. Code will be publicly available.

Index Terms—Offline signature verification, Biometrics, Selfsupervised learning, Diffusion models

## I. INTRODUCTION

H ANDWRITTEN signature verification (HSV) has been a longstanding and critical biometric means for distinguishing genuine signatures from forgeries [1]. Depending on signature’s properties, this technique can be broadly categorized into online and offline forms. Online HSV [2, 3] processes dynamically captured digital signatures, while the offline counterpart [4, 5] analyzes static signatures scanned or photographed from paper. Despite the growing prevalence of electronic signing devices, the traditional use of offline signatures on physical documents is still prevalent across various domains such as financial transactions, legal proceedings, and government administration. Consequently, offline HSV remains crucial for ensuring the security and trustworthiness of these applications. In this paper, we focus on offline HSV.

Over the past few decades, the growing demand for realworld HSV has spurred substantial efforts to enhance the reliability of HSV systems. Mainstream methods involve feature representation extraction followed by similarity measurement, in which learning high-quality representations is the key to building a reliable HSV system. Supervised learning techniques have been intensively studied to enhance feature extraction, such as local pattern learning with CNN [6] and global contextual modeling using Transformer [5]. However, the surge of self-supervised representation learning in other biometric domains [7, 8] presents a promising avenue for HSV, particularly given the limited availability of HSV data due to the challenge of collecting skilled forgeries. Despite its potential, self-supervised learning remains largely shadowed in the HSV domain, with only a few studies delving into its effectiveness thus far [9, 10].

Generative representation learning is currently one of the dominant self-supervised learning paradigms in computer vision, as opposed to contrastive learning [11]. Following the principle of “comprehension by generation”, it typically reconstructs corrupted (masked/augmented) images to extract transferable feature representations for downstream tasks. Masked image modeling (MIM), standing as a prominent exemplar within this paradigm, has achieved remarkable success across myriad vision tasks [12, 13]. However, as exemplified in Fig. 1 (a), MIM struggles to accurately restore fine-grained textures and strokes within signatures, yielding only blurred edges and oversimplified structures. This suggests MIM is a suboptimal pretext task for learning effective signature representation.

Beyond MIM, diffusion-based approaches [14, 15, 16, 17] have emerged as a potent force for generative representation learning. Diffusion models progressively add noise to corrupt the clean image into pure Gaussian noise in a forward process, then generate targets via backward denoising. Recent studies [14, 16, 18] exploit visual encoders alongside the denoiser to extract latent codes from auxiliary images, aiming to assist synthesis and facilitate representation learning. Intuitively, akin to how one discerns the genuineness and authorship of a partially crumpled signature from its visible regions, the corruption process of diffusion models parallels the physical crumpling of a signature, while the visual encoders’ latent encoding closely resembles human visual perception of intact regions. This congruence offers a compelling routine for signature representation learning, in which the diffusion model can be tasked with reconstructing the “crumpled” part through iterative denoising, guided by the visual features of “uncrumpled” details extracted by an additional visual encoder. This approach allows the visual encoder to perceive individualized signature patterns and encode them into latent features, achieving representation learning. Furthermore, the diffusion model’s accurate reconstruction of signature details facilitates global visual perception, thereby refining representation quality.

![](images/3e39192c587ec50c2a69b687e8f2c4a4c12df51fb9683b145cc1047163db79dc.jpg)

![](images/d81773ad80612eaeaa8578ccbef44b7e96e4e316ad2fc7cba15bda8a0af75892.jpg)  
(a) Masked Image Modeling (MIM)

![](images/e27d23f5ff8e04f770a82adba22397c0db5bda0715e66f712fc2039b5ec5fb10.jpg)  
(b) Reconstruction from Ruins (RFR)  
Fig. 1: Qualitative and quantitative (Structural Similarity Index (SSIM)) comparisons between MIM and our proposed RFR. (a) MIM exhibits blurred, oversimplified structures that fail to recover complex strokes, resulting in suboptimal representations. (b) In contrast, RFR faithfully reconstructs stroke details and character structures despite severe crumpling, achieving higher SSIM metrics. Although not exact replicas, the fidelity suffices for capturing effective signature representation.

Motivated by this insight, we propose SigDiffuser, a novel diffusion framework that leverages a visual encoder to condition a diffusion denoiser for signature representation learning. SigDiffuser introduces two key innovations. First, we propose Reconstruction from Ruins (RFR; Fig. 1 (b)), a novel pretext task that reconstructs a severely corrupted signature region using a spatially distinct clean portion. Unlike existing pretext tasks that employ mild deformation (e.g., color jittering, image rotation) [14, 15, 16, 19], RFR inflicts an unprecedented level of corruption, i.e., crumpling, upon signatures, obliterating their text structures and writing styles into a “ruined” state. By compelling SigDiffuser to restore these “ruins”, RFR incentivizes the visual encoder to perceive sufficient writing traits of the clean portion to aid in reconstruction, thereby enhancing latent representation quality. Second, existing methods [14, 16] bridge the encoder and denoiser with a single latent vector, limiting representation capacity. We address this by introducing a pyramidal feature aggregation module (PFAM) that aggregates multi-scale encoder features into hierarchical latent representations. These features are injected into corresponding denoiser layers for fine-grained modulation, empowering the encoder to learn more informative representations.

We adopt HSV as the downstream task and evaluate SigDiffuser on multiple HSV datasets spanning different languages, including BHSig260-B (Bengali) [20], BHSig260-H (Hindi) [20], HanSig (Chinese) [21], UTSig (Persian) [22], and CEDAR (English) [23]. Extensive experiments demonstrate that SigDiffuser significantly outperforms not only existing self-supervised representation learning methods but also stateof-the-art (SOTA) HSV models. We further observe that increasing the deformation degree enhances representation quality, substantiating the effectiveness of applying severe destruction within the RFR task.

Our main contributions are summarized as follows:

• We propose SigDiffuser, a pioneering framework that integrates diffusion models with self-supervised learning for signature representation. It leverages a visual encoder to capture unique writing patterns from intact signature regions, enabling effective reconstruction of severely crumpled areas and learning informative representations.

• We propose a novel Reconstruction from Ruins (RFR) pretext task tailored for signature representation learning. RFR applies severe signature crumpling rather than mild corruptions, effectively incentivizing representation learning by reconstructing severely crumpled signatures.

• We introduce a pyramidal feature aggregation module (PFAM) to strengthen representation learning. Through multi-scale feature aggregation and hierarchical modulation, PFAM enables SigDiffuser to extract richer latent representations than single latent embedding approaches.

• Experiments demonstrate SigDiffuser’s superior performance over both existing representation learning methods and SOTA HSV models. Furthermore, we reveal that increasing the crumpling severity improves representation quality, substantiating the effectiveness of the RFR task.

## II. RELATED WORK

## A. Generative Representation Learning

1) Non-Diffusion-Based Methods: Early generative attempts mainly exploit AutoEncoders and Generative Adversarial Networks (GAN) [24] as latent feature learners. Inspired by masked language modeling, the focus transitions to masked image modeling (MIM). Context Encoders [25] pioneers MIM via image inpainting while Vision Transformer (ViT) [26] advances to predict the mean color of masked patches. Sim-MIM [13] and MAE [12] revisit MIM by processing only unmasked patches or all patches, respectively. SparK [27] adopts CNN for MIM, in contrast to ViT. Besides, iGPT [28] introduces autoregressive prediction to sequentially predict unknown pixels. SimAN [19] restores augmented scene text by aligning styles from its neighboring patches.

2) Diffusion-Based Methods: Beyond their renowned generation fidelity, diffusion models have emerged as powerful representation learners [29]. DiffAE [14] conditions the denoiser on a visual encoder to capture semantic traits. DDAE [15] unifies self-supervised generative and discriminative pretraining via a single denoiser. SODA [16] condenses latent representations using an information bottleneck. l-DAE [17] employs a PCA for latent space encoding. DBAE [30] bridges latent features with the generation endpoint to retain complete source information. Unlike prior methods that use single representations to bridge the encoder and denoiser (e.g., DiffAE, SODA, and DBAE), we modulate the denoiser using multi-scale representations aggregated via a feature pyramid to enhance feature extraction. Additionally, we propose a novel RFR pretext task, conditioning the denoiser on severely corrupted signature parts, enforcing the encoder to capture sufficient writing features for restoration.

![](images/b187705b94563a5773446d710820e1977b6cd044c904a6a603e0d9a698ce2821.jpg)  
(a) Reconstruction from Ruins (RFR)  
(b) SigDiffuser: Signature Representation Learning Diffusion Model  
Fig. 2: Overall framework of SigDiffuser. (a) Signature data construction for RFR. We apply crumpling on the signature image I to obtain $I _ { c }$ and segment square patches in different positions to acquire $I _ { o r g } , I _ { s t y } ,$ and $I _ { c o r }$ for RFR pretraining. (b) Detailed model architecture and self-supervised pretraining process. The visual encoder encodes the intact region $I _ { s t y }$ into latent representations. These features are hierarchically modulated to the denoiser to assist in reconstructing the severely crumpled portion $I _ { c o r }$ , thereby enforcing the visual encoder to learn meaningful representations.

## B. Offline Handwritten Signature Verification

Offline HSV methods typically consist of two stages: feature extraction and similarity measurement. Based on the feature extraction technique, existing methods can be grouped into hand-crafted feature-based and neural network-based methods.

1) Hand-Crafted Feature-Based Method: Conventional HSV relies heavily on hand-crafted feature descriptors, generally categorized as shape-based and texture-based. Shapebased features capture global appearance and geometric structure [31, 32, 33]. Texture-based features capture local texture patterns and micro-structures inside signatures [34, 35, 36]. Kumar et al. [37] model both texture and shape with surroundedness features. Other works also explore using KAZE-based Fisher Vectors [38] and learning SPD-manifold distances from covariance matrices [39]. For similarity measurement, Support Vector Machines [33, 34, 35], ensemble methods [32, 36] and manifold-based distance measures [39] are explored.

2) Neural Network-Based Method: Currently, deep neural networks dominate HSV due to their powerful representation capabilities. Early methods used CNNs as simple feature extractors [40], while recent works combine Siamese architectures with CNNs for local feature learning [6, 41, 42, 43, 44, 45]. For instance, MSDN [43] integrates region-based metric learning with a Siamese DenseNet. Subsequently, prompted by its effectiveness, ViT has been integrated into Siamese networks for global feature learning, utilizing mutual attention [5] or hybrid CNN-ViT architectures [46, 47]. Two-channel models that process stacked signatures represent an alternative routine [42, 48]. Beyond architecture, representations are further enriched by using text recognition for multi-task learning [49] or fusion with handcrafted features [50]. For similarity measurement, the paradigm evolves from decoupled verifiers (distance-based [41] or SVM-based [40, 51]) to end-to-end methods (thresholded sigmoid/softmax functions [5, 43, 45]).

However, existing methods overwhelmingly concentrate on supervised learning, neglecting self-supervised learning’s potential in this field. While a few works have explored selfsupervision [9, 10], such as reconstruction with attention guidance [9] and feature decorrelation [10], they yield inferior HSV performance. In this paper, we propose the first diffusionbased signature representation learning scheme, establishing a new paradigm for HSV through the integration of diffusion models and self-supervised learning.

## III. METHODOLOGY

Fig. 2 illustrates SigDiffuser’s overall framework. To extract deep representations, we introduce the Reconstruction from Ruins (RFR) pretext task that compels the model to reconstruct a severely crumpled signature portion using an intact reference. We harness the diffusion process to fulfill the reconstruction, with a pyramidal feature aggregation module (PFAM) to boost representation quality. The pretrained encoder is transferred to downstream HSV tasks.

## A. Pretext Task: Reconstruction from Ruins

The aim of RFR lies in reconstructing an utterly ruined signature through the assistance of an intact portion, thus enabling signature representation learning. To this end, we construct pairwise ruined-intact data. Given a clean and flat signature image $I ~ \in ~ \mathbb { R } ^ { 1 \times H \times W }$ (H is height and W is width), we generate a displacement field D with a size of $2 \times H \times W$ (2 is the channel number) to emulate crumpling. This field assigns a 2D vector $( d x , d y ) = ( d x _ { i } , d y _ { i } ) , i \in \{ 0 , . . . , H \times W - 1 \}$ to each pixel i, dictating its horizontal and vertical distortion. $( d x , d y )$ is further processed and applied to clean signatures to create the crumpled outputs. This process is formulated as:

$$
D = ( d x , d y ) = ( \mathcal { R } ( H , W ) \times 2 - 1 ) \times \alpha ,\tag{1}
$$

$$
\mathcal { G } ( i , j , \sigma ) = \frac { 1 } { 2 \pi \sigma ^ { 2 } } e ^ { - \frac { i ^ { 2 } + j ^ { 2 } } { 2 \sigma ^ { 2 } } } ,
$$

$$
\begin{array} { r l } & { d x _ { s } = ( d x \ast \mathcal { G } ) ( i , j ) } \\ & { \quad \quad = \sum _ { i = 0 } ^ { K - 1 } \sum _ { j = 0 } ^ { K - 1 } d x ( x - i , y - j ) \cdot \mathcal { G } ( i , j , \sigma ) , } \end{array}
$$

$$
\begin{array} { l } { d y _ { s } = ( d y \ast \mathcal { G } ) ( i , j ) \quad } \\ { \quad = \displaystyle \sum _ { i = 0 } ^ { K - 1 } \sum _ { j = 0 } ^ { K - 1 } d y ( x - i , y - j ) \cdot \mathcal { G } ( i , j , \sigma ) , } \end{array}\tag{2}
$$

where G specifies a Gaussian kernel (size K, standard deviation σ), R denotes a random noise generation function, and ${ \mathcal { R } } ( H , W )$ generates a random displacement field with a size of $H \times W$ . α acts as a scale factor. By prescribing extremely large values for $\alpha , \sigma ,$ and K, we induce a highly intense elastic distortion field dx and $d y$ along the x and y axes as per Eq. 1. Thereafter, we apply large-kernel Gaussian smoothing to ensure continuity as in Eq. 2. The smoothed distortion fields then undergo random shifts and scaling, with pixel adjustments based on their distance from the image center, yielding modified displacement fields $d \boldsymbol { x } _ { s } ^ { \prime }$ and $d y _ { s } ^ { \prime }$ Ultimately, the distorted displacement fields $d \bar { x _ { s } ^ { \prime } }$ and $d y _ { s } ^ { \prime }$ are added to the original pixel coordinates to obtain the “crumpled” coordinates, and we remap I according to these coordinates to create a severely crumpled signature $I _ { c } \mathbf { : }$

$$
\begin{array} { c } { D = [ d x _ { s } ^ { \prime } , d y _ { s } ^ { \prime } ] , } \\ { I _ { c } = I ^ { \prime } ( x , y ) = I ( x + D ( 0 , x , y ) , y + D ( 1 , x , y ) ) . } \end{array}\tag{3}
$$

Following this, we apply Gaussian noise to the crumpled signatures. We then crop a square patch $I _ { o r g }$ of side length l (the shorter side of I) at a random position of $I ,$ and crop a corresponding crumpled patch $I _ { c o r }$ from $I _ { c }$ at the same position. An equal-sized patch $I _ { s t y }$ is cropped from I elsewhere to serve as the style reference. This process is illustrated in Fig. 2 (a). Mirroring the human cognitive process of inferring holistic writing styles from visible parts, RFR requires the SigDiffuser model to restore the crumpled patch $I _ { c o r }$ to its clean version $I _ { o r g }$ under the guidance of $I _ { s t y }$ . We use a visual encoder E (detailed in the next subsection) to extract writing style representations $z _ { s t y } ~ = ~ \mathcal { E } ( I _ { s t y } )$ of the same signature from $I _ { s t y }$ to aid in reconstruction.

Technical foundation of RFR: shortcut-resistant information bottleneck. Technically, RFR induces a shortcutresistant information bottleneck by suppressing the direct information path from $I _ { c o r }$ to $I _ { o r g } \mathrm { , }$ thereby making reconstruction depend more on the style representation $z _ { s t y }$ . Let $\mathcal { T } ( \cdot ; \cdot )$ denote mutual information. The reconstruction-relevant information can be decomposed as:

$$
\mathcal { T } ( I _ { o r g } ; z _ { s t y } , I _ { c o r } ) = \mathcal { Z } ( I _ { o r g } ; I _ { c o r } ) + \mathcal { Z } ( I _ { o r g } ; z _ { s t y } \mid I _ { c o r } ) .\tag{4}
$$

In self-supervised tasks with mild image transformations [19, 52] or simple noise addition [14, 15, 16], the corrupted input preserves strong local structural correspondence with the target. Similarly, conventional reconstruction paradigms like Masked Image Modeling (MIM) [12, 13] lack an explicit, spatially independent style reference, relying heavily on local spatial cues from unmasked patches. In these cases, the first term $\mathcal { I } ( I _ { o r g } ; I _ { c o r } )$ remains dominant, allowing the denoiser/generator to exploit low-level pixel or geometry shortcuts and reducing the learning pressure on the encoder. In contrast, the severe crumpling introduced in RFR destroys local stroke continuity and spatial layouts, driving the direct target-corrupted mutual information toward a small residual, i.e., $\begin{array} { r } { \mathcal { T } ( I _ { o r g } ; I _ { c o r } )  \epsilon _ { \mathrm { : } } } \end{array}$ , forcing the reconstruction predominantly rely on $\mathcal { T } ( I _ { o r g } ; z _ { s t y } \mid I _ { c o r } )$

To ensure $z _ { s t y }$ provides meaningful guidance for this conditional dependence, the source patch $I _ { s t y }$ is intentionally cropped from a spatially distinct position from $I _ { o r g }$ . This prevents the visual encoder from trivially copying pixels, compelling it to encode intrinsic writing traits into $z _ { s t y }$ , such as stroke dynamics (stroke connections and turns), slant tendency, and pressure variations, that transfer across spatial locations. In this context, $z _ { s t y }$ supplies rich style cues to maximize the conditional mutual information $\mathcal { T } ( I _ { o r g } ; z _ { s t y } \mid I _ { c o r } )$ , effectively preventing superficial pixel shortcuts during reconstruction and improving the encoder’s representation quality.

## B. SigDiffuser

We present the architecture of the SigDiffuser model in ${ \mathrm { F i g . ~ 2 ~ ( b ) } }$ , consisting of an encoder E, a pyramidal feature aggregation module (PFAM) P, and a denoiser D.

Encoder. The encoder E is a ResNet-18 [53] without the second and last downsampling operations. It encodes the style signature patch $I _ { s t y }$ to derive hierarchical feature maps:

$$
\{ f _ { i } \} = \{ f _ { 1 } , f _ { 2 } , . . . , f _ { n } \} = { \mathcal { E } } ( I _ { s t y } ) , i \in \{ 1 , . . . , n \} ,\tag{5}
$$

where n denotes the number of hierarchies in E. These feature maps are then fed into the PFAM for hierarchical aggregation.

Pyramidal feature aggregation module. Existing models like DiffAE [14] and SODA [16] input only a single latent representation (typically from the last layer) to the denoiser, leading to limited representational capacity. To address this issue, we introduce the PFAM $( { \mathcal { P } } ;$ see Fig. 2 (b)) to extract richer representations by hierarchically aggregating multi-scale features from different layers of the visual encoder. PFAM is simplified from the feature pyramid network [54] by removing the final $3 \times 3$ convolution on the merged feature map. It takes the feature maps $\{ f _ { i } \}$ outputted by the encoder E, deriving a set of compact latent representations $z = \{ z _ { i } \} , i \in \{ 1 , . . . , n \} $

$$
z _ { i } = \left\{ \begin{array} { l l } { \mathcal { P } _ { i } ( f _ { i } ) = \psi _ { i } ( f _ { i } ) , \mathrm { i f } \ i = 1 , } \\ { \mathcal { P } _ { i } ( f _ { i - 1 } , f _ { i } ) = \psi _ { i } ( f _ { i - 1 } ) \oplus \phi _ { i } ( f _ { i } ) , \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{6}
$$

where $\mathcal { P } _ { i }$ is the $i ^ { t h }$ layer of the PFAM, $\psi _ { i }$ is a 1×1 convolution layer for channel projection, $\phi _ { i }$ is an upsample operation for spatial dimension matching through bilinear interpolation, and ⊕ is element-wise addition. Through this hierarchical aggregation process (Eq. 6), the $i ^ { t h }$ layer of PFAM amalgamates feature maps $f _ { i - 1 }$ and $f _ { i }$ to encode multi-scale writing traits, producing a more informative latent representation $z _ { i }$ at the current hierarchy. These hierarchical embeddings $z = \{ z _ { i } \}$ are subsequently injected into different layers of the denoiser D for layer-wise modulation. n is set to 4.

Denoiser. We build the denoiser D employing a conditional Denoising Diffusion Implicit Model (DDIM) [55], which is conditioned on the hierarchical latent representations $z =$ $\{ z _ { i } \} , i \ \in \ \{ 1 , . . . , n \}$ . It incorporates a forward process that progressively corrupts the target distribution $\begin{array} { l } { \pmb { x } _ { 0 } } \ \sim \ q ( \pmb { x } _ { 0 } )  \end{array}$ into pure Gaussian noise $\mathbf { \boldsymbol { x } } _ { T } \sim \mathcal { N } ( \mathbf { \boldsymbol { 0 } } , \pmb { I } )$ , complemented by a backward process that reconstructs the target by T steps iterative denoising from the conditional input ${ \pmb x } _ { T } ^ { \prime } = ( { \pmb x } _ { T } , z )$

The forward process is a Markov chain that incrementally adds noise to the target, as formulated by:

$$
{ \pmb x } _ { t } = \sqrt { \bar { \alpha _ { t } } } { \pmb x } _ { 0 } + \sqrt { 1 - \bar { \alpha _ { t } } } { \pmb \epsilon } ; \ \bar { \alpha } _ { t } = \prod _ { s = 1 } ^ { t } ( 1 - \beta _ { s } ) ,\tag{7}
$$

where $t ~ \sim ~ [ 0 , T ] .$ , T is the maximum sampling timestep. $\beta \sim ( 0 , 1 )$ is a fixed hyperparameter that indicates the noise levels. The backward process takes ${ \pmb x } _ { T } ^ { \prime } = ( { \pmb x } _ { T } , z )$ as input and deterministically approximates the reverse mapping:

$$
\begin{array} { c } { { { \pmb x } _ { t - 1 } = \sqrt { \bar { \alpha } _ { t - 1 } } \displaystyle \frac { { \pmb x } _ { t } - \sqrt { 1 - \bar { \alpha } _ { t } } { \pmb \epsilon } _ { \theta } ^ { t } ( { \pmb x } _ { t } , t , z ) } { \sqrt { \bar { \alpha } _ { t } } } \displaystyle + } } \\ { { \sqrt { 1 - \bar { \alpha } _ { t - 1 } } \displaystyle \epsilon _ { \theta } ^ { t } ( { \pmb x } _ { t } , t , z ) . } } \end{array}\tag{8}
$$

The noise term $\epsilon _ { \theta } ^ { t } ( x _ { t } , t , z )$ in Eq. 8 is predicted by the denoiser under the supervision of the MSE loss $| | \epsilon _ { \theta } ^ { t } ( x _ { t } , t , z ) - \epsilon | |$ As illustrated in ${ \mathrm { F i g . ~ } } 2 { \mathrm { ~ ( b ) } }$ , the denoiser adopts a four-layer, symmetric UNet architecture [56]. Since we compute four latent representations $\{ z _ { i } \} , i \in \{ 1 , . . . , 4 \}$ , we symmetrically add $z _ { i }$ into the $i ^ { t h }$ UNet layer through addition, achieving hierarchical modulation. From a global perspective, the denoiser is primarily responsible for image reconstruction, thus liberating the encoder to focus on latent encoding. Given the demanding reconstruction task (i.e., RFR), the encoder is driven to leverage any available writing features to assist the denoiser, which in turn incentivizes itself to extract more informative writing patterns, resulting in enhanced latent representations.

Notably, closer to our work are SODA [16] and DiffAE [14], yet SigDiffuser differs in two main aspects. (1) For the pretext task, SigDiffuser reconstructs severely crumpled signatures (the proposed RFR). However, SODA synthesizes a novel view of the image, while DiffAE performs reconstruction with the same image as semantic guidance. (2) Regarding architecture, SigDiffuser introduces a multi-scale, hierarchical feature modulation architecture for enriched latent features, whereas both SODA and DiffAE employ a single-representation approach. These approaches together result in better self-supervised learning performance on HSV as validated in Table II.

![](images/d978d88cfd8e544006bb609d64e567172d8aae8760ecb07dec0b5dfa7fe91a3a.jpg)  
Fig. 3: Downstream model architecture for HSV. (a) Training. The encoder pretrained on the RFR pretext task is transferred to the downstream model for HSV fine-tuning. (b) Testing. All signatures are input to the model to extract corresponding feature vectors, which are then used for verification.

## C. Downstream HSV Model

After the self-supervised pretraining with the RFR pretext task, we fine-tune a specific model for the downstream HSV task, as illustrated in Fig. 3. This model’s fine-tuning is independent of SigDiffuser’s pretraining (Sec. III-B), and all testing results (Sec. IV) are obtained using this model. Model architecture, training, and testing details are elaborated below.

Model architecture. As shown in Fig. 3 (a), the encoder of SigDiffuser pretrained via the RFR pretext task transfers directly as the HSV model encoder. Inspired by Spatial Pyramid Pooling (SPP) [57], we design an improved Normalized SPP (NSPP) module, which is placed after the encoder to reduce the spatial dimension of the last feature map into feature vectors. Given the final feature map $\boldsymbol { x } \in \mathbb { R } ^ { 5 1 \overline { { 2 } } \times h \times w }$ , where 512 is the channel number, $h$ and w respectively denote the height and width, it is first compressed by average pooling to derive local and global features $x _ { l }$ and $x _ { g } \mathrm { : }$

$$
x _ { g } = \mathrm { A d a A v g P o o l } _ { 2 \times 2 } ( x ) , x _ { l } = \mathrm { A d a A v g P o o l } _ { 1 \times 1 } ( x ) .\tag{9}
$$

To ensure scale invariance and enhance discriminability, both features are L2-normalized to obtain $\widetilde { x } _ { g }$ and $\widetilde { x } _ { l }$ . They are then transformed into 1D vectors using convolutional and fused:

$$
\begin{array} { r l } & { \hat { \boldsymbol { x } } _ { g } = \mathrm { c o n v } _ { 2 \times 2 } ( \widetilde { \boldsymbol { x } } _ { g } ) , \ \hat { \boldsymbol { x } } _ { l } = \mathrm { c o n v } _ { 1 \times 1 } ( \widetilde { \boldsymbol { x } } _ { l } ) , } \\ & { \quad \quad \quad f = \mathrm { S E L U } ( \hat { \boldsymbol { x _ { g } } } + \hat { \boldsymbol { x _ { l } } } ) , f \in \mathbb { R } ^ { 5 1 2 } . } \end{array}\tag{10}
$$

The final feature vector $f$ is used for loss computation or identity verification as described below.

Downstream model training. As shown in Fig. 3(a), we append a Position-Dependent Siamese Network (PDSN) [51] after the NSPP module strictly for training. For each writer, we sample triplets $\left( I _ { a } , I _ { p } , I _ { n } \right)$ comprising an anchor (genuine), a positive (genuine), and a negative (skilled forgery) signature. The encoder and NSPP extract corresponding features $( f _ { a } , f _ { p } , f _ { n } )$ . PDSN processes a collection of pairs $[ ( f _ { a } , f _ { n } ) , ( f _ { p } , f _ { n } ) , ( f _ { a } , f _ { p } ) ]$ to compute a logit triplet, which is supervised by binary cross-entropy loss $\mathcal { L } _ { B C E }$ against groundtruth labels [1, 1, 0]. Additionally, the feature vectors pass through a linear layer for writer ID classification, optimized via label-smoothed cross-entropy loss $\mathcal { L } _ { I D }$ (label smoothing

TABLE I: Dataset details and data splitting. #WriterTraining and #WriterTesting respectively denote the number of writers for training and testing. Sig. denotes signature. /writer denotes per writer.
<table><tr><td>Dataset</td><td>Language</td><td>#Writer</td><td>#WriterTaining</td><td> $\overline { { \# \mathrm { W i t e r } _ { \mathrm { T e s t i n g } } } }$ </td><td>#Genuine Sig./writer</td><td>#Forged Sig./writer</td><td>#Split</td><td>#Training</td><td>#Testing</td><td>#Total</td></tr><tr><td>BHSig260-B [20]</td><td>Bengali</td><td>100</td><td>50</td><td>50</td><td>24</td><td>30</td><td>5</td><td>2,700</td><td>2,700</td><td>5,400</td></tr><tr><td>BHSig260-H [20]</td><td>Hindi</td><td>160</td><td>100</td><td>60</td><td>24</td><td>30</td><td>5</td><td>5,400</td><td>3,240</td><td>8,640</td></tr><tr><td>HanSig [21]</td><td>Chinese</td><td>238</td><td>213</td><td>25</td><td>20</td><td>20</td><td>10</td><td>31,800</td><td>3,600</td><td>35,400</td></tr><tr><td>UTSig [22]</td><td>Persian</td><td>115</td><td>60</td><td>55</td><td>27</td><td>45</td><td>5</td><td>4,320</td><td>3,960</td><td>8,280</td></tr><tr><td>CEDAR [23]</td><td>English</td><td>55</td><td>50</td><td>5</td><td>24</td><td>24</td><td>11</td><td>2,400</td><td>240</td><td>2,640</td></tr></table>

0.1). The final optimization objective is $\mathcal { L } ~ = ~ 5 \mathcal { L } _ { B C E } ~ + ~$ $0 . 0 1 \mathcal { L } _ { I D }$ . We process B triplets from distinct writers per batch, resulting in an effective batch size of 24.

Downstream model testing. During testing, genuine signatures and skilled forgeries are processed to extract feature vectors via the NSPP module (bypassing the PDSN), as shown in Fig. 3 (b). These features are used to determine the genuineness of each sample and evaluate model performance.

## IV. EXPERIMENT

## A. Experimental Protocol

1) Dataset: We evaluate our method on multiple HSV datasets using two data-splitting settings: (i) Standard split, following existing training/testing partitions; and (ii) Split with validation, which holds out a validation subset from the training data for early stopping and checkpoint selection. As these datasets lack official splits, we mitigate evaluation bias by employing standard K-fold cross-validation (where test folds are mutually exclusive). For datasets indivisible into K folds under prior partitions (e.g., 100/60 for BHSig260-H), we adopt full-coverage random splits, ensuring every writer appears in the test set at least once, with some writers recurring in multiple test splits. These protocols are applied to all datasets, with per-dataset details presented below.

BHSig260. BHSig260 [20] contains BHSig260-B (5,400 Bengali signatures from 100 writers) and BHSig260-H (8,640 Hindi signatures from 160 writers), with each writer contributing 24 genuine signatures and 30 skilled forgeries. Under setting (i), BHSig260-B is split into 50/50 writers for training/testing, and BHSig260-H is split into 100/60, following [5, 6, 41, 44]. Under setting (ii), 10 and 20 training writers are further reserved for validation on BHSig260-B and BHSig260-H, respectively, yielding 40/10/50 and 80/20/60 training/validation/testing splits. We perform 5 full-coverage random splits on each of BHSig260-B and BHSig260-H.

HanSig. HanSig [21] is a Chinese signature dataset containing 885 names signed by 238 writers. Each name has 20 genuine signatures in three styles (neat, normal, and stylish) and an equal number of skilled forgeries, totaling 35,400 signatures. Under setting (i), 795 names from 213 writers are used for training and 90 names from 25 writers for testing, following the original protocol. Under setting (ii), 60 training names are reserved for validation, giving a 735/60/90 split. We perform 10 full-coverage random splits.

UTSig. UTSig [22] includes 8,280 Persian signatures from 115 individuals, each providing 27 genuine signatures and 45 forgeries (42 skilled forgeries and three opposite-hand written). Under setting (i), 60/55 writers are used for training/testing following the original protocol. Under setting (ii),

10 training writers are reserved for validation, giving a   
50/10/55 split. We perform 5 full-coverage random splits.

CEDAR. CEDAR [23] comprises 2,640 English signatures from 55 writers, with 24 genuine signatures and 24 skilled forgeries per writer. Under setting (i), 50/5 writers are used for training/testing following [5, 43]. Under setting (ii), 5 training writers are reserved for validation, giving a 45/5/5 split. Unlike the other datasets, which use full-coverage random splits, we perform 11-fold cross-validation on CEDAR.

Data splitting details are summarized in Table I. Note that writer sets for training and testing are disjoint, ensuring openset evaluations. For BHSig260, we pretrain on the union of the BHSig260-B and BHSig260-H training sets, but fine-tune and evaluate on each separately. For other datasets, pretraining and fine-tuning share the corresponding training set. To prevent data leakage, pretraining and fine-tuning are independently rerun per split or fold. We report the mean and standard deviation of the performance as mean (±std).

2) Evaluation Metric: We employ four metrics: Equal Error Rate (EER), False Acceptance Rate (FAR), False Rejection Rate (FRR), and Accuracy (Acc). For fairness, we follow the most stringent evaluation setting from [5, 43, 44], implementing an open-set verifier based on three realistic design choices. (1) Since HSV compares a query against one or more references, we adopt the more difficult yet realistic single-reference (1-vs-1) protocol, despite multiple references typically yielding higher accuracy. (2) We use a global threshold across all writers rather than a user-specific one for decision-making, enabling more reliable and objective assessment. (3) We evaluate under the writer-independent scenario, verifying signatures without user-specific training during enrollment. This better reflects real-world deployment than writer-dependent methods that train personalized classifiers per writer. Implementation details are presented below.

Specifically, given a query signature $x ^ { q }$ claiming to be writer w and a reference signature $x ^ { w }$ of writer w, we extract their 512-dimensional feature vectors $f _ { q } = \phi ( x _ { q } )$ and $f _ { w } = \phi ( x _ { w } )$ using the encoder and pooling layer ϕ of our downstream model, and measure their similarity by:

$$
s ^ { w } ( x ^ { q } ) = | | f _ { q } - f _ { w } | | _ { 2 } / \sqrt { \overline { { d _ { w } } } } ,\tag{11}
$$

where $| | \cdot | | _ { 2 }$ refers to the Euclidean distance, $\overline { { d _ { w } } }$ denotes the mean reference-to-reference distance for writer w, following [66]; in our 1-vs-1 setting with a single reference, $\overline { { d _ { w } } } = 1$ Given a threshold c, the decision rule is defined as:

$$
\delta _ { c } ( x ^ { q } , x ^ { w } ) = \mathbb { 1 } [ s ^ { w } ( x ^ { q } ) < c ] ,\tag{12}
$$

where $\mathbb { 1 } [ \cdot ]$ is the indicator function; $\delta _ { c } ~ = ~ 1$ accepts $x ^ { q }$ as

TABLE II: Linear probing and downstream fine-tuning results on BHSig260-B [20] and BHSig260-H [20]. The best and secondbest results are marked in bold and with underlines, with the same annotations throughout. Superscripts $^ { * } , ^ { \dag } , ^ { \star }$ indicate models utilizing their original backbones: ResNet-50 [53], a modified 10-layer VGG [58], and ConvNeXt-Small [59], respectively.
<table><tr><td rowspan="3">Method</td><td rowspan="3">Venue</td><td colspan="5">Linear Probing</td><td colspan="2"></td><td colspan="6">Downstream Fine-tuning</td></tr><tr><td rowspan="2">BHSig260-B</td><td colspan="4"></td><td colspan="2">BHSig260-H</td><td colspan="2">BHSig260-B</td><td colspan="2"></td><td colspan="2">BHSig260-H</td></tr><tr><td>EER FAR↓</td><td></td><td>FRR</td><td>Acc ↑</td><td>EER↓ FAR</td><td>FRR↓</td><td>Acc ↑ EER</td><td>FAR ↓</td><td>FRR↓ Acc ↑</td><td>EER↓</td><td>FAR↓</td><td>FRR↓</td><td>Acc ↑</td></tr><tr><td>trastive SSRL Methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SimCLR [60] SimCLR* [60]</td><td>ICML&#x27;20</td><td>34.96 (±0.57) 39.59 (±0.98)</td><td>34.96 (±0.57) 34.96 (±0.57) 65.04 (±0.57) 39.59 (±0.98)</td><td>)60.42 (±0.99)</td><td>35.55 (±0.54)</td><td>35.56 (±0.54)</td><td>35.54 (±0.54) 64.45 (±0.54)</td><td>8.28 (±1.52) 61.65 (±0.68)</td><td>8.28 (±1.53) 9.67 (±1.58)</td><td>8.28 (±1.52) 91.72 (±1.52)</td><td>6.36 (±1.23) 8.54 (±1.53)</td><td>6.37 (±1.22) 8.54 (±1.54)</td><td>6.36 (±1.23)</td><td>93.63 (±1.22)</td></tr><tr><td>BYOL [61]</td><td>ICML&#x27;20 NeurIPS&#x27;20</td><td>39.58 (±0.99) 13.51 (±1.69)</td><td>13.52 (±1.69) 13.51 (±1.70) 86.48 (±1.69)</td><td></td><td>38.35 (±0.68) 27.08 (±0.90)</td><td>38.35 (±0.69) 27.08 (±0.90)</td><td>38.36 (±0.67) 27.08 (±0.90)</td><td>9.67 (±1.58) 5.86 (±0.73)</td><td>9.67 (±1.58) 5.86 (±0.73) 5.85 (±0.73)</td><td>90.33 (±1.58) 94.14 (±0.73)</td><td>7.11 (±1.67)</td><td>7.11 (±1.67)</td><td>8.54 (±1.53) 7.11 (±1.67)</td><td>91.46 (±1.54) 92.89 (±1.67)</td></tr><tr><td>BYOL* [61]</td><td>ICML&#x27;20</td><td>23.64 (±1.69) 23.64 (±1.69)</td><td>23.64 (±1.69)</td><td>76.36 (±1.69)</td><td>28.37 (±1.23)</td><td>28.37 (±1.23)</td><td>28.38 (±1.23)</td><td>71.63 (±1.23) 5.98 (±1.78)</td><td>5.98 (±1.78) 5.98 (±1.78)</td><td>94.02 (±1.78)</td><td>8.05 (±1.98)</td><td>8.05 (±1.98)</td><td>8.05 (±1.99)</td><td>91.95 (±1.98)</td></tr><tr><td>SeqCLR [62]</td><td>CVPR&#x27;21</td><td>20.25 (±1.56) 20.25 (±1.56)</td><td>20.26 (±1.56)</td><td></td><td>22.99 (±0.65)</td><td>22.99 (±0.64)</td><td>22.99 (±0.65)</td><td>77.01 (±0.64) 10.81 (±0.67)</td><td>10.82 (±0.67) 10.81 (±0.67)</td><td>89.19 (±0.67)</td><td>9.51 (±1.48)</td><td>9.50 (±1.48)</td><td>9.51 (±1.48)</td><td>90.50 (±1.48)</td></tr><tr><td>ChaCo [63]</td><td>ICFHR&#x27;22</td><td>11.10 (±1.04) 11.10 (±1.04)</td><td>11.10 (±1.04)</td><td>88.90 (±1.04)</td><td>16.60 (±0.72)</td><td>16.60 (±0.71)</td><td>16.60 (±0.72)</td><td>83.40 (±0.72) 6.73 (±1.83)</td><td>6.73 (±1.83) 6.73 (±1.82)</td><td>93.27 (±1.83)</td><td>5.35 (±1.18)</td><td>5.35 (±1.18)</td><td>5.34 (±1.18)</td><td>94.65 (±1.18)</td></tr><tr><td>CMT-Co [64]</td><td>ACCV&#x27;22</td><td>15.81 (±1.62) 15.82 (±1.62)</td><td></td><td></td><td>19.89 (±0.99)</td><td>19.90 (±0.99)</td><td></td><td>7.35 (±1.40)</td><td>7.35 (±1.41)</td><td>7.34 (±1.40) 92.65 (±1.41)</td><td>5.83 (±1.88)</td><td>5.83 (±1.88)</td><td>5.83 (±1.88)</td><td>94.17 (±1.88)</td></tr><tr><td>SWIS [10]</td><td>ICIP&#x27;22</td><td>13.35 (±0.65) 13.35</td><td>(±0.65) 13.35 (±0.66)</td><td></td><td>17.65 (±1.26)</td><td>17.65 (±1.26)</td><td>17.65 (±1.26)</td><td>82.35 (±1.26) 8.46 (±1.56)</td><td>8.46 (±1.56) 8.45 (±1.56)</td><td>91.54 (±1.56)</td><td>9.52 (±4.50)</td><td>9.52 (±4.49)</td><td>9.53 (±4.50)</td><td>90.48 (±4.49)</td></tr><tr><td>Generative SSRL</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SURDS [9]</td><td>ICPR&#x27;22 CVPR&#x27;22</td><td>16.17 (±2.17) 16.16 (±2.17) 19.66 (±1.41)</td><td>16.17 (±2.17) 19.65 (±1.41) 19.66 (±1.42)</td><td>83.84 (±2.17) 80.34 (±1.41)</td><td>26.83 (±0.86) 30.33 (±0.72)</td><td>26.83 (±0.86)</td><td>26.83 (±0.86)</td><td>73.17 (±0.86) 7.21 (±1.05)</td><td>7.21 (±1.05) 7.22 (±1.06)</td><td>92.79 (±1.05)</td><td>6.20 (±1.02)</td><td>6.20 (±1.02)</td><td>6.20 (±1.02)</td><td>93.80 (±1.02)</td></tr><tr><td>SimAN [19]</td><td>CVPR&#x27;22</td><td>28.18 (±1.56)</td><td>28.17 (±1.58) 28.15 (±1.53)</td><td>71.83 (±1.56)</td><td>36.59 (±0.49)</td><td>30.33 (±0.73)</td><td>30.33 (±0.72)</td><td>69.67 (±0.72) 5.93 (±0.46)</td><td>5.93 (±0.46)</td><td>5.92 (±0.46) 94.07 (±0.46)</td><td>6.52 (±1.41)</td><td>6.52 (±1.41)</td><td>6.52 (±1.41)</td><td>93.48 (±1.41)</td></tr><tr><td>SimAN† [19]</td><td>CVPR&#x27;22</td><td>30.89 (±2.03)</td><td>30.88 (±2.02)</td><td></td><td>41.98 (±0.74)</td><td>36.62 (±0.42) 41.98 (±0.75)</td><td>36.56 (±0.55)</td><td>63.40 (±0.46) 12.14 (±8.50)</td><td>12.16 (±8.54) 15.24 (±1.70)</td><td>12.12 (±8.44) 87.86 (±8.52) 15.25 (±1.66)</td><td>7.77 (±1.26) 37.86 (±1.78)</td><td>7.73 (±1.30) 38.18 (±1.47)</td><td>7.83 (±1.20) 37.57 (±2.05)</td><td>92.24 (±1.27)</td></tr><tr><td>MAE [12]</td><td>CVPR&#x27;22</td><td>29.99 (±3.59)</td><td>29.99 (±3.59) 29.99 (±3.59)</td><td></td><td></td><td>29.89 (±2.59) 29.89 (±2.58)</td><td>41.99 (±0.73) 29.89 (±2.59)</td><td>58.02 (±0.74) 15.24 (±1.68) 70.11 (±2.58) 7.40 (±1.72)</td><td>7.40 (±1.73) 7.40 (±1.72)</td><td>84.76 (±1.69) 92,60 (±1.73)</td><td>9.21 (±1.93)</td><td>9.21 (±1.93)</td><td>9.21 (±1.93)</td><td>61.99 (±1.63)</td></tr><tr><td>DiffAE [14] SparK [27]]</td><td>ICLR&#x27;23</td><td>13.54 (±1.19)</td><td>13.54 (±1.19) 13.54 (±1.19)</td><td>86.46(±1.19)</td><td>20.45 (±0.85)</td><td>20.45 (±0.85)</td><td>20.46 (±0.85)</td><td>79.55 (±0.85) 5.42 (±1.42)</td><td>5.42 (±1.42) 5.42 (±1.41)</td><td>94.58 (±1.42)</td><td>7.00 (±1.42)</td><td>7.00 (±1.42)</td><td>7.00 (±1.43)</td><td>90.79 (±1.93) 93.00 (±1.42)</td></tr><tr><td>SparK* [27]</td><td>ICLR&#x27;23</td><td>25.46 (±1.14)</td><td>25.46 (±1.13) 25.47 (±1.14)</td><td></td><td>35.44 (±1.27)</td><td>35.44 (±1.28)</td><td>35.45 (±1.27)</td><td>64.56 (±1.28) 14.16 (±0.80)</td><td>14.16 (±0.80)</td><td>14.15 (±0.79) 85.84 (±0.80)</td><td>11.71 (±1.69)</td><td>11.70 (±1.69)</td><td>11.71 (±1.69)</td><td>88.30 (±1.69)</td></tr><tr><td>DDAE [15]</td><td>ICCV&#x27;23</td><td>10.32 (±1.23)</td><td>10.32 (±1.23)</td><td>10.32 (±1.24) 89.68 (±1.23)</td><td>17.11 (±0.84)</td><td>17.10 (±0.84)</td><td>17.11 (±0.84) 82.89 (±0.84)</td><td>4.18 (±0.84)</td><td>4.18 (±0.84)</td><td>4.18 (±0.84) 95.82 (±0.84)</td><td>4.59 (±1.01)</td><td>4.59 (±1.01)</td><td>4.59 (±1.01)</td><td>95.41 (±1.01)</td></tr><tr><td>SODA [16]</td><td>CVPR&#x27;24</td><td>11.56 (±1.85) 11.56 (±1.85) 10.12 (±1.41)</td><td>10.12 (±1.41)</td><td>11.56 (±1.85) 88.44 (±1.85 10.12 (±1.41) 89.88 (±1.41)</td><td></td><td>13.02 (±2.16)</td><td></td><td>2.82 (±0.54)</td><td>2.82 (±0.55)</td><td>2.83 (±0.55) 97.18 (±0.55)</td><td>3.99 (±1.13)</td><td>3.98 (±1.13)</td><td>3.99 (±1.13)</td><td>96.02 (±1.13)</td></tr><tr><td>SODA* [16]</td><td>CVPR&#x27;24</td><td>25.62 (±1.55)</td><td>25.62 (±1.56) 25.62 (±1.54) 74.38 (±1.55)</td><td></td><td>17.47 (±1.09)</td><td></td><td>17.47 (±1.08) 82.53 (±1.09)</td><td>3.34 (±0.20)</td><td>11.27 (±1.06)</td><td>96.66 (±0.20)</td><td>6.47 (±1.54)</td><td>6.47 (±1.54)</td><td>6.47 (±1.53)</td><td>93.53 (±1.54)</td></tr><tr><td>l-DAE [17]</td><td>ICLR&#x27;25 ICLR&#x27;25</td><td>18.32 (±0.92)</td><td>18.33 (±0.92)</td><td>18.32 (±0.92) 81.67 (±0.92)</td><td>20.48 (±1.01)</td><td>20.48 (±1.01)</td><td>40.00 (±3.95) 40.00 (±3.96) 40.01 (±3.94) 60.00 (±3.96) 20.49 (±1.01) 79.52 (±1.01)</td><td>11.27 (±1.05) 6.07 (±0.77)</td><td>6.07 (±0.77)</td><td>11.27 (±1.04) 88.73 (±1.06) 6.07 (±0.76) 93.93 (±0.77)</td><td>38.24 (±12.62) 6.28 (±1.41)</td><td>38.23 (±12.60) 6.28 (±1.41)</td><td>38.26 (±12.64)</td><td>61.76 (±12.61)</td></tr><tr><td>DBAE [30]</td><td>TCSVT&#x27;26</td><td>34.01 (±1.17)</td><td>34.03 (±1.17)</td><td>33.98 (±1.18) 65.98 (±1.17)</td><td>33.88 (±1.24)</td><td>33.88 (±1.26)</td><td>33.88 (±1.22)</td><td>66.12 (±1.25) 16.76 (±2.25)</td><td>16.76 (±2.25)</td><td>16.76 (±2.25) 83.24 (±2.25)</td><td>19.72 (±1.28)</td><td>19.72 (±1.28)</td><td>6.28 (±1.41) 19.72 (±1.29)</td><td>93.72 (±1.41)</td></tr><tr><td>CAMD [8]</td><td>CVPR&#x27;26</td><td>23.68(±1.86) 23.68 (±1.86)</td><td></td><td>23.68 (±1.85) 76.32 (±1.86)</td><td>29.61 (±0.88)</td><td>29.60 (±0.88)</td><td>29.62(±0.88)</td><td>70.39 (±0.88) 12.38 (±1.77)</td><td>12.38 (±1.77)</td><td>12.38 (±1.77) 87.62 (±1.77)</td><td>16.41 (±1.38)</td><td>16.41 (±1.39)</td><td>16.41 (±1.38)</td><td>80.28 (±1.28) 83.59 (±1.38)</td></tr><tr><td>DIS [65] Baseline (Ours)</td><td>This work</td><td>19.92 (±1.47) 19.91 (±1.47)</td><td>9.56 (±1.13)</td><td>19.92 (±1.47) 80.09 (±1.47) 9.56(±1.13) 90.44 (±1.13)</td><td>30.02 (±0.38)</td><td>30.02 (±0.38)</td><td>30.02 (±0.38)</td><td>69.98 (±0.38) 6.23 (±0.45)</td><td>6.23 (±0.45) 1.85 (±0.46)</td><td>6.23 (±0.45) 93.77 (±0.45)</td><td>6.88 (±1.31)</td><td>6.89 (±1.31)</td><td>6.88 (±1.31)</td><td>93.11 (±1.31)</td></tr><tr><td>SigDiffuser (Ours)</td><td>This work</td><td>9.56 (±1.13)</td><td></td><td></td><td></td><td></td><td>13.02 (±0.61) 13.01 (±0.61) 86,98 (±0.61)</td><td>1.85 (±0.46)</td><td></td><td>1.84 (±0.46) 98.15 (±0.46)</td><td>3.13 (±0.33)</td><td>3.13 (±0.33)</td><td>3.13 (±0.33)</td><td>96.87 (±0.33)</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

genuine, and $\delta _ { c } = 0$ rejects it as a forgery.

Consider a dataset comprising W writers, each possessing N genuine signatures and M forged signatures. To achieve 1-vs-1 evaluation, we designate each genuine sample in turn as the reference while treating the remaining N − 1 genuine signatures and M forged signatures as queries. Since the comparison score is symmetric, each genuine-genuine pair is counted once, yielding $N \times ( N - 1 ) / 2$ genuine-genuine scores and $N \times M$ genuine-forged scores per writer. We then pool all the comparison scores from all writers into a single global evaluation set, resulting in $W \times ( N \times ( N - 1 ) / 2 + N \times M )$ scores. Given a global threshold c, each score is converted into a binary decision according to Eq. 12. We compare these decisions with the corresponding pair labels and count the numbers of True Positive (TP), False Positive (FP), True Negative (TN), and False Negative (FN). The FAR, FRR, and Acc at threshold c are computed as:

$$
\begin{array} { c c c } { F A R ( c ) = \displaystyle \frac { F P ( c ) } { T N ( c ) + F P ( c ) } , } \\ { F R R ( c ) = \displaystyle \frac { F N ( c ) } { T P ( c ) + F N ( c ) } , } \\ { A c c ( c ) = \displaystyle \frac { T P ( c ) + T N ( c ) } { T P ( c ) + T N ( c ) + F P ( c ) + F N ( c ) } . } \end{array}\tag{13}
$$

EER is the error rate at the threshold where FAR equals FRR. Let C denote the candidate threshold set consisting of all distinct comparison scores. Given the discrete and finite nature of C, an exact threshold satisfying $F A R ( c ) = F R R ( c )$ generally does not exist within C. We therefore first search the candidate set C for a discrete threshold $c ^ { * }$ that minimizes the gap between FAR and FRR:

$$
c ^ { * } = \arg \operatorname* { m i n } _ { c \in \mathcal { C } } | F A R ( c ) - F R R ( c ) | .\tag{14}
$$

Since the computation of FAR, FRR, and Acc requires a valid confusion matrix at an actual discrete threshold, we report them at this determined operating point:

$$
\mathrm { F A R } = F A R ( c ^ { * } ) , \mathrm { F R R } = F R R ( c ^ { * } ) , \mathrm { A c c } = A c c ( c ^ { * } ) .\tag{15}
$$

Subsequently, we compute EER at an exact threshold obtained through linear interpolation. While the discrete approximation $\scriptstyle { \frac { 1 } { 2 } } ( F A R ( c ^ { * } ) + F R R ( c ^ { * } ) )$ can provide a direct estimate, it may deviate from the EER at the true FAR-FRR intersection. To obtain a precise EER, we linearly interpolate between c<sup>∗</sup> and its adjacent threshold $c ^ { \dagger } \in { \mathcal { C } }$ lying on the opposite side of the FAR–FRR crossing to locate the crossing point, i.e., the exact threshold. Let $D ( c ) = F A R ( c ) - F R R ( c )$ , this threshold c˜ is given by:

$$
\tilde { c } = c ^ { * } + \frac { D ( c ^ { * } ) } { D ( c ^ { * } ) - D ( c ^ { \dagger } ) } ( c ^ { \dagger } - c ^ { * } ) .\tag{16}
$$

At this threshold, the FAR and FRR equalize, allowing us to compute the EER as:

$$
\mathrm { E E R } = F A R ( \tilde { c } ) = F R R ( \tilde { c } ) .\tag{17}
$$

Note that when $c ^ { * } = { \tilde { c } } , \operatorname { E E R } = \operatorname { F A R } = \operatorname { F R R } = 1 - \operatorname { A c c } .$

Importantly, $c ^ { * }$ (and consequently c˜) is derived from the score distribution pooled across all writers, representing a global threshold rather than a writer-specific one. This implementation also accommodates an unlimited number of test writers, ensuring writer independence and generalization to unseen writers. All metrics are reported as percentages.

## B. Implementation Detail

The maximum time step T for denoiser sampling is set to $1 0 0 0 . \ \alpha , \ \sigma ,$ , and K in RFR (Eq. 1 and 2) are set to 18, 072, $2 . 0 4 8 \times 1 0 ^ { 6 }$ , 91, respectively. During pretraining, the side length l for patches $I _ { o r g } , \ I _ { c o r } ,$ , and $I _ { s t y }$ is 96. During finetuning, all datasets use 96×336 (height × width) image inputs. More data processing details are included in Appendix A.

For pretraining, SigDiffuser is trained from scratch for 800 epochs (20 warmup epochs) using AdamW $( \beta _ { 1 } = 0 . 9 $ $\beta _ { 2 } ~ = ~ 0 . 9 9 9$ , and weight decay 0.05), a batch size of 32, and a learning rate of 1e-4. The additive noise term ϵ in the denoiser follows a cosine schedule. For downstream training, we train the model for 80 epochs with a batch size of 24, using AdamW with a weight decay of 1e-5 and a cosine learning rate decay from 1e-4 to 1e-7. The baseline is a randomly initialized downstream model fine-tuned per dataset.

TABLE III: Performance comparison on BHSig260-B [20].
<table><tr><td>Method</td><td>Venue</td><td>EER↓</td><td>FAR↓</td><td>FRR↓</td><td>Acc ↑</td></tr><tr><td>SigNet [41]</td><td>arXiv&#x27;17</td><td>13.89</td><td>13.89</td><td>13.89</td><td>86.11</td></tr><tr><td>SigNet-F [40]</td><td>PR&#x27;17</td><td>20.44</td><td>20.30</td><td>20.87</td><td>79.45</td></tr><tr><td>FMPNet [51]</td><td>ICFHR&#x27;18</td><td>13.36</td><td>13.33</td><td>13.37</td><td>86.65</td></tr><tr><td>DeepHSV [42]</td><td>ICDAR&#x27;19</td><td>11.92</td><td></td><td></td><td>88.08</td></tr><tr><td>IDN [6]</td><td>CVPR&#x27;19</td><td></td><td>4.12</td><td>5.24</td><td>95.32</td></tr><tr><td>SDINet [67]</td><td>AAAI&#x27;21</td><td></td><td>3.30</td><td>7.86</td><td>94.42</td></tr><tr><td>CaP [68]</td><td>ICPR&#x27;21</td><td>3.96</td><td>3.96</td><td>3.96</td><td></td></tr><tr><td>AVN [44]</td><td>TMM&#x27;22</td><td>6.14</td><td></td><td></td><td>93.80</td></tr><tr><td>2C2S [48]</td><td>EAAI&#x27;23</td><td>6.75</td><td>5.37</td><td>8.11</td><td>93.25</td></tr><tr><td>MGRNet [21]</td><td>arXiv&#x27;23</td><td>6.12</td><td>5.93</td><td>6.20</td><td></td></tr><tr><td>LwR [69]</td><td>PR&#x27;23</td><td>6.10</td><td></td><td></td><td>94.50</td></tr><tr><td>TransOSV [5]</td><td>PR&#x27;24</td><td>9.90</td><td>9.90</td><td>9.90</td><td></td></tr><tr><td>SPD Manifold [39]</td><td>TIFS&#x27;24</td><td>11.10</td><td></td><td></td><td></td></tr><tr><td>CWA [45]</td><td>ICDAR&#x27;24</td><td>2.53</td><td>2.95</td><td>2.14</td><td>97.27</td></tr><tr><td>DetailSemNet [46]</td><td>ECCV&#x27;24</td><td>10.19</td><td>5.96</td><td>18.24</td><td>90.64</td></tr><tr><td>HTCSigNet [47]</td><td>PR&#x27;25</td><td>8.52</td><td>9.20</td><td>7.83</td><td>91.17</td></tr><tr><td>Baseline (Ours)</td><td>This work</td><td>6.23 (±0.45)</td><td>6.23 (±0.45)</td><td>6.23 (±0.45)</td><td>93.77 (±0.45)</td></tr><tr><td>SigDiffuserval (Ours)</td><td>This work</td><td>2.23 (±0.53)</td><td>2.23 (±0.53)</td><td>2.23 (±0.53)</td><td>97.77 (±0.53)</td></tr><tr><td>SigDiffuser (Ours)</td><td>This work</td><td>1.85 (±0.46)</td><td>1.85 (±0.46)</td><td>1.84 (±0.46)</td><td>98.15 (±0.46)</td></tr></table>

TABLE V: Performance comparison on HanSig [21].
<table><tr><td>Method</td><td>Venue</td><td>EER↓</td><td>FAR↓</td><td>FRR↓</td><td>Acc ↑</td></tr><tr><td>SigNet [41]</td><td>arXiv&#x27;17</td><td>33.07</td><td>33.10</td><td>33.05</td><td>66.92</td></tr><tr><td>SigNet-F [40]</td><td>PR&#x27;17</td><td>41.71</td><td>41.80</td><td>41.58</td><td>58.31</td></tr><tr><td>FMPNet [51]</td><td>ICFHR&#x27;18</td><td>13.13</td><td>13.10</td><td>13.16</td><td>86.87</td></tr><tr><td>MGRNet [21]</td><td>arXiv&#x27;23</td><td>9.93</td><td>11.85</td><td>7.69</td><td></td></tr><tr><td>CWA [45]</td><td>ICDAR&#x27;24</td><td>9.47</td><td>6.61</td><td>16.23</td><td>90.26</td></tr><tr><td>DetailSemNet [46]</td><td>ECCV&#x27;24</td><td>16.00</td><td>10.19</td><td>24.62</td><td>85.17</td></tr><tr><td>HTCSigNet [47]</td><td>PR&#x27;25</td><td>22.18</td><td>22.18</td><td>22.18</td><td>77.82</td></tr><tr><td>Baseline (Ours)</td><td>This work</td><td>19.22 (±8.99)</td><td>19.22 (±8.98)</td><td>19.23 (±8.99)</td><td>80.78 (±8.99)</td></tr><tr><td>SigDiffuserval (Ours)</td><td>This work</td><td>7.69 (±2.23)</td><td>7.69 (±2.23)</td><td>7.69 (±2.24)</td><td>92.31 (±2.23)</td></tr><tr><td>SigDiffuser (Ours)</td><td>This work</td><td>7.39 (±1.64)</td><td>7.39 (±1.63)</td><td>7.39 (±1.64)</td><td>92.61 (±1.64)</td></tr></table>

TABLE VII: Performance comparison on CEDAR [23].
<table><tr><td>Method</td><td>Venue</td><td>EER ↓</td><td>FAR↓</td><td>FRR↓</td><td>Acc ↑</td></tr><tr><td>SigNet-F [40]</td><td>PR&#x27;17</td><td>5.92</td><td></td><td></td><td></td></tr><tr><td>FMPNet [51]</td><td>ICFHR&#x27;18</td><td>12.17</td><td>12.08</td><td>12.17</td><td>87.87</td></tr><tr><td>IDN [6]</td><td>CVPR&#x27;19</td><td>3.62</td><td>5.87</td><td>2.17</td><td></td></tr><tr><td>SDINet [67]</td><td>AAAI&#x27;21</td><td>1.75</td><td>0.73</td><td>3.42</td><td></td></tr><tr><td>MSDN [43]</td><td>PR&#x27;21</td><td>6.74</td><td></td><td></td><td></td></tr><tr><td>CaP [68]</td><td>ICPR&#x27;21</td><td>4.34</td><td>4.34</td><td>4.34</td><td></td></tr><tr><td>ILNNeck [70]</td><td>IJCNN&#x27;22</td><td>9.21</td><td></td><td></td><td>90.79</td></tr><tr><td>QSLS [71]</td><td>Neural Comput. Appl.&#x27;23</td><td>1.93</td><td>3.35</td><td>2.39</td><td></td></tr><tr><td>MGRNet [21]</td><td>arXiv&#x27;23</td><td>3.51</td><td>3.33</td><td>3.55</td><td></td></tr><tr><td>SPD Manifold [39]</td><td>TIFS&#x27;24</td><td>8.53</td><td></td><td></td><td></td></tr><tr><td>DetailSemNet [46]</td><td>ECCV&#x27;24</td><td>24.61</td><td>10.97</td><td>40.29</td><td>79.53</td></tr><tr><td>DCCCAE [50]</td><td>PR&#x27;25</td><td>10.81</td><td></td><td></td><td></td></tr><tr><td>Baseline (Ours)</td><td>This work</td><td>12.77 (±8.42)</td><td>12.77 (±8.42)</td><td>12.75 (±8.42)</td><td>87.23 (±8.42)</td></tr><tr><td>SigDiffuserval (Ours)</td><td>This work</td><td>8.55 (±6.91)</td><td>8.55 (±6.91)</td><td>8.54 (±6.91)</td><td>91.45 (±6.91)</td></tr><tr><td>SigDiffuser (Ours)</td><td>This work</td><td>6.92 (±5.88)</td><td>6.92 (±5.87)</td><td>6.92 (±5.88)</td><td>93.08 (±5.88)</td></tr></table>

Data augmentation is limited to random translation (≤5 pixels, padded with white). Training time scales with the dataset size. For example, on the largest HanSig dataset, pretraining consumes around four days and fine-tuning takes around two hours, respectively, on a single NVIDIA 3090 GPU. For standard data splits, we evaluate the final epoch’s checkpoint. For splits with validation, we apply early stopping with a patience of 20 epochs and evaluate the best checkpoint.

## C. Linear Probing and Downstream Fine-Tuning Evaluation

We investigate SigDiffuser’s representation quality through probe evaluation (freezing the encoder, training only NSPP and PDSN) and downstream fine-tuning (full-parameter training). Baselines include contrastive methods (e.g., SimCLR [60], SeqCLR [62]), MIM-based generative methods (e.g., MAE [12], SparK [27], CAMD [8]), diffusion-based generative methods (e.g., DiffAE [14], SODA [16], l-DAE [17]). For fair comparison, most baselines are reimplemented using SigDiffuser’s ResNet-18 encoder. Exceptions retaining their native architectures are MAE, CAMD, and DIS (ViT-Tiny [26]), and l-DAE (DiT-S [72]). Baselines evaluated with their original encoders are marked with \*, <sup>†</sup>, and <sup>⋆</sup>.

TABLE IV: Performance comparison on BHSig260-H [20].
<table><tr><td>Method</td><td>Venue</td><td>EER↓</td><td>FAR↓</td><td>FRR↓</td><td>Acc ↑</td></tr><tr><td>SigNet [41]</td><td>arXiv&#x27;17</td><td>15.36</td><td>15.36</td><td>15.36</td><td>84.64</td></tr><tr><td>SigNet-F [40]</td><td>PR&#x27;17</td><td>18.30</td><td>18.22</td><td>18.37</td><td>81.71</td></tr><tr><td>FMPNet [51]</td><td>ICFHR&#x27;18</td><td>14.28</td><td>14.17</td><td>14.48</td><td>85.70</td></tr><tr><td>DeepHSV [42]</td><td>ICDAR&#x27;19</td><td>13.34</td><td></td><td></td><td>86.66</td></tr><tr><td>IDN [6]</td><td>CVPR&#x27;19</td><td></td><td>8.99</td><td>4.93</td><td>93.04</td></tr><tr><td>SDINet [67]</td><td>AAAI&#x27;21</td><td></td><td>6.24</td><td>3.77</td><td>95.00</td></tr><tr><td>CaP [68]</td><td>ICPR&#x27;21</td><td>5.97</td><td>5.97</td><td>5.97</td><td></td></tr><tr><td>AVN [44]</td><td>TMM&#x27;22</td><td>5.65</td><td></td><td></td><td>94.32</td></tr><tr><td>2C2S [48]</td><td>EAAI&#x27;23</td><td>9.32</td><td>8.66</td><td>9.98</td><td>90.68</td></tr><tr><td>MGRNet [21]</td><td>arXiv&#x27;23</td><td>6.68</td><td>6.76</td><td>6.56</td><td></td></tr><tr><td>LwR [69]</td><td>PR&#x27;23</td><td>8.80</td><td></td><td></td><td>92.60</td></tr><tr><td>TransOSV [5]</td><td>PR&#x27;24</td><td>3.24</td><td>3.24</td><td>3.24</td><td></td></tr><tr><td>SPD Manifold [39]</td><td>TIFS&#x27;24</td><td>15.60</td><td></td><td></td><td></td></tr><tr><td>CWA [45]</td><td>ICDAR&#x27;24</td><td>9.55</td><td>7.11</td><td>12.14</td><td>91.42</td></tr><tr><td>DetailSemNet [46]</td><td>ECCV&#x27;24</td><td>13.31</td><td>7.89</td><td>22.32</td><td>88.11</td></tr><tr><td>HTCSigNet [47]</td><td>PR&#x27;25</td><td>4.63</td><td>4.61</td><td>4.65</td><td>95.26</td></tr><tr><td>Baseline (Ours)</td><td>This work</td><td>6.88 (±1.31)</td><td>6.89 (±1.31)</td><td>6.88 (±1.31)</td><td>93.11 (±1.31)</td></tr><tr><td>SigDiffuserval (Ours)</td><td>This work</td><td>3.65 (±0.44)</td><td>3.65 (±0.44)</td><td>3.65 (±0.44)</td><td>96.35 (±0.44)</td></tr><tr><td>SigDiffuser (Ours)</td><td>This work</td><td>3.13 (±0.33)</td><td>3.13 (±0.33)</td><td>3.13 (±0.33)</td><td>96.87 (±0.33)</td></tr></table>

TABLE VI: Performance comparison on UTSig [22].
<table><tr><td>Method</td><td>Venue</td><td>EER↓</td><td>FAR ↓</td><td>FRR↓</td><td>Acc ↑</td></tr><tr><td>UTSig-S1 [22]</td><td>IET Biom&#x27;16</td><td>29.71</td><td>21.29</td><td>39.27</td><td></td></tr><tr><td>SigNet [41]</td><td>arXiv&#x27;17</td><td>47.08</td><td>46.71</td><td>47.38</td><td>53.05</td></tr><tr><td>SigNet-F [40]</td><td>PR&#x27;17</td><td>43.20</td><td>42.93</td><td>43.64</td><td>56.81</td></tr><tr><td>FMPNet [51]</td><td>ICFHR&#x27;18</td><td>32.95</td><td>32.81</td><td>33.04</td><td>67.11</td></tr><tr><td>LwR [69]</td><td>PR&#x27;23</td><td>29.70</td><td></td><td></td><td>72.70</td></tr><tr><td>DetailSemNet [46]</td><td>ECCV&#x27;24</td><td>39.54</td><td>0.78</td><td>96.31</td><td>77.80</td></tr><tr><td>HTCSigNet [47]</td><td>PR&#x27;25</td><td>28.93</td><td>28.92</td><td>28.96</td><td>71.08</td></tr><tr><td>Baseline (Ours)</td><td>This work</td><td>29.81 (±1.51)</td><td>29.81 (±1.51)</td><td>29.81 (±1.51)</td><td>70.19 (±1.51)</td></tr><tr><td>SigDiffuserval (Ours)</td><td>This work</td><td>27.39 (±1.62)</td><td>27.38 (±1.61)</td><td>27.40 (±1.62)</td><td>72.61 (±1.62)</td></tr><tr><td>SigDiffuser (Ours)</td><td>This work</td><td>23.48 (±1.81)</td><td>23.48 (±1.81) 23.49 (±1.82) 76.52 (±1.81)</td><td></td><td></td></tr></table>

Results are summarized in Table II. (1) SigDiffuser consistently achieves the best performance in both probe and finetuning evaluations. Compared to the randomly initialized baseline, the pretrained model achieves superior mean performance and generally lower across-split variance, demonstrating that our pretraining extracts high-quality representations, directly improving verification stability. (2) Most competing methods, even those with pretext tasks tailored for handwritten/scene text self-supervised representation learning (SSRL) [19, 62, 63, 64], perform significantly worse, exhibiting lower means and higher across-split variance. SigDiffuser also surpasses other diffusion-based schemes, such as DiffAE and SODA, by considerable margins. (3) Models using their original encoders, whether deeper (ResNet-50), modern-designed (ConvNeXt-Small), or lightweight (VGG-10), consistently underperform our default ResNet-18. Although native architectures align better with their vanilla designs, these results indicate that ResNet-18 provides a superior balance of capacity and architectural suitability for SSRL in HSV.

(4) Generative methods generally outperform contrastive methods, suggesting contrastive learning is less suited for HSV. Among generative methods, diffusion-based approaches mostly surpass MIM-based models, consistent with the blurred reconstruction results shown in Fig. 1 (a). This is potentially due to MIM’s difficulty in perceiving fine textures amidst white backgrounds when large portions of strokes are masked. (5) SigDiffuser’s superiority mainly stems from the proposed RFR. Unlike pretext tasks with mild deformations that permit local pixel-copying shortcuts, RFR requires the model to restore a severely ruined signature region guided by a spatially distinct style patch. As discussed in Sec. III-A, this design suppresses reconstruction shortcuts and forces reliance on intrinsic writing-style evidence encoded from $I _ { s t y }$ . Hence, the encoder captures writing-style cues across spatial locations (e.g., stroke dynamics, spacing, and slant) rather than trivial pixel relations, which are essential for HSV. Further visual analyses are presented in Sec. IV-H.

TABLE VIII: Ablation study on BHSig260-B [20] and BHSig260-H [20]. SSRL denotes self-supervised representation learning.
<table><tr><td rowspan="2">#Line</td><td rowspan="2">Case</td><td colspan="4">BHSig260-B</td><td colspan="4">BHSig260-H</td></tr><tr><td>EER↓</td><td>FAR↓</td><td>FRR↓</td><td>Acc ↑</td><td>EER↓</td><td>FAR↓</td><td>FRR↓</td><td>Acc ↑</td></tr><tr><td>1</td><td>Baseline</td><td>5.77</td><td>5.78</td><td>5.76</td><td>94.23</td><td>5.95</td><td>5.96</td><td>5.95</td><td>94.04</td></tr><tr><td>2</td><td>Baseline w/ RFR-based SSRL; w/  $I _ { c o r }$ </td><td>2.60</td><td>2.60</td><td>2.60</td><td>97.40</td><td>5.18</td><td>5.19</td><td>5.18</td><td>94.82</td></tr><tr><td>3</td><td>Baseline w/ RFR-based SSRL; w/  $I _ { c o r } ;$  w/ PFAM</td><td>1.58</td><td>1.58</td><td>1.57</td><td>98.43</td><td>2.59</td><td>2.59</td><td>2.58</td><td>97.41</td></tr><tr><td>4</td><td>Baseline w/ RFR-based SSRL; w/o  $I _ { c o r } ;$  w/ PFAM</td><td>4.34</td><td>4.34</td><td>4.33</td><td>95.66</td><td>3.31</td><td>3.31</td><td>3.31</td><td>96.69</td></tr><tr><td>5</td><td>Baseline w/ MIM-based SSRL; w/  $I _ { c o r } ;$  w/ PFAM</td><td>3.39</td><td>3.39</td><td>3.39</td><td>96.61</td><td>4.13</td><td>4.13</td><td>4.13</td><td>95.87</td></tr></table>

## D. Comparison with Existing HSV Method

We compare the full-parameter fine-tuned model with SOTA HSV methods in Tables III to VI. For fair comparison, we only include prior results obtained under identical training/testing splits and evaluation settings. For instance, we report singlereference rather than multi-reference results for HTCSigNet [47] and the 50/50 writer split for TransOSV [5] on BHSig260- B (which also reports 80/20). Here we report the multi-split results, with per-split details provided in Appendix B.

As confirmed by the results, even under the multi-split protocol, SigDiffuser outperforms preceding models in almost all cases. On BHSig260-B and BHSig260-H (Tables III and IV), SigDiffuser sets new records by improving EERs from 2.53% and 3.24% to 1.85% and 3.13%, respectively. Although the second-best method, CWA, uses an 80/20 writer split (easier than our 50/50 split), SigDiffuser maintains its superiority. HanSig is a challenging Chinese HSV dataset characterized by significant variation in writing style. Despite the heightened difficulty, SigDiffuser outperforms all existing methods in Table V. On UTSig (Table VI), SigDiffuser shows significant advantages over existing methods, even approaching the performance of 12-reference methods despite using only one reference. It is noteworthy that the signatures across different datasets diverge drastically in text structure and writing styles due to language and cultural discrepancies. Overall, the strong average results across multiple data splits demonstrate SigDiffuser’s effectiveness in learning informative writing patterns that transcend linguistic and stylistic boundaries.

We further evaluate on the CEDAR dataset. Several prior works [41, 42, 45, 48] report seemingly perfect metrics (0% EER and 100% Acc) by relying on trivial background cues (genuine signatures are gray, forgeries are white) rather than actual strokes. To prevent this, we remove all backgrounds during data preprocessing. Table VII shows that RFR pretraining substantially improves SigDiffuser over the baseline. Notably, although the inherent difficulty of certain writers and samples in CEDAR causes high performance variance, our pretraining effectively reduces this instability (±6.91→±5.88). While not achieving the lowest EER, SigDiffuser remains competitive.

![](images/07368a441759967c5e4c0e679a6530030b3d0a4153e1517987de362ebd12bc2e.jpg)  
(a) Corruption Degree Comparison (Setting 1)

Fig. 4: Visualized comparison of varying corruption degrees.

Other than standard data splits, we also evaluate using splits with validation (Sec. IV-A1), denoted as SigDiffuser<sup>val</sup>. As observed, performance consistently drops across all datasets. We attribute this degradation to the reduced training data volume. Since offline HSV datasets are inherently small, mainstream methods typically use only training-testing splits to maximize training data; introducing a validation set inevitably hampers performance. Despite this decline, SigDiffuser<sup>val</sup> maintains optimal or near-optimal performance on most datasets.

## E. Ablation Study

We analyze the effect of SigDiffuser’s key components, including the RFR pretext task, the pyramidal feature aggregation module (PFAM), and the corrupted signature input $I _ { c o r } ,$ on a single data split, with results presented in Table VIII. Comparing Lines 1 and 2, self-supervised learning with RFR effectively improves HSV performance. In Line 3, adding PFAM to form the complete SigDiffuser further improves EER by 1.02% and 2.59%, confirming PFAM’s efficacy.

Additionally, since using $I _ { s t y }$ and $I _ { o r g }$ also enables representation learning, we investigate the efficacy and necessity of introducing the corrupted signature region $I _ { c o r }$ . Conceptually, removing $I _ { c o r }$ degrades our RFR pretext task to novel view synthesis. As shown in Lines 3 and 4, removing $I _ { c o r }$ leads to significant performance declines, especially on BHSig260-B. This suggests that reconstructing from the corrupted signatures $I _ { c o r }$ is effective to boost model performance, underscoring its necessity as RFR’s core component. Furthermore, substituting RFR with MIM drops overall performance (Lines 3 vs. 5). This indicates RFR’s superiority over MIM in HSV modeling, aligning with our findings in Sec. IV-C. Additional ablation studies regarding PFAM’s implementation variants, the isolated effect of diffusion modeling in SigDiffuser, the number of pretraining epochs, and patch side length l used in RFR are detailed in Appendices C, D, E, and F, respectively.

## F. Cross-Dataset Generalization Analysis

To investigate cross-domain generalization of SigDiffuser, we conduct cross-dataset validation on four datasets, as shown in Table IX. We measure performance degradation from source to target domain via accuracy drop $( \Delta _ { \mathrm { A c c } } )$ . All methods suffer noticeable degradation when transferred to unseen target datasets, with drops ranging from 3% to 43%. The decline is attributable to the domain-shift problem inherent in HSV [70, 73], where variations in language, writing style, and data volume introduce significant discrepancies between datasets. Despite this challenging setting, SigDiffuser consistently yields smaller $\Delta _ { \mathrm { A c c } }$ than the baseline across different source-target pairs. For instance, when trained on BHSig260- H, it reduces the accuracy drops by 1.44%, 5.85%, and 4.56% on BHSig260-B, CEDAR, and HanSig, respectively. It also achieves the smallest performance drop among compared methods in many settings, such as BHSig260-H→BHSig260- B and HanSig→CEDAR. These results indicate that SigDiffuser degrades more gracefully under domain shift than both the baseline and existing HSV methods.

TABLE IX: Cross-dataset validation. Models are trained on the source dataset and tested on the target dataset. $\Delta _ { \mathrm { A c c } }$ denotes the accuracy drop from the best in-domain accuracy, where smaller values indicate better cross-dataset generalization
<table><tr><td rowspan="3">Source</td><td rowspan="3">Model</td><td rowspan="3">Venue</td><td colspan="10">Target</td></tr><tr><td colspan="3">BHSig260-B</td><td colspan="3">BHSig260-H</td><td colspan="2">CEDAR</td><td colspan="3"></td></tr><tr><td>EER↓</td><td>Acc ↑</td><td> $\Delta _ { \mathrm { A c c } } \downarrow$ </td><td>EER↓</td><td>Acc ↑  ${ \underline { { \Delta _ { \mathrm { A c c } } } } } \downarrow$ </td><td></td><td>EER↓</td><td>Acc ↑  $\Delta _ { \mathrm { A c c } } \downarrow$ </td><td>EER↓</td><td>HanSig Acc ↑</td><td> ${ \underline { { \Delta _ { \mathrm { A c c } } } } } \downarrow$ </td></tr><tr><td rowspan="7">BHSig260-B</td><td>SigNet [41]</td><td>arXiv&#x27;17</td><td>33.89</td><td>86.11 0.00</td><td></td><td>52.78</td><td>33.33</td><td></td><td>50.00</td><td>36.11</td><td></td><td></td><td></td></tr><tr><td>IDN [6]</td><td>CVPR&#x27;19</td><td></td><td>95.32 0.00</td><td></td><td>74.30</td><td>21.02</td><td></td><td>50.00</td><td>45.32</td><td></td><td></td><td></td></tr><tr><td>CaP [68]</td><td>ICDAR&#x27;21</td><td>3.96</td><td>96.04</td><td>0.00</td><td>69.59</td><td>26.45</td><td></td><td>66.29</td><td>29.75</td><td></td><td></td><td></td></tr><tr><td>ILNNeck [70]</td><td>IJCNN&#x27;22</td><td>5.72</td><td>94.28</td><td>0.00</td><td>79.21</td><td>15.07</td><td></td><td>80.34</td><td>13.94</td><td></td><td></td><td></td></tr><tr><td>DetailSemNet [46]</td><td>ECCV’24</td><td>10.19</td><td>90.64</td><td>0.00</td><td>26.00 78.69</td><td>11.95</td><td>33.40</td><td>72.35</td><td>18.29</td><td>36.01</td><td>70.41</td><td>20.23</td></tr><tr><td>Baseline (Ours)</td><td>This work</td><td>5.77</td><td>94.23</td><td>0.00</td><td>22.69 77.32</td><td>16.91</td><td>28.85</td><td>71.15</td><td>23.08</td><td>24.70</td><td>75.31</td><td>18.92</td></tr><tr><td>SigDiffuser (Ours)</td><td>This work</td><td>1.58</td><td>98.43</td><td>0.00</td><td>12.44 87.56</td><td>10.87</td><td>16.95</td><td>83.05</td><td>15.38</td><td>19.06</td><td>80.94</td><td>17.49</td></tr><tr><td rowspan="7">BHSig260-H</td><td>SigNet</td><td>arXiv&#x27;17</td><td></td><td>60.65</td><td>23.99 18.92</td><td>15.36 84.64</td><td>0.00</td><td></td><td>59.57</td><td>25.07</td><td></td><td></td><td></td></tr><tr><td>IDN</td><td>CVPR&#x27;19</td><td></td><td>74.12</td><td></td><td>93.04</td><td>0.00</td><td></td><td>50.00</td><td>43.04</td><td></td><td></td><td></td></tr><tr><td>CaP</td><td>ICDAR&#x27;21</td><td></td><td>85.34</td><td>8.69</td><td>5.97 94.03</td><td>0.00</td><td></td><td>70.51</td><td>23.52</td><td></td><td></td><td></td></tr><tr><td>ILNNeck</td><td>IJCNN&#x27;22</td><td></td><td>87.57</td><td>8.22</td><td>4.21 95.79</td><td>0.00</td><td></td><td>79.76</td><td>16.03</td><td></td><td></td><td></td></tr><tr><td>DetailSemNet</td><td>ECCV’24</td><td>22.07</td><td>80.52</td><td>7.59</td><td>13.31 88.11</td><td>0.00</td><td>35.94</td><td>71.13</td><td>16.98</td><td>36.46</td><td>69.72</td><td>18.39</td></tr><tr><td>Baseline (Ours)</td><td>This work</td><td>11.27</td><td>88.73</td><td>5.31</td><td>5.95 94.04</td><td>0.00</td><td>21.53</td><td>78.47</td><td>15.57</td><td>24.93 20.37</td><td>75.07</td><td>18.97</td></tr><tr><td>SigDiffuser (Ours)</td><td>This work</td><td>6.45</td><td>93.55</td><td>3.86</td><td>2.59 97.41</td><td>0.00</td><td>15.68</td><td>84.32</td><td>13.09</td><td></td><td>79.63</td><td>17.78</td></tr><tr><td rowspan="6">CEDAR</td><td>IDN</td><td>CVPR&#x27;19</td><td></td><td>50.01</td><td>46.37</td><td></td><td>50.36 46.02</td><td>3.62</td><td>96.38</td><td>0.00</td><td></td><td></td><td></td></tr><tr><td>CaP</td><td>ICDAR&#x27;21</td><td></td><td>61.93</td><td>33.63 5.86</td><td>60.92</td><td>35.46</td><td>4.34</td><td>95.56</td><td>0.00</td><td></td><td></td><td></td></tr><tr><td>ILNNeck</td><td>IJCNN&#x27;22</td><td></td><td>84.93</td><td></td><td>80.83</td><td>9.96</td><td>9.21</td><td>90.79</td><td>0.00</td><td></td><td></td><td></td></tr><tr><td>DetailSemNet</td><td>ECCV&#x27;24</td><td>32.00</td><td>73.75</td><td>5.78</td><td>33.67 73.40</td><td>6.13</td><td>24.61</td><td>79.53</td><td>0.00</td><td>39.71</td><td>68.19</td><td>10.34</td></tr><tr><td>Baseline (Ours)</td><td>This work</td><td>32.54</td><td>67.46</td><td>27.42</td><td>32.25 67.70</td><td>27.18</td><td>5.11</td><td>94.88</td><td>0.00</td><td>27.70</td><td>72.33</td><td>22.55</td></tr><tr><td>SigDiffuser (Ours)</td><td>This work</td><td>21.08</td><td>78.92</td><td>18.62</td><td>29.04 70.97</td><td>26.57</td><td>2.46</td><td>97.54</td><td>0.00</td><td>23.90</td><td>76.10</td><td>21.44</td></tr><tr><td rowspan="4">HanSig</td><td>FMPNet [51]</td><td>ICFHR&#x27;18</td><td>23.16</td><td>76.83</td><td>9.53 4.45</td><td>26.52 21.92</td><td>73.47 12.37</td><td>22.25</td><td>77.75</td><td>8.10</td><td>14.15</td><td>85.85</td><td>0.00</td></tr><tr><td>DetailSemNet</td><td>ECCV’24</td><td>20.85</td><td>80.72</td><td></td><td>80.46 69.82</td><td>4.71</td><td>22.54</td><td>81.01 83.40</td><td>4.16</td><td>16.00</td><td>85.17</td><td>0.00</td></tr><tr><td>Baseline (Ours)</td><td>This work</td><td>14.61</td><td>85.39</td><td>4.53</td><td>30.18 25.99</td><td>20.10 18.62</td><td>16.59</td><td>88.69</td><td>6.52 3.95</td><td>10.07</td><td>89.92</td><td>0.00</td></tr><tr><td>SigDiffuser (Ours)</td><td>This work</td><td>10.66</td><td>89.34</td><td>3.30</td><td></td><td>74.02</td><td>11.32</td><td></td><td></td><td>7.36</td><td>92.64</td><td>0.00</td></tr></table>

TABLE X: Comparison of corruption degrees on BHSig260-B [20] and BHSig260-H [20]. (a) Setting 1. We use α to control the crumpling severity, ranging from 0 (no crumpling) to 27,108 (highly intensified). (b) Setting 2. We use existing dataaugmentation techniques to control the crumpling degree: None (SimAN [19]), Mild (CMT-Co [64]), and Full (our RFR).

(a) Setting 1: Controlled by α as described in Sec. III-A.
<table><tr><td rowspan="2">α</td><td colspan="4"> $\mathrm { B H S i g 2 6 0 \mathrm { - } B }$ </td><td colspan="4">BHSig260-H</td></tr><tr><td>EER↓</td><td>FAR↓</td><td>FRR↓</td><td>Acc ↑</td><td>EER↓</td><td>FAR↓</td><td>FRR↓</td><td>Acc ↑</td></tr><tr><td>0</td><td>6.39</td><td>6.39</td><td>6.39</td><td>93.61</td><td>5.42</td><td>5.41</td><td>5.42</td><td>94.59</td></tr><tr><td>9,036</td><td>1.88</td><td>1.88</td><td>1.88</td><td>98.12</td><td>3.21</td><td>3.21</td><td>3.21</td><td>96.79</td></tr><tr><td>18,072</td><td>1.58</td><td>1.58</td><td>1.57</td><td>98.43</td><td>2.59</td><td>2.59</td><td>2.59</td><td>97.41</td></tr><tr><td>27,108</td><td>2.19</td><td>2.19</td><td>2.19</td><td>97.81</td><td>3.05</td><td>3.05</td><td>3.05</td><td>96.95</td></tr></table>

Nevertheless, the absolute cross-dataset performance remains limited. SigDiffuser’s accuracy falls below 80% in 5 of the 12 cross-dataset settings, indicating a substantial gap from its in-domain performance (mostly above 92%). This difficulty is not specific to our method: under the same protocol, previous SOTA methods such as ILNNeck and DetailSemNet exhibit comparable or more severe degradation, as evidenced by DetailSemNet falling below 80% in 8 of the 12 settings. These observations suggest that cross-domain transfer in HSV remains an open problem, where inherent morphological differences between scripts introduce representation gaps that current methods, including ours, do not fully resolve.

(b) Setting 2: Controlled by existing implementations.
<table><tr><td rowspan="2">Degree</td><td colspan="4">BHSig260-B</td><td colspan="4">BHSig260-H</td></tr><tr><td>EER↓</td><td>FAR↓</td><td>FRR↓</td><td>Acc ↑</td><td>EER↓</td><td>FAR↓</td><td>FRR↓</td><td>Acc ↑</td></tr><tr><td>None</td><td>4.59</td><td>4.59</td><td>4.59</td><td>95.41</td><td>5.73</td><td>5.73</td><td>5.72</td><td>94.27</td></tr><tr><td>Mild</td><td>2.00</td><td>2.00</td><td>2.01</td><td>98.00</td><td>3.68</td><td>3.68</td><td>3.68</td><td>96.32</td></tr><tr><td>Full</td><td>1.58</td><td>1.58</td><td>1.57</td><td>98.43</td><td>2.59</td><td>2.59</td><td>2.59</td><td>97.41</td></tr></table>

Beyond script differences, cross-dataset generalization is further influenced by experimental factors. Following previous studies, we adopt the standard training/testing splits for each dataset, which results in varying numbers of training writers (e.g., 50 for CEDAR and 100 for BHSig260-H). This leads to asymmetric generalization behaviors, where models trained on larger splits naturally acquire richer representations. For instance, BHSig260-B and BHSig260-H share similar collection environments, but transferring from the larger BHSig260- H (100 training writers) to BHSig260-B (50 training writers) yields a much smaller performance drop than the reverse. This confirms that scaling up training writers helps mitigate the domain gap. This advantage is further magnified by writer diversity in HanSig. Results show that transferring to HanSig incurs larger drops than transferring from it to other datasets. This stems from its large training split (795 names, 31,800 samples), which exposes models to far more diverse writing styles, resulting in better cross-domain results.

![](images/f58f7310c5884bb784f6ce9f10def4e2870837c02f263d7e8b6ef9223be5fae2.jpg)  
(a) BHSig260-B.

![](images/91a76077c69001f03e08d1cc42c6ff65b2bc7280e7599b940af3ecb5cc3201c2.jpg)  
(b) BHSig260-H.

Fig. 5: Grad-CAM [74] visualizations of downstream models. While MAE produces sharp responses on local, isolated strokes, SigDiffuser highlights broader, discriminative structures like stroke connections and turns, capturing rich personal writing styles.  
![](images/d42b0325e0db031e9c6083c9200e0c38ab38b4afd73b51725bbab355469ecdf5.jpg)  
(a) BHSig260-B: 12/24 samples per writer for 25/50 tested writers.

![](images/f089643bca8303af0d330c10cdede18dff5df5885aa5eb6a41d6f92c50ed67fb.jpg)  
(b) BHSig260-H. 12/24 samples per writer for 30/60 tested writers.  
Fig. 6: t-SNE visualizations of downstream HSV embeddings (partial samples shown; full results in Appendix H). We use the Silhouette score [75] to quantify the separation degree (higher is better). Compared to the baseline and MAE, SigDiffuser forms more compact writer-level clusters with significantly higher scores, demonstrating superior feature separability.

## G. Effectiveness Analysis of RFR’s Crumpling Strategy

Given the unprecedentedly severe crumpling applied to signatures in RFR, we explore the effectiveness of crumpling degrees through two strategies. (1) Scale factor control. We vary α in Eq. 1 across 0, 9, 036, 18, 072, and 27, 108, representing moderate to severe crumpling. (2) Existing augmentation control. We apply data augmentation methods from SimAN [19] and CMT-Co [64] on clean signatures to emulate no spatial distortion (None) and mild crumpling (Mild), with vanilla RFR crumpling as Full. These proxies offer an alternative yet convincing evaluation of crumpling severity, with visualized and quantitative results shown in Fig. 4 and Table X.

Table X (a) shows that performance improves as α increases, but excessive stroke warping (α = 27, 108) exceeds reconstruction capacity and degrades results. Table X (b) similarly shows that amplifying distortion progressively improves model performance. The consistency under both settings implies the effectiveness of introducing severe crumpling in RFR for empowering signature representation learning, which is a core difference against other generative methods that apply mild deformation [19] or simple noise addition [14, 15, 16]. In practice, one can tune α to find the optimal crumpling degree for application. Furthermore, we investigate employing forged signatures as natural crumpling in comparison to the artificial crumpling of RFR in Appendix G, which yields similar findings that reinforce RFR’s efficacy.

## H. Visualization

We first evaluate feature discriminability using Grad-CAM [74] and t-SNE. Fig. 5 shows Grad-CAM heatmaps using the ground-truth writer identity as targets. As observed, the baseline (no pretraining) generally fails to focus on valid signatures, while MAE (pretrained with MIM) sparsely activates on isolated stroke fragments, indicating shortcut learning based on superficial patterns. In contrast, SigDiffuser (pretrained with RFR) accurately attends to structurally significant regions like stroke connections and turns, which reflect intrinsic, unconscious personal writing habits. This confirms that the severe corruption design in RFR successfully suppresses direct reconstruction shortcuts, forcing the model to capture stable, discriminative writing-style evidence.

Subsequently, we present t-SNE visualizations of genuine signature features in Fig. 6 (partial samples shown for clarity; full results in Appendix H), illustrating how this attention translates into the global feature space. While baseline and MAE features exhibit noticeable class overlap and scattered distributions, SigDiffuser forms compact, well-separated writer clusters. This demonstrates that SigDiffuser effectively minimizes intra-writer variations and maximizes inter-writer separability, yielding discriminative representations. We further visualize SigDiffuser’s reconstruction examples and failure cases in Appendices I and J.

## V. LIMITATION

Apart from the strong HSV performance, some potential limitations warrant discussion. First, SigDiffuser primarily targets enhancing representation learning rather than image synthesis. Compared to dedicated image synthesis methods [16, 76], SigDiffuser may falter in producing a disentangled latent space and struggle with controllable image generation such as attribute manipulation. Second, while incorporating a broader range of signature data could further boost pretraining performance, we adhere to standard self-supervised protocol [12, 16] that restrict pretraining to the same dataset as finetuning to ensure fair comparisons. Third, while SigDiffuser achieves leading performance under in-domain settings (Tables II to VI), its absolute accuracy under cross-domain transfer (e.g., Bengali↔English↔Chinese) remains limited, as discussed in Sec. IV-F. This is a field-wide challenge in HSV, and designing pretext tasks or domain-adaptation techniques that yield domain-invariant representations remains an important direction for future work.

## VI. CONCLUSION

We propose SigDiffuser, a novel diffusion-based representation learning method for handwritten signature verification (HSV), comprising a Reconstruction from Ruins (RFR) pretext task and a pyramidal feature aggregation module (PFAM). RFR introduces severe crumpling corruption on signatures, creating a challenging reconstruction problem that compels the model’s deep understanding of discriminative writing patterns. PFAM aggregates multi-scale encoder features into latent embeddings, elevating representation quality.

Experiments across multiple HSV datasets demonstrate SigDiffuser’s superior performance and adaptability across diverse linguistic and stylistic contexts. We reveal that increasing crumpling severity yields better signature representations, validating RFR’s effectiveness. Unlike existing works [14, 16] that apply simple noise on clean images, our work innovatively proves that extreme destruction (i.e., severe crumpling) can enhance latent encoding for more informative representations. By leveraging denoising diffusion to reconstruct severely “ruined” signatures, we naturally bridge diffusion models and self-supervised learning, paving a new avenue for signature representation and potentially inspiring advances in other biometric domains, such as fingerprint verification.

## APPENDIX A

## DATA PREPROCESSING

For data preprocessing, we first convert signature images into grayscale and remove the signature background with white pixel substitution. Gaussian filtering is then applied to filter out noise. Next, we excise the peripheral white pixel borders and resize the signature images to a fixed size of 96×336 (height × width), preserving the aspect ratio through white pixel padding until the preset dimensions are achieved to avoid distortions. Consequently, during pretraining, the patch side length l is set to 96, resulting in a size of 96 × 96 for cropped image patches $I _ { o r g } , I _ { c o r }$ , and $I _ { s t y }$ . During downstream fine-tuning, we utilize $9 6 \times 3 3 6$ as the standard input size across all datasets.

## APPENDIX B

## PER-SPLIT PERFORMANCE OF SIGDIFFUSER

For transparency and reproducibility, we present the persplit performance of SigDiffuser on different datasets, along with comparisons against existing SOTA models. Results are presented in Tables XI to XV. As observed, SigDiffuser surpasses previous SOTA in most single-split cases, especially on BHSig260-B, BHSig260-H, HanSig, and UTSig. This demonstrates that our model’s superiority holds not only in the multi-split average but also across single splits. Yet, the EER on CEDAR fluctuates from 1.59% (Split-1) to 18.55% (Split-11), resulting in a suboptimal multi-split average. This sensitivity to data partitioning highlights an inherent limitation of traditional single-split protocols, where performance can be heavily biased by a specific random split. Therefore, it underscores the necessity of our multi-split evaluation for a more objective and reliable assessment.

TABLE XI: Per-split performance on BHSig260-B [20].
<table><tr><td>Model</td><td>EER↓</td><td>FAR↓</td><td>FRR↓</td><td>Acc↑</td></tr><tr><td>Existing SOTA Model</td><td></td><td></td><td></td><td></td></tr><tr><td>CWA [45]</td><td>2.53</td><td>2.95</td><td>2.14</td><td>97.27</td></tr><tr><td>Ours</td><td></td><td></td><td></td><td></td></tr><tr><td>Split-1</td><td>1.59</td><td>1.59</td><td>1.59</td><td>98.41</td></tr><tr><td>Split-2</td><td>2.45</td><td>2.44</td><td>2.45</td><td>97.55</td></tr><tr><td>Split-3</td><td>2.23</td><td>2.23</td><td>2.22</td><td>97.77</td></tr><tr><td>Split-4</td><td>1.40</td><td>1.40</td><td>1.39</td><td>98.60</td></tr><tr><td>Split-5</td><td>1.58</td><td>1.58</td><td>1.57</td><td>98.43</td></tr><tr><td>Mean (±std)</td><td>1.85 (±0.46)</td><td>1.85 (±0.46)</td><td>1.84 (±0.45)</td><td>98.15 (±0.46)</td></tr></table>

## APPENDIX C

## EXTENDED ABLATION STUDY: HIERARCHICAL ARCHITECTURE IMPLEMENTATION

To validate the effectiveness of the proposed pyramidal feature aggregation Module (PFAM) in hierarchical latent encoding, we conduct an ablation study regarding the implementation of multi-scale feature aggregation architectures. We implement three alternative hierarchical approaches to substitute the PFAM inside SigDiffuser, including channel-wise concatenation, cross-attention aggregation, and hierarchical bottlenecks. The first two directly integrate multi-scale features into decoder layers through concatenation or cross-attention. The hierarchical bottlenecks, which are an extension of the single bottleneck of SODA [16], compress multi-scale features into latent vectors and integrate them into decoder layers through addition. Table XVI presents the comparative results. It can be seen that PFAM exhibits the optimal performance among the four methods, underscoring its superiority. By cumulatively compounding multi-scale handwriting feature maps, PFAM produces more informative latent representations than other architectures that linearly use multi-scale features without further aggregation. The enriched latent representations, in turn, enhance the encoder’s feature extraction capabilities, leading to more effective representation learning.

TABLE XII: Per-split performance on BHSig260-H [20].
<table><tr><td>Model</td><td>EER↓</td><td>FAR↓</td><td>FRR↓</td><td>Acc↑</td></tr><tr><td>Existing SOTA Model</td><td></td><td></td><td></td><td></td></tr><tr><td>TransOSV [5]</td><td>3.24</td><td>3.24</td><td>3.24</td><td>-</td></tr><tr><td>Ours</td><td></td><td></td><td></td><td></td></tr><tr><td>Split-1</td><td>3.13</td><td>3.13</td><td>3.13</td><td>96.87</td></tr><tr><td>Split-2</td><td>3.31</td><td>3.31</td><td>3.31</td><td>96.69</td></tr><tr><td>Split-3</td><td>3.44</td><td>3.44</td><td>3.44</td><td>96.56</td></tr><tr><td>Split-4</td><td>3.18</td><td>3.18</td><td>3.19</td><td>96.82</td></tr><tr><td>Split-5</td><td>2.59</td><td>2.59</td><td>2.58</td><td>97.41</td></tr><tr><td>Mean (±std)</td><td colspan="4">3.13 (±0.33) 3.13 (±0.33) 3.13 (±0.33) 96.87 (±0.33)</td></tr></table>

TABLE XIII: Per-split performance on HanSig [21].
<table><tr><td>Model</td><td>EER↓</td><td>FAR↓</td><td>FRR↓</td><td>Acc↑</td></tr><tr><td>Existing SOTA Model</td><td></td><td></td><td></td><td></td></tr><tr><td>MGRNet [21]</td><td>9.93</td><td>11.85</td><td>7.69</td><td></td></tr><tr><td>CWA [45]</td><td>9.47</td><td>6.61</td><td>16.23</td><td>90.26</td></tr><tr><td>Ours</td><td></td><td></td><td></td><td></td></tr><tr><td>Split-1</td><td>5.23</td><td>5.23</td><td>5.25</td><td>94.76</td></tr><tr><td>Split-2</td><td>7.36</td><td>7.36</td><td>7.36</td><td>92.64</td></tr><tr><td>Split-3</td><td>7.13</td><td>7.13</td><td>7.13</td><td>92.87</td></tr><tr><td>Split-4</td><td>5.81</td><td>5.81</td><td>5.81</td><td>94.19</td></tr><tr><td>Split-5</td><td>6.69</td><td>6.69</td><td>6.69</td><td>93.31</td></tr><tr><td>Split-6</td><td>6.84</td><td>6.84</td><td>6.83</td><td>93.16</td></tr><tr><td>Split-7</td><td>8.10</td><td>8.10</td><td>8.11</td><td>91.90</td></tr><tr><td>Split-8</td><td>6.83</td><td>6.82</td><td>6.83</td><td>93.18</td></tr><tr><td>Split-9</td><td>11.00</td><td>10.99</td><td>11.00</td><td>89.01</td></tr><tr><td>Split-10</td><td>8.89</td><td>8.89</td><td>8.89</td><td>91.11</td></tr><tr><td>Mean (±std)</td><td>7.39 (±1.64)</td><td>7.39 (±1.63)</td><td>7.39 (±1.64)</td><td>92.61 (±1.64)</td></tr></table>

## APPENDIX D

## EXTENDED ABLATION STUDY: EFFECT OF DIFFUSION MODELING IN SIGDIFFUSER

In the RFR pretext task, SigDiffuser leverages the iterative denoising mechanism of diffusion to reconstruct severely crumpled signature patches for representation learning. To isolate the effect of the diffusion process, we remove it from the framework for an ablation. In this case, the framework degenerates into a single-step image-to-image translation model conditioned on an extra encoder. As shown in Table XVII, performance drops significantly without the diffusion process. This is because restoring a heavily crumpled signature is far beyond the capability of a single-step mapping, which consequently hampers representation learning. In contrast, the multi-step iterative denoising of diffusion effectively handles such reconstruction, enabling the model to learn meaningful representations. These results highlight the indispensable role of diffusion modeling in our method, resonating with our motivation for coupling the RFR pretext task with a diffusionbased framework.

TABLE XIV: Per-split performance on UTSig [22].
<table><tr><td>Model</td><td>EER↓</td><td>FAR↓</td><td>FRR↓</td><td>Acc↑</td></tr><tr><td colspan="5">Existing SOTA Model</td></tr><tr><td>UTSig-S1 [22]</td><td>29.71</td><td>21.29</td><td>39.27</td><td>=</td></tr><tr><td>HTCSigNet [47]</td><td>28.93</td><td>28.92</td><td>28.96</td><td>71.08</td></tr><tr><td colspan="5">Ours</td></tr><tr><td>Split-1</td><td>24.22</td><td>24.22</td><td>24.23</td><td>75.78</td></tr><tr><td>Split-2</td><td>20.50</td><td>20.51</td><td>20.50</td><td>79.49</td></tr><tr><td>Split-3</td><td>25.18</td><td>25.17</td><td>25.19</td><td>74.83</td></tr><tr><td>Split-4</td><td>24.35</td><td>24.35</td><td>24.36</td><td>75.65</td></tr><tr><td>Split-5</td><td>23.14</td><td>23.14</td><td>23.15</td><td>76.86</td></tr><tr><td>Mean (±std)</td><td>23.48 (±1.81)</td><td>23.48 (±1.81)</td><td>23.49 (±1.82)</td><td>76.52 (±1.81)</td></tr></table>

TABLE XV: Per-split performance on CEDAR [23].
<table><tr><td>Model</td><td>EER↓</td><td>FAR↓</td><td>FRR↓</td><td>Acc↑</td></tr><tr><td>Existing SOTA Model</td><td></td><td></td><td></td><td></td></tr><tr><td>IDN [6]</td><td>3.62</td><td>5.87</td><td>2.17</td><td>-</td></tr><tr><td>SDINet [67]</td><td>1.75</td><td>0.73</td><td>3.42</td><td>-</td></tr><tr><td>Ours</td><td></td><td></td><td></td><td></td></tr><tr><td>Split-1</td><td>1.59</td><td>1.60</td><td>1.59</td><td>98.40</td></tr><tr><td>Split-2</td><td>3.26</td><td>3.26</td><td>3.26</td><td>96.74</td></tr><tr><td>Split-3</td><td>2.46</td><td>2.47</td><td>2.46</td><td>97.54</td></tr><tr><td>Split-4</td><td>2.25</td><td>2.26</td><td>2.25</td><td>97.75</td></tr><tr><td>Split-5</td><td>4.62</td><td>4.62</td><td>4.64</td><td>95.38</td></tr><tr><td>Split-6</td><td>5.02</td><td>5.03</td><td>5.00</td><td>94.98</td></tr><tr><td>Split-7</td><td>3.38</td><td>3.37</td><td>3.41</td><td>96.62</td></tr><tr><td>Split-8</td><td>8.06</td><td>8.06</td><td>8.04</td><td>91.95</td></tr><tr><td>Split-9</td><td>16.50</td><td>16.49</td><td>16.52</td><td>83.50</td></tr><tr><td>Split-10</td><td>10.42</td><td>10.42</td><td>10.43</td><td>89.58</td></tr><tr><td>Split-11</td><td>18.55</td><td>18.54</td><td>18.55</td><td>81.46</td></tr><tr><td>Mean (±std)</td><td>6.92 (±5.88)</td><td>6.92 (±5.87)</td><td>6.92 (±5.88)</td><td>93.08 (±5.88)</td></tr></table>

## APPENDIX E

## EXTENDED ABLATION STUDY: EFFECT OF PRETRAINING EPOCHS

We further conduct an ablation study on the number of self-supervised pretraining epochs, with results shown in Table XVIII. As observed, the performance improves as the pretraining schedule becomes longer, despite minor fluctuations. The best results are obtained at 800 epochs. This is particularly evident on BHSig260-H, where EER drops from 3.45% to 2.59% and Acc rises to 97.41%.

These findings align with prior self-supervised learning methods, such as SimCLR [60], MoCo v2 [77], SwAV [78], and MAE [12], which demonstrate that longer pretraining generally improves downstream performance. Accordingly, they often adopt pretraining schedules of several hundred to over 1000 epochs, and inspire our adoption of 800 pretraining epochs. Although extended pretraining schedules are often studied on large-scale datasets such as ImageNet [79], our results indicate that this practice also benefits the proposed RFR task on relatively smaller signature datasets. Therefore, we adopt 800 epochs as the default pretraining schedule.

## APPENDIX F

## EXTENDED ABLATION STUDY: EFFECT OF PATCH SIDE LENGTH

The side length l of the signature square patch used in RFR could essentially affect model performance. Consequently, we conduct ablation studies on the side length, with results presented in Table XIX. From the outcomes, increasing l steadily improves the downstream performance, achieving the optimal results at l = 96. Further increasing l to 128 leads to an obvious drop in performance. The optimal value of l = 96 naturally aligns with our downstream evaluation setup, which uses input dimensions of 96 × 336, ensuring better model perception of signature pixel distributions. Based on these results, we adopt l = 96 for all experiments.

TABLE XVI: Comparison of different multi-scale feature aggregation architectures used for SigDiffuser on BHSig260- B [20] and BHSig260-H [20]. PFAM denotes the pyramidal feature aggregation module of SigDiffuser.
<table><tr><td rowspan="2">Architecture</td><td colspan="4">BHSig260-B</td><td colspan="4">BHSig260-H</td></tr><tr><td>EER↓</td><td>FAR↓</td><td>FRR↓</td><td>Acc ↑</td><td>EER↓</td><td>FAR↓</td><td>FRR↓</td><td>Acc ↑</td></tr><tr><td>Concatenation</td><td>1.77</td><td>1.77</td><td>1.77</td><td>98.23</td><td>3.83</td><td>3.83</td><td>3.83</td><td>96.17</td></tr><tr><td>Cross-Attention</td><td>4.78</td><td>4.78</td><td>4.77</td><td>95.22</td><td>7.11</td><td>7.11</td><td>7.11</td><td>92.89</td></tr><tr><td>Hierarchical Bottleneck</td><td>1.97</td><td>1.98</td><td>1.97</td><td>98.03</td><td>2.27</td><td>2.27</td><td>2.27</td><td>97.73</td></tr><tr><td>PFAM</td><td>1.58</td><td>1.58</td><td>1.57</td><td>98.43</td><td>2.59</td><td>2.59</td><td>2.59</td><td>97.41</td></tr></table>

TABLE XVII: Performance comparison of SigDiffuser pretrained with versus without diffusion modeling on BHSig260- B [20] and BHSig260-H [20].
<table><tr><td rowspan="2">Method</td><td colspan="4">BHSig260-B</td><td colspan="4">BHSig260-H</td></tr><tr><td>EER↓</td><td>FAR↓</td><td>FRR↓</td><td>Acc ↑</td><td>EER↓</td><td>FAR↓</td><td>FRR↓</td><td>Acc ↑</td></tr><tr><td>w/o Diffusion</td><td>6.85</td><td>6.85</td><td>6.86</td><td>93.15</td><td>5.67</td><td>5.67</td><td>5.67</td><td>94.33</td></tr><tr><td>w/ Diffusion</td><td>1.58</td><td>1.58</td><td>1.57</td><td>98.43</td><td>2.59</td><td>2.59</td><td>2.59</td><td>97.41</td></tr></table>

## APPENDIX G

## EFFECTIVENESS ANALYSIS OF RFR’S CRUMPLING STRATEGY: USING FORGERY AS NATURAL CRUMPLING

While RFR employs artificial crumpling, skillfully forged signatures represent natural deformations [81] of genuine signatures. Therefore, we compare using forgeries as corrupted images in RFR against the vanilla crumpling, with results presented in Table XX. As observed, downstream performance decreases when forged signatures are used as crumpling. The reasons could be two-fold. (1) Using forged signatures as corruptions reduces the pretraining data volume by over half, which could decline downstream performance. (2) The natural variations of forged signatures exhibit fewer spatial deformations compared to vanilla corrupted ones, weakening the model’s representation learning ability. This aligns with the finding in Sec. IV-G that a lower corruption degree declines model performance, reinforcing the efficacy of using severe crumpling in RFR.

## APPENDIX H

## FULL T-SNE VISUALIZATION

Complementing Sec. IV-H1 in the main paper, we present the full t-SNE visualization results in Fig. 7. Compared with the baseline model and MAE, SigDiffuser forms significantly more compact and better separated writer-level clusters, with higher Silhouette scores. The visualized and numerical results collectively validate SigDiffuser’s effectiveness in learning discriminative signature representations.

## APPENDIX I

## RECONSTRUCTION EXAMPLE

We present reconstruction examples across datasets in Fig. 8. Comparing the reconstructed signatures $I _ { g e n }$ to the original uncrumpled ones $I _ { o r g } ,$ writing styles and textures are faithfully restored, while the content is sometimes imprecisely replicated as visually plausible but garbled characters. Nonetheless, unlike traditional font/text generation methods [82, 83], the primary objective of RFR is on capturing signature styles rather than precise verbatim replication. Despite these discrepancies, SigDiffuser achieves high-quality signature representations through self-supervised pretraining, as demonstrated across various datasets (Tables II to VI in the main paper). Furthermore, as shown in Fig. 1, SigDiffuser’s reconstructions significantly surpass those of Masked Image

![](images/1b090642cfeba9f63ddd16a4f7d5b2f1a27e3c304df9b688683cd8d1aff3c376.jpg)  
(a) BHSig260-B.  
(b) BHSig260-H.

Fig. 7: Full t-SNE visualizations [80] of downstream HSV embeddings. We use the Silhouette score [75] to quantify the compactness and separation (higher is better). SigDiffuser yields more compact writer-level feature clusters with significantl larger Silhouette scores than the baseline and MAE, validating its effectiveness in learning separable writer-level representations  
![](images/5b189d9223ba98412a16d30292102cce3aff36e98a51407bdf27a50d915b043d.jpg)  
Fig. 8: Reconstruction visualizations of SigDiffuser on different datasets. $I _ { s t y }$ denotes the intact signature patches used as the style reference, which is input to the visual encoder. $I _ { c o r }$ denotes the corrupted signature patch. $I _ { o r g }$ denotes the clean version of $I _ { c o r } \mathrm { . }$ , which is also the reconstruction target of the model. $I _ { g e n }$ denotes the reconstructed signature patches.

TABLE XVIII: Performance comparison under different numbers of pretraining epochs in the RFR task on BHSig260-B [20] and BHSig260-H [20].
<table><tr><td rowspan="2">#Epoch</td><td colspan="4">BHSig260-B</td><td colspan="4">BHSig260-H</td></tr><tr><td>EER↓</td><td>FAR↓</td><td>FRR↓</td><td>Acc ↑</td><td>EER↓</td><td>FAR↓</td><td>FRR↓</td><td>Acc ↑</td></tr><tr><td>300</td><td>2.13</td><td>2.13</td><td>2.14</td><td>97.87</td><td>3.19</td><td>3.19</td><td>3.19</td><td>96.81</td></tr><tr><td>500</td><td>1.90</td><td>1.90</td><td>1.90</td><td>98.10</td><td>3.21</td><td>3.21</td><td>3.21</td><td>96.79</td></tr><tr><td>700</td><td>2.14</td><td>2.13</td><td>2.14</td><td>97.86</td><td>2.63</td><td>2.63</td><td>2.63</td><td>97.37</td></tr><tr><td>800</td><td>1.58</td><td>1.58</td><td>1.57</td><td>98.43</td><td>2.59</td><td>2.59</td><td>2.59</td><td>97.41</td></tr></table>

TABLE XIX: Performance comparison of varying signature patch size (side length) l used in the RFR task on BHSig260- B [20] and BHSig260-H [20].
<table><tr><td rowspan="2">l</td><td colspan="4">BHSig260-B</td><td colspan="4">BHSig260-H</td></tr><tr><td>EER↓</td><td>FAR ↓</td><td>FRR↓</td><td>Acc ↑</td><td>EER↓</td><td>FAR↓</td><td>FRR↓</td><td>Acc ↑</td></tr><tr><td>32</td><td>4.92</td><td>4.92</td><td>4.91</td><td>95.08</td><td>4.44</td><td>4.44</td><td>4.44</td><td>95.56</td></tr><tr><td>48</td><td>3.07</td><td>3.08</td><td>3.07</td><td>96.93</td><td>4.69</td><td>4.70</td><td>4.69</td><td>95.30</td></tr><tr><td>64</td><td>2.38</td><td>2.38</td><td>2.38</td><td>97.62</td><td>3.38</td><td>3.38</td><td>3.38</td><td>96.62</td></tr><tr><td>96</td><td>1.58</td><td>1.58</td><td>1.57</td><td>98.43</td><td>2.59</td><td>2.59</td><td>2.59</td><td>97.41</td></tr><tr><td>128</td><td>1.66</td><td>1.67</td><td>1.66</td><td>98.34</td><td>4.02</td><td>4.02</td><td>4.03</td><td>95.98</td></tr></table>

TABLE XX: Comparison between using forged signatures (natural destruction) and RFR’s severe crumpling as destruction methods on BHSig260-B [20] and BHSig260-H [20].
<table><tr><td rowspan="2">Crumpling Effect</td><td colspan="4">BHSig260-B</td><td colspan="4">BHSig260-H</td></tr><tr><td>EER↓</td><td>FAR↓</td><td>FRR↓</td><td>Acc ↑</td><td>EER↓</td><td>FAR↓</td><td>FRR↓</td><td>Acc ↑</td></tr><tr><td>Forged Signature</td><td>3.04</td><td>3.04</td><td>3.04</td><td>96.96</td><td>4.09</td><td>4.09</td><td>4.09</td><td>95.91</td></tr><tr><td>Vanilla Crumpling</td><td>1.58</td><td>1.58</td><td>1.57</td><td>98.43</td><td>2.59</td><td>2.59</td><td>2.59</td><td>97.41</td></tr></table>

Modeling (MIM), which only restores blurred structures. This superior structural fidelity demonstrates a deep understanding of signature writing patterns, confirming that precise content replication is inconsequential to our method’s overall effectiveness as an HSV pretraining approach.

![](images/76dd5511d191bd9b5ae9ad1fe51273dbde58f5fd63346450014e05d07f1072ff.jpg)

![](images/8bdf8082bd31732be632f4ec46b1837ebd16e63170c7704cb258461550d0c559.jpg)  
Fig. 9: Visualization of false rejection and false acceptance examples of SigDiffuser.

## APPENDIX J FAILURE CASE ANALYSIS

We present some incorrectly verified samples of SigDiffuser across different datasets in Fig. 9, including false rejection and false acceptance. As observed, the false rejections typically exhibit significant deviations from their reference templates. For example, in CEDAR, the falsely rejected samples show larger inclination angles and contrasting size proportions (shorter overall length) compared to the template. In HanSig, the template signature displays a free-flowing writing style with obvious connected strokes, whereas the false rejection lacks stroke connections and exhibits a rigid writing style. The huge differences lead to their misclassification as forgeries, despite originating from the same writers of templates.

Regarding false acceptances, forged signatures often exhibit local writing patterns that resemble their corresponding template. For instance, the central portion of Bengali forgeries, the stroke inclination in Hindi forgeries, and the free-flowing writing style in Chinese forgeries all demonstrate similarities to their respective templates. The model may not always identify these deceptive details, leading to erroneous classification. These failed cases reveal that self-supervised learning remains challenged by some high-quality forged samples and struggles with genuine samples that exhibit significant stylistic variations from templates, despite improved HSV performance. Notably, Tables III to VII show that SigDiffuser maintains closely balanced FAR and FRR, suggesting the need for comprehensive optimization across all failure modes rather than targeting specific error types.

## REFERENCES

[1] M. Diaz, M. A. Ferrer, D. Impedovo et al., “A Perspective Analysis of Handwritten Signature Technology,” ACM Computing Survey, vol. 51, no. 6, Jan 2019.

[2] P. Zhang, J. Jiang, Y. Liu, and L. Jin, “MSDS: A Large-Scale Chinese Signature and Token Digit String Dataset for Handwriting Verification,” in NeurIPS, vol. 35, 2022, pp. 36 507–36 519.

[3] P. Zhang, K. Ding, and L. Jin, “Capturing More: Learning Multi-Domain Representations for Robust Online Handwriting Verification,” in ACM MM, 2025, p. 1471–1479.

[4] R. Sabourin, G. Genest, and F. Preteux, “Off-Line Signature Verification by Local Granulometric Size Distributions,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 19, no. 9, pp. 976–988, 1997.

[5] H. Li, P. Wei, Z. Ma et al., “TransOSV: Offline Signature Verification with Transformers,” Pattern Recognition, vol. 145, p. 109882, 2024.

[6] P. Wei, H. Li, and P. Hu, “Inverse Discriminative Networks for Handwritten Signature Verification,” in CVPR, June 2019, pp. 5757–5765.

[7] C. Fan, S. Hou, J. Wang et al., “Learning Gait Representation From Massive Unlabelled Walking Videos: A Benchmark,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 12, pp. 14 920–14 937, 2023.

[8] S. Wang, S. Chen, D.-H. Wang et al., “CAMD: Context-Aware Masked Distillation for General Self-Supervised Facial Representation Pre-Training,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 36, no. 4, pp. 4663–4677, 2026.

[9] S. Chattopadhyay, S. Manna, S. Bhattacharya, and U. Pal, “SURDS: Self-Supervised Attention-guided Reconstruc-

tion and Dual Triplet Loss for Writer Independent Offline Signature Verification,” in ICPR, 2022, pp. 1600–1606.

[10] S. Manna, S. Chattopadhyay, S. Bhattacharya, and U. Pal, “SWIS: Self-Supervised Representation Learning for Writer Independent Offline Signature Verification,” in ICIP, 2022, pp. 1411–1415.

[11] X. Liu, F. Zhang, Z. Hou et al., “Self-Supervised Learning: Generative or Contrastive,” IEEE Transactions on Knowledge and Data Engineering, vol. 35, no. 1, pp. 857–876, 2023.

[12] K. He, X. Chen, S. Xie et al., “Masked Autoencoders are Scalable Vision Learners,” in CVPR, June 2022, pp. 16 000–16 009.

[13] Z. Xie, Z. Zhang, Y. Cao et al., “SimMIM: A Simple Framework for Masked Image Modeling,” in CVPR, June 2022, pp. 9653–9663.

[14] K. Preechakul, N. Chatthee, S. Wizadwongsa, and S. Suwajanakorn, “Diffusion Autoencoders: Toward a Meaningful and Decodable Representation,” in CVPR, June 2022, pp. 10 619–10 629.

[15] W. Xiang, H. Yang, D. Huang, and Y. Wang, “Denoising Diffusion Autoencoders are Unified Self-supervised Learners,” in ICCV, October 2023, pp. 15 802–15 812.

[16] D. A. Hudson, D. Zoran, M. Malinowski et al., “SODA: Bottleneck Diffusion Models for Representation Learning,” in CVPR, June 2024, pp. 23 115–23 127.

[17] X. Chen, Z. Liu, S. Xie, and K. He, “Deconstructing Denoising Diffusion Models for Self-Supervised Learning,” in ICLR, 2025.

[18] Z. Zhang, Z. Zhao, and Z. Lin, “Unsupervised Representation Learning from Pre-trained Diffusion Probabilistic Models,” in NeurIPS, vol. 35, 2022, pp. 22 117–22 130.

[19] C. Luo, L. Jin, and J. Chen, “SimAN: Exploring Self-Supervised Representation Learning of Scene Text via Similarity-Aware Normalization,” in CVPR, June 2022, pp. 1039–1048.

[20] S. Pal, A. Alaei, U. Pal, and M. Blumenstein, “Performance of an Off-Line Signature Verification Method Based on Texture Features on a Large Indic-Script Signature Dataset,” in DAS, 2016, pp. 72–77.

[21] F.-H. Huang and H.-M. Lu, “Multiscale Global and Regional Feature Learning using Co-Tuplet Loss for Offline Handwritten Signature Verification,” arXiv preprint arXiv:2308.00428, 2023.

[22] A. Soleimani, K. Fouladi, and B. N. Araabi, “UTSig: A Persian Offline Signature Dataset,” IET Biometrics, vol. 6, no. 1, pp. 1–8, 2017.

[23] M. K. KALERA, S. SRIHARI, and A. XU, “Offline Signature Verification and Identification using Distance Statistics,” International Journal of Pattern Recognition and Artificial Intelligence, vol. 18, no. 07, pp. 1339– 1360, 2004.

[24] A. Radford, L. Metz, and S. Chintala, “Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks,” arXiv preprint arXiv:1511.06434, 2015.

[25] D. Pathak, P. Krahenbuhl, J. Donahue et al., “Context Encoders: Feature Learning by Inpainting,” in CVPR,

June 2016, pp. 2536–2544.

[26] A. Dosovitskiy, L. Beyer, A. Kolesnikov et al., “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale,” in ICLR, 2021.

[27] K. Tian, Y. Jiang, Q. Diao et al., “Designing BERT for Convolutional Networks: Sparse and Hierarchical Masked Modeling,” in ICLR, 2023.

[28] M. Chen, A. Radford, R. Child, J. Wu, H. Jun, D. Luan, and I. Sutskever, “Generative Pretraining From Pixels,” in ICML, vol. 119, 2020, pp. 1691–1703.

[29] M. Fuest, P. Ma, M. Gui et al., “Diffusion Models and Representation Learning: A Survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 48, no. 7, pp. 7209–7228, 2026.

[30] Y. Kim, K. Lee, M. Park et al., “Diffusion Bridge AutoEncoders for Unsupervised Representation Learning,” in ICLR, 2025, pp. 84 846–84 887.

[31] S. Chen and S. Srihari, “Use of Exterior Contours and Shape Features in Off-Line Signature Verification,” in ICDAR, 2005, pp. 1280–1284.

[32] D. Bertolini, L. Oliveira, E. Justino, and R. Sabourin, “Reducing Forgeries in Writer-Independent Off-Line Signature Verification through Ensemble of Classifiers,” Pattern Recognition, vol. 43, no. 1, pp. 387–396, 2010.

[33] A. Hamadene and Y. Chibani, “One-Class Writer-Independent Offline Signature Verification using Feature Dissimilarity Thresholding,” IEEE Transactions on Information Forensics and Security, vol. 11, no. 6, pp. 1226– 1238, 2016.

[34] J. Vargas, M. Ferrer, C. Travieso, and J. Alonso, “Off-Line Signature Verification Based on Grey Level Information using Texture Features,” Pattern Recognition, vol. 44, no. 2, pp. 375–385, 2011.

[35] M. A. Ferrer, J. F. Vargas, A. Morales, and A. Ordonez, “Robustness of Offline Signature Verification based on Gray Level Features,” IEEE Transactions on Information Forensics and Security, vol. 7, no. 3, pp. 966–977, 2012.

[36] J. Hu and Y. Chen, “Offline Signature Verification using Real Adaboost Classifier Combination of Pseudo-Dynamic Features,” in ICDAR, 2013, pp. 1345–1349.

[37] R. Kumar, J. Sharma, and B. Chanda, “Writer-Independent Off-Line Signature Verification using Surroundedness Feature,” Pattern Recognition Letters, vol. 33, no. 3, pp. 301–308, 2012.

[38] M. Okawa, “Synergy of Foreground–Background Images for Feature Extraction: Offline Signature Verification using Fisher Vector with Fused KAZE Features,” Pattern Recognition, vol. 79, pp. 480–489, 2018.

[39] E. N. Zois, D. Tsourounis, and D. Kalivas, “Similarity Distance Learning on SPD Manifold for Writer Independent Offline Signature Verification,” IEEE Transactions on Information Forensics and Security, vol. 19, pp. 1342– 1356, 2024.

[40] L. G. Hafemann, R. Sabourin, and L. S. Oliveira, “Learning Features for Offline Handwritten Signature Verification using Deep Convolutional Neural Networks,” Pattern Recognition, vol. 70, pp. 163–176, 2017.

[41] S. Dey, A. Dutta, J. I. Toledo et al., “SigNet:

Convolutional Siamese Network for Writer Independent Offline Signature Verification,” arXiv preprint arXiv:1707.02131, 2017.

[42] C. Li, F. Lin, Z. Wang et al., “DeepHSV: User-Independent Offline Signature Verification using Two-Channel CNN,” in ICDAR, 2019, pp. 166–171.

[43] L. Liu, L. Huang, F. Yin et al., “Offline Signature Verification using a Region Based Deep Metric Learning Network,” Pattern Recognition, vol. 118, p. 108009, 2021.

[44] H. Li, P. Wei, and P. Hu, “AVN: An Adversarial Variation Network Model for Handwritten Signature Verification,” IEEE Transactions on Multimedia, vol. 24, pp. 594–608, 2022.

[45] L.-R. Ling, H. Zhang, Y. Fei, and C.-L. Liu, “Deep Metric Learning with Cross-Writer Attention for Offline Signature Verification,” in ICDAR, 2024, pp. 250–267.

[46] M.-C. Shih, T.-L. Huang, Y.-H. Shih et al., “DetailSem-Net: Elevating Signature Verification Through Detail-Semantic Integration,” in ECCV, 2024, pp. 449–466.

[47] L. Zheng, D. Wu, S. Xu, and Y. Zheng, “HTCSigNet: A Hybrid Transformer and Convolution Signature Network for Offline Signature Verification,” Pattern Recognition, vol. 159, p. 111146, 2025.

[48] J.-X. Ren, Y.-J. Xiong, H. Zhan et al., “2C2S: A Two-Channel and Two-Stream Transformer Based Framework for Offline Signature Verification,” Engineering Applications of Artificial Intelligence, vol. 118, p. 105639, 2023.

[49] C.-Q. Lin, D.-H. Wang, Y.-F. Su et al., “OCR4HSV: A Multi-task Learning Approach for Handwritten Signature Verification,” in ICPR, 2025, pp. 287–302.

[50] X. Zhao, L. Zheng, P. Yuan et al., “Fusing Deep and Hand-Crafted Features by Deep Canonically Correlated Contractive Autoencoder for Offline Signature Verification,” Pattern Recognition, vol. 168, p. 111834, 2025.

[51] S. Lai and L. Jin, “Learning Discriminative Feature Hierarchies for Off-Line Signature Verification,” in ICFHR, 2018, pp. 175–180.

[52] S. Gidaris, P. Singh, and N. Komodakis, “Unsupervised Representation Learning by Predicting Image Rotations,” in ICLR, 2018.

[53] K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” in CVPR, June 2016.

[54] T.-Y. Lin, P. Dollar, R. Girshick, K. He, B. Hariharan, and S. Belongie, “Feature Pyramid Networks for Object Detection,” in CVPR, July 2017.

[55] J. Song, C. Meng, and S. Ermon, “Denoising Diffusion Implicit Models,” in ICLR, 2021.

[56] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” in MICCAI, 2015, pp. 234–241.

[57] K. He, X. Zhang, S. Ren et al., “Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 37, no. 9, pp. 1904–1916, 2015.

[58] K. Simonyan and A. Zisserman, “Very Deep Convolutional Networks for Large-Scale Image Recognition,” in

ICLR, 2015.

[59] Z. Liu, H. Mao, C.-Y. Wu et al., “A ConvNet for the 2020s,” in CVPR, June 2022, pp. 11 976–11 986.

[60] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A Simple Framework for Contrastive Learning of Visual Representations,” in ICML, vol. 119, 13–18 Jul 2020, pp. 1597–1607.

[61] J.-B. Grill, F. Strub, F. Altche´ et al., “Bootstrap Your Own Latent - A New Approach to Self-Supervised Learning,” in NeurIPS, vol. 33, 2020, pp. 21 271–21 284.

[62] A. Aberdam, R. Litman, S. Tsiper et al., “Sequence-to-Sequence Contrastive Learning for Text Recognition,” in CVPR, June 2021, pp. 15 302–15 312.

[63] X. Zhang, T. Wang, J. Wang et al., “ChaCo: Character Contrastive Learning for Handwritten Text Recognition,” in ICFHR, 2022, pp. 345–359.

[64] X. Zhang, J. Wang, L. Jin et al., “CMT-Co: Contrastive Learning with Character Movement Task for Handwritten Text Recognition,” in ACCV, 2022, pp. 3104–3120.

[65] X. Zhang, R. Jiang, W. Gao et al., “Residual Connections Harm Generative Representation Learning,” in CVPR, June 2026, pp. 39 669–39 679.

[66] S. Lai, L. Jin, Y. Zhu et al., “SynSig2Vec: Forgery-Free Learning of Dynamic Signature Representations by Sigma Lognormal-Based Synthesis and 1D CNN,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 10, pp. 6472–6485, 2022.

[67] H. Li, P. Wei, and P. Hu, “Static-Dynamic Interaction Networks for Offline Signature Verification,” AAAI, vol. 35, pp. 1893–1901, May 2021.

[68] X. Lu, L. Huang, and F. Yin, “Cut and Compare: Endto-end Offline Signature Verification Network,” in ICPR, 2021, pp. 3589–3596.

[69] X. ji, D. Suehiro, and S. Uchida, “Paired Contrastive Feature for Highly Reliable Offline Signature Verification,” Pattern Recognition, vol. 144, p. 109816, 2023.

[70] X. Cairang, D. Zhaxi, X. Yang et al., “Learning Generalisable Representations for Offline Signature Verification,” in IJCNN, 2022, pp. 1–7.

[71] M. Ajij, S. Pratihar, S. R. Nayak et al., “Off-Line Signature Verification using Elementary Combinations of Directional Codes from Boundary Pixels,” Neural Computing and Applications, pp. 1–18, 2023.

[72] W. Peebles and S. Xie, “Scalable Diffusion Models with Transformers,” in ICCV, October 2023, pp. 4195–4205.

[73] D. Engin, A. Kantarci, S. Arslan, and H. K. Ekenel, “Offline Signature Verification on Real-World Documents,” in CVPRW, June 2020.

[74] R. R. Selvaraju, M. Cogswell, A. Das et al., “Grad-CAM: Visual Explanations From Deep Networks via Gradient-Based Localization,” in ICCV, Oct 2017.

[75] P. J. Rousseeuw, “Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis,” Journal of Computational and Applied Mathematics, vol. 20, pp. 53–65, 1987.

[76] G. Ding, X. Han, S. Wang et al., “Stable Attribute Group Editing for Reliable Few-Shot Image Generation,” IEEE Transactions on Circuits and Systems for Video

Technology, vol. 35, no. 12, pp. 12 719–12 733, 2025.

[77] X. Chen, H. Fan, R. Girshick, and K. He, “Improved Baselines with Momentum Contrastive Learning,” arXiv preprint arXiv:2003.04297, 2020.

[78] M. Caron, I. Misra, J. Mairal, P. Goyal, P. Bojanowski, and A. Joulin, “Unsupervised Learning of Visual Features by Contrasting Cluster Assignments,” in NeurIPS, vol. 33, 2020, pp. 9912–9924.

[79] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “ImageNet: A Large-Scale Hierarchical Image Database,” in CVPR, 2009, pp. 248–255.

[80] L. Van der Maaten and G. Hinton, “Visualizing Data using t-SNE,” JMLR, vol. 9, no. 11, 2008.

[81] P. Zhang, Y. Liu, S. Lai et al., “Privacy-Preserving Biometric Verification with Handwritten Random Digit String,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 4, pp. 3049–3066, 2025.

[82] Z. Yang, D. Peng, Y. Kong et al., “FontDiffuser: One-Shot Font Generation via Denoising Diffusion with Multi-Scale Content Aggregation and Style Contrastive Learning,” AAAI, vol. 38, no. 7, pp. 6603–6611, 2024.

[83] C. Luo, Y. Zhu, L. Jin, Z. Li, and D. Peng, “SLOGAN: Handwriting Style Synthesis for Arbitrary-Length and Out-of-Vocabulary Text,” IEEE Transactions on Neural Networks and Learning Systems, vol. 34, no. 11, pp. 8503–8515, 2023.