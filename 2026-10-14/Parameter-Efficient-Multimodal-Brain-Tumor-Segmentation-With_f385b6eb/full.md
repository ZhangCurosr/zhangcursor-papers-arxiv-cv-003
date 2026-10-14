# Parameter-Efficient Multimodal Brain Tumor Segmentation With Stage-Wise Shared Adapters

MILAD ROSTAMIAN<sup>1</sup>, MOHAMMADREZA MOHAMMADI<sup>1</sup>, ENDER KONUKOGLU<sup>2</sup>, and MOHSEN SORYANI<sup>1</sup>

<sup>1</sup>School of Computer Engineering, Iran University of Science and Technology (IUST), Tehran, Iran (e-mail: milad\_rostamian@comp.iust.ac.ir; mrmohammadi@iust.ac.ir; soryani@iust.ac.ir)

<sup>2</sup>Computer Vision Laboratory, ETH Zurich, 8092 Zurich, Switzerland (e-mail: ender.konukoglu@vision.ee.ethz.ch)

Corresponding author: Mohsen Soryani (e-mail: soryani@iust.ac.ir).

ABSTRACT Different MRI modalities reveal different parts of a brain tumor. Segmentation models combine information from these modalities to outline tumor regions. Many models use a separate encoder for each modality. However, exchanging features among these encoders at several stages can increase model complexity. We introduce the Block-Wise Shared Adapter (BWSA), which combines compressed features from the modality encoders in a shared path and returns adapted features to each stream. In its fixed-blending form, BWSA adds the adapted features to the original features. Its trainable-gated variant (TG-BWSA) instead learns separate weights for the original and adapted features at each modality and encoder stage. We evaluate both variants on BraTS 2020 and BraTS 2023 with four segmentation backbones. On the official BraTS 2023 online validation set, BWSA increases average region-wise Dice by 1.3–7.6 percentage points relative to the corresponding unmodified backbones. TG-BWSA increases it by 2.8–7.7 points. Both variants add approximately 0.85–3.73% parameters and reduce average HD95 relative to the unmodified backbones.

INDEX TERMS brain tumor segmentation, multimodal MRI, adapter, feature fusion, parameter-efficient learning

## . INTRODUCTION

Brain tumor segmentation from magnetic resonance imaging (MRI) supports diagnosis, treatment planning, and lon-<sup>.</sup> gitudinal assessment [1]. Multimodal protocols commonly include contrast-enhanced T1-weighted (T1ce), T1-weighted<sub>6</sub> (T1), fluid-attenuated inversion recovery (FLAIR), and T2-2 weighted (T2) images, which provide complementary infor-<sup>:</sup> mation about tumor and surrounding tissue [2].i

<sub>Fusion</sub> <sub>strategies</sub> <sub>are</sub> <sub>commonly</sub> <sub>classified</sub> <sub>by</sub> <sub>where</sub> <sub>they</sub>X combine the modalities [3]. Early fusion concatenates modal-<sup>r</sup> ities at the input, whereas late fusion merges independently produced outputs. Early fusion allows interaction throughout the network but does not preserve separate modality streams. Late fusion preserves those streams but postpones their interaction until the output. Intermediate fusion combines encoder features at one or more depths. Methods that use attention or gates can model cross-modal dependencies at these depths, but modality-specific branches and dense attention can increase parameter and computational costs [4]–[6].

Adapters have also been used in intermediate fusion. Shi et al. place fusion adapters at multiple layers of two pretrained Segment Anything Model (SAM) encoders to combine two MRI modality groups for glioma segmentation [7]. Lu et al. use a dual-stream adapter to align modality features before fusing them at multiple scales [8]. We test whether a shared adapter can exchange features among four MRI encoders with a small increase in parameters. We also compare fixed and learned weights for combining the original and adapted features.

We introduce the Block-Wise Shared Adapter (BWSA) for multimodal fusion between encoder stages (Fig. 1c). BWSA combines the adapted and original features with fixed unit weights. Its gated extension, the Trainable Gated BWSA (TG-BWSA), learns two nonnegative scalars (α, β) for each modality and encoder stage. One weights the adapted feature and the other weights the original feature. These weights stay the same for every test image. Both forms use a single shared path rather than a full adapter for every modality. Separate expansion layers then return the shared features to the corresponding encoder streams.

We evaluate BWSA with fixed and learned stream weights on BraTS 2020 and BraTS 2023 using four CNN, Transformer, and hybrid backbones. We report local validation results and results from the BraTS 2023 online validation server. On online validation, fixed-weight BWSA raises average Dice by 1.3–7.6 percentage points relative to the unmodified backbones. The gated extension raises it by 2.8–7.7 percentage points. Both forms have nearly the same number of parameters because the gates add only two scalar weights per modality and stage. The reported overhead of 0.85–3.73% compares the gated adapter with each unmodified backbone. We also evaluate RFNet [9] with both forms of BWSA under 14 incomplete input configurations and report the complete input result separately.

![](images/f6a25a90a970df36f0e93b976cb77b6d7f1be2694e06944b1e5b5bbc306eb362.jpg)  
FIGURE 1: Representative fusion placements. (a) Early fusion concatenates the modalities at the input. (b) Late fusion combines independently produced predictions. (c) BWSA inserts a shared adaptation path between encoder stages. Its gated extension learns scalar weights for the original and adapted streams of each modality at each stage.

Contributions. The main contributions of this work are:

• We introduce BWSA, a stage-wise shared adapter that can be inserted between encoder blocks in the four evaluated backbones.

• We extend BWSA to TG-BWSA by learning two scalar weights per modality and stage for the adapted multimodal stream and the original stream. The shared adapter path remains unchanged.

• We evaluate the shared adapter with fixed and learned weights. We also compare it with separate adapters for each modality.

• Missing modality experiments cover all 14 incomplete input configurations and report the complete input case separately.

## II. RELATED WORK

## A. MULTIMODAL FUSION METHODS

Some models combine MRI sequences before feature extraction. Swin UNETR takes the four sequences as channels of one input and learns a joint representation [10]. This gives the encoder access to all sequences from its first layer, but does not keep a separate stream for each one. To retain modalityspecific features, Nie et al. encode each modality separately and merge their high-level features before segmentation [11]. Fang et al. also use separate modality inputs and combine their features through hybrid attention, together with selfsupervised feature learning [12].

With separate streams, fusion can use learned interactions between their features. NestedFormer uses attention to combine high-level modality features and gating at lower scales [4]. mmFormer uses an inter-modal Transformer to relate features from modality-specific encoders [5]. F2Net maintains both modality-specific and shared representations, with modules that exchange information between them [6].

## B. MODALITY WEIGHTING AND MISSING INPUTS

Keeping the modalities separate also allows a model to vary their contributions. RFNet learns which modality features matter in different tumor regions [9]. A2FSeg averages the available features and then uses attention to adjust their fusion [13]. These methods derive their weights from the input features. TG-BWSA instead learns one pair of stream weights for each modality and stage; the weights stay fixed across test inputs.

When a sequence is missing, the set of features available for fusion changes. HeMIS handles different subsets by averaging the embeddings of the available modalities [14]. Chen et al. combine feature disentanglement and gated fusion to learn from incomplete inputs [15]. Qiu et al. show that adding a modality can harm predictions in some regions and calibrate the fusion result accordingly [16].

## C. ADAPTERS FOR MULTIMODAL FUSION

Adapters add a small trainable path to a model. Med-SA uses them to adapt pretrained SAM to medical image segmentation [17]. Adapters can also take part in fusion. Shi et al. use fusion adapters between two pretrained SAM encoders to combine MRI modality groups at multiple layers [7]. Lu et al. use a dual-stream adapter to align modality features before multi-scale fusion [8]. BWSA connects four separate modality streams through one shared adapter path at each encoder stage.

## III. BLOCK-WISE SHARED ADAPTER FRAMEWORK

BWSA uses a shared path between encoder blocks to process compressed features from all modalities. It adds the adapted feature to the original feature of each modality. Its gated extension, TG-BWSA, learns separate nonnegative scalar weights for the original and adapted features at each modality and stage.

## A. ARCHITECTURAL FRAMEWORK

We use a multi-encoder, single-decoder design (Fig. 2). Let M be the number of modalities. For BraTS, M=4 and the modalities are T1ce, T1, FLAIR, and T2. Each modality m is processed by an encoder $E _ { m } ( \cdot )$ producing features at L stages. At stage l, the modality-specific 3D feature maps are

$$
\{ X _ { m } ^ { ( l ) } \} _ { m = 1 } ^ { M } , \qquad X _ { m } ^ { ( l ) } \in \mathbb { R } ^ { C \times D \times H \times W } ,\tag{1}
$$

where C denotes the channel dimension and $( D , H , W )$ are depth, height, and width, respectively. For the 2D backbones, the depth dimension is omitted and the convolutions use the corresponding 2D kernels. An adapter transforms $\{ X _ { m } ^ { ( l ) } \} _ { m = 1 } ^ { M }$ into $\{ \bar { Y } _ { m } ^ { ( l ) } \} _ { m = 1 } ^ { M ^ { - } }$ , which are forwarded to stage (l+1). At each stage, the updated modality features are concatenated to form the corresponding skip input to the shared decoder. The adapter changes the connections between encoder blocks but leaves the internal operations of those blocks unchanged.

![](images/ad7f23273f882a80de8530cd3a877cf3b9a9893ace966d5a3c449e15de13713d.jpg)  
FIGURE 2: Overall multimodal framework. Each modality is processed by a separate encoder branch. The adapters connect successive encoder blocks, and their outputs are passed to a shared decoder through skip connections.

## B. BWSA MODULE AND GATED EXTENSION

Both variants use (i) channel compression and multimodal aggregation, (ii) shared cross-modal channel recalibration, (iii) spatial refinement, and (iv) residual blending. TG-BWSA adds learned weights in the final step. Figure 3a illustrates the TG-BWSA overview, and Figure 3b provides the CMCR details.

## 1) Compression and Multimodal Aggregation

Each modality feature map is first compressed to $C ^ { \prime } = \lfloor r C \rfloor$ channels, with $r = 0 . 5$ , using a per-modality $1 \times 1 \times 1$ convolution:

$$
\begin{array} { r } { X _ { m } ^ { \prime ( l ) } = \mathrm { R e L U } \Big ( \mathrm { B N } \Big ( \mathrm { C o n v } _ { 1 \times 1 \times 1 } ^ { ( m ) } ( X _ { m } ^ { ( l ) } ) \Big ) \Big ) \in \mathbb { R } ^ { C ^ { \prime } \times D \times H \times W } . } \end{array}\tag{2}
$$

The compressed features are concatenated across modalities:

$$
X _ { \mathrm { c a t } } ^ { \prime ( l ) } = \left[ X _ { 1 } ^ { \prime ( l ) } , \dots , X _ { M } ^ { \prime ( l ) } \right] \in \mathbb { R } ^ { M C ^ { \prime } \times D \times H \times W } .\tag{3}
$$

## 2) Cross-Modal Channel Recalibration (CMCR)

To model cross-modal dependencies in the shared latent space, we apply a channel-attention (SE-style) recalibration [18] to $X _ { \mathrm { c a t } } ^ { \prime ( l ) }$ . The excitation path includes dropout with $\scriptstyle p = 0 . 1 $

$$
\begin{array} { r l } & { \quad \quad \quad \boldsymbol { a } ^ { ( l ) } = \sigma \Bigl ( \mathbf { W } _ { \mathrm { 2 } } \operatorname { D r o p o u t } \Bigl ( \operatorname { R e L U } \Bigl ( \mathbf { W } _ { \mathrm { 1 } } \mathrm { G A P } ( X _ { \mathrm { c a t } } ^ { \prime ( l ) } ) \Bigr ) \Bigr ) \Bigr ) , } \\ & { \quad \quad X ^ { \prime \prime ( l ) } = X _ { \mathrm { c a t } } ^ { \prime ( l ) } \odot \boldsymbol { a } ^ { ( l ) } . } \end{array}\tag{4}
$$

where $a ^ { ( l ) }$ contains the channel weights, $\mathrm { G A P } ( \cdot )$ is global average pooling, $\mathbf { W } _ { 1 } , \mathbf { W } _ { 2 }$ are fully-connected layers, $\sigma ( \cdot )$ is sigmoid, and $\odot$ denotes channel-wise multiplication broadcast over $D \times H \times W$

## 3) Spatial Refinement and Expansion

We apply a depthwise separable convolution to refine $X ^ { \prime \prime ( l ) }$ spatially:

$$
\begin{array} { r } { X ^ { \prime \prime \prime ( l ) } = \mathrm { R e L U } \Big ( \mathrm { B N } \Big ( \mathrm { C o n v } _ { 1 \times 1 \times 1 } \big ( \mathrm { D W C } _ { 3 \times 3 \times 3 } \big ( X ^ { \prime \prime ( l ) } \big ) \big ) \Big ) . } \end{array}\tag{5}
$$

Then, for each modality we project back to the original channel dimension C to obtain an adapted feature map:

$$
Z _ { m } ^ { ( l ) } = \operatorname { R e L U } \Bigl ( \operatorname { B N } \Bigl ( \operatorname { C o n v } _ { 1 \times 1 \times 1 } ^ { ( m ) } \bigl ( X ^ { \prime \prime \prime ( l ) } \bigr ) \Bigr ) \Bigr ) \in \mathbb { R } ^ { C \times D \times H \times W } .\tag{6}
$$

The recalibration and refinement path up to $X ^ { \prime \prime \prime ( l ) }$ is shared across modalities. The final expansion to $Z _ { m } ^ { ( l ) }$ is performed separately for each modality.

## 4) Residual Blending and Trainable Gates

BWSA adds the original and adapted features with unit weights:

$$
\begin{array} { r } { Y _ { m , \mathrm { B W S A } } ^ { ( l ) } = X _ { m } ^ { ( l ) } + Z _ { m } ^ { ( l ) } . } \end{array}\tag{7}
$$

TG-BWSA instead learns two scalar gates per modality and stage:

$$
\begin{array} { r } { Y _ { m } ^ { ( l ) } = \underbrace { \left( \beta _ { m } ^ { ( l ) } + \epsilon \right) X _ { m } ^ { ( l ) } } _ { \mathrm { P r e s e r v a t i o n } } + \underbrace { \alpha _ { m } ^ { ( l ) } Z _ { m } ^ { ( l ) } } _ { \mathrm { A d a p t a t i o n } } , \qquad \epsilon = 1 0 ^ { - 3 } . } \end{array}\tag{8}
$$

To enforce non-negativity, gates are parameterized by unconstrained variables and rectified via Softplus:

$$
\alpha _ { m } ^ { ( l ) } = \mathrm { S o f t p l u s } ( \hat { \alpha } _ { m } ^ { ( l ) } ) , \qquad \beta _ { m } ^ { ( l ) } = \mathrm { S o f t p l u s } ( \hat { \beta } _ { m } ^ { ( l ) } ) .\tag{9}
$$

TG-BWSA adds 2ML trainable scalar weights beyond BWSA, where M is the number of modalities and L is the number of adapter stages. With four modalities and four stages, this is $2 \times 4 \times 4 = 3 2$ weights. Both forms retain the same compression, channel-recalibration, spatial-refinement, and expansion layers. We initialize $\hat { \alpha } _ { m } ^ { ( l ) }$ and $\hat { \beta } _ { m } ^ { ( l ) }$ such that $\alpha _ { m } ^ { ( l ) }$ ≈ 0.5 and $\beta _ { m } ^ { ( \bar { l } ) }$ ≈ 1.0 after Softplus, giving greater initial weight to the original feature. We add an $\ell _ { 2 }$ penalty to the gates at stages 2 through L:

$$
\mathcal { L } _ { \mathrm { g a t e } } = \lambda \sum _ { l = 2 } ^ { L } \sum _ { m = 1 } ^ { M } \left( ( \alpha _ { m } ^ { ( l ) } ) ^ { 2 } + ( \beta _ { m } ^ { ( l ) } ) ^ { 2 } \right) , \qquad \lambda = 1 0 ^ { - 4 } .\tag{10}
$$

![](images/1bf116f9a77ae1e7d729abe04a71e44d143faae09b7ede6d620e57ad038c2add.jpg)

(a) TG-BWSA overview.  
![](images/f944729b1703fd94ec329d4665a7892368aa4ec7d003583acc854db8965cd953.jpg)  
(b) CMCR details.  
FIGURE 3: TG-BWSA. (a) Compression → shared recalibration/refinement → gated redistribution. (b) CMCR operating on the concatenated multimodal latent tensor.

## TABLE 1: TG-BWSA forward pass at encoder stage l.

1: Input: modality features $\{ X _ { m } ^ { ( l ) } \} _ { m = 1 } ^ { M }$   
2: Parameters: compression ratio $\ddot { r } { = } \bar { 0 } . 5$ and $\scriptstyle \epsilon = 1 0 ^ { - 3 }$   
3: for $m \not = 1$ to M do   
4: $X _ { m } ^ { \prime ( l ) } \gets \mathrm { R e L U } ( \mathrm { B N } ( \mathrm { C o n v } _ { 1 \times 1 \times 1 } ^ { ( m ) } ( X _ { m } ^ { ( l ) } ) ) )$   
5: end for   
6: $X _ { \mathrm { c a t } } ^ { \prime ( l ) }  \mathrm { C o n c a t } ( X _ { 1 } ^ { \prime ( l ) } , \dots , X _ { M } ^ { \prime ( l ) } )$   
7: $\begin{array} { r } { X ^ { ' \prime \prime ( l ) } \gets X _ { \mathrm { c a t } } ^ { \prime ( l ) } \odot \bar { \sigma } ( \mathbf { W } _ { 2 } \operatorname { D r o p o u t } ( \mathrm { R e L U } ( \mathbf { W } _ { 1 } \mathrm { G A P } ( X _ { \mathrm { c a t } } ^ { \prime ( l ) } ) ) ) ) } \end{array}$   
8: $\begin{array} { r } { X ^ { \prime \prime \prime ( l ) } \gets \tilde { \mathrm { R e L U } } ( \mathrm { B N } ( \mathrm { C o n v } _ { 1 \times 1 \times 1 } ( \mathrm { D W C } _ { 3 \times 3 \times 3 } ( X ^ { \prime \prime ( l ) } \xrightarrow ) ) ) ) } \end{array}$   
9: for m = 1 to M do   
10: $Z _ { m } ^ { ( l ) } \gets \mathrm { R e L U } ( \mathrm { B N } ( \mathrm { C o n v } _ { 1 \times 1 \times 1 } ^ { ( m ) } ( X ^ { \prime \prime \prime ( l ) } ) ) )$   
11: $\alpha _ { m } ^ { ( l ) } \gets \mathrm { S o f t p l u s } ( \hat { \alpha } _ { m } ^ { ( l ) } )$   
12: $\beta _ { m } ^ { \ddot { ( l ) } }  \mathrm { S o f t p l u s } ( \hat { \beta } _ { m } ^ { \ddot { ( l ) } } )$   
13: $\begin{array} { r } { Y _ { m } ^ { ( l ) } \gets ( \beta _ { m } ^ { ( l ) ^ { \ast } } + \epsilon ) X _ { m } ^ { ( l ) ^ { \prime } } + \alpha _ { m } ^ { ( l ) } Z _ { m } ^ { ( l ) } } \end{array}$   
14: end for   
15: return $\{ Y _ { m } ^ { ( l ) } \} _ { m = 1 } ^ { M }$

During TG-BWSA training, the total objective is $\mathcal { L } _ { \mathrm { s e g } } + \mathcal { L } _ { \mathrm { g a t e } } ,$ where $\mathcal { L } _ { \mathrm { s e g } }$ is the segmentation loss retained from the corresponding backbone. BWSA uses $\mathcal { L } _ { \mathrm { s e g } }$ without the gate penalty.

## Design choices.

We use scalar gates to limit the number of added parameters. Channel-wise or spatial gates would provide finer control, but CMCR already weights channels in the combined features. Each learned gate applies to one modality and stage and remains fixed across test inputs. One shared path processes features from all modalities. Separate paths would repeat the same operations without combining their features inside the adapter. We compare the shared and separate paths in the

PMA ablation under a similar parameter budget.

## IV. EXPERIMENTAL SETUP AND EVALUATION A. DATASETS AND EVALUATION METRICS

We conduct experiments on the BraTS 2020 (369 subjects) and BraTS 2023 (1,251 subjects) datasets [2], [19]–[21]. Each dataset contains co-registered T1ce, T1, FLAIR, and T2 volumes with voxel-wise tumor annotations. We randomly assign 80% of the subjects to training and 20% to local validation. For BraTS 2023, we also report results on the official online validation set of 219 subjects. Dice is reported for all experiments. The local-validation Dice scores for 2D models are calculated on axial slices rather than reconstructed volumes and are not directly comparable with 3D localvalidation or official online Dice scores. HD95 is also reported for the BraTS 2023 online validation set.

BraTS online evaluation metrics: We report region-wise (voxel-wise) Dice and HD95 for ET, TC, and WT. Lesionwise results from the same online submission are reported in Table 9.

Data split and reproducibility: All train/validation splits are performed at the subject level to prevent slice-level leakage. For each dataset, we randomly split subjects into 80% training and 20% validation once and keep this split fixed for all methods. Local-validation results are reported as mean±standard deviation over three independent runs with different random seeds for weight initialization and data order. Each run uses the same split and checkpoint-selection rule.

## B. IMPLEMENTATION DETAILS

Backbone Architectures: We integrated BWSA with fixed weights and with trainable gates into two 3D backbones, RFNet [9] and mmFormer [5], and two 2D backbones, a Multi-Encoder U-Net (MEUNet), implemented by us for this study, and Multi-Encoder FCT (MEFCT) [22]. F2Net [6] uses Cross-modal Feature-enhanced Modules and Multimodal Collaboration Modules at multiple stages. We ran the authors’ F2Net code and calculated Dice for its predictions using the same Dice calculation as for our 2D models. For the adapter comparisons, we retained the decoder and loss function of each backbone.

Training Scenarios: We evaluated two training scenarios. RFNet, mmFormer, and MEUNet were trained from scratch. In the second scenario, the MEFCT encoders were initialized with self-supervised weights from RePre [23]. The encoder freezing protocol for pretrained MEFCT, used by TG-BWSA, BWSA, and LoRA [24] variants, is described below.

3D Model Settings: For RFNet and mmFormer, inputs were z-score normalized volumetric patches. RFNet was trained for 300 epochs on $8 0 \times 8 0 \times 8 0$ patches with a batch size of 2 on a single NVIDIA A6000 GPU, while mmFormer was trained for 1000 epochs on $1 2 8 \times 1 2 8 \times 1 2 8$ patches with a batch size of 2 on the same hardware. Both used the Adam optimizer and a polynomial learning rate decay schedule. The checkpoint with the highest mean validation Dice was used for evaluation.

2D Model Settings: MEUNet and MEFCT were trained for 100 epochs on min-max normalized 224 × 224 axial slices with a batch size of 16, distributed across four NVIDIA A6000 GPUs using data parallelism. The AdamW optimizer was employed, and the total loss combined weighted Categorical Cross-Entropy and Dice Loss.

Self-supervised pretraining of FCT encoders: Following RePre [23], the four modality-specific encoders in MEFCT are pretrained individually on unlabeled BraTS 2023 training images in the order T1ce, T1, FLAIR, and T2. For local validation, pretraining uses only images from the 80% of subjects assigned to the training split. The held-out 20% are excluded from both pretraining and supervised optimization. Each encoder jointly optimizes masked-patch inpainting and contrastive instance discrimination. We uniformly mask 15% of non-overlapping $1 6 ~ \times ~ 1 6$ patches per slice and reconstruct them with a convolutional decoder using SSIM loss. A contrastive branch applies NT-Xent [25] to projection heads over two independently augmented views. The total loss is ${ \mathcal { L } } _ { \mathrm { t o t a l } } = \lambda _ { \mathrm { S S I M } } { \mathcal { L } } _ { \mathrm { S S I M } } + \lambda _ { \mathrm { N T - X e n t } } { \mathcal { L } } _ { \mathrm { c o n t r a s t } } ,$ , with $\lambda _ { \mathrm { S S I M } } ~ = ~ 0 . 8 .$ $\lambda _ { \mathrm { N T - X e n t } } = 0 . 2$ , and temperature $\tau = 0 . 2$

Partial fine-tuning of pretrained MEFCT encoders: We use a two-phase partialfine-tuning protocol for the pretrained MEFCT encoders. In Phase 1, all encoder layers are frozen. The decoder is trained for every variant, together with the BWSA, TG-BWSA, or LoRA modules when present. In Phase 2, the encoder bottleneck blocks are also unfrozen while the earlier encoder layers remain fixed. This protocol applies to the MEFCT variants in Table 7 and those marked with <sup>†</sup> in the BraTS 2023 online validation table. Augmentations, losses, optimizer, scheduler, early stopping, and evaluation are held constant across the adapter variants.

LoRA baseline (fairness and configuration): For MEFCT, we attach LoRA blocks after each self-attention block in the four encoders, including Blocks 1–4 and the bottleneck. Each LoRA module applies a low rank residual of the form $\begin{array} { r } { \tilde { { x } } \ = \ x + \ \frac { \alpha } { r } ( x A ) B } \end{array}$ . We set $\alpha \ = \ 2 r$ , use r=16 and α=32 for the primary baseline, and apply dropout with p=0.1. The LoRA and TG-BWSA comparisons use the same data augmentation, losses, and optimization schedule. In both cases, the decoder and encoder bottlenecks are trainable. LoRA adds approximately $1 . 3 \times 1 0 ^ { 5 }$ trainable parameters across the four encoders, whereas TG-BWSA adds approximately $9 . 0 \times 1 0 ^ { 5 }$ parameters to MEFCT. The comparison therefore controls the training protocol but not the number or placement of added parameters.

Per-modality adapters (PMA): PMA replicates the TG-BWSA structure (compression, CMCR, dual gates, refinement) independently for each modality, resulting in a comparable but slightly larger parameter budget than the shared TG-BWSA.

## C. RESULTS

We report results for BWSA with fixed weights and its gated extension, TG-BWSA. The reported 0.85–3.73% parameter increase is measured from each unmodified backbone to its TG-BWSA version. The evaluations include local full-modality validation, official BraTS 2023 online validation (Table 8), and missing-modality inputs (Table 14). Section IV-D reports parameter overhead and inference cost. Section IV-E reports component ablations, gate-initialization sensitivity, missing-modality experiments, and training analyses.

## 1) Local Validation Results

Baselines and Notation: We evaluate BWSA with fixed stream weights and its gated extension. BWSA contains the shared CMCR and depthwise separable convolution path with α=1 and β=1. TG-BWSA learns these two stream weights while retaining the shared path. The following notation is used in the tables.

• Models augmented with the static BWSA are denoted with a single asterisk (\*).

• Models using the gated extension, TG-BWSA, are denoted with a double asterisk (\*\*).

• In detailed metric tables, bold marks the best value in each column, and underlining marks the second-best.

BWSA with fixed weights increases mean Dice for RFNet, mmFormer, and MEUNet on both local validation sets, and the gated extension increases it further (Table 2). Detailed region-wise results are reported in Tables 3–6, and the pretrained MEFCT comparison is reported in Table 7. MEUNet+TG-BWSA exceeds the F2Net result by 0.0230 Dice on BraTS 2020 (0.8974 vs. 0.8744) and by 0.0192 on BraTS 2023 (0.9240 vs. 0.9048).

Region-wise local-validation results: Tables 3–6 report the region-wise WT, TC, and ET results for the BraTS 2020 and BraTS 2023 local validation sets.

Adapter comparison on pretrained MEFCT: TG-BWSA has the highest mean Dice among the tested variants on BraTS 2020 and BraTS 2023 (Table 7). On BraTS 2023, its mean Dice is 0.0036 higher than the reported LoRA baseline with r=16 (0.9095 vs. 0.9059). TG-BWSA adds approximately 0.9M parameters, compared with 0.127M for LoRA. The comparison uses the same training protocol, but the methods differ in parameter count and module placement.

## 2) Region-wise Online Validation

BraTS 2023 online validation protocol: For each model, we trained three runs on the same 80% subject-level training split. We selected the run with the highest mean Dice on the held-out 20% local validation split and submitted predictions from that run to the official BraTS 2023 online validation server. Neither local nor online validation subjects were used for training.

On the official BraTS 2023 online validation set (Table 8), RFNet has an average Dice of 0.8314 without an adapter. BWSA raises it to 0.8639, and learning the weights raises it to 0.8654. The corresponding average HD95 values are 28.80, 14.37, and 16.34 mm. Thus, fixed weights give RFNet a lower average HD95 than learned weights. With mmFormer,

TABLE 2: Mean local full-modality Dice over WT, TC, and ET across three runs on BraTS 2020 and 2023. The 2D scores are slice-wise and are not directly comparable with the 3D scores. Bold marks the best variant within each backbone row. The last column shows the parameter increase from each unmodified backbone to its TG-BWSA version. Region-wise results and pretrained MEFCT comparisons appear in Tables 3–6 and Table 7, respectively.
<table><tr><td>Backbone</td><td colspan="3">BraTS 2020 mean Dice</td><td colspan="3">BraTS 2023 mean Dice</td><td>TG-BWSA overhead</td></tr><tr><td></td><td>Base</td><td>+BWSA</td><td>+TG-BWSA</td><td>Base</td><td>+BWSA</td><td>+TG-BWSA</td><td></td></tr><tr><td>RFNet [9]</td><td>0.8623</td><td>0.8693</td><td>0.8712</td><td>0.8941</td><td>0.8955</td><td>0.8982</td><td>3.73%</td></tr><tr><td>mmFormer [5]</td><td>0.8635</td><td>0.8695</td><td>0.8715</td><td>0.8976</td><td>0.8995</td><td>0.9013</td><td>0.92%</td></tr><tr><td>MEUNet</td><td>0.8873</td><td>0.8957</td><td>0.8974</td><td>0.9169</td><td>0.9198</td><td>0.9240</td><td>2.94%</td></tr><tr><td>F2Net [6]</td><td>0.8744</td><td></td><td></td><td>0.9048</td><td></td><td></td><td></td></tr></table>

TABLE 3: BraTS 2020 local-validation Dice for 3D models (mean±standard deviation over three runs).
<table><tr><td>Model</td><td>WT</td><td>TC</td><td>ET</td><td>Mean</td></tr><tr><td>RFNet [9]</td><td> $0 . 9 1 1 0 { \scriptstyle \pm 0 . 0 0 3 2 }$ </td><td> $0 . 8 8 2 5 { \scriptstyle \pm 0 . 0 0 3 1 }$ </td><td> $0 . 7 9 3 5 { \scriptstyle \pm 0 . 0 0 2 9 }$ </td><td>0.8623</td></tr><tr><td>RFNet*</td><td> $0 . 9 1 3 8 { \pm } 0 . 0 0 1 6$ </td><td> $0 . 8 8 4 7 { \scriptstyle \pm 0 . 0 0 1 9 }$ </td><td> $0 . 8 0 9 4 { \scriptstyle \pm 0 . 0 0 1 7 }$ </td><td>0.8693</td></tr><tr><td>RFNet**</td><td> $0 . 9 1 6 1 { \scriptstyle \pm 0 . 0 0 1 8 }$ </td><td> $\mathbf { 0 . 8 8 6 1 { \scriptstyle \pm 0 . 0 0 2 0 } }$ </td><td> $0 . 8 1 1 4 { \scriptstyle \pm 0 . 0 0 1 1 }$ </td><td>0.8712</td></tr><tr><td>mmFormer [5]</td><td> $\overline { { 0 . 9 1 3 3 } } \pm 0 . 0 0 3 4$ </td><td> $0 . 8 7 9 8 { \scriptstyle \pm 0 . 0 0 3 0 }$ </td><td> $0 . 7 9 7 4 { \scriptstyle \pm 0 . 0 0 2 8 }$ </td><td>0.8635</td></tr><tr><td>mmFormer*</td><td> $0 . 9 1 5 1 { \scriptstyle \pm 0 . 0 0 2 8 }$ </td><td> $0 . 8 8 2 9 { \scriptstyle \pm 0 . 0 0 2 3 }$ </td><td> $0 . 8 1 0 5 { \scriptstyle \pm 0 . 0 0 2 6 }$ </td><td>0.8695</td></tr><tr><td> $\mathrm { m m F o r m e r ^ { * * } }$ </td><td> $\mathbf { 0 . 9 1 6 5 } \pm 0 . 0 0 1 7$ </td><td> $0 . 8 8 4 7 { \scriptstyle \pm 0 . 0 0 2 5 }$ </td><td> $\mathbf { 0 . 8 1 3 4 } \pm 0 . 0 0 2 3$ </td><td>0.8715</td></tr></table>

TABLE 4: BraTS 2020 slice-wise local-validation Dice for 2D models with trainable encoders (mean±standard deviation over three runs).
<table><tr><td>Model</td><td>WT</td><td>TC</td><td>ET</td><td>Mean</td></tr><tr><td>MEUNet</td><td> $0 . 8 7 2 4 { \scriptstyle \pm 0 . 0 0 2 9 }$ </td><td> $0 . 8 9 2 5 { \scriptstyle \pm 0 . 0 0 3 7 }$ </td><td> $0 . 8 9 7 1 { \scriptstyle \pm 0 . 0 0 3 1 }$ </td><td>0.8873</td></tr><tr><td>MEUNet*</td><td> $0 . 8 8 3 8 { \pm } 0 . 0 0 1 1$ </td><td> $0 . 9 0 1 9 { \scriptstyle \pm 0 . 0 0 1 5 }$ </td><td> $0 . 9 0 1 4 { \scriptstyle \pm 0 . 0 0 1 2 }$ </td><td>0.8957</td></tr><tr><td> $\mathbf { M E U N e t ^ { * * } }$ </td><td> $\overline { { 0 . 8 8 5 3 } } \pm 0 . 0 0 1 3$ </td><td> $\mathbf { 0 . 9 0 4 1 } { \scriptstyle \pm 0 . 0 0 1 1 }$ </td><td> $\overline { { { \bf 0 . 9 0 2 9 } } } \pm 0 . 0 0 1 0$ </td><td>0.8974</td></tr><tr><td>F2Net [6]</td><td> $0 . 8 2 1 3 { \scriptstyle \pm 0 . 0 0 8 6 }$ </td><td> $0 . 9 0 3 4 { \pm } 0 . 0 0 3 5$ </td><td> $0 . 8 9 8 5 { \scriptstyle \pm 0 . 0 0 4 4 }$ </td><td>0.8744</td></tr></table>

TABLE 5: BraTS 2023 local-validation Dice for 3D models (mean±standard deviation over three runs).
<table><tr><td>Model</td><td>WT</td><td>TC</td><td>ET</td><td>Mean</td></tr><tr><td>RFNet [9]</td><td> $0 . 9 2 4 4 { \pm } 0 . 0 0 2 2$ </td><td> $0 . 8 9 7 6 { \scriptstyle \pm 0 . 0 0 2 3 }$ </td><td> $0 . 8 6 0 3 { \scriptstyle \pm 0 . 0 0 1 9 }$ </td><td>0.8941</td></tr><tr><td>RFNet*</td><td> $0 . 9 2 6 1 { \scriptstyle \pm 0 . 0 0 1 5 }$ </td><td> $0 . 8 9 9 0 { \scriptstyle \pm 0 . 0 0 0 2 }$ </td><td> $0 . 8 6 1 3 { \scriptstyle \pm 0 . 0 0 1 5 }$ </td><td>0.8955</td></tr><tr><td>RFNet**</td><td> $0 . 9 2 9 3 { \scriptstyle \pm 0 . 0 0 0 9 }$ </td><td> $0 . 9 0 0 7 { \scriptstyle \pm 0 . 0 0 1 1 }$ </td><td> $0 . 8 6 4 7 { \scriptstyle \pm 0 . 0 0 0 8 }$ </td><td>0.8982</td></tr><tr><td>mmFormer [5]</td><td> $0 . 9 2 9 3 { \scriptstyle \pm 0 . 0 0 4 3 }$ </td><td> $0 . 9 0 1 1 { \scriptstyle \pm 0 . 0 0 4 0 }$ </td><td> $0 . 8 6 2 4 { \scriptstyle \pm 0 . 0 0 4 2 }$ </td><td>0.8976</td></tr><tr><td>mmFormer*</td><td> $0 . 9 3 0 7 { \scriptstyle \pm 0 . 0 0 1 7 }$ </td><td> $0 . 9 0 2 9 { \scriptstyle \pm 0 . 0 0 1 3 }$ </td><td> $\mathbf { 0 . 8 6 4 8 { \scriptstyle \pm 0 . 0 0 1 0 } }$ </td><td>0.8995</td></tr><tr><td>mmFormer**</td><td> $\overline { { 0 . 9 3 2 2 } } \pm 0 . 0 0 1 5$ </td><td> $\mathbf { 0 . 9 0 7 0 { \scriptstyle \pm 0 . 0 0 1 6 } }$ </td><td> $0 . 8 6 4 7 { \scriptstyle \pm 0 . 0 0 1 3 }$ </td><td>0.9013</td></tr></table>

TABLE 6: BraTS 2023 slice-wise local-validation Dice for 2D MEUNet (mean±standard deviation over three runs).
<table><tr><td>Model</td><td>WT</td><td>TC</td><td>ET</td><td>Mean</td></tr><tr><td>MEUNet</td><td> $0 . 8 9 5 5 { \scriptstyle \pm 0 . 0 0 3 3 }$ </td><td> $0 . 9 2 9 1 { \scriptstyle \pm 0 . 0 0 3 0 }$ </td><td> $0 . 9 2 6 0 { \scriptstyle \pm 0 . 0 0 3 4 }$ </td><td>0.9169</td></tr><tr><td>MEUNet*</td><td> $0 . 9 0 1 7 { \scriptstyle \pm 0 . 0 0 1 2 }$ </td><td> $0 . 9 3 0 6 { \scriptstyle \pm 0 . 0 0 1 3 }$ </td><td> $\underline { { 0 . 9 2 7 1 } } \pm 0 . 0 0 1 0$ </td><td>0.9198</td></tr><tr><td>MEUNet**</td><td> $\mathbf { 0 . 9 0 3 7 } \pm 0 . 0 0 1 1$ </td><td> $\mathbf { 0 . 9 3 4 5 { \scriptstyle \pm 0 . 0 0 1 3 } }$ </td><td> $\mathbf { 0 . 9 3 3 8 { \pm } 0 . 0 0 1 2 }$ </td><td>0.9240</td></tr><tr><td>F2Net [6]</td><td> $0 . 8 6 3 4 { \scriptstyle \pm 0 . 0 2 4 9 }$ </td><td> $0 . 9 2 7 1 { \scriptstyle \pm 0 . 0 0 3 2 }$ </td><td> $0 . 9 2 4 0 { \scriptstyle \pm 0 . 0 0 2 4 }$ </td><td>0.9048</td></tr></table>

MEUNet, and MEFCT, the gated form increases average Dice over the unmodified backbone by 0.0769, 0.0283, and 0.0450, respectively. It also reduces average HD95 for these three backbones. The added effect of learned weights varies by region and backbone.

## 3) Lesion-wise Online Validation

The BraTS 2023 segmentation evaluation reports both regionwise and lesion-wise Dice and HD95. The lesion-wise metrics, which are used for challenge ranking, evaluate connected lesions and penalize missed and false-positive components [26], [27]. Table 8 reports region-wise results for the evaluated models. Table 9 includes lesion-wise results from the same RFNet+BWSA and RFNet+TG-BWSA online submissions, alongside published comparators. Region-wise Dice measures voxel overlap within each tumor region. We therefore report the two types separately.

TABLE 7: Adapter comparison for 2D MEFCT on BraTS 2020 and 2023 local validation sets using slice-wise Dice under partial fine-tuning of pretrained encoders (bottleneck blocks unfrozen). Values are mean±standard deviation over three runs.
<table><tr><td colspan="5">(a) BraTS 2020</td></tr><tr><td>Model</td><td>WT</td><td>TC</td><td>ET</td><td>Mean</td></tr><tr><td>MEFCT</td><td>0.8614±0.0041</td><td> $0 . 8 7 9 2 { \scriptstyle \pm 0 . 0 0 3 5 }$ </td><td> $0 . 8 7 7 5 { \scriptstyle \pm 0 . 0 0 4 7 }$ </td><td>0.8727</td></tr><tr><td>MEFCT*</td><td>0.8708±0.0013</td><td> $0 . 8 8 0 2 { \scriptstyle \pm 0 . 0 0 1 0 }$ </td><td> $0 . 8 8 9 4 { \scriptstyle \pm 0 . 0 0 1 4 }$ </td><td>0.8801</td></tr><tr><td>MEFCT**</td><td> $\textcircled { 0 . 8 7 2 0 } \pm 0 . 0 0 1 1$ </td><td> $\mathbf { 0 . 8 8 7 1 { \scriptstyle \pm 0 . 0 0 0 9 } }$ </td><td> $\overline { { 0 . 8 9 1 1 } } \pm 0 . 0 0 0 8$ </td><td>0.8834</td></tr><tr><td> $\mathrm { M E F C T - L o R A } \left[ 2 4 \right]$ </td><td> $0 . 8 6 9 4 { \scriptstyle \pm 0 . 0 0 1 4 }$ </td><td> $0 . 8 8 1 3 { \pm } 0 . 0 0 1 9$ </td><td> $0 . 8 8 7 3 { \scriptstyle \pm 0 . 0 0 0 7 }$ </td><td>0.8793</td></tr></table>

<table><tr><td colspan="5">(b) BraTS 2023</td></tr><tr><td>Model</td><td>WT</td><td>TC</td><td>ET</td><td>Mean</td></tr><tr><td>MEFCT</td><td>0.8741±0.0044</td><td> $0 . 9 1 5 2 { \scriptstyle \pm 0 . 0 0 3 7 }$ </td><td> $0 . 9 0 8 0 { \scriptstyle \pm 0 . 0 0 4 2 }$ </td><td>0.8991</td></tr><tr><td>MEFCT*</td><td> $0 . 8 8 7 9 { \scriptstyle \pm 0 . 0 0 1 3 }$ </td><td> $0 . 9 1 8 3 { \scriptstyle \pm 0 . 0 0 0 9 }$ </td><td> $0 . 9 1 4 4 { \pm } 0 . 0 0 1 4$ </td><td>0.9069</td></tr><tr><td>MEFCT**</td><td> $\overline { { { \bf 0 . 8 9 0 2 } } } \pm 0 . 0 0 1 1$ </td><td> $\mathbf { 0 . 9 2 2 3 { \scriptstyle \pm 0 . 0 0 0 8 } }$ </td><td> $\overline { { { \bf 0 . 9 1 6 0 } } } \pm 0 . 0 0 1 0$ </td><td>0.9095</td></tr><tr><td> $\mathrm { M E F C T - L o R A } \left[ 2 4 \right]$ </td><td> $0 . 8 8 5 9 { \pm } 0 . 0 0 1 5$ </td><td> $0 . 9 1 9 0 { \scriptstyle \pm 0 . 0 0 0 8 }$ </td><td> $0 . 9 1 2 9 { \scriptstyle \pm 0 . 0 0 1 6 }$ </td><td>0.9059</td></tr></table>

Lesion-wise comparison: RFNet+TG-BWSA has an average lesion-wise Dice of 0.852 and average lesion-wise HD95 of 20.61, compared with 0.848 and 23.33 for the method of Zheng et al. [28]. The three highest-scoring listed teams have average lesion-wise Dice values of 0.876–0.888, above the 0.852 obtained by RFNet+TG-BWSA. The listed teams used different training and inference pipelines. Their scores show how our results compare with other submissions, but they do not isolate the effect of the adapter.

## 4) Qualitative Analysis

Figure 4 shows cases sampled at random from the BraTS 2023 local validation set for the 2D and 3D backbones, without ranking cases by model performance. In the displayed 2D cases, the adapter variants recover parts of the annotated tumor extent that are absent or undersegmented in some backbone predictions. TG-BWSA also yields boundaries that more closely follow the ground truth in several examples.

TABLE 8: BraTS 2023 online validation results for all models (Dice/HD95). <sup>†</sup>Pretrained encoders use partial fine-tuning according to the protocol in Sec. IV-B.
<table><tr><td>Model</td><td>Dice ET</td><td>Dice TC</td><td>Dice WT</td><td>HD95 ET</td><td>HD95 TC</td><td>HD95 WT</td><td>Dice Avg</td><td>HD95 Avg</td></tr><tr><td>RFNet [9]</td><td>0.7889</td><td>0.8201</td><td>0.8853</td><td>35.4500</td><td>26.8100</td><td>24.1500</td><td>0.8314</td><td>28.8033</td></tr><tr><td>RFNet + BWSA</td><td>0.8313</td><td>0.8511</td><td>0.9093</td><td>20.5500</td><td>16.3800</td><td>6.1700</td><td>0.8639</td><td>14.3667</td></tr><tr><td>RFNet + TG-BWSA</td><td>0.8364</td><td>0.8472</td><td>0.9127</td><td>18.9200</td><td>23.3400</td><td>6.7700</td><td>0.8654</td><td>16.3433</td></tr><tr><td>mmFormer [5]</td><td>0.6209</td><td>0.7025</td><td>0.8289</td><td>48.1700</td><td>33.2500</td><td>24.7900</td><td>0.7174</td><td>35.4033</td></tr><tr><td>mmFormer + BWSA</td><td>0.7502</td><td>0.7745</td><td>0.8563</td><td>26.9400</td><td>28.6200</td><td>16.5000</td><td>0.7937</td><td>24.0200</td></tr><tr><td>mmFormer + TG-BWSA</td><td>0.7685</td><td>0.7751</td><td>0.8394</td><td>23.5300</td><td>22.4800</td><td>19.8300</td><td>0.7943</td><td>21.9467</td></tr><tr><td>Multi-Enc U-Net</td><td>0.7467</td><td>0.8147</td><td>0.8918</td><td>26.4400</td><td>16.9600</td><td>9.1200</td><td>0.8177</td><td>17.5067</td></tr><tr><td>Multi-Enc U-Net + BWSA</td><td>0.7607</td><td>0.8329</td><td>0.8972</td><td>22.5500</td><td>13.2300</td><td>7.0100</td><td>0.8303</td><td>14.2633</td></tr><tr><td>Multi-Enc U-Net + TG-BWSA</td><td>0.7962</td><td>0.8378</td><td>0.9040</td><td>18.0200</td><td>12.5800</td><td>7.8800</td><td>0.8460</td><td>12.8267</td></tr><tr><td>Multi-Enc FCT† Multi-Enc FCT + BWSA†</td><td>0.7160</td><td>0.7533</td><td>0.8691</td><td>31.7300</td><td>29.9700</td><td>9.9900</td><td>0.7795</td><td>23.8967</td></tr><tr><td>Multi-Enc FCT + TG-BWSA†</td><td>0.7652</td><td>0.8120</td><td>0.8856</td><td>18.5655</td><td>18.8615</td><td>7.9558</td><td>0.8209</td><td>15.1276</td></tr><tr><td></td><td>0.7732</td><td>0.8133</td><td>0.8871</td><td>18.7600</td><td>18.1300</td><td>7.7400</td><td>0.8245</td><td>14.8767</td></tr><tr><td> $\mathrm { M u l t i  – E n c ~ F C T } + \mathrm { L o R A } ^ { \dag } \ [ 2 4 ]$ </td><td>0.7692</td><td>0.8024</td><td>0.8847</td><td>20.2674</td><td>24.2854</td><td>7.7655</td><td>0.8188</td><td>17.4394</td></tr></table>

TABLE 9: BraTS 2023 online validation lesion-wise Dice and HD95. Comparator values and anonymized team identifiers are transcribed from Zheng et al. [28].
<table><tr><td>Model</td><td>L-Dice ET</td><td>L-Dice TC</td><td>L-Dice WT</td><td>L-HD95 ET</td><td>L-HD95 TC</td><td>L-HD95 WT</td><td>Avg L-Dice</td><td>Avg L-HD95</td></tr><tr><td>Team @s7zuqer</td><td>0.866</td><td>0.89</td><td>0.909</td><td>10.594</td><td>7.228</td><td>11.086</td><td>0.888</td><td>9.636</td></tr><tr><td>Team @simis</td><td>0.857</td><td>0.876</td><td>0.908</td><td>13.649</td><td>12.825</td><td>11.270</td><td>0.880</td><td>12.581</td></tr><tr><td>Team @qian8</td><td>0.856</td><td>0.868</td><td>0.904</td><td>14.070</td><td>13.778</td><td>10.735</td><td>0.876</td><td>12.861</td></tr><tr><td>Faking_it [29]</td><td>0.854</td><td>0.858</td><td>0.887</td><td>17.321</td><td>18.007</td><td>19.322</td><td>0.866</td><td>18.217</td></tr><tr><td>BiomedMBZ [30]</td><td>0.837</td><td>0.859</td><td>0.903</td><td>25.451</td><td>19.185</td><td>12.685</td><td>0.866</td><td>19.107</td></tr><tr><td>Shenhai Zheng [28]</td><td>0.823</td><td>0.860</td><td>0.862</td><td>25.602</td><td>15.110</td><td>29.291</td><td>0.848</td><td>23.334</td></tr><tr><td>Ours (RFNet + BWSA)</td><td>0.834</td><td>0.828</td><td>0.883</td><td>21.585</td><td>31.274</td><td>17.343</td><td>0.848</td><td>23.401</td></tr><tr><td>Ours  $( \mathbf { R F N e t } + \mathbf { T G } \mathbf { - B } \mathbf { W S A } )$ </td><td>0.832</td><td>0.837</td><td>0.888</td><td>21.309</td><td>25.153</td><td>15.369</td><td>0.852</td><td>20.610</td></tr></table>

In the displayed 3D cases, BWSA and TG-BWSA reduce several isolated false positive regions and correct some areas of oversegmentation or undersegmentation relative to the corresponding backbones.

## D. PARAMETER OVERHEAD AND INFERENCE COST

## 1) Parameter Overhead

Table 10 compares each unmodified backbone with its TG-BWSA version. BWSA uses the same adapter layers and lacks only the 2ML scalar gates, so its parameter count is nearly identical to that of TG-BWSA. The percentage overhead in the table is relative to the unmodified backbone and is not the cost of gating alone.

TABLE 10: Parameter counts (millions) for each unmodified backbone and its TG-BWSA version. The reported overhead compares these two models. Counts are rounded to one decimal place; percentages may differ from calculations using the displayed counts.
<table><tr><td>Backbone</td><td> $P _ { \mathrm { b a s e } }$ </td><td>P+TG-BWSA</td><td> $\Delta \%$ </td></tr><tr><td>RFNet</td><td>9.0</td><td>9.3</td><td>3.73</td></tr><tr><td>mmFormer</td><td>36.7</td><td>37.0</td><td>0.92</td></tr><tr><td>Multi-Enc. U-Net</td><td>29.8</td><td>30.7</td><td>2.94</td></tr><tr><td>Multi-Enc. FCT</td><td>102.7</td><td>103.6</td><td>0.85</td></tr></table>

## 2) Inference Cost

We measured wall-clock latency for a single forward pass with an $8 0 ^ { 3 }$ input, peak GPU memory, and theoretical GFLOPs for RFNet with and without TG-BWSA on one

NVIDIA A6000 GPU. Table 11 shows increases of 2.4% (37 ms) in latency, 8.2% in peak memory, and 4.1% in GFLOPs. These measurements apply to the reported input size, backbone, implementation, and hardware configuration.

TABLE 11: Inference cost for one forward pass with an $8 0 ^ { 3 }$ input on an NVIDIA A6000 GPU.
<table><tr><td>Model</td><td>Latency (s/input)</td><td>GPU Memory (GB)</td><td>GFLOPs</td></tr><tr><td>RFNet [9]</td><td>1.544</td><td>1.95</td><td>102.8</td></tr><tr><td>RFNet**</td><td>1.581</td><td>2.11</td><td>107.0</td></tr></table>

## E. ABLATION STUDIES AND TRAINING ANALYSIS

We use RFNet on the BraTS 2023 local validation set for the component and missing-modality analyses. We also examine gate initialization, regularization, training loss, and the learned gate coefficients (Figs. 5 and 6).

## 1) Component-wise evaluation of BWSA and its gated extension

Table 12 compares fixed and learned weights with changes to the main adapter components over three runs. The noresidual variant removes the direct original-feature path. In the RFNet ablation, BWSA scores 0.8955 mean Dice with fixed weights and 0.8982 with learned gates, a difference of 0.0027. Removing the original-feature path lowers the mean to 0.8944. PMA uses four independent adapters with the same internal operations and a similar total parameter count, but does not combine modality features within an adapter. BWSA with learned gates scores 0.8982, compared with 0.8947 for PMA under this RFNet protocol.

![](images/2493e0b62330622326c433f2feda4962f94ae3cb9c19a27802a36fb76bcffed1.jpg)  
(a) 2D models (MEFCT and MEUNet)

![](images/165621f2f8a129f0946f581bee47c509964946c8bc12b8ca3294d4d280c31db2.jpg)  
(b) 3D models (RFNet and mmFormer)

FIGURE 4: Qualitative segmentation comparisons on randomly sampled BraTS 2023 local validation cases. Predictions are shown for the unmodified backbones and their adapter variants. Cyan denotes non-enhancing tumor, green denotes edema, and yellow denotes enhancing tumor.  
![](images/2a74401eda435e4a8414d4065ec578779a846b3a97ac2fcc0c072f9523770f08.jpg)  
FIGURE 5: Training loss for RFNet and RFNet+TG-BWSA. The dashed line marks activation of the region-fusion loss near epoch 100. The curves describe optimization on the training data and should be interpreted together with the validation results.

## 2) Gate initialization and regularization

We evaluated five combinations of $\alpha _ { 0 } , \beta _ { 0 }$ , and λ (Table 13). Across these settings, WT Dice ranges from 0.9263 to 0.9293, TC from 0.8964 to 0.9007, and ET from 0.8598 to 0.8647. The narrow ranges show limited sensitivity to the five tested settings.

TABLE 12: Component-wise evaluation of adapter variants on BraTS 2023 using RFNet (mean±standard deviation over three runs). Best results are in bold, and second-best results are underlined.
<table><tr><td>Configuration</td><td>WT</td><td>TC</td><td>ET</td><td>Mean</td></tr><tr><td>TG-BWSA (learned gates)</td><td>0.9293±0.0009</td><td>0.9007±0.0011</td><td>0.8647±0.0008</td><td>0.8982</td></tr><tr><td>BWSA (fixed weights)</td><td>0.9261±0.0015</td><td>0.8990±0.0002</td><td>0.8613±0.0015</td><td>0.8955</td></tr><tr><td>Per-Modality Adapters (PMA)</td><td>0.9250±0.0018</td><td>0.8985±0.0009</td><td>0.8605±0.0012</td><td>0.8947</td></tr><tr><td>No Residual Path</td><td>0.9230±0.0025</td><td>0.9001±0.0018 0.8601±0.0020</td><td></td><td>0.8944</td></tr><tr><td>Baseline (RFNet)</td><td>0.9244±0.0022</td><td>0.8976±0.0023</td><td>0.8603±0.0019</td><td>0.8941</td></tr><tr><td>w/o CMCR (no channel recalibration)</td><td>0.9265±0.0024</td><td>0.8964±0.0020</td><td>0.8562±0.0022</td><td>0.8930</td></tr><tr><td>w/o Depthwise Separable Convolution</td><td>0.9251±0.0027 0.8955±0.0022 0.8547±0.0023</td><td></td><td></td><td>0.8918</td></tr><tr><td>Basic Adapter (1 × 1 conv only)</td><td>0.9175±0.0031 0.8870±0.0026 0.8394±0.0028 0.8813</td><td></td><td></td><td></td></tr></table>

TABLE 13: Effect of gate initialization and regularization on BraTS 2023 with RFNet.
<table><tr><td>α0</td><td> $\beta _ { 0 }$ </td><td>λ</td><td>WT</td><td>TC</td><td>ET</td></tr><tr><td>0.5</td><td>1.0</td><td>1e-4</td><td>0.9293</td><td>0.9007</td><td>0.8647</td></tr><tr><td>0.1</td><td>1.0</td><td>1e-4</td><td>0.9272</td><td>0.8991</td><td>0.8614</td></tr><tr><td>0.5</td><td>0.5</td><td>1e-4</td><td>0.9269</td><td>0.8970</td><td>0.8598</td></tr><tr><td>0.5</td><td>1.0</td><td>1e-5</td><td>0.9280</td><td>0.9001</td><td>0.8631</td></tr><tr><td>0.5</td><td>1.0</td><td>1e-3</td><td>0.9263</td><td>0.8964</td><td>0.8602</td></tr></table>

## 3) Performance with Missing Modalities

Ablation Protocol: At test time, a missing modality is simulated by zeroing its input volume while keeping standard preprocessing for the others. Inference is run in evaluation mode to prevent normalization updates.

Results: We evaluate RFNet with and without BWSA, using either fixed weights or learned gates, over the 14 incomplete subsets of {T1ce, T1, FLAIR, T2}. The input containing all four modalities is reported separately with the complete input results. Table 14 averages only the incomplete input configurations. Both forms of BWSA improve mean WT, TC, and ET Dice over RFNet. Fixed weights give higher TC Dice (0.796 vs. 0.779), while learned weights give slightly higher WT (0.834 vs. 0.825) and ET (0.629 vs. 0.627) Dice.

![](images/2e02feb886e8a23c8c9929cc1c6fd8987043b885b503ab77d529f600bebf2f56.jpg)

![](images/a86de81f6cdfda4fea6a80f6fd0f361aae2251b386153f5241c55c7c8b7a3a7d.jpg)

![](images/e28cb2846edcc088fff26a2252efae49c9891f38b4ce0577072a3cb21d388fab.jpg)

![](images/dcd0e9cfb932e521515368894fb6f2c1daf2149707f82d59390a44521802835c.jpg)

(a) Gate Trajectories During Training (Encoder Stages 1-4)  
![](images/1a9492bdceb52ad8bb04643ab63be76a589eaccc573c58a2dd81516bd6c6bac2.jpg)

![](images/634ab92c3de6c7603a2034c4502af6a2399dd174f5a68d60e00561492629f121.jpg)  
(b) Coefficient Values Across Training Epochs

![](images/b0e658a725508abee6c934b4e8f3b32b479fb88aede5ff09147f338ed9d2b8bc.jpg)

![](images/b28582b202407f704c2229118d4869fe5835b0e1386fc9043faf033bf06fe878.jpg)  
(c) Final Learned Gate Heatmaps  
FIGURE 6: Learned gate coefficients for RFNet+TG-BWSA over 300 training epochs. (a) Coefficient trajectories for each modality and encoder stage. (b) Distributions of values recorded across training epochs. (c) Final coefficients after epoch 300. α weights the adapted stream and β weights the original stream.

TABLE 14: Mean Dice across the 14 incomplete-input modality configurations on BraTS 2023. The complete-input case is excluded and reported with the full-modality results. Differences in the text are calculated from unrounded means.
<table><tr><td>Region</td><td>RFNet</td><td>+BWSA</td><td>+TG-BWSA</td></tr><tr><td>WT</td><td>0.754</td><td>0.825</td><td>0.834</td></tr><tr><td>TC</td><td>0.531</td><td>0.796</td><td>0.779</td></tr><tr><td>ET</td><td>0.430</td><td>0.627</td><td>0.629</td></tr></table>

## 4) Training Loss and Gate Analysis

Figure 5 compares the recorded RFNet and RFNet+TG-BWSA training losses. The TG-BWSA curve is lower over most of the plotted epochs, but training loss alone does not establish convergence rate or validation performance.

Figure 6 shows the trajectories of α and β over 300 training epochs, distributions of their recorded values across those epochs, and their final values for each modality and stage. The boxplot medians summarize the training trajectories and need not equal the final coefficients. After training, the coefficients are fixed model parameters and do not adapt to individual test inputs.

For FLAIR at Stage 4, α increases from its initialization near 0.5 to 0.9220. At Stage 2, the final α and β values for T1ce are close (0.7056 and 0.7016). The final coefficients differ across modalities and stages. They weight the adapted and original streams, but do not establish a link between an individual gate and a particular tumor region.

## V. DISCUSSION

BWSA increases mean Dice over the unmodified model across the four evaluated backbones in local and online validation. Learning the weights does not improve every metric. On BraTS 2023 online validation, the gated form has higher average Dice on all four backbones, but fixed weights give RFNet a lower average HD95 (14.37 vs. 16.34 mm; Table 8). Under incomplete RFNet inputs, fixed weights give higher mean TC Dice, whereas learned weights give higher WT and ET Dice (Table 14). For MEUNet, learning the weights raises average online Dice from 0.8303 to 0.8460. Both forms of BWSA improved average Dice and HD95 over the unmodified backbones on BraTS 2023 online validation. Their parameter counts are nearly identical. For a given backbone, the choice between fixed blending and learned weights should be guided by validation performance on the metric of interest.

The gate coefficients are fixed after training and cannot adjust to the reliability of an individual test image. Their trajectories do not explain the observed differences between tumor regions. In the RFNet ablation, TG-BWSA has higher mean Dice than PMA with a similar parameter count (0.8982 vs. 0.8947; Table 12). The measured latency, memory, and GFLOPs apply to RFNet on an NVIDIA A6000 (Table 11); the cost for the other backbones was not measured.

For the lesion-wise online metrics, RFNet+TG-BWSA has higher average Dice (0.852 vs. 0.848) and lower average HD95 (20.61 vs. 23.40 mm) than RFNet+BWSA (Table 9). Its average lesion-wise Dice remains below the three highest-scoring teams listed in that table (0.876–0.888). The compared submissions use different training and inference pipelines, so these scores provide benchmark context rather than an isolated comparison of architectures.

## LIMITATIONS AND FUTURE WORK

The evaluation uses BraTS 2020 and BraTS 2023 without an external cohort. Each online score comes from one of three runs selected by local validation Dice, so variation in online scores across seeds was not measured. Missing modalities are simulated by zeroing complete input volumes, and modality dropout is not used during training. The LoRA comparison differs in parameter count and module placement, so it does not measure efficiency per parameter. Future studies can test scanner and protocol shifts, partial image corruption, and modality dropout during training.

## VI. CONCLUSION

BWSA lets separate modality encoders combine features through one shared adapter between stages. With fixed weights, it adds the adapted features to the original features. TG-BWSA extends the same adapter by learning those weights for each modality and stage. We evaluated both forms with four backbones on BraTS 2020 and BraTS 2023. On the official BraTS 2023 online validation set, BWSA increased average region-wise Dice by 1.3–7.6 percentage points relative to the unmodified backbones. The gated extension increased it by 2.8–7.7 points. Both forms increased parameter counts by approximately 0.85–3.73% relative to the unmodified backbones. Both forms reduced average HD95 relative to the corresponding unmodified backbones. The choice between fixed blending and learned weights should be guided by validation results for the chosen backbone and evaluation metric.

## DATA AVAILABILITY

The BraTS 2020 and BraTS 2023 datasets analyzed in this study are available from the BraTS challenge repositories, subject to the respective access and data-use conditions. BraTS 2023 is distributed via Synapse.

## ETHICS STATEMENT

This secondary analysis used publicly available and deidentified BraTS 2020 and BraTS 2023 data. The authors recruited no participants, collected no new human subject data, and accessed no identifiable information. Therefore, no additional institutional review board approval or informed consent was required. Original data collection and release followed the ethics and data use procedures of the contributing institutions and the BraTS organizers.

## CODE AVAILABILITY

Code and experiment scripts are available at https://github.   
com/milad67/TG-BWSA.

## AUTHOR CONTRIBUTIONS

M.R. contributed to conceptualization, methods, software, analysis, and writing. M.M. contributed to methods and design. E.K. contributed to interpretation and revision. M.S. supervised the study. All authors approved the manuscript.

## REFERENCES

[1] P. Jyothi and A. R. Singh, ‘‘Deep learning models and traditional automated techniques for brain tumor segmentation in MRI: A review,’’ Artif. Intell. Rev., vol. 56, no. 4, pp. 2923–2969, Apr. 2023, doi: 10.1007/s10462-022-10245-x.

[2] U. Baid et al., ‘‘The RSNA-ASNR-MICCAI BraTS 2021 benchmark on brain tumor segmentation and radiogenomic classification,’’ arXiv:2107.02314, 2021, doi: 10.48550/arXiv.2107.02314.

[3] S. R. Stahlschmidt, B. Ulfenborg, and J. Synnergren, ‘‘Multimodal deep learning for biomedical data fusion: A review,’’ Brief. Bioinform., vol. 23, no. 2, Art. no. bbab569, Mar. 2022, doi: 10.1093/bib/bbab569.

[4] Z. Xing, L. Yu, L. Wan, T. Han, and L. Zhu, ‘‘NestedFormer: Nested modality-aware transformer for brain tumor segmentation,’’ in Proc. Int. Conf. Med. Image Comput. Comput.-Assist. Intervent. (MICCAI), 2022, pp. 140–150, doi: 10.1007/978-3-031-16443-9\_14.

[5] Y. Zhang et al., ‘‘mmFormer: Multimodal medical transformer for incomplete multimodal learning of brain tumor segmentation,’’ in Proc. Int. Conf. Med. Image Comput. Comput.-Assist. Intervent. (MICCAI), 2022, pp. 107–117, doi: 10.1007/978-3-031-16443-9\_11.

[6] H. Yang, T. Zhou, Y. Zhou, Y. Zhang, and H. Fu, ‘‘Flexible fusion network for multi-modal brain tumor segmentation,’’ IEEE J. Biomed. Health Inform., vol. 27, no. 7, pp. 3349–3359, Jul. 2023, doi: 10.1109/JBHI.2023.3271808.

[7] X. Shi et al., ‘‘Multi-modal medical SAM: An adaptation method of segment anything model (SAM) for glioma segmentation using multimodal MR images,’’ ACM Trans. Comput. Healthcare, vol. 6, no. 2, Art. no. 24, pp. 1–21, Apr. 2025, doi: 10.1145/3712297.

[8] H. Lu, M. Yu, X. Wei, X. Xu, and J. Xu, ‘‘Unbiased multimodal fusion for medical image segmentation based on dual-stream adapter,’’ Knowl.-Based Syst., vol. 330, Art. no. 114653, Nov. 2025, doi: 10.1016/j.knosys.2025.114653.

[9] Y. Ding, X. Yu, and Y. Yang, ‘‘RFNet: Region-aware fusion network for incomplete multi-modal brain tumor segmentation,’’ in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2021, pp. 3955–3964, doi: 10.1109/ICCV48922.2021.00394.

[10] A. Hatamizadeh, V. Nath, Y. Tang, D. Yang, H. R. Roth, and D. Xu, ‘‘Swin UNETR: Swin transformers for semantic segmentation of brain tumors in MRI images,’’ in Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries (Lecture Notes in Computer Science), vol. 12962. Cham, Switzerland: Springer, 2022, pp. 272–284, doi: 10.1007/978-3-031-08999-2\_22.

[11] D. Nie, L. Wang, Y. Gao, and D. Shen, ‘‘Fully convolutional networks for multi-modality isointense infant brain image segmentation,’’ in Proc. IEEE 13th Int. Symp. Biomed. Imag. (ISBI), 2016, pp. 1342–1345, doi: 10.1109/ISBI.2016.7493515.

[12] F. Fang, Y. Yao, T. Zhou, G. Xie, and J. Lu, ‘‘Self-supervised multimodal hybrid fusion network for brain tumor segmentation,’’ IEEE J. Biomed. Health Inform., vol. 26, no. 11, pp. 5310–5320, Nov. 2022, doi: 10.1109/JBHI.2021.3109301.

[13] Z. Wang and Y. Hong, ‘‘A2FSeg: Adaptive multi-modal fusion network for medical image segmentation,’’ in Proc. Int. Conf. Med. Image Comput. Comput.-Assist. Intervent. (MICCAI), 2023, pp. 673–681, doi: 10.1007/978-3-031-43901-8\_64.

[14] M. Havaei, N. Guizard, N. Chapados, and Y. Bengio, ‘‘HeMIS: Heteromodal image segmentation,’’ in Proc. Int. Conf. Med. Image Comput. Comput.-Assist. Intervent. (MICCAI), 2016, pp. 469–477, doi: 10.1007/978-3-319-46723-8\_54.

[15] C. Chen, Q. Dou, Y. Jin, H. Chen, J. Qin, and P.-A. Heng, ‘‘Robust multimodal brain tumor segmentation via feature disentanglement and gated fusion,’’ in Proc. Int. Conf. Med. Image Comput. Comput.-Assist. Intervent. (MICCAI), 2019, pp. 447–456, doi: 10.1007/978-3-030-32248- 9\_50.

[16] Y. Qiu, K. Jiang, H. Yao, Z. Wang, and S. Satoh, ‘‘Does adding a modality really make positive impacts in incomplete multi-modal brain tumor segmentation?,’’ IEEE Trans. Med. Imag., vol. 44, no. 5, pp. 2194–2205, May 2025, doi: 10.1109/TMI.2025.3526818.

[17] J. Wu et al., ‘‘Medical SAM adapter: Adapting segment anything model for medical image segmentation,’’ Med. Image Anal., vol. 102, Art. no. 103547, May 2025, doi: 10.1016/j.media.2025.103547.

[18] J. Hu, L. Shen, and G. Sun, ‘‘Squeeze-and-excitation networks,’’ in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018, pp. 7132– 7141, doi: 10.1109/CVPR.2018.00745.

[19] B. H. Menze et al., ‘‘The multimodal brain tumor image segmentation benchmark (BRATS),’’ IEEE Trans. Med. Imag., vol. 34, no. 10, pp. 1993–2024, Oct. 2015, doi: 10.1109/TMI.2014.2377694.

[20] S. Bakas et al., ‘‘Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features,’’ Sci. Data, vol. 4, Art. no. 170117, Sep. 2017, doi: 10.1038/sdata.2017.117.

[21] ‘‘BraTS 2023 Challenge,’’ Synapse, 2023. Accessed: Oct. 4, 2026. [Online]. Available: https://www.synapse.org/Synapse:syn51156910/wiki/ 622351

[22] A. Tragakis, C. Kaul, R. Murray-Smith, and D. Husmeier, ‘‘The fully convolutional transformer for medical image segmentation,’’ in Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV), 2023, pp. 3649– 3658, doi: 10.1109/WACV56688.2023.00365.

[23] L. Wang, F. Liang, Y. Li, H. Zhang, W. Ouyang, and J. Shao, ‘‘RePre: Improving self-supervised vision transformer with reconstructive pretraining,’’ in Proc. 31st Int. Joint Conf. Artif. Intell. (IJCAI), 2022, pp. 1437–1443, doi: 10.24963/IJCAI.2022/200.

[24] E. J. Hu et al., ‘‘LoRA: Low-rank adaptation of large language models,’ in Proc. Int. Conf. Learn. Representations (ICLR), 2022. [Online]. Avail able: https://openreview.net/forum?id=nZeVKeeFYf9

[25] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, ‘‘A simple framework for contrastive learning of visual representations,’’ in Proc. 37th Int. Conf. Mach. Learn. (ICML) (Proceedings of Machine Learning Research), vol. 119, 2020, pp. 1597–1607.

[26] R. Saluja, ‘‘BraTS 2023 lesion-wise performance metrics for segmentation challenges,’’ GitHub repository. Accessed: Oct. 4, 2026. [Online]. Available: https://github.com/rachitsaluja/BraTS-2023-Metrics

[27] R. Saluja, J. Rudie, and U. Baid, ‘‘BraTS 2023 performance evaluation and ranking,’’ Synapse discussion. Accessed: Oct. 4, 2026. [Online]. Available: https://www.synapse.org/Synapse: syn51156910/discussion/threadId=10357

[28] S. Zheng et al., ‘‘Asymmetric adaptive heterogeneous network for multimodality medical image segmentation,’’ IEEE Trans. Med. Imag., vol. 44, no. 4, pp. 1836–1852, Apr. 2025, doi: 10.1109/TMI.2025.3526604.

[29] A. Ferreira et al., ‘‘How we won BraTS 2023 adult glioma challenge? Just faking it! Enhanced synthetic data augmentation and model ensemble for brain tumour segmentation,’’ arXiv:2402.17317, 2024, doi: 10.48550/arXiv.2402.17317.

[30] F. Maani, A. U. R. Hashmi, M. Aljuboory, N. Saeed, I. Sobirov, and M. Yaqub, ‘‘Advanced tumor segmentation in medical imaging: An ensemble approach for BraTS 2023 adult glioma and pediatric tumor tasks,’’ in Brain Tumor Segmentation, and Cross-Modality Domain Adap tation for Medical Image Segmentation (Lecture Notes in Computer Science), vol. 14669. Cham, Switzerland: Springer, 2024, pp. 264–277, doi: 10.1007/978-3-031-76163-8\_24.