# RLDF: A Region-Level Deepfake Dataset and Benchmark for Partial-Face Manipulation Detection

Shukesh Reddy Abhijit Das Machine Intelligence Group, Department of CS&IS Birla Institute of Technology and Science, Pilani, Hyderabad Campus, India

## Abstract

Existing deepfake detection models are built upon manipulated images that are tampered with either exclusively fully swapped or fully synthesised faces, which leads us to an open question: Can off-the-shelf deepfake models detect manipulation effectively while tampering is confned to a local facial sub-region? To fill this gap, we introduce RLDF (Region-Level Deepfake), the first dataset for region-level deepfake evaluation. Employing the proposed region-aware partial swap synthesis model, we generate over 2M partial face-swap images by using eight face-swapping methods. Altogether, we considered seven face regions, among which there are four discrete regions (eye, nose, mouth and eyebrow) and four composite facial regions (eye and nose; eye, nose and mouth; eye, nose, mouth and eyebrow; and full face). In addition, we define three evaluation protocols and benchmark 12 state-of-the-art detectors across the five popular standard cross-dataset evaluations, DF40, and the proposed RLDF, providing a comprehensive audit of where and why existing detectors underperformed. Our evaluation reveals that existing detectors fail to generalize consistently across partial facial regions, resulting in an overall drop of 10.5% AUC and 10% accuracy under region-level evaluation. Further, we propose a mask-guided cross-attention model as a baseline approach that can effectively detect region-level manipulation. The proposed dataset, code, and checkpoints are available at https://github.com/RLDF-Dataset/RLDF.

## 1. Introduction

Rapid advancements in the use of deep generative [16, 24], diffusion [21, 50], and Large Language Models (LLMs) [1, 4], have dramatically lowered the barrier towords generating content creation across image, video, and audio domains. This has enabled the generation of highly realistic synthetic media, which is often indistinguishable from real content. Face manipulation, in particular, has seen rapid growth through various approaches, including face swapping [7, 28, 31, 42, 51, 61, 63], face reenactment [45, 49, 62, 71], facial attribute editing [9, 10, 43, 59], and fully synthetic face generation [6, 25, 44, 50]. The misuse of such techniques for identity fraud, misinformation, and non-consensual content has made deepfake detection an active research problem.

To mitigate these, numerous deepfake detection methods have been proposed [5, 32, 47, 52, 64] which are broadly categorized into CNN-based classifiers [2, 41, 52], spatialdomain detectors [32, 54, 60, 64], and frequency-domain methods [36, 39, 47]. More recently, Vision Transformer and CLIP-based detectors [8, 19, 37, 66] have leveraged large-scale pretrained representations to improve generalization. Although these methods exhibit strong in-domain performance, they demonstrate a significant drop when evaluated in a cross domain/cross-dataset settings [65, 67], as they tend to overfit to dataset-specific artifacts rather than learning the underlying manipulation patterns.

In this direction, several benchmark datasets have been introduced, including FaceForensics++ [52], CelebDF [35], DFDC [13], DFDCP [15], and most recently DF40 [67]. For example, DF40 aggregates 40 distinct generation methods across face swapping, reenactment, full synthesis, and editing. However, all existing datasets share a common limitation that the tampering is committed at the global image level with a binary label of annotation, i.e., indicating whether the image is manipulated or not. Hence, existing deepfake detectors [8, 32, 37, 66, 72] trained based on the above-mentioned datasets attempt to learn a global representation to distinguish whether an image is fake or real. Therefore, it can be assumed that the representation of these detectors remains unaware of explicit quantified spatial manipulated cues. Consequently, when manipulation is confined to a specific iota of the region, the existing models struggle to detect the manipulation. To the best of our knowledge, no existing work or datasets have studied deepfake detection in which manipulation is performed at the distinct face region. This leaves a critical and unexplored gap in the literature. To address this gap, we introduce RLDF, the first dataset designed for partial or region-level deepfake evaluation. By leveraging existing full face-swapping methods [7, 18, 38, 42, 51, 55, 61, 63]. We proposed a regionaware partial swap model that can manipulate specific region such as the eye, nose, mouth, and eyebrow and four composite regions that progressively expand in spatial coverage from paired regions up to full-face replacement (Section 3). RLDF comprises over 2M, 8 face-swapping methods, and 8 facial regions. Our specific contributions are as follows: (1) The first region-level deepfake dataset with partial faceswap annotations. (2) Benchmarking across three evaluation protocols including 21 manipulation methods from existing datasets, and RLDF region-level evaluation, rebelling that performance of existing detectors drop significantly under partial-face conditions. (3) A mask-guided cross-attention baseline that sets a starting point for future work, confirming that patch-level forgery grounding is a promising direction for region-level manipulation detection.

Table 1. Comparison of existing deepfake datasets. All prior datasets annotate manipulation with a binary image-level label and contain only full-face swaps. RLDF is the first benchmark to provide region-level annotations for partially manipulated faces. Region Labels: per-region GT labels across 8 sub-facial areas. Localization GT: spatial ground-truth for forgery localization. Partial-Face: manipulation on sub-regions only.
<table><tr><td>Dataset</td><td>Pub.</td><td>#Methods</td><td>Fake Images</td><td>Region Labels</td><td>Localization GT</td><td>Partial-Face</td></tr><tr><td>DF-TIMIT [27]</td><td>ArXiv&#x27;18</td><td>2</td><td>一</td><td>x</td><td>X</td><td>x</td></tr><tr><td>UADFV [68]</td><td>ICASSP’19</td><td>1</td><td>252</td><td>x</td><td>x</td><td>x</td></tr><tr><td>FaceForensics++ [52]</td><td>ICCV&#x27;19</td><td>4</td><td></td><td>x</td><td>x</td><td>x</td></tr><tr><td>DeepFakeDetection [17]</td><td></td><td>5</td><td></td><td>x</td><td>x</td><td>x</td></tr><tr><td>CDF [35]</td><td>CVPR&#x27;20</td><td>1</td><td></td><td>x</td><td>x</td><td>x</td></tr><tr><td>DFFD [12]</td><td>CVPR&#x27;20</td><td>7</td><td>0.2M+</td><td>x</td><td>x</td><td>x</td></tr><tr><td>DeeperForensics [22]</td><td>CVPR&#x27;20</td><td>1</td><td></td><td>x</td><td>x</td><td>x</td></tr><tr><td>DFDC [13]</td><td>ArXiv&#x27;20</td><td>8</td><td></td><td>x</td><td>x</td><td>x</td></tr><tr><td>ForgeryNet [20]</td><td>CVPR&#x27;21</td><td>15</td><td>1M+</td><td>X</td><td>x</td><td>x</td></tr><tr><td>FakeAVCeleb [26]</td><td>NeurIPS&#x27;21</td><td>4</td><td></td><td>x</td><td>x</td><td>x</td></tr><tr><td>KoDF [29]</td><td>ICCV&#x27;21</td><td>6</td><td></td><td>x</td><td>X</td><td>X</td></tr><tr><td>FFIW [73]</td><td>CVPR&#x27;21</td><td>3</td><td></td><td>X</td><td>x</td><td>X</td></tr><tr><td>DF3 [23]</td><td>TMM&#x27;22</td><td>6</td><td></td><td>x</td><td>x</td><td>x</td></tr><tr><td>DeepFakeFace [56]</td><td>ArXiv&#x27;23</td><td>3</td><td>90K</td><td>x</td><td>x</td><td>x</td></tr><tr><td>DF-Platter [40]</td><td>CVPR&#x27;23</td><td>3</td><td></td><td>x</td><td>x</td><td>x</td></tr><tr><td>DiffusionDeepfake [3]</td><td>ArXiv&#x27;24</td><td>2</td><td>0.1M+</td><td>x</td><td>X</td><td>X</td></tr><tr><td>DF40 [67]</td><td>NeurIPS&#x27;24</td><td>40</td><td>1M+</td><td>x</td><td>x</td><td>x</td></tr><tr><td>RLDF (Ours)</td><td></td><td>8</td><td>2M+</td><td></td><td>v</td><td></td></tr></table>

## 2. Related Work

Deepfake Detection: Seminal deepfake detectors explores physiological signals [34, 69] and generation artefacts in spatial and frequency domains [33, 47]. Following works in the literature paved a complementary line of research by designing attention mechanisms, contrastive objectives, and feature-disentanglement strategies to learn more discriminative representations [12, 64]. More recently, Vision Transformer and CLIP-based features [8, 19, 37, 66] are employed to improve cross-dataset generalization [30, 32, 54]. Further, pseudo-fake synthesis methods are proposed which simulate blending artefacts at training time to reduce overfitting to seen forgeries [65, 67]. Despite these advances, all existing detectors are designed and evaluated on fully swapped or fully synthesized faces. When subjected to region-level evaluation, where manipulation is confined to a specific facial sub-region, performance drops sharply across all methods, a failure mode that standard benchmarks cannot reveal.

Deepfake Datasets: Several Deepfake datasets have been proposed in the literature, such as UADFV [68], DFTIMIT [27], FaceForensics++ [52], Celeb-DF [35], DFDC [13], DeeperForensics [22], and DF40 [67]. Yet across all of these datasets, manipulation is annotated at the image-level or video label. Moreover, the manipulation is performed considering the full face. There exists no dataset that provides region-level annotations, partial-face samples, or spatial ground-truth (Table 1). We proposed RLDF to mitigate the above gaps. Evaluating existing detectors on RLDF reveals a consistent drop in performance. AUC of FatFormer [37] and DFD-FCG [19] drop to 0.745 and 0.806 respectively on RLDF, a degradation of up to 10% AUC from their DF40 scores. (Refer to Section 4 for detailed evaluation protocol).

## 3. Methodology

We present a unified contribution addressing region-level deepfake detection through two tightly coupled components: a large-scale benchmark that systematically exposes how existing detectors fail under partial-face manipulation, and a mask-guided detection framework.

## 3.1. RLDF Dataset

Let $\varphi : \mathcal { T } _ { s } \times \mathcal { T } _ { t } \to \mathcal { T }$ denote a face-swapping method that transfers the identity of a source image $\mathbf { \bar { \Gamma } } _ { \bar { I _ { s } } } \in \mathbf { \mathbb { R } ^ { 3 \times H \times W } }$ to a target image $I _ { t } \in \mathbf { \bar { \mathbb { R } } ^ { 3 \times H \times W } }$ , producing a complete faceswap output $\hat { \mathbf { I } } _ { f } = \varphi ( I _ { s } , I _ { t } )$ that preserves the target's pose, expression, and background while replacing the facial identity. We consider a diverse set of eight state-of-the-art facemanipulation methods: Φ = BlendFace, E4S, FaceDancer, FSGAN, GHOST, InSwapper, SimSwap, UniFace. These are based on GAN-based, latent-space, reenactment, and diffusion-based generation paradigms. No method in Φ supports partial manipulation; all operate exclusively on the full facial region. Modifying these methods for localized manipulation is infeasible at scale across eight architecturally distinct systems. We therefore introduce a region-aware partial swap synthesis that composites only the designated facial region from $\hat { \mathbf { I } } _ { f }$ back onto $I _ { t }$ using a semantically derived soft mask, producing partial fakes without any modification to $\varphi .$ Figure 1 illustrates the pipeline and Algorithm 1 formalizes the complete procedure.

Algorithm 1 Region-Aware Partial Swap Synthesis   
Require: Video pairs $\{ ( v _ { s } ^ { ( k ) } , v _ { t } ^ { ( k ) } ) \} _ { k = 1 } ^ { 1 0 0 0 }$ , swap methods Φ, re  
gions R, frames per pair $T = 3 2$   
Ensure: Dataset DRLDF with tuples $\left( \mathbf { I } _ { i } , r _ { i } , y _ { i } \right)$   
1: for each pair $( v _ { s } , v _ { t } )$ do   
2: Sample T frames via linspace, detect and align [14], crop   
to 224×224   
3: for each target frame $I _ { t }$ do   
4: Add $( I _ { t } ,$ ful1, 0) to $\mathcal { D } _ { \mathrm { R L D F } }$ ▷ y = 0: real   
5: end for   
6: for each method $\varphi \in \Phi$ do   
7: Generate full swap: $\hat { \mathbf { I } } _ { f } = \varphi ( I _ { s } , I _ { t } )$   
8: Parse $I _ { t }$ via BiSeNetV2, dilate per region, Gaussian   
smooth $( \sigma { = } 3 . 0 ) \to \mathbf { M } _ { \tau }$   
9: for each region $r \in$ R do   
10: $\mathbf { I } _ { p } ^ { ( r ) }  \mathbf { M } _ { r } \odot \hat { \mathbf { I } } _ { f } + ( 1 - \mathbf { M } _ { r } ) \odot I _ { t }$   
11: Add $( \mathbf { I } _ { p } ^ { ( r ) } , r , 1 )$ to DRLDF y = 1: fake   
12: end for   
13: end for   
14: end for   
15: Partition into train/val/test (720/140/140 pairs, no identity over  
lap)

From the FaceForensics++ corpus [52], we form 1,000 source-target pairs $( v _ { s } , v _ { t } )$ and sample $T \ = \ 3 2$ temporally aligned target frames via linspace indexing: $t _ { j } ~ =$ $\left\lfloor j ( F _ { t } - 1 ) / ( T - 1 ) \right\rfloor$ , where $F _ { t }$ is the total frame count. For each frame, InsightFace Antelope [14] detects the primary face at $6 4 0 \times 6 4 0$ resolution and produces a $2 2 4 \times 2 2 4$ similarity-transformed aligned crop normalizing pose and scale. Frames with no detected face $( < 3 \% )$ are discarded. The full swap is then generated per Eq. (1), and all mask extraction and blending operate on this aligned representation.

$$
\hat { \mathrm { \bf ~ I } } _ { f } = \varphi ( I _ { s } , I _ { t } ) , \qquad \varphi \in \Phi .\tag{1}
$$

For each target frame It, region masks $\mathbf { M } _ { r } \in [ 0 , 1 ] ^ { H \times W }$ are obtained via BiSeNetV2 [70] facial parsing, followed by morphological dilation to compensate for undersegmentation in small regions (eyes, eyebrow, nose, mouth) and Gaussian smoothing $\left( \sigma { = } 3 . 0 \right)$ to produce natural soft boundaries. Critically, masks are parsed from the unmodified target $I _ { t } ,$ so region boundaries follow real facial anatomy rather than the swap-induced artifacts present in $\hat { \mathbf { I } } _ { f }$ . The partial fake is then synthesized as:

$$
\mathbf { I } _ { p } ^ { ( r ) } = \mathbf { M } _ { r } \odot \hat { \mathbf { I } } _ { f } ~ + ~ \left( 1 - \mathbf { M } _ { r } \right) \odot I _ { t } ,\tag{2}
$$

where pixels inside the region (Mr ≈ 1) carry swap artifacts and pixels outside $( \mathbf { M } _ { r } \approx 0 )$ are drawn exactly from $I _ { t } .$ Detection is thus strictly harder than whole-face forgery: the model must localize artifacts within a semantically confined sub-region while ignoring the surrounding authentic context. The eight regions R comprise four individual regions (eye, eyebrow, nose, mouth) and four composite regions of progressively increasing spatial coverage: Composite-1 (eye + nose), Composite-2 (eye + nose + mouth), and Composite-3 (eye + nose + mouth + eyebrow), plus Full Face as an upper bound, yielding eight manipulation in total.

Applying Eq. (2) across all |Φ|=8 methods, |R|=8 regions, $T { = } 3 2$ frames, and 1,000 pairs yields $1 , 0 0 0 \times 3 2 \times$ 64 = 2,048,000 fake images, balanced with an equal number of unmodified target frames as real samples (y=0).

## 3.2. Mask-Guided Cross-Attention Framework

Existing detectors predict a binary label $y \in \{ 0 , 1 \}$ from the entire image of the face, without knowing where the manipulation occurred on the face. This works for full face fake images, but fails for region-level fakes where the forged area is small. To address this, we propose MCA that predicts both a binary label and a spatial forgery map:

$$
f : \mathbf { I } \longrightarrow ( \widehat { y } \in [ 0 , 1 ] , \widehat { \mathbf { m } } \in [ 0 , 1 ] ^ { N } ) ,\tag{3}
$$

where $\hat { m } _ { i } \approx 1$ means patch i is likely forged. The map î is not a standalone output it guides a cross-attention module that builds a forgery-focused feature, which is then combined with a global feature for the final prediction. Figure 2 shows an overview.

Given an image $\mathbf { I } ~ \in ~ \mathbb { R } ^ { 3 \times 2 2 4 \times 2 2 4 }$ , we use CLIP ViT-L/14 [48] to extract $N { = } 2 5 6$ patch tokens $\textbf { P } \in \mathbb { R } ^ { N \times D }$ (D=1024) from a 16×16 grid. The first 12 of 24 transformer blocks are frozen. The last 12 are fine-tuned with $\mathrm { l r } _ { \mathrm { e n c } } = 1 0 ^ { - 6 }$

Forgery MaskNet: MaskNet assigns a forgery score to each of the N=256 patches independently. A lightweight three-layer MLP processes each patch token and outputs a probability $\hat { m } _ { i } \in [ 0 , 1 ]$

$$
\begin{array} { r } { \hat { \mathbf { m } } = \sigma \left( \mathbf { M L P } \big ( \mathbf { L N } ( \mathbf { P } ) \big ) \right) , \quad \quad } \\ { \mathbb { R } ^ { 1 0 2 4 } \xrightarrow { \mathrm { G E L U } } \mathbb { R } ^ { 5 1 2 } \xrightarrow { \mathrm { G E L U } } \mathbb { R } ^ { 2 5 6 } \xrightarrow { \sigma } \mathbb { R } ^ { 1 } . } \end{array}\tag{4}
$$

Each patch is scored independently, without looking at neighbouring patches. This is intentional, as sharing information between patches can cause the mask to collapse to a trivial all-zero or all-one output. The scores are learned entirely from binary real/fake labels, without any spatial annotations. Masked Cross-Attention (MCA): A standard mean-pooled feature treats all patches equally, irrespective of whether they belong to a forged or authentic region. To focus on the forged area, we introduce MCA, which builds a feature $\mathbf { F } _ { f }$ that attends selectively to suspicious patches.

![](images/b24fd677e10aa56e6f393364051643266f1c8d53a92358cbde38583f23f35a52.jpg)  
P = Partial Fake S = Swapped Full Face (from Face Swap Model) M = Soft Region Mask (from BiSeNetV2) T = Target Face

Figure 1. Overview of the proposed region-aware partial swap synthesis: Given source-target video pairs, (1) aligned face crops are extracted and passed through eight face-swapping methods to generate full face-swaps, (2) a BiSeNetV2 facial parser derives soft region masks from the target frame, and (3) the designated facial region from the full swap is composited back onto the target image, producing partial fakes across eight facial regions. Representative samples for every region and swap method are provided in Appendix F  
![](images/9ab5350bca38b54442f7aedfd1a2a74e5c0a27e9351bcd7a6a9b242ebfa33fc8.jpg)  
Figure 2. Overview of the proposed mask-guided cross-attention framework. A partially fine-tuned CLIP ViT-L/14 encoder extracts N=256 patch tokens. Forgery MaskNet assigns each patch a forgery score $\hat { m } _ { i } \in [ 0 , 1 ]$ . Masked Cross-Attention uses these scores as attention biases so the model focuses on suspicious patches. A global feature (mean-pooled) and the region-focused feature are concatenated and passed to a classifier.

We use a single learnable query vector $\mathbf { q } \in \mathbb { R } ^ { 1 \times D }$ that represents what a forged region looks like. It is projected to $\mathbf { Q } = \mathbf { W } _ { Q } \mathbf { q } .$ , and all patch tokens form keys ${ \bf K } = { \bf W } _ { K } { \bf P }$ and values $\dot { \mathbf { V } } = \mathbf { W } _ { V } \mathbf { P }$ . The forgery scores m from MaskNet are added to the attention logits as a log-space bias, yielding $\mathbf { F } _ { f }$ with d=128 (8 heads) and $\epsilon { = } 1 0 ^ { - 6 }$ defined in Eq. (5):

$$
\mathbf { F } _ { f } = \mathrm { L N } \Bigg ( \mathbf { W } _ { O } \cdot \mathrm { S o f t m a x } \Bigg ( \frac { \mathbf { Q } \mathbf { K } ^ { \top } } { \sqrt { d } } + \log ( \hat { \mathbf { m } } + \epsilon ) \Bigg ) \cdot \mathbf { V } \Bigg ) ~ .\tag{5}
$$

The bias acts as a soft mask over the attention logits, where a forged patch $( \hat { m } _ { i } \approx 1 )$ adds ≈ 0 and is included normally, while an authentic patch $( \hat { m } _ { i } \approx 0 )$ adds —∞ and is suppressed. As a result, $\mathbf { F } _ { f }$ captures only the forged regions. This soft mask is fully differentiable, so MaskNet and MCA improve each other during training without any spatial supervision. On real images, all scores are near zero, so $\mathbf { F } _ { f } \approx \mathbf { F } _ { g }$ and no false forgery signal is produced.

Dual-Path Fusion and Classification: We compute a global feature $\begin{array} { r } { \mathbf { F } _ { g } = \frac { 1 } { N } \sum _ { i } \mathbf { P } _ { i } } \end{array}$ by averaging all patch tokens. This captures the overall global feature from the input image. We concatenate $\mathbf { F } _ { g }$ and $\mathbf { F } _ { f }$ and pass the combined feature to a classifier C:

$$
\begin{array} { r l } & { \hat { \boldsymbol { y } } = \sigma \big ( \mathrm { c l a m p } \big ( \mathcal { C } ( [ \mathbf { F } _ { g } ; \mathbf { F } _ { f } ] ) , - 1 5 , 1 5 \big ) \big ) , } \\ & { \mathbb { R } ^ { 2 0 4 8 } \xrightarrow { \mathrm { G E L U } } \mathbb { R } ^ { 1 0 2 4 } \xrightarrow { \mathrm { G E L U } } \mathbb { R } ^ { 5 1 2 } } \\ & { \qquad \xrightarrow { \mathrm { G E L U } } \mathbb { R } ^ { 2 5 6 } \xrightarrow { \sigma } \mathbb { R } ^ { 1 } , } \end{array}\tag{6}
$$

where C is a four-layer MLP with LayerNorm, GELU, and Dropout(0.3). Concatenation keeps both signals separate: $\mathbf { F } _ { g }$ provides full face context, $\mathbf { F } _ { f }$ provides localised forgery feature.

Training Stability: At the start of training, MaskNet scores are random, which makes $\mathbf { F } _ { f }$ noisy and unstable.

To handle this, we blend $\mathbf { F } _ { f }$ toward $\mathbf { F } _ { g }$ during an early warmup phase: ${ \tilde { \bf F } } _ { f } ~ = ~ w { \bf F } _ { g } + \left( 1 { - } w \right) { \bf F } _ { f }$ , where $w =$ $\operatorname* { m a x } ( 1 - e / T _ { w } , ~ 1 - 0 . 6 7 )$ decreases over ${ T _ { w } } { = } 3$ epochs. After warmup, $\mathbf { F } _ { f }$ contributes up to 67% of the fused signal. This blending applies only during training. We also maintain an exponential moving average $( \mathrm { E M A } , \rho { = } 0 . 9 9 9 )$ of all weights and use the EMA model for validation.

Loss Function: The total loss combines binary classification with three mask regularisation terms:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { c l s } } + 0 . 1 \mathcal { L } _ { \mathrm { e n t } } + 0 . 5 \mathcal { L } _ { \mathrm { r e a l } } + 0 . 3 \mathcal { L } _ { \mathrm { f a k e } } .\tag{7}
$$

$\mathcal { L } _ { \mathrm { c l s } }$ is binary cross-entropy with label smoothing $( \alpha { = } 0 . 1 )$ $\mathcal { L } _ { \mathrm { e n t } }$ penalises masks that are all-zero or all-one, keeping them informative. $\mathcal { L } _ { \mathrm { r e a l } }$ pushes mask scores toward zero on real images. $\mathcal { L } _ { \mathrm { f a k e } }$ pushes them toward one on fake images. Without the last two terms, the mask collapses and $\mathbf { F } _ { f }$ degenerates to a copy of $\mathbf { F } _ { g }$

## 4. Benchmarking and Findings

Implementation Details: All models process 224×224 crops (InsightFace Antelope alignment [14]). Backbone: CLIP ViT-L/14 [48], N=256 patch tokens, D=1024; first 12 of 24 blocks frozen, last 12 fine-tuned $( \eta _ { \mathrm { e n c } } { = } 1 0 ^ { - 6 } )$ Task heads use $\eta _ { \mathrm { h e a d s } } { = } 5 { \times } 1 0 ^ { - 5 }$ , AdamW, cosine schedule $( T _ { \mathrm { m a x } } { = } 3 0 , \eta _ { \mathrm { m i n } } { = } 1 0 ^ { - 7 } )$ , weight decay 0.05, gradient clip 1.0, batch 64, single NVIDIA A100. Loss: $\lambda _ { \mathrm { e n t } } \mathrm { = } 0 . 1$ $\lambda _ { \mathrm { { r e a l } } } { = } 0 . 5 , \lambda _ { \mathrm { { f a k e } } } { = } 0 . 3$ , label smoothing α=0.1. All detectors are trained exclusively on FF++ (C23) [52] (720/140/140 split) and tested with the official checkpoint provided in the paper, with no additional data or tuning. We evaluate on three protocols: P#1: five classical benchmarks (CelebDFv1/v2, DFDC, DFDCP, UADFV); P#2: eight DF40 generation methods [67]; P#3: our RLDF test set (\~286K images, 8 face-swap methods × 8 regions). All cells report ACC/ AUC; weighted means are image-count weighted.

Protocol Results. P#1 (Table 2): Our 12-Block achieves the highest weighted mean AUC (0.86, ACC 0.78), outperforming all baselines including ForAda [11], which leads on DFDC and DFDCP individually (AUC 0.84/0.89) but ranks second overall at 0.87. Frequency-domain detectors (LGrad, FreqNet) collapse to near-random performance (AUC ≈0.50), confirming that artefacts introduced by FF++ C23 compression are dataset-specific and do not transfer across benchmarks.

P#2 (Table 3): Our 12-Block leads on weighted ACC (0.80), with particularly strong AUC on DeepFaceLab (0.97) and HeyGen (0.90) generation methods that produce localised blending boundaries well-suited to MCA's patchlevel attention. Mid-Journey6 proves universally challenging (AUC 0.55), as it synthesises faces from scratch without any identity transfer, leaving no spatial forgery cues for detection.

P#3 (Table 4): Unlike P#1 and P#2 where the entire face is manipulated, P#3 restricts forgery to individual facial subregions, which makes the detection task progressively harder as the manipulated area shrinks. Our 8-Block leads all methods (ACC 0.81, AUC 0.87). Notably, it outperforms the 12- Block, in contrast to P#1 and P#2, revealing a fundamental trade-off in encoder fine-tuning depth. The 12-Blocks, finetuned more aggressively on full face FF++ forgeries, learns features biased toward holistic manipulation patterns and loses sensitivity to localized patch-level cues. The 8-Blocks, with the, majority of CLIP layers frozen, retains more general spatial representations that generalize better to partialface manipulation. Across methods, E4S is the most detectable (AUC ≥0.96) owing to its pronounced style-transfer boundaries, while BlendFace is the hardest (AUC 0.85) as its latent-space blending produces minimal visible seam artifacts. Table 5 consolidates per-protocol weighted means, extended analysis including ROC curves, AUC heatmaps, confusion matrices, t-SNE projections, MaskNet activation maps are provided in Appendices G and H.

Finding 1: No existing method generalises across all three protocols. ProDet [8] (AUC 0.89 on P#3) collapses to 0.57 on P#2, FatFormer [37] (0.85 on P#2) drops to 0.74 on P#3, ForAda [11] leads P#1 (0.87) but falls to 0.73 on P#2. Among all evaluated detectors, our 12-Block baseline consistently maintains strong performance across all three protocols (mean ACC 0.78, AUC 0.85), suggesting that explicit spatial grounding via MaskNet and MCA offers a generalization advantage orthogonal to encoder pre-training and frequency analysis.

Finding 2: The three protocols are orthogonal. Rank ordering of methods differs substantially across protocols: strong cross-dataset performance on P#1 does not predict region-level sensitivity on P#3, confirming that RLDF probes a capability not captured by any existing benchmark.

Finding 3: Region-level evaluation reveals systematic failure. Detectors that score above 0.82 AUC on P#1 drop by up to 10.5% when evaluated on P#3, validating the RLDF benchmark's role in exposing a critical and previously invisible generalization gap.

Overall, existing detectors fail to generalize consistently across partial facial regions, with an average drop of 10.5% AUC and 10% accuracy under region-level evaluation. This degradation is not uniform, it varies across swap methods and facial regions, with some region method pairs proving significantly harder than others. A full per-region and permethod breakdown is provided in Appendices G and H.

## 4.1. Ablation Study

Table 6 evaluates each component by removing it while keeping the rest fixed. Removing mask guidance or the forged path $\mathbf { F } _ { f }$ has negligible impact on P#1 (≤0.001 ACC) but causes the largest drops on P#3 (0.155 and 0.157 ACC), confirming that detecting region-level forgery demands spatial localization, not just a global feature. Removing the global path $\mathbf { F } _ { g }$ hurts P#2 most (0.061 AUC), where generators like

![](images/423089798d350f0d4ded0f7d1f8712edfb358cee88eb4f6284c485aac343bb1a.jpg)  
Figure 3. t-SNE visualization of penultimate features on the RLDF test set (Protocol #3). Blue = real, red = fake.

Table 2. Protocol #1 (Cross-Dataset, Classic Deepfake Benchmarks): Frame-level performance. Models are trained on FF++ and tested on five widely-used deepfake benchmarks. Bold = best; underline = second-best (per metric, per column). Weighted mean is image-count weighted across five datasets (total 297,914 images; DFDC+DFDCP account for 92.5% of weight).
<table><tr><td colspan="2">Detector Venue</td><td colspan="2">CelebDF-v1</td><td colspan="2">CelebDF-v2</td><td colspan="2">DFDC</td><td colspan="2">DFDCP</td><td colspan="2">UADFV</td><td colspan="2">Wtd. Mean</td></tr><tr><td></td><td></td><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td></tr><tr><td>Xception [52]</td><td>ICCV&#x27;19</td><td>0.69</td><td>0.77</td><td>0.71</td><td>0.74</td><td>0.64</td><td>0.71</td><td>0.68</td><td>0.73</td><td>0.79</td><td>0.94</td><td>0.67</td><td>0.73</td></tr><tr><td>F3Net [46]</td><td>ECCV’20</td><td>0.68</td><td>0.75</td><td>0.61</td><td>0.73</td><td>0.63</td><td>0.70</td><td>0.64</td><td>0.70</td><td>0.81</td><td>0.89</td><td>0.63</td><td>0.71</td></tr><tr><td>RECCÈ [5]</td><td>CVPR&#x27;22</td><td>0.69</td><td>0.75</td><td>0.70</td><td>0.75</td><td>0.62</td><td>0.68</td><td>0.66</td><td>0.70</td><td>0.78</td><td>0.89</td><td>0.64</td><td>0.70</td></tr><tr><td>SBI [53]</td><td>CVPR’22</td><td>0.75</td><td>0.81</td><td>0.71</td><td>0.77</td><td>0.62</td><td>0.71</td><td>0.74</td><td>0.83</td><td>0.84</td><td>0.94</td><td>0.69</td><td>0.77</td></tr><tr><td>LGrad [57]</td><td>CVPR&#x27;23</td><td>0.39</td><td>0.54</td><td>0.35</td><td>0.53</td><td>0.48</td><td>0.50</td><td>0.35</td><td>0.50</td><td>0.50</td><td>0.51</td><td>0.41</td><td>0.50</td></tr><tr><td>UCF [64]</td><td>ICCV’23</td><td>0.72</td><td>0.81</td><td>0.66</td><td>0.77</td><td>0.66</td><td>0.73</td><td>0.60</td><td>0.69</td><td>0.82</td><td>0.92</td><td>0.63</td><td>0.72</td></tr><tr><td>FatFormer [37]</td><td>CVPR&#x27;24</td><td>0.40</td><td>0.74</td><td>0.36</td><td>0.60</td><td>0.56</td><td>0.63</td><td>0.59</td><td>0.59</td><td>0.60</td><td>0.78</td><td>0.56</td><td>0.61</td></tr><tr><td>FreqNet [58]</td><td>AAAI&#x27;24</td><td>0.44</td><td>0.44</td><td>0.38</td><td>0.56</td><td>0.48</td><td>0.51</td><td>0.40</td><td>0.53</td><td>0.45</td><td>0.45</td><td>0.43</td><td>0.52</td></tr><tr><td>ProDet [8]</td><td>NeurIPS&#x27;24</td><td>0.77</td><td>0.86</td><td>0.77</td><td>0.83</td><td>0.64</td><td>0.70</td><td>0.72</td><td>0.77</td><td>0.82</td><td>0.95</td><td>0.69</td><td>0.75</td></tr><tr><td>DFD-FČG [19]</td><td>CVPR&#x27;25</td><td>0.70</td><td>0.89</td><td>0.74</td><td>0.87</td><td>0.65</td><td>0.80</td><td>0.73</td><td>0.83</td><td>0.66</td><td>0.98</td><td>0.69</td><td>0.82</td></tr><tr><td>Effort [66]</td><td>ICML&#x27;25</td><td>0.79</td><td>0.92</td><td>0.80</td><td>0.87</td><td>0.74</td><td>0.82</td><td>0.72</td><td>0.84</td><td>0.83</td><td>0.97</td><td>0.74</td><td>0.83</td></tr><tr><td>ForAda [11]</td><td>CVPR&#x27;25</td><td>0.82</td><td>0.91</td><td>0.82</td><td>0.90</td><td>0.75</td><td>0.84</td><td>0.73</td><td>0.89</td><td>0.89</td><td>0.98</td><td>0.75</td><td>0.87</td></tr><tr><td>Ours (8-Blocks)</td><td></td><td>0.83</td><td>0.92</td><td>0.80</td><td>0.89</td><td>0.73</td><td>0.81</td><td>0.74</td><td>0.81</td><td>0.89</td><td>0.98</td><td>0.74</td><td>0.82</td></tr><tr><td>Ours (12-Blocks)</td><td></td><td>0.84</td><td>0.93</td><td>0.84</td><td>0.92</td><td>0.75</td><td>0.84</td><td>0.79</td><td>0.86</td><td>0.90</td><td>0.98</td><td>0.78</td><td>0.86</td></tr></table>

![](images/c03702baf600494212831ccbb6c2e5525b1b4023a9e9d7668bede2d3cc34a0f0.jpg)  
(a) Protocol #1

![](images/b22054d99aa97cd872ec3dcfe305b7c3a25ef2230c367ba81914b99b6ef6828d.jpg)  
(b) Protocol #2

![](images/2383f19d66c7ae20ff1711ef695ed82ddf194b8c4d93cdca4515e83a97fc497b.jpg)  
(c) Protocol #3  
Figure 4. ROC curves across all three evaluation protocols. Our baseline (8-Blk, 12-Blk) achieves competitive AUC across all protocols (P#1: 0.906, P#2: 0.788, P#3: 0.88), while existing detectors show inconsistent performance under region-level evaluation.

Mind-Journey6 leave no local forgery cue and whole-face context is the only reliable signal. Replacing the soft mask with a hard binary threshold drops P#3 by 0.113 ACC, as binarisation severs the gradient between MCA and MaskNet, preventing them from jointly refining spatial localization. These results validate all four design choices, each targeting a distinct failure mode, and removing any one degrades 5erloimitations and FutureMork

RLDF is deliberately scoped to face swapping: all eight generators transfer identity, so region-level behaviour under reenactment, entire-face synthesis and attribute editing remains open. Partial fakes are produced by compositing a region of a full swap back onto the target through a smoothed parsing mask rather than by region-native generation, so the resulting boundaries reflect both the generator and the compositing step, and the region masks inherit any errors of the face parser. Evaluation is frame-level, leaving the temporal cues available in video unused by both the benchmark and the MCA baseline. The perceptual study is indicative rather than conclusive, covering 14 images rated by 37 non-

Table 3. Protocol #2 (Cross-Dataset, DF40 Test Set): Frame-level performance. Models are trained on FF++ and tested on eight generation methods from DF40. Bold = best; underline = second-best (per metric, per column). Weighted mean is image-count weighted across the eight sub-datasets (total 31,238 images).
<table><tr><td rowspan="2">Detector</td><td rowspan="2">Venue</td><td colspan="2">Deep- FaceLab</td><td colspan="2">Mid- Journey6</td><td colspan="2">Hey- Gen</td><td colspan="2">Star- GAN</td><td colspan="2">Star- GAN2</td><td colspan="2">Style- CLIP</td><td colspan="2">Collab- Diff</td><td colspan="2">Which- FaceReal</td><td colspan="2">Wtd. Mean</td></tr><tr><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td></tr><tr><td>Xception [52]</td><td>ICCV&#x27;19</td><td>0.66</td><td>0.72</td><td>0.62</td><td>0.57</td><td>0.79</td><td>0.77</td><td>0.57</td><td>0.62</td><td>0.48</td><td>0.42</td><td>0.65</td><td>0.78</td><td>0.69</td><td>0.75</td><td>0.47</td><td>0.45</td><td>0.65</td><td>0.67</td></tr><tr><td>F3Net [46]</td><td>ECCV&#x27;20</td><td>0.58</td><td>0.71</td><td>0.72</td><td>0.78</td><td>0.50</td><td>0.50</td><td>0.59</td><td>0.67</td><td>0.48</td><td>0.39</td><td>0.42</td><td>0.44</td><td>0.54</td><td>0.42</td><td>0.44</td><td>0.34</td><td>0.52</td><td>0.53</td></tr><tr><td>RECCE [5]</td><td>CVPR&#x27;22</td><td>0.57</td><td>0.64</td><td>0.68</td><td>0.72</td><td>0.71</td><td>0.55</td><td>0.55</td><td>0.61</td><td>0.48</td><td>0.38</td><td>0.50</td><td>0.61</td><td>0.60</td><td>0.56</td><td>0.47</td><td>0.40</td><td>0.59</td><td>0.56</td></tr><tr><td>SBI [53]</td><td>CVPR&#x27;22</td><td>0.72</td><td>0.80</td><td>0.55</td><td>0.57</td><td>0.67</td><td>0.71</td><td>0.71</td><td>0.81</td><td>0.52</td><td>0.52</td><td>0.67</td><td>0.68</td><td>0.53</td><td>0.55</td><td>0.40</td><td>0.34</td><td>0.63</td><td>0.67</td></tr><tr><td>LGrad [57]</td><td>CVPR&#x27;23</td><td>0.33</td><td>0.48</td><td>0.99</td><td>1.00</td><td>0.75</td><td>0.44</td><td>0.50</td><td>0.53</td><td>0.50</td><td>0.49</td><td>0.26</td><td>0.46</td><td>0.51</td><td>0.61</td><td>0.67</td><td>0.76</td><td>0.56</td><td>0.53</td></tr><tr><td>UCF [64]</td><td>ICCV&#x27;23</td><td>0.64</td><td>0.74</td><td>0.70</td><td>0.76</td><td>0.61</td><td>0.73</td><td>0.61</td><td>0.70</td><td>0.49</td><td>0.44</td><td>0.51</td><td>0.62</td><td>0.60</td><td>0.58</td><td>0.47</td><td>0.41</td><td>0.58</td><td>0.65</td></tr><tr><td>FatFormer [37]</td><td>CVPR&#x27;24</td><td>0.35</td><td>0.53</td><td>0.95</td><td>0.85</td><td>0.86</td><td>0.94</td><td>0.98</td><td>1.00</td><td>0.68</td><td>0.82</td><td>0.79</td><td>0.89</td><td>0.68</td><td>0.75</td><td>0.87</td><td>0.98</td><td>0.76</td><td>0.85</td></tr><tr><td>FreqNet [58]</td><td>AAAI&#x27;24</td><td>0.32</td><td>0.42</td><td>0.97</td><td>1.00</td><td>0.86</td><td>0.71</td><td>0.50</td><td>0.50</td><td>0.50</td><td>0.50</td><td>0.26</td><td>0.49</td><td>0.61</td><td>0.85</td><td>0.71</td><td>0.79</td><td>0.59</td><td>0.61</td></tr><tr><td>ProDet [8]</td><td>NeurIPS&#x27;24</td><td>0.73</td><td>0.77</td><td>0.54</td><td>0.64</td><td>0.31</td><td>0.44</td><td>0.58</td><td>0.63</td><td>0.48</td><td>0.47</td><td>0.68</td><td>0.66</td><td>0.52</td><td>0.61</td><td>0.47</td><td>0.44</td><td>0.51</td><td>0.57</td></tr><tr><td>DFD-FCG [19]</td><td>CVPR&#x27;25</td><td>0.76</td><td>0.90</td><td>0.75</td><td>0.85</td><td>0.59</td><td>0.89</td><td>0.74</td><td>0.97</td><td>0.61</td><td>0.71</td><td>0.81</td><td>0.86</td><td>0.72</td><td>0.86</td><td>0.53</td><td>0.53</td><td>0.68</td><td>0.85</td></tr><tr><td>Effort [66]</td><td>ICML&#x27;25</td><td>0.84</td><td>0.93</td><td>0.58</td><td>0.57</td><td>0.81</td><td>0.80</td><td>0.83</td><td>0.99</td><td>0.79</td><td>0.85</td><td>0.82</td><td>0.88</td><td>0.65</td><td>0.71</td><td>0.50</td><td>0.52</td><td>0.77</td><td>0.82</td></tr><tr><td>ForAda [11]</td><td>CVPR&#x27;25</td><td>0.85</td><td>0.92</td><td>0.57</td><td>0.46</td><td>0.87</td><td>0.68</td><td>0.89</td><td>0.95</td><td>0.66</td><td>0.73</td><td>0.52</td><td>0.69</td><td>0.57</td><td>0.64</td><td>0.48</td><td>0.47</td><td>0.74</td><td>0.73</td></tr><tr><td>Ours (8-Blocks)</td><td></td><td>0.88</td><td>0.94</td><td>0.54</td><td>0.41</td><td>0.87</td><td>0.85</td><td>0.83</td><td>0.92</td><td>0.70</td><td>0.76</td><td>0.75</td><td>0.81</td><td>0.65</td><td>0.69</td><td>0.55</td><td>0.47</td><td>0.78</td><td>0.80</td></tr><tr><td>Ours (12-Blocks)</td><td></td><td>0.92</td><td>0.97</td><td>0.60</td><td>0.55</td><td>0.89</td><td>0.90</td><td>0.84</td><td>0.93</td><td>0.72</td><td>0.78</td><td>0.80</td><td>0.86</td><td>0.68</td><td>0.73</td><td>0.56</td><td>0.52</td><td>0.80</td><td>0.84</td></tr></table>

Table 4. Protocol #3 (RLDF Region-Level Test Set): Frame-level performance. Models are trained on FF++ and evaluated on our proposed RLDF benchmark across eight face-swap methods. Bold = best; underline = second-best (per metric, per column). Weighted mean is image-count weighted (total 286,456 images; counts are near-uniform so weighting is effectively equal).
<table><tr><td rowspan=2 colspan=23>Wtd.Detector         Venue BlendFace   E4S   FaceDancer FSGAN   Ghost  InSwapperSimSwap  UniFace RLDF: All  MeanACCAUCACCAUCACCAUCACCAUCACCAUCACCAUC</td></tr><tr><td rowspan=1 colspan=2>ACC AUC</td><td rowspan=1 colspan=6>ACC AUC ACC AUC ACC AUC</td></tr><tr><td rowspan=3 colspan=3>Xception [52]    ICCV&#x27;19 0.660.73</td><td rowspan=3 colspan=2>0.700.78</td><td rowspan=3 colspan=1>0.69</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=2 colspan=1>0.76</td><td rowspan=2 colspan=1>0.80</td><td rowspan=2 colspan=1>0.88</td><td rowspan=2 colspan=1>0.69</td><td rowspan=2 colspan=1>0.77</td><td rowspan=2 colspan=3>0.79</td><td rowspan=2 colspan=1>0.87</td><td rowspan=2 colspan=1>0.66</td><td rowspan=2 colspan=1>0.74</td><td rowspan=2 colspan=2>0.680.74</td><td rowspan=2 colspan=2>0.710.78</td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>0.71</td><td rowspan=1 colspan=1>0.78</td></tr><tr><td rowspan=1 colspan=1>F3Net [46]      ECCV&#x27;20</td><td rowspan=1 colspan=1>0.59</td><td rowspan=1 colspan=1>0.69</td><td rowspan=1 colspan=2>0.690.79</td><td rowspan=1 colspan=1>0.61</td><td rowspan=1 colspan=1>0.72</td><td rowspan=1 colspan=1>0.81</td><td rowspan=1 colspan=1>0.92</td><td rowspan=1 colspan=1>0.66</td><td rowspan=1 colspan=1>0.80</td><td rowspan=1 colspan=3>0.84</td><td rowspan=1 colspan=1>0.93</td><td rowspan=1 colspan=1>0.63</td><td rowspan=1 colspan=1>0.77</td><td rowspan=1 colspan=1>0.59</td><td rowspan=1 colspan=1>0.69</td><td rowspan=1 colspan=1>0.68</td><td rowspan=1 colspan=1>0.79</td><td rowspan=1 colspan=1>0.68</td><td rowspan=1 colspan=1>0.79</td></tr><tr><td rowspan=1 colspan=1>RECCE [5]      CVPR&#x27;22</td><td rowspan=1 colspan=1>0.62</td><td rowspan=1 colspan=1>0.71</td><td rowspan=1 colspan=1>0.75</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>0.66</td><td rowspan=1 colspan=1>0.75</td><td rowspan=1 colspan=1>0.82</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.70</td><td rowspan=1 colspan=1>0.80</td><td rowspan=1 colspan=3>0.84</td><td rowspan=1 colspan=1>0.92</td><td rowspan=1 colspan=1>0.67</td><td rowspan=1 colspan=1>0.77</td><td rowspan=1 colspan=1>0.64</td><td rowspan=1 colspan=1>0.73</td><td rowspan=1 colspan=1>0.71</td><td rowspan=1 colspan=1>0.80</td><td rowspan=1 colspan=1>0.71</td><td rowspan=1 colspan=1>0.81</td></tr><tr><td rowspan=1 colspan=1>SBI [53]         CVPR&#x27;22</td><td rowspan=1 colspan=1>0.70</td><td rowspan=1 colspan=1>0.81</td><td rowspan=1 colspan=1>0.93</td><td rowspan=1 colspan=1>0.98</td><td rowspan=1 colspan=1>0.62</td><td rowspan=1 colspan=1>0.74</td><td rowspan=1 colspan=1>0.77</td><td rowspan=1 colspan=1>0.88</td><td rowspan=1 colspan=1>0.70</td><td rowspan=1 colspan=1>0.79</td><td rowspan=1 colspan=3>0.81</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>0.67</td><td rowspan=1 colspan=1>0.78</td><td rowspan=1 colspan=1>0.7</td><td rowspan=1 colspan=1>0.821</td><td rowspan=1 colspan=1>0.74</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>0.74</td><td rowspan=1 colspan=1>0.84</td></tr><tr><td rowspan=3 colspan=1>LGrad [57]      CVPR&#x27;23</td><td rowspan=3 colspan=1>0.50</td><td rowspan=3 colspan=1>0.54</td><td rowspan=3 colspan=1>0.50</td><td rowspan=3 colspan=1>0.57</td><td rowspan=3 colspan=1>0.50</td><td rowspan=3 colspan=1>0.53</td><td rowspan=3 colspan=1>0.49</td><td rowspan=3 colspan=1>0.48</td><td rowspan=3 colspan=1>0.50</td><td rowspan=3 colspan=1>0.57</td><td rowspan=3 colspan=3>0.50</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=2 colspan=1>0</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>0.59</td><td rowspan=1 colspan=1>0.50</td><td rowspan=1 colspan=1>0.58</td><td rowspan=1 colspan=1>0.50</td><td rowspan=1 colspan=1>0.53</td><td rowspan=1 colspan=1>0.50</td><td rowspan=1 colspan=1>0.55</td><td rowspan=1 colspan=1>0.50</td><td rowspan=1 colspan=1>0.55</td></tr><tr><td rowspan=1 colspan=1>UCF [64]       ICCV&#x27;23</td><td rowspan=1 colspan=1>0.64</td><td rowspan=1 colspan=1>0.73</td><td rowspan=1 colspan=1>0.82</td><td rowspan=1 colspan=1>0.90</td><td rowspan=1 colspan=1>0.68</td><td rowspan=1 colspan=1>0.78</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.95</td><td rowspan=1 colspan=1>0.77</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=2>0.</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>0.74</td><td rowspan=1 colspan=1>0.85</td><td rowspan=1 colspan=1>0.6</td><td rowspan=1 colspan=1>0.734</td><td rowspan=1 colspan=1>0.75</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1>0.85</td></tr><tr><td rowspan=1 colspan=1>FatFormer [37]  CVPR&#x27;24</td><td rowspan=1 colspan=1>0.53</td><td rowspan=1 colspan=1>0.61</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>0.98</td><td rowspan=1 colspan=1>0.57</td><td rowspan=1 colspan=1>0.72</td><td rowspan=1 colspan=1>0.57</td><td rowspan=1 colspan=1>0.70</td><td rowspan=1 colspan=1>0.57</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=3>0.55</td><td rowspan=1 colspan=1>0.75</td><td rowspan=1 colspan=1>0.56</td><td rowspan=1 colspan=1>0.72</td><td rowspan=1 colspan=1>0.56</td><td rowspan=1 colspan=1>0.71</td><td rowspan=1 colspan=1>0.61</td><td rowspan=1 colspan=1>0.75</td><td rowspan=1 colspan=1>0.61</td><td rowspan=1 colspan=1>0.74</td></tr><tr><td rowspan=1 colspan=1>FreqNet [58]    AAAI&#x27;24</td><td rowspan=1 colspan=1>0.38</td><td rowspan=1 colspan=1>0.45</td><td rowspan=1 colspan=2>0.380.46</td><td rowspan=1 colspan=1>0.39</td><td rowspan=1 colspan=1>0.47</td><td rowspan=1 colspan=1>0.32</td><td rowspan=1 colspan=1>0.360</td><td rowspan=1 colspan=1>.38</td><td rowspan=1 colspan=1>0.460</td><td rowspan=1 colspan=3>.38</td><td rowspan=1 colspan=1>0.47</td><td rowspan=1 colspan=1>0.38</td><td rowspan=1 colspan=1>0.46</td><td rowspan=1 colspan=1>0.39</td><td rowspan=1 colspan=1>0.46</td><td rowspan=1 colspan=1>0.38</td><td rowspan=1 colspan=1>0.45</td><td rowspan=1 colspan=1>0.38</td><td rowspan=1 colspan=1>0.45</td></tr><tr><td rowspan=1 colspan=1>ProDet [8]      NeurIPS&#x27;24</td><td rowspan=1 colspan=1>0.81</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.97</td><td rowspan=1 colspan=1>0.74</td><td rowspan=1 colspan=1>0.82</td><td rowspan=1 colspan=1>0.86</td><td rowspan=1 colspan=1>0.95</td><td rowspan=1 colspan=1>0.78</td><td rowspan=1 colspan=1>0.86</td><td rowspan=1 colspan=3>0.79</td><td rowspan=1 colspan=1>0.86</td><td rowspan=1 colspan=1>0.77</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>0.79</td><td rowspan=1 colspan=1>0.88</td><td rowspan=1 colspan=1>0.80</td><td rowspan=1 colspan=1>0.88</td><td rowspan=1 colspan=1>0.80</td><td rowspan=1 colspan=1>0.89</td></tr><tr><td rowspan=1 colspan=1>DFD-FCG [19]  CVPR&#x27;25</td><td rowspan=1 colspan=1>0.61</td><td rowspan=1 colspan=1>0.67</td><td rowspan=1 colspan=1>0.71</td><td rowspan=1 colspan=1>0.98</td><td rowspan=1 colspan=1>0.66</td><td rowspan=1 colspan=1>0.79</td><td rowspan=1 colspan=1>0.69</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.63</td><td rowspan=1 colspan=1>0.74</td><td rowspan=1 colspan=3>0.66</td><td rowspan=1 colspan=1>0.80</td><td rowspan=1 colspan=1>0.57</td><td rowspan=1 colspan=1>0.62</td><td rowspan=1 colspan=1>0.65</td><td rowspan=1 colspan=1>0.78</td><td rowspan=1 colspan=1>0.67</td><td rowspan=1 colspan=1>0.81</td><td rowspan=1 colspan=1>0.65</td><td rowspan=1 colspan=1>0.78</td></tr><tr><td rowspan=1 colspan=1>Effort [66]       ICML&#x27;25</td><td rowspan=1 colspan=1>0.62</td><td rowspan=1 colspan=1>0.67</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>0.99</td><td rowspan=1 colspan=1>0.77</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>0.93</td><td rowspan=1 colspan=1>0.78</td><td rowspan=1 colspan=1>0.85</td><td rowspan=1 colspan=3>0.80</td><td rowspan=1 colspan=1>0.88</td><td rowspan=1 colspan=1>0.68</td><td rowspan=1 colspan=1>0.74</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1>0.83</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>0.77</td><td rowspan=1 colspan=1>0.84</td></tr><tr><td rowspan=1 colspan=1>ForAda [11]     CVPR&#x27;25</td><td rowspan=1 colspan=2>0.680.75</td><td rowspan=1 colspan=2>0.920.99</td><td rowspan=1 colspan=1>0.79</td><td rowspan=1 colspan=1>0.86</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.94</td><td rowspan=1 colspan=1>0.74</td><td rowspan=1 colspan=1>0.81</td><td rowspan=1 colspan=3>0.80</td><td rowspan=1 colspan=1>0.86</td><td rowspan=1 colspan=1>0.66</td><td rowspan=1 colspan=1>0.72</td><td rowspan=1 colspan=2>0.780.85</td><td rowspan=1 colspan=4>0.780.850.780.85</td></tr><tr><td rowspan=1 colspan=1>Ours (8-Blocks)</td><td rowspan=1 colspan=2>0.750.85</td><td rowspan=1 colspan=12>0.920.960.810.900.840.910.800.880.800.88</td><td rowspan=1 colspan=1>0.80</td><td rowspan=2 colspan=7>0.700.790.880.800.880.810.870.770.880.760.860.760.86</td></tr><tr><td rowspan=1 colspan=1>Ours (12-Blocks)</td><td rowspan=1 colspan=4>0.690.820.920.97</td><td rowspan=1 colspan=1>0.79</td><td rowspan=1 colspan=5>0.890.810.900.740.85</td><td rowspan=1 colspan=4>0.740.85</td><td rowspan=1 colspan=1>0.63</td><td rowspan=1 colspan=1>0.75</td></tr></table>

Table 5. Cross-Protocol Generalization. Our detector is the only method maintaining strong performance across all three regimes. Bold = best, underline = second-best (per metric, per column).
<table><tr><td rowspan="2">Detector</td><td rowspan="2">Venue</td><td colspan="2">P#1 (Classic)</td><td colspan="2">P#2 (DF40)</td><td colspan="2">P#3 (RLDF)</td><td colspan="2">Mean</td></tr><tr><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td></tr><tr><td>Xception [52]</td><td>ICCV&#x27;19</td><td>0.67</td><td>0.73</td><td>0.65</td><td>0.67</td><td>0.71</td><td>0.78</td><td>0.68</td><td>0.73</td></tr><tr><td>F3Net [46]</td><td>ECCV’20</td><td>0.63</td><td>0.71</td><td>0.52</td><td>0.53</td><td>0.68</td><td>0.79</td><td>0.61</td><td>0.68</td></tr><tr><td>RECCÈ [5]</td><td>CVPR&#x27;22</td><td>0.64</td><td>0.70</td><td>0.59</td><td>0.56</td><td>0.71</td><td>0.81</td><td>0.65</td><td>0.69</td></tr><tr><td>SBI [53]</td><td>CVPR&#x27;22</td><td>0.69</td><td>0.77</td><td>0.63</td><td>0.67</td><td>0.74</td><td>0.84</td><td>0.69</td><td>0.76</td></tr><tr><td>LGrad [57]</td><td>CVPR&#x27;23</td><td>0.41</td><td>0.50</td><td>0.56</td><td>0.53</td><td>0.50</td><td>0.55</td><td>0.49</td><td>0.53</td></tr><tr><td>UCF [64]</td><td>ICCV’23</td><td>0.63</td><td>0.72</td><td>0.58</td><td>0.65</td><td>0.76</td><td>0.85</td><td>0.66</td><td>0.74</td></tr><tr><td>FatFormer [37]</td><td>CVPR&#x27;24</td><td>0.56</td><td>0.61</td><td>0.76</td><td>0.85</td><td>0.61</td><td>0.74</td><td>0.64</td><td>0.73</td></tr><tr><td>FreqNet [58]</td><td>AAAI&#x27;24</td><td>0.43</td><td>0.52</td><td>0.59</td><td>0.61</td><td>0.38</td><td>0.45</td><td>0.47</td><td>0.53</td></tr><tr><td>ProDet [8]</td><td>NeurIPS&#x27;24</td><td>0.69</td><td>0.75</td><td>0.51</td><td>0.57</td><td>0.80</td><td>0.89</td><td>0.67</td><td>0.74</td></tr><tr><td>DFD-FČG [19]</td><td>CVPR’25</td><td>0.69</td><td>0.82</td><td>0.68</td><td>0.85</td><td>0.65</td><td>0.78</td><td>0.67</td><td>0.82</td></tr><tr><td>Effort [66]</td><td>ICML&#x27;25</td><td>0.74</td><td>0.83</td><td>0.77</td><td>0.82</td><td>0.77</td><td>0.84</td><td>0.76</td><td>0.83</td></tr><tr><td>ForAda [11]</td><td>CVPR’25</td><td>0.75</td><td>0.87</td><td>0.74</td><td>0.73</td><td>0.78</td><td>0.85</td><td>0.76</td><td>0.82</td></tr><tr><td>Ours (8-Blocks)</td><td></td><td>0.74</td><td>0.82</td><td>0.78</td><td>0.80</td><td>0.81</td><td>0.87</td><td>0.78</td><td>0.83</td></tr><tr><td>Ours (12-Blocks)</td><td></td><td>0.78</td><td>0.86</td><td>0.80</td><td>0.84</td><td>0.76</td><td>0.86</td><td>0.78</td><td>0.85</td></tr></table>

Table 6. Ablation Study. Component-wise analysis of the proposed Mask-Guided Cross-Attention (MCA) framework across three evaluation protocols: P1 (classic cross-dataset), P2 (DF40), and P3 (RLDF region-level benchmark). ∆ denotes the difference from the full model (negative indicates performance drop).
<table><tr><td rowspan="3">Variant</td><td colspan="6">Absolute Performance</td><td colspan="6">∆ vs Full Model</td></tr><tr><td colspan="2">P1 (Classic)</td><td colspan="2">P2 (DF40)</td><td colspan="2">P3 (RLDF)</td><td colspan="2">P1 (Classic)</td><td colspan="2">P2 (DF40)</td><td colspan="2">P3 (RLDF)</td></tr><tr><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td><td>∆ACC</td><td>ΔAUC</td><td>ΔACC</td><td>ΔAUC</td><td>∆ACC</td><td>ΔAUC</td></tr><tr><td>w/o mask guidance</td><td>0.772</td><td>0.852</td><td>0.749</td><td>0.780</td><td>0.607</td><td>0.850</td><td>+0.001</td><td>+0.000</td><td>-0.051</td><td>-0.060</td><td>-0.155</td><td>-0.011</td></tr><tr><td>w/o forged path  $( \mathbf { F } _ { f } { = } \mathbf { F } _ { g } )$ </td><td>0.771</td><td>0.851</td><td>0.748</td><td>0.774</td><td>0.605</td><td>0.850</td><td>+0.001</td><td>-0.001</td><td>-0.052</td><td>-0.066</td><td>-0.157</td><td>-0.012</td></tr><tr><td>w/o global path  $( \mathbf { F } _ { g } { = } \mathbf { F } _ { f } )$ </td><td>0.770</td><td>0.850</td><td>0.749</td><td>0.779</td><td>0.617</td><td>0.850</td><td>-0.001</td><td>-0.001</td><td>-0.051</td><td>-0.061</td><td>-0.144</td><td>-0.011</td></tr><tr><td>Hard mask (binary m)</td><td>0.764</td><td>0.851</td><td>0.748</td><td>0.780</td><td>0.648</td><td>0.851</td><td>-0.007</td><td>-0.001</td><td>-0.052</td><td>-0.060</td><td>-0.113</td><td>-0.011</td></tr><tr><td>Full model (baseline)</td><td>0.771</td><td>0.852</td><td>0.800</td><td>0.840</td><td>0.761</td><td>0.861</td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

specialist participants. Extending RLDF to reenactment methods and to video-level temporal analysis are therefore the most immediate directions for future work.

## 6. Conclusion

We presented RLDF, the first dataset and benchmark for evaluating deepfake detectors at the region level. RLDF contains over 2M images produced by 8 face-swapping methods applied to 8 distinct facial regions, built using a regionaware partial swap synthesis. We evaluated 12 detectors across three protocols. First finding is that, no single detector performs well across all three protocols, indicating that existing methods are not generalizable when the evaluation paradigm change. Second, the three protocols are orthogonal and perform well on classic or DF40 benchmarks, but exhibit a significant performance drop when evaluated at the region level. Third, existing detectors fail to generalize consistently across partial facial regions. Performance drops by up to 10.5% AUC and 10% accuracy when the manipulated area shrinks from full-face to a single region. To provide a starting point, we propose a mask-guided cross-attention (MCA) baseline that grounds forgery detection at the patch level. Our ablation shows that spatial localization, not just global image-level features, drives gains in region-level evaluation. Ethical considerations. RLDF is built from FaceForensics++ [52] real videos under its academic research licence and is released strictly for non-commercial detection research; no new identities were collected. Users must respect that licence and those of the eight swapping methods used, and must not employ the data for impersonation or any purpose harming the privacy or reputation of the source individuals (Appendix E). Data, code and checkpoints are released under the same non-commercial terms.

## References

[1] Josh Achiam, Steven Adler, Sandhini Agarwal, et al. GPT-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 1

[2] Darius Afchar, Vincent Nozick, Junichi Yamagishi, and Isao Echizen. Mesonet: a compact facial video forgery detection network. In 2018 IEEE International Workshop on Information Forensics and Security, pages 1–7. IEEE, 2018. 1

[3] Chaitali Bhattacharyya, Hanxiao Wang, Feng Zhang, Sungho Kim, and Xiatian Zhu. Diffusion deepfake. arXiv preprint arXiv:2404.01579, 2024. 2

[4] Tom Brown, Benjamin Mann, Nick Ryder, et al. Language models are few-shot learners. Advances in Neural Information Processing Systems, 2020. 1

[5] Junyi Cao, Chao Ma, Taiping Yao, Shen Chen, Shouhong Ding, and Xiaokang Yang. End-to-end reconstructionclassification learning for face forgery detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4113–4122, 2022. 1, 6, 7, 8, 40, 42, 46, 47, 48

[6] Junsong Chen, Jincheng Yu, Chongjian Ge, et al. PixArt-α: Fast training of diffusion transformer for photorealistic textto-image synthesis. In International Conference on Learning Representations, 2024. 1

[7] Renwang Chen, Xuanhong Chen, Bingbing Ni, and Yanhao Ge. Simswap: An efficient framework for high fidelity face swapping. In MM '20: The 28th ACM International Conference on Multimedia, 2020. 1, 2, 16

[8] Jikang Cheng, Zhiyuan Yan, Ying Zhang, Yuhao Luo, Zhongyuan Wang, and Chen Li. Can we leave deepfake data behind in training deepfake detector? arXiv preprint arXiv:2408.17052, 2024. 1, 2, 5, 6, 7, 8, 40, 42, 46, 47, 48

[9] Yunjey Choi, Minje Choi, Munyoung Kim, Jung-Woo Ha, Sunghun Kim, and Jaegul Choo. Stargan: Unified generative adversarial networks for multi-domain image-to-image translation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 8789–8797, 2018. 1

[10] Yunjey Choi, Youngjung Uh, Jaejun Yoo, and Jung-Woo Ha. Stargan v2: Diverse image synthesis for multiple domains. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8188–8197, 2020. 1

[11] Xinjie Cui, Yuezun Li, Ao Luo, Jiaran Zhou, and Junyu Dong. Forensics adapter: Adapting clip for generalizable face forgery detection. In Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), pages 19207–19217, 2025. 5, 6, 7, 8, 40, 42, 46, 47, 48

[12] Hao Dang, Feng Liu, Joel Stehouwer, Xiaoming Liu, and Anil K Jain. On the detection of digital face manipulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020. 2

[13] Deepfake Detection Challenge. Deepfake detection challenge (DFDC).https://www.kaggle.com/c/deepfakedetection-challenge, 2020. Accessed: 2021-04-24. 1,2

[14] Jiankang Deng, Jia Guo, Jing Yang, Niannan Xue, Irene Kotsia, and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. IEEE Transactions

on Pattern Analysis and Machine Intelligence, 44(10):5962– 5979, 2022. 3, 5

[15] Brian Dolhansky, Russ Howes, Ben Pflaum, Nicole Baram, and Cristian Canton Ferrer. The deepfake detection challenge (dfdc) preview dataset. arXiv preprint arXiv:1910.08854, 2019.1

[16] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in Neural Information Processing Systems, 2014. 1

[17] Google AI. Contributing data to deepfake detection research. https://ai.googleblog.com/2019/09/ contributing-data-to-deepfake-detection. htm1, 2019. Accessed: 2021-04-24. 2

[18] Alexander Groshev, Anastasia Maltseva, Daniil Chesakov, Andrey Kuznetsov, and Denis Dimitrov. Ghost—a new face swap approach for image and video domains. IEEE Access, 10:83452–83462, 2022. 2, 16

[19] Yue-Hua Han, Tai-Ming Huang, Kai-Lung Hua, and Jun-Cheng Chen. Towards more general video-based deepfake detection through facial component guided adaptation for foundation model. In Proceedings of the Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 1, 2, 6, 7, 8, 40, 42, 46, 47, 48

[20] Yinan He, Bei Gan, Siyu Chen, Yichun Zhou, Guojun Yin, Luchuan Song, Lu Sheng, Jing Shao, and Ziwei Liu. Forgerynet: A versatile benchmark for comprehensive forgery analysis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4360–4369, 2021. 2

[21] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, 2020. 1

[22] Liming Jiang, Ren Li, Wayne Wu, Chen Qian, and Chen Change Loy. Deeperforensics-1.0: A large-scale dataset for real-world face forgery detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020. 2

[23] Yan Ju, Shan Jia, Jialing Cai, Haiying Guan, and Siwei Lyu. Glff: Global and local feature fusion for ai-synthesized image detection. IEEE Transactions on Multimedia, 2023. 2

[24] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019. 1

[25] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of StyleGAN. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020. 1

[26] Hasam Khalid, Shahroz Tariq, Minha Kim, and Simon S Woo. Fakeavceleb: A novel audio-video multimodal deepfake dataset. arXiv preprint arXiv:2108.05080, 2021. 2

[27] Pavel Korshunov and Sébastien Marcel. Deepfakes: a new threat to face recognition? assessment and detection. arXiv preprint arXiv:1812.08685, 2018. 2

[28] Marek Kowalski. FaceSwap. https://github.com/ MarekKowalski/FaceSwap, 2020. Accessed: 2020-05- 10.1

[29] Patrick Kwon, Jaeseong You, Gyuhyeon Nam, Sungwoo Park, and Gyeongsu Chae. Kodf: A large-scale korean deepfake detection dataset. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10744–10753, 2021.2

[30] Hanzhe Li, Jiaran Zhou, Yuezun Li, Baoyuan Wu, Bin Li, and Junyu Dong. Freqblender: Enhancing deepfake detection by blending frequency knowledge. In Conference on Neural Information Processing Systems, 2024. 2

[31] Lingzhi Li, Jianmin Bao, Hao Yang, Dong Chen, and Fang Wen. Advancing high fidelity identity swapping for forgery detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5074–5083, 2020.1

[32] Lingzhi Li, Jianmin Bao, Ting Zhang, Hao Yang, Dong Chen, Fang Wen, and Baining Guo. Face x-ray for more general face forgery detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020.1, 2

[33] Yuezun Li and Siwei Lyu. Exposing deepfake videos by detecting face warping artifacts. In IEEE Conference on Computer Vision and Pattern Recognition Workshops, 2019. 2

[34] Yuezun Li, Ming-Ching Chang, and Siwei Lyu. In ictu oculi: Exposing AI generated fake face videos by detecting eye blinking. In IEEE International Workshop on Information Forensics and Security, 2018. 2

[35] Yuezun Li, Xin Yang, Pu Sun, Honggang Qi, and Siwei Lyu. Celeb-df: A new dataset for deepfake forensics. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020. 1, 2

[36] Honggu Liu, Xiaodan Li, Wenbo Zhou, Yuefeng Chen, Yuan He, Hui Xue, Weiming Zhang, and Nenghai Yu. Spatialphase shallow learning: rethinking face forgery detection in frequency domain. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021. 1

[37] Huan Liu, Zichang Tan, Chuangchuang Tan, Yunchao Wei, Jingdong Wang, and Yao Zhao. Forgery-aware adaptive transformer for generalizable synthetic image detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 1, 2, 5, 6, 7, 8, 40, 42, 46, 47, 48

[38] Zhian Liu, Maomao Li, Yong Zhang, Cairong Wang, Qi Zhang, Jue Wang, and Yongwei Nie. Fine-grained face swapping via regional gan inversion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8578–8587, 2023. 2, 16

[39] Yuchen Luo, Yong Zhang, Junchi Yan, and Wei Liu. Generalizing face forgery detection with high-frequency features. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021. 1

[40] Kartik Narayan, Harsh Agarwal, Kartik Thakral, Surbhi Mittal, Mayank Vatsa, and Richa Singh. Df-platter: multiface heterogeneous deepfake dataset. In Proceedings of

the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9739–9748, 2023. 2

[41] Huy H Nguyen, Junichi Yamagishi, and Isao Echizen. Capsule-forensics: Using capsule networks to detect forged images and videos. In IEEE International Conference on Acoustics, Speech and Signal Processing, pages 2307–2311. IEEE, 2019. 1

[42] Yuval Nirkin, Yosi Keller, and Tal Hassner. Fsgan: Subject agnostic face swapping and reenactment. In Proceedings of the IEEE/CVF international conference on computer vision, pages 7184–7193, 2019. 1, 2, 16

[43] Or Patashnik, Zongze Wu, Eli Shechtman, Daniel Cohen-Or, and Dani Lischinski. Styleclip: Text-driven manipulation of stylegan imagery. In Proceedings of the IEEE/CVF international conference on computer vision, pages 2085–2094, 2021.1

[44] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023. 1

[45] KR Prajwal, Rudrabha Mukhopadhyay, Vinay P Namboodiri, and CV Jawahar. A lip sync expert is all you need for speech to lip generation in the wild. In Proceedings of the 28th ACM international conference on multimedia, pages 484–492, 2020.1

[46] Yuyang Qian, Guojun Yin, Lu Sheng, Zixuan Chen, and Jing Shao. Thinking in frequency: Face forgery detection by mining frequency-aware clues. In Computer Vision – ECCV 2020, pages 86–103, Cham, 2020. Springer International Publishing. 6, 7, 8, 40, 42, 46, 47, 48

[47] Yuyang Qian, Guojun Yin, Lu Sheng, Zixuan Chen, and Jing Shao. Thinking in frequency: Face forgery detection by mining frequency-aware clues. In European Conference Computer Vision, pages 86–103. Springer, 2020. 1, 2

[48] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, pages 8748–8763. PMLR, 2021. 3, 5

[49] Yurui Ren, Ge Li, Yuanqi Chen, Thomas H Li, and Shan Liu. Pirenderer: Controllable portrait image generation via semantic neural rendering. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 13759– 13768, 2021. 1

[50] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022. 1

[51] Felix Rosberg, Eren Erdal Aksoy, Fernando Alonso-Fernandez, and Cristofer Englund. Facedancer: Pose-and occlusion-aware high fidelity face swapping. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 3454–3463, 2023. 1, 2, 16

[52] Andreas Rössler, Davide Cozzolino, Luisa Verdoliva, Christian Riess, Justus Thies, and Matthias Nießner. FaceForen-

sics++: Learning to detect manipulated facial images. In International Conference on Computer Vision (ICCV), 2019. 1, 2, 3, 5, 6, 7, 8, 16, 40, 42, 46, 47, 48

[53] Kaede Shiohara and Toshihiko Yamasaki. Detecting deepfakes with self-blended images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18720–18729, 2022. 6, 7, 8, 40, 42, 46, 47, 48

[54] Kaede Shiohara and Toshihiko Yamasaki. Detecting deepfakes with self-blended images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18720–18729, 2022. 1, 2

[55] Kaede Shiohara, Xingchao Yang, and Takafumi Taketomi. Blendface: Re-designing identity encoders for face-swapping. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7634–7644, 2023. 2, 16

[56] Haixu Song, Shiyu Huang, Yinpeng Dong, and Wei-Wei Tu. Robustness and generalizability of deepfake detection: A study with diffusion models. arXiv preprint arXiv:2309.02218, 2023. 2

[57] Chuangchuang Tan, Yao Zhao, Shikui Wei, Guanghua Gu, and Yunchao Wei. Learning on gradients: Generalized artifacts representation for gan-generated images detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12105–12114, 2023. 6, 7, 8, 40, 42, 46, 47, 48

[58] Chuangchuang Tan, Yao Zhao, Shikui Wei, Guanghua Gu, Ping Liu, and Yunchao Wei. Frequency-aware deepfake detection: Improving generalizability through frequency space learning, 2024. 6, 7, 8, 40, 42, 46, 47, 48

[59] Omer Tov, Yuval Alaluf, Yotam Nitzan, Or Patashnik, and Daniel Cohen-Or. Designing an encoder for stylegan image manipulation. ACM Transactions on Graphics (TOG), 40(4): 1–14, 2021. 1

[60] Chengrui Wang and Weihong Deng. Representative forgery mining for fake face detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021. 1

[61] Haofan Wang. inswapper: One-click face swapper. https : //github.com/haofanwang/inswapper,2023. 1, 2,16

[62] Yaohui Wang, Di Yang, Francois Bremond, and Antitza Dantcheva. Latent image animator: Learning to animate images via latent space navigation. arXiv preprint arXiv:2203.09043, 2022. 1

[63] Chao Xu, Jiangning Zhang, Yue Han, Guanzhong Tian, Xianfang Zeng, Ying Tai, Yabiao Wang, Chengjie Wang, and Yong Liu. Designing one unified framework for high-fidelity face reenactment and swapping. In European conference on computer vision, pages 54–71. Springer, 2022. 1, 2, 16

[64] Zhiyuan Yan, Yong Zhang, Yanbo Fan, and Baoyuan Wu. Ucf: Uncovering common features for generalizable deepfake detection. arXiv preprint arXiv:2304.13949, 2023. 1, 2, 6, 7, 8, 40, 42, 46, 47, 48

[65] Zhiyuan Yan, Yong Zhang, Xinhang Yuan, Siwei Lyu, and Baoyuan Wu. Deepfakebench: A comprehensive benchmark of deepfake detection. In Advances in Neural Information

Processing Systems, pages 4534–4565. Curran Associates, Inc., 2023. 1, 2

[66] Zhiyuan Yan, Jiangming Wang, Zhendong Wang, Peng Jin, Ke-Yue Zhang, Shen Chen, Taiping Yao, Shouhong Ding, Baoyuan Wu, and Li Yuan. Effort: Efficient orthogonal modeling for generalizable ai-generated image detection. arXiv preprint arXiv:2411.15633, 2024. 1, 2, 6, 7, 8, 40, 42, 46, 47, 48

[67] Zhiyuan Yan, Taiping Yao, Shen Chen, Yandan Zhao, Xinghe Fu, Junwei Zhu, Donghao Luo, Li Yuan, Chengjie Wang, Shouhong Ding, et al. Df40: Toward next-generation deepfake detection. arXiv preprint arXiv:2406.13495, 2024. 1, 2, 5

[68] Xin Yang, Yuezun Li, and Siwei Lyu. Exposing deep fakes using inconsistent head poses. In ICASSP 2019-2019 IEEE international conference on acoustics, speech and signal processing (ICASSP), pages 8261–8265. IEEE, 2019. 2

[69] Xin Yang, Yuezun Li, and Siwei Lyu. Exposing deep fakes using inconsistent head poses. In IEEE International Conference on Acoustics, Speech and Signal Processing, 2019. 2

[70] Changqian Yu, Changxin Gao, Jingbo Wang, Gang Yu, Chunhua Shen, and Nong Sang. Bisenet v2: Bilateral network with guided aggregation for real-time semantic segmentation, 2020.3

[71] Wenxuan Zhang, Xiaodong Cun, Xuan Wang, Yong Zhang, Xi Shen, Yu Guo, Ying Shan, and Fei Wang. Sadtalker: Learning realistic 3d motion coefficients for stylized audiodriven single image talking face animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8652–8661, 2023. 1

[72] Hanqing Zhao, Wenbo Zhou, Dongdong Chen, Tianyi Wei, Weiming Zhang, and Nenghai Yu. Multi-attentional deepfake detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021. 1

[73] Tianfei Zhou, Wenguan Wang, Zhiyuan Liang, and Jianbing Shen. Face forensics in the wild. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5778–5788, 2021. 2

# RLDF: A Region-Level Deepfake Dataset and Benchmark for Partial-Face Manipulation Detection Supplementary Material

## A. Content Overview

This appendix provides supplementary material for RLDF. Due to space constraints in the main paper, detailed results, visualisations, and analyses are presented here. The content is organised as follows:

• Section B: Datasheet for RLDF. Full documentation of the dataset following Datasheets for Datasets, covering motivation, composition, collection process, preprocessing, recommended and discouraged uses, distribution terms, and maintenance.

• Section C: Face-Swapping Method Selection. Criteria and characteristics of the eight face-swapping methods used in RLDF construction, covering technical paradigm, publication venue, and visual artefact profile.

• Section D: Human Perceptual Realism Study. A 37-participant survey rating 14 region-level deepfake failure cases on a 0 to 5 fakeness scale, confirming that partial-face forgeries are perceptually ambiguous even to non-specialist humans.

• Section E: Ethical Statement. Usage constraints, licence compliance, and data provenance for RLDF.

• Section F: Dataset Visualisations. Representative partial-face swap samples across all eight methods and seven manipulation regions (Eye, Nose, Mouth, Composite-1/2/3, Full Face), illustrating the region isolation pipeline and soft region masks.

• Section G: Region-Level Performance Breakdown. Comprehensive quantitative analysis under Protocol #3, including:

– ROC curves averaged per region and per swap method;

– Per-detector AUC heatmaps across all method×region pairs;

– Confusion matrices analysing false-positive and false-negative patterns;

– t-SNE feature space projections per detector and per region.

• Section H: MaskNet Spatial Analysis and Score Distributions. Analysis of MaskNet's emergent spatial attention including:

– Spatial activation maps for BlendFace across all 7 regions (validating unsupervised spatial grounding);

– Activation maps for Eye-region fakes across all 8 swap methods (showing intensity correlates with detectability).

– Cross-dataset activation maps on CelebDF, DFDC, DFDCP, and UADFV (demonstrating dataset-agnostic generalization).

– Real/fake score distribution analysis across regions and detectors.

• Section I: Per-Method Region-Level Analysis (Protocol #3). Complete ACC/ AUC tables for all 13 detectors across seven manipulation regions, reported separately for each of the eight face-swapping methods.

• Section J: Qualitative Failure Cases. False-negative examples (fake images misclassified as real) from Ours (12-Blk), organised by manipulation region and sorted by detection confidence.

## B. Datasheet for RLDF

<table><tr><td>Motivation</td></tr></table>

For what purpose was the dataset created? Was there a specific task in mind? Was there a specific gap that needed to be filled? RLDF was created to evaluate deepfake detection when manipulation is confined to a localised facial sub-region rather than the whole face. Existing benchmarks (FaceForensics++, Celeb-DF, DFDC, DFDCP, UADFV, DF40) annotate manipulation at the global image level with a binary label, and every forgery they contain replaces or synthesises the entire face. No prior dataset supports region-level evaluation, so it was not previously possible to ask whether detectors remain reliable as the manipulated area shrinks. RLDF fills that gap by providing partial face swaps with per-region annotations.

Who created this dataset and on behalf of which entity? Anonymised for double-blind review.

Who funded the creation of the dataset? If there is an associated grant, please provide the name of the grantor and the grant name and number. Anonymised for double-blind review.

Any other comments? None

<table><tr><td>Composition</td></tr></table>

What do the instances that comprise the dataset represent? Are there multiple types of instances? Each instance is a single 224× 224 aligned face crop, labelled real or fake. A fake instance is additionally annotated with the face-swapping method used and the facial region that was manipulated. There is one instance type, and real and fake instances differ only in their label and annotations.

How many instances are there in total? Applying the region-aware partial swap synthesis across |Φ|=8 face-swapping methods, |R|=8 facial regions, T=32 frames per pair and 1,000 source-target video pairs yields 1,000 × 32 × 64 = 2,048,000 fake images, balanced with an equal number of unmodified target frames as real samples. The held-out test split contains approximately 286K images.

Does the dataset contain all possible instances or is it a sample? If the dataset is a sample, what is the larger set? Is the sample representative? RLDF is a sample. The real source footage is drawn from FaceForensics++, and 1,000 source-target video pairs were used, with 32 frames sampled per pair by linear spacing. Frames on which the face detector failed were discarded. Representativeness relative to the full FaceForensics++ distribution has not been formally validated, and RLDF inherits the demographic and capture-condition composition of that corpus.

What data does each instance consist of? “Raw” data or features? Raw image data. Each instance is an RGB face crop, together with a metadata record giving its binary label, and for fakes the swap method and manipulated region. The soft region masks derived during construction are released alongside the images

Is there a label or target associated with each instance? If so, please provide a description. Yes. Every instance carries a binary authenticity label (y=1 fake, y=0 real). Fake instances additionally carry a region label drawn from four discrete regions (eye, nose, mouth, eyebrow) and four composite regions (Composite-1: eye+nose, Composite-2: +mouth, Composite-3: +eyebrow, and Full Face), plus the identity of the generating method.

Is any information missing from individual instances? If so, please explain why. No fields are missing for retained instances Frames on which face detection or alignment failed were excluded during construction rather than retained with partial annotation.

Are relationships between individual instances made explicit? If so, please describe how. Yes. Instances derived from the same source-target video pair share an identifier prefix, and each fake is explicitly linked to the unmodified target frame it was composited onto. This makes the 64 method ×region variants of a given frame identifiable as a group.

Are there recommended data splits? If so, please describe them and the rationale. Yes. RLDF uses a 720/140/140 train/validation/test split defined over source video pairs, with no identity overlap between splits. Splitting at the pair level rather than the frame level prevents frames of the same subject appearing in both training and evaluation.

Are there any errors, sources of noise, or redundancies? If so, please describe. Yes. Region masks are produced by an automatic BiSeNetV2 face parser, so parsing errors propagate into the composited boundary. Because partial fakes are composited from a full swap through a smoothed mask rather than generated region-natively, blend boundaries reflect both the generator and the compositing step. Consecutive sampled frames from the same clip are visually redundant.

Is the dataset self-contained, or does it rely on external resources? If it links to external resources, are there guarantees they will remain constant, and are there restrictions? RLDF depends on FaceForensics++ for its real source footage, which must be obtained separately under that dataset's own licence, and on the eight public face-swapping implementations used in construction (Appendix C). Those repositories are third-party and carry no guarantee of permanence.

Does the dataset contain data that might be considered confidential? If so, please describe. No. RLDF is derived entirely from FaceForensics++ footage, which is publicly released for research, and inherits whatever protections that release applied.

Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or otherwise cause anxiety? If so, please describe why. RLDF consists of manipulated facial imagery. No content-moderation pass was performed beyond that already applied by FaceForensics++. The dataset depicts identifiable people whose likenesses have been altered without their involvement, which some viewers may find uncomfortable, and its purpose is to support detection of exactly that harm.

Does the dataset relate to people? If not, you may skip the remaining questions in this section. Yes. Every instance is a human face.

Does the dataset identify any subpopulations? If so, please describe their distributions. No. RLDF adds no demographic annotation. The underlying distribution is that of FaceForensics++, which has not been rebalanced here. Consequently, per-region detection results reported in this paper should not be read as uniform across demographic groups.

Is it possible to identify individuals, directly or indirectly? If so, please describe how. Yes, visually. RLDF adds no names or explicit identifiers, but the imagery depicts identifiable individuals to the same extent as FaceForensics++, and the real target frames are unmodified.

Does the dataset contain data that might be considered sensitive? If so, please describe. Facial imagery is biometric data. RLDF annotates no other sensitive attribute, though apparent characteristics could in principle be inferred from the images themselves, as in the source corpus.

Any other comments? None.

<table><tr><td>Collection Process</td></tr><tr><td>How was the data associated with each instance acquired? Was it directly observable, reported by subjects, or in- ferred/derived? If inferred, was it validated? Derived. Real footage was taken from FaceForensics++. Fakes were produced by the region-aware partial swap synthesis described in the paper: a full face swap is generated with each method, a soft region mask is derived from the target frame, and only the designated region is composited back onto the target. Region labels follow deterministically from the</td></tr><tr><td>What mechanisms or procedures were used to collect the data? How were these validated? Faces were detected and aligned with InsightFace Antelope at 640×640 and cropped to 224×224. Region masks were produced by a BiSeNetV2 face parser, dilated per region and Gaussian-smoothed (σ=3.0) to avoid hard seams. The eight swap methods were used through their public reference</td></tr><tr><td>If the dataset is a sample from a larger set, what was the sampling strategy? Deterministic. 1,000 source-target pairs were used, and 32 frames per pair were sampled by linear spacing across the clip rather than at random, so the sample is reproducible.</td></tr><tr><td>Who was involved in the data collection process and how were they compensated? Construction was carried out by the authors. The perceptual realism study (Appendix D) involved 37 volunteer participants from academic backgrounds, none of them deepfake</td></tr><tr><td>specialists. Participants were fellow research scholars who assisted voluntarily, and no compensation was provided. Over what timeframe was the data collected? Does this match the creation timeframe of the underlying data? Content generation was carried out between November 2025 and May 2026. This does not match the creation timeframe of the underlying footage,</td></tr><tr><td>Were any ethical review processes conducted? If so, please describe the process and outcome. No formal institutional ethics review was sought, as none was required for a study of this kind: participation was voluntary, no personal or demographic data were</td></tr><tr><td>collected, only anonymous ratings were recorded, and the imagery shown was already public within FaceForensics++. The dataset&#x27;s intended use, licence constraints and prohibited uses are set out in Appendix E. Does the dataset relate to people? If not, you may skip the remaining questions in this section. Yes.</td></tr><tr><td>Did you collect the data from the individuals in question directly, or via third parties? Via a third party. All footage originates from FaceForensics++, and no individual was recorded for this work.</td></tr><tr><td>Were the individuals in question notified about the data collection? If so, how was notice provided? Not by the authors. Notification was a matter for the original FaceForensics++ collection. Did the individuals consent to the collection and use of their data? If so, how was consent requested and provided? Not</td></tr><tr><td>to the authors directly. FaceForensics++ is derived from consented, publicly available video content, and RLDF relies on that consent and adds no separate process. Individuals did not specifically consent to having their likeness partially swapped. If consent was obtained, were individuals provided with a mechanism to revoke it? If so, please describe. Not by the</td></tr><tr><td>authors. Any revocation mechanism is that of FaceForensics++, and a withdrawal upstream would require the corresponding RLDF instances</td></tr><tr><td>to be removed. Has an analysis of the potential impact of the dataset on data subjects been conducted? If so, please describe the</td></tr></table>

Was any preprocessing/cleaning/labeling done? If so, please describe. If not, you may skip the remainder of this section. Yes. Faces were detected, aligned and cropped to 224×224, and frames with no detected face were discarded. Region masks were dilated and Gaussian-smoothed before compositing. Splits were formed at the video-pair level to prevent identity leakage.

Was the “raw" data saved in addition to the preprocessed data? If so, please provide a link. The unmodified target frames are retained as the real half of the dataset. The original FaceForensics++ videos are not redistributed and must be obtained from that project.

Is the software used to preprocess/clean/label the instances available? If so, please provide a link. Yes. The construction pipeline, including mask derivation and compositing, is released with the code (Algorithm 1 in the main paper gives the procedure in full).

Any other comments? None.

<table><tr><td>Uses</td></tr><tr><td></td></tr></table>

Has the dataset been used for any tasks already? If so. please describe. Yes. RLDF forms Protocol #3 of the benchmark reported in this paper, on which 12 state-of-the-art detectors were evaluated alongside two classic cross-dataset protocols. It was also used to train and evaluate the mask-guided cross-attention baseline.

Is there a repository that links to papers or systems that use the dataset? If so, please provide a link. Not at present.

What (other) tasks could the dataset be used for? Forgery localisation and segmentation, since the region of manipulation is known, analysis of detector sensitivity as a function of manipulated area, robustness studies across swap methods, and as pretraining data for spatially grounded detectors.

Is there anything about the composition or collection process that might impact future uses? Is there anything a future user should know to avoid unfair treatment or other harms? Three points. First, partial fakes are composited rather than generated region-natively, so a detector may learn the compositing signature instead of a generator artefact, and conclusions should be checked against region-native manipulations where possible. Second, RLDF covers face swapping only, not reenactment, entire-face synthesis or attribute editing. Third, it inherits the demographic composition of FaceForensics++ and carries no demographic labels, so per-group performance cannot be measured on it and should not be assumed uniform.

Are there tasks for which the dataset should not be used? If so, please describe. RLDF must not be used to build or improve face-swapping or impersonation systems, for political messaging, or for any application that could harm the privacy or reputation of the individuals in the source footage (Appendix E). Detectors trained on it should not be deployed to make consequential decisions about people without independent validation on the target domain.

Any other comments? None

<table><tr><td>Distribution</td></tr><tr><td></td></tr></table>

Will the dataset be distributed to third parties outside of the entity on behalf of which it was created? If so, please describe. Yes. RLDF is intended for public release to the research community.

How will the dataset be distributed? Does the dataset have a digital object identifier (DOl)? Via Hugging Face, with the construction and evaluation code released separately. The release URL is withheld here to preserve double-blind review and will be given in the camera-ready version. No DOI has been minted.

When will the dataset be distributed? Publicly by the camera-ready deadline, pending acceptance.

Will the dataset be distributed under a copyright or other IP license, and/or under applicable terms of use? If so, please describe. Access is conditioned on compliance with the FaceForensics++ academic research licence and with the licences of all eight face-swapping methods used in construction. Use is restricted to non-commercial research. The licence applied to RLDF's own contributed content (region masks, annotations and composited images) will be stated on release, upon acceptance.

Have any third parties imposed IP-based or other restrictions on the data? If so, please describe. Yes. FaceForensics++ and each of the eight swap-method implementations carry their own licences, with which users must independently comply.

Do any export controls or other regulatory restrictions apply? If so, please describe. None known to the authors.

Any other comments? None.

<table><tr><td>Maintenance</td></tr><tr><td></td></tr></table>

Who will be supporting/hosting/maintaining the dataset? Anonymised for double-blind review.

How can the owner/curator/manager of the dataset be contacted? Anonymised for double-blind review.

Is there an erratum? If so, please provide a link. No. Errata will be recorded in the release repository if any arise.

Will the dataset be updated? If so. please describe how often. by whom. and how updates will be communicated. Corrections to region masks or labels will be issued if errors are identified. There is no fixed release schedule, and updates will be published as revisions of the Hugging Face repository, and a changelog will be maintained from the public release onward.

If the dataset relates to people, are there applicable limits on the retention of the data? If so, please describe and explain enforcement. RLDF collects no data directly from individuals, and retention is governed by the FaceForensics++ terms. A withdrawal upstream would require removal of the corresponding RLDF instances.

Will older versions of the dataset continue to be supported/hosted/maintained? If so, please describe how. Releases will be versioned so that published results remain reproducible. Because the repository is git-backed, superseded revisions remain retrievable by their commit identifier and will not be deleted.

If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so? If so please describe. The construction pipeline is released, so others can extend RLDF to further swap methods or regions. Contributions are not centrally validated by the authors.

Any other comments? None.

## C. Face-Swapping Method Selection

RLDF uses eight face-swapping methods chosen to maximise diversity across three axes: technical paradigm (identity disentanglement, GAN-based blending, latent-space inversion, reenactment-guided, and lightweight mobile), publication venue and recency (2019–2023; ArXiv, CVPR, ICCV, ECCV, and WACV), and visual artifact profile (seam-based, style-transfer, colour-shift, and blending artefacts). None supports partial manipulation natively, all operate on the full face, making them suitable for region-aware partial swap synthesis without architectural modification. All eight are publicly available and widely benchmarked in deepfake detection literature.

Table S1. Face-swapping methods used in RLDF construction.
<table><tr><td>#</td><td>Method</td><td>Paradigm</td><td>Venue</td><td>Key characteristic</td><td>Code</td></tr><tr><td></td><td>BlendFace [55]</td><td>GAN, disentangle</td><td>ICCV 2023</td><td>Seam-minimising latent blending; hardest to detect</td><td>Link</td></tr><tr><td>2</td><td>E4S [38]</td><td>Latent inversion</td><td>CVPR 2023</td><td>Style-transfer boundaries; most detectable</td><td>Link</td></tr><tr><td>3</td><td>FaceDancer [51]</td><td>GAN, disentangle</td><td>WACV 2023</td><td>Adaptive feature injection across identities</td><td>Link</td></tr><tr><td>4</td><td>FSGAN [42]</td><td>Reenactment-guided</td><td>ArXiv 2019</td><td>Parsing-mask-based subject-agnostic swap</td><td>Link</td></tr><tr><td>5</td><td>GHOST [18]</td><td>GAN, lightweight</td><td>ArXiv 2022</td><td>Efficient one-shot swap with strong colour consistency</td><td>Link</td></tr><tr><td>6</td><td>InSwapper [61]</td><td>GAN (Roop)</td><td></td><td>Commercial-grade; widely used in real-world deepfakes</td><td>Link</td></tr><tr><td>7</td><td>SimSwap [7]</td><td>GAN, disentangle</td><td>ICCV 2019</td><td>Weak feature matching preserves target attributes</td><td>Link</td></tr><tr><td>8</td><td>UniFace [63]</td><td>GAN, disentangle</td><td>ECCV 2022</td><td>Unified framework for arbitrary face swapping</td><td>Link</td></tr></table>

## D. Human Perceptual Realism Study

To validate that RLDF failure cases are genuinely difficult for humans not just detectors we conducted a perceptual realism survey. Participants were shown 14 region-level deepfake images (2 samples per region across 7 regions: Eye, Nose, Mouth, C-1, C-2, C-3, Full Face) and asked to rate each image on a scale of 0 (looks completely real) to 5 (looks completely fake). All 14 images were drawn from detector failure cases identified in Protocol #3 (Appendix J).

37 participants completed the survey, recruited from diverse academic backgrounds spanning Computer Science, Electronics, Electrical Engineering, Mechanical Engineering, Civil Engineering, and Humanities & Social Sciences. Participants included undergraduate (BE), postgraduate (ME), and doctoral (PhD) scholars, none of whom were deepfake specialists.

Figure S1 reports the mean and standard deviation of perceived fakeness per image. The overall mean score is 2.34/ 5.0, indicating that participants were largely uncertain leaning toward perceiving these images as real. Certain partial-region samples received the lowest scores (Nose S2: 1.11, C-3 S1: 1.24), confirming that small-region forgeries are perceptually near-indistinguishable from authentic faces. In particular, Full-Face samples scored highest (2.92 and 2.38), consistent with the expectation that full-face swaps introduce more visible artifacts than partial-region fakes. The high standard deviations (\~1.7) reflect genuine inter-rater ambiguity rather than systematic disagreement. These results support the core motivation for RLDF: region-level fakes represent a qualitatively harder detection regime for both humans and detectors

## E. Ethical Statement

RLDF is constructed from FaceForensics++ [52] real videos under its academic research licence and is intended strictly for non-commercial deepfake detection research. The dataset must not be used for impersonation, political messaging, or any application that could harm the privacy or reputation of individuals in the source videos. Users must comply with the FaceForensics++ licence [52] and the licences of all face-swapping methods used in dataset construction. All images in RLDF are derived from consented public-domain video content; no new identities were collected or recorded for this work.

![](images/088c9406ca5f40c36d6fad81f115b0927298dbfd4452df9ca72d64b875f021dd.jpg)  
Figure S1. Human Perceptual Realism Survey. Per-region mean fakeness score (0 = real, 5 = obviously fake) rated by 37 participants across 14 region-level deepfake images. Dashed line = midpoint (2.5); dotted red line = overall mean (2.34). Partial-region fakes score below the midpoint; Full-Face fakes are the most perceptually detectable.

## F. RLDF: Dataset Visualizations

This section presents representative visual samples from the RLDF dataset, illustrating the region-isolation pipeline across all eight faceswapping methods and seven manipulation regions. Figures S2 to S9 show sample images spanning single-region manipulations (Eye, Nose, Mouth) through composite regions (Composite-1/2/3) to full face swap, along with their corresponding soft region masks, demonstrating the progressive increase in spatial coverage and detection difficulty as the manipulated region shrinks.

## G. Region-Level Performance Breakdown

ROC Curves: Figures S10 and S11 show ROC curves under Protocol #3 averaged per region and per swap method respectively. Detection difficulty monotonically increases as the manipulated area shrinks: Full Face fakes are easiest (AUC ≥ 0.90 for most detectors) while single-region fakes, particularly the Eye region (≈5% of face area), push all detectors toward chance level. Across swap methods, E4S is the most detectable owing to its visible style-transfer boundaries, while BlendFace is the hardest due to its seam-minimising latent blending.

AUC Heatmaps: Figures S12-S16 show AUC heatmaps across all eight swap methods (rows) and seven manipulation regions (columns) for five detectors. Each cell is the AUC on that specific (method, region) pair from Protocol #3. Our 12-Block detector (Figure S12) shows the most uniform AUC distribution, sustaining strong performance across all cells. Specialist detectors such as ProDet (Figure S16) achieve high AUC on full face and composite regions but degrade sharply for single-region (Eye, Nose, Mouth) fakes, revealing that their decision boundaries are calibrated for global forgery signals. FatFormer (Figure S15) achieves high AUC on E4S across all regions due to its style-transfer artefacts being globally visible, but collapses on BlendFace entirely.

Confusion Matrices: Figures S17–S20 show confusion matrices on the combined RLDF test set. Our 8-Block model achieves the most balanced true positive and true negative rates. ProDet and Effort exhibit high false-negative rates on region-level fakes consistently predicting partial-face manipulations as real consistent with training and design for full face forgery only. FatFormer shows an elevated false-positive rate, flagging many real images as fake, likely because its frequency-sensitive features are triggered by natural facial texture variation.

Feature Space Analysis (t-SNE): Figure S21 compares penultimate-layer t-SNE projections across all detectors on the RLDF test set. Our models show cleaner real/fake separation than competitor methods, particularly for composite and full face regions. Figures S22-S27 show per-region breakdowns for each detector: separation quality degrades as region size decreases for all methods, but degrades most sharply for Effort and FatFormer at the Eye region, where their features are not spatially grounded.

![](images/5c347fdf2fd170345f5b35ace37d35fcd6a80a7a8f07ab91665119962610f876.jpg)  
Manipulation Region

EYE

NOSE

MOUTH

COMPOSITE-1

COMPOSITE-2

COMPOSITE-3

FULL SWAPPED FACE

![](images/f440040543a818ed7c2ad3a8f7cc0e5be93fe76cef567dd6bca3eb7844820412.jpg)

![](images/b9e4ab21212fb6a4199f7945ff342461ffa1fd82d31a7d25c96a0bb4a3cbdaf7.jpg)  
Figure S2. RLDF sample images across 8 face-swapping methods (rows) and seven manipulation regions of increasing spatial coverage from single regions (Eye, Nose, Mouth) to full face swap (columns).

![](images/70eb5f803700ee116579b6e544fe9e1923db6de5af3dd3545c5dfbfb3833563e.jpg)

<table><tr><td></td><td rowspan=1 colspan=1>0.66</td><td rowspan=1 colspan=1>0.66</td><td rowspan=1 colspan=1>0.67</td><td rowspan=1 colspan=1>0.70</td><td rowspan=1 colspan=1>0.73</td><td rowspan=1 colspan=1>0.75</td><td rowspan=1 colspan=1>0.78</td><td></td><td rowspan=1 colspan=1>0.71</td></tr><tr><td></td><td rowspan=1 colspan=1>0.82</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.86</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.85</td><td></td><td rowspan=1 colspan=1>0.86</td></tr><tr><td></td><td rowspan=1 colspan=1>0.70</td><td rowspan=1 colspan=1>0.69</td><td rowspan=1 colspan=1>0.68</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1>0.79</td><td rowspan=1 colspan=1>0.80</td><td rowspan=1 colspan=1>0.84</td><td></td><td rowspan=1 colspan=1>0.75</td></tr><tr><td rowspan=4 colspan=1>Swattp hod  FSGANGHOSTInSwapperSimSwap</td><td rowspan=1 colspan=1>0.74</td><td rowspan=1 colspan=1>0.79</td><td rowspan=1 colspan=1>0.78</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.87</td><td></td><td rowspan=1 colspan=1>0.82</td></tr><tr><td rowspan=1 colspan=1>0.71</td><td rowspan=1 colspan=1>0.71</td><td rowspan=1 colspan=1>0.73</td><td rowspan=1 colspan=1>0.77</td><td rowspan=1 colspan=1>0.81</td><td rowspan=1 colspan=1>0.83</td><td rowspan=1 colspan=1>0.81</td><td></td><td rowspan=1 colspan=1>0.77</td></tr><tr><td rowspan=1 colspan=1>0.74</td><td rowspan=1 colspan=1>0.78</td><td rowspan=1 colspan=1>0.79</td><td rowspan=1 colspan=1>0.83</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.88</td><td></td><td rowspan=1 colspan=1>0.82</td></tr><tr><td rowspan=1 colspan=1>0.67</td><td rowspan=1 colspan=1>0.68</td><td rowspan=1 colspan=1>0.69</td><td rowspan=1 colspan=1>0.71</td><td rowspan=1 colspan=1>0.75</td><td rowspan=1 colspan=1>0.77</td><td rowspan=1 colspan=1>0.79</td><td></td><td rowspan=1 colspan=1>0.72</td></tr><tr><td rowspan=1 colspan=1>UniFace</td><td rowspan=1 colspan=1>0.69</td><td rowspan=1 colspan=1>0.69</td><td rowspan=1 colspan=1>0.69</td><td rowspan=1 colspan=1>0.75</td><td rowspan=1 colspan=1>0.79</td><td rowspan=1 colspan=1>0.79</td><td rowspan=1 colspan=1>0.84</td><td></td><td rowspan=1 colspan=1>0.75</td></tr><tr><td rowspan=1 colspan=1>Avg.</td><td rowspan=1 colspan=1>0.71</td><td rowspan=1 colspan=1>0.73</td><td rowspan=1 colspan=1>0.74</td><td rowspan=1 colspan=1>0.78</td><td rowspan=1 colspan=1>0.81</td><td rowspan=1 colspan=1>0.82</td><td rowspan=1 colspan=1>0.83</td><td></td><td rowspan=1 colspan=1>0.77</td></tr></table>

Figure S3. RLDF overview: (a) full face vs. all region-level combined AUC drop across 12 detectors, showing consistent degradation under evaluation, and (b) per-region AUC heatmap averaged across all detectors and swap methods, exposing systematic failure on individual facial regions.

![](images/b821996ad032ae746de17f79aaf611389147934a7a6062a4af40c944f46af0cd.jpg)  
Figure S4. Dataset samples: Eye region (pair 081/087).

![](images/7de5688e09a07d2b51146176054cb5eb94bd41e183967556bbee3542aa3acf4a.jpg)  
Figure S5. Dataset samples: Nose region (pair 081/087).

![](images/da4943001a0642dad8081769baf7ab0a97f1eedc2293f2abbac8ff695dd19ee3.jpg)  
Figure S6. Dataset samples: Mouth region (pair 081/087).

![](images/e311a71ab0f016dbcb5e0ab5a50646af6429aab3a7966aff27caf6fe7a587aee.jpg)  
Figure S7. Dataset samples: Composite-1 (Eye+Nose) region (pair 081/087).

![](images/c9612792ec6e9adee0a57d1fe26dfcf0d317607bb7054c7067349d6a77e57785.jpg)  
Figure S8. Dataset samples: Composite-2 (Eye+Nose+Mouth) region (pair 081/087).

![](images/1986f6e47fc4e80f865f5c9c0b21d65ecef1e17f2c779a74f65c84fd6a1beacd.jpg)  
Figure S9. Dataset samples: Composite-3 (Eye+Nose+Mouth+Eyebrow) region (pair 081/087).

![](images/83bd4e81e1660cfe668731febee3d9553f77fcb42783eec2d2db0eb9f9087d3c.jpg)

Per ds  d d  ddd   n vdds  
![](images/579c916bad8dfd78dcae7f9ae49d3bf94e6fa0e3b9e09d21266ace73feeabb6e.jpg)  
Riser t    esep s es e ee  e (# eo e ot o eea e  n edet

![](images/d0bd3320b194e93cfcc20c05f1f0596c4e7a10815ae5ae28b0218e41ad78ff68.jpg)

![](images/0b99cb72f3d4d8521e30b3e1129a485c779b3c407fb048819ff5f140e546adac.jpg)

![](images/fd80c17f6c66c2ff9abd6260d2046bb775e7b0c46a1256435837e1d12f843bae.jpg)

![](images/290038b7ff670db2e38f104ba0ead3e5a868e52ec09e8a2d91069bc903a30433.jpg)

![](images/a609175e6ea551fb1c75df369108eeaaea24da73b2792a1f9687428e5f853101.jpg)

![](images/25f9d8cc953c073079bca884a554aafc546c37b532c91b9cb841d45ad6280fe2.jpg)  
ise         (# o  da- d  s s   st.

Ours (12-BIk): AUC per Method × Region Mean AUC = 0.882  
![](images/809317fdaf763f901625a3a24e364c802cdf1b59a508321b2a5f4aa315b177ea.jpg)  
Figure S12. AUC heatmap: Ours (12-Blk). Most uniform coverage.

![](images/927303420e1f0ba4e0ffae07067c01f1e91c454c5a62c428945f656205345113.jpg)  
Figure S13. AUC heatmap: Ours (8-Blk). Slightly stronger on small regions.

![](images/d4e7eeb77bce8b19e8b3b420583a7ed0ef668b1a03a3723ef403ecedd006f90f.jpg)  
Figure S14. AUC heatmap: Effort. Strong on full face, weakens for single-region.

![](images/29a03bf82d462e9302dbfe80f0284d5ee03fe6b8005a6bf271aaa4345a8737f7.jpg)  
Figure S15. AUC heatmap: FatFormer. High on E4S, collapses on BlendFace.

![](images/79fc6b737c88d12a5ff7a06488e7449c2a682d5c16fec7735c1cd7565a096b7c.jpg)  
Figure S16. AUC heatmap: ProDet. Strong on Full Face, poor on small regions.

![](images/bd95d98e7325e36ad984dfff2b98bdfc36cc20a4ba7bf594be2064603d9b2c02.jpg)

Ours (8-BIk): Confusion Matrix per Region (all 8 methods combined)  
![](images/c4290ce6f5c15e312678f83ecf3a5873b86b562f483e0b9fc3ba3261fd0127b2.jpg)

![](images/66da2b810b78002751d5f0ad992ad2dc61a0c302bfe54175dd71a9adeb0b8b90.jpg)

![](images/8b9ba936e823b8ef26bf8e29dca7c8cac4ab2403063f0b56b06a6c19eef5d8d7.jpg)

![](images/d7ef185f8f57dedcebe17a7e9490a932d153da52f6eecc64c9cdffe83cbf0669.jpg)

![](images/66bf2fe75c2c28539843d4771d1207da0c2c6927b5359016981f66e8dfc5ddb3.jpg)  
Figure S17. Confusion matrix: Ours (8-Blk). Most balanced FP/FN across all regions.  
Effort: Confusion Matrix per Region (all 8 methods combined)

![](images/01e2a97b499a3ee728ac64dfce8fc379bc23cce56a8a7d4a41853541c871c177.jpg)

![](images/d687833f52441329146268c598e0dc48cee2923ce05fd8c1b43eb90847d69dc1.jpg)

![](images/da7c52d7e0193544097d9599afdb7afc58969f83cc0678f38edf90be0785603e.jpg)

![](images/6bdfbb3809ddd860e6b7678b3864ed40678c4eed90f61a07e79d50c3ce1387f3.jpg)

![](images/ee0a4d5979e43aade1b17516344484c65bd2c44a2e6936a01b57ae6796ee63d9.jpg)

![](images/fe7d14f0040d33a2599ff6071008c61c53abd28e9bf89800073295268541f018.jpg)  
Figure S18. Confusion matrix: Effort. High false-negative rate on single-region fakes.  
ProDet: Confusion Matrix per Region (all 8 methods combined)

![](images/14f197a92fc869e818115d38db6d254a96b27d1af98f1e0dbb266f0f26ad0b60.jpg)

![](images/78b2786ccd9d11775d5f5af26f293fe039ea2c9d3efc5b383e32c7f9faa5d93b.jpg)

![](images/ffe3a1f088f91684f0e0856bdb5db302002183a6271d60521d651ed9e6cd23b1.jpg)

![](images/cd996b2a85545824f676ab8c91b44a45b05f80fbb542df0cc79a666fde60d691.jpg)

![](images/15b09c901ccee0fe97b3e6fb8177a4b741f2c0a85f4df7f9e2672470b7b5cf44.jpg)

![](images/b1c0d4f753f9b4ef5c9b24ba3f242d979c72462685e5a9c3ee61441818975cfb.jpg)

![](images/ab2a596a2c2f019d8a5549ee058f4c14f980f41c8e374f2da257e1407c0b40bd.jpg)

Figure S19. Confusion matrix: ProDet. High FN on sub-region fakes, strong on full face.  
![](images/6762c2bdc7fd5dc46f1679ca0b725b50ec79e427d1507b8748b3321fddfb8cbf.jpg)

![](images/5f210207f5c48391638b5dbfdbb25ff4b21a64a8394c52e1f672d040fefa9871.jpg)

FatFormer: Confusion Matrix per Region (all 8 methods combined)  
![](images/30f0cda9a3e5e0bd137349ade1e2a5ae80826b4c008ff434a4f1c2c76998c2bd.jpg)  
Figure S20. Confusion matrix: FatFormer. Elevated false-positive rate on real images.

![](images/8c644cd199d5b6dae23c841897cdca0afd1ef79de86dba887f8a49fe6a1fa562.jpg)  
Figure S21. t-SNE of penultimate features on the RLDF test set. Blue = Real, Red = Fake. Our models achieve cleaner separation than competitors.

![](images/5c861e3edb990e6d1d946fc64d87ac594cbb3d1bce2e7a9140e88f3f59d7f47d.jpg)  
Fi - u-o :- io  uon

![](images/b15b823fea408e501e6bd1919193a0890fce7bd8387a84282489f28c769311dd.jpg)  
i - -o :(- o  ton

![](images/b258adc210f3c8016e7841883637e7bf0308807aec097a194e73fd1cb1b52de1.jpg)  
Fio PF- t :o Pt ton

![](images/b33cfca01af8b45530623fdca93c0218dbee32c0568b8d225d4eaa4abf1ee0f1.jpg)  
Fi P-  :Pto t tton

![](images/5025d6ecfa8fb4158be0324bde11d14a0e81a07c3b41073d9d91bf261ca7c299.jpg)  
Ri P- - :o P on

![](images/c6603f0453ff7558cce8762601f825bb4962c3192f41ef76f27117c2cd940f42.jpg)  
Fi F- -sr :o  tton

## H. MaskNet Spatial Analysis and Score Distributions

MaskNet Spatial Analysis. Figures S29 and S30 show the spatial distribution of per-patch forgery scores mî predicted by MaskNet, visualised as 16× 16 heatmaps (one cell per CLIP patch position), averaged over all test images for a given (method, region) combination.

Region progression (Figure S29): For BlendFace fakes, high mask scores progressively expand as the manipulated region grows from a tight cluster at the eye patches for Eye-only fakes, through overlapping nose/mouth areas for Composite regions, to broad coverage for Full Face fakes. This spatial alignment emerges purely from the classification gradient flowing through the log-space soft mask in Eq. (3), with no spatial supervision at any point during training. This directly validates the differentiable gate design: MaskNet and MCA co-adapt to localise the forgery without pixel-level annotation.

Method variation at Eye region (Figure S30): Across all eight swap methods, MaskNet consistently concentrates activation around the eye patches for Eye-region fakes. Mask intensity varies by method: BlendFace produces the weakest activation (seam-minimising; hardest to detect, AUC 0.85), while E4S and FaceDancer produce the strongest (style-transfer artefacts; $\mathrm { A U C } \geq 0 . 9 6 )$ . This variation directly predicts per-method detectability, confirming that MaskNet's learned spatial response serves as an implicit indicator of forgery strength.

Cross-dataset generalization (Figure S28): On unseen cross-dataset benchmarks (CelebDF-v1/v2, DFDC, DFDCP, UADFV), MaskNet has never seen these images during training yet produces informative spatial masks. Scores cluster around the central face region eyes, nose bridge, and mouth, which are the dominant loci of identity and the areas most altered by full face swapping methods. Crucially, the masks do not collapse to uniform noise (which would happen if MaskNet had no forgery-relevant signal), nor do they saturate the entire image (which would indicate over-confident global detection). Instead they exhibit structured, face-centric activation that generalises the spatial prior learned on FF++ training data to entirely different datasets and generation methods, providing qualitative evidence that MaskNet learns a dataset-agnostic notion of “where faces get swapped" rather than overfitting to FF++ artefact statistics.

![](images/b8bbca1b2aca55a64b54870742269608a3df61b8023984edba6f524a7e3d3b7b.jpg)

![](images/aa4d6b74aa4bfba528c7456ec94a2551be6192410b945d598b0fb96577ba4bad.jpg)  
CelebDF-v1

![](images/75d516e9436b061adc0fdf4187b356cbf5a9df19c5071bada25ab1f643551cc1.jpg)

![](images/07b94ed2e1448703f2cd40f839ca0ff019c5473c3982825910b458928bb47a4b.jpg)  
CelebDF-v2

![](images/3f885866f2e186e045a7a047fd85c478d01a31e272bbf094581bc5c6665ad430.jpg)

![](images/9539c634bbaa1a32fb4e9279a759319a6c1c3c6e28c225adf377ddeca82130c8.jpg)

![](images/a611885287312d9e9c5fa351bbfc39b04be137ceb203463d1b0a3bc927556320.jpg)

![](images/7e0b7ba67a231ed5900003c7f64370d71e289717c7c3371854055400b1a0eda7.jpg)  
DFDCP

DFDC  
![](images/a25f52ac83f250fb4cd4e80883cc13eedb8fc0b4f5c5812b1e0e7d5cc20ad2eb.jpg)

![](images/32ab6538a7863991fc7b7400ea3ea4d903d674c7969344d896983cc512bf765d.jpg)  
UADFV  
Figure S28. MaskNet activation on unseen cross-dataset benchmarks. Despite training exclusively on FF++, MaskNet produces structured, face-centric masks across CelebDF, DFDC, DFDCP, and UADFV, consistently activating around eyes, nose, and mouth.

![](images/162c59624dd158a8f04d6d4a7e4afd9cfc2d5ed6afcb950bf052f5fd750bce93.jpg)  
e sn aes e e Me  es e e-d ae t s  n s  e ss e : e  et ·roo oe w o  os s  osowss

![](images/3ec6f24439c28b664b55ae1992036d6d2cab8cd2a1dbef8025744271c62f333a.jpg)  
ds  (  )      st rd ( nns).

## I. Per-Method Region-Level Analysis (Protocol #3)

Tables S2-S9 report the complete per-method, per-region detection performance under Protocol #3 across all 13 evaluated detectors and seven manipulation regions (Eye, Nose, Mouth, Composite-1, Composite-2, Composite-3, Full Face).

Several consistent trends emerge across all eight methods. First, detection accuracy increases monotonically with region size: the Eye and Nose columns consistently show the lowest AUC across all detectors and all methods, while the Full Face column shows the highest. This confirms that detection difficulty is primarily governed by the spatial extent of the manipulation rather than the generation method itself. Table S2. Protocol #3 (BlendFace): Region-level performance (ACC/ AUC). Bold = best, underline = second-best (per metric, per column). C-1: eye+nose, C-2: eye+nose+mouth, C-3: eye+nose+mouth+eyebrow.
<table><tr><td rowspan="2">Detector</td><td rowspan="2">Venue</td><td colspan="2">Eye</td><td colspan="2">Nose</td><td colspan="2">Mouth</td><td colspan="2">C-1</td><td colspan="2">C-2</td><td colspan="2">C-3</td><td colspan="2">Full</td></tr><tr><td colspan="2">ACC AUC</td><td colspan="2">ACC</td><td colspan="2">AUC ACC</td><td colspan="2">AUC ACC</td><td colspan="2">AUC ACC AUC</td><td colspan="2">ACC AUC</td><td colspan="2">ACC AUC</td></tr><tr><td></td><td></td><td>0.62</td><td></td><td></td><td>0.70</td><td></td><td>0.73</td><td>0.66</td><td>0.72</td><td>0.70</td><td>0.78</td><td></td><td>0.79</td><td></td><td></td></tr><tr><td>Xception [52] F3Net [46]</td><td>ICCV&#x27;19 ECCV’20</td><td>0.56 0.66</td><td>0.68</td><td>0.64 0.55</td><td>0.64</td><td>0.66 0.59</td><td>0.69</td><td>0.56</td><td>0.67</td><td>0.60</td><td>0.72</td><td>0.71 0.62</td><td>0.75</td><td>0.72 0.67</td><td>0.83 0.81</td></tr><tr><td>RECCE [5]</td><td>CVPR&#x27;22</td><td>0.59</td><td>0.68</td><td>0.59</td><td>0.68</td><td>0.58</td><td>0.69</td><td>0.61</td><td>0.71</td><td></td><td>0.61 0.73</td><td>0.64</td><td>0.76</td><td>0.67</td><td>0.78</td></tr><tr><td>SBI [53]</td><td>CVPR&#x27;22</td><td>0.62</td><td>0.73</td><td>0.62</td><td>0.73</td><td>0.60</td><td>0.72</td><td>0.70</td><td>0.82</td><td>0.76</td><td>0.87</td><td>0.77</td><td>0.89</td><td>0.85</td><td>0.94</td></tr><tr><td>LGrad [57]</td><td>CVPR’23</td><td>0.50</td><td>0.54</td><td>0.50</td><td>0.53</td><td>0.50</td><td>0.53</td><td>0.50</td><td>0.54</td><td>0.50</td><td>0.54</td><td>0.50</td><td>0.54</td><td>0.50</td><td>0.53</td></tr><tr><td>UCF [64]</td><td>ICCV’23</td><td>0.59</td><td>0.68</td><td>0.59</td><td>0.68</td><td>0.61</td><td>0.72</td><td>0.61</td><td>0.72</td><td>0.65</td><td>0.77</td><td>0.69</td><td>0.81</td><td>0.73</td><td>0.84</td></tr><tr><td>FatFormer [37]</td><td>CVPR&#x27;24</td><td>0.51</td><td>0.57</td><td>0.51</td><td>0.59</td><td>0.51</td><td>0.57</td><td>0.52</td><td>0.61</td><td>0.53</td><td>0.62</td><td>0.54</td><td>0.64</td><td>0.57</td><td>0.70</td></tr><tr><td>FreqNet [58]</td><td>AAAI&#x27;24</td><td>0.39</td><td>0.45</td><td>0.39</td><td>0.45</td><td>0.38</td><td>0.45</td><td>0.39</td><td>0.45</td><td>0.39</td><td>0.45</td><td>0.38</td><td>0.45</td><td>0.37</td><td>0.44</td></tr><tr><td>ProDet [8]</td><td>NeurIPS’24</td><td>0.75</td><td>0.82</td><td>0.78</td><td>0.86</td><td>0.77</td><td>0.84</td><td>0.82</td><td>0.91</td><td>0.86</td><td>0.94 </td><td>0.85</td><td>0.94</td><td>0.87</td><td>0.96</td></tr><tr><td>DFD-FCG [19]</td><td>CVPR&#x27;25</td><td>0.58</td><td>0.59</td><td>0.57</td><td>0.58</td><td>0.59</td><td>0.63</td><td>0.61</td><td>0.65</td><td>0.66</td><td>0.74</td><td>0.67</td><td>0.77</td><td>0.69</td><td>0.81</td></tr><tr><td>Effort [66]</td><td>ICML’25</td><td>0.58 0.63</td><td></td><td>0.57</td><td>0.60</td><td>0.57</td><td>0.61</td><td>0.63</td><td>0.67</td><td>0.65</td><td>0.71 </td><td>0.68</td><td>0.74</td><td>0.71</td><td>0.77</td></tr><tr><td>ForAda [11]</td><td>CVPR’25</td><td>0.65</td><td>0.72</td><td>0.66</td><td>0.72</td><td>0.63</td><td>0.71</td><td>0.70</td><td>0.76</td><td>0.71</td><td>0.77</td><td>0.73</td><td>0.80</td><td>0.73</td><td>0.81</td></tr><tr><td>Ours (12-Blk)</td><td></td><td>0.61</td><td>0.77</td><td>0.62</td><td>0.78</td><td>0.59</td><td>0.77</td><td>0.72</td><td>0.85</td><td>0.77</td><td>0.89</td><td>0.82</td><td>0.93</td><td>0.90</td><td>0.96</td></tr></table>

Second, detector behaviour is strongly method-dependent. E4S (Table S3) is the most detectable across all detectors AUC values exceed 0.90 for most methods even at the Eye region owing to its visible style-transfer boundaries that are detectable even in small regions. BlendFace (Table S2) is the hardest: its seam-minimising latent blending produces the lowest AUC values consistently, with several detectors falling below 0.70 at the Eye region. SimSwap (Table S8) similarly challenges all detectors, particularly at single-region granularities, due to its weak feature matching that preserves target attributes and minimises swap-induced discontinuities. Third, frequency-based detectors (LGrad, FreqNet) fail systematically across all methods and all regions (AUC near 0.50), confirming that the C23 compression artefacts exploited during training do not transfer to region-level manipulations. Conversely, our proposed 8-Block and 12-Block baselines maintain competitive AUC across most method-region pairs, with performance particularly strong on FSGAN (Table S5), FaceDancer (Table S4), and UniFace (Table S9), where localised blending seams provide strong patch-level signals for MaskNet. Score Distributions: Figure S32 directly compares real/fake score distributions for Eye-region fakes vs Full Face fakes. The Eye region shows near-complete overlap between real and fake distributions, explaining the low detection accuracy, full face shows clear bimodal separation. Detection difficulty scales monotonically with the manipulated area, validating RLDF's progressive region granularity as a diagnostic tool. Figure S32 confirms this across all detectors: mean P(fake) on fake samples rises monotonically from Eye to Full Face for every method evaluated. Figures S31–S35 show per-region real/fake KDE score distributions for four detectors

![](images/cc38dd2f51449978d23e7455a35711b75f6347000702a5bc6d59299c0f31d758.jpg)

Table S3. Protocol #3 (E4S): Region-level performance (ACC/ AUC). Bold = best, underline = second-best (per metric, per column). C-1: eye+nose, C-2: eye+nose+mouth, C-3: eye+nose+mouth+eyebrow.
<table><tr><td rowspan="2">Detector</td><td rowspan="2">Venue</td><td colspan="2">Eye</td><td colspan="2">Nose</td><td colspan="2">Mouth</td><td colspan="2">C-1</td><td colspan="2">C-2</td><td colspan="2">C-3</td><td colspan="2">Full</td></tr><tr><td>ACC</td><td>CAUC</td><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td></tr><tr><td>Xception [52]</td><td>ICCV&#x27;19</td><td>0.70</td><td>0.75</td><td>0.70</td><td>0.77</td><td>0.74 </td><td>0.82</td><td>0.71</td><td>0.77</td><td>0.67</td><td>0.74</td><td>0.70</td><td>0.78</td><td>0.71</td><td>0.83</td></tr><tr><td>F3Net [46]</td><td>ECCV&#x27;20</td><td>0.68</td><td>0.78</td><td>0.70</td><td>0.79</td><td>0.72</td><td>0.82</td><td>0.67</td><td>0.76</td><td>0.68</td><td>0.76</td><td>0.69</td><td>0.79</td><td>0.55</td><td>0.66</td></tr><tr><td>RECCE [5]</td><td>CVPR&#x27;22</td><td>0.74</td><td>0.81</td><td>0.80</td><td>0.88</td><td>0.66</td><td>0.77</td><td>0.81</td><td>0.89</td><td>0.73</td><td>0.83</td><td>0.76</td><td>0.85</td><td>0.64</td><td>0.76</td></tr><tr><td>SBI [53]</td><td>CVPR&#x27;22</td><td>0.88</td><td>0.92</td><td>0.92</td><td>0.98</td><td>0.89</td><td>0.96</td><td>0.95</td><td>1.00</td><td>0.95</td><td>1.00</td><td>0.95</td><td>1.00</td><td>0.95</td><td>1.00</td></tr><tr><td>LGrad [57]</td><td>CVPR&#x27;23</td><td>0.50</td><td>0.56</td><td>0.50</td><td>0.56</td><td>0.50</td><td>0.57</td><td>0.50</td><td>0.57</td><td>0.50</td><td>0.58</td><td>0.50</td><td>0.57</td><td>0.50</td><td>0.58</td></tr><tr><td>UCF [64]</td><td>ICCV’23</td><td>0.74</td><td>0.83</td><td>0.84</td><td>0.91</td><td>0.83</td><td>0.92</td><td>0.82</td><td>0.91</td><td>0.81</td><td>0.91</td><td>0.83</td><td>0.92</td><td>0.65</td><td>0.79</td></tr><tr><td>FatFormer [37]</td><td>CVPR&#x27;24</td><td>0.89</td><td>0.93</td><td>0.94</td><td>0.99</td><td>0.91</td><td>0.99</td><td>0.98</td><td>0.99</td><td>0.99</td><td>0.99</td><td>0.99</td><td>0.99</td><td>0.99</td><td>1.00</td></tr><tr><td>FreqNet [58]</td><td>AAAI&#x27;24</td><td>0.37</td><td>0.44</td><td>0.39</td><td>0.47</td><td>0.38</td><td>0.45</td><td>0.38</td><td>0.46</td><td>0.38</td><td>0.46</td><td>0.39</td><td>0.47</td><td>0.35</td><td>0.42</td></tr><tr><td>ProDet [8]</td><td>NeurIPS’24</td><td>0.83</td><td>0.93</td><td>0.88</td><td>0.98</td><td>0.86</td><td>0.95</td><td>0.88</td><td>0.99</td><td>0.88</td><td>0.99</td><td>0.88</td><td>0.99</td><td>0.88</td><td>0.99</td></tr><tr><td>DFD-FCG [19]</td><td>CVPR&#x27;25</td><td>0.70</td><td>0.91</td><td>0.74</td><td>0.99</td><td>0.74</td><td>0.99</td><td>0.74</td><td>1.00</td><td>0.74</td><td>1.00 </td><td>0.74</td><td>1.00</td><td>0.74</td><td>1.00</td></tr><tr><td>Effort [66]</td><td>ICML’25</td><td>0.84</td><td>0.93</td><td>0.90</td><td>1.00</td><td>0.90</td><td>0.99</td><td>0.90</td><td>1.00</td><td>0.90</td><td>1.00</td><td>0.90</td><td>1.00</td><td>0.90</td><td>1.00</td></tr><tr><td>ForAda [11]</td><td>CVPR&#x27;25</td><td>0.88</td><td>0.94</td><td>0.94</td><td>1.00 </td><td>0.93</td><td>0.99</td><td>0.94</td><td>1.00 </td><td></td><td>0.94 1.00</td><td>0.94</td><td>1.00</td><td>0.93</td><td>0.99</td></tr><tr><td>Ours (12-Blk)</td><td></td><td>0.88</td><td>0.93</td><td>0.96</td><td>0.99</td><td>0.84</td><td>0.93</td><td>0.96</td><td>0.99</td><td>0.96</td><td>0.99</td><td>0.96</td><td>0.99</td><td>0.96</td><td>0.99</td></tr></table>

![](images/99efd7402309aa3b75ab6ff92c31bcc6108211b91509a5dcb2ae821e9c086b3a.jpg)

![](images/10a23ad293a704c5ef22a903297a81390091d996301f03f8d3aa554a4b531143.jpg)

![](images/cdb12ff513612e42ef1b781656501a8c1a3991ee583c41d7bd9504df9c23ab29.jpg)

![](images/240b3a3cf03dd7a5ba98102b298cba04d3941bf0b005ab12eeee3226ab18fb23.jpg)

![](images/db8eac9bc3c9fa16cdf9edb7d0b7eda91c90de4e1c57220e850c0ab595197d74.jpg)

![](images/4ed380b02d3206ba6ae47ee410674b27235cd89c7c791f37db667e0f083136b7.jpg)

![](images/e0ffb47d1da4c8aa5ec41e6c2f3375475719e2ab48ed834c83fed1d5006cbf55.jpg)

![](images/d762676e285bc543d11f19c48a4fe6f1e2959de079ca78cd6d8b9bea5f6752fc.jpg)

![](images/fb28e4565779778c62a1ff75148a3d6127a7eed4cf2062034c24324d9bf46044.jpg)

![](images/ad538e24cde0ef65d7711a60692ac063b352a694e2054148c4c8471befa5e418.jpg)  
Figure S32. Eye vs Full Face score distributions. Eye shows near-complete real/fake overlap, Full Face shows clear bimodal separation.

![](images/338321f939858471c9aae84e5557c5eade3ed00b7900d00080dbe9bdd4af4aa3.jpg)

![](images/1e46886733324b6574f08140a347357ff5cbef5023f8bc31783de36d87941de9.jpg)

![](images/a053abaaf3e8a9ca2df55cd2a70616549e68a9ff6fd29db83574cce9fec2e0c1.jpg)

Table S4. Protocol #3 (FaceDancer): Region-level performance (ACC / AUC). Bold = best; underline = second-best (per metric, per column). C-1: eye+nose, C-2: eye+nose+mouth, C-3: eye+nose+mouth+eyebrow.
<table><tr><td rowspan="2">Detector</td><td rowspan="2">Venue</td><td colspan="2">Eye</td><td colspan="2">Nose</td><td colspan="2">Mouth</td><td colspan="2">C-1</td><td colspan="2">C-2</td><td colspan="2">C-3</td><td colspan="2">Full</td></tr><tr><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td></tr><tr><td>Xception [52]</td><td>ICCV&#x27;19</td><td>0.63</td><td>0.69</td><td>0.67</td><td>0.74</td><td>0.67</td><td>0.75</td><td>0.68</td><td>0.76</td><td>0.74</td><td>0.81 </td><td>0.73</td><td>0.80</td><td>0.73</td><td>0.85</td></tr><tr><td>F3Net [46]</td><td>ECCV’20</td><td>0.58</td><td>0.67</td><td>0.61</td><td>0.72 </td><td>0.58</td><td>0.68 </td><td>0.62</td><td>0.73</td><td>0.65</td><td>0.76 </td><td>0.65</td><td>0.76</td><td>0.74</td><td>0.84</td></tr><tr><td>RECCE [5]</td><td>CVPR&#x27;22</td><td>0.61</td><td>0.68</td><td>0.65</td><td>0.73</td><td>0.64</td><td>0.75</td><td>0.66</td><td>0.74</td><td>0.71</td><td>0.81</td><td>0.70</td><td>0.80</td><td>0.79</td><td>0.87</td></tr><tr><td>SBI [53]</td><td>CVPR&#x27;22</td><td>0.58</td><td>0.68</td><td>0.56</td><td>0.67</td><td>0.56</td><td>0.65</td><td>0.61</td><td>0.74</td><td>0.63</td><td>0.77</td><td>0.64</td><td>0.78</td><td>0.77</td><td>0.89</td></tr><tr><td>LGrad [57]</td><td>CVPR’23</td><td>0.50</td><td>0.53</td><td>0.50</td><td>0.54</td><td>0.50</td><td>0.54</td><td>0.50</td><td>0.53</td><td>0.50</td><td>0.53</td><td>0.50</td><td>0.52</td><td>0.50</td><td>0.52</td></tr><tr><td>UCF [64]</td><td>ICCV’23</td><td>0.62</td><td>0.72</td><td>0.66</td><td>0.77</td><td>0.64</td><td>0.75</td><td>0.68</td><td>0.79</td><td>0.71</td><td>0.83</td><td>0.70</td><td>0.82</td><td>0.79</td><td>0.89</td></tr><tr><td>FatFormer</td><td>CVPR&#x27;24</td><td>0.52</td><td>0.67</td><td>0.51</td><td>0.60</td><td>0.51</td><td>0.61</td><td>0.54</td><td>0.71</td><td>0.56</td><td>0.74</td><td>0.65</td><td>0.84</td><td>0.70</td><td>0.87</td></tr><tr><td>FreqNet [58]</td><td>AAAI&#x27;24</td><td>0.38</td><td>0.46</td><td>0.39</td><td>0.47</td><td>0.39</td><td>0.46</td><td>0.39</td><td>0.47</td><td>0.40</td><td>0.47</td><td>0.40</td><td>0.48</td><td>0.41</td><td>0.48</td></tr><tr><td>ProDet [8]</td><td>NeurIPS’24</td><td>0.70</td><td>0.77</td><td>0.70</td><td>0.77</td><td>0.69</td><td>0.76</td><td>0.76</td><td>0.83</td><td>0.78</td><td>0.86 </td><td>0.78</td><td>0.85</td><td>0.81</td><td>0.90</td></tr><tr><td>DFD-FCG [19]</td><td>CVPR&#x27;25</td><td>0.65</td><td>0.73</td><td>0.62</td><td>0.67</td><td>0.61</td><td>0.65</td><td>0.69</td><td>0.82</td><td></td><td>0.71 0.87</td><td>0.71</td><td>0.90</td><td>0.72</td><td>0.93</td></tr><tr><td>Effort [66]</td><td>ICML&#x27;25</td><td>0.72</td><td>0.77</td><td>0.66</td><td>0.71</td><td>0.66</td><td>0.71</td><td>0.79</td><td>0.87</td><td>0.83</td><td>0.91</td><td>0.86</td><td>0.95</td><td>0.87</td><td>0.96</td></tr><tr><td>ForAda [11]</td><td>CVPR&#x27;25</td><td>0.73</td><td>0.78</td><td>0.74</td><td>0.80</td><td>0.69</td><td>0.76</td><td>0.82</td><td>0.89</td><td>0.84</td><td>0.90</td><td>0.87</td><td>0.93</td><td>0.88</td><td>0.95</td></tr><tr><td>Ours (12-Blk)</td><td></td><td>0.75</td><td>0.87</td><td></td><td>0.70</td><td>0.84</td><td>0.61 0.79</td><td></td><td>0.86</td><td>0.94</td><td>0.89 0.96</td><td>0.92</td><td>0.97</td><td>0.94</td><td>0.98</td></tr></table>

Table S5. Protocol #3 (FSGAN): Region-level performance (ACC / AUC). Bold = best, underline = second-best (per metric, per column). C-1: eye+nose, C-2: eye+nose+mouth, C-3: eye+nose+mouth+eyebrow.
<table><tr><td>Detector</td><td>Venue</td><td colspan="2">Eye</td><td colspan="2">Nose</td><td colspan="2">Mouth</td><td colspan="2">C-1</td><td colspan="2">C-2</td><td colspan="2">C-3</td><td colspan="2">Full</td></tr><tr><td></td><td></td><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td></tr><tr><td>Xception [52]</td><td>ICCV&#x27;19</td><td>0.67</td><td>0.75</td><td>0.76</td><td>0.84</td><td>0.82</td><td>0.91</td><td>0.79</td><td>0.88</td><td>0.86</td><td>0.96</td><td>0.85</td><td>0.94</td><td>0.85</td><td>0.95</td></tr><tr><td>F3Net [46]</td><td>ECCV’20</td><td>0.64</td><td>0.80</td><td>0.76</td><td>0.89</td><td>0.80</td><td>0.91</td><td>0.81</td><td>0.92</td><td>0.90</td><td>0.97</td><td>0.87</td><td>0.95</td><td>0.92</td><td>0.98</td></tr><tr><td>RECCE [5]</td><td>CVPR&#x27;22</td><td>0.68</td><td>0.79</td><td>0.78</td><td>0.88</td><td>0.86</td><td>0.93</td><td>0.82</td><td>0.91</td><td>0.90</td><td>0.97</td><td>0.88</td><td>0.95</td><td>0.91</td><td>0.98</td></tr><tr><td>SBI [53]</td><td>CVPR&#x27;22</td><td>0.64</td><td>0.77</td><td>0.69</td><td>0.83</td><td>0.66</td><td>0.79</td><td>0.80</td><td>0.91</td><td>0.86</td><td>0.95</td><td>0.86</td><td>0.95</td><td>0.87</td><td>0.96</td></tr><tr><td>LGrad [57]</td><td>CVPR&#x27;23</td><td>0.49</td><td>0.48</td><td>0.49</td><td>0.48</td><td>0.49</td><td>0.48</td><td>0.49</td><td>0.48</td><td>0.49</td><td>0.49</td><td>0.49</td><td>0.49</td><td>0.49</td><td>0.48</td></tr><tr><td>UCF [64]</td><td>ICCV’23</td><td>0.77</td><td>0.87</td><td>0.85</td><td>0.93</td><td>0.87</td><td>0.95</td><td>0.89</td><td>0.96</td><td>0.92</td><td>0.98</td><td>0.91</td><td>0.98</td><td>0.93</td><td>0.99</td></tr><tr><td>FatFormer [37]</td><td>CVPR&#x27;24</td><td>0.54</td><td>0.65</td><td>0.54</td><td>0.64</td><td>0.51</td><td>0.56</td><td>0.59</td><td>0.75</td><td>0.60</td><td>0.75</td><td>0.63</td><td>0.79</td><td>0.59</td><td>0.75</td></tr><tr><td>FreqNet [58]</td><td>AAAI&#x27;24</td><td>0.32</td><td>0.35</td><td>0.32</td><td>0.35</td><td>0.33</td><td>0.35</td><td>0.32</td><td>0.35</td><td>0.32</td><td>0.36</td><td>0.32</td><td>0.36</td><td>0.31</td><td>0.33</td></tr><tr><td>ProDet [8]</td><td>NeurIPS’24</td><td>0.83</td><td>0.92</td><td>0.84</td><td>0.92</td><td>0.86</td><td>0.95</td><td>0.86</td><td>0.95</td><td>0.88</td><td>0.98</td><td>0.88</td><td>0.98</td><td>0.88</td><td>0.98</td></tr><tr><td>DFD-FCG [19]</td><td>CVPR&#x27;25</td><td>0.65</td><td>0.77</td><td>0.67</td><td>0.80</td><td>0.67</td><td>0.79</td><td>0.72</td><td>0.92</td><td>0.73</td><td>0.96</td><td>0.73</td><td>0.97</td><td>0.73</td><td>0.97</td></tr><tr><td>Effort [66]</td><td>ICML’25</td><td>0.77</td><td>0.85</td><td>0.81</td><td>0.89</td><td>0.75</td><td>0.83</td><td>0.87</td><td>0.97</td><td>0.89</td><td>0.99</td><td>0.89</td><td>0.99</td><td>0.89</td><td>0.99</td></tr><tr><td>ForAda [11]</td><td>CVPR&#x27;25</td><td>0.76</td><td>0.84</td><td>0.85</td><td>0.92</td><td>0.83</td><td>0.91</td><td>0.91</td><td>0.97</td><td>0.93</td><td>0.99</td><td>0.93</td><td>0.99</td><td>0.93</td><td>0.99</td></tr><tr><td>Ours (12-Blk)</td><td></td><td>0.69</td><td>0.81</td><td>0.79</td><td>0.88</td><td>0.64</td><td>0.80</td><td>0.90</td><td></td><td>0.96</td><td>0.93 0.97</td><td>0.94</td><td>0.98</td><td>0.96</td><td>0.99</td></tr></table>

Table S6. Protocol #3 (GHOST): Region-level performance (ACC/ AUC). Bold = best, underline = second-best (per metric, per column). C-1: eye+nose, C-2: eye+nose+mouth, C-3: eye+nose+mouth+eyebrow.
<table><tr><td>Detector</td><td>Venue</td><td colspan="2">Eye</td><td colspan="2">Nose</td><td colspan="2">Mouth</td><td colspan="2">C-1</td><td colspan="2">C-2</td><td colspan="2">C-3</td><td colspan="2">Full</td></tr><tr><td></td><td></td><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td></tr><tr><td>Xception [52]</td><td>ICCV&#x27;19</td><td>0.61</td><td>0.68</td><td>0.65</td><td>0.72</td><td>0.74</td><td>0.82</td><td>0.64</td><td>0.73</td><td>0.74</td><td>0.82</td><td>0.76</td><td>0.84</td><td>0.77</td><td>0.88</td></tr><tr><td>F3Net [46]</td><td>ECCV’20</td><td>0.610.76</td><td></td><td>0.61</td><td>0.76</td><td>0.67</td><td>0.82</td><td>0.61</td><td>0.77</td><td>0.67</td><td>0.81</td><td>0.73</td><td>0.87</td><td>0.61</td><td>0.79</td></tr><tr><td>RECCE [5]</td><td>CVPR&#x27;22</td><td>0.64 0.73</td><td></td><td>0.67</td><td>0.77</td><td>0.72</td><td>0.83</td><td>0.66</td><td>0.78</td><td>0.73</td><td>0.83</td><td>0.76</td><td>0.86</td><td>0.57</td><td>0.65</td></tr><tr><td>SBI [53]</td><td>CVPR&#x27;22</td><td>0.61</td><td>0.70</td><td>0.62</td><td>0.70</td><td>0.60</td><td>0.67</td><td>0.72</td><td>0.83</td><td>0.78</td><td>0.88</td><td>0.77</td><td>0.88</td><td>0.77</td><td>0.90</td></tr><tr><td>LGrad [57]</td><td>CVPR&#x27;23</td><td>0.49</td><td>0.57</td><td>0.50</td><td>0.57</td><td>0.50</td><td>0.57</td><td>0.50</td><td>0.57</td><td>0.50</td><td>0.57</td><td>0.50</td><td>0.57</td><td>0.50</td><td>0.57</td></tr><tr><td>UCF [64]</td><td>ICCV’23</td><td>0.71</td><td>0.84</td><td>0.74</td><td>0.86</td><td>0.75</td><td>0.87</td><td>0.75</td><td>0.87</td><td>0.78</td><td>0.89</td><td>0.83</td><td>0.93</td><td>0.69</td><td>0.83</td></tr><tr><td>FatFormer [37]</td><td>CVPR&#x27;24</td><td>0.53</td><td>0.70</td><td>0.51</td><td>0.68</td><td>0.51</td><td>0.68</td><td>0.56</td><td>0.75</td><td>0.58</td><td>0.78</td><td>0.62</td><td>0.81</td><td>0.69</td><td>0.87</td></tr><tr><td>FreqNet [58]</td><td>AAAI&#x27;24</td><td>0.38</td><td>0.45</td><td>0.38</td><td>0.45</td><td>0.38</td><td>0.46</td><td>0.38</td><td>0.46</td><td>0.39</td><td>0.46</td><td>0.39</td><td>0.46</td><td>0.37</td><td>0.44</td></tr><tr><td>ProDet [8]</td><td>NeurIPS’24</td><td>0.73</td><td>0.82</td><td>0.77</td><td>0.85</td><td>0.76</td><td>0.84</td><td>0.81</td><td>0.89</td><td>0.84</td><td>0.92</td><td>0.84</td><td>0.92</td><td>0.77</td><td>0.84</td></tr><tr><td>DFD-FCG [19]</td><td>CVPR&#x27;25</td><td>0.59</td><td>0.64</td><td>0.56</td><td>0.59</td><td>0.63</td><td>0.70</td><td>0.62</td><td>0.71</td><td>0.68</td><td>0.81</td><td>0.70</td><td>0.84</td><td>0.71</td><td>0.87</td></tr><tr><td>Effort [66]</td><td>ICML&#x27;25</td><td>0.73</td><td>0.80</td><td>0.68</td><td>0.73</td><td>0.69</td><td>0.76</td><td>0.80</td><td>0.88</td><td>0.84</td><td>0.92</td><td>0.85</td><td>0.94</td><td>0.88</td><td>0.96</td></tr><tr><td>ForAda [11]</td><td>CVPR&#x27;25</td><td>0.66 0.72</td><td></td><td>0.67</td><td>0.74</td><td>0.68</td><td>0.75</td><td>0.75</td><td>0.83</td><td>0.81</td><td>0.89</td><td>0.84</td><td>0.91 </td><td>0.82</td><td>0.89</td></tr><tr><td>Ours (12-Blk)</td><td></td><td>0.65</td><td>0.79</td><td>0.65</td><td></td><td>0.77 0.62</td><td>0.77</td><td>0.79</td><td>0.89</td><td></td><td>0.86 0.94</td><td>0.91</td><td>0.96</td><td>0.93</td><td>0.97</td></tr></table>

Table S7. Protocol #3 (InSwapper): Region-level performance (ACC/ AUC). Bold = best, underline = second-best (per metric, per column). C-1: eye+nose, C-2: eye+nose+mouth, C-3: eye+nose+mouth+eyebrow.
<table><tr><td rowspan="2">Detector</td><td rowspan="2">Venue</td><td colspan="2">Eye</td><td colspan="2">Nose</td><td colspan="2">Mouth</td><td colspan="2">C-1</td><td colspan="2">C-2</td><td colspan="2">C-3</td><td colspan="2">Full</td></tr><tr><td colspan="2">ACC AUC</td><td colspan="2">ACC</td><td colspan="2">AUC ACC AUC</td><td colspan="2">ACC AUC</td><td colspan="2">ACC AUC</td><td colspan="2">ACC AUC</td><td colspan="2">ACC AUC</td></tr><tr><td>Xception [52]</td><td>ICCV&#x27;19</td><td>0.64</td><td>0.73</td><td>0.79</td><td>0.87</td><td>0.81</td><td>0.88</td><td>0.81</td><td>0.89</td><td>0.86</td><td>0.94</td><td>0.83</td><td>0.91</td><td></td><td>0.93</td></tr><tr><td>F3Net [46]</td><td>ECCV&#x27;20</td><td>0.70</td><td>0.86</td><td>0.84</td><td>0.94</td><td>0.80</td><td>0.91</td><td>0.89</td><td>0.96</td><td>0.92</td><td>0.98</td><td>0.88</td><td>0.96</td><td>0.84 0.86</td><td>0.95</td></tr><tr><td>RECCE [5]</td><td>CVPR&#x27;22</td><td>0.67</td><td>0.79</td><td>0.85</td><td>0.93</td><td>0.86</td><td>0.95</td><td>0.84</td><td>0.93</td><td>0.90</td><td>0.98</td><td>0.88</td><td>0.96</td><td>0.87</td><td>0.95</td></tr><tr><td>SBI [53]</td><td>CVPR&#x27;22</td><td>0.68</td><td>0.79</td><td>0.73</td><td>0.85</td><td>0.61</td><td>0.73</td><td>0.88</td><td>0.96</td><td>0.93</td><td>0.98</td><td>0.92</td><td>0.98</td><td>0.91</td><td>0.97</td></tr><tr><td>LGrad [57]</td><td>CVPR&#x27;23</td><td>0.50</td><td>0.59</td><td>0.50</td><td>0.60</td><td>0.50</td><td>0.60</td><td>0.50</td><td>0.59</td><td>0.49</td><td>0.59</td><td>0.50</td><td>0.58</td><td>0.50</td><td>0.59</td></tr><tr><td>UCF [64]</td><td>ICCV’23</td><td>0.80</td><td>0.91</td><td>0.89</td><td>0.96</td><td>0.88</td><td>0.95</td><td>0.91</td><td>0.97</td><td>0.92</td><td>0.98</td><td>0.91</td><td>0.98</td><td>0.91</td><td>0.97</td></tr><tr><td>FatFormer [37]</td><td>CVPR&#x27;24</td><td>0.52</td><td>0.71</td><td>0.50</td><td>0.69</td><td>0.51</td><td>0.69</td><td>0.53</td><td>0.75</td><td>0.54</td><td>0.75</td><td>0.56</td><td>0.77</td><td>0.66</td><td>0.87</td></tr><tr><td>FreqNet [58]</td><td>AAAI&#x27;24</td><td>0.38</td><td>0.46</td><td>0.38</td><td>0.46</td><td>0.38</td><td>0.46</td><td>0.39</td><td>0.46</td><td>0.39</td><td>0.47</td><td>0.39</td><td>0.47</td><td>0.40</td><td>0.47</td></tr><tr><td>ProDet [8]</td><td>NeurIPS’24</td><td>0.69</td><td>0.78</td><td>0.79</td><td>0.86</td><td>0.78</td><td>0.86</td><td>0.83</td><td>0.90</td><td>0.85</td><td>0.92</td><td>0.83</td><td>0.89</td><td>0.80</td><td>0.87</td></tr><tr><td>DFD-FCG [19]</td><td>CVPR&#x27;25</td><td>0.63</td><td>0.70</td><td>0.61</td><td>0.66</td><td>0.68</td><td>0.79</td><td>0.67</td><td>0.78</td><td>0.72</td><td>0.90</td><td>0.73</td><td>0.91</td><td>0.73</td><td>0.94</td></tr><tr><td>Effort [66]</td><td>ICML’25</td><td>0.73</td><td>0.80</td><td>0.70</td><td>0.76</td><td>0.74</td><td>0.81</td><td>0.81</td><td>0.89</td><td>0.87</td><td>0.95</td><td>0.88</td><td>0.96</td><td>0.89</td><td>0.98</td></tr><tr><td>ForAda [11]</td><td>CVPR&#x27;25</td><td>0.66</td><td>0.74</td><td>0.71</td><td>0.78</td><td>0.75</td><td>0.83</td><td>0.82</td><td>0.88</td><td>0.89</td><td>0.95</td><td>0.89</td><td>0.95</td><td>0.89</td><td>0.94</td></tr><tr><td>Ours (12-Blk)</td><td></td><td>0.62</td><td>0.76</td><td>0.64</td><td>0.77</td><td>0.65</td><td>0.80</td><td>0.77</td><td>0.88</td><td>0.88</td><td>0.95</td><td>0.90</td><td>0.96</td><td>0.94</td><td>0.98</td></tr></table>

Table S8. Protocol #3 (SimSwap): Region-level performance (ACC / AUC). Bold = best, underline = second-best (per metric, per column). C-1: eye+nose, C-2: eye+nose+mouth, C-3: eye+nose+mouth+eyebrow.
<table><tr><td rowspan="2">Detector</td><td rowspan="2">Venue</td><td colspan="2">Eye</td><td colspan="2">Nose</td><td colspan="2">Mouth</td><td colspan="2">C-1</td><td colspan="2">C-2</td><td colspan="2">C-3</td><td colspan="2">Full</td></tr><tr><td colspan="2">ACC AUC</td><td colspan="2">ACC AUC</td><td colspan="2">ACC AUC</td><td colspan="2">ACC AUC</td><td colspan="2">ACC AUC</td><td colspan="2">ACC AUC</td><td colspan="2">ACC AUC</td></tr><tr><td></td><td>ICCV&#x27;19</td><td>0.62</td><td>0.69</td><td>0.63</td><td>0.70</td><td></td><td>0.78</td><td>0.63</td><td>0.71</td><td>0.69</td><td>0.77</td><td></td><td>0.79</td><td></td><td>0.82</td></tr><tr><td>Xception [52] F3Net [46]</td><td>ECCV&#x27;20</td><td>0.60</td><td>0.75</td><td>0.60</td><td>0.75</td><td>0.70 0.64</td><td>0.79 </td><td>0.61</td><td>0.74</td><td>0.63</td><td>0.78 </td><td>0.70 0.64</td><td>0.80</td><td>0.70 0.59</td><td>0.73</td></tr><tr><td>RECCE [5]</td><td>CVPR&#x27;22</td><td>0.63</td><td>0.73</td><td>0.63</td><td>0.73</td><td>0.69</td><td>0.80</td><td>0.65</td><td>0.75</td><td>0.68</td><td>0.80</td><td>0.69</td><td>0.81</td><td>0.61</td><td>0.70</td></tr><tr><td>SBI [53]</td><td>CVPR&#x27;22</td><td>0.58</td><td>0.67</td><td>0.58</td><td>0.68</td><td>0.58</td><td>0.67</td><td>0.67</td><td>0.78</td><td>0.74</td><td>0.86</td><td>0.74</td><td>0.86</td><td>0.82</td><td>0.92</td></tr><tr><td>LGrad [57]</td><td>CVPR&#x27;23</td><td>0.50</td><td>0.58</td><td>0.50</td><td>0.58</td><td>0.50</td><td>0.58</td><td>0.50</td><td>0.58</td><td>0.50</td><td>0.58</td><td>0.50</td><td>0.58</td><td>0.50</td><td>0.58</td></tr><tr><td>UCF [64]</td><td>ICCV&#x27;23</td><td>0.70</td><td>0.83</td><td>0.71</td><td>0.84</td><td>0.72</td><td>0.85</td><td>0.73</td><td>0.85</td><td>0.74</td><td>0.86</td><td>0.77</td><td>0.89</td><td>0.70</td><td>0.83</td></tr><tr><td>FatFormer [37]</td><td>CVPR&#x27;24</td><td>0.51</td><td>0.65</td><td>0.51</td><td>0.67</td><td>0.50</td><td>0.64</td><td>0.53</td><td>0.70</td><td>0.54</td><td>0.72</td><td>0.57</td><td>0.75</td><td>0.75</td><td>0.90</td></tr><tr><td>FreqNet [58]</td><td>AAAI&#x27;24</td><td>0.38</td><td>0.45</td><td>0.38</td><td>0.46</td><td>0.38</td><td>0.46</td><td>0.39</td><td>0.46</td><td>0.39</td><td>0.47</td><td>0.39</td><td>0.47</td><td>0.39</td><td>0.46</td></tr><tr><td>ProDet [8]</td><td>NeurIPS’24</td><td>0.71</td><td>0.79</td><td>0.75</td><td>0.82</td><td>0.74</td><td>0.83</td><td>0.77</td><td>0.84</td><td>0.81</td><td>0.89</td><td>0.81</td><td>0.88</td><td>0.80</td><td>0.88</td></tr><tr><td>DFD-FCG [19]</td><td>CVPR&#x27;25</td><td>0.54 </td><td>0.55</td><td>0.51</td><td>0.51</td><td>0.58</td><td>0.61</td><td>0.56</td><td>0.58</td><td>0.62</td><td>0.68</td><td>0.63</td><td>0.71</td><td>0.68</td><td>0.80</td></tr><tr><td>Effort [66]</td><td>ICML&#x27;25</td><td>0.62</td><td>0.67</td><td>0.61</td><td>0.66</td><td>0.60</td><td>0.65</td><td>0.68</td><td>0.73</td><td>0.71</td><td>0.78</td><td>0.74</td><td>0.81</td><td>0.82</td><td>0.89</td></tr><tr><td>ForAda [11]</td><td>CVPR&#x27;25</td><td>0.600.65</td><td></td><td>0.64</td><td>0.69</td><td>0.63</td><td>0.68</td><td>0.67</td><td>0.73</td><td>0.71</td><td>0.78</td><td>0.73</td><td>0.80</td><td>0.70</td><td>0.76</td></tr><tr><td>Ours (12-Blk)</td><td></td><td>0.54</td><td>0.65</td><td>0.59</td><td>0.72</td><td>0.55</td><td>0.66</td><td>0.65</td><td>0.79</td><td>0.70</td><td>0.83</td><td>0.77</td><td>0.88</td><td>0.84</td><td>0.93</td></tr></table>

Table S9. Protocol #3 (UniFace): Region-level performance (ACC / AUC). Bold = best, underline = second-best (per metric, per column). C-1: eye+nose, C-2: eye+nose+mouth, C-3: eye+nose+mouth+eyebrow.
<table><tr><td rowspan="2">Detector</td><td rowspan="2">Venue</td><td colspan="2">Eye</td><td colspan="2">Nose</td><td colspan="2">Mouth</td><td colspan="2">C-1</td><td colspan="2">C-2</td><td colspan="2">C-3</td><td colspan="2">Full</td></tr><tr><td colspan="2">ACC AUC</td><td colspan="2">ACC AUC</td><td colspan="2">ACC AUC</td><td colspan="2">ACC AUC</td><td colspan="2">ACC AUC</td><td colspan="2">ACC AUC</td><td colspan="2">ACC AUC</td></tr><tr><td>Xception [52]</td><td>ICCV&#x27;19</td><td>0.61</td><td>0.67</td><td>0.67</td><td>0.73</td><td>0.67</td><td>0.74</td><td>0.68</td><td>0.74</td><td>0.71</td><td>0.79</td><td>0.70</td><td>0.78</td><td></td><td>0.81</td></tr><tr><td>F3Net [46]</td><td>ECCV&#x27;20</td><td>0.55</td><td>0.63</td><td>0.57</td><td>0.68</td><td>0.57</td><td>0.69</td><td>0.58</td><td>0.68</td><td>0.60</td><td>0.72</td><td>0.60</td><td>0.71</td><td>0.71 0.67</td><td>0.80</td></tr><tr><td>RECCE [5]</td><td>CVPR&#x27;22</td><td>0.59</td><td>0.67</td><td>0.63</td><td>0.72</td><td>0.61</td><td>0.72</td><td>0.66</td><td>0.75</td><td>0.68</td><td>0.78 </td><td>0.67</td><td>0.77</td><td>0.78</td><td>0.86</td></tr><tr><td>SBI [53]</td><td>CVPR&#x27;22</td><td>0.67</td><td>0.79</td><td>0.60</td><td>0.72</td><td>0.59</td><td>0.71</td><td>0.72</td><td>0.84</td><td>0.77</td><td>0.88</td><td>0.78</td><td>0.89</td><td>0.83</td><td>0.93</td></tr><tr><td>LGrad [57]</td><td>CVPR&#x27;23</td><td>0.50</td><td>0.53</td><td>0.50</td><td>0.53</td><td>0.50</td><td>0.54</td><td>0.50</td><td>0.53</td><td>0.50</td><td>0.53</td><td>0.50</td><td>0.53</td><td>0.50</td><td>0.52</td></tr><tr><td>UCF [64]</td><td>ICCV’23</td><td>0.59</td><td>0.67</td><td>0.60</td><td>0.70</td><td>0.62</td><td>0.73</td><td>0.63</td><td>0.73</td><td>0.65</td><td>0.76</td><td>0.64</td><td>0.75</td><td>0.72</td><td>0.84</td></tr><tr><td>FatFormer [37]</td><td>CVPR&#x27;24</td><td>0.52</td><td>0.63</td><td>0.51</td><td>0.59</td><td>0.51</td><td>0.60</td><td>0.53</td><td>0.69</td><td>0.56</td><td>0.74</td><td>0.60</td><td>0.79</td><td>0.73</td><td>0.89</td></tr><tr><td>FreqNet [58]</td><td>AAAI&#x27;24</td><td>0.38</td><td>0.45</td><td>0.39</td><td>0.46</td><td>0.39</td><td>0.45</td><td>0.39</td><td>0.46</td><td>0.39</td><td>0.46</td><td>0.39</td><td>0.46</td><td>0.38</td><td>0.46</td></tr><tr><td>ProDet [8]</td><td>NeurIPS’24</td><td>0.74</td><td>0.82</td><td>0.76</td><td>0.84</td><td>0.76</td><td>0.83</td><td>0.81</td><td>0.89</td><td>0.83</td><td>0.92</td><td>0.83</td><td>0.91</td><td>0.85</td><td>0.94</td></tr><tr><td>DFD-FCG [19]</td><td>CVPR&#x27;25</td><td>0.64</td><td>0.71</td><td>0.59</td><td>0.63</td><td>0.61</td><td>0.65</td><td>0.69</td><td>0.81</td><td></td><td>0.71 0.87</td><td>0.72</td><td>0.89</td><td>0.72</td><td>0.93</td></tr><tr><td>Effort [66]</td><td>ICML’25</td><td>0.71</td><td>0.77</td><td>0.64</td><td>0.69</td><td>0.65</td><td>0.70</td><td>0.79</td><td>0.86</td><td>0.83</td><td>0.91</td><td>0.85</td><td>0.94</td><td>0.87</td><td>0.96</td></tr><tr><td>ForAda [11]</td><td>CVPR&#x27;25</td><td>0.71</td><td>0.78</td><td>0.73</td><td>0.80</td><td>0.69</td><td>0.77</td><td>0.81</td><td>0.87</td><td>0.85</td><td>0.91</td><td>0.86</td><td>0.93</td><td>0.87</td><td>0.94</td></tr><tr><td>Ours (12-Blk)</td><td></td><td>0.70</td><td>0.84</td><td>0.68</td><td>0.82</td><td>0.60</td><td>0.78</td><td>0.83</td><td>0.92</td><td></td><td>0.88 0.95</td><td>0.91</td><td>0.97</td><td>0.95</td><td>0.98</td></tr></table>

## J. Qualitative Analysis and Failure Cases

Figures S36 and S37–S43 show MaskNet predictions and failure cases of Ours (12-Blk). MaskNet (Fig. S36) highlights manipulated regions, while failures correspond to false negatives, mainly in Eye and Nose, where the manipulated area is <10% and artifacts are subtle. Many Eye failures arise from BlendFace and SimSwap, indicating difficulty due to both region size and subtle artifacts.

## MaskNet Visualization for Cross Dataset Samples

![](images/d89aee07b9b2efa2b2f3b979d1222ea2320916bbc57bd667f34c153bf9aa6e06.jpg)  
Figure S36. MaskNet visualization: predicted forgery responses highlighting manipulated regions.

Eye Region (Ours 12-BIk) Fake predicted as Real

![](images/7068497565ff98518c339da9de9a4aeb342f25c76d3a7959938b22edb2441326.jpg)  
Figure S37. Failure cases: Eye region. Fake images predicted as Real by Ours (12-Blk).

Nose Region (Ours 12-BIk) Fake predicted as Real

![](images/9906d63e99b906e40a20dae424f78f01f3f19981e8ef86991a98e0b2d409a48c.jpg)  
Figure S38. Failure cases: Nose region.

Mouth Region (Ours 12-BIk) Fake predicted as Real

![](images/3c77dbec6d3827dfe2808b2b14914d75b376dba300dccb39854a780e0f80bc17.jpg)  
Figure S39. Failure cases: Mouth region.

Composite-1 Region (Ours 12-BIk) Fake predicted as Real

![](images/899182027d6d910108525c9db40504b216ad9fcbd1f672059c211331126b983a.jpg)  
Figure S40. Failure cases: Composite-1 (Eye+Nose).

Composite-2 Region (Ours 12-BIk) Fake predicted as Real

![](images/11ab24e96948d5f0831dac200d3a3128b6a4ff6a96307f8f7167211a1bcfe25f.jpg)  
Figure S41. Failure cases: Composite-2 (Eye+Nose+Mouth)

Composite-3 Region (Ours 12-BIk) Fake predicted as Real

![](images/577de3f7d9c7a299ba8341ddba80d11e65b6d331f4387fa563bde6de841ae3b8.jpg)  
Figure S42. Failure cases: Composite-3 (Eye+Nose+Mouth+Eyebrow).

![](images/3275d95cca6acc00bc034193ca917bdf4817e69c4734dde67762cc25b02e1732.jpg)  
Figure S43. Failure cases: Full Face region.