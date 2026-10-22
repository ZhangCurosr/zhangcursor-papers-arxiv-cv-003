# Spatially Gated Demographic Steering for Debiasing Multi-Person Text-to-Image Generation

Venkatesh Thirugnana Sambandham Technische Hochschule Ingolstadt Ingolstadt, Germany

venkatesh.thirugnanasambandham@thi.de

Torsten Schon¨ Technische Hochschule Ingolstadt Ingolstadt, Germany torsten.schoen@thi.de

## Abstract

Text-to-Image models reproduce societal stereotypes, and existing mitigation applies one demographic correction to the whole scene. In multi-person scene generation this creates a trade-off: global steering improves aggregate balance only by homogenizing the people within an image. Steering-based methods like Embedding Arithmetic drive the intra-image diversity below the unsteered baseline. Text- and latent-projection approaches shift the aggregate skew with only marginal gains in intra-image diversity. We present Spatially Gated Steering (SGS), which steers the demographics of each person independently by extracting per-person masks from the Diffusion Transformer’s (DiT) internal attention maps at a critical denoising timestep and applying a distinct demographic vector within each mask for the remaining steps. SGS needs no fine-tuning, inversion or layout specifications, and uses a lightweight segmentor over frozen DiT attention features to localize subjects at early timesteps. On skewed occupational prompts, SGS reaches high intra-image racial and gender diversity while keeping aggregate balance competitive and image quality near baseline, a result we validate on two architecturally distinct diffusion transformers, FLUX.1-Dev and SD3.5-Large. Codebase and model weights are available at https://github.com/cvims/spatiallygated-steering

## 1. Introduction

Text-to-Image (T2I) generation has transformed digital content creation, yet foundational models such as FLUX [24] and Stable Diffusion [10] systematically reproduce societal stereotypes [9]. Gender-neutral prompts like a carpenter yield predominantly male-presenting images; a nurse yields predominantly female-presenting images [28]. Studies show these outputs shift user beliefs about real-world distributions [14], making fairness mitigation essential for

![](images/0e3c49cee71ca86730a4227d0ee789673ae226de513f0230e1df4175ca1637fe.jpg)  
Figure 1. SGS intuition. The base trajectory (black) reaches x<sup>base</sup><sub>0</sub> in a stereotyped region of p<sub>data</sub>; global mitigation [39] (brown dashed) shifts it to one alternative mode. SGS instead intercepts at x<sub>start</sub>, extracts per-person masks, and steers each individual independently (orange/blue dashed) toward distinct demographic attractors within the same image x<sub>0</sub>.

responsible deployment.

Several mitigation strategies have been proposed and they operate at different levels of the diffusion pipeline [37]: They modify the text embeddings [6, 22, 39, 43] or fine-tune model weights [13, 36] to shift the aggregate output distribution. While effective for scenes with a single person, global mitigation methods share a fundamental limitation: They apply identical steering to every person in a scene. A prompt like ”a group of nurses” receives one demographic vector, pushing all the persons in the image in the same direction and failing to diversify individuals independently.

This limitation is not merely theoretical: Wan and Chang [41] show that multi-person prompts expose a distinct, compounding class of bias that single-subject evaluations cannot capture. Co-occurring subjects amplify stereotypes even when the same model behaves anti-stereotypically on

![](images/3e9ce1350a1db9a2c3485df20b09ad8fabab765dc19ea2f7ca162206ee28d527.jpg)  
Figure 2. Qualitative comparison across three profession types spanning different stereotype directions. Prompts contain no demo graphic descriptors. Base is demographically homogeneous; EA shifts everyone uniformly; FairImagen is statistically indistinguishable from baseline on FLUX; SGS steers each instance toward a distinct intersectional target while preserving occupational context and scene coherence. Rightmost column: segmentation masks (nearest-neighbor upsampled from 32 × 32) with assigned demographic labels.

single-person prompts.

Attempts to address this gap remain limited. Existing approaches either rewrite the generation prompt with a single corrective signal applied to the whole scene [8, 41], optimize demographic distributions at the aggregate level without per-instance control [18], or require an explicit spatial layout as input [7]. To the best of our knowledge, no existing method localizes individuals from the model’s own internal representations and applies independent demographic steering, to achieve intra-image diversity. Figure 1 illustrates this gap: while global mitigation shifts all persons uniformly toward a single demographic mode, our approach steers each individual independently through their spatial mask, reaching distinct attractors within p<sub>data</sub> in a single denoising pass.

We introduce Spatially Gated Steering (SGS), a framework for per-instance demographic steering in multi-person T2I scene generation. SGS operates by 1) Identifying and generating the masks of instances of people during the intermediate denoising steps using a lightweight detector, 2) Steering the demographics of the detected individuals, while also maintaining concept coherence.

Our contributions are as follows:

• A trade-off in multi-person fairness. Global mitigation faces an inherent tension: improving aggregate balance requires collapsing intra-image diversity, sometimes below the unsteered baseline. Since any per-instance diversification trivially raises intra-image entropy, the meaningful result is high entropy at a competitive aggregate balance. SGS reaches high intra-image diversity with a competitive aggregate balance by steering per-instance without fine-tuning, flow-inversion or layout.

• Instance segmentation from DiT features. We show that self-attention maps of DiTs at a critical denoising timestep encode spatially separable person instances, and train a lightweight DETR-style [4] segmentor on these frozen features that isolates individuals across crowd sizes and scene complexities.

• Benchmark and analysis. We introduce a benchmark measuring fairness along both intra-image diversity and aggregate deviation, surfacing the homogenization failure mode of global methods and showing SGS is strong on both axes, on both FLUX.1-Dev and SD3.5-Large.

## 2. Related Works

Bias and Fairness in Text-to-Image Models. T2I models have been extensively shown to amplify demographic stereotypes encoded in training data, producing outputs skewed by gender, race and cultural identity across a wide range of prompt types [1, 5, 28, 35]. Beyond documenting these skews, a parallel line of work formalizes their measurement: Dehdashtian et al. [9] quantify stereotypical attributes in the generated set and trace their origin within the model, reporting that even high-fidelity systems such as FLUX retain strong stereotypical predispositions. Mitigation strategies operate at different points in the generation pipeline: prompt engineering injects demographic descriptors at inference [2, 11, 43], embedding manipulation steers text representations prior to image generation [6, 15, 22, 25, 26, 39], and parameter-efficient fine-tuning edits model weights to shift output distributions [13, 36]. Park et al.’s Entanglement-Free Attention [30] shares our goal of preserving non-target attributes during bias mitigation, training cross-attention predictors under a mask-constrained regularization loss so background edits are suppressed. However, like other global methods, it applies one scene-level attribute correction rather than steering each person independently. Despite this breadth, all existing methods treat the generated image as a single unit, applying one steering signal to the entire scene. This is sufficient when a prompt produces one person, but breaks down entirely for multiperson scenes where distinct individuals require independent control.

Multi-person Bias and Instance-level Control. This breakdown in multi-subject generation is primarily driven by semantic leakage, a phenomenon where the attention mechanisms of diffusion architectures inadvertently blend the visual and demographic features of different subjects during the denoising trajectory. Wan and Chang [41] demonstrate the consequences of this at scale: their Paired Stereotype Test shows that over 74% of DALL-E 3 images depicting two individuals with contrastive social identities display gender-occupational biases, even when the same model produces anti-stereotypical outputs for single-person prompts. Jung et al. [18] propose a Multi-Group Proportional Representation metric to measure and optimize demographic distributions across intersectional groups, but operate at the aggregate level without per-instance control. A related direction targets stereotypes that emerge from the co-occurrence of multiple concepts within a single prompt: Zhou et al. [47] show that pairing a demographic descriptor with an unrelated object can induce stereotypi cal associations invisible to single-concept audits, mitigated via a trained Sensitive Transformer that constrains the output distribution toward a stereotype-free target. This targets a distinct problem from ours, stereotyped associations between a subject and its surrounding context, corrected globally over the whole generation, rather than diversifying multiple people already explicitly present in a scene. Training-free inference methods such as Bounded Attention [7] attempt to prevent semantic leakage by confining attention flows within manually predefined spatial bounding boxes, but rely on rigid user-provided layouts rather than organic scene composition. Reinforcement learning approaches such as DisCo [3] directly fine-tune models to penalize intra-image facial similarity, but use face similarity as a proxy for demographic diversity rather than steering specific attribute vectors, and incur significant computational overhead. Global steering techniques such as Embedding Arithmetic [39] modify text embeddings uniformly, which fundamentally fails to produce heterogeneous intraimage diversity.

Diffusion Features for Spatial Understanding. Intermediate diffusion activations encode rich spatial structure without additional training: DIFT [38] shows middenoising U-Net features support zero-shot dense correspondence; ODISE [42] uses Stable Diffusion U-Net features for open-vocabulary panoptic segmentation. This extends to Diffusion Transformers only recently: ConceptAttention [16] shows DiT attention-layer projections yield sharper concept saliency than cross-attention alone; TRACE [17] shows diffusion models isolate object boundaries at a specific timestep, enabling spatial masks from internal attention. SGS extends this line of work to intraimage demographic fairness in multi-person scenes generated by T2I models.

## 3. Preliminaries

## 3.1. Rectified Flow Denoising

Rectified flow [27] learns a velocity field v<sub>θ</sub> (a neural network with parameters θ) that transports samples from a Gaussian noise distribution to the data distribution along approximately straight trajectories. FLUX [24] adopts this formulation, initializing generation from $x _ { 1 } \sim \mathcal { N } ( 0 , I )$ and iteratively denoising via:

$$
x _ { t - \Delta t } = x _ { t } \ : - \ : \Delta t \cdot v _ { \theta } ( x _ { t } , t , \mathbf c ) ,\tag{1}
$$

where $t \in [ 0 , 1 ]$ decreases from 1 (pure noise) to 0 (clean image) over T steps, $\Delta t = 1 / T$ , and c is the encoded text condition [31].

## 3.2. Multimodal Diffusion Transformer (MM-DiT)

The MM-DiT [10] architecture patchifies the noisy image latent $\boldsymbol { x } _ { t } \in \mathbb { R } ^ { H W \times d }$ , where $H W$ are the latent height and width and d is the token dimension, into image tokens, concatenates them with text conditioning tokens c [31], and processes them through a stack of N transformer layers, each applying joint image-text self-attention and a feedforward MLP. FLUX.1-Dev [24] uses $N { = } 1 9$ double-stream layers; SD3.5-Large [10] uses N=38 joint transformer blocks.

## 3.3. Embedding Arithmetic

Embedding Arithmetic [39] computes a demographic attribute vector as the mean difference between embeddings generated by a biased prompt $( p _ { \mathrm { t a r g e t } } )$ and neutral prompts $( p _ { \mathrm { n e u t r a l } } )$ over a set of profession prompts $\mathcal { P } \mathrm { : }$

$$
\mathbf { v } _ { A } = \frac { 1 } { | \mathcal { P } | } \sum _ { p \in \mathcal { P } } \bigl ( \mathbf { e } ( p _ { \mathrm { t a r g e t } } ) - \mathbf { e } ( p _ { \mathrm { n e u t r a l } } ) \bigr ) ,\tag{2}
$$

where ${ \bf e } ( \cdot ) \in \mathbb { R } ^ { D }$ denotes the text embedding and A indexes the target demographic attribute. FLUX conditions denoising on two text encoders: a pooled CLIP-L/14 embedding (768-dim) that modulates the network globally, and a T5-XXL sequence embedding that carries the prompt text via cross-attention. Embedding Arithmetic’s direction vectors are computed and validated in the CLIP pooled space $- \mathbf { e } ( \cdot )$ and c in Eq. (1) denote this pooled CLIP embedding, not the T5 sequence. $p _ { \mathrm { t a r g e t } }$ and $p _ { \mathrm { n e u t r a l } }$ share an identical template except for an inserted demographic descriptor (e.g. “a photo of a {demographic} {profession}” vs. “a photo of a {profession}”), so $\mathbf { v } _ { A }$ isolates the demographic direction rather than any profession-specific signal. At inference, the original method shifts the conditioning vector globally as $\tilde { \mathbf { c } } = \mathbf { c } + \boldsymbol { \alpha } \cdot \mathbf { v } _ { A }$ , where $\alpha > 0$ controls the steering strength.

## 4. Methods

SGS performs per-instance demographic steering within the standard rectified flow denoising loop, requiring no model fine-tuning, no trajectory inversion, and no external spatial layout specification. Crucially, instance masks are predicted directly from the MM-DiT’s attention features at a critical denoising step, rather than decoding the noisy latents for an external segmentor - which would be unreliable while the latent is still far from $x _ { 0 }$ , and redundant since these features are already computed by the base pass. Figure 1 illustrates the intuition behind our approach and Algorithm 1 summarizes the complete inference procedure. The denoising loop is divided into three phases governed by two hyperparameters: $t _ { \mathrm { s t a r t } }$ and $t _ { \mathrm { e n d } }$ . Before $t _ { \mathrm { s t a r t } }$ , generation proceeds without intervention. $\mathrm { A t } t _ { \mathrm { s t a r t } }$ , instance masks are extracted and cached. Between $t _ { \mathrm { s t a r t } }$ and $t _ { \mathrm { e n d } } .$ , a separate steered forward pass is run per detected instance at each denoising step, and the predictions are spatially accumulated. After $t _ { \mathrm { e n d } }$ , standard denoising resumes to completion.

## 4.1. Instance Detection

At step $t = t _ { \mathrm { s t a r t } } .$ , the base forward pass is run with attention extraction enabled, returning the MM-DiT self-attention maps $\{ \mathrm { S A } _ { t } ^ { \ell } \} _ { \ell = 1 } ^ { N }$

Algorithm 1 SGS Inference   
Require: Noisy image latent $x _ { 1 } .$ , prompt embeddings (c), at  
tribute vectors $\{ \mathbf { v } _ { A } \}$ , strength α, λ, window $[ t _ { \mathrm { s t a r t } } , t _ { \mathrm { e n d } } )$   
1: for $t = 0 , \ldots , T - 1$ do   
2: $\hat { v } ^ { \mathrm { b a s e } } \gets v _ { \theta } ( x _ { t } , t , \mathbf { c } )$ ▷ base pass   
3: if $t = t _ { \mathrm { s t a r t } }$ then   
4: $\mathbf { F } _ { t } \gets \mathrm { B i n A t t n } ( \{ \mathrm { S A } _ { t } ^ { \ell } \} )$ ▷ Eq. (3)   
5: $\{ M _ { t } \} , K \gets \mathrm { S e g m e n t o r } ( \mathbf { F } _ { t } )$ ▷ Eq. (4)   
6: $\{ A _ { t } \} \gets \mathrm { A s s i g n D e m o g r a p h i c s } ( K )$   
7: end if   
8: i $\mathrm { ~ f ~ } t _ { \mathrm { s t a r t } } \leq t < t _ { \mathrm { e n d } }$ and $K > 0$ then   
9: $\hat { v } ^ { \mathrm { f i n a l } }  \hat { v } ^ { \mathrm { b a s e } }$   
10: for $i = 1 , \ldots , K$ do   
11: $\tilde { \mathbf { c } } ^ { ( i ) } \gets \mathbf { c } ^ { ( i ) } + \boldsymbol { \alpha } \cdot \mathbf { v } _ { A _ { i } }$ ▷ Eq. (6)   
12: $\hat { v } ^ { ( i ) }  v _ { \theta } ( x _ { t } , t , \tilde { \mathbf { c } } ^ { ( i ) } )$ ▷ Eq. (7)   
13: $\hat { v } ^ { \mathrm { f i n a l } } + = \dot { \lambda } ( \hat { v } ^ { ( i ) } - \hat { v } ^ { \mathrm { b a s e } } ) \odot M _ { i }$ ▷ Eq. (9)   
14: end for   
15: else   
16: $\hat { v } ^ { \mathrm { f i n a l } }  \hat { v } ^ { \mathrm { b a s e } }$   
17: end if   
18: $x _ { t - \Delta t } \gets x _ { t } + ( t _ { \mathrm { p r e v } } - t _ { \mathrm { c u r r } } ) \cdot \hat { v } ^ { \mathrm { f i n a l } }$ ▷ Eq. (10)   
19: end for   
20: return x<sub>0</sub>

Self-attention maps. The self-attention map for layer ℓ is computed independently for each attention head and averaged across heads, capturing pairwise spatial affinities between image tokens [17]; the full expression is given in supplementary Section B.

Layer binning. In order to save compute and memory usage, rather than operating on all N transformer layers independently, we partition them into B=4 nonoverlapping bins of approximately 5 consecutive layers each for FLUX.1-Dev and 9–11 layers each for SD3.5- Large. We then compute the layer-wise mean of the selfattention maps within each bin, yielding four spatially resolved feature maps $\lbrace \widehat { \mathrm { S A } } _ { t } ^ { b } \rbrace _ { b = 1 } ^ { 4 }$ per timestep. Through an ablation study over all bin combinations (Section 5.3), we find that concatenating bins b=3 and b=4 yields the best instance segmentation performance. We therefore define the feature representation as:

$$
\mathbf { F } _ { t } = \widehat { \mathrm { S A } } _ { t } ^ { 3 } \Vert \widehat { \mathrm { S A } } _ { t } ^ { 4 } \in \mathbb { R } ^ { H W \times 2 H W } ,\tag{3}
$$

where ∥ denotes channel-wise concatenation. $\mathbf { F } _ { t }$ is passed directly to the instance segmentor described below.

Segmentor Architecture. $\mathbf { F } _ { t }$ is passed to a DETR-style [4] instance segmentor. The model comprises a three-layer CNN backbone that projects $\mathbf { F } _ { t } \in \mathbb { R } ^ { H \dot { W } \times 2 H W }$ into a spatial feature map, followed by a linear pixel projection that yields a sequence of pixel embeddings $\mathbf { P } ^ { \mathsf { ^ { - } } } \in \mathsf { \mathbb { R } } ^ { H W \times D _ { \epsilon } }$ where $D _ { e }$ is the transformer embedding dimension. A set of $N _ { q }$ learnable instance queries $\mathbf { Q } \in \mathbb { R } ^ { N _ { q } \times D _ { e } }$ are refined by a

L-layer transformer decoder attending over P. The denoising timestep t is incorporated via a sinusoidal embedding concatenated to the input features, conditioning the segmentor on the noise level at which features were extracted. Each refined query produces three outputs via dedicated linear heads:

$$
{ \mathrm { c o n f } } _ { i } \in [ 0 , 1 ] , \quad { \mathrm { c e n t e r } } _ { i } \in [ 0 , 3 2 ] ^ { 2 } , \quad M _ { i } \in [ 0 , 1 ] ^ { 3 2 \times 3 2 } ,\tag{4}
$$

where conf is the instance confidence, center is the predicted center coordinate, and $M _ { i }$ is the instance mask produced via a dot product between the query mask embedding and the pixel embedding sequence. Queries with confidence below a threshold τ are discarded, yielding $K \leq N _ { q }$ active instances. More details, along with the hyperparameters of the segmentor can be found in the supplementary material (Section B).

Training Data. The segmentor is trained on a purposebuilt dataset of 1440 generated multi-person images spanning $k { \in } \{ 2 , 3 , 4 , 5 \}$ persons across 9 scene types and 3 difficulty levels, with instance pseudo-labels produced by SAM2 [32] automatic mask generation. Full dataset details and the train/val split are provided in the supplementary material.

Training. The segmentor is trained with a Hungarian matching loss [4]. For each image, $N _ { q }$ predicted queries are optimally assigned to K ground-truth instances by minimising a cost matrix.

$$
\mathcal { C } _ { i j } = w _ { c } \frac { | \hat { c } _ { i } - c _ { j } | _ { 2 } } { 3 2 } - \log \hat { p } _ { i } + w _ { m } \left( 1 - \operatorname { D i c e } ( \hat { M } _ { i } , M _ { j } ) \right)
$$

where $\hat { c } _ { i } , \hat { p } _ { i } , \hat { M } _ { i }$ are the predicted center, confidence, and mask for query $i , c _ { j } , M _ { j }$ are the ground-truth center and mask, and $w _ { c } , w _ { m }$ weight the center and mask loss terms (values in supplementary Section B). Unmatched queries are supervised with a no-instance label. The mask head is trained with Dice loss [29] and binary cross-entropy on the matched pairs.

Mask caching and demographic assignment. The K binary masks $\{ M _ { i } \} _ { i = 1 } ^ { K }$ are cached and reused for all subsequent steps in the steering window without re-extraction. Each instance i is assigned a target demographic attribute $A _ { i }$ by uniform sampling over the intersection of gender and racial attribute categories, which can be overridden at inference time to reflect user-specified fairness targets.

## 4.2. Per-Instance Steering

For each timestep t in the window $t _ { \mathrm { s t a r t } } \leq t \leq t _ { \mathrm { e n d } }$ , the base velocity prediction $\hat { v } _ { \theta }$ is first obtained from the standard forward process:

$$
\hat { v } _ { \theta } ^ { \mathrm { b a s e } } = v _ { \theta } ( x _ { t } , t , \mathbf { c } )\tag{5}
$$

where c is the conditional embeddings from the text encoder. Rather than the single global shift Embedding Arithmetic applies (Section 3.3), SGS shifts the conditioning vector independently per detected instance:

$$
\begin{array} { r } { \tilde { \mathbf { c } } ^ { ( i ) } = \mathbf { c } ^ { ( i ) } + \alpha \cdot \mathbf { v } _ { A _ { i } } , } \end{array}\tag{6}
$$

where $\alpha > 0$ controls the steering strength and $A _ { i }$ is the demographic attribute assigned to the instance i. SGS precomputes $\{ \mathbf { v } _ { A } \}$ offline for a set of gender and racial attributes and applies Eq. 6 independently per detected person within the steering window described in Section 5.4.

Fair Forward Pass. For each instance $i \in \{ 1 , \ldots , K \}$ , a separate fair forward pass is run with the conditioning embedding c˜ steered via Eq. (6):

$$
\hat { v } _ { \theta } ^ { ( i ) } = v _ { \theta } \Big ( x _ { t } , t , \tilde { \mathbf { c } } ^ { ( i ) } \Big )\tag{7}
$$

Spatially Gated Accumulation The per-instance steering vector is the difference between the fair and base predictions:

$$
\delta ^ { ( i ) } = \hat { v } _ { \theta } ^ { ( i ) } - \hat { v } _ { \theta } ^ { \mathrm { b a s e } }\tag{8}
$$

The steered velocity is accumulated across all K instances, with each steering vector spatially gated by the corresponding cached mask $M _ { i }$ of the instance:

$$
\hat { v } _ { \theta } ^ { \mathrm { f i n a l } } = \hat { v } _ { \theta } ^ { \mathrm { b a s e } } + \lambda \sum _ { i = 1 } ^ { K } \delta ^ { ( i ) } \odot M _ { i }\tag{9}
$$

where $\lambda > 0$ is the diversity strength and $\odot$ denotes the element-wise multiplication with $M _ { i }$ broadcast over the feature dimension. The latent is then updated via Eq. (1).

$$
\begin{array} { r } { x _ { t - \Delta t } = x _ { t } + ( t _ { \mathrm { p r e v } } - t _ { \mathrm { c u r r } } ) \cdot \hat { v } _ { \theta } ^ { \mathrm { f n a l } } . } \end{array}\tag{10}
$$

Outside the steering window, the latent is updated using the base prediction only and no attention extraction, fair forward pass, or mask application occurs.

## 5. Experiments

## 5.1. Implementation Details

Model and Hardware Configurations. Our primary experiments use FLUX.1-Dev<sup>1</sup> in bfloat16 precision [19] on a single NVIDIA A6000 GPU (48GB VRAM), with 21 denoising steps and a guidance scale of 4.5. To evaluate cross-architecture generalization, we additionally port SGS and all four baselines to SD3.5-Large<sup>2</sup> in bfloat16 precision, run on a single NVIDIA A100 GPU (39GB MIG slice), at 1024×1024 resolution (its native training resolution) with 28 denoising steps and a guidance scale of

![](images/013a0c4e007e2df9cd05c8a4cd102f90d36bd8d9590e58842239765f6d7798a9.jpg)  
Figure 3. Instance segmentation analysis (n=720): segmentation performance across the FLUX denoising trajectory for four architectures. All methods peak around t≈0.7–0.8; DETR consistently outperforms all baselines. Layer bin ablation motivating our choice of Bins 3+4 as $\mathbf { F } _ { t }$ (Eq. (3)) is in supplementary Section B.

3.5. SGS’s instance-detection and per-instance steering mechanism (Section 4.1, Section 3.3) makes no FLUXspecific assumption beyond the existence of a joint imagetext self-attention map inside the denoising transformer, which SD3.5’s MM-DiT also provides. SD3.5-Large’s EA direction vectors live in the joint pooled embedding of its two CLIP encoders (CLIP-L and CLIP-G, concatenated to 2048-dim), rather than FLUX’s single 768-dim CLIP-L pooled space; T5 remains excluded from vector construction in both cases.

## 5.2. Evaluation Metrics

Intersectional Entropy $( H _ { i x } )$ Let ${ \mathcal { G } } { = } { \mathcal { R } } \times { \mathcal { S } }$ denote the set of intersectional demographic groups over racial categories R={white, black, asian, indian} and genders ${ \boldsymbol { S } } =$ {male, female}, giving |G| = 8 groups<sup>3</sup>. For a single generated image with n detected faces, let $p _ { g }$ be the ratio of faces classified as group $g .$ We compute per-image intersectional entropy as:

$$
H _ { i x } = - \frac { 1 } { \log \operatorname* { m i n } ( n , | \mathcal { G } | ) } \sum _ { g \in \mathcal { G } } p _ { g } \log p _ { g } \ \in [ 0 , 1 ] ,\tag{11}
$$

and report its mean over all generated images. Normalizing the entropy with log min(n, |G|) ensures that a $k { = } 2$ image in which both persons belong to a distinct intersectional groups achieves $H _ { i x } { = } 1 . 0$ , independent of k. This captures intra-image demographic diversity which is the core objective of Spatially Gated demographic Steering, rather than aggregated statistics.

Deviation Ratio (DR) Following [25, 40], for an attribute with category set C let $N _ { c }$ be the total number of detected faces classified as category c and $\begin{array} { r } { N = \sum _ { c } N _ { c } ; } \end{array}$

$$
\mathit { D R } = \operatorname* { m a x } _ { c \in \mathcal { C } } \ \frac { \left| \ N _ { c } / N - 1 / | \mathcal { C } | \right| } { 1 - 1 / | \mathcal { C } | } ,\tag{12}
$$

where $D R { = } 0$ is perfectly balanced and $D R { = } 1$ is maximally skewed. We report $D R _ { \mathrm { r a c e } } \ ( | { \mathcal { C } } | = 4 ;$ white, black, asian, indian) and $D R _ { \mathrm { g e n d e r } } \ : ( | { \mathcal { C } } | = 2 { : }$ male, female). Unlike the original single-person formulation, we aggregate facelevel counts across all k persons per image, which we find more appropriate for multi-person generation.

Image Quality We report CLIP score [31] for prompt fidelity and aesthetic score delta (Aes∆) relative to the unsteered baseline [33]. Face crops from the generated images are extracted using MTCNN [44]. Similar to prior works [39, 41], Demographic labels are produced by FairFace [20] detector, which we validate directly against known ground truth rather than assuming its accuracy (supplementary Section E).

## 5.3. Instance Segmentation

Evaluation Protocol. We evaluate instance segmentation on a held-out test set of our dataset as described in Section 4.1. The test set has $n { = } 7 2 0$ images across all $k \in$ $\{ 2 , 3 , 4 , 5 \}$ person counts and three difficulty levels. We report mean intersection-over-union (mIoU), average precision at IoU threshold (AP50) and average precision averaged over thresholds 50%–95% (AP50-95). Features are extracted at each denoising timestep $t \in [ 0 , 1 ]$ and we report performance across the full temporal sweep.

Architectural Comparison We compare four segmentation approaches of increasing supervisory complexity: K-Means (naive unsupervised baseline), Spectral clustering (following MaskDiffusion [21]), CenterNet [48] (supervised, lightweight), and our DETR-style model (Section 4.1). TRACE [17], which also extracts instance boundaries from diffusion self-attention maps, is a natural comparison, but its authors had not released training or inference code at submission time, precluding a faithful reproduction.

Figure 3 reports performance across FLUX denoising timesteps. All methods improve as denoising progresses toward t≈0.7–0.8 (step 4 of 21), then plateau or degrade as features shift toward fine-grained texture, confirming MM-DiT features become instance-discriminative at intermediate timesteps [17, 38]; together with the steering-window ablation (Section 5.4), this motivates $t _ { \mathrm { s t a r t } } { = } 4 .$ DETR outperforms all baselines at every timestep, peaking at mIoU 0.664, AP50 0.797, AP50–95 0.416 (vs. CenterNet’s 0.618/0.762/0.337), reflecting the gap between supervised and unsupervised instance separation from DiT features. A DETR trained analogously on SD3.5-Large reaches a comparable peak (Section F), so this timestep-localized discriminability is not FLUX-specific. Per-difficulty and qualitative detail: supplementary Sections C, G; Section D shows the segmentor is not the bottleneck behind SGS’s steering accuracy at $K { \le } 3 ,$ but shows weaker performance at $K { = } 4$

Layer Bin Ablation. Performance improves monotonically from Bin 1 to Bin 4, reflecting the progression of spatial structure across MM-DiT layer depth - earlier layers encode coarse layout while later layers capture finegrained instance boundaries. Concatenating Bins 3 and 4 achieves the best performance across all metrics (mIoU: 0.631, AP50: 0.747, AP50-95: 0.368, averaged across all timesteps), outperforming other bin configurations (full perbin table in supplementary Section B). This result directly motivates our choice of $\mathbf { F } _ { t } = \widehat { \mathrm { S A } } _ { t } ^ { 3 } \| \widehat { \mathrm { S A } } _ { t } ^ { 4 }$ in Eq. (3).

## 5.4. Steering Parameters Ablation

We evaluate on a set of 8 occupations: 4 male-skewed (lawyer, physician, developer, politician) and 4 femaleskewed (social worker, hairdresser, cashier, receptionist), generating 10 images per occupation for $k { \in } \{ 2 , 3 \}$ persons with randomly assigned target demographics per detected instance. We report five metrics: race and gender steering accuracy $( \operatorname { A c c } _ { \mathrm { r a c e } } , \operatorname { A c c } _ { \mathrm { g e n d e r } } )$ against the assigned target; aesthetic score delta $( A e s \Delta )$ [33] relative to the unsteered baseline; background preservation $\mathrm { ( L P I P S _ { b g } }$ [45]) outside the instance masks; and prompt–image CLIP similarity delta (CLIP∆), where values closer to zero indicate less semantic drift.

Steering Hyperparameters. We ablate EA strength $\alpha \in$ {1.5, 2.0, 2.5, 3.0, 3.5} and spatial diversity strength $\lambda \ \in$ $\{ 3 . 0 , 4 . 0 , 5 . 0 , 6 . 0 , 7 . 0 \}$ , reporting steering accuracy, aesthetic delta, background preservation, and CLIP similarity delta. We select $\alpha { = } 2 . 0$ and $\lambda { = } 4 . 0$ as our steering parameters; full sensitivity analysis is provided in Table S5 of the supplementary material.

Steering Window. We additionally sweep detection/steering start step $t _ { \mathrm { s t a r t } } \in \{ 2 , 4 , 6 \}$ and end step $t _ { \mathrm { e n d } }$ with $\alpha { = } 2 . 0$ and $\lambda { = } 4 . 0$ fixed (full grid and discussion in supplementary Section H, Table S6). Early detection $( t _ { \mathrm { s t a r t } } { = } 2 )$ yields the highest steering accuracy but degrades background preservation and aesthetic quality, since MM-DiT features this early are insufficiently structured for reliable instance segmentation. Late detection $( t _ { \mathrm { s t a r t } } { = } 6 )$ preserves background coherence but leaves too short a window to shift demographics effectively. We select $t _ { \mathrm { s t a r t } } { = } 4$ $t _ { \mathrm { e n d } } { = } 1 0$ as our operating point for FLUX, balancing steering accuracy against background preservation and semantic drift. Similar Hyperparameter testing was done for SD3.5-Large and the results can be found in Section I.

## 5.5. Benchmark Comparison

We evaluate on 30 held-out WinoBias occupations (disjoint from the 10 used to fit EA’s direction vectors) $\times 5$ prompt templates × $k \in \{ 2 , 3 , 4 \} \times 1 0$ seeds, on both FLUX.1-Dev and SD3.5-Large (Section 5.1; full list in supplementary Section J), yielding 4,500 images/method/architecture. We compare against Embedding Arithmetic [39], SDID [25], RespoDiff [40], and FairImagen [12]; DisCo [3] targets a related goal but requires fine-tuning and has no public implementation, precluding a faithful comparison. Each baseline was originally designed for UNet-based diffusion models; how we faithfully reimplemented them for MM-DiT is detailed in supplementary Section N. $H _ { i x } .$ , CLIP, and $A e s \Delta$ are mean ±95% CI (normal approximation); DR uses a percentile bootstrap (1000 resamples), since it is a populationlevel statistic rather than a per-image mean.

Table 1 reports intra-image diversity $( H _ { i x } )$ against aggregate balance (DR) on both architectures under one protocol. Since $H _ { i x }$ rewards any per-instance diversification, the two must be read jointly: global methods steer every person identically and so cannot raise intra-image diversity. Embedding Arithmetic reaches the lowest $D R _ { r a c e }$ on both models, but its $H _ { i x }$ falls well below the unsteered baseline on both – a single demographic vector pushes everyone toward the same group, improving aggregate counts while erasing intra-image variation. SDID, RespoDiff, and Fair-Imagen leave DR close to baseline on both architectures, confirming that global latent and text-embedding projections do not transfer to either model’s joint-attention formulation. SGS is the only method to break this trade-off on both: highest $H _ { i x }$ by a wide margin, $D R _ { r a c e }$ second only to EA, and prompt fidelity near baseline throughout. This pattern replicates on a second, architecturally distinct diffusion transformer, not just FLUX.

## 5.6. Human Evaluation

The metrics above rely on an automatic race/gender classifier (Section 5.1). To check that SGS’s gains are also visible to real people, not just to one automatic tool, we ran a small anonymous online study. Raters were shown two images side by side, one from SGS and one from a baseline, and asked which has more diverse people and which better matches the prompt, plus an overall preference. We collected 65 completed sessions (657 screens, 1971 judgments in total), comparing SGS against all five baselines.

Table 2 shows the share of judgments that favored SGS. Raters preferred SGS on diversity by a wide margin against every baseline (88–90%, all differences statistically significant). On matching the prompt, results are mostly a tie, except SGS is clearly preferred over EA (89%). The overall preference question shows the same pattern: SGS is clearly favored over EA (76%) and roughly tied with the rest. Agreement between raters (Krippendorff’s alpha, where 1.0 means perfect agreement and 0 means no better than chance) was 0.73 for diversity – strong agreement – and lower, 0.43–0.44, for the more subjective faithfulness/preference questions, which is expected for personaltaste judgments. Full study design, recruiting, and perquestion detail are in the supplementary material (Sec-

Table 1. Quantitative results on the extended WinoBias occupation benchmark (30 occupations $\times 5$ templates × $k \in \{ 2 , 3 , 4 \} \times$ 10 seeds, n=4,500/method/architecture), on FLUX.1-Dev and SD3.5-Large. $H _ { i x } \colon$ intra-image intersectional entropy. DR: deviation ratio from uniform (lower is better). Bold and Underlined values represent the best and second-best methods per metric, within each architecture.
<table><tr><td></td><td colspan="5">FLUX.1-Dev</td><td colspan="5">SD3.5-Large</td></tr><tr><td>Method</td><td> $H _ { i x } \uparrow$ </td><td>CLIP↑</td><td> $A e s \Delta \uparrow$ </td><td> $\mathrm { D R } _ { \mathrm { r a c e } } \downarrow$ </td><td> $\mathrm { D R } _ { \mathrm { g e n d e r } } \downarrow$ </td><td> $H _ { i x } \uparrow$ </td><td>CLIP↑</td><td>Aes∆↑</td><td> $\mathrm { D R } _ { \mathrm { r a c e } } \downarrow$ </td><td> $\mathrm { D R } _ { \mathrm { g e n d e r } } \downarrow$ </td></tr><tr><td>Baseline</td><td>0.524±0.016</td><td>0.321±0.001</td><td></td><td>0.748±0.015</td><td>0.608±0.022</td><td>0.549±0.015</td><td>0.325±0.001</td><td></td><td>0.615±0.018</td><td>0.367±0.026</td></tr><tr><td>EA [39]</td><td>0.285±0.013</td><td>0.236±0.002</td><td>+0.336±0.011</td><td>0.079±0.016</td><td>0.434±0.027</td><td>0.319±0.012</td><td>0.323±0.001</td><td>+0.163±0.010</td><td>0.069±0.016</td><td>0.303±0.029</td></tr><tr><td>SDID [25]</td><td>0.515±0.016</td><td>0.314±0.001</td><td>+0.118±0.008</td><td>0.797±0.013</td><td>0.593±0.023</td><td>0.573±0.015</td><td>0.322±0.001</td><td>+0.006±0.015</td><td>0.628±0.017</td><td>0.456±0.026</td></tr><tr><td>RespoDiff [40]</td><td>0.540±0.017</td><td>0.315±0.001</td><td>-0.055±0.006</td><td>0.737±0.017</td><td>0.624±0.022</td><td>0.575±0.015</td><td>0.326±0.001</td><td>-0.042±0.009</td><td>0.584±0.018</td><td>0.458±0.025</td></tr><tr><td>FairImagen [12]</td><td>0.534±0.016</td><td>0.320±0.001</td><td>+0.004±0.005</td><td>0.744±0.015</td><td>0.574±0.025</td><td>0.539±0.015</td><td>0.319±0.001</td><td>+0.057±0.011</td><td>0.385±0.019</td><td>0.293±0.028</td></tr><tr><td>SGS (ours)</td><td>0.794±0.014</td><td>0.318±0.001</td><td>-0.000±0.007</td><td>0.299±0.007</td><td>0.504±0.025</td><td>0.832±0.011</td><td>0.324±0.001</td><td>-0.037±0.010</td><td>0.306±0.006</td><td>0.233±0.028</td></tr></table>

Table 2. Human evaluation: share of judgments preferring SGS over each baseline (n=65 raters). <sup>∗</sup>: statistically significant preference (p<0.05).
<table><tr><td>VS.</td><td>Diversity</td><td>Faithful.</td><td>Preference</td></tr><tr><td>Baseline</td><td>88.5%*</td><td>46.2%</td><td>47.4%</td></tr><tr><td>EA</td><td>90.5%*</td><td>89.4%*</td><td>75.6%*</td></tr><tr><td>FairImagen</td><td>88.8%*</td><td>45.8%</td><td>46.8%</td></tr><tr><td>RespoDiff</td><td>89.0%*</td><td>51.9%</td><td>51.3%</td></tr><tr><td>SDID</td><td>90.5%*</td><td>52.3%</td><td>55.1%</td></tr></table>

tion O).

## 5.7. Qualitative Analysis

Figure 2’s three occupations span orthogonal stereotype directions (nurses female-skewed; CEOs, farmers maleskewed), yet base outputs are homogeneous in all three. EA’s unifordm shift is clearest for nurses, where all four figures converge on one demographic, eliminating the intraimage variation that drives its $H _ { i x }$ below baseline. SGS instead maintains scene coherence across all three directions, confining steering to person regions rather than leaking into occupational cues like attire or background. Additional results are in the supplementary material (Section L).

Table 3. Inference time per-image (A6000 GPU, 512×512, 21 denoising steps). FPs: total forward passes through the DiT per image. Overhead relative to unsteered baseline.
<table><tr><td>Method</td><td>FPs</td><td>Time (s)</td><td>Overhead</td></tr><tr><td>Baseline</td><td>21</td><td> $7 . 8 \pm 0 . 1$ </td><td>1.0</td></tr><tr><td>EA</td><td>21</td><td> $7 . 9 \pm 0 . 0$ </td><td>1.0</td></tr><tr><td>SGS (K=2)</td><td>33</td><td> $1 2 . 3 \pm 0 . 0$ </td><td>1.6</td></tr><tr><td>SGS (K=3)</td><td>39</td><td> $1 4 . 6 \pm 0 . 0$ </td><td>1.9</td></tr><tr><td>SGS (K=4)</td><td>45</td><td> $1 6 . 8 \pm 0 . 0$ </td><td>2.2</td></tr></table>

## 5.8. Compute Cost

Table 3 reports per-image denoising time measured on a single A6000 GPU (48GB) across 10 runs after 3 warmup passes. EA introduces no additional forward passes and is within measurement noise of the baseline. SGS incurs a deterministic overhead of exactly $K \times ( t _ { \mathrm { e n d } } - t _ { \mathrm { s t a r t } } ) = 6 K$ additional forward passes. For K = 3 the generation takes 14.6s versus 7.8s for the baseline. This cost is proportional to the scene complexity and can be reduced by narrowing the steering window at the expense of steering accuracy (Table S6, supplementary Section H).

## 6. Limitations

Three limitations bound the current system. First, segmentor accuracy degrades for k≥4 persons in heavily occluded scenes, this also impacts the steering accuracy. Second, the per-instance forward pass introduces a compute overhead proportional to K. Batching would reduce wall-clock time at the cost of VRAM, a trade-off we could not explore within our hardware budget. Third, SGS applies steering regardless of prompt intent; a query such as “four male doctors” will still redistribute demographics. A lightweight prompt parser that zeroes the steering signal for explicitly constrained slots could address this.

## 7. Conclusion

We presented SGS, a per-instance demographic steering framework for multi-person T2I generation requiring no fine-tuning, inversion, or layout input. Global steering improves aggregate balance only by driving intra-image diversity below the unsteered baseline; SGS is the only evaluated method to break this trade-off, raising intersectional entropy to 0.794 (from 0.524) on FLUX.1-Dev and to 0.832 (from 0.549) on SD3.5-Large, while recovering near-best aggregate balance and preserving semantic coherence on both architectures. Per-instance spatial steering thus emerges as a necessary step toward fair multi-person text-to-image generation, generalizing across architecturally distinct diffusion transformers rather than being a property of any single model.

## References

[1] Federico Bianchi, Pratyusha Kalluri, Esin Durmus, Faisal Ladhak, Myra Cheng, Debora Nozza, Tatsunori Hashimoto, Dan Jurafsky, James Zou, and Aylin Caliskan. Easily accessible text-to-image generation amplifies demographic stereotypes at large scale. In Proceedings of the 2023 ACM conference on fairness, accountability, and transparency, pages 1493–1504, 2023. 3

[2] Sarah Bonna, Yu-Cheng Huang, Ekaterina Novozhilova, Sejin Paik, Zhengyang Shan, Michelle Yilin Feng, Ge Gao, Yonish Tayal, Rushil Kulkarni, Jialin Yu, et al. Debiaspi: inference-time debiasing by prompt iteration of a text-toimage generative model. In European Conference on Computer Vision, pages 68–83. Springer, 2024. 3

[3] Shubhankar Borse, Farzad Farhadzadeh, Munawar Hayat, and Fatih Porikli. Disco: Reinforcement with diversity constraints for multi-human generation. arXiv preprint arXiv:2510.01399, 2025. 3, 7

[4] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-toend object detection with transformers. In European conference on computer vision, pages 213–229. Springer, 2020. 2, 4, 5

[5] Jaemin Cho, Abhay Zala, and Mohit Bansal. Dall-eval: Probing the reasoning skills and social biases of text-toimage generation models. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3043– 3054, 2023. 3

[6] Ching-Yao Chuang, Varun Jampani, Yuanzhen Li, Antonio Torralba, and Stefanie Jegelka. Debiasing visionlanguage models via biased prompts. arXiv preprint arXiv:2302.00070, 2023. 1, 3

[7] Omer Dahary, Or Patashnik, Kfir Aberman, and Daniel Cohen-Or. Be yourself: Bounded attention for multi-subject text-to-image generation. In European Conference on Computer Vision, pages 432–448. Springer, 2024. 2, 3

[8] Farbod Davoodi, Pooriya Safaei, Sana Harighi, Parsa Gholami, Amirali Amini, Kimia Vanaei, Emad Firoozi, Parham Abed Azad, Babak Khalaj, Siavash Ahmadi, et al. Kg-fairdiff: Knowledge graph-guided prompt refinement for demographically fair text-to-image generation. In Culture x AI: Evaluating AI as a Cultural Technology (ICML 2026), 2026. 2

[9] Sepehr Dehdashtian, Gautam Sreekumar, and Vishnu Boddeti. Oasis uncovers: High-quality t2i models, same old stereotypes. In The Thirteenth International Conference on Learning Representations, 2025. 1, 3

[10] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Muller, Harry Saini, Yam Levi, Dominik¨ Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis. In Proceedings of the 41st International Conference on Ma chine Learning, pages 12606–12633. PMLR, 2024. 1, 3, 4

[11] Felix Friedrich, Manuel Brack, Lukas Struppek, Dominik Hintersdorf, Patrick Schramowski, Sasha Luccioni, and Kristian Kersting. Auditing and instructing text-to-image

generation models on fairness. AI and Ethics, 5(3):2103– 2123, 2025. 3

[12] Zihao Fu, Ryan Brown, Shun Shao, Kai Rawal, Eoin Delaney, and Chris Russell. Fairimagen: Post-processing for bias mitigation in text-to-image models. Advances in Neural Information Processing Systems, 38:29488–29512, 2026. 7, 8, 18

[13] Rohit Gandikota, Hadas Orgad, Yonatan Belinkov, Joanna Materzynska, and David Bau. Unified concept editing in´ diffusion models. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 5111– 5120, 2024. 1, 3

[14] Douglas Guilbeault, Solene Delecourt, Tasker Hull, Bhar-\` gav Srinivasa Desikan, Mark Chu, and Ethan Nadler. Online images amplify gender bias. Nature, 626(8001):1049–1055, 2024. 1

[15] Boyu Han, Qianqian Xu, Shilong Bao, Zhiyong Yang, Kangli Zi, and Qingming Huang. Lightfair: Towards an efficient alternative for fair t2i diffusion via debiasing pre-trained text encoders. Advances in Neural Information Processing Sys tems, 38:22671–22724, 2026. 3

[16] Alec Helbling, Tuna Han Salih Meral, Benjamin Hoover, Pinar Yanardag, and Duen Horng Chau. Conceptattention: Diffusion transformers learn highly interpretable features. In Forty-second International Conference on Machine Learning, 2025. 3

[17] Sanghyun Jo, Ziseok Lee, Wooyeol Lee, Jonghyun Choi, Jaesik Park, and Kyungsu Kim. Trace: Your diffusion model is secretly an instance edge detector. arXiv preprint arXiv:2503.07982, 2025. 3, 4, 6

[18] Sangwon Jung, Alex Oesterling, Claudio Mayrink Verdun, Sajani Vithana, Taesup Moon, and Flavio P. Calmon. Multigroup proportional representations for text-to-image models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 2, 3

[19] Dhiraj Kalamkar, Dheevatsa Mudigere, Naveen Mellempudi, Dipankar Das, Kunal Banerjee, Sasikanth Avancha, Dharma Teja Vooturi, Nataraj Jammalamadaka, Jianyu Huang, Hector Yuen, et al. A study of bfloat16 for deep learning training. arXiv preprint arXiv:1905.12322, 2019. 5

[20] Kimmo Karkk ¨ ainen and Jungseock Joo. Fairface: Face at-¨ tribute dataset for balanced race, gender, and age. arXiv preprint arXiv:1908.04913, 2019. 6, 14

[21] Yasufumi Kawano and Yoshimitsu Aoki. Maskdiffusion: Exploiting pre-trained diffusion models for semantic seg mentation. IEEE Access, 12:127283–127293, 2024. 6

[22] Eunji Kim, Siwon Kim, Minjun Park, Rahim Entezari, and Sungroh Yoon. Rethinking training for de-biasing text-toimage generation: Unlocking the potential of stable diffusion. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 13361–13370, 2025. 1, 3

[23] Klaus Krippendorff. Measuring the reliability of qualita tive text analysis data. Quality and quantity, 38(6):787–800, 2004. 20

[24] Black Forest Labs. Flux. https://github.com/ black-forest-labs/flux, 2024. 1, 3, 4

[25] Hang Li, Chengzhi Shen, Philip Torr, Volker Tresp, and Jindong Gu. Self-discovering interpretable diffusion latent directions for responsible text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 3, 6, 7, 8, 18

[26] Jia Li, Lijie Hu, Jingfeng Zhang, Tianhang Zheng, Hua Zhang, and Di Wang. Fair text-to-image diffusion via fair mapping. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 26256–26264, 2025. 3

[27] Xingchao Liu, Chengyue Gong, et al. Flow straight and fast: Learning to generate and transfer data with rectified flow. In The Eleventh International Conference on Learning Representations, 2022. 3

[28] Sasha Luccioni, Christopher Akiki, Margaret Mitchell, and Yacine Jernite. Stable bias: Evaluating societal representations in diffusion models. Advances in Neural Information Processing Systems, 36:56338–56351, 2023. 1, 3

[29] Fausto Milletari, Nassir Navab, and Seyed-Ahmad Ahmadi. V-net: Fully convolutional neural networks for volumetric medical image segmentation. In 2016 fourth international conference on 3D vision (3DV), pages 565–571. Ieee, 2016. 5

[30] Jeonghoon Park, Juyoung Lee, Chaeyeon Chung, Jaeseong Lee, Jaegul Choo, and Jindong Gu. Fair generation without unfair distortions: Debiasing text-to-image generation with entanglement-free attention. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 17567–17576. IEEE, 2025. 3

[31] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 3, 6, 14

[32] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Radle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Seg-¨ ment anything in images and videos. In International Conference on Learning Representations, pages 28085–28128, 2025. 5, 12

[33] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in neural information processing systems, 35:25278–25294, 2022. 6, 7

[34] Sefik Ilkin Serengil and Alper Ozpinar. Boosted lightface: A hybrid dnn and gbm model for boosted facial recognition. Gazi University Journal of Science, 39(1):452–466, 2026. 14

[35] Preethi Seshadri, Sameer Singh, and Yanai Elazar. The bias amplification paradox in text-to-image generation. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 6367–6384, 2024. 3

[36] Xudong Shen, Chao Du, Tianyu Pang, Min Lin, Yongkang Wong, and Mohan Kankanhalli. Finetuning text-to-image

diffusion models for fairness. In The Twelfth International Conference on Learning Representations, 2024. 1, 3, 14

[37] Megan Smith, Venkatesh Thirugnana Sambandham, Florian Richter, Matthias Uhl, Laura Crompton, and Torsten Schon.¨ Operationalizing fairness in text-to-image models: A survey of bias, fairness audits and mitigation strategies. In ICLR 2026 Workshop on Algorithmic Fairness Across Alignment Procedures and Agentic Systems, 2026. 1

[38] Luming Tang, Menglin Jia, Qianqian Wang, Cheng Perng Phoo, and Bharath Hariharan. Emergent correspondence from image diffusion. In Thirty-seventh Conference on Neu ral Information Processing Systems, 2023. 3, 6

[39] Venkatesh Thirugnana Sambandham and Torsten Schon.¨ Embedding arithmetic: A lightweight, tuning-free framework for post-hoc bias mitigation in text-to-image models. In International Conference on Pattern Recognition, pages 308–323. Springer, 2026. 1, 3, 4, 6, 7, 8, 14

[40] Silpa Vadakkeeveetil Sreelatha, Sauradip Nag, Muhammad Awais, Serge Belongie, and Anjan Dutta. Respodiff: Dualmodule bottleneck transformation for responsible &amp; faithful t2i generation. In Advances in Neural Information Processing Systems, pages 42551–42583. Curran Associates, Inc., 2025. 6, 7, 8, 18

[41] Yixin Wan and Kai-Wei Chang. The male ceo and the female assistant: Evaluation and mitigation of gender biases in text-to-image generation of dual subjects. In Proceedings of the 63rd Annual Meeting of the Association for Compu tational Linguistics (Volume 1: Long Papers), pages 9174– 9190, 2025. 1, 2, 3, 6

[42] Jiarui Xu, Sifei Liu, Arash Vahdat, Wonmin Byeon, Xiaolong Wang, and Shalini De Mello. Open-vocabulary panoptic segmentation with text-to-image diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2955–2966, 2023. 3

[43] Cheng Zhang, Xuanbai Chen, Siqi Chai, Chen Henry Wu, Dmitry Lagun, Thabo Beeler, and Fernando De la Torre. Itigen: Inclusive text-to-image generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3969–3980, 2023. 1, 3

[44] K. Zhang, Z. Zhang, Z. Li, and Y. Qiao. Joint face detection and alignment using multitask cascaded convolutional networks. IEEE Signal Processing Letters, 23(10):1499–1503, 2016. 6

[45] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018. 7

[46] Jieyu Zhao, Tianlu Wang, Mark Yatskar, Vicente Ordonez, and Kai-Wei Chang. Gender bias in coreference resolution: Evaluation and debiasing methods. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), pages 15–20, 2018. 16

[47] Junlei Zhou, Jiashi Gao, Xiangyu Zhao, Xin Yao, and Xuetao Wei. Association of objects may engender stereotypes: Mitigating association-engendered stereotypes in text-to-image

generation. Advances in Neural Information Processing Systems, 37:51754–51786, 2024. 3

[48] Xingyi Zhou, Dequan Wang, and Philipp Krahenb¨ uhl. Ob-¨ jects as points. arXiv preprint arXiv:1904.07850, 2019. 6

## Supplementary Material

## A. Segmentor - Training Dataset Details

The segmentor is trained on a purpose-built dataset of 1440 generated multi-person images spanning $k \in \{ 2 , 3 , 4 , 5 \}$ persons across 9 scene types and 3 difficulty levels. The 9 scene types are: market, bus stop, city landmark, outdoor concert, train station, backyard barbecue, city park, street food fair, and mountain trail. Difficulty encodes camera framing and occlusion: easy uses a wide shot with subjects standing apart facing the camera; medium uses a medium shot with subjects side by side; hard uses tight close-up shots with partial occlusion among subjects. All images were generated using FLUX.1-Dev at $5 1 2 \times 5 1 2$ resolution with guidance scale 4.5 and 21 denoising steps.

Train/val/test split The 1440 training images are split into 80% training and 20% validation images, stratified by k and difficulty level. The test set (n=720) is generated from a disjoint set of 9 out-of-distribution scene types with different framing protocol to evaluate generalisation beyond the training distribution.

Pseudo-label generation Instance segmentation masks are produced by SAM2 [32] automatic mask generation applied to each generated image. Predicted masks are filtered by minimum area and matched to ground-truth person count k via Hungarian matching on mask centroid distance. Duplicate predictions with pairwise IoU > 0.5 are suppressed by retaining the higher confidence mask.

## B. Segmentor - Architecture

We describe each component of the instance segmentor with exact hyper-parameter values as used in all reported experiments.

Input Features. The segmentor receives the binned selfattention feature map $\mathbf { F } _ { t } ,$ formed by concatenating layer bins b=3 and $b { = } 4$ of the MM-DiT self-attention at the detection timestep $t _ { \mathrm { s t a r t } }$ . Features are min-max normalized per channel before being passed to the backbone. This choice of bins is itself the result of an ablation (Table S1, moved here from the main paper’s Section 5.3 for space): performance improves monotonically from Bin 1 to Bin 4 as layer depth increases, and concatenating Bins 3+4 outperforms every single-bin or other-pair configuration tested.

Timestep Conditioning The continuous FLUX timestep $t \in [ 0 , 1 ]$ is encoded via a sinusoidal embedding followed by a learned linear projection with SiLU activation. The embedding is spatially broadcast and concatenated channelwise to $\mathbf { F } _ { t }$ before the backbone, conditioning the segmentor on the noise level at which features were extracted.

Table S1. Layer bin ablation (n=720). Mean mIoU, AP50, and AP50-95 averaged across all denoising timesteps using DETR. Best in bold.
<table><tr><td>Bin</td><td>mIoU</td><td>AP50</td><td>AP50-95</td></tr><tr><td>Bin 1</td><td>0.539</td><td>0.581</td><td>0.225</td></tr><tr><td>Bin 2</td><td>0.583</td><td>0.661</td><td>0.287</td></tr><tr><td>Bin 3</td><td>0.606</td><td>0.710</td><td>0.333</td></tr><tr><td>Bin 4</td><td>0.626</td><td>0.736</td><td>0.360</td></tr><tr><td>3+4</td><td>0.631</td><td>0.747</td><td>0.368</td></tr></table>

CNN Backbone Three ConvBnReLU blocks (Conv2d, BatchNorm2d, ReLU, same-padding) progressively compress the high-dimensional attention channels into a compact spatial feature map while preserving the $3 2 \times 3 2$ resolution throughout. A final 1×1 convolution projects to the transformer embedding dimension $D _ { e } { = } 1 2 8$ , yielding pixel embeddings $\mathbf { P } \in \mathbb { R } ^ { 1 , 0 ^ { \smile } 4 \times 1 2 8 }$ after spatial flattening.

Transformer decoder. $N _ { q } { = } 8$ learnable instance queries are refined by L=2 decoder layers. Each layer applies selfattention among the queries, cross-attention from queries to P, and a two-layer FFN with GELU activation, all with residual connections and LayerNorm (H=4 attention heads throughout).

Output heads. Each refined query produces three outputs via dedicated linear heads: a scalar confidence $\mathrm { c o n f } _ { i } \in$ [0, 1] a center coordinate center<sub>i</sub> $\in [ 0 , 3 2 ] ^ { 2 }$ , and an instance mask $M _ { i } \in [ 0 , 1 ] ^ { 3 2 \times 3 2 }$ computed as a dot-product between a per-query mask embedding and P, reshaped to $3 2 \times 3 2$ center<sub>i</sub> is used only during training to stabilize the Hungarian matching.

Inference Queries with con $\mathrm { f } _ { i } < \tau { = } 0 . 5$ are discarded and the retained masks are binarized at $> 0 . 5 ,$ , yielding $K \leq 8$ binary instance masks. No NMS is applied - K is determined by the confidence threshold.

Training. The model is trained for 50 epochs with Adam $( \mathrm { l r } { = } 1 0 ^ { - \bar { 4 } }$ , batch size 8, 5 linear warm-up epochs). The 1,440 training images are split 80/20 into 1,152 training and 288 validation samples, stratified by k and difficulty. Loss weights: $w _ { c } { = } 2 . 0 , w _ { m } { = } 2 . 0$ . The complete segmentor has 1.2M parameters, lightweight relative to the FLUX DiT backbone it operates on.

## C. Segmentor Results - Per Difficulty

Figure 4 breaks down DETR segmentor performance across scene difficulty (left) and person count K (right). AP50 remains consistently high across all cases, while AP50- 95 drops substantially, indicating that instances are reliably detected but mask boundaries lack the precision required at stricter IoU thresholds. This gap widens at higher

![](images/9a82e6d599c870e1b78946a6659a9a72b9536d3bbc3a35a8e4ef0e44566b725a.jpg)

![](images/16c9b406c3d2efdf23cd7956dd04fe5fa8392eed9304ff337b50190154c89115.jpg)  
Figure 4. Segmentation performance of the SGS instance segmentor (DETR variant) broken down by scene difficulty (left) and person count K (right). Left: mIoU, AP50, and AP50-95 averaged over all K for easy, medium, and hard scenes. Performance degrades with difficulty primarily due to increased occlusion and smaller per-person regions. Right: The same metrics averaged over difficulty levels per K. All three metrics decline as K increases, reflecting the greater spatial competition among instances at higher person counts. Results are reported at denoising step t = 4.

K, where spatial competition between instances further degrades boundary delineation.

## D. Steering Performance by Person Count and Oracle-Mask Validation

Figure 4 above shows that segmentor detection quality (mIoU/AP) degrades with $K ;$ here we ask the downstream question directly: does that degradation propagate into steering accuracy, and how much of any remaining error is attributable to the segmentor rather than the steering mechanism. Both experiments below use FLUX.1-Dev only, at the operating point $\alpha { = } 2 . 0 , \lambda { = } 4 . 0 .$ , window [4, 10).

Steering accuracy by K, at the operating point. Table S2 slices the steering-window ablation (Section H) by person count at $t _ { \mathrm { s t a r t } } { = } 4 , ~ t _ { \mathrm { e n d } } { = } 1 0$ . This ablation’s evaluation set covers $k \in \{ 2 , 3 \}$ only, the same scope as the original steering-hyperparameter ablation it is drawn from; it does not include $k { = } 4$

Table S2. Steering accuracy by person count at the selected operating point (n=80/K).
<table><tr><td>K</td><td> $\operatorname { A c c } _ { \operatorname { r a c e } }$ </td><td> $\operatorname { A c c } _ { \mathrm { g e n d e r } }$ </td><td> $K _ { \mathrm { d e t e c t e d } }$  (mean)</td><td> $H _ { i x }$ </td></tr><tr><td>2</td><td>0.733</td><td>0.748</td><td>2.050</td><td>0.662</td></tr><tr><td>3</td><td>0.683</td><td>0.766</td><td>3.075</td><td>0.861</td></tr></table>

Oracle-mask (SAM2) vs. DETR segmentor, matched pairs. To isolate the segmentor’s contribution specifically, we re-run SGS with SAM2-derived masks (extracted from the finished, unsteered baseline image) in place of the

DETR segmentor, on the same 445 images for both (identical occupation/K/template/seed; only the mask source differs).

Table S3. DETR vs. oracle (SAM2) masks, matched pairs, $n { = } 4 4 5$ overall (149/148/148 at $K { = } 2 / 3 / 4 )$
<table><tr><td>K</td><td>Mask source</td><td> $\operatorname { A c c } _ { \operatorname { r a c e } }$ </td><td> $\operatorname { A c c } _ { \mathrm { g e n d e r } }$ </td><td> $H _ { i x }$ </td><td> $\mathrm { L P I P S _ { b g } }$ </td></tr><tr><td rowspan="2">2</td><td>DETR</td><td>0.843</td><td>0.896</td><td>0.792</td><td>0.743</td></tr><tr><td>Oracle</td><td>0.812</td><td>0.849</td><td>0.799</td><td>0.750</td></tr><tr><td rowspan="2">3</td><td>DETR</td><td>0.722</td><td>0.806</td><td>0.835</td><td>0.738</td></tr><tr><td>Oracle</td><td>0.705</td><td>0.776</td><td>0.902</td><td>0.740</td></tr><tr><td rowspan="2">4</td><td>DETR</td><td>0.643</td><td>0.720</td><td>0.834</td><td>0.756</td></tr><tr><td>Oracle</td><td>0.701</td><td>0.764</td><td>0.874</td><td>0.753</td></tr><tr><td rowspan="2">Overall</td><td>DETR</td><td>0.714</td><td>0.808</td><td>0.820</td><td>0.746</td></tr><tr><td> $\mathrm { O r a c l e }$ </td><td>0.740</td><td>0.796</td><td>0.858</td><td>0.748</td></tr></table>

The DETR segmentor matches or exceeds oracle-mask race accuracy at $K { = } 2$ and K=3 (0.843 vs. 0.812; 0.722 vs. 0.705), but falls behind it at $K { = } 4 ~ ( 0 . 6 4 3 ~ \mathrm { v s . } ~ 0 . 7 0 1$ a gap of 0.058). The Oracle SAM2 masks recover the race-steering accuracy that DETR loses once scenes become more crowded. Gender accuracy shows a related but smaller pattern: DETR leads at $K = 2$ and K = 3 (0.896 vs. 0.849; 0.806 vs. 0.776) while the oracle mask lead at K = 4 by 0.04 (0.764 vs. 0.720). Background preservation $\mathrm { ( L P I P S _ { b g } ) }$ is flat across mask source at every K, so this K=4 gap is not a fidelity tradeoff. Taken together, these results indicate the live segmentor is not the limiting factor behind SGS’s residual steering error at K=2 and K=3. The gap to perfect accuracy comes from elsewhere in the pipeline (steering strength, classifier noise).

![](images/1637424c648def9004659a2f041d6b389e28a210170fc90a666d025ef55c2015.jpg)  
Figure 5. Qualitative comparison of instance segmentation methods across four scenes of increasing difficulty. Each row shows the generated image, SAM2 oracle masks, and predictions from K-Means, Spectral clustering, CenterNet, and our DETR-based segmentor. K-Means and Spectral produce fragmented, noisy masks in complex scenes (rows 3–4), while CenterNet sometimes over-predicts the instance count under occlusion. DETR consistently recovers the correct number of instances with coherent boundaries across all difficulty levels.

## E. Classifier Validation

All race/gender numbers in this paper depend on one automatic tool, FairFace [20]. To read demographics off a generated face, before trusting it, we check how accurate it actually is against a set of images where the true demographic is known for certain, and we also check two other classifiers (a CLIP [31] zero-shot classifier, and DeepFace [34]) for comparison.

Building a set of images with known demographics. We generate images where every person in the group is asked for the same demographic (e.g. “a group of three Black women”), so the prompt itself has a cue for every face in that image. This is done for both FLUX.1-Dev and SD3.5-Large, across all 8 race×gender combinations and group sizes $k \in \{ 2 , 3 , 4 \}$ . We keep only images where a face detector actually finds exactly k faces, to avoid mixing in blank or wrong-count images; this leaves 1,832 individ ual face crops with a known-correct label.

Results. Table S4 reports accuracy for each classifier on this set. FairFace and CLIP are both highly accurate on race (95–96%) and near-perfect on gender (>99%); Deep-Face is noticeably weaker on race (88%) and gender (93%). “Abstain” means the classifier’s raw output fell into a category we do not use (e.g. Middle Eastern, Latino/Hispanic, dropped due to high confusion with other categories reported in several previous works [36, 39]); this happens occasionally for FairFace and DeepFace and never for CLIP, which only ever outputs one of our four race categories directly.

Do the three classifiers agree with each other? Separately from whether each classifier is correct, we check how often the three classifiers agree with each other (Krippendorff’s alpha, where 1.0 means perfect agreement and 0 means no better than chance). Agreement is high on race (0.94) and on gender (0.94), and every pairwise comparison between two classifiers also shows strong agreement (Cohen’s kappa between 0.85 and 1.00 in every case). This gives an independent check on the results above: it is not just that FairFace happens to agree with the ground-truth labels we built, three separately-built tools broadly agree with each other too.

Table S4. Classifier validation against known-correct labels (n=1,832 face crops, both FLUX.1-Dev and SD3.5-Large).
<table><tr><td rowspan="2"></td><td colspan="2">Race</td><td colspan="2">Gender</td></tr><tr><td>Classifier Accuracy</td><td>Abstain</td><td>Accuracy</td><td>Abstain</td></tr><tr><td>FairFace</td><td>95.7%</td><td>2.5%</td><td>99.7%</td><td>0.0%</td></tr><tr><td>CLIP</td><td>95.3%</td><td>0.0%</td><td>99.9%</td><td>0.0%</td></tr><tr><td>DeepFace</td><td>87.7%</td><td>7.5%</td><td>92.7%</td><td>0.0%</td></tr></table>

## F. Segmentor Timestep Sweep – SD3.5-Large

We repeat the segmentor timestep sweep of Section 5.3 on SD3.5-Large, training a separate DETR-style segmentor on SD3.5’s own self-attention features at $1 0 2 4 \times 1 0 2 4$ resolution (same architecture and training recipe as the FLUX segmentor, Section B), and evaluating on a held-out validation split (n=288) of SD3.5-generated images. Across SD3.5’s 28-step denoising schedule, segmentation quality peaks at timestep index 17, reaching mIoU of 0.664, AP50 of 0.841, and AP50-95 of 0.413 which is comparable to FLUX’s own peak (mIoU 0.664, AP50 0.797, AP50-95 0.416, Section 5.3), indicating that MM-DiT self-attention features become instance-discriminative at an analogous point in SD3.5’s denoising trajectory as well.

![](images/46a8a89c6d4cebeeb63e23883d17666671828a2356c63391aa146a6c1361936f.jpg)

![](images/b41bbdbadd5d98ce8c58a7e026d6da6e532bff83cff03903d236792d3ce2152b.jpg)

![](images/3a8f894f4385a0d0fea784c0631ad6d96514c02d8bc51a666153e4c0609d38d9.jpg)  
Figure 6. SD3.5-Large DETR segmentor, timestep sweep (val split, n=288). mIoU, AP50, and AP50-95 across SD3.5’s 28-step denoising schedule, mirroring Figure 3’s FLUX sweep. Performance peaks at timestep index 17 and plateaus around it, the same qualitative shape as FLUX’s own sweep.

## G. Segmentor Results - Qualitative

Figure 5 compares instance segmentation predictions across four scenes spanning easy to hard difficulty. In the simpler market and tourist scenes (rows 1–2), all methods recover the correct instance count, though K-Means and Spectral clustering exhibit scattered noise pixels caused by highfrequency variation in the attention feature space. The gap widens in the close-up street scene and the outdoor hikers scene (rows 3–4): K-Means and Spectral produce heavily fragmented masks that merge or split individuals, while CenterNet over-estimates the instance count due to ambiguous overlap regions. DETR-based segmentor produces coherent per-person masks with the correct instance counts, demonstrating that the cross-attention mechanism of DETR learns to separate spatially proximate instances that confound clustering-based approaches.

Table S5. Steering hyperparameter ablation. Bold = selected configuration. Aes∆: aesthetic score delta vs. unsteered baseline (higher = better quality). $\mathrm { L P I P S _ { b g } \mathrm { : } }$ background preservation. CLIP∆: prompt–image CLIP similarity delta (closer to 0 = less semantic drift).
<table><tr><td></td><td> $_ \mathrm { A c c _ { r a c e } } \uparrow$ </td><td> $\operatorname { A c c } _ { \mathrm { g e n d e r } } \uparrow$ </td><td>Aes∆↑</td><td> $\mathrm { L P I P S _ { b g } \uparrow }$ </td><td>CLIP∆(→ 0)</td></tr><tr><td colspan="6">(a) EA strength α [λ=4.0 fixed]</td></tr><tr><td>1.5</td><td>0.573</td><td>0.722</td><td>+0.009</td><td>0.916</td><td>-0.026</td></tr><tr><td>2.0</td><td>0.708</td><td>0.757</td><td>+0.027</td><td>0.900</td><td>-0.033</td></tr><tr><td>2.5</td><td>0.723</td><td>0.740</td><td>+0.010</td><td>0.890</td><td>-0.042</td></tr><tr><td>3.0</td><td>0.636</td><td>0.738</td><td>-0.134</td><td>0.876</td><td>-0.052</td></tr><tr><td>3.5</td><td>0.438</td><td>0.566</td><td>-0.762</td><td>0.842</td><td>-0.104</td></tr><tr><td colspan="6">(b) Spatial strength λ [α=2.0 fixed]</td></tr><tr><td>3.0</td><td>0.644</td><td>0.734</td><td>+0.020</td><td>0.914</td><td>-0.028</td></tr><tr><td>4.0</td><td>0.708</td><td>0.757</td><td>+0.027</td><td>0.900</td><td>-0.033</td></tr><tr><td>5.0</td><td>0.736</td><td>0.742</td><td>+0.034</td><td>0.889</td><td>-0.039</td></tr><tr><td>6.0</td><td>0.767</td><td>0.746</td><td>-0.019</td><td>0.876</td><td>-0.046</td></tr><tr><td>7.0</td><td>0.758</td><td>0.742</td><td>-0.120</td><td>0.864</td><td>-0.053</td></tr></table>

## H. Steering Hyperparameter Sensitivity

Table S5 reports the full sensitivity analysis for EA strength α and spatial diversity strength λ, using the evaluation setup described in Section 5.4 of the main paper.

For α (panel a), steering accuracy and aesthetic quality improve from α=1.5 to α=2.0, then degrade at higher values. α=2.5 achieves marginally higher race accuracy (0.723 vs. 0.708) but at the cost of gender accuracy, aesthetic quality, background preservation, and semantic coherence. At $\alpha { \geq } 3 . 0 .$ , aesthetic quality falls below the unsteered baseline and CLIP similarity drops sharply, indicating the shifted embedding has diverged from the original prompt semantics.

For λ (panel b), λ=4.0 achieves the best joint performance. Higher values improve race accuracy marginally but incur increasing aesthetic degradation and semantic drift; λ=7.0 produces a negative aesthetic delta (−0.120) and a CLIP drop of −0.053. We select $\alpha { = } 2 . 0$ and λ=4.0 as our operating point.

Table S6. Steering window placement ablation (FLUX.1-Dev, α=2.0, λ=4.0). Bold = selected operating point. CLIP∆: prompt–image similarity delta vs. baseline.
<table><tr><td>Start</td><td>End</td><td> $_ \mathrm { A c c _ { r a c e } \uparrow }$ </td><td> $\operatorname { A c c } _ { \mathrm { g e n d e r } } \uparrow$ </td><td>Aes∆↑</td><td> $\mathrm { L P I P S _ { b g } \uparrow }$ </td><td> $\mathrm { C L I P } \Delta \to 0$ </td></tr><tr><td rowspan="4">2</td><td>8</td><td>0.713</td><td>0.784</td><td>-0.097</td><td>0.760</td><td>-0.097</td></tr><tr><td>10</td><td>0.718</td><td>0.783</td><td>-0.123</td><td>0.759</td><td>-0.102</td></tr><tr><td>12</td><td>0.749</td><td>0.791</td><td>-0.179</td><td>0.758</td><td>-0.105</td></tr><tr><td>14</td><td>0.754</td><td>0.783</td><td>-0.237</td><td>0.757</td><td>-0.106</td></tr><tr><td rowspan="4">4</td><td>8</td><td>0.632</td><td>0.744</td><td>+0.008</td><td>0.904</td><td>-0.029</td></tr><tr><td>10</td><td>0.708</td><td>0.757</td><td>+0.027</td><td>0.900</td><td>-0.033</td></tr><tr><td>12</td><td>0.741</td><td>0.743</td><td>+0.036</td><td>0.899</td><td>-0.038</td></tr><tr><td>14</td><td>0.769</td><td>0.745</td><td>+0.036</td><td>0.898</td><td>-0.041</td></tr><tr><td rowspan="3">6</td><td>10</td><td>0.476</td><td>0.649</td><td>-0.001</td><td>0.961</td><td>-0.013</td></tr><tr><td>12</td><td>0.545</td><td>0.652</td><td>+0.002</td><td>0.958</td><td>-0.016</td></tr><tr><td>14</td><td>0.594</td><td>0.649</td><td>+0.007</td><td>0.956</td><td>-0.018</td></tr></table>

Steering window Early detection and steering at $t _ { \mathrm { s t a r t } } { = } 2$ yields the highest steering accuracy across all end points but produces consistently negative aesthetic deltas and poor background preservation $( \mathrm { L P I P S _ { b g } } { \leq } 0 . 7 5 9 )$ , indicating that MM-DiT features at step 2 are insufficiently structured for reliable instance segmentation, causing the steering signal to leak beyond the intended mask regions. Late detection and steering at $t _ { \mathrm { s t a r t } } { = } 6$ preserves background coherence well $( \mathrm { L P I P S _ { b g } } { \ge } 0 . 9 5 6 )$ but achieves substantially lower steering accuracy $( \mathrm { A c c . } _ { \mathrm { r a c e } } { \le } 0 . 5 9 4$ $\mathrm { A c c } _ { \mathrm { g e n d e r } } { \le } 0 . 6 5 2 )$ , as the remaining steering window is too short to shift demographics effectively. $t _ { \mathrm { s t a r t } } { = } 4$ achieves the best balance across all metrics: extending the end step from 8 to 10 improves steering accuracy (0.632→0.708 race, 0.744→0.757 gender) with only marginal reductions in background preservation, while further extension $\mathbf { \nabla } [ 0 ~ t _ { \mathrm { e n d } } ] = 1 2$ or 14 offers diminishing accuracy gains at the cost of increasing semantic drift. We therefore select $t _ { \mathrm { s t a r t } } { = } 4$ and $t _ { \mathrm { e n d } } = 1 0$ as our operating point, corresponding to steps 4–10 of the 21- step denoising trajectory.

## I. Steering Hyperparameter and Window Ablation – SD3.5-Large

We repeat the steering-strength and steering-window ablations on SD3.5-Large.

Table S7 sweeps $\begin{array} { r l r } { \alpha } & { { } \in } & { \{ 1 . 0 , 1 . 5 , 2 . 0 \} } \end{array}$ and $\lambda \in$ {1.0, 2.0, 4.0}. Unlike FLUX, accuracy improves monotonically across the tested range on both parameters, with α=2.0, λ=4.0 the best-performing configuration tested; we select this as SD3.5’s operating point.

Table S8 sweeps detection/steering start step $\in$ $\{ 3 , 4 , 5 , 6 \}$ and end step, out of SD3.5’s 28-step schedule. start=4, end=24 achieves the highest accuracy among the configurations tested.

Table S7. SD3.5-Large steering-strength ablation. $\mathrm { L P I P S _ { b g } \mathrm { : } }$ background preservation.
<table><tr><td>α λ</td><td> $_ \mathrm { A c c _ { r a c e } } \uparrow$ </td><td> $\operatorname { A c c } _ { \mathrm { g e n d e r } } \uparrow$ </td><td> $\mathrm { L P I P S _ { b g } \uparrow }$ </td></tr><tr><td>1.0</td><td>1.0 0.443</td><td>0.620</td><td>0.958</td></tr><tr><td>1.0</td><td>2.0 0.526</td><td>0.646</td><td>0.938</td></tr><tr><td>1.0</td><td>4.0 0.707</td><td>0.693</td><td>0.904</td></tr><tr><td>1.5</td><td>1.0 0.561</td><td>0.653</td><td>0.946</td></tr><tr><td>1.5</td><td>2.0 0.694</td><td>0.677</td><td>0.912</td></tr><tr><td>1.5</td><td>4.0 0.809</td><td>0.722</td><td>0.877</td></tr><tr><td>2.0 1.0</td><td>0.658</td><td>0.660</td><td>0.931</td></tr><tr><td>2.0 2.0</td><td>0.752</td><td>0.682</td><td>0.893</td></tr><tr><td>2.0 4.0</td><td>0.800</td><td>0.759</td><td>0.855</td></tr></table>

Table S8. SD3.5-Large steering-window ablation, $\alpha { = } 2 . 0 .$ $\lambda { = } 4 . 0 .$
<table><tr><td>Start</td><td>End</td><td> $\operatorname { A c c } _ { \operatorname { r a c e } }$ </td><td> $\operatorname { A c c } _ { \mathrm { g e n d e r } }$ </td><td> $\mathrm { L P I P S _ { b g } }$ </td></tr><tr><td rowspan="3">3</td><td>10</td><td>0.391</td><td>0.624</td><td>0.921</td></tr><tr><td>12</td><td>0.402</td><td>0.624</td><td>0.920</td></tr><tr><td>14</td><td>0.427</td><td>0.642</td><td>0.919</td></tr><tr><td rowspan="5">4</td><td>11</td><td>0.378</td><td>0.623</td><td>0.945</td></tr><tr><td>13</td><td>0.397</td><td>0.617</td><td>0.944</td></tr><tr><td>16</td><td>0.424</td><td>0.617</td><td>0.944</td></tr><tr><td>19</td><td>0.453</td><td>0.626</td><td>0.944</td></tr><tr><td>24</td><td>0.514</td><td>0.622</td><td>0.944</td></tr><tr><td rowspan="3">5</td><td>10</td><td>0.338</td><td>0.610</td><td>0.964</td></tr><tr><td>12</td><td>0.357</td><td>0.613</td><td>0.963</td></tr><tr><td>14</td><td>0.386</td><td>0.613</td><td>0.962</td></tr><tr><td rowspan="3">6</td><td>10</td><td>0.293</td><td>0.590</td><td>0.977</td></tr><tr><td>12</td><td>0.310</td><td>0.590</td><td>0.976</td></tr><tr><td>14</td><td>0.340</td><td>0.595</td><td>0.975</td></tr></table>

## J. Extended Benchmark – Occupations and Prompt Templates

Table 1 in the main paper (Section 5.5) evaluates on 30 WinoBias occupations [46], disjoint from the 10 occupations used to fit the EA direction vectors (carpenter, mechanic, construction worker, laborer, sheriff, teacher, nurse, housekeeper,flight attendant, librarian), so all baselines are evaluated out-of-distribution with respect to their own calibration set.

Occupations. 15 male-skewed: driver, mover, developer, farmer, guard, chief, janitor, lawyer, cook, physician, CEO, analyst, manager, supervisor, salesperson. 15 femaleskewed: editor, designer, accountant, auditor, writer, baker, clerk, cashier, counselor, attendant, tailor, librarian, assistant, cleaner, hairdresser.

Prompt templates. Each occupation is instantiated under 5 templates, with {k} and {occupation} substituted as in

Table S9. Per-occupation demographic metrics, baseline vs. SGS (FLUX.1-Dev, extended benchmark, n=150/occupation). $H _ { i x } \mathrm { : }$ : intra image intersectional entropy (higher = more diverse). $D R _ { \mathrm { r a c e } } , D R _ { \mathrm { g e n d e r } } \colon$ deviation ratio from uniform (lower = more balanced). Lef block: male-skewed occupations. Right block: female-skewed occupations.
<table><tr><td rowspan="2">Occupation</td><td colspan="3">Baseline</td><td colspan="3">SGS</td></tr><tr><td> $H _ { i x }$ </td><td> $D R _ { r }$ </td><td> $D R _ { g }$ </td><td> $H _ { i x }$ </td><td>DRr</td><td> $D R _ { g }$ </td></tr><tr><td>Driver</td><td>0.480</td><td>0.544</td><td>1.000</td><td>0.578</td><td>0.324</td><td>0.680</td></tr><tr><td>Mover</td><td>0.406</td><td>0.486</td><td>0.920</td><td>0.717</td><td>0.297</td><td>0.627</td></tr><tr><td>Developer</td><td>0.482</td><td>0.794</td><td>0.987</td><td>0.843</td><td>0.279</td><td>0.787</td></tr><tr><td>Farmer</td><td>0.426</td><td>0.759</td><td>0.987</td><td>0.761</td><td>0.324</td><td>0.587</td></tr><tr><td>Guard</td><td>0.503</td><td>0.333</td><td>0.813</td><td>0.661</td><td>0.270</td><td>0.720</td></tr><tr><td>Chief</td><td>0.505</td><td>0.116</td><td>0.707</td><td>0.859</td><td>0.315</td><td>0.267</td></tr><tr><td>Janitor</td><td>0.547</td><td>0.674</td><td>0.800</td><td>0.810</td><td>0.306</td><td>0.600</td></tr><tr><td>Lawyer</td><td>0.582</td><td>0.866</td><td>0.840</td><td>0.735</td><td>0.289</td><td>0.880</td></tr><tr><td>Cook</td><td>0.489</td><td>0.687</td><td>0.893</td><td>0.858</td><td>0.421</td><td>0.267</td></tr><tr><td>Physician</td><td>0.598</td><td>0.640</td><td>0.640</td><td>0.862</td><td>0.280</td><td>0.587</td></tr><tr><td>CEO</td><td>0.385</td><td>0.907</td><td>0.987</td><td>0.793</td><td>0.288</td><td>0.987</td></tr><tr><td>Analyst</td><td>0.563</td><td>0.850</td><td>1.000</td><td>0.852</td><td>0.297</td><td>0.947</td></tr><tr><td>Manager</td><td>0.469</td><td>0.916</td><td>0.987</td><td>0.859</td><td>0.280</td><td>0.840</td></tr><tr><td>Supervisor</td><td>0.472</td><td>0.935</td><td>0.973</td><td>0.852</td><td>0.289</td><td>0.867</td></tr><tr><td>Salesperson</td><td>0.748</td><td>0.917</td><td>0.733</td><td>0.888</td><td>0.387</td><td>0.627</td></tr></table>

the main paper:

1. ”A portrait photo of {k} {occupation}s”

2. ”A photo of {k} {occupation}s”

3. ”A group of {k} {occupation}s”

4. ”A realistic image of {k} {occupation}s at work”

5. ”{k} {occupation}s standing together”

Combined with $k \in \{ 2 , 3 , 4 \}$ and 10 seeds, this yields 30 × $5 \times 3 \times 1 0 = 4 { , } 5 0 0$ images per method.

## K. Per-Occupation Demographic Distribution

Table S9 breaks down intra-image intersectional entropy $( H _ { i x } )$ and aggregate deviation ratio $( D R _ { \mathrm { r a c e } } , \ D R _ { \mathrm { g e n d e r } } )$ per occupation, for the baseline and SGS, across all 30 occupations of the extended WinoBias benchmark (Section 5.5 of the main paper; FLUX.1-Dev). Occupations are grouped by the male-skewed/female-skewed split used to construct the benchmark (Section J).

Entropy. SGS raises $H _ { i x }$ over the baseline for every one of the 30 occupations without exception – the smallest gain is Driver (0.480→0.578) and the largest is Tailor (0.395→0.861), with no occupation where SGS’s perinstance steering fails to increase intra-image diversity relative to the unsteered model.

Aggregate deviation. $D R _ { \mathrm { r a c e } }$ drops for every occupation under SGS, typically substantially (e.g. Manager: 0.916→0.280; CEO: 0.907→0.288), confirming the aggregate race-balance improvement seen in Table 1 holds occupation-by-occupation, not just on average. $D R _ { \mathrm { g e n d e r } }$ is noisier at the per-occupation level and does not improve uniformly – most occupations move closer to balance (e.g.

<table><tr><td rowspan="2">Occupation</td><td colspan="3">Baseline</td><td colspan="3">SGS</td></tr><tr><td> $H _ { i x }$ </td><td> $D R _ { r }$ </td><td> $D R _ { g }$ </td><td> $H _ { i x }$ </td><td>DRr</td><td> $D R _ { g }$ </td></tr><tr><td>Editor</td><td>0.605</td><td>0.872</td><td>0.573</td><td>0.772</td><td>0.298</td><td>0.387</td></tr><tr><td>Designer</td><td>0.603</td><td>0.732</td><td>0.147</td><td>0.777</td><td>0.333</td><td>0.133</td></tr><tr><td>Accountant</td><td>0.642</td><td>0.927</td><td>0.813</td><td>0.869</td><td>0.374</td><td>0.853</td></tr><tr><td>Auditor</td><td>0.492</td><td>0.816</td><td>0.973</td><td>0.758</td><td>0.324</td><td>0.813</td></tr><tr><td>Writer</td><td>0.622</td><td>0.813</td><td>0.440</td><td>0.826</td><td>0.306</td><td>0.200</td></tr><tr><td>Baker</td><td>0.420</td><td>0.621</td><td>0.787</td><td>0.806</td><td>0.298</td><td>0.280</td></tr><tr><td>Clerk</td><td>0.534</td><td>0.890</td><td>0.947</td><td>0.857</td><td>0.289</td><td>0.760</td></tr><tr><td>Cashier</td><td>0.514</td><td>0.854</td><td>0.440</td><td>0.783</td><td>0.270</td><td>0.040</td></tr><tr><td>Counselor</td><td>0.623</td><td>0.937</td><td>0.080</td><td>0.824</td><td>0.279</td><td>0.200</td></tr><tr><td>Attendant</td><td>0.686</td><td>0.841</td><td>0.413</td><td>0.922</td><td>0.289</td><td>0.347</td></tr><tr><td>Tailor</td><td>0.395</td><td>0.756</td><td>0.973</td><td>0.861</td><td>0.324</td><td>0.653</td></tr><tr><td>Librarian</td><td>0.515</td><td>0.776</td><td>0.467</td><td>0.722</td><td>0.494</td><td>0.040</td></tr><tr><td>Assistant</td><td>0.566</td><td>0.757</td><td>0.080</td><td>0.805</td><td>0.324</td><td>0.107</td></tr><tr><td>Cleaner</td><td>0.507</td><td>0.569</td><td>0.360</td><td>0.650</td><td>0.297</td><td>0.520</td></tr><tr><td>Hairdresser</td><td>0.355</td><td>0.911</td><td>0.547</td><td>0.667</td><td>0.391</td><td>0.333</td></tr></table>

Chief: 0.707→0.267; Cashier: 0.440→0.040) but five of the 30 move further from it (Lawyer: 0.840→0.880; Accountant: 0.813→0.853; Counselor: 0.080→0.200; Assistant: 0.080→0.107; Cleaner: 0.360→0.520) – consistent with gender assignment being a coarser 2-category balancing problem where per-occupation sample size (n=150) leaves more room for variance than the pooled Table 1 figures show.

Intersectional Observation. Taken together, Table S9 confirms the core motivation for $H _ { i x }$ holds at the level of individual occupations, not only in aggregate: the baseline is both gender-stereotyped and racially homogeneous on essentially every occupation tested, and SGS’s per-instance steering raises intra-image diversity consistently across all of them.

## L. Extended Qualitative Analysis

Figure 7 presents an extended qualitative comparison across four occupations spanning both stereotype directions and three person counts $k \in \{ 2 , 3 , 4 \}$ . Occupations include two female-dominated professions (Nurse, Housekeeping) and two male-dominated professions (Sheriffs, Construction Workers).

Baseline. Across all four occupations and all values of k, the unsteered baseline produces demographically homogeneous outputs that reflect occupational stereotypes; Nurses and Housekeepers are consistently female-presenting or White, while Sheriffs and Construction workers are consistently male-presenting and White. This homogeneity persists regardless of k, confirming that the stereotyping is driven by the text prompt rather than any scene composition effect introduced by specifying multiple persons.

SGS across person counts. SGS consistently produces intersectional demographic diversity across all k values. $\mathrm { A t } \ k \ = \ 2$ , each pair contains two persons steered to distinct intersectional targets– for example, a Black male nurse is paired with an Asian female nurse, or an Indian male housekeeper paired with an Asian female housekeeper. At $k \ = \ 3 \ \mathrm { a n d } \ k \ = \ 4 .$ , the segmentor reliably separates all persons and assigns independent demographic vectors to each, with the resulting groups spanning multiple racial categories and both genders within the same image. The consistency across k values demonstrates that the instance segmentor remains reliable as scene complexity increases.

Occupational context preservation. A key requirement of practical fairness steering is that demographic edits do not disrupt occupational cues. Across all conditions, SGS preserves the visual markers of each profession: nurses retain scrubs and stethoscopes, sheriffs retain badges and hats, housekeepers retain aprons, and construction workers retain safety helmets and reflective vests. This confirms that the spatially gated steering is confined to person regions via the instance masks, preventing demographic steering from leaking into occupational attire or scene background.

## M. Qualitative Comparison – SD3.5-Large

Figure 8 repeats the main paper’s qualitative comparison (Figure 2) on SD3.5-Large, across four prompts and all five baselines. Images from Base, EA, FairImagen, RespoDiff, and SDID remain demographically homogeneous or only mildly shifted across all four scenes, while SGS diversifies every scene independently per person while preserving occupational context (aprons, rice-field setting, formal attire). This confirms the pattern shown quantitatively in Table 1 holds qualitatively on a second architecture as well.

## N. Baseline Adaptation to Diffusion Transformers

FairImagen [12], SDID [25], and RespoDiff [40] were each designed and released against older UNet-based diffusion models – FairImagen against SD1.5 and SDXL, RespoDiff against SDXL, and SDID against a UNet-based Stable Diffusion pipeline – where generation is driven by a singlestream convolutional UNet with cross-attention to a text encoder. FLUX and SD3.5-Large instead use a Multimodal Diffusion Transformer (MM-DiT, Section 3.2): image and text tokens are processed jointly through a stack of selfattention blocks, with no UNet and no single bottleneck layer. None of the three methods’ released implementations run on this architecture family as-is. We faithfully reimple mented each method’s core mechanism for MM-DiT, verified directly against its own released code, rather than mechanically porting UNet-specific hooks.

FairImagen. The original method operates on the model’s pooled text-conditioning vector. For each demographic group g, it defines an empirical bias direction

$$
\nu _ { g } = \frac { 1 } { \vert X ^ { \left( g \right) } \vert } \sum _ { \bar { E } _ { p } \in X ^ { \left( g \right) } } \bar { E } _ { p } - \bar { E } ,\tag{13}
$$

and obtains a projector P by solving

$$
\operatorname* { m i n } _ { P ^ { \top } P = I } - \mathrm { T r } ( P ^ { \top } \Sigma _ { X } P ) + \lambda \| B P \| _ { F } ^ { 2 } ,\tag{14}
$$

where $B = Z ^ { \top } X \in \mathbb { R } ^ { G \times D }$ collects the per-group mean directions and $Z ~ \in ~ \{ 0 , 1 \} ^ { n \times G }$ is a one-hot group indicator; P thus spans the top-variance directions within the null space of the group-dependent signal B. At inference, the projector is applied orthogonally to the pooled embedding, $\bar { \bar { E } } _ { p } ^ { \prime } = P P ^ { \top } \bar { \bar { E } } _ { p }$ , before it conditions the UNet. MM-DiT models expose an analogous pooled conditioning vector alongside their token-level sequence conditioning, so we fit and apply this same projector on this real pooled vector, leaving the token sequence untouched.

SDID. The original method learns a single concept offset $c \in \mathbb { R } ^ { D }$ per demographic direction by

$$
c ^ { * } = \arg \operatorname* { m i n } _ { c } \sum _ { x , y \sim \mathcal { D } } \sum _ { t \sim [ 0 , T ] } \| \epsilon - \epsilon _ { \theta } ( x _ { t } ^ { + } , t , \pi ( y ^ { - } ) , c ) \| ^ { 2 } ,\tag{15}
$$

where $x ^ { + }$ is an image generated from a concept-inclusive prompt and $y ^ { - }$ is the same prompt with the concept term removed, with the base model $\epsilon _ { \theta }$ kept frozen throughout. At inference, the offset is added directly to the UNet’s imagelatent bottleneck activation, $h  h + c .$ , a single injection point on the image side of the network. MM-DiT has no UNet bottleneck to target, so we carry the same design principle over: the learned offset is added to the hidden state of one representative block within the joint-transformer stack, preserving a single image-side injection point rather than broadcasting the offset into the text-token stream.

RespoDiff. The original method decomposes its responsible transformation as

$$
\hat { f } ( y _ { \mathrm { n e u } } ) = g \big ( T _ { \theta } ^ { \mathrm { r e s p } , s } ( h _ { \mathrm { n e u } } ) + T _ { \theta } ^ { \mathrm { s e m } , s } ( h _ { \mathrm { n e u } } ) \big ) ,\tag{16}
$$

training its Responsible Concept Alignment Module $T _ { \theta } ^ { \mathrm { r e s p } , s }$ (our race-aware/gender-aware modules) and Semantic Alignment Module $T _ { \theta } ^ { \mathrm { s e m } , s }$ with

$$
\mathcal { L } _ { \mathrm { r e s p } } = \mathbb { E } _ { z _ { t , \mathrm { n e u } } } \big [ \| \epsilon _ { f _ { \mathrm { r e s p } } } ( z _ { t , \mathrm { n e u } } , y _ { \mathrm { n e u } } ) - \epsilon _ { f } ( z _ { t , \mathrm { n e u } } , y _ { \mathrm { t a r } } ^ { s } ) \| _ { 2 } ^ { 2 } \big ] ,\tag{17}
$$

$$
\mathcal { L } _ { \mathrm { s e m } } = \mathbb { E } _ { z _ { t , \mathrm { n e u } } } \left[ \Vert \epsilon _ { \hat { f } } ( z _ { t , \mathrm { n e u } } , y _ { \mathrm { n e u } } ) - \epsilon _ { f } ( z _ { t , \mathrm { n e u } } , y _ { \mathrm { n e u } } ) \Vert _ { 2 } ^ { 2 } \right] ,\tag{18}
$$

![](images/f7ac5b0c26e8e63426cb9b71f62cbd9a41ac5972e177fd37b67f8b2c2366db79.jpg)  
Figure 7. Extended qualitative comparison across four occupations and three person counts $k \in \{ 2 , 3 , 4 \}$ . Left: baseline FLUX outputs. Right: SGS outputs. Rows show female-dominant (Nurses, Housekeepers) and male-dominant (Sheriffs, Construction Workers) professions. Baseline outputs are demographically homogeneous across all k values and both stereotype directions. SGS consistently produces intersectional demographic diversity — each person within the same scene is steered to a distinct demographic target — while also preserving occupational context (uniform, apron, badge, hard hat) and scene coherence across all conditions. Performance is consistent across $k \in \{ 2 , 3 , 4 \}$ , demonstrating that the segmentor reliably separates and independently steers two, three, and four persons within the same image.

and injects their combined residual once at the UNet’s bottleneck hidden state h. Our first attempt at reimplementing this for MM-DiT broadcast the same residual across every block of the joint-transformer stack; this reliably broke generation, producing incoherent output regardless of prompt or seed. Injecting the residual at a single representative block instead resolved this and produced coherent, steerable output. This also brings the reimplementation closer to the original method’s own single-bottleneck design than the all-block version we started from.

Shared adaptations. All three methods are applied only to the conditional half of the classifier-free-guidance batch, never the unconditional branch, consistent with how each was originally deployed. Training uses each target architecture’s own native noise parametrization and scheduler rather than a generic substitute.

Code Availability. Reference implementations of all three adapted baselines are included in the supplementary code (baselines/flux/ and baselines/sd35/): fairimagen.py, sdid.py, and respodiff.py, each containing the fit/training function and manipulation function described above with inline comments matching this section. SGS itself is in sgs/ (FLUX) and sgs sd3/ (SD3.5-Large).

## O. Human Evaluation Protocol

Automatic demographic classifiers may not fully capture how a human observer perceives diversity, prompt fidelity and preference. So we conducted a blind pairwise human evaluation comparing SGS against five baselines (Baseline, EA, FairImagen, RespoDiff and SDID) across 33 scenes spanning WinoBias occupations at $K \in \{ 2 , 3 , 4 \}$ people per image, using a fixed seed per scene shared across all six methods. Each rater was shown two images side by side from the same scene, generated by two different methods (one always SGS), with method identity withheld (labeled only ’Image A’/’Image B’) and left – right placement randomized per pair. For each pair, raters answered three questions (refer Figure 9):

![](images/cb11b11b24371c32897f2f0260d9ef4a1b2ba8b13aeb959459c5c89e17bb6000.jpg)  
Figure 8. Qualitative comparison on SD3.5-Large, mirroring Figure 2. Base, EA, FairImagen, RespoDiff, and SDID remain homoge neous or only mildly shifted across all four prompts; SGS independently diversifies each detected person while preserving occupationa context.

1. Which image shows more diversity in race and gender?   
2. Which image better matches the caption above?

3. Given a choice, which image would you prefer?

The first two questions offered a three way choice (Image A / about equal / Image B), while the last was a forced binary choice used to obtain a holistic preference signal. Each participants rated 10 pairs, two per comparison, ensuring balanced coverage across all five comparisons within a session.

Sessions in which a rater selected the identical response to every question throughout were excluded prior to analysis. For each comparison and question, we report the percentage of judgments preferring SGS (ties counted as half a preference) with a 95% Wilson score confidence interval and a two-sided exact binomial test against the 50% chance level (ties excluded from the significance test), alongside Krippendorff’s α (nominal) as a measure of interrater reliability[23]; results are reported in Table 2.

![](images/3d16f71bf7e5b179529c056d8755227678ab7b765cde820561e557782dae6ed6.jpg)  
Figure 9. The human evaluation interface. Raters see a captioned imagepair with method identity and position blinded, and answer three questions per pair: perceived diversity, prompt faithfulness, and overall preference.