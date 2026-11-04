# ScoreDial: Precise Control of Visual Attributes in Image Generation via Regressor Guidance from Limited Human Ratings

Jianbo Tang<sup>1</sup> Keita Ishii<sup>2</sup> Hiroki Sakamoto<sup>2</sup> Manabu Kikuchi<sup>2</sup> Katsuma Kitamura<sup>2</sup> Chiho Semba<sup>2</sup> Kenta Nishiyama<sup>2</sup> Teppei Mori<sup>2</sup> Yukihiro Kiwaki<sup>2</sup> Takeo Kato<sup>1\*</sup> <sup>1</sup>Keio University, Japan <sup>2</sup>Bridgestone Corporation, Japan

tang-jianbo@keio.jp, kato@mech.keio.ac.jp

{keita.ishii, hiroki.sakamoto2, manabu.kikuchi, katsuma.kitamura,

chiho.semba, kenta.nishiyama1, teppei.mori, yukihiro.kiwaki}@bridgestone.com

## Abstract

Requested-value control is stricter than continuous steering: given c on a human-rating scale, the generated image should attain that value from only limited ratings. We present ScoreDial, which fits a differentiable regressor R to 100–479 human-rated images; once R is fixed, a requested value c is defined on its output scale, and learned target conditioning is paired with error-adaptive guidance that repeatedly reduces |R(x) − c| during sampling. Across tire slip resistance, interior visual complexity, food desire-toeat, and soap translucency, using one set of sampler settings shared across all four, ScoreDial reaches requested values and achieves, on all four attributes, R-MAE at most 0.013 over a range of width 0.8 and Spearman ρ = 0.980, the maximum attainable under our evaluation protocol. Because R also drives sampling, we evaluate agreement with R separately from matched sweeps, an independent auditor fitted to the same ratings on a backbone none of the deployed regressors uses, and blinded human judgments. MPGD-flow also reaches the requested values under R, but ScoreDial has larger auditor displacement on every attribute, the largest among all compared methods, and a stronger MPGD-flow setting gives up target agreement without closing this gap. On every attribute, removing target conditioning leaves low R-MAE but more than halves auditor displacement, separating precise score correction from attribute movement. Blinded rating studies with 15 raters per attribute show that mean human ratings track c on tire and soap, with linear R<sup>2</sup> = 0.665 and

R<sup>2</sup> = 0.786, and in forced-choice comparisons of matched sweeps, raters chose ScoreDial’s as the clearer progression in at least 84% of judgments against every baseline.

## 1. Introduction

Visual appearance influences how products are perceived and evaluated [3, 11]. Such judgments concern perceived attributes of an appearance [11], and collecting consumer evaluations at scale can be expensive [4]. Generative models can automate the creation of design images [4, 37], but useful control may require more than asking for an attribute to become “more” or “less” pronounced: the desired outcome can be a particular numeric level [6, 8, 27].

We study requested-value control of task-specific human-rated attributes from only a few hundred rated images. Continuous conditional generators are developed in label-rich settings and condition on scalar regression labels [8, 13, 15, 16, 43], but sampling is open-loop to realized label error: the current sample’s discrepancy is not measured and fed back. Slider methods expose a continuous strength or direction [2, 7, 18, 19, 39, 40], not a requested value calibrated to a task-specific human-rating scale. Other controls derive attributes from image statistics, classifiers, or vision–language models [6, 43]; Hu et al. describe CLIPScore as “unreliable and often inaccurate” for text–image faithfulness [26]. Our setting combines limited human ratings with a numeric request on that scale.

ScoreDial turns those ratings into a closed-loop generator. A frozen pretrained visual backbone and a linear ridge head fitted to the human ratings form the differentiable regressor R, while a pretrained diffusion model is adapted to each image domain. Target conditioning, trained jointly with the domain LoRA on R-pseudo-labeled images, injects c into every transformer block and positions generation open-loop. Error-adaptive regressor guidance then closes the loop by repeatedly measuring R(x) − c on the clean-image estimate and applying a gradient-normalized, noise-weighted correction that shrinks with the remaining error. We write R-MAE for the mean of |R(x) − c| over a stated grid of targets and seeds.

![](images/39a2e1f8b6e3b776e6093315c927f8f500a80b36248ccfa24d09363378ea9809.jpg)  
Figure 1. Matched requested-value sweeps for the four compared methods (MPGD-flow at its selected strength); c increases from left to right in steps of 0.2. All methods share the frozen SD3.5 model, resolution, base sampler settings, prompts, and—within each attribute—the same seed.

Because R also drives sampling, low R-MAE establishes controller agreement but not by itself perceptual change. Training-free loss-based guidance can be more susceptible to adversarial gradients than classifier guidance [33], and learned objectives can be over-optimized [10, 20, 42]. We therefore separate target agreement under R from attribute movement, assessed by matched sweeps, an independent auditor using the same ratings but a backbone absent from all deployed regressors, and blinded human judgments.

Contributions. (1) We formulate and evaluate requestedvalue generation for task-specific human-rated visual attributes with only 100–479 rated images per attribute. (2) We introduce ScoreDial, which couples learned target conditioning with error-adaptive regressor feedback so that generation is first positioned by c and then corrected according to the realized score error. (3) Across four disparate attributes, we show empirically that convergence under the guidance regressor R and movement of the intended attribute are not interchangeable: MPGD-flow, and ScoreDial’s own feedback without target conditioning, reach the requested values under R but move the attribute less than ScoreDial, which has the largest auditor displacement among the compared methods on all four attributes. Blinded ratings on tire and soap confirm that mean perceived levels track c, and in forced-choice judgments raters chose ScoreDial’s progression as the clearer one in at least 84% of comparisons with every baseline.

## 2. Related Work

Human-rated objectives in design and affective generation. Human ratings have also shaped design-generation pipelines. Burnap et al. predict consumer aesthetic scores and automatically generate product designs [4]; Hayakawa et al. train StyleGAN on tread images in the top 40% under a slip-resistance regressor [22]; Yang et al. score and filter diffusion-generated products with a Kansei predictor [37]. Ibarrola et al. instead target requested valence–arousal– dominance values using an affect predictor trained from 13,913 rated words and 1,194 rated images [27]. Their optimization-based generators include affect-prediction error in the objective, while their Stable Diffusion procedure optimizes text-conditioning embeddings for the requested affect. ScoreDial targets task-specific scales learned from a few hundred rated images and repeatedly corrects the realized image score during sampling.

Continuous scalar-conditioned generation. CcGAN [13, 14], CCDM [15], and iCCDM [16] learn image generators conditioned on continuous scalar regression labels. Their published evaluation protocols use regressors trained on real images to measure label consistency after generation, but their sampling procedures do not feed the predicted label error back to update each sample. Att-Adapter adapts a pretrained diffusion model for continuous multi-attribute control using 16,336 face images, whose attributes come from landmarks and pretrained face estimators, and 2,030 car images [8]. CompSlider turns sixteen attribute values, obtained from attribute classifiers over roughly three million training images, into the conditioning a frozen textto-image model consumes [43]. Our conditioning stage follows the same principle, but its pseudo-labels come from an R fitted to only 100–479 human-rated images rather than the much larger labeled sets these methods use, and it is paired at inference with feedback based on $R ( \mathbf { x } ) - c .$

Intensity controls and sliders. AttriCtrl maps concrete attributes and vision–language-derived quantities to a unified [0, 1] scale and injects user values through learned encoders with a frozen diffusion backbone [6]. Slider methods instead learn or identify continuous directions in model parameters, text embeddings, or latent representations [2, 7, 18, 19, 39, 40]; their coefficients are not fitted to ratings of a task-specific human attribute. Across these works, reported attribute change is scored by vision– language models: ∆CLIP [2, 7, 18, 19], CLIP/SigLIP/BLIP similarity [40], or a multimodal language model [39], rather than ratings of that attribute.

Inference-time guidance and reliability. Universal Guidance [1] and MPGD [23] inject differentiable objectives during diffusion sampling. Related guidance methods include DPS [9], LGD [35], and TFG [38]. Gebhardt et al. use a learned regressor to guide image editing [21], whereas ScoreDial generates from noise. Our sampler adapts MPGD-LDM’s clean-estimate update and Universal Guidance’s stepwise refinement, but makes the correction magnitude depend explicitly on the current absolute score error and pairs this feedback stage with learned target conditioning.

## 3. Method

ScoreDial first fits a differentiable regressor R to human ratings (Sec. 3.1), then combines target conditioning that places generation near c (Sec. 3.2) with error-adaptive guidance that closes the residual gap during sampling (Sec. 3.3; Fig. 2). Both operate on R’s scale.

## 3.1. Attribute regressor from few labels

A frozen pretrained backbone maps each image to $\mathbf { f } \in \mathbb { R } ^ { d }$ and the human ratings fit a ridge head. We consider seven backbone–representation candidates: for base DINOv2 and DINOv3 [29, 34] (excluding register tokens), the class token, patch-token mean, and their concatenation; for CLIP ViT-B/16 [31], the projected pooled representation. Selection and guidance use identical preprocessing: RGB [0, 1], bicubic short-edge resize to 256, 224 × 224 center crop, and pretrained normalization (Sec. 4.2).

A ridge-regression head is fitted to standardized features:

$$
R ( { \bf x } ) = \beta ^ { \top } \tilde { \bf f } + b , \qquad \tilde { \bf f } = ( { \bf f } - \pmb { \mu } _ { f } ) / \pmb { \sigma } _ { f } .\tag{1}
$$

Here $\mu _ { f }$ and $\sigma _ { f }$ are the per-dimension mean and scale estimated from the training features, and $\beta$ and b are the ridge head parameters. Freezing the feature extractor keeps the learned component small, while the resulting image-toscore function remains end-to-end differentiable, allowing gradients to propagate from R through the decoded clean estimate to the predicted clean latent.

## 3.2. Domain adaptation and target conditioning

The generator is the official Stable Diffusion 3.5- medium release, whose rectified-flow transformer follows SD3 [17].<sup>1</sup> A single LoRA [25] adapts the transformer to each image domain and is trained jointly with the target-value embedding below; Sec. 4.1 specifies the fixed training pools.

To place generation according to the requested value, we inject c through the transformer’s existing scalarconditioning pathway. We first standardize it,

$$
\bar { c } = \frac { c - \mu _ { R } } { \sigma _ { R } } ,\tag{2}
$$

where $\mu _ { R }$ and $\sigma _ { R }$ are the mean and standard deviation of $R \ ' \mathrm { s }$ predictions on the domain images (Sec. 4.1). A sinusoidal projection followed by a two-layer MLP embeds c¯;

![](images/84a4bfbdb086f83c508b1698af5ac044e988b28cc0a6bb0367ade961ddd14272.jpg)  
Figure 2. ScoreDial separates target positioning from residual correction. (A) A frozen pretrained visual backbone and a fitted linear ridge head form the differentiable attribute regressor R. (B) A learned embedding of the requested value c is injected into every transformer block and trained jointly with a domain LoRA, providing open-loop target conditioning. During active guidance steps, R evaluates the decoded clean estimate; the squared-error gradient is normalized, and the clean latent is corrected with a step size determined by the current error and noise level.

the result is added to the combined timestep and pooled-text embedding that conditions every transformer block. The same value embedding is supplied to both conditional and unconditional classifier-free-guidance branches, preventing CFG from multiplying the requested value.

The deployed R pseudo-labels every domain-training image, so conditioning and feedback use the same scale. These labels supervise joint flow-matching training of the LoRA and value embedding; no additional human ratings are used. The embedding’s last layer is zero-initialized and is present at every denoising step. This path is open-loop: it sees c but not the current sample’s score, leaving residual error for feedback.

## 3.3. Error-adaptive regressor guidance

Target conditioning provides a location on the scale; guidance now corrects the realized sample. We use 30 denoising steps, indexed by $t \in \{ 0 , \ldots , 2 9 \}$ . At each guidance pass, let z be the current noisy latent and $\sigma \in ( 0 , 1 ]$ the current noise level. We compute the classifier-free-guidance (CFG) velocity [24], its clean prediction, the decoded RGB estimate, and the signed score error:

$$
\mathbf { v } = \mathbf { v } _ { \theta } ^ { \mathrm { u } } + \omega \big ( \mathbf { v } _ { \theta } ^ { \mathrm { c } } - \mathbf { v } _ { \theta } ^ { \mathrm { u } } \big ) ,\tag{3a}
$$

$$
\hat { \mathbf { z } } _ { 0 } = \mathbf { z } - \sigma \mathbf { v } ,\tag{3b}
$$

$$
\tilde { \mathbf { x } } _ { 0 } = \frac { \mathrm { c l i p } ( \mathcal { D } ( \hat { \mathbf { z } } _ { 0 } ) , - 1 , 1 ) + 1 } { 2 } ,\tag{3c}
$$

$$
\mathrm { e r r } = R ( \tilde { \mathbf { x } } _ { 0 } ) - c .\tag{3d}
$$

ω is the CFG scale; both branches share the target-value embedding (Sec. 3.2). D is the frozen Stable Diffusion 3.5 VAE decoder. Clipping and remapping to [0, 1] remain inside the computational graph and are followed by the resize, crop, and model-specific normalization of Sec. 3.1; on the rated images, this frozen autoencoder path alone changes R by at most 1.84% of the control range (supplementary material). We then differentiate the squared target error with respect to the predicted clean latent:

$$
\mathbf { g } = \nabla _ { \hat { \mathbf { z } } _ { 0 } } \mathrm { e r r } ^ { 2 } = 2 \operatorname { e r r } \nabla _ { \hat { \mathbf { z } } _ { 0 } } R ( \tilde { \mathbf { x } } _ { 0 } ) ,\tag{4a}
$$

$$
\hat { \mathbf { g } } = \frac { \mathbf { g } } { \left. \mathbf { g } \right. + 1 0 ^ { - 8 } } ,\tag{4b}
$$

where the norm is computed independently for each latent. We use the gradient for direction but not for step magnitude: after normalization, the guided clean prediction is

$$
\hat { \mathbf { z } } _ { 0 } ^ { g } = \hat { \mathbf { z } } _ { 0 } - s _ { t } \Vert \hat { \mathbf { z } } _ { 0 } \Vert \hat { \mathbf { g } } .\tag{5}
$$

The correction magnitude is set explicitly by the current error and noise level:

$$
\begin{array} { c } { { s _ { t } = \lambda _ { t } | \mathrm { e r r } | w ( \sigma ) , } } \\ { { w ( \sigma ) = w _ { \mathrm { m i n } } + ( 1 - w _ { \mathrm { m i n } } ) \sigma ^ { q } . } } \end{array}\tag{6a}
$$

(6b)

This form makes the feedback self-attenuating: $s _ { t }$ is proportional to $| R ( \tilde { \mathbf { x } } _ { 0 } ) - c |$ and approaches zero as the requested value is reached. Gradient normalization prevents the raw norm of ∇R from determining the update size, while $\left\| \hat { \mathbf { z } } _ { 0 } \right\|$ expresses the displacement relative to the clean-latent magnitude. Following guidance scaling [12] and timestep-dependent scaling in Universal Guidance [1], w(σ) weakens the correction as the sample becomes cleaner while retaining a floor $w _ { \mathrm { m i n } }$

With normalized progress $ { p } ^ { \mathrm { ~ ~ } } =  { \ t / 3 0 }$ , the guidance

coefficient is

$$
\lambda _ { t } = \left\{ \begin{array} { l l } { 0 , } & { p < 0 . 2 , } \\ { \lambda _ { \operatorname* { m a i n } } , } & { 0 . 2 \le p \le 0 . 8 , } \\ { \lambda _ { \mathrm { t a i l } } , } & { p > 0 . 8 . } \end{array} \right.\tag{7}
$$

The first 20% of the trajectory therefore uses targetconditioned CFG without regressor guidance, and the error factor and noise weighting remain active in both guidance intervals. We set $\lambda _ { \operatorname* { m a i n } } = 0 . 4 , \lambda _ { \mathrm { t a i l } } = 0 . 0 5 , w _ { \operatorname* { m i n } } = 0 . 2 ,$ and $q = 1$ once in preliminary tire experiments and reuse them unchanged for all four attributes; no sampler setting is tuned per attribute.

Sampling transition and refinement. We apply the correction to the predicted clean latent rather than back-propagating through the denoiser, following the clean-estimate update in MPGD-LDM [23] (Algorithm 2). The current velocity also defines the corresponding noise prediction

$$
\hat { \pmb { \epsilon } } = \mathbf { z } + ( 1 - \sigma ) \mathbf { v } .\tag{8}
$$

After the final guidance pass at a denoising step, the next rectified-flow state is

$$
\begin{array} { r } { \mathbf { z } _ { \mathrm { n e x t } } = ( 1 - \sigma _ { \mathrm { n e x t } } ) \hat { \mathbf { z } } _ { 0 } ^ { g } + \sigma _ { \mathrm { n e x t } } \hat { \mathbf { \epsilon } } . } \end{array}\tag{9}
$$

Here $\sigma _ { \mathrm { n e x t } }$ is the next schedule value. For every step with $\lambda _ { t } ~ > ~ 0$ , we additionally adapt Universal Guidance’s Universal Stepwise Refinement [1]. We use $k = 2$ guidance passes in total. Between the two passes, the state is re-noised to a level $\sigma _ { \mathrm { b a c k } }$ as far above σ as $\sigma _ { \mathrm { n e x t } }$ is below it:

$$
\sigma _ { \mathrm { b a c k } } = \sigma + ( \sigma - \sigma _ { \mathrm { n e x t } } ) , \qquad \kappa = \frac { 1 - \sigma _ { \mathrm { b a c k } } } { 1 - \sigma _ { \mathrm { n e x t } } } ,
$$

$$
\mathbf { z } \gets \kappa \mathbf { z } _ { \mathrm { n e x t } } + \sqrt { \sigma _ { \mathrm { b a c k } } ^ { 2 } - ( \kappa \sigma _ { \mathrm { n e x t } } ) ^ { 2 } } \epsilon , \epsilon \sim \mathcal { N } ( 0 , \mathbf { I } ) .\tag{10}
$$

Because $\kappa { \bf z } _ { \mathrm { n e x t } } = ( 1 - \sigma _ { \mathrm { b a c k } } ) \hat { { \bf z } } _ { 0 } ^ { g } + \kappa \sigma _ { \mathrm { n e x t } } \hat { \epsilon }$ , the estimated noise is carried into the refinement state and only the remaining noise is sampled afresh. The second pass recomputes the velocity, clean estimate, score error, and gradient at $\sigma _ { \mathrm { b a c k } }$ , using the nearest schedule timestep for denoiser conditioning; the final transition still advances to this step’s $\sigma _ { \mathrm { n e x t } }$ . Steps with $\lambda _ { t } ~ = ~ 0$ use a single target-conditioned CFG Euler transition with neither regressor guidance nor refinement.

## 4. Experiments

We test whether ScoreDial reaches requested values and moves the corresponding attribute (Sec. 4.3), which components supply precision and movement (Sec. 4.4), and whether people perceive the requested progression, also relative to the baselines (Sec. 4.5).

## 4.1. Setup

Source supervision. We evaluate four attributes with different semantics and rating protocols. Tire slip resistance uses five-point Likert ratings from Hayakawa et al. [22]. Their 120 tread entries contain one duplicated image with conflicting means; dropping both entries leaves 118 rated images. Interior visual complexity uses the complete 100-image Interior Design category of SAVOIAS [32], whose 0–100 scores were derived by the dataset authors from crowdsourced pairwise comparisons using Bradley–Terry and matrix completion. Food desireto-eat uses the 479 CROCUFID images with 0–100 visual-analogue ratings [36]. Soap translucency uses the 150 real photographs rated on seven-point scales by Liao et al. [28], excluding their 150 model-generated soap images; following their protocol, we min–max normalize each observer’s ratings and then average across observers.

Generator-training images. Domain adaptation and target conditioning share one image pool per attribute: 984 tires, 100 interiors, 479 foods, and 800 soaps. The tire pool begins with all 534 unique tread images in the patent-sourced archive of Hayakawa et al. [22]; the 535-file archive contains one exact byte-level duplicate, and 118 unique images carry ratings. For generator training only, the original patent line drawings of these treads are upsampled to $1 0 2 4 ^ { 2 }$ with Topaz Gigapixel, median-filtered with a $7 \times 7$ window, and standardized per image to mean 128 and standard deviation 55, clipped to [0, 255]. R is always fitted to the rated archive images, and no upsampled image enters a reported measurement. To reduce rendering artifacts observed at extreme-score tire targets, we oversample the score tails deterministically: using deployed-R scores of the archive images, horizontal, vertical, and both-axis flips are added for every image with score $\leq 0 . 2 5 2$ (65 images) or $\geq ~ 0 . 7 4 6$ (85 images), producing 984 training images. Interior and food use exactly their rated sets for generator training. The 800 soaps are sampled uniformly at random from the 8,085-photograph Translucent Image Dataset [28] before scoring. An SSCD copy-detection check [30] finds no reported image unusually close to its generator-training pool (supplementary material).

Score ranges and requested targets. All requested values are represented on [0, 1], so the error factor $| R ( \mathbf { x } ) - c |$ has a common scale across attributes: the observed tire and food label ranges are mapped affinely, interior’s defined 0–100 scale by its endpoints, and the averaged soap scores already lie in [0, 1]. Every control experiment and ablation uses the same five targets, $c \in \{ 0 . 1 , 0 . 3 , 0 . 5 , 0 . 7 , 0 . 9 \}$ , with ten seeds per target.

Training and sampling. We use Stable Diffusion 3.5- medium through Hugging Face diffusers in PyTorch on one NVIDIA RTX 5090. The fixed domain prompts are “a tire tread pattern, top-down view, grayscale,” “a photograph of a room interior,” “a plate of food on white background,” and “a bar of soap.” For tire/interior/food/soap respectively, a rank-32 LoRA on the transformer’s attention projections and the target-value embedding are trained jointly for 3,000/3,000/2,000/6,000 flow-matching steps, with batch 4, AdamW at constant learning rate $1 0 ^ { - \overline { { 4 } } }$ , and horizontal-flip probability 0.5. Training and all reported generation use $1 0 2 4 ^ { 2 }$ images. ScoreDial uses 30 Euler steps, CFG 5, and the guidance schedule of Sec. $3 . 3 . \quad " \pm "$ denotes mean ± population SD over the stated runs or seeds.

## 4.2. Attribute regressors: selection and evaluation

Because the same labeled set is used to choose a representation and to fit the final regressor, we separate evaluation of the selection procedure from selection of the deployment model [5]. Repeated nested cross-validation of the full selection procedure (supplementary material) gives tire $R ^ { 2 } = 0 . 6 9 7 \pm 0 . 0 4 5$ , interior $R ^ { 2 } = 0 . 6 1 3 \pm 0 . 0 3 2 .$ food $R ^ { 2 } ~ = ~ 0 . 7 0 5 \pm 0 . 0 0 9$ , and soap $R ^ { 2 } ~ = ~ 0 . 9 0 9 \pm 0 . 0 0 6$ For deployment, repeated 5-fold CV on all labels selects DINOv2 patch-token mean with $\alpha ~ = ~ 5 0$ for tire, CLIP projected pooled output with $\alpha ~ = ~ 1 0 0$ for interior and food, and DINOv3 class-token plus patch-token-mean concatenation with $\alpha = 1 0 0$ for soap; each head is refit on all labels and fixed before generation begins.

## 4.3. Requested-value control: agreement and movement

We evaluate two axes because R participates in generation: target agreement, whether R scores the image at the requested value, and attribute movement, whether changing the request changes the attribute under evidence the sampler never optimizes.

ScoreDial target agreement. A requested value is counted as reached when $| R ( \mathbf { x } ) - c | \leq 0 . 0 5$ . ScoreDial reaches every tested cell. Across the 200 target–seed cells, the median error is at most 0.0124, the 90th percentile at most 0.0224, and the largest single-cell error is 0.0331 on food. The requested variable has five tied values while generated scores are distinct, so Spearman correlation cannot exceed 0.980 on this grid; ScoreDial reaches that ceiling on all four attributes, with no target cell ranked out of order.

Matched comparison protocol. Numeric prompt, MPGDflow, UG-flow, and ScoreDial use the same frozen SD3.5 model, $1 0 2 4 ^ { 2 }$ resolution, 30-step Euler base schedule, CFG 5, fixed domain prompts, and test seeds. Baseline LoRAs are trained exactly as ScoreDial’s except that they do not receive the target-value embedding. MPGD-flow and UGflow use R only during sampling; Numeric prompt uses no regressor guidance. With batch size 1 on the RTX 5090 of Sec. 4.1, per-image sampling time averaged over the ten test seeds and the four attributes is 5.5 s for Numeric prompt, 16.6 s for MPGD-flow, 48.2 s for UG-flow, and 29.5 s

for ScoreDial, at peak memory 7.7/11.9/26.0/11.9 GB;   
per-attribute values differ from these means by at most 2%.

Numeric prompt appends “with perceived [attribute] rated c on a zero-to-one scale,” using the mapped scales of Sec. 4.1. MPGD-flow adapts MPGD-LDM Algorithm 2 [23] to rectified flow: one fixed-coefficient loss-gradient step is applied to the predicted clean latent, with neither refinement nor denoiser back-propagation. UG-flow adapts forward Universal Guidance [1] to rectified flow and differentiates through the denoiser. It uses no backward optimization steps, matching Universal Guidance’s Stable Diffusion setting, and uses Universal Stepwise Refinement with the same $k = 2$ as ScoreDial.

The continuous scalar-conditioned generators discussed in Sec. 2 train their own scalar-conditioned models rather than operating as control mechanisms on the matched frozen SD3.5 backbone [13, 15, 16]; both these and the closest methods built on pretrained diffusion models [8, 43] report published settings substantially more label-rich than the 100–479 human-rated images available here. We therefore do not include them in this comparison. The conditioning-only ablation (Sec. 4.4) additionally evaluates the same open-loop scalar-conditioning principle under the matched protocol on every attribute.

Guidance-strength selection. Strengths are chosen using ten seeds disjoint from all test seeds. For each guidance baseline we search {1, 3, 10, 30, 100, 300} at endpoints 0.1 and 0.9 and choose the lowest endpoint R-MAE among settings that still produce images of the domain; endpoint selection yields one fixed coefficient for the full requested range. MPGD-flow selects 100/30/30/10 for tire/interior/food/soap, each strictly inside the ladder, with error increasing on both neighboring sides. UG-flow’s error decreases at every rung, but strengths from 100 upward produce hard-edged color blocks rather than domain images for every attribute. We therefore report UG-flow at 30, the strongest domain-valid rung and hence the lowest-error eligible setting. This rule optimizes target agreement alone. UG-flow’s selected rung is already its strongest domain-valid one, whereas MPGD-flow’s lies inside the ladder, so a stronger setting could trade agreement for movement. We thus also report MPGD-flow one rung stronger, at 300/100/100/30 (MPGD-flow<sup>+</sup>), a setting less restrictive in the precision/movement trade-off.

What target agreement alone says. Numeric prompt has the largest R-MAE on every attribute and near-zero rank correlation. UG-flow and MPGD-flow<sup>+</sup> improve on it but remain above the 0.05 tolerance on all four, while ScoreDial has lower R-MAE than both in every domain. More revealingly, applying the same tolerance to the grid mean, MPGD-flow and ScoreDial are within it on all four, while the other three settings are not. Under R alone, MPGD-flow and ScoreDial therefore both appear to solve

Table 1. Matched comparison. Ten test seeds are shared by all methods and all requested values (N=50 images per method and attribute); $\mathbf { M P G D - f l o w } ^ { + }$ is MPGD-flow one strength rung above its selected one (Sec. 4.3). Top: agreement with the deployed regressor $R ; \pm { \mathrm { i s } }$ the SD of the per-seed mean, and $\rho$ is Spearman correlation between requested value and R of the generated image. Bottom: independent-auditor endpoint displacement $A ( 0 . 9 ) \ : - \ : A ( 0 . 1 )$ within one seed (mean ± SD over the 10 seeds); auditors are attribute-specific, so comparisons are within a row.
<table><tr><td rowspan="2">Attribute</td><td colspan="2">Numeric prompt</td><td colspan="2">MPGD-flow</td><td colspan="2">MPGD-flow+</td><td colspan="2">UG-flow</td><td colspan="2">ScoreDial</td></tr><tr><td>R-MAE↓</td><td> $\rho \uparrow$ </td><td>R-MAE↓</td><td> $\rho \uparrow$ </td><td>R-MAE↓</td><td>ρ↑</td><td>R-MAE↓</td><td> $\rho \uparrow$ </td><td>R-MAE↓</td><td> $\rho \uparrow$ </td></tr><tr><td>Tire</td><td>0.2672±0.0303</td><td>-0.061</td><td> $0 . 0 0 3 1 { \pm } 0 . 0 0 1 5$ </td><td>0.980</td><td> $0 . 1 2 7 6 { \scriptstyle \pm 0 . 0 9 0 6 }$ </td><td>0.862</td><td> $0 . 1 7 8 8 { \pm } 0 . 0 2 7 9$ </td><td>0.764</td><td> $0 . 0 1 2 9 { \pm } 0 . 0 0 2 8$ </td><td>0.980</td></tr><tr><td>Interior</td><td> $0 . 3 9 0 5 { \pm } 0 . 0 5 9 2$ </td><td>-0.039</td><td> $0 . 0 2 4 2 { \pm } 0 . 0 3 8 4$ </td><td>0.961</td><td> $0 . 1 3 0 4 { \scriptstyle \pm 0 . 0 6 5 5 }$ </td><td>0.869</td><td> $0 . 1 7 9 1 { \scriptstyle \pm 0 . 0 5 5 1 }$ </td><td>0.789</td><td> $0 . 0 0 6 9 { \pm } 0 . 0 0 0 9$ </td><td>0.980</td></tr><tr><td>Food</td><td> $0 . 2 6 0 0 { \scriptstyle \pm 0 . 0 1 8 7 }$ </td><td>-0.073</td><td> $0 . 0 1 4 9 { \pm } 0 . 0 1 1 8$ </td><td>0.980</td><td> $0 . 0 8 3 2 { \scriptstyle \pm 0 . 0 4 7 0 }$ </td><td>0.906</td><td> $0 . 1 8 6 6 { \pm } 0 . 0 3 5 4$ </td><td>0.664</td><td> $0 . 0 0 8 6 { \pm } 0 . 0 0 2 6$ </td><td>0.980</td></tr><tr><td>Soap</td><td> $0 . 2 6 0 0 { \pm } 0 . 0 2 1 6$ </td><td>0.027</td><td> $0 . 0 1 3 4 { \pm } 0 . 0 1 1 4$ </td><td>0.970</td><td> $0 . 0 7 0 2 { \scriptstyle \pm 0 . 0 2 5 6 }$ </td><td>0.936</td><td> $0 . 1 8 1 6 { \pm } 0 . 0 3 9 6$ </td><td>0.718</td><td> $0 . 0 0 7 2 { \scriptstyle \pm 0 . 0 0 2 0 }$ </td><td>0.980</td></tr><tr><td colspan="9">Independent-auditor endpoint displacement A(0.9) — A(0.1) ↑</td><td colspan="2"></td></tr><tr><td>Tire</td><td colspan="2"> $- 0 . 0 1 4 { \pm } 0 . 0 8 1$ </td><td colspan="2"> $- 0 . 0 2 1 { \scriptstyle \pm 0 . 0 9 5 }$ </td><td colspan="2"> $0 . 0 5 3 { \pm } 0 . 1 3 9$ </td><td colspan="2"> $0 . 0 9 6 { \pm } 0 . 1 9 6$ </td><td colspan="2"> $0 . 2 8 5 { \pm } 0 . 2 0 9$ </td></tr><tr><td>Interior</td><td colspan="2"> $0 . 0 0 1 { \scriptstyle \pm 0 . 0 5 6 }$ </td><td colspan="2"> $0 . 0 5 5 { \pm } 0 . 0 6 2$ </td><td colspan="2"> $0 . 1 5 7 { \pm } 0 . 1 0 2$ </td><td colspan="2"> $0 . 1 8 5 { \pm } 0 . 1 7 1$ </td><td colspan="2"> $0 . 4 1 0 { \pm } 0 . 0 8 2$ </td></tr><tr><td>Food</td><td colspan="2"> $- 0 . 0 1 7 { \pm } 0 . 0 8 3$ </td><td colspan="2"> $0 . 1 2 4 \pm 0 . 0 7 4$ </td><td colspan="2"> $0 . 1 3 4 \pm 0 . 1 1 8$ </td><td colspan="2"> $0 . 0 7 6 { \scriptstyle \pm 0 . 1 0 6 }$ </td><td colspan="2"> $0 . 1 8 7 { \pm } 0 . 1 0 0$ </td></tr><tr><td>Soap</td><td colspan="2"> $0 . 0 2 7 { \scriptstyle \pm 0 . 0 4 1 }$ </td><td colspan="2"> $0 . 0 7 8 { \scriptstyle \pm 0 . 0 8 1 }$ </td><td colspan="2"> $0 . 1 5 2 { \pm } 0 . 1 0 3$ </td><td colspan="2"> $0 . 3 1 2 { \scriptstyle \pm 0 . 1 9 7 }$ </td><td colspan="2"> $0 . 4 5 2 { \pm } 0 . 1 3 7$ </td></tr></table>

requested-value control.

Independent attribute movement. For each attribute we fit an independent auditor: a ridge head on frozen SigLIP features [41], a backbone none of the deployed regressors uses. It uses the same human ratings as R, holding the attribute definition fixed while changing representation and predictor, and is never used for generator training, sampling, or strength selection. Each auditor attains nested-CV $R ^ { 2 } \geq$ 0.61, following the practice of evaluating generated samples with a predictor trained apart from the generator [13, 15].

Because sampling explicitly closes the error under $R ,$ not under the auditor, we do not treat the auditor as a second calibration target or compare its absolute reading A directly with c. Instead, for each seed we measure how far that reading travels across the requested range, $A ( 0 . 9 ) { - } A ( 0 . 1 )$ with both endpoint images generated from the same noise.

The matched sweeps change the interpretation: Numeric prompt and MPGD-flow vary little between endpoint requests, whereas ScoreDial progresses across the range (Fig. 1; all test images in the supplementary material). ScoreDial also has larger auditor displacement than MPGDflow on every attribute (Table 1, bottom). MPGD-flow thus exposes the central failure mode: reaching the requested value under R does not imply comparable movement under an independent predictor of the same human-rated attribute. A stronger setting does not close this gap: MPGD-flow<sup>+</sup> has larger auditor displacement than MPGD-flow on every attribute but gives up target agreement (Table 1), and its displacement remains below ScoreDial’s on all four.

UG-flow shows the complementary failure mode: substantial movement without reaching the requested values under R. Its interior sweep also changes scene content between neighboring targets—washbasin, kitchen, living room—whereas ScoreDial keeps one room and changes its complexity. Together, the two panels of Table 1 show that requested-value control needs both target agreement and attribute movement; ScoreDial is the only compared method that reaches the requested values on all four attributes and has the largest auditor displacement on all four.

![](images/af998b636b8c3420db88b530ec26013d3d2e16241e8d93c01b1defa9cd4e2bc8.jpg)  
Figure 3. Matched Full and G-only tire outputs at requested values 0.1/0.5/0.9; these are the only two configurations that retain low error under R on every attribute (Table 2).

## 4.4. Ablation of ScoreDial

We ablate ScoreDial on all four attributes to isolate the roles of its two control paths and the design of its feedback update. Full is ScoreDial as proposed. G-only removes the target-value offset while retaining the same jointly trained LoRA, so c enters only through regressor guidance.

No |R(x) − c| factor replaces Eq. (6a) by $s _ { t } = \lambda _ { t } w ( \sigma )$ while retaining gradient normalization, latent-norm scaling, and refinement. Direct raw gradient uses $\hat { \mathbf { z } } _ { 0 } ^ { g } = \hat { \mathbf { z } } _ { 0 } - \lambda _ { t } \mathbf { g } ,$ retaining $\lambda _ { t }$ and refinement but removing gradient normalization, latent-norm scaling, and w(σ). Conditioning only keeps the target-value embedding and disables regressor guidance.

The update design matters for target precision (Table 2). On every attribute, removing only the $| R ( { \bf x } ) - c |$ factor raises R-MAE from Full’s 0.007–0.013 to 0.067–0.082, and stepping along the raw gradient raises it further, to 0.178–0.215. Conditioning alone has R-MAE 0.159–0.241.

Table 2. Ablation on all four attributes: R-agreement and independent-auditor endpoint displacement ∆ on the same $5 \times 1 0$ target–seed grid (N=50 per configuration and attribute). Full is the ScoreDial run of Table 1.
<table><tr><td></td><td colspan="2">Tire</td><td colspan="2">Interior</td><td colspan="2">Food</td><td colspan="2">Soap</td></tr><tr><td>Configuration</td><td>R-MAE↓</td><td>∆↑</td><td>R-MAE↓</td><td>∆↑</td><td>R-MAE↓</td><td>∆↑</td><td>R-MAE↓</td><td>∆↑</td></tr><tr><td>Full</td><td>0.0129±0.0028</td><td></td><td>0.285±0.209 0.0069±0.0009</td><td> $0 . 4 1 0 { \scriptstyle \pm 0 . 0 8 2 }$ </td><td> $0 . 0 0 8 6 \pm 0 . 0 0 2 6$ </td><td> $0 . 1 8 7 { \pm } 0 . 1 0 0$ </td><td>0.0072±0.0020 0.452±0.137</td><td></td></tr><tr><td>G-only</td><td>0.0154±0.0041</td><td> $0 . 0 4 1 { \pm } 0 . 0 7 5$ </td><td> $0 . 0 0 8 4 \pm 0 . 0 0 2 3$ </td><td> $0 . 1 6 0 { \pm } 0 . 1 2 0$ </td><td> $0 . 0 0 8 4 \pm 0 . 0 0 4 1$ </td><td> $0 . 0 9 0 { \scriptstyle \pm 0 . 0 9 5 }$ </td><td>0.0086±0.0023 0.114±0.088</td><td></td></tr><tr><td>Conditioning only</td><td>0.1960±0.0482</td><td> $0 . 1 8 4 \pm 0 . 1 8 1$ </td><td> $0 . 2 0 1 1 { \scriptstyle \pm 0 . 0 4 9 6 }$ </td><td> $0 . 2 9 5 { \pm } 0 . 1 7 4$ </td><td> $0 . 2 4 0 6 \pm 0 . 0 6 4 7$ </td><td> $0 . 0 7 9 { \pm } 0 . 1 3 9$ </td><td>0.1594±0.0616 0.465±0.169</td><td></td></tr><tr><td>Direct raw gradient</td><td>0.1789±0.0316</td><td> $0 . 2 5 4 { \pm } 0 . 2 0 2$ </td><td> $0 . 1 8 0 1 { \scriptstyle \pm 0 . 0 4 6 3 }$ </td><td> $0 . 3 2 0 { \pm } 0 . 1 1 3$ </td><td>0.2146±0.0490 0.084±0.115</td><td></td><td> $0 . 1 7 8 0 { \scriptstyle \pm 0 . 0 5 8 3 }$ </td><td> $0 . 3 6 7 { \scriptstyle \pm 0 . 1 4 6 }$ </td></tr><tr><td>No |R(x) − c| factor</td><td> $0 . 0 7 1 5 { \pm } 0 . 0 1 1 0$ </td><td> $0 . 1 5 9 { \pm } 0 . 1 2 7$ </td><td> $0 . 0 6 6 5 { \scriptstyle \pm 0 . 0 1 0 5 }$ </td><td> $0 . 0 6 0 { \scriptstyle \pm 0 . 0 6 5 }$ </td><td> $0 . 0 7 6 9 { \scriptstyle \pm 0 . 0 1 9 9 }$ </td><td> $0 . 0 0 4 \pm 0 . 0 9 0$ </td><td> $0 . 0 8 2 5 { \pm } 0 . 0 1 9 7$ </td><td> $0 . 1 7 5 { \pm } 0 . 1 0 4$ </td></tr></table>

G-only is the only ablation besides Full with low R-MAE on every attribute (0.008–0.015). Feedback alone can therefore drive R close to c (Fig. 3; all attributes and configurations in the supplementary material), yet removing target conditioning more than halves auditor displacement on every attribute. Conditioning-only gives auditor displacement of 0.079–0.465 but poor R-agreement. The two paths thus separate: target conditioning contributes movement, while feedback supplies precise correction. G-only reproduces the MPGD-flow pattern—low error under R is not sufficient evidence of attribute movement.

## 4.5. Human evaluation

We finally ask whether direct human perception follows the requested value. We run the same blinded protocol on tire and soap. For each attribute, 15 raters score 40 ScoreDial images (five requested values × eight seeds) without seeing the target. Ratings use the source scale for that attribute: 5 points for tire slip resistance [22] (1: looks easy to slip; 5: looks difficult to slip) and 7 points for soap translucency [28] (1: low; 7: high). Each participant sees 48 randomized presentations including 8 blinded repeats.

Participation is voluntary with informed consent; identifiers are pseudonymized, and no demographic or sensitive data are collected. R is never shown to raters and plays no role in rating collection. Before analysis, a rater would be excluded for using at most two distinct values or for mean absolute repeat discrepancy above two points; no rater meets either criterion in either study. Repeated presentations are averaged within rater and then across raters for each image. Mean human ratings track c in both studies: tire gives linear $R ^ { 2 } ~ = ~ 0 . 6 6 5$ and tie-corrected Spearman $\rho = 0 . 8 2 7 \mathrm { , }$ , while soap gives $R ^ { 2 } = 0 . 7 8 6$ and $\rho = 0 . 8 6 0$

Forced-choice comparison with baselines. The rating studies cover ScoreDial alone, so we compare perceived progression with the baselines in a two-alternative forcedchoice (2AFC) study on tire and soap, with the same recruitment and consent. Each screen shows two rows of five images generated from the same test seed at $c = 0 . 1 , \ldots , 0 . 9$ from left to right: one row from ScoreDial and one from Numeric prompt, MPGD-flow, MPGDflow<sup>+</sup>, or UG-flow, giving 40 pairs over the four baselines and ten test seeds. Raters are told that both rows request an increasing level from left to right and choose the row that becomes more clearly slip-resistant (tire) or translucent (soap), judging the change rather than image quality; there is no tie option. ScoreDial’s row is on top in a random half of the pairs for each rater, and 8 pairs are repeated with the rows swapped. Under criteria fixed before analysis, a rater is excluded for giving the same answer on every screen or for choosing the same method on a repeat and its original in at most 3 of the 8 repeats, leaving 15 tire and 15 soap raters. Table 3 reports the fraction of non-repeat judgments preferring ScoreDial. ScoreDial is preferred in 84–98% of judgments across baselines and attributes.

Table 3. 2AFC: fraction of judgments preferring ScoreDial’s sweep over the baseline’s, with 95% percentile bootstrap intervals over raters and seeds (0.5 means no preference).
<table><tr><td>ScoreDial vs.</td><td>Tire</td><td>Soap</td></tr><tr><td>Numeric prompt</td><td>0.98 [0.94, 1.00]</td><td>0.98 [0.93, 1.00]</td></tr><tr><td>MPGD-flow</td><td>0.95 [0.90, 0.99]</td><td>0.97 [0.92, 1.00]</td></tr><tr><td> $\mathbf { M P G D - f l o w ^ { + } }$ </td><td>0.98 [0.92, 1.00]</td><td>0.97 [0.90, 1.00]</td></tr><tr><td>UG-flow</td><td>0.85 [0.71, 0.95]</td><td>0.86 [0.77, 0.94]</td></tr></table>

## 5. Conclusion

ScoreDial couples learned target conditioning with erroradaptive feedback for requested-value control from a few hundred human ratings. One sampler configuration reaches every tested cell on all four attributes $( R { \mathbf { - } } { \mathbf { M A E } } \leq 0 . 0 1 3 ,$ $\rho = 0 . 9 8 0 )$ . Crucially, agreement with the guidance regressor R is not equivalent to attribute control: MPGD-flow also reaches the requested values under R, while ScoreDial has larger independent-auditor displacement on every attribute, even against a stronger MPGD-flow setting, and the largest among all compared methods. On every attribute, feedbackonly retains low R error while auditor displacement drops by more than half, whereas conditioning-only lacks accurate requested-value agreement. Blinded human studies on tire and soap show mean ratings tracking c, and in forcedchoice comparisons raters chose ScoreDial’s progression as the clearer one in at least 84% of judgments against every baseline. Code and generation manifests are available at https://github.com/Kamijo-Gen/ScoreDial. Limitations. The soap regressor is cross-validated by image rather than by physical soap, so its $R ^ { 2 }$ may be optimistic.

## References

[1] A. Bansal et al. Universal guidance for diffusion models. In ICLR, 2024.

[2] S. A. Baumann et al. Continuous, subject-specific attribute control in T2I models by identifying semantic directions. In CVPR, pages 13231–13241, 2025.

[3] P. H. Bloch. Seeking the ideal form: Product design and consumer response. Journal ofMarketing, 59(3):16–29, 1995.

[4] A. Burnap, J. R. Hauser, and A. Timoshenko. Product aesthetic design: A machine learning augmentation. Marketing Science, 42(6):1029–1056, 2023.

[5] G. C. Cawley and N. L. C. Talbot. On over-fitting in model selection and subsequent selection bias in performance evaluation. Journal of Machine Learning Research, 11:2079–2107, 2010.

[6] D. Chen et al. AttriCtrl: A generalizable framework for controlling semantic attribute intensity in diffusion models. In ICLR, 2026.

[7] P.-Y. Chiu, I.-S. Fang, and J.-C. Chen. Text Slider: Efficient and plug-and-play continuous concept control for image/video synthesis via LoRA adapters. In WACV, pages 613–622, 2026.

[8] W. Cho et al. Att-Adapter: A robust and precise domainspecific multi-attributes T2I diffusion adapter via conditional VAE. In ICCV, pages 15626–15635, 2025.

[9] H. Chung et al. Diffusion posterior sampling for general noisy inverse problems. In ICLR, 2023.

[10] K. Clark et al. Directly fine-tuning diffusion models on differentiable rewards. In ICLR, 2024.

[11] N. Crilly, J. Moultrie, and P. J. Clarkson. Seeing things: Consumer response to the visual domain in product design. Design Studies, 25(6):547–577, 2004.

[12] P. Dhariwal and A. Nichol. Diffusion models beat GANs on image synthesis. In NeurIPS, 2021.

[13] X. Ding et al. CcGAN: Continuous conditional generative adversarial networks for image generation. In ICLR, 2021.

[14] X. Ding et al. Continuous conditional generative adversarial networks: Novel empirical losses and label input mechanisms. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(7):8143–8158, 2023.

[15] X. Ding et al. CCDM: Continuous conditional diffusion models for image generation. IEEE Transactions on Multimedia, 28:4360–4372, 2026.

[16] X. Ding et al. Enhancing diffusion-based quantitatively controllable image generation via matrix-form EDM and adaptive vicinal training. arXiv:2602.02114, 2026.

[17] P. Esser et al. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024.

[18] R. Ezra et al. FreeSliders: Training-free, modality-agnostic concept sliders for fine-grained diffusion control in images, audio, and video. arXiv:2511.00103, 2025.

[19] R. Gandikota et al. Concept sliders: LoRA adaptors for precise control in diffusion models. In ECCV, pages 172–188, 2024.

[20] L. Gao, J. Schulman, and J. Hilton. Scaling laws for reward model overoptimization. In ICML, pages 10835–10866, 2023.

[21] C. Gebhardt et al. Regressor-guided generative image editing balances user emotions to reduce time spent online. arXiv:2501.12289, 2025.

[22] R. Hayakawa, R. Noji, and T. Kato. Generating tyre tread designs using a sensory evaluation regression model and a generative model. Journal of Engineering Design, 36(1):19–51, 2025.

[23] Y. He et al. Manifold preserving guided diffusion. In ICLR, 2024.

[24] J. Ho and T. Salimans. Classifier-free diffusion guidance. arXiv:2207.12598, 2022.

[25] E. J. Hu et al. LoRA: Low-rank adaptation of large language models. In ICLR, 2022.

[26] Y. Hu, B. Liu, J. Kasai, Y. Wang, M. Ostendorf, R. Krishna, and N. A. Smith. TIFA: Accurate and interpretable text-toimage faithfulness evaluation with question answering. In ICCV, pages 20406–20417, 2023.

[27] F. Ibarrola, R. Lulham, and K. Grace. Affect-conditioned im age generation. IEEE Transactions on Affective Computing, 15(4):2169–2179, 2024.

[28] C. Liao, M. Sawayama, and B. Xiao. Unsupervised learning reveals interpretable latent representations for translucency perception. PLoS Computational Biology, 19(2):e1010878, 2023.

[29] M. Oquab et al. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024.

[30] E. Pizzi, S. D. Roy, S. N. Ravindra, P. Goyal, and M. Douze. A self-supervised descriptor for image copy detection. In CVPR, pages 14512–14522, 2022.

[31] A. Radford et al. Learning transferable visual models from natural language supervision. In ICML, 2021.

[32] E. Saraee, M. Jalal, and M. Betke. SAVOIAS: A diverse, multi-category visual complexity dataset. arXiv:1810.01771, 2018.

[33] Y. Shen et al. Understanding and improving training-free loss-based diffusion guidance. In NeurIPS, 2024.

[34] O. Siméoni et al. DINOv3. arXiv:2508.10104, 2025.

[35] J. Song et al. Loss-guided diffusion models for plug-and play controllable generation. In ICML, 2023.

[36] A. Toet et al. CROCUFID: A cross-cultural food image database for research on food elicited affective responses. Frontiers in Psychology, 10:58, 2019.

[37] C. Yang, F. Liu, and J. Ye. A product form design method integrating Kansei engineering and diffusion model. Advanced Engineering Informatics, 57:102058, 2023.

[38] H. Ye et al. TFG: Unified training-free guidance for diffusion models. In NeurIPS, 2024.

[39] W. Ye et al. All-in-One Slider for attribute manipulation in diffusion models. In CVPR, pages 18693–18702, 2026.

[40] A. Zarei et al. SliderEdit: Continuous image editing with fine-grained instruction control. In CVPR, pages 14430–14439, 2026.

[41] X. Zhai et al. Sigmoid loss for language image pre-training. In ICCV, pages 11975–11986, 2023.

[42] Z. Zhang et al. Confronting reward overoptimization for diffusion models: A perspective of inductive and primacy biases. In ICML, pages 60396–60413, 2024.

[43] Z. Zhu et al. CompSlider: Compositional slider for disentangled multiple-attribute image generation. In ICCV, pages 16682–16691, 2025.

# ScoreDial: Precise Control of Visual Attributes in Image Generation via Regressor Guidance from Limited Human Ratings

Supplementary Material

This supplement accompanies the main paper. Sections $\mathrm { A } { - } \mathrm { C }$ give the details that the main paper refers to: the autoencoder round-trip check (Section 3.3 of the main paper), the training-pool memorization check (Section 4.1), and the regressor-selection protocol (Section 4.2). Sections D and E show every test image behind Tables 1 and 2.

## A. Autoencoder round-trip check

Section 3.3 of the main paper states that, on the rated images, the frozen autoencoder path alone changes R by at most 1.84% of the control range. Each of the 847 rated images (118 tire, 100 interior, 479 food, 150 soap) is resized to $1 0 2 4 ^ { 2 }$ , encoded and decoded by the frozen SD3.5 autoencoder used during sampling, and scored by the deployed R before and after. Before running, we fixed the criterion that the round trip is negligible for an attribute if the mean $| \Delta R |$ is below 5% of the control range (0.8, the width of the requested values 0.1–0.9) and the Pearson correlation between the two scores is at least 0.99. The mean $| \Delta R |$ is 0.91%, 1.83%, 1.16%, and 0.42% of the range for tire, interior, food, and soap, with $r ~ \geq ~ 0 . 9 9 8 6$ , so every attribute meets the criterion. As a reference, pairing each original with a randomly permuted reconstruction gives the $| \Delta R |$ between unrelated images of the same attribute; the actual round-trip shift is at most 6% of that value. The check isolates the decoder path; attribute movement is evaluated separately in Section 4.3 of the main paper.

## B. Training-pool memorization check

Section 4.1 of the main paper states that an SSCD copydetection check finds no reported image unusually close to its generator-training pool. For each attribute we take the 50 reported ScoreDial images (the Full cells of Tables 1 and 2) and the generator-training pool of Section 4.1, embed both with the released SSCD descriptor sscd\_disc\_mixup (Pizzi et al., CVPR 2022), and measure cosine similarity. Each generated image is matched to its nearest training image. As a reference, each training image’s nearest neighbour within the pool, excluding itself, gives the similarity the pool already reaches internally. Table S1 lists the results: on every attribute, even the closest generated image lies inside the pool’s own nearest-neighbour distribution, at its 15th–74th percentile. The tire pool contains flipped copies of its tail images (Section 4.1), which raises its internal similarity.

Table S1. SSCD nearest-neighbour cosine similarity between the 50 reported images per attribute and the generator-training pool. $\mathrm { \ddot { P c t l . } } \ '$ is the percentile of the largest generated-image similarity within the pool’s internal nearest-neighbour distribution.
<table><tr><td rowspan="2"></td><td rowspan="2">Pool</td><td colspan="3">Generated NN Pool NN</td><td rowspan="2">Pctl.</td></tr><tr><td>size mean</td><td>max</td><td>mean</td></tr><tr><td>Attribute Tire</td><td>984</td><td>0.59</td><td>0.69</td><td>0.87</td><td>15.5</td></tr><tr><td>Interior</td><td>100</td><td>0.26</td><td>0.32</td><td>0.28</td><td>74.0</td></tr><tr><td>Food</td><td>479</td><td>0.50</td><td>0.66</td><td>0.66</td><td>48.2</td></tr><tr><td>Soap</td><td>800</td><td>0.44</td><td>0.57</td><td>0.56</td><td>57.9</td></tr></table>

## C. Regressor-selection protocol

Section 4.2 of the main paper reports repeated nested cross-validation of the full selection procedure. In each of five repeats, the labelled images are split into five outer folds. Within each outer-training split, a five-fold inner cross-validation scores all 28 configurations, the seven backbone–representation candidates of Section 3.1 combined with ridge $\alpha \in \{ 1 , 1 0 , 5 0 , 1 0 0 \}$ , and keeps the one with the highest inner $R ^ { 2 }$ . That configuration is refit on the outer-training split and predicts the held-out outer fold. Each repeat yields one $R ^ { 2 }$ over all outer predictions, and the main paper reports their $\mathrm { m e a n } \pm \mathrm { S D }$ . These scores estimate the selection procedure as a whole: the configurations chosen in the outer folds are recorded only to inspect their stability and do not choose the deployed regressor, which is selected separately as described in Section 4.2.

## D. All test images of the matched comparison

Table 1 of the main paper is computed from 50 test images per method and attribute (five requested values × ten test seeds), and Figure 1 shows one seed per attribute. Figures S1–S12 show all of them. Each block is one test seed, with the methods as rows and the requested values ${ c = 0 . 1 , 0 . 3 , . . . , 0 . 9 }$ as columns; within a block, all methods use the same seed. MPGD-flow is shown at its selected strength and $\mathbf { M P G D - f l o w ^ { + } }$ one strength rung above it.

![](images/37bb58521667e4d35cf33a06ef1e52a910f10b1616d5274b300bcf8b6c46fe44.jpg)  
Figure S1. Tire slip resistance, matched comparison, test seeds 0–3.

![](images/eb87f333d8f0b20c6fe4ef6b3ed46e54faadbb9db9402647f5742e03813acc63.jpg)  
Figure S2. Tire slip resistance, matched comparison, test seeds 4–7.

![](images/6132c65f819a8e39dc40bf51c9e8080494551e8e4eda25c2646eec2e42199b88.jpg)  
Figure S3. Tire slip resistance, matched comparison, test seeds 8–9.

![](images/b9344440b71d2ae99246cbbed3df495faf268efeb0fc305bac0e9465c612ae22.jpg)  
Figure S4. Interior visual complexity, matched comparison, test seeds 0–3.

![](images/c2d5a2d3cc868f06c7680698044e45585e7f842b9ae80f0d35c246d06c9cc5f8.jpg)  
Figure S5. Interior visual complexity, matched comparison, test seeds 4–7.

![](images/f6b07cc2059a86c50b23a21d407b9e497a3d964c9c653d50c495d4907e81a408.jpg)  
Figure S6. Interior visual complexity, matched comparison, test seeds 8–9.

Seed 1

![](images/a35411551f781182175d470a48fe40d75850df2d3d2ea45045a1d2beefabc59f.jpg)  
Figure S7. Food desire-to-eat, matched comparison, test seeds 0–3.

![](images/e9f4735f5ed5fae958499514c95dee6f22c48e5e766c5996162a0047e36f8d93.jpg)  
Figure S8. Food desire-to-eat, matched comparison, test seeds 4–7.

![](images/f583fe965b261ef98d8ac79fc8a0fabc0991562117db556f0ebf142783c781f3.jpg)  
Figure S9. Food desire-to-eat, matched comparison, test seeds 8–9.

![](images/b908f23e9a1a349750f60c09d529186fd37fa3287a798fdec1508ccb39e114a8.jpg)  
Figure S10. Soap translucency, matched comparison, test seeds 0–3.

![](images/b5487b160be1909791b2cd1c66998598113a30c2169b0309e07dcb8910b68b9a.jpg)  
Figure S11. Soap translucency, matched comparison, test seeds 4–7.

![](images/ec3f303b4ce419303fb79c30b71581e340a71a653e3e23dc9d16eb83b1211f19.jpg)  
Figure S12. Soap translucency, matched comparison, test seeds 8–9.

## E. All test images of the ablation

Table 2 of the main paper reports five configurations on the same $5 \times 1 0$ target–seed grid, and Figure 3 shows Full and G-only for tire at three requested values. Figures S13–S24 show every image of every configuration and attribute in the layout of Section D, with the five configurations as rows. Full is the ScoreDial run of Table 1, so its row repeats the ScoreDial row of the corresponding comparison figure.

![](images/ecac838b50416ca46fbf53e2e243d88e043e9f3d475415f119b31361d60c6d82.jpg)  
Figure S13. Tire slip resistance, ablation, test seeds 0–3.

![](images/f8a07d0cfe374a076006d1c2f68580d19a3b2879f398eab23f898c49ed23bb63.jpg)  
Figure S14. Tire slip resistance, ablation, test seeds 4–7.

![](images/a5dac1bda92d4121a38205ff7ef2394f95576d78ef6931989b8710a0f71a6c05.jpg)  
Figure S15. Tire slip resistance, ablation, test seeds 8–9.

![](images/83f1159e72e99e45d9e49e750dcfff8a91d5863520c1cfa422cc58d0ab50ec9f.jpg)  
Figure S16. Interior visual complexity, ablation, test seeds 0–3.

![](images/9c7dc94d004dbe5a51e682b540c33c72f57917ef55faa1e56f4f6d92e3d2fc8d.jpg)  
Figure S17. Interior visual complexity, ablation, test seeds 4–7.

![](images/d77c9fe0f9b969deed2dd53ba03dfb781b3f031c7b1ececc25d3036a6b555a36.jpg)  
Figure S18. Interior visual complexity, ablation, test seeds 8–9.

![](images/a1809215ed1db95f252ed679091bba8738d29bb1acedfdd2b086ae910d30037c.jpg)  
Figure S19. Food desire-to-eat, ablation, test seeds 0–3.

![](images/cac8140300262ee784c0850ae3e757ccbcfb812c20efb0bacaec4d96d5c994a5.jpg)  
Figure S20. Food desire-to-eat, ablation, test seeds 4–7.

![](images/1337fe5b9d08dc4f4d98fd969eac3f5b56e471be2f3e8d88aedef97f47aaebbe.jpg)  
Figure S21. Food desire-to-eat, ablation, test seeds 8–9.

![](images/53ed81d23711defed55e2f070525ad754374eb005bb004a10cada9b1deb1dbe3.jpg)  
Figure S22. Soap translucency, ablation, test seeds 0–3.

![](images/a43c52b88f846182bb11437c40e8b7705332b7cf3af51e9e56895d0ff27f27b1.jpg)  
Figure S23. Soap translucency, ablation, test seeds 4–7.

![](images/2c368fe59f4e8a4d083c13798dca66402145f63439448949c16d9c4c6c859657.jpg)  
Figure S24. Soap translucency, ablation, test seeds 8–9.