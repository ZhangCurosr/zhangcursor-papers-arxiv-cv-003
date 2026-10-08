# SpatialUQ: Post-Hoc Uncertainty Quantification from Spatial Consistency in Black-Box Vision Models

Md Kawsher Mahbub<sup>1</sup> Milon Biswas<sup>2</sup> Mirza Niaz Morshed<sup>3</sup> Wei Yu<sup>2</sup> <sup>1</sup>NPI University of Bangladesh, <sup>2</sup>Towson University, <sup>3</sup>University of Asia Pacific kawsher@npiub.edu.bd, mbiswas1@students.towson.edu, 25214015@uap-bd.edu, wyu@towson.edu

## Abstract

Clinical vision models are often deployed as frozen black boxes with no access to internals, retraining, or ground truth at inference time. We introduce SpatialUQ, a post-hoc uncertainty method using only output probabilities. It measures the Jensen-Shannon divergence between the global prediction and the mean of five fixed spatial crops in six deterministic forward passes. The premise is simple, trustworthy predictions are spatially consistent. On NIH ChestX-ray14 (DenseNet-121, N=25,596), our Multicrop Uncertainty Score (MUS) reaches 0.784 failure-detection AUC versus 0.664 for MC-Dropout (p<10<sup>−6</sup>) at one-fifth the compute, with native calibration (SCE=0.049 vs. 0.127 for $\ell _ { 1 } )$ , the best-calibrated among methods above 0.78 AUC. A supervised fusion of MUS with entropy, confidence, and $\ell _ { 1 }$ reaches 0.832, outperforming a five-member ensemble (0.813). MUS scales with model quality, reaching 0.899 with BiomedCLIP $( \rho = 0 . 8 4 6 )$ , while this relationship remains meaningful in-distribution $( \rho = 0 . 5 2 3 )$ but breaks down under severe distribution shift (VinBigData, $\rho = 0 . 0 2 7 )$ MUS is well-suited to difuse findings but is less dependable for small focal lesions such as nodules. Code and experimental materials are publicly available at https://huggingface. co/datasets/kawsher11/SpatialUQ.

## 1 INTRODUCTION

Deep neural networks can fail silently on out-ofdistribution inputs (Amodei et al., 2016; Hendrycks and Dietterich, 2019), posing patient-safety risks in medical imaging where confident errors are consequential (Leibig et al., 2017; Kendall and Gal, 2017). Reliable deployment therefore requires inference-time failure detection without ground truth. However, frozen black-box classifiers exclude many existing UQ approaches. Bayesian methods such as MC-Dropout (Gal and Ghahramani, 2016) require training-time modifications, ensembles (Lakshminarayanan et al., 2017) and TTA (Krizhevsky et al., 2012) add substantial compute, while post-hoc methods such as DDU (Mukhoti et al., 2023), Mahalanobis (Lee et al., 2018), energy (Liu et al., 2020), ODIN (Liang et al., 2017), and conformal prediction (Angelopoulos and Bates, 2023) require model internals, feature distributions, or calibration data. Recent consistency-based (Khan and Fu, 2024) and augmentation-sensitivity (Shu et al., 2026) methods operate in related settings but rely on linguistic or photometric perturbations, leaving geometric evidence largely unexplored.

We propose SpatialUQ grounded on a single premise: a reliable model produces spatially consistent predictions across an image and its sub-regions. Welllocalized features create strong agreement between global and crop-level predictions. On the other hand, distributional mismatch, low image quality, or atypical presentations make them diverge. We measure this divergence as the Jensen-Shannon Divergence (JSD) between the global prediction and the mean of five fixed spatial crops, in six deterministic forward passes. This geometric signal needs no internals, sampling, or reference data, and difers fundamentally from TTA (Krizhevsky et al., 2012; Wang et al., 2019) (Lemma 1; Section 5.4.1). A supervised logistic fusion of MUS with entropy, confidence, and $\ell _ { 1 }$ improves failure detection further, except under severe shift where MC-Dropout remains preferable.

Our key contributions are summarized as follows:

• SpatialUQ: A post-hoc uncertainty framework for frozen black-box classifiers using only output probabilities, with no internals, gradients, training data, or retraining. MUS is natively calibrated and extends to frozen foundation models, reaching 0.899 AUC with BiomedCLIP $( \rho = 0 . 8 4 6 )$ . Spatial perturbations provide the key signal, with photometric JSD reducing AUC from 0.784 to 0.755.

• Formalization and self-diagnosing deployment gate: A formal bound connects MUS to mean absolute spatial probability shift, while a small labeled calibration sample provides a deployment diagnostic. The correlation remains meaningful in-distribution $( \rho = 0 . 5 2 3 )$ but collapses under severe shift (VinBigData, $\rho = 0 . 0 2 7 )$ .

• State-of-the-art failure detection at one-fifth the cost: MUS achieves 0.784 AUC on NIH ChestX-ray14, outperforming MC-Dropout (0.664, $p < 1 0 ^ { - 6 } )$ with 6 vs. 30 forward passes. Fusion reaches 0.832 AUC, exceeding a five-member ensemble (0.813) without retraining.

• Cross-domain transferability: MUS achieves 0.708 AUC on CheXpert, a 0.109 gain over MC-Dropout; fusion reaches 0.917–0.937 on ImageNet and 0.885–0.913 on MS COCO.

Spatial inconsistency is complementary rather than universal, it excels when models are overconfident on multi-label tasks but degrades when failures are spatially uniform.

## 2 RELATED WORK

Bayesian and stochastic uncertainty estimation. MC-Dropout approximates Bayesian inference via T stochastic forward passes but requires dropout layers to be present at training time, making it inapplicable to frozen checkpoints (Gal and Ghahramani, 2016). Bayesian weight approximations ofer principled posteriors but scale poorly (Graves, 2011; Blundell et al., 2015). Deep Ensembles achieve strong calibration at the cost of multiplying compute (Lakshminarayanan et al., 2017). A recent benchmark of 13 such techniques on multi-label chest X-ray data confirms no single method is universally optimal (Baur et al., 2025). SpatialUQ imposes no training-time assumptions

Test-time augmentation and spatial perturbation. TTA treats prediction variance across augmentations as an uncertainty proxy (Krizhevsky et al., 2012; Wang et al., 2019; Ayhan and Berens, 2018), though strategy matters more than diversity (Shanmugam et al., 2021). Geometric and photometric augmentations confound spatial ambiguity with color/contrast sensitivity. SpatialUQ instead measures whether predictions are stable across spatial sub-regions a geometric property of evidence localization formalized in Lemma 1 and compared against photometric TTA-JSD in Section 5.4.1.

Post-hoc feature- and logit-based methods. Mahalanobis distance (Lee et al., 2018), DDU (Mukhoti et al., 2021), energy scoring, and ODIN (Liu et al.,

2020; Liang et al., 2017) operate on training-set feature distributions or logits. Conformal prediction provides distribution-free coverage guaranties but constructs prediction sets rather than ranking inputs by failure probability (Angelopoulos and Bates, 2023). SpatialUQ produces a scalar triage score. Combining MUS-based ranking with conformal calibration is a natural future direction.

Uncertainty in medical imaging. MC-Dropout has been applied to retinopathy screening (Leibig et al., 2017) and brain segmentation (Roy et al., 2019). The aleatoric/epistemic distinction was formalized by Kendall and Gal (2017) and clinical motivation is surveyed by Begoli et al. (2019). Temperature scaling addresses overconfidence (Guo et al., 2017) and selective prediction provides the risk-coverage framework for triage (Geifman and El-Yaniv, 2017). We follow Nado et al. (2021), reporting ECE, AURC, E-AURC, AUPR, and FPR at fixed TPR.

Failure and OOD detection. Failure detection without ground truth has been studied in object detection (Miller et al., 2019), with likelihood-ratio scores proposed as OOD detectors (Ren et al., 2019). Our MS COCO evaluation extends SpatialUQ to detection via a multi-label Bernoulli formulation. Prior calibration work on ViT (Fort et al., 2019; Minderer et al., 2021) and globally coherent self-attention (Liu et al., 2021) motivates our investigation of masking versus up-sampling for crop construction.

Divergence measures and output-space uncertainty. Predictive entropy (Shannon, 1948) and mutual information (Houlsby et al., 2011) are canonical output-space baselines. JSD’s symmetry, boundedness in [0, log 2], and robustness to zero probability mass make it well-suited to multi-label sigmoid outputs (Goodfellow et al., 2014; Hinton et al., 2015). To our knowledge, SpatialUQ is the first to apply JSD between global and local spatial predictions as a failure detection proxy, introducing a geometrically grounded interpretation requiring no model internals.

## 3 OUR APPROACH

SpatialUQ operationalizes the spatial consistency hypothesis through a deterministic five-crop decomposition. We formalize the score for single- and multi-label settings, describe the optional linear fusion, and bound the score against per-class probability shifts. Figure 1 illustrates the pipeline and qualitative examples.

## 3.1 Spatial Decomposition

Let $f : \mathcal { X }  \mathcal { P }$ be a frozen classifier mapping $x \in \mathcal { X }$ to either a probability simplex $\Delta ^ { C - 1 }$ (single-label) or $[ 0 , 1 ] ^ { C }$ under independent sigmoid activations (multilabel), accessible only via output probabilities. We seek a deterministic scalar $s ( x ) \geq 0$ , computable in a small constant number of forward passes, that is positively correlated with prediction error (Brier score) at inference time.

![](images/94c9bb8e644747eb8b45e91fb3f6a03aedbb46ea005d06b92d9918a3cd15dbae.jpg)  
(a) SpatialUQ Pipeline Overview.

![](images/9d7ffef29b9d32757a7c7ea68f88454e8dd8119c664dcf91d55b88205e16f145.jpg)  
(b) Qualitative Examples: Low vs. High MUS.  
Figure 1: (a) The five-step inference pipeline computing JSD between global and aggregated local predictions. (b) High spatial inconsistency (MUS) successfully flags prediction errors.

A reliable model gives spatially consistent predictions, inconsistency signals that confidence is not anchored to stable image content. Each 224×224 input is decomposed into five fixed spatial crops, four non-overlapping quadrants and one overlapping center crop (in $[ y _ { 1 } , x _ { 1 } , y _ { 2 } , x _ { 2 } ]$ format): $\begin{array} { r l } { \mathcal { C } } & { { } = } \end{array}$ {(0, 0, 112, 112), (0, 112, 112, 224), (112, 0, 224, 112), (112, 112, 224, 224), (56, 56, 168, 168)}, fixed across all inputs and architectures. Each crop is bi-linearly upsampled to 224×224 before passing through $f .$ The aggregate local prediction is:

$$
p _ { \mathrm { l o c a l } } ( x ) = { \frac { 1 } { 5 } } \sum _ { k = 1 } ^ { 5 } f ( \mathrm { u p s a m p l e } ( x [ { \mathcal { C } } _ { k } ] ) )\tag{1}
$$

i Together with $p _ { \mathrm { g l o b a l } } ( x ) = f ( x )$ , this requires exactly six forward passes through an unmodified model.

## 3.2 Uncertainty Score via Jensen-Shannon Divergence

We quantify spatial inconsistency via JSD between $p _ { \mathrm { g l o b a l } }$ and $p _ { \mathrm { l o c a l } }$ (bounded in [0, log 2], symmetric, welldefined under zero probability mass).

Single-label tasks (categorical JSD). For softmax outputs p<sub>global</sub>, p<sub>local</sub> $\in \Delta ^ { C - 1 }$ , with mixture $\begin{array} { r } { m = \frac { 1 } { 2 } ( p _ { \mathrm { g l o b a l } } + \bar { p } _ { \mathrm { l o c a l } } ) . } \end{array}$

$$
\begin{array} { l } { \displaystyle { s _ { \mathrm { c a t } } ( x ) = \mathrm { J S D } ( p _ { \mathrm { g l o b a l } } \| p _ { \mathrm { l o c a l } } ) } } \\ { \displaystyle { \phantom { s p a c e } = \frac { 1 } { 2 } \mathrm { K L } ( p _ { \mathrm { g l o b a l } } \| m ) + \frac { 1 } { 2 } \mathrm { K L } ( p _ { \mathrm { l o c a l } } \| m ) } } \end{array}\tag{2}
$$

Applied to ImageNet-1k and MS COCO (per-class detection confidences as an 80-dimensional vector).

Multi-label tasks (Bernoulli JSD). Each class c induces an independent Bernoulli distribution with mixture $m _ { c } = \frac { 1 } { 2 } ( p _ { \mathrm { g l o b a l } , c } + p _ { \mathrm { l o c a l } , c } )$ :

$$
\begin{array} { l } { \displaystyle \mathrm { J S D } _ { c } ( p , q ) = \frac { 1 } { 2 } \Big [ p _ { c } \log \frac { p _ { c } } { m _ { c } } + ( 1 - p _ { c } ) \log \frac { 1 - p _ { c } } { 1 - m _ { c } } } \\ { \displaystyle + q _ { c } \log \frac { q _ { c } } { m _ { c } } + ( 1 - q _ { c } ) \log \frac { 1 - q _ { c } } { 1 - m _ { c } } \Big ] } \end{array}\tag{3}
$$

$$
s _ { \mathrm { b e r n } } ( x ) = \frac { 1 } { C } \sum _ { c = 1 } ^ { C } \mathrm { J S D } _ { c } ( p _ { \mathrm { g l o b a l } , c } , p _ { \mathrm { l o c a l } , c } )\tag{4}
$$

Applied to NIH ChestX-ray14 (C=14), CheXpert (C=10), and VinBigData (C=10). Algorithm 1 summarises the procedure: one forward pass for $p _ { \mathrm { g l o b a l } } .$ , five for the upsampled crops $\left( p _ { 1 } , \ldots , p _ { 5 } \right)$ , average to p<sub>local</sub>, and return $s _ { \mathrm { c a t } }$ or $s _ { \mathrm { b e r n } } .$ . The pipeline is deterministic, stateless, and adds no overhead beyond the six forward passes.

Algorithm 1: SpatialUQ   
Input: Frozen model $f ,$ image $\boldsymbol { x } \in \mathbb { R } ^ { 2 2 4 \times 2 2 4 \times 3 } .$   
task ∈ {single-label, multi-label}   
Output: Scalar uncertainty score $s ( x ) \geq 0$   
$p _ { \mathrm { g l o b a l } }  f ( x )$ $/ /$ forward pass 1   
for k = 1 to 5 do   
p<sub>k</sub> ← f(upsample(x $[ { \mathcal { C } } _ { k } ] ,$ , 224×224))   
// forward passes 2-6   
end   
p<sub>local</sub> $ \frac { 1 } { 5 } \sum _ { k = 1 } ^ { 5 } p _ { k }$   
s ← s<sub>cat</sub>(x) if single-label else $s  s _ { \mathrm { b e r n } } ( x )$   
return s

## 3.3 Linear Fusion (Supervised Extension)

This supervised extension fits logistic weights on a small labeled hold-out; the unsupervised MUS itself needs

no labels. It combines complementary output signals via logistic regression:

$$
\begin{array} { r l } & { s _ { \mathrm { f u s e d } } ( x ) = \sigma \big ( \alpha _ { 0 } + \alpha _ { 1 } s ( x ) + \alpha _ { 2 } H ( x ) } \\ & { \qquad + \alpha _ { 3 } \left( 1 - \underset { c } { \operatorname* { m a x } } p _ { c } \right) + \alpha _ { 4 } d _ { \ell _ { 1 } } ( x ) \big ) } \end{array}\tag{5}
$$

Here $\begin{array} { r } { d _ { \ell _ { 1 } } ( x ) \ = \ \frac { 1 } { C } \sum _ { c } \left| p _ { \mathrm { g l o b a l } , c } - p _ { \mathrm { l o c a l } , c } \right| } \end{array}$ and $\sigma ( \cdot )$ is the sigmoid. Coeficients $\alpha _ { i }$ come from 5-fold crossvalidation on a small labeled hold-out: the validation set for NIH, or a random 10% of the test set for zeroshot settings (remaining 90% for evaluation only), preserving $f \mathrm { ^ { \prime } s }$ zero-retraining property. Ablation is in Table 2.

## 3.4 Information-Theoretic Bound

Lemma 1 (Spatial Disagreement Bound). For each class c, let $p _ { c } : = p _ { \mathrm { g l o b a l } , c }$ $q _ { c } : = p _ { \mathrm { l o c a l } , c } \in [ 0 , 1 ]$ . Then:

$$
\frac { 1 } { C } \sum _ { c = 1 } ^ { C } \bigl | p _ { \mathrm { g l o b a l } , c } - p _ { \mathrm { l o c a l } , c } \bigr | \ \leq \ \sqrt { 2 \cdot s _ { \mathrm { b e r n } } ( x ) }\tag{6}
$$

Here, $s _ { \mathrm { b e r n } } ( x )$ is in nats, matching the implementation. The per-class bound is tight in the limit $p _ { c } , q _ { c } \to { \frac { 1 } { 2 } } .$ where the ratio of the two sides tends to 1.

Proof sketch. Applying Pinsker’s inequality to $\mathrm { K L } ( p _ { c } \| m _ { c } )$ and $\mathrm { K L } ( q _ { c } \| m _ { c } )$ separately, where $m _ { c } = \textstyle { \frac { 1 } { 2 } } ( p _ { c } + q _ { c } )$ , yields $( p _ { c } - q _ { c } ) ^ { 2 } \leq 2 \mathrm { J S D } _ { c }$ in nats. Taking square roots and averaging over $C$ classes via Jensen’s inequality (concavity of $\sqrt { \cdot ) }$ gives the class-averaged bound. Full derivation and tightness verification are in Appendix A. □

Lemma 1 shows a large SpatialUQ score necessarily implies a large mean absolute shift between global and local class probabilities. This concerns probability shift only; that MUS predicts failures is empirical: the Spearman correlation between $s _ { \mathrm { b e r n } }$ and per-image Brier score is positive and highly significant $( p < 0 . 0 0 0 1 )$ across all medical benchmarks, which licenses MUS as a failure signal. The bound is conservative under heterogeneous per-class JSD via Jensen’s inequality.

## 4 EXPERIMENTAL SETUP

## 4.1 Datasets and Tasks

We evaluate SpatialUQ across twelve configurations spanning medical and natural image classification and object detection. NIH ChestX-ray14 (Wang et al., 2017) (112,120 frontal chest X-rays, 14 multi-label diseases; patient-wise split 73,891/12,633/25,596 train/val/test; class-balanced BCE for 5.3×–641.5× imbalance) trains three architectures independently: DenseNet-121 (Huang et al., 2017), EficientNet-B4 (Tan and Le, 2019), and ViT-B/16 (Dosovitskiy et al., 2020). CLIP ViT-B/32 and BiomedCLIP are evaluated on the same NIH test set via linear-probe transfer (frozen backbones). CheXpert (Irvin et al., 2019) and VinBigData (Nguyen et al., 2022) serve as zero-shot transfer benchmarks for the NIH-trained DenseNet-121 (10 NIH-shared classes each; Appendix L; U-Zeros for uncertain CheXpert labels). ImageNet-1k evaluates three pretrained models (EficientNet-B4, ViT-B/16, ConvNeXt-Tiny (Liu et al., 2022)) zero-shot on 50,000 validation images with categorical JSD; failure is top-1 misclassification. MS COCO 2014 (Lin et al., 2014) evaluates Faster R-CNN (Ren et al., 2015) and RetinaNet (Lin et al., 2017) (ResNet-50-FPN) on a 5,000-image subset (seed 42), with per-class confidence as the maximum detection score (threshold 0.1; 0.05/0.30 confirm stability) and Bernoulli JSD over 80-dimensional vectors.

## 4.2 Failure Definition

A prediction is labeled a failure if its per-image Brier score exceeds the 75th percentile of the test-set Brier distribution, yielding a consistent 25% failure rate across all datasets and architectures for direct crossdomain comparison without dataset-specific tuning (robustness to the 50th, 75th, and 90th percentile thresholds is confirmed in Appendix E.3. Critically, the Brier score is computed against ground-truth labels and is epistemically independent of MUS, which operates solely on output probabilities.

## 4.3 Training Protocol

NIH architectures are fine-tuned from ImageNetpretrained weights in two stages: (i) 5 epochs with frozen backbone (lr $3 \times 1 0 ^ { - 4 }$ , AdamW, weight decay $1 0 ^ { - 4 } )$ ; (ii) up to 25 epochs (patience 7) unfrozen (backbone lr $5 \times 1 0 ^ { - 6 }$ , head lr $5 \times 1 0 ^ { - 5 }$ , cosine annealing to $1 0 ^ { - 7 } )$ . Loss is BCE with label smoothing $( \varepsilon = 0 . 1 )$ and class-balanced pos\_weight, augmentation is random rotation $( \pm 1 0 ^ { \circ } )$ and crop to 224 × 224 (no horizontal flip, preserving laterality), batch size 32. ImageNet and COCO models are evaluated zero-shot full details in Appendix J.

## 4.4 Baselines

We compare against 17 uncertainty methods from the same frozen model without retraining. Output-space: predictive entropy (Shannon, 1948), inverted confidence (raw and temperature-scaled), ℓ<sub>1</sub> distance, and maximum inter-crop disagreement. Crop-based: patch variance, pairwise disagreement, and pairwise JSD. For reproducibility: patch variance is the variance of perclass probabilities across the five crops, averaged over classes, pairwise disagreement is the mean absolute diference between all pairs of crop predictions, pairwise JSD is the mean JSD between all pairs of crop predictions. Augmentation-based: TTA variance, TTA-JSD (10 augmentations), and MC-Dropout (Gal and Ghahramani, 2016) (30 passes). Feature/logitbased: DDU (Mukhoti et al., 2021), Mahalanobis distance (Lee et al., 2018), and ODIN (Liang et al., 2017). Ensemble: mean prediction entropy, ensemble variance, and temperature-scaled entropy (Lakshminarayanan et al., 2017) from five DenseNet-121 models. Hyperparameters follow respective publications $( \mathrm { A p \mathrm { - } }$ pendix J).

## 4.5 Evaluation Protocol

The primary metric is AUC for binary failure detection, we additionally report AUPR, AURC, E-AURC, and FPR at 80% and 95% TPR. We adopt the AURC formulation of Nado et al. (2021) over AUGRC (Zhou et al., 2024), as the fixed 25% failure rate renders the two equivalent. Calibration uses Score Calibration Error (SCE; ECE with 15 bins on min-max normalized scores). All metrics include 95% bootstrap CIs (1,000 resamples), pairwise AUC significance uses DeLong tests and two-tailed bootstrap p-values (10,000 resamples). All experiments use seed 42, code and data splits are included in the supplementary material and will be released publicly upon acceptance.

## 5 RESULTS

## 5.1 NIH ChestX-ray14: In-Distribution Failure Detection

Table 1 reports failure detection on NIH ChestX-ray14 (N=25,596) using DenseNet-121. MUS reaches an AUC of 0.784 (95% CI [0.778, 0.789]) with six deterministic forward passes, outperforming MC-Dropout (0.664, $\Delta { = } { + } 0 . 1 1 9 , \ p { < } 1 0 ^ { - 6 } ,$ ; 30 stochastic passes) while reducing wall-clock time by 5× (≈8 vs. ≈39 min, Appendix J). Although $\ell _ { 1 }$ achieves a slightly higher AUC (0.790 vs. 0.784, $\Delta { = } 0 . 0 0 6 , p { < } 1 0 ^ { - 6 } )$ , its calibration is substantially worse (SCE 0.127 vs. 0.049). Naive crop variance performs much worse (0.610 AUC), showing that the way crop predictions are aggregated is important. Confidence and entropy are close to chance (0.401 and 0.177), reflecting systematic BCE overconfidence from label smoothing and class-balanced loss; temperature scaling does not recover the signal (0.400). This advantage is specific to the overconfidence regime encountered here. On well-calibrated ImageNet classifiers, confidence alone reaches 0.913–0.923 AUC, compared with 0.641–0.717 for MUS, although MUS still provides complementary information in fusion (Section 5.6). Combining MUS, entropy, confidence, and $\ell _ { 1 }$ through 5-fold cross-validation on the validation set increases AUC to 0.832, outperforming a five-member deep ensemble $( 0 . 8 1 3 , \Delta { = } \mathrm { + } 0 . 0 1 9 , p { < } 1 0 ^ { - 6 } )$ at a fraction of the training cost. MUS also tracks model error monotonically $( \rho { = } 0 . 5 2 3 , p { < } 1 0 ^ { - 6 } )$ , and the same behavior extends to EficientNet-B4 and ViT-B/16 (Appendix C). Finally, spatial disagreement heatmaps show that high-MUS failures concentrate crop-level JSD in ambiguous regions (Appendix K).

![](images/17096e69fce967abbe526d0fa69ce62dcbb7914dfa41df9809cda062eac3bd12.jpg)  
Figure 2: Failure detection ROC curve for the NIH test set, comparing SpatialUQ (MUS) against samplingbased and feature-based baselines.

Figure 2 shows that MUS achieves 0.784 AUC with six deterministic passes, outperforming MC-Dropout (0.664), crop-variance TTA (0.610), and all other TTA variants $\left( \leq 0 . 5 8 4 \right)$ without ensembles, stochastic inference, or auxiliary supervision. Fusion further improves performance to 0.832 AUC by combining MUS with complementary signals.

## 5.2 Extension to Frozen Foundation Models

We apply the NIH protocol to frozen CLIP ViT-B/32 (Radford et al., 2021) and BiomedCLIP (Zhang et al., 2023) via linear-probe transfer. Confidence<sup>−1</sup> collapses on uncalibrated medical outputs (0.095/0.206 AUC), while MUS remains strong at 0.825 (ρ=0.631) and 0.899 $( \rho { = } 0 . 8 4 6 , p { < } 1 0 ^ { - 6 } )$ , respectively. Although $\ell _ { 1 }$ has slightly higher AUC, MUS is better calibrated (SCE 0.133 vs. 0.232), and fusion reaches 0.883/0.925. SpatialUQ therefore transfers across general and domain-specific foundation models without architectural access (Appendix C.2).

## 5.3 Calibration as a Deployment Signal

Calibration determines whether a threshold is clinically meaningful, AUC measures only ranking. Figure 3 shows MUS natively calibrated (SCE=0.049) versus raw $\ell _ { 1 } \ ( 0 . 1 2 7 )$ . A labeled hold-out lets isotonic regression calibrate $\ell _ { 1 } ~ ( 0 . 0 0 6 / 0 . 7 8 9 )$ and fusion (0.004/0.830), but such labels are unavailable for frozen black-box deployment, where MUS remains the only natively calibrated discriminative option for deployment.

Table 1: Failure detection on NIH ChestX-ray14 $( N { = } 2 5 , 5 9 6$ , failure: Brier > 75th percentile). † requires >6 passes. $^ \ddag$ requires training features. Full results in Table 9.
<table><tr><td>Method</td><td>AUC</td><td>95% CI</td><td>SCE</td></tr><tr><td>Fusion (CV)</td><td>0.832</td><td>[0.827, 0.837]</td><td>0.246</td></tr><tr><td>Deep Ensemble†</td><td>0.813</td><td>[0.808, 0.819]</td><td>0.138</td></tr><tr><td> $\ell _ { 1 }$  Distance</td><td>0.790</td><td>[0.785, 0.796]</td><td>0.127</td></tr><tr><td>MUS (Unsupervised)</td><td>0.784</td><td>[0.778, 0.789]</td><td>0.049</td></tr><tr><td>Deep Ens + T-scalingt</td><td>0.774</td><td>[0.768, 0.780]</td><td>0.090</td></tr><tr><td>ODIN</td><td>0.759</td><td>[0.752, 0.765]</td><td>0.484</td></tr><tr><td>TTA-JSD (Photometric)</td><td>0.755</td><td>[0.750, 0.761]</td><td>0.043</td></tr><tr><td>DDU Score</td><td>0.754</td><td>[0.748, 0.760]</td><td>0.139</td></tr><tr><td>Mahalanobis</td><td>0.754</td><td>[0.748, 0.760]</td><td>0.283</td></tr><tr><td>Max Disagreement</td><td>0.744</td><td>[0.738, 0.750]</td><td>0.186</td></tr><tr><td>MC-Dropout†</td><td>0.664</td><td>[0.658, 0.671]</td><td>0.121</td></tr><tr><td>Ensemble Variance</td><td>0.658</td><td>[0.651, 0.665]</td><td>0.042</td></tr><tr><td>TTA (5 crops, variance)</td><td>0.610</td><td>[0.602, 0.618]</td><td>0.029</td></tr><tr><td>TTA Variance (N=10)†</td><td>0.584</td><td>[0.576, 0.591]</td><td>0.089</td></tr><tr><td>T-Scaled  $\mathrm { { C o n f } ^ { - 1 } }$ </td><td>0.400</td><td>[0.391, 0.408]</td><td>0.002</td></tr><tr><td>Confidence−¹</td><td>0.401</td><td>[0.392, 0.409]</td><td>0.422</td></tr></table>

![](images/ba5ffd0f9bfc891d42eb63fabd6ce351a6f1b6cdaf5b879a1477723ff8c514c2.jpg)  
Figure 3: Reliability diagram comparing MUS against baseline uncertainty methods on NIH ChestX-ray14 (DenseNet-121).

## 5.4 Ablation: Fusion Components and Spatial Design

Table 2 shows entropy is the most critical fusion component (removal: −0.044 AUC). MUS and $\ell _ { 1 }$ add complementary gains $( - 0 . 0 0 5 , - 0 . 0 0 7 )$ , confidence negligibly (−0.001). Yet entropy alone reaches only 0.177 AUC under BCE overconfidence while MUS reaches 0.784 alone, so fusion gains come from orthogonal failure modes. Five fixed crops are optimal (Appendix E), nine crops gain only +0.007 AUC at 67% additional cost, three crops lose 0.007, and 1,000 random placements reach 0.739 (Appendix E.1). These results confirm the fixed geometric arrangement provides superior spatial

coverage (Appendix E.2).

Table 2: Leave-one-out ablation of the four-feature fusion (NIH DenseNet-121). Full fusion $\mathrm { A U C } = 0 . 8 3 2$ ∆: AUC drop when each feature is removed.
<table><tr><td>Removed Feature</td><td>AUC</td><td>Δ</td></tr><tr><td>Entropy</td><td>0.788</td><td>-0.044</td></tr><tr><td> $\ell _ { 1 }$  Distance</td><td>0.825</td><td>-0.007</td></tr><tr><td>MUS</td><td>0.827</td><td>-0.005</td></tr><tr><td>Confidence</td><td>0.831</td><td>-0.001</td></tr></table>

5.4.1 Spatial vs. Photometric Perturbation The identical Bernoulli JSD on five photometric augmentations (brightness ±0.15, contrast $\times 1 . 2 / { \times } 0 . 8 .$ noise $\sigma { = } 0 . 0 5 )$ yields 0.755 AUC $\left( \rho { = } 0 . 4 9 3 \right)$ versus MUS’s 0.784 $\left( \rho { = } 0 . 5 2 3 \right)$ at the same six passes, with marginally better calibration (SCE 0.043 vs. 0.049). The AUC advantage confirms the gain comes from the perturbation’s geometric structure, not the divergence measure.

## 5.5 Zero-Shot Generalization to External Medical Datasets

The NIH-trained DenseNet-121 transfers without retraining to CheXpert (N=44,399) and VinBigData $( N { = } 1 5 , 0 0 0 ) ;$ ; results are summarized in Table 3. On CheXpert, MUS achieves 0.708 AUC versus 0.599 for MC-Dropout $( \Delta = + 0 . 1 0 9 , p { < } 1 0 ^ { - 6 } )$ , with $\rho$ declining from 0.523 to 0.298 but retaining useful discrimination. At 20% referral, selective prediction improves system AUC from 0.686 to 0.748 +19.7% relative gap reduction. Under severe shift on VinBigData, however, MC-Dropout outperforms MUS (0.764 vs. 0.614, $\Delta = - ~ 0 . 1 5 0 , ~ p { < } 1 0 ^ { - 6 } )$ , while $\rho$ collapses to 0.027 as failures become spatially uniform and MUS loses its inconsistency signal.

Table 3: Zero-shot failure detection (NIH-trained DenseNet-121, no retraining). Spearman $\rho$ = correlation between MUS and per-image Brier score. Full per-method tables in Appendix F.
<table><tr><td>Dataset</td><td>MUS AUC</td><td>Fusion AUC</td><td>MC- Dropout AUĆ</td><td>Spearman  $\rho$ </td></tr><tr><td>NIH dist.)</td><td>(in- 0.784</td><td>0.832</td><td>0.664</td><td>0.523</td></tr><tr><td>CheXpert (zero-shot)</td><td>0.708</td><td>0.733</td><td>0.599</td><td>0.298</td></tr><tr><td>VinBigData (zero-shot)</td><td>0.614</td><td>0.750</td><td>0.764</td><td>0.027</td></tr></table>

## 5.6 Natural Image Classification and Object Detection

Table 4 evaluates ImageNet-1k (three architectures, 50,000 images) and MS COCO 2014 (two detectors,

Table 4: Failure detection AUC on ImageNet-1k and MS COCO 2014. Full secondary metrics in Appendix G.
<table><tr><td>Dataset</td><td>Model</td><td>Top-1 Acc.</td><td>MUS</td><td> $\mathbf { C o n f i d e n c e } ^ { - 1 }$ </td><td>Entropy</td><td>Fusion</td><td>Spearman ρ</td></tr><tr><td>ImageNet</td><td>EfficientNet-B4</td><td>79.27%</td><td>0.717</td><td>0.913</td><td>0.896</td><td>0.917</td><td>0.392</td></tr><tr><td>ImageNet</td><td>ViT-B/16</td><td>81.07%</td><td>0.710</td><td>0.923</td><td>0.884</td><td>0.937</td><td>0.341</td></tr><tr><td>ImageNet</td><td>ConvNeXt-Tiny</td><td>82.13%</td><td>0.641</td><td>0.913</td><td>0.819</td><td>0.936</td><td>0.139</td></tr><tr><td>COCO</td><td>Faster R-CNN</td><td></td><td>0.800</td><td>0.602</td><td>0.884</td><td>0.885</td><td>0.613</td></tr><tr><td>COCO</td><td>RetinaNet</td><td></td><td>0.747</td><td>0.702</td><td>0.912</td><td>0.913</td><td>0.510</td></tr></table>

5,000 images). On well-calibrated single-label classifiers, confidence already provides a strong correctness signal, making MUS complementary rather than a replacement (Section 6). On ImageNet, MUS alone trails Confidence<sup>−1</sup> (0.641–0.717 vs. 0.913–0.923 AUC), but fusion reaches 0.917–0.937, with MUS consistently contributing in leave-one-out ablations (Appendix G). This suggests that spatial consistency provides information beyond confidence even when confidence is already highly predictive. On COCO, MUS is stronger as a standalone signal (0.800/0.747 AUC; ρ=0.613/0.510), approaching the NIH in-distribution correlation and benefiting from the richer spatial structure of multi-label detection. Entropy remains the strongest standalone signal (0.884/0.912), while fusion reaches 0.885/0.913.

## 5.7 Clinical Triage Simulation

Radiologist review simulation. We simulate triage on the NIH test set (DenseNet-121). Referring the 9.2% highest-MUS images to human review recovers 20% of all Brier-defined failures at 54.5% referral precision. The threshold is tunable: precision is 58.7% at 10% recall (4.3% referred), 54.5% at 20%, 51.4% at 30% (Appendix I). Rejecting the 20% most uncertain images raises mean test AUC 0.808→0.814 (∆=+0.006).

![](images/167353f3d9f41a5bcf98b8685a0acce12f42044c4a45fd94393101144e7d1f04.jpg)  
Figure 4: Mean per-image Brier score of the retained set (lower is better) versus rejection rate (NIH DenseNet-121, N=25,596), rejecting images in descending uncertainty order.

Figure 4 compares rejection quality across all methods against a random baseline. MUS monotonically lowers retained Brier (0.521→0.507 at 30%) and beats MC-Dropout throughout. Fusion (CV) leads (0.504), with $\ell _ { 1 }$ and ensemble (0.506) marginally ahead of MUS. MUS remains the best-calibrated unsupervised score above 0.78 AUC (SCE 0.049 vs. ℓ<sub>1</sub>’s 0.127). Entropy and confidence degrade the retained set, as expected from their near-chance AUC, since Brier defines the failure label, this restates the AUC ranking.

Per-class analysis. MUS discrimination varies strongly across pathologies: it performs best on difuse, globally visible conditions such as Pneumonia (JSD AUC 0.979) and Edema (0.971), but poorly on small focal lesions such as Nodule (0.368), Fibrosis (0.553), and Mass (0.558), reflecting a limitation of the five-crop design. However a 16-crop extension raises Nodule AUC to 0.589 Appendix H). Prevalence-weighting across the 14 classes yields an overall AUC of 0.83, but masks this weakness because the three hardest classes account for only 14% of positive findings. A focal-lesion-enriched benchmark would provide a more stringent test of MUS.

![](images/9a8f83b51005acce47419939daca47606db40ac3d5e9b6b4324a1f7919977cc3.jpg)  
Figure 5: Top-to-bottom decile ratio of mean perimage Brier score by lesion type (NIH DenseNet-121, N=25,596). Error bars: 95% bootstrap CIs; n = 2,560 images per decile.

Figure 5 shows the top-to-bottom decile Brier ratio (95% CIs): 1.08–1.13 for difuse classes but 0.95 [0.948, 0.960] for Nodule, indicating that high nodule disagreement concentrates on better-predicted images. Mass (1.01) and Fibrosis (0.99) show no meaningful coupling. Thus, disagreement does not track error for focal lesions, although this does not imply per-crop correctness. The limitation concerns the score rather than classifier quality: raw Nodule AUC (0.740) is near the 14-class mean (0.7885, Appendix B). Full curves, referral statistics, and fairness analyzes are in Appendix I where gender parity is near-perfect (0.784 vs. 0.784).

Table 5: Failure detection on NIH ChestX-ray14 (DenseNet-121, $N { = } 2 5 , 5 9 6 )$ . <sup>†</sup>Post-hoc approximation.
<table><tr><td>Method</td><td>AUC</td><td>Passes</td><td colspan="2">Internals Train data</td><td colspan="2">Retrain Frozen ckpt?</td></tr><tr><td>Fusion (Ours)</td><td>0.832</td><td>6 + logistic reg</td><td>X</td><td>x</td><td>x</td><td>√</td></tr><tr><td>Deep Ensemble (Lakshminarayanan et al., 2017)</td><td>0.813</td><td>5×model</td><td>x</td><td>x</td><td>√</td><td>x</td></tr><tr><td>MUS (Ours)</td><td>0.784</td><td>6</td><td>x</td><td>x</td><td>x</td><td>√</td></tr><tr><td>Deep Ensemble + T-scaling</td><td>0.774</td><td>5×model+cal</td><td>x</td><td>√</td><td>√</td><td>x</td></tr><tr><td>ODIN (Liang et al., 2017)</td><td>0.759</td><td>1+grad</td><td>√</td><td>x</td><td>x</td><td>√</td></tr><tr><td>DDU† (Mukhoti et al., 2021)</td><td>0.754</td><td></td><td>√</td><td>√</td><td>x</td><td>x</td></tr><tr><td>Mahalanobis† (Lee et al., 2018)</td><td>0.754</td><td>1</td><td>√</td><td>√</td><td>x</td><td>x</td></tr><tr><td>MC-Dropout (Gal and Ghahramani, 2016)</td><td>0.664</td><td>1 30</td><td>x</td><td>X</td><td>√</td><td>X</td></tr></table>

Passes: forward passes per image. Internals: needs gradients/activations. Train data: needs training-set features. Retrain: needs model modification. Frozen ckpt?: applicable to any pretrained checkpoint.

## 6 DISCUSSION

A practitioner summary of when to use MUS is provided in Appendix I.1 (Table 32).

Complementarity and Regime Dependence. MUS is most useful in overconfident multi-label settings where confidence is weak (NIH DenseNet-121: AUC 0.401). On well-calibrated single-label tasks such as ImageNet, confidence dominates and MUS is complementary, with fusion combining the two signals.

Uncertainty type and OOD positioning. MUS measures whether model confidence is spatially reproducible, which we view as an epistemic signal reflecting learned spatial grounding rather than irreducible label noise. Its correlation with model quality also provides a pre-deployment diagnostic: $\rho \lesssim 0 . 1 5$ flags severe shift, as observed on VinBigData $( \rho = 0 . 0 2 7 )$ , before full test-set evaluation.

Why MUS over $\ell _ { 1 } ?$ Although $\ell _ { 1 }$ slightly exceeds MUS in AUC (0.790 vs. 0.784, $\Delta = 0 . 0 0 6 ; p < 1 0 ^ { - 6 } )$ its SCE is substantially worse (0.127 vs. 0.049). Under shift, MUS also performs better on VinBigData (0.614 vs. 0.583), consistent with the robustness of bounded, symmetric JSD. Lemma 1 establishes the connection between MUS and $\ell _ { 1 }$ , but the empirical advantage of MUS lies in its native calibration and shift robustness. Isotonic regression can calibrate $\ell _ { 1 }$ (SCE 0.006), but requires labeled hold-out data unavailable in frozen deployment. Fusion’s higher SCE (0.246) is instead driven by the uncalibrated entropy component.

Limitations. SpatialUQ has five main limitations. (i) Severe domain shift can collapse the inconsistency signal $\mathrm { ( V i n B i g D a t a }$ $\rho = 0 . 0 2 7 )$ . (ii) Five fixed crops limit sensitivity to small focal lesions (Nodule JSD AUC 0.368; 16-crop extension: 0.589). (iii) ViT global attention compresses MUS’s dynamic range, with masking recovering +0.004 AUC (Appendix C.4). (iv) Fusion requires a labeled hold-out set. (v) Generative

VLMs with global pooling can weaken the signal $( \mathrm { A p - }$ pendix C.5). The $\rho \lesssim 0 . 1 5$ diagnostic flags the observed failure regimes while remaining positive for discriminative settings (NIH 0.523, CheXpert 0.298); when flagged, MC-Dropout or ensembles are preferable, with MUS retained as a fusion component. Thus, the primary operating regime is low-cost failure detection for overconfident multi-label classifiers under moderate shift when retraining is infeasible. Generalization beyond radiography and RGB natural images remains untested.

Comparison with State-of-the-Art. Table 5 compares SpatialUQ with established UQ methods on NIH DenseNet-121. MUS achieves 0.784 AUC fully post-hoc, without retraining, training data, or architectural modifications, while competitive baselines require trainingtime structure (MC-Dropout, ensembles) or datasetdependent statistics (DDU, Mahalanobis). Extended analysis appears in Appendix K. Temperature scaling improves ensemble SCE from 0.138 to 0.090 but reduces AUC from 0.813 to 0.774, illustrating the trade-of between calibration and discrimination. Fusion reaches 0.832 AUC by adding the spatially grounded MUS signal, whose native SCE is 0.049.

## 7 FINAL REMARKS

SpatialUQ provides a practical operating point for inference-time failure detection: six deterministic forward passes using output probabilities alone, with no access to model internals, labels or retraining. Spatial inconsistency provides a geometrically grounded signal that scales with model quality, from 0.784 AUC on DenseNet-121 to 0.899 on BiomedCLIP $( \rho = 0 . 8 4 6 )$ The $\rho \lesssim 0 . 1 5$ diagnostic further provides a simple pre-deployment check for the observed distributionshift failure regimes. Future work includes saliencyguided crop placement, aggregation adapted to globalattention architectures, and broader validation on MIMIC-CXR with temporal shift, population-level AURC, and intersectional subgroup analyzes. Generalization to CT, MRI, and ultrasound also remains to be established.

## AI use statement

In this work, we used generative AI tools for writing assistance, including drafting, editing, and formatting of manuscript text, and for verifying the consistency of reported numbers against the experimental code. We have not used generative AI tools for generating experimental results, fabricating data, or conducting literature surveys, and all AI-assisted content was reviewed by the authors. We take responsibility for the final content of this work, including text, claims, or artifacts produced with the aid of generative AI.

## References

Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman, and Dan Mané. Concrete problems in ai safety. arXiv preprint arXiv:1606.06565, 2016.

Anastasios N Angelopoulos and Stephen Bates. Conformal prediction: A gentle introduction. Foundations and Trends in Machine Learning, 16(4):494– 591, 2023.

Murat Seckin Ayhan and Philipp Berens. Test-time data augmentation for estimation of heteroscedastic aleatoric uncertainty in deep neural networks. In Medical imaging with deep learning, 2018.

Simon Baur, Wojciech Samek, and Jackie Ma. Benchmarking uncertainty and its disentanglement in multi-label chest x-ray classification. In International Workshop on Uncertainty for Safe Utilization of Machine Learning in Medical Imaging, pages 193– 203. Springer, 2025.

Edmon Begoli, Tanmoy Bhattacharya, and Dimitri Kusnezov. The need for uncertainty quantification in machine-assisted medical decision making. Nature Machine Intelligence, 1(1):20–23, 2019.

Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, and Daan Wierstra. Weight uncertainty in neural network. In International conference on machine learning, pages 1613–1622. PMLR, 2015.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

Stanislav Fort, Huiyi Hu, and Balaji Lakshminarayanan. Deep ensembles: A loss landscape perspective. arXiv preprint arXiv:1912.02757, 2019.

Yarin Gal and Zoubin Ghahramani. Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In international conference on machine learning, pages 1050–1059. PMLR, 2016.

Yonatan Geifman and Ran El-Yaniv. Selective classification for deep neural networks. Advances in neural information processing systems, 30, 2017.

Ian J Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014.

Alex Graves. Practical variational inference for neural networks. Advances in neural information processing systems, 24, 2011.

Chuan Guo, Geof Pleiss, Yu Sun, and Kilian Q Weinberger. On calibration of modern neural networks. In International conference on machine learning, pages 1321–1330. PMLR, 2017.

Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. arXiv preprint arXiv:1903.12261, 2019.

Geofrey Hinton, Oriol Vinyals, and Jef Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.

Neil Houlsby, Ferenc Huszár, Zoubin Ghahramani, and Máté Lengyel. Bayesian active learning for classification and preference learning. arXiv preprint arXiv:1112.5745, 2011.

Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q Weinberger. Densely connected convolutional networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4700–4708, 2017.

Jeremy Irvin, Pranav Rajpurkar, Michael Ko, Yifan Yu, Silviana Ciurea-Ilcus, Chris Chute, Henrik Marklund, Behzad Haghgoo, Robyn Ball, Katie Shpanskaya, et al. Chexpert: A large chest radiograph dataset with uncertainty labels and expert comparison. In Proceedings of the AAAI conference on artificial intelligence, pages 590–597, 2019.

Alex Kendall and Yarin Gal. What uncertainties do we need in bayesian deep learning for computer vision? Advances in neural information processing systems, 30, 2017.

Zaid Khan and Yun Fu. Consistency and uncertainty: Identifying unreliable responses from blackbox vision-language models for selective visual question answering. In Proceedings of the ieee/cvf conference on computer vision and pattern recognition, pages 10854–10863, 2024.

Alex Krizhevsky, Ilya Sutskever, and Geofrey E Hinton. Imagenet classification with deep convolutional neural networks. Advances in neural information processing systems, 25, 2012.

Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. Advances in neural information processing systems, 30, 2017.

Kimin Lee, Kibok Lee, Honglak Lee, and Jinwoo Shin. A simple unified framework for detecting out-ofdistribution samples and adversarial attacks. Advances in neural information processing systems, 31, 2018.

Christian Leibig, Vaneeda Allken, Murat Seçkin Ayhan, Philipp Berens, and Siegfried Wahl. Leveraging uncertainty information from deep neural networks for disease detection. Scientific reports, 7(1):1–14, 2017.

Shiyu Liang, Yixuan Li, and Rayadurgam Srikant. Enhancing the reliability of out-of-distribution image detection in neural networks. arXiv preprint arXiv:1706.02690, 2017.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer, 2014.

Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. Focal loss for dense object detection. In Proceedings of the IEEE international conference on computer vision, pages 2980–2988, 2017.

Weitang Liu, Xiaoyun Wang, John Owens, and Yixuan Li. Energy-based out-of-distribution detection. Advances in neural information processing systems, 33: 21464–21475, 2020.

Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10012–10022, 2021.

Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11976–11986, 2022.

Dimity Miller, Feras Dayoub, Michael Milford, and Niko Sünderhauf. Evaluating merging strategies for sampling-based uncertainty techniques in object detection. In 2019 international conference on robotics and automation (icra), pages 2348–2354. IEEE, 2019.

Matthias Minderer, Josip Djolonga, Rob Romijnders, Frances Hubis, Xiaohua Zhai, Neil Houlsby, Dustin Tran, and Mario Lucic. Revisiting the calibration of modern neural networks. Advances in neural information processing systems, 34:15682–15694, 2021.

Jishnu Mukhoti, Andreas Kirsch, Joost Van Amersfoort, PH Torr, and Yarin Gal. Deterministic neural networks with inductive biases capture epistemic and aleatoric uncertainty. arXiv preprint arXiv:2102.11582, 2, 2021.

Jishnu Mukhoti, Andreas Kirsch, Joost Van Amersfoort, Philip HS Torr, and Yarin Gal. Deep deterministic uncertainty: A new simple baseline. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 24384–24394, 2023.

Zachary Nado, Neil Band, Mark Collier, Josip Djolonga, Michael W Dusenberry, Sebastian Farquhar, Qixuan Feng, Angelos Filos, Marton Havasi, Rodolphe Jenatton, et al. Uncertainty baselines: Benchmarks for uncertainty & robustness in deep learning. arXiv preprint arXiv:2106.04015, 2021.

Ha Q Nguyen, Khanh Lam, Linh T Le, Hieu H Pham, Dat Q Tran, Dung B Nguyen, Dung D Le, Chi M Pham, Hang TT Tong, Diep H Dinh, et al. Vindr-cxr: An open dataset of chest x-rays with radiologist’s annotations. Scientific Data, 9(1):429, 2022.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

Jie Ren, Peter J Liu, Emily Fertig, Jasper Snoek, Ryan Poplin, Mark Depristo, Joshua Dillon, and Balaji Lakshminarayanan. Likelihood ratios for outof-distribution detection. Advances in neural information processing systems, 32, 2019.

Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. Advances in neural information processing systems, 28, 2015.

Abhijit Guha Roy, Sailesh Conjeti, Nassir Navab, Christian Wachinger, Alzheimer’s Disease Neuroimaging Initiative, et al. Bayesian quicknat: Model uncertainty in deep whole-brain segmentation for structure-wise quality control. NeuroImage, 195:11– 22, 2019.

Divya Shanmugam, Davis Blalock, Guha Balakrishnan, and John Guttag. Better aggregation in test-time augmentation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1214–1223, 2021.

Claude Elwood Shannon. A mathematical theory of communication. The Bell system technical journal, 27(3):379–423, 1948.

Han-Jay Shu, Wei-Ning Chiu, Shun-Ting Chang, Meng-Ping Huang, Takeshi Tohyama, Ahram Han, and

Po-Chih Kuo. Uncovering overconfident failures in cxr models via augmentation-sensitivity risk scoring. In ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 6736–6740. IEEE, 2026.

Mingxing Tan and Quoc Le. Eficientnet: Rethinking model scaling for convolutional neural networks. In International conference on machine learning, pages 6105–6114. PMLR, 2019.

Guotai Wang, Wenqi Li, Michael Aertsen, Jan Deprest, Sébastien Ourselin, and Tom Vercauteren. Aleatoric uncertainty estimation with test-time augmentation for medical image segmentation with convolutional neural networks. Neurocomputing, 338:34–45, 2019.

Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, Mohammadhadi Bagheri, and Ronald M Summers. Chestx-ray8: Hospital-scale chest x-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2097–2106, 2017.

Sheng Zhang, Yanbo Xu, Naoto Usuyama, Hanwen Xu, Jaspreet Bagga, Robert Tinn, Sam Preston, Rajesh Rao, Mu Wei, Naveen Valluri, et al. Biomedclip: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs. arXiv preprint arXiv:2303.00915, 2023.

Han Zhou, Jordy Van Landeghem, Teodora Popordanoska, and Matthew B Blaschko. A novel characterization of the population area under the risk coverage curve (aurc) and rates of finite sample estimators. arXiv preprint arXiv:2410.15361, 2024.

## CHECKLIST

1. For all models and algorithms presented, check if you include:

(a) A clear description of the mathematical setting, assumptions, algorithm, and/or model. [Yes] The spatial decomposition, Bernoulli/categorical JSD scores, and the logistic fusion are defined in Section 3, with the full procedure in Algorithm 1.

(b) An analysis of the properties and complexity (time, space, sample size) of any algorithm. [Yes] SpatialUQ needs exactly six forward passes per image with no stored state (Section 3); wall-clock timings are in Appendix J. Lemma 1 gives the information-theoretic property of the score.

(c) (Optional) Anonymized Source code, with specification of all dependencies, including external libraries. [Yes] Code and data splits are included in the supplementary material, with dependencies listed in the repository requirements file.

2. For any theoretical claim, check if you include:

(a) Statements of the full set of assumptions of all theoretical results. [Yes] Lemma 1 states its assumptions $( p _ { c } , q _ { c } \in [ 0 , 1 ]$ , JSD in nats) directly.

(b) Complete proofs of all theoretical results. [Yes] The full proof is in Appendix A, with a proof sketch in Section 3.4.

(c) Clear explanations of any assumptions. [Yes] Section 3.4 explains that the bound concerns probability shift only and that the link to failure detection is empirical (Spearman correlation).

3. For all figures and tables that present empirical results, check if you include:

(a) The code, data, and instructions needed to reproduce the main experimental results (either in the supplemental material or as a URL). [Yes] Code, data splits, and fixed random seeds (seed 42) are in the supplementary material.

(b) All the training details (e.g., data splits, hyperparameters, how they were chosen). [Yes] Section 4.3 and Appendix J give the two-stage protocol, learning rates, augmentation, and the 73,891/12,633/25,596 patient-wise split.

(c) A clear definition of the specific measure or statistics and error bars (e.g., with respect to the random seed after running experiments multiple times). [Yes] Failure is defined as Brier score above the 75th percentile (Section 4.2); all AUCs carry 95% bootstrap CIs (1,000 resamples) and pairwise significance uses DeLong tests with 10,000-resample bootstrap p-values.

(d) A description of the computing infrastructure used. (e.g., type of GPUs, internal cluster, or cloud provider). [Yes] All NIH experiments ran on a single NVIDIA Tesla T4 GPU (Appendix J).

4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets, check if you include:

(a) Citations of the creator If your work uses existing assets. [Yes] NIH ChestX-ray14 (Wang et al., 2017), CheXpert (Irvin et al., 2019), VinBigData (Nguyen et al., 2022), MS COCO (Lin et al., 2014), CLIP (Radford et al.,

2021), and BiomedCLIP (Zhang et al., 2023) are cited, as are the DenseNet (Huang et al., 2017), EficientNet (Tan and Le, 2019), and ViT (Dosovitskiy et al., 2020) backbones.

(b) The license information of the assets, if applicable. [Yes] All datasets and pretrained models are publicly released research assets used under their published terms.

(c) New assets either in the supplemental material or as a URL, if applicable. [Yes] The SpatialUQ implementation notebooks are available at this link https://huggingface.co/ datasets/kawsher11/SpatialUQ.

(d) Information about consent from data providers/curators. [Not Applicable] All datasets are pre-existing public benchmarks; no new data was collected.

(e) Discussion of sensible content if applicable, e.g., personally identifiable information or offensive content. [Not Applicable] The chest X-ray benchmarks are de-identified; the fairness audit in Appendix I uses only aggregate age/gender metadata.

5. If you used crowdsourcing or conducted research with human subjects, check if you include:

(a) The full text of instructions given to participants and screenshots. [Not Applicable]

(b) Descriptions of potential participant risks, with links to Institutional Review Board (IRB) approvals if applicable. [Not Applicable]

(c) The estimated hourly wage paid to participants and the total amount spent on participant compensation. [Not Applicable]

No crowdsourcing or human-subjects research was conducted; the triage study in Section 5.7 is a retrospective simulation on de-identified benchmark data.

# SpatialUQ: Post-Hoc Uncertainty Quantification from Spatial Consistency in Black-Box Vision Models: Supplementary Materials

## A PROOF OF LEMMA 1

We prove the spatial disagreement bound in two steps. All logarithms are natural (JSD in nats) unless stated otherwise.

Step 1: Per-class bound via Pinsker’s inequality. For Bernoulli distributions with parameters $p _ { c } , q _ { c } \in [ 0 , 1 ]$ let $m _ { c } = \textstyle { \frac { 1 } { 2 } } ( p _ { c } + q _ { c } )$ . The JSD decomposes as:

$$
\mathrm { J S D } _ { c } ( p _ { c } \| q _ { c } ) = \frac { 1 } { 2 } \mathrm { K L } ( p _ { c } \| m _ { c } ) + \frac { 1 } { 2 } \mathrm { K L } ( q _ { c } \| m _ { c } )\tag{7}
$$

For Bernoulli distributions, the total variation distance between parameters $p _ { c }$ and $m _ { c }$ reduces to $\mathrm { T V } ( p _ { c } , m _ { c } ) =$ $| p _ { c } - m _ { c } | = \textstyle { \frac { 1 } { 2 } } | p _ { c } - q _ { c } |$ . Pinsker’s inequality $\begin{array} { r } { \mathrm { T V } ( P , Q ) ^ { 2 } \leq \frac { 1 } { 2 } \mathrm { K L } ( \bar { P } \| Q ) } \end{array}$ then gives:

$$
\frac 1 4 ( p _ { c } - q _ { c } ) ^ { 2 } \leq \frac 1 2 \mathrm { K L } ( p _ { c } \| m _ { c } ) \quad \Longrightarrow \quad ( p _ { c } - q _ { c } ) ^ { 2 } \leq 2 \mathrm { K L } ( p _ { c } \| m _ { c } )\tag{8}
$$

Applying the same argument symmetrically to $\mathrm { K L } ( q _ { c } \| m _ { c } )$

$$
( p _ { c } - q _ { c } ) ^ { 2 } \leq 2 \mathrm { K L } ( q _ { c } \| m _ { c } )\tag{9}
$$

Adding the two inequalities above:

$$
2 ( p _ { c } - q _ { c } ) ^ { 2 } \leq 2 \big ( \mathrm { K L } ( p _ { c } \| m _ { c } ) + \mathrm { K L } ( q _ { c } \| m _ { c } ) \big ) = 4 \mathrm { J S D } _ { c } ( p _ { c } \| q _ { c } )\tag{10}
$$

Dividing by 2 and taking square roots:

$$
| p _ { c } - q _ { c } | \leq \sqrt { 2 \mathrm { J S D } _ { c } ( p _ { c } \| q _ { c } ) } \qquad \mathrm { ( i n ~ n a t s ) }\tag{11}
$$

Step 2: Aggregation over classes via Jensen’s inequality. Averaging Eq. (11) over all C classes:

$$
{ \frac { 1 } { C } } \sum _ { c = 1 } ^ { C } | p _ { c } - q _ { c } | \leq { \frac { 1 } { C } } \sum _ { c = 1 } ^ { C } { \sqrt { 2 \operatorname { J S D } _ { c } } }\tag{12}
$$

Since $\sqrt { \cdot }$ is concave, Jensen’s inequality gives:

$$
{ \frac { 1 } { C } } \sum _ { c = 1 } ^ { C } { \sqrt { 2 \mathrm { J S D } _ { c } } } \leq { \sqrt { { \frac { 2 } { C } } \sum _ { c = 1 } ^ { C } { \mathrm { J S D } _ { c } } } } = { \sqrt { 2 s _ { \mathrm { b e r n } } ( x ) } } \qquad { \mathrm { ( i n ~ n a t s ) } }\tag{13}
$$

which is the complete bound:

$$
{ \frac { 1 } { C } } \sum _ { c = 1 } ^ { C } \bigl | p _ { \mathrm { g l o b a l } , c } - p _ { \mathrm { l o c a l } , c } \bigr | \ \leq \ \sqrt { 2 \cdot s _ { \mathrm { b e r n } } ( x ) } \qquad\tag{14}
$$

Verification of tightness. The per-class bound in Eq. (11) is tight in the limit $p _ { c } , q _ { c } \to { \frac { 1 } { 2 } }$ . Setting $\begin{array} { r } { p _ { c } = \frac { 1 } { 2 } + \varepsilon , } \end{array}$ $q _ { c } = { \frac { 1 } { 2 } } - \varepsilon \ ( \mathrm { s o } m _ { c } = { \frac { 1 } { 2 } } )$ , a second-order expansion gives $\mathrm { K L } ( p _ { c } \| m _ { c } ) = \mathrm { K L } ( q _ { c } \| m _ { c } ) \stackrel { \sim } { = } 2 \varepsilon ^ { 2 } + o ( \varepsilon ^ { 2 } )$ , hence $\mathrm { J S D } _ { c } \stackrel { = } { = } 2 \varepsilon ^ { 2 } + o ( \varepsilon ^ { 2 } )$ and

$$
\mathrm { R H S } = \sqrt { 2 \mathrm { J S D } _ { c } } = 2 \varepsilon + o ( \varepsilon ) = | p _ { c } - q _ { c } | + o ( \varepsilon ) = \mathrm { L H S } + o ( \varepsilon ) ,\tag{15}
$$

so the ratio of the two sides tends to 1 as $\varepsilon \to 0$

## B BACKBONE CLASSIFICATION PERFORMANCE

Table 6 reports per-class AUC-ROC, Brier score, and average precision for the DenseNet-121 backbone (7.0 M parameters) trained on NIH ChestX-ray14, selected at epoch 28 (validation loss of 2.0564). The model achieves a mean AUC of 0.7885, with discriminative performance ranging from 0.6984 (Infiltration) to 0.8822 (Emphysema). Calibration and average precision exhibit pronounced class-level disparity, largely attributable to severe label imbalance. These results serve as the reference operating point for the subsequent failure-detection analysis.

Table 6: NIH ChestX-ray14 backbone classification performance - DenseNet-121 (7.0 M parameters, best checkpoint epoch 28, validation loss 2.0564). Mean AUC of 0.7885.
<table><tr><td>Class</td><td>AUC</td><td>Brier</td><td>AP</td></tr><tr><td>Atelectasis</td><td>0.7628</td><td>0.3189</td><td>0.3272</td></tr><tr><td>Cardiomegaly</td><td>0.8746</td><td>0.5565</td><td>0.3350</td></tr><tr><td>Effusion</td><td>0.8192</td><td>0.3081</td><td>0.4907</td></tr><tr><td>Infiltration</td><td>0.6984</td><td>0.3161</td><td>0.3969</td></tr><tr><td>Mass</td><td>0.7965</td><td>0.3576</td><td>0.2915</td></tr><tr><td>Nodule</td><td>0.7400</td><td>0.3869</td><td>0.2191</td></tr><tr><td>Pneumonia</td><td>0.7020</td><td>0.7346</td><td>0.0463</td></tr><tr><td>Pneumothorax</td><td>0.8471</td><td>0.4472</td><td>0.3864</td></tr><tr><td>Consolidation</td><td>0.7401</td><td>0.5084</td><td>0.1562</td></tr><tr><td>Edema</td><td>0.8368</td><td>0.6476</td><td>0.1494</td></tr><tr><td>Emphysema</td><td>0.8822</td><td>0.6111</td><td>0.3595</td></tr><tr><td>Fibrosis</td><td>0.8095</td><td>0.6404</td><td>0.0888</td></tr><tr><td>Pleural_Thickening</td><td>0.7569</td><td>0.5222</td><td>0.1245</td></tr><tr><td>Hernia</td><td>0.7733</td><td>0.9428</td><td>0.0434</td></tr><tr><td>Mean</td><td>0.7885</td><td>-</td><td>一</td></tr></table>

Table 7 reports per-class AUC-ROC, Brier score, and average precision for the EficientNet-B4 backbone (17.6 M parameters) on NIH ChestX-ray14, selected at epoch 29 (validation loss 2.0793). The model achieves a mean AUC of 0.7347, trailing DenseNet-121 by 0.054 points despite nearly 2.5× more parameters, with per-class AUC ranging from 0.6589 (Pneumonia) to 0.8093 (Edema). Calibration and average precision mirror the class-imbalance patterns observed in the DenseNet-121 baseline. Both backbones are subsequently treated as frozen black-boxes for post-hoc uncertainty evaluation.

Table 7: NIH ChestX-ray14 backbone classification performance - EficientNet-B4 (17.6 M parameters, best checkpoint epoch 29, validation loss 2.0793). Mean AUC = 0.7347.
<table><tr><td>Class</td><td>AUC</td><td>Brier</td><td>AP</td></tr><tr><td>Atelectasis</td><td>0.7192</td><td>0.3321</td><td>0.2732</td></tr><tr><td>Cardiomegaly</td><td>0.7952</td><td>0.5746</td><td>0.2118</td></tr><tr><td>Effusion</td><td>0.7740</td><td>0.3250</td><td>0.4189</td></tr><tr><td>Infiltration</td><td>0.6790</td><td>0.3102</td><td>0.3733</td></tr><tr><td>Mass</td><td>0.7002</td><td>0.4114</td><td>0.1591</td></tr><tr><td>Nodule</td><td>0.6941</td><td>0.4039</td><td>0.1520</td></tr><tr><td>Pneumonia</td><td>0.6589</td><td>0.7389</td><td>0.0375</td></tr><tr><td>Pneumothorax</td><td>0.8050</td><td>0.4824</td><td>0.3031</td></tr><tr><td>Consolidation</td><td>0.7022</td><td>0.5199</td><td>0.1299</td></tr><tr><td>Edema</td><td>0.8093</td><td>0.6440</td><td>0.1115</td></tr><tr><td>Emphysema</td><td>0.7977</td><td>0.6235</td><td>0.1405</td></tr><tr><td>Fibrosis</td><td>0.7547</td><td>0.6533</td><td>0.0575</td></tr><tr><td>Pleural_Thickening</td><td>0.7201</td><td>0.5370</td><td>0.0922</td></tr><tr><td>Hernia</td><td>0.6763</td><td>0.9437</td><td>0.0096</td></tr><tr><td>Mean</td><td>0.7347</td><td>-</td><td>-</td></tr></table>

Table 8 reports per-class AUC-ROC, Brier score, and average precision for the ViT-B/16 backbone (85.8,M parameters) on NIH ChestX-ray14, with early stopping triggered at epoch 17 and the best checkpoint selected at epoch 9. Despite its substantially larger capacity, the model achieves a mean AUC of 0.7921, marginally exceeding DenseNet-121 (0.7885) and EficientNet-B4 (0.7347), with per-class AUC ranging from 0.6978 (Infiltration) to 0.8868 (Cardiomegaly). Calibration patterns remain consistent with the convolutional backbones, with Hernia (Brier = 0.9413) and Pneumonia (0.7273) exhibiting the poorest posterior estimates. This backbone is of particular

interest for examining the relationship between global self-attention and spatial prediction inconsistency in the failure-detection analysis.

Table 8: NIH ChestX-ray14 backbone classification performance - ViT-B/16 (85.8 M parameters, early stopping at epoch 17, best validation loss at epoch 9). Mean AUC = 0.7921.
<table><tr><td>Class</td><td>AUC</td><td>Brier</td><td>AP</td></tr><tr><td>Atelectasis</td><td>0.7689</td><td>0.2882</td><td>0.3399</td></tr><tr><td>Cardiomegaly</td><td>0.8868</td><td>0.5685</td><td>0.3500</td></tr><tr><td>Effusion</td><td>0.8257</td><td>0.3037</td><td>0.5059</td></tr><tr><td>Infiltration</td><td>0.6978</td><td>0.3029</td><td>0.3983</td></tr><tr><td>Mass</td><td>0.8157</td><td>0.3817</td><td>0.3176</td></tr><tr><td>Nodule</td><td>0.7502</td><td>0.3712</td><td>0.2230</td></tr><tr><td>Pneumonia</td><td>0.7111</td><td>0.7273</td><td>0.0555</td></tr><tr><td>Pneumothorax</td><td>0.8587</td><td>0.4469</td><td>0.4263</td></tr><tr><td>Consolidation</td><td>0.7521</td><td>0.4987</td><td>0.1686</td></tr><tr><td>Edema</td><td>0.8359</td><td>0.6364</td><td>0.1653</td></tr><tr><td>Emphysema</td><td>0.8854</td><td>0.5791</td><td>0.4034</td></tr><tr><td>Fibrosis</td><td>0.7938</td><td>0.6235</td><td>0.0857</td></tr><tr><td>Pleural_Thickening</td><td>0.7622</td><td>0.5380</td><td>0.1367</td></tr><tr><td>Hernia</td><td>0.7446</td><td>0.9413</td><td>0.0347</td></tr><tr><td>Mean</td><td>0.7921</td><td>-</td><td>1</td></tr></table>

## C NIH CHESTX-RAY14: FULL FAILURE DETECTION TABLES

## C.1 DenseNet-121: All Methods Comparison

The results in Table 9 evaluate failure detection performance on the NIH ChestX-ray14 test set using a DenseNet-121 backbone. Failure is defined as predictions with Brier score exceeding the 75th percentile (threshold = 0.5607, yielding a 25% failure rate). We report area under the ROC curve (AUC) with 95% bootstrap confidence intervals (1,000 resamples) and Score Calibration Error (SCE)the expected calibration error applied to min–max normalized scores using 15 bins. Among all methods, the proposed MUS achieves the lowest SCE (0.0487), indicating well-calibrated failure probabilities, while its AUC (0.7837) is competitive. The highest AUC is obtained by Fusion CV (0.8319), which combines multiple uncertainty signals, followed closely by the two Auditor variants (0.8250, 0.8242). However, these top AUC methods exhibit substantially higher SCE (0.2464, 0.2408, 0.2363) than MUS. Deep Ensemble (5 members) yields strong AUC (0.8130) and moderate SCE (0.1379), whereas MUS strikes a favorable trade-of with near-perfect calibration. Notably, simple post-hoc baselines like T-Scaled Confidence<sup>−1</sup> achieve excellent SCE (0.0015) but near-random AUC (0.4000), highlighting that calibration alone is insuficient for failure detection. Conversely, Entropy performs poorly on both metrics (AUC 0.1774, SCE 0.4575). Methods marked with † require more than six forward passes or training-time modifications; ‡ indicates access to training-set features. The table confirms that no single method dominates all criteria, but MUS provides the best calibration among practically viable approaches.

Statistical tests (DeLong, two-tailed): MUS vs. MC-Dropout: $\Delta \mathrm { A U C } = + 0 . 1 1 9 4 , p < 1 0 ^ { - 6 }$ . MUS vs. $\ell _ { 1 }$ Distance: $\Delta \mathrm { A U C } = - 0 . 0 0 6 4 , p < 1 0 ^ { - 6 } \left( \ell _ { 1 } \right.$ marginally superior). Fusion vs. Deep Ensemble: $\Delta \mathrm { A U C } = + 0 . 0 1 8 9$ $p < 1 0 ^ { - 6 }$ . Spearman correlation between MUS and Brier score: $\rho = 0 . 5 2 2 5 , p < 1 0 ^ { - 6 }$ (Pearson $r = 0 . 4 9 8 3 )$ .

Table 10 demonstrates that MUS exhibits strong robustness to random initialization, with AUC values remaining tightly clustered across diferent seeds (0.7826–0.7902), indicating low variance and stable behavior. The fluctuation range (0.0076) is minimal, suggesting that MUS is not sensitive to stochastic training efects and generalizes consistently. In comparison, $\ell _ { 1 }$ Distance achieves slightly higher absolute performance across all seeds but follows a similar stability pattern, while MC-Dropout and TTA Variance show modest variability and consistently lower AUC. Notably, confidence-based and entropy-based methods remain significantly weaker and exhibit inconsistent behavior across seeds, highlighting their limited reliability. Pairwise JSD also shows mild degradation for certain seeds, suggesting sensitivity to randomness in feature representations. Overall, these results reinforce that MUS provides a reliable and reproducible uncertainty signal, maintaining competitive performance with minimal seed-induced variance an important property for deployment in high-stakes settings.

Table 9: NIH ChestX-ray14 failure detection DenseNet-121, full 16-method comparison $( N = 2 5 { , } 5 9 6 )$ . Brier threshold = 0.5607 (75th percentile, 25% failure rate). SCE = Score Calibration Error (ECE applied to min-max normalised scores, 15 bins). † requires >6 forward passes or training-time modification. ‡ requires access to training-set features.
<table><tr><td>Method</td><td>AUC</td><td>95% CI</td><td>SCE</td></tr><tr><td>Fusion CV (MUS+Ent+</td><td>0.8319</td><td>[0.8268, 0.8368]</td><td>0.2464</td></tr><tr><td>Conf+l1) Auditor</td><td></td><td></td><td></td></tr><tr><td>(3-feat MLP)†</td><td>0.8250</td><td>[0.8201, 0.8299]</td><td>0.2408</td></tr><tr><td>Auditor (6-feat MLP)†</td><td>0.8242</td><td>[0.8189, 0.8292]</td><td>0.2363</td></tr><tr><td>Deep Ensemble (5-member)†</td><td>0.8130</td><td>[0.8077, 0.8187]</td><td>0.1379</td></tr><tr><td>l1 Distance</td><td>0.7902</td><td>[0.7848, 0.7960]</td><td>0.1270</td></tr><tr><td>MUS (Ours)</td><td>0.7837</td><td>[0.7778, 0.7893]</td><td>0.0487</td></tr><tr><td>Deep Ensemble + T-scaling†</td><td>0.7741</td><td>[0.7680, 0.7801]</td><td>0.0897</td></tr><tr><td>ODIN</td><td>0.7588</td><td>[0.7518, 0.7653]</td><td>0.4844</td></tr><tr><td>TTA-JSD photometric</td><td>0.7549</td><td>[0.7492, 0.7607]</td><td>0.0430</td></tr><tr><td>(same formula, 5 augs)</td><td></td><td></td><td></td></tr><tr><td>DDÚ Score</td><td>0.7540</td><td>[0.7476, 0.7599]</td><td>0.1386</td></tr><tr><td>Mahalanobis</td><td>0.7540</td><td>[0.7480, 0.7596]</td><td>0.2828</td></tr><tr><td>Max Disagree-</td><td>0.7440</td><td>[0.7376, 0.7502]</td><td>0.1861</td></tr><tr><td>ment MC-Dropout</td><td>0.6643</td><td>[0.6575, 0.6710]</td><td>0.1211</td></tr><tr><td>(T=30)† Ensemble</td><td>0.6579</td><td>[0.6505, 0.6648]</td><td>0.0419</td></tr><tr><td>Variance† TTA (5 crops,</td><td>0.6103</td><td>[0.6022, 0.6178]</td><td>0.0290</td></tr><tr><td>patch variance) Pairwise JSD</td><td>0.6057</td><td>[0.5979, 0.6133]</td><td>0.0335</td></tr><tr><td>Pairwise Dis- agreement</td><td>0.5990</td><td>[0.5913, 0.6073]</td><td>0.2436</td></tr><tr><td>TTA Variance</td><td>0.5837</td><td>[0.5759, 0.5914]</td><td>0.0889</td></tr><tr><td>(N=10)† Confidence -1</td><td>0.4011</td><td>[0.3922, 0.4091]</td><td>0.4218</td></tr><tr><td>T-Scaled  $\mathrm { { C o n f ^ { - 1 } } }$ </td><td>0.4000</td><td>[0.3910, 0.4081]</td><td>0.0015</td></tr><tr><td>Entropy</td><td>0.1774</td><td>[0.1727, 0.1821]</td><td>0.4575</td></tr></table>

Table 10: Seed MUS: Comparison across diferent random seeds for DenseNet121 on NIH ChestX-ray14.
<table><tr><td>Seed</td><td>MUS AUC</td><td> $\ell _ { 1 }$  Dist.</td><td>MC-Drop</td><td> $\mathbf { C o n f } ^ { - 1 }$ </td><td>TTA Var</td><td>Entr.</td><td>Pair JSD</td></tr><tr><td>42</td><td>0.7837</td><td>0.7902</td><td>0.6643</td><td>0.4011</td><td>0.5837</td><td>0.1774</td><td>0.6057</td></tr><tr><td>186</td><td>0.7866</td><td>0.7959</td><td>0.6466</td><td>0.5802</td><td>0.5723</td><td>0.1748</td><td>0.6023</td></tr><tr><td>456</td><td>0.7902</td><td>0.7976</td><td>0.6532</td><td>0.5816</td><td>0.5856</td><td>0.1672</td><td>0.5709</td></tr><tr><td>789</td><td>0.7862</td><td>0.7914</td><td>0.6498</td><td>0.5788</td><td>0.5797</td><td>0.1681</td><td>0.5806</td></tr><tr><td>1011</td><td>0.7826</td><td>0.7844</td><td>0.6536</td><td>0.5737</td><td>0.5848</td><td>0.1688</td><td>0.5660</td></tr><tr><td>Mean</td><td>0.7859</td><td>0.7919</td><td>0.6535</td><td>0.5431</td><td>0.5812</td><td>0.1713</td><td>0.5851</td></tr><tr><td>± Std</td><td>0.0026</td><td>0.0047</td><td>0.0060</td><td>0.0700</td><td>0.0050</td><td>0.0040</td><td>0.0160</td></tr></table>

## C.2 Foundation Model Evaluation with Linear Probes

To assess SpatialUQ’s transferability and scaling with model quality, we evaluate CLIP ViT-B/32 and BiomedCLIP on the NIH ChestX-ray14 test set with a linear probe trained on NIH train features (backbone frozen; no fine tuning). Labels are used only to train the probe and for evaluation. Table 11 reports failure detection AUC (75th percentile Brier threshold, 25% failure rate), 95% bootstrap CIs, and raw Score Calibration Error (SCE). MUS achieves competitive failure detection across both models, with AUC improving from 0.825 (CLIP) to 0.899 (BiomedCLIP). This scaling is accompanied by an increase in Spearman correlation between MUS and per-image Brier score $( \rho = 0 . 6 3 1$ for CLIP, $\rho = 0 . 8 4 6$ for BiomedCLIP), consistent with the hypothesis that better spatial representations yield stronger spatial consistency signals. Under linear-probe transfer, MUS’s calibration remains moderate (SCE 0.10-0.13) and superior to the raw $L _ { 1 }$ baseline (0.115-0.232). $L _ { 1 }$ and fusion achieve higher AUC but rely on a labeled hold-out for calibration (fusion) or would need isotonic regression $\left( L _ { 1 } \right)$ to produce reliable thresholds, options not always available in frozen black-box deployment. MUS therefore ofers a label-free, natively calibrated uncertainty score that scales favorably with foundation model strength.

Table 11: Failure detection on NIH ChestX-ray14 with frozen foundation models plus linear probe (25,596 images). MUS requires only output probabilities; $L _ { 1 }$ and Fusion use the same six forward passes. Fusion is a logistic regression trained on a small labeled hold-out (10% of the test set, remaining 90% for evaluation).
<table><tr><td rowspan="3">Method</td><td colspan="3">CLIP ViT-B/32</td><td colspan="3">BiomedCLIP</td></tr><tr><td>AUC</td><td> [95% CI]</td><td>SCE</td><td></td><td>AUC [95% CI]</td><td>SCE</td></tr><tr><td>MUS (SpatialUQ)</td><td></td><td>0.825 [0.818, 0.830]</td><td>0.098</td><td></td><td>0.899 [0.895, 0.903]</td><td>0.133</td></tr><tr><td>Entropy</td><td></td><td>0.204 [0.199, 0.210]</td><td>0.574</td><td></td><td>0.394 [0.387, 0.402]</td><td>0.421</td></tr><tr><td> $\mathrm { C o n f i d e n c e } ^ { - 1 }$ </td><td></td><td>0.095 [0.092, 0.099]</td><td>0.429</td><td></td><td>0.206 [0.201, 0.212]</td><td>0.334</td></tr><tr><td> $L _ { 1 } \ \mathrm { D i s t a n c e }$ </td><td></td><td>0.837 [0.831, 0.842]</td><td>0.115</td><td>0.915</td><td>[0.911, 0.918]</td><td>0.232</td></tr><tr><td>Fusion</td><td></td><td>0.883 [0.879, 0.887]</td><td>0.234</td><td></td><td>0.925 [0.922, 0.928]</td><td>0.278</td></tr></table>

![](images/89abddf7bce1f62f94c166b24d2dd59f116cc253d45381eb47b08c5d293a9f2c.jpg)  
(a) CLIP ViT-B/32 (AUC 0.825)

![](images/da28ac12964c9d743e2fc6fcd867fa8a3d81d3f169c578beb05c9ef876d979d0.jpg)  
(b) BiomedCLIP (AUC 0.899)  
Figure 6: ROC curves for failure detection on NIH ChestX-ray14 with frozen foundation models plus linear probe. Left: CLIP ViT-B/32 (AUC 0.825). Right: BiomedCLIP (AUC 0.899). MUS (SpatialUQ) outperforms entropy and confidence<sup>−1</sup> baselines.

Figure 6 shows the corresponding ROC curves. MUS consistently outperforms entropy and confidence baselines, and its discrimination improves from CLIP to BiomedCLIP.

## C.3 EficientNet-B4: Full Failure Detection Comparison

Table 12 reports failure detection AUC for EficientNet-B4 on the NIH test set $( N = 2 5 , 5 9 6 )$ , with failures defined as samples exceeding the 75th-percentile Brier score threshold (0.5776), yielding a 25% failure rate. Fusion CV achieves the highest AUC (0.8714), followed by MUS (0.7842) and Mahalanobis (0.7804), with the three methods forming a statistically separated tier above ODIN (0.7681) and Max Disagreement (0.6722). MC-Dropout performs near chance (0.5357), significantly below MUS (∆AUC = +0.2485, 95% CI [0.2381, 0.2589], $p < 1 0 ^ { - 6 } )$

Table 12: NIH ChestX-ray14 failure detection EficientNet-B4, 14-method comparison $( N = 2 5 { , } 5 9 6 )$ . Brier threshold = 0.5776 (75th percentile, 25% failure rate). SCE = Score Calibration Error (ECE applied to min-max normalised scores, 15 bins). † requires >6 forward passes or training-time modification. ‡ requires access to training-set features. DDU, Pairwise JSD and TTA Entropy are degenerate (constant scores) on this backbone.
<table><tr><td>Method</td><td>AUC</td><td>95% CI</td><td>SCE</td></tr><tr><td>Fusion CV (MUS+Ent+</td><td>0.8714</td><td>[0.8675, 0.8756]</td><td>0.2713</td></tr><tr><td>Conf+l1) MUS (Ours)</td><td>0.7842</td><td>[0.7777, 0.7900]</td><td>0.0666</td></tr><tr><td>Mahalanobis</td><td>0.7804</td><td>[0.7747, 0.7863]</td><td>0.3243</td></tr><tr><td> $\ell _ { 1 }$  Distance</td><td>0.7746</td><td>[0.7683, 0.7809]</td><td>0.0873</td></tr><tr><td>ODIN</td><td>0.7682</td><td>[0.7622, 0.7742]</td><td>0.2185</td></tr><tr><td>Max Disagree-</td><td>0.6722</td><td>[0.6644, 0.6794]</td><td>0.1391</td></tr><tr><td>ment MC-Dropout</td><td>0.5357</td><td>[0.5274, 0.5439]</td><td>0.0726</td></tr><tr><td>(T=30)† TTA Variance</td><td>0.6050</td><td>[0.5969, 0.6129]</td><td>0.2324</td></tr><tr><td>(N=10)† Pairwise Dis-</td><td>0.5877</td><td>[0.5800, 0.5953]</td><td>0.1315</td></tr><tr><td>agreement TTA (5 crops,</td><td>0.5823</td><td>[0.5744, 0.5906]</td><td>0.0414</td></tr><tr><td>patch variance) DDU Score‡</td><td></td><td></td><td></td></tr><tr><td>Pairwise JSD</td><td>0.5000 0.5000</td><td>[0.5000, 0.5000] [0.5000, 0.5000]</td><td>0.1315</td></tr><tr><td>Confidence−¹</td><td>0.3777</td><td>[0.3696, 0.3862]</td><td>0.1910</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>Entropy</td><td>0.1208</td><td>[0.1172, 0.1250]</td><td>0.4917</td></tr></table>

## C.4 ViT-B/16: Full Failure Detection Comparison

Table 13 reports failure detection AUC for ViT-B/16 on the NIH test set $( N = 2 5 , 5 9 6 )$ , with the failure threshold set at the 75th-percentile Brier score (0.5538). Fusion CV again leads (0.8314), with ODIN ranking second (0.8005) a notable reversal from the EficientNet-B4 ordering followed by MUS (0.7497). MC-Dropout remains the weakest method (0.6502), though the margin over MUS narrows considerably relative to the convolutional backbone $( \Delta \mathrm { A U C } = + 0 . 0 9 9 6 , p < 1 0 ^ { - 6 } )$ , suggesting that ViT’s global attention mechanism partially reduces the failure signal exploitable by sampling-based approaches.

ViT crop artifact investigation. ViT-B/16 exhibits compressed MUS dynamic range due to global selfattention. Masking-based crops (non-crop region zeroed to ImageNet mean) improve AUROC marginally over upsampling-based crops (0.7465 vs. 0.7421, $\Delta = + 0 . 0 0 4 5 )$ . All reported ViT results use upsampling for consistency with other architectures.

The results in Table 14 show that the impact of spatial aggregation strategy is strongly architecture-dependent. For convolutional backbones, upsampling consistently outperforms masking, with DenseNet-121 exhibiting a modest drop $( \Delta = - 0 . 0 1 4 1 )$ ) and EficientNet-B4 showing a substantial degradation $( \Delta = - 0 . 2 2 5 1 )$ under masking. This suggests that masking disrupts local feature continuity, which CNN-based models rely on for stable uncertainty estimation. In contrast, ViT-B/16 demonstrates the opposite trend: masking yields a slight improvement in failure detection performance $( \Delta = + 0 . 0 0 4 5 )$ , indicating that transformer-based models may better tolerate or even benefit from structured occlusion due to their global attention mechanism. Overall, these results highlight that the choice between upsampling and masking is not universally optimal and should be matched to the inductive biases of the underlying architecture.

## C.5 BioViL-T: Generative VLM Failure Regime

Table 15 reports SpatialUQ applied to BioViL-T, a generative vision-language model whose image encoder produces globally pooled cross-modal representations designed for image-text alignment rather than spatial discrimination. MUS collapses to 0.533 AUC (Spearman $\rho { = } { - } 0 . 1 8 7 )$ , correctly flagged by the $\rho \lesssim 0 . 1 5$ diagnostic as a known failure regime. This result is consistent with the mechanism described in Section 6: spatial inconsistency requires architectures with spatially grounded feature extraction; globally pooled representations eliminate the crop-level inconsistency gradient MUS relies upon. Fusion recovers to 0.838 AUC driven by entropy and confidence signals, demonstrating graceful degradation when MUS is uninformative.

Architectural interpretation. BioViL-T’s image encoder is trained via contrastive alignment with radiology reports at the global image level, producing representations that are semantically rich but spatially uniform across crops. Unlike contrastive models trained on image patches (CLIP, BiomedCLIP), BioViL-T does not develop spatially discriminative crop-level features, removing the inconsistency signal MUS requires. This finding clarifies the architectural precondition for SpatialUQ: the model must produce spatially varying representations across sub-regions, a property present in CNNs and patch-based contrastive ViTs but absent in globally pooled generative VLMs. Practitioners can verify this precondition in under one minute using the $\rho$ diagnostic on a small unlabeled sample before committing to MUS-based deployment.

Table 13: NIH ChestX-ray14 failure detection ViT-B/16, 14-method comparison $( N = 2 5 , 5 9 6 )$ . Brier threshold = 0.5538 (75th percentile, 25% failure rate). SCE = Score Calibration Error (ECE applied to min-max normalised scores, 15 bins). † requires >6 forward passes or training-time modification. ‡ requires access to training-set features. DDU, Pairwise JSD and TTA Entropy are degenerate (constant scores) on this backbone.
<table><tr><td>Method</td><td>AUC</td><td>95% CI</td><td>SCE</td></tr><tr><td>Fusion CV (MUS+Ent+</td><td>0.8314</td><td>[0.8265, 0.8363]</td><td>0.2662</td></tr><tr><td>Conf+l1)</td><td></td><td></td><td></td></tr><tr><td>ODIN</td><td>0.8003</td><td>[0.7945, 0.8062]</td><td>0.4059</td></tr><tr><td>MUS (Ours)</td><td>0.7497</td><td>[0.7435, 0.7563]</td><td>0.0455</td></tr><tr><td> $\ell _ { 1 }$  Distance</td><td>0.7324</td><td>[0.7255, 0.7390]</td><td>0.1412</td></tr><tr><td>Mahalanobis</td><td>0.7282</td><td>[0.7223, 0.7344]</td><td>0.2030</td></tr><tr><td>Max Disagree- ment</td><td>0.7084</td><td>[0.7019, 0.7154]</td><td>0.1545</td></tr><tr><td>TTA Variance (N=10)†</td><td>0.6212</td><td>[0.6139, 0.6287]</td><td>0.0690</td></tr><tr><td>Pairwise Dis- agreement</td><td>0.6203</td><td>[0.6126, 0.6282]</td><td>0.2070</td></tr><tr><td>TTA (5 crops, patch variance)</td><td>0.6089</td><td>[0.6012, 0.6167]</td><td>0.0871</td></tr><tr><td>MC-Dropout (T=30)†</td><td>0.6502</td><td>[0.6439, 0.6569]</td><td>0.1429</td></tr><tr><td>DDU Ścore</td><td>0.5000</td><td>[0.5000, 0.5000]</td><td></td></tr><tr><td>Pairwise JSD</td><td>0.5000</td><td>[0.5000, 0.5000]</td><td>0.2070</td></tr><tr><td>Confidence−¹</td><td>0.4215</td><td>[0.4133, 0.4297]</td><td>0.1881</td></tr><tr><td>Entropy</td><td>0.1695</td><td>[0.1645, 0.1745]</td><td>0.4775</td></tr></table>

Table 14: Comparison of failure detection AUC using upsampling versus masking strategies across diferent architectures (separate ablation run; see Appendix C.4).
<table><tr><td>Architecture</td><td>Upsample AUC</td><td>Mask AUC ∆</td><td></td></tr><tr><td>DenseNet-121</td><td>0.7837</td><td>0.7697</td><td>-0.0141</td></tr><tr><td>EfficientNet-B4</td><td>0.7791</td><td>0.5540</td><td>-0.2251</td></tr><tr><td>ViT-B/16</td><td>0.7421</td><td>0.7465</td><td>+0.0045</td></tr></table>

Table 15: SpatialUQ on BioViL-T (NIH ChestX-ray14, $N { = } 2 5 , 5 9 6 .$ , failure: Brier $> 7 5 \mathrm { t h }$ percentile). Near-chance MUS performance is correctly flagged by the $\rho \lesssim 0 . 1 5$ diagnostic $( \rho { = } { - } 0 . 1 8 7 )$
<table><tr><td>Method</td><td>AUC</td><td>95% CI</td><td>SCE</td><td>Spearman  $\rho$ </td></tr><tr><td>Fusion</td><td>0.838</td><td>[0.832, 0.843]</td><td>0.069</td><td></td></tr><tr><td>MUS (SpatialUQ)</td><td>0.533</td><td>[0.526, 0.541]</td><td>0.202</td><td>-0.187</td></tr><tr><td> $\ell _ { 1 }$  Distance</td><td>0.533</td><td>[0.525, 0.541]</td><td>0.120</td><td></td></tr><tr><td>Entropy</td><td>0.242</td><td>[0.236, 0.248]</td><td>0.723</td><td></td></tr><tr><td> $\mathrm { C o n f i d e n c e ^ { - 1 } }$ </td><td>0.166</td><td>[0.160, 0.171]</td><td>0.572</td><td></td></tr></table>

## D PER-CLASS JSD AUC (ALL ARCHITECTURES AND DATASETS)

## D.1 NIH ChestX-ray14 - DenseNet-121

Table 16 reports per-class failure detection AUC using the class-specific JSD score for DenseNet-121, with per-class failure defined as Brier score exceeding the 75th percentile of each class’s marginal distribution. JSD AUC varies markedly across pathologies, from 0.9789 (Pneumonia) to 0.3680 (Nodule), revealing a strong dependence on spatial extent: difuse findings such as Pneumonia, Edema, and Consolidation consistently achieve JSD AUC ≥ 0.95, whereas small focal lesions Nodule (0.368), Fibrosis (0.553), and Mass (0.558) fall well below the aggregate performance. This pattern confirms that MUS exploits spatial prediction consistency most efectively when pathological regions occupy large, contiguous image areas, and degrades systematically as lesion size decreases.

Table 16: Per-class JSD AUC for failure detection DenseNet-121 on NIH ChestX-ray14. Classes sorted by JSD AUC (descending). MUS excels for difuse, globally visible pathologies and degrades for small focal lesions.
<table><tr><td>Class</td><td>Backbone AUC</td><td>JSD AUC</td><td>N Pos</td></tr><tr><td>Pneumonia</td><td>0.7020</td><td>0.9789</td><td>555</td></tr><tr><td>Edema</td><td>0.8368</td><td>0.9710</td><td>925</td></tr><tr><td>Consolidation</td><td>0.7401</td><td>0.9564</td><td>1,815</td></tr><tr><td>Pleural_Thickening</td><td>0.7569</td><td>0.9496</td><td>1,143</td></tr><tr><td>Cardiomegaly</td><td>0.8746</td><td>0.9518</td><td>1,069</td></tr><tr><td>Atelectasis</td><td>0.7628</td><td>0.9145</td><td>3,279</td></tr><tr><td>Pneumothorax</td><td>0.8471</td><td>0.8943</td><td>2,665</td></tr><tr><td>Hernia</td><td>0.7733</td><td>0.8837</td><td>86</td></tr><tr><td>Infiltration</td><td>0.6984</td><td>0.8649</td><td>6,112</td></tr><tr><td>Effusion</td><td>0.8192</td><td>0.8641</td><td>4,658</td></tr><tr><td>Emphysema</td><td>0.8822</td><td>0.6697</td><td>1,093</td></tr><tr><td>Mass</td><td>0.7965</td><td>0.5580</td><td>1,748</td></tr><tr><td>Fibrosis</td><td>0.8095</td><td>0.5528</td><td>435</td></tr><tr><td>Nodule</td><td>0.7400</td><td>0.3680</td><td>1,623</td></tr></table>

## D.2 NIH ChestX-ray14- EficientNet-B4

Table 17 reports per-class JSD AUC for EficientNet-B4 on NIH ChestX-ray14. In contrast to DenseNet-121, all 14 classes fall within a narrow band of 0.75-0.89, indicating uniformly reliable failure detection across pathology types. Most strikingly, Nodule JSD AUC rises to 0.8822 compared to 0.3680 for DenseNet-121, an architecture-dependent gap consistent with EficientNet’s compound scaling allocating greater capacity to fine-grained spatial features. The pronounced class-level disparity observed under DenseNet-121 particularly the collapse for small focal lesions is thus largely absent, suggesting that backbone inductive bias is a primary determinant of MUS’s per-class failure detection profile.

Table 17: Per-class JSD AUC - EficientNet-B4 on NIH. Notable: Nodule AUC = 0.8822 vs. 0.3680 for DenseNet 121.
<table><tr><td>Class</td><td>Backbone AUC</td><td>JSD AUC</td></tr><tr><td>Pneumonia</td><td>0.6589</td><td>0.89</td></tr><tr><td>Edema</td><td>0.8093</td><td>0.88</td></tr><tr><td>Nodule</td><td>0.6941</td><td>0.8822</td></tr><tr><td>Consolidation</td><td>0.7022</td><td>0.87</td></tr><tr><td>Pleural_Thickening</td><td>0.7201</td><td>0.86</td></tr><tr><td>Cardiomegaly</td><td>0.7952</td><td>0.86</td></tr><tr><td>Atelectasis</td><td>0.7192</td><td>0.85</td></tr><tr><td>Pneumothorax</td><td>0.8050</td><td>0.84</td></tr><tr><td>Effusion</td><td>0.7740</td><td>0.83</td></tr><tr><td>Infiltration</td><td>0.6790</td><td>0.82</td></tr><tr><td>Emphysema</td><td>0.7977</td><td>0.80</td></tr><tr><td>Mass</td><td>0.7002</td><td>0.79</td></tr><tr><td>Fibrosis</td><td>0.7547</td><td>0.77</td></tr><tr><td>Hernia</td><td>0.6763</td><td>0.75</td></tr></table>

## D.3 NIH ChestX-ray14 - ViT-B/16

Table 18 reports per-class JSD AUC for ViT-B/16 on NIH ChestX-ray14. The architecture reproduces the spatialextent dependence observed in DenseNet-121, with difuse pathologies such as Pneumonia (0.88) and Edema (0.87) retaining strong failure detection, while small focal lesions collapse sharply Nodule (0.3037), Mass (0.4182), and Emphysema (0.4024). We attribute this degradation to ViT’s global self-attention, which produces spatially coherent crop-level predictions regardless of lesion localisation, thereby compressing the inconsistency signal that

MUS relies upon. Notably, even Pneumothorax (0.74) and Hernia (0.65) underperform their EficientNet-B4 counterparts, reinforcing that attention-based inductive bias systematically limits MUS’s discriminative range for spatially confined findings.

Table 18: Per-class JSD AUC - ViT-B/16 on NIH. ViT shows substantially worse per-class performance for small focal lesions compared to DenseNet-121 and EficientNet-B4.
<table><tr><td>Class</td><td>Backbone AUC</td><td>JSD AUC</td></tr><tr><td>Pneumonia</td><td>0.7111</td><td>0.88</td></tr><tr><td>Edema</td><td>0.8359</td><td>0.87</td></tr><tr><td>Cardiomegaly</td><td>0.8868</td><td>0.85</td></tr><tr><td>Consolidation</td><td>0.7521</td><td>0.84</td></tr><tr><td>Pleural_Thickening</td><td>0.7622</td><td>0.83</td></tr><tr><td>Atelectasis</td><td>0.7689</td><td>0.81</td></tr><tr><td>Effusion</td><td>0.8257</td><td>0.79</td></tr><tr><td>Infiltration</td><td>0.6978</td><td>0.77</td></tr><tr><td>Pneumothorax</td><td>0.8587</td><td>0.74</td></tr><tr><td>Hernia</td><td>0.7446</td><td>0.65</td></tr><tr><td>Fibrosis</td><td>0.7938</td><td>0.53</td></tr><tr><td>Emphysema</td><td>0.8854</td><td>0.4024</td></tr><tr><td>Mass</td><td>0.8157</td><td>0.4182</td></tr><tr><td>Nodule</td><td>0.7502</td><td>0.3037</td></tr></table>

## D.4 CheXpert Zero-Shot - Per-Class JSD AUC

Table 19 reports per-class JSD AUC for zero-shot transfer of the NIH-trained DenseNet-121 to CheXpert (N=44,399, 10 mapped classes). MUS generalises robustly for difuse pathologies, with eight of ten classes exceeding 0.71, led by Pneumonia (0.9678) and Pleural Thickening (0.9675). Nodule, however, degrades severely to 0.1446 well below its in-distribution JSD AUC indicating that domain shift disproportionately erodes the spatial consistency signal for small focal lesions, compounding the architectural limitations observed on NIH.

Table 19: Per-class JSD AUC - zero-shot transfer to CheXpert (NIH-trained DenseNet-121, N = 44,399, 10 mapped classes). Nodule degrades severely (0.1446) under domain shift.

<table><tr><td>Class</td><td>JSD AUC</td></tr><tr><td>Pneumonia</td><td>0.9678</td></tr><tr><td>Pleural_Thickening</td><td>0.9675</td></tr><tr><td>Consolidation</td><td>0.9589</td></tr><tr><td>Atelectasis</td><td>0.9218</td></tr><tr><td>Cardiomegaly</td><td>0.9216</td></tr><tr><td>Pneumothorax</td><td>0.9149</td></tr><tr><td>Edema</td><td>0.8633</td></tr><tr><td>Infiltration</td><td></td></tr><tr><td></td><td>0.7190</td></tr><tr><td>Effusion</td><td>0.6709</td></tr><tr><td>Nodule</td><td>0.1446</td></tr></table>

## D.5 VinBigData Zero-Shot Per-Class JSD AUC

Table 20 reports per-class JSD AUC for zero-shot transfer to VinBigData (N=15,000, 10 mapped classes). Performance bifurcates sharply: Atelectasis (0.9467), Infiltration (0.9222), and Pneumothorax (0.9181) retain strong failure detection, while Consolidation (0.3375), Mass (0.1644), and Nodule (0.1608) collapse near chance. Relative to CheXpert transfer, the degradation is both more severe and more broadly distributed across classes, suggesting that extreme domain shift renders model failures spatially uniform eliminating the crop-level inconsistency signal that MUS relies upon rather than selectively impairing small focal lesions as observed under moderate shift.

## E ABLATION STUDIES

## E.1 Sensitivity to Crop Placement (Random vs. Fixed)

To evaluate whether the discriminative power of MUS is dependent on specific geometric coordinates, we conducted a control experiment substituting the fixed five-crop set with 1,000 randomly sampled 112 × 112 crops. This configuration yielded a failure detection AUC of 0.7389, compared to 0.784 for the standard fixed-crop MUS. While the fixed geometric arrangement provides a superior and more eficient signal, the continued eficacy of random placements confirms that the principle of spatial inconsistency is robust to crop positions and reflects a fundamental model characteristic rather than a sensitive hyperparameter.

Table 20: Per-class JSD AUC - zero-shot transfer to VinBigData (NIH-trained DenseNet-121, $N = 1 5 , 0 0 0 .$ , 10 mapped classes). Severe domain shift collapses JSD AUC for common pathologies (Consolidation: 0.3375; Mass: 0.1644; Nodule: 0.1608).
<table><tr><td>Class</td><td>JSD AUC</td></tr><tr><td>Atelectasis</td><td>0.9467</td></tr><tr><td>Infiltration</td><td>0.9222</td></tr><tr><td>Pneumothorax</td><td>0.9181</td></tr><tr><td>Cardiomegaly</td><td>0.8845</td></tr><tr><td>Pleural_Thickening</td><td>0.6678</td></tr><tr><td>Effusion</td><td>0.4229</td></tr><tr><td>Fibrosis</td><td>0.3490</td></tr><tr><td>Consolidation</td><td>0.3375</td></tr><tr><td>Mass</td><td>0.1644</td></tr><tr><td>Nodule</td><td>0.1608</td></tr></table>

## E.2 Crop Count Ablation

Table 21 reports MUS failure detection AUC on NIH DenseNet-121 as a function of the number of spatial crops. Five crops achieves the best cost-eficiency trade-of: nine crops yields a marginal +0.007 AUC improvement (0.7902 vs. 0.7837) at 67% more inference cost, while three crops reduces AUC by 0.007 (0.7764 vs. 0.7837). The intentional overlap between the center crop and all four quadrants provides double coverage of the image center a region of heightened diagnostic importance in frontal chest radiography without any additional forward passes. The apparent reversal at $\mathrm { K } { = } 4 \ \mathrm { ( A U C } = 0 . 7 8 5 9 > \mathrm { K } { = } 5 \ \mathrm { A U C } = 0 . 7 8 3 7 )$ is not statistically meaningful: the diference of 0.0022 lies well within the 95% bootstrap CI width of ±0.006 for MUS on this dataset (bootstrap p > 0.3). This is a sampling artifact rather than a structural advantage of the four-crop arrangement.

Table 21: Crop count ablation MUS AUC on NIH DenseNet-121 (failure threshold = 0.5607). Inference cost scales linearly with crop count plus one global pass.
<table><tr><td>Crops</td><td>MUS AUC</td><td>Forward Passes</td></tr><tr><td>3</td><td>0.7764</td><td>4</td></tr><tr><td>4</td><td>0.7859</td><td>5</td></tr><tr><td>5</td><td>0.7837</td><td>6</td></tr><tr><td>6</td><td>0.7801</td><td>7</td></tr><tr><td>7</td><td>0.7816</td><td>8</td></tr><tr><td>9</td><td>0.7902</td><td>10</td></tr></table>

The non-monotonic AUC curve in Table 21 (dipping at K=6 before recovering at K=7 and K=9) reflects a geometric property of fixed crop placement. At K=5, the four quadrant crops and one center crop provide non-redundant spatial coverage with overlap only at the center region. Adding a sixth crop in a fixed scheme necessarily introduces either partial redundancy with an existing crop or an extreme peripheral crop that covers mainly background pixels in standard chest X-ray framing. The recovery at K=7 and K=9 occurs when enough additional crops are added to restore meaningful spatial diversity. This behavior motivates adaptive or saliency guided crop placement as a future direction, which could potentially recover K=9 performance at lower inference cost.

Beyond AUC, as we can see from Figure 7, the five crop design is geometrically motivated. The four-quadrant arrangement alone leaves the central 112×112 pixel region without a dedicated crop each quadrant contributes only a partial corner overlap. In frontal chest radiography, this central region contains diagnostically critical structures including the cardiac silhouette, mediastinum, and central airways. The fifth center crop at coordinates (56, 56, 168, 168) provides direct dedicated coverage of this region, making the five-crop set the minimum spatially complete decomposition of a 224×224 image. K=4 cannot be considered spatially complete regardless of its AUC on any single dataset.

Figure 8 (left) confirms K=5 as the optimal cost-eficiency operating point; K=9 yields only +0.007 AUC at 67% additional inference cost. Figure 8 (right) shows that MUS and Fusion CV produce the clearest failure/success separation; entropy is anti-correlated with failure due to systematic overconfidence under BCE training.

(a) K = 4 — Four Quadrant Crops Only  
(b) K = 5 — Four Quadrants + Center Crop  
![](images/b96a1c1f1f31aee87c8d38ac360d1d3436442d8261d640706518d9379c0e3e98.jpg)

Figure 7: Geometric coverage of fixed spatial crops. (a) K = 4 (quadrants only): four non-overlapping quadrant crops leave the central 112×112 px region without a dedicated crop. (b) K = 5 (quadrants + center): adding a fixed center crop provides full coverage of the diagnostically critical central region, with partial overlap across quadrants. Despite nearly identical performance (AUC of 0.7859 vs. 0.7837; ∆ = 0.0022, within bootstrap CI), K = 5 is preferred for its strictly stronger geometric coverage guarantees  
![](images/b420b04cf8632ea26ddb1fe8cb5d4b55f9b1f6f99363db1f9d5362616fa0fffc.jpg)

![](images/ac0214eeb9262f9c81c322bbc39b529c6e09afeed8d2b1af4d1b15788363c0f0.jpg)  
Figure 8: Ablation studies on NIH DenseNet-121. Left: Failure detection AUC as a function of the number of spatial crops K. Five crops achieves the best cost-eficiency trade-of; nine crops yields a marginal +0.007 AUC gain at 67% more inference cost, while three crops reduces AUC by 0.007. Right: Normalised uncertainty score distributions stratified by prediction outcome (failure vs. success) for six methods. SpatialUQ (MUS) and Fusion CV show the clearest separation; Entropy is anti-correlated with failure (AUC = 0.177) under the overconfident multi-label training regime.

## E.3 Failure Threshold Robustness

Table 22 evaluates failure detection performance under increasingly stringent definitions of failure, revealing clear stratification across uncertainty estimation methods. As expected, naive baselines such as Random remain at chance level, while confidence-based proxies (Confidence<sup>−1</sup>, Entropy) perform poorly and degrade further at higher thresholds, indicating limited alignment with true model errors. Classical disagreement-based approaches (Patch Variance, Pairwise Disagreement, JSD) provide moderate gains and show slight improvement as the failure definition becomes stricter, suggesting some sensitivity to severe errors. Among single-model uncertainty methods, $\ell _ { 1 }$ Distance and ODIN demonstrate strong performance, particularly at higher thresholds, while TTA-based and MC-Dropout methods degrade, reflecting instability under stricter failure criteria. Ensemble-based approaches remain among the strongest standalone baselines, with Deep Ensemble achieving consistently high AUC and improving substantially at the 90th percentile. MUS (ours) delivers competitive performance across all thresholds, outperforming most single-model and stochastic methods while remaining close to ensemble-level performance. Notably, augmenting MUS with lightweight auditor models yields further gains, and the Fusion (CV) approach achieves the best overall performance across all thresholds, reaching an AUC of 0.8712 at the 90th percentile. These results indicate that MUS provides a robust and scalable uncertainty signal, and that combining complementary features can further enhance failure detection, especially under stringent error definitions.

Table 22: Failure detection AUC across diferent Brier score failure thresholds on NIH ChestX-ray14. Thresholds correspond to the 50<sup>th</sup> (0.5170), 75<sup>th</sup> (0.5607), and $9 0 ^ { \mathrm { t h } }$ (0.5993) percentiles of the test set Brier scores.
<table><tr><td>Method</td><td>AUC@50th</td><td>AUC@75th</td><td>AUC@90th</td></tr><tr><td>Random</td><td>0.4985</td><td>0.4979</td><td>0.4960</td></tr><tr><td>Confidence−¹</td><td>0.4079</td><td>0.4011</td><td>0.3886</td></tr><tr><td>Entropy</td><td>0.1936</td><td>0.1774</td><td>0.1319</td></tr><tr><td>Patch Variance</td><td>0.6045</td><td>0.6103</td><td>0.6236</td></tr><tr><td>Pairwise Disagr.</td><td>0.5910</td><td>0.5990</td><td>0.6124</td></tr><tr><td>Pairwise JSD</td><td>0.5997</td><td>0.6057</td><td>0.6184</td></tr><tr><td>TTA Variance</td><td>0.5994</td><td>0.5837</td><td>0.5789</td></tr><tr><td>TTA Entropy</td><td>0.2182</td><td>0.2092</td><td>0.1729</td></tr><tr><td>l1 Distance</td><td>0.7747</td><td>0.7902</td><td>0.8282</td></tr><tr><td>DDU Score</td><td>0.7627</td><td>0.7540</td><td>0.7634</td></tr><tr><td>Mahalanobis</td><td>0.7627</td><td>0.7540</td><td>0.7634</td></tr><tr><td>ODIN</td><td>0.7730</td><td>0.7588</td><td>0.7735</td></tr><tr><td>MC-Dropout</td><td>0.7001</td><td>0.6643</td><td>0.6461</td></tr><tr><td>Deep Ensemble (5-mem)</td><td>0.7979</td><td>0.8130</td><td>0.8562</td></tr><tr><td>MUS (Ours)</td><td>0.7698</td><td>0.7837</td><td>0.8226</td></tr><tr><td>Auditor (3-feat)</td><td>0.8124</td><td>0.8250</td><td>0.8681</td></tr><tr><td>Auditor (6-feat)</td><td>0.8110</td><td>0.8242</td><td>0.8683</td></tr><tr><td>Fusion (ČV)</td><td>0.8170</td><td>0.8319</td><td>0.8712</td></tr></table>

The monotonic increase of MUS AUC with Brier threshold stringency (0.770 → 0.784 → 0.823 across the 50th, 75th, and 90th percentiles) has a natural interpretation: spatial inconsistency is a stronger signal for the most severely incorrect predictions, where the model has entirely failed to localize discriminative features. At the 50th percentile threshold, half of all predictions are defined as failures, including many borderline cases where the model’s prediction is only marginally incorrect and spatial inconsistency may be modest. At the 90th percentile, only the 10% most severely incorrect predictions are defined as failures cases, where the model is confidently wrong, which are precisely the cases where spatial incoherence is most pronounced. This monotonicity is preserved on CheXpert $( 0 . 6 4 7  0 . 7 0 8  0 . 7 5 7 )$ , confirming that the 75th-percentile operating point used throughout the paper is neither optimistically nor pessimistically chosen.

## E.4 Leave-One-Out Fusion Ablation

Table 23 reports a leave-one-out ablation of the four-feature fusion for NIH DenseNet-121, where each feature is removed in turn from the full fusion (AUC of = 0.8319) to quantify its marginal contribution. Entropy emerges as the dominant contributor by a substantial margin: its removal degrades AUC by 0.0437, reducing the fusion to 0.7881 approaching the standalone performance of individual uncertainty estimators. $\ell _ { 1 }$ Distance and MUS each provide modest but non-redundant complementary signal (−0.0071 and −0.0053, respectively), confirming that spatial inconsistency and distributional divergence capture failure modes not fully explained by predictive entropy alone. Confidence contributes negligibly (−0.0010), suggesting that its information is largely subsumed by entropy within the fusion. Collectively, these results indicate that fusion performance is primarily driven by entropy, with MUS and $\ell _ { 1 }$ Distance providing independent secondary contributions that collectively account for the gap between entropy-only and full-fusion performance.

Table 23: Leave-one-out fusion ablation - NIH DenseNet-121. Full 4-feature fusion AUC = 0.8319.
<table><tr><td>Removed Feature</td><td>Fusion AUC</td><td> $\pmb { \Delta }$ </td></tr><tr><td>Entropy</td><td>0.7881</td><td>-0.0437</td></tr><tr><td> $\ell _ { 1 }$  Distance</td><td>0.8248</td><td>-0.0071</td></tr><tr><td>MUS</td><td>0.8265</td><td>-0.0053</td></tr><tr><td>Confidence</td><td>0.8309</td><td>-0.0010</td></tr></table>

Table 24 reports the leave-one-out fusion ablation for ViT-B/16 (full fusion AUC = 0.8314). Entropy dominates even more strongly than in DenseNet-121, with its removal incurring a drop of −0.0780 AUC nearly twice the corresponding penalty observed for the convolutional backbone. $\ell _ { 1 }$ Distance and MUS contribute marginally (−0.0022 and −0.0015, respectively), while Confidence proves slightly redundant, with its removal yielding a negligible improvement of +0.0014, suggesting its information is fully subsumed by entropy under the global attention regime. This pattern is consistent with ViT’s compressed spatial inconsistency signal, which diminishes the independent contribution of MUS and shifts the fusion’s discriminative burden almost entirely onto entropybased uncertainty.

Table 24: Leave-one-out fusion ablation - NIH ViT-B/16. Full 4-feature fusion AUC = 0.8314.
<table><tr><td>Removed Feature</td><td>Fusion AUC</td><td> $\pmb { \Delta }$ </td></tr><tr><td>Entropy</td><td>0.7534</td><td>-0.0780</td></tr><tr><td> $\ell _ { 1 }$  Distance</td><td>0.8292</td><td>-0.0022</td></tr><tr><td>MUS</td><td>0.8299</td><td>-0.0015</td></tr><tr><td>Confidence</td><td>0.8328</td><td>+0.0014</td></tr></table>

Table 25 reports leave-one-out fusion ablations across three ImageNet-1k architectures, revealing a consistent reversal relative to the NIH chest X-ray setting: confidence is the dominant fusion signal in all three models, with removal penalties of −0.0934 (ConvNeXt-Tiny), −0.0407 (ViT-B/16), and −0.0168 (EficientNet-B4). This dominance is consistent with the well-calibrated, single-label nature of ImageNet classification, where predictive confidence alone carries strong failure-discriminative information. Entropy contributes modestly across architectures, while MUS provides a small but consistent complementary signal (−0.0035 to −0.0056), with the exception of EficientNet-B4 where entropy removal incurs no penalty (±0.0000), suggesting near-complete redundancy with confidence in that setting.

Table 25: Leave-one-out fusion ablation ImageNet-1k. Confidence dominates on single-label well-calibrated classifiers; MUS provides complementary but secondary signal.
<table><tr><td>Model</td><td>Removed</td><td>LOO AUC</td><td> $\pmb { \Delta }$ </td></tr><tr><td>ConvNeXt-Tiny</td><td>Confidence</td><td>0.8422</td><td>-0.0934</td></tr><tr><td>ConvNeXt-Tiny</td><td>Entropy</td><td>0.9217</td><td>-0.0139</td></tr><tr><td>ConvNeXt-Tiny</td><td>MUS</td><td>0.9322</td><td>-0.0035</td></tr><tr><td>ViT-B/16</td><td>Confidence</td><td>0.8962</td><td>-0.0407</td></tr><tr><td>ViT-B/16</td><td>Entropy</td><td>0.9317</td><td>-0.0051</td></tr><tr><td>ViT-B/16</td><td>MUS</td><td>0.9324</td><td>-0.0045</td></tr><tr><td>EfficientNet-B4</td><td>Confidence</td><td>0.9005</td><td>-0.0168</td></tr><tr><td>EfficientNet-B4</td><td>Entropy</td><td>0.9174</td><td>±0.0000</td></tr><tr><td>EfficientNet-B4</td><td>MUS</td><td>0.9117</td><td>-0.0056</td></tr></table>

## E.5 Permutation Feature Importance (Learned Auditor)

Permutation feature importance for the 3-feature learned auditor (MLP) on NIH DenseNet-121 (baseline test AUC = 0.8250):

• Entropy: AUC drop = 0.3516 (dominant signal)

• Confidence: AUC drop = 0.0078

• MUS: AUC drop = 0.0035

These magnitudes confirm the LOO ablation finding: entropy is the primary discriminator in the learned auditor context, while MUS and confidence provide secondary, independent contributions.

## F ZERO-SHOT TRANSFER: FULL METRIC TABLES

## F.1 CheXpert Zero-Shot: Complete Evaluation

Table 26 reports failure detection AUC and full secondary metrics for all methods on CheXpert. The NIH-trained DenseNet-121 is applied without any retraining or calibration. Failure threshold = 0.5130 (Brier 75th percentile, 25% failure rate).

Table 26: Zero-shot failure detection on CheXpert (NIH-trained DenseNet-121, N = 44,399, failure threshold = 0.5130). All metrics computed on the full test set without any retraining. AURC and E-AURC lower is better.
<table><tr><td>Method</td><td>AUC</td><td>ECE</td><td>AUPR</td><td>FPR@80%</td><td>FPR@95%</td><td>AURC</td><td>E-AURC</td></tr><tr><td>Fusion CV</td><td>0.7330</td><td>0.0601</td><td>0.4248</td><td>0.4581</td><td>0.6279</td><td>0.11467</td><td>0.08343</td></tr><tr><td>l1 Distance</td><td>0.7102</td><td>0.1372</td><td>0.4048</td><td>0.4889</td><td>0.6976</td><td>0.12686</td><td>0.09562</td></tr><tr><td>MUS (Ours)</td><td>0.7076</td><td>0.0446</td><td>0.4023</td><td>0.4967</td><td>0.6956</td><td>0.12739</td><td>0.09614</td></tr><tr><td>TTA Variance</td><td>0.6004</td><td>0.0353</td><td>0.3206</td><td>0.6773</td><td>0.8854</td><td>0.18897</td><td>0.15772</td></tr><tr><td>MC-Dropout</td><td>0.5991</td><td>0.1275</td><td>0.3004</td><td>0.6611</td><td>0.8691</td><td>0.18347</td><td>0.15223</td></tr><tr><td>Patch Variance</td><td>0.5634</td><td>0.0499</td><td>0.2883</td><td>0.7187</td><td>0.9174</td><td>0.21090</td><td>0.17965</td></tr><tr><td>Confidence -1</td><td>0.3473</td><td>0.3279</td><td>0.1810</td><td>0.8520</td><td>0.9547</td><td>0.31020</td><td>0.27895</td></tr><tr><td>Entropy</td><td>0.2656</td><td>0.4257</td><td>0.1651</td><td>0.9263</td><td>0.9848</td><td>0.37997</td><td>0.34873</td></tr></table>

Bootstrap 95% CI $( N = 1 , 0 0 0$ resamples): MUS: 0.7076 [0.7026, 0.7124]. Fusion: 0.7330 [0.7282, 0.7373]. Spearman ρ (MUS vs. Brier) = 0.2983, $p < 1 0 ^ { - 6 }$ (reduced from 0.5225 on NIH, reflecting moderate domain shift while retaining discriminative power). MUS vs. MC-Dropout: $\Delta \mathrm { A U C } = + 0 . 1 0 8 5 , p < 1 0 ^ { - 6 }$

Clinical utility (CheXpert): Selective prediction at 20% referral improves system AUC from 0.6860 to 0.7478. Relative gap reduction: 19.7%. The drop in Spearman correlation between MUS and Brier score from 0.523 (NIH in-distribution) to 0.298 (CheXpert zero-shot) reflects moderate domain shift: the DenseNet-121 trained on NIH encounters a diferent patient population, scanner characteristics, and label distribution at $\mathrm { C h e X p e r t }$ . Despite this, MUS retains meaningful discriminative power (AUC = 0.708), suggesting that spatial inconsistency is a partially domain-agnostic signal the model’s spatial coherence degrades on genuinely uncertain inputs even under population shift. The ECE of MUS on CheXpert (0.0446) remains low, confirming that calibration is largely preserved under moderate shift. The selective prediction result at 20% referral (system AUC improving from 0.686 to 0.748, a 19.7% relative gap reduction) demonstrates practical clinical utility even without retraining.

## F.2 VinBigData Zero-Shot: Complete Evaluation

Table 27 reports full metrics for VinBigData. This is the identified failure regime for SpatialUQ: severe domain shift causes the model to fail uniformly across global and local spatial views rather than selectively. Spearman ρ (MUS vs. Brier) = 0.0269, $p = 9 . 8 6 \times 1 0 ^ { - 4 }$ near-zero correlation confirming that spatial inconsistency no longer tracks prediction error under extreme domain shift. MUS vs. MC-Dropout: $\Delta \mathrm { A U C } = - 0 . 1 4 9 9 , p < 1 0 ^ { - 6 }$ Bootstrap 95% CI: MUS = 0.6141 [0.6029, 0.6258]; MC-Dropout = 0.7641 [0.7564, 0.7725]. Clinical utility: selective prediction at 20% referral improves AUC from 0.8014 to 0.8084; relative gap reduction only 3.5% (substantially lower than CheXpert’s 19.7%), reflecting the dominance of uniform model error over selective spatial incoherence.

Despite MUS collapsing to near-chance on VinBigData (AUC = 0.614, Spearman $\rho = 0 . 0 2 7 )$ , the linear Fusion model retains 0.750 AUC. A leave-one-out analysis on the VinBigData 10% calibration split confirms that this gain is driven entirely by MC-Dropout (AUC contribution ≈ 0.140) and entropy (contribution ≈ 0.095), with MUS contributing negligibly (≈ 0.002). This confirms that Fusion’s robustness on VinBigData comes from its MC-Dropout component rather than spatial consistency, and that the 0.750 Fusion AUC should not be interpreted as evidence that SpatialUQ transfers to extreme-shift settings. Practitioners deploying in settings with suspected severe domain shift should prefer MC-Dropout or ensemble methods over MUS as the primary uncertainty signal, using MUS as a complementary component only when compute permits.

Table 27: Zero-shot failure detection on VinBigData (NIH-trained DenseNet-121, N = 15,000, failure threshold = 0.4117). MC-Dropout significantly outperforms MUS in this extreme-shift regime.
<table><tr><td>Method</td><td>AUC</td><td>ECE</td><td>AUPR</td><td>FPR@80%</td><td>FPR@95%</td><td>AURC</td><td>E-AURC</td></tr><tr><td>MC-Dropout</td><td>0.7641</td><td>0.1414</td><td>0.4304</td><td>0.3743</td><td>0.5991</td><td>0.10452</td><td>0.07327</td></tr><tr><td>Fusion CV</td><td>0.7502</td><td>0.1161</td><td>0.3931</td><td>0.3631</td><td>0.5938</td><td>0.10707</td><td>0.07582</td></tr><tr><td>TTA Variance</td><td>0.6127</td><td>0.0771</td><td>0.3057</td><td>0.6196</td><td>0.8381</td><td>0.17268</td><td>0.14143</td></tr><tr><td>MUS (Ours)</td><td>0.6141</td><td>0.1254</td><td>0.3589</td><td>0.7343</td><td>0.9479</td><td>0.20936</td><td>0.17811</td></tr><tr><td>l1 Distance</td><td>0.5827</td><td>0.0712</td><td>0.3426</td><td>0.7764</td><td>0.9530</td><td>0.22480</td><td>0.19355</td></tr><tr><td>Patch Variance</td><td>0.5003</td><td>0.0925</td><td>0.2514</td><td>0.8006</td><td>0.9560</td><td>0.25270</td><td>0.22145</td></tr></table>

## G FULL METRIC TABLES: IMAGENET AND MS COCO

## G.1 ImageNet-1k: Full Metric Breakdown

Table 28 reports full secondary metrics for failure detection on ImageNet-1k across three architectures. Confidence inv is the dominant standalone signal, achieving AUC of 0.9132, 0.9233, and 0.9131 for EficientNet-B4, ViT-B/16, and ConvNeXt-Tiny respectively, consistent with the ablation findings. Fusion CV recovers near-ceiling performance across all architectures (0.9173, 0.9369, 0.9356), with notably lower FPR@80%TPR (0.1237, 0.0872, 0.0700). MUS as a standalone method performs substantially weaker across all architectures (AUC 0.6415-0.7171), with Spearman correlations against Brier score declining from 0.3918 (EficientNet-B4) to 0.1391 (ConvNeXt-Tiny), mirroring the degradation in standalone AUC and confirming that spatial inconsistency provides limited discriminative signal for well-calibrated single-label classifiers where confidence already captures the dominant failure mode.

Table 28: ImageNet-1k failure detection - full secondary metrics (50,000 images, failure = top-1 misclassification). Confidence-inv dominates on all architectures as a standalone signal; fusion recovers near-ceiling performance.
<table><tr><td>Model</td><td>Method</td><td>AUC</td><td>AUPR</td><td>FPR@80%</td><td>FPR@95%</td><td>AURC</td></tr><tr><td>EfficientNet-B4</td><td>MUS (Ours)</td><td>0.7171</td><td>0.4491</td><td>0.5062</td><td>0.7670</td><td>0.3890</td></tr><tr><td>EfficientNet-B4</td><td>Entropy</td><td>0.8961</td><td>0.7785</td><td>0.1468</td><td>0.4742</td><td>0.5252</td></tr><tr><td>EfficientNet-B4</td><td>Confidence-inv</td><td>0.9132</td><td>0.8424</td><td>0.1389</td><td>0.4398</td><td>0.5442</td></tr><tr><td>EfficientNet-B4</td><td>Fusion CV</td><td>0.9173</td><td>0.8496</td><td>0.1237</td><td></td><td></td></tr><tr><td>ViT-B/16</td><td>MUS (Ours)</td><td>0.7095</td><td>0.4066</td><td>0.4945</td><td>0.7555</td><td>0.3686</td></tr><tr><td>ViT-B/16</td><td>Entropy</td><td>0.8840</td><td>0.7856</td><td>0.1691</td><td>0.5967</td><td>0.5247</td></tr><tr><td>ViT-B/16</td><td>Confidence-inv</td><td>0.9233</td><td>0.8781</td><td>0.0973</td><td>0.4607</td><td>0.5546</td></tr><tr><td>ViT-B/16</td><td>Fusion CV</td><td>0.9369</td><td>0.8894</td><td>0.0872</td><td></td><td></td></tr><tr><td>ConvNeXt-Tiny</td><td>MUS (Ours)</td><td>0.6415</td><td>0.3143</td><td>0.5655</td><td>0.8059</td><td>0.3048</td></tr><tr><td>ConvNeXt-Tiny</td><td>Entropy</td><td>0.8194</td><td>0.6632</td><td>0.3306</td><td>0.6926</td><td>0.4787</td></tr><tr><td>ConvNeXt-Tiny</td><td>Confidence-inv</td><td>0.9131</td><td>0.8701</td><td>0.1081</td><td>0.5654</td><td>0.5506</td></tr><tr><td>ConvNeXt-Tiny</td><td>Fusion CV</td><td>0.9356</td><td>0.8912</td><td>0.0700</td><td></td><td></td></tr></table>

## G.2 MS COCO 2014: Full Metric Breakdown and Score Threshold Sensitivity

Table 29 reports failure detection metrics for Faster R-CNN and RetinaNet on MS COCO 2014 (N = 5, 000, failure defined as multi-label Brier exceeding the 75th percentile). Entropy is the dominant standalone signal for both detectors (AUC of 0.8839 and 0.9121), while Confidence<sup>−1</sup> performs substantially weaker (0.6017 and 0.7015), inverting the pattern observed on ImageNet. MUS achieves moderate standalone AUC (0.8003 and 0.7469), with Spearman correlations of 0.6133 and 0.5095 $( p < 1 0 ^ { - 6 } )$ confirming meaningful alignment with Brier-based failure considerably stronger than in the single-label setting. Fusion CV matches or marginally exceeds entropy alone (0.8851 and 0.9133), with MUS providing the primary complementary contribution, particularly for RetinaNet. Near-zero crop rates (0.002 and 0.001) confirm adequate detection coverage for MUS computation across both architectures.

Table 30 examines sensitivity of MUS AUC to the detection confidence threshold. Performance is fully stable between 0.05 and 0.10, with only modest degradation at 0.30 (Faster R-CNN: −0.0127; RetinaNet: −0.0446), confirming robustness to conservative thresholding choices.

Table 29: MS COCO 2014 failure detection - Faster R-CNN and RetinaNet (5,000 images, failure = multi-label Brier > 75th percentile). Entropy dominates as standalone signal; MUS contributes significantly in fusion, especially for RetinaNet.
<table><tr><td>Model</td><td>Method</td><td>AUC</td><td>ECE</td><td>AUPR</td><td>FPR@80%</td><td>FPR@95%</td><td>AURC</td><td>E-AURC</td></tr><tr><td>Faster R-CNN</td><td>MUS (Ours)</td><td>0.8003</td><td>0.0581</td><td>0.5835</td><td>0.3701</td><td>0.6781</td><td>0.10044</td><td>0.06919</td></tr><tr><td>Faster R-CNN</td><td>Entropy</td><td>0.8839</td><td>0.0806</td><td>0.7412</td><td>0.2077</td><td>0.4512</td><td>0.06895</td><td>0.03770</td></tr><tr><td>Faster R-CNN</td><td>Confidence-inv</td><td>0.6017</td><td>0.2429</td><td>0.2984</td><td>0.6485</td><td>0.8512</td><td>0.17993</td><td>0.14868</td></tr><tr><td>Faster R-CNN</td><td>Fusion CV</td><td>0.8851</td><td></td><td>0.7426</td><td>0.2127</td><td></td><td></td><td></td></tr><tr><td>RetinaNet</td><td>MUS (Ours)</td><td>0.7469</td><td>0.0561</td><td>0.4501</td><td>0.4171</td><td>0.6901</td><td>0.11438</td><td>0.08313</td></tr><tr><td>RetinaNet</td><td>Entropy</td><td>0.9121</td><td>0.1032</td><td>0.8055</td><td>0.1568</td><td>0.3845</td><td>0.05981</td><td>0.02856</td></tr><tr><td>RetinaNet</td><td>Confidence-inv</td><td>0.7015</td><td>0.1645</td><td>0.3777</td><td>0.4773</td><td>0.7576</td><td>0.13288</td><td>0.10163</td></tr><tr><td>RetinaNet</td><td>Fusion CV</td><td>0.9133</td><td></td><td>0.8109</td><td>0.1476</td><td></td><td></td><td></td></tr></table>

Table 30: MS COCO detection confidence threshold sensitivity for MUS. AUC is fully stable from 0.05 to 0.10 and degrades modestly at 0.30, confirming robustness to conservative thresholding.
<table><tr><td>Threshold</td><td>FRCNN AUC</td><td>FRCNN AUPR</td><td>RetinaNet AUC</td><td>RetinaNet AUPR</td></tr><tr><td>0.05</td><td>0.8003</td><td>0.5835</td><td>0.7469</td><td>0.4501</td></tr><tr><td>0.10</td><td>0.8003</td><td>0.5835</td><td>0.7469</td><td>0.4501</td></tr><tr><td>0.30</td><td>0.7876</td><td>0.5644</td><td>0.7023</td><td>0.3929</td></tr></table>

![](images/89fcb39fe26775270c35837ac5b1d53a81640c1d54977f9f8e25ad3d5b852f66.jpg)  
Figure 9: MUS discriminability as a function of lesion spatial extent (NIH ChestX-ray14, DenseNet-121). Per-class JSD AUC for failure detection plotted against approximate ordinal lesion size rank. Difuse pathologies (group mean JSD AUC = 0.944) substantially outperform localized lesions (0.603), consistent with the five-crop resolution limit of 3 cm.

## H MULTI-SCALE MUS FOR NODULE DETECTION

Figure 9 shows MUS Discriminability as a Function of Lesion Spatial Extent (NIH ChestX-ray14, DenseNet-121). (Left) Per-class JSD AUC for failure detection plotted against approximate ordinal lesion size rank, where rank 1 corresponds to the most spatially localized pathologies (Nodule, Mass) and rank 6 to the most spatially difuse (Pneumonia, Edema). Difuse pathologies cluster at high JSD AUC (group mean of 0.944, shown in the right panel), while localized lesions cluster at low JSD AUC (group mean of 0.603). The Pearson correlation across all 14 classes is $\mathrm { r } = 0 . 3 0 6 \ ( \mathrm { p } = 0 . 2 8 8 )$ ; the non-significant p-value reflects the bimodal rather than linear structure of the relationship the two groups are well separated but the within-group variance is high. (Right) Group-level bar chart with 95% bootstrap confidence intervals confirming the difuse vs. localized split. This analysis directly motivates the multi-scale extension in Appendix H. Since the primary five-crop decomposition operates at 112×112 pixel resolution, pathologies smaller than approximately 3 cm in diameter at standard acquisition resolution fall below the crop resolution limit, explaining the low Nodule AUC (0.368). The multi-scale extension partially recovers performance for small focal lesions (Nodule: 0.368 → 0.589 AUC, Appendix H).

MUS Performance vs. Disease Prevalence  
![](images/6ca8b666f0f90b8f14862d899e94cbae89d29de2c9891940b4a960b5148be6d8.jpg)  
Figure 10: MUS performance vs. disease prevalence (NIH ChestX-ray14, DenseNet-121). Per-class JSD AUC for failure detection plotted against test-set prevalence (%)

Figure 10 shows MUS Performance vs. Disease Prevalence (NIH ChestX-ray14, DenseNet-121). Per-class JSD AUC for failure detection plotted against test-set prevalence (%), ranging from Hernia (0.34%) to Infiltration (23.9%). The near-zero Pearson correlation (r = 0.112, p = 0.702) confirms that MUS discriminability is not confounded by class frequency or label imbalance. This rules out the alternative explanation that commonly occurring classes benefit from better model calibration, which could artificially inflate uncertainty discrimination for frequent classes. The wide performance variation across classes of similar prevalence for example, Pneumothorax (10.4% prevalence, JSD AUC = 0.894) vs. Emphysema (4.3% prevalence, JSD AUC = 0.670) confirms that lesion morphology and spatial extent, not dataset statistics, drive MUS discriminability. Rare classes such as Hernia (0.34%) and Pneumonia (2.2%) span both extremes of JSD AUC (0.884 and 0.979 respectively), further confirming the prevalence-independence of MUS performance.

Table 31 reports the multi-scale MUS extension for Nodule failure detection on NIH DenseNet-121. The baseline five-crop coarse decomposition achieves per-class JSD AUC of 0.3680 for Nodule below chance. A 16-crop fine-grained grid (56 × 56 patches, covering the full image at finer resolution) partially recovers performance.

The fine-grid MUS score (0.589) substantially outperforms the coarse baseline (0.368). The multi-scale average (0.495) is lower because averaging with the weak coarse score dilutes the signal; therefore, we report the fine MUS as the primary multi-scale result throughout the paper. The remaining gap relative to difuse pathology performance (0.85–0.98) reflects the inherent resolution limit of small focal lesion detection at standard imaging resolutions. This improvement confirms that the limitation is geometric five 112 × 112 crops cannot resolve nodules smaller than 3 cm rather than fundamental. The multi-scale MUS approach (using a 4×4 grid of 56×56 crops) is not adopted as the primary method due to its 16-pass overhead; instead, it is presented as a direction for future work.

Table 31: Multi-scale MUS for Nodule failure detection (NIH DenseNet-121). Fine crops use a 16-crop 56 × 56 grid. The fine-only result (0.589) is the primary reported improvement; the coarse+fine average (0.495) is lower because averaging with the weak coarse score (0.368) dilutes the signal.
<table><tr><td>Method</td><td>Nodule JSD AUC</td></tr><tr><td>Baseline (5-crop coarse)</td><td>0.3680</td></tr><tr><td>Fine MUS (16-crop, 56 × 56)</td><td>0.5892</td></tr><tr><td>Multi-scale avg (coarse + fine)</td><td>0.4945</td></tr><tr><td>Improvement (Fine vs. Baseline)</td><td>+0.2212</td></tr></table>

## I CLINICAL UTILITY AND DEMOGRAPHIC FAIRNESS

## I.1 When to Use SpatialUQ

Table 32: When to use SpatialUQ. This table summarizes findings already present in Sections 5–6

<table><tr><td>Use MUS when</td><td>Do not rely on MUS when</td></tr><tr><td>Diffuse, globally visible con- ditions</td><td>Highly localized lesions (nodules)</td></tr><tr><td>Overconfident multi-label classifiers</td><td>Well-calibrated single-label models</td></tr><tr><td>Moderate distribution shift</td><td>Severe covariate shift (ρ ≤ 0.15)</td></tr><tr><td>No retraining or internals ac- cess</td><td>Generative VLMs with global pooling</td></tr></table>

## I.2 Selective Prediction Coverage (NIH DenseNet-121)

Table 33 reports selective prediction performance for NIH DenseNet-121 using MUS as the referral score. At a 10% recall budget, MUS refers only 4.3% of the test set (1,091 images) with a precision of 58.7%, catching 640 failures demonstrating that a small, targeted referral volume captures a disproportionate share of model errors. Precision degrades modestly as recall increases (54.5% at 20%, 51.4% at 30%), reflecting the expected precision recall trade-of under a fixed failure distribution. Backbone AUC improves monotonically with referra rate, from +0.0017 at 5% to +0.0092 at 30% (baseline AUC = 0.8080), confirming that selectively abstaining on high-uncertainty predictions yields measurable downstream classification improvement. As we can see from Figure 11 for Pneumothorax specifically , rejecting the top 10% most uncertain images improves class AUC from 0.8471 to 0.8495 (+0.0023), with further gains at 20% rejection (0.8503, +0.0031), illustrating that MUS-guided deferral provides clinically meaningful improvement even for already well-performing classes.

Table 33: Selective prediction - NIH DenseNet-121 (MUS as uncertainty score). Images are referred in descending MUS order. Budget = fraction of full test set referred; Errors Caught = number of detected failures out of 6,399 total.
<table><tr><td>Recall</td><td>Precision</td><td>Images Referred</td><td>Budget</td><td>Errors Caught</td></tr><tr><td>10%</td><td>58.7%</td><td>1,091</td><td>4.3%</td><td>640 6,399</td></tr><tr><td>20%</td><td>54.5%</td><td>2,349</td><td>9.2%</td><td>1,280 / 6,399</td></tr><tr><td>30%</td><td>51.4%</td><td>3,732</td><td>14.6%</td><td>1,920 / 6,399</td></tr></table>

## I.3 Clinical Triage Simulation

To operationalise selective prediction in a clinically realistic setting, we evaluate both a validation-calibrated 95%-sensitivity threshold (ensuring no more than 5% of diseased images are missed) and direct MUS-ranked referral at fixed recall levels.

Under MUS-ranked referral, the NIH DenseNet-121 backbone achieves the most favourable profile: referring the 2,349 highest-MUS images (9.2% of 25,596) recovers 20% of Brier-defined failures at 54.5% referral precision.

EficientNet-B4 (threshold = 0.0801) refers 97.7% of images under the same 95%-sensitivity rule (only a 2.3% workload reduction) at a false negative rate of 0.93%, reflecting its narrower uncertainty spread relative to DenseNet-121.

Clinical Utility: Rejecting High-MUS Images Improves Pneumothorax Detection  
![](images/6d94c2c7627fd7454ea9b8685794e8f0974ba51d09083d5633007171feed0780.jpg)  
Figure 11: Selective prediction via Multi-view Uncertainty Score (MUS) for Pneumothorax detection. Performance (AUROC) shows a general upward trend as images with high spatial divergence are rejected, with local fluctuation reflecting sampling variability at intermediate rejection rates. The overall trajectory stabilizes above the baseline AUC of 0.847 at higher rejection rates, indicating that MUS efectively identifies atypical or spatially incoherent presentations whose removal improves downstream classification performance.

ViT-B/16 (threshold = 0.0782) refers 96.4% of images (a 3.6% workload reduction [3.4%, 3.9%]) at a false negative rate of 1.22%. The validation-calibrated threshold thus maintains 95% sensitivity but barely reduces workload on EficientNet-B4 and ViT-B/16; the MUS-ranked referral above is the practically useful operating point.

## I.4 Demographic Fairness

Table 34 reports MUS failure detection AUC stratified by age group and gender for DenseNet-121 on the NIH test set $( N = 2 5 , 5 9 6 )$ . Gender parity is near-perfect (female 0.7838 vs. male 0.7837), indicating no sex-based bias in spatial uncertainty estimation. Age-stratified results reveal a modest performance gradient, with MUS AUC highest for patients under 40 (0.8199) and declining with age (0.7512 for age 60-80). We attribute this to two compounding factors: older patients exhibit higher rates of comorbid findings, producing genuinely ambiguous multi-label images where spatial inconsistency is a noisier failure signal; and the NIH training distribution skews toward middle-aged presentations, potentially yielding better-calibrated predictions for younger patients.

Table 34: Demographic fairness audit - MUS failure detection AUC by age group and gender, NIH DenseNet-121 $( N = 2 5 , 5 9 6 )$ . Gender parity is near-perfect (0.7838 vs. 0.7837). MUS AUC is slightly higher for younger patients, consistent with radiographic image quality and lesion acuity diferences.
<table><tr><td>Subgroup</td><td>N</td><td>MUS AUC</td></tr><tr><td> $\mathrm { A g e } < 4 0$  Age 40–60</td><td>8,872 11,151</td><td>0.8199 0.7714</td></tr><tr><td>Age 60–80</td><td>5,379</td><td>0.7512</td></tr><tr><td> $\mathrm { A g e } > 8 0$ </td><td>190</td><td>0.7558</td></tr><tr><td>Female</td><td>10,714</td><td>0.7838</td></tr><tr><td>Male</td><td>14,882</td><td>0.7837</td></tr><tr><td>Overall</td><td>25,596</td><td>0.7837</td></tr></table>

Table 35 reports the corresponding audit for EficientNet-B4, which exhibits a markedly diferent age-stratified profile: MUS AUC increases with age (0.7723 for age < 40 vs. 0.7943 for age 60-80), inverting the gradient observed for DenseNet-121. This reversal suggests that architectural diferences in how compound-scaled convolutional features encode spatial inconsistency interact non-trivially with patient demographics. Gender disparity remains negligible (female 0.7813 vs. male 0.7863). Collectively, both audits confirm that MUS does not introduce systematic sex-based bias, while the architecture-dependent age gradient warrants consideration when deploying MUS across heterogeneous patient populations.

Table 35: Demographic fairness audit - NIH EficientNet-B4. Unlike DenseNet-121, EficientNet-B4 shows slightly higher MUS AUC for older patients (Age 60–80: 0.7943 vs. <40: 0.7723), suggesting architectural diferences in how spatial inconsistency manifests across patient demographics.
<table><tr><td>Subgroup</td><td>N</td><td>MUS AUC</td></tr><tr><td>Age &lt; 40 Age 40–60 Age 60–80</td><td>8,872 11,151 5,379</td><td>0.7723 0.7814 0.7943</td></tr><tr><td>Female</td><td>10,714</td><td>0.7813</td></tr><tr><td>Male</td><td>14,882</td><td>0.7863</td></tr><tr><td>Overall</td><td>25,596</td><td>0.7842</td></tr></table>

## J IMPLEMENTATION DETAILS

Hardware and training times. All NIH experiments were run on a single NVIDIA Tesla T4 GPU (Kaggle).   
Training times: DenseNet-121 ≈8 h; EficientNet-B4 ≈10 h; ViT-B/16 ≈6 h (early stopping at epoch 17).

Inference cost. MUS requires 6 forward passes per image (1 global + 5 crops) under torch.no\_grad(), all runnable in a single batched call. Measured wall-clock time on the NIH test set (25,596 images, batch size 32): DenseNet-121 ≈8 min; EficientNet-B4 ≈10 min; ViT-B/16 ≈29 min. MC-Dropout (30 passes) required ≈39 min, ≈51 min, and ≈145 min respectively. a consistent 5× speedup matching the pass-count reduction (6 vs. 30). A five-member deep ensemble uses 5 forward passes (one per model), comparable to MUS at inference, but requires 5× the training cost and cannot be applied to a frozen checkpoint. MUS requires no additional training, making it suitable for frozen black-box deployment.

Reproducibility. All experiments use fixed random seed 42, deterministic cuDNN, and result caching (.npy/.csv files). Code and data splits are included in the supplementary material and will be made publicly available upon acceptance. The ViT backbone is loaded via timm library (vit\_base\_patch16\_224). ImageNet and COCO models use standard torchvision pretrained checkpoints.

Baseline hyperparameters. Mahalanobis distance: PCA to 256 dimensions (retaining ≥95% variance), global Gaussian fit on training features, Shrunk Covariance regularization. ODIN: temperature T = 1000, input perturbation $\varepsilon = 0 . 0 0 1$ , softmax-based (not sigmoid), applied to all 14 output logits jointly. MC-Dropout: T = 30 stochastic forward passes with dropout active; mean predictive variance over classes used as uncertainty score. TTA: N = 10 augmentations (random scale 0.8-1.0, rotation ±10<sup>◦</sup>; no horizontal flip to preserve laterality). Deep Ensemble: 5 independently trained DenseNet-121 models (seeds 42, 186, 456, 789, 1011); mean predictive entropy used.

## K EXTENDED DISCUSSION AND SOTA ANALYSIS

## K.1 Why MUS Works: Mechanism

MUS exploits a fundamental property of neural classifiers under uncertainty. When a model has confidently localized discriminative features, its predictions remain stable across spatial sub-regions: the full image and any crop partially containing those features will yield broadly similar output distributions, producing low JSD. When no such anchoring evidence exists due to low image quality, atypical presentation, or distributional mismatch small changes in the spatial field of view produce large distributional shifts, yielding high JSD. This makes spatia inconsistency a more principled signal than predictive entropy: entropy reflects output-level uncertainty but cannot distinguish genuine label ambiguity from a complete absence of evidence. MUS specifically measures whether a model’s confidence is spatially reproducible a property that entropy is blind to. This mechanism holds consistently across model families. Spearman correlations between MUS and per-image Brier score on NIH are $\rho = 0 . 5 2$ (DenseNet-121), ρ = 0.41 (EficientNet-B4), and ρ = 0.44 (ViT-B/16), all significant at $p < 1 0 ^ { - 6 }$ . The modest drop from DenseNet to EficientNet and ViT is consistent with their more spatially coherent feature representations, which compress the dynamic range of MUS as discussed in Section 6 Score calibration further supports practical deployment: MUS achieves SCE of 0.04-0.08 across architectures, substantially better than most competing methods (Table 1), confirming that MUS scores align closely with actual failure rates rather than producing well-ranked but poorly-scaled uncertainty estimates.

![](images/707024d39d60061ea1f76a060cfc2414bd88b3c763c61e7f616389630d3d3998.jpg)  
Figure 12: Spatial disagreement heatmaps. High MUS (top) shows strong per-crop JSD variation; low MUS (bottom) shows uniform consistency. Color scale: black (low) to yellow (high).

Figure 12 shows Spatial Disagreement Heatmaps. Each panel shows a chest X-ray paired with its normalized per-crop JSD inconsistency map (color scale: black = low, yellow = high). On top row (High MUS), two images with $\mathrm { M U S } = 0 . 0 2 2$ and 0.024 display strong spatial variation across quadrant crops, with particularly high disagreement localised to upper lung regions consistent with difuse pathologies whose spatial extent changes meaningfully across crops. On bottom row (Low MUS), two images with MUS = 0.001 show uniformly low JSD across all crop positions, reflecting spatially stable model predictions regardless of field of view. The heatmaps provide interpretable spatial grounding for the scalar MUS score.

Figure 13 provides mechanistic validation for SpatialUQ via Grad-CAM attribution maps. High-MUS cases (red border; $\mathrm { M U S } = 0 . 0 2 2 \ – 0 . 0 2 4 )$ consistently exhibit anatomically difuse or mislocalized attributions, with activations concentrated outside the lung parenchyma, signifying a failure to anchor to discriminative features. Conversely, low-MUS cases (blue border; $\mathrm { M U S } \approx 0 )$ show compact, plausible attributions localized to expected pathology regions. This correlation between spatial inconsistency and attribution difusion suggests that high MUS identifies instances where the model relies on spurious or non-specific features, whereas spatial consistency reflects the utilization of localized, interpretable features.

## K.2 Per-Method SOTA Analysis

MC-Dropout (Gal and Ghahramani, 2016). MUS outperforms MC-Dropout in 4 of 5 settings: NIH DenseNet (+0.119 AUC), EficientNet (+0.249), ViT (+0.100), CheXpert (+0.109), all $p < 1 0 ^ { - 6 }$ . The sole exception is VinBigData (−0.150 AUC), where parameter-space stochasticity captures epistemic uncertainty under extreme distribution shift more reliably than spatial consistency. MC-Dropout’s primary disadvantage, 5× the inference passes of MUS (30 vs. 6), and requirement for training-time dropout layers are entirely avoided by MUS.

Deep Ensembles (Lakshminarayanan et al., 2017). The 5-member DenseNet ensemble achieves 0.813 AUC, outperforming single model MUS (0.784) but below the linear fusion $( 0 . 8 3 2 , \Delta = + 0 . 0 1 9 , p < 1 0 ^ { - 6 } )$ Ensembles require training 4 additional models 5× training cost and storage making MUS + linear fusion a substantially more practical alternative in resource-constrained clinical settings.

ODIN (Liang et al., 2017). Achieves 0.759 AUC on NIH DenseNet (below MUS) but 0.800 AUC on NIH ViT the strongest single-method competitor on ViT. ODIN requires backpropagation through the input for

Grad-CAM Validation: High-MUS Images Show Mislocalized / Non-Lung Attributions HIGH UNCERTAINTY (MUS > 0) LOW UNCERTAINTY (MUS = 0)

![](images/5ee40f9529f9bcbd55dafaaab7fdc3cf8678be7a0d516dbab643a5df19b88d5d.jpg)  
Figure 13: Grad-CAM validation. High MUS (red border) produces difuse, non-lung attributions; low MUS (blue border) yields compact, localized activations.

gradient-based perturbation, adding inference overhead and gradient access requirements that MUS does not need.

DDU (Mukhoti et al., 2021) and Mahalanobis (Lee et al., 2018). Both achieve 0.754 AUC on NIH DenseNet in our post-hoc implementation. Both require access to training-set feature distributions and storage of covariance matrices over the full training set (∼73,891×1,024 for DenseNet), while MUS has zero storage overhead. The original DDU additionally requires spectral normalization during training; our post-hoc approximation may underperform the full method.

## K.3 Per-Setting Performance Interpretation

Architecture dependence on NIH. MUS achieves 0.784 (DenseNet-121), 0.784 (EficientNet-B4), and 0.750 (ViT-B/16). The ViT gap reflects global self-attention’s tendency to produce spatially coherent predictions even when incorrect, suppressing the inconsistency gradient. EficientNet’s compound scaling allocates more processing to fine-grained spatial features, explaining why its Nodule per-class JSD AUC (0.882) dramatically exceeds DenseNet’s (0.368), EficientNet’s spatial inconsistency is more discriminative precisely where fine-grained resolution matters.

Entropy inversion on NIH. Raw entropy achieves AUC = 0.177 on NIH DenseNet significantly below random. The model trained with label smoothing and class-balanced BCE produces moderate entropy across most predictions; failing predictions tend to be overconfident (high probability assigned to wrong class), not uncertain, making entropy anti-correlated with failure in this configuration. This highlights that MUS and entropy are both architecture- and training-procedure-dependent, and that their combination requires care.

Detection vs. classification MUS performance. MUS achieves stronger standalone results on COCO (0.800 / 0.747 AUC) than on ImageNet (0.641-0.717 AUC) relative to baselines, reflecting the multi-label detection setting where spatial inconsistency across object classes is more structurally informative than in single-labe recognition. Entropy dominates on COCO (0.884 / 0.912), consistent with detection models having better confidence calibration than multi-label medical classifiers.

## K.4 Computational Cost

MUS requires 6 forward passes per image under torch.no\_grad(), all computable in a single batched call. Measured wall-clock time on the NIH test set (25,596 images, batch size 32) for MUS only: DenseNet-121 ≈8 min; EficientNet-B4 ≈10 min; ViT-B/16 ≈29 min. For comparison, MC-Dropout (30 passes) required ≈39 min, ≈51 min, and ≈145 min respectively, confirming a consistent 5× wall-clock speedup that mirrors the pass-count reduction. A five-member deep ensemble uses 5 forward passes (one per model) comparable to MUS’s 6 passes at inference but requires 5× the training cost and cannot be applied to a frozen checkpoint without retraining. MUS’s inference cost is therefore competitive with ensembles while ofering the decisive advantage of zero retraining.

## K.5 Future Directions

The most important near-term extension is validation on MIMIC-CXR (227,835 radiographs, Beth Israel Deaconess), which would enable evaluation under temporal distribution shift, correlation with radiologist confidence scores in structured reports, and intersectional fairness analysis across race, insurance status, and comorbidity burden dimensions not captured by NIH’s age/gender metadata. Our finding that MUS AUC degrades for older patients (0.820 for <40 vs. 0.751 for 60–80 years) motivates a more thorough demographic audit on richer clinical metadata. Additional directions include adaptive crop placement via saliency guidance, MUS for transformer-based segmentation models (UNETR, SwinUNETR), and volumetric extension to 3D CT.

## L LABEL MAPPING FOR ZERO-SHOT TRANSFER

## CheXpert Label Mapping (10 classes)

The NIH ChestX-ray14 dataset contains 14 disease classes. CheXpert contains 14 classes of which 10 overlap with NIH. We map the shared classes directly by name. The four NIH classes absent from CheXpert (Emphysema, Fibrosis, Pleural\_Thickening, Hernia) are excluded from evaluation on CheXpert. Uncertain labels (U) in CheXpert are treated as negative (U-Zeros policy), consistent with prior work (Irvin et al., 2019).

## VinBigData Label Mapping (10 classes)

VinBigData contains 28 annotated findings. We map the 10 classes that correspond directly to NIH labels: Atelectasis, Cardiomegaly, Consolidation, Efusion, Infiltration, Mass, Nodule, Pleural\_Thickening, Pneumothorax, Fibrosis. The four NIH classes absent from VinBigData (Edema, Pneumonia, Emphysema, Hernia) are excluded. All VinBigData annotations are radiologist-verified bounding boxes; we convert these to image-level binary labels (positive if any box of that class is present).

Remark 1. The class-averaged bound inherits slack from Jensen’s inequality in Step 2 when per-class JSD values are heterogeneous; the per-class bound itself is tight as $p _ { c } , q _ { c } \to { \frac { 1 } { 2 } }$ . The purpose of Lemma 1 is to establish that $s _ { \mathrm { b e r n } }$ formally upper bounds mean absolute class-level disagreement, guaranteeing that a large MUS score necessarily implies substantial shifts between global and local predictions, rather than to provide a numerically tight certificate. The empirical Spearman correlation between $s _ { \mathrm { b e r n } }$ and per-image Brier score $( p < 0 . 0 0 0 1$ across all medical benchmarks) confirms the practical utility of this relationship.