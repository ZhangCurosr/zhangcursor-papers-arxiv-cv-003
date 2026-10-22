# Supervision-Adaptive Curriculum and Guarded Anti-Curriculum Learning for Dense Prediction

Chunming He, Dingming Zhang, Fengyang Xiao, Zhiwen Cao, and Sina Farsiu, Fellow, IEEE

![](images/d12b358ad075b8b25cb2cd272d3a0bbf45b297c6ff1ed7319850642513773f2e.jpg)

![](images/fc07f1628395ab7ebd1ad9cd03c6d4bd297887490601bc9d0384ef4eef56f325.jpg)

![](images/3d70918faa2b7f6ad279eceadba51ff0098481f1b19457dd28789ef343facfe0.jpg)  
Fig. 1. PaceShift across dense prediction tasks. Left: Ground-truth masks and predictions from the baseline and its PaceShift-trained counterpart, overlaid on the inputs. Right: Radar chart comparing the baseline, CurriSeg, and PaceShift, with each axis labeled “Task (Dataset, Metric)”. Each axis is independently scaled so that the outer ring represents the best performance. For BER, lower values are mapped farther outward. PaceShift adds no inference-time parameters.

<sub>Temporal</sub>Abstract—Curriculum learning orders samples by difficulty, <sup>Train statistics</sup> <sup>hardD d</sup> <sup>IoU</sup> <sup>GT</sup> <sup>= −1</sup> <sup>(</sup> <sup>)</sup> <sup>P D d</sup> <sup>IoU</sup> <sup>GT</sup> <sup>= −1</sup> <sup>(</sup> <sup>)</sup> but difficulty alone does not establish supervision reliability for dense prediction. Easy samples can expose shortcuts, hard samples can be informative, and noisy or ambiguous samples can <sup>Static</sup> <sup>difficulty</sup> <sup>amplifies</sup> <sup>shortcut</sup> <sup>reliance.</sup> <sup>Dense-mask</sup> <sup>dependent</sup>mislead training. Building on our conference method CurriSeg, we present PaceShift, a stabilize-then-challenge paradigm combining supervision-adaptive curriculum selection with guarded spectral post-training. A learning-dynamics view motivates separating supervision reliability from temporal update value and applying spectral challenges after stabilization. The first phase estimates reliability from available supervision, augmentation consistency, model disagreement, and prediction certainty. Temporal statistics guide sample weighting, while pixel-uncertainty weights reduce the early influence of uncertain regions. The second phase ranks predictions from multiple spectral views using supervision agreement, robustness, reference consistency, and penalties. A pairwise preference objective measured against the frozen Phase-I reference, together with an anchor, refines the prediction. The framework supports full, weak, semi-supervised, and modelderived supervision. Experiments across context-entangled content segmentation (CECS), broader dense-prediction tasks, and foundation-model and cross-modal adaptations show accuracy gains. Targeted diagnostics indicate improved pooled calibration and corruption robustness in the evaluated settings. PaceShift preserves the deployed architecture and adds no inference computation. Code will be publicly released.

Index Terms—Anti-Curriculum Learning, Curriculum Learning, Dense Prediction, Preference Learning, Supervision Reliability, Training Dynamics.

## I. INTRODUCTION

<sup>Temporal</sup>statistics <sup>Multi-radius</sup>+ Guard Q PTrain <sup>Single-radius</sup>SBFT   P D d = −1  TrainD <sup>ENSE</sup> <sup>prediction</sup> <sup>assigns</sup> <sup>a</sup> <sup>structured</sup> <sup>label</sup> <sup>to</sup> <sup>every</sup> <sup>pixel</sup>and underpins perception across medical, autonomous, <sub>Four</sub> <sub>cues:</sub> <sub>sup.</sub> <sub>/</sub> <sub>cons.</sub> <sub>/</sub> <sub>dis.</sub> <sub>/</sub> <sub>ent.</sub>and industrial applications [1]–[5]. Most progress has been hallenge unguarded. Supervision-adaptive difficulty with a guarded multi-radius challenge.architectural: stronger backbones [6], dedicated detectors with multi-scale aggregation [7]–[9], attention [10], and frequencyaware modules [11]. We instead argue that dense prediction is limited not only by network capacity but also by training-signal reliability and the order in which it is consumed (Fig. 1).

The reliability problem is sharpest in context-entangled content segmentation (CECS), where objects share intrinsic visual patterns with their background, as in camouflaged object detection [7], transparent object detection [12], and concealed defect detection [4], [13]. We also include polyp segmentation as a CECS-related task with low boundary contrast, local ambiguity, and annotation uncertainty. Existing methods ad dress the resulting feature ambiguity mainly by enhancing representations [11], [14]. Yet a more basic bottleneck is the learning dynamics that arise when the data distribution is entangled. We make this concrete with three observations that, while most visible in CECS, recur across dense prediction.

Difficulty and learning value are not the same quantity. Curriculum learning [15] typically orders samples from easy to hard, but in CECS, apparent easiness can arise from residual texture contrast that reveals the object. Prioritizing samples with low loss or error may therefore encourage reliance on fragile high-frequency cues instead of shape and context. This concern is supported by our experiments: under a matched budget, naive easy-to-hard ordering performs no better than random ordering, while conventional schedules can reduce accuracy. Frequency-intervention results further show greater sensitivity to high-frequency perturbations after naive easy-first training.

Hard samples pose a complementary challenge: high prediction error can reflect either informative visual difficulty or unreliable supervision. A well-concealed object may be difficult yet informative, whereas weak annotations, imperfect ground-truth quality [16], label drift, or intrinsically ambiguous regions can produce high errors without equally useful training signals. Since a static difficulty score cannot reliably distinguish these cases, difficulty-based selection may discard informative samples while retaining unreliable ones, increasing the risk of overfitting.

![](images/ec97ab89587365618cb4026699be28c01f218c3340e4cd754c556956a798517b.jpg)  
Fig. 2. Curriculum paradigms for dense prediction. A naive curriculum orders samples easy-to-hard by a static ground-truth difficulty and can strengthen reliance on fragile texture cues. CurriSeg adds temporal statistics and a stabilize-then-challenge schedule, but its score is defined by dense-mask error and its single-radius challenge is unguarded. PaceShift keeps the same overall two-phase training skeleton and changes two core components: a four-cue supervision-adaptive score $\rho _ { i }$ and a guarded multi-radius spectral challenge during post-training, without changing the deployed model architecture.

Confidence, finally, is not reliability. Uncertain boundary and low-contrast regions can dominate early gradients, while high confidence need not imply correctness. Directly rewarding confidence can therefore reinforce errors and degrade calibration.

Together, these three observations suggest that reliable training must distinguish supervision quality from temporal learning value, regulate uncertain pixels, and challenge shortcut reliance only after initial stabilization is reached.

From CurriSeg to PaceShift. Our conference work CurriSeg [17], published at ICML 2026, introduced a two-phase stabilize-then-challenge framework for CECS. It combined dense-mask difficulty trajectories, temporal sample weighting, pixel uncertainty weighting, and single-radius spectralblindness fine-tuning (SBFT). However, its difficulty score required dense ground-truth masks and could not be applied uniformly across weak, semi-supervised, and model-derived supervision. Moreover, its single-radius spectral stage lacked a mechanism for assessing the reliability of alternative predictions. To address these limitations, we develop PaceShift (Fig. 2), which retains the two-phase design while introducing supervision-adaptive curriculum selection and guarded multiradius post-training with reference-relative preference learning. These changes extend the training framework without modifying the deployed architecture or increasing inference cost.

Scope. CECS serves as our primary setting for studying supervision reliability and texture shortcuts together. Experiments on broader dense-prediction tasks evaluate the transfer of the full training framework, while mechanism-specific evidence is estab lished primarily in CECS and unreliable-supervision settings.

This paper makes four contributions:

(1) A learning-dynamics view of dense prediction (Sec. III-B). We explain why naive easy-first curricula can amplify fragile shortcuts and why the stabilize-then-challenge framework succeeds, separating supervision reliability from temporal update value and treating SBFT as a post-stabilization frequency bottleneck whose order matters.

(2) Supervision-adaptive curriculum selection (Sec. III-C). We replace ground-truth difficulty with a reliability score combining available supervision, augmentation consistency, model disagreement, and prediction uncertainty, yielding one curriculum under full, weak, semi-supervised, and modelderived supervision while reducing confirmation bias.

(3) Guarded anti-curriculum post-training (Sec. III-D). SBFT-P (SBFT with preference-guarded post-training) combines multi-radius spectral challenges with a supervisionadaptive preference guard based on correctness, robustness, and consistency, with uncertainty as a weak penalty and not as the sole selection rule, improving robustness while reducing overconfidence in the evaluated settings and recovering the original SBFT as a special case under full supervision.

(4) Systematic validation and diagnostics (Sec. IV). We evaluate PaceShift across multiple dense-prediction families, supervision regimes, architectures, data-centric baselines, and foundation-model and cross-modal settings. Targeted experiments examine training dynamics, supervision reliability, candidate selection, corruption robustness, and calibration without additional inference computation.

Beyond the methodological changes, this journal extension substantially expands the evaluation of CurriSeg [17]. We extend the broader dense-prediction evaluation to clinical, di chotomous, and salient segmentation; add vision-language and reference-guided evaluations; expand data-centric comparisons; and report paired-run statistical tests. Targeted analyses of supervision reliability, temporal update value, guard selection, phase ordering, frequency sensitivity, corruption robustness, and calibration further examine the underlying mechanisms.

## II. RELATED WORK

Context-entangled and dense prediction segmentation. CECS spans camouflaged object detection [3], [18], transparent object detection [19], and concealed defect detection [13], with polyp segmentation [20] as a closely related low-contrast task. Progress has been mostly architectural: encoder-decoder backbones with multi-scale aggregation [7], attention-based refinement [9], edge- or uncertainty-aware branches [10], [21], transformer context modeling [22], and frequency-aware modules [11]. These works strengthen representation but keep a standard supervised routine with shuffled mini-batches, so the roles of sample difficulty, label ambiguity, and texture shortcuts in the learning dynamics remain underexplored. Reversible unfolding [12] and nested unfolding [14] designs have begun to inject inductive structure into the decoder, yet the optimization schedule itself is rarely treated as an explicit design variable. Curriculum learning and self-paced learning. Curriculum learning orders training from easy to hard to improve generalization [15], [23], with variants in self-paced learning [24], teacherstudent curricula [25], [26], and loss- or hardness-based ranking [27], [28]. These methods assume that easy samples give reliable supervision and that difficulty tracks learning utility. Both break in CECS: easy samples may carry spurious texture that promotes shortcuts, and hard samples can be at once informative and ambiguous, consistent with analyses showing that the benefit of ordering depends sharply on data structure [26]. Most curricula also ignore the pixel-level uncertainty of dense prediction, and almost none deliberately increase difficulty after a stable solution is reached. PaceShift instead separates reliability from temporal update value, extends the curriculum to pixels, and introduces a post-stabilization anti-curriculum.

Data-centric learning: training dynamics and data efficiency. Training-dynamics methods characterize examples through evolving predictions, losses, margins, or gradients, including forgetting events [29], [30], EL2N and GraNd [31], dataset cartography [32], and AUM [33]. Dyn-Unc [34] and TDDS [35] further exploit temporal variability, while InfoBatch [36] and SeTa [37] dynamically remove low-value samples for efficiency. ADELE [38] instead uses early-learning dynamics to correct noisy labels. Most of these formulations assume sample-level labels and do not directly address sparse, pixel-level, or modelderived supervision. PaceShift builds on this line of work but derives the trajectory from supervision-adaptive dense reliability, couples it with pixel-level uncertainty, and follows stabilization with guarded post-training. Its aim is accuracy, robustness, and calibration under spatially and supervisiondependent unreliability, not pruning or label correction.

Learning under incomplete supervision. Weakly and semi-supervised segmentation learn from sparse annotations or unlabeled data through consistency regularization and pseudolabeling [39], [40]. A recurring failure is confirmation bias: training on the model’s own confident predictions reinforces existing errors [41]. In CECS this is acute, since confidence and correctness diverge in concealed regions, so a confidence threshold both keeps wrong pseudo-labels and discards hard but correct ones. PaceShift instead combines confidence, disagreement, consistency, and an external foundation-model cue for reliability estimation, whereas SEE [42] generates changing pseudo-labels from a frozen SAM and SCALER [43] jointly adapts SAM and the segmenter.

Shortcut bias, texture bias, and frequency-domain learning. Networks trained by SGD favor the simplest features that separate the data even when more robust ones exist, studied as simplicity bias [44], [45] and, for vision, texture bias [46]. This differs from the spectral bias of network functions, which learn the low-frequency components of the input-output map first [47]. The two concern different frequencies, image versus function, and conflating them yields contradictory predictions about a low-pass operation. We therefore frame SBFT as suppressing a high-frequency image shortcut to counteract texture bias without invoking function-space spectral bias.

Foundation-model adaptation and post-training. Segment Anything-style models [48] are adapted to specialized domains through adapters, prompt modules, or low-rank updates [49], [50]. Related segmentation systems condition on text or image prompts [51], [52], open-vocabulary alignment [53], or reference images [54]. Such conditioning introduces another source of unreliability, namely the alignment between the conditioning signal and the predicted mask, which PaceShift treats as an additional reliability cue without adding a new mechanism. Separately, DPO [55] learns from preferred and dispreferred outputs without fitting a separate explicit reward model. We adopt a related pairwise objective for dense predictions and define preferences through correctness, robustness, and consistency on dense masks.

## III. METHOD

## A. Problem Formulation and Overview

Let $\mathcal { D } = \{ ( x _ { i } , s _ { i } ) \} _ { i = 1 } ^ { N }$ be a dense prediction dataset, where $\boldsymbol { x } _ { i } \in \mathbb { R } ^ { H _ { s } \times \tilde { W _ { s } } \times 3 }$ is an image and $s _ { i }$ is its supervision. In full supervision, $s _ { i }$ is a dense mask $\bar { y _ { i } } \in \{ 0 , 1 \} ^ { \cdot } \bar { H } _ { s } \times W _ { s }$ . In weak supervision, $s _ { i }$ is a sparse cue such as a scribble, point, or box. In semi-supervision, $s _ { i }$ is a dense mask for a small labeled subset $\mathcal { D } _ { L }$ and empty for the unlabeled subset $\mathcal { D } _ { U }$ . A model $f _ { \theta }$ predicts a per-pixel logit map ${ \hat { y } } _ { i } = f _ { \theta } ( x _ { i } )$ . The conference framework assumed $s _ { i } = y _ { i }$ for all i. PaceShift keeps the same network and the same stabilize-then-challenge schedule, but reorganizes training around the reliability of $s _ { i }$ so that the schedule operates for any of the regimes above.

As shown in Fig. 3, PaceShift runs in two phases over $T$ epochs. Phase I, supervision-adaptive robust curriculum selection (SA-RCS, epochs 1 to $T _ { c } )$ , builds a stable representation by progressively selecting reliable yet informative samples and by damping uncertain pixels. Phase II, guarded anti-curriculum promotion (SBFT-P, epochs $T _ { c } + 1 \ \mathrm { t o } \ T )$ , deliberately raises difficulty through a multi-radius spectral challenge and refines the model through a guarded preference objective.

The two phases can be viewed as a time-varying objective: Phase I minimizes $\begin{array} { r } { \sum _ { i \in \mathcal { S } _ { t } } \omega _ { i } \mathcal { L } _ { \mathrm { s e g } } \big ( f _ { \theta } ( x _ { i } ) , t _ { i } , W _ { i } \big ) } \end{array}$ over the growing reliable subset $S _ { t }$ , where $\mathcal { L } _ { \mathrm { s e g } }$ is a weighted segmentation loss, $t _ { i }$ is the supervision-adaptive target, and $\omega _ { i }$ and $W _ { i }$ are the sample and pixel weights, while Phase II minimizes the post-training objective of Eq. (18). The schedule hence moves from maximizing reliable information intake to minimizing reliance on nuisance frequencies, and both phases use the same supervision-evidence and source-trust construction, for reliable intake in Phase I and guarded selection in Phase II.

## B. A Learning-Dynamics View

We first lay out the learning dynamics motivating the design. 1) Easy samples amplify a high-frequency shortcut: In a context-entangled scene, an object is concealed when its appearance matches the background. In CECS, one recurring source of apparent easiness is residual local high-frequency texture contrast that separates the object from its background. A model trained on these first can minimize the loss by keying on that texture difference, a feature that is cheap to fit and locally discriminative but does not transfer to well-concealed cases. This is an instance of simplicity and texture bias [44]: gradient descent prefers the simplest separating feature even when a more robust shape or context feature exists. This is not the classical spectral bias of network functions [47], [56], which concerns the order in which frequencies of the input-output map are learned. Here the relevant frequency is in image space, and the pathology is that early easy-first training increases

![](images/de20c39bcaaebdf3d060cb90cb418a08805a3bcb40351ab0d10b7e4d8fe09818.jpg)  
Fig. 3. Framework of PaceShift. Top (schematic): performance rises in Phase I, is perturbed by the spectral challenge at $T _ { c } ,$ and settles higher in Phase II. Phase I (SA-RCS): four reliability cues form a supervision-adaptive difficulty $d _ { i } = 1 - \rho _ { i }$ . After a warm-up on all samples, a percentile curriculum selects ${ \mathbf { } } S _ { t } ,$ and TSSW and PUE reweight the loss at sample and pixel level. Phase II (SBFT-P): a multi-radius challenge yields candidates that a guard scores into a preferred and dispreferred mask, refined against the reference $\theta _ { 0 } .$ . No inference parameters are added.

reliance on a high-frequency image cue. This motivates the two phases: keep shortcut-prone easy samples from dominating early training, then suppress the high-frequency channel once a sufficiently stable representation exists.

2) Temporal dynamics reveal update value: A single difficulty value cannot tell an informative hard sample from a noisy one, while the training trajectory can. For sample i let $\bar { d } _ { i } ^ { ( k ) }$ denote its difficulty at checkpoint k. Two patterns are used as heuristics. A high variance of $\{ d _ { i } ^ { ( k ) } \} _ { I }$ reflects strong temporal variation, which is common near decision boundaries or under intrinsic ambiguity. A high mean with low variance reflects persistent failure, which can arise from outliers or label errors that the model cannot resolve, and the empirical rule down-weights it. This reading aligns with example-difficulty and forgetting analyses [29], [31] and with Dyn-Unc and TDDS [34], [35], which likewise exploit training dynamics instead of a single value. We import neither score directly: we model the mean, the variance, and the persistent-failure pattern of high mean and low variance as separate, individually ablatable terms on the supervisionadaptive difficulty of Sec. III-C, because the noise-ambiguity dilemma of CECS requires distinguishing all three under any supervision regime. This is what motivates the parameterfree mid-variance preference and the outlier penalty $\omega ^ { \mathrm { o u t } }$ in Sec. III-C: intermediate temporal variation serves as a heuristic for potentially informative hardness, whereas persistent stability and persistent failure are down-weighted.

3) SBFT is a post-stabilization frequency bottleneck: Split the input into a low-frequency component $x ^ { \mathrm { l o } }$ and a highfrequency component $x ^ { \mathrm { { h i } } }$ . A representation useful for dense prediction should retain the information needed for the target without relying on the nuisance channel $x ^ { \mathrm { { h i } } }$ . SBFT attenuates $x ^ { \mathrm { { h i } } }$ while requiring consistent prediction through the supervised or guarded objective, a frequency-level bottleneck inspired by, but not formally an instance of, the information bottleneck [57]. The phase order is not interchangeable: if the bottleneck is imposed before a stable representation exists, the model is deprived of the cues it needs to bootstrap structure, and applying the challenge directly after warm-up falls below standard training on every reported metric (Table XXII).

## C. Supervision-Adaptive Robust Curriculum Selection

SA-RCS targets the noise-ambiguity dilemma by monitoring the temporal behavior of samples and the spatial uncertainty of pixels, and it does so without assuming a complete mask. It has four parts: a warm-up curriculum, a supervision-adaptive difficulty, a temporal sample weighting, and a pixel uncertainty weighting, combined in a single loss.

1) Supervision-adaptive difficulty: The difficulty defined in the conference version is the per-sample segmentation error of a periodically saved checkpoint $f _ { \theta } ^ { ( k ) }$

$$
d _ { i } ^ { \operatorname { f u l l } } = 1 - \operatorname { I o U } \bigl ( f _ { \theta } ^ { ( k ) } ( x _ { i } ) , y _ { i } \bigr ) ,\tag{1}
$$

which requires the dense mask $y _ { i } .$ For binary segmentation, IoUbased difficulty, reliability, and guard scores use the thresholded prediction $\mathbf { 1 } [ \sigma ( f _ { \theta } ( x ) ) \geq 0 . 5 ]$ . We replace it with a difficulty $d _ { i } = 1 - \rho _ { i }$ derived from a reliability score $\rho _ { i } \in [ 0 , 1 ]$ that aggregates the supervision signals that are actually available for the current training sample. We use four normalized cues, each in [0, 1] and oriented so that larger means more reliable:

• supervision agreement $s _ { i } ^ { \mathrm { s u p } }$ : for full supervision the IoU between $f _ { \theta } ^ { ( k ) } ( x _ { i } )$ and $y _ { i } ,$ , for weak supervision the satisfied ratio of the available sparse cue (scribble overlap, point coverage, or box containment), with the point-supervised setting averaging point-cue satisfaction and agreement with a frozen SAM3 proposal, for unlabeled samples the agreement with an exponential-moving-average teacher or a foundation-model mask proposal.

TABLE I  
SUPERVISION-TRUST PRESETS. IN EQ. (2), $a ^ { \mathrm { s u p } } { = } \tau _ { \mathrm { t r u s t } }$ AND $( a ^ { \mathrm { c o n s } } , a ^ { \mathrm { d i s } } , a ^ { \mathrm { e n t } } ) = ( 1 - \tau _ { \mathrm { t r u s t } } )$ π WITH THE FIXED TEMPLATE $\pi = ( 0 . 4 , 0 . 4 , 0 . 2 )$ , AND THE GUARD IN EQ. (11) REUSES τ<sub>trust</sub> WITH THE FIXED PENALTY SCALE ε. POINT SUPERVISION AVERAGES CUE SATISFACTION WITH FROZEN-PROPOSAL AGREEMENT WITHOUT CHANGING THE PRESET.
<table><tr><td>Supervision sources</td><td>Trust  $\tau _ { \mathrm { t r u s t } }$ </td><td> $Q _ { \mathrm { s u p } }$  source</td></tr><tr><td>Dense human mask (full; semi, labeled part)</td><td>0.7</td><td>GT mask</td></tr><tr><td>Sparse human cue (scribble, point, box)</td><td>0.4</td><td>weak-cue evidence</td></tr><tr><td>Model-derived supervision (EMA teacher, CAM-derived pseudo-mask)</td><td>0.3</td><td>teacher / pseudo-mask agree</td></tr></table>

• augmentation consistency $s _ { i } ^ { \mathrm { c o n s . } }$ : one minus the mean disagreement between predictions on two stochastic aug mentations of $x _ { i } ,$ , where each prediction is inverse-warped to the original image coordinates and the disagreement is measured by 1 − IoU over the aligned masks.

• model agreement $1 - \bar { s } _ { i } ^ { \mathrm { d i s } }$ : one minus the disagreement across Monte-Carlo dropout predictions, or among ensemble predictions when an ensemble is used, an internal model-uncertainty cue, distinct from the external proposal that provides mask-level evidence for $s _ { i } ^ { \mathrm { s u p } }$ where used.

• prediction certainty $1 - \bar { s } _ { i } ^ { \mathrm { e n t } }$ : one minus the image-level average of normalized pixel entropy over the probability map, an image-level cue for $\rho _ { i }$ that is distinct from the pixel-level entropy weight $W _ { h , w }$ in Eq. (8).

The reliability is their convex combination,

$$
\begin{array} { r l } & { \rho _ { i } = a ^ { \mathrm { s u p } } s _ { i } ^ { \mathrm { s u p } } + a ^ { \mathrm { c o n s } } s _ { i } ^ { \mathrm { c o n s } } + a ^ { \mathrm { d i s } } \left( 1 - \bar { s } _ { i } ^ { \mathrm { d i s } } \right) } \\ & { \qquad + a ^ { \mathrm { e n t } } \left( 1 - \bar { s } _ { i } ^ { \mathrm { e n t } } \right) , } \end{array}\tag{2}
$$

with $a ^ { \mathrm { s u p } } + a ^ { \mathrm { c o n s } } + a ^ { \mathrm { d i s } } + a ^ { \mathrm { e n t } } = 1$ , and the difficulty used everywhere below is $d _ { i } = 1 - \rho _ { i }$ . Instead of setting the four weights per regime, we derive all of them from a single sourcespecific supervision-trust level $\tau _ { \mathrm { t r u s t } } \in [ 0 , 1 ]$ that reflects how verifiable the supervision source is: $a ^ { \mathrm { s u p } } = \tau _ { \mathrm { t r u s t } }$ , and the residual mass is shared by the auxiliary cues through a fixed global template, $( a ^ { \mathrm { c o n s } } , a ^ { \mathrm { \bar { d i s } } } , a ^ { \mathrm { e n t } } ) = ( \mathrm { \bar { 1 } } - \tau _ { \mathrm { t r u s t } } )$ π with $\pi =$ $( 0 . 4 , 0 . 4 , 0 . 2 )$ , so that consistency and disagreement remain the main auxiliary signals and entropy a weak one, consistent with the design point below. $\tau _ { \mathrm { t r u s t } }$ takes one of three levels according to the primary supervision source (Table I): $\tau _ { \mathrm { t r u s t } } = 0 . 7$ for a dense human mask, which concentrates mass on $s ^ { \mathrm { s u p } }$ and recovers Eq. (1) in the limit $\tau _ { \mathrm { t r u s t } } \to 1 , \tau _ { \mathrm { t r u s t } } = 0 . 4$ for a sparse human cue, and $\tau _ { \mathrm { t r u s t } } = 0 . 3$ for a model-derived signal, where the agreement term is itself uncertain.

Reliability is not confidence. Here $\rho _ { i }$ measures the evidence conditioned reliability of the current training signal, not annotation correctness alone: $\tau _ { \mathrm { t r u s t } }$ acts as a source-trust prior, while the sample-dependent cues measure how strongly the available evidence supports learning from sample i at the current stage. The entropy term alone would select the samples the model is already sure about and would close the self-training feedback loop. Combining it with disagreement and consistency, and anchoring $s ^ { \mathrm { s u p } }$ on an external foundation-model proposal that is independent of the current model parameters where such a proposal is used, selects samples that are stable and still informative. This construction helps reduce confirmation bias in self-training. The trust level $\tau _ { \mathrm { t r u s t } }$ is the only regime-level quantity and the template π is global, so a new regime requires choosing one trust level, with no search over weights. The presets perform close to validation-tuned values and clearly better than uniform cue weights (Table XXIII), so the gains are not the product of per-dataset search.

2) Warm-up curriculum: Following the conference [17], we expand the active subset along a percentile schedule after a short warm-up. For the first $T _ { w }$ epochs all samples are used without filtering, so the model acquires a basic representation before difficulty-based selection begins. Let $p ( t )$ be the selection percentile at epoch t and Per $( \bar { \{ d _ { n } \} } _ { n = 1 } ^ { N } , \bar { p ( t ) } )$ the corresponding threshold. The active subset is

$$
S _ { t } = \mathcal { D } , \qquad t \le T _ { w } ,
$$

$$
\begin{array} { r } { S _ { t } = \big \{ i \mid d _ { i } \le \mathrm { P e r } ( \{ d _ { n } \} _ { n = 1 } ^ { N } , p ( t ) ) \big \} , \quad T _ { w } < t \le T _ { c } , } \end{array}\tag{3}
$$

$$
\begin{array} { r } { p ( t ) = p _ { \operatorname* { m i n } } + ( 1 - p _ { \operatorname* { m i n } } ) \frac { t - T _ { w } - 1 } { T _ { c } - T _ { w } - 1 } , } \end{array}
$$

with $p _ { \mathrm { m i n } } ~ = ~ 0 . 6$ and $T _ { w } = 1 0$ . Training starts from highreliability samples, selected by the reliability score instead of difficulty alone, and progressively admits lower-reliability, harder samples as the percentile $p ( t )$ grows.

3) Temporal statistics-based sample weighting: Within $S _ { t }$ samples still differ in learning value. To capture this, we compute the difficulty at refresh checkpoints. Let $f _ { \theta } ^ { ( k ) }$ denote checkpoints saved at epochs kH $< T _ { c } , k = 1 , \dots , K .$ , where $\textstyle K = \left\lfloor { \frac { T _ { c } - 1 } { H } } \right\rfloor ( K { = } 5 $ at the defaults), and let $\dot { d } _ { i } ^ { ( k ) } = 1 - \rho _ { i } ^ { ( k ) }$ with $\bar { \rho _ { i } ^ { ( k ) } }$ the reliability of Eq. (2) evaluated under $f _ { \theta } ^ { ( k ) }$ . The history $B _ { i } = \{ d _ { i } ^ { ( 1 ) } , \ldots \bar { , } d _ { i } ^ { ( k ) } \}$ stores these values, and

$$
\mu _ { i } = \frac { 1 } { | \mathcal { B } _ { i } | } \sum _ { k } d _ { i } ^ { ( k ) } , \qquad \sigma _ { i } ^ { 2 } = \frac { 1 } { | \mathcal { B } _ { i } | } \sum _ { k } \big ( d _ { i } ^ { ( k ) } - \mu _ { i } \big ) ^ { 2 } ,\tag{4}
$$

so the statistics aggregate distinct training states, not repeated copies of one checkpoint. Until two checkpoints exist we set $\omega _ { i } = 1$ . As argued in Sec. III-B, moderate variance is used as a heuristic for informative hardness, very high variance is common for ambiguous or unstable samples, and a high mean with low variance is treated as persistent failure. After min-max normalizing to $\tilde { \mu } _ { i }$ and $\tilde { \sigma } _ { i } ^ { 2 }$ , we map the statistics to weights,

$$
\omega _ { i } ^ { \mu } = 1 - \tilde { \mu } _ { i } , \quad \quad \omega _ { i } ^ { \sigma } = 4 \tilde { \sigma } _ { i } ^ { 2 } \big ( 1 - \tilde { \sigma } _ { i } ^ { 2 } \big ) ,\tag{5}
$$

a parameter-free mid-variance preference that peaks at the midpoint of the normalized range and suppresses both alreadystable and highly unstable samples. This parameter-free parabolic variance weight replaces the Gaussian variance weight of CurriSeg [17]. The final sample weight:

$$
\omega _ { i } = w _ { \mathrm { m i n } } + \left( 1 - w _ { \mathrm { m i n } } \right) \omega _ { i } ^ { \mu } \omega _ { i } ^ { \sigma } \omega _ { i } ^ { \mathrm { o u t } } , \omega _ { i } ^ { \mathrm { o u t } } = 1 - \tilde { \mu } _ { i } ( 1 - \tilde { \sigma } _ { i } ^ { 2 } ) ,\tag{6}
$$

with $w _ { \mathrm { m i n } } = 0 . 1$ 1 a floor that maintains gradient flow and is shared with the pixel weighting in Eq. (8). We use intermediate temporal variation as a heuristic for potentially informative hardness, while the high-mean, low-variance factor suppresses persistent failures. Section IV-K tests this heuristic directly against update alignment. Under incomplete supervision the same statistics operate on $d _ { i } = 1 - \rho _ { i }$ without modification.

4) Pixel uncertainty weighting (PUE): Boundary and lowcontrast pixels are harder than homogeneous ones and their gradients can dominate early optimization. Unlike pseudolabel filtering [42], we down-weight uncertain pixels on the predicted mask itself without hiding them. For a logit $\hat { y } _ { h , w }$ with probability $p _ { h , w } = \sigma ( \hat { y } _ { h , w } )$ , the normalized entropy is

$$
E _ { h , w } = - p _ { h , w } \log _ { 2 } p _ { h , w } - ( 1 - p _ { h , w } ) \log _ { 2 } ( 1 - p _ { h , w } ) ,\tag{7}
$$

which peaks at $p _ { h , w } = 0 . 5$ and vanishes at certainty. The soft pixel weight at epoch t is formulated as:

$$
W _ { h , w } ( t ) = w _ { \mathrm { m i n } } + ( 1 - w _ { \mathrm { m i n } } ) \big ( 1 - \beta ( t ) E _ { h , w } \big ) ,\tag{8}
$$

with $w _ { \mathrm { m i n } }$ the shared non-zero floor of Eq. (6) and $\beta ( t ) =$ ma $\mathrm { x } ( 0 , 1 - t / T _ { c } )$ a curriculum-aware coefficient. The weight $W$ is detached from the graph, so Eq. (8) reweights the loss without back-propagating through the entropy-derived weights. Early on, high-entropy pixels are strongly down-weighted, and as $\beta ( t )$ decays to zero the pixel weights return to one and remain uniform throughout Phase II, while the floor keeps gradient flow from all pixels.

5) Unified curriculum loss: For a sample $i \in S _ { t }$ the loss combines the two granularities,

$$
\mathcal { L } _ { \mathrm { c u r r } } = \omega _ { i } \mathcal { L } _ { \mathrm { s e g } } ^ { w } ( \hat { y } _ { i } , t _ { i } , W _ { i } ) ,\tag{9}
$$

where $\mathcal { L } _ { \mathrm { s e g } } ^ { w }$ is the regime-appropriate weighted segmentation loss, a weighted BCE plus IoU under full supervision, the corresponding scribble, point, or box loss under weak supervision, and an entropy-gated pseudo-label loss for unlabeled samples, applied to the matching supervision-adaptive target $t _ { i } .$ Pixels without any supervision contribute zero to $\mathcal { L } _ { \mathrm { s e g } } ^ { w } .$ . Their predictions still enter the consistency and entropy cues of $\operatorname { E q . } \left( 2 \right)$ , so the curriculum remains well defined across regimes.

The same formulation applies across supervision sources within the same framework by re-instantiating $s _ { i } ^ { \mathrm { s u p } } , \ t _ { i } .$ , and $Q _ { \mathrm { s u p } } \mathrm { : }$ dense supervision uses ground-truth masks, sparse supervision uses cue evidence with the corresponding weak loss, and unlabeled samples use available EMA-teacher and frozen proposal evidence, and when both are present, their agreement is averaged and disagreement pixels are down-weighted. Trust follows the primary supervision source, with the supervisionspecific definitions given in the Supp.

6) Beyond binary masks: Multi-class and instance segmentation need no new component: the binary entropy becomes a normalized multi-class entropy, agreement is measured by class-averaged IoU or by Hungarian-matched mask IoU with an object-count term, and the degeneration penalty penalizes collapse to background, onto a single class, or in object count. The trust and loss coefficients $\tau _ { \mathrm { t r u s t } } , \varepsilon .$ , and $\lambda _ { \mathrm { a n c } }$ are shared, with fixed output-specific collapse checks detailed in the Supp., and Eqs. (3), (4) and (6) transfer unchanged: they operate on the scalar $d _ { i } = 1 - \rho _ { i }$ and never see the output format.

7) Cross-modal conditioning: Conditioning on a language prompt or a reference image adds one axis of unreliability: the mask may disagree with the conditioning even when the model is internally confident. We instantiate the existing cues rather than introducing new ones, leaving $\rho _ { i }$ and $Q ( m )$ unchanged and $s _ { i } ^ { \mathrm { s u p } }$ mask-level. Language conditioning uses promptparaphrase consistency, the agreement across semantically equivalent rephrasings, and vision-text disagreement, the gap to a vision-only prediction. Reference conditioning analogously uses reference-swap consistency and vision-reference disagreement. Conditioned predictions provide reliability evidence, not fixed training labels, while the trust level continues to follow the primary supervision source (Table I).

## D. Guarded Anti-Curriculum Promotion

Once SA-RCS stabilizes at epoch $T _ { c } ,$ we raise difficulty on purpose. The conference SBFT applied a single low-pass filter and fine-tuned against the mask. We extend it into a guarded post-training stage, SBFT-P, that creates a spectral challenge of several strengths, scores the resulting candidate predictions with a supervision-adaptive guard, and refines the model with a guarded preference objective. The guard is what keeps the stage from reducing to confidence maximization.

1) Multi-radius spectral challenge: We low-pass the input in the frequency domain with a circular mask,

$$
\widetilde { x } _ { i } ^ { 1 . 0 0 } = x _ { i } , \quad \widetilde { x } _ { i } ^ { r } = \mathcal { F } ^ { - 1 } \big ( \mathcal { F } ( x _ { i } ) \odot M _ { r } \big ) \mathrm { f o r } r < 1 ,
$$

$$
\begin{array} { r } { M _ { r } ( u , v ) = \mathbf { 1 } \Big [ ( u - c _ { u } ) ^ { 2 } + ( v - c _ { v } ) ^ { 2 } \leq \big ( \frac { r \operatorname* { m i n } ( H _ { s } , W _ { s } ) } { 2 } \big ) ^ { 2 } \Big ] , } \end{array}\tag{10}
$$

where $\mathcal { F }$ and ${ \mathcal { F } } ^ { - 1 }$ denote the centered Fourier transform and its inverse, $\left( c _ { u } , c _ { v } \right)$ is the frequency origin, and 1[·] is the indicator. We use a small set of radii $r \in \{ 0 . 9 0 , 0 . 9 5 , 1 . 0 0 \}$ The identity view $r = 1 . 0 0$ is the clean prediction by definition, while smaller radii progressively remove the high-frequency texture channel. Passing each view through the segmenter gives detached candidate masks $m _ { i } ^ { r } = \mathrm { s g } \big [ \sigma \big ( f _ { \theta } ( \tilde { x } _ { i } ^ { r } ) \big ) \big ]$ , where $\mathrm { s g } [ \cdot ]$ denotes stop-gradient. The thresholded masks $\mathbf { \bar { 1 } } [ m _ { i } ^ { r } \geq$ 0.5] enter the overlap and area checks of the guard, and the soft probabilities enter the entropy term and the preference objective. The three radii expose how predictions change as high-frequency content is progressively removed, which the guard turns into a robustness signal.

2) Supervision-adaptive guard: We score each candidate mask m with the following evaluation function:

$$
\begin{array} { r l } & { Q ( m ) = \tau _ { \mathrm { t r u s t } } Q _ { \mathrm { s u p } } ( m ) } \\ & { ~ + \frac { 1 - \tau _ { \mathrm { t r u s t } } } { 2 } \big [ Q _ { \mathrm { r o b } } ( m ) + Q _ { \mathrm { c o n s } } ( m ) \big ] } \\ & { ~ - \varepsilon \big [ Q _ { \mathrm { d e g } } ( m ) + Q _ { \mathrm { u n c } } ( m ) \big ] , } \end{array}\tag{11}
$$

whose terms are: a correctness term $Q _ { \mathrm { s u p } } ,$ which uses the same supervision source as $s _ { i } ^ { \mathrm { s u p } }$ in Eq. (2) (GT overlap, weak-cue satisfaction, or teacher/proposal agreement) but is evaluated on each candidate mask, a robustness term $Q _ { \mathrm { r o b } }$ , the mean agreement of m with the other candidates across radii, a consistency term $Q _ { \mathrm { { c o n s } } }$ , the agreement of $m$ with the cleanview prediction of a frozen reference, by default the Phase-I checkpoint $\theta _ { 0 } .$ , which anchors consistency to the Phase-I model instead of an evolving clean prediction, a degeneration penalty $Q _ { \mathrm { d e g } } ,$ , which discourages all-background masks, area collapse, and over-smoothing, and an uncertainty term $Q _ { \mathrm { u n c } }$ , the mean entropy. Positive weights sum to one and shift from correctness toward robustness and consistency as supervision becomes less verifiable, while $\varepsilon = 0 . 1$ is a fixed penalty scale shared across regimes. Detached candidates prevent gradients through selfgenerated targets. The preferred and dispreferred masks are $r _ { i } ^ { + } = \arg \operatorname* { m a x } _ { r } Q ( m _ { i } ^ { r } ) , r _ { i } ^ { - } = \arg \operatorname* { m i n } _ { r } Q ( m _ { i } ^ { r } ) , m _ { i } ^ { \pm } = m _ { i } ^ { r _ { i } ^ { \pm } }$ . (12)

Because Q is dominated by accuracy, stability under spectral perturbation, and non-degeneracy, with confidence entering only as a weak penalty, selection favors predictions supported by multiple cues, not confidence alone, and Table XXIV reports that this does not inflate confidence.

3) Guarded preference objective: We adopt a pairwise preference objective in the spirit of direct preference optimization [55], adapted to dense masks. Let

$$
\begin{array} { c l c r } { { \ell _ { \theta } ( m  { | } x ) = \frac { 1 } { | \Omega | } \displaystyle \sum _ { h , w } \bigl [ m _ { h , w } \log p _ { h , w } } } \\ { { } } & { { } } \\ { { } } & { { + \left( 1 - m _ { h , w } \right) \log ( 1 - p _ { h , w } ) \bigr ] } } \end{array}\tag{13}
$$

be the pixel-averaged log-likelihood of mask m under the model, with $p = \sigma ( f _ { \theta } ( x ) )$ , and let $\theta _ { 0 }$ be the reference saved at the end of SA-RCS. Writing $\Delta \ell _ { \theta } ( m ) = \ell _ { \theta } ( m ) - \ell _ { \theta _ { 0 } } ( m )$ for the log-likelihood margin of the model against the reference, the preference loss prefers $m ^ { + }$ over $m ^ { - }$

$$
\mathcal { L } _ { \mathrm { p r e f } } = - \log \sigma \big ( \Delta \ell _ { \theta } ( m ^ { + } ) - \Delta \ell _ { \theta } ( m ^ { - } ) \big ) ,\tag{14}
$$

evaluated on the clean input $x _ { i } ,$ not on a challenge view, so post-training transfers the preferred mask back onto the deployed inference path. Since $\ell _ { \theta }$ is affine in the target mask, the preference margin has the equivalent logit form

$$
\Delta \ell _ { \theta } ( m ^ { + } ) - \Delta \ell _ { \theta } ( m ^ { - } ) = \frac { 1 } { | \Omega | } \sum _ { j \in \Omega } ( m _ { j } ^ { + } - m _ { j } ^ { - } ) \Delta z _ { j } ,\tag{15}
$$

where $\Delta z _ { j } ~ = ~ z _ { \theta , j } - z _ { \theta _ { 0 } , j }$ denotes the clean-view logit difference, with $z _ { \theta } = f _ { \theta } ( x _ { i } )$ and Ω the pixel set. The objective favors higher relative logits where $m _ { j } ^ { + } > m _ { j } ^ { - }$ and lower ones where $m _ { j } ^ { + } < m _ { j } ^ { - }$ . A clean-view anchor keeps the model close to the frozen reference and limits preference-driven drift,

$$
\mathcal { L } _ { \mathrm { a n c } } = \mathrm { K L } \big ( p _ { \theta _ { 0 } } ( x _ { i } ) \| p _ { \theta } ( x _ { i } ) \big ) ,\tag{16}
$$

written with the frozen reference $\theta _ { 0 }$ as the teacher distribution, the standard reference-distillation direction. The conference SBFT term is retained,

$$
\mathcal { L } _ { \mathrm { S B F T } } = \mathcal { L } _ { \mathrm { B C E } } ^ { w } ( \tilde { p } _ { i } , t _ { i } ) + \mathcal { L } _ { \mathrm { I o U } } ^ { w } ( \tilde { p } _ { i } , t _ { i } ) , \tilde { p } _ { i } = \sigma \big ( f _ { \theta } ( \tilde { x } _ { i } ^ { 0 . 9 5 } ) \big ) ,\tag{17}
$$

where $t _ { i }$ is the supervision-adaptive target (the dense mask, or the guarded preferred mask $m _ { i } ^ { + }$ when no dense mask exists). The superscript w follows the pixel-weighted loss convention of Phase I, and because $\beta = 0$ in Phase II, $W _ { h , w } = 1$ for all pixels. The post-training loss is formulated as follows:

$$
\mathcal { L } _ { \mathrm { p o s t } } = \mathcal { L } _ { \mathrm { S B F T } } + \mathcal { L } _ { \mathrm { p r e f } } + \lambda _ { \mathrm { a n c } } \mathcal { L } _ { \mathrm { a n c } } ,\tag{18}
$$

where the preference and temperature scales are fixed to one and $\lambda _ { \mathrm { a n c } } = 0 . 1$ is a global anchor strength reused across all datasets and regimes. Under full supervision, dropping the preference and anchor terms, i.e., keeping only $\mathcal { L } _ { \mathrm { S B F T } }$ with the single radius $r = 0 . 9 5$ , recovers the conference SBFT objective. Without dense masks the target $t _ { i }$ is the guarded $m _ { i } ^ { + }$ , which the conference formulation did not define. The ablations in Table XXI isolate the contribution of each added term.

## E. Overall Training Pipeline

PaceShift leaves the architecture unchanged and adds no inference cost, with modest overhead incurred only during training and the full procedure listed in the Supp.

Overhead. Modest training overhead comes only from the periodic augmented and stochastic forward predictions that refresh the Phase-I reliability cues, and from the spectralcandidate and frozen-reference evaluations of Phase II. The temporal history stores one scalar per refresh checkpoint per sample in memory, and the reduced active subset in Phase I offsets part of the extra computation (Table III).

Hyperparameter footprint. All reliability and guard weights derive from the trust level $\tau _ { \mathrm { t r u s t } }$ of Table I. The mid-variance map is parameter-free, the preference and temperature scales are both fixed to one, β(t) follows the prescribed schedule, and $T _ { c }$ admits an adaptive trigger with comparable performance (Table XXII). The remaining global constants $p _ { \operatorname* { m i n } } , w _ { \operatorname* { m i n } } , \varepsilon$ and $\lambda _ { \mathrm { a n c } }$ are fixed once and reused unchanged across all tasks, backbones, and regimes, while H and the phase boundaries follow the common budget-scaling rule in the Supp. Sensitivity is reported in the Supp., and no per-dataset tuning is used.

## IV. EXPERIMENTS

## A. Experimental Setup

Tasks, datasets, and metrics. We group three core CECS tasks and the related polyp-segmentation task into the CECS suite: camouflaged object detection (COD) on CAMO [64], COD10K [1], CHAMELEON [65], and NC4K [66], polyp image segmentation (PIS) trained on Kvasir [67] and ClinicDB [68] and evaluated on CVC-ColonDB [69] and ETIS [70], transparent object detection (TOD) on GDD [19] and GSD [71], and zero-shot concealed defect detection (CDD) on CDS2K [13]. We further test broader generality across infrared small-target detection (IRSTD-1k [72]), shadow detection (SBU [73]), medical lesion and fluid segmentation (ISIC 2018 [74] and RETOUCH [75]), dichotomous and salient segmentation (DIS VD [76] and DUTS-TE [77]), and semantic (ADE20K [78]) and instance (COCO [79]) segmentation, and supervision generality under point and semi-supervised settings, foundationmodel adaptation, and language- and reference-conditioned segmentation. For COD, we report mean absolute error (M), adaptive F-measure $( F _ { \beta } )$ [80], mean E-measure $( E _ { \phi } )$ [81], and structure measure $( S _ { \alpha } )$ [82]. PIS uses mean Dice, mIoU, and $S _ { \alpha } ,$ , TOD uses mIoU, $F _ { \beta } ^ { \mathrm { m a x } }$ , and M, and CDD adds the mean F-measure $( F _ { \beta } ^ { m } )$ to the COD metrics. Other tasks use their standard metrics. We assess calibration using pixel-pooled ECE [83]. We also report a ground-truth-stratified confidence– accuracy gap (GT-CAG, the mean of the gaps computed on ground-truth foreground and background pixels), a boundary restricted gap $\mathrm { ( C A G _ { b n d } ) }$ , and WConf, the mean confidence on incorrectly predicted pixels. These three are complementary diagnostics of where confidence departs from accuracy, not measures of global calibration, since a calibrated predictor can have a nonzero stratified gap. Please see details in the Supp. Implementation. PaceShift is implemented in PyTorch with Adam, $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 9 9 )$ . All epoch-scale constants share a single dynamics horizon H = 10: an H-epoch warm-up without filtering, a difficulty checkpoint refreshed every H epochs, and temporal statistics computed over the Phase-I refresh checkpoints (five at the defaults). The curriculum phase covers epochs 1 to $T _ { c } = 6 0$ and the anti-curriculum post-training phase $T _ { c } + 1$ to $T = 7 0$ , and the percentile schedule grows from $p _ { \operatorname* { m i n } } = 0 . 6 \mathrm { t o } 1 . 0$ . Standard augmentations (random flip, crop, and photometric jitter) are shared by all baselines and by their PaceShift counterparts, so comparisons are controlled. Inputs are resized to 352×352 for the canonical CECS experiments, while cross-task experiments preserve each host’s official resolution, optimizer, learning rate, batch size, augmentation, and total training budget, with PaceShift phase boundaries scaled proportionally with that budget (see Supp.).

TABLE II  
CAMOUFLAGED OBJECT DETECTION. “+” AND “++” DENOTE TRAINING THE BASELINE UNDER CURRISEG [17] AND PACESHIFT. ∆: MEAN RELATIVE IMPROVEMENT OVER THE BASELINE ACROSS ALL METRICS AND DATASETS, WITH THE SIGN REVERSED FOR LOWER-IS-BETTER METRICS SO THAT POSITIVE VALUES INDICATE IMPROVEMENT. BOLD: BEST WITHIN EACH BACKBONE BLOCK. <sup>†</sup>: OUR DINO V2-L RETRAINING OF VSCODE UNDER THE PROTOCOL OF THIS TABLE. <sup>‡</sup>: EVALUATED BY US FROM THE AUTHORS’ RELEASED MODEL UNDER THE PROTOCOL OF THIS TABLE.
<table><tr><td rowspan=1 colspan=18>CHAMELEON                CAMO                  COD10K                  NC4KMethods        Backbone                                                                                                             ∆(%)</td></tr><tr><td rowspan=1 colspan=18>M↓  $F _ { \beta } \uparrow$    $E _ { \phi } \uparrow$    $S _ { \alpha } \uparrow$  M↓  $F _ { \beta } \uparrow$    $E _ { \phi } \uparrow$   $S _ { \alpha } \uparrow$  M↓  $F _ { \beta } \uparrow$    $E _ { \phi } \uparrow$    $S _ { \alpha } \uparrow$  M↓  $F _ { \beta } \uparrow$    $E _ { \phi } \uparrow$    $S _ { \alpha } \uparrow$ </td></tr><tr><td rowspan=1 colspan=3>MGL-R [58]     ResNet50  0.0310.825</td><td rowspan=1 colspan=1>0.917</td><td rowspan=1 colspan=2>0.8910.088</td><td rowspan=1 colspan=10>0.7380.8120.7750.0350.6800.8510.8140.0530.7780.867</td><td rowspan=1 colspan=1>0.833</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>PreyNet [59]     ResNet50   0.027</td><td rowspan=1 colspan=1>0.844</td><td rowspan=1 colspan=1>0.948</td><td rowspan=1 colspan=1>0.895</td><td rowspan=1 colspan=1>0.077</td><td rowspan=1 colspan=1>0.763</td><td rowspan=1 colspan=1>0.854</td><td rowspan=1 colspan=1>0.790</td><td rowspan=1 colspan=1>0.034</td><td rowspan=1 colspan=1>0.715</td><td rowspan=1 colspan=2>0.8940.813</td><td rowspan=1 colspan=1>0.047</td><td rowspan=1 colspan=1>0.798</td><td rowspan=1 colspan=1>0.887</td><td rowspan=1 colspan=1>0.838</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>FEDER [10]     ResNet50   0.028</td><td rowspan=1 colspan=1>0.850</td><td rowspan=1 colspan=1>0.944</td><td rowspan=1 colspan=1>0.892</td><td rowspan=1 colspan=1>0.070</td><td rowspan=1 colspan=1>0.775</td><td rowspan=1 colspan=1>0.870</td><td rowspan=1 colspan=1>0.802</td><td rowspan=1 colspan=1>0.032</td><td rowspan=1 colspan=1>0.715</td><td rowspan=1 colspan=2>0.8920.810</td><td rowspan=1 colspan=1>0.046</td><td rowspan=1 colspan=1>0.808</td><td rowspan=1 colspan=1>0.900</td><td rowspan=1 colspan=1>0.842</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>FEDER+ [17]    ResNet50  0.026</td><td rowspan=1 colspan=1>0.858</td><td rowspan=1 colspan=1>0.952</td><td rowspan=1 colspan=1>0.898</td><td rowspan=1 colspan=1>0.068</td><td rowspan=1 colspan=1>0.790</td><td rowspan=1 colspan=1>0.881</td><td rowspan=1 colspan=1>0.807</td><td rowspan=1 colspan=1>0.030</td><td rowspan=1 colspan=1>0.736</td><td rowspan=1 colspan=2>0.9100.818</td><td rowspan=1 colspan=1>0.043</td><td rowspan=1 colspan=1>0.825</td><td rowspan=1 colspan=1>0.912</td><td rowspan=1 colspan=1>0.850</td><td rowspan=1 colspan=1>2.46 ↑</td></tr><tr><td rowspan=1 colspan=2>FEDER++ (Ours)ResNet50  0.025</td><td rowspan=1 colspan=1>0.863</td><td rowspan=1 colspan=1>0.957</td><td rowspan=1 colspan=1>0.898</td><td rowspan=1 colspan=1>0.066</td><td rowspan=1 colspan=1>0.797</td><td rowspan=1 colspan=1>0.889</td><td rowspan=1 colspan=1>0.810</td><td rowspan=1 colspan=2>0.0290.742</td><td rowspan=1 colspan=3>0.9170.8200.042</td><td rowspan=1 colspan=1>0.830</td><td rowspan=1 colspan=1>0.916</td><td rowspan=1 colspan=1>0.851</td><td rowspan=1 colspan=1>3.59 ↑</td></tr><tr><td rowspan=1 colspan=2>FSEL [11]      ResNet50   0.029</td><td rowspan=1 colspan=1>0.847</td><td rowspan=1 colspan=2>0.9410.893</td><td rowspan=1 colspan=1>0.069</td><td rowspan=1 colspan=3>0.7790.8810.816</td><td rowspan=1 colspan=2>0.0320.722</td><td rowspan=1 colspan=2>0.8910.822</td><td rowspan=1 colspan=1>0.045</td><td rowspan=1 colspan=1>0.807</td><td rowspan=1 colspan=1>0.901</td><td rowspan=1 colspan=1>0.847</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>FSEL+ [17]     ResNet50  0.028</td><td rowspan=1 colspan=1>0.856</td><td rowspan=1 colspan=2>0.9490.898</td><td rowspan=1 colspan=1>0.067</td><td rowspan=1 colspan=3>0.7920.8890.819</td><td rowspan=1 colspan=2>0.0300.742</td><td rowspan=1 colspan=2>0.9090.831</td><td rowspan=1 colspan=1>0.042</td><td rowspan=1 colspan=1>0.823</td><td rowspan=1 colspan=1>0.919</td><td rowspan=1 colspan=1>0.855</td><td rowspan=1 colspan=1>2.22 ↑</td></tr><tr><td rowspan=1 colspan=2>FSEL++ (Ours)  ResNet50  0.027</td><td rowspan=1 colspan=1>0.863</td><td rowspan=1 colspan=3>0.9530.9000.066</td><td rowspan=1 colspan=3>0.7950.8940.824</td><td rowspan=1 colspan=2>0.0290.746</td><td rowspan=1 colspan=2>0.9160.833</td><td rowspan=1 colspan=1>0.041</td><td rowspan=1 colspan=1>0.831</td><td rowspan=1 colspan=1>0.927</td><td rowspan=1 colspan=1>0.857</td><td rowspan=1 colspan=1>3.28 ↑</td></tr><tr><td rowspan=1 colspan=2>RUN [12]       ResNet50  0.027</td><td rowspan=1 colspan=1>0.855</td><td rowspan=1 colspan=3>0.9520.8950.070</td><td rowspan=1 colspan=3>0.7810.8680.806</td><td rowspan=1 colspan=2>0.0300.747</td><td rowspan=1 colspan=2>0.9030.827</td><td rowspan=1 colspan=1>0.042</td><td rowspan=1 colspan=1>0.824</td><td rowspan=1 colspan=1>0.908</td><td rowspan=1 colspan=1>0.851</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>RUN+ [17]      ResNet50  0.025</td><td rowspan=1 colspan=9>0.8630.9600.9000.0680.8010.8790.8130.0290.765</td><td rowspan=1 colspan=3>0.9170.8360.040</td><td rowspan=1 colspan=1>0.835</td><td rowspan=1 colspan=1>0.921</td><td rowspan=1 colspan=1>0.858</td><td rowspan=1 colspan=1>2.13 ↑</td></tr><tr><td rowspan=1 colspan=1>RUN++ (Ours)</td><td rowspan=1 colspan=1>ResNet50  0.023</td><td rowspan=1 colspan=1>0.870</td><td rowspan=1 colspan=1>0.967</td><td rowspan=1 colspan=1>0.902</td><td rowspan=1 colspan=2>0.0660.808</td><td rowspan=1 colspan=2>0.8860.819</td><td rowspan=1 colspan=1>0.027</td><td rowspan=1 colspan=1>0.775</td><td rowspan=1 colspan=3>0.9260.8380.038</td><td rowspan=1 colspan=1>0.841</td><td rowspan=1 colspan=1>0.930</td><td rowspan=1 colspan=1>0.862</td><td rowspan=1 colspan=1>4.04 ↑</td></tr><tr><td rowspan=1 colspan=1>NUN [14]</td><td rowspan=1 colspan=1>ResNet50  0.026</td><td rowspan=1 colspan=1>0.862</td><td rowspan=1 colspan=1>0.955</td><td rowspan=1 colspan=1>0.899</td><td rowspan=1 colspan=1>0.069</td><td rowspan=1 colspan=1>0.788</td><td rowspan=1 colspan=1>0.890</td><td rowspan=1 colspan=1>0.818</td><td rowspan=1 colspan=1>0.029</td><td rowspan=1 colspan=1>0.759</td><td rowspan=1 colspan=2>0.9080.831</td><td rowspan=1 colspan=1>0.042</td><td rowspan=1 colspan=1>0.828</td><td rowspan=1 colspan=1>0.913</td><td rowspan=1 colspan=1>0.855</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>NUN+ [17]</td><td rowspan=1 colspan=1>ResNet50  0.025</td><td rowspan=1 colspan=1>0.868</td><td rowspan=1 colspan=1>0.960</td><td rowspan=1 colspan=1>0.903</td><td rowspan=1 colspan=1>0.068</td><td rowspan=1 colspan=1>0.797</td><td rowspan=1 colspan=1>0.898</td><td rowspan=1 colspan=1>0.824</td><td rowspan=1 colspan=1>0.028</td><td rowspan=1 colspan=1>0.765</td><td rowspan=1 colspan=2>0.9150.835</td><td rowspan=1 colspan=2>0.0410.833</td><td rowspan=1 colspan=1>0.917</td><td rowspan=1 colspan=1>0.858</td><td rowspan=1 colspan=1>1.19 ↑</td></tr><tr><td rowspan=1 colspan=2>NUN++ (Ours)  ResNet50  0.023</td><td rowspan=1 colspan=1>0.881</td><td rowspan=1 colspan=1>0.972</td><td rowspan=1 colspan=1>0.908</td><td rowspan=1 colspan=1>0.066</td><td rowspan=1 colspan=1>0.809</td><td rowspan=1 colspan=2>0.9100.830</td><td rowspan=1 colspan=5>0.0260.7780.9270.8410.039</td><td rowspan=1 colspan=1>0.842</td><td rowspan=1 colspan=1>0.922</td><td rowspan=1 colspan=1>0.866</td><td rowspan=1 colspan=1>3.41 ↑</td></tr><tr><td rowspan=1 colspan=3>BSA-Net [60]    Res2Net50 0.0270.851</td><td rowspan=1 colspan=1>0.946</td><td rowspan=1 colspan=1>0.895</td><td rowspan=1 colspan=1>0.079</td><td rowspan=1 colspan=1>0.768</td><td rowspan=1 colspan=1>0.851</td><td rowspan=1 colspan=1>0.796</td><td rowspan=1 colspan=1>0.034</td><td rowspan=1 colspan=1>0.723</td><td rowspan=1 colspan=2>0.8910.818</td><td rowspan=1 colspan=1>0.048</td><td rowspan=1 colspan=1>0.805</td><td rowspan=1 colspan=1>0.897</td><td rowspan=1 colspan=1>0.841</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>RUN [12]</td><td rowspan=1 colspan=1>Res2Net50 0.024</td><td rowspan=1 colspan=1>0.879</td><td rowspan=1 colspan=1>0.956</td><td rowspan=1 colspan=1>0.907</td><td rowspan=1 colspan=1>0.066</td><td rowspan=1 colspan=1>0.815</td><td rowspan=1 colspan=1>0.905</td><td rowspan=1 colspan=1>0.843</td><td rowspan=1 colspan=1>0.028</td><td rowspan=1 colspan=1>0.764</td><td rowspan=1 colspan=1>0.914</td><td rowspan=1 colspan=1>0.849</td><td rowspan=1 colspan=1>0.041</td><td rowspan=1 colspan=1>0.830</td><td rowspan=1 colspan=1>0.917</td><td rowspan=1 colspan=1>0.859</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>RUN+ [17]</td><td rowspan=1 colspan=1>Res2Net50 0.023</td><td rowspan=1 colspan=1>0.891</td><td rowspan=1 colspan=1>0.963</td><td rowspan=1 colspan=1>0.911</td><td rowspan=1 colspan=1>0.065</td><td rowspan=1 colspan=1>0.820</td><td rowspan=1 colspan=1>0.912</td><td rowspan=1 colspan=1>0.845</td><td rowspan=1 colspan=1>0.026</td><td rowspan=1 colspan=1>0.785</td><td rowspan=1 colspan=1>0.933</td><td rowspan=1 colspan=1>0.857</td><td rowspan=1 colspan=1>0.038</td><td rowspan=1 colspan=1>0.852</td><td rowspan=1 colspan=1>0.932</td><td rowspan=1 colspan=1>0.870</td><td rowspan=1 colspan=1>2.23 ↑</td></tr><tr><td rowspan=1 colspan=1>RUN++ (Ours)</td><td rowspan=1 colspan=1>Res2Net50 0.022</td><td rowspan=1 colspan=1>0.896</td><td rowspan=1 colspan=1>0.968</td><td rowspan=1 colspan=1>0.913</td><td rowspan=1 colspan=1>0.064</td><td rowspan=1 colspan=1>0.830</td><td rowspan=1 colspan=1>0.922</td><td rowspan=1 colspan=1>0.849</td><td rowspan=1 colspan=1>0.025</td><td rowspan=1 colspan=1>0.792</td><td rowspan=1 colspan=1>0.938</td><td rowspan=1 colspan=1>0.861</td><td rowspan=1 colspan=1>0.037</td><td rowspan=1 colspan=1>0.859</td><td rowspan=1 colspan=1>0.936</td><td rowspan=1 colspan=1>0.873</td><td rowspan=1 colspan=1>3.44↑</td></tr><tr><td rowspan=1 colspan=1>NUN [14]</td><td rowspan=1 colspan=1>Res2Net50 0.023</td><td rowspan=1 colspan=1>0.883</td><td rowspan=1 colspan=1>0.959</td><td rowspan=1 colspan=1>0.910</td><td rowspan=1 colspan=1>0.066</td><td rowspan=1 colspan=1>0.822</td><td rowspan=1 colspan=1>0.909</td><td rowspan=1 colspan=1>0.846</td><td rowspan=1 colspan=1>0.026</td><td rowspan=1 colspan=1>0.781</td><td rowspan=1 colspan=1>0.920</td><td rowspan=1 colspan=1>0.852</td><td rowspan=1 colspan=1>0.040</td><td rowspan=1 colspan=1>0.836</td><td rowspan=1 colspan=1>0.922</td><td rowspan=1 colspan=1>0.864</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>NUN+ [17]</td><td rowspan=1 colspan=1>Res2Net50 0.022</td><td rowspan=1 colspan=1>0.889</td><td rowspan=1 colspan=1>0.967</td><td rowspan=1 colspan=1>0.915</td><td rowspan=1 colspan=1>0.064</td><td rowspan=1 colspan=1>0.827</td><td rowspan=1 colspan=1>0.915</td><td rowspan=1 colspan=1>0.849</td><td rowspan=1 colspan=1>0.025</td><td rowspan=1 colspan=1>0.789</td><td rowspan=1 colspan=1>0.927</td><td rowspan=1 colspan=1>0.856</td><td rowspan=1 colspan=1>0.038</td><td rowspan=1 colspan=1>0.843</td><td rowspan=1 colspan=1>0.928</td><td rowspan=1 colspan=1>0.867</td><td rowspan=1 colspan=1>1.50 ↑</td></tr><tr><td rowspan=1 colspan=2>NUN++ (Ours)   Res2Net50 0.021</td><td rowspan=1 colspan=1>0.895</td><td rowspan=1 colspan=1>0.973</td><td rowspan=1 colspan=1>0.919</td><td rowspan=1 colspan=1>0.062</td><td rowspan=1 colspan=1>0.834</td><td rowspan=1 colspan=1>0.923</td><td rowspan=1 colspan=1>0.853</td><td rowspan=1 colspan=1>0.023</td><td rowspan=1 colspan=1>0.797</td><td rowspan=1 colspan=2>0.9340.862</td><td rowspan=1 colspan=1>0.035</td><td rowspan=1 colspan=1>0.860</td><td rowspan=1 colspan=1>0.938</td><td rowspan=1 colspan=1>0.872</td><td rowspan=1 colspan=1>3.54 ↑</td></tr><tr><td rowspan=1 colspan=2>CamoDiff [61]   PVT V2   0.022</td><td rowspan=1 colspan=1>0.868</td><td rowspan=1 colspan=1>0.952</td><td rowspan=1 colspan=1>0.908</td><td rowspan=1 colspan=1>0.042</td><td rowspan=1 colspan=1>0.853</td><td rowspan=1 colspan=2>0.9360.878</td><td rowspan=1 colspan=1>0.019</td><td rowspan=1 colspan=1>0.815</td><td rowspan=1 colspan=2>0.9430.883</td><td rowspan=1 colspan=1>0.028</td><td rowspan=1 colspan=1>0.858</td><td rowspan=1 colspan=1>0.942</td><td rowspan=1 colspan=1>0.895</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>RUN [12]</td><td rowspan=1 colspan=1>PVT V2   0.021</td><td rowspan=1 colspan=1>0.877</td><td rowspan=1 colspan=1>0.958</td><td rowspan=1 colspan=1>0.916</td><td rowspan=1 colspan=1>0.045</td><td rowspan=1 colspan=1>0.861</td><td rowspan=1 colspan=1>0.934</td><td rowspan=1 colspan=1>0.877</td><td rowspan=1 colspan=1>0.021</td><td rowspan=1 colspan=1>0.810</td><td rowspan=1 colspan=1>0.941</td><td rowspan=1 colspan=1>0.878</td><td rowspan=1 colspan=1>0.030</td><td rowspan=1 colspan=1>0.868</td><td rowspan=1 colspan=1>0.940</td><td rowspan=1 colspan=1>0.892</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>RUN+ [17]</td><td rowspan=1 colspan=1>PVT V2   0.019</td><td rowspan=1 colspan=1>0.893</td><td rowspan=1 colspan=1>0.971</td><td rowspan=1 colspan=1>0.922</td><td rowspan=1 colspan=1>0.042</td><td rowspan=1 colspan=1>0.879</td><td rowspan=1 colspan=1>0.952</td><td rowspan=1 colspan=1>0.885</td><td rowspan=1 colspan=1>0.018</td><td rowspan=1 colspan=1>0.828</td><td rowspan=1 colspan=1>0.957</td><td rowspan=1 colspan=1>0.886</td><td rowspan=1 colspan=1>0.026</td><td rowspan=1 colspan=1>0.889</td><td rowspan=1 colspan=1>0.958</td><td rowspan=1 colspan=1>0.903</td><td rowspan=1 colspan=1>3.94 ↑</td></tr><tr><td rowspan=1 colspan=1>RUN++ (Ours)</td><td rowspan=1 colspan=1>PVT V2   0.018</td><td rowspan=1 colspan=1>0.899</td><td rowspan=1 colspan=1>0.975</td><td rowspan=1 colspan=1>0.925</td><td rowspan=1 colspan=1>0.041</td><td rowspan=1 colspan=1>0.885</td><td rowspan=1 colspan=1>0.958</td><td rowspan=1 colspan=1>0.888</td><td rowspan=1 colspan=1>0.017</td><td rowspan=1 colspan=1>0.836</td><td rowspan=1 colspan=1>0.963</td><td rowspan=1 colspan=1>0.889</td><td rowspan=1 colspan=1>0.025</td><td rowspan=1 colspan=1>0.896</td><td rowspan=1 colspan=1>0.967</td><td rowspan=1 colspan=1>0.908</td><td rowspan=1 colspan=1>5.34 ↑</td></tr><tr><td rowspan=1 colspan=1>NUN [14]</td><td rowspan=1 colspan=1>PVT V2   0.021</td><td rowspan=1 colspan=1>0.888</td><td rowspan=1 colspan=1>0.963</td><td rowspan=1 colspan=1>0.918</td><td rowspan=1 colspan=1>0.042</td><td rowspan=1 colspan=1>0.868</td><td rowspan=1 colspan=1>0.939</td><td rowspan=1 colspan=1>0.885</td><td rowspan=1 colspan=2>0.0180.839</td><td rowspan=1 colspan=3>0.9480.8900.027</td><td rowspan=1 colspan=1>0.880</td><td rowspan=1 colspan=1>0.947</td><td rowspan=1 colspan=1>0.897</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>NUN+ [17]</td><td rowspan=1 colspan=1>PVT V2   0.020</td><td rowspan=1 colspan=1>0.897</td><td rowspan=1 colspan=1>0.970</td><td rowspan=1 colspan=1>0.925</td><td rowspan=1 colspan=1>0.040</td><td rowspan=1 colspan=1>0.879</td><td rowspan=1 colspan=1>0.951</td><td rowspan=1 colspan=1>0.892</td><td rowspan=1 colspan=2>0.0170.845</td><td rowspan=1 colspan=4>0.9520.8910.0260.885</td><td rowspan=1 colspan=1>0.951</td><td rowspan=1 colspan=1>0.898</td><td rowspan=1 colspan=1>1.69 ↑</td></tr><tr><td rowspan=1 colspan=3>NUN++ (Ours)  PVT V2   0.0180.908</td><td rowspan=1 colspan=1>0.980</td><td rowspan=1 colspan=12>0.9300.0380.8870.9680.8990.0160.8570.9620.8940.0240.8970.960</td><td rowspan=1 colspan=1>0.905</td><td rowspan=1 colspan=1>4.16 ↑</td></tr><tr><td rowspan=1 colspan=3>VSCode† [62]   DINO V2-L0.0200.882</td><td rowspan=1 colspan=15>0.9540.9150.0430.8600.9310.8820.0200.8180.9350.8800.0300.8670.9400.895</td></tr><tr><td rowspan=1 colspan=1>C3Net‡ [63]</td><td rowspan=1 colspan=1>DINO V2-L0.016</td><td rowspan=1 colspan=1>0.905</td><td rowspan=1 colspan=1>0.968</td><td rowspan=1 colspan=12>0.9270.0310.8920.9530.9040.0160.8520.9610.8980.0240.8940.958</td><td rowspan=1 colspan=1>0.913</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>NUN [14]</td><td rowspan=1 colspan=1>DINO V2-L0.015</td><td rowspan=1 colspan=1>0.918</td><td rowspan=1 colspan=1>0.975</td><td rowspan=1 colspan=12>0.9310.0280.9080.9650.9110.0140.8800.9720.9050.0200.9130.967</td><td rowspan=1 colspan=1>0.920</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>NUN+ [17]</td><td rowspan=1 colspan=1>DINO V2-L0.014</td><td rowspan=1 colspan=1>0.922</td><td rowspan=1 colspan=1>0.978</td><td rowspan=1 colspan=12>0.9320.0270.9120.9700.9130.0130.8860.9760.9080.0180.9190.973</td><td rowspan=1 colspan=1>0.923</td><td rowspan=1 colspan=1>2.03 ↑</td></tr><tr><td rowspan=1 colspan=3>NUN++ (Ours)   DINO V2-L0.0140.929</td><td rowspan=1 colspan=15>0.9820.9350.0260.9180.9770.9160.0120.8910.9820.9100.0160.9270.9780.9253.71 ↑</td></tr></table>

TABLE III

TRAINING COST UNDER FULL SUPERVISION (RESNET50, BATCH SIZE 2, SINGLE RTX 4090). VANILLA BASELINES FOLLOW OFFICIAL SCHEDULES (E.G., FEDER: 100 EPOCHS), WHILE CURRISEG AND PACESHIFT USE 70 EPOCHS. RELATIVE TIME/EPOCH IS WALL-CLOCK TIME PER EPOCH NORMALIZED BY THE CORRESPONDING VANILLA BASELINE. PEAK MEMORY USES MAX\_M EMOR Y\_A LLOC A T E D, WITH SBFT-P VIEWS PROCESSED SEQUENTIALLY AND THE FROZEN REFERENCE RESIDENT IN PHASE II. ∆ IS THE MEAN RELATIVE ACCURACY GAIN FROM TABLE II.
<table><tr><td>Methods</td><td>Training Time (h) Relative Time/Epoch</td><td>GPU Memory (G)</td><td>∆(%)</td></tr><tr><td>FEDER</td><td>9.62</td><td>1.00× 1.53</td><td></td></tr><tr><td>FEDER+</td><td>6.84</td><td>1.02× 1.62</td><td>2.46↑</td></tr><tr><td>FEDER++</td><td>7.95</td><td>1.18× 1.81</td><td>3.59↑</td></tr><tr><td>FSEL</td><td>11.54</td><td>1.00× 2.83</td><td></td></tr><tr><td>FSEL+</td><td>5.96</td><td>1.03× 2.92</td><td>2.22↑</td></tr><tr><td>FSEL++</td><td>6.93</td><td>1.20× 3.15</td><td>3.28↑</td></tr><tr><td>RUN</td><td>12.64</td><td>1.00× 3.66</td><td></td></tr><tr><td>RUN+</td><td>8.32</td><td>1.03× 3.75</td><td>2.13↑</td></tr><tr><td>RUN++</td><td>9.68</td><td>1.20× 4.02</td><td>4.04↑</td></tr></table>

## B. Comparative Evaluation on CECS

Camouflaged object detection. Table II inserts CurriSeg and PaceShift into four detectors across four backbones and compares them with graph-based, attention-based, transformer, and diffusion detectors. PaceShift improves or matches every detector on every dataset, with mean relative gains of 3.3%– 5.3%, and NUN++ with DINO V2-L achieves the strongest overall performance. Fig. 4 shows more faithful masks and fewer texture-driven false positives in highly concealed scenes. Table III reports training cost under each recipe’s actual schedule. PaceShift increases average per-epoch cost by about 20%, with most overhead concentrated in the ten Phase-II epochs. Its reported total wall-clock time is nevertheless lower because the curriculum recipes use 70 epochs whereas the vanilla baselines follow their longer official schedules. For example, FEDER uses 100 epochs, while FEDER+ and FEDER++ use 70 and achieve higher accuracy. Inference cost is unchanged. Table IV reports five paired runs on COD10K. Both CurriSeg (+) and PaceShift (++) improve the vanilla baseline at $p < 0 . 0 0 1$ on all three metrics. On the same paired runs, PaceShift further improves over CurriSeg on $F _ { \beta }$ and $E _ { \phi }$ for all three detectors $( p < 0 . 0 1 )$ , whereas the smaller $S _ { \alpha }$ gains (0.002) are not statistically significant.

Polyp, transparent, and concealed defect. Tables V to VII cover the remaining three CECS tasks. On PIS and TOD, PaceShift improves both a task-specific detector, MEGANet [8] and RFENet [89] respectively, and the RUN segmenter, and it consistently exceeds the conference recipe. On CDD, where a COD-trained model is transferred to CDS2K without adaptation, the relative gains are the largest of the four tasks, roughly 7.7% to 7.9%, and the gain of PaceShift substantially exceeds that of CurriSeg. Fig. 4 gives qualitative comparisons on all four tasks.

![](images/ee472f30ec20201066693cbd06e334d392f2884edda59556e511078121c2b4a5.jpg)  
(a) Camouflaged Object Detection

![](images/ef96fab984e53f4320e0e5087a5f5f68ea87adfe8a7754ddcbb24d1ccd497c9b.jpg)  
(b) Polyp Image Segmentation

![](images/d3a864ad21a10300bd2a274c0552dd976d85679facbb99d099481bceb5d32e9f.jpg)  
(c) Transparent Object Detection

![](images/66ca0ac2bccf213fa7d0f9e26948a81d9032838b161327d14fb448acb786735c.jpg)  
(d) Concealed Defect Detection

Fig. 4. Visual comparisons on COD, PIS, TOD, and CDD using the RUN baseline and its CurriSeg- and PaceShift-trained variants. PaceShift better recovers concealed structures and boundaries, with fewer false positives in texture-rich backgrounds.  
TABLE IV  
STATISTICAL SIGNIFICANCE ON COD10K (RESNET50), MEAN±STD OVER FIVE PAIRED RUNS WITH SHARED SEEDS. TWO-SIDED PAIRED t-TESTS REPORT THE MAXIMUM p-VALUE OVER $F _ { \beta } ,$ $E _ { \phi } ,$ AND $S _ { \alpha } .$ . TESTS OF ++ AGAINST + ARE REPORTED IN THE TEXT.
<table><tr><td>Methods</td><td> $F _ { \beta } \uparrow$ </td><td> $E _ { \phi } \uparrow$ </td><td> $S _ { \alpha } \uparrow$ </td><td>max. p (vs. base)</td></tr><tr><td>FEDER</td><td> $0 . 7 1 5 { \pm } . 0 0 4$ </td><td> $0 . 8 9 2 \pm . 0 0 3$ </td><td> $0 . 8 1 0 \pm . 0 0 2$ </td><td></td></tr><tr><td>FEDER+</td><td> $0 . 7 3 6 \pm . 0 0 3$ </td><td> $0 . 9 1 0 { \scriptstyle \pm . 0 0 3 }$ </td><td> $0 . 8 1 8 \pm . 0 0 2$ </td><td> $< 0 . 0 0 1$ </td></tr><tr><td>FEDER++</td><td> $\mathbf { 0 . 7 4 2 \pm . 0 0 3 }$ </td><td> ${ \bf 0 . 9 1 7 \pm . 0 0 2 }$ </td><td> $\mathbf { 0 . 8 2 0 \pm . 0 0 1 }$ </td><td>&lt;0.001</td></tr><tr><td>FSEL</td><td> $0 . 7 2 2 \pm . 0 0 4$ </td><td> $0 . 8 9 1 \pm . 0 0 3$ </td><td> $0 . 8 2 2 \pm . 0 0 2$ </td><td></td></tr><tr><td>FSEL+</td><td> $0 . 7 4 2 \pm . 0 0 3$ </td><td> $0 . 9 0 9 \pm . 0 0 2$ </td><td> $0 . 8 3 1 \pm . 0 0 1$ </td><td>&lt;0.001</td></tr><tr><td>FSEL++</td><td> $\mathbf { 0 . 7 4 6 \pm . 0 0 3 }$ </td><td> $\mathbf { 0 . 9 1 6 } \pm . 0 0 2$ </td><td> $\mathbf { 0 . 8 3 3 \pm . 0 0 1 }$ </td><td>&lt;0.001</td></tr><tr><td>RUN</td><td> $0 . 7 4 7 \pm . 0 0 4$ </td><td> $0 . 9 0 3 \pm . 0 0 3$ </td><td> $0 . 8 2 7 \pm . 0 0 2$ </td><td></td></tr><tr><td>RUN+</td><td> $0 . 7 6 5 \pm . 0 0 3$ </td><td> $0 . 9 1 7 { \scriptstyle \pm . 0 0 2 }$ </td><td> $0 . 8 3 6 \pm . 0 0 1$ </td><td>&lt;0.001</td></tr><tr><td>RUN++</td><td> $\mathbf { 0 . 7 7 5 \pm . 0 0 2 }$ </td><td> $\mathbf { 0 . 9 2 6 } \pm . 0 0 2$ </td><td> $\mathbf { 0 . 8 3 8 \pm . 0 0 1 }$ </td><td>&lt;0.001</td></tr></table>

TABLE V

POLYP IMAGE SEGMENTATION (PVT V2 BACKBONE).
<table><tr><td rowspan="2">Methods</td><td colspan="3">CVC-ColonDB</td><td colspan="3">ETIS</td><td rowspan="2">PIS ∆(%)</td></tr><tr><td>mDice↑</td><td>mIoU↑</td><td>Sα↑</td><td>mDice↑</td><td>mIoU↑</td><td> $S _ { \alpha } \uparrow$ </td></tr><tr><td>PolypPVT [20]</td><td>0.808</td><td>0.727</td><td>0.865</td><td>0.787</td><td>0.706</td><td>0.871</td><td></td></tr><tr><td>CoInNet [84]</td><td>0.797</td><td>0.729</td><td>0.875</td><td>0.759</td><td>0.690</td><td>0.859</td><td></td></tr><tr><td>LSSNet [85]</td><td>0.820</td><td>0.741</td><td>0.867</td><td>0.779</td><td>0.701</td><td>0.867</td><td></td></tr><tr><td>MEGANet [8]</td><td>0.793</td><td>0.714</td><td>0.854</td><td>0.739</td><td>0.665</td><td>0.836</td><td></td></tr><tr><td>MEGANet+ [17]</td><td>0.815</td><td>0.736</td><td>0.866</td><td>0.755</td><td>0.682</td><td>0.848</td><td>2.24 ↑</td></tr><tr><td>MEGANet++ (Ours)</td><td>0.821</td><td>0.743</td><td>0.869</td><td>0.765</td><td>0.688</td><td>0.851</td><td>3.02 ↑</td></tr><tr><td>RUN [12]</td><td>0.822</td><td>0.742</td><td>0.880</td><td>0.788</td><td>0.709</td><td>0.878</td><td></td></tr><tr><td>RUN+ [17]</td><td>0.845</td><td>0.763</td><td>0.893</td><td>0.806</td><td>0.720</td><td>0.890</td><td>2.05↑</td></tr><tr><td>RUN++ (Ours)</td><td>0.852</td><td>0.772</td><td>0.896</td><td>0.815</td><td>0.728</td><td>0.894</td><td>2.91 ↑</td></tr></table>

## C. Generalization across Architectures

PaceShift replaces a training schedule, not a module, so it should compose with any segmenter. Table VIII attaches the identical recipe, with identical global constants, to thirteen configurations of twelve detectors spanning encoder-decoder, graph, bio-inspired, frequency, attention, uncertainty, diffusion, unfolding, and foundation-model designs. Each baseline keeps its own backbone, optimizer, and augmentation, and only the training recipe changes. Six rows reuse runs reported elsewhere: FEDER, FSEL, RUN, and the two NUN rows are the COD10K results of the canonical runs in Table II, and the SAM-Adapter row repeats Table XIV. The remaining seven detectors are trained under their native epoch budgets with the phase boundaries scaled as in the Supp. Three observations follow. (1) Every configuration improves, within a band of 3.8 to 6.5 percent, and PaceShift exceeds CurriSeg on all thirteen rows. (2) It is complementary to architectural modeling of the very quantities it controls: UGTR already reasons about uncertainty inside the network and is nonetheless among the larger beneficiaries, and FSEL, which models frequency, still gains, because such a module changes how features are computed while the schedule changes what training signal the optimizer receives. (3) The gains persist under the adaptation regime that is replacing task-specific training: DINO V2 and the frozen SAM adapter improve by 4.3 and 4.7 percent without any change to their adapters or frozen weights.

TABLE VI  
TRANSPARENT OBJECT DETECTION (PVT V2 BACKBONE).
<table><tr><td rowspan="2">Methods</td><td colspan="3">GDD</td><td colspan="3">GSD</td><td rowspan="2">TOD ∆(%)</td></tr><tr><td>mIoU↑</td><td> $F _ { \beta } ^ { \operatorname* { m a x } } \uparrow$ </td><td> $M \downarrow$ </td><td>mIoU↑</td><td>Fmax ↑ β</td><td>M↓</td></tr><tr><td>EBLNet [86]</td><td>0.870</td><td>0.922</td><td>0.064</td><td>0.817</td><td>0.878</td><td>0.059</td><td></td></tr><tr><td>IEBAF [87]</td><td>0.887</td><td>0.944</td><td>0.056</td><td>0.861</td><td>0.926</td><td>0.049</td><td></td></tr><tr><td>GhostingNet [88]</td><td>0.893</td><td>0.943</td><td>0.054</td><td>0.838</td><td>0.904</td><td>0.055</td><td></td></tr><tr><td>RFENet [89]</td><td>0.886</td><td>0.938</td><td>0.057</td><td>0.865</td><td>0.931</td><td>0.048</td><td></td></tr><tr><td>RFENet+ [17]</td><td>0.893</td><td>0.952</td><td>0.052</td><td>0.875</td><td>0.939</td><td>0.045</td><td>3.22 ↑</td></tr><tr><td>RFENet++ (Ours)</td><td>0.898</td><td>0.965</td><td>0.050</td><td>0.878</td><td>0.942</td><td>0.043</td><td>4.94↑</td></tr><tr><td>RUN [12]</td><td>0.895</td><td>0.952</td><td>0.051</td><td>0.866</td><td>0.938</td><td>0.043</td><td></td></tr><tr><td>RUN+ [17]</td><td>0.902</td><td>0.966</td><td>0.047</td><td>0.878</td><td>0.945</td><td>0.039</td><td>3.59 ↑</td></tr><tr><td> $\mathbf { R U N + + \left( O u r s \right) }$ </td><td>0.906</td><td>0.979</td><td>0.045</td><td>0.887</td><td>0.958</td><td>0.037</td><td>5.72 ↑</td></tr></table>

TABLE VII  
CONCEALED DEFECT DETECTION ON CDS2K.
<table><tr><td>Methods</td><td> $S _ { \alpha } \uparrow$ </td><td>M↓</td><td> $E _ { \phi } \uparrow$ </td><td> $F _ { \beta } \uparrow$ </td><td> $F _ { \beta } ^ { m } \uparrow$ </td><td>∆(%)</td></tr><tr><td>SINet V2 [7]</td><td>0.551</td><td>0.102</td><td>0.567</td><td>0.223</td><td>0.248</td><td></td></tr><tr><td>HitNet [90]</td><td>0.563</td><td>0.118</td><td>0.564</td><td>0.298</td><td>0.298</td><td></td></tr><tr><td>OAFormer [91]</td><td>0.541</td><td>0.121</td><td>0.535</td><td>0.216</td><td>0.239</td><td></td></tr><tr><td>CamoFormer [22]</td><td>0.589</td><td>0.100</td><td>0.588</td><td>0.330</td><td>0.329</td><td></td></tr><tr><td>CamoFormer+ [17]</td><td>0.599</td><td>0.091</td><td>0.603</td><td>0.345</td><td>0.346</td><td>4.59 ↑</td></tr><tr><td>CamoFormer++ (Ours)</td><td>0.607</td><td>0.088</td><td>0.617</td><td>0.359</td><td>0.361</td><td>7.70↑</td></tr><tr><td>RUN [12]</td><td>0.590</td><td>0.068</td><td>0.595</td><td>0.298</td><td>0.299</td><td></td></tr><tr><td>RUN+ [17]</td><td>0.598</td><td>0.061</td><td>0.612</td><td>0.309</td><td>0.310</td><td>4.38 ↑</td></tr><tr><td> $\mathbf { R U N + + \left( O u r s \right) }$ </td><td>0.607</td><td>0.058</td><td>0.626</td><td>0.323</td><td>0.324</td><td>7.91↑</td></tr></table>

TABLE VIII  
GENERALITY ACROSS ARCHITECTURES ON COD10K. $" + "$ AND “++” DENOTE CURRISEG [17] AND PACESHIFT. BASE IS THE BASELINE’S S<sub>α</sub>, AND THE TWO ∆ COLUMNS REPORT THE MEAN RELATIVE GAIN OF EACH RECIPE OVER ITS BASELINE ACROSS FOUR METRICS (M, $F _ { \beta } , E _ { \phi } , S _ { \alpha } )$ . FOR UGTR, THE + CELL REPORTS THE STANDARD CURRISEG RECIPE IN [17].
<table><tr><td>Paradigms</td><td>Model</td><td>Backbone</td><td>Base  $( S _ { \alpha } )$ </td><td>∆(%, +)</td><td>∆(%, ++)</td></tr><tr><td colspan="6">Convolutional detectors</td></tr><tr><td>Enc.-dec.</td><td>SINet-V2 [7]</td><td>Res2Net50</td><td>0.815</td><td>1.76↑</td><td>4.23 ↑</td></tr><tr><td>Graph reason.</td><td>MGL-R [58]</td><td>ResNet50</td><td>0.814</td><td>2.49 ↑</td><td>4.77 ↑</td></tr><tr><td>Bio-inspired</td><td>PreyNet [59]</td><td>ResNet50</td><td>0.813</td><td>2.54↑</td><td>5.10 ↑</td></tr><tr><td>Edge+freq.</td><td>FEDER [10]</td><td>ResNet50</td><td>0.810</td><td>3.05 ↑</td><td>4.30 ↑</td></tr><tr><td colspan="6">Attention and transformer detectors</td></tr><tr><td>Masked attn.</td><td>CamoFormer [22]</td><td>PVT V2</td><td>0.872</td><td>3.22 ↑</td><td>5.88 ↑</td></tr><tr><td>Multi-scale</td><td>ZoomNeXt [9]</td><td>PVT V2</td><td>0.887</td><td>2.07↑</td><td>4.14↑</td></tr><tr><td>Freq. attn.</td><td>FSEL [11]</td><td>ResNet50</td><td>0.822</td><td>3.03 ↑</td><td>4.21 ↑</td></tr><tr><td>Uncertainty</td><td>UGTR [21]</td><td>ResNet50</td><td>0.817</td><td>2.69 ↑</td><td>5.99 ↑</td></tr><tr><td colspan="6">Generative and unfolding models</td></tr><tr><td>Diffusion</td><td>CamoDiff [61]</td><td>PVT V2</td><td>0.883</td><td>1.90 ↑</td><td>3.79 ↑</td></tr><tr><td>Unfolding</td><td>RUN [12]</td><td>PVT V2</td><td>0.878</td><td>4.78↑</td><td>6.46 ↑</td></tr><tr><td>Nested unfold.</td><td>NUN [14]</td><td>PVT V2</td><td>0.890</td><td>1.70 ↑</td><td>3.80 ↑</td></tr><tr><td colspan="6">Foundation-model based</td></tr><tr><td>Found. backb.</td><td>NUN [14]</td><td>DINO V2-L</td><td>0.905</td><td>2.14 ↑</td><td>4.28 ↑</td></tr><tr><td>Found. adapt.</td><td>SAM-a [49]</td><td>ViT-H</td><td>0.883</td><td>3.29 ↑</td><td>4.70 ↑</td></tr></table>

TABLE IX

GENERALIZATION TO BROADER DENSE PREDICTION FAMILIES. ONE PRIMARY METRIC PER FAMILY. $\mathrm { \ddot { \Phi } + \mathrm { \stackrel {  } { \partial } \ A N D \ } \mathrm { \stackrel {  } { \Phi } + + \mathrm { \uparrow } } } ,$ DENOTE THE CURRISEG AND PACESHIFT RECIPES ON THE LISTED BASELINE, AND ∆ IS THE RELATIVE GAIN OF “++” OVER IT. ONLY THE TRAINING STRATEGY CHANGES, AND SPLITS FOLLOW EACH BASELINE.
<table><tr><td>Tasks</td><td>Datasets</td><td>Baselines</td><td>Metrics Base</td><td></td><td>+</td><td>++</td><td>∆(%)</td></tr><tr><td colspan="8">Tier 1: low-contrast detection with ambiguous annotation</td></tr><tr><td>IRSTD</td><td>IRSTD-1k [72]</td><td>IRSAM [92]</td><td>IoU↑</td><td>73.7</td><td>75.5</td><td>77.2</td><td>4.75 ↑</td></tr><tr><td>Shadow det.</td><td>SBU [73]</td><td>Spider [2]</td><td>BER↓</td><td>0.0400.038</td><td></td><td>0.037</td><td>7.50↑</td></tr><tr><td colspan="8">Tier 2: low-contrast clinical dense prediction</td></tr><tr><td>Skin lesion</td><td>ISIC 2018 [74]</td><td>DconnNet [93] DSC↑</td><td></td><td>90.4</td><td>90.7</td><td>92.8</td><td>2.65 ↑</td></tr><tr><td>Reti. OCT</td><td></td><td>RETOUCH [75] DconnNet [93] DSC ↑</td><td></td><td>87.7</td><td>87.5</td><td>89.6</td><td>2.17 ↑</td></tr><tr><td colspan="8">Tier 3: standard dense prediction</td></tr><tr><td>Dicho. seg.</td><td>DIS-VD [76]</td><td>MVANet [94]</td><td> $F _ { \beta } ^ { \omega } \uparrow$ </td><td>0.855 0.871</td><td></td><td>0.887</td><td>3.74↑</td></tr><tr><td>SOD</td><td>DUTS-TE [77]</td><td>VST-T++ [95]</td><td> $F _ { \beta } \uparrow$ </td><td>0.869 0.882</td><td></td><td>0.897</td><td>3.22 ↑</td></tr><tr><td></td><td>Semantic seg. ADE20K [78]</td><td>CGRSeg [96]</td><td>mIoU↑</td><td>45.5</td><td>46.8</td><td>47.8</td><td>5.05↑</td></tr><tr><td>Instance seg.</td><td>COCO [79]</td><td>M2Form [97]</td><td>AP↑</td><td>38.0</td><td>38.6</td><td>39.4</td><td>3.68↑</td></tr></table>

## D. Generalization to Broader Dense Prediction Families

To assess PaceShift beyond CECS, we evaluate increasingly diverse dense-prediction tasks exhibiting high-frequency shortcuts, supervision-reliability variation, both, or neither prominently (Sec. III-B). Tier 1 collects low-contrast detection with ambiguous annotation, Tier 2 low-contrast clinical dense prediction, and Tier 3 salient object detection and dichotomous segmentation with standard semantic and instance segmentation, the last using the multi-class instantiation of Sec. III-C. For each family, Table IX keeps one strong published baseline’s architecture, optimizer, augmentation, and epoch budget, changes only the training strategy with no constant retuned, and reports the family’s primary metric.

Tier 1: low-contrast detection with ambiguous annotation. Infrared small-target detection (IRSTD) and shadow detection are low-contrast but not texture-matched, and both carry substantial annotation ambiguity [73]. Gains persist here although the texture shortcut is weaker, indicating that the full formulation is not limited to shortcut-dominated data.

Tier 2: clinical dense prediction. Skin lesion and retinal OCT fluid segmentation combine low boundary contrast with costly, uncertain annotation, making supervision quality and boundary ambiguity particularly relevant to the reliability formulation. On RETOUCH, CurriSeg falls marginally below the baseline, whereas PaceShift recovers this loss and further improves performance under the same clinical setting.

TABLE X  
ONLINE TRAINING STRATEGIES UNDER FULL DATASET ACCESS (RUN, RESNET50, COD10K) AT A FIXED TRAINING HORIZON. “SA-RCS ONLY” USES THE COMPLETE PHASE-I RECIPE AND REMOVES ONLY PHASE II.
<table><tr><td>Strategies</td><td>Method source</td><td>M↓</td><td> $F _ { \beta } \uparrow$ </td><td> $E _ { \phi } \uparrow$ </td><td> $S _ { \alpha } \uparrow$ </td><td>∆(%)</td></tr><tr><td>Vanilla (RUN) [12]</td><td>ICML25</td><td>0.030</td><td>0.747</td><td>0.903</td><td>0.827</td><td></td></tr><tr><td>Loss curriculum</td><td></td><td>0.031</td><td>0.742</td><td>0.898</td><td>0.824</td><td>1.23 ↓</td></tr><tr><td>Co-teaching [98]</td><td>NeurIPS18</td><td>0.030</td><td>0.752</td><td>0.905</td><td>0.829</td><td>0.28 ↑</td></tr><tr><td>Meta-Weight-Net [99]</td><td>NeurIPS19</td><td>0.029</td><td>0.751</td><td>0.908</td><td>0.830</td><td>1.20 ↑</td></tr><tr><td>InfoBatch [36]</td><td>ICLR24</td><td>0.030</td><td>0.748</td><td>0.905</td><td>0.828</td><td>0.12 ↑</td></tr><tr><td>SeTa [37]</td><td>CVPR25</td><td>0.030</td><td>0.753</td><td>0.906</td><td>0.829</td><td>0.34 ↑</td></tr><tr><td>IES [100]</td><td>ICLR25</td><td>0.029</td><td>0.752</td><td>0.909</td><td>0.831</td><td>1.29 ↑</td></tr><tr><td>Dyn-Unc (online) [34]</td><td>CVPRW24</td><td>0.029</td><td>0.758</td><td>0.910</td><td>0.832</td><td>1.55↑</td></tr><tr><td>TDDS (online) [35]</td><td>CVPR24</td><td>0.028</td><td>0.757</td><td>0.912</td><td>0.833</td><td>2.43 ↑</td></tr><tr><td>+CurriSeg [17]</td><td>ICML26</td><td>0.029</td><td>0.765</td><td>0.917</td><td>0.836</td><td>2.10 ↑</td></tr><tr><td>+ SA-RCS only (Ours)</td><td></td><td>0.029</td><td>0.763</td><td>0.918</td><td>0.835</td><td>2.03 ↑</td></tr><tr><td>+PaceShift (Ours)</td><td></td><td>0.027</td><td>0.775</td><td>0.926</td><td>0.838</td><td>4.41 ↑</td></tr></table>

![](images/a3bbb9f3ace46e4f748d509158c4f03135cf407a68ea7784d71f8e9ed83a51dd.jpg)  
Fig. 5. One-shot coreset selection on COD10K (RUN, ResNet50), showing absolute $F _ { \beta }$ differences $( \times 1 0 ^ { - 2 } )$ from random selection at each retention rate. We compare the SA-RCS diagnostic with Moderate [101], CCS [102], Dyn-Unc [34], TDDS [35], and label-free ELFS [103]. All differences vanish at 100% retention, where selection is inactive. Vanilla achieves $F _ { \beta } { = } 0 . 7 4 7$

Tier 3: standard dense prediction. Dichotomous, salient, semantic, and instance segmentation are not defined by a texture-matched background. PaceShift nevertheless yields consistent gains under the native host budgets, with the same proportional stabilize-then-challenge schedule and without introducing task-specific constants. The multi-class and instance rows use the corresponding output-form instantiations of Sec. III-C. These results support transfer of the full training paradigm beyond context-entangled settings.

## E. Comparison with Data-Centric Training and Selection

Data-centric methods split into two families with different objectives: online strategies change which samples contribute during one run, whereas one-shot coreset scores select a subset before retraining and collapse to the vanilla result at full retention. We therefore compare the two families under separate protocols, all on a fixed RUN (ResNet50) backbone on COD10K with shared optimization rules.

Adaptation of scores designed for classification. Per-pixel quantities use a class-balanced spatial mean so background pixels do not dominate, trajectory scores share one checkpoint cache, and coreset methods read the same pooled encoder feature. For the online comparison, Dyn-Unc and TDDS are mapped to sample weights by the same normalization and floor $w _ { \mathrm { m i n } }$ as Eq. (6), and the retention study keeps their one-shot form. Per-score definitions are in the Supp.

TABLE XI  
WEAK SUPERVISION ON COD10K (SUPERVISION: POINT, BACKBONE: PVT V2, FOUNDATION MODEL: SAM3). “+” DENOTES CURRISEG ADAPTED TO THIS REGIME AS DESCRIBED IN THE SUPP., $\mathbf { A N D \Phi } ^ { * * } + + \mathbf { \Phi } ^ { * }$ DENOTES PACESHIFT.
<table><tr><td>Methods</td><td>Method source</td><td>M↓</td><td> $F _ { \beta } \uparrow$ </td><td> $E _ { \phi } \uparrow$ </td><td> $S _ { \alpha } \uparrow$ </td><td> $\Delta ( \% )$ </td></tr><tr><td>WS-SAM [104]</td><td>NeurIPS23</td><td>0.035</td><td>0.724</td><td>0.865</td><td>0.808</td><td></td></tr><tr><td>SEE [42]</td><td>TPAMI25</td><td>0.033</td><td>0.735</td><td>0.866</td><td>0.814</td><td></td></tr><tr><td>SEE+ [17]</td><td>ICML26</td><td>0.031</td><td>0.746</td><td>0.878</td><td>0.821</td><td>2.45 ↑</td></tr><tr><td>SEE++ (Ours)</td><td></td><td>0.029</td><td>0.761</td><td>0.892</td><td>0.827</td><td>5.06 ↑</td></tr><tr><td>SCALER [43]</td><td>ECCV26</td><td>0.029</td><td>0.758</td><td>0.897</td><td>0.827</td><td></td></tr><tr><td>SCALER+ [17]</td><td>ICML26</td><td>0.028</td><td>0.767</td><td>0.906</td><td>0.829</td><td>1.47 ↑</td></tr><tr><td>SCALER++ (Ours)</td><td></td><td>0.026</td><td>0.780</td><td>0.915</td><td>0.834</td><td>4.03 ↑</td></tr></table>

TABLE XII

SEMI-SUPERVISION ON COD10K (1/16 LABELED TRAINING DATA, PVT V2, SAM3). “+” DENOTES CURRISEG ADAPTED TO THIS REGIME AS DETAILED IN THE SUPP., AND “++” DENOTES PACESHIFT.
<table><tr><td>Methods</td><td>Method source</td><td>M↓</td><td> $F _ { \beta } \uparrow$ </td><td> $E _ { \phi } \uparrow$ </td><td> $S _ { \alpha } \uparrow$ </td><td> $\Delta ( \% )$ </td></tr><tr><td>CoSOD [105]</td><td>WACV24</td><td>0.053</td><td>0.674</td><td>0.807</td><td>0.766</td><td></td></tr><tr><td>SEE [42]</td><td>TPAMI25</td><td>0.043</td><td>0.713</td><td>0.825</td><td>0.771</td><td></td></tr><tr><td>SEE+ [17]</td><td>ICML26</td><td>0.041</td><td>0.723</td><td>0.836</td><td>0.776</td><td>2.01 ↑</td></tr><tr><td>SEE++ (Ours)</td><td></td><td>0.038</td><td>0.740</td><td>0.848</td><td>0.783</td><td>4.94 ↑</td></tr><tr><td>SCALER [43]</td><td>ECCV26</td><td>0.034</td><td>0.733</td><td>0.849</td><td>0.790</td><td></td></tr><tr><td>SCALER+ [17]</td><td>ICML26</td><td>0.033</td><td>0.739</td><td>0.855</td><td>0.793</td><td>1.21 ↑</td></tr><tr><td> $\mathbf { S C A L E R + + \thinspace ( O u r s ) }$ </td><td></td><td>0.031</td><td>0.748</td><td>0.863</td><td>0.797</td><td>3.35 ↑</td></tr></table>

TABLE XIII

Online strategies. Three observations follow from Table X. A naive loss curriculum hurts under this schedule, consistent with Sec. III-B and with its null effect under the matched budget of Table XIX: easy, clean samples first let the model commit to the texture shortcut. Second, the trajectory scores Dyn-Unc and TDDS are the strongest prior baselines, which supports our premise that the relevant signal is a trajectory property. Third, SA-RCS alone exceeds both on $F _ { \beta } , E _ { \phi } ,$ and $S _ { \alpha }$ while remaining comparable in M, and the full PaceShift widens the margin: the distinction is the supervision-adaptive reliability score, its online refresh, the pixel-level coupling, and the guarded post-training, none of which these efficiencyoriented scores designed for classification provide.

Fixed-retention coreset study. Fig. 5 evaluates a representative one-shot score per family as the $F _ { \beta }$ gain over random selection. Since PaceShift is online and emits no static coreset, its diagnostic averages the Phase-I weight, $\begin{array} { r } { s _ { i } ^ { \mathrm { S A } } ~ = ~ \frac { 1 } { T _ { c } - T _ { w } } \sum _ { t = T _ { w } + 1 } ^ { T _ { c } } \omega _ { i } ^ { ( t ) } } \end{array}$ , retains the top fraction, and retrains from scratch. Our diagnostic gives the largest gain at 40–80% retention and a competitive one at 20%. PaceShift does not compete there, since it uses the full set and targets accuracy, robustness, and calibration instead of subset reduction.

## F. Learning under Incomplete Supervision

Because “supervision-adaptive” is central to our claim, weak and semi-supervision are treated as primary experiments: a point-annotated weak regime and a scarce 1/16 semisupervised regime, both driving curriculum selection by the reliability score $\rho _ { i }$ of Eq. (2) in place of a ground-truth IoU. Weak supervision. Table XI layers PaceShift on two pointsupervised COD segmenters, SEE [42] and SCALER [43]. For a point cue the supervision-agreement term of Eq. (2) averages point-cue satisfaction (foreground-disk coverage with

CORRUPTED PSEUDO-LABEL DETECTION ON COD10K UNDER THE 1/16   
SEMI-SUPERVISED SETTING. WE CORRUPT 20% OF THE EMA-TEACHER   
PSEUDO-MASKS, AND THE FROZEN SAM3 PROPOSAL IS NEVER   
CORRUPTED. EACH CRITERION FLAGS ITS LOWEST-SCORING 20%,   
MATCHING THE CORRUPTION RATE, SO FLAGGING PRECISION AND RECALL   
COINCIDE (FLAG P/R, RANDOM 0.200). “RET. PL $S _ { \alpha } \mathbf { \backslash }$ ” IS THE MEAN   
QUALITY OF THE RETAINED PSEUDO-LABELS AGAINST HELD-OUT GROUND   
TRUTH, USED ONLY FOR EVALUATION. AUROC AND AUPRC ARE   
THRESHOLD-FREE (RANDOM 0.50 AND ABOUT 0.20).   
“PSEUDO-MASK–SAM3 AGREEMENT” SCORES EVERY EVALUATED   
PSEUDO-MASK, CORRUPTED OR NOT, DIRECTLY BY ITS IOU WITH THE   
FROZEN PROPOSAL, WITHOUT THE SEGMENTER.

<table><tr><td>Criteria</td><td>Flag P/R↑</td><td>Ret.  $\mathrm { P L } ~ S _ { \alpha } \uparrow$ </td><td>AUROC↑</td><td>AUPRC↑</td></tr><tr><td>Confidence filter</td><td>0.589</td><td>0.801</td><td>0.682</td><td>0.641</td></tr><tr><td>Entropy only</td><td>0.631</td><td>0.807</td><td>0.708</td><td>0.669</td></tr><tr><td>Peer small-loss</td><td>0.643</td><td>0.810</td><td>0.723</td><td>0.687</td></tr><tr><td>Dyn-Unc [34]</td><td>0.702</td><td>0.817</td><td>0.774</td><td>0.735</td></tr><tr><td>Pseudo-mask-SAM3 agreement</td><td>0.735</td><td>0.822</td><td>0.806</td><td>0.774</td></tr><tr><td>Prediction-SAM3 agreement</td><td>0.681</td><td>0.814</td><td>0.759</td><td>0.719</td></tr><tr><td>ρi w/o ext. proposal</td><td>0.724</td><td>0.820</td><td>0.792</td><td>0.761</td></tr><tr><td>Reliability  $\rho _ { i }$ </td><td>0.769</td><td>0.827</td><td>0.827</td><td>0.801</td></tr></table>

TABLE XIV

FOUNDATION-MODEL ADAPTERS ON COD10K UNDER FULL SUPERVISION. $" + "$ AND $\ " { + + \prime } $ DENOTE CURRISEG AND PACESHIFT RECIPES.
<table><tr><td>Methods</td><td>M↓</td><td> $F _ { \beta } \uparrow$ </td><td> $E _ { \phi } \uparrow$ </td><td>Sα↑</td><td>∆(%)</td></tr><tr><td>SAM-Adapter [49]</td><td>0.025</td><td>0.800</td><td>0.918</td><td>0.883</td><td></td></tr><tr><td>SAM-Adapter+ [17]</td><td>0.023</td><td>0.822</td><td>0.935</td><td>0.888</td><td>3.29 ↑</td></tr><tr><td>SAM-Adapter++ (Ours)</td><td>0.022</td><td>0.827</td><td>0.940</td><td>0.892</td><td>4.70 ↑</td></tr><tr><td>SAM2-Adapter [107]</td><td>0.018</td><td>0.848</td><td>0.950</td><td>0.899</td><td></td></tr><tr><td>SAM2-Adapter+ [17]</td><td>0.016</td><td>0.869</td><td>0.962</td><td>0.903</td><td>3.82 ↑</td></tr><tr><td>SAM2-Adapter++ (Ours)</td><td>0.015</td><td>0.876</td><td>0.969</td><td>0.906</td><td>5.69 ↑</td></tr><tr><td>SAM3-Adapter [108]</td><td>0.015</td><td>0.883</td><td>0.965</td><td>0.927</td><td></td></tr><tr><td>SAM3-Adapter+ [17]</td><td>0.014</td><td>0.892</td><td>0.968</td><td>0.930</td><td>2.08 ↑</td></tr><tr><td>SAM3-Adapter++ (Ours)</td><td>0.013</td><td>0.901</td><td>0.975</td><td>0.932</td><td>4.24 ↑</td></tr></table>

a background-seed penalty) and agreement with the SAM3 mask proposal [106], while the augmentation-consistency, model-disagreement, and entropy cues are unchanged, so the curriculum discounts pseudo-masks that disagree with all available cues, not the model’s own confident ones. Using no additional dense label, PaceShift (“++”) improves both baselines on every metric, and its mean relative gain more than doubles that of CurriSeg $( ^ { 6 6 } + 7 )$ in each case, precisely where the ground-truth difficulty $d _ { i } ^ { \mathrm { f u l l } }$ of Eq. (1) is undefined.

Semi-supervision. Table XII repeats the protocol in the particularly scarce 1/16 regime, with CoSOD [105] for context. The unlabeled samples are routed by $\rho _ { i } ,$ , which fuses agreement with an EMA teacher and the SAM3 proposal, augmentation consistency, model disagreement, and prediction entropy, so pseudo-mask reliability is assessed from several complementary cues, not from confidence alone. PaceShift again improves both segmenters on every metric despite the severely limited labeled supervision, and again more than doubles the conference gain. Pseudo-label triage. To test whether $\rho _ { i }$ captures pseudo-label quality or acts only as a regularizer, we corrupt 20% of the pseudo-labels in the 1/16 regime with an equal mix of boundary displacement, false-positive patches, and morphological distortion. Each criterion flags its lowest-scoring 20%. Ground truth is used only for evaluation. The external SAM3 proposal remains uncorrupted, so this diagnostic assesses corruption detection with an available external comparison signal. Table XIII reports flagging precision/recall, which coincide by construction, retained pseudo-label quality, and threshold-free AUROC and AUPRC: First, $\rho _ { i }$ outperforms the confidence, entropy, small-loss, and Dyn-Unc [34] on both detection and retained quality, so no single temporal or confidence signal explains its advantage. Second, the control rows isolate the external cue. Scoring each pseudo-mask directly by its agreement with the uncorrupted proposal is a strong detector on its own (AUROC 0.806), stronger than agreement between the model prediction and the proposal (0.759), yet $\rho _ { i }$ improves on direct checking on all four measures, by 2.1 AUROC points (paired bootstrap 95% CI 1.0 to 3.2, Supp.) and 3.4 flagging points, and $\rho _ { i }$ without any proposal already beats every nonexternal baseline, so the internal cues add information beyond the proposal and the gain comes from combining the two.

TABLE XV  
VLM-BASED WEAKLY SUPERVISED SEMANTIC SEGMENTATION (MIOU %).$\ " { + + \prime } $ ADDS OUR PACESHIFT TO EACH CLIP-BASED BASELINE.
<table><tr><td rowspan="2">Methods</td><td rowspan="2">Method source</td><td colspan="2">VOC</td><td>COCO</td><td rowspan="2">∆(%)</td></tr><tr><td>Val.</td><td>Test</td><td>Val.</td></tr><tr><td>DIAL [109]</td><td>ECCV24</td><td>74.5</td><td>74.9</td><td>44.4</td><td>一</td></tr><tr><td>WeCLIP [110]</td><td>CVPR24</td><td>76.4</td><td>77.2</td><td>47.1</td><td></td></tr><tr><td>WeCLIP++ (Ours)</td><td></td><td>77.3</td><td>77.8</td><td>48.1</td><td> $1 . 3 6 ~ \uparrow$ </td></tr><tr><td>ExCEL [111]</td><td>CVPR25</td><td>78.4</td><td>78.5</td><td>50.3</td><td></td></tr><tr><td>ExCEL++ (Ours)</td><td></td><td>79.2</td><td>79.2</td><td>50.9</td><td> $1 . 0 4 ~ \uparrow$ </td></tr></table>

TABLE XVI

VLM-BASED OPEN-VOCABULARY CAMOUFLAGED OBJECT SEGMENTATION ON OVCAMO. THE c· MEASURES ARE THE OPEN-VOCABULARY COS METRICS AND CIOU IS THE CLASS-AWARE IOU.
<table><tr><td>Methods</td><td> $c S _ { \alpha } ^ { \mathrm { ~ ~ } }$  个</td><td> ${ c F } _ { \beta } ^ { w } \uparrow$ </td><td>cM↓</td><td> $c F _ { \beta } \uparrow$  人</td><td> $c E _ { \phi } \uparrow$ </td><td>cIoU↑</td><td>∆(%)</td></tr><tr><td>FC-CLIP [112]</td><td>0.080</td><td>0.076</td><td>0.872</td><td>0.090</td><td>0.191</td><td>0.072</td><td></td></tr><tr><td>OVCoser [113]</td><td>0.579</td><td>0.490</td><td>0.336</td><td>0.520</td><td>0.616</td><td>0.443</td><td></td></tr><tr><td>OVCoser++ (Ours)</td><td>0.593</td><td>0.526</td><td>0.312</td><td>0.556</td><td>0.651</td><td>0.466</td><td>5.78 ↑</td></tr><tr><td>COCUS [114]</td><td>0.668</td><td>0.615</td><td>0.265</td><td>0.631</td><td>0.697</td><td>0.568</td><td></td></tr><tr><td>COCUS++ (Ours)</td><td>0.683</td><td>0.647</td><td>0.252</td><td>0.662</td><td>0.725</td><td>0.584</td><td>4.02 ↑</td></tr></table>

## G. Foundation-Model Adaptation

Modern dense prediction often adapts frozen foundation backbones using lightweight adapters. We evaluate SAM-, SAM2-, and SAM3-Adapters [49], [107], [108] on COD under the canonical full-supervision protocol, comparing each baseline with CurriSeg and PaceShift. The promptensembled frozen foundation model serves as teacher and the adapter-tuned model as student. Their disagreement provides reliability evidence, while the teacher mask is never treated as GT. The proposal informs the disagreement cue in $\rho _ { i }$ and the consistency term $Q _ { \mathrm { { c o n s } } }$ of Eq. (11), while $Q _ { \mathrm { s u p } }$ uses available mask supervision. Both phases share the same supervision-evidence construction without relying on teacher pseudo-labels. Table XIV shows that PaceShift improves every adapter and consistently exceeds CurriSeg.

## H. Results on VLM- and Reference-Guided Segmentation

A growing family of segmenters uses vision-language or reference information. We apply PaceShift to three such settings for controlled comparison, each with the original baseline and a stronger recent method, preserving the architecture, input protocol, optimizer, training horizon, and trainable parameter count while changing only data scheduling and post-training. Each setting is evaluated on its native benchmark with the corresponding official metrics. The unreliability PaceShift addresses is instantiated per case. The conference recipe is compared in the CECS and incomplete-supervision experiments, and this section reports PaceShift against the host baselines.

TABLE XVII  
REFERENCE-GUIDED CAMOUFLAGED OBJECT SEGMENTATION ON R2C7K.
<table><tr><td>Methods</td><td>M↓</td><td> $F _ { \beta } ^ { w } \uparrow$ </td><td> $E _ { \phi } \uparrow$ </td><td> $S _ { \alpha } \uparrow$ </td><td>∆(%)</td></tr><tr><td>SINetV2-Ref [7], [54]</td><td>0.033</td><td>0.700</td><td>0.888</td><td>0.823</td><td></td></tr><tr><td>R2CNet [54]</td><td>0.036</td><td>0.669</td><td>0.879</td><td>0.805</td><td></td></tr><tr><td>R2CNet++ (Ours)</td><td>0.033</td><td>0.692</td><td>0.896</td><td>0.815</td><td>3.74 ↑</td></tr><tr><td>RefOnce [115]</td><td>0.019</td><td>0.819</td><td>0.937</td><td>0.890</td><td></td></tr><tr><td>RefOnce++ (Ours)</td><td>0.018</td><td>0.832</td><td>0.951</td><td>0.896</td><td>2.26 ↑</td></tr></table>

VLM-based weakly supervised segmentation. CLIP-based segmenters derive supervision from CAM pseudo-masks whose quality varies widely across images and classes. Since no dense GT exists at training time, the difficulty stays $d _ { i } = 1 - \rho _ { i }$ , with class-balanced CAM agreement serving as the supervision signal $s _ { i } ^ { \mathrm { s u p } }$ , vision-language evidence entering through the consistency and disagreement cues of Sec. III-C, and normalized multi-class entropy completing $\rho _ { i } ,$ , so pseudo-mask reliability is assessed from several decoupled cues instead of CAM confidence alone. Table XV plugs PaceShift into WeCLIP [110] and ExCEL [111] on PASCAL VOC [116] and MS COCO [79], with DIAL [109] for context, improving both models on every evaluated split.

VLM-based open-vocabulary camouflaged segmentation. Open-vocabulary segmentation requires both localizing camouflaged objects and recognizing potentially unseen classes. We incorporate vision-language agreement into the existing disagreement and consistency cues, while $Q _ { \mathrm { s u p } }$ uses available mask supervision for each candidate. Table XVI shows that PaceShift consistently improves OVCoser [113] and COCUS [114], including class-aware IoU. The generic FC-CLIP [112] performs worse on OVCamo, highlighting the difficulty of this task for general-purpose segmenters.

Reference-guided camouflaged segmentation. Referencetarget agreement provides condition-specific reliability evidence during training, computed in an independent feature space so it does not merely echo the segmenter’s own matching branch, and enters through the same cues. Table XVII plugs PaceShift into R2CNet [54], which introduced the R2C7K benchmark, and the stronger RefOnce [115], which distills references into a prototype memory and needs none at inference. Both improve consistently on every metric.

## I. Ablation Study and Hyperparameter Sensitivity

Unless otherwise stated, ablations use RUN with PVT V2 on COD10K, and supervision-dependent ablations switch the regime, not the dataset. Single-run Phase-II controls share one checkpoint $\theta _ { 0 } .$ , and the five-run Phase-II comparison uses one Phase-I endpoint per run.

Curriculum components and their complementarity. Table XVIII runs the complete factorial over the three Phase-I mechanisms without post-training, so the phase is isolated. Selection is the strongest standalone mechanism and temporal weighting adds a smaller, consistent gain, and pixel weighting is weak alone but improves all four metrics on top of RCS and TSSW, acting as complementary refinement.

TABLE XVIII  
FULL FACTORIAL OVER PHASE-I WITHOUT POST-TRAINING. RCS: ROBUSTCURRICULUM SELECTION, THE PERCENTILE SELECTION OF EQ. (3), TSSW:TEMPORAL STATISTICS-BASED SAMPLE WEIGHTING, PUE: PIXELUNCERTAINTY WEIGHTING. THE ALL-ON CONFIGURATION CORRESPONDSTO SA-RCS.
<table><tr><td>RCS</td><td>TSSW</td><td>PUE</td><td>M↓</td><td> $F _ { \beta } \uparrow$ </td><td> $E _ { \phi } \uparrow$ </td><td> $S _ { \alpha } \uparrow$ </td></tr><tr><td></td><td></td><td></td><td>0.021</td><td>0.810</td><td>0.941</td><td>0.878</td></tr><tr><td></td><td></td><td></td><td>0.020</td><td>0.817</td><td>0.946</td><td>0.879</td></tr><tr><td></td><td>√</td><td></td><td>0.021</td><td>0.814</td><td>0.943</td><td>0.879</td></tr><tr><td></td><td></td><td></td><td>0.021</td><td>0.813</td><td>0.942</td><td>0.878</td></tr><tr><td> $\checkmark$ </td><td> $\checkmark$ </td><td></td><td>0.020</td><td>0.821</td><td>0.949</td><td>0.880</td></tr><tr><td> $\checkmark$ </td><td></td><td></td><td>0.020</td><td>0.819</td><td>0.947</td><td>0.879</td></tr><tr><td> $-$ </td><td> $\checkmark$ </td><td>√</td><td>0.020</td><td>0.816</td><td>0.945</td><td>0.879</td></tr><tr><td> $\checkmark$ </td><td> $\checkmark$ </td><td></td><td>0.019</td><td>0.824</td><td>0.952</td><td>0.881</td></tr></table>

Curriculum ordering. The factorial shows that selection helps. Table XIX asks what it should be based on. Every row keeps the same progressive schedule $p ( t )$ , subset sizes, optimizer-step counts, and weighting. All dynamic rankings are refreshed at the same cadence, Static $\rho _ { i }$ is computed once after warm-up and frozen, and Random fixes one random order. The random order row remains close to the no-selection configuration of Table XVIII, with only small gains in $F _ { \beta }$ and $E _ { \phi } ,$ , so the progressive schedule alone does not explain the full improvement. Loss- and confidence-based rankings give inconsistent and limited gains over this control, in line with the weak link between instantaneous loss and useful updates in Sec. IV-K, whereas reliability-based rankings improve all four metrics. The online score keeps a modest, consistent margin over the RCS ranking, and freezing $\rho _ { i }$ after warm-up removes that margin and becomes comparable to RCS, suggesting that online tracking of the learning dynamics contributes to the improvement.

Inside the supervision-adaptive curriculum. Table XX(a) isolates the reliability cues of $\rho _ { i } ,$ , the main Phase-I change from CurriSeg. Supervision evidence alone performs substantially below the full reliability score, and removing model disagreement causes the largest drop among the drop-one variants. These results support combining complementary cues instead of relying on supervision agreement alone, and the external proposal is isolated in Table XIII. In (b) and (c), the parameterfree mid-variance weight and the pixel weight with floor and decay match or exceed every compared variant on all metrics.

The guarded anti-curriculum. Beyond the four accuracy metrics, Table XXI reports pixel-pooled ECE, the groundtruth-stratified confidence–accuracy gap (GT-CAG), and $S _ { \alpha } { } ^ { c } .$ the absolute $S _ { \alpha }$ averaged over the corruption suite of Fig. 7, first within each family and then across the four families, over the five nonzero severities. The upper part uses the five runs of Fig. 6. Every post-training variant starts from the Phase-I endpoint of its run with the same budget, and the rows compare alternative designs, not a cumulative path. The lower part reports single-run controls from the canonical run. Multi-radius supervision improves over the single-radius conference operator but stays below the full design. Guarded $m ^ { + }$ distillation, which differs from full SBFT-P only in the clean-view term, recovers most of its clean-accuracy improvement, with small differences in the five-run mean accuracy scores. Replacing distillation with the reference-relative pairwise objective improves $S _ { \alpha } { } ^ { c }$ by $0 . 0 0 5 8 \pm 0 . 0 0 1 8$ and reduces ECE by $0 . 0 0 1 0 \pm 0 . 0 0 0 3$ and

TABLE XIX  
CURRICULUM ORDERING UNDER A MATCHED BUDGET ON COD10K.
<table><tr><td>Ranking criteria</td><td>M↓</td><td> $F _ { \beta } \uparrow$ </td><td> $E _ { \phi } \uparrow$ </td><td> $S _ { \alpha } \uparrow$ </td></tr><tr><td>Random</td><td>0.021</td><td>0.814</td><td>0.944</td><td>0.878</td></tr><tr><td>Loss easy-first</td><td>0.021</td><td>0.812</td><td>0.943</td><td>0.877</td></tr><tr><td>Loss hard-first</td><td>0.020</td><td>0.816</td><td>0.945</td><td>0.878</td></tr><tr><td>Confidence easy-first</td><td>0.021</td><td>0.813</td><td>0.943</td><td>0.877</td></tr><tr><td>RCS ranking (CurriSeg)</td><td>0.020</td><td>0.820</td><td>0.949</td><td>0.880</td></tr><tr><td>Static  $\rho _ { i }$ </td><td>0.020</td><td>0.818</td><td>0.947</td><td>0.879</td></tr><tr><td>Online SA-RCS</td><td>0.019</td><td>0.824</td><td>0.952</td><td>0.881</td></tr></table>

GT-CAG by $0 . 0 0 2 7 \pm 0 . 0 0 0 9$ in all five runs (mean paired difference ± 95% CI half-width, Holm-adjusted paired t-test $p ~ < ~ 0 . 0 1 )$ . In the canonical run, random preferences stay at the level of multi-radius supervision, confidence-ranked preferences recover some accuracy and robustness but widen GT-CAG, and removing the anchor costs a margin similar to that of the pairwise objective. The guard drop-one ablations (Supp.) largely align with the intended roles: removing $Q _ { \mathrm { r o b } }$ mainly costs corruption robustness, $Q _ { \mathrm { u n c } }$ the confidence gap, and $Q _ { \mathrm { s u p } }$ clean accuracy. Swapping the frozen SA-RCS reference in $Q _ { \mathrm { { c o n s } } }$ for the current prediction costs substantially more corruption robustness than clean accuracy, supporting a fixed consistency reference over self-agreement. The final row is a direct confidence-reward negative control (defined in the Supp.) that leaves candidate selection unchanged and instead minimizes prediction entropy during post-training. Combined with the confidence-preference row, it separates confidence-based candidate selection from directly optimizing confidence, and Table XXIV reports its calibration effect.

Phase ordering. Table XXII examines whether the two phases are interchangeable. Challenge-first schedules, guarded or not, fall substantially below standard training, consistent with an early spectral challenge interfering with representation formation. Reversing the phases recovers much of the damage but stays below standard training in $S _ { \alpha }$ and $S _ { \alpha } { } ^ { c } .$ , so later stabilization does not recover the loss from an early challenge in this setting. After stabilization, the conference operator realizes part of the gain and SBFT-P the rest, largest in $S _ { \alpha } { } ^ { c }$ and GT-CAG, the properties Phase II targets. Alternating the two losses recovers part of the benefit but stays below the strict twostage order, again with the clearest gaps in $S _ { \alpha } { } ^ { c }$ and GT-CAG. An adaptive trigger that starts Phase II at the validation plateau performs on par with the fixed $T _ { c }$ , so the result is not sensitive to the transition epoch. Fig. 6 complements these end points with the training dynamics of all three recipes.

Trust parameterization. The source-type presets are the only regime-level constants of the method, so Table XXIII stresses them in two regimes that together cover all three supervision sources: semi-supervised $1 / 1 6$ (Table XII), where dense human masks coexist with model-derived pseudo-labels, and point supervision (Table XI), which exercises the sparse-human preset. Uniform cue weights and a source-agnostic $\tau _ { \mathrm { t r u s t } } { = } 0 . 5$ both reduce performance, and mismatched source trust degrades the most, supporting source-dependent trust over a single scalar. The presets are also insensitive to their exact values: ten orderpreserving ±20% perturbations stay close to the default, and validation tuning improves $S _ { \alpha }$ by only 0.001.

Global constants. One-at-a-time sweeps under the fixed budget

TABLE XX  
INSIDE THE SUPERVISION-ADAPTIVE CURRICULUM. EACH PHASE-I VARIANT IS INSERTED INTO THE OTHERWISE UNCHANGED FULL PACESHIFT PIPELINE, INCLUDING PHASE II. SO EVERY PANEL REPORTS FINAL PERFORMANCE. (A) RELIABILITY CUES UNDER 1/16 SEMI-SUPERVISION (SCALER. SAM3). (B) TEMPORAL WEIGHTING SHAPE UNDER FULL SUPERVISION. (C) PIXEL UNCERTAINTY WEIGHTING UNDER FULL SUPERVISION.  
(a) Reliability cues  
(b) Temporal weighting
<table><tr><td>ρi variants</td><td>M↓  $F _ { \beta } \uparrow$   $E _ { \phi } \uparrow$ </td><td> $S _ { \alpha } \uparrow$ </td></tr><tr><td>Supervision only</td><td>0.035 0.730 0.848</td><td>0.788</td></tr><tr><td>w/o supervision</td><td>0.033 0.740 0.855</td><td>0.792</td></tr><tr><td>w/o consistency</td><td>0.032 0.742 0.857</td><td>0.793</td></tr><tr><td>w/o disagreement</td><td>0.034 0.735 0.851</td><td>0.789</td></tr><tr><td>w/o entropy</td><td>0.032 0.744 0.859</td><td>0.795</td></tr><tr><td>Full  $\rho _ { i }$ </td><td>0.031 0.748</td><td>0.863 0.797</td></tr></table>

<table><tr><td>ωi variants</td><td> $M \downarrow$   $F _ { \beta } \uparrow$ </td><td> $E _ { \phi } \uparrow$   $S _ { \alpha } \uparrow$ </td></tr><tr><td>CurriSeg TSSW</td><td>0.018 0.832 0.959</td><td>0.887</td></tr><tr><td>Mono. incr. σ</td><td>0.018 0.829 0.957</td><td>0.885</td></tr><tr><td>Mono. decr. σ</td><td>0.018 0.831 0.958</td><td>0.886</td></tr><tr><td>w/o  $\omega ^ { \mathrm { o u t } }$ </td><td>0.017 0.833 0.960</td><td>0.887</td></tr><tr><td>Unimodal mean</td><td>0.017 0.834 0.961</td><td>0.888</td></tr><tr><td>Param.-free full</td><td>0.017 0.836 0.963</td><td>0.889</td></tr></table>

TABLE XXI

PHASE-II DESIGNS ON COD10K (RUN, PVT V2). UPPER PART: MEAN OVER THE FIVE RUNS OF FIG. 6, WITH THE SAMPLE STANDARD DEVIATION IN THE ROW BENEATH. WITHIN EACH RUN, ALL POST-TRAINING VARIANTS SHARE THE PHASE-I ENDPOINT AND THE OPTIMIZATION-STEP BUDGET, AND SA-RCS ONLY EVALUATES THAT ENDPOINT WITHOUT POST-TRAINING. LOWER PART: SINGLE-RUN CONTROLS FROM THE CANONICAL RUN (RUN 1). ECE IS PIXEL-POOLED, GT-CAG IS A CONFIDENCE DIAGNOSTIC, AND $S _ { \alpha } { } ^ { c }$ IS THE CORRUPTION-AVERAGED $S _ { \alpha } .$ . GUARD DROP-ONE ABLATIONS ARE IN THE SUPP.
<table><tr><td>Variants</td><td>M↓</td><td> $F _ { \beta } \uparrow$ </td><td></td><td></td><td>Eφ↑ Sα↑ Sαc↑1</td><td>ECE↓</td><td>GT-CAG↓</td></tr><tr><td colspan="8">Five runs: mean (first row), std (second row)</td></tr><tr><td>SA-RCS only (no post)</td><td>0.019 0.823 0.951 0.880 0.788 ±.0003 ±.002 ±.002 ±.001 ±.003</td><td></td><td></td><td></td><td></td><td>0.015 ±.001</td><td>0.048 ±.001</td></tr><tr><td>Single-r SBFT (conf.)</td><td>0.018</td><td>0.827</td><td></td><td>0.9550.883</td><td>0.802</td><td>0.014</td><td>0.044</td></tr><tr><td>Multi-r supervised SBFT</td><td>±.0003 0.018</td><td></td><td></td><td></td><td>±.002 ±.002 ±.001 ±.003 0.829 0.957 0.884 0.811</td><td>±.001 0.013</td><td>±.001 0.040</td></tr><tr><td>SBFT + guarded  $m ^ { + }$  distill.</td><td>±.0003 0.017</td><td>0.835</td><td>0.961</td><td>0.888</td><td>±.002 ±.001 ±.001 ±.003 0.825</td><td>±.001 0.012</td><td>±.001 0.038</td></tr><tr><td>SBFT-P (full)</td><td>±.0003 0.017 ±.0002</td><td>±.003 0.835 ±.002</td><td>0.962</td><td>0.888</td><td>±.002 ±.001 ±.002 0.831 ±.003</td><td>±.001 0.011 ±.001</td><td>±.001 0.036 ±.001</td></tr><tr><td colspan="8">Single run (run 1)</td></tr><tr><td>SBFT-P (full, run 1)</td><td>0.017</td><td>0.836</td><td>0.963</td><td>0.889</td><td>0.832</td><td>0.011</td><td>0.035</td></tr><tr><td>SBFT + random pref.</td><td>0.018</td><td>0.829</td><td>0.957</td><td>0.884</td><td>0.809</td><td>0.013</td><td>0.041</td></tr><tr><td>SBFT + confidence pref.</td><td>0.018</td><td>0.831</td><td>0.959</td><td>0.886</td><td>0.815</td><td>0.015</td><td>0.044</td></tr><tr><td>SBFT + guarded, w/o anchor</td><td>0.017</td><td>0.833</td><td>0.960</td><td>0.887</td><td>0.824</td><td>0.012</td><td>0.039</td></tr><tr><td>Direct conf. reward</td><td>0.018</td><td></td><td></td><td>0.831 0.958 0.886 0.806</td><td></td><td>0.024</td><td>0.071</td></tr></table>

TABLE XXII

PHASE ORDER ON COD10K (SINGLE RUN). ALL SCHEDULES SHARE THE SAME WARM-UP, AND FOR CHALLENGE-FIRST VARIANTS THE FROZEN WARM-UP CHECKPOINT SERVES AS THE REFERENCE.
<table><tr><td>Training order</td><td> $M \downarrow$   $F _ { \beta } \uparrow$ </td><td> $E _ { \phi } \uparrow$   $S _ { \alpha } \uparrow$ </td><td> $S _ { \alpha } { } ^ { c } \uparrow$ </td><td>GT-CAG↓</td></tr><tr><td>Standard training</td><td>0.021 0.810 0.941 0.8780.778</td><td></td><td></td><td>0.056</td></tr><tr><td>SBFT after warm-up</td><td>0.023 0.785 0.924 0.8580.744</td><td></td><td></td><td>0.062</td></tr><tr><td>SBFT-P after warm-up</td><td>0.022 0.792 0.930 0.865 0.752</td><td></td><td></td><td>0.059</td></tr><tr><td>SBFT-P → SA-RCS</td><td>0.020 0.812 0.943 0.8760.771</td><td></td><td></td><td>0.052</td></tr><tr><td>SA-RCS → SBFT (conf.)</td><td>0.018 0.828 0.956 0.8840.803</td><td></td><td></td><td>0.043</td></tr><tr><td>SA-RCS → SBFT-P (ours)</td><td>0.017 0.836 0.963 0.889 0.832</td><td></td><td></td><td>0.035</td></tr><tr><td>Alternating SA-RCS / SBFT-P 0.018 0.830 0.957 0.885 0.816</td><td></td><td></td><td></td><td>0.039</td></tr><tr><td>Adaptive  $\breve { T } _ { c }$  trigger</td><td>0.017 0.836 0.962 0.8880.828</td><td></td><td></td><td>0.036</td></tr></table>

(c) Pixel weighting

T=70 leave performance stable over broad ranges, with the largest changes in GT-CAG in the two Phase-II safety scales: ε=0 raises GT-CAG to 0.041 and $\lambda _ { \mathrm { a n c } } { = } 0$ raises it to 0.039, reproducing the no-anchor configuration of Table XXI. Joint perturbation of the five non-transition constants keeps mean $S _ { \alpha }$ within 0.002 of the default at $\pm 2 0 \%$ and produces no abrupt collapse $\mathrm { a t \pm 4 0 \% }$ , and full sweeps are in the Supp.

Training dynamics. Fig. 6 shows $F _ { \beta }$ on the COD10K test set across training checkpoints, averaged over five runs with one-standard-deviation bands. All schedules, hyperparameters, and reported checkpoints are fixed before these retrospective evaluations, so no test result enters selection or tuning. The three trajectories are indistinguishable through the warm-up. Phase I then separates them twice over: both curricula pull

<table><tr><td>PUE variants</td><td>M↓  $F _ { \beta } \uparrow$   $E _ { \phi } \uparrow$ </td><td> $S _ { \alpha } \uparrow$ </td></tr><tr><td>No PUE</td><td>0.018 0.830 0.958</td><td>0.886</td></tr><tr><td>Hard mask</td><td>0.018 0.829 0.956</td><td>0.884</td></tr><tr><td>Soft only</td><td>0.018 0.830 0.957</td><td>0.885</td></tr><tr><td>+ decay only</td><td>0.017 0.832 0.959</td><td>0.886</td></tr><tr><td>+ floor only</td><td>0.017 0.834 0.961</td><td>0.888</td></tr><tr><td>+ floor + decay</td><td>0.017 0.836 0.963</td><td>0.889</td></tr></table>

TABLE XXIII <sub>M</sub>

TRUST PARAMETERIZATION (1/16 SEMI-SUPERVISION, PLUS POINT6 <sub>Dyn-Unc</sub> <sub>TDDS</sub>SUPERVISION FOR THE SPARSE-HUMAN PRESET). UNIFORM: EQUAL CUECOEFFICIENTS, SINGLE $\tau _ { \mathrm { t r u s t } } .$ ONE SHARED TRUST FOR ALL SOURCES,“MISMATCHED”: SUPERVISION RECEIVES AN INCORRECT SOURCE-TYPE<sub>4</sub>mPRESET (IN SEMI THE DENSE-HUMAN AND MODEL-DERIVED PRESETS ARE<sub>d</sub>SWAPPED, AND IN POINT THE SPARSE-HUMAN SOURCE RECEIVES THE<sub>a</sub>MODEL-DERIVED PRESET). THE<sub>2</sub>. $\pm 2 0 \%$ ROW REPORTS MEAN±STD OVER

TEN ORDER-PRESERVING RANDOM DRAWS. VARIATION IS ACROSS<sup>v</sup> CONFIGURATIONS, NOT SEEDS. PER-REGIME TUNING SELECTS $\tau _ { \mathrm { t r u s t } }$ ON HELD-OUT TRAINING DATA ONLY.
<table><tr><td>Trust design</td><td>M↓</td><td> $F _ { \beta } \uparrow$ </td><td> $E _ { \phi } \uparrow$ </td><td> $S _ { \alpha } \uparrow$ </td><td> $S _ { \alpha }$  (point)↑</td></tr><tr><td>Uniform cue</td><td>0.034</td><td>0.731</td><td>0.848</td><td>0.786</td><td>0.820</td></tr><tr><td>Single  $\tau _ { \mathrm { t r u s t } } { = } 0 . 5$ </td><td>0.033</td><td>0.738</td><td>0.854</td><td>0.791</td><td>0.826</td></tr><tr><td>Mismatched trust</td><td>0.036</td><td>0.719</td><td>0.838</td><td>0.777</td><td>0.812</td></tr><tr><td>Preset ±20%</td><td> $0 . 0 3 2 { \scriptstyle \pm 0 . 0 0 1 }$ </td><td></td><td></td><td>0.745±0.002 0.861±0.002 0.796±0.001</td><td>0.831±0.001</td></tr><tr><td>Source-type preset</td><td>0.031</td><td>0.748</td><td>0.863</td><td>0.797</td><td>0.834</td></tr><tr><td>Val.-tuned  $\tau _ { \mathrm { t r u s t } }$ </td><td>0.031</td><td>0.749</td><td>0.864</td><td>0.798</td><td>0.835</td></tr></table>

![](images/b3db88667c1b06cabb06ce696b292e58d661901b08f604d2ed3607d569dd628b.jpg)  
Fig. 6. Training dynamics on COD10K (RUN with PVT V2): $F _ { \beta }$ on the test set at each checkpoint, mean over five runs with one-standard-deviation bands. The marks on the epoch axis indicate $T _ { w } , T _ { c } ,$ , and T, and ten extra diagnostic epochs beyond T are shown for reference.

away from standard training, and PaceShift keeps a modest, <sub>0.95</sub>consistent margin over CurriSeg. After T the challenge briefly costs both recipes accuracy, but PaceShift drops less, recovers <sup>0.90</sup>earlier, and ends higher with the narrowest band. Ten extra <sub>t</sub>i<sup>o</sup>epochs change every curve by at most 0.001, so the $T { = } 7 0$ t<sup>e</sup>checkpoints are not cut short.

## J. Robustness and Calibration

This section reports how the trained model behaves when the input degrades and whether its confidence tracks its correctness. The mechanism behind both is analyzed in Sec. IV-K.

Robustness under diverse corruptions. Complementing the controlled single-band interventions of Sec. IV-K, we evaluate robustness under a broad suite of common image corruptions, grouped into four families: noise (Gaussian, shot, impulse),

![](images/2c58c3bcfb5a6b7c5726933acd5814663d4cf3dbdfe2190492453b12cbc344c1.jpg)  
Fig. 7. Robustness on COD10K as retention of each recipe’s own clean $S _ { \alpha }$ Severity 0 is the clean test set. At each severity, retention is averaged within each corruption family and then equally over the four families.  
TABLE XXIV

CALIBRATION AND CONFIDENCE DIAGNOSTICS. ECE: PIXEL-POOLED   
EXPECTED CALIBRATION ERROR, USED TO ASSESS CALIBRATION.   
GT-CAG: GROUND-TRUTH-STRATIFIED CONFIDENCE–ACCURACY GAP,   
THE MEAN OF THE GAPS ON GROUND-TRUTH FOREGROUND AND   
BACKGROUND PIXELS, $\mathrm { C A G } _ { \mathrm { b n d } } \colon$ THE SAME GAP WITHIN A DILATED   
BOUNDARY BAND, AND WCONF: MEAN CONFIDENCE ON INCORRECT   
PIXELS, ALL THREE BEING COMPLEMENTARY DIAGNOSTICS. TS [83]:   
POST-HOC TEMPERATURE SCALING, A SINGLE SCALAR FITTED ON   
HELD-OUT VALIDATION DATA. (A) RECIPE-LEVEL COMPARISON ON   
COD10K (SINGLE RUN), WHERE THE FINAL COLUMN IS THE DIRECT   
CONFIDENCE-REWARD CONTROL DEFINED IN THE SUPP. (B) THE GUARDED   
STAGE ACROSS TASKS AND SUPERVISION REGIMES (RUN WITH PVT V2   
FOR PIS AND TOD, SCALER FOR THE COD REGIMES).

(a) Recipe-level comparison on COD10K
<table><tr><td>Metrics</td><td>RUN</td><td>+TS</td><td>+CurriSeg</td><td>+PaceShift</td><td>Direct conf. reward</td></tr><tr><td>ECE↓</td><td>0.019</td><td>0.010</td><td>0.016</td><td>0.011</td><td>0.024</td></tr><tr><td>GT-CAG ↓</td><td>0.056</td><td>0.034</td><td>0.046</td><td>0.035</td><td>0.071</td></tr><tr><td> $\mathrm { C A G } _ { \mathrm { b n d } } \downarrow$ </td><td>0.121</td><td>0.102</td><td>0.096</td><td>0.074</td><td>0.148</td></tr><tr><td>WConf ↓</td><td>0.86</td><td>0.80</td><td>0.82</td><td>0.79</td><td>0.94</td></tr></table>

(b) Generality across tasks and regimes
<table><tr><td rowspan="2">Tasks / regimes</td><td colspan="4">SA-RCS only</td><td colspan="4">+ SBFT-P</td></tr><tr><td>ECE</td><td>GT-CAG</td><td> $\mathrm { C A G } _ { \mathrm { b n d } }$ </td><td>WConf ECE</td><td></td><td>GT-CAG</td><td> $\mathrm { C A G } _ { \mathrm { b n d } }$ </td><td>WConf</td></tr><tr><td>PIS</td><td>0.018</td><td>0.042</td><td>0.098</td><td>0.80</td><td>0.014</td><td>0.032</td><td>0.072</td><td>0.77</td></tr><tr><td>TOD</td><td>0.018</td><td>0.038</td><td>0.089</td><td>0.78</td><td>0.013</td><td>0.030</td><td>0.067</td><td>0.75</td></tr><tr><td>COD, point</td><td>0.019</td><td>0.054</td><td>0.127</td><td>0.86</td><td>0.015</td><td>0.040</td><td>0.091</td><td>0.81</td></tr><tr><td>COD, semi 1/16 0.017</td><td></td><td>0.051</td><td>0.119</td><td>0.85</td><td>0.014</td><td>0.037</td><td>0.085</td><td>0.80</td></tr></table>

TABLE XXV

blur (defocus, motion), weather and visibility (fog, frost), and digital or photometric (contrast, brightness, JPEG), each at five severities. Because the three recipes differ in clean accuracy, Fig. 7 reports the retention $S _ { \alpha } ( s ) / S _ { \alpha } ( 0 )$ , so that every curve starts at one and the separation reflects robustness relative to each model’s own clean performance. At each severity, retention is averaged first over corruptions within each family and then equally over the four families. Both recipes degrade more gracefully than the base schedule, and PaceShift retains more than CurriSeg at every nonzero severity, with the margin widening as corruption strengthens.

Calibration. Table XXIV reports pixel-pooled ECE for calibration, alongside three complementary confidence diagnostics. In panel (a), PaceShift reduces ECE from 0.019 to 0.011 on COD10K, below CurriSeg (0.016), and improves all three diagnostics. Post-hoc temperature scaling [83], which needs a separate labeled calibration split, reaches a slightly lower ECE and GT-CAG, whereas PaceShift obtains its calibration from training alone and gives a smaller boundary gap and lower confidence on incorrect pixels. Direct confidence reward retains competitive clean accuracy (Table XXI) but worsens ECE and the confidence diagnostics. Panel (b) shows consistent

VALIDATION OF PACESHIFT’S INTERNAL TRAINING SIGNALS.   
(A) PSEUDO-LABEL QUALITY ACROSS $\rho _ { i }$ QUINTILES ON THE UNLABELED SPLIT OF THE SEMI-SUPERVISED 1/16 REGIME. GROUND TRUTH IS   
WITHHELD FROM TRAINING AND USED ONLY POST HOC, AND r<sub>s</sub>=0.64 IS COMPUTED OVER INDIVIDUAL PSEUDO-LABELS BEFORE BINNING.   
(B) ALIGNMENT OF SAMPLE SCORES ON $\mathcal { D } _ { s }$ WITH THE UPDATE-DIRECTION PROXY A<sup>opt</sup>: SPEARMAN AND TOP-20% OVERLAP (RANDOM: 20%), REPORTED AS MEAN±STD OVER THE FOUR PHASE-I REFRESH CHECKPOINTS WITH AT LEAST TWO HISTORY ENTRIES. VARIATION IS ACROSS CHECKPOINTS, NOT TRAINING SEEDS.

(a) $\rho _ { i }$ vs. pseudo-label quality  
(b) Alignment with A<sup>opt</sup>
<table><tr><td>ρi quin.</td><td>IoU↑</td><td>bnd. F↑</td><td>pix. err. ↓</td></tr><tr><td>Q5 (high)</td><td>0.871</td><td>0.822</td><td>0.032</td></tr><tr><td>Q4</td><td>0.804</td><td>0.741</td><td>0.058</td></tr><tr><td>Q3</td><td>0.742</td><td>0.668</td><td>0.083</td></tr><tr><td>Q2</td><td>0.668</td><td>0.585</td><td>0.121</td></tr><tr><td>Q1 (low)</td><td>0.549</td><td>0.468</td><td>0.192</td></tr></table>

<table><tr><td>Scores</td><td>Spearman↑</td><td>Top-20%↑</td></tr><tr><td>Instant. loss</td><td>−0.25±0.02</td><td>11±1%</td></tr><tr><td>GraNd</td><td>0.20±0.01</td><td>23±1%</td></tr><tr><td>Forgetting</td><td>0.29±0.01</td><td>25±1%</td></tr><tr><td>ρi</td><td>0.32±0.01</td><td> $2 8 \pm 1 \%$ </td></tr><tr><td>ωi</td><td> $\mathbf { 0 . 5 3 \pm 0 . 0 2 }$ </td><td> $41 \pm 2 \%$ </td></tr></table>

TABLE XXVI

GUARD SELECTION AGAINST HELD-OUT GT, SHOWN ONLY WHERE $Q _ { \mathrm { s u p } }$ IS SURROGATE-BASED. TOP-1 IS THE RATE OF SELECTING A CANDIDATE WITHIN $1 0 ^ { - 3 }$ TRUE IOU OF THE ORACLE CANDIDATE, AND REGRET IS THE TRUE IOU LEFT ON THE TABLE. RANDOM REPORTS THE EXACT EXPECTATION UNDER UNIFORM SELECTION AMONG THE THREE CANDIDATES, INCLUDING TOLERANCE-INDUCED ORACLE TIES.
<table><tr><td rowspan="2">Selection rules</td><td colspan="2">Semi 1/16 (unlab.)</td><td colspan="2">Point</td></tr><tr><td>Top-1 (%) ↑</td><td>Regret ↓</td><td>Top-1 (%) ↑</td><td>Regret↓</td></tr><tr><td>Random (expected)</td><td>36.7</td><td>0.052</td><td>34.9</td><td>0.061</td></tr><tr><td>Confidence only</td><td>47.1</td><td>0.038</td><td>44.8</td><td>0.046</td></tr><tr><td>Consistency only</td><td>54.6</td><td>0.031</td><td>52.9</td><td>0.038</td></tr><tr><td>Guard w/o  $Q \mathrm { s u p }$ </td><td>58.3</td><td>0.027</td><td>57.1</td><td>0.033</td></tr><tr><td>Full guard Q(m)</td><td>71.8</td><td>0.014</td><td>68.4</td><td>0.019</td></tr></table>

improvements across various tasks and supervision regimes.

## K. Mechanism Analysis

We next assess whether $\rho _ { i }$ tracks supervision quality, ω<sub>i</sub> reflects update value, $Q ( m )$ ranks better candidates, and spectral post-training reduces high-frequency reliance.

1) Does reliability track supervision quality: In the triage of Table XIII, the full $\rho _ { i }$ gives the best corruption detection and retained quality. Removing the external proposal lowers AUROC by 3.5 points, and direct pseudo-mask–proposal agreement reaches 0.806, so $\rho _ { i }$ adds 2.1 points over direct external checking and the gain comes from fusing external and internal evidence, not from either alone. On naturally occurring pseudo-labels (Table XXV(a)), quality decreases monotonically across the five $\rho _ { i }$ quintiles, with a 32.2-point top-to-bottom IoU gap and a sample-level Spearman correlation of 0.64 with true IoU. Thus, in the pseudo-label regime, $\rho _ { i }$ ranks true supervision quality beyond model confidence.

2) Does the temporal weight align with useful updates: On a fixed diagnostic subset $\mathcal { D } _ { s } ,$ , we compare sample scores against an optimizer-conditioned first-order alignment proxy $\bar { A } _ { i } ^ { \mathrm { o p t } }$ , measured against a 200-image probe set $\mathcal { D } _ { p }$ held out from optimization and tuning. The proxy uses the unweighted loss and a frozen Adam preconditioner, so no compared weighting enters its own target. Correlations are computed independently over the four Phase-I refresh checkpoints with at least two history entries and then averaged. Full protocol details are in the Supp. Table XXV(b) shows that instantaneous loss is negatively correlated with update alignment, whereas $\rho _ { i }$ is moderately positive and $\omega _ { i }$ achieves the strongest correlation and top-20% overlap. The two therefore answer different questions: $\rho _ { i }$ asks whether supervision can be trusted, and $\omega _ { i }$ asks whether a sample is worth updating on now.

TABLE XXVII  
TEST-TIME FREQUENCY INTERVENTION ON COD10K (RUN, PVT V2). EACH CELL IS THE DROP FROM THAT MODEL’S OWN CLEAN $S _ { \alpha }$ . THE NAIVE EASY-FIRST ROW IS A CONVENTIONAL EASY-TO-HARD SCHEDULE, UNLIKE THE MATCHED-BUDGET CONTROL OF TABLE XIX.
<table><tr><td rowspan="2">Training</td><td rowspan="2">Clean↑</td><td colspan="3">drop under intervention</td></tr><tr><td>HF rem. ↓</td><td>HF noise↓</td><td>LF supp. (DC retained)</td></tr><tr><td>Standard</td><td>0.878</td><td>0.086</td><td>0.177</td><td>0.276</td></tr><tr><td>Naive easy-first CL</td><td>0.871</td><td>0.104</td><td>0.201</td><td>0.249</td></tr><tr><td>SA-RCS only</td><td>0.881</td><td>0.071</td><td>0.142</td><td>0.288</td></tr><tr><td>PaceShift</td><td>0.889</td><td>0.049</td><td>0.098</td><td>0.309</td></tr></table>

3) Does the guard select the better mask, and does the challenge reduce high-frequency reliance:

Guard selection. We evaluate samples for which $Q _ { \mathrm { s u p } }$ is surrogate-based, namely semi-supervised unlabeled and point supervised samples. Selection is scored by post-hoc Top-1 accuracy, the fraction for which $m ^ { + }$ is within $1 0 ^ { - 3 }$ of the highest true IoU among the three candidates, and by regret ma $\mathrm { x } _ { r } \mathrm { I o U } ( m ^ { r } , y ) - \mathrm { I o U } ( m ^ { + } , y )$ . Table XXVI shows that the full guard halves regret or better relative to the better single-cue rule, substantially exceeds the uniform-random Top-1 baseline, and improves over the guard without $Q _ { \mathrm { s u p } } .$ . Confidence is the weaker single cue, and consistency performs better but can still favor stable errors. Selection is also not trivial: the clean candidate accounts for at most 42% of selections.

Frequency intervention. We probe frequency reliance at an unseen cutoff of 0.70 of the Nyquist radius, outside the training radii {0.90, 0.95, 1.00}, using high-frequency removal, low-frequency suppression, and band-limited high-frequency noise, with full protocol details in the Supp. Table XXVII reports drops relative to each model’s clean score. Relative to standard training, PaceShift reduces sensitivity to highfrequency removal and noise by 43% and 45%, while becoming more sensitive to low-frequency suppression, consistent with reduced reliance on high-frequency content. Naive easy-first training is both less accurate and more sensitive to the highfrequency interventions, consistent with the shortcut hypothesis.

## V. DISCUSSION

## A. Why the Paradigm Works

The diagnostics of Sec. IV-K say more when considered together than separately. Reliability and update value are two different quantities: $\rho _ { i }$ ranks held-out target quality more effectively than the evaluated confidence-based criteria (Tables XXV(a) and XIII), whereas $\omega _ { i }$ is the score most positively associated with the optimizer-conditioned alignment proxy, and instantaneous loss is negatively associated with it (Table XXV(b)). The guard selects low-regret candidates that confidence alone would not (Table XXVI), both phases reduce high-frequency sensitivity (Table XXVII), and the proposed order outperforms reversed and challenge-first schedules (Table XXII). Together, these diagnostics support treating training order, supervision reliability, and post-training calibration as connected aspects of the training design.

## B. Scope and Limitations

Spectral intervention is most useful when targets and backgrounds share low-level statistics, as in camouflaged, polyp, and transparent segmentation, and this mechanism is most directly supported by the CECS diagnostics. The broader experiments evaluate the full training recipe without isolating the contribution of each phase (Table IX). When an object occupies only a few pixels, spectral attenuation may remove useful target evidence along with nuisance cues, and residual failures also include overlapping instances and heavy occlusion (see the Supp.). Reliability-aware scheduling may offer smaller gains when sample reliability varies little, and incomplete-supervision settings that use external teacher or proposal evidence depend on its quality. Finally, Phase II adds modest training-time computation (Table III) but no inference-time parameters or cost.

## VI. CONCLUSION

PaceShift turns CurriSeg’s setting-specific training recipe into a unified supervision-adaptive formulation and replaces its unguarded single-radius post-training with guarded multiradius preference learning. Across CECS tasks, incomplete supervision, foundation-model adaptation, and broader dense prediction, it improves accuracy, and targeted evaluations show improved pooled calibration and corruption robustness in the evaluated settings and support the proposed mechanism.

## REFERENCES

[1] D.-P. Fan, G.-P. Ji, G. Sun, M.-M. Cheng, J. Shen, and L. Shao, “Camouflaged object detection,” in CVPR, 2020, pp. 2777–2787.

[2] X. Zhao et al., “Spider: a unified framework for context-dependent concept segmentation,” in ICML, 2024, pp. 60 906–60 926.

[3] C. He et al., “Strategic preys make acute predators: Enhancing camouflaged object detectors by generating camouflaged objects,” in ICLR, 2024.

[4] F. Xiao et al., “A survey of camouflaged object detection and beyond,” CAAI Artif. Intell. Res., vol. 3, 2024, art. no. 9150044.

[5] C. He et al., “RIDE: Retinex-informed decoupling for exposing concealed objects,” arXiv preprint arXiv:2605.15450, 2026.

[6] W. Wang et al., “PVT v2: Improved baselines with Pyramid Vision Transformer,” Comput. Vis. Media, vol. 8, no. 3, pp. 415–424, 2022.

[7] D.-P. Fan, G.-P. Ji, M.-M. Cheng, and L. Shao, “Concealed object detection,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 44, no. 10, pp. 6024–6042, 2022.

[8] N.-T. Bui, D.-H. Hoang, Q.-T. Nguyen, M.-T. Tran, and N. Le, “MEGANet: Multi-scale edge-guided attention network for weak boundary polyp segmentation,” in WACV, 2024, pp. 7985–7994.

[9] Y. Pang, X. Zhao, T.-Z. Xiang, L. Zhang, and H. Lu, “ZoomNeXt: A unified collaborative pyramid network for camouflaged object detection,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 46, no. 12, pp. 9205–9220, 2024.

[10] C. He et al., “Camouflaged object detection with feature decomposition and edge reconstruction,” in CVPR, 2023, pp. 22 046–22 055.

[11] Y. Sun, C. Xu, J. Yang, H. Xuan, and L. Luo, “Frequency-spatial entanglement learning for camouflaged object detection,” in ECCV, 2024, pp. 343–360.

[12] C. He et al., “RUN: Reversible unfolding network for concealed object segmentation,” in ICML, 2025, pp. 22 853–22 864.

[13] D.-P. Fan, G.-P. Ji, P. Xu, M.-M. Cheng, C. Sakaridis, and L. Van Gool, “Advances in deep concealed scene understanding,” Visual Intell., vol. 1, no. 1, 2023, art. no. 16.

[14] C. He et al., “NUN: Nested unfolding network for real-world concealed object segmentation,” in ECCV, 2026.

[15] Y. Bengio, J. Louradour, R. Collobert, and J. Weston, “Curriculum learning,” in ICML, 2009, pp. 41–48.

[16] F. Xiao et al., “Beyond ground-truth: Leveraging image quality priors for real-world image restoration,” in CVPR, 2026, pp. 29 897–29 908.

[17] C. He, R. Zhang, F. Xiao, D. Zhang, Z. Cao, and S. Farsiu, “Refining context-entangled content segmentation via curriculum selection and anti-curriculum promotion,” in ICML, 2026, pp. 42 267–42 279.

[18] C. He et al., “Unfoldcrf: Structured mask refinement with imageconditioned latent regions,” arXiv preprint arXiv:2609.33996, 2026.

[19] H. Mei, X. Yang, Y. Wang, Y. Liu, and S. He, “Don’t hit me! glass detection in real-world scenes,” in CVPR, 2020, pp. 3687–3696.

[20] B. Dong, W. Wang, D.-P. Fan, J. Li, H. Fu, and L. Shao, “Polyp-PVT: Polyp segmentation with Pyramid Vision Transformers,” CAAI Artif. Intell. Res., vol. 2, 2023, art. no. 9150015.

[21] F. Yang et al., “Uncertainty-guided transformer reasoning for camouflaged object detection,” in ICCV, 2021, pp. 4146–4155.

[22] B. Yin et al., “CamoFormer: Masked separable attention for camouflaged object detection,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 46, no. 12, pp. 10 362–10 374, 2024.

[23] P. Soviany, R. T. Ionescu, P. Rota, and N. Sebe, “Curriculum learning: A survey,” Int. J. Comput. Vis., vol. 130, no. 6, pp. 1526–1565, 2022.

[24] M. P. Kumar, B. Packer, and D. Koller, “Self-paced learning for latent variable models,” in NeurIPS, vol. 23, 2010.

[25] T. Matiisen, A. Oliver, T. Cohen, and J. Schulman, “Teacher–student curriculum learning,” IEEE Trans. Neural Networks Learn. Syst., vol. 31, no. 9, pp. 3732–3740, 2020.

[26] L. Saglietti, S. Mannelli, and A. Saxe, “An analytical theory of curriculum learning in teacher-student networks,” in NeurIPS, vol. 35, 2022, pp. 21 113–21 127.

[27] A. Pentina, V. Sharmanska, and C. H. Lampert, “Curriculum learning of multiple tasks,” in CVPR, 2015, pp. 5492–5500.

[28] P. Soviany, “Curriculum learning with diversity for supervised computer vision tasks,” arXiv preprint arXiv:2009.10625, 2020.

[29] M. Toneva, A. Sordoni, R. Tachet des Combes, A. Trischler, Y. Bengio, and G. J. Gordon, “An empirical study of example forgetting during deep neural network learning,” in ICLR, 2019.

[30] C. He et al., “Hard, yet reducible: Controlled forward transfer for synthetic degradation curation,” arXiv preprint arXiv:2610.09849, 2026.

[31] M. Paul, S. Ganguli, and G. K. Dziugaite, “Deep learning on a data diet: Finding important examples early in training,” in NeurIPS, vol. 34, 2021, pp. 20 596–20 607.

[32] S. Swayamdipta et al., “Dataset cartography: Mapping and diagnosing datasets with training dynamics,” in EMNLP, 2020, pp. 9275–9293.

[33] G. Pleiss, T. Zhang, E. Elenberg, and K. Q. Weinberger, “Identifying mislabeled data using the area under the margin ranking,” in NeurIPS, vol. 33, 2020, pp. 17 044–17 056.

[34] M. He, S. Yang, T. Huang, and B. Zhao, “Large-scale dataset pruning with dynamic uncertainty,” in CVPRW, 2024, pp. 7713–7722.

[35] X. Zhang, J. Du, Y. Li, W. Xie, and J. T. Zhou, “Spanning training progress: Temporal dual-depth scoring (TDDS) for enhanced dataset pruning,” in CVPR, 2024, pp. 26 223–26 232.

[36] Z. Qin et al., “InfoBatch: Lossless training speed up by unbiased dynamic data pruning,” in ICLR, 2024.

[37] Q. Zhou, J. Gao, and Q. Wang, “Scale efficient training for large datasets,” in CVPR, 2025, pp. 20 458–20 467.

[38] S. Liu, K. Liu, W. Zhu, Y. Shen, and C. Fernandez-Granda, “Adaptive early-learning correction for segmentation from noisy annotations,” in CVPR, 2022, pp. 2606–2616.

[39] A. Tarvainen and H. Valpola, “Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results,” in NeurIPS, vol. 30, 2017.

[40] K. Sohn et al., “FixMatch: Simplifying semi-supervised learning with consistency and confidence,” in NeurIPS, vol. 33, 2020, pp. 596–608.

[41] E. Arazo, D. Ortego, P. Albert, N. E. O’Connor, and K. McGuinness, “Pseudo-labeling and confirmation bias in deep semi-supervised learning,” in IJCNN, 2020, pp. 1–8.

[42] C. He et al., “Segment concealed objects with incomplete supervision,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 47, no. 9, pp. 7832–7851, 2025.

[43] C. He, D. Zhang, L. Tang, Z. Yang, F. Xiao, and S. Farsiu, “When specialists meet generalists: Segmenter-coordinated asymmetric learning for label-deficient concealed object segmentation,” in ECCV, 2026.

[44] H. Shah, K. Tamuly, A. Raghunathan, P. Jain, and P. Netrapalli, “The pitfalls of simplicity bias in neural networks,” in NeurIPS, vol. 33, 2020, pp. 9573–9585.

[45] M. Pezeshki, O. Kaba, Y. Bengio, A. C. Courville, D. Precup, and G. Lajoie, “Gradient starvation: A learning proclivity in neural networks,” in NeurIPS, vol. 34, 2021, pp. 1256–1272.

[46] K. L. Hermann, T. Chen, and S. Kornblith, “The origins and prevalence of texture bias in convolutional neural networks,” in NeurIPS, vol. 33, 2020, pp. 19 000–19 015.

[47] N. Rahaman et al., “On the spectral bias of neural networks,” in ICML, 2019, pp. 5301–5310.

[48] A. Kirillov et al., “Segment anything,” in ICCV, 2023, pp. 4015–4026.

[49] T. Chen et al., “SAM-Adapter: Adapting segment anything in underperformed scenes,” in ICCVW, 2023, pp. 3367–3375.

[50] E. J. Hu et al., “LoRA: Low-rank adaptation of large language models,” in ICLR, 2022.

[51] T. Lüddecke and A. Ecker, “Image segmentation using text and image prompts,” in CVPR, 2022, pp. 7086–7096.

[52] Y. Zhang et al., “EVF-SAM: Early vision-language fusion for textprompted segment anything model,” Image Vis. Comput., vol. 173, 2026, art. no. 106028.

[53] S. Cho, H. Shin, S. Hong, A. Arnab, P. H. Seo, and S. Kim, “CAT-Seg: Cost aggregation for open-vocabulary semantic segmentation,” in CVPR, 2024, pp. 4113–4123.

[54] X. Zhang, B. Yin, Z. Lin, Q. Hou, D.-P. Fan, and M.-M. Cheng, “Referring camouflaged object detection,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 47, no. 5, pp. 3597–3610, 2025.

[55] R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn, “Direct preference optimization: Your language model is secretly a reward model,” in NeurIPS, vol. 36, 2023, pp. 53 728–53 741.

[56] Z.-Q. J. Xu, Y. Zhang, T. Luo, Y. Xiao, and Z. Ma, “Frequency principle: Fourier analysis sheds light on deep neural networks,” Commun. Comput. Phys., vol. 28, no. 5, pp. 1746–1767, 2020.

[57] N. Tishby and N. Zaslavsky, “Deep learning and the information bottleneck principle,” in ITW, 2015, pp. 1–5.

[58] Q. Zhai, X. Li, F. Yang, C. Chen, H. Cheng, and D.-P. Fan, “Mutual graph learning for camouflaged object detection,” in CVPR, 2021, pp. 2997–3007.

[59] M. Zhang, S. Xu, Y. Piao, D. Shi, S. Lin, and H. Lu, “PreyNet: Preying on camouflaged objects,” in ACM MM, 2022, pp. 5323–5332.

[60] H. Zhu et al., “I can find you! boundary-guided separated attention network for camouflaged object detection,” in AAAI, vol. 36, 2022, pp. 3608–3616.

[61] K. Sun, Z. Chen, X. Lin, X. Sun, H. Liu, and R. Ji, “Conditional diffusion models for camouflaged and salient object detection,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 47, no. 4, pp. 2833–2848, 2025.

[62] Z. Luo et al., “VSCode: General visual salient and camouflaged object detection with 2D prompt learning,” in CVPR, 2024, pp. 17 169–17 180.

[63] B. Jan, A. H. El-Maleh, A. J. Siddiqui, A. Bais, and S. Anwar, “C3Net: Context-contrast network for camouflaged object detection,” arXiv preprint arXiv:2511.12627, 2025.

[64] T.-N. Le, T. V. Nguyen, Z. Nie, M.-T. Tran, and A. Sugimoto, “Anabranch network for camouflaged object segmentation,” Comput. Vis. Image Underst., vol. 184, pp. 45–56, 2019.

[65] P. Skurowski, H. Abdulameer, J. Błaszczyk, T. Depta, A. Kornacki, and P. Kozieł, “Animal camouflage analysis: CHAMELEON database,” Unpublished manuscript, 2018.

[66] Y. Lv et al., “Simultaneously localize, segment and rank the camouflaged objects,” in CVPR, 2021, pp. 11 591–11 601.

[67] D. Jha et al., “Kvasir-SEG: A segmented polyp dataset,” in MMM, 2020, pp. 451–462.

[68] J. Bernal, F. J. Sánchez, G. Fernández-Esparrach, D. Gil, C. Rodríguez, and F. Vilariño, “WM-DOVA maps for accurate polyp highlighting in colonoscopy: Validation vs. saliency maps from physicians,” Comput. Med. Imaging Graph., vol. 43, pp. 99–111, 2015.

[69] J. Bernal, J. Sánchez, and F. Vilariño, “Towards automatic polyp detection with a polyp appearance model,” Pattern Recognit., vol. 45, no. 9, pp. 3166–3182, 2012.

[70] J. Silva, A. Histace, O. Romain, X. Dray, and B. Granado, “Toward embedded detection of polyps in WCE images for early diagnosis of colorectal cancer,” Int. J. Comput. Assist. Radiol. Surg., vol. 9, no. 2, pp. 283–293, 2014.

[71] J. Lin and Z. He, “Rich context aggregation with reflection prior for glass surface detection,” in CVPR, 2021, pp. 13 415–13 424.

[72] M. Zhang, R. Zhang, Y. Yang, H. Bai, J. Zhang, and J. Guo, “ISNet: Shape matters for infrared small target detection,” in CVPR, 2022, pp. 877–886.

[73] T. F. Y. Vicente, L. Hou, C.-P. Yu, M. Hoai, and D. Samaras, “Largescale training of shadow detectors with noisily-annotated shadow examples,” in ECCV, 2016, pp. 816–832.

[74] N. Codella et al., “Skin lesion analysis toward melanoma detection 2018: A challenge hosted by the International Skin Imaging Collaboration (ISIC),” arXiv preprint arXiv:1902.03368, 2019.

[75] H. Bogunovic´ et al., “RETOUCH: The retinal OCT fluid detection and segmentation benchmark and challenge,” IEEE Trans. Med. Imaging, vol. 38, no. 8, pp. 1858–1874, 2019.

[76] X. Qin, H. Dai, X. Hu, D.-P. Fan, L. Shao, and L. Van Gool, “Highly accurate dichotomous image segmentation,” in ECCV, 2022, pp. 38–56.

[77] L. Wang et al., “Learning to detect salient objects with image-level supervision,” in CVPR, 2017, pp. 136–145.

[78] B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso, and A. Torralba, “Scene parsing through ADE20K dataset,” in CVPR, 2017, pp. 633–641.

[79] T.-Y. Lin et al., “Microsoft COCO: Common objects in context,” in ECCV, 2014, pp. 740–755.

[80] R. Achanta, S. Hemami, F. Estrada, and S. Süsstrunk, “Frequency-tuned salient region detection,” in CVPR, 2009, pp. 1597–1604.

[81] D.-P. Fan, C. Gong, Y. Cao, B. Ren, M.-M. Cheng, and A. Borji, “Enhanced-alignment measure for binary foreground map evaluation,” in IJCAI, 2018, pp. 698–704.

[82] D.-P. Fan, M.-M. Cheng, Y. Liu, T. Li, and A. Borji, “Structure-measure: A new way to evaluate foreground maps,” in ICCV, 2017, pp. 4548– 4557.

[83] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in ICML, 2017, pp. 1321–1330.

[84] S. Jain et al., “CoInNet: A convolution-involution network with a novel statistical attention for automatic polyp segmentation,” IEEE Trans. Med. Imaging, vol. 42, no. 12, pp. 3987–4000, 2023.

[85] W. Wang, H. Sun, and X. Wang, “LSSNet: A method for colon polyp segmentation based on local feature supplementation and shallow feature supplementation,” in MICCAI, 2024, pp. 446–456.

[86] H. He et al., “Enhanced boundary learning for glass-like object segmentation,” in ICCV, 2021, pp. 15 859–15 868.

[87] D. Han et al., “Internal-external boundary attention fusion for glass surface segmentation,” Neural Netw., vol. 195, 2026, art. no. 108232.

[88] T. Yan et al., “GhostingNet: A novel approach for glass surface detection with ghosting cues,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 47, no. 1, pp. 323–337, 2025.

[89] K. Fan, C. Wang, Y. Wang, C. Wang, R. Yi, and L. Ma, “RFENet: towards reciprocal feature evolution for glass segmentation,” in IJCAI, 2023, pp. 717–725.

[90] X. Hu et al., “High-resolution iterative feedback network for camouflaged object detection,” in AAAI, vol. 37, no. 1, 2023, pp. 881–889.

[91] X. Yang, H. Zhu, G. Mao, and S. Xing, “OAFormer: Occlusion aware transformer for camouflaged object detection,” in ICME, 2023, pp. 1421–1426.

[92] M. Zhang, Y. Wang, J. Guo, Y. Li, X. Gao, and J. Zhang, “IRSAM: Advancing segment anything model for infrared small target detection,” in ECCV, 2024, pp. 233–249.

[93] Z. Yang and S. Farsiu, “Directional connectivity-based segmentation of medical images,” in CVPR, 2023, pp. 11 525–11 535.

[94] Q. Yu, X. Zhao, Y. Pang, L. Zhang, and H. Lu, “Multi-view aggregation network for dichotomous image segmentation,” in CVPR, 2024, pp. 3921–3930.

[95] N. Liu, Z. Luo, N. Zhang, and J. Han, “VST++: Efficient and stronger visual saliency transformer,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 46, no. 11, pp. 7300–7316, 2024.

[96] Z. Ni, X. Chen, Y. Zhai, Y. Tang, and Y. Wang, “Context-guided spatial feature reconstruction for efficient semantic segmentation,” in ECCV, 2024, pp. 239–255.

[97] B. Cheng, I. Misra, A. G. Schwing, A. Kirillov, and R. Girdhar, “Masked attention mask transformer for universal image segmentation,” in CVPR, 2022, pp. 1290–1299.

[98] B. Han et al., “Co-teaching: Robust training of deep neural networks with extremely noisy labels,” in NeurIPS, vol. 31, 2018.

[99] J. Shu et al., “Meta-Weight-Net: Learning an explicit mapping for sample weighting,” in NeurIPS, vol. 32, 2019.

[100] S. Yuan, R. Lin, L. Feng, B. Han, and T. Liu, “Instance-dependent early stopping,” in ICLR, 2025.

[101] X. Xia, J. Liu, J. Yu, X. Shen, B. Han, and T. Liu, “Moderate coreset: A universal method of data selection for real-world data-efficient deep learning,” in ICLR, 2023.

[102] H. Zheng, R. Liu, F. Lai, and A. Prakash, “Coverage-centric coreset selection for high pruning rates,” in ICLR, 2023.

[103] H. Zheng et al., “ELFS: Label-free coreset selection with proxy training dynamics,” in ICLR, 2025.

[104] C. He et al., “Weakly-supervised concealed object segmentation with SAM-based pseudo labeling and multi-scale feature grouping,” in NeurIPS, vol. 36, 2023, pp. 30 726–30 737.

[105] S. Chakraborty, S. Naha, M. Bastan, A. K. K. C., and D. Samaras, “Unsupervised and semi-supervised co-salient object detection via segmentation frequency statistics,” in WACV, 2024, pp. 332–342.

[106] N. Carion, L. Gustafson, Y.-T. Hu, S. Debnath, R. Hu et al., “SAM 3: Segment anything with concepts,” arXiv preprint arXiv:2511.16719, 2025.

[107] T. Chen et al., “SAM2-Adapter: Evaluating & adapting segment anything 2 in downstream tasks: Camouflage, shadow, medical image segmentation, and more,” arXiv preprint arXiv:2408.04579, 2024.

[108] T. Chen et al., “SAM3-Adapter: Efficient adaptation of segment anything 3 for camouflage object segmentation, shadow detection, and medical image segmentation,” arXiv preprint arXiv:2511.19425, 2025.

[109] S. Jang, J. Yun, J. Kwon, E. Lee, and Y. Kim, “DIAL: Dense image-text alignment for weakly supervised semantic segmentation,” in ECCV, 2024, pp. 248–266.

[110] B. Zhang, S. Yu, Y. Wei, Y. Zhao, and J. Xiao, “Frozen CLIP: A strong backbone for weakly supervised semantic segmentation,” in CVPR, 2024, pp. 3796–3806.

[111] Z. Yang, Y. Meng, K. Fu, F. Tang, S. Wang, and Z. Song, “Exploring CLIP’s dense knowledge for weakly supervised semantic segmentation,” in CVPR, 2025, pp. 20 223–20 232.

[112] Q. Yu, J. He, X. Deng, X. Shen, and L.-C. Chen, “Convolutions die hard: Open-vocabulary segmentation with single frozen convolutional CLIP,” in NeurIPS, vol. 36, 2023, pp. 32 215–32 234.

[113] Y. Pang, X. Zhao, J. Zuo, L. Zhang, and H. Lu, “Open-vocabulary camouflaged object segmentation,” in ECCV, 2024, pp. 476–495.

[114] K. Zhao et al., “Open-vocabulary camouflaged object segmentation with cascaded vision language models,” Comput. Vis. Media, vol. 12, no. 2, pp. 473–492, 2026.

[115] Y.-H. Wu, Z.-X. Zhu, Y. Wang, L. Zhen, and D.-P. Fan, “RefOnce: Distilling references into a prototype memory for referring camouflaged object detection,” arXiv preprint arXiv:2511.20989, 2025.

[116] M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman, “The PASCAL visual object classes (VOC) challenge,” Int. J. Comput. Vis., vol. 88, no. 2, pp. 303–338, 2010.

Chunming He received the B.S. degree from Nanjing University of Posts and Telecommunications, China, and the M.E. degree from Tsinghua University, China. He is currently a Ph.D. student at Duke University, Durham, USA. His research interests include computer vision and biomedical image analysis.

Dingming Zhang received the B.S. degree from Zhejiang Sci-Tech University, China. He is currently pursuing the M.S. degree at Duke University, Durham, USA. His research interests include computer vision.

Fengyang Xiao received the B.S. degree from Nanjing University of Posts and Telecommunications, China, and the M.S. degree from Sun Yat-sen University, China. She is currently a Ph.D. student at Duke University, Durham, USA. Her research interests include differential equations and computer vision.

Zhiwen Cao received the Ph.D. degree from Purdue University, USA. He is currently a Gen AI Researcher at Adobe. His research interests include image, video and audio generation and editing.

Sina Farsiu (Fellow, IEEE) received his Ph.D. degree in electrical engineering from the University of California, Santa Cruz in 2005. He is now the Anderson-Rupp Professor of BME, with secondary appointments at the Departments of ECE and CS at Duke University. He has served as the Senior Area Editor of IEEE Transactions on Image Processing, Deputy Editor of the Biomedical Optics Express, and the Associate Editor of SIAM Journal on Imaging Sciences. He received the “Outstanding Member of the Editorial Board Award” from the IEEE Signal Processing Society in 2018. He is a Fellow of IEEE, Optica, SPIE, ARVO, and AIMBE.