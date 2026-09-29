# Mixed-Prior Decision Risk for Calibrated Uncertainty Estimation in Open-Set Recognition<sup>1</sup>

L. A. Erlygin<sup>∗</sup> , P. D. Proskura<sup>∗∗∗</sup> , A. A. Zaytsev<sup>∗</sup> <sup>∗∗</sup>

∗Skolkovo Institute of Science and Technology (Skoltech), Moscow, Russia ∗∗Risk Management, Sber, Moscow, Russia

∗∗∗The Institute for Information Transmission Problems (IITP RAS), Moscow, Russia Received September 15, 2025

Abstract—In open-set recognition (OSR), a probe must either be identified as one of the known gallery classes or rejected as unknown, so three error types coexist: false acceptance, false rejection, and misidentification. An uncertainty score for selective recognition should rank probes by the risk of the decision the system has made.

Bayesian gallery-aware models such as Holistic Uncertainty Estimation (HolUE) summarize the posterior over known and unknown classes by Kullback–Leibler (KL) divergence components and map them to an uncertainty score with a supervised nonlinear calibrator. We show that the KL summary is not generally monotone in decision risk: linear fusion of the KL components tuned on validation data yields negative filtering quality on several benchmarks.

We propose MPRisk, a mixed-prior posterior decision-risk score that keeps the same Bayesian posterior but directly scores the error events associated with the selected decision: falseacceptance, misidentification, and false-rejection risks, plus a non-specificity penalty for rejections, enabled by modeling unknown identities as a continuous component. Four nonnegative weights tuned on a validation set sufice for ranking; no nonlinear supervised model is required. Across nine image, audio, and text benchmarks, MPRisk achieves the best or tied-best Prediction Rejection Ratio at every operating point on the image and audio benchmarks and on most text operating points, with bootstrap-confirmed gains over HolUE on five benchmarks (up to +0.19 PRR) at comparable or lower runtime.

KEYWORDS: machine learning, open-set recognition, uncertainty estimation, selective prediction, Bayesian decision theory, probabilistic embeddings, risk decomposition.

## 1. INTRODUCTION

Open-set recognition (OSR) is the standard formulation for recognition systems deployed in open-world environments: the system must both identify samples from known classes and reject samples from unknown classes. This setting appears in face recognition, speaker identification, whale and dolphin identification, authorship attribution, intent classification, and topic recognition [1,2,3,4,5,6]. A standard OSR pipeline compares the probe embedding to the gallery prototypes by cosine similarity: the probe is accepted if the best similarity exceeds a threshold and is then assigned the identity of the closest prototype; otherwise it is rejected as unknown. With modern metric-learning losses such as ArcFace and CosFace, this paradigm achieves high recognition accuracy [7, 8].

High accuracy alone is not suficient in risk-sensitive applications. The system must also estimate the reliability of each decision, so that unreliable decisions can be deferred to a human operator or resolved by acquiring another sample. This is the selective-recognition setting, where an uncertainty score is evaluated by how well it ranks erroneous decisions ahead of correct ones [9, 10]. In OSR the ranking task is non-trivial because three qualitatively diferent errors coexist:

1. False acceptance (FA): an unknown probe is accepted as a known gallery class.

2. False rejection (FR): a known probe is rejected as unknown.

3. Misidentification (ID): a known probe is accepted but assigned the wrong identity.

Gallery-aware Bayesian models address this multiplicity by reconstructing a posterior distribution over known and unknown classes. Probabilistic embeddings such as PFE and SCF capture sample-quality ambiguity [11,12]; gallery-aware models additionally use the relative position of the probe and the gallery prototypes [13]. Holistic Uncertainty Estimation (HolUE) combines both sources: it summarizes the posterior by two Kullback–Leibler (KL) divergence components and maps them to an uncertainty score with a supervised nonlinear calibrator trained on validation data.

The starting point of this paper is a general question: are information-gain summaries appropriate for ranking the risk of an OSR decision? KL divergence measures how far the posterior moved from the prior, whereas selective recognition requires a score aligned with the probability or cost of the actual decision error. These two orderings can disagree: a posterior split between two plausible gallery identities is far from a uniform prior, yet the maximum-posterior decision carries high risk. Section 3.4 makes this mismatch precise, and Section 5.6 confirms it empirically: linear fusion of the HolUE KL components, tuned on validation data, produces negative filtering quality on several benchmarks, while the same features perform well under HolUE’s nonlinear calibration.

We propose MPRisk, a mixed-prior posterior decision-risk score that avoids the need for nonlinear calibration by scoring the error events of the selected decision directly. MPRisk reuses the Bayesian posterior over known and unknown classes and combines four components: falseacceptance risk $r _ { \mathrm { F A } }$ , misidentification risk $r _ { \mathrm { I D } } .$ , false-rejection risk $r _ { \mathrm { F R } }$ , and a reject non-specificity penalty r . The last component exists only because the unknown part of the class space is modeled as a continuous component representing unknown identities: it asks whether a rejection is supported by a specific unknown-identity hypothesis or merely by difuse evidence, and thereby detects poor-quality in-gallery probes that are confidently but wrongly rejected. The ranking score contains no nonlinear fitted model; four nonnegative weights are selected on validation data, and an optional monotone calibration step converts the score into an error-probability estimate for reliability analysis. Figure 1 summarizes the approach.

We make three contributions:

– Analysis. We show that KL-based summaries of the gallery-aware posterior are not generally ordered by the risk of the selected OSR decision, and that linear fusion of the HolUE KL features performs poorly on several benchmarks while nonlinear calibration recovers useful rankings.

– Method. We derive MPRisk, which combines posterior terms for false acceptance, misidentification, and false rejection with a closed-form non-specificity penalty for rejected probes based on the continuous unknown component of the mixed prior.

– Evaluation. Under matched validation budgets we compare MPRisk with KL-based, posthoc, and supervised scores on nine image, audio, and text benchmarks, including component ablations, error-type analysis, calibration diagnostics, and runtime measurements.

Section 2 reviews related work; Section 3 formalizes the problem and the KL–risk mismatch; Section 4 presents MPRisk; Section 5 reports experiments; Sections 6–7 discuss and conclude.

![](images/79673ceb251e9c8f9f5cb82c0b6ccbdc40001e14729ce737210a646cf086294a.jpg)  
Figure 1. MPRisk overview. (a) Embedding space of a fixed OSR system with K gallery classes (prototypes , samples ■) under the mixed prior: unknown identities form a continuous component $c \in ( K , K + 1 ]$ (tinted reject region); dashed lines are decision boundaries; blue shading shows the decision-risk field $1 - \operatorname* { m a x } _ { a } p ( a \mid \mathbf { x } )$ Markers show the three error types: false acceptance (⋆), misidentification (♦), and false rejection $( \times )$ a corrupted known probe whose difuse embedding distribution (green density, low $\kappa _ { \mathbf { x } } )$ yields a non-specific unknown posterior, i.e., high $\mathcal { N } _ { 0 } ( \mathbf { x } )$ . (b) Top: the KL summary measures information gain and can invert the risk ordering of two posteriors (A vs. B), which is why KL features require the nonlinear calibration of HolUE. Bottom: MPRisk scores the selected decision with four components r<sub>FA</sub>, r<sub>ID</sub>, r<sub>FR</sub>, and $r _ { \mathrm { N S } } = P _ { 0 } \mathcal { N } _ { 0 } ;$ stacked bars show the components at the three marked probes — for the confidently rejected corrupted probe only $r _ { \mathrm { N S } }$ flags the error. The final score $u \lambda$ combines the components with four nonnegative validation-tuned weights.

## 2. RELATED WORK

## 2.1. Open-set recognition and open-world learning

Open-set recognition was introduced to describe recognition systems that must handle classes absent from training or enrollment [1]. The broader open-world learning literature studies unknownclass rejection, open-set classification, and incremental discovery of new classes [14,15]. Deep openset methods such as OpenMax and DOC modify the classifier head or activation statistics to reduce open-space risk [16,17]; a further line of work improves the discriminative separation of known and unknown classes with specialized losses and architectures [14]. Most of this literature improves recognition or unknown-detection accuracy, whereas we estimate the reliability of decisions of a fixed OSR system without changing the embedding model, gallery, or acceptance rule.

## 2.2. Open-set recognition in text

Several text tasks naturally follow the OSR protocol. In intent classification a conversational system must reject out-of-scope requests; CLINC150 is the standard benchmark [5]. Topic classification datasets such as Yahoo Answers, AGNews, and DBPedia are converted into open-set protocols by treating a subset of topics as known [6, 18]. Authorship attribution and verification are inherently open-set [19, 20, 21]; PAN protocols provide challenging settings with dynamic galleries [22, 23]. Our previous work adapted gallery-aware KL-based uncertainty estimation to open-set text classification; here we use both modality groups within the identical Bayesian model.

## 2.3. Aleatoric uncertainty and probabilistic embeddings

Uncertainty in deep learning is commonly decomposed into epistemic and aleatoric components [24, 25, 26, 27]; in our fixed-backbone, fixed-gallery pipeline the dominant source is aleatoric, arising from the relation between a probe and the gallery. Probabilistic embeddings represent this uncertainty directly: PFE predicts feature uncertainty in Euclidean space [11], and the SCF model predicts a von Mises–Fisher (vMF) distribution on the unit hypersphere [12],

$$
p ( \mathbf { z } | \mathbf { x } ) = C _ { d } ( \kappa _ { \mathbf { x } } ) \exp ( \kappa _ { \mathbf { x } } \pmb { \mu } _ { \mathbf { x } } ^ { \top } \mathbf { z } ) ,
$$

where $\pmb { \mu } _ { \mathbf { x } }$ is a unit mean direction and $\kappa _ { \mathbf { x } }$ a concentration [28]; large $\kappa _ { \mathbf { x } }$ corresponds to a high-quality embedding. MPRisk uses this vMF distribution twice: to average the embedding-conditioned posterior into probe-level class probabilities and to obtain a closed-form non-specificity of the unknown-identity posterior (Section 4.2).

## 2.4. Selective prediction and calibration

Selective prediction evaluates whether uncertain samples are rejected before confident ones [9,10]. Classical decision theory prescribes rejecting when the conditional risk of the best action exceeds the rejection cost (Chow’s rule); under 0–1 loss the optimal score is the posterior error probability $1 - \operatorname* { m a x } _ { a } p ( a | \mathbf { x } )$ , not an entropy- or divergence-based functional. MPRisk instantiates this principle for the OSR action space, where the loss structure difers across accept and reject decisions. Calibration, in contrast, concerns whether predicted probabilities match empirical frequencies [29, 30, 31]; we treat ranking (evaluated by the Prediction Rejection Ratio [32]) and probability calibration (evaluated by ECE) as separate goals.

## 2.5. Relation to prior work

We are not aware of prior work that decomposes OSR uncertainty into decision-conditioned posterior risks while retaining a continuous unknown-identity component. Existing gallery-aware estimators use raw posterior confidence [13] or information-theoretic summaries with supervised calibration; existing selective-prediction scores operate on logits or sample quality and ignore the accept/reject decision structure.

## 3. BACKGROUND AND PROBLEM STATEMENT

## 3.1. Open-set recognition protocol

Let $\mathcal { G } = \{ g _ { 1 } , \dotsc , g _ { K } \}$ be a gallery of known classes, each represented by a prototype $\pmb { \mu _ { c } } \in \mathbb { S } ^ { d - 1 }$ At inference the system receives a probe x and applies the fixed decision rule

$$
s ( \mathbf { x } ) = \operatorname* { m a x } _ { c \in \{ 1 , \ldots , K \} } \mu _ { c } ^ { \top } \mathbf { z } ( \mathbf { x } ) , \qquad \mathbf { x } \mathrm { ~ i s ~ a c c e p t e d ~ i f ~ } s ( \mathbf { x } ) \geq \tau ,
$$

$$
\hat { c } ( { \bf x } ) = \underset { c \in \{ 1 , . . . , K \} } { \mathrm { a r g m a x } } ~ { \pmb \mu } _ { c } ^ { \top } { \bf z } ( { \bf x } ) \quad \mathrm { i f ~ a c c e p t e d } .
$$

Uncertainty estimation does not alter this decision; it assigns a score used to decide whether to trust it. All compared methods therefore share identical recognition performance at zero filtering and difer only in how they order probes for rejection-by-uncertainty.

$$
\mathrm { \Delta H H \Phi O P M A I I / H O H H b I E ~ I I P O I I E C C b I ~ T O M ~ 2 5 ~ \mathrm { \sc ~ \mathcal { N } ^ { o } ~ 1 ~ } ~ 2 0 2 5 ~ \mathrm { \sc ~ \mathcal { N } ^ { o } ~ 2 ~ } }
$$

## 3.2. What the uncertainty score should measure

For selective OSR the ideal score ranks probes by the probability (or cost) of a recognition error given the decision made. If the probe was accepted, the relevant errors are false acceptance and misidentification; if it was rejected, the relevant error is false rejection. The posterior terms must therefore be conditioned on the action taken: the posterior unknown probability $P _ { 0 } ( \mathbf { x } )$ is a falseacceptance risk only for accepted probes, while for rejected probes it is evidence that the rejection was correct.

## 3.3. Bayesian mixed-prior posterior

We adopt the Bayesian model over embeddings and class labels used by gallery-aware recognition [13]. The class variable is mixed:

$$
c \in \{ 1 , \ldots , K \} \cup ( K , K + 1 ] ,
$$

with discrete values for known classes and the continuous interval indexing the continuum of unknown identities. The prior is

$$
p ( c ) = \frac { 1 - \beta } { K } \sum _ { i = 1 } ^ { K } \delta ( c - i ) + \beta \mathbb { I } \{ c \in ( K , K + 1 ] \} ,
$$

where $\delta$ is the Dirac delta function, I is the indicator function, and $\beta \in ( 0 , 1 )$ is the prior probability mass assigned to the unknown continuum, i.e., the prior probability that a probe belongs to an identity outside the gallery. Known-class embedding densities are vMF with shared concentration $\kappa _ { g } \mathrm { : }$

$$
p ( \mathbf { z } | c = i ) = C _ { d } ( \kappa _ { g } ) \exp ( \kappa _ { g } \mu _ { i } ^ { \top } \mathbf { z } ) ,
$$

each unknown identity $c \in ( K , K + 1 ]$ has a point embedding $\pmb { \mu } _ { c } ^ { \circ }$ distributed uniformly on $\mathbb { S } ^ { d - 1 }$ ， so the aggregate unknown embedding density is $p _ { 0 } ( \mathbf { z } ) = 1 / S _ { d - 1 }$ , where $S _ { d - 1 }$ is the surface area of $\mathbb { S } ^ { d - 1 }$ , and the marginal is

$$
m ( \mathbf { z } ) = \frac { 1 - \beta } { K } \sum _ { i = 1 } ^ { K } p ( \mathbf { z } | c = i ) + \frac { \beta } { S _ { d - 1 } } .
$$

The probe-level posterior integrates over the SCF embedding distribution:

$$
P _ { i } ( \mathbf { x } ) = \int _ { \mathbb { S } ^ { d - 1 } } \frac { \frac { 1 - \beta } { K } p ( \mathbf { z } | c = i ) } { m ( \mathbf { z } ) } p ( \mathbf { z } | \mathbf { x } ) d \mathbf { z } , \qquad P _ { 0 } ( \mathbf { x } ) = \int _ { \mathbb { S } ^ { d - 1 } } \frac { \beta / S _ { d - 1 } } { m ( \mathbf { z } ) } p ( \mathbf { z } | \mathbf { x } ) d \mathbf { z } ,
$$

with $\begin{array} { r } { P _ { 0 } ( \mathbf { x } ) + \sum _ { i = 1 } ^ { K } P _ { i } ( \mathbf { x } ) = 1 } \end{array}$

Mean-embedding approximation. Following the practical approximation validated for HolUE, we evaluate the embedding-conditioned posterior at the mean embedding $\pmb { \mu } _ { \mathbf { x } }$

$$
P _ { i } ( { \bf x } ) \approx \frac { \frac { 1 - \beta } { K } p ( { \pmb \mu } _ { \bf x } | c = i ) } { m ( { \pmb \mu } _ { \bf x } ) } , \qquad P _ { 0 } ( { \bf x } ) \approx \frac { \beta / S _ { d - 1 } } { m ( { \pmb \mu } _ { \bf x } ) } .
$$

This avoids Monte Carlo noise and costs $O ( K d )$ per probe; the concentration $\kappa _ { \mathbf { x } }$ still enters the method through the non-specificity term of Section 4.2. Monte Carlo posterior estimates with up to 32 samples did not improve PRR in preliminary experiments while multiplying the cost.

Table 1. Disagreement between the raw (uncalibrated) KL summary and the error-risk ordering. Spearman: rank correlation between the KL summary and the per-probe error indicator. Inv. rate: fraction of (erroneous, correct) probe pairs ordered incorrectly by the KL summary. KL AUROC/AUPRC: any-error detection quality of the raw KL summary used directly as an uncertainty score.
<table><tr><td>Dataset</td><td></td><td></td><td>Spearman Inv. rate KL AUROC</td><td>KL AUPRC</td></tr><tr><td>IJB-C</td><td>-0.03</td><td>0.55</td><td>0.45</td><td>0.04</td></tr><tr><td>Yahoo Answers</td><td>-0.47</td><td>0.81</td><td>0.18</td><td>0.15</td></tr></table>

## 3.4. KL summaries and decision risk

HolUE scores each probe by how far the reconstructed posterior $p ( c | \mathbf { x } )$ has moved away from the prior $p ( c ) { : }$ ; we refer to this information-gain quantity, $D _ { \mathrm { K L } } ( p ( c | \mathbf { x } ) \| p ( c ) )$ , as the KL summary of the posterior. The summary splits into a gallery term $\mathrm { K L _ { 1 } }$ , accumulated over the known classes, and an unknown-mass term $\mathrm { K L _ { 2 } }$ , accumulated over the continuous unknown component. HolUE does not use these components directly: it normalizes them on a validation set and fuses them with a supervised nonlinear calibrator trained on validation error labels — an exponential-transform model in the biometric setting and a small MLP in the text setting. HolUE is therefore a validationsupervised uncertainty estimator, and throughout this paper it is compared as such.

Why is the nonlinear step needed? For a uniform prior over N outcomes,

$$
D _ { \mathrm { K L } } ( p \Vert p _ { \mathrm { u n i f } } ) = \log N - H ( p ) ,
$$

so the KL summary ranks probes by negative posterior entropy, whereas the decision risk of the maximum-posterior action is

$$
R ( \mathbf { x } ) = 1 - \operatorname* { m a x } _ { a } p ( a | \mathbf { x } ) .
$$

For $N \geq 3 ,$ entropy and maximum probability are not monotone functions of each other: a posterior split between two identities — the misidentification-prone case — is under-penalized by an entropytype score relative to a mildly difuse posterior. Two further mismatches are specific to OSR. First, KL aggregates evidence about all actions, whereas risk concerns only the action taken: a rejected probe with a sharp two-way split among gallery classes carries large $\mathrm { K L _ { 1 } }$ that is irrelevant to the rejection’s correctness. Second, $\mathrm { K L _ { 2 } }$ grows with the concentration $\kappa _ { \mathbf { x } }$ , so its relation to error probability reverses between accepted and rejected probes. $\mathrm { A }$ fixed monotone transformation of each component cannot in general account for this change of meaning across decision contexts; HolUE therefore learns a flexible nonlinear mapping on validation data.

Table 1 quantifies the raw mismatch on two representative datasets. Here Spearman is the rank correlation between the raw (uncalibrated) KL summary and the per-probe error indicator; the inversion rate is the fraction of (erroneous, correct) probe pairs to which the KL summary assigns the wrong order; KL AUROC and KL AUPRC measure any-error detection quality of the raw KL summary used directly as a score. The raw summary is nearly uncorrelated with the error ordering on IJB-C and anti-correlated on Yahoo Answers.

## 4. MPRISK METHOD

## 4.1. Decision-conditioned risk components

Let the fixed OSR decision for probe x be either rejection or acceptance as class cˆ. We call a component risk-aligned if, within the decision context in which it is active, it is monotone in the posterior probability of its associated error event. MPRisk uses three risk-aligned posterior error risks and, in the next subsection, one penalty term.

## ИНФОРМАЦИОННЫЕ ПРОЦЕССЫ ТОМ 25 № 1 2025

Acceptance risks. If x is accepted as ${ \hat { c } } ,$ the probe may be unknown or may belong to a diferent known class:

$$
r _ { \mathrm { F A } } ( \mathbf { x } ) = \mathbb { I } \{ \mathbf { x } { \mathrm { ~ a c c e p t e d } } \} P _ { 0 } ( \mathbf { x } ) , \qquad r _ { \mathrm { I D } } ( \mathbf { x } ) = \mathbb { I } \{ \mathbf { x } { \mathrm { ~ a c c e p t e d } } \} \sum _ { j = 1 } ^ { K } P _ { j } ( \mathbf { x } ) .
$$

For rejected probes both are identically zero.

Rejection risk. If x is rejected, an error occurs when the probe belongs to some gallery class:

$$
r _ { \mathrm { F R } } ( { \bf x } ) = \mathbb { I } \{ { \bf x } \ \mathrm { r e j e c t e d } \} \left( 1 - P _ { 0 } ( { \bf x } ) \right) .
$$

Under equal error costs, $r _ { \mathrm { F A } } + r _ { \mathrm { I D } } + r _ { \mathrm { F R } }$ equals the posterior probability that the taken decision is wrong — Chow’s conditional risk for the OSR action space.

## 4.2. Mixed-prior reject non-specificity

The rejection risk $1 - P _ { 0 } ( \mathbf { x } )$ is blind to an important failure mode. For a known probe whose embedding drifted far from the gallery due to corruption, the model assigns $P _ { 0 } ( \mathbf { x } ) \approx 1$ and the rejection appears maximally confident, although it is a false rejection. If the unknown class were a single collapsed atom, no posterior quantity could distinguish this case from a genuine unknown.

The mixed prior enables a sharper question: is the unknown explanation specific or difuse? Conditioned on the unknown event, the posterior density over unknown identity centers $\mathbf { u } \in \mathbb { S } ^ { d - }$ 1 is

$$
p \big ( \mathbf { u } \big | \mathbf { x } , c \in ( K , K + 1 ] \big ) = \frac { \frac { \beta } { S _ { d - 1 } } \frac { p ( \mathbf { u } | \mathbf { x } ) } { m ( \mathbf { u } ) } } { P _ { 0 } ( \mathbf { x } ) } .
$$

A high-quality unknown probe induces a concentrated posterior over latent unknown identity centers, whereas a corrupted known probe induces a difuse one: its large $P _ { 0 }$ then reflects poor embedding quality rather than evidence for a coherent unknown identity. We quantify concentration by the normalized collision probability

$$
\mathcal { C } _ { 0 } ( \mathbf { x } ) = S _ { d - 1 } \int _ { \mathbb { S } ^ { d - 1 } } p \big ( \mathbf { u } | \mathbf { x } , c \in \left( K , K + 1 \right] \big ) ^ { 2 } d \mathbf { u } , \qquad \mathcal { N } _ { 0 } ( \mathbf { x } ) = \frac { 1 } { \mathcal { C } _ { 0 } ( \mathbf { x } ) } \in ( 0 , 1 ] ,
$$

so that $\mathcal { N } _ { 0 }  1$ for a fully non-specific unknown posterior and $\mathcal { N } _ { 0 }  0$ for a concentrated one. The non-specificity penalty is

$$
r _ { \mathrm { N S } } ( { \bf x } ) = \mathbb { I } \{ { \bf x } \ \mathrm { r e j e c t e d } \} \ P _ { 0 } ( { \bf x } ) \mathcal { N } _ { 0 } ( { \bf x } ) :
$$

a rejection is flagged when it places large mass on the unknown component without committing to any specific unknown identity. Unlike the three error risks, $r _ { \mathrm { N S } }$ is not itself a posterior error probability; we refer to the four terms collectively as components.

Closed form. Approximating the unknown-identity posterior by the vMF embedding distribution predicted by SCF, the collision integral of a vMF density is analytic:

$$
\int _ { \mathbb { S } ^ { d - 1 } } p ( \mathbf { z } | \mathbf { x } ) ^ { 2 } d \mathbf { z } = { \frac { C _ { d } ( \kappa _ { \mathbf { x } } ) ^ { 2 } } { C _ { d } ( 2 \kappa _ { \mathbf { x } } ) } } , \qquad \Longrightarrow \qquad N _ { 0 } ( \mathbf { x } ) \approx { \frac { C _ { d } ( 2 \kappa _ { \mathbf { x } } ) } { S _ { d - 1 } C _ { d } ( \kappa _ { \mathbf { x } } ) ^ { 2 } } } .
$$

Low-quality samples have small $\kappa _ { \mathbf { x } }$ and hence large $\mathcal { N } _ { 0 }$ . Note that $\mathcal { N } _ { 0 }$ alone is a monotone transform of the SCF quality score; the value added by r<sub>NS</sub> lies in its decision gating and weighting by $P _ { 0 }$ which the ablations of Section 5.8 isolate.

4.3. Final score, weight tuning, and probability calibration

The MPRisk score is a nonnegative combination of the four components:

$$
u _ { \lambda } ( \mathbf { x } ) = \lambda _ { \mathrm { F A } } r _ { \mathrm { F A } } ( \mathbf { x } ) + \lambda _ { \mathrm { I D } } r _ { \mathrm { I D } } ( \mathbf { x } ) + \lambda _ { \mathrm { F R } } r _ { \mathrm { F R } } ( \mathbf { x } ) + \lambda _ { \mathrm { N S } } r _ { \mathrm { N S } } ( \mathbf { x } ) .
$$

The equal-cost version, $\lambda \equiv 1$ , is reported as MPRisk raw. The tuned version selects

$$
\lambda ^ { \star } = \operatorname { a r g m a x } _ { \lambda \in \mathbb { R } _ { + } ^ { 4 } } \operatorname { P R R } _ { \mathrm { v a l } } ^ { F _ { 1 } } ( u _ { \lambda } )
$$

on a disjoint validation set by a randomized search over log-uniform weight candidates (the score is scale-invariant in λ). The weights can be interpreted as relative costs assigned to the components; diferent datasets and FPIR operating points assign diferent efective importance to the error types (Section 5.9), so cost selection is part of the method. The four-parameter score has substantially lower capacity than the supervised baselines considered in Section 5.6. Note that weight tuning uses validation error labels through the PRR objective; what MPRisk avoids is a nonlinear fitted model in the ranking pipeline.

An optional scalar monotone calibrator (fitted on validation error indicators) maps $u _ { \lambda ^ { \star } }$ to a calibrated error probability; this variant, MPRisk cal, preserves the ranking up to ties and is used for reliability analysis only.

## 4.4. Computational cost

All components are functions of quantities already computed for the recognition decision plus the analytic ${ \mathcal { N } } _ { 0 } ;$ the per-probe overhead is $O ( K d )$ for the posterior and $O ( 1 )$ for the risk formula. MPRisk raw requires no tuning at all; weight tuning and probability calibration are ofline, one-of procedures on the validation set. Measured runtimes are reported in Section 5.12.

## 5. EXPERIMENTS

## 5.1. Datasets, protocols, and baselines

We evaluate on four image and audio identification benchmarks — IJB-C [33], IJB-B [34], Whale [4], and VoxBlink (VB-Eval-L-5) [3] — and five text benchmarks — Yahoo Answers, AG-News, DBPedia [6], CLINC150 [5], and PAN-20-AV [22]. Backbones, galleries, thresholds, and OSR protocols follow our previous work exactly: ArcFace/SCF backbones for image and audio, the Whale OSR protocol built on HappyWhale, and frozen-BERT SCF heads with the gallery constructions of our previous work for text. Recognition decisions are fixed throughout; methods are compared purely as uncertainty estimators. Validation splits are disjoint from test in identities/authors/topics and are used for all weight tuning, supervised fitting, and calibration; every validation-consuming method receives the same validation data.

Baselines include SCF (sample quality), AccScr (distance to the rejection threshold), MSP and Margin (post-hoc confidence over gallery-augmented logits with calibrated temperature), GalUE (gallery ambiguity), and HolUE in its original validation-calibrated form (Section 3.4) [13]. The prior mass is $\beta = 0 . 5$ everywhere; robustness of the shared posterior to $\beta$ was established previously and carries over to MPRisk.

## 5.2. Metrics

Our primary metric is the Prediction Rejection Ratio (PRR) [32], computed from the $F _ { 1 }$ score as probes are removed in decreasing order of uncertainty:

$$
\mathrm { P R R } = \frac { \mathrm { A U C } _ { \mathrm { u n c } } - \mathrm { A U C } _ { \mathrm { r a n d o m } } } { \mathrm { A U C } _ { \mathrm { o r a c l e } } - \mathrm { A U C } _ { \mathrm { r a n d o m } } } ,
$$

$$
\mathrm { \Delta H H \Phi O P M A I I / H O H H b I E ~ I I P O I I E C C b I ~ T O M ~ 2 5 ~ \mathrm { \sc ~ \mathcal { N } ^ { o } ~ 1 ~ } ~ 2 0 2 5 ~ \mathrm { \sc ~ \mathcal { N } ^ { o } ~ 2 ~ } }
$$

Table 2. Prediction Rejection Ratios (PRR, ) for F<sub>1</sub> filtering on image, Whale, and audio open-set recognition benchmarks. HolUE uses its validation-trained calibration; MPRisk uses validation-tuned component weights; both receive identical validation data. Best results in bold, second-best underlined.
<table><tr><td>Method</td><td colspan="3">IJB-C</td><td colspan="3">IJB-B</td><td colspan="3">Whale</td><td colspan="3">VB-Eval-L-5</td></tr><tr><td></td><td>0.05</td><td>0.1</td><td>0.2</td><td>0.05</td><td>0.1</td><td>0.2</td><td>0.05</td><td>0.1</td><td>0.2</td><td>0.01</td><td>0.05</td><td>0.1</td></tr><tr><td>SCF</td><td>0.40</td><td>0.31</td><td>0.23</td><td>0.29</td><td>0.25</td><td>0.22</td><td>0.16</td><td>0.02</td><td>-0.06</td><td>0.55</td><td>0.26</td><td>0.12</td></tr><tr><td>AccScr</td><td>0.73</td><td>0.72</td><td>0.66</td><td>0.65</td><td>0.68</td><td>0.62</td><td>0.77</td><td>0.75</td><td>0.66</td><td>0.66</td><td>0.76</td><td>0.72</td></tr><tr><td>MSP</td><td>0.74</td><td>0.75</td><td>0.70</td><td>0.66</td><td>0.70</td><td>0.65</td><td>0.77</td><td>0.77</td><td>0.70</td><td>0.38</td><td>0.88</td><td>0.86</td></tr><tr><td>Margin</td><td>0.74</td><td>0.75</td><td>0.70</td><td>0.66</td><td>0.70</td><td>0.66</td><td>0.77</td><td>0.77</td><td>0.70</td><td>0.68</td><td>0.88</td><td>0.86</td></tr><tr><td>GalUE</td><td>0.74</td><td>0.74</td><td>0.67</td><td>0.66</td><td>0.69</td><td>0.60</td><td>0.78</td><td>0.76</td><td>0.70</td><td>0.69</td><td>0.89</td><td>0.87</td></tr><tr><td>HolUE</td><td>0.76</td><td>0.81</td><td>0.73</td><td>0.54</td><td>0.71</td><td>0.63</td><td>0.79</td><td>0.82</td><td>0.83</td><td>0.74</td><td>0.81</td><td>0.89</td></tr><tr><td>MPRisk raw (ours)</td><td>0.60</td><td>0.44</td><td>0.31</td><td>0.54</td><td>0.38</td><td>0.29</td><td>0.78</td><td>0.76</td><td>0.70</td><td>0.65</td><td>0.88</td><td>0.87</td></tr><tr><td>MPRisk (ours)</td><td></td><td>0.76 0.83</td><td>0.91</td><td>0.67</td><td>0.76</td><td>0.87</td><td>0.81</td><td>0.88</td><td>0.94</td><td>0.75</td><td>0.89</td><td>0.94</td></tr></table>

so that 1 corresponds to oracle filtering, 0 to random filtering, and negative values to a score worse than random. We complement PRR with per-error-type detection AUROC (any error, FA, FR, ID), paired bootstrap confidence intervals for PRR diferences (resampling over probes), expected calibration error (ECE) with reliability diagrams for the calibrated variants, and wall-clock perprobe runtime.

## 5.3. Compared estimators under a fixed validation budget

Both HolUE and tuned MPRisk consume the same validation set, so the head-to-head comparison below is tuned-versus-tuned. To isolate what the feature representation contributes beyond the shared validation access, we additionally evaluate a family of estimators built on the same posterior and validation data:

– Linear KL fusion: the KL evidence restricted to a linear rule. The features $- \mathrm { K L _ { 1 } , - K L _ { 2 } } .$ , and $- ( \mathrm { K L _ { 1 } + K L _ { 2 } } )$ are sign-expanded (each feature also enters negated, so arbitrary signed combinations are reachable), standardized on validation, and combined with weights selected by the same randomized $\mathrm { P R R } _ { \mathrm { v a l } } ^ { F _ { 1 } }$ search used for λ<sup>⋆</sup>. Comparing this row with HolUE isolates the contribution of the nonlinear calibrator; comparing it with tuned MPRisk isolates the contribution of the risk features under identical linear tuning.

– Tuned simple: the same linear protocol applied to the post-hoc scores {SCF, AccScr, MSP, Margin}.

– Rejected-SCF: the SCF quality score gated by the rejection decision (accepted probes receive a constant low value), isolating the gating idea from the $P _ { 0 }$ weighting in $r _ { \mathrm { N S } }$

– Supervised logistic / Supervised MLP: error detectors trained on validation with binary any-error labels over a 14-dimensional feature vector combining the four post-hoc scores with all posterior features $\mathrm { ( - K L _ { 1 } , - K L _ { 2 } }$ , their sum, $P _ { 0 } , \mathcal { N } _ { 0 }$ , the four MPRisk components, and the raw MPRisk sum). The logistic model uses balanced class weights; the MLP is a single-hiddenlayer network (16 units, weight decay, early stopping). These are stronger supervised baselines with a larger feature set; note they difer from HolUE’s calibrator, which operates on only two normalized features with a task-specific objective.

– Hybrid KL+MPRisk: a tuned linear combination of the Linear-KL-fusion score and the tuned MPRisk score, testing whether information gain and decision risk are complementary.

## 5.4. Main results: image, Whale, and audio

Table 2 reports PRR on IJB-C, IJB-B, Whale, and VoxBlink at three FPIR operating points each; Figure 2 shows the corresponding rejection curves.

![](images/bc608fc56f7d781333ebf4780dd7e78a9d5a338604ff98ef74d74e0913f0a5e4.jpg)

![](images/7f856adcfa529a4b3e0fea96f1235b5e88d32ee3a2beaa3da11c4d4a880df208.jpg)

![](images/eaeaff3ec31ec588f7b28534d91d7903ad11a4aa421eeba0b9990e8241d2722a.jpg)  
Figure 2. Risk-controlled filtering curves on IJB-C at FPIR 0.05 (left), Whale at FPIR 0.1 (middle), and VoxBlink at FPIR 0.05 (right). Each panel shows F , FPIR, and FNIR versus the fraction of filtered probes; probes are removed in decreasing order of uncertainty; PRR values are given in the legends, with Random and Oracle curves as references. MPRisk tracks the Oracle most closely on $F _ { 1 } ;$ unlike sample-quality scores (SCF) it reduces FPIR rapidly, and unlike gallery-only scores (AccScr, GalUE) it also reduces FNIR by filtering confident non-specific rejections early. Best viewed zoomed in.

Table 3. Prediction Rejection Ratios (PRR, ) for $F _ { 1 }$ filtering on open-set text classification benchmarks. HolUE uses its validation-trained MLP calibration; MPRisk uses validation-tuned component weights. Best results in bold, second-best underlined.
<table><tr><td rowspan="2">Method</td><td colspan="6">Yahoo</td><td colspan="6">AGNews</td><td colspan="6">DBPedia</td><td colspan="6">CLINC150</td><td colspan="4">PAN-20-AV</td></tr><tr><td>0.1</td><td>0.2</td><td>0.3</td><td>0.4</td><td>0.5</td><td>0.1</td><td>0.2</td><td></td><td>0.3</td><td>0.4</td><td>0.5</td><td>0.1</td><td></td><td>0.2</td><td>0.3</td><td>0.4</td><td>0.5</td><td>0.1</td><td>0.2</td><td>0.3</td><td></td><td>0.4</td><td>0.5</td><td>0.1</td><td>0.2</td><td>0.3</td><td>0.4</td><td>0.5</td></tr><tr><td>SCF</td><td>0.17</td><td>0.28</td><td>0.40</td><td>0.49</td><td>0.55</td><td></td><td>-0.07</td><td>-0.07</td><td>-0.14</td><td></td><td>-0.14-0.11</td><td>0.18</td><td>0.30</td><td></td><td>0.41</td><td>0.48</td><td>0.56</td><td>0.50</td><td>0.49</td><td>0.47</td><td>0.46</td><td></td><td>0.48</td><td>0.02</td><td>0.03</td><td>0.05</td><td>0.20</td><td>0.23</td></tr><tr><td>AccScr</td><td>-0.14</td><td>0.28</td><td>0.43</td><td>0.48</td><td>0.49</td><td></td><td>-0.02</td><td>0.39</td><td>0.48</td><td>0.51</td><td>0.50</td><td>0.39</td><td></td><td>0.78</td><td>0.83</td><td>0.84</td><td>0.79</td><td>0.15</td><td>0.40</td><td>0.49</td><td>0.56</td><td></td><td>0.59</td><td>-0.20</td><td>0.31</td><td>0.44</td><td>0.37</td><td>0.43</td></tr><tr><td>MSP</td><td>0.24</td><td>0.35</td><td>0.41</td><td>0.43</td><td>0.45</td><td></td><td>0.06</td><td>0.43</td><td>0.48</td><td>0.51</td><td>0.19</td><td></td><td>0.39</td><td>0.78</td><td>0.83</td><td>0.84</td><td>0.79</td><td>0.68</td><td>0.62</td><td>0.59</td><td></td><td>0.53</td><td>0.55</td><td>0.17</td><td>0.39</td><td>0.50</td><td>0.50</td><td>0.59</td></tr><tr><td>Margin</td><td>-0.14</td><td>0.28</td><td>0.43</td><td>0.48</td><td>0.49</td><td></td><td>-0.02</td><td>0.39</td><td>0.48</td><td>0.51</td><td>0.50</td><td></td><td>0.39</td><td>0.78</td><td>0.83</td><td>0.84</td><td>0.79</td><td>0.14</td><td>0.36</td><td>0.40</td><td>0.39</td><td></td><td>0.38</td><td>-0.23</td><td>0.31</td><td>0.46</td><td>0.42</td><td>0.51</td></tr><tr><td>GalUE</td><td>-0.15</td><td>0.27</td><td>0.43</td><td>0.48</td><td>0.49</td><td></td><td>-0.03</td><td>0.38</td><td>0.47</td><td>0.51</td><td>0.51</td><td></td><td>0.77</td><td>0.78</td><td>0.83</td><td>0.84</td><td>0.79</td><td>0.14</td><td>0.35</td><td>0.39</td><td>0.38</td><td></td><td>0.37</td><td>-0.18</td><td>0.37</td><td>0.46</td><td>0.45</td><td>0.53</td></tr><tr><td>HolUE</td><td>0.77</td><td>0.71</td><td>0.69</td><td>0.71</td><td>0.74</td><td></td><td>0.47</td><td>0.45</td><td>0.52</td><td>0.62</td><td>0.71</td><td></td><td>0.52</td><td>0.72</td><td>0.81</td><td>0.94</td><td>0.96</td><td>0.69</td><td>0.62</td><td>0.57</td><td>0.56</td><td></td><td>0.59</td><td>0.18</td><td>0.35</td><td>0.40</td><td>0.49</td><td>0.43</td></tr><tr><td>MPRisk raw (ours)</td><td>-0.15</td><td>0.27</td><td>0.43</td><td>0.48</td><td>0.49</td><td></td><td>0.02</td><td>0.38</td><td>0.47</td><td>0.51</td><td>0.51</td><td>0.65</td><td></td><td>0.78</td><td>0.83</td><td>0.83</td><td>0.79</td><td>0.14</td><td>0.36</td><td>0.40</td><td>0.40</td><td></td><td>0.41</td><td>-0.14</td><td>0.43</td><td>0.51</td><td>0.53</td><td>0.60</td></tr><tr><td>MPRisk (ours)</td><td>0.66</td><td>0.52</td><td>0.46</td><td>0.40</td><td>0.64</td><td></td><td>0.54</td><td>0.39</td><td>0.48</td><td>0.58</td><td>0.76</td><td>0.72</td><td></td><td>0.85</td><td>0.92</td><td>0.94</td><td>0.95</td><td></td><td>0.71 0.65</td><td>0.59</td><td>90.57</td><td></td><td>0.59</td><td>0.37</td><td>0.43</td><td>0.48</td><td>0.44 0.62</td><td></td></tr></table>

MPRisk is best or tied-best at all 12 reported operating points. The gains grow with FPIR: at FPIR 0.2 MPRisk reaches 0.91 on IJB-C (+0.18 over HolUE), 0.87 on IJB-B, and 0.94 on Whale (+0.11 over HolUE) — higher FPIR admits more unknown probes, so the acceptance-conditioned risks carry more of the error mass. MPRisk raw is substantially weaker than the tuned version (e.g., 0.44 vs. 0.83 on IJB-C at FPIR 0.1): the equal-cost sum is dominated by the numerous small rejection risks, so cost selection is integral to the method.

## 5.5. Main results: text

Table 3 reports PRR on the five text benchmarks at five operating points each; Figure 3 shows rejection curves.

MPRisk performs best on DBPedia (e.g., 0.92 vs. 0.81 for HolUE at FPIR 0.3), CLINC150 (best at all five points), and PAN-20-AV at most points (e.g., 0.37 vs. 0.18 at FPIR 0.1), and takes the extreme operating points on AGNews. HolUE is consistently stronger on Yahoo Answers. The error-type analysis in Section 5.9 explains the exception: Yahoo errors are almost exclusively false rejections, a regime in which the quality-driven $\mathrm { K L _ { 2 } }$ feature under HolUE’s calibration is a near-ideal specialist. This dataset-level complementarity motivates the hybrid estimator studied in Section 5.6.

## 5.6. Feature-space study

Tables 4 and 5 evaluate the estimator family of Section 5.3: all rows share the validation split, the tuning objective $\mathrm { P R R } _ { \mathrm { v a l } } ^ { F _ { 1 } }$ , and (for the linear rows) the same randomized search budget.

Four conclusions follow.

![](images/f0ddaf2424dc2aa052e05518615579571aa323555ad742a6554dc150bbfa8aff.jpg)

![](images/daa877877cb2af3d44b8ec04f6b4151a7fe424fb91d6df71f08531cba91314b4.jpg)

![](images/e3a97b58bbf035cd380629b81d8624a2e5668161fa1df115184b13325a0ccc6b.jpg)  
Figure 3. Risk-controlled filtering curves on the text benchmarks at FPIR 0.1: Yahoo Answers, CLINC150, and PAN-20-AV (left to right). Each panel shows F<sub>1</sub>, FPIR, and FNIR versus filter-out rate, with PRR in the legends and Random/Oracle references. On CLINC150 and PAN-20-AV, MPRisk dominates the $F _ { 1 }$ curve and lowers FPIR fastest; on Yahoo Answers, HolUE’s curve is higher because the error mass consists almost entirely of false rejections (cf. Table 11). Best viewed zoomed in.

Table 4. Feature-space study on image, Whale, and audio benchmarks (PRR, ). All rows use the same validation split and, where applicable, the same tuning objective and search budget. Linear KL fusion applies MPRisk’s linear tuning protocol to the KL components (with sign expansion); it should be contrasted with HolUE in Table 2, which fuses the same KL components with a supervised nonlinear calibrator. MPRisk rows correspond to an independent re-run of the λ search and may difer from Table 2 by 0.01.
<table><tr><td rowspan="2">Method</td><td colspan="3">IJB-C</td><td colspan="3">IJB-B</td><td colspan="3">Whale</td><td colspan="3">VB-Eval-L-5</td></tr><tr><td>0.05</td><td>0.1</td><td>0.2</td><td>0.05</td><td>0.1</td><td>0.2</td><td>0.05</td><td>0.1</td><td>0.2</td><td>0.01</td><td>0.05</td><td>0.1</td></tr><tr><td>MPRisk raw</td><td>0.6</td><td>0.44</td><td>0.31</td><td>0.54</td><td>0.38</td><td>0.29</td><td>0.78</td><td>0.76</td><td>0.7</td><td>0.65</td><td>0.88</td><td>0.87</td></tr><tr><td>MPRisk tuned no rNS</td><td>0.76</td><td>0.84</td><td>0.9</td><td>0.66</td><td>0.77</td><td>0.87</td><td>0.82</td><td>0.89</td><td>0.94</td><td>0.71</td><td>0.82</td><td>0.95</td></tr><tr><td>MPRisk</td><td>0.76</td><td>0.84</td><td>0.91</td><td>0.66</td><td>0.77</td><td>0.88</td><td>0.81</td><td>0.89</td><td>0.94</td><td>0.75</td><td>0.89</td><td>0.95</td></tr><tr><td>Linear KL fusion</td><td>-0.67</td><td>-0.5</td><td>0.17</td><td>-1.31</td><td>-0.18</td><td>0.49</td><td>0.61</td><td>0.6</td><td>0.61</td><td>0.58</td><td>0.31</td><td>0.43</td></tr><tr><td>Tuned simple</td><td>0.7</td><td>0.82</td><td>0.84</td><td>-2.89</td><td>-2.34</td><td>-1.88</td><td>0.8</td><td>0.81</td><td>0.76</td><td>0.77</td><td>0.89</td><td>0.87</td></tr><tr><td>Rejected-SCF</td><td>0.28</td><td>0.15</td><td>0.08</td><td>0.29</td><td>0.16</td><td>0.08</td><td>0.27</td><td>0.11</td><td>0.04</td><td>0.68</td><td>0.07</td><td>0</td></tr><tr><td>Supervised logistic</td><td>0.7</td><td>0.83</td><td>0.91</td><td>0.66</td><td>0.78</td><td>0.87</td><td>0.86</td><td>0.91</td><td>0.95</td><td>0.88</td><td>0.89</td><td>0.91</td></tr><tr><td>Supervised MLP</td><td>-0.7</td><td>0.23</td><td>0.67</td><td>0.56</td><td>0.67</td><td>0.82</td><td>-0.38</td><td>0.46</td><td>0.88</td><td>0.34</td><td>0.01</td><td>-0.03</td></tr><tr><td>Hybrid KL+MPRisk</td><td>0.79</td><td>0.87</td><td>0.93</td><td>0.68</td><td>0.8</td><td>0.89</td><td>0.84</td><td>0.9</td><td>0.95</td><td>0.83</td><td>0.89</td><td>0.93</td></tr></table>

(i) Linear fusion is efective for MPRisk features but not for the KL features. Linear KL fusion produces negative PRR on IJB-C (−0.67/−0.50) and IJB-B (−1.31) and trails elsewhere, whereas the same KL evidence fused by HolUE’s nonlinear calibrator reaches 0.76/0.81 on IJB-C (Table 2). This is the analysis of Section 3.4 made quantitative. MPRisk reaches 0.76/0.84/0.91 on IJB-C with four nonnegative weights: the risk decomposition yields features that combine efectively under a linear score.

(ii) Linear fusion of heterogeneous post-hoc scores generalizes poorly. Tuned simple is competitive on IJB-C and VoxBlink but substantially negative on IJB-B (−2.89) and weak on DBPedia; scores on incompatible scales transfer poorly from validation to test even after standardization, whereas the MPRisk components share a common posterior-probability scale.

(iii) MPRisk is competitive with supervised detectors at much lower capacity. Supervised logistic — trained on error labels over the full 14-dimensional feature set — exceeds tuned MPRisk only modestly (e.g., Whale 0.86 vs. 0.81 at FPIR 0.05; DBPedia 0.89 vs. 0.72 at FPIR 0.1) and loses at some points (VoxBlink at FPIR 0.1). The supervised MLP varies substantially across datasets, from 0.88 on Yahoo to −0.70 on IJB-C and −0.38 on Whale; this does not contradict the stability of HolUE’s two-input, task-objective MLP — on moderate validation sets, performance tracks how tightly the hypothesis class matches the task rather than raw capacity.

Table 5. Feature-space study on open-set text benchmarks (PRR, ). Conventions as in Table $4 ;$ the HolUE reference (KL features with supervised MLP calibration) is given in Table 3.
<table><tr><td rowspan="2">Method</td><td colspan="3">Yahoo Answers</td><td colspan="3">AGNews</td><td colspan="3">DBPedia</td><td colspan="3">CLINC150</td><td colspan="3">PAN-20-AV</td></tr><tr><td>0.1</td><td>0.3</td><td>0.5</td><td>0.1</td><td>0.3</td><td>0.5</td><td>0.1</td><td>0.3</td><td>0.5</td><td>0.1</td><td>0.3</td><td>0.5</td><td>0.1</td><td>0.3</td><td>0.5</td></tr><tr><td>MPRisk raw</td><td>-0.15</td><td>0.43</td><td>0.49</td><td>0.02</td><td>0.47</td><td>0.51</td><td>0.65</td><td>0.83</td><td>0.79</td><td>0.14</td><td>0.4</td><td>0.41</td><td>-0.14</td><td>0.51</td><td>0.6</td></tr><tr><td>MPRisk tuned no  $r _ { N S }$ </td><td>0.66</td><td>0.51</td><td>0.64</td><td>0.54</td><td>0.48</td><td>0.76</td><td>0.72</td><td>0.92</td><td>0.95</td><td>0.69</td><td>0.6</td><td>0.59</td><td>0.33</td><td>0.49</td><td>0.64</td></tr><tr><td>MPRisk</td><td>0.66</td><td>0.46</td><td>0.64</td><td>0.53</td><td>0.48</td><td>0.76</td><td>0.72</td><td>0.92</td><td>0.95</td><td>0.71</td><td>0.59</td><td>0.59</td><td>0.31</td><td>0.47</td><td>0.61</td></tr><tr><td>Linear KL fusion</td><td>0.71</td><td>0.36</td><td>0.51</td><td>0.5</td><td>0.43</td><td>0.41</td><td>0.44</td><td>0.42</td><td>0.53</td><td>0.64</td><td>0.55</td><td>0.55</td><td>0.26</td><td>0.42</td><td>0.5</td></tr><tr><td>Tuned simple</td><td>0.33</td><td>0.5</td><td>0.72</td><td>0.07</td><td>0.16</td><td>0.27</td><td>0.17</td><td>0.22</td><td>0.43</td><td>0.69</td><td>0.6</td><td>0.63</td><td>0.31</td><td>0.46</td><td>0.47</td></tr><tr><td>Rejected-SCF</td><td>0.18</td><td>0.18</td><td>-0.07</td><td>0.22</td><td>0.01</td><td>-0.22</td><td>0.05</td><td>-0.08</td><td>-0.21</td><td>0.59</td><td>0.27</td><td>0</td><td>0.25</td><td>0.22</td><td>0.28</td></tr><tr><td>Supervised logistic</td><td>0.73</td><td>0.68</td><td>0.75</td><td>0.32</td><td>0.62</td><td>0.75</td><td>0.89</td><td>0.93</td><td>0.97</td><td>0.76</td><td>0.62</td><td>0.61</td><td>0.33</td><td>0.45</td><td>0.61</td></tr><tr><td>Supervised MLP</td><td>0.88</td><td>0.86</td><td>0.86</td><td>0.53</td><td>-0.37</td><td>0.72</td><td>0.89</td><td>0.8</td><td>0.95</td><td>0.76</td><td>0.58</td><td>-0.18</td><td>0.19</td><td>-0.33</td><td>-0.22</td></tr><tr><td>Hybrid KL+MPRisk</td><td>0.73</td><td>0.49</td><td>0.63</td><td>0.54</td><td>0.57</td><td>0.79</td><td>0.71</td><td>0.92</td><td>0.95</td><td>0.74</td><td>0.59</td><td>0.59</td><td>0.43</td><td>0.61</td><td>0.67</td></tr></table>

Table 6. Full MPRisk component ablation on image, Whale, and audio benchmarks. PRR for $F _ { 1 }$ filtering is reported.
<table><tr><td>Variant</td><td></td><td>IJB-C</td><td></td><td></td><td>IJB-B</td><td></td><td></td><td>Whale</td><td></td><td></td><td>VB-Eval-L-5</td><td></td></tr><tr><td></td><td>0.05</td><td>0.1</td><td>0.2</td><td>0.05</td><td>0.1</td><td>0.2</td><td>0.05</td><td>0.1</td><td>0.2</td><td>0.01</td><td>0.05</td><td>0.1</td></tr><tr><td>rFA</td><td>-0.92</td><td>0.34</td><td>0.76</td><td>-1.05</td><td>0.22</td><td>0.72</td><td>-3.17</td><td>-0.63</td><td>0.84</td><td>-3.54</td><td>0.53</td><td>0.85</td></tr><tr><td>rID</td><td>-0.91</td><td>0.35</td><td>0.77</td><td>-1.06</td><td>0.22</td><td>0.73</td><td>-3.21</td><td>-0.66</td><td>0.8</td><td>-0.65</td><td>0.56</td><td>0.87</td></tr><tr><td>rFR</td><td>0.22</td><td>0.11</td><td>0.05</td><td>0.25</td><td>0.13</td><td>0.06</td><td>0.26</td><td>0.1</td><td>0.03</td><td>0.8</td><td>0.08</td><td>0.01</td></tr><tr><td>rNS</td><td>0.28</td><td>0.15</td><td>0.06</td><td>0.29</td><td>0.16</td><td>0.08</td><td>0.27</td><td>0.11</td><td>0.04</td><td>0.67</td><td>0.07</td><td>0</td></tr><tr><td> $r _ { F A } + r _ { I D }$ </td><td>-0.91</td><td>0.35</td><td>0.77</td><td>-1.05</td><td>0.22</td><td>0.73</td><td>-3.18</td><td>-0.63</td><td>0.83</td><td>-3.52</td><td>0.59</td><td>0.88</td></tr><tr><td> $r _ { F R } + r _ { N S }$ </td><td>0.22</td><td>0.11</td><td>0.05</td><td>0.25</td><td>0.13</td><td>0.06</td><td>0.26</td><td>0.1</td><td>0.03</td><td>0.81</td><td>0.08</td><td>0.01</td></tr><tr><td>Ordinary no  $r _ { N S }$ </td><td>0.6</td><td>0.44</td><td>0.31</td><td>0.54</td><td>0.38</td><td>0.29</td><td>0.78</td><td>0.76</td><td>0.7</td><td>0.65</td><td>0.88</td><td>0.87</td></tr><tr><td>Full raw</td><td>0.6</td><td>0.44</td><td>0.31 0.9</td><td>0.54 0.66</td><td>0.38 0.77</td><td>0.29 0.87</td><td>0.78 0.82</td><td>0.76</td><td>0.7</td><td>0.65</td><td>0.88</td><td>0.87</td></tr><tr><td>Tuned no  $r _ { N S }$ </td><td>0.76</td><td>0.84 0.84</td><td>0.91</td><td>0.66</td><td>0.77</td><td>0.88</td><td>0.81</td><td>0.89 0.89</td><td>0.94</td><td>0.71</td><td>0.82</td><td>0.95</td></tr><tr><td>MPRisk</td><td>0.76</td><td></td><td></td><td></td><td>0.16</td><td>0.08</td><td>0.27</td><td></td><td>0.94</td><td>0.75</td><td>0.89</td><td>0.95</td></tr><tr><td>Rejected-SCF</td><td>0.28</td><td>0.15</td><td>0.08</td><td>0.29</td><td></td><td></td><td></td><td>0.11</td><td>0.04</td><td>0.68</td><td>0.07</td><td>0</td></tr></table>

(iv) Decision risk and information gain are complementary. Hybrid KL+MPRisk, a tuned combination of two scalars, obtains the highest PRR at 9 of 12 image/audio operating points and on AGNews and PAN-20-AV (e.g., 0.43 at PAN FPIR 0.1, +0.25 over HolUE); on Yahoo it recovers most of HolUE’s advantage (0.73 vs. 0.77).

## 5.7. Component ablation

Tables 6 and 7 decompose MPRisk into its parts.

Three facts emerge. First, single components are not scores: each decision-conditioned component is identically zero on the complementary decision set, so in isolation it ranks half of the probes arbitrarily. This is why r<sub>FA</sub> alone reaches PRR −3.54 at low FPIR (where almost no unknown probes are accepted) yet 0.85 at high FPIR. Second, which component dominates is dataset- and operating-point-specific: acceptance risks dominate DBPedia and high-FPIR points, while rejection components dominate Yahoo, CLINC150, and PAN at low FPIR (r alone achieves 0.66 on Yahoo). No fixed weighting serves all regimes. Third, the tuned combination is uniformly near the per-cell maximum over all variants, while the equal-cost combinations inherit the weaknesses of their dominant untuned term. The addition of $r _ { \mathrm { N S } }$ on top of the tuned three-component score gives small consistent gains at specific operating points (+0.01 on IJB-C/IJB-B at FPIR 0.2; $+ 0 . 0 4 / + 0 . 0 7$ on VoxBlink; +0.03 on DBPedia at FPIR 0.1; +0.28 on Yahoo at FPIR 0.3); its main value, shown in Sections 5.8–5.9, is rejection-specific detection and interpretability.

Table 7. Full MPRisk component ablation on open-set text benchmarks. PRR for $F _ { 1 }$ filtering is reported.
<table><tr><td rowspan="2">Variant</td><td colspan="3">Yahoo Answers</td><td colspan="3">AGNews</td><td colspan="3">DBPedia</td><td colspan="3">CLINC150</td><td colspan="3">PAN-20-AV</td></tr><tr><td>0.1</td><td>0.3</td><td>0.5</td><td>0.1</td><td>0.3</td><td>0.5</td><td>0.1</td><td>0.3</td><td>0.5</td><td>0.1</td><td>0.3</td><td>0.5</td><td>0.1</td><td>0.3</td><td>0.5</td></tr><tr><td>rFA</td><td>-1.44</td><td>0.09</td><td>0.5</td><td>-1.51</td><td>0.38</td><td>0.66</td><td>0.55</td><td>0.92</td><td>0.95</td><td>-0.49</td><td>0.15</td><td>0.42</td><td>-1.44</td><td>-0.58</td><td>0.2</td></tr><tr><td>rID</td><td>-0.43</td><td>-0.21</td><td>0.64</td><td>-0.53</td><td>-0.29</td><td>0.76</td><td>-0.04</td><td>-0.05</td><td>-0.12</td><td>-0.45</td><td>0.01</td><td>0.13</td><td>-1.41</td><td>-0.45</td><td>0.58</td></tr><tr><td>rFR</td><td>0.66</td><td>0.31</td><td>-0.04</td><td>0.53</td><td>0.05</td><td>-0.19</td><td>-0.47</td><td>-0.27</td><td>-0.21</td><td>0.69</td><td>0.29</td><td>0.01</td><td>0.35</td><td>0.31</td><td>0.28</td></tr><tr><td>rNS</td><td>0.17</td><td>0.18</td><td>-0.07</td><td>0.22</td><td>0.01</td><td>-0.22</td><td>0.06</td><td>-0.08</td><td>-0.21</td><td>0.55</td><td>0.26</td><td>0</td><td>0.25</td><td>0.22</td><td>0.28</td></tr><tr><td> $r _ { F A } + r _ { I D }$ </td><td>-1.44</td><td>0.09</td><td>0.5</td><td>-1.51</td><td>0.38</td><td>0.66</td><td>0.55</td><td>0.92</td><td>0.95</td><td>-0.4</td><td>0.2</td><td>0.33</td><td>-1.39</td><td>-0.38</td><td>0.65</td></tr><tr><td> $r _ { F R } + r _ { N S }$ </td><td>0.66</td><td>0.31</td><td>-0.04</td><td>0.57</td><td>0.05</td><td>-0.19</td><td>0.15</td><td>-0.08</td><td>-0.21</td><td>0.71</td><td>0.29</td><td>0.01</td><td>0.35</td><td>0.31</td><td>0.28</td></tr><tr><td>Ordinary no  $r _ { N S }$ </td><td>-0.15</td><td>0.43</td><td>0.49</td><td>0.02</td><td>0.47</td><td>0.51</td><td>0.65</td><td>0.83</td><td>0.79</td><td>0.14</td><td>0.4</td><td>0.41</td><td>-0.14</td><td>0.51</td><td>0.6</td></tr><tr><td>Full raw</td><td>-0.15</td><td>0.43</td><td>0.49</td><td>0.02</td><td>0.47</td><td>0.51</td><td>0.65</td><td>0.83</td><td>0.79</td><td>0.14</td><td>0.4</td><td>0.41</td><td>-0.14</td><td>0.51</td><td>0.6</td></tr><tr><td>Tuned no rNS</td><td>0.66</td><td>0.18</td><td>0.64</td><td>0.54</td><td>0.48</td><td>0.76</td><td>0.69</td><td>0.92</td><td>0.95</td><td>0.71</td><td>0.6</td><td>0.59</td><td>0.33</td><td>0.49</td><td>0.64</td></tr><tr><td>MPRisk</td><td>0.66</td><td>0.46</td><td>0.64</td><td>0.53</td><td>0.48</td><td>0.76</td><td>0.72</td><td>0.92</td><td>0.95</td><td>0.71</td><td>0.59</td><td>0.59</td><td>0.31</td><td>0.47</td><td>0.61</td></tr><tr><td>Rejected-SCF</td><td>0.18</td><td>0.18</td><td>-0.07</td><td>0.22</td><td>0.01</td><td>-0.22</td><td>0.05</td><td>-0.08</td><td>-0.21</td><td>0.59</td><td>0.27</td><td>0</td><td>0.25</td><td>0.22</td><td>0.28</td></tr></table>

Table 8. Mixed-prior necessity ablation on image and Whale benchmarks. PRR for F<sub>1</sub> filtering is reported. The collapsed-unknown variant replaces the continuous unknown component by a single reject class, making r unavailable; consequently the first three rows coincide by construction.
<table><tr><td>Variant</td><td>IJB-C 0.05 0.1</td><td>0.2</td><td>0.05</td><td>Whale 0.1 0.2</td></tr><tr><td>Collapsed unknown</td><td>0.6 0.44</td><td>0.31</td><td>0.78</td><td>0.76 0.7</td></tr><tr><td>No rNs</td><td>0.6 0.44</td><td>0.31</td><td>0.78</td><td>0.76 0.7</td></tr><tr><td>Equal full</td><td>0.6 0.44</td><td>0.31</td><td>0.78</td><td>0.76 0.7</td></tr><tr><td>rNS</td><td>0.28 0.15</td><td>0.06</td><td>0.27</td><td>0.11 0.04</td></tr><tr><td> $\mathcal { N } _ { 0 }$ </td><td>0.4 0.31</td><td>0.23</td><td>0.16</td><td>0.02 -0.06</td></tr><tr><td>MPRisk</td><td>0.76 0.83 0.91 0.81 0.88 0.94</td><td></td><td></td><td></td></tr></table>

## 5.8. Is the mixed prior necessary?

Tables 8 and 9 isolate the contribution of modeling the unknown class as a continuum.

Under a collapsed unknown class the three ordinary risks are unchanged, so the first three rows coincide: the only score the mixed prior adds is r<sub>NS</sub>. Two nested comparisons quantify its ingredients. The raw non-specificity $\mathcal { N } _ { 0 }$ alone equals the SCF quality score up to a monotone transform (compare its IJB-C row, $0 . 4 0 / 0 . 3 1 / 0 . 2 3$ , with SCF in Table 2) and is weak. Gating and $P _ { 0 }$ weighting change its behavior qualitatively — as a standalone score it becomes a rejection specialist nearly identical to Rejected-SCF — and its value materializes inside the tuned combination. Most of the improvement over the equal-weight score comes from tuning the three standard decision risks (Tables 6–7); the non-specificity penalty provides smaller aggregate gains but adds a rejectionspecific signal and makes suspicious rejects interpretable.

## 5.9. Error-type detection

PRR aggregates all error types into one ranking; Tables 10 and 11 disaggregate them. MPRisk cal is reported alongside MPRisk: monotone calibration preserves AUROC up to ties, so the rows are near-identical.

On the image and audio benchmarks, tuned MPRisk is the best overall error detector and the best FA and ID detector (IJB-C: FA 0.98, ID 0.97; Whale: FA 0.96, ID 0.94), at a moderate cost in FR AUROC relative to threshold-based scores. On Yahoo Answers, where errors are almost entirely false rejections, optimization for aggregate PRR pushes MPRisk’s weights toward $r _ { \mathrm { F R } } / r _ { \mathrm { N S } } .$ its FR AUROC reaches 0.90, close to HolUE’s 0.92, while its FA AUROC drops to 0.20 because the objective assigns little weight to the rare FA errors. On PAN-20-AV, MPRisk is the best any-error and FR detector (0.68 and 0.66) while conceding FA and ID to gallery scores; MPRisk raw is the best ID detector there (0.91). These trade-ofs stem from compressing four costs into one scalar for one metric, not from the decomposition itself: a deployment with asymmetric error costs can tune λ against its own cost matrix, which single-summary scores do not support.

Table 9. Mixed-prior necessity ablation on text benchmarks. PRR for $F _ { 1 }$ filtering is reported.
<table><tr><td>Variant</td><td colspan="3">Yahoo Answers</td><td colspan="3">DBPedia</td><td colspan="3">CLINC150</td><td colspan="3">PAN-20-AV</td></tr><tr><td></td><td>0.1</td><td>0.3</td><td>0.5</td><td>0.1</td><td>0.3</td><td>0.5</td><td>0.1</td><td>0.3</td><td>0.5</td><td>0.1</td><td>0.3</td><td>0.5</td></tr><tr><td>Collapsed unknown -0.15</td><td></td><td>0.43</td><td>0.49</td><td>0.65</td><td>0.83</td><td>0.79</td><td>0.14</td><td>0.4</td><td>0.41</td><td>-0.14</td><td>0.51</td><td>0.6</td></tr><tr><td>No rNs</td><td>-0.15</td><td>0.43</td><td>0.49</td><td>0.65</td><td>0.83</td><td>0.79</td><td>0.14</td><td>0.4</td><td>0.41</td><td>-0.14</td><td>0.51</td><td>0.6</td></tr><tr><td>Equal full</td><td>-0.15</td><td>0.43</td><td>0.49</td><td>0.65</td><td>0.83</td><td>0.79</td><td>0.14</td><td>0.4</td><td>0.41</td><td>-0.14</td><td>0.51</td><td>0.6</td></tr><tr><td>rNS</td><td>0.17</td><td>0.18</td><td>-0.07</td><td>0.06</td><td>-0.08</td><td>-0.21</td><td>0.55</td><td>0.26</td><td>0</td><td>0.25</td><td>0.22</td><td>0.28</td></tr><tr><td> $\mathcal { N } _ { 0 }$ </td><td>0.16</td><td>0.39</td><td>0.55</td><td>0.01</td><td>0.14</td><td>0.31</td><td>-0.03</td><td>-0.16</td><td>-0.26</td><td>0.02</td><td>0.05</td><td>0.22</td></tr><tr><td>MPRisk</td><td>0.66</td><td>0.46</td><td>0.64</td><td>0.72</td><td>0.92</td><td>0.95</td><td>0.71</td><td>0.59</td><td>0.59</td><td>0.37</td><td>0.48</td><td>0.62</td></tr></table>

Table 10. Error-type detection quality on IJB-C at FPIR 0.05 and Whale at FPIR 0.1. AUROC is reported for detecting any OSR error (Any), false acceptance (FA), false rejection (FR), and misidentification (ID).
<table><tr><td colspan="5"></td><td colspan="4">Whale (FPIR 0.1)</td></tr><tr><td>Method</td><td>Any</td><td>FA</td><td>FR</td><td>ID</td><td>Any</td><td>FA</td><td>FR</td><td>ID</td></tr><tr><td>SCF</td><td>0.70</td><td>0.61</td><td>0.77</td><td>0.91</td><td>0.61</td><td>0.55</td><td>0.84</td><td>0.86</td></tr><tr><td>AccScr</td><td>0.87</td><td>0.91</td><td>0.80</td><td>0.85</td><td>0.88</td><td>0.87</td><td>0.86</td><td>0.81</td></tr><tr><td>MSP</td><td>0.87</td><td>0.91</td><td>0.80</td><td>0.88</td><td>0.88</td><td>0.88</td><td>0.85</td><td>0.88</td></tr><tr><td>GalUE</td><td>0.88</td><td>0.92</td><td>0.80</td><td>0.90</td><td>0.89</td><td>0.88</td><td>0.86</td><td>0.90</td></tr><tr><td>HolUE</td><td>0.87</td><td>0.96</td><td>0.70</td><td>0.97</td><td>0.93</td><td>0.93</td><td>0.85</td><td>0.90</td></tr><tr><td>MPRisk raw</td><td>0.81</td><td>0.80</td><td>0.79</td><td>0.84</td><td>0.89</td><td>0.88</td><td>0.86</td><td>0.90</td></tr><tr><td>MPRisk</td><td>0.89</td><td>0.98</td><td>0.75</td><td>0.97</td><td>0.95</td><td>0.96</td><td>0.79</td><td>0.94</td></tr><tr><td>MPRisk cal</td><td>0.89</td><td>0.98</td><td>0.75</td><td>0.97</td><td>0.95</td><td>0.96</td><td>0.79</td><td>0.94</td></tr></table>

## 5.10. Statistical significance

Tables 12 and 13 report paired bootstrap confidence intervals for PRR diferences.

On the image and audio benchmarks, the 95% intervals for the gain over HolUE exclude zero on IJB-B $( \Delta = 0 . 1 2 , [ 0 . 0 7 , 0 . 1 7 ] )$ , Whale (0.06, [0.04, 0.08]), and VoxBlink (0.08, [0.06, 0.10]); IJB-C at FPIR 0.05 is a statistical tie. On text, the intervals exclude zero on DBPedia (0.19, [0.15, 0.25]) and CLINC150 $( p = 0 . 0 4 6 )$ , include zero on AGNews $( p = 0 . 0 7 3 )$ and PAN-20-AV $( p = 0 . 0 5 4 ;$ the wide interval reflects the small PAN test set), and MPRisk is significantly worse on Yahoo Answers (−0.11). The gain of tuned MPRisk over MPRisk raw exceeds zero in every tested comparison, and monotone calibration leaves all conclusions unchanged (MPRisk cal rows).

## 5.11. Calibration

Table 14 evaluates whether the scores double as error-probability estimates; Figure 4 shows reliability diagrams. Both compared quantities are validation-calibrated — HolUE by its calibration model, MPRisk cal by the monotone calibrator — so this comparison is also symmetric.

Calibration results are mixed: MPRisk cal has lower ECE on IJB-C (0.020 vs. 0.049), Yahoo Answers, and PAN-20-AV, while HolUE is lower on Whale, VoxBlink, DBPedia, and CLINC150 by

Table 11. Error-type detection quality on Yahoo Answers and PAN-20-AV, both at FPIR 0.1. AUROC is reported for detecting any OSR error (Any), false acceptance (FA), false rejection (FR), and misidentification (ID); misidentification is absent in the Yahoo Answers protocol at this operating point.
<table><tr><td rowspan="2">Method</td><td colspan="4">Yahoo Answers ( (FPIR 0.1)</td><td colspan="4">PAN-20-AV (FPIR 0.1)</td></tr><tr><td>Any</td><td>FA</td><td>FR</td><td>ID</td><td>Any</td><td>FA</td><td>FR</td><td>ID</td></tr><tr><td>SCF</td><td>0.45</td><td>0.66</td><td>0.41</td><td></td><td>0.50</td><td>0.37</td><td>0.56</td><td>0.36</td></tr><tr><td>AccScr</td><td>0.59</td><td>0.77</td><td>0.54</td><td></td><td>0.61</td><td>0.89</td><td>0.48</td><td>0.83</td></tr><tr><td>MSP</td><td>0.54</td><td>0.76</td><td>0.48</td><td></td><td>0.64</td><td>0.82</td><td>0.53</td><td>0.86</td></tr><tr><td>GalUE</td><td>0.58</td><td>0.77</td><td>0.53</td><td></td><td>0.61</td><td>0.88</td><td>0.48</td><td>0.90</td></tr><tr><td>HolUE</td><td>0.89</td><td>0.58</td><td>0.92</td><td></td><td>0.59</td><td>0.75</td><td>0.52</td><td>0.70</td></tr><tr><td>MPRisk raw</td><td>0.59</td><td>0.77</td><td>0.53</td><td></td><td>0.61</td><td>0.88</td><td>0.48</td><td>0.91</td></tr><tr><td>MPRisk</td><td>0.79</td><td>0.20</td><td>0.90</td><td></td><td>0.68</td><td>0.62</td><td>0.66</td><td>0.66</td></tr><tr><td>MPRisk cal</td><td>0.79</td><td>0.27</td><td>0.88</td><td></td><td>0.68</td><td>0.62</td><td>0.66</td><td>0.66</td></tr></table>

Table 12. Paired bootstrap confidence intervals for PRR diferences on image, Whale, and audio benchmarks. A reported $p ( \Delta \leq 0 )$ of 0 indicates that no bootstrap resample produced $\Delta \leq 0 .$
<table><tr><td>Dataset</td><td>FPIR</td><td>Comparison</td><td>PRR A PRR B</td><td></td><td>∆ PRR [95% CI]</td><td> $p ( \Delta \leq 0 )$ </td></tr><tr><td>IJB-C</td><td>0.05</td><td>MPRisk - HolUE</td><td>0.76</td><td>0.76</td><td>0.01 [−0.01, 0.03]</td><td>0.166</td></tr><tr><td>IJB-B</td><td>0.05</td><td>MPRisk - HolUE</td><td>0.67</td><td>0.54</td><td>0.12 [0.07, 0.17]</td><td>0</td></tr><tr><td>Whale</td><td>0.1</td><td>MPRisk - HolUE</td><td>0.88</td><td>0.82</td><td>0.06 [0.04, 0.08]</td><td>0</td></tr><tr><td>VB-Eval-L-5</td><td>0.05</td><td>MPRisk – HolUE</td><td>0.89</td><td>0.81</td><td>0.08 [0.06, 0.1]</td><td>0</td></tr><tr><td>IJB-C</td><td>0.05</td><td>MPRisk cal - HolUE</td><td>0.76</td><td>0.76</td><td>0.01 [−0.01, 0.03]</td><td>0.178</td></tr><tr><td>IJB-B</td><td>0.05</td><td>MPRisk cal – HolUE</td><td>0.67</td><td>0.54</td><td>0.12 [0.07,0.17]</td><td>0</td></tr><tr><td>Whale</td><td>0.1</td><td>MPRisk cal – HolUE</td><td>0.88</td><td>0.82</td><td>0.06 [0.04, 0.08]</td><td>0</td></tr><tr><td>VB-Eval-L-5</td><td>0.05</td><td>MPRisk cal – HolUE</td><td>0.89</td><td>0.81</td><td>0.08 [0.06, 0.1]</td><td>0</td></tr><tr><td>IJB-C</td><td>0.05</td><td>MPRisk – MPRisk raw</td><td>0.76</td><td>0.6</td><td>0.16 [0.13, 0.19]</td><td>0</td></tr><tr><td>IJB-C</td><td>0.05</td><td>MPRisk − MPRisk no rNS</td><td>0.76</td><td>0.6</td><td>0.16 [0.13, 0.19]</td><td>0</td></tr><tr><td>Whale</td><td>0.1</td><td>MPRisk – MPRisk raw</td><td>0.88</td><td>0.76</td><td>0.12 [0.1, 0.14]</td><td>0</td></tr><tr><td>Whale</td><td>0.1</td><td>MPRisk – MPRisk no rNs</td><td>0.88</td><td>0.76</td><td>0.12 [0.1, 0.14]</td><td>0</td></tr></table>

smaller margins. Since calibration does not change the selective-recognition ranking (Tables 12– 13), the calibrated variant can be used whenever an absolute error-probability readout is needed at no ranking cost.

## 5.12. Runtime

Table 15 reports online per-probe cost, excluding the shared backbone forward pass.

MPRisk raw costs 0.05–1.6 ms per probe: its inputs are the posterior probabilities already computed for the decision plus one analytic formula. Tuned MPRisk and MPRisk cal are comparable to or cheaper than HolUE on all benchmarks, and over four times cheaper on PAN-20-AV (31–32 vs. 132 ms), where the dynamic per-query gallery makes the $\mathrm { K L _ { 2 } }$ computation of HolUE expensive.

## 6. DISCUSSION

What the experiments establish. HolUE and MPRisk use the same posterior and validation split; they difer in the features and the final scoring model. The KL components contain useful information, but their ordering is not generally consistent with decision risk, so their usefulness depends on the learned nonlinear calibration: the same features under linear tuning produce negative PRR on several benchmarks. The decision-conditioned components, in contrast, combine efectively under a four-weight linear rule, generalize stably from validation to test, and outperform the calibrated KL score at 25 of 27 image/audio operating points and on most text operating points. Overall: gains whose bootstrap intervals exclude zero on five of nine benchmarks, ties or inconclusive diferences on three, and one clear HolUE win (Yahoo Answers).

Table 13. Paired bootstrap confidence intervals for PRR diferences on text benchmarks. A reported p(∆ 0) of 0 indicates that no bootstrap resample produced $\Delta \leq 0 .$
<table><tr><td>Dataset</td><td>FPIR</td><td>Comparison</td><td>PRR A</td><td>PRR B</td><td>∆ PRR [95% CI]</td><td></td><td> $p ( \Delta \leq 0 )$ </td></tr><tr><td>Yahoo Answers</td><td>0.1</td><td>MPRisk – HolUE</td><td>0.66</td><td>0.77</td><td></td><td>-0.11 [−0.13, -0.09]</td><td>1</td></tr><tr><td>AGNews</td><td>0.1</td><td>MPRisk – HolUE</td><td>0.54</td><td>0.47</td><td></td><td>0.07 [–0.02, 0.17]</td><td>0.073</td></tr><tr><td>DBPedia</td><td>0.1</td><td>MPRisk – HolUE</td><td>0.72</td><td>0.52</td><td></td><td>0.19 [0.15, 0.25]</td><td>0</td></tr><tr><td>CLINC150</td><td>0.1</td><td>MPRisk – HolUE</td><td>0.71</td><td>0.69</td><td></td><td>0.02 [0, 0.04]</td><td>0.046</td></tr><tr><td>PAN-20-AV</td><td>0.1</td><td>MPRisk – HolUE</td><td>0.37</td><td>0.18</td><td></td><td>0.18 [-0.05, 0.4]</td><td>0.054</td></tr><tr><td>Yahoo Answers</td><td>0.1</td><td>MPRisk cal - HolUE</td><td>0.66</td><td>0.77</td><td></td><td>-0.11 [−0.13, -0.09]</td><td>1</td></tr><tr><td>AGNews</td><td>0.1</td><td>MPRisk cal - HolUE</td><td>0.54</td><td>0.47</td><td></td><td>0.07 [-0.02, 0.16]</td><td>0.063</td></tr><tr><td>DBPedia</td><td>0.1</td><td>MPRisk cal – HolUE</td><td>0.72</td><td>0.52</td><td></td><td>0.19 [0.14, 0.25]</td><td>0</td></tr><tr><td>CLINC150</td><td>0.1</td><td>MPRisk cal – HolUE</td><td>0.69</td><td>0.69</td><td></td><td>-0.01 [-0.03, 0.02]</td><td>0.558</td></tr><tr><td>PAN-20-AV</td><td>0.1</td><td>MPRisk cal – HolUE</td><td>0.37</td><td>0.18</td><td></td><td>0.18 [-0.04, 0.39]</td><td>0.062</td></tr><tr><td>Yahoo Answers</td><td>0.1</td><td>MPRisk – MPRisk raw</td><td>0.66</td><td>-0.15</td><td></td><td>0.8 [0.77, 0.86]</td><td>0</td></tr><tr><td>Yahoo Answers</td><td>0.1</td><td>MPRisk – MPRisk no  $r _ { N S }$ </td><td>0.66</td><td>-0.15</td><td></td><td>0.8 [0.77, 0.87]</td><td>0</td></tr><tr><td>PAN-20-AV</td><td>0.1</td><td>MPRisk – MPRisk raw</td><td>0.37</td><td>-0.14</td><td></td><td>0.51 [0.2, 0.78]</td><td>0</td></tr><tr><td>PAN-20-AV</td><td>0.1</td><td>MPRisk – MPRisk no  $r _ { N S }$ </td><td>0.37</td><td>-0.14</td><td></td><td>0.51 [0.21, 0.8]</td><td>0.001</td></tr></table>

Table 14. Calibration quality (ECE, ) on image, Whale, audio, and text benchmarks. The FPIR operating point is indicated below each dataset name.
<table><tr><td>Method</td><td>IJB-C 0.05</td><td>0.1</td><td>0.05</td><td>Whale VB-Eval-L-5 Yahoo Answers 0.1</td><td>0.1</td><td>0.1</td><td>DBPedia CLINC150 PAN-20-AV 0.1</td></tr><tr><td>HolUE</td><td>0.049</td><td>0.125</td><td>0.111</td><td>0.314</td><td>0.407</td><td>0.240</td><td>0.207</td></tr><tr><td>MPRisk cal</td><td>0.020</td><td>0.137</td><td>0.165</td><td>0.228</td><td>0.416</td><td>0.254</td><td>0.192</td></tr></table>

When KL features remain useful. The Yahoo result is diagnostic. Its error mass is almost purely false rejections of ambiguous multi-topic questions, for which the quality-driven $\mathrm { K L _ { 2 } }$ feature under HolUE’s calibration is a near-ideal single signal. MPRisk’s scalar objective gains FR sensitivity (FR AUROC 0.90) at the expense of FA sensitivity (FA AUROC 0.20). Two remedies follow from our results: tune λ against the deployment’s own cost matrix, or use the hybrid, which recovers most of HolUE’s Yahoo performance while preserving MPRisk’s gains elsewhere.

The role of the mixed prior. The continuous unknown component contributes on three levels: small consistent PRR gains at specific operating points; a rejection-specific detection signal unavailable to a collapsed reject class; and an explanation of why a rejection is suspicious — high unknown mass with no specific unknown hypothesis — which single-scalar scores cannot articulate.

Capacity and validation size. Logistic regression on the full 14-dimensional feature set exceeds tuned MPRisk only modestly and not uniformly; the generic 14-input MLP varies substantially across datasets, whereas HolUE’s two-input, task-objective MLP is stable. On moderate validation sets, performance tracks how tightly the hypothesis class matches the task rather than raw capacity; MPRisk’s four-weight rule over semantically fixed components is the lowest-capacity option in this family and achieves comparable performance. When abundant validation errors are available, the hybrid or a logistic detector on the joint features is a reasonable recipe; when validation data are limited, tuned MPRisk is a lower-risk choice and MPRisk raw a tuning-free fallback.

![](images/527dafc4211f2a6bcb429d1647c3d52519e4f21f2e501206b1959f593f46c0df.jpg)

![](images/2031a87a08b385826eb8aeadb16bca0f51c86dcaa519434c1cb911f4e642077e.jpg)

![](images/267dd087e0ca00bb38fe7043570d2507eeeef6b4f5fd27efd73eca06a9242b10.jpg)

![](images/d86fe2b5599c10cf3c6d06ed752238dfbe8a5d3faeac3b3b3ebec07cc752815b.jpg)  
Figure 4. Reliability diagrams for calibrated error probabilities of HolUE and MPRisk cal on Whale at FPIR 0.1 (top) and Yahoo Answers at FPIR 0.1 (bottom). Predicted error probability (binned) is plotted against empirical error frequency per bin; the diagonal denotes perfect calibration.

Table 15. Runtime overhead on image, Whale, audio, and text benchmarks. Online time in milliseconds per probe is reported (backbone forward pass excluded); the FPIR operating point is indicated below each dataset name.
<table><tr><td>Method</td><td>IJB-C 0.05</td><td>0.1</td><td>0.05</td><td>Whale VB-Eval-L-5 Yahoo Answers DBPedia CLINC150 PAN-20-AV 0.1</td><td>0.1</td><td>0.1</td><td>0.1</td></tr><tr><td>HolUE</td><td>2.08</td><td>3.03</td><td>2.95</td><td>3.68</td><td>2.03</td><td>4.82</td><td>132.43</td></tr><tr><td>MPRisk raw</td><td>0.67</td><td>1.35</td><td>1.55</td><td>0.09</td><td>0.05</td><td>0.14</td><td>0.45</td></tr><tr><td>MPRisk</td><td>1.89</td><td>3.13</td><td>2.33</td><td>2.56</td><td>1.41</td><td>1.84</td><td>31.17</td></tr><tr><td>MPRisk cal</td><td>2.07</td><td>3.08</td><td>2.38</td><td>2.55</td><td>1.54</td><td>2.04</td><td>32.14</td></tr></table>

Limitations. Tuned MPRisk depends on a validation split matched to the test operating point, and the scalar PRR objective induces error-type trade-ofs (Section 5.9); multi-objective or costmatrix tuning is a direct extension. The method inherits the assumptions of the Bayesian posterior and the quality of the SCF embedding distribution: the mean-embedding approximation bounds posterior fidelity in extremely low-κ regimes, and the interpretation of non-specificity as evidence of false rejection has been validated only within the tested models. Robustness under gallery shift and validation/test cost mismatch was not evaluated. Finally, calibration results are mixed, and both methods over-predict error probability in the tails on text.

## 7. CONCLUSION

We proposed MPRisk, a mixed-prior posterior decision-risk score for uncertainty estimation in open-set recognition. Instead of summarizing the gallery-aware Bayesian posterior by information gain and calibrating the result with a supervised nonlinear model, MPRisk directly scores the error events associated with the selected decision — false acceptance, misidentification, and false rejection — and adds a closed-form non-specificity penalty that flags confident but poorly supported rejections, enabled by modeling unknown identities as a continuous component. Four nonnegative validation-tuned weights sufice for ranking.

Across nine image, audio, and text benchmarks under matched validation budgets, MPRisk is best or tied-best at all image and audio operating points and on most text operating points, with bootstrap-confirmed gains over HolUE on five benchmarks, at comparable or lower runtime. Linear fusion of the KL features performs poorly, whereas the MPRisk components remain competitive with supervised error detectors; combining the two scores gives further gains in several settings. These results suggest that decision-conditioned posterior risks provide a simple basis for selective OSR, while KL features supply complementary information when suficient validation data are avail able. Future work includes cost-matrix tuning of the weights and transferring decision-conditioned risk scoring to selective generation and hallucination detection in large language models [35].

## REFERENCES

1. W. J. Scheirer, A. de Rezende Rocha, A. Sapkota, and T. E. Boult. Toward open set recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 35(7):1757–1772, 2013.

2. Anil K. Jain Stan Z. Li. Handbook of Face Recognition. Springer London, 2011.

3. Yuke Lin, Ming Cheng, Fulin Zhang, Yingying Gao, Shilei Zhang, and Ming Li. VoxBlink2: A 100K+ speaker recognition corpus and the open-set speaker-identification benchmark. In Proc. Interspeech 2024, pages 4263–4267, 2024.

4. Philip T. Patton, Ted Cheeseman, Kenshin Abe, Taiki Yamaguchi, Walter Reade, Ken Southerland, Addison Howard, Erin M. Oleson, Jason B. Allen, Erin Ashe, et al. A deep learning approach to photoidentification demonstrates high performance on two dozen cetacean species. Methods in Ecology and Evolution, 14(10):2611–2625, 2023.

5. Stefan Larson, Anish Mahendran, Joseph J. Peper, Christopher Clarke, Andrew Lee, Parker Hill, Jonathan K. Kummerfeld, Kevin Leach, Michael A. Laurenzano, Lingjia Tang, and Jason Mars. An evaluation dataset for intent classification and out-of-scope prediction. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan, editors, Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 1311–1316, Hong Kong, China, November 2019. Association for Computational Linguistics.

6. Junfan Chen, Richong Zhang, Junchi Chen, Chunming Hu, and Yongyi Mao. Open-set semi-supervised text classification with latent outlier softening. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD ’23, page 226236, New York, NY, USA, 2023. Association for Computing Machinery.

7. Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4690–4699, 2019.

8. Hao Wang, Yitong Wang, Zheng Zhou, Xing Ji, Dihong Gong, Jingchao Zhou, Zhifeng Li, and Wei Liu. Cosface: Large margin cosine loss for deep face recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 5265–5274, 2018.

9. C. Chow. On optimum recognition error and reject tradeof. IEEE Transactions on Information Theory, 16(1):41–46, 1970.

10. Yonatan Geifman and Ran El-Yaniv. Selective classification for deep neural networks. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017.

11. Yichun Shi and Anil K. Jain. Probabilistic face embeddings. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2019.

12. Shen Li, Jianqing Xu, Xiaqing Xu, Pengcheng Shen, Shaoxin Li, and Bryan Hooi. Spherical confidence learning for face recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15629–15637, June 2021.

13. Leonid Erlygin and Alexey Zaytsev. Holistic uncertainty estimation for open-set recognition. IEEE Access, 14:18868–18880, 2026.

14. Jitendra Parmar, Satyendra Chouhan, Vaskar Raychoudhury, and Santosh Rathore. Open-world machine learning: Applications, challenges, and opportunities. ACM Comput. Surv., 55(10), February 2023.

15. Chuanxing Geng, Sheng jun Huang, and Songcan Chen. Recent advances in open set recognition: A survey. 2018.

16. Abhijit Bendale and Terrance E Boult. Towards open set deep networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1563–1572, 2016.

17. Lei Shu, Hu Xu, and Bing Liu. DOC: Deep open classification of text documents. In Martha Palmer, Rebecca Hwa, and Sebastian Riedel, editors, Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pages 2911–2916, Copenhagen, Denmark, September 2017. Association for Computational Linguistics.

18. Noam Slonim and Naftali Tishby. Document clustering using word clusters via the information bottleneck method. In Proceedings of the 23rd Annual International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’00, page 208215, New York, NY, USA, 2000. Association for Computing Machinery.

19. Efstathios Stamatatos. Authorship attribution using text distortion. In Mirella Lapata, Phil Blunsom, and Alexander Koller, editors, Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 1, Long Papers, pages 1138–1149, Valencia, Spain, April 2017. Association for Computational Linguistics.

20. Patrick Juola. Authorship attribution. Found. Trends Inf. Retr., 1(3):233334, December 2006.

21. Sarkhan Badirli, Mary Borgo Ton, Abdulmecit Gungor, and Murat Dundar. Open set authorship attribution toward demystifying victorian periodicals. In Document Analysis and Recognition ICDAR 2021: 16th International Conference, Lausanne, Switzerland, September 510, 2021, Proceedings, Part IV, page 221235, Berlin, Heidelberg, 2021. Springer-Verlag.

22. Mike Kestemont, Enrique Manjavacas, Ilia Markov, Janek Bevendorf, Matti Wiegmann, Efstathios Stamatatos, Martin Potthast, and Benno Stein. Overview of the cross-domain authorship verification task at PAN 2020. In Linda Cappellato, Carsten Eickhof, Nicola Ferro, and Aur´elie N´ev´eol, editors, Working Notes of CLEF 2020 - Conference and Labs of the Evaluation Forum, Thessaloniki, Greece, September 22-25, 2020, volume 2696 of CEUR Workshop Proceedings. CEUR-WS.org, 2020.

23. Andrei Manolache, Florin Brad, Elena Burceanu, Antonio Barbalau, Radu Tudor Ionescu, and Marius Popescu. Transferring bert-like transformers’ knowledge for authorship verification. CoRR, abs/2112.05125, 2021.

24. Alex Kendall and Yarin Gal. What uncertainties do we need in bayesian deep learning for computer vision? In Advances in neural information processing systems, volume 30, 2017.

25. Stefan Depeweg, Jos´e Miguel Hern´andez-Lobato, Finale Doshi-Velez, and Stefen Udluft. Decomposition of uncertainty in Bayesian deep learning for eficient and risk-sensitive learning. In International Conference on Machine Learning (ICML), 2018.

26. Yarin Gal and Zoubin Ghahramani. Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In International conference on machine learning, pages 1050–1059, 2016.

27. Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. In Advances in neural information processing systems, volume 30, 2017.

28. Nicholas I. Fisher, Toby Lewis, and Brian J. J. Embleton. Statistical Analysis of Spherical Data. Cambridge University Press, Cambridge, UK, 1993.

29. Chuan Guo, Geof Pleiss, Yixuan Sun, and Kilian Q Weinberger. On calibration of modern neural networks. In International conference on machine learning, pages 1321–1330, 2017.

30. Mahdi Pakdaman Naeini, Gregory F. Cooper, and Milos Hauskrecht. Obtaining well calibrated probabilities using bayesian binning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 29, 2015.

31. Allan H. Murphy. A new vector partition of the probability score. Journal of Applied Meteorology, 12(4):595–600, 1973.

32. E. Fadeeva, R. Vashurin, A. Tsvigun, and et al. Lm-polygraph: Uncertainty estimation for language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, 2023.

33. Brianna Maze, Jocelyn Adams, James A. Duncan, Nathan Kalka, Tim Miller, Charles Otto, Anil K. Jain, William T. Niggel, Janet Anderson, Jordan Cheney, and Patrick Grother. Iarpa janus benchmark c: Face dataset and protocol. In 2018 International Conference on Biometrics (ICB), pages 158–165, 2018.

34. Cameron Whitelam, Emma Taborsky, Austin Blanton, Brianna Maze, Jocelyn Adams, Tim Miller, Nathan Kalka, Anil K. Jain, James A. Duncan, Kristen Allen, Jordan Cheney, and Patrick Grother. Iarpa janus benchmark-b face dataset. In 2017 IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 592–600, 2017.

35. Alexandra Bazarova, Aleksandr Yugay, Andrey Shulga, Alina Ermilova, Andrei Volodichev, Konstantin Polev, Julia Belikova, Rauf Parchiev, Dmitry Simakov, Maxim Savchenko, Andrey Savchenko, Serguei Barannikov, and Alexey Zaytsev. Hallucination detection in llms with topological divergence on attention graphs, 2025.