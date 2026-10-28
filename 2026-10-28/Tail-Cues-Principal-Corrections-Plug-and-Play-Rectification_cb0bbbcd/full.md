# Tail Cues, Principal Corrections: Plug-and-Play Rectification for Open-Set Test-Time Adaptation

Yingkai Yang Chaoqi Chen<sup>∗</sup> Hui Huang

Guangdong Provincial Key Laboratory of Visual Media and Multidimensional Intelligence College of Computer Science and Software Engineering, Shenzhen University ekyleyang@gmail.com, cqchen1994@gmail.com, hhzhiyan@gmail.com

## Abstract

Open-Set Test-Time Adaptation (OSTTA) operates on non-stationary target streams where covariate and semantic shifts coexist. Existing TTA methods often obtain adaptation signals by updating model parameters or maintaining mutable memories, which can be costly and vulnerable to unknown samples. We present TALOR, a lightweight, backward-free rectification module that uses the source linear head as a geometric anchor. The head-induced basis decomposes each normalized feature into high-gain principal coordinates and low-gain tail coordinates. TALOR estimates a soft-routed affine correction in the principal subspace, combining a regime-level principal bias with a tail-conditioned slope that captures sample-level tail-to-principal coupling. Using weighted ridge regression, it subtracts this drift estimate from the principal coordinates, reconstructs the feature and feeds it into the source head. On the ImageNet-C benchmark with six csOOD datasets, TALOR outperforms the next-best UniEnt by 2.2% H-score while running 2.5× faster and using only 12% of its GPU memory; as a post-correction plug-in, it further boosts COME to 68.4% H-score with only 2% additional memory overhead. Code is available at https://github.com/ekyle0522/TALOR.

## 1 Introduction

Modern recognition systems are increasingly deployed in non-stationary environments, where test inputs may drift away from the source distribution and may also contain previously unseen semantics. Test-Time Adaptation (TTA) [6, 39, 51, 22, 3, 35, 13, 42, 24, 12] aims to improve robustness using unlabeled test data at deployment time. Open-Set TTA further requires adapting to covariate-shifted in-distribution (csID) samples while simultaneously rejecting covariate-shifted out-of-distribution (csOOD) samples [11]. This setting is challenging for common TTA strategies: gradient updates and cache-based modules rely on online adaptation signals that can be distorted by unknown samples, while their additional optimization or memory overhead can be undesirable in real-time deployment. These limitations motivate a lightweight alternative that corrects the test-time representation itself without updating model parameters.

A domain shift is expressed not only in the input space but also in the latent space. When the feature direction deviates from the source geometry, even a small displacement can damage predictions if it lies along directions that the classifier amplifies. This suggests that feature correction should not be treated as an unconstrained high-dimensional update: the linear head already identifies latent space directions to which its outputs are most sensitive. This observation raises a feature-level question:

Can the response of latent features to domain shift be approximated from the unlabeled non-stationary stream and then removed without updating the model?

![](images/9009e8b0e2cc73d53fe17de4ca8ba081d65ff6624823da11f72e08a331c3aa48.jpg)  
Figure 1: Runtime–performance–memory comparison of TALOR and representative baselines. Circle size denotes GPU usage. The dashed line marks Source+TALOR: even as a standalone plug-in, it surpasses existing methods while using the least runtime and memory. When attached to other TTA methods, TALOR substantially improves their performance with only marginal additional budgets.

Most classification models end with a linear head $h ( v ) = W v + b ,$ which is usually kept fixed during test-time adaptation to preserve the source semantic decision boundary. We observe that this fixed head is not merely a readout layer. It induces a task-aligned coordinate system over penultimate features, and we use its singular spectrum to partition the axes into high-gain and lowgain components. Concretely, the dominant right-singular vectors of W form a principal subspace: changes along these axes are magnified by large singular values and therefore can strongly affect logits, margins, and confidence. The remaining basis vectors form a tail subspace: they have lower direct gain under the classifier, but remain sensitive to target shifts. In online batches, the principal coordinates can contain a removable domain response: a shared principal mean together with sample-specific deviations that are statistically predictable from tail coordinates. We call the latter coupling tail-to-principal leakage. This leads to a latent view of adaptation: domain shift displacesfeatures relative to the geometry assumed by the fixed source head, and adaptation should restore their compatibility with this geometry.

Building on this motivation, we propose TALOR (Test-time Adaptation via Linear Online Rectification), a plug-and-play rectification module inserted between the backbone feature and the linear head. TALOR uses the head to decompose normalized features into high-gain principal coordinates and low-gain tail coordinates. It then treats the principal coordinates as correction targets and the tail coordinates as shift-sensitive cues, estimating an affine principal drift online in mixed csID/csOOD streams. The rectified feature is then evaluated by the unchanged source head. TALOR is lightweight because it avoids backward passes, augmentation loops, and long-horizon caches. It is also composable: since it only post-corrects features, it can be applied to a frozen source model or stacked on top of backbone-updating TTA methods as a stabilizing post-correction (Fig. 1).

Our contributions are summarized as follows:

• We identify a head-induced feature structure for OSTTA: domain shift induces a removable response in classifier-amplified principal coordinates, while low-gain tail coordinates provide shift-sensitive cues for estimating this response.

• We propose TALOR, a plug-and-play, backward-free rectification module that couples this classifier-head-induced sensor–target decomposition with soft-routed, tail-conditioned estimation of affine principal drift, removing the drift before the unchanged source head without source data or source statistics.

• TALOR is lightweight and broadly effective: it surpasses next-best UniEnt as a standalone method by 2.2% H-score, boosts COME by 4.7% with only 2% memory overhead, and generalizes beyond open-set adaptation to closed-set and dynamic continual protocols.

## 2 Related Work

Prior works on test-time adaptation can be broadly grouped into two lines: continual TTA (CTTA) and open-set TTA (OSTTA). CTTA methods aim to adapt online to evolving covariate shifts while avoiding error accumulation and catastrophic forgetting, typically through regularization, parameterefficient updates, or reset-based stabilization. OSTTA methods further consider semantic shifts, requiring the model to adapt to known classes while reliably rejecting unknown samples. Closest in spirit, SSA [1] aligns source and target statistics within an output-significant PCA subspace of source features; TALOR instead uses a head-induced basis, keeps the complementary tail as a drift sensor, and removes drift sample-wise without backpropagation or source statistics. Appendix A details representative methods.

![](images/656a9f40ed305d7326fa23893c06c166607150ab9bcff1703b81b3dccadb1582.jpg)  
Figure 2: Pipeline. TALOR is a gradient-free layer between feature extractor and source linear head. ⃝1 Head-induced coordinates: the normalized feature vˆ is projected into principal coordinates a and tail coordinates u using the right-singular basis of W. ⃝2 & ⃝3 Affine principal drift: TALOR establishes the dual-regime soft routing via tail-energy, then models removable principal drift. ⃝4 Rectified inference: The estimated principal drift is subtracted, then the feature is reconstructed from the unchanged tail coordinates and the original feature-norm scale before applying the classifier.

## 3 Preliminaries

Problem Setup. We study Open-Set TTA on non-stationary target streams where covariate shift and unknown semantics coexist. A model consists of a backbone $\overline { { g } } : \mathcal { X }  \mathbb { R } ^ { D }$ mapping an input x to a feature vector $v = g ( x ) \in \mathbb { R } ^ { D }$ , and a linear classifier head $h ( v ) = W v + b$ , with $W \in \mathbb { R } ^ { C \times D }$ and $b \in \mathbb { R } ^ { C }$ . In general, the head parameters $\{ W , b \}$ typically remain fixed to preserve the semantic decision boundary. However, the backbone g may essentially operate in two modes: it is either frozen or continuously updated. Our method is agnostic to this choice, as it operates directly on the streaming feature v. The source logits are $z ( x ) \overset { \vartriangle } { = } W g ( x ) + b$ . At test time $t ,$ each step provides an unlabeled mini-batch $\boldsymbol { B } _ { t } = \{ \boldsymbol { x } _ { i } \} _ { i = 1 } ^ { B }$ drawn from a mixture distribution

$$
p _ { t } ( x ) = ( 1 - \pi _ { t } ) p _ { t } ^ { \mathrm { i d } } ( x ) + \pi _ { t } p _ { t } ^ { \mathrm { o o d } } ( x ) ,\tag{1}
$$

where $p _ { t } ^ { \mathrm { i d } }$ denotes the csID component whose labels belong to the source label set ${ \mathcal { V } } _ { \mathrm { i d } } = \{ 1 , \ldots , C \}$ and $p _ { t } ^ { \mathrm { o o d } }$ denotes the csOOD component with unknown semantics relative to ${ \mathcal { N } } _ { \mathrm { i d } }$ . The stream is non-stationary because both $p _ { t } ^ { \mathrm { i d } }$ and $p _ { t } ^ { \mathrm { o o d } }$ may change over time. The goal of OSTTA is to produce reliable ID predictions for samples from $p _ { t } ^ { \mathrm { i d } }$ while rejecting samples from $p _ { t } ^ { \mathrm { o o d } }$ , using only unlabeled test inputs and without access to source training data.

## 4 Proposed Method

Overview. TALOR performs test-time adaptation by post-correcting features (Fig. 2). The module relies on a head-induced asymmetry: high-gain principal coordinates serve as correction targets, while low-gain tail coordinates provide cues for estimating the domain-induced response.

## 4.1 Head-induced sensor–target decomposition

The frozen linear head is usually treated as a readout layer in test-time adaptation, but its weight matrix also defines an anisotropic coordinate system over classifier-input features. Directions associated with large singular values are strongly amplified in the logits, so small perturbations along these axes can substantially change margins, confidence, and open-set scores. In contrast, low-gain directions have weaker direct logit influence, but they can still respond to distributional shift. This motivates a sensor–target decomposition: principal coordinates are the correction targets, while tail coordinates are used as cues for estimating how the principal coordinates should be corrected.

![](images/210027267b0589cd6e3a93e1e8644737c5dea4f1e0cb4d299795f639dfa41b59.jpg)  
(a) Logit impact.

![](images/36fdffc5bae461ca7884eab106f51a8f8ef37ead96c4ca49e0dd3c1df4b5a768.jpg)  
(b) Relative shift.

![](images/a1ec0097b25b819cdfe9ec9cf2e1f082d03895fb2bee8dbd2345f766803577bd.jpg)  
(c) Tail-to-principal CCA.  
Figure 3: Paired clean-corrupted diagnostics in the head-induced basis. The results show that principal coordinates dominate logit impact (a), while tail coordinates remain shift-responsive (b) and linearly predictive of principal drift (c). The vertical dashed line marks the τ split.

Formally, let the singular value decomposition of the frozen head weight be

$$
\boldsymbol { W } = \boldsymbol { U \Sigma V } ^ { \top } .\tag{2}
$$

where $W \in \mathbb { R } ^ { C \times D } , V \in \mathbb { R } ^ { D \times D }$ contains the right singular vectors, and the singular values are zero-padded to $\sigma _ { 1 } , \dots , \sigma _ { D }$ with $\sigma _ { 1 } \geq \cdot \cdot \cdot \geq \sigma _ { D } \geq 0 . \operatorname { L e t } s = \operatorname* { m i n } ( C , D )$ . We choose a split point k using an energy threshold $\tau \in ( 0 , 1 )$ :

$$
k _ { 0 } = \operatorname* { m i n } \left\{ m : \frac { \sum _ { j = 1 } ^ { m } \sigma _ { j } ^ { 2 } } { \sum _ { j = 1 } ^ { D } \sigma _ { j } ^ { 2 } } \geq \tau \right\} , \qquad k = \operatorname* { m i n } \{ k _ { 0 } , s - 1 \} .\tag{3}
$$

The first k right-singular vectors span the principal basis, and the remaining vectors form the tail basis:

$$
B _ { p } = V _ { : , 1 : k } ^ { \top } \in \mathbb { R } ^ { k \times D } , \qquad B _ { t } = V _ { : , k + 1 : D } ^ { \top } \in \mathbb { R } ^ { ( D - k ) \times D } .\tag{4}
$$

For later use, denote $d = D - k$ . The two bases satisfy $B _ { p } B _ { p } ^ { \top } = I _ { k } , B _ { t } B _ { t } ^ { \top } = I _ { d } .$ and $B _ { p } ^ { \top } B _ { p } +$ $B _ { t } ^ { \top } B _ { t } = I _ { D } \left[ 9 \right]$ . When $D > C$ , head-null directions are automatically included in the tail basis; App. B details how the same construction applies to both $D < C$ and $\bar { D ^ { \circ } } > C$ backbones.

For a streaming feature $v _ { i } = g ( x _ { i } )$ , TALOR first separates its norm and direction:

$$
\rho _ { i } = \| v _ { i } \| _ { 2 } , \qquad \hat { v } _ { i } = \frac { v _ { i } } { \rho _ { i } } , \qquad a _ { i } = B _ { p } \hat { v } _ { i } \in \mathbb { R } ^ { k } , \qquad u _ { i } = B _ { t } \hat { v } _ { i } \in \mathbb { R } ^ { d } .\tag{5}
$$

Here $a _ { i }$ and $u _ { i }$ are the principal and tail coordinates of the feature direction. Since $( B _ { p } , B _ { t } )$ form an orthonormal basis of the feature space, the normalized feature can be reconstructed as

$$
\begin{array} { r } { \hat { v } _ { i } = B _ { p } ^ { \top } a _ { i } + B _ { t } ^ { \top } u _ { i } . } \end{array}\tag{6}
$$

The corresponding source logits admit the decomposition

$$
z _ { i } = W v _ { i } + b = \rho _ { i } U \Sigma \left[ { a _ { i } } \right] + b .\tag{7}
$$

Eq. 7 makes the asymmetry explicit: the same amount of coordinate drift has a much larger logit effect in high-singular-value principal directions than in low-gain tail directions.

We verify this structure with paired clean-corrupted diagnostics in Fig. 3; App. C defines the paired protocol and diagnostic metrics. Let $c _ { i } = V ^ { \top } \hat { v } _ { i }$ denote the full head-induced coordinate vector and define the coordinate drift $\Delta c _ { i j } = c _ { i j } ^ { \mathrm { c o r r } } - c _ { i j } ^ { \mathrm { c l e a n } }$ . We measure the head-weighted logit impact of the normalized feature drift $\sigma _ { j } ^ { 2 } \mathbb { E } _ { i } [ ( \Delta c _ { i j } ) ^ { 2 } ]$ , the relative shift E $_ i [ ( \Delta c _ { i j } ) ^ { 2 } ] / ( \mathrm { V a r } _ { i } ( c _ { i j } ^ { \mathrm { c l e a n } } ) + \epsilon )$ , and the canonical correlation between tail drift $\Delta u$ and principal drift $\Delta a$ . The diagnostics show that principal coordinates dominate logit impact, tail coordinates remain shift-responsive, and tail drift linearly decodes a nontrivial component of principal drift. TALOR therefore corrects only the principal coordinates and keeps the tail coordinates as shift sensors rather than correction targets: in the oracle ablation of App. C.2, restoring the principal coordinates alone closes 97.0% (ViT-B/16) and 96.7% (ResNet-50) of the clean–corrupt accuracy gap, while additionally restoring the tail adds only 1.2 and 1.8 pp.

Given an estimated principal drift $\hat { \Delta } _ { i } \in \mathbb { R } ^ { k }$ , TALOR applies the principal correction

$$
\tilde { a } _ { i } = a _ { i } - \hat { \Delta } _ { i } .\tag{8}
$$

We next model $\hat { \Delta } _ { i }$ , estimate it from unlabeled mixed streams, and remove it at inference.

## 4.2 Regime-conditioned affine principal drift

Open-set target batches contain two latent regimes: covariate-shifted in-distribution samples and covariate-shifted out-of-distribution samples. We denote the regimes by $\mathcal { Q } = \{ \mathrm { I D } , \mathrm { O O D } \}$ }. These regimes need not share the same principal-coordinate mean or the same tail-to-principal coupling. Known and unknown class samples can induce different shared principal offsets, and the principal deviation predictable from tail coordinates can also vary across regimes. A single global rectifier would average these effects; TALOR therefore maintains a regime-conditioned affine drift model.

For a regime $q \in \mathcal { Q } .$ , TALOR models the removable principal drift as

$$
\Delta _ { i } ^ { ( q ) } \approx \underbrace { \mu _ { a } ^ { ( q ) } } _ { \mathrm { p r i n c i p a l ~ m e a n } } + \underbrace { R ^ { ( q ) \top } \left( u _ { i } - \mu _ { u } ^ { ( q ) } \right) } _ { \mathrm { t a i l - t o - p r i n c i p a l ~ s l o p e } } ,\tag{9}
$$

where $\mu _ { a } ^ { ( q ) } \in \mathbb { R } ^ { k }$ is the regime-level principal mean, $\mu _ { u } ^ { ( q ) } \in \mathbb { R } ^ { d }$ is the regime-level tail mean, and $R ^ { ( q ) } \in \mathbb { R } ^ { d \times k }$ is a tail-to-principal regression matrix. We use affine principal drift to denote the whole removable term in Eq. 9; we use tail-to-principal leakage specifically for the tail-conditioned slope component. Because the slope is fitted on centered coordinates, it captures only sample-specific deviations and cannot absorb a branch-level shift. The principal mean $\mu _ { a } ^ { ( q ) }$ therefore provides a branch-specific translation in the normalized principal coordinates; samples with identical routing weights receive the same translation and retain their pairwise differences (App. D.3). The slope term $R ^ { ( q ) \top } ( u _ { i } - \mu _ { u } ^ { ( q ) } )$ ) captures the sample-specific principal deviation linearly predictable from centered tail coordinates.

Why tail cues? A centered principal deviation $a _ { i } - \mu _ { a } ^ { ( q ) }$ mixes useful class/instance variation with domain-induced nuisance; without labels, paired clean samples, or source statistics, a principal-only estimator has no external evidence for which part of this deviation to remove. Tail coordinates instead offer an auxiliary low-gain view of the same shift, allowing the tail-conditioned slope to target principal variation supported by cross-subspace evidence. Since singular values establish only the impact asymmetry, this requires an explicit condition on the centered coordinates $a _ { i } ^ { \circ } = a _ { i } - \mu _ { a } ^ { \left( q \right) }$ and $u _ { i } ^ { \circ } = u _ { i } - \mu _ { u } ^ { ( q ) }$

Assumption 1 (Shared nuisance). Within each regime $q ,$ the centered coordinates admit the sharednuisance decomposition

$$
a _ { i } ^ { \circ } = L _ { q } \xi _ { i } + s _ { i } + \varepsilon _ { i } ^ { a } , \qquad u _ { i } ^ { \circ } = T _ { q } \xi _ { i } + \varepsilon _ { i } ^ { u } .
$$

Here $\xi _ { i }$ is a zero-mean shift-induced nuisance, $s _ { i }$ represents principal semantic (class/instance) variation, and $\varepsilon _ { i } ^ { a } , \varepsilon _ { i } ^ { u }$ are residuals. The tail cue is uncorrelated with $s _ { i }$ and $\varepsilon _ { i } ^ { a } , \mathrm { i . e . , } \mathbb { E } _ { q } [ u _ { i } ^ { \circ } s _ { i } ^ { \top } ] =$ $\mathbb { E } _ { q } [ u _ { i } ^ { \circ } ( \varepsilon _ { i } ^ { a } ) ^ { \top } ] = 0$

Proposition 1 (Tail-conditioned nuisance identification). Under Assumption 1, let $\Sigma _ { \xi u } = \mathbb { E } _ { q } [ \xi _ { i } u _ { i } ^ { \circ } { } ^ { \top } ]$ and $\Sigma _ { u u } = \mathbb { E } _ { q } [ u _ { i } ^ { \circ } u _ { i } ^ { \circ } { } ^ { \top } ]$ . For any $\lambda \geq 0 ,$ , the population ridge estimator

$$
R _ { q , \lambda } ^ { * } = \arg \operatorname* { m i n } _ { \boldsymbol { R } } \mathbb { E } _ { q } \left[ \| \boldsymbol { a } _ { i } ^ { \circ } - \boldsymbol { R } ^ { \top } \boldsymbol { u } _ { i } ^ { \circ } \| _ { 2 } ^ { 2 } \right] + \lambda \| \boldsymbol { R } \| _ { F } ^ { 2 }\tag{10}
$$

(with the minimum-norm solution at $\lambda = 0 )$ satisfies

$$
R _ { q , \lambda } ^ { * \top } u _ { i } ^ { \circ } = L _ { q } \Sigma _ { \xi u } ( \Sigma _ { u u } + \lambda I _ { d } ) ^ { \dagger } u _ { i } ^ { \circ } \in \mathrm { c o l } ( L _ { q } ) .\tag{11}
$$

Thus, the predicted component is uncorrelated with $s _ { i } ,$ which remains in the residual. $A t \lambda = 0 ,$ , the prediction is the minimum-MSE linear estimate of $L _ { q } \xi _ { i }$ from $u _ { i } ^ { \circ }$ . It recovers $L _ { q } \xi _ { i }$ exactly $i f \varepsilon _ { i } ^ { u } = 0$ $\mathrm { \bar { C o v } } _ { q } ( \xi _ { i } ) \succ 0 .$ , and $T _ { q }$ hasfull column rank.

Proposition 1 characterizes the population target of tail-conditioned regression. TALOR approximates this mechanism through finite-batch, soft-weighted ridge regression and EMA updates; the complete proof and scope are given in App. D.1.

Independent structural support comes from SoMA [52], whose SVD analysis of pre-trained weights finds general, class-shared features in large singular components and texture-biased, context-specific features in minor ones (App. A). Since tail-correlated semantic variation can also be removed, we assess the empirical plausibility of the shared-nuisance interpretation through complementary covariance, correction, and negative-control diagnostics (App. D.2): tail and principal drifts are coupled well above a permutation baseline $( \mathrm { F i g . ~ } \bar { 3 } \mathrm { c } )$ , the observed tail covaries more strongly with the corruption-induced principal displacement than with the clean principal coordinates for every corruption, and without covariate shift TALOR stays within 0.5 pp of Source accuracy (App. E).

## 4.3 Soft routing and online affine estimation

Tail-energy responsibilities. The affine model in Eq. 9 is regime-conditioned, but the online stream is unlabeled. TALOR therefore estimates the two regime models with soft responsibilities rather than hard pseudo-labels. Following the use of residual-subspace energy in [37], we use the tail energy

$$
e _ { i } = \| u _ { i } \| _ { 2 } ^ { 2 }
$$

as a weak routing statistic. Given a mini-batch $B _ { t }$ , TALOR fits a two-component Gaussian mixture model to $\{ e _ { i } \} _ { i = 1 } ^ { B }$ . Let the mixture parameters be $( \alpha _ { 1 } , m _ { 1 } , s _ { 1 } ^ { 2 } )$ and $( \alpha _ { 2 } , \dot { m } _ { 2 } , s _ { 2 } ^ { 2 } )$ , where $\alpha _ { j }$ is the mixture weight, $m _ { j }$ is the component mean, and $s _ { j } ^ { 2 }$ is the component variance. The posterior responsibility of component j for sample i is

$$
r _ { i j } = \frac { \alpha _ { j } \mathcal { N } ( e _ { i } \mid m _ { j } , s _ { j } ^ { 2 } ) } { \sum _ { \ell = 1 } ^ { 2 } \alpha _ { \ell } \mathcal { N } ( e _ { i } \mid m _ { \ell } , s _ { \ell } ^ { 2 } ) } .\tag{12}
$$

The tail energy measures how much the normalized feature occupies the low-gain tail subspace induced by the source head. Although corrupted csID samples are shifted, they still belong to the source label space and empirically tend to preserve stronger occupancy in this tail subspace. In contrast, csOOD samples lack a matching source class and typically exhibit weaker normalized occupancy there. We therefore assign the lower-energy component to the OOD-dominated branch, j<sub>OOD</sub> = arg min $\cdot j \in \{ 1 , 2 \}  ^ { m _ { j } }$ , and define

$$
w _ { i } ^ { ( \mathrm { O O D } ) } : = r _ { i j \mathrm { O O D } } , \qquad w _ { i } ^ { ( \mathrm { I D } ) } : = 1 - w _ { i } ^ { ( \mathrm { O O D } ) } .\tag{13}
$$

The GMM is only a weak router on the scalar $e _ { i } ,$ not an assumption that classes or corruption modes are Gaussian [11, 26]; its weights serve solely as regression responsibilities and to mix drift estimates, so tail energy is a lightweight soft-routing cue rather than a universally calibrated ID/OOD score, and TALOR does not require perfect open-set routing (Sec. 4.5).

Weighted ridge estimation. For each branch $q \in \mathcal { Q }$ , TALOR first computes branch-weighted batch means:

$$
\bar { u } ^ { ( q ) } = \frac { \sum _ { i = 1 } ^ { B } w _ { i } ^ { ( q ) } u _ { i } } { \sum _ { i = 1 } ^ { B } w _ { i } ^ { ( q ) } } , \qquad \bar { a } ^ { ( q ) } = \frac { \sum _ { i = 1 } ^ { B } w _ { i } ^ { ( q ) } a _ { i } } { \sum _ { i = 1 } ^ { B } w _ { i } ^ { ( q ) } } .\tag{14}
$$

Define the batch-centered coordinates $\hat { u } _ { i } ^ { \circ ( q ) } \ : = \ : u _ { i } \ : - \ : \bar { u } ^ { ( q ) }$ and $\hat { a } _ { i } ^ { \circ ( q ) } \ : = \ : a _ { i } \ : - \ : \bar { a } ^ { ( q ) }$ , as empirical counterparts of the regime-centered coordinates in Sec. 4.2. Let $\dot { X } ^ { ( q ) } \in \mathbb { R } ^ { B \times d }$ and $Y ^ { ( q ) } \in \mathbb { R } ^ { B \times k }$ be row-stacked matrices whose i-th rows are $( \hat { u } _ { i } ^ { \circ ( q ) } ) ^ { \top }$ and $( \hat { a } _ { i } ^ { \circ ( q ) } ) ^ { \top }$ , respectively, and let $\Omega ^ { ( q ) } =$ dia $\mathrm { g } ( w _ { 1 } ^ { ( q ) } , \dots , w _ { B } ^ { ( q ) } )$ . For $\lambda > 0$ , TALOR estimates the tail-conditioned slope by weighted ridge regression:

$$
\hat { R } ^ { ( q ) } = \arg \operatorname* { m i n } _ { R } \Big \| ( \Omega ^ { ( q ) } ) ^ { 1 / 2 } \left( X ^ { ( q ) } R - Y ^ { ( q ) } \right) \Big \| _ { F } ^ { 2 } + \lambda \| R \| _ { F } ^ { 2 } .\tag{15}
$$

The primal closed-form solution is

$$
\hat { R } ^ { ( q ) } = \left( X ^ { ( q ) \top } \Omega ^ { ( q ) } X ^ { ( q ) } + \lambda I _ { d } \right) ^ { - 1 } X ^ { ( q ) \top } \Omega ^ { ( q ) } Y ^ { ( q ) } .\tag{16}
$$

Because the ridge problem is solved on centered coordinates, $\hat { R } ^ { \left( q \right) }$ estimates only the slope between tail and principal deviations. The principal mean $\bar { a } ^ { ( q ) }$ is retained separately as the affine intercept.

Efficient small-batch solver. Streaming TTA often uses small mini-batches, while the tail dimension $d$ can be several hundred or larger. We therefore solve Eq. 15 in the dual. Let $X _ { w } ^ { ( q ) } = ( \Omega ^ { ( q ) } ) ^ { 1 / 2 } X ^ { ( q ) }$ and $Y _ { w } ^ { ( q ) } = ( \Omega ^ { ( q ) } ) ^ { 1 / 2 } Y ^ { ( q ) }$ . Using the push-through identity [47, 15], we obtain

$$
\hat { R } ^ { ( q ) } = ( X _ { w } ^ { ( q ) } ) ^ { \top } \left( X _ { w } ^ { ( q ) } ( X _ { w } ^ { ( q ) } ) ^ { \top } + \lambda I _ { B } \right) ^ { - 1 } Y _ { w } ^ { ( q ) } .\tag{17}
$$

In practice, we solve the corresponding symmetric positive definite linear system without forming an explicit inverse [18]. The dual formulation reduces the matrix factorization cost from $O ( d ^ { 3 } )$ to $O ( B ^ { 3 } )$ when $B \ll d \left[ 1 4 \right]$

Online update. To smooth noisy mini-batch estimates while tracking non-stationary streams, TALOR updates the branch states with exponential moving averages:

$$
\mathbf { s } _ { t } ^ { ( q ) } \gets \beta \mathbf { s } _ { t - 1 } ^ { ( q ) } + ( 1 - \beta ) \hat { \mathbf { s } } ^ { ( q ) } ,\tag{18}
$$

where ${ \bf s } _ { t } ^ { ( q ) } = ( R _ { t } ^ { ( q ) } , \mu _ { u , t } ^ { ( q ) } , \mu _ { a , t } ^ { ( q ) } )$ denotes the branch states, $\hat { \mathbf { s } } ^ { ( q ) } = ( \hat { R } ^ { ( q ) } , \bar { u } ^ { ( q ) } , \bar { a } ^ { ( q ) } )$ contains the corresponding mini-batch estimates, and $\beta \in [ 0 , 1 )$ is the EMA momentum.

## 4.4 Rectified inference

After the online states are updated, TALOR applies the learned affine rectifier without modifying either the backbone or the source head. For each sample, it first predicts branch-specific principal drift estimates and then averages them using the current soft responsibilities:

$$
\hat { \Delta } _ { i } ^ { ( q ) } = \mu _ { a , t } ^ { ( q ) } + R _ { t } ^ { ( q ) \top } \left( { u } _ { i } - { \mu } _ { u , t } ^ { ( q ) } \right) , \quad q \in \mathcal { Q } , \qquad \hat { \Delta } _ { i } = \sum _ { q \in \mathcal { Q } } w _ { i } ^ { ( q ) } \hat { \Delta } _ { i } ^ { ( q ) } .\tag{19}
$$

The resulting drift estimate is subtracted only from the principal coordinates. The tail coordinates are left unchanged; the corrected coordinate vector is mapped back to feature space and rescaled by the original feature norm $\rho _ { i }$ , which acts as a norm anchor rather than an exact constraint since $\| \tilde { \hat { v } } _ { i } \| _ { 2 }$ need not equal 1, before applying the frozen classifier:

$$
\begin{array} { r } { \widetilde { a } _ { i } = a _ { i } - \hat { \Delta } _ { i } , \qquad \widetilde { \widetilde { v } } _ { i } = B _ { p } ^ { \top } \widetilde { a } _ { i } + B _ { t } ^ { \top } u _ { i } , \qquad \widetilde { v } _ { i } = \rho _ { i } \widetilde { \widetilde { v } } _ { i } , \quad \widetilde { z } _ { i } = W \widetilde { v } _ { i } + b . } \end{array}\tag{20}
$$

## 4.5 Mechanism Analysis

We further examine whether the online affine drift estimated by TALOR captures a removable principal component.

Protocol. A frozen ViT-B/16 processes the 15 ImageNet-C corruptions at severity 5 (5,000 clean/corrupted pairs each, batch size 64); only the unlabeled corrupted stream drives TALOR, while clean views define the oracle drift $\Delta a _ { i } = a _ { i } ^ { \mathrm { c o r r } } - a _ { i } ^ { \mathrm { c l e a n } }$ . We report the cosine between $\hat { \Delta } _ { i }$ and $\Delta a _ { i }$ and the principal-drift reduction $\mathrm { P D R } = \mathbf { \bar { ( } | | \Delta \bar { a _ { i } } | | _ { 2 } - | | \Delta { a _ { i } } - }$ $\hat { \Delta } _ { i } \| _ { 2 } ) / ( \| \Delta a _ { i } \| _ { 2 } + \epsilon ) ( \mathrm { A p p . ~ C ) }$

![](images/c477917c42964a0e0a36475813fa6d5b958818eca285f21962696854b8a85d0d.jpg)

![](images/463c5ce7f9ba3f2e6ed1fe42b8e676fe8cc6f92d68fb547e7ab647ed3ed3374c.jpg)  
(a) Correction diagnostics.  
(b) ID/OOD slope subspaces.  
Figure 4: (a) Contrast’s clean-corrupted analysis of the predicted affine principal drift. (b) Energy-truncated principal-angle spectrum between ID and OOD tail-to-principal slope subspaces.

Removing principal drift. Fig. 4(a) shows that the predicted drift aligns with the oracle principal drift, reduces the measured drift after subtraction, and improves corrupted-sample accuracy, supporting the central modeling assumption that a substantial part of the harmful high-gain shift is predictable from online tail-to-principal statistics. The slope alone reduces the squared principal drift for all 15 corruptions, and the pattern persists in the open-set benchmark stream (App. C.4 and D.2).

Shared and regime-specific slopes. Fig. 4(b) compares the energy-truncated top-r left-singular subspaces of the ID and OOD slope maps (App. F). Each subspace retains at least 90% of its slopemap Frobenius energy (r = 38 for the displayed cell), which avoids dimension-forced zero angles in the full column spaces. Over the first ∼10 directions, the ID/OOD cross-half spectrum stays far below the same-dimensional Haar-random reference and close to the ID split-half spectrum; beyond them, it departs toward the random reference. This indicates a shared leading-energy component together with residual regime-specific directions rather than equality of the full slope maps.

This evidence is consistent with the routing ablation in Tab. 4: a single affine rectifier can exploit the shared leading component, while soft routing can refine residual ID/OOD-specific structure. It also helps explain why TALOR is less brittle when unsupervised routing is imperfect.

## 5 Experiments

## 5.1 Experimental Setup

Benchmarks and Baselines. To evaluate performance under non-stationary open-set streams, we construct protocols combining covariate-shifted in-distribution and out-of-distribution samples. For the csID component, we employ the standard ImageNet-C [16] benchmark. To assess generalization to additional domain shifts, we also use the highly nonlinear ImageNet-R [17], ImageNet-Sketch [43] and the recently released LAION-C [25] benchmark. For the csOOD component, we follow established practices by applying the corresponding corruptions to a diverse set of OOD datasets, including Places365 [56], Textures [8], NINCO [2], iNaturalist [40], SSB-Hard [41], and SUN [48]. We compare our method against a comprehensive suite of state-of-the-art baselines, including strong continual TTA methods (CoTTA [45], SAR [34], EATA [33], ViDA [27], DPCore [54]) and specialized Open-Set TTA frameworks (STAMP [50], OSTTA [23], UniEnt [11], COME [53]).

Table 1: Results (%) for ImageNet-C benchmark across six csOOD datasets. Improvement and degradation compared to the baseline are highlighted in green or red respectively. All the results are averaged over 15 corruptions.
<table><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td>Improve</td><td>+ 4 +10</td></tr><tr><td>39.9 63. 33 55 6 40.4 TALOR 3 9 .0 78 +</td><td>66. 55. .8 58. 53 5 63 6 45 62. 6 48. .2 .5 83 56. .4 69. 54. .2 76. 46.0 73 4 .9 5 3 .4 +1 + +0. .6 + .8 +5.2</td></tr><tr><td>Improve +0. .5 + 13. 3.6 +0. - S 5 0 .4 46. 2 + OR 5 3 3 4 50. .9 87 52 Improve +3. .2 2 3 +6. +4 X</td><td>+ 45. .6 48 5 49. 2 9 62 54 65 9 55.2 51 .6 92 54. .0 9 6 .5 3 8 5 6 3 8 54.4 84.3 65.4</td></tr><tr><td>50.8 .8 66.6 55 7 3 4 + 5.8</td><td>+4.8 +2. 2 + 15 .2 +4. .2 +29. +14. × 7 58. .0 3 61 .8 92.9 58. 56.8 60.2 2 .4 85.3 68.1</td></tr><tr><td>Improve +20. + +33. .9 +1.3 +25. 50. .4 65. 3 .3 68. 50.3 76.9 LOR 56.5 80.2 56. 3 86.0 56.4 93.5 + Improve +6. +1 9 +6. +6.1 +16. 6</td><td>+34. 2 + +1.2 +23.5 +9.6 69.5 × 50.3 50.6 56.4 50.4 64. 2 50.4 66.7 56.4 56.4 91 56.5 .0 56.5 80. 56.4 83.7 66.6 +23. +5.9 .8</td></tr><tr><td>iEnt 59. 74.0 59 .4 83.4 59. 81.2 + TALOR 60. 60.2 90.6 59.9 96.0 Improve +0. 9 +0. 0.8 +7.2 +4.9</td><td>+6.1 + +6. +1 +6. 0 +17 + 44.2 78.0 59.5 62. 58. 7 .8 56. 76. 63. .9 44.7 81.5 60.4 72 59.6 82.6 .5 84. 67. +0.5 +3.5 +0.9 +0. 9 .8 +7.4 +3.4 3 6 +</td></tr><tr><td>S 5 .0 .9 78. .4 51.9 TA LÖR 54.0 82 3 90.3 54.3 + Improve +2.5 +2.3 +2.4 +10.4 +2.4 4+15. +11. .9</td><td>85. 52.0 78.9 60.9 .9 2 9 60. 2 96. 54.5 94. 54.5 54.2 .4 3 86. 65. 9 +2.5 +10.8 +2.4 9 +5.7</td></tr><tr><td>58.4 76.0 58.0 78.7 51.4 81.5 59.0 + TALOR 59.2 84.4 58.6 90.9 52.1 91.5 59.6</td><td>+2.4 +11.6 + 82.0 59.0 58.4 56.3 71 57.0 74.6 63.7 94.2 59.6 75.3 57.1 82.5 57.7 86.5 68.4</td></tr><tr><td>Improve +0.8 +8.4 +0.6 +0.7+10.0 27 56. 33.9 78.3 32.5 84.0 ore .5 5 56.7 R 88.1 92.0 + +0.2 +0.4 +9.9 +0.4</td><td>+0.6+16.9 +0.8 + +0.7 + +4. 32.4 58. 65. 55.6 83.3 55.9 2 76.6 63. 5 92.0 76.2 56.0 85.6 66.6 +7.9 +0.3 +10.3 +0.2 +0.4 +7.6 +1 +0.3 +9.0 +3.</td></tr></table>

Evaluation Protocol. We employ three quantitative metrics to provide a holistic assessment of OSTTA performance. First, we report the ID classification accuracy (Acc) to measure the model’s domain generalization capability on known classes. Second, we use the area under the ROC curve (AUC) to evaluate the effectiveness of outlier detection in a threshold-free manner. Finally, to balance these two objectives, we report the H-score, defined as the harmonic mean of Acc and AUC, which serves as the primary indicator of overall performance in open-set adaptation scenarios <sup>2</sup>.

Implementation Details. To ensure consistency and reproducibility, our experimental framework is built upon the codebase of DPCore[54]. The primary results utilize a ViT-B/16 backbone pretrained on ImageNet-1K from the timm library, additional results using ResNet-50 are detailed in App. G. For OOD detection, we employ the energy score [28] as the scoring function to compute AUC. We fix the batch size to 64 and the global csID:csOOD ratio to 1:1, the common OSTTA protocol [11, 23, 53, 26]; samples are mixed at the dataset level, so the per-batch composition fluctuates naturally (Experiments with different ratios are provided in App. H).

To simulate realistic deployment conditions and ensure a fair blind evaluation, we follow the protocol in [50]: hyperparameters for all competing methods are tuned exclusively on the first dataset-domain combination. These settings are subsequently frozen and applied directly to all unseen datasets to rigorously evaluate generalization robustness. To ensure consistency and avoid dataset-specific tuning, we fix τ=0.8, β=0.9, and λ=0.01 for all main results before any exposure to target data to fairly quantify the incremental performance gains attributable to the TALOR mechanism. Regarding specific baselines, we select the strongest configuration for UniEnt by integrating EATA with the UniEnt+ variant. For COME, we adopt its optimal version reported for both continual and open-world adaptation scenarios.

Table 2: Results (%) on LAION-C benchmark. Dataset and mean columns report H-score.
<table><tr><td>Method</td><td>Places365-C</td><td>Textures-C</td><td>iNaturalist-C</td><td>SUN-C</td><td>SSB-Hard-C</td><td>NINCO-C</td><td>Mean H-s</td></tr><tr><td>Source CoTTA EATA SAR OSTTA ViDA UniEnt STAMP</td><td>55.4 (+10.5) 54.4 (+11.5)  $6 2 . 0 \overset { \cdot } { ( } + 7 . 2 \overset { \cdot } { ) }$  38.2 (+18.4) 57.1 (+11.3) 54.4 (+11.6) 65.6 (+5.4) 56.6 (+9.6)</td><td>57.5 (+10.7) 55.8 (+12.4)  $6 5 . 3 \ : \dot { ( } + 7 . 8 ) \ : \mathring { \ : }$   $3 9 . 7 \ ( \div 1 5 . 2 )$   $5 4 . 1 \ \AA + 1 5 . 3 \AA$   $5 6 . 6 \stackrel { . } { ( } + 1 1 . 7 \stackrel { . } { ) }$   $7 0 . 3 \ : \dot { ( } + 3 . 9 ) $ </td><td> $6 0 . 5 \ : ( + 9 . 7 )$   $5 8 . 5 ( + 1 1 . 7 ) $   $6 6 . 2 \stackrel { \cdot } { ( } + 7 . 0 ) ^ { \prime }$   $3 8 . 5 \ : ( + 2 1 . 2 ) $   $5 5 . 5 \ : ( + 1 4 . 6 )$   $5 8 . 0 \dot { ( } + 1 1 . 8 \dot { ) }$   $7 2 . 4 \dot { ( } + 3 . 2 ) ^ { }$ </td><td> $5 8 . 4 \ : ( + 1 1 . 1 ) $   $5 7 . 0 \ : \dot { ( } + 1 2 . 6 \dot { ) }$   $6 5 . 3 \ : \dot { ( } + 8 . 0 ) \ : \dot  $   $4 0 . 2 \ : ( \dot { + } 1 8 . 6 )$   $5 6 . 6 \ : ( + 1 3 . 8 )$   $5 6 . 9 \ \dot { ( } + 1 2 . 5 \dot { ) }$   $7 0 . 8 \ : ( + 3 . 8 )$ </td><td> $5 4 . 8 \ : ( + 6 . 9 )$   $5 4 . 6 \ : ( + 7 . 1 )$   $6 2 . 1 \ \mathring { ( + 6 . 0 ) }$   $5 0 . 3 \ \AA + 9 . 6 \AA$   $6 0 . 5 \ : ( + 5 . 4 )$   $5 4 . 6 \ : ( + 7 . 0 ) $   $ \bar { 6 } 2 . 8 \ \backslash + \bar { 5 } . 1 \backslash$ </td><td>56.8 (+8.7)  $5 6 . 0 \ : \AA + 9 . 6 \AA$   $6 2 . 2 \ ( + 8 . 2 ) $   $4 5 . 1 ( + 1 4 . 4 )$   $6 0 . 6 \dot { ( + 8 . 3 ) }$  56.1 (+9.4)</td><td> $5 7 . 2 \ : ( + 9 . 6 )$   $5 6 . 1 \ : ( \dot { + } 1 0 . 8 )$   $6 3 . 8 \ : \dot { ( } + 7 . 4 ) \ : $   $4 2 . 0 \ : ( \dot { + } 1 6 . 2 )$   $5 7 . 4 \ : \AA ( + 1 1 . 5 )$   $5 6 . 1 \ \AA + 1 0 . 7 \AA$ </td></tr></table>

Table 3: Closed-set accuracy (%) results under five evaluation settings. Each entry reports the original accuracy with the TALOR improvement in parentheses.
<table><tr><td>Method</td><td>LAION-C</td><td>ImgNet-R</td><td>ImgNet-Sketch</td><td>Structured (CSC)</td><td>Dynamic (CDC)</td></tr><tr><td>Source</td><td>51.4 (+6.1)</td><td>42.9 (+2.8)</td><td>45.8 (+3.2)</td><td>49.8 (+6.1)</td><td>49.8 (+5.5)</td></tr><tr><td>CoTTA</td><td>51.4 (+6.2)</td><td>43.2 (+2.7)</td><td>46.1 (+3.2)</td><td>50.0 (+6.0)</td><td>49.4 (+6.6)</td></tr><tr><td>EATA</td><td>61.1 (+1.6)</td><td>50.9 (+0.2)</td><td>52.6 (+1.1)</td><td>59.5 (+0.8)</td><td>58.5 (+0.8)</td></tr><tr><td>SAR</td><td>49.1 (+8.0)</td><td>45.9 (+3.1)</td><td>50.0 (+2.0)</td><td>58.3 (+1.5)</td><td>48.8 (+4.2)</td></tr><tr><td>OSTTA</td><td>56.4 (+3.4)</td><td>46.2 (+2.5)</td><td>49.1 (+1.9)</td><td>58.4 (+1.5)</td><td>58.2 (+1.3)</td></tr><tr><td>ViDA</td><td>51.1 (+6.2)</td><td>43.1 (+2.7)</td><td>46.2 (+3.1)</td><td>50.5 (+5.7)</td><td>50.8 (+5.1)</td></tr><tr><td>UniEnt</td><td>60.1 (+1.8)</td><td>47.6 (+1.1)</td><td>49.5 (+1.8)</td><td>59.4 (+1.3)</td><td>58.6 (+1.0)</td></tr><tr><td>STAMP</td><td>51.4 (+5.3)</td><td>43.5 (+2.7)</td><td>46.3 (+3.0)</td><td>51.5 (+4.0)</td><td>51.8 (+3.1)</td></tr><tr><td>COME</td><td>59.6 (+1.5)</td><td>49.6 (+0.2)</td><td>51.6 (+1.1)</td><td>55.4 (+1.1)</td><td>55.1 (+0.9)</td></tr><tr><td>DPCore</td><td>65.3 (+0.1)</td><td>47.6 (+0.1)</td><td>48.3 (+0.8)</td><td>60.5 (+0.4)</td><td>57.4 (+0.6)</td></tr></table>

## 5.2 Main Results

ImageNet-C Benchmark. Tab. 1 presents the comprehensive performance comparison on the ImageNet-C benchmark with 15 corruptions at highest severity level. TALOR delivers consistent gains across all baselines, improving the mean H-score by a clear margin. These consistent gains suggest that head-induced principal correction is complementary to current TTA objectives and provides a useful post-correction signal beyond their standard adaptation losses. Even when deployed as a standalone plug-in on the source backbone, TALOR achieves a mean H-score of 66.1%, surpassing the strongest original baseline UniEnt (63.9%) while maintaining near-inference speed. Notably, TALOR mitigates the severe degradation of SAR under continual domain and semantic shifts, boosting its mean H-score from 51.1% to 65.4%, a substantial improvement of +14.4%. Furthermore, augmenting COME with TALOR attains the state-of-the-art mean H-score of 68.4% with a +4.7% gain, establishing it as the most competitive solution under this protocol. The gains persist under imbalance: standalone TALOR improves the Source H-score by +8.2% at csID:csOOD=8:2 and +10.2% at 3:7 (App. H). These gains come with negligible additional time and GPU memory overhead, indicating that TALOR improves robustness with minimal cost. The efficiency comparison is provided in App. I.

LAION-C Benchmark. As shown in Tab. 2, TALOR also generalizes to the more recent LAION-C benchmark with six challenging domain shifts, most of which are nonlinear shifts. Even without parameter updates, the standalone plug-in yields a +9.6% gain on the frozen source model. Substantial improvements are observed across representative adaptation methods, boosting CoTTA by +10.8%, EATA by +7.4%, OSTTA by +11.5%, and ViDA by +10.7% in mean H-score. Furthermore, it advances the strongest baselines, upgrading UniEnt to about 72.5% and COME to 71.2%, indicating that TALOR provides a complementary correction across diverse shift types.

Robustness across Closed-Set and Continual Protocols. To test whether the correction mechanism transfers beyond open-set streams, we evaluate TALOR in the closed-set setting where only csID samples are present (Tab. 3). Since the test stream contains no unknowns, both soft-routed branches of TALOR are estimated from csID samples only. Despite closed-set TTA being a thoroughly explored field, TALOR consistently yields stable gains across all baselines, suggesting that head-induced principal correction can remain useful even when the label space is fixed. The first three columns highlight TALOR’s broad generalization across diverse domain shifts. On benchmarks ranging from the synthetic corruptions of LAION-C to the natural shifts of ImageNet-R and Sketch, TALOR delivers uniform improvements. Furthermore, we adopt the CSC and CDC protocols from [54] to assess stability under complex temporal dynamics. As shown in last two columns, TALOR proves effective not only in the structured transitions of CSC but also in the chaotic, irregular recurrence of CDC. Collectively, these results indicate that TALOR can be effective in both open- and closed-set scenarios and remains stable across static and dynamic data streams.

![](images/ef2355e54d20f4a4b8291a3f225573365002854eef799b960957fe641287d82b.jpg)  
(a) Energy threshold τ

![](images/e9b8fcdea6c1f3288cdf82f4d1641593d76992d6a0dec3ed254db37fb8d5378e.jpg)  
(b) EMA momentum $\beta$

![](images/34739d9a91b2fa99da91fc4d179e32b7102d622f241c2f79b31ddd76a8f144fa.jpg)  
(c) Ridge coefficient λ

![](images/3dc92ea7c4621ac90c3a6ce662b44421c7439956a411e1de4e5dcbdcbd700e17.jpg)  
(d) Batch Size

Figure 5: Ablation studies on ImageNet-C benchmark averaged on six csOOD datasets. Table 4: Module ablation study.
<table><tr><td rowspan="2">Method</td><td colspan="3">Source</td><td colspan="3">EATA</td><td colspan="3">DPCore</td><td colspan="3">UniEnt</td><td colspan="3">COME</td><td colspan="3">OSTTA</td><td colspan="3">SAR</td></tr><tr><td></td><td></td><td>Acc AUC H-s</td><td></td><td>Acc AUC</td><td>H-s</td><td></td><td></td><td>Acc AUC H-s</td><td>Acc AUC</td><td></td><td>H-s</td><td>Acc</td><td></td><td>AUC H-s</td><td>Acc AUC H-s</td><td></td><td></td><td>Acc</td><td>AUC</td><td>H-s</td></tr><tr><td>Origin</td><td>49.8</td><td>68.0</td><td>56.4</td><td>45.5</td><td>62.6</td><td>48.2</td><td>54.8</td><td>76.6</td><td>63.5</td><td>56.7</td><td>76.7</td><td>63.9</td><td>57.0</td><td>74.6</td><td>63.7</td><td>56.1</td><td>61.8</td><td>58.4</td><td>50.2</td><td>55.2</td><td>51.1</td></tr><tr><td>Mean-only</td><td>54.8</td><td>75.4</td><td>62.3</td><td>45.8</td><td>65.3</td><td>50.2</td><td>54.8</td><td>77.3</td><td>63.7</td><td>57.3</td><td>78.2</td><td>64.9</td><td>57.3</td><td>79.0</td><td>65.5</td><td>57.0</td><td>73.7</td><td>63.7</td><td>53.7</td><td>76.9</td><td>62.3</td></tr><tr><td>Slope-only</td><td>51.0</td><td>80.8</td><td>61.4</td><td>45.6</td><td>73.1</td><td>52.4</td><td>55.1</td><td>85.3</td><td>66.5</td><td>56.8</td><td>83.6</td><td>66.6</td><td>57.3</td><td>85.2</td><td>67.8</td><td>56.5</td><td>82.1</td><td>66.5</td><td>49.7</td><td>73.6</td><td>58.4</td></tr><tr><td>Full</td><td>55.9</td><td>83.5</td><td>66.1</td><td>46.0</td><td>73.9</td><td>53.4</td><td>55.1</td><td>85.6</td><td>66.6</td><td>57.5</td><td>84.1</td><td>67.3</td><td>57.7</td><td>86.5</td><td>68.4</td><td>57.4</td><td>85.3</td><td>68.1</td><td>54.4</td><td>84.3</td><td>65.4</td></tr><tr><td>Single Branch</td><td>55.6</td><td>82.4</td><td>65.4</td><td>45.8</td><td>73.4</td><td>53.0</td><td>54.9</td><td>84.9</td><td>66.1</td><td>57.3</td><td>83.3</td><td>66.8</td><td>57.3</td><td>86.1</td><td>68.0</td><td>57.2</td><td>84.7</td><td>67.6</td><td>54.3</td><td>83.5</td><td>64.9</td></tr><tr><td>Oracle</td><td>55.7</td><td>86.3</td><td>66.9</td><td>47.1</td><td>76.0</td><td>55.0</td><td>55.0</td><td>88.4</td><td>67.4 57.5</td><td></td><td>86.9</td><td>68.2</td><td>57.6</td><td>88.6</td><td>69.0</td><td>57.2</td><td>88.4</td><td>69.0</td><td>54.1</td><td>86.6</td><td>65.9</td></tr></table>

## 5.3 Ablation Studies

Robustness to hyperparameters and batch size. We study the sensitivity of TALOR on the ImageNet-C benchmark. SVD energy threshold. τ controls the split point k between principal and tail subspaces; TALOR exhibits strong stability across different values, with H-score remaining near-plateau over a wide range (Fig. 5a). EMA momentum. β smooths the online updates of branch statistics and drift estimates; performance is similarly stable for moderate-to-large β, while removing EMA (very small β) is more prone to noisy batch-wise estimation (Fig. 5b). Ridge coefficient. λ regularizes the weighted ridge regressor for the tail-to-principal slope term; TALOR remains robust for small-to-moderate λ and only degrades slightly when over-regularized (Fig. 5c). Batch size. While TTA methods typically benefit from the more stable statistics provided by larger batch sizes (Fig. 5d), TALOR demonstrates remarkable robustness to batch size variations. Notably, even under small batches (B=16), TALOR delivers significant improvements over the baselines, achieving high performance comparable to or exceeding that of baselines with larger batches.

Ablating affine terms and soft regime routing. Tab. 4 ablates two design axes of TALOR: the affine form of the principal drift estimator and the use of soft ID/OOD regime routing. (1) Mean-only uses only the branch principal mean, i.e., $\hat { \Delta } _ { i } ^ { ( q ) } = \mu _ { a , t } ^ { ( q ) }$ , while Slope-only uses only the tail-to-principal slope, i.e., $\hat { \Delta } _ { i } ^ { ( q ) } = R _ { t } ^ { ( q ) \top } ( u _ { i } - \mu _ { u , t } ^ { ( q ) } )$ . The two terms are complementary: the mean term mainly improves ID accuracy by recentering the target principal coordinates, whereas the slope term mainly improves AUC by correcting sample-level tail-to-principal leakage, which is larger and better aligned with the paired drift for csOOD than for csID samples (App. C.4). The Full model combines both terms and therefore gives a synergistic Acc–AUC gain across baselines. (2) The routing variants isolate the value of regime separation. Single Branch uses one affine principal rectifier for the whole mixed stream. It remains competitive, consistent with the substantial shared leading-energy slope component in Fig. 4(b), while also providing a degenerate yet effective form of TALOR when extremely severe shifts make routing unreliable. Nevertheless, the H-score and especially AUC generally follow the hierarchy Single Branch < Full TALOR < Oracle across baselines. This ordering is consistent with soft ID/OOD routing refining residual regime-specific structure: the unsupervised GMM moves the estimator beyond a mixed regressor, while GT routing gives the upper bound when that residual structure is separated perfectly.

## 6 Conclusion

We presented TALOR, a lightweight plug-in for Open-Set TTA that uses the frozen source linear head as a geometric anchor for feature rectification. TALOR innovatively estimates a removable affine principal drift through a principal mean and a tail-to-principal slope map, and subtracts this drift estimate, reconstructs the tailored feature and feeds it to the source head. Extensive experiments show that TALOR is effective as a standalone rectifier and as a stabilizing post-correction module across open-set and closed-set continual protocols.

## Acknowledgments and Disclosures

Funding. This work was supported in part by Guangdong S&T Program (2024B0101050004), NSFC (W2611096, Overseas), ICFCRT (W2441020), Shenzhen Science and Technology Program (KJZD20240903100022028, KQTD20210811090044003), and Guangdong Provincial Key Laboratory of Visual Media and Multidimensional Intelligence (2026B1212010012).

Competing Interests. The authors declare no competing interests.

## References

[1] Kazuki Adachi, Shin’ya Yamaguchi, Atsutoshi Kumagai, and Tomoki Hamagami. Test-time adaptation for regression by subspace alignment. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=SXtl7NRyE5.

[2] Julian Bitterwolf, Maximilian Mueller, and Matthias Hein. In or out? fixing imagenet out-ofdistribution detection evaluation. arXiv preprint arXiv:2306.00826, 2023.

[3] Malik Boudiaf, Romain Mueller, Ismail Ben Ayed, and Luca Bertinetto. Parameter-free online test-time adaptation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8344–8353, 2022.

[4] Chentao Cao, Zhun Zhong, Zhanke Zhou, Tongliang Liu, Yang Liu, Kun Zhang, and Bo Han. Noisy test-time adaptation in vision-language models. arXiv preprint arXiv:2502.14604, 2025.

[5] Haozhi Cao, Yuecong Xu, Jianfei Yang, Pengyu Yin, Shenghai Yuan, and Lihua Xie. Multimodal continual test-time adaptation for 3d semantic segmentation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 18809–18819, 2023.

[6] Dian Chen, Dequan Wang, Trevor Darrell, and Sayna Ebrahimi. Contrastive test-time adaptation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 295–305, 2022.

[7] Ziyang Chen, Yongsheng Pan, Yiwen Ye, Mengkang Lu, and Yong Xia. Each test image deserves a specific prompt: Continual test-time adaptation for 2d medical image segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11184–11193, 2024.

[8] Mircea Cimpoi, Subhransu Maji, Iasonas Kokkinos, Sammy Mohamed, and Andrea Vedaldi. Describing textures in the wild. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3606–3613, 2014.

[9] Carl Eckart and Gale Young. The approximation of one matrix by another of lower rank. Psychometrika, 1(3):211–218, 1936.

[10] Yulu Gan, Yan Bai, Yihang Lou, Xianzheng Ma, Renrui Zhang, Nian Shi, and Lin Luo. Decorate the newcomers: Visual domain prompt for continual test time adaptation. In Proceedings ofthe AAAI conference on artificial intelligence, volume 37, pages 7595–7603, 2023.

[11] Zhengqing Gao, Xu-Yao Zhang, and Cheng-Lin Liu. Unified entropy optimization for open-set test-time adaptation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23975–23984, 2024.

[12] Mingrong Gong, Chao-Qi Chen, Lu-Yao Tang, Yuxiao Wang, and Sergio Escalera. 4d point cloud segmentation via active test-time adaptation. In AAAI Conference on Artificial Intelligence, 2026. URL https://api.semanticscholar.org/CorpusID:286691532.

[13] Shurui Gui, Xiner Li, and Shuiwang Ji. Active test-time adaptation: Theoretical analyses and an algorithm. arXiv preprint arXiv:2404.05094, 2024.

[14] Trevor Hastie, Robert Tibshirani, Jerome Friedman, and James Franklin. The elements of statistical learning: data mining, inference and prediction. The Mathematical Intelligencer, 27 (2):83–85, 2005.

[15] Harold V Henderson and Shayle R Searle. On deriving the inverse of a sum of matrices. SIAM review, 23(1):53–60, 1981.

[16] Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. Proceedings of the International Conference on Learning Representations, 2019.

[17] Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu, Samyak Parajuli, Mike Guo, Dawn Song, Jacob Steinhardt, and Justin Gilmer. The many faces of robustness: A critical analysis of out-of-distribution generalization. ICCV, 2021.

[18] Nicholas J Higham. Accuracy and stability ofnumerical algorithms. SIAM, 2002.

[19] Junyuan Hong, Lingjuan Lyu, Jiayu Zhou, and Michael Spranger. Mecta: Memory-economic continual test-time model adaptation. In 2023 International Conference on Learning Representations, 2023.

[20] Qinting Jiang, Chuyang Ye, Dongyan Wei, Bingli Wang, Yuan Xue, Jingyan Jiang, and Zhi Wang. Feature-based instance neighbor discovery: Advanced stable test-time adaptation in dynamic world. arXiv preprint arXiv:2506.06782, 2025.

[21] Daeun Lee, Jaehong Yoon, and Sung Ju Hwang. Becotta: Input-dependent online blending of experts for continual test-time adaptation. arXiv preprint arXiv:2402.08712, 2024.

[22] Jonghyun Lee, Dahuin Jung, Saehyung Lee, Junsung Park, Juhyeon Shin, Uiwon Hwang, and Sungroh Yoon. Entropy is not enough for test-time adaptation: From the perspective of disentangled factors. arXiv preprint arXiv:2403.07366, 2024.

[23] Jungsoo Lee, Debasmit Das, Jaegul Choo, and Sungha Choi. Towards open-set test-time adaptation utilizing the wisdom of crowds in entropy minimization. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 16380–16389, 2023.

[24] Taeckyung Lee, Sorn Chottananurak, Junsu Kim, Jinwoo Shin, Taesik Gong, and Sung-Ju Lee. Test-time adaptation with binary feedback. arXiv preprint arXiv:2505.18514, 2025.

[25] Fanfei Li, Thomas Klein, Wieland Brendel, Robert Geirhos, and Roland S Zimmermann. Laion-c: An out-of-distribution benchmark for web-scale vision models. arXiv preprint arXiv:2506.16950, 2025.

[26] Yushu Li, Xun Xu, Yongyi Su, and Kui Jia. On the robustness of open-world test-time training: Self-training with dynamic prototype expansion. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11836–11846, 2023.

[27] Jiaming Liu, Senqiao Yang, Peidong Jia, Renrui Zhang, Ming Lu, Yandong Guo, Wei Xue, and Shanghang Zhang. Vida: Homeostatic visual domain adapter for continual test time adaptation. arXiv preprint arXiv:2306.04344, 2023.

[28] Weitang Liu, Xiaoyun Wang, John Owens, and Yixuan Li. Energy-based out-of-distribution detection. Advances in neural information processing systems, 33:21464–21475, 2020.

[29] Yingnan Liu, Rui Qiao, Mong Li Lee, and Wynne Hsu. Test-time adaptation by causal trimming. arXiv preprint arXiv:2510.11133, 2025.

[30] Sarthak Kumar Maharana, Baoming Zhang, and Yunhui Guo. Palm: Pushing adaptive learning rate mechanisms for continual test-time adaptation. ArXiv, abs/2403.10650, 2024. URL https://api.semanticscholar.org/CorpusID:268513104.

[31] Sarthak Kumar Maharana, Shambhavi Mishra, Yunbei Zhang, Shuaicheng Niu, Taki Hasan Rafi, Jihun Hamm, Marco Pedersoli, Jose Dolz, and Yunhui Guo. Continual test-time adaptation in computer vision: Methods, benchmarks, and future directions. Transactions on Machine Learning Research, 2026. ISSN 2835-8856. URL https://openreview.net/forum?id= mM3r03Xw1V. Survey Certification.

[32] Chenggong Ni, Fan Lyu, Jiayao Tan, Fuyuan Hu, Rui Yao, and Tao Zhou. Maintaining consistent inter-class topology in continual test-time adaptation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 15319–15328, 2025.

[33] Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, Yaofo Chen, Shijian Zheng, Peilin Zhao, and Mingkui Tan. Efficient test-time model adaptation without forgetting. In International conference on machine learning, pages 16888–16905. PMLR, 2022.

[34] Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, Zhiquan Wen, Yaofo Chen, Peilin Zhao, and Mingkui Tan. Towards stable test-time adaptation in dynamic wild world. arXiv preprint arXiv:2302.12400, 2023.

[35] Shuaicheng Niu, Chunyan Miao, Guohao Chen, Pengcheng Wu, and Peilin Zhao. Test-time model adaptation with only forward passes. arXiv preprint arXiv:2404.01650, 2024.

[36] Mincheol Park, Heeji Won, Won Woo Ro, and Suhyun Kim. Rethinking entropy in test-time adaptation: The missing piece from energy duality. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[37] Sanqing Qu, Tianpei Zou, Lianghua He, Florian Röhrbein, Alois Knoll, Guang Chen, and Changjun Jiang. Lead: Learning decomposition for source-free universal domain adaptation. In CVPR, 2024.

[38] Manogna Sreenivas and Soma Biswas. Efficient open set single image test time adaptation of vision language models. arXiv preprint arXiv:2406.00481, 2024.

[39] Yushun Tang, Shuoshuo Chen, Jiyuan Jia, Yi Zhang, and Zhihai He. Domain-conditioned transformer for fully test-time adaptation. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 6260–6269, 2024.

[40] Grant Van Horn, Oisin Mac Aodha, Yang Song, Yin Cui, Chen Sun, Alex Shepard, Hartwig Adam, Pietro Perona, and Serge Belongie. The inaturalist species classification and detection dataset. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 8769–8778, 2018.

[41] Sagar Vaze, Kai Han, Andrea Vedaldi, and Andrew Zisserman. Open-set recognition: a good closed-set classifier is all you need? In International Conference on Learning Representations, 2022.

[42] Guowei Wang and Changxing Ding. Effortless active labeling for long-term test-time adaptation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 25633– 25642, 2025.

[43] Haohan Wang, Songwei Ge, Zachary Lipton, and Eric P Xing. Learning robust global representations by penalizing local predictive power. In Advances in Neural Information Processing Systems, pages 10506–10518, 2019.

[44] Kunyu Wang, Xueyang Fu, Yuanfei Bao, Chengjie Ge, Chengzhi Cao, Wei Zhai, and Zheng-Jun Zha. Paid: Pairwise angular-invariant decomposition for continual test-time adaptation. arXiv preprint arXiv:2506.02453, 2025.

[45] Qin Wang, Olga Fink, Luc Van Gool, and Dengxin Dai. Continual test-time domain adaptation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7201–7211, 2022.

[46] Ran Wang, Hua Zuo, Zhen Fang, and Jie Lu. Multiple teacher model for continual test-time domain adaptation. In Australasian Joint Conference on Artificial Intelligence, pages 304–314. Springer, 2023.

[47] Max A Woodbury. Inverting modified matrices. Department of Statistics, Princeton University, 1950.

[48] Jianxiong Xiao, James Hays, Krista A Ehinger, Aude Oliva, and Antonio Torralba. Sun database: Large-scale scene recognition from abbey to zoo. In 2010 IEEE computer society conference on computer vision and pattern recognition, pages 3485–3492. IEEE, 2010.

[49] Yingkai Yang, Chaoqi Chen, and Hui Huang. Back to source: Open-set continual test-time adaptation via domain compensation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 7957–7966, June 2026.

[50] Yongcan Yu, Lijun Sheng, Ran He, and Jian Liang. Stamp: Outlier-aware test-time adaptation with stable memory replay. In European Conference on Computer Vision, pages 375–392. Springer, 2024.

[51] Yige Yuan, Bingbing Xu, Liang Hou, Fei Sun, Huawei Shen, and Xueqi Cheng. Tea: Test-time energy adaptation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23901–23911, 2024.

[52] Seokju Yun, Seunghye Chae, Dongheon Lee, and Youngmin Ro. Soma: Singular value decomposed minor components adaptation for domain generalizable representation learning. In Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), pages 25602–25612, June 2025.

[53] Qingyang Zhang, Yatao Bian, Xinke Kong, Peilin Zhao, and Changqing Zhang. Come: Test-time adaption by conservatively minimizing entropy. arXiv preprint arXiv:2410.10894, 2024.

[54] Yunbei Zhang, Akshay Mehra, Shuaicheng Niu, and Jihun Hamm. Dpcore: Dynamic prompt coreset for continual test-time adaptation. arXiv preprint arXiv:2406.10737, 2024.

[55] Wenjie Zhao, Jia Li, Xinyao Dong, Yapeng Tian, Yu Xiang, and Yunhui Guo. Mitigating the id–ood tradeoff in open-set test-time adaptation. 2026 IEEE/CVF Conference on Computer Vision and Pattern Recognition- FINDINGS Track (CVPRF), pages 6792–6801, 2026. URL https://api.semanticscholar.org/CorpusID:287072352.

[56] Bolei Zhou, Agata Lapedriza, Aditya Khosla, Aude Oliva, and Antonio Torralba. Places: A 10 million image database for scene recognition. IEEE transactions on pattern analysis and machine intelligence, 40(6):1452–1464, 2017.

[57] Tianpei Zou, Guo Yu, Ya Wu, Fan Lu, Zhongcong Xu, Zhang Bo, Ziqiao Wang, Sanqing Qu, and Guang Chen. GOOD: Geometry-guided out-of-distribution modeling for open-set test-time adaptation in point cloud semantic segmentation. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=HyNWlZd4iO.

## A Related Work

Continual Test-Time Adaptation. Continual Test-Time Adaptation (CTTA) aims to cope with error accumulation and catastrophic forgetting under non-stationary test distributions [45, 46, 5, 21, 7, 19, 54, 32, 36, 29–31]. A common strategy is to stabilize online updates via regularization: EATA [33] and SAR [34] constrain overly aggressive parameter changes, while FIND [20] decouples features to normalize statistics robustly under dynamic shifts. Parameter-efficient methods further isolate domain-specific knowledge to reduce interference: VDP [10] uses visual prompts to capture distribution shifts, ViDA [27] allocates low-/high-rank adapters homeostatically, and PAID [44] restricts updates to orthogonal transformations to preserve pre-trained geometry.

Open-Set Test-Time Adaptation. OSTTA addresses covariate and semantic shifts by adapting to known classes while rejecting unknowns [49, 55, 57]. STAMP [50] uses class-balanced memory and self-weighted entropy to stabilize updates and outlier detection. OSTTA [23] mitigates open-set noise by filtering losses via confidence gaps, while UniEnt [11] employs distribution-aware filtering to minimize entropy for knowns and maximize it for unknowns. To curb overconfidence, COME [53] models predictive uncertainty for conservative entropy minimization. OWTTT [26] proposes openworld self-training with dynamic prototype expansion, progressively absorbing reliable novel clusters while using prototype-based rejection to separate unknowns. For foundation models, ZS-TTA [4] couples zero-shot VLM classification with an online noise detector trained on frozen VLM features to gate entropy-based adaptation, while ROSITA [38] enables single-image OSTTA through contrastive separation and dynamic feature banks. A unified standard is currently missing regarding whether OSTTA should simultaneously accommodate continual adaptation without model resetting. While OSTTA and UniEnt adopt the non-resetting protocol, methods such as STAMP, COME, and OWTTT employ resetting between distributions. This work follows the more challenging non-stationary setting established by the former.

Subspace-Aware Adaptation. A recent line of work recognizes that different subspaces of the feature representation contribute unequally to model outputs, motivating selective adaptation strategies. SSA [1] identifies the output-significant subspace via PCA of the source-feature covariance and aligns source and target marginal distributions only within this subspace, excluding complementary dimensions from adaptation. SSA focuses on closed-set classification and regression tasks and requires source feature statistics (mean, PCA basis, and variances) at test time; it does not address open-set scenarios or provide regime-conditioned mechanisms for unknown rejection. Independently, SoMA [52] analyzes the weight matrices of vision foundation models via SVD and finds that singular components with large singular values extract general features spanning many classes, while minor singular components mainly capture fine-grained, texture-biased, and context-specific features. Although SoMA studies backbone weights rather than a classifier head, this functional separation suggests that semantic and nuisance variation in TALOR’s principal and tail subspaces are driven by largely distinct underlying factors, providing independent structural support for Assumption 1.

## B Implementation Details

Backbone-dependent head geometry. TALOR constructs its coordinate system once from the frozen linear head before online inference. For ImageNet-1K, the number of classes is $C = 1 0 0 0$ while the classifier-input feature dimension D depends on the backbone. Let $W \in \mathbb { R } ^ { C \times D }$ denote the head weight and $s = \mathsf { \bar { m i n } } ( C , D )$ . We compute the full singular value decomposition

$$
W = U \Sigma V ^ { \top } , \qquad U \in \mathbb { R } ^ { C \times C } , \quad \Sigma \in \mathbb { R } ^ { C \times D } , \quad V \in \mathbb { R } ^ { D \times D } .\tag{21}
$$

The decomposition provides s singular values, including any zeros due to rank deficiency. When $D > C$ , the additional D − s right-singular vectors span directions in the null space of W and have zero singular values. The split point k is selected using the singular-value energy criterion in Eq. 3 and clamped to $k \le s - 1$ , ensuring a non-empty tail subspace. The principal and tail bases used in Sec. 4.1 are defined as

$$
B _ { p } = ( V _ { : , 1 : k } ) ^ { \top } \in \mathbb { R } ^ { k \times D } , \qquad B _ { t } = ( V _ { : , k + 1 : D } ) ^ { \top } \in \mathbb { R } ^ { ( D - k ) \times D } .\tag{22}
$$

Since V is orthogonal, these bases form an orthogonal decomposition of the full feature space, satisfying $B _ { p } B _ { p } ^ { \top } \overset { = } { = } I _ { k } , B _ { t } B _ { t } ^ { \top } = I _ { D - k }$ , and $B _ { p } ^ { \top } B _ { p } + B _ { t } ^ { \top } B _ { t } = \dot { I } _ { D }$ . Thus, TALOR always operates in the D-dimensional feature space rather than the C-dimensional class space; the class dimension influences the decomposition only through the geometry of the frozen classifier head.

ViT-B/16: $D < C .$ . For ViT-B/16, the classifier head has shape $W \in \mathbb { R } ^ { 1 0 0 0 \times 7 6 8 } , \mathrm { { s o } } D = 7 6 8 <$ $C = 1 0 0 0$ and $s = 7 6 8$ . The right-singular basis spans the entire 768-dimensional feature space. After the energy split, TALOR obtains

$$
B _ { p } \in \mathbb { R } ^ { k \times 7 6 8 } , \qquad B _ { t } \in \mathbb { R } ^ { ( 7 6 8 - k ) \times 7 6 8 } .\tag{23}
$$

There is no additional classifier null space beyond the feature dimension; the tail consists of the lower-gain axes within the ViT feature space.

ResNet-50: $D > C .$ . For ResNet-50, the classifier head has shape $W \in \mathbb { R } ^ { 1 0 0 0 \times 2 0 4 8 } , \mathrm { { s o } } D = 2 0 4 8 >$ $C = 1 0 0 0$ and $s = 1 0 0 0$ . The full right-singular basis contains both classifier-supported axes and feature axes that lie in the null space of the head. After selecting $k \leq 9 9 9$ , TALOR forms

$$
B _ { p } \in \mathbb { R } ^ { k \times 2 0 4 8 } , \qquad B _ { t } \in \mathbb { R } ^ { ( 2 0 4 8 - k ) \times 2 0 4 8 } .\tag{24}
$$

The tail therefore contains the low-gain singular vectors $V _ { : , k + 1 : 1 0 0 0 }$ together with the head-null vectors $V _ { : , 1 0 0 1 : 2 0 4 8 }$ . This is the same construction as in the ViT case: changing the backbone only changes the shape of W, and the SVD automatically supplies the appropriate tail geometry. This economy is central to TALOR: a single head-induced decomposition covers both $D < C$ and $D > C$ without architecture-specific rules.

Routing fail-safe. The two-component GMM of Sec. 4.3 is refitted in every batch on the scalar tail energies. If the fit raises a numerical exception, TALOR assigns neutral responsibilities $w _ { i } ^ { \mathrm { ( I D ) } } =$ $w _ { i } ^ { \mathrm { ( O O D ) } } = 0 . 5$ . No label- or dataset-dependent confidence threshold is used, and only the affine branch states are smoothed across batches by the EMA in Eq. 18.

## C Paired Corruption Diagnostics

This appendix defines the paired diagnostics used to validate the rectification mechanism. We use paired samples $( x _ { i } , x _ { i } ^ { \mathrm { c o r r } } )$ where $x _ { i } ^ { \mathrm { c o r r } }$ is a corruption version of $x _ { i }$ . Projecting the normalized features onto the head-induced basis yields principal/tail coordinates $( a _ { i } , u _ { i } )$ for $x _ { i }$ and $( a _ { i } ^ { \mathrm { c o r r } } , u _ { i } ^ { \mathrm { c o r r } } )$ for $x _ { i } ^ { \mathrm { { c o r r } } }$ . Define the coordinate drifts

$$
\Delta a _ { i } = a _ { i } ^ { \mathrm { c o r r } } - a _ { i } ,
$$

$$
\Delta u _ { i } = u _ { i } ^ { \mathrm { c o r r } } - u _ { i } .\tag{25}
$$

## C.1 CCA coupling spectrum

To test whether low-gain drift contains a linearly decodable high-gain drift component, we compute canonical correlation analysis (CCA) between $\dot { \Delta u }$ and $\Delta a .$ . Let X and $Y$ be the centered row-stacked matrices of $\Delta u _ { i }$ and $\Delta a _ { i }$ , respectively. With covariance estimates

$$
\Sigma _ { u u } = \frac { 1 } { n - 1 } X ^ { \top } X , \qquad \Sigma _ { a a } = \frac { 1 } { n - 1 } Y ^ { \top } Y , \qquad \Sigma _ { u a } = \frac { 1 } { n - 1 } X ^ { \top } Y ,\tag{26}
$$

and a small regularizer γ, we form

$$
K = ( \Sigma _ { u u } + \gamma I ) ^ { - 1 / 2 } \Sigma _ { u a } ( \Sigma _ { a a } + \gamma I ) ^ { - 1 / 2 } .\tag{27}
$$

The CCA spectrum is the sequence of singular values

$$
\rho _ { j } = \sigma _ { j } ( K ) , \qquad j = 1 , 2 , \ldots ,\tag{28}
$$

where a large $\rho _ { j }$ means that one linear combination of tail/null drift and one linear combination of principal drift are strongly correlated. For the permutation baseline, we randomly shuffle the rows of $\Delta u .$ , recompute the spectrum, and plot the 95th percentile over permutations. The ViT-B/16 diagnostics are shown in Fig. 3; the ResNet-50 counterpart in Fig. 6 additionally reflects the classifier null axes available when $D > C . { \mathrm { A p p } }$ . D.2 complements this spectrum with a regression-based test of drift predictability against a permutation null.

## C.2 Oracle target ablation

The CCA diagnostic tests whether tail/null drift can serve as a linear sensor. Furthermore, we separately test which subspace is the effective correction target by an oracle ablation. For a target subspace S, the oracle replaces only that corrupted coordinate block with its clean-paired value,

$$
\tilde { c } _ { i , S } ^ { \mathrm { c o r r } } = c _ { i , S } ^ { \mathrm { c o r r } } - \left( c _ { i , S } ^ { \mathrm { c o r r } } - c _ { i , S } \right) = c _ { i , S } ,\tag{29}
$$

![](images/ff84114eaee4c1745a7ef9a5471deb1028a12ce18f2b672345c866a8e5de17d6.jpg)  
(a) Logit impact.

![](images/ce44b7f944eee697951ed30ecee146a44353987a20ffa8d98f48edc354dc12c7.jpg)  
(b) Relative shift.

![](images/65e057d3864e191ab25243df4f9c8ceaade277a978b29958e04b7a46214aaa0b.jpg)  
(c) Tail/null-to-principal CCA.  
Figure 6: ResNet-50 paired motivation diagnostics in the head-induced basis. The same sensor-target pattern holds beyond ViT-B/16; the CCA panel additionally shows that classifier-null coordinates, despite having no direct logit gain, contain linearly decodable information about principal drift.

Table 5: Oracle target correction on paired ImageNet-C diagnostics. Accuracy gains are reported in percentage points; gap closed is the percentage of the clean–corrupt accuracy gap recovered by restoring only the indicated subspace.
<table><tr><td>Backbone</td><td>Metric</td><td>Principal</td><td>Supported tail</td><td>Null only</td><td>Tail+null</td><td>All</td></tr><tr><td rowspan="2">ViT-B/16</td><td>Acc. gain (pp)</td><td>40.9</td><td>15.9</td><td>一</td><td>15.9</td><td>42.1</td></tr><tr><td>Gap closed (%)</td><td>97.0</td><td>38.9</td><td>一</td><td>38.9</td><td>100.0</td></tr><tr><td rowspan="2">ResNet-50</td><td>Acc. gain (pp)</td><td>56.9</td><td>7.2</td><td>0.0</td><td>7.2</td><td>58.7</td></tr><tr><td>Gap closed (%)</td><td>96.7</td><td>13.2</td><td>0.0</td><td>13.2</td><td>100.0</td></tr></table>

while leaving all other coordinates unchanged. We report the accuracy gain

$$
\mathrm { G a i n } ( S ) = \mathrm { A c c } _ { \mathrm { o r a c l e } } ( S ) - \mathrm { A c c } _ { \mathrm { c o r r } } ,\tag{30}
$$

and the fraction of the clean–corrupt gap closed,

$$
\operatorname { G a p C l o s e d } ( S ) = { \frac { \operatorname { A c c } _ { \mathrm { o r a c l e } } ( S ) - \operatorname { A c c } _ { \mathrm { c o r r } } } { \operatorname { A c c } _ { \mathrm { c l e a n } } - \operatorname { A c c } _ { \mathrm { c o r r } } } } .\tag{31}
$$

Table 5 shows that principal coordinates are the dominant correction target, whereas tail/null coordinates are primarily useful as sensors.

## C.3 Per-batch correction curves

Principal drift ratio (mean & std). For each sample we measure how much of the paired drift lies in the principal coordinates:

$$
\ell _ { i } = \frac { \| \Delta a _ { i } \| _ { 2 } } { \| \Delta a _ { i } \| _ { 2 } + \| \Delta u _ { i } \| _ { 2 } } .\tag{32}
$$

For a mini-batch $B _ { t }$ , we plot the mean and standard deviation

$$
\mathrm { L R } _ { t } = \frac { 1 } { | \boldsymbol { \beta } _ { t } | } \sum _ { i \in \boldsymbol { B } _ { t } } \ell _ { i } ,
$$

$$
\mathrm { S t d ( L R ) } _ { t } = \sqrt { \frac { 1 } { \left| \mathcal { B } _ { t } \right| } \sum _ { i \in \mathcal { B } _ { t } } \ell _ { i } ^ { 2 } - \mathrm { L R } _ { t } ^ { 2 } } .\tag{33}
$$

Cosine alignment to the predicted correction (mean & std). TALOR predicts an affine principal correction $\hat { \Delta } _ { i }$ from branch statistics and tail-coordinate deviations, and rectifies the principal coordinate by $\tilde { a } _ { i } = a _ { i } - \hat { \Delta } _ { i }$ (see Eqns. 19 and Eqns. 20 in the main text). For paired diagnostics, we compare $\hat { \Delta } _ { i }$ against the observed principal drift $\Delta a _ { i }$ using cosine similarity:

$$
c _ { i } = \frac { \langle \hat { \Delta } _ { i } , \Delta a _ { i } \rangle } { \| \hat { \Delta } _ { i } \| _ { 2 } \| \Delta a _ { i } \| _ { 2 } + \epsilon } .\tag{34}
$$

We plot batch mean/std as

$$
\mathrm { C o s } _ { t } = \frac { 1 } { | \mathcal { B } _ { t } | } \sum _ { i \in \mathcal { B } _ { t } } c _ { i } , \qquad \mathrm { S t d } ( \mathrm { C o s } ) _ { t } = \sqrt { \frac { 1 } { | \mathcal { B } _ { t } | } \sum _ { i \in \mathcal { B } _ { t } } c _ { i } ^ { 2 } - \mathrm { C o s } _ { t } ^ { 2 } } .\tag{35}
$$

Principal-drift reduction. We quantify how much rectification reduces the paired principal drift magnitude:

$$
d _ { i } ^ { \mathrm { b e f o r e } } = \lVert \Delta a _ { i } \rVert _ { 2 } = \lVert a _ { i } ^ { \mathrm { c o r r } } - a _ { i } \rVert _ { 2 } , ~ d _ { i } ^ { \mathrm { a f t e r } } = \lVert \tilde { a } _ { i } ^ { \mathrm { c o r r } } - a _ { i } \rVert _ { 2 } , ~ \tilde { a } _ { i } ^ { \mathrm { c o r r } } = a _ { i } ^ { \mathrm { c o r r } } - \hat { \Delta } _ { i } ,\tag{36}
$$

and define the normalized reduction

$$
r _ { i } = \frac { d _ { i } ^ { \mathrm { b e f o r e } } - d _ { i } ^ { \mathrm { a f t e r } } } { d _ { i } ^ { \mathrm { b e f o r e } } + \epsilon } , \qquad \mathrm { D R } _ { t } = \frac { 1 } { | \mathscr { B } _ { t } | } \sum _ { i \in \mathscr { B } _ { t } } r _ { i } .\tag{37}
$$

Accuracy curves. Top-1 accuracy on corrupted samples and rectified logits are computed with ground-truth labels,

$$
\operatorname { A c c } _ { t } ^ { \mathrm { c o r r } } = \frac { 1 } { \left| { \cal B } _ { t } \right| } \sum _ { i \in { \cal B } _ { t } } \mathbb { I } \Bigl [ \arg \operatorname* { m a x } _ { c } z _ { i , c } ^ { \mathrm { c o r r } } = y _ { i } \Bigr ] , \quad \operatorname { A c c } _ { t } ^ { \mathrm { r e c t } } = \frac { 1 } { \left| { \cal B } _ { t } \right| } \sum _ { i \in { \cal B } _ { t } } \mathbb { I } \Bigl [ \arg \operatorname* { m a x } _ { c } \tilde { z } _ { i , c } = y _ { i } \Bigr ] ,\tag{38}
$$

where $\tilde { z } _ { i }$ is obtained by recomputing logits from the rectified feature after rescaling by the original feature norm and using the frozen source head. These accuracies are not required at test time; they serve only to validate the proposed mechanism in Fig. 4a.

## C.4 Component-wise alignment with the paired drift

Fig. 4(a) evaluates the full correction. To separate the two affine terms, we decompose $\hat { \Delta } _ { i } =$ $\hat { \Delta } _ { i } ^ { \mathrm { m e a n } } + \hat { \Delta } _ { i } ^ { \mathrm { s l o p e } }$ , with the routed mean $\begin{array} { r } { \hat { \Delta } _ { i } ^ { \mathrm { m e a n } } = \sum _ { q } w _ { i } ^ { ( q ) } \mu _ { a , t } ^ { ( q ) } } \end{array}$ and the routed slope $\hat { \Delta } _ { i } ^ { \mathrm { s l o p e } } =$ $\begin{array} { r } { \sum _ { q } w _ { i } ^ { ( q ) } R _ { t } ^ { ( q ) \top } ( u _ { i } - \mu _ { u , t } ^ { ( q ) } ) } \end{array}$ , along one online trajectory, so that all readouts share sample order, soft responsibilities, and ridge/EMA states. We report the cosine alignment with the oracle drift $\Delta a _ { i }$ and the PDR of Sec. 4.5 (Tab. 6). The main open-set protocol replays Source+TALOR on the benchmark stream of Sec. 5; the paired clean views of both csID and csOOD samples are used only for analysis. Each term aligns with the oracle drift and reduces it, and their sum performs best; the full correction has the highest cosine in all 90 cells. Splitting the open-set cells by regime, the slope correction is larger and better aligned for csOOD than for csID samples (correction norm 0.299 vs. 0.246, cosine 0.362 vs. 0.336, PDR 0.057 vs. 0.048), whereas the mean term behaves similarly in both regimes (cosine 0.337 vs. 0.325). This is consistent with the slope term contributing mainly to ID/OOD separation in Tab. 4.

Table 6: Component-wise alignment of TALOR’s correction with the paired oracle principal drift in the main open-set benchmark stream. Mean ± std over the 90 cells (6 csOOD datasets $\times \ 1 5$ corruptions).
<table><tr><td>Metric</td><td>Mean only</td><td>Slope only</td><td>Full</td></tr><tr><td>Cosine</td><td> $0 . 3 3 1 \pm 0 . 1 3 5$ </td><td> $0 . 3 4 9 \pm 0 . 0 5 0$ </td><td> $0 . 4 8 3 \pm 0 . 1 0 5$ </td></tr><tr><td>PDR</td><td> $0 . 0 5 1 \pm 0 . 0 7 6$ </td><td> $0 . 0 5 3 \pm 0 . 0 2 6$ </td><td> $0 . 1 1 3 \pm 0 . 0 8 2$ </td></tr></table>

## D Shared-Nuisance Condition and Drift Identification

This appendix proves Proposition 1 and states the scope of Assumption 1, reports empirical checks of the condition and of the deployed online correction, and discusses the mean term.

## D.1 Proof and scope

We use the notation of Sec. 4.2, with $\Sigma _ { \xi u } = \mathbb { E } _ { q } [ \xi _ { i } u _ { i } ^ { \circ } { } ^ { \top } ]$ and $\Sigma _ { u u } = \mathbb { E } _ { q } [ u _ { i } ^ { \circ } u _ { i } ^ { \circ } { } ^ { \top } ]$

Proof of Proposition 1. Let $\Sigma _ { a u } = \mathbb { E } _ { q } [ a _ { i } ^ { \circ } u _ { i } ^ { \circ } { } ^ { \top } ]$ . The objective is a convex quadratic in R with normal equations $( \Sigma _ { u u } + \lambda I _ { d } ) R = \Sigma _ { a u } ^ { \top }$ ; its minimum-norm minimizer is $R _ { q , \lambda } ^ { * } = ( \Sigma _ { u u } + \lambda I _ { d } ) ^ { \dagger } \Sigma _ { a u } ^ { \top }$ , which is the unique minimizer for $\lambda > 0$ . By Assumption 1,

$$
\Sigma _ { a u } = L _ { q } \Sigma _ { \xi u } + \mathbb { E } _ { q } [ s _ { i } { u _ { i } ^ { \circ } } ^ { \top } ] + \mathbb { E } _ { q } [ \varepsilon _ { i } ^ { a } { u _ { i } ^ { \circ } } ^ { \top } ] = L _ { q } \Sigma _ { \xi u } .
$$

To clarify the statistical meaning of the resulting prediction, consider the auxiliary population ridge regression of the latent nuisance $\xi _ { i }$ on the tail coordinates:

$$
\boldsymbol { C } _ { \boldsymbol { q } , \lambda } ^ { * } = \arg \operatorname* { m i n } _ { \boldsymbol { C } } \mathbb { E } _ { \boldsymbol { q } } \| \boldsymbol { \xi } _ { \boldsymbol { i } } - \boldsymbol { C } ^ { \top } \boldsymbol { u } _ { \boldsymbol { i } } ^ { \circ } \| _ { 2 } ^ { 2 } + \lambda \| \boldsymbol { C } \| _ { F } ^ { 2 } = ( \boldsymbol { \Sigma } _ { \boldsymbol { u } \boldsymbol { u } } + \lambda \boldsymbol { I } _ { \boldsymbol { d } } ) ^ { \dagger } \boldsymbol { \Sigma } _ { \boldsymbol { \xi } \boldsymbol { u } } ^ { \top } ,
$$

where the minimum-norm solution is taken at $\lambda = 0$ . Since $\Sigma _ { a u } = L _ { q } \Sigma _ { \xi u }$ , the two population ridge solutions satisfy

$$
R _ { q , \lambda } ^ { * } = C _ { q , \lambda } ^ { * } L _ { q } ^ { \top } .
$$

Consequently,

$$
\begin{array} { r } { R _ { \boldsymbol { q } , \lambda } ^ { * \top } \boldsymbol { u } _ { i } ^ { \circ } = L _ { \boldsymbol { q } } C _ { \boldsymbol { q } , \lambda } ^ { * \top } \boldsymbol { u } _ { i } ^ { \circ } = L _ { \boldsymbol { q } } \hat { \xi } _ { i , \lambda } , \qquad \hat { \xi } _ { i , \lambda } : = \Sigma _ { \boldsymbol { \xi } \boldsymbol { u } } ( \Sigma _ { \boldsymbol { u } \boldsymbol { u } } + \lambda I _ { d } ) ^ { \dagger } \boldsymbol { u } _ { i } ^ { \circ } . } \end{array}
$$

Thus, although the fitted regression targets $a _ { i } ^ { \circ } { \mathrm { . } }$ , its population prediction is exactly the linear image of a ridge estimate of the shared nuisance $\xi _ { i } ,$ , and therefore lies in col $( L _ { q } )$ . For any fixed matrix M, $\mathbb { E } _ { q } [ ( M ^ { \top } u _ { i } ^ { \circ } ) s _ { i } ^ { \top } ] = M ^ { \top } \mathbb { E } _ { q } [ u _ { i } ^ { \circ } s _ { i } ^ { \top } ] = 0 ;$ taking $M = R _ { q , \lambda } ^ { * }$ shows that the prediction is uncorrelated with $s _ { i } ,$ and the residual $a _ { i } ^ { \circ } - R _ { q , \lambda } ^ { * \top } u _ { i } ^ { \circ } = L _ { q } ( \xi _ { i } - \hat { \xi } _ { i , \lambda } ) + s _ { i } + \varepsilon _ { i } ^ { a }$ contains $s _ { i }$ unchanged. At $\lambda = 0 , \hat { \xi } _ { i , 0 } = \Sigma _ { \xi u } \Sigma _ { u u } ^ { \dag } u _ { i } ^ { \circ }$ is the minimum-MSE linear estimate of the zero-mean factor $\xi _ { i }$ from $u _ { i } ^ { \circ }$ , and by linearity $\boldsymbol { L _ { q } } \boldsymbol { \hat { \xi } } _ { i , 0 }$ is that of $L _ { q } \xi _ { i }$ . Finally, let $\varepsilon _ { i } ^ { u } = 0 , \Sigma _ { \xi } = \mathrm { C o v } _ { q } ( \xi _ { i } ) \succ 0$ , and let $T _ { q }$ have full column rank. Then $M _ { q } = \bar { T _ { q } } \Sigma _ { \xi } ^ { 1 / 2 }$ also has full column rank, and $u _ { i } ^ { \circ } = M _ { q } \Sigma _ { \xi } ^ { - 1 / 2 } \xi _ { i }$ $\Sigma _ { u u } = M _ { q } M _ { q } ^ { \top }$ , and $\Sigma _ { \xi u } = \Sigma _ { \xi } ^ { 1 / 2 } M _ { q } ^ { \top }$ . Using $M _ { q } ^ { \top } ( M _ { q } M _ { q } ^ { \top } ) ^ { \dagger } = M _ { q } ^ { + }$ and $M _ { q } ^ { + } M _ { q } = I$ , we obtain $\hat { \xi } _ { i , 0 } = \Sigma _ { \xi } ^ { 1 / 2 } M _ { q } ^ { + } M _ { q } \Sigma _ { \xi } ^ { - 1 / 2 } \xi _ { i } = \xi _ { i }$ □

Scope. Proposition 1 characterizes the population target of the slope term; TALOR estimates it from finite, soft-weighted batches, where ridge regularization and EMA smoothing stabilize the estimate. Assumption 1 idealizes the representation and is not implied by the SVD of the head, whose singular values only establish which directions are amplified. When part of the principal semantic variation is linearly correlated with the tail cue, i.e., $\mathbb { E } _ { q } \big [ { u } _ { i } ^ { \circ } s _ { i } ^ { \top } \big ] \ne 0$ , the slope term also removes that tail-predictable part. To assess departures from the idealized assumption, we examine the relative strength of the two cross-covariance terms:

$$
\eta _ { C } ^ { ( q ) } = \frac { \left. \mathbb { E } _ { q } [ ( s _ { i } + \varepsilon _ { i } ^ { a } ) u _ { i } ^ { \circ ^ { \top } } ] \right. _ { 2 } } { \left. \mathbb { E } _ { q } [ ( L _ { q } \xi _ { i } ) u _ { i } ^ { \circ ^ { \top } } ] \right. _ { 2 } } .\tag{39}
$$

Smaller values indicate stronger drift dominance in cross-covariance. App. D.2 evaluates this diagnostic alongside the drift reduction achieved by the deployed estimator; the mean term is analyzed separately in App. D.3.

## D.2 Empirical checks of the condition and of the online correction

Let $( a ^ { \mathrm { c } } , u ^ { \mathrm { c } } )$ and $( a ^ { \mathrm { t } } , u ^ { \mathrm { t } } )$ denote the clean and corrupted coordinates and $\delta a = a ^ { \mathrm { t } } - a ^ { \mathrm { c } }$ . Paired views identify the two principal components of the corrupted stream in Assumption 1: the corruptioninduced displacement δa plays the role of $L _ { q } \xi _ { i }$ and the clean principal coordinates $a ^ { \mathrm { c } }$ that of $s _ { i } + \varepsilon _ { i } ^ { a }$ while the regressor input is the observed corrupted tail $u ^ { \mathrm { t } }$ , exactly as online. Tab. 7 reports three checks per corruption.

(i) Drift predictability beyond high-dimensional fitting. We fit a pooled, centered ridge map from $u ^ { \mathrm { t } }$ to δa on the 5,000 pairs and report its in-sample $R ^ { 2 } , \operatorname { i . e . }$ , the fraction of centered drift energy that it explains. Because in-sample ridge fitting can explain part of an unrelated target through finite-sample correlations, we compare it with 200 within-corruption permutations of the rows of $\bar { \delta } \boldsymbol { a } .$ , which break the sample pairing while keeping both marginals. The real $R ^ { 2 }$ is 0.312 on average (0.248–0.382), whereas the permutation null is $\bar { 0 . 0 8 3 } \pm 0 . 0 \bar { 0 } 1$ for every corruption; none of the 3,000 permutations reaches the real value (z-scores 325–608). This regression-based test complements the CCA spectrum of Fig. 3c: the observed tail linearly predicts a substantial part of the principal drift.

(ii) Relative strength of the cross-covariance terms. The cross-covariance seen by the online regression splits exactly by prediction target, $\operatorname { C o v } ( a ^ { \mathrm { t } } , u ^ { \mathrm { t } } ) = \operatorname { C o v } ( \delta a , u ^ { \mathrm { t } } ) + \operatorname { C o v } ( a ^ { \mathrm { c } } , u ^ { \mathrm { t } } )$ , whose two terms are the empirical counterparts of the denominator and numerator of Eq. 39. Their spectral-norm ratio $\eta _ { C } = \| \mathrm { C o v } ( a ^ { \mathrm { c } } , u ^ { \mathrm { t } } ) \| _ { 2 } / \| \dot { \mathrm { C o v } } ( \delta a , u ^ { \mathrm { t } } ) \| _ { 2 }$ is below one for all 15 corruptions, with mean 0.234.

The margin is largest for global corruptions such as fog, frost, and contrast (0.08–0.13) and smallest for mild ones such as brightness, pixelate, and JPEG compression (0.38–0.54), indicating stronger drift dominance in cross-covariance for the former.

(iii) The deployed online estimator. We run the TALOR module itself (GMM routing, weighted ridge, and EMA; batch size 64; state reset per corruption) on the corrupted stream and measure the reduction of squared principal drift, $\begin{array} { r } { \mathrm { D R } = 1 - \sum _ { i } \| \delta a _ { i } - \hat { \Delta } _ { i } \| _ { 2 } ^ { 2 } / \sum _ { i } \| \delta a _ { i } \| _ { 2 } ^ { 2 } } \end{array}$ , for the full correction and for its slope and mean terms. We also report $\mathrm { D R } _ { \sigma }$ , the corresponding reduction for the full correction after mapping principal directional drift to logits with each sample’s corrupted feature norm held fixed, using weights $\rho _ { i } \sigma _ { j } \ ( \mathrm { E q . \ 7 } )$ . The full correction and the slope term reduce the drift for all 15 corruptions and the mean term for 14, and $\mathrm { D R } _ { \sigma }$ exceeds DR for every corruption (0.355 vs. 0.276 on average): the correction is most effective along the directions that the head amplifies.

Table 7: Per-corruption checks of the shared-nuisance condition and of the deployed online correction (ViT-B/16, ImageNet-C at severity 5, 5,000 pairs per corruption). $R ^ { 2 } \colon$ in-sample ridge fit of the centered drift from the corrupted tail (permutation null $0 . 0 8 3 \pm 0 . 0 0 1$ for every corruption). η<sub>C</sub>: clean-target to drift-target cross-covariance ratio (smaller values indicate stronger drift dominance in cross-covariance). DR: reduction of squared principal drift by the online TALOR correction and its terms; $\operatorname { D R } _ { \sigma } \colon$ logit-space DR of the full correction.
<table><tr><td></td><td colspan="2">Condition</td><td colspan="4">Online correction</td></tr><tr><td>Corruption</td><td> $R ^ { 2 }$ </td><td> $\eta _ { C }$ </td><td>DR (full)</td><td>DR (slope)</td><td>DR (mean)</td><td>DRσ</td></tr><tr><td>Gaussian noise</td><td>0.317</td><td>0.136</td><td>0.304</td><td>0.162</td><td>0.145</td><td>0.408</td></tr><tr><td>Shot noise</td><td>0.303</td><td>0.171</td><td>0.243</td><td>0.164</td><td>0.077</td><td>0.330</td></tr><tr><td>Impulse noise</td><td>0.314</td><td>0.159</td><td>0.265</td><td>0.163</td><td>0.099</td><td>0.361</td></tr><tr><td>Defocus blur</td><td>0.335</td><td>0.184</td><td>0.340</td><td>0.160</td><td>0.189</td><td>0.470</td></tr><tr><td>Glass blur</td><td>0.363</td><td>0.206</td><td>0.301</td><td>0.192</td><td>0.111</td><td>0.397</td></tr><tr><td>Motion blur</td><td>0.300</td><td>0.221</td><td>0.250</td><td>0.153</td><td>0.100</td><td>0.348</td></tr><tr><td>Zoom blur</td><td>0.324</td><td>0.215</td><td>0.270</td><td>0.164</td><td>0.109</td><td>0.335</td></tr><tr><td>Snow</td><td>0.288</td><td>0.331</td><td>0.236</td><td>0.128</td><td>0.111</td><td>0.271</td></tr><tr><td>Frost</td><td>0.382</td><td>0.106</td><td>0.318</td><td>0.210</td><td>0.110</td><td>0.405</td></tr><tr><td>Fog</td><td>0.375</td><td>0.075</td><td>0.534</td><td>0.085</td><td>0.451</td><td>0.656</td></tr><tr><td>Brightness</td><td>0.249</td><td>0.537</td><td>0.058</td><td>0.085</td><td>-0.025</td><td>0.085</td></tr><tr><td>Contrast</td><td>0.296</td><td>0.131</td><td>0.458</td><td>0.069</td><td>0.394</td><td>0.605</td></tr><tr><td>Elastic transform</td><td>0.334</td><td>0.269</td><td>0.280</td><td>0.152</td><td>0.131</td><td>0.305</td></tr><tr><td>Pixelate</td><td>0.248</td><td>0.397</td><td>0.118</td><td>0.101</td><td>0.018</td><td>0.154</td></tr><tr><td>JPEG compression</td><td>0.255</td><td>0.375</td><td>0.158</td><td>0.094</td><td>0.063</td><td>0.202</td></tr><tr><td>Mean</td><td>0.312</td><td>0.234</td><td>0.276</td><td>0.139</td><td>0.139</td><td>0.355</td></tr></table>

## D.3 Mean term safety

The mean term $\mu _ { a } ^ { ( q ) }$ is the affine intercept of Eq. 9. Because the slope is fitted on centered coordinates, it cannot represent a principal offset shared within a branch, so the weighted target mean supplies this intercept; an explicit source feature mean is not required for this role. The observed target mean can contain both a nuisance component (reflecting domain shift) and a semantic component (reflecting the class composition of the current stream), and TALOR subtracts its smoothed estimate without decomposing it. Two properties limit the risk of removing useful semantics.

First, consider the mean term in isolation at a fixed online state t. Define the mean-only corrected coordinates as

$$
a _ { i } ^ { \mathrm { { m e a n } } } : = a _ { i } - \hat { \Delta } _ { i } ^ { \mathrm { { m e a n } } } , \qquad \hat { \Delta } _ { i } ^ { \mathrm { { m e a n } } } : = \sum _ { q \in \mathcal { Q } } w _ { i } ^ { ( q ) } \mu _ { a , t } ^ { ( q ) } .
$$

For any two samples,

$$
a _ { i } ^ { \mathrm { m e a n } } - a _ { j } ^ { \mathrm { m e a n } } = ( a _ { i } - a _ { j } ) - \sum _ { q \in \mathcal { Q } } ( w _ { i } ^ { ( q ) } - w _ { j } ^ { ( q ) } ) \mu _ { a , t } ^ { ( q ) } .\tag{40}
$$

Thus, samples with identical routing weights receive the same translation, and their pairwise differences are preserved:

$$
a _ { i } ^ { \mathrm { m e a n } } - a _ { j } ^ { \mathrm { m e a n } } = a _ { i } - a _ { j } .
$$

This includes hard routing to the same branch. Within any group receiving the same translation, centered covariance is also unchanged. Under soft routing, differences in the weighted means can change pairwise differences.

Second, if the removed mean were predominantly discriminative, ID accuracy would drop; instead, Mean-only improves or preserves ID accuracy for all seven hosts in Tab. 4 (e.g., from 49.8% to 54.8% on Source). Without covariate shift, TALOR changes accuracy by only +0.12 pp on an unshifted clean stream and by at most −0.46 pp under pure label shift, which alters the target principal mean through the class prior alone (App. E). Hence the mean term corrects shift-induced bias without materially distorting the semantic geometry.

## E Negative Controls: Unshifted and Pure Label-Shift Streams

To verify that TALOR does not over-correct when no covariate shift is present, we conduct two negative-control experiments on ViT-B/16: an unshifted clean stream and a pure label-shift stream. In both settings, $p _ { t } ( x \mid y ) \approx p _ { s } ( x \mid y )$ , so there is no domain shift to remove; the relevant criterion is whether TALOR preserves the Source accuracy. Throughout this section, TALOR denotes standalone Source + TALOR; neither stream contains OOD samples, so we report ID accuracy only.

Unshifted clean stream. All methods use the same balanced stream of 5,000 clean ImageNet-val samples (5 per class), with no corruption or label-space change.

Table 8: Unshifted clean stream (ViT-B/16). Top-1 accuracy and change relative to Source (pp).
<table><tr><td>Metric</td><td>Source</td><td>TALOR</td><td>CoTTA</td><td>EATA</td><td>UniEnt</td><td>SAR</td><td>OSTTA</td><td>ViDA</td><td>STAMP</td><td>COME</td><td>DPCore</td></tr><tr><td>Top-1 Acc. (%)</td><td>84.48</td><td>84.60</td><td>84.52</td><td>84.60</td><td>84.84</td><td>84.56</td><td>84.54</td><td>84.22</td><td>84.72</td><td>84.16</td><td>84.62</td></tr><tr><td>∆ Acc (pp)</td><td>0.00</td><td>+0.12</td><td>+0.04</td><td>+0.12</td><td>+0.36</td><td>+0.08</td><td>+0.06</td><td>-0.26</td><td>+0.24</td><td>-0.32</td><td>+0.14</td></tr></table>

TALOR changes accuracy by only +0.12 pp, indicating near-identity behavior at the decision level on clean data.

Pure label shift. We keep $p _ { t } ( x \mid y )$ unchanged and alter only $p _ { t } ( y )$ using a long-tailed prior $q _ { r } \propto \mathrm { I F } ^ { - r / ( C - 1 ) }$ , where r is the class rank and $C = 1 0 0 0$ . Each stream contains 5,000 clean samples, with $\mathrm { I F } \in \{ 1 , 1 0 , 1 0 0 \}$ and a shared manifest. Source accuracies are 85.32%, 86.16%, and 84.86%, respectively.

Table 9: Pure label-shift streams (ViT-B/16). ∆ pp in accuracy relative to Source under varying imbalance factors.
<table><tr><td>IF</td><td>TALOR</td><td>CoTTA</td><td>EATA</td><td>UniEnt</td><td>SAR</td><td>OSTTA</td><td>COME</td></tr><tr><td>1</td><td>-0.06</td><td>+0.14</td><td>-0.14</td><td>-0.12</td><td>0.00</td><td>-0.08</td><td>+0.26</td></tr><tr><td>10</td><td>-0.16</td><td>0.00</td><td>-0.02</td><td>-0.12</td><td>-0.28</td><td>-0.24</td><td>-0.50</td></tr><tr><td>100</td><td>-0.46</td><td>-0.02</td><td>-0.04</td><td>-0.18</td><td>-0.08</td><td>+0.06</td><td>-0.48</td></tr><tr><td>Mean</td><td>-0.23</td><td>+0.04</td><td>-0.07</td><td>-0.14</td><td>-0.12</td><td>-0.09</td><td>-0.24</td></tr></table>

TALOR’s accuracy changes are −0.06, −0.16, and −0.46 pp for $\mathrm { I F } = 1 , 1 0 , 1 0 0 .$ , respectively, with a mean change of −0.23 pp. Even under the extreme $\mathrm { I F } = 1 0 0$ prior shift, the change remains below 0.5 pp, with no instability or accuracy collapse. This is comparable to established TTA methods such as COME (−0.24 pp), UniEnt (−0.14 pp), and SAR (−0.12 pp).

This experiment directly stresses the case where the target principal mean changes because of classprior variation while no covariate drift is present. TALOR nevertheless remains close to Source across all imbalance factors. Together with the consistent Mean-only gains under covariate shift (Tab. 4), these results show that the principal-mean term does not substantially erase normal semantic structure; its main benefit appears when a shared high-gain domain response is present.

## F Measuring ID/OOD Leading-Energy Slope Subspaces

Energy-truncated slope subspaces. Since each slope map is numerically full column rank with shape $4 1 5 \times 3 5 3$ , their full column spaces necessarily intersect in at least 291 dimensions, yielding the same number of trivial zero principal angles regardless of empirical alignment. To avoid this dimension-induced degeneracy, we compare their leading-energy subspaces, using a 90% Frobeniusenergy target subject to the rank cap $2 r \le D - k$ . For each regime, write the compact SVD as

$$
R _ { t } ^ { ( q ) } = U _ { q } \mathrm { d i a g } ( s _ { q , 1 } , \ldots , s _ { q , m _ { q } } ) V _ { q } ^ { \top } , \qquad s _ { q , 1 } \geq \cdot \cdot \cdot \geq s _ { q , m _ { q } } > 0 .\tag{41}
$$

Let $r _ { q }$ be the smallest rank retaining 90% of the slope-map Frobenius energy,

$$
r _ { q } = \operatorname* { m i n } \left\{ \ell : \frac { \sum _ { j = 1 } ^ { \ell } s _ { q , j } ^ { 2 } } { \sum _ { j = 1 } ^ { m _ { q } } s _ { q , j } ^ { 2 } } \geq 0 . 9 0 \right\} , \qquad r = \operatorname* { m i n } \left( \operatorname* { m a x } ( r _ { \mathrm { I D } } , r _ { \mathrm { O O D } } ) , \left\lfloor \frac { D - k } { 2 } \right\rfloor \right) .\tag{42}
$$

We compare the leading-energy tail subspaces $S _ { r } ^ { ( q ) } : = \operatorname { c o l } ( U _ { q } [ : , 1 : r ] ) \subset \mathbb { R } ^ { D - k }$ . The cap $2 r \le D - k$ excludes dimension-forced intersections; for the displayed cell (contrast, iNaturalist- $C ) , r = 3 8 $

Principal-angle calibration. With $U _ { q , r } = U _ { q } [ : , 1 : r ]$ , we compute

$$
U _ { \mathrm { I D } , r } ^ { \top } U _ { \mathrm { O O D } , r } = P \mathrm { d i a g } ( \sigma _ { 1 } , \ldots , \sigma _ { r } ) T ^ { \top } , \qquad \theta _ { j } = \operatorname { a r c c o s } ( \sigma _ { j } ) \in [ 0 , \frac { \pi } { 2 } ] .\tag{43}
$$

We evaluate the spectrum using cross-halves of the ID and OOD streams, and calibrate it against within-regime split-half spectra and pairs of independent Haar-random r-dimensional subspaces in $\mathbb { R } ^ { D - k }$ . The split-half references quantify finite-stream estimation noise, whereas the random reference quantifies the overlap expected for unrelated subspaces of the same dimension.

In Fig. 4b, the ID/OOD cross-half spectrum is well below the Haar-random reference and close to the ID split-half spectrum over the first ∼10 directions, and then departs toward the random reference. The remaining gap to the within-regime spectra shows that the two maps are not identical and is consistent with residual regime-specific structure. Accordingly, the diagnostic supports—but does not by itself prove—the routing interpretation in Tab. 4: a mixed rectifier can exploit the shared leading component, while ID/OOD routing can refine residual structure.

## G Result on ResNet-50 backbone

Table 10: Results on the ResNet-50 backbone and ImageNet-to-ImageNet-C in OSTTA with TALOR plugin across six covariate-shifted OOD datasets. All the results are averaged over 15 corruptions.
<table><tr><td rowspan="2">Method</td><td colspan="2">Places.-C Acc AUC</td><td rowspan="2">Textures-C iNatur.-C</td><td colspan="2" rowspan="2">SUN-C Acc AUC</td><td rowspan="2">SSB-H.C NINCO-</td><td colspan="2">Mean</td></tr><tr><td>Acc AUC</td><td>Acc AUC</td><td>Acc AUC Acc AUC</td><td>Acc AUC H-s</td></tr><tr><td>Source + TALOR Improve +4.2</td><td>17.7 64.7 21.9 74.0 +9.4</td><td>17.7 60.1 21.8 74.8 +4.2 +14.7</td><td>17.7 75.1 21.8 90.4 +4.1 +15.3</td><td>17.7 70.2 21.9 86.0 +4.3 3+15.8</td><td>17.7 54.0 21.9 58.1 +4.2 +4.0</td><td>17.7 60.6 21.8 67.5 +4.2 +6.9</td><td>17.7 64.1 25.9 21.9 75.1 32.0 +4.2 +11.0 +6.1</td></tr><tr><td>EATA +TALÓR Improve +0.4</td><td>32.2 76.2 32.6 79. .4 +3.2</td><td>29.9 53.7 30.8 64.8 +0.9 +11.2</td><td>13.5 63.5 27. 5.2 14.6 76.1 28. +1.1 +12.6 +0.7</td><td>82. .3 35.6 87. .6 36.1 +5.3 +0.5</td><td>56.0 32. 57.9 33.0 +1.9 +0.7</td><td>.4 63. .4 67.2 +3.9 +0.7</td><td>28.5 65 82 37.2 29.2 72. 39.0 +6.3 +1.8</td></tr><tr><td>SAR + TALÓR Improve +3.9</td><td>13.7 65.3 17.6 70.8 +5.5</td><td>8.2 36.3 11.6 56.7 +3.4 +20.4</td><td>13.4 62.4 17.0 86.3 +3.6 +23.9</td><td>14.2 66.2 17.7 85.9 +3.6 +19.7</td><td>11. .3 48.4 15.3 51.5 +4.0 +3.1</td><td>7.8 47.3 10.3 54.5</td><td>11.4 54.3 17.8 14.9 67.6 23.4</td></tr><tr><td>OSTTA + TALOR Improve</td><td>20.2 63.9 22.5 70.6</td><td>20.1 41.4 22.3 59.3 +2.2 +17.8</td><td>16.8 49.1 17.5 19.5 78.6 20.1</td><td>57.4 77.6 28.0</td><td>+2.5 26.1 51.0 22.1 53.8 24.5</td><td>+7.2 51.0 61.3 22. .8</td><td>+3.5 +13.3 +5.7 20.5 52.3 28.5 66.9 33.0</td></tr><tr><td>ViDA + TALÓR 22.3</td><td>+2.2 +6.7 18.0 63.9 74.3</td><td>17.9 57.6 22.2 73.9</td><td>+2.7 7+29.5 18.0 72.5 18.0 22.2 90.2 22.3</td><td>+2.6 +20.2 +2.0 67.4 18.0 85.9 22.3</td><td>+2.8 54.1 58.3</td><td>+2.4 +10.3 18.0 59.8 22.2 67.4</td><td>+2.3 3+14.6 +4.5 18.0 62.6 26.0</td></tr><tr><td>Improve STAMP +TALOR 28.5 70.2 Improve +0.2 +0.0</td><td>+4.3 +10.4 28.3 70.2</td><td>+4.3 +16.3 27.3 63.1 27.7 70.7 +0.4 +7.6 +0.8 +0.5</td><td>+4.2 +17.7 26.5 81.9 27.1 27.4 82.5 27.8 +0.7 +0.3</td><td>+4.3 3+18.5 78.6 78.9</td><td>+4.2 +4.2 29.2 59.0 29.5 57.2 +0.3 -1.8 +0.3</td><td>+4.2 +7.6 29.0 62.4 29.2 65.7</td><td>22.2 75.0 32.4 +4.3 3+12.4 +6.4 27.9 69.2 37.9 28.4 70.9 38.7</td></tr></table>

Table 11: Results (%) for ImageNet-to-ImageNet-C in csID:csOOD=8:2 ratio with the TALOR plug-in across six covariate-shifted OOD datasets. Improvement and degradation compared to the baseline are highlighted in green or red respectively. All the results are averaged over 15 corruptions.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Places.-C Acc AUC</td><td rowspan="2">Textures-C Acc AU Acc AU</td><td rowspan="2">SUN-C Acc AUC</td><td rowspan="2">SSB-H.C Acc A</td><td rowspan="2">NINC Acc AU Acc</td></tr><tr><td>iNatur.-C</td></tr><tr><td>Source 49.8 TALOR 55. .9 + Improve +6.1</td><td>49 88 49 88 75. .6 55 55: +8.8 +6.0 +10</td><td>78. 9 49. 2.8 90. 7 55.9 +6.1 +11 .8 +6.1 +16.6</td><td>4 49 8∞ 9 49 88 64. .8 88.0 55. 64. .6 55. 74 1 +6. +8.7 +6. +9 .8</td><td>49. 88 68.0 56. 3.5 55: 79.1 64. +6.1 +11.0 +8.2</td></tr><tr><td>CoTTA 49. .9 65. TALOR 56.0 + Improve +6.1 +9. .9</td><td>95.8 55. 5.8 68 3.2 8 +12</td><td>78. 3 55: .8 90. .2 56.0 +11. 1.8 +6.0</td><td>69 .9 49 89 56. 3 49 6 88 3 55. 64.0 55 9 +18.4 +6. +7.7 +6. 5.3</td><td>64. 49 67. .2 56. 74.7 55.9 79.0 64.5 +10.1 +6.1 +11.8 +8.4</td></tr><tr><td>EATA 59. .2 65. .0 TALÓR 60.0 74.1 + Improve +0.8 +9.1</td><td>56. 69 82 57 83 +0.</td><td>55 85 73. .5 89 ×</td><td>.9 8.8 68. 59 4.3 6 58. .6 60. 67 +0.7 +17.7 +0. 9 +6.0</td><td>.9 66 23 538 67 4 61 3 .8 75 79.2 66. 4 +0 .9 +9 +0. +11.8 +5</td></tr><tr><td>SAR 58.1 64. + TALOR Improve +1.6 +12.9</td><td>.5 55 5 77.4 57 +1.9 +32.</td><td>5.5 55 85 79 32 55 57. 93 57. + -1.6 +13. .9 +1.9</td><td>.6 67 58. .4 64. 64 .6 90.6 59.8</td><td>58 2 70. .5 56. 85 67. .0 60. 59. .8 78. .4 58 82.5 67.9</td></tr><tr><td>OSTTA 58.5 64.6 + TALOR 60.0 77.4 Improve +1.4 +12.9</td><td>58. .0 62 59.4 8 +25</td><td>1 .9 77 .5 58.2 93.8 59.6</td><td>+23.4 +1.4 +2.9 68.1 58. .8 63. 91.1 60.2</td><td>+1.6 +7. .8 +1.7 +15.5 +7.1 58. .5 68. .7 58. .3 67. .4 62. 60.0 77 .8 59.7 82: 5 68.8</td></tr><tr><td>ViDA 50.5 66. .5 TALÓR 56.4 75. 83 + +5.9 +9</td><td>+1.4 3013 50. 70. 56.</td><td>+1.3 +16.3 +1.4 50. 79 50.4 .5 3 91.0 56.3</td><td>+23.0 +1.4 +1.5 50.5 56. 50. 0.5 71. .4 88.3 56.3 65.0 56.2</td><td>+9 +1.4 +15.0 +6. 95 50. 68. 25 55.0 .2 .5 56.3 .0 .0 79</td></tr><tr><td>Improve UniEnt 50.5 59. 73 .9 TALOR 79 1 + 6.2</td><td>+11 58. 4.4 80 4 .9</td><td>+5.8 +12.0 +5.8 59 88. 3.2 50.0. .4 59 81 60 4</td><td>+17.2 +5.7 +8.5 +5 59. 3 62 4.4 59 2 60. .6 60.4 1</td><td>.8 +9 .9 +5.8 +11.3 +7.9 .4 79 9.2 59 1 76. 7.2 66 60.1 83. 69</td></tr><tr><td>Improve +1.1 +5 STAMP 55 .7 69. .9 +TALOR 4</td><td>+0. .9 +6.5 50.3 72. 84 83</td><td>+ +6. 9 +1.1 +8.6 55 83. .6 1 75 5 5 9 2 2 .9 89.0</td><td>+1.3 +5 .0 + .5 55 .6 59 2.2 55 .6 65 5 4</td><td>+6.4 +1.1 +6.4 +3. 68 28 59. .0 75 64.7</td></tr><tr><td>Improve +3. .8 +7.6 COME 58 75. .5 TALOR 7 80.7 +</td><td>+3.6 +10.5 58 .0 77 .7</td><td>+3. +8. .6 +3.7 +13.4 1 85. .8 59 3 82. 7 94. 6 59. .9 92.3</td><td>48 +3. +6.0 +3 .0 58 6.3 59 .2</td><td>.8 + +9.0 +5.7 6 58. .6 72 55 58 .4 75. .4 65</td></tr><tr><td>Improve +0.8 +5.1 +0.7 58.8 76.5 59.8 DPCore + TALOR 80.5 60.1</td><td>+9.5 +0.6 80.9 56.3 87.1 56.6</td><td>+8.9 +0.6 +10.3 +0. .8 93.9 83.3 60.5 1 60.6</td><td>59 4 +7.9 +0.8 58. 60. 62. 2.6 .2 60.3 68.2 58.6</td><td>3 78. 1 83. 4 68. 7 +6.0 +0.7 +8.0 +3.4 75.53 58. .9 79.4 67.1 59.2 84.1 69.0 +0.3 +4.7</td></tr></table>

## H Results under Different OOD Ratios

## I Computational Complexity

## J Limitations and Broader Impacts

Limitations. TALOR is built upon the geometry induced by a frozen linear classifier head, which provides a simple, task-aligned anchor for test-time feature rectification. Our extensive diagnostics and ablations show that, across different backbones, OOD ratios, open-set streams, closed-set settings, and dynamic continual protocols, harmful target-stream shifts often contain a removable component in the classifier-amplified principal coordinates, while tail coordinates provide useful cues for estimating this component. Extending the head-induced rectification principle to richer prediction tasks, multimodal models, and alternative classifier parameterizations is an interesting direction for future work. Since TALOR is backward-free, source-free, and modular, such extensions may be achieved by constructing task-aligned coordinate systems and lightweight online estimators without changing the underlying model architecture.

Broader impacts. TALOR offers an innovative and lightweight route toward robust test-time adaptation. By using the frozen source head as a geometric anchor, TALOR performs feature rectification without backpropagation, costly augmentation loops, or large mutable memories. This makes it suitable for low-latency and resource-constrained deployment scenarios, such as edge recognition systems, real-time monitoring, and continually changing visual environments. In openset streams, TALOR avoids directly updating model parameters from unlabeled target samples, reducing the risk that unknown samples contaminate the source decision boundary. More broadly, its plug-and-play design can make robust adaptation more accessible across deployed recognition systems, especially when computational budgets, privacy constraints, or deployment stability make conventional online optimization undesirable. As with other robustness methods, TALOR should be used together with standard monitoring and calibration practices when deployed in high-stakes applications.

Table 12: Results (%) for ImageNet-to-ImageNet-C in csID:csOOD=3:7 ratio with the TALOR plug-in across six covariate-shifted OOD datasets. Improvement and degradation compared to the baseline are highlighted in green or red respectively. All the results are averaged over 15 corruptions.
<table><tr><td rowspan="2">Method</td><td>Places.-C</td><td rowspan="2">Textures-C Acc AUC</td><td rowspan="2">iNatur.-C Acc AUC</td><td rowspan="2">SUN-C SSB-H.C Acc AUC Acc AUC</td><td rowspan="2">NINCO-C Acc AUC</td><td rowspan="2">Mean Acc AUC H-s</td></tr><tr><td>Acc AUC</td></tr><tr><td>Source 49.5 67.0 49.5 70.7 + TALOR 55.6 81.9 55.3 87.6 Improve +6.1 +14.9 +5.8 +17.0</td><td></td><td>49.5 78.7 49.5 55.4 94.2 55.5 92.8 +5.9 +15.5 +6.0 +21.3</td><td>71.5 49.5 56.2 55.4 73.2 +5.9 +17.0</td><td>49.5 64.6 49.5 68.1 56.2 55.5 81.7 55.5 85.3 66.4 +6.0 +17.1 +6.0 +17.1 +10.2</td></tr><tr><td>CoTTA 49.4 63.2 + TALOR 55.4 81.1 Improve +6.1 +18.0 +6.4 +21.6</td><td>48.7 65.8 55.1 87.4</td><td>49.075.6 49.2 66.0 55.2 93.5 55.5 92.8 +6.2 +17.9 +6.2 +26.8</td><td>49.554.7 55.4 71.9 +5.9 +17.2 +6.3 +18.8</td><td>48.9 61.8 49.1 64.5 54.5 55.2 80.6 55.3 84.5 66.0 +6.2 +20.0 +11.5</td></tr><tr><td>EATA 50.2 58.9 + TALOR 50.7 74.1 Improve +0.5 +15.2</td><td>22.6 56.8 22.9 63.9 +0.4 +7.1</td><td>37.5 63.1 38.2 79.8 +0.7 +16.7</td><td>45.5 65.6 40.7 55.1 46.3 81.1 41.2 67.0 +0.8 +15.4 +0.6 +11.9</td><td>48.5 61.5 40.8 60.2 44.8 49.2 76.0 41.4 73.6 50.0</td></tr><tr><td>SAR 48.6 56.0 + TALOR 52.7 80.6 Improve +4.1 +24.7</td><td>43.6 49.6 50.5 87.6 +6.9 +37.9</td><td>43.2 49.9 47.2 88.7</td><td>41.8 44.1 55.1 59.7 46.4 90.5 56.8 74.6</td><td>+0.7 +14.4 +0.6+13.5 +5.1 50.2 62.3 47.1.53.6 48.2 53.2 81.6 51.1 83.9 62.3</td></tr><tr><td>OSTTA 55.9 + TALOR R57.2 Improve +1.3 +23.5 +1.5 +37.8</td><td>58.2 52.4 52.2 81.8 53.8 90.0</td><td>+4.0 +38.8 45.6 60.3 47.0 91.0</td><td>+4.7 +46.5 +1.7 +14.8 53.4 57.2 57.6 60.3 54.5 92.5 59.0 76.4</td><td>+2.9 +19.3 +4.1 +30.3 +14.1 55.5 65.3 53.4 58.9 55.5 56.8 83.3 54.7 85.8 66.2</td></tr><tr><td>ViDA 49.8 65.2 + TALOR 55.8 81.9</td><td>49.7 68.3 55.6 87.7</td><td>+1.4 +30.8 +1.1 +35.4 49.676.6 49.7 68.4 55.7 94.2 55.7 92.8</td><td>+1.4+16.1 50.1 56.1 55.8 73.6 55.9 81.9</td><td>+1.2 +18.0 +1.3 +26.9 +10.7 49.8 64.0 49.8 66.5 55.9 55.8 85.3 66.6</td></tr><tr><td>Improve UniEnt 56.6 71.2 + TALOR 57.2 80.8</td><td>+6.1 +16.7 +5.9 +19.3 51.0 76.1 51.9 85.1</td><td>+6.1 +17.6 54.1 86.7 41.6 54.8 93.0</td><td>+6.1 +24.4 +5.8 +17.5 76.1 58.2 61.4 42.2 80.5 59.1 75.2</td><td>+6.1 +17.9 +6.0 +18.9 +10.8 57.6 69.9 53.2 73.6 59.9</td></tr><tr><td>Improve +0.7 STAMP 51.6 70.3 51.4 76.1 + TALOR 54.4 83.9 53.8 91.0</td><td>+9.6 +0.8 +9.0 54.0 96.4</td><td>+0.7 +6.2 +0.5 +4.4 51.5 84.8 51.5 77.1 54.1 94.5</td><td>+1.0+13.8 51.3 59.7 53.8 73.9 53.8 83.5</td><td>58.5 83.9 53.9 83.1 64.1 +0.9 +14.0 +0.8 +9.5 +4.2 51.4 69.2 51.5 72.9 59.5</td></tr><tr><td>Improve +2.7 +13.6 COME 12.9 49.5 + TALOR 13.2 60.1 54.6 91.2</td><td>+2.4 +14.9 53.6 77.4</td><td>+2.4 +11.6 +2.6 +17.4 54.1 83.8 54.4 80.7 54.8 95.2 55.4 94.1</td><td>+2.5 +14.2 53.6 56.9 54.4 75.9</td><td>54.0 87.2 65.9 +2.4 +14.3 +2.5 +14.3 +6.5 51.8 70.3 46.7 69.8 53.5 52.7 82.3 47.5 83.1 58.1</td></tr><tr><td colspan="2">Improve +0.2 +10.7 +0.9 +13.7 +0.7 +11.4 DPCore 55.9 75.7 53.9 77.3 46.6 + TALOR 56.0 83.7 54.4 88.8</td><td colspan="2">+1.1 +13.4 +0.8 +19.0 81.5 51.3 81.3 56.2 64.8 47.2 89.9 51.6 90.9 56.5 78.6 +0.5 +8.4 +0.3 +9.6 +0.3 +13.8</td><td>+1.0 +12.0 +0.8 +13.4 +4.7 50.0 73.5 52.3 75.7 61.4 50.5 81.7 52.7 85.6 64.8 +0.5 +8.2 +0.4 +9.9 +3.4</td></tr></table>

Table 13: Comparisons w.r.t. computation complexity. Accuracy, AUC, and H-score are reported in %. TALOR adds negligible computational cost. The wall-clock time (Min) and memory usage (MiB) are measured for processing 15 consecutive corruptions, where each domain is constructed by mixing 5,000 csID and 5,000 csOOD images, batch size = 64; all Time/Memory values are kept as integers. Mem↑ (%) denotes the relative GPU-memory increment after adding TALOR, i.e., $\begin{array} { r } { \frac { \mathsf { M e m o r y } _ { + T A L O R } - \mathsf { M e m o r y } _ { O r i g i n } } { \mathsf { M e m o r y } _ { O r i g i n } } \times 1 0 0 \% } \end{array}$ . All experiments are conducted on an NVIDIA Quadro P6000.
<table><tr><td rowspan="2">Method</td><td colspan="4">Origin</td><td colspan="6">+TALOR</td><td colspan="4">∆ (+TALOR − Origin)</td></tr><tr><td>Acc AUC</td><td>H-s</td><td>|Time</td><td>Mem</td><td></td><td>|Acc AUC</td><td>H-s</td><td>Time</td><td>Mem</td><td></td><td>Acc AUC</td><td>H-s</td><td></td><td>|Time Mem(%)</td></tr><tr><td>Source</td><td>49.8 68.0</td><td>56.4</td><td>29</td><td>1,565</td><td>55.9</td><td>983.5</td><td>66.1</td><td>30</td><td>1,717</td><td></td><td>+6.1 +15.5</td><td>+9.7</td><td>+1</td><td>+10</td></tr><tr><td>CoTTA</td><td>49.6 64.6</td><td>54.9</td><td>212</td><td>21,064</td><td></td><td>55.682.7</td><td>65.6</td><td>214</td><td>21,197</td><td></td><td>+6.0 +18.2 +10.8</td><td></td><td>+2</td><td>+1</td></tr><tr><td>EATA</td><td>45.5 62.6 48.2</td><td></td><td>42</td><td>7,123</td><td></td><td>46.073.9</td><td>53.4</td><td>43</td><td>7,255</td><td></td><td>+0.5 +11.3 +5.2</td><td></td><td>+1</td><td>+2</td></tr><tr><td>SAR</td><td>50.2 55.2</td><td>51.1</td><td>114</td><td>7,073</td><td></td><td>54.4 84.3</td><td>65.4</td><td>115</td><td>7,353</td><td></td><td>+4.2 +29.0 +14.4</td><td></td><td>+1</td><td>+4</td></tr><tr><td>OSTTA</td><td>56.1 61.8</td><td>58.4</td><td>101</td><td>12,891</td><td></td><td>57.4 85.3</td><td>68.1</td><td>103</td><td>13,209</td><td></td><td>+1.2 +23.5 +9.6</td><td></td><td>+2</td><td>+2</td></tr><tr><td>ViDA</td><td>50.4 66.7 56.4</td><td></td><td>469</td><td>12,089</td><td>56.4</td><td>83.7</td><td>66.6</td><td>465</td><td>12,253</td><td></td><td>+6.0 +17.1 +10.1</td><td></td><td>-4</td><td>+1</td></tr><tr><td>UniEnt</td><td>56.7 76.7 63.9</td><td></td><td>74</td><td>14,217</td><td>57.5</td><td>84.1</td><td>67.3</td><td>75</td><td>14,349</td><td></td><td>+0.8 +7.4</td><td>+3.4</td><td>+1</td><td>+1</td></tr><tr><td>STAMP</td><td>51.974.2</td><td>60.2</td><td>181</td><td>11,307</td><td>54.3</td><td>86.0</td><td>65.9</td><td>182</td><td>11,165</td><td></td><td>+2.4 +11.9</td><td>+5.7</td><td>+1</td><td>-1</td></tr><tr><td>COME</td><td>57.074.6</td><td>63.7</td><td>42</td><td>7,123</td><td>57.7</td><td>86.5</td><td>68.4</td><td>42</td><td>7,257</td><td></td><td>+0.7 +11.9</td><td>+4.7</td><td>0</td><td>+2</td></tr><tr><td>DPCore</td><td>54.8 76.6 63.5</td><td></td><td>116</td><td>11,007</td><td></td><td></td><td>55.1 85.6 66.6</td><td>115</td><td>11,141</td><td></td><td>+0.3 +9.0</td><td>+3.1</td><td>-1</td><td>+1</td></tr></table>

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction state the scope of TALOR as a backward-free feature-rectification plug-in for open-set test-time adaptation. The main claims are supported by the method description, main OSTTA results, mechanism validation experiment, closedset evaluations, ablations, and computational analysis.

Guidelines:

• The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.

• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.

• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Limitations are discussed in App. J.

Guidelines:

• The answer [N/A] means that the paper has no limitation while the answer [No] means that the paper has limitations, but those are not discussed in the paper.

• The authors are encouraged to create a separate “Limitations” section in their paper.

• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

## Answer: [Yes]

Proposition 1 is formally stated under Assumption 1 in Sec. 4.2, with its complete proof provided in App. D.1. The result characterizes the population tail-conditioned ridge estimator, and its scope and limitations are explicitly discussed in App. D.

## Guidelines:

• The answer [N/A] means that the paper does not include theoretical results.

• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

• All assumptions should be clearly stated or referenced in the statement of any theorems.

• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

• Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: The paper specifies the TALOR algorithm, evaluation protocol, benchmarks, evaluation metrics, batch size, OOD scoring rule, and fixed hyperparameters used for the main experiments. Additional backbone, OOD-ratio, and computational details are provided in the appendices.

## Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• If the paper includes experiments, a [No] answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: The code will be released upon acceptance.

Guidelines:

• The answer [N/A] means that paper does not include experiments requiring code.

• Please see the NeurIPS code and data submission guidelines (https://neurips.cc/ public/guides/CodeSubmissionPolicy) for more details.

• While we encourage the release of code and data, we understand that this might not be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //neurips.cc/public/guides/CodeSubmissionPolicy) for more details.

• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [Yes]

Justification: Already illustrate in Experimental result reproducibility part.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

• The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No]

Justification: The paper reports averages over corruption types and multiple datasets. We focus on consistent improvements across datasets, baselines, ablations, and computational comparisons.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The authors should answer [Yes] if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)

• The assumptions made should be given (e.g., Normally distributed errors).

• It should be clear whether the error bar is the standard deviation or the standard error of the mean.

• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g., negative error rates).

• If error bars are reported in tables or plots, the authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: App. I reports wall-clock time and GPU memory usage for the main baselines with and without TALOR. The experiments are conducted on a single NVIDIA Quadro P6000, and the table specifies the processed stream size and batch size.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: The research uses public benchmarks and does not involve human-subject experiments, private data. We have reviewed the NeurIPS Code of Ethics and do not identify any deviation from it.

Guidelines:

• The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.

• If the authors answer [No], they should explain the special circumstances that require a deviation from the Code of Ethics.

• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: Broader impacts are discussed in App. J.

## Guidelines:

• The answer [N/A] means that there is no societal impact of the work performed.

• If the authors answer [N/A] or [No], they should explain why their work has no societal impact or why the paper does not address societal impact.

• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

Answer: [N/A]

Justification: The work proposes a test-time feature-rectification algorithm evaluated on existing public benchmarks.

Guidelines:

• The answer [N/A] means that the paper poses no such risks.

• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [No]

Justification: Licenses will be included with the code release.

Guidelines:

• The answer [N/A] means that the paper does not use existing assets.

• The authors should cite the original paper that produced the code package or dataset.

• The authors should state which version of the asset is used and, if possible, include a URL.

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.

• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

• If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [N/A]

Justification: The paper does not introduce a new dataset, benchmark, pretrained model, or other released asset.

Guidelines:

• The answer [N/A] means that the paper does not release new assets.

• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

• The paper should discuss whether and how consent was obtained from people whose asset is used.

• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [N/A]

Justification: The work does not involve crowdsourcing.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A]

Justification: The work does not involve human-subject research.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

Answer: [N/A]

Justification: The core method development, experiments, and scientific claims do not rely on LLMs as an important, original, or non-standard component. Any writing, editing, or formatting assistance does not affect the methodology, scientific rigor, or originality of the research.

## Guidelines:

• The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.

• Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.