# ROBUST EXPLANATORY ANALYSIS FOR STRUCTURED OPTIMIZATION NETWORKS(REASON) IN LEARNING

Vahid Jebraeeli<sup>1</sup>, Hamid Krim<sup>1</sup>, Keisuke Fujii<sup>2</sup>

<sup>1</sup>ECE Department, North Carolina State University, Raleigh, NC, USA <sup>2</sup>Graduate School of Informatics, Nagoya University, Nagoya, Japan vjebrae@ncsu.edu ahk@ncsu.edu fujii@i.nagoya-u.ac.jp

## ABSTRACT

We seek a hierarchical resolution of residual error under a composite learning objective as a principled mechanism for identifying and exploiting additional information, particularly under distribution shift and perturbation. Inspired by the System 1/System 2 paradigm, we formulate this principle as a two-loop mechanism serving a common inference goal. A fast, System 1-like loop resolves the residual addressable by the available evidence; the remaining epistemic residual triggers a medium speed, System 2-like loop that actively acquires additional evidence and measures its informational value through residual reduction. Over nested solution spaces, the mediumloop optimum is provably no worse than the fast-loop optimum. Experiments show that the resulting two-loop model consistently improves upon the fast loop at approximately half the computation of the medium loop, including under unseen corruptions, and, when provided a cleaner observation, outperforms staged classifiers and learning-to-defer methods at equal computational cost.

Index Terms: Two-loop model, selective prediction, active perception, adaptive computation, epistemic uncertainty.

## 1. INTRODUCTION

While machine learning has achieved remarkable success, much of this progress has come at the expense of increasing model and sample complexity while still falling short on generalization. Transformers have partially advanced beyond this paradigm by explicitly modeling cross-contextual coherence through attention[1], thereby enabling a modest but important form of relational reasoning. Yet this reasoning remains largely associative and compositional, rather than reflecting a deeper structural organization of memory, inference, and adaptive computation. The notion of reasoning has long been regarded as an inseparable component of intelligent learning, yet its principled and coherent integration into learning systems has remained elusive. A major conceptual breakthrough emerged from the work of Kahneman and Tversky[2], and the subsequent fast–slow, or System 1/System 2, view of cognition, which provided a simple yet powerful framework for rationalizing how learning and decisionmaking may operate across distinct computational regimes: a fast, intuitive, and largely automatic process, complemented by a slower, deliberative, and reasoning-intensive process. Complementarily, the consciousness-prior perspective [3] suggests that such higher-level processing may operate on a selectively filtered subset of the available representation, with attention providing a natural mechanism for funneling task-relevant information toward more deliberate reasoning. In a different but related way, we propose learning as a hierarchy of loops that resolve residual uncertainty over progressively richer solution spaces. The Fast Loop (FL) resolves errors addressable by the current amortized inference map, while persistent residuals activate a

Medium Speed Loop (MSL) that explores alternative reasoning trajectories without altering the underlying representation. Failure of this expanded search to sufficiently reduce the residual is then interpreted as evidence of a representational limitation, ideally triggering a creative loop that operates over an architectural state space to adapt the hypothesis class itself, a third level we defer to future work. Our proposed approach shares with selective classification, failure detection, learning-to-defer, and adaptive-inference methods, the recognition at a high level that computational effort should be non-uniform across samples. Classical reject-option and selective-prediction methods determine whether an inference should be accepted or withheld [4, 5], while failure-detection and calibration methods seek reliable indicators of when a prediction is likely to be erroneous or out of distribution [6, 7, 8]. Learning-to-defer methods extend this principle by routing uncertain decisions to a downstream expert or model [9, 10, 11], whereas staged and early-exit architectures adapt computational cost by allowing easy examples to terminate early and difficult examples to invoke additional processing [12, 13, 14]. Our formulation views failure beyond the criterion for rejection, deferral, or additional computation, but as a structured residual which determines the type of computation subsequently activated. Thus, rather than implementing a confidence-based sequence of predictors of increasing complexity, the proposed hierarchy progressively expands the admissible modes of inference, reasoning, and representation according to the nature of the unresolved residual.

To sum up and in contrast to fore-noted existing work hinging on pre-selected gating mechanism as well as to the Bayesian interpretation of System 1/System 2 in [15], the central objective of this paper is the realization of a data-driven learning system inspired by the latter paradigm and built around a two-loop learning architecture (Fast/Medium) resulting from a hierarchical optimization formulation of a learning functional objective. Our implementation also refers inputs by a calibrated confidence threshold; what differs from prior gating is the action it triggers, the acquisition of new evidence.

Two key questions are addressed in this contribution. First, can decision-making be improved by selectively acquiring additional evidence to reduce the residual error, rather than through a uniformly optimized decision rule of comparable model complexity governed by a preselected threshold gate? This directly examines the learning value of the System 1/System 2 paradigm, wherein additional computational and informational resources are invoked only when the fast-loop inference leaves a significant unresolved residual, thereby allocating learning and reasoning resources where they are most needed.

Second, what is the value of the MSL operating in REVEAL mode, where additional evidence is actively sought to resolve uncertainty remaining after fast-loop inference? The utility of the acquired evidence is measured by its informational novelty relative to the current representation and, critically, by its reduction of the epistemic residual $R _ { \mathrm { e p i } }$ . It is thus assessed not by the mere acquisition of additional evidence, but by its ability to acquire informative evidence that measurably resolves epistemic uncertainty. Extensive experiments are conducted across four fine-grained recognition benchmarks: Stanford Dogs, CUB-200-2011, FGVC-Aircraft, and Oxford-IIIT Pets.

Here, we shift focus from efficient learning with fixed data availability[16, 17] to inference, where computational and informational resources are rather adaptively allocated on a per-input basis. The remainder of the paper formulates the problem and develops the proposed solution, presents the experimental results in Section 4, and concludes in Section 5.

## 2. PROBLEM STATEMENT AND SOLUTION

## 2.1. Residual decomposition over nested regimes

We formulate learning as the hierarchical resolution of residual uncertainty over progressively richer solution spaces. A FL, instantiated as a conventional CNN, is trained to minimize the inference residual $R _ { \mathrm { i n f } }$ associated with the current inference map. Persistent residual uncertainty $R _ { \mathrm { e p i } }$ activates a MSL, which, upon request from the FL, explores alternative decisions using additional evidence. As noted earlier, the natural triggering of a creative loop (CL) upon failure of these two procedures is deferred to future work. Accordingly, the associated architecture state space may be defined as

$$
\Omega = \mathcal { C } \times \mathcal { L } \times \mathcal { P } ,
$$

where $\mathcal { C } , \mathcal { L } ,$ and $\mathcal { P }$ respectively parameterize the search over channel capacity, network depth, and interaction complexity/order. A complete state is defined as $X = ( z , \theta , \psi )$ , with z the latent inference state of an observation, θ the fast predictor which consults <sub>ψ</sub> the predictive model as the medium regime. The residual of a decision decomposes approximately into a part the current inference map can already resolve and a part left by insufficient evidence,

$$
R \approx R _ { \mathrm { i n f } } + R _ { \mathrm { e p i } } ,\tag{1}
$$

and the two regimes target the two parts: the FL $\mathcal { T } _ { f }$ reduces $R _ { \mathrm { i n f } }$ by amortized inference, and the MSL $\mathcal { T } _ { m }$ acts only on what it leaves. A complete sweep is therefore a composition,

$$
X ^ { ( k + 1 ) } = \mathcal { T } _ { \mathfrak { m } } \circ \mathcal { T } _ { f } \bigl ( X ^ { ( k ) } \bigr ) , \qquad \mathcal { T } _ { \mathfrak { m } } \mathrm { a c t i v e \ i f f } R _ { \mathrm { e p i } } > \varepsilon _ { m } ,\tag{2}
$$

in which the medium speed loop is bypassed and the fast solution retained wherever the epistemic residual is already small. The organization is a nonlinear block-coordinate relaxation: distinct mechanisms address distinct components of one residual, and the slower one is invoked only for what the faster one cannot resolve.

Both regimes descend a single functional,

$$
\begin{array} { r } { \mathcal { J } ( X ) = \mathcal { L } _ { \mathrm { t a s k } } + \lambda _ { p } \mathcal { L } _ { \mathrm { p r e d } } , } \end{array}\tag{3}
$$

encoding task performance and predictive consistency, over hierarchically nested feasible sets $S _ { F } \subset S _ { M } \colon$ in $S _ { F }$ the latent state is what the fast map returns, while $S _ { M }$ also admits a state revised on further evidence. Nesting alone then gives the following.

Proposition 1. Consider a two-loop learning system consisting of a fast loop F and a medium speed loop M, operating over hierarchically nested admissible solution spaces $S _ { F } \subseteq S _ { M }$ . Let X denote an admissible learning state, including the latent representation and the parameters active at the corresponding loop, and let both loops be governed by the common objective

$$
{ \mathcal { J } } ( X ) = { \mathcal { L } } _ { \mathrm { t a s k } } ( X ) + \lambda _ { p } { \mathcal { L } } _ { \mathrm { p r e d } } ( X ) , \qquad \lambda _ { p } \geq 0 .
$$

Assume that a minimizer of $\mathcal { J }$ exists over each admissible space. Define $X _ { F } ^ { \star } = \arg \operatorname* { m i n } _ { X \in S _ { F } } \bar { \mathcal { J } } ( X )$ ， $X _ { M } ^ { \star } = \arg \operatorname* { m i n } _ { X \in S _ { M } } \mathcal { J } ( X )$ Then

$$
\begin{array} { r } { \mathcal { J } \left( X _ { M } ^ { \star } \right) = \operatorname* { m i n } _ { X \in S _ { M } } \mathcal { J } \left( X \right) \leq \operatorname* { m i n } _ { X \in S _ { F } } \mathcal { J } \left( X \right) = \mathcal { J } \left( X _ { F } ^ { \star } \right) . } \end{array}
$$

Sketch of Proof: Since $S _ { F } \subseteq S _ { M } .$ , the fast-loop optimum $X _ { F } ^ { \star }$ is feasible in $S _ { M }$ . Hence, minimizing the same objective over the enlarged space yields $\mathcal { J } \left( X _ { M } ^ { \star } \right) \leq \mathcal { J } \left( \bar { X } _ { F } ^ { \star } \right)$ .

The inequality establishes that the optimal medium-regime performance is no worse than that of the fast regime and can improve when additional evidence is effectively exploited. In the experiments, however, θ and <sub>ψ</sub> are trained separately, each on its own objective rather than by minimizing $\mathcal { J }$ jointly, and then held fixed; only the latent state z varies per input, and medium speed loop activation is governed by a threshold rather than direct minimization. The proposition therefore motivates the design but does not guarantee a gain for the implemented gate; the relevant empirical questions are when the inequality is strict, the magnitude of the improvement, and its computational cost.

## 2.2. Fast Loop and Epistemic Action

With fixed parameters, the FL maps each observation to a decision,

$$
\begin{array} { r } { s = \Phi _ { \theta } ( o ) , \qquad \hat { y } _ { 1 } = \arg \operatorname* { m a x } _ { \boldsymbol { v } } \pi _ { \theta } ( \boldsymbol { y } \mid s ) , \qquad c ( s ) = \underset { \boldsymbol { v } } { \operatorname* { m a x } } \pi _ { \theta } ( \boldsymbol { y } \mid s ) , } \end{array}\tag{4}
$$

where

o: observation; s: feature representation; $\pi _ { \theta } \colon$ FL decision distribution;   
$c ( s ) \colon$ confidence.

The epistemic residual $R _ { \mathrm { e p i } }$ is assessed through confidence, with low confidence triggering the MSL:

$$
\begin{array} { r } { c ( s ) < c _ { 0 } , \qquad c _ { 0 } = Q \gamma ( c ) , } \end{array}\tag{5}
$$

where

$Q _ { \gamma } .$ the <sub>γ</sub>-quantile of pooled calibration confidence, with <sub>γ</sub> the lowconfidence fraction.

Thus, $c _ { 0 }$ operationalizes $\varepsilon _ { m } ,$ with low confidence indicating a large unresolved residual. Its quantile definition provides a consistent low-confidence fraction across benchmarks. The sole parameter <sub>γ</sub> is set on the calibration set, never on the test set.

## 2.3. The second observation

A COMMIT action preserves the FL decision, whereas REVEAL acquires supplemental evidence $o ^ { \prime }$ for the medium speed loop, as illustrated in Fig. 1:

$$
\begin{array} { r } { \hat { y } = \left\{ \begin{array} { l l } { \hat { y } _ { 2 } = \arg \operatorname* { m a x } _ { y } \pi _ { \psi } ( y \mid o ^ { \prime } ) , } & { \mathrm { R E V E A L } , } \\ { \hat { y } _ { 1 } , } & { \mathrm { C O M M I T } , } \end{array} \right. } \end{array}\tag{6}
$$

where

$\pi _ { \psi } \mathrm { : }$ fixed-parameter MSL decision distribution; $o ^ { \prime } { : }$ supplemental evidence, comprising either a degraded sample, an independent draw under the same corruption, or a clean view.

The FL incurs cost $c _ { 1 } .$ , while each REVEAL incurs an additional MSL cost $c _ { 2 } .$ . Thus, at low-confidence fraction <sub>γ</sub>, the average cost per input is

$$
\bar { c } ( \gamma ) = c _ { 1 } + \gamma c _ { 2 } ,\tag{7}
$$

where

$\gamma \mathrm { : }$ low-confidence fraction in Eq. (5), equivalently the referral rate.   
Cost is measured in GMACs $( 1 0 ^ { \overset { \cdot } { 9 } }$ multiply-accumulate operations);   
evidence acquisition is assigned zero cost.

![](images/ff59f692ff9e10cc04f99fafa6e039dae65448d978420e03320cd01681fdf586.jpg)  
Fig. 1. The FL processes each input and either adopts its decision (COMMIT) or, when confidence is insufficient, invokes the MSL to acquire and process additional evidence (REVEAL).

The three cases provide objective benchmarks for evidence quality: the clean view establishes an upper bound, reprocessing the degraded input provides a baseline, and an independent draw provides fresh evidence of the same quality.

At a set <sub>γ</sub>, all three options incur the same cost, so that any difference between them is attributable to the observation alone.

## 3. EXPERIMENTAL DESIGN

Benchmarks. Four fine-grained benchmarks are used: Stanford Dogs [18], CUB-200-2011 [19], FGVC-Aircraft [20], and Oxford-IIIT Pets [21] (Dogs, CUB, Aircraft, and Pets). Dogs is split per class into 14,355 training, 3,025 calibration, and 3,200 test images; its accuracies are therefore not comparable to published results. Its images are also drawn from ImageNet [22], which is part of the pretraining data of both backbones, so its absolute accuracies may be optimistic, although all compared models share this pretraining. The other three benchmarks retain their official test sets, with calibration sets of 1,200, 1,300, and 736 images held out from their training data, on which every operating point is selected and which no FL has seen.

Models. The FL is MobileNetV3-Large [23] at 224 224 px (pixels), and the MSL is DINOv2-B [24] at 518 px, adapted with LoRA [25] and classifying by distance to class exemplars; both are fine-tuned once and then held fixed. The initial CUB and Aircraft FLs fell short of the published accuracy of their backbone and were therefore retrained with a procedure selected on the calibration set, each test set being evaluated only once: clean accuracy rose from 68.31 to 81.72 on CUB and from 58.54 to 87.70 on Aircraft, against published values of 73.91 and 85.8 [26, 27]. The Dogs and Pets FLs are retained as initially trained: the Pets FL already exceeds its published value, and the Dogs FL is 5.1 points below a linear classifier on its frozen backbone (1.5 for Pets). The single model of Section 4 is the same MSL retrained at 392 px under the identical protocol.

Conditions. In addition to the clean images, seven perturbations of our own construction are applied, following [28]: occlusion at three sizes, cutout, feature-region masking, additive noise, and blur. Unless stated otherwise, each reported accuracy is the mean over these eight conditions, with the MSL reprocessing the degraded input.

Equal cost. The three cases refer the same inputs in each condition and hence incur the same cost. The FL requires $c _ { 1 } = 0 . 2 2$ GMAC per input (0.29 on CUB) and the MSL c<sub>2</sub> = 154.94 GMAC.

The operating point. The threshold $c _ { 0 }$ is chosen so that the MSL processes the half of the inputs for which the FL is least confident, i.e., $\gamma = 1 / 2$ . This operating point is selected on the calibration set to maximize the gain over the FL while referring no more than half of the inputs to the MSL.

![](images/ffcb45796de66e25ef4351acf0a4ca0ca1b257f31c804eef413deddbec0869bf.jpg)  
Fig. 2. The FL alone, the two-loop model (FL + MSL), and the MSL alone, means over eight conditions (top), cost per input (bottom).

![](images/827ef8635c800dc454b931198c8613b8f0287705b6a7adfcc3ba9229f539c2ea.jpg)  
Fig. 3. Staged classifiers [12] and learning to defer (best of three post-hoc rules [10, 11]), run with our FL and MSL at the same cost of about 78 GMAC per input, reprocess the input the FL saw, as in Fig. 2; ours receives a clean view instead, so the gap measures the value of better evidence, acquisition cost excluded. Means over eight conditions.

Training and testing. The FL and MSL are trained separately on each benchmark’s training set, including images degraded by perturbations of the same types as the test conditions, and are then frozen. Using held-out calibration images never seen during training, c<sub>0</sub> is set to the median FL confidence across all eight conditions. At test time, the FL first classifies every image: if its confidence is at least c<sub>0</sub>, the decision is retained (COMMIT); otherwise, the MSL processes a second observation of the same object (the same degraded image, an independent draw of the same perturbation, or a clean view), and its decision is used instead (REVEAL). Generalization beyond the eight conditions is evaluated by holding each condition out when setting c<sub>0</sub> and by testing on fifteen corruption types unseen during training (Section 4.3). Performance differences are reported with 95 percent confidence intervals over test images.

## 4. RESULTS

Sections 4.1 and 4.2 address the two questions of Section 1 in turn, and Section 4.3 tests generalization.

## 4.1. Accuracy and Cost

Fig. 2 sets the FL, the two-loop model (FL + MSL) and the MSL side by side, with their respective cost; every model processes the degraded input.

Analysis. The two-loop model improves on the FL on every benchmark, by 8–13 points when the MSL reprocesses the degraded input and by 19–30 points when it processes a clean view. With the degraded input, it comes close to the MSL alone on three benchmarks and exceeds it on Aircraft, at roughly half its cost (Fig. 2). The confidence ranking accounts for less than half of this gain: against a random referral of the same size, it adds 4–5 points with the degraded input and 5–7 with a clean view.

A single model at the same cost. Retraining the MSL at 392 px, with all else unchanged, yields a single model requiring 80.31 GMAC per input, slightly more than the two-loop model on three benchmarks. On the degraded input, it outperforms the two-loop model on Dogs, CUB, and Pets by 1.3–2.6 points and underperforms it on Aircraft by 3.6 points, all four differences being significant.

Choosing between the two designs. The two-loop model assigns the more confident half of the inputs to the FL alone, reducing computation without sacrificing accuracy only when the FL is at least as accurate as the MSL on these retained inputs. On the calibration set, the FL exceeds the MSL on this subset by 9.4 points for Aircraft but underperforms it on the other three benchmarks; the test sets preserve these signs within 0.5 points. Because the two-loop model and MSL make identical decisions on the referred fraction <sub>γ</sub>, their accuracy difference is exactly [acc(FL  kept) acc(MSL  kept)](1 <sub>γ</sub>). Thus, since the single model exceeds the MSL by 0.3–1.0 points on every benchmark, a negative gap favors the single model, whereas the large positive gap on Aircraft allows the two-loop model to prevail.

Comparison with staged classifiers and learning-to-defer. The distinction is not the gate, but the action it triggers. At the same referral rate, entropy, energy, margin, and temperature-scaled confidence are within 1.4 points of max-softmax, while the best post-hoc learned deferral rule [10, 11] is within 0.4 points [8]. Staged classifiers [12], early exit [13], and learning-to-defer reprocess the available input; REVEAL instead permits acquisition of new evidence. At matched cost, a clean view improves over reprocessing by 6–17 points and over the MSL alone by 6–18 points at half its cost (Fig. 3), whereas an independent draw under the same corruption yields no gain (Section 4.2).

## 4.2. Gain of Supplemental Evidence

The same two-loop model is run three times at the same operating point and cost, changing only the evidence supplied to the MSL: the degraded input processed by the FL, an independent draw of the same corruption at the same severity, or a clean view. Since the referred inputs are identical across the three runs, any difference is attributable to the observation alone.

Analysis. A clean view gains 6–17 points over reprocessing across the eight conditions and 2–4 on the three described next. For cutout, noise, and blur, an independent draw is simply a fresh sample from the same corruption distribution: for the MSL applied to every input, it changes accuracy by 0.7 to 0.0 points relative to reprocessing, with two intervals below zero and none above; for the two-loop model, the change is within 0.1 points of zero, with every interval straddling zero. Thus, acquisition benefits primarily from improved evidence quality, not novelty.

Feature masking is the exception. Since each draw masks one of five regions, redrawing leaves aggregate FL performance essentially unchanged (within 0.6 points), but thresholding selects cases in which the discriminative region was masked. A fresh draw then improves the FL by 3–11 points and the two-loop model by 0.2–2.5 points (significant only on CUB). A clean view still gains 2–5 points, at least twice the redraw gain. The three graded occlusions are excluded because their initial occluder is centered whereas the redraw is uniform, creating a more favorable distribution under which the FL alone gains 2–6 points.

Finally, REVEAL replaces rather than fuses observations. An oracle selecting, per image, the better of reprocessing and an independent draw would raise the MSL’s accuracy by 1–4 points over reprocessing; the present design cannot attain this bound because the MSL does not retain the first observation.

The gain holds on 15 corruption types never seen  
![](images/fed72339908a7e85eb8038cae5a2c0a8b51bcee798107909ccc4ad667aec1168.jpg)  
Fig. 4. The FL alone, our two-loop model, and the MSL alone on the fifteen ImageNet-C corruption types [28] at severity three, unseen in training and calibration.

## 4.3. Generalization

Held-out conditions. Generalization is assessed by leaving one condition out at a time [29, 30]: the threshold $c _ { 0 }$ is set to the median calibration confidence over the remaining seven conditions, and the two-loop model is tested on the held-out one. On every held-out condition, the two-loop model remains above the FL, on average by 8–13 points per benchmark when reprocessing the degraded input and by 19–30 points with a clean view.

Unseen corruptions. Each benchmark’s threshold $c _ { 0 }$ is then applied, unchanged, to the fifteen ImageNet-C corruption types [28] at severity three, generated with the implementation of [31]. The two-loop model outperforms the FL on every benchmark and corruption type (Fig. 4); it is within three points of the MSL alone on every benchmark and above it on Aircraft, at 58–79 percent of its cost. The referral rate is not prescribed but follows from the threshold: on Dogs it ranges from 30 percent of the inputs under brightness to all of them under Gaussian noise, where the FL accuracy falls below one percent and the two-loop model then requires more computation than the MSL alone. Here too, an independent draw yields no gain.

## 5. DISCUSSION AND CONCLUSION

Two-Loop Model: Strengths and Limitations. With a fixed FL–MSL pair, confidence-thresholded referral, and a second observation acquired only when the FL is uncertain, the two-loop model gains 8–13 points over the FL at roughly half the MSL cost. The gain persists across all held-out conditions and an unseen corruption benchmark, exceeds random referral at matched cost, and satisfies the inequality of Section 2.1 on every benchmark. However, retraining the MSL at the computational cost of the two-loop model performs better on three of four benchmarks. The exception is precisely where the FL classifies its retained inputs more accurately than the MSL, a condition detectable from calibration before training the alternative. Moreover, the benefit of the second observation arises primarily from improved quality rather than novelty. Since Eq. (7) assigns zero cost to evidence acquisition, selective acquisition is advantageous only when acquiring evidence itself carries a cost, as when a second viewpoint, a higher-resolution or longer-exposure capture, or a retake must be requested; otherwise, a single model of the same computational cost is preferable.

Running time and alternatives. In running time rather than arithmetic operations, the reduction is smaller: about 70 percent of the MSL’s time on Dogs, rather than half. Given the same clean view, the FL alone still trails the two-loop model by 3.8–5.5 points.

Future work. Two directions remain open: fusing both observations, and an epistemic-residual test that improves upon confidence.

## 6. COMPLIANCE WITH ETHICAL STANDARDS

This study involved no human or animal subjects. It used four publicly available benchmarks of photographs of dogs, birds, aircraft, and pets [18, 19, 20, 21], the Stanford Dogs images being drawn from ImageNet [22], each used for non-commercial research under its license or terms of use, together with publicly released pretrained models [23, 24]. No data were collected and no live animals were involved. Some photographs incidentally show people; no individual was identified or analyzed, and no dataset images are reproduced in this paper. Ethical approval was therefore not required.

## 7. ACKNOWLEDGMENTS

This work was supported by the College of Engineering, North Carolina State University, and the Graduate School of Informatics, Nagoya University. The authors have no relevant financial or nonfinancial interests to disclose.

## 8. REFERENCES

[1] Ashish Vaswani et al., “Attention is all you need,” in Advances in Neural Information Processing Systems (NeurIPS), 2017, pp. 5998–6008.

[2] Daniel Kahneman, Thinking, Fast and Slow, Farrar, Straus and Giroux, New York, 2011.

[3] Yoshua Bengio, “The consciousness prior,” arXiv preprint arXiv:1709.08568, 2017.

[4] C. K. Chow, “On optimum recognition error and reject tradeoff,” IEEE Trans. Inf. Theory, vol. 16, no. 1, pp. 41–46, 1970.

[5] Yonatan Geifman and Ran El-Yaniv, “Selective classification for deep neural networks,” in Advances in Neural Information Processing Systems (NeurIPS), 2017, pp. 4878–4887.

[6] Dan Hendrycks and Kevin Gimpel, “A baseline for detecting misclassified and out-of-distribution examples in neural networks,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2017.

[7] Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger, “On calibration of modern neural networks,” in Proc. Int. Conf. Mach. Learn. (ICML), 2017, pp. 1321–1330.

[8] Paul F. Jaeger, Carsten T. Lüth, Lukas Klein, and Till J. Bungert, “A call to reflect on evaluation practices for failure detection in image classification,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2023.

[9] Hussein Mozannar and David Sontag, “Consistent estimators for learning to defer to an expert,” in Proc. Int. Conf. Mach. Learn. (ICML), 2020, pp. 7076–7087.

[10] Harikrishna Narasimhan, Wittawat Jitkrittum, Aditya Krishna Menon, Ankit Singh Rawat, and Sanjiv Kumar, “Post-hoc estimators for learning to defer to an expert,” in Advances in Neural Information Processing Systems (NeurIPS), 2022.

[11] Wittawat Jitkrittum, Neha Gupta, Aditya Krishna Menon, Harikrishna Narasimhan, Ankit Singh Rawat, and Sanjiv Kumar, “When does confidence-based cascade deferral suffice?,” in Advances in Neural Information Processing Systems (NeurIPS), 2023, pp. 9891–9906.

[12] Paul Viola and Michael Jones, “Rapid object detection using a boosted cascade of simple features,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2001, pp. 511–518.

[13] Surat Teerapittayanon, Bradley McDanel, and H. T. Kung, “BranchyNet: Fast inference via early exiting from deep neural

networks,” in Proc. Int. Conf. Pattern Recognit. (ICPR), 2016, pp. 2464–2469.

[14] Tolga Bolukbasi, Joseph Wang, Ofer Dekel, and Venkatesh Saligrama, “Adaptive neural networks for efficient inference,” in Proc. Int. Conf. Mach. Learn. (ICML), 2017, pp. 527–536.

[15] Karl Friston, “The free-energy principle: a unified brain theory?,” Nature Reviews Neuroscience, pp. 127–138, 2010.

[16] Vahid Jebraeeli, Bo Jiang, Derya Cansever, and Hamid Krim, “Koopcon: A new approach towards smarter and less complex learning,” in Proc. IEEE Int. Conf. Image Process. (ICIP), 2024.

[17] Vahid Jebraeeli, Bo Jiang, Hamid Krim, and Derya Cansever, “Generative expansion of small datasets: An expansive graph approach,” in Proc. IEEE Int. Conf. Acoust., Speech, Signal Process. (ICASSP), 2025.

[18] Aditya Khosla, Nityananda Jayadevaprakash, Bangpeng Yao, and Li Fei-Fei, “Novel dataset for fine-grained image categorization: Stanford Dogs,” in Proc. CVPR Workshop on Fine-Grained Visual Categorization (FGVC), 2011.

[19] Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie, “The Caltech-UCSD birds-200-2011 dataset,” Tech. Rep. CNS-TR-2011-001, California Institute of Technology, 2011.

[20] Subhransu Maji, Esa Rahtu, Juho Kannala, Matthew Blaschko, and Andrea Vedaldi, “Fine-grained visual classification of aircraft,” arXiv preprint arXiv:1306.5151, 2013.

[21] Omkar M. Parkhi, Andrea Vedaldi, Andrew Zisserman, and C. V. Jawahar, “Cats and dogs,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2012, pp. 3498–3505.

[22] Jia Deng et al., “ImageNet: A large-scale hierarchical image database,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2009, pp. 248–255.

[23] Andrew Howard et al., “Searching for MobileNetV3,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2019, pp. 1314– 1324.

[24] Maxime Oquab et al., “DINOv2: Learning robust visual features without supervision,” Trans. Mach. Learn. Res., 2024.

[25] Edward J. Hu et al., “LoRA: Low-rank adaptation of large language models,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2022.

[26] Pranav Jeevan and Amit Sethi, “Which backbone to use: A resource-efficient domain specific comparison for computer vision,” arXiv preprint arXiv:2406.05612, 2024.

[27] Simon Kornblith, Jonathon Shlens, and Quoc V. Le, “Do better ImageNet models transfer better?,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2019, pp. 2661–2671.

[28] Dan Hendrycks and Thomas G. Dietterich, “Benchmarking neural network robustness to common corruptions and perturba tions,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2019.

[29] Ishaan Gulrajani and David Lopez-Paz, “In search of lost domain generalization,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2021.

[30] François Chollet, “On the measure of intelligence,” arXiv preprint arXiv:1911.01547, 2019.

[31] Claudio Michaelis et al., “Benchmarking robustness in object detection: Autonomous driving when winter is coming,” arXiv preprint arXiv:1907.07484, 2019.