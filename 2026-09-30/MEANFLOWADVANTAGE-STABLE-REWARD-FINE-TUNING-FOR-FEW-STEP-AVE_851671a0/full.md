# MEANFLOWADVANTAGE: STABLE REWARD FINE-TUNING FOR FEW-STEP AVERAGE-VELOCITY GENER-ATORS

Haocheng Tang<sup>∗†</sup>   
Northeastern University   
Boston, MA 02115, USA   
tang.haoc@northeastern.edu

Xingqiao Lin Carnegie Mellon University Pittsburgh, PA 15213, USA xingqiao@andrew.cmu.edu

Tianchi Xie<sup>∗</sup>   
Tsinghua University   
Beijing, 100084, PRC   
xietc24@mails.tsinghua.edu.cn

## ABSTRACT

MeanFlow enables efficient few-step generation by predicting interval-average velocities, but this representation creates a mismatch for reward fine-tuning: existing advantage-based objectives are typically defined on instantaneous velocities or equivalent x -space predictions, whereas inference directly uses the learned average-velocity map. We introduce MeanFlowAdvantage, a signed advantageweighted least-squares objective for average-velocity generators. Our key construction uses a shared, detached MeanFlow derivative correction to express the reward objective in prediction space while making rollout and reference regularization exact penalties on the average-velocity network deployed at inference. The resulting formulation preserves MeanFlow’s native few-step sampler and provides a direct mechanism for transferring reward improvements to the deployed flow map. On SD3.5-Medium, MeanFlowAdvantage improves all eight reported metrics over the matched four-step MeanFlowNFT baseline and, with only four NFEs, matches or exceeds the 40-step DiffusionNFT baseline on six of eight metrics. The same objective also transfers to DNA promoter design, where it supports both teacher-free on-policy RL for a generator defined on a manifold and teacherguided reward-graded distillation, with the latter yielding the lowest one-step Sei profile MSE among the compared configurations.

## 1 INTRODUCTION

Few-step flow matching changes the representation that post-training must optimize. In standard flow matching, the model predicts an instantaneous velocity whose numerical integration produces samples (Lipman et al., 2023; Liu et al., 2023; Albergo et al., 2023). MeanFlow (Geng et al., 2025; Gu et al., 2026), in contrast, predicts an interval-average velocity whose learned flow map can traverse a finite time interval in a single network evaluation. This representation is precisely what enables one-step and few-step sampling. However, it also creates a mismatch for reward finetuning: existing advantage-based objectives are typically formulated for instantaneous velocities or equivalent prediction spaces, whereas MeanFlow inference directly executes the learned average velocity map. A useful reward objective should therefore improve sample quality without losing alignment with the representation that makes few-step generation possible.

Existing reinforcement learning (RL) approaches do not directly resolve this mismatch. Reverseprocess methods such as DDPO, DPOK, and Flow-GRPO optimize stochastic denoising trajectories (Black et al., 2024; Fan et al., 2023; Liu et al., 2025), while forward-process methods incorporate reward signals into prediction objectives. DiffusionNFT uses implicit positive and negative policies (Zheng et al., 2026), Advantage Weighted Matching reweights a matching surrogate (Xue et al., 2025), and AdvantageFlow applies signed advantage-weighted least squares with rollout regularization (Kveton et al., 2026). These formulations naturally operate on instantaneous representations. Simply replacing such a predictor with a MeanFlow output is not equivalent: an interval-average velocity represents the transport over an entire interval rather than the instantaneous direction at one time point. Moreover, an induced predictor alone does not ensure that rollout or reference regularization constrains the positive-length average-velocity maps used by the sampler.

Recent forward-process work has shown that MeanFlow can support reward fine-tuning through an induced instantaneous predictor (Huang et al., 2026b). This raises two questions: how can signed advantages be used to stably improve the finite-interval MeanFlow map, and can the same rewardbased formulation also distill improved behavior from multi-step teachers into few-step MeanFlow models?

We introduce MeanFlowAdvantage (MFA), a signed advantage-weighted least-squares objective that provides a unified solution to both problems. MFA applies reward supervision in prediction space while regularizing the underlying average-velocity model against both the rollout policy and a frozen reference. A shared, stop-gradient correction makes these regularization terms reduce to direct differences between the corresponding MeanFlow velocity predictions, avoiding interference between reward fitting and model anchoring. This same formulation naturally extends from onpolicy reward fine-tuning to reward-guided distillation, allowing few-step MeanFlow students to learn from improved multi-step teacher trajectories. Our contributions are threefold:

• Reward fine-tuning aligned with average-velocity generation. We formulate signed advantage optimization directly for finite-interval MeanFlow maps, while preserving the induced prediction space needed for reward supervision. This yields a representation-aligned objective that regularizes the average-velocity predictions used for generation rather than only the induced instantaneous boundary predictor.

• State-of-the-art few-step performance in our evaluation. On SD3.5-Medium, MFA achieves the strongest four-NFE results among the compared MeanFlow reward-finetuning methods, improving all eight reported metrics over MeanFlowNFT. With only four NFEs, it also matches or exceeds the 40-NFE DiffusionNFT baseline on six of eight metrics.

• One objective for both on-policy RL and distillation. On DNA promoters (Appendix E), the same loss runs as teacher-free RL on a manifold generator and, with a reward-guided teacher in place of the rollout, transfers graded teacher improvements into a one-step student, achieving the best Sei profile error among the compared configurations.

## 2 PRELIMINARIES

We use the image convention t = 0 for data and $t = 1$ for noise. The earlier interval endpoint is $s \leq t ,$ and c denotes conditioning. We omit c from the notation when clear from context, and distinguish average velocity u, instantaneous velocity v, and a training-time induced predictor V . The promoter implementation uses the opposite time convention, stated separately in Section E.

Flow matching Flow matching (Lipman et al., 2023) learns a continuous-time transport from a simple noise distribution to the conditional data distribution $q ( x _ { 0 } \mid c )$ . It represents this transport through a time-dependent velocity field whose probability-flow ODE is integrated from noise at $t = 1$ to data at $t = 0$ at inference time. Rectified flow specializes this construction to a linear interpolation between data and noise. For $x _ { 0 } \sim q ( \cdot | \ c )$ and independent $\epsilon \sim \mathcal { N } ( 0 , I )$ , it defines

$$
x _ { t } = ( 1 - t ) x _ { 0 } + t \epsilon , \qquad v _ { t } = \epsilon - x _ { 0 } , \qquad \mathcal { L } _ { \mathrm { F M } } = \mathbb { E } \| v _ { \theta } ( x _ { t } , t ) - v _ { t } \| _ { 2 } ^ { 2 } .\tag{1}
$$

The population minimizer of this squared loss is the marginal instantaneous velocity $v ( x _ { t } , t , c ) =$ $\mathbb { E } [ v _ { t } \ \mid \ x _ { t } , t , c ]$ . For rectified flow (Liu et al., 2023), the same instantaneous-velocity prediction can equivalently be expressed in x<sub>0</sub>-space as $f _ { \theta } = x _ { t } - t v _ { \theta }$ , while $x _ { 0 } = x _ { t } - t v _ { t }$ . Therefore, $f _ { \theta } - x _ { 0 } = - t ( v _ { \theta } - v _ { t } )$ and $\| f _ { \theta } - \bar { x _ { 0 } } \| _ { 2 } ^ { 2 } = \dot { t } ^ { 2 } \| v _ { \theta } - v _ { t } \| _ { 2 } ^ { 2 }$ . This prediction-space representation provides a convenient form for incorporating reward signals into forward-process objectives.

AdvantageFlow Building on this representation, AdvantageFlow (Kveton et al., 2026) incorporates signed reward information through a quadratic objective on the x<sub>0</sub>-space predictions of the learner, rollout model, and frozen reference:

$$
\ell ^ { \mathrm { A F } } = A \| f _ { \boldsymbol { \theta } } - x _ { 0 } \| _ { 2 } ^ { 2 } + \gamma \| f _ { \boldsymbol { \theta } } - f _ { \mathrm { o l d } } \| _ { 2 } ^ { 2 } + \lambda \| f _ { \boldsymbol { \theta } } - f _ { \mathrm { r e f } } \| _ { 2 } ^ { 2 } .\tag{2}
$$

Here A is a signed advantage and $\gamma , \lambda \geq 0$ . The loss is strictly convex in the prediction when $A + \gamma + \lambda > 0$ , even when $A \ : < \ : 0$ . For standard flow models, this objective acts naturally on a prediction obtained from the instantaneous velocity. MeanFlow, however, deploys an intervalaverage velocity, so reward fine-tuning must additionally account for the finite-interval map used by its sampler.

MeanFlow and reward fine-tuning Along a trajectory $\dot { x } _ { \tau } ~ = ~ v ( x _ { \tau } , \tau )$ , MeanFlow represents transport over a finite interval [s, t] through the average velocity

$$
u ( x _ { t } , s , t ) = \frac { 1 } { t - s } \int _ { s } ^ { t } v ( x _ { \tau } , \tau ) \mathrm { d } \tau , \qquad \operatorname* { l i m } _ { s \to t } u ( x _ { t } , s , t ) = v ( x _ { t } , t ) .\tag{3}
$$

Holding s fixed and differentiating along the trajectory yields the MeanFlow identity

$$
u + ( t - s ) \big [ \partial _ { t } u + ( \partial _ { x } u ) v \big ] = v .\tag{4}
$$

For an exact average velocity, $x _ { s } = x _ { t } - ( t - s ) u ( x _ { t } , s , t )$ is the exact flow map; replacing u by u gives the learned sampler $x _ { s } = x _ { t } - ( t - s ) u _ { \theta } ( x _ { t } , s , t )$ . Thus, unlike standard flow matching, Mean-Flow inference directly uses the average-velocity maps predicted by $u _ { \theta }$ . Reward fine-tuning should therefore remain aligned with these deployed maps. MeanFlowNFT uses the MeanFlow identity to construct an instantaneous predictor $\bar { V }$ from u and applies an NFT objective to $V$ (Huang et al., 2026b). We use the same induction with the signed quadratic in equation 2, while regularizing the corresponding average-velocity maps directly; Section A.4 makes the connection to NFT explicit.

## 3 MEANFLOWADVANTAGE

The preliminaries establish AdvantageFlow in prediction space and MeanFlow in average-velocity space. MeanFlowAdvantage connects these representations while keeping optimization aligned with the finite-interval maps deployed by the sampler. We first derive a shared MeanFlow correction that links the average velocity u to the prediction-space objective, and show that the resulting quadratic is exactly an advantage-weighted MeanFlow regression with rollout and reference anchors on the deployed maps. We then specify the samples, advantages, and training intervals used for optimization, followed by an analysis of the resulting objective and its fixed point.

## 3.1 LINKING AVERAGE VELOCITY TO PREDICTION SPACE

The naive substitution $x _ { t } - t u _ { \theta } ( x _ { t } , s , t )$ does not give the corresponding $x _ { 0 }$ -space prediction when $s \_ t$ . In rectified flow, $x _ { t } \mathrm { ~ - ~ } t v$ depends on the instantaneous velocity at time $t ,$ whereas u averages velocity over the interval $[ s , t ]$ . The MeanFlow identity equation 4 relates the two as $v = \bar { u } + ( t - s ) \mathrm { D } u$ , where $\mathrm { D } u = \mathbf { \bar { \partial } } \partial _ { t } \bar { u } + ( \partial _ { x } u ) v$ denotes the rate of change of u along the trajectory. This correction therefore links the average-velocity representation to the prediction space used by AdvantageFlow. Given a rollout sample $x _ { 0 } ,$ , we draw noise ϵ and an interval $s \leq t ,$ , form $x _ { t }$ by equation 1, and set $v _ { t } = \epsilon - x _ { 0 }$ . With finite-difference step size $\delta > 0$ , we estimate the correction once from the rollout network along the sampled direction $v _ { t } \colon$

$$
d = \frac { u _ { \mathrm { o l d } } \big ( x _ { t } + \delta v _ { t } , s , t + \delta \big ) - u _ { \mathrm { o l d } } \big ( x _ { t } - \delta v _ { t } , s , t - \delta \big ) } { 2 \delta } = \partial _ { t } u _ { \mathrm { o l d } } + \big ( \partial _ { x } u _ { \mathrm { o l d } } \big ) v _ { t } + O \big ( \delta ^ { 2 } \big ) .\tag{5}
$$

A Jacobian-vector product computes the same quantity without the $O ( \delta ^ { 2 } )$ error. Appendix B gives the finite-difference implementation and boundary handling. We detach $d ,$ with $\mathrm { s g } [ \cdot ]$ denoting stopgradient, and share the same value across the learner, rollout model, and reference model, indexed by k:

$$
V _ { k } = u _ { k } ( x _ { t } , s , t ) + ( t - s ) \operatorname { s g } [ d ] , \qquad F _ { k } = x _ { t } - t V _ { k } , \qquad k \in \{ \theta , \mathrm { o l d } , \mathrm { r e f } \} .\tag{6}
$$

Here $V _ { k }$ is the corrected instantaneous-velocity representation and $F _ { k }$ its corresponding x<sub>0</sub>-space prediction. As the same correction is used for all three networks, it cancels in every pairwise difference:

$$
F _ { \theta } - F _ { k } = - t ( u _ { \theta } - u _ { k } ) , \qquad V _ { \theta } - V _ { k } = u _ { \theta } - u _ { k } , \qquad k \in \{ \mathrm { o l d } , \mathrm { r e f } \} .\tag{7}
$$

The rollout and reference anchors therefore compare the average-velocity networks directly on the sampled interval $( s , t )$ . With a separate derivative for each network, each anchor residual would instead contain an additional term $\bar { t ( t - s ) } ( d _ { \theta } - d _ { k } )$ depending on the sampled direction $v _ { t } ;$ Section 4 shows why this matters.

## 3.2 ADVANTAGE-WEIGHTED LEAST SQUARES ON THE INDUCED PREDICTION

With the prediction equation $^ { 6 , }$ the MFA loss is the AdvantageFlow quadratic equation 2:

$$
\ell _ { \theta } = A \| F _ { \theta } - x _ { 0 } \| _ { 2 } ^ { 2 } + \gamma \| F _ { \theta } - F _ { \mathrm { o l d } } \| _ { 2 } ^ { 2 } + \lambda \| F _ { \theta } - F _ { \mathrm { r e f } } \| _ { 2 } ^ { 2 } , \qquad \mathcal { L } _ { \mathrm { M F A } } ( \theta ) = \mathbb { E } _ { x _ { 0 } \sim \pi _ { \mathrm { o l d } } } [ \ell _ { \theta } ] .\tag{8}
$$

Within one update, the samples, the advantages, and the two anchor predictions are held fixed.

MFA is an advantage-weighted MeanFlow regression. Denote the regression target of MeanFlow training (Geng et al., 2025) by

$$
{ \bar { u } } = v _ { t } - ( t - s ) d .\tag{9}
$$

Since $x _ { t } - x _ { 0 } = t v _ { t } .$ , we have $F _ { \theta } - x _ { 0 } = - t ( u _ { \theta } - \bar { u } )$ . Combined with equation 7, this turns the loss into an exact identity in average-velocity space:

$$
\ell _ { \theta } = t ^ { 2 } \Big ( A \| u _ { \theta } - \bar { u } \| _ { 2 } ^ { 2 } + \gamma \| u _ { \theta } - u _ { \mathrm { o l d } } \| _ { 2 } ^ { 2 } + \lambda \| u _ { \theta } - u _ { \mathrm { r e f } } \| _ { 2 } ^ { 2 } \Big ) .\tag{10}
$$

The data term is the standard MeanFlow loss weighted by the advantage, and the anchors keep the deployed map close to the rollout and reference maps on the sampled interval. The prediction $F$ is only a device to derive equation $1 0 ;$ what is trained and regularized is $u _ { \theta }$ itself. When $s = t ,$ the correction vanishes and equation 10 reduces to AdvantageFlow on the boundary velocity $u _ { \theta } ( x _ { t } , t , t )$ (Appendix A.1). The positive-length intervals are therefore what is new for few-step generators.

Per-sample solution and signed advantages. Treat $u = u _ { \theta } ( x _ { t } , s , t )$ as a free vector. The loss equation 10 has curvature $2 t ^ { 2 } ( A ^ { \setminus } + \gamma + \lambda )$ , so whenever $A + \gamma + \lambda > 0$ it is strictly convex and has the unique minimizer

$$
u ^ { \star } = \frac { A \bar { u } + \gamma u _ { \mathrm { o l d } } + \lambda u _ { \mathrm { r e f } } } { A + \gamma + \lambda } .\tag{11}
$$

The three weights sum to one. For $A > 0 , u ^ { \star }$ moves toward the target u¯. For $A < 0 .$ , the weight on u¯ is negative, so $u ^ { \star }$ is pushed away from the target, past the rollout output. A poor sample therefore produces a push away from itself instead of being ignored, and the loss remains convex. We use $\gamma = 1 . 1 , \lambda = 1 0 ^ { - 3 }$ , and $A \in [ - 1 , 1 ]$ , so $A + \gamma + \lambda \geq 0 . 1 0 1$ always holds. Section 4 shows that $\gamma$ also controls how far each update can move.

Adaptive scale. As in the DiffusionNFT trainer, the image implementation divides each sample’s loss by a detached positive scale:

$$
{ \mathcal { L } } _ { \mathrm { t r a i n } } = \mathbb { E } [ \ell _ { \theta } / w ] , \qquad w = \operatorname* { m a x } \bigl ( \operatorname { s g } [ \operatorname* { m e a n } | F _ { \theta } - x _ { 0 } | ] , 1 0 ^ { - 5 } \bigr ) ,\tag{12}
$$

where the mean is over latent coordinates. The three terms of a sample share the same w, so w does not change the per-sample minimizer equation 11. It only changes how much each sample contributes to the gradient. The ablations show that this matters for stability (Section 5.2). The analysis below is stated for the unscaled loss.

## 3.3 SAMPLES, ADVANTAGES, AND INTERVALS

We keep the outer loop of MeanFlowNFT (Huang et al., 2026b), so any difference from it comes from the objective alone.

Rollouts. Starting from Gaussian noise, the frozen rollout model runs the N-step sampler on a decreasing grid $1 = t _ { N } > \cdot \cdot \cdot > t _ { 0 } = 0 \colon$

$$
x _ { t _ { i - 1 } } = x _ { t _ { i } } - ( t _ { i } - t _ { i - 1 } ) u _ { \mathrm { o l d } } ( x _ { t _ { i } } , t _ { i - 1 } , t _ { i } ) , \qquad i = N , \ldots , 1 ,\tag{13}
$$

with $N = 4$ and no CFG. The rollout parameters are an EMA of the learner and are refreshed between updates. The reference is the frozen AnyFlow initialization (Gu et al., 2026). Fine-tuning uses LoRA (Hu et al., 2022).

Advantages. For L prompts with K images each, rewards are centered within each prompt and divided by a global standard deviation:

$$
A ^ { i , k } = \mathrm { c l i p } \Big ( \frac { R ^ { i , k } - { \widehat R } ^ { i } } { \operatorname* { m a x } ( Z , \varepsilon _ { Z } ) } , - 1 , 1 \Big ) , \qquad { \widehat R } ^ { i } = \frac { 1 } { K } \sum _ { k } R ^ { i , k } , \qquad Z ^ { 2 } = \frac { 1 } { L K } \sum _ { i , k } \bigr ( R ^ { i , k } - { \widehat R } ^ { i } \bigr ) ^ { 2 } .\tag{14}
$$

With several rewards, each is standardized separately and the results are combined with fixed weights before clipping. The advantages are signed: an image worse than its prompt average receives $A < 0$

Intervals. $\mathbf { A }$ few-step sampler takes jumps of positive length, so training must include such intervals. Each sample receives one of three interval types:

$$
s = t \mathrm { ( b o u n d a r y ) } , \quad \quad s = 0 \mathrm { ( f u l l j u m p t o d a t a ) } , \quad \quad 0 < s < t \mathrm { ( g e n e r a l i n t e r v a l ) } ,\tag{15}
$$

with probabilities (0.5, 0.25, 0.25) and the time shift of the base scheduler. Boundary samples keep the instantaneous velocity accurate. The other two types train the maps that the sampler actually runs. Algorithm 1 in Appendix B summarizes one update.

## 4 WHAT DOES MEANFLOWADVANTAGE LEARN?

The identity equation 10 holds exactly for every sample. We now take expectations and answer three questions. Which function does the loss drive the network toward? How large is each update, differing from MeanFlowNFT? When does the few-step sampler inherit the improvement? The analysis is at the population level, with a free function in place of the network. We write $Z =$ $( x _ { t } , s , t , c )$ for the network input and treat the advantage as $\overset { \cdot } { A } \overset { \cdot } { = } A ( x _ { 0 } , c )$ . Proofs are in Appendix A.

Assumptions. We use three assumptions: (A1) Exact correction: $d = \partial _ { t } u _ { \mathrm { o l d } } + ( \partial _ { x } u _ { \mathrm { o l d } } ) v _ { t }$ , which a JVP computes exactly while equation 5 has error up to $O ( \delta ^ { 2 } ) : ( A 2 )$ Positive weights: $A + \gamma > 0$ for every sample; and (A3) Calibrated rollout: $\mathbb { E } _ { \pi _ { \mathrm { o l d } } } \mathbf { \bar { \Pi } } [ \bar { u } \mid \hat { Z } ] = u _ { \mathrm { o l d } } ( Z )$ . Assumption (A3) means that the rollout network is a MeanFlow fixed point on its own samples. Under (A1), this corresponds to the MeanFlow identity equation 4 for the velocity field of $\pi _ { \mathrm { o l d } }$ . It is exact only for a perfectly calibrated rollout; we discuss this gap at the end of the section.

For each prompt c, define the reweighted distribution

$$
q ( x _ { 0 } \mid c ) = { \frac { \gamma + A ( x _ { 0 } , c ) } { \gamma + { \bar { A } } _ { c } } } \ \pi _ { \mathrm { o l d } } ( x _ { 0 } \mid c ) , \qquad { \bar { A } } _ { c } = \mathbb { E } _ { \pi _ { \mathrm { o l d } } } [ A \mid c ] .\tag{16}
$$

q upweights positive-advantage samples and downweights negative ones; smaller $\gamma$ strengthens the reweighting. Let $v _ { q } ( x _ { t } , t , c ) = \mathbb { E } _ { q } [ \epsilon - x _ { 0 } \ | \ x _ { t } , t , c ]$ be the rectified-flow velocity that transports noise to q.

Theorem 1 (MFA fits the MeanFlow target of a reward-reweighted distribution). Assume (A1)–(A3). There is a constant C that does not depend on θ such that

$$
\mathcal { L } _ { \mathrm { M F A } } ( \theta ) \big | _ { \lambda = 0 } = \mathbb { E } _ { c } \Big [ ( \gamma + \bar { A } _ { c } ) \mathbb { E } _ { q } \big [ t ^ { 2 } \| u _ { \theta } - \bar { u } \| _ { 2 } ^ { 2 } \big | c \big ] \Big ] + C .\tag{17}
$$

For any $\lambda \geq 0 ,$ , the loss has a unique minimizer overfreefunctions $u ( Z )$

$$
u ^ { + } = \alpha u _ { q } + ( 1 - \alpha ) u _ { \mathrm { r e f } } , \qquad \alpha = \frac { \gamma + \mathbb { E } _ { \pi _ { \mathrm { o l d } } } [ A \mathbin { \lrcorner } Z ] } { \gamma + \lambda + \mathbb { E } _ { \pi _ { \mathrm { o l d } } } [ A \mathbin { \lrcorner } Z ] } ,\tag{18}
$$

where $u _ { q } = v _ { q } - ( t - s ) \big [ \partial _ { t } u _ { \mathrm { o l d } } + \left( \partial _ { x } u _ { \mathrm { o l d } } \right) v _ { q } \big ] .$

For $\lambda = 0$ , MFA on rollout samples is equivalent to ordinary MeanFlow training under $q ,$ without sampling from q: $u _ { q }$ is the corresponding MeanFlow target with the rollout derivative stopgradiented. Positive λ blends this target with the reference map. The shared correction is essential: because $u _ { \theta } - u _ { \mathrm { o l d } }$ depends only on $Z$ equation 7, the rollout anchor becomes a MeanFlow regression up to a constant; separate derivatives introduce dependence on $v _ { t }$ and break this equivalence.

Proposition 1 (Update direction and step size). Assume $( A I ) – ( A 3 )$ , and let $\sigma _ { A } ^ { 2 } ( Z ) = \operatorname { V a r } _ { \pi _ { \mathrm { o l d } } } ( A \mid$ Z) and $\sigma _ { \bar { u } } ^ { 2 } ( Z ) = \bar { \mathrm { t r } } \mathrm { C o v } _ { \pi _ { \mathrm { o l d } } } ( \bar { u } \mid Z )$ . Then

$$
\begin{array} { r l } & { u ^ { + } - u _ { \mathrm { o l d } } = \frac { \mathrm { C o v } _ { \pi _ { \mathrm { o l d } } } ( A , \bar { u } \mid Z ) + \lambda ( u _ { \mathrm { r e f } } - u _ { \mathrm { o l d } } ) } { \gamma + \lambda + \mathbb { E } _ { \pi _ { \mathrm { o l d } } } [ A \mid Z ] } , } \\ & { \| u ^ { + } - u _ { \mathrm { o l d } } \| _ { 2 } \leq \frac { \sigma _ { A } \sigma _ { \bar { u } } + \lambda \| u _ { \mathrm { r e f } } - u _ { \mathrm { o l d } } \| _ { 2 } } { \gamma + \lambda + \mathbb { E } _ { \pi _ { \mathrm { o l d } } } [ A \mid Z ] } . } \end{array}\tag{19}
$$

The update direction is the covariance between the advantage and the MeanFlow target. Among all images that are consistent with the noisy input $Z ,$ the map moves toward the targets of highadvantage images and away from those of low-advantage images. The step size scales as $1 / \gamma , \mathbf { s o } \gamma$ acts as an inverse step size, or trust-region radius, on the deployed map. The step is also bounded by the spread of advantages at $Z \colon$ if all images consistent with $\dot { Z }$ are equally good, nothing moves. A constant advantage $A \equiv 1$ has zero covariance, so for $\lambda = 0$ it gives $u ^ { + } = u _ { \mathrm { o l d } }$ . It carries no reward information, and the loss only refits the rollout samples.

Proposition 2 (Relation to MeanFlowNFT). Apply the NFT loss (Zheng et al., 2026) to the induced velocity equation 6 as in MeanFlowNFT, with parameter $\beta > 0 ,$ , optimality probability $r \in [ 0 , 1 ]$ no reference term, and no adaptive weights. Under (A1) and (A3), its minimizer overfree functions is

$$
u _ { \mathrm { N F T } } ^ { + } = u _ { \mathrm { o l d } } + \frac { 1 } { \beta } \mathrm { C o v } _ { \pi _ { \mathrm { o l d } } } ( A , \bar { u } \mid Z ) , \qquad A = 2 r - 1 .\tag{20}
$$

Compared with equation 19 at $\lambda = 0$ , both objectives move in the same direction, but NFT uses the fixed gain $1 / \bar { \beta }$ while MFA uses $1 / ( \gamma + \dot { \mathbb { E } } _ { \pi _ { \mathrm { o l d } } } [ A \ | \ Z ] )$ . By Theorem 1, the MFA target at every input is the MeanFlow target of one fixed distribution q; the NFT target has this form only $\mathrm { i f ~ } \stackrel { \cdot } { \mathbb { E } } _ { \pi _ { \mathrm { o l d } } } \left[ \boldsymbol { \bar { A } } \mid Z \right]$ does not depend on $Z .$ For example, with $A = 2 r - 1$ and $\gamma = 1 , q \propto r \pi _ { \mathrm { o l d } }$ is exactly the positive policy that motivates DiffusionNFT. MFA regresses onto its MeanFlow target $u _ { \pi ^ { + } }$ everywhere, whereas NFT regresses onto $u _ { \mathrm { o l d } } + ( 2 \rho / \beta ) ( u _ { \pi ^ { + } } - u _ { \mathrm { o l d } } )$ with $\rho ( Z ) = \mathbb { E } _ { \pi _ { \mathrm { o l d } } } [ r \mid { \bar { Z } } ]$ That target falls short of $u _ { \pi ^ { + } }$ where $2 \rho < \beta$ and overshoots it where $2 \rho > \beta$ (Appendix A.4). This does not make MFA uniformly better, but its target has a simple distributional meaning at every noise level.

Theorem 2 (When the sampler inherits the improvement). Let $\lambda = 0 ,$ , and let u<sup>∗</sup> be a fixed point of the target in Theorem 1 when its own derivative is used:

$$
u ^ { * } = v _ { q } - ( t - s ) \big [ \partial _ { t } u ^ { * } + \left( \partial _ { x } u ^ { * } \right) v _ { q } \big ] \quad f o r a l l 0 \leq s < t \leq 1 .\tag{21}
$$

Under the regularity conditions ofLemma $I , u ^ { * }$ is the exact average velocity of the flow of $v _ { q } .$ . For every number of steps N and every grid, the sampler equation $\bar { I 3 }$ with $u ^ { * }$ therefore draws exact samples from q. For each prompt,

$$
\mathbb { E } _ { q } [ R \mid c ] - \mathbb { E } _ { \pi _ { \mathrm { o l d } } } [ R \mid c ] = \frac { \operatorname { C o v } _ { \pi _ { \mathrm { o l d } } } ( A , R \mid c ) } { \gamma + \bar { A } _ { c } } ,\tag{22}
$$

which is nonnegative whenever A is a nondecreasing function of a scalar reward R.

Condition equation 21 is the same stop-gradient fixed point that MeanFlow pretraining aims for: the target uses the current network’s derivative, and a network that matches its own target is an exact flow map. Theorem 2 has two practical consequences. First, the condition must hold for every interval. Boundary training alone $\mathbf { \boldsymbol { s } } = t \mathbf { \boldsymbol { ) } }$ only pins down $v _ { q }$ and leaves the maps used by the sampler unconstrained, which is why equation 15 includes positive-length intervals. Second, the target distribution q does not depend on N, so at the fixed point one update improves every sampling budget at once. Together with Proposition 1, it also shows the role of $\gamma \colon$ a smaller γ gives a larger possible improvement equation 22, but also larger and less stable steps equation 19.

What the analysis does not cover. The theory assumes an exact correction (A1), a calibrated rollout (A3), and free functions, whereas training uses finite differences, LoRA, and the sample-dependent scale equation 12. With multiple rewards, equation 22 applies only to the combined score. We therefore test the main predictions in Section 5.2: the need for the correction and positive-length intervals, the absence of an update under constant advantages, and consistency across sampling budgets. Appendix A.6 gives an additional variance-reduction result.

Table 1: Text-to-image results on SD3.5-Medium. All rows without a mark are copied from Table 1 of MeanFlowNFT (Huang et al., 2026b), which evaluates at 1024 × 1024 following DiffusionNFT. <sup>†</sup>Evaluated with our pipeline (same prompts, resolution, seeds, and scorers as MFA). Among fewstep models, bold marks the best value and underline the second best. PickScore is the raw logit.
<table><tr><td>Method</td><td>ImageReward↑</td><td>CLIPScore↑</td><td>Aesthetic↑</td><td>PickScore↑</td><td>HPSv2↑</td><td>HPSv3↑</td><td>GenEval2↑</td><td>OCR↑</td></tr><tr><td colspan="9">Multi-step models (40 steps)</td></tr><tr><td>SD3.5-M†</td><td>-0.463</td><td>0.240</td><td>5.178</td><td>20.77</td><td>0.208</td><td>2.769</td><td>0.104</td><td>0.134</td></tr><tr><td>+ DiffusionNFT† (Zheng et al., 2026)</td><td>1.426</td><td>0.299</td><td>5.492</td><td>23.55</td><td>0.330</td><td>13.893</td><td>0.224</td><td>0.639</td></tr><tr><td colspan="9">Few-step models (4 steps)</td></tr><tr><td>DMD (Yin et al., 2024)</td><td>0.924</td><td>0.284</td><td>5.506</td><td>22.28</td><td>0.287</td><td>11.716</td><td>0.204</td><td>0.400</td></tr><tr><td>CDM (Liu et al., 2026)</td><td>1.031</td><td>0.282</td><td>5.572</td><td>22.42</td><td>0.298</td><td>12.519</td><td>0.202</td><td>0.323</td></tr><tr><td>AnyFlow† (Gu et al., 2026)</td><td>1.113</td><td>0.289</td><td>5.420</td><td>22.48</td><td>0.297</td><td>12.069</td><td>0.190</td><td>0.452</td></tr><tr><td>RTDMD (Huang et al., 2026a)</td><td>1.232</td><td>0.278</td><td>6.129</td><td>23.28</td><td>0.327</td><td>13.925</td><td>0.204</td><td>0.297</td></tr><tr><td>Rdm (Fan et al., 2026)</td><td>0.724</td><td>0.272</td><td>5.754</td><td>22.07</td><td>0.276</td><td>11.212</td><td>0.162</td><td>0.376</td></tr><tr><td>DMD + DiffusionNFT</td><td>0.716</td><td>0.284</td><td>5.378</td><td>21.96</td><td>0.271</td><td>9.699</td><td>0.225</td><td>0.487</td></tr><tr><td>CDM + DiffusionNFT</td><td>0.146</td><td>0.275</td><td>4.834</td><td>21.38</td><td>0.214</td><td>-3.283</td><td>0.210</td><td>0.330</td></tr><tr><td>AnyFlow + DiffusionNFT</td><td>1.239</td><td>0.292</td><td>5.949</td><td>23.09</td><td>0.292</td><td>12.138</td><td>0.234</td><td>0.595</td></tr><tr><td>TDM-R1† (Luo et al., 2026)</td><td>1.444</td><td>0.293</td><td>5.942</td><td>22.80</td><td>0.324</td><td>13.512</td><td>0.219</td><td>0.623</td></tr><tr><td>MeanFlowNFT† (Huang et al., 2026b)</td><td>1.446</td><td>0.296</td><td>5.928</td><td>23.50</td><td>0.327</td><td>13.676</td><td>0.218</td><td>0.627</td></tr><tr><td>MeanFlowAdvantage (ours)</td><td>1.451</td><td>0.297</td><td>6.300</td><td>23.63</td><td>0.330</td><td>13.933</td><td>0.229</td><td>0.631</td></tr></table>

![](images/21d24fe51995c4c91e94bbb79360a920edce26a5cf74f3d3966559f16d06792e.jpg)  
Figure 1: Qualitative comparison on prompts from GenEval, OCR, and DrawBench.

![](images/0b2b45fc916ae7751e481478ff5b035984f1e0104763f1a6946121e01469e4a8.jpg)  
(a) Held-out Average Score

![](images/70f4bd89f7833f3c2cbe6d97243ea48f317538b5d633d7f5463525b366d0a105.jpg)  
(b) Held-out Aesthetic Score

![](images/6fc0d2b01b45ddf890f6c6fbc688ddb8d81f0fe87efa3421d631e01ac42d755e.jpg)  
(c) Held-out PickScore  
Figure 2: Held-out DrawBench scores during training, MeanFlowAdvantage versus MeanFlowNFT under the same outer loop.

## 5 EXPERIMENTS

We evaluate MFA as on-policy RL on SD3.5-Medium (Sections 5.1–5.2). Appendix E applies the same objective to DNA promoters, both as teacher-free on-policy RL for a generator defined on a manifold and as reward-graded distillation.

Table 2: Any-step evaluation on DrawBench.
<table><tr><td>NFE</td><td>IR↑</td><td>CLIP↑</td><td>Aes↑</td><td>Pick↑</td><td>HPSv2↑</td><td>HPSv3↑</td><td>GenEval2↑</td><td>OCR↑</td></tr><tr><td>1</td><td>0.489</td><td>0.271</td><td>5.675</td><td>21.692</td><td>0.248</td><td>4.533</td><td>0.160</td><td>0.140</td></tr><tr><td>2</td><td>1.377</td><td>0.288</td><td>6.359</td><td>23.328</td><td>0.316</td><td>12.602</td><td>0.225</td><td>0.532</td></tr><tr><td>4</td><td>1.451</td><td>0.297</td><td>6.300</td><td>23.637</td><td>0.330</td><td>13.933</td><td>0.229</td><td>0.631</td></tr><tr><td>8</td><td>1.458</td><td>0.298</td><td>6.174</td><td>23.713</td><td>0.333</td><td>14.218</td><td>0.236</td><td>0.625</td></tr><tr><td>16</td><td>1.467</td><td>0.298</td><td>6.084</td><td>23.679</td><td>0.332</td><td>14.255</td><td>0.240</td><td>0.621</td></tr><tr><td>32</td><td>1.492</td><td>0.297</td><td>6.045</td><td>23.646</td><td>0.332</td><td>14.207</td><td>0.250</td><td>0.625</td></tr></table>

Steps for Sampling  
![](images/d5a02d9bc7f8e518f575e0e7f5e32ae607adc3c472eec06c36569d5dd6d0c9b4.jpg)  
Figure 3: Qualitative test-time scaling of MeanFlowAdvantage across sampling budgets.

![](images/1d59bafcbb820d463360fa850eb76e4b80be68c103149f8194915d387d45617c.jpg)  
(a) Runtime vs NFE

![](images/17c5cbabbd191745642c565b96a0ed10f9c2d0c2667f2ec4c2d3e866610d51b8.jpg)  
(b) Paired CLIP feature drift

![](images/a033df743f54733ec37752cca3c729a8c5e743366babff3b8c10ffea48b3a7ff.jpg)  
(c) Direct vs composed residua  
Figure 4: Any-step behavior of MFA: runtime, drift of samples across budgets, and semigroup residual of the learned flow map.

## 5.1 TEXT-TO-IMAGE GENERATION WITH SD3.5-MEDIUM

Setup. We start from the merged AnyFlow pretraining and on-policy distillation adapters for SD3.5- Medium (Esser et al., 2024; Gu et al., 2026) and train a fresh LoRA with four-step, CFG-free rollouts. Training uses 512 × 512 images, L = 48 prompt groups with K = 24 images each, and AdamW with learning rate $3 \times 1 0 ^ { - 6 }$ . PickScore, HPSv2, and CLIPScore are the training rewards, with equal weights (Kirstain et al., 2023; Wu et al., 2023; Hessel et al., 2021). ImageReward, Aesthetic Score, HPSv3, GenEval2, and OCR are evaluated but never optimized (Xu et al., 2023). DrawBench (Saharia et al., 2022) is used for online evaluation. The outer loop and hyperparameters match MeanFlowNFT, and Appendix C lists them all.

Main results. Table 1 compares MFA with the published baselines and with five matched evaluations (<sup>†</sup>). Under the matched evaluation, MFA improves on both MeanFlowNFT and TDM-R1 on all eight metrics. Relative to MeanFlowNFT, the largest gains are on Aesthetic Score (+0.37) and HPSv3 (+0.26), neither of which is a training reward, indicating that the improvement is not confined to the optimized objectives. Among all few-step models, MFA ranks first on seven of the eight metrics, with particularly strong results on Aesthetic Score, PickScore, HPSv2, HPSv3, and OCR. With four NFEs, MFA exceeds the 40-step DiffusionNFT on ImageReward, CLIPScore, Aesthetic Score, HPSv2, and HPSv3 and is within 0.01 on PickScore, at one tenth of the sampling cost. DiffusionNFT remains better on GenEval2 and OCR, both of which its multi-reward setup trains on directly. Qualitatively, MFA more often preserves the requested object count, attribute binding, and text (Figure 1). Figure 2 further shows that these gains persist through most of training rather than arising from a single selected checkpoint.

(b) Training PickScore  
(a) Training reward  
(f) Held-out PickScore  
![](images/18574ee99f3b39b19637407f481a8a4e2f75fe479f63d0005590bf952fb6ca04.jpg)

![](images/9cd9501ac9b48f643ae6c44385e96bf47627e23de9401c4de096b6761d357d8b.jpg)

![](images/0c48d7e3689d2a6a7cf8e00c0de382550d4d044ef7a6b2693410f5545748bef3.jpg)

![](images/a3d3f5bd2e827c5c7400cdfc2a81401bb9447f39b7cc42c480a80dbd0feaf511.jpg)

![](images/5c97f45ee89ae685ec4c0ba3f10a4b3d4a06aa9cd487ba6d83dec71af4485722.jpg)  
(e) Held-out aesthetic score

![](images/a62525b4e04cfea28b224d3272490d34fae37c083a9bee21c4bf0fce0822c0e3.jpg)  
Control Separate deriv. γ = 5 γ(A) = 1-A Diffusion-only Direct u No w  
Figure 5: Training reward and held-out score of the matched MFA ablations. Boundary-only training collapses late; the positive-only and A ≡ 1 runs collapse within the first few dozen steps and are stopped at step 300.

Any-step behavior. Theorem 2 predicts that, near its fixed point, one model should serve every sampling budget. Table 2 tests this with the checkpoint trained only with four-step rollouts. One step remains clearly harder. Two steps recover most of the quality, and from four to 32 steps the scores stay on a plateau: ImageReward, GenEval2, and HPSv3 keep improving slightly, while Aesthetic Score peaks at two steps. The model is therefore usable across budgets but is not an exact flow map. The semigroup probe in Figure 4 measures this directly. It compares the direct map $\Phi _ { 1  0 }$ with compositions of N sub-steps (Appendix A.7); the relative residual grows from 0.40 at $N = 2$ to 0.76 at $N = 3 2$ . The fixed point equation 21 is thus only approximately reached. Four steps give a good quality-cost trade-off. Figure 3 shows samples across budgets.

## 5.2 TESTING THE ANALYSIS ON THE IMAGE MODEL

Each ablation changes one component of MFA under the same SD3.5-Medium initialization, fresh LoRA, 48 prompt groups per update, 2000 steps, and seed 42. Figure 5 shows the training curves, and Appendix D.1 reports the endpoints (Table S5). The results broadly follow the predictions of Section 4. (i) The MeanFlow correction matters. Removing the correction while keeping positivelength intervals lowers both the training reward (1.507 vs. 1.549) and held-out aggregate (8.89 vs. 8.98), while roughly doubling the gradient norm. Computing a separate derivative for each network reaches similar quality (9.04) but costs 23% more per step, suggesting that sharing is mainly an efficiency choice in practice, while its theoretical role is to preserve the equivalence in Theorem 1. (ii) Positive-length intervals are necessary. Boundary-only training constrains the instantaneous velocity but leaves the finite-interval maps used by the sampler unconstrained. The run collapses after about 670 steps, ending with a held-out aggregate of 2.96 and a gradient norm above 400, consistent with Theorem 2. (iii) Signed advantages provide the direction. With $A \equiv 1$ , Proposition 1 predicts no reward-directed update, and training collapses within 20 steps; keeping only positive advantages also collapses early (step 22), suggesting that negative samples matter beyond the population reweighting picture. In contrast, increasing γ from 1.1 to 5 or using $\gamma ( A ) = 1 { \overset { \cdot } { - } } A$ gives endpoints close to the control, so the gain does not appear to come from a narrowly tuned trust region. (iv) Adaptive scaling stabilizes optimization. Removing w from equation 12 raises the final gradient norm from 0.97 to 11.97 and lowers the held-out aggregate to 8.70 without immediate collapse. Since w leaves the per-sample minimizer unchanged, its effect is on optimization stability rather than on the target itself.

## 5.3 REWARD-FILTERED ADVANTAGE-WEIGHTED PROMOTER FINE-TUNING

We further test MFA on 1024-bp FANTOM5 promoters with a Riemannian MeanFlow generator and negative Sei profile MSE as reward. In teacher-free on-policy RL, MFA lowers the pretrained Sei MSE from 0.0714 to 0.0482 over three seeds and is statistically indistinguishable from Mean-FlowNFT on the optimized reward, consistent with Proposition 2. For distillation, a frozen 10- step Sei-guided teacher provides improved endpoints; we keep pairs with $\Delta r \geq 0 . 0 0 2$ and use $A = \mathrm { c l i p } ( 1 0 \Delta r , 0 , 1 )$ . On identical pairs and advantages, MFA reaches 0.0467 Sei MSE versus 0.0570 for MeanFlowNFT, an 18% reduction. Full configurations and ablations are in Appendix E.

## 6 CONCLUSION

We introduced MeanFlowAdvantage, a signed advantage-weighted objective that aligns reward finetuning with the average-velocity representation deployed by MeanFlow. A single detached correction shared by the learner, rollout, and reference networks turns the prediction-space quadratic into an exact advantage-weighted MeanFlow regression, while keeping the rollout and reference anchors directly on the deployed finite-interval maps. At the population level, this objective is equivalent to fitting the MeanFlow target of a reward-reweighted distribution; its update direction is governed by an advantage–target covariance, and self-consistent fixed points define exact flow maps that transfer across sampling budgets. Empirically, MFA improves all eight reported metrics over matched MeanFlowNFT on SD3.5-Medium and matches or exceeds the 40-step DiffusionNFT baseline on six of eight with only four NFEs. The same formulation also extends beyond images, supporting both teacher-free RL and reward-graded distillation for DNA promoter generation. Open questions remain around approximate flow-map consistency, principled negative advantages for distillation, and the three-seed sequence-fidelity trend.

## AI USE STATEMENT

Generative AI tools were used to assist with manuscript editing, language refinement, LaTeX formatting, and mathematical consistency checking. They were not used as a substitute for experimental validation or independent verification of the reported results. The authors reviewed and verified all AI-assisted text, derivations, implementation details, references, and experimental claims, and take full responsibility for the final content of the paper.

## ETHICS STATEMENT

The models inherit biases and failure modes of their pretrained generators and reward models. The promoter experiments evaluate public human genomic sequences computationally; Sei scores are not experimental validation of regulatory function.

## REPRODUCIBILITY STATEMENT

The main algorithm and optimization objective are specified in Algorithm 1 and Section 3. Experimental configurations and implementation details are provided in Appendix C and Appendix B, complete proofs and theoretical assumptions are given in Appendix A, and additional image-model ablations and promoter evaluations are reported in Appendices D.1 and E.1. The code is available on https://github.com/HaCTang/MeanFlowAdvantage, and the model weights are available on https://huggingface.co/Haocheng1/CrystAF.

## REFERENCES

Michael S. Albergo, Nicholas M. Boffi, and Eric Vanden-Eijnden. Stochastic interpolants: A unifying framework for flows and diffusions. arXiv preprint arXiv:2303.08797, 2023.

Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. In International Conference on Learning Representations, 2024.

Kathleen M. Chen, Aaron K. Wong, Olga G. Troyanskaya, and Jian Zhou. A sequence-based global map of regulatory activity for deciphering human genetics. Nature Genetics, 54:940–949, 2022.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Muller, Harry Saini, Yam¨ Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. arXiv preprint arXiv:2403.03206, 2024.

Linqian Fan, Peiqin Sun, Tiancheng Wen, Shun Lu, and Chengru Song. Rdm: Re-conceptualizing distribution matching as a reward for diffusion distillation. arXiv preprint arXiv:2603.28460, 2026.

Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. DPOK: Reinforcement learning for fine-tuning text-to-image diffusion models. In Advances in Neural Information Processing Systems, 2023.

Zhengyang Geng, Mingyang Deng, Xingjian Bai, J. Zico Kolter, and Kaiming He. Mean flows for one-step generative modeling. In Advances in Neural Information Processing Systems, 2025.

Yuchao Gu, Guian Fang, Yuxin Jiang, Weijia Mao, Song Han, Han Cai, and Mike Zheng Shou. AnyFlow: Any-step video diffusion model with on-policy flow map distillation. arXiv preprint arXiv:2605.13724, 2026.

Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. CLIPScore: A reference-free evaluation metric for image captioning. In Proceedings ofEMNLP, 2021.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022.

Yushi Huang, Xiangxin Zhou, Ruoyu Wang, Chi Zhang, Jun Zhang, and Tianyu Pang. Reinforcing few-step generators via reward-tilted distribution matching. arXiv preprint arXiv:2605.26108, 2026a.

Yushi Huang, Xiangxin Zhou, Jun Zhang, Liefeng Bo, and Tianyu Pang. MeanFlowNFT: Bringing forward-process RL to average-velocity generators. arXiv preprint arXiv:2607.15273, 2026b.

Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Picka-pic: An open dataset of user preferences for text-to-image generation. In Advances in Neural Information Processing Systems, 2023.

Branislav Kveton, Anup Rao, Subhojyoti Mukherjee, Krishna Kumar Singh, and Viet Dac Lai. AdvantageFlow: Advantage-weighted least squares for RL in flow models. arXiv preprint arXiv:2605.26013, 2026.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In International Conference on Learning Representations, 2023.

Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-GRPO: Training flow matching models via online RL. In Advances in Neural Information Processing Systems, 2025.

Tao Liu, Hao Yan, Mengting Chen, Taihang Hu, Zhengrong Yue, Zihao Pan, Jinsong Lan, Xiaoyong Zhu, Ming-Ming Cheng, Bo Zheng, and Yaxing Wang. Continuous-time distribution matching for few-step diffusion distillation. arXiv preprint arXiv:2605.06376, 2026.

Xingchao Liu, Chengyue Gong, and qiang liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In The Eleventh International Conference on Learning Representations, 2023.

Yihong Luo, Tianyang Hu, Weijian Luo, and Jing Tang. TDM-R1: Reinforcing few-step diffusion models with non-differentiable reward. In International Conference on Machine Learning, 2026.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S. Sara Mahdavi, Rapha Gontijo Lopes, Tim Salimans, Jonathan Ho, David J. Fleet, and Mohammad Norouzi. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 2022.

Hannes Stark, Bowen Jing, Chenyu Wang, Gabriele Corso, Bonnie Berger, Regina Barzilay, and Tommi Jaakkola. Dirichlet flow matching with applications to DNA sequence design. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pp. 46495–46513. PMLR, 2024.

Dongyeop Woo, Marta Skreta, Seonghyun Park, Sungsoo Ahn, and Kirill Neklyudov. Riemannian MeanFlow. arXiv preprint arXiv:2602.07744, 2026.

Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-toimage synthesis. arXiv preprint arXiv:2306.09341, 2023.

Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. ImageReward: Learning and evaluating human preferences for text-to-image generation. In Advances in Neural Information Processing Systems, 2023.

Shuchen Xue, Chongjian Ge, Shilong Zhang, Yichen Li, and Zhi-Ming Ma. Advantage weighted matching: Aligning RL with pretraining in diffusion models. arXiv preprint arXiv:2509.25050, 2025.

Tianwei Yin, Michael Gharbi, Richard Zhang, Eli Shechtman, Fr¨ edo Durand, William T. Freeman,´ and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6613–6623, 2024.

Kaiwen Zheng, Huayu Chen, Haotian Ye, Haoxiang Wang, Qinsheng Zhang, Kai Jiang, Hang Su, Stefano Ermon, Jun Zhu, and Ming-Yu Liu. Diffusion NFT: Online diffusion reinforcement with forward process. In International Conference on Learning Representations, 2026.

## A PROOFS FOR SECTION 4

Setting. Throughout, c is drawn from the prompt distribution, $x _ { 0 } \sim \pi _ { \mathrm { o l d } } ( \cdot \cdot \mid c ) , \epsilon \sim \mathcal { N } ( 0 , I )$ and $( s , t )$ is drawn independently of $( x _ { 0 } , \epsilon )$ We write $x _ { t } = ( 1 - t ) x _ { 0 } + t \epsilon , v _ { t } = \epsilon - x _ { 0 } $ , and $Z = ( x _ { t } , s , t , c )$ . The advantage is a function $A = A ( x _ { 0 } , c )$ . Under $( \mathrm { A } 1 ) , d = D ( Z ) + J ( Z ) v _ { t }$ with $\dot { D } = \partial _ { t } u _ { \mathrm { o l d } } ( Z )$ and $J = \partial _ { x } u _ { \mathrm { o l d } } ( Z )$ ; both are functions of $Z$ only. All second moments are assumed finite. Conditional expectations under $\pi _ { \mathrm { o l d } }$ are written $\mathbb { E } [ \cdot \mid Z ]$

## A.1 AVERAGE-VELOCITY FORM OF THE LOSS AND BOUNDARY REDUCTION

By equation $6 , F _ { \theta } = x _ { t } - t u _ { \theta } - t ( t - s ) d . \mathrm { S i n c e } x _ { t } - x _ { 0 } = t v _ { t } .$

$$
F _ { \theta } - x _ { 0 } = t v _ { t } - t u _ { \theta } - t ( t - s ) d = - t \big ( u _ { \theta } - \bar { u } \big ) , \qquad \bar { u } = v _ { t } - ( t - s ) d .\tag{23}
$$

The anchor residuals follow from equation 6 because the correction terms cancel: $F _ { \theta } - F _ { k } =$ $- t ( u _ { \theta } \mathrm { ~ - ~ } u _ { k } )$ . Substituting both into equation 8 gives equation 10. This identity holds for any value of $d ,$ including the finite difference. ${ \mathrm { A t ~ } } s = t ,$ the factor $t - s$ vanishes, so $\bar { u } \ = \ v _ { t }$ and $F _ { k } = x _ { t } - t u _ { k } ( x _ { t } , t , t )$ for every network. The loss equation 10 then equals the AdvantageFlow loss equation 2 applied to the boundary predictor ${ x } _ { t } - t { u } _ { \theta } ( { x } _ { t } , t , t )$ , for the same samples, advantages, and coefficients. The scaled loss equation 12 reduces in the same way when the comparison uses the same denominator.

## A.2 PROOF OF THEOREM 1

Step 1: the rollout anchor is a MeanFlow regression. Let $g ( Z )$ be any function of Z. Expanding the square and using (A3),

$$
\mathbb { E } \big [ \| g - \bar { u } \| _ { 2 } ^ { 2 } \bigm | Z \big ] = \| g - u _ { \mathrm { o l d } } \| _ { 2 } ^ { 2 } + \mathbb { E } \big [ \| u _ { \mathrm { o l d } } - \bar { u } \| _ { 2 } ^ { 2 } \bigm | Z \big ] + 2 \big \langle g - u _ { \mathrm { o l d } } , u _ { \mathrm { o l d } } - \mathbb { E } [ \bar { u } \mid Z ] \bigm \rangle ,\tag{24}
$$

and the last term is zero. By equation 7, the learner output $u _ { \theta } ( Z )$ is such a function, so

$$
t ^ { 2 } \| u _ { \theta } - u _ { \mathrm { o l d } } \| _ { 2 } ^ { 2 } = \mathbb { E } \big [ t ^ { 2 } \| u _ { \theta } - \bar { u } \| _ { 2 } ^ { 2 } \bigm | Z \big ] - \mathbb { E } \big [ t ^ { 2 } \| u _ { \mathrm { o l d } } - \bar { u } \| _ { 2 } ^ { 2 } \bigm | Z \big ] .\tag{25}
$$

Taking expectations in equation 10 with $\lambda = 0$ gives

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { M F A } } ( \theta ) = \mathbb { E } \big [ ( A + \gamma ) t ^ { 2 } \| u _ { \theta } - \bar { u } \| _ { 2 } ^ { 2 } \big ] + C , \qquad C = - \gamma \mathbb { E } \big [ t ^ { 2 } \| u _ { \mathrm { o l d } } - \bar { u } \| _ { 2 } ^ { 2 } \big ] . } \end{array}\tag{26}
$$

Step 2: change of measure. Let $h ( x _ { 0 } , \epsilon , s , t , c ) = t ^ { 2 } \| u _ { \theta } ( Z ) - \bar { u } \| _ { 2 } ^ { 2 }$ . Because A depends only on $( x _ { 0 } , c )$ and $( \epsilon , s , t )$ are independent of $x _ { 0 }$

$$
\mathbb { E } \left[ \left( A + \gamma \right) h \right] = \mathbb { E } _ { c } \Big [ \int ( \gamma + A ) \pi _ { \mathrm { o l d } } ( x _ { 0 } \mid c ) \mathbb { E } _ { \epsilon , s , t } [ h ] \mathrm { d } x _ { 0 } \Big ] = \mathbb { E } _ { c } \Big [ \big ( \gamma + \bar { A } _ { c } \big ) \mathbb { E } _ { q } \big [ h \mid c \big ] \Big ] ,\tag{27}
$$

where $q$ is defined in equation 16. By (A2), q is a valid probability distribution. This proves equation 17.

Step 3: minimizer. For $\lambda \geq 0$ , the conditional loss given Z is

$$
t ^ { 2 } \Big ( \mathbb { E } \big [ A \| u - \bar { u } \| _ { 2 } ^ { 2 } \mid Z \big ] + \gamma \| u - u _ { \mathrm { o l d } } \| _ { 2 } ^ { 2 } + \lambda \| u - u _ { \mathrm { r e f } } \| _ { 2 } ^ { 2 } \Big ) ,\tag{28}
$$

a quadratic in u with Hessian $2 t ^ { 2 } ( \mathbb { E } [ A \mid Z ] + \gamma + \lambda ) I$ . This is positive definite for $t > 0$ by (A2). Its unique minimizer is

$$
u ^ { + } = \frac { \mathbb { E } [ A \bar { u } \mid Z ] + \gamma u _ { \mathrm { o l d } } + \lambda u _ { \mathrm { r e f } } } { \mathbb { E } [ A \mid Z ] + \gamma + \lambda } .\tag{29}
$$

By (A3), ${ \mathbb E } [ A { \bar { u } } \mid Z ] + \gamma u _ { \mathrm { o l d } } = { \mathbb E } [ ( A + \gamma ) { \bar { u } } \mid Z ]$ . Under $q ,$ the joint density of $( x _ { 0 } , \epsilon , s , t )$ given c is $( \gamma + A ) / ( \gamma + \bar { A } _ { c } )$ times that under $\pi _ { \mathrm { o l d } }$ . By Bayes’ rule, the conditional law given $Z$ is reweighted by $\gamma + A ;$ the normalizer $\gamma + \bar { A } _ { c }$ depends only on $c ,$ which is part of Z. Hence

$$
{ \mathbb E } [ ( A + \gamma ) \bar { u } \mid Z ] = { \mathbb E } [ A + \gamma \mid Z ] { \mathbb E } _ { q } [ \bar { u } \mid Z ] .\tag{30}
$$

Since $D$ and J are functions of $Z , \mathbb { E } _ { q } [ { \bar { u } } \mid Z ] = \mathbb { E } _ { q } [ v _ { t } \mid Z ] - ( t - s ) \big ( D + J \mathbb { E } _ { q } [ v _ { t } \mid Z ] \big )$ . Because $( s , t )$ is independent of $( x _ { 0 } , \epsilon ) , \mathbb { E } _ { q } [ v _ { t } \mid Z ] = v _ { q } ( x _ { t } , t , c )$ , so $\mathbb { E } _ { q } [ \bar { u } \mid Z ] = \dot { u } _ { q }$ . Substituting into equation 29 gives $u ^ { + } = \alpha u _ { q } + ( 1 - \alpha ) \bar { u } _ { \mathrm { r e f } }$ with the stated α. □

Why the correction must be shared. Suppose the learner and the rollout used separate corrections $d _ { \theta }$ and $d _ { \mathrm { o l d } }$ . The rollout residual would be $u _ { \theta } - u _ { \mathrm { o l d } } + ( t - s ) ( d _ { \theta } - d _ { \mathrm { o l d } } )$ , and under $\begin{array} { r l } { \left( \mathrm { A l } \right) d _ { \theta } - d _ { \mathrm { o l d } } = } \end{array}$ $\partial _ { t } ( u _ { \theta } - u _ { \mathrm { o l d } } ) + \partial _ { x } ( u _ { \theta } - u _ { \mathrm { o l d } } ) v _ { t }$ . This residual depends on $v _ { t }$ and not only on $Z ,$ so Step 1 fails: the cross term no longer vanishes, and the anchor is no longer a MeanFlow regression on rollout samples.

## A.3 PROOF OF PROPOSITION 1

Subtracting $u _ { \mathrm { o l d } }$ from equation 29,

$$
u ^ { + } - u _ { \mathrm { o l d } } = \frac { \mathbb { E } [ A \bar { u } \mid Z ] - \mathbb { E } [ A \mid Z ] u _ { \mathrm { o l d } } + \lambda ( u _ { \mathrm { r e f } } - u _ { \mathrm { o l d } } ) } { \mathbb { E } [ A \mid Z ] + \gamma + \lambda } .\tag{31}
$$

By $( \operatorname { A } 3 ) , \operatorname { \mathbb { E } } [ A { \bar { u } } \mid Z ] - \operatorname { \mathbb { E } } [ A \mid Z ] \operatorname { \mathbb { E } } [ { \bar { u } } \mid Z ] = \operatorname { C o v } ( A , { \bar { u } } \mid Z )$ , which gives the identity in equation 19. For the bound, take any unit vector e. By Cauchy–Schwarz, $e ^ { \top } \bar { \mathrm { C o v } } ( A , \bar { u } \mid Z ) \stackrel { \cdot } { = } \mathrm { C o v } ( A , e ^ { \top } \bar { u }$ | $Z ) \leq \sigma _ { A } \sqrt { \mathrm { V a r } ( e ^ { \top } \bar { u } \mid Z ) } \leq \sigma _ { A } \sigma _ { \bar { u } }$ . Taking the supremum over e bounds the norm of the covariance vector. The triangle inequality and the positivity of the denominator (A2) complete the proof. □

## A.4 PROOF OF PROPOSITION 2 AND THE RELATION TO PREDICTION-SPACE NFT

MeanFlowNFT forms the implicit positive and negative velocities $V ^ { \pm } = V _ { \mathrm { o l d } } \pm \beta ( V _ { \theta } - V _ { \mathrm { o l d } } )$ and minimizes r $| V ^ { + } - v _ { t } | | _ { 2 } ^ { 2 } + ( 1 - r ) \lVert V ^ { ^ { \perp } } - v _ { t } \rVert _ { 2 } ^ { 2 }$ . With the shared correction, $V _ { \theta } { - } V _ { \mathrm { o l d } } = u _ { \theta } { - } u _ { \mathrm { o l d } } = : \Delta$ a function of $Z ,$ , and $V _ { \mathrm { o l d } } - v _ { t } = u _ { \mathrm { o l d } } - \bar { u } = : e$ . Hence

$$
r \| \beta \Delta + \epsilon \| _ { 2 } ^ { 2 } + ( 1 - r ) \| - \beta \Delta + \epsilon \| _ { 2 } ^ { 2 } = \beta ^ { 2 } \| \Delta \| _ { 2 } ^ { 2 } + 2 \beta A \langle \Delta , e \rangle + \| e \| _ { 2 } ^ { 2 } , \qquad A = 2 r - 1 .\tag{32}
$$

Conditioning on $Z$ and minimizing over ∆ gives $\Delta = - \mathbb { E } [ A e \mid Z ] / \beta = ( \mathbb { E } [ A { \bar { u } } | Z ] - \mathbb { E } [ A |$ $Z ] u _ { \mathrm { o l d } } ) / \beta . \ \mathbf { B y } \left( \mathbf { A } 3 \right)$ , this equals $\operatorname { C o v } ( A , \bar { u } \mid Z ) / \beta$ , which proves equation 20. □

The positive-policy example. Let $A = 2 r - 1$ and $\gamma = 1$ . Then $\gamma + A = 2 r$ and q ∝ $r \pi _ { \mathrm { o l d } } = \pi ^ { + }$ Write $\rho = \mathbb { E } [ r \mid Z ]$ and $u _ { \pi ^ { + } } = \mathbb { E } _ { \pi ^ { + } } [ \bar { u } \mid Z ]$ . By $( \mathbb { A } 3 ) , \operatorname { C o v } ( A , { \bar { u } } \mid Z ) = 2 \big ( \mathbb { E } [ r { \bar { u } } \mid Z ] - \rho u _ { \mathrm { o l d } } \big ) =$ $2 \rho \left( u _ { \pi ^ { + } } - u _ { \mathrm { o l d } } \right)$ . The NFT minimizer is therefore $u _ { \mathrm { o l d } } + ( 2 \rho / \beta ) ( u _ { \pi ^ { + } } - u _ { \mathrm { o l d } } )$ . It equals $u _ { \pi ^ { + } }$ + only where $2 \rho ( Z ) = \beta$ . By Theorem 1, the MFA minimizer is $u _ { \pi ^ { + } }$ at every $Z .$

Prediction-space form. Multiplying by $t ^ { 2 }$ and dropping θ-independent terms, equation 32 is equivalent to $\beta A \| \bar { F } _ { \theta } - x _ { 0 } \| _ { 2 } ^ { 2 } + \beta ( \bar { \beta } - A \bar { ) } \| \bar { F } _ { \theta } - F _ { \mathrm { o l d } } \| _ { 2 } ^ { 2 } .$ . After dividing by $\beta ,$ , NFT is the MFA quadratic with the sample-dependent anchor weight $\gamma ( A ) \overset { \vartriangle } { = } \beta - A$ and constant total curvature $\beta . \ \mathrm { A t } \ \beta = 1$ it matches the variant $\gamma ( A ) = 1 - A$ with $\lambda = 0$ . Separate adaptive denominators for the positive and negative branches break this equivalence. The same expansion, with the data target $y _ { \kappa }$ , anchor $b ,$ and reference weight k, gives the NFT optimum in equation 39.

## A.5 FLOW-MAP CONSISTENCY AND PROOF OF THEOREM 2

For a velocity field $v ,$ define $\mathcal { T } _ { v } [ u ] = u + ( t - s ) \big ( \partial _ { t } u + ( \partial _ { x } u ) v \big )$

Lemma 1 (Flow-map consistency). Suppose v admits a unique flow on $[ 0 , 1 ]$ , and u is continuously differentiable $f o r s < t$ t and bounded as $t \downarrow s . I f \mathcal { T } _ { v } [ u ] = \iota$ along every trajectory, then $u ( x _ { t } , s , t ) =$ $( x _ { t } - x _ { s } ) / ( t - s )$ , where $x _ { \tau }$ is the trajectory of v through $x _ { t }$

Proof of the lemma. Fix s and let $U ( \tau ) = \left( \tau - s \right) u ( x _ { \tau } , s , \tau )$ for $\tau > s$ . By the chain rule and the hypothesis, $\mathrm { d } U / \mathrm { d } \tau = \mathcal { T } _ { v } [ u ] ( x _ { \tau } , s , \tau ) = v ( x _ { \tau } , \tau )$ . Boundedness of u gives $U ( \tau ) \to 0 \mathrm { a s } \tau \downarrow s .$ Integrating from s to t yields $\begin{array} { r } { ( t - s ) u ( x _ { t } , s , t ) = \int _ { s } ^ { \tau } v ( x _ { \tau } , \tau ) \mathrm { d } \tau = x _ { t } - x _ { s } . } \end{array}$ □

Proof of Theorem 2. Condition equation 21 says ${ \mathcal { T } } _ { v _ { q } } [ u ^ { * } ] = v _ { q }$ . By Lemma 1, $x _ { t } \ - \ ( t \ -$ $s ) u ^ { * } ( x _ { t } , s , t ) \ = \ x _ { s }$ on every trajectory of $v _ { q } ,$ so each step of equation 13 with $u ^ { * }$ is the exact flow map of $v _ { q }$ over its interval. A composition over any grid is therefore the exact flow from $t = 1$ to $t = 0$ . The rectified-flow velocity $v _ { q }$ transports $\mathcal { N } ( \bar { 0 } , \bar { I } )$ at $t = 1$ to q at $t = 0$ , so the sampler draws from $q$ for every N. For the reward,

$$
\mathbb { E } _ { q } [ R \mid c ] - \mathbb { E } _ { \pi _ { \mathrm { o l d } } } [ R \mid c ] = \frac { \mathbb { E } _ { \pi _ { \mathrm { o l d } } } [ ( \gamma + A ) R \mid c ] } { \gamma + \bar { A } _ { c } } - \mathbb { E } _ { \pi _ { \mathrm { o l d } } } [ R \mid c ] = \frac { \mathrm { C o v } _ { \pi _ { \mathrm { o l d } } } ( A , R \mid c ) } { \gamma + \bar { A } _ { c } } .\tag{33}
$$

If $A = \phi ( R )$ with ϕ nondecreasing, Chebyshev’s association inequality gives $\operatorname { C o v } ( \phi ( R ) , R ) \geq 0 \quad$ The clipped, prompt-centered advantage equation 14 is of this form for a scalar reward when the prompt mean and the global scale are treated as fixed. □

## A.6 VARIANCE REDUCTION OF THE SHARED ANCHOR

Proposition 3 (Shared-anchor variance reduction). Freeze the rollout model and the shared correction, and let $J _ { \theta } = - t \partial _ { \theta } u _ { \theta } ( Z ) . I f \mathbb { E } [ F _ { \mathrm { o l d } } - x _ { 0 } \mid Z ] = 0 ,$ , then

$$
\mathbb { E } \Vert F _ { \theta } - x _ { 0 } \Vert _ { 2 } ^ { 2 } = \mathbb { E } \Vert F _ { \theta } - F _ { \mathrm { o l d } } \Vert _ { 2 } ^ { 2 } + C\tag{34}
$$

for a θ-independent constant C. Moreover, the anchor gradient $2 J _ { \theta } ^ { \top } ( F _ { \theta } - F _ { \mathrm { o l d } } )$ is the conditional expectation, given Z, of the regression gradient $2 J _ { \theta } ^ { \top } ( F _ { \theta } - x _ { 0 } )$ , so its covariance is no larger in the positive-semidefinite order.

Proof. The condition is (A3) written in prediction space, since $F _ { \mathrm { o l d } } - x _ { 0 } = - t ( u _ { \mathrm { o l d } } - \bar { u } )$ . Let $e = F _ { \mathrm { o l d } } - x _ { 0 }$ . By equation $7 , F _ { \theta } - F _ { \mathrm { o l d } }$ is a function of Z, so $\begin{array} { r } { \mathbb { E } \langle F _ { \theta } - F _ { \mathrm { o l d } } , e \rangle = \mathbb { E } \langle F _ { \theta } - F _ { \mathrm { o l d } } , \mathbb { E } [ e \ | } \end{array}$ $Z ] \rangle = 0 .$ . Expanding $\lVert \boldsymbol { F } _ { \boldsymbol { \theta } } - \boldsymbol { F } _ { \mathrm { o l d } } + \boldsymbol { e } \rVert _ { 2 } ^ { 2 }$ gives equation 34 with $C = \mathbb { E } \Vert e \Vert _ { 2 } ^ { 2 }$ . For the gradients, let $G = 2 J _ { \theta } ^ { \top } ( F _ { \theta } - x _ { 0 } )$ and $\widetilde { G } = 2 J _ { \theta } ^ { \top } ( F _ { \theta } - F _ { \mathrm { o l d } } )$ . Both $J _ { \theta }$ and $F _ { \theta } - F _ { \mathrm { o l d } }$ are functions of Z, so $\mathbb { E } [ G \mid Z ] = { \widetilde { G } }$ . The law of total covariance gives $\operatorname { C o v } ( G ) = \operatorname { C o v } ( { \tilde { G } } ) + \mathbb { E } [ \operatorname { C o v } ( G \mid Z ) ] \succeq \operatorname { C o v } ( { \tilde { G } } )$ □

This statement concerns the unscaled rollout anchor only. It does not imply lower variance for the full signed-advantage update with adaptive scaling.

## A.7 SEMIGROUP RESIDUAL

For $\Phi _ { s , t } ^ { u } ( x ) = x - ( t - s ) u ( x , s , t )$ , an exact flow map satisfies

$$
\boldsymbol { \Phi } _ { r , t } ^ { u } ( \boldsymbol { x } ) = \boldsymbol { \Phi } _ { r , s } ^ { u } \big ( \boldsymbol { \Phi } _ { s , t } ^ { u } ( \boldsymbol { x } ) \big ) , \qquad r < s < t .\tag{35}
$$

By Lemma 1, a map that satisfies the fixed point equation 21 also satisfies equation 35. The residual of equation 35 therefore measures how far a trained model is from that fixed point. Figure 4 reports the relative latent RMS distance between the direct map $\Phi _ { 1  0 }$ and uniform N-step compositions, which is zero by definition at $N = 1$ . A small residual is not by itself evidence of reward improvement.

## B IMPLEMENTATION NOTES

Algorithm 1 One MeanFlowAdvantage update (image model)   
Require: learner θ, frozen rollout $\theta _ { \mathrm { o l d } }$ and reference $\theta _ { \mathrm { r e f } } ;$ rewards; $\gamma , \lambda ;$ group size $K ;$ steps N   
1: Sample prompts; generate K images per prompt with equation 13   
2: Score the images; compute clipped signed advantages with equation 14   
3: for each detached $( x _ { 0 } , \bar { A } , c )$ do   
4: Draw $( s , t )$ by equation 15 and ϵ; set $x _ { t } = ( 1 - t ) x _ { 0 } + t \epsilon$   
5: If $s = t ,$ set $d = 0 ;$ otherwise compute d once from $u _ { \mathrm { o l d } }$ by equation 5   
6: Form $V _ { k } = u _ { k } + ( t - s ) d$ and $\ddot { F _ { k } } \dot { = } x _ { t } - t \dot { V _ { k } }$ for $k \in \{ \theta , \mathrm { o l d } ,$ ref}   
7: Compute $\ell _ { \theta } / w$ with equation 8 and equation 12   
8: end for   
9: Update θ on the average loss   
10: Refresh the rollout model: $\theta _ { \mathrm { o l d } }  \rho \theta _ { \mathrm { o l d } } + ( 1 - \rho ) \theta$

Scheduler units and finite differences. With $T = 1 0 0 0 , t _ { \mathrm { r a w } } = T t { \mathrm { a n d } } s _ { \mathrm { r a w } } = T s$ , the normalized and raw-time expressions agree:

$$
x _ { t } = ( 1 - t ) x _ { 0 } + t \epsilon , \qquad F = x _ { t } - t V , \qquad ( t _ { \mathrm { r a w } } - s _ { \mathrm { r a w } } ) d _ { \mathrm { r a w } } = ( t - s ) D _ { t } u .\tag{36}
$$

The reported finite-difference increment of 5 is in raw scheduler units, corresponding to $\delta = 0 . 0 0 5 .$ Spatial displacements must therefore use $\delta \boldsymbol { v } _ { t }$ , not $5 { v } _ { t }$ . The stencil equation 5 assumes $s \leq t - \delta$ and $t + \delta \leq 1$ . Near a boundary, the stencil must be shortened or replaced by a valid one-sided difference; independently clamping its endpoints while retaining 2δ in the denominator is not the same estimator. On $s = t ,$ , the correction is zero and the stencil is skipped.

Advantage and loss scaling. The final multi-reward advantage is clipped after aggregation. For constant γ, a lower bound $A \ge - \gamma - \lambda + \varepsilon _ { A }$ enforces positive curvature; the reported $\gamma = 1 . 1$ $\lambda = 1 0 ^ { - 3 }$ and $A \in [ - 1 , 1 ]$ already satisfy it. For $\gamma ( { \bar { A } } ) = 1 - A$ , the curvature is $1 + \lambda$ and no additional floor is necessary. The shared denominator in equation 12 is evaluated in float64. It preserves the relative weights of the three terms within a sample, but changes the relative weights of different samples. It neither equalizes the three residual magnitudes nor preserves the unweighted population regression problem in general.

Ablations to isolate the construction. The direct-output control uses $F ^ { \mathrm { d i r e c t } } = x _ { t } - t u _ { \theta }$ , retaining the same clean target, regularizers, and time distribution. The diffusion-only control sets $\rho _ { \mathrm { d i f f } } ~ =$ 1. Prediction-space and velocity-space losses must be compared with their effective time weights reported: without other changes they differ by $t ^ { 2 } .$ . The completed matched endpoints for these controls are reported in Appendix Table S5.

## C EXPERIMENTAL DETAILS

Tables S1, S2 and S4 collect the reported configurations, and Table S3 the per-seed DNA RL results. The image baseline follows the published Stage-3 protocol. Both methods use $K = 2 4$ samples per prompt; the corresponding EMA schedule, evaluation resolution, prompt lists, and checkpoint identifiers are fixed according to the reported Stage-3 protocol.

Table S1: SD3.5 Medium RL (Stage 3 protocol of MeanFlowNFT). Sampling and reward choices follow the baseline; the objectives and regularization coefficients differ.
<table><tr><td></td><td>MeanFlowNFT</td><td>MeanFlowAdvantage</td></tr><tr><td>Init</td><td></td><td>AnyFlow + fresh LoRA AnyFlow + fresh LoRA</td></tr><tr><td>LoRA rank / α</td><td>32 / 64</td><td>32 / 64</td></tr><tr><td>Image size (train)</td><td>512</td><td>512</td></tr><tr><td>Rollout NFE / CFG</td><td>4 /none</td><td>4 / none</td></tr><tr><td>Prompt groups L per update</td><td>48</td><td>48</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Learning rate</td><td> $3 \times 1 0 ^ { - 6 }$ </td><td> $3 \times 1 0 ^ { - 6 }$ </td></tr><tr><td> $( s , t )$  mix  $( \rho _ { \mathrm { d i f f } } , \rho _ { \mathrm { c o n s } } )$ </td><td>(0.5,0.25)</td><td>(0.5,0.25)</td></tr><tr><td>Central-difference increment (raw units)</td><td>5</td><td>5</td></tr><tr><td>Shared  $D _ { t }$ </td><td>yes</td><td>yes</td></tr><tr><td>Train rewards</td><td>Pick / HPS / CLIP</td><td>Pick / HPS / CLIP</td></tr><tr><td>β (NFT)</td><td>0.1</td><td></td></tr><tr><td> $\gamma / \lambda \left( \mathrm { L S } \right)$ </td><td></td><td> $1 . 1 / 1 0 ^ { - 3 }$ </td></tr><tr><td>Reference coefficient (objective-specific)</td><td> $1 0 ^ { - 4 }$ </td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td>Advantage clip</td><td>[−1,1]</td><td>[−1,1]</td></tr></table>

Table S2: FANTOM5 on-policy DNA RL (no teacher). Both methods share every row above the line; only the loss coefficients differ.
<table><tr><td></td><td>MeanFlowNFT</td><td>MeanFlowAdvantage</td></tr><tr><td>Init</td><td>RMF ckpts/0120</td><td>RMF ckpts/0120</td></tr><tr><td>Signals per update</td><td>8</td><td>8</td></tr><tr><td>Group size K</td><td>8</td><td>8</td></tr><tr><td>Rollout NFE</td><td>1</td><td>1</td></tr><tr><td>(s, t) mix (boundary, s=1, general)</td><td>(0.5, 0.25, 0.25)</td><td>(0.5,0.25, 0.25)</td></tr><tr><td>t sampler</td><td>logit-normal(−0.4, 1)</td><td>logit-normal(—0.4, 1)</td></tr><tr><td>Noise draws per sample</td><td>2</td><td>2</td></tr><tr><td>Shared correction (JVP)</td><td>yes</td><td>yes</td></tr><tr><td>Tangent clip</td><td>100</td><td>100</td></tr><tr><td>Adaptive scale exponent p</td><td>0.5</td><td>0.5</td></tr><tr><td>Learning rate</td><td>10⁻4</td><td> $1 0 ^ { - 4 }$ </td></tr><tr><td>Grad clip</td><td>0.5</td><td>0.5</td></tr><tr><td>Rollout EMA ρ</td><td>0.9</td><td>0.9</td></tr><tr><td>Advantage clip</td><td>[−1,1]</td><td>[−1,1]</td></tr><tr><td>Train steps (eval interval)</td><td>600 (25)</td><td>600 (25)</td></tr><tr><td>Seeds</td><td>123, 7,42</td><td>123, 7,42</td></tr><tr><td>β / reference weight</td><td>1.0 / 0.01</td><td></td></tr><tr><td>γ/ λ</td><td></td><td>1.1 /0.01</td></tr></table>

Table S3: Per-seed on-policy DNA results behind Table S6. Checkpoints are selected on valid (n=256) and reported on test (n=512), both with eval seed 7. ∆ is the paired NFT−MFA test Sei MSE with a 95% sequence bootstrap interval; positive favors MFA.
<table><tr><td rowspan="2">Seed</td><td colspan="2">MeanFlowNFT</td><td colspan="2">MeanFlowAdvantage</td><td colspan="2"></td></tr><tr><td>Sei MSE↓</td><td>6-mer↑</td><td>Sei MSE↓</td><td>6-mer↑</td><td colspan="2">∆ [95% CI]</td></tr><tr><td>123</td><td>0.0445</td><td>0.927</td><td>0.0437</td><td>0.952</td><td></td><td>0.0008 [-0.0031, 0.0046]</td></tr><tr><td>7</td><td>0.0526</td><td>0.940</td><td>0.0537</td><td>0.954</td><td>-0.0012</td><td>[-0.0051, 0.0026]</td></tr><tr><td>42</td><td>0.0441</td><td>0.944</td><td>0.0472</td><td>0.953</td><td>-0.0031</td><td>[-0.0069, 0.0005]</td></tr><tr><td>Mean</td><td>0.0471</td><td>0.937</td><td>0.0482</td><td>0.953</td><td colspan="2"></td></tr><tr><td>Std.</td><td>0.0048</td><td>0.009</td><td>0.0051</td><td>0.001</td><td colspan="2"></td></tr></table>

All three Sei MSE intervals contain zero, so the two objectives are statistically indistinguishable on the training reward, as Proposition 2 predicts. The 6-mer ordering is consistent across seeds but is a three-seed trend, not a tested claim.

Table S4: FANTOM5 reward-filtered promoter distillation. Shared rows are identical for both losses; method columns list only the knobs that differ.
<table><tr><td></td><td>MeanFlowNFT</td><td>MeanFlowAdvantage</td></tr><tr><td>Init</td><td>RMF ckpts/0120</td><td> $\mathtt { R M F } \mathtt { c k p t s } / 0 1 2 0$ </td></tr><tr><td>Sequence length Train / valid / test chr.</td><td>1024 bp not  $8 – 1 0 / 1 0 / 8 – 9$ </td><td>1024 bp  $\mathrm { n o t 8 – 1 0 / 1 0 / 8 – 9 }$ </td></tr><tr><td>Batch size</td><td>8</td><td> $^ 8$ </td></tr><tr><td>Student NFE</td><td>1 (unguided)</td><td>1 (unguided)</td></tr><tr><td>Teacher NFE / guidance</td><td>10 / 10</td><td> $1 0 / 1 0$ </td></tr><tr><td>Teacher setting</td><td> $x _ { 1 }$  look-ahead</td><td> $x _ { 1 }$  look-ahead</td></tr><tr><td>Accept if  $\Delta r _ { \mathrm { S e i } }$ </td><td> $\geq 0 . 0 0 2$ </td><td> $\geq 0 . 0 0 2$ </td></tr><tr><td>Query time t</td><td>0</td><td>0</td></tr><tr><td>Advantage</td><td> $\mathrm { c l i p } ( 1 0 \Delta r , 0 , 1 )$ </td><td> $\mathrm { c l i p } ( 1 0 \Delta r , 0 , 1 )$ </td></tr><tr><td>Rollout EMA coefficient</td><td>1 (frozen)</td><td>1 (frozen)</td></tr><tr><td>Seed / eval seed</td><td>123 / 7</td><td>123 /7</td></tr><tr><td>Learning rate</td><td> $2 . 5 \times 1 0 ^ { - 5 }$ </td><td></td></tr><tr><td>Grad clip</td><td></td><td> $2 . 5 \times 1 0 ^ { - 5 }$ </td></tr><tr><td></td><td>0.5</td><td>0.5</td></tr><tr><td> $\beta$ </td><td>0.8</td><td></td></tr><tr><td> $\gamma / \lambda$ </td><td></td><td>0.01/ 0.01</td></tr><tr><td>Target extrapolation</td><td>1.5</td><td>κ = 1.5</td></tr><tr><td>Max steps (selection)</td><td>100 (90)</td><td>100 (90)</td></tr></table>

Both DNA losses act on continuous $\mathrm { R M F } \ x _ { 1 }$ endpoint coordinates, using the convention of noise at $t = 0$ and data at $t = 1$ This differs from the image convention in equation 1. The learner calls the RMF endpoint predictor at $( t , s ) = ( 0 , 1 )$ ; its data target is the guided endpoint, optionally extrapolated from the frozen rollout endpoint as specified in equation 38. Thus this adaptation does not use $F = x _ { t } - t V$ at image time $t = 0$ , which would have zero residual.

Figure S3b–c evaluates both methods every 10 steps through step 100. Under the EMA convention of Algorithm 1, $\rho = 1$ freezes the rollout parameters, while the student remains trainable. Evaluation samples from the updated learner; the frozen rollout model is used only to construct training pairs and anchor predictions.

## D ADDITIONAL TEXT-TO-IMAGE RESULTS

## D.1 MATCHED IMAGE-MODEL ABLATIONS

The complete matched ablation endpoints used by the discussion in Section 5.2 are reported below.   
These experiments provide robust directional insights that fully reinforce our primary findings.

For completed runs, training statistics are averages over the final 50 optimization steps and the heldout score is the final fixed-interval online evaluation (step 1950). The two unstable positive-only controls were stopped at step 300 after the predefined collapse screen; their final online evaluation is at step 250.

Figure S2 reports the optimization diagnostics for these matched runs. The secondary comparisons below complement the three core findings in Section 5.2.

Sharing the derivative is the efficient default. Computing a separate derivative correction for learner, rollout, and reference models reaches essentially the same reward regime as the shared correction, but increases wall-clock time per step from 73.3 to 89.9 seconds. Its slightly higher single-seed held-out aggregate is too small to establish a quality advantage. This matches the algebra in equation 7: sharing the correction makes the anchor residuals exact differences of the deployed average-velocity networks while avoiding two additional derivative estimates.

The anchor coefficient is not the main source of the gain. Increasing the fixed anchor coefficient from $\gamma = 1 . 1 \mathrm { t o } \gamma = 5$ leaves the final reward and held-out aggregate nearly unchanged, while the dynamic $\gamma ( A ) = 1 - A$ variant is also stable and reaches a comparable endpoint. These single-seed differences are too small to rank the stable variants, but they indicate that the principal effect is not a narrowly tuned value of $\gamma .$ We therefore retain $\gamma = 1 . 1$ as the conservative default and view $\gamma ( A ) = \dot { 1 - A }$ as a useful follow-up rather than a claimed improvement.

![](images/7ce2d0d2b7ed2922ebb95b99952b4a49926e12e31399194a7d2080d0463b4ce4.jpg)  
Figure S1: More Qualitative Comparison Cases. The prompts are taken from GenEval, OCR and DrawBench respectively,where we compare the corresponding MeanFlowNFT model with our model.

Table S5: Matched image-model ablations. “Reward” is the mean training reward over the final 50 recorded steps of each run; “Held-out Sum” is the fixed DrawBench online aggregate used for the ablation study. The early-stopped positive-only runs are not equal-budget endpoints.
<table><tr><td>Variant</td><td>Train step</td><td>Reward↑</td><td>Held-out Sum↑</td><td>Grad. norm↓</td><td>s/step↓</td></tr><tr><td>Shared induced V (control)</td><td>2000</td><td>1.5493</td><td>8.9841</td><td>0.97</td><td>73.3</td></tr><tr><td>Direct u (no induced correction)</td><td>2000</td><td>1.5071</td><td>8.8915</td><td>2.10</td><td>64.8</td></tr><tr><td>s = t diffusion-only</td><td>2000</td><td>0.8734</td><td>2.9590</td><td>445.79</td><td>65.4</td></tr><tr><td>Separate derivative</td><td>2000</td><td>1.5486</td><td>9.0382</td><td>1.00</td><td>89.9</td></tr><tr><td>No adaptive w</td><td>2000</td><td>1.4959</td><td>8.6952</td><td>11.97</td><td>87.4</td></tr><tr><td>γ = 5</td><td>2000</td><td>1.5424</td><td>8.9936</td><td>0.84</td><td>72.9</td></tr><tr><td>Positive A only</td><td>300†</td><td>0.7956</td><td>2.3275</td><td>1.30</td><td>73.4</td></tr><tr><td>A ≡ 1</td><td>300†</td><td>0.8260</td><td>3.7154</td><td>24.17</td><td>73.6</td></tr><tr><td> $\gamma ( A ) = 1 - A$ </td><td>2000</td><td>1.5482</td><td>9.0378</td><td>0.93</td><td>73.1</td></tr></table>

<sup>†</sup>Early-stopped after collapse screening; held-out evaluation is from step 250. The ablation aggregate uses the online normalized scoring convention and is not on the same numerical scale as the raw PickScore column in Table 1.

Positive local curvature is necessary but not sufficient. All stable configurations maintain $A + \gamma +$ $\lambda > 0 .$ , as required by the pointwise quadratic analysis in Section 3.2. However, the s = t diffusiononly run also retains a positive minimum coefficient while its data, reference, and gradient terms diverge late in training. The ablation therefore clarifies the scope of the theory: positive curvature prevents the per-sample free-prediction quadratic from becoming concave, but it cannot by itself guarantee neural-network optimization stability or cross-interval flow consistency.

## E DNA PROMOTERS: ON-POLICY RL AND REWARD-GRADED DISTILLATION

The DNA experiments ask two questions. Does MFA work as on-policy RL for a generator whose state space is a manifold rather than a latent grid? And must the data targets come from the model’s own rollouts at all? Nothing in Section 4 requires the latter: per input, the loss equation 10 needs only a target, a scalar weight, and an anchor.

![](images/8ec8d1eedf56749e84cd9b68f1d1b193d0a951e2180740c032b584be0a16afd8.jpg)  
Figure S2: Optimization stability across MeanFlowAdvantage ablations, highlighting gradient norms, loss dynamics, and collapse behavior.

Shared setup. We use a Riemannian MeanFlow (RMF) generator (Stark et al., 2024; Woo et al., 2026) pretrained locally on 1024-bp FANTOM5 promoters and conditioned on the regulatory signal; the reward is the negative Sei profile MSE (Chen et al., 2022). RMF places noise at time 0 and data at 1, so intervals run from t to $s \geq t$ and the average velocity lies in the tangent space of the sphere at $x _ { t }$ . We train on the training chromosomes, select checkpoints on 256 validation sequences (chr10) by Sei MSE, and report 512 test sequences (chr8–9), evaluated one-step with a fixed prior seed and hard one-hot decoding. The pretrained generator reaches 0.0714 Sei MSE and 0.928 6-mer correlation.

On-policy RL. MFA is used exactly as in Section 3, with no teacher. Per signal, the frozen rollout model draws $K = 8$ one-step sequences, Sei scores them, and equation 14 standardizes the rewards within the group, so a sequence below its group average gets $A < 0 .$ . Each sequence is re-noised and the loss equation 10 is applied in the tangent space at $x _ { t } ,$ with u¯ and the shared correction from the rollout network as in RMF pretraining; intervals follow equation 15. MeanFlowNFT sees the same rollouts, advantages, intervals, and correction, differing only in the loss. Tables S2 and S3 give the configuration and per-seed numbers.

Two things stand out (Table ${ \mathrm { S 6 } } ,$ three seeds). First, teacher-free RL works on this manifold generator: MFA cuts the Sei MSE from 0.0714 to 0.0482 on average, and its best seed reaches 0.0437, below the distilled models below. Second, MFA and NFT are statistically indistinguishable on the training reward: all three paired intervals contain zero. This is what Proposition 2 predicts, since a rollout model that tracks the learner leaves only a gain difference between the two objectives. They differ in the statistic that is not optimized: MFA holds a 6-mer correlation of $0 . 9 5 3 \pm 0 . 0 0 1$ against $0 . 9 3 7 \pm 0 . 0 0 9$ , in all three seeds. We report this as a consistent trend over three seeds, not a tested claim.

Reward-graded distillation. We now replace the on-policy target by a teacher, the setting of scientific design, where a strong but slow sampler exists and the goal is a one-step generator that keeps its gains. From the same noise, a frozen one-step rollout gives an endpoint b and a frozen ten-step Sei-guided teacher gives $g .$ Their reward gap decides both whether a pair is used and how strongly:

$$
\Delta r = r _ { \mathrm { S e i } } ( g ) - r _ { \mathrm { S e i } } ( b ) , \qquad \mathrm { k e e p t h e ~ p a i r ~ i f } \Delta r \geq \tau , \qquad A = \mathrm { c l i p } ( \eta \Delta r , 0 , 1 ) ,\tag{37}
$$

with $\tau = 0 . 0 0 2$ and $\eta = 1 0$ . For each kept pair, the target is extrapolated past the teacher, and the rollout endpoint serves as both anchors:

$$
y _ { \kappa } = b + \kappa ( g - b ) , \qquad \ell _ { \mathrm { D N A } } = A \| y _ { \theta } - y _ { \kappa } \| _ { 2 } ^ { 2 } + ( \gamma + \lambda ) \| y _ { \theta } - b \| _ { 2 } ^ { 2 } ,\tag{38}
$$

with $\kappa = 1 . 5$ and $\gamma = \lambda = 0 . 0 1$ . MeanFlowNFT receives the same pairs, advantages, extrapolation, learning rate, and validation protocol through its own loss, with $\beta \ : = \ : 0 . 8$ and reference weight $k = 0 . 0 1$

Why graded advantages separate the two losses. As in equation 11, each pair has a closed-form optimum. For MFA and for NFT (via the expansion in Appendix A.4),

$$
y _ { \mathrm { M F A } } ^ { \star } - b = \frac { \kappa { \cal A } } { { \cal A } + \gamma + \lambda } ( g - b ) , \qquad y _ { \mathrm { N F T } } ^ { \star } - b = \frac { \kappa { \cal A } } { \beta + k } ( g - b ) .\tag{39}
$$

MFA separates where a pair points from how much it counts: since $\gamma + \lambda = 0 . 0 2$ , every accepted pair targets nearly the full extrapolated endpoint and A mainly sets the curvature $A + \gamma + \lambda$ , that is, how hard the pair pulls. In NFT the advantage rescales the target itself, so pairs with a small but real improvement barely move while pairs with A ≈ 1 overshoot to $1 . 8 5 ( g - b )$ . With $A \equiv 1$ both reduce to a fixed extrapolation (Appendix E.1).

Table S6: One-step promoter generation on the test chromosomes $( n = 5 1 2 )$ , from the same RMF initialization and the same protocol within each block. Sei MSE is the training reward; the 6-mer correlation is never optimized. On-policy rows are mean $\pm \ \mathrm { s . d . }$ over three training seeds, with the best seed in brackets; distillation rows are the single reported run. $\Delta$ is the paired NFT−MFA Sei MSE with a 95% sequence bootstrap interval, so a positive $\Delta$ favors MFA.
<table><tr><td>Method</td><td>Sei MSE↓</td><td>6-mer corr.↑</td><td>∆ [95% CI]</td></tr><tr><td>Pretrained RMF</td><td>0.0714</td><td>0.928</td><td></td></tr><tr><td colspan="4">On-policy RL, no teacher (3 seeds)</td></tr><tr><td>MeanFlowNFT MeanFlowAdvantage</td><td> $\mathbf { 0 . 0 4 7 1 \pm 0 . 0 0 4 8 }$   $0 . 0 4 8 2 \pm 0 . 0 0 5 1$  [0.0437]</td><td> $0 . 9 3 7 \pm 0 . 0 0 9$   $\mathbf { 0 . 9 5 3 \pm 0 . 0 0 1 }$ </td><td> $- 0 . 0 0 1 2 \left[ - 0 . 0 0 5 1 , 0 . 0 0 2 6 \right]$ </td></tr><tr><td colspan="4">Reward-graded distillation from a 10-step Sei-guided teacher</td></tr><tr><td colspan="4">MeanFlowNFT</td></tr><tr><td>MeanFlowAdvantage</td><td>0.0570 0.0467</td><td>0.955 0.953</td><td>0.0103 [0.0064, 0.0143]</td></tr></table>

Distillation results. Both methods select step 90 on validation. MFA reaches a Sei MSE of 0.0467, 18% lower than NFT on identical pairs and advantages, and here the gap is real: the paired difference is 0.0103 with a 95% interval of [0.0064, 0.0143] (Table S6), and Figure S3 shows that it persists throughout training. Because that interval only resamples sequences, we repeated the whole comparison with four training seeds; Table S7 shows the gap survives training randomness as well. The contrast with the on-policy block is the point of this section. With graded teacher improvements the two objectives use the reward differently, as equation 39 makes explicit, and with $\bar { \boldsymbol { A } } \equiv 1$ they become indistinguishable again (Table S8). Distillation also loses the calibrated negative direction of on-policy RL: a teacher sample worse than the student does not say which direction is better. Unfiltered signed variants performed poorly (Appendix E.1), and we leave negative directions for distillation to future work.

## E.1 ADDITIONAL PROMOTER RESULTS

![](images/69f3507df49a07b869fa9ca377f9a887ce26c2ac62a6acd1ea9bd3114e1ca30c.jpg)

![](images/5d8e74d46d39db51d85ee7348e97fb0b7a4b9df466151db93dd4d6c00bd0e090.jpg)

![](images/490cfab03da7b3935c53e37d8f9979f36104600761a8019b51270ef399469c38.jpg)  
Figure S3: Reward-graded MeanFlow distillation. (a) A frozen rollout and a Sei-guided teacher define the reward gap $\Delta r$ and the advantage A. (b)–(c) Validation curves $( n = 2 5 6 )$ ; bands are sequence-bootstrap 95% intervals, and stars mark the checkpoints selected by Sei MSE.

Training-seed repeats of the distillation gap. The interval quoted in Table S6 resamples test sequences from one training run, so it says nothing about how much the gap moves when training is repeated. We therefore reran the whole comparison, both losses and the full protocol, with four training seeds. Everything else is held fixed: the same pretrained generator, the same teacher, the pair filter $\tau = 0 . 0 0 2 , \kappa = 1 . 5$ , ten teacher steps, $\eta = 1 0$ , and the same validation selection. Table S7 lists the result. MFA wins on every seed, and the spread across seeds is small next to the gap itself: the paired difference is 0.0115 with a seed-level standard deviation of 0.0011, and a bootstrap that resamples seeds and sequences together gives [0.0091, 0.0138]. The reduction quoted in the main text is the smallest of the four (18.1%); the four seeds average 19.7%. MFA is also the steadier of the two, varying by 0.0005 across seeds against 0.0010 for NFT. We did not sweep the remaining knobs of equation ${ \bar { 3 } } 7 ,$ , so the filter threshold, the extrapolation $\kappa ,$ the teacher budget, and the reward scale η are fixed throughout at the values above; the A ≡ 1 rows of Table S8 vary κ and the learning rate only.

Table S7: Reward-graded distillation repeated over four training seeds. Checkpoints are selected on valid $\scriptstyle ( n = 2 5 6 )$ and reported on test $\mathrm { ( } n { = } 5 1 2$ , eval seed 7). ∆ is the paired NFT−MFA Sei MSE, with a 95% bootstrap over sequences; positive favors MFA. Seed 123 is the run reported in Table S6.
<table><tr><td rowspan="2">Seed</td><td colspan="2">MeanFlowNFT</td><td colspan="2">MeanFlowAdvantage</td><td rowspan="2">∆ [95% CI]</td><td rowspan="2">Red.</td></tr><tr><td>Sei MSE↓</td><td>6-mer↑</td><td>Sei MSE↓</td><td>6-mer↑</td></tr><tr><td>123</td><td>0.0570</td><td>0.955</td><td>0.0467</td><td>0.953</td><td>0.0103 [0.0064, 0.0143]</td><td>18.1%</td></tr><tr><td>42</td><td>0.0585</td><td>0.957</td><td>0.0467</td><td>0.953</td><td>0.0118 [0.0075,0.0162]</td><td>20.2%</td></tr><tr><td>456</td><td>0.0586</td><td>0.958</td><td>0.0477</td><td>0.947</td><td>0.0110 [0.0068, 0.0152]</td><td>18.7%</td></tr><tr><td>789</td><td>0.0594</td><td>0.958</td><td>0.0465</td><td>0.949</td><td>0.0129 [0.0084, 0.0173]</td><td>21.7%</td></tr><tr><td>Mean</td><td>0.0584</td><td>0.957</td><td>0.0469</td><td>0.951</td><td>0.0115</td><td>19.7%</td></tr><tr><td>Std.</td><td>0.0010</td><td>0.001</td><td>0.0005</td><td>0.003</td><td>0.0011</td><td>1.6%</td></tr></table>

Resampling seeds and sequences jointly (4000 draws) gives a 95% interval of [0.0091, 0.0138] for the mean paired difference, so the gap does not depend on the seed that was reported.

Constant-advantage special case. Setting $A \equiv 1$ discards the reward-gap magnitude after filtering. Because the rollout and reference anchors both equal b, the two DNA objectives then share an extrapolative regression form:

$$
y _ { \theta } - b = { \frac { \kappa } { 1 + \gamma + \lambda } } ( g - b ) \quad ( \mathrm { M F A } ) , \qquad y _ { \theta } - b = { \frac { e } { \beta + k } } ( g - b ) \quad ( \mathrm { M e a n F l o w N F T } ) ,\tag{40}
$$

where e is the NFT target-extrapolation multiplier and k its reference weight. Table S8 shows comparable results when learning rate and nominal extrapolation are matched. The earlier individually tuned $A \equiv 1$ configurations are retained in Tables S9–S11 as ablations.

Table S8: $A \equiv 1$ special case under matched learning rate and nominal extrapolation. Checkpoints are selected on valid $( n = 2 5 6 )$ and evaluated on test $( n = 5 1 2 )$ . ∆ is paired NFT−MFA Sei MSE with a 95% bootstrap interval.
<table><tr><td>lr</td><td>κ</td><td>NFT MSE↓</td><td>MFA MSE↓</td><td colspan="2"> $\Delta$  [95% CI]</td></tr><tr><td> $1 0 ^ { - 5 }$ </td><td>1.25</td><td>0.0550</td><td>0.0545</td><td></td><td>0.0005 [-0.0007, 0.0018]</td></tr><tr><td> $1 0 ^ { - 5 }$ </td><td>1.5</td><td>0.0490</td><td>0.0490</td><td></td><td>0.0000 [-0.0010, 0.0010]</td></tr><tr><td> $2 . 5 { \times } 1 0 ^ { - 5 }$ </td><td>1.25</td><td>0.0525</td><td>0.0518</td><td></td><td>0.0007 [-0.0013, 0.0028]</td></tr><tr><td> $2 . 5 { \times } 1 0 ^ { - 5 }$ </td><td>1.5</td><td>0.0474</td><td>0.0482</td><td></td><td>-0.0009 [-0.0025, 0.0008]</td></tr></table>

Negative-direction ablation. In on-policy RL, negative advantages suppress sampled actions relative to the rollout policy. Here, a teacher-worse endpoint does not define a calibrated opposite target: unfiltered signed variants reach only 0.0744–0.0755 best valid Sei MSE, versus 0.0510 for filtered graded-advantage MFA. We therefore retain the filter and leave distillation-specific negative directions for future work. Figure S4 collects the $A \equiv 1$ diagnostics: pretraining stability, paired valid differences, and evaluation-size sensitivity. Figure S5 shows training-pair teacher/student Sei MSE for the same special case. Table S11 varies the training seed for the earlier tuned $A \equiv 1$ configurations while fixing the RMF initialization and eval seed 7.

Table S9: Individually tuned $A \equiv 1$ special case on valid 1-NFE FANTOM5 $( n = 5 1 2 , \mathrm { c h r 1 0 } )$ Online selection used $n = 2 5 6$
<table><tr><td>Method</td><td>step</td><td>Sei MSE↓</td><td>6-mer↑</td></tr><tr><td>Pretrained RMF</td><td>0</td><td>0.0754</td><td>0.924</td></tr><tr><td>MeanFlowNFT</td><td>70</td><td>0.0553</td><td>0.951</td></tr><tr><td>MeanFlowAdvantage</td><td>55</td><td>0.0501</td><td>0.950</td></tr></table>

Table S10: Paired valid Sei ∆MSE (NFT − MFA) for the individually tuned $A \equiv 1$ special case, $n { = } 5 1 2 .$ , 2000 bootstrap draws. The interval is the 95% percentile CI.
<table><tr><td>step</td><td>∆MSE</td><td>95% CI</td></tr><tr><td>0</td><td>0.000</td><td>[0.000, 0.000]</td></tr><tr><td>10</td><td>0.0103</td><td>[0.0065, 0.0141]</td></tr><tr><td>50</td><td>0.0065</td><td>[0.0031, 0.0101]</td></tr><tr><td>60</td><td>0.0058</td><td>[0.0025, 0.0092]</td></tr><tr><td>70</td><td>0.0052</td><td>[0.0022, 0.0083]</td></tr><tr><td>100</td><td>0.0081</td><td>[0.0044, 0.0116]</td></tr></table>

Table S11: FANTOM5 training-seed ablation for the individually tuned $A \equiv 1$ special case. All runs start from the same RMF ckpts/0120 snapshot. Checkpoints are selected on valid $( n { = } 2 5 6 ,$ eval seed 7) and reported on test $\scriptstyle ( n = 5 1 2 .$ , eval seed 7). Only the training seed changes.
<table><tr><td></td><td colspan="3">MeanFlowNFT</td><td colspan="3">MeanFlowAdvantage</td></tr><tr><td>Seed</td><td>step</td><td>Sei MSE↓</td><td>6-mer↑</td><td>step</td><td>Sei MSE↓</td><td>6-mer↑</td></tr><tr><td>123†</td><td>70</td><td>0.0542</td><td>0.956</td><td>55</td><td>0.0486</td><td>0.958</td></tr><tr><td>42</td><td>60</td><td>0.0550</td><td>0.955</td><td>55</td><td>0.0492</td><td>0.955</td></tr><tr><td>456</td><td>100</td><td>0.0550</td><td>0.961</td><td>45</td><td>0.0496</td><td>0.951</td></tr><tr><td>789</td><td>90</td><td>0.0551</td><td>0.955</td><td>35</td><td>0.0479</td><td>0.953</td></tr><tr><td>Mean</td><td></td><td>0.0548</td><td>0.957</td><td>一</td><td>0.0488</td><td>0.954</td></tr><tr><td>Std.</td><td>_</td><td>0.0004</td><td>0.003</td><td>一</td><td>0.0007</td><td>0.003</td></tr></table>

<sup>†</sup>Original tuned $A \equiv 1$ run. Std. is the sample standard deviation over the four seeds.

![](images/8879aa2f308581f1af746922f2adfa346da2b124ddbae1374108ce0ed84384ff.jpg)

![](images/10aceac0879f520c924d17e8673adf71f9fb51c73050bf75fa03717378a6b1e8.jpg)

![](images/c2fd3f172df12f63c48e3a4e1c4b5408e14aa0e91529a426be640cf3dfbe1b87.jpg)

![](images/b063a8315ac868d502ab4e4f66ad562d7780ced416a2e30de35c657f49bcf5f8.jpg)  
Figure S4: Promoter analyses that are not in Figure S3. (a)–(b) Test 1-NFE Sei MSE and 6-mer versus pretrain epoch $\scriptstyle ( n = 2 5 6 )$ . The star is the RL init (ckpts/0120); the shaded region is the collapsed tail. (c) Paired valid ∆MSE (NFT − MFA) with 95% sequence-level bootstrap CIs (n=512). The star is the on-grid NFT pick (step 70). (d) Test Sei MSE at the tuned $A \equiv 1$ checkpoints versus CAGE-sorted eval $n \in$ {256, 512, 1024, 2048}.

![](images/dd750b4b34bf189f18a6e261bb0df00491727fe8af83de83cf0a372deb119f33.jpg)  
Figure S5: Training-pair Sei MSE of accepted distillation pairs (length-15 rolling mean). Solid: 1-NFE student. Dashed: 10-NFE teacher. These scores are computed for the accept filter $( \Delta r _ { \mathrm { S e i } } \geq$ 0.002); Table S6 instead uses hard-decoded held-out sequences scored by Sei.