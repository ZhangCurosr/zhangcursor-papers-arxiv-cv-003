# Uncertainty-Aware Trajectory Forecasting from Imperfect Tracking

Stephane Da Silva Martins<sup>1</sup>, Victor Petrovic<sup>2</sup>, Emanuel Aldea<sup>1</sup>, and Sylvie Le Hégarat-Mascle<sup>1</sup>

1 SATIE – CNRS UMR 8029, Université Paris-Saclay, France {stephane.da-silva-martins, emanuel.aldea, sylvie.le-hegarat}@universite-paris-saclay.fr 2 ENS Paris-Saclay, Université Paris-Saclay, France victor.petrovic@ens-paris-saclay.fr

Abstract. Most trajectory forecasting models are trained on clean annotated histories, and are often evaluated under the same idealized assumption, although practical deployments rely on trajectories produced by imperfect multi-object trackers. The real-world observations exhibit localization jitter, missed or unstable detections, and data-association ambiguity, which are usually either ignored or removed through denoising. This paper instead treats tracking-derived reliability cues as an informative signal to be propagated to the predictor. We propose a plug-in uncertainty-aware formulation in which each observed state is encoded as an uncertain state representation, modeled by a Gaussian distribution whose covariance combines detection-level localization uncertainty and association-level ambiguity through the law of total variance. Existing backbones are adapted with minimal architectural changes: input trajectories are represented as Gaussian observations, and predicted trajectories are produced as Gaussian forecasts rather than deterministic coordinates. To train predictors that remain robust under structured observation noise, we combine temporally correlated Ornstein-Uhlenbeck perturbations with response-based knowledge distillation from a teacher trained on clean trajectories. Experiments on Oxford Town Centre and VIRAT using real tracker outputs, together with a complementary ETH/UCY pseudo-detection protocol, show that the proposed formulation improves displacement accuracy and the reliability–sharpness trade-of of probabilistic forecasts.

Keywords: Trajectory Forecasting · Tracking Uncertainty · Uncertainty Quantification · Probabilistic Reliability · Knowledge Distillation

## 1 Introduction

Multi-agent trajectory forecasting is a key component of robotic navigation, video understanding, and safety-critical decision-making systems. Modern predictors have made substantial progress on standard pedestrian benchmarks by modeling social interactions, scene context, and the multimodality of future motion using graph neural networks, Transformers, recurrent architectures, and generative models [22, 40, 42]. Yet most of these methods implicitly assume that the observed past trajectory is clean, temporally consistent, and directly available at inference time.

This assumption is rarely satisfied in operational settings. Observed histories are usually obtained from an upstream multi-object tracker, whose outputs may contain localization noise, confidence fluctuations, missed detections, and identity switches. Importantly, these imperfections are not only sources of error; they also carry information about the reliability of the observations: unstable detections, low-confidence boxes or ambiguous associations often reflect challenging scene conditions such as occlusion or crowding. Removing or ignoring these signals may therefore discard useful information for the forecasting model. Recent work on prediction from raw videos and robust trajectory forecasting has shown that tracking errors can strongly afect downstream performance [25, 41], motivating predictors that propagate observation uncertainty to the predictor instead of treating the past trajectory as clean and deterministic. In this work, we specifically focus on localization uncertainty and association ambiguity at observed time steps. Explicit handling of missing observations and recovery from completed identity switches are outside the scope of the present formulation.

We address this problem by propagating tracking uncertainty through the forecasting pipeline. Instead of representing the past as deterministic points, each observation is modeled as a Gaussian state whose mean is the estimated pedestrian position and whose covariance matrix quantifies the uncertainty about the pedestrian’s true position. We decompose this covariance into two interpretable components: localization uncertainty, induced by uncertain detections, and association uncertainty, induced by multiple plausible matching candidates. These two terms are combined through the law of total variance, yielding a compact uncertainty signal that can be consumed by existing trajectory forecasting backbones.

The resulting formulation is intentionally simple. A standard coordinate embedding is replaced by an uncertainty-aware embedding of the Gaussian state parameters $( \mu _ { x } , \mu _ { y } , \sigma _ { x } , \sigma _ { y } , \rho _ { x y } )$ , where $\rho _ { x y }$ encodes the correlation term of the covariance matrix, and the output head predicts Gaussian moments for the predicted positions, optimized with a negative log-likelihood objective. This keeps the core predictor unchanged while allowing the model to account for observations with large estimated uncertainty and produce probabilistic forecasts. During training, we simulate temporally correlated tracking errors with an Ornstein– Uhlenbeck process parameterized from empirical tracker residuals. We further use response-based knowledge distillation: a teacher trained on clean trajectories provides privileged predictions, while the student learns from corrupted probabilistic observations. This encourages the student to learn motion dynamics closer to those inferred from clean trajectories, without forcing it to copy the teacher’s covariance.

Our main contributions are summarized as follows:

1. We formulate trajectory forecasting using imperfect tracker outputs as an uncertainty propagation problem, representing observed and predicted states as Gaussian states rather than deterministic points.

2. We derive a tracker-output uncertainty estimate that separates localization uncertainty from association ambiguity and combines these two components through the law of total variance.

3. We introduce an uncertainty-aware training strategy that couples temporally correlated noise injection with response-based distillation from a teacher trained on clean trajectories.

4. We evaluate robustness and probabilistic reliability on real tracking outputs from Oxford Town Centre and VIRAT, and on a complementary ETH/UCY pseudo-detection protocol.

## 2 Related Work

## 2.1 Trajectory Prediction under Clean Observations

Trajectory prediction has been extensively studied under the assumption that past trajectories are accurately observed. Prior work has explored several complementary aspects of this setting, including the modeling of social interactions between agents, the incorporation of scene context, and the extension of prediction horizons from short-term extrapolation to longer-range reasoning [2, 4, 8, 10, 15, 19, 28, 29, 39, 42, 44]. To address these challenges, the literature has progressively moved from recurrent and pooling-based models to graphbased, attention-based, and transformer-based architectures, and more recently to state-space models and semantic priors that improve long-range reasoning and scene awareness [9, 34].

Since human motion is inherently stochastic and multimodal, a large body of work has further modeled future uncertainty through generative mechanisms such as GANs [15], VAEs [21, 40], difusion models [14, 26], and flow-matching methods [7]. These approaches are efective at representing multiple plausible futures conditioned on past motion, social interactions, or scene context. However, they generally assume that the observed past is clean and complete. Our work targets this complementary source of uncertainty: the reliability of the input trajectory itself, which may be noisy, partial, or corrupted in unconstrained perception settings.

## 2.2 Robust Forecasting from Noisy Observations

Robust trajectory forecasting addresses the gap between benchmark annotations and real perception outputs. One line of work bypasses or reduces the dependence on explicit tracking by predicting from raw videos, detections, or multihypothesis tracking structures [36, 37, 41, 45]. Another line explicitly attempts to denoise or disentangle corrupted histories before forecasting. OosTraj [43] introduces a vision–positioning denoising module for out-of-sight trajectories, CaDeT [33] uses causal disentanglement, and NATRA [25] learns noise-agnostic representations through mutual-information constraints. These methods share our motivation of improving robustness to imperfect observations, but they typically treat tracking errors as a nuisance to remove. We instead expose the predictor to an uncertainty representation derived from the tracking process, so that the model can modulate its reliance on each observation according to its estimated uncertainty.

## 2.3 Tracking Uncertainty and Privileged Distillation

Uncertainty estimation is essential when trajectory forecasting relies on imperfect perception outputs. In multi-object tracking, localization covariance, detector confidence, and association ambiguity provide cues about the uncertainty of the estimated state [1, 5, 38]. We use these cues to build a Gaussian observation representation by combining detection-level and association-level uncertainty through moment matching. Knowledge distillation has also been used to transfer privileged or teacher information to a student model [16, 27, 35]. Our setting follows this learning-using-privileged-information view. During training, the teacher has access to clean histories, while the student receives corrupted probabilistic histories. The distillation objective encourages the student to align its predictions with clean history predictions, without forcing it to mimic the teacher’s uncertainty estimate.

## 3 Methodology

Given a pedestrian i, a trajectory predictor observes a past sequence ${ \bf { X } } _ { i } \ =$ $\{ ( x _ { t } ^ { i } , y _ { t } ^ { i } ) \} _ { t = 1 } ^ { T _ { \mathrm { o b s } } }$ and forecasts future positions $\mathbf { Y } _ { i } ~ = ~ \{ ( x _ { t } ^ { i } , y _ { t } ^ { i } ) \} _ { t = T _ { \mathrm { o b s } } + 1 } ^ { T _ { \mathrm { o b s } } + \bar { T } _ { \mathrm { p r e d } } }$ . In this paper, we assume that $\mathbf { X } _ { i }$ is produced by an upstream tracker and therefore comes with observation reliability that should be propagated to the forecasting model.

Our framework has three components. First, tracker outputs are converted into Gaussian observation representations by combining detection-level and association-level uncertainty. Second, we adapt existing backbones in order to process these representations and to predict Gaussian parameters for future positions. Third, training combines structured corruption and teacher–student distillation so that the student aligns its predictions made from noisy observations with clean-history predictions, while retaining its own uncertainty estimates. Figure 1 summarizes this training pipeline.

## 3.1 Probabilistic Modeling of Tracking Uncertainty

Most predictors assume deterministic inputs $\mathbf { X } _ { i }$ . In real-world scenarios, $\mathbf { X } _ { i }$ is produced by an upstream MOT and is corrupted by localization noise and association ambiguity. We therefore model the input as a sequence of random variables.

![](images/4d451d27736e715f232754fcc302e3e1c9d762b25d51eddd7fafd1ce8571c1d4.jpg)  
Fig. 1: Overview of the proposed uncertainty-aware training pipeline. A teacher model is trained on clean trajectories, whereas a student model receives probabilistic observations corrupted by temporally correlated tracking-like noise. The student is optimized with a supervised negative log-likelihood and a response-based consistency loss on the teacher mean.

For clarity in the following, we omit the pedestrian index i. Let $\mathbf { x } _ { t } \in \mathbb { R } ^ { 2 }$ denote the random vector representing the true position of a pedestrian at time t. Instead of representing the tracker output as a deterministic point, we model each observation as a probability distribution $p ( \mathbf { x } _ { t } )$ over positions (e.g., a bivariate Gaussian), parameterized from the tracker outputs.

Law of Total Variance for Tracker Uncertainty. At time t, the tracker provides a set of N candidate detections $\mathcal { D } _ { t } = \{ d _ { k } \} _ { k = 1 } ^ { N }$ . Each candidate detection $d _ { k }$ is characterized by a 2D position $\pmb { \mu } _ { k } \in \mathbb { R } ^ { 2 }$ , a bounding-box size $( W _ { k } , H _ { k } )$ , and a confidence score $s _ { k }$

Soft association over candidates. Let $c _ { k }$ denote an association cost between the current track state and candidate $d _ { k }$ . From these costs, we define normalized association weights $( w _ { k } ) _ { k = 1 } ^ { N }$ via a softmax with the temperature $\tau > 0$

$$
w _ { k } \triangleq \frac { \exp ( - c _ { k } / \tau ) } { \sum _ { j = 1 } ^ { N } \exp ( - c _ { j } / \tau ) } , \qquad \mathrm { s o ~ t h a t } \quad w _ { k } \geq 0 , \sum _ { k = 1 } ^ { N } w _ { k } = 1 .\tag{1}
$$

In this study, we use $\tau = 1$ in all experiments. Before applying the softmax, association costs are normalized per frame and per cost type, so that the fixed temperature is applied on comparable scales. We then introduce a discrete random variable $Z \in [ [ 1 , N ]$ representing the association hypothesis, with $\mathbb { P } ( Z = k ) = w _ { k }$ . Note that $( w _ { k } )$ is a soft assignment induced by the costs (with $\tau )$ , not necessarily a calibrated posterior from the tracker.

Observation model (detection noise). We model the observed 2D position at time t as a random vector $\tilde { \mathbf { x } } _ { t } \in \mathbb { R } ^ { 2 }$ . Conditioned on the association hypothesis $\{ Z = k \}$ , we assume

$$
\tilde { \mathbf { x } } _ { t } \mid \{ Z = k \} \sim \mathcal { N } ( \mu _ { k } , \Sigma _ { k } ) ,\tag{2}
$$

where $\pmb { \mu } _ { k }$ is the detected position and $\Sigma _ { k }$ encodes the spatial uncertainty associated with detection candidate $d _ { k }$ . Inspired by box-geometry cues used in standard trackers [38] and recent attempts to modulate measurement noise with detection confidence [11], we use a confidence-aware localization proxy rather than a calibrated detector posterior. This deliberately simple mapping should not be interpreted as a calibrated estimate of detector uncertainty; it only provides a monotonic reliability cue, with larger boxes and lower-confidence detections producing broader spatial uncertainty. Detector-specific or calibrated covariance estimates could replace this proxy without changing the remainder of the framework. We set the covariance of each candidate as:

$$
\begin{array} { r } { \Sigma _ { k } = \left( \begin{array} { c c } { \frac { W _ { k } ^ { 2 } } { s _ { k } + \epsilon _ { s } } } & { 0 } \\ { 0 } & { \frac { H _ { k } ^ { 2 } } { s _ { k } + \epsilon _ { s } } } \end{array} \right) , } \end{array}\tag{3}
$$

so that low-confidence detections, for which $s _ { k }$ is close to $0 ,$ yield large uncertainty, whereas high-confidence detections, for which $s _ { k }$ is close to 1, keep the variance close to the object’s squared spatial extent. Here, $W _ { k }$ and $H _ { k }$ denote the bounding-box dimensions and $s _ { k } \in ( 0 , 1 ]$ denotes the detection confidence. Moment matching. The marginal distribution induced by Eqs. (1) and (2) is a Gaussian mixture, which we reduce to a single Gaussian by matching its first two moments. By total expectation,

$$
\mu _ { \mathrm { t o t a l } } = \mathbb { E } [ \tilde { \mathbf { x } } _ { t } ] = \sum _ { k = 1 } ^ { N } w _ { k } \pmb { \mu } _ { k } ,\tag{4}
$$

and by total covariance,

$$
{ \boldsymbol { \Sigma } } _ { \mathrm { t o t a l } } = \underbrace { \sum _ { k = 1 } ^ { N } w _ { k } { \boldsymbol { \Sigma } } _ { k } } _ { { \boldsymbol { \Sigma } } _ { \mathrm { p o s } } } + \underbrace { \sum _ { k = 1 } ^ { N } w _ { k } ( { \boldsymbol { \mu } } _ { k } - { \boldsymbol { \mu } } _ { \mathrm { t o t a l } } ) ( { \boldsymbol { \mu } } _ { k } - { \boldsymbol { \mu } } _ { \mathrm { t o t a l } } ) } _ { { \boldsymbol { \Sigma } } _ { \mathrm { a s s o c } } } .\tag{5}
$$

Here, $\pmb { \Sigma } _ { \mathrm { p o s } }$ is the weighted localization uncertainty of the candidate detections, whereas $\pmb { \Sigma } _ { \mathrm { a s s o c } }$ measures their spatial dispersion and therefore association ambiguity. In the single-candidate case, $\pmb { \Sigma } _ { \mathrm { a s s o c } } = \mathbf { 0 } ;$ ; it also remains small for spatially concentrated candidates and increases when several separated candidates receive comparable weights. The two components are used to construct $\pmb { \Sigma } _ { \mathrm { t o t a l } }$ and are not passed separately to the forecasting network.

Finally, we parameterize the input to the prediction model as

$$
\tilde { \mathbf { x } } _ { t } \approx \mathcal { N } ( \mu _ { \mathrm { t o t a l } } , \Sigma _ { \mathrm { t o t a l } } ) .\tag{6}
$$

Unified uncertainty interface. Training-time perturbations and inferencetime tracking ambiguity are expressed through the same covariance interface $\pmb { \Sigma } _ { \mathrm { t o t a l } }$ . The predictor therefore receives both the estimated position and an explicit descriptor of its reliability, without requiring the forecasting architecture to distinguish the underlying source of uncertainty.

Black-box deployment: proxy association costs. In deployment, the Multi-Object Tracking (MOT) may be a black box and may not expose its internal association cost matrix. $\mathrm { T o }$ retain a multi-hypothesis formulation, we reconstruct proxy costs $c _ { k } ^ { \prime }$ a posteriori by comparing the current track state to each detection candidate $d _ { k } \ \in \ { \mathcal { D } } _ { t }$ . This formulation implicitly propagates uncertainty from t 1: as the track’s variance or bounding box size increases, the resulting costs $c _ { k } ^ { \prime } \ ( \mathrm { e . g . }$ ., Mahalanobis or IoU) reflect a higher association ambiguity. When the deployed tracker relies on specific metrics (e.g., Mahalanobis gating and ReID cosine distance as in DeepSORT [38] or BoT-SORT [1]), we approximate these components to compute $c _ { k } ^ { \prime }$ and obtain weights $w _ { k }$ via Eq. (1).

When appearance features are unavailable, we approximate association costs via two IoU-based proxies. Raw IoU computes the overlap directly between the last observed track box and each detection candidate $d _ { k }$ . Projected IoU first compensates for the track’s motion by projecting its position forward using the estimated velocity at the last time step.

## 3.2 Gaussian Observation Representation and Input Embedding

To incorporate the probabilistic input derived above into modern trajectory prediction backbones such as SingularTrajectory [3], VISTA [10] or MART [22], we modify the input embedding layer. Standard models typically employ a linear projection layer $\phi : \mathbb { R } ^ { 2 }  \mathbb { R } ^ { d }$ to map deterministic 2D coordinates $\mathbf { x } _ { t } = ( x _ { t } , y _ { t } )$ into a d-dimensional latent feature space.

In our framework, the observation at time t is parameterized by the Gaussian statistics $( \mu _ { \mathrm { t o t a l } } , \Sigma _ { \mathrm { t o t a l } } )$ . To obtain a compact uncertainty descriptor while preserving the geometry of $\pmb { \Sigma } _ { \mathrm { t o t a l } }$ , we encode the covariance by its marginal standard deviations and correlation coeficient. Let

$$
\sigma _ { x } = \sqrt { \left[ \boldsymbol { \Sigma } _ { \mathrm { t o t a l } } \right] _ { x x } } , \qquad \sigma _ { y } = \sqrt { \left[ \boldsymbol { \Sigma } _ { \mathrm { t o t a l } } \right] _ { y y } } , \qquad \rho _ { x y } = \frac { \left[ \boldsymbol { \Sigma } _ { \mathrm { t o t a l } } \right] _ { x y } } { \sigma _ { x } \sigma _ { y } + \epsilon } ,\tag{7}
$$

where ϵ is a small numerical constant and $\rho _ { x y }$ is clipped to $[ - 1 + \epsilon , 1 - \epsilon ]$ when needed. We then adapt the embedding layer to project this five-dimensional Gaussian observation representation into the same latent dimension d:

$$
\mathbf { E } _ { t } = \phi _ { \mathrm { p r o b } } \left( \mu _ { x } , \mu _ { y } , \sigma _ { x } , \sigma _ { y } , \rho _ { x y } \right) \in \mathbb { R } ^ { d } .\tag{8}
$$

By keeping the embedding dimension d identical to that of the original backbone, our embedding $\mathbf { E } _ { t }$ remains a plug-and-play modification. As a result, the proposed uncertainty-aware input can complement certain existing deterministic SOTA predictors with minimal architectural changes, without modifying their core forecasting modules. The predictor receives the combined statistics $( \mu _ { \mathrm { t o t a l } } , \Sigma _ { \mathrm { t o t a l } } ) ; \Sigma _ { \mathrm { p o s } }$ and $\pmb { \Sigma } _ { \mathrm { a s s o c } }$ are not separate input channels.

## 3.3 Probabilistic Output and Training Loss

We preserve each backbone’s native stochastic trajectory-generation mechanism and replace only its final two-dimensional position head by a five-dimensional

Gaussian head. Consequently, every native future sample k is represented at each prediction step t by $\mathcal { N } ( \mu _ { \mathrm { p r e d } , t } ^ { ( k ) } , \Sigma _ { \mathrm { p r e d } , t } ^ { ( k ) } )$ , where the mean $\pmb { \mu } _ { \mathrm { p r e d } , t } ^ { ( k ) } = ( \mu _ { x } , \mu _ { y } )$ replaces the original predicted position and $( \sigma _ { x } , \sigma _ { y } , \rho _ { x y } )$ parameterize its covariance. For SingularTrajectory, the difusion sampling process is therefore unchanged; only the final mapping from each generated sample to a 2D position is replaced by this Gaussian head. The same principle is used for MART and VISTA. We optimize the Gaussian outputs with the negative log-likelihood (NLL):

$$
\mathcal { L } _ { \mathrm { N L L } } = \frac { 1 } { 2 } \sum _ { t = 1 } ^ { T _ { \mathrm { p r e d } } } \left( \log | \Sigma _ { \mathrm { p r e d } , t } | + ( \mathbf { y } _ { t } - \boldsymbol { \mu } _ { \mathrm { p r e d } , t } ) ^ { \top } ( \Sigma _ { \mathrm { p r e d } , t } ) ^ { - 1 } ( \mathbf { y } _ { t } - \boldsymbol { \mu } _ { \mathrm { p r e d } , t } ) \right) ,\tag{9}
$$

where $\mathbf { y } _ { t }$ denotes the ground-truth position. $\mathrm { B y }$ minimizing $\mathcal { L } _ { \mathrm { N L L } }$ , the model is trained not only to minimize displacement error but also to learn a covariance that captures predictive dispersion. Its empirical reliability is evaluated using coverage and area metrics.

## 3.4 Robust Training via Empirical Noise Injection

Training a model directly on outputs from a specific tracker can imprint trackerspecific error patterns and limit generalization. Conversely, training only on clean data does not expose the model to the noise distribution it will face at inference time. To bridge this gap, we propose a tracking-noise-aware training strategy. Rather than relying on standard i.i.d. Gaussian noise, which ignores the temporal structure of tracking errors, we simulate realistic dynamics by coupling an empirical distribution with an Ornstein-Uhlenbeck process.

Empirical Error Distribution. To capture the stochastic properties of realworld tracking failures, we first analyze the error residuals of a baseline tracker evaluated on the training set. We apply kernel density estimation (KDE) to estimate the probability density function (PDF) of the error magnitudes. During training, a base error scale $\sigma _ { i }$ is sampled from this empirical distribution for each pedestrian $i ,$ so that the injected noise reflects diferent levels of tracking dificulty.

Ornstein-Uhlenbeck Drift Process. Unlike i.i.d. Gaussian noise, real-world tracking errors often exhibit temporal correlation due to the inertia of filtering algorithms $( \mathrm { e . g . }$ , Kalman filters). This “colored noise” efect [5] implies that an error at time t is likely to persist at $t + 1$

To replicate these dynamics, we model the error residuals $\mathbf { n } _ { t }$ as an Ornstein-Uhlenbeck process , defined by the following Stochastic Diferential Equation (SDE):

$$
d \mathbf { n } _ { t } = - \theta ( \mathbf { n } _ { t } - \mu \mathbf { 1 } _ { 2 } ) d t + \sigma _ { i } \ { d \mathbf { W } } _ { t } ,\tag{10}
$$

where:

– θ is the mean-reversion rate (stifness), pulling both spatial components of the error back towards the scalar mean $\mu$ (typically $\mu = 0$ for an unbiased tracker ).

$- \ \sigma _ { i }$ is the volatility (difusion coeficient) specific to pedestrian i.

$- \textbf { W } _ { t }$ denotes a standard Wiener process (Brownian motion).

For numerical implementation during training, we apply the Euler-Maruyama discretization to $\operatorname { E q . } \quad ( 1 0 )$ with a time step $\varDelta t$ . The update rule for the noise injected into pedestrian $i \mathrm { \ ' } _ { \mathrm { S } }$ trajectory becomes:

$$
\begin{array} { r } { \mathbf n _ { i , t + 1 } = \mathbf n _ { i , t } - \underbrace { \theta ( \mathbf n _ { i , t } - \mu \mathbf 1 _ { 2 } ) \Delta t } _ { \mathrm { D r i f t } } + \underbrace { \sigma _ { i } \sqrt { \Delta t } \cdot \pmb \xi _ { i , t } } _ { \mathrm { D i f f u s i o n } } , } \end{array}\tag{11}
$$

where $\pmb { \xi } _ { i , t } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { 2 } )$ is a standard Gaussian vector sampled at each step. This formulation generates smooth, temporally correlated error trajectories that mimic the drift behavior of imperfect trackers, consistent with recent trajectory simulation studies [20].

## 3.5 Privileged Knowledge Transfer via Response-Based Distillation

To mitigate performance degradation due to input corruption, we adopt a Knowledge Distillation (KD) [16] strategy within the Learning Using Privileged Information (LUPI) paradigm [35]. The link between KD and LUPI [27] has been used to bridge modality gaps and improve model robustness to noisy or missing inputs in vision tasks [12, 17]. Here, ground-truth trajectories are available during training as privileged information, to guide the learning of a model operating on imperfect data at test time.

We use response-based rather than feature-based distillation [13]. Matching only model outputs avoids architecture-specific mappings between intermediate representations and keeps the distillation mechanism portable across backbones. Teacher-Student Formulation. As illustrated in Figure 1, we instantiate a dual-network architecture comprising a Teacher model $\tau$ and a Student model ${ \mathcal { S } } .$ While both share the same backbone architecture , they operate on diferent inputs:

Teacher $\tau { : }$ Receives ground truth coordinates $\mathbf { X } _ { \mathrm { G T } }$ . To keep the input interface consistent with the Student, we cast these inputs into a probabilistic form : a Gaussian distribution centered on the ground truth with a small, fixed isotropic standard deviation $( \mathrm { e . g . , ~ } \sigma \mathrm { ~ = ~ } 0 . 1 \mathrm { ~ m } )$ to reflect annotation noise. This constitutes the privileged information and is available only during training.

– Student : Receives the probabilistic input $( \mu _ { \mathrm { n o i s y } } , \Sigma _ { \mathrm { n o i s y } } )$ generated by the noise-injection module, simulating the uncertainty of real-world tracking outputs.

The Teacher is used only during training and is discarded at inference time. It is therefore a privileged-information training component rather than a competing deployable predictor, since its clean observation history is unavailable in the target setting. Since the Teacher $\tau$ is trained on clean trajectories, it provides a predictive signal conditioned on clean observations. Distillation transfers this signal to the Student : by minimizing a consistency loss that treats the Teacher’s mean prediction as a pseudo-target, we encourage the Student to produce plausible futures that align with ground-truth dynamics, even if the observations are corrupted.

Objective Function and Predictive Uncertainty. Let $S ( \cdot \mid \bf { X } _ { \mathrm { n o i s y } } )$ denote the Student model, which outputs a probabilistic forecast parameterized as a sequence of bivariate Gaussians $\mathcal { N } ( \mu _ { t } ^ { \tilde { S } } , \Sigma _ { t } ^ { S } )$ for each future time step t. Similarly, the Teacher $\mathcal { T } ( \cdot \mid \mathbf { X } _ { \mathrm { G T } } )$ predicts Gaussians $\mathcal { N } ( \mu _ { t } ^ { \mathcal { T } } , \Sigma _ { t } ^ { \mathcal { T } } )$ ). In our distillation training, we deliberately use only the Teacher’s mean prediction $\mu _ { t } ^ { \mathcal { T } }$ as a pseudotarget.

To transfer the Teacher’s clean predictive signal to the Student, the training objective combines supervised learning with a distillation constraint. The total loss $\mathcal { L } _ { t o t a l }$ is a weighted sum:

$$
{ \mathcal { L } } _ { \mathrm { t o t a l } } = \lambda _ { \mathrm { t r a j } } { \mathcal { L } } _ { \mathrm { N L L } } ( \mathbf { Y } _ { \mathrm { G T } } \mid { \mathcal { S } } ) + \lambda _ { \mathrm { d i s t } } { \mathcal { L } } _ { \mathrm { c o n s i s t } } ( { \pmb { \mu } } ^ { { \mathcal { T } } } \mid { \mathcal { S } } ) ,\tag{12}
$$

e $\lambda _ { \mathrm { t r a j } }$ and $\lambda _ { \mathrm { d i s t } }$ balance the two objectives.

Supervised Loss $( \ { \mathcal { L } } _ { \mathrm { N L L } } )$ . The first term is the negative log-likelihood of the future ground truth $\mathbf { Y } _ { \mathrm { G T } }$ under the Student’s predictive distribution, encouraging accurate forecasting and reliable aleatoric uncertainty.

Consistency Distillation $( \ \mathcal { L } _ { \mathrm { c o n s i s t } } )$ . The second term enforces consistency between the Student’s probabilistic output and the Teacher’s mean prediction. We formulate it as the (Gaussian) NLL of the Teacher’s mean under the Student’s predictive distribution:

$$
\mathcal { L } _ { \mathrm { c o n s i s t } } = \frac { 1 } { 2 } \sum _ { t = 1 } ^ { T _ { \mathrm { p r e d } } } \left( \log \left| \boldsymbol { \Sigma } _ { t } ^ { S } \right| + ( \mu _ { t } ^ { \mathcal { T } } - \mu _ { t } ^ { S } ) ^ { \top } ( \boldsymbol { \Sigma } _ { t } ^ { S } ) ^ { - 1 } ( \mu _ { t } ^ { \mathcal { T } } - \mu _ { t } ^ { S } ) \right) .\tag{13}
$$

We use this NLL-based consistency rather than a full Kullback-Leibler (KL) divergence between the Teacher’s and Student’s predictive distributions. A KL divergence would encourage the Student to match the Teacher’s covariance $( \pmb { \Sigma } ^ { s }$ ≈ $\Sigma ^ { \mathcal { T } } )$ . However, the Teacher’s uncertainty $\pmb { \Sigma } ^ { \mathcal { T } }$ reflects future stochasticity based on clean past observations, whereas the Student must account for both future stochasticity and uncertainty induced by noisy tracking inputs. $\mathrm { B y }$ applying the NLL only to the Teacher’s mean $\mu ^ { \mathcal { T } }$ (and discarding its covariance), we distill clean motion dynamics while allowing the Student to adapt its own covariance $\pmb { \Sigma } ^ { s }$ to reflect observation reliability.

## 4 Experiments

## 4.1 Experimental Setup

Datasets. We use a tracking-derived uncertainty protocol on Oxford Town Centre (OTC) [6] and VIRAT [31]. For ETH/UCY [23, 32], where bounding-box annotations are not directly available, we use the detector-adaptation protocol described in the supplementary material before converting detections into Gaussian observation representations. This protocol is complementary and is not intended as a standard ETH/UCY benchmark comparison.

Data Splits and Real-World Evaluation. For Oxford Town Centre and VIRAT, each video is split chronologically: the first 80% of its duration is used for training and the final 20% for testing. Training and test examples therefore come from temporally disjoint frames. Clean-training configurations use groundtruth trajectories from the first 80%, while evaluation uses uncorrected MOT trajectories from the final 20%; tracker-specific adaptation uses tracker outputs only from the training portion. The split is temporal rather than identity-based, so identities are not explicitly constrained to be disjoint across the two portions.

Coordinate transformation via homography. Pedestrian detections are first obtained in the image plane. We use the bottom-center point of each bounding box as the pedestrian footprint and project it onto the ground plane using dataset homographies. Covariances defined in image coordinates are projected to the ground plane using the first-order Jacobian of the homography, $\boldsymbol { \Sigma } _ { \mathrm { g r o u n d } } = \mathbf { J } _ { h } \boldsymbol { \Sigma } _ { \mathrm { i m a g e } } \mathbf { J } _ { h } ^ { \top }$ , which accounts for perspective distortion while keeping the uncertainty representation compact [30].

Evaluation Protocols, Training Configurations, and Scope of Comparisons. We evaluate the proposed formulation with three recent trajectory prediction backbones: SingularTrajectory [3], MART [22] and VISTA [10]. These models are benchmarked under five training configurations, across which the probabilistic input embedding and the NLL loss are kept identical, so that comparisons primarily reflect the training data distribution, input preprocessing, and the presence or absence of the distillation objective.

1. Baseline: The backbone is optimized on ground-truth trajectories. When evaluated on noisy tracking inputs, this configuration quantifies the performance degradation induced by the distribution shift between training and inference.

2. Kalman Filtering: The backbone trained on clean ground-truth data is then evaluated on tracker outputs pre-processed by a causal Kalman filter, assessing the contribution of input denoising as a strong signal-processing baseline.

3. Tracker-Specific Adaptation: The model is trained on uncorrected outputs produced by the test-time tracker on the training split, aligning the training and inference distributions while avoiding access to test trajectories. This configuration may nevertheless overfit to that tracker’s artifacts and failure modes.

4. Stochastic Noise Injection: The model is trained on ground-truth data perturbed by the Ornstein-Uhlenbeck noise. This setup isolates the contribution of temporally correlated noise modeling compared to empirical trackerspecific adaptation.

5. Tracking-Noise-Aware Distillation: In the complete proposed framework, the Student network operates on probabilistic inputs synthesized via the OU noise process, guided by the Teacher network through the NLL consistency objective.

The most directly related method, NATRA [25], targets a similar setting but does not provide public code. We therefore report a NATRA-inspired reimplementation from the published description and distinguish it from an oficial implementation. OosTraj [43] and CaDeT [33] rely on substantially diferent sensing assumptions, making direct transfer to our surveillance setting non-trivial.

Unless otherwise stated, the main quantitative comparison uses BoT-SORT with its internal association costs. Results obtained with ByteTrack and with the Raw-IoU and Projected-IoU proxy costs are reported in the supplementary material.

Evaluation metrics. Each backbone retains its native inference procedure and generates $K = 2 0$ future samples, each represented by a sequence of samplespecific Gaussians $\{ \mathcal { N } ( \pmb { \mu } _ { t } ^ { ( k ) } , \hat { \pmb { \Sigma } _ { t } ^ { ( k ) } } ) \} _ { t = 1 } ^ { T _ { \mathrm { p r e d } } }$ . For displacement metrics, the Gaussian means form the sampled trajectories used for min $\mathrm { { A D E } _ { 2 0 } }$ and minF $\mathrm { \Delta D E _ { 2 0 } }$ NLL, empirical coverage, and ellipse area are computed from the corresponding sample-specific Gaussian forecasts and aggregated across samples. Additional MeanADE<sub>20</sub>, MeanFDE<sub>20</sub>, AUC [24], and 80%/90% coverage results are reported in the supplementary material.

Implementation details. Networks are trained with Adam [18]. The teacher is trained on clean ground-truth trajectories and then frozen. The student receives corrupted probabilistic observations and is optimized with the supervised NLL and response-based consistency terms. For Ornstein–Uhlenbeck noise injection, we use a mean-reversion rate $\theta = 0 . 1 5$ and long-term mean $\mu = 0$ unless otherwise specified; sensitivity to these parameters is reported in the supplementary material.

## 4.2 Quantitative Results

Displacement accuracy. Table 1 reports the results obtained on tracker trajectories. Models trained on ground-truth data lose accuracy under this shift. Kalman filtering helps, but the gap remains. Training on tracker outputs reduces this gap, at the cost of being tied to one tracker. The distillation setting gives lower ADE/FDE in most cases and stays stable across datasets and backbones. Probabilistic reliability. Table 1 also shows the efect on probabilistic reliability. When the past trajectory is treated as exact, the model often assigns too little uncertainty to noisy observations. With probabilistic inputs and NLL consistency, the student produces forecasts whose covariance better reflects the reliability of noisy inputs while maintaining competitive prediction area, improving the observed trade-of among NLL, coverage, area, and displacement accuracy.

Input–output uncertainty ablation. Table 2 separates the roles of observation and output uncertainty in the noisy-GT setting. Probabilistic inputs improve all three backbones, and the fully probabilistic variant gives the best overall results. This establishes the benefit of providing the covariance descriptor, but does not by itself isolate how the backbone internally exploits its fine-grained temporal variations.

Table 1: Quantitative evaluation on Oxford Town Centre, VIRAT, and ETH/UCY datasets. Filt. means Filtering. Distill. means Distillation. Coverage and area are reported only at the 95% and 99% confidence levels. Bold and underlined values indicate the lowest and second-lowest NLL, min $\mathrm { { A D E } _ { 2 0 } }$ , and minF $\mathrm { \Delta D E _ { 2 0 } }$ , respectively, within each dataset/backbone block.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Model</td><td rowspan="2">Training</td><td rowspan="2">NLL</td><td rowspan="2">minADE20</td><td rowspan="2">minFDE20</td><td colspan="2">Coverage (%)</td><td colspan="2">Area (m2)</td></tr><tr><td>95%</td><td>99%</td><td>95%</td><td>99%</td></tr><tr><td rowspan="2">OTO</td><td rowspan="2">VISTA</td><td>GT Kalman Filt. Tracking Noisy GT Distill.</td><td>41.47 14.54 4.93 9.15 2.30</td><td>1.57 0.69 0.91 0.64 0.56</td><td>1.77 0.82 0.73 0.65 0.54</td><td>25.6 45.6 73.2 59.3 86.2</td><td>31.6 53.9 81.6 68.1 91.5</td><td>0.84 0.61 4.04 1.48 8.03</td><td>1.29 0.94 6.21 2.28 12.4</td></tr><tr><td>MART</td><td>GT Kalman Filt. Tracking Noisy GT Distill.</td><td>16.55 0.81 14.82 0.76 -11.57 0.68 -14.69 0.58 -15.56 0.51</td><td>1.32 1.20 1.13 0.89 0.77</td><td>83.3 75.1 95.0 97.1 97.5</td><td>89.6 86.7 98.5 99.7</td><td>10.8 4.69 8.83 11.7</td><td>16.6 7.21 13.6 18.0</td></tr><tr><td></td><td>SingularTrajectory</td><td>GT Kalman Filt. Tracking Noisy GT Distill.</td><td>12.02 7.99 -5.01 -9.02 -10.99</td><td>0.91 0.76 0.72 0.60 0.54</td><td>1.46 1.25 1.15 1.02 0.89</td><td>75.02 70.01 85.99 94.01 95.02</td><td>87.99 84.02 94.98 97.98 97.99</td><td>5.62 3.42 6.39 8.31 6.99</td><td>8.59 5.22 9.81 12.81</td></tr><tr><td rowspan="2">VIAT</td><td rowspan="2">VISTA</td><td>GT Kalman Filt. Tracking Noisy GT</td><td>12.62 7.75 2.97 1.46 1.39</td><td>0.76 0.58 0.85 0.58 0.52</td><td>1.13 0.57 0.70 0.42</td><td>38.3 45.6 77.2 92.6</td><td>47.5 56.0 85.9 95.9</td><td>0.54 0.56 4.42 6.50</td><td>10.79 0.83 0.86 6.80</td></tr><tr><td>MART</td><td>Distill. GT Kalman Filt. Tracking Noisy GT</td><td>-3.14 1.09 -6.83 -12.66 -23.30 0.55</td><td>0.38 1.98 1.48 1.16 0.87</td><td>92.5 75.5 72.3 83.9 98.1</td><td>96.1 88.2 86.1 94.3</td><td>5.83 11.2 3.69 2.92</td><td>8.97 17.2 5.67 4.49</td></tr><tr><td></td><td>SingularTrajectory</td><td>Distill. GT Kalman Filt. Tracking Noisy GT Distill.</td><td>-30.17 3.02 0.02 -6.01 -14.01 -18.01</td><td>0.43 0.90 0.72 0.61 0.51 0.43</td><td>0.70 1.62 1.29 1.09 0.83</td><td>98.5 72.01 70.01 84.02 95.98</td><td>99.6 85.98 85.01 94.02 98.99</td><td>4.94 5.22 2.98 4.02 6.51</td><td>7.59 7.99 4.62 6.21 10.02</td></tr><tr><td rowspan="2">E/CY</td><td rowspan="2">VISTA</td><td>Kalman Filt. Tracking Noisy GT Distill.</td><td>21.49 11.49 5.51 2.99 2.39</td><td>0.66 0.58 0.64 0.52 0.47</td><td>1.09 0.99 0.98 0.83</td><td>47.48 52.48 67.51 82.49</td><td>62.48 67.48 79.99 92.52</td><td>0.76 0.77 2.98 4.71</td><td>1.12 1.19 4.61 7.22</td></tr><tr><td>MART</td><td>GT Kalman Filt. Tracking Noisy GT</td><td>5.51 1.99 -6.98 -14.99</td><td>0.72 0.61 0.56 0.48</td><td>0.80 1.18 1.02 0.96</td><td>86.48 62.49 66.49 75.02</td><td>94.02 76.52 80.01 87.02</td><td>4.12 6.31 8.62 13.22 3.91 6.02 5.18 7.99</td></tr><tr><td></td><td>SingularTrajectory</td><td>Distill. GT Kalman Filt. Tracking Noisy GT</td><td>-18.01 8.02 4.98 0.51 -5.02</td><td>0.42 0.64 0.55 0.53 0.44</td><td>0.83 0.71 0.98 0.88 0.87</td><td>90.48 92.48 57.49 62.48 71.02</td><td>96.51 96.98 72.51 76.52 86.01</td><td>7.78 6.12 2.82 2.49</td><td>12.01 9.39 4.32 3.78</td></tr></table>

## 4.3 Ablation Study

Noise model and distillation objective. Table 3 shows that Gaussian augmentation improves over no augmentation and OU noise further reduces errors. NLL-based distillation also outperforms KL for all backbones without forcing the Student covariance to match the clean Teacher’s covariance.

Comparison with robust forecasting. On OTC, our NATRA-inspired reimplementation is weaker for VISTA (0.66/0.68/1.21 vs. 0.56/0.54/0.99), MART (0.63/0.86/1.38 vs. 0.51/0.77/1.11), and SingularTrajectory (0.65/0.99/1.04 vs. 0.54/0.89/0.87), reported as minADE/minFDE/AUC. These results concern our reimplementation, since oficial NATRA code is unavailable.

Table 2: Input/output uncertainty ablation on OTC (noisy-GT). Each cell is minADE/minFDE/AUC; $\mathrm { D . / P . }$ denote deterministic/probabilistic input/output. Lower is better.
<table><tr><td>Model</td><td> $\mathrm { D } . / \mathrm { D } .$ </td><td> $\mathrm { D } . / \mathrm { P } .$ </td><td> $\mathrm { P . / D . }$ </td><td> $\mathrm { P } . / \mathrm { P } .$ </td></tr><tr><td>MART</td><td> $0 . 6 1 / 1 . 0 2 / 1 . 5 4$ </td><td> $0 . 6 2 / 0 . 9 5 / 1 . 4 3$ </td><td> $0 . 6 0 / 0 . 8 9 / 1 . 4 3$ </td><td> $\mathbf { 0 . 5 8 / 0 . 8 9 / 1 . 2 2 }$ </td></tr><tr><td>VISTA</td><td> $0 . 7 2 \dot { / } 0 . 7 9 \dot { / } 2 . 0 6$ </td><td> $0 . 6 9 / 0 . 7 0 \dot { / } 1 . 8 9$ </td><td> $0 . 6 7 / 0 . 6 6 \mathrm { ~ / 1 . 7 7 }$ </td><td> $\mathbf { 0 . 6 4 } / 0 . 6 5 / 1 . 6 2$ </td></tr><tr><td></td><td>SingularTraj. 0.70/1.18/1.81</td><td> $0 . 6 7 / 1 . 1 0 \dot { / } 1 . 6 8$ </td><td> $0 . 6 3 \dot { / } 1 . 0 5 \dot { / } 1 . 5 9$ </td><td>0.60/1.02/1.47</td></tr></table>

Table 3: Noise-model ablation on OTC (top) and distillation-loss ablation on VIRAT (bottom). Entries are min $\mathrm { { A D E } _ { 2 0 } / }$ min $\mathrm { F D E _ { 2 0 } ; }$ lower is better.
<table><tr><td>Variant</td><td>MART</td><td>VISTA</td><td>SingularTraj.</td></tr><tr><td>No augmentation</td><td>0.79/1.33</td><td>1.56/1.76</td><td>0.91/1.46</td></tr><tr><td>Gaussian noise</td><td>0.60/0.94</td><td>0.92/1.13</td><td>0.70/1.15</td></tr><tr><td>OU noise</td><td>0.58/0.89</td><td>0.64/0.65</td><td>0.60/1.02</td></tr><tr><td>Distill. with KL</td><td>0.56/0.96</td><td>0.58/0.54</td><td>0.52/0.84</td></tr><tr><td>Distill. with NLL</td><td>0.43/0.70</td><td>0.52/0.38</td><td>0.43/0.71</td></tr></table>

Tracker and association-proxy robustness. On OTC, Raw/Projected IoU gives lower NLL than BoT-SORT internal costs for VISTA (2.30 vs. 1.34 and 1.24), MART ( 15.56 vs. 23.49 and 22.20), and SingularTrajectory ( 10.99 vs. 14.62 and 15.04). Tracker costs can mix uncalibrated cues, whereas IoU is geometric; this does not establish IoU as universally superior, but shows that proprietary internal costs are unnecessary.

## 5 Conclusion

This paper addressed trajectory forecasting from imperfect tracker outputs. Instead of first denoising the past trajectory, we estimate uncertainty from the tracker outputs and pass it to the predictor. Each observation is written as a Gaussian state, with a covariance that combines localization and association uncertainty. The backbone therefore receives both the estimated position and an explicit descriptor of its reliability. We also train the student with OU noise and a clean teacher, so that it learns from corrupted inputs while keeping a clean motion target. Experiments on Oxford Town Centre and VIRAT with real tracker outputs, together with the ETH/UCY pseudo-detection protocol, show improvements in displacement error metrics and probabilistic reliability.

## References

1. Aharon, N., Orfaig, R., Bobrovsky, B.Z.: BoT-SORT: Robust associations multipedestrian tracking. arXiv preprint arXiv:2206.14651 (2022). https://doi.org/ 10.48550/arXiv.2206.14651

2. Alahi, A., Goel, K., Ramanathan, V., Robicquet, A., Fei-Fei, L., Savarese, S.: Social LSTM: Human trajectory prediction in crowded spaces. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2016)

3. Bae, I., Park, Y.J., Jeon, H.G.: SingularTrajectory: Universal trajectory predictor using difusion model. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 17890–17901 (2024)

4. Bai, J., Shim, I.: SceneAware: Scene-constrained pedestrian trajectory prediction with LLM-guided walkability. arXiv preprint arXiv:2506.14144 (2025)

5. Bar-Shalom, Y., Li, X.R., Kirubarajan, T.: Estimation with Applications to Tracking and Navigation. John Wiley & Sons (2001)

6. Benfold, B., Reid, I.: Stable multi-target tracking in real-time surveillance video. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2011)

7. Brinke, K.B.t., Minartz, K., Menkovski, V.: STFlow: Data-coupled flow matching for geometric trajectory simulation. arXiv preprint arXiv:2505.18647 (2025)

8. Cheng, H., Liu, M., Chen, L., Broszio, H., Sester, M., Yang, M.Y.: GatTraj: A graph- and attention-based multi-agent trajectory prediction model. ISPRS Journal of Photogrammetry and Remote Sensing 205, 163–175 (2023)

9. Chib, P.S., Singh, P.: LG-Traj: LLM-guided pedestrian trajectory prediction. In: Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops (ICCVW) (2025)

10. Da Silva Martins, S., Aldea, E., Le Hégarat-Mascle, S.: VISTA: A vision and intent-aware social attention framework for multi-agent trajectory prediction. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). pp. 287–296 (2026)

11. Du, Y., Zhao, Z., Song, Y., Zhao, Y., Su, F., Gong, T., Meng, H.: StrongSORT: Make DeepSORT great again. IEEE Transactions on Multimedia 25, 8725–8737 (2023)

12. Garcia, N.C., Morerio, P., Murino, V.: Modality distillation with multiple stream networks for action recognition. In: European Conference on Computer Vision (ECCV) (2018)

13. Gou, J., Yu, B., Maybank, S.J., Tao, D.: Knowledge distillation: A survey. International Journal of Computer Vision 129(6), 1789–1819 (2021)

14. Gu, T., Chen, G., Li, J., Lin, C., Rao, Y., Zhou, J., Lu, J.: Stochastic trajectory prediction via motion indeterminacy difusion. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2022)

15. Gupta, A., Johnson, J., Fei-Fei, L., Savarese, S., Alahi, A.: Social GAN: Socially acceptable trajectories with generative adversarial networks. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2018)

16. Hinton, G., Vinyals, O., Dean, J.: Distilling the knowledge in a neural network. In: NIPS Deep Learning and Representation Learning Workshop (2015)

17. Hofman, J., Gupta, S., Darrell, T.: Learning with side information through modality hallucination. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2016)

18. Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. In: International Conference on Learning Representations (ICLR) (2015)

19. Kothari, P., Kreiss, S., Alahi, A.: Human trajectory forecasting in crowds: A deep learning perspective. IEEE Transactions on Intelligent Transportation Systems 23(7), 7386–7400 (2022). https://doi.org/10.1109/TITS.2021.3069362

20. Langås, E.F., Zafar, M.H., Nyberg, S.O., Sanfilippo, F.: Human trajectory simulation in industrial settings using the ornstein–uhlenbeck process and deep learning based classification. In: Proceedings of the 10th IEEE International Conference on Automation, Robotics and Applications (ICARA). pp. 427–432 (2024)

21. Lee, M., Sohn, S.S., Moon, S., Yoon, S., Kapadia, M., Pavlovic, V.: MUSE-VAE: Multi-scale VAE for environment-aware long-term trajectory prediction. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2022)

22. Lee, S., Lee, J., Yu, Y., Kim, T., Lee, K.: MART: Multiscale relational transformer networks for multi-agent trajectory prediction. In: European Conference on Computer Vision (ECCV). pp. 89–107. Springer (2024)

23. Lerner, A., Chrysanthou, Y., Lischinski, D.: Crowds by example. Computer Graphics Forum 26(3), 655–664 (2007). https://doi.org/10.1111/j.1467-8659.2007. 01089.x

24. Li, L., Lin, X., Huang, Y., Zhang, Z., Hu, J.F.: Beyond minimum-of-n: Rethinking the evaluation and methods of pedestrian trajectory prediction. IEEE Transactions on Circuits and Systems for Video Technology 34(12), 12880–12893 (2024). https: //doi.org/10.1109/TCSVT.2024.3439128

25. Li, R., Li, C., Lv, R., Li, Y., Gao, Y., Zhang, X., Zhou, J.: NATRA: Noise-agnostic framework for trajectory prediction with noisy observations. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 27872– 27884 (2025)

26. Li, R., Li, C., Ren, D., Chen, G., Yuan, Y., Wang, G.: BCDif: Bidirectional consistent difusion for instantaneous trajectory prediction. In: Advances in Neural Information Processing Systems (2023)

27. Lopez-Paz, D., Bottou, L., Schölkopf, B., Vapnik, V.: Unifying distillation and privileged information. In: International Conference on Learning Representations (ICLR) (2016)

28. Mohamed, A., Qian, K., Elhoseiny, M., Claudel, C.: Social-STGCNN: A social spatio-temporal graph convolutional neural network for human trajectory prediction. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2020)

29. Na, K.I., Kim, U.H., Kim, J.H.: SPU-BERT: Faster human multi-trajectory prediction from socio-physical understanding of BERT. Knowledge-Based Systems 274, 110637 (2023)

30. Ochoa, B., Belongie, S.: Covariance propagation for guided matching. In: Workshop on Statistical Methods in Multi-Image and Video Processing (SMVP). Graz, Austria (2006)

31. Oh, S., Hoogs, A., Perera, A., Cuntoor, N., Chen, C.C., Lee, J.T., Mukherjee, S., Aggarwal, J., Lee, H., Davis, L., et al.: A large-scale benchmark dataset for event recognition in surveillance video. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2011)

32. Pellegrini, S., Ess, A., Schindler, K., Van Gool, L.: You’ll never walk alone: Modeling social behavior for multi-target tracking. In: Proceedings of the IEEE International Conference on Computer Vision (ICCV) (2009)

33. Pourkeshavarz, M., Zhang, J., Rasouli, A.: CaDeT: A causal disentanglement approach for robust trajectory prediction in autonomous driving. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2024)

34. Ren, Z., Wei, P., Tang, H., Li, H., Yang, J., Qin, J.: Stochastic-aware Mamba difusion for pedestrian trajectory prediction. In: Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (2025)

35. Vapnik, V., Vashist, A.: A new learning paradigm: Learning using privileged information. Neural Networks 22(5-6), 544–557 (2009). https://doi.org/10.1016/j. neunet.2009.06.042

36. Weng, X., Ivanovic, B., Kitani, K., Pavone, M.: Whose track is it anyway? improving robustness to tracking errors with afinity-based trajectory prediction. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2022)

37. Weng, X., Ivanovic, B., Pavone, M.: MTP: Multi-hypothesis tracking and prediction for reduced error propagation. In: Proceedings of the IEEE Intelligent Vehicles Symposium (IV) (2022)

38. Wojke, N., Bewley, A., Paulus, D.: Simple online and realtime tracking with a deep association metric. In: Proceedings of the IEEE International Conference on Image Processing (ICIP) (2017)

39. Xu, C., Tan, R.T., Tan, Y., Chen, S., Wang, Y., Wang, X., Wang, Y.: EqMotion: Equivariant multi-agent motion prediction with invariant interaction reasoning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2023)

40. Xu, P., Hayet, J.B., Karamouzas, I.: SocialVAE: Human trajectory prediction using timewise latents. In: European Conference on Computer Vision (ECCV) (2022)

41. Yu, R., Zhou, Z.: Towards robust human trajectory prediction in raw videos. In: Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) (2021)

42. Yuan, Y., Weng, X., Ou, Y., Kitani, K.M.: AgentFormer: Agent-aware transformers for socio-temporal multi-agent forecasting. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) (2021)

43. Zhang, H., Xu, Y., Lu, H., Shimizu, T., Fu, Y.: OosTraj: Out-of-sight trajectory prediction with vision-positioning denoising. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2024)

44. Zhang, P., Xue, J., Zhang, P., Zheng, N., Ouyang, W.: Social-aware pedestrian trajectory prediction via states refinement LSTM. IEEE Transactions on Pattern Analysis and Machine Intelligence 44(05), 2742–2759 (2022)

45. Zhang, P., Bai, L., Wang, Y., Fang, J., Xue, J., Zheng, N., Ouyang, W.: Towards trajectory forecasting from detection. IEEE Transactions on Pattern Analysis and Machine Intelligence 45(10), 12550–12561 (2023)