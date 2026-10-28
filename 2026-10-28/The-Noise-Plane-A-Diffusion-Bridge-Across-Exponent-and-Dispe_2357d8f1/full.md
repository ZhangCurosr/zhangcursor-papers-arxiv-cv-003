# The Noise Plane: A Diffusion Bridge Across Exponent and Dispersion

Xuran Hu, Yuru Jia, Tengxi Wang, Miloš Dakovic,´ Member, IEEE, Ljubiša Stankovic,´ Fellow, IEEE, and Wufan Zhao, Member, IEEE

Abstract—Diffusion-based denoising commonly uses Gaussian perturbations, whereas sensor noise follows diverse signaldependent statistical laws. In this paper, we propose p-Bridge, a unified measurement-indexed diffusion framework for Tweedie noise. The noise exponent p describes the variance–intensity relation, while accumulated measurement ν controls the noise level. Together, they define a noise plane that includes Gaussian, Poisson, and Gamma models. The stochastic bridge connects measurement distributions through independent increments, giving denoising a physical interpretation as progressive measurement accumulation. Both the starting and stopping indices can be specified, allowing restoration from an observed measurement level to a prescribed higher level. Measurement superposition yields a shared reversemean update, paired with a deviance-based objective for learning the required conditional mean. A latent compound Poisson– Gamma construction enables sampling at intermediate exponents. A single exponent-conditioned network adapts to different Tweedie noise models without separate training for each exponent. Experiments show that exponent conditioning improves crossexponent robustness over fixed-exponent models. Using estimated exponents, the shared network achieves robustness comparable to an oracle-selected set of five specialist models. Experiments on synthetic images, FMD fluorescence microscopy, and multi-sensor SAR images further demonstrate its effectiveness across different signal-dependent noise regimes.

Index Terms—diffusion bridge, image restoration, denoising, exponential dispersion models, Tweedie family.

## I. INTRODUCTION

MAGE denoising is a fundamental problem in computational imaging, where the objective is to recover scene information from measurements corrupted during acquisition. Classical methods combine an observation model with image priors based on sparsity or nonlocal similarity [1]. These approaches established effective ways to separate image structure from noise, while subsequent plug-and-play methods incorporated learned denoisers into model-based optimization [2]. Their formulation requires both a useful prior for the clean image and an appropriate statistical description of the observation. In physical imaging, however, the observation statistics vary substantially across sensing mechanisms. Photon counting produces signal-dependent fluctuations in cameras and fluorescence microscopy [3], [4], whereas coherent imaging introduces multiplicative speckle [5]. Consequently, a noise level alone is insufficient to characterize the degradation across these different imaging regimes.

![](images/4657f78403899fb76cd4969955b5b2f09c2a1d4e1604295513557cac3b46ff29.jpg)  
Fig. 1. Noise regimes along the Tweedie exponent axis. Top: illustrative noise at representative exponents. Bottom: real observations with estimated exponents pˆ.

Learning-based methods have substantially improved denoising by learning image statistics and restoration mappings from data. Convolutional networks provide efficient inference [6], [7], Transformer-based models capture long-range image dependencies [8], and noise-conditioned models allow one network to handle different corruption levels [9]. Nevertheless, supervised restoration depends on how well the training data represent the deployment conditions, while collecting accurate clean references can be difficult in real photography and scientific imaging [10], [11]. Learning from noisy observations reduces this dependence. Noise2Noise uses corrupted training targets [12], and Noise2Self exploits noise independence to learn without clean targets [13]. These methods broaden the available training settings, but their validity and performance still depend on the statistical structure of the observations. Blind restoration also benefits from explicitly estimating the degradation. VIRNet, for example, jointly estimates noise and restores the image under a spatially varying Gaussian likelihood [14]. Such developments make denoising more adaptive, while leaving the choice of the underlying noise distribution an important modeling decision.

A complementary line of research addresses this modeling decision through sensor physics and noise transformations. Physics-based noise models describe the acquisition process and generate realistic training pairs [15], while recent physicsinformed approaches learn difficult-to-model sensor components from calibration data such as dark frames [16]. These methods improve the connection between synthetic training noise and real measurements, although adapting them to another sensing mechanism requires a corresponding noise model and calibration procedure. Variance-stabilizing transformations offer another route by reducing signal dependence before Gaussian denoising [17]–[19]. Noise2VST extends this idea by learning a model-free transformation from the noisy input and using a frozen Gaussian denoiser [20]. This demonstrates that a Gaussian denoiser can remain useful for non-Gaussian observations when coupled with suitable adaptation. These approaches focus on modeling or transforming an observation without specifying how its distribution should evolve along a denoising trajectory.

Diffusion-based restoration makes this trajectory an explicit part of the model. Restoration bridges such as I<sup>2</sup>SB learn processes connecting degraded and clean image distributions [21], while diffusion posterior sampling incorporates measurement likelihoods, including Poisson noise, into reconstruction with a diffusion prior [22]. Beyond Gaussian formulations, Xie et al. construct denoising diffusion processes for Gaussian, Gamma, and Poisson noise with a unified training strategy [23]. Gamma Bridge further gives the trajectory a physical coordinate through the number of looks, allowing restoration between specified measurement levels for Gamma speckle [24]. These advances establish both noise-aware diffusion and measurementindexed restoration, but these constructions treat particular noise laws or discrete families. Extending this perspective to a continuously varying variance–intensity relation requires a common construction that adapts the noise law while preserving the measurement interpretation of the intermediate states.

In this work, we propose p-Bridge, a unified measurementindexed diffusion bridge based on the Tweedie family [25], [26]. Our starting point is that accumulating independent measurements of the same scene reduces noise while preserving the expected intensity. The Tweedie variance relation $\mu ^ { p } / \nu$ allows us to apply this principle across different noise laws. Here, p describes how variance depends on intensity, while ν represents accumulated exposure, dose, or number of looks. These two coordinates distinguish changes in the noise law (Figure 1) from movement along a measurement trajectory (Figure 2). At a fixed exponent, increments that are independent conditional on the clean image connect the measurement distributions into a stochastic bridge.

In this construction, the network predicts the clean signal needed for each reverse update, while exponent conditioning allows its prediction to adapt to the observation’s noise statistics. We evaluate the role of this adaptation through controlled exponent-mismatch and estimation experiments, and assess restoration quality on synthetic images, FMD fluorescence microscopy, and SAR images from multiple sensors.

The main contributions are as follows:

1) We propose p-Bridge, a unified measurement-indexed diffusion framework for Tweedie noise. Its two noise coordinates have distinct roles. The exponent p specifies the variance–intensity relation, while accumulated measurement ν controls the noise level. The stochastic bridge connects the corresponding measurement distributions, allowing both the starting and stopping measurement levels to be specified.

![](images/f9667e770dc48589f9550ce19d688004108428e963f6cf60de560ed57fceeb6a.jpg)  
Fig. 2. Measurement-indexed denoising trajectories on the noise plane. p controls signal dependence, while increasing ν reduces noise.

2) We derive a shared reverse-mean update from measurement superposition, with a mixing coefficient independent of the noise exponent. A unit-deviance reconstruction objective targets the conditional mean required by this update. A latent compound Poisson–Gamma construction enables sampling of measurement transitions at intermediate exponents.

3) We develop a single exponent-conditioned network that adapts to different Tweedie noise models without separate training for each exponent. At inference, the network uses an exponent estimated from the observation. This shared model achieves cross-exponent robustness comparable to an oracle-selected set of five specialist models.

## II. RELATED WORK

## A. Diffusion models, bridges and inverse problems

Widely used denoising diffusion models [27] and their scorebased formulations [28] add Gaussian noise according to a prescribed variance schedule. Subsequent work has improved sampling [29] and network parameterization [30]. Restorationoriented methods also redesign the trajectory. For example, ResShift introduces residual shifting to reduce the number of sampling steps [31].

Restoration bridges connect clean and degraded image distributions. Schrödinger bridge formulations connect prescribed endpoint distributions [32], with I<sup>2</sup>SB applying this principle to image restoration [21]. Other formulations use direct endpoint interpolation [33] or mean-reverting stochastic differential equations [34]. More recent developments target spatial adaptation and efficiency. RDBM uses endpoint residuals to modulate noise injection and removal across image regions [35], while E-Bridge shortens the trajectory and learns a consistency-based mapping for single-step restoration [36]. System-Embedded Diffusion Bridge Models incorporate the forward system explicitly, embedding a known linear operator and noise covariance into a matrix-valued SDE under an additive Gaussian measurement model [37].

![](images/a10d4a3367b6ac959b522ae02143663772d4dfc553c7da1cefbdfb3f11a08c3b.jpg)  
Fig. 3. Network architecture and training losses of p-Bridge, with a deterministic restoration trajectory on an FMD crop.

Another class of methods adapts a pretrained diffusion prior to the observation. DDRM uses the singular-value decomposition of the forward operator [38], whereas DPS introduces an approximate likelihood score [22]. DPS supports Poisson measurement noise while retaining a Gaussian diffusion prior. IRBridge transfers pretrained generative priors to restoration bridges by relating processes with a common endpoint distribution [39].

Non-Gaussian constructions extend the choice of diffusion distributions. Xie et al. formulate diffusion denoising for Gaussian, Gamma, and Poisson corruption with a unified training strategy [23]. Learning to Jump uses count thinning and reverse count thickening for generative modeling [40], while Star-Shaped DDPM develops non-Gaussian models within exponential families [41]. In statistical inference, data thinning splits observations from convolution-closed distributions, including Gaussian, Poisson, and Gamma laws, into independent components [42]. Gamma Bridge indexes a speckle-denoising trajectory by the number of looks [24]. p-Bridge extends this measurement-indexed formulation to the Gaussian point and the Poisson-to-Gamma continuum, using a common superposition construction and a single exponent-conditioned network.

## B. Signal-dependent denoising and noise estimation

Variance-stabilizing transformations reduce signal dependence before Gaussian denoising. Representative choices include the Anscombe transform for Poisson data [17], with a corrected inverse at low counts [43], and the logarithmic transform for speckle [19]. Noise2VST learns a model-free transform from a single noisy image and combines it with frozen Gaussian denoisers [20].

Sensor-specific methods incorporate the statistics of the acquisition process. Photon-limited methods use variancestabilized nonlocal regression for microscopy [4], raw-data networks [44], or physics-based sensor noise models [45]. Speckle methods exploit its multiplicative statistics [5] through self-supervised networks [46] or closed-form shrinkage [47].

Single-image noise estimation is often formulated within a specified noise family. Gaussian noise estimators use patch statistics and covariance eigenvalues [48]. Poisson–Gaussian fitting uses local mean–variance relations to estimate shotnoise and read-noise parameters from a raw frame [3]. VIRNet jointly estimates degradation and restores the image under a spatially varying Gaussian likelihood [14]. Blind2Sound couples noise-level sensing with self-supervised denoising for Poisson–Gaussian corruption [49]. In p-Bridge, the estimated exponent p selects the bridge’s measurement law, while the estimated index ν determines its starting noise level. The two parameters are fitted jointly as the scaling of ν depends on p.

## III. A UNIFIED MEASUREMENT-INDEXED BRIDGE

In p-Bridge, denoising is modeled as the accumulation of independent measurements of the same scene, which reduces noise while preserving the underlying signal. We use the accumulated measurement amount to index a bridge across Gaussian, Poisson, Gamma, and intermediate noise models. The exponent p selects the noise model, while the measurement index ν controls its noise level. We first construct this bridge, then derive its reverse update and reconstruction loss, and finally describe sampling and training. Figure 3 shows the network pipelines.

## A. Preliminaries and notation

We first introduce the distributional notation used to describe measurements. Let µ denote a generic mean intensity and $\nu > 0$ the accumulated measurement index. The accumulated signal Z

follows an additive exponential dispersion model [26], written $Z \sim \mathrm { E D } ^ { * } ( \theta , \nu )$ , with density

$$
f _ { Z } ( z ; \theta , \nu ) = c ^ { * } ( z ; \nu ) \exp \{ \theta z - \nu \kappa ( \theta ) \} .\tag{1}
$$

Here θ is the natural parameter, κ is the cumulant function, and $c ^ { * }$ is a nonnegative factor independent of θ. The density is taken with respect to a suitable dominating measure, so that continuous, discrete, and mixed distributions are all covered. Normalization makes $\nu \kappa ( \theta )$ the log-normalizer. Its first two derivatives give

$$
\mathbb { E } [ Z ] = \nu \kappa ^ { \prime } ( \theta ) , \qquad \mathrm { V a r } ( Z ) = \nu \kappa ^ { \prime \prime } ( \theta ) .\tag{2}
$$

The measured intensity is $x = Z / \nu .$ . Its mean and variance are determined by the mean map τ and variance function $V ,$

$$
\mu = \tau ( \theta ) = \kappa ^ { \prime } ( \theta ) , \quad \quad V ( \mu ) = \kappa ^ { \prime \prime } \big ( \tau ^ { - 1 } ( \mu ) \big ) .\tag{3}
$$

Consequently, the intensity has mean $\mu$ and variance $V ( \mu ) / \nu .$ For a fixed noise family, neither τ nor V depends on $\nu .$ Changing the measurement index therefore changes the noise level without changing the underlying mean. Since $\tau ^ { \prime } ( \theta ) = \kappa ^ { \prime \prime } ( \theta ) > 0$ in the interior of the natural-parameter domain, τ is invertible onto the mean domain Ω. Its inverse maps an intensity to the corresponding natural parameter and is the canonical link in generalized linear models.

Tweedie models have the power variance function $V ( \mu ) =$ $\mu ^ { p } ~ [ 2 5 ]$ , [26]. The exponent $p$ specifies the variance–intensity relation, while $\mu$ and $\nu$ determine the mean intensity and measurement level. For the Gaussian branch, we also allow a fixed variance scale $\sigma ^ { 2 } ;$ setting $\sigma ^ { 2 } = 1$ recovers $V ( \mu ) = \mu ^ { 0 }$ The family-specific functions are listed in Table S1 of the supplement.

The index ν represents accumulated measurement rather than noise amplitude. Larger indices correspond to less noisy measurements of the same scene. Accordingly, the forward process decreases $\nu ,$ whereas denoising increases it. A finite index still describes a noisy measurement; the clean signal is approached as $\nu \to \infty$

## B. Measurement-indexed formulation

Let $x _ { \star }$ denote the clean image and $x _ { \nu }$ an intensity measurement at index $\nu > 0$ . Depending on the sensor, ν represents exposure, dose, or number of looks, expressed in the units of the noise model. We assume

$$
\mathbb { E } [ x _ { \nu } \mid x _ { \star } ] = x _ { \star } , \qquad \mathrm { V a r } ( x _ { \nu } \mid x _ { \star } ) = \frac { V ( x _ { \star } ) } { \nu } .\tag{4}
$$

These moments follow from Equations (2) and (3) with $\mu = x _ { \star }$ Any additional dispersion scale is absorbed into ν.

The bridge also requires a rule for combining measurements. For two measurements of the same scene that are independent conditional on $x _ { \star }$ , their average weighted by $\nu _ { 1 }$ and $\nu _ { 2 }$ must remain in the same family at index $\nu _ { 1 } + \nu _ { 2 }$ . The Tweedie models above have this property. It is simplest to express it using the accumulated signal $Z _ { \nu } = \nu x _ { \nu }$ . Write its distribution as $\mathrm { E D } ^ { * } ( \theta , \nu )$ , where the natural parameter θ specifies the underlying mean. For independent $Z _ { 1 } ~ \sim ~ \mathrm { E D } ^ { * } ( \theta , \nu _ { 1 } )$ and $Z _ { 2 } \sim \mathrm { E D } ^ { * } ( \theta , \nu _ { 2 } )$

$$
Z _ { 1 } + Z _ { 2 } \sim \mathrm { E D } ^ { * } ( \theta , \nu _ { 1 } + \nu _ { 2 } ) .\tag{5}
$$

Dividing the sum by $\nu _ { 1 } + \nu _ { 2 }$ gives the required intensity average. This includes Poisson superposition and Gamma addition at a common rate. Closure is not guaranteed for different natural parameters.

We now arrange these measurement distributions along a decreasing schedule $\nu ( 0 ) = \nu _ { \mathrm { m a x } } > \cdots > \nu ( T - 1 ) = \nu _ { \mathrm { o b s } } .$ At each step, define

$$
\boldsymbol { Z } _ { t } \mid \boldsymbol { x } _ { \star } \sim \mathrm { E D } ^ { \ast } ( \theta _ { \star } , \nu ( t ) ) , \qquad \boldsymbol { x } _ { t } = \frac { \boldsymbol { Z } _ { t } } { \nu ( t ) } ,\tag{6}
$$

where $\theta _ { \star } = \tau ^ { - 1 } ( x _ { \star } )$ is the natural parameter corresponding to the clean image. The final marginal matches the observation model at $\nu _ { \mathrm { o b s } } ,$ whereas the finite initial state $x _ { 0 }$ still contains noise. The clean image is approached in conditional mean square as $\nu \to \infty$

Each marginal in Equation (6) follows the assumed sensor model. To connect these marginals into a joint process, we use independent increments with the same underlying mean, as defined next. All identities apply pixelwise. The regularity conditions, including the availability of arbitrary positive measurement increments and the invertibility of τ on the mean domain Ω, are stated in Section S1.

## C. Family-wide properties

1) Superposition and the reverse mean: A reverse step moves from $\nu ( t )$ to the larger measurement amount $\nu ( t - 1 )$ By Equation (5), the missing amount $\Delta _ { t } = \nu ( t - 1 ) - \nu ( t ) > 0$ can be supplied by an independent measurement increment. This defines the following coupling.

Theorem 1 (Superposition coupling). For $t = 1 , \dots , T - 1$ let $Y _ { t  t - 1 } \sim \mathrm { E D } ^ { * } ( \theta _ { \star } , \Delta _ { t } )$ be independent of $Z _ { t }$ . Then $Z _ { t } +$ $Y _ { t  t - 1 } \sim \mathrm { E D } ^ { * } ( \theta _ { \star } , \nu ( t - 1 ) )$ ). Equivalently,

$$
x _ { t - 1 } = \alpha _ { t } x _ { t } + { \frac { Y _ { t  t - 1 } } { \nu ( t - 1 ) } } , \qquad \alpha _ { t } = { \frac { \nu ( t ) } { \nu ( t - 1 ) } } .\tag{7}
$$

The stochastic reverse therefore adds an independent measurement increment. $\mathrm { A t } ~ p = 2$ it recovers the Gamma Bridge update [24], and at $p = 1$ it gives the corresponding Poisson construction. Replacing x with a network prediction $\hat { x } _ { \star }$ defines a plug-in stochastic transition. Replacing its increment with its conditional mean gives a deterministic update.

Theorem 2 (p-independence of the reverse mean). Given $( x _ { t } , \hat { x } _ { \star } )$ with $\begin{array} { r l r } { \hat { x } _ { \star } } & { { } \in } & { \Omega , } \end{array}$ draw $\begin{array} { r l r l } { Y _ { t  t - 1 } } & { { } | } & { ( x _ { t } , \hat { x } _ { \star } ) } & { { } \sim } & { } \end{array}$ $\mathrm { E D } ^ { * } ( \tau ^ { - 1 } ( \hat { x } _ { \star } ) , \Delta _ { t } )$ . Under this model transition,

$$
\mathbb { E } [ x _ { t - 1 } \mid x _ { t } , \hat { x } _ { \star } ] = \alpha _ { t } x _ { t } + ( 1 - \alpha _ { t } ) \hat { x } _ { \star } .\tag{8}
$$

The coefficient $\alpha _ { t }$ depends only on the measurement schedule, not on p.

The update is the conditional mean of the plug-in transition defined above, obtained from $\mathbb { E } [ Y _ { t  t - 1 } \mid x _ { t } , \hat { x } _ { \star } ] = \Delta _ { t } \hat { x } ,$ <sub>⋆</sub>. With oracle predictions $\hat { x } _ { \star } = x _ { \star }$ and the correct observation-model initialization, independent increments reproduce the joint law of the superposition coupling. A fixed observation instead defines its conditional trajectory.

The deterministic update preserves the oracle mean but not the bridge marginals. For a bridge-distributed input, its conditional variance is $\alpha _ { t } V ( x _ { \star } ) / \nu ( t - 1 )$ rather than $V ( x _ { \star } ) / \nu ( t - 1 )$

Physical measurement marginals and oracle distributional exactness therefore apply to the stochastic construction, while the deterministic sampler follows its reverse mean (Section S2).

2) The unit deviance: For $y , \mu \in \Omega$ , the unit deviance equals twice the Bregman divergence generated by the convex conjugate $\boldsymbol { \psi } = \boldsymbol { \kappa } ^ { * } ~ [ 5 0 ]$ :

$$
d ( y , \mu ) = 2 B _ { \psi } ( y , \mu ) .\tag{9}
$$

Theorem 3 (Mean-optimal reconstruction). Let $Y$ take values in the domain of $d ,$ with $\mathbb { E } [ | Y | ] < \infty , m = \mathbb { E } [ Y ] \in \Omega ,$ , and $\mathbb { E } [ d ( Y , m ) ] < \infty$ . Then

$$
\arg \operatorname* { m i n } _ { \mu \in \Omega } \mathbb { E } [ d ( Y , \mu ) ] = m
$$

uniquely. Applied conditionally to the clean image, the optimal reconstruction under the same conditions is

$$
\mathbb { E } [ x _ { \star } \mid x _ { t } , x _ { \mathrm { o b s } } ] .
$$

The Bregman representation and mean-optimality proof are derived in Section S3. Our bounded clean-image range $\mathcal { X } = [ \epsilon , 1 ] , \epsilon > 0$ , satisfies these conditions. The result aligns the reconstruction term with the conditional mean used in Equation (8).

Differentiating the deviance gives a shared gradient for the power-variance family:

$$
\partial _ { \mu } d _ { p } ( y , \mu ) = - \frac { 2 ( y - \mu ) } { \mu ^ { p } } .\tag{10}
$$

With $d _ { p } ( y , y ) = 0 \quad$ , this yields squared error at $p = 0$ , Poisson deviance at $p = 1$ , and twice the Itakura–Saito divergence at $p = 2 ,$ , as well as the intermediate deviance forms.

## D. Exponent-specific thinning

Training couplings and measurement-level initialization require a downward transition that removes part of an accumulated measurement. Under the superposition coupling, its density or mass is

$$
q ( z _ { t } \mid z _ { t - 1 } ; \theta _ { \star } ) = \frac { c ^ { * } ( z _ { t } ; \nu ( t ) ) c ^ { * } ( z _ { t - 1 } - z _ { t } ; \Delta _ { t } ) } { c ^ { * } ( z _ { t - 1 } ; \nu ( t - 1 ) ) } .\tag{11}
$$

The exponential factor depends only on the conditioned total $z _ { t - 1 }$ and cancels. Hence the kernel is independent of $\theta _ { \star }$ . For compound laws, Equation (11) includes endpoint atoms as well as a continuous component. The corresponding measures are specified in Section S4.

At $\displaystyle p = 0 , 1 , 2 ,$ , the kernels are Gaussian, Binomial, and scaled Beta, respectively (Table I). For $1 < p < 2$ , the compound Poisson–Gamma representation instead provides a two-stage construction.

Proposition 4 (Latent thinning for $\ : 1 \ : < \ : p \ : < \ : 2 )$ . Let $\begin{array} { r } { Z _ { \nu } \overset { - } { = } \sum _ { i = 1 } ^ { N _ { \nu } } J _ { i } , } \end{array}$ where $N _ { \nu } \sim \mathrm { P o i s } ( \lambda \nu )$ and the jumps are independent $\mathsf { \bar { T } } ( \alpha , \beta )$ variables in shape–scale parameterization, independent of $N _ { \nu } .$ Set

$$
\begin{array} { l l } { \displaystyle \lambda = \frac { \mu ^ { 2 - p } } { 2 - p } , } & { \displaystyle \alpha = \frac { 2 - p } { p - 1 } , } \\ { \displaystyle \beta = ( p - 1 ) \mu ^ { p - 1 } . } \end{array}\tag{12}
$$

Then $\mathbb { E } [ Z _ { \nu } ] ~ = ~ \nu \mu$ and $\mathrm { V a r } ( Z _ { \nu } ) ~ = ~ \nu \mu ^ { p }$ . Conditional on $N _ { t - 1 } = n$ and $Z _ { t - 1 } = z ,$ , sample $N _ { t } = n _ { 1 } \sim$ Binomial $. ( n , \alpha _ { t } )$ For $0 < n _ { 1 } < n ,$

$$
Z _ { t } / z \sim \mathrm { B e t a } ( n _ { 1 } \alpha , ( n - n _ { 1 } ) \alpha ) .
$$

If $n _ { 1 } = 0 ,$ set $Z _ { t } = 0 ; i f n _ { 1 } = n > 0 ,$ set $Z _ { t } = z . \ I f n = 0 ,$ both totals are zero and no ratio is formed.

This combines Poisson thinning of the counts with Beta– Gamma splitting of the amplitudes. When a simulator retains the count, these steps require no count inference. For an observed total alone, the count must first be sampled from its posterior. The weights are independent of the unknown mean and are given in Section S4-D. The finite truncation used for real observations makes that implementation approximate. The synthetic severity-transfer experiment uses simulator counts when thinning the conditioning channel, a protocolspecific source of information not available at deployment (Section S10-C).

Within the specified compound model, a Gamma-distributed gain G has excess-noise factor $F ^ { 2 } = \mathbb { E } [ G ^ { 2 } ] / \mathbb { E } [ G ] ^ { 2 } = 1 + 1 / \alpha ,$ yielding

$$
p = 2 - \frac { 1 } { F ^ { 2 } } .\tag{13}
$$

This relation holds under the mean-dependent jump scale and rate in Equation (12). At fixed $\mu$ and $\nu ,$ the Tweedie family approaches Poisson and Gamma as $p$ approaches 1 and 2. By contrast, changing only the event rate at fixed gain produces a linear variance–mean relation, but not generally a Poisson distribution.

## E. Output parameterization and measurement calibration

For positive clean intensities, the conditional coefficient of variation is

$$
\operatorname { C V } ( x _ { t } \mid x _ { \star } ) = { \frac { x _ { \star } ^ { p / 2 - 1 } } { \sqrt { \nu ( t ) } } } .\tag{14}
$$

It is independent of intensity at $p = 2$ and increases towards dark regions for $p < 2 .$ This motivates our residual anchoring at $p = 2$ and direct log-intensity prediction for $1 \leq p < 2$ Positive means are enforced with an exponential output map for $p \geq 1$ , while the Gaussian branch uses an identity map.

We distinguish the mean domain Ω, the normalized cleanimage range $\mathcal { X } ~ = ~ [ \epsilon , 1 ]$ , and the observation support. The sampler clips predictions $\hat { x } _ { \star }$ to $x ,$ , not sampled states. This bounded-range projection is an implementation choice, since positivity of the mean alone imposes no upper bound. In particular, compound observations can equal zero or exceed the clean-image range.

For a specified exponent $p ,$ define on a homogeneous patch $\hat { \nu } = V ( \bar { x } ) / \hat { \sigma } ^ { 2 }$ . A radiometric rescaling $x \mapsto c x , c > 0$ , gives

$$
\hat { \nu } \longmapsto c ^ { p - 2 } \hat { \nu } .\tag{15}
$$

Only at $p = 2$ is the estimated index invariant to multiplicative gain. At other exponents, the index must be reported relative to a specified intensity scale. This motivates calibrating the intensity scale and jointly estimating $( p , \nu )$

TABLE I  
BRIDGE STATES AND THINNING KERNELS ACROSS THE SUPPORTED EXPONENT AXIS.
<table><tr><td>p</td><td> $x _ { \nu }$ </td><td>index</td><td> $c ^ { * } ( z ; \nu )$ </td><td> $q ( \boldsymbol { z } _ { t } \mid \boldsymbol { z } )$ </td></tr><tr><td>0</td><td> $x _ { \star } + \sigma \xi / \sqrt { \nu }$ </td><td>frames</td><td> $\propto e ^ { - z ^ { 2 } / ( 2 \nu \sigma ^ { 2 } ) }$ </td><td> $\mathcal { N } ( \alpha _ { t } z , \sigma ^ { 2 } \alpha _ { t } \Delta _ { t } )$ </td></tr><tr><td>1</td><td> $\mathrm { P o i s } ( \nu x _ { \star } ) / \nu$ </td><td>exposure</td><td> $\nu ^ { z } / z !$ </td><td> $\mathrm { B i n o m i a l } ( z , \alpha _ { t } )$ </td></tr><tr><td>(1,2)</td><td> $Z _ { \nu } / \nu$ </td><td>exposure</td><td> $\mathrm { s e r i e s ^ { * } }$ </td><td> $\mathrm { B i n o m . } \to \mathrm { B e t a } ^ { * }$ </td></tr><tr><td>2</td><td> $\Gamma ( \nu , \nu / x _ { \star } )$ </td><td>looks</td><td> $z ^ { \nu - 1 } / \Gamma ( \nu )$ </td><td> $z _ { t } / z \sim \mathrm { B e t a } ( \nu ( t ) , \Delta _ { t } )$ </td></tr></table>

$\mathrm { I e r e } \ z : = z _ { t - 1 } , \alpha _ { t } = \nu ( t ) / \nu ( t - 1 ) , \Delta _ { t } = \nu ( t - 1 ) - \nu ( t )$ , and $\xi \sim \mathcal { N } ( 0 , 1 )$ . For $1 < p < 2 , Z _ { \nu }$ is compound Poisson–Gamma and the asterisk denotes its latent two-stage construction. The Gamma entry in this table uses the shape–rate parameterization. Here, x<sub>⋆</sub> denotes the clean image.

## F. Exponent range and Gaussian special case

We formulate p-Bridge over $\{ 0 \} \cup \{ 1 , 2 \}$ , covering the Gaussian model and the Poisson-to-Gamma continuum.

At $p = 0 .$ , allowing a fixed variance scale $V ( \mu ) = \sigma ^ { 2 }$ and writing $\sigma _ { t } ^ { 2 } = \sigma ^ { 2 } / \nu ( t )$ gives

$$
\begin{array} { r } { x _ { t } = x _ { \star } + \sigma _ { t } \xi , \qquad \xi \sim \mathcal { N } ( 0 , I ) . } \end{array}\tag{16}
$$

These marginals have the VE-SDE / EDM noise parameterization [28], [30]. Their deterministic update in Equation (8) is nevertheless different from an EDM probability-flow Euler step. The coefficients on the current state are $\sigma _ { t - 1 } ^ { 2 } / \sigma _ { t } ^ { 2 }$ and $\sigma _ { t - 1 } / \sigma _ { t }$ respectively. The unit-variance convention gives squared-error deviance. Retaining $\sigma ^ { 2 }$ scales it by $\sigma ^ { - 2 }$ . Our conditional network covers $p \in [ 1 , 2 ]$ , with $p = 0$ evaluated as a separate branch.

Adding independent Gaussian read noise generally leaves the single-exponent family. Moreover, a fixed read variance per exposure contributes $\sigma _ { \mathrm { r } } ^ { 2 } / \bar { \nu } ^ { 2 }$ to normalized intensity, rather than the $1 / \nu$ scaling in Equation (4). Extending the bridge to this setting requires a cumulative-noise model with compatible independent increments. For real data, the fitted exponent summarizes the variance–intensity relation over the calibration range.

## G. Training objective

We optimize ${ \mathcal { L } } = \lambda _ { \mathrm { { r e c } } } { \mathcal { L } } _ { \mathrm { { r e c } } } + \lambda _ { \mathrm { { m o m } } } { \mathcal { L } } _ { \mathrm { { m o m } } } + \lambda _ { \mathrm { { c o n s } } } { \mathcal { L } } _ { \mathrm { { c o n s } } } .$ . The reconstruction term uses a constant schedule weight,

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { r e c } } = \mathbb { E } [ w d _ { p } ( x _ { \star } , \hat { x } _ { \star } ( x _ { t } ; t , x _ { \mathrm { o b s } } ) ) ] , } \\ & { \quad w = \mathbb { E } _ { t } [ \nu ( t ) / \nu _ { \mathrm { o b s } } ] , } \end{array}\tag{17}
$$

where t follows the training sampling distribution. The likelihood-ratio scale gives a time-dependent weight proportional to $\nu ( t )$ . We replace the normalized weight $\nu ( t ) / \nu _ { \mathrm { o b s } }$ with its mean over training steps. This preserves the average weight and the pointwise reconstruction optimum while distributing weight uniformly across steps. The corresponding ablation is reported in Table VI. The conditional model additionally uses the exponent-dependent loss weight $\mu _ { \mathrm { r e f } } ^ { 2 - p }$

The moment term is

$$
{ \mathcal { L } } _ { \mathrm { m o m } } = ( { \bar { \varepsilon } } ) ^ { 2 } + { \widehat { \left( \mathrm { V a r } ( \varepsilon ) - 1 \right) ^ { 2 } } } ,\tag{18}
$$

using the Pearson residual

$$
\varepsilon = \frac { x _ { \mathrm { o b s } } - \hat { x } _ { \star } } { \sqrt { V ( \hat { x } _ { \star } ) / \nu _ { \mathrm { o b s } } } } .\tag{19}
$$

Under oracle predictions, each residual has zero mean and unit variance. Higher-order moments depend on the noise model. At $p = 1$ , residual skewness is $( \nu _ { \mathrm { o b s } } x _ { \star } ) ^ { - 1 / 2 } . \mathrm { A t } \ p = 2$ , the term reduces to a constraint on the speckle ratio variance, whereas at $p = 1$ , it standardizes count residuals by their predicted Poisson variance. Spatial dependence affects the reliability of pooled estimates. For learned predictions, this term regularizes residual statistics and can shift the reconstruction optimum.

The trajectory term supervises a network evaluation at a state generated by one deterministic update:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { c o n s } } = \mathbb { E } [ \nu ( t ^ { \prime } ) d _ { p } ( x _ { \star } , \hat { x } _ { \star } ( x _ { t ^ { \prime } } ; t ^ { \prime } , x _ { \mathrm { o b s } } ) ) ] , } \\ & { \quad x _ { t ^ { \prime } } = \tilde { \alpha } x _ { t } + ( 1 - \tilde { \alpha } ) \mathrm { s g } ( \hat { x } _ { \star } ) , } \\ & { \quad \quad \tilde { \alpha } = \nu ( t ) / \nu ( t ^ { \prime } ) , \qquad t ^ { \prime } < t . } \end{array}\tag{20}
$$

The stop-gradient operator blocks differentiation through the prediction that generates $\boldsymbol { x } _ { t ^ { \prime } }$ . Reconstruction optimality on the forward training distribution need not imply stationarity on this induced distribution, even when their supports overlap (Section S7). Training settings and implementation details are provided in Sections IV-A and S10.

## IV. EXPERIMENTS

We evaluate p-Bridge on synthetic noise spanning the exponent and severity axes, followed by real fluorescence microscopy and SAR data. The experiments examine exponent mismatch and conditioning, compare restoration methods, and evaluate noise parameter estimation and robustness to exponent.

## A. Experimental setup

Training data. We use a pool of 1,100 clean natural images from BSDS500 and DIV2K [53], [54], with 256 × 256 training crops. Clean intensities are mapped to $\mathcal { X } = [ 0 . 0 5 , 1 ]$ using the intensity offset. Noisy observations are generated from the corresponding Tweedie model.

Synthetic evaluation. We evaluate on 64 held-out crops with noise exponents $p _ { \mathrm { d a t a } } \in \{ 1 , 1 . 2 5 , 1 . 5 , 1 . 7 5 , 2 \}$ and severities $\eta \in \{ 0 . 2 5 , 0 . 5 , 1 \}$ (mild, moderate, and severe). Each fixedexponent model is tested across all five exponents, with results averaged over three corruption seeds. Comparisons with external methods use $p _ { \mathrm { d a t a } } \in \{ 1 , 1 . 5 , 2 \}$

Real-data evaluation. Noise calibration covers 12 FMD fluorescence microscopy configurations [11] and three SAR sensors: miniSAR, FARAD-X, and TerraSAR-X. Restoration comparisons use three FMD configurations, following the prior protocol, and all three SAR sensors. For each FMD configuration, we use 19 fields for fine-tuning over 20,000 steps and one held-out field for evaluation. Dataset-specific preprocessing and calibration are described in the corresponding experimental subsections.

TABLE II  
MINIMUM PSNR (DB) ACROSS TESTED NOISE EXPONENTS, AVERAGED OVER THREE CORRUPTION SEEDS AT NFE = 5. “TOLD” AND “BLIND” USE TRUE AND ESTIMATED EXPONENTS, RESPECTIVELY. “ENVELOPE” USES ORACLE-SELECTED SPECIALISTS. ∆ IS THE BLIND MODEL’S GAIN OVER THE BEST FIXED-EXPONENT MODEL. BOLD HIGHLIGHTS THE BEST FIXED-EXPONENT RESULT AND THE CONDITIONAL RESULT.
<table><tr><td rowspan="2">severity</td><td colspan="5">fixed specialist, worst case over pdata</td><td colspan="3">conditional</td><td rowspan="2">envelope</td></tr><tr><td>p=1</td><td>1.25</td><td>1.5</td><td>1.75</td><td>2</td><td>told  $p _ { \mathrm { d a t a } }$ </td><td>blind p</td><td>∆</td></tr><tr><td>severe</td><td>18.997</td><td>20.150</td><td>21.339</td><td>20.036</td><td>18.007</td><td>22.260</td><td>22.260</td><td>+0.921</td><td>22.254</td></tr><tr><td>moderate</td><td>20.921</td><td>21.899</td><td>23.112</td><td>22.903</td><td>21.747</td><td>24.635</td><td>24.637</td><td>+1.525</td><td>24.631</td></tr><tr><td>mild</td><td>24.142</td><td>25.052</td><td>26.022</td><td>26.336</td><td>25.551</td><td>27.436</td><td>27.435</td><td>+1.099</td><td>27.431</td></tr></table>

TABLE III

SYNTHETIC-IMAGE DENOISING RESULTS. PSNR (DB) / SSIM ARE AVERAGED OVER THREE CORRUPTION SEEDS. BOLD AND UNDERLINE MARK THE BEST AND SECOND-BEST RESULTS PER METRIC. “—” DENOTES NO TRAINED PARAMETERS.
<table><tr><td rowspan="2">method</td><td rowspan="2">params (M) NFE</td><td rowspan="2"></td><td colspan="3">severe, η=1</td><td colspan="3">moderate, η=0.5</td><td colspan="3">mild,  $\eta { = } 0 . 2 5$ </td></tr><tr><td>pdata=1</td><td>1.5</td><td>2</td><td>1</td><td>1.5</td><td>2</td><td>1</td><td>1.5</td><td>2</td></tr><tr><td colspan="10">Classical, transform-based, and Gaussian-bridge baselines</td></tr><tr><td>BM3D [1]</td><td></td><td></td><td>18.63/.493</td><td>17.59/.479</td><td>16.87/.467</td><td>23.19/.632</td><td>22.70/.622</td><td>22.08/.609</td><td>26.58/.763</td><td>26.21/.747</td><td>25.63/.724</td></tr><tr><td>VST + BM3D [17], [43]</td><td></td><td>1</td><td>21.21/.485</td><td>20.12/.434</td><td>19.86/.462</td><td>23.87/.634</td><td>23.77/.638</td><td>23.62/.641</td><td>26.89/.778</td><td>26.76/.778</td><td>26.59/.776</td></tr><tr><td>Noise2VST [20]</td><td>32.64</td><td>1</td><td>21.99/.543</td><td>21.53/.521</td><td>21.05/.499</td><td>24.71/.691</td><td>24.56/.689</td><td>24.40/.685</td><td>27.66/.812</td><td>27.55/.812</td><td>27.38/.811</td></tr><tr><td>p-Bridge, p=0</td><td>15.19</td><td>5</td><td></td><td>18.67/.40717.51/.37416.97/.357</td><td></td><td>21.39/.541</td><td>20.04/.503</td><td>19.13/.480</td><td>24.85/.701</td><td>23.52/.666</td><td>22.42/.642</td></tr><tr><td>Supervised backbones</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>NAFNet [7]</td><td>29.16</td><td>1</td><td></td><td>22.30/.57222.23/.575</td><td>22.16/.57624.64/.692</td><td></td><td></td><td>24.57/.693 24.48/.695</td><td></td><td></td><td>27.50/.81027.41/.81127.28/.811</td></tr><tr><td>Restormer [8]</td><td>26.11</td><td>1</td><td>22.17/.570</td><td>22.09/.570</td><td>22.02/.573</td><td>24.55/.688</td><td>24.47/.689</td><td>24.37/.691</td><td>27.43/.80627.35/.808</td><td></td><td>27.22/.808</td></tr><tr><td>EAMamba [51]</td><td>25.32</td><td>1</td><td>22.28/.574</td><td>22.21/.57622.13/.578</td><td></td><td>24.66/.695</td><td>24.59/.697</td><td>24.50/.698</td><td></td><td></td><td>27.50/.811 27.44/.813 27.31/.813</td></tr><tr><td>Diffusion-based methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DDRM [38]</td><td>15.11</td><td>5</td><td>20.36/.445</td><td>19.54/.401 18.74/.371 22.25/.542</td><td></td><td></td><td></td><td>21.36/.492 20.52/.448</td><td></td><td></td><td>23.40/.62723.31/.62023.13/.605</td></tr><tr><td>I2SB [21]</td><td>15.11</td><td>5</td><td></td><td>21.61/.455 21.58/.461</td><td>21.51/.465</td><td>23.95/.608</td><td>23.94/.612</td><td>23.85/.612</td><td>27.07/,774</td><td>27.10/.781</td><td>26.99/.781</td></tr><tr><td>ResShift [31]</td><td>15.11</td><td>5</td><td></td><td>21.68/.54421.60/.54521.51/.54624.05/.667</td><td></td><td></td><td>23.99/.669</td><td></td><td></td><td></td><td>23.90/.67026.77/.78926.73/.79126.63/.791</td></tr><tr><td>Noise-model estimation methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FBI-Denoiser [52]</td><td>2.49</td><td>1</td><td>17.44/.238</td><td>16.42/.210</td><td>12.96/.110</td><td>22.74/.509</td><td>20.19/.350</td><td>15.23/.165</td><td>18.85/.240</td><td>15.94/.178</td><td>12.97/.103</td></tr><tr><td>VIRNet [14]</td><td>10.53</td><td>1</td><td></td><td>22.32/.57022.26/.572</td><td>22.18/.574</td><td>24.72/.695</td><td>24.66/.697</td><td>24.58/.700</td><td>27.55/.812</td><td></td><td>27.48/.81427.36/.814</td></tr><tr><td>Ours</td><td></td><td>5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>p-Bridge, fixed p=1.5</td><td>15.19 15.28</td><td>5</td><td>21.35/.465 22.39/.577</td><td>22.34/.58021.34/.51623.83/.61124.73/.70423.11/.627 22.33/.579</td><td>22.26/.58224.79/.70024.73/.703</td><td></td><td></td><td>24.64/.704</td><td>27.64/.816</td><td></td><td>26.99/.77127.58/.81926.02/.763</td></tr><tr><td>p-Bridge, conditional blind p</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>27.57/.818</td><td>27.44/.819</td></tr></table>

TABLE IV

FMD DENOISING RESULTS. PSNR (DB) / SSIM ARE AVERAGED OVER THREE SEEDS ON THE HELD-OUT FIELD. BOLD AND UNDERLINE MARK THE BEST AND SECOND-BEST RESULTS PER METRIC. <sup>∗</sup>WITHOUT FINE-TUNING.
<table><tr><td>method</td><td>CF-FISH</td><td>CF-MICE</td><td>TP-MICE</td><td>mean</td></tr><tr><td>BM3D [1]*</td><td>27.01/.826</td><td>33.18/.942</td><td>31.63/.886</td><td>30.61/.885</td></tr><tr><td>VST + BM3D [17], [43]*</td><td>25.73/.809</td><td>33.62/.946</td><td>31.55/.889</td><td>30.30/.881</td></tr><tr><td>Noise2VST [20]*</td><td>28.58/.879</td><td>34.02/.949</td><td>32.14/.895</td><td>31.58/.908</td></tr><tr><td>p-Bridge,  $p _ { \mathrm { m o d e l } } { = } 0 ^ { * }$ </td><td>24.68/.811</td><td>31.18/.932</td><td>30.45/.887</td><td>28.77/.877</td></tr><tr><td>NAFNet [7]</td><td>28.23/.878</td><td>34.06/.951</td><td>32.61/.898</td><td>31.63/.909</td></tr><tr><td>Restormer [8]</td><td>28.00/.873</td><td>34.19/.951</td><td>32.39/.898</td><td>31.52/.908</td></tr><tr><td>EAMamba [51]</td><td>28.00/.874</td><td>34.13/.951</td><td>32.51/.898</td><td>31.55/.908</td></tr><tr><td>DDRM [38]†</td><td>22.67/.539</td><td>24.98/.546</td><td>24.88/.399</td><td>24.18/.495</td></tr><tr><td>I2SB [21]</td><td>28.25/.867</td><td>34.03/.948</td><td>31.19/.880</td><td>31.15/.898</td></tr><tr><td>ResShift [31]</td><td>27.98/.871</td><td>33.28/.946</td><td>31.70/.893</td><td>30.99/.903</td></tr><tr><td>FBI-Denoiser [52]</td><td>21.67/.624</td><td>32.10/.888</td><td>30.57/.844</td><td>28.12/.785</td></tr><tr><td>VIRNet [14]</td><td>28.70/.883</td><td>34.24/.952</td><td>32.55/.899</td><td>31.83/.911</td></tr><tr><td>p-Bridge,  $p _ { \mathrm { m o d e l } } { = } 1 . 5 ^ { \ast }$ </td><td>27.47/.868</td><td>33.47/.944</td><td>31.70/.891</td><td>30.88/.901</td></tr><tr><td>p-Bridge, conditional</td><td>28.59/.885</td><td>34.01/.952 32.68/.899</td><td></td><td>31.76/.912</td></tr></table>

<sup>†</sup>DDRM uses the realised per-image noise level.

Evaluation metrics. We report PSNR and SSIM against clean synthetic images or the FMD frame-average reference. The supplementary sampling-budget study also reports dark-region log-RMSE (Table S2). For real SAR, where clean references are unavailable, we report the ratio-image measures M [55] and RIS [56] alongside visual comparisons.

TABLE V  
REAL SAR DESPECKLING RESULTS. M/RIS ARE AVERAGED OVER SENSORS. LOWER IS BETTER. BOLD AND UNDERLINE MARK THE BEST AND SECOND-BEST RESULTS PER METRIC.
<table><tr><td rowspan="2">method</td><td colspan="4">M/RIS, lower better</td></tr><tr><td></td><td></td><td>miniSAR FARAD-X TerraSAR-X</td><td>mean</td></tr><tr><td>BM3D [1]</td><td>32.5/55.1</td><td>41.0/78.1</td><td>49.2/109.5</td><td>40.9/80.9</td></tr><tr><td>VST + BM3D [17], [43]</td><td>34.7/46.4</td><td>35.8/64.2</td><td>45.4/77.9</td><td>38.6/62.8</td></tr><tr><td>Noise2VST [20]</td><td>25.5/37.9</td><td>30.5/50.3</td><td>39.6/77.5</td><td>31.9/55.2</td></tr><tr><td>p-Bridge, pmodel=0</td><td>36.3/63.8</td><td>39.9/73.6</td><td>46.7/100.9</td><td>40.9/79.4</td></tr><tr><td>NAFNet [7]</td><td>12.0/17.4</td><td>21.8/34.7</td><td>32.2/60.8</td><td>22.0/37.7</td></tr><tr><td>Restormer [8]</td><td>29.9/48.2</td><td>41.3/78.0</td><td>47.4/103.3</td><td>39.5/76.5</td></tr><tr><td>EAMamba [51]</td><td>25.4/39.6</td><td>31.5/53.6</td><td>43.4/88.9</td><td>33.4/60.7</td></tr><tr><td>DDRM [38]</td><td>20.9/30.8</td><td>25.2/38.4</td><td>34.2/58.8</td><td>26.7/42.7</td></tr><tr><td>I2SB [21]</td><td>12.7/18.5</td><td>30.0/50.1</td><td>34.3/65.6</td><td>25.7/44.7</td></tr><tr><td>ResShift [31]</td><td>23.8/37.1</td><td>35.4/62.6</td><td>42.7/88.1</td><td>33.9/62.6</td></tr><tr><td>FBI-Denoiser [52]</td><td>37.2/65.0</td><td>43.0/81.4</td><td>45.5/93.9</td><td>41.9/80.1</td></tr><tr><td>VIRNet [14]</td><td>18.2/28.0</td><td>29.8/50.3</td><td>38.6/75.5</td><td>28.8/51.3</td></tr><tr><td>p-Bridge,  $p _ { \mathrm { m o d e l } } { = } 1 . 5$ </td><td>33.3/56.4</td><td>43.2/83.8</td><td>49.6/111.7</td><td>42.0/84.0</td></tr><tr><td>p-Bridge, conditional</td><td>7.3/14.2</td><td>22.4/37.4</td><td>28.6/54.4</td><td>19.4/35.3</td></tr></table>

Model and training. We use a UNet with time embedding, containing 15.19M parameters for fixed-exponent models and

15.28M with exponent conditioning. The conditional model samples $p$ uniformly from $\{ 1 , 1 . 0 5 , \ldots , 2 \}$ per micro-batch. Each bridge model is trained once for 50,000 steps with global batch size 24, 1,000 warmup steps, and cosine learning rate decay from $1 0 ^ { - 4 } ~ \mathrm { { t o } ~ 1 0 ^ { - 7 } }$ . Loss weights are fixed at $( \lambda _ { \mathrm { r e c } } , \lambda _ { \mathrm { m o m } } , \lambda _ { \mathrm { c o n s } } ) = ( 1 0 , 1 , 5 )$ . Implementation details are provided in the supplement.

Noise calibration. We compare different noise models by matching their relative noise at a reference intensity. We use $\mu _ { \mathrm { r e f } } = 0 . 4 6 4 8 6 4$ , the mean intensity of the offset training pool, and define the noise level η as the ratio of noise standard deviation to this reference intensity. The required measurement index is then

$$
\nu _ { \mathrm { o b s } } ( p ) = \eta ^ { - 2 } \mu _ { \mathrm { r e f } } ^ { p - 2 } .\tag{21}
$$

This calibration matches noise levels at $\mu _ { \mathrm { r e f } }$ . Away from this reference, relative noise increases towards darker regions for $p < 2 ,$ whereas it remains constant across intensities for $p = 2$ Figure 4 illustrates how the noise exponent affects the spatial distribution of relative noise.

Inference and severity transfer. Unless otherwise stated, p-Bridge uses deterministic sampling with five network evaluations. Models trained at severe noise $( \eta = 1 )$ are also evaluated at moderate and mild noise without retraining. For a less noisy input, sampling starts at its corresponding measurement index within the trained range. For real-image evaluation, calibrated noise-plane coordinates $( p , \nu )$ configure the bridge and select the starting point for denoising.

## B. Behaviour along the exponent axis

The cost of a mismatched exponent. Figure 5 compares models trained at different noise exponents on the same noisy inputs. Each column fixes $p _ { \mathrm { d a t a } }$ , allowing direct comparison with the matched model. At severe noise, applying a Gammatrained model $( p _ { \mathrm { m o d e l } } = 2 )$ to Poisson data $( p _ { \mathrm { d a t a } } = 1 )$ reduces PSNR by 4.40 dB, the largest loss across the three noise levels. The reverse mismatch costs approximately 3.3 dB, showing that the two directions are not equally harmful. Across the mismatched pairs in the $p \in \{ 1 , 1 . 5 , 2 \}$ subgrid, the average loss at severe noise is 2.00 dB.

On the $5 \times 5$ evaluation grid, the matched model achieves the highest PSNR in every column at all three noise levels. Performance differences remain observable even between adjacent exponents separated by 0.25, indicating sensitivity to relatively small mismatches. Since no single fixed-exponent model is optimal across the full range, we next examine which exponent provides the best worst-case performance when the noise exponent is unknown.

The minimax exponent. When the noise exponent is unknown, we evaluate each fixed-exponent model by its lowest PSNR across the tested noise exponents. Table II shows that $p = 1 . 5$ maximizes this worst-case PSNR at severe and moderate noise, reaching 21.339 and 23.112 dB, respectively. At mild noise, the optimum shifts to $p = 1 . 7 5$ , with 26.336 dB. Thus, an interior exponent offers the strongest worst-case performance among the tested fixed models, but no single exponent is optimal at every noise level. We therefore use $p = 1 . 5$ as a fixed-model baseline and next examine whether conditioning on the noise exponent improves robustness across the full range.

![](images/74721334e4dfc1290922263c44f408f5a0c534695fc04503b6d0aa6ff69f0620.jpg)  
Fig. 4. Effect of the noise exponent at matched relative noise. Top: clean image $x _ { 0 }$ and noisy observations; percentages give the fraction of pixels at the intensity floor. Bottom: relative noise $\mathrm { S D } / x _ { 0 } .$ , normalized at $\mu _ { \mathrm { r e f } } ;$ values give the maximum-to-minimum ratio.

The p-conditional network. To cover different noise exponents with a single model, we condition the network on $p$ through an embedding alongside the time embedding. During training, p is sampled uniformly from $\{ 1 , 1 . 0 5 , \ldots , 2 \}$ for each micro-batch.

Table II compares the conditional network with fixedexponent models in terms of worst-case PSNR across noise exponents. Using estimated rather than true exponents, the conditional network improves over the fixed $p = 1 . 5$ model by 0.921, 1.525, and 1.413 dB at severe, moderate, and mild noise, respectively. Against the best fixed-exponent model at each severity, the gains are 0.921, 1.525, and 1.099 dB. The smaller mild-noise gain reflects the stronger $p = 1 . 7 5$ baseline.

We also compare the conditional network with an oracleselected set of five specialist models, one for each tested exponent. At severe, moderate, and mild noise, the specialists achieve minimum PSNRs across the tested exponents of 22.254, 24.631, and 27.431 dB, compared with 22.260, 24.637, and 27.435 dB for the conditional network using estimated exponents. The minimum PSNRs differ by at most 0.006 dB across the three noise levels.

Using estimated exponents yields nearly identical minimum PSNR to using the true exponents. In Table II, the differences between these two inputs to the same conditional network are 0.000, +0.002, and −0.001 dB at severe, moderate, and mild noise, respectively.

To verify that the network uses the exponent input, we vary the conditioning exponent while keeping the data exponent fixed. The matching exponent achieves the highest PSNR for all five tested data exponents at every noise level, confirming that the conditioning input affects restoration performance.

## C. Comparison

We compare p-Bridge with baseline methods on synthetic images, FMD fluorescence microscopy, and real SAR data. We report PSNR and SSIM for synthetic and FMD images, and ratio-image metrics with visual comparisons for SAR.

Synthetic images. Table III reports results for nine combinations of noise exponent and severity. Compared with external baselines, conditional p-Bridge achieves the highest PSNR in eight settings and the highest SSIM in all nine. Noise2VST leads only in PSNR at mild noise with $p _ { \mathrm { d a t a } } = 1$ , by 0.02 dB. Improvements over VIRNet are small but consistent, ranging from 0.06 to 0.09 dB. Performance also varies less across noise exponents. At severe noise, PSNR drops by 0.13 dB from $p _ { \mathrm { d a t a } } = 1 ~ \mathrm { t o } ~ 2$ , compared with 0.94 dB for Noise2VST. Figure 6 provides visual comparisons.

<table><tr><td rowspan=3 colspan=1>1</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>severe</td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=6>moderate</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>mild</td><td rowspan=7 colspan=1>-4.0-3.53.0</td></tr><tr><td rowspan=2 colspan=1>22.41</td><td rowspan=2 colspan=1>22.09</td><td rowspan=2 colspan=1>20.99</td><td rowspan=2 colspan=1>19.89</td><td rowspan=2 colspan=1>19.00</td><td></td><td rowspan=2 colspan=1>24.81</td><td rowspan=2 colspan=1>24.38</td><td rowspan=2 colspan=1>23.07</td><td rowspan=2 colspan=1>21.87</td><td rowspan=1 colspan=1>20.92</td><td rowspan=1 colspan=1></td><td rowspan=2 colspan=1>27.67</td><td rowspan=2 colspan=1>27.33</td><td rowspan=2 colspan=1>26.24</td><td rowspan=2 colspan=1>25.10</td><td rowspan=2 colspan=1>24.14</td></tr><tr><td></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=4 colspan=1>1.25</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=3 colspan=1>22.23</td><td rowspan=3 colspan=1>22.37</td><td rowspan=3 colspan=1>22.10</td><td rowspan=3 colspan=1>21.15</td><td rowspan=2 colspan=1>20.15</td><td></td><td rowspan=2 colspan=1>24.62</td><td rowspan=3 colspan=1>24.78</td><td rowspan=3 colspan=1>24.38</td><td rowspan=3 colspan=1>23.11</td><td rowspan=2 colspan=1>21.90</td><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1>27.53</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td rowspan=2 colspan=1>27.62</td><td rowspan=2 colspan=1>27.31</td><td rowspan=2 colspan=1>26.23</td><td rowspan=2 colspan=1>25.05</td></tr><tr><td rowspan=1 colspan=1></td><td></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=7 colspan=1>pmodel1.51.752</td><td rowspan=1 colspan=1>21.35</td><td rowspan=1 colspan=1>22.21</td><td rowspan=1 colspan=1>22.34</td><td rowspan=1 colspan=1>22.10</td><td rowspan=1 colspan=1>21.34</td><td></td><td rowspan=1 colspan=1>23.83</td><td rowspan=1 colspan=1>24.59</td><td rowspan=1 colspan=1>24.73</td><td rowspan=1 colspan=1>24.35</td><td rowspan=1 colspan=1>23.11</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>26.99</td><td rowspan=1 colspan=1>27.46</td><td rowspan=1 colspan=1>27.58</td><td rowspan=1 colspan=1>27.21</td><td rowspan=1 colspan=1>26.02</td><td rowspan=1 colspan=1>-2.5-2.0</td></tr><tr><td rowspan=3 colspan=1>20.04</td><td rowspan=3 colspan=1>21.46</td><td rowspan=3 colspan=1>22.18</td><td rowspan=3 colspan=1>22.28</td><td rowspan=3 colspan=1>22.11</td><td></td><td rowspan=3 colspan=1>22.90</td><td rowspan=3 colspan=1>23.90</td><td rowspan=3 colspan=1>24.55</td><td rowspan=3 colspan=1>24.67</td><td rowspan=3 colspan=1>24.36</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1>26.34</td><td rowspan=2 colspan=1>26.94</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td rowspan=1 colspan=1>27.41</td><td rowspan=1 colspan=1>27.50</td><td rowspan=1 colspan=1>27.16</td><td rowspan=1 colspan=1>-1.5-1.0</td></tr><tr><td rowspan=1 colspan=1>18.01</td><td rowspan=1 colspan=1>19.66</td><td rowspan=1 colspan=1>21.32</td><td rowspan=1 colspan=1>22.11</td><td rowspan=1 colspan=1>22.25</td><td></td><td rowspan=1 colspan=1>21.75</td><td rowspan=1 colspan=1>22.78</td><td rowspan=1 colspan=1>23.81</td><td rowspan=1 colspan=1>24.48</td><td rowspan=1 colspan=1>24.63</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>25.55</td><td rowspan=1 colspan=1>26.19</td><td rowspan=1 colspan=1>26.84</td><td rowspan=1 colspan=1>27.29</td><td rowspan=1 colspan=1>27.43</td><td rowspan=3 colspan=1>-0.5B-0.0</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1.25</td><td rowspan=1 colspan=1>1.5</td><td rowspan=1 colspan=1>1.75</td><td rowspan=1 colspan=2>2</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1.25</td><td rowspan=1 colspan=1>1.5</td><td rowspan=1 colspan=1>1.75</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1.25</td><td rowspan=1 colspan=1>1.5</td><td rowspan=1 colspan=1>1.75</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Pdata</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Pdata</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Pdata</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr></table>

Fig. 5. PSNR (dB) across model exponents (rows) and data exponents (columns), averaged over three corruption seeds at NFE = 5. Bold marks matched exponents, and shading shows the PSNR loss relative to the best result in each column.

observation  
BM3D  
VST+BM3D  
Noise2VST  
Gaussian-bridge  
NAFNet  
Restormer  
![](images/4c829c0089fa1a0631a92a0e361413a87a0230639003d73a4ded7466b4ec8f68.jpg)  
Fig. 6. Synthetic-image restorations at moderate noise $( \eta = 0 . 5 )$ , with one row per noise exponent. Corner values show PSNR (dB).

Fluorescence microscopy (FMD). Table IV reports results on three FMD configurations. Conditional p-Bridge achieves the highest mean SSIM (0.912) and the second-highest mean PSNR (31.76 dB), compared with 31.83 dB for VIRNet. The PSNR ranking varies by configuration. Our model trails VIRNet by 0.11 and 0.23 dB on CF-FISH and CF-MICE, respectively, but leads on TP-MICE with 32.68 dB, a 0.13 dB gain. For SSIM, it ranks first on CF-FISH and shares first place with VIRNet on the other two configurations. Its mean PSNR also exceeds I<sup>2</sup>SB by 0.61 dB. Noise parameters are calibrated from training-field frame stacks. All comparisons follow the same protocol. The table distinguishes fine-tuned methods from those transferred without fine-tuning. Figure 7 shows the restoration results.

SAR images. Table V reports results on miniSAR, FARAD-X, and TerraSAR-X, using eight windows per sensor. Among the ranked methods, p-Bridge achieves the lowest mean M and RIS (19.4/35.3), followed by NAFNet (22.0/37.7). It ranks first on both metrics for miniSAR and TerraSAR-X, second for FARAD-X, where NAFNet scores lower by 0.6 and 2.7, respectively. Figure 8 complements these ratio-image metrics with visual comparisons of noise removal and detail preservation. Noise calibration is detailed in the supplement.

![](images/f8ded0893ec6fa0c411ec6ed7bd500eee7f865a8bf399a65c3ceaa9576b4fb14.jpg)

Fig. 7. FMD restorations, one crop per configuration. Corner values show PSNR (dB).  
![](images/81954744a957112d5ed30587f0f0f1b547d68a60134ef0dd31356ed582ada79b.jpg)  
Fig. 8. Real SAR restorations, one window per sensor. Corner values show M (lower is better).

TABLE VI  
COMPONENT ABLATIONS AT SEVERE NOISE AND ${ \mathrm { N F E } } = 5 ,$ , USING THE TRUE EXPONENT. EACH ROW CHANGES ONE SETTING. MINIMUM PSNR (DB) IS TAKEN OVER THE TESTED DATA EXPONENTS. ∆ IS THE DIFFERENCE FROM THE FULL MODEL.
<table><tr><td>component</td><td>full model</td><td>ablation</td><td>min. PSNR</td><td> $\Delta$ </td></tr><tr><td>Full conditional model</td><td></td><td></td><td>22.260</td><td>1</td></tr><tr><td>Reconstruction weight</td><td> $\mathbb { E } _ { t } [ \nu ( t ) / \nu _ { \mathrm { o b s } } ]$ </td><td> $\nu ( t ) / \nu _ { \mathrm { o b s } }$ </td><td>21.916</td><td>-0.345</td></tr><tr><td>Exponent-dependent weight</td><td> $\mu _ { \mathrm { r e f } } ^ { 2 - p }$ </td><td>1</td><td>22.248</td><td>-0.012</td></tr><tr><td>Moment regularization</td><td> $\lambda _ { \mathrm { m o m } } = 1$ </td><td> $\lambda _ { \mathrm { m o m } } = 0$ </td><td>22.251</td><td>-0.010</td></tr><tr><td>Consistency regularization</td><td> $\lambda _ { \mathrm { c o n s } } = 5$ </td><td> $\lambda _ { \mathrm { c o n s } } = 0$ </td><td>22.051</td><td>-0.210</td></tr><tr><td>Index schedule</td><td>Geometric</td><td> $\mathrm { L i n e a r }$ </td><td>21.992</td><td>-0.268</td></tr><tr><td>Reverse update</td><td> $\alpha _ { t } x _ { t } + ( 1 - \alpha _ { t } ) \hat { x } _ { \star }$ </td><td> $\operatorname* { m a x } ( \hat { x } _ { \star } , 0 )$ </td><td>22.075</td><td>-0.186</td></tr><tr><td>Output-range projection</td><td>Enabled</td><td>Disabled</td><td>22.260</td><td>-0.000</td></tr></table>

Reconstruction weights have the same mean over sampled training steps, and loss coefficients are unchanged in this comparison. In the reverse update, $\alpha _ { t } = \nu ( t ) / \nu ( t - 1 )$ . The ablated update replaces the state with the current prediction, with nonnegative clipping.

![](images/1c56f440358da3d83d6e52b0adbe0154088391d2c1b9411b87b8e2552b449f31.jpg)  
Fig. 9. Qualitative component ablation at severe noise, $p _ { \mathrm { d a t a } } = 1 . 5 .$ . Corner values report PSNR (dB).

## D. Ablation studies

Evaluation protocol. Table VI summarizes the component ablations of the conditional model. Each variant changes one setting while keeping the others fixed. Training-side variants are retrained with the same data and budget, whereas inferenceonly variants reuse the full model’s checkpoint.

Loss weighting and index schedule. Measurement-dependent weighting and a linear index schedule reduce PSNR by 0.345 and 0.268 dB, respectively. Although the mean reconstruction weight is unchanged, measurement-dependent weighting assigns greater weight to cleaner states. Likewise, a linear schedule allocates fewer training steps near the noisy observation than a geometric schedule over the same index range. These results support the use of constant weighting and geometric spacing to emphasize noisy states under a fixed training budget.

Regularization and reverse update. At five network evaluations, removing consistency regularization and replacing the reverse mixture with direct prediction reduce PSNR by 0.210 and 0.186 dB, respectively. These losses increase to 0.452 and 1.654 dB at 25 evaluations. At one evaluation, no reverse update is applied, so the two samplers produce identical outputs.

![](images/1e26119ffd8e6ae3785d85a472b6bc89cee5374e9762926f02dd0586b5f8e4cc.jpg)  
Fig. 10. Excess relative error over the matched-exponent model, by intensity decile and signed exponent error. The box marks $| \hat { p } - p | \leq 0 . 2 \dot { 5 }$ , and the gray strip shows the range of the estimator errors. Rows average exponent pairs with the same error.

TABLE VII  
PSNR GAINS (DB) OVER THE FIXED $p = 1 . 5$ MODEL UNDER WORST-CASE EXPONENT ERRORS. THE CONDITIONING EXPONENT SATISFIES $| \hat { p } - p _ { \mathrm { d a t a } } | \leq \delta$ WITHIN THE TESTED EXPONENT RANGE. $\delta = 0$ USES THE TRUE EXPONENT, WHIL $\operatorname { E } \delta = \infty$ REMOVES THE ERROR CONSTRAINT. POSITIVE VALUES FAVOR THE CONDITIONAL MODEL.
<table><tr><td>vs. fixed  $\scriptstyle p = 1 . 5$ </td><td>δ=0</td><td> $\delta { = } 0 . 2 5$ </td><td> $\delta { = } 0 . 5$ </td><td> $\delta \mathrm { = } \infty$ </td></tr><tr><td>severe</td><td> $+ 0 . 9 2 2$ </td><td> $+ 0 . 8 3 0$ </td><td> $+ 0 . 1 0 7$ </td><td> $- 1 . 7 0 7$ </td></tr><tr><td>moderate</td><td> $+ 1 . 5 2 3$ </td><td> $+ 1 . 3 1 1$ </td><td> $+ 0 . 1 8 4$ </td><td> $- 1 . 8 6 3$ </td></tr><tr><td>mild</td><td> $+ 1 . 4 1 4$ </td><td> $+ 1 . 2 3 3$ </td><td> $+ 0 . 1 7 7$ </td><td> $- 1 . 7 9 6$ </td></tr></table>

Removing consistency regularization changes PSNR by only 0.004 dB. By comparison, removing moment regularization or exponent-dependent loss weighting reduces PSNR by only 0.010 and 0.012 dB, respectively.

Figure 9 illustrates the structural degradation of ablations. No-projection result remains visually close to the full model. Output-range projection. With matched exponents, removing projection changes PSNR by at most 0.001 dB at one, five, and 25 network evaluations. Its benefit is more evident when the conditioning exponent is below the data exponent. For Gamma noise $( p _ { \mathrm { d a t a } } = 2 )$ with the conditioning exponent set to 1, removing projection reduces PSNR by 0.223, 0.168, and 0.139 dB, respectively. Projection therefore provides a modest benefit under exponent mismatch, with little effect on matchednoise performance.

## E. Noise parameter estimation and robustness

Tolerance to exponent estimation error. We first examine how accurately the exponent must be estimated to retain the benefit of conditioning. Table VII compares the conditional network with the fixed $p = 1 . 5$ model under worst-case errors satisfying $| \hat { p } - p _ { \mathrm { d a t a } } | \leq \delta .$ At $\delta \ : = \ : 0 . 2 5$ , the PSNR gains remain 0.830–1.311 dB across the three noise levels, only 0.09–0.21 dB below exact conditioning. At $\delta = 0 . 5$ , the gains decrease to 0.107–0.184 dB but remain positive. Without an error constraint over the tested exponent range, the conditional model falls below the fixed baseline by 1.707–1.863 dB. Thus, errors within 0.25 preserve most of the conditioning benefit in these tests.

Figure 10 shows how this cost varies with intensity. Overestimating the exponent concentrates excess error in dark regions, whereas underestimating it affects bright regions. The observed estimator errors remain within $\delta = 0 . 2 5$

![](images/1376365116b09637b103ed80a371ab765bee6163bd6af51a5dee86f74dcfda1f.jpg)  
Fig. 11. Variance–mean relations by detector class. Each configuration is normalized by its fitted variance at the median signal. Dashed lines mark $p = 0 , 1 , 2 ;$ shaded bands span the fitted exponents, and corner values give the class mean.

![](images/4960e8c85859e4bf7346d5e20822c39b95a9a7cffe813f8c14fb374a751b5ffa.jpg)  
Fig. 12. Fitted exponents by imaging configuration and detector class. Dots show medians, thick bars interquartile ranges, and thin lines full ranges; carets indicate values outside the displayed range.

Noise parameter estimation. We estimate the noise parameters directly from the noisy image without additional training. Taking logarithms of $\operatorname { V a r } ( x _ { \mathrm { o b s } } \mid \mu ) = \mu ^ { p } / \nu$ gives a linear relation with slope $p$ and intercept − log ν. We fit a robust regression of estimated log-noise variance on log-mean intensity. The fitted slope and intercept jointly determine $p$ and $\nu ,$ with dispersion absorbed into the measurement index. Details of local-statistic estimation and regression are provided in Section S10-D.

On the evaluation grid of five exponents, three noise levels, and three corruption seeds, the reported worst-case absolute error in $p$ is 0.0914, below the 0.25 target identified above. The median relative error in ν ranges from 2.8% to 4.1%. Mild noise is more challenging to estimate because, at a fixed exponent, ν is 16 times its severe-noise value, leaving a weaker noise signal. Additional tests of estimation bias and sensitivity to estimator settings are provided in Section S9-B.

Estimates outside the supported range. The regression does not constrain $\hat { p }$ to [1, 2]. In five of the 45 evaluated configurations, estimates fall outside this interval by at most 0.032, all at $p _ { \mathrm { { d a t a } } } = 1 ~ \mathrm { { o r } ~ 2 }$ . We clip $\hat { p }$ to [1, 2] to match the range supported by the conditional model and its bridge implementation. The blind results in Table II use these clipped estimates.

![](images/22af6a983609d177e05477a993908a748a8f5419e67b2950ec0c5039e7be8661.jpg)  
Fig. 13. Real and simulated FMD frames using the fitted exponent and index. Rows show the clean signal, real and simulated observations, and middle-band noise histograms (real filled; simulated outlined). Numbers are simulated-to real noise-standard-deviation ratios averaged over the dark and bright bands; values closer to 1 indicate better agreement.

![](images/5ee3719635804fd692bc27e13e2ff87309e60f5731d45452c82d0951cd8892a5.jpg)  
Fig. 14. Noise statistics versus candidate exponent for the fields in Figure 13. Left: simulated-to-real noise-standard-deviation ratios; right: simulated skew ness, for dark and bright bands. Curves closer to the horizontal match lines agree better with real data; vertical lines mark the fitted exponents.

Real-data calibration. Figure 11 shows the variance–mean relation used to fit the real-data exponents, and Figure 12 summarizes the estimates across imaging configurations. The fitted slopes differ across detector classes and configurations, supporting configuration-specific calibration rather than a single fixed exponent.

Calibration check on FMD. To check whether the fitted law reproduces the observed noise, Figure 13 compares real and simulated FMD frames at matched clean signal, exponent, and index. Figure 14 then varies the candidate exponent while refitting the corresponding index, showing how the simulated noise level and skew change around the fitted value.

## V. CONCLUSION

$p \textmd { - }$ Bridge connects statistical noise modeling with the design of a denoising trajectory. Measurement accumulation provides the common bridge structure, while exponent conditioning adapts the learned prediction to the noise law. The experiments show that this adaptation matters beyond selecting a noise level. Fixed-exponent models lose accuracy under mismatched noise laws, whereas a shared conditional network achieves cross-exponent robustness comparable to oracle-selected specialists. Using estimated exponents leaves this robustness nearly unchanged, and moderate conditioning errors preserve most of the benefit. Results on synthetic images, fluorescence microscopy, and SAR further support the usefulness of the formulation across imaging regimes. Together, these findings support adapting to different Tweedie noise models by making the noise law an explicit condition within a shared restoration model, without a separate checkpoint for each exponent.

## REFERENCES

[1] K. Dabov, A. Foi, V. Katkovnik, and K. Egiazarian, “Image denoising by sparse 3-d transform-domain collaborative filtering,” IEEE Transactions on image processing, vol. 16, no. 8, pp. 2080–2095, 2007.

[2] K. Zhang, Y. Li, W. Zuo, L. Zhang, L. Van Gool, and R. Timofte, “Plug and-play image restoration with deep denoiser prior,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 10, pp. 6360– 6376, 2021.

[3] A. Foi, M. Trimeche, V. Katkovnik, and K. Egiazarian, “Practical poissonian-gaussian noise modeling and fitting for single-image raw-data,” IEEE transactions on image processing, vol. 17, no. 10, pp. 1737–1754, 2008.

[4] J. Boulanger, C. Kervrann, P. Bouthemy, P. Elbau, J.-B. Sibarita, and J. Salamero, “Patch-based nonlocal functional for denoising fluorescence microscopy image sequences,” IEEE transactions on medical imaging, vol. 29, no. 2, pp. 442–454, 2009.

[5] J. W. Goodman, “Some fundamental properties of speckle,” JOSA, vol. 66, no. 11, pp. 1145–1150, 1976.

[6] K. Zhang, W. Zuo, Y. Chen, D. Meng, and L. Zhang, “Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising,” IEEE transactions on image processing, vol. 26, no. 7, pp. 3142–3155, 2017.

[7] L. Chen, X. Chu, X. Zhang, and J. Sun, “Simple baselines for image restoration,” in European conference on computer vision. Springer, 2022, pp. 17–33.

[8] S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, and M.-H. Yang, “Restormer: Efficient transformer for high-resolution image restoration,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 5728–5739.

[9] K. Zhang, W. Zuo, and L. Zhang, “Ffdnet: Toward a fast and flexible solution for cnn-based image denoising,” IEEE Transactions on Image Processing, vol. 27, no. 9, pp. 4608–4622, 2018.

[10] A. Abdelhamed, S. Lin, and M. S. Brown, “A high-quality denoising dataset for smartphone cameras,” in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition. IEEE, 2018, pp. 1692–1700.

[11] Y. Zhang, Y. Zhu, E. Nichols, Q. Wang, S. Zhang, C. Smith, and S. Howard, “A poisson-gaussian denoising dataset with real fluorescence microscopy images,” in 2019 IEEE/CVF conference on computer vision and pattern recognition (CVPR). IEEE, 2019, pp. 11 702–11 710.

[12] J. Lehtinen, J. Munkberg, J. Hasselgren, S. Laine, T. Karras, M. Aittala, and T. Aila, “Noise2Noise: Learning image restoration without clean data,” in ICML, ser. Proceedings of Machine Learning Research, vol. 80, 2018, pp. 2965–2974. [Online]. Available: https://proceedings.mlr.press/v80/lehtinen18a.html

[13] J. Batson and L. Royer, “Noise2self: Blind denoising by self-supervision,” in International conference on machine learning. PMLR, 2019, pp. 524–533.

[14] Z. Yue, H. Yong, Q. Zhao, L. Zhang, D. Meng, and K.-Y. K. Wong, “Deep variational network toward blind image restoration,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 11, pp. 7011– 7026, 2024.

[15] K. Wei, Y. Fu, Y. Zheng, and J. Yang, “Physics-based noise modeling for extreme low-light photography,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 11, pp. 8520–8537, 2021.

[16] H. Feng, L. Wang, Y. Huang, Y. Wang, L. Zhu, and H. Huang, “Learning physics-informed noise models from dark frames for low-light raw image denoising,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026.

[17] F. J. Anscombe, “The transformation of poisson, binomial and negativebinomial data,” Biometrika, vol. 35, no. 3/4, pp. 246–254, 1948.

[18] M. Makitalo and A. Foi, “Optimal inversion of the generalized anscombe transformation for poisson-gaussian noise,” IEEE transactions on image processing, vol. 22, no. 1, pp. 91–103, 2012.

[19] E. Dalsasso, X. Yang, L. Denis, F. Tupin, and W. Yang, “Sar image despeckling by deep neural networks: From a pre-trained model to an end-to-end training strategy,” Remote Sensing, vol. 12, no. 16, p. 2636, 2020.

[20] S. Herbreteau and M. Unser, “Self-calibrated variance-stabilizing transformations for real-world image denoising,” in 2025 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2025, pp. 10 496–10 506.

[21] G.-H. Liu, A. Vahdat, D.-A. Huang, E. A. Theodorou, W. Nie, and A. Anandkumar, “I<sup>2</sup>SB: Image-to-image schrödinger bridge,” arXiv preprint arXiv:2302.05872, 2023.

[22] H. Chung, J. Kim, M. T. Mccann, M. L. Klasky, and J. C. Ye, “Diffusion posterior sampling for general noisy inverse problems,” in The Eleventh International Conference on Learning Representations, 2023. [Online]. Available: https://openreview.net/forum?id=OnD9zGAGT0k

[23] Y. Xie, M. Yuan, B. Dong, and Q. Li, “Diffusion model for generative image denoising,” arXiv preprint arXiv:2302.02398, 2023.

[24] X. Hu, Y. Zhu, T. Wang, J. Li, and W. Zhao, “γ-Bridge: A look-parametric diffusion bridge,” arXiv preprint arXiv:2607.22719, 2026.

[25] M. C. Tweedie et al., “An index which distinguishes between some important exponential families,” in Statistics: Applications and new directions: Proc. Indian statistical institute golden Jubilee International conference, vol. 579, 1984, pp. 579–604.

[26] B. Jørgensen, “Exponential dispersion models,” Journal of the Royal Statistical Society Series B: Statistical Methodology, vol. 49, no. 2, pp. 127–145, 1987.

[27] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in neural information processing systems, vol. 33, pp. 6840– 6851, 2020.

[28] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole, “Score-based generative modeling through stochastic differential equations,” arXiv preprint arXiv:2011.13456, 2020.

[29] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” arXiv preprint arXiv:2010.02502, 2020.

[30] T. Karras, M. Aittala, T. Aila, and S. Laine, “Elucidating the design space of diffusion-based generative models,” Advances in neural information processing systems, vol. 35, pp. 26 565–26 577, 2022.

[31] Z. Yue, J. Wang, and C. C. Loy, “Efficient diffusion model for image restoration by residual shifting,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 1, pp. 116–130, 2024.

[32] V. De Bortoli, J. Thornton, J. Heng, and A. Doucet, “Diffusion schrödinger bridge with applications to score-based generative modeling,” Advances in neural information processing systems, vol. 34, pp. 17 695– 17 709, 2021.

[33] M. Delbracio and P. Milanfar, “Inversion by direct iteration: An alternative to denoising diffusion for image restoration,” arXiv preprint arXiv:2303.11435, 2023.

[34] Z. Luo, F. K. Gustafsson, Z. Zhao, J. Sjölund, and T. B. Schön, “Image restoration with mean-reverting stochastic differential equations,” in Proceedings of the 40th International Conference on Machine Learning. PMLR, 2023, pp. 23 045–23 066.

[35] H. Wang, J. Zhang, H. Chen, H. Guo, D. Wang, J. Ma, and B. Du, “Residual diffusion bridge model for image restoration,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 8375–8386.

[36] H. Jinhui, Z. Zhu, and J. Hou, “Energy-oriented diffusion bridge for image restoration with foundational diffusion models,” in The Fourteenth International Conference on Learning Representations.

[37] B. Sobieski, M. Tivnan, Y. Wang, P. Jin, D. Wu, Q. Li, P. Biecek et al., “System-embedded diffusion bridge models,” Advances in Neural Information Processing Systems, vol. 38, pp. 17 006–17 043, 2026.

[38] B. Kawar, M. Elad, S. Ermon, and J. Song, “Denoising diffusion restoration models,” Advances in neural information processing systems, vol. 35, pp. 23 593–23 606, 2022.

[39] H. Wang, T. Jin, W. Lin, S. Wang, H. Huang, S. Ji, and Z. Zhao, “IRBridge: Solving image restoration bridge with pre-trained generative diffusion models,” in ICML, ser. Proceedings of Machine Learning Research, vol. 267, 2025, pp. 63 266–63 289.

[40] T. Chen and M. Zhou, “Learning to jump: Thinning and thickening latent counts for generative modeling, 2023,” URL https://arxiv. org/abs/2305.18375.

[41] A. Okhotin, D. Molchanov, A. Vladimir, G. Bartosh, V. Ohanesian, A. Alanov, and D. P. Vetrov, “Star-shaped denoising diffusion probabilistic models,” Advances in Neural Information Processing Systems, vol. 36, pp. 10 038–10 067, 2023.

[42] A. Neufeld, A. Dharamshi, L. L. Gao, and D. Witten, “Data thinning for convolution-closed distributions,” Journal of Machine Learning Research, vol. 25, no. 57, pp. 1–35, 2024.

[43] M. Makitalo and A. Foi, “Optimal inversion of the anscombe transformation in low-count poisson image denoising,” IEEE transactions on Image Processing, vol. 20, no. 1, pp. 99–109, 2010.

[44] C. Chen, Q. Chen, J. Xu, and V. Koltun, “Learning to see in the dark,” in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition. Ieee, 2018, pp. 3291–3300.

[45] K. Wei, Y. Fu, J. Yang, and H. Huang, “A physics-based noise formation model for extreme low-light raw denoising,” in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2020, pp. 2755–2764.

[46] E. Dalsasso, L. Denis, and F. Tupin, “As if by magic: Self-supervised training of deep despeckling networks with merlin,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–13, 2021.

[47] X. Hu, M. Zhu, D. Stankovic, Y. Zhu, Z. Feng, Y. Ban, and L. Stankovi´ c,´ “Closed-form nonlocal shrinkage for multiplicative image denoising and sar despeckling,” arXiv preprint arXiv:2608.15028, 2026.

[48] X. Liu, M. Tanaka, and M. Okutomi, “Single-image noise level estimation for blind denoising,” IEEE transactions on image processing, vol. 22, no. 12, pp. 5226–5237, 2013.

[49] J. Liu, Z. Wang, B. Chen, and H. Han, “Blind2sound: self-supervised image denoising without residual noise,” in 2025 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2025, pp. 12 937–12 946.

[50] A. Banerjee, S. Merugu, I. S. Dhillon, and J. Ghosh, “Clustering with bregman divergences,” Journal of machine learning research, vol. 6, no. Oct, pp. 1705–1749, 2005.

[51] Y.-C. Lin, Y.-S. Xu, H.-W. Chen, H.-K. Kuo, and C.-Y. Lee, “Eamamba: Efficient all-around vision state space model for image restoration,” in 2025 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2025, pp. 11 708–11 719.

[52] J. Byun, S. Cha, and T. Moon, “Fbi-denoiser: Fast blind image denoiser for poisson-gaussian noise,” in 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2021, pp. 5764–5773.

[53] P. Arbelaez, M. Maire, C. Fowlkes, and J. Malik, “Contour detection and hierarchical image segmentation,” IEEE transactions on pattern analysis and machine intelligence, vol. 33, no. 5, pp. 898–916, 2010.

[54] E. Agustsson and R. Timofte, “Ntire 2017 challenge on single image super-resolution: Dataset and study,” in 2017 IEEE conference on computer vision and pattern recognition workshops (CVPRW). IEEE, 2017, pp. 1122–1131.

[55] L. Gomez, R. Ospina, and A. C. Frery, “Unassisted quantitative evaluation of despeckling filters,” Remote Sensing, vol. 9, no. 4, p. 389, 2017.

[56] S. Vitale, D. Cozzolino, G. Scarpa, L. Verdoliva, and G. Poggi, “Guided patchwise nonlocal sar despeckling,” IEEE Transactions on Geoscience and Remote Sensing, vol. 57, no. 9, pp. 6484–6498, 2019.

[57] E. Lukacs, “A characterization of the gamma distribution,” The Annals of Mathematical Statistics, vol. 26, no. 2, pp. 319–324, 1955.

[58] P. K. Dunn and G. K. Smyth, “Series evaluation of tweedie exponential dispersion model densities,” Statistics and Computing, vol. 15, no. 4, pp. 267–280, 2005.

[59] B. Jorgensen, The theory of dispersion models. CRC Press, 1997.

## Supplementary Material

## The Noise Plane: A Diffusion Bridge Across Exponent and Dispersion

## CONTENTS

S1 Measurement model and assumptions .   
S1-B Conditions for the bridge construction   
S1-C Clean limit and intensity domains   
S2 Superposition and the reverse mean . .   
S2-A Closure under measurement accumulation   
S2-B The superposition coupling .   
S2-C Exponent-independent mixing coefficient   
S2-D Measurement-weighted intensity averaging   
S2-E Oracle preservation properties   
S3 Unit deviance and conditional-mean prediction   
S3-A Bregman representation .   
S3-B Mean-optimal reconstruction   
S3-C Tweedie deviance formulas   
S4 Measurement thinning kernels . .   
S4-A Independence from the clean-signal parameter   
S4-B Gaussian, Poisson, and Gamma kernels   
S4-C Latent thinning for intermediate exponents   
S4-D Latent-count inference for real observations   
S4-E Relation to the excess-noise factor .   
S5 Output parameterization and index calibration   
S5-A Relative noise and residual anchoring   
S5-B Effect of radiometric gain   
S5-C Prediction domain . 6   
S6 Exponent range and Gaussian special case . 7   
S6-A Admissible exponents . 7   
S6-B Gaussian marginals and reverse update . 7   
S7 Role of the consistency objective 7   
S8 Evaluation conventions . . 7   
S8-A Number of function evaluations S8-B Evaluation across noise exponents <sup>7</sup><sub>8</sub>   
S9 Additional experimental results . .   
S9-A Additional restoration examples   
S9-B Sensitivity to estimator settings   
S10 Implementation details . . . .   
S10-A Conditioning and training-step sampling   
S10-B The conditional model   
S10-C Noise-severity transfer   
S10-D Noise-exponent estimation   
S10-E Baseline configurations   
S11 SAR data and supplementary evaluation . 8

## S1. MEASUREMENT MODEL AND ASSUMPTIONS

## A. Accumulated signal and intensity

The measurement model is defined pixelwise by a clean mean and a measurement index ν. We derive the moment identities in Equation (2) from the additive density in Equation (1). The natural parameter θ lies in the interior of its domain Θ. For s such that $\theta + s \in$ int Θ, normalization of the density gives

$$
\begin{array} { l } { \displaystyle \mathbb { E } [ e ^ { s Z } ] = \int e ^ { s z } c ^ { * } ( z ; \nu ) e ^ { \theta z - \nu \kappa ( \theta ) } \mathrm { d } m _ { \nu } ( z ) } \\ { \displaystyle = e ^ { \nu [ \kappa ( \theta + s ) - \kappa ( \theta ) ] } \int f _ { Z } ( z ; \theta + s , \nu ) \mathrm { d } m _ { \nu } ( z ) } \\ { \displaystyle = e ^ { \nu [ \kappa ( \theta + s ) - \kappa ( \theta ) ] } . } \end{array}
$$

Here $m _ { \nu }$ denotes the dominating measure. Taking the logarithm yields the cumulant-generating function

$$
K _ { Z } ( s ) = \nu [ \kappa ( \theta + s ) - \kappa ( \theta ) ] .\tag{S1}
$$

Its first two derivatives at zero give the mean and variance in Equation (2). Using the mean map and variance function defined in Equation (3), normalizing the accumulated signal gives the intensity $x = Z / \nu ,$ also written in reproductive form as $x \sim \mathrm { E D } ( \mu , 1 / \nu )$ . Therefore,

$$
{ \begin{array} { r } { \mathbb { E } [ x ] = { \frac { \mathbb { E } [ Z ] } { \nu } } = \mu , } \\ { \operatorname { V a r } ( x ) = { \frac { \operatorname { V a r } ( Z ) } { \nu ^ { 2 } } } = { \frac { V ( \mu ) } { \nu } } . } \end{array} }
$$

Setting $\mu = x _ { \star }$ recovers the conditional measurement moments in Equation (4). The power variance function $V ( \mu ) = \mu ^ { p }$ gives the Tweedie family [25]. We consider $p \in \{ 0 \} \cup [ 1 , 2 ]$ , covering Gaussian, Poisson, compound Poisson–Gamma, and Gamma models. These distributions provide both the required moments and the closure property used to construct the bridge.

Table S1 lists the distributional functions for these models. They follow by integrating $\mathrm { d } \mu / \mathrm { d } \theta = V ( \mu )$ and $\mathrm { d } \kappa / \mathrm { d } \theta = \mu ,$ with conventional choices of integration constants. Here $b =$ $( 1 - p ) \theta > 0$ for $1 < p < 2 .$ The natural parameter ranges over R for $p = 0 , 1$ and is negative for $1 < p \leq 2$

The measurement index incorporates the noise scale. If a physical measurement amount a gives variance $\phi V ( \mu ) / a$ for a fixed dispersion scale $\phi > 0 .$ , setting $\nu = a / \phi$ recovers $V ( \mu ) / \nu$ . Thus, calibration determines the conversion from physical measurement amount to index.

## B. Conditions for the bridge construction

The bridge uses the following assumptions.

(A1) The additive family is infinitely divisible and defined for every index $\nu \in \Lambda = ( 0 , \infty )$

(A2) The mean map $\tau = \kappa ^ { \prime }$ is a bijection from int Θ onto the mean domain Ω. Hence

$$
\theta _ { \star } = \tau ^ { - 1 } ( x _ { \star } )
$$

exists and is unique for each $x _ { \star } \in \Omega$

(A3) The schedule is strictly decreasing, with

$$
0 < \nu _ { \mathrm { o b s } } = \nu ( T - 1 ) < \cdots < \nu ( 0 ) = \nu _ { \mathrm { m a x } } < \infty ,
$$

so every reverse step has a positive increment

$$
\Delta _ { t } = \nu ( t - 1 ) - \nu ( t ) > 0 , \qquad t = 1 , \ldots , T - 1 .
$$

TABLE S1  
DISTRIBUTIONAL FUNCTIONS OF THE SUPPORTED TWEEDIE MODELS.
<table><tr><td>p</td><td>Gaussian 0</td><td>Poisson 1</td><td>Poisson-Gamma (1,2)</td><td>Gamma 2</td></tr><tr><td> $\kappa ( \theta )$ </td><td> $\sigma ^ { 2 } \theta ^ { 2 } / 2$ </td><td> $e ^ { \theta }$ </td><td> $b ^ { ( 2 - p ) / ( 1 - p ) } / ( 2 - p )$ </td><td> $- \log ( - \theta )$ </td></tr><tr><td> $\tau ( \theta )$ </td><td> $\sigma ^ { 2 } \theta$ </td><td> $e ^ { \theta }$ </td><td> $b ^ { 1 / ( 1 - p ) }$ </td><td> $- 1 / \theta$ </td></tr><tr><td> $V ( \mu )$ </td><td> $\sigma ^ { 2 }$ </td><td> $\mu$ </td><td> $\mu ^ { p }$ </td><td> $\mu ^ { 2 }$ </td></tr><tr><td> $\tau ^ { - 1 } ( \mu )$ </td><td> $\mu / \sigma ^ { 2 }$ </td><td> $\log \mu$ </td><td> $\mu ^ { 1 - p } / ( 1 - p )$ </td><td> $- 1 / \mu$ </td></tr></table>

(A4) Conditional on $x _ { \star } ,$ the initial reverse state $Z _ { T - 1 }$ and all added increments are mutually independent and share the natural parameter $\theta _ { \star }$ . Each earlier state is formed by adding the next increment to the accumulated state.

(A5) All distributional identities and moment statements are interpreted pixelwise, with variances referring to individual pixel intensities.

Condition (A1) permits arbitrary positive measurement increments. Without it, a family may allow only a discrete index set. For example, a binomial model parameterized by its number of trials requires integer indices and cannot support an arbitrary real-valued measurement schedule. Condition (A4) supplies the independence needed for superposition. It does not assert independence between the bridge states, which share accumulated measurements.

## C. Clean limit and intensity domains

For a fixed clean intensity with finite $V ( x _ { \star } )$ , unbiasedness gives

$$
\begin{array} { r l } { \mathbb { E } [ ( x _ { \nu } - x _ { \star } ) ^ { 2 } \mid x _ { \star } ] = \mathrm { V a r } ( x _ { \nu } \mid x _ { \star } ) } \\ { = \displaystyle \frac { V ( x _ { \star } ) } { \nu } \longrightarrow 0 \quad } & { ( \nu  \infty ) . } \end{array}
$$

This establishes the conditional mean-square limit stated in the main paper. At finite $\nu _ { \mathrm { m a x } } ,$ , the initial bridge state instead has variance $V ( x _ { \star } ) / \nu _ { \mathrm { m a x } }$ and is not identified with the clean image.

The mean domain, the clean-image range, and the observation support serve different roles. Under (A2), the positive Tweedie branches have $\Omega = ( 0 , \infty )$ , so their clean means are strictly positive but have no model-imposed upper bound. The implementation uses the normalized clean-image range $\mathcal { X } = [ \epsilon , 1 ] , \epsilon > 0$ , and clips predicted clean images to this range. This is an implementation convention, and sampled states are not clipped. In particular, for $1 < p < 2 ,$ , observations can equal zero or exceed the clean-image range.

## S2. SUPERPOSITION AND THE REVERSE MEAN

We prove Theorems 1 and 2 and derive the oracle preservation properties.

## A. Closure under measurement accumulation

Consider independent accumulated signals $Z _ { 1 } \sim \mathrm { E D } ^ { * } ( \theta , \nu _ { 1 } )$ and $Z _ { 2 } \sim \mathrm { E D } ^ { * } ( \theta , \nu _ { 2 } )$ with the same natural parameter. Inde-

pendence makes their cumulant-generating functions additive. Using Equation (S1),

$$
\begin{array} { r l } & { K _ { Z _ { 1 } + Z _ { 2 } } ( s ) = K _ { Z _ { 1 } } ( s ) + K _ { Z _ { 2 } } ( s ) } \\ & { \qquad = \nu _ { 1 } [ \kappa ( \theta + s ) - \kappa ( \theta ) ] } \\ & { \qquad + \nu _ { 2 } [ \kappa ( \theta + s ) - \kappa ( \theta ) ] } \\ & { \qquad = ( \nu _ { 1 } + \nu _ { 2 } ) [ \kappa ( \theta + s ) - \kappa ( \theta ) ] . } \end{array}
$$

The result is the cumulant-generating function of $\mathrm { E D } ^ { * } ( \theta , \nu _ { 1 } +$ $\nu _ { 2 } )$ , proving Equation (5). A shared natural parameter ensures closure. With different parameters, the sum generally does not have this form.

## B. The superposition coupling

To prove Theorem 1, condition on the clean image. By (A1), an increment with index $\Delta _ { t }$ exists; by (A4), it shares $\theta _ { \star }$ with $Z _ { t }$ and is independent of $Z _ { t }$ . Applying Equation (5) and using $\nu ( t ) + \Delta _ { t } = \nu ( t - 1 )$ gives

$$
Z _ { t - 1 } = Z _ { t } + Y _ { t  t - 1 } \sim \mathrm { E D } ^ { * } \bigl ( \theta _ { \star } , \nu ( t - 1 ) \bigr ) ,
$$

This is the marginal at $t - 1$ prescribed by Equation (6).

The intensity-domain form follows by substituting $Z _ { t } ~ =$ $\nu ( t ) x _ { t }$ and dividing through by $\nu ( t - 1 )$

$$
\begin{array} { l } { x _ { t - 1 } = \displaystyle \frac { Z _ { t - 1 } } { \nu ( t - 1 ) } } \\ { \displaystyle \quad = \frac { \nu ( t ) \ : x _ { t } + Y _ { t  t - 1 } } { \nu ( t - 1 ) } } \\ { \displaystyle \quad = \frac { \nu ( t ) } { \nu ( t - 1 ) } \displaystyle x _ { t } + \frac { 1 } { \nu ( t - 1 ) } Y _ { t  t - 1 } } \\ { \displaystyle \quad = \alpha _ { t } x _ { t } + \frac { 1 } { \nu ( t - 1 ) } Y _ { t  t - 1 } , } \end{array}
$$

which is Equation (7).

## C. Exponent-independent mixing coefficient

For Theorem 2, condition on the current state and prediction $( x _ { t } , \hat { x } _ { \star } )$ . The model draws an increment with natural parameter $\hat { \theta } _ { \star } = \tau ^ { - 1 } ( \hat { x } _ { \star } )$ and index $\Delta _ { t }$ . The first moment in Equation (2) gives

$$
\begin{array} { r } { \mathbb { E } \big [ Y _ { t  t - 1 } \mid x _ { t } , \hat { x } _ { \star } \big ] = \Delta _ { t } \kappa ^ { \prime } ( \hat { \theta } _ { \star } ) = \Delta _ { t } \hat { x } _ { \star } , } \end{array}
$$

so taking the conditional expectation of Equation (7) given $( x _ { t } , \hat { x } _ { \star } )$ ,

$$
\begin{array} { l } { \displaystyle \mathbb { E } \big [ x _ { t - 1 } \mid x _ { t } , \hat { x } _ { \star } \big ] = \alpha _ { t } x _ { t } + \frac { \Delta _ { t } } { \nu ( t - 1 ) } \hat { x } _ { \star } } \\ { \displaystyle = \alpha _ { t } x _ { t } + \left( 1 - \frac { \nu ( t ) } { \nu ( t - 1 ) } \right) \hat { x } _ { \star } } \\ { \displaystyle = \alpha _ { t } x _ { t } + ( 1 - \alpha _ { t } ) \hat { x } _ { \star } , } \end{array}
$$

where $\Delta _ { t } / \nu ( t - 1 ) = 1 - \alpha _ { t }$ . The mixing coefficient depends only on the measurement indices, whereas the increment distribution and network prediction can depend on $p .$ The result gives the conditional mean of the specified plug-in transition.

## D. Measurement-weighted intensity averaging

The normalized form of Equation (5) gives the measurementweighted averaging rule. If $x _ { i } \sim \mathrm { E D } ( \mu , 1 / w _ { i } )$ are independent with positive measurement weights $w _ { i } .$ put $Z _ { i } : = w _ { i } x _ { i }$ , so that $\bar { Z } _ { i } \sim \mathrm { E D } ^ { * } ( \theta , w _ { i } )$ with the common $\theta = \tau ^ { - 1 } ( \mu )$ . Then Equation (5) gives

$$
\sum _ { i } Z _ { i } \sim \mathrm { E D } ^ { * } \Big ( \theta , \sum _ { i } w _ { i } \Big ) ,
$$

and dividing by $\sum _ { i } w _ { i }$ converts back to the reproductive form:

$$
\frac { \sum _ { i } w _ { i } x _ { i } } { \sum _ { i } w _ { i } } \sim \mathrm { E D } \left( \mu , \frac { 1 } { \sum _ { i } w _ { i } } \right) .\tag{S2}
$$

Thus the weighted average remains in the same family, with mean $\mu$ and measurement index $\sum _ { i } w _ { i }$ . For measurements of a fixed scene, take $\mu = x _ { \star }$ and interpret independence conditionally on that scene. Exposure, dose, or number of looks can serve as the index when the corresponding sensor model and calibration satisfy these assumptions.

## E. Oracle preservation properties

We now establish the preservation properties under oracle predictions $\hat { x } _ { \star } = x _ { \star }$

First, the stochastic update preserves the prescribed bridge marginals by Theorem 1. The deterministic update preserves their mean. Taking the expectation of Equation (8) over $x _ { t } \sim$ $\mathrm { E D } ( x _ { \star } , 1 / \nu ( t ) )$ gives

$$
\begin{array} { r l } & { \mathbb { E } [ x _ { t - 1 } ] = \alpha _ { t } \mathbb { E } [ x _ { t } ] + \left( 1 - \alpha _ { t } \right) x _ { \star } } \\ & { \quad \quad \quad = \alpha _ { t } x _ { \star } + ( 1 - \alpha _ { t } ) x _ { \star } = x _ { \star } . } \end{array}
$$

Induction extends this mean-preservation result along the schedule. Replacing the random increment by its mean reduces the variance, so the deterministic update does not preserve the bridge distribution. For a bridge-distributed input,

$$
\operatorname { V a r } ( x _ { t - 1 } ^ { \mathrm { d e t } } \mid x _ { \star } ) = \alpha _ { t } ^ { 2 } { \frac { V ( x _ { \star } ) } { \nu ( t ) } } = \alpha _ { t } { \frac { V ( x _ { \star } ) } { \nu ( t - 1 ) } } ,
$$

which is smaller than the target variance $V ( x _ { \star } ) / \nu ( t - 1 )$

Second, the stochastic sampler reproduces the joint trajectory of the superposition coupling. Starting from $Z _ { T - 1 } .$ , each earlier state is generated by an increment independent of all states at times $\geq t .$ . Therefore,

$$
q ( z _ { t - 1 } \mid z _ { t : T - 1 } , x _ { \star } ) = q ( z _ { t - 1 } \mid z _ { t } , x _ { \star } ) ,
$$

so the chain rule collapses to

$$
q ( z _ { T - 1 } , \dots , z _ { 0 } \mid x _ { \star } ) = q ( z _ { T - 1 } \mid x _ { \star } ) \prod _ { t = 1 } ^ { T - 1 } q ( z _ { t - 1 } \mid z _ { t } , x _ { \star } ) ,\tag{S3}
$$

Thus, the coupling is Markov in the reverse-time index, with the stochastic transition in Theorem 1. Initialization from $Z _ { T - 1 } \sim \mathrm { E D } ^ { * } ( \theta _ { \star } , \nu _ { \mathrm { o b s } } )$ recovers the full joint law. For a fixed observation $Z _ { T - 1 } = z _ { \mathrm { o b s } } ,$ , the same transitions instead generate the trajectory conditional on that observation.

These properties concern the superposition coupling. The marginals in Equation (6) alone do not determine a unique joint law across time.

## S3. UNIT DEVIANCE AND CONDITIONAL-MEAN PREDICTION

The Bregman representation in Equation (9) provides a shared reconstruction objective across noise models. Let $\psi : =$ $\kappa ^ { * }$ . Convex conjugacy gives

$$
\psi ( \mu ) = \mu \tau ^ { - 1 } ( \mu ) - \kappa \big ( \tau ^ { - 1 } ( \mu ) \big ) , \quad \quad \psi ^ { \prime } ( \mu ) = \tau ^ { - 1 } ( \mu ) = \theta .\tag{S4}
$$

## A. Bregman representation

Write $\tilde { \theta } = \tau ^ { - 1 } ( y )$ and $\theta = \tau ^ { - 1 } ( \mu )$ , and expand the Bregman divergence generated by ψ:

$$
\begin{array} { l } { { B _ { \psi } ( y , \mu ) = \psi ( y ) - \psi ( \mu ) - \psi ^ { \prime } ( \mu ) \left( y - \mu \right) } } \\ { { \ \qquad = \left[ y \tilde { \theta } - \kappa ( \tilde { \theta } ) \right] - \left[ \mu \theta - \kappa ( \theta ) \right] - \theta \left( y - \mu \right) } } \\ { { \ \qquad = y \tilde { \theta } - \kappa ( \tilde { \theta } ) - \mu \theta + \kappa ( \theta ) - \theta y + \theta \mu } } \\ { { \ \qquad = y ( \tilde { \theta } - \theta ) - \kappa ( \tilde { \theta } ) + \kappa ( \theta ) } } \\ { { \ \qquad = { \frac { 1 } { 2 } } d ( y , \mu ) , } } \end{array}
$$

The second line substitutes Equation (S4). Cancelling the $\mu \theta$ terms then gives half the unit deviance.

## B. Mean-optimal reconstruction

We first derive the pointwise gradient, then establish Theorem 3 by an expectation decomposition. Using $\psi ^ { \prime \prime } ( \mu ) =$ $1 / V ( \mu )$

$$
\begin{array} { l } { { \partial _ { \mu } B _ { \psi } ( y , \mu ) = \partial _ { \mu } \bigl [ \psi ( y ) - \psi ( \mu ) - \psi ^ { \prime } ( \mu ) ( y - \mu ) \bigr ] } } \\ { { \qquad = - \psi ^ { \prime } ( \mu ) - \psi ^ { \prime \prime } ( \mu ) ( y - \mu ) + \psi ^ { \prime } ( \mu ) } } \\ { { \qquad = - \psi ^ { \prime \prime } ( \mu ) \left( y - \mu \right) } } \\ { { \qquad = - \displaystyle \frac { y - \mu } { V ( \mu ) } , } } \end{array}
$$

The two $\psi ^ { \prime } ( \mu )$ terms cancel by the product rule. When differentiation and expectation can be interchanged,

$$
\frac { d } { d \mu } \mathbb { E } \big [ d ( Y , \mu ) \big ] = - \frac { 2 } { { V } ( \mu ) } \big ( \mathbb { E } [ Y ] - \mu \big ) ,\tag{S5}
$$

Since $V ( \mu ) > 0$ on the mean domain, the only stationary point is $\mu = \mathbb { E } [ Y ]$ To establish optimality without interchanging differentiation and expectation, let $m : = \mathbb { E } [ Y ]$ and expand Equation (9) about m:

$$
{ \mathbb E } \big [ d ( Y , \mu ) \big ] = { \mathbb E } \big [ d ( Y , m ) \big ] + d ( m , \mu ) , \qquad \mu \in \Omega ,\tag{S6}
$$

The cross term has zero expectation because it is linear in $Y - m . \mathrm { I f } \ m \in \Omega$ and $\mathbb { E } [ d ( \bar { Y } , m ) ] < \infty$ , strict convexity of $\psi$ makes $d ( m , \mu )$ uniquely minimal at $\mu = m$ . These conditions establish Theorem 3. In our experiments, $x _ { \star } \in \mathcal { X } = [ \epsilon , 1 ]$ so the conditional mean remains in the mean domain and the expected deviance is finite. Applying the decomposition conditionally gives

$$
\arg \operatorname* { m i n } _ { \hat { x } _ { \star } } \mathbb { E } \big [ d ( x _ { \star } , \hat { x } _ { \star } ) \mid x _ { t } , x _ { \mathrm { o b s } } \big ] = \mathbb { E } \big [ x _ { \star } \mid x _ { t } , x _ { \mathrm { o b s } } \big ] ,
$$

Thus, the reconstruction objective predicts the conditional mean used by the reverse update in Equation (8).

## C. Tweedie deviance formulas

Substituting $V ( \mu ) = \mu ^ { p }$ yields the gradient in Equation (10):

$$
\partial _ { \mu } d _ { p } ( y , \mu ) = - \frac { 2 ( y - \mu ) } { \mu ^ { p } } = - 2 y \mu ^ { - p } + 2 \mu ^ { 1 - p } ,
$$

and integrating in $\mu ,$ with the constant of integration fixed by $d _ { p } ( y , y ) = 0$ , gives

$$
d _ { p } ( y , \mu ) = 2 \left[ \frac { y ^ { 2 - p } } { ( 1 - p ) ( 2 - p ) } - \frac { y \mu ^ { 1 - p } } { 1 - p } + \frac { \mu ^ { 2 - p } } { 2 - p } \right]\tag{S7}
$$

for $p \notin \{ 0 , 1 , 2 \}$ , together with

$$
\begin{array} { r } { d _ { p } ( y , \mu ) = \left\{ \begin{array} { l l } { ( y - \mu ) ^ { 2 } , } & { p = 0 , } \\ { 2 \big [ y \log ( y / \mu ) - ( y - \mu ) \big ] , } & { p = 1 , } \\ { 2 \big [ y / \mu - \log ( y / \mu ) - 1 \big ] , } & { p = 2 , } \end{array} \right. } \end{array}\tag{S8}
$$

The Gaussian case follows by setting $p = 0$ in Equation (S7). The Poisson and Gamma cases follow from the limits $p  1$ and $p  2$ , where the power terms yield logarithms.

## S4. MEASUREMENT THINNING KERNELS

The forward transition reduces the measurement amount through the thinning kernel $q ( z _ { t } \mid z _ { t - 1 } ; \theta _ { \star } )$ in Equation (11). We derive this kernel, give its Gaussian, Poisson, and Gamma forms, and establish the latent construction in Theorem 4. We then relate the compound model to the excess-noise factor in Equation (13).

## A. Independence from the clean-signal parameter

For $p = 1$ , the density in Equation (1) is a mass function with respect to counting measure. For $1 < p < 2 $ , use the parameterindependent dominating measure $\delta _ { 0 } + \mathrm { L e b } _ { ( 0 , \infty ) }$ . Under the canonical compound parameterization, $c ^ { * } ( 0 ; \dot { \nu } ) = 1$ , while the probability mass at zero is $\mathrm { P r } ( Z _ { \nu } = 0 ) \stackrel { \cdot } { = } e ^ { - { \not \nu } \lambda }$ . The latter is part of the probability law, not part of the dominating measure.

Conditioning on a positive total z gives a mixed kernel on [0, z] with a Lebesgue density on $( 0 , z )$ and atoms at both 0 and z. Equation (11) gives its density with respect to $\delta _ { 0 } +$ $\mathrm { L e b } _ { ( 0 , z ) } + \delta _ { z }$ . In particular,

$$
\begin{array} { r l } & { \operatorname* { P r } ( Z _ { t } = 0 \mid Z _ { t - 1 } = z ) = \frac { c ^ { * } ( z ; \Delta _ { t } ) } { c ^ { * } ( z ; \nu ( t - 1 ) ) } , } \\ & { \operatorname* { P r } ( Z _ { t } = z \mid Z _ { t - 1 } = z ) = \frac { c ^ { * } ( z ; \nu ( t ) ) } { c ^ { * } ( z ; \nu ( t - 1 ) ) } . } \end{array}
$$

For $z = 0 ,$ , the conditional law is a point mass at zero. The latent construction below realizes all these components without evaluating the compound density.

By (A4) the pair $( Z _ { t } , Z _ { t - 1 } - Z _ { t } )$ consists of independent variables at the same $\theta _ { \star }$ , with indices $\nu ( t )$ and $\Delta _ { t }$ . Its joint density is therefore the product of two copies of Equation (1),

$$
c ^ { * } \big ( z _ { t } ; \nu ( t ) \big ) c ^ { * } \big ( z _ { t - 1 } - z _ { t } ; \Delta _ { t } \big ) e ^ { \theta _ { \star } z _ { t - 1 } - \nu ( t - 1 ) \kappa ( \theta _ { \star } ) } ,
$$

where $z _ { t } + ( z _ { t - 1 } - z _ { t } ) = z _ { t - 1 }$ and $\nu ( t ) + \Delta _ { t } = \nu ( t - 1 )$ . Dividing by the marginal density of $Z _ { t - 1 }$ cancels the exponential factor, giving

$$
q ( z _ { t } \mid z _ { t - 1 } ; \theta _ { \star } ) = \frac { c ^ { * } \big ( z _ { t } ; \nu ( t ) \big ) c ^ { * } \big ( z _ { t - 1 } - z _ { t } ; \Delta _ { t } \big ) } { c ^ { * } \big ( z _ { t - 1 } ; \nu ( t - 1 ) \big ) } ,
$$

This recovers Equation (11). Conditioning on the sufficient statistic $Z _ { t - 1 }$ removes $\theta _ { \star }$ for every supported exponent, including $1 < p < 2$

## B. Gaussian, Poisson, and Gamma kernels

For $p \in \{ 0 , 1 , 2 \}$ , the normalizing factor $c ^ { * }$ has an explicit form that gives a standard thinning distribution. We identify each kernel by substituting $c ^ { * }$ into Equation (11). Throughout, $\alpha _ { t } = \nu ( t ) / \nu ( t - 1 )$ and $\Delta _ { t } / \nu ( t - 1 ) = 1 - \alpha _ { t }$ . The results are summarized in Table I.

Gaussian $( p \ = \ 0 )$ . Here $c ^ { * } ( z ; \nu ) \ \propto \ e ^ { - z ^ { 2 } / 2 \nu \sigma ^ { 2 } }$ , so the numerator is proportional to

$$
\exp \left\{ - \frac { z _ { t } ^ { 2 } } { 2 \nu ( t ) \sigma ^ { 2 } } - \frac { ( z _ { t - 1 } - z _ { t } ) ^ { 2 } } { 2 \Delta _ { t } \sigma ^ { 2 } } \right\} ,
$$

a Gaussian in $z _ { t } .$ Collecting the quadratic and linear coefficients,

$$
\mathrm { p r e c i s i o n } = \frac { 1 } { \sigma ^ { 2 } } \left[ \frac { 1 } { \nu ( t ) } + \frac { 1 } { \Delta _ { t } } \right] = \frac { \nu ( t - 1 ) } { \sigma ^ { 2 } \nu ( t ) \Delta _ { t } } ,
$$

z<sub>t−1</sub> linear coefficient = ∆<sub>t</sub>σ<sup>2</sup>

The mean is the ratio of the two and the variance the reciprocal of the precision:

$$
\begin{array} { r l } & { \mathrm { m e a n } = \displaystyle \frac { z _ { t - 1 } } { \Delta _ { t } \sigma ^ { 2 } } \cdot \frac { \sigma ^ { 2 } \nu ( t ) \Delta _ { t } } { \nu ( t - 1 ) } = \displaystyle \frac { \nu ( t ) } { \nu ( t - 1 ) } z _ { t - 1 } = \alpha _ { t } z _ { t - 1 } , } \\ & { \mathrm { v a r i a n c e } = \displaystyle \frac { \sigma ^ { 2 } \nu ( t ) \Delta _ { t } } { \nu ( t - 1 ) } = \sigma ^ { 2 } \alpha _ { t } \Delta _ { t } . } \end{array}
$$

Thus, $z _ { t } \mid z _ { t - 1 } \sim \mathcal { N } ( \alpha _ { t } z _ { t - 1 } , \sigma ^ { 2 } \alpha _ { t } \Delta _ { t } )$ , with a mean proportional to the observed total and a variance determined by the schedule.

Poisson $( p = 1 )$ . Here $\kappa ( \theta ) = e ^ { \theta }$ and $Z \sim \operatorname { P o i s } ( \nu e ^ { \theta } )$ , so

$$
c ^ { * } ( z ; \nu ) = \frac { \nu ^ { z } } { z ! } , \qquad z \in \mathbb { N } .
$$

Substituting,

$$
\begin{array} { c } { { q ( z _ { t } \mid z _ { t - 1 } ) = \frac { \nu ( t ) ^ { z _ { t } } } { z _ { t } ! } \cdot \frac { \Delta _ { t } ^ { z _ { t - 1 } - z _ { t } } } { ( z _ { t - 1 } - z _ { t } ) ! } \cdot \frac { z _ { t - 1 } ! } { \nu ( t - 1 ) ^ { z _ { t - 1 } } } } } \\ { { = \left( { z _ { t - 1 } } \right) \left( \frac { \nu ( t ) } { \nu ( t - 1 ) } \right) ^ { z _ { t } } \left( \frac { \Delta _ { t } } { \nu ( t - 1 ) } \right) ^ { z _ { t - 1 } - z _ { t } } } } \\ { { = \left( { z _ { t } } \right) \alpha _ { t } ^ { z _ { t } } \left( 1 - \alpha _ { t } \right) ^ { z _ { t - 1 } - z _ { t } } , } } \end{array}
$$

The resulting kernel is Binomia $\displaystyle ( z _ { t - 1 } , \alpha _ { t } )$ , which retains each count independently with probability $\alpha _ { t }$

Gamma $\left( p = 2 \right)$ . Here $\kappa ( \theta ) = - \log ( - \theta )$ and $Z \sim \Gamma ( \nu , - \theta )$ so

$$
c ^ { * } ( z ; \nu ) = \frac { z ^ { \nu - 1 } } { \Gamma ( \nu ) } .
$$

Substituting,

$$
\begin{array} { l } { \displaystyle q ( z _ { t } \mid z _ { t - 1 } ) = \frac { \Gamma \big ( \nu ( t - 1 ) \big ) } { \Gamma \big ( \nu ( t ) \big ) \Gamma ( \Delta _ { t } ) } } \\ { \displaystyle \qquad \times \ z _ { t } ^ { \nu ( t ) - 1 } ( z _ { t - 1 } - z _ { t } ) ^ { \Delta _ { t } - 1 } z _ { t - 1 } ^ { 1 - \nu ( t - 1 ) } . } \end{array}
$$

Change variable to $u = z _ { t } / z _ { t - 1 }$ . The powers of $z _ { t - 1 }$ collect as

$$
z _ { t - 1 } ^ { \nu ( t ) - 1 } \cdot z _ { t - 1 } ^ { \Delta _ { t } - 1 } \cdot z _ { t - 1 } ^ { 1 - \nu ( t - 1 ) } \cdot z _ { t - 1 } \ = \ z _ { t - 1 } ^ { \nu ( t ) + \Delta _ { t } - \nu ( t - 1 ) } \ = \ z _ { t - 1 } ^ { 0 } ,
$$

The final factor is the Jacobian $d z _ { t } = z _ { t - 1 }$ du. All dependence on $z _ { t - 1 }$ cancels, leaving the Beta density

$$
f _ { U } ( u ) = \frac { u ^ { \nu ( t ) - 1 } ( 1 - u ) ^ { \Delta _ { t } - 1 } } { B \big ( \nu ( t ) , \Delta _ { t } \big ) } , \qquad 0 < u < 1 .
$$

Hence $\begin{array} { r l r } { U } & { { } \sim } & { \mathrm { B e t a } ( \nu ( t ) , \Delta _ { t } ) } \end{array}$ independently of the total $Z _ { t - 1 } \ [ 5 7 ]$ , and thinning scales that total by $\dot { U } .$

## C. Latent thinning for intermediate exponents

For $1 < p < 2 $ , Theorem 4 combines binomial thinning of the latent count with Beta thinning of the accumulated amplitude.

The jump count satisfies $N _ { \nu } \sim \mathrm { P o i s } ( \lambda \nu )$ . Applying the Poisson thinning kernel gives

$$
N _ { t } \mid N _ { t - 1 } = n \sim \mathrm { B i n o m i a l } ( n , \alpha _ { t } ) , \qquad \alpha _ { t } = { \frac { \nu ( t ) } { \nu ( t - 1 ) } } .
$$

Because jump amplitudes are independent of the count process, their distribution is unchanged by count thinning.

For the amplitude layer, condition on $N _ { t } = n _ { 1 }$ and $N _ { t - 1 } =$ $n .$ . The total at t is a sum of $n _ { 1 }$ independent $\Gamma ( \alpha , \beta )$ jumps and the discarded part is the sum of the remaining $n - n _ { 1 }$ , so by Gamma additivity at the common scale $\beta ,$

$$
Z _ { t } \sim \Gamma ( n _ { 1 } \alpha , \beta ) , \qquad Z _ { t - 1 } - Z _ { t } \sim \Gamma \bigl ( ( n - n _ { 1 } ) \alpha , \beta \bigr ) ,
$$

These totals are independent and share a scale. For $0 < n _ { 1 } < n ,$ the Beta–Gamma identity [57] gives

$$
\frac { Z _ { t } } { Z _ { t } + ( Z _ { t - 1 } - Z _ { t } ) } = \frac { Z _ { t } } { Z _ { t - 1 } } \sim \mathrm { B e t a } \big ( n _ { 1 } \alpha , ( n - n _ { 1 } ) \alpha \big ) ,
$$

independent of $Z _ { t - 1 }$ given the counts. The scale $\beta$ cancels in both layers, so neither depends on it.

For the boundary counts, $n _ { 1 } = 0$ gives $Z _ { t } = 0 ,$ , and $n _ { 1 } = n$ gives $Z _ { t } = Z _ { t - 1 }$ . When $n = 0 ,$ , both totals are zero and no ratio is formed. These cases recover the atoms of the mixed thinning kernel without evaluating a Beta distribution with a zero shape parameter.

The shape $\alpha = ( 2 - p ) / ( p - 1 )$ is fixed by the Tweedie parameterization with variance function $\mu ^ { p }$ . It equals the negative of the transformed exponent in Equation (S11) for $1 < p < 2 .$

## D. Latent-count inference for real observations

Both thinning layers are exact conditional on N. Under synthetic degradation the simulator can retain the generated count alongside the state, avoiding count inference. A real observation supplies only the total ${ \bar { Z } } ,$ and the count has to be sampled from its posterior. Writing the compound density as a sum over n and cancelling every factor that does not depend on n, the weights are, at unit dispersion and for $z > 0 ,$

$$
\begin{array} { r } { \operatorname* { P r } [ N _ { \nu } = n \mid Z _ { \nu } = z ] \propto \frac { A ^ { n } } { n ! \Gamma ( n \alpha ) } , \ ~ } \\ { A = \frac { \nu z ^ { \alpha } } { ( 2 - p ) ( p - 1 ) ^ { \alpha } } , \ ~ } \end{array}\tag{S9}
$$

for $n \geq 1$ , with $z = 0$ forcing $n = 0$ . The mean drops out of the n-dependence: the two places $\mu$ enters, the Poisson rate $\lambda \propto \mu ^ { 2 - p }$ and the jump scale $\beta \propto \mu ^ { p - 1 }$ , cancel because

$\alpha ( p - 1 ) = 2 - p$ . Equation (S9) is the Tweedie series [58] read as a distribution over n rather than as a density in z.

The implementation uses $n _ { \mathrm { m a x } } ~ = ~ 6 4$ , normalizes these weights over $1 \leq n \leq n _ { \operatorname* { m a x } } ,$ and samples by inverse cumulative probability. For fixed $( p , \nu , z )$ , define the omitted posterior mass

$$
\epsilon _ { \mathrm { t a i l } } ( z ) = \frac { \sum _ { n > n _ { \mathrm { m a x } } } A ^ { n } / [ n ! \Gamma ( n \alpha ) ] } { \sum _ { n \geq 1 } A ^ { n } / [ n ! \Gamma ( n \alpha ) ] } .
$$

The omitted posterior mass $\epsilon _ { \mathrm { t a i l } } ( z )$ equals the total variation error of the truncated count distribution.

Interior-exponent thinning of real observations requires this posterior step, whereas Gamma thinning uses the exact Beta kernel. Our synthetic severity-transfer protocol instead uses simulator-retained counts to thin the conditioning channel. The reported transfer results therefore apply to retained-count thinning. Transfer using inferred counts remains to be evaluated.

## E. Relation to the excess-noise factor

To derive Equation (13), let the gain of each primary event follow $G \sim \Gamma ( k , \beta )$ in shape–scale form. Its excess-noise factor is

$$
F ^ { 2 } : = { \frac { \mathbb { E } [ G ^ { 2 } ] } { \mathbb { E } [ G ] ^ { 2 } } } .
$$

For a Gamma gain, $\mathbb { E } [ G ] = k \beta$ and $\mathrm { V a r } ( G ) = k \beta ^ { 2 }$ , so

$$
F ^ { 2 } = 1 + { \frac { \mathrm { V a r } ( G ) } { \mathbb { E } [ G ] ^ { 2 } } } = 1 + { \frac { k \beta ^ { 2 } } { k ^ { 2 } \beta ^ { 2 } } } = 1 + { \frac { 1 } { k } } ,\tag{S10}
$$

This factor is independent of $\beta .$ Identifying the gain shape with the jump shape in Theorem 4, $\alpha = k = ( 2 - p ) / ( p - 1 )$ , gives

$$
{ \frac { 2 - p } { p - 1 } } = k \quad \Longrightarrow \quad 2 - p = k ( p - 1 ) \quad \Longrightarrow \quad p = { \frac { 2 + k } { 1 + k } } .
$$

Inverting Equation (S10) as $k = 1 / ( F ^ { 2 } - 1 )$ and substituting,

$$
p = { \frac { 2 + { \cfrac { 1 } { F ^ { 2 } - 1 } } } { 1 + { \cfrac { 1 } { F ^ { 2 } - 1 } } } } = { \frac { 2 ( F ^ { 2 } - 1 ) + 1 } { ( F ^ { 2 } - 1 ) + 1 } } = { \frac { 2 F ^ { 2 } - 1 } { F ^ { 2 } } } = 2 - { \frac { 1 } { F ^ { 2 } } } ,
$$

which is Equation (13). The map is strictly increasing in $F ^ { 2 }$ with

$$
p \to 1 ^ { + } \mathrm { a s } F ^ { 2 } \to 1 ^ { + } , \qquad p \to 2 ^ { - } \mathrm { a s } F ^ { 2 } \to \infty ,
$$

hence a bijection onto (1, 2). Holding the mean and measurement index fixed while varying the jump rate and scale according to the Tweedie parameterization recovers the Poisson and Gamma marginals at the endpoints.

This relation applies within the Tweedie compound parameterization. With dispersion $\phi ,$ its parameters are

$$
\lambda = \frac { \mu ^ { 2 - p } } { \phi ( 2 - p ) } , \qquad \alpha = \frac { 2 - p } { p - 1 } , \qquad \beta = \phi ( p - 1 ) \mu ^ { p - 1 } ,
$$

which reduces to Equation (12) at $\phi = 1$ . In particular, the jump scale depends on the mean. For comparison, hold the

gain parameters α and $\beta$ fixed and vary only the event rate. With $N \sim \mathrm { P o i s } ( \lambda ) , X _ { i } \overset { i i d } { \sim } \Gamma ( \alpha , \beta )$ , and $\begin{array} { r } { Z = \sum _ { i < N } X _ { i } } \end{array}$ ,

$$
\begin{array} { r l } & { \mathbb { E } [ Z ] = \lambda \alpha \beta , } \\ & { \mathrm { V a r } ( Z ) = \lambda \mathbb { E } [ X _ { 1 } ^ { 2 } ] = \lambda \alpha ( \alpha + 1 ) \beta ^ { 2 } , } \\ & { \frac { \mathrm { V a r } ( Z ) } { \mathbb { E } [ Z ] } = ( \alpha + 1 ) \beta , } \end{array}
$$

The variance-to-mean ratio is independent of $\lambda ,$ giving a linear variance law with dispersion $( \alpha + 1 ) \beta$ . The distribution remains compound Poisson, rather than the scaled Poisson law of a Tweedie $p = 1$ model. Thus, Equation (13) relates gain shape to the exponent under the full Tweedie construction. Assigning an exponent to a sensor also requires variance–intensity calibration, as in the FMD and SAR experiments.

## S5. OUTPUT PARAMETERIZATION AND INDEX CALIBRATION

The variance law in Equation (4) explains the exponent dependence of relative noise and the sensitivity of the measurement index to intensity scaling.

## A. Relative noise and residual anchoring

By Equation (4), $\mathbb { E } [ x _ { t } ] = x _ { \star }$ and $\mathrm { V a r } ( x _ { t } ) = V ( x _ { \star } ) / \nu ( t )$ so with $V ( \mu ) = \mu ^ { p }$

$$
\begin{array} { r l } & { \operatorname { C V } ( x _ { t } \mid x _ { \star } ) = \frac { \sqrt { \operatorname { V a r } ( x _ { t } ) } } { \mathbb { E } \left[ x _ { t } \right] } = \frac { \sqrt { V ( x _ { \star } ) / \nu ( t ) } } { x _ { \star } } } \\ & { \quad \quad = \frac { x _ { \star } ^ { p / 2 } } { x _ { \star } \sqrt { \nu ( t ) } } = \frac { x _ { \star } ^ { p / 2 - 1 } } { \sqrt { \nu ( t ) } } , } \end{array}
$$

This recovers Equation (14). At a fixed measurement index, relative noise is signal-independent for $p = 2$ and increases towards low intensities for $p < 2$ . This behavior motivates log-intensity prediction for $1 \le p < 2$ and residual anchoring at $p = 2 .$

## B. Effect of radiometric gain

On a homogeneous patch write $\hat { \nu } : = V ( \bar { x } ) / \hat { \sigma } ^ { 2 }$ with x¯ the sample mean and $\hat { \sigma } ^ { 2 }$ the sample variance. Under a radiometric rescaling $x \mapsto$ cx the two sample moments transform as

$$
\bar { x } \mapsto c \bar { x } , \qquad \hat { \sigma } ^ { 2 } \mapsto c ^ { 2 } \hat { \sigma } ^ { 2 } ,
$$

so with $V ( \mu ) = \mu ^ { p } ;$

$$
\hat { \nu } = \frac { V ( \bar { x } ) } { \hat { \sigma } ^ { 2 } } \longmapsto \frac { ( c \bar { x } ) ^ { p } } { c ^ { 2 } \hat { \sigma } ^ { 2 } } = c ^ { p - 2 } \frac { \bar { x } ^ { p } } { \hat { \sigma } ^ { 2 } } = c ^ { p - 2 } \hat { \nu } ,
$$

This gives Equation (15). Gamma index estimation is gaininvariant because $p - 2 = 0 { \mathrm { ~ a t ~ } } p = 2$ . For other exponents, the estimated index depends on the intensity units, so calibration requires a fixed normalization convention.

## C. Prediction domain

The supported $p \geq 1$ models have positive mean domains and nonnegative measurement support. We enforce positive predictions through the output parameterization. ${ \bf A t } \ \boldsymbol { p } = 0 ;$ the Gaussian mean domain is R, and we use an identity map. Projection onto the bounded working range $\mathcal { X }$ is a separate implementation choice.

## S6. EXPONENT RANGE AND GAUSSIAN SPECIAL CASE

The main text studies $\{ 0 \} \cup \{ 1 , 2 \}$ . Here we summarize the exponent domain of the Tweedie family and derive the Gaussian special case.

## A. Admissible exponents

Tweedie exponential dispersion models with variance function $V ( \mu ) ~ = ~ \mu ^ { p }$ exist for $p \notin \mathsf { \Gamma } ( 0 , 1 )$ [26], [58], [59]. A commonly used exponent transformation is

$$
\alpha _ { \mathrm { { s } } } = { \frac { p - 2 } { p - 1 } } ,\tag{S11}
$$

for $p \neq 1$ . In the compound Poisson–Gamma range, $- \alpha _ { \mathrm { { s } } } =$ $( 2 - p ) / ( p - 1 )$ gives the positive jump shape rather than an ordinary stable-law index. Positive-support models also exist for $p > 2 .$ , including the inverse Gaussian model at $p = 3 ,$ for which bridge simulation methods are available. These models lie outside the scope of the present framework.

## B. Gaussian marginals and reverse update

At $p = 0$ the variance function is the constant $\sigma ^ { 2 }$ , so

$$
\begin{array} { r } { \kappa ( \theta ) = \frac 1 2 \sigma ^ { 2 } \theta ^ { 2 } , \qquad \mathrm { E D } ^ { * } ( \theta , \nu ) = \mathcal N \big ( \nu \sigma ^ { 2 } \theta , ~ \nu \sigma ^ { 2 } \big ) . } \end{array}
$$

Since $\tau ( \theta ) = \kappa ^ { \prime } ( \theta ) = \sigma ^ { 2 } \theta$ , the bridge’s natural parameter is

$$
\theta _ { \star } = \tau ^ { - 1 } ( x _ { \star } ) = \frac { x _ { \star } } { \sigma ^ { 2 } } ,
$$

so that, writing $\sigma _ { t } ^ { 2 } : = \sigma ^ { 2 } / \nu ( t )$

$$
Z _ { t } \sim { \mathcal { N } } { \big ( } \nu ( t ) x _ { \star } , \nu ( t ) \sigma ^ { 2 } { \big ) } \quad \implies \quad x _ { t } = { \frac { Z _ { t } } { \nu ( t ) } } \sim { \mathcal { N } } { \big ( } x _ { \star } , \sigma _ { t } ^ { 2 } { \big ) } ,
$$

This gives the Gaussian reparameterization $x _ { t } = x _ { \star } + \sigma _ { t } \xi$ in Equation (16), with $\xi \sim \mathcal { N } ( 0 , I )$ . The conditional variance is independent of the clean intensity, and the shared reverse-mean update in Equation (8) applies directly.

Although the marginal parameterization is Gaussian, the deterministic update differs from a probability-flow ODE step. Since $\alpha _ { t } = \nu ( \dot { t } ) / \nu ( t - 1 ) = \sigma _ { t - 1 } ^ { 2 } / \dot { \sigma } _ { t } ^ { 2 }$ , Equation (8) gives

$$
x _ { t - 1 } = \hat { x } _ { \star } + \frac { \sigma _ { t - 1 } ^ { 2 } } { \sigma _ { t } ^ { 2 } } \left( x _ { t } - \hat { x } _ { \star } \right) ,
$$

whereas an Euler step on the EDM probability-flow ODE in the same σ parameterisation [30] gives

$$
x _ { t - 1 } = \hat { x } _ { \star } + \frac { \sigma _ { t - 1 } } { \sigma _ { t } } ( x _ { t } - \hat { x } _ { \star } ) ,
$$

The coefficients are therefore the variance ratio and standarddeviation ratio, respectively. For example, a step from $\sigma = 2$ to $\sigma = 1$ retains one quarter of the displacement under our update and one half under the Euler update. Our coefficient follows from the conditional mean of the superposition coupling in Theorem 2.

TABLE S2  
EFFECT OF THE NUMBER OF FUNCTION EVALUATIONS AT SEVERE NOISE. “DIAGONAL” DENOTES MATCHED MODEL AND DATA EXPONENTS. “WORST” IS THE MINIMUM OVER THE TESTED DATA EXPONENTS. BOLD MARKS THE PREFERRED VALUE IN EACH ROW.
<table><tr><td>severe</td><td>NFE=1</td><td>NFE=5</td><td>NFE=25</td></tr><tr><td>diagonal mean PSNR</td><td>22.326</td><td>22.330</td><td>22.225</td></tr><tr><td>p=1.5 worst over pdata</td><td>20.739</td><td>21.339</td><td>21.594</td></tr><tr><td>mean off-diagonal penalty</td><td>+1.601</td><td>+1.256</td><td>+0.924</td></tr><tr><td>diagonal dark log-RMSE</td><td>0.2841</td><td>0.2831</td><td>0.2854</td></tr><tr><td>diagonal res. var.</td><td>0.926</td><td>0.949</td><td>0.968</td></tr></table>

## S7. ROLE OF THE CONSISTENCY OBJECTIVE

The reconstruction and consistency objectives both compare predictions with $x _ { \star } ,$ but evaluate the network on different state distributions. Reconstruction uses states from the forward process, whereas consistency uses states generated by a model update.

Suppose the network is pointwise optimal, meaning

$$
{ \hat { x } } _ { \star } ( x _ { t } ; t , x _ { \mathrm { o b s } } ) = \mathbb { E } [ x _ { \star } \mid x _ { t } , x _ { \mathrm { o b s } } ] \quad { \mathrm { a t ~ e v e r y ~ } } ( x _ { t } , t ) { \mathrm { ~ v i s i t e d } } ,
$$

so that $\nabla _ { \theta } { \mathcal { L } } _ { \mathrm { r e c } } ~ = ~ 0$ there. Applying the chain rule to Equation (20), with the unified gradient Equation (10) supplying the inner derivative,

$$
\nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { c o n s } } = \mathbb { E } \left[ - \frac { 2 \nu ( t ^ { \prime } ) \left( x _ { \star } - \hat { x } _ { \star } ( x _ { t ^ { \prime } } ) \right) } { V \left( \hat { x } _ { \star } ( x _ { t ^ { \prime } } ) \right) } \nabla _ { \boldsymbol { \theta } } \hat { x } _ { \star } ( x _ { t ^ { \prime } } ) \right] .\tag{S12}
$$

The stop-gradient removes derivatives through the generated state $x _ { t ^ { \prime } }$ , while the network is still evaluated on that state. The induced joint distribution of $( x _ { \star } , x _ { t ^ { \prime } } , x _ { \mathrm { o b s } } )$ can differ from the forward training distribution at $t ^ { \prime } .$ . Its optimal conditional-mean predictor can therefore differ as well, so optimal reconstruction on forward states does not imply that Equation (S12) vanishes. The consistency term addresses this possible distribution shift.

## S8. EVALUATION CONVENTIONS

We provide additional results on the sampling budget and worst-case evaluation.

## A. Number of function evaluations

Table S2 evaluates fixed-exponent models with one, five, and 25 network evaluations at severe noise. For matched exponents, five evaluations achieve a mean PSNR of 22.330 dB, compared with 22.326 dB for one evaluation and 22.225 dB for 25. Dark-region log-RMSE is also lowest at five evaluations (0.2831), compared with 0.2841 and 0.2854 at one and 25 evaluations, respectively. Additional evaluations are more helpful under exponent mismatch. From one to 25 evaluations, the minimum PSNR of the fixed $p \ = \ 1 . 5$ model across tested data exponents increases by 0.855 dB, while the mean mismatch penalty decreases from 1.601 to 0.924 dB. Thus, longer sampling reduces mismatch losses but does not improve matched-exponent PSNR in this comparison.

We further evaluate the conditional model with NFE ∈ {1, 3, 5, 8, 10, 15, 25} on a held-out split. PSNR peaks at three evaluations and decreases slightly thereafter. Relative to this peak, one, five, and 25 evaluations reduce PSNR by 0.021, 0.006, and 0.092 dB, respectively. All runs use deterministic sampling, and varying the projection onto X changes PSNR by at most 0.002 dB. We retain five evaluations as the common sampling budget, with a gap of only 0.006 dB to the best setting in this sweep.

## B. Evaluation across noise exponents

Mean mismatch loss and worst-case PSNR describe different aspects of exponent robustness. The former summarizes the cost of mismatched model–data exponent pairs and depends on the evaluation grid. At severe noise with five network evaluations, it is 2.000 dB on the $3 \times 3$ grid and 1.256 dB on the $5 \times 5$ grid. The latter is the minimum PSNR across the tested data exponents. We use this common criterion in Table II to compare fixed-exponent models with the conditional network. Among the fixed models, the minimax model maximizes this minimum.

## S9. ADDITIONAL EXPERIMENTAL RESULTS

## A. Additional restoration examples

Figures S1 and S2 show synthetic-image restorations across noise exponents at severe and mild noise, complementing the moderate-noise comparison in Figure 6. Corner values report PSNR for each displayed crop.

## B. Sensitivity to estimator settings

We assess the sensitivity of noise-exponent estimation to three settings: local window sizes {9, 15, 25}, intensity-bin counts {6, 10, 16}, and retained fractions {0.1, 0.3, 0.5} of the flattest samples within each bin. Across the tested settings, the maximum absolute exponent error $| { \hat { p } } - p |$ ranges from 0.067 to 0.089. On constant-intensity tiles, $| \hat { p } - p | \leq 0 . 0 0 6 6 .$ , and ν is recovered to three significant figures.

## S10. IMPLEMENTATION DETAILS

## A. Conditioning and training-step sampling

The observation is supplied through a separate conditioning channel at the training measurement index $\nu _ { \mathrm { o b s } }$ . This input remains fixed throughout sampling and is adjusted separately for noise-severity transfer.

We sample training steps uniformly from $\{ t _ { \operatorname* { m i n } } , \dots , T - 1 \}$ The expectation in Equation (17) is therefore evaluated as

$$
w = \frac { 1 } { T - t _ { \mathrm { m i n } } } \sum _ { t = t _ { \mathrm { m i n } } } ^ { T - 1 } \frac { \nu ( t ) } { \nu _ { \mathrm { o b s } } } .
$$

## B. The conditional model

We cache the 21 bridge configurations corresponding to the training exponents. Each optimizer step accumulates gradients over 12 micro-batches, each with its own sampled exponent. Exponent-dependent loss weighting. During conditional training, we multiply the reconstruction deviance by $\mu _ { \mathrm { r e f } } ^ { 2 - p }$ as an empirical weighting across exponents. Table VI evaluates this choice by replacing the factor with 1.

At inference, we update the exponent embedding and bridge configuration together, including the measurement index corresponding to the selected exponent.

Algorithm S1 Single-image noise-parameter estimation   
Require: Noisy intensity image x; h = 15, B = 10, ρ = 0.3   
Ensure: Regression estimates (ˆp, νˆ) under ϕ = 1   
1: $r _ { i , j } \gets \frac { 1 } { 2 } ( x _ { i , j } - x _ { i + 1 , j } \underset { \cdot } { - } x _ { i , j + 1 } + x _ { i + 1 , j + 1 } )$   
2: $\bar { x } \mathbin {  } \mathrm { B o x M E A N } ( x , h \mathbin { \times } h )$   
3: Partition samples into B bins by x¯   
4: for each intensity bin b do   
5: Retain the flattest fraction ρ of samples   
6: Estimate bin mean m and noise variance v   
from the retained local means and residuals   
7: end for   
8: $( \hat { p } , a ) \gets \mathrm { T H E I L S E N } \big ( \{ ( \log m _ { b } , \log v _ { b } ) \} _ { b = 1 } ^ { B } \big )$   
9: $\tilde { \nu } \gets \mathrm { e x p } ( - a )$   
10: return $( \hat { p } , \hat { \nu } )$  
TABLE S3

SAR NOISE-EXPONENT ESTIMATES ACROSS LOCAL-ENL WINDOW SELECTIONS (MEDIANS OVER IMAGES). NOMINAL AND DISPERSION-IMPLIED LOOKS, L, ARE INCLUDED FOR REFERENCE.
<table><tr><td rowspan="2">sensor</td><td rowspan="2">nominal L</td><td colspan="5">p at ENL band</td><td rowspan="2">implied L</td></tr><tr><td>all</td><td>5-95</td><td>10-95</td><td>25-95</td><td>40-95</td></tr><tr><td>miniSAR</td><td>1.00</td><td>2.150</td><td>2.042</td><td>2.014</td><td>2.005</td><td>2.002</td><td>1.06</td></tr><tr><td>FARAD-X</td><td>1.00</td><td>2.103</td><td>2.071</td><td>2.058</td><td>2.030</td><td>2.013</td><td>0.92</td></tr><tr><td>TerraSAR-X</td><td>1.16</td><td>2.106</td><td>2.067</td><td>2.044</td><td>2.014</td><td>2.005</td><td>1.37</td></tr></table>

## C. Noise-severity transfer

For noise-severity transfer, only the conditioning input is thinned to the training index $\nu _ { \mathrm { o b s } } .$ , while the observed image used to initialize sampling is unchanged. For $1 < p < 2 ,$ synthetic evaluations use simulator-retained latent counts for this thinning step.

## D. Noise-exponent estimation

Algorithm S1 summarizes the single-image estimator. The cross-difference residual removes locally affine signals. Under independent pixel noise, its variance equals the average noise variance of the four pixels. A 15 × 15 box filter supplies the local mean intensities used for binning. Retaining the flattest 30% within each of 10 bins limits the contribution of image structure to the variance estimates, which are then used in the Theil–Sen fit. The pseudocode returns the regression estimates before the range clipping described in the main paper.

## E. Baseline configurations

I<sup>2</sup>SB and ResShift use our UNet and training pool. ResShift operates in pixel space with five stochastic sampling steps. DDRM uses an unconditional DDPM prior trained on the clean pool. On synthetic data, its scalar noise level is computed from the true per-pixel variance. Noise2VST fits a transform to each test image for 2,000 iterations with frozen pretrained denoisers. Its reported NFE = 1 counts the final denoising pass. EAMamba uses 128×128 training crops, while evaluation uses 256 × 256 crops.

## S11. SAR DATA AND SUPPLEMENTARY EVALUATION

We convert Sandia miniSAR/FARAD complex data and ESA TerraSAR-X 16-bit MGD-SE amplitudes to linear intensity, excluding display-rendered or terrain-resampled images. Table S3 shows local-ENL sensitivity: at the 40–95 percentiles, fitted slopes span 2.002–2.016, close to the Gamma branch’s quadratic variance law.

![](images/fc3d0baed211a9604f3406eaa6222d69fc5465db2971e8de5958497f21ed4c32.jpg)  
Fig. S1. Synthetic-image restorations at severe noise $( \eta = 1 ) ,$ with one row per noise exponent. Corner values show PSNR (dB).

![](images/4fc2282e5f46df58c6858f8d619bc8ebb9d3abf67f08d85e427ab9196deb3d8c.jpg)  
Fig. S2. Synthetic-image restorations at mild noise $( \eta = 0 . 2 5 )$ , with one row per noise exponent. Corner values show PSNR (dB).