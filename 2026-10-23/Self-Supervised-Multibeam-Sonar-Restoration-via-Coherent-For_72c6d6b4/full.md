# Self-Supervised Multibeam Sonar Restoration via Coherent Forward Modeling

Zixin Wang, James Hopgood, and Mike Davies

Abstract—Multibeam imaging sonar produces intensity images degraded by coherent speckle and system-induced blur, while clean reflectivity targets are rarely available. Existing despeckling methods often rely on simplified intensity-domain noise models, synthetic clean targets, or blind-spot self-supervision, and typically do not jointly model deblurring and speckle suppression.

We propose a physics-consistent self-supervised framework for joint despeckling and deblurring of multibeam imaging sonar. A stochastic coherent image-formation model yields a conditionalmean forward operator that relates latent scene reflectivity to expected intensity through the squared magnitude of the system point spread function. We then use registered multi-frame sonar sequences to train without clean ground truth images, showing that, under an idealized inter-frame conditional-mean condition, the expected paired loss equals the corresponding forwardprediction loss up to a network-independent constant. To stabilize the ill-conditioned inverse problem, we further adapt robust equivariant imaging using stochastic coherent simulation and introduce a paired forward prediction error metric for groundtruth-free evaluation. Experiments on two real multibeam sonar datasets demonstrate a favourable balance between forwardprediction agreement, homogeneous-region speckle suppression, and preservation of visible structural boundaries compared with representative baselines.

Index Terms—Multibeam imaging sonar, coherent image formation, spatially correlated speckle, self-supervised image reconstruction, equivariant imaging.

## I. INTRODUCTION

M <sup>ULTIBEAM</sup> <sup>imaging</sup> <sup>sonar</sup> <sup>is</sup> <sup>widely</sup> <sup>used</sup> <sup>for</sup> <sup>under-</sup> water perception, mapping, inspection, and robotic navigation because it remains effective in turbid water and lowvisibility environments [1], [2], [3]. However, sonar images are often degraded by coherent speckle and system-induced blur, which obscure small targets, weaken object boundaries, and reduce the reliability of downstream tasks such as detection, segmentation, and scene interpretation [4], [5], [6]. Effective restoration therefore requires not only speckle suppression, but also recovery of structurally faithful and spatially sharp image content.

A central difficulty is that practical imaging sonars usually provide detected intensity images, while the dominant degradations arise from coherent acoustic image formation. The beamformed complex field is formed by the coherent superposition of echoes from unresolved scatterers and is subsequently converted to intensity. This process gives rise to signal-dependent, non-Gaussian speckle rather than simple additive noise [7], [8], [9]. In addition, the system response acts on the coherent field before intensity detection, so speckle correlation and blur are coupled through the effective point spread function (PSF) of the imaging system [10], [11], [12]. Since clean reflectivity ground truth is rarely available, imagingsonar restoration must address a coupled inverse problem without ground truth: joint despeckling and deblurring from intensity-only observations.

Existing sonar despeckling methods, together with the broader Synthetic Aperture Radar (SAR) despeckling literature, are often based on simplified intensity-domain statistical models. Classical approaches commonly treat speckle as multiplicative noise, frequently with exponential or Gamma statistics, and sometimes transform it into an additive form in the logarithmic domain [10], [13], [14], [15]. These assumptions have motivated local adaptive filters, Bayesian estimators, diffusion-based methods, and nonlocal patch aggregation approaches [16], [17], [18], [19]. However, these formulations primarily describe marginal or locally filtered intensity statistics. Spatial correlation is typically introduced through filtering windows, patch matching, aggregation, or hand-specified correlation structures [20]. Consequently, they do not provide a sonar-specific analytical observation model in which spatially correlated speckle and point spread function (PSF) blur are generated jointly.

Learning-based despeckling methods can achieve strong empirical performance, but supervised training generally requires clean reflectivity targets, which are rarely obtainable in real sonar imaging [15], [21], [22], [23], [24]. Many methods therefore rely on synthetic clean images, simulated speckle corruption, or architectures transferred from optical denoising [25], [26]. These strategies alleviate the lack of labeled sonar data, but they can introduce a mismatch between the training degradation and real sonar observations, where coherent speckle, spatial correlation, system blur, platform motion, and sensor-specific effects are coupled by the acquisition process. A recent despeckling method adopts a multiplicative y = xn model with Gamma-distributed speckle and constrains log-domain residual statistics [27]. However, its formulation remains a single-image despeckling framework and does not explicitly model PSF-coupled coherent image formation or the spatially correlated speckle induced by the imaging process.

Self-supervised methods reduce the dependence on clean targets by constructing supervision from noisy measurements themselves. Blind-spot, masking-based, and subsamplingbased approaches prevent the network from directly observing the prediction target and have been widely used for image despeckling [28], [29], [30], [31], [32], [33], [34], [35], [36]. These methods usually rely on a conditional-independence assumption: the corruption at the prediction target should be independent of the visible measurements given the underlying clean signal. This assumption is difficult to satisfy in imaging sonar because coherent propagation and the system response introduce spatially correlated speckle. Enlarging the blind spot or excluding neighbouring pixels can reduce noise leakage, but it also removes local spatial context that is important for preserving structure and undoing system blur.

Noisy-pair learning provides a complementary route to selfsupervision. Noise2Noise showed that clean targets can be replaced by independently corrupted observations with the same conditional mean, and Frame2Frame extended this idea to registered neighbouring video frames [37], [38]. Related principles have also been explored for multitemporal SAR despeckling and registered sonar sequences [39], [40]. In coherent sonar, however, neighbouring frames are not simply independent additive-noise observations: they are affected by viewpoint change, platform motion, registration error, coherent speckle decorrelation, and the system point spread function. Therefore, a registered-frame objective must specify not only how frames are paired, but also which physical domain the network output represents.

Our preliminary work, PI-SSD [40], used registered sonar frame pairs for self-supervised despeckling. However, it matches the network output directly to the second observation, $\| f _ { \theta } ( y _ { 1 } ) - y _ { 2 } \| _ { 2 } ^ { 2 }$ , without applying the PSF-dependent forward operator to the output. The network therefore targets a still PSF-blurred intensity image.

Building on PI-SSD, the present work uses the coherent image-formation model explicitly in the learning objective to formulate a self-supervised inverse problem, rather than treating paired frames as generic noisy images. We derive the conditional mean intensity learning goal for the coherent speckle-contaminated forward model, showing how the expected intensity observation is related to the scene reflectivity and the squared magnitude of the system PSF. This provides a physically meaningful forward prediction target for training from registered multi-frame sonar observations without clean reflectivity ground truth. The resulting inverse problem remains ill-conditioned, since different reflectivity estimates can produce similar intensity observations. We therefore extend robust equivariant imaging (REI) [41], [42] to sonar restoration by adapting it to the derived conditional mean learning goal and incorporating it within a Noise2Noisetype self-supervised training framework. This combines a physics-consistent paired loss, which exploits registered sonar frames with approximately decorrelated speckle, with REI regularization to constrain the reconstruction under known transformations. Finally, using the same paired-frame protocol, we introduce paired forward prediction error (PFPE) to assess ground-truth-free forward-prediction consistency with registered paired observations.

This work treats multibeam-sonar restoration as a physicsconsistent self-supervised inverse problem, rather than as generic image denoising. The network estimates a latent reflectivity image, and supervision is imposed only after projection through a coherent conditional-mean forward model. This distinguishes the proposed framework from blind-spot denoising and direct noisy-pair despeckling methods, which operate primarily in the observation/image domain. The main contributions of this work are summarized as follows:

• We develop a coherent-formation-based statistical observation model for imaging sonar restoration. The model explains spatially correlated speckle and system-induced blur as coupled consequences of coherent propagation and the sonar PSF, rather than imposing speckle correlation through intensity-domain post-filtering.

• We derive the corresponding conditional mean intensity learning goal for the coherent speckle-contaminated forward model. This provides a physics-consistent forward prediction objective for learning from detected intensity measurements without requiring clean reflectivity ground truth.

• We propose a self-supervised reconstruction framework for joint despeckling and deblurring using registered multi-frame sonar observations. Unlike direct noisypair matching, the paired loss constrains the forward projection $A f _ { \theta } ( y _ { 1 } )$ of the latent reconstruction, while REI regularization uses stochastic coherent simulation from the same image-formation model to reduce the illconditioning of sonar intensity reconstruction.

• Finally, based on the same multi-frame analysis, we introduce paired forward prediction error (PFPE) as a groundtruth-free measure of measurement-domain forwardprediction consistency under a fixed paired-frame protocol. PFPE provides a complementary evaluation criterion for real sonar data where clean reflectivity ground truth is unavailable.

## II. PHYSICAL OBSERVATION MODEL

Multibeam imaging sonar systems provide detected intensity images, but these images are generally impacted by speckle and blur as a byproduct of the coherent imaging process. Let Ω denote the discrete range-angle image grid of the sonar, with pixel index $i \in \Omega$ . We write $x \in \mathbb { R } _ { + } ^ { | \widetilde { \Omega } | }$ for the latent scene reflectivity in intensity units, where $x _ { i } \geq 0$ is the reflectivity at pixel i. An observed intensity image is denoted by $\boldsymbol { y } \in \mathbb { R } _ { + } ^ { | \Omega | }$

We model one observed intensity realization as

$$
y = \left| u * h \right| ^ { 2 } = \left| ( { \sqrt { x } } \cdot g ) * h \right| ^ { 2 } ,\tag{1}
$$

where $u = \sqrt { x } \cdot g \in \mathbb { C } ^ { | \Omega | }$ is the speckle-modulated coherent field, · denotes pointwise multiplication, and ∗ denotes spatial convolution. The magnitude square in (1) is applied pointwise.

Under the fully developed speckle model [12], $g \in \mathbb { C } ^ { | \Omega | }$ is a circular complex Gaussian speckle field. At each range-angle bin $i \in \Omega$

$$
g _ { i } \sim \mathcal { C N } ( 0 , 1 ) , \qquad \mathbb { E } [ | g _ { i } | ^ { 2 } ] = 1 .\tag{2}
$$

Before processing by the imaging system, this gives $\mathbb { E } [ | u _ { i } | ^ { 2 } \ |$ $x ] = x _ { i }$

Here, $h \in \mathbb { C } ^ { | \Omega | }$ is the coherent point spread function (PSF) of the sonar system, written as

$$
h : = h ( t , \vartheta ) = R ( t ) B ( \vartheta ) ,\tag{3}
$$

![](images/33e31163663803c499c3d78ea59bb3a2ba03cc270561bda5b379be30322e7148.jpg)  
Fig. 1. Empirical spatial correlation of detected sonar intensity measured from homogeneous ROIs in a representative Boat frame. Correlation is computed along the range and azimuth directions; offsets start from one pixel to exclude the trivial zero-offset self-correlation. Curves show the mean across ROIs, with shaded bands indicating one standard deviation.

where t denotes two-way travel time, $R ( t )$ is the temporal response, and $B ( \vartheta )$ is the angular beam pattern [11], [43], [44]. For simplicity, we use a local shift-invariant approximation of $h$ evaluated around the central beam direction, which keeps the forward operator tractable for local image regions.

In imaging sonar, the coherent PSF mixes neighbouring coherent returns before magnitude-square detection. Under fully developed speckle, the single-look intensity has exponential marginal statistics [8], [12]. Since the coherent field is filtered before detection, the resulting intensity samples are spatially correlated rather than independent pointwise speckle. This spatial correlation is illustrated in Fig. 1, where empirical correlation decays with range and azimuth spatial offset. This also explains a limitation of blind-spot self-supervision for sonar restoration: reducing leakage from spatially correlated speckle requires masking a local neighbourhood around the prediction target, but this removes precisely the nearby structural information that is important for joint despeckling and deblurring.

## III. METHODOLOGY

## A. Problem Formulation

Given detected sonar intensity observations generated by the coherent model in Section II, our goal is to estimate the latent reflectivity x without clean ground truth. Fig. 2 summarizes the proposed self-supervised framework. The reconstruction network is defined as

$$
{ \hat { x } } = f _ { \theta } ( y ) .\tag{4}
$$

This inverse problem is ill-conditioned because coherent speckle introduces signal-dependent stochastic fluctuations, while the forward operator induced by the sonar PSF suppresses high-frequency reflectivity structure. Therefore, training must constrain the reconstruction in the measurement domain while regularizing the latent reflectivity estimate.

## B. Paired Observation Construction

Multi-frame sonar sequences provide the opportunity to exploit repeated observations when viewing an approximately static scene. To this end, we select and register pairs of frames to form observations $( y _ { 1 } , y _ { 2 } ^ { \mathrm { r e g } } )$ that share the same latent reflectivity but have different coherent speckle realizations. The paired frame can be selected from a short temporal window around $y _ { 1 }$ . For short acquisition intervals, the dominant inter-frame motion is approximated by a two-dimensional translation $\Delta \rho = \left( \Delta \rho _ { 1 } , \Delta \rho _ { 2 } \right)$ , estimated by phase correlation [45], [46]. We denote the resulting translation warp by $W _ { 2 \to 1 }$ so that $y _ { 2 } ^ { \mathrm { r e g } } = W _ { 2  1 } ( y _ { 2 } ^ { \mathrm { r a w } } )$ aligns the paired frame to $y _ { 1 }$ . After registration, the paired frames are assumed to have aligned conditional means,

$$
\mathbb { E } [ y _ { 2 } ^ { \mathrm { r e g } } [ \pmb { \rho } ] | x ] \approx \mathbb { E } [ y _ { 1 } [ \pmb { \rho } ] | x ] ,\tag{5}
$$

where $\rho ~ = ~ \left( \rho _ { 1 } , \rho _ { 2 } \right)$ denotes a discrete range-angle image coordinate. For notational simplicity, we write $y _ { 2 }$ for $y _ { 2 } ^ { \mathrm { r e g } }$ below. A representative before-and-after registration check is provided in the Supplementary Section S1 to illustrate the structural alignment obtained by this procedure.

## C. Statistical Assumptions and Stochastic Forward Model

For a registered pair observing the same local scene, we model

$$
y _ { k } = \left| \left( { \sqrt { x } } \cdot g _ { k } \right) * h \right| ^ { 2 } , \qquad k \in \{ 1 , 2 \} ,\tag{6}
$$

where $x$ is the shared latent reflectivity, h is the coherent PSF, and $g _ { k }$ is the coherent speckle modulation field of the k-th observation. We assume approximate scene stationarity after registration and reduced inter-frame speckle correlation. For the paired-loss analysis, this is expressed through the conditional-mean relation

$$
\operatorname { \mathbb { E } } [ y _ { 2 } \mid x , y _ { 1 } ] \approx \operatorname { \mathbb { E } } [ y _ { 2 } \mid x ] = \mu ( x ) , \qquad \operatorname { \mathbb { E } } [ y _ { 1 } \mid x ] = \mu ( x ) .\tag{7}
$$

Exact conditional independence, $y _ { 1 } \perp y _ { 2 } \mid x ,$ is a sufficient but stronger condition for this relation.

For a fixed reflectivity $x ,$ the measurement-domain target under the mean squared error criterion is

$$
\mu ( x ) \triangleq \mathbb { E } _ { g } [ y \mid x ] .\tag{8}
$$

Let $v$ denote the coherent field after PSF convolution:

$$
v = ( { \sqrt { x } } \cdot g ) * h , \qquad y = | v | ^ { 2 } .\tag{9}
$$

For a two-dimensional index $p ~ = ~ ( p _ { 1 } , p _ { 2 } ) ~ \in ~ \Omega$ and $q =$ $( q _ { 1 } , q _ { 2 } ) \in \Omega$ , the spatial convolution can be written as

$$
v _ { p } = \sum _ { q \in \Omega } h _ { p - q } \sqrt { x _ { q } } g _ { q } .\tag{10}
$$

Then,

$$
\begin{array} { l } { \mu _ { p } ( x ) = \mathbb { E } _ { g } [ | v _ { p } | ^ { 2 } \mid x ] } \\ { = \displaystyle \sum _ { q \in \Omega } \displaystyle \sum _ { r \in \Omega } h _ { p - q } h _ { p - r } ^ { * } \sqrt { x _ { q } x _ { r } } \mathbb { E } [ g _ { q } g _ { r } ^ { * } ] . } \end{array}\tag{11}
$$

(12)

Using the spatial whiteness and unit second moment of the coherent speckle modulation field [7],

$$
\mathbb { E } _ { g } [ g _ { q } g _ { r } ^ { * } ] = \delta _ { q r } .\tag{13}
$$

![](images/32076f60fa8fc96c600fdf7e93a938233893355f3653214a2a944aed91509ad2.jpg)  
Fig. 2. Overview of the proposed framework. The framework contains four components: a coherent physical observation model, an image-registration module that constructs registered paired observations, a physics-consistency branch that compares the forward projection of the reconstruction with a registered frame, and a robust equivariant imaging branch that regularizes the latent reconstruction. At inference time, only the trained reconstruction network is used and it operates on a single sonar image.

where $\delta _ { q r }$ is the Kronecker delta. Hence, all cross terms vanish, giving

$$
\mu _ { p } ( x ) = \sum _ { q } x _ { q } | h _ { p - q } | ^ { 2 } .\tag{14}
$$

Thus, under the local shift-invariant approximation, despite the stochastic observation model being nonlinear in x and $^ { g , }$ the conditional mean can be expressed as a linear function of x:

$$
\mu ( x ) = A x = x * | h | ^ { 2 } ,\tag{15}
$$

where A is the conditional-mean forward operator induced by the coherent observation model.

The same forward model is also used to simulate sonar measurements from a candidate reflectivity image z. We define the stochastic coherent simulator

$$
S _ { g } ( z ) : = \left| ( \sqrt { z } \cdot g ) * h \right| ^ { 2 } .\tag{16}
$$

Following the speckle-averaging principle in coherent imaging [12], we use a J-look averaged simulator

$$
\bar { S } _ { J } ( z ) : = \frac { 1 } { J } \sum _ { j = 1 } ^ { J } S _ { g _ { j } } ( z ) ,\tag{17}
$$

where $g _ { 1 } , \ldots , g _ { J }$ are independent speckle fields. Provided each simulated look satisfies $\mathbb { E } _ { g _ { j } } [ S _ { g _ { j } } ( z ) \mid z ] = A z ,$ , the averaged simulator preserves the same conditional mean,

$$
\mathbb { E } _ { g _ { 1 } , \ldots , g _ { J } } \big [ \bar { S } _ { J } ( z ) \mid z \big ] = A z ,\tag{18}
$$

Equivalently, writing $\varepsilon _ { j } ( z ) = S _ { g _ { j } } ( z ) - A z$ , we have $\mathbb { E } [ \varepsilon _ { j } ( z ) ~ | ~ z ] ~ = ~ 0$ and $\begin{array} { r } { \bar { S } _ { J } ( z ) \mathrm { ~ - ~ } A \bar { z ^ { \mathrm { ~ } } } = \mathrm { ~ } J ^ { - 1 } \sum _ { j = 1 } ^ { J } \varepsilon _ { j } ( z ) } \end{array}$

Independence of the simulated looks eliminates the cross terms in expectation, giving

$$
\begin{array} { r l } & { \mathbb { E } _ { g _ { 1 } , \dots , g _ { J } } \Big [ \big \| \bar { S } _ { J } ( z ) - A z \big \| _ { 2 } ^ { 2 } \Big | z \Big ] } \\ & { \qquad = \cfrac { 1 } { J } \mathbb { E } _ { g } \Big [ \big \| S _ { g } ( z ) - A z \big \| _ { 2 } ^ { 2 } \Big | z \Big ] . } \end{array}\tag{19}
$$

Thus, J controls the strength of the simulated speckle while preserving the conditional mean learning goal. This supports the interpretation of $\bar { S } _ { J }$ as a J-look simulated sonar measurement: $\textit { J } = \textit { 1 }$ corresponds to a single-look simulated observation, whereas $\bar { S } _ { J } ( z )$ converges in mean square to $A ( z )$ as $J \to \infty$ under the independent-look simulator.

## D. Physics Consistency Learning

Our objective is to learn a reconstruction network that approximates the conditional mean of the latent reflectivity,

$$
\hat { x } = f _ { \theta } ( y _ { 1 } ) \approx \mathbb { E } [ x \mid y _ { 1 } ] .\tag{20}
$$

However, clean samples of x are not available. Using the registered-pair assumptions in Section III-C and the conditional mean relation in (15), we train through the measurement-domain prediction $A f _ { \theta } ( y _ { 1 } )$ . The physics consistency loss is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { P C } } ( \boldsymbol { \theta } ) = \mathbb { E } _ { ( y _ { 1 } , y _ { 2 } ) \sim p _ { \mathrm { p a i r } } } \left[ \left| \left| y _ { 2 } - A f _ { \boldsymbol { \theta } } ( y _ { 1 } ) \right| \right| _ { 2 } ^ { 2 } \right] , } \end{array}\tag{21}
$$

where $p _ { \mathrm { p a i r } }$ denotes the distribution of registered paired observations. Under the assumptions above, this loss decomposes as

$$
\mathcal { L } _ { \mathrm { P C } } ( \theta ) = \mathbb { E } _ { x , y _ { 1 } } \left[ \| A x - A f _ { \theta } ( y _ { 1 } ) \| _ { 2 } ^ { 2 } \right] + C ,\tag{22}
$$

where

$$
C = \mathbb { E } _ { x } \left[ \mathbb { E } _ { y _ { 2 } | x } \left[ \| y _ { 2 } - \mu ( x ) \| _ { 2 } ^ { 2 } \right] \right]\tag{23}
$$

is independent of θ.

Under the idealized condition $\mathbb { E } [ y _ { 2 } \mid x , y _ { 1 } ] = A x$ , the cross term in the squared-loss expansion vanishes, yielding (22), with the network-independent constant defined in (23). Thus, minimizing $\mathcal { L } _ { \mathrm { P C } }$ matches the conditional mean projected onto the measurement domain, rather than fitting a single noisy realization.

The minimizers are not necessarily unique in the reflectivity domain. If $A ^ { \dagger }$ denotes the Moore–Penrose pseudoinverse, a solution can be written as

$$
f _ { \theta } ( y _ { 1 } ) = A ^ { \dagger } A \mathbb { E } [ x \mid y _ { 1 } ] + \nu ( y _ { 1 } ) , \qquad \nu ( y _ { 1 } ) \in \mathrm { n u l l } ( A ) ,\tag{24}
$$

consistent with the equivariant splitting ambiguity of inverse problems [47, Lemma 1]. In practice, even when A is full rank, it is a blur operator and can be severely ill-conditioned. Therefore, the physics consistency loss alone weakly determines high-frequency reflectivity components, motivating the equivariant regularization introduced next.

E. Robust Equivariant Imaging with Stochastic Forward Simulation

To further constrain the latent reflectivity, we adapt robust equivariant imaging (REI) [42] to the coherent speckle noise model introduced above. REI enforces consistency of the reconstruction system under transformations that preserve plausible reflectivity structure. Let $T _ { \tau }$ denote such a transformation, for example a small rotation or shift. Given

$$
\hat { x } = f _ { \theta } ( y _ { 1 } ) , \qquad z = T _ { \tau } \hat { x } ,\tag{25}
$$

the REI branch requires the network to recover z from a simulated sonar measurement generated using the J-look coherent simulator $\bar { S } _ { J }$ . The sonar-adapted REI loss is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { R E I } } ^ { ( J ) } ( \theta ) = \mathbb { E } _ { y _ { 1 } , \tau , g _ { 1 } , \ldots , g _ { J } } \Big [ \big | \big | f _ { \theta } \big ( \bar { S } _ { J } ( T _ { \tau } f _ { \theta } ( y _ { 1 } ) ) \big ) } \\ { - T _ { \tau } f _ { \theta } ( y _ { 1 } ) \big | \big | _ { 2 } ^ { 2 } \Big ] . } \end{array}\tag{26}
$$

The final training objective is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { t o t a l } } ( \theta ) = \mathcal { L } _ { \mathrm { P C } } ( \theta ) + \alpha \mathcal { L } _ { \mathrm { R E I } } ^ { ( J ) } ( \theta ) , } \end{array}\tag{27}
$$

where $\alpha > 0$ controls the REI regularization strength.

## IV. EXPERIMENTS

We evaluate the proposed framework on two real multibeam imaging sonar datasets acquired with a Tritech Gemini 720ik sonar. In the absence of clean reflectivity ground truth, realsonar evaluation cannot directly measure latent reflectivity accuracy. However, when registered pairs of corrupted sonar observations are available, the same Noise2Noise principle used for training provides a paired-frame proxy for measurementdomain predictive consistency. We therefore assess restoration using complementary evidence: the proposed paired forward prediction error (PFPE) measures measurement-domain forward-prediction consistency, while equivalent number of looks (ENL), edge preservation index (EPI), and visual comparisons assess speckle suppression, edge preservation, and structural plausibility. In this setting, deblurring is evaluated indirectly through consistency with the sonar forward model and preservation of visible structures.

## A. Evaluation Metrics

The PFPE metric tests whether a restoration from one frame can predict a registered paired observation. Let $y _ { 1 }$ be the input frame, $y _ { 2 }$ its registered paired observation, and xˆ the restored output of an evaluated method. For the proposed method, ${ \hat { x } } =$ $f _ { \boldsymbol { \theta } } ( y _ { 1 } )$ ). We define

$$
\mathrm { P F P E } ( y _ { 1 } , y _ { 2 } ) = \left\| A \hat { x } - y _ { 2 } \right\| _ { 2 } ^ { 2 } .\tag{28}
$$

The aggregate PFPE improvement is computed from the ratio of empirical mean errors as

$$
\mathrm { P F P E _ { i m p } ^ { a g g } } = 1 0 0 \left( 1 - \frac { \overline { { \mathrm { P F P E } ( y _ { 1 } , y _ { 2 } ) } } } { \overline { { \| y _ { 1 } - y _ { 2 } \| _ { 2 } ^ { 2 } } } } \right) ,\tag{29}
$$

Here, squared norms are normalized by the number of evaluated image pixels, and the overline denotes the empirical mean over all evaluated test pairs. Tables report PFPE as mean ± standard deviation across these pairs.

Lower PFPE and higher $\mathrm { P F P E _ { i m p } ^ { a g g } }$ scores indicate better agreement under the fixed paired-frame protocol. Under the idealized conditional-mean condition in Section III-C, expected PFPE equals expected conditional-mean measurement error plus a reconstruction-independent constant. Unlike single-frame consistency $\lVert A f _ { \theta } ( y _ { 1 } ) - y _ { 1 } \rVert _ { 2 } ^ { 2 }$ , the heldout criterion is less biased toward preserving input-specific speckle because the prediction is compared with a registered neighbouring observation. All restored outputs are scored using the same sonar operator A to predict a registered paired observation, so PFPE is interpreted under a common latentreflectivity hypothesis rather than as a full-reference quality metric or an evaluation of each baseline’s native prediction target. We therefore interpret PFPE jointly with ENL, EPI, $D _ { \mathrm { E P I } }$ , and visual results.

We also report ENL and edge preservation index deviation $( D _ { \mathrm { E P I } } )$ as complementary no-reference indicators. These metrics characterize homogeneous-region smoothing and edgestrength preservation, respectively, but they do not measure reconstruction accuracy in the absence of ground truth. ENL measures speckle suppression in homogeneous regions,

$$
\mathrm { E N L } = \frac { \mu _ { \hat { x } , \Omega _ { \mathrm { h o m } } } ^ { 2 } } { \sigma _ { \hat { x } , \Omega _ { \mathrm { h o m } } } ^ { 2 } } ,\tag{30}
$$

where $\Omega _ { \mathrm { h o m } }$ denotes homogeneous regions and $\mu _ { \hat { x } , \Omega _ { \mathrm { h o m } } }$ and $\sigma _ { \hat { x } , \Omega _ { \mathrm { h o m } } } ^ { 2 }$ are the mean and variance of xˆ within these regions. Higher ENL indicates stronger speckle suppression, but may favor oversmoothing.

EPI measures relative edge strength in selected edge regions:

$$
\mathrm { E P I } _ { n } = \frac { \sum _ { ( i , j ) \in \Omega _ { n , \mathrm { e d g e } } } | \nabla \hat { x } _ { n } ( i , j ) | } { \sum _ { ( i , j ) \in \Omega _ { n , \mathrm { e d g e } } } | \nabla y _ { n } ( i , j ) | } ,\tag{31}
$$

where $\Omega _ { n , \mathrm { e d g e } }$ is selected from the observed image $y _ { n }$ and held fixed across methods, and ∇ denotes a finite-difference gradient operator. We report the mean per-image deviation from unity:

$$
D _ { \mathrm { E P I } } = \frac { 1 0 0 } { N } \sum _ { n = 1 } ^ { N } \left| 1 - \mathrm { E P I } _ { n } \right| ,\tag{32}
$$

where N is the number of evaluated images. A smaller $D _ { \mathrm { E P I } }$ indicates closer agreement of edge strength with the observed images.

For ENL, homogeneous ROI candidates are required to have sufficient mean intensity, valid-pixel support, and low Sobel gradient energy, and the retained candidates are ranked by coefficient of variation; for EPI and $D _ { \mathrm { E P I } }$ , edge ROIs are selected separately as the patches with the highest Sobel gradient energy. All ROIs are selected from the input observation and then fixed for all compared methods. This avoids selecting regions according to the restored output and reduces manual selection bias.

## B. Datasets

We use two datasets acquired with a Tritech Gemini 720ik sonar. The sensor outputs intensity images formed from 512 beams, operates at a centre frequency of $7 2 0 \mathrm { k H z } ,$ and provides a $1 2 0 ^ { \circ }$ field of view. Raw measurements are represented as range-beam images in polar coordinates and treated as singlechannel intensity images.

Boat. The Boat dataset contains complex underwater scenes with cluttered object responses and strong background speckle, seabed texture, acoustic shadows, and heterogeneous scattering from object boundaries and surrounding structures. These factors make the dataset challenging for joint deblurring and despeckling, since the method must suppress coherent speckle while preserving structural details and boundaries. It was acquired with a maximum imaging range of 50 m and a frame rate of 3 frames per second. The dataset contains 1455 raw frames.

Underwater Structure. The Underwater Structure dataset contains more regular structural scenes and provides a complementary evaluation under relatively stable underwater inspection conditions. It was acquired with a maximum imaging range of 25.5 m and a frame rate of 10 frames per second. The dataset contains 1282 raw frames.

## C. Implementation Details

For each raw sequence, we use temporally contiguous training, validation, and test blocks with an approximate $8 : 1 : 1$ ratio before fixed $5 1 2 \times 5 1 2$ patch extraction. A small buffer of unused frames is inserted between adjacent blocks, and paired-frame samples are formed only within the same split. This prevents temporally adjacent or registered pairs from crossing the train/validation/test boundaries. During testing, the fully convolutional network is applied to individual sonar frames. We compare the proposed method with representative model-based despeckling methods, including PPB [18] and SAR-BM3D [19], and recent learning-based self-supervised baselines, including Noise2Void [29], Blind2Unblind (B2UB) [35], LG-BPN [33], SS-BSN [34], SEGSID [36], and PI-SSD [40].

All paired-frame experiments use a fixed main protocol with frame gap 1. The gap is chosen as the smallest reliable gap that reduces inter-frame correlation while preserving scene match and registration quality. Pairs are retained only when the estimated translation satisfies $| \Delta \rho _ { 1 } | \leq 1 0$ pixels and $| \Delta \rho _ { 2 } | \le 5$ pixels, which removes pairs with large apparent motion, scene mismatch, or failed registration. This yields 1328 and 1137 retained registered pairs for the Boat and Underwater Structure datasets, respectively, under the final pairing protocol. The full gap-correlation statistics and displacement diagnostics are reported in Supplementary Section S2. All methods are evaluated using the same fixed test pairs.

The PSF is generated from the Gemini 720ik geometry and parameters reported in Section IV-B, together with a sound speed of 1540 $\mathrm { m } / \mathrm { s }$ and a 96-element receive aperture with half-wavelength spacing.

For the network, we use a standard U-Net backbone implemented with deepinv [48]. This compact four-scale U-Net keeps the inference model lightweight and is chosen to support future low-latency deployment on robotic sonar platforms. All models are trained with Adam using a learning rate of $1 \times 1 0 ^ { - 4 }$ , a weight decay of $5 \times 1 0 ^ { - 7 }$ , and a batch size of 2. The checkpoint with the lowest validation loss is selected for testing. Unless otherwise stated, the main experiments use $J \ = \ 6 .$ , with $T _ { \tau }$ implemented as random rotations with a maximum angle of 20<sup>◦</sup> for REI regularization. The REI weight is set to $\alpha = 0 . 0 3$ for Boat and $\alpha = 0 . 0 5$ for Underwater Structure, based on validation performance. This dataset-specific choice reflects the different registration accuracy and structural complexity of the two datasets, and no test-set results were used for this selection. Experiments are implemented in PyTorch 2.2.2 and run on a single NVIDIA RTX 3090 GPU.

## D. Performance on Boat Dataset

Quantitative evaluation. The Boat results are reported in the first four metric columns of Table I, and the corresponding metric distributions are shown in Fig. 3. Additional qualitative examples and complete metric values are provided in Supplementary Section S3, including Fig. 7 and Table V. PFPE measures the raw held-out forward-prediction error, while $\mathrm { P F P E _ { i m p } ^ { a g g } }$ reports the aggregate improvement over the noisy registered-pair baseline.

The proposed method achieves the lowest PFPE and the highest aggregate PFPE improvement of 14.10% under the stated common latent-reflectivity scoring protocol, indicating the strongest overall forward-prediction consistency on the Boat dataset. Blind2Unblind (B2UB) obtains a similar aggregate PFPE improvement of 13.39%, but its ENL decreases from the raw image value of 8.18 to 7.74, indicating limited speckle suppression. In contrast, the proposed method increases ENL to 12.81 while keeping $D _ { \mathrm { E P I } }$ among the lowest restored results. Methods with larger ENL, such as SS-BSN, LG-BPN, and PI-SSD, show much larger edge-strength deviation and lower PFPE improvement, suggesting that their higher smoothness is associated with oversmoothing rather than stronger forward-prediction agreement. The ablation results further show why PFPE should not be interpreted in isolation. As reported in Supplementary Section S5-A and Supplementary Table VII, a physics-consistency-only $( \mathcal { L } _ { \mathrm { P C } ^ { - } }$ only) variant improves PFPE but yields nearly zero ENL and extremely large $D _ { \mathrm { E P I } }$ , demonstrating that forward consistency alone does not ensure a plausible reconstruction. This supports the use of REI regularization and motivates the multi-metric evaluation protocol: PFPE is treated as a measurement-domain consistency metric and interpreted together with ENL, $D _ { \mathrm { E P I } }$ and visual structural evidence.

![](images/39ac0b71beb6ded4b56fc0b454f521df47fb991d9c873280faf96adf47a564e7.jpg)  
Fig. 3. The top panel shows $\mathrm { P F P E _ { i m p } ^ { a g g } }$ relative to the noisy baseline; the remaining three panels show distributions of per-pair PFPE and per-image ENL and EPI on the Boat dataset. In the three boxplots, boxes show interquartile ranges, horizontal lines indicate medians, and dots denote individual test images. Dashed gray lines and shaded gray bands indicate the corresponding noisy means and mean ± one standard deviation, respectively.

Qualitative evaluation. Fig. 4 shows representative visual comparisons on Boat. PPB and SAR-BM3D reduce speckle but blur local structures, while several learning-based baselines either retain granular texture or over-smooth object boundaries. The proposed method suppresses background speckle while preserving clearer separation of high-reflectivity structures in the zoomed regions. These observations agree with the quantitative results, where the proposed method achieves the best aggregate PFPE improvement while maintaining low edge-strength deviation.

## E. Performance on Underwater Structure Dataset

Quantitative evaluation. The Underwater Structure results are reported in the last four metric columns of Table I. Additional qualitative examples and complete metric values for this dataset are provided in Supplementary Section S4, including Fig. 8 and Table VI.

The proposed method achieves the lowest PFPE and the highest aggregate PFPE improvement of 4.36% under the latent-reflectivity hypothesis, indicating the strongest overall agreement with the paired forward-prediction criterion on this dataset. As in the Boat experiment, methods with substantially higher ENL, such as SS-BSN, PI-SSD, and LG-BPN, also show much larger $D _ { \mathrm { E P I } }$ values, indicating that stronger smoothing does not necessarily preserve observed edge strength. In contrast, the proposed method provides the best aggregate PFPE improvement while keeping $D _ { \mathrm { E P I } }$ relatively low, showing a favourable balance between forwardprediction consistency, speckle suppression, and structure preservation.

Qualitative evaluation. Fig. 5 shows representative visual comparisons on the Underwater Structure dataset. Modelbased methods retain residual speckle or blur weak structural details, while several self-supervised baselines produce smoother backgrounds at the cost of less distinct object boundaries. The proposed method suppresses background speckle while preserving clearer structural continuity, especially in the zoomed regions where the boundary between the underwater structure and the surrounding background remains more distinguishable.

## F. Computational Cost

Table II reports the computational cost on $5 1 2 \times 5 1 2$ images. Parameters and MACs are computed at the model level for learning-based methods, while time per image is measured using the actual test-time inference path used to generate the reported results. For the proposed method, only the trained reconstruction network is used at inference; the paired observation branch, stochastic simulator, and REI loss are used only during training.

The proposed method has the same backbone as PI-SSD and therefore the same parameter count and MACs, with a measured inference time of 0.043 s per image. This corresponds to approximately 23 frames per second, which is faster than the acquisition rates of both datasets considered here: 3 frames per second for Boat and 10 frames per second for Underwater Structure. This suggests that the physics-consistent self-supervised training objective improves restoration quality without requiring a heavier test-time model, and remains compatible with future real-time multibeam sonar processing.

## V. CONCLUSION

This work formulated multibeam imaging sonar restoration as a physics-consistent self-supervised inverse problem for joint despeckling and deblurring. Starting from a coherent stochastic observation model, we derived the conditional-mean intensity for the forward model that links latent reflectivity to detected sonar intensity. This model enables paired physicsconsistency learning from registered multi-frame observations without clean reflectivity targets. To regularize the illconditioned inverse problem, we further introduced robust equivariant imaging based on sonar-specific forward simulation.

Experiments on the Boat and Underwater Structure datasets show that the proposed method improves paired forward prediction while achieving a favourable balance between speckle suppression and edge preservation. Compared with representative model-based and learning-based baselines, the proposed framework produces reconstructions that are more consistent with the physical forward model and less prone to either residual speckle or excessive structural smoothing.

TABLE I  
QUANTITATIVE COMPARISON ON THE BOAT AND UNDERWATER STRUCTURE DATASETS. PFPE IS REPORTED IN UNITS OF $1 0 ^ { - 4 } ;$ , AND $D _ { \mathrm { E P I } }$ IS REPORTED IN PERCENT. $\operatorname { P F P E } _ { \operatorname { i m p } } ^ { \mathrm { a g g } }$ DENOTES THE AGGREGATE PERCENTAGE IMPROVEMENT OVER THE NOISY-FRAME BASELINE. THE LOWEST PFPE AND HIGHEST $\mathrm { P F P E _ { i m p } ^ { a g g } }$ ARE HIGHLIGHTED IN BOLD. ENL AND $D _ { \mathrm { E P I } }$ ARE COMPLEMENTARY INDICATORS OF SPECKLE SUPPRESSION AND EDGE-STRENGTH DEVIATION, RESPECTIVELY.
<table><tr><td></td><td colspan="4">Boat</td><td colspan="4">Underwater Structure</td></tr><tr><td>Method</td><td> $\mathrm { P F P E } \downarrow$ </td><td> $\mathrm { P F P E _ { i m p } ^ { a g g } }$  ←</td><td> $\mathrm { E N L } \uparrow$ </td><td> $D _ { \mathrm { E P I } } \downarrow$ </td><td> $\mathrm { P F P E } \downarrow$ </td><td> $\mathrm { P F P E _ { i m p } ^ { a g g } } \ .$  个</td><td> $\mathrm { E N L } \uparrow$ </td><td> $D _ { \mathrm { E P I } } \downarrow$ </td></tr><tr><td>Noisy</td><td> $1 . 5 9 \pm 0 . 3 6$ </td><td> $0 . 0 0$ </td><td> $8 . 1 8 \pm 1 . 1 2$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $6 . 7 2 \pm 3 . 0 6$ </td><td> $0 . 0 0$ </td><td> $3 . 3 8 \pm 0 . 3 6$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>PPB [18]</td><td> $1 . 7 9 \pm 0 . 2 6$ </td><td>-13.06</td><td> $1 2 . 7 3 \pm 2 . 7 1$ </td><td> $1 9 . 3 2 \pm 2 . 2 8$ </td><td> $7 . 3 3 \pm 2 . 3 4$ </td><td>-9.01</td><td> $6 . 9 8 \pm 1 . 3 6$ </td><td> $2 7 . 1 7 \pm 4 . 1 6$ </td></tr><tr><td>SAR-BM3D [19]</td><td> $1 . 4 0 \pm 0 . 2 9$ </td><td>11.53</td><td> $1 2 . 1 4 \pm 2 . 4 0$ </td><td> $2 9 . 8 7 \pm 1 . 6 0$ </td><td> $6 . 7 2 \pm 2 . 1 8$ </td><td>-0.02</td><td> $4 . 7 1 \pm 0 . 5 9$ </td><td> $2 5 . 4 2 \pm 1 . 0 5$ </td></tr><tr><td>Noise2Void [29]</td><td> $1 . 7 7 \pm 0 . 4 0$ </td><td>-11.59</td><td> $1 1 . 0 3 \pm 1 . 6 2$ </td><td> $2 6 . 6 1 \pm 1 . 3 6$ </td><td> $6 . 8 2 \pm 2 . 2 4$ </td><td>-1.49</td><td> $4 . 0 9 \pm 0 . 4 6$ </td><td> $1 7 . 2 0 \pm 0 . 5 6$ </td></tr><tr><td>B2UB [35]</td><td> $1 . 3 7 \pm 0 . 2 9$ </td><td>13.39</td><td> $7 . 7 4 \pm 1 . 0 3$ </td><td> $4 . 6 8 \pm 0 . 3 6$ </td><td> $6 . 5 7 \pm 2 . 1 8$ </td><td>2.25</td><td> $3 . 3 6 \pm 0 . 3 5$ </td><td> $1 . 1 1 \pm 0 . 1 3$ </td></tr><tr><td>LG-BPN [33]</td><td> $1 . 5 7 \pm 0 . 3 1$ </td><td>1.19</td><td> $1 8 . 5 7 \pm 5 . 6 0$ </td><td> $4 5 . 8 9 \pm 2 . 2 7$ </td><td> $7 . 8 0 \pm 2 . 3 5$ </td><td>-16.05</td><td> $7 . 1 1 \pm 1 . 3 5$ </td><td> $1 4 . 1 9 \pm 2 . 1 6$ </td></tr><tr><td>SS-BSN [34]</td><td> $1 . 6 2 \pm 0 . 3 2$ </td><td>-2.24</td><td> $2 5 . 0 4 \pm 8 . 7 9$ </td><td> $5 5 . 9 2 \pm 3 . 0 6$ </td><td> $8 . 8 5 \pm 2 . 8 4$ </td><td>-31.66</td><td> $1 5 . 5 0 \pm 5 . 2 2$ </td><td> $7 1 . 2 4 \pm 3 . 8 8$ </td></tr><tr><td>PI-SSD [40]</td><td> $1 . 4 4 \pm 0 . 2 7$ </td><td>9.39</td><td> $1 6 . 9 7 \pm 4 . 8 1$ </td><td> $4 3 . 2 2 \pm 3 . 2 4$ </td><td> $7 . 2 1 \pm 2 . 2 5$ </td><td>-7.36</td><td> $7 . 7 6 \pm 2 . 0 2$ </td><td> $5 1 . 2 9 \pm 2 . 1 9$ </td></tr><tr><td>SEGSID [36]</td><td> $1 . 4 6 \pm 0 . 3 1$ </td><td>8.20</td><td> $1 5 . 8 3 \pm 3 . 8 1$ </td><td> $4 0 . 8 2 \pm 2 . 1 6$ </td><td> $6 . 8 6 \pm 2 . 2 2$ </td><td>-2.02</td><td> $4 . 7 4 \pm 0 . 6 6$ </td><td> $2 8 . 8 6 \pm 1 . 4 4$ </td></tr><tr><td>Proposed</td><td> ${ \bf 1 . 3 6 \pm 0 . 2 8 }$ </td><td>14.10</td><td> $1 2 . 8 1 \pm 2 . 6 8$ </td><td> $4 . 6 5 \pm 4 . 1 3$ </td><td> ${ \bf 6 . 4 3 \pm 2 . 0 8 }$ </td><td>4.36</td><td> $6 . 3 7 \pm 2 . 8 4$ </td><td> $6 . 8 3 \pm 2 . 9 5$ </td></tr></table>

TABLE II

COMPUTATIONAL COST ON BOAT IMAGES. MACS ARE REPORTED USING THE MODEL-LEVEL PROTOCOL, AND TIME PER IMAGE IS MEASURED USING THE ACTUAL INFERENCE PATH.
<table><tr><td>Method</td><td>Params (M)</td><td>MACs (G)</td><td>Time/img (s) ↓</td></tr><tr><td>PPB</td><td></td><td></td><td>151.219</td></tr><tr><td>SAR-BM3D</td><td></td><td></td><td>324.225</td></tr><tr><td>Noise2Void</td><td>0.80</td><td>52.26</td><td>0.147</td></tr><tr><td>B2UB</td><td>1.10</td><td>73.94</td><td>0.018</td></tr><tr><td>LG-BPN</td><td>6.16</td><td>1605.23</td><td>13.789</td></tr><tr><td>SS-BSN</td><td>6.41</td><td>1757.44</td><td>7.334</td></tr><tr><td>PI-SSD</td><td>8.56</td><td>198.94</td><td>0.044</td></tr><tr><td>SEGSID</td><td>22.95</td><td>2044.56</td><td>112.511</td></tr><tr><td>Proposed</td><td>8.56</td><td>198.94</td><td>0.043</td></tr></table>

The current framework relies on registered paired observations and an approximate local PSF, and may be affected by severe registration errors, dynamic scene changes, or forwardmodel mismatch. A remaining limitation is that real-sonar deblurring is evaluated indirectly, since clean latent reflectivity ground truth is unavailable. Future work will focus on controlled simulations or calibration-target experiments to directly quantify reflectivity accuracy and PSF-mismatch robustness, as well as broader deployment across sonar platforms, imaging geometries, and more diverse underwater environments.

## REFERENCES

[1] H. Huang, M. Zhang, L. Zhang, D. Wang, and B. Wang, “A novel sonar image preprocessing method for AUV positioning based on underwater SLAM,” IEEE Trans. Instrum. Meas., vol. 74, pp. 1–14, 2025.

[2] M. Aubard, A. Madureira, L. Teixeira, and J. Pinto, “Sonar-based deep learning in underwater robotics: Overview, robustness, and challenges,” IEEE J. Ocean. Eng., vol. 50, no. 3, pp. 1866–1884, 2025.

[3] S. Das, P. K. Malik, and A. Pandey, “Challenges and advances in underwater sonar systems and AI-driven signal processing for modern naval operations: A systematic review,” J. Field Robot., vol. 43, no. 2, pp. 899–931, 2026.

[4] Y. Lu, J. Zhang, Q. Chen, C. Xu, M. Irfan, and Z. Chen, “AquaYOLO: Enhancing YOLOv8 for accurate underwater object detection for sonar images,” J. Mar. Sci. Eng., vol. 13, no. 1, 2025, Art. no. 73.

[5] J. Si, T. Zhou, X. Yu, W. Du, and S. Xu, “An unsupervised method for detecting and segmenting shadow areas of sunken targets in sonar images,” IEEE Trans. Instrum. Meas., vol. 74, pp. 1–15, 2025.

[6] M. Elmezain, L. Saad Saoud, A. Sultan, M. Heshmat, L. Seneviratne, and I. Hussain, “Advancing underwater vision: A survey of deep learning models for underwater object recognition and tracking,” IEEE Access, vol. 13, pp. 17 830–17 867, 2025.

[7] D. H. Johnson and D. E. Dudgeon, Array Signal Processing: Concepts and Techniques. Englewood Cliffs, NJ: Prentice-Hall, 1993.

[8] J. W. Goodman, “Some fundamental properties of speckle,” J. Opt. Soc. Am., vol. 66, no. 11, pp. 1145–1150, 1976.

[9] Y. Wang, X. Li, Y. Zhang, Z. Li, S. Li, and B. Shen, “Deep learning methods for sonar image despeckling: Advances and challenges,” IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens., vol. 19, pp. 17 281– 17 305, 2026.

[10] S. E. Skipetrov, J. Peuser, R. Cerbino, P. Zakharov, B. Weber, and F. Scheffold, “Noise in laser speckle correlation and imaging techniques,” Opt. Express, vol. 18, no. 14, pp. 14 519–14 534, 2010.

[11] M. A. Ainslie, Principles of Sonar Performance Modelling. Berlin, Heidelberg: Springer, 2010.

[12] J. W. Goodman, Speckle Phenomena in Optics: Theory and Applications. Englewood, CO, USA: Roberts and Company Publishers, 2007.

[13] V. S. Frost, J. A. Stiles, K. S. Shanmugan, and J. C. Holtzman, “A model for radar images and its application to adaptive digital filtering of multiplicative noise,” IEEE Trans. Pattern Anal. Mach. Intell., vol. PAMI-4, no. 2, pp. 157–166, 1982.

[14] F. Argenti, A. Lapini, T. Bianchi, and L. Alparone, “A tutorial on speckle reduction in synthetic aperture radar images,” IEEE Geosci. Remote Sens. Mag., vol. 1, no. 3, pp. 6–35, 2013.

[15] Z. Li, Z. Wang, D. Chen, T. L. Yip, and A. P. Teixeira, “RepDNet: A re-parameterization despeckling network for autonomous underwater side-scan sonar imaging with prior-knowledge customized convolution,” Def. Technol., vol. 35, pp. 259–274, 2024.

[16] J.-S. Lee, “Digital image enhancement and noise filtering by use of local statistics,” IEEE Trans. Pattern Anal. Mach. Intell., vol. PAMI-2, no. 2, pp. 165–168, 1980.

[17] D. T. Kuan, A. A. Sawchuk, T. C. Strand, and P. Chavel, “Adaptive noise smoothing filter for images with signal-dependent noise,” IEEE Trans. Pattern Anal. Mach. Intell., vol. PAMI-7, no. 2, pp. 165–177, 1985.

[18] C.-A. Deledalle, L. Denis, and F. Tupin, “Iterative weighted maximum likelihood denoising with probabilistic patch-based weights,” IEEE Trans. Image Process., vol. 18, no. 12, pp. 2661–2672, 2009.

[19] S. Parrilli, M. Poderico, C. V. Angelino, and L. Verdoliva, “A nonlocal SAR image denoising algorithm based on LLMMSE wavelet shrinkage,” IEEE Trans. Geosci. Remote Sens., vol. 50, no. 2, pp. 606–616, 2012.

[20] Y. Huang, W. Li, and F. Yuan, “Speckle noise reduction in sonar image based on adaptive redundant dictionary,” J. Mar. Sci. Eng., vol. 8, no. 10, 2020, Art. no. 761.

[21] Y. Lu, M. Yang, and R. W. Liu, “DSPNet: Deep learning-enabled blind

![](images/837f6432e0714c67e685005e88ce8a070c938a78c6c8dfac1e56e63eb91867c3.jpg)

![](images/8885859d54993478d6c5a2d50d4889d845383d87e998895410f477621179c7c9.jpg)  
(g) SS-BSN  
(f) LG-BPN  
(c) SAR-BM3D

![](images/725f1b55275e66856c7f7db15189f66dea2aa7dd48f45557bbd14a5ff5acaa54.jpg)  
(e) Noise2Void  
(d) Blind2Unblind

![](images/06df671be2c6db6a94a5be3e5a27665a2870d0da7944f3af219ddb2a6cedea49.jpg)  
(h) PI-SSD  
(i) SEGSID  
(j) Proposed  
Fig. 4. Qualitative comparison on the Boat dataset. Red and blue boxes show zoomed regions around high-reflectivity structures and background-shadow transitions. The proposed method reduces speckle while preserving separable local structures and clearer object-background transitions compared with representative model-based and self-supervised methods.

reduction of speckle noise,” in Proc. 25th Int. Conf. Pattern Recognit. (ICPR). IEEE, 2021, pp. 3475–3482.

[22] H. Long, L. Shen, Z. Wang, and J. Chen, “Underwater forward-looking sonar images target detection via speckle reduction and scene prior,” IEEE Trans. Geosci. Remote Sens., vol. 61, pp. 1–13, 2023.

[23] T. Yang, T. Zhang, and Y. Yao, “SimNFND: A forward-looking sonar denoising model trained on simulated noise-free and noisy data,” Remote Sens., vol. 16, no. 15, 2024, Art. no. 2815.

[24] X. Zhou, K. Tian, Z. Zhou, B. Ning, and Y. Wang, “SID-TGAN: A transformer-based generative adversarial network for sonar image despeckling,” Remote Sens., vol. 15, no. 20, 2023, Art. no. 5072.

[25] X. Zhou, X. Pang, H. Ma, B. Ning, and Y. Wang, “A multivariate calibration framework with global-local interaction and edge-aware enhancement for sonar image despeckling,” Image Vis. Comput., vol. 173, 2026, Art. no. 106079.

[26] Z. Wang et al., “Can optical denoising clean sonar images? A benchmark and fusion approach,” 2025, arXiv:2503.01655.

[27] S. Pillai, S. S. Savner, and S. K. Sahoo, “Physics-guided self-supervised statistical residual learning for sonar despeckling with improved generalization,” IEEE Signal Process. Lett., vol. 33, pp. 2395–2399, 2026.

[28] A. B. Molini, D. Valsesia, G. Fracastoro, and E. Magli, “Speckle2Void: Deep self-supervised SAR despeckling with blind-spot convolutional

neural networks,” IEEE Trans. Geosci. Remote Sens., vol. 60, pp. 1– 17, 2022.

[29] A. Krull, T.-O. Buchholz, and F. Jug, “Noise2Void - Learning denoising from single noisy images,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2019, pp. 2129–2137.

[30] J. Batson and L. Royer, “Noise2Self: Blind denoising by selfsupervision,” in Proc. Int. Conf. Mach. Learn. (ICML). PMLR, 2019, pp. 524–533.

[31] T. Huang, S. Li, X. Jia, H. Lu, and J. Liu, “Neighbor2Neighbor: Selfsupervised denoising from single noisy images,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2021, pp. 14 781–14 790.

[32] W. Lee, S. Son, and K. M. Lee, “AP-BSN: Self-supervised denoising for real-world images via asymmetric PD and blind-spot network,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022, pp. 17 725–17 734.

[33] Z. Wang, Y. Fu, J. Liu, and Y. Zhang, “LG-BPN: Local and global blind-patch network for self-supervised real-world denoising,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023, pp. 18 156–18 165.

[34] Y.-J. Han and H.-J. Yu, “SS-BSN: Attentive blind-spot network for selfsupervised denoising with nonlocal self-similarity,” in Proc. 32nd Int. Joint Conf. Artif. Intell. (IJCAI), Aug. 2023, pp. 800–809.

(f) LG-BPN  
![](images/974f92c00607bb2144829c3cc767164bba73c20edcd6c842f3a1f40ca7327548.jpg)  
(g) SS-BSN  
(h) PI-SSD  
(i) SEGSID  
(j) Proposed  
Fig. 5. Qualitative comparison on the Underwater Structure dataset. Red and blue boxes show zoomed regions around structural boundaries and background regions. The proposed method suppresses background speckle while preserving clearer structural continuity and weak local responses compared with representative baselines.

[35] Z. Wang, J. Liu, G. Li, and H. Han, “Blind2Unblind: Self-supervised image denoising with visible blind spots,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022, pp. 2027–2036.

[36] S. Liu, J. Lu, H. Dou, J. Li, and Y. Deng, “SEGSID: A semantic-guided framework for sonar image despeckling,” IEEE Trans. Image Process., vol. 34, pp. 652–666, 2025.

[37] J. Lehtinen et al., “Noise2Noise: Learning image restoration without clean data,” in Proc. 35th Int. Conf. Mach. Learn. (ICML), ser. Proceedings of Machine Learning Research, vol. 80. PMLR, 2018, pp. 2965–2974.

[38] T. Ehret, A. Davy, J.-M. Morel, G. Facciolo, and P. Arias, “Model-blind video denoising via frame-to-frame training,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2019, pp. 11 369–11 378.

[39] E. Dalsasso, L. Denis, and F. Tupin, “SAR2SAR: A semi-supervised despeckling algorithm for SAR images,” IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens., vol. 14, pp. 4321–4329, 2021.

[40] Z. Wang, J. Hopgood, and M. Davies, “Physics-informed self-supervised despeckling for coherent imaging sonar,” in Proc. 34th Eur. Signal Process. Conf. (EUSIPCO), Bruges, Belgium, 2026, pp. 1871–1875.

[41] D. Chen, J. Tachella, and M. E. Davies, “Equivariant imaging: Learning beyond the range space,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2021, pp. 4379–4388.

[42] D. Chen, J. Tachella, and M. E. Davies, “Robust equivariant imaging: A fully unsupervised framework for learning to image from noisy and partial measurements,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022, pp. 5647–5656.

[43] M. Ali, D. Magee, and U. Dasgupta, “Signal processing overview of ultrasound systems for medical imaging,” Texas Instruments, White Paper SPRAB12, Nov. 2008, accessed: Oct. 8, 2026. [Online]. Available: https://www.ti.com/lit/pdf/sprab12

[44] N. Z. Kolev, Ed., Sonar Systems. London, U.K.: IntechOpen, Sep.

2011.

[45] B. S. Reddy and B. N. Chatterji, “An FFT-based technique for translation, rotation, and scale-invariant image registration,” IEEE Trans. Image Process., vol. 5, no. 8, pp. 1266–1271, 1996.

[46] N. Hurtos, D. Ribas, X. Cuf´ı, Y. Petillot, and J. Salvi, “Fourier-based registration for robust forward-looking sonar mosaicing in low-visibility underwater environments,” J. Field Robot., vol. 32, no. 1, pp. 123–151, 2015.

[47] V. Sechaud, J. Scanvic, Q. Barthelemy, P. Abry, and J. Tachella,´ “Equivariant splitting: Self-supervised learning from incomplete data,” in Proc. 14th Int. Conf. Learn. Represent. (ICLR), 2026.

[48] J. Tachella et al., “DeepInverse: A Python package for solving imaging inverse problems with deep learning,” J. Open Source Softw., vol. 10, no. 115, 2025, Art. no. 8923.

# Supplementary Material

Self-Supervised Multibeam Sonar Restoration via Coherent Forward Modeling

Zixin Wang, James Hopgood, Mike Davies

## S1. IMAGE REGISTRATION VALIDATION

This section supports the paired-observation assumption used in the main paper. The image registration example illustrates the alignment of scene structures before paired training and PFPE evaluation, while the frame-gap analysis examines how temporal separation reduces input-target correlation.

Registered frame pairs are used for paired training and PFPE evaluation, whereas inference uses a single input frame. Candidate pairs are aligned by phase correlation following the paired observation registration described in the main paper. Fig. 6 shows a representative Boat example with an estimated translation of $\Delta \rho _ { 1 } ~ = ~ 9$ pixels and $\Delta \rho _ { 2 } ~ = ~ 0$ pixels. The reference frame y<sub>1</sub> is shown in red and the paired frame y<sub>2</sub> is shown in green; regions that overlap after alignment appear yellow. The increased yellow overlap after registration indicates improved alignment of persistent scene structures.

![](images/53fc5132eb893b367e2e9566f0c6fedfca6ec4e0877501aeca39521485cf6dda.jpg)

![](images/6af5a1c1e0832303e89d24d005a8d75f02613acbe4b8efc0f76e41dae0a6defb.jpg)  
(a) Before registration  
(b) After registration  
Fig. 6. Representative registration check on the Boat dataset. The reference frame $y _ { 1 }$ is shown in red and the paired frame $y _ { 2 }$ is shown in green; yellow indicates spatial overlap between the two frames. Registration increases the overlap of persistent scene structures, supporting the use of the pair for selfsupervised training and PFPE evaluation.

## S2. FRAME-PAIR CORRELATION ANALYSIS

This section evaluates whether the registered frame pairs used for self-supervised training and PFPE evaluation provide a practical balance between reduced inter-frame correlation and reliable registration. Pairs that are too highly correlated may contain nearly identical speckle fluctuations, whereas pairs with too large a temporal separation may suffer from scene mismatch, viewpoint change, and registration uncertainty.

TABLE III  
EMPIRICAL ROI PEARSON CORRELATION BETWEEN REGISTERED FRAMEPAIRS AT DIFFERENT FRAME GAPS. THE REDUCED-CORRELATION RATIODENOTES THE PERCENTAGE OF VALID REGISTERED PAIRS WITH $r < 0 . 5$
<table><tr><td rowspan="2">Frame gap</td><td colspan="2">Boat</td><td colspan="2">Underwater Structure</td></tr><tr><td>Pearson r</td><td> $r < 0 . 5$ </td><td>Pearson r</td><td> $r < 0 . 5$ </td></tr><tr><td>1</td><td> $0 . 4 8 5 \pm 0 . 1 3 8$ </td><td>56.59%</td><td> $0 . 2 6 9 \pm 0 . 1 7 4$ </td><td>88.98%</td></tr><tr><td>2</td><td> $0 . 3 7 0 \pm 0 . 1 5 1$ </td><td>79.29%</td><td> $0 . 1 9 1 \pm 0 . 1 2 1$ </td><td>96.71%</td></tr><tr><td>3</td><td> $0 . 2 9 4 \pm 0 . 1 5 6$ </td><td>87.46%</td><td> $0 . 1 5 6 \pm 0 . 1 0 0$ </td><td>98.90%</td></tr><tr><td>4</td><td> $0 . 2 4 0 \pm 0 . 1 5 7$ </td><td>94.15%</td><td> $0 . 1 3 3 \pm 0 . 0 8 9$ </td><td>99.61%</td></tr><tr><td>5</td><td> $0 . 2 0 8 \pm 0 . 1 5 3$ </td><td>96.87%</td><td> $0 . 1 1 8 \pm 0 . 0 8 3$ </td><td>99.92%</td></tr><tr><td>10</td><td> $0 . 1 3 3 \pm 0 . 1 3 5$ </td><td>98.91%</td><td> $0 . 0 8 1 \pm 0 . 0 6 6$ </td><td>100.00%</td></tr></table>

a) Correlation diagnostic: For each frame gap $\ell ,$ we compute the Pearson correlation coefficient r between registered frame pairs within homogeneous ROIs. These ROIs are selected to avoid obvious edges and structural boundaries, following the same principle used for ENL evaluation. Table III reports the percentage of valid registered pairs below a practical reduced-correlation threshold of $r \ < \ 0 . 5$ as a function of frame gap. For each dataset, we select the smallest gap for which this percentage is at least 50%, subject to the displacement bounds used for registration. This rule helps avoid near-duplicate training pairs while limiting the viewpoint variation and registration uncertainty that arise at larger frame gaps. The threshold is not intended to prove statistical independence or pure speckle decorrelation, since real sonar ROIs may still contain residual texture, weak unresolved structure, and registration error.

b) Displacement and overlap diagnostics: Since reduced inter-frame correlation does not by itself guarantee reliable paired observations, we also examine geometric reliability after registration. Increasing the frame gap can reduce residual correlation, but it can also increase apparent motion, reduce the common valid image support, and introduce scene mismatch. We therefore report displacement and overlap diagnostics in addition to the ROI correlation analysis.

For each frame gap ℓ, phase correlation estimates the translational displacement $( \Delta \rho _ { 1 , \ell } , \Delta \rho _ { 2 , \ell } )$ between the input frame and its registered paired observation. We summarize the typical motion using the median absolute displacement

$$
\mathrm { M e d } _ { \Delta \rho } ( \ell ) = \left( \mathrm { m e d i a n } \left( \left| \Delta \rho _ { 1 , \ell } \right| \right) , \mathrm { m e d i a n } \left( \left| \Delta \rho _ { 2 , \ell } \right| \right) \right) ,\tag{33}
$$

which is less sensitive than the mean to occasional largemotion or imperfect-registration pairs.

We also report the percentage of candidate pairs satisfying the fixed small-displacement condition

$$
P _ { s } ( \ell ) = 1 0 0 \mathrm { P r } \left( | \Delta \rho _ { 1 , \ell } | \leq 1 0 , | \Delta \rho _ { 2 , \ell } | \leq 5 \right) ,\tag{34}
$$

where the probability is computed over candidate registered pairs. A lower $P _ { s } ( \ell )$ indicates that a larger fraction of frame pairs require substantial translation, which increases the risk of registration uncertainty and local scene mismatch.

TABLE IV  
FRAME-PAIR DISPLACEMENT AND OVERLAP DIAGNOSTICS FOR THE BOAT AND UNDERWATER STRUCTURE DATASETS. $\operatorname { M e d } _ { \Delta \rho }$ DENOTES THE MEDIAN ABSOLUTE DISPLACEMENT PAIR (median $\Delta \dot { \rho } _ { 1 } |$ , median $\Delta \rho _ { 2 } | )$ . P<sub>s</sub> DENOTES THE PERCENTAGE OF CANDIDATE PAIRS SATISFYING $| \Delta \rho _ { 1 } | \leq 1 0 ~ \mathrm { A N D }$ $| \Delta \rho _ { 2 } | \leq 5 . \ : P _ { o }$ DENOTES THE PERCENTAGE OF CANDIDATE PAIRS WITH VALID-OVERLAP RATIO $q _ { i , \ell } > 0 . 9 5$ AFTER TRANSLATION.
<table><tr><td rowspan="2">Gap</td><td colspan="3">Boat</td><td colspan="3">Underwater Structure</td></tr><tr><td> $\operatorname { M e d } _ { \Delta \rho }$ </td><td> $P _ { s } ( \% )$ </td><td> $P _ { o } ( \% )$ </td><td> $\operatorname { M e d } _ { \Delta \rho }$ </td><td> $P _ { s } ( \% )$ </td><td> $P _ { o } ( \% )$ </td></tr><tr><td>1</td><td>(3,0)</td><td>91.78</td><td>99.17</td><td>(3,0)</td><td>88.83</td><td>99.84</td></tr><tr><td>2</td><td>(5,0)</td><td>68.24</td><td>94.48</td><td>(5,0)</td><td>69.56</td><td>93.82</td></tr><tr><td>3</td><td>(8,1)</td><td>47.73</td><td>83.89</td><td>(6,0)</td><td>60.66</td><td>85.58</td></tr><tr><td>4</td><td>(10, 1)</td><td>38.89</td><td>72.35</td><td>(6, 0)</td><td>56.75</td><td>80.85</td></tr><tr><td>5</td><td>(14, 1)</td><td>30.17</td><td>64.03</td><td>(6, 0)</td><td>54.25</td><td>75.94</td></tr><tr><td>10</td><td>(28, 2)</td><td>17.00</td><td>39.37</td><td>(4, 1)</td><td>42.87</td><td>69.02</td></tr></table>

To measure how much image support remains valid after translation, we report the percentage of high-overlap pairs:

$$
P _ { o } ( \ell ) = 1 0 0 \mathrm { P r } \left( \frac { \left( H - | \Delta \rho _ { 2 , \ell } | \right) \left( W - | \Delta \rho _ { 1 , \ell } | \right) } { H W } > 0 . 9 5 \right) ,\tag{35}
$$

where H and W denote the image height and width. A lower $P _ { o } ( \ell )$ indicates that larger frame gaps more frequently produce pairs with reduced valid support after registration.

Table IV shows that frame-gap selection involves a tradeoff: larger gaps can reduce residual correlation, but they also make the paired observation geometrically less reliable. For this reason, all main cross-method comparisons use the same fixed evaluation gap, while larger-gap results are reported only as sensitivity analysis.

c) Physical Interpretation of Correlation Decay: The two datasets exhibit different empirical correlation decay rates, even under the same frame-gap protocol. This difference should not be interpreted as a pure speckle statistic, because the measured ROI correlation is affected by platform motion, imaging range, scene geometry, viewpoint change, registration residuals, and weak unresolved structure within the selected homogeneous ROIs. Among the available acquisition parameters, imaging range provides useful physical context. The Boat dataset was acquired with a maximum range of 50 m, whereas the Underwater Structure dataset used a shorter maximum range of 25.5 m. For a lateral platform displacement $\Delta s$ at range R, the angular change scales approximately as $\Delta \theta ~ \approx ~ \Delta s / R$ . Thus, for the same physical displacement, closer-range observations can undergo larger viewpoint changes, which alter the relative path lengths and phases of unresolved scatterers within a resolution cell. This can lead to stronger speckle decorrelation after registration. This physical interpretation offers one possible explanation for why the empirical correlation diagnostic is dataset dependent and supports using it only as a practical check of paired-frame suitability, rather than as proof of statistical independence.

## S3. RESULTS ON THE BOAT DATASET

This section provides the complete Boat results supporting the main-paper comparison. Table V reports the raw PFPE values, $\mathrm { P F P E _ { i m p } ^ { a g g } }$ , ENL, EPI, and $D _ { \mathrm { E P I } }$ . The additional metrics show why ENL alone is insufficient: several compared methods, such as SS-BSN and LG-BPN, increase ENL but also produce large $D _ { \mathrm { E P I } }$ , indicating that the apparent speckle reduction is associated with edge attenuation or oversmoothing rather than accurate forward prediction. In contrast, the proposed method achieves the highest $\mathrm { P F P E _ { i m p } ^ { a g g } }$ while keeping EPI close to 1, supporting improved forward predictive agreement with limited deviation in observed edge strength. Fig. 7 provides an additional qualitative example consistent with this trend.

## S4. RESULTS ON THE UNDERWATER STRUCTURE DATASET

This section provides the complete Underwater Structure results supporting the main-paper comparison. Table VI shows that the proposed method obtains the highest $\mathrm { P F P E _ { i m p } ^ { a g g } }$ while maintaining an EPI close to one. As in the Boat experiment, methods with substantially higher ENL, such as SS-BSN, also show much larger $D _ { \mathrm { E P I } }$ , indicating that stronger homogeneous-region smoothing does not necessarily imply better preservation of observed edge strength. Fig. 8 provides an additional qualitative comparison on a representative structural scene.

## S5. ABLATION AND SENSITIVITY ANALYSIS

This section reports additional Boat ablations and sensitivity analyses. Each block varies selected factors around the stated reference settings, so results are interpreted within each block rather than as main cross-method rankings.

Table VII separates component ablations from hyperparameter sensitivity analyses, covering the paired physicsconsistency loss, registration, REI transformation type, regularization weight, rotation magnitude, and number of simulated looks. We also analyze the effect of the training and evaluation frame gaps to distinguish restoration behaviour from the gapdependent scaling of aggregate PFPE improvement. Since the Boat dataset exhibits more challenging scene variation and inter-frame displacement in our experiments, we use it for these additional analyses.

## A. Ablation Analysis

The results show that the physics-consistency loss alone can improve the paired forward-prediction score, but does not produce a meaningful reflectivity reconstruction. Adding REI regularization is therefore necessary to mitigate the ill-conditioning left by the blurred forward operator. The physics-consistency-only $( \mathcal { L } _ { \mathrm { P C } } )$ variant obtains high PFPE improvement but collapses to a degenerate solution, with nearly zero ENL and an extremely large $D _ { \mathrm { E P I } }$ . This behaviour is consistent with the analysis in the main paper: the physics-consistency loss constrains $A f _ { \theta } ( y )$ in the measurement domain, but weakly determines high-frequency or weakly observed components of the latent reflectivity. The no-registration variant performs poorly in PFPE, confirming that paired supervision requires spatially aligned observations before the forward-prediction loss is meaningful.

TABLE V  
COMPLETE OUANTITATIVE COMPARISON ON THE BOAT DATASET. $\mathrm { P F P E _ { i m p } ^ { a g g } }$ IS COMPUTED FROM THE RATIO OF SUMMED ERRORS. WHICH IS RELATIVE TO THE NOISY PAIRED-FRAME BASELINE. ENL MEASURES HOMOGENEOUS-REGION SMOOTHING, WHILE EPI AND $D _ { \mathrm { E P I } }$ CHARACTERIZE EDGE-STRENGTH PRESERVATION RELATIVE TO THE OBSERVED IMAGE. HIGH ENL SHOULD THEREFORE BE INTERPRETED TOGETHER WITH PFPE AND $D _ { \mathrm { E P I } } .$ , SINCE IT CAN ALSO RESULT FROM OVERSMOOTHING.
<table><tr><td>Category</td><td>Method</td><td> $\mathrm { P F P E } \ ( \times 1 0 ^ { - 4 } ) \downarrow$ </td><td> $\mathrm { P F P E } _ { \mathrm { i m p } } ^ { \mathrm { a g g } } ~ ( \% ) ~ \uparrow$ </td><td>ENL↑</td><td> $\mathrm { E P I }  1$ </td><td> $D _ { \mathrm { E P I } } \left( \% \right) \downarrow$ </td></tr><tr><td>Baseline</td><td>Noisy</td><td> $1 . 5 9 \pm 0 . 3 6$ </td><td>0.00</td><td> $8 . 1 8 \pm 1 . 1 2$ </td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>Model-based</td><td>PPB SAR-BM3D</td><td> $1 . 7 9 \pm 0 . 2 6$   $1 . 4 0 \pm 0 . 2 9$ </td><td>-13.06 11.53</td><td> $1 2 . 7 3 \pm 2 . 7 1$   $1 2 . 1 4 \pm 2 . 4 0$ </td><td> $0 . 8 0 7 \pm 0 . 0 2 3$   $0 . 7 0 1 \pm 0 . 0 1 6$ </td><td> $1 9 . 3 2 \pm 2 . 2 8$   $2 9 . 8 7 \pm 1 . 6 0$ </td></tr><tr><td rowspan="6">Self-supervised</td><td>Noise2Void</td><td> $1 . 7 7 \pm 0 . 4 0$ </td><td>-11.59</td><td> $1 1 . 0 3 \pm 1 . 6 2$ </td><td> $0 . 7 3 4 \pm 0 . 0 1 4$ </td><td> $2 6 . 6 1 \pm 1 . 3 6$ </td></tr><tr><td>Blind2Unblind</td><td> $1 . 3 7 \pm 0 . 2 9$ </td><td></td><td> $7 . 7 4 \pm 1 . 0 3$ </td><td> $1 . 0 4 7 \pm 0 . 0 0 4$ </td><td> $4 . 6 8 \pm 0 . 3 6$ </td></tr><tr><td>LG-BPN</td><td></td><td>13.39</td><td></td><td></td><td></td></tr><tr><td></td><td> $1 . 5 7 \pm 0 . 3 1$ </td><td>1.19</td><td> $1 8 . 5 7 \pm 5 . 6 0$   $2 5 . 0 4 \pm 8 . 7 9$ </td><td> $0 . 5 4 1 \pm 0 . 0 2 3$ </td><td> $4 5 . 8 9 \pm 2 . 2 7$ </td></tr><tr><td>SS-BSN PI-SSD</td><td> $1 . 6 2 \pm 0 . 3 2$   $1 . 4 4 \pm 0 . 2 7$ </td><td>-2.24</td><td> $1 6 . 9 7 \pm 4 . 8 1$ </td><td> $0 . 4 4 1 \pm 0 . 0 3 1$   $0 . 5 6 8 \pm 0 . 0 3 2$ </td><td> $5 5 . 9 2 \pm 3 . 0 6$ </td></tr><tr><td>SEGSID</td><td> $1 . 4 6 \pm 0 . 3 1$ </td><td>9.39 8.20</td><td> $1 5 . 8 3 \pm 3 . 8 1$ </td><td> $0 . 5 9 2 \pm 0 . 0 2 2$ </td><td> $4 3 . 2 2 \pm 3 . 2 4$   $4 0 . 8 2 \pm 2 . 1 6$ </td></tr><tr><td>Proposed</td><td>Proposed</td><td> ${ \bf 1 . 3 6 \pm 0 . 2 8 }$ </td><td>14.10</td><td> $1 2 . 8 1 \pm 2 . 6 8$ </td><td> $0 . 9 9 2 \pm 0 . 0 6 2$ </td><td> $4 . 6 5 \pm 4 . 1 3$ </td></tr></table>

TABLE VI

COMPLETE QUANTITATIVE COMPARISON ON THE UNDERWATER STRUCTURE DATASET. $\mathrm { P F P E _ { i m p } ^ { a g g } }$ IS COMPUTED FROM THE RATIO OF SUMMED ERRORS,WHICH IS RELATIVE TO THE NOISY PAIRED-FRAME BASELINE. ENL MEASURES HOMOGENEOUS-REGION SMOOTHING, WHILE EPI AND $D _ { \mathrm { E P I } }$ MEASUREEDGE-STRENGTH PRESERVATION RELATIVE TO THE OBSERVED IMAGE.
<table><tr><td>Category</td><td>Method</td><td> $\mathrm { P F P E } \ ( \times 1 0 ^ { - 4 } ) \downarrow$ </td><td> $\mathrm { P F P E _ { i m p } ^ { a g g } ~ ( \% ) ~ } \uparrow$ </td><td>ENL↑</td><td>EPI → 1</td><td> $D _ { \mathrm { E P I } } \left( \% \right) \downarrow$ </td></tr><tr><td>Baseline</td><td>Noisy</td><td> $6 . 7 2 \pm 3 . 0 6$ </td><td>0.00</td><td> $3 . 3 8 \pm 0 . 3 6$ </td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>Model-based</td><td>PPB SAR-BM3D</td><td> $7 . 3 3 \pm 2 . 3 4$   $6 . 7 2 \pm 2 . 1 8$ </td><td>-9.01 -0.02</td><td> $6 . 9 8 \pm 1 . 3 6$   $4 . 7 1 \pm 0 . 5 9$ </td><td> $0 . 7 2 8 \pm 0 . 0 4 2$   $0 . 7 4 6 \pm 0 . 0 1 1$ </td><td> $2 7 . 1 7 \pm 4 . 1 6$   $2 5 . 4 2 \pm 1 . 0 5$ </td></tr><tr><td rowspan="6">Self-supervised</td><td>Noise2Void</td><td> $6 . 8 2 \pm 2 . 2 4$ </td><td>-1.49</td><td> $4 . 0 9 \pm 0 . 4 6$ </td><td> $0 . 8 2 8 \pm 0 . 0 0 6$ </td><td> $1 7 . 2 0 \pm 0 . 5 6$ </td></tr><tr><td>Blind2Unblind</td><td> $6 . 5 7 \pm 2 . 1 8$ </td><td>2.25</td><td> $3 . 3 6 \pm 0 . 3 5$ </td><td> $1 . 0 1 1 \pm 0 . 0 0 1$ </td><td> $1 . 1 1 \pm 0 . 1 3$ </td></tr><tr><td>LG-BPN</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td> $7 . 8 0 \pm 2 . 3 5$ </td><td>-16.05</td><td> $7 . 1 1 \pm 1 . 3 5$ </td><td> $0 . 8 5 8 \pm 0 . 0 2 2$ </td><td> $1 4 . 1 9 \pm 2 . 1 6$ </td></tr><tr><td>SS-BSN PI-SSD</td><td> $8 . 8 5 \pm 2 . 8 4$   $7 . 2 1 \pm 2 . 2 5$ </td><td>-31.66</td><td> $1 5 . 5 0 \pm 5 . 2 2$   $7 . 7 6 \pm 2 . 0 2$ </td><td> $0 . 2 8 8 \pm 0 . 0 3 9$   $0 . 4 8 7 \pm 0 . 0 2 2$ </td><td> $7 1 . 2 4 \pm 3 . 8 8$ </td></tr><tr><td>SEGSID</td><td> $6 . 8 6 \pm 2 . 2 2$ </td><td>-7.36</td><td> $4 . 7 4 \pm 0 . 6 6$ </td><td> $0 . 7 1 1 \pm 0 . 0 1 4$ </td><td> $5 1 . 2 9 \pm 2 . 1 9$   $2 8 . 8 6 \pm 1 . 4 4$ </td></tr><tr><td>Proposed</td><td>Proposed</td><td> ${ \bf 6 . 4 3 \pm 2 . 0 8 }$ </td><td>-2.02 4.36</td><td> $6 . 3 7 \pm 2 . 8 4$ </td><td> $0 . 9 9 0 \pm 0 . 0 7 4$ </td><td> $6 . 8 3 \pm 2 . 9 5$ </td></tr></table>

## B. Sensitivity Analysis

The sensitivity blocks in Table VII show that the proposed method is governed by a trade-off between forward prediction, speckle suppression, and structural preservation. In each block, one factor is varied while the remaining settings are kept fixed, so the trends can be interpreted within each group.

For the transformation and rotation-angle blocks, $\alpha = 0 . 0 5$ and $J = 8$ are fixed; moderate geometric perturbations give the best balance, whereas very large rotations are less compatible with the locally shift-invariant PSF approximation and the anisotropic range-azimuth structure of sonar images.

For the REI-weight block, $\textit { J } = \textit { 8 }$ and the maximum rotation angle is fixed at 20<sup>◦</sup>. Increasing α raises ENL, but overly strong regularization reduces PFPE improvement and attenuates edges, as indicated by lower EPI and larger $D _ { \mathrm { E P I } } .$ This confirms that higher homogeneous-region smoothness is not sufficient to indicate better restoration, since excessive regularization can lead to oversmoothing.

For the simulated-look block, $\alpha = 0 . 0 5$ and the maximum rotation angle is fixed at 20<sup>◦</sup>. The number of simulated looks J controls the stochasticity of the REI forward simulation. Very small J yields poorer PFPE and larger edge-strength deviation despite a higher ENL, indicating an unfavourable smoothing– preservation trade-off. As J increases, the averaged simulator approaches the conditional-mean forward model; the observed PFPE improves and ENL decreases, while $D _ { \mathrm { E P I } }$ remains similar for $J = 8$ and $J = 1 6 .$

The reported setting is therefore used as a balanced operating point across PFPE, ENL, EPI, and $D _ { \mathrm { E P I } }$ , rather than being selected by optimizing any single metric.

## C. Frame Gap Analysis

This section analyzes how the training frame gap affects the restoration behaviour of the proposed method. All variants in Table VIII use the same configuration, with $\alpha \ = \ 0 . 0 5 .$ $J \ = \ 8 ,$ , and T<sub>τ</sub> implemented as rotations with a maximum angle of 20<sup>◦</sup>. For each training gap, displacement tolerances are set to meet a 90% pair-retention target to remove infeasible registration outliers, while the model and optimization settings are held fixed.

PFPE and $\mathrm { P F P E _ { i m p } ^ { a g g } }$ are evaluated using fixed test gaps 1, 2, and 3, whereas ENL, EPI, and D<sub>EPI</sub> are reported once for each trained checkpoint because they are computed from the restored single-frame output.

The $\mathrm { P F P E _ { i m p } ^ { a g g } }$ is gap-dependent because it is normalized by the noisy paired-frame discrepancy. At a fixed test gap, identical test pairs, preprocessing, and evaluated image support yield a common noisy-pair denominator across methods; thus, aggregate normalization preserves the ranking of mean PFPE. Across test gaps, both the noisy baseline (the aggregate denominator changes with the paired-frame discrepancy at that gap) and the paired prediction error can change, so the scores are not directly comparable absolute image-quality measures. The main comparisons therefore use the common test gap 1, supported by the correlation and registration diagnostics in Section S2.

![](images/f82f1409d1943ced051d86a7e7b2e71d912fc77c9c20d65db3345364b496b2d6.jpg)  
(f) LG-BPN  
(g) SS-BSN  
(i) SEGSID  
(h) PI-SSD  
(j) Proposed  
Fig. 7. Additional qualitative comparison on the Boat dataset. Red and blue boxes indicate zoomed regions containing highreflectivity structures and background-shadow transitions. The proposed method suppresses speckle while preserving separable local structures and clearer transition boundaries. This visual behavior is consistent with its higher PFPE improvement and lower $D _ { \mathrm { E P I } }$ in Table V.

Table VIII shows this effect empirically. The noisy PFPE baseline increases from $1 . 5 9 \times 1 0 ^ { - 4 }$ at test gap 1 to $2 . 3 5 \times 1 0 ^ { - 4 }$ and $2 . 9 4 \times 1 0 ^ { - 4 }$ at test gaps 2 and 3, respectively. Under this pair-retention protocol, increasing the training gap from 1 to 2 and 3 increases ENL from 12.40 to 15.31 and 16.48, but also increases $D _ { \mathrm { E P I } }$ from 5.17% to 15.82% and 20.91%. This indicates stronger smoothing but larger deviation from the observed edge strength.

The gap-1 trained model provides the most favourable balance in this analysis. It achieves the highest $\mathrm { P F P E _ { i m p } ^ { a g g } }$ at test gaps 1 and 2 and the lowest $D _ { \mathrm { E P I } }$ . At test gap 3, the traingap-3 model yields a slightly higher aggregate improvement (24.98% versus 24.50%), but a substantially larger $D _ { \mathrm { E P I } }$ (20.91% versus 5.17%). Under this auxiliary configuration, these results support the choice of frame gap 1 for training in the main experiments.

![](images/96a318db8c4da2d6fe1f78b270b91cbc858fc68d95126f6ce50ee416016fc7c4.jpg)

![](images/899675b440bc8f86b00e1d632da4b3112607d9ab0d9f101379b5dd61c1b83351.jpg)

![](images/9f75be77dc99a54e1f7679304b1d1b7d1e1fa6ea9bc71319f2e89719cc311ca4.jpg)  
(c) SAR-BM3D

![](images/2a561cd34ee2378dea17a7a0ffc3b1f2e64b4fac2fa2f0f17f9aed30a190eb74.jpg)  
(d) Blind2Unblind

![](images/c1e94124bc894b64bd9759d44681c1e453ea9af8a1c62facc62dee5a000cf398.jpg)  
(e) Noise2Void

(f) LG-BPN  
(g) SS-BSN  
![](images/9dae1fdf4da4271ddc2e01a143d8110b79ac2e4e466f661df6c289cbef4f3e93.jpg)  
(h) PI-SSD

![](images/a3e33b13457516715a1df6af19b315db34fc34902cc8eccd19e2c481ff125738.jpg)  
(i) SEGSID

![](images/f2b4ab83f419fac68189632eac023cb057c24c110db7e67a6ac726c4fa417ca2.jpg)  
(j) Proposed  
Fig. 8. Additional qualitative comparison on the Underwater Structure dataset. The zoomed regions highlight structura boundaries, weak internal reflectors, and background speckle. The proposed method suppresses speckle while preserving clearer structural continuity than representative baselines, consistent with its lower PFPE and relatively small $D _ { \mathrm { E P I } }$ in Table VI.

TABLE VII  
ABLATION AND SENSITIVITY ANALYSIS ON THE BOAT DATASET $\mathrm { P F P E _ { i m p } ^ { a g g } }$ IS COMPUTED FROM THE RATIO OF SUMMED ERRORS. ENL MEASURES HOMOGENEOUS-REGION SMOOTHING, WHILE $D _ { \mathrm { E P I } }$ MEASURES MEAN PER-IMAGE DEVIATION FROM UNITY IN RELATIVE EDGE STRENGTH. IN EACH BLOCK, ONE DESIGN FACTOR IS VARIED WHILE THE REMAINING SETTINGS ARE KEPT FIXED WHENEVER POSSIBLE. THE RESULTS ARE INTERPRETED JOINTLY IN TERMS OF PAIRED-FRAME FORWARD PREDICTION, SPECKLE SUPPRESSION, AND STRUCTURAL PRESERVATION.
<table><tr><td>Variant</td><td> $\mathrm { P F P E } \ ( \times 1 0 ^ { - 4 } ) \downarrow$ </td><td> $\mathrm { P F P E _ { i m p } ^ { a g g } ~ ( \% ) ~ } \uparrow$ </td><td>ENL ↑</td><td>EPI → 1</td><td> $D _ { \mathrm { E P I } } \left( \% \right) \downarrow$ </td></tr><tr><td colspan="6">Loss and registration</td></tr><tr><td>LPC-only</td><td> $1 . 2 4 \pm 0 . 2 9$ </td><td>22.02</td><td> $0 . 0 2 \pm 0 . 0 0$ </td><td> $2 3 . 0 9 3 \pm 1 . 4 1 9$ </td><td> $2 2 0 9 . 3 5 \pm 1 4 1 . 8 9$ </td></tr><tr><td>No registration</td><td> $1 . 6 2 \pm 0 . 2 9$ </td><td>-2.32</td><td> $1 4 . 8 8 \pm 3 . 8 0 $ </td><td> $0 . 5 4 9 \pm 0 . 0 2 6$ </td><td> $4 5 . 0 9 \pm 2 . 5 9$ </td></tr><tr><td colspan="6">Transformation</td></tr><tr><td>Scale</td><td> $1 . 3 8 \pm 0 . 2 8$ </td><td>13.07</td><td> $1 2 . 7 4 \pm 2 . 5 3$ </td><td> $0 . 9 5 6 \pm 0 . 0 5 4$ </td><td> $5 . 6 9 \pm 3 . 9 6$ </td></tr><tr><td>Shift</td><td> $1 . 4 0 \pm 0 . 3 0$ </td><td>11.97</td><td> $1 4 . 2 9 \pm 3 . 4 7$ </td><td> $0 . 9 2 6 \pm 0 . 0 7 6$ </td><td> $8 . 7 9 \pm 5 . 9 5$ </td></tr><tr><td>CPAB</td><td> $1 . 4 1 \pm 0 . 2 8$ </td><td>10.83</td><td> $1 3 . 5 8 \pm 3 . 2 3$ </td><td> $1 . 0 1 3 \pm 0 . 0 5 0$ </td><td> $3 . 9 2 \pm 3 . 3 3$ </td></tr><tr><td>Rot. + scale</td><td> $1 . 4 2 \pm 0 . 2 9$ </td><td>10.54</td><td> $1 4 . 7 7 \pm 3 . 2 4$ </td><td> $0 . 9 2 6 \pm 0 . 0 6 8$ </td><td> $8 . 2 3 \pm 5 . 7 5$ </td></tr><tr><td>Rot. + shift</td><td> $1 . 4 9 \pm 0 . 3 1$ </td><td>6.10</td><td> $1 5 . 1 9 \pm 3 . 9 8$ </td><td> $0 . 7 7 8 \pm 0 . 0 5 5$ </td><td> $2 2 . 2 3 \pm 5 . 5 0$ </td></tr><tr><td colspan="6">Rotation angle</td></tr><tr><td> $1 0 ^ { \circ }$ </td><td> $1 . 3 7 \pm 0 . 2 9$ </td><td>13.88</td><td> $1 1 . 3 5 \pm 2 . 1 2$ </td><td> $0 . 9 4 4 \pm 0 . 0 4 3$ </td><td> $6 . 1 8 \pm 3 . 4 6$ </td></tr><tr><td> $2 0 ^ { \circ }$ </td><td> $1 . 3 7 \pm 0 . 3 0$ </td><td>13.37</td><td> $1 2 . 4 0 \pm 2 . 7 3$ </td><td> $0 . 9 6 8 \pm 0 . 0 5 8$ </td><td> $5 . 1 7 \pm 4 . 0 4$ </td></tr><tr><td> $4 5 ^ { \circ }$ </td><td> $1 . 4 2 \pm 0 . 3 0$ </td><td>10.56</td><td> $1 0 . 8 1 \pm 2 . 0 6$ </td><td> $0 . 8 2 2 \pm 0 . 0 3 9$ </td><td> $1 7 . 7 8 \pm 3 . 8 6$ </td></tr><tr><td>90°</td><td> $1 . 4 0 \pm 0 . 3 1$ </td><td>11.86</td><td> $1 0 . 2 4 \pm 1 . 7 6$ </td><td> $0 . 8 8 7 \pm 0 . 0 4 2$ </td><td> $1 1 . 3 5 \pm 4 . 2 1$ </td></tr><tr><td colspan="6">REI weight α</td></tr><tr><td> $\alpha = 0 . 0 3$ </td><td> $1 . 3 5 \pm 0 . 2 9$ </td><td>14.97</td><td> $9 . 9 8 \pm 1 . 7 0 $ </td><td> $1 . 0 7 9 \pm 0 . 0 7 7$ </td><td> $8 . 5 6 \pm 6 . 9 7$ </td></tr><tr><td> $\alpha = 0 . 0 5$ </td><td> $1 . 3 7 \pm 0 . 3 0$ </td><td>13.37</td><td> $1 2 . 4 0 \pm 2 . 7 3$ </td><td> $0 . 9 6 8 \pm 0 . 0 5 8$ </td><td> $5 . 1 7 \pm 4 . 0 4$ </td></tr><tr><td> $\alpha = 0 . 1 0$ </td><td> $1 . 5 8 \pm 0 . 3 2$ </td><td>0.06</td><td> $1 7 . 3 7 \pm 5 . 8 1$ </td><td> $0 . 7 1 8 \pm 0 . 0 5 7$ </td><td> $2 8 . 2 2 \pm 5 . 7 1$ </td></tr><tr><td> $\alpha = 0 . 5 0$ </td><td> $1 . 6 5 \pm 0 . 3 0$ </td><td>-3.78</td><td> $2 6 . 5 7 \pm 1 1 . 3 8$ </td><td> $0 . 5 4 7 \pm 0 . 0 4 5$ </td><td> $4 5 . 3 0 \pm 4 . 5 3$ </td></tr><tr><td colspan="6">Simulated looks J</td></tr><tr><td> $J = 1$ </td><td> $1 . 8 3 \pm 0 . 3 2$ </td><td>-15.13</td><td> $2 0 . 3 2 \pm 7 . 0 9$ </td><td> $0 . 5 0 0 \pm 0 . 0 3 1$ </td><td> $4 9 . 9 8 \pm 3 . 1 4$ </td></tr><tr><td> $J = 4$ </td><td> $1 . 7 0 \pm 0 . 3 3$ </td><td>-7.22</td><td> $1 9 . 1 0 \pm 6 . 5 5$ </td><td> $0 . 7 3 5 \pm 0 . 0 5 1$ </td><td> $2 6 . 4 9 \pm 5 . 1 3$ </td></tr><tr><td> $J = 8$ </td><td> $1 . 3 7 \pm 0 . 3 0$ </td><td>13.37</td><td> $1 2 . 4 0 \pm 2 . 7 3$ </td><td> $0 . 9 6 8 \pm 0 . 0 5 8$ </td><td> $5 . 1 7 \pm 4 . 0 4$ </td></tr><tr><td> $J = 1 6$ </td><td> $1 . 3 4 \pm 0 . 2 9$ </td><td>15.35</td><td> $1 2 . 1 0 \pm 2 . 4 2$ </td><td> $1 . 0 0 2 \pm 0 . 0 6 6$ </td><td> $5 . 2 4 \pm 4 . 0 6$ </td></tr></table>

TABLE VIII  
EFFECT OF THE TRAINING FRAME GAP ON BOAT TEST PERFORMANCE. MODEL AND OPTIMIZATION SETTINGS ARE FIXED, WHILE GAP-SPECIFIC DISPLACEMENT TOLERANCES ARE SET TO MEET A COMMON 90% PAIR-RETENTION TARGET. PFPE IS REPORTED IN UNITS OF $1 0 ^ { - 4 } ,$ , AND IMP. DENOTES $\mathrm { P F P E _ { i m p } ^ { a g g } }$ IN PERCENT. PFPE AND IMP. ARE EVALUATED USING FIXED PFPE TEST GAPS 1. 2. AND 3. FOR EACH TRAINED CHECKPOINT. ENL. EPL AND $D _ { \mathrm { E P I } }$ ARE REPORTED ONCE BECAUSE THEY ARE COMPUTED FROM THE RESTORED SINGLE-FRAME OUTPUT.
<table><tr><td></td><td colspan="2">Test gap 1</td><td colspan="2">Test gap 2</td><td colspan="2">Test gap 3</td><td colspan="3"></td></tr><tr><td>Variant</td><td> $\mathrm { P F P E } \downarrow$ </td><td>Imp. ↑</td><td>PFPE↓</td><td>Imp. ↑</td><td>PFPE↓</td><td>Imp. ↑</td><td> $\mathrm { E N L } \uparrow$ </td><td> $\mathrm { E P I }  1$ </td><td> $D _ { \mathrm { E P I } } \left( \% \right) \downarrow$ </td></tr><tr><td>Noisy</td><td> $1 . 5 9 \pm 0 . 3 6$ </td><td>0.00</td><td> $2 . 3 5 \pm 0 . 5 8$ </td><td>0.00</td><td> $2 . 9 4 \pm 0 . 8 2$ </td><td>0.00</td><td> $8 . 1 8 \pm 1 . 1 2$ </td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>Train gap 1</td><td> $1 . 3 7 \pm 0 . 3 0$ </td><td>13.37</td><td> $1 . 8 2 \pm 0 . 4 8$ </td><td>22.20</td><td> $2 . 2 2 \pm 0 . 7 5$ </td><td>24.50</td><td> $1 2 . 4 0 \pm 2 . 7 3$ </td><td> $0 . 9 6 8 \pm 0 . 0 5 8$ </td><td> $5 . 1 7 \pm 4 . 0 4$ </td></tr><tr><td>Train gap 2</td><td> $1 . 5 2 \pm 0 . 3 0$ </td><td>4.04</td><td> $1 . 9 1 \pm 0 . 4 6$ </td><td>18.58</td><td> $2 . 2 7 \pm 0 . 7 1$ </td><td>22.60</td><td> $1 5 . 3 1 \pm 4 . 6 7$ </td><td> $0 . 8 4 2 \pm 0 . 0 6 6$ </td><td> $1 5 . 8 2 \pm 6 . 6 0$ </td></tr><tr><td>Train gap 3</td><td> $1 . 5 5 \pm 0 . 2 6$ </td><td>2.30</td><td> $1 . 8 9 \pm 0 . 3 8$ </td><td>19.45</td><td> $2 . 2 0 \pm 0 . 5 9$ </td><td>24.98</td><td> $1 6 . 4 8 \pm 3 . 9 0$ </td><td> $0 . 7 9 1 \pm 0 . 0 5 8$ </td><td> $2 0 . 9 1 \pm 5 . 7 9$ </td></tr></table>