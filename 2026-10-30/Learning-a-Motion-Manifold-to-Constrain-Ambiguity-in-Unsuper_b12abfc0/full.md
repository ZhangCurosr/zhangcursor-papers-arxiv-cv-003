# Learning a Motion Manifold to Constrain Ambiguity in Unsupervised 4D Medical Image Interpolation

Haojin Li, Huazhu Fu, Heng Li, and Jiang Liu

Abstract—Unsupervised 4D medical image interpolation reconstructs intermediate anatomical states from temporally sparse observations without intermediate supervision. Distinct motion fields may attain comparable endpoint agreement while producing different intermediate anatomy, giving rise to motion ambiguity. This ambiguity is difficult to constrain when many independently adjustable motion variables must be estimated from the observed endpoints. We develop a framework that learns a motion manifold to restrict candidate motion fields while retaining sample-specific variation. Cross-Sample Spectral Basis Learning extracts recurring motion structure from endpointderived dense estimates and encodes it in a shared spectral basis. Sample-Conditioned Cayley Basis Reorientation uses predicted motion coordinates to rotate the effective motion directions for each endpoint pair without increasing the number of independent coordinates. Their composition defines a nonlinear family of complete velocity fields across endpoint pairs, accommodating anatomical and motion-state differences within a restricted representation. Integrating each predicted field supports bidirectional reconstruction at arbitrary intermediate times. Theoretical analysis establishes sufficient conditions under which endpoint observations distinguish nearby motions within the learned family. Experiments on ACDC and 4D-Lung demonstrate state-of-theart performance among the evaluated methods trained without intermediate volumes and competitive results against methods using intermediate supervision.

Index Terms—4D medical image interpolation, unsupervised learning, motion ambiguity, manifold learning, deformation modeling.

## I. INTRODUCTION

4D medical imaging captures anatomical change over time and supports functional assessment [1] and image-guided intervention [2]. Volumetric acquisitions, including dynamic CT and MRI, face trade-offs between temporal resolution, spatial fidelity, and modality-dependent constraints such as acquisition time and radiation exposure [3], [4]. Interpolating missing volumes offers a computational route to temporally denser anatomical observations without requiring equally dense acquisition [5], [6].

![](images/874c8f8460811e273a8801bd4d882c0fdf6170d854df7ab1957bc059a2725d44.jpg)  
Fig. 1. Constraining motion ambiguity with a learned motion manifold. (a) The same endpoint volumes can admit different intervening motions. (b) Learning shared motion structure restricts the candidate motions available for endpoint fitting while accommodating sample-specific variation.

Unsupervised interpolation learns without intermediatevolume targets, deriving supervision from the observed endpoints [6]. Endpoint reconstruction evaluates anatomical alignment but does not specify the intervening motion. As shown in Fig. 1, distinct motion fields may attain comparable endpoint agreement while producing different intermediate anatomy, giving rise to motion ambiguity [7], [8]. Learned correspondence [9], [10], temporal consistency [6], and generative priors [11], [12] guide estimation, yet endpoint fitting can retain many spatially distributed motion variables with only indirect evidence of how their changes should be coordinated.

Learning a motion manifold offers a representation-level route to constrain this ambiguity. It limits estimation to a family of complete motion fields described by a small set of motion coordinates [13]. These coordinates couple changes across the volume, allowing endpoint fitting to draw on recurring relationships between spatially separated regions. Statistical and learned models capture recurring anatomical motion [14], [15], alongside variation across individuals [16] and respiratory states [17]. A useful motion family should make this shared structure available as a constraint while accommodating the anatomy and motion state of each endpoint pair.

Two challenges arise in learning this family from endpoint observations. 1) Shared motion structure must be inferred without intermediate motion targets. Endpoint-derived dense estimates provide motion evidence, but each is only one possible explanation of an endpoint pair. Their recurring variation must be identified across samples to establish a shared reference. 2) Sample-specific motion must remain expressive within the learned family. The restriction should accommodate changes in anatomy and motion state without restoring a large number of independently adjustable motion variables.

We develop a framework that learns a motion manifold from endpoint observations and uses it to construct continuous deformations for interpolation. Cross-Sample Spectral Basis Learning summarizes endpoint-derived dense estimates in a shared spectral basis, providing a data-derived reference for motion synthesis. Sample-Conditioned Cayley Basis Reorientation uses motion coordinates predicted from each endpoint pair to rotate the effective directions within this representation. This coordinate-dependent reorientation supports nonlinear motion variation while keeping the number of independent coordinates fixed. The learned family constrains ambiguity by restricting the motion fields considered in endpoint fitting. Integrating the resulting velocity field supports bidirectional reconstruction at arbitrary query times. Theoretical analysis characterizes conditions under which endpoint observations distinguish nearby motion coordinates.

The contributions of this work are summarized as follows:

• We develop a framework that learns a motion manifold to constrain ambiguity in unsupervised 4D medical image interpolation while retaining sample-specific motion variation.

• We introduce Cross-Sample Spectral Basis Learning, which extracts recurring motion structure from provisional endpoint-derived estimates to establish the shared basis of the motion manifold.

• We introduce Sample-Conditioned Cayley Basis Reorientation, which enables sample-specific motion expression by varying effective motion directions at a fixed number of independent motion coordinates.

• Experiments on ACDC and 4D-Lung demonstrate stateof-the-art accuracy among the evaluated methods trained without intermediate volumes and competitive performance against methods using intermediate supervision.

## II. RELATED WORK

## A. 4D Medical Image Interpolation

Research on 4D medical image interpolation has developed motion representations and synthesis models for recovering unobserved anatomical changes [5], [6]. More broadly, joint motion estimation and image reconstruction exploit the dependence of recovered detail on correspondence quality [18], [19]. Among image interpolation methods, motion-guided approaches such as SVIN and MPVF use multi-scale representations to accommodate different motion magnitudes [5], [20]. AMT strengthens matching through all-pairs correspondence and multiple motion fields [21], while VFIFME reduces reconstruction’s dependence on imperfect motion estimates [22]. Generative modeling supplies temporal priors through deformation-aware diffusion in DDM [11], latent dynamics in LDDM [23], and Fourier guidance in FB-Diff [12]. These priors complement correspondence-based synthesis by drawing on recurring patterns of anatomical appearance and change.

The supervision used to learn motion varies across approaches. TMSDF transfers intermediate motion cues into time-modulated deformation fusion [24]. For unsupervised interpolation without intermediate training targets, UVI-Net uses cyclic endpoint reconstruction [6], while TTT4MII uses instance-specific self-supervision [25]. Temporal parameterization and spatial consistency also influence the recovered anatomical sequence [26], [27]. These designs strengthen reconstruction and temporal modeling, although endpoint agreement alone does not determine the intervening anatomical motion.

## B. Motion Ambiguity in Deformation Modeling

Motion ambiguity arises when different motion fields explain the same observed anatomy [28]. Spatial regularization in VM and TM [9], [10] and velocity integration [29], [30] guide deformation through local coherence and transformation structure. Shape-aware tracking uses anatomical boundaries to constrain correspondence [31]; physical formulations incorporate tissue mechanics [32] or signal consistency [33] to supplement image agreement. Fourier-Net+ [34] and LRVF [35] restrict motion variation through band-limited and low-rank representations, with coefficient-based respiratory models providing related constraints [2], [36]. Spatial penalties couple neighboring displacements, while compact coefficients link distant regions through shared directions. These restrictions narrow the search but do not specify every unobserved anatomical state. Recent work examines non-unique motion [7], including velocity ambiguity in VD [8]. Probabilistic motion formulations model deformation distributions under partial observations [37], [38].

A learned motion manifold restricts candidate fields through coordinates describing their coordinated variation [13]. Statistical respiratory models [14] and learned deformation representations [39], [40] extract recurring structure; manifold regularization guides estimated fields toward a learned family [41]. Related cardiac representations encode shape and temporal variation [42]–[44], with conditional neural fields relating geometry to individual characteristics [45]. Case-specific neural fields [46]–[48] and anatomy-conditioned motion modes [49] make the representation responsive to individual observations. The shared family provides a reusable constraint across cases; conditioning or case-wise fitting determines which combinations of this structure are available for each observed anatomy. For unsupervised interpolation, the challenge is to learn a shared motion manifold from endpoint-derived evidence while retaining the motion needed by each pair.

![](images/2fca44b8c56dcda752d7ef531289477cbf9c8a53bbefb040f4af0b6829b88828.jpg)  
Fig. 2. Overview of the proposed framework for unsupervised 4D medical image interpolation. (a) Endpoint-derived dense estimates are aggregated in the DCT domain to learn a shared spectral reference $( \mu , \bar { B } , \Lambda _ { \epsilon } )$ . (b) Predicted motion coordinates condition the reorientation of shared directions and their combination into a complete velocity field. (c) Velocity integration, endpoint warping, and time-weighted blending generate intermediate volumes. The shared reference is fixed at inference; dashed arrows indicate training-only paths.

## III. METHODOLOGY

## A. Problem Formulation and Overview

Let $I _ { 0 } , I _ { 1 } : \Omega \ \to \ \mathbb { R }$ denote two observed volumes on a spatial domain $ { \Omega } \subset \mathbb { R } ^ { 3 }$ with their acquisition times normalized to 0 and 1. Unsupervised interpolation estimates $\widehat { I } _ { t }$ for any $t \in ( 0 , 1 )$ without intermediate-volume supervision. We represent motion by a stationary velocity field (SVF) v, whose integrated deformations reconstruct the endpoints and synthesize intermediate volumes through backward warping, $\mathcal { W } ( I , \phi ) = I \circ \phi$ . Endpoint reconstruction supplies the training signal, but distinct velocity fields can explain the same observations and produce different intermediate anatomy.

Our framework learns a motion manifold that restricts the candidate velocity fields:

$$
\mathcal { M } = \{ F ( z ) : z \in \mathcal { Z } \subset \mathbb { R } ^ { d } \} ,\tag{1}
$$

where $z$ contains the intrinsic motion coordinates and $F$ maps them to a complete 3D velocity field. As illustrated in Fig. 2, Cross-Sample Spectral Basis Learning extracts shared motion structure from endpoint-derived estimates, and Sample-Conditioned Cayley Basis Reorientation determines how this structure is expressed for the current endpoint pair. The resulting family limits independent motion variation while accommodating sample-specific directions.

## B. Cross-Sample Spectral Basis Learning

Shared motion structure must be learned from endpoint observations, since intermediate motion is unavailable. A shared 3D encoder and an auxiliary prediction head estimate a dense SVF $v _ { \mathrm { d e n s e } }$ from each training endpoint pair. Bidirectional endpoint reconstruction and spatial regularization train this head [9], providing provisional motion estimates across different anatomical states. Across training pairs, their recurring, spatially coordinated variation supplies the evidence for learning the shared basis.

We apply an orthonormal discrete cosine transform (DCT) [50] to each velocity component and retain a lowfrequency block, giving $a _ { \mathrm { d e n s e } } ~ = ~ \mathcal { C } ( v _ { \mathrm { d e n s e } } ) ~ \in ~ \mathbb { R } ^ { M }$ , where $M \ = \ 3 K _ { D } K _ { H } K _ { W }$ and $( K _ { D } , K _ { H } , K _ { W } )$ are the retained block dimensions. The transpose operator $\mathcal { C } ^ { \top }$ performs the inverse DCT (iDCT) after zero-padding the omitted coefficients. Within this spectral representation, an ordered basis $B = [ b _ { 1 } , \dots , b _ { R } ] \in \bar { \mathbb { R } } ^ { M \times R }$ , with $B ^ { \top } B = I _ { R }$ and $R < M$ is learned online from centered coefficients. For their buffered covariance Σb, we use the update direction:

$$
\Delta B = \widehat { \Sigma } B - B \mathrm { t r i u } ( B ^ { \top } \widehat { \Sigma } B ) ,\tag{2}
$$

where triu(·) retains the upper-triangular entries. The step along $\Delta B$ is normalized by the centered buffered coefficients mean squared norm and followed by QR orthonormalization. This ordered covariance update [51] accumulates dominant variation across endpoint pairs while the dense estimates evolve during training.

Let $\mu$ denote the running coefficient mean, and let the positive diagonal matrix $\Lambda _ { \epsilon }$ contain the running mean squared centered projections along the basis directions, floored at ϵ for numerical normalization. Using the shared spectral reference $( \mu , B , \Lambda _ { \epsilon } )$ , we project the dense estimate into motion coordinates and reconstruct its velocity field as follows:

$$
\begin{array} { r } { c _ { \mathrm { d e n s e } } = \Lambda _ { \epsilon } ^ { - 1 / 2 } B ^ { \top } ( a _ { \mathrm { d e n s e } } - \mu ) , } \\ { v _ { \mathrm { p r o j } } = \mathcal { C } ^ { \top } ( \mu + B \Lambda _ { \epsilon } ^ { 1 / 2 } c _ { \mathrm { d e n s e } } ) . } \end{array}\tag{3}
$$

The motion coordinate $c _ { \mathrm { d e n s e } } \in \mathbb { R } ^ { R }$ and its reconstructed field $v _ { \mathrm { p r o j } }$ provide coordinate- and velocity-level targets for the sample-specific representation. Each learned direction can combine retained frequencies and velocity components, so a change in one motion coordinate describes coordinated variation in a complete field. These targets express the endpointderived motion evidence through the shared basis, linking the motion needed by an endpoint pair to patterns supported across training cases.

## C. Sample-Conditioned Cayley Basis Reorientation

The shared reference constrains the available motion patterns, while each endpoint pair determines their samplespecific expression. We construct d intrinsic motion coordinates from the shared encoder features $h \colon$

$$
z = [ p _ { \theta } ( h ) + c _ { \mathrm { m o t i o n } } ] _ { 1 : d } \in \mathbb { R } ^ { d } ,\tag{4}
$$

where $p _ { \theta } ( h ) \in \mathbb { R } ^ { R }$ is a learned coordinate prediction, $[ \cdot ] _ { 1 : d }$ truncates the summed coordinates to the leading $d \ < \ R$ components, and $c _ { \mathrm { m o t i o n } } ~ \in ~ \mathbb { R } ^ { R }$ supplements the prediction with a motion-derived cue. This cue applies the same centered, energy-normalized spectral projection to an upsampled coarse velocity proxy $v _ { \mathrm { p r o x y } }$ produced within the coordinate predictor.

To let the same number of coordinates express different motion directions across samples, we embed z into $\mathbb { R } ^ { R }$ using a fixed matrix $E \in \mathbb { R } ^ { R \times d }$ that appends $R { - } d$ zeros. An MLP $G _ { \psi }$ takes z as input and predicts two factors $U ( z ) , V ( z ) \in \mathbb { R } ^ { R \times s }$ which define the skew-symmetric generator:

$$
K ( z ) = \frac { U ( z ) V ( z ) ^ { \top } - V ( z ) U ( z ) ^ { \top } } { \sqrt { s R } } .\tag{5}
$$

A bounded scalar gate $\alpha = g _ { \mathrm { m a x } } \operatorname { t a n h } ( \rho )$ scales the reorientation generator. With $S ( z ) = \alpha K ( z ) / 2$ , the Cayley map in Fig. 3 gives:

$$
Q ( z ) = ( I _ { R } - S ( z ) ) ^ { - 1 } ( I _ { R } + S ( z ) ) .\tag{6}
$$

Skew-symmetry makes $Q ( z )$ orthogonal, preserving the length of the normalized motion coordinates [52]. The factorization restricts its departure from the identity to rank at most $2 s \mathrm { . }$ . Initializing the learnable scalar $\rho$ at zero starts the model with the unrotated embedding, from which sample-conditioned reorientation is learned.

The reoriented motion coordinate is:

$$
c ( z ) = Q ( z ) E z .\tag{7}
$$

Although $c ( z )$ can use all $R$ components, each remains determined by the same $d$ intrinsic coordinates. Its combination with the shared basis yields the motion map:

$$
F ( z ) = \mathcal { C } ^ { \top } \Big ( \mu + B \Lambda _ { \epsilon } ^ { 1 / 2 } c ( z ) \Big ) .\tag{8}
$$

The columns of $B \Lambda _ { \epsilon } ^ { 1 / 2 } Q ( z ) E$ specify the effective spectral directions for velocity synthesis. A fixed embedding would use the same d-dimensional span for every pair. Reorientation allows different pairs to draw on different combinations of the R shared directions. The same d coordinates determine both their combination and orientation, allowing the motion family to accommodate individual anatomy and motion state without additional independently adjustable per-pair variables.

## D. Theoretical Analysis of Motion Ambiguity

Locally, the benefit of restricting motion to the learned manifold depends on how its coordinates affect endpoint reconstruction. For scalar volumes with N voxels, let $g ( z ) \in$ $\mathbb { R } ^ { 2 N }$ concatenate the residuals $\mathcal { W } ( I _ { 0 } , \exp ( F ( z ) ) ) - I _ { 1 }$ and $\mathcal { W } ( I _ { 1 } , \exp ( - F ( z ) ) ) - I _ { 0 }$ , where W denotes image warping and exp integrates an SVF. Its coordinate sensitivity at $z _ { \star }$ is $H _ { \star } = D g ( z _ { \star } )$

![](images/2935617784e3402192b451763de94333eead8a26ad5b704490a622e206196c73.jpg)  
Fig. 3. Conditional reorientation of motion coordinates. The motion coordinates z generate a Cayley rotation $Q ( z )$ that acts on their embedding $E z .$ This changes the effective motion directions for each pair while preserving coordinate length and the number of independent coordinates.

Proposition 1 (Local identifiability of motion coordinates). Assume that F and the endpoint-warping residual are continuously differentiable locally. The paired residual of an unrestricted $v \in  { \mathbb { R } } ^ { 3 N }$ has at least N first-order null directions. If H has full column rank $d ,$ then z is locally identifiable from the paired residual and $F$ is locally an immersion. If ${ \cal D } g$ is also L-Lipschitz, with $L > 0$ , and $\kappa _ { \star } = \sigma _ { \operatorname* { m i n } } ( H _ { \star } ) > 0$ , the following bound holds:

$$
\| g ( z _ { \star } + \delta z ) - g ( z _ { \star } ) \| _ { 2 } \geq \frac { \kappa _ { \star } } { 2 } \| \delta z \| _ { 2 } ,\tag{9}
$$

for $\| \delta z \| _ { 2 } \leq \kappa _ { \star } / L$ whose connecting segment stays within the neighborhood.

Proof. The dense residual Jacobian has $2 N$ rows and 3N columns, so rank–nullity gives a nullity of at least N. Full column rank of $H _ { \star }$ provides a nonsingular d×d row minor; the inverse-function theorem applied to those residual components gives local injectivity. The chain rule also implies that $D F ( z _ { \star } )$ has rank d. Finally, the first-order remainder is bounded by $L \| \delta z \| _ { 2 } ^ { 2 } / 2$ . Subtracting it from $\lVert H _ { \star } \delta z \rVert _ { 2 } \geq \kappa _ { \star } \lVert \delta z \rVert _ { 2 }$ yields the bound.

The result has two implications. 1) Freely varying dense fields contain motion changes to which endpoint reconstruction is insensitive at first order. 2) Within the learned family, full column rank means that every local coordinate direction affects the endpoint residual. The bound quantifies this sensitivity: sufficiently small coordinate changes produce a proportionate residual change. Restricting the motion family can therefore make nearby motions distinguishable when its retained coordinates are observable from the endpoints.

## E. Motion Reconstruction and Learning

For a velocity field $u , \mathcal { L } _ { \mathrm { r e c } } ( u )$ sums bidirectional endpoint reconstruction penalties combining Charbonnier intensity discrepancy and local normalized cross-correlation [6]. The dense objective $\mathcal { L } _ { \mathrm { d e n s e } }$ adds mean squared spatial velocity gradients with weight $\lambda _ { r }$ to $\mathcal { L } _ { \mathrm { r e c } } ( v _ { \mathrm { d e n s e } } )$ , while ${ \mathcal { L } } _ { \mathrm { m a n } } = { \mathcal { L } } _ { \mathrm { r e c } } ( v )$ trains the manifold prediction $v = F ( z )$ . Projection, velocity, and coordinate consistency couple these predictions to the shared reference:

TABLE I  
QUANTITATIVE COMPARISON ON ACDC AND 4D-LUNG. RESULTS ARE MEAN ± STANDARD DEVIATION OVER FIVE TRAINING RUNS. SUP. INDICATES INTERMEDIATE-VOLUME SUPERVISION. MODEL SIZE (#P) IS ALSO REPORTED. BEST RECONSTRUCTION SCORES ARE IN BOLD.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Venue</td><td rowspan="2"> $\operatorname { S u p . }$ </td><td colspan="4"></td><td colspan="6">4D-Lung</td><td rowspan="2">#P (M)</td></tr><tr><td> $\mathrm { P S N R } _ { \mathrm { ~ d B ~ } } ^ { \uparrow }$ </td><td></td><td> $\mathbf { N M } _ { \times 1 0 ^ { - 2 } } ^ { \uparrow }$ </td><td> $\mathrm { S S I M \Sigma _ { \times 1 0 ^ { - 2 } } ^ { \uparrow } }$ </td><td></td><td> $\mathrm { P S N R } _ { \mathrm { ~ d B ~ } } ^ { \uparrow }$ </td><td> $\mathbf { N M } _ { \times 1 0 ^ { - 2 } } ^ { \uparrow }$ </td><td></td><td> $\mathbf { S S I M } _ { \times 1 0 ^ { - 2 } } ^ { \uparrow }$ </td><td></td></tr><tr><td>VM [9]</td><td>TMI&#x27;19</td><td>x</td><td>25.694</td><td>±0.074</td><td>53.543 ±0.367</td><td> $9 4 . 6 7 8 \ \pm 0 . 1 0 4$ </td><td>26.077</td><td>±0.066</td><td>43.982</td><td>±0.355</td><td>86.055</td><td>±0.292</td><td>10.044</td></tr><tr><td>TM [10]</td><td>MedIA&#x27;22</td><td>x</td><td>27.125</td><td>±0.200</td><td>64.628 ±0.671</td><td> $9 6 . 6 4 4 \pm 0 . 1 2 3$ </td><td>26.683</td><td>±0.051</td><td>48.945</td><td>±0.104</td><td>84.779</td><td>±0.125</td><td>10.804</td></tr><tr><td>SVIN [5]</td><td>CVPR&#x27;20</td><td>x</td><td>28.434</td><td>±0.443</td><td>60.178 ±0.335</td><td> $9 7 . 3 5 2 \pm 0 . 1 5 2$ </td><td>27.312</td><td>±0.280</td><td>43.171</td><td>±0.392</td><td>85.442</td><td>±0.286</td><td>12.393</td></tr><tr><td>UVI-Net [6]</td><td>CVPR&#x27;24</td><td>x</td><td>31.903</td><td>±0.118</td><td>72.252 ±0.166</td><td> $9 8 . 6 1 8 \scriptstyle \pm 0 . 0 2 7$ </td><td>28.948</td><td>±0.277</td><td>55.157</td><td>±0.841</td><td>88.750</td><td>±0.668</td><td>14.273</td></tr><tr><td>TTT4MII [25]</td><td>IJCNN&#x27;25</td><td>x</td><td>32.136</td><td>±0.140</td><td>72.514 ±0.174</td><td> $9 8 . 6 7 1 \scriptstyle \pm 0 . 0 3 1$ </td><td>28.322</td><td>±0.529</td><td>50.590</td><td>±1.563</td><td>83.495</td><td>±1.280</td><td>12.006</td></tr><tr><td>Fourier-Net+ [34]</td><td>TNNLS&#x27;26</td><td>x</td><td>28.815</td><td>±0.072</td><td>65.098 ±0.039</td><td> $9 7 . 2 9 0 \pm 0 . 0 2 4$ </td><td></td><td> $2 6 . 7 8 2 \scriptstyle \pm 0 . 1 9 1$ </td><td>49.032</td><td>±0.194</td><td></td><td>84.814 ±0.424</td><td>12.414</td></tr><tr><td>LRVF [35]</td><td>MICCAI&#x27;26</td><td>x</td><td>31.961</td><td>±0.168</td><td>72.610 ±0.138</td><td> $9 8 . 6 2 3 \scriptstyle \pm 0 . 0 2 8$ </td><td></td><td> $2 9 . 8 2 3 \ \pm 0 . 1 0 6$ </td><td>62.337</td><td>±0.205</td><td>91.286</td><td>±0.126</td><td>33.917</td></tr><tr><td>MPVF [20]</td><td>JBHI&#x27;23</td><td>√</td><td>32.144</td><td>±0.011</td><td>70.658 ±0.038</td><td> $9 8 . 6 6 9 \scriptstyle \pm 0 . 0 0 3$ </td><td>29.909</td><td>±0.035</td><td></td><td>53.074 ±0.389</td><td>90.180</td><td>) ±0.053</td><td>14.533</td></tr><tr><td>LDDM [23]</td><td>MICCAI&#x27;24</td><td></td><td>31.839</td><td>±0.033</td><td>72.333 ±0.302</td><td> $9 8 . 5 5 1 \scriptstyle \pm 0 . 0 1 3$ </td><td>27.563</td><td>±0.111</td><td></td><td>56.577 ±8.077</td><td>83.499</td><td>±12.092</td><td>15.545</td></tr><tr><td>AMT [21]</td><td>CVPR&#x27;23</td><td></td><td>31.009</td><td>±0.267</td><td>71.461 ±0.438</td><td> $9 8 . 3 8 9 \scriptstyle \pm 0 . 0 7 7$ </td><td></td><td>28.881 ±0.773</td><td></td><td>62.101 ±1.080</td><td></td><td>87.954 ±1.723</td><td>23.336</td></tr><tr><td>VFIFME [22]</td><td>CVPR&#x27;23</td><td></td><td>29.627</td><td>±3.297</td><td>66.215 ±8.075</td><td> $9 7 . 2 5 0 \pm 4 . 6 2 4$ </td><td></td><td> $2 9 . 7 1 1 \pm 1 . 1 8 3$ </td><td>50.843 ±4.208</td><td></td><td>89.677 ±1.887</td><td></td><td>24.089</td></tr><tr><td>FB-Diff [12]</td><td>ICCV’25</td><td>√</td><td>29.387</td><td>±0.103</td><td>62.135 ±0.274</td><td> $9 7 . 4 8 3 \pm 0 . 1 6 3$ </td><td></td><td>27.617 ±0.867</td><td>49.275</td><td>±1.345</td><td>80.913</td><td>±3.712</td><td>33.527</td></tr><tr><td>DDM [11]</td><td>MICCAI&#x27;22</td><td>√</td><td></td><td> $2 8 . 5 5 8 \scriptstyle \pm 1 . 6 7 9$ </td><td> $6 4 . 1 4 2 \pm 1 . 3 4 9$ </td><td> $9 7 . 2 3 7 \pm 0 . 8 7 2$ </td><td></td><td> $2 6 . 8 9 6 \pm 0 . 1 0 1$ </td><td></td><td>42.544 ±0.894</td><td>85.229</td><td>±0.270</td><td>23.178</td></tr><tr><td>TMSDF [24]</td><td>MICCAI&#x27;25</td><td>√</td><td></td><td> $3 1 . 9 7 4 \scriptstyle \pm 0 . 2 2 5$ </td><td> $6 6 . 4 0 1 \scriptstyle \pm 0 . 8 3 3$ </td><td> $9 8 . 5 5 8 \scriptstyle \pm 0 . 0 7 2$ </td><td></td><td> $2 9 . 8 4 8 \ \pm 0 . 0 2 3$ </td><td></td><td> $4 8 . 3 6 0 \ \pm 0 . 1 1 3$ </td><td>90.289</td><td>9 ±1.086</td><td>42.283</td></tr><tr><td>VD [8]</td><td>TPAMI&#x27;26</td><td>√</td><td></td><td> $3 2 . 1 0 7 ~ \pm 0 . 9 6 2$ </td><td> $7 2 . 1 3 6 \pm 2 . 6 1 2$ </td><td> $9 8 . 6 8 0 \scriptstyle \pm 0 . 3 6 8$ </td><td></td><td> $2 8 . 9 8 2 \scriptstyle \pm 0 . 8 7 7$ </td><td></td><td> $6 0 . 6 5 7 \scriptstyle \pm 1 . 4 4 3$ </td><td></td><td> $9 0 . 1 5 0 \pm 1 . 1 3 5$ </td><td>23.336</td></tr><tr><td>Ours</td><td></td><td>x</td><td></td><td> $\mathbf { 3 2 . 2 0 5 \_ } _ { \pm 0 . 0 0 4 }$ </td><td> $7 2 . 7 3 4 \scriptstyle \pm 0 . 0 0 1$ </td><td> $\mathbf { 9 8 . 6 8 5 \ } _ { \pm 0 . 0 0 1 }$ </td><td></td><td> $2 9 . 8 7 4 \scriptstyle \pm 0 . 0 1 7$ </td><td></td><td> ${ \bf 6 2 . 4 4 2 _ { \Omega \pm 0 . 0 6 8 } }$ </td><td> $\mathbf { 9 1 . 3 0 2 \ } _ { \pm 0 . 0 2 8 }$ </td><td></td><td>15.100</td></tr></table>

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { p r o j } } = \| v _ { \mathrm { d e n s e } } - v _ { \mathrm { p r o j } } \| _ { 1 } , ~ } \\ { \mathcal { L } _ { \mathrm { v e l } } = \| v - v _ { \mathrm { p r o j } } \| _ { 1 } , \quad \mathcal { L } _ { \mathrm { c o o r d } } = \| c ( z ) - c _ { \mathrm { d e n s e } } \| _ { 1 } , } \end{array}\tag{10}
$$

where $\lVert \cdot \rVert _ { 1 }$ denotes mean absolute discrepancy. The full objective is:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { d e n s e } } + \mathcal { L } _ { \mathrm { m a n } } + \lambda _ { p } \mathcal { L } _ { \mathrm { p r o j } } + \lambda _ { v } \mathcal { L } _ { \mathrm { v e l } } + \lambda _ { c } \mathcal { L } _ { \mathrm { c o o r d } } .\tag{11}
$$

The projection term keeps the dense estimates compatible with the shared basis. Velocity and coordinate consistency use the projected field and coordinates as fixed targets to guide sample-specific prediction at field and coordinate levels, respectively.

Inference. With the basis and running statistics fixed, the coordinate predictor estimates one $\mathrm { S V F } \ v = F ( z )$ for the endpoint pair from shared encoder features, without the auxiliary dense head. For query time t, scaling-and-squaring [30] gives $\phi _ { t } ~ = \exp ( t v )$ and $\bar { \phi } _ { t } = \exp ( - ( 1 - t ) v )$ , and bidirectional fusion yields:

$$
\widehat { I } _ { t } = ( 1 - t ) \mathcal { W } ( I _ { 0 } , \phi _ { t } ) + t \mathcal { W } ( I _ { 1 } , \bar { \phi } _ { t } ) .\tag{12}
$$

## IV. EXPERIMENTS

## A. Experimental Settings

Datasets. We evaluate on ACDC [53] and 4D-Lung [54]. ACDC contains 150 cardiac MRI cases, divided into 80/20/50 cases for training/validation/testing. End-diastolic and endsystolic volumes serve as endpoints, and volumes are resized to $1 6 0 \times 1 6 0 \times 1 6$ . Our processed 4D-Lung subset contains 500 respiratory samples, split at the patient level into 306/84/110 samples for training/validation/testing. The 0% and 50% respiratory phases serve as endpoints, with volumes resized to $1 2 8 \times 1 2 8 \times 3 2$ . All volumes are histogram-equalized and linearly normalized to [0, 1].

Evaluation Protocol. Fifteen baselines cover deformationbased modeling, motion-guided interpolation, and generative synthesis under common data partitions, preprocessing, and test volumes. Table I identifies intermediate-volume supervision. Our training losses use only endpoints; intervening volumes are reserved for validation-based model selection and evaluation. Reconstruction is evaluated at recorded acquisition times using PSNR, NMI, and SSIM [55], with scores aggregated over test volumes and standard deviations based on five training runs.

Implementation Details. The PyTorch model uses a shared four-stage 3D convolutional encoder with channel widths $4 0 / 8 0 / 1 6 0 / 3 2 0 .$ an auxiliary dense head, and a motioncoordinate predictor, and is trained on NVIDIA RTX A6000 GPUs. The retained DCT block is $8 \times 3 2 \times 3 2 .$ , with $R = 8 0 .$ $d \ = \ 4 0 .$ , Cayley factor width $s \ = \ 4 ,$ and gate bound $g _ { \mathrm { m a x } } = 0 . 2 5$ . We set $\lambda _ { r } = 0 . 0 1 , \lambda _ { p } = 1 0 , \lambda _ { v } = 1$ , and $\lambda _ { c } = 5 ,$ with relative weights 10 : 1 for the Charbonnier and correlation penalties. AdamW uses learning rates of $1 0 ^ { - 3 }$ for the dense branch and $3 \times 1 0 ^ { - 4 }$ for the remaining components, with cosine annealing. Training runs for 500 epochs on ACDC and 200 on 4D-Lung, using batch size 1 and gradient accumulation over eight iterations. Velocity integration uses seven scalingand-squaring steps.

## B. Quantitative Comparison

Table I contrasts methods with different sources of motion guidance. Cyclic reconstruction in UVI-Net [6] and instancespecific refinement in TTT4MII [25] substantially improve ACDC reconstruction over VM, TM, and Fourier-Net+. Their less uniform gains on 4D-Lung, particularly across NMI and SSIM, underline the difficulty of inferring intervening anatomy from endpoint supervision alone. Our method maintains strong performance across all three criteria on both datasets. Cardiac contraction and respiratory deformation produce different patterns of image change; consistent results in both settings support a data-derived motion constraint across these distinct anatomical dynamics. Agreement across intensity, statistical dependence, and structural similarity measures also provides complementary evidence of reconstruction fidelity.

![](images/92821a10884142a451363b401d32ffe92c61ae1b42e2b2007be12f7b0ce3f41a.jpg)  
Fig. 4. Qualitative comparison on ACDC and 4D-Lung. Enlarged regions highlight the ventricular blood-pool boundary and internal low-signal structures in ACDC, and fine elongated intrapulmonary structures in 4D-Lung.

(a) Held-out spectral energy retention  
![](images/7ebbb62660e69d332821dbc14a2a7bf0b246f1e8fa64af027c412c813cf3b09f.jpg)

(b) Reconstruction versus spectral rank  
![](images/abdc63abf7ba0b770623d3c0808b3408eebc93f85b69431e52d6a026d068623f.jpg)

(c) Reconstruction after spectral projection  
![](images/80b5b6e9ef82463988b3e20fa1345c02faf37da5198e195069ab632f605176ef.jpg)  
Fig. 5. Shared-basis analyses on ACDC. (a) Retained energy fraction for learned, PCA, frequency-ordered DCT, and random directions. (b) Reconstruction versus R at $d = 4 0$ , with and without projection consistency P. (c) Change in intermediate-volume MSE after projecting the dense field, plotted against its original MSE; positive ∆MSE favors projection.

Intermediate-supervised methods provide a complementary comparison. Multi-scale deformation fusion in MPVF and TMSDF [20], [24], multiple motion fields in AMT [21], and distance-indexed guidance in VD [8] enrich the motion information used for synthesis. Their strengths differ across criteria: on 4D-Lung, MPVF achieves high PSNR and AMT high NMI, while their SSIM is less competitive. Intermediate supervision enriches the training cues, but reconstruction still depends on how these cues organize the estimated motion. Our method exceeds most of these approaches and remains close to the strongest results. Its near-leading respiratory PSNR is accompanied by strong NMI and SSIM, showing that similar overall intensity error can coexist with different levels of structural agreement. Achieving this balance without intermediate training targets supports the shared motion constraint as useful

interpolation guidance.

## C. Qualitative Comparison

As shown in Fig. 4, the methods recover broadly similar anatomy, but differ in how local boundaries and fine structures are reconstructed. In ACDC, our result retains the indentations along the ventricular blood-pool boundary and the separation of internal low-signal structures, which become smoother or less distinct in several predictions. On 4D-Lung, our reconstruction preserves narrow intrapulmonary profiles close to the reference. These differences reflect the spatial arrangement of local structures as well as their sharpness. A plausible-looking boundary may still have the wrong extent, while averaging misaligned structures can broaden or merge a narrow profile. The enlarged regions show that the learned motion family preserves this localized variation alongside the overall anatomical change.

## D. Analysis of Spectral Basis Learning

Spectral energy retention. Fig. 5(a) compares retained energy fractions, $\| B ^ { \top } ( a _ { \mathrm { d e n s e } } - \mu ) \| _ { 2 } ^ { 2 } / \| a _ { \mathrm { d e n s e } } - \mu \| _ { 2 } ^ { 2 }$ , averaged over held-out cases [56]. PCA is fitted to training dense-motion coefficients; frequency-ordered DCT and random orthonormal directions provide fixed alternatives in the same DCT block. The learned directions approach PCA and substantially exceed these alternatives at matched R. Their alignment with recurring motion variation uses the shared direction budget more effectively than frequency order alone.

(a) Reorientation across intrinsic dimensions  
![](images/781090102eff62c844c1ed00ce3bd74ee1dca91ae8f5d418b798a96d35fe6288.jpg)

(b) Global versus sample-conditioned reorientation  
![](images/16722b1678e056b9e1d0e6112121eda50178a45b00a7d246524851b0620d8672.jpg)

(c) Performance across reorientation mappings  
![](images/5f1fee9c8a00f1cf4f2d25d996627d298b1ca4a7b061d4f2a9795280afe5d316.jpg)  
Fig. 6. Reorientation analyses on ACDC. (a) Fixed embedding versus sample-conditioned reorientation at $R = 8 0 .$ . (b) Global versus sample-conditioned reorientation across low, medium, and high image-change groups at d = 40. The α = 0 column gives absolute PSNR; remaining columns show gains over this reference. (c) Cayley, matrix-exponential, QR-based, and normalized-MLP mappings, with PSNR expressed as deviations from each case’s mean across the four mappings.

Reconstruction versus spectral rank. Increasing R at fixed $d \ = \ 4 0$ first improves reconstruction and then reduces $\mathbf { i t } ,$ with a peak near $R ~ = ~ 8 0$ in Fig. 5(b). Here P denotes projection consistency $\mathcal { L } _ { \mathrm { p r o j } }$ . Since d is unchanged, this comparison varies the breadth of the shared reference while preserving the number of coordinates assigned to each pair. The trend distinguishes coverage of dense variation from useful interpolation capacity: a wider set of available directions does not necessarily improve the intermediate anatomy. The consistently higher scores with $P$ further indicate the value of keeping the evolving dense evidence aligned with the shared representation.

Reconstruction after spectral projection. For each endpoint pair, we synthesize intermediate volumes using the dense field and its projection through the same integration and blending procedure. Each point in Fig. 5(c) reports ∆MSE = $\mathrm { M S E _ { d e n s e } - M S E _ { p r o j } }$ , averaged over that case’s intermediate volumes. The small differences and predominance of positive values show that projection largely preserves reconstructionrelevant information through velocity integration and image synthesis. Energy retention measures coverage of dense variation, while this reconstruction-level check evaluates restriction after integration and warping. Positive differences also occur for cases with larger dense reconstruction errors. The selected directions retain variation that remains useful through the nonlinear synthesis process, helping explain why projected fields preserve or improve intermediate reconstruction.

## E. Analysis of Sample-Conditioned Reorientation

Reorientation across intrinsic dimensions. Fixed uses the unrotated embedding $c = E z$ , while Active uses $c = Q ( z ) E z$ with other settings held fixed. Active improves reconstruction throughout the tested range of d in Fig. 6(a), and its performance levels off near $d = 4 0 . \mathrm { ~ A ~ }$ smaller Active coordinate set can also match or exceed larger Fixed configurations. The advantage reflects how effectively the shared directions are used as well as the number of independent motion variables. The fixed embedding gives every pair the same motion directions, even though the anatomy and motion state differ. Sample-conditioned reorientation preserves useful individual variation within a restricted coordinate budget, reducing the need to recover expression solely by enlarging that budget.

Global versus sample-conditioned reorientation. Global learns one orientation shared by all cases, whereas Sample predicts the orientation from each pair’s motion coordinates; other settings are held fixed. Cases are ranked by mean absolute foreground intensity change between sampled reference volumes and divided into low, medium, and high thirds. In Fig. 6(b), Sample yields larger gains at matched generator strengths α, with both variants returning to the unrotated embedding at $\alpha = 0$ . The consistent advantage favors adapting the shared directions to each endpoint pair. Even the lowchange group benefits, indicating that sample dependence is useful beyond cases with large image changes. Moderate generator strengths give the largest gains, balancing directional flexibility with controlled departures from the unrotated shared reference.

Alternative reorientation mappings. We compare Cayley with matrix-exponential and QR-based orthogonal mappings, and with an MLP that directly predicts length-normalized coordinates. To remove between-case difficulty offsets, each PSNR is centered by subtracting that case’s mean across the four mappings. The distributions in Fig. 6(c) place Cayley close to the other orthogonal constructions and above the normalized MLP. The MLP’s lower scores favor structured directional mappings over directly predicting a length-matched coordinate vector. The comparable performance of the orthogonal variants supports structured directional adaptation as the central benefit of reorientation. Cayley provides a competitive realization of this flexibility with norm preservation and a lowrank-controlled departure from the shared directions.

## F. Additional Experiments

Loss configuration. Table II compares combinations of $\mathcal { L } _ { \mathrm { p r o j } } .$ ${ \mathcal { L } } _ { \mathrm { v e l } } ,$ and $\mathcal { L } _ { \mathrm { c o o r d } }$ . Projection consistency improves the base configuration, while coupling it with the sample-specific consistency terms yields further gains. Combining these terms brings the shared structure into both dense motion estimation and sample-specific prediction. Velocity consistency ties the guidance to the field used for image synthesis, while coordinate consistency aligns the compact representation. Their joint advantage suggests that constraining the final field and organizing its predicted coordinates play complementary roles in interpolation.

TABLE II  
LOSS ABLATION ON ACDC.
<table><tr><td> $\mathcal { L } _ { \mathrm { p r o j } }$ </td><td> $\mathcal { L } _ { \mathrm { v e l } }$ </td><td> $\mathcal { L } _ { \mathrm { c o o r d } }$ </td><td>PSNR  $^ { \uparrow } _ { \mathrm { d B } }$ </td><td>NMI  $\stackrel { \uparrow } { \times } 1 0 ^ { - 2 }$ </td><td>SSIM  $\stackrel { \uparrow } { \times } 1 0 ^ { - 2 }$ </td></tr><tr><td rowspan="3">√</td><td rowspan="3">√</td><td></td><td>30.855</td><td>66.625</td><td>98.295</td></tr><tr><td></td><td>31.520</td><td>68.754</td><td>98.507</td></tr><tr><td>√</td><td>31.907</td><td>72.416</td><td>98.614</td></tr><tr><td>√</td><td></td><td>√</td><td>32.030</td><td>72.672</td><td>98.645</td></tr><tr><td>√</td><td>√</td><td></td><td>32.045</td><td>72.666</td><td>98.648</td></tr><tr><td>√</td><td>√</td><td>√</td><td>32.205</td><td>72.734</td><td>98.685</td></tr></table>

TABLE III

MEAN PAIRWISE DISPLACEMENT DIFFERENCE ACROSS TRAINING RUNS ON ACDC AT $t = 0 . 5$ (voxel, ↓).
<table><tr><td></td><td>Ours</td><td>MPVF</td><td>Fourier-Net+</td><td>DDM</td></tr><tr><td>Difference</td><td>0.03989</td><td>0.04572</td><td>0.14694</td><td>0.23164</td></tr></table>

Motion estimation consistency. For each endpoint pair, we compare the backward displacement used to warp $I _ { 0 }$ to the midpoint across training runs, averaging pairwise Euclidean differences over voxels and cases. This evaluates reproducibility of the recovered correspondence for the same observations, complementing tests of reconstructed appearance. Our method has low disagreement, comparable to MPVF and substantially below Fourier-Net+ and DDM in Table III. Endpoint image agreement can conceal different underlying correspondences. Small field differences together with accurate reconstruction support a stable choice of motion explanation, consistent with the intended restriction of motion ambiguity.

Temporal reconstruction fidelity. As shown in Fig. 7(a), our reconstructions follow the evolving ventricular contours and blood-pool appearance, while local boundaries and internal contrast are less faithfully recovered by LDDM and Fourier-Net+. This comparison tests whether detail preserved at one query also changes appropriately at other times, including the relative positions of local boundaries as the ventricle deforms. A shared velocity representation links these reconstructions, allowing the observed anatomy to evolve coherently across intermediate queries.

Relative change error measures the mean absolute discrepancy between predicted and reference adjacent-volume change rates, normalized by each case’s time-averaged absolute reference-change rate. Rates use the actual time separation between volumes, with spatial averaging over anatomy identified by intensity variation in the reference sequence. Our method closely tracks MPVF and remains below LDDM and Fourier-Net+ across the intervals in Fig. 7(b), and below DDM in four of five intervals. These results complement framewise similarity: the preserved ventricular detail is accompanied by competitive agreement with reference change rates across the sampled intervals, linking anatomical appearance to the progression of image change over time.

(a) Anatomical evolution across time  
![](images/2b3404c226eece69d88c583afd11917df28aff527741038f64892c8bae913654.jpg)

(b) Normalized MAE: predicted vs. GT frame changes  
![](images/175302243857da9d0aa6d451489e30c51ee3490fec137d84bd1a24f610af487e.jpg)  
Fig. 7. Temporal reconstruction on ACDC. (a) Ventricular anatomy and enlarged details across intermediate query times. (b) Relative change error across normalized time intervals, evaluating agreement with the reference sequence’s image evolution. Lower values indicate more faithful temporal reconstruction.

(a) Segmentation consistency on ACDC  
![](images/55f7a952ab4d5c262c14ef5ff6de087e998ac2e796c65dc4f866127f1f53bfa3.jpg)

(b) Segmentation visualization  
![](images/c1e049f42907dd7bed864f0c66bf0abe5a83eb657cb557acc166e4fa51d1ff8c.jpg)  
Fig. 8. Segmentation consistency on ACDC. (a) Dice agreement between masks predicted from reconstructed and reference volumes, averaged over test volumes including endpoints. RV, MYO, and LV denote the right ventricular cavity, left ventricular myocardium, and left ventricular cavity. (b) Reference and reconstructed images with their model-predicted masks.

Segmentation consistency. A standard 3D U-Net [57] trained on ACDC training cases is frozen and applied to both reconstructed and reference volumes; Dice measures agreement between its predicted masks. Our method approaches MPVF and exceeds Fourier-Net+ and DDM across the three structures in Fig. 8(a). Its advantage over LDDM in MYO and LV indicates that maintaining blood-pool appearance alone does not ensure consistent myocardial extent. Cavity and myocardial masks depend on neighboring boundaries, so a local contour shift can alter their relative extent even when the overall appearance changes little. Agreement across these regions supports preservation of their relative configuration during interpolation. The overlays in Fig. 8(b) connect reconstruction fidelity with consistent structural delineation.

COMPUTATIONAL COST ON ACDC. QPS COUNTS COMPLETE 3D MIDPOINT QUERIES. MPVF TRAINING TIME INCLUDES ALL TRAINING STAGES.
<table><tr><td>Method</td><td colspan="2">Est. train time (s/epoch)↓</td></tr><tr><td></td><td>#P (M)</td><td>QPS ↑ 40.52</td></tr><tr><td>Ours MPVF</td><td>15.100 14.536</td><td>19.18 105.14</td></tr><tr><td>DDM</td><td>23.164</td><td>4.61 18.42 14.04</td></tr><tr><td>LDDM</td><td>15.653</td><td>51.52 2.72</td></tr><tr><td></td><td></td><td>121.16</td></tr><tr><td>Fourier-Net+</td><td>12.414</td><td>1.84</td></tr></table>

Computational cost. Our method provides higher query throughput than MPVF, DDM, and LDDM, with estimated training time close to DDM, as shown in Table IV. Training uses the dense head to build motion evidence; query-time synthesis uses the shared reference and predicted motion coordinates without this auxiliary head. Fourier-Net+ is faster, but the reconstruction comparisons favor our method. The resulting trade-off places the framework between fast band-limited synthesis and more expensive motion-guided or generative alternatives, retaining sample-specific motion expression at a moderate synthesis cost and making that flexibility practical for repeated queries in temporally dense reconstruction.

## V. CONCLUSION

We presented a framework that learns a motion manifold to constrain ambiguity in unsupervised 4D medical image interpolation. Cross-sample spectral basis learning captures recurring motion structure from endpoint-derived estimates, while sample-conditioned Cayley reorientation varies the effective motion directions without increasing the number of independent motion coordinates. The resulting representation accommodates sample-specific variation within a restricted motion family. Experiments on ACDC and 4D-Lung demonstrate state-of-the-art performance among the evaluated methods trained without intermediate volumes and competitive results against methods using intermediate supervision. Module analyses, temporal evaluation, and motion-estimation consistency link these reconstruction results to effective use of the shared directions and reproducible motion estimates. Learning a shared motion family while adapting its expression to individual observations provides a useful representation-level constraint on ambiguity in endpoint-supervised interpolation.

[1] Q. Meng, W. Bai, D. P. O’Regan, and D. Rueckert, “Deepmesh: meshbased cardiac motion tracking using deep learning,” IEEE transactions on medical imaging, vol. 43, no. 4, pp. 1489–1500, 2024.

[2] N. R. Huttinga, T. Bruijnen, C. A. Van Den Berg, and A. Sbrizzi, “Real-time non-rigid 3d respiratory motion estimation for mr-guided radiotherapy using mr-motus,” IEEE Transactions on Medical Imaging, vol. 41, no. 2, pp. 332–346, 2022.

[3] G. Reiter, C. Reiter, I. Ovcina, M. Fuchsjager, and U. Reiter, “Four-¨ dimensional flow mri for a dynamic perspective on the heart and adjacent great vessels,” Radiology, vol. 316, no. 2, p. e242972, 2025.

[4] S. Demehri, F. I. Baffour, J. G. Klein, E. Ghotbi, H. A. Ibad, K. Moradi, K. Taguchi, J. Fritz, J. A. Carrino, A. Guermazi, E. K. Fishman, and W. B. Zbijewski, “Musculoskeletal ct imaging: State-of-the-art advancements and future directions,” Radiology, vol. 308, no. 2, p. e230344, 2023.

[5] Y. Guo, L. Bi, E. Ahn, D. Feng, Q. Wang, and J. Kim, “A spatiotemporal volumetric interpolation network for 4d dynamic medical image,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 4726–4735.

[6] J. Kim, H. Yoon, G. Park, K. Kim, and E. Yang, “Data-efficient unsupervised interpolation without any intermediate frame for 4d medical images,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 11 353–11 364.

[7] K. Zhou, W. Li, X. Han, and J. Lu, “Exploring motion ambiguity and alignment for high-quality video frame interpolation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 22 169–22 179.

[8] Z. Zhong, Y. Zhang, W. Wang, X. Sun, Y. Qiao, G. Krishnan, S. Ma, and J. Wang, “Velocity disambiguation for video frame interpolation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 48, no. 7, pp. 7938–7953, 2026.

[9] G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag, and A. V. Dalca, “Voxelmorph: a learning framework for deformable medical image registration,” IEEE transactions on medical imaging, vol. 38, no. 8, pp. 1788–1800, 2019.

[10] J. Chen, E. C. Frey, Y. He, W. P. Segars, Y. Li, and Y. Du, “Transmorph: Transformer for unsupervised medical image registration,” Medical image analysis, vol. 82, p. 102615, 2022.

[11] B. Kim and J. C. Ye, “Diffusion deformable model for 4d temporal medical image generation,” in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2022, pp. 539–548.

[12] X. You, R. Yang, C. Zhang, Z. Jiang, J. Yang, and N. Navab, “Fb-diff: Fourier basis-guided diffusion for temporal interpolation of 4d medical imaging,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 28 010–28 020.

[13] M. Meila and H. Zhang, “Manifold learning: What, how, and why,”˘ Annual Review of Statistics and Its Application, vol. 11, pp. 393–417, 2024.

[14] J. Ehrhardt, R. Werner, A. Schmidt-Richberg, and H. Handels, “Statistical modeling of 4d respiratory lung motion using diffeomorphic image registration,” IEEE Transactions on Medical Imaging, vol. 30, no. 2, pp. 251–265, 2011.

[15] M. Qiao, S. Wang, H. Qiu, A. de Marvao, D. P. O’Regan, D. Rueckert, and W. Bai, “Cheart: A conditional spatio-temporal generative model for cardiac anatomy,” IEEE Transactions on Medical Imaging, vol. 43, no. 3, pp. 1259–1269, 2024.

[16] M. Qiao, K. A. McGurk, S. Wang, P. M. Matthews, D. P. O’Regan, and W. Bai, “A personalized time-resolved 3D mesh generative model for unveiling normal heart dynamics,” Nature Machine Intelligence, vol. 7, pp. 800–811, 2025.

[17] Y. Huang, B. Eiben, K. Thielemans, and J. R. McClelland, “Resolving variable respiratory motion from unsorted 4D computed tomography,” in Medical Image Computing and Computer Assisted Intervention – MICCAI 2024, ser. Lecture Notes in Computer Science, vol. 15001. Springer Nature Switzerland, 2024, pp. 588–597.

[18] S. Zhi, Y. Wang, H. Xiao, T. Bai, B. Li, Y. Tang, C. Liu, W. Li, T. Li, H. Ge et al., “Coarse–super-resolution–fine network (cosf-net): a unified end-to-end neural network for 4d-mri with simultaneous motion estimation and super-resolution,” IEEE transactions on medical imaging, vol. 43, no. 1, pp. 162–174, 2024.

[19] J. Pan, W. Huang, D. Rueckert, T. Kustner, and K. Hammernik, “Motion-¨ compensated MR CINE reconstruction with reconstruction-driven motion estimation,” IEEE Transactions on Medical Imaging, vol. 43, no. 7, pp. 2420–2433, 2024.

[20] T.-T. Wei, C. Kuo, Y.-C. Tseng, and J.-J. Chen, “Mpvf: 4d medical image inpainting by multi-pyramid voxel flows,” IEEE Journal of Biomedical and Health Informatics, vol. 27, no. 12, pp. 5872–5882, 2023.

[21] Z. Li, Z.-L. Zhu, L.-H. Han, Q. Hou, C.-L. Guo, and M.-M. Cheng, “Amt: All-pairs multi-field transforms for efficient frame interpolation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 9801–9810.

[22] Z. Yu, Y. Zhang, D. Zou, X. Chen, J. S. Ren, and S. Ren, “Rangenullspace video frame interpolation with focalized motion estimation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 22 159–22 168.

[23] T. Chen, Y. Shi, Z. Zheng, B. Yan, J. Hu, X. X. Zhu, and L. Mou, “Ultrasound image-to-video synthesis via latent dynamic diffusion models,” in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2024, pp. 764–774.

[24] J. Zhang, D. Ai, Z. Gan, T. Fu, J. Fan, H. Song, D. Xiao, and J. Yang, “Temporal modulated multi-scale deformation fusion via knowledge distillation for 4d medical image interpolation,” in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2025, pp. 551–561.

[25] Q. Zhang, Y. Lei, Z. Zheng, Z. Chen, and Z. Xie, “Test time training for 4d medical image interpolation,” in 2025 International Joint Conference on Neural Networks (IJCNN). IEEE, 2025, pp. 1–8.

[26] H. Li, H. Wang, Z. Ma, M. Ou, H. Li, and J. Liu, “Phase-aligned finitefourier periodic deformation for 4D medical image interpolation,” in Proceedings of the 34th ACM International Conference on Multimedia, 2026.

[27] X. Zhou, J. Liu, S. Yu, H. Yang, C. Li, T. Tan, and S. Wang, “A diffusiondriven temporal super-resolution and spatial consistency enhancement framework for 4d mri imaging,” in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2025, pp. 3–12.

[28] A. Reithmeir, V. Spieker, V. Sideri-Lampretsa, D. Rueckert, J. A. Schnabel, and V. A. Zimmer, “From model based to learned regularization in medical image registration: A comprehensive review,” Medical Image Analysis, vol. 108, p. 103854, 2026.

[29] M. F. Beg, M. I. Miller, A. Trouve, and L. Younes, “Computing large´ deformation metric mappings via geodesic flows of diffeomorphisms,” International journal of computer vision, vol. 61, no. 2, pp. 139–157, 2005.

[30] T. Vercauteren, X. Pennec, A. Perchant, and N. Ayache, “Diffeomorphic demons: Efficient non-parametric image registration,” NeuroImage, vol. 45, no. 1, pp. S61–S72, 2009.

[31] Q. Meng, C. Qin, W. Bai, T. Liu, A. de Marvao, D. P. O’Regan, and D. Rueckert, “MulViMotion: Shape-aware 3D myocardial motion tracking from multi-view cardiac MRI,” IEEE Transactions on Medical Imaging, vol. 41, no. 8, pp. 1961–1974, 2022.

[32] P. Arratia Lopez, H. Mella, S. Uribe, D. E. Hurtado, and F. Sahli Costa-´ bal, “WarpPINN: Cine-MR image registration with physics-informed neural networks,” Medical Image Analysis, vol. 89, p. 102925, 2023.

[33] E. Hanania, A. Zehavi-Lenz, I. Volovik, D. Link-Sourani, I. Cohen, and M. Freiman, “MBSS-T1: Model-based subject-specific self-supervised motion correction for robust cardiac T1 mapping,” Medical Image Analysis, vol. 102, p. 103495, 2025.

[34] X. Jia, A. Thorley, A. Gomez, W. Lu, D. Kotecha, A. F. Frangi, and J. Duan, “Fourier-net+: Band-limited spatial representation for efficient medical image registration,” IEEE Transactions on Neural Networks and Learning Systems, pp. 1–15, 2026.

[35] H. Li, H. Wang, C. Liu, Z. Ma, H. Li, and J. Liu, “Low-rank velocity fields as a structural prior for unsupervised 4d medical image interpolation,” in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2026, pp. 238–248.

[36] Y. Huang, I. Singh, T. Joyce, K. Thielemans, and J. R. McClelland, “DIGS: Dynamic CBCT reconstruction using deformation-informed 4D gaussian splatting and a low-rank free-form deformation model,” in Medical Image Computing and Computer Assisted Intervention – MICCAI 2025, ser. Lecture Notes in Computer Science, vol. 15963. Springer Nature Switzerland, 2025, pp. 131–141.

[37] S. Thibeault, L. Vazquez Romaguera, and S. Kadoury, “Conditional 4D motion diffusion models with masked observations to forecast deformations,” in Medical Image Computing and Computer Assisted Intervention – MICCAI 2024, ser. Lecture Notes in Computer Science, vol. 15006. Springer Nature Switzerland, 2024, pp. 89–98.

[38] N. R. F. Huttinga, T. Bruijnen, C. A. T. van den Berg, and A. Sbrizzi, “Gaussian processes for real-time 3D motion and uncertainty estimation during MR-guided radiotherapy,” Medical Image Analysis, vol. 88, p. 102843, 2023.

[39] J. Krebs, H. Delingette, B. Mailhe, N. Ayache, and T. Mansi, “Learning´ a probabilistic model for diffeomorphic registration,” IEEE Transactions on Medical Imaging, vol. 38, no. 9, pp. 2165–2176, 2019.

[40] T. Mezheritsky, L. Vazquez Romaguera, W. L. Le, and S. Kadoury,´ “Population-based 3D respiratory motion modelling from convolutional autoencoders for 2D ultrasound-guided radiotherapy,” Medical Image Analysis, vol. 75, p. 102260, 2022.

[41] R. Bhalodia, S. Y. Elhabian, L. Kavan, and R. T. Whitaker, “A cooperative autoencoder for population-based regularization of CNN image registration,” in International Conference on Medical Image Computing and Computer-Assisted Intervention, ser. Lecture Notes in Computer Science, vol. 11765. Springer, 2019, pp. 391–400.

[42] M.-M. Rohe, M. Sermesant, and X. Pennec, “Low-dimensional represen-´ tation of cardiac motion using barycentric subspaces: A new group-wise paradigm for estimation, analysis, and reconstruction,” Medical Image Analysis, vol. 45, pp. 1–12, 2018.

[43] H. Dou, J. Huang, A. Zakeri, Z. Zhou, T. Mu, J. Duan, and A. F. Frangi, “4D CardioSynth: Synthesising dynamic virtual heart populations through spatiotemporal disentanglement,” in Medical Image Computing and Computer Assisted Intervention – MICCAI 2025, ser. Lecture Notes in Computer Science, vol. 15962. Springer Nature Switzerland, 2025, pp. 3–12.

[44] M. Qiao, J. Zheng, W. Zhang, Q. Ma, L. Li, B. Kainz, D. P. O’Regan, P. M. Matthews, S. Niederer, and W. Bai, “Mesh4D: A motion-aware multi-view variational autoencoder for 3D+t mesh reconstruction,” in Medical Image Computing and Computer Assisted Intervention – MIC-CAI 2025, ser. Lecture Notes in Computer Science, vol. 15975. Springer Nature Switzerland, 2025, pp. 343–353.

[45] K. Sørensen, P. Diez, J. Margeta, Y. El Youssef, M. Pham, J. J. Pedersen, T. Kuhl, O. de Backer, K. Kofoed, O. Camara, and R. Paulsen, “Spatio-¨ temporal neural distance fields for conditional generative modeling of the heart,” in Medical Image Computing and Computer Assisted Intervention – MICCAI 2024, ser. Lecture Notes in Computer Science, vol. 15003. Springer Nature Switzerland, 2024, pp. 422–432.

[46] S. Sun, K. Han, C. You, H. Tang, D. Kong, J. Naushad, X. Yan, H. Ma, P. Khosravi, J. S. Duncan et al., “Medical image registration via neural fields,” Medical Image Analysis, vol. 97, p. 103249, 2024.

[47] C. Shen, H. Zhu, Y. Zhou, Y. Liu, S. Yi, L. Dong, W. Zhao, D. J. Brady, X. Cao, Z. Ma et al., “Continuous 3d myocardial motion tracking via echocardiography,” IEEE Transactions on Medical Imaging, vol. 43, no. 12, pp. 4236–4252, 2024.

[48] J. B. Gebauer, L. E. Buttgen, T. Sentker, and R. Werner, “Non-¨ linear INR-based motion modeling for 4D radiotherapy,” in Medical Image Computing and Computer Assisted Intervention – MICCAI 2026. Springer Nature Switzerland, 2026.

[49] Y. Zhu, M. Chen, Q. Wang, X. Gu, and W. Lu, “Learning what can move: Patient-specific motion subspaces from a single scan,” in Medical Image Computing and Computer Assisted Intervention – MICCAI 2026. Springer Nature Switzerland, 2026.

[50] Y. Wang and D. Mukherjee, “The discrete cosine transform and its impact on visual compression: Fifty years from its invention,” IEEE Signal Processing Magazine, vol. 40, no. 6, pp. 14–17, 2023.

[51] T. D. Sanger, “Optimal unsupervised learning in a single-layer linear feedforward neural network,” Neural Networks, vol. 2, no. 6, pp. 459– 473, 1989.

[52] K. Helfrich, D. Willmott, and Q. Ye, “Orthogonal recurrent neural networks with scaled Cayley transform,” in Proceedings of the 35th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 80. PMLR, 2018, pp. 1969–1978.

[53] O. Bernard, A. Lalande, C. Zotti, F. Cervenansky, X. Yang, P.-A. Heng, I. Cetin, K. Lekadir, O. Camara, M. A. G. Ballester et al., “Deep learning techniques for automatic mri cardiac multi-structures segmentation and diagnosis: is the problem solved?” IEEE transactions on medical imaging, vol. 37, no. 11, pp. 2514–2525, 2018.

[54] G. D. Hugo, E. Weiss, W. C. Sleeman, S. Balik, P. J. Keall, J. Lu, and J. F. Williamson, “Data from 4D lung imaging of NSCLC patients,” The Cancer Imaging Archive, 2016, data set.

[55] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: From error visibility to structural similarity,” IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600–612, 2004.

[56] N. Halko, P.-G. Martinsson, and J. A. Tropp, “Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions,” SIAM Review, vol. 53, no. 2, pp. 217–288, 2011.

[57] O. C¸ ic¸ek, A. Abdulkadir, S. S. Lienkamp, T. Brox, and O. Ronneberger,<sup>¨</sup> “3D U-Net: Learning dense volumetric segmentation from sparse annotation,” in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2016, pp. 424–432.