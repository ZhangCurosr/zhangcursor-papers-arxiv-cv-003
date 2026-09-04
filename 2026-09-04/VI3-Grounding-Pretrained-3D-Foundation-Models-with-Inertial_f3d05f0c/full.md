# VI3: Grounding Pretrained 3D Foundation Models with Inertial Cues

Ernesto Lozano Alberto Jaenal Javier Civera

I3A, Universidad de Zaragoza, Spain {e.lozano, ajaenal, jcivera}@unizar.es

![](images/af6cb590c52ddbb0bfea84090a9889559062e8951f7a38392d1256f25f9f0a4f.jpg)  
Figure 1. VI3 grounds the predictions of a pretrained 3D foundation model (trajectory and scene reconstruction shown in red) using inertial data from an onboard IMU (inertial trajectory shown in blue) as GT-free supervision. This leads to a physics-aware, scene-specific trajectory refinement and metric scale recovery, that keeps geometry consistent as seen in the right figure (VI3 results shown in green).

## Abstract

3Dfoundation models (3DFMs) excel at predicting camera poses and dense depth from multiple views of a scene, showcasing strong zero-shot generalization. However, as metric scale is not observablefrom monocular images, their absolute scale predictions are typically inaccurate. Inertial measurement units (IMUs), present in most devices, naturally complement monocular cameras by observing scaled motion. We introduce VI3, a model-agnostic framework that metrically anchors a pretrained 3DFM using only IMU readings. VI3 initializes and preintegrates the IMU to obtain a metric motion reference, which is then used to recover the scale of the 3DFM outputs. Our method includes adaptable anchoring strategies tailored to diverse 3DFM architectures. Experiments on synthetic and real aerial datasets demonstrate that VI3 recovers metric scale without ground-truth supervision while preserving geometric consistency, acting as a fine refinement under well-conditioned motion and as a strong prior when motion is less informative. https://ernestolozano.github.io/vi3

## 1. Introduction

Estimating 3D scene attributes from unstructured images is a cornerstone of computer vision, with applications in robotics [30] and neural rendering [44] among others. For decades, the prevailing paradigm has relied on featurebased pipelines, e.g. Structure-from-Motion (SfM) [1, 36, 41] or visual SLAM [20, 33, 34]. Despite their trajectory accuracy, these methods remain brittle in unconstrained environments [37, 40], yielding sparse point clouds rather than the dense geometry required for modern spatial AI [2, 44]. Recently, 3D foundation models (3DFMs) have reshaped this landscape. Trained on large, diverse image collections, models such as DUSt3R [50], VGGT [47], π³ [52], and a growing list of successors [21, 24, 27, 38, 48] learn strong geometry and motion priors, predicting camera poses and dense per-pixel geometry from unposed images in a single feed-forward pass.

Nevertheless, 3DFMs process images independently through monocular backbones [35]. While this enables them to preserve highly accurate relative motion and scene structure, their reconstructions remain defined only up to an unknown scale factor and are therefore decoupled from real-world metric dimensions. This limitation derives from their training: 3DFMs learn under scale-invariant supervision objectives (usually distilled from SfM pipelines [41] rather than metric signals), making them fundamentally incapable of determining the metric scale of the scene. This remains an open gap: turning a scale-free prediction into a physically anchored, metric reconstruction. Recent work have attempted to enforce metric scale by conditioning inference using camera priors [16, 30] or through dedicated tokens [21]. However, their strong data-driven priors may override explicit metric signals, yielding reconstructions that fail to reflect the real-world dimensions. Crucially, such methods still depend on precise camera calibration (often unavailable in unconstrained settings) or upstream metric poses to produce physically-anchored reconstructions.

This gap can be fully addressed by aligning vision with physical measurements from a sensor ubiquitous across most platforms: an inertial measurement unit (IMU). By providing metric readings, the IMU renders real-world scale observable, with a decade of visual-inertial research [6, 12, 13] sustaining its sufficiency to metrically ground vision.

We introduce a model-agnostic Visual Inertial grounding for 3DFMs (VI3) that enables metric 3DFM reconstruction by just using an IMU; entirely bypassing optimization backends, maps, or external finetuning corpora. Given a short clip, the IMU is initialized leveraging the outputs of the 3DFM (without requiring any GT or external signal) and then preintegrated into a physics-consistent, gravity-aware motion reference that anchors the 3DFM to the real-world. Due to the diversity of 3DFM models, we propose different ways of injecting the metric scale, such as input conditioning (for compatible modes) or test-time refinement [15], thereby scaling the predicted geometry without explicit 3D supervision.

In summary, our contributions are threefold. (1) To the best of our knowledge, VI3 is the first method to bring together at inference time the dense 3DFM priors and the metric grounding of inertial sensing with the aim of physically anchoring any model, only requiring an on-board IMU. (2) We propose an IMU intialization method that uses the 3DFM and the inertial readings to estimate the IMU state. (3) We present a test-time refinement formulation to inject physical awareness into 3DFMs and render metric-scale reconstructions. Through extensive experiments on diverse aerial datasets and across several representative 3DFM baselines, we show that VI3 metrically grounds a frozen 3DFM from a coarse inertial motion estimate, yielding reconstruction scales close to the real-world.

## 2. Related Work

Feedforward 3D reconstruction. 3DFMs replace the iterative correspondence-and-optimization loop of classical pipelines [33, 41], by regressing camera poses and dense per-pixel geometry from unposed images in a feedforward pass. DUSt3R [50] and MASt3R [24] predict pointmaps from image pairs, with a family of successors focusing on incremental reconstructions [5, 8, 9, 23, 26, 28, 29, 31, 45, 49, 53, 58]; while VGGT [47], $\pi ^ { 3 }$ [52], and subsequent models [11, 27, 42, 48, 54] extend this to multiple views and estimate a set of common 3D attributes jointly (e.g. depth, cameras, pointmaps). These models provide strong and dense geometric priors but are trained under SfM labels, predicting geometry only up to an unknown global scale.

Scale-aware 3DFMs. Recent works have attempted to inject metric scale into 3DFM reconstructions. Some works [25, 32] integrate it directly into the attention mechanism. Other approaches opt to use modular auxiliary heads to ingest priors (intrinsics, poses, metric depth) to condition the network during inference. AMB3R [46] uses a head specifically trained to regress a data-driven scale from the model features. In turn, Pi3X [52], Depth Anything 3 [27] and others [16, 22, 30] directly produce metric reconstructions from the scaled inputs, while MapAnything [21] and variants [19] use a scale token that allows them to regress the metric scale factor. However, the prior metric scale is still learned, inherited from the training distribution rather than measured; while any metric input depends on a dedicated frontend application. We propose a model-agnostic framework that physically anchors any 3DFM only using inertial signals; conditioning or refining the model output.

Visual-inertial estimation. Inertial sensing resolves the scale ambiguity of monocular vision. Classical visualinertial systems [3, 6, 13, 39] estimate metric, gravityaligned trajectories in real time, but their reconstruction is sparse, covering only trackable features, and runs on a stateful backend tuned to a fixed rig. Closer to our approach, recent work couples unconstrained, learned 3D with inertial data: MASt3R-Fusion [56] fuses feedforward pointmap correspondences with IMU preintegration in a factor-graph backend, and VIGS-SLAM [57] optimizes an explicit perscene 3D Gaussian representation jointly with the IMU states. In both, the inertial signal drives an external optimization, and the pretrained weights are never adapted. This work builds no explicit representation nor recurrent state, instead we use the inertial signal as a lightweight geometric anchor to condition or adapt the model output.

Test-time adaptation of 3D feedforward models. To avoid costly retraining, test-time adaptation updates the weights of the pretrained foundation model during inference to meet a target objective. Generic vision test-time training [15] tunes parameters to each test input to improve robustness under distribution shifts. For 3DFM reconstruction, recent work [8, 17, 49, 55] focuses on refinement for long sequence stability and affordability, to achieve lineartime, drift-free visual and temporal consistency. We propose refinement as a way to adapt the 3DFM weights by injecting metric scale to a short monocular video, anchoring the resulting geometry to a physical reference.

![](images/04474ccb5f3e2837cd4a2b4d3b1c6011e40009e565bcc7e41f31fa5535a5c988.jpg)  
Figure 2. VI3’s block diagram, consisting of two branches. Visual: a pretrained 3DFM takes unposed input images and yields depth and poses. Inertial: the on-board IMU stream, gt-free initialized and preintegrated, provides a scene-specific target trajectory. On-board inertial cues (green dashed arrows), introduced either via pose conditioning, or through adaption (TTR), ground the reconstruction metric.

## 3. Up-to-Scale Geometry from 3DFMs

A 3DFM M receives as input n unposed consecutive frames $\{ \mathcal { T } _ { i } \} _ { i = 1 } ^ { n }$ , with $\mathcal { T } _ { i } \in \{ 0 , \ldots , 2 5 5 \} ^ { w \times h \times 3 }$ , and predicts a set of geometric outputs that depends on the specific model, but typically includes pixel-wise depth $\{ \mathcal { D } _ { i } ^ { \mathcal { M } } \} _ { i = 1 } ^ { n } ,$ $\mathcal { D } _ { i } ^ { \mathcal { M } } \ \in \ \mathbb { R } _ { + } ^ { \dot { w } \times h }$ and camera extrinsics $\{ \mathbf { p } _ { i } ^ { \mathcal { M } } , \mathbf { R } _ { i } ^ { \mathcal { M } } \} _ { i = 1 } ^ { n } \in$ $\mathbb { R } ^ { 3 } \times S O ^ { 3 }$ for each input image. We assume that camera intrinsics are pre-calibrated, although 3DFMs can also predict them. Point cloud representations $\mathcal { X } _ { i } ^ { \mathcal { M } } \in \mathbb { R } ^ { w \times h \times 3 }$ can be computed per image from the depth maps and intrinsics.

## 4. Inertial Preintegration

An IMU provides at a discrete time instant k measurements of the angular velocity $\tilde { \omega } _ { k } ~ \in ~ \mathbb { R } ^ { 3 }$ and linear acceleration $\tilde { \mathbf { a } } _ { k } \in \mathbb { R } ^ { 3 }$ in the local reference frame, which are affected by additive, slowly-varying biases $b _ { k } ^ { g } \in \mathbb { R } ^ { 3 }$ and $b _ { k } ^ { a } \in \mathbb { R } ^ { 3 }$ and and zero-mean Gaussian noises $\hat { \pmb { \eta } } ^ { g } \sim \mathcal { N } ( \mathbf { 0 } , { \pmb { \Sigma } } _ { g } ) \in \mathbb { R } ^ { 3 }$ and $\pmb { \eta } ^ { a } \sim \mathcal { N } ( \mathbf { 0 } , \pmb { \Sigma } _ { a } ) \in \mathbb { R } ^ { 3 }$

$$
\omega _ { k } = \tilde { \omega } _ { k } - b _ { k } ^ { g } - \eta ^ { g } , \qquad a _ { k } = \tilde { a } _ { k } - b _ { k } ^ { a } - \eta ^ { a } ,\tag{1}
$$

where $\omega _ { k }$ and $\mathbf { \alpha } _ { a _ { k } }$ refer to the true values for angular velocity and acceleration, respectively. As we work with short visual-inertial clips, we will assume constant biases in the considered intervals, i.e., $b _ { k } ^ { g } \approx b ^ { g }$ and $b _ { k } ^ { a } \approx \boldsymbol { b } ^ { a }$

## 4.1. Preintegration and state propagation

Preintegrating [12] the m bias- and noise-corrected IMU readings between two consecutive camera frames, i and $i + 1$ , yields the relative rotation $\Delta \mathsf { R } _ { i } = \mathsf { R } _ { i } ^ { \top } \mathsf { R } _ { i + 1 }$ , relative position $\Delta \mathbf { p } _ { i } = \mathbf { p } _ { i + 1 } - \mathbf { p } _ { i }$ and velocity change $\Delta { { \bf { v } } _ { i } } \ =$ $\mathbf { v } _ { i + 1 } - \mathbf { v } _ { i }$ between them:

$$
\Delta \mathtt { R } _ { i } = \prod _ { k = 1 } ^ { m } \mathrm { E x p } \left( \left( \tilde { \omega } _ { k } - b ^ { g } - \pmb { \eta } ^ { g } \right) \Delta t \right) ,\tag{2}
$$

$$
\Delta { { \bf { v } } _ { i } } = { \bf { g } } \Delta T + \sum _ { k = 1 } ^ { m } { { { \sf { R } } _ { k } } \left( { { { \tilde { a } } _ { k } } - b ^ { a } - \eta ^ { a } } \right) \Delta t } ,\tag{3}
$$

$$
\Delta \mathbf { p } _ { i } = \sum _ { k = 1 } ^ { m } \left[ \mathbf { v } _ { k } \Delta t + { \frac { 1 } { 2 } } \mathbf { g } \Delta t ^ { 2 } + { \frac { 1 } { 2 } } \mathrm { R } _ { k } \left( { \tilde { \mathbf { a } } } _ { k } - { \mathbf { b } } ^ { a } - \eta ^ { a } \right) \Delta t ^ { 2 } \right]\tag{4}
$$

where ∆t stands for the time between IMU readings, $\Delta T = m \Delta t$ for the time between consecutive frames, Exp for the exponential map from so(3) to SO(3), and $\mathbf { g }$ for the gravity vector.

Our goal is to recover the inertial states $\mathtt { R } _ { i } , \mathbf { p } _ { i }$ and $\mathbf { v } _ { i }$ for every frame. We will assume the origin at the first frame, hence $\mathrm { R _ { 1 } } ~ = ~ \mathrm { I _ { 3 } }$ and $\mathbf { p } _ { 1 } = \mathbf { 0 }$ , and the states will be recoverable if we know the values for the biases $\boldsymbol { b } ^ { g }$ and $\pmb { b } ^ { a }$ , the gravity g and the velocity in the first frame $\mathbf { v } _ { 1 }$

## 4.2. Inertial state initialization

As shown in by Kaiser et al. [18], the effect of $\pmb { b } ^ { a }$ is negligible for short-time integrations, so we will assume $\pmb { b } ^ { a } \approx \mathbf { 0 }$ and will only estimate the values for $ { b ^ { g } } ,  { \mathbf { v } } _ { 1 }$ and $\mathbf { g }$ from the up-to-scale geometry estimated by the 3DFM and the IMU readings.

Gyroscope bias $( b ^ { g } )$ . As observed in Equation $2 , \mathbf { b } _ { g }$ can be estimated from the angular velocity measurements given an estimate of the relative rotation $\Delta \mathtt { R } _ { i }$ , which we obtain from the 3DFM. We adopt the closed-form solution by Cerezo et al. [7], which, assuming small-angle rotations, estimates ${ \bf b } _ { g }$ from two consecutive frames $\mathcal { T } _ { i }$ and $\mathcal { T } _ { i + 1 }$ as

$$
\begin{array} { r } { \pmb { b } _ { i } ^ { g } \approx - \frac { 1 } { \Delta t } \mathrm { L o g } \Big ( \mathrm { E x p } ( \bar { \omega } _ { i } \Delta t ) ^ { \top } \mathrm { E x p } \left( \frac { 1 } { m } \mathrm { L o g } \Delta \bar { \mathrm { R } } _ { i } ^ { \mathcal { M } } \right) \Big ) , } \end{array}\tag{5}
$$

where $\begin{array} { r } { \bar { \boldsymbol { \omega } } _ { i } = \frac { 1 } { m } \sum _ { k = 1 } ^ { m } \tilde { \boldsymbol { \omega } } _ { k } } \end{array}$ is the average of angular velocity measurements between the two frames, Log is the logarithmic map from SO(3) to so(3), and $\Delta \mathbf { R } _ { i } ^ { \mathcal { M } } = \mathbf { R } _ { i } ^ { \mathcal { M } ^ { \top } } \mathbf { R } _ { i + 1 } ^ { \mathcal { M } }$ is the relative rotation estimated by the 3DFM. The final estimate for $\hat { b } ^ { g }$ is obtained averaging the values for all $n - 1$ pairs over the sequence $\begin{array} { r } { \hat { \pmb { b } } ^ { g } = \frac { 1 } { n - 1 } \sum _ { i = 1 } ^ { n - 1 } { \pmb { b } } _ { i } ^ { g } } \end{array}$

Initial velocity $( \mathbf { v } _ { 1 } )$ . Up-to-scale velocities can be computed from the 3DFM poses using discrete differences as $\mathbf { v } _ { i } ^ { \mathcal { M } } = \Delta \mathbf { p } _ { i } ^ { \mathcal { M } } / \Delta T$ Assuming for now known gravity g, something that we will resolve in the next paragraphs, the 3DFM scale factor s can be constrained using the velocity increments in Eq. 3, yielding $s \Delta \mathbf { v } _ { i } ^ { \mathcal { M } } = \Delta \mathbf { v } _ { i }$ where $\begin{array} { r c l } { \dot { \Delta \mathbf { v } _ { i } ^ { \mathcal { M } } } } & { = } & { \mathbf { v } _ { i + 1 } ^ { \mathcal { M } } - \mathbf { \dot { v } } _ { i } ^ { \mathcal { M } } } \end{array}$ Stacking these constraints over all $n - 1$ image pairs gives the linear system $\mathbf { \Delta } _ { s } \mathbf { V } ^ { \mathcal { M } } = \mathbf { V }$ , with $\mathbf { V } ^ { \mathcal { M } } = \left( \Delta \mathbf { v } _ { 1 } ^ { \mathcal { M } ^ { \top } } \quad \ldots \quad \Delta \mathbf { v } _ { n - 1 } ^ { \mathcal { M } } { } ^ { \top } \right)$ and $\mathbf { V } = \left( \Delta \mathbf { v } _ { 1 } ^ { \top } \quad \ldots \quad \Delta \mathbf { v } _ { n - 1 } ^ { \top } \right) ^ { \top }$ . Since both velocity estimates are noisy, we formulate the problem as a total least squares (TLS) optimization rather than ordinary least squares, and obtain an initial estimate sˆ by solving

$$
\begin{array} { r } { \hat { s } _ { \mathrm { i n i t } } = \arg \underset { s , \Delta \mathbf { V } ^ { \mathcal { M } } , \Delta \mathbf { V } } { \operatorname* { m i n } } | | \Delta \mathbf { V } ^ { \mathcal { M } } \Delta \mathbf { V } | | _ { F } ^ { 2 } } \\ { \mathrm { s . t . } s ( \mathbf { V } ^ { \mathcal { M } } + \Delta \mathbf { V } ^ { \mathcal { M } } ) = \mathbf { V } + \Delta \mathbf { V } } \end{array}\tag{6}
$$

which, in this one-dimensional case, has the following closed-form solution

$$
\hat { s } = \mathrm { s i g n } \left( \left. \mathbf { V } , \mathbf { V } ^ { \mathcal { M } } \right. \right) \frac { \| \mathbf { V } \| } { \| \mathbf { V } ^ { \mathcal { M } } \| } .\tag{7}
$$

As this initial estimation of the scale may be affected by large errors in the velocity estimations, in particular for low-parallax frames, we refine this estimation as follows. We align the initial velocity’s orientation $\breve { \mathbf { v } } _ { 1 } = \mathbf { v } _ { 1 } ^ { \mathcal { M } } / \| \mathbf { v } _ { 1 } ^ { \mathcal { M } } \|$ with the gravity-free first pair, and we compute its magnitude $\mu$ as the median of back-projecting all frames:

$$
\begin{array} { r } { \boldsymbol { \mu } = \operatorname * { m e d i a n } _ { k } \left. \hat { s } \mathbf { v } _ { k } ^ { \mathcal { M } } - \sum _ { 1 < l \leq k } \Delta \mathbf { v } _ { l } , \ \hat { \mathbf { v } } _ { 1 } \right. } \end{array}\tag{8}
$$

The initial velocity is finally computed as $\mathbf { v } _ { 1 } = \mu \breve { \mathbf { v } } _ { 1 }$ . Two sanity checks safeguard the optimization: s falls back to 1 if visual and inertial motions are anti-aligned, and $\mathbf { v } _ { 1 }$ reverts to $s \mathbf { v } _ { 1 } ^ { \mathcal { M } } \operatorname { i f } \mu \leq 0$ due to an inverted gravity vector.

Gravity $( \mathbf { g } ) .$ . The gravity vector can be estimated from image cues using GeoCalib [43], yielding an estimate that we will denote as $\mathbf { g } _ { \mathrm { g c } }$ However, we observed that these estimates can degrade severely on poorly textured, out-ofdistribution images, while GeoCalib’s confidence measure does not reliably reflect these large errors. To detect such failure cases, we compute approximate gravity estimates using two additional methods and assess their mutual consensus.

Assuming negligible accelerometer bias and small linear accelerations, we can obtain a coarse gravity estimate $\mathbf { g } _ { \mathrm { i m u } }$ by averaging the measured accelerations

$$
\mathbf { g } _ { \mathrm { i m u } } = - 9 . 8 1 \frac { \sum _ { k = 1 } ^ { ( n - 1 ) m } \mathsf { R } _ { k } \tilde { \mathbf { a } } _ { k } } { ( n - 1 ) m \parallel \sum _ { k = 1 } ^ { ( n - 1 ) m } \mathsf { R } _ { k } \tilde { \mathbf { a } } _ { k } \parallel }\tag{9}
$$

Additionally, we can also derive the gravity with a kinematic least-squares procedure, an alternative that we will denote as $\mathbf { g } _ { \mathrm { l s } } .$ . Based on the preintegrated velocity update [12] across frame pair (i, j):

$$
\mathbf { v } _ { j } = \mathbf { v } _ { i } + \mathbf { g } \Delta t _ { i j } + \sum _ { i < k \leq j } \mathbf { R } _ { k - 1 } \Delta \mathbf { v } _ { k } ,\tag{10}
$$

we substitute the difference between the visual observation scaled to metric units $\mathbf { v } _ { j } - \mathbf { v } _ { i } = s \Delta \mathbf { v } _ { i j } ^ { { M } }$ , isolating the velocity residuals in the integrated inertial motion:

$$
\mathbf { r } _ { i j } : = \ s \Delta \mathbf { v } _ { i j } ^ { M } - \sum _ { i < k \leq j } \mathbf { R } _ { k - 1 } \Delta \mathbf { v } _ { k } \ = \ \mathbf { g } \Delta t _ { i j } ,\tag{11}
$$

Minimizing the squared norm $\begin{array} { r } { \sum _ { i < j } \| \Delta t _ { i j } \mathbf { g } - \mathbf { r } _ { i j } \| ^ { 2 } } \end{array}$ over all image pairs $i < j$ yields a closed-form solution for the unit gravity direction:

$$
\mathbf { g } _ { \mathrm { l s } } = 9 . 8 1 , \frac { \mathbf { n } } { \| \mathbf { n } \| } , \mathbf { n } = \frac { \sum _ { i < j } \Delta t _ { i j } \mathbf { r } _ { i j } } { \sum _ { i < j } \Delta t _ { i j } ^ { 2 } } = \sum _ { i < j } w _ { i j } \ \frac { \mathbf { r } _ { i j } } { \Delta t _ { i j } } ,\tag{12}
$$

where the weighting term $\begin{array} { r l r } { w _ { i j } } & { { } = } & { \Delta t _ { i j } ^ { 2 } / \sum \Delta t _ { i j } ^ { 2 } } \end{array}$ makes long-baseline pairs dominate quadratically, insulating $\mathbf { g } _ { \mathrm { l s } }$ against high-frequency visual tracking noise while locking the magnitude to the gravity modulus 9.81. There is a circular dependence between the term $\mathbf { g } _ { \mathrm { l s } }$ and the scale s, that was mentioned when estimating the initial velocity $\mathbf { v } _ { 1 }$ , which can be modeled as the function $s = S ( \mathbf { g } )$ This requires to tightly couple both variables through a fixed-point iteration scheme $\begin{array} { r } { s ^ { ( n + 1 ) } = S \big ( \mathbf { g } ^ { ( n ) } \big ) , \mathbf { g } ^ { ( \overline { { n } } + 1 ) } = \mathbf { g } _ { \mathrm { l s } } \big ( \overline { { s } } ^ { ( n + 1 ) } \big ) } \end{array}$ , which starts from a scale-free seed $\mathbf { g } ^ { ( 0 ) } ~ ( \mathbf { g } _ { \mathrm { g c } } ~ \mathrm { o r } ~ \mathbf { g } _ { \mathrm { i m u } } )$ and converges within two to three iterations to solution $\mathrm { F P } ( \mathbf { g } ^ { ( 0 ) } )$ ).

The consensus is evaluated using the two independent seeds for the iterative process: visual $\hat { \bf g } _ { \mathrm { g c } } = \mathrm { F P } ( \mathbf { g } _ { \mathrm { g c } } )$ and inertial $\hat { \bf g } _ { \mathrm { i m u } } = \mathrm { F P } ( \bf g _ { \mathrm { i m u } } )$ . When the motion is sufficient, both seeds converge to a common direction within approximately $1 ^ { \circ } .$ , establishing a consensus error $\angle ( \hat { \bf g } _ { \mathrm { g c } } , \hat { \bf g } _ { \mathrm { i m u } } )$ that reflects how inertial observations pin gravity, and demonstrating that $\mathbf { g } _ { \mathrm { g c } }$ is accurate in most cases.

Table 1. Up-to-scale models. GT-free inertial test-time refinement (TTR) of the pretrained up-to-scale 3D foundation models (Base), which predict geometry without metric scale. The base models are ${ \sim } 3 { - } 7 \times$ under-scaled and thus score ≈0 in $\delta { < } 1 . 2 5 $ , while TTR restores absolute metric depth and pulls both trajectory and depth scale toward 1, respecting the geometry. Note, in addition, that rotation metric are also substantially improved.
<table><tr><td rowspan="2"></td><td rowspan="2">Model</td><td colspan="2">Trans  $( \mathbf { m } ) \downarrow$ </td><td colspan="2"> $\operatorname { R o t } \left( ^ { \circ } \right) \downarrow$ </td><td colspan="2"> $\mathrm { S c a l e ( T r a j . )  1 }$ </td><td colspan="2">AbsRel ↓</td><td colspan="2"> $\delta { < } 1 . 2 5 \uparrow$ </td><td colspan="2">Scale (Depth) →1</td></tr><tr><td>Base</td><td>TTR</td><td>Base</td><td>TTR</td><td>Base</td><td>TTR</td><td>Base</td><td>TTR</td><td>Base</td><td>TTR</td><td>Base</td><td>TTR</td></tr><tr><td></td><td>VGGT</td><td>1.130</td><td>0.358</td><td>0.933</td><td>0.410</td><td>0.14</td><td>1.05</td><td>0.052</td><td>0.089</td><td>0.000</td><td>0.515</td><td>0.17</td><td>1.11</td></tr><tr><td>TartirvV2</td><td>VGGT-Ω</td><td>1.110</td><td>0.378</td><td>0.725</td><td>0.354</td><td>0.17</td><td>1.10</td><td>0.033</td><td>0.080</td><td>0.000</td><td>0.596</td><td>0.19</td><td>1.12</td></tr><tr><td></td><td> $\pi ^ { 3 }$ </td><td>1.047</td><td>0.350</td><td>0.560</td><td>0.296</td><td>0.20</td><td>1.09</td><td>0.041</td><td>0.083</td><td>0.000</td><td>0.556</td><td>0.21</td><td>1.12</td></tr><tr><td></td><td>VGGT</td><td>0.192</td><td>0.061</td><td>0.293</td><td>0.285</td><td>0.27</td><td>0.98</td><td>0.239</td><td>0.234</td><td>0.004</td><td>0.324</td><td>0.28</td><td>1.00</td></tr><tr><td>EC</td><td>VGGT-Ω</td><td>0.176</td><td>0.074</td><td>0.291</td><td>0.252</td><td>0.32</td><td>1.05</td><td>0.212</td><td>0.215</td><td>0.003</td><td>0.243</td><td>0.31</td><td>1.10</td></tr><tr><td></td><td> $\pi ^ { 3 }$ </td><td>0.178</td><td>0.078</td><td>0.234</td><td>0.164</td><td>0.40</td><td>1.08</td><td>0.204</td><td>0.203</td><td>0.008</td><td>0.251</td><td>0.42</td><td>1.18</td></tr><tr><td></td><td>VGGT</td><td>1.109</td><td>0.497</td><td>2.152</td><td>0.301</td><td>0.22</td><td>0.81</td><td>0.209</td><td>0.203</td><td>0.002</td><td>0.201</td><td>0.35</td><td>0.92</td></tr><tr><td>UΛZ-FFPV</td><td>VGGT-Ω</td><td>1.089</td><td>0.328</td><td>0.762</td><td>0.192</td><td>0.20</td><td>0.92</td><td>0.244</td><td>0.237</td><td>0.007</td><td>0.217</td><td>0.31</td><td>1.20</td></tr><tr><td></td><td> $\pi ^ { 3 }$ </td><td>0.891</td><td>0.396</td><td>0.734</td><td>0.246</td><td>0.29</td><td>0.79</td><td>0.241</td><td>0.249</td><td>0.039</td><td>0.138</td><td>0.45</td><td>1.07</td></tr></table>

We classify $\mathbf { g } _ { \mathrm { g c } }$ as an outlier if it diverges beyond the consensus error, enforcing the condition:

$$
\operatorname* { m i n } _ { X \in \{ \mathrm { g c } , \mathrm { i m u } \} } { \cal Z } \big ( \mathbf { g } _ { \mathrm { g c } } , \mathbf { g } _ { X } \big ) \ > \ \operatorname* { m a x } \big ( { \cal Z } ( \hat { \mathbf { g } } _ { \mathrm { g c } } , \hat { \mathbf { g } } _ { \mathrm { i m u } } ) , \gamma _ { \mathrm { m i n } } \big ) ,\tag{13}
$$

where $\begin{array} { r l r } { \gamma _ { \mathrm { m i n } } } & { { } = } & { 2 0 ^ { \circ } } \end{array}$ acts as a lower noise floor. When this condition triggers, the system rejects $\mathbf { g } _ { \mathrm { g c } }$ and adopts the purely inertial fixed point $\hat { \bf g } _ { \mathrm { i m u } } ,$ subject to $\mathrm { ( s i g n } \big ( \hat { \mathbf { g } } _ { \mathrm { i m u } } \big ) \geq 0 \big )$ , alongside its corresponding scale $s _ { \mathrm { i m u } }$ and initial velocity $\mathbf { v } _ { 1 } ( \mathbf { g } _ { \mathrm { i m u } } , s _ { \mathrm { i m u } } )$ , ensuring robust initialization even under severe visual prior degradation.

Once the initialization parameters are estimated, we use the in the preintegration process described in Sec. 4.1 to construct the metric, frame-1-anchored target trajectory $\mathcal { T } ^ { \mathrm { i m u } } = \left\{ \mathbf { p } _ { i } ^ { \mathrm { i m u } } , \mathsf { R } _ { i } ^ { \mathrm { i m u } } \right\} _ { i = 1 } ^ { \dot { n } }$ used as supervisory signal driving the anchoring.

## 5. Metrically anchoring 3DFMs

VI3 aims to inject the metric information encoded in the inertial trajectory into the pretrained 3DFM at test time. We consider two possible approaches to achieve this.

Input conditioning As stated in Sec. 2, certain models [27, 52] can use optional inputs during inference to metrically scale the output. Thus, we feed $\mathcal { T } ^ { \mathrm { i m u } }$ into the 3DFM to obtain a physically-informed reconstruction.

Test-time refinement The objective of the proposed testtime refinement process is to find a joint parameter subset $( \alpha , \hat { \theta } )$ of scale and model weights such that the geometric output of the refined model is physically aligned to the trajectory $\mathcal { T } ^ { \mathrm { i m u } }$ . Since the geometry predicted by the 3DFM is correct up to a global scale factor, we introduce a single learnable scalar $\alpha > 0$ , which from onwards acts on the raw translation $\mathbf { p } _ { i } ^ { \mathcal { M } } \mapsto \alpha \mathbf { p } _ { i } ^ { \mathcal { M } }$ and depth $\mathcal { D } _ { i } ^ { \mathcal { M } } \mapsto \alpha \mathcal { D } _ { i } ^ { \mathcal { M } }$ , tying both representations and leaving rotation and intrinsics untouched. This factor is initialized by directly comparing the ratios between the path lengths $\begin{array} { r } { \alpha _ { 0 } = \bigg ( \sum _ { i } \| \mathbf { p } _ { i } ^ { \mathrm { i m u } } \| _ { 2 } \bigg ) \bigg / \bigg ( \sum _ { i } \| \mathbf { p } ^ { \mathcal { M } } { } _ { i } \| _ { 2 } \bigg ) } \end{array}$

We propose a pose supervision term that takes place on consecutive frames, making the objective globally invariant (rigid transforms shared by prediction and target cancel):

$$
\mathcal { L } _ { \mathrm { p o s } } = \frac { 1 } { n - 1 } \sum _ { i = 2 } ^ { n } \left. ( \mathbf { p } _ { i } ^ { \mathcal { M } } - \mathbf { p } _ { i - 1 } ^ { \mathcal { M } } ) - \Delta \mathbf { p } _ { i - 1 } \right. _ { 1 } ,\tag{14}
$$

$$
\mathcal { L } _ { \mathrm { r o t } } = \frac { 1 } { n - 1 } \sum _ { i = 2 } ^ { n } \left. \mathrm { L o g } \left( \left( \mathbb { R } _ { i - 1 } ^ { \mathcal { M } } \mathrm { ^ T } { \mathbb R } _ { i } ^ { \mathcal { M } } \right) ^ { \top } \Delta \mathrm { R } _ { i - 1 } \right) \right. _ { 2 } .\tag{15}
$$

Since this procedure operates without an available 3D ground truth, we need to enforce cross-view geometric consistency by coupling depth maps and camera poses via a self-supervised consistency loop. We thus introduce a geometric depth-consistency loss that anchors predicted depths directly to the inertial recovered pose scale. For every pair $( i , j )$ where $i \neq j$ , we penalize discrepancies between the reprojected depth and the source depth sampled at the valid reprojected coordinates [14]:

$$
\mathcal { L } _ { \mathrm { d c o n s } } = \frac { 1 } { n } \sum _ { t = 1 } ^ { n } \left[ \operatorname* { m i n } _ { i \neq j } \left| [ \pi _ { i } ( \mathcal { X } _ { j } ^ { \mathcal { M } } ) ] z - \mathcal { D } _ { i } ^ { \mathcal { M } } \Big ( \pi _ { i } ( \mathcal { X } _ { j } ^ { \mathcal { M } } ) \Big ) \right| \right] _ { \mathrm { v a l i d } } ,\tag{16}
$$

with $\pi _ { i }$ the projection operator projecting into the i-th frame the 3D world points ${ \mathcal { X } } _ { j } ^ { { \mathcal { M } } }$ resulting of backprojecting depth predictions from frame j; and $[ ] _ { Z }$ the z-coordinate.

Table 2. Metric models: learned priors vs. inertial cues. Each backbone’s native metric prediction (Base), the same pretrained model conditioned on our GT-free IMU-integrated poses (IMU-cond, pose-only, feed-forward), and, conversely, using inertial test-time refine ment (TTR). <sup>†</sup>IMU-cond additionally feeds camera intrinsics (required by DA3’s pose encoder); Pi3X’s IMU-cond is pose-only.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Model</td><td colspan="3">Trans (m) ↓</td><td colspan="3"> $\operatorname { R o t } \left( ^ { \circ } \right) \downarrow$ </td><td colspan="3">Scale (Traj.) →1</td><td colspan="3">Scale (Depth) →1</td></tr><tr><td>Base</td><td>IMU-cond</td><td>TTR</td><td>Base</td><td>IMU-cond</td><td>TTR</td><td>Base</td><td>IMU-cond</td><td>TTR</td><td>Base</td><td>IMU-cond</td><td>TTR</td></tr><tr><td rowspan="2">TartanAirV2</td><td>Pi3X</td><td>0.392</td><td>0.371</td><td>0.382</td><td>0.802</td><td>0.546</td><td>0.316</td><td>1.18</td><td>1.20</td><td>1.08</td><td>1.26</td><td>1.25</td><td>1.11</td></tr><tr><td>DA3</td><td>0.322</td><td>0.446†</td><td>0.486</td><td>1.379</td><td>0.801†</td><td>0.351</td><td>0.90</td><td>0.83†</td><td>1.06</td><td>0.99</td><td>0.90†</td><td>1.19</td></tr><tr><td rowspan="2">EuRoC</td><td>Pi3X</td><td>0.071</td><td>0.081</td><td>0.067</td><td>0.257</td><td>0.306</td><td>0.254</td><td>0.83</td><td>0.82</td><td>1.04</td><td>0.91</td><td>0.93</td><td>1.11</td></tr><tr><td>DA3</td><td>0.061</td><td>0.055†</td><td>0.092</td><td>0.261</td><td>0.326†</td><td>0.312</td><td>0.72</td><td> $0 . 7 5 ^ { \dagger }$ </td><td>1.13</td><td>0.74</td><td>0.73†</td><td>1.20</td></tr><tr><td rowspan="2">UZH-FPV</td><td>Pi3X</td><td>0.576</td><td>0.535</td><td>0.368</td><td>0.933</td><td>0.738</td><td>0.233</td><td>1.11</td><td>1.15</td><td>0.82</td><td>1.63</td><td>1.65</td><td>1.10</td></tr><tr><td>DA3</td><td>0.538</td><td>0.452†</td><td>0.379</td><td>1.252</td><td>0.626†</td><td>0.231</td><td>0.96</td><td>0.83†</td><td>0.88</td><td>1.39</td><td>1.05†</td><td>1.07</td></tr></table>

Table 3. Calibration-parameter init error. Accuracy of the backbone-independent, fully-GT-free estimates of the two IMU calibration parameters that seed refinement: gravity direction and gyro bias, the latter as absolute error with relative in parentheses. GT gyro bias is only available on EuRoC (– elsewhere).
<table><tr><td>Dataset</td><td>Gravity (°)↓</td><td>Gyro-bias (rad/s) ↓</td></tr><tr><td>TartanAirV2</td><td>1.05</td><td></td></tr><tr><td>EuRoC</td><td>1.08</td><td>0.009 (11.3%)</td></tr><tr><td>UZH-FPV</td><td>6.61</td><td></td></tr></table>

On top of these, we add two regularization terms that protect the structural integrity: (i) a frame-1 positional anchor on the first frame predicted position $\hat { \mathbf { p } } _ { 1 } ^ { \bar { M } }$ , restricting the coordinate origin $( \mathcal { L } _ { \mathrm { a } } ^ { p } ~ = ~ \| \hat { \mathbf { p } } _ { 1 } ^ { \mathcal { M } } \| _ { 1 } )$ , and (ii) a depthto-init anchor, which pins the depth to its zero-shot shape $\begin{array} { r } { \mathcal { L } _ { \mathrm { a } } ^ { D } = \frac { 1 } { n } \sum _ { i } \| \mathcal { D } _ { i } ^ { M } - \mathcal { D } _ { i } ^ { M } ^ { ( 0 ) } \| _ { 1 } } \end{array}$ . This allows the network to adapt to the input stream photometric distribution while ensuring the geometry remains strictly bounded by the 3DFM structural prior. The total objective amounts to:

$$
\begin{array} { r } { \mathcal { L } ( \theta ) = w _ { r } \mathcal { L } _ { \mathrm { r o t } } + w _ { p } \mathcal { L } _ { \mathrm { p o s } } + w _ { \mathrm { d } } \mathcal { L } _ { \mathrm { d c o n s } } + w _ { a _ { p } } \mathcal { L } _ { \mathrm { a } } ^ { p } + w _ { a _ { d } } \mathcal { L } _ { \mathrm { a } } ^ { D } } \\ { ( 1 7 ) } \end{array}
$$

which we minimize by unfreezing only the camera- and depth-estimation heads, together with the last few L interframe (global) attention blocks of the aggregator in the 3DFM, plus the scalar $\alpha ,$ running K gradient steps on the single window with T<sup>imu</sup> held fixed.

## 6. Experiments

We evaluate the performance of VI3 through quantitative and qualitative results, and ablating design choices.

Datasets. We use three visual-inertial aerial datasets spanning synthetic, micro-aerial and drone racing nature. TartanAirV2 [51] contributes 24 synthetic drone-flight sequences coming from four varied environments: indoors and outdoors (House, Hospital, ModernCityDowntown, ForestEnv) at the two existing difficulties (easy/hard) and on the first two trajectories (P000, P002). We take the full EuRoC-MAV [4] dataset, consisting of 11 sequences: Vicon-Room (GT-depth obtained from Leica MS50 laser scanner) and Machine-Hall (GT-depth stereoderived). UZH-FPV [10] brings 16 drone-racing sequences with public GT (Snapdragon indoor/outdoor, forward- and 45<sup>◦</sup>-facing cameras); the event-camera and no-GT competition splits are omitted. GT-depth is stereo-derived. Notably, TartanAirV2 (or its earlier version TartanAir), appears in the original training corpus of all the backbones we use, except VGGT. Its inclusion aims to show that results, despite being closer to in-distribution rather than zero-shot, still exhibit a clear scene-specific gap we target. The datasets utilized in this work span the whole motion regime range: noise-free (synthetic) and high-parallax (TartanAirV2); moderate motion (EuRoC); and highly dynamic, low-parallax (UZH-FPV).

Table 4. Initial velocity estimation. Recovery of the direction and magnitude, for each specific dataset and backbone.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Metric</td><td colspan="3">Up-to-scale</td><td rowspan="2">Scale-aware</td></tr><tr><td>VGGT VGGT-Ω</td><td></td><td> $\pi ^ { 3 }$  Pi3X DA3</td></tr><tr><td rowspan="2">TartanAirV2</td><td> $v _ { 1 }$  dir. (°) ↓</td><td>6.2</td><td>4.3</td><td>5.4</td><td>6.6 11.0</td></tr><tr><td> $v _ { 1 }$  mag. ↓</td><td>0.21</td><td>0.22</td><td>0.20</td><td>0.20 0.22</td></tr><tr><td>EuRoC</td><td>v1 dir. (°) ↓ v1 mag. ↓</td><td>10.0 0.21</td><td>3.6 0.23</td><td>9.3 0.26</td><td>8.8 5.6 0.17 0.22</td></tr><tr><td>UZH-FPV</td><td>v1 dir. (°) ↓ v1 mag. ↓</td><td>10.4 0.24</td><td>4.9 0.17</td><td>4.6 0.27</td><td>7.2 6.3 0.17 0.16</td></tr></table>

## 6.1. Experimental setup

Consistent with the nature of 3DFMs, which operate on short context windows, we evaluate on clips: short segments sampled from each sequence consisting of a window of $n { = } 8$ frames, with a stride of two. We sample three clips per sequence (near the start, middle, and end) for diverse coverage. All reported metrics are therefore aggregated as the median over all clips across a dataset’s sequences.

Backbones. We extensively analyze the behaviour of testtime refinement on five relevant pretrained 3DFMs (using their released weights): three of them up-to-scale, VGGT [47], VGGT-Ω [48] and $\pi ^ { 3 }$ [52]; and other two scale-aware, Pi3X [52] and Depth Anything 3 (DA3) [27]. Optimization. For the TTR, we experimentally set $w _ { r } = 2 0$ $w _ { t } \mathrm { = } 1 0 , \mathrm { ~ } w _ { \mathrm { d } } \mathrm { = } 1 0 , \mathrm { ~ } w _ { a _ { p } } \mathrm { = } 1 0 0 0$ , and ${ w _ { a } } _ { d } \mathrm { = } 1$ , as they bring a good trade-off between pose refinement and scale recovery, without corrupting depth. We optimize with AdamW for a fixed budget of $K = 1 6 0$ steps per clip. Experiments run on a single NVIDIA RTX 5090. TTR optimizes the scale factor $\alpha ,$ alongside the model’s camera head, depth head, and the final aggregator blocks (L = 8), while keeping all other parameters frozen.

Evaluation metrics. We evaluate camera pose, scale, and dense depth. Unlike conventional visual-odometry practice, we do not align trajectories before computing pose error, as our goal is to assess absolute metric accuracy. Trans is the RMSE of per-frame camera positions against GT (meters); Rot is the mean geodesic angle to GT orientations (degrees). Scale (Traj.) is the median per-frame ratio $| t _ { \mathrm { p r e d } } | / | t _ { \mathrm { g t } } |$ , where 1 is ideal and values below/above indicate under/over-scaling, excluding near-stationary (GT displacement <1cm) clips, for which the ratio is undefined. For dense depth we report two complementary metrics. Shape is invariant to global scale and isolates geometric fidelity, measured via per-frame median scalealignment AbsRel (↓) and the absolute inlier ratio δ<1.25 (↑), which measures the pixels within 1.25× of GT without alignment. Scale (Depth), measured without alignment as median $\left( D _ { \mathrm { p r e d } } / D _ { \mathrm { g t } } \right)$ , again targets 1. Reporting them separately distinguishes a reconstruction correctly shaped but mis-scaled, from one correct on both, only differentiated by the Scale (Depth) column.

All results are seeded; on fixed hardware, re-runs report variances within 0.01 m, 0.04<sup>◦</sup> and 0.03 in scale, far below the reported anchoring effects.

## 6.2. Results

Fig. 3 illustrates qualitative results, further examples can be found in the supplementary. Tables 1 and 2 report the median results respectively for up-to-scale and metric backbones across sequences and clips of the raw pretrained model (Base), IMU conditioning (if applicable) and our TTR. The results show that inertial measures are useful to inject physical knowledge in models spanning both natures. Up-to-scale models can only recover a correct geometry, but at most up to a 45% of the original scale, reflected also by position and the poor results in $\delta { < } 1 . 2 5$ , which confirms structions in all cases (scale ≈ 1), which is propagated to translation. Additionally, we observe a consistent improvement in rotation, while our proposed cross-view loss helps to preserve the geometry, meaning that the injected scale does not corrupt 3D awareness. Metric-aware models produce scale estimates closer to 1 and much better translation results (especially on TartanAirV2, consistent with its indistribution status), but we show that the IMU can improve this substantially in out-of-distribution cases both by TTR or conditioning on auxiliary geometry at test time. We also observe a slight but consistent improvement in the rotation, meaning that the inertial measurements can inject accurate information for the six degrees of freedom of the motion. We can see that, in some cases, the performance does not vary by just conditioning the network by metric inputs, suggesting that the data-driven prior is overriding any input. In these cases, TTR proves to be usually a better alternative, since the backbone parameters are adapted to fit the specific scene.

![](images/ce230c865f0f0665dd9148b7d126ce9330dcb2f413e565c4c82ad15ef77289bc.jpg)  
Figure 3. Qualitative examples. From top to bottom: TartanAirV2 ModernCityDowntown P000; EuRoC Vicon-Room V2-03; UZH-FPV indoor-forward-9. VI3 (green) correctly recovers, thanks to the inertial signal (blue), metric, scene-consistent geometry, in contrast to the pretrained 3DFM (red). Best in color. the monocular scale ambiguity. In turn, our proposed refinement is able to successfully retrieve near metric recon-

![](images/069143dec6e4218c4564ae82ea8c4e72a8430c3d1181ca9213ba461e11dfee49.jpg)  
perturbation (°)

![](images/43193b7dc13d711c57ce692920b64986a99a6967cd1e466fed76c77e4ac9f991.jpg)  
perturbation (×)

![](images/0d4d85669eb48f594e5ec1a7429f72368634a37dc77c3f53671530910babda63.jpg)  
perturbation (°)

![](images/5ebc6ce6806389961d703ad7254ca0119180d65d990630efa6bf220070bb9a92.jpg)  
perturbation (rad/s)

![](images/ce1db44fbbbca14109bce8f133f3e81033172d9ed9e8b270f0817e997d3bee56.jpg)  
perturbation (m/s2)  
Figure 4. Sensitivity to inertial initialization of recovered metric scale (VGGT backbone). Downstream scale (→ 1 ideal) as each GT initialization component is independently perturbed (growing values of error applied to initial linear velocity, gravity and IMU biases).

## 6.3. Additional analyses

The most relevant analyses are included here, while further ones can be found in the supplementary material.

Inertial initialization sensitivity. We test how IMU initialization errors affect the recovered metric scale, applying varied noise over the preintegrated trajectory, then injected in a VGGT backbone using TTR. Fig. 4 shows that VI3’s scale recovery is robust against IMU biases and gravity noise, while being greatly affected by deviations in the initial velocity estimate: direction and especially magnitude errors degrade performance substantially.

Initialization accuracy. We evaluate the error obtained by the initialization procedure described in Sec. 4.2, analyzing two sensor-derived variables, less dependent on the choice of backbone: absolute error angle in the gravity direction gˆ and the relative gyro bias error $\bar { \lVert \boldsymbol { b } _ { g } - \boldsymbol { b } _ { g } \rVert / \lVert \boldsymbol { b } _ { g } \rVert }$ . Tab. 3 shows the results (only EuRoC has GT bias), where we observe that our initialization process achieves good estimates for both variables, with a worse performance in the UZH-FPV dataset, more challenging due to the high speed.

We also evaluate the error obtained in the estimation of the most sensitive initialization parameter, the initial velocity. Since this metric depends on the employed backbone, we evaluate the error in both components: direction and magnitude factor $( | | \hat { v } _ { 0 } | - | v _ { 0 } | | / | v _ { 0 } | )$ . Tab. 4 shows that the direction error stays small across backbones while the magnitude spreads more spreads widely.

## 7. Conclusion

In this work, we have demonstrated that the IMU is a suitable source of physical knowledge to anchor scaleambiguous geometric outputs from any pretrained 3DFM into metric observations. For that, we introduced VI3, a sensor-in-the-loop framework that reframes physical anchoring, with diverse ways of injecting metric scale, including a dedicated refinement method. Through extensive experiments, we show that VI3 generalizes across backbones of different nature and diverse scenes; and that neither learned scale priors nor input conditioning are reliable enough alternatives, a reminder that physical properties need to be anchored in physical measurement, as data-driven priors can be inaccurate and prone to out-ofdistribution failures.

The same philosophy extends as future work to any metric signal available at deployment, or a fusion of several (e.g. GNSS, wheel odometry, depth), and to downstream pipelines (feedforward novel-view synthesis, mapping) that depend on accurate metric geometry, pointing toward foundation models that specialize to the physical world they are dropped into.

## References

[1] Sameer Agarwal, Yasutaka Furukawa, Noah Snavely, Ian Simon, Brian Curless, Steven M Seitz, and Richard Szeliski. Building rome in a day. Communications of the ACM, 54 (10):105–112, 2011. 1

[2] Yanqi Bao, Tianyu Ding, Jing Huo, Yaoli Liu, Yuxin Li, Wenbin Li, Yang Gao, and Jiebo Luo. 3d gaussian splatting: Survey, technologies, challenges, and opportunities. IEEE

Transactions on Circuits and Systems for Video Technology, 35(7):6832–6852, 2025. 1

[3] Simon Boche, Jaehyung Jung, Sebastian Barbas Laina, and´ Stefan Leutenegger. Okvis2-x: Open keyframe-based visualinertial slam configurable with dense depth or lidar, and gnss. IEEE Transactions on Robotics, 2025. 2

[4] Michael Burri, Janosch Nikolic, Pascal Gohl, Thomas Schneider, Joern Rehder, Sammy Omari, Markus W Achtelik, and Roland Siegwart. The euroc micro aerial vehicle datasets. The International Journal of Robotics Research, 2016. 6

[5] Yohann Cabon, Lucas Stoffl, Leonid Antsfeld, Gabriela Csurka, Boris Chidlovskii, Jerome Revaud, and Vincent Leroy. Must3r: Multi-view network for stereo 3d reconstruction. In CVPR, pages 1050–1060, 2025. 2

[6] Carlos Campos, Richard Elvira, Juan J. Gomez Rodr ´ ´ıguez, Jose M. M. Montiel, and Juan D. Tard´ os. ORB-SLAM3:´ An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM. IEEE Transactions on Robotics, 37 (6):1874–1890, 2021. arXiv:2007.11898 [cs.RO]. 2

[7] Samuel Cerezo, Seong Hun Lee, and Javier Civera. An efficient closed-form solution to full visual-inertial state initialization. IEEE Robotics and Automation Letters, 2026. 4

[8] Xingyu Chen, Yue Chen, Yuliang Xiu, Andreas Geiger, and Anpei Chen. TTT3R: 3D Reconstruction as Test-Time Training, 2026. arXiv:2509.26645 [cs]. 2, 3

[9] Zhuoguang Chen, Minghui Qin, Tianyuan Yuan, Zhe Liu, and Hang Zhao. Long3r: Long sequence streaming 3d reconstruction. arXiv preprint arXiv:2507.18255, 2025. 2

[10] Jeffrey Delmerico, Titus Cieslewski, Henri Rebecq, Matthias Faessler, and Davide Scaramuzza. Are We Ready for Autonomous Drone Racing? The UZH-FPV Drone Racing Dataset. In 2019 International Conference on Robotics and Automation (ICRA), pages 6713–6719, Montreal, QC, Canada, 2019. IEEE. 6

[11] Sven Elflein, Qunjie Zhou, and Laura Leal-Taixe. Light3r-´ sfm: Towards feed-forward structure-from-motion. In CVPR, pages 16774–16784, 2025. 2

[12] Christian Forster, Luca Carlone, Frank Dellaert, and Davide Scaramuzza. On-Manifold Preintegration for Real-Time Visual-Inertial Odometry. IEEE Transactions on Robotics, 33(1):1–21, 2017. arXiv:1512.02363 [cs.RO]. 2, 3, 4

[13] Patrick Geneva, Kevin Eckenhoff, Woosik Lee, Yulin Yang, and Guoquan Huang. OpenVINS: A research platform for visual-inertial estimation. In Proc. of the IEEE International Conference on Robotics and Automation, Paris, France, 2020. 2

[14] Clement Godard, Oisin Mac Aodha, Michael Firman, and´ Gabriel J. Brostow. Digging into self-supervised monocular depth prediction. The International Conference on Computer Vision (ICCV), 2019. 5

[15] Dongchen Han, Yining Li, Tianyu Li, Zixuan Cao, Ziming Wang, Jun Song, Yu Cheng, Bo Zheng, and Gao Huang. Vit<sup>3</sup>: Unlocking test-time training in vision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026. 2

[16] Wonbong Jang, Philippe Weinzaepfel, Vincent Leroy, Lourdes Agapito, and Jerome Revaud. Pow3R: Empowering Unconstrained 3D Reconstruction with Camera and Scene Priors. 2025. 2

[17] Haian Jin, Rundi Wu, Tianyuan Zhang, Ruiqi Gao, Jonathan T. Barron, Noah Snavely, and Aleksander Holynski. ZipMap: Linear-time stateful 3d reconstruction via testtime training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026. 3

[18] Jacques Kaiser, Agostino Martinelli, Flavio Fontana, and Davide Scaramuzza. Simultaneous state initialization and gyroscope bias calibration in visual inertial aided navigation. IEEE Robotics and Automation Letters, 2(1):18–25, 2016. 3

[19] Jay Karhade, Nikhil Keetha, Yuchen Zhang, Tanisha Gupta, Akash Sharma, Sebastian Scherer, and Deva Ramanan. Any4d: Unified feed-forward metric 4d reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14578–14589, 2026. 2

[20] Nikhil Keetha, Jay Karhade, Krishna Murthy Jatavallabhula, Gengshan Yang, Sebastian Scherer, Deva Ramanan, and Jonathon Luiten. Splatam: Splat, track & map 3d gaussians for dense rgb-d slam. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21357–21366. IEEE, 2024. 1

[21] Nikhil Keetha, Norman Muller, Johannes Sch ¨ onberger,¨ Lorenzo Porzi, Yuchen Zhang, Tobias Fischer, Arno Knapitsch, Duncan Zauss, Ethan Weber, Nelson Antunes, Jonathon Luiten, Manuel Lopez-Antequera, Samuel Rota Bulo, Christian Richardt, Deva Ramanan, Sebastian Scherer,\` and Peter Kontschieder. MapAnything: Universal Feed-Forward Metric 3D Reconstruction, 2026. arXiv:2509.13414 [cs]. 1, 2

[22] Ramil Khafizov, Artem Komarichev, Ruslan Rakhimov, Peter Wonka, and Evgeny Burnaev. G-cut3r: Guided 3d reconstruction with camera and depth prior integration. arXiv preprint arXiv:2508.11379, 2025. 2

[23] Yushi Lan, Yihang Luo, Fangzhou Hong, Shangchen Zhou, Honghua Chen, Zhaoyang Lyu, Shuai Yang, Bo Dai, Chen Change Loy, and Xingang Pan. Stream3r: Scalable sequential 3d reconstruction with causal transformer. arXiv preprint arXiv:2508.10893, 2025. 2

[24] Vincent Leroy, Yohann Cabon, and Jer´ ome Revaud.ˆ Grounding Image Matching in 3D with MASt3R, 2024. arXiv:2406.09756 [cs]. 1, 2

[25] Ruilong Li, Brent Yi, Junchen Liu, Hang Gao, Yi Ma, and Angjoo Kanazawa. Cameras as relative positional encoding. Advances in Neural Information Processing Systems, 38:15984–16009, 2026. 2

[26] Zizun Li, Jianjun Zhou, Yifan Wang, Haoyu Guo, Wenzheng Chang, Yang Zhou, Haoyi Zhu, Junyi Chen, Chunhua Shen, and Tong He. Wint3r: Window-based streaming reconstruction with camera token pool. arXiv preprint arXiv:2509.05296, 2025. 2

[27] Haotong Lin, Sili Chen, Junhao Liew, Donny Y. Chen, Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth Anything 3: Recovering the Visual Space from Any Views, 2025. arXiv:2511.10647 [cs]. 1, 2, 5, 7

[28] Yuzheng Liu, Siyan Dong, Shuzhe Wang, Yingda Yin, Yanchao Yang, Qingnan Fan, and Baoquan Chen. Slam3r: Realtime dense scene reconstruction from monocular rgb videos. In CVPR, pages 16651–16662, 2025. 2

[29] Jiahao Ma, Lei Wang, David Ahmedt-Aristizabal, Chuong Nguyen, et al. Puzzles: Unbounded video-depth augmentation for scalable end-to-end 3d reconstruction. arXiv preprint arXiv:2506.23863, 2025. 2

[30] Mohammad Mahdavian, Gordon Tan, Binbin Xu, Yuan Ren, Dongfeng Bai, and Bingbing Liu. Uniscale: Unified scale-aware 3d reconstruction for multi-view understanding via prior injection for robotic perception. arXiv preprint arXiv:2602.23224, 2026. 1, 2

[31] Soroush Mahdi, Fardin Ayar, Ehsan Javanmardi, Manabu Tsukada, and Mahdi Javanmardi. Evict3r: Training-free token eviction for memory-bounded streaming visual geometry transformers. arXiv preprint arXiv:2509.17650, 2025. 2

[32] Takeru Miyato, Bernhard Jaeger, Max Welling, and Andreas Geiger. Gta: A geometry-aware attention mechanism for multi-view transformers. In International Conference on Learning Representations, pages 8172–8208, 2024. 2

[33] Raul Mur-Artal and Juan D Tardos. Orb-slam2: An open-´ source slam system for monocular, stereo, and rgb-d cameras. IEEE transactions on robotics, 33(5):1255–1262, 2017. 1, 2

[34] Raul Mur-Artal, J. M. M. Montiel, and Juan D. Tardos. Orbslam: A versatile and accurate monocular slam system. IEEE Transactions on Robotics, 31(5):1147–1163, 2015. 1

[35] Maxime Oquab, Timothee Darcet, Th´ eo Moutakanni, Huy´ Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve Je-´ gou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning Robust Visual Features without Supervision, 2024. arXiv:2304.07193 [cs]. 2

[36] Linfei Pan, Daniel Bar´ ath, Marc Pollefeys, and Johannes L´ Schonberger. Global structure-from-motion revisited. In¨ European Conference on Computer Vision, pages 58–77. Springer, 2024. 1

[37] Vojtech Panek, Qunjie Zhou, Yaqing Ding, Sergio´ Agostinho, Zuzana Kukelova, Torsten Sattler, and Laura Leal-Taixe. A guide to structureless visual localization: V.´ panek et al. International Journal of Computer Vision, 134 (6):263, 2026. 1

[38] Haosong Peng, Hao Li, Yalun Dai, Yushi Lan, Yihang Luo, Tianyu Qi, Zhengshen Zhang, Yufeng Zhan, Junfei Zhang, Wenchao Xu, et al. Omnivggt: Omni-modality driven visual geometry grounded transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 36485–36497, 2026. 1

[39] Tong Qin, Peiliang Li, and Shaojie Shen. Vins-mono: A robust and versatile monocular visual-inertial state estimator. IEEE transactions on robotics, 34(4):1004–1020, 2018. 2

[40] Torsten Sattler, Akihiko Torii, Josef Sivic, Marc Pollefeys, Hajime Taira, Masatoshi Okutomi, and Tomas Pajdla. Are

large-scale 3d models really necessary for accurate visual localization? In 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 6175–6184. IEEE, 2017. 1

[41] Johannes Lutz Schonberger and Jan-Michael Frahm.¨ Structure-from-Motion Revisited. In Conference on Computer Vision and Pattern Recognition (CVPR), 2016. 1, 2

[42] Zhenggang Tang, Yuchen Fan, Dilin Wang, Hongyu Xu, Rakesh Ranjan, Alexander Schwing, and Zhicheng Yan. Mv-dust3r+: Single-stage scene reconstruction from sparse views in 2 seconds. In CVPR, pages 5283–5293, 2025. 2

[43] Alexander Veicht, Paul-Edouard Sarlin, Philipp Lindenberger, and Marc Pollefeys. Geocalib: Learning singleimage calibration with geometric optimization. In European Conference on Computer Vision, pages 1–20. Springer, 2024. 4

[44] Guangming Wang, Lei Pan, Songyou Peng, Shaohui Liu, Chenfeng Xu, Yanzi Miao, Wei Zhan, Masayoshi Tomizuka, Marc Pollefeys, and Hesheng Wang. Nerfs in robotics: A survey. The International Journal of Robotics Research, 45 (6):858–892, 2026. 1

[45] Hengyi Wang and Lourdes Agapito. 3d reconstruction with spatial memory. In Proceedings of the International Conference on 3D Vision (3DV), pages 78–89. IEEE, 2025. 2

[46] Hengyi Wang and Lourdes Agapito. Amb3r: Accurate feedforward metric-scale 3d reconstruction with backend. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14612–14625, 2026. 2

[47] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. VGGT: Visual Geometry Grounded Transformer, 2025. arXiv:2503.11651 [cs]. 1, 2, 7

[48] Jianyuan Wang, Minghao Chen, Shangzhan Zhang, Nikita Karaev, Johannes Schonberger, Patrick Labatut, Piotr Bo-¨ janowski, David Novotny, Andrea Vedaldi, and Christian Rupprecht. VGGT-Ω, 2026. 1, 2, 7

[49] Qianqian Wang\*, Yifei Zhang\*, Aleksander Holynski, Alexei A. Efros, and Angjoo Kanazawa. Continuous 3d perception model with persistent state. In CVPR, 2025. 2, 3

[50] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. DUSt3R: Geometric 3D Vision Made Easy, 2023. arXiv:2312.14132 [cs]. 1, 2

[51] Wenshan Wang, Delong Zhu, Xiangwei Wang, Yaoyu Hu, Yuheng Qiu, Chen Wang, Yafei Hu, Ashish Kapoor, and Sebastian Scherer. TartanAir: A Dataset to Push the Limits of Visual SLAM, 2020. arXiv:2003.14338 [cs.RO]. 6

[52] Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chunhua Shen, and Tong He. π<sup>3</sup>: Permutation-equivariant visual geometry learning, 2026. 1, 2, 5, 7

[53] Yuqi Wu, Wenzhao Zheng, Jie Zhou, and Jiwen Lu. Point3r: Streaming 3d reconstruction with explicit spatial pointer memory. arXiv preprint arXiv:2507.02863, 2025. 2

[54] Jianing Yang, Alexander Sax, Kevin J Liang, Mikael Henaff, Hao Tang, Ang Cao, Joyce Chai, Franziska Meier, and Matt Feiszli. Fast3r: Towards 3d reconstruction of 1000+ images in one forward pass. In CVPR, pages 21924–21935, 2025. 2

[55] Junyi Zhang, Charles Herrmann, Junhwa Hur, Chen Sun, Ming-Hsuan Yang, Forrester Cole, Trevor Darrell, and Deqing Sun. Loger: Long-context geometric reconstruction with hybrid memory. arXiv preprint arXiv:2603.03269, 2026. 3

[56] Yuxuan Zhou, Xingxing Li, Shengyu Li, Zhuohao Yan, Chunxi Xia, and Shaoquan Feng. Mast3r-fusion: Integrating feed-forward visual model with imu, gnss for highfunctionality slam, 2025. 2

[57] Zihan Zhu, Wei Zhang, Moyang Li, Norbert Haala, Marc Pollefeys, and Daniel Barath. Vigs-slam: Visual inertia gaussian splatting slam. In European Conference on Computer Vision (ECCV), 2026. 2

[58] Dong Zhuo, Wenzhao Zheng, Jiahe Guo, Yuqi Wu, Jie Zhou, and Jiwen Lu. Streaming 4d visual geometry transformer. arXiv preprint arXiv:2507.11539, 2025. 2

# VI3: Grounding Pretrained 3D Foundation Models with Inertial Cues

Supplementary Material

## A. Additional results

We provide further experiments and ablation studies, spanning the analysis of specific components of VI3, specifically the inertial initialization and the test-time refinement.

## A.1. Inertial initialization

## Ground-truth inertial initialization

Tables 8 and 9 compare VI3 using our ground-truth-free (GT-free) inertial initialization from Sec. 4.2 against using the ground truth (GT) inertial parameters. As before, we separately assess its impact on TTR for the up-to-scale models, and on both input conditioning and TTR for the metric baselines. These results validate our proposed GT-free initialization, which maintains a small performance gap with respect to the GT oracle in rotation and depth, although the gap is larger for translation due to the sensitivity of the estimated scale to errors in the initial velocity. This trend is consistent across datasets and models, for both metric and up-to-scale settings. Overall, the results demonstrate that TTR can effectively absorb the errors introduced by the initial inertial estimate and refine the predictions, making it a more effective mechanism than input conditioning for injecting inertial cues into pretrained 3DFMs.

## Gravity consensus

Table 5 provides additional results for our gravity consensus mechanism. The image-based gravity estimation $\mathbf { g } _ { \mathrm { g c } }$ typically fails on texture-less scenarios, as reflected by large mean errors in TartanAirV2 and UZH-FPV. The use of an alternative, rough, inertial-only gravity estimation $\left( \bf g _ { \mathrm { i m u } } \right)$ brings the possibility to detect large errors of $\mathbf { g } _ { \mathrm { g c } }$ , and swap to a visual-inertial least-squares estimate $\left( \mathbf { g } _ { \mathrm { l s } } \right)$ , which prevents from using largely erroneous values and, on average, shows a configuration which outperforms every gravity estimate alone. Note, in the right part of the table, the specific percentage of gravity swaps from $\mathbf { g } _ { \mathrm { g c } }$ to $\mathbf { g } _ { \mathrm { l s } } ,$ and how the error is substantially reduced for these specific cases. This shows how our consensus method effectively detects large errors from image-based gravity preditions.

## Initial velocity-scale estimation

Table. 6 supports our decision of formulating the initial velocity estimation as total least squares (TLS), instead of as an ordinary least squares (OLS). The assumption underlying TLS is that both the visual and inertial counterparts are noisy. On TartanAirV2, for which IMU data is noiseless, this assumption does not hold and both scale estimates show similar errors. However, on the two real (noisy) datasets (EuRoC and UZH-FPV) TLS leads to clear improvements over OLS, with scale estimates closer to the ideal value (1).

## Initial velocity magnitude

When applying the regressed velocity scale to obtain the magnitude of the initial velocity, two options arise: scaling only the first pair of frames, or backprojecting all pairs to the initial one and taking the median as a robust representative. Table 7 shows that using all pairs brings slightly, but clearly lower translation errors and a better scale estimates for out-of-distribution scenes with moderate parallax, in particular in EuRoC.

Table 5. Init: gravity consensus. VGGT model. Mean frame-0 gravity error (<sup>◦</sup> vs. GT). For diverging consensus, we report Fires on % of windows, replacing the default single-image $\mathbf { g } _ { \mathrm { g c } }$ with the motion-compensated $\mathbf { g } _ { \mathrm { l s } }$ only there; our estimate beats both sources over the full suite. Mean over sequences×windows.
<table><tr><td rowspan="2">Dataset</td><td colspan="4">Full-suite (°) ↓</td><td colspan="2">Diverging consensus</td></tr><tr><td> $\mathbf { g } _ { \mathrm { g c } }$ </td><td> $\mathbf { g } _ { \mathrm { i m u } }$ </td><td> $\mathbf { g } _ { \mathrm { l s } }$ </td><td>Ours</td><td>Fires (%)</td><td> $\mathbf { g } _ { \mathrm { g c } } {  } \mathbf { g } _ { \mathrm { l s } } ( { \mathrm { \mathrm { \mathrm { \infty } } } } )$ </td></tr><tr><td>TartanAirV2</td><td>9.67</td><td>8.58</td><td>4.31</td><td>1.96</td><td>10</td><td> $7 8 . 3 \substack {  4 . 3 }$ </td></tr><tr><td>EuRoC</td><td>1.98</td><td>4.22</td><td>2.62</td><td>1.98</td><td>0</td><td></td></tr><tr><td>UZH-FPV</td><td>23.75</td><td>22.38</td><td>15.52 10.32</td><td></td><td>31</td><td> $6 1 . 8 \substack {  } 1 8 . 8$ </td></tr></table>

Table 6. Init: velocity-scale estimation – OLS vs. TLS. VGGT model. Implied $| \hat { v } _ { 1 } | / | v _ { 1 } ^ { \mathrm { g t } } |$ (ideal 1.0) of the ordinary- and totalleast-squares scalar scale. OLS attenuates under predictor noise (regression dilution) ⇒ systematic under-scale; the errors-invariables TLS is unbiased.
<table><tr><td>Dataset</td><td>OLS</td><td>TLS</td></tr><tr><td>TartanAirV2</td><td>0.89</td><td>1.12</td></tr><tr><td>EuRoC</td><td>0.76</td><td>0.96</td></tr><tr><td>UZH-FPV</td><td>0.60</td><td>0.70</td></tr></table>

Table 7. Init: velocity magnitude. Using the TLS scale on the first pair alone (first-pair, Eq. 7) vs. the robust median backprojection over all pairs (all-pairs, Eq. 8. The all-pairs back-out lowers translation error and pulls both trajectory and depth scale toward 1 where the single-pair estimate is noisy. VGGT.
<table><tr><td rowspan="2">Dataset</td><td colspan="2">Trans (m) ↓</td><td colspan="2"> $\mathrm { S c a l e ( T r a j . )  1 }$ </td><td colspan="2">Scale (Depth) →1</td></tr><tr><td>first</td><td>all</td><td>first</td><td>all</td><td>first</td><td>all</td></tr><tr><td>TartanAirV2</td><td>0.505</td><td>0.351</td><td>1.00</td><td>1.05</td><td>1.05</td><td>1.11</td></tr><tr><td>EuRoC</td><td>0.060</td><td>0.060</td><td>0.91</td><td>0.99</td><td>1.08</td><td>1.00</td></tr><tr><td>UZH-FPV</td><td>0.537</td><td>0.495</td><td>0.81</td><td>0.80</td><td>0.94</td><td>0.94</td></tr></table>

Table 8. Initialization of up-to-scale models: GT-free vs. GT (Oracle). Inertial test-time refinement with fully-GT-free init vs. an oracle GT init (GT gravity, v<sub>1</sub>, gyro bias). Each cell is GT-free / GT; better in bold. GT-free reaches near-oracle quality.
<table><tr><td colspan="2" rowspan="2">Model Init</td><td>Trans (m) ↓</td><td>Rot (°) ↓</td><td>Scale (Traj.) →1</td><td>AbsRel ↓</td><td>δ&lt;1.25 ↑</td><td>Scale (Depth) →1</td></tr><tr><td>GT-free / GT</td><td>GT-free / GT</td><td>GT-free / GT</td><td>GT-free / GT</td><td>GT-free / GT</td><td>GT-free / GT</td></tr><tr><td></td><td>VGGT</td><td>0.358 / 0.058</td><td>0.410 / 0.348</td><td>1.05 / 0.99</td><td>0.089 / 0.076</td><td>0.515 / 0.882</td><td>1.11 / 1.06</td></tr><tr><td>TarAir</td><td>VGGT-Ω</td><td>0.378 / 0.053</td><td>0.354 / 0.248</td><td>1.10 / 1.00</td><td>0.080 / 0.077</td><td>0.596 / 0.938</td><td>1.12 / 1.04</td></tr><tr><td></td><td> $\pi ^ { 3 }$ </td><td>0.350 / 0.053</td><td>0.296 / 0.307</td><td>1.09 / 0.99</td><td>0.083 / 0.071</td><td>0.556 / 0.921</td><td>1.12 / 1.01</td></tr><tr><td></td><td>VGGT</td><td>0.061 / 0.015</td><td>0.285 / 0.118</td><td>0.98 / 1.00</td><td>0.234 / 0.240</td><td>0.324 / 0.465</td><td>1.00 / 1.08</td></tr><tr><td>EOC</td><td>VGGT-Ω</td><td>0.074 / 0.016</td><td>0.252 / 0.062</td><td>1.05 / 1.00</td><td>0.215 / 0.219</td><td>0.243 / 0.653</td><td>1.10 / 1.03</td></tr><tr><td></td><td> $\pi ^ { 3 }$ </td><td>0.078 / 0.012</td><td>0.164 / 0.105</td><td>1.08 / 0.99</td><td>0.203 / 0.204</td><td>0.251 / 0.672</td><td>1.18 / 1.07</td></tr><tr><td></td><td>VGGT</td><td>0.497 / 0.027</td><td>0.301 / 0.344</td><td>0.80 / 1.00</td><td>0.203 / 0.209</td><td>0.201 / 0.379</td><td>0.92 / 1.28</td></tr><tr><td>UΛZ-HPV</td><td>VGGT-Ω</td><td>0.328 / 0.009</td><td>0.192 / 0.189</td><td>0.92 / 1.00</td><td>0.237 / 0.242</td><td>0.217 / 0.309</td><td>1.20 / 1.30</td></tr><tr><td></td><td> $\pi ^ { 3 }$ </td><td>0.396 / 0.022</td><td>0.246 / 0.242</td><td>0.79 / 1.00</td><td>0.249 / 0.239</td><td>0.138 / 0.247</td><td>1.07 / 1.34</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 9. Initialization of scale-aware models: GT-free vs. GT IMU-pose conditioning (IMU-cond, feed-forward) and inertial test-time refinement (TTR), each shown as GT-free/GT: our fully-GT-free init vs. an oracle init (GT gravity, v<sub>0</sub>, gyro bias). Better in bold. <sup>†</sup>DA3’s IMU-cond additionally feeds camera intrinsics (required by its pose encoder); Pi3X’s is pose-only. Median over sequences×windows.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Model</td><td colspan="2">Trans (m) ↓</td><td colspan="2">Rot (°) ↓</td><td colspan="2">Scale (Traj.) →1</td><td colspan="2">Scale (Depth) →1</td></tr><tr><td>IMU-cond GT-free/GT</td><td>TTR GT-free/GT</td><td>IMU-cond GT-free/GT</td><td>TTR GT-free/GT</td><td>IMU-cond GT-free/GT</td><td>TTR GT-free/GT</td><td>IMU-cond GT-free/GT</td><td>TTR GT-free/GT</td></tr><tr><td rowspan="2">TartanAir</td><td>Pi3X</td><td>0.371 / 0.339</td><td>0.417 / 0.061</td><td>0.546 / 0.589</td><td>0.310 / 0.320</td><td>1.20 / 1.20</td><td>1.09 / 0.99</td><td>1.25 / 1.26</td><td>1.10 / 1.03</td></tr><tr><td>DA3</td><td>0.446 / 0.248†</td><td>0.486 / 0.061</td><td>0.801 / 0.683†</td><td>0.351 / 0.345</td><td>0.83 / 0.81†</td><td>1.06 / 1.01</td><td>0.90 / 0.88†</td><td>1.19 / 1.16</td></tr><tr><td rowspan="2">EuRoC</td><td>Pi3X</td><td>0.081 / 0.070</td><td>0.071 / 0.015</td><td>0.306 / 0.257</td><td>0.247 / 0.138</td><td>0.82 / 0.85</td><td>1.02 / 1.00</td><td>0.93 / 0.93</td><td>1.12 / 1.07</td></tr><tr><td>DA3</td><td>0.055 / 0.057†</td><td>0.092 / 0.027</td><td>0.326 / 0.235†</td><td>0.312 / 0.141</td><td>0.75 / 0.76†</td><td>1.13 / 1.01</td><td>0.73 / 0.73†</td><td>1.20 / 1.04</td></tr><tr><td rowspan="2">UZH-FPV</td><td>Pi3X</td><td>0.535 / 0.494</td><td>0.404 / 0.036</td><td>0.738 / 0.812</td><td>0.231 / 0.214</td><td>1.15 /1.15</td><td>0.82 / 1.00</td><td>1.65 / 1.64</td><td>1.09 / 1.38</td></tr><tr><td>DA3</td><td>0.452 / 0.377†</td><td>0.379 / 0.020</td><td>0.626 / 0.423†</td><td>0.231 / 0.228</td><td>0.83 / 0.84†</td><td>0.88 / 0.99</td><td>1.05 / 1.06†</td><td>1.07 / 1.27</td></tr></table>

## A.2. Test-time refinement

## Supervision objective

Table 10 breaks down the supervision objective adopted in the our TTR. Exclusively supervising pose (keeping depth frozen) limits the benefits that multiple views can bring into the refinement process. Adding reprojection depth consistency, unlocks multi-view constraints in a selfsupervised manner, leading to pose improvement. However, the inclusion of this up-to-scale loss degrades the scale estimates. Our adopted configuration solves this problem by regularizing depth, leading to a significant growth in depth metric correctness (δ<1.25).

## Frame stride: parallax

Figure 5 analyzes the effect of the frame stride parameter, a direct tuning knob which allows adjusting parallax by skipping a number of consecutive sequence frames. Figure 5 (A) shows absolute pose error increases with growing stride, which is a consequence of the absolute translation being bigger. Figure 5 (B) normalizes the error by the translation length, which makes a more relevant pattern to emerge. Strides of 3 or 2 bring a good balance between parallax and translation in the datasets considered, as below them the parallax is insufficient, and above it the multi-view overlap diminishes and perspective effects change texture. The initial velocity, influenced by the translation accuracy commented, shows the same pattern, with stride 2 being optimal for initial velocity direction, and 3 for its magnitude counterpart, as Figure 5 (C) exposes. We set stride 2 in the results presented in the main paper.

## Computational cost

On Table 11 we show the backbone-specific initialization and test-time refinement cost. We use the consumer-grade NVIDIA RTX 5090 GPU. Note some models (†) implement gradient checkpointing to fit in memory, increasing the cost for the choice of consumer-grade hardware. We also show, for the lightest model, VGGT-Ω, the impact of fine-tuning capacity. More unfrozen blocks (higher capacity), implies longer overall processing time and a higher memory consumption. We find 8 to be a sweet spot between refinement capacity (see rotation column) but also computational time and memory, as for the choice of n = 8 context frames,

![](images/826c38bea13e6c936967220166961ebc6971496278b7e426b1b3e2adec128220.jpg)

![](images/3134a7d80fb98841b6c87641dffa898a956364c70bddeac3ef44f4a5cd54a16b.jpg)

![](images/b5427264b08815a183e724faee7a249e9b5d9500c7048684ef58f8bd9a53b6b2.jpg)  
Figure 5. Frame-stride (parallax) effect absolute and relative poses, and initial velocity estimation.

Table 10. TTR: supervision objective ablation. Results shown for VGGT, on EuRoC, using GT-free init (VI3). Adding our selfsupervised depth-consistency term to the pose loss improves trajectory refinement and metric depth; anchoring then fixes the metric-depth drift the consistency term alone erodes.
<table><tr><td>Loss</td><td>Trans (m) ↓</td><td>Rot (°) ↓</td><td>Scale (Traj.) →1</td><td>AbsRel ↓</td><td>δ&lt;1.25 ↑</td><td>Scale (Depth) →1</td></tr><tr><td>Pose-only</td><td>0.078</td><td>0.441</td><td>0.983</td><td>0.240</td><td>0.275</td><td>0.994</td></tr><tr><td>+ SSL dcons</td><td>0.063</td><td>0.334</td><td>0.981</td><td>0.242</td><td>0.246</td><td>0.917</td></tr><tr><td>+ Anchoring</td><td>0.062</td><td>0.290</td><td>0.988</td><td>0.234</td><td>0.324</td><td>1.003</td></tr></table>

8 is the upper bound to fit in memory some of the models utilized in this work.

## B. Additional qualitative results

We show further examples of qualitative results. Figures 6 and 7 correspond to the four synthetic environments we evaluate on from TartanAirV2 (ForestEnv, ModernCity-Downtown, Hospital and House). Figure 8 presents an additional Vicon Room 2 reconstruction, with moderate motion and various scene objects; and Figure 9 shows how VI3 is capable of recovering from an overscaled metric-aware pretrained baseline, despite the high-speed motion profile of UZH-FPV Indoor Forward scenes.

## C. Limitations

Our approach materializes a way to add inertial cues to pretrained 3DFMs without any ground-truth. We have shown VI3 robustly recovers metric motion and geometry across backbones and datasets. It is, however, bounded by what visual-inertial cues cannot observe. Metric scale is only recoverable where motion provides sufficient translational parallax; a fundamental observability limit rather than a failure of estimation. Models differ in how gracefully they degrade in this regime, since the estimate operates on their predicted inter-frame velocities.

Table 11. Test-time refinement cost on a single NVIDIA RTX 5090, (n = 8, stride 2). Init is the one-time GT-free initialization, with model-weight loading excluded as it is amortized across windows. ms / iteration is the amortized per-step wall-time. Total is the time to reach the converged operating point of 160 iterations, i.e. Init + 160×(ms / iteration). The initialization is essentially free (∼ 230 ms) and model independent; the cost is dominated by the refinement steps. <sup>†</sup>Uses gradient checkpointing to be able to fit n = 8, trading recompute for memory (reflected in the periteration cost).

<table><tr><td colspan="3">Initialization and optimization steps</td></tr><tr><td>Model</td><td>Init</td><td>ms / iteration Total (160 it.)</td></tr><tr><td>VGGT†</td><td>232 ms</td><td>113 s</td></tr><tr><td>VGGT-Ω</td><td>706 232 ms 302</td><td>48s</td></tr><tr><td>π3</td><td>232 ms 492</td><td>79s</td></tr><tr><td>Pi3X†</td><td>232 ms 739</td><td>118s</td></tr><tr><td>DA3†</td><td>232 ms 802</td><td>128 s</td></tr><tr><td colspan="3">Number of trainable blocks (VGGT-Ω)</td></tr><tr><td>Blocks</td><td>Rot. (°) ms / iteration</td><td>Total (160 it.)</td></tr><tr><td>0</td><td>0.315 255</td><td>41 s</td></tr><tr><td>4</td><td>0.291 274</td><td>44s</td></tr><tr><td>8</td><td>0.228 302</td><td>48s</td></tr><tr><td>16</td><td>0.206 381</td><td>61 s</td></tr><tr><td>24 (Full)</td><td>0.234 421</td><td>67s</td></tr></table>

![](images/97e88a997c1fd4c5a72ed9d363ee425757d4da2f26dd5df9376f3d89cc0f72f7.jpg)

![](images/ae24c260c96c618070bb859a3dbc6da61cbaa7231eae262053c5b91cda0644ff.jpg)

![](images/4f6af815fd68bfc8ca01538a35acfab889db92a5d01114245f9919120675c914.jpg)

![](images/d131c96f16034c7c2e5525ae5c6e79969b067e3f38fa039cecbfc76b6d767a1f.jpg)  
Figure 6. TartanAirV2 qualitative results: ForestEnv and House environments. Pretrained 3DFM (red); inertial trajectory (blue); VI3 (green); GT pointcloud (normal texture).

![](images/0392d4c9992fe8510a5ac0b5d307fda99168dc3af527fc9c027b5eebada95da1.jpg)

![](images/c364f0e684545657c92197f6bf0a828b294361154fd01fdd0e53e01e1e4c4635.jpg)

![](images/940adde0a98fe7819f2ad432ba07d62914ef01850abcc3f5424044977185c0f2.jpg)

![](images/c75d27fc508f393348db7dffdf7e992dc965e030d4d9fe72f2cab4b26c6c53b1.jpg)

![](images/7f65c9887218fbf8098b7a843b0022cba47290fc359d5bba6a5af188aed655da.jpg)  
Figure 7. TartanAirV2 qualitative results: ModernCityDowntown and Hospital environments. Pretrained 3DFM (red); inertial trajectory (blue); VI3 (green); GT pointcloud (normal texture).

![](images/6cf52c92da1aeec7025a79b0f1ad3e991c6e713e436596e6e872ba1f7fdbefe2.jpg)

![](images/96f0e0d9a44a14035f1eee9f8c0bf19193dce861e61dcbd64be740d5dadfe872.jpg)

![](images/3d218cb5e1d5c6345601d7fb921bc2c39f5698c03414a0fc45c8209fe4d3b24b.jpg)  
Figure 8. EuRoC qualitative results: Vicon Room 2 scene. Pretrained 3DFM (red); inertial trajectory (blue); VI3 (green); GT pointcloud (normal texture).

![](images/26c83dcdd30f25a498e9ded725e49d1c5c0799e713b226412597e201f955fdf6.jpg)

![](images/054e8056607afc127120781c16a9fedebbb37a8796b51264f4a4fa75de1d9812.jpg)  
Figure 9. UZH-FPV qualitative results: Indoor Forward scene. Pretrained 3DFM (red); inertial trajectory (blue); VI3 (green); GT pointcloud (normal texture), stereo-derived (noisy).